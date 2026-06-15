# -*- coding: utf-8 -*-
"""
神经网络代理模型架构：U-Net / CNN encoder-decoder

输入: 掩模图案 (B, 1, H, W)，值域 [0, 1]
输出: 空间像 (B, 1, H, W)，值域 [0, 1]

支持的架构:
- UNet: 经典 U-Net，带跳跃连接的 encoder-decoder
- 可配置深度、通道数、是否使用批归一化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class SurrogateModelConfig:
    """
    代理模型架构配置

    Attributes:
        model_type: 模型类型，'unet' 或 'cnn_encoder_decoder'
        in_channels: 输入通道数（掩模为1）
        out_channels: 输出通道数（空间像为1）
        base_channels: 第一层卷积通道数，每层翻倍
        num_levels: U-Net 层数（encoder/decoder 级数）
        use_batch_norm: 是否使用批归一化
        dropout_rate: Dropout 比例，0 表示不使用
        activation: 激活函数类型: 'relu', 'leaky_relu', 'gelu'
        final_activation: 输出层激活: 'sigmoid', 'tanh', None
        bilinear: 上采样是否使用双线性插值（True）或转置卷积（False）
    """
    model_type: str = 'unet'
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    num_levels: int = 4
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    activation: str = 'relu'
    final_activation: Optional[str] = 'sigmoid'
    bilinear: bool = True

    def to_dict(self) -> dict:
        return {
            'model_type': self.model_type,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'base_channels': self.base_channels,
            'num_levels': self.num_levels,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'final_activation': self.final_activation,
            'bilinear': self.bilinear,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SurrogateModelConfig':
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def _get_activation(name: str) -> nn.Module:
    """获取激活函数模块"""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    else:
        logger.warning(f"未知激活函数 {name}，使用 ReLU")
        return nn.ReLU(inplace=True)


class DoubleConv(nn.Module):
    """
    U-Net 标准双卷积块: (Conv -> [BN] -> Act) × 2
    """

    def __init__(self, in_channels: int, out_channels: int,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 dropout_rate: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []

        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(_get_activation(activation))

        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))

        layers.append(nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(_get_activation(activation))

        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    U-Net 下采样块: MaxPool -> DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 dropout_rate: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batch_norm,
                       activation, dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    U-Net 上采样块: Upsample / ConvTranspose -> 拼接 -> DoubleConv
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 dropout_rate: float = 0.0,
                 bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(
                in_channels + skip_channels, out_channels,
                use_batch_norm, activation, dropout_rate
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(
                in_channels // 2 + skip_channels, out_channels,
                use_batch_norm, activation, dropout_rate
            )

    @staticmethod
    def _pad_to_match(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """对 x 做零填充以匹配 target 的空间尺寸"""
        diff_y = target.size()[2] - x.size()[2]
        diff_x = target.size()[3] - x.size()[3]
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2
            ])
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._pad_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net 模型：输入掩模，输出空间像

    Architecture:
        Input (B, 1, H, W)
         ↓
        [DoubleConv] → skip1
         ↓
        [DownBlock] × (num_levels-1) → skips
         ↓
        [Bottleneck DoubleConv]
         ↓
        [UpBlock] × (num_levels-1)  (使用对应的 skip)
         ↓
        [1×1 Conv] + final_activation
         ↓
        Output (B, 1, H, W)
    """

    def __init__(self, config: Optional[SurrogateModelConfig] = None):
        super().__init__()
        self.config = config or SurrogateModelConfig()
        cfg = self.config

        self.inc = DoubleConv(
            cfg.in_channels, cfg.base_channels,
            cfg.use_batch_norm, cfg.activation, cfg.dropout_rate
        )

        encoder_channels = [cfg.base_channels]
        self.down_blocks = nn.ModuleList()
        for i in range(1, cfg.num_levels):
            ch_in = encoder_channels[-1]
            ch_out = min(ch_in * 2, 512)
            self.down_blocks.append(DownBlock(
                ch_in, ch_out, cfg.use_batch_norm,
                cfg.activation, cfg.dropout_rate
            ))
            encoder_channels.append(ch_out)

        bottleneck_in = encoder_channels[-1]
        bottleneck_out = min(bottleneck_in * 2, 512)
        self.bottleneck = DoubleConv(
            bottleneck_in, bottleneck_out,
            cfg.use_batch_norm, cfg.activation, cfg.dropout_rate
        )

        self.up_blocks = nn.ModuleList()
        decoder_channels = [bottleneck_out]
        for i in reversed(range(cfg.num_levels - 1)):
            skip_ch = encoder_channels[i]
            in_ch = decoder_channels[-1]
            out_ch = encoder_channels[i]
            self.up_blocks.append(UpBlock(
                in_ch, skip_ch, out_ch,
                cfg.use_batch_norm, cfg.activation,
                cfg.dropout_rate, cfg.bilinear
            ))
            decoder_channels.append(out_ch)

        self.outc = nn.Conv2d(
            decoder_channels[-1], cfg.out_channels, kernel_size=1
        )

        if cfg.final_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif cfg.final_activation == 'tanh':
            self.final_act = nn.Tanh()
        else:
            self.final_act = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """He 初始化卷积权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.inc(x)
        skips.append(x)

        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        for i, up in enumerate(self.up_blocks):
            skip_idx = len(self.down_blocks) - i
            x = up(x, skips[skip_idx])

        logits = self.outc(x)
        return self.final_act(logits)

    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNEncoderDecoder(nn.Module):
    """
    简化版 CNN encoder-decoder（无跳跃连接），适合小尺寸/快速推理
    """

    def __init__(self, config: Optional[SurrogateModelConfig] = None):
        super().__init__()
        self.config = config or SurrogateModelConfig()
        cfg = self.config

        encoder_layers: List[nn.Module] = []
        ch_in = cfg.in_channels
        ch_out = cfg.base_channels
        encoder_channels = [ch_in]

        for i in range(cfg.num_levels):
            encoder_layers.append(nn.Conv2d(
                ch_in, ch_out, kernel_size=4, stride=2, padding=1,
                bias=not cfg.use_batch_norm
            ))
            if cfg.use_batch_norm:
                encoder_layers.append(nn.BatchNorm2d(ch_out))
            encoder_layers.append(_get_activation(cfg.activation))
            if cfg.dropout_rate > 0:
                encoder_layers.append(nn.Dropout2d(cfg.dropout_rate))

            encoder_channels.append(ch_out)
            ch_in = ch_out
            ch_out = min(ch_out * 2, 512)

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        ch_in = encoder_channels[-1]
        for i in range(cfg.num_levels):
            level = cfg.num_levels - 1 - i
            ch_out = encoder_channels[level] if level > 0 else cfg.base_channels

            if cfg.bilinear:
                decoder_layers.append(nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True
                ))
                decoder_layers.append(nn.Conv2d(
                    ch_in, ch_out, kernel_size=3, padding=1,
                    bias=not cfg.use_batch_norm
                ))
            else:
                decoder_layers.append(nn.ConvTranspose2d(
                    ch_in, ch_out, kernel_size=4, stride=2, padding=1,
                    bias=not cfg.use_batch_norm
                ))

            if i < cfg.num_levels - 1:
                if cfg.use_batch_norm:
                    decoder_layers.append(nn.BatchNorm2d(ch_out))
                decoder_layers.append(_get_activation(cfg.activation))
                if cfg.dropout_rate > 0:
                    decoder_layers.append(nn.Dropout2d(cfg.dropout_rate))

            ch_in = ch_out

        decoder_layers.append(nn.Conv2d(
            ch_in, cfg.out_channels, kernel_size=3, padding=1
        ))
        self.decoder = nn.Sequential(*decoder_layers)

        if cfg.final_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif cfg.final_activation == 'tanh':
            self.final_act = nn.Tanh()
        else:
            self.final_act = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        logits = self.decoder(x)
        return self.final_act(logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: SurrogateModelConfig) -> nn.Module:
    """
    根据配置构建模型

    Args:
        config: 模型配置

    Returns:
        PyTorch 模型
    """
    model_type = config.model_type.lower()
    if model_type == 'unet':
        model = UNet(config)
    elif model_type in ('cnn_encoder_decoder', 'cnn', 'encoder_decoder'):
        model = CNNEncoderDecoder(config)
    else:
        raise ValueError(f"未知模型类型: {model_type}，支持 'unet', 'cnn_encoder_decoder'")

    logger.info(
        f"构建 {model_type} 模型: 参数={model.count_parameters():,}, "
        f"levels={config.num_levels}, base_ch={config.base_channels}"
    )
    return model
