# -*- coding: utf-8 -*-
"""
版图热点机器学习预测模块

使用卷积神经网络（CNN）对版图 patch 进行二分类："高风险 / 低风险"。
在完整光学仿真之前快速筛出疑似热点区域，仅对高风险区域触发 OPC / ILT 精修，
大幅缩短大芯片的全局扫描时间。

核心组件
---------
1. HotspotCNN            : 轻量级 CNN 分类器（VGG-like / ResNet-like 两种架构）
2. HotspotPatchDataset   : patch 数据集（版图切片 + EPE/CD 风险标签）
3. generate_hotspot_dataset : 从完整版图 + 光学仿真自动生成标注数据集
4. train_hotspot_predictor   : 完整训练流水线（训练/验证/checkpoint/早停/导出）
5. scan_layout_for_hotspots  : 滑窗扫描完整版图，聚合输出热点 bbox 列表
6. load_hotspot_predictor    : 从 checkpoint 加载训练好的模型
7. export_hotspot_predictor  : 导出 TorchScript / ONNX 供生产部署

典型使用流程
-------------
    # 阶段 1：数据生成（离线）
    dataset = generate_hotspot_dataset(mask_layouts, target_layouts, optical_sys)

    # 阶段 2：模型训练（离线）
    train_hotspot_predictor(dataset, training_cfg, output_dir='./hotspot_model')

    # 阶段 3：全局热点扫描（在线推理）
    predictor = load_hotspot_predictor('./hotspot_model/best.pth')
    scan_result = scan_layout_for_hotspots(predictor, full_mask)
    high_risk_bboxes = scan_result.high_risk_bboxes   # 送入 OPC/ILT 精修
"""

import os
import sys
import copy
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path
from enum import Enum

import numpy as np
from scipy.ndimage import (
    binary_dilation, label, find_objects, generate_binary_structure,
    distance_transform_edt, gaussian_filter
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# PyTorch 依赖（可选，用于训练/推理）
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam, AdamW, SGD
    from torch.optim.lr_scheduler import (
        ReduceLROnPlateau, CosineAnnealingLR, StepLR
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch 未安装，热点预测模块的训练/推理功能将不可用。\n"
        "请安装: pip install torch torchvision"
    )


# ============================================================================
# 1. 配置与数据结构
# ============================================================================

class CNNArchitectureType(Enum):
    """CNN 分类器架构类型"""
    VGG_LIKE = "vgg_like"       # 经典 VGG 风格（小尺寸快速推理）
    RESNET_LIKE = "resnet_like"  # 带残差连接，精度更高


@dataclass
class HotspotPredictorConfig:
    """
    热点 CNN 分类模型架构配置

    Attributes
    ----------
    architecture : CNN 架构类型
    in_channels : 输入通道数（版图掩模=1，若拼接目标/距离变换可增大）
    patch_size : 输入 patch 尺寸（像素，正方形）
    base_channels : 第一层卷积通道数，每阶段翻倍
    num_stages : 卷积阶段数（每个阶段 = 下采样 + ConvBlock）
    use_batch_norm : 是否使用批归一化
    dropout_rate : Dropout 比例，0 表示不使用
    activation : 激活函数: 'relu', 'leaky_relu', 'gelu', 'silu'
    hidden_dims : 分类头 MLP 隐藏层维度列表（空则仅使用全局池化+线性）
    num_classes : 输出类别数（2=二分类高/低风险；也可支持多等级风险）
    use_se_block : 是否在每个 stage 末尾加入 SE 通道注意力
    """
    architecture: str = "vgg_like"
    in_channels: int = 1
    patch_size: int = 64
    base_channels: int = 32
    num_stages: int = 4
    use_batch_norm: bool = True
    dropout_rate: float = 0.2
    activation: str = "relu"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 64])
    num_classes: int = 2
    use_se_block: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HotspotPredictorConfig':
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass
class HotspotDatasetConfig:
    """
    热点 patch 数据集生成配置

    Attributes
    ----------
    patch_size : 切片尺寸（像素，正方形）
    stride : 滑窗步长（像素），用于从完整版图切 patch
    epe_threshold_nm : EPE 阈值（nm），超过则标记为高风险
    cd_error_threshold_nm : CD 误差阈值（nm，绝对值），超过则标记高风险
    min_risk_pct : patch 内风险像素占比阈值，大于则标记为高风险样本
    balance_classes : 是否自动平衡正负样本比例（欠采样多数类/过采样少数类）
    positive_ratio : 当 balance_classes=True 时，正样本(高风险)占比目标
    augment : 是否做数据增强（旋转/翻转）
    pixel_size : 像素尺寸 (nm)，用于阈值换算
    use_distance_transform : 是否将距离变换作为额外通道（辅助边缘信息）
    """
    patch_size: int = 64
    stride: int = 32
    epe_threshold_nm: float = 3.0
    cd_error_threshold_nm: float = 5.0
    min_risk_pct: float = 0.02
    balance_classes: bool = True
    positive_ratio: float = 0.4
    augment: bool = True
    pixel_size: float = 1.0
    use_distance_transform: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HotspotDatasetConfig':
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass
class HotspotTrainingConfig:
    """
    热点预测模型训练超参数配置

    Attributes
    ----------
    epochs : 总训练 epoch 数
    batch_size : 批大小
    learning_rate : 初始学习率
    weight_decay : L2 权重衰减
    optimizer : 优化器类型: 'adam', 'adamw', 'sgd'
    grad_clip : 梯度裁剪最大范数，0 表示不裁剪

    loss_type : 损失函数: 'ce'(交叉熵), 'focal', 'weighted_ce'
    focal_gamma : Focal Loss 的 gamma 参数
    class_weights : 类别权重 [w_low_risk, w_high_risk]，None 自动计算

    scheduler : 学习率调度器: 'plateau', 'cosine', 'step', None
    scheduler_patience : ReduceLROnPlateau 耐心值
    scheduler_factor : ReduceLROnPlateau 衰减因子
    scheduler_step_size : StepLR 步长
    min_lr : 最小学习率

    train_ratio : 训练集比例（其余为验证集）
    seed : 随机种子
    device : 计算设备: 'auto', 'cpu', 'cuda', 'mps'
    num_workers : DataLoader 加载线程数
    pin_memory : DataLoader 是否锁页内存

    output_dir : 输出目录
    checkpoint_freq : 每隔多少 epoch 保存一次 checkpoint
    save_best_only : 是否只保存验证集最优模型
    early_stop_patience : 早停耐心值，0 表示不早停
    log_interval : 每隔多少 batch 打印一次训练日志

    export_onnx : 训练完成后是否导出 ONNX
    export_torchscript : 训练完成后是否导出 TorchScript
    onnx_opset_version : ONNX opset 版本
    """
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    grad_clip: float = 1.0

    loss_type: str = "focal"
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None

    scheduler: Optional[str] = "plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 15
    min_lr: float = 1e-7

    train_ratio: float = 0.85
    seed: int = 42
    device: str = "auto"
    num_workers: int = 2
    pin_memory: bool = True

    output_dir: str = "./hotspot_model"
    checkpoint_freq: int = 5
    save_best_only: bool = True
    early_stop_patience: int = 12
    log_interval: int = 20

    export_onnx: bool = True
    export_torchscript: bool = True
    onnx_opset_version: int = 17

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HotspotTrainingConfig':
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass
class HotspotScanResult:
    """
    全局版图热点扫描结果

    Attributes
    ----------
    high_risk_bboxes : 高风险区域 bbox 列表，格式 (y_min, y_max, x_min, x_max)
    risk_scores : 对应每个 bbox 的平均风险概率 (0~1)
    risk_heatmap : 与输入版图同尺寸的风险概率热力图（numpy 数组）
    patch_predictions : 每个 patch 的预测结果列表 [(y,x,p,label), ...]
    num_patches_scanned : 扫描的总 patch 数
    num_high_risk_patches : 被判定为高风险的 patch 数
    high_risk_area_ratio : 高风险区域占版图总面积比例
    scan_time_sec : 扫描耗时（秒）
    """
    high_risk_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    risk_scores: List[float] = field(default_factory=list)
    risk_heatmap: Optional[np.ndarray] = None
    patch_predictions: List[Tuple[int, int, float, int]] = field(default_factory=list)
    num_patches_scanned: int = 0
    num_high_risk_patches: int = 0
    high_risk_area_ratio: float = 0.0
    scan_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "high_risk_bboxes": [list(b) for b in self.high_risk_bboxes],
            "risk_scores": self.risk_scores,
            "num_patches_scanned": self.num_patches_scanned,
            "num_high_risk_patches": self.num_high_risk_patches,
            "high_risk_area_ratio": self.high_risk_area_ratio,
            "scan_time_sec": self.scan_time_sec,
        }


# ============================================================================
# 2. CNN 分类模型
# ============================================================================

def _get_activation_torch(name: str) -> 'nn.Module':
    """获取激活函数模块（PyTorch）"""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 不可用")
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'silu' or name == 'swish':
        return nn.SiLU(inplace=True)
    else:
        logger.warning(f"未知激活函数 {name}，使用 ReLU")
        return nn.ReLU(inplace=True)


class SEBlock(nn.Module if _TORCH_AVAILABLE else object):
    """
    Squeeze-and-Excitation 通道注意力模块
    参考: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, channels: int, reduction: int = 16):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用")
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        scale = self.excitation(self.squeeze(x))
        return x * scale


class VGGConvBlock(nn.Module if _TORCH_AVAILABLE else object):
    """
    VGG 风格卷积块: 2 × (Conv3x3 → [BN] → Act) → MaxPool
    """

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True,
                 activation: str = 'relu', dropout_rate: float = 0.0,
                 use_se: bool = False):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用")
        super().__init__()
        layers: List[nn.Module] = []

        for _ in range(2):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,
                                    bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(_get_activation_torch(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            in_ch = out_ch

        if use_se:
            layers.append(SEBlock(out_ch))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        return self.block(x)


class ResidualBlock(nn.Module if _TORCH_AVAILABLE else object):
    """
    残差块: Conv3x3 → BN → Act → Conv3x3 → BN + shortcut
    末尾接 MaxPool 下采样
    """

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True,
                 activation: str = 'relu', dropout_rate: float = 0.0,
                 use_se: bool = False):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用")
        super().__init__()
        act_cls = lambda: _get_activation_torch(activation)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act1 = act_cls()
        self.drop1 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch
            else nn.Identity()
        )
        self.act2 = act_cls()
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        identity = self.shortcut(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out = self.se(out) + identity
        out = self.act2(out)
        return self.pool(out)


class HotspotCNN(nn.Module if _TORCH_AVAILABLE else object):
    """
    版图热点 CNN 二分类器

    输入: (B, in_channels, patch_size, patch_size)，值域 [0, 1]
    输出: (B, num_classes) logits 或 (B,) 风险概率（根据 inference_mode）

    流程:
        Patch → [Conv Stage × N] → GlobalAvgPool → [MLP] → 分类 logits
    """

    def __init__(self, config: Optional[HotspotPredictorConfig] = None):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch 未安装，无法构建 HotspotCNN。\n"
                "请安装: pip install torch torchvision"
            )
        super().__init__()
        self.config = config or HotspotPredictorConfig()
        cfg = self.config

        arch = CNNArchitectureType(cfg.architecture)

        # ---- 骨干网络 ----
        in_ch = cfg.in_channels
        out_ch = cfg.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3,
                      bias=not cfg.use_batch_norm),
            nn.BatchNorm2d(out_ch) if cfg.use_batch_norm else nn.Identity(),
            _get_activation_torch(cfg.activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        in_ch = out_ch

        self.stages = nn.ModuleList()
        for _ in range(cfg.num_stages):
            out_ch = min(in_ch * 2, 512)
            if arch == CNNArchitectureType.VGG_LIKE:
                stage = VGGConvBlock(
                    in_ch, out_ch, use_bn=cfg.use_batch_norm,
                    activation=cfg.activation, dropout_rate=cfg.dropout_rate,
                    use_se=cfg.use_se_block
                )
            else:
                stage = ResidualBlock(
                    in_ch, out_ch, use_bn=cfg.use_batch_norm,
                    activation=cfg.activation, dropout_rate=cfg.dropout_rate,
                    use_se=cfg.use_se_block
                )
            self.stages.append(stage)
            in_ch = out_ch

        # ---- 分类头 ----
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        fc_layers: List[nn.Module] = []
        prev_dim = in_ch
        for h_dim in cfg.hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, h_dim))
            fc_layers.append(nn.BatchNorm1d(h_dim))
            fc_layers.append(_get_activation_torch(cfg.activation))
            if cfg.dropout_rate > 0:
                fc_layers.append(nn.Dropout(cfg.dropout_rate))
            prev_dim = h_dim
        fc_layers.append(nn.Linear(prev_dim, cfg.num_classes))
        self.classifier = nn.Sequential(*fc_layers)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """He 初始化卷积权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: 'torch.Tensor',
                return_probs: bool = False) -> 'torch.Tensor':
        """
        Args:
            x: 输入张量 (B, C, H, W)
            return_probs: True 时返回风险概率 (Sigmoid/Softmax)，否则返回 logits
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.global_pool(x).flatten(1)
        logits = self.classifier(x)

        if return_probs:
            if self.config.num_classes == 2:
                return F.softmax(logits, dim=1)[:, 1:2]  # P(high_risk)
            else:
                return F.softmax(logits, dim=1)
        return logits

    # ------------------------------------------------------------------
    def predict_risk(self, x: 'torch.Tensor') -> np.ndarray:
        """便捷推理：返回高风险概率 (numpy, 0~1)"""
        self.eval()
        with torch.no_grad():
            probs = self.forward(x, return_probs=True)
        return probs.detach().cpu().numpy().squeeze()

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
def build_hotspot_cnn(
    config: Optional[HotspotPredictorConfig] = None
) -> 'HotspotCNN':
    """
    根据配置构建热点 CNN 分类模型

    Args:
        config: 模型配置，None 则使用默认

    Returns:
        HotspotCNN 模型实例
    """
    cfg = config or HotspotPredictorConfig()
    model = HotspotCNN(cfg)
    logger.info(
        f"构建 HotspotCNN [{cfg.architecture}]: "
        f"参数={model.count_parameters():,}, "
        f"stages={cfg.num_stages}, base_ch={cfg.base_channels}, "
        f"patch={cfg.patch_size}, SE={cfg.use_se_block}"
    )
    return model


# ============================================================================
# 3. 数据集：patch 生成 + 风险标签
# ============================================================================

class HotspotPatchDataset(Dataset if _TORCH_AVAILABLE else object):
    """
    热点 patch 数据集（PyTorch Dataset）

    样本构成:
        - input:  版图 patch 张量 (C, patch_size, patch_size)，值域 [0, 1]
        - label:  风险标签 (0=低风险, 1=高风险)
        - meta:   元信息字典（epe_mean, cd_error_mean, origin bbox 等）
    """

    def __init__(
        self,
        patches: List[np.ndarray],
        labels: List[int],
        metas: Optional[List[Dict[str, Any]]] = None,
        augment: bool = False,
        transform=None,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用")
        super().__init__()
        self.patches = [p.astype(np.float32) for p in patches]
        self.labels = list(labels)
        self.metas = metas or [{} for _ in patches]
        self.augment = augment
        self.transform = transform
        assert len(self.patches) == len(self.labels) == len(self.metas), \
            "patches / labels / metas 长度不一致"

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple['torch.Tensor', int, Dict[str, Any]]:
        patch = self.patches[idx].copy()
        label = self.labels[idx]
        meta = self.metas[idx]

        if self.augment:
            k = np.random.randint(0, 4)
            patch = np.rot90(patch, k=k, axes=(-2, -1))
            if np.random.random() < 0.5:
                patch = np.flip(patch, axis=-1).copy()
            if np.random.random() < 0.5:
                patch = np.flip(patch, axis=-2).copy()

        if self.transform is not None:
            patch = self.transform(patch)

        if patch.ndim == 2:
            patch = patch[np.newaxis, ...]

        tensor = torch.from_numpy(np.ascontiguousarray(patch))
        return tensor, int(label), meta


# ---------------------------------------------------------------------------
# 标签生成辅助：计算 patch 风险掩码
# ---------------------------------------------------------------------------

def _compute_risk_mask(
    wafer_binary: np.ndarray,
    target_binary: np.ndarray,
    pixel_size: float,
    epe_threshold_nm: float,
    cd_error_threshold_nm: float,
) -> np.ndarray:
    """
    基于 EPE + CD 误差计算逐像素风险掩码（1=风险像素）

    思路：
    1. 用距离变换得到 wafer 边缘与 target 边缘的逐像素偏差
    2. EPE 阈值: 边缘像素偏差 > epe_threshold_nm
    3. CD 阈值: 内部大面积缺失/冗余 (> cd_error_threshold_nm)
    """
    from core.litho_metrics import extract_edges

    wafer_edge = extract_edges(wafer_binary) > 0.5
    target_edge = extract_edges(target_binary) > 0.5

    dist_to_target = distance_transform_edt(~target_edge) * pixel_size
    dist_to_wafer = distance_transform_edt(~wafer_edge) * pixel_size

    epe_risk = np.zeros_like(wafer_binary, dtype=bool)
    epe_risk[wafer_edge] = dist_to_target[wafer_edge] > epe_threshold_nm
    epe_risk[target_edge] = dist_to_wafer[target_edge] > epe_threshold_nm

    cd_map = (wafer_binary.astype(np.float32) - target_binary.astype(np.float32))
    cd_risk = np.abs(cd_map) * distance_transform_edt(cd_map != 0) * pixel_size \
        > cd_error_threshold_nm

    return epe_risk | cd_risk


def generate_hotspot_dataset(
    mask_layouts: List[np.ndarray],
    target_layouts: Optional[List[np.ndarray]] = None,
    optical_system=None,
    wafer_threshold: float = 0.3,
    dataset_config: Optional[HotspotDatasetConfig] = None,
    process_condition=None,
) -> HotspotPatchDataset:
    """
    从完整版图 + （可选）光学仿真自动生成带标签的热点 patch 数据集

    当 target_layouts 为 None 时：
        自动调用 simulate_wafer_image 得到 wafer，与 mask 对比生成标签。
        如果 optical_system 也为 None，则退化为**启发式规则标注**（仅依赖
        mask 的几何复杂度，如线宽、拐角密度，可用于无仿真环境的预训练）。

    Args:
        mask_layouts: 掩模版图列表，每个为 (H, W) 二值/浮点 numpy 数组
        target_layouts: 目标版图列表（理想成像），可为 None
        optical_system: core.imaging.OpticalSystem 实例，用于仿真 wafer
        wafer_threshold: wafer 二值化阈值
        dataset_config: 数据集配置，None 使用默认
        process_condition: core.imaging.ProcessCondition 工艺条件

    Returns:
        HotspotPatchDataset 实例
    """
    cfg = dataset_config or HotspotDatasetConfig()
    ps = cfg.patch_size
    stride = cfg.stride
    px = cfg.pixel_size

    all_patches: List[np.ndarray] = []
    all_labels: List[int] = []
    all_metas: List[Dict[str, Any]] = []

    n_layouts = len(mask_layouts)
    logger.info(f"开始生成热点数据集: {n_layouts} 张版图, "
                f"patch={ps}, stride={stride}")

    for layout_idx, mask in enumerate(mask_layouts):
        mask_bin = (np.asarray(mask) >= 0.5).astype(np.float64)
        H, W = mask_bin.shape

        # ---- (A) 若有 target / 仿真器，计算 wafer 与风险掩码 ----
        target = (
            target_layouts[layout_idx] if target_layouts is not None
            else mask_bin
        )
        target_bin = (np.asarray(target) >= 0.5).astype(np.float64)

        if optical_system is not None:
            try:
                from core.imaging import simulate_wafer_image
                if process_condition is not None:
                    wafer, _ = simulate_wafer_image(
                        optical_system, mask_bin, process_condition
                    )
                else:
                    wafer, _ = simulate_wafer_image(
                        optical_system, mask_bin
                    )
                wafer_bin = (wafer >= wafer_threshold).astype(np.float64)
                risk_mask = _compute_risk_mask(
                    wafer_bin, target_bin, px,
                    cfg.epe_threshold_nm, cfg.cd_error_threshold_nm
                )
            except Exception as exc:
                logger.warning(f"版图 {layout_idx} 仿真失败，退化几何标注: {exc}")
                risk_mask = _heuristic_risk_mask(mask_bin, px,
                                                 cfg.epe_threshold_nm)
        else:
            risk_mask = _heuristic_risk_mask(mask_bin, px,
                                             cfg.epe_threshold_nm)

        # ---- (B) 距离变换通道 ----
        if cfg.use_distance_transform:
            dt = distance_transform_edt(target_bin > 0.5)
            dt_norm = dt / (dt.max() + 1e-8)
            mask_input = np.stack([mask_bin, dt_norm.astype(np.float32)], axis=0)
        else:
            mask_input = mask_bin[np.newaxis, ...]

        # ---- (C) 滑窗切 patch ----
        ys = list(range(0, max(1, H - ps + 1), stride))
        xs = list(range(0, max(1, W - ps + 1), stride))
        if ys[-1] != H - ps:
            ys.append(H - ps)
        if xs[-1] != W - ps:
            xs.append(W - ps)

        for y in ys:
            for x in xs:
                patch = mask_input[:, y:y + ps, x:x + ps]
                risk_patch = risk_mask[y:y + ps, x:x + ps]

                risk_pct = risk_patch.sum() / max(1, risk_patch.size)
                patch_risk_mean = risk_pct

                is_high_risk = risk_pct >= cfg.min_risk_pct

                all_patches.append(patch.astype(np.float32))
                all_labels.append(1 if is_high_risk else 0)
                all_metas.append({
                    'layout_idx': layout_idx,
                    'bbox': (y, y + ps, x, x + ps),
                    'risk_pct': float(patch_risk_mean),
                    'epe_max_nm': float(risk_patch.sum() * px),  # proxy
                })

    logger.info(f"滑窗切片完成: 共 {len(all_patches)} 个 patch, "
                f"高风险={sum(all_labels)}, 低风险={len(all_labels) - sum(all_labels)}")

    # ---- (D) 类别均衡 ----
    if cfg.balance_classes:
        all_patches, all_labels, all_metas = _balance_dataset(
            all_patches, all_labels, all_metas, cfg.positive_ratio
        )

    return HotspotPatchDataset(
        all_patches, all_labels, all_metas, augment=cfg.augment
    )


def _heuristic_risk_mask(
    mask_bin: np.ndarray,
    pixel_size: float,
    epe_threshold_nm: float,
) -> np.ndarray:
    """
    无仿真器时的启发式几何风险掩码（预训练可用）

    规则（近似模拟光刻热点）：
    1. 细线宽区域: 距离变换 < 阈值
    2. 高密度拐角: Laplacian 响应强
    """
    from scipy.ndimage import convolve

    risk = np.zeros_like(mask_bin, dtype=bool)
    dt_inside = distance_transform_edt(mask_bin > 0.5) * pixel_size
    dt_outside = distance_transform_edt(mask_bin <= 0.5) * pixel_size
    threshold = max(epe_threshold_nm * 2.0, 2 * pixel_size)
    risk |= (dt_inside > 0) & (dt_inside < threshold)
    risk |= (dt_outside > 0) & (dt_outside < threshold)

    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    resp = np.abs(convolve(mask_bin.astype(np.float32), laplacian))
    risk |= resp >= 0.3
    return risk


def _balance_dataset(
    patches: List[np.ndarray],
    labels: List[int],
    metas: List[Dict[str, Any]],
    target_positive_ratio: float,
) -> Tuple[List, List, List]:
    """平衡正负样本（优先欠采样多数类，不足再过采样少数类）"""
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        logger.warning("数据集仅含单一类别，无法平衡，返回原样")
        return patches, labels, metas

    target_n_neg = int(n_pos * (1 - target_positive_ratio) / target_positive_ratio)
    if target_n_neg < n_neg:
        rng = np.random.RandomState(42)
        sel_neg = rng.choice(neg_idx, size=target_n_neg, replace=False)
        sel_idx = list(pos_idx) + list(sel_neg)
    else:
        target_n_pos = int(n_neg * target_positive_ratio / (1 - target_positive_ratio))
        rng = np.random.RandomState(42)
        if target_n_pos > n_pos:
            extra = rng.choice(pos_idx, size=target_n_pos - n_pos, replace=True)
            sel_pos = pos_idx + list(extra)
        else:
            sel_pos = list(rng.choice(pos_idx, size=target_n_pos, replace=False))
        sel_idx = sel_pos + neg_idx

    rng2 = np.random.RandomState(7)
    rng2.shuffle(sel_idx)
    return (
        [patches[i] for i in sel_idx],
        [labels[i] for i in sel_idx],
        [metas[i] for i in sel_idx],
    )


# ============================================================================
# 4. 训练流水线
# ============================================================================

class FocalLoss(nn.Module if _TORCH_AVAILABLE else object):
    """Focal Loss: FL(pt) = -(1 - pt)^γ * log(pt)"""

    def __init__(self, gamma: float = 2.0,
                 weight: Optional['torch.Tensor'] = None,
                 reduction: str = 'mean'):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用")
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: 'torch.Tensor',
                targets: 'torch.Tensor') -> 'torch.Tensor':
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return fl.mean()
        elif self.reduction == 'sum':
            return fl.sum()
        return fl


def _get_device(device_name: str) -> 'torch.device':
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 不可用")
    name = device_name.lower()
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(name)


def _split_dataset(dataset: HotspotPatchDataset, train_ratio: float, seed: int):
    n = len(dataset)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    from torch.utils.data import Subset
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_hotspot_predictor(
    dataset: HotspotPatchDataset,
    model_config: Optional[HotspotPredictorConfig] = None,
    training_config: Optional[HotspotTrainingConfig] = None,
) -> Dict[str, Any]:
    """
    完整训练热点预测 CNN 模型

    Args:
        dataset: 样本数据集
        model_config: 模型架构配置，None 则根据 dataset patch 尺寸自适应
        training_config: 训练超参数配置，None 用默认

    Returns:
        训练日志字典，包含 metrics 曲线、最佳 epoch 等信息
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch 未安装，无法训练热点预测模型。\n"
            "请安装: pip install torch torchvision"
        )

    t_cfg = training_config or HotspotTrainingConfig()
    torch.manual_seed(t_cfg.seed)
    np.random.seed(t_cfg.seed)

    sample_ch = dataset[0][0].shape[0]
    sample_ps = dataset[0][0].shape[-1]
    m_cfg = model_config or HotspotPredictorConfig(
        in_channels=sample_ch, patch_size=sample_ps
    )
    m_cfg.in_channels = sample_ch
    m_cfg.patch_size = sample_ps

    out_dir = Path(t_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "model_config.json", 'w') as f:
        json.dump(m_cfg.to_dict(), f, indent=2, ensure_ascii=False)
    with open(out_dir / "training_config.json", 'w') as f:
        json.dump(t_cfg.to_dict(), f, indent=2, ensure_ascii=False)

    device = _get_device(t_cfg.device)
    logger.info(f"训练设备: {device}")

    model = build_hotspot_cnn(m_cfg).to(device)

    # ---- 数据加载器 ----
    train_ds, val_ds = _split_dataset(dataset, t_cfg.train_ratio, t_cfg.seed)
    logger.info(f"数据集划分: train={len(train_ds)}, val={len(val_ds)}")

    g = torch.Generator()
    g.manual_seed(t_cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=t_cfg.batch_size, shuffle=True,
        num_workers=t_cfg.num_workers, pin_memory=t_cfg.pin_memory,
        drop_last=True, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=t_cfg.batch_size * 2, shuffle=False,
        num_workers=t_cfg.num_workers, pin_memory=t_cfg.pin_memory
    )

    # ---- 优化器 ----
    opt_name = t_cfg.optimizer.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == 'adam':
        optimizer = Adam(params, lr=t_cfg.learning_rate,
                         weight_decay=t_cfg.weight_decay)
    elif opt_name == 'adamw':
        optimizer = AdamW(params, lr=t_cfg.learning_rate,
                          weight_decay=t_cfg.weight_decay)
    elif opt_name == 'sgd':
        optimizer = SGD(params, lr=t_cfg.learning_rate, momentum=0.9,
                        weight_decay=t_cfg.weight_decay, nesterov=True)
    else:
        raise ValueError(f"不支持的优化器: {opt_name}")

    # ---- 损失函数 ----
    class_weights_tensor = None
    if t_cfg.loss_type in ('weighted_ce', 'focal') and t_cfg.class_weights is None:
        labels_train = [dataset.labels[i] for i in train_ds.indices]
        n_pos = sum(labels_train)
        n_neg = len(labels_train) - n_pos
        if n_pos > 0 and n_neg > 0:
            inv = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float32)
            class_weights_tensor = inv / inv.sum() * 2
            logger.info(f"自动类别权重: {class_weights_tensor.tolist()}")
    elif t_cfg.class_weights is not None:
        class_weights_tensor = torch.tensor(
            t_cfg.class_weights, dtype=torch.float32
        )

    if class_weights_tensor is not None:
        class_weights_tensor = class_weights_tensor.to(device)

    if t_cfg.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif t_cfg.loss_type == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif t_cfg.loss_type == 'focal':
        criterion = FocalLoss(gamma=t_cfg.focal_gamma,
                              weight=class_weights_tensor)
    else:
        raise ValueError(f"不支持的损失函数: {t_cfg.loss_type}")

    # ---- 学习率调度 ----
    scheduler = None
    if t_cfg.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=t_cfg.scheduler_factor,
            patience=t_cfg.scheduler_patience, min_lr=t_cfg.min_lr, verbose=True
        )
    elif t_cfg.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg.epochs,
                                      eta_min=t_cfg.min_lr)
    elif t_cfg.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=t_cfg.scheduler_step_size,
                           gamma=t_cfg.scheduler_factor)

    # ---- 训练循环 ----
    history: Dict[str, List[float]] = {
        'train_loss': [], 'train_acc': [], 'train_recall': [],
        'val_loss': [], 'val_acc': [], 'val_recall': [],
        'val_precision': [], 'val_f1': [], 'lr': [],
    }
    best_val_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    early_stop_cnt = 0

    global_step = 0
    t_start = time.time()

    for epoch in range(1, t_cfg.epochs + 1):
        model.train()
        losses, corrects, totals = 0.0, 0, 0
        tp, fn = 0, 0

        for batch_i, (x, y, _meta) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, return_probs=False)
            loss = criterion(logits, y)
            loss.backward()
            if t_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
            optimizer.step()

            bs = y.size(0)
            losses += loss.item() * bs
            preds = logits.argmax(dim=1)
            corrects += (preds == y).sum().item()
            totals += bs
            tp += ((preds == 1) & (y == 1)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()
            global_step += 1

            if (batch_i + 1) % t_cfg.log_interval == 0:
                cur_loss = losses / max(1, totals)
                cur_acc = corrects / max(1, totals)
                cur_rec = tp / max(1, tp + fn)
                logger.info(
                    f"[Epoch {epoch}/{t_cfg.epochs}] "
                    f"batch={batch_i + 1}/{len(train_loader)} "
                    f"loss={cur_loss:.4f} acc={cur_acc:.4f} "
                    f"recall={cur_rec:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        train_loss = losses / max(1, totals)
        train_acc = corrects / max(1, totals)
        train_rec = tp / max(1, tp + fn)

        # ---- 验证 ----
        val_metrics = _evaluate_model(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]['lr']

        for k in ('train_loss', 'train_acc', 'train_recall',
                  'val_loss', 'val_acc', 'val_recall',
                  'val_precision', 'val_f1', 'lr'):
            history[k].append({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_recall': train_rec,
                **val_metrics,
                'lr': cur_lr,
            }[k])

        logger.info(
            f"==== Epoch {epoch}/{t_cfg.epochs} ====\n"
            f"  Train: loss={train_loss:.4f} acc={train_acc:.4f} "
            f"recall={train_rec:.4f}\n"
            f"  Val:   loss={val_metrics['val_loss']:.4f} "
            f"acc={val_metrics['val_acc']:.4f} "
            f"recall={val_metrics['val_recall']:.4f} "
            f"precision={val_metrics['val_precision']:.4f} "
            f"F1={val_metrics['val_f1']:.4f}   lr={cur_lr:.2e}"
        )

        # ---- Checkpoint & Best ----
        if val_metrics['val_f1'] > best_val_f1:
            best_val_f1 = val_metrics['val_f1']
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            early_stop_cnt = 0
            save_path = out_dir / "best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state_dict,
                'model_config': m_cfg.to_dict(),
                'val_f1': best_val_f1,
                'history': history,
            }, save_path)
            logger.info(f"  ✅ 新最佳模型已保存: {save_path}  (F1={best_val_f1:.4f})")
        else:
            early_stop_cnt += 1

        if not t_cfg.save_best_only and epoch % t_cfg.checkpoint_freq == 0:
            ckpt_path = out_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_config': m_cfg.to_dict(),
                'history': history,
            }, ckpt_path)

        # ---- Scheduler 步进 ----
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['val_f1'])
            else:
                scheduler.step()

        # ---- 早停 ----
        if t_cfg.early_stop_patience > 0 and early_stop_cnt >= t_cfg.early_stop_patience:
            logger.info(f"⏹  早停触发: {early_stop_cnt} 个 epoch 无提升")
            break

    t_total = time.time() - t_start
    logger.info(f"训练完成: 耗时 {t_total:.1f}s, 最佳 epoch={best_epoch}, "
                f"最佳 Val F1={best_val_f1:.4f}")

    # ---- 恢复最佳权重并导出 ----
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history_path = out_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()},
                  f, indent=2)

    if t_cfg.export_torchscript:
        try:
            _export_torchscript(model, m_cfg, device, out_dir / "hotspot_predictor.pt")
        except Exception as exc:
            logger.warning(f"TorchScript 导出失败: {exc}")

    if t_cfg.export_onnx:
        try:
            _export_onnx(model, m_cfg, device,
                         out_dir / "hotspot_predictor.onnx",
                         opset=t_cfg.onnx_opset_version)
        except Exception as exc:
            logger.warning(f"ONNX 导出失败: {exc}")

    return {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'total_time_sec': t_total,
        'history': history,
        'output_dir': str(out_dir),
    }


def _evaluate_model(model, loader, criterion, device) -> Dict[str, float]:
    """单轮验证评估"""
    model.eval()
    losses, corrects, totals = 0.0, 0, 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            logits = model(x, return_probs=False)
            loss = criterion(logits, y)

            bs = y.size(0)
            losses += loss.item() * bs
            preds = logits.argmax(dim=1)
            corrects += (preds == y).sum().item()
            totals += bs
            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    acc = corrects / max(1, totals)
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        'val_loss': losses / max(1, totals),
        'val_acc': acc,
        'val_recall': recall,
        'val_precision': precision,
        'val_f1': f1,
    }


# ============================================================================
# 5. 模型加载 / 保存 / 导出
# ============================================================================

def load_hotspot_predictor(
    checkpoint_path: Union[str, Path],
    device: str = 'auto',
) -> HotspotCNN:
    """
    从 checkpoint 加载训练好的 HotspotCNN 模型

    Args:
        checkpoint_path: .pth / .pt 文件路径
        device: 推理设备

    Returns:
        eval 模式的 HotspotCNN 实例
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 未安装，无法加载模型")

    ckpt = torch.load(str(checkpoint_path),
                      map_location=_get_device('cpu'))
    cfg_dict = ckpt.get('model_config')
    cfg = (HotspotPredictorConfig.from_dict(cfg_dict)
           if cfg_dict else HotspotPredictorConfig())
    model = build_hotspot_cnn(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    dev = _get_device(device)
    model = model.to(dev).eval()
    logger.info(f"已加载热点预测模型: {checkpoint_path} -> {dev}")
    return model


def _export_torchscript(model: HotspotCNN, cfg: HotspotPredictorConfig,
                        device, path: Path):
    """导出 TorchScript"""
    model.eval()
    dummy = torch.randn(1, cfg.in_channels, cfg.patch_size, cfg.patch_size).to(device)
    scripted = torch.jit.trace(model, dummy)
    scripted.save(str(path))
    logger.info(f"TorchScript 已导出: {path}")


def _export_onnx(model: HotspotCNN, cfg: HotspotPredictorConfig,
                 device, path: Path, opset: int = 17):
    """导出 ONNX（要求 onnx 包已安装）"""
    model.eval()
    dummy = torch.randn(1, cfg.in_channels, cfg.patch_size, cfg.patch_size).to(device)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    logger.info(f"ONNX 已导出: {path}")


def export_hotspot_predictor(
    model: HotspotCNN,
    output_dir: Union[str, Path],
    export_onnx: bool = True,
    export_torchscript: bool = True,
    onnx_opset: int = 17,
    device: str = 'auto',
) -> Dict[str, str]:
    """
    将训练好的模型导出为部署格式

    Args:
        model: 训练好的 HotspotCNN
        output_dir: 输出目录
        export_onnx: 是否导出 ONNX
        export_torchscript: 是否导出 TorchScript
        onnx_opset: ONNX opset 版本
        device: 导出设备

    Returns:
        导出文件路径字典
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 不可用")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    dev = _get_device(device)
    model = model.to(dev).eval()

    paths: Dict[str, str] = {}

    with open(out_dir / "model_config.json", 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    ckpt_path = out_dir / "hotspot_predictor.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': cfg.to_dict(),
    }, ckpt_path)
    paths['checkpoint'] = str(ckpt_path)

    if export_torchscript:
        ts_path = out_dir / "hotspot_predictor.pt"
        _export_torchscript(model, cfg, dev, ts_path)
        paths['torchscript'] = str(ts_path)

    if export_onnx:
        onnx_path = out_dir / "hotspot_predictor.onnx"
        _export_onnx(model, cfg, dev, onnx_path, onnx_opset)
        paths['onnx'] = str(onnx_path)

    return paths


# ============================================================================
# 6. 全局版图热点扫描（在线推理）
# ============================================================================

def scan_layout_for_hotspots(
    model: HotspotCNN,
    mask_layout: np.ndarray,
    stride: Optional[int] = None,
    risk_threshold: float = 0.5,
    batch_size: int = 128,
    device: str = 'auto',
    use_distance_transform: bool = False,
    merge_overlap: int = 8,
    bbox_padding: int = 8,
    min_bbox_size: int = 16,
    return_heatmap: bool = True,
) -> HotspotScanResult:
    """
    使用训练好的 CNN 对完整版图进行滑窗扫描，输出高风险 bbox 列表

    这是 OPC / ILT 精修的前置过滤步骤：只将返回的高风险 bbox 送入
    OPC/ILT，其余区域保持原掩模不变，可大幅降低计算量。

    Args:
        model: 训练好的 HotspotCNN
        mask_layout: 完整掩模版图 (H, W)，二值或浮点数组
        stride: 滑窗步长，None 默认取 patch_size // 2
        risk_threshold: 高风险判定概率阈值 (0~1)，越低越敏感
        batch_size: 推理批大小（越大越快，占显存越多）
        device: 推理设备
        use_distance_transform: 是否拼接距离变换通道（需与训练一致）
        merge_overlap: 重叠像素阈值内的相邻高风险 patch 合并为同一 bbox
        bbox_padding: 最终 bbox 外扩像素
        min_bbox_size: 最小 bbox 尺寸（过滤噪声小方块）
        return_heatmap: 是否生成并返回风险热力图

    Returns:
        HotspotScanResult，包含高风险 bbox、评分、热力图等
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 不可用")

    t0 = time.time()
    cfg = model.config
    ps = cfg.patch_size
    stride = stride or max(1, ps // 2)

    dev = _get_device(device)
    model = model.to(dev).eval()

    mask = np.asarray(mask_layout, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask_layout 需要 2D 数组，当前 shape={mask.shape}")
    H, W = mask.shape

    # 边界填充确保覆盖完整
    pad_h = (ps - (H % ps)) if H % ps != 0 else 0
    pad_w = (ps - (W % ps)) if W % ps != 0 else 0
    mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
    HP, WP = mask_padded.shape

    # 输入预处理
    mask_in = (mask_padded >= 0.5).astype(np.float32)
    if use_distance_transform:
        dt = distance_transform_edt(mask_in > 0.5)
        dt_norm = dt / (dt.max() + 1e-8)
        input_stack = np.stack([mask_in, dt_norm.astype(np.float32)], axis=0)
    else:
        input_stack = mask_in[np.newaxis, ...]

    # 收集 patch 位置
    ys = list(range(0, HP - ps + 1, stride))
    xs = list(range(0, WP - ps + 1, stride))
    positions: List[Tuple[int, int]] = []
    for y in ys:
        for x in xs:
            positions.append((y, x))

    num_patches = len(positions)
    logger.info(f"开始扫描版图 {H}×{W} -> {HP}×{WP} (pad), "
                f"共 {num_patches} 个 patch (stride={stride})")

    # ---- 批推理 ----
    patch_preds: List[Tuple[int, int, float, int]] = []
    risk_mask = np.zeros((HP, WP), dtype=np.float32) if return_heatmap else None
    count_mask = np.zeros((HP, WP), dtype=np.float32) if return_heatmap else None

    num_batches = (num_patches + batch_size - 1) // batch_size
    high_risk_patch_count = 0

    with torch.no_grad():
        for bi in range(num_batches):
            start = bi * batch_size
            end = min(num_patches, start + batch_size)
            batch_pos = positions[start:end]
            actual_bs = len(batch_pos)

            batch_np = np.zeros(
                (actual_bs, cfg.in_channels, ps, ps), dtype=np.float32
            )
            for bi2, (y, x) in enumerate(batch_pos):
                batch_np[bi2] = input_stack[:, y:y + ps, x:x + ps]

            x_t = torch.from_numpy(batch_np).to(dev, non_blocking=True)
            probs = model(x_t, return_probs=True)[:, -1].detach().cpu().numpy()

            for (y, x), p in zip(batch_pos, probs):
                label = int(p >= risk_threshold)
                patch_preds.append((y, x, float(p), label))
                if label == 1:
                    high_risk_patch_count += 1
                if return_heatmap and risk_mask is not None:
                    risk_mask[y:y + ps, x:x + ps] += float(p)
                    count_mask[y:y + ps, x:x + ps] += 1.0

    if return_heatmap and risk_mask is not None:
        valid = count_mask > 0
        risk_mask[valid] /= count_mask[valid]
        risk_heatmap = risk_mask[:H, :W]
        risk_heatmap = gaussian_filter(risk_heatmap, sigma=1.0)
    else:
        risk_heatmap = None

    # ---- bbox 聚合 ----
    bboxes, scores = _aggregate_patch_bboxes(
        patch_preds, ps, (H, W), merge_overlap, bbox_padding, min_bbox_size,
        risk_threshold
    )

    # 计算高风险区域面积占比
    if risk_heatmap is not None:
        high_area = (risk_heatmap >= risk_threshold).sum()
        ratio = float(high_area / max(1, H * W))
    else:
        merged = np.zeros((H, W), dtype=bool)
        for (y0, y1, x0, x1) in bboxes:
            merged[y0:y1, x0:x1] = True
        ratio = float(merged.sum() / max(1, H * W))

    result = HotspotScanResult(
        high_risk_bboxes=bboxes,
        risk_scores=scores,
        risk_heatmap=risk_heatmap,
        patch_predictions=patch_preds,
        num_patches_scanned=num_patches,
        num_high_risk_patches=high_risk_patch_count,
        high_risk_area_ratio=ratio,
        scan_time_sec=time.time() - t0,
    )
    logger.info(
        f"扫描完成: 耗时 {result.scan_time_sec:.2f}s, "
        f"高风险 patch={high_risk_patch_count}/{num_patches} "
        f"({100 * high_risk_patch_count / max(1, num_patches):.2f}%), "
        f"合并后 bbox={len(bboxes)} 个, "
        f"高风险面积占比={100 * ratio:.2f}%"
    )
    return result


def _aggregate_patch_bboxes(
    patch_preds: List[Tuple[int, int, float, int]],
    patch_size: int,
    original_shape: Tuple[int, int],
    merge_overlap: int,
    bbox_padding: int,
    min_bbox_size: int,
    risk_threshold: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """
    将高风险 patch 聚合成合并后的 bbox 列表

    方法：
    1. 绘制逐 patch 置信度的高风险掩码
    2. 阈值化 → 连通域分析 → bbox + 平均评分
    3. 膨胀合并相近区域 → 外扩 padding → 过滤小 bbox
    """
    H, W = original_shape
    ps = patch_size
    canvas = np.zeros((H, W), dtype=np.float32)
    hit = np.zeros((H, W), dtype=np.float32)

    for (y, x, p, label) in patch_preds:
        if label != 1:
            continue
        y0, y1 = y, min(y + ps, H)
        x0, x1 = x, min(x + ps, W)
        canvas[y0:y1, x0:x1] += p
        hit[y0:y1, x0:x1] += 1.0

    if canvas.sum() <= 0:
        return [], []

    valid = hit > 0
    canvas[valid] /= hit[valid]
    canvas = gaussian_filter(canvas, sigma=1.0)
    bin_mask = canvas >= risk_threshold * 0.6  # 聚合时放宽阈值，避免间隙

    struct = generate_binary_structure(2, 2)
    if merge_overlap > 0:
        bin_mask = binary_dilation(bin_mask, structure=struct,
                                   iterations=max(1, merge_overlap // 2))

    labeled, ncomps = label(bin_mask, structure=struct)
    objs = find_objects(labeled)

    bboxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []

    for comp_idx, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        (sy, sx) = sl
        y0, y1 = sy.start, sy.stop
        x0, x1 = sx.start, sx.stop

        comp_mask = (labeled[y0:y1, x0:x1] == comp_idx)
        if comp_mask.sum() == 0:
            continue

        avg_score = float(canvas[y0:y1, x0:x1][comp_mask].mean())

        y0 = max(0, y0 - bbox_padding)
        y1 = min(H, y1 + bbox_padding)
        x0 = max(0, x0 - bbox_padding)
        x1 = min(W, x1 + bbox_padding)

        if (y1 - y0) < min_bbox_size or (x1 - x0) < min_bbox_size:
            continue

        bboxes.append((int(y0), int(y1), int(x0), int(x1)))
        scores.append(avg_score)

    order = sorted(range(len(bboxes)), key=lambda i: -scores[i])
    bboxes = [bboxes[i] for i in order]
    scores = [scores[i] for i in order]
    return bboxes, scores
