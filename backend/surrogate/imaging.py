# -*- coding: utf-8 -*-
"""
代理模型推理接口

SurrogateImaging 类:
- 与 PartialCoherentImaging 类似的接口，方便直接替换
- predict(mask) -> aerial_image，快速神经网络推理
- compare_with_ground_truth(mask) -> (surrogate_aerial, real_aerial, metrics)
  对比代理模型与真实成像的精度

主要特性:
- 自动设备选择（CPU/CUDA/MPS）
- 批量推理支持（多张掩模同时预测）
- 推理时间统计
- 与真实成像结果的对比评估
"""

import os
import sys
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Union
import copy

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.imaging import (
    OpticalSystem,
    PartialCoherentImaging,
)

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch 未安装，无法使用 SurrogateImaging。\n"
        "请安装: pip install torch"
    )

from .train import (
    load_trained_model,
    select_device,
    ssim_numpy,
    psnr_numpy,
)
from .model import SurrogateModelConfig, build_model


@dataclass
class SurrogateComparisonResult:
    """代理模型 vs 真实模型对比结果"""
    surrogate_aerial: np.ndarray
    real_aerial: np.ndarray
    mask: np.ndarray
    metrics: Dict[str, float]
    surrogate_time_ms: float
    real_time_ms: float
    speedup: float

    def summary(self) -> str:
        m = self.metrics
        return (
            f"代理模型评估:\n"
            f"  加速比: {self.speedup:.2f}x "
            f"(代理={self.surrogate_time_ms:.2f}ms vs 真实={self.real_time_ms:.2f}ms)\n"
            f"  精度: MSE={m['mse']:.6f}, MAE={m['mae']:.6f}\n"
            f"        SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.2f} dB"
        )


class SurrogateImaging:
    """
    神经网络代理成像模型，接口与 PartialCoherentImaging 对齐

    用法 1: 从训练好的 checkpoint 加载:
        imaging = SurrogateImaging.from_checkpoint('path/to/best_model.pt')
        aerial = imaging.predict(mask)

    用法 2: 传入已加载的 PyTorch 模型:
        model, info = load_trained_model('path/to/best_model.pt')
        imaging = SurrogateImaging(model, optical_system=optics)
        aerial = imaging.predict(mask)

    用法 3: 精度对比:
        result = imaging.compare_with_ground_truth(mask)
        print(result.summary())
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optical_system: Optional[OpticalSystem] = None,
        image_size: Optional[Tuple[int, int]] = None,
        device: str = 'auto',
        model_config: Optional[SurrogateModelConfig] = None,
    ):
        """
        Args:
            model: 已加载并加载好权重的 PyTorch 模型
            optical_system: 对应的光学系统参数（可选，用于真实对比 & metadata）
            image_size: 图像尺寸，None 则在第一次推理时自动推断
            device: 推理设备
            model_config: 模型配置
        """
        self.model = model
        self.optical_system = optical_system
        self._model_config = model_config
        self._device = select_device(device)
        self.model = self.model.to(self._device)
        self.model.eval()

        self._image_size = image_size
        self._real_imaging_model: Optional[PartialCoherentImaging] = None

        self._inference_count = 0
        self._total_inference_time_ms = 0.0

        logger.info(
            f"SurrogateImaging 初始化完成: 设备={self._device}, "
            f"参数={sum(p.numel() for p in model.parameters()):,}"
        )

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        optical_system: Optional[OpticalSystem] = None,
        device: str = 'auto',
    ) -> 'SurrogateImaging':
        """
        从 checkpoint 文件加载并构建 SurrogateImaging

        Args:
            checkpoint_path: .pt 文件路径
            optical_system: 可选的光学系统参数
            device: 推理设备

        Returns:
            SurrogateImaging 实例
        """
        model, extra = load_trained_model(checkpoint_path, device=device)
        model_cfg = extra.get('model_config')
        return cls(
            model=model,
            optical_system=optical_system,
            device=device,
            model_config=model_cfg,
        )

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def average_inference_time_ms(self) -> float:
        """平均推理时间（毫秒）"""
        if self._inference_count == 0:
            return 0.0
        return self._total_inference_time_ms / self._inference_count

    def _prepare_input(
        self, mask: Union[np.ndarray, List[np.ndarray]]
    ) -> Tuple[torch.Tensor, int]:
        """将输入掩模转换为模型需要的 (B, 1, H, W) tensor"""
        if isinstance(mask, list):
            masks = [np.asarray(m, dtype=np.float32) for m in mask]
            if self._image_size is None:
                self._image_size = masks[0].shape
            batch = np.stack([m for m in masks], axis=0)
        elif isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                if self._image_size is None:
                    self._image_size = mask.shape
                batch = mask.astype(np.float32)[np.newaxis, ...]
            elif mask.ndim == 3:
                if self._image_size is None:
                    self._image_size = mask.shape[1:]
                batch = mask.astype(np.float32)
            elif mask.ndim == 4:
                if self._image_size is None:
                    self._image_size = mask.shape[2:]
                batch = mask.astype(np.float32)[:, 0] if mask.shape[1] == 1 else mask[:, :, :, 0]
            else:
                raise ValueError(
                    f"mask 维度不支持: {mask.ndim}，支持 2/3/4 维"
                )
        else:
            raise TypeError(f"mask 类型不支持: {type(mask)}")

        if batch.ndim == 3:
            batch = batch[:, np.newaxis, :, :]

        batch_tensor = torch.from_numpy(batch).to(self._device)
        B = batch_tensor.size(0)
        return batch_tensor, B

    def predict(
        self,
        mask: Union[np.ndarray, List[np.ndarray]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        预测空间像

        Args:
            mask: 掩模，支持:
                - 单张 (H, W) float32/64 ndarray -> 输出 (H, W)
                - 多张 (N, H, W) ndarray 或 list of (H, W) -> 输出 (N, H, W)
            batch_size: 批大小，None 则一次推理全部

        Returns:
            空间像 ndarray，值域 [0, 1]
        """
        was_single = isinstance(mask, np.ndarray) and mask.ndim == 2

        with torch.no_grad():
            input_tensor, B = self._prepare_input(mask)

            if batch_size is None or B <= batch_size:
                t0 = time.time()
                output = self.model(input_tensor)
                elapsed = (time.time() - t0) * 1000
                self._inference_count += B
                self._total_inference_time_ms += elapsed * B
                output_np = output.cpu().numpy()
            else:
                outputs = []
                for start in range(0, B, batch_size):
                    end = min(start + batch_size, B)
                    chunk = input_tensor[start:end]
                    t0 = time.time()
                    out_chunk = self.model(chunk)
                    elapsed = (time.time() - t0) * 1000
                    self._inference_count += (end - start)
                    self._total_inference_time_ms += elapsed * (end - start)
                    outputs.append(out_chunk.cpu().numpy())
                output_np = np.concatenate(outputs, axis=0)

        result = output_np[:, 0]
        result = np.clip(result, 0.0, 1.0)

        if was_single:
            return result[0]
        return result

    def __call__(
        self,
        mask: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """与 PartialCoherentImaging 对齐：直接调用 compute_aerial_image"""
        return self.predict(mask)

    # ------------------------------------------------------------------
    # 兼容 PartialCoherentImaging 接口
    # ------------------------------------------------------------------

    def compute_aerial_image(self, mask: np.ndarray) -> np.ndarray:
        """与 PartialCoherentImaging.compute_aerial_image 接口一致"""
        return self.predict(mask)

    def compute_image_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像对掩模的梯度（用于 OPC/SMO 优化）

        使用 PyTorch autograd 计算 d(aerial)/d(mask)

        Args:
            mask: (H, W) 掩模

        Returns:
            梯度数组 (H, W)
        """
        self.model.eval()

        mask_tensor = torch.from_numpy(
            mask.astype(np.float32)[np.newaxis, np.newaxis, ...]
        ).to(self._device)
        mask_tensor.requires_grad_(True)

        aerial = self.model(mask_tensor)
        aerial.sum().backward()

        grad = mask_tensor.grad.detach().cpu().numpy()[0, 0]
        return grad.astype(np.float64)

    # ------------------------------------------------------------------
    # 精度对比
    # ------------------------------------------------------------------

    def _ensure_real_imaging(self, image_size: Tuple[int, int]):
        """确保真实成像模型已初始化"""
        if self._real_imaging_model is None:
            if self.optical_system is None:
                raise ValueError(
                    "需要设置 optical_system 才能进行真实成像对比"
                )
            self._real_imaging_model = PartialCoherentImaging(
                self.optical_system, image_size
            )

    def compare_with_ground_truth(
        self,
        mask: np.ndarray,
        verbose: bool = True,
    ) -> SurrogateComparisonResult:
        """
        对比代理模型与真实 PartialCoherentImaging 的精度

        Args:
            mask: 输入掩模 (H, W)
            verbose: 是否打印结果

        Returns:
            SurrogateComparisonResult
        """
        image_size = mask.shape
        self._ensure_real_imaging(image_size)

        t0 = time.time()
        surrogate_aerial = self.predict(mask)
        surrogate_time = (time.time() - t0) * 1000

        t0 = time.time()
        real_aerial = self._real_imaging_model.compute_aerial_image(mask)
        real_time = (time.time() - t0) * 1000

        mse_val = float(np.mean((surrogate_aerial - real_aerial) ** 2))
        mae_val = float(np.mean(np.abs(surrogate_aerial - real_aerial)))
        try:
            ssim_val = ssim_numpy(surrogate_aerial, real_aerial)
        except Exception:
            ssim_val = float('nan')
        psnr_val = psnr_numpy(surrogate_aerial, real_aerial)

        metrics = {
            'mse': mse_val,
            'mae': mae_val,
            'ssim': ssim_val,
            'psnr': psnr_val,
        }
        speedup = real_time / max(surrogate_time, 1e-6)

        result = SurrogateComparisonResult(
            surrogate_aerial=surrogate_aerial,
            real_aerial=real_aerial,
            mask=mask,
            metrics=metrics,
            surrogate_time_ms=surrogate_time,
            real_time_ms=real_time,
            speedup=speedup,
        )

        if verbose:
            logger.info(result.summary())

        return result

    def batch_compare(
        self,
        masks: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        在批量数据上对比代理模型与真实成像

        Args:
            masks: (N, H, W) 掩模数组
            verbose: 是否打印结果

        Returns:
            聚合指标字典: MSE/MAE/SSIM/PSNR 均值 + 加速比
        """
        N = len(masks)
        if N == 0:
            return {}

        results: List[SurrogateComparisonResult] = []
        for i, mask in enumerate(masks):
            r = self.compare_with_ground_truth(mask, verbose=False)
            results.append(r)
            if verbose and (i + 1) % max(1, N // 10) == 0:
                logger.info(f"对比进度: {i + 1}/{N}")

        mses = [r.metrics['mse'] for r in results]
        maes = [r.metrics['mae'] for r in results]
        ssims = [r.metrics['ssim'] for r in results if not np.isnan(r.metrics['ssim'])]
        psnrs = [r.metrics['psnr'] for r in results]
        speedups = [r.speedup for r in results]

        agg = {
            'num_samples': N,
            'mse_mean': float(np.mean(mses)),
            'mse_std': float(np.std(mses)),
            'mae_mean': float(np.mean(maes)),
            'ssim_mean': float(np.mean(ssims)) if ssims else float('nan'),
            'psnr_mean': float(np.mean(psnrs)),
            'speedup_mean': float(np.mean(speedups)),
            'speedup_median': float(np.median(speedups)),
            'surrogate_time_ms_avg': float(np.mean([r.surrogate_time_ms for r in results])),
            'real_time_ms_avg': float(np.mean([r.real_time_ms for r in results])),
        }

        if verbose:
            logger.info(
                f"批量对比完成 (N={N}):\n"
                f"  MSE: {agg['mse_mean']:.6f} ± {agg['mse_std']:.6f}\n"
                f"  SSIM: {agg['ssim_mean']:.4f}, PSNR: {agg['psnr_mean']:.2f} dB\n"
                f"  加速比: 均值 {agg['speedup_mean']:.2f}x / 中位数 {agg['speedup_median']:.2f}x"
            )

        return agg

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def reset_stats(self):
        """重置推理统计"""
        self._inference_count = 0
        self._total_inference_time_ms = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        return {
            'inference_count': self._inference_count,
            'total_time_ms': self._total_inference_time_ms,
            'avg_time_ms': self.average_inference_time_ms,
            'device': str(self._device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
        }


# ======================================================================
# 在线自适应模块：经验回放、增量微调、精度监控、多保真度切换
# ======================================================================

from enum import Enum
from collections import deque
from dataclasses import asdict
import random


class FidelityMode(Enum):
    """保真度模式枚举"""
    SURROGATE_ONLY = "surrogate_only"
    ADAPTIVE = "adaptive"
    GROUND_TRUTH = "ground_truth"


@dataclass
class ExperienceReplayConfig:
    """经验回放缓冲区配置"""
    capacity: int = 10000
    batch_size: int = 32
    priority_sampling: bool = False
    alpha: float = 0.6
    beta: float = 0.4


@dataclass
class OnlineFineTuningConfig:
    """在线微调配置"""
    enabled: bool = True
    update_interval: int = 50
    min_samples_before_update: int = 64
    learning_rate: float = 1e-5
    num_steps: int = 10
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    freeze_encoder: bool = False
    replay_ratio: float = 0.8
    loss_type: str = "mse"


@dataclass
class AccuracyMonitorConfig:
    """精度监控配置"""
    enabled: bool = True
    check_interval: int = 20
    min_samples_for_check: int = 16
    mse_threshold: float = 0.001
    ssim_threshold: float = 0.95
    psnr_threshold: float = 30.0
    consecutive_failures_before_fallback: int = 3
    window_size: int = 10
    auto_recovery: bool = True
    recovery_check_interval: int = 50


@dataclass
class AdaptiveSurrogateConfig:
    """自适应代理模型完整配置"""
    fidelity_mode: FidelityMode = FidelityMode.ADAPTIVE
    experience_replay: ExperienceReplayConfig = field(
        default_factory=ExperienceReplayConfig
    )
    fine_tuning: OnlineFineTuningConfig = field(
        default_factory=OnlineFineTuningConfig
    )
    accuracy_monitor: AccuracyMonitorConfig = field(
        default_factory=AccuracyMonitorConfig
    )
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'fidelity_mode': self.fidelity_mode.value,
            'experience_replay': asdict(self.experience_replay),
            'fine_tuning': asdict(self.fine_tuning),
            'accuracy_monitor': asdict(self.accuracy_monitor),
            'verbose': self.verbose,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AdaptiveSurrogateConfig':
        cfg = cls()
        if 'fidelity_mode' in d:
            cfg.fidelity_mode = FidelityMode(d['fidelity_mode'])
        if 'experience_replay' in d:
            cfg.experience_replay = ExperienceReplayConfig(
                **d['experience_replay']
            )
        if 'fine_tuning' in d:
            cfg.fine_tuning = OnlineFineTuningConfig(**d['fine_tuning'])
        if 'accuracy_monitor' in d:
            cfg.accuracy_monitor = AccuracyMonitorConfig(
                **d['accuracy_monitor']
            )
        if 'verbose' in d:
            cfg.verbose = d['verbose']
        return cfg


class ExperienceReplayBuffer:
    """
    经验回放缓冲区

    存储 (mask, aerial_image) 样本对，支持：
    - FIFO 管理，自动淘汰旧样本
    - 随机采样用于经验回放训练
    - 优先级采样（可选），优先采样误差大的样本
    """

    def __init__(self, config: ExperienceReplayConfig):
        self.config = config
        self.buffer: deque = deque(maxlen=config.capacity)
        self.priorities: deque = deque(maxlen=config.capacity)
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        mask: np.ndarray,
        aerial: np.ndarray,
        priority: Optional[float] = None,
    ):
        """添加样本到缓冲区"""
        self.buffer.append((mask.astype(np.float32), aerial.astype(np.float32)))
        if priority is None:
            priority = 1.0
        self.priorities.append(priority)

    def add_batch(
        self,
        masks: np.ndarray,
        aerials: np.ndarray,
        priorities: Optional[np.ndarray] = None,
    ):
        """批量添加样本"""
        if priorities is None:
            priorities = np.ones(len(masks))
        for i in range(len(masks)):
            self.add(masks[i], aerials[i], priorities[i])

    def sample(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从缓冲区采样一批样本

        Returns:
            (masks, aerials) - (B, H, W) 数组
        """
        if len(self.buffer) == 0:
            raise ValueError("缓冲区为空，无法采样")

        bs = batch_size or self.config.batch_size
        bs = min(bs, len(self.buffer))

        if self.config.priority_sampling and len(self.priorities) > 0:
            priorities = np.array(self.priorities)
            probs = priorities ** self.config.alpha
            probs = probs / probs.sum()
            indices = self.rng.choice(
                len(self.buffer), size=bs, replace=True, p=probs
            )
        else:
            indices = self.rng.choice(
                len(self.buffer), size=bs, replace=True
            )

        masks = []
        aerials = []
        for idx in indices:
            mask, aerial = self.buffer[idx]
            masks.append(mask)
            aerials.append(aerial)

        return np.stack(masks), np.stack(aerials)

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """更新样本优先级"""
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = max(prio, 1e-8)

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.priorities.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            'size': len(self.buffer),
            'capacity': self.config.capacity,
            'fill_ratio': len(self.buffer) / self.config.capacity,
        }


@dataclass
class FineTuningResult:
    """微调结果"""
    success: bool
    num_steps: int
    initial_loss: float
    final_loss: float
    loss_history: List[float]
    time_sec: float
    message: str = ""

    def summary(self) -> str:
        if not self.success:
            return f"微调失败: {self.message}"
        return (
            f"微调完成: {self.num_steps} 步, "
            f"损失 {self.initial_loss:.6f} → {self.final_loss:.6f}, "
            f"耗时 {self.time_sec:.2f}s"
        )


class OnlineFineTuner:
    """
    在线微调器

    基于经验回放缓冲区对代理模型进行增量微调，支持：
    - 小批量快速微调
    - 冻结编码器防止灾难性遗忘
    - 经验回放混合新样本训练
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: OnlineFineTuningConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.update_counter = 0
        self.rng = np.random.default_rng()

        self.optimizer = self._build_optimizer()
        self.loss_fn = self._build_loss_fn()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建优化器，支持冻结编码器"""
        if self.config.freeze_encoder:
            params = []
            for name, p in self.model.named_parameters():
                if 'decoder' in name or 'outc' in name:
                    params.append(p)
                else:
                    p.requires_grad_(False)
        else:
            params = self.model.parameters()

        return torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_loss_fn(self):
        """构建损失函数"""
        loss_type = self.config.loss_type.lower()
        if loss_type == 'mse':
            return torch.nn.MSELoss()
        elif loss_type == 'l1':
            return torch.nn.L1Loss()
        elif loss_type == 'huber':
            return torch.nn.HuberLoss(delta=0.1)
        else:
            logger.warning(f"未知损失类型 {loss_type}，使用 MSE")
            return torch.nn.MSELoss()

    def fine_tune(
        self,
        replay_buffer: ExperienceReplayBuffer,
        new_masks: Optional[np.ndarray] = None,
        new_aerials: Optional[np.ndarray] = None,
    ) -> FineTuningResult:
        """
        执行一次微调

        Args:
            replay_buffer: 经验回放缓冲区
            new_masks: 新采集的掩模样本
            new_aerials: 新采集的空间像样本

        Returns:
            FineTuningResult
        """
        if not self.config.enabled:
            return FineTuningResult(
                success=False, num_steps=0,
                initial_loss=0.0, final_loss=0.0,
                loss_history=[], time_sec=0.0,
                message="微调未启用"
            )

        t0 = time.time()
        self.model.train()
        loss_history: List[float] = []

        try:
            if len(replay_buffer) < self.config.min_samples_before_update:
                return FineTuningResult(
                    success=False, num_steps=0,
                    initial_loss=0.0, final_loss=0.0,
                    loss_history=[], time_sec=0.0,
                    message=f"样本不足: {len(replay_buffer)} < {self.config.min_samples_before_update}"
                )

            replay_size = int(self.config.batch_size * self.config.replay_ratio)
            new_size = self.config.batch_size - replay_size

            initial_loss = None
            final_loss = None

            for step in range(self.config.num_steps):
                self.optimizer.zero_grad()

                replay_masks, replay_aerials = replay_buffer.sample(replay_size)

                if (new_masks is not None and new_aerials is not None
                        and len(new_masks) > 0 and new_size > 0):
                    new_indices = self.rng.choice(
                        len(new_masks),
                        size=min(new_size, len(new_masks)),
                        replace=False
                    )
                    batch_masks = np.concatenate([
                        replay_masks, new_masks[new_indices]
                    ])
                    batch_aerials = np.concatenate([
                        replay_aerials, new_aerials[new_indices]
                    ])
                else:
                    batch_masks = replay_masks
                    batch_aerials = replay_aerials

                mask_tensor = torch.from_numpy(
                    batch_masks[:, np.newaxis, :, :]
                ).to(self.device)
                aerial_tensor = torch.from_numpy(
                    batch_aerials[:, np.newaxis, :, :]
                ).to(self.device)

                pred = self.model(mask_tensor)
                loss = self.loss_fn(pred, aerial_tensor)

                if initial_loss is None:
                    initial_loss = float(loss.item())

                loss.backward()

                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                self.optimizer.step()

                loss_val = float(loss.item())
                loss_history.append(loss_val)
                final_loss = loss_val

            self.update_counter += 1
            self.model.eval()

            return FineTuningResult(
                success=True,
                num_steps=self.config.num_steps,
                initial_loss=initial_loss or 0.0,
                final_loss=final_loss or 0.0,
                loss_history=loss_history,
                time_sec=time.time() - t0,
            )

        except Exception as e:
            logger.error(f"微调失败: {e}")
            self.model.eval()
            return FineTuningResult(
                success=False, num_steps=0,
                initial_loss=0.0, final_loss=0.0,
                loss_history=loss_history,
                time_sec=time.time() - t0,
                message=str(e)
            )

    def reset_optimizer(self):
        """重置优化器状态"""
        self.optimizer = self._build_optimizer()

    def get_stats(self) -> Dict[str, Any]:
        return {
            'update_counter': self.update_counter,
            'learning_rate': self.config.learning_rate,
            'freeze_encoder': self.config.freeze_encoder,
        }


@dataclass
class AccuracyCheckResult:
    """精度检查结果"""
    passed: bool
    num_samples: int
    mse_mean: float
    mse_std: float
    ssim_mean: float
    psnr_mean: float
    metrics: Dict[str, float]
    failures: List[int]
    time_sec: float
    message: str = ""

    def summary(self) -> str:
        status = "通过" if self.passed else "未通过"
        return (
            f"精度检查{status} (N={self.num_samples}): "
            f"MSE={self.mse_mean:.6f}±{self.mse_std:.6f}, "
            f"SSIM={self.ssim_mean:.4f}, PSNR={self.psnr_mean:.2f}dB, "
            f"耗时 {self.time_sec:.2f}s"
        )


class AccuracyMonitor:
    """
    精度监控器

    定期采样真实仿真验证代理模型精度，支持：
    - 多指标监控（MSE/SSIM/PSNR）
    - 滑动窗口统计
    - 连续失败检测
    - 自动回退触发
    """

    def __init__(
        self,
        config: AccuracyMonitorConfig,
        real_imaging: PartialCoherentImaging,
    ):
        self.config = config
        self.real_imaging = real_imaging
        self.check_counter = 0
        self.consecutive_failures = 0
        self.total_failures = 0
        self.total_checks = 0
        self.is_fallback_mode = False
        self.recovery_check_counter = 0

        self.mse_history: deque = deque(maxlen=config.window_size)
        self.ssim_history: deque = deque(maxlen=config.window_size)
        self.psnr_history: deque = deque(maxlen=config.window_size)

    def check_accuracy(
        self,
        surrogate_predict_fn,
        masks: np.ndarray,
    ) -> AccuracyCheckResult:
        """
        检查代理模型精度

        Args:
            surrogate_predict_fn: 代理模型预测函数
            masks: 用于检查的掩模样本

        Returns:
            AccuracyCheckResult
        """
        if not self.config.enabled:
            return AccuracyCheckResult(
                passed=True, num_samples=0,
                mse_mean=0.0, mse_std=0.0,
                ssim_mean=1.0, psnr_mean=100.0,
                metrics={}, failures=[], time_sec=0.0,
            )

        if len(masks) < self.config.min_samples_for_check:
            return AccuracyCheckResult(
                passed=True, num_samples=len(masks),
                mse_mean=0.0, mse_std=0.0,
                ssim_mean=1.0, psnr_mean=100.0,
                metrics={}, failures=[], time_sec=0.0,
                message=f"样本不足: {len(masks)} < {self.config.min_samples_for_check}"
            )

        t0 = time.time()
        self.check_counter += 1
        self.total_checks += 1

        mses = []
        ssims = []
        psnrs = []
        failures = []

        for i, mask in enumerate(masks):
            surrogate_aerial = surrogate_predict_fn(mask)
            real_aerial = self.real_imaging.compute_aerial_image(mask)

            mse = float(np.mean((surrogate_aerial - real_aerial) ** 2))
            mses.append(mse)

            try:
                ssim = ssim_numpy(surrogate_aerial, real_aerial)
            except Exception:
                ssim = float('nan')
            ssims.append(ssim)

            psnr = psnr_numpy(surrogate_aerial, real_aerial)
            psnrs.append(psnr)

            if (mse > self.config.mse_threshold
                    or ssim < self.config.ssim_threshold
                    or psnr < self.config.psnr_threshold):
                failures.append(i)

        mse_mean = float(np.mean(mses))
        mse_std = float(np.std(mses))
        ssim_mean = float(np.nanmean(ssims)) if not np.isnan(ssims).all() else float('nan')
        psnr_mean = float(np.mean(psnrs))

        self.mse_history.append(mse_mean)
        self.ssim_history.append(ssim_mean)
        self.psnr_history.append(psnr_mean)

        passed = (
            mse_mean <= self.config.mse_threshold
            and ssim_mean >= self.config.ssim_threshold
            and psnr_mean >= self.config.psnr_threshold
        )

        if passed:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.total_failures += 1

        if (self.consecutive_failures
                >= self.config.consecutive_failures_before_fallback):
            self.is_fallback_mode = True
            logger.warning(
                f"精度连续 {self.consecutive_failures} 次不达标，"
                f"触发回退到全精度仿真"
            )

        metrics = {
            'mse_mean': mse_mean,
            'mse_std': mse_std,
            'ssim_mean': ssim_mean,
            'psnr_mean': psnr_mean,
            'failure_rate': len(failures) / len(masks),
        }

        return AccuracyCheckResult(
            passed=passed,
            num_samples=len(masks),
            mse_mean=mse_mean,
            mse_std=mse_std,
            ssim_mean=ssim_mean,
            psnr_mean=psnr_mean,
            metrics=metrics,
            failures=failures,
            time_sec=time.time() - t0,
        )

    def check_recovery(
        self,
        surrogate_predict_fn,
        masks: np.ndarray,
    ) -> bool:
        """检查是否可以从回退模式恢复"""
        if not self.is_fallback_mode:
            return True

        if not self.config.auto_recovery:
            return False

        self.recovery_check_counter += 1
        if self.recovery_check_counter < self.config.recovery_check_interval:
            return False

        self.recovery_check_counter = 0
        result = self.check_accuracy(surrogate_predict_fn, masks)

        if result.passed:
            self.is_fallback_mode = False
            self.consecutive_failures = 0
            logger.info(f"精度恢复，切换回代理模式")
            return True
        return False

    def reset(self):
        """重置监控状态"""
        self.check_counter = 0
        self.consecutive_failures = 0
        self.total_failures = 0
        self.total_checks = 0
        self.is_fallback_mode = False
        self.recovery_check_counter = 0
        self.mse_history.clear()
        self.ssim_history.clear()
        self.psnr_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            'check_counter': self.check_counter,
            'total_checks': self.total_checks,
            'total_failures': self.total_failures,
            'consecutive_failures': self.consecutive_failures,
            'is_fallback_mode': self.is_fallback_mode,
            'mse_window_mean': float(np.mean(self.mse_history)) if self.mse_history else 0.0,
            'ssim_window_mean': float(np.nanmean(self.ssim_history)) if self.ssim_history else 0.0,
            'psnr_window_mean': float(np.mean(self.psnr_history)) if self.psnr_history else 0.0,
        }


@dataclass
class AdaptiveUpdateResult:
    """自适应更新结果"""
    update_triggered: bool
    accuracy_checked: bool
    accuracy_result: Optional[AccuracyCheckResult]
    fine_tuning_result: Optional[FineTuningResult]
    mode_switched: bool
    new_mode: Optional[FidelityMode]
    new_samples_added: int

    def summary(self) -> str:
        parts = []
        if self.accuracy_checked and self.accuracy_result:
            parts.append(self.accuracy_result.summary())
        if self.update_triggered and self.fine_tuning_result:
            parts.append(self.fine_tuning_result.summary())
        if self.mode_switched:
            parts.append(f"模式切换 → {self.new_mode.value if self.new_mode else 'N/A'}")
        if self.new_samples_added > 0:
            parts.append(f"新增样本: {self.new_samples_added}")
        return " | ".join(parts) if parts else "无更新"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'update_triggered': self.update_triggered,
            'accuracy_checked': self.accuracy_checked,
            'accuracy_result': (
                asdict(self.accuracy_result)
                if self.accuracy_result else None
            ),
            'fine_tuning_result': (
                asdict(self.fine_tuning_result)
                if self.fine_tuning_result else None
            ),
            'mode_switched': self.mode_switched,
            'new_mode': self.new_mode.value if self.new_mode else None,
            'new_samples_added': self.new_samples_added,
        }


class AdaptiveSurrogateImaging(SurrogateImaging):
    """
    自适应代理成像模型，支持在线更新和多保真度切换

    核心特性：
    1. 经验回放：存储历史样本用于增量训练
    2. 在线微调：定期用真实仿真数据更新模型
    3. 精度监控：实时跟踪模型精度
    4. 多保真度切换：精度退化时自动回退到全精度仿真

    用法：
        imaging = AdaptiveSurrogateImaging.from_checkpoint(
            'path/to/model.pt', optical_system=optics
        )
        aerial = imaging.predict(mask)  # 自动选择保真度
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optical_system: OpticalSystem,
        image_size: Optional[Tuple[int, int]] = None,
        device: str = 'auto',
        model_config: Optional[SurrogateModelConfig] = None,
        adaptive_config: Optional[AdaptiveSurrogateConfig] = None,
    ):
        super().__init__(
            model=model,
            optical_system=optical_system,
            image_size=image_size,
            device=device,
            model_config=model_config,
        )

        if optical_system is None:
            raise ValueError(
                "AdaptiveSurrogateImaging 需要 optical_system 用于真实仿真"
            )

        self.adaptive_config = adaptive_config or AdaptiveSurrogateConfig()
        self._current_mode = self.adaptive_config.fidelity_mode

        self._ensure_real_imaging(image_size or (128, 128))

        self.replay_buffer = ExperienceReplayBuffer(
            self.adaptive_config.experience_replay
        )
        self.fine_tuner = OnlineFineTuner(
            self.model, self.adaptive_config.fine_tuning, self._device
        )
        self.accuracy_monitor = AccuracyMonitor(
            self.adaptive_config.accuracy_monitor, self._real_imaging_model
        )

        self._inference_since_last_update = 0
        self._inference_since_last_check = 0
        self._recent_masks: List[np.ndarray] = []
        self._pending_new_masks: List[np.ndarray] = []
        self._pending_new_aerials: List[np.ndarray] = []
        self._last_update_result: Optional[AdaptiveUpdateResult] = None

        self.rng = np.random.default_rng()

        logger.info(
            f"AdaptiveSurrogateImaging 初始化完成: "
            f"初始模式={self._current_mode.value}, "
            f"设备={self._device}"
        )

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        optical_system: OpticalSystem,
        device: str = 'auto',
        adaptive_config: Optional[AdaptiveSurrogateConfig] = None,
    ) -> 'AdaptiveSurrogateImaging':
        """从 checkpoint 加载并构建 AdaptiveSurrogateImaging"""
        model, extra = load_trained_model(checkpoint_path, device=device)
        model_cfg = extra.get('model_config')
        return cls(
            model=model,
            optical_system=optical_system,
            device=device,
            model_config=model_cfg,
            adaptive_config=adaptive_config,
        )

    # ------------------------------------------------------------------
    # 模式管理
    # ------------------------------------------------------------------

    @property
    def current_mode(self) -> FidelityMode:
        return self._current_mode

    @property
    def is_fallback_mode(self) -> bool:
        return (
            self._current_mode == FidelityMode.GROUND_TRUTH
            or self.accuracy_monitor.is_fallback_mode
        )

    def set_mode(self, mode: FidelityMode):
        """手动设置保真度模式"""
        old_mode = self._current_mode
        self._current_mode = mode
        if mode == FidelityMode.GROUND_TRUTH:
            self.accuracy_monitor.is_fallback_mode = True
        else:
            self.accuracy_monitor.is_fallback_mode = False
        logger.info(f"保真度模式切换: {old_mode.value} → {mode.value}")

    # ------------------------------------------------------------------
    # 自适应预测接口
    # ------------------------------------------------------------------

    def predict(
        self,
        mask: Union[np.ndarray, List[np.ndarray]],
        force_mode: Optional[FidelityMode] = None,
        enable_adaptive: bool = True,
    ) -> np.ndarray:
        """
        预测空间像，支持自适应保真度选择

        Args:
            mask: 输入掩模
            force_mode: 强制使用指定模式
            enable_adaptive: 是否启应逻辑

        Returns:
            空间像 ndarray
        """
        mode = force_mode or self._current_mode

        if mode == FidelityMode.GROUND_TRUTH:
            return self._predict_ground_truth(mask)

        if (mode == FidelityMode.ADAPTIVE
                and self.accuracy_monitor.is_fallback_mode):
            result = self._predict_ground_truth(mask)
            if enable_adaptive:
                self._maybe_check_recovery(mask)
            return result

        result = super().predict(mask)

        if enable_adaptive and mode == FidelityMode.ADAPTIVE:
            self._track_mask_for_update(mask)
            self._maybe_trigger_update(mask)

        return result

    def _predict_ground_truth(
        self,
        mask: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """使用真实仿真预测"""
        was_single = isinstance(mask, np.ndarray) and mask.ndim == 2

        if isinstance(mask, list):
            masks = [np.asarray(m) for m in mask]
            results = [
                self._real_imaging_model.compute_aerial_image(m)
                for m in masks
            ]
            return np.stack(results)
        elif isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                return self._real_imaging_model.compute_aerial_image(mask)
            elif mask.ndim == 3:
                results = [
                    self._real_imaging_model.compute_aerial_image(m)
                    for m in mask
                ]
                return np.stack(results)
            elif mask.ndim == 4:
                results = [
                    self._real_imaging_model.compute_aerial_image(mask[i, 0])
                    for i in range(mask.shape[0])
                ]
                return np.stack(results)

        raise ValueError(f"不支持的 mask 格式: {type(mask)}, shape={getattr(mask, 'shape', 'N/A')}")

    def _track_mask_for_update(self, mask: Union[np.ndarray, List[np.ndarray]]):
        """跟踪用于更新的掩模样本"""
        if isinstance(mask, list):
            for m in mask:
                if m.ndim == 2:
                    self._recent_masks.append(m.copy())
        elif isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                self._recent_masks.append(mask.copy())
            elif mask.ndim >= 3:
                for i in range(mask.shape[0]):
                    self._recent_masks.append(mask[i].copy())

        max_recent = 2 * self.adaptive_config.accuracy_monitor.min_samples_for_check
        if len(self._recent_masks) > max_recent:
            self._recent_masks = self._recent_masks[-max_recent:]

    def _maybe_trigger_update(self, mask: Union[np.ndarray, List[np.ndarray]]):
        """检查是否需要触发更新流程"""
        self._inference_since_last_update += 1
        self._inference_since_last_check += 1

        update_cfg = self.adaptive_config.fine_tuning
        monitor_cfg = self.adaptive_config.accuracy_monitor

        need_accuracy_check = (
            monitor_cfg.enabled
            and self._inference_since_last_check >= monitor_cfg.check_interval
        )

        need_update = (
            update_cfg.enabled
            and self._inference_since_last_update >= update_cfg.update_interval
        )

        if need_accuracy_check or need_update:
            self._perform_adaptive_update(
                perform_accuracy_check=need_accuracy_check,
                perform_fine_tuning=need_update,
            )

    def _maybe_check_recovery(self, mask: Union[np.ndarray, List[np.ndarray]]):
        """检查是否可以从回退模式恢复"""
        if not self.accuracy_monitor.is_fallback_mode:
            return

        if isinstance(mask, np.ndarray) and mask.ndim == 2:
            self._recent_masks.append(mask.copy())

        if (len(self._recent_masks)
                >= self.adaptive_config.accuracy_monitor.min_samples_for_check):
            check_masks = np.stack(self._recent_masks[
                -self.adaptive_config.accuracy_monitor.min_samples_for_check:
            ])
            self.accuracy_monitor.check_recovery(self._surrogate_predict_only, check_masks)

    def _surrogate_predict_only(self, mask: np.ndarray) -> np.ndarray:
        """仅使用代理模型预测（用于恢复检查）"""
        return super().predict(mask)

    # ------------------------------------------------------------------
    # 自适应更新核心逻辑
    # ------------------------------------------------------------------

    def _perform_adaptive_update(
        self,
        perform_accuracy_check: bool = True,
        perform_fine_tuning: bool = True,
    ) -> AdaptiveUpdateResult:
        """
        执行自适应更新流程

        1. 精度检查（可选）
        2. 采集真实样本（如果需要更新）
        3. 添加到经验回放缓冲区
        4. 在线微调（可选）
        5. 模式切换判定
        """
        if self.adaptive_config.verbose:
            logger.info("开始执行自适应更新流程...")

        accuracy_result = None
        fine_tuning_result = None
        new_samples_added = 0
        mode_switched = False
        new_mode = None

        if perform_accuracy_check and len(self._recent_masks) >= self.adaptive_config.accuracy_monitor.min_samples_for_check:
            n = min(len(self._recent_masks), 32)
            check_indices = self.rng.choice(len(self._recent_masks), size=n, replace=False)
            check_masks = np.stack([self._recent_masks[i] for i in check_indices])

            accuracy_result = self.accuracy_monitor.check_accuracy(
                self._surrogate_predict_only, check_masks
            )

            if self.adaptive_config.verbose:
                logger.info(accuracy_result.summary())

            failures = accuracy_result.failures
            if failures:
                self._collect_ground_truth_samples(check_masks[failures])
                new_samples_added += len(failures)

            self._inference_since_last_check = 0

        if perform_fine_tuning:
            if len(self._recent_masks) >= self.adaptive_config.fine_tuning.min_samples_before_update:
                n = min(len(self._recent_masks), 32)
                sample_indices = self.rng.choice(
                    len(self._recent_masks), size=n, replace=False
                )
                sample_masks = np.stack([self._recent_masks[i] for i in sample_indices])
                self._collect_ground_truth_samples(sample_masks)
                new_samples_added += len(sample_masks)

            if len(self.replay_buffer) >= self.adaptive_config.fine_tuning.min_samples_before_update:
                new_masks = None
                new_aerials = None
                if self._pending_new_masks:
                    new_masks = np.stack(self._pending_new_masks)
                    new_aerials = np.stack(self._pending_new_aerials)

                fine_tuning_result = self.fine_tuner.fine_tune(
                    self.replay_buffer, new_masks, new_aerials
                )

                if self.adaptive_config.verbose:
                    logger.info(fine_tuning_result.summary())

                self._pending_new_masks.clear()
                self._pending_new_aerials.clear()

            self._inference_since_last_update = 0

        if self.accuracy_monitor.is_fallback_mode:
            if self._current_mode != FidelityMode.GROUND_TRUTH:
                old_mode = self._current_mode
                self._current_mode = FidelityMode.GROUND_TRUTH
                mode_switched = True
                new_mode = FidelityMode.GROUND_TRUTH
                logger.warning(
                    f"精度不达标，自动切换到全精度仿真模式: "
                    f"{old_mode.value} → {new_mode.value}"
                )
        elif self._current_mode == FidelityMode.GROUND_TRUTH:
            self._current_mode = self.adaptive_config.fidelity_mode
            mode_switched = True
            new_mode = self._current_mode
            logger.info(f"恢复到 {new_mode.value} 模式")

        result = AdaptiveUpdateResult(
            update_triggered=perform_fine_tuning,
            accuracy_checked=perform_accuracy_check,
            accuracy_result=accuracy_result,
            fine_tuning_result=fine_tuning_result,
            mode_switched=mode_switched,
            new_mode=new_mode,
            new_samples_added=new_samples_added,
        )

        self._last_update_result = result

        if self.adaptive_config.verbose and result.summary():
            logger.info(f"自适应更新完成: {result.summary()}")

        return result

    def _collect_ground_truth_samples(self, masks: np.ndarray):
        """采集真实仿真样本并添加到经验回放缓冲区"""
        for mask in masks:
            try:
                surrogate_aerial = self._surrogate_predict_only(mask)
                real_aerial = self._real_imaging_model.compute_aerial_image(mask)

                error = float(np.mean((surrogate_aerial - real_aerial) ** 2))
                priority = error + 1e-8

                self.replay_buffer.add(mask, real_aerial, priority=priority)

                self._pending_new_masks.append(mask)
                self._pending_new_aerials.append(real_aerial)

            except Exception as e:
                logger.warning(f"采集样本失败: {e}")

    def manual_trigger_update(
        self,
        masks: Optional[np.ndarray] = None,
        perform_accuracy_check: bool = True,
        perform_fine_tuning: bool = True,
    ) -> AdaptiveUpdateResult:
        """
        手动触发更新流程

        Args:
            masks: 额外的掩模样本用于更新
            perform_accuracy_check: 是否执行精度检查
            perform_fine_tuning: 是否执行微调

        Returns:
            AdaptiveUpdateResult
        """
        if masks is not None:
            if masks.ndim == 2:
                self._recent_masks.append(masks.copy())
            elif masks.ndim >= 3:
                for i in range(masks.shape[0]):
                    self._recent_masks.append(masks[i].copy())

        return self._perform_adaptive_update(
            perform_accuracy_check=perform_accuracy_check,
            perform_fine_tuning=perform_fine_tuning,
        )

    # ------------------------------------------------------------------
    # 接口兼容
    # ------------------------------------------------------------------

    def compute_aerial_image(self, mask: np.ndarray) -> np.ndarray:
        """兼容 PartialCoherentImaging 接口"""
        return self.predict(mask)

    def compute_image_gradient(self, mask: np.ndarray) -> np.ndarray:
        """计算梯度（始终使用代理模型）"""
        return super().compute_image_gradient(mask)

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------

    def reset_adaptive_state(self):
        """重置所有自适应状态"""
        self.replay_buffer.clear()
        self.fine_tuner.reset_optimizer()
        self.accuracy_monitor.reset()
        self._inference_since_last_update = 0
        self._inference_since_last_check = 0
        self._recent_masks.clear()
        self._pending_new_masks.clear()
        self._pending_new_aerials.clear()
        self._last_update_result = None
        self._current_mode = self.adaptive_config.fidelity_mode
        logger.info("自适应状态已重置")

    def get_adaptive_stats(self) -> Dict[str, Any]:
        """获取自适应系统统计信息"""
        return {
            'current_mode': self._current_mode.value,
            'is_fallback_mode': self.is_fallback_mode,
            'inference_since_last_update': self._inference_since_last_update,
            'inference_since_last_check': self._inference_since_last_check,
            'replay_buffer': self.replay_buffer.get_stats(),
            'fine_tuner': self.fine_tuner.get_stats(),
            'accuracy_monitor': self.accuracy_monitor.get_stats(),
            'recent_masks_count': len(self._recent_masks),
            'pending_samples_count': len(self._pending_new_masks),
            'last_update_result': (
                self._last_update_result.to_dict()
                if self._last_update_result else None
            ),
        }

    def save_state(self, filepath: str):
        """保存自适应状态到文件"""
        state = {
            'adaptive_config': self.adaptive_config.to_dict(),
            'current_mode': self._current_mode.value,
            'replay_buffer': {
                'buffer': list(self.replay_buffer.buffer),
                'priorities': list(self.replay_buffer.priorities),
            },
            'accuracy_monitor': self.accuracy_monitor.get_stats(),
            'fine_tuner': self.fine_tuner.get_stats(),
        }
        torch.save(state, filepath)
        logger.info(f"自适应状态已保存到 {filepath}")

    def load_state(self, filepath: str):
        """从文件加载自适应状态"""
        state = torch.load(filepath, map_location=self._device, weights_only=False)
        self.adaptive_config = AdaptiveSurrogateConfig.from_dict(
            state['adaptive_config']
        )
        self._current_mode = FidelityMode(state['current_mode'])

        if 'replay_buffer' in state:
            self.replay_buffer.buffer.clear()
            self.replay_buffer.priorities.clear()
            for (mask, aerial), prio in zip(
                state['replay_buffer']['buffer'],
                state['replay_buffer']['priorities']
            ):
                self.replay_buffer.buffer.append((mask, aerial))
                self.replay_buffer.priorities.append(prio)

        logger.info(f"自适应状态已从 {filepath} 加载")
