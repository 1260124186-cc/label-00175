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
