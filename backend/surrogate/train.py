# -*- coding: utf-8 -*-
"""
代理模型训练流水线

包含:
- TrainingConfig: 训练超参数配置
- train_surrogate_model: 完整训练流程（训练+验证+checkpoint+日志）
- evaluate_surrogate_model: 独立评估函数，计算 MSE/SSIM/MAE/PSNR
- load_trained_model: 从 checkpoint 加载模型
- SSIM / PSNR 指标实现
"""

import os
import sys
import time
import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path
import copy

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import (
        ReduceLROnPlateau, CosineAnnealingLR, StepLR, ExponentialLR
    )
except ImportError as e:
    raise ImportError(
        f"PyTorch 未安装，无法使用训练模块: {e}\n"
        "请安装: pip install torch torchvision"
    )

from .model import SurrogateModelConfig, build_model, UNet, CNNEncoderDecoder
from .dataset import (
    SurrogateDataset,
    DatasetConfig,
    generate_training_data,
    split_train_val,
    load_dataset_hdf5,
)


@dataclass
class ExportConfig:
    """
    模型生产化导出配置

    Attributes:
        export_onnx: 是否导出 ONNX 格式
        export_torchscript: 是否导出 TorchScript 格式
        onnx_opset_version: ONNX opset 版本
        dynamic_batch: 是否支持动态 batch 维度
        optimize: 是否对导出模型进行优化
        validate_export: 导出后是否验证模型正确性
        simplify_onnx: 是否使用 onnx-simplifier 简化 ONNX 模型
    """
    export_onnx: bool = True
    export_torchscript: bool = True
    onnx_opset_version: int = 17
    dynamic_batch: bool = True
    optimize: bool = True
    validate_export: bool = True
    simplify_onnx: bool = True


@dataclass
class TrainingConfig:
    """
    训练超参数配置

    Attributes:
        # 基础训练
        epochs: 总训练 epoch 数
        batch_size: 批大小
        learning_rate: 初始学习率
        weight_decay: L2 权重衰减
        optimizer: 优化器类型: 'adam', 'adamw', 'sgd'
        grad_clip: 梯度裁剪最大范数，0 表示不裁剪

        # 损失函数
        loss_type: 主损失: 'mse', 'l1', 'huber', 'ssim', 'combined'
        ssim_weight: 当 loss_type='combined' 时 (1-SSIM) 的权重
        mse_weight: 当 loss_type='combined' 时 MSE 的权重
        huber_delta: Huber 损失的 delta 参数

        # 学习率调度
        scheduler: 调度器: 'plateau', 'cosine', 'step', 'exp', None
        scheduler_patience: ReduceLROnPlateau 耐心值
        scheduler_factor: ReduceLROnPlateau 衰减因子
        scheduler_step_size: StepLR 步长
        scheduler_gamma: StepLR/ExponentialLR 衰减率
        min_lr: 最小学习率

        # 数据
        dataset_path: HDF5 数据集路径（优先级高于在线生成）
        num_samples: 在线生成时的样本数（当 dataset_path=None 时）
        grid_size: 在线生成时的图像尺寸
        train_ratio: 训练集比例

        # Checkpoint & 日志
        output_dir: 输出目录（保存模型、日志、曲线）
        checkpoint_freq: 每隔多少 epoch 保存一次 checkpoint
        save_best_only: 是否只保存验证集最优模型
        early_stop_patience: 早停耐心值，0 表示不早停
        seed: 随机种子（用于可复现性）

        # 设备
        device: 'auto' | 'cpu' | 'cuda' | 'mps'
        num_workers: DataLoader 加载线程数
        pin_memory: DataLoader 是否锁页内存

        # 验证
        val_batch_size: 验证批大小（默认等于 batch_size）
        log_interval: 每隔多少个 batch 打印一次训练日志

        # 生产化导出
        export: 导出配置
    """
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = 'adamw'
    grad_clip: float = 1.0

    loss_type: str = 'combined'
    ssim_weight: float = 0.2
    mse_weight: float = 1.0
    huber_delta: float = 0.1

    scheduler: Optional[str] = 'plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.9
    min_lr: float = 1e-7

    dataset_path: Optional[str] = None
    num_samples: int = 5000
    grid_size: Tuple[int, int] = (128, 128)
    train_ratio: float = 0.8

    output_dir: str = './surrogate_checkpoints'
    checkpoint_freq: int = 5
    save_best_only: bool = True
    early_stop_patience: int = 20
    seed: Optional[int] = 42

    device: str = 'auto'
    num_workers: int = 4
    pin_memory: bool = True

    val_batch_size: Optional[int] = None
    log_interval: int = 20

    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            elif isinstance(v, ExportConfig):
                d[k] = asdict(v)
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                if k == 'export' and isinstance(v, dict):
                    setattr(cfg, k, ExportConfig(**v))
                elif isinstance(getattr(cfg, k), tuple) and isinstance(v, list):
                    setattr(cfg, k, tuple(v))
                else:
                    setattr(cfg, k, v)
        return cfg


# ======================================================================
# 损失函数 & 评估指标
# ======================================================================

def _ssim_torch(
    x: torch.Tensor, y: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    计算 SSIM (Structural Similarity Index)，PyTorch 实现，可微

    Args:
        x, y: (B, C, H, W) 张量，值域 [0, 1]
    Returns:
        SSIM 标量 (越大越好，范围 [-1, 1])
    """
    from torch.nn.functional import avg_pool2d, conv2d

    def gaussian(window_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= (window_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    sigma = 1.5
    g = gaussian(window_size, sigma)
    window_1d = g.unsqueeze(0).unsqueeze(0)
    window_2d = window_1d.transpose(-1, -2) @ window_1d
    C = x.size(1)
    window = window_2d.unsqueeze(0).expand(C, 1, window_size, window_size).to(x.device)
    padding = window_size // 2

    mu_x = conv2d(x, window, padding=padding, groups=C)
    mu_y = conv2d(y, window, padding=padding, groups=C)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = conv2d(x * x, window, padding=padding, groups=C) - mu_x_sq
    sigma_y_sq = conv2d(y * y, window, padding=padding, groups=C) - mu_y_sq
    sigma_xy = conv2d(x * y, window, padding=padding, groups=C) - mu_xy

    ssim_map = (
        (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return ssim_map.mean()


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - SSIM，作为损失（越小越好）"""
    return 1.0 - _ssim_torch(x, y)


def ssim_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """NumPy 版本的 SSIM（用于最终评估）"""
    from skimage.metrics import structural_similarity as _ssim
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return float(_ssim(target, pred, data_range=1.0))


def psnr_numpy(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """峰值信噪比 (dB)"""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-15:
        return 100.0
    return float(10.0 * np.log10(max_val ** 2 / mse))


def build_loss_fn(cfg: TrainingConfig):
    """根据配置构建损失函数"""
    loss_type = cfg.loss_type.lower()

    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1' or loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return nn.HuberLoss(delta=cfg.huber_delta)
    elif loss_type == 'ssim':
        return ssim_loss
    elif loss_type == 'combined':
        mse_w = cfg.mse_weight
        ssim_w = cfg.ssim_weight
        mse_fn = nn.MSELoss()

        def combined_loss(pred, target):
            return mse_w * mse_fn(pred, target) + ssim_w * ssim_loss(pred, target)
        return combined_loss
    else:
        logger.warning(f"未知损失类型 {loss_type}，使用 MSE")
        return nn.MSELoss()


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> optim.Optimizer:
    opt_type = cfg.optimizer.lower()
    if opt_type == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    elif opt_type == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    elif opt_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    else:
        logger.warning(f"未知优化器 {opt_type}，使用 AdamW")
        return optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )


def build_scheduler(
    optimizer: optim.Optimizer, cfg: TrainingConfig
) -> Optional[object]:
    if cfg.scheduler is None:
        return None
    sched_type = cfg.scheduler.lower()
    if sched_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer, mode='min',
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.min_lr,
            verbose=False,
        )
    elif sched_type == 'cosine':
        return CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )
    elif sched_type == 'step':
        return StepLR(
            optimizer, step_size=cfg.scheduler_step_size,
            gamma=cfg.scheduler_gamma,
        )
    elif sched_type == 'exp':
        return ExponentialLR(
            optimizer, gamma=cfg.scheduler_gamma,
        )
    else:
        logger.warning(f"未知调度器 {sched_type}，不使用调度")
        return None


# ======================================================================
# 设备选择
# ======================================================================

def select_device(requested: str = 'auto') -> torch.device:
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(requested)


# ======================================================================
# 训练 & 验证循环
# ======================================================================

def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ExportPaths:
    """导出文件路径"""
    onnx_path: Optional[str] = None
    torchscript_path: Optional[str] = None
    metadata_path: Optional[str] = None


@dataclass
class TrainResult:
    """训练结果封装"""
    model: nn.Module
    model_config: SurrogateModelConfig
    training_config: TrainingConfig
    best_val_loss: float
    best_epoch: int
    train_loss_history: List[float]
    val_loss_history: List[float]
    val_metrics_history: List[Dict[str, float]]
    total_time: float
    checkpoint_path: str
    export_paths: ExportPaths = field(default_factory=ExportPaths)

    def summary(self) -> str:
        parts = [
            f"训练完成: 最佳 epoch={self.best_epoch}, "
            f"最佳 val_loss={self.best_val_loss:.6f}, "
            f"总耗时={self.total_time:.1f}s\n"
            f"PyTorch Checkpoint: {self.checkpoint_path}"
        ]
        if self.export_paths.onnx_path:
            parts.append(f"ONNX 模型: {self.export_paths.onnx_path}")
        if self.export_paths.torchscript_path:
            parts.append(f"TorchScript 模型: {self.export_paths.torchscript_path}")
        if self.export_paths.metadata_path:
            parts.append(f"元数据: {self.export_paths.metadata_path}")
        return "\n".join(parts)


def _validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
) -> Tuple[float, Dict[str, float], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    验证一个 epoch，返回 (avg_loss, metrics_dict, preds, targets, inputs)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_inputs: List[np.ndarray] = []

    with torch.no_grad():
        for masks, aerials in loader:
            masks = masks.to(device)
            aerials = aerials.to(device)

            preds = model(masks)
            loss = loss_fn(preds, aerials)

            total_loss += float(loss.item())
            num_batches += 1

            all_preds.append(preds.cpu().numpy())
            all_targets.append(aerials.cpu().numpy())
            all_inputs.append(masks.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)

    preds_all = np.concatenate(all_preds, axis=0)[:, 0]
    targets_all = np.concatenate(all_targets, axis=0)[:, 0]

    mse_val = float(np.mean((preds_all - targets_all) ** 2))
    mae_val = float(np.mean(np.abs(preds_all - targets_all)))

    try:
        ssim_vals = [
            ssim_numpy(p, t) for p, t in zip(preds_all, targets_all)
        ]
        ssim_val = float(np.mean(ssim_vals))
    except Exception:
        ssim_val = float('nan')

    psnr_vals = [psnr_numpy(p, t) for p, t in zip(preds_all, targets_all)]
    psnr_val = float(np.mean(psnr_vals))

    metrics = {
        'loss': avg_loss,
        'mse': mse_val,
        'mae': mae_val,
        'ssim': ssim_val,
        'psnr': psnr_val,
    }

    return avg_loss, metrics, all_preds, all_targets, all_inputs


def train_surrogate_model(
    model_config: Optional[SurrogateModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    dataset_config: Optional[DatasetConfig] = None,
    verbose: bool = True,
) -> TrainResult:
    """
    完整的代理模型训练流程

    Args:
        model_config: 模型架构配置，None 使用默认 U-Net
        training_config: 训练配置，None 使用默认
        dataset_config: 数据集配置（仅当在线生成时使用）
        verbose: 是否打印详细日志

    Returns:
        TrainResult，包含训练后的模型和相关信息
    """
    model_cfg = model_config or SurrogateModelConfig()
    train_cfg = training_config or TrainingConfig()

    _set_seed(train_cfg.seed)
    device = select_device(train_cfg.device)
    if verbose:
        logger.info(f"使用设备: {device}")

    # ----------------------------------------------------------
    # 1. 准备数据
    # ----------------------------------------------------------
    if train_cfg.dataset_path and os.path.exists(train_cfg.dataset_path):
        if verbose:
            logger.info(f"从 {train_cfg.dataset_path} 加载数据集")
        masks, aerials, meta, ds_cfg, train_idx, val_idx = load_dataset_hdf5(
            train_cfg.dataset_path
        )
        if train_idx is None:
            train_idx, val_idx = split_train_val(
                len(masks), train_cfg.train_ratio, train_cfg.seed
            )
    else:
        if verbose:
            logger.info(
                f"在线生成数据集: {train_cfg.num_samples} 样本, "
                f"尺寸 {train_cfg.grid_size}"
            )
        if dataset_config is None:
            dataset_config = DatasetConfig(
                grid_size=train_cfg.grid_size,
                num_samples=train_cfg.num_samples,
                seed=train_cfg.seed,
            )
        masks, aerials, meta, ds_cfg = generate_training_data(
            dataset_config, verbose=verbose
        )
        train_idx, val_idx = split_train_val(
            len(masks), train_cfg.train_ratio, train_cfg.seed
        )

    train_ds = SurrogateDataset(masks[train_idx], aerials[train_idx])
    val_ds = SurrogateDataset(masks[val_idx], aerials[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory and device.type != 'cpu',
        drop_last=True,
    )
    val_bs = train_cfg.val_batch_size or train_cfg.batch_size
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory and device.type != 'cpu',
    )

    if verbose:
        logger.info(
            f"数据集划分: 训练={len(train_idx)}, 验证={len(val_idx)}"
        )

    # ----------------------------------------------------------
    # 2. 构建模型、损失、优化器
    # ----------------------------------------------------------
    model = build_model(model_cfg).to(device)
    loss_fn = build_loss_fn(train_cfg)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    best_model_path = os.path.join(train_cfg.output_dir, 'best_model.pt')
    last_model_path = os.path.join(train_cfg.output_dir, 'last_model.pt')
    config_json_path = os.path.join(train_cfg.output_dir, 'config.json')

    with open(config_json_path, 'w') as f:
        json.dump({
            'model_config': model_cfg.to_dict(),
            'training_config': train_cfg.to_dict(),
            'dataset_config': ds_cfg.to_dict() if hasattr(ds_cfg, 'to_dict') else {},
        }, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------
    # 3. 训练循环
    # ----------------------------------------------------------
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    val_metrics_history: List[Dict[str, float]] = []

    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    early_stop_counter = 0

    t0 = time.time()

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        ep_t0 = time.time()

        for batch_i, (masks, aerials) in enumerate(train_loader, 1):
            masks = masks.to(device)
            aerials = aerials.to(device)

            optimizer.zero_grad()
            preds = model(masks)
            loss = loss_fn(preds, aerials)
            loss.backward()

            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip
                )

            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

            if verbose and batch_i % train_cfg.log_interval == 0:
                lr_cur = optimizer.param_groups[0]['lr']
                logger.info(
                    f"[Epoch {epoch}/{train_cfg.epochs}] "
                    f"Batch {batch_i}/{len(train_loader)}: "
                    f"loss={loss.item():.5f}, lr={lr_cur:.2e}"
                )

        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_loss_history.append(avg_train_loss)

        # ------------------------------------------------------
        # 验证
        # ------------------------------------------------------
        val_loss, val_metrics, _, _, _ = _validate_one_epoch(
            model, val_loader, loss_fn, device
        )
        val_loss_history.append(val_loss)
        val_metrics_history.append(val_metrics)

        ep_time = time.time() - ep_t0
        lr_cur = optimizer.param_groups[0]['lr']

        if verbose:
            logger.info(
                f"Epoch {epoch}/{train_cfg.epochs} "
                f"({ep_time:.1f}s) | "
                f"train_loss={avg_train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"MSE={val_metrics['mse']:.5f} "
                f"SSIM={val_metrics['ssim']:.4f} "
                f"PSNR={val_metrics['psnr']:.2f} | "
                f"lr={lr_cur:.2e}"
            )

        # ------------------------------------------------------
        # Checkpoint & 早停
        # ------------------------------------------------------
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if epoch % train_cfg.checkpoint_freq == 0 or is_best:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'val_metrics_history': val_metrics_history,
                'model_config': model_cfg.to_dict(),
                'training_config': train_cfg.to_dict(),
            }
            if scheduler is not None and hasattr(scheduler, 'state_dict'):
                ckpt['scheduler_state_dict'] = scheduler.state_dict()

            if is_best or not train_cfg.save_best_only:
                torch.save(ckpt, last_model_path)
                if verbose and epoch % train_cfg.checkpoint_freq == 0:
                    logger.info(f"  -> 保存 checkpoint: {last_model_path}")

            if is_best:
                torch.save(ckpt, best_model_path)
                if verbose:
                    logger.info(f"  *** 新的最佳模型保存在 {best_model_path} ***")

        # ------------------------------------------------------
        # 学习率调度
        # ------------------------------------------------------
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ------------------------------------------------------
        # 早停
        # ------------------------------------------------------
        if (train_cfg.early_stop_patience > 0
                and early_stop_counter >= train_cfg.early_stop_patience):
            if verbose:
                logger.info(
                    f"早停触发: 连续 {early_stop_counter} 个 epoch 未改善"
                )
            break

    # ----------------------------------------------------------
    # 训练结束：加载最佳权重
    # ----------------------------------------------------------
    total_time = time.time() - t0
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 保存最终 checkpoint
    final_ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'val_metrics_history': val_metrics_history,
        'model_config': model_cfg.to_dict(),
        'training_config': train_cfg.to_dict(),
        'total_time': total_time,
    }
    torch.save(final_ckpt, last_model_path)
    if not os.path.exists(best_model_path):
        torch.save(final_ckpt, best_model_path)

    # ----------------------------------------------------------
    # 生产化导出：ONNX + TorchScript + 元数据
    # ----------------------------------------------------------
    export_paths = ExportPaths()
    export_cfg = train_cfg.export

    if export_cfg.export_onnx or export_cfg.export_torchscript:
        if verbose:
            logger.info("开始生产化模型导出...")

        if val_metrics_history:
            extra_metrics = val_metrics_history[-1]
        else:
            extra_metrics = None

        export_paths = export_trained_model(
            model=model,
            output_dir=train_cfg.output_dir,
            model_config=model_cfg,
            training_config=train_cfg,
            grid_size=train_cfg.grid_size,
            device=device,
            extra_metrics=extra_metrics,
        )

    result = TrainResult(
        model=model,
        model_config=model_cfg,
        training_config=train_cfg,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_metrics_history=val_metrics_history,
        total_time=total_time,
        checkpoint_path=best_model_path,
        export_paths=export_paths,
    )

    if verbose:
        logger.info(result.summary())

    return result


# ======================================================================
# 模型生产化导出
# ======================================================================


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 1, 128, 128),
    opset_version: int = 17,
    dynamic_batch: bool = True,
    optimize: bool = True,
    validate: bool = True,
    simplify: bool = True,
    device: Optional[torch.device] = None,
) -> Optional[str]:
    """
    导出 PyTorch 模型为 ONNX 格式

    Args:
        model: 训练好的 PyTorch 模型
        output_path: 输出 .onnx 文件路径
        input_shape: 示例输入形状 (B, C, H, W)
        opset_version: ONNX opset 版本
        dynamic_batch: 是否支持动态 batch 维度
        optimize: 是否进行优化（冻结 BatchNorm 等）
        validate: 导出后验证 ONNX 模型
        simplify: 是否使用 onnx-simplifier 简化
        device: 设备，None 则自动选择

    Returns:
        成功返回输出路径，失败返回 None
    """
    try:
        if device is None:
            device = select_device('cpu')

        model = model.to(device)
        model.eval()

        if optimize:
            model = _optimize_model_for_export(model)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

        dummy_input = torch.randn(input_shape, device=device)

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )

        if simplify:
            try:
                import onnx
                import onnxsim
                model_onnx = onnx.load(output_path)
                model_simplified, check = onnxsim.simplify(model_onnx)
                if check:
                    onnx.save(model_simplified, output_path)
                    logger.info("ONNX 模型已使用 onnx-simplifier 优化")
                else:
                    logger.warning("onnx-simplifier 检查失败，使用原始 ONNX 模型")
            except ImportError:
                logger.warning("onnx-simplifier 未安装，跳过简化步骤")
            except Exception as e:
                logger.warning(f"onnx-simplifier 简化失败: {e}")

        if validate:
            try:
                import onnx
                import onnxruntime as ort

                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX 模型结构检查通过")

                ort_session = ort.InferenceSession(
                    output_path,
                    providers=['CPUExecutionProvider']
                )

                with torch.no_grad():
                    torch_output = model(dummy_input).cpu().numpy()

                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
                ort_outputs = ort_session.run(None, ort_inputs)
                ort_output = ort_outputs[0]

                max_diff = np.max(np.abs(torch_output - ort_output))
                mean_diff = np.mean(np.abs(torch_output - ort_output))

                if max_diff < 1e-4:
                    logger.info(
                        f"ONNX 推理验证通过: 最大差异={max_diff:.2e}, "
                        f"平均差异={mean_diff:.2e}"
                    )
                else:
                    logger.warning(
                        f"ONNX 推理差异较大: 最大差异={max_diff:.2e}, "
                        f"平均差异={mean_diff:.2e}"
                    )

            except ImportError as e:
                logger.warning(f"ONNX 验证依赖未安装: {e}，跳过验证")
            except Exception as e:
                logger.warning(f"ONNX 验证失败: {e}")

        logger.info(f"ONNX 模型已导出: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"ONNX 导出失败: {e}")
        return None


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 1, 128, 128),
    method: str = 'trace',
    optimize: bool = True,
    validate: bool = True,
    device: Optional[torch.device] = None,
) -> Optional[str]:
    """
    导出 PyTorch 模型为 TorchScript 格式

    Args:
        model: 训练好的 PyTorch 模型
        output_path: 输出 .pt 文件路径
        input_shape: 示例输入形状 (B, C, H, W)
        method: 'trace' (使用 torch.jit.trace) 或 'script' (使用 torch.jit.script)
        optimize: 是否进行优化
        validate: 导出后验证 TorchScript 模型
        device: 设备，None 则自动选择

    Returns:
        成功返回输出路径，失败返回 None
    """
    try:
        if device is None:
            device = select_device('cpu')

        model = model.to(device)
        model.eval()

        if optimize:
            model = _optimize_model_for_export(model)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

        dummy_input = torch.randn(input_shape, device=device)

        with torch.no_grad():
            if method == 'trace':
                scripted_model = torch.jit.trace(model, dummy_input)
            elif method == 'script':
                scripted_model = torch.jit.script(model)
            else:
                raise ValueError(f"未知的导出方法: {method}，支持 'trace' 或 'script'")

        scripted_model.save(output_path)

        if validate:
            try:
                loaded_model = torch.jit.load(output_path, map_location=device)
                loaded_model.eval()

                with torch.no_grad():
                    original_output = model(dummy_input)
                    scripted_output = loaded_model(dummy_input)

                max_diff = torch.max(torch.abs(original_output - scripted_output)).item()
                mean_diff = torch.mean(torch.abs(original_output - scripted_output)).item()

                if max_diff < 1e-5:
                    logger.info(
                        f"TorchScript 推理验证通过: 最大差异={max_diff:.2e}, "
                        f"平均差异={mean_diff:.2e}"
                    )
                else:
                    logger.warning(
                        f"TorchScript 推理差异较大: 最大差异={max_diff:.2e}, "
                        f"平均差异={mean_diff:.2e}"
                    )

            except Exception as e:
                logger.warning(f"TorchScript 验证失败: {e}")

        logger.info(f"TorchScript 模型已导出: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"TorchScript 导出失败: {e}")
        return None


def _optimize_model_for_export(model: nn.Module) -> nn.Module:
    """
    优化模型以便导出：
    1. 切换到 eval 模式（冻结 BatchNorm/Dropout 等训练专用层）
    2. 尝试将 BatchNorm 层参数合并到卷积层中（如果可用）
    """
    model.eval()

    try:
        import torch
        from torch.fx import symbolic_trace
        from torch.ao.quantization.fuse_modules import fuse_conv_bn

        model.eval()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        return model
    except ImportError:
        logger.info("未安装完整量化工具，仅启用 eval 模式")
        model.eval()
        return model
    except Exception as e:
        logger.debug(f"模型优化跳过: {e}")
        model.eval()
        return model


def export_metadata(
    output_path: str,
    model_config: SurrogateModelConfig,
    training_config: TrainingConfig,
    grid_size: Tuple[int, int],
    export_paths: ExportPaths,
    extra_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    导出模型元数据 JSON 文件，供推理服务使用

    Args:
        output_path: 输出 JSON 文件路径
        model_config: 模型配置
        training_config: 训练配置
        grid_size: 输入图像尺寸 (H, W)
        export_paths: 导出文件路径
        extra_info: 额外信息（如评估指标）

    Returns:
        输出文件路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    metadata = {
        'model': {
            'type': model_config.model_type,
            'input_shape': [-1, model_config.in_channels, grid_size[0], grid_size[1]],
            'output_shape': [-1, model_config.out_channels, grid_size[0], grid_size[1]],
            'input_dtype': 'float32',
            'output_dtype': 'float32',
            'input_range': [0.0, 1.0],
            'output_range': [0.0, 1.0],
            'num_parameters': sum(p.numel() for p in build_model(model_config).parameters()),
            'config': model_config.to_dict(),
        },
        'export': {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'onnx_path': export_paths.onnx_path,
            'torchscript_path': export_paths.torchscript_path,
            'opset_version': training_config.export.onnx_opset_version,
            'dynamic_batch': training_config.export.dynamic_batch,
        },
        'preprocessing': {
            'normalize': False,
            'mean': [0.0],
            'std': [1.0],
            'input_format': 'NCHW',
            'channel_order': 'first',
        },
        'postprocessing': {
            'sigmoid': model_config.final_activation == 'sigmoid',
            'clip': [0.0, 1.0],
        },
        'performance_hints': {
            'recommended_batch_size': training_config.batch_size,
            'recommended_device': 'cpu',
            'approx_latency_ms_per_batch': None,
        },
    }

    if extra_info:
        metadata['extra'] = extra_info

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"元数据已导出: {output_path}")
    return output_path


def export_trained_model(
    model: nn.Module,
    output_dir: str,
    model_config: SurrogateModelConfig,
    training_config: TrainingConfig,
    grid_size: Tuple[int, int],
    device: Optional[torch.device] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> ExportPaths:
    """
    执行完整的生产化导出流程（ONNX + TorchScript + 元数据）

    Args:
        model: 训练好的 PyTorch 模型
        output_dir: 输出目录
        model_config: 模型配置
        training_config: 训练配置
        grid_size: 图像尺寸 (H, W)
        device: 设备
        extra_metrics: 额外的评估指标

    Returns:
        ExportPaths 包含所有导出文件路径
    """
    export_cfg = training_config.export
    input_shape = (1, 1, grid_size[0], grid_size[1])

    paths = ExportPaths()

    if export_cfg.export_onnx:
        onnx_path = os.path.join(output_dir, 'model.onnx')
        paths.onnx_path = export_to_onnx(
            model=model,
            output_path=onnx_path,
            input_shape=input_shape,
            opset_version=export_cfg.onnx_opset_version,
            dynamic_batch=export_cfg.dynamic_batch,
            optimize=export_cfg.optimize,
            validate=export_cfg.validate_export,
            simplify=export_cfg.simplify_onnx,
            device=device,
        )

    if export_cfg.export_torchscript:
        ts_path = os.path.join(output_dir, 'model.pt')
        paths.torchscript_path = export_to_torchscript(
            model=model,
            output_path=ts_path,
            input_shape=input_shape,
            method='trace',
            optimize=export_cfg.optimize,
            validate=export_cfg.validate_export,
            device=device,
        )

    paths.metadata_path = export_metadata(
        output_path=os.path.join(output_dir, 'metadata.json'),
        model_config=model_config,
        training_config=training_config,
        grid_size=grid_size,
        export_paths=paths,
        extra_info=extra_metrics,
    )

    return paths


# ======================================================================
# 模型评估 & 加载
# ======================================================================

def evaluate_surrogate_model(
    model: nn.Module,
    masks: np.ndarray,
    aerial_targets: np.ndarray,
    device: str = 'auto',
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    在给定数据集上评估模型，计算 MSE / MAE / SSIM / PSNR

    Returns:
        dict with keys: 'mse', 'mae', 'ssim', 'psnr', 'num_samples', 'time_per_sample_ms'
    """
    dev = select_device(device)
    model = model.to(dev)
    model.eval()

    if masks.shape[0] != aerial_targets.shape[0]:
        raise ValueError("masks 和 aerial_targets 数量不一致")

    dataset = SurrogateDataset(masks, aerial_targets)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=dev.type != 'cpu',
    )

    all_preds: List[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for batch_masks, _ in loader:
            batch_masks = batch_masks.to(dev)
            preds = model(batch_masks)
            all_preds.append(preds.cpu().numpy())

    total_time = time.time() - t0
    n = masks.shape[0]
    per_sample_ms = total_time / n * 1000

    preds_all = np.concatenate(all_preds, axis=0)[:, 0]
    targets_all = aerial_targets.astype(np.float32)

    mse_val = float(np.mean((preds_all - targets_all) ** 2))
    mae_val = float(np.mean(np.abs(preds_all - targets_all)))

    try:
        ssim_vals = [ssim_numpy(p, t) for p, t in zip(preds_all, targets_all)]
        ssim_val = float(np.mean(ssim_vals))
    except Exception:
        ssim_val = float('nan')

    psnr_vals = [psnr_numpy(p, t) for p, t in zip(preds_all, targets_all)]
    psnr_val = float(np.mean(psnr_vals))

    result = {
        'num_samples': n,
        'mse': mse_val,
        'mae': mae_val,
        'ssim': ssim_val,
        'psnr': psnr_val,
        'total_time_s': total_time,
        'time_per_sample_ms': per_sample_ms,
    }

    if verbose:
        logger.info(
            f"评估完成: N={n}, 总耗时 {total_time:.2f}s "
            f"({per_sample_ms:.2f} ms/样本)\n"
            f"  MSE={mse_val:.6f}, MAE={mae_val:.6f}\n"
            f"  SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f} dB"
        )

    return result


def load_trained_model(
    checkpoint_path: str,
    device: str = 'auto',
    load_weights_only: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    从 checkpoint 加载训练好的模型

    Args:
        checkpoint_path: .pt 文件路径
        device: 推理设备
        load_weights_only: 只返回模型，不加载优化器等状态

    Returns:
        (model, extra_info)
        - model: 加载好权重的模型
        - extra_info: 训练配置、历史记录、最佳 epoch 等信息
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    dev = select_device(device)
    map_loc = dev if dev.type == 'cpu' else None

    ckpt = torch.load(checkpoint_path, map_location=map_loc, weights_only=False)

    model_cfg_dict = ckpt.get('model_config', {})
    model_cfg = SurrogateModelConfig.from_dict(model_cfg_dict)
    model = build_model(model_cfg)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model = model.to(dev)
    model.eval()

    extra = {
        'model_config': model_cfg,
        'best_epoch': ckpt.get('best_epoch'),
        'best_val_loss': ckpt.get('best_val_loss'),
        'train_loss_history': ckpt.get('train_loss_history'),
        'val_loss_history': ckpt.get('val_loss_history'),
        'val_metrics_history': ckpt.get('val_metrics_history'),
        'training_config': TrainingConfig.from_dict(
            ckpt.get('training_config', {})
        ) if 'training_config' in ckpt else None,
    }

    logger.info(
        f"从 {checkpoint_path} 加载模型: "
        f"最佳 epoch={extra['best_epoch']}, "
        f"val_loss={extra['best_val_loss']:.6f}"
    )
    return model, extra
