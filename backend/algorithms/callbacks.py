# -*- coding: utf-8 -*-
"""
回调模块：训练过程中的各种回调

提供统一的回调接口，支持学习率调度、早停、checkpoint、
中间掩模保存、收敛曲线绘制等功能。
同时提供工作流级别的 Checkpoint 管理，支持多阶段流程（SMO/OPC/ILT等）
的断点续跑，持久化光源图、掩模状态、迭代计数与随机种子。
"""

import os
import time
import json
import pickle
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from io import BytesIO
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """
    训练器状态，在回调之间共享

    Attributes:
        epoch: 当前 epoch/迭代次数
        loss: 当前损失值
        learning_rate: 当前学习率
        mask: 当前掩模
        best_loss: 历史最优损失
        best_mask: 历史最优掩模
        loss_history: 损失历史
        lr_history: 学习率历史
        mask_history: 中间掩模历史（可选，用于批量评估）
        stop_training: 是否停止训练
        logs: 其他日志信息
    """
    epoch: int = 0
    loss: float = float('inf')
    learning_rate: float = 0.0
    mask: Optional[np.ndarray] = None
    best_loss: float = float('inf')
    best_mask: Optional[np.ndarray] = None
    loss_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    mask_history: List[np.ndarray] = field(default_factory=list)
    stop_training: bool = False
    logs: Dict[str, Any] = field(default_factory=dict)


class Callback(ABC):
    """
    回调基类

    所有回调都应该继承此类，并重写需要的钩子方法。
    """

    def __init__(self):
        self.state: Optional[TrainerState] = None
        self.params: Dict[str, Any] = {}

    def set_params(self, params: Dict[str, Any]):
        """设置训练参数"""
        self.params = params

    def set_state(self, state: TrainerState):
        """设置训练状态引用"""
        self.state = state

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始时调用"""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束时调用"""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 epoch 开始时调用"""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 epoch 结束时调用"""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 batch 开始时调用"""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 batch 结束时调用"""
        pass


class CallbackList:
    """
    回调列表管理器

    按顺序执行多个回调。
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks: List[Callback] = callbacks or []
        self.state = TrainerState()
        self.params: Dict[str, Any] = {}

    def append(self, callback: Callback):
        """添加回调"""
        self.callbacks.append(callback)
        if self.state is not None:
            callback.set_state(self.state)
        if self.params:
            callback.set_params(self.params)

    def set_params(self, params: Dict[str, Any]):
        """设置参数到所有回调"""
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_state(self, state: TrainerState):
        """设置状态到所有回调"""
        self.state = state
        for callback in self.callbacks:
            callback.set_state(state)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始"""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束"""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """epoch 开始"""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """
        epoch 结束

        Returns:
            是否应该停止训练
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
        return self.state.stop_training


class LambdaCallback(Callback):
    """
    Lambda 回调：使用简单函数作为回调

    用于快速创建简单的回调，无需定义类。
    """

    def __init__(self,
                 on_train_begin: Optional[Callable] = None,
                 on_train_end: Optional[Callable] = None,
                 on_epoch_begin: Optional[Callable] = None,
                 on_epoch_end: Optional[Callable] = None):
        super().__init__()
        self._on_train_begin = on_train_begin
        self._on_train_end = on_train_end
        self._on_epoch_begin = on_epoch_begin
        self._on_epoch_end = on_epoch_end

    def on_train_begin(self, logs=None):
        if self._on_train_begin:
            self._on_train_begin(logs)

    def on_train_end(self, logs=None):
        if self._on_train_end:
            self._on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        if self._on_epoch_begin:
            self._on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if self._on_epoch_end:
            self._on_epoch_end(epoch, logs)


class LearningRateSchedulerCallback(Callback):
    """
    学习率调度器回调

    支持多种学习率调度策略：step、exponential、cosine、reduce_on_plateau
    """

    def __init__(self,
                 initial_lr: float,
                 scheduler_type: str = 'step',
                 decay: float = 0.95,
                 step_size: int = 20,
                 min_lr: float = 1e-6,
                 patience: int = 10,
                 factor: float = 0.5,
                 min_delta: float = 1e-6):
        """
        初始化学习率调度器

        Args:
            initial_lr: 初始学习率
            scheduler_type: 调度器类型: 'step', 'exponential', 'cosine', 'reduce_on_plateau'
            decay: 衰减率
            step_size: 步长（step/cosine 调度使用）
            min_lr: 最小学习率
            patience: ReduceLROnPlateau 的耐心值
            factor: ReduceLROnPlateau 的衰减因子
            min_delta: ReduceLROnPlateau 的最小改善量
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.scheduler_type = scheduler_type.lower()
        self.decay = decay
        self.step_size = step_size
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.min_delta = min_delta
        self.current_lr = initial_lr

        self._plateau_counter = 0
        self._best_loss = float('inf')

    def on_train_begin(self, logs=None):
        self.current_lr = self.initial_lr
        self._plateau_counter = 0
        self._best_loss = float('inf')
        if self.state is not None:
            self.state.learning_rate = self.current_lr

    def on_epoch_begin(self, epoch, logs=None):
        if self.scheduler_type == 'step':
            self.current_lr = self.initial_lr * (self.decay ** (epoch // self.step_size))
        elif self.scheduler_type == 'exponential':
            self.current_lr = self.initial_lr * (self.decay ** epoch)
        elif self.scheduler_type == 'cosine':
            self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                             (1 + np.cos(np.pi * epoch / self.step_size))

        self.current_lr = max(self.current_lr, self.min_lr)

        if self.state is not None:
            self.state.learning_rate = self.current_lr
            if 'learning_rate' not in self.state.logs:
                self.state.logs['learning_rate'] = self.current_lr

    def on_epoch_end(self, epoch, logs=None):
        if self.scheduler_type == 'reduce_on_plateau':
            current_loss = logs.get('loss', float('inf')) if logs else float('inf')

            if current_loss < self._best_loss - self.min_delta:
                self._best_loss = current_loss
                self._plateau_counter = 0
            else:
                self._plateau_counter += 1
                if self._plateau_counter >= self.patience:
                    self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                    self._plateau_counter = 0
                    logger.info(f"学习率衰减: {self.current_lr / self.factor:.6e} -> {self.current_lr:.6e}")

            if self.state is not None:
                self.state.learning_rate = self.current_lr
                self.state.logs['learning_rate'] = self.current_lr


class EarlyStoppingCallback(Callback):
    """
    早停回调

    当验证损失在连续 patience 个 epoch 内没有改善时，停止训练。
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-6,
                 monitor: str = 'loss',
                 mode: str = 'min',
                 restore_best: bool = True):
        """
        初始化早停

        Args:
            patience: 耐心值（连续多少次无改善则停止）
            min_delta: 最小改善量
            monitor: 监控的指标名称
            mode: 'min' 或 'max'，指标越小越好还是越大越好
            restore_best: 是否恢复到最优模型
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best = restore_best

        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.best_mask = None

    def on_train_begin(self, logs=None):
        self.counter = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.best_mask = None

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor, float('inf')) if logs else float('inf')

        improved = False
        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                improved = True
        else:
            if current_value > self.best_value + self.min_delta:
                improved = True

        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            if self.state is not None and self.state.mask is not None:
                self.best_mask = self.state.mask.copy()
                self.state.best_loss = current_value
                self.state.best_mask = self.best_mask.copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.state is not None:
                    self.state.stop_training = True
                    if self.restore_best and self.best_mask is not None:
                        self.state.mask = self.best_mask.copy()
                        self.state.loss = self.best_value
                    logger.info(
                        f"早停触发: 在 epoch {epoch} 停止，"
                        f"最优值 {self.best_value:.6e} (epoch {self.best_epoch})"
                    )


class ModelCheckpointCallback(Callback):
    """
    模型 checkpoint 回调

    定期保存训练状态（checkpoint），支持断点续训。
    """

    def __init__(self,
                 checkpoint_dir: str = './checkpoints',
                 save_freq: int = 10,
                 save_best_only: bool = False,
                 monitor: str = 'loss',
                 mode: str = 'min',
                 max_checkpoints: int = 5,
                 prefix: str = 'checkpoint'):
        """
        初始化 checkpoint 回调

        Args:
            checkpoint_dir: checkpoint 保存目录
            save_freq: 保存频率（每多少个 epoch 保存一次）
            save_best_only: 是否只保存最优的
            monitor: 监控指标
            mode: 'min' 或 'max'
            max_checkpoints: 最多保留的 checkpoint 数量
            prefix: 文件名前缀
        """
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.max_checkpoints = max_checkpoints
        self.prefix = prefix

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.saved_files: List[Path] = []

    def on_train_begin(self, logs=None):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.saved_files = []

    def on_epoch_end(self, epoch, logs=None):
        should_save = False
        is_best = False

        if self.save_best_only:
            current_value = logs.get(self.monitor, float('inf')) if logs else float('inf')
            if self.mode == 'min':
                if current_value < self.best_value:
                    self.best_value = current_value
                    is_best = True
                    should_save = True
            else:
                if current_value > self.best_value:
                    self.best_value = current_value
                    is_best = True
                    should_save = True
        else:
            if epoch % self.save_freq == 0:
                should_save = True

        if should_save and self.state is not None:
            filename = f"{self.prefix}_epoch_{epoch:04d}"
            if is_best:
                filename += "_best"
            filepath = self.checkpoint_dir / f"{filename}.npz"

            self._save_checkpoint(filepath, epoch)
            logger.info(f"保存 checkpoint: {filepath}")

            self.saved_files.append(filepath)

            if self.max_checkpoints > 0 and len(self.saved_files) > self.max_checkpoints:
                old_file = self.saved_files.pop(0)
                if old_file.exists() and "_best" not in old_file.name:
                    old_file.unlink()
                    logger.debug(f"删除旧 checkpoint: {old_file}")

    def _save_checkpoint(self, filepath: Path, epoch: int):
        """保存 checkpoint"""
        state = self.state
        save_dict = {
            'epoch': epoch,
            'loss': state.loss,
            'learning_rate': state.learning_rate,
            'best_loss': state.best_loss,
            'loss_history': np.array(state.loss_history),
            'lr_history': np.array(state.lr_history),
        }

        if state.mask is not None:
            save_dict['mask'] = state.mask

        if state.best_mask is not None:
            save_dict['best_mask'] = state.best_mask

        for key, value in state.logs.items():
            if isinstance(value, (int, float, np.ndarray)):
                save_dict[f'log_{key}'] = value

        np.savez(filepath, **save_dict)

    @classmethod
    def load_checkpoint(cls, filepath: str) -> Dict[str, Any]:
        """
        从 checkpoint 文件加载状态

        Args:
            filepath: checkpoint 文件路径

        Returns:
            状态字典
        """
        data = np.load(filepath, allow_pickle=True)
        result: Dict[str, Any] = {}

        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            if key.startswith('log_'):
                result.setdefault('logs', {})[key[4:]] = value
            else:
                result[key] = value

        if 'loss_history' in result:
            result['loss_history'] = list(result['loss_history'])
        if 'lr_history' in result:
            result['lr_history'] = list(result['lr_history'])

        return result


class MaskSnapshotCallback(Callback):
    """
    中间掩模快照回调

    每隔 N 步保存当前的中间掩模图像，用于观察优化过程。
    """

    def __init__(self,
                 save_dir: str = './snapshots',
                 save_freq: int = 10,
                 save_best: bool = True,
                 image_format: str = 'png',
                 save_npy: bool = True):
        """
        初始化掩模快照回调

        Args:
            save_dir: 保存目录
            save_freq: 保存频率（每多少 epoch 保存一次）
            save_best: 是否保存最优掩模
            image_format: 图像格式 (png, npy)
            save_npy: 是否同时保存 numpy 格式
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.save_best = save_best
        self.image_format = image_format
        self.save_npy = save_npy

        self.best_loss = float('inf')

    def on_train_begin(self, logs=None):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if self.state is None or self.state.mask is None:
            return

        current_loss = logs.get('loss', float('inf')) if logs else float('inf')

        save_regular = (epoch % self.save_freq == 0)

        is_best = False
        if self.save_best and current_loss < self.best_loss:
            self.best_loss = current_loss
            is_best = True

        if save_regular or is_best:
            if save_regular:
                self._save_mask(self.state.mask, f"mask_epoch_{epoch:04d}")
            if is_best:
                self._save_mask(self.state.mask, "mask_best")

    def _save_mask(self, mask: np.ndarray, filename: str):
        """保存掩模"""
        try:
            from utils.visualization import plot_mask
            import matplotlib
            matplotlib.use('Agg')

            img_path = self.save_dir / f"{filename}.{self.image_format}"
            plot_mask(
                mask,
                title=f"Mask - {filename}",
                save_path=str(img_path),
                show=False,
                figsize=(8, 8)
            )
        except Exception as e:
            logger.debug(f"保存掩模图像失败: {e}")

        if self.save_npy:
            npy_path = self.save_dir / f"{filename}.npy"
            np.save(npy_path, mask)


class ConvergencePlotCallback(Callback):
    """
    收敛曲线绘制回调

    实时绘制损失和学习率曲线，保存为图片。
    """

    def __init__(self,
                 save_dir: str = './plots',
                 plot_freq: int = 10,
                 log_scale: bool = True,
                 plot_lr: bool = True,
                 live_update: bool = False):
        """
        初始化收敛曲线回调

        Args:
            save_dir: 保存目录
            plot_freq: 绘图频率（每多少 epoch 更新一次）
            log_scale: 是否使用对数坐标
            plot_lr: 是否同时绘制学习率曲线
            live_update: 是否实时更新（用于有显示环境的情况）
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.plot_freq = plot_freq
        self.log_scale = log_scale
        self.plot_lr = plot_lr
        self.live_update = live_update

    def on_train_begin(self, logs=None):
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.state is None:
            return
        if epoch % self.plot_freq != 0 and epoch > 0:
            return

        try:
            import matplotlib
            if not self.live_update:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            n_plots = 2 if self.plot_lr else 1
            fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
            if n_plots == 1:
                axes = [axes]

            loss_history = self.state.loss_history
            epochs = list(range(len(loss_history)))

            axes[0].plot(epochs, loss_history, 'b-', linewidth=1.5)
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)
            if self.log_scale and len(loss_history) > 0 and loss_history[-1] > 0:
                axes[0].set_yscale('log')

            if loss_history:
                axes[0].annotate(
                    f'final: {loss_history[-1]:.4e}',
                    xy=(len(loss_history) - 1, loss_history[-1]),
                    xytext=(-10, 10), textcoords='offset points',
                    fontsize=9, color='red'
                )

            if self.plot_lr and len(self.state.lr_history) > 0:
                lr_history = self.state.lr_history
                axes[1].plot(epochs[:len(lr_history)], lr_history, 'g-', linewidth=1.5)
                axes[1].set_title('Learning Rate')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('LR')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_yscale('log')

            plt.tight_layout()
            save_path = self.save_dir / 'convergence_curve.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

            if self.live_update:
                plt.pause(0.01)

            plt.close(fig)

        except Exception as e:
            logger.debug(f"绘制收敛曲线失败: {e}")


class LoggerCallback(Callback):
    """
    日志回调

    定期输出训练日志信息。
    """

    def __init__(self,
                 log_freq: int = 10,
                 show_lr: bool = True,
                 show_time: bool = True):
        """
        初始化日志回调

        Args:
            log_freq: 日志输出频率
            show_lr: 是否显示学习率
            show_time: 是否显示耗时
        """
        super().__init__()
        self.log_freq = log_freq
        self.show_lr = show_lr
        self.show_time = show_time
        self.start_time = 0.0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        logger.info("=" * 60)
        logger.info("训练开始")
        logger.info("=" * 60)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_freq != 0:
            return

        msg = f"Epoch {epoch:4d}"

        if logs:
            for key in ['loss', 'mse', 'ssim']:
                if key in logs:
                    msg += f" | {key}: {logs[key]:.6e}"

        if self.show_lr and self.state is not None:
            msg += f" | lr: {self.state.learning_rate:.4e}"

        if self.show_time:
            elapsed = time.time() - self.start_time
            msg += f" | {elapsed:.1f}s"

        logger.info(msg)

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info(f"训练结束，总耗时: {total_time:.2f} 秒")
        if logs and 'loss' in logs:
            logger.info(f"最终损失: {logs['loss']:.6e}")
        logger.info("=" * 60)


class HistoryCallback(Callback):
    """
    历史记录回调

    记录所有训练历史数据。
    """

    def __init__(self, save_masks: bool = False):
        """
        初始化历史记录回调

        Args:
            save_masks: 是否保存每一步的中间掩模（内存开销较大，按需开启）
        """
        super().__init__()
        self.save_masks = save_masks
        self.epoch_history: List[int] = []
        self.loss_history: List[float] = []
        self.lr_history: List[float] = []
        self.mask_history: List[np.ndarray] = []
        self.logs_history: List[Dict[str, Any]] = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_history.append(epoch)
        if logs and 'loss' in logs:
            self.loss_history.append(logs['loss'])
        elif self.state is not None:
            self.loss_history.append(self.state.loss)

        if self.state is not None:
            self.lr_history.append(self.state.learning_rate)
            if self.save_masks and self.state.mask is not None:
                self.mask_history.append(self.state.mask.copy())
                self.state.mask_history.append(self.state.mask.copy())

        if logs:
            self.logs_history.append(dict(logs))

    def get_history(self) -> Dict[str, Any]:
        """获取历史记录"""
        result = {
            'epochs': self.epoch_history,
            'loss': self.loss_history,
            'learning_rate': self.lr_history,
            'logs': self.logs_history,
        }
        if self.save_masks:
            result['masks'] = self.mask_history
        return result


class AnimationCallback(Callback):
    """
    优化过程动画生成回调

    每 N 步记录当前掩模、空间像、(可选)wafer图像、误差图，训练结束后生成
    GIF 或 MP4 动画文件，直观展示优化演进过程。支持附带收敛曲线子图。
    """

    def __init__(self,
                 save_dir: str = './animations',
                 save_freq: int = 1,
                 output_format: str = 'gif',
                 fps: int = 10,
                 dpi: int = 100,
                 figsize: Optional[Tuple[int, int]] = None,
                 compute_aerial_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 compute_wafer_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 target_image: Optional[np.ndarray] = None,
                 filename_prefix: str = 'optimization_animation',
                 show_title_info: bool = True,
                 show_convergence: bool = True,
                 consistent_error_scale: bool = True,
                 cmap_mask: str = 'gray',
                 cmap_aerial: str = 'gray',
                 cmap_wafer: str = 'gray',
                 cmap_error: str = 'hot'):
        """
        初始化动画回调

        Args:
            save_dir: 动画文件保存目录
            save_freq: 保存频率（每多少个 epoch 记录一帧）
            output_format: 输出格式: 'gif' 或 'mp4'
            fps: 动画帧率
            dpi: 输出分辨率
            figsize: 图像尺寸 (width, height)，为 None 时根据子图数量自动推算
            compute_aerial_fn: 计算空间像的函数: fn(mask) -> aerial_image
            compute_wafer_fn: 计算wafer图像的函数: fn(mask) -> wafer_image (可选)
            target_image: 目标图像，用于计算误差图
            filename_prefix: 输出文件名前缀
            show_title_info: 是否在标题中显示 epoch/loss 等信息
            show_convergence: 是否附带显示损失收敛曲线子图
            consistent_error_scale: 是否在所有帧中使用一致的误差色标范围（便于对比）
            cmap_mask: 掩模图的颜色映射
            cmap_aerial: 空间像的颜色映射
            cmap_wafer: wafer图像的颜色映射
            cmap_error: 误差图的颜色映射
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_freq = max(1, int(save_freq))
        self.output_format = output_format.lower()
        self.fps = fps
        self.dpi = dpi
        self.figsize = figsize
        self.compute_aerial_fn = compute_aerial_fn
        self.compute_wafer_fn = compute_wafer_fn
        self.target_image = target_image
        self.filename_prefix = filename_prefix
        self.show_title_info = show_title_info
        self.show_convergence = show_convergence
        self.consistent_error_scale = consistent_error_scale
        self.cmap_mask = cmap_mask
        self.cmap_aerial = cmap_aerial
        self.cmap_wafer = cmap_wafer
        self.cmap_error = cmap_error

        self._frames: List[Dict[str, Any]] = []
        self._matplotlib_imported = False
        self._plt = None
        self._animation = None
        self._import_matplotlib()

        self.output_path: Optional[Path] = None
        self._error_vmax: Optional[float] = None
        self._n_image_cols: int = 3  # 默认 mask / aerial / error

    def _import_matplotlib(self):
        """延迟导入 matplotlib，避免无显示环境报错"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            self._plt = plt
            self._animation = animation
            self._matplotlib_imported = True
        except Exception as e:
            logger.warning(f"matplotlib 导入失败，动画回调将不生效: {e}")
            self._matplotlib_imported = False

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始，初始化保存目录和帧缓存"""
        self._frames.clear()
        self.output_path = None
        self._error_vmax = None
        if self._matplotlib_imported:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 epoch 结束，按需记录一帧"""
        if not self._matplotlib_imported:
            return
        if self.state is None or self.state.mask is None:
            return
        if epoch % self.save_freq != 0 and epoch > 0:
            return

        try:
            mask = self.state.mask.copy()
            loss = float(logs.get('loss', self.state.loss)) if logs else float(self.state.loss)
            lr = float(self.state.learning_rate)

            aerial = None
            wafer = None
            error = None
            if self.compute_aerial_fn is not None:
                try:
                    aerial = self.compute_aerial_fn(mask)
                except Exception as e:
                    logger.debug(f"计算空间像失败: {e}")
                    aerial = None

            if self.compute_wafer_fn is not None:
                try:
                    wafer = self.compute_wafer_fn(mask)
                except Exception as e:
                    logger.debug(f"计算wafer图像失败: {e}")
                    wafer = None

            ref_image = None
            if wafer is not None:
                ref_image = wafer
            elif aerial is not None:
                ref_image = aerial

            if ref_image is not None and self.target_image is not None:
                try:
                    if ref_image.shape == self.target_image.shape:
                        error = np.abs(ref_image - self.target_image)
                except Exception as e:
                    logger.debug(f"计算误差图失败: {e}")
                    error = None

            self._frames.append({
                'epoch': epoch,
                'mask': mask,
                'aerial': aerial,
                'wafer': wafer,
                'error': error,
                'loss': loss,
                'learning_rate': lr,
            })
        except Exception as e:
            logger.debug(f"记录动画帧失败 (epoch {epoch}): {e}")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束，生成并保存动画文件"""
        if not self._matplotlib_imported or len(self._frames) == 0:
            if len(self._frames) == 0:
                logger.debug("没有可用的动画帧，跳过动画生成")
            return

        if self.consistent_error_scale:
            errors = [f['error'] for f in self._frames if f['error'] is not None]
            if errors:
                self._error_vmax = float(max(np.max(e) for e in errors))
                if self._error_vmax <= 0:
                    self._error_vmax = 1.0

        self._n_image_cols = 3
        if self._frames[0].get('wafer') is not None:
            self._n_image_cols = 4

        try:
            self._generate_animation()
        except Exception as e:
            logger.warning(f"生成动画失败: {e}")

    def _figure_layout(self) -> Tuple[Any, List[Any], Optional[Any]]:
        """
        创建 Figure 与子图布局

        Returns:
            (fig, image_axes, convergence_ax_or_None)
        """
        plt = self._plt
        n_img = self._n_image_cols

        if self.show_convergence:
            fig = plt.figure(figsize=self.figsize or (5 * n_img + 3, 5.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
            gs_img = gs[0].subgridspec(1, n_img, wspace=0.15)
            image_axes = [fig.add_subplot(gs_img[i]) for i in range(n_img)]
            conv_ax = fig.add_subplot(gs[1])
            return fig, image_axes, conv_ax
        else:
            fig, axes = plt.subplots(1, n_img, figsize=self.figsize or (5 * n_img, 4.5))
            if n_img == 1:
                axes = [axes]
            return fig, list(axes), None

    def _draw_images(self, image_axes: List[Any], frame: Dict[str, Any]):
        """在 image_axes 上绘制各幅图像（不创建新的 colorbar，而是复用已有的）"""
        plt = self._plt
        mask = frame['mask']
        aerial = frame.get('aerial')
        wafer = frame.get('wafer')
        error = frame.get('error')

        col_idx = 0

        ax = image_axes[col_idx]
        ax.imshow(mask, cmap=self.cmap_mask, vmin=0, vmax=1)
        ax.set_title('Mask')
        ax.axis('off')
        col_idx += 1

        if wafer is not None:
            ax = image_axes[col_idx]
            ax.imshow(aerial, cmap=self.cmap_aerial, vmin=0, vmax=1)
            ax.set_title('Aerial Image')
            ax.axis('off')
            col_idx += 1

            ax = image_axes[col_idx]
            ax.imshow(wafer, cmap=self.cmap_wafer, vmin=0, vmax=1)
            ax.set_title('Wafer Image')
            ax.axis('off')
            col_idx += 1
        else:
            ax = image_axes[col_idx]
            if aerial is not None:
                ax.imshow(aerial, cmap=self.cmap_aerial, vmin=0, vmax=1)
                ax.set_title('Aerial Image')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_title('Aerial Image')
            ax.axis('off')
            col_idx += 1

        ax = image_axes[col_idx]
        if error is not None:
            vmax = self._error_vmax if self._error_vmax is not None else (
                float(np.max(error)) if np.max(error) > 0 else 1.0
            )
            im = ax.imshow(error, cmap=self.cmap_error, vmin=0, vmax=vmax)
            cbar_ax = getattr(ax, '_cbar_ax', None)
            if cbar_ax is None:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
                ax._cbar_ax = cbar_ax
                fig = ax.figure
                fig.colorbar(im, cax=cbar_ax)
            else:
                cbar_ax.clear()
                fig = ax.figure
                fig.colorbar(im, cax=cbar_ax)
            ax.set_title(f'Error Map (max={vmax:.3f})')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Error Map')
        ax.axis('off')

    def _draw_convergence(self, conv_ax: Any, frame: Dict[str, Any]):
        """在 conv_ax 上绘制到当前帧为止的损失曲线"""
        if self.state is None or not self.state.loss_history:
            losses = [f['loss'] for f in self._frames]
        else:
            losses = list(self.state.loss_history)

        conv_ax.clear()
        epochs = list(range(1, len(losses) + 1))
        conv_ax.plot(epochs, losses, 'b-', linewidth=1.2, label='loss')
        if losses:
            current_epoch = frame['epoch']
            current_idx = min(len(losses) - 1, max(0, current_epoch - 1))
            conv_ax.axvline(x=epochs[current_idx], color='r', linestyle='--',
                            linewidth=1, alpha=0.7, label=f'current')
            conv_ax.scatter([epochs[current_idx]], [losses[current_idx]],
                            color='r', s=25, zorder=5)
        conv_ax.set_title('Loss Convergence')
        conv_ax.set_xlabel('Epoch')
        conv_ax.set_ylabel('Loss')
        conv_ax.grid(True, alpha=0.3)
        if losses and min(losses) > 0 and (max(losses) / max(min(losses), 1e-12)) > 100:
            conv_ax.set_yscale('log')
        conv_ax.legend(loc='upper right', fontsize=8)

    def _render_frame(self, frame_idx: int) -> Any:
        """渲染第 frame_idx 帧，返回 matplotlib Figure"""
        frame = self._frames[frame_idx]
        fig, image_axes, conv_ax = self._figure_layout()
        self._draw_images(image_axes, frame)
        if conv_ax is not None:
            self._draw_convergence(conv_ax, frame)
        if self.show_title_info:
            info = (f"Epoch {frame['epoch']} | "
                    f"Loss {frame['loss']:.3e} | "
                    f"LR {frame['learning_rate']:.3e}")
            fig.suptitle(info, fontsize=12, y=1.02)
        return fig

    def _generate_animation(self):
        """使用 FuncAnimation 生成动画并保存"""
        plt = self._plt
        animation = self._animation
        n_frames = len(self._frames)

        fig, image_axes_ref, conv_ax_ref = self._figure_layout()
        self._draw_images(image_axes_ref, self._frames[0])
        if conv_ax_ref is not None:
            self._draw_convergence(conv_ax_ref, self._frames[0])
        if self.show_title_info:
            frame = self._frames[0]
            info = (f"Epoch {frame['epoch']} | "
                    f"Loss {frame['loss']:.3e} | "
                    f"LR {frame['learning_rate']:.3e}")
            fig.suptitle(info, fontsize=12, y=1.02)

        def update(frame_idx: int):
            """更新第 frame_idx 帧内容（复用已有 figure）"""
            frame = self._frames[frame_idx]
            self._draw_images(image_axes_ref, frame)
            if conv_ax_ref is not None:
                self._draw_convergence(conv_ax_ref, frame)
            if self.show_title_info:
                info = (f"Epoch {frame['epoch']} | "
                        f"Loss {frame['loss']:.3e} | "
                        f"LR {frame['learning_rate']:.3e}")
                fig.suptitle(info, fontsize=12, y=1.02)
            return []

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=int(1000 / max(1, self.fps)),
            blit=False,
            repeat=True,
        )

        save_kwargs = dict(fps=self.fps, dpi=self.dpi)

        if self.output_format == 'gif':
            filename = f'{self.filename_prefix}.gif'
            self.output_path = self.save_dir / filename
            writer = None
            try:
                ani.save(str(self.output_path), writer='pillow', **save_kwargs)
            except Exception as e:
                logger.debug(f"pillow writer 失败，尝试 imagemagick: {e}")
                try:
                    ani.save(str(self.output_path), writer='imagemagick', **save_kwargs)
                except Exception as e2:
                    logger.warning(f"GIF 保存失败 (pillow & imagemagick): {e2}")
                    plt.close(fig)
                    return
        elif self.output_format == 'mp4':
            filename = f'{self.filename_prefix}.mp4'
            self.output_path = self.save_dir / filename
            try:
                writer = animation.FFMpegWriter(
                    fps=self.fps,
                    metadata=dict(artist='MaskOptimizer'),
                    bitrate=2000,
                )
                ani.save(str(self.output_path), writer=writer, dpi=self.dpi)
            except Exception as e:
                logger.debug(f"ffmpeg writer 失败，回退到 GIF: {e}")
                try:
                    filename = f'{self.filename_prefix}.gif'
                    self.output_path = self.save_dir / filename
                    ani.save(str(self.output_path), writer='pillow', **save_kwargs)
                    logger.info("ffmpeg 不可用，已回退为 GIF 格式输出")
                except Exception as e2:
                    logger.warning(f"MP4/GIF 保存均失败: {e2}")
                    plt.close(fig)
                    return
        else:
            logger.warning(f"未知的动画格式: {self.output_format}，回退为 GIF")
            filename = f'{self.filename_prefix}.gif'
            self.output_path = self.save_dir / filename
            ani.save(str(self.output_path), writer='pillow', **save_kwargs)

        plt.close(fig)
        logger.info(f"优化动画已保存: {self.output_path} (共 {n_frames} 帧, {self.fps} fps)")

    def get_frames(self) -> List[Dict[str, Any]]:
        """获取所有记录的帧数据"""
        return list(self._frames)


class ExperimentTrackingCallback(Callback):
    """
    实验追踪回调

    将训练过程中的配置、指标、耗时等信息记录到实验追踪系统中。
    支持 MLflow、WandB 和本地文件后端。
    """

    def __init__(self,
                 backend: str = "local",
                 experiment_name: str = "mask_optimization",
                 run_name: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None,
                 tracking_dir: str = "./mlruns",
                 tracking_uri: Optional[str] = None,
                 wandb_project: Optional[str] = None,
                 wandb_entity: Optional[str] = None,
                 log_config: bool = True,
                 log_metrics_freq: int = 1,
                 log_system_metrics: bool = False):
        """
        初始化实验追踪回调

        Args:
            backend: 追踪后端: 'local', 'mlflow', 'wandb'
            experiment_name: 实验名称
            run_name: 运行名称
            tags: 标签字典
            tracking_dir: 本地追踪目录（local 后端）
            tracking_uri: MLflow tracking URI（mlflow 后端）
            wandb_project: WandB 项目名（wandb 后端）
            wandb_entity: WandB 实体名（wandb 后端）
            log_config: 是否记录配置
            log_metrics_freq: 记录指标的频率（每多少 epoch 记录一次）
            log_system_metrics: 是否记录系统指标
        """
        super().__init__()
        self.backend = backend.lower()
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.tracking_dir = tracking_dir
        self.tracking_uri = tracking_uri
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.log_config = log_config
        self.log_metrics_freq = max(1, log_metrics_freq)
        self.log_system_metrics = log_system_metrics

        self._tracker = None
        self._run_id: Optional[str] = None
        self._start_time: float = 0.0
        self._epoch_count: int = 0

    def _init_tracker(self):
        """初始化追踪器"""
        from utils.experiment_tracking import create_tracker

        kwargs = {
            "experiment_name": self.experiment_name,
        }

        if self.backend == "local":
            kwargs["tracking_dir"] = self.tracking_dir
        elif self.backend == "mlflow":
            if self.tracking_uri:
                kwargs["tracking_uri"] = self.tracking_uri
        elif self.backend == "wandb":
            if self.wandb_project:
                kwargs["project"] = self.wandb_project
            if self.wandb_entity:
                kwargs["entity"] = self.wandb_entity

        self._tracker = create_tracker(self.backend, **kwargs)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始，初始化实验追踪"""
        try:
            self._init_tracker()

            tags = dict(self.tags)
            if self.params:
                for k, v in self.params.items():
                    if isinstance(v, (str, int, float, bool)):
                        tags[f"param_{k}"] = str(v)

            self._run_id = self._tracker.start_run(
                run_name=self.run_name,
                tags=tags,
            )
            self._start_time = time.time()
            self._epoch_count = 0

            if self.log_config and self.params:
                self._tracker.log_config(self.params)

            logger.info(f"实验追踪已启动: {self._run_id} (后端: {self.backend})")

        except Exception as e:
            logger.warning(f"实验追踪初始化失败，将跳过追踪: {e}")
            self._tracker = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 epoch 结束，记录指标"""
        if self._tracker is None:
            return

        if epoch % self.log_metrics_freq != 0:
            return

        self._epoch_count = epoch

        try:
            metrics = {}

            if logs:
                for key, value in logs.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        metrics[key] = float(value)

            if self.state is not None:
                metrics["learning_rate"] = float(self.state.learning_rate)

            if metrics:
                self._tracker.log_metrics(metrics, step=epoch)

        except Exception as e:
            logger.debug(f"记录实验指标失败 (epoch {epoch}): {e}")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束，记录最终结果和耗时"""
        if self._tracker is None:
            return

        try:
            total_time = time.time() - self._start_time
            self._tracker.log_metric("total_time", total_time, step=self._epoch_count)

            if self.state is not None:
                self._tracker.log_metric("best_loss", float(self.state.best_loss), step=self._epoch_count)

            if logs:
                final_metrics = {}
                for key, value in logs.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        final_metrics[f"final_{key}"] = float(value)
                if final_metrics:
                    self._tracker.log_metrics(final_metrics, step=self._epoch_count)

            self._tracker.set_tag("status", "completed")
            self._tracker.end_run(status="completed")

            logger.info(f"实验追踪已完成: {self._run_id}")

        except Exception as e:
            logger.warning(f"结束实验追踪时出错: {e}")
            try:
                self._tracker.end_run(status="failed")
            except Exception:
                pass

    @property
    def run_id(self) -> Optional[str]:
        """获取当前运行 ID"""
        return self._run_id

    @property
    def tracker(self):
        """获取底层追踪器"""
        return self._tracker


# ============================================================================
# 工作流级别 Checkpoint 支持（用于多阶段流程：SMO/OPC/ILT等）
# ============================================================================


@dataclass
class WorkflowCheckpointState:
    """
    工作流 Checkpoint 完整状态

    用于在多阶段工作流（SMO/OPC/ILT 等）中持久化完整运行状态，
    支持 Docker 容器重启或集群节点故障后从最近 checkpoint 恢复。

    Attributes:
        workflow_type: 工作流类型标识 ('SMO', 'OPC', 'ILT', 'HYBRID' 等)
        workflow_version: 工作流代码版本标识（用于兼容性检查）
        checkpoint_id: 唯一 checkpoint 标识（时间戳+哈希）
        created_at: 创建时间戳（秒）
        outer_iteration: 当前外层迭代次数（如 SMO 的外层交替次数）
        inner_iteration: 当前内层迭代次数（如 SMO 的子阶段迭代）
        current_phase: 当前子阶段标识（如 SMO 的 'source'/'mask'/'joint'）
        source: 像素化光源分布（2D ndarray，SMO 特有）
        mask: 当前掩模状态（2D ndarray）
        best_loss: 历史最优损失值
        best_mask: 历史最优掩模
        best_source: 历史最优光源（SMO 特有）
        loss_history: 损失历史列表
        loss_components_history: 各损失分量历史
        random_seed_numpy: numpy 随机状态
        random_seed_python: Python random 模块状态
        extra_data: 工作流特有的额外数据字典
                     - SMO: source_grid_size, source_constraints, patience_counter 等
                     - OPC: srafs, hotspots, transform_history 等
        config_hash: 配置参数的哈希（用于验证配置一致性）
    """
    workflow_type: str
    workflow_version: str = '1.0.0'
    checkpoint_id: str = ''
    created_at: float = 0.0

    outer_iteration: int = 0
    inner_iteration: int = 0
    current_phase: str = ''

    source: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    best_loss: float = float('inf')
    best_mask: Optional[np.ndarray] = None
    best_source: Optional[np.ndarray] = None

    loss_history: List[float] = field(default_factory=list)
    loss_components_history: List[Dict[str, float]] = field(default_factory=list)

    random_seed_numpy: Optional[Tuple[Any, ...]] = None
    random_seed_python: Optional[Tuple[Any, ...]] = None

    extra_data: Dict[str, Any] = field(default_factory=dict)
    config_hash: str = ''

    def save(self, filepath: Union[str, Path]) -> None:
        """
        保存 checkpoint 状态到文件

        使用 npz + pickle 组合格式，确保大型数组高效存储。
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if not self.checkpoint_id:
            self.checkpoint_id = self._generate_checkpoint_id()
        if not self.created_at:
            self.created_at = time.time()

        npz_data: Dict[str, Any] = {
            'workflow_type': self.workflow_type,
            'workflow_version': self.workflow_version,
            'checkpoint_id': self.checkpoint_id,
            'created_at': self.created_at,
            'outer_iteration': self.outer_iteration,
            'inner_iteration': self.inner_iteration,
            'current_phase': self.current_phase,
            'best_loss': self.best_loss,
            'config_hash': self.config_hash,
        }

        if self.source is not None:
            npz_data['source'] = self.source
        if self.mask is not None:
            npz_data['mask'] = self.mask
        if self.best_mask is not None:
            npz_data['best_mask'] = self.best_mask
        if self.best_source is not None:
            npz_data['best_source'] = self.best_source
        if self.loss_history:
            npz_data['loss_history'] = np.array(self.loss_history)

        np.savez(filepath.with_suffix('.npz'), **npz_data)

        pickle_data = {
            'loss_components_history': self.loss_components_history,
            'random_seed_numpy': self.random_seed_numpy,
            'random_seed_python': self.random_seed_python,
            'extra_data': self.extra_data,
        }
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta_path = filepath.with_suffix('.json')
        meta = {
            'workflow_type': self.workflow_type,
            'workflow_version': self.workflow_version,
            'checkpoint_id': self.checkpoint_id,
            'created_at': self.created_at,
            'created_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self.created_at)),
            'outer_iteration': self.outer_iteration,
            'inner_iteration': self.inner_iteration,
            'current_phase': self.current_phase,
            'best_loss': self.best_loss,
            'loss_count': len(self.loss_history),
            'has_source': self.source is not None,
            'has_mask': self.mask is not None,
            'config_hash': self.config_hash,
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"工作流 checkpoint 已保存: {filepath} (外层迭代={self.outer_iteration}, "
                   f"阶段={self.current_phase}, best_loss={self.best_loss:.4e})")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'WorkflowCheckpointState':
        """
        从文件加载 checkpoint 状态

        优先读取 .npz 和 .pkl，若存在 .json 则校验元数据一致性。
        """
        filepath = Path(filepath)
        base = filepath.with_suffix('')
        npz_path = base.with_suffix('.npz')
        pkl_path = base.with_suffix('.pkl')
        json_path = base.with_suffix('.json')

        if not npz_path.exists():
            raise FileNotFoundError(f"Checkpoint 文件不存在: {npz_path}")

        npz = np.load(npz_path, allow_pickle=True)
        data: Dict[str, Any] = {}
        for k in npz.files:
            v = npz[k]
            if isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
            data[k] = v

        state = cls(workflow_type=str(data.get('workflow_type', 'UNKNOWN')))
        state.workflow_version = str(data.get('workflow_version', '1.0.0'))
        state.checkpoint_id = str(data.get('checkpoint_id', ''))
        state.created_at = float(data.get('created_at', 0.0))
        state.outer_iteration = int(data.get('outer_iteration', 0))
        state.inner_iteration = int(data.get('inner_iteration', 0))
        state.current_phase = str(data.get('current_phase', ''))
        state.best_loss = float(data.get('best_loss', float('inf')))
        state.config_hash = str(data.get('config_hash', ''))

        if 'source' in data:
            state.source = data['source']
        if 'mask' in data:
            state.mask = data['mask']
        if 'best_mask' in data:
            state.best_mask = data['best_mask']
        if 'best_source' in data:
            state.best_source = data['best_source']
        if 'loss_history' in data:
            state.loss_history = list(data['loss_history'])

        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    pickle_data = pickle.load(f)
                state.loss_components_history = pickle_data.get('loss_components_history', [])
                state.random_seed_numpy = pickle_data.get('random_seed_numpy', None)
                state.random_seed_python = pickle_data.get('random_seed_python', None)
                state.extra_data = pickle_data.get('extra_data', {})
            except Exception as e:
                logger.warning(f"加载 checkpoint pickle 数据失败（可能为旧格式）: {e}")

        logger.info(f"工作流 checkpoint 已加载: {filepath} (外层迭代={state.outer_iteration}, "
                   f"阶段={state.current_phase}, best_loss={state.best_loss:.4e})")
        return state

    def capture_random_state(self) -> None:
        """捕获当前 numpy 和 Python 的随机种子状态"""
        self.random_seed_numpy = np.random.get_state()
        self.random_seed_python = random.getstate()

    def restore_random_state(self) -> None:
        """将 numpy 和 Python 的随机种子恢复到 checkpoint 时的状态"""
        if self.random_seed_numpy is not None:
            np.random.set_state(self.random_seed_numpy)
            logger.debug("numpy 随机状态已恢复")
        if self.random_seed_python is not None:
            random.setstate(self.random_seed_python)
            logger.debug("Python random 随机状态已恢复")

    @staticmethod
    def _generate_checkpoint_id() -> str:
        """生成唯一 checkpoint ID：时间戳+随机哈希"""
        timestamp = int(time.time() * 1000)
        rand = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        return f"ckpt_{timestamp}_{rand}"


class WorkflowCheckpointManager:
    """
    工作流 Checkpoint 管理器

    负责 checkpoint 的保存调度、历史管理、查找最近 checkpoint 等功能。
    支持多阶段工作流（SMO/OPC/ILT 等）的统一管理。

    典型使用：
        mgr = WorkflowCheckpointManager(
            checkpoint_dir='./checkpoints/smo_run_001',
            workflow_type='SMO',
            save_freq_outer=1,   # 每 N 次外层迭代保存
            max_checkpoints=10,
        )
        # 启动时尝试恢复
        latest = mgr.find_latest_checkpoint()
        if latest:
            state = mgr.load_checkpoint(latest)
            state.restore_random_state()
            # ... 从 state 中恢复工作流状态

        # 每次外层迭代结束时
        mgr.save_if_needed(state, outer_iter, force=False)
    """

    def __init__(self,
                 checkpoint_dir: Union[str, Path],
                 workflow_type: str,
                 save_freq_outer: int = 1,
                 save_freq_inner: int = 0,
                 max_checkpoints: int = 10,
                 save_best_only: bool = False,
                 filename_prefix: str = 'workflow',
                 config: Optional[Any] = None):
        """
        初始化 Checkpoint 管理器

        Args:
            checkpoint_dir: checkpoint 保存根目录
            workflow_type: 工作流类型标识（'SMO'/'OPC'/'ILT' 等）
            save_freq_outer: 外层迭代保存频率（每 N 次外层迭代保存一次），0 禁用
            save_freq_inner: 内层迭代保存频率（每 N 次内层迭代保存一次），0 禁用
            max_checkpoints: 最多保留的 checkpoint 文件数量（含 best），0 不限制
            save_best_only: 是否只保留最优 checkpoint
            filename_prefix: 文件名前缀
            config: 工作流配置对象（用于 config_hash 一致性校验）
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.workflow_type = workflow_type
        self.save_freq_outer = max(0, int(save_freq_outer))
        self.save_freq_inner = max(0, int(save_freq_inner))
        self.max_checkpoints = max(0, int(max_checkpoints))
        self.save_best_only = save_best_only
        self.filename_prefix = filename_prefix

        self._saved_files: List[Path] = []
        self._best_filepath: Optional[Path] = None
        self._best_loss: float = float('inf')

        self.config_hash = ''
        if config is not None:
            self.config_hash = self._compute_config_hash(config)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _compute_config_hash(self, config: Any) -> str:
        """计算配置的哈希值，用于恢复时验证配置一致性"""
        try:
            if hasattr(config, 'to_dict'):
                cfg_str = json.dumps(config.to_dict(), sort_keys=True, default=str)
            elif isinstance(config, dict):
                cfg_str = json.dumps(config, sort_keys=True, default=str)
            else:
                cfg_str = str(config)
            return hashlib.sha256(cfg_str.encode('utf-8')).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"计算配置哈希失败: {e}")
            return ''

    def should_save(self,
                    outer_iteration: int,
                    inner_iteration: int = 0,
                    force: bool = False) -> bool:
        """
        判断当前是否应该保存 checkpoint

        Args:
            outer_iteration: 当前外层迭代次数
            inner_iteration: 当前内层迭代次数
            force: 是否强制保存

        Returns:
            是否需要保存
        """
        if force:
            return True
        if self.save_freq_outer > 0 and outer_iteration > 0 and \
                outer_iteration % self.save_freq_outer == 0:
            return True
        if self.save_freq_inner > 0 and inner_iteration > 0 and \
                inner_iteration % self.save_freq_inner == 0:
            return True
        return False

    def save_checkpoint(self,
                        state: WorkflowCheckpointState,
                        outer_iteration: int,
                        current_loss: Optional[float] = None,
                        force: bool = False) -> Optional[Path]:
        """
        根据条件决定是否保存 checkpoint

        Args:
            state: 要保存的状态
            outer_iteration: 当前外层迭代次数
            current_loss: 当前损失值（用于判断 best）
            force: 是否强制保存

        Returns:
            若保存了则返回文件路径，否则返回 None
        """
        state.workflow_type = self.workflow_type
        if not state.config_hash and self.config_hash:
            state.config_hash = self.config_hash

        state.capture_random_state()

        is_best = False
        if current_loss is not None and current_loss < self._best_loss:
            self._best_loss = current_loss
            is_best = True

        if self.save_best_only and not is_best:
            return None

        if not self.should_save(outer_iteration, state.inner_iteration, force) and not is_best:
            return None

        if is_best and not self.should_save(outer_iteration, state.inner_iteration, force):
            pass

        filename_parts = [
            self.filename_prefix,
            f"outer_{outer_iteration:04d}",
        ]
        if state.inner_iteration > 0:
            filename_parts.append(f"inner_{state.inner_iteration:04d}")
        if state.current_phase:
            filename_parts.append(state.current_phase)
        if is_best:
            filename_parts.append('best')

        filename = '_'.join(filename_parts)
        filepath = self.checkpoint_dir / filename

        state.save(filepath)

        if is_best:
            self._best_filepath = filepath
            best_link = self.checkpoint_dir / f'{self.filename_prefix}_latest_best'
            for suffix in ['.npz', '.pkl', '.json']:
                src = filepath.with_suffix(suffix)
                dst = best_link.with_suffix(suffix)
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if src.exists():
                    try:
                        if hasattr(os, 'symlink'):
                            os.symlink(src, dst)
                        else:
                            import shutil
                            shutil.copy2(src, dst)
                    except Exception as e:
                        logger.debug(f"创建 best checkpoint 链接失败: {e}")

        if not is_best:
            self._saved_files.append(filepath)
            self._rotate_checkpoints()

        return filepath

    def _rotate_checkpoints(self) -> None:
        """轮转 checkpoint，删除最旧的以保持 max_checkpoints 限制"""
        if self.max_checkpoints <= 0:
            return
        while len(self._saved_files) > self.max_checkpoints:
            old = self._saved_files.pop(0)
            if old == self._best_filepath:
                continue
            for suffix in ['.npz', '.pkl', '.json']:
                f = old.with_suffix(suffix)
                if f.exists():
                    try:
                        f.unlink()
                    except Exception as e:
                        logger.warning(f"删除旧 checkpoint 失败 {f}: {e}")
            logger.debug(f"已删除旧 checkpoint: {old}")

    def _iter_checkpoint_candidates(self) -> List[Path]:
        """
        枚举所有真实 checkpoint（排除 _latest_best 快捷符号），返回 .json 元数据路径列表

        注意：文件名末尾带 _best 的真实 checkpoint（如 test_outer_0005_best.json）
        是被标记为 best 的真实保存文件，**必须保留**；仅跳过
        `{prefix}_latest_best.json` 这种专门的快捷方式链接/拷贝。
        """
        if not self.checkpoint_dir.exists():
            return []
        json_files: List[Path] = []
        latest_best_tag = f'{self.filename_prefix}_latest_best.json'
        for f in self.checkpoint_dir.glob(f'{self.filename_prefix}_*.json'):
            if f.name == latest_best_tag:
                continue
            json_files.append(f)
        return json_files

    def _rank_checkpoint_by_recency(self, json_files: List[Path]) -> List[Path]:
        """
        按 最近→最旧 排序 checkpoint：
        优先级：outer_iteration 降序 → inner_iteration 降序 → created_at 降序 → mtime 降序
        """
        def _sort_key(f: Path):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    meta = json.load(fh)
                outer = int(meta.get('outer_iteration', 0))
                inner = int(meta.get('inner_iteration', 0))
                created = float(meta.get('created_at', 0.0))
            except Exception:
                outer, inner, created = 0, 0, 0.0
            try:
                mtime = f.stat().st_mtime
            except Exception:
                mtime = 0.0
            return (-outer, -inner, -created, -mtime)

        return sorted(json_files, key=_sort_key)

    def _validate_checkpoint(self,
                             base_path: Path,
                             validate_config: bool,
                             expected_hash: Optional[str]) -> bool:
        """验证 checkpoint 元数据合法性（配置哈希等）"""
        try:
            meta_path = base_path.with_suffix('.json')
            if meta_path.exists() and validate_config:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                file_hash = meta.get('config_hash', '') or ''
                if expected_hash is not None and file_hash:
                    if file_hash != expected_hash:
                        logger.warning(
                            f"跳过 checkpoint {base_path.name}: 配置哈希不一致 "
                            f"(期望 {expected_hash[:8] if len(expected_hash) >= 8 else expected_hash}..., "
                            f"文件 {file_hash[:8]}...)"
                        )
                        return False
            npz_path = base_path.with_suffix('.npz')
            if not npz_path.exists():
                logger.warning(f"跳过 checkpoint {base_path.name}: 缺少 .npz 文件")
                return False
            return True
        except Exception as e:
            logger.warning(f"检查 checkpoint {base_path} 失败: {e}")
            return False

    def find_latest_checkpoint(self,
                               validate_config: bool = True,
                               expected_config_hash: Optional[str] = None) -> Optional[Path]:
        """
        在 checkpoint_dir 中查找**最近一次**保存的 checkpoint 文件（严格按时间/迭代号最近）

        排序优先级：outer_iteration 降序 → inner_iteration 降序 → created_at 降序 → 文件 mtime 降序
        不返回 _latest_best 快捷方式（它指向 best，可能不是最近）。

        Args:
            validate_config: 是否验证配置哈希一致
            expected_config_hash: 期望的配置哈希（默认使用 self.config_hash）

        Returns:
            checkpoint 文件基路径（不含后缀），未找到返回 None
        """
        if not self.checkpoint_dir.exists():
            return None

        expected_hash = expected_config_hash or self.config_hash
        json_files = self._iter_checkpoint_candidates()
        if not json_files:
            return None

        ranked = self._rank_checkpoint_by_recency(json_files)
        for json_file in ranked:
            base = json_file.with_suffix('')
            if self._validate_checkpoint(base, validate_config, expected_hash):
                return base
        return None

    def find_best_checkpoint(self,
                             validate_config: bool = True,
                             expected_config_hash: Optional[str] = None) -> Optional[Path]:
        """
        查找**最优损失（best_loss 最小）**的 checkpoint（供对比或回退使用，非断点续跑默认入口）

        Args:
            validate_config: 是否验证配置哈希一致
            expected_config_hash: 期望的配置哈希

        Returns:
            checkpoint 文件基路径（不含后缀），未找到返回 None
        """
        if not self.checkpoint_dir.exists():
            return None
        expected_hash = expected_config_hash or self.config_hash

        best_link = self.checkpoint_dir / f'{self.filename_prefix}_latest_best'
        if best_link.with_suffix('.npz').exists() or best_link.with_suffix('.json').exists():
            if self._validate_checkpoint(best_link, validate_config, expected_hash):
                return best_link

        json_files = self._iter_checkpoint_candidates()
        if not json_files:
            return None

        best_path: Optional[Path] = None
        best_loss = float('inf')
        for json_file in json_files:
            base = json_file.with_suffix('')
            if not self._validate_checkpoint(base, validate_config, expected_hash):
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                loss = float(meta.get('best_loss', float('inf')))
            except Exception:
                continue
            if loss < best_loss:
                best_loss = loss
                best_path = base
        return best_path

    def load_checkpoint(self, filepath: Union[str, Path]) -> WorkflowCheckpointState:
        """加载 checkpoint 状态的便捷方法"""
        return WorkflowCheckpointState.load(filepath)

    def list_all_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出目录中所有 checkpoint 的元信息

        Returns:
            元信息字典列表（按最近→最旧排序：outer_iteration desc → inner_iteration desc → created_at desc）
        """
        if not self.checkpoint_dir.exists():
            return []

        json_files = self._iter_checkpoint_candidates()
        ranked = self._rank_checkpoint_by_recency(json_files)

        result = []
        for json_file in ranked:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta['filepath'] = str(json_file.with_suffix(''))
                result.append(meta)
            except Exception:
                continue
        return result

    def cleanup_all(self, keep_best: bool = True) -> int:
        """
        清理所有 checkpoint 文件

        Args:
            keep_best: 是否保留标记为 best 的 checkpoint

        Returns:
            删除的文件数量
        """
        count = 0
        if not self.checkpoint_dir.exists():
            return count

        for suffix in ['.npz', '.pkl', '.json']:
            pattern = f'{self.filename_prefix}_*{suffix}'
            for f in self.checkpoint_dir.glob(pattern):
                if keep_best and '_best.' in f.name:
                    continue
                if keep_best and '_latest_best.' in f.name:
                    continue
                try:
                    f.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"删除 checkpoint 失败 {f}: {e}")

        self._saved_files.clear()
        logger.info(f"已清理 {count} 个 checkpoint 文件")
        return count


class WorkflowCheckpointCallback(Callback):
    """
    工作流级别的 Checkpoint 回调（适配 Callback 接口）

    与 WorkflowCheckpointManager 配合使用，在训练回调的生命周期中
    自动管理 checkpoint。适用于将工作流嵌入到 Trainer 的场景。
    """

    def __init__(self,
                 checkpoint_manager: WorkflowCheckpointManager,
                 state_provider: Callable[[], WorkflowCheckpointState],
                 loss_provider: Optional[Callable[[], float]] = None):
        """
        初始化工作流 Checkpoint 回调

        Args:
            checkpoint_manager: checkpoint 管理器实例
            state_provider: 调用时返回当前 WorkflowCheckpointState 的函数
            loss_provider: 调用时返回当前损失值的函数（可选）
        """
        super().__init__()
        self.manager = checkpoint_manager
        self.state_provider = state_provider
        self.loss_provider = loss_provider
        self._last_saved_path: Optional[Path] = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 epoch 结束时尝试保存 checkpoint"""
        state = self.state_provider()
        state.outer_iteration = epoch
        if logs and 'loss' in logs:
            current_loss = float(logs['loss'])
        elif self.loss_provider is not None:
            current_loss = self.loss_provider()
        else:
            current_loss = None

        path = self.manager.save_checkpoint(state, epoch, current_loss)
        if path is not None:
            self._last_saved_path = path

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束时强制保存最终 checkpoint"""
        state = self.state_provider()
        if state.outer_iteration == 0 and self.state is not None:
            state.outer_iteration = self.state.epoch
        current_loss = None
        if logs and 'loss' in logs:
            current_loss = float(logs['loss'])
        elif self.loss_provider is not None:
            current_loss = self.loss_provider()
        elif self.state is not None:
            current_loss = self.state.best_loss

        self.manager.save_checkpoint(state, state.outer_iteration, current_loss, force=True)

    @property
    def last_saved_path(self) -> Optional[Path]:
        """返回最近一次保存的 checkpoint 路径"""
        return self._last_saved_path

