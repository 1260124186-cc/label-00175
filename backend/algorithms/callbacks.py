# -*- coding: utf-8 -*-
"""
回调模块：训练过程中的各种回调

提供统一的回调接口，支持学习率调度、早停、checkpoint、
中间掩模保存、收敛曲线绘制等功能。
"""

import os
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

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
