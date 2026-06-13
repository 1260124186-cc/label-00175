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
from io import BytesIO

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
