# -*- coding: utf-8 -*-
"""
掩模优化模块：完整的掩模图案优化流程

该模块实现"掩模图案→光学成像→误差计算→参数更新"的迭代优化流程，
支持早停、学习率调度等功能。
"""

import numpy as np
from typing import Optional, Callable, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from core.imaging import (
    OpticalSystem, PartialCoherentImaging, simulate_wafer_image,
    ProcessCondition, ProcessWindow, MultiProcessSimulationResult,
    simulate_multi_process, create_focus_dose_window, create_full_process_window,
    downsample_mask, upsample_mask, build_pyramid_scales,
    split_tiles, merge_tiles_with_blend
)
from core.rigorous_sim import (
    SimulationBackend,
    RCWAConfig,
    FDTDConfig,
    simulate as unified_simulate,
    simulate_multi_process_unified,
)
from core.fft import (
    create_bandlimit_mask, bandlimit_projection,
    bandlimited_gradient_projection, BandlimitType
)
from core.metrics import (
    mse, mae, ssim, evaluate_all, MetricsResult,
    ssim_loss_gradient,
    total_variation, total_variation_gradient,
    total_variation_anisotropic, total_variation_isotropic,
    total_variation_isotropic_gradient,
    pvb, pvb_gradient,
    l1_regularization, l1_regularization_gradient,
    l2_regularization, l2_regularization_gradient,
    tv_regularization, tv_regularization_gradient,
    manhattan_distance_penalty, manhattan_distance_penalty_gradient,
    binary_entropy_penalty, binary_entropy_penalty_gradient,
    edge_placement_error, edge_placement_error_gradient,
    soft_edge_placement_error, soft_edge_placement_error_gradient,
    min_feature_size_morphology, min_feature_size_frequency,
    soft_min_feature_size_morphology, soft_min_feature_size_morphology_gradient,
    min_feature_size_frequency_gradient,
    min_feature_size_combined, min_feature_size_combined_gradient,
    CompositeLossComponents,
    SpatialWeightConfig, generate_spatial_weight_mask,
    weighted_mse, weighted_mae,
    weighted_mse_gradient, weighted_mae_gradient
)
from algorithms.optimizer import (
    BaseOptimizer, GradientDescentOptimizer, BFGSOptimizer,
    NewtonOptimizer, OptimizationResult, AdamOptimizer, RMSpropOptimizer
)
from algorithms.advanced_optimizer import (
    BaseHeuristicOptimizer, GeneticAlgorithmOptimizer, ParticleSwarmOptimizer,
    SimulatedAnnealingOptimizer, DifferentialEvolutionOptimizer, CMAESOptimizer
)
from algorithms.callbacks import (
    Callback, CallbackList, TrainerState,
    LearningRateSchedulerCallback, EarlyStoppingCallback,
    ModelCheckpointCallback, MaskSnapshotCallback,
    ConvergencePlotCallback, LoggerCallback, HistoryCallback,
    LambdaCallback, AnimationCallback
)

try:
    from surrogate import SurrogateImaging
    _HAS_SURROGATE = True
except ImportError:
    _HAS_SURROGATE = False

logger = logging.getLogger(__name__)


def _apply_threshold_for_loss(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    可微近似阈值函数（sigmoid平滑），用于wafer图像损失计算

    使用sigmoid函数近似硬阈值，使得梯度可以传递。

    Args:
        image: 输入光强图像
        threshold: 阈值

    Returns:
        平滑阈值后的图像
    """
    k = 50.0
    return 1.0 / (1.0 + np.exp(-k * (image - threshold)))


@dataclass
class LossWeights:
    """
    复合损失函数各分量权重配置

    loss = w_mse * MSE + w_ssim * (1-SSIM) + w_pvb * PVB + w_mask_complexity * mask_complexity
           + w_binary * binary_penalty + w_tv_smooth * TV_smooth
           + w_epe * EPE + w_min_feature * min_feature_size
           + w_weighted_mse * WMSE + w_weighted_mae * WMAE

    Attributes:
        mse: MSE（均方误差）权重
        ssim: (1-SSIM) 结构相似性损失权重
        pvb: PVB（Process Variation Band，工艺变化带宽）权重
        mask_complexity: 掩模复杂度（总变差TV）权重
        binary_penalty: 二值化惩罚权重（曼哈顿距离/熵）
        tv_smooth: 各向同性TV平滑权重
        epe: 边缘放置误差（EPE）权重
        min_feature: 最小特征尺寸约束权重
        weighted_mse: 空间加权MSE权重（热点区域更受关注）
        weighted_mae: 空间加权MAE权重
    """
    mse: float = 1.0
    ssim: float = 0.0
    pvb: float = 0.0
    mask_complexity: float = 0.0
    binary_penalty: float = 0.0
    tv_smooth: float = 0.0
    epe: float = 0.0
    min_feature: float = 0.0
    weighted_mse: float = 0.0
    weighted_mae: float = 0.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, float]]) -> 'LossWeights':
        """从字典创建，缺失键使用默认值"""
        if d is None:
            return cls()
        defaults = {
            'mse': 1.0, 'ssim': 0.0, 'pvb': 0.0, 'mask_complexity': 0.0,
            'binary_penalty': 0.0, 'tv_smooth': 0.0, 'epe': 0.0, 'min_feature': 0.0,
            'weighted_mse': 0.0, 'weighted_mae': 0.0
        }
        defaults.update(d)
        return cls(**defaults)

    def to_dict(self) -> Dict[str, float]:
        return {
            'mse': self.mse,
            'ssim': self.ssim,
            'pvb': self.pvb,
            'mask_complexity': self.mask_complexity,
            'binary_penalty': self.binary_penalty,
            'tv_smooth': self.tv_smooth,
            'epe': self.epe,
            'min_feature': self.min_feature,
            'weighted_mse': self.weighted_mse,
            'weighted_mae': self.weighted_mae
        }

    def total_weight(self) -> float:
        return (self.mse + self.ssim + self.pvb + self.mask_complexity +
                self.binary_penalty + self.tv_smooth + self.epe + self.min_feature +
                self.weighted_mse + self.weighted_mae)


@dataclass
class RegularizationConfig:
    """
    正则化配置

    Attributes:
        type: 正则化类型: 'l1', 'l2', 'tv', 'tv_isotropic',
              'manhattan', 'binary_entropy', 'epe', 'epe_soft',
              'min_feature_morph', 'min_feature_freq', 'min_feature_combined', None
        strength: 正则化强度系数
        params: 额外参数字典（如 min_size, sigma 等）
    """
    type: Optional[str] = None  # 'l1', 'l2', 'tv', 'tv_isotropic', 'manhattan',
                                 # 'binary_entropy', 'epe', 'epe_soft',
                                 # 'min_feature_morph', 'min_feature_freq', 'min_feature_combined'
    strength: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'RegularizationConfig':
        """从字典创建，缺失键使用默认值"""
        if d is None:
            return cls()
        return cls(
            type=d.get('type', None),
            strength=float(d.get('strength', 0.0)),
            params=dict(d.get('params', {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'strength': self.strength,
            'params': self.params
        }


@dataclass
class OptimizationConfig:
    """
    优化配置类

    Attributes:
        optimizer_type: 优化器类型
        max_iter: 最大迭代次数
        learning_rate: 学习率（梯度下降）
        tol: 收敛容差
        early_stop_patience: 早停耐心值
        lr_scheduler: 学习率调度器类型
        metric: 优化目标指标（兼容旧接口，当 use_composite_loss=False 时生效）
        use_composite_loss: 是否使用复合损失函数（优先级高于 metric）
        loss_weights: 复合损失各分量权重（use_composite_loss=True 时生效）
        regularization: 正则化配置
        bounds: 掩模值边界
        verbose: 是否输出详细信息
        random_seed: 随机种子（用于结果复现）

        # 多工艺条件联合优化配置
        use_multi_process: 是否启用多工艺条件联合优化
        process_conditions: 多工艺条件列表（与use_multi_process配合使用）
        process_window_mode: 工艺窗口模式
            - None: 不使用工艺窗口
            - 'focus_dose': 使用 focus-dose 二维窗口
            - 'full': 使用 focus-dose-NA-sigma 四维窗口
            - 'custom': 使用自定义 process_conditions
        focus_range: 离焦量扫描范围 (start, stop, n_points)
        dose_range: 曝光剂量扫描范围 (start, stop, n_points)
        na_range: 数值孔径扫描范围
        sigma_range: 部分相干因子扫描范围
        process_center_weight: 工艺窗口中心条件额外权重倍率
        process_edge_weight: 工艺窗口边界条件基础权重
        robustness_loss_weight: 工艺鲁棒性（方差/极差）损失权重，
                                用于同时最小化均值误差和条件间差异
        threshold: 光刻胶阈值（用于wafer图像损失）
        use_wafer_image_loss: 是否使用wafer图像（阈值后）计算损失

        # 掩模约束与正则化参数
        binary_penalty_type: str = 'manhattan'  # 'manhattan' 或 'entropy'
        epe_threshold: float = 0.5  # EPE 边缘检测阈值
        epe_sigma: float = 1.0  # 软 EPE 的高斯平滑 sigma
        epe_use_soft: bool = True  # 是否使用可微的软 EPE
        min_feature_size: int = 3  # 最小特征尺寸（像素）
        min_feature_method: str = 'combined'  # 'morphology', 'frequency', 'combined'
        min_feature_alpha: float = 0.5  # 联合方法中形态学权重
        pixel_size: float = 1.0  # 像素尺寸（用于EPE和最小特征尺寸的物理单位转换）
    """
    optimizer_type: str = 'gradient_descent'
    max_iter: int = 100
    learning_rate: float = 0.01
    tol: float = 1e-6
    early_stop_patience: int = 10
    lr_scheduler: Optional[str] = None  # 'step', 'exponential', 'cosine'
    lr_decay: float = 0.95
    lr_step_size: int = 20
    metric: str = 'mse'  # 'mse', 'mae', 'ssim'（兼容旧接口）
    use_composite_loss: bool = False  # 是否启用复合损失
    loss_weights: LossWeights = field(default_factory=LossWeights)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    bounds: tuple = (0.0, 1.0)
    verbose: bool = True
    random_seed: Optional[int] = None

    # 启发式算法参数
    population_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    n_jobs: int = 1

    # 多工艺条件联合优化
    use_multi_process: bool = False
    process_conditions: Optional[List[ProcessCondition]] = None
    process_window_mode: Optional[str] = None  # None, 'focus_dose', 'full', 'custom'
    focus_range: tuple = (-100, 100, 5)
    dose_range: tuple = (0.8, 1.2, 5)
    na_range: tuple = (1.30, 1.35, 3)
    sigma_range: tuple = (0.65, 0.75, 0.85, 3)
    process_center_weight: Optional[float] = 2.0
    process_edge_weight: float = 1.0
    robustness_loss_weight: float = 0.0  # 0表示不使用鲁棒性正则化
    threshold: float = 0.3
    use_wafer_image_loss: bool = False  # False使用aerial图像计算损失，True使用wafer图像

    # 掩模约束与正则化参数
    binary_penalty_type: str = 'manhattan'  # 'manhattan' 或 'entropy'
    epe_threshold: float = 0.5  # EPE 边缘检测阈值
    epe_sigma: float = 1.0  # 软 EPE 的高斯平滑 sigma
    epe_use_soft: bool = True  # 是否使用可微的软 EPE
    min_feature_size: int = 3  # 最小特征尺寸（像素）
    min_feature_method: str = 'combined'  # 'morphology', 'frequency', 'combined'
    min_feature_alpha: float = 0.5  # 联合方法中形态学权重
    pixel_size: float = 1.0  # 像素尺寸

    # 频域正则化（带限约束投影）配置
    use_frequency_bandlimit: bool = False  # 是否启用频域带限投影
    bandlimit_type: str = 'lowpass'  # 'lowpass', 'highpass', 'bandpass', 'bandstop', 'circular', 'rectangular', 'directional', 'custom'
    bandlimit_inner_radius: float = 0.0  # 内半径（归一化0-1）
    bandlimit_outer_radius: float = 0.5  # 外半径（归一化0-1）
    bandlimit_fx_range: tuple = (0.0, 0.5)  # 矩形带通x方向频率范围
    bandlimit_fy_range: tuple = (0.0, 0.5)  # 矩形带通y方向频率范围
    bandlimit_angle_range: tuple = (0.0, 6.283185307179586)  # 方向带通角度范围（弧度）
    bandlimit_smooth: bool = False  # 是否使用平滑过渡
    bandlimit_order: int = 4  # 巴特沃斯阶数（smooth=True时）
    bandlimit_preserve_dc: bool = True  # 是否保留直流分量
    bandlimit_projection_freq: int = 1  # 投影频率（每N次迭代投影一次）
    bandlimit_project_gradient: bool = False  # 是否对梯度也施加频域投影
    bandlimit_custom_mask: Any = None  # 自定义频域掩模（bandlimit_type='custom'时使用）

    # 多尺度/分块大尺寸掩模优化配置
    use_multiscale: bool = False  # 是否启用金字塔多尺度优化
    multiscale_mode: str = 'pyramid'  # 'pyramid' 或 'tile'
    # 金字塔多尺度参数
    pyramid_min_size: int = 64  # 金字塔最低分辨率的最小尺寸
    pyramid_scales: int = 3  # 金字塔层数（不含原始分辨率）
    pyramid_iter_ratio: float = 0.3  # 低分辨率层迭代数占总迭代数的比例
    # Tile分块参数
    tile_size: int = 256  # 单个 tile 的尺寸（像素）
    tile_overlap: int = 32  # 相邻 tile 的重叠区域（像素）
    tile_blend_sigma: float = 8.0  # 边界融合的高斯 sigma

    # Callback 系统配置
    use_callbacks: bool = True  # 是否启用 callback 系统
    callback_log_freq: int = 10  # 日志输出频率
    # 早停配置（通过 callback 实现）
    early_stopping_enable: bool = True  # 是否启用早停
    early_stopping_min_delta: float = 1e-6  # 早停最小改善量
    early_stopping_restore_best: bool = True  # 早停后是否恢复最优
    # 学习率调度器增强配置
    lr_scheduler_patience: int = 10  # ReduceLROnPlateau 耐心值
    lr_scheduler_factor: float = 0.5  # ReduceLROnPlateau 衰减因子
    lr_min: float = 1e-7  # 最小学习率

    # 空间加权误差配置（热点区域更关注）
    spatial_weight: SpatialWeightConfig = field(default_factory=SpatialWeightConfig)

    # Checkpoint 配置
    checkpoint_enable: bool = False  # 是否启用 checkpoint 保存
    checkpoint_dir: str = './checkpoints'  # checkpoint 保存目录
    checkpoint_freq: int = 50  # checkpoint 保存频率
    checkpoint_save_best_only: bool = False  # 是否只保存最优的
    checkpoint_max_keep: int = 5  # 最多保留的 checkpoint 数量
    resume_from_checkpoint: Optional[str] = None  # 从哪个 checkpoint 恢复训练

    # 中间掩模快照配置
    snapshot_enable: bool = False  # 是否启用中间掩模快照
    snapshot_dir: str = './snapshots'  # 快照保存目录
    snapshot_freq: int = 20  # 快照保存频率
    snapshot_save_best: bool = True  # 是否保存最优掩模
    snapshot_save_npy: bool = True  # 是否保存 numpy 格式

    # 收敛曲线绘制配置
    plot_enable: bool = False  # 是否启用收敛曲线绘制
    plot_dir: str = './plots'  # 曲线图保存目录
    plot_freq: int = 10  # 曲线更新频率
    plot_log_scale: bool = True  # 是否使用对数坐标
    plot_lr: bool = True  # 是否同时绘制学习率曲线
    plot_live_update: bool = False  # 是否实时更新显示

    # 优化过程动画配置（GIF/MP4）
    animation_enable: bool = False  # 是否启用优化过程动画
    animation_dir: str = './animations'  # 动画保存目录
    animation_freq: int = 1  # 每多少个 epoch 记录一帧
    animation_format: str = 'gif'  # 输出格式: 'gif' 或 'mp4'
    animation_fps: int = 10  # 动画帧率
    animation_dpi: int = 100  # 动画分辨率 (DPI)
    animation_figsize: tuple = None  # 图像尺寸 (width, height)，None 自动推算
    animation_show_info: bool = True  # 是否显示 epoch/loss 标题信息
    animation_show_convergence: bool = True  # 是否显示损失收敛曲线子图
    animation_consistent_error: bool = True  # 误差图是否使用一致色标（便于帧间对比）
    animation_show_wafer: bool = True  # 是否同时显示Wafer图像（光刻胶阈值后）

    # 中间掩模历史记录配置（用于批量评估与Pareto前沿分析）
    save_mask_history: bool = False  # 是否在内存中保存每一步的中间掩模

    # 实验追踪配置
    experiment_tracking_enable: bool = False  # 是否启用实验追踪
    experiment_tracking_backend: str = 'local'  # 追踪后端: 'local', 'mlflow', 'wandb'
    experiment_name: str = 'mask_optimization'  # 实验名称
    run_name: Optional[str] = None  # 运行名称
    experiment_tags: Optional[Dict[str, str]] = None  # 实验标签
    tracking_dir: str = './mlruns'  # 本地追踪目录
    tracking_uri: Optional[str] = None  # MLflow tracking URI
    wandb_project: Optional[str] = None  # WandB 项目名
    wandb_entity: Optional[str] = None  # WandB 实体名
    log_experiment_config: bool = True  # 是否记录配置
    log_metrics_freq: int = 1  # 记录指标的频率

    # 严格电磁仿真后端配置（与标量 Hopkins 并行）
    simulation_backend: str = 'hopkins'  # 'hopkins' (标量) | 'rcwa' (矢量) | 'fdtd' (严格)
    rcwa_config: Optional[RCWAConfig] = None   # RCWA 求解器细粒度参数
    fdtd_config: Optional[FDTDConfig] = None   # FDTD 求解器细粒度参数
    # 评估阶段切换为矢量后端（用于高精度最终评估）
    use_vector_backend_for_evaluation: bool = False

    # 神经网络代理模型（Surrogate Model）配置
    use_surrogate: bool = False  # 是否启用代理模型做快速近似优化
    surrogate_checkpoint_path: Optional[str] = None  # 训练好的 .pt 模型路径
    use_surrogate_for_gradient: bool = True  # 梯度计算也使用代理模型（更快）
    use_real_model_for_final_evaluation: bool = True  # 优化结束后用真实模型验证
    surrogate_device: str = 'auto'  # 代理模型推理设备: 'auto' | 'cpu' | 'cuda' | 'mps'

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'OptimizationConfig':
        if d is None:
            return cls()
        cfg = cls()
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        for key, value in d.items():
            if key in field_names:
                if key == 'loss_weights':
                    cfg.loss_weights = LossWeights.from_dict(value)
                elif key == 'regularization':
                    cfg.regularization = RegularizationConfig.from_dict(value)
                elif key == 'spatial_weight':
                    cfg.spatial_weight = SpatialWeightConfig.from_dict(value)
                elif key == 'process_conditions' and value is not None:
                    cfg.process_conditions = [
                        ProcessCondition(**pc) if isinstance(pc, dict) else pc
                        for pc in value
                    ]
                elif key == 'rcwa_config' and value is not None:
                    cfg.rcwa_config = RCWAConfig(**value) if isinstance(value, dict) else value
                elif key == 'fdtd_config' and value is not None:
                    cfg.fdtd_config = FDTDConfig(**value) if isinstance(value, dict) else value
                else:
                    setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for f in self.__dataclass_fields__.values():
            val = getattr(self, f.name)
            if f.name == 'loss_weights':
                result[f.name] = val.to_dict()
            elif f.name == 'regularization':
                result[f.name] = val.to_dict()
            elif f.name == 'spatial_weight':
                result[f.name] = val.to_dict()
            elif f.name == 'process_conditions' and val is not None:
                result[f.name] = [pc.to_dict() if hasattr(pc, 'to_dict') else pc for pc in val]
            elif f.name in ('rcwa_config', 'fdtd_config') and val is not None:
                d = {}
                for k, v in val.__dict__.items():
                    if isinstance(v, (np.ndarray, complex)):
                        d[k] = str(v)
                    else:
                        d[k] = v
                result[f.name] = d
            else:
                result[f.name] = val
        return result


class LearningRateScheduler:
    """学习率调度器"""

    def __init__(self,
                 initial_lr: float,
                 scheduler_type: str,
                 decay: float = 0.95,
                 step_size: int = 20,
                 min_lr: float = 1e-6):
        """
        初始化学习率调度器

        Args:
            initial_lr: 初始学习率
            scheduler_type: 调度器类型
            decay: 衰减率
            step_size: 步长（用于step调度）
            min_lr: 最小学习率
        """
        self.initial_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.decay = decay
        self.step_size = step_size
        self.min_lr = min_lr
        self.current_lr = initial_lr

    def step(self, epoch: int) -> float:
        """
        更新学习率

        Args:
            epoch: 当前迭代次数

        Returns:
            更新后的学习率
        """
        if self.scheduler_type == 'step':
            # 阶梯衰减
            self.current_lr = self.initial_lr * (self.decay ** (epoch // self.step_size))

        elif self.scheduler_type == 'exponential':
            # 指数衰减
            self.current_lr = self.initial_lr * (self.decay ** epoch)

        elif self.scheduler_type == 'cosine':
            # 余弦退火
            self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                             (1 + np.cos(np.pi * epoch / self.step_size))

        self.current_lr = max(self.current_lr, self.min_lr)
        return self.current_lr


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        """
        初始化早停

        Args:
            patience: 耐心值（连续多少次无改善则停止）
            min_delta: 最小改善量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, loss: float) -> bool:
        """
        检查是否应该早停

        Args:
            loss: 当前损失值

        Returns:
            是否应该停止
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


@dataclass
class MaskOptimizationResult:
    """掩模优化结果"""
    optimized_mask: np.ndarray
    initial_mask: np.ndarray
    target_image: np.ndarray
    final_wafer_image: np.ndarray
    initial_wafer_image: np.ndarray
    final_metrics: MetricsResult
    initial_metrics: MetricsResult
    loss_history: List[float]
    total_iterations: int
    total_time: float
    converged: bool
    message: str
    multi_process_result: Optional[MultiProcessSimulationResult] = None
    per_condition_losses: Optional[List[float]] = None
    process_conditions: Optional[List[ProcessCondition]] = None
    mask_history: Optional[List[np.ndarray]] = None


class MaskOptimizer:
    """
    掩模优化器

    实现完整的掩模图案优化流程。
    支持多工艺条件联合优化：对 focus、dose、NA、sigma 等参数扫描，
    同时约束工艺窗口中心与边界的成像质量。
    """

    def __init__(self,
                 optical_system: Optional[OpticalSystem] = None,
                 config: Optional[OptimizationConfig] = None,
                 callbacks: Optional[List[Callback]] = None):
        """
        初始化掩模优化器

        Args:
            optical_system: 光学系统参数
            config: 优化配置
            callbacks: 自定义回调列表
        """
        self.optical_system = optical_system or OpticalSystem()
        self.config = config or OptimizationConfig()

        self._imaging_model: Optional[PartialCoherentImaging] = None
        self._target_image: Optional[np.ndarray] = None
        self._optimizer: Optional[Union[BaseOptimizer, BaseHeuristicOptimizer]] = None
        self._lr_scheduler: Optional[LearningRateScheduler] = None
        self._early_stopping: Optional[EarlyStopping] = None
        self._spatial_weight_mask: Optional[np.ndarray] = None

        self._multi_imaging_models: Optional[List[PartialCoherentImaging]] = None
        self._multi_conditions: Optional[List[ProcessCondition]] = None
        self._multi_weights: Optional[List[float]] = None

        self._surrogate_imaging: Optional['SurrogateImaging'] = None
        self._surrogate_multi_models: Optional[List['SurrogateImaging']] = None

        self._callbacks: CallbackList = CallbackList(callbacks)
        self._history_callback: Optional[HistoryCallback] = None
        self._trainer_state: Optional[TrainerState] = None

        self._bandlimit_mask: Optional[np.ndarray] = None

    def _setup_bandlimit_mask(self, image_size: tuple):
        """设置频域带限约束掩模"""
        cfg = self.config
        if not cfg.use_frequency_bandlimit:
            self._bandlimit_mask = None
            return

        try:
            custom_mask = cfg.bandlimit_custom_mask
            if (cfg.bandlimit_type.lower() == 'custom'
                and custom_mask is not None):
                if custom_mask.shape != tuple(image_size):
                    logger.warning(
                        f"自定义频域掩模形状 {custom_mask.shape} "
                        f"与图像形状 {image_size} 不匹配，调整大小"
                    )
                    from scipy.ndimage import zoom
                    sy = image_size[0] / custom_mask.shape[0]
                    sx = image_size[1] / custom_mask.shape[1]
                    custom_mask = zoom(custom_mask, (sy, sx), order=1)

            self._bandlimit_mask = create_bandlimit_mask(
                shape=image_size,
                bandlimit_type=cfg.bandlimit_type,
                inner_radius=cfg.bandlimit_inner_radius,
                outer_radius=cfg.bandlimit_outer_radius,
                fx_range=cfg.bandlimit_fx_range,
                fy_range=cfg.bandlimit_fy_range,
                angle_range=cfg.bandlimit_angle_range,
                smooth=cfg.bandlimit_smooth,
                order=cfg.bandlimit_order,
                custom_mask=custom_mask if cfg.bandlimit_type.lower() == 'custom' else None
            )
            logger.info(
                f"频域带限投影已启用: 类型={cfg.bandlimit_type}, "
                f"内半径={cfg.bandlimit_inner_radius}, "
                f"外半径={cfg.bandlimit_outer_radius}"
            )
        except Exception as e:
            logger.warning(f"创建频域带限掩模失败，跳过频域约束: {e}")
            self._bandlimit_mask = None

    def _apply_bandlimit_projection(self, mask: np.ndarray) -> np.ndarray:
        """应用频域带限投影"""
        if self._bandlimit_mask is None:
            return mask
        return bandlimit_projection(
            mask,
            self._bandlimit_mask,
            preserve_dc=self.config.bandlimit_preserve_dc
        )

    def _apply_gradient_bandlimit(self, gradient: np.ndarray) -> np.ndarray:
        """对梯度应用频域带限投影"""
        if (self._bandlimit_mask is None
            or not self.config.bandlimit_project_gradient):
            return gradient
        return bandlimited_gradient_projection(gradient, self._bandlimit_mask)

    def _should_use_surrogate(self, for_evaluation: bool = False, for_gradient: bool = False) -> bool:
        """判断是否应该使用代理模型

        Args:
            for_evaluation: 是否为评估阶段（最终评估优先使用真实模型）
            for_gradient: 是否用于梯度计算

        Returns:
            是否使用代理模型
        """
        cfg = self.config
        if not cfg.use_surrogate:
            return False
        if not _HAS_SURROGATE:
            return False
        if for_evaluation and cfg.use_real_model_for_final_evaluation:
            return False
        if for_gradient and not cfg.use_surrogate_for_gradient:
            return False
        if for_gradient:
            return self._surrogate_imaging is not None or self._surrogate_multi_models is not None
        return self._surrogate_imaging is not None or self._surrogate_multi_models is not None

    def _setup_surrogate_model(self, image_size: tuple):
        """设置神经网络代理模型

        当 config.use_surrogate=True 且指定了 checkpoint_path 时，
        加载训练好的 SurrogateImaging 模型。

        Args:
            image_size: 图像尺寸 (height, width)
        """
        cfg = self.config
        self._surrogate_imaging = None
        self._surrogate_multi_models = None

        if not cfg.use_surrogate:
            return

        if not _HAS_SURROGATE:
            logger.warning("surrogate 模块不可用（请安装 PyTorch），将回退到真实成像模型")
            cfg.use_surrogate = False
            return

        checkpoint_path = cfg.surrogate_checkpoint_path
        if checkpoint_path is None:
            logger.warning("use_surrogate=True 但未指定 surrogate_checkpoint_path，将回退到真实成像模型")
            cfg.use_surrogate = False
            return

        try:
            device = cfg.surrogate_device
            self._surrogate_imaging = SurrogateImaging.from_checkpoint(
                checkpoint_path=checkpoint_path,
                optical_system=self.optical_system,
                device=device,
            )
            self._surrogate_imaging._image_size = tuple(image_size)

            if self._multi_conditions is not None:
                self._surrogate_multi_models = []
                for cond in self._multi_conditions:
                    try:
                        cond_optics = cond.to_optical_system(base_optics=self.optical_system)
                        surr = SurrogateImaging.from_checkpoint(
                            checkpoint_path=checkpoint_path,
                            optical_system=cond_optics,
                            device=device,
                        )
                        surr._image_size = tuple(image_size)
                        self._surrogate_multi_models.append(surr)
                    except Exception as e:
                        logger.warning(f"多工艺条件代理模型初始化失败（cond={cond}），该条件回退到真实模型: {e}")
                        self._surrogate_multi_models.append(None)

            logger.info(
                f"神经网络代理模型加载成功: {checkpoint_path}"
                + (f", 多工艺条件 {len(self._surrogate_multi_models)} 个" if self._surrogate_multi_models else "")
            )
        except Exception as e:
            logger.warning(f"加载代理模型失败，将回退到真实成像模型: {e}")
            cfg.use_surrogate = False
            self._surrogate_imaging = None
            self._surrogate_multi_models = None

    def _surrogate_compute_aerial(self, mask: np.ndarray, for_evaluation: bool = False) -> Optional[np.ndarray]:
        """通过代理模型计算空间像（单工艺条件）

        Args:
            mask: 掩模图案
            for_evaluation: 是否为评估阶段

        Returns:
            空间像，若不使用代理模型返回 None
        """
        if not self._should_use_surrogate(for_evaluation=for_evaluation):
            return None
        if self._surrogate_imaging is None:
            return None
        try:
            return self._surrogate_imaging.compute_aerial_image(mask)
        except Exception as e:
            logger.warning(f"代理模型推理失败，回退到真实模型: {e}")
            return None

    def _surrogate_compute_gradient(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """通过代理模型计算成像梯度 d(aerial)/d(mask)

        Args:
            mask: 掩模图案

        Returns:
            梯度数组，若不使用代理模型返回 None
        """
        if not self._should_use_surrogate(for_gradient=True):
            return None
        if self._surrogate_imaging is None:
            return None
        try:
            return self._surrogate_imaging.compute_image_gradient(mask)
        except Exception as e:
            logger.warning(f"代理模型梯度计算失败，回退到真实模型: {e}")
            return None

    def _multi_surrogate_compute_aerial(self, idx: int, mask: np.ndarray, for_evaluation: bool = False) -> Optional[np.ndarray]:
        """多工艺条件下，通过指定索引的代理模型计算空间像

        Args:
            idx: 工艺条件索引
            mask: 掩模图案
            for_evaluation: 是否为评估阶段

        Returns:
            空间像，若不使用代理模型返回 None
        """
        if not self._should_use_surrogate(for_evaluation=for_evaluation):
            return None
        if (self._surrogate_multi_models is None
                or idx >= len(self._surrogate_multi_models)
                or self._surrogate_multi_models[idx] is None):
            return None
        try:
            return self._surrogate_multi_models[idx].compute_aerial_image(mask)
        except Exception as e:
            logger.warning(f"多工艺条件代理模型[{idx}]推理失败，回退到真实模型: {e}")
            return None

    def _multi_surrogate_compute_gradient(self, idx: int, mask: np.ndarray) -> Optional[np.ndarray]:
        """多工艺条件下，通过指定索引的代理模型计算成像梯度

        Args:
            idx: 工艺条件索引
            mask: 掩模图案

        Returns:
            梯度数组，若不使用代理模型返回 None
        """
        if not self._should_use_surrogate(for_gradient=True):
            return None
        if (self._surrogate_multi_models is None
                or idx >= len(self._surrogate_multi_models)
                or self._surrogate_multi_models[idx] is None):
            return None
        try:
            return self._surrogate_multi_models[idx].compute_image_gradient(mask)
        except Exception as e:
            logger.warning(f"多工艺条件代理模型[{idx}]梯度失败，回退到真实模型: {e}")
            return None

    def _setup_imaging_model(self, image_size: tuple):
        """设置成像模型（标量 Hopkins 模型，始终初始化以用于梯度回退）"""
        self._imaging_model = PartialCoherentImaging(
            self.optical_system, image_size
        )
        self._image_size = tuple(image_size)
        if self.config.verbose:
            logger.info(
                f"仿真后端配置: "
                f"训练={self.config.simulation_backend}, "
                f"评估={'矢量' if self.config.use_vector_backend_for_evaluation else '标量Hopkins'}"
            )

    # ------------------------------------------------------------------
    # 统一仿真后端集成接口
    # ------------------------------------------------------------------
    def _effective_backend(self, for_evaluation: bool = False) -> str:
        """根据阶段（训练/评估）返回实际生效的仿真后端"""
        if for_evaluation and self.config.use_vector_backend_for_evaluation:
            # 评估阶段强制使用更精确的后端（若用户配置了 RCWA/FDTD）
            cfg_backend = str(self.config.simulation_backend).lower()
            if cfg_backend in ("rcwa", "fdtd"):
                return cfg_backend
            return "rcwa"  # 默认评估阶段回退到 RCWA
        return str(self.config.simulation_backend).lower()

    def _simulate_aerial(self, mask: np.ndarray, for_evaluation: bool = False):
        """
        统一空间像仿真入口：根据配置选择 代理模型/Hopkins/RCWA/FDTD 后端。

        优先级：代理模型（仅训练阶段）→ RCWA/FDTD → Hopkins
        """
        if not for_evaluation:
            surrogate_aerial = self._surrogate_compute_aerial(mask, for_evaluation=for_evaluation)
            if surrogate_aerial is not None:
                return surrogate_aerial

        backend = self._effective_backend(for_evaluation=for_evaluation)
        if backend == SimulationBackend.HOPKINS.value:
            return self._imaging_model.compute_aerial_image(mask)

        sim_res = unified_simulate(
            mask=mask,
            backend=backend,
            optical_system=self.optical_system,
            threshold=self.config.threshold,
            apply_resist=False,
            pixel_size_nm=self.optical_system.pixel_size,
            rcwa_config=self.config.rcwa_config,
            fdtd_config=self.config.fdtd_config,
        )
        return sim_res.aerial_image

    def _simulate_wafer(self, mask: np.ndarray, for_evaluation: bool = False):
        """
        统一晶圆图仿真入口（含阈值/光刻胶模型）。
        """
        if not for_evaluation:
            surrogate_aerial = self._surrogate_compute_aerial(mask, for_evaluation=for_evaluation)
            if surrogate_aerial is not None:
                if self.config.use_wafer_image_loss:
                    return _apply_threshold_for_loss(surrogate_aerial, self.config.threshold)
                return surrogate_aerial

        backend = self._effective_backend(for_evaluation=for_evaluation)
        if backend == SimulationBackend.HOPKINS.value:
            aerial = self._imaging_model.compute_aerial_image(mask)
            if self.config.use_wafer_image_loss:
                return _apply_threshold_for_loss(aerial, self.config.threshold)
            return aerial

        sim_res = unified_simulate(
            mask=mask,
            backend=backend,
            optical_system=self.optical_system,
            threshold=self.config.threshold,
            apply_resist=self.config.use_wafer_image_loss,
            pixel_size_nm=self.optical_system.pixel_size,
            rcwa_config=self.config.rcwa_config,
            fdtd_config=self.config.fdtd_config,
        )
        return sim_res.wafer_image if self.config.use_wafer_image_loss else sim_res.aerial_image

    def _setup_optimizer(self):
        """设置优化器"""
        opt_type = self.config.optimizer_type.lower()
        seed = self.config.random_seed

        if opt_type == 'gradient_descent':
            self._optimizer = GradientDescentOptimizer(
                learning_rate=self.config.learning_rate,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                verbose=self.config.verbose
            )
        elif opt_type == 'adam':
            self._optimizer = AdamOptimizer(
                learning_rate=self.config.learning_rate,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                verbose=self.config.verbose
            )
        elif opt_type == 'rmsprop':
            self._optimizer = RMSpropOptimizer(
                learning_rate=self.config.learning_rate,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                verbose=self.config.verbose
            )
        elif opt_type == 'bfgs':
            self._optimizer = BFGSOptimizer(
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                verbose=self.config.verbose
            )
        elif opt_type == 'newton':
            self._optimizer = NewtonOptimizer(
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                verbose=self.config.verbose
            )
        elif opt_type == 'genetic':
            self._optimizer = GeneticAlgorithmOptimizer(
                population_size=self.config.population_size,
                max_iter=self.config.max_iter,
                crossover_rate=self.config.crossover_rate,
                mutation_rate=self.config.mutation_rate,
                verbose=self.config.verbose,
                seed=seed,
                n_jobs=self.config.n_jobs
            )
        elif opt_type == 'pso':
            self._optimizer = ParticleSwarmOptimizer(
                population_size=self.config.population_size,
                max_iter=self.config.max_iter,
                verbose=self.config.verbose,
                seed=seed,
                n_jobs=self.config.n_jobs
            )
        elif opt_type in ('sa', 'simulated_annealing'):
            self._optimizer = SimulatedAnnealingOptimizer(
                max_iter=self.config.max_iter,
                verbose=self.config.verbose,
                seed=seed
            )
        elif opt_type in ('de', 'differential_evolution'):
            self._optimizer = DifferentialEvolutionOptimizer(
                population_size=self.config.population_size,
                max_iter=self.config.max_iter,
                verbose=self.config.verbose,
                seed=seed,
                n_jobs=self.config.n_jobs
            )
        elif opt_type in ('cma_es', 'cmaes'):
            self._optimizer = CMAESOptimizer(
                population_size=self.config.population_size,
                max_iter=self.config.max_iter,
                verbose=self.config.verbose,
                seed=seed
            )
        else:
            raise ValueError(f"未知的优化器类型: {opt_type}")

    def _setup_lr_scheduler(self):
        """设置学习率调度器"""
        if self.config.lr_scheduler:
            self._lr_scheduler = LearningRateScheduler(
                initial_lr=self.config.learning_rate,
                scheduler_type=self.config.lr_scheduler,
                decay=self.config.lr_decay,
                step_size=self.config.lr_step_size
            )

    def _setup_early_stopping(self):
        """设置早停"""
        self._early_stopping = EarlyStopping(
            patience=self.config.early_stop_patience
        )

    def _setup_callbacks(self, initial_mask: np.ndarray):
        """
        设置回调系统

        根据配置创建并注册所有回调。
        """
        cfg = self.config
        callbacks = self._callbacks

        self._trainer_state = TrainerState()
        self._trainer_state.mask = initial_mask.copy()
        self._trainer_state.learning_rate = cfg.learning_rate

        callbacks.set_state(self._trainer_state)
        callbacks.set_params({
            'max_iter': cfg.max_iter,
            'learning_rate': cfg.learning_rate,
            'optimizer_type': cfg.optimizer_type,
        })

        self._history_callback = HistoryCallback(save_masks=cfg.save_mask_history)
        callbacks.append(self._history_callback)

        if cfg.use_callbacks:
            if cfg.verbose:
                callbacks.append(LoggerCallback(
                    log_freq=cfg.callback_log_freq,
                    show_lr=True,
                    show_time=True
                ))

            if cfg.lr_scheduler:
                callbacks.append(LearningRateSchedulerCallback(
                    initial_lr=cfg.learning_rate,
                    scheduler_type=cfg.lr_scheduler,
                    decay=cfg.lr_decay,
                    step_size=cfg.lr_step_size,
                    min_lr=cfg.lr_min,
                    patience=cfg.lr_scheduler_patience,
                    factor=cfg.lr_scheduler_factor,
                    min_delta=cfg.early_stopping_min_delta
                ))

            if cfg.early_stopping_enable and cfg.early_stop_patience > 0:
                callbacks.append(EarlyStoppingCallback(
                    patience=cfg.early_stop_patience,
                    min_delta=cfg.early_stopping_min_delta,
                    monitor='loss',
                    mode='min',
                    restore_best=cfg.early_stopping_restore_best
                ))

            if cfg.checkpoint_enable:
                callbacks.append(ModelCheckpointCallback(
                    checkpoint_dir=cfg.checkpoint_dir,
                    save_freq=cfg.checkpoint_freq,
                    save_best_only=cfg.checkpoint_save_best_only,
                    monitor='loss',
                    mode='min',
                    max_checkpoints=cfg.checkpoint_max_keep,
                    prefix='mask_opt'
                ))

            if cfg.snapshot_enable:
                callbacks.append(MaskSnapshotCallback(
                    save_dir=cfg.snapshot_dir,
                    save_freq=cfg.snapshot_freq,
                    save_best=cfg.snapshot_save_best,
                    save_npy=cfg.snapshot_save_npy
                ))

            if cfg.plot_enable:
                callbacks.append(ConvergencePlotCallback(
                    save_dir=cfg.plot_dir,
                    plot_freq=cfg.plot_freq,
                    log_scale=cfg.plot_log_scale,
                    plot_lr=cfg.plot_lr,
                    live_update=cfg.plot_live_update
                ))

            if cfg.animation_enable:
                imaging_model_for_anim = self._imaging_model
                if (cfg.use_multi_process
                        and self._multi_imaging_models is not None
                        and len(self._multi_imaging_models) > 0):
                    mid_idx = len(self._multi_imaging_models) // 2
                    imaging_model_for_anim = self._multi_imaging_models[mid_idx]

                if imaging_model_for_anim is not None:
                    def compute_aerial_wrapper(mask, model=imaging_model_for_anim):
                        return model.compute_aerial_image(mask)

                    compute_wafer_wrapper = None
                    if cfg.animation_show_wafer:
                        wafer_threshold = cfg.threshold
                        wafer_use_resist = cfg.use_wafer_image_loss
                        dose_for_anim = 1.0
                        if (cfg.use_multi_process
                                and self._multi_conditions is not None
                                and len(self._multi_conditions) > 0):
                            mid_idx2 = len(self._multi_conditions) // 2
                            dose_for_anim = self._multi_conditions[mid_idx2].dose

                        def compute_wafer_wrapper(mask,
                                                  model=imaging_model_for_anim,
                                                  dose=float(dose_for_anim),
                                                  use_resist=wafer_use_resist,
                                                  thresh=wafer_threshold):
                            aerial = model.compute_aerial_image(mask)
                            if dose != 1.0:
                                aerial = np.clip(aerial * dose, 0.0, 1.0)
                            if use_resist:
                                return _apply_threshold_for_loss(aerial, thresh)
                            return aerial

                    callbacks.append(AnimationCallback(
                        save_dir=cfg.animation_dir,
                        save_freq=cfg.animation_freq,
                        output_format=cfg.animation_format,
                        fps=cfg.animation_fps,
                        dpi=cfg.animation_dpi,
                        figsize=tuple(cfg.animation_figsize) if cfg.animation_figsize is not None else None,
                        compute_aerial_fn=compute_aerial_wrapper,
                        compute_wafer_fn=compute_wafer_wrapper,
                        target_image=self._target_image,
                        show_title_info=cfg.animation_show_info,
                        show_convergence=cfg.animation_show_convergence,
                        consistent_error_scale=cfg.animation_consistent_error,
                    ))

            if cfg.experiment_tracking_enable:
                from algorithms.callbacks import ExperimentTrackingCallback
                callbacks.append(ExperimentTrackingCallback(
                    backend=cfg.experiment_tracking_backend,
                    experiment_name=cfg.experiment_name,
                    run_name=cfg.run_name,
                    tags=cfg.experiment_tags,
                    tracking_dir=cfg.tracking_dir,
                    tracking_uri=cfg.tracking_uri,
                    wandb_project=cfg.wandb_project,
                    wandb_entity=cfg.wandb_entity,
                    log_config=cfg.log_experiment_config,
                    log_metrics_freq=cfg.log_metrics_freq,
                ))

    def add_callback(self, callback: Callback):
        """
        添加自定义回调

        Args:
            callback: 回调实例
        """
        self._callbacks.append(callback)

    def _supports_step_training(self) -> bool:
        """
        检查当前优化器是否支持逐步训练

        只有梯度类优化器支持逐步训练和完整的 callback 功能。
        """
        opt_type = self.config.optimizer_type.lower()
        return opt_type in ['gradient_descent', 'adam', 'rmsprop']

    def _step_train(self,
                    initial_mask: np.ndarray,
                    old_callback: Optional[Callable] = None
                    ) -> OptimizationResult:
        """
        逐步训练循环（支持完整 callback 系统）

        适用于 gradient_descent、adam、rmsprop 等梯度优化器。

        Args:
            initial_mask: 初始掩模
            old_callback: 旧版回调函数（兼容接口）

        Returns:
            OptimizationResult 对象
        """
        cfg = self.config
        x = initial_mask.copy()
        shape = x.shape

        self._setup_callbacks(initial_mask)
        state = self._trainer_state

        nfev = 1
        f_val = self._compute_loss(x)
        state.loss = f_val
        state.loss_history.append(f_val)
        state.lr_history.append(state.learning_rate)

        success = False
        message = "达到最大迭代次数"
        nit = 0

        self._callbacks.on_train_begin({'loss': f_val})

        if old_callback is not None:
            old_callback(0, x, f_val)

        opt_type = cfg.optimizer_type.lower()

        if opt_type == 'gradient_descent':
            velocity = np.zeros_like(x.flatten())
        elif opt_type == 'adam':
            m = np.zeros_like(x)
            v = np.zeros_like(x)
        elif opt_type == 'rmsprop':
            eg2 = np.zeros_like(x)
            velocity = np.zeros_like(x)

        for epoch in range(1, cfg.max_iter + 1):
            nit = epoch

            self._callbacks.on_epoch_begin(epoch)

            lr = state.learning_rate

            grad = self._compute_gradient(x)
            nfev += 1

            grad = self._apply_gradient_bandlimit(grad)

            if opt_type == 'gradient_descent':
                x_flat = x.flatten()
                grad_flat = grad.flatten()

                if hasattr(self._optimizer, 'momentum') and self._optimizer.momentum > 0:
                    velocity = self._optimizer.momentum * velocity - lr * grad_flat
                    x_new_flat = x_flat + velocity
                else:
                    x_new_flat = x_flat - lr * grad_flat

                x_new = x_new_flat.reshape(shape)

            elif opt_type == 'adam':
                beta1 = 0.9
                beta2 = 0.999
                epsilon = 1e-8

                if hasattr(self._optimizer, 'beta1'):
                    beta1 = self._optimizer.beta1
                if hasattr(self._optimizer, 'beta2'):
                    beta2 = self._optimizer.beta2
                if hasattr(self._optimizer, 'epsilon'):
                    epsilon = self._optimizer.epsilon

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)

                m_hat = m / (1 - beta1 ** epoch)
                v_hat = v / (1 - beta2 ** epoch)

                x_new = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)

            elif opt_type == 'rmsprop':
                alpha = 0.9
                epsilon = 1e-8
                momentum = 0.0

                if hasattr(self._optimizer, 'alpha'):
                    alpha = self._optimizer.alpha
                if hasattr(self._optimizer, 'epsilon'):
                    epsilon = self._optimizer.epsilon
                if hasattr(self._optimizer, 'momentum'):
                    momentum = self._optimizer.momentum

                eg2 = alpha * eg2 + (1 - alpha) * (grad ** 2)

                if momentum > 0:
                    velocity = momentum * velocity - lr * grad / (np.sqrt(eg2) + epsilon)
                    x_new = x + velocity
                else:
                    x_new = x - lr * grad / (np.sqrt(eg2) + epsilon)

            else:
                x_new = x - lr * grad

            x_new = self._clip_to_bounds(x_new)

            proj_freq = max(1, self.config.bandlimit_projection_freq)
            if epoch % proj_freq == 0:
                x_new = self._apply_bandlimit_projection(x_new)
                x_new = self._clip_to_bounds(x_new)

            f_new = self._compute_loss(x_new)
            nfev += 1

            state.epoch = epoch
            state.loss = f_new
            state.mask = x_new.copy()
            state.loss_history.append(f_new)
            state.lr_history.append(lr)

            logs = {'loss': f_new, 'learning_rate': lr}

            stop = self._callbacks.on_epoch_end(epoch, logs)

            if old_callback is not None:
                old_callback(epoch, x_new, f_new)

            if self._check_convergence(f_val, f_new, x, x_new):
                success = True
                message = f"在第{epoch}次迭代收敛"
                x = x_new
                f_val = f_new
                break

            if stop:
                success = True
                message = f"早停触发，在第{epoch}次迭代停止"
                if state.mask is not None:
                    x = state.mask.copy()
                    f_val = state.loss
                break

            x = x_new
            f_val = f_new

        f_final = state.best_loss if state.best_loss < f_val else f_val
        x_final = state.best_mask if (state.best_mask is not None and state.best_loss < f_val) else x

        self._callbacks.on_train_end({'loss': f_final})

        return OptimizationResult(
            x=x_final,
            fun=f_final,
            nit=nit,
            nfev=nfev,
            success=success,
            message=message,
            history=state.loss_history
        )

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """将值裁剪到边界内"""
        return np.clip(x, self.config.bounds[0], self.config.bounds[1])

    def _check_convergence(self,
                           f_old: float,
                           f_new: float,
                           x_old: np.ndarray,
                           x_new: np.ndarray) -> bool:
        """检查是否收敛"""
        f_change = abs(f_new - f_old) / (abs(f_old) + 1e-10)
        x_change = np.linalg.norm(x_new - x_old) / (np.linalg.norm(x_old) + 1e-10)
        return f_change < self.config.tol or x_change < self.config.tol

    def _load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        从 checkpoint 文件加载状态

        Args:
            filepath: checkpoint 文件路径

        Returns:
            状态字典
        """
        return ModelCheckpointCallback.load_checkpoint(filepath)

    def _build_process_conditions(self) -> Tuple[List[ProcessCondition], List[float]]:
        """
        根据 OptimizationConfig 构建多工艺条件列表和权重

        根据 process_window_mode 选择不同的条件生成策略：
        - 'custom': 使用 config.process_conditions
        - 'focus_dose': 使用 focus_range 和 dose_range 生成二维窗口
        - 'full': 使用全部四个维度范围生成四维窗口
        - None 且 process_conditions 非空: 直接使用 process_conditions

        Returns:
            (conditions, weights) 二元组
        """
        cfg = self.config
        mode = cfg.process_window_mode

        if mode == 'custom' or (mode is None and cfg.process_conditions is not None):
            conditions = cfg.process_conditions
            if conditions is None or len(conditions) == 0:
                raise ValueError("process_window_mode 为 custom 但 process_conditions 为空")
            weights = [c.weight for c in conditions]
            return conditions, weights

        if mode == 'focus_dose':
            conditions = create_focus_dose_window(
                focus_range=cfg.focus_range,
                dose_range=cfg.dose_range,
                na=self.optical_system.na,
                sigma=self.optical_system.sigma,
                wavelength=self.optical_system.wavelength,
                center_weight=cfg.process_center_weight,
                edge_weight=cfg.process_edge_weight
            )
        elif mode == 'full':
            conditions = create_full_process_window(
                focus_values=cfg.focus_range,
                dose_values=cfg.dose_range,
                na_values=cfg.na_range,
                sigma_values=cfg.sigma_range,
                wavelength=self.optical_system.wavelength,
                center_weight_boost=cfg.process_center_weight
            )
        else:
            raise ValueError(
                f"无法确定多工艺条件生成方式: "
                f"process_window_mode={mode}, process_conditions={cfg.process_conditions}"
            )

        weights = [c.weight for c in conditions]
        return conditions, weights

    def _setup_multi_process_models(self, image_size: tuple):
        """
        设置多工艺条件成像模型

        为每个工艺条件创建独立的 PartialCoherentImaging 实例，
        并构建归一化权重。

        Args:
            image_size: 图像尺寸 (height, width)
        """
        conditions, weights = self._build_process_conditions()
        self._multi_conditions = conditions
        self._multi_weights = weights

        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError("多工艺条件权重总和必须大于0")
        self._multi_weights = [w / total_w for w in weights]

        self._multi_imaging_models = []
        for cond in conditions:
            optics = cond.to_optical_system(base_optics=self.optical_system)
            model = PartialCoherentImaging(optics, image_size)
            self._multi_imaging_models.append(model)

        logger.info(
            f"多工艺条件联合优化: {len(conditions)} 个条件, "
            f"权重归一化后范围 [{min(self._multi_weights):.3f}, {max(self._multi_weights):.3f}]"
        )

    def _prepare_image(self, aerial: np.ndarray, dose: float = 1.0) -> np.ndarray:
        """
        对空间像做剂量缩放和（可选）光刻胶阈值平滑处理

        Args:
            aerial: 空间像
            dose: 曝光相对剂量

        Returns:
            处理后的图像，用于损失计算
        """
        if dose != 1.0:
            aerial = np.clip(aerial * dose, 0.0, 1.0)
        if self.config.use_wafer_image_loss:
            return _apply_threshold_for_loss(aerial, self.config.threshold)
        return aerial

    def _compute_regularization_loss(self, mask: np.ndarray) -> float:
        """
        计算正则化项损失

        Args:
            mask: 掩模图案

        Returns:
            正则化损失值
        """
        reg_cfg = self.config.regularization
        if reg_cfg.type is None or reg_cfg.strength <= 0:
            return 0.0

        params = reg_cfg.params
        reg_type = reg_cfg.type.lower()

        if reg_type == 'l1':
            return reg_cfg.strength * l1_regularization(mask)
        elif reg_type == 'l2':
            return reg_cfg.strength * l2_regularization(mask)
        elif reg_type == 'tv':
            return reg_cfg.strength * tv_regularization(mask)
        elif reg_type == 'tv_isotropic':
            return reg_cfg.strength * total_variation_isotropic(mask)
        elif reg_type == 'manhattan':
            return reg_cfg.strength * manhattan_distance_penalty(mask)
        elif reg_type == 'binary_entropy':
            return reg_cfg.strength * binary_entropy_penalty(mask)
        elif reg_type == 'epe':
            epe_threshold = params.get('threshold', self.config.epe_threshold)
            pixel_size = params.get('pixel_size', self.config.pixel_size)
            return reg_cfg.strength * edge_placement_error(
                mask, self._target_image, epe_threshold, pixel_size
            )
        elif reg_type == 'epe_soft':
            sigma = params.get('sigma', self.config.epe_sigma)
            pixel_size = params.get('pixel_size', self.config.pixel_size)
            return reg_cfg.strength * soft_edge_placement_error(
                mask, self._target_image, sigma, pixel_size
            )
        elif reg_type == 'min_feature_morph':
            min_size = params.get('min_size', self.config.min_feature_size)
            return reg_cfg.strength * soft_min_feature_size_morphology(mask, min_size)
        elif reg_type == 'min_feature_freq':
            min_size = params.get('min_size', self.config.min_feature_size)
            pixel_size = params.get('pixel_size', self.config.pixel_size)
            return reg_cfg.strength * min_feature_size_frequency(mask, min_size, pixel_size)
        elif reg_type == 'min_feature_combined':
            min_size = params.get('min_size', self.config.min_feature_size)
            pixel_size = params.get('pixel_size', self.config.pixel_size)
            alpha = params.get('alpha', self.config.min_feature_alpha)
            return reg_cfg.strength * min_feature_size_combined(
                mask, min_size, pixel_size, alpha
            )
        else:
            logger.warning(f"未知的正则化类型: {reg_cfg.type}，跳过")
            return 0.0

    def _compute_regularization_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算正则化项的梯度

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        reg_cfg = self.config.regularization
        if reg_cfg.type is None or reg_cfg.strength <= 0:
            return np.zeros_like(mask)

        params = reg_cfg.params
        reg_type = reg_cfg.type.lower()

        if reg_type == 'l1':
            return reg_cfg.strength * l1_regularization_gradient(mask)
        elif reg_type == 'l2':
            return reg_cfg.strength * l2_regularization_gradient(mask)
        elif reg_type == 'tv':
            return reg_cfg.strength * tv_regularization_gradient(mask)
        elif reg_type == 'tv_isotropic':
            return reg_cfg.strength * total_variation_isotropic_gradient(mask)
        elif reg_type == 'manhattan':
            return reg_cfg.strength * manhattan_distance_penalty_gradient(mask)
        elif reg_type == 'binary_entropy':
            return reg_cfg.strength * binary_entropy_penalty_gradient(mask)
        elif reg_type == 'epe':
            epe_threshold = params.get('threshold', self.config.epe_threshold)
            eps = params.get('eps', 1e-5)
            return reg_cfg.strength * edge_placement_error_gradient(
                mask, self._target_image, epe_threshold, eps
            )
        elif reg_type == 'epe_soft':
            sigma = params.get('sigma', self.config.epe_sigma)
            return reg_cfg.strength * soft_edge_placement_error_gradient(
                mask, self._target_image, sigma
            )
        elif reg_type == 'min_feature_morph':
            min_size = params.get('min_size', self.config.min_feature_size)
            return reg_cfg.strength * soft_min_feature_size_morphology_gradient(mask, min_size)
        elif reg_type == 'min_feature_freq':
            min_size = params.get('min_size', self.config.min_feature_size)
            pixel_size = params.get('pixel_size', self.config.pixel_size)
            return reg_cfg.strength * min_feature_size_frequency_gradient(
                mask, min_size, pixel_size
            )
        elif reg_type == 'min_feature_combined':
            min_size = params.get('min_size', self.config.min_feature_size)
            pixel_size = params.get('pixel_size', self.config.pixel_size)
            alpha = params.get('alpha', self.config.min_feature_alpha)
            return reg_cfg.strength * min_feature_size_combined_gradient(
                mask, min_size, pixel_size, alpha
            )
        else:
            return np.zeros_like(mask)

    def _compute_image_loss_components(self, image: np.ndarray,
                                       target: np.ndarray) -> CompositeLossComponents:
        """
        针对单幅图像计算 MSE、(1-SSIM)、加权MSE/MAE 等逐像素损失分量（不含 PVB/正则化）

        Args:
            image: 处理后的成像结果
            target: 目标图像

        Returns:
            CompositeLossComponents（填充 mse、ssim、weighted_mse、weighted_mae 字段）
        """
        comp = CompositeLossComponents()
        lw = self.config.loss_weights

        if lw.mse > 0:
            comp.mse = lw.mse * mse(image, target)
        if lw.ssim > 0:
            comp.ssim = lw.ssim * (1.0 - ssim(image, target))
        if lw.weighted_mse > 0 and self._spatial_weight_mask is not None:
            comp.weighted_mse = lw.weighted_mse * weighted_mse(
                image, target, self._spatial_weight_mask
            )
        if lw.weighted_mae > 0 and self._spatial_weight_mask is not None:
            comp.weighted_mae = lw.weighted_mae * weighted_mae(
                image, target, self._spatial_weight_mask
            )

        return comp

    def _compute_mask_constraints(self, mask: np.ndarray) -> CompositeLossComponents:
        """
        计算掩模约束项（二值化惩罚、TV平滑、EPE、最小特征尺寸）

        Args:
            mask: 掩模图案

        Returns:
            CompositeLossComponents，填充 binary_penalty、tv_smooth、epe、min_feature 字段
        """
        comp = CompositeLossComponents()
        lw = self.config.loss_weights
        cfg = self.config

        if lw.binary_penalty > 0:
            if cfg.binary_penalty_type == 'entropy':
                comp.binary_penalty = lw.binary_penalty * binary_entropy_penalty(mask)
            else:
                comp.binary_penalty = lw.binary_penalty * manhattan_distance_penalty(mask)

        if lw.tv_smooth > 0:
            comp.tv_smooth = lw.tv_smooth * total_variation_isotropic(mask)

        if lw.epe > 0:
            if cfg.epe_use_soft:
                comp.epe = lw.epe * soft_edge_placement_error(
                    mask, self._target_image, cfg.epe_sigma, cfg.pixel_size
                )
            else:
                comp.epe = lw.epe * edge_placement_error(
                    mask, self._target_image, cfg.epe_threshold, cfg.pixel_size
                )

        if lw.min_feature > 0:
            method = cfg.min_feature_method.lower()
            if method == 'morphology':
                comp.min_feature = lw.min_feature * soft_min_feature_size_morphology(
                    mask, cfg.min_feature_size
                )
            elif method == 'frequency':
                comp.min_feature = lw.min_feature * min_feature_size_frequency(
                    mask, cfg.min_feature_size, cfg.pixel_size
                )
            else:
                comp.min_feature = lw.min_feature * min_feature_size_combined(
                    mask, cfg.min_feature_size, cfg.pixel_size, cfg.min_feature_alpha
                )

        return comp

    def _compute_mask_constraints_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算掩模约束项的梯度

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        grad = np.zeros_like(mask, dtype=np.float64)
        lw = self.config.loss_weights
        cfg = self.config

        if lw.binary_penalty > 0:
            if cfg.binary_penalty_type == 'entropy':
                grad += lw.binary_penalty * binary_entropy_penalty_gradient(mask)
            else:
                grad += lw.binary_penalty * manhattan_distance_penalty_gradient(mask)

        if lw.tv_smooth > 0:
            grad += lw.tv_smooth * total_variation_isotropic_gradient(mask)

        if lw.epe > 0:
            if cfg.epe_use_soft:
                grad += lw.epe * soft_edge_placement_error_gradient(
                    mask, self._target_image, cfg.epe_sigma
                )
            else:
                grad += lw.epe * edge_placement_error_gradient(
                    mask, self._target_image, cfg.epe_threshold
                )

        if lw.min_feature > 0:
            method = cfg.min_feature_method.lower()
            if method == 'morphology':
                grad += lw.min_feature * soft_min_feature_size_morphology_gradient(
                    mask, cfg.min_feature_size
                )
            elif method == 'frequency':
                grad += lw.min_feature * min_feature_size_frequency_gradient(
                    mask, cfg.min_feature_size, cfg.pixel_size
                )
            else:
                grad += lw.min_feature * min_feature_size_combined_gradient(
                    mask, cfg.min_feature_size, cfg.pixel_size, cfg.min_feature_alpha
                )

        return grad

    def _compute_composite_single_condition(self, mask: np.ndarray,
                                            imaging_model: PartialCoherentImaging,
                                            dose: float = 1.0,
                                            for_evaluation: bool = False,
                                            multi_idx: Optional[int] = None) -> Tuple[float, np.ndarray, CompositeLossComponents]:
        """
        计算单工艺条件下的复合损失、成像结果及各分量（不含 PVB 和正则化）

        支持统一仿真后端：代理模型 → RCWA/FDTD → Hopkins

        Args:
            mask: 掩模
            imaging_model: 成像模型（其 optics 作为当前工艺条件的光学参数）
            dose: 曝光剂量
            for_evaluation: 是否为评估阶段（可切换到矢量后端）
            multi_idx: 多工艺条件索引（用于选择对应代理模型）

        Returns:
            (loss_value, processed_image, components)
        """
        aerial = None
        if not for_evaluation:
            if multi_idx is not None:
                aerial = self._multi_surrogate_compute_aerial(multi_idx, mask, for_evaluation=for_evaluation)
            else:
                aerial = self._surrogate_compute_aerial(mask, for_evaluation=for_evaluation)

        if aerial is None:
            backend = self._effective_backend(for_evaluation=for_evaluation)
            if backend == SimulationBackend.HOPKINS.value:
                aerial = imaging_model.compute_aerial_image(mask)
            else:
                sim_res = unified_simulate(
                    mask=mask,
                    backend=backend,
                    optical_system=imaging_model.optics,
                    threshold=self.config.threshold,
                    apply_resist=False,
                    pixel_size_nm=imaging_model.optics.pixel_size,
                    rcwa_config=self.config.rcwa_config,
                    fdtd_config=self.config.fdtd_config,
                )
                aerial = sim_res.aerial_image

        image = self._prepare_image(aerial, dose)
        components = self._compute_image_loss_components(image, self._target_image)
        loss = components.mse + components.ssim + components.weighted_mse + components.weighted_mae
        return loss, image, components

    def _compute_single_condition_loss(self, mask: np.ndarray,
                                       imaging_model: PartialCoherentImaging,
                                       dose: float = 1.0,
                                       for_evaluation: bool = False,
                                       multi_idx: Optional[int] = None) -> float:
        """
        计算单组工艺条件下的损失

        当 config.use_composite_loss=True 时，使用复合损失（MSE/SSIM 加权）；
        否则回退到旧的单一 metric 逻辑。

        Args:
            mask: 掩模图案
            imaging_model: 成像模型（其 optics 作为当前工艺条件的光学参数）
            dose: 曝光相对剂量
            for_evaluation: 是否为评估阶段（可切换到矢量后端）
            multi_idx: 多工艺条件索引（用于选择对应代理模型）

        Returns:
            损失值
        """
        if self.config.use_composite_loss:
            loss, _, _ = self._compute_composite_single_condition(
                mask, imaging_model, dose, for_evaluation=for_evaluation, multi_idx=multi_idx
            )
            return loss

        aerial = None
        if not for_evaluation:
            if multi_idx is not None:
                aerial = self._multi_surrogate_compute_aerial(multi_idx, mask, for_evaluation=for_evaluation)
            else:
                aerial = self._surrogate_compute_aerial(mask, for_evaluation=for_evaluation)

        if aerial is None:
            backend = self._effective_backend(for_evaluation=for_evaluation)
            if backend == SimulationBackend.HOPKINS.value:
                aerial = imaging_model.compute_aerial_image(mask)
            else:
                sim_res = unified_simulate(
                    mask=mask,
                    backend=backend,
                    optical_system=imaging_model.optics,
                    threshold=self.config.threshold,
                    apply_resist=False,
                    pixel_size_nm=imaging_model.optics.pixel_size,
                    rcwa_config=self.config.rcwa_config,
                    fdtd_config=self.config.fdtd_config,
                )
                aerial = sim_res.aerial_image

        if self.config.use_wafer_image_loss:
            if dose != 1.0:
                aerial = np.clip(aerial * dose, 0.0, 1.0)
            image = _apply_threshold_for_loss(aerial, self.config.threshold)
        else:
            if dose != 1.0:
                aerial = np.clip(aerial * dose, 0.0, 1.0)
            image = aerial

        metric = self.config.metric.lower()
        if metric == 'mse':
            return mse(image, self._target_image)
        elif metric == 'mae':
            return mae(image, self._target_image)
        elif metric == 'ssim':
            return 1.0 - ssim(image, self._target_image)
        else:
            raise ValueError(f"未知的评估指标: {metric}")

    def _compute_multi_process_loss(self, mask: np.ndarray) -> float:
        """
        计算多工艺条件加权损失

        当 use_composite_loss=True 时，复合损失形式：
        L = Σ_i w_i * [w_mse*MSE_i + w_ssim*(1-SSIM_i)]
            + w_pvb * PVB({I_i})
            + w_mask_complexity * TV(mask)
            + R(mask)
            + λ_robust * Var(L_i)

        否则使用旧的单一 metric 逻辑。

        Args:
            mask: 掩模图案

        Returns:
            加权总损失
        """
        cfg = self.config

        if cfg.use_composite_loss:
            lw = cfg.loss_weights
            per_images = []
            per_losses = []
            total_img_loss = 0.0

            for idx, (model, cond, w) in enumerate(zip(
                self._multi_imaging_models,
                self._multi_conditions,
                self._multi_weights
            )):
                loss_i, img_i, _ = self._compute_composite_single_condition(
                    mask, model, cond.dose, multi_idx=idx
                )
                per_images.append(img_i)
                per_losses.append(loss_i)
                total_img_loss += w * loss_i

            self._last_per_condition_losses = per_losses

            total_loss = total_img_loss

            if lw.pvb > 0:
                pvb_val = pvb(per_images)
                total_loss += lw.pvb * pvb_val

            if lw.mask_complexity > 0:
                total_loss += lw.mask_complexity * total_variation(mask)

            mask_constraints = self._compute_mask_constraints(mask)
            total_loss += (mask_constraints.binary_penalty +
                           mask_constraints.tv_smooth +
                           mask_constraints.epe +
                           mask_constraints.min_feature)

            total_loss += self._compute_regularization_loss(mask)

            if cfg.robustness_loss_weight > 0 and len(per_losses) > 1:
                loss_arr = np.array(per_losses)
                robustness = float(np.var(loss_arr))
                total_loss += cfg.robustness_loss_weight * robustness

            return total_loss

        per_losses = []
        for idx, (model, cond, w) in enumerate(zip(
            self._multi_imaging_models,
            self._multi_conditions,
            self._multi_weights
        )):
            loss_i = self._compute_single_condition_loss(
                mask, model, cond.dose, multi_idx=idx
            )
            per_losses.append(loss_i)

        self._last_per_condition_losses = per_losses

        total_loss = 0.0
        for w, loss_i in zip(self._multi_weights, per_losses):
            total_loss += w * loss_i

        if self.config.robustness_loss_weight > 0 and len(per_losses) > 1:
            loss_arr = np.array(per_losses)
            robustness = float(np.var(loss_arr))
            total_loss += self.config.robustness_loss_weight * robustness

        return total_loss

    def _compute_multi_process_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算多工艺条件加权损失的梯度

        当 use_composite_loss=True 时：
        dL/dmask = Σ_i w_i * d[w_mse*MSE_i + w_ssim*(1-SSIM_i)]/dmask
                 + w_pvb * dPVB/dmask
                 + w_mask_complexity * dTV(mask)/dmask
                 + dR(mask)/dmask
                 + λ_robust * dVar(L_i)/dmask

        对于包含 PVB 或使用矢量仿真后端 (RCWA/FDTD) 的情况，退化为数值梯度。
        支持代理模型（Surrogate Model）快速计算 aerial 和 gradient。

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        cfg = self.config
        train_backend = self._effective_backend(for_evaluation=False)

        if train_backend != SimulationBackend.HOPKINS.value:
            return self._numerical_gradient(mask)

        if cfg.use_composite_loss:
            lw = cfg.loss_weights
            if lw.pvb > 0:
                return self._numerical_gradient(mask)

            gradient = np.zeros_like(mask)
            per_images = []
            per_losses = []

            for idx, (model, cond, w) in enumerate(zip(
                self._multi_imaging_models,
                self._multi_conditions,
                self._multi_weights
            )):
                aerial = self._multi_surrogate_compute_aerial(idx, mask)
                if aerial is None:
                    aerial = model.compute_aerial_image(mask)
                image = self._prepare_image(aerial, cond.dose)
                per_images.append(image)

                error_grad = np.zeros_like(image)

                if lw.mse > 0:
                    error_grad += lw.mse * (2.0 * (image - self._target_image) / mask.size)

                if lw.ssim > 0:
                    error_grad += lw.ssim * ssim_loss_gradient(image, self._target_image)

                if lw.weighted_mse > 0 and self._spatial_weight_mask is not None:
                    error_grad += lw.weighted_mse * weighted_mse_gradient(
                        image, self._target_image, self._spatial_weight_mask
                    )

                if lw.weighted_mae > 0 and self._spatial_weight_mask is not None:
                    error_grad += lw.weighted_mae * weighted_mae_gradient(
                        image, self._target_image, self._spatial_weight_mask
                    )

                imaging_grad = self._multi_surrogate_compute_gradient(idx, mask)
                if imaging_grad is None:
                    imaging_grad = model.compute_image_gradient(mask)

                if cond.dose != 1.0:
                    error_grad = error_grad * cond.dose

                if cfg.use_wafer_image_loss:
                    aerial_dosed = aerial if cond.dose == 1.0 else np.clip(aerial * cond.dose, 0.0, 1.0)
                    threshold_grad = (aerial_dosed >= cfg.threshold).astype(np.float64)
                    error_grad = error_grad * threshold_grad

                gradient += w * (error_grad * imaging_grad)

                loss_i, _, _ = self._compute_composite_single_condition(
                    mask, model, cond.dose, multi_idx=idx
                )
                per_losses.append(loss_i)

            if lw.mask_complexity > 0:
                gradient += lw.mask_complexity * total_variation_gradient(mask)

            gradient += self._compute_mask_constraints_gradient(mask)

            gradient += self._compute_regularization_gradient(mask)

            if cfg.robustness_loss_weight > 0 and len(per_losses) > 1:
                loss_arr = np.array(per_losses)
                mean_loss = np.mean(loss_arr)
                n = len(per_losses)
                for idx, (model, cond, w) in enumerate(zip(
                    self._multi_imaging_models,
                    self._multi_conditions,
                    self._multi_weights
                )):
                    factor = 2.0 * cfg.robustness_loss_weight * (per_losses[idx] - mean_loss) / (n * n)
                    if abs(factor) < 1e-12:
                        continue
                    aerial = self._multi_surrogate_compute_aerial(idx, mask)
                    if aerial is None:
                        aerial = model.compute_aerial_image(mask)
                    image = self._prepare_image(aerial, cond.dose)
                    error_grad = np.zeros_like(image)
                    if lw.mse > 0:
                        error_grad += lw.mse * (2.0 * (image - self._target_image) / mask.size)
                    if lw.ssim > 0:
                        error_grad += lw.ssim * ssim_loss_gradient(image, self._target_image)
                    if lw.weighted_mse > 0 and self._spatial_weight_mask is not None:
                        error_grad += lw.weighted_mse * weighted_mse_gradient(
                            image, self._target_image, self._spatial_weight_mask
                        )
                    if lw.weighted_mae > 0 and self._spatial_weight_mask is not None:
                        error_grad += lw.weighted_mae * weighted_mae_gradient(
                            image, self._target_image, self._spatial_weight_mask
                        )
                    imaging_grad = self._multi_surrogate_compute_gradient(idx, mask)
                    if imaging_grad is None:
                        imaging_grad = model.compute_image_gradient(mask)
                    if cond.dose != 1.0:
                        error_grad = error_grad * cond.dose
                    gradient += factor * (error_grad * imaging_grad)

            return gradient

        metric = self.config.metric.lower()
        gradient = np.zeros_like(mask)
        per_losses = []

        for idx, (model, cond, w) in enumerate(zip(
            self._multi_imaging_models,
            self._multi_conditions,
            self._multi_weights
        )):
            aerial = self._multi_surrogate_compute_aerial(idx, mask)
            if aerial is None:
                aerial = model.compute_aerial_image(mask)

            if cond.dose != 1.0:
                aerial_dosed = np.clip(aerial * cond.dose, 0.0, 1.0)
            else:
                aerial_dosed = aerial

            if self.config.use_wafer_image_loss:
                image = _apply_threshold_for_loss(aerial_dosed, self.config.threshold)
            else:
                image = aerial_dosed

            if metric == 'mse':
                error_grad = 2 * (image - self._target_image) / mask.size
            elif metric == 'mae':
                error_grad = np.sign(image - self._target_image) / mask.size
            elif metric == 'ssim':
                error_grad = ssim_loss_gradient(image, self._target_image)
            else:
                return self._numerical_gradient(mask)

            imaging_grad = self._multi_surrogate_compute_gradient(idx, mask)
            if imaging_grad is None:
                imaging_grad = model.compute_image_gradient(mask)

            if cond.dose != 1.0:
                error_grad = error_grad * cond.dose

            if self.config.use_wafer_image_loss:
                threshold_grad = (aerial_dosed >= self.config.threshold).astype(np.float64)
                error_grad = error_grad * threshold_grad

            gradient += w * (error_grad * imaging_grad)

            loss_i = self._compute_single_condition_loss(
                mask, model, cond.dose, multi_idx=idx
            )
            per_losses.append(loss_i)

        if self.config.robustness_loss_weight > 0 and len(per_losses) > 1:
            loss_arr = np.array(per_losses)
            mean_loss = np.mean(loss_arr)
            n = len(per_losses)
            for idx, (model, cond, w) in enumerate(zip(
                self._multi_imaging_models,
                self._multi_conditions,
                self._multi_weights
            )):
                factor = 2.0 * self.config.robustness_loss_weight * (per_losses[idx] - mean_loss) / (n * n)
                if abs(factor) < 1e-12:
                    continue
                aerial = self._multi_surrogate_compute_aerial(idx, mask)
                if aerial is None:
                    aerial = model.compute_aerial_image(mask)
                if cond.dose != 1.0:
                    aerial_dosed = np.clip(aerial * cond.dose, 0.0, 1.0)
                else:
                    aerial_dosed = aerial
                if metric == 'mse':
                    error_grad = 2 * (aerial_dosed - self._target_image) / mask.size
                elif metric == 'mae':
                    error_grad = np.sign(aerial_dosed - self._target_image) / mask.size
                elif metric == 'ssim':
                    error_grad = ssim_loss_gradient(aerial_dosed, self._target_image)
                imaging_grad = self._multi_surrogate_compute_gradient(idx, mask)
                if imaging_grad is None:
                    imaging_grad = model.compute_image_gradient(mask)
                if cond.dose != 1.0:
                    error_grad = error_grad * cond.dose
                gradient += factor * (error_grad * imaging_grad)

        return gradient

    def _compute_loss(self, mask: np.ndarray) -> float:
        """
        计算损失函数

        当启用多工艺条件联合优化时，自动调度到
        _compute_multi_process_loss；否则使用标称条件。

        当 use_composite_loss=True 时，使用复合损失：
        L = w_mse*MSE + w_ssim*(1-SSIM) + w_mask_complexity*TV(mask) + R(mask)

        Args:
            mask: 掩模图案

        Returns:
            损失值
        """
        if self.config.use_multi_process and self._multi_imaging_models is not None:
            return self._compute_multi_process_loss(mask)

        if self.config.use_composite_loss:
            lw = self.config.loss_weights
            loss, _, _ = self._compute_composite_single_condition(
                mask, self._imaging_model, dose=1.0
            )
            if lw.mask_complexity > 0:
                loss += lw.mask_complexity * total_variation(mask)

            mask_constraints = self._compute_mask_constraints(mask)
            loss += (mask_constraints.binary_penalty +
                     mask_constraints.tv_smooth +
                     mask_constraints.epe +
                     mask_constraints.min_feature)

            loss += self._compute_regularization_loss(mask)
            return loss

        wafer_image = self._surrogate_compute_aerial(mask)
        if wafer_image is None:
            wafer_image = self._imaging_model.compute_aerial_image(mask)

        if self.config.use_wafer_image_loss:
            wafer_image = _apply_threshold_for_loss(wafer_image, self.config.threshold)

        metric = self.config.metric.lower()
        if metric == 'mse':
            return mse(wafer_image, self._target_image)
        elif metric == 'mae':
            return mae(wafer_image, self._target_image)
        elif metric == 'ssim':
            return 1.0 - ssim(wafer_image, self._target_image)
        else:
            raise ValueError(f"未知的评估指标: {metric}")

    def _compute_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算损失函数对掩模的梯度

        当启用多工艺条件联合优化时，自动调度到
        _compute_multi_process_gradient。

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        if self.config.use_multi_process and self._multi_imaging_models is not None:
            return self._compute_multi_process_gradient(mask)

        cfg = self.config

        if cfg.use_composite_loss:
            lw = cfg.loss_weights

            aerial = self._surrogate_compute_aerial(mask)
            if aerial is None:
                aerial = self._imaging_model.compute_aerial_image(mask)
            image = self._prepare_image(aerial, dose=1.0)

            imaging_grad = self._surrogate_compute_gradient(mask)
            if imaging_grad is None:
                imaging_grad = self._imaging_model.compute_image_gradient(mask)

            error_grad = np.zeros_like(image)

            if lw.mse > 0:
                error_grad += lw.mse * (2.0 * (image - self._target_image) / mask.size)

            if lw.ssim > 0:
                error_grad += lw.ssim * ssim_loss_gradient(image, self._target_image)

            if lw.weighted_mse > 0 and self._spatial_weight_mask is not None:
                error_grad += lw.weighted_mse * weighted_mse_gradient(
                    image, self._target_image, self._spatial_weight_mask
                )

            if lw.weighted_mae > 0 and self._spatial_weight_mask is not None:
                error_grad += lw.weighted_mae * weighted_mae_gradient(
                    image, self._target_image, self._spatial_weight_mask
                )

            if cfg.use_wafer_image_loss:
                threshold_grad = (aerial >= cfg.threshold).astype(np.float64)
                error_grad = error_grad * threshold_grad

            gradient = error_grad * imaging_grad

            if lw.mask_complexity > 0:
                gradient += lw.mask_complexity * total_variation_gradient(mask)

            gradient += self._compute_mask_constraints_gradient(mask)

            gradient += self._compute_regularization_gradient(mask)

            return gradient

        wafer_image = self._surrogate_compute_aerial(mask)
        if wafer_image is None:
            wafer_image = self._imaging_model.compute_aerial_image(mask)

        if self.config.use_wafer_image_loss:
            wafer_image_for_grad = _apply_threshold_for_loss(wafer_image, self.config.threshold)
        else:
            wafer_image_for_grad = wafer_image

        if self.config.metric.lower() == 'mse':
            error_grad = 2 * (wafer_image_for_grad - self._target_image) / mask.size
        elif self.config.metric.lower() == 'mae':
            error_grad = np.sign(wafer_image_for_grad - self._target_image) / mask.size
        elif self.config.metric.lower() == 'ssim':
            error_grad = ssim_loss_gradient(wafer_image_for_grad, self._target_image)
        else:
            return self._numerical_gradient(mask)

        imaging_grad = self._surrogate_compute_gradient(mask)
        if imaging_grad is None:
            imaging_grad = self._imaging_model.compute_image_gradient(mask)

        if self.config.use_wafer_image_loss:
            threshold_grad = (wafer_image >= self.config.threshold).astype(np.float64)
            error_grad = error_grad * threshold_grad

        gradient = error_grad * imaging_grad

        return gradient

    def _numerical_gradient(self, mask: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """数值梯度计算"""
        gradient = np.zeros_like(mask)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask_plus = mask.copy()
                mask_plus[i, j] += eps
                mask_minus = mask.copy()
                mask_minus[i, j] -= eps

                gradient[i, j] = (self._compute_loss(mask_plus) -
                                 self._compute_loss(mask_minus)) / (2 * eps)

        return gradient

    def _make_callback_driven_objective(self,
                                     original_obj: Callable[[np.ndarray], float],
                                     original_mask_shape: Tuple[int, int],
                                     base_epoch_interval: int = 1):
        """
        为非 step-training 优化器包装目标函数，用于驱动 callback 系统。

        Args:
            original_obj: 原始目标函数 f(mask_2d) -> float
            original_mask_shape: 原始掩模 2D 形状
            base_epoch_interval: 每多少次函数评估对应一个逻辑 epoch（对于启发式算法，因为每代调用次数很多。）

        Returns:
            (wrapped_obj, initial_epoch_counter)  —— wrapped_obj(x_flat_or_2d) -> float
        """
        cfg = self.config
        callbacks = self._callbacks
        state = self._trainer_state

        interval = max(1, int(base_epoch_interval))
        counter = {'eval_count': 0, 'logical_epoch': 0}

        def wrapped(x: np.ndarray) -> float:
            if x.ndim == 1:
                mask_2d = x.reshape(original_mask_shape)
            else:
                mask_2d = x
            loss = float(original_obj(mask_2d))

            counter['eval_count'] += 1

            if state is not None:
                state.mask = mask_2d.copy()
                state.loss = loss
                if not state.loss_history or state.loss_history[-1] != loss:
                    state.loss_history.append(loss)
                if not state.lr_history:
                    state.lr_history.append(cfg.learning_rate)
                if state.best_mask is None or loss < state.best_loss:
                    state.best_loss = loss
                    state.best_mask = mask_2d.copy()

            new_logical = counter['eval_count'] // interval
            if new_logical > counter['logical_epoch'] and state is not None:
                counter['logical_epoch'] = new_logical
                epoch = counter['logical_epoch']
                state.epoch = epoch
                logs = {'loss': loss, 'learning_rate': state.learning_rate}
                try:
                    callbacks.on_epoch_begin(epoch)
                except Exception:
                    pass
                try:
                    callbacks.on_epoch_end(epoch, logs)
                except Exception:
                    pass

            return loss

        return wrapped

    def optimize(self,
                 initial_mask: np.ndarray,
                 target_image: np.ndarray,
                 callback: Optional[Callable[[int, np.ndarray, float], None]] = None
                 ) -> MaskOptimizationResult:
        """
        执行掩模优化

        当 config.use_multi_process=True 时，自动构建多工艺条件
        成像模型并进行联合优化，同时约束工艺窗口中心与边界。

        支持 callback 系统：LearningRateScheduler、EarlyStopping、
        Checkpoint、MaskSnapshot、ConvergencePlot 等。

        Args:
            initial_mask: 初始掩模图案
            target_image: 目标图像
            callback: 回调函数 callback(iteration, current_mask, current_loss)

        Returns:
            MaskOptimizationResult对象
        """
        start_time = time.time()

        self._target_image = target_image.astype(np.float64)
        self._spatial_weight_mask = generate_spatial_weight_mask(
            self._target_image, self.config.spatial_weight
        )

        self._setup_imaging_model(initial_mask.shape)
        self._setup_bandlimit_mask(initial_mask.shape)

        use_step_training = self._supports_step_training() and self.config.use_callbacks

        if not use_step_training:
            self._setup_optimizer()
            self._setup_lr_scheduler()
            self._setup_early_stopping()

        initial_wafer = self._imaging_model.compute_aerial_image(initial_mask)
        initial_metrics = evaluate_all(initial_wafer, target_image)

        starting_mask = initial_mask.copy()
        start_epoch = 0

        if self.config.resume_from_checkpoint:
            try:
                ckpt_data = self._load_checkpoint(self.config.resume_from_checkpoint)
                if 'mask' in ckpt_data and ckpt_data['mask'] is not None:
                    starting_mask = ckpt_data['mask']
                    if starting_mask.shape != initial_mask.shape:
                        logger.warning(
                            f"Checkpoint 掩模形状 {starting_mask.shape} "
                            f"与初始掩模形状 {initial_mask.shape} 不匹配，使用初始掩模"
                        )
                        starting_mask = initial_mask.copy()
                    else:
                        start_epoch = int(ckpt_data.get('epoch', 0))
                        logger.info(
                            f"从 checkpoint 恢复训练: epoch={start_epoch}, "
                            f"loss={ckpt_data.get('loss', 'N/A')}"
                        )

                        if self._trainer_state is None:
                            self._trainer_state = TrainerState()

                        if 'loss_history' in ckpt_data:
                            self._trainer_state.loss_history = list(ckpt_data['loss_history'])
                        if 'lr_history' in ckpt_data:
                            self._trainer_state.lr_history = list(ckpt_data['lr_history'])
                        if 'best_loss' in ckpt_data:
                            self._trainer_state.best_loss = float(ckpt_data['best_loss'])
                        if 'best_mask' in ckpt_data:
                            self._trainer_state.best_mask = ckpt_data['best_mask']
            except Exception as e:
                logger.warning(f"加载 checkpoint 失败，从初始状态开始: {e}")
                starting_mask = initial_mask.copy()
                start_epoch = 0

        needs_callback_driver = (not use_step_training) and self.config.use_callbacks and (
            self.config.animation_enable
            or self.config.snapshot_enable
            or self.config.plot_enable
            or self.config.checkpoint_enable
            or self.config.early_stopping_enable
            or self.config.save_mask_history
        )

        if needs_callback_driver:
            self._setup_callbacks(starting_mask)
            initial_loss = self._compute_loss(starting_mask)
            self._trainer_state.loss = initial_loss
            if not self._trainer_state.loss_history:
                self._trainer_state.loss_history = [initial_loss]
            if not self._trainer_state.lr_history:
                self._trainer_state.lr_history = [self.config.learning_rate]
            if self._trainer_state.best_mask is None or initial_loss < self._trainer_state.best_loss:
                self._trainer_state.best_loss = initial_loss
                self._trainer_state.best_mask = starting_mask.copy()
            self._callbacks.on_train_begin({'loss': initial_loss})

        if self.config.use_multi_process:
            self._setup_multi_process_models(initial_mask.shape)
        else:
            self._multi_imaging_models = None
            self._multi_conditions = None
            self._multi_weights = None

        self._setup_surrogate_model(initial_mask.shape)

        if self.config.use_multi_process:
            logger.info(f"开始多工艺条件联合掩模优化，{len(self._multi_conditions)} 个工艺条件，"
                       f"初始MSE: {initial_metrics.mse:.6e}")
        else:
            logger.info(f"开始掩模优化，初始MSE: {initial_metrics.mse:.6e}")

        if use_step_training:
            result = self._step_train(starting_mask, old_callback=callback)
        elif isinstance(self._optimizer, BaseHeuristicOptimizer):
            interval = max(1, self.config.population_size)

            if self._bandlimit_mask is not None:
                def original_obj(x):
                    x_proj = self._apply_bandlimit_projection(
                        x.reshape(starting_mask.shape)
                    )
                    x_proj = self._clip_to_bounds(x_proj)
                    return self._compute_loss(x_proj)
                starting_mask = self._apply_bandlimit_projection(starting_mask)
                starting_mask = self._clip_to_bounds(starting_mask)

                if needs_callback_driver:
                    wrapped_obj = self._make_callback_driven_objective(
                        original_obj, starting_mask.shape, interval
                    )
                else:
                    wrapped_obj = original_obj

                result = self._optimizer.optimize(
                    objective=wrapped_obj,
                    x0=starting_mask,
                    bounds=self.config.bounds
                )
            else:
                original_obj = self._compute_loss
                if needs_callback_driver:
                    wrapped_obj = self._make_callback_driven_objective(
                        original_obj, starting_mask.shape, interval
                    )
                else:
                    wrapped_obj = original_obj

                result = self._optimizer.optimize(
                    objective=wrapped_obj,
                    x0=starting_mask,
                    bounds=self.config.bounds
                )
        else:
            if self._bandlimit_mask is not None:
                def original_obj(x):
                    x_proj = self._apply_bandlimit_projection(
                        x.reshape(starting_mask.shape)
                    )
                    x_proj = self._clip_to_bounds(x_proj)
                    return self._compute_loss(x_proj)
                def wrapped_grad_base(x):
                    x_proj = self._apply_bandlimit_projection(
                        x.reshape(starting_mask.shape)
                    )
                    x_proj = self._clip_to_bounds(x_proj)
                    g = self._compute_gradient(x_proj)
                    g = self._apply_gradient_bandlimit(g)
                    return g.flatten()
                starting_mask = self._apply_bandlimit_projection(starting_mask)
                starting_mask = self._clip_to_bounds(starting_mask)

                if needs_callback_driver:
                    wrapped_obj = self._make_callback_driven_objective(
                        original_obj, starting_mask.shape, 1
                    )
                else:
                    wrapped_obj = original_obj

                result = self._optimizer.optimize(
                    objective=wrapped_obj,
                    x0=starting_mask,
                    gradient=wrapped_grad_base,
                    bounds=self.config.bounds
                )
            else:
                original_obj = self._compute_loss
                def grad_base(x):
                    if x.ndim == 1:
                        x = x.reshape(starting_mask.shape)
                    return self._compute_gradient(x).flatten()

                if needs_callback_driver:
                    wrapped_obj = self._make_callback_driven_objective(
                        original_obj, starting_mask.shape, 1
                    )
                else:
                    wrapped_obj = original_obj

                result = self._optimizer.optimize(
                    objective=wrapped_obj,
                    x0=starting_mask,
                    gradient=grad_base,
                    bounds=self.config.bounds
                )

        if needs_callback_driver:
            final_loss = float(getattr(result, 'fun', float('inf')))
            final_mask = getattr(result, 'x', None)
            if final_mask is not None and final_mask.ndim == 1:
                try:
                    final_mask = final_mask.reshape(starting_mask.shape)
                except Exception:
                    pass
            self._callbacks.on_train_end({'loss': final_loss})

        optimized_mask = result.x
        if self._bandlimit_mask is not None:
            optimized_mask = self._apply_bandlimit_projection(optimized_mask)
            optimized_mask = self._clip_to_bounds(optimized_mask)
        final_wafer = self._imaging_model.compute_aerial_image(optimized_mask)
        final_metrics = evaluate_all(final_wafer, target_image)

        total_time = time.time() - start_time

        multi_process_result = None
        per_condition_losses = None
        process_conditions = None

        if self.config.use_multi_process and self._multi_conditions is not None:
            multi_process_result = simulate_multi_process(
                optimized_mask,
                self._multi_conditions,
                base_optics=self.optical_system,
                threshold=self.config.threshold,
                apply_resist=self.config.use_wafer_image_loss
            )
            per_condition_losses = getattr(self, '_last_per_condition_losses', None)
            process_conditions = self._multi_conditions

        mask_history = None
        if (self._history_callback is not None
                and self._history_callback.save_masks
                and self._history_callback.mask_history):
            mask_history = self._history_callback.mask_history
        elif (self._trainer_state is not None
              and self._trainer_state.mask_history):
            mask_history = self._trainer_state.mask_history

        logger.info(f"优化完成，最终MSE: {final_metrics.mse:.6e}，"
                   f"耗时: {total_time:.2f}秒")

        return MaskOptimizationResult(
            optimized_mask=optimized_mask,
            initial_mask=initial_mask,
            target_image=target_image,
            final_wafer_image=final_wafer,
            initial_wafer_image=initial_wafer,
            final_metrics=final_metrics,
            initial_metrics=initial_metrics,
            loss_history=result.history,
            total_iterations=result.nit,
            total_time=total_time,
            converged=result.success,
            message=result.message,
            multi_process_result=multi_process_result,
            per_condition_losses=per_condition_losses,
            process_conditions=process_conditions,
            mask_history=mask_history
        )

    def _optimize_pyramid(self,
                          initial_mask: np.ndarray,
                          target_image: np.ndarray,
                          callback: Optional[Callable[[int, np.ndarray, float], None]] = None
                          ) -> MaskOptimizationResult:
        """
        金字塔多尺度优化

        先在低分辨率粗优化，逐步上采样到高分辨率细化。

        Args:
            initial_mask: 初始掩模
            target_image: 目标图像
            callback: 回调函数

        Returns:
            MaskOptimizationResult
        """
        start_time = time.time()
        cfg = self.config
        scales = build_pyramid_scales(
            initial_mask.shape,
            min_size=cfg.pyramid_min_size,
            n_scales=cfg.pyramid_scales
        )
        logger.info(
            f"金字塔多尺度优化: {len(scales)} 级, "
            f"尺寸序列 {scales}"
        )
        total_max_iter = cfg.max_iter
        coarse_ratio = cfg.pyramid_iter_ratio
        n_coarse_levels = len(scales) - 1
        if n_coarse_levels > 0:
            coarse_iters = max(int(total_max_iter * coarse_ratio) // n_coarse_levels, 5)
            fine_iters = total_max_iter - coarse_iters * n_coarse_levels
        else:
            coarse_iters = 0
            fine_iters = total_max_iter

        current_mask = initial_mask.astype(np.float64)
        self._target_image = target_image.astype(np.float64)
        loss_history = []

        for level_idx, scale_shape in enumerate(scales):
            is_last = (level_idx == len(scales) - 1)
            if not is_last:
                current_mask = downsample_mask(current_mask, 2)
                if current_mask.shape[0] < scale_shape[0] or current_mask.shape[1] < scale_shape[1]:
                    current_mask = upsample_mask(current_mask, scale_shape)
                elif current_mask.shape != scale_shape:
                    current_mask = upsample_mask(current_mask, scale_shape)
                target_at_scale = downsample_mask(self._target_image, 2)
                if target_at_scale.shape[0] < scale_shape[0] or target_at_scale.shape[1] < scale_shape[1]:
                    target_at_scale = upsample_mask(target_at_scale, scale_shape)
                elif target_at_scale.shape != scale_shape:
                    target_at_scale = upsample_mask(target_at_scale, scale_shape)
                n_iter = coarse_iters
            else:
                target_at_scale = self._target_image.copy()
                n_iter = fine_iters

            level_config = OptimizationConfig(
                optimizer_type=cfg.optimizer_type,
                max_iter=n_iter,
                learning_rate=cfg.learning_rate,
                tol=cfg.tol,
                early_stop_patience=cfg.early_stop_patience,
                lr_scheduler=cfg.lr_scheduler,
                lr_decay=cfg.lr_decay,
                lr_step_size=cfg.lr_step_size,
                metric=cfg.metric,
                use_composite_loss=cfg.use_composite_loss,
                loss_weights=cfg.loss_weights,
                regularization=cfg.regularization,
                bounds=cfg.bounds,
                verbose=cfg.verbose,
                random_seed=cfg.random_seed,
                use_multi_process=cfg.use_multi_process,
                process_conditions=cfg.process_conditions,
                process_window_mode=cfg.process_window_mode,
                focus_range=cfg.focus_range,
                dose_range=cfg.dose_range,
                na_range=cfg.na_range,
                sigma_range=cfg.sigma_range,
                process_center_weight=cfg.process_center_weight,
                process_edge_weight=cfg.process_edge_weight,
                robustness_loss_weight=cfg.robustness_loss_weight,
                threshold=cfg.threshold,
                use_wafer_image_loss=cfg.use_wafer_image_loss,
                binary_penalty_type=cfg.binary_penalty_type,
                epe_threshold=cfg.epe_threshold,
                epe_sigma=cfg.epe_sigma,
                epe_use_soft=cfg.epe_use_soft,
                min_feature_size=cfg.min_feature_size,
                min_feature_method=cfg.min_feature_method,
                min_feature_alpha=cfg.min_feature_alpha,
                pixel_size=cfg.pixel_size,
            )

            level_optimizer = MaskOptimizer(
                optical_system=self.optical_system,
                config=level_config
            )
            level_result = level_optimizer.optimize(
                initial_mask=current_mask,
                target_image=target_at_scale,
                callback=callback
            )
            current_mask = level_result.optimized_mask
            loss_history.extend(level_result.loss_history)

            if not is_last:
                current_mask = upsample_mask(current_mask, scales[level_idx + 1])

        self._setup_imaging_model(current_mask.shape)
        final_wafer = self._imaging_model.compute_aerial_image(current_mask)
        final_metrics = evaluate_all(final_wafer, self._target_image)
        initial_wafer = self._imaging_model.compute_aerial_image(initial_mask)
        initial_metrics = evaluate_all(initial_wafer, self._target_image)

        total_time = time.time() - start_time

        logger.info(
            f"金字塔多尺度优化完成，最终MSE: {final_metrics.mse:.6e}，"
            f"耗时: {total_time:.2f}秒"
        )

        return MaskOptimizationResult(
            optimized_mask=current_mask,
            initial_mask=initial_mask,
            target_image=target_image,
            final_wafer_image=final_wafer,
            initial_wafer_image=initial_wafer,
            final_metrics=final_metrics,
            initial_metrics=initial_metrics,
            loss_history=loss_history,
            total_iterations=len(loss_history),
            total_time=total_time,
            converged=True,
            message="金字塔多尺度优化完成"
        )

    def _optimize_tile(self,
                       initial_mask: np.ndarray,
                       target_image: np.ndarray,
                       callback: Optional[Callable[[int, np.ndarray, float], None]] = None
                       ) -> MaskOptimizationResult:
        """
        分块 tile 优化

        将大尺寸掩模分割为重叠的 tile 块，逐块优化后拼接融合。

        Args:
            initial_mask: 初始掩模
            target_image: 目标图像
            callback: 回调函数

        Returns:
            MaskOptimizationResult
        """
        start_time = time.time()
        cfg = self.config
        tile_size = cfg.tile_size
        overlap = cfg.tile_overlap
        blend_sigma = cfg.tile_blend_sigma

        ny, nx = initial_mask.shape
        needs_tiling = (ny > tile_size or nx > tile_size)

        if not needs_tiling:
            logger.info("掩模尺寸不超过 tile 大小，回退到普通优化")
            return self.optimize(initial_mask, target_image, callback)

        logger.info(
            f"分块 tile 优化: tile_size={tile_size}, overlap={overlap}, "
            f"blend_sigma={blend_sigma}, mask_shape=({ny}, {nx})"
        )

        self._target_image = target_image.astype(np.float64)
        current_mask = initial_mask.astype(np.float64)
        loss_history = []

        n_epochs = max(1, cfg.max_iter)
        for epoch in range(n_epochs):
            tiles = split_tiles(current_mask, tile_size, overlap)

            for t_idx, tile_info in enumerate(tiles):
                rs = tile_info['row_start']
                cs = tile_info['col_start']
                re = tile_info['row_end']
                ce = tile_info['col_end']
                tile_data = tile_info['data']
                tile_target = self._target_image[rs:re, cs:ce].copy()

                tile_config = OptimizationConfig(
                    optimizer_type=cfg.optimizer_type,
                    max_iter=max(5, n_epochs // 2),
                    learning_rate=cfg.learning_rate,
                    tol=cfg.tol,
                    early_stop_patience=5,
                    lr_scheduler=cfg.lr_scheduler,
                    lr_decay=cfg.lr_decay,
                    lr_step_size=cfg.lr_step_size,
                    metric=cfg.metric,
                    use_composite_loss=cfg.use_composite_loss,
                    loss_weights=cfg.loss_weights,
                    regularization=cfg.regularization,
                    bounds=cfg.bounds,
                    verbose=False,
                    random_seed=cfg.random_seed,
                    threshold=cfg.threshold,
                    use_wafer_image_loss=cfg.use_wafer_image_loss,
                    binary_penalty_type=cfg.binary_penalty_type,
                    epe_threshold=cfg.epe_threshold,
                    epe_sigma=cfg.epe_sigma,
                    epe_use_soft=cfg.epe_use_soft,
                    min_feature_size=cfg.min_feature_size,
                    min_feature_method=cfg.min_feature_method,
                    min_feature_alpha=cfg.min_feature_alpha,
                    pixel_size=cfg.pixel_size,
                )

                tile_optimizer = MaskOptimizer(
                    optical_system=self.optical_system,
                    config=tile_config
                )
                tile_result = tile_optimizer.optimize(
                    initial_mask=tile_data,
                    target_image=tile_target
                )
                tile_info['data'] = tile_result.optimized_mask
                loss_history.extend(tile_result.loss_history)

            current_mask = merge_tiles_with_blend(
                tiles, (ny, nx), overlap, blend_sigma
            )

            if callback is not None:
                self._setup_imaging_model(current_mask.shape)
                aerial = self._imaging_model.compute_aerial_image(current_mask)
                current_loss = mse(aerial, self._target_image)
                callback(epoch, current_mask, current_loss)

            if cfg.verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                self._setup_imaging_model(current_mask.shape)
                aerial = self._imaging_model.compute_aerial_image(current_mask)
                current_loss = mse(aerial, self._target_image)
                logger.info(f"Tile epoch {epoch:4d}: loss={current_loss:.6e}")

        self._setup_imaging_model(current_mask.shape)
        final_wafer = self._imaging_model.compute_aerial_image(current_mask)
        final_metrics = evaluate_all(final_wafer, self._target_image)
        initial_wafer = self._imaging_model.compute_aerial_image(initial_mask)
        initial_metrics = evaluate_all(initial_wafer, self._target_image)

        total_time = time.time() - start_time

        logger.info(
            f"分块 tile 优化完成，最终MSE: {final_metrics.mse:.6e}，"
            f"耗时: {total_time:.2f}秒"
        )

        return MaskOptimizationResult(
            optimized_mask=current_mask,
            initial_mask=initial_mask,
            target_image=target_image,
            final_wafer_image=final_wafer,
            initial_wafer_image=initial_wafer,
            final_metrics=final_metrics,
            initial_metrics=initial_metrics,
            loss_history=loss_history,
            total_iterations=len(loss_history),
            total_time=total_time,
            converged=True,
            message="分块 tile 优化完成"
        )

    def optimize_multiscale(self,
                            initial_mask: np.ndarray,
                            target_image: np.ndarray,
                            callback: Optional[Callable[[int, np.ndarray, float], None]] = None
                            ) -> MaskOptimizationResult:
        """
        多尺度/分块优化入口

        根据 config.use_multiscale 和 config.multiscale_mode 选择优化策略：
        - 'pyramid': 金字塔多尺度优化（先低分辨率粗优化，再逐级上采样细化）
        - 'tile': 分块优化后拼接融合

        当 use_multiscale=False 时回退到普通 optimize。

        Args:
            initial_mask: 初始掩模图案
            target_image: 目标图像
            callback: 回调函数 callback(iteration, current_mask, current_loss)

        Returns:
            MaskOptimizationResult
        """
        if not self.config.use_multiscale:
            return self.optimize(initial_mask, target_image, callback)

        mode = self.config.multiscale_mode.lower()
        if mode == 'pyramid':
            return self._optimize_pyramid(initial_mask, target_image, callback)
        elif mode == 'tile':
            return self._optimize_tile(initial_mask, target_image, callback)
        else:
            raise ValueError(
                f"未知的多尺度模式: {mode}，支持 'pyramid' 或 'tile'"
            )

    def optimize_with_custom_objective(self,
                                       initial_mask: np.ndarray,
                                       objective_func: Callable[[np.ndarray], float],
                                       gradient_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
                                       ) -> OptimizationResult:
        """
        使用自定义目标函数进行优化

        Args:
            initial_mask: 初始掩模
            objective_func: 自定义目标函数
            gradient_func: 自定义梯度函数（可选）

        Returns:
            OptimizationResult对象
        """
        self._setup_optimizer()

        if isinstance(self._optimizer, BaseHeuristicOptimizer):
            return self._optimizer.optimize(
                objective=objective_func,
                x0=initial_mask,
                bounds=self.config.bounds
            )
        else:
            return self._optimizer.optimize(
                objective=objective_func,
                x0=initial_mask,
                gradient=gradient_func,
                bounds=self.config.bounds
            )


class OptimizationStrategy(Enum):
    """多层掩模优化策略"""
    ALTERNATING = "alternating"
    JOINT = "joint"


@dataclass
class MultiLayerOptimizationConfig:
    """
    多层掩模联合优化配置

    Attributes:
        strategy: 优化策略 - 'alternating'（交替优化）或 'joint'（联合优化）
        wafer_mask_config: 晶圆掩模优化配置
        source_mask_config: 光源掩模优化配置
        source_learning_rate: 光源掩模学习率
        alternating_inner_iter: 交替优化时每轮内层迭代次数
        alternating_warmup_iters: 先单独优化晶圆掩模的预热迭代次数
        source_bounds: 光源掩模值边界
        source_regularization: 光源正则化配置
        source_smoothness_weight: 光源平滑度权重
        source_sparsity_weight: 光源稀疏性权重
    """
    strategy: OptimizationStrategy = OptimizationStrategy.ALTERNATING
    wafer_mask_config: Optional[OptimizationConfig] = None
    source_mask_config: Optional[OptimizationConfig] = None
    source_learning_rate: float = 0.001
    alternating_inner_iter: int = 5
    alternating_warmup_iters: int = 0
    source_bounds: Tuple[float, float] = (0.0, 1.0)
    source_regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    source_smoothness_weight: float = 0.0
    source_sparsity_weight: float = 0.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'MultiLayerOptimizationConfig':
        """从字典创建"""
        if d is None:
            return cls()
        cfg = cls()
        if 'strategy' in d:
            cfg.strategy = OptimizationStrategy(d['strategy'])
        if 'wafer_mask_config' in d:
            cfg.wafer_mask_config = OptimizationConfig.from_dict(d['wafer_mask_config'])
        if 'source_mask_config' in d:
            cfg.source_mask_config = OptimizationConfig.from_dict(d['source_mask_config'])
        if 'source_learning_rate' in d:
            cfg.source_learning_rate = float(d['source_learning_rate'])
        if 'alternating_inner_iter' in d:
            cfg.alternating_inner_iter = int(d['alternating_inner_iter'])
        if 'alternating_warmup_iters' in d:
            cfg.alternating_warmup_iters = int(d['alternating_warmup_iters'])
        if 'source_bounds' in d:
            cfg.source_bounds = tuple(d['source_bounds'])
        if 'source_regularization' in d:
            cfg.source_regularization = RegularizationConfig.from_dict(d['source_regularization'])
        if 'source_smoothness_weight' in d:
            cfg.source_smoothness_weight = float(d['source_smoothness_weight'])
        if 'source_sparsity_weight' in d:
            cfg.source_sparsity_weight = float(d['source_sparsity_weight'])
        return cfg


@dataclass
class MultiLayerOptimizationResult:
    """多层掩模联合优化结果"""
    optimized_wafer_mask: np.ndarray
    optimized_source_mask: np.ndarray
    initial_wafer_mask: np.ndarray
    initial_source_mask: np.ndarray
    target_image: np.ndarray
    final_wafer_image: np.ndarray
    initial_wafer_image: np.ndarray
    final_metrics: MetricsResult
    initial_metrics: MetricsResult
    loss_history: List[float]
    source_loss_history: List[float]
    wafer_loss_history: List[float]
    total_iterations: int
    total_time: float
    converged: bool
    message: str
    strategy: OptimizationStrategy
    multi_process_result: Optional[MultiProcessSimulationResult] = None
    process_conditions: Optional[List[ProcessCondition]] = None


class MultiLayerMaskOptimizer:
    """
    多层掩模联合优化器（SMO/DMD）

    支持光源掩模（source mask）与晶圆掩模（wafer mask）的交替优化或联合优化。
    损失函数对两层掩模分别计算梯度并交替更新。

    优化策略:
    - ALTERNATING（交替优化）: 固定光源优化晶圆掩模若干步，然后固定晶圆掩模优化光源若干步，交替进行
    - JOINT（联合优化）: 同时计算损失对光源和晶圆掩模的梯度，同时更新两者
    """

    def __init__(self,
                 optical_system: Optional[OpticalSystem] = None,
                 config: Optional[OptimizationConfig] = None,
                 ml_config: Optional[MultiLayerOptimizationConfig] = None):
        """
        初始化多层掩模优化器

        Args:
            optical_system: 光学系统参数
            config: 基础优化配置（兼容单掩模优化）
            ml_config: 多层优化专用配置
        """
        self.optical_system = optical_system or OpticalSystem()
        self.config = config or OptimizationConfig()
        self.ml_config = ml_config or MultiLayerOptimizationConfig()

        if self.ml_config.wafer_mask_config is not None:
            self.wafer_config = self.ml_config.wafer_mask_config
        else:
            self.wafer_config = self.config

        if self.ml_config.source_mask_config is not None:
            self.source_config = self.ml_config.source_mask_config
        else:
            self.source_config = OptimizationConfig(
                learning_rate=self.ml_config.source_learning_rate,
                max_iter=self.ml_config.alternating_inner_iter,
                use_composite_loss=True,
                loss_weights=LossWeights(mse=1.0)
            )

        self._wafer_optimizer = MaskOptimizer(
            optical_system=self.optical_system,
            config=self.wafer_config
        )

        self._imaging_model: Optional[PartialCoherentImaging] = None
        self._target_image: Optional[np.ndarray] = None

        self._current_source_mask: Optional[np.ndarray] = None
        self._current_wafer_mask: Optional[np.ndarray] = None

        self._multi_imaging_models: Optional[List[PartialCoherentImaging]] = None
        self._multi_conditions: Optional[List[ProcessCondition]] = None
        self._multi_weights: Optional[List[float]] = None

        self._source_lr_scheduler: Optional[LearningRateScheduler] = None
        self._source_early_stopping: Optional[EarlyStopping] = None

    def _setup_imaging_model(self, image_size: Tuple[int, int]):
        """设置成像模型"""
        self._imaging_model = PartialCoherentImaging(
            self.optical_system, image_size
        )

    def _setup_multi_process_models(self, image_size: Tuple[int, int]):
        """设置多工艺条件成像模型"""
        self._wafer_optimizer._build_process_conditions = self._wafer_optimizer._build_process_conditions
        conditions, weights = self._wafer_optimizer._build_process_conditions()

        self._multi_conditions = conditions
        total_w = sum(weights)
        self._multi_weights = [w / total_w for w in weights]

        self._multi_imaging_models = []
        for cond in conditions:
            optics = cond.to_optical_system(base_optics=self.optical_system)
            model = PartialCoherentImaging(optics, image_size)
            self._multi_imaging_models.append(model)

    def _setup_source_optimization(self):
        """设置光源优化相关组件"""
        self._source_lr_scheduler = LearningRateScheduler(
            initial_lr=self.ml_config.source_learning_rate,
            scheduler_type='cosine',
            decay=0.95,
            step_size=50,
            min_lr=1e-7
        )
        self._source_early_stopping = EarlyStopping(
            patience=15,
            min_delta=1e-7
        )

    def _prepare_image(self, aerial: np.ndarray, dose: float = 1.0, config: Optional[OptimizationConfig] = None) -> np.ndarray:
        """对空间像做预处理用于损失计算"""
        cfg = config or self.config
        if dose != 1.0:
            aerial = np.clip(aerial * dose, 0.0, 1.0)
        if cfg.use_wafer_image_loss:
            return _apply_threshold_for_loss(aerial, cfg.threshold)
        return aerial

    def _compute_source_regularization_loss(self, source: np.ndarray) -> float:
        """计算光源正则化损失"""
        loss = 0.0
        cfg = self.ml_config

        if cfg.source_smoothness_weight > 0:
            loss += cfg.source_smoothness_weight * total_variation_isotropic(source)

        if cfg.source_sparsity_weight > 0:
            loss += cfg.source_sparsity_weight * l1_regularization(source)

        reg_cfg = cfg.source_regularization
        if reg_cfg.type is not None and reg_cfg.strength > 0:
            reg_type = reg_cfg.type.lower()
            if reg_type == 'l1':
                loss += reg_cfg.strength * l1_regularization(source)
            elif reg_type == 'l2':
                loss += reg_cfg.strength * l2_regularization(source)
            elif reg_type == 'tv' or reg_type == 'tv_isotropic':
                loss += reg_cfg.strength * total_variation_isotropic(source)

        return loss

    def _compute_source_regularization_gradient(self, source: np.ndarray) -> np.ndarray:
        """计算光源正则化梯度"""
        grad = np.zeros_like(source)
        cfg = self.ml_config

        if cfg.source_smoothness_weight > 0:
            grad += cfg.source_smoothness_weight * total_variation_isotropic_gradient(source)

        if cfg.source_sparsity_weight > 0:
            grad += cfg.source_sparsity_weight * l1_regularization_gradient(source)

        reg_cfg = cfg.source_regularization
        if reg_cfg.type is not None and reg_cfg.strength > 0:
            reg_type = reg_cfg.type.lower()
            if reg_type == 'l1':
                grad += reg_cfg.strength * l1_regularization_gradient(source)
            elif reg_type == 'l2':
                grad += reg_cfg.strength * l2_regularization_gradient(source)
            elif reg_type == 'tv' or reg_type == 'tv_isotropic':
                grad += reg_cfg.strength * total_variation_isotropic_gradient(source)

        return grad

    def _compute_loss_for_source(self, source_mask: np.ndarray,
                                 wafer_mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算给定光源和晶圆掩模下的损失，以及损失对光源的梯度

        Args:
            source_mask: 光源掩模
            wafer_mask: 晶圆掩模（固定）

        Returns:
            (loss_value, gradient) 二元组
        """
        self._imaging_model.update_source(source_mask)

        if self.config.use_multi_process and self._multi_imaging_models is not None:
            return self._compute_multi_process_loss_for_source(source_mask, wafer_mask)

        return self._compute_single_process_loss_for_source(source_mask, wafer_mask)

    def _compute_single_process_loss_for_source(self, source_mask: np.ndarray,
                                                wafer_mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """单工艺条件下的光源损失和梯度计算"""
        cfg = self.config
        lw = cfg.loss_weights

        aerial = self._imaging_model.compute_aerial_image(wafer_mask)
        image = self._prepare_image(aerial, dose=1.0, config=cfg)

        loss = 0.0
        error_grad = np.zeros_like(image)

        if cfg.use_composite_loss:
            if lw.mse > 0:
                mse_val = mse(image, self._target_image)
                loss += lw.mse * mse_val
                error_grad += lw.mse * (2.0 * (image - self._target_image) / image.size)
            if lw.ssim > 0:
                ssim_val = 1.0 - ssim(image, self._target_image)
                loss += lw.ssim * ssim_val
                error_grad += lw.ssim * ssim_loss_gradient(image, self._target_image)
        else:
            metric = cfg.metric.lower()
            if metric == 'mse':
                loss = mse(image, self._target_image)
                error_grad = 2.0 * (image - self._target_image) / image.size
            elif metric == 'mae':
                loss = mae(image, self._target_image)
                error_grad = np.sign(image - self._target_image) / image.size
            elif metric == 'ssim':
                loss = 1.0 - ssim(image, self._target_image)
                error_grad = ssim_loss_gradient(image, self._target_image)

        if cfg.use_wafer_image_loss:
            threshold_grad = (aerial >= cfg.threshold).astype(np.float64)
            error_grad = error_grad * threshold_grad

        source_grad = self._imaging_model.compute_source_gradient(wafer_mask)
        gradient = error_grad.mean() * source_grad

        gradient += self._compute_source_regularization_gradient(source_mask)
        loss += self._compute_source_regularization_loss(source_mask)

        return loss, gradient

    def _compute_multi_process_loss_for_source(self, source_mask: np.ndarray,
                                               wafer_mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """多工艺条件下的光源损失和梯度计算"""
        for model in self._multi_imaging_models:
            model.update_source(source_mask)

        cfg = self.config
        lw = cfg.loss_weights
        gradient = np.zeros_like(source_mask)
        total_loss = 0.0
        per_losses = []

        for model, cond, w in zip(
            self._multi_imaging_models,
            self._multi_conditions,
            self._multi_weights
        ):
            aerial = model.compute_aerial_image(wafer_mask)
            image = self._prepare_image(aerial, cond.dose, config=cfg)

            error_grad = np.zeros_like(image)
            loss_i = 0.0

            if cfg.use_composite_loss:
                if lw.mse > 0:
                    loss_i += lw.mse * mse(image, self._target_image)
                    error_grad += lw.mse * (2.0 * (image - self._target_image) / image.size)
                if lw.ssim > 0:
                    loss_i += lw.ssim * (1.0 - ssim(image, self._target_image))
                    error_grad += lw.ssim * ssim_loss_gradient(image, self._target_image)
            else:
                metric = cfg.metric.lower()
                if metric == 'mse':
                    loss_i = mse(image, self._target_image)
                    error_grad = 2.0 * (image - self._target_image) / image.size
                elif metric == 'mae':
                    loss_i = mae(image, self._target_image)
                    error_grad = np.sign(image - self._target_image) / image.size
                elif metric == 'ssim':
                    loss_i = 1.0 - ssim(image, self._target_image)
                    error_grad = ssim_loss_gradient(image, self._target_image)

            if cfg.use_wafer_image_loss:
                aerial_dosed = aerial if cond.dose == 1.0 else np.clip(aerial * cond.dose, 0.0, 1.0)
                threshold_grad = (aerial_dosed >= cfg.threshold).astype(np.float64)
                error_grad = error_grad * threshold_grad

            source_grad_i = model.compute_source_gradient(wafer_mask)
            gradient += w * error_grad.mean() * source_grad_i
            total_loss += w * loss_i
            per_losses.append(loss_i)

        gradient += self._compute_source_regularization_gradient(source_mask)
        total_loss += self._compute_source_regularization_loss(source_mask)

        if cfg.robustness_loss_weight > 0 and len(per_losses) > 1:
            loss_arr = np.array(per_losses)
            robustness = float(np.var(loss_arr))
            total_loss += cfg.robustness_loss_weight * robustness

            mean_loss = np.mean(loss_arr)
            n = len(per_losses)
            for model, cond, w in zip(
                self._multi_imaging_models,
                self._multi_conditions,
                self._multi_weights
            ):
                idx = self._multi_conditions.index(cond)
                factor = 2.0 * cfg.robustness_loss_weight * (per_losses[idx] - mean_loss) / (n * n)
                if abs(factor) < 1e-12:
                    continue
                source_grad_i = model.compute_source_gradient(wafer_mask)
                gradient += factor * source_grad_i

        return total_loss, gradient

    def _optimize_wafer_step(self, source_mask: np.ndarray,
                             wafer_mask: np.ndarray,
                             n_iter: int) -> Tuple[np.ndarray, float, List[float]]:
        """
        固定光源，优化晶圆掩模

        Args:
            source_mask: 当前光源掩模（固定）
            wafer_mask: 当前晶圆掩模（待优化）
            n_iter: 迭代次数

        Returns:
            (optimized_mask, final_loss, loss_history) 三元组
        """
        self._imaging_model.update_source(source_mask)
        if self._multi_imaging_models is not None:
            for model in self._multi_imaging_models:
                model.update_source(source_mask)

        old_max_iter = self._wafer_optimizer.config.max_iter
        self._wafer_optimizer.config.max_iter = n_iter

        try:
            result = self._wafer_optimizer.optimize(
                initial_mask=wafer_mask,
                target_image=self._target_image
            )
            final_loss = result.loss_history[-1] if result.loss_history else float('inf')
            return result.optimized_mask, final_loss, result.loss_history
        finally:
            self._wafer_optimizer.config.max_iter = old_max_iter

    def _optimize_source_step(self, source_mask: np.ndarray,
                              wafer_mask: np.ndarray,
                              n_iter: int,
                              epoch: int = 0) -> Tuple[np.ndarray, float, List[float]]:
        """
        固定晶圆掩模，优化光源

        Args:
            source_mask: 当前光源掩模（待优化）
            wafer_mask: 当前晶圆掩模（固定）
            n_iter: 迭代次数
            epoch: 当前外层迭代次数（用于学习率调度）

        Returns:
            (optimized_source, final_loss, loss_history) 三元组
        """
        current_source = source_mask.copy()
        loss_history = []
        lr = self._source_lr_scheduler.step(epoch) if self._source_lr_scheduler else self.ml_config.source_learning_rate

        for i in range(n_iter):
            loss, grad = self._compute_loss_for_source(current_source, wafer_mask)
            loss_history.append(loss)

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e3:
                grad = grad / grad_norm * 1e3

            current_source = current_source - lr * grad
            current_source = np.clip(current_source, *self.ml_config.source_bounds)

            total = np.sum(current_source)
            if total > 0:
                current_source = current_source / total

            if np.linalg.norm(grad) < self.config.tol:
                break

        final_loss = loss_history[-1] if loss_history else float('inf')
        return current_source, final_loss, loss_history

    def _optimize_alternating(self,
                              initial_wafer_mask: np.ndarray,
                              initial_source_mask: np.ndarray,
                              callback: Optional[Callable[[int, np.ndarray, np.ndarray, float], None]] = None
                              ) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], List[float]]:
        """
        交替优化策略

        Args:
            initial_wafer_mask: 初始晶圆掩模
            initial_source_mask: 初始光源掩模
            callback: 回调函数 callback(epoch, wafer_mask, source_mask, current_loss)

        Returns:
            (final_wafer_mask, final_source_mask, total_loss_history, wafer_loss_history, source_loss_history)
        """
        wafer_mask = initial_wafer_mask.copy()
        source_mask = initial_source_mask.copy()

        total_loss_history = []
        wafer_loss_history = []
        source_loss_history = []

        max_iter = self.config.max_iter
        inner_iter = self.ml_config.alternating_inner_iter
        warmup_iters = self.ml_config.alternating_warmup_iters

        if warmup_iters > 0:
            logger.info(f"开始晶圆掩模预热优化，{warmup_iters} 次迭代")
            wafer_mask, wafer_loss, hist = self._optimize_wafer_step(
                source_mask, wafer_mask, warmup_iters
            )
            wafer_loss_history.extend(hist)

        for epoch in range(max_iter):
            wafer_mask, wafer_loss, wafer_hist = self._optimize_wafer_step(
                source_mask, wafer_mask, inner_iter
            )
            wafer_loss_history.extend(wafer_hist)

            source_mask, source_loss, source_hist = self._optimize_source_step(
                source_mask, wafer_mask, inner_iter, epoch
            )
            source_loss_history.extend(source_hist)

            total_loss = (wafer_loss + source_loss) / 2
            total_loss_history.append(total_loss)

            if self.config.verbose and (epoch % 10 == 0 or epoch == max_iter - 1):
                logger.info(
                    f"Epoch {epoch:4d}: total_loss={total_loss:.6e}, "
                    f"wafer_loss={wafer_loss:.6e}, source_loss={source_loss:.6e}"
                )

            if callback is not None:
                callback(epoch, wafer_mask, source_mask, total_loss)

            if self._source_early_stopping and self._source_early_stopping(total_loss):
                logger.info(f"早停触发，在 epoch {epoch} 停止优化")
                break

        return wafer_mask, source_mask, total_loss_history, wafer_loss_history, source_loss_history

    def _optimize_joint(self,
                        initial_wafer_mask: np.ndarray,
                        initial_source_mask: np.ndarray,
                        callback: Optional[Callable[[int, np.ndarray, np.ndarray, float], None]] = None
                        ) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], List[float]]:
        """
        联合优化策略：同时更新晶圆掩模和光源

        Args:
            initial_wafer_mask: 初始晶圆掩模
            initial_source_mask: 初始光源掩模
            callback: 回调函数 callback(epoch, wafer_mask, source_mask, current_loss)

        Returns:
            (final_wafer_mask, final_source_mask, total_loss_history, wafer_loss_history, source_loss_history)
        """
        wafer_mask = initial_wafer_mask.copy()
        source_mask = initial_source_mask.copy()

        total_loss_history = []
        wafer_loss_history = []
        source_loss_history = []

        max_iter = self.config.max_iter
        wafer_lr = self.wafer_config.learning_rate
        source_lr = self.ml_config.source_learning_rate

        wafer_grad_optimizer = GradientDescentOptimizer(
            learning_rate=wafer_lr,
            max_iter=1,
            tol=self.config.tol,
            verbose=False
        )

        for epoch in range(max_iter):
            self._imaging_model.update_source(source_mask)
            if self._multi_imaging_models is not None:
                for model in self._multi_imaging_models:
                    model.update_source(source_mask)

            wafer_loss = self._wafer_optimizer._compute_loss(wafer_mask)
            wafer_grad = self._wafer_optimizer._compute_gradient(wafer_mask)

            wafer_result = wafer_grad_optimizer.optimize(
                objective=lambda x: self._wafer_optimizer._compute_loss(x),
                x0=wafer_mask,
                gradient=lambda x: self._wafer_optimizer._compute_gradient(x),
                bounds=self.config.bounds
            )
            wafer_mask = wafer_result.x

            source_loss, source_grad = self._compute_loss_for_source(source_mask, wafer_mask)

            grad_norm = np.linalg.norm(source_grad)
            if grad_norm > 1e3:
                source_grad = source_grad / grad_norm * 1e3

            source_mask = source_mask - source_lr * source_grad
            source_mask = np.clip(source_mask, *self.ml_config.source_bounds)

            total = np.sum(source_mask)
            if total > 0:
                source_mask = source_mask / total

            total_loss = (wafer_loss + source_loss) / 2
            total_loss_history.append(total_loss)
            wafer_loss_history.append(wafer_loss)
            source_loss_history.append(source_loss)

            if self.config.verbose and (epoch % 10 == 0 or epoch == max_iter - 1):
                logger.info(
                    f"Epoch {epoch:4d}: total_loss={total_loss:.6e}, "
                    f"wafer_loss={wafer_loss:.6e}, source_loss={source_loss:.6e}"
                )

            if callback is not None:
                callback(epoch, wafer_mask, source_mask, total_loss)

            if self._source_early_stopping and self._source_early_stopping(total_loss):
                logger.info(f"早停触发，在 epoch {epoch} 停止优化")
                break

        return wafer_mask, source_mask, total_loss_history, wafer_loss_history, source_loss_history

    def optimize(self,
                 initial_wafer_mask: np.ndarray,
                 initial_source_mask: np.ndarray,
                 target_image: np.ndarray,
                 callback: Optional[Callable[[int, np.ndarray, np.ndarray, float], None]] = None
                 ) -> MultiLayerOptimizationResult:
        """
        执行多层掩模联合优化

        Args:
            initial_wafer_mask: 初始晶圆掩模图案
            initial_source_mask: 初始光源掩模图案
            target_image: 目标图像
            callback: 回调函数 callback(iteration, wafer_mask, source_mask, current_loss)

        Returns:
            MultiLayerOptimizationResult 对象
        """
        start_time = time.time()

        self._target_image = target_image.astype(np.float64)
        image_size = initial_wafer_mask.shape

        self._setup_imaging_model(image_size)
        self._setup_source_optimization()

        if self.config.use_multi_process:
            self._setup_multi_process_models(image_size)
        else:
            self._multi_imaging_models = None
            self._multi_conditions = None
            self._multi_weights = None

        self._wafer_optimizer._target_image = self._target_image
        self._wafer_optimizer._setup_imaging_model(image_size)
        self._wafer_optimizer._setup_optimizer()
        self._wafer_optimizer._setup_lr_scheduler()
        self._wafer_optimizer._setup_early_stopping()

        if self.config.use_multi_process:
            self._wafer_optimizer._setup_multi_process_models(image_size)

        initial_source = initial_source_mask.copy()
        self._imaging_model.update_source(initial_source)
        self._wafer_optimizer._imaging_model.update_source(initial_source)
        if self._multi_imaging_models is not None:
            for model, wafer_model in zip(self._multi_imaging_models, self._wafer_optimizer._multi_imaging_models):
                model.update_source(initial_source)
                wafer_model.update_source(initial_source)

        initial_wafer = self._imaging_model.compute_aerial_image(initial_wafer_mask)
        initial_metrics = evaluate_all(initial_wafer, target_image)

        logger.info(
            f"开始多层掩模{self.ml_config.strategy.value}优化，"
            f"策略: {self.ml_config.strategy.value}, "
            f"初始MSE: {initial_metrics.mse:.6e}"
        )

        if self.ml_config.strategy == OptimizationStrategy.ALTERNATING:
            opt_wafer, opt_source, total_hist, wafer_hist, source_hist = self._optimize_alternating(
                initial_wafer_mask, initial_source_mask, callback
            )
        else:
            opt_wafer, opt_source, total_hist, wafer_hist, source_hist = self._optimize_joint(
                initial_wafer_mask, initial_source_mask, callback
            )

        self._imaging_model.update_source(opt_source)
        final_wafer = self._imaging_model.compute_aerial_image(opt_wafer)
        final_metrics = evaluate_all(final_wafer, target_image)

        total_time = time.time() - start_time

        multi_process_result = None
        process_conditions = None

        if self.config.use_multi_process and self._multi_conditions is not None:
            for model in self._multi_imaging_models:
                model.update_source(opt_source)
            multi_process_result = simulate_multi_process(
                opt_wafer,
                self._multi_conditions,
                base_optics=self.optical_system,
                threshold=self.config.threshold,
                apply_resist=self.config.use_wafer_image_loss
            )
            process_conditions = self._multi_conditions

        converged = len(total_hist) > 0 and (
            self._source_early_stopping.should_stop if self._source_early_stopping else False
        )

        logger.info(
            f"优化完成，最终MSE: {final_metrics.mse:.6e}，"
            f"耗时: {total_time:.2f}秒"
        )

        return MultiLayerOptimizationResult(
            optimized_wafer_mask=opt_wafer,
            optimized_source_mask=opt_source,
            initial_wafer_mask=initial_wafer_mask,
            initial_source_mask=initial_source_mask,
            target_image=target_image,
            final_wafer_image=final_wafer,
            initial_wafer_image=initial_wafer,
            final_metrics=final_metrics,
            initial_metrics=initial_metrics,
            loss_history=total_hist,
            source_loss_history=source_hist,
            wafer_loss_history=wafer_hist,
            total_iterations=len(total_hist),
            total_time=total_time,
            converged=converged,
            message="优化完成" if converged else "达到最大迭代次数",
            strategy=self.ml_config.strategy,
            multi_process_result=multi_process_result,
            process_conditions=process_conditions
        )
