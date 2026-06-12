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
    simulate_multi_process, create_focus_dose_window, create_full_process_window
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
    CompositeLossComponents
)
from algorithms.optimizer import (
    BaseOptimizer, GradientDescentOptimizer, BFGSOptimizer,
    NewtonOptimizer, OptimizationResult
)
from algorithms.advanced_optimizer import (
    BaseHeuristicOptimizer, GeneticAlgorithmOptimizer, ParticleSwarmOptimizer
)

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

    Attributes:
        mse: MSE（均方误差）权重
        ssim: (1-SSIM) 结构相似性损失权重
        pvb: PVB（Process Variation Band，工艺变化带宽）权重
        mask_complexity: 掩模复杂度（总变差TV）权重
        binary_penalty: 二值化惩罚权重（曼哈顿距离/熵）
        tv_smooth: 各向同性TV平滑权重
        epe: 边缘放置误差（EPE）权重
        min_feature: 最小特征尺寸约束权重
    """
    mse: float = 1.0
    ssim: float = 0.0
    pvb: float = 0.0
    mask_complexity: float = 0.0
    binary_penalty: float = 0.0
    tv_smooth: float = 0.0
    epe: float = 0.0
    min_feature: float = 0.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, float]]) -> 'LossWeights':
        """从字典创建，缺失键使用默认值"""
        if d is None:
            return cls()
        defaults = {
            'mse': 1.0, 'ssim': 0.0, 'pvb': 0.0, 'mask_complexity': 0.0,
            'binary_penalty': 0.0, 'tv_smooth': 0.0, 'epe': 0.0, 'min_feature': 0.0
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
            'min_feature': self.min_feature
        }

    def total_weight(self) -> float:
        return (self.mse + self.ssim + self.pvb + self.mask_complexity +
                self.binary_penalty + self.tv_smooth + self.epe + self.min_feature)


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


class MaskOptimizer:
    """
    掩模优化器

    实现完整的掩模图案优化流程。
    支持多工艺条件联合优化：对 focus、dose、NA、sigma 等参数扫描，
    同时约束工艺窗口中心与边界的成像质量。
    """

    def __init__(self,
                 optical_system: Optional[OpticalSystem] = None,
                 config: Optional[OptimizationConfig] = None):
        """
        初始化掩模优化器

        Args:
            optical_system: 光学系统参数
            config: 优化配置
        """
        self.optical_system = optical_system or OpticalSystem()
        self.config = config or OptimizationConfig()

        self._imaging_model: Optional[PartialCoherentImaging] = None
        self._target_image: Optional[np.ndarray] = None
        self._optimizer: Optional[Union[BaseOptimizer, BaseHeuristicOptimizer]] = None
        self._lr_scheduler: Optional[LearningRateScheduler] = None
        self._early_stopping: Optional[EarlyStopping] = None

        self._multi_imaging_models: Optional[List[PartialCoherentImaging]] = None
        self._multi_conditions: Optional[List[ProcessCondition]] = None
        self._multi_weights: Optional[List[float]] = None

    def _setup_imaging_model(self, image_size: tuple):
        """设置成像模型"""
        self._imaging_model = PartialCoherentImaging(
            self.optical_system, image_size
        )

    def _setup_optimizer(self):
        """设置优化器"""
        opt_type = self.config.optimizer_type.lower()
        seed = self.config.random_seed  # 从配置获取随机种子

        if opt_type == 'gradient_descent':
            self._optimizer = GradientDescentOptimizer(
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
        针对单幅图像计算 MSE、(1-SSIM) 等逐像素损失分量（不含 PVB/正则化）

        Args:
            image: 处理后的成像结果
            target: 目标图像

        Returns:
            CompositeLossComponents（仅填充 mse、ssim 字段）
        """
        comp = CompositeLossComponents()
        lw = self.config.loss_weights

        if lw.mse > 0:
            comp.mse = lw.mse * mse(image, target)
        if lw.ssim > 0:
            comp.ssim = lw.ssim * (1.0 - ssim(image, target))

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
                                            dose: float = 1.0) -> Tuple[float, np.ndarray, CompositeLossComponents]:
        """
        计算单工艺条件下的复合损失、成像结果及各分量（不含 PVB 和正则化）

        Args:
            mask: 掩模
            imaging_model: 成像模型
            dose: 曝光剂量

        Returns:
            (loss_value, processed_image, components)
        """
        aerial = imaging_model.compute_aerial_image(mask)
        image = self._prepare_image(aerial, dose)
        components = self._compute_image_loss_components(image, self._target_image)
        loss = components.mse + components.ssim
        return loss, image, components

    def _compute_single_condition_loss(self, mask: np.ndarray,
                                       imaging_model: PartialCoherentImaging,
                                       dose: float = 1.0) -> float:
        """
        计算单组工艺条件下的损失

        当 config.use_composite_loss=True 时，使用复合损失（MSE/SSIM 加权）；
        否则回退到旧的单一 metric 逻辑。

        Args:
            mask: 掩模图案
            imaging_model: 成像模型
            dose: 曝光相对剂量

        Returns:
            损失值
        """
        if self.config.use_composite_loss:
            loss, _, _ = self._compute_composite_single_condition(mask, imaging_model, dose)
            return loss

        aerial = imaging_model.compute_aerial_image(mask)

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

            for model, cond, w in zip(
                self._multi_imaging_models,
                self._multi_conditions,
                self._multi_weights
            ):
                loss_i, img_i, _ = self._compute_composite_single_condition(mask, model, cond.dose)
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
        for model, cond, w in zip(
            self._multi_imaging_models,
            self._multi_conditions,
            self._multi_weights
        ):
            loss_i = self._compute_single_condition_loss(mask, model, cond.dose)
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

        对于包含 PVB 的情况，退化为数值梯度。

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        cfg = self.config

        if cfg.use_composite_loss:
            lw = cfg.loss_weights
            if lw.pvb > 0:
                return self._numerical_gradient(mask)

            gradient = np.zeros_like(mask)
            per_images = []
            per_losses = []

            for model, cond, w in zip(
                self._multi_imaging_models,
                self._multi_conditions,
                self._multi_weights
            ):
                aerial = model.compute_aerial_image(mask)
                image = self._prepare_image(aerial, cond.dose)
                per_images.append(image)

                error_grad = np.zeros_like(image)

                if lw.mse > 0:
                    error_grad += lw.mse * (2.0 * (image - self._target_image) / mask.size)

                if lw.ssim > 0:
                    error_grad += lw.ssim * ssim_loss_gradient(image, self._target_image)

                imaging_grad = model.compute_image_gradient(mask)

                if cond.dose != 1.0:
                    error_grad = error_grad * cond.dose

                if cfg.use_wafer_image_loss:
                    aerial_dosed = aerial if cond.dose == 1.0 else np.clip(aerial * cond.dose, 0.0, 1.0)
                    threshold_grad = (aerial_dosed >= cfg.threshold).astype(np.float64)
                    error_grad = error_grad * threshold_grad

                gradient += w * (error_grad * imaging_grad)

                loss_i, _, _ = self._compute_composite_single_condition(mask, model, cond.dose)
                per_losses.append(loss_i)

            if lw.mask_complexity > 0:
                gradient += lw.mask_complexity * total_variation_gradient(mask)

            gradient += self._compute_mask_constraints_gradient(mask)

            gradient += self._compute_regularization_gradient(mask)

            if cfg.robustness_loss_weight > 0 and len(per_losses) > 1:
                loss_arr = np.array(per_losses)
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
                    aerial = model.compute_aerial_image(mask)
                    image = self._prepare_image(aerial, cond.dose)
                    error_grad = np.zeros_like(image)
                    if lw.mse > 0:
                        error_grad += lw.mse * (2.0 * (image - self._target_image) / mask.size)
                    if lw.ssim > 0:
                        error_grad += lw.ssim * ssim_loss_gradient(image, self._target_image)
                    imaging_grad = model.compute_image_gradient(mask)
                    if cond.dose != 1.0:
                        error_grad = error_grad * cond.dose
                    gradient += factor * (error_grad * imaging_grad)

            return gradient

        metric = self.config.metric.lower()
        gradient = np.zeros_like(mask)
        per_losses = []

        for model, cond, w in zip(
            self._multi_imaging_models,
            self._multi_conditions,
            self._multi_weights
        ):
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

            imaging_grad = model.compute_image_gradient(mask)

            if cond.dose != 1.0:
                error_grad = error_grad * cond.dose

            if self.config.use_wafer_image_loss:
                threshold_grad = (aerial_dosed >= self.config.threshold).astype(np.float64)
                error_grad = error_grad * threshold_grad

            gradient += w * (error_grad * imaging_grad)

            loss_i = self._compute_single_condition_loss(mask, model, cond.dose)
            per_losses.append(loss_i)

        if self.config.robustness_loss_weight > 0 and len(per_losses) > 1:
            loss_arr = np.array(per_losses)
            mean_loss = np.mean(loss_arr)
            n = len(per_losses)
            for model, cond, w in zip(
                self._multi_imaging_models,
                self._multi_conditions,
                self._multi_weights
            ):
                idx = self._multi_conditions.index(cond)
                factor = 2.0 * self.config.robustness_loss_weight * (per_losses[idx] - mean_loss) / (n * n)
                if abs(factor) < 1e-12:
                    continue
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

            aerial = self._imaging_model.compute_aerial_image(mask)
            image = self._prepare_image(aerial, dose=1.0)
            imaging_grad = self._imaging_model.compute_image_gradient(mask)

            error_grad = np.zeros_like(image)

            if lw.mse > 0:
                error_grad += lw.mse * (2.0 * (image - self._target_image) / mask.size)

            if lw.ssim > 0:
                error_grad += lw.ssim * ssim_loss_gradient(image, self._target_image)

            if cfg.use_wafer_image_loss:
                threshold_grad = (aerial >= cfg.threshold).astype(np.float64)
                error_grad = error_grad * threshold_grad

            gradient = error_grad * imaging_grad

            if lw.mask_complexity > 0:
                gradient += lw.mask_complexity * total_variation_gradient(mask)

            gradient += self._compute_mask_constraints_gradient(mask)

            gradient += self._compute_regularization_gradient(mask)

            return gradient

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

    def optimize(self,
                 initial_mask: np.ndarray,
                 target_image: np.ndarray,
                 callback: Optional[Callable[[int, np.ndarray, float], None]] = None
                 ) -> MaskOptimizationResult:
        """
        执行掩模优化

        当 config.use_multi_process=True 时，自动构建多工艺条件
        成像模型并进行联合优化，同时约束工艺窗口中心与边界。

        Args:
            initial_mask: 初始掩模图案
            target_image: 目标图像
            callback: 回调函数 callback(iteration, current_mask, current_loss)

        Returns:
            MaskOptimizationResult对象
        """
        start_time = time.time()

        self._target_image = target_image.astype(np.float64)

        self._setup_imaging_model(initial_mask.shape)
        self._setup_optimizer()
        self._setup_lr_scheduler()
        self._setup_early_stopping()

        if self.config.use_multi_process:
            self._setup_multi_process_models(initial_mask.shape)
        else:
            self._multi_imaging_models = None
            self._multi_conditions = None
            self._multi_weights = None

        initial_wafer = self._imaging_model.compute_aerial_image(initial_mask)
        initial_metrics = evaluate_all(initial_wafer, target_image)

        if self.config.use_multi_process:
            logger.info(f"开始多工艺条件联合掩模优化，{len(self._multi_conditions)} 个工艺条件，"
                       f"初始MSE: {initial_metrics.mse:.6e}")
        else:
            logger.info(f"开始掩模优化，初始MSE: {initial_metrics.mse:.6e}")

        if isinstance(self._optimizer, BaseHeuristicOptimizer):
            result = self._optimizer.optimize(
                objective=self._compute_loss,
                x0=initial_mask,
                bounds=self.config.bounds
            )
        else:
            result = self._optimizer.optimize(
                objective=self._compute_loss,
                x0=initial_mask,
                gradient=self._compute_gradient,
                bounds=self.config.bounds
            )

        optimized_mask = result.x
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
            process_conditions=process_conditions
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
