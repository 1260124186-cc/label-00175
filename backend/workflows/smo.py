# -*- coding: utf-8 -*-
"""
SMO (Source-Mask Optimization) 光源掩模协同优化工作流模块

实现光刻系统中的光源与掩模联合优化：
    输入原始版图 + 初始照明条件 → 像素化光源参数化 →
    联合成像前向模型 → 交替/联合梯度下降 →
    约束投影（能量守恒、sigma等效、平滑正则化） →
    输出最优光源分布与校正后掩模

主要组件：
    1. PixelatedSource: 像素化光源模型，独立参数化，支持约束投影
    2. SMOImagingModel: 联合成像前向模型，扩展 PartialCoherentImaging
       支持 source ⊗ mask 的联合频域卷积（SOCS / FULL_TCC 形式）
    3. SourceConstraints: 光源约束（总能量守恒、sigma等效、平滑正则化）
    4. SMOConfig: SMO 优化配置
    5. SMOResult: SMO 单次迭代结果
    6. SMOWorkflowResult: SMO 工作流最终结果
    7. SMOWorkflow: 完整 SMO 工作流封装，支持交替优化与联合梯度下降
    8. run_smo_workflow: 便捷入口函数
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from scipy.ndimage import gaussian_filter

from core.imaging import (
    OpticalSystem, PartialCoherentImaging,
    ProcessCondition, ProcessWindow,
    IlluminationType, TCCMode,
    generate_source, _shift_pupil,
    compute_zernike_phase, _compute_pupil_with_aberrations,
    socs_decomposition, simulate_wafer_image,
    ResistModel, apply_resist_model,
)
from core.fft import WindowType
from core.metrics import (
    mse, mae, ssim, edge_placement_error,
    total_variation_isotropic,
)
from core.litho_metrics import compute_epe
from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig, LossWeights
from utils.config import load_config, save_config
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


# ============================================================================
# 枚举与数据结构定义
# ============================================================================

class SMOptimizationStrategy(Enum):
    """SMO 优化策略枚举"""
    ALTERNATING = 'alternating'        # 交替优化：固定source优化mask → 固定mask优化source
    JOINT_GRADIENT = 'joint_gradient'  # 联合梯度下降：同时更新source和mask
    SOURCE_FIRST = 'source_first'      # 先优化source若干轮，再固定source优化mask


class SourceInitializationType(Enum):
    """光源初始化类型枚举"""
    CONVENTIONAL = 'conventional'      # 传统圆形照明
    ANNULAR = 'annular'                # 环形照明
    DIPOLE = 'dipole'                  # 偶极照明
    QUASAR = 'quasar'                  # 四极照明
    UNIFORM_DISK = 'uniform_disk'      # 均匀圆盘
    RANDOM = 'random'                  # 随机分布（约束后）
    CUSTOM = 'custom'                  # 自定义光源分布


@dataclass
class SourceConstraintsConfig:
    """
    光源约束配置

    Attributes:
        energy_conservation: 是否启用总能量守恒约束（归一化到1）
        energy_target: 目标总能量，默认为1.0
        sigma_target: 目标等效sigma值，None则不强制约束
        sigma_tolerance: sigma约束容差
        smoothness_weight: 平滑正则化权重（TV/L2平滑）
        smoothness_type: 平滑类型 'tv' 或 'gaussian'
        gaussian_sigma: 高斯平滑 sigma（像素单位），仅 smoothness_type='gaussian'
        non_negative: 是否强制光源强度非负
        support_radius: 光源最大支持半径（归一化sigma单位），None不限制
        support_radius_inner: 光源最小内半径（归一化sigma单位），用于环形约束
    """
    energy_conservation: bool = True
    energy_target: float = 1.0
    sigma_target: Optional[float] = None
    sigma_tolerance: float = 0.02
    smoothness_weight: float = 0.01
    smoothness_type: str = 'tv'
    gaussian_sigma: float = 1.5
    non_negative: bool = True
    support_radius: Optional[float] = None
    support_radius_inner: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'SourceConstraintsConfig':
        if d is None:
            return cls()
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            'energy_conservation': self.energy_conservation,
            'energy_target': self.energy_target,
            'sigma_target': self.sigma_target,
            'sigma_tolerance': self.sigma_tolerance,
            'smoothness_weight': self.smoothness_weight,
            'smoothness_type': self.smoothness_type,
            'gaussian_sigma': self.gaussian_sigma,
            'non_negative': self.non_negative,
            'support_radius': self.support_radius,
            'support_radius_inner': self.support_radius_inner,
        }


@dataclass
class SMOConfig:
    """
    SMO 工作流配置

    Attributes:
        strategy: 优化策略 (alternating / joint_gradient / source_first)
        max_outer_iterations: 外层交替优化最大迭代次数
        source_max_iter: 每轮光源优化最大迭代次数
        mask_max_iter: 每轮掩模优化最大迭代次数
        joint_max_iter: 联合梯度下降最大迭代次数（仅joint策略）
        source_learning_rate: 光源优化学习率
        mask_learning_rate: 掩模优化学习率
        joint_learning_rate_source: 联合优化时光源学习率
        joint_learning_rate_mask: 联合优化时掩模学习率
        tol: 收敛容差
        convergence_patience: 收敛耐心值

        source_grid_size: 像素化光源网格尺寸 (ny, nx)
        source_init_type: 光源初始化类型
        source_init_params: 光源初始化额外参数
        source_constraints: 光源约束配置

        wafer_threshold: 晶圆成像二值化阈值
        pixel_size: 像素尺寸 (nm)
        use_wafer_image_loss: 是否使用wafer图像计算损失

        mask_loss_weights: 掩模侧复合损失权重
        source_loss_weights: 光源侧损失权重（配合 EPE/MSE 等）

        process_conditions: 多工艺条件列表（离焦/剂量组合），None 则仅用标称条件
        pvb_weight: 工艺变化带宽（PVB）损失权重，仅多工艺条件时有效

        source_snapshot_freq: 光源快照保存频率（每N次外层迭代）
        verbose: 是否输出详细日志
    """
    strategy: SMOptimizationStrategy = SMOptimizationStrategy.ALTERNATING
    max_outer_iterations: int = 20
    source_max_iter: int = 50
    mask_max_iter: int = 100
    joint_max_iter: int = 200
    source_learning_rate: float = 0.005
    mask_learning_rate: float = 0.01
    joint_learning_rate_source: float = 0.003
    joint_learning_rate_mask: float = 0.008
    tol: float = 1e-5
    convergence_patience: int = 5

    source_grid_size: Optional[Tuple[int, int]] = None
    source_init_type: SourceInitializationType = SourceInitializationType.CONVENTIONAL
    source_init_params: Dict[str, Any] = field(default_factory=dict)
    source_constraints: SourceConstraintsConfig = field(default_factory=SourceConstraintsConfig)

    wafer_threshold: float = 0.3
    pixel_size: float = 1.0
    use_wafer_image_loss: bool = True

    mask_loss_weights: LossWeights = field(default_factory=LossWeights)
    source_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'mse': 1.0,
        'epe': 0.5,
    })

    process_conditions: Optional[List[Dict[str, float]]] = None
    pvb_weight: float = 0.0

    source_snapshot_freq: int = 1
    verbose: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'SMOConfig':
        if d is None:
            return cls()
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                if key == 'strategy':
                    cfg.strategy = SMOptimizationStrategy(value) if isinstance(value, str) else value
                elif key == 'source_init_type':
                    cfg.source_init_type = SourceInitializationType(value) if isinstance(value, str) else value
                elif key == 'source_constraints':
                    cfg.source_constraints = SourceConstraintsConfig.from_dict(value)
                elif key == 'mask_loss_weights':
                    cfg.mask_loss_weights = LossWeights.from_dict(value)
                elif key == 'source_grid_size' and value is not None:
                    cfg.source_grid_size = tuple(value)
                else:
                    setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'SMOConfig':
        config_dict = load_config(config_path)
        smo_config = config_dict.get('smo', config_dict)
        return cls.from_dict(smo_config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy.value,
            'max_outer_iterations': self.max_outer_iterations,
            'source_max_iter': self.source_max_iter,
            'mask_max_iter': self.mask_max_iter,
            'joint_max_iter': self.joint_max_iter,
            'source_learning_rate': self.source_learning_rate,
            'mask_learning_rate': self.mask_learning_rate,
            'joint_learning_rate_source': self.joint_learning_rate_source,
            'joint_learning_rate_mask': self.joint_learning_rate_mask,
            'tol': self.tol,
            'convergence_patience': self.convergence_patience,
            'source_grid_size': list(self.source_grid_size) if self.source_grid_size else None,
            'source_init_type': self.source_init_type.value,
            'source_init_params': dict(self.source_init_params),
            'source_constraints': self.source_constraints.to_dict(),
            'wafer_threshold': self.wafer_threshold,
            'pixel_size': self.pixel_size,
            'use_wafer_image_loss': self.use_wafer_image_loss,
            'mask_loss_weights': self.mask_loss_weights.to_dict(),
            'source_loss_weights': dict(self.source_loss_weights),
            'process_conditions': self.process_conditions,
            'pvb_weight': self.pvb_weight,
            'source_snapshot_freq': self.source_snapshot_freq,
            'verbose': self.verbose,
        }

    def to_yaml(self, config_path: Union[str, Path]) -> None:
        save_config({'smo': self.to_dict()}, config_path)


@dataclass
class SMOIterationResult:
    """
    SMO 单次外层迭代结果

    Attributes:
        iteration: 外层迭代次数
        phase: 优化阶段 ('source' / 'mask' / 'joint')
        loss_before: 优化前总损失
        loss_after: 优化后总损失
        source_before: 优化前光源
        source_after: 优化后光源
        mask_before: 优化前掩模
        mask_after: 优化后掩模
        aerial_before: 优化前空间像
        aerial_after: 优化后空间像
        wafer_before: 优化前晶圆图
        wafer_after: 优化后晶圆图
        epe_before: 优化前 EPE 统计
        epe_after: 优化后 EPE 统计
        source_effective_sigma: 优化后光源等效 sigma
        loss_components: 各损失分量明细
    """
    iteration: int
    phase: str
    loss_before: float
    loss_after: float
    source_before: np.ndarray
    source_after: np.ndarray
    mask_before: np.ndarray
    mask_after: np.ndarray
    aerial_before: np.ndarray
    aerial_after: np.ndarray
    wafer_before: np.ndarray
    wafer_after: np.ndarray
    epe_before: Dict[str, float]
    epe_after: Dict[str, float]
    source_effective_sigma: float
    loss_components: Dict[str, float] = field(default_factory=dict)

    @property
    def loss_improvement(self) -> float:
        return self.loss_before - self.loss_after

    @property
    def loss_improvement_ratio(self) -> float:
        if abs(self.loss_before) > 1e-12:
            return self.loss_improvement / abs(self.loss_before)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'phase': self.phase,
            'loss_before': self.loss_before,
            'loss_after': self.loss_after,
            'loss_improvement': self.loss_improvement,
            'loss_improvement_ratio': self.loss_improvement_ratio,
            'epe_before_mean': self.epe_before.get('epe_mean', 0.0),
            'epe_after_mean': self.epe_after.get('epe_mean', 0.0),
            'source_effective_sigma': self.source_effective_sigma,
            'loss_components': self.loss_components,
        }


@dataclass
class SMOWorkflowResult:
    """
    SMO 工作流最终结果

    Attributes:
        initial_mask: 初始掩模
        initial_source: 初始光源
        optimal_mask: 最优掩模
        optimal_source: 最优光源
        initial_wafer: 初始晶圆成像
        optimal_wafer: 最优晶圆成像
        initial_epe: 初始 EPE 统计
        final_epe: 最终 EPE 统计
        iterations: 所有迭代结果列表
        source_history: 光源演化历史（快照）
        mask_history: 掩模演化历史（快照）
        loss_history: 总损失收敛历史
        converged: 是否收敛
        reason: 终止原因
        total_time: 总耗时（秒）
    """
    initial_mask: np.ndarray
    initial_source: np.ndarray
    optimal_mask: np.ndarray
    optimal_source: np.ndarray
    initial_wafer: np.ndarray
    optimal_wafer: np.ndarray
    initial_epe: Dict[str, float]
    final_epe: Dict[str, float]
    iterations: List[SMOIterationResult] = field(default_factory=list)
    source_history: List[np.ndarray] = field(default_factory=list)
    mask_history: List[np.ndarray] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    converged: bool = False
    reason: str = ''
    total_time: float = 0.0

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    @property
    def total_epe_improvement(self) -> float:
        return self.initial_epe.get('epe_mean', 0.0) - self.final_epe.get('epe_mean', 0.0)

    @property
    def total_epe_improvement_ratio(self) -> float:
        init = self.initial_epe.get('epe_mean', 0.0)
        if init > 1e-12:
            return self.total_epe_improvement / init
        return 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            'initial_epe': self.initial_epe,
            'final_epe': self.final_epe,
            'total_epe_improvement': self.total_epe_improvement,
            'total_epe_improvement_ratio': self.total_epe_improvement_ratio,
            'num_iterations': self.num_iterations,
            'converged': self.converged,
            'reason': self.reason,
            'total_time': self.total_time,
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'initial_loss': self.loss_history[0] if self.loss_history else None,
        }


# ============================================================================
# 像素化光源模型
# ============================================================================

class PixelatedSource:
    """
    像素化光源模型

    将照明光源表示为 2D 强度分布，与掩模独立参数化。
    支持初始化、约束投影、等效 sigma 计算、梯度计算。
    """

    def __init__(self,
                 grid_size: Tuple[int, int],
                 optical_system: OpticalSystem,
                 init_type: SourceInitializationType = SourceInitializationType.CONVENTIONAL,
                 init_params: Optional[Dict[str, Any]] = None,
                 constraints: Optional[SourceConstraintsConfig] = None,
                 custom_source: Optional[np.ndarray] = None):
        """
        初始化像素化光源

        Args:
            grid_size: 光源网格尺寸 (ny, nx)
            optical_system: 光学系统参数（用于提供 cutoff 等）
            init_type: 光源初始化类型
            init_params: 初始化额外参数
            constraints: 光源约束配置
            custom_source: 自定义初始光源（当 init_type=CUSTOM 时）
        """
        self.grid_size = grid_size
        self.ny, self.nx = grid_size
        self.optics = optical_system
        self.init_type = init_type
        self.init_params = init_params or {}
        self.constraints = constraints or SourceConstraintsConfig()

        self._setup_frequency_grid()
        self._initialize(custom_source)

    def _setup_frequency_grid(self):
        """设置光源频率网格"""
        ny, nx = self.grid_size
        cutoff = self.optics.cutoff_frequency

        fx = np.fft.fftfreq(nx, self.optics.pixel_size)
        fy = np.fft.fftfreq(ny, self.optics.pixel_size)
        self.fx, self.fy = np.meshgrid(fx, fy)
        self.dfx = 1.0 / (nx * self.optics.pixel_size)
        self.dfy = 1.0 / (ny * self.optics.pixel_size)

        self.rho_norm = np.sqrt(self.fx ** 2 + self.fy ** 2) / cutoff
        self.theta = np.arctan2(self.fy, self.fx)

    def _initialize(self, custom_source: Optional[np.ndarray]):
        """初始化光源分布"""
        cutoff = self.optics.cutoff_frequency
        params = dict(self.init_params)

        if self.init_type == SourceInitializationType.CUSTOM and custom_source is not None:
            if custom_source.shape != self.grid_size:
                raise ValueError(
                    f"自定义光源形状 {custom_source.shape} 与网格尺寸 {self.grid_size} 不匹配"
                )
            self.intensity = custom_source.astype(np.float64).copy()

        elif self.init_type == SourceInitializationType.UNIFORM_DISK:
            sigma_val = params.get('sigma', self.optics.sigma)
            self.intensity = np.zeros(self.grid_size, dtype=np.float64)
            mask = self.rho_norm <= sigma_val
            self.intensity[mask] = 1.0

        elif self.init_type == SourceInitializationType.RANDOM:
            np.random.seed(params.get('seed', 42))
            self.intensity = np.random.rand(*self.grid_size).astype(np.float64)
            sigma_val = params.get('sigma', self.optics.sigma)
            self.intensity[self.rho_norm > sigma_val] = 0.0

        else:
            illum_map = {
                SourceInitializationType.CONVENTIONAL: IlluminationType.CONVENTIONAL,
                SourceInitializationType.ANNULAR: IlluminationType.ANNULAR,
                SourceInitializationType.DIPOLE: IlluminationType.DIPOLE,
                SourceInitializationType.QUASAR: IlluminationType.QUASAR,
            }
            illum_type = illum_map.get(self.init_type, IlluminationType.CONVENTIONAL)

            if not params:
                if self.init_type == SourceInitializationType.CONVENTIONAL:
                    params = {
                        'sigma_inner': 0.0,
                        'sigma_outer': self.optics.sigma
                    }
                elif self.init_type == SourceInitializationType.ANNULAR:
                    params = {
                        'sigma_inner': 0.6 * self.optics.sigma,
                        'sigma_outer': self.optics.sigma
                    }
                elif self.init_type == SourceInitializationType.DIPOLE:
                    params = {
                        'sigma_inner': 0.5 * self.optics.sigma,
                        'sigma_outer': 0.8 * self.optics.sigma,
                        'angle': 0.0,
                        'opening_angle': 60.0
                    }
                elif self.init_type == SourceInitializationType.QUASAR:
                    params = {
                        'sigma_inner': 0.5 * self.optics.sigma,
                        'sigma_outer': 0.8 * self.optics.sigma,
                        'angle': 45.0,
                        'opening_angle': 30.0
                    }

            self.intensity = generate_source(
                self.fx, self.fy, illum_type, params, cutoff
            )

        self.project_constraints()

    def project_constraints(self):
        """
        施加光源约束投影

        顺序：非负性 → 支持域约束 → sigma 等效约束 → 能量守恒 → 平滑
        """
        cfg = self.constraints

        if cfg.non_negative:
            self.intensity = np.clip(self.intensity, 0.0, None)

        if cfg.support_radius is not None:
            self.intensity[self.rho_norm > cfg.support_radius] = 0.0

        if cfg.support_radius_inner is not None:
            self.intensity[self.rho_norm < cfg.support_radius_inner] = 0.0

        if cfg.sigma_target is not None:
            self._project_sigma_constraint()

        if cfg.energy_conservation:
            total = np.sum(self.intensity)
            if total > 1e-15:
                self.intensity = self.intensity / total * cfg.energy_target
            else:
                center = (self.ny // 2, self.nx // 2)
                self.intensity[center] = cfg.energy_target

        if cfg.smoothness_weight > 0:
            self._apply_smoothing()

    def _project_sigma_constraint(self):
        """投影 sigma 等效约束（调整光源径向分布）"""
        cfg = self.constraints
        target_sigma = cfg.sigma_target
        tol = cfg.sigma_tolerance

        for _ in range(10):
            current_sigma = self.compute_effective_sigma()
            sigma_diff = current_sigma - target_sigma

            if abs(sigma_diff) <= tol:
                break

            scale_factor = target_sigma / max(current_sigma, 1e-8)

            from scipy.ndimage import map_coordinates
            cy, cx = self.ny // 2, self.nx // 2
            y_coords, x_coords = np.mgrid[0:self.ny, 0:self.nx]
            y_centered = y_coords - cy
            x_centered = x_coords - cx

            new_y = cy + y_centered / scale_factor
            new_x = cx + x_centered / scale_factor

            coords = np.vstack([new_y.ravel(), new_x.ravel()])
            scaled = map_coordinates(self.intensity, coords, order=1, mode='constant', cval=0.0)
            self.intensity = scaled.reshape(self.grid_size)

            self.intensity = np.clip(self.intensity, 0.0, None)

    def _apply_smoothing(self):
        """应用平滑正则化"""
        cfg = self.constraints
        weight = min(cfg.smoothness_weight, 1.0)

        if cfg.smoothness_type == 'gaussian':
            smoothed = gaussian_filter(self.intensity, sigma=cfg.gaussian_sigma)
            self.intensity = (1 - weight) * self.intensity + weight * smoothed
        else:
            kernel_size = 3
            from scipy.ndimage import uniform_filter
            smoothed = uniform_filter(self.intensity, size=kernel_size)
            self.intensity = (1 - weight) * self.intensity + weight * smoothed

    def compute_effective_sigma(self) -> float:
        """
        计算光源等效 sigma（基于一阶矩的加权平均）

        sigma_eff = ∫ S(f) * |f| / f_c df / ∫ S(f) df

        Returns:
            等效 sigma 值
        """
        total = np.sum(self.intensity)
        if total <= 1e-15:
            return 0.0
        weighted_rho = np.sum(self.intensity * self.rho_norm)
        return float(weighted_rho / total)

    def compute_smoothness_penalty(self) -> float:
        """
        计算平滑惩罚项（各向同性 TV）

        Returns:
            TV 平滑惩罚值
        """
        return float(total_variation_isotropic(self.intensity))

    def compute_smoothness_gradient(self) -> np.ndarray:
        """
        计算平滑惩罚梯度（各向同性 TV 梯度的向量化实现）

        Returns:
            梯度数组，与光源形状相同
        """
        intensity = self.intensity
        ny, nx = intensity.shape
        grad = np.zeros_like(intensity)

        dy_pos = np.zeros_like(intensity)
        dy_pos[:-1, :] = intensity[1:, :] - intensity[:-1, :]
        dy_neg = np.zeros_like(intensity)
        dy_neg[1:, :] = intensity[:-1, :] - intensity[1:, :]

        dx_pos = np.zeros_like(intensity)
        dx_pos[:, :-1] = intensity[:, 1:] - intensity[:, :-1]
        dx_neg = np.zeros_like(intensity)
        dx_neg[:, 1:] = intensity[:, :-1] - intensity[:, 1:]

        eps = 1e-8
        grad += dy_pos / np.sqrt(dy_pos ** 2 + eps)
        grad += dy_neg / np.sqrt(dy_neg ** 2 + eps)
        grad += dx_pos / np.sqrt(dx_pos ** 2 + eps)
        grad += dx_neg / np.sqrt(dx_neg ** 2 + eps)

        return grad

    def get_intensity(self) -> np.ndarray:
        """获取光源强度分布"""
        return self.intensity.copy()

    def set_intensity(self, new_intensity: np.ndarray, auto_project: bool = True):
        """
        设置光源强度

        Args:
            new_intensity: 新的光源强度
            auto_project: 是否自动施加约束投影
        """
        if new_intensity.shape != self.grid_size:
            raise ValueError(
                f"光源形状 {new_intensity.shape} 与网格尺寸 {self.grid_size} 不匹配"
            )
        self.intensity = new_intensity.astype(np.float64).copy()
        if auto_project:
            self.project_constraints()

    def get_visualization(self) -> np.ndarray:
        """获取 fftshift 后的光源分布，便于可视化"""
        return np.fft.fftshift(self.intensity)


# ============================================================================
# 联合成像前向模型
# ============================================================================

class SMOImagingModel:
    """
    联合成像前向模型

    扩展 PartialCoherentImaging，支持：
    - 像素化光源与掩模独立参数化
    - source ⊗ mask 的联合频域卷积（FULL_TCC / SOCS 形式）
    - 对光源和掩模的梯度计算
    - 动态更新光源并重新计算传递函数
    """

    def __init__(self,
                 optical_system: OpticalSystem,
                 image_size: Tuple[int, int],
                 window_type: Optional[Union[WindowType, str]] = None,
                 pad_width: Optional[Union[int, Tuple[int, int]]] = None,
                 tukey_alpha: float = 0.5,
                 tcc_mode: Optional[TCCMode] = None,
                 socs_num_terms: int = 8):
        """
        初始化联合成像模型

        Args:
            optical_system: 光学系统参数
            image_size: 掩模图像尺寸 (ny, nx)
            window_type: 窗函数类型
            pad_width: 零填充宽度
            tukey_alpha: Tukey 窗渐变因子
            tcc_mode: TCC 计算模式（覆盖 optical_system 中的设置）
            socs_num_terms: SOCS 分解项数
        """
        self.base_optics = optical_system
        self.image_size = image_size
        self.window_type = window_type
        self.pad_width = pad_width
        self.tukey_alpha = tukey_alpha
        self._tcc_mode_override = tcc_mode
        self._socs_num_terms_override = socs_num_terms

        self._imaging: Optional[PartialCoherentImaging] = None
        self._current_source: Optional[np.ndarray] = None

        self._rebuild_imaging(optical_system)

    def _rebuild_imaging(self, optics: OpticalSystem):
        """重建内部 PartialCoherentImaging 实例"""
        mode = self._tcc_mode_override or optics.tcc_mode
        socs_terms = self._socs_num_terms_override or optics.socs_num_terms

        new_optics = OpticalSystem(
            wavelength=optics.wavelength,
            na=optics.na,
            sigma=optics.sigma,
            pixel_size=optics.pixel_size,
            defocus=optics.defocus,
            magnification=optics.magnification,
            illumination_type=optics.illumination_type,
            source_params=dict(optics.source_params),
            tcc_mode=mode,
            socs_num_terms=socs_terms,
            custom_source=optics.custom_source,
            zernike_coefficients=dict(optics.zernike_coefficients)
        )

        self._imaging = PartialCoherentImaging(
            new_optics, self.image_size,
            window_type=self.window_type,
            pad_width=self.pad_width,
            tukey_alpha=self.tukey_alpha
        )
        self._current_optics = new_optics

    def update_source(self, pixelated_source: PixelatedSource):
        """
        更新光源并重新计算传递函数

        Args:
            pixelated_source: 像素化光源实例
        """
        source_intensity = pixelated_source.get_intensity()

        expected_shape = self._imaging.source.shape
        if source_intensity.shape != expected_shape:
            import scipy.ndimage as ndi
            zoom_factors = (
                expected_shape[0] / source_intensity.shape[0],
                expected_shape[1] / source_intensity.shape[1]
            )
            source_intensity = ndi.zoom(source_intensity, zoom_factors, order=1)
            total = np.sum(source_intensity)
            if total > 0:
                source_intensity = source_intensity / total

        self._imaging.update_source(source_intensity)
        self._current_source = self._imaging.source.copy()

    def update_defocus(self, defocus: float):
        """更新离焦量（用于多工艺条件）"""
        self._current_optics.defocus = defocus
        self._rebuild_imaging(self._current_optics)
        if self._current_source is not None:
            self._imaging.update_source(self._current_source)

    def compute_aerial_image(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像

        Args:
            mask: 掩模图案

        Returns:
            归一化空间像光强分布
        """
        return self._imaging.compute_aerial_image(mask)

    def compute_wafer_image(self,
                            mask: np.ndarray,
                            threshold: float = 0.3,
                            apply_resist_flag: bool = True,
                            resist_model: Optional[ResistModel] = None) -> np.ndarray:
        """
        计算晶圆成像

        Args:
            mask: 掩模图案
            threshold: 光刻胶阈值
            apply_resist_flag: 是否应用光刻胶响应
            resist_model: 高级光刻胶模型

        Returns:
            晶圆成像结果
        """
        aerial = self.compute_aerial_image(mask)
        if apply_resist_flag:
            return apply_resist_model(aerial, resist_model=resist_model, threshold=threshold)
        return aerial

    def compute_mask_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像对掩模的梯度

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        return self._imaging.compute_image_gradient(mask)

    def compute_source_gradient(self,
                                mask: np.ndarray,
                                dLoss_dAerial: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算损失对光源分布的梯度 dLoss/dS。

        根据 Hopkins 公式和链式法则：
            dLoss/dS(fs_i) = Σ_{x,y} dLoss/dI(x,y) · |FFT^{-1}[M(f) · P(f - fs_i)]|^2

        Args:
            mask: 掩模图案
            dLoss_dAerial: 损失对空间像的梯度 (H, W)，可选；None 时默认全1

        Returns:
            光源梯度数组，形状与光源相同
        """
        return self._imaging.compute_source_gradient(mask, dLoss_dAerial)

    def get_source(self) -> np.ndarray:
        """获取当前光源分布"""
        return self._imaging.source.copy()

    def get_source_visualization(self) -> np.ndarray:
        """获取 fftshift 后的光源用于可视化"""
        return self._imaging.get_source_image()

    def get_effective_tcc_kernel(self) -> Optional[np.ndarray]:
        """获取等效 TCC 核（2D对角近似）"""
        return self._imaging.tcc_kernel.copy() if self._imaging.tcc_kernel is not None else None


# ============================================================================
# 光源优化器
# ============================================================================

class SourceOptimizer:
    """
    光源梯度优化器

    固定掩模，通过梯度下降优化像素化光源分布。
    支持：约束投影、平滑正则化、学习率调度。
    """

    def __init__(self,
                 imaging_model: SMOImagingModel,
                 config: SMOConfig):
        """
        初始化光源优化器

        Args:
            imaging_model: 联合成像模型
            config: SMO 配置
        """
        self.imaging = imaging_model
        self.config = config

    def _compute_loss(self,
                      mask: np.ndarray,
                      target: np.ndarray,
                      pixelated_source: PixelatedSource) -> Tuple[float, Dict[str, float], np.ndarray]:
        """
        计算损失与损失对空间像的梯度

        Returns:
            (total_loss, components, dLoss_dAerial) 三元组
        """
        self.imaging.update_source(pixelated_source)

        aerial = self.imaging.compute_aerial_image(mask)

        if self.config.use_wafer_image_loss:
            k = 50.0
            wafer = 1.0 / (1.0 + np.exp(-k * (aerial - self.config.wafer_threshold)))
            diff = wafer - target
            dLoss_dWafer = 2.0 * diff / diff.size
            dWafer_dAerial = k * wafer * (1.0 - wafer)
            dLoss_dAerial = dLoss_dWafer * dWafer_dAerial
            mse_val = float(np.mean(diff ** 2))
        else:
            diff = aerial - target
            dLoss_dAerial = 2.0 * diff / diff.size
            mse_val = float(np.mean(diff ** 2))

        w = self.config.source_loss_weights
        total_loss = w.get('mse', 1.0) * mse_val

        components = {'mse': mse_val}

        if w.get('epe', 0.0) > 0:
            wafer_binary = (aerial >= self.config.wafer_threshold).astype(np.float64)
            epe_stats = compute_epe(wafer_binary, target, pixel_size=self.config.pixel_size)
            epe_mean = epe_stats.get('epe_mean', 0.0)
            total_loss += w['epe'] * epe_mean
            components['epe'] = epe_mean

        if pixelated_source.constraints.smoothness_weight > 0:
            tv_val = pixelated_source.compute_smoothness_penalty()
            tv_w = pixelated_source.constraints.smoothness_weight
            total_loss += tv_w * tv_val
            components['source_smoothness_tv'] = tv_val

        return total_loss, components, dLoss_dAerial

    def optimize(self,
                 initial_source: PixelatedSource,
                 mask: np.ndarray,
                 target: np.ndarray,
                 max_iter: Optional[int] = None,
                 learning_rate: Optional[float] = None) -> Tuple[PixelatedSource, List[float]]:
        """
        执行光源优化（固定掩模）

        Args:
            initial_source: 初始像素化光源
            mask: 固定的掩模
            target: 目标图案
            max_iter: 覆盖配置中的最大迭代次数
            learning_rate: 覆盖配置中的学习率

        Returns:
            (优化后的光源, 损失历史)
        """
        max_iter = max_iter or self.config.source_max_iter
        lr = learning_rate or self.config.source_learning_rate

        source = initial_source
        loss_history = []
        best_loss = float('inf')
        best_source_intensity = source.get_intensity().copy()
        patience_counter = 0

        for it in range(max_iter):
            loss_val, components, dLoss_dAerial = self._compute_loss(mask, target, source)
            loss_history.append(loss_val)

            if loss_val < best_loss - self.config.tol:
                best_loss = loss_val
                best_source_intensity = source.get_intensity().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.convergence_patience:
                    if self.config.verbose:
                        logger.info(f"  光源优化在第 {it+1} 次迭代提前收敛（耐心值耗尽）")
                    break

            source_grad_raw = self.imaging.compute_source_gradient(mask, dLoss_dAerial)

            if source.constraints.smoothness_weight > 0:
                smooth_grad = source.compute_smoothness_gradient()
                source_grad = source_grad_raw + source.constraints.smoothness_weight * smooth_grad
            else:
                source_grad = source_grad_raw

            grad_norm = np.max(np.abs(source_grad)) + 1e-12
            normalized_grad = source_grad / grad_norm

            new_intensity = source.intensity - lr * normalized_grad
            source.set_intensity(new_intensity, auto_project=True)

            if self.config.verbose and (it + 1) % max(1, max_iter // 5) == 0:
                sigma_eff = source.compute_effective_sigma()
                logger.info(f"  光源迭代 {it+1}/{max_iter}: loss={loss_val:.6f}, "
                           f"sigma_eff={sigma_eff:.4f}, components={components}")

        source.set_intensity(best_source_intensity, auto_project=True)
        return source, loss_history


# ============================================================================
# 掩模优化器（封装 MaskOptimizer）
# ============================================================================

class MaskOptimizerForSMO:
    """
    为 SMO 工作流封装的掩模优化器

    固定当前光源，使用 MaskOptimizer 优化掩模。
    """

    def __init__(self,
                 imaging_model: SMOImagingModel,
                 config: SMOConfig):
        """
        初始化掩模优化器

        Args:
            imaging_model: 联合成像模型
            config: SMO 配置
        """
        self.imaging = imaging_model
        self.config = config
        self._current_source_intensity: Optional[np.ndarray] = None

    def optimize(self,
                 initial_mask: np.ndarray,
                 target: np.ndarray,
                 pixelated_source: PixelatedSource,
                 max_iter: Optional[int] = None,
                 learning_rate: Optional[float] = None) -> Tuple[np.ndarray, List[float]]:
        """
        执行掩模优化（固定光源）

        Args:
            initial_mask: 初始掩模
            target: 目标图案
            pixelated_source: 固定的光源
            max_iter: 覆盖配置中的最大迭代次数
            learning_rate: 覆盖配置中的学习率

        Returns:
            (优化后的掩模, 损失历史)
        """
        self.imaging.update_source(pixelated_source)
        self._current_source_intensity = pixelated_source.get_intensity()

        max_iter = max_iter or self.config.mask_max_iter
        lr = learning_rate or self.config.mask_learning_rate

        source_for_optics = self.imaging.get_source()

        smo_optics = OpticalSystem(
            wavelength=self.imaging._current_optics.wavelength,
            na=self.imaging._current_optics.na,
            sigma=self.imaging._current_optics.sigma,
            pixel_size=self.imaging._current_optics.pixel_size,
            defocus=self.imaging._current_optics.defocus,
            magnification=self.imaging._current_optics.magnification,
            illumination_type=IlluminationType.CUSTOM,
            source_params={},
            tcc_mode=TCCMode.SOCS,
            socs_num_terms=self.imaging._current_optics.socs_num_terms,
            custom_source=source_for_optics,
            zernike_coefficients=dict(self.imaging._current_optics.zernike_coefficients)
        )

        opt_config = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=max_iter,
            learning_rate=lr,
            use_composite_loss=True,
            loss_weights=self.config.mask_loss_weights,
            tol=self.config.tol,
            early_stop_patience=self.config.convergence_patience,
            threshold=self.config.wafer_threshold,
            use_wafer_image_loss=self.config.use_wafer_image_loss,
            pixel_size=self.config.pixel_size,
            verbose=False
        )

        optimizer = MaskOptimizer(
            optical_system=smo_optics,
            config=opt_config
        )

        try:
            result = optimizer.optimize(initial_mask, target)
            optimized_mask = result.optimized_mask
            loss_history = result.loss_history if hasattr(result, 'loss_history') and result.loss_history else [getattr(result, 'final_loss', float('inf'))]
        except Exception as e:
            logger.warning(f"MaskOptimizer 失败，回退到简单梯度下降: {e}")
            optimized_mask, loss_history = self._simple_gradient_mask(
                initial_mask, target, max_iter, lr
            )

        optimized_mask = np.clip(optimized_mask, 0.0, 1.0)
        return optimized_mask, loss_history

    def _simple_gradient_mask(self,
                              initial_mask: np.ndarray,
                              target: np.ndarray,
                              max_iter: int,
                              lr: float) -> Tuple[np.ndarray, List[float]]:
        """回退方案：简单梯度下降优化掩模"""
        mask = initial_mask.copy()
        loss_history = []

        for it in range(max_iter):
            aerial = self.imaging.compute_aerial_image(mask)
            diff = aerial - target
            loss_val = float(np.mean(diff ** 2))
            loss_history.append(loss_val)

            grad = self.imaging.compute_mask_gradient(mask)
            dLoss_dMask = 2.0 * grad * (np.sum(grad * diff) / (np.sum(grad * grad) + 1e-12))
            mask = mask - lr * dLoss_dMask
            mask = np.clip(mask, 0.0, 1.0)

        return mask, loss_history


# ============================================================================
# 联合梯度下降优化器
# ============================================================================

class JointGradientOptimizer:
    """
    联合梯度下降优化器

    同时更新光源和掩模，根据联合损失的梯度分别对两者下降。
    """

    def __init__(self,
                 imaging_model: SMOImagingModel,
                 config: SMOConfig):
        self.imaging = imaging_model
        self.config = config

    def _compute_joint_loss(self,
                            mask: np.ndarray,
                            target: np.ndarray,
                            pixelated_source: PixelatedSource) -> Tuple[float, Dict[str, float], np.ndarray]:
        self.imaging.update_source(pixelated_source)
        aerial = self.imaging.compute_aerial_image(mask)

        if self.config.use_wafer_image_loss:
            k = 50.0
            wafer = 1.0 / (1.0 + np.exp(-k * (aerial - self.config.wafer_threshold)))
            diff = wafer - target
            dLoss_dWafer = 2.0 * diff / diff.size
            dWafer_dAerial = k * wafer * (1.0 - wafer)
            dLoss_dAerial = dLoss_dWafer * dWafer_dAerial
            mse_val = float(np.mean(diff ** 2))
        else:
            diff = aerial - target
            dLoss_dAerial = 2.0 * diff / diff.size
            mse_val = float(np.mean(diff ** 2))

        total_loss = self.config.mask_loss_weights.mse * mse_val
        components = {'mse': mse_val}

        if pixelated_source.constraints.smoothness_weight > 0:
            tv_val = pixelated_source.compute_smoothness_penalty()
            total_loss += pixelated_source.constraints.smoothness_weight * tv_val
            components['source_tv'] = tv_val

        return total_loss, components, dLoss_dAerial

    def optimize(self,
                 initial_source: PixelatedSource,
                 initial_mask: np.ndarray,
                 target: np.ndarray,
                 max_iter: Optional[int] = None) -> Tuple[PixelatedSource, np.ndarray, List[float]]:
        """
        执行联合梯度下降

        Returns:
            (优化后光源, 优化后掩模, 损失历史)
        """
        max_iter = max_iter or self.config.joint_max_iter
        lr_s = self.config.joint_learning_rate_source
        lr_m = self.config.joint_learning_rate_mask

        source = initial_source
        mask = initial_mask.copy()
        loss_history = []
        best_loss = float('inf')
        best_source_intensity = source.get_intensity().copy()
        best_mask = mask.copy()
        patience_counter = 0

        for it in range(max_iter):
            loss_val, components, dLoss_dAerial = self._compute_joint_loss(mask, target, source)
            loss_history.append(loss_val)

            if loss_val < best_loss - self.config.tol:
                best_loss = loss_val
                best_source_intensity = source.get_intensity().copy()
                best_mask = mask.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.convergence_patience:
                    if self.config.verbose:
                        logger.info(f"  联合优化在第 {it+1} 次迭代提前收敛")
                    break

            mask_grad = self.imaging.compute_mask_gradient(mask)
            m_grad_norm = np.max(np.abs(mask_grad)) + 1e-12
            mask = mask - lr_m * mask_grad / m_grad_norm
            mask = np.clip(mask, 0.0, 1.0)

            source_grad_raw = self.imaging.compute_source_gradient(mask, dLoss_dAerial)
            s_grad_norm = np.max(np.abs(source_grad_raw)) + 1e-12

            if source.constraints.smoothness_weight > 0:
                smooth_grad = source.compute_smoothness_gradient()
                total_s_grad = source_grad_raw / s_grad_norm + source.constraints.smoothness_weight * smooth_grad
            else:
                total_s_grad = source_grad_raw / s_grad_norm

            new_intensity = source.intensity - lr_s * total_s_grad
            source.set_intensity(new_intensity, auto_project=True)

            if self.config.verbose and (it + 1) % max(1, max_iter // 5) == 0:
                sigma_eff = source.compute_effective_sigma()
                logger.info(f"  联合迭代 {it+1}/{max_iter}: loss={loss_val:.6f}, "
                           f"sigma_eff={sigma_eff:.4f}, components={components}")

        source.set_intensity(best_source_intensity, auto_project=True)
        return source, best_mask, loss_history


# ============================================================================
# SMO 工作流主类
# ============================================================================

class SMOWorkflow:
    """
    SMO (Source-Mask Optimization) 工作流主类

    封装完整的光源-掩模协同优化流程：
    1. 像素化光源初始化
    2. 交替优化 / 联合梯度下降
    3. 约束投影与收敛检查
    4. 历史记录与结果封装

    典型使用：
        workflow = SMOWorkflow(config=smo_config, optical_system=optics)
        result = workflow.run(initial_mask, target)
    """

    def __init__(self,
                 config: Optional[SMOConfig] = None,
                 optical_system: Optional[OpticalSystem] = None):
        """
        初始化 SMO 工作流

        Args:
            config: SMO 配置，None 则使用默认
            optical_system: 基础光学系统参数，None 则使用默认
        """
        self.config = config or SMOConfig()
        self.base_optics = optical_system or OpticalSystem()

        self._source_optimizer: Optional[SourceOptimizer] = None
        self._mask_optimizer: Optional[MaskOptimizerForSMO] = None
        self._joint_optimizer: Optional[JointGradientOptimizer] = None
        self._imaging: Optional[SMOImagingModel] = None

    def _evaluate_state(self,
                        mask: np.ndarray,
                        target: np.ndarray,
                        pixelated_source: PixelatedSource) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, Dict[str, float]]:
        """评估当前状态：计算总损失、空间像、晶圆图、EPE统计"""
        self._imaging.update_source(pixelated_source)
        aerial = self._imaging.compute_aerial_image(mask)
        wafer_cont = simulate_wafer_image(
            mask,
            optical_system=self._imaging._current_optics,
            threshold=self.config.wafer_threshold,
            apply_resist=True
        )
        wafer_binary = (wafer_cont >= self.config.wafer_threshold).astype(np.float64)

        diff = aerial - target
        mse_val = float(np.mean(diff ** 2))

        w = self.config.mask_loss_weights
        total_loss = w.mse * mse_val

        epe_stats = compute_epe(wafer_binary, target, pixel_size=self.config.pixel_size)
        if w.epe > 0:
            total_loss += w.epe * epe_stats.get('epe_mean', 0.0)

        if pixelated_source.constraints.smoothness_weight > 0:
            tv_val = pixelated_source.compute_smoothness_penalty()
            total_loss += pixelated_source.constraints.smoothness_weight * tv_val

        components = {
            'mse': mse_val,
            'epe_mean': epe_stats.get('epe_mean', 0.0),
            'source_sigma': pixelated_source.compute_effective_sigma(),
        }

        if self.config.process_conditions and self.config.pvb_weight > 0:
            pvb_val = self._compute_pvb(mask, target, pixelated_source)
            total_loss += self.config.pvb_weight * pvb_val
            components['pvb'] = pvb_val

        return total_loss, components, aerial, wafer_binary, epe_stats

    def _compute_pvb(self,
                     mask: np.ndarray,
                     target: np.ndarray,
                     pixelated_source: PixelatedSource) -> float:
        """
        计算工艺变化带宽（PVB）

        在多工艺条件下计算晶圆成像的最大-最小差异（带宽），
        用于最大化工艺窗口。

        Args:
            mask: 当前掩模
            target: 目标图案
            pixelated_source: 像素化光源

        Returns:
            PVB 值（平均工艺变化带宽）
        """
        if not self.config.process_conditions:
            return 0.0

        wafer_images = []
        for cond_dict in self.config.process_conditions:
            cond = ProcessCondition(
                defocus=cond_dict.get('defocus', 0.0),
                dose=cond_dict.get('dose', 1.0),
                weight=cond_dict.get('weight', 1.0),
            )
            cond_optics = cond.to_optical_system(base_optics=self._imaging._current_optics)
            wafer_i = simulate_wafer_image(
                mask,
                optical_system=cond_optics,
                threshold=self.config.wafer_threshold,
                apply_resist=True
            )
            wafer_images.append(wafer_i)

        if len(wafer_images) < 2:
            return 0.0

        wafer_stack = np.stack(wafer_images, axis=0)
        pvb_map = np.max(wafer_stack, axis=0) - np.min(wafer_stack, axis=0)
        return float(np.mean(pvb_map))

    def run(self,
            initial_mask: np.ndarray,
            target: np.ndarray) -> SMOWorkflowResult:
        """
        运行完整的 SMO 工作流

        Args:
            initial_mask: 初始掩模图案
            target: 目标版图图案

        Returns:
            SMOWorkflowResult 结果对象
        """
        import time
        start_time = time.time()

        image_size = initial_mask.shape
        if self.config.verbose:
            logger.info("=" * 70)
            logger.info("SMO (Source-Mask Optimization) 工作流启动")
            logger.info("=" * 70)
            logger.info(f"  策略: {self.config.strategy.value}")
            logger.info(f"  掩模尺寸: {image_size}")
            logger.info(f"  最大外层迭代: {self.config.max_outer_iterations}")

        source_grid = self.config.source_grid_size or image_size

        self._imaging = SMOImagingModel(
            self.base_optics, image_size,
            tcc_mode=TCCMode.SOCS,
            socs_num_terms=max(self.base_optics.socs_num_terms, 8)
        )

        pixelated_source = PixelatedSource(
            grid_size=source_grid,
            optical_system=self.base_optics,
            init_type=self.config.source_init_type,
            init_params=self.config.source_init_params,
            constraints=self.config.source_constraints
        )

        self._source_optimizer = SourceOptimizer(self._imaging, self.config)
        self._mask_optimizer = MaskOptimizerForSMO(self._imaging, self.config)
        self._joint_optimizer = JointGradientOptimizer(self._imaging, self.config)

        initial_source_arr = pixelated_source.get_intensity().copy()
        initial_mask_arr = initial_mask.copy()
        current_mask = initial_mask.copy()

        init_loss, _, init_aerial, init_wafer, init_epe = self._evaluate_state(
            current_mask, target, pixelated_source
        )

        iterations: List[SMOIterationResult] = []
        source_history: List[np.ndarray] = [pixelated_source.get_visualization()]
        mask_history: List[np.ndarray] = [current_mask.copy()]
        loss_history: List[float] = [init_loss]

        best_loss = init_loss
        best_source_intensity = pixelated_source.get_intensity().copy()
        best_mask = current_mask.copy()
        patience_counter = 0
        converged = False
        reason = ''

        if self.config.verbose:
            logger.info(f"\n初始状态: loss={init_loss:.6f}, EPE_mean={init_epe.get('epe_mean', 0):.3f}nm")
            logger.info(f"  等效 sigma: {pixelated_source.compute_effective_sigma():.4f}")

        strategy = self.config.strategy

        if strategy == SMOptimizationStrategy.SOURCE_FIRST:
            if self.config.verbose:
                logger.info(f"\n阶段 1/2: 单独优化光源 ({self.config.source_max_iter} 次迭代)...")
            pixelated_source, _ = self._source_optimizer.optimize(
                pixelated_source, current_mask, target
            )
            source_history.append(pixelated_source.get_visualization())

            if self.config.verbose:
                logger.info(f"阶段 2/2: 固定光源，优化掩模...")

        for outer_iter in range(self.config.max_outer_iterations):
            if self.config.verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"外层迭代 {outer_iter + 1}/{self.config.max_outer_iterations}")
                logger.info(f"{'='*60}")

            mask_before = current_mask.copy()
            source_before = pixelated_source.get_intensity().copy()

            loss_before, _, aerial_before, wafer_before, epe_before = self._evaluate_state(
                current_mask, target, pixelated_source
            )

            if strategy in (SMOptimizationStrategy.ALTERNATING, SMOptimizationStrategy.SOURCE_FIRST):

                if self.config.verbose:
                    logger.info("  [子阶段 1/2] 固定掩模 → 优化光源...")
                pixelated_source, source_loss_hist = self._source_optimizer.optimize(
                    pixelated_source, current_mask, target
                )

                loss_after_s, _, aerial_after_s, wafer_after_s, epe_after_s = self._evaluate_state(
                    current_mask, target, pixelated_source
                )

                source_iter_result = SMOIterationResult(
                    iteration=outer_iter * 2,
                    phase='source',
                    loss_before=loss_before,
                    loss_after=loss_after_s,
                    source_before=source_before,
                    source_after=pixelated_source.get_intensity(),
                    mask_before=mask_before,
                    mask_after=current_mask,
                    aerial_before=aerial_before,
                    aerial_after=aerial_after_s,
                    wafer_before=wafer_before,
                    wafer_after=wafer_after_s,
                    epe_before=epe_before,
                    epe_after=epe_after_s,
                    source_effective_sigma=pixelated_source.compute_effective_sigma(),
                    loss_components={'phase': 'source', 'sub_loss': source_loss_hist[-1] if source_loss_hist else loss_after_s}
                )
                iterations.append(source_iter_result)

                if self.config.verbose:
                    logger.info(f"    光源优化完成: loss {loss_before:.6f} → {loss_after_s:.6f} "
                               f"(改善 {source_iter_result.loss_improvement_ratio*100:.1f}%)")
                    logger.info(f"    EPE: {epe_before.get('epe_mean', 0):.3f} → {epe_after_s.get('epe_mean', 0):.3f} nm")
                    logger.info(f"    等效 sigma: {pixelated_source.compute_effective_sigma():.4f}")

                if self.config.verbose:
                    logger.info("  [子阶段 2/2] 固定光源 → 优化掩模...")

                mask_loss_before = loss_after_s
                current_mask, mask_loss_hist = self._mask_optimizer.optimize(
                    current_mask, target, pixelated_source
                )

                loss_after_m, components_m, aerial_after_m, wafer_after_m, epe_after_m = self._evaluate_state(
                    current_mask, target, pixelated_source
                )

                mask_iter_result = SMOIterationResult(
                    iteration=outer_iter * 2 + 1,
                    phase='mask',
                    loss_before=mask_loss_before,
                    loss_after=loss_after_m,
                    source_before=pixelated_source.get_intensity(),
                    source_after=pixelated_source.get_intensity(),
                    mask_before=mask_before,
                    mask_after=current_mask,
                    aerial_before=aerial_after_s,
                    aerial_after=aerial_after_m,
                    wafer_before=wafer_after_s,
                    wafer_after=wafer_after_m,
                    epe_before=epe_after_s,
                    epe_after=epe_after_m,
                    source_effective_sigma=pixelated_source.compute_effective_sigma(),
                    loss_components=components_m
                )
                iterations.append(mask_iter_result)

                current_total_loss = loss_after_m

                if self.config.verbose:
                    logger.info(f"    掩模优化完成: loss {mask_loss_before:.6f} → {loss_after_m:.6f} "
                               f"(改善 {mask_iter_result.loss_improvement_ratio*100:.1f}%)")
                    logger.info(f"    EPE: {epe_after_s.get('epe_mean', 0):.3f} → {epe_after_m.get('epe_mean', 0):.3f} nm")

            else:
                if self.config.verbose:
                    logger.info("  [联合梯度下降] 同时更新光源和掩模...")

                pixelated_source, current_mask, joint_loss_hist = self._joint_optimizer.optimize(
                    pixelated_source, current_mask, target
                )

                loss_after_j, components_j, aerial_after_j, wafer_after_j, epe_after_j = self._evaluate_state(
                    current_mask, target, pixelated_source
                )

                joint_iter_result = SMOIterationResult(
                    iteration=outer_iter,
                    phase='joint',
                    loss_before=loss_before,
                    loss_after=loss_after_j,
                    source_before=source_before,
                    source_after=pixelated_source.get_intensity(),
                    mask_before=mask_before,
                    mask_after=current_mask,
                    aerial_before=aerial_before,
                    aerial_after=aerial_after_j,
                    wafer_before=wafer_before,
                    wafer_after=wafer_after_j,
                    epe_before=epe_before,
                    epe_after=epe_after_j,
                    source_effective_sigma=pixelated_source.compute_effective_sigma(),
                    loss_components=components_j
                )
                iterations.append(joint_iter_result)
                current_total_loss = loss_after_j

                if self.config.verbose:
                    logger.info(f"    联合优化完成: loss {loss_before:.6f} → {loss_after_j:.6f} "
                               f"(改善 {joint_iter_result.loss_improvement_ratio*100:.1f}%)")
                    logger.info(f"    EPE: {epe_before.get('epe_mean', 0):.3f} → {epe_after_j.get('epe_mean', 0):.3f} nm")

            loss_history.append(current_total_loss)

            if (outer_iter + 1) % self.config.source_snapshot_freq == 0:
                source_history.append(pixelated_source.get_visualization())
                mask_history.append(current_mask.copy())

            if current_total_loss < best_loss - self.config.tol:
                best_loss = current_total_loss
                best_source_intensity = pixelated_source.get_intensity().copy()
                best_mask = current_mask.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if self.config.verbose:
                    logger.info(f"  连续 {patience_counter} 轮无显著改善（耐心值 {self.config.convergence_patience}）")

                if patience_counter >= self.config.convergence_patience:
                    converged = True
                    reason = (f"外层迭代 {outer_iter+1}: 连续 {patience_counter} 轮"
                             f"损失改善低于阈值 {self.config.tol}")
                    if self.config.verbose:
                        logger.info(f"  收敛触发: {reason}")
                    break

        if not converged:
            reason = f"达到最大外层迭代次数 {self.config.max_outer_iterations}"

        pixelated_source.set_intensity(best_source_intensity, auto_project=True)
        self._imaging.update_source(pixelated_source)

        final_aerial = self._imaging.compute_aerial_image(best_mask)
        final_wafer_cont = simulate_wafer_image(
            best_mask,
            optical_system=self._imaging._current_optics,
            threshold=self.config.wafer_threshold,
            apply_resist=True
        )
        final_wafer = (final_wafer_cont >= self.config.wafer_threshold).astype(np.float64)
        final_epe = compute_epe(final_wafer, target, pixel_size=self.config.pixel_size)

        total_time = time.time() - start_time

        result = SMOWorkflowResult(
            initial_mask=initial_mask_arr,
            initial_source=initial_source_arr,
            optimal_mask=best_mask,
            optimal_source=best_source_intensity,
            initial_wafer=init_wafer,
            optimal_wafer=final_wafer,
            initial_epe=init_epe,
            final_epe=final_epe,
            iterations=iterations,
            source_history=source_history,
            mask_history=mask_history,
            loss_history=loss_history,
            converged=converged,
            reason=reason,
            total_time=total_time
        )

        if self.config.verbose:
            logger.info(f"\n{'='*70}")
            logger.info("SMO 工作流完成")
            logger.info(f"{'='*70}")
            logger.info(f"  收敛: {'是' if converged else '否'} — {reason}")
            logger.info(f"  总耗时: {total_time:.2f} 秒")
            logger.info(f"  外层迭代: {len([it for it in iterations if it.phase in ('alternating_outer','joint','source')])}")
            logger.info(f"  初始 EPE: {init_epe.get('epe_mean', 0):.3f} nm → "
                       f"最终 EPE: {final_epe.get('epe_mean', 0):.3f} nm")
            logger.info(f"  EPE 改善: {result.total_epe_improvement:.3f} nm "
                       f"({result.total_epe_improvement_ratio*100:.1f}%)")
            logger.info(f"  初始损失: {loss_history[0]:.6f} → 最终损失: {best_loss:.6f}")

        return result


# ============================================================================
# 便捷入口函数
# ============================================================================

def run_smo_workflow(initial_mask: np.ndarray,
                     target: np.ndarray,
                     config: Optional[SMOConfig] = None,
                     optical_system: Optional[OpticalSystem] = None) -> SMOWorkflowResult:
    """
    便捷入口：运行 SMO 工作流

    Args:
        initial_mask: 初始掩模图案
        target: 目标版图
        config: SMO 配置，None 则使用默认
        optical_system: 光学系统参数，None 则使用默认

    Returns:
        SMOWorkflowResult 结果对象
    """
    workflow = SMOWorkflow(config=config, optical_system=optical_system)
    return workflow.run(initial_mask, target)
