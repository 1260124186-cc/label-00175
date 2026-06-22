# -*- coding: utf-8 -*-
"""
水平集拓扑优化器：整合成像链、SIMP 材料、H-J 演化与拓扑约束

LevelSetTopologyOptimizer 是与梯度投影 ILT 平行的独立方法论：
  - ILT：像素级 [0,1] 参数化 + 梯度投影 + 离散量化
  - Level Set：连续域 SDF 参数化 + 边界演化 + 天然光滑边界

优化流程：
  1. 初始化：从目标图案构造水平集 φ₀
  2. 前向传播：φ → ρ = H_ε(φ) → mask → aerial → wafer
  3. 损失计算：L = w_mse·MSE + w_epe·EPE + w_perimeter·P + ...
  4. 反向传播：dL/dmask → dL/dρ → dL/dφ（SIMP 链式求导）
  5. 速度计算：V_n = -(dL/dφ)/|∇φ| + V_constraint
  6. 水平集演化：φ^{n+1} = φ^n - Δt·V_n·|∇φ|
  7. 重初始化 + 约束执行 → 回到 2

与 MaskOptimizer/ILTWorkflow 的接口一致性：
  - 相同的输入：initial_mask, target, optical_system
  - 相同的输出：TopologyOptimizationResult（兼容 MaskOptimizationResult）
  - 可通过配置切换优化方法
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from core.imaging import (
    OpticalSystem, PartialCoherentImaging,
    simulate_wafer_image, ProcessCondition,
)
from core.metrics import (
    mse, mae, ssim, evaluate_all,
    total_variation_isotropic, total_variation_isotropic_gradient,
    edge_placement_error,
)
from core.litho_metrics import compute_epe
from topology.level_set import LevelSetFunction
from topology.simp_material import SIMPMaterialModel
from topology.hamilton_jacobi import (
    HamiltonJacobiEvolver, ShapeVelocityCalculator,
)
from topology.constraints import TopologyConstraints

logger = logging.getLogger(__name__)


class TopologyMethod(Enum):
    """拓扑优化方法枚举"""
    LEVEL_SET = 'level_set'
    LEVEL_SET_SIMP = 'level_set_simp'


@dataclass
class TopologyOptConfig:
    """
    水平集拓扑优化配置

    Attributes:
        max_iter: 最大迭代次数
        convergence_tol: 收敛容差
        convergence_patience: 收敛耐心值

        method: 拓扑优化方法
        dx: 网格间距

        heaviside_epsilon: Heaviside 投影正则化参数
        heaviside_decay_schedule: Heaviside epsilon 衰减策略
            - 'constant': 固定不变
            - 'linear': 线性衰减
            - 'exponential': 指数衰减
        heaviside_decay_rate: 衰减率

        simp_penalty: SIMP 惩罚指数
        simp_min_density: SIMP 最小密度
        simp_projection_method: Heaviside 投影方法

        evolution_dt: 演化时间步长，None 时 CFL 自适应
        evolution_cfl: CFL 安全系数
        reinit_interval: 重初始化间隔
        reinit_iters: 重初始化子迭代数

        curvature_weight: 曲率平滑权重
        min_feature_width: 最小特征宽度（0=不约束）
        min_feature_weight: 最小特征尺寸约束权重
        perimeter_target: 目标周长
        perimeter_weight: 周长约束权重
        volume_fraction: 目标面积比例
        volume_weight: 面积约束权重

        morphological_enforce_interval: 形态学约束执行间隔（0=不执行）
        morphological_enforce_iters: 形态学约束迭代次数

        loss_mse_weight: MSE 损失权重
        loss_epe_weight: EPE 损失权重
        loss_tv_weight: TV 正则化权重
        loss_binary_weight: 二值化惩罚权重
        wafer_threshold: 光刻胶阈值
        resist_steepness: soft resist sigmoid 陡度

        filter_radius: 灵敏度过滤半径
        filter_method: 过滤方法

        verbose: 是否输出详细日志
    """
    max_iter: int = 200
    convergence_tol: float = 1e-6
    convergence_patience: int = 20

    method: TopologyMethod = TopologyMethod.LEVEL_SET_SIMP
    dx: float = 1.0

    heaviside_epsilon: float = 2.0
    heaviside_decay_schedule: str = 'exponential'
    heaviside_decay_rate: float = 0.98

    simp_penalty: float = 3.0
    simp_min_density: float = 1e-3
    simp_projection_method: str = 'arctan'

    evolution_dt: Optional[float] = None
    evolution_cfl: float = 0.45
    reinit_interval: int = 5
    reinit_iters: int = 5

    curvature_weight: float = 0.05
    min_feature_width: float = 0.0
    min_feature_weight: float = 1.0
    perimeter_target: Optional[float] = None
    perimeter_weight: float = 0.1
    volume_fraction: Optional[float] = None
    volume_weight: float = 1.0

    morphological_enforce_interval: int = 0
    morphological_enforce_iters: int = 1

    loss_mse_weight: float = 1.0
    loss_epe_weight: float = 0.0
    loss_tv_weight: float = 0.0
    loss_binary_weight: float = 0.0
    wafer_threshold: float = 0.3
    resist_steepness: float = 50.0

    filter_radius: float = 0.0
    filter_method: str = 'gaussian'

    verbose: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'TopologyOptConfig':
        if d is None:
            return cls()
        cfg = cls()
        field_names = {f.name for f in cfg.__dataclass_fields__.values()}
        for key, value in d.items():
            if key in field_names:
                if key == 'method' and isinstance(value, str):
                    setattr(cfg, key, TopologyMethod(value))
                else:
                    setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for f in self.__dataclass_fields__.values():
            val = getattr(self, f.name)
            if isinstance(val, Enum):
                result[f.name] = val.value
            else:
                result[f.name] = val
        return result


@dataclass
class TopologyIterationResult:
    """单次迭代结果"""
    iteration: int
    loss: float
    loss_components: Dict[str, float]
    mask: np.ndarray
    wafer_continuous: np.ndarray
    velocity_norm: float = 0.0
    phi_range: Tuple[float, float] = (0.0, 0.0)
    epsilon_current: float = 1.0
    constraint_contributions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'loss': self.loss,
            'loss_components': self.loss_components,
            'velocity_norm': self.velocity_norm,
            'phi_range': self.phi_range,
            'epsilon_current': self.epsilon_current,
            'constraint_contributions': self.constraint_contributions,
        }


@dataclass
class TopologyOptimizationResult:
    """
    水平集拓扑优化结果

    兼容 MaskOptimizationResult 的接口，可无缝替换 ILT/像素优化。
    """
    optimized_mask: np.ndarray
    initial_mask: np.ndarray
    target_image: np.ndarray
    final_wafer_image: np.ndarray
    initial_wafer_image: np.ndarray
    final_loss: float
    initial_loss: float
    loss_history: List[float]
    total_iterations: int
    total_time: float
    converged: bool
    message: str
    final_level_set: Optional[LevelSetFunction] = None
    iterations: List[TopologyIterationResult] = field(default_factory=list)
    mask_history: List[np.ndarray] = field(default_factory=list)

    @property
    def loss_improvement(self) -> float:
        return self.initial_loss - self.final_loss

    @property
    def loss_improvement_ratio(self) -> float:
        if abs(self.initial_loss) > 1e-12:
            return self.loss_improvement / abs(self.initial_loss)
        return 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            'initial_loss': self.initial_loss,
            'final_loss': self.final_loss,
            'loss_improvement': self.loss_improvement,
            'loss_improvement_ratio': self.loss_improvement_ratio,
            'total_iterations': self.total_iterations,
            'converged': self.converged,
            'message': self.message,
            'total_time': self.total_time,
        }


class LevelSetTopologyOptimizer:
    """
    水平集拓扑优化器

    与 ILTWorkflow/GradientProjector 并行的方法论：
    - ILT: 像素级 [0,1] 参数化 → 梯度投影 → 离散量化
    - Level Set: 连续域 SDF 参数化 → 边界演化 → 天然光滑

    关键优势：
    1. 天然光滑边界（零水平集是 C¹ 连续曲线）
    2. 最小特征尺寸通过曲率流/形态学约束直接保证
    3. 拓扑灵活性（自动处理孔洞生成/合并）
    4. 物理意义明确（边界法向速度 = 形状灵敏度）
    """

    def __init__(self,
                 optical_system: Optional[OpticalSystem] = None,
                 config: Optional[TopologyOptConfig] = None):
        self.optical_system = optical_system or OpticalSystem()
        self.config = config or TopologyOptConfig()

        self._imaging: Optional[PartialCoherentImaging] = None
        self._material_model: Optional[SIMPMaterialModel] = None
        self._evolver: Optional[HamiltonJacobiEvolver] = None
        self._velocity_calc: Optional[ShapeVelocityCalculator] = None
        self._constraints: Optional[TopologyConstraints] = None
        self._target: Optional[np.ndarray] = None

    def _setup(self, image_size: Tuple[int, int]):
        cfg = self.config
        dx = cfg.dx

        self._imaging = PartialCoherentImaging(self.optical_system, image_size)

        self._material_model = SIMPMaterialModel(
            penalty=cfg.simp_penalty,
            epsilon=cfg.heaviside_epsilon,
            min_density=cfg.simp_min_density,
            projection_method=cfg.simp_projection_method,
            filter_radius=cfg.filter_radius,
            filter_method=cfg.filter_method,
            dx=dx,
        )

        self._evolver = HamiltonJacobiEvolver(
            cfl_number=cfg.evolution_cfl,
            reinit_interval=cfg.reinit_interval,
            reinit_iters=cfg.reinit_iters,
            curvature_weight=0.0,
            max_dt=1.0,
            dx=dx,
        )

        self._velocity_calc = ShapeVelocityCalculator(
            normalize=True,
            extension_method='gradient',
            dx=dx,
        )

        self._constraints = TopologyConstraints(
            curvature_weight=cfg.curvature_weight,
            min_feature_width=cfg.min_feature_width,
            min_feature_weight=cfg.min_feature_weight,
            perimeter_target=cfg.perimeter_target,
            perimeter_weight=cfg.perimeter_weight,
            volume_fraction=cfg.volume_fraction,
            volume_weight=cfg.volume_weight,
            dx=dx,
        )

    def _compute_current_epsilon(self, iteration: int) -> float:
        cfg = self.config
        eps0 = cfg.heaviside_epsilon
        schedule = cfg.heaviside_decay_schedule
        rate = cfg.heaviside_decay_rate

        if schedule == 'constant' or iteration == 0:
            return eps0
        elif schedule == 'linear':
            progress = min(iteration / max(cfg.max_iter, 1), 1.0)
            return max(eps0 * (1.0 - 0.8 * progress), 0.1)
        elif schedule == 'exponential':
            return max(eps0 * (rate ** iteration), 0.1)
        return eps0

    def _compute_loss(self,
                      mask: np.ndarray,
                      target: np.ndarray,
                      wafer: np.ndarray) -> Tuple[float, Dict[str, float]]:
        cfg = self.config
        components: Dict[str, float] = {}
        total = 0.0

        if cfg.loss_mse_weight > 0:
            mse_val = float(mse(wafer, target))
            components['mse'] = mse_val
            total += cfg.loss_mse_weight * mse_val

        if cfg.loss_epe_weight > 0:
            threshold = cfg.wafer_threshold
            wafer_bin = (wafer >= threshold).astype(np.float64)
            target_bin = (target >= threshold).astype(np.float64)
            epe_stats = compute_epe(wafer_bin, target_bin, pixel_size=cfg.dx)
            epe_mean = epe_stats.get('epe_mean', 0.0)
            components['epe'] = epe_mean
            total += cfg.loss_epe_weight * epe_mean

        if cfg.loss_tv_weight > 0:
            tv_val = float(total_variation_isotropic(mask))
            components['tv'] = tv_val
            total += cfg.loss_tv_weight * tv_val

        if cfg.loss_binary_weight > 0:
            bp = float(np.mean(4.0 * mask * (1.0 - mask)))
            components['binary_penalty'] = bp
            total += cfg.loss_binary_weight * bp

        components['total'] = total
        return total, components

    def _compute_loss_gradient(self,
                               mask: np.ndarray,
                               target: np.ndarray,
                               wafer: np.ndarray,
                               aerial: np.ndarray) -> np.ndarray:
        cfg = self.config
        N = mask.shape[0] * mask.shape[1]
        grad = np.zeros_like(mask)

        if cfg.loss_mse_weight > 0:
            k = cfg.resist_steepness
            t = cfg.wafer_threshold
            wafer_soft = 1.0 / (1.0 + np.exp(-np.clip(k * (aerial - t), -500, 500)))
            dW_dI = k * wafer_soft * (1.0 - wafer_soft)
            dI_dM = self._imaging.compute_image_gradient(mask)
            dL_dW = 2.0 * cfg.loss_mse_weight * (wafer_soft - target) / N
            grad += dL_dW * dW_dI * dI_dM

        if cfg.loss_tv_weight > 0:
            grad += cfg.loss_tv_weight * total_variation_isotropic_gradient(mask)

        if cfg.loss_binary_weight > 0:
            grad += cfg.loss_binary_weight * 4.0 * (1.0 - 2.0 * mask) / N

        return grad

    def optimize(self,
                 initial_mask: np.ndarray,
                 target: np.ndarray) -> TopologyOptimizationResult:
        """
        执行水平集拓扑优化

        Args:
            initial_mask: 初始掩模图案
            target: 目标图案（晶圆上期望的二值图）

        Returns:
            TopologyOptimizationResult
        """
        start_time = time.time()
        cfg = self.config
        dx = cfg.dx

        image_size = initial_mask.shape
        self._setup(image_size)
        self._target = target

        ls = LevelSetFunction.from_binary_mask(
            initial_mask, smooth_sigma=1.0, dx=dx
        )

        density = ls.to_density(epsilon=cfg.heaviside_epsilon)
        initial_aerial = self._imaging.compute_aerial_image(density)
        k = cfg.resist_steepness
        t = cfg.wafer_threshold
        initial_wafer = 1.0 / (1.0 + np.exp(-np.clip(k * (initial_aerial - t), -500, 500)))

        initial_loss, initial_components = self._compute_loss(
            density, target, initial_wafer
        )

        result = TopologyOptimizationResult(
            optimized_mask=density.copy(),
            initial_mask=initial_mask.copy(),
            target_image=target.copy(),
            final_wafer_image=initial_wafer.copy(),
            initial_wafer_image=initial_wafer.copy(),
            final_loss=initial_loss,
            initial_loss=initial_loss,
            loss_history=[initial_loss],
            total_iterations=0,
            total_time=0.0,
            converged=False,
            message='',
            final_level_set=ls.copy(),
        )

        best_loss = initial_loss
        best_ls = ls.copy()
        best_mask = density.copy()
        best_wafer = initial_wafer.copy()
        patience_counter = 0
        prev_loss = initial_loss

        if cfg.verbose:
            logger.info(
                f"水平集拓扑优化开始: max_iter={cfg.max_iter}, "
                f"method={cfg.method.value}, dx={dx}, "
                f"ε₀={cfg.heaviside_epsilon}"
            )
            logger.info(f"初始损失: {initial_loss:.6f}")

        for iteration in range(1, cfg.max_iter + 1):
            current_epsilon = self._compute_current_epsilon(iteration)
            self._material_model.epsilon = current_epsilon

            density = ls.to_density(epsilon=current_epsilon)
            mask_for_imaging = np.clip(density, 0.0, 1.0)

            aerial = self._imaging.compute_aerial_image(mask_for_imaging)
            wafer = 1.0 / (1.0 + np.exp(-np.clip(k * (aerial - t), -500, 500)))

            current_loss, loss_components = self._compute_loss(
                mask_for_imaging, target, wafer
            )

            dL_dmask = self._compute_loss_gradient(
                mask_for_imaging, target, wafer, aerial
            )

            dL_dphi = self._material_model.density_gradient_to_phi(
                dL_dmask, ls.phi
            )

            v_shape = self._velocity_calc.compute(dL_dphi, ls)

            v_constraint, constraint_contributions = (
                self._constraints.compute_constraint_velocity(ls)
            )

            v_total = v_shape + v_constraint

            v_norm = float(np.sqrt(np.mean(v_total ** 2)))

            ls = self._evolver.evolve(ls, v_total, n_steps=1)

            if (cfg.morphological_enforce_interval > 0
                    and iteration % cfg.morphological_enforce_interval == 0):
                self._constraints.enforce_min_feature_morphological(
                    ls, cfg.morphological_enforce_iters
                )

            density_new = ls.to_density(epsilon=current_epsilon)
            mask_new = np.clip(density_new, 0.0, 1.0)

            iter_result = TopologyIterationResult(
                iteration=iteration,
                loss=current_loss,
                loss_components=loss_components,
                mask=mask_new.copy(),
                wafer_continuous=wafer.copy(),
                velocity_norm=v_norm,
                phi_range=(float(ls.phi.min()), float(ls.phi.max())),
                epsilon_current=current_epsilon,
                constraint_contributions=constraint_contributions,
            )
            result.iterations.append(iter_result)
            result.loss_history.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                best_ls = ls.copy()
                best_mask = mask_new.copy()
                best_wafer = wafer.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if abs(prev_loss - current_loss) < cfg.convergence_tol:
                patience_counter += 1

            if patience_counter >= cfg.convergence_patience:
                result.converged = True
                result.message = (
                    f"收敛：连续 {cfg.convergence_patience} 次 "
                    f"损失改善 < {cfg.convergence_tol}"
                )
                if cfg.verbose:
                    logger.info(f"迭代 {iteration}: {result.message}")
                break

            prev_loss = current_loss

            if cfg.verbose and iteration % max(1, cfg.max_iter // 20) == 0:
                logger.info(
                    f"迭代 {iteration}/{cfg.max_iter}: "
                    f"loss={current_loss:.6f}, "
                    f"v_norm={v_norm:.4f}, "
                    f"ε={current_epsilon:.3f}, "
                    f"φ=[{ls.phi.min():.1f},{ls.phi.max():.1f}]"
                )

        result.optimized_mask = best_mask
        result.final_wafer_image = best_wafer
        result.final_loss = best_loss
        result.final_level_set = best_ls
        result.total_iterations = len(result.iterations)
        result.total_time = time.time() - start_time

        if not result.converged:
            result.message = f"达到最大迭代次数 {cfg.max_iter}"

        if cfg.verbose:
            logger.info(f"水平集拓扑优化完成: {result.message}")
            logger.info(
                f"损失: {initial_loss:.6f} → {best_loss:.6f} "
                f"(改善 {initial_loss - best_loss:.6f})"
            )
            logger.info(f"总耗时: {result.total_time:.2f}s")

        return result


def run_topology_optimization(
        initial_mask: np.ndarray,
        target: np.ndarray,
        optical_system: Optional[OpticalSystem] = None,
        config: Optional[TopologyOptConfig] = None) -> TopologyOptimizationResult:
    """
    水平集拓扑优化便捷入口函数

    Args:
        initial_mask: 初始掩模图案
        target: 目标图案
        optical_system: 光学系统参数
        config: 拓扑优化配置

    Returns:
        TopologyOptimizationResult
    """
    optimizer = LevelSetTopologyOptimizer(
        optical_system=optical_system,
        config=config,
    )
    return optimizer.optimize(initial_mask, target)
