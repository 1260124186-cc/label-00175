# -*- coding: utf-8 -*-
"""
ILT (Inverse Lithography Technology) 反演光刻工作流模块

实现反演光刻技术的完整流程，区别于通用 MaskOptimizer 的黑盒优化，
采用 ILT 特有的梯度投影与掩模量化策略。

核心组件：
    1. DifferentiableImagingChain: 可微成像链，确保从掩模到晶圆的全链路可微
       （含 soft resist、可微阈值）
    2. GradientProjector: 梯度投影优化，每步梯度更新后投影到 [0,1]，
       并可选投影到离散透射率等级
    3. MaskComplexityPenalty: 掩模复杂度控制，在目标函数中显式加入
       掩模周长、顶点数、辅助特征数量惩罚
    4. MultiObjectiveILT: 多目标 ILT，同时优化多个目标图案，
       加权求和
    5. ILTConfig: ILT 工作流配置
    6. ILTWorkflow: 完整 ILT 工作流封装
    7. run_ilt_workflow: 便捷入口函数
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
from scipy.ndimage import (
    gaussian_filter, binary_dilation, binary_erosion,
    label, find_objects, generate_binary_structure, sobel
)

from core.imaging import (
    OpticalSystem, PartialCoherentImaging,
    ProcessCondition, simulate_wafer_image,
    ResistModel, apply_resist_model,
)
from core.metrics import (
    mse, mae, total_variation_isotropic,
    total_variation_isotropic_gradient,
)
from core.litho_metrics import compute_epe, extract_edges
from utils.config import load_config, save_config

logger = logging.getLogger(__name__)


class TransmissionLevel(Enum):
    """离散透射率等级枚举"""
    BINARY = 'binary'              # 0/1 二值
    TERNARY = 'ternary'            # 0/0.5/1 三值
    CONTINUOUS = 'continuous'      # 连续 [0,1]


class ILTOptimizerType(Enum):
    """ILT 优化器类型枚举"""
    GRADIENT_PROJECTION = 'gradient_projection'
    ADAM_PROJECTION = 'adam_projection'
    SGD_PROJECTION = 'sgd_projection'


class ILTLossComponent(Enum):
    """ILT 损失分量枚举"""
    MSE = 'mse'
    L2_WAFER = 'l2_wafer'
    PERIMETER = 'perimeter'
    VERTEX_COUNT = 'vertex_count'
    SUB_FEATURE_COUNT = 'sub_feature_count'
    BINARY_PENALTY = 'binary_penalty'
    TV_SMOOTH = 'tv_smooth'


@dataclass
class ILTComplexityConfig:
    """
    掩模复杂度惩罚配置

    Attributes:
        perimeter_weight: 掩模周长惩罚权重
        vertex_weight: 顶点数惩罚权重
        sub_feature_weight: 辅助特征数量惩罚权重
        sub_feature_min_area: 辅助特征最小面积阈值（像素）
        sub_feature_max_area: 辅助特征最大面积阈值（像素）
    """
    perimeter_weight: float = 0.0
    vertex_weight: float = 0.0
    sub_feature_weight: float = 0.0
    sub_feature_min_area: int = 2
    sub_feature_max_area: int = 100

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'ILTComplexityConfig':
        if d is None:
            return cls()
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            'perimeter_weight': self.perimeter_weight,
            'vertex_weight': self.vertex_weight,
            'sub_feature_weight': self.sub_feature_weight,
            'sub_feature_min_area': self.sub_feature_min_area,
            'sub_feature_max_area': self.sub_feature_max_area,
        }


@dataclass
class ILTConfig:
    """
    ILT 工作流配置

    Attributes:
        max_iter: 最大迭代次数
        learning_rate: 学习率
        optimizer_type: 优化器类型
        convergence_tol: 收敛容差
        convergence_patience: 收敛耐心值

        transmission_level: 离散透射率等级
        quantization_start_iter: 开始量化的迭代数（延迟量化，前期连续优化）
        quantization_schedule: 量化调度类型
            - 'step': 一步切换到离散
            - 'linear': 线性渐增量化强度
            - 'cosine': 余弦渐增量化强度
        quantization_strength: 量化强度（0~1），1表示完全量化

        resist_steepness: soft resist sigmoid 陡度参数 k
        wafer_threshold: 光刻胶阈值

        l2_wafer_weight: 晶圆图 L2 损失权重
        complexity: 掩模复杂度惩罚配置
        binary_penalty_weight: 二值化惩罚权重
        tv_smooth_weight: TV 平滑权重

        multi_objective_conditions: 多目标工艺条件列表，
            每项为 dict 含 'defocus', 'dose', 'weight' 键
        multi_objective_targets: 多目标图案列表（不同 focus 下的目标），
            None 则所有条件共用同一 target

        pixel_size: 像素尺寸 (nm)
        verbose: 是否输出详细日志
    """
    max_iter: int = 200
    learning_rate: float = 0.01
    optimizer_type: ILTOptimizerType = ILTOptimizerType.ADAM_PROJECTION
    convergence_tol: float = 1e-6
    convergence_patience: int = 20

    transmission_level: TransmissionLevel = TransmissionLevel.CONTINUOUS
    quantization_start_iter: int = 100
    quantization_schedule: str = 'linear'
    quantization_strength: float = 1.0

    resist_steepness: float = 50.0
    wafer_threshold: float = 0.3

    l2_wafer_weight: float = 1.0
    complexity: ILTComplexityConfig = field(default_factory=ILTComplexityConfig)
    binary_penalty_weight: float = 0.0
    tv_smooth_weight: float = 0.0

    multi_objective_conditions: Optional[List[Dict[str, float]]] = None
    multi_objective_targets: Optional[List[np.ndarray]] = None

    pixel_size: float = 1.0
    verbose: bool = True
    imaging_model: Optional[Any] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'ILTConfig':
        if d is None:
            return cls()
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key) and key != 'imaging_model':
                if key == 'optimizer_type':
                    cfg.optimizer_type = ILTOptimizerType(value) if isinstance(value, str) else value
                elif key == 'transmission_level':
                    cfg.transmission_level = TransmissionLevel(value) if isinstance(value, str) else value
                elif key == 'complexity':
                    cfg.complexity = ILTComplexityConfig.from_dict(value)
                else:
                    setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'ILTConfig':
        config_dict = load_config(config_path)
        ilt_config = config_dict.get('ilt', config_dict)
        return cls.from_dict(ilt_config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate,
            'optimizer_type': self.optimizer_type.value,
            'convergence_tol': self.convergence_tol,
            'convergence_patience': self.convergence_patience,
            'transmission_level': self.transmission_level.value,
            'quantization_start_iter': self.quantization_start_iter,
            'quantization_schedule': self.quantization_schedule,
            'quantization_strength': self.quantization_strength,
            'resist_steepness': self.resist_steepness,
            'wafer_threshold': self.wafer_threshold,
            'l2_wafer_weight': self.l2_wafer_weight,
            'complexity': self.complexity.to_dict(),
            'binary_penalty_weight': self.binary_penalty_weight,
            'tv_smooth_weight': self.tv_smooth_weight,
            'multi_objective_conditions': self.multi_objective_conditions,
            'pixel_size': self.pixel_size,
            'verbose': self.verbose,
        }

    def to_yaml(self, config_path: Union[str, Path]) -> None:
        save_config({'ilt': self.to_dict()}, config_path)


@dataclass
class ILTIterationResult:
    """
    ILT 单次迭代结果

    Attributes:
        iteration: 迭代次数
        loss: 总损失
        loss_components: 各损失分量明细
        mask: 当前掩模
        wafer_continuous: 当前连续晶圆图
        wafer_binary: 当前二值化晶圆图
        gradient_norm: 梯度范数
        learning_rate: 当前学习率
        quantization_strength_current: 当前实际量化强度
    """
    iteration: int
    loss: float
    loss_components: Dict[str, float]
    mask: np.ndarray
    wafer_continuous: np.ndarray
    wafer_binary: np.ndarray
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    quantization_strength_current: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'loss': self.loss,
            'loss_components': self.loss_components,
            'gradient_norm': self.gradient_norm,
            'learning_rate': self.learning_rate,
            'quantization_strength_current': self.quantization_strength_current,
        }


@dataclass
class ILTWorkflowResult:
    """
    ILT 工作流最终结果

    Attributes:
        initial_mask: 初始掩模
        optimal_mask: 最优掩模
        initial_wafer: 初始晶圆成像
        optimal_wafer: 最优晶圆成像
        initial_epe: 初始 EPE 统计
        final_epe: 最终 EPE 统计
        initial_loss: 初始总损失
        final_loss: 最终总损失
        iterations: 所有迭代结果列表
        mask_history: 掩模演化历史
        loss_history: 损失收敛历史
        converged: 是否收敛
        reason: 终止原因
        total_time: 总耗时（秒）
        final_quantization_strength: 最终量化强度
    """
    initial_mask: np.ndarray
    optimal_mask: np.ndarray
    initial_wafer: np.ndarray
    optimal_wafer: np.ndarray
    initial_epe: Dict[str, float]
    final_epe: Dict[str, float]
    initial_loss: float = 0.0
    final_loss: float = 0.0
    iterations: List[ILTIterationResult] = field(default_factory=list)
    mask_history: List[np.ndarray] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    converged: bool = False
    reason: str = ''
    total_time: float = 0.0
    final_quantization_strength: float = 0.0

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    @property
    def total_loss_improvement(self) -> float:
        return self.initial_loss - self.final_loss

    @property
    def total_loss_improvement_ratio(self) -> float:
        if abs(self.initial_loss) > 1e-12:
            return self.total_loss_improvement / abs(self.initial_loss)
        return 0.0

    @property
    def total_epe_improvement(self) -> float:
        return self.initial_epe.get('epe_mean', 0.0) - self.final_epe.get('epe_mean', 0.0)

    def summary(self) -> Dict[str, Any]:
        return {
            'initial_epe': self.initial_epe,
            'final_epe': self.final_epe,
            'total_epe_improvement': self.total_epe_improvement,
            'initial_loss': self.initial_loss,
            'final_loss': self.final_loss,
            'total_loss_improvement': self.total_loss_improvement,
            'total_loss_improvement_ratio': self.total_loss_improvement_ratio,
            'num_iterations': self.num_iterations,
            'converged': self.converged,
            'reason': self.reason,
            'total_time': self.total_time,
            'final_quantization_strength': self.final_quantization_strength,
        }


class DifferentiableImagingChain:
    """
    可微成像链

    确保从掩模到晶圆的全链路可微，包括：
    1. 光学成像：通过 PartialCoherentImaging 计算空间像
    2. Soft Resist：使用 sigmoid 函数近似光刻胶响应
    3. 可微阈值：通过 sigmoid 实现可微的二值化

    全链路梯度：
        dL/dM = dL/dW · dW/dI · dI/dM
        其中 W = sigmoid(k·(I - t)), I = imaging(M)
    """

    def __init__(self,
                 optical_system: OpticalSystem,
                 image_size: Tuple[int, int],
                 resist_steepness: float = 50.0,
                 wafer_threshold: float = 0.3,
                 imaging_model: Optional[Any] = None):
        """
        初始化可微成像链

        Args:
            optical_system: 光学系统参数
            image_size: 图像尺寸 (ny, nx)
            resist_steepness: sigmoid 陡度参数
            wafer_threshold: 光刻胶阈值
            imaging_model: 可选的外部成像模型（如自适应代理模型），
                需实现 compute_aerial_image(mask) 接口
        """
        self.optical_system = optical_system
        self.image_size = image_size
        self.resist_steepness = resist_steepness
        self.wafer_threshold = wafer_threshold

        if imaging_model is not None:
            self._imaging = imaging_model
        else:
            self._imaging = PartialCoherentImaging(
                optical_system, image_size
            )

    def forward(self, mask: np.ndarray, dose: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播：掩模 → 空间像 → 晶圆图

        Args:
            mask: 掩模图案
            dose: 曝光剂量比例

        Returns:
            (aerial_image, wafer_continuous) 空间像与连续晶圆图
        """
        aerial = self._imaging.compute_aerial_image(mask)
        if dose != 1.0:
            aerial = np.clip(aerial * dose, 0.0, None)
        wafer = self.soft_resist(aerial)
        return aerial, wafer

    def soft_resist(self, aerial_image: np.ndarray) -> np.ndarray:
        """
        Soft resist：使用 sigmoid 近似光刻胶显影

        W = 1 / (1 + exp(-k · (I - t)))

        Args:
            aerial_image: 空间像光强

        Returns:
            连续晶圆图（0~1之间，可微）
        """
        k = self.resist_steepness
        t = self.wafer_threshold
        z = k * (aerial_image - t)
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_wafer_gradient(self, aerial_image: np.ndarray) -> np.ndarray:
        """
        计算 dW/dI（soft resist 对空间像的梯度）

        dW/dI = k · W · (1 - W)

        Args:
            aerial_image: 空间像光强

        Returns:
            梯度数组
        """
        wafer = self.soft_resist(aerial_image)
        k = self.resist_steepness
        return k * wafer * (1.0 - wafer)

    def compute_mask_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算 dI/dM（空间像对掩模的梯度）

        通过 PartialCoherentImaging 的 Hopkins 梯度计算。

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        return self._imaging.compute_image_gradient(mask)

    def compute_full_gradient(self,
                              mask: np.ndarray,
                              dL_dW: np.ndarray,
                              dose: float = 1.0) -> np.ndarray:
        """
        计算全链路梯度 dL/dM = dL/dW · dW/dI · dI/dM

        Args:
            mask: 掩模图案
            dL_dW: 损失对晶圆图的梯度
            dose: 曝光剂量比例

        Returns:
            损失对掩模的梯度
        """
        aerial, wafer = self.forward(mask, dose)
        dW_dI = self.compute_wafer_gradient(aerial)
        dI_dM = self.compute_mask_gradient(mask)
        if dose != 1.0:
            dW_dI = dW_dI * dose
        dL_dI = dL_dW * dW_dI
        return dL_dI * dI_dM


class GradientProjector:
    """
    梯度投影优化器

    每步梯度更新后执行投影操作：
    1. 连续投影：clip 到 [0, 1]
    2. 离散投影：投影到最近的透射率等级

    支持延迟量化（前期连续优化，后期逐步量化），
    通过 quantization_schedule 控制量化强度渐进增加。
    """

    def __init__(self, config: ILTConfig):
        """
        初始化梯度投影优化器

        Args:
            config: ILT 配置
        """
        self.config = config
        self._adam_m: Optional[np.ndarray] = None
        self._adam_v: Optional[np.ndarray] = None
        self._adam_t: int = 0
        self._sgd_velocity: Optional[np.ndarray] = None

    def project_continuous(self, mask: np.ndarray) -> np.ndarray:
        """
        连续投影：裁剪到 [0, 1]

        Args:
            mask: 掩模

        Returns:
            投影后的掩模
        """
        return np.clip(mask, 0.0, 1.0)

    def project_discrete(self,
                         mask: np.ndarray,
                         strength: float = 1.0) -> np.ndarray:
        """
        离散投影：投影到最近的透射率等级

        使用软量化实现可微近似：
            Q(M) = (1-α)·M + α·nearest_level(M)
        其中 α = strength 控制量化程度。

        支持的透射率等级：
        - BINARY: {0, 1}
        - TERNARY: {0, 0.5, 1}
        - CONTINUOUS: 不做离散投影

        Args:
            mask: 掩模
            strength: 量化强度 (0~1)

        Returns:
            量化投影后的掩模
        """
        if strength <= 0.0:
            return mask

        level = self.config.transmission_level

        if level == TransmissionLevel.CONTINUOUS:
            return mask

        if level == TransmissionLevel.BINARY:
            quantized = np.where(mask < 0.5, 0.0, 1.0)
        elif level == TransmissionLevel.TERNARY:
            quantized = np.zeros_like(mask)
            quantized[mask < 0.25] = 0.0
            quantized[(mask >= 0.25) & (mask < 0.75)] = 0.5
            quantized[mask >= 0.75] = 1.0
        else:
            return mask

        blended = (1.0 - strength) * mask + strength * quantized
        return np.clip(blended, 0.0, 1.0)

    def compute_quantization_strength(self, iteration: int) -> float:
        """
        根据迭代次数和调度策略计算当前量化强度

        Args:
            iteration: 当前迭代次数

        Returns:
            当前量化强度 (0~1)
        """
        cfg = self.config
        start = cfg.quantization_start_iter
        max_strength = cfg.quantization_strength

        if iteration < start:
            return 0.0

        if cfg.transmission_level == TransmissionLevel.CONTINUOUS:
            return 0.0

        progress = (iteration - start) / max(cfg.max_iter - start, 1)
        progress = min(progress, 1.0)

        if cfg.quantization_schedule == 'step':
            return max_strength
        elif cfg.quantization_schedule == 'linear':
            return max_strength * progress
        elif cfg.quantization_schedule == 'cosine':
            return max_strength * 0.5 * (1.0 - np.cos(np.pi * progress))
        else:
            return max_strength * progress

    def step(self,
             mask: np.ndarray,
             gradient: np.ndarray,
             iteration: int) -> np.ndarray:
        """
        执行一步梯度投影更新

        Args:
            mask: 当前掩模
            gradient: 损失对掩模的梯度
            iteration: 当前迭代次数

        Returns:
            更新后的掩模
        """
        lr = self.config.learning_rate
        opt_type = self.config.optimizer_type

        if opt_type == ILTOptimizerType.ADAM_PROJECTION:
            updated = self._adam_step(mask, gradient, lr)
        elif opt_type == ILTOptimizerType.SGD_PROJECTION:
            updated = self._sgd_step(mask, gradient, lr)
        else:
            updated = mask - lr * gradient

        updated = self.project_continuous(updated)

        q_strength = self.compute_quantization_strength(iteration)
        updated = self.project_discrete(updated, q_strength)

        return updated

    def _adam_step(self,
                   mask: np.ndarray,
                   gradient: np.ndarray,
                   lr: float) -> np.ndarray:
        """
        Adam 优化器一步

        Args:
            mask: 当前掩模
            gradient: 梯度
            lr: 学习率

        Returns:
            更新后的掩模
        """
        if self._adam_m is None:
            self._adam_m = np.zeros_like(mask)
            self._adam_v = np.zeros_like(mask)
            self._adam_t = 0

        self._adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        self._adam_m = beta1 * self._adam_m + (1 - beta1) * gradient
        self._adam_v = beta2 * self._adam_v + (1 - beta2) * gradient ** 2

        m_hat = self._adam_m / (1 - beta1 ** self._adam_t)
        v_hat = self._adam_v / (1 - beta2 ** self._adam_t)

        return mask - lr * m_hat / (np.sqrt(v_hat) + eps)

    def _sgd_step(self,
                  mask: np.ndarray,
                  gradient: np.ndarray,
                  lr: float) -> np.ndarray:
        """
        SGD + momentum 一步

        Args:
            mask: 当前掩模
            gradient: 梯度
            lr: 学习率

        Returns:
            更新后的掩模
        """
        if self._sgd_velocity is None:
            self._sgd_velocity = np.zeros_like(mask)

        momentum = 0.9
        self._sgd_velocity = momentum * self._sgd_velocity - lr * gradient
        return mask + self._sgd_velocity

    def reset_state(self):
        """重置优化器状态"""
        self._adam_m = None
        self._adam_v = None
        self._adam_t = 0
        self._sgd_velocity = None


class MaskComplexityPenalty:
    """
    掩模复杂度控制

    在目标函数中显式加入掩模复杂度惩罚，包括：
    1. 周长惩罚：控制掩模边界的总长度
    2. 顶点数惩罚：控制掩模形状的顶点数量
    3. 辅助特征数量惩罚：控制 SRAF 等辅助特征的数量

    所有惩罚项均提供可微梯度，支持端到端优化。
    """

    def __init__(self, config: ILTComplexityConfig):
        """
        初始化掩模复杂度惩罚

        Args:
            config: 复杂度惩罚配置
        """
        self.config = config

    def compute_perimeter(self, mask: np.ndarray) -> float:
        """
        计算掩模周长（可微近似）

        使用梯度幅值之和作为周长的可微近似：
            P ≈ Σ |∇M| ≈ Σ sqrt((∂M/∂x)² + (∂M/∂y)² + ε)

        Args:
            mask: 掩模图案

        Returns:
            周长值
        """
        gy = np.zeros_like(mask)
        gx = np.zeros_like(mask)
        gy[:-1, :] = mask[1:, :] - mask[:-1, :]
        gx[:, :-1] = mask[:, 1:] - mask[:, :-1]
        eps = 1e-8
        return float(np.sum(np.sqrt(gx ** 2 + gy ** 2 + eps)))

    def compute_perimeter_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算周长惩罚的梯度

        dP/dM 的向量化解法。

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        gy = np.zeros_like(mask)
        gx = np.zeros_like(mask)
        gy[:-1, :] = mask[1:, :] - mask[:-1, :]
        gx[:, :-1] = mask[:, 1:] - mask[:, :-1]

        eps = 1e-8
        mag = np.sqrt(gx ** 2 + gy ** 2 + eps)

        grad = np.zeros_like(mask)
        dx_pos = np.zeros_like(mask)
        dx_pos[:, :-1] = gx[:, :-1] / mag[:, :-1]
        dx_neg = np.zeros_like(mask)
        dx_neg[:, 1:] = -gx[:, :-1] / mag[:, :-1]

        dy_pos = np.zeros_like(mask)
        dy_pos[:-1, :] = gy[:-1, :] / mag[:-1, :]
        dy_neg = np.zeros_like(mask)
        dy_neg[1:, :] = -gy[:-1, :] / mag[:-1, :]

        grad = dx_pos + dx_neg + dy_pos + dy_neg
        return grad

    def compute_vertex_count(self, mask: np.ndarray) -> float:
        """
        计算掩模顶点数（可微近似）

        通过检测二值化后拐角点的数量估计顶点数。
        使用可微近似：对角方向二阶差的绝对值之和。

        Args:
            mask: 掩模图案

        Returns:
            顶点数估计
        """
        gy = np.zeros_like(mask)
        gx = np.zeros_like(mask)
        gy[:-1, :] = mask[1:, :] - mask[:-1, :]
        gx[:, :-1] = mask[:, 1:] - mask[:, :-1]

        gyy = np.zeros_like(mask)
        gxx = np.zeros_like(mask)
        gxy = np.zeros_like(mask)

        gyy[1:-1, :] = gy[1:-1, :] - gy[:-2, :]
        gxx[:, 1:-1] = gx[:, 1:-1] - gx[:, :-2]
        gxy[:-1, :-1] = (mask[1:, 1:] - mask[1:, :-1]
                          - mask[:-1, 1:] + mask[:-1, :-1])

        corner_response = np.abs(gxx * gyy - gxy ** 2)
        threshold = 0.01
        return float(np.sum(corner_response[corner_response > threshold]))

    def compute_vertex_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算顶点数惩罚的梯度

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        grad = np.zeros_like(mask)
        h, w = mask.shape

        if h < 3 or w < 3:
            return grad

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                shifted_y_s = max(0, dy)
                shifted_y_e = min(h, h + dy)
                src_y_s = max(0, -dy)
                src_y_e = min(h, h - dy)

                shifted_x_s = max(0, dx)
                shifted_x_e = min(w, w + dx)
                src_x_s = max(0, -dx)
                src_x_e = min(w, w - dx)

                sign = 1.0 if (dy + dx) % 2 == 0 else -1.0
                grad[src_y_s:src_y_e, src_x_s:src_x_e] += (
                    sign * 0.25
                )

        return grad

    def compute_sub_feature_count(self, mask: np.ndarray) -> float:
        """
        计算辅助特征数量

        对掩模做二值化后，检测面积在指定范围内的小连通域，
        统计其数量作为辅助特征数。

        Args:
            mask: 掩模图案

        Returns:
            辅助特征数量
        """
        binary = (mask >= 0.5).astype(np.int32)
        struct = generate_binary_structure(2, 2)
        eroded = binary_erosion(binary, structure=struct)
        skeleton_features = binary & ~eroded

        labeled, num_features = label(skeleton_features)
        if num_features == 0:
            return 0.0

        objects = find_objects(labeled)
        count = 0
        for idx, obj in enumerate(objects):
            if obj is None:
                continue
            region = (labeled == (idx + 1))
            area = int(np.sum(region))
            if (self.config.sub_feature_min_area <= area
                    <= self.config.sub_feature_max_area):
                count += 1

        return float(count)

    def compute_sub_feature_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算辅助特征数量惩罚的梯度

        使用形态学近似的可微梯度：辅助特征倾向于出现在
        细小突起处，梯度指向腐蚀方向。

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        gy = np.zeros_like(mask)
        gx = np.zeros_like(mask)
        gy[:-1, :] = mask[1:, :] - mask[:-1, :]
        gx[:, :-1] = mask[:, 1:] - mask[:, :-1]

        thinness = np.abs(gx) + np.abs(gy)
        threshold = 0.5
        grad = thinness * (1.0 - 2.0 * mask)
        grad[thinness < threshold] = 0.0

        return grad

    def compute_total_penalty(self,
                              mask: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        计算总掩模复杂度惩罚

        Args:
            mask: 掩模图案

        Returns:
            (总惩罚值, 各分量明细)
        """
        components: Dict[str, float] = {}
        total = 0.0

        if self.config.perimeter_weight > 0:
            p = self.compute_perimeter(mask)
            components['perimeter'] = p
            total += self.config.perimeter_weight * p

        if self.config.vertex_weight > 0:
            v = self.compute_vertex_count(mask)
            components['vertex_count'] = v
            total += self.config.vertex_weight * v

        if self.config.sub_feature_weight > 0:
            s = self.compute_sub_feature_count(mask)
            components['sub_feature_count'] = s
            total += self.config.sub_feature_weight * s

        return total, components

    def compute_total_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算总掩模复杂度惩罚的梯度

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        grad = np.zeros_like(mask)

        if self.config.perimeter_weight > 0:
            grad += self.config.perimeter_weight * self.compute_perimeter_gradient(mask)

        if self.config.vertex_weight > 0:
            grad += self.config.vertex_weight * self.compute_vertex_gradient(mask)

        if self.config.sub_feature_weight > 0:
            grad += self.config.sub_feature_weight * self.compute_sub_feature_gradient(mask)

        return grad


class MultiObjectiveILT:
    """
    多目标 ILT

    同时优化多个目标图案（如不同 focus 下的成像），加权求和。

    每个目标对应一个工艺条件（defocus, dose 等），
    可以指定独立的权重和目标图案。

    总损失 = Σ_i w_i · L_i(mask, target_i, condition_i)
    """

    def __init__(self,
                 imaging_chain: DifferentiableImagingChain,
                 config: ILTConfig):
        """
        初始化多目标 ILT

        Args:
            imaging_chain: 可微成像链
            config: ILT 配置
        """
        self.imaging_chain = imaging_chain
        self.config = config
        self._condition_imagers: List[DifferentiableImagingChain] = []

    def setup_conditions(self,
                         optical_system: OpticalSystem,
                         conditions: List[Dict[str, float]]):
        """
        为每个工艺条件创建独立的成像链

        Args:
            optical_system: 基础光学系统
            conditions: 工艺条件列表，每项含 'defocus', 'dose' 键
        """
        self._condition_imagers = []
        for cond in conditions:
            defocus = cond.get('defocus', 0.0)
            cond_optics = OpticalSystem(
                wavelength=optical_system.wavelength,
                na=optical_system.na,
                sigma=optical_system.sigma,
                pixel_size=optical_system.pixel_size,
                defocus=defocus,
                magnification=optical_system.magnification,
                illumination_type=optical_system.illumination_type,
                source_params=dict(optical_system.source_params),
                tcc_mode=optical_system.tcc_mode,
                socs_num_terms=optical_system.socs_num_terms,
                custom_source=optical_system.custom_source,
                zernike_coefficients=dict(optical_system.zernike_coefficients),
            )
            chain = DifferentiableImagingChain(
                cond_optics,
                self.imaging_chain.image_size,
                resist_steepness=self.config.resist_steepness,
                wafer_threshold=self.config.wafer_threshold,
            )
            self._condition_imagers.append(chain)

    def compute_multi_objective_loss(
            self,
            mask: np.ndarray,
            targets: List[np.ndarray],
            weights: List[float],
    ) -> Tuple[float, List[Dict[str, float]]]:
        """
        计算多目标加权损失

        Args:
            mask: 当前掩模
            targets: 每个条件对应的目标图案列表
            weights: 每个条件的权重

        Returns:
            (总损失, 各条件损失明细列表)
        """
        total_loss = 0.0
        all_components: List[Dict[str, float]] = []

        for i, (target, weight) in enumerate(zip(targets, weights)):
            if i < len(self._condition_imagers):
                chain = self._condition_imagers[i]
            else:
                chain = self.imaging_chain

            _, wafer = chain.forward(mask)
            diff = wafer - target
            l2_loss = float(np.mean(diff ** 2))
            condition_loss = weight * self.config.l2_wafer_weight * l2_loss

            components = {
                f'condition_{i}_l2_wafer': l2_loss,
                f'condition_{i}_weighted_loss': condition_loss,
                f'condition_{i}_weight': weight,
            }
            all_components.append(components)
            total_loss += condition_loss

        return total_loss, all_components

    def compute_multi_objective_gradient(
            self,
            mask: np.ndarray,
            targets: List[np.ndarray],
            weights: List[float],
    ) -> np.ndarray:
        """
        计算多目标加权梯度

        Args:
            mask: 当前掩模
            targets: 每个条件对应的目标图案列表
            weights: 每个条件的权重

        Returns:
            总梯度
        """
        total_grad = np.zeros_like(mask)
        N = mask.shape[0] * mask.shape[1]

        for i, (target, weight) in enumerate(zip(targets, weights)):
            if i < len(self._condition_imagers):
                chain = self._condition_imagers[i]
            else:
                chain = self.imaging_chain

            aerial, wafer = chain.forward(mask)
            dL_dW = 2.0 * self.config.l2_wafer_weight * (wafer - target) / N
            dW_dI = chain.compute_wafer_gradient(aerial)
            dI_dM = chain.compute_mask_gradient(mask)
            grad_i = weight * dL_dW * dW_dI * dI_dM
            total_grad += grad_i

        return total_grad


class ILTWorkflow:
    """
    ILT 工作流

    完整的反演光刻优化流程：
    1. 初始化可微成像链
    2. 设置梯度投影优化器
    3. 配置掩模复杂度惩罚
    4. 迭代优化：前向传播 → 损失计算 → 反向传播 → 梯度投影 → 量化
    5. 输出最优掩模

    与通用 MaskOptimizer 的区别：
    - MaskOptimizer 是黑盒优化，可切换各种通用优化器（GD/Adam/BFGS/GA 等）
    - ILTWorkflow 是专用的梯度投影优化，采用 ILT 特有的量化策略和复杂度控制
    - ILT 的全链路可微通过 soft resist + 可微阈值实现，
      而非 MaskOptimizer 的硬阈值后评估
    """

    def __init__(self,
                 optical_system: Optional[OpticalSystem] = None,
                 config: Optional[ILTConfig] = None):
        """
        初始化 ILT 工作流

        Args:
            optical_system: 光学系统参数
            config: ILT 配置
        """
        self.optical_system = optical_system or OpticalSystem()
        self.config = config or ILTConfig()

        self._imaging_chain: Optional[DifferentiableImagingChain] = None
        self._gradient_projector: Optional[GradientProjector] = None
        self._complexity_penalty: Optional[MaskComplexityPenalty] = None
        self._multi_objective: Optional[MultiObjectiveILT] = None

    def optimize(self,
                 initial_mask: np.ndarray,
                 target: np.ndarray) -> ILTWorkflowResult:
        """
        执行 ILT 优化

        Args:
            initial_mask: 初始掩模图案
            target: 目标图案（晶圆上期望的二值图）

        Returns:
            ILT 工作流结果
        """
        start_time = time.time()
        cfg = self.config
        mask = initial_mask.copy().astype(np.float64)
        image_size = mask.shape

        self._imaging_chain = DifferentiableImagingChain(
            self.optical_system, image_size,
            resist_steepness=cfg.resist_steepness,
            wafer_threshold=cfg.wafer_threshold,
            imaging_model=cfg.imaging_model,
        )
        self._gradient_projector = GradientProjector(cfg)
        self._complexity_penalty = MaskComplexityPenalty(cfg.complexity)

        is_multi = (cfg.multi_objective_conditions is not None
                    and len(cfg.multi_objective_conditions) > 0)

        if is_multi:
            self._multi_objective = MultiObjectiveILT(
                self._imaging_chain, cfg
            )
            self._multi_objective.setup_conditions(
                self.optical_system, cfg.multi_objective_conditions
            )

        _, initial_wafer = self._imaging_chain.forward(mask)
        initial_binary = (initial_wafer >= cfg.wafer_threshold).astype(np.float64)
        initial_epe = compute_epe(
            initial_binary, target, pixel_size=cfg.pixel_size
        )
        initial_loss, _ = self._compute_total_loss(mask, target)

        result = ILTWorkflowResult(
            initial_mask=initial_mask.copy(),
            optimal_mask=mask.copy(),
            initial_wafer=initial_wafer.copy(),
            optimal_wafer=initial_wafer.copy(),
            initial_epe=initial_epe,
            final_epe=initial_epe,
            initial_loss=initial_loss,
            final_loss=initial_loss,
        )

        best_loss = initial_loss
        best_mask = mask.copy()
        patience_counter = 0
        prev_loss = initial_loss

        if cfg.verbose:
            logger.info(f"ILT 优化开始: max_iter={cfg.max_iter}, "
                       f"lr={cfg.learning_rate}, "
                       f"optimizer={cfg.optimizer_type.value}, "
                       f"transmission={cfg.transmission_level.value}")
            logger.info(f"初始损失: {initial_loss:.6f}, "
                       f"初始 EPE: {initial_epe['epe_mean']:.3f} nm")

        for iteration in range(1, cfg.max_iter + 1):
            gradient = self._compute_total_gradient(mask, target)
            grad_norm = float(np.sqrt(np.mean(gradient ** 2)))

            mask = self._gradient_projector.step(mask, gradient, iteration)

            q_strength = self._gradient_projector.compute_quantization_strength(iteration)

            current_loss, loss_components = self._compute_total_loss(mask, target)

            _, wafer_cont = self._imaging_chain.forward(mask)
            wafer_bin = (wafer_cont >= cfg.wafer_threshold).astype(np.float64)

            iter_result = ILTIterationResult(
                iteration=iteration,
                loss=current_loss,
                loss_components=loss_components,
                mask=mask.copy(),
                wafer_continuous=wafer_cont.copy(),
                wafer_binary=wafer_bin.copy(),
                gradient_norm=grad_norm,
                learning_rate=cfg.learning_rate,
                quantization_strength_current=q_strength,
            )
            result.iterations.append(iter_result)
            result.loss_history.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                best_mask = mask.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if abs(prev_loss - current_loss) < cfg.convergence_tol:
                patience_counter += 1

            if patience_counter >= cfg.convergence_patience:
                result.converged = True
                result.reason = (f"收敛：连续 {cfg.convergence_patience} 次 "
                                f"损失改善 < {cfg.convergence_tol}")
                if cfg.verbose:
                    logger.info(f"迭代 {iteration}: {result.reason}")
                break

            prev_loss = current_loss

            if cfg.verbose and iteration % max(1, cfg.max_iter // 20) == 0:
                logger.info(
                    f"迭代 {iteration}/{cfg.max_iter}: "
                    f"loss={current_loss:.6f}, "
                    f"grad_norm={grad_norm:.6f}, "
                    f"q_strength={q_strength:.3f}"
                )

        result.optimal_mask = best_mask
        _, optimal_wafer = self._imaging_chain.forward(best_mask)
        optimal_binary = (optimal_wafer >= cfg.wafer_threshold).astype(np.float64)
        final_epe = compute_epe(
            optimal_binary, target, pixel_size=cfg.pixel_size
        )
        final_loss, _ = self._compute_total_loss(best_mask, target)

        result.optimal_wafer = optimal_wafer
        result.final_epe = final_epe
        result.final_loss = final_loss
        result.total_time = time.time() - start_time
        result.final_quantization_strength = (
            self._gradient_projector.compute_quantization_strength(
                result.num_iterations
            )
        )

        if not result.converged:
            result.reason = f"达到最大迭代次数 {cfg.max_iter}"

        if cfg.verbose:
            logger.info(f"ILT 优化完成: {result.reason}")
            logger.info(f"损失: {initial_loss:.6f} → {final_loss:.6f} "
                       f"(改善 {initial_loss - final_loss:.6f})")
            logger.info(f"EPE: {initial_epe['epe_mean']:.3f} → "
                       f"{final_epe['epe_mean']:.3f} nm")
            logger.info(f"总耗时: {result.total_time:.2f}s")

        return result

    def _compute_total_loss(self,
                            mask: np.ndarray,
                            target: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        计算总损失

        L = w_l2 · L2(wafer, target)
          + w_perimeter · P(mask) + w_vertex · V(mask) + w_subfeat · S(mask)
          + w_binary · B(mask)
          + w_tv · TV(mask)
          + Σ_i w_i · L_i (多目标)

        Args:
            mask: 当前掩模
            target: 目标图案

        Returns:
            (总损失, 各分量明细)
        """
        cfg = self.config
        components: Dict[str, float] = {}
        total = 0.0

        _, wafer = self._imaging_chain.forward(mask)

        is_multi = (cfg.multi_objective_conditions is not None
                    and len(cfg.multi_objective_conditions) > 0)

        if is_multi:
            targets = cfg.multi_objective_targets or [target]
            while len(targets) < len(cfg.multi_objective_conditions):
                targets.append(target)

            weights = [c.get('weight', 1.0) for c in cfg.multi_objective_conditions]
            w_sum = sum(weights) or 1.0
            norm_weights = [w / w_sum for w in weights]

            multi_loss, multi_components = (
                self._multi_objective.compute_multi_objective_loss(
                    mask, targets, norm_weights
                )
            )
            total += multi_loss
            for comp in multi_components:
                components.update(comp)
        else:
            N = mask.shape[0] * mask.shape[1]
            l2_loss = float(np.mean((wafer - target) ** 2))
            components['l2_wafer'] = l2_loss
            total += cfg.l2_wafer_weight * l2_loss

        complexity_total, complexity_components = (
            self._complexity_penalty.compute_total_penalty(mask)
        )
        total += complexity_total
        components.update(complexity_components)

        if cfg.binary_penalty_weight > 0:
            bp = self._compute_binary_penalty(mask)
            components['binary_penalty'] = bp
            total += cfg.binary_penalty_weight * bp

        if cfg.tv_smooth_weight > 0:
            tv = float(total_variation_isotropic(mask))
            components['tv_smooth'] = tv
            total += cfg.tv_smooth_weight * tv

        components['total'] = total
        return total, components

    def _compute_total_gradient(self,
                                mask: np.ndarray,
                                target: np.ndarray) -> np.ndarray:
        """
        计算总损失对掩模的梯度

        Args:
            mask: 当前掩模
            target: 目标图案

        Returns:
            梯度数组
        """
        cfg = self.config
        grad = np.zeros_like(mask)
        N = mask.shape[0] * mask.shape[1]

        is_multi = (cfg.multi_objective_conditions is not None
                    and len(cfg.multi_objective_conditions) > 0)

        if is_multi:
            targets = cfg.multi_objective_targets or [target]
            while len(targets) < len(cfg.multi_objective_conditions):
                targets.append(target)

            weights = [c.get('weight', 1.0) for c in cfg.multi_objective_conditions]
            w_sum = sum(weights) or 1.0
            norm_weights = [w / w_sum for w in weights]

            grad += self._multi_objective.compute_multi_objective_gradient(
                mask, targets, norm_weights
            )
        else:
            aerial, wafer = self._imaging_chain.forward(mask)
            dL_dW = 2.0 * cfg.l2_wafer_weight * (wafer - target) / N
            grad += self._imaging_chain.compute_full_gradient(mask, dL_dW)

        grad += self._complexity_penalty.compute_total_gradient(mask)

        if cfg.binary_penalty_weight > 0:
            grad += cfg.binary_penalty_weight * self._compute_binary_penalty_gradient(mask)

        if cfg.tv_smooth_weight > 0:
            grad += cfg.tv_smooth_weight * total_variation_isotropic_gradient(mask)

        return grad

    def _compute_binary_penalty(self, mask: np.ndarray) -> float:
        """
        计算二值化惩罚（曼哈顿距离）

        B = Σ 4·M·(1-M)，在 M=0.5 处取最大值 1，在 M=0 或 M=1 处为 0。

        Args:
            mask: 掩模图案

        Returns:
            二值化惩罚值
        """
        return float(np.mean(4.0 * mask * (1.0 - mask)))

    def _compute_binary_penalty_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算二值化惩罚的梯度

        dB/dM = 4·(1 - 2·M) / N

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        N = mask.shape[0] * mask.shape[1]
        return 4.0 * (1.0 - 2.0 * mask) / N


def run_ilt_workflow(initial_mask: np.ndarray,
                     target: np.ndarray,
                     optical_system: Optional[OpticalSystem] = None,
                     config: Optional[ILTConfig] = None,
                     config_path: Optional[Union[str, Path]] = None) -> ILTWorkflowResult:
    """
    ILT 工作流便捷入口函数

    Args:
        initial_mask: 初始掩模图案
        target: 目标图案
        optical_system: 光学系统参数，None 则使用默认参数
        config: ILT 配置，None 则使用默认配置
        config_path: YAML 配置文件路径，优先于 config 参数

    Returns:
        ILT 工作流结果
    """
    if config_path is not None:
        config = ILTConfig.from_yaml(config_path)

    workflow = ILTWorkflow(
        optical_system=optical_system,
        config=config,
    )
    return workflow.optimize(initial_mask, target)
