# -*- coding: utf-8 -*-
"""
多保真度贝叶斯优化 (MFBO) 数据结构定义

包含所有核心数据类、枚举和配置，用于：
- 保真度层级定义
- MFBO 配置参数
- 优化历史记录
- 优化结果封装
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FidelityLevel(str, Enum):
    """
    保真度层级枚举

    定义仿真精度从低到高的层级：
    - LOW: 廉价代理仿真（低分辨率/近似模型，快速但精度低）
    - MEDIUM: 中等精度仿真（平衡速度与精度）
    - HIGH: 全精度仿真（高分辨率/严格物理模型，精确但昂贵）
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_int(cls, level: int) -> 'FidelityLevel':
        """从整数转换（0=LOW, 1=MEDIUM, 2=HIGH）"""
        mapping = {0: cls.LOW, 1: cls.MEDIUM, 2: cls.HIGH}
        if level not in mapping:
            raise ValueError(f"Invalid fidelity level integer: {level}")
        return mapping[level]

    def to_int(self) -> int:
        """转换为整数"""
        mapping = {FidelityLevel.LOW: 0, FidelityLevel.MEDIUM: 1, FidelityLevel.HIGH: 2}
        return mapping[self]

    def __lt__(self, other):
        if isinstance(other, FidelityLevel):
            return self.to_int() < other.to_int()
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, FidelityLevel):
            return self.to_int() <= other.to_int()
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, FidelityLevel):
            return self.to_int() > other.to_int()
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, FidelityLevel):
            return self.to_int() >= other.to_int()
        return NotImplemented


class AcquisitionFunctionType(str, Enum):
    """获取函数类型枚举"""
    EI = "ei"              # Expected Improvement
    UCB = "ucb"            # Upper Confidence Bound
    PI = "pi"              # Probability of Improvement
    MES = "mes"            # Max-value Entropy Search
    EIV = "eiv"            # Expected Improvement per Unit Cost (Cost-Aware)
    KG = "kg"              # Knowledge Gradient
    MFES = "mfes"          # Multi-Fidelity Expected Improvement (with cost)


class KernelType(str, Enum):
    """多保真度核函数类型枚举"""
    AR1 = "ar1"                # AR1 (Auto-Regressive 1) 核心，Kennedy & O'Hagan 2000
    COKriging = "cokriging"    # Co-Kriging 核心
    LinearCoregional = "lcm"   # Linear Model of Coregionalization
    NARGP = "nargp"            # Nonlinear Auto-Regressive GP (深度型)


class FidelitySelectionStrategy(str, Enum):
    """保真度选择策略枚举"""
    COST_AWARE = "cost_aware"           # 成本感知获取函数（默认）
    INFORMATION_GAIN = "info_gain"      # 基于信息增益（KG/MES）
    BUDGET_PROPORTIONAL = "budget_prop" # 按预算比例分配
    SCHEDULED = "scheduled"             # 预设调度表
    ADAPTIVE_THRESHOLD = "adaptive"     # 自适应阈值切换


@dataclass
class FidelityCost:
    """
    各保真度层级的计算成本配置

    Attributes:
        costs: 各层级的相对成本字典，HIGH通常设为1.0作为基准
        absolute_times: 各层级的绝对计算时间（秒），用于实际预算跟踪
    """
    costs: Dict[FidelityLevel, float] = field(default_factory=lambda: {
        FidelityLevel.LOW: 0.01,
        FidelityLevel.MEDIUM: 0.1,
        FidelityLevel.HIGH: 1.0,
    })
    absolute_times: Dict[FidelityLevel, float] = field(default_factory=lambda: {
        FidelityLevel.LOW: 0.01,
        FidelityLevel.MEDIUM: 0.1,
        FidelityLevel.HIGH: 1.0,
    })

    def get_cost(self, level: FidelityLevel) -> float:
        """获取指定保真度的相对成本"""
        return self.costs.get(level, 1.0)

    def get_absolute_time(self, level: FidelityLevel) -> float:
        """获取指定保真度的绝对计算时间"""
        return self.absolute_times.get(level, 1.0)

    def cost_ratio(self, from_level: FidelityLevel, to_level: FidelityLevel) -> float:
        """计算两个保真度间的成本比率"""
        return self.get_cost(to_level) / self.get_cost(from_level)


@dataclass
class SearchSpace:
    """
    搜索空间定义

    Attributes:
        bounds: 每个维度的边界 [(low, high), ...]
        dimensions: 维度数量
        names: 可选的维度名称
        types: 可选的维度类型 ('continuous', 'integer', 'categorical')
    """
    bounds: List[Tuple[float, float]]
    dimensions: int = 0
    names: Optional[List[str]] = None
    types: Optional[List[str]] = None

    def __post_init__(self):
        if self.dimensions == 0:
            self.dimensions = len(self.bounds)
        if self.names is None:
            self.names = [f"x{i}" for i in range(self.dimensions)]
        if self.types is None:
            self.types = ['continuous'] * self.dimensions

    def sample(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """从搜索空间均匀采样"""
        if rng is None:
            rng = np.random.default_rng()
        samples = np.zeros((n_samples, self.dimensions))
        for i, (low, high) in enumerate(self.bounds):
            if self.types[i] == 'integer':
                samples[:, i] = rng.integers(int(low), int(high) + 1, size=n_samples)
            else:
                samples[:, i] = rng.uniform(low, high, size=n_samples)
        return samples

    def clip(self, x: np.ndarray) -> np.ndarray:
        """将值裁剪到搜索空间边界内"""
        x = np.atleast_2d(x)
        result = np.zeros_like(x)
        for i, (low, high) in enumerate(self.bounds):
            if self.types[i] == 'integer':
                result[:, i] = np.clip(np.round(x[:, i]), int(low), int(high))
            else:
                result[:, i] = np.clip(x[:, i], low, high)
        return result


@dataclass
class MFBOConfig:
    """
    多保真度贝叶斯优化完整配置

    Attributes:
        n_init_low: 低保真度初始样本数
        n_init_medium: 中保真度初始样本数
        n_init_high: 高保真度初始样本数
        max_iterations: 最大迭代次数
        max_budget: 最大计算预算（以HIGH保真度成本为单位）
        target_fidelity: 最终优化目标的保真度层级
        kernel_type: 多保真度核函数类型
        acquisition_type: 获取函数类型
        fidelity_strategy: 保真度选择策略
        ucb_beta: UCB的探索系数 β
        noise_variance: 观测噪声方差
        optimizer_restarts: GP超参数优化重启次数
        acq_optimizer: 获取函数优化方法 ('lbfgs', 'cmaes', 'random')
        acq_n_candidates: 获取函数优化随机候选点数
        early_stop_patience: 无改善提前停止的迭代数
        min_improvement_threshold: 最小改善阈值
        random_seed: 随机种子
        progress_callback: 进度回调函数
    """
    n_init_low: int = 10
    n_init_medium: int = 5
    n_init_high: int = 3
    max_iterations: int = 100
    max_budget: float = 50.0
    target_fidelity: FidelityLevel = FidelityLevel.HIGH

    kernel_type: KernelType = KernelType.AR1
    acquisition_type: AcquisitionFunctionType = AcquisitionFunctionType.EIV
    fidelity_strategy: FidelitySelectionStrategy = FidelitySelectionStrategy.COST_AWARE

    ucb_beta: float = 2.0
    noise_variance: float = 1e-6
    optimizer_restarts: int = 5
    acq_optimizer: str = "lbfgs"
    acq_n_candidates: int = 500

    early_stop_patience: int = 30
    min_improvement_threshold: float = 1e-8
    random_seed: Optional[int] = None

    cost_config: FidelityCost = field(default_factory=FidelityCost)

    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化配置到字典"""
        return {
            "n_init_low": self.n_init_low,
            "n_init_medium": self.n_init_medium,
            "n_init_high": self.n_init_high,
            "max_iterations": self.max_iterations,
            "max_budget": self.max_budget,
            "target_fidelity": self.target_fidelity.value,
            "kernel_type": self.kernel_type.value,
            "acquisition_type": self.acquisition_type.value,
            "fidelity_strategy": self.fidelity_strategy.value,
            "ucb_beta": self.ucb_beta,
            "noise_variance": self.noise_variance,
            "optimizer_restarts": self.optimizer_restarts,
            "acq_optimizer": self.acq_optimizer,
            "acq_n_candidates": self.acq_n_candidates,
            "early_stop_patience": self.early_stop_patience,
            "min_improvement_threshold": self.min_improvement_threshold,
            "random_seed": self.random_seed,
            "cost_config": {
                "costs": {k.value: v for k, v in self.cost_config.costs.items()},
                "absolute_times": {k.value: v for k, v in self.cost_config.absolute_times.items()},
            }
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MFBOConfig':
        """从字典反序列化配置"""
        cfg = cls()
        for k, v in d.items():
            if k == 'target_fidelity':
                cfg.target_fidelity = FidelityLevel(v)
            elif k == 'kernel_type':
                cfg.kernel_type = KernelType(v)
            elif k == 'acquisition_type':
                cfg.acquisition_type = AcquisitionFunctionType(v)
            elif k == 'fidelity_strategy':
                cfg.fidelity_strategy = FidelitySelectionStrategy(v)
            elif k == 'cost_config':
                costs = {FidelityLevel(k2): v2 for k2, v2 in v['costs'].items()}
                times = {FidelityLevel(k2): v2 for k2, v2 in v['absolute_times'].items()}
                cfg.cost_config = FidelityCost(costs=costs, absolute_times=times)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass
class Observation:
    """
    单次观测记录

    Attributes:
        x: 输入参数向量 (D,)
        y: 观测目标值（标量，最小化方向）
        fidelity: 观测的保真度层级
        cost: 此次观测消耗的计算成本
        time: 观测实际耗时（秒）
        metadata: 额外元数据
    """
    x: np.ndarray
    y: float
    fidelity: FidelityLevel
    cost: float = 0.0
    time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationRecord:
    """
    每次迭代的详细记录

    Attributes:
        iteration: 迭代编号
        selected_x: 选择的输入点
        selected_fidelity: 选择的保真度
        acquisition_value: 获取函数值
        predicted_mean: GP预测均值
        predicted_std: GP预测标准差
        observed_y: 实际观测值
        cost_spent: 此迭代消耗的成本
        best_y_so_far: 至今最优目标值（target_fidelity）
        total_budget_used: 已使用总预算
        surrogate_error: 保真度间代理误差（如可用）
    """
    iteration: int
    selected_x: np.ndarray
    selected_fidelity: FidelityLevel
    acquisition_value: float
    predicted_mean: float
    predicted_std: float
    observed_y: float
    cost_spent: float
    best_y_so_far: float
    total_budget_used: float
    surrogate_error: Optional[float] = None


@dataclass
class MFBOResult:
    """
    多保真度贝叶斯优化结果

    Attributes:
        best_x: 最优输入参数
        best_y: 最优目标值（target_fidelity下）
        best_fidelity: 最优解的保真度层级
        n_iterations: 完成的迭代次数
        total_budget_used: 总消耗计算成本
        total_time: 总运行时间
        observations: 所有观测记录
        history: 迭代历史记录
        final_gp_nll: 最终GP负对数似然
        convergence_plot_data: 用于收敛绘图的数据
    """
    best_x: np.ndarray
    best_y: float
    best_fidelity: FidelityLevel
    n_iterations: int
    total_budget_used: float
    total_time: float
    observations: List[Observation]
    history: List[IterationRecord]
    final_gp_nll: float = 0.0
    convergence_plot_data: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """生成结果摘要文本"""
        lines = []
        lines.append("=" * 60)
        lines.append("多保真度贝叶斯优化 (MFBO) 结果摘要")
        lines.append("=" * 60)
        lines.append(f"  迭代次数: {self.n_iterations}")
        lines.append(f"  最优目标值: {self.best_y:.6e}")
        lines.append(f"  最优保真度: {self.best_fidelity.value}")
        lines.append(f"  最优参数: {self.best_x}")
        lines.append(f"  总消耗成本: {self.total_budget_used:.2f} (high-fidelity units)")
        lines.append(f"  总运行时间: {self.total_time:.2f}s")

        fidelity_counts = {}
        for obs in self.observations:
            fidelity_counts[obs.fidelity.value] = fidelity_counts.get(obs.fidelity.value, 0) + 1
        lines.append(f"  各保真度样本数: {fidelity_counts}")
        lines.append(f"  最终GP NLL: {self.final_gp_nll:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def get_convergence_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取收敛曲线数据

        Returns:
            (iterations, budget_used, best_y_so_far)
        """
        iters = np.array([h.iteration for h in self.history])
        budgets = np.array([h.total_budget_used for h in self.history])
        bests = np.array([h.best_y_so_far for h in self.history])
        return iters, budgets, bests

    def to_dict(self, include_observations: bool = False) -> Dict[str, Any]:
        """序列化结果到字典"""
        result = {
            "best_x": self.best_x.tolist(),
            "best_y": float(self.best_y),
            "best_fidelity": self.best_fidelity.value,
            "n_iterations": self.n_iterations,
            "total_budget_used": float(self.total_budget_used),
            "total_time": float(self.total_time),
            "final_gp_nll": float(self.final_gp_nll),
            "history": [
                {
                    "iteration": h.iteration,
                    "selected_x": h.selected_x.tolist(),
                    "selected_fidelity": h.selected_fidelity.value,
                    "acquisition_value": float(h.acquisition_value),
                    "predicted_mean": float(h.predicted_mean),
                    "predicted_std": float(h.predicted_std),
                    "observed_y": float(h.observed_y),
                    "cost_spent": float(h.cost_spent),
                    "best_y_so_far": float(h.best_y_so_far),
                    "total_budget_used": float(h.total_budget_used),
                    "surrogate_error": float(h.surrogate_error) if h.surrogate_error is not None else None,
                }
                for h in self.history
            ],
        }
        if include_observations:
            result["observations"] = [
                {
                    "x": obs.x.tolist(),
                    "y": float(obs.y),
                    "fidelity": obs.fidelity.value,
                    "cost": float(obs.cost),
                    "time": float(obs.time),
                    "metadata": obs.metadata,
                }
                for obs in self.observations
            ]
        return result
