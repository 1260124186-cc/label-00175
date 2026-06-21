# -*- coding: utf-8 -*-
"""
MPC (Model Predictive Control) 自适应 OPC 工作流模块

将 OPC 建模为滚动时域优化问题（Receding Horizon Optimization）：
每轮根据预测的未来若干步 EPE 变化趋势调整边缘偏移量，
而非仅贪婪最小化当前 EPE，用于应对工艺漂移或批次间变化的在线校正场景。

核心思想：
    传统 OPC: min EPE(current_mask)  → 贪婪优化
    MPC-OPC:  min Σ EPE_k  (k=1..N)  → 滚动时域优化
              s.t. 边缘偏移约束
              只执行第一步偏移，然后滚动到下一轮

主要组件：
    1. EPEPredictor: EPE 预测模型，支持在线递推最小二乘(RLS)更新
    2. ProcessDriftEstimator: 工艺漂移估计器，跟踪 EPE 基线漂移
    3. MPCOptimizer: 滚动时域优化器，求解有限时域最优控制序列
    4. MPCOPCWorkflow: MPC-OPC 完整工作流封装
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import logging
from scipy.optimize import minimize
from scipy.ndimage import (
    distance_transform_edt, binary_dilation, binary_erosion,
    generate_binary_structure
)

from core.imaging import OpticalSystem, simulate_wafer_image
from core.litho_metrics import compute_epe, extract_edges
from workflows.opc import (
    OPCConfig,
    HotspotDetector,
    OPCTransformApplier,
    HotspotRegion,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class MPCConfig:
    """
    MPC-OPC 配置

    Attributes:
        prediction_horizon: 预测时域 N（预测未来多少步 EPE）
        control_horizon: 控制时域 M（优化多少步控制量，M <= N）
        epe_weight: 预测 EPE 在目标函数中的权重
        control_weight: 控制量（边缘偏移量）在目标函数中的权重
        control_rate_weight: 控制量变化率惩罚权重（防止抖动）
        drift_compensation_weight: 漂移补偿权重
        max_edge_offset: 单次最大边缘偏移量（像素）
        max_total_offset: 累计最大边缘偏移量（像素）
        max_offset_change: 相邻两次偏移的最大变化量（像素）

        use_drift_estimation: 是否启用工艺漂移估计
        drift_window_size: 漂移估计滑动窗口大小
        drift_forgetting_factor: 漂移估计遗忘因子 (0, 1]

        predictor_type: 预测模型类型 ('linear', 'arx', 'quadratic')
        rls_forgetting_factor: RLS 遗忘因子 (0, 1]
        rls_initial_covariance: RLS 初始协方差

        online_update: 是否在线更新预测模型
        adaptation_rate: 自适应学习率

        opc_config: 基础 OPC 配置
        verbose: 是否输出详细日志
    """
    prediction_horizon: int = 5
    control_horizon: int = 3
    epe_weight: float = 1.0
    control_weight: float = 0.01
    control_rate_weight: float = 0.05
    drift_compensation_weight: float = 0.5
    max_edge_offset: float = 2.0
    max_total_offset: float = 6.0
    max_offset_change: float = 1.0

    use_drift_estimation: bool = True
    drift_window_size: int = 10
    drift_forgetting_factor: float = 0.9

    predictor_type: str = 'linear'
    rls_forgetting_factor: float = 0.95
    rls_initial_covariance: float = 1000.0

    online_update: bool = True
    adaptation_rate: float = 0.1

    opc_config: Optional[OPCConfig] = None
    verbose: bool = True

    def __post_init__(self):
        if self.opc_config is None:
            self.opc_config = OPCConfig()

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'MPCConfig':
        if d is None:
            return cls()
        config = cls()
        opc_dict = d.pop('opc_config', None)
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        if opc_dict is not None:
            config.opc_config = OPCConfig.from_dict(opc_dict)
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prediction_horizon': self.prediction_horizon,
            'control_horizon': self.control_horizon,
            'epe_weight': self.epe_weight,
            'control_weight': self.control_weight,
            'control_rate_weight': self.control_rate_weight,
            'drift_compensation_weight': self.drift_compensation_weight,
            'max_edge_offset': self.max_edge_offset,
            'max_total_offset': self.max_total_offset,
            'max_offset_change': self.max_offset_change,
            'use_drift_estimation': self.use_drift_estimation,
            'drift_window_size': self.drift_window_size,
            'drift_forgetting_factor': self.drift_forgetting_factor,
            'predictor_type': self.predictor_type,
            'rls_forgetting_factor': self.rls_forgetting_factor,
            'rls_initial_covariance': self.rls_initial_covariance,
            'online_update': self.online_update,
            'adaptation_rate': self.adaptation_rate,
            'opc_config': self.opc_config.to_dict() if self.opc_config else None,
            'verbose': self.verbose,
        }


@dataclass
class PredictionResult:
    """
    EPE 预测结果

    Attributes:
        predicted_epe: 预测的未来 EPE 序列 (nm)，长度为 prediction_horizon
        predicted_epe_std: 预测标准差
        drift_rate: 估计的漂移速率 (nm/步)
        features: 用于预测的特征向量
    """
    predicted_epe: np.ndarray
    predicted_epe_std: np.ndarray
    drift_rate: float = 0.0
    features: Optional[np.ndarray] = None


@dataclass
class MPCOptimizationResult:
    """
    MPC 优化结果

    Attributes:
        optimal_offsets: 最优边缘偏移序列（像素），长度为 control_horizon
        predicted_epe: 对应预测的 EPE 序列 (nm)
        objective_value: 目标函数值
        constraints_satisfied: 是否满足所有约束
        first_offset: 第一步偏移量（实际执行的偏移）
    """
    optimal_offsets: np.ndarray
    predicted_epe: np.ndarray
    objective_value: float
    constraints_satisfied: bool = True
    first_offset: float = 0.0

    def __post_init__(self):
        if len(self.optimal_offsets) > 0:
            self.first_offset = self.optimal_offsets[0]


@dataclass
class MPCOPCIterationResult:
    """
    MPC-OPC 单次迭代结果

    Attributes:
        iteration: 迭代次数
        mask_before: 迭代前掩模
        mask_after: 迭代后掩模
        epe_before: 迭代前 EPE 统计
        epe_after: 迭代后 EPE 统计
        edge_offset_applied: 本次应用的边缘偏移量（像素）
        cumulative_offset: 累计边缘偏移量（像素）
        prediction_result: EPE 预测结果
        optimization_result: MPC 优化结果
        drift_rate: 当前估计的漂移速率 (nm/迭代)
        hotspots_before: 迭代前热点
        hotspots_after: 迭代后热点
    """
    iteration: int
    mask_before: np.ndarray
    mask_after: np.ndarray
    epe_before: Dict[str, float]
    epe_after: Dict[str, float]
    edge_offset_applied: float
    cumulative_offset: float
    prediction_result: Optional[PredictionResult] = None
    optimization_result: Optional[MPCOptimizationResult] = None
    drift_rate: float = 0.0
    hotspots_before: List[HotspotRegion] = field(default_factory=list)
    hotspots_after: List[HotspotRegion] = field(default_factory=list)

    @property
    def epe_improvement(self) -> float:
        return self.epe_before.get('epe_mean', 0.0) - self.epe_after.get('epe_mean', 0.0)

    @property
    def epe_improvement_ratio(self) -> float:
        before = self.epe_before.get('epe_mean', 0.0)
        if before > 0:
            return self.epe_improvement / before
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        pred_epe = None
        if self.prediction_result is not None:
            pred_epe = self.prediction_result.predicted_epe.tolist()
        opt_offsets = None
        if self.optimization_result is not None:
            opt_offsets = self.optimization_result.optimal_offsets.tolist()
        return {
            'iteration': self.iteration,
            'epe_before_mean': self.epe_before.get('epe_mean', 0.0),
            'epe_after_mean': self.epe_after.get('epe_mean', 0.0),
            'epe_improvement': self.epe_improvement,
            'epe_improvement_ratio': self.epe_improvement_ratio,
            'edge_offset_applied': self.edge_offset_applied,
            'cumulative_offset': self.cumulative_offset,
            'drift_rate': self.drift_rate,
            'predicted_epe': pred_epe,
            'optimal_offsets': opt_offsets,
            'hotspots_before_count': len(self.hotspots_before),
            'hotspots_after_count': len(self.hotspots_after),
        }


@dataclass
class MPCOPCWorkflowResult:
    """
    MPC-OPC 工作流最终结果

    Attributes:
        initial_mask: 初始掩模
        corrected_mask: 校正后掩模
        initial_epe: 初始 EPE 统计
        final_epe: 最终 EPE 统计
        iterations: 所有迭代结果列表
        converged: 是否收敛
        reason: 终止原因
        final_drift_rate: 最终估计的漂移速率
        total_epe_improvement: 总 EPE 改善量
    """
    initial_mask: np.ndarray
    corrected_mask: np.ndarray
    initial_epe: Dict[str, float]
    final_epe: Dict[str, float]
    iterations: List[MPCOPCIterationResult] = field(default_factory=list)
    converged: bool = False
    reason: str = ''
    final_drift_rate: float = 0.0

    @property
    def total_epe_improvement(self) -> float:
        return self.initial_epe.get('epe_mean', 0.0) - self.final_epe.get('epe_mean', 0.0)

    @property
    def total_epe_improvement_ratio(self) -> float:
        init = self.initial_epe.get('epe_mean', 0.0)
        if init > 0:
            return self.total_epe_improvement / init
        return 0.0

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    def summary(self) -> Dict[str, Any]:
        return {
            'initial_epe_mean': self.initial_epe.get('epe_mean', 0.0),
            'final_epe_mean': self.final_epe.get('epe_mean', 0.0),
            'total_epe_improvement': self.total_epe_improvement,
            'total_epe_improvement_ratio': self.total_epe_improvement_ratio,
            'num_iterations': self.num_iterations,
            'converged': self.converged,
            'reason': self.reason,
            'final_drift_rate': self.final_drift_rate,
        }


# ============================================================================
# EPE 预测模型
# ============================================================================

class EPEPredictor:
    """
    EPE 预测模型

    基于历史 EPE 和控制量（边缘偏移量）预测未来 EPE 变化趋势。
    支持在线递推最小二乘(RLS)更新，适应工艺漂移。

    预测模型（线性 ARX 形式）：
        EPE[k+1] = a0 + a1*EPE[k] + a2*offset[k] + drift * k + noise

    其中：
        - EPE[k]: 第 k 步的 EPE 值
        - offset[k]: 第 k 步施加的边缘偏移量
        - drift: 工艺漂移速率
    """

    def __init__(self, config: MPCConfig):
        """
        初始化 EPE 预测器

        Args:
            config: MPC 配置
        """
        self.config = config
        self.predictor_type = config.predictor_type

        self.weights: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self._init_model()

        self.history_epe: List[float] = []
        self.history_offsets: List[float] = []
        self.history_steps: List[int] = []

        self._prev_epe: Optional[float] = None
        self._prev_offset: Optional[float] = None
        self._prev_step: Optional[int] = None

    def _init_model(self):
        """初始化模型参数"""
        if self.predictor_type == 'linear':
            n_features = 3
        elif self.predictor_type == 'quadratic':
            n_features = 5
        else:
            n_features = 3

        self.weights = np.zeros(n_features)
        self.weights[0] = 1.0
        self.covariance = np.eye(n_features) * self.config.rls_initial_covariance

    def reset(self):
        """重置预测器"""
        self._init_model()
        self.history_epe.clear()
        self.history_offsets.clear()
        self.history_steps.clear()
        self._prev_epe = None
        self._prev_offset = None
        self._prev_step = None

    def _extract_features(self, epe: float, offset: float, step: int) -> np.ndarray:
        """
        提取特征向量

        Args:
            epe: 当前 EPE 值
            offset: 当前边缘偏移量
            step: 当前步数（用于漂移项）

        Returns:
            特征向量
        """
        if self.predictor_type == 'linear':
            return np.array([epe, offset, 1.0])
        elif self.predictor_type == 'quadratic':
            return np.array([epe, offset, offset ** 2, step, 1.0])
        else:
            return np.array([epe, offset, 1.0])

    def update(self, epe: float, offset: float, step: int = 0):
        """
        使用新观测值在线更新模型（RLS）

        ARX 模型：EPE[k+1] = f(EPE[k], offset[k])
        因此使用前一时刻的 (EPE, offset) 作为特征，
        当前时刻的 EPE 作为目标值进行更新。

        首次调用时只记录状态，不进行更新。

        Args:
            epe: 实测 EPE 值
            offset: 施加的边缘偏移量（累计）
            step: 当前步数
        """
        self.history_epe.append(epe)
        self.history_offsets.append(offset)
        self.history_steps.append(step)

        if not self.config.online_update:
            self._prev_epe = epe
            self._prev_offset = offset
            self._prev_step = step
            return

        if self._prev_epe is not None:
            x = self._extract_features(self._prev_epe, self._prev_offset, self._prev_step)
            y = epe

            lamb = self.config.rls_forgetting_factor

            try:
                P = self.covariance
                w = self.weights

                y_pred = np.dot(w, x)
                error = y - y_pred

                Px = np.dot(P, x)
                gain = Px / (lamb + np.dot(x, Px))

                self.weights = w + gain * error
                self.covariance = (P - np.outer(gain, np.dot(x, P))) / lamb
            except Exception as e:
                logger.warning(f"RLS 更新失败: {e}")

        self._prev_epe = epe
        self._prev_offset = offset
        self._prev_step = step

    def predict(self,
                current_epe: float,
                current_offset: float,
                offset_sequence: np.ndarray,
                start_step: int = 0) -> PredictionResult:
        """
        预测未来 N 步的 EPE

        Args:
            current_epe: 当前 EPE 值
            current_offset: 当前累计偏移量
            offset_sequence: 未来的边缘偏移量序列（控制输入）
            start_step: 起始步数（用于漂移项）

        Returns:
            预测结果
        """
        N = len(offset_sequence)
        predicted_epe = np.zeros(N)
        predicted_std = np.zeros(N)

        epe_k = current_epe
        offset_k = current_offset

        for k in range(N):
            offset_k += offset_sequence[k]
            x = self._extract_features(epe_k, offset_k, start_step + k)
            epe_k = float(np.dot(self.weights, x))
            predicted_epe[k] = epe_k

            if self.covariance is not None:
                var = np.dot(x, np.dot(self.covariance, x))
                predicted_std[k] = np.sqrt(max(var, 0.0))

        drift_rate = 0.0
        if len(self.history_epe) >= 2:
            recent_epe = self.history_epe[-min(5, len(self.history_epe)):]
            if len(recent_epe) >= 2:
                drift_rate = (recent_epe[-1] - recent_epe[0]) / (len(recent_epe) - 1)

        return PredictionResult(
            predicted_epe=predicted_epe,
            predicted_epe_std=predicted_std,
            drift_rate=drift_rate,
        )

    def predict_with_drift(self,
                           current_epe: float,
                           current_offset: float,
                           offset_sequence: np.ndarray,
                           drift_rate: float = 0.0,
                           start_step: int = 0) -> PredictionResult:
        """
        带漂移补偿的 EPE 预测

        Args:
            current_epe: 当前 EPE 值
            current_offset: 当前累计偏移量
            offset_sequence: 未来的边缘偏移量序列
            drift_rate: 估计的漂移速率 (nm/步)
            start_step: 起始步数

        Returns:
            预测结果
        """
        result = self.predict(current_epe, current_offset, offset_sequence, start_step)

        if self.config.use_drift_estimation and abs(drift_rate) > 1e-10:
            drift_comp = drift_rate * self.config.drift_compensation_weight
            for k in range(len(result.predicted_epe)):
                result.predicted_epe[k] += drift_comp * (k + 1)

        result.drift_rate = drift_rate
        return result


# ============================================================================
# 工艺漂移估计器
# ============================================================================

class ProcessDriftEstimator:
    """
    工艺漂移估计器

    使用滑动窗口和指数加权移动平均(EWMA)估计工艺漂移速率。
    漂移表现为 EPE 基线随时间的系统性变化。
    """

    def __init__(self, config: MPCConfig):
        """
        初始化漂移估计器

        Args:
            config: MPC 配置
        """
        self.config = config
        self.window_size = config.drift_window_size
        self.forgetting_factor = config.drift_forgetting_factor

        self.epe_history: List[float] = []
        self.offset_history: List[float] = []
        self.step_history: List[int] = []

        self.ewma_epe: Optional[float] = None
        self.ewma_offset: Optional[float] = None
        self.drift_rate: float = 0.0

    def reset(self):
        """重置估计器"""
        self.epe_history.clear()
        self.offset_history.clear()
        self.step_history.clear()
        self.ewma_epe = None
        self.ewma_offset = None
        self.drift_rate = 0.0

    def update(self, epe: float, offset: float, step: int = 0):
        """
        更新漂移估计

        Args:
            epe: 实测 EPE 值
            offset: 当前累计偏移量
            step: 当前步数
        """
        self.epe_history.append(epe)
        self.offset_history.append(offset)
        self.step_history.append(step)

        if len(self.epe_history) > self.window_size:
            self.epe_history.pop(0)
            self.offset_history.pop(0)
            self.step_history.pop(0)

        alpha = 1.0 - self.forgetting_factor
        if self.ewma_epe is None:
            self.ewma_epe = epe
            self.ewma_offset = offset
        else:
            self.ewma_epe = alpha * epe + (1 - alpha) * self.ewma_epe
            self.ewma_offset = alpha * offset + (1 - alpha) * self.ewma_offset

        self._estimate_drift_rate()

    def _estimate_drift_rate(self):
        """估计漂移速率"""
        if len(self.epe_history) < 3:
            self.drift_rate = 0.0
            return

        epe_arr = np.array(self.epe_history)
        offset_arr = np.array(self.offset_history)
        step_arr = np.array(self.step_history, dtype=float)

        if len(step_arr) < 2:
            self.drift_rate = 0.0
            return

        weights = np.exp(-0.1 * (np.arange(len(step_arr))[::-1]))
        weights = weights / weights.sum()

        try:
            epe_norm = epe_arr - np.average(epe_arr, weights=weights)
            step_norm = step_arr - np.average(step_arr, weights=weights)

            numerator = np.sum(weights * epe_norm * step_norm)
            denominator = np.sum(weights * step_norm ** 2)

            if abs(denominator) > 1e-10:
                raw_drift = numerator / denominator
            else:
                raw_drift = 0.0

            offset_effect = self._estimate_offset_effect(epe_arr, offset_arr)
            self.drift_rate = raw_drift - offset_effect * np.mean(np.diff(offset_arr))
        except Exception:
            self.drift_rate = 0.0

    def _estimate_offset_effect(self, epe_arr: np.ndarray, offset_arr: np.ndarray) -> float:
        """估计边缘偏移对 EPE 的影响系数"""
        if len(epe_arr) < 3:
            return -1.0

        try:
            epe_norm = epe_arr - epe_arr.mean()
            offset_norm = offset_arr - offset_arr.mean()

            numerator = np.sum(epe_norm * offset_norm)
            denominator = np.sum(offset_norm ** 2)

            if abs(denominator) > 1e-10:
                return numerator / denominator
            return -1.0
        except Exception:
            return -1.0

    def get_drift_rate(self) -> float:
        """获取当前估计的漂移速率 (nm/步)"""
        return self.drift_rate

    def predict_drift(self, steps: int) -> np.ndarray:
        """
        预测未来若干步的漂移累积量

        Args:
            steps: 预测步数

        Returns:
            每步的累积漂移量数组
        """
        return np.arange(1, steps + 1) * self.drift_rate


# ============================================================================
# MPC 滚动时域优化器
# ============================================================================

class MPCOptimizer:
    """
    滚动时域优化器 (Receding Horizon Optimizer)

    在给定的预测时域内，求解最优边缘偏移序列，
    最小化未来 EPE 的加权和 + 控制量惩罚 + 控制量变化率惩罚。

    优化问题：
        min_{u_0..u_{M-1}}  Σ_{k=1}^{N} w_epe * EPE_k^2
                            + Σ_{k=0}^{M-1} w_u * u_k^2
                            + Σ_{k=1}^{M-1} w_Δu * (u_k - u_{k-1})^2

        s.t.  |u_k| <= u_max
              |Σ_{i=0}^{k} u_i| <= u_total_max
              |u_k - u_{k-1}| <= Δu_max

    其中：
        - N: 预测时域
        - M: 控制时域
        - u_k: 第 k 步的边缘偏移量
        - EPE_k: 预测的第 k 步 EPE
    """

    def __init__(self, config: MPCConfig, predictor: EPEPredictor):
        """
        初始化 MPC 优化器

        Args:
            config: MPC 配置
            predictor: EPE 预测器
        """
        self.config = config
        self.predictor = predictor

    def solve(self,
              current_epe: float,
              current_offset: float,
              drift_rate: float = 0.0,
              start_step: int = 0) -> MPCOptimizationResult:
        """
        求解滚动时域优化问题

        Args:
            current_epe: 当前 EPE 值 (nm)
            current_offset: 当前累计边缘偏移量 (像素)
            drift_rate: 估计的漂移速率 (nm/步)
            start_step: 当前步数（用于漂移项）

        Returns:
            优化结果
        """
        N = self.config.prediction_horizon
        M = self.config.control_horizon
        M = min(M, N)

        u0 = np.zeros(M)

        bounds = [(-self.config.max_edge_offset, self.config.max_edge_offset)
                  for _ in range(M)]

        constraints = []

        def objective(u: np.ndarray) -> float:
            u_full = np.zeros(N)
            u_full[:M] = u
            if M < N:
                u_full[M:] = u[-1]

            pred = self.predictor.predict_with_drift(
                current_epe, current_offset, u_full, drift_rate, start_step
            )

            epe_term = self.config.epe_weight * np.sum(pred.predicted_epe ** 2)
            control_term = self.config.control_weight * np.sum(u ** 2)

            rate_term = 0.0
            if len(u) > 1:
                rate_term = self.config.control_rate_weight * np.sum(np.diff(u) ** 2)

            return epe_term + control_term + rate_term

        def cumulative_offset_constraint(u: np.ndarray) -> float:
            cum = np.cumsum(u)
            total = current_offset + cum
            return self.config.max_total_offset - np.max(np.abs(total))

        def rate_constraint(u: np.ndarray) -> float:
            if len(u) <= 1:
                return self.config.max_offset_change
            diffs = np.abs(np.diff(u))
            return self.config.max_offset_change - np.max(diffs)

        constraints.append({
            'type': 'ineq',
            'fun': cumulative_offset_constraint
        })
        constraints.append({
            'type': 'ineq',
            'fun': rate_constraint
        })

        try:
            result = minimize(
                objective,
                u0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 100,
                    'ftol': 1e-6,
                    'disp': False,
                }
            )

            optimal_u = result.x
            constraints_ok = result.success

            u_full = np.zeros(N)
            u_full[:M] = optimal_u
            if M < N:
                u_full[M:] = optimal_u[-1]

            final_pred = self.predictor.predict_with_drift(
                current_epe, current_offset, u_full, drift_rate, start_step
            )

            return MPCOptimizationResult(
                optimal_offsets=optimal_u,
                predicted_epe=final_pred.predicted_epe,
                objective_value=result.fun,
                constraints_satisfied=constraints_ok,
            )

        except Exception as e:
            logger.warning(f"MPC 优化失败，回退到贪婪策略: {e}")

            greedy_offset = np.clip(
                -current_epe * self.config.adaptation_rate,
                -self.config.max_edge_offset,
                self.config.max_edge_offset
            )

            greedy_offsets = np.full(M, greedy_offset)

            u_full = np.zeros(N)
            u_full[:M] = greedy_offsets
            if M < N:
                u_full[M:] = greedy_offset

            fallback_pred = self.predictor.predict_with_drift(
                current_epe, current_offset, u_full, drift_rate, start_step
            )

            return MPCOptimizationResult(
                optimal_offsets=greedy_offsets,
                predicted_epe=fallback_pred.predicted_epe,
                objective_value=float('inf'),
                constraints_satisfied=False,
            )


# ============================================================================
# MPC-OPC 工作流主类
# ============================================================================

class MPCOPCWorkflow:
    """
    MPC 自适应 OPC 工作流

    将 OPC 建模为滚动时域优化问题：
    每轮预测未来若干步 EPE 变化趋势，优化边缘偏移序列，
    只执行第一步偏移，然后滚动到下一轮。

    相比传统贪婪 OPC 的优势：
        1. 考虑未来 EPE 趋势，避免短视决策
        2. 自动补偿工艺漂移
        3. 在线学习模型参数，适应批次间变化
        4. 控制量平滑，防止偏移抖动

    工作流程：
        1. 初始化：模拟初始 EPE，初始化预测模型和漂移估计器
        2. 滚动优化循环：
           a. 预测未来 N 步 EPE
           b. MPC 优化求解最优 M 步偏移序列
           c. 执行第一步偏移
           d. 重新模拟得到新 EPE
           e. 在线更新预测模型和漂移估计
           f. 收敛检查
        3. 输出最终掩模
    """

    def __init__(self,
                 config: Optional[MPCConfig] = None,
                 optical_system: Optional[OpticalSystem] = None):
        """
        初始化 MPC-OPC 工作流

        Args:
            config: MPC 配置，None 则使用默认配置
            optical_system: 光学系统参数，None 则使用默认参数
        """
        self.config = config or MPCConfig()
        self.optical_system = optical_system or OpticalSystem()

        opc_cfg = self.config.opc_config or OPCConfig()
        self.opc_config = opc_cfg

        self.hotspot_detector = HotspotDetector(opc_cfg)
        self.transform_applier = OPCTransformApplier(opc_cfg)

        self.predictor = EPEPredictor(self.config)
        self.drift_estimator = ProcessDriftEstimator(self.config)
        self.mpc_optimizer = MPCOptimizer(self.config, self.predictor)

        if self.config.verbose:
            logger.info("MPC-OPC 工作流已初始化")
            logger.info(f"预测时域: {self.config.prediction_horizon}, "
                       f"控制时域: {self.config.control_horizon}")
            logger.info(f"漂移估计: {'启用' if self.config.use_drift_estimation else '禁用'}")
            logger.info(f"在线更新: {'启用' if self.config.online_update else '禁用'}")

    def _simulate_epe(self, mask: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        模拟晶圆成像并计算 EPE

        Args:
            mask: 掩模图案
            target: 目标图案

        Returns:
            (晶圆二值图, EPE 统计字典)
        """
        wafer_cont = simulate_wafer_image(
            mask,
            optical_system=self.optical_system,
            threshold=self.opc_config.wafer_threshold,
            apply_resist=True
        )
        wafer_bin = (wafer_cont >= self.opc_config.wafer_threshold).astype(np.float64)

        epe_stats = compute_epe(
            wafer_bin, target,
            pixel_size=self.opc_config.pixel_size
        )

        return wafer_bin, epe_stats

    def _apply_global_edge_offset(self,
                                   mask: np.ndarray,
                                   offset: float) -> np.ndarray:
        """
        对掩模应用全局边缘偏移

        使用带符号距离变换实现确定性的边缘偏移，
        支持亚像素精度的连续偏移量。

        带符号距离约定：
            - 掩模内部：距离为正（到边缘的距离
            - 掩模外部：距离为负（到边缘的距离）
            - 偏移 > 0：向外膨胀（特征变大）
            - 偏移 < 0：向内腐蚀（特征变小）

        Args:
            mask: 原始掩模（二值或灰度）
            offset: 偏移量（像素）

        Returns:
            偏移后的二值掩模
        """
        mask_bin = mask >= 0.5

        if abs(offset) < 1e-3:
            return mask.copy()

        dist_inside = distance_transform_edt(mask_bin)
        dist_outside = distance_transform_edt(~mask_bin)

        signed_dist = np.where(mask_bin, dist_inside, -dist_outside)

        shifted = signed_dist + offset
        result = (shifted >= 0).astype(np.float64)

        return result

    def _check_convergence(self,
                            epe_stats: Dict[str, float],
                            iteration: int,
                            prev_epe: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
        """
        检查收敛条件

        Args:
            epe_stats: 当前 EPE 统计
            iteration: 当前迭代次数
            prev_epe: 上一轮 EPE 统计

        Returns:
            (是否收敛, 终止原因)
        """
        epe_mean = epe_stats.get('epe_mean', float('inf'))

        if epe_mean <= self.opc_config.epe_convergence_threshold:
            return True, f"EPE 已收敛到阈值以下: {epe_mean:.3f} nm <= {self.opc_config.epe_convergence_threshold} nm"

        if iteration >= self.opc_config.max_iterations:
            return True, f"达到最大迭代次数: {self.opc_config.max_iterations}"

        if prev_epe is not None:
            prev_mean = prev_epe.get('epe_mean', float('inf'))
            improvement = prev_mean - epe_mean
            if improvement < self.opc_config.epe_convergence_threshold * 0.05:
                return True, f"EPE 改善停滞: {improvement:.4f} nm"

        return False, ""

    def run(self,
            initial_mask: np.ndarray,
            target: np.ndarray,
            drift_rate_guess: float = 0.0) -> MPCOPCWorkflowResult:
        """
        运行 MPC-OPC 工作流

        Args:
            initial_mask: 初始掩模图案
            target: 目标图案（原始版图）
            drift_rate_guess: 初始漂移速率猜测 (nm/迭代)

        Returns:
            MPC-OPC 工作流结果
        """
        if self.config.verbose:
            logger.info("\n" + "=" * 60)
            logger.info("MPC-OPC 工作流开始")
            logger.info("=" * 60)

        self.predictor.reset()
        self.drift_estimator.reset()

        current_mask = initial_mask.copy()
        cumulative_offset = 0.0
        iterations: List[MPCOPCIterationResult] = []

        wafer_initial, initial_epe = self._simulate_epe(initial_mask, target)

        if self.config.verbose:
            logger.info(f"初始 EPE: mean={initial_epe['epe_mean']:.3f} nm, "
                       f"max={initial_epe['epe_max']:.3f} nm")

        self.predictor.update(initial_epe['epe_mean'], 0.0, step=0)
        self.drift_estimator.update(initial_epe['epe_mean'], 0.0, step=0)

        prev_epe = None
        converged = False
        reason = ""

        for iteration in range(1, self.opc_config.max_iterations + 1):
            if self.config.verbose:
                logger.info(f"\n{'=' * 50}")
                logger.info(f"MPC-OPC 迭代 {iteration}")
                logger.info(f"{'=' * 50}")

            wafer_before, epe_before = self._simulate_epe(current_mask, target)

            hotspots_before = self.hotspot_detector.detect(
                current_mask, target,
                wafer_binary=wafer_before,
                optical_system=self.optical_system
            )

            drift_rate = self.drift_estimator.get_drift_rate()
            if iteration == 1 and drift_rate_guess != 0.0:
                drift_rate = drift_rate_guess

            opt_result = self.mpc_optimizer.solve(
                current_epe=epe_before['epe_mean'],
                current_offset=cumulative_offset,
                drift_rate=drift_rate,
                start_step=iteration
            )

            applied_offset = opt_result.first_offset

            predicted_epe_mean = None
            if len(opt_result.predicted_epe) > 0:
                predicted_epe_mean = opt_result.predicted_epe[0]

            pred_result = PredictionResult(
                predicted_epe=opt_result.predicted_epe,
                predicted_epe_std=np.zeros_like(opt_result.predicted_epe),
                drift_rate=drift_rate,
            )

            new_mask = self._apply_global_edge_offset(current_mask, applied_offset)
            cumulative_offset += applied_offset

            wafer_after, epe_after = self._simulate_epe(new_mask, target)

            hotspots_after = self.hotspot_detector.detect(
                new_mask, target,
                wafer_binary=wafer_after,
                optical_system=self.optical_system
            )

            if self.config.online_update:
                self.predictor.update(epe_after['epe_mean'], cumulative_offset, step=iteration)
                self.drift_estimator.update(epe_after['epe_mean'], cumulative_offset, step=iteration)

            iter_result = MPCOPCIterationResult(
                iteration=iteration,
                mask_before=current_mask,
                mask_after=new_mask,
                epe_before=epe_before,
                epe_after=epe_after,
                edge_offset_applied=applied_offset,
                cumulative_offset=cumulative_offset,
                prediction_result=pred_result,
                optimization_result=opt_result,
                drift_rate=drift_rate,
                hotspots_before=hotspots_before,
                hotspots_after=hotspots_after,
            )
            iterations.append(iter_result)

            if self.config.verbose:
                logger.info(
                    f"迭代 {iteration}: EPE {epe_before['epe_mean']:.3f} → "
                    f"{epe_after['epe_mean']:.3f} nm "
                    f"(改善 {iter_result.epe_improvement:.3f} nm, "
                    f"{iter_result.epe_improvement_ratio * 100:.1f}%)"
                )
                logger.info(
                    f"边缘偏移: {applied_offset:+.3f} px (累计: {cumulative_offset:+.3f} px)"
                )
                logger.info(f"漂移速率估计: {drift_rate:.4f} nm/迭代")
                logger.info(f"热点数量: {len(hotspots_before)} → {len(hotspots_after)}")

            converged, reason = self._check_convergence(
                epe_after, iteration, epe_before
            )

            current_mask = new_mask
            prev_epe = epe_after

            if converged:
                if self.config.verbose:
                    logger.info(f"\n收敛: {reason}")
                break

        if not converged:
            reason = f"达到最大迭代次数: {self.opc_config.max_iterations}"

        final_wafer, final_epe = self._simulate_epe(current_mask, target)

        result = MPCOPCWorkflowResult(
            initial_mask=initial_mask,
            corrected_mask=current_mask,
            initial_epe=initial_epe,
            final_epe=final_epe,
            iterations=iterations,
            converged=converged,
            reason=reason,
            final_drift_rate=self.drift_estimator.get_drift_rate(),
        )

        if self.config.verbose:
            logger.info("\n" + "=" * 60)
            logger.info("MPC-OPC 工作流完成")
            logger.info(f"初始 EPE: {initial_epe['epe_mean']:.3f} nm")
            logger.info(f"最终 EPE: {final_epe['epe_mean']:.3f} nm")
            logger.info(f"总改善: {result.total_epe_improvement:.3f} nm ({result.total_epe_improvement_ratio * 100:.1f}%)")
            logger.info(f"迭代次数: {result.num_iterations}")
            logger.info(f"累计偏移: {cumulative_offset:+.3f} px")
            logger.info(f"最终漂移率: {result.final_drift_rate:.4f} nm/迭代")
            logger.info("=" * 60)

        return result

    def run_with_batch_adaptation(self,
                                   initial_mask: np.ndarray,
                                   target: np.ndarray,
                                   historical_epe_list: Optional[List[float]] = None,
                                   historical_offset_list: Optional[List[float]] = None) -> MPCOPCWorkflowResult:
        """
        带历史批次数据的在线自适应运行

        利用历史批次数据预热预测模型，更好地应对批次间变化。

        Args:
            initial_mask: 初始掩模
            target: 目标图案
            historical_epe_list: 历史批次的 EPE 数据
            historical_offset_list: 历史批次的偏移量数据

        Returns:
            MPC-OPC 工作流结果
        """
        if historical_epe_list is not None and historical_offset_list is not None:
            if len(historical_epe_list) == len(historical_offset_list):
                if self.config.verbose:
                    logger.info(f"使用 {len(historical_epe_list)} 条历史数据预热预测模型")
                for i, (epe, offset) in enumerate(zip(historical_epe_list, historical_offset_list)):
                    self.predictor.update(epe, offset, step=-(len(historical_epe_list) - i))
                    self.drift_estimator.update(epe, offset, step=-(len(historical_epe_list) - i))

        return self.run(initial_mask, target)


def run_mpc_opc_workflow(
        initial_mask: np.ndarray,
        target: np.ndarray,
        config: Optional[MPCConfig] = None,
        optical_system: Optional[OpticalSystem] = None,
        **kwargs) -> MPCOPCWorkflowResult:
    """
    便捷函数：运行 MPC-OPC 工作流

    Args:
        initial_mask: 初始掩模图案
        target: 目标图案
        config: MPC 配置
        optical_system: 光学系统参数
        **kwargs: 额外配置参数，将覆盖 config 中的对应项

    Returns:
        MPC-OPC 工作流结果
    """
    if config is None:
        config = MPCConfig()

    if kwargs:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    workflow = MPCOPCWorkflow(config=config, optical_system=optical_system)
    return workflow.run(initial_mask, target)
