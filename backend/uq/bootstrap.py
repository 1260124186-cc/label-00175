# -*- coding: utf-8 -*-
"""
Bootstrap 重采样分析模块

提供多种 Bootstrap 方法用于不确定性量化：
- 非参数 Bootstrap (Non-parametric Bootstrap)
- 残差 Bootstrap (Residual Bootstrap)
- 百分位数法置信区间
- BCa (Bias-corrected and accelerated) 置信区间
- 正态近似置信区间

适用于：
1. 已有蒙特卡洛仿真样本的不确定性量化
2. 模型参数估计的置信区间计算
3. 失效率、P PM 等小概率事件的不确定性估计
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats

from uq.schemas import ConfidenceInterval

logger = logging.getLogger(__name__)


class BootstrapMethod(Enum):
    """Bootstrap 方法枚举"""
    NONPARAMETRIC = "nonparametric"
    RESIDUAL = "residual"
    PARAMETRIC = "parametric"
    BAYESIAN = "bayesian"


@dataclass
class BootstrapConfig:
    """
    Bootstrap 配置

    Attributes:
        n_bootstrap: Bootstrap 重采样次数
        method: Bootstrap 方法
        ci_method: 置信区间计算方法
            - 'percentile': 百分位数法
            - 'normal': 正态近似
            - 'bca': BCa 方法 (Bias-corrected and accelerated)
            - 'basic': 基本 bootstrap 区间
        confidence_level: 置信水平（默认 0.95）
        statistic: 要计算的统计量函数，None 则计算均值
        random_seed: 随机种子
    """
    n_bootstrap: int = 1000
    method: BootstrapMethod = BootstrapMethod.NONPARAMETRIC
    ci_method: str = "percentile"
    confidence_level: float = 0.95
    statistic: Optional[Callable[[np.ndarray], float]] = None
    random_seed: Optional[int] = None


@dataclass
class BootstrapResult:
    """
    Bootstrap 分析结果

    Attributes:
        original_statistic: 原始样本的统计量值
        bootstrap_statistics: 所有 Bootstrap 样本的统计量值数组
        standard_error: Bootstrap 标准误
        confidence_intervals: 不同方法的置信区间
        bias: Bootstrap 偏差估计
        bias_corrected: 偏差校正后的统计量估计
        n_bootstrap: 实际执行的 Bootstrap 次数
        convergence: 是否收敛（当 n_bootstrap 足够大时标准误稳定）
    """
    original_statistic: float
    bootstrap_statistics: np.ndarray
    standard_error: float
    confidence_intervals: Dict[str, ConfidenceInterval] = field(default_factory=dict)
    bias: Optional[float] = None
    bias_corrected: Optional[float] = None
    n_bootstrap: int = 0
    convergence: bool = False

    @property
    def original_stat(self) -> float:
        return self.original_statistic

    @property
    def converged(self) -> bool:
        return self.convergence

    @property
    def bias_corrected_estimate(self) -> Optional[float]:
        return self.bias_corrected

    def to_dict(self, include_bootstrap_stats: bool = False) -> Dict[str, Any]:
        result = {
            "original_statistic": float(self.original_statistic),
            "standard_error": float(self.standard_error),
            "confidence_intervals": {
                k: v.to_dict() for k, v in self.confidence_intervals.items()
            },
            "bias": float(self.bias) if self.bias is not None else None,
            "bias_corrected": float(self.bias_corrected) if self.bias_corrected is not None else None,
            "n_bootstrap": int(self.n_bootstrap),
            "convergence": bool(self.convergence),
        }
        if include_bootstrap_stats:
            result["bootstrap_statistics"] = self.bootstrap_statistics.tolist()
        return result

    def summary(self) -> str:
        lines = ["=== Bootstrap 分析结果 ==="]
        lines.append(f"  原始统计量值: {self.original_statistic:.6f}")
        lines.append(f"  Bootstrap 次数: {self.n_bootstrap}")
        lines.append(f"  标准误 (SE): {self.standard_error:.6f}")
        if self.bias is not None:
            lines.append(f"  偏差估计: {self.bias:.6f}")
        if self.bias_corrected is not None:
            lines.append(f"  偏差校正估计: {self.bias_corrected:.6f}")
        for name, ci in self.confidence_intervals.items():
            lines.append(f"  {name}: {ci.summary()}")
        return "\n".join(lines)


class BootstrapAnalyzer:
    """
    Bootstrap 分析器

    提供多种 Bootstrap 方法用于不确定性量化。
    """

    def __init__(self, config: Optional[BootstrapConfig] = None):
        """
        初始化 Bootstrap 分析器

        Args:
            config: Bootstrap 配置
        """
        self.config = config if config is not None else BootstrapConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def _compute_statistic(self, data: np.ndarray) -> float:
        """计算统计量"""
        if self.config.statistic is not None:
            return float(self.config.statistic(data))
        return float(np.mean(data))

    def nonparametric_bootstrap(
        self,
        data: np.ndarray,
        n_bootstrap: Optional[int] = None,
    ) -> BootstrapResult:
        """
        非参数 Bootstrap

        通过有放回地从原始数据中重采样来估计统计量的不确定性。

        Args:
            data: 原始样本数据，形状 (n_samples,) 或 (n_samples, n_features)
            n_bootstrap: 重采样次数，None 则使用配置值

        Returns:
            BootstrapResult
        """
        if n_bootstrap is None:
            n_bootstrap = self.config.n_bootstrap

        data = np.asarray(data)
        n = len(data)
        if n < 2:
            raise ValueError("样本量至少需要 2 个才能进行 Bootstrap")

        original_stat = self._compute_statistic(data)

        bootstrap_stats = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            indices = self.rng.integers(0, n, size=n)
            sample = data[indices]
            bootstrap_stats[i] = self._compute_statistic(sample)

        return self._build_result(original_stat, bootstrap_stats, data)

    def residual_bootstrap(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
        model_predict: Callable[[np.ndarray], np.ndarray],
        x: Optional[np.ndarray] = None,
        n_bootstrap: Optional[int] = None,
    ) -> BootstrapResult:
        """
        残差 Bootstrap

        适用于回归模型的不确定性估计：对残差重采样，生成新的响应变量。

        Args:
            y: 观测值
            y_pred: 模型预测值
            model_predict: 模型预测函数 predict(x) -> y_pred
            x: 自变量（可选）
            n_bootstrap: 重采样次数

        Returns:
            BootstrapResult
        """
        if n_bootstrap is None:
            n_bootstrap = self.config.n_bootstrap

        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        residuals = y - y_pred
        n = len(residuals)

        original_stat = self._compute_statistic(y_pred)

        bootstrap_stats = np.zeros(n_bootstrap)
        centered_residuals = residuals - np.mean(residuals)

        for i in range(n_bootstrap):
            indices = self.rng.integers(0, n, size=n)
            resampled_residuals = centered_residuals[indices]
            y_star = y_pred + resampled_residuals

            if x is not None:
                x_star = x[indices] if x.ndim > 1 else x[indices]
                y_pred_star = model_predict(x_star)
            else:
                y_pred_star = model_predict(y_star)

            bootstrap_stats[i] = self._compute_statistic(y_pred_star)

        return self._build_result(original_stat, bootstrap_stats, y_pred)

    def parametric_bootstrap(
        self,
        data: np.ndarray,
        distribution: str = "normal",
        n_bootstrap: Optional[int] = None,
    ) -> BootstrapResult:
        """
        参数 Bootstrap

        假设数据服从某种分布，先拟合分布参数，再从拟合的分布中采样。

        Args:
            data: 原始样本数据
            distribution: 分布类型 ('normal', 'lognormal', 'gamma', 'exponential', 'poisson')
            n_bootstrap: 重采样次数

        Returns:
            BootstrapResult
        """
        if n_bootstrap is None:
            n_bootstrap = self.config.n_bootstrap

        data = np.asarray(data)
        n = len(data)
        original_stat = self._compute_statistic(data)

        if distribution == "normal":
            mu, sigma = np.mean(data), np.std(data, ddof=1)
        elif distribution == "lognormal":
            log_data = np.log(np.maximum(data, 1e-10))
            mu, sigma = np.mean(log_data), np.std(log_data, ddof=1)
        elif distribution == "gamma":
            shape, loc, scale = stats.gamma.fit(data, floc=0)
        elif distribution == "exponential":
            loc, scale = stats.expon.fit(data, floc=0)
        else:
            mu, sigma = np.mean(data), np.std(data, ddof=1)
            distribution = "normal"

        bootstrap_stats = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            if distribution == "normal":
                sample = self.rng.normal(mu, sigma, size=n)
            elif distribution == "lognormal":
                sample = self.rng.lognormal(mu, sigma, size=n)
            elif distribution == "gamma":
                sample = self.rng.gamma(shape, scale, size=n)
            elif distribution == "exponential":
                sample = self.rng.exponential(scale, size=n) + loc
            else:
                sample = self.rng.normal(mu, sigma, size=n)

            bootstrap_stats[i] = self._compute_statistic(sample)

        return self._build_result(original_stat, bootstrap_stats, data)

    def _build_result(
        self,
        original_stat: float,
        bootstrap_stats: np.ndarray,
        data: np.ndarray,
    ) -> BootstrapResult:
        """构建 BootstrapResult 结果"""
        se = float(np.std(bootstrap_stats, ddof=1))
        bias = float(np.mean(bootstrap_stats) - original_stat)
        bias_corrected = float(original_stat - bias)

        result = BootstrapResult(
            original_statistic=original_stat,
            bootstrap_statistics=bootstrap_stats,
            standard_error=se,
            bias=bias,
            bias_corrected=bias_corrected,
            n_bootstrap=len(bootstrap_stats),
            convergence=self._check_convergence(bootstrap_stats),
        )

        alpha = 1 - self.config.confidence_level

        ci_percentile = ConfidenceInterval(
            lower=float(np.percentile(bootstrap_stats, 100 * alpha / 2)),
            upper=float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))),
            level=self.config.confidence_level,
            method="percentile",
            point_estimate=float(np.median(bootstrap_stats)),
            standard_error=se,
        )
        result.confidence_intervals["percentile"] = ci_percentile

        z = stats.norm.ppf(1 - alpha / 2)
        ci_normal = ConfidenceInterval(
            lower=float(original_stat - z * se),
            upper=float(original_stat + z * se),
            level=self.config.confidence_level,
            method="normal",
            point_estimate=original_stat,
            standard_error=se,
        )
        result.confidence_intervals["normal"] = ci_normal

        try:
            ci_bca = self._compute_bca_ci(
                data, bootstrap_stats, original_stat, self.config.confidence_level
            )
            result.confidence_intervals["bca"] = ci_bca
        except Exception as e:
            logger.debug(f"BCa 区间计算失败: {e}")

        ci_basic = ConfidenceInterval(
            lower=float(2 * original_stat - np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))),
            upper=float(2 * original_stat - np.percentile(bootstrap_stats, 100 * alpha / 2)),
            level=self.config.confidence_level,
            method="basic",
            point_estimate=original_stat,
            standard_error=se,
        )
        result.confidence_intervals["basic"] = ci_basic

        return result

    def _compute_bca_ci(
        self,
        data: np.ndarray,
        bootstrap_stats: np.ndarray,
        original_stat: float,
        confidence_level: float,
    ) -> ConfidenceInterval:
        """
        计算 BCa (Bias-corrected and accelerated) 置信区间

        BCa 方法对偏差和偏度进行校正，在大多数情况下优于普通百分位数法。
        """
        alpha = 1 - confidence_level
        n = len(data)

        prop_less = np.mean(bootstrap_stats < original_stat)
        prop_less = min(max(prop_less, 1e-10), 1 - 1e-10)
        z0 = stats.norm.ppf(prop_less)

        jackknife_stats = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            jack_sample = data[mask]
            jackknife_stats[i] = self._compute_statistic(jack_sample)

        jack_mean = np.mean(jackknife_stats)
        numerator = np.sum((jack_mean - jackknife_stats) ** 3)
        denominator = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5
        a_hat = numerator / denominator if abs(denominator) > 1e-15 else 0.0

        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)

        def adjust_z(z_val):
            return z0 + (z0 + z_val) / (1 - a_hat * (z0 + z_val))

        alpha1 = stats.norm.cdf(adjust_z(z_alpha_low))
        alpha2 = stats.norm.cdf(adjust_z(z_alpha_high))

        lower = float(np.percentile(bootstrap_stats, 100 * alpha1))
        upper = float(np.percentile(bootstrap_stats, 100 * alpha2))

        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            level=confidence_level,
            method="bca",
            point_estimate=float(np.median(bootstrap_stats)),
            standard_error=float(np.std(bootstrap_stats, ddof=1)),
        )

    def _check_convergence(
        self, bootstrap_stats: np.ndarray, window_size: int = 50
    ) -> bool:
        """检查 Bootstrap 是否收敛"""
        n = len(bootstrap_stats)
        if n < 100:
            return n >= 30

        cumsum = np.cumsum(bootstrap_stats)
        running_mean = cumsum / np.arange(1, n + 1)

        ws = min(window_size, n // 4)
        recent_means = running_mean[-ws:]
        total_std = np.std(bootstrap_stats)
        if total_std < 1e-12:
            return True
        std_ratio = np.std(recent_means) / total_std
        return std_ratio < 0.05

    def analyze(
        self,
        data: np.ndarray,
        method: Optional[BootstrapMethod] = None,
        **kwargs,
    ) -> BootstrapResult:
        """
        综合分析接口

        Args:
            data: 样本数据
            method: 方法类型，None 则使用配置中的方法
            **kwargs: 传递给具体方法的额外参数

        Returns:
            BootstrapResult
        """
        if method is None:
            method = self.config.method

        if method == BootstrapMethod.NONPARAMETRIC:
            return self.nonparametric_bootstrap(data, **kwargs)
        elif method == BootstrapMethod.PARAMETRIC:
            return self.parametric_bootstrap(data, **kwargs)
        elif method == BootstrapMethod.RESIDUAL:
            return self.residual_bootstrap(data, **kwargs)
        else:
            return self.nonparametric_bootstrap(data, **kwargs)


def bootstrap_ci(
    data: np.ndarray,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    ci_method: str = "percentile",
    statistic: Optional[Callable[[np.ndarray], float]] = None,
    random_seed: Optional[int] = None,
) -> ConfidenceInterval:
    """
    便捷函数：计算 Bootstrap 置信区间

    Args:
        data: 样本数据
        confidence_level: 置信水平
        n_bootstrap: 重采样次数
        ci_method: 置信区间方法 ('percentile', 'normal', 'bca', 'basic')
        statistic: 统计量函数，None 则计算均值
        random_seed: 随机种子

    Returns:
        ConfidenceInterval
    """
    config = BootstrapConfig(
        n_bootstrap=n_bootstrap,
        ci_method=ci_method,
        confidence_level=confidence_level,
        statistic=statistic,
        random_seed=random_seed,
    )
    analyzer = BootstrapAnalyzer(config)
    result = analyzer.nonparametric_bootstrap(data)

    if ci_method in result.confidence_intervals:
        return result.confidence_intervals[ci_method]
    return result.confidence_intervals["percentile"]
