# -*- coding: utf-8 -*-
"""
失效概率与风险评估模块

基于蒙特卡洛仿真结果，评估随机效应导致的失效概率，
分析各随机过程的敏感度，为随机效应敏感结构提供风险评估。

核心功能：
1. 失效模式识别（桥接、断线、CD超标等）
2. 失效概率估计（蒙特卡洛估计、重要性采样、外推法）
3. 敏感度分析（Sobol指数、SALib集成）
4. 风险等级划分
5. 风险可视化指标（RPN风险优先数
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.stats import norm, lognorm, beta, gamma, gaussian_kde

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """失效模式枚举"""
    CD_OVERSHOOT = "cd_overshoot"           # CD 超标（过大）
    CD_UNDERSHOOT = "cd_undershoot"       # CD 不足（过小）
    BRIDGING = "bridging"                   # 桥接（相邻线条短路）
    OPEN_CIRCUIT = "open_circuit"           # 断线（线条断开）
    EDGE_ROUGHNESS_EXCEED = "ler_exceed"     # 边缘粗糙度超标
    LWR_EXCEED = "lwr_exceed"             # 线宽粗糙度超标
    EPE_EXCEED = "epe_exceed"             # 边缘放置误差超标
    PATTERN_FAILURE = "pattern_failure"       # 图形完全失效


class RiskLevel(Enum):
    """风险等级枚举"""
    VERY_LOW = "very_low"    # 极低风险
    LOW = "low"            # 低风险
    MEDIUM = "medium"      # 中等风险
    HIGH = "high"          # 高风险
    VERY_HIGH = "very_high"  # 极高风险


class SensitivityMetric(Enum):
    """敏感度度量类型"""
    FIRST_ORDER = "first_order"      # 一阶敏感度
    TOTAL_ORDER = "total_order"        # 总敏感度
    MUTUAL_INFORMATION = "mi"         # 互信息
    PEARSON_CORRELATION = "pearson"   # 皮尔逊相关系数
    SPEARMAN_RANK = "spearman"         # 斯皮尔曼秩相关


@dataclass
class FailureCriteria:
    """
    失效判据配置

    Attributes:
        cd_target: CD 目标值 (nm)
        cd_tolerance: CD 公差 (nm)
        cd_lower_limit: CD 下限 (nm)
        cd_upper_limit: CD 上限 (nm)
        ler_limit: LER 上限 (nm)
        lwr_limit: LWR 上限 (nm)
        epe_limit: EPE 上限 (nm)
        bridging_threshold: 桥接判定阈值（最小间距)
        min_cd_for_break: 断线判定阈值（最小 CD）
    """
    cd_target: Optional[float] = None
    cd_tolerance: Optional[float] = None
    cd_lower_limit: Optional[float] = None
    cd_upper_limit: Optional[float] = None
    ler_limit: Optional[float] = None
    lwr_limit: Optional[float] = None
    epe_limit: Optional[float] = None
    bridging_threshold: Optional[float] = None
    min_cd_for_break: Optional[float] = None

    def get_cd_limits(self) -> Tuple[Optional[float], Optional[float]]:
        """获取 CD 上下限"""
        lower = self.cd_lower_limit
        upper = self.cd_upper_limit

        if lower is None and self.cd_target is not None and self.cd_tolerance is not None:
            lower = self.cd_target - self.cd_tolerance
            upper = self.cd_target + self.cd_tolerance

        return lower, upper


@dataclass
class FailureCount:
    """
    失效计数结果

    Attributes:
        total_samples: 总样本数
        failed_samples: 失效样本数
        failure_count: 各失效模式的失效次数
        failure_probability: 各失效模式的失效概率
    """
    total_samples: int = 0
    failed_samples: int = 0
    failure_count: Dict[str, int] = field(default_factory=dict)
    failure_probability: Dict[str, float] = field(default_factory=dict)


@dataclass
class FailureProbabilityResult:
    """
    失效概率估计结果

    Attributes:
        failure_mode: 失效模式
        estimate_method: 估计方法
        probability: 失效概率估计值
        confidence_interval: 置信区间 (lower, upper)
        confidence_level: 置信水平
        standard_error: 标准误差
        ppm: PPM 缺陷率
        log_ppm: 对数 PPM
    """
    failure_mode: str
    estimate_method: str
    probability: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    standard_error: float
    ppm: float
    log_ppm: float

    def summary(self) -> str:
        """生成摘要字符串"""
        lines = [
            f"=== {self.failure_mode} 失效概率 ===",
            f"  估计方法: {self.estimate_method}",
            f"  失效概率: {self.probability:.2e}",
            f"  {int(self.confidence_level * 100):.0f}% 置信区间: [{self.confidence_interval[0]:.2e}, {self.confidence_interval[1]:.2e}]",
            f"  标准误差: {self.standard_error:.2e}",
            f"  PPM 缺陷率: {self.ppm:.2f} ppm",
        ]
        return "\n".join(lines)


@dataclass
class SensitivityAnalysisResult:
    """
    敏感度分析结果

    Attributes:
        parameter: 参数名称
        metric: 敏感度度量类型
        value: 敏感度值
        confidence_interval: 置信区间
        rank: 敏感度排序
    """
    parameter: str
    metric: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    rank: Optional[int] = None


@dataclass
class RiskAssessmentResult:
    """
    风险评估完整结果

    Attributes:
        overall_failure_probability: 总体失效概率
        failure_probabilities: 各失效模式的失效概率结果
        sensitivity_analysis: 敏感度分析结果
        risk_level: 整体风险等级
        risk_levels: 各失效模式的风险等级
        risk_priority_number: RPN 风险优先数
        risk_mitigation: 风险缓解建议
    """
    overall_failure_probability: float = 0.0
    failure_probabilities: Dict[str, FailureProbabilityResult] = field(default_factory=dict)
    sensitivity_analysis: List[SensitivityAnalysisResult] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    risk_levels: Dict[str, RiskLevel] = field(default_factory=dict)
    risk_priority_number: Dict[str, float] = field(default_factory=dict)
    risk_mitigation: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """生成摘要字符串"""
        lines = [
            "=== 随机效应风险评估结果 ===",
            f"  总体失效概率: {self.overall_failure_probability:.2e}",
            f"  整体风险等级: {self.risk_level.value}",
        ]

        if self.failure_probabilities:
            lines.append("")
            lines.append("  各失效模式概率:")
            for mode, result in self.failure_probabilities.items():
                lines.append(f"    {mode}: {result.probability:.2e} ({result.ppm:.1f} ppm)")

        if self.sensitivity_analysis:
            lines.append("")
            lines.append("  关键敏感度分析:")
            for sa in sorted(self.sensitivity_analysis):
                rank_str = f"#{sa.rank} " if sa.rank else ""
                lines.append(f"    {rank_str}{sa.parameter}: {sa.value:.3f} ({sa.metric})")

        if self.risk_mitigation:
            lines.append("")
            lines.append("  风险缓解建议:")
            for i, mitigation in enumerate(self.risk_mitigation, 1):
                lines.append(f"    {i}. {mitigation}")

        return "\n".join(lines)


class FailureProbabilityEstimator:
    """
    失效概率估计器

    提供多种失效概率估计方法：
    1. 蒙特卡洛直接估计
    2. 二项分布置信区间
    3. 基于分布的外推估计
    4. 重要性采样（预留接口）
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        初始化估计器

        Args:
            confidence_level: 置信水平
        """
        self.confidence_level = confidence_level

    def estimate_monte_carlo(
        self,
        failure_indicators: np.ndarray,
        failure_mode: str,
    ) -> FailureProbabilityResult:
        """
        蒙特卡洛直接估计失效概率

        Args:
            failure_indicators: 失效指示数组（True表示失效）
            failure_mode: 失效模式名称

        Returns:
            失效概率估计结果
        """
        n = len(failure_indicators)
        if n == 0:
            return FailureProbabilityResult(
                failure_mode=failure_mode,
                estimate_method="monte_carlo",
                probability=0.0,
                confidence_interval=(0.0, 0.0),
                confidence_level=self.confidence_level,
                standard_error=0.0,
                ppm=0.0,
                log_ppm=0.0,
            )

        n_fail = np.sum(failure_indicators)
        p_hat = n_fail / n
        se = np.sqrt(p_hat * (1 - p_hat) / np.sqrt(n))

        alpha = 1 - self.confidence_level
        z = norm.ppf(1 - alpha / 2)
        ci_lower = max(0.0, p_hat - z * se)
        ci_upper = min(1.0, p_hat + z * se)

        ppm = p_hat * 1e6
        log_ppm = -np.log10(max(p_hat, 1e-10)) + 6

        return FailureProbabilityResult(
            failure_mode=failure_mode,
            estimate_method="monte_carlo",
            probability=float(p_hat),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            confidence_level=self.confidence_level,
            standard_error=float(se),
            ppm=float(ppm),
            log_ppm=float(log_ppm),
        )

    def estimate_extrapolation(
        self,
        metric_values: np.ndarray,
        limit_value: float,
        failure_mode: str,
        distribution_type: str = "normal",
        tail: str = "upper",
    ) -> FailureProbabilityResult:
        """
        基于分布拟合的尾部分布外推估计

        Args:
            metric_values: 度量值数组
            limit_value: 失效阈值
            failure_mode: 失效模式名称
            distribution_type: 分布类型 ('normal', 'lognormal', 'gamma', 'kde')
            tail: 尾部类型 ('upper' 或 'lower')

        Returns:
            失效概率估计结果
        """
        n = len(metric_values)
        if n < 10:
            logger.warning("样本量过小，使用蒙特卡洛估计替代")
            if tail == "upper":
                indicators = metric_values > limit_value
            else:
                indicators = metric_values < limit_value
            return self.estimate_monte_carlo(indicators, failure_mode)

        try:
            if distribution_type == "normal":
                mu, sigma = np.mean(metric_values), np.std(metric_values)
                if tail == "upper":
                    p_fail = 1 - norm.cdf(limit_value, mu, sigma)
                else:
                    p_fail = norm.cdf(limit_value, mu, sigma)
                se = self._bootstrap_se(metric_values, limit_value, tail, "normal")
                estimate_method = "normal_extrapolation"

            elif distribution_type == "lognormal":
                shape, loc, scale = lognorm.fit(metric_values, floc=0)
                if tail == "upper":
                    p_fail = 1 - lognorm.cdf(limit_value, shape, loc, scale)
                else:
                    p_fail = lognorm.cdf(limit_value, shape, loc, scale)
                se = self._bootstrap_se(metric_values, limit_value, tail, "lognormal")
                estimate_method = "lognormal_extrapolation"

            elif distribution_type == "gamma":
                shape, loc, scale = gamma.fit(metric_values, floc=0)
                if tail == "upper":
                    p_fail = 1 - gamma.cdf(limit_value, shape, loc, scale)
                else:
                    p_fail = gamma.cdf(limit_value, shape, loc, scale)
                se = self._bootstrap_se(metric_values, limit_value, tail, "gamma")
                estimate_method = "gamma_extrapolation"

            elif distribution_type == "kde":
                kde = gaussian_kde(metric_values)
                if tail == "upper":
                    p_fail = 1 - kde.integrate_box_1d(limit_value, np.inf)
                else:
                    p_fail = kde.integrate_box_1d(-np.inf, limit_value)
                se = self._bootstrap_se(metric_values, limit_value, tail, "kde")
                estimate_method = "kde_extrapolation"

            else:
                raise ValueError(f"未知分布类型: {distribution_type}")

        except Exception as e:
            logger.warning(f"分布拟合失败 ({e})，使用蒙特卡洛估计")
            if tail == "upper":
                indicators = metric_values > limit_value
            else:
                indicators = metric_values < limit_value
            return self.estimate_monte_carlo(indicators, failure_mode)

        alpha = 1 - self.confidence_level
        z = norm.ppf(1 - alpha / 2)
        ci_lower = max(0.0, p_fail - z * se)
        ci_upper = min(1.0, p_fail + z * se)

        ppm = p_fail * 1e6
        log_ppm = -np.log10(max(p_fail, 1e-10)) + 6

        return FailureProbabilityResult(
            failure_mode=failure_mode,
            estimate_method=estimate_method,
            probability=float(p_fail),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            confidence_level=self.confidence_level,
            standard_error=float(se),
            ppm=float(ppm),
            log_ppm=float(log_ppm),
        )

    def _bootstrap_se(
        self,
        metric_values: np.ndarray,
        limit_value: float,
        tail: str,
        distribution_type: str,
        n_bootstrap: int = 100,
    ) -> float:
        """
        Bootstrap 估计标准误差

        Args:
            metric_values: 度量值数组
            limit_value: 失效阈值
            tail: 尾部类型
            distribution_type: 分布类型
            n_bootstrap: Bootstrap 次数

        Returns:
            标准误差
        """
        n = len(metric_values)
        if n < 20:
            return np.sqrt(0.01 * (1 - 0.01) / np.sqrt(n))

        p_estimates = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(metric_values, size=n, replace=True)
            try:
                if distribution_type == "normal":
                    mu, sigma = np.mean(sample), np.std(sample)
                    if tail == "upper":
                        p = 1 - norm.cdf(limit_value, mu, sigma)
                    else:
                        p = norm.cdf(limit_value, mu, sigma)
                elif distribution_type == "lognormal":
                    shape, loc, scale = lognorm.fit(sample, floc=0)
                    if tail == "upper":
                        p = 1 - lognorm.cdf(limit_value, shape, loc, scale)
                    else:
                        p = lognorm.cdf(limit_value, shape, loc, scale)
                elif distribution_type == "gamma":
                    shape, loc, scale = gamma.fit(sample, floc=0)
                    if tail == "upper":
                        p = 1 - gamma.cdf(limit_value, shape, loc, scale)
                    else:
                        p = gamma.cdf(limit_value, shape, loc, scale)
                elif distribution_type == "kde":
                    kde = gaussian_kde(sample)
                    if tail == "upper":
                        p = 1 - kde.integrate_box_1d(limit_value, np.inf)
                    else:
                        p = kde.integrate_box_1d(-np.inf, limit_value)
                else:
                    continue
                p_estimates.append(p)
            except Exception:
                continue

        if len(p_estimates) < 10:
            return np.sqrt(0.01 * (1 - 0.01) / np.sqrt(n))

        return float(np.std(p_estimates))


class FailureDetector:
    """
    失效检测器

    从蒙特卡洛仿真结果中检测各种失效模式。
    """

    def __init__(self, criteria: FailureCriteria):
        """
        初始化检测器

        Args:
            criteria: 失效判据
        """
        self.criteria = criteria

    def detect_cd_failure(
        self,
        cd_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测 CD 失效

        Args:
            cd_values: CD 值数组

        Returns:
            (cd_overshoot_indicators, cd_undershoot_indicators)
        """
        lower_limit, upper_limit = self.criteria.get_cd_limits()

        overshoot = np.zeros_like(cd_values, dtype=bool)
        undershoot = np.zeros_like(cd_values, dtype=bool)

        if upper_limit is not None:
            overshoot = cd_values > upper_limit

        if lower_limit is not None:
            undershoot = cd_values < lower_limit

        return overshoot, undershoot

    def detect_ler_failure(
        self,
        ler_values: np.ndarray,
    ) -> np.ndarray:
        """
        检测 LER 失效

        Args:
            ler_values: LER 值数组

        Returns:
            LER 失效指示数组
        """
        if self.criteria.ler_limit is None:
            return np.zeros_like(ler_values, dtype=bool)

        return ler_values > self.criteria.ler_limit

    def detect_lwr_failure(
        self,
        lwr_values: np.ndarray,
    ) -> np.ndarray:
        """
        检测 LWR 失效

        Args:
            lwr_values: LWR 值数组

        Returns:
            LWR 失效指示数组
        """
        if self.criteria.lwr_limit is None:
            return np.zeros_like(lwr_values, dtype=bool)

        return lwr_values > self.criteria.lwr_limit

    def detect_epe_failure(
        self,
        epe_values: np.ndarray,
    ) -> np.ndarray:
        """
        检测 EPE 失效

        Args:
            epe_values: EPE 值数组

        Returns:
            EPE 失效指示数组
        """
        if self.criteria.epe_limit is None:
            return np.zeros_like(epe_values, dtype=bool)

        return np.abs(epe_values) > self.criteria.epe_limit

    def detect_bridging_failure(
        self,
        wafer_binary_list: List[np.ndarray],
        nominal_wafer: np.ndarray,
        pixel_size: float = 1.0,
    ) -> np.ndarray:
        """
        检测桥接失效

        Args:
            wafer_binary_list: 二值化晶圆图列表
            nominal_wafer: 标称晶圆图
            pixel_size: 像素尺寸

        Returns:
            桥接失效指示数组
        """
        n = len(wafer_binary_list)
        indicators = np.zeros(n, dtype=bool)

        if self.criteria.bridging_threshold is None:
            return indicators

        nominal_pattern = nominal_wafer > 0.5

        for i, wafer in enumerate(wafer_binary_list):
            extra_material = (wafer > 0.5) & (~nominal_pattern)

            if np.any(extra_material):
                kernel = np.ones((3, 3))
                from scipy.ndimage import label
                labeled, n_regions = label(extra_material)

                if n_regions > 0:
                    for region_id in range(1, n_regions + 1):
                        region = labeled == region_id
                        region_size = np.sum(region) * pixel_size ** 2
                        if region_size > self.criteria.bridging_threshold:
                            indicators[i] = True
                            break

        return indicators

    def detect_open_circuit_failure(
        self,
        cd_values: np.ndarray,
        wafer_binary_list: List[np.ndarray],
    ) -> np.ndarray:
        """
        检测断线失效

        Args:
            cd_values: CD 值数组
            wafer_binary_list: 二值化晶圆图列表

        Returns:
            断线失效指示数组
        """
        n = len(wafer_binary_list)
        indicators = np.zeros(n, dtype=bool)

        for i, wafer in enumerate(wafer_binary_list):
            if self.criteria.min_cd_for_break is not None:
                if np.any(cd_values[i] < self.criteria.min_cd_for_break):
                    indicators[i] = True
                    continue

            pattern = wafer > 0.5
            if np.sum(pattern) == 0:
                indicators[i] = True

        return indicators

    def detect_all_failures(
        self,
        cd_values: Optional[np.ndarray],
        ler_values: Optional[np.ndarray] = None,
        lwr_values: Optional[np.ndarray] = None,
        epe_values: Optional[np.ndarray] = None,
        wafer_binary_list: Optional[List[np.ndarray]] = None,
        nominal_wafer: Optional[np.ndarray] = None,
        pixel_size: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        检测所有失效模式

        Args:
            cd_values: CD 值数组
            ler_values: LER 值数组
            lwr_values: LWR 值数组
            epe_values: EPE 值数组
            wafer_binary_list: 二值化晶圆图列表
            nominal_wafer: 标称晶圆图
            pixel_size: 像素尺寸

        Returns:
            各失效模式的指示数组字典
        """
        failures = {}

        if cd_values is not None:
            overshoot, undershoot = self.detect_cd_failure(cd_values)
            failures[FailureMode.CD_OVERSHOOT.value] = overshoot
            failures[FailureMode.CD_UNDERSHOOT.value] = undershoot

        if ler_values is not None:
            failures[FailureMode.EDGE_ROUGHNESS_EXCEED.value] = self.detect_ler_failure(ler_values)

        if lwr_values is not None:
            failures[FailureMode.LWR_EXCEED.value] = self.detect_lwr_failure(lwr_values)

        if epe_values is not None:
            failures[FailureMode.EPE_EXCEED.value] = self.detect_epe_failure(epe_values)

        if wafer_binary_list is not None and nominal_wafer is not None:
            failures[FailureMode.BRIDGING.value] = self.detect_bridging_failure(
                wafer_binary_list, nominal_wafer, pixel_size
            )
            failures[FailureMode.OPEN_CIRCUIT.value] = self.detect_open_circuit_failure(
                cd_values if cd_values is not None else np.zeros(len(wafer_binary_list)),
                wafer_binary_list,
            )

        return failures


class SensitivityAnalyzer:
    """
    敏感度分析器

    分析各随机参数对输出度量的敏感度。
    """

    def __init__(self):
        self.confidence_level = 0.95

    def analyze_pearson(
        self,
        parameter_samples: np.ndarray,
        output_metric: np.ndarray,
        parameter_names: List[str],
    ) -> List[SensitivityAnalysisResult]:
        """
        皮尔逊相关系数分析

        Args:
            parameter_samples: 参数样本数组 (n_samples, n_params)
            output_metric: 输出度量数组 (n_samples,)
            parameter_names: 参数名称列表

        Returns:
            敏感度分析结果列表
        """
        results = []
        n_params = parameter_samples.shape[1]

        for i in range(n_params):
            param = parameter_samples[:, i]
            corr, p_value = stats.pearsonr(param, output_metric)
            results.append(SensitivityAnalysisResult(
                parameter=parameter_names[i],
                metric=SensitivityMetric.PEARSON_CORRELATION.value,
                value=float(abs(corr)),
            ))

        results.sort(key=lambda x: -x.value)
        for i, r in enumerate(results, 1):
            r.rank = i

        return results

    def analyze_spearman(
        self,
        parameter_samples: np.ndarray,
        output_metric: np.ndarray,
        parameter_names: List[str],
    ) -> List[SensitivityAnalysisResult]:
        """
        斯皮尔曼秩相关分析

        Args:
            parameter_samples: 参数样本数组 (n_samples, n_params)
            output_metric: 输出度量数组 (n_samples,)
            parameter_names: 参数名称列表

        Returns:
            敏感度分析结果列表
        """
        results = []
        n_params = parameter_samples.shape[1]

        for i in range(n_params):
            param = parameter_samples[:, i]
            corr, p_value = stats.spearmanr(param, output_metric)
            results.append(SensitivityAnalysisResult(
                parameter=parameter_names[i],
                metric=SensitivityMetric.SPEARMAN_RANK.value,
                value=float(abs(corr)),
            ))

        results.sort(key=lambda x: -x.value)
        for i, r in enumerate(results, 1):
            r.rank = i

        return results

    def analyze_mutual_information(
        self,
        parameter_samples: np.ndarray,
        output_metric: np.ndarray,
        parameter_names: List[str],
        n_bins: int = 20,
    ) -> List[SensitivityAnalysisResult]:
        """
        互信息分析

        Args:
            parameter_samples: 参数样本数组 (n_samples, n_params)
            output_metric: 输出度量数组 (n_samples,)
            parameter_names: 参数名称列表
            n_bins: 分箱数

        Returns:
            敏感度分析结果列表
        """
        results = []
        n_params = parameter_samples.shape[1]

        for i in range(n_params):
            param = parameter_samples[:, i]
            mi = self._compute_mutual_information(param, output_metric, n_bins)
            results.append(SensitivityAnalysisResult(
                parameter=parameter_names[i],
                metric=SensitivityMetric.MUTUAL_INFORMATION.value,
                value=float(mi),
            ))

        results.sort(key=lambda x: -x.value)
        for i, r in enumerate(results, 1):
            r.rank = i

        return results

    def _compute_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int,
    ) -> float:
        """
        计算互信息

        Args:
            x: 第一个变量
            y: 第二个变量
            n_bins: 分箱数

        Returns:
            互信息值
        """
        c_xy = np.histogram2d(x, y, n_bins)[0]
        c_xy = c_xy / c_xy.sum()

        c_x = c_xy.sum(axis=1)
        c_y = c_xy.sum(axis=0)

        c_x = c_x[c_x > 0]
        c_y = c_y[c_y > 0]

        h_x = -np.sum(c_x * np.log2(c_x))
        h_y = -np.sum(c_y * np.log2(c_y))
        h_xy = -np.sum(c_xy[c_xy > 0] * np.log2(c_xy[c_xy > 0]))

        return h_x + h_y - h_xy


class RiskAssessor:
    """
    风险评估器

    集成失效概率估计和敏感度分析，
    提供完整的风险评估结果。
    """

    def __init__(
        self,
        criteria: FailureCriteria,
        confidence_level: float = 0.95,
        risk_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        初始化评估器

        Args:
            criteria: 失效判据
            confidence_level: 置信水平
            risk_thresholds: 风险等级阈值
        """
        self.criteria = criteria
        self.confidence_level = confidence_level
        self.failure_detector = FailureDetector(criteria)
        self.probability_estimator = FailureProbabilityEstimator(confidence_level)
        self.sensitivity_analyzer = SensitivityAnalyzer()

        self.risk_thresholds = risk_thresholds or {
            "very_low": 1e-9,
            "low": 1e-6,
            "medium": 1e-4,
            "high": 1e-2,
            "very_high": 1.0,
        }

    def determine_risk_level(self, probability: float) -> RiskLevel:
        """
        根据失效概率确定风险等级

        Args:
            probability: 失效概率

        Returns:
            风险等级
        """
        thresholds = self.risk_thresholds

        if probability < thresholds["very_low"]:
            return RiskLevel.VERY_LOW
        elif probability < thresholds["low"]:
            return RiskLevel.LOW
        elif probability < thresholds["medium"]:
            return RiskLevel.MEDIUM
        elif probability < thresholds["high"]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def compute_rpn(
        self,
        probability: float,
        severity: float = 1.0,
        detection: float = 1.0,
    ) -> float:
        """
        计算风险优先数 (RPN)

        RPN = 发生概率 × 严重程度 × 检测难度

        Args:
            probability: 失效概率
            severity: 严重程度 (1-10)
            detection: 检测难度 (1-10)

        Returns:
            RPN 值
        """
        return probability * severity * detection * 1e6

    def generate_mitigation_suggestions(
        self,
        risk_level: RiskLevel,
        sensitivity_results: List[SensitivityAnalysisResult],
    ) -> List[str]:
        """
        生成风险缓解建议

        Args:
            risk_level: 风险等级
            sensitivity_results: 敏感度分析结果

        Returns:
            缓解建议列表
        """
        suggestions = []

        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            suggestions.append("建议重新设计掩模图形，降低随机效应敏感度")
            suggestions.append("考虑增加剂量或优化工艺条件")

        top_params = [sa for sa in sensitivity_results[:3]]
        for sa in top_params:
            if "photon" in sa.parameter.lower() or "shot" in sa.parameter.lower():
                suggestions.append(f"建议增加光子通量以降低光子散粒噪声影响")
            elif "diffusion" in sa.parameter.lower() or "pag" in sa.parameter.lower():
                suggestions.append(f"建议优化光酸扩散长度或 PAG 浓度分布")
            elif "threshold" in sa.parameter.lower():
                suggestions.append(f"建议优化显影工艺以降低阈值波动")

        if not suggestions:
            suggestions.append("当前风险水平可接受，继续监控")

        return suggestions

    def assess(
        self,
        cd_values: Optional[np.ndarray],
        ler_values: Optional[np.ndarray] = None,
        lwr_values: Optional[np.ndarray] = None,
        epe_values: Optional[np.ndarray] = None,
        wafer_binary_list: Optional[List[np.ndarray]] = None,
        nominal_wafer: Optional[np.ndarray] = None,
        parameter_samples: Optional[np.ndarray] = None,
        parameter_names: Optional[List[str]] = None,
        pixel_size: float = 1.0,
        use_extrapolation: bool = True,
    ) -> RiskAssessmentResult:
        """
        执行完整的风险评估

        Args:
            cd_values: CD 值数组
            ler_values: LER 值数组
            lwr_values: LWR 值数组
            epe_values: EPE 值数组
            wafer_binary_list: 二值化晶圆图列表
            nominal_wafer: 标称晶圆图
            parameter_samples: 参数样本数组 (n_samples, n_params)
            parameter_names: 参数名称列表
            pixel_size: 像素尺寸
            use_extrapolation: 是否使用外推估计

        Returns:
            风险评估结果
        """
        result = RiskAssessmentResult()

        failures = self.failure_detector.detect_all_failures(
            cd_values=cd_values,
            ler_values=ler_values,
            lwr_values=lwr_values,
            epe_values=epe_values,
            wafer_binary_list=wafer_binary_list,
            nominal_wafer=nominal_wafer,
            pixel_size=pixel_size,
        )

        overall_failures = np.zeros(len(next(iter(failures.values()))), dtype=bool)

        for mode, indicators in failures.items():
            overall_failures |= indicators

            if use_extrapolation and cd_values is not None and mode in [
                FailureMode.CD_OVERSHOOT.value,
                FailureMode.CD_UNDERSHOOT.value,
            ]:
                lower_limit, upper_limit = self.criteria.get_cd_limits()
                if mode == FailureMode.CD_OVERSHOOT.value and upper_limit is not None:
                    fp_result = self.probability_estimator.estimate_extrapolation(
                        cd_values,
                        upper_limit,
                        mode,
                        distribution_type="normal",
                        tail="upper",
                    )
                elif mode == FailureMode.CD_UNDERSHOOT.value and lower_limit is not None:
                    fp_result = self.probability_estimator.estimate_extrapolation(
                        cd_values,
                        lower_limit,
                        mode,
                        distribution_type="normal",
                        tail="lower",
                    )
                else:
                    fp_result = self.probability_estimator.estimate_monte_carlo(
                        indicators, mode
                    )
            else:
                fp_result = self.probability_estimator.estimate_monte_carlo(
                    indicators, mode
                )

            result.failure_probabilities[mode] = fp_result
            result.risk_levels[mode] = self.determine_risk_level(fp_result.probability)
            result.risk_priority_number[mode] = self.compute_rpn(fp_result.probability)

        overall_fp = self.probability_estimator.estimate_monte_carlo(
            overall_failures, "overall"
        )
        result.overall_failure_probability = overall_fp.probability
        result.risk_level = self.determine_risk_level(overall_fp.probability)

        if parameter_samples is not None and parameter_names is not None and cd_values is not None:
            try:
                result.sensitivity_analysis = self.sensitivity_analyzer.analyze_spearman(
                    parameter_samples,
                    cd_values,
                    parameter_names,
                )
            except Exception as e:
                logger.warning(f"敏感度分析失败: {e}")

        result.risk_mitigation = self.generate_mitigation_suggestions(
            result.risk_level,
            result.sensitivity_analysis,
        )

        return result


def assess_stochastic_risk(
    cd_values: np.ndarray,
    cd_target: float,
    cd_tolerance: float,
    ler_values: Optional[np.ndarray] = None,
    lwr_values: Optional[np.ndarray] = None,
    epe_values: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
) -> RiskAssessmentResult:
    """
    便捷函数：执行随机效应风险评估

    Args:
        cd_values: CD 值数组
        cd_target: CD 目标值 (nm)
        cd_tolerance: CD 公差 (nm)
        ler_values: LER 值数组
        lwr_values: LWR 值数组
        epe_values: EPE 值数组
        confidence_level: 置信水平

    Returns:
        风险评估结果
    """
    criteria = FailureCriteria(
        cd_target=cd_target,
        cd_tolerance=cd_tolerance,
    )

    assessor = RiskAssessor(criteria, confidence_level=confidence_level)
    return assessor.assess(
        cd_values=cd_values,
        ler_values=ler_values,
        lwr_values=lwr_values,
        epe_values=epe_values,
    )
