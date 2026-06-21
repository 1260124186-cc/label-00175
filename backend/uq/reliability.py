# -*- coding: utf-8 -*-
"""
可靠性分析模块

评估光刻优化结果在真实制造环境中的可靠性：
- 失效概率估计（蒙特卡洛、FORM/SORM）
- 可靠性指标 β (Hasofer-Lind)
- 敏感度分析（一阶/总阶 Sobol 指数）
- 参数贡献度分析
- 风险等级划分与改进建议
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.stats import norm

from uq.schemas import (
    ConfidenceInterval,
    FailureProbabilityResult,
    ReliabilityResult,
)
from uq.bootstrap import BootstrapAnalyzer, BootstrapConfig

logger = logging.getLogger(__name__)


class FailureCriterion:
    """
    失效判据定义

    定义极限状态函数 g(X) > 0 表示安全，g(X) ≤ 0 表示失效。

    对于光刻成像问题，典型的失效模式包括：
    - CD 超出规格: g = CD_tolerance - |CD - CD_target|
    - EPE 超出规格: g = EPE_limit - EPE
    - NILS 不足: g = NILS - NILS_min
    """

    def __init__(
        self,
        name: str,
        limit_state_function: Callable[[Dict[str, float]], float],
        description: str = "",
    ):
        """
        初始化失效判据

        Args:
            name: 失效模式名称
            limit_state_function: 极限状态函数 g(metrics_dict) -> float
                g > 0 表示安全，g <= 0 表示失效
            description: 描述
        """
        self.name = name
        self.limit_state_function = limit_state_function
        self.description = description

    def evaluate(self, metrics: Dict[str, float]) -> float:
        """计算极限状态函数值"""
        return float(self.limit_state_function(metrics))

    def is_safe(self, metrics: Dict[str, float]) -> bool:
        """判断是否安全"""
        return self.evaluate(metrics) > 0

    def is_failed(self, metrics: Dict[str, float]) -> bool:
        """判断是否失效"""
        return self.evaluate(metrics) <= 0

    @staticmethod
    def cd_failure(
        cd_target: float,
        cd_tolerance: float = 0.1,
        tolerance_type: str = "relative",
    ) -> "FailureCriterion":
        """
        创建 CD 失效判据

        Args:
            cd_target: 目标 CD (nm)
            cd_tolerance: CD 容差
            tolerance_type: 'relative' (相对) 或 'absolute' (绝对)

        Returns:
            FailureCriterion
        """
        if tolerance_type == "relative":
            cd_lower = cd_target * (1 - cd_tolerance)
            cd_upper = cd_target * (1 + cd_tolerance)
        else:
            cd_lower = cd_target - cd_tolerance
            cd_upper = cd_target + cd_tolerance

        def g(metrics):
            cd = metrics.get("cd", metrics.get("cd_mean", cd_target))
            return min(cd - cd_lower, cd_upper - cd)

        desc = f"CD ∈ [{cd_lower:.2f}, {cd_upper:.2f}] nm"
        return FailureCriterion("cd_out_of_spec", g, desc)

    @staticmethod
    def epe_failure(epe_limit: float = 3.0) -> "FailureCriterion":
        """
        创建 EPE 失效判据

        Args:
            epe_limit: EPE 上限 (nm)

        Returns:
            FailureCriterion
        """
        def g(metrics):
            epe = metrics.get("epe", metrics.get("epe_mean", 0.0))
            return epe_limit - epe

        return FailureCriterion("epe_exceed", g, f"EPE ≤ {epe_limit:.2f} nm")

    @staticmethod
    def nils_failure(nils_min: float = 2.0) -> "FailureCriterion":
        """
        创建 NILS 失效判据

        Args:
            nils_min: NILS 下限（通常 > 2 为合格）

        Returns:
            FailureCriterion
        """
        def g(metrics):
            nils = metrics.get("nils", metrics.get("nils_mean", 0.0))
            return nils - nils_min

        return FailureCriterion("nils_insufficient", g, f"NILS ≥ {nils_min:.2f}")

    @staticmethod
    def ils_failure(ils_min: float = 0.03) -> "FailureCriterion":
        """
        创建 ILS 失效判据

        Args:
            ils_min: ILS 下限 (1/nm)

        Returns:
            FailureCriterion
        """
        def g(metrics):
            ils = metrics.get("ils", metrics.get("ils_mean", 0.0))
            return ils - ils_min

        return FailureCriterion("ils_insufficient", g, f"ILS ≥ {ils_min:.4f} 1/nm")


def compute_failure_probability(
    metric_samples: Dict[str, np.ndarray],
    failure_criteria: List[FailureCriterion],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = None,
) -> Dict[str, FailureProbabilityResult]:
    """
    基于蒙特卡洛样本估计各失效模式的失效概率

    Args:
        metric_samples: 各指标的样本值字典 {metric_name: array(n_samples,)}
        failure_criteria: 失效判据列表
        confidence_level: 置信水平
        n_bootstrap: Bootstrap 次数（用于置信区间）
        random_seed: 随机种子

    Returns:
        {failure_mode: FailureProbabilityResult} 字典
    """
    n_samples = len(next(iter(metric_samples.values())))
    rng = np.random.default_rng(random_seed)

    results = {}
    overall_indicators = np.zeros(n_samples, dtype=bool)

    for criterion in failure_criteria:
        g_values = np.zeros(n_samples)
        for i in range(n_samples):
            single_metrics = {k: v[i] for k, v in metric_samples.items()}
            g_values[i] = criterion.evaluate(single_metrics)

        failure_indicators = g_values <= 0
        n_fail = int(np.sum(failure_indicators))
        p_fail = n_fail / n_samples if n_samples > 0 else 0.0

        se = np.sqrt(p_fail * (1 - p_fail) / n_samples) if n_samples > 0 else 0.0

        try:
            boot_config = BootstrapConfig(
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                statistic=lambda x: float(np.mean(x <= 0)),
                random_seed=int(rng.integers(0, 2**31)),
            )
            boot_analyzer = BootstrapAnalyzer(boot_config)
            boot_result = boot_analyzer.nonparametric_bootstrap(g_values)

            ci_key = "bca" if "bca" in boot_result.confidence_intervals else "percentile"
            ci = boot_result.confidence_intervals.get(
                ci_key, boot_result.confidence_intervals["percentile"]
            )
        except Exception:
            z = norm.ppf(1 - (1 - confidence_level) / 2)
            ci = ConfidenceInterval(
                lower=max(0.0, p_fail - z * se),
                upper=min(1.0, p_fail + z * se),
                level=confidence_level,
                method="normal",
                point_estimate=p_fail,
                standard_error=se,
            )

        ppm = p_fail * 1e6
        log_ppm = -np.log10(max(p_fail, 1e-15)) + 6

        results[criterion.name] = FailureProbabilityResult(
            failure_mode=criterion.name,
            probability=float(p_fail),
            confidence_interval=ci,
            standard_error=float(se),
            ppm=float(ppm),
            log_ppm=float(log_ppm),
            n_failures=n_fail,
            n_samples=n_samples,
            estimate_method="monte_carlo",
        )

        overall_indicators |= failure_indicators

    return results


def compute_reliability_index(
    failure_probability: float,
    method: str = "hasofer_lind",
) -> float:
    """
    计算可靠性指标 β

    β = -Φ^{-1}(P_f)，其中 Φ 是标准正态 CDF。
    β 越大表示可靠性越高：
    - β = 1 → P_f ≈ 15.9%
    - β = 2 → P_f ≈ 2.3%
    - β = 3 → P_f ≈ 0.135%
    - β = 4 → P_f ≈ 31.7 ppm
    - β = 5 → P_f ≈ 0.287 ppm
    - β = 6 → P_f ≈ 0.001 ppm

    Args:
        failure_probability: 失效概率 P_f
        method: 方法 ('hasofer_lind')

    Returns:
        可靠性指标 β
    """
    p_f = max(min(failure_probability, 1.0 - 1e-15), 1e-15)
    return float(-norm.ppf(p_f))


def first_order_reliability(
    nominal_metrics: Dict[str, float],
    parameter_uncertainties: Dict[str, Tuple[float, float]],
    metric_functions: Dict[str, Callable[[Dict[str, float]], float]],
    failure_criterion: FailureCriterion,
) -> Dict[str, Any]:
    """
    一阶可靠性方法 (FORM) 近似

    通过在标称点处线性化极限状态函数，估计失效概率和可靠性指标。

    Args:
        nominal_metrics: 标称条件下的指标值
        parameter_uncertainties: {param_name: (nominal, std)} 参数不确定性
        metric_functions: {metric_name: function(params) -> metric_value} 指标函数
        failure_criterion: 失效判据

    Returns:
        包含 beta, pf, design_point, sensitivities 的字典
    """
    param_names = list(parameter_uncertainties.keys())
    n_params = len(param_names)

    nominal_params = {name: val[0] for name, val in parameter_uncertainties.items()}
    param_stds = {name: val[1] for name, val in parameter_uncertainties.items()}

    nominal_g = failure_criterion.evaluate(nominal_metrics)

    gradients = {}
    eps = 1e-6
    for name in param_names:
        params_plus = dict(nominal_params)
        params_plus[name] = nominal_params[name] + eps * max(abs(nominal_params[name]), 1e-6)

        metrics_plus = {}
        for metric_name, func in metric_functions.items():
            metrics_plus[metric_name] = func(params_plus)

        g_plus = failure_criterion.evaluate(metrics_plus)
        dg_dparam = (g_plus - nominal_g) / (eps * max(abs(nominal_params[name]), 1e-6))
        gradients[name] = dg_dparam * param_stds[name]

    gradient_norm_sq = sum(g ** 2 for g in gradients.values())
    gradient_norm = np.sqrt(gradient_norm_sq)

    if gradient_norm < 1e-15:
        beta = float("inf") if nominal_g > 0 else float("-inf")
    else:
        beta = nominal_g / gradient_norm

    pf = float(norm.cdf(-beta))

    sensitivities = {}
    if gradient_norm > 1e-15:
        for name in param_names:
            sensitivities[name] = float(-gradients[name] / gradient_norm)

    design_point = {}
    for name in param_names:
        alpha = sensitivities.get(name, 0.0)
        design_point[name] = nominal_params[name] + alpha * param_stds[name] * beta

    return {
        "beta": float(beta),
        "pf": float(pf),
        "nominal_g": float(nominal_g),
        "design_point": design_point,
        "sensitivities": sensitivities,
        "method": "FORM",
    }


class ReliabilityAnalyzer:
    """
    可靠性分析器

    集成失效概率估计、可靠性指标计算、敏感度分析等功能，
    提供完整的可靠性评估。
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        risk_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        初始化可靠性分析器

        Args:
            confidence_level: 置信水平
            risk_thresholds: 风险等级阈值 {level_name: max_probability}
        """
        self.confidence_level = confidence_level
        self.risk_thresholds = risk_thresholds or {
            "very_low": 1e-9,
            "low": 1e-6,
            "medium": 1e-4,
            "high": 1e-2,
            "very_high": 1.0,
        }

    def _determine_risk_level(self, probability: float) -> str:
        """根据失效概率确定风险等级"""
        if probability < self.risk_thresholds["very_low"]:
            return "very_low"
        elif probability < self.risk_thresholds["low"]:
            return "low"
        elif probability < self.risk_thresholds["medium"]:
            return "medium"
        elif probability < self.risk_thresholds["high"]:
            return "high"
        else:
            return "very_high"

    def _compute_sobol_indices(
        self,
        parameter_samples: np.ndarray,
        metric_values: np.ndarray,
        parameter_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        近似计算 Sobol 敏感度指数

        基于样本协方差的一阶敏感度估计（无需专门的实验设计）。
        """
        n_params = len(parameter_names)
        result = {}

        total_var = np.var(metric_values)
        if total_var < 1e-15:
            for name in parameter_names:
                result[name] = {"first_order": 0.0, "total_order": 0.0}
            return result

        for j, name in enumerate(parameter_names):
            x = parameter_samples[:, j]
            y = metric_values

            corr, _ = stats.pearsonr(x, y)
            first_order = float(corr ** 2)

            rank_corr, _ = stats.spearmanr(x, y)
            total_order = float(rank_corr ** 2)

            result[name] = {
                "first_order": first_order,
                "total_order": max(first_order, total_order),
            }

        return result

    def _compute_parameter_contributions(
        self,
        parameter_samples: np.ndarray,
        failure_indicators: np.ndarray,
        parameter_names: List[str],
    ) -> Dict[str, float]:
        """计算各参数对失效概率的贡献度"""
        n_params = len(parameter_names)
        contributions = {}

        total_var = np.var(failure_indicators.astype(float))
        if total_var < 1e-15:
            for name in parameter_names:
                contributions[name] = 0.0
            return contributions

        for j, name in enumerate(parameter_names):
            x = parameter_samples[:, j]
            y = failure_indicators.astype(float)

            rank_corr, _ = stats.spearmanr(x, y)
            contributions[name] = float(rank_corr ** 2)

        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions

    def _generate_recommendations(
        self,
        overall_risk: str,
        parameter_contributions: Dict[str, float],
        failure_probabilities: Dict[str, FailureProbabilityResult],
    ) -> List[str]:
        """生成可靠性改进建议"""
        suggestions = []

        if overall_risk in ("high", "very_high"):
            suggestions.append("当前风险较高，建议重新优化掩模图形以提高工艺鲁棒性")
            suggestions.append("考虑增加工艺窗口约束或使用鲁棒优化方法")

        if overall_risk == "medium":
            suggestions.append("当前风险中等，建议进行进一步验证或优化")

        top_params = sorted(parameter_contributions.items(), key=lambda x: -x[1])[:3]
        for param, contrib in top_params:
            if contrib < 0.05:
                continue
            if "focus" in param.lower() or "defocus" in param.lower():
                suggestions.append(f"离焦量 ({param}) 贡献了 {contrib*100:.1f}% 的不确定性，建议评估焦深是否足够")
            elif "dose" in param.lower():
                suggestions.append(f"剂量 ({param}) 贡献了 {contrib*100:.1f}% 的不确定性，建议优化曝光宽容度")
            elif "na" in param.lower() or "sigma" in param.lower():
                suggestions.append(f"光学参数 ({param}) 贡献了 {contrib*100:.1f}% 的不确定性，建议评估照明条件优化空间")
            elif "aberration" in param.lower() or "zernike" in param.lower():
                suggestions.append(f"像差 ({param}) 贡献了 {contrib*100:.1f}% 的不确定性，建议进行像差校准")
            elif "threshold" in param.lower():
                suggestions.append(f"光刻胶阈值 ({param}) 贡献了 {contrib*100:.1f}% 的不确定性，建议优化光刻胶工艺")

        for mode, fp in failure_probabilities.items():
            if fp.probability > 1e-4:
                if "cd" in mode.lower():
                    suggestions.append(f"CD 失效概率较高 ({fp.probability:.2e})，建议检查 CD 均匀性和工艺窗口")
                elif "epe" in mode.lower():
                    suggestions.append(f"EPE 失效概率较高 ({fp.probability:.2e})，建议优化边缘放置精度")
                elif "nils" in mode.lower() or "ils" in mode.lower():
                    suggestions.append(f"NILS/ILS 失效概率较高 ({fp.probability:.2e})，建议优化对比度或使用 OPC 辅助特征")

        if not suggestions:
            suggestions.append("当前可靠性水平可接受，建议定期监控工艺参数波动")

        return suggestions

    def analyze(
        self,
        metric_samples: Dict[str, np.ndarray],
        failure_criteria: List[FailureCriterion],
        parameter_samples: Optional[np.ndarray] = None,
        parameter_names: Optional[List[str]] = None,
        n_bootstrap: int = 1000,
        random_seed: Optional[int] = None,
    ) -> ReliabilityResult:
        """
        执行完整的可靠性分析

        Args:
            metric_samples: {metric_name: array(n_samples,)} 各指标的样本值
            failure_criteria: 失效判据列表
            parameter_samples: (n_samples, n_params) 参数样本数组（用于敏感度分析）
            parameter_names: 参数名称列表
            n_bootstrap: Bootstrap 次数
            random_seed: 随机种子

        Returns:
            ReliabilityResult
        """
        failure_probabilities = compute_failure_probability(
            metric_samples=metric_samples,
            failure_criteria=failure_criteria,
            confidence_level=self.confidence_level,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

        n_samples = len(next(iter(metric_samples.values())))
        overall_failures = np.zeros(n_samples, dtype=bool)
        for criterion in failure_criteria:
            g_values = np.zeros(n_samples)
            for i in range(n_samples):
                single_metrics = {k: v[i] for k, v in metric_samples.items()}
                g_values[i] = criterion.evaluate(single_metrics)
            overall_failures |= (g_values <= 0)

        overall_pf = float(np.mean(overall_failures))
        overall_beta = compute_reliability_index(overall_pf)
        risk_level = self._determine_risk_level(overall_pf)

        sensitivity_indices = {}
        parameter_contributions = {}

        if parameter_samples is not None and parameter_names is not None:
            for metric_name, values in metric_samples.items():
                sensitivity_indices[metric_name] = self._compute_sobol_indices(
                    parameter_samples, values, parameter_names
                )

            parameter_contributions = self._compute_parameter_contributions(
                parameter_samples, overall_failures, parameter_names
            )

        recommendations = self._generate_recommendations(
            risk_level, parameter_contributions, failure_probabilities
        )

        for fp in failure_probabilities.values():
            fp.reliability_index = compute_reliability_index(fp.probability)

        return ReliabilityResult(
            overall_failure_probability=overall_pf,
            failure_probabilities=failure_probabilities,
            reliability_index=overall_beta,
            sensitivity_indices=sensitivity_indices,
            parameter_contributions=parameter_contributions,
            risk_level=risk_level,
            recommendations=recommendations,
        )
