# -*- coding: utf-8 -*-
"""
UQ 主分析器模块

集成工艺参数扰动仿真、模型参数不确定性传播与统计推断，
输出优化掩模的完整不确定性量化结果，回答：
"该优化结果在真实 fab 中有多可靠？"
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import time

from uq.schemas import (
    UQConfig,
    UQMethod,
    MetricUncertainty,
    ReliabilityResult,
    FailureProbabilityResult,
    UncertaintyType,
    ProcessUncertaintyConfig,
    ModelUncertaintyConfig,
)
from uq.parameter_uncertainty import (
    ProcessPerturbationSampler,
    ModelUncertaintySampler,
)
from uq.bootstrap import BootstrapAnalyzer, BootstrapConfig
from uq.reliability import (
    ReliabilityAnalyzer,
    FailureCriterion,
)

logger = logging.getLogger(__name__)


@dataclass
class SingleRunResult:
    """
    单次工艺条件下的仿真结果

    Attributes:
        run_id: 运行 ID
        process_condition: 工艺条件（字典形式）
        model_perturbation: 模型参数扰动
        metrics: 成像指标字典
        aerial_image: 空间像（可选）
        wafer_image: 晶圆图（可选）
    """
    run_id: int
    process_condition: Dict[str, float]
    model_perturbation: Dict[str, float]
    metrics: Dict[str, float]
    aerial_image: Optional[np.ndarray] = None
    wafer_image: Optional[np.ndarray] = None


class UQAnalyzer:
    """
    不确定性量化主分析器

    工作流程：
    1. 根据工艺不确定性配置采样多组工艺条件
    2. 根据模型不确定性配置采样模型参数扰动
    3. 对每个采样点执行光刻仿真，获取成像指标
    4. 对指标集合执行统计推断（Bootstrap 或贝叶斯）
    5. 估计失效概率与可靠性指标
    """

    def __init__(
        self,
        config: Optional[UQConfig] = None,
        optical_system: Any = None,
        imaging_simulator: Optional[Callable] = None,
        metric_computers: Optional[Dict[str, Callable]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        初始化 UQ 分析器

        Args:
            config: UQ 配置
            optical_system: 光学系统实例（core.imaging.OpticalSystem）
            imaging_simulator: 仿真函数，签名为
                simulator(mask, process_condition, model_params) ->
                    {'aerial_image', 'wafer_image', ...}
            metric_computers: {metric_name: callable(aerial, wafer, **kwargs)}
                指标计算函数字典
            random_seed: 随机种子
        """
        self.config = config if config is not None else UQConfig()
        self.optical_system = optical_system
        self.imaging_simulator = imaging_simulator
        self.metric_computers = metric_computers or {}
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self._process_sampler = ProcessPerturbationSampler(
            self.config.process_uncertainty,
            self.optical_system,
            random_seed=random_seed,
        )
        self._model_sampler = ModelUncertaintySampler(
            self.config.model_uncertainty,
            random_seed=random_seed,
        )
        self._reliability_analyzer = ReliabilityAnalyzer(
            confidence_level=self.config.confidence_level,
        )

    def run_simulations(
        self,
        mask: np.ndarray,
        target: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
        save_images: bool = False,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> List[SingleRunResult]:
        """
        批量执行工艺参数扰动下的光刻仿真

        Args:
            mask: 掩模图形 (2D 数组)
            target: 目标晶圆图（用于 EPE 计算）
            n_samples: 样本数，None 使用配置值
            save_images: 是否保存中间图像
            progress_callback: 进度回调 callback(current, total, elapsed)

        Returns:
            单次仿真结果列表
        """
        if n_samples is None:
            n_samples = self.config.n_samples

        t_start = time.time()

        process_conditions = self._process_sampler.sample(n_samples)
        model_perturbations = self._model_sampler.sample(n_samples)

        results: List[SingleRunResult] = []

        for i in range(n_samples):
            pc = process_conditions[i]
            mp = model_perturbations[i]

            sim_output = self._call_simulator(mask, pc, mp)

            metrics = self._compute_metrics(
                sim_output=sim_output,
                pc=pc,
                mp=mp,
                target=target,
            )

            run = SingleRunResult(
                run_id=i,
                process_condition=self._pc_to_dict(pc),
                model_perturbation=mp,
                metrics=metrics,
                aerial_image=sim_output.get("aerial_image") if save_images else None,
                wafer_image=sim_output.get("wafer_image") if save_images else None,
            )
            results.append(run)

            if progress_callback is not None:
                elapsed = time.time() - t_start
                progress_callback(i + 1, n_samples, elapsed)

        return results

    def _call_simulator(
        self,
        mask: np.ndarray,
        process_condition: Any,
        model_perturbation: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        """调用仿真器，兼容不同签名"""
        if self.imaging_simulator is None:
            return self._default_simulator(mask, process_condition, model_perturbation)

        try:
            return self.imaging_simulator(
                mask=mask,
                process_condition=process_condition,
                model_perturbation=model_perturbation,
            )
        except TypeError:
            try:
                return self.imaging_simulator(mask, process_condition)
            except Exception as e:
                logger.warning(f"仿真器调用失败，使用默认仿真器: {e}")
                return self._default_simulator(mask, process_condition, model_perturbation)

    def _default_simulator(
        self,
        mask: np.ndarray,
        process_condition: Any,
        model_perturbation: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        """
        默认简化仿真器（无真实成像模型时使用）

        仅做基于参数扰动的近似指标估计，用于功能演示和无成像器时的 fallback。
        """
        try:
            from core.imaging import PartialCoherentImaging

            optics = process_condition.to_optical_system(self.optical_system)
            imager = PartialCoherentImaging(optics)
            aerial = imager.simulate(mask)

            threshold = 0.5
            if "threshold_factor" in model_perturbation:
                threshold *= model_perturbation["threshold_factor"]

            wafer = (aerial >= threshold).astype(np.float64)
            return {"aerial_image": aerial, "wafer_image": wafer}

        except Exception as e:
            logger.info(f"使用极简仿真 fallback: {e}")
            blurred = self._gaussian_blur(mask.astype(np.float64), sigma=1.5)
            dose_factor = getattr(process_condition, "dose", 1.0)
            aerial = np.clip(blurred * dose_factor, 0, 1)
            threshold = 0.5
            if "threshold_factor" in model_perturbation:
                threshold *= model_perturbation["threshold_factor"]
            wafer = (aerial >= threshold).astype(np.float64)
            return {"aerial_image": aerial, "wafer_image": wafer}

    @staticmethod
    def _gaussian_blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """简化高斯模糊（避免 scipy.ndimage 依赖）"""
        if sigma <= 0:
            return img
        radius = int(np.ceil(sigma * 3))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()

        result = np.zeros_like(img)
        for i in range(img.shape[0]):
            result[i] = np.convolve(img[i], kernel, mode="same")
        for j in range(img.shape[1]):
            result[:, j] = np.convolve(result[:, j], kernel, mode="same")
        return result

    def _compute_metrics(
        self,
        sim_output: Dict[str, np.ndarray],
        pc: Any,
        mp: Dict[str, float],
        target: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """计算成像指标"""
        aerial = sim_output.get("aerial_image")
        wafer = sim_output.get("wafer_image")

        metrics: Dict[str, float] = {}

        if self.metric_computers:
            for name, func in self.metric_computers.items():
                try:
                    metrics[name] = float(
                        func(aerial=aerial, wafer=wafer, target=target)
                    )
                except Exception as e:
                    logger.debug(f"指标 {name} 计算失败: {e}")
        else:
            metrics = self._default_metrics(aerial, wafer, target)

        metrics["defocus"] = float(getattr(pc, "defocus", 0.0))
        metrics["dose"] = float(getattr(pc, "dose", 1.0))
        metrics["na"] = float(getattr(pc, "na", 1.0))
        metrics["sigma"] = float(getattr(pc, "sigma", 0.75))
        for k, v in mp.items():
            metrics[k] = float(v)

        return metrics

    @staticmethod
    def _default_metrics(
        aerial: Optional[np.ndarray],
        wafer: Optional[np.ndarray],
        target: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """默认指标计算（无自定义 metric_computers 时）"""
        metrics = {}

        if wafer is not None:
            metrics["wafer_mean"] = float(np.mean(wafer))
            metrics["wafer_std"] = float(np.std(wafer))

            if target is not None:
                diff = np.abs(wafer - target)
                metrics["epe_mean"] = float(np.mean(diff))
                metrics["epe_max"] = float(np.max(diff))
                metrics["mse"] = float(np.mean((wafer - target) ** 2))

                intersection = np.sum(np.minimum(wafer, target))
                union = np.sum(np.maximum(wafer, target))
                metrics["iou"] = float(intersection / max(union, 1e-10))

            if aerial is not None:
                mid = wafer.shape[0] // 2
                line = aerial[mid, :]
                edge_positions = np.where(np.diff((line >= 0.5).astype(int)) != 0)[0]
                if len(edge_positions) >= 2:
                    cd = edge_positions[-1] - edge_positions[0]
                    metrics["cd"] = float(cd)

                    slopes = np.abs(np.gradient(line))
                    if len(edge_positions) >= 2:
                        e1, e2 = edge_positions[0], edge_positions[1]
                        ils = slopes[e1:e2 + 1]
                        if len(ils) > 0:
                            metrics["ils"] = float(np.max(ils))

                    if cd > 0:
                        nils = (np.max(np.abs(np.gradient(line))) * cd) / max(np.max(line), 1e-10)
                        metrics["nils"] = float(nils)

        return metrics

    @staticmethod
    def _pc_to_dict(pc: Any) -> Dict[str, float]:
        """工艺条件转字典"""
        if hasattr(pc, "to_dict"):
            d = pc.to_dict()
            return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}
        else:
            result = {}
            for attr in ["defocus", "dose", "na", "sigma", "wavelength"]:
                if hasattr(pc, attr):
                    result[attr] = float(getattr(pc, attr))
            return result

    def analyze_uncertainty(
        self,
        runs: List[SingleRunResult],
        metric_names: Optional[List[str]] = None,
        method: Optional[UQMethod] = None,
    ) -> Dict[str, MetricUncertainty]:
        """
        对仿真结果执行统计推断，计算置信区间

        Args:
            runs: 仿真结果列表
            metric_names: 要分析的指标名，None 则分析所有数值指标
            method: 统计推断方法

        Returns:
            {metric_name: MetricUncertainty}
        """
        if method is None:
            method = self.config.method

        if metric_names is None:
            metric_names = self._collect_numeric_metric_names(runs)

        metric_samples = self._collect_metric_samples(runs, metric_names)

        results: Dict[str, MetricUncertainty] = {}

        if method == UQMethod.BOOTSTRAP:
            analyzer = BootstrapAnalyzer(
                BootstrapConfig(
                    n_bootstrap=self.config.n_bootstrap,
                    confidence_level=self.config.confidence_level,
                    ci_method="bca",
                    random_seed=int(self.rng.integers(0, 2**31)),
                )
            )
            for name in metric_names:
                samples = metric_samples[name]
                try:
                    boot_result = analyzer.nonparametric_bootstrap(samples)
                    ci = boot_result.confidence_intervals.get(
                        "bca",
                        boot_result.confidence_intervals.get("percentile"),
                    )
                    mu = MetricUncertainty(
                        metric_name=name,
                        samples=samples,
                        nominal_value=float(np.mean(samples)),
                        standard_error=float(boot_result.standard_error),
                        confidence_interval=ci,
                        bias=float(boot_result.bias),
                        bias_corrected=float(boot_result.bias_corrected_estimate),
                        uncertainty_type=UncertaintyType.ALEATORY,
                    )
                    results[name] = mu
                except Exception as e:
                    logger.warning(f"Bootstrap 分析指标 {name} 失败: {e}")
                    results[name] = self._fallback_metric_uncertainty(name, samples)

        elif method == UQMethod.MONTE_CARLO:
            for name in metric_names:
                results[name] = self._fallback_metric_uncertainty(name, metric_samples[name])

        else:
            logger.info(f"方法 {method.value} 目前使用 Monte Carlo 近似")
            for name in metric_names:
                results[name] = self._fallback_metric_uncertainty(name, metric_samples[name])

        return results

    def _fallback_metric_uncertainty(
        self, metric_name: str, samples: np.ndarray
    ) -> MetricUncertainty:
        """近似正态置信区间 fallback"""
        from uq.schemas import ConfidenceInterval
        from scipy.stats import norm

        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))
        se = std / np.sqrt(len(samples)) if len(samples) > 1 else std

        alpha = 1 - self.config.confidence_level
        z = norm.ppf(1 - alpha / 2)

        ci = ConfidenceInterval(
            lower=mean - z * se,
            upper=mean + z * se,
            level=self.config.confidence_level,
            method="normal",
            point_estimate=mean,
            standard_error=float(se),
        )

        return MetricUncertainty(
            metric_name=metric_name,
            samples=samples,
            nominal_value=mean,
            standard_error=float(se),
            confidence_interval=ci,
            uncertainty_type=UncertaintyType.MIXED,
        )

    @staticmethod
    def _collect_numeric_metric_names(runs: List[SingleRunResult]) -> List[str]:
        """收集所有数值指标名"""
        if not runs:
            return []
        names = []
        for k, v in runs[0].metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                names.append(k)
        return names

    @staticmethod
    def _collect_metric_samples(
        runs: List[SingleRunResult], metric_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """收集各指标的样本数组"""
        samples: Dict[str, np.ndarray] = {}
        for name in metric_names:
            vals = []
            for r in runs:
                v = r.metrics.get(name)
                if v is not None and np.isfinite(float(v)):
                    vals.append(float(v))
            samples[name] = np.array(vals, dtype=np.float64)
        return samples

    def analyze_reliability(
        self,
        runs: List[SingleRunResult],
        failure_criteria: List[FailureCriterion],
        metric_uncertainties: Optional[Dict[str, MetricUncertainty]] = None,
    ) -> ReliabilityResult:
        """
        执行可靠性分析

        Args:
            runs: 仿真结果列表
            failure_criteria: 失效判据列表
            metric_uncertainties: 指标不确定性（可选）

        Returns:
            ReliabilityResult
        """
        metric_samples = self._collect_metric_samples(
            runs, self._collect_numeric_metric_names(runs)
        )

        param_samples_list = []
        param_names = ["defocus", "dose", "na", "sigma"]
        for r in runs:
            row = [
                r.process_condition.get("defocus", 0.0),
                r.process_condition.get("dose", 1.0),
                r.process_condition.get("na", 1.0),
                r.process_condition.get("sigma", 0.75),
            ]
            param_samples_list.append(row)
        parameter_samples = np.array(param_samples_list, dtype=np.float64)

        return self._reliability_analyzer.analyze(
            metric_samples=metric_samples,
            failure_criteria=failure_criteria,
            parameter_samples=parameter_samples,
            parameter_names=param_names,
            n_bootstrap=self.config.n_bootstrap,
            random_seed=int(self.rng.integers(0, 2**31)),
        )

    def full_analysis(
        self,
        mask: np.ndarray,
        target: Optional[np.ndarray] = None,
        failure_criteria: Optional[List[FailureCriterion]] = None,
        n_samples: Optional[int] = None,
        metric_names: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> "UQResult":
        """
        执行完整 UQ 分析

        Args:
            mask: 优化后的掩模图形
            target: 目标晶圆图
            failure_criteria: 失效判据列表，None 则使用默认 CD/EPE/NILS 判据
            n_samples: 样本数
            metric_names: 需分析的指标名
            progress_callback: 进度回调 callback(current, total, elapsed)

        Returns:
            UQResult
        """
        from uq.schemas import UQResult

        t_start = time.time()

        runs = self.run_simulations(
            mask=mask,
            target=target,
            n_samples=n_samples,
            save_images=False,
            progress_callback=progress_callback,
        )

        metric_uncertainties = self.analyze_uncertainty(
            runs=runs,
            metric_names=metric_names,
        )

        if failure_criteria is None:
            failure_criteria = self._default_failure_criteria(metric_uncertainties)

        reliability = self.analyze_reliability(
            runs=runs,
            failure_criteria=failure_criteria,
            metric_uncertainties=metric_uncertainties,
        )

        total_time = time.time() - t_start

        return UQResult(
            method=self.config.method,
            n_samples=len(runs),
            confidence_level=self.config.confidence_level,
            metric_uncertainties=metric_uncertainties,
            reliability=reliability,
            total_time=total_time,
        )

    @staticmethod
    def _default_failure_criteria(
        metric_uncertainties: Dict[str, MetricUncertainty],
    ) -> List[FailureCriterion]:
        """根据指标值自动生成默认失效判据"""
        criteria = []

        if "epe_mean" in metric_uncertainties:
            mean_epe = metric_uncertainties["epe_mean"].nominal_value
            epe_limit = max(mean_epe * 2.0, 3.0)
            criteria.append(FailureCriterion.epe_failure(epe_limit=float(epe_limit)))

        if "cd" in metric_uncertainties:
            mean_cd = metric_uncertainties["cd"].nominal_value
            if mean_cd > 0:
                criteria.append(
                    FailureCriterion.cd_failure(
                        cd_target=float(mean_cd), cd_tolerance=0.10
                    )
                )

        if "nils" in metric_uncertainties:
            mean_nils = metric_uncertainties["nils"].nominal_value
            nils_min = min(max(mean_nils * 0.7, 2.0), mean_nils)
            criteria.append(FailureCriterion.nils_failure(nils_min=float(nils_min)))

        if not criteria:
            def g(metrics):
                iou = metrics.get("iou", 1.0)
                return iou - 0.9
            criteria.append(FailureCriterion("iou_below_90", g, "IoU ≥ 0.9"))

        return criteria


def run_uq_analysis(
    mask: np.ndarray,
    target: Optional[np.ndarray] = None,
    optical_system: Any = None,
    imaging_simulator: Optional[Callable] = None,
    config: Optional[UQConfig] = None,
    **kwargs,
):
    """
    便捷函数：运行完整 UQ 分析

    Args:
        mask: 优化后的掩模
        target: 目标晶圆图
        optical_system: 光学系统
        imaging_simulator: 仿真函数
        config: UQ 配置
        **kwargs: 其他参数

    Returns:
        UQResult
    """
    analyzer = UQAnalyzer(
        config=config,
        optical_system=optical_system,
        imaging_simulator=imaging_simulator,
    )
    return analyzer.full_analysis(mask=mask, target=target, **kwargs)
