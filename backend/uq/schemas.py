# -*- coding: utf-8 -*-
"""
UQ 数据结构模块

定义不确定性量化所需的所有数据结构：
- 参数不确定性配置（工艺参数、模型参数）
- 置信区间
- 指标不确定性结果
- 失效概率结果
- 可靠性分析结果
- UQ 整体结果
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field, InitVar
from enum import Enum


class UncertaintyType(Enum):
    """不确定性类型枚举"""
    ALEATORY = "aleatory"
    EPISTEMIC = "epistemic"
    MIXED = "mixed"


class UQMethod(Enum):
    """UQ 分析方法枚举"""
    BOOTSTRAP = "bootstrap"
    BAYESIAN_MCMC = "bayesian_mcmc"
    BAYESIAN_ABC = "bayesian_abc"
    MONTE_CARLO = "monte_carlo"
    FIRST_ORDER = "first_order"
    PROBABILISTIC_COLLOCATION = "probabilistic_collocation"


@dataclass
class ParameterDistribution:
    """
    参数概率分布描述

    Attributes:
        name: 参数名称
        distribution_type: 分布类型 ('normal', 'uniform', 'triangular', 'lognormal', 'gamma', 'beta', 'custom')
        nominal: 标称值（分布中心或众数）
        params: 分布参数字典
            - normal: {'std': float}
            - uniform: {'low': float, 'high': float}
            - triangular: {'low': float, 'high': float, 'mode': float}
            - lognormal: {'sigma': float, 'mu': float}
            - gamma: {'shape': float, 'scale': float}
            - beta: {'alpha': float, 'beta': float}
            - custom: {'samples': np.ndarray} 或 {'pdf': Callable}
        uncertainty_type: 不确定性类型（偶然/认知/混合）
        bounds: 可选的物理边界 [low, high]，用于截断采样
        description: 可选的参数描述
    """
    name: str
    distribution_type: str = "normal"
    nominal: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    uncertainty_type: UncertaintyType = UncertaintyType.ALEATORY
    bounds: Optional[Tuple[float, float]] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "distribution_type": self.distribution_type,
            "nominal": self.nominal,
            "params": self.params,
            "uncertainty_type": self.uncertainty_type.value,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "description": self.description,
        }

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        从分布中采样

        Args:
            n: 采样数量
            rng: 随机数生成器，None 则使用默认

        Returns:
            样本数组 (n,)
        """
        if rng is None:
            rng = np.random.default_rng()
        dtype = self.distribution_type.lower()

        if dtype == "normal":
            std = self.params.get("std", 1.0)
            samples = rng.normal(loc=self.nominal, scale=std, size=n)
        elif dtype == "uniform":
            low = self.params.get("low", self.nominal - 1.0)
            high = self.params.get("high", self.nominal + 1.0)
            samples = rng.uniform(low=low, high=high, size=n)
        elif dtype == "triangular":
            low = self.params.get("low", self.nominal - 1.0)
            high = self.params.get("high", self.nominal + 1.0)
            mode = self.params.get("mode", self.nominal)
            samples = rng.triangular(left=low, mode=mode, right=high, size=n)
        elif dtype == "lognormal":
            sigma = self.params.get("sigma", 1.0)
            mu = self.params.get("mu", np.log(max(self.nominal, 1e-10)))
            samples = rng.lognormal(mean=mu, sigma=sigma, size=n)
        elif dtype == "gamma":
            shape = self.params.get("shape", 1.0)
            scale = self.params.get("scale", 1.0)
            samples = rng.gamma(shape=shape, scale=scale, size=n)
        elif dtype == "beta":
            alpha = self.params.get("alpha", 1.0)
            beta = self.params.get("beta", 1.0)
            low = self.params.get("low", 0.0)
            high = self.params.get("high", 1.0)
            samples = rng.beta(a=alpha, b=beta, size=n) * (high - low) + low
        else:
            std = self.params.get("std", 1.0)
            samples = rng.normal(loc=self.nominal, scale=std, size=n)

        if self.bounds is not None:
            samples = np.clip(samples, self.bounds[0], self.bounds[1])
        return samples

    def pdf(self, x: float) -> float:
        """计算概率密度函数值"""
        from scipy import stats as _stats

        dtype = self.distribution_type.lower()
        try:
            if dtype == "normal":
                std = self.params.get("std", 1.0)
                return float(_stats.norm.pdf(x, self.nominal, std))
            elif dtype == "uniform":
                low = self.params.get("low", self.nominal - 1.0)
                high = self.params.get("high", self.nominal + 1.0)
                return float(_stats.uniform.pdf(x, low, high - low))
            elif dtype == "lognormal":
                sigma = self.params.get("sigma", 1.0)
                mu = self.params.get("mu", np.log(max(self.nominal, 1e-10)))
                return float(_stats.lognorm.pdf(x, sigma, scale=np.exp(mu)))
            else:
                return 1.0
        except Exception:
            return 1.0

    @classmethod
    def normal(
        cls,
        name: str,
        nominal: float,
        std: float,
        description: str = "",
        uncertainty_type: UncertaintyType = UncertaintyType.ALEATORY,
    ) -> "ParameterDistribution":
        """创建正态分布参数"""
        return cls(
            name=name,
            distribution_type="normal",
            nominal=nominal,
            params={"std": float(std)},
            uncertainty_type=uncertainty_type,
            description=description,
        )

    @classmethod
    def uniform(
        cls,
        name: str,
        low: float,
        high: float,
        description: str = "",
        uncertainty_type: UncertaintyType = UncertaintyType.ALEATORY,
    ) -> "ParameterDistribution":
        """创建均匀分布参数"""
        nominal = 0.5 * (low + high)
        return cls(
            name=name,
            distribution_type="uniform",
            nominal=nominal,
            params={"low": float(low), "high": float(high)},
            uncertainty_type=uncertainty_type,
            description=description,
            bounds=(float(low), float(high)),
        )

    @classmethod
    def lognormal(
        cls,
        name: str,
        mu: float,
        sigma: float,
        description: str = "",
        uncertainty_type: UncertaintyType = UncertaintyType.ALEATORY,
    ) -> "ParameterDistribution":
        """创建对数正态分布参数"""
        nominal = float(np.exp(mu + 0.5 * sigma ** 2))
        return cls(
            name=name,
            distribution_type="lognormal",
            nominal=nominal,
            params={"mu": float(mu), "sigma": float(sigma)},
            uncertainty_type=uncertainty_type,
            description=description,
        )


@dataclass
class ProcessUncertaintyConfig:
    """
    工艺参数不确定性配置

    描述光刻工艺中可控参数的波动：
    - 离焦量 (defocus)
    - 曝光剂量 (dose)
    - 数值孔径 (NA)
    - 部分相干因子 (sigma)
    - 波长 (wavelength)
    - Flare 系数
    - Zernike 像差系数

    Attributes:
        focus_std: 离焦量标准差 (nm)，默认 30nm (典型 fab 工艺波动)
        dose_std: 曝光剂量相对标准差（相对于标称剂量 1.0），默认 3%
        na_std: 数值孔径相对标准差，默认 0.5%
        sigma_std: 部分相干因子相对标准差，默认 1%
        wavelength_std: 波长标准差 (nm)，默认 0.1nm
        flare_std: Flare 系数标准差，默认 0.5%
        aberration_std: Zernike 像差系数标准差（单位: 波长λ）
            - float: 所有像差使用相同标准差
            - Dict[int, float]: 按 Zernike 索引指定
        zernike_indices: 要考虑的 Zernike 系数索引列表（Noll 索引, 0-based）
        distribution: 分布类型 ('normal', 'uniform', 'triangular')
        random_seed: 随机种子
        custom_parameters: 自定义工艺参数分布列表
    """
    focus_std: float = 30.0
    dose_std: float = 0.03
    na_std: float = 0.005
    sigma_std: float = 0.01
    wavelength_std: float = 0.1
    flare_std: float = 0.005
    aberration_std: Optional[Union[float, Dict[int, float]]] = None
    zernike_indices: Optional[List[int]] = None
    distribution: str = "normal"
    random_seed: Optional[int] = None
    custom_parameters: List[ParameterDistribution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focus_std": self.focus_std,
            "dose_std": self.dose_std,
            "na_std": self.na_std,
            "sigma_std": self.sigma_std,
            "wavelength_std": self.wavelength_std,
            "flare_std": self.flare_std,
            "aberration_std": (
                self.aberration_std
                if isinstance(self.aberration_std, (int, float)) or self.aberration_std is None
                else {str(k): v for k, v in self.aberration_std.items()}
            ),
            "zernike_indices": self.zernike_indices,
            "distribution": self.distribution,
            "random_seed": self.random_seed,
            "custom_parameters": [p.to_dict() for p in self.custom_parameters],
        }


@dataclass
class ModelUncertaintyConfig:
    """
    模型参数不确定性配置

    描述光刻仿真模型中的参数不确定性：
    - 光刻胶阈值
    - 光酸扩散长度
    - 显影速率参数
    - MEEF 模型参数
    - 像差校准误差

    Attributes:
        threshold_std: 光刻胶阈值相对标准差，默认 5%
        diffusion_length_std: 光酸扩散长度相对标准差，默认 10%
        resist_model_params_std: 光刻胶模型其他参数的相对标准差
        meef_std: MEEF 估计的相对标准差，默认 10%
        aberration_calibration_std: 像差校准误差标准差（单位: 波长λ）
        model_form_uncertainty: 模型形式不确定性（0-1），默认 0.05
        distribution: 分布类型
        random_seed: 随机种子
        custom_parameters: 自定义模型参数分布列表
    """
    threshold_std: float = 0.05
    diffusion_length_std: float = 0.10
    resist_model_params_std: float = 0.05
    meef_std: float = 0.10
    aberration_calibration_std: float = 0.005
    model_form_uncertainty: float = 0.05
    distribution: str = "normal"
    random_seed: Optional[int] = None
    custom_parameters: List[ParameterDistribution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold_std": self.threshold_std,
            "diffusion_length_std": self.diffusion_length_std,
            "resist_model_params_std": self.resist_model_params_std,
            "meef_std": self.meef_std,
            "aberration_calibration_std": self.aberration_calibration_std,
            "model_form_uncertainty": self.model_form_uncertainty,
            "distribution": self.distribution,
            "random_seed": self.random_seed,
            "custom_parameters": [p.to_dict() for p in self.custom_parameters],
        }


@dataclass
class UQConfig:
    """
    UQ 分析整体配置

    Attributes:
        n_samples: 采样/仿真次数
        confidence_level: 置信水平（默认 0.95 = 95%）
        process_uncertainty: 工艺参数不确定性配置
        model_uncertainty: 模型参数不确定性配置
        uq_method: 使用的 UQ 方法
        metrics_to_analyze: 要分析的指标列表，None 则分析所有指标
            可选: ['cd', 'cd_error', 'epe', 'mse', 'ssim', 'ils', 'nils', 'pw_area']
        enable_sensitivity_analysis: 是否启用敏感度分析
        enable_reliability_analysis: 是否启用可靠性分析
        progress_callback: 进度回调 callback(current, total, elapsed)
        random_seed: 全局随机种子
    """
    n_samples: int = 200
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    process_uncertainty: ProcessUncertaintyConfig = field(
        default_factory=ProcessUncertaintyConfig
    )
    model_uncertainty: ModelUncertaintyConfig = field(
        default_factory=ModelUncertaintyConfig
    )
    uq_method: UQMethod = UQMethod.MONTE_CARLO
    metrics_to_analyze: Optional[List[str]] = None
    enable_sensitivity_analysis: bool = True
    enable_reliability_analysis: bool = True
    progress_callback: Optional[Callable[[int, int, float], None]] = None
    random_seed: Optional[int] = None

    @property
    def method(self) -> UQMethod:
        return self.uq_method

    @method.setter
    def method(self, value: UQMethod):
        self.uq_method = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "confidence_level": self.confidence_level,
            "process_uncertainty": self.process_uncertainty.to_dict(),
            "model_uncertainty": self.model_uncertainty.to_dict(),
            "uq_method": self.uq_method.value,
            "metrics_to_analyze": self.metrics_to_analyze,
            "enable_sensitivity_analysis": self.enable_sensitivity_analysis,
            "enable_reliability_analysis": self.enable_reliability_analysis,
            "random_seed": self.random_seed,
        }


@dataclass
class ConfidenceInterval:
    """
    置信区间结果

    Attributes:
        lower: 置信区间下界
        upper: 置信区间上界
        level: 置信水平（如 0.95）
        method: 计算方法 ('percentile', 'normal', 'bca', 'hpd')
        point_estimate: 点估计值（均值/中位数）
        standard_error: 标准误差
        width: 置信区间宽度
    """
    lower: float
    upper: float
    level: float = 0.95
    method: str = "percentile"
    point_estimate: Optional[float] = None
    standard_error: Optional[float] = None

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def relative_width(self) -> float:
        if self.point_estimate is None or abs(self.point_estimate) < 1e-15:
            return float("inf")
        return self.width / abs(self.point_estimate)

    def contains(self, value: float) -> bool:
        """判断值是否在置信区间内"""
        return self.lower <= float(value) <= self.upper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "level": float(self.level),
            "method": self.method,
            "point_estimate": float(self.point_estimate) if self.point_estimate is not None else None,
            "standard_error": float(self.standard_error) if self.standard_error is not None else None,
            "width": float(self.width),
        }

    def summary(self) -> str:
        return (
            f"{int(self.level * 100):.0f}% CI [{self.lower:.4f}, {self.upper:.4f}] "
            f"(width={self.width:.4f}, {self.method})"
        )


@dataclass
class MetricUncertainty:
    """
    单个成像指标的不确定性分析结果

    Attributes:
        metric_name: 指标名称
        nominal_value: 标称值（无不确定性时的计算结果）
        samples: 所有采样/仿真得到的指标值数组
        mean: 均值
        std: 标准差
        variance: 方差
        median: 中位数
        percentile_5: 5% 分位数（悲观估计）
        percentile_95: 95% 分位数（乐观估计）
        cv: 变异系数 (std/mean)
        confidence_intervals: 不同方法/水平的置信区间字典
        distribution_type: 拟合的分布类型
        skewness: 偏度
        kurtosis: 峰度
    """
    metric_name: str
    nominal_value: float
    samples: Optional[np.ndarray] = None
    mean: float = 0.0
    std: float = 0.0
    variance: float = 0.0
    median: float = 0.0
    percentile_5: float = 0.0
    percentile_95: float = 0.0
    cv: float = 0.0
    confidence_intervals: Dict[str, ConfidenceInterval] = field(default_factory=dict)
    distribution_type: str = "unknown"
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    uncertainty_type: UncertaintyType = UncertaintyType.MIXED
    bias: Optional[float] = None
    bias_corrected: Optional[float] = None
    bias_corrected_estimate: Optional[float] = None
    standard_error: Optional[float] = None
    confidence_interval: InitVar[Optional[ConfidenceInterval]] = None

    def __post_init__(self, confidence_interval: Optional[ConfidenceInterval] = None):
        if confidence_interval is not None:
            self.confidence_intervals[confidence_interval.method] = confidence_interval
        if self.bias_corrected_estimate is None and self.bias_corrected is not None:
            self.bias_corrected_estimate = self.bias_corrected
        if self.bias_corrected is None and self.bias_corrected_estimate is not None:
            self.bias_corrected = self.bias_corrected_estimate
        if self.samples is not None and len(self.samples) > 0 and self.mean == 0 and self.std == 0:
            try:
                self.compute_stats()
            except Exception:
                pass

    @property
    def confidence_interval(self) -> Optional[ConfidenceInterval]:
        """获取首选的置信区间"""
        if not self.confidence_intervals:
            return None
        for key in ("bca", "percentile", "normal", "basic", "hpd"):
            if key in self.confidence_intervals:
                return self.confidence_intervals[key]
        return next(iter(self.confidence_intervals.values()))

    @confidence_interval.setter
    def confidence_interval(self, value: ConfidenceInterval):
        self.confidence_intervals[value.method] = value

    def compute_stats(self) -> Dict[str, float]:
        """从 samples 计算所有统计量，返回统计字典"""
        result: Dict[str, float] = {}
        if self.samples is None or len(self.samples) == 0:
            return result

        arr = np.asarray(self.samples, dtype=np.float64)
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr))
        self.variance = float(np.var(arr))
        self.median = float(np.median(arr))
        self.percentile_5 = float(np.percentile(arr, 5))
        self.percentile_95 = float(np.percentile(arr, 95))
        if abs(self.mean) > 1e-12:
            self.cv = float(self.std / abs(self.mean))

        try:
            from scipy import stats
            self.skewness = float(stats.skew(arr))
            self.kurtosis = float(stats.kurtosis(arr))
        except Exception:
            pass

        result = {
            "mean": self.mean,
            "std": self.std,
            "variance": self.variance,
            "median": self.median,
            "percentile_5": self.percentile_5,
            "percentile_95": self.percentile_95,
            "cv": self.cv,
        }
        if self.skewness is not None:
            result["skewness"] = self.skewness
        if self.kurtosis is not None:
            result["kurtosis"] = self.kurtosis
        return result

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        result = {
            "metric_name": self.metric_name,
            "nominal_value": float(self.nominal_value),
            "mean": float(self.mean),
            "std": float(self.std),
            "variance": float(self.variance),
            "median": float(self.median),
            "percentile_5": float(self.percentile_5),
            "percentile_95": float(self.percentile_95),
            "cv": float(self.cv),
            "confidence_intervals": {
                k: v.to_dict() for k, v in self.confidence_intervals.items()
            },
            "distribution_type": self.distribution_type,
            "skewness": float(self.skewness) if self.skewness is not None else None,
            "kurtosis": float(self.kurtosis) if self.kurtosis is not None else None,
        }
        if include_samples and self.samples is not None:
            result["samples"] = self.samples.tolist()
        return result

    def summary(self) -> str:
        lines = [f"=== {self.metric_name} 不确定性分析 ==="]
        lines.append(f"  标称值: {self.nominal_value:.4f}")
        lines.append(f"  均值 ± 标准差: {self.mean:.4f} ± {self.std:.4f}")
        lines.append(f"  中位数: {self.median:.4f}")
        lines.append(f"  变异系数 CV: {self.cv:.4f} ({self.cv * 100:.2f}%)")
        lines.append(f"  90% 范围: [{self.percentile_5:.4f}, {self.percentile_95:.4f}]")
        for name, ci in self.confidence_intervals.items():
            lines.append(f"  {name}: {ci.summary()}")
        return "\n".join(lines)


@dataclass
class FailureProbabilityResult:
    """
    失效概率估计结果

    Attributes:
        failure_mode: 失效模式名称
        probability: 估计的失效概率
        confidence_interval: 失效概率的置信区间
        standard_error: 标准误差
        ppm: PPM 缺陷率（每百万）
        log_ppm: log10(PPM) + 6，便于比较极低概率
        n_failures: 观测到的失效数
        n_samples: 总采样数
        estimate_method: 估计方法 ('monte_carlo', 'is', 'form', 'sorm', 'extrapolation')
        reliability_index: 可靠性指标 β（FORM/SORM）
    """
    failure_mode: str
    probability: float
    confidence_interval: Optional[ConfidenceInterval] = None
    standard_error: Optional[float] = None
    ppm: Optional[float] = None
    log_ppm: Optional[float] = None
    n_failures: int = 0
    n_samples: int = 0
    estimate_method: str = "monte_carlo"
    reliability_index: Optional[float] = None

    def __post_init__(self):
        if self.ppm is None:
            self.ppm = self.probability * 1e6
        if self.log_ppm is None:
            self.log_ppm = -np.log10(max(self.probability, 1e-15)) + 6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_mode": self.failure_mode,
            "probability": float(self.probability),
            "confidence_interval": (
                self.confidence_interval.to_dict()
                if self.confidence_interval is not None
                else None
            ),
            "standard_error": float(self.standard_error) if self.standard_error is not None else None,
            "ppm": float(self.ppm) if self.ppm is not None else None,
            "log_ppm": float(self.log_ppm) if self.log_ppm is not None else None,
            "n_failures": int(self.n_failures),
            "n_samples": int(self.n_samples),
            "estimate_method": self.estimate_method,
            "reliability_index": float(self.reliability_index) if self.reliability_index is not None else None,
        }

    def summary(self) -> str:
        lines = [f"=== {self.failure_mode} 失效概率 ==="]
        lines.append(f"  估计方法: {self.estimate_method}")
        lines.append(f"  失效概率: {self.probability:.2e}")
        if self.ppm is not None:
            lines.append(f"  PPM 缺陷率: {self.ppm:.2f} ppm")
        if self.reliability_index is not None:
            lines.append(f"  可靠性指标 β: {self.reliability_index:.3f}")
        if self.confidence_interval is not None:
            lines.append(f"  {self.confidence_interval.summary()}")
        lines.append(f"  失效数/样本数: {self.n_failures}/{self.n_samples}")
        return "\n".join(lines)


@dataclass
class ReliabilityResult:
    """
    可靠性分析完整结果

    Attributes:
        overall_failure_probability: 总体失效概率
        failure_probabilities: 各失效模式的失效概率
        reliability_index: 总体可靠性指标 β
        sensitivity_indices: 各参数的敏感度指数（一阶/总阶）
        parameter_contributions: 各参数对失效概率的贡献百分比
        risk_level: 风险等级 ('very_low', 'low', 'medium', 'high', 'very_high')
        recommendations: 可靠性改进建议
    """
    overall_failure_probability: float = 0.0
    failure_probabilities: Dict[str, FailureProbabilityResult] = field(default_factory=dict)
    reliability_index: Optional[float] = None
    sensitivity_indices: Dict[str, Dict[str, float]] = field(default_factory=dict)
    parameter_contributions: Dict[str, float] = field(default_factory=dict)
    risk_level: str = "low"
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_failure_probability": float(self.overall_failure_probability),
            "failure_probabilities": {
                k: v.to_dict() for k, v in self.failure_probabilities.items()
            },
            "reliability_index": float(self.reliability_index) if self.reliability_index is not None else None,
            "sensitivity_indices": self.sensitivity_indices,
            "parameter_contributions": {
                k: float(v) for k, v in self.parameter_contributions.items()
            },
            "risk_level": self.risk_level,
            "recommendations": list(self.recommendations),
        }

    def summary(self) -> str:
        lines = ["=== 可靠性分析结果 ==="]
        lines.append(f"  总体失效概率: {self.overall_failure_probability:.2e}")
        if self.reliability_index is not None:
            lines.append(f"  可靠性指标 β: {self.reliability_index:.3f}")
        lines.append(f"  风险等级: {self.risk_level}")
        lines.append("")
        lines.append("  各失效模式:")
        for mode, fp in self.failure_probabilities.items():
            lines.append(f"    {mode}: P={fp.probability:.2e} ({fp.ppm:.1f} ppm)")
        if self.parameter_contributions:
            lines.append("")
            lines.append("  参数贡献度:")
            for param, contrib in sorted(
                self.parameter_contributions.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {param}: {contrib * 100:.1f}%")
        if self.recommendations:
            lines.append("")
            lines.append("  改进建议:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")
        return "\n".join(lines)


@dataclass
class UQResult:
    """
    UQ 分析完整结果

    Attributes:
        config: UQ 配置
        method: 使用的 UQ 方法
        metric_uncertainties: 各成像指标的不确定性分析结果
        reliability: 可靠性分析结果（如启用）
        sensitivity_analysis: 敏感度分析结果（如启用）
        total_time: 总计算时间（秒）
        n_samples_completed: 实际完成的采样数
        nominal_metrics: 标称条件下的指标值
        sampled_conditions: 采样的工艺/模型条件列表（可选）
    """
    config: Optional[UQConfig] = None
    method: UQMethod = UQMethod.MONTE_CARLO
    metric_uncertainties: Dict[str, MetricUncertainty] = field(default_factory=dict)
    reliability: Optional[ReliabilityResult] = None
    sensitivity_analysis: Optional[Dict[str, Any]] = None
    total_time: float = 0.0
    n_samples_completed: int = 0
    nominal_metrics: Dict[str, float] = field(default_factory=dict)
    sampled_conditions: Optional[List[Dict[str, Any]]] = None
    n_samples: int = 0
    confidence_level: float = 0.95

    def __post_init__(self):
        if self.config is not None:
            if self.n_samples == 0:
                self.n_samples = self.n_samples_completed or self.config.n_samples
            if self.confidence_level == 0.95:
                self.confidence_level = self.config.confidence_level
        else:
            if self.n_samples_completed == 0:
                self.n_samples_completed = self.n_samples

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict() if self.config is not None else None,
            "method": self.method.value,
            "n_samples": int(self.n_samples),
            "confidence_level": float(self.confidence_level),
            "metric_uncertainties": {
                k: v.to_dict(include_samples=include_samples)
                for k, v in self.metric_uncertainties.items()
            },
            "reliability": self.reliability.to_dict() if self.reliability is not None else None,
            "sensitivity_analysis": self.sensitivity_analysis,
            "total_time": float(self.total_time),
            "n_samples_completed": int(self.n_samples_completed),
            "nominal_metrics": {k: float(v) for k, v in self.nominal_metrics.items()},
            "sampled_conditions": (
                self.sampled_conditions if include_samples and self.sampled_conditions is not None
                else None
            ),
        }

    def summary(self) -> str:
        _cl = self.config.confidence_level if self.config is not None else self.confidence_level
        _n = self.n_samples_completed if self.n_samples_completed > 0 else self.n_samples
        lines = [
            "=" * 60,
            "不确定性量化 (UQ) 分析报告",
            "研究问题: 优化结果在真实 fab 中有多可靠？",
            "=" * 60,
            "",
            f"分析方法: {self.method.value}",
            f"采样次数: {_n}",
            f"置信水平: {int(_cl * 100)}%",
            f"总耗时: {self.total_time:.1f}s",
            "",
        ]

        lines.append("--- 标称条件指标 ---")
        for name, val in self.nominal_metrics.items():
            lines.append(f"  {name}: {val:.4f}")

        lines.append("")
        lines.append("--- 各指标不确定性 ---")
        for name, mu in self.metric_uncertainties.items():
            lines.append("")
            lines.append(mu.summary())

        if self.reliability is not None:
            lines.append("")
            lines.append(self.reliability.summary())

        if self.sensitivity_analysis:
            lines.append("")
            lines.append("--- 敏感度分析 ---")
            for metric, indices in self.sensitivity_analysis.items():
                lines.append(f"  {metric}:")
                if isinstance(indices, dict):
                    for param, val in sorted(indices.items(), key=lambda x: -abs(x[1])):
                        lines.append(f"    {param}: {val:.4f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
