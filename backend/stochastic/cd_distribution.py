# -*- coding: utf-8 -*-
"""
CD 分布统计分析模块

对蒙特卡洛仿真得到的 CD 数据进行统计分析，包括：
1. 基础统计量（均值、方差、标准差、偏度、峰度）
2. 直方图与概率密度函数（PDF）拟合
3. 累积分布函数（CDF）
4. 正态性检验
5. 分位数分析与置信区间
6. 过程能力指数（Cp, Cpk）
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.stats import (
    norm, lognorm, skewnorm, gamma, beta,
    kurtosis, skew, probplot, gaussian_kde,
)
import logging

from .monte_carlo import MonteCarloStochasticResult

logger = logging.getLogger(__name__)


class DistributionType(Enum):
    """分布类型枚举"""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    SKEWNORMAL = "skewnormal"
    GAMMA = "gamma"
    BETA = "beta"
    KDE = "kde"


@dataclass
class CDBasicStats:
    """
    CD 基础统计量

    Attributes:
        n_samples: 样本数量
        mean: 均值 (nm)
        std: 标准差 (nm)
        variance: 方差 (nm²)
        min: 最小值 (nm)
        max: 最大值 (nm)
        range: 极差 (nm)
        median: 中位数 (nm)
        mode: 众数 (nm)
        skew: 偏度
        kurtosis: 峰度（ excess kurtosis）
        q1: 第一四分位数 (nm)
        q3: 第三四分位数 (nm)
        iqr: 四分位距 (nm)
        ci_lower: 95%置信区间下界 (nm)
        ci_upper: 95%置信区间上界 (nm)
        standard_error: 标准误 (nm)
    """
    n_samples: int
    mean: float
    std: float
    variance: float
    min: float
    max: float
    range: float
    median: float
    mode: float
    skew: float
    kurtosis: float
    q1: float
    q3: float
    iqr: float
    ci_lower: float
    ci_upper: float
    standard_error: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'n_samples': self.n_samples,
            'mean': self.mean,
            'std': self.std,
            'variance': self.variance,
            'min': self.min,
            'max': self.max,
            'range': self.range,
            'median': self.median,
            'mode': self.mode,
            'skew': self.skew,
            'kurtosis': self.kurtosis,
            'q1': self.q1,
            'q3': self.q3,
            'iqr': self.iqr,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'standard_error': self.standard_error,
        }


@dataclass
class DistributionFitResult:
    """
    分布拟合结果

    Attributes:
        distribution_type: 分布类型
        parameters: 分布参数
        ks_statistic: Kolmogorov-Smirnov 检验统计量
        ks_pvalue: KS 检验 p 值
        ad_statistic: Anderson-Darling 检验统计量
        ad_pvalue: AD 检验 p 值（近似）
        log_likelihood: 对数似然值
        aic: AIC 信息准则
        bic: BIC 信息准则
        pdf: 概率密度函数采样点 (x, y)
        cdf: 累积分布函数采样点 (x, y)
    """
    distribution_type: DistributionType
    parameters: Dict[str, float]
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float
    ad_pvalue: float
    log_likelihood: float
    aic: float
    bic: float
    pdf: Tuple[np.ndarray, np.ndarray]
    cdf: Tuple[np.ndarray, np.ndarray]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'distribution_type': self.distribution_type.value,
            'parameters': self.parameters,
            'ks_statistic': self.ks_statistic,
            'ks_pvalue': self.ks_pvalue,
            'ad_statistic': self.ad_statistic,
            'ad_pvalue': self.ad_pvalue,
            'log_likelihood': self.log_likelihood,
            'aic': self.aic,
            'bic': self.bic,
        }


@dataclass
class ProcessCapability:
    """
    过程能力指数

    Attributes:
        cp: 潜在过程能力指数 Cp = (USL - LSL) / (6σ)
        cpu: 上单侧过程能力指数 CPU = (USL - μ) / (3σ)
        cpl: 下单侧过程能力指数 CPL = (μ - LSL) / (3σ)
        cpk: 实际过程能力指数 Cpk = min(CPU, CPL)
        cpm: 目标值过程能力指数 Cpm
        pp: 过程性能指数 Pp
        ppk: 过程性能指数 Ppk
        usl: 上规格限 (nm)
        lsl: 下规格限 (nm)
        target: 目标值 (nm)
        yield_estimate: 估算良率 (%)
        ppm_outside: 百万分率缺陷数
    """
    cp: float
    cpu: float
    cpl: float
    cpk: float
    cpm: float
    pp: float
    ppk: float
    usl: float
    lsl: float
    target: float
    yield_estimate: float
    ppm_outside: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'cp': self.cp,
            'cpu': self.cpu,
            'cpl': self.cpl,
            'cpk': self.cpk,
            'cpm': self.cpm,
            'pp': self.pp,
            'ppk': self.ppk,
            'usl': self.usl,
            'lsl': self.lsl,
            'target': self.target,
            'yield_estimate': self.yield_estimate,
            'ppm_outside': self.ppm_outside,
        }


@dataclass
class CDDistributionAnalysis:
    """
    CD 分布完整分析结果

    Attributes:
        cd_values: 原始 CD 值数组 (nm)
        nominal_cd: 标称 CD (无噪声时的 CD, nm)
        basic_stats: 基础统计量
        distribution_fits: 各分布拟合结果（按 AIC 排序）
        best_fit: 最佳拟合分布
        process_capability: 过程能力指数（如果提供了规格限）
        histogram: 直方图数据 (counts, bin_edges)
        outliers: 异常值检测结果
    """
    cd_values: np.ndarray
    nominal_cd: Optional[float] = None
    basic_stats: Optional[CDBasicStats] = None
    distribution_fits: List[DistributionFitResult] = field(default_factory=list)
    best_fit: Optional[DistributionFitResult] = None
    process_capability: Optional[ProcessCapability] = None
    histogram: Optional[Tuple[np.ndarray, np.ndarray]] = None
    outliers: Optional[Dict[str, Any]] = None

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        result = {
            'nominal_cd': self.nominal_cd,
            'basic_stats': self.basic_stats.to_dict() if self.basic_stats else None,
            'best_fit': self.best_fit.to_dict() if self.best_fit else None,
            'process_capability': (
                self.process_capability.to_dict() if self.process_capability else None
            ),
            'n_distribution_fits': len(self.distribution_fits),
            'outliers': self.outliers,
        }
        if include_samples:
            result['cd_values'] = self.cd_values.tolist()
        return result

    def summary(self) -> str:
        lines = ["=== CD 分布统计分析 ==="]

        if self.nominal_cd is not None:
            lines.append(f"  标称CD: {self.nominal_cd:.2f} nm")

        if self.basic_stats:
            s = self.basic_stats
            lines.append("")
            lines.append("  基础统计:")
            lines.append(f"    样本数: {s.n_samples}")
            lines.append(f"    均值: {s.mean:.2f} nm")
            lines.append(f"    标准差: {s.std:.2f} nm (3σ = {3*s.std:.2f} nm)")
            lines.append(f"    范围: [{s.min:.2f}, {s.max:.2f}] nm")
            lines.append(f"    中位数: {s.median:.2f} nm")
            lines.append(f"    偏度: {s.skew:.3f}")
            lines.append(f"    峰度: {s.kurtosis:.3f}")
            lines.append(f"    95% CI: [{s.ci_lower:.2f}, {s.ci_upper:.2f}] nm")

        if self.best_fit:
            lines.append("")
            lines.append("  最佳分布拟合:")
            lines.append(f"    分布类型: {self.best_fit.distribution_type.value}")
            lines.append(f"    KS 统计量: {self.best_fit.ks_statistic:.4f} (p={self.best_fit.ks_pvalue:.4f})")
            lines.append(f"    AIC: {self.best_fit.aic:.2f}")

        if self.process_capability:
            pc = self.process_capability
            lines.append("")
            lines.append("  过程能力:")
            lines.append(f"    规格限: [{pc.lsl:.2f}, {pc.usl:.2f}] nm (目标: {pc.target:.2f} nm)")
            lines.append(f"    Cp: {pc.cp:.3f}")
            lines.append(f"    Cpk: {pc.cpk:.3f}")
            lines.append(f"    Cpm: {pc.cpm:.3f}")
            lines.append(f"    估算良率: {pc.yield_estimate:.2f}%")
            lines.append(f"    PPM缺陷: {pc.ppm_outside:.1f}")

        if self.outliers:
            lines.append("")
            lines.append("  异常值检测:")
            lines.append(f"    方法: {self.outliers.get('method', 'IQR')}")
            lines.append(f"    异常值数量: {self.outliers.get('n_outliers', 0)}")
            if 'indices' in self.outliers:
                lines.append(f"    异常值: {self.outliers['indices'][:5]}...")

        return "\n".join(lines)


class CDDistributionAnalyzer:
    """
    CD 分布分析器

    对蒙特卡洛仿真得到的 CD 数据进行全面统计分析。

    使用方式::

        analyzer = CDDistributionAnalyzer()
        analysis = analyzer.analyze(
            cd_values=result.all_cd_horizontal,
            nominal_cd=nominal_cd,
            usl=cd_target * 1.1,
            lsl=cd_target * 0.9,
        )
        print(analysis.summary())
    """

    def __init__(self):
        """初始化分析器"""
        pass

    def compute_basic_stats(
        self,
        cd_values: np.ndarray,
        confidence_level: float = 0.95,
    ) -> CDBasicStats:
        """
        计算基础统计量

        Args:
            cd_values: CD 值数组 (nm)
            confidence_level: 置信水平 (默认 95%)

        Returns:
            CDBasicStats 基础统计量
        """
        arr = np.asarray(cd_values, dtype=np.float64)
        n = len(arr)

        if n == 0:
            raise ValueError("CD 值数组为空")

        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))
        var_val = float(np.var(arr, ddof=1))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        range_val = max_val - min_val
        median_val = float(np.median(arr))

        try:
            mode_val = float(stats.mode(arr, keepdims=False).mode)
        except Exception:
            mode_val = median_val

        skew_val = float(skew(arr))
        kurt_val = float(kurtosis(arr))

        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1

        se = std_val / np.sqrt(n) if n > 1 else 0.0
        alpha = 1 - confidence_level
        if n > 30:
            z = stats.norm.ppf(1 - alpha / 2)
            ci_lower = mean_val - z * se
            ci_upper = mean_val + z * se
        else:
            t = stats.t.ppf(1 - alpha / 2, df=n - 1)
            ci_lower = mean_val - t * se
            ci_upper = mean_val + t * se

        return CDBasicStats(
            n_samples=n,
            mean=mean_val,
            std=std_val,
            variance=var_val,
            min=min_val,
            max=max_val,
            range=range_val,
            median=median_val,
            mode=mode_val,
            skew=skew_val,
            kurtosis=kurt_val,
            q1=q1,
            q3=q3,
            iqr=iqr,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            standard_error=float(se),
        )

    def _fit_distribution(
        self,
        data: np.ndarray,
        dist_type: DistributionType,
        x_eval: np.ndarray,
    ) -> Optional[DistributionFitResult]:
        """
        拟合单个分布

        Args:
            data: 数据数组
            dist_type: 分布类型
            x_eval: PDF/CDF 评估点

        Returns:
            DistributionFitResult 或 None（拟合失败时）
        """
        try:
            if dist_type == DistributionType.NORMAL:
                params = norm.fit(data)
                dist = norm(*params)
                param_dict = {'loc': params[0], 'scale': params[1]}
                n_params = 2

            elif dist_type == DistributionType.LOGNORMAL:
                params = lognorm.fit(data, floc=0)
                dist = lognorm(*params)
                param_dict = {'s': params[0], 'loc': params[1], 'scale': params[2]}
                n_params = 3

            elif dist_type == DistributionType.SKEWNORMAL:
                params = skewnorm.fit(data)
                dist = skewnorm(*params)
                param_dict = {'a': params[0], 'loc': params[1], 'scale': params[2]}
                n_params = 3

            elif dist_type == DistributionType.GAMMA:
                params = gamma.fit(data, floc=0)
                dist = gamma(*params)
                param_dict = {'a': params[0], 'loc': params[1], 'scale': params[2]}
                n_params = 3

            elif dist_type == DistributionType.BETA:
                data_min = data.min()
                data_max = data.max()
                data_range = data_max - data_min + 1e-10
                data_scaled = (data - data_min) / data_range
                data_scaled = np.clip(data_scaled, 0.01, 0.99)
                params = beta.fit(data_scaled, floc=0, fscale=1)
                dist = beta(params[0], params[1], loc=data_min, scale=data_range)
                param_dict = {'a': params[0], 'b': params[1], 'loc': data_min, 'scale': data_range}
                n_params = 4

            elif dist_type == DistributionType.KDE:
                if len(data) < 5:
                    return None
                kde = gaussian_kde(data)
                pdf_y = kde(x_eval)
                cdf_y = np.array([kde.integrate_box_1d(-np.inf, x) for x in x_eval])
                log_likelihood = float(np.sum(np.log(kde(data) + 1e-300)))
                n_params = len(data)
                aic = 2 * n_params - 2 * log_likelihood
                bic = n_params * np.log(len(data)) - 2 * log_likelihood

                return DistributionFitResult(
                    distribution_type=dist_type,
                    parameters={'bandwidth': float(kde.factor)},
                    ks_statistic=0.0,
                    ks_pvalue=0.0,
                    ad_statistic=0.0,
                    ad_pvalue=0.0,
                    log_likelihood=log_likelihood,
                    aic=aic,
                    bic=bic,
                    pdf=(x_eval, pdf_y),
                    cdf=(x_eval, cdf_y),
                )

            else:
                return None

            pdf_y = dist.pdf(x_eval)
            cdf_y = dist.cdf(x_eval)

            ks_stat, ks_p = stats.kstest(data, dist.cdf)

            try:
                ad_result = stats.anderson(data, dist='norm' if dist_type == DistributionType.NORMAL else 'expon')
                ad_stat = float(ad_result.statistic)
                ad_p = 0.05
            except Exception:
                ad_stat = 0.0
                ad_p = 0.0

            log_likelihood = float(np.sum(np.log(dist.pdf(data) + 1e-300)))
            n = len(data)
            aic = 2 * n_params - 2 * log_likelihood
            bic = n_params * np.log(n) - 2 * log_likelihood

            return DistributionFitResult(
                distribution_type=dist_type,
                parameters=param_dict,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_p),
                ad_statistic=float(ad_stat),
                ad_pvalue=float(ad_p),
                log_likelihood=log_likelihood,
                aic=float(aic),
                bic=float(bic),
                pdf=(x_eval, pdf_y),
                cdf=(x_eval, cdf_y),
            )

        except Exception as e:
            logger.debug(f"拟合 {dist_type.value} 分布失败: {e}")
            return None

    def fit_distributions(
        self,
        cd_values: np.ndarray,
        distributions: Optional[List[DistributionType]] = None,
        n_eval_points: int = 200,
    ) -> List[DistributionFitResult]:
        """
        拟合多种分布并排序

        Args:
            cd_values: CD 值数组 (nm)
            distributions: 要拟合的分布类型列表，None 则使用所有分布
            n_eval_points: PDF/CDF 评估点数

        Returns:
            按 AIC 升序排列的分布拟合结果列表（AIC越小越好）
        """
        arr = np.asarray(cd_values, dtype=np.float64)

        if distributions is None:
            distributions = [
                DistributionType.NORMAL,
                DistributionType.LOGNORMAL,
                DistributionType.SKEWNORMAL,
                DistributionType.GAMMA,
                DistributionType.KDE,
            ]

        data_min = arr.min()
        data_max = arr.max()
        margin = 0.1 * (data_max - data_min + 1e-10)
        x_eval = np.linspace(data_min - margin, data_max + margin, n_eval_points)

        results = []
        for dist_type in distributions:
            result = self._fit_distribution(arr, dist_type, x_eval)
            if result is not None:
                results.append(result)

        results.sort(key=lambda r: r.aic)

        return results

    def compute_process_capability(
        self,
        cd_values: np.ndarray,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
        target: Optional[float] = None,
    ) -> Optional[ProcessCapability]:
        """
        计算过程能力指数

        Args:
            cd_values: CD 值数组 (nm)
            usl: 上规格限 (nm)，None 则使用 mean + 3*std
            lsl: 下规格限 (nm)，None 则使用 mean - 3*std
            target: 目标值 (nm)，None 则使用均值

        Returns:
            ProcessCapability 过程能力指数
        """
        arr = np.asarray(cd_values, dtype=np.float64)
        n = len(arr)

        if n < 2:
            return None

        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))
        std_within = float(np.std(arr, ddof=1))

        if target is None:
            target = mean_val

        if usl is None:
            usl = mean_val + 3 * std_val
        if lsl is None:
            lsl = mean_val - 3 * std_val

        six_sigma = 6 * std_val
        three_sigma = 3 * std_val

        if six_sigma > 0:
            cp = (usl - lsl) / six_sigma
        else:
            cp = float('inf')

        if three_sigma > 0:
            cpu = (usl - mean_val) / three_sigma
            cpl = (mean_val - lsl) / three_sigma
        else:
            cpu = float('inf')
            cpl = float('inf')

        cpk = min(cpu, cpl)

        sigma_t = std_val if std_val > 0 else 1e-10
        cpm = (usl - lsl) / (6 * np.sqrt(1 + ((mean_val - target) / sigma_t) ** 2) * sigma_t)

        if std_within > 0:
            pp = (usl - lsl) / (6 * std_within)
            ppu = (usl - mean_val) / (3 * std_within)
            ppl = (mean_val - lsl) / (3 * std_within)
            ppk = min(ppu, ppl)
        else:
            pp = float('inf')
            ppk = float('inf')

        if std_val > 0:
            p_above = 1 - norm.cdf(usl, loc=mean_val, scale=std_val)
            p_below = norm.cdf(lsl, loc=mean_val, scale=std_val)
            p_outside = p_above + p_below
            yield_estimate = (1 - p_outside) * 100
            ppm_outside = p_outside * 1e6
        else:
            yield_estimate = 100.0
            ppm_outside = 0.0

        return ProcessCapability(
            cp=float(cp),
            cpu=float(cpu),
            cpl=float(cpl),
            cpk=float(cpk),
            cpm=float(cpm),
            pp=float(pp),
            ppk=float(ppk),
            usl=float(usl),
            lsl=float(lsl),
            target=float(target),
            yield_estimate=float(yield_estimate),
            ppm_outside=float(ppm_outside),
        )

    def detect_outliers(
        self,
        cd_values: np.ndarray,
        method: str = 'iqr',
        iqr_factor: float = 1.5,
        z_threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """
        检测异常值

        Args:
            cd_values: CD 值数组 (nm)
            method: 检测方法 ('iqr', 'zscore', 'mad')
            iqr_factor: IQR 方法的倍数因子
            z_threshold: Z-score 方法的阈值

        Returns:
            异常值检测结果字典
        """
        arr = np.asarray(cd_values, dtype=np.float64)
        n = len(arr)

        if method == 'iqr':
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_factor * iqr
            upper_bound = q3 + iqr_factor * iqr
            outlier_mask = (arr < lower_bound) | (arr > upper_bound)
            bounds = {'lower': float(lower_bound), 'upper': float(upper_bound)}

        elif method == 'zscore':
            mean_val = np.mean(arr)
            std_val = np.std(arr, ddof=1)
            if std_val > 0:
                z_scores = np.abs((arr - mean_val) / std_val)
                outlier_mask = z_scores > z_threshold
            else:
                outlier_mask = np.zeros(n, dtype=bool)
            bounds = {'z_threshold': z_threshold}

        elif method == 'mad':
            median_val = np.median(arr)
            mad = np.median(np.abs(arr - median_val))
            if mad > 0:
                modified_z = 0.6745 * (arr - median_val) / mad
                outlier_mask = np.abs(modified_z) > z_threshold
            else:
                outlier_mask = np.zeros(n, dtype=bool)
            bounds = {'mad': float(mad), 'threshold': z_threshold}

        else:
            raise ValueError(f"未知的异常值检测方法: {method}")

        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = arr[outlier_mask].tolist()

        return {
            'method': method,
            'n_outliers': int(np.sum(outlier_mask)),
            'indices': outlier_indices,
            'values': outlier_values,
            'bounds': bounds,
        }

    def analyze(
        self,
        cd_values: np.ndarray,
        nominal_cd: Optional[float] = None,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
        target: Optional[float] = None,
        confidence_level: float = 0.95,
        n_bins: Optional[int] = None,
        distributions: Optional[List[DistributionType]] = None,
        outlier_method: str = 'iqr',
    ) -> CDDistributionAnalysis:
        """
        完整 CD 分布分析

        Args:
            cd_values: CD 值数组 (nm)
            nominal_cd: 标称 CD (无噪声时的 CD, nm)
            usl: 上规格限 (nm)
            lsl: 下规格限 (nm)
            target: 目标 CD (nm)，None 则使用 nominal_cd 或均值
            confidence_level: 置信水平
            n_bins: 直方图箱数，None 则自动计算
            distributions: 要拟合的分布列表
            outlier_method: 异常值检测方法

        Returns:
            CDDistributionAnalysis 完整分析结果
        """
        arr = np.asarray(cd_values, dtype=np.float64)

        if n_bins is None:
            n_bins = min(int(np.sqrt(len(arr))) + 1, 50)

        histogram = np.histogram(arr, bins=n_bins)

        basic_stats = self.compute_basic_stats(arr, confidence_level)

        dist_fits = self.fit_distributions(arr, distributions)
        best_fit = dist_fits[0] if dist_fits else None

        if target is None and nominal_cd is not None:
            target = nominal_cd

        process_cap = None
        if usl is not None or lsl is not None:
            process_cap = self.compute_process_capability(arr, usl, lsl, target)

        outliers = self.detect_outliers(arr, method=outlier_method)

        return CDDistributionAnalysis(
            cd_values=arr,
            nominal_cd=nominal_cd,
            basic_stats=basic_stats,
            distribution_fits=dist_fits,
            best_fit=best_fit,
            process_capability=process_cap,
            histogram=histogram,
            outliers=outliers,
        )

    def analyze_from_monte_carlo(
        self,
        mc_result: MonteCarloStochasticResult,
        direction: str = 'horizontal',
        **kwargs,
    ) -> CDDistributionAnalysis:
        """
        从蒙特卡洛结果进行 CD 分布分析

        Args:
            mc_result: 蒙特卡洛仿真结果
            direction: CD 方向 ('horizontal' 或 'vertical')
            **kwargs: 传递给 analyze() 的额外参数

        Returns:
            CDDistributionAnalysis 完整分析结果
        """
        if direction == 'horizontal':
            cd_values = mc_result.all_cd_horizontal
        elif direction == 'vertical':
            cd_values = mc_result.all_cd_vertical
        else:
            raise ValueError(f"未知方向: {direction}")

        if cd_values is None or len(cd_values) == 0:
            raise ValueError(f"没有 {direction} 方向的 CD 数据")

        nominal_cd = None
        if mc_result.nominal_wafer_binary is not None:
            from core.litho_metrics import compute_cd
            nomimal_stats = compute_cd(
                mc_result.nominal_wafer_binary,
                direction=direction,
                pixel_size=mc_result.config.pixel_size,
            )
            nominal_cd = nomimal_stats['cd_mean']

        return self.analyze(
            cd_values=cd_values,
            nominal_cd=nominal_cd,
            **kwargs,
        )


def analyze_cd_distribution(
    cd_values: np.ndarray,
    **kwargs,
) -> CDDistributionAnalysis:
    """
    便捷函数：分析 CD 分布

    Args:
        cd_values: CD 值数组 (nm)
        **kwargs: 传递给 CDDistributionAnalyzer.analyze() 的参数

    Returns:
        CDDistributionAnalysis 完整分析结果
    """
    analyzer = CDDistributionAnalyzer()
    return analyzer.analyze(cd_values, **kwargs)
