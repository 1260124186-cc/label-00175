# -*- coding: utf-8 -*-
"""
线边缘粗糙度（LER/LWR）评估模块

对蒙特卡洛仿真结果中的边缘位置进行统计分析，
计算线边缘粗糙度（LER）、线宽粗糙度（LWR）、
功率谱密度（PSD）、自相关函数（ACF）等指标。

核心功能：
1. LER/LWR 统计量计算
2. 功率谱密度（PSD）分析
3. 自相关函数（ACF）与相关长度
4. 边缘形态特征提取
5. 多实现结果的统计聚合
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import signal, stats

logger = logging.getLogger(__name__)


class EdgeDirection(Enum):
    """边缘方向枚举"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class RoughnessMetric(Enum):
    """粗糙度度量类型"""
    LER = "LER"           # 线边缘粗糙度 (3σ)
    LWR = "LWR"           # 线宽粗糙度 (3σ)
    LER_MEAN = "LER_mean"  # LER 均值
    LWR_MEAN = "LWR_mean"  # LWR 均值
    CORRELATION_LENGTH = "correlation_length"  # 相关长度
    RMS_ROUGHNESS = "RMS_roughness"            # RMS 粗糙度


@dataclass
class EdgeProfile:
    """
    单条边缘轮廓数据

    Attributes:
        positions: 边缘位置数组 (沿走线方向的位置)
        edge_indices: 边缘索引数组（垂直于走线方向）
        pixel_size: 像素尺寸 (nm)
        direction: 边缘方向
    """
    positions: np.ndarray
    edge_indices: np.ndarray
    pixel_size: float
    direction: EdgeDirection

    @property
    def length_nm(self) -> float:
        """边缘长度 (nm)"""
        return len(self.positions) * self.pixel_size

    @property
    def edge_nm(self) -> np.ndarray:
        """边缘位置 (nm)"""
        return self.edge_indices * self.pixel_size

    def detrend(self, order: int = 1) -> np.ndarray:
        """
        去除边缘趋势（多项式拟合去趋势）

        Args:
            order: 多项式阶数

        Returns:
            去趋势后的边缘位置
        """
        if len(self.edge_indices) < order + 1:
            return self.edge_indices.copy()

        x = np.arange(len(self.edge_indices))
        coeffs = np.polyfit(x, self.edge_indices, order)
        trend = np.polyval(coeffs, x)
        return self.edge_indices - trend


@dataclass
class SingleEdgeRoughnessResult:
    """
    单条边缘的粗糙度分析结果

    Attributes:
        direction: 边缘方向
        edge_id: 边缘编号（左/右/上/下）
        edge_profile: 原始边缘轮廓
        detrended_edge: 去趋势后的边缘轮廓
        ler_3sigma: LER (3σ) 值 (nm)
        ler_mean: LER 均值 (nm)
        rms_roughness: RMS 粗糙度 (nm)
        psd_frequencies: PSD 频率轴 (1/nm)
        psd_values: PSD 值 (nm³)
        acf_lags: ACF 滞后轴 (nm)
        acf_values: ACF 值
        correlation_length: 相关长度 (nm)
        skew: 偏度
        kurtosis: 峰度
    """
    direction: EdgeDirection
    edge_id: str
    edge_profile: EdgeProfile
    detrended_edge: np.ndarray
    ler_3sigma: float
    ler_mean: float
    rms_roughness: float
    psd_frequencies: np.ndarray
    psd_values: np.ndarray
    acf_lags: np.ndarray
    acf_values: np.ndarray
    correlation_length: float
    skew: float
    kurtosis: float

    def summary(self) -> str:
        """生成摘要字符串"""
        lines = [
            f"=== {self.direction.value} 边缘 {self.edge_id} 粗糙度分析 ===",
            f"  LER (3σ): {self.ler_3sigma:.2f} nm",
            f"  LER 均值: {self.ler_mean:.2f} nm",
            f"  RMS 粗糙度: {self.rms_roughness:.2f} nm",
            f"  相关长度: {self.correlation_length:.2f} nm",
            f"  偏度: {self.skew:.3f}",
            f"  峰度: {self.kurtosis:.3f}",
        ]
        return "\n".join(lines)


@dataclass
class LWRResult:
    """
    线宽粗糙度分析结果

    Attributes:
        direction: 方向
        line_id: 线条编号
        width_profile: 线宽轮廓 (nm)
        lwr_3sigma: LWR (3σ) 值 (nm)
        lwr_mean: LWR 均值 (nm)
        mean_width: 平均线宽 (nm)
        rms_width: 线宽 RMS (nm)
        psd_frequencies: PSD 频率轴 (1/nm)
        psd_values: PSD 值 (nm³)
        acf_lags: ACF 滞后轴 (nm)
        acf_values: ACF 值
        correlation_length: 相关长度 (nm)
    """
    direction: EdgeDirection
    line_id: str
    width_profile: np.ndarray
    lwr_3sigma: float
    lwr_mean: float
    mean_width: float
    rms_width: float
    psd_frequencies: np.ndarray
    psd_values: np.ndarray
    acf_lags: np.ndarray
    acf_values: np.ndarray
    correlation_length: float

    def summary(self) -> str:
        """生成摘要字符串"""
        lines = [
            f"=== {self.direction.value} 线条 {self.line_id} 线宽粗糙度 ===",
            f"  平均线宽: {self.mean_width:.2f} nm",
            f"  LWR (3σ): {self.lwr_3sigma:.2f} nm",
            f"  LWR 均值: {self.lwr_mean:.2f} nm",
            f"  线宽 RMS: {self.rms_width:.2f} nm",
            f"  相关长度: {self.correlation_length:.2f} nm",
        ]
        return "\n".join(lines)


@dataclass
class SingleRealizationRoughnessResult:
    """
    单次实现的粗糙度分析结果

    Attributes:
        realization_id: 实现ID
        edge_roughness: 各边缘的粗糙度结果
        line_roughness: 各线条的粗糙度结果
        overall_ler: 整体 LER 统计
        overall_lwr: 整体 LWR 统计
    """
    realization_id: int
    edge_roughness: Dict[str, SingleEdgeRoughnessResult] = field(default_factory=dict)
    line_roughness: Dict[str, LWRResult] = field(default_factory=dict)
    overall_ler: Dict[str, float] = field(default_factory=dict)
    overall_lwr: Dict[str, float] = field(default_factory=dict)


@dataclass
class MonteCarloRoughnessResult:
    """
    蒙特卡洛粗糙度统计结果

    Attributes:
        n_realizations: 实现次数
        ler_distributions: 各边缘 LER 分布
        lwr_distributions: 各线条 LWR 分布
        ler_stats: LER 统计量
        lwr_stats: LWR 统计量
        mean_psd: 平均 PSD
        mean_acf: 平均 ACF
        correlation_length_distribution: 相关长度分布
    """
    n_realizations: int = 0
    ler_distributions: Dict[str, np.ndarray] = field(default_factory=dict)
    lwr_distributions: Dict[str, np.ndarray] = field(default_factory=dict)
    ler_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    lwr_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mean_psd: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    mean_acf: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    correlation_length_distribution: Dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> str:
        """生成摘要字符串"""
        lines = [
            "=== 蒙特卡洛线边缘粗糙度统计结果 ===",
            f"  实现次数: {self.n_realizations}",
        ]

        for edge_key, stats_dict in self.ler_stats.items():
            lines.append("")
            lines.append(f"  {edge_key} LER 统计:")
            lines.append(f"    均值 LER: {stats_dict['mean']:.2f} nm")
            lines.append(f"    标准差 LER: {stats_dict['std']:.2f} nm")
            lines.append(f"    95% 分位数: [{stats_dict['p5']:.2f}, {stats_dict['p95']:.2f}] nm")

        for line_key, stats_dict in self.lwr_stats.items():
            lines.append("")
            lines.append(f"  {line_key} LWR 统计:")
            lines.append(f"    均值 LWR: {stats_dict['mean']:.2f} nm")
            lines.append(f"    标准差 LWR: {stats_dict['std']:.2f} nm")
            lines.append(f"    95% 分位数: [{stats_dict['p5']:.2f}, {stats_dict['p95']:.2f}] nm")

        return "\n".join(lines)


class EdgeRoughnessAnalyzer:
    """
    线边缘粗糙度分析器

    提供完整的 LER/LWR 分析功能，包括：
    - 单实现边缘粗糙度分析
    - 多实现蒙特卡洛统计聚合
    - PSD/ACF 分析
    - 相关长度估计
    """

    def __init__(
        self,
        pixel_size: float = 1.0,
        detrend_order: int = 1,
        psd_window: str = "hann",
        psd_nperseg: Optional[int] = None,
        acf_max_lag: Optional[int] = None,
    ):
        """
        初始化分析器

        Args:
            pixel_size: 像素尺寸 (nm)
            detrend_order: 去趋势多项式阶数
            psd_window: PSD 窗函数类型
            psd_nperseg: PSD 每段长度
            acf_max_lag: ACF 最大滞后
        """
        self.pixel_size = pixel_size
        self.detrend_order = detrend_order
        self.psd_window = psd_window
        self.psd_nperseg = psd_nperseg
        self.acf_max_lag = acf_max_lag

    def extract_edge_profiles(
        self,
        edge_positions: Dict[str, np.ndarray],
        direction: EdgeDirection,
    ) -> List[EdgeProfile]:
        """
        从边缘位置数据中提取边缘轮廓

        Args:
            edge_positions: 边缘位置字典（来自蒙特卡洛结果）
            direction: 边缘方向

        Returns:
            边缘轮廓列表
        """
        key = direction.value
        if key not in edge_positions or edge_positions[key] is None:
            return []

        edges_array = edge_positions[key]
        profiles = []

        for edge_idx in range(edges_array.shape[1]):
            edge_col = edges_array[:, edge_idx]
            valid_mask = ~np.isnan(edge_col)

            if not np.any(valid_mask):
                continue

            positions = np.where(valid_mask)[0]
            edge_indices = edge_col[valid_mask]

            profile = EdgeProfile(
                positions=positions,
                edge_indices=edge_indices,
                pixel_size=self.pixel_size,
                direction=direction,
            )
            profiles.append(profile)

        return profiles

    def compute_psd(
        self,
        signal_data: np.ndarray,
        fs: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算功率谱密度（PSD）

        使用 Welch 方法估计 PSD。

        Args:
            signal_data: 输入信号
            fs: 采样频率 (1/nm)

        Returns:
            (频率轴, PSD值)
        """
        if len(signal_data) < 4:
            return np.array([]), np.array([])

        nperseg = self.psd_nperseg or min(len(signal_data), 256)
        nperseg = min(nperseg, len(signal_data))

        try:
            freqs, psd = signal.welch(
                signal_data,
                fs=fs,
                window=self.psd_window,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                scaling="density",
            )
            return freqs, psd
        except Exception as e:
            logger.warning(f"PSD计算失败: {e}")
            return np.array([]), np.array([])

    def compute_acf(
        self,
        signal_data: np.ndarray,
        max_lag: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算自相关函数（ACF）

        Args:
            signal_data: 输入信号
            max_lag: 最大滞后

        Returns:
            (滞后轴, ACF值)
        """
        n = len(signal_data)
        if n < 2:
            return np.array([]), np.array([])

        max_lag = max_lag or self.acf_max_lag or min(n // 2, 100)
        max_lag = min(max_lag, n - 1)

        signal_centered = signal_data - np.mean(signal_data)
        variance = np.var(signal_data)

        if np.isclose(variance, 0):
            lags = np.arange(max_lag + 1)
            acf = np.zeros_like(lags, dtype=float)
            acf[0] = 1.0
            return lags, acf

        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0

        for lag in range(1, max_lag + 1):
            cov = np.mean(signal_centered[:-lag] * signal_centered[lag:])
            acf[lag] = cov / variance

        lags = np.arange(max_lag + 1) * self.pixel_size

        return lags, acf

    def estimate_correlation_length(
        self,
        acf_lags: np.ndarray,
        acf_values: np.ndarray,
        threshold: float = 1.0 / np.e,
    ) -> float:
        """
        估计相关长度

        Args:
            acf_lags: ACF 滞后轴 (nm)
            acf_values: ACF 值
            threshold: 相关长度阈值（默认 1/e）

        Returns:
            相关长度 (nm)
        """
        if len(acf_values) < 2:
            return 0.0

        for i in range(1, len(acf_values)):
            if acf_values[i] <= threshold:
                if i == 1:
                    return acf_lags[1]

                x1, x2 = acf_lags[i - 1], acf_lags[i]
                y1, y2 = acf_values[i - 1], acf_values[i]

                if np.isclose(y1, y2):
                    return x2

                t = (threshold - y1) / (y2 - y1)
                return x1 + t * (x2 - x1)

        return acf_lags[-1]

    def analyze_single_edge(
        self,
        edge_profile: EdgeProfile,
        edge_id: str,
    ) -> SingleEdgeRoughnessResult:
        """
        分析单条边缘的粗糙度

        Args:
            edge_profile: 边缘轮廓
            edge_id: 边缘标识

        Returns:
            单边缘粗糙度分析结果
        """
        detrended = edge_profile.detrend(order=self.detrend_order)
        edge_nm = detrended * self.pixel_size

        ler_3sigma = 3.0 * np.std(edge_nm)
        ler_mean = np.mean(np.abs(edge_nm - np.mean(edge_nm)))
        rms_roughness = np.sqrt(np.mean(edge_nm ** 2))

        fs = 1.0 / self.pixel_size
        freqs, psd = self.compute_psd(edge_nm, fs)
        acf_lags, acf = self.compute_acf(edge_nm)

        correlation_length = self.estimate_correlation_length(acf_lags, acf)

        skew = stats.skew(edge_nm) if len(edge_nm) > 2 else 0.0
        kurtosis = stats.kurtosis(edge_nm) if len(edge_nm) > 3 else 0.0

        return SingleEdgeRoughnessResult(
            direction=edge_profile.direction,
            edge_id=edge_id,
            edge_profile=edge_profile,
            detrended_edge=detrended,
            ler_3sigma=ler_3sigma,
            ler_mean=ler_mean,
            rms_roughness=rms_roughness,
            psd_frequencies=freqs,
            psd_values=psd,
            acf_lags=acf_lags,
            acf_values=acf,
            correlation_length=correlation_length,
            skew=skew,
            kurtosis=kurtosis,
        )

    def analyze_line_width(
        self,
        left_edge: EdgeProfile,
        right_edge: EdgeProfile,
        line_id: str,
    ) -> Optional[LWRResult]:
        """
        分析线宽粗糙度

        Args:
            left_edge: 左边缘轮廓
            right_edge: 右边缘轮廓
            line_id: 线条标识

        Returns:
            线宽粗糙度分析结果（如果边缘不匹配则返回 None）
        """
        common_positions = np.intersect1d(left_edge.positions, right_edge.positions)
        if len(common_positions) < 4:
            logger.warning(f"线条 {line_id} 边缘不匹配，无法计算 LWR")
            return None

        left_indices = np.searchsorted(left_edge.positions, common_positions)
        right_indices = np.searchsorted(right_edge.positions, common_positions)

        left_vals = left_edge.edge_indices[left_indices]
        right_vals = right_edge.edge_indices[right_indices]

        width_profile = np.abs(right_vals - left_vals) * self.pixel_size

        lwr_3sigma = 3.0 * np.std(width_profile)
        lwr_mean = np.mean(np.abs(width_profile - np.mean(width_profile)))
        mean_width = np.mean(width_profile)
        rms_width = np.sqrt(np.mean(width_profile ** 2))

        fs = 1.0 / self.pixel_size
        freqs, psd = self.compute_psd(width_profile, fs)
        acf_lags, acf = self.compute_acf(width_profile)

        correlation_length = self.estimate_correlation_length(acf_lags, acf)

        return LWRResult(
            direction=left_edge.direction,
            line_id=line_id,
            width_profile=width_profile,
            lwr_3sigma=lwr_3sigma,
            lwr_mean=lwr_mean,
            mean_width=mean_width,
            rms_width=rms_width,
            psd_frequencies=freqs,
            psd_values=psd,
            acf_lags=acf_lags,
            acf_values=acf,
            correlation_length=correlation_length,
        )

    def analyze_single_realization(
        self,
        edge_positions: Dict[str, np.ndarray],
        realization_id: int = 0,
    ) -> SingleRealizationRoughnessResult:
        """
        分析单次实现的粗糙度

        Args:
            edge_positions: 边缘位置字典
            realization_id: 实现ID

        Returns:
            单次实现粗糙度分析结果
        """
        result = SingleRealizationRoughnessResult(realization_id=realization_id)

        for direction in [EdgeDirection.HORIZONTAL, EdgeDirection.VERTICAL]:
            profiles = self.extract_edge_profiles(edge_positions, direction)

            if len(profiles) == 0:
                continue

            for i, profile in enumerate(profiles):
                edge_id = f"edge_{i}"
                edge_result = self.analyze_single_edge(profile, edge_id)
                key = f"{direction.value}_{edge_id}"
                result.edge_roughness[key] = edge_result

            for i in range(0, len(profiles) - 1, 2):
                line_id = f"line_{i // 2}"
                lwr_result = self.analyze_line_width(profiles[i], profiles[i + 1], line_id)
                if lwr_result is not None:
                    key = f"{direction.value}_{line_id}"
                    result.line_roughness[key] = lwr_result

        ler_values = [r.ler_3sigma for r in result.edge_roughness.values()]
        lwr_values = [r.lwr_3sigma for r in result.line_roughness.values()]

        if ler_values:
            result.overall_ler = {
                "mean": float(np.mean(ler_values)),
                "std": float(np.std(ler_values)),
                "min": float(np.min(ler_values)),
                "max": float(np.max(ler_values)),
            }

        if lwr_values:
            result.overall_lwr = {
                "mean": float(np.mean(lwr_values)),
                "std": float(np.std(lwr_values)),
                "min": float(np.min(lwr_values)),
                "max": float(np.max(lwr_values)),
            }

        return result

    def analyze_monte_carlo(
        self,
        all_edge_positions: List[Dict[str, np.ndarray]],
    ) -> MonteCarloRoughnessResult:
        """
        分析蒙特卡洛仿真的粗糙度结果

        Args:
            all_edge_positions: 所有实现的边缘位置列表

        Returns:
            蒙特卡洛粗糙度统计结果
        """
        n = len(all_edge_positions)
        if n == 0:
            return MonteCarloRoughnessResult(n_realizations=0)

        result = MonteCarloRoughnessResult(n_realizations=n)

        all_single_results: List[SingleRealizationRoughnessResult] = []
        for i, edge_pos in enumerate(all_edge_positions):
            single_result = self.analyze_single_realization(edge_pos, realization_id=i)
            all_single_results.append(single_result)

        all_edge_keys = set()
        all_line_keys = set()
        for sr in all_single_results:
            all_edge_keys.update(sr.edge_roughness.keys())
            all_line_keys.update(sr.line_roughness.keys())

        for edge_key in all_edge_keys:
            ler_values = []
            psd_list = []
            acf_list = []
            corr_lengths = []
            freq_ref = None
            lag_ref = None

            for sr in all_single_results:
                if edge_key in sr.edge_roughness:
                    er = sr.edge_roughness[edge_key]
                    ler_values.append(er.ler_3sigma)
                    corr_lengths.append(er.correlation_length)

                    if freq_ref is None and len(er.psd_frequencies) > 0:
                        freq_ref = er.psd_frequencies
                        psd_list.append(er.psd_values)
                    elif freq_ref is not None and len(er.psd_frequencies) == len(freq_ref):
                        psd_list.append(er.psd_values)

                    if lag_ref is None and len(er.acf_lags) > 0:
                        lag_ref = er.acf_lags
                        acf_list.append(er.acf_values)
                    elif lag_ref is not None and len(er.acf_lags) == len(lag_ref):
                        acf_list.append(er.acf_values)

            if ler_values:
                ler_arr = np.array(ler_values)
                result.ler_distributions[edge_key] = ler_arr
                result.ler_stats[edge_key] = {
                    "mean": float(np.mean(ler_arr)),
                    "std": float(np.std(ler_arr)),
                    "min": float(np.min(ler_arr)),
                    "max": float(np.max(ler_arr)),
                    "p5": float(np.percentile(ler_arr, 5)),
                    "p25": float(np.percentile(ler_arr, 25)),
                    "p50": float(np.percentile(ler_arr, 50)),
                    "p75": float(np.percentile(ler_arr, 75)),
                    "p95": float(np.percentile(ler_arr, 95)),
                }

                if psd_list:
                    mean_psd = np.mean(np.array(psd_list), axis=0)
                    result.mean_psd[edge_key] = (freq_ref, mean_psd)

                if acf_list:
                    mean_acf = np.mean(np.array(acf_list), axis=0)
                    result.mean_acf[edge_key] = (lag_ref, mean_acf)

                result.correlation_length_distribution[edge_key] = np.array(corr_lengths)

        for line_key in all_line_keys:
            lwr_values = []
            psd_list = []
            acf_list = []
            freq_ref = None
            lag_ref = None

            for sr in all_single_results:
                if line_key in sr.line_roughness:
                    lr = sr.line_roughness[line_key]
                    lwr_values.append(lr.lwr_3sigma)

                    if freq_ref is None and len(lr.psd_frequencies) > 0:
                        freq_ref = lr.psd_frequencies
                        psd_list.append(lr.psd_values)
                    elif freq_ref is not None and len(lr.psd_frequencies) == len(freq_ref):
                        psd_list.append(lr.psd_values)

                    if lag_ref is None and len(lr.acf_lags) > 0:
                        lag_ref = lr.acf_lags
                        acf_list.append(lr.acf_values)
                    elif lag_ref is not None and len(lr.acf_lags) == len(lag_ref):
                        acf_list.append(lr.acf_values)

            if lwr_values:
                lwr_arr = np.array(lwr_values)
                result.lwr_distributions[line_key] = lwr_arr
                result.lwr_stats[line_key] = {
                    "mean": float(np.mean(lwr_arr)),
                    "std": float(np.std(lwr_arr)),
                    "min": float(np.min(lwr_arr)),
                    "max": float(np.max(lwr_arr)),
                    "p5": float(np.percentile(lwr_arr, 5)),
                    "p25": float(np.percentile(lwr_arr, 25)),
                    "p50": float(np.percentile(lwr_arr, 50)),
                    "p75": float(np.percentile(lwr_arr, 75)),
                    "p95": float(np.percentile(lwr_arr, 95)),
                }

                if psd_list:
                    mean_psd = np.mean(np.array(psd_list), axis=0)
                    result.mean_psd[line_key] = (freq_ref, mean_psd)

                if acf_list:
                    mean_acf = np.mean(np.array(acf_list), axis=0)
                    result.mean_acf[line_key] = (lag_ref, mean_acf)

        return result


def compute_ler_from_edges(
    edge_positions: np.ndarray,
    pixel_size: float = 1.0,
    detrend_order: int = 1,
) -> Dict[str, float]:
    """
    便捷函数：从边缘位置数组计算 LER

    Args:
        edge_positions: 边缘位置数组 (n_positions, n_edges)
        pixel_size: 像素尺寸 (nm)
        detrend_order: 去趋势多项式阶数

    Returns:
        包含 LER 统计量的字典
    """
    if edge_positions.ndim == 1:
        edge_positions = edge_positions.reshape(-1, 1)

    ler_values = []
    rms_values = []

    for edge_idx in range(edge_positions.shape[1]):
        edge_col = edge_positions[:, edge_idx]
        valid_mask = ~np.isnan(edge_col)

        if np.sum(valid_mask) < 4:
            continue

        edge_data = edge_col[valid_mask]

        x = np.arange(len(edge_data))
        coeffs = np.polyfit(x, edge_data, detrend_order)
        trend = np.polyval(coeffs, x)
        detrended = edge_data - trend

        edge_nm = detrended * pixel_size
        ler_3sigma = 3.0 * np.std(edge_nm)
        rms = np.sqrt(np.mean(edge_nm ** 2))

        ler_values.append(ler_3sigma)
        rms_values.append(rms)

    if not ler_values:
        return {"LER_3sigma": 0.0, "LER_mean": 0.0, "RMS": 0.0, "n_edges": 0}

    return {
        "LER_3sigma": float(np.mean(ler_values)),
        "LER_3sigma_std": float(np.std(ler_values)),
        "LER_mean": float(np.mean(np.abs(edge_nm - np.mean(edge_nm)))),
        "RMS": float(np.mean(rms_values)),
        "n_edges": len(ler_values),
    }


def compute_lwr_from_edges(
    edge_positions: np.ndarray,
    pixel_size: float = 1.0,
) -> Dict[str, float]:
    """
    便捷函数：从边缘位置数组计算 LWR

    Args:
        edge_positions: 边缘位置数组 (n_positions, n_edges)，
                       相邻两列为一对线条的左右边缘
        pixel_size: 像素尺寸 (nm)

    Returns:
        包含 LWR 统计量的字典
    """
    if edge_positions.ndim == 1:
        return {"LWR_3sigma": 0.0, "n_lines": 0}

    n_edges = edge_positions.shape[1]
    lwr_values = []

    for i in range(0, n_edges - 1, 2):
        left_edge = edge_positions[:, i]
        right_edge = edge_positions[:, i + 1]

        valid_mask = ~np.isnan(left_edge) & ~np.isnan(right_edge)
        if np.sum(valid_mask) < 4:
            continue

        width = np.abs(right_edge[valid_mask] - left_edge[valid_mask]) * pixel_size
        lwr_3sigma = 3.0 * np.std(width)
        lwr_values.append(lwr_3sigma)

    if not lwr_values:
        return {"LWR_3sigma": 0.0, "LWR_mean": 0.0, "n_lines": 0}

    return {
        "LWR_3sigma": float(np.mean(lwr_values)),
        "LWR_3sigma_std": float(np.std(lwr_values)),
        "LWR_mean": float(np.mean(np.abs(width - np.mean(width)))),
        "mean_width": float(np.mean(width)),
        "n_lines": len(lwr_values),
    }
