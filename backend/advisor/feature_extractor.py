# -*- coding: utf-8 -*-
"""
版图特征提取器

从输入版图（二值掩模）中提取以下关键特征：
    1. 频谱特征（SpectralFeatures）：FFT 频谱分析
    2. 最小 CD (Critical Dimension)
    3. 拐角密度 (Corner Density)
    4. 周期性评分 (Periodicity Score)
    5. 占空比 (Duty Cycle)
    6. 填充率 (Fill Ratio)
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from scipy.ndimage import (
    binary_erosion, binary_dilation, label, find_objects,
    generate_binary_structure, distance_transform_edt, sobel
)
from scipy.signal import find_peaks
import logging

from advisor.schemas import LayoutFeatures, SpectralFeatures

logger = logging.getLogger(__name__)


class LayoutFeatureExtractor:
    """
    版图特征提取器

    从二值掩模图案中提取 RET 策略选择所需的版图特征。
    所有方法均为静态方法，无需实例化即可使用。

    典型用法：
        features = LayoutFeatureExtractor.extract(mask, pixel_size=1.0)
    """

    @staticmethod
    def extract(
        mask: np.ndarray,
        pixel_size: float = 1.0,
        technology_node: str = 'duv_arf',
        wavelength: float = 193.0,
        na: float = 1.35,
    ) -> LayoutFeatures:
        """
        从二值掩模中提取完整版图特征

        Args:
            mask: 二值掩模图案 (0/1 或 0.0/1.0)
            pixel_size: 像素尺寸 (nm)
            technology_node: 技术节点 ('duv_arf' / 'euv')
            wavelength: 光源波长 (nm)
            na: 数值孔径

        Returns:
            LayoutFeatures 完整版图特征数据结构
        """
        mask = (mask >= 0.5).astype(np.float64)
        spectral = LayoutFeatureExtractor.extract_spectral_features(mask, pixel_size)
        min_cd = LayoutFeatureExtractor.extract_min_cd(mask, pixel_size)
        corner_density = LayoutFeatureExtractor.extract_corner_density(mask)
        periodicity, dominant_pitch = LayoutFeatureExtractor.extract_periodicity(mask, pixel_size)
        duty_cycle = LayoutFeatureExtractor.extract_duty_cycle(mask)
        fill_ratio = float(np.mean(mask))

        return LayoutFeatures(
            min_cd_nm=min_cd,
            corner_density=corner_density,
            periodicity_score=periodicity,
            dominant_pitch_nm=dominant_pitch,
            duty_cycle=duty_cycle,
            fill_ratio=fill_ratio,
            spectral=spectral,
            technology_node=technology_node,
            wavelength=wavelength,
            na=na,
            pixel_size=pixel_size,
            image_shape=mask.shape,
        )

    @staticmethod
    def extract_spectral_features(
        mask: np.ndarray,
        pixel_size: float = 1.0,
    ) -> SpectralFeatures:
        """
        提取频谱特征

        通过 2D FFT 分析版图的频域特性，提取：
        - dominant_frequency: 主频（最高能量频率分量），单位 cycles/μm
        - bandwidth_3db: 3dB 带宽
        - spectral_entropy: 频谱熵（归一化到 [0,1]），衡量频率分布的均匀性
        - high_freq_energy_ratio: 高频能量占比（> 2×主频的能量 / 总能量）
        - low_freq_energy_ratio: 低频能量占比（< 0.5×主频的能量 / 总能量）
        - peak_count: 频谱中的峰值数量
        - spectral_centroid: 频谱重心

        Args:
            mask: 二值掩模
            pixel_size: 像素尺寸 (nm)

        Returns:
            SpectralFeatures 频谱特征
        """
        ny, nx = mask.shape
        fft2 = np.fft.fft2(mask - np.mean(mask))
        power_spectrum = np.abs(np.fft.fftshift(fft2)) ** 2

        fy = np.fft.fftshift(np.fft.fftfreq(ny, pixel_size))
        fx = np.fft.fftshift(np.fft.fftfreq(nx, pixel_size))
        fy_grid, fx_grid = np.meshgrid(fy, fx, indexing='ij')
        freq_magnitude = np.sqrt(fy_grid ** 2 + fx_grid ** 2)

        total_energy = np.sum(power_spectrum)
        if total_energy < 1e-15:
            return SpectralFeatures()

        center_y, center_x = ny // 2, nx // 2
        mask_dc = np.zeros_like(power_spectrum, dtype=bool)
        dc_radius = max(2, int(0.02 * min(ny, nx)))
        y_grid, x_grid = np.ogrid[:ny, :nx]
        dc_region = ((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2) <= dc_radius ** 2
        ps_no_dc = power_spectrum.copy()
        ps_no_dc[dc_region] = 0
        energy_no_dc = np.sum(ps_no_dc)

        if energy_no_dc < 1e-15:
            return SpectralFeatures()

        radial_profile = LayoutFeatureExtractor._compute_radial_profile(
            ps_no_dc, freq_magnitude
        )

        freq_bins = radial_profile['freq']
        energy_bins = radial_profile['energy']

        peak_idx = np.argmax(energy_bins)
        dominant_frequency = freq_bins[peak_idx] if peak_idx < len(freq_bins) else 0.0

        half_max = energy_bins[peak_idx] / 2.0 if peak_idx < len(energy_bins) else 0.0
        above_half = np.where(energy_bins >= half_max)[0]
        if len(above_half) > 0:
            bandwidth_3db = freq_bins[above_half[-1]] - freq_bins[above_half[0]]
        else:
            bandwidth_3db = 0.0

        prob = energy_bins / (np.sum(energy_bins) + 1e-15)
        prob = prob[prob > 1e-15]
        spectral_entropy = float(-np.sum(prob * np.log2(prob + 1e-15)))
        max_entropy = np.log2(len(prob)) if len(prob) > 1 else 1.0
        spectral_entropy = spectral_entropy / max(max_entropy, 1e-8)
        spectral_entropy = min(spectral_entropy, 1.0)

        if dominant_frequency > 0:
            high_freq_mask = freq_magnitude > 2.0 * dominant_frequency
            low_freq_mask = (freq_magnitude > 0) & (freq_magnitude < 0.5 * dominant_frequency)
        else:
            high_freq_mask = freq_magnitude > np.max(freq_magnitude) * 0.5
            low_freq_mask = freq_magnitude < np.max(freq_magnitude) * 0.1

        high_freq_energy_ratio = float(np.sum(ps_no_dc[high_freq_mask]) / energy_no_dc)
        low_freq_energy_ratio = float(np.sum(ps_no_dc[low_freq_mask]) / energy_no_dc)

        peak_indices, _ = find_peaks(energy_bins, height=half_max * 0.1, distance=3)
        peak_count = len(peak_indices)

        spectral_centroid = float(np.sum(freq_bins * energy_bins) / (np.sum(energy_bins) + 1e-15))

        return SpectralFeatures(
            dominant_frequency=dominant_frequency * 1000.0,
            bandwidth_3db=bandwidth_3db * 1000.0,
            spectral_entropy=spectral_entropy,
            high_freq_energy_ratio=high_freq_energy_ratio,
            low_freq_energy_ratio=low_freq_energy_ratio,
            peak_count=peak_count,
            spectral_centroid=spectral_centroid * 1000.0,
        )

    @staticmethod
    def _compute_radial_profile(
        power_spectrum: np.ndarray,
        freq_magnitude: np.ndarray,
        num_bins: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        计算径向平均功率谱

        Args:
            power_spectrum: 2D 功率谱
            freq_magnitude: 对应的频率幅度网格
            num_bins: 径向分 bin 数

        Returns:
            {'freq': freq_bins, 'energy': energy_bins}
        """
        max_freq = np.max(freq_magnitude)
        if max_freq < 1e-15:
            return {'freq': np.zeros(1), 'energy': np.zeros(1)}

        bin_edges = np.linspace(0, max_freq, num_bins + 1)
        freq_bins = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        energy_bins = np.zeros(num_bins)

        flat_freq = freq_magnitude.ravel()
        flat_ps = power_spectrum.ravel()
        bin_indices = np.digitize(flat_freq, bin_edges) - 1
        valid = (bin_indices >= 0) & (bin_indices < num_bins)

        for i in range(num_bins):
            mask_bin = valid & (bin_indices == i)
            if np.any(mask_bin):
                energy_bins[i] = np.sum(flat_ps[mask_bin])

        return {'freq': freq_bins, 'energy': energy_bins}

    @staticmethod
    def extract_min_cd(mask: np.ndarray, pixel_size: float = 1.0) -> float:
        """
        提取最小 CD (Critical Dimension)

        使用距离变换估算版图中最细的线宽/间距。
        对亮区和暗区分别计算内部距离场，对每个连通域取最大距离
        （代表该域最粗处的半径），然后取 2×该值作为该域的 CD，
        最后取所有域中最小的 CD 作为全局最小 CD。

        对于周期性线/空间结构，这等价于测量最细线条的宽度。

        Args:
            mask: 二值掩模
            pixel_size: 像素尺寸 (nm)

        Returns:
            最小 CD (nm)
        """
        binary = mask >= 0.5

        if np.all(binary) or np.all(~binary):
            return float(min(mask.shape)) * pixel_size

        min_cd = float('inf')

        for region_binary, region_inverse in [(binary, ~binary), (~binary, binary)]:
            dist = distance_transform_edt(region_binary)
            struct = generate_binary_structure(2, 2)
            labeled_arr, num_features = label(region_binary, structure=struct)

            if num_features == 0:
                continue

            objects = find_objects(labeled_arr)
            for idx, obj in enumerate(objects):
                if obj is None:
                    continue
                component_mask = (labeled_arr == (idx + 1))
                area = int(np.sum(component_mask))
                if area < 2:
                    continue
                max_dist = float(np.max(dist[component_mask]))
                cd_estimate = 2.0 * max_dist
                if cd_estimate < min_cd:
                    min_cd = cd_estimate

        if min_cd == float('inf') or min_cd <= 0:
            min_cd = 1.0

        return min_cd * pixel_size

    @staticmethod
    def extract_corner_density(mask: np.ndarray) -> float:
        """
        提取拐角密度

        使用 Sobel 算子检测边缘，然后检测拐角点，
        拐角密度 = 拐角像素数 / 边缘像素数。

        高拐角密度意味着版图中有大量拐角结构
        (L 形、T 形结等)，这些区域通常需要更强的 RET。

        Args:
            mask: 二值掩模

        Returns:
            拐角密度 [0, 1]
        """
        binary = (mask >= 0.5).astype(np.float64)
        if np.all(binary < 0.5) or np.all(binary >= 0.5):
            return 0.0

        sy = sobel(binary, axis=0)
        sx = sobel(binary, axis=1)
        edge_magnitude = np.sqrt(sx ** 2 + sy ** 2)
        edge_mask = edge_magnitude > np.max(edge_magnitude) * 0.3
        edge_count = np.sum(edge_mask)

        if edge_count < 4:
            return 0.0

        syy = sobel(sy, axis=0)
        sxx = sobel(sx, axis=1)
        sxy = sobel(sy, axis=1)

        corner_response = (sxx * syy - sxy ** 2) - 0.04 * ((sxx + syy) ** 2)

        threshold = np.percentile(
            corner_response[edge_mask], 90
        ) if edge_count > 10 else np.max(corner_response) * 0.5
        corner_mask = (corner_response > threshold) & edge_mask
        corner_count = np.sum(corner_mask)

        return float(corner_count / max(edge_count, 1))

    @staticmethod
    def extract_periodicity(
        mask: np.ndarray,
        pixel_size: float = 1.0,
    ) -> Tuple[float, float]:
        """
        提取周期性评分和主导间距

        通过自相关函数分析版图的周期性：
        - periodicity_score: 周期性评分 [0, 1]，1 表示完全周期性
        - dominant_pitch: 主导间距 (nm)

        Args:
            mask: 二值掩模
            pixel_size: 像素尺寸 (nm)

        Returns:
            (periodicity_score, dominant_pitch_nm)
        """
        binary = (mask >= 0.5).astype(np.float64)
        ny, nx = binary.shape

        if np.sum(binary) < 4:
            return 0.0, 0.0

        center_y, center_x = ny // 2, nx // 2
        max_lag = min(ny, nx) // 4
        min_lag = 3

        row_means = np.mean(binary, axis=1)
        col_means = np.mean(binary, axis=0)

        row_var = np.var(row_means)
        col_var = np.var(col_means)

        if row_var < 1e-15 and col_var < 1e-15:
            return 0.0, 0.0

        best_score = 0.0
        best_pitch = 0.0

        if row_var >= col_var:
            signal = row_means
            for lag in range(min_lag, max_lag):
                if lag >= len(signal):
                    break
                corr = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
                if not np.isnan(corr) and abs(corr) > best_score:
                    best_score = abs(corr)
                    best_pitch = lag * pixel_size
        else:
            signal = col_means
            for lag in range(min_lag, max_lag):
                if lag >= len(signal):
                    break
                corr = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
                if not np.isnan(corr) and abs(corr) > best_score:
                    best_score = abs(corr)
                    best_pitch = lag * pixel_size

        periodicity_score = min(best_score, 1.0)

        return periodicity_score, best_pitch

    @staticmethod
    def extract_duty_cycle(mask: np.ndarray) -> float:
        """
        提取占空比

        占空比 = 亮区面积 / (亮区面积 + 暗区面积)
        对于周期性结构，duty_cycle ≈ CD / pitch

        Args:
            mask: 二值掩模

        Returns:
            占空比 [0, 1]
        """
        binary = mask >= 0.5
        fill = np.mean(binary)
        return float(fill)
