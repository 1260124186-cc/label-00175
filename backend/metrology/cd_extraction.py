# -*- coding: utf-8 -*-
"""
CD (Critical Dimension) 提取算法模块

对仿真晶圆图沿指定测量线提取线宽，支持多种经典计量算法：
1. Threshold Crossing: 阈值穿通法，基于灰度阈值检测边缘位置
2. Derivative Peak: 导数峰值法，基于信号导数的极大/极小值定位边缘
3. Linear Regression: 线性回归法，对边缘过渡区进行线性拟合
4. Polynomial Fit: 多项式拟合法，对边缘轮廓进行高阶多项式拟合
"""

import numpy as np
from numba import jit
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy.ndimage import map_coordinates, gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


class CDExtractionMethod(Enum):
    """CD 提取方法枚举"""
    THRESHOLD_CROSSING = "threshold_crossing"
    DERIVATIVE_PEAK = "derivative_peak"
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL_FIT = "polynomial_fit"


@dataclass
class MeasurementLine:
    """
    测量线定义

    Attributes:
        start: 起点坐标 (y, x)，像素单位
        end: 终点坐标 (y, x)，像素单位
        direction: 测量方向标签 ('horizontal', 'vertical', 'diagonal')
        name: 测量线名称，用于报告标识
    """
    start: Tuple[float, float]
    end: Tuple[float, float]
    direction: str = "horizontal"
    name: str = "ML1"


@dataclass
class CDExtractionResult:
    """
    CD 提取结果

    Attributes:
        cd_value: 提取的 CD 值 (nm)
        left_edge_pos: 左边缘位置 (nm, 沿测量线方向)
        right_edge_pos: 右边缘位置 (nm, 沿测量线方向)
        method: 使用的提取方法
        profile_intensity: 沿测量线的强度分布
        profile_positions: 沿测量线的位置坐标 (nm)
        edge_positions_raw: 原始像素级边缘位置
        confidence: 提取置信度 (0~1)
    """
    cd_value: float
    left_edge_pos: float
    right_edge_pos: float
    method: str
    profile_intensity: np.ndarray
    profile_positions: np.ndarray
    edge_positions_raw: Tuple[float, float]
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cd_value': self.cd_value,
            'left_edge_pos': self.left_edge_pos,
            'right_edge_pos': self.right_edge_pos,
            'method': self.method,
            'confidence': self.confidence,
        }


@jit(nopython=True, cache=True)
def _bilinear_interpolate(image: np.ndarray, y: float, x: float) -> float:
    """
    双线性插值获取亚像素精度的图像值

    Args:
        image: 输入图像 (2D)
        y: 纵坐标 (浮点)
        x: 横坐标 (浮点)

    Returns:
        插值后的像素值
    """
    ny, nx = image.shape
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, ny - 1)
    x1 = min(x0 + 1, nx - 1)

    y0 = max(y0, 0)
    x0 = max(x0, 0)

    fy = y - y0
    fx = x - x0

    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]

    v0 = v00 * (1 - fx) + v01 * fx
    v1 = v10 * (1 - fx) + v11 * fx

    return v0 * (1 - fy) + v1 * fy


def extract_profile(image: np.ndarray,
                    line: MeasurementLine,
                    pixel_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    沿测量线提取一维强度分布

    使用双线性插值获取亚像素精度的采样。

    Args:
        image: 输入晶圆图像 (2D)
        line: 测量线定义
        pixel_size: 像素尺寸 (nm)

    Returns:
        (intensity_profile, position_profile)
        - intensity_profile: 沿测量线的强度值
        - position_profile: 沿测量线的位置坐标 (nm)，从起点开始
    """
    y0, x0 = line.start
    y1, x1 = line.end

    length_pix = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
    n_samples = int(np.ceil(length_pix * 2)) + 1

    t = np.linspace(0, 1, n_samples)
    ys = y0 + t * (y1 - y0)
    xs = x0 + t * (x1 - x0)

    coords = np.vstack([ys, xs])
    profile = map_coordinates(image.astype(np.float64), coords, order=1, mode='nearest')

    positions = np.linspace(0, length_pix * pixel_size, n_samples)

    return profile, positions


@jit(nopython=True, cache=True)
def _find_threshold_crossings(profile: np.ndarray,
                               threshold: float) -> np.ndarray:
    """
    查找 profile 中所有穿过阈值的位置（亚像素精度）

    Args:
        profile: 一维强度信号
        threshold: 阈值

    Returns:
        穿通点索引数组（浮点索引，可插值）
    """
    crossings = []
    n = len(profile)

    for i in range(n - 1):
        v0 = profile[i]
        v1 = profile[i + 1]

        if (v0 - threshold) * (v1 - threshold) < 0:
            frac = (threshold - v0) / (v1 - v0)
            crossings.append(float(i) + frac)
        elif abs(v0 - threshold) < 1e-10 and (v1 - threshold) * (v0 - threshold) <= 0:
            crossings.append(float(i))

    return np.array(crossings, dtype=np.float64)


def cd_threshold_crossing(profile: np.ndarray,
                          positions: np.ndarray,
                          threshold: Optional[float] = None,
                          threshold_pct: float = 0.5) -> CDExtractionResult:
    """
    阈值穿通法 (Threshold Crossing) 提取 CD

    计算信号最大值与最小值之间的指定百分比阈值，
    找到阈值穿通点，相邻两个穿通点间距即为 CD。

    Args:
        profile: 一维强度分布
        positions: 位置坐标 (nm)
        threshold: 显式阈值，若为 None 则使用 threshold_pct 计算
        threshold_pct: 阈值百分比 (0~1)，默认 0.5 (50%阈值)

    Returns:
        CD 提取结果
    """
    if threshold is None:
        p_min = np.min(profile)
        p_max = np.max(profile)
        threshold = p_min + threshold_pct * (p_max - p_min)

    crossings_idx = _find_threshold_crossings(profile, threshold)

    if len(crossings_idx) < 2:
        return CDExtractionResult(
            cd_value=0.0,
            left_edge_pos=0.0,
            right_edge_pos=0.0,
            method=CDExtractionMethod.THRESHOLD_CROSSING.value,
            profile_intensity=profile,
            profile_positions=positions,
            edge_positions_raw=(0.0, 0.0),
            confidence=0.0,
        )

    left_idx = crossings_idx[0]
    right_idx = crossings_idx[-1]

    step = positions[1] - positions[0] if len(positions) > 1 else 1.0
    left_pos = left_idx * step + positions[0]
    right_pos = right_idx * step + positions[0]
    cd = right_pos - left_pos

    p_min = np.min(profile)
    p_max = np.max(profile)
    contrast = (p_max - p_min) / (p_max + p_min + 1e-10)
    confidence = min(1.0, contrast * 2.0)

    return CDExtractionResult(
        cd_value=float(cd),
        left_edge_pos=float(left_pos),
        right_edge_pos=float(right_pos),
        method=CDExtractionMethod.THRESHOLD_CROSSING.value,
        profile_intensity=profile,
        profile_positions=positions,
        edge_positions_raw=(float(left_idx), float(right_idx)),
        confidence=float(confidence),
    )


def cd_derivative_peak(profile: np.ndarray,
                       positions: np.ndarray,
                       smooth_sigma: float = 1.0) -> CDExtractionResult:
    """
    导数峰值法 (Derivative Peak) 提取 CD

    对强度分布求一阶导数，导数的极大值对应上升沿（左边缘），
    极小值对应下降沿（右边缘），两峰值位置间距即为 CD。

    Args:
        profile: 一维强度分布
        positions: 位置坐标 (nm)
        smooth_sigma: 高斯平滑 sigma (像素单位)，用于降噪

    Returns:
        CD 提取结果
    """
    if smooth_sigma > 0:
        profile_smooth = gaussian_filter1d(profile.astype(np.float64), sigma=smooth_sigma)
    else:
        profile_smooth = profile.astype(np.float64)

    derivative = np.gradient(profile_smooth)

    peak_idx = int(np.argmax(derivative))
    valley_idx = int(np.argmin(derivative))

    if peak_idx >= valley_idx:
        return CDExtractionResult(
            cd_value=0.0,
            left_edge_pos=0.0,
            right_edge_pos=0.0,
            method=CDExtractionMethod.DERIVATIVE_PEAK.value,
            profile_intensity=profile,
            profile_positions=positions,
            edge_positions_raw=(0.0, 0.0),
            confidence=0.0,
        )

    left_idx = float(peak_idx)
    right_idx = float(valley_idx)

    step = positions[1] - positions[0] if len(positions) > 1 else 1.0
    left_pos = left_idx * step + positions[0]
    right_pos = right_idx * step + positions[0]
    cd = right_pos - left_pos

    peak_amplitude = derivative[peak_idx] - derivative[valley_idx]
    noise_level = np.std(derivative[:max(3, peak_idx // 2)]) + 1e-10
    snr = peak_amplitude / (noise_level * 2 + 1e-10)
    confidence = min(1.0, snr / 5.0)

    return CDExtractionResult(
        cd_value=float(cd),
        left_edge_pos=float(left_pos),
        right_edge_pos=float(right_pos),
        method=CDExtractionMethod.DERIVATIVE_PEAK.value,
        profile_intensity=profile,
        profile_positions=positions,
        edge_positions_raw=(left_idx, right_idx),
        confidence=float(confidence),
    )


def cd_linear_regression(profile: np.ndarray,
                         positions: np.ndarray,
                         threshold_pct: float = 0.5,
                         fit_window: int = 5) -> CDExtractionResult:
    """
    线性回归法 (Linear Regression) 提取 CD

    对边缘过渡区附近的数据点进行线性拟合，
    以拟合线与阈值的交点作为亚像素精度的边缘位置。

    Args:
        profile: 一维强度分布
        positions: 位置坐标 (nm)
        threshold_pct: 阈值百分比 (0~1)
        fit_window: 拟合窗口半宽 (像素数)

    Returns:
        CD 提取结果
    """
    p_min = np.min(profile)
    p_max = np.max(profile)
    threshold = p_min + threshold_pct * (p_max - p_min)

    crossings_idx = _find_threshold_crossings(profile, threshold)

    if len(crossings_idx) < 2:
        return CDExtractionResult(
            cd_value=0.0,
            left_edge_pos=0.0,
            right_edge_pos=0.0,
            method=CDExtractionMethod.LINEAR_REGRESSION.value,
            profile_intensity=profile,
            profile_positions=positions,
            edge_positions_raw=(0.0, 0.0),
            confidence=0.0,
        )

    edge_positions = []
    r_squared_values = []

    for cross_idx in crossings_idx[:2]:
        center = int(round(cross_idx))
        i_start = max(0, center - fit_window)
        i_end = min(len(profile), center + fit_window + 1)

        x_local = np.arange(i_start, i_end, dtype=np.float64)
        y_local = profile[i_start:i_end].astype(np.float64)

        if len(x_local) < 3:
            edge_positions.append(cross_idx)
            r_squared_values.append(0.0)
            continue

        coeffs = np.polyfit(x_local, y_local, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        if abs(slope) < 1e-10:
            edge_positions.append(cross_idx)
            r_squared_values.append(0.0)
            continue

        edge_idx = (threshold - intercept) / slope

        y_pred = np.polyval(coeffs, x_local)
        ss_res = np.sum((y_local - y_pred) ** 2)
        ss_tot = np.sum((y_local - np.mean(y_local)) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-10)

        edge_positions.append(edge_idx)
        r_squared_values.append(r_squared)

    step = positions[1] - positions[0] if len(positions) > 1 else 1.0
    left_idx = edge_positions[0]
    right_idx = edge_positions[-1]
    left_pos = left_idx * step + positions[0]
    right_pos = right_idx * step + positions[0]
    cd = right_pos - left_pos

    confidence = float(np.mean(r_squared_values)) if r_squared_values else 0.0

    return CDExtractionResult(
        cd_value=float(cd),
        left_edge_pos=float(left_pos),
        right_edge_pos=float(right_pos),
        method=CDExtractionMethod.LINEAR_REGRESSION.value,
        profile_intensity=profile,
        profile_positions=positions,
        edge_positions_raw=(float(left_idx), float(right_idx)),
        confidence=max(0.0, min(1.0, confidence)),
    )


def cd_polynomial_fit(profile: np.ndarray,
                      positions: np.ndarray,
                      threshold_pct: float = 0.5,
                      poly_order: int = 3,
                      fit_window: int = 8) -> CDExtractionResult:
    """
    多项式拟合法 (Polynomial Fit) 提取 CD

    对边缘过渡区进行高阶多项式拟合，更精确地建模边缘的 S 型曲线，
    以拟合曲线与阈值的交点作为边缘位置。

    Args:
        profile: 一维强度分布
        positions: 位置坐标 (nm)
        threshold_pct: 阈值百分比 (0~1)
        poly_order: 多项式阶数 (默认 3 次)
        fit_window: 拟合窗口半宽 (像素数)

    Returns:
        CD 提取结果
    """
    p_min = np.min(profile)
    p_max = np.max(profile)
    threshold = p_min + threshold_pct * (p_max - p_min)

    crossings_idx = _find_threshold_crossings(profile, threshold)

    if len(crossings_idx) < 2:
        return CDExtractionResult(
            cd_value=0.0,
            left_edge_pos=0.0,
            right_edge_pos=0.0,
            method=CDExtractionMethod.POLYNOMIAL_FIT.value,
            profile_intensity=profile,
            profile_positions=positions,
            edge_positions_raw=(0.0, 0.0),
            confidence=0.0,
        )

    edge_positions = []
    r_squared_values = []

    for cross_idx in crossings_idx[:2]:
        center = int(round(cross_idx))
        i_start = max(0, center - fit_window)
        i_end = min(len(profile), center + fit_window + 1)

        x_local = np.arange(i_start, i_end, dtype=np.float64)
        y_local = profile[i_start:i_end].astype(np.float64)

        n_points = len(x_local)
        if n_points < poly_order + 2:
            edge_positions.append(cross_idx)
            r_squared_values.append(0.0)
            continue

        try:
            coeffs = np.polyfit(x_local, y_local, poly_order)

            roots = np.roots(coeffs - np.array([0.0] * poly_order + [threshold]))
            valid_roots = [r.real for r in roots
                           if abs(r.imag) < 1e-6
                           and i_start - 0.5 <= r.real <= i_end + 0.5]

            if valid_roots:
                edge_idx = min(valid_roots, key=lambda r: abs(r - cross_idx))
            else:
                edge_idx = cross_idx

            y_pred = np.polyval(coeffs, x_local)
            ss_res = np.sum((y_local - y_pred) ** 2)
            ss_tot = np.sum((y_local - np.mean(y_local)) ** 2)
            r_squared = 1.0 - ss_res / (ss_tot + 1e-10)

        except np.linalg.LinAlgError:
            edge_idx = cross_idx
            r_squared = 0.0

        edge_positions.append(edge_idx)
        r_squared_values.append(r_squared)

    step = positions[1] - positions[0] if len(positions) > 1 else 1.0
    left_idx = edge_positions[0]
    right_idx = edge_positions[-1]
    left_pos = left_idx * step + positions[0]
    right_pos = right_idx * step + positions[0]
    cd = right_pos - left_pos

    confidence = float(np.mean(r_squared_values)) if r_squared_values else 0.0

    return CDExtractionResult(
        cd_value=float(cd),
        left_edge_pos=float(left_pos),
        right_edge_pos=float(right_pos),
        method=CDExtractionMethod.POLYNOMIAL_FIT.value,
        profile_intensity=profile,
        profile_positions=positions,
        edge_positions_raw=(float(left_idx), float(right_idx)),
        confidence=max(0.0, min(1.0, confidence)),
    )


def extract_cd(image: np.ndarray,
               line: MeasurementLine,
               method: Union[str, CDExtractionMethod] = CDExtractionMethod.THRESHOLD_CROSSING,
               pixel_size: float = 1.0,
               **kwargs) -> CDExtractionResult:
    """
    通用 CD 提取接口

    对晶圆图像沿指定测量线提取 CD，支持多种算法。

    Args:
        image: 输入晶圆图像 (2D 灰度图)
        line: 测量线定义
        method: CD 提取方法
        pixel_size: 像素尺寸 (nm)
        **kwargs: 传递给具体算法的额外参数

    Returns:
        CD 提取结果
    """
    if isinstance(method, str):
        method = CDExtractionMethod(method)

    profile, positions = extract_profile(image, line, pixel_size)

    if method == CDExtractionMethod.THRESHOLD_CROSSING:
        return cd_threshold_crossing(profile, positions, **kwargs)
    elif method == CDExtractionMethod.DERIVATIVE_PEAK:
        return cd_derivative_peak(profile, positions, **kwargs)
    elif method == CDExtractionMethod.LINEAR_REGRESSION:
        return cd_linear_regression(profile, positions, **kwargs)
    elif method == CDExtractionMethod.POLYNOMIAL_FIT:
        return cd_polynomial_fit(profile, positions, **kwargs)
    else:
        raise ValueError(f"未知的 CD 提取方法: {method}")


def extract_cd_multiline(image: np.ndarray,
                         lines: List[MeasurementLine],
                         method: Union[str, CDExtractionMethod] = CDExtractionMethod.THRESHOLD_CROSSING,
                         pixel_size: float = 1.0,
                         **kwargs) -> List[CDExtractionResult]:
    """
    批量沿多条测量线提取 CD

    Args:
        image: 输入晶圆图像 (2D)
        lines: 测量线列表
        method: CD 提取方法
        pixel_size: 像素尺寸 (nm)
        **kwargs: 额外参数

    Returns:
        CD 提取结果列表，与输入 lines 一一对应
    """
    results = []
    for line in lines:
        result = extract_cd(image, line, method, pixel_size, **kwargs)
        results.append(result)
    return results
