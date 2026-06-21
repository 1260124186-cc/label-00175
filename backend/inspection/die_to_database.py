# -*- coding: utf-8 -*-
"""
Die-to-Database 差异图计算模块

实现待测掩模图像与数据库参考图像的差异计算，
支持自适应阈值、噪声抑制、候选缺陷区域提取等功能，
用于掩模检测机台的缺陷检测算法研究。

核心功能:
1. 图像配准对齐（亚像素精度）
2. 差异图计算（绝对/相对/归一化）
3. 自适应阈值分割
4. 候选缺陷区域提取与分析
5. 差异直方图统计
"""

import numpy as np
from numba import jit, prange
from typing import Optional, List, Tuple, Dict, Any
from scipy.ndimage import (
    gaussian_filter,
    median_filter,
    label,
    find_objects,
    center_of_mass,
)
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.optimize import minimize
import logging

from inspection.schemas import (
    DifferenceMapResult,
    InspectionImageResult,
    InspectionAnalysisConfig,
)

logger = logging.getLogger(__name__)


def _compute_cross_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
) -> np.ndarray:
    """
    计算两幅图像的互相关图

    Args:
        img1: 图像1
        img2: 图像2

    Returns:
        互相关图
    """
    fft1 = np.fft.fft2(img1)
    fft2 = np.fft.fft2(img2)
    cc = np.fft.ifft2(fft1 * np.conj(fft2))
    return np.abs(np.fft.fftshift(cc))


def _align_images(
    test_image: np.ndarray,
    reference_image: np.ndarray,
    max_shift_pixels: int = 5,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    对齐测试图像与参考图像

    使用互相关法估计亚像素级位移，
    并对测试图像进行平移校正。

    Args:
        test_image: 待测图像
        reference_image: 参考图像
        max_shift_pixels: 最大允许位移 (像素)

    Returns:
        (对齐后的测试图像, 估计的位移 (dy, dx))
    """
    cc = _compute_cross_correlation(test_image, reference_image)

    center_y, center_x = cc.shape[0] // 2, cc.shape[1] // 2

    y_min = max(0, center_y - max_shift_pixels)
    y_max = min(cc.shape[0], center_y + max_shift_pixels + 1)
    x_min = max(0, center_x - max_shift_pixels)
    x_max = min(cc.shape[1], center_x + max_shift_pixels + 1)

    cc_search = cc[y_min:y_max, x_min:x_max]
    peak_idx = np.unravel_index(np.argmax(cc_search), cc_search.shape)

    dy = peak_idx[0] + y_min - center_y
    dx = peak_idx[1] + x_min - center_x

    if dy == 0 and dx == 0:
        return test_image.copy(), (0, 0)

    aligned = np.zeros_like(test_image)
    src_y_start = max(0, -dy)
    src_y_end = min(test_image.shape[0], test_image.shape[0] - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(test_image.shape[1], test_image.shape[1] - dx)

    dst_y_start = max(0, dy)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, dx)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    aligned[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        test_image[src_y_start:src_y_end, src_x_start:src_x_end]

    return aligned, (dy, dx)


def compute_difference_map(
    test_image: np.ndarray,
    reference_image: np.ndarray,
    align: bool = True,
    normalize: bool = True,
    smooth_sigma: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Die-to-Database 差异图

    支持图像对齐、平滑去噪、归一化等预处理步骤。

    Args:
        test_image: 待测图像 (0~1)
        reference_image: 参考图像 (0~1)
        align: 是否进行图像配准对齐
        normalize: 是否进行局部归一化（减少光照不均影响）
        smooth_sigma: 平滑 sigma (像素)，0 表示不平滑

    Returns:
        (绝对差异图, 带符号差异图)
    """
    if test_image.shape != reference_image.shape:
        raise ValueError(
            f"图像尺寸不匹配: test={test_image.shape}, ref={reference_image.shape}"
        )

    test_img = test_image.astype(np.float64).copy()
    ref_img = reference_image.astype(np.float64).copy()

    if align:
        test_img, shift = _align_images(test_img, ref_img)
        logger.debug(f"图像配准位移: dy={shift[0]}, dx={shift[1]}")

    if normalize:
        test_norm = _local_normalization(test_img)
        ref_norm = _local_normalization(ref_img)
    else:
        test_norm = test_img
        ref_norm = ref_img

    signed_diff = test_norm - ref_norm

    if smooth_sigma > 0:
        signed_diff = gaussian_filter(signed_diff, sigma=smooth_sigma)

    abs_diff = np.abs(signed_diff)

    return abs_diff, signed_diff


def _local_normalization(image: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    局部归一化，减少光照不均的影响

    Args:
        image: 输入图像
        window_size: 局部窗口大小

    Returns:
        归一化后的图像
    """
    img = image.astype(np.float64)

    mean_local = gaussian_filter(img, sigma=window_size / 4.0)
    std_local = np.sqrt(gaussian_filter((img - mean_local) ** 2, sigma=window_size / 4.0))

    std_local = np.maximum(std_local, 1e-6)

    normalized = (img - mean_local) / std_local

    norm_min, norm_max = normalized.min(), normalized.max()
    if norm_max > norm_min:
        normalized = (normalized - norm_min) / (norm_max - norm_min)

    return normalized


def compute_detection_threshold(
    difference_map: np.ndarray,
    threshold_abs: float = 0.1,
    threshold_rel: float = 3.0,
    adaptive: bool = True,
) -> float:
    """
    计算缺陷检测阈值

    综合考虑绝对阈值和相对阈值（基于统计分布）。

    Args:
        difference_map: 差异图（绝对值）
        threshold_abs: 绝对阈值下限
        threshold_rel: 相对阈值（标准差倍数）
        adaptive: 是否使用自适应阈值

    Returns:
        最终使用的检测阈值
    """
    if not adaptive:
        return threshold_abs

    diff_flat = difference_map.flatten()

    q1 = np.percentile(diff_flat, 25)
    q3 = np.percentile(diff_flat, 75)
    iqr = q3 - q1

    robust_std = iqr / 1.349
    median_val = np.median(diff_flat)

    rel_threshold = median_val + threshold_rel * robust_std

    final_threshold = max(threshold_abs, rel_threshold)

    logger.debug(
        f"阈值计算: 绝对={threshold_abs:.4f}, "
        f"相对={rel_threshold:.4f}, "
        f"最终={final_threshold:.4f}"
    )

    return float(final_threshold)


def threshold_difference_map(
    difference_map: np.ndarray,
    threshold: float,
    min_area_pixels: int = 3,
    max_area_pixels: Optional[int] = None,
    connectivity: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对差异图进行阈值分割，提取候选缺陷区域

    Args:
        difference_map: 差异图（绝对值）
        threshold: 检测阈值
        min_area_pixels: 最小缺陷面积 (像素)
        max_area_pixels: 最大缺陷面积 (像素)，None 表示不限制
        connectivity: 连通域分析连接性 (4 或 8)

    Returns:
        (二值化掩模, 标记图)
    """
    binary_map = difference_map > threshold

    structure = np.ones((3, 3)) if connectivity == 8 else np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])

    labeled, num_features = label(binary_map.astype(np.int32), structure=structure)

    if num_features > 0 and (min_area_pixels > 1 or max_area_pixels is not None):
        for label_idx in range(1, num_features + 1):
            region_mask = labeled == label_idx
            area = int(np.sum(region_mask))

            remove = False
            if min_area_pixels > 1 and area < min_area_pixels:
                remove = True
            if max_area_pixels is not None and area > max_area_pixels:
                remove = True

            if remove:
                binary_map[region_mask] = False
                labeled[region_mask] = 0

        labeled, _ = label(binary_map.astype(np.int32), structure=structure)

    return binary_map.astype(np.uint8), labeled


def _analyze_region(
    region_mask: np.ndarray,
    difference_map: np.ndarray,
    signed_difference: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pixel_size_nm: float = 1.0,
) -> Dict[str, Any]:
    """
    分析单个候选缺陷区域

    Args:
        region_mask: 区域二值掩模
        difference_map: 绝对差异图
        signed_difference: 带符号差异图
        bbox: 边界框 (y1, y2, x1, x2)
        pixel_size_nm: 像素尺寸 (nm/pixel)

    Returns:
        区域特征字典
    """
    y1, y2, x1, x2 = bbox
    region_diff = difference_map[y1:y2, x1:x2][region_mask]
    region_signed = signed_difference[y1:y2, x1:x2][region_mask]

    area_pixels = int(np.sum(region_mask))
    area_nm2 = area_pixels * pixel_size_nm * pixel_size_nm
    equivalent_diameter_nm = 2.0 * np.sqrt(area_nm2 / np.pi)

    if len(region_diff) == 0:
        mean_diff = 0.0
        max_diff = 0.0
        std_diff = 0.0
        mean_signed = 0.0
        polarity = "positive"
    else:
        mean_diff = float(np.mean(region_diff))
        max_diff = float(np.max(region_diff))
        std_diff = float(np.std(region_diff))
        mean_signed = float(np.mean(region_signed))
        polarity = "positive" if mean_signed > 0 else "negative"

    cy_full, cx_full = center_of_mass(region_mask.astype(np.float64))

    perimeter = _compute_perimeter(region_mask)
    if perimeter > 0:
        circularity = 4 * np.pi * area_pixels / (perimeter * perimeter)
    else:
        circularity = 1.0

    aspect_ratio = max(y2 - y1, x2 - x1) / max(min(y2 - y1, x2 - x1), 1)

    intensity_score = min(mean_diff / max(difference_map.max(), 1e-6), 1.0)
    size_score = min(area_pixels / 50.0, 1.0)
    confidence = 0.6 * intensity_score + 0.4 * size_score

    return {
        'center_y': float(cy_full),
        'center_x': float(cx_full),
        'bbox': [int(y1), int(y2), int(x1), int(x2)],
        'area_pixels': area_pixels,
        'area_nm2': float(area_nm2),
        'size_nm': float(equivalent_diameter_nm),
        'mean_difference': mean_diff,
        'max_difference': max_diff,
        'std_difference': std_diff,
        'mean_signed': mean_signed,
        'polarity': polarity,
        'perimeter_pixels': int(perimeter),
        'circularity': float(circularity),
        'aspect_ratio': float(aspect_ratio),
        'confidence': float(confidence),
    }


def _compute_perimeter(binary_mask: np.ndarray) -> int:
    """
    计算二值区域的周长（像素数）

    Args:
        binary_mask: 二值掩模

    Returns:
        周长（像素数）
    """
    eroded = binary_erosion(binary_mask)
    boundary = binary_mask & (~eroded)
    return int(np.sum(boundary))


def extract_candidate_regions(
    thresholded_map: np.ndarray,
    labeled_map: np.ndarray,
    difference_map: np.ndarray,
    signed_difference: np.ndarray,
    pixel_size_nm: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    从阈值分割结果中提取候选缺陷区域

    Args:
        thresholded_map: 二值化掩模
        labeled_map: 连通域标记图
        difference_map: 绝对差异图
        signed_difference: 带符号差异图
        pixel_size_nm: 像素尺寸 (nm/pixel)

    Returns:
        候选区域列表，每个区域为特征字典
    """
    regions = []
    num_regions = labeled_map.max()

    if num_regions == 0:
        return regions

    objects = find_objects(labeled_map)

    for idx, bbox in enumerate(objects, start=1):
        if bbox is None:
            continue

        y1, y2 = bbox[0].start, bbox[0].stop
        x1, x2 = bbox[1].start, bbox[1].stop

        region_mask = labeled_map[y1:y2, x1:x2] == idx

        if not np.any(region_mask):
            continue

        region_info = _analyze_region(
            region_mask,
            difference_map,
            signed_difference,
            (y1, y2, x1, x2),
            pixel_size_nm,
        )
        region_info['region_id'] = idx
        regions.append(region_info)

    regions.sort(key=lambda r: r['max_difference'], reverse=True)

    return regions


def compute_difference_histogram(
    difference_map: np.ndarray,
    num_bins: int = 100,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    计算差异图的直方图统计

    Args:
        difference_map: 差异图
        num_bins: 直方图 bins 数量
        range_min: 直方图最小值，None 则自动计算
        range_max: 直方图最大值，None 则自动计算

    Returns:
        字典，包含 counts, bin_edges, bin_centers
    """
    diff_flat = difference_map.flatten()

    if range_min is None:
        range_min = diff_flat.min()
    if range_max is None:
        range_max = diff_flat.max()

    counts, bin_edges = np.histogram(
        diff_flat,
        bins=num_bins,
        range=(range_min, range_max),
        density=True,
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return {
        'counts': counts,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
    }


def compute_die_to_database(
    test_image: np.ndarray,
    reference_image: np.ndarray,
    config: Optional[InspectionAnalysisConfig] = None,
    pixel_size_nm: float = 1.0,
    align: bool = True,
) -> DifferenceMapResult:
    """
    完整的 Die-to-Database 差异图计算

    从图像对齐、差异计算、阈值分割到候选区域提取的完整流程。

    Args:
        test_image: 待测检测图像 (0~1)
        reference_image: 参考检测图像 (0~1)
        config: 分析配置，None 则使用默认配置
        pixel_size_nm: 像素尺寸 (nm/pixel)
        align: 是否进行图像配准对齐

    Returns:
        DifferenceMapResult，包含差异图及分析结果

    使用示例::

        result = compute_die_to_database(test_img, ref_img)
        print(f"检测到 {len(result.candidate_regions)} 个候选缺陷区域")
        print(f"最大差异: {result.max_difference:.4f}")
    """
    if config is None:
        config = InspectionAnalysisConfig()

    difference_map, signed_difference = compute_difference_map(
        test_image,
        reference_image,
        align=align,
        normalize=True,
        smooth_sigma=0.5,
    )

    threshold = compute_detection_threshold(
        difference_map,
        threshold_abs=config.diff_threshold_abs,
        threshold_rel=config.diff_threshold_rel,
        adaptive=config.inspection_config.adaptive_threshold,
    )

    thresholded_map, labeled_map = threshold_difference_map(
        difference_map,
        threshold,
        min_area_pixels=config.min_area_pixels,
        max_area_pixels=config.max_area_pixels,
        connectivity=config.connectivity,
    )

    candidate_regions = extract_candidate_regions(
        thresholded_map,
        labeled_map,
        difference_map,
        signed_difference,
        pixel_size_nm=pixel_size_nm,
    )

    histogram = compute_difference_histogram(difference_map)

    mean_diff = float(np.mean(difference_map))
    max_diff = float(np.max(difference_map))
    std_diff = float(np.std(difference_map))

    return DifferenceMapResult(
        difference_map=difference_map,
        signed_difference=signed_difference,
        thresholded_map=thresholded_map,
        threshold_used=threshold,
        candidate_regions=candidate_regions,
        difference_histogram=histogram,
        mean_difference=mean_diff,
        max_difference=max_diff,
        std_difference=std_diff,
    )


def compute_die_to_database_from_result(
    inspection_result: InspectionImageResult,
    config: Optional[InspectionAnalysisConfig] = None,
    align: bool = True,
) -> DifferenceMapResult:
    """
    从检测图像仿真结果直接计算 Die-to-Database 差异图

    Args:
        inspection_result: 检测图像仿真结果
        config: 分析配置
        align: 是否进行图像配准对齐

    Returns:
        DifferenceMapResult
    """
    pixel_size_nm = inspection_result.config.optics.pixel_size_nm

    return compute_die_to_database(
        inspection_result.inspection_image,
        inspection_result.reference_image,
        config=config,
        pixel_size_nm=pixel_size_nm,
        align=align,
    )


def compute_aligned_difference(
    test_img: np.ndarray,
    ref_img: np.ndarray,
    method: str = 'absolute',
) -> np.ndarray:
    """
    计算对齐后的差异图（多种差异度量）

    Args:
        test_img: 待测图像
        ref_img: 参考图像
        method: 差异计算方法:
            - 'absolute': 绝对差异 |I1 - I2|
            - 'squared': 平方差异 (I1 - I2)²
            - 'relative': 相对差异 |I1 - I2| / (|I1| + |I2| + eps)
            - 'normalized': 归一化互相关差异 1 - NCC

    Returns:
        差异图
    """
    eps = 1e-10

    if method == 'absolute':
        return np.abs(test_img - ref_img)

    elif method == 'squared':
        return (test_img - ref_img) ** 2

    elif method == 'relative':
        return np.abs(test_img - ref_img) / (np.abs(test_img) + np.abs(ref_img) + eps)

    elif method == 'normalized':
        t = test_img - test_img.mean()
        r = ref_img - ref_img.mean()
        ncc = np.sum(t * r) / (np.sqrt(np.sum(t**2)) * np.sqrt(np.sum(r**2)) + eps)
        return np.ones_like(test_img) * (1.0 - ncc)

    else:
        raise ValueError(f"未知的差异计算方法: {method}")
