# -*- coding: utf-8 -*-
"""
光刻专用指标模块：基于二值化晶圆图与目标图边缘提取的工艺评估指标

该模块实现了以下光刻专用指标：
1. EPE (Edge Placement Error) - 边缘放置误差
2. CD (Critical Dimension) Error - 关键尺寸误差
3. ILS (Image Log Slope) - 对数像斜率
4. NILS (Normalized Image Log Slope) - 归一化对数像斜率
5. PW (Process Window) Area - 工艺窗口面积
6. MEEF (Mask Error Enhancement Factor) - 掩模误差增强因子
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
from scipy.signal import convolve2d


def extract_edges(binary_image: np.ndarray,
                  method: str = 'morphological') -> np.ndarray:
    """
    从二值图像中提取边缘

    Args:
        binary_image: 二值图像（0或1）
        method: 边缘提取方法
            - 'morphological': 形态学边缘检测（原图 - 腐蚀图）
            - 'sobel': Sobel梯度边缘检测

    Returns:
        边缘图（1表示边缘，0表示非边缘）
    """
    img = binary_image.astype(np.float64)
    img_bin = img >= 0.5

    if method == 'morphological':
        struct = np.ones((3, 3), dtype=bool)
        eroded = binary_erosion(img_bin, structure=struct)
        edges = img_bin & ~eroded
        return edges.astype(np.float64)
    elif method == 'sobel':
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        grad_y = convolve2d(img, sobel_y, mode='same', boundary='symm')
        grad_x = convolve2d(img, sobel_x, mode='same', boundary='symm')
        grad_mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
        grad_max = grad_mag.max()
        if grad_max > 0:
            grad_mag = grad_mag / grad_max
        edges = (grad_mag >= 0.3).astype(np.float64)
        return edges
    else:
        raise ValueError(f"未知的边缘提取方法: {method}")


def compute_epe(wafer_binary: np.ndarray,
                target_binary: np.ndarray,
                pixel_size: float = 1.0,
                edge_method: str = 'morphological') -> Dict[str, float]:
    """
    计算边缘放置误差 (Edge Placement Error, EPE)

    对二值化后的晶圆图与目标图做边缘提取后，
    计算两组边缘之间的平均最小距离。

    EPE = mean( min_distance(wafer_edge_pixel, target_edge) )
        + mean( min_distance(target_edge_pixel, wafer_edge) )

    还返回 EPE 的最大值和标准差。

    Args:
        wafer_binary: 二值化晶圆图像（0或1）
        target_binary: 二值化目标图像（0或1）
        pixel_size: 像素尺寸（nm），用于将像素距离转换为物理距离
        edge_method: 边缘提取方法

    Returns:
        字典，包含:
            - 'epe_mean': 平均EPE (nm)
            - 'epe_max': 最大EPE (nm)
            - 'epe_std': EPE标准差 (nm)
            - 'epe_median': EPE中位数 (nm)
    """
    wafer_edge = extract_edges(wafer_binary, method=edge_method)
    target_edge = extract_edges(target_binary, method=edge_method)

    if np.sum(target_edge) == 0 and np.sum(wafer_edge) == 0:
        return {'epe_mean': 0.0, 'epe_max': 0.0, 'epe_std': 0.0, 'epe_median': 0.0}

    dist_to_wafer = distance_transform_edt(1.0 - wafer_edge)
    dist_to_target = distance_transform_edt(1.0 - target_edge)

    wafer_edge_distances = dist_to_target[wafer_edge > 0.5]
    target_edge_distances = dist_to_wafer[target_edge > 0.5]

    all_distances = np.concatenate([wafer_edge_distances, target_edge_distances])

    result = {
        'epe_mean': float(np.mean(all_distances) * pixel_size),
        'epe_max': float(np.max(all_distances) * pixel_size),
        'epe_std': float(np.std(all_distances) * pixel_size),
        'epe_median': float(np.median(all_distances) * pixel_size),
    }
    return result


def _measure_cd_along_scanline(scanline: np.ndarray,
                                threshold: float = 0.5) -> List[float]:
    """
    沿一条扫描线测量所有关键尺寸

    找到所有 0→1 和 1→0 的跳变点，相邻一对构成一个特征的 CD。

    Args:
        scanline: 一维信号（已二值化或连续值）
        threshold: 连续信号的二值化阈值

    Returns:
        CD 列表（像素单位）
    """
    binary = (scanline >= threshold).astype(np.int32)
    n = len(binary)

    transitions_up = []
    transitions_down = []

    for i in range(1, n):
        if binary[i] == 1 and binary[i - 1] == 0:
            transitions_up.append(i)
        elif binary[i] == 0 and binary[i - 1] == 1:
            transitions_down.append(i)

    if binary[0] == 1:
        transitions_up.insert(0, 0)
    if binary[-1] == 1:
        transitions_down.append(n)

    cds = []
    n_pairs = min(len(transitions_up), len(transitions_down))
    for k in range(n_pairs):
        cd = transitions_down[k] - transitions_up[k]
        if cd > 0:
            cds.append(float(cd))
    return cds


def compute_cd(binary_image: np.ndarray,
               direction: str = 'both',
               pixel_size: float = 1.0,
               threshold: float = 0.5) -> Dict[str, float]:
    """
    测量二值图像的关键尺寸 (Critical Dimension)

    沿水平和/或垂直扫描线测量所有特征的宽度（CD），
    返回统计量。

    Args:
        binary_image: 二值化图像
        direction: 扫描方向
            - 'horizontal': 仅水平扫描
            - 'vertical': 仅垂直扫描
            - 'both': 两个方向
        pixel_size: 像素尺寸 (nm)
        threshold: 二值化阈值

    Returns:
        字典，包含:
            - 'cd_mean': 平均CD (nm)
            - 'cd_min': 最小CD (nm)
            - 'cd_max': 最大CD (nm)
            - 'cd_std': CD标准差 (nm)
            - 'n_features': 检测到的特征数
    """
    img = binary_image.astype(np.float64)
    all_cds = []

    if direction in ('horizontal', 'both'):
        for row in range(img.shape[0]):
            cds = _measure_cd_along_scanline(img[row, :], threshold)
            all_cds.extend(cds)

    if direction in ('vertical', 'both'):
        for col in range(img.shape[1]):
            cds = _measure_cd_along_scanline(img[:, col], threshold)
            all_cds.extend(cds)

    all_cds = np.array(all_cds) * pixel_size

    if len(all_cds) == 0:
        return {
            'cd_mean': 0.0,
            'cd_min': 0.0,
            'cd_max': 0.0,
            'cd_std': 0.0,
            'n_features': 0,
        }

    return {
        'cd_mean': float(np.mean(all_cds)),
        'cd_min': float(np.min(all_cds)),
        'cd_max': float(np.max(all_cds)),
        'cd_std': float(np.std(all_cds)),
        'n_features': len(all_cds),
    }


def compute_cd_error(wafer_binary: np.ndarray,
                     target_binary: np.ndarray,
                     direction: str = 'both',
                     pixel_size: float = 1.0) -> Dict[str, float]:
    """
    计算关键尺寸误差 (CD Error)

    分别测量晶圆图和目标图的 CD，然后计算差异。

    CD_error = CD_wafer - CD_target

    Args:
        wafer_binary: 二值化晶圆图像
        target_binary: 二值化目标图像
        direction: 扫描方向
        pixel_size: 像素尺寸 (nm)

    Returns:
        字典，包含:
            - 'cd_error_mean': 平均CD误差 (nm)
            - 'cd_error_relative': 相对CD误差 (%)
            - 'cd_wafer_mean': 晶圆图平均CD (nm)
            - 'cd_target_mean': 目标图平均CD (nm)
            - 'cd_wafer': 晶圆图CD统计
            - 'cd_target': 目标图CD统计
    """
    cd_wafer = compute_cd(wafer_binary, direction=direction, pixel_size=pixel_size)
    cd_target = compute_cd(target_binary, direction=direction, pixel_size=pixel_size)

    cd_wafer_mean = cd_wafer['cd_mean']
    cd_target_mean = cd_target['cd_mean']

    if cd_target_mean > 1e-10:
        cd_error_relative = (cd_wafer_mean - cd_target_mean) / cd_target_mean * 100.0
    else:
        cd_error_relative = 0.0

    return {
        'cd_error_mean': float(cd_wafer_mean - cd_target_mean),
        'cd_error_relative': float(cd_error_relative),
        'cd_wafer_mean': cd_wafer_mean,
        'cd_target_mean': cd_target_mean,
        'cd_wafer': cd_wafer,
        'cd_target': cd_target,
    }


def compute_ils(aerial_image: np.ndarray,
                threshold: float = 0.3,
                pixel_size: float = 1.0) -> Dict[str, float]:
    """
    计算对数像斜率 (Image Log Slope, ILS)

    ILS = |dI/dn| / I

    其中 dI/dn 是沿边缘法线方向的梯度，I 是该点的光强。
    ILS 衡量像在边缘处的陡峭程度，越大表示边缘越锐利。

    在阈值轮廓附近采样计算。

    Args:
        aerial_image: 空间像（连续光强分布，归一化到 [0, 1]）
        threshold: 阈值（在接近此阈值的轮廓处计算 ILS）
        pixel_size: 像素尺寸 (nm)

    Returns:
        字典，包含:
            - 'ils_mean': 平均ILS (1/nm)
            - 'ils_min': 最小ILS (1/nm)
            - 'ils_max': 最大ILS (1/nm)
            - 'ils_std': ILS标准差 (1/nm)
            - 'n_sample_points': 采样点数
    """
    img = aerial_image.astype(np.float64)
    ny, nx = img.shape

    grad_y = np.zeros_like(img)
    grad_x = np.zeros_like(img)
    grad_y[:-1, :] = np.diff(img, axis=0)
    grad_x[:, :-1] = np.diff(img, axis=1)

    grad_mag = np.sqrt(grad_y ** 2 + grad_x ** 2 + 1e-30)

    threshold_band = np.abs(img - threshold) < 0.1
    edge_region = threshold_band & (img > 1e-10)

    ils_map = grad_mag / (img + 1e-30)

    ils_values = ils_map[edge_region]

    if len(ils_values) == 0:
        return {
            'ils_mean': 0.0,
            'ils_min': 0.0,
            'ils_max': 0.0,
            'ils_std': 0.0,
            'n_sample_points': 0,
        }

    ils_values_physical = ils_values / pixel_size

    return {
        'ils_mean': float(np.mean(ils_values_physical)),
        'ils_min': float(np.min(ils_values_physical)),
        'ils_max': float(np.max(ils_values_physical)),
        'ils_std': float(np.std(ils_values_physical)),
        'n_sample_points': int(len(ils_values)),
    }


def compute_nils(aerial_image: np.ndarray,
                 cd_target: float,
                 threshold: float = 0.3,
                 pixel_size: float = 1.0) -> Dict[str, float]:
    """
    计算归一化对数像斜率 (Normalized Image Log Slope, NILS)

    NILS = ILS * CD_target = (|dI/dn| / I) * CD_target

    NILS 是量纲为一的指标，消除了特征尺寸的影响。
    NILS > 2 通常被认为是可接受的光刻质量阈值。

    Args:
        aerial_image: 空间像（连续光强分布）
        cd_target: 目标关键尺寸 (nm)
        threshold: 阈值
        pixel_size: 像素尺寸 (nm)

    Returns:
        字典，包含:
            - 'nils_mean': 平均NILS (无量纲)
            - 'nils_min': 最小NILS
            - 'nils_max': 最大NILS
            - 'nils_std': NILS标准差
            - 'n_sample_points': 采样点数
            - 'ils': 原始ILS统计
    """
    ils_result = compute_ils(aerial_image, threshold=threshold, pixel_size=pixel_size)

    nils_mean = ils_result['ils_mean'] * cd_target
    nils_min = ils_result['ils_min'] * cd_target
    nils_max = ils_result['ils_max'] * cd_target
    nils_std = ils_result['ils_std'] * cd_target

    return {
        'nils_mean': float(nils_mean),
        'nils_min': float(nils_min),
        'nils_max': float(nils_max),
        'nils_std': float(nils_std),
        'n_sample_points': ils_result['n_sample_points'],
        'ils': ils_result,
    }


def compute_process_window_area(
        conditions: List,
        cd_values: np.ndarray,
        cd_target: float,
        cd_tolerance: float = 0.1,
        focus_key: str = 'defocus',
        dose_key: str = 'dose') -> Dict[str, float]:
    """
    计算工艺窗口 (Process Window) 面积

    在 focus-dose 空间中，找到 CD 满足规格
    (|CD - CD_target| <= cd_tolerance * CD_target) 的所有工艺条件，
    计算该区域的面积。

    Args:
        conditions: ProcessCondition 列表（来自 imaging.py）
        cd_values: 每个工艺条件对应的 CD 值数组 (nm)，长度与 conditions 相同
        cd_target: 目标CD (nm)
        cd_tolerance: CD容差（相对值），默认10%
        focus_key: 离焦量属性名
        dose_key: 剂量属性名

    Returns:
        字典，包含:
            - 'pw_area': 工艺窗口面积 (nm * 相对剂量)
            - 'pw_ratio': 工艺窗口面积占扫描总面积的比例
            - 'n_passing': 通过条件的数量
            - 'n_total': 总条件数量
            - 'focus_range': 通过条件的离焦范围 (nm)
            - 'dose_range': 通过条件的剂量范围
    """
    n_total = len(conditions)
    if n_total == 0:
        return {
            'pw_area': 0.0, 'pw_ratio': 0.0,
            'n_passing': 0, 'n_total': 0,
            'focus_range': (0.0, 0.0), 'dose_range': (0.0, 0.0),
        }

    cd_values = np.asarray(cd_values, dtype=np.float64)
    cd_lower = cd_target * (1.0 - cd_tolerance)
    cd_upper = cd_target * (1.0 + cd_tolerance)

    passing = (cd_values >= cd_lower) & (cd_values <= cd_upper)
    n_passing = int(np.sum(passing))

    focus_values = np.array([getattr(c, focus_key) for c in conditions], dtype=np.float64)
    dose_values = np.array([getattr(c, dose_key) for c in conditions], dtype=np.float64)

    total_focus_span = float(np.max(focus_values) - np.min(focus_values)) if n_total > 1 else 0.0
    total_dose_span = float(np.max(dose_values) - np.min(dose_values)) if n_total > 1 else 0.0
    total_area = total_focus_span * total_dose_span

    if n_passing == 0:
        return {
            'pw_area': 0.0, 'pw_ratio': 0.0,
            'n_passing': 0, 'n_total': n_total,
            'focus_range': (0.0, 0.0), 'dose_range': (0.0, 0.0),
        }

    passing_focus = focus_values[passing]
    passing_dose = dose_values[passing]

    pw_focus_span = float(np.max(passing_focus) - np.min(passing_focus))
    pw_dose_span = float(np.max(passing_dose) - np.min(passing_dose))

    pw_area = pw_focus_span * pw_dose_span
    pw_ratio = pw_area / total_area if total_area > 0 else 0.0

    return {
        'pw_area': float(pw_area),
        'pw_ratio': float(pw_ratio),
        'n_passing': n_passing,
        'n_total': n_total,
        'focus_range': (float(np.min(passing_focus)), float(np.max(passing_focus))),
        'dose_range': (float(np.min(passing_dose)), float(np.max(passing_dose))),
    }


def compute_meef(mask: np.ndarray,
                 optical_system,
                 threshold: float = 0.3,
                 delta: float = 0.02,
                 direction: str = 'both',
                 pixel_size: float = 1.0,
                 resist_model=None) -> Dict[str, float]:
    """
    计算掩模误差增强因子 (Mask Error Enhancement Factor, MEEF)

    MEEF = ΔCD_wafer / ΔCD_mask

    通过对掩模做微小扰动（膨胀/腐蚀），分别计算扰动前后
    晶圆图的 CD 和掩模的 CD，然后求比值。

    Args:
        mask: 掩模图案（2D数组，值范围 [0, 1]）
        optical_system: 光学系统参数（OpticalSystem实例）
        threshold: 光刻胶阈值
        delta: 掩模扰动量（像素膨胀/腐蚀次数），或浮点比例
        direction: CD测量方向
        pixel_size: 像素尺寸 (nm)
        resist_model: 光刻胶模型

    Returns:
        字典，包含:
            - 'meef': MEEF值（量纲一）
            - 'cd_wafer_original': 原始晶圆CD (nm)
            - 'cd_wafer_perturbed': 扰动后晶圆CD (nm)
            - 'cd_mask_original': 原始掩模CD (nm)
            - 'cd_mask_perturbed': 扰动后掩模CD (nm)
            - 'delta_cd_wafer': 晶圆CD变化量 (nm)
            - 'delta_cd_mask': 掩模CD变化量 (nm)
    """
    from core.imaging import PartialCoherentImaging, apply_resist_model, _apply_threshold

    mask_float = mask.astype(np.float64)
    mask_bin = (mask_float >= 0.5).astype(np.float64)

    struct = np.ones((3, 3), dtype=bool)
    mask_perturbed_bin = binary_dilation(mask_bin, structure=struct).astype(np.float64)

    imaging_model = PartialCoherentImaging(optical_system, mask.shape)

    aerial_original = imaging_model.compute_aerial_image(mask_float)
    aerial_perturbed = imaging_model.compute_aerial_image(mask_perturbed_bin)

    if resist_model is not None:
        wafer_original = apply_resist_model(aerial_original, resist_model=resist_model)
        wafer_perturbed = apply_resist_model(aerial_perturbed, resist_model=resist_model)
    else:
        wafer_original = _apply_threshold(aerial_original, threshold)
        wafer_perturbed = _apply_threshold(aerial_perturbed, threshold)

    cd_mask_orig = compute_cd(mask_bin, direction=direction, pixel_size=pixel_size)
    cd_mask_pert = compute_cd(mask_perturbed_bin, direction=direction, pixel_size=pixel_size)
    cd_wafer_orig = compute_cd(wafer_original, direction=direction, pixel_size=pixel_size)
    cd_wafer_pert = compute_cd(wafer_perturbed, direction=direction, pixel_size=pixel_size)

    delta_cd_mask = cd_mask_pert['cd_mean'] - cd_mask_orig['cd_mean']
    delta_cd_wafer = cd_wafer_pert['cd_mean'] - cd_wafer_orig['cd_mean']

    if abs(delta_cd_mask) > 1e-10:
        meef = delta_cd_wafer / delta_cd_mask
    else:
        meef = 0.0

    return {
        'meef': float(meef),
        'cd_wafer_original': cd_wafer_orig['cd_mean'],
        'cd_wafer_perturbed': cd_wafer_pert['cd_mean'],
        'cd_mask_original': cd_mask_orig['cd_mean'],
        'cd_mask_perturbed': cd_mask_pert['cd_mean'],
        'delta_cd_wafer': float(delta_cd_wafer),
        'delta_cd_mask': float(delta_cd_mask),
    }


def compute_meef_simple(mask: np.ndarray,
                        threshold: float = 0.3,
                        delta: float = 0.02,
                        direction: str = 'both',
                        pixel_size: float = 1.0) -> Dict[str, float]:
    """
    简化版 MEEF 计算（不需要光学系统，仅在掩模空间估算）

    通过掩模二值图的膨胀/腐蚀估算 MEEF。
    适用于快速评估，不涉及光学成像仿真。

    MEEF_simple ≈ CD_wafer_perturbed / CD_mask_perturbed

    此处用二值化掩模直接作为"晶圆图"的近似。

    Args:
        mask: 掩模图案
        threshold: 二值化阈值
        delta: 扰动强度（形态学膨胀/腐蚀的像素数）
        direction: CD测量方向
        pixel_size: 像素尺寸 (nm)

    Returns:
        MEEF 字典
    """
    mask_bin = (mask.astype(np.float64) >= threshold).astype(np.float64)
    struct = np.ones((3, 3), dtype=bool)

    mask_dilated = binary_dilation(mask_bin, structure=struct).astype(np.float64)
    mask_eroded = binary_erosion(mask_bin, structure=struct).astype(np.float64)

    cd_nominal = compute_cd(mask_bin, direction=direction, pixel_size=pixel_size)
    cd_dilated = compute_cd(mask_dilated, direction=direction, pixel_size=pixel_size)
    cd_eroded = compute_cd(mask_eroded, direction=direction, pixel_size=pixel_size)

    cd_nom = cd_nominal['cd_mean']
    cd_dil = cd_dilated['cd_mean']
    cd_ero = cd_eroded['cd_mean']

    delta_mask = cd_dil - cd_ero
    delta_wafer = cd_dil - cd_ero

    if abs(delta_mask) > 1e-10:
        meef = delta_wafer / delta_mask
    else:
        meef = 0.0

    return {
        'meef': float(meef),
        'cd_nominal': cd_nom,
        'cd_dilated': cd_dil,
        'cd_eroded': cd_ero,
        'delta_cd': float(delta_wafer),
    }


@dataclass
class LithoMetricsResult:
    """光刻专用指标评估结果"""
    epe: Dict[str, float] = field(default_factory=dict)
    cd_error: Dict[str, float] = field(default_factory=dict)
    ils: Dict[str, float] = field(default_factory=dict)
    nils: Dict[str, float] = field(default_factory=dict)
    pw_area: Dict[str, float] = field(default_factory=dict)
    meef: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """转换为嵌套字典"""
        return {
            'epe': self.epe,
            'cd_error': self.cd_error,
            'ils': self.ils,
            'nils': self.nils,
            'pw_area': self.pw_area,
            'meef': self.meef,
        }

    def summary(self) -> str:
        """生成可读的摘要字符串"""
        lines = ["=== 光刻专用指标 ==="]
        if self.epe:
            lines.append(f"EPE mean: {self.epe.get('epe_mean', 0):.2f} nm")
        if self.cd_error:
            lines.append(f"CD error: {self.cd_error.get('cd_error_mean', 0):.2f} nm "
                         f"({self.cd_error.get('cd_error_relative', 0):.1f}%)")
        if self.ils:
            lines.append(f"ILS mean: {self.ils.get('ils_mean', 0):.4f} 1/nm")
        if self.nils:
            lines.append(f"NILS mean: {self.nils.get('nils_mean', 0):.2f}")
        if self.pw_area:
            lines.append(f"PW area: {self.pw_area.get('pw_area', 0):.1f} nm·dose "
                         f"({self.pw_area.get('pw_ratio', 0) * 100:.1f}%)")
        if self.meef:
            lines.append(f"MEEF: {self.meef.get('meef', 0):.2f}")
        return "\n".join(lines)


def evaluate_litho_metrics(
        wafer_binary: np.ndarray,
        target_binary: np.ndarray,
        aerial_image: Optional[np.ndarray] = None,
        cd_target: Optional[float] = None,
        pixel_size: float = 1.0,
        threshold: float = 0.3,
        edge_method: str = 'morphological',
        cd_direction: str = 'both') -> LithoMetricsResult:
    """
    一次性评估所有光刻专用指标

    Args:
        wafer_binary: 二值化晶圆图像
        target_binary: 二值化目标图像
        aerial_image: 空间像（用于计算 ILS/NILS），可选
        cd_target: 目标CD (nm)，可选；None则从目标图自动测量
        pixel_size: 像素尺寸 (nm)
        threshold: 光刻胶阈值
        edge_method: 边缘提取方法
        cd_direction: CD测量方向

    Returns:
        LithoMetricsResult 对象
    """
    epe = compute_epe(wafer_binary, target_binary,
                      pixel_size=pixel_size, edge_method=edge_method)

    cd_err = compute_cd_error(wafer_binary, target_binary,
                              direction=cd_direction, pixel_size=pixel_size)

    ils_result = {}
    nils_result = {}
    if aerial_image is not None:
        ils_result = compute_ils(aerial_image, threshold=threshold,
                                pixel_size=pixel_size)

        if cd_target is None:
            cd_target_measured = cd_err.get('cd_target_mean', 0.0)
        else:
            cd_target_measured = cd_target

        if cd_target_measured > 0:
            nils_result = compute_nils(aerial_image, cd_target=cd_target_measured,
                                      threshold=threshold, pixel_size=pixel_size)

    return LithoMetricsResult(
        epe=epe,
        cd_error=cd_err,
        ils=ils_result,
        nils=nils_result,
        pw_area={},
        meef={},
    )


@dataclass
class ProcessWindowScanResult:
    """
    Focus-Dose 工艺窗口扫描结果

    用于存储在 focus-dose 网格上各工艺条件的评估指标，
    为 Bossung 图和可打印区域热力图提供数据。
    """
    focus_values: np.ndarray
    dose_values: np.ndarray
    unique_focus: np.ndarray
    unique_dose: np.ndarray
    cd_matrix: np.ndarray
    cd_error_matrix: np.ndarray
    epe_matrix: np.ndarray
    mse_matrix: np.ndarray
    ssim_matrix: np.ndarray
    ils_matrix: np.ndarray
    nils_matrix: np.ndarray
    passing_mask: Optional[np.ndarray] = None

    @property
    def n_focus(self) -> int:
        return len(self.unique_focus)

    @property
    def n_dose(self) -> int:
        return len(self.unique_dose)


def extract_process_window_scan(
        multi_result,
        target_binary: np.ndarray,
        cd_target: Optional[float] = None,
        cd_tolerance: float = 0.1,
        pixel_size: float = 1.0,
        threshold: float = 0.3) -> ProcessWindowScanResult:
    """
    从多工艺仿真结果中提取 focus-dose 扫描的指标矩阵

    对每个 (focus, dose) 工艺条件，计算 CD、CD误差、EPE、MSE、SSIM、ILS、NILS 等指标，
    并整理为二维矩阵形式，便于后续可视化。

    Args:
        multi_result: MultiProcessSimulationResult 实例
        target_binary: 二值化目标图像
        cd_target: 目标CD (nm)；None 则从目标图自动测量
        cd_tolerance: CD 相对容差，用于判定可打印区域
        pixel_size: 像素尺寸 (nm)
        threshold: 光刻胶阈值

    Returns:
        ProcessWindowScanResult，包含各指标的二维矩阵
    """
    from core.metrics import mse, ssim

    conditions = multi_result.conditions
    wafer_images = multi_result.wafer_images
    aerial_images = multi_result.aerial_images

    n = len(conditions)
    if n == 0:
        raise ValueError("工艺条件列表为空")

    focus_values = np.array([c.defocus for c in conditions], dtype=np.float64)
    dose_values = np.array([c.dose for c in conditions], dtype=np.float64)

    unique_focus = np.sort(np.unique(focus_values))
    unique_dose = np.sort(np.unique(dose_values))
    n_f = len(unique_focus)
    n_d = len(unique_dose)

    focus_to_idx = {f: i for i, f in enumerate(unique_focus)}
    dose_to_idx = {d: j for j, d in enumerate(unique_dose)}

    cd_matrix = np.full((n_f, n_d), np.nan)
    cd_error_matrix = np.full((n_f, n_d), np.nan)
    epe_matrix = np.full((n_f, n_d), np.nan)
    mse_matrix = np.full((n_f, n_d), np.nan)
    ssim_matrix = np.full((n_f, n_d), np.nan)
    ils_matrix = np.full((n_f, n_d), np.nan)
    nils_matrix = np.full((n_f, n_d), np.nan)

    if cd_target is None:
        cd_target = compute_cd(target_binary, pixel_size=pixel_size)['cd_mean']

    for idx in range(n):
        fi = focus_to_idx[focus_values[idx]]
        di = dose_to_idx[dose_values[idx]]

        wafer = wafer_images[idx]
        wafer_bin = (wafer >= threshold).astype(np.float64)

        cd_info = compute_cd(wafer_bin, pixel_size=pixel_size)
        cd_matrix[fi, di] = cd_info['cd_mean']

        cd_err_info = compute_cd_error(wafer_bin, target_binary, pixel_size=pixel_size)
        cd_error_matrix[fi, di] = cd_err_info['cd_error_mean']

        epe_info = compute_epe(wafer_bin, target_binary, pixel_size=pixel_size)
        epe_matrix[fi, di] = epe_info['epe_mean']

        mse_matrix[fi, di] = mse(wafer, target_binary)
        ssim_matrix[fi, di] = ssim(wafer, target_binary)

        if idx < len(aerial_images) and aerial_images[idx] is not None:
            ils_info = compute_ils(aerial_images[idx], threshold=threshold, pixel_size=pixel_size)
            ils_matrix[fi, di] = ils_info['ils_mean']
            if cd_target and cd_target > 0:
                nils_info = compute_nils(aerial_images[idx], cd_target=cd_target,
                                         threshold=threshold, pixel_size=pixel_size)
                nils_matrix[fi, di] = nils_info['nils_mean']

    cd_lower = cd_target * (1.0 - cd_tolerance)
    cd_upper = cd_target * (1.0 + cd_tolerance)
    passing_mask = (cd_matrix >= cd_lower) & (cd_matrix <= cd_upper)

    return ProcessWindowScanResult(
        focus_values=focus_values,
        dose_values=dose_values,
        unique_focus=unique_focus,
        unique_dose=unique_dose,
        cd_matrix=cd_matrix,
        cd_error_matrix=cd_error_matrix,
        epe_matrix=epe_matrix,
        mse_matrix=mse_matrix,
        ssim_matrix=ssim_matrix,
        ils_matrix=ils_matrix,
        nils_matrix=nils_matrix,
        passing_mask=passing_mask,
    )
