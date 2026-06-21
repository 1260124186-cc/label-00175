# -*- coding: utf-8 -*-
"""
掩模检测图像仿真模块

模拟掩模检测机台的成像过程，支持：
1. 明场检测 (Bright Field) - 常规透射式成像
2. 暗场检测 (Dark Field) - 散射成像，增强小缺陷对比度
3. 相位对比检测 (Phase Contrast) - 相位缺陷增强
4. 偏振检测 (Polarization) - 基于偏振差异的缺陷检测

包含光学模糊、噪声、对比度增强、缺陷信号增强等物理模型。
"""

import numpy as np
from numba import jit, prange
from typing import Optional, Dict, Any, Tuple
from scipy.ndimage import gaussian_filter, sobel, distance_transform_edt
from scipy.signal import fftconvolve
import logging

from inspection.schemas import (
    InspectionMode,
    InspectionConfig,
    InspectionImageResult,
)

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def _gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    """
    生成 2D 高斯核

    Args:
        size: 核大小 (奇数)
        sigma: 高斯 sigma

    Returns:
        归一化的 2D 高斯核
    """
    kernel = np.zeros((size, size), dtype=np.float64)
    half = size // 2
    two_sigma_sq = 2.0 * sigma * sigma
    norm = 0.0

    for i in range(size):
        for j in range(size):
            y = i - half
            x = j - half
            val = np.exp(-(x * x + y * y) / two_sigma_sq)
            kernel[i, j] = val
            norm += val

    if norm > 0:
        for i in range(size):
            for j in range(size):
                kernel[i, j] /= norm

    return kernel


def _create_optical_blur_kernel(blur_sigma: float) -> np.ndarray:
    """
    根据光学模糊 sigma 生成模糊核

    Args:
        blur_sigma: 模糊 sigma (像素)

    Returns:
        2D 模糊核
    """
    sigma_pix = max(blur_sigma, 0.1)
    kernel_size = int(np.ceil(sigma_pix * 6)) | 1
    kernel_size = max(kernel_size, 3)
    return _gaussian_kernel_2d(kernel_size, sigma_pix)


def _compute_edge_strength(image: np.ndarray) -> np.ndarray:
    """
    计算图像边缘强度图

    Args:
        image: 输入图像 (2D)

    Returns:
        边缘强度图 (0~1 归一化)
    """
    grad_y = sobel(image.astype(np.float64), axis=0, mode='reflect')
    grad_x = sobel(image.astype(np.float64), axis=1, mode='reflect')
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    edge_max = edge_mag.max()
    if edge_max > 0:
        edge_mag = edge_mag / edge_max

    return edge_mag


def _simulate_bright_field(mask: np.ndarray, config: InspectionConfig) -> np.ndarray:
    """
    明场检测成像仿真

    明场模式：透射光成像，亮背景暗图案。
    铬层（不透明区域）吸收光线，呈现暗色；
    石英衬底（透明区域）透过光线，呈现亮色。

    Args:
        mask: 掩模图案 (0=不透光, 1=透光)
        config: 检测配置

    Returns:
        明场检测图像 (0~1)
    """
    img = mask.astype(np.float64).copy()

    if config.mode == InspectionMode.BRIGHT_FIELD:
        base_contrast = config.contrast_enhancement
        img = 0.1 + 0.8 * img
        img = np.clip(img, 0.0, 1.0)

        img = (img - 0.5) * base_contrast + 0.5
        img = np.clip(img, 0.0, 1.0)

    return img


def _simulate_dark_field(mask: np.ndarray, config: InspectionConfig) -> np.ndarray:
    """
    暗场检测成像仿真

    暗场模式：散射光成像，暗背景亮缺陷。
    只有散射光被收集，图案边缘和缺陷产生散射信号。
    小缺陷在暗场下对比度显著提高。

    Args:
        mask: 掩模图案 (0=不透光, 1=透光)
        config: 检测配置

    Returns:
        暗场检测图像 (0~1)
    """
    edge_map = _compute_edge_strength(mask)

    base_bg = 0.05
    edge_strength = config.contrast_enhancement * 1.5
    img = base_bg + edge_strength * edge_map

    dt_foreground = distance_transform_edt(mask > 0.5)
    dt_background = distance_transform_edt(mask <= 0.5)
    near_edge = np.minimum(dt_foreground, dt_background)
    near_edge_mask = near_edge < 5.0

    corner_effect = np.zeros_like(edge_map)
    corner_effect[near_edge_mask] = 0.3 * edge_map[near_edge_mask]
    img += corner_effect

    img = np.clip(img, 0.0, 1.0)
    return img


def _simulate_phase_contrast(mask: np.ndarray, config: InspectionConfig) -> np.ndarray:
    """
    相位对比检测成像仿真

    相位对比模式：对相位变化敏感，可检测相位缺陷
    和极小的透过率变化。基于 Zernike 相位板原理。

    Args:
        mask: 掩模图案 (0=不透光, 1=透光)
        config: 检测配置

    Returns:
        相位对比检测图像 (0~1)
    """
    edge_map = _compute_edge_strength(mask)

    img = mask.astype(np.float64).copy()
    phase_contrast = 0.3 + 0.4 * img + 0.5 * edge_map

    laplacian = np.zeros_like(mask, dtype=np.float64)
    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            laplacian[i, j] = (
                4 * mask[i, j]
                - mask[i-1, j] - mask[i+1, j]
                - mask[i, j-1] - mask[i, j+1]
            )

    laplacian_norm = np.abs(laplacian)
    lap_max = laplacian_norm.max() if laplacian_norm.max() > 0 else 1.0
    laplacian_norm = laplacian_norm / lap_max

    phase_contrast += 0.4 * config.contrast_enhancement * laplacian_norm
    phase_contrast = np.clip(phase_contrast, 0.0, 1.0)

    return phase_contrast


def _simulate_polarization(mask: np.ndarray, config: InspectionConfig) -> np.ndarray:
    """
    偏振检测成像仿真

    偏振模式：利用偏振差异检测缺陷。
    不同材料和结构对偏振态的影响不同，
    缺陷区域的偏振特性可能与正常区域不同。

    Args:
        mask: 掩模图案 (0=不透光, 1=透光)
        config: 检测配置

    Returns:
        偏振检测图像 (0~1)
    """
    edge_map = _compute_edge_strength(mask)

    s_component = mask.astype(np.float64)
    p_component = mask.astype(np.float64)

    edge_modulation = 0.15 * edge_map
    s_component = s_component * (1.0 + edge_modulation)
    p_component = p_component * (1.0 - edge_modulation * 0.5)

    pol_state = config.optics.polarization_state
    if pol_state == 's':
        img = s_component
    elif pol_state == 'p':
        img = p_component
    elif pol_state == 'circular':
        img = 0.5 * (s_component + p_component) + 0.3 * np.abs(s_component - p_component)
    else:
        img = 0.5 * (s_component + p_component) + 0.2 * config.contrast_enhancement * edge_map

    img = np.clip(img, 0.0, 1.0)
    return img


def _apply_defect_boost(
    inspection_image: np.ndarray,
    reference_image: np.ndarray,
    defect_mask: np.ndarray,
    boost_factor: float,
) -> np.ndarray:
    """
    增强缺陷区域的对比度

    Args:
        inspection_image: 检测图像
        reference_image: 参考图像（无缺陷）
        defect_mask: 缺陷位置二值掩模
        boost_factor: 增强倍数 (1~5)

    Returns:
        缺陷增强后的检测图像
    """
    if boost_factor <= 1.0 or not np.any(defect_mask):
        return inspection_image

    diff = np.abs(inspection_image - reference_image)
    defect_regions = defect_mask > 0.5

    enhanced = inspection_image.copy()
    boost = np.clip(boost_factor, 1.0, 5.0)

    if np.any(defect_regions):
        defect_diff = diff[defect_regions]
        if np.mean(defect_diff) > 0.01:
            max_diff = defect_diff.max() if defect_diff.max() > 0 else 1.0
            local_boost = 1.0 + (boost - 1.0) * (defect_diff / max_diff)
            enhanced[defect_regions] = reference_image[defect_regions] + \
                (inspection_image[defect_regions] - reference_image[defect_regions]) * local_boost

    enhanced = np.clip(enhanced, 0.0, 1.0)
    return enhanced


def _apply_optical_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    应用光学模糊

    Args:
        image: 输入图像
        kernel: 模糊核

    Returns:
        模糊后的图像
    """
    return fftconvolve(image, kernel, mode='same')


def _add_detector_noise(
    image: np.ndarray,
    noise_level: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    添加检测器噪声

    噪声模型：
    1. 散粒噪声 (Poisson) - 信号相关
    2. 读出噪声 (Gaussian) - 加性噪声

    Args:
        image: 输入图像 (0~1)
        noise_level: 噪声水平 (0~1)
        rng: 随机数生成器

    Returns:
        (含噪声图像, 噪声图)
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image.astype(np.float64).copy()
    noise_map = np.zeros_like(img)

    if noise_level > 0:
        photons_per_pixel = 500.0 / max(noise_level, 0.01)
        expected_counts = img * photons_per_pixel
        expected_counts = np.clip(expected_counts, 0.0, None)
        noisy_counts = rng.poisson(expected_counts)
        shot_noise = (noisy_counts - expected_counts) / photons_per_pixel
        noise_map += shot_noise

        read_noise_std = 0.01 * noise_level
        read_noise = rng.normal(0, read_noise_std, img.shape)
        noise_map += read_noise

        img = img + noise_map
        img = np.clip(img, 0.0, 1.0)

    return img, noise_map


def _apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    应用 Gamma 校正

    Args:
        image: 输入图像 (0~1)
        gamma: Gamma 指数

    Returns:
        校正后图像
    """
    if gamma == 1.0:
        return image
    return np.power(np.clip(image, 0.0, 1.0), max(gamma, 0.1))


def simulate_inspection_image(
    mask_test: np.ndarray,
    mask_reference: Optional[np.ndarray] = None,
    config: Optional[InspectionConfig] = None,
    defect_mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> InspectionImageResult:
    """
    生成掩模检测仿真图像

    根据配置的检测模式（明场/暗场/相位对比/偏振），
    模拟检测机台的成像过程，包括光学模糊、噪声、
    对比度增强和缺陷信号增强。

    Args:
        mask_test: 待测掩模图案 (0=不透光, 1=透光)，可能包含缺陷
        mask_reference: 参考掩模图案（数据库标称值），None 则使用无缺陷的 mask_test
        config: 检测配置，None 则使用默认配置
        defect_mask: 已知缺陷位置二值掩模，None 则自动从两幅掩模差异计算
        seed: 随机数种子，用于噪声复现

    Returns:
        InspectionImageResult，包含检测图像及中间结果

    使用示例::

        config = InspectionConfig(mode=InspectionMode.DARK_FIELD)
        result = simulate_inspection_image(mask_defective, mask_nominal, config)
        plt.imshow(result.inspection_image, cmap='gray')
    """
    if config is None:
        config = InspectionConfig()

    rng = np.random.default_rng(seed)

    if mask_test.ndim != 2:
        raise ValueError(f"mask_test 必须是 2D 数组，当前维度: {mask_test.ndim}")

    if mask_reference is None:
        mask_reference = mask_test.copy()

    if defect_mask is None:
        defect_mask = np.abs(mask_test.astype(np.float64) - mask_reference.astype(np.float64)) > 0.1

    mode = config.mode

    if mode == InspectionMode.BRIGHT_FIELD:
        ref_img = _simulate_bright_field(mask_reference, config)
        test_img = _simulate_bright_field(mask_test, config)
    elif mode == InspectionMode.DARK_FIELD:
        ref_img = _simulate_dark_field(mask_reference, config)
        test_img = _simulate_dark_field(mask_test, config)
    elif mode == InspectionMode.PHASE_CONTRAST:
        ref_img = _simulate_phase_contrast(mask_reference, config)
        test_img = _simulate_phase_contrast(mask_test, config)
    elif mode == InspectionMode.POLARIZATION:
        ref_img = _simulate_polarization(mask_reference, config)
        test_img = _simulate_polarization(mask_test, config)
    else:
        raise ValueError(f"不支持的检测模式: {mode}")

    blur_kernel = _create_optical_blur_kernel(config.blur_sigma)
    ref_blurred = _apply_optical_blur(ref_img, blur_kernel)
    test_blurred = _apply_optical_blur(test_img, blur_kernel)

    test_boosted = _apply_defect_boost(
        test_blurred, ref_blurred, defect_mask, config.defect_boost
    )

    test_noisy, noise_map = _add_detector_noise(test_boosted, config.noise_level, rng)
    ref_noisy, _ = _add_detector_noise(ref_blurred, config.noise_level, rng)

    test_final = _apply_gamma_correction(test_noisy, config.gamma)
    ref_final = _apply_gamma_correction(ref_noisy, config.gamma)

    edge_map = _compute_edge_strength(mask_test.astype(np.float64))

    return InspectionImageResult(
        inspection_image=test_final,
        reference_image=ref_final,
        defect_mask=defect_mask.astype(np.uint8),
        edge_map=edge_map,
        noise_map=noise_map,
        config=config,
        mode=mode,
    )


def simulate_multi_mode_inspection(
    mask_test: np.ndarray,
    mask_reference: Optional[np.ndarray] = None,
    modes: Optional[List[InspectionMode]] = None,
    base_config: Optional[InspectionConfig] = None,
    seed: Optional[int] = None,
) -> Dict[InspectionMode, InspectionImageResult]:
    """
    多模式检测图像仿真

    同时生成多种检测模式下的图像，用于对比分析
    不同模式对特定缺陷类型的检测能力。

    Args:
        mask_test: 待测掩模图案
        mask_reference: 参考掩模图案
        modes: 检测模式列表，None 则使用所有模式
        base_config: 基础检测配置
        seed: 随机数种子

    Returns:
        字典，键为检测模式，值为对应的检测图像结果

    使用示例::

        results = simulate_multi_mode_inspection(mask_def, mask_ref)
        for mode, result in results.items():
            print(f"{mode.value}: 最大差异 = {np.max(np.abs(result.inspection_image - result.reference_image)):.4f}")
    """
    if modes is None:
        modes = [
            InspectionMode.BRIGHT_FIELD,
            InspectionMode.DARK_FIELD,
            InspectionMode.PHASE_CONTRAST,
            InspectionMode.POLARIZATION,
        ]

    if base_config is None:
        base_config = InspectionConfig()

    results = {}
    for i, mode in enumerate(modes):
        config = InspectionConfig(
            mode=mode,
            optics=base_config.optics,
            noise_level=base_config.noise_level,
            contrast_enhancement=base_config.contrast_enhancement,
            defect_boost=base_config.defect_boost,
            threshold_abs=base_config.threshold_abs,
            threshold_rel=base_config.threshold_rel,
            blur_sigma=base_config.blur_sigma,
            gamma=base_config.gamma,
            adaptive_threshold=base_config.adaptive_threshold,
            min_defect_size_nm=base_config.min_defect_size_nm,
        )
        mode_seed = seed + i if seed is not None else None
        results[mode] = simulate_inspection_image(
            mask_test, mask_reference, config, seed=mode_seed
        )

    return results


def compute_defect_contrast(
    inspection_result: InspectionImageResult,
    defect_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算缺陷对比度

    评估检测图像中缺陷区域与背景的对比度，
    用于量化检测模式对缺陷的增强效果。

    Args:
        inspection_result: 检测图像结果
        defect_mask: 缺陷掩模，None 则使用结果中的 defect_mask

    Returns:
        字典，包含多种对比度指标：
            - 'mean_contrast': 平均绝对对比度
            - 'max_contrast': 最大对比度
            - 'defect_mean': 缺陷区域平均强度
            - 'background_mean': 背景区域平均强度
            - 'defect_std': 缺陷区域强度标准差
            - 'background_std': 背景区域强度标准差
            - 'snr': 信噪比 (|μ_defect - μ_bg| / σ_bg)
            - 'cnr': 对比度噪声比 (|μ_defect - μ_bg| / sqrt(σ_defect² + σ_bg²))
    """
    if defect_mask is None:
        defect_mask = inspection_result.defect_mask

    img = inspection_result.inspection_image
    defect_regions = defect_mask > 0.5
    background_regions = ~defect_regions

    if not np.any(defect_regions) or not np.any(background_regions):
        return {
            'mean_contrast': 0.0,
            'max_contrast': 0.0,
            'defect_mean': 0.0,
            'background_mean': 0.0,
            'defect_std': 0.0,
            'background_std': 0.0,
            'snr': 0.0,
            'cnr': 0.0,
        }

    defect_intensities = img[defect_regions]
    background_intensities = img[background_regions]

    defect_mean = float(np.mean(defect_intensities))
    bg_mean = float(np.mean(background_intensities))
    defect_std = float(np.std(defect_intensities))
    bg_std = float(np.std(background_intensities))

    diff = np.abs(img - inspection_result.reference_image)
    mean_contrast = float(np.mean(diff[defect_regions]))
    max_contrast = float(np.max(diff[defect_regions])) if np.any(defect_regions) else 0.0

    noise_level = bg_std if bg_std > 1e-6 else 1e-6
    snr = abs(defect_mean - bg_mean) / noise_level

    noise_combined = np.sqrt(defect_std**2 + bg_std**2) if (defect_std**2 + bg_std**2) > 1e-12 else 1e-6
    cnr = abs(defect_mean - bg_mean) / noise_combined

    return {
        'mean_contrast': mean_contrast,
        'max_contrast': max_contrast,
        'defect_mean': defect_mean,
        'background_mean': bg_mean,
        'defect_std': defect_std,
        'background_std': bg_std,
        'snr': snr,
        'cnr': cnr,
    }


from typing import List
