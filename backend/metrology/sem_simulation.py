# -*- coding: utf-8 -*-
"""
SEM/CD-SEM 图像仿真模块

基于晶圆拓扑（光强分布、光刻胶轮廓）生成类 SEM 灰度图像，
模拟扫描电子显微镜的成像物理过程，用于视觉对比与算法验证。

仿真模型包括：
1. 二次电子发射 (Secondary Electron Emission, SEE) 模型
2. 边缘增强效应 (Edge Enhancement)
3. 电子束模糊 (Beam Blur) - 有限束斑尺寸
4. 噪声模型 (Shot Noise、Gaussian Noise)
5. 对比度与亮度控制
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy.ndimage import gaussian_filter, sobel, laplace, binary_erosion, distance_transform_edt
from scipy.signal import fftconvolve
import logging

logger = logging.getLogger(__name__)


class SEMDetectorMode(Enum):
    """SEM 探测器模式"""
    INLENS = "inlens"              # 镜内探测器，高分辨率，边缘增强明显
    EVERHART_THORNLEY = "et"       # Everhart-Thornley 探测器，常规成像
    BACKSCATTERED = "bse"          # 背散射电子探测器，成分对比


@dataclass
class SEMSimConfig:
    """
    SEM 图像仿真参数配置

    Attributes:
        beam_diameter_nm: 电子束直径 (nm)，决定模糊程度
        acceleration_voltage_kv: 加速电压 (kV)，影响穿透深度与 SE 产额
        working_distance_mm: 工作距离 (mm)，影响像差与分辨率
        edge_enhancement: 边缘增强强度 (0~2)，越大边缘越亮
        noise_level: 噪声强度 (0~1)，0 为无噪声
        shot_noise: 是否添加散粒噪声 (Poisson 分布)
        gaussian_noise_std: 高斯噪声标准差 (0~1 归一化强度)
        brightness: 亮度偏移 (-1~1)
        contrast: 对比度增益 (0.5~3)
        gamma: Gamma 校正指数
        detector_mode: 探测器模式
        pixel_size_nm: 图像像素尺寸 (nm)，用于束斑计算
    """
    beam_diameter_nm: float = 3.0
    acceleration_voltage_kv: float = 1.0
    working_distance_mm: float = 5.0
    edge_enhancement: float = 1.0
    noise_level: float = 0.05
    shot_noise: bool = True
    gaussian_noise_std: float = 0.02
    brightness: float = 0.0
    contrast: float = 1.2
    gamma: float = 1.0
    detector_mode: SEMDetectorMode = SEMDetectorMode.INLENS
    pixel_size_nm: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d['detector_mode'] = self.detector_mode.value
        return d


@dataclass
class SEMSimResult:
    """
    SEM 图像仿真结果

    Attributes:
        sem_image: 生成的 SEM 灰度图 (0~1 归一化)
        se_yield_map: 二次电子产额分布图
        edge_map: 边缘强度图
        beam_blur_kernel: 使用的电子束模糊核
        config: 使用的仿真配置
    """
    sem_image: np.ndarray
    se_yield_map: np.ndarray
    edge_map: np.ndarray
    beam_blur_kernel: np.ndarray
    config: SEMSimConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sem_image_shape': list(self.sem_image.shape),
            'config': self.config.to_dict(),
        }


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


def _create_beam_blur_kernel(beam_diameter_nm: float,
                              pixel_size_nm: float) -> np.ndarray:
    """
    根据电子束直径生成模糊核

    使用高斯函数近似电子束的高斯强度分布，
    束斑直径定义为强度降至 1/e² 处的全宽。

    Args:
        beam_diameter_nm: 电子束直径 (nm)
        pixel_size_nm: 像素尺寸 (nm)

    Returns:
        2D 模糊核
    """
    sigma_pix = (beam_diameter_nm / pixel_size_nm) / (2.0 * np.sqrt(2.0))
    sigma_pix = max(sigma_pix, 0.1)

    kernel_size = int(np.ceil(sigma_pix * 6)) | 1
    kernel_size = max(kernel_size, 3)

    return _gaussian_kernel_2d(kernel_size, sigma_pix)


def _compute_edge_strength(topography: np.ndarray) -> np.ndarray:
    """
    计算晶圆拓扑的边缘强度图

    使用 Sobel 算子计算梯度幅值，用于模拟 SE 边缘增强。

    Args:
        topography: 晶圆拓扑/光强分布 (2D)

    Returns:
        边缘强度图 (0~1 归一化)
    """
    grad_y = sobel(topography.astype(np.float64), axis=0, mode='reflect')
    grad_x = sobel(topography.astype(np.float64), axis=1, mode='reflect')
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    edge_max = edge_mag.max()
    if edge_max > 0:
        edge_mag = edge_mag / edge_max

    return edge_mag


def _compute_se_yield(topography: np.ndarray,
                       edge_map: np.ndarray,
                       config: SEMSimConfig) -> np.ndarray:
    """
    计算二次电子 (SE) 产额分布

    SE 产额模型：
    - 基底产额：正比于材料原子序数，此处简化为与拓扑高度相关
    - 边缘增强：边缘处 SE 产额显著提高（电子逃逸概率高）
    - 探测器模式影响：InLens 模式边缘增强更强

    Args:
        topography: 晶圆拓扑/光强分布 (0~1)
        edge_map: 边缘强度图 (0~1)
        config: SEM 仿真配置

    Returns:
        SE 产额图 (0~1 归一化)
    """
    topo_norm = (topography - topography.min()) / (topography.max() - topography.min() + 1e-10)

    if config.detector_mode == SEMDetectorMode.INLENS:
        edge_factor = 2.5 * config.edge_enhancement
    elif config.detector_mode == SEMDetectorMode.EVERHART_THORNLEY:
        edge_factor = 1.5 * config.edge_enhancement
    else:
        edge_factor = 0.5 * config.edge_enhancement

    base_yield = 0.3 + 0.5 * topo_norm
    edge_yield = edge_factor * edge_map * (0.3 + 0.7 * topo_norm)

    se_yield = base_yield + edge_yield

    se_min, se_max = se_yield.min(), se_yield.max()
    if se_max > se_min:
        se_yield = (se_yield - se_min) / (se_max - se_min)

    return se_yield


def _apply_beam_blur(se_yield: np.ndarray,
                      kernel: np.ndarray) -> np.ndarray:
    """
    应用电子束模糊

    Args:
        se_yield: SE 产额图
        kernel: 模糊核

    Returns:
        模糊后的图像
    """
    return fftconvolve(se_yield, kernel, mode='same')


def _add_noise(image: np.ndarray,
                config: SEMSimConfig,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    添加 SEM 噪声

    噪声模型：
    1. 散粒噪声 (Shot Noise)：服从 Poisson 分布，正比于信号强度
    2. 高斯噪声：探测器读出噪声，加性

    Args:
        image: 输入图像 (0~1)
        config: SEM 仿真配置
        rng: 随机数生成器

    Returns:
        含噪声图像
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image.astype(np.float64).copy()
    img = np.clip(img, 0.0, 1.0)

    if config.shot_noise and config.noise_level > 0:
        electrons_per_pixel = 1000.0 / max(config.noise_level, 0.01)
        expected_counts = img * electrons_per_pixel
        expected_counts = np.clip(expected_counts, 0.0, None)
        expected_counts = np.nan_to_num(expected_counts, nan=0.0, posinf=0.0, neginf=0.0)
        noisy_counts = rng.poisson(expected_counts)
        img = noisy_counts / electrons_per_pixel

    if config.gaussian_noise_std > 0:
        gaussian_noise = rng.normal(0, config.gaussian_noise_std, img.shape)
        img = img + gaussian_noise

    img = np.clip(img, 0.0, 1.0)
    return img


def _apply_brightness_contrast(image: np.ndarray,
                                config: SEMSimConfig) -> np.ndarray:
    """
    应用亮度、对比度、Gamma 校正

    Args:
        image: 输入图像 (0~1)
        config: SEM 仿真配置

    Returns:
        校正后图像
    """
    img = image.astype(np.float64)

    img = (img - 0.5) * config.contrast + 0.5 + config.brightness

    img = np.clip(img, 0.0, 1.0)

    if config.gamma != 1.0 and config.gamma > 0:
        img = np.power(img, config.gamma)

    return img


def simulate_sem_image(topography: np.ndarray,
                        config: Optional[SEMSimConfig] = None,
                        seed: Optional[int] = None) -> SEMSimResult:
    """
    从晶圆拓扑生成 SEM 灰度图像

    完整仿真流程：
    1. 计算边缘强度图
    2. 计算二次电子产额分布
    3. 应用电子束模糊
    4. 添加噪声
    5. 亮度/对比度/Gamma 校正

    Args:
        topography: 晶圆拓扑/光强分布 (2D 数组)，可为光强图或光刻胶轮廓
        config: SEM 仿真配置，为 None 时使用默认参数
        seed: 随机数种子，用于复现噪声

    Returns:
        SEMSimResult，包含生成的图像及中间结果
    """
    if config is None:
        config = SEMSimConfig()

    rng = np.random.default_rng(seed)

    topo = topography.astype(np.float64)
    if topo.ndim != 2:
        raise ValueError(f"topography 必须是 2D 数组，当前维度: {topo.ndim}")

    edge_map = _compute_edge_strength(topo)

    se_yield = _compute_se_yield(topo, edge_map, config)

    blur_kernel = _create_beam_blur_kernel(
        config.beam_diameter_nm,
        config.pixel_size_nm
    )

    blurred = _apply_beam_blur(se_yield, blur_kernel)

    noisy = _add_noise(blurred, config, rng)

    sem_image = _apply_brightness_contrast(noisy, config)

    return SEMSimResult(
        sem_image=sem_image,
        se_yield_map=se_yield,
        edge_map=edge_map,
        beam_blur_kernel=blur_kernel,
        config=config,
    )


def simulate_cd_sem_line_scan(topography: np.ndarray,
                               line_y: int,
                               config: Optional[SEMSimConfig] = None,
                               seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    模拟 CD-SEM 线扫描 (Line Scan) 信号

    沿指定水平线提取一维 SEM 强度信号，
    包含束斑模糊和噪声，用于 CD 算法测试。

    Args:
        topography: 晶圆拓扑 (2D)
        line_y: 扫描线的行索引
        config: SEM 仿真配置
        seed: 随机种子

    Returns:
        字典，包含：
            - 'position': 位置坐标 (像素)
            - 'ideal_profile': 理想拓扑轮廓 (无噪声无模糊)
            - 'sem_profile': SEM 扫描信号 (含模糊和噪声)
    """
    if config is None:
        config = SEMSimConfig()

    sim_result = simulate_sem_image(topography, config, seed)

    line_y_clamped = max(0, min(line_y, topography.shape[0] - 1))

    position = np.arange(topography.shape[1], dtype=np.float64)
    ideal_profile = topography[line_y_clamped, :].astype(np.float64)
    sem_profile = sim_result.sem_image[line_y_clamped, :].astype(np.float64)

    return {
        'position': position,
        'ideal_profile': ideal_profile,
        'sem_profile': sem_profile,
    }


@jit(nopython=True, cache=True)
def _compute_charging_map_1d(profile: np.ndarray,
                              charging_strength: float = 0.3) -> np.ndarray:
    """
    计算一维荷电效应 (Charging) 强度

    绝缘体区域会积累电荷，导致局部亮度变化。

    Args:
        profile: 一维信号
        charging_strength: 荷电强度 (0~1)

    Returns:
        荷电效应偏移图
    """
    n = len(profile)
    charging = np.zeros(n, dtype=np.float64)

    low_threshold = np.min(profile) + 0.2 * (np.max(profile) - np.min(profile))
    insulator_mask = profile < low_threshold

    for i in range(n):
        if insulator_mask[i]:
            left = max(0, i - 10)
            right = min(n, i + 11)
            local_mean = 0.0
            count = 0
            for j in range(left, right):
                local_mean += profile[j]
                count += 1
            if count > 0:
                local_mean /= count
            charging[i] = charging_strength * (local_mean - profile[i])

    return charging


def apply_charging_effect(sem_image: np.ndarray,
                           strength: float = 0.2) -> np.ndarray:
    """
    对 SEM 图像施加荷电效应

    光刻胶等绝缘材料在电子束照射下会积累电荷，
    导致局部亮度异常（通常是暗区变亮）。

    Args:
        sem_image: SEM 图像 (0~1)
        strength: 荷电强度 (0~1)

    Returns:
        施加荷电效应后的图像
    """
    img = sem_image.astype(np.float64)
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        charging_1d = _compute_charging_map_1d(img[i, :], strength)
        result[i, :] = np.clip(img[i, :] + charging_1d, 0.0, 1.0)

    return result
