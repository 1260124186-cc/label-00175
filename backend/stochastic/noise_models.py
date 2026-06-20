# -*- coding: utf-8 -*-
"""
随机噪声模型模块

实现光刻仿真中的三个核心随机过程：
1. 光子散粒噪声 (Photon Shot Noise) - 光强的泊松统计波动
2. 光酸扩散随机性 (Photoacid Diffusion Stochasticity) - PAG分解与扩散的空间随机性
3. 显影阈值波动 (Development Threshold Fluctuation) - 光刻胶显影反应的分子级随机性
"""

import numpy as np
from numba import jit
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """噪声类型枚举"""
    PHOTON_SHOT = "photon_shot"
    PHOTOACID_DIFFUSION = "photoacid_diffusion"
    DEVELOPMENT_THRESHOLD = "development_threshold"
    ALL = "all"


@dataclass
class NoiseConfig:
    """
    随机噪声配置

    控制各类随机噪声的强度和参数。

    Attributes:
        photon_shot_noise_enabled: 是否启用光子散粒噪声
        photon_fluence: 标称光子注量 (photons/nm²)，用于计算散粒噪声强度
        photon_gain: 光子-光酸转换增益系数
        photoacid_diffusion_enabled: 是否启用光酸扩散随机性
        pag_concentration: PAG(光酸产生剂)标称浓度 (mol/cm³)
        pag_distribution_std: PAG空间分布的相对标准差
        diffusion_length_mean: 平均扩散长度 (nm)
        diffusion_length_std: 扩散长度的标准差 (nm)
        development_threshold_enabled: 是否启用显影阈值波动
        threshold_mean: 平均显影阈值 (归一化浓度单位)
        threshold_std: 显影阈值的标准差
        threshold_correlation_length: 阈值波动的空间相关长度 (nm)
        random_seed: 随机种子，用于结果复现
    """
    photon_shot_noise_enabled: bool = True
    photon_fluence: float = 100.0
    photon_gain: float = 0.8

    photoacid_diffusion_enabled: bool = True
    pag_concentration: float = 1e-3
    pag_distribution_std: float = 0.15
    diffusion_length_mean: float = 3.0
    diffusion_length_std: float = 0.5

    development_threshold_enabled: bool = True
    threshold_mean: float = 0.5
    threshold_std: float = 0.03
    threshold_correlation_length: float = 2.0

    random_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'photon_shot_noise_enabled': self.photon_shot_noise_enabled,
            'photon_fluence': self.photon_fluence,
            'photon_gain': self.photon_gain,
            'photoacid_diffusion_enabled': self.photoacid_diffusion_enabled,
            'pag_concentration': self.pag_concentration,
            'pag_distribution_std': self.pag_distribution_std,
            'diffusion_length_mean': self.diffusion_length_mean,
            'diffusion_length_std': self.diffusion_length_std,
            'development_threshold_enabled': self.development_threshold_enabled,
            'threshold_mean': self.threshold_mean,
            'threshold_std': self.threshold_std,
            'threshold_correlation_length': self.threshold_correlation_length,
            'random_seed': self.random_seed,
        }

    @classmethod
    def euv_default(cls, random_seed: Optional[int] = None) -> 'NoiseConfig':
        """创建 EUV 典型参数配置"""
        return cls(
            photon_fluence=50.0,
            photon_gain=0.6,
            pag_distribution_std=0.2,
            diffusion_length_mean=2.5,
            diffusion_length_std=0.8,
            threshold_std=0.04,
            threshold_correlation_length=1.5,
            random_seed=random_seed,
        )

    @classmethod
    def duv_arf_default(cls, random_seed: Optional[int] = None) -> 'NoiseConfig':
        """创建 DUV ArF 典型参数配置"""
        return cls(
            photon_fluence=200.0,
            photon_gain=0.9,
            pag_distribution_std=0.1,
            diffusion_length_mean=4.0,
            diffusion_length_std=0.4,
            threshold_std=0.02,
            threshold_correlation_length=3.0,
            random_seed=random_seed,
        )


@dataclass
class NoiseRealization:
    """
    单次噪声实现结果

    存储一次随机采样产生的各类噪声场。

    Attributes:
        photon_noise: 光子散粒噪声场 (相对强度波动)
        photoacid_concentration: 光酸浓度场 (包含PAG分布随机性)
        diffusion_length_field: 空间变化的扩散长度场 (nm)
        threshold_field: 空间变化的显影阈值场
        effective_threshold: 有效显影阈值（考虑所有噪声后的结果）
        noise_config: 产生该实现的噪声配置
        seed: 使用的随机种子
    """
    photon_noise: Optional[np.ndarray] = None
    photoacid_concentration: Optional[np.ndarray] = None
    diffusion_length_field: Optional[np.ndarray] = None
    threshold_field: Optional[np.ndarray] = None
    effective_threshold: Optional[np.ndarray] = None
    noise_config: Optional[NoiseConfig] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'has_photon_noise': self.photon_noise is not None,
            'has_photoacid_concentration': self.photoacid_concentration is not None,
            'has_diffusion_length': self.diffusion_length_field is not None,
            'has_threshold_field': self.threshold_field is not None,
            'seed': self.seed,
        }


class NoiseGenerator:
    """
    随机噪声生成器

    统一管理各类随机噪声的生成，支持独立控制各类噪声的开关和参数。

    使用方式::

        config = NoiseConfig()
        generator = NoiseGenerator(config)
        noise = generator.generate(shape=(256, 256), pixel_size=1.0)
    """

    def __init__(self, config: Optional[NoiseConfig] = None):
        """
        初始化噪声生成器

        Args:
            config: 噪声配置，None 则使用默认配置
        """
        self.config = config if config is not None else NoiseConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def reseed(self, seed: Optional[int] = None):
        """重置随机种子"""
        self.config.random_seed = seed
        self._rng = np.random.default_rng(seed)

    def _get_seed_for_realization(self) -> int:
        """为单次实现生成一个种子"""
        return int(self._rng.integers(0, 2**31 - 1))

    def generate_photon_shot_noise(
        self,
        shape: Tuple[int, int],
        pixel_size: float,
        nominal_intensity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        生成光子散粒噪声

        光子散粒噪声遵循泊松分布，方差等于均值。
        噪声强度与 sqrt(光子数) 成反比。

        Args:
            shape: 输出形状 (ny, nx)
            pixel_size: 像素尺寸 (nm)
            nominal_intensity: 标称光强分布 (归一化到 [0, 1])，None 则使用均匀光强 1.0

        Returns:
            包含散粒噪声的相对光强场 (均值约为 1.0)
        """
        if not self.config.photon_shot_noise_enabled:
            return np.ones(shape, dtype=np.float64)

        area_per_pixel = pixel_size ** 2
        mean_photons_per_pixel = self.config.photon_fluence * area_per_pixel

        if nominal_intensity is None:
            nominal_intensity = np.ones(shape, dtype=np.float64)

        expected_photons = mean_photons_per_pixel * nominal_intensity
        expected_photons = np.maximum(expected_photons, 1.0)

        noisy_photons = self._rng.poisson(expected_photons)
        relative_intensity = noisy_photons / expected_photons

        logger.debug(
            f"光子散粒噪声: 平均光子数={mean_photons_per_pixel:.1f}, "
            f"相对强度范围=[{relative_intensity.min():.3f}, {relative_intensity.max():.3f}]"
        )

        return relative_intensity.astype(np.float64)

    def generate_pag_distribution(
        self,
        shape: Tuple[int, int],
        pixel_size: float,
    ) -> np.ndarray:
        """
        生成 PAG (光酸产生剂) 空间分布

        PAG 分子在光刻胶中呈随机分布，导致光酸产生效率的空间波动。

        Args:
            shape: 输出形状 (ny, nx)
            pixel_size: 像素尺寸 (nm)

        Returns:
            归一化的 PAG 浓度场 (均值为 1.0)
        """
        if not self.config.photoacid_diffusion_enabled:
            return np.ones(shape, dtype=np.float64)

        local_pag = self._rng.normal(
            loc=1.0,
            scale=self.config.pag_distribution_std,
            size=shape,
        )
        local_pag = np.maximum(local_pag, 0.1)

        corr_length = max(1.0, self.config.threshold_correlation_length)
        sigma_pixels = corr_length / pixel_size
        if sigma_pixels > 0.5:
            local_pag = gaussian_filter(local_pag, sigma=sigma_pixels)

        pag_mean = local_pag.mean()
        if pag_mean > 1e-10:
            local_pag = local_pag / pag_mean

        logger.debug(
            f"PAG分布: std={self.config.pag_distribution_std:.3f}, "
            f"实际范围=[{local_pag.min():.3f}, {local_pag.max():.3f}]"
        )

        return local_pag.astype(np.float64)

    def generate_diffusion_length_field(
        self,
        shape: Tuple[int, int],
        pixel_size: float,
    ) -> np.ndarray:
        """
        生成空间变化的扩散长度场

        光酸扩散长度受局部环境影响，存在空间随机性。

        Args:
            shape: 输出形状 (ny, nx)
            pixel_size: 像素尺寸 (nm)

        Returns:
            扩散长度场 (nm)
        """
        if not self.config.photoacid_diffusion_enabled:
            return np.full(shape, self.config.diffusion_length_mean, dtype=np.float64)

        diffusion_lengths = self._rng.normal(
            loc=self.config.diffusion_length_mean,
            scale=self.config.diffusion_length_std,
            size=shape,
        )
        diffusion_lengths = np.maximum(diffusion_lengths, 0.5)

        corr_length = max(1.0, self.config.threshold_correlation_length)
        sigma_pixels = corr_length / pixel_size
        if sigma_pixels > 0.5:
            diffusion_lengths = gaussian_filter(diffusion_lengths, sigma=sigma_pixels)

        logger.debug(
            f"扩散长度: 均值={diffusion_lengths.mean():.2f}nm, "
            f"范围=[{diffusion_lengths.min():.2f}, {diffusion_lengths.max():.2f}]nm"
        )

        return diffusion_lengths.astype(np.float64)

    def generate_threshold_field(
        self,
        shape: Tuple[int, int],
        pixel_size: float,
    ) -> np.ndarray:
        """
        生成空间变化的显影阈值场

        显影阈值受光刻胶分子分布和局部浓度影响，存在空间相关性。

        Args:
            shape: 输出形状 (ny, nx)
            pixel_size: 像素尺寸 (nm)

        Returns:
            显影阈值场 (归一化单位)
        """
        if not self.config.development_threshold_enabled:
            return np.full(shape, self.config.threshold_mean, dtype=np.float64)

        raw_noise = self._rng.normal(
            loc=self.config.threshold_mean,
            scale=self.config.threshold_std,
            size=shape,
        )

        corr_length = max(0.1, self.config.threshold_correlation_length)
        sigma_pixels = corr_length / pixel_size
        if sigma_pixels > 0.5:
            raw_noise = gaussian_filter(raw_noise, sigma=sigma_pixels)

        filtered_std = raw_noise.std()
        if filtered_std > 1e-10:
            raw_noise = (
                self.config.threshold_mean
                + (raw_noise - raw_noise.mean())
                * self.config.threshold_std
                / filtered_std
            )

        raw_noise = np.clip(
            raw_noise,
            self.config.threshold_mean - 3 * self.config.threshold_std,
            self.config.threshold_mean + 3 * self.config.threshold_std,
        )

        logger.debug(
            f"显影阈值: 均值={raw_noise.mean():.4f}, "
            f"std={raw_noise.std():.4f}, "
            f"范围=[{raw_noise.min():.4f}, {raw_noise.max():.4f}]"
        )

        return raw_noise.astype(np.float64)

    def generate(
        self,
        shape: Tuple[int, int],
        pixel_size: float = 1.0,
        nominal_intensity: Optional[np.ndarray] = None,
        noise_types: Optional[Union[NoiseType, list]] = None,
    ) -> NoiseRealization:
        """
        生成完整的噪声实现

        Args:
            shape: 输出形状 (ny, nx)
            pixel_size: 像素尺寸 (nm)
            nominal_intensity: 标称光强分布，用于光子散粒噪声计算
            noise_types: 要生成的噪声类型，None 则生成所有启用的噪声

        Returns:
            NoiseRealization 实例，包含各类噪声场
        """
        realization_seed = self._get_seed_for_realization()
        realization_rng = np.random.default_rng(realization_seed)

        temp_rng = self._rng
        self._rng = realization_rng

        try:
            photon_noise = None
            photoacid_conc = None
            diffusion_length = None
            threshold_field = None

            if noise_types is None or noise_types == NoiseType.ALL:
                generate_photon = self.config.photon_shot_noise_enabled
                generate_pag = self.config.photoacid_diffusion_enabled
                generate_threshold = self.config.development_threshold_enabled
            elif isinstance(noise_types, NoiseType):
                generate_photon = noise_types in (NoiseType.PHOTON_SHOT, NoiseType.ALL)
                generate_pag = noise_types in (NoiseType.PHOTOACID_DIFFUSION, NoiseType.ALL)
                generate_threshold = noise_types in (NoiseType.DEVELOPMENT_THRESHOLD, NoiseType.ALL)
            else:
                generate_photon = NoiseType.PHOTON_SHOT in noise_types
                generate_pag = NoiseType.PHOTOACID_DIFFUSION in noise_types
                generate_threshold = NoiseType.DEVELOPMENT_THRESHOLD in noise_types

            if generate_photon:
                photon_noise = self.generate_photon_shot_noise(
                    shape, pixel_size, nominal_intensity
                )

            if generate_pag:
                photoacid_conc = self.generate_pag_distribution(shape, pixel_size)
                diffusion_length = self.generate_diffusion_length_field(shape, pixel_size)

            if generate_threshold:
                threshold_field = self.generate_threshold_field(shape, pixel_size)

            if threshold_field is not None:
                effective_threshold = threshold_field.copy()
            else:
                effective_threshold = np.full(shape, self.config.threshold_mean)

            return NoiseRealization(
                photon_noise=photon_noise,
                photoacid_concentration=photoacid_conc,
                diffusion_length_field=diffusion_length,
                threshold_field=threshold_field,
                effective_threshold=effective_threshold,
                noise_config=self.config,
                seed=realization_seed,
            )
        finally:
            self._rng = temp_rng


@jit(nopython=True, parallel=True, cache=True)
def apply_stochastic_diffusion(
    latent_image: np.ndarray,
    diffusion_length_field: np.ndarray,
    pixel_size: float,
) -> np.ndarray:
    """
    应用空间变化的高斯扩散（Numba加速）

    对每个像素使用其局部扩散长度进行高斯滤波。

    Args:
        latent_image: 潜像 (曝光后的光酸浓度分布)
        diffusion_length_field: 空间变化的扩散长度场 (nm)
        pixel_size: 像素尺寸 (nm)

    Returns:
        扩散后的光酸浓度分布
    """
    ny, nx = latent_image.shape
    result = np.zeros_like(latent_image)

    max_sigma_pix = int(np.ceil(diffusion_length_field.max() / pixel_size * 3)) + 1
    max_sigma_pix = max(max_sigma_pix, 1)

    for y in range(ny):
        for x in range(nx):
            sigma = diffusion_length_field[y, x] / pixel_size
            if sigma <= 0.1:
                result[y, x] = latent_image[y, x]
                continue

            half_kernel = int(np.ceil(sigma * 3))
            half_kernel = min(half_kernel, max_sigma_pix)

            y_start = max(0, y - half_kernel)
            y_end = min(ny, y + half_kernel + 1)
            x_start = max(0, x - half_kernel)
            x_end = min(nx, x + half_kernel + 1)

            kernel_sum = 0.0
            value_sum = 0.0
            two_sigma_sq = 2.0 * sigma * sigma

            for yy in range(y_start, y_end):
                for xx in range(x_start, x_end):
                    dy = (yy - y) * pixel_size
                    dx = (xx - x) * pixel_size
                    dist_sq = dy * dy + dx * dx
                    weight = np.exp(-dist_sq / two_sigma_sq)
                    kernel_sum += weight
                    value_sum += weight * latent_image[yy, xx]

            if kernel_sum > 0:
                result[y, x] = value_sum / kernel_sum
            else:
                result[y, x] = latent_image[y, x]

    return result


def apply_stochastic_lithography(
    aerial_image: np.ndarray,
    noise: NoiseRealization,
    pixel_size: float = 1.0,
    base_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用完整的随机光刻流程

    1. 光子散粒噪声 → 2. PAG效率波动 → 3. 空间变化扩散 → 4. 空间变化阈值

    Args:
        aerial_image: 标称空间像光强分布
        noise: 噪声实现
        pixel_size: 像素尺寸 (nm)
        base_threshold: 基础显影阈值（当 noise.effective_threshold 为 None 时使用）

    Returns:
        (latent_image, resist_image)
        - latent_image: 扩散后的潜像（光酸浓度）
        - resist_image: 显影后的光刻胶图像（二值化前的连续值）
    """
    ny, nx = aerial_image.shape

    intensity = aerial_image.astype(np.float64).copy()

    if noise.photon_noise is not None:
        intensity = intensity * noise.photon_noise

    if noise.photoacid_concentration is not None:
        latent = intensity * noise.photoacid_concentration
    else:
        latent = intensity.copy()

    if noise.diffusion_length_field is not None:
        latent = apply_stochastic_diffusion(latent, noise.diffusion_length_field, pixel_size)
    else:
        sigma = 3.0 / pixel_size
        if sigma > 0.5:
            latent = gaussian_filter(latent, sigma=sigma)

    if noise.effective_threshold is not None:
        threshold = noise.effective_threshold
    else:
        threshold = np.full_like(latent, base_threshold)

    resist_image = latent - threshold

    return latent, resist_image


def create_euv_noise_config(
    photon_fluence: float = 50.0,
    diffusion_length_mean: float = 2.0,
    threshold_std: float = 0.05,
    random_seed: Optional[int] = None,
) -> NoiseConfig:
    """
    创建 EUV 典型参数的噪声配置

    Args:
        photon_fluence: 光子通量 (光子/nm²)
        diffusion_length_mean: 平均扩散长度 (nm)
        threshold_std: 阈值标准差
        random_seed: 随机种子

    Returns:
        EUV 噪声配置
    """
    return NoiseConfig(
        photon_shot_noise_enabled=True,
        photon_fluence=photon_fluence,
        photon_gain=0.7,
        photoacid_diffusion_enabled=True,
        pag_distribution_std=0.2,
        diffusion_length_mean=diffusion_length_mean,
        diffusion_length_std=0.6,
        development_threshold_enabled=True,
        threshold_mean=0.5,
        threshold_std=threshold_std,
        threshold_correlation_length=1.5,
        random_seed=random_seed,
    )


def create_duv_noise_config(
    photon_fluence: float = 200.0,
    diffusion_length_mean: float = 4.0,
    threshold_std: float = 0.02,
    random_seed: Optional[int] = None,
) -> NoiseConfig:
    """
    创建 DUV 典型参数的噪声配置

    Args:
        photon_fluence: 光子通量 (光子/nm²)
        diffusion_length_mean: 平均扩散长度 (nm)
        threshold_std: 阈值标准差
        random_seed: 随机种子

    Returns:
        DUV 噪声配置
    """
    return NoiseConfig(
        photon_shot_noise_enabled=True,
        photon_fluence=photon_fluence,
        photon_gain=0.9,
        photoacid_diffusion_enabled=True,
        pag_distribution_std=0.1,
        diffusion_length_mean=diffusion_length_mean,
        diffusion_length_std=0.4,
        development_threshold_enabled=True,
        threshold_mean=0.5,
        threshold_std=threshold_std,
        threshold_correlation_length=2.5,
        random_seed=random_seed,
    )
