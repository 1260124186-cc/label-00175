# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from scipy.ndimage import convolve, rotate
import logging

logger = logging.getLogger(__name__)


class EtchAnisotropyMode(Enum):
    VERTICAL_DOMINANT = "vertical_dominant"
    CONE = "cone"
    TRAPEZOIDAL = "trapezoidal"
    ELLIPTICAL = "elliptical"
    CUSTOM_KERNEL = "custom_kernel"


@dataclass
class AnisotropicEtchConfig:
    mode: EtchAnisotropyMode = EtchAnisotropyMode.CONE
    vertical_rate: float = 1.0
    lateral_rate: float = 0.1
    kernel_size: int = 15
    etch_time: float = 1.0
    sidewall_angle_deg: float = 88.0
    pixel_size: float = 1.0
    custom_kernel: Optional[np.ndarray] = None
    rotation_angle: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.value,
            'vertical_rate': self.vertical_rate,
            'lateral_rate': self.lateral_rate,
            'kernel_size': self.kernel_size,
            'etch_time': self.etch_time,
            'sidewall_angle_deg': self.sidewall_angle_deg,
            'pixel_size': self.pixel_size,
            'rotation_angle': self.rotation_angle,
        }


@dataclass
class AnisotropicEtchResult:
    etched_image: np.ndarray
    kernel: np.ndarray
    mode: str
    lateral_bias_nm: float = 0.0
    effective_anisotropy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'lateral_bias_nm': self.lateral_bias_nm,
            'effective_anisotropy': self.effective_anisotropy,
        }


def _build_cone_kernel(kernel_size: int,
                       lateral_rate: float,
                       vertical_rate: float) -> np.ndarray:
    half = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    if vertical_rate <= 0:
        return kernel
    aspect = lateral_rate / vertical_rate
    for i in range(kernel_size):
        for j in range(kernel_size):
            dy = (i - half)
            dx = (j - half)
            r = np.sqrt(dx ** 2 + dy ** 2) * aspect
            if r <= half:
                kernel[i, j] = max(0.0, 1.0 - r / (half + 1e-10))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def _build_trapezoidal_kernel(kernel_size: int,
                              sidewall_angle_deg: float,
                              lateral_rate: float,
                              vertical_rate: float) -> np.ndarray:
    half = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    tan_angle = np.tan(np.deg2rad(sidewall_angle_deg))
    for i in range(kernel_size):
        for j in range(kernel_size):
            dy = abs(i - half)
            dx = abs(j - half)
            r = np.sqrt(dx ** 2 + dy ** 2)
            r_lateral = r * lateral_rate / (vertical_rate + 1e-10)
            r_max = half * tan_angle / (tan_angle + 1e-10) if tan_angle > 0.01 else half
            if r_lateral <= r_max:
                kernel[i, j] = max(0.0, 1.0 - r_lateral / (r_max + 1e-10))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def _build_elliptical_kernel(kernel_size: int,
                             lateral_rate: float,
                             vertical_rate: float) -> np.ndarray:
    half = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    a = half * lateral_rate / (vertical_rate + 1e-10)
    b = float(half)
    for i in range(kernel_size):
        for j in range(kernel_size):
            dy = i - half
            dx = j - half
            if a > 1e-10 and b > 1e-10:
                ellipse_val = (dx / a) ** 2 + (dy / b) ** 2
            else:
                ellipse_val = (dx / (half + 1e-10)) ** 2 + (dy / (half + 1e-10)) ** 2
            if ellipse_val <= 1.0:
                kernel[i, j] = 1.0 - ellipse_val
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def build_etch_kernel(config: AnisotropicEtchConfig) -> np.ndarray:
    if config.mode == EtchAnisotropyMode.CUSTOM_KERNEL and config.custom_kernel is not None:
        kernel = config.custom_kernel.astype(np.float64)
        total = kernel.sum()
        if total > 0:
            kernel /= total
        return kernel

    ks = config.kernel_size
    if ks % 2 == 0:
        ks += 1

    if config.mode == EtchAnisotropyMode.VERTICAL_DOMINANT:
        kernel = np.zeros((ks, ks), dtype=np.float64)
        center = ks // 2
        kernel[center, center] = 1.0 - config.lateral_rate / (config.vertical_rate + 1e-10)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                kernel[center + di, center + dj] = (config.lateral_rate /
                                                      (config.vertical_rate + 1e-10)) / 8.0
        total = kernel.sum()
        if total > 0:
            kernel /= total
        return kernel

    elif config.mode == EtchAnisotropyMode.CONE:
        kernel = _build_cone_kernel(ks, config.lateral_rate, config.vertical_rate)
    elif config.mode == EtchAnisotropyMode.TRAPEZOIDAL:
        kernel = _build_trapezoidal_kernel(ks, config.sidewall_angle_deg,
                                           config.lateral_rate, config.vertical_rate)
    elif config.mode == EtchAnisotropyMode.ELLIPTICAL:
        kernel = _build_elliptical_kernel(ks, config.lateral_rate, config.vertical_rate)
    else:
        kernel = np.zeros((ks, ks), dtype=np.float64)
        center = ks // 2
        kernel[center, center] = 1.0

    if config.rotation_angle != 0.0:
        kernel = rotate(kernel, config.rotation_angle, reshape=False, order=1)
        total = kernel.sum()
        if total > 0:
            kernel /= total

    return kernel


def apply_anisotropic_etch(wafer_binary: np.ndarray,
                           config: AnisotropicEtchConfig) -> AnisotropicEtchResult:
    kernel = build_etch_kernel(config)
    wafer_float = wafer_binary.astype(np.float64)
    etch_response = convolve(wafer_float, kernel, mode='wrap')

    lateral_bias_nm = config.lateral_rate * config.etch_time * config.pixel_size * 2.0
    threshold = 1.0 - config.lateral_rate / (config.vertical_rate + 1e-10) * 0.5
    threshold = max(0.3, min(0.7, threshold))

    etched = (etch_response >= threshold).astype(np.float64)

    anisotropy = config.vertical_rate / (config.vertical_rate + config.lateral_rate + 1e-10)

    return AnisotropicEtchResult(
        etched_image=etched,
        kernel=kernel,
        mode=config.mode.value,
        lateral_bias_nm=lateral_bias_nm,
        effective_anisotropy=float(anisotropy),
    )


class AnisotropicEtchModel:
    def __init__(self, config: Optional[AnisotropicEtchConfig] = None):
        self.config = config or AnisotropicEtchConfig()
        self._kernel = None

    @property
    def kernel(self) -> np.ndarray:
        if self._kernel is None:
            self._kernel = build_etch_kernel(self.config)
        return self._kernel

    def apply(self, wafer_binary: np.ndarray) -> AnisotropicEtchResult:
        return apply_anisotropic_etch(wafer_binary, self.config)

    def compute_etch_gradient(self, wafer_binary: np.ndarray) -> np.ndarray:
        kernel = self.kernel
        wafer_float = wafer_binary.astype(np.float64)
        grad = convolve(wafer_float, kernel, mode='wrap')
        return grad
