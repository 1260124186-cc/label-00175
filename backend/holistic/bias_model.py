# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy.ndimage import uniform_filter, gaussian_filter
import logging

logger = logging.getLogger(__name__)


class BiasModelType(Enum):
    CONSTANT = "constant"
    DENSITY_DEPENDENT = "density_dependent"
    PITCH_DEPENDENT = "pitch_dependent"
    LOADING_EFFECT = "loading_effect"


@dataclass
class BiasModelConfig:
    model_type: BiasModelType = BiasModelType.DENSITY_DEPENDENT
    constant_bias_nm: float = 3.0
    density_filter_size: int = 15
    density_bias_coefficients: Dict[str, float] = field(default_factory=lambda: {
        'a0': 2.0,
        'a1': -4.0,
        'a2': 2.0,
    })
    pitch_sensitivity: float = 0.01
    min_pitch_nm: float = 40.0
    max_pitch_nm: float = 500.0
    loading_coefficient: float = 0.3
    pixel_size: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type.value,
            'constant_bias_nm': self.constant_bias_nm,
            'density_filter_size': self.density_filter_size,
            'density_bias_coefficients': dict(self.density_bias_coefficients),
            'pitch_sensitivity': self.pitch_sensitivity,
            'min_pitch_nm': self.min_pitch_nm,
            'max_pitch_nm': self.max_pitch_nm,
            'loading_coefficient': self.loading_coefficient,
            'pixel_size': self.pixel_size,
        }


@dataclass
class BiasModelResult:
    bias_map: np.ndarray
    biased_image: np.ndarray
    model_type: str
    bias_mean_nm: float = 0.0
    bias_std_nm: float = 0.0
    bias_max_nm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'bias_mean_nm': self.bias_mean_nm,
            'bias_std_nm': self.bias_std_nm,
            'bias_max_nm': self.bias_max_nm,
        }


def compute_local_density(wafer_binary: np.ndarray,
                          filter_size: int = 15) -> np.ndarray:
    wafer_float = wafer_binary.astype(np.float64)
    if filter_size < 3:
        return wafer_float
    return uniform_filter(wafer_float, size=filter_size, mode='wrap')


def compute_bias_constant(shape: tuple,
                          bias_nm: float,
                          pixel_size: float = 1.0) -> np.ndarray:
    bias_pix = bias_nm / pixel_size
    return np.full(shape, bias_pix, dtype=np.float64)


def compute_bias_density_dependent(wafer_binary: np.ndarray,
                                   config: BiasModelConfig) -> np.ndarray:
    density = compute_local_density(wafer_binary, config.density_filter_size)
    coeffs = config.density_bias_coefficients
    a0 = coeffs.get('a0', 2.0)
    a1 = coeffs.get('a1', -4.0)
    a2 = coeffs.get('a2', 2.0)
    bias_nm = a0 + a1 * density + a2 * density ** 2
    bias_pix = bias_nm / config.pixel_size
    return bias_pix


def compute_bias_pitch_dependent(wafer_binary: np.ndarray,
                                 config: BiasModelConfig) -> np.ndarray:
    density = compute_local_density(wafer_binary, config.density_filter_size)
    pitch_nm = np.where(density > 0.01,
                        config.pixel_size / (density + 1e-10),
                        config.max_pitch_nm)
    pitch_nm = np.clip(pitch_nm, config.min_pitch_nm, config.max_pitch_nm)
    pitch_norm = (pitch_nm - config.min_pitch_nm) / (config.max_pitch_nm - config.min_pitch_nm + 1e-10)
    bias_nm = config.constant_bias_nm * (1.0 + config.pitch_sensitivity * pitch_norm)
    bias_pix = bias_nm / config.pixel_size
    return bias_pix


def compute_bias_loading_effect(wafer_binary: np.ndarray,
                                config: BiasModelConfig) -> np.ndarray:
    density = compute_local_density(wafer_binary, config.density_filter_size)
    bias_nm = config.constant_bias_nm * (1.0 + config.loading_coefficient * density)
    bias_pix = bias_nm / config.pixel_size
    return bias_pix


def apply_bias_to_image(wafer_binary: np.ndarray,
                        bias_map: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
    wafer_bin = (wafer_binary >= 0.5).astype(np.float64)
    dist_foreground = distance_transform_edt(1.0 - wafer_bin)
    dist_background = distance_transform_edt(wafer_bin)

    etched = np.zeros_like(wafer_bin)
    mask_region = wafer_bin > 0.5
    etched[mask_region & (dist_foreground <= np.abs(bias_map) + 1e-6)] = 0.0
    etched[mask_region & (dist_foreground > np.abs(bias_map) + 1e-6)] = 1.0
    expand_mask = (~mask_region) & (dist_background <= np.abs(bias_map) + 1e-6)
    etched[expand_mask] = 1.0
    etched[~mask_region & ~expand_mask] = 0.0
    return etched


class EtchBiasModel:
    def __init__(self, config: Optional[BiasModelConfig] = None):
        self.config = config or BiasModelConfig()

    def compute_bias(self, wafer_binary: np.ndarray) -> np.ndarray:
        if self.config.model_type == BiasModelType.CONSTANT:
            return compute_bias_constant(wafer_binary.shape,
                                         self.config.constant_bias_nm,
                                         self.config.pixel_size)
        elif self.config.model_type == BiasModelType.DENSITY_DEPENDENT:
            return compute_bias_density_dependent(wafer_binary, self.config)
        elif self.config.model_type == BiasModelType.PITCH_DEPENDENT:
            return compute_bias_pitch_dependent(wafer_binary, self.config)
        elif self.config.model_type == BiasModelType.LOADING_EFFECT:
            return compute_bias_loading_effect(wafer_binary, self.config)
        else:
            raise ValueError(f"Unknown bias model type: {self.config.model_type}")

    def apply(self, wafer_binary: np.ndarray) -> BiasModelResult:
        bias_map = self.compute_bias(wafer_binary)
        biased_image = apply_bias_to_image(wafer_binary, bias_map)
        bias_nm = bias_map * self.config.pixel_size
        return BiasModelResult(
            bias_map=bias_map,
            biased_image=biased_image,
            model_type=self.config.model_type.value,
            bias_mean_nm=float(np.mean(bias_nm)),
            bias_std_nm=float(np.std(bias_nm)),
            bias_max_nm=float(np.max(np.abs(bias_nm))),
        )
