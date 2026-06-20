# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from scipy.ndimage import distance_transform_edt, gaussian_filter
import logging

logger = logging.getLogger(__name__)


class TrenchProfileModel(Enum):
    RECTANGULAR = "rectangular"
    TRAPEZOIDAL = "trapezoidal"
    ROUNDED_BOTTOM = "rounded_bottom"
    BOWING = "bowing"


@dataclass
class EtchProcessConfig:
    etch_rate_si_nm_per_s: float = 50.0
    etch_rate_resist_nm_per_s: float = 5.0
    etch_time_s: float = 10.0
    selectivity: float = 10.0
    resist_thickness_nm: float = 100.0
    pixel_size: float = 1.0
    trench_profile: TrenchProfileModel = TrenchProfileModel.TRAPEZOIDAL
    sidewall_angle_deg: float = 88.0
    bowing_factor: float = 0.1
    micro_loading_factor: float = 0.05
    aspect_ratio_dependent_etch: bool = True
    arde_coefficient: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            'etch_rate_si_nm_per_s': self.etch_rate_si_nm_per_s,
            'etch_rate_resist_nm_per_s': self.etch_rate_resist_nm_per_s,
            'etch_time_s': self.etch_time_s,
            'selectivity': self.selectivity,
            'resist_thickness_nm': self.resist_thickness_nm,
            'pixel_size': self.pixel_size,
            'trench_profile': self.trench_profile.value,
            'sidewall_angle_deg': self.sidewall_angle_deg,
            'bowing_factor': self.bowing_factor,
            'micro_loading_factor': self.micro_loading_factor,
            'aspect_ratio_dependent_etch': self.aspect_ratio_dependent_etch,
            'arde_coefficient': self.arde_coefficient,
        }


@dataclass
class TrenchDepthResult:
    depth_map: np.ndarray
    depth_mean_nm: float
    depth_min_nm: float
    depth_max_nm: float
    depth_std_nm: float
    resist_consumed_nm: float
    etch_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'depth_mean_nm': self.depth_mean_nm,
            'depth_min_nm': self.depth_min_nm,
            'depth_max_nm': self.depth_max_nm,
            'depth_std_nm': self.depth_std_nm,
            'resist_consumed_nm': self.resist_consumed_nm,
            'etch_time_s': self.etch_time_s,
        }


@dataclass
class PostEtchCDResult:
    cd_resist_nm: float
    cd_etch_nm: float
    cd_shift_nm: float
    cd_top_nm: float
    cd_bottom_nm: float
    trench_profile: str
    sidewall_angle_deg: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cd_resist_nm': self.cd_resist_nm,
            'cd_etch_nm': self.cd_etch_nm,
            'cd_shift_nm': self.cd_shift_nm,
            'cd_top_nm': self.cd_top_nm,
            'cd_bottom_nm': self.cd_bottom_nm,
            'trench_profile': self.trench_profile,
            'sidewall_angle_deg': self.sidewall_angle_deg,
        }


@dataclass
class MorphologyPredictionResult:
    trench_depth: TrenchDepthResult
    post_etch_cd: PostEtchCDResult
    etched_image: np.ndarray
    depth_map: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trench_depth': self.trench_depth.to_dict(),
            'post_etch_cd': self.post_etch_cd.to_dict(),
        }


def _compute_arde_factor(cd_nm: float, depth_nm: float,
                         arde_coefficient: float) -> float:
    if cd_nm <= 0:
        return 0.0
    aspect_ratio = depth_nm / (cd_nm + 1e-10)
    return 1.0 / (1.0 + arde_coefficient * aspect_ratio ** 2)


def _compute_micro_loading(wafer_binary: np.ndarray,
                           pixel_size: float,
                           loading_factor: float) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    density = uniform_filter(wafer_binary.astype(np.float64), size=21, mode='wrap')
    return 1.0 + loading_factor * (density - 0.5)


def predict_trench_depth(wafer_binary: np.ndarray,
                         config: EtchProcessConfig) -> TrenchDepthResult:
    wafer_bin = (wafer_binary >= 0.5).astype(np.float64)
    dist = distance_transform_edt(wafer_bin)

    nominal_depth = config.etch_rate_si_nm_per_s * config.etch_time_s
    resist_consumed = config.etch_rate_resist_nm_per_s * config.etch_time_s
    resist_consumed = min(resist_consumed, config.resist_thickness_nm)

    loading_map = _compute_micro_loading(wafer_bin, config.pixel_size,
                                         config.micro_loading_factor)

    depth_map = np.zeros_like(wafer_bin, dtype=np.float64)
    trench_mask = wafer_bin < 0.5
    depth_map[trench_mask] = nominal_depth * loading_map[trench_mask]

    if config.aspect_ratio_dependent_etch:
        from core.litho_metrics import compute_cd
        cd_info = compute_cd(wafer_bin, direction='both', pixel_size=config.pixel_size)
        cd_mean = cd_info['cd_mean']
        if cd_mean > 0:
            arde = _compute_arde_factor(cd_mean, nominal_depth, config.arde_coefficient)
            depth_map[trench_mask] *= arde

    if config.trench_profile == TrenchProfileModel.ROUNDED_BOTTOM:
        depth_map *= (1.0 - 0.1 * np.exp(-dist / (config.pixel_size * 5 + 1e-10)))
    elif config.trench_profile == TrenchProfileModel.BOWING:
        bowing_profile = config.bowing_factor * np.sin(np.pi * dist /
                                                        (dist.max() + 1e-10) * 0.5)
        depth_map += bowing_profile * nominal_depth * 0.05

    depth_map_nm = depth_map * config.pixel_size if depth_map.max() < 1e3 else depth_map
    trench_depths = depth_map_nm[trench_mask]

    if len(trench_depths) > 0:
        depth_mean = float(np.mean(trench_depths))
        depth_min = float(np.min(trench_depths))
        depth_max = float(np.max(trench_depths))
        depth_std = float(np.std(trench_depths))
    else:
        depth_mean = depth_min = depth_max = depth_std = 0.0

    return TrenchDepthResult(
        depth_map=depth_map,
        depth_mean_nm=depth_mean,
        depth_min_nm=depth_min,
        depth_max_nm=depth_max,
        depth_std_nm=depth_std,
        resist_consumed_nm=resist_consumed,
        etch_time_s=config.etch_time_s,
    )


def predict_post_etch_cd(wafer_binary: np.ndarray,
                         etched_binary: np.ndarray,
                         config: EtchProcessConfig) -> PostEtchCDResult:
    from core.litho_metrics import compute_cd

    cd_resist_info = compute_cd(wafer_binary, direction='both', pixel_size=config.pixel_size)
    cd_etch_info = compute_cd(etched_binary, direction='both', pixel_size=config.pixel_size)

    cd_resist = cd_resist_info['cd_mean']
    cd_etch = cd_etch_info['cd_mean']
    cd_shift = cd_etch - cd_resist

    tan_angle = np.tan(np.deg2rad(config.sidewall_angle_deg))
    nominal_depth = config.etch_rate_si_nm_per_s * config.etch_time_s
    lateral_shift = nominal_depth / (tan_angle + 1e-10)
    cd_top = cd_etch + 2 * lateral_shift
    cd_bottom = cd_etch - 2 * lateral_shift
    cd_bottom = max(0.0, cd_bottom)

    return PostEtchCDResult(
        cd_resist_nm=cd_resist,
        cd_etch_nm=cd_etch,
        cd_shift_nm=cd_shift,
        cd_top_nm=cd_top,
        cd_bottom_nm=cd_bottom,
        trench_profile=config.trench_profile.value,
        sidewall_angle_deg=config.sidewall_angle_deg,
    )


class MorphologyPredictor:
    def __init__(self, etch_config: Optional[EtchProcessConfig] = None):
        self.etch_config = etch_config or EtchProcessConfig()

    def predict_depth(self, wafer_binary: np.ndarray) -> TrenchDepthResult:
        return predict_trench_depth(wafer_binary, self.etch_config)

    def predict_cd(self, wafer_binary: np.ndarray,
                   etched_binary: np.ndarray) -> PostEtchCDResult:
        return predict_post_etch_cd(wafer_binary, etched_binary, self.etch_config)

    def predict(self, wafer_binary: np.ndarray,
                etched_binary: np.ndarray) -> MorphologyPredictionResult:
        depth_result = self.predict_depth(wafer_binary)
        cd_result = self.predict_cd(wafer_binary, etched_binary)
        return MorphologyPredictionResult(
            trench_depth=depth_result,
            post_etch_cd=cd_result,
            etched_image=etched_binary,
            depth_map=depth_result.depth_map,
        )
