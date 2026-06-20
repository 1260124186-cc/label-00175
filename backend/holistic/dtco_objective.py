# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DTCOObjectiveMode(Enum):
    POST_ETCH_MSE = "post_etch_mse"
    POST_ETCH_EPE = "post_etch_epe"
    POST_ETCH_CD_ERROR = "post_etch_cd_error"
    COMPOSITE = "composite"


@dataclass
class DTCOObjectiveConfig:
    mode: DTCOObjectiveMode = DTCOObjectiveMode.COMPOSITE
    weight_mse: float = 1.0
    weight_epe: float = 2.0
    weight_cd_error: float = 3.0
    weight_depth_error: float = 1.0
    weight_sidewall_angle: float = 0.5
    target_depth_nm: float = 500.0
    target_cd_nm: float = 40.0
    target_sidewall_angle_deg: float = 90.0
    depth_tolerance_nm: float = 50.0
    cd_tolerance_nm: float = 4.0
    pixel_size: float = 1.0
    include_resist_matching: bool = True
    resist_weight: float = 0.3
    etch_weight: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.value,
            'weight_mse': self.weight_mse,
            'weight_epe': self.weight_epe,
            'weight_cd_error': self.weight_cd_error,
            'weight_depth_error': self.weight_depth_error,
            'weight_sidewall_angle': self.weight_sidewall_angle,
            'target_depth_nm': self.target_depth_nm,
            'target_cd_nm': self.target_cd_nm,
            'target_sidewall_angle_deg': self.target_sidewall_angle_deg,
            'depth_tolerance_nm': self.depth_tolerance_nm,
            'cd_tolerance_nm': self.cd_tolerance_nm,
            'pixel_size': self.pixel_size,
            'include_resist_matching': self.include_resist_matching,
            'resist_weight': self.resist_weight,
            'etch_weight': self.etch_weight,
        }


@dataclass
class DTCOObjectiveResult:
    total_loss: float
    mse_loss: float
    epe_loss: float
    cd_error_loss: float
    depth_error_loss: float
    sidewall_angle_loss: float
    resist_loss: float
    etch_loss: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_loss': self.total_loss,
            'mse_loss': self.mse_loss,
            'epe_loss': self.epe_loss,
            'cd_error_loss': self.cd_error_loss,
            'depth_error_loss': self.depth_error_loss,
            'sidewall_angle_loss': self.sidewall_angle_loss,
            'resist_loss': self.resist_loss,
            'etch_loss': self.etch_loss,
            'metrics': self.metrics,
        }

    def summary(self) -> str:
        lines = [
            "=== DTCO Objective ===",
            f"Total Loss: {self.total_loss:.6f}",
            f"  MSE Loss: {self.mse_loss:.6f}",
            f"  EPE Loss: {self.epe_loss:.6f}",
            f"  CD Error Loss: {self.cd_error_loss:.6f}",
            f"  Depth Error Loss: {self.depth_error_loss:.6f}",
            f"  Sidewall Angle Loss: {self.sidewall_angle_loss:.6f}",
            f"  Resist Loss: {self.resist_loss:.6f}",
            f"  Etch Loss: {self.etch_loss:.6f}",
        ]
        return "\n".join(lines)


def _compute_mse_loss(predicted: np.ndarray, target: np.ndarray) -> float:
    diff = predicted.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(diff ** 2))


def _compute_epe_loss(wafer_binary: np.ndarray,
                      target_binary: np.ndarray,
                      pixel_size: float = 1.0) -> float:
    from core.litho_metrics import compute_epe
    epe_info = compute_epe(wafer_binary, target_binary, pixel_size=pixel_size)
    return epe_info.get('epe_mean', 0.0)


def _compute_cd_error_loss(wafer_binary: np.ndarray,
                           target_binary: np.ndarray,
                           target_cd_nm: float,
                           pixel_size: float = 1.0) -> float:
    from core.litho_metrics import compute_cd_error
    cd_err = compute_cd_error(wafer_binary, target_binary, pixel_size=pixel_size)
    absolute_cd_error = abs(cd_err.get('cd_error_mean', 0.0))
    relative_error = absolute_cd_error / (target_cd_nm + 1e-10)
    return float(relative_error)


def _compute_depth_loss(depth_mean_nm: float,
                        target_depth_nm: float,
                        tolerance_nm: float) -> float:
    error = abs(depth_mean_nm - target_depth_nm)
    if error <= tolerance_nm:
        return 0.0
    return float((error - tolerance_nm) / (target_depth_nm + 1e-10))


def _compute_sidewall_angle_loss(measured_angle_deg: float,
                                 target_angle_deg: float) -> float:
    error = abs(measured_angle_deg - target_angle_deg)
    return float(error / 90.0)


def compute_post_etch_mse_objective(etched_image: np.ndarray,
                                    target_image: np.ndarray,
                                    config: DTCOObjectiveConfig) -> DTCOObjectiveResult:
    mse_loss = _compute_mse_loss(etched_image, target_image)
    return DTCOObjectiveResult(
        total_loss=mse_loss,
        mse_loss=mse_loss,
        epe_loss=0.0,
        cd_error_loss=0.0,
        depth_error_loss=0.0,
        sidewall_angle_loss=0.0,
        resist_loss=0.0,
        etch_loss=mse_loss,
    )


def compute_post_etch_epe_objective(etched_image: np.ndarray,
                                    target_image: np.ndarray,
                                    config: DTCOObjectiveConfig) -> DTCOObjectiveResult:
    epe_loss = _compute_epe_loss(etched_image, target_image, config.pixel_size)
    return DTCOObjectiveResult(
        total_loss=epe_loss,
        mse_loss=0.0,
        epe_loss=epe_loss,
        cd_error_loss=0.0,
        depth_error_loss=0.0,
        sidewall_angle_loss=0.0,
        resist_loss=0.0,
        etch_loss=epe_loss,
    )


def compute_post_etch_cd_objective(etched_image: np.ndarray,
                                   target_image: np.ndarray,
                                   config: DTCOObjectiveConfig) -> DTCOObjectiveResult:
    cd_loss = _compute_cd_error_loss(etched_image, target_image,
                                     config.target_cd_nm, config.pixel_size)
    return DTCOObjectiveResult(
        total_loss=cd_loss,
        mse_loss=0.0,
        epe_loss=0.0,
        cd_error_loss=cd_loss,
        depth_error_loss=0.0,
        sidewall_angle_loss=0.0,
        resist_loss=0.0,
        etch_loss=cd_loss,
    )


def compute_composite_objective(wafer_binary: np.ndarray,
                                etched_image: np.ndarray,
                                target_binary: np.ndarray,
                                depth_mean_nm: float,
                                sidewall_angle_deg: float,
                                config: DTCOObjectiveConfig) -> DTCOObjectiveResult:
    mse_loss = _compute_mse_loss(etched_image, target_binary)
    epe_loss = _compute_epe_loss(etched_image, target_binary, config.pixel_size)
    cd_error_loss = _compute_cd_error_loss(etched_image, target_binary,
                                           config.target_cd_nm, config.pixel_size)
    depth_loss = _compute_depth_loss(depth_mean_nm, config.target_depth_nm,
                                     config.depth_tolerance_nm)
    angle_loss = _compute_sidewall_angle_loss(sidewall_angle_deg,
                                              config.target_sidewall_angle_deg)

    etch_loss = (config.weight_mse * mse_loss +
                 config.weight_epe * epe_loss / (config.pixel_size + 1e-10) +
                 config.weight_cd_error * cd_error_loss +
                 config.weight_depth_error * depth_loss +
                 config.weight_sidewall_angle * angle_loss)

    resist_loss = 0.0
    if config.include_resist_matching:
        resist_mse = _compute_mse_loss(wafer_binary, target_binary)
        resist_epe = _compute_epe_loss(wafer_binary, target_binary, config.pixel_size)
        resist_loss = config.weight_mse * resist_mse + config.weight_epe * resist_epe

    total_loss = config.etch_weight * etch_loss + config.resist_weight * resist_loss

    metrics = {
        'cd_etch_nm': 0.0,
        'cd_target_nm': config.target_cd_nm,
        'depth_mean_nm': depth_mean_nm,
        'depth_target_nm': config.target_depth_nm,
        'sidewall_angle_deg': sidewall_angle_deg,
    }

    return DTCOObjectiveResult(
        total_loss=total_loss,
        mse_loss=mse_loss,
        epe_loss=epe_loss,
        cd_error_loss=cd_error_loss,
        depth_error_loss=depth_loss,
        sidewall_angle_loss=angle_loss,
        resist_loss=resist_loss,
        etch_loss=etch_loss,
        metrics=metrics,
    )


class DTCOObjective:
    def __init__(self, config: Optional[DTCOObjectiveConfig] = None):
        self.config = config or DTCOObjectiveConfig()

    def evaluate(self,
                 wafer_binary: np.ndarray,
                 etched_image: np.ndarray,
                 target_binary: np.ndarray,
                 depth_mean_nm: float = 0.0,
                 sidewall_angle_deg: float = 90.0) -> DTCOObjectiveResult:
        if self.config.mode == DTCOObjectiveMode.POST_ETCH_MSE:
            return compute_post_etch_mse_objective(etched_image, target_binary, self.config)
        elif self.config.mode == DTCOObjectiveMode.POST_ETCH_EPE:
            return compute_post_etch_epe_objective(etched_image, target_binary, self.config)
        elif self.config.mode == DTCOObjectiveMode.POST_ETCH_CD_ERROR:
            return compute_post_etch_cd_objective(etched_image, target_binary, self.config)
        elif self.config.mode == DTCOObjectiveMode.COMPOSITE:
            return compute_composite_objective(
                wafer_binary, etched_image, target_binary,
                depth_mean_nm, sidewall_angle_deg, self.config
            )
        else:
            raise ValueError(f"Unknown DTCO objective mode: {self.config.mode}")

    def as_loss_function(self,
                         target_binary: np.ndarray,
                         simulate_fn: Callable,
                         etch_pipeline_fn: Callable) -> Callable:
        def loss_fn(mask: np.ndarray) -> float:
            wafer_binary = simulate_fn(mask)
            etched_image, depth_nm, sidewall_deg = etch_pipeline_fn(wafer_binary)
            result = self.evaluate(wafer_binary, etched_image, target_binary,
                                   depth_nm, sidewall_deg)
            return result.total_loss
        return loss_fn
