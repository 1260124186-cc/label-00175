# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from holistic.bias_model import (
    EtchBiasModel,
    BiasModelConfig,
    BiasModelType,
    BiasModelResult,
)
from holistic.anisotropic_etch import (
    AnisotropicEtchModel,
    AnisotropicEtchConfig,
    EtchAnisotropyMode,
    AnisotropicEtchResult,
)
from holistic.morphology_predictor import (
    MorphologyPredictor,
    EtchProcessConfig,
    TrenchProfileModel,
    MorphologyPredictionResult,
    TrenchDepthResult,
    PostEtchCDResult,
)
from holistic.dtco_objective import (
    DTCOObjective,
    DTCOObjectiveConfig,
    DTCOObjectiveMode,
    DTCOObjectiveResult,
)

logger = logging.getLogger(__name__)


class CoSimPipelineMode(Enum):
    BIAS_ONLY = "bias_only"
    ANISOTROPIC_ONLY = "anisotropic_only"
    BIAS_THEN_ANISOTROPIC = "bias_then_anisotropic"
    ANISOTROPIC_THEN_BIAS = "anisotropic_then_bias"
    FULL = "full"


@dataclass
class CoSimConfig:
    pipeline_mode: CoSimPipelineMode = CoSimPipelineMode.FULL
    bias_config: BiasModelConfig = field(default_factory=BiasModelConfig)
    anisotropic_config: AnisotropicEtchConfig = field(default_factory=AnisotropicEtchConfig)
    etch_config: EtchProcessConfig = field(default_factory=EtchProcessConfig)
    dtco_config: DTCOObjectiveConfig = field(default_factory=DTCOObjectiveConfig)
    enable_morphology_prediction: bool = True
    enable_dtco_evaluation: bool = True
    pixel_size: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pipeline_mode': self.pipeline_mode.value,
            'bias_config': self.bias_config.to_dict(),
            'anisotropic_config': self.anisotropic_config.to_dict(),
            'etch_config': self.etch_config.to_dict(),
            'dtco_config': self.dtco_config.to_dict(),
            'enable_morphology_prediction': self.enable_morphology_prediction,
            'enable_dtco_evaluation': self.enable_dtco_evaluation,
            'pixel_size': self.pixel_size,
        }


@dataclass
class CoSimStepResult:
    step_name: str
    input_image: np.ndarray
    output_image: np.ndarray
    step_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoSimResult:
    wafer_image: np.ndarray
    etched_image: np.ndarray
    depth_map: np.ndarray
    depth_mean_nm: float
    post_etch_cd_nm: float
    cd_shift_nm: float
    sidewall_angle_deg: float
    dtco_result: Optional[DTCOObjectiveResult] = None
    morphology_result: Optional[MorphologyPredictionResult] = None
    step_results: List[CoSimStepResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'depth_mean_nm': self.depth_mean_nm,
            'post_etch_cd_nm': self.post_etch_cd_nm,
            'cd_shift_nm': self.cd_shift_nm,
            'sidewall_angle_deg': self.sidewall_angle_deg,
        }
        if self.dtco_result is not None:
            result['dtco'] = self.dtco_result.to_dict()
        if self.morphology_result is not None:
            result['morphology'] = self.morphology_result.to_dict()
        return result

    def summary(self) -> str:
        lines = [
            "=== Litho-Etch Co-Simulation Result ===",
            f"Trench Depth: {self.depth_mean_nm:.1f} nm",
            f"Post-Etch CD: {self.post_etch_cd_nm:.1f} nm",
            f"CD Shift: {self.cd_shift_nm:.1f} nm",
            f"Sidewall Angle: {self.sidewall_angle_deg:.1f} deg",
        ]
        if self.dtco_result is not None:
            lines.append(self.dtco_result.summary())
        return "\n".join(lines)


class LithoEtchCoSimPipeline:
    def __init__(self, config: Optional[CoSimConfig] = None):
        self.config = config or CoSimConfig()
        self._bias_model = EtchBiasModel(self.config.bias_config)
        self._anisotropic_model = AnisotropicEtchModel(self.config.anisotropic_config)
        self._morphology_predictor = MorphologyPredictor(self.config.etch_config)
        self._dtco_objective = DTCOObjective(self.config.dtco_config)

    def _apply_bias_step(self, wafer_binary: np.ndarray,
                         step_results: List[CoSimStepResult]) -> np.ndarray:
        bias_result = self._bias_model.apply(wafer_binary)
        step_results.append(CoSimStepResult(
            step_name="bias_etch",
            input_image=wafer_binary.copy(),
            output_image=bias_result.biased_image.copy(),
            step_metrics=bias_result.to_dict(),
        ))
        return bias_result.biased_image

    def _apply_anisotropic_step(self, wafer_binary: np.ndarray,
                                step_results: List[CoSimStepResult]) -> np.ndarray:
        aniso_result = self._anisotropic_model.apply(wafer_binary)
        step_results.append(CoSimStepResult(
            step_name="anisotropic_etch",
            input_image=wafer_binary.copy(),
            output_image=aniso_result.etched_image.copy(),
            step_metrics=aniso_result.to_dict(),
        ))
        return aniso_result.etched_image

    def run(self,
            wafer_image: np.ndarray,
            target_image: Optional[np.ndarray] = None) -> CoSimResult:
        wafer_bin = (wafer_image >= 0.5).astype(np.float64)
        step_results: List[CoSimStepResult] = []

        step_results.append(CoSimStepResult(
            step_name="input_wafer",
            input_image=wafer_bin.copy(),
            output_image=wafer_bin.copy(),
            step_metrics={},
        ))

        current = wafer_bin.copy()
        mode = self.config.pipeline_mode

        if mode == CoSimPipelineMode.BIAS_ONLY:
            current = self._apply_bias_step(current, step_results)
        elif mode == CoSimPipelineMode.ANISOTROPIC_ONLY:
            current = self._apply_anisotropic_step(current, step_results)
        elif mode == CoSimPipelineMode.BIAS_THEN_ANISOTROPIC:
            current = self._apply_bias_step(current, step_results)
            current = self._apply_anisotropic_step(current, step_results)
        elif mode == CoSimPipelineMode.ANISOTROPIC_THEN_BIAS:
            current = self._apply_anisotropic_step(current, step_results)
            current = self._apply_bias_step(current, step_results)
        elif mode == CoSimPipelineMode.FULL:
            current = self._apply_bias_step(current, step_results)
            current = self._apply_anisotropic_step(current, step_results)

        etched_image = current
        depth_result = self._morphology_predictor.predict_depth(wafer_bin)
        cd_result = self._morphology_predictor.predict_cd(wafer_bin, etched_image)

        morphology_result = None
        if self.config.enable_morphology_prediction:
            morphology_result = self._morphology_predictor.predict(wafer_bin, etched_image)

        dtco_result = None
        if self.config.enable_dtco_evaluation and target_image is not None:
            dtco_result = self._dtco_objective.evaluate(
                wafer_bin, etched_image, target_image,
                depth_result.depth_mean_nm,
                cd_result.sidewall_angle_deg,
            )

        return CoSimResult(
            wafer_image=wafer_bin,
            etched_image=etched_image,
            depth_map=depth_result.depth_map,
            depth_mean_nm=depth_result.depth_mean_nm,
            post_etch_cd_nm=cd_result.cd_etch_nm,
            cd_shift_nm=cd_result.cd_shift_nm,
            sidewall_angle_deg=cd_result.sidewall_angle_deg,
            dtco_result=dtco_result,
            morphology_result=morphology_result,
            step_results=step_results,
        )

    def run_batch(self,
                  wafer_images: List[np.ndarray],
                  target_image: Optional[np.ndarray] = None) -> List[CoSimResult]:
        return [self.run(img, target_image) for img in wafer_images]


def run_litho_etch_cosim(wafer_image: np.ndarray,
                          target_image: Optional[np.ndarray] = None,
                          config: Optional[CoSimConfig] = None) -> CoSimResult:
    pipeline = LithoEtchCoSimPipeline(config)
    return pipeline.run(wafer_image, target_image)


def create_dtco_aware_simulate_fn(optical_system,
                                   co_sim_config: Optional[CoSimConfig] = None) -> Callable:
    from core.imaging import PartialCoherentImaging, apply_resist_model

    config = co_sim_config or CoSimConfig()
    pipeline = LithoEtchCoSimPipeline(config)
    imaging_model = PartialCoherentImaging(optical_system, (256, 256))

    def simulate_fn(mask: np.ndarray) -> Tuple[np.ndarray, CoSimResult]:
        aerial = imaging_model.compute_aerial_image(mask)
        wafer = _apply_threshold(aerial, 0.3)
        co_sim_result = pipeline.run(wafer)
        return wafer, co_sim_result

    return simulate_fn


def _apply_threshold(image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    return (image >= threshold).astype(np.float64)
