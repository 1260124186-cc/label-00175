# -*- coding: utf-8 -*-
"""
RET 推荐引擎数据结构定义

定义版图特征、推荐结果、历史实验记录等核心数据结构。
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class RETStrategy(Enum):
    OPC_ONLY = 'opc_only'
    OPC_SRAF = 'opc_sraf'
    ILT = 'ilt'
    SMO_ILT = 'smo_ilt'


@dataclass
class SpectralFeatures:
    dominant_frequency: float = 0.0
    bandwidth_3db: float = 0.0
    spectral_entropy: float = 0.0
    high_freq_energy_ratio: float = 0.0
    low_freq_energy_ratio: float = 0.0
    peak_count: int = 0
    spectral_centroid: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dominant_frequency': self.dominant_frequency,
            'bandwidth_3db': self.bandwidth_3db,
            'spectral_entropy': self.spectral_entropy,
            'high_freq_energy_ratio': self.high_freq_energy_ratio,
            'low_freq_energy_ratio': self.low_freq_energy_ratio,
            'peak_count': self.peak_count,
            'spectral_centroid': self.spectral_centroid,
        }


@dataclass
class LayoutFeatures:
    min_cd_nm: float = 0.0
    corner_density: float = 0.0
    periodicity_score: float = 0.0
    dominant_pitch_nm: float = 0.0
    duty_cycle: float = 0.0
    fill_ratio: float = 0.0
    spectral: SpectralFeatures = field(default_factory=SpectralFeatures)
    technology_node: str = 'duv_arf'
    wavelength: float = 193.0
    na: float = 1.35
    pixel_size: float = 1.0
    image_shape: Tuple[int, int] = (0, 0)

    def k1_factor(self) -> float:
        if self.wavelength <= 0 or self.na <= 0:
            return float('inf')
        return self.min_cd_nm * self.na / self.wavelength

    def complexity_score(self) -> float:
        k1 = self.k1_factor()
        score = 0.0
        if k1 < 0.35:
            score += 0.4
        elif k1 < 0.5:
            score += 0.25
        elif k1 < 0.7:
            score += 0.1
        if self.corner_density > 0.05:
            score += 0.25
        elif self.corner_density > 0.02:
            score += 0.15
        if self.spectral.high_freq_energy_ratio > 0.3:
            score += 0.2
        elif self.spectral.high_freq_energy_ratio > 0.15:
            score += 0.1
        if self.periodicity_score < 0.3:
            score += 0.15
        return min(score, 1.0)

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.min_cd_nm / 100.0,
            self.k1_factor(),
            self.corner_density,
            self.periodicity_score,
            self.duty_cycle,
            self.fill_ratio,
            self.spectral.high_freq_energy_ratio,
            self.spectral.spectral_entropy,
            self.spectral.peak_count / 20.0,
            1.0 if self.technology_node == 'euv' else 0.0,
        ], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'min_cd_nm': self.min_cd_nm,
            'corner_density': self.corner_density,
            'periodicity_score': self.periodicity_score,
            'dominant_pitch_nm': self.dominant_pitch_nm,
            'duty_cycle': self.duty_cycle,
            'fill_ratio': self.fill_ratio,
            'spectral': self.spectral.to_dict(),
            'technology_node': self.technology_node,
            'wavelength': self.wavelength,
            'na': self.na,
            'pixel_size': self.pixel_size,
            'image_shape': list(self.image_shape),
            'k1_factor': self.k1_factor(),
            'complexity_score': self.complexity_score(),
        }


@dataclass
class RETRecommendation:
    strategy: RETStrategy = RETStrategy.OPC_ONLY
    confidence: float = 0.0
    reason: str = ''
    opc_params: Dict[str, Any] = field(default_factory=dict)
    ilt_params: Dict[str, Any] = field(default_factory=dict)
    smo_params: Dict[str, Any] = field(default_factory=dict)
    optical_system_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy.value,
            'confidence': self.confidence,
            'reason': self.reason,
            'opc_params': self.opc_params,
            'ilt_params': self.ilt_params,
            'smo_params': self.smo_params,
            'optical_system_hints': self.optical_system_hints,
        }


@dataclass
class RETRecommendationResult:
    primary: RETRecommendation = field(default_factory=RETRecommendation)
    alternatives: List[RETRecommendation] = field(default_factory=list)
    features: Optional[LayoutFeatures] = None
    matched_experiments: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary': self.primary.to_dict(),
            'alternatives': [a.to_dict() for a in self.alternatives],
            'features': self.features.to_dict() if self.features else None,
            'matched_experiments': self.matched_experiments,
            'warnings': self.warnings,
        }


@dataclass
class ExperimentRecord:
    id: str = ''
    layout_type: str = ''
    technology_node: str = 'duv_arf'
    wavelength: float = 193.0
    na: float = 1.35
    min_cd_nm: float = 45.0
    corner_density: float = 0.0
    periodicity_score: float = 0.0
    high_freq_energy_ratio: float = 0.0
    strategy: str = 'opc_only'
    final_epe_nm: float = 0.0
    epe_improvement_pct: float = 0.0
    convergence: bool = False
    total_time_sec: float = 0.0
    opc_params: Dict[str, Any] = field(default_factory=dict)
    ilt_params: Dict[str, Any] = field(default_factory=dict)
    smo_params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ''

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.min_cd_nm / 100.0,
            self.min_cd_nm * self.na / self.wavelength,
            self.corner_density,
            self.periodicity_score,
            0.5,
            0.5,
            self.high_freq_energy_ratio,
            0.5,
            0.5,
            1.0 if self.technology_node == 'euv' else 0.0,
        ], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'layout_type': self.layout_type,
            'technology_node': self.technology_node,
            'wavelength': self.wavelength,
            'na': self.na,
            'min_cd_nm': self.min_cd_nm,
            'corner_density': self.corner_density,
            'periodicity_score': self.periodicity_score,
            'high_freq_energy_ratio': self.high_freq_energy_ratio,
            'strategy': self.strategy,
            'final_epe_nm': self.final_epe_nm,
            'epe_improvement_pct': self.epe_improvement_pct,
            'convergence': self.convergence,
            'total_time_sec': self.total_time_sec,
            'opc_params': self.opc_params,
            'ilt_params': self.ilt_params,
            'smo_params': self.smo_params,
            'notes': self.notes,
        }
