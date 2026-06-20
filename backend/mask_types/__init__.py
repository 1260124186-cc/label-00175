# -*- coding: utf-8 -*-
"""
掩模类型模块

提供各种光刻掩模类型的复数透过率建模，包括：
- 相位偏移掩模 (PSM)
  - 交替 PSM (Alt-PSM)
  - 衰减式 PSM (Att-PSM)
  - 连续相位掩模
  - 幅度-相位联合掩模
- 成像系统集成封装
"""

from .psm import (
    MaskType,
    PSMConfig,
    PhaseShiftMask,
    BinaryMask,
    AlternatingPSM,
    AttenuatedPSM,
    ContinuousPhaseMask,
    AmplitudePhaseMask,
    create_mask_model,
    compute_complex_gradient,
    verify_gradient_numerical,
    PSMImagingWrapper,
    PhaseOnlyImagingWrapper,
    AmplitudePhaseImagingWrapper,
    verify_end_to_end_gradient_numerical,
)

__all__ = [
    'MaskType',
    'PSMConfig',
    'PhaseShiftMask',
    'BinaryMask',
    'AlternatingPSM',
    'AttenuatedPSM',
    'ContinuousPhaseMask',
    'AmplitudePhaseMask',
    'create_mask_model',
    'compute_complex_gradient',
    'verify_gradient_numerical',
    'PSMImagingWrapper',
    'PhaseOnlyImagingWrapper',
    'AmplitudePhaseImagingWrapper',
    'verify_end_to_end_gradient_numerical',
]
