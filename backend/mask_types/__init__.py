# -*- coding: utf-8 -*-
"""
掩模类型模块

提供各种光刻掩模类型的复数透过率建模，包括：
- 相位偏移掩模 (PSM)
  - 交替 PSM (Alt-PSM)
  - 衰减式 PSM (Att-PSM)
"""

from .psm import (
    MaskType,
    PhaseShiftMask,
    AlternatingPSM,
    AttenuatedPSM,
    BinaryMask,
    create_mask_model,
)

__all__ = [
    'MaskType',
    'PhaseShiftMask',
    'AlternatingPSM',
    'AttenuatedPSM',
    'BinaryMask',
    'create_mask_model',
]
