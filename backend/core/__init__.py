# -*- coding: utf-8 -*-
"""
核心模块：光学成像、傅里叶变换、误差评估
"""

from core.imaging import (
    OpticalSystem,
    PartialCoherentImaging,
    simulate_wafer_image,
    IlluminationType,
    generate_source,
    compute_tcc_kernel_2d,
    compute_tcc_full,
    socs_decomposition,
    ProcessCondition,
    ProcessWindow,
    MultiProcessSimulationResult,
    simulate_multi_process,
    create_focus_dose_window,
    create_full_process_window,
    ResistType,
    ResistThresholdMode,
    ResistModel,
    apply_resist_model,
    downsample_mask,
    upsample_mask,
    build_pyramid_scales,
    split_tiles,
    merge_tiles_with_blend
)
from core.fft import fft2d, ifft2d, fft1d, ifft1d, frequency_filter, phase_modulation
from core.metrics import mse, mae, ssim, normalized_correlation, batch_evaluate

__all__ = [
    'OpticalSystem',
    'PartialCoherentImaging',
    'simulate_wafer_image',
    'IlluminationType',
    'generate_source',
    'compute_tcc_kernel_2d',
    'compute_tcc_full',
    'socs_decomposition',
    'ProcessCondition',
    'ProcessWindow',
    'MultiProcessSimulationResult',
    'simulate_multi_process',
    'create_focus_dose_window',
    'create_full_process_window',
    'ResistType',
    'ResistThresholdMode',
    'ResistModel',
    'apply_resist_model',
    'downsample_mask',
    'upsample_mask',
    'build_pyramid_scales',
    'split_tiles',
    'merge_tiles_with_blend',
    'fft2d',
    'ifft2d',
    'fft1d',
    'ifft1d',
    'frequency_filter',
    'phase_modulation',
    'mse',
    'mae',
    'ssim',
    'normalized_correlation',
    'batch_evaluate'
]
