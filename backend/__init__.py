# -*- coding: utf-8 -*-
"""
计算光刻与版图优化仿真框架

该框架用于掩模图案优化的代码开发与算法验证，核心功能包括：
- 光学成像建模（部分相干成像模型）
- 傅里叶变换计算
- 优化算法迭代
- 误差评估（MSE、MAE、SSIM等）

Author: Lithography Simulation Team
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Lithography Simulation Team'

from .core import (
    OpticalSystem,
    PartialCoherentImaging,
    simulate_wafer_image,
    fft2d,
    ifft2d,
    mse,
    mae,
    ssim
)

from .algorithms import (
    GradientDescentOptimizer,
    BFGSOptimizer,
    MaskOptimizer,
    OptimizationConfig
)

from .utils import (
    load_image,
    save_image,
    load_config,
    setup_logger
)

__all__ = [
    # Core
    'OpticalSystem',
    'PartialCoherentImaging',
    'simulate_wafer_image',
    'fft2d',
    'ifft2d',
    'mse',
    'mae',
    'ssim',
    # Algorithms
    'GradientDescentOptimizer',
    'BFGSOptimizer',
    'MaskOptimizer',
    'OptimizationConfig',
    # Utils
    'load_image',
    'save_image',
    'load_config',
    'setup_logger'
]
