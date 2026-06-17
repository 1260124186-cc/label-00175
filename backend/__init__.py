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

import sys as _sys
from pathlib import Path as _Path

# ---------------------------------------------------------------------------
# 路径兼容：当以 `python -m backend.xxx` 方式从仓库根目录启动时，
# 先将 backend/ 自身加入 sys.path，确保所有子包（core / utils / workflows ...）
# 内的裸导入（如 `from core.imaging import ...`）能正常工作。
# 同时也保留了从 backend/ 目录直接启动的传统方式。
# ---------------------------------------------------------------------------
_BACKEND_DIR = str(_Path(__file__).resolve().parent)
if _BACKEND_DIR not in _sys.path:
    _sys.path.insert(0, _BACKEND_DIR)

__version__ = '1.0.0'
__author__ = 'Lithography Simulation Team'

try:
    from core import (
        OpticalSystem,
        PartialCoherentImaging,
        simulate_wafer_image,
        fft2d,
        ifft2d,
        mse,
        mae,
        ssim
    )

    from algorithms import (
        GradientDescentOptimizer,
        BFGSOptimizer,
        MaskOptimizer,
        OptimizationConfig
    )

    from utils import (
        load_image,
        load_gds_layer,
        save_image,
        load_config,
        setup_logger
    )
except ImportError:
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
        load_gds_layer,
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
    'load_gds_layer',
    'save_image',
    'load_config',
    'setup_logger'
]
