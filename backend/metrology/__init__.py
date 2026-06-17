# -*- coding: utf-8 -*-
"""
计量与 SEM/CD-SEM 仿真对接模块

模拟晶圆成像后的计量过程，连接仿真结果与工艺可制造性评估。

子模块：
- cd_extraction: CD 提取算法（Threshold Crossing、Derivative Peak 等）
- sem_simulation: SEM/CD-SEM 图像仿真（边缘增强、噪声、电子束模糊）
- metrics_report: 计量一致性报告（Uniformity、Linearity、Cp/Cpk 等）
"""

try:
    from metrology.cd_extraction import (
        CDExtractionMethod,
        MeasurementLine,
        CDExtractionResult,
        extract_profile,
        cd_threshold_crossing,
        cd_derivative_peak,
        cd_linear_regression,
        cd_polynomial_fit,
        extract_cd,
        extract_cd_multiline,
    )
    from metrology.sem_simulation import (
        SEMDetectorMode,
        SEMSimConfig,
        SEMSimResult,
        simulate_sem_image,
        simulate_cd_sem_line_scan,
        apply_charging_effect,
    )
    from metrology.metrics_report import (
        CDTarget,
        UniformityMetrics,
        LinearityMetrics,
        PrecisionMetrics,
        ProcessCapabilityMetrics,
        CDMeasurementPoint,
        MetrologyReport,
        compute_uniformity,
        compute_linearity,
        compute_precision,
        compute_process_capability,
        generate_metrology_report,
    )
except ImportError:
    from .cd_extraction import (
        CDExtractionMethod,
        MeasurementLine,
        CDExtractionResult,
        extract_profile,
        cd_threshold_crossing,
        cd_derivative_peak,
        cd_linear_regression,
        cd_polynomial_fit,
        extract_cd,
        extract_cd_multiline,
    )
    from .sem_simulation import (
        SEMDetectorMode,
        SEMSimConfig,
        SEMSimResult,
        simulate_sem_image,
        simulate_cd_sem_line_scan,
        apply_charging_effect,
    )
    from .metrics_report import (
        CDTarget,
        UniformityMetrics,
        LinearityMetrics,
        PrecisionMetrics,
        ProcessCapabilityMetrics,
        CDMeasurementPoint,
        MetrologyReport,
        compute_uniformity,
        compute_linearity,
        compute_precision,
        compute_process_capability,
        generate_metrology_report,
    )

__all__ = [
    'CDExtractionMethod',
    'MeasurementLine',
    'CDExtractionResult',
    'extract_profile',
    'cd_threshold_crossing',
    'cd_derivative_peak',
    'cd_linear_regression',
    'cd_polynomial_fit',
    'extract_cd',
    'extract_cd_multiline',
    'SEMDetectorMode',
    'SEMSimConfig',
    'SEMSimResult',
    'simulate_sem_image',
    'simulate_cd_sem_line_scan',
    'apply_charging_effect',
    'CDTarget',
    'UniformityMetrics',
    'LinearityMetrics',
    'PrecisionMetrics',
    'ProcessCapabilityMetrics',
    'CDMeasurementPoint',
    'MetrologyReport',
    'compute_uniformity',
    'compute_linearity',
    'compute_precision',
    'compute_process_capability',
    'generate_metrology_report',
]
