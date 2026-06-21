# -*- coding: utf-8 -*-
"""
掩模检测图像仿真与可检测性分析模块

模拟掩模检测机台的成像过程，支持明场/暗场等多种检测模式，
实现 Die-to-Database 差异图计算，评估 OPC 后掩模的
可检测性与假缺陷率。

核心功能:
1. 检测图像仿真: 明场、暗场、相位对比、偏振检测模式
2. Die-to-Database: 差异图计算、候选缺陷提取
3. 可检测性分析: 检测率、假警报率、ROC/AUC、印刷性评估

使用示例::

    from inspection import (
        run_full_inspection_analysis,
        InspectionConfig,
        InspectionMode,
        InspectionAnalysisConfig,
    )

    # 创建配置
    config = InspectionAnalysisConfig(
        inspection_config=InspectionConfig(
            mode=InspectionMode.DARK_FIELD,
            defect_boost=2.0,
        )
    )

    # 运行完整分析
    result = run_full_inspection_analysis(
        mask_defective, mask_nominal, config
    )
    print(result.summary())
"""

from inspection.schemas import (
    InspectionMode,
    DefectClass,
    DieType,
    InspectionOptics,
    InspectionConfig,
    InspectionImageResult,
    DifferenceMapResult,
    DefectCandidate,
    DetectabilityResult,
    InspectionAnalysisConfig,
    FullInspectionResult,
)
from inspection.inspection_simulator import (
    simulate_inspection_image,
    simulate_multi_mode_inspection,
    compute_defect_contrast,
)
from inspection.die_to_database import (
    compute_difference_map,
    compute_detection_threshold,
    threshold_difference_map,
    extract_candidate_regions,
    compute_difference_histogram,
    compute_die_to_database,
    compute_die_to_database_from_result,
    compute_aligned_difference,
)
from inspection.detectability_analysis import (
    analyze_detectability,
    run_full_inspection_analysis,
    compute_false_defect_rate,
    evaluate_detection_performance,
)

__all__ = [
    'InspectionMode',
    'DefectClass',
    'DieType',
    'InspectionOptics',
    'InspectionConfig',
    'InspectionImageResult',
    'DifferenceMapResult',
    'DefectCandidate',
    'DetectabilityResult',
    'InspectionAnalysisConfig',
    'FullInspectionResult',
    'simulate_inspection_image',
    'simulate_multi_mode_inspection',
    'compute_defect_contrast',
    'compute_difference_map',
    'compute_detection_threshold',
    'threshold_difference_map',
    'extract_candidate_regions',
    'compute_difference_histogram',
    'compute_die_to_database',
    'compute_die_to_database_from_result',
    'compute_aligned_difference',
    'analyze_detectability',
    'run_full_inspection_analysis',
    'compute_false_defect_rate',
    'evaluate_detection_performance',
]
