# -*- coding: utf-8 -*-
"""
掩模制造(Mask Manufacturing)成本评估模块

评估掩模图案从版图设计到掩模写入(Mask Writer)的制造复杂度与成本，
核心指标包括：
    1. 多边形顶点数量 (Polygon Vertex Count)
    2. 曼哈顿化矩形 Shot 数量 (Manhattanized Shot Count)
    3. 数据体积 (Data Volume)
    4. 预估写入时间 (Estimated Write Time)

所有指标综合为 制造复杂度分数 (Manufacturing Complexity Score)，
可作为 SMO/ILT 等掩模优化流程中的附加惩罚项，平衡成像质量与掩模厂制造成本。

主要组件：
    - ManufacturingCostConfig: 成本评估配置
    - ManufacturingCostResult: 成本评估结果数据结构
    - MaskManufacturingCostEvaluator: 掩模制造成本评估器核心类
    - MaskManufacturingPenalty: 可微惩罚项封装，用于 SMO/ILT 优化
"""

from .cost_evaluator import (
    ManufacturingCostConfig,
    ManufacturingCostResult,
    MaskManufacturingCostEvaluator,
    MaskWriterType,
    ShotFracturingStrategy,
    RectangleShot,
    estimate_vertex_count,
    manhattanize_polygon,
    fracturing_to_shots,
    estimate_shot_count,
    estimate_data_volume,
    estimate_write_time,
    compute_complexity_score,
)

from .penalty import (
    ManufacturingPenaltyConfig,
    MaskManufacturingPenalty,
    compute_vertex_penalty,
    compute_shot_penalty,
    compute_data_penalty,
    compute_write_time_penalty,
    compute_manufacturing_penalty,
    compute_manufacturing_penalty_gradient,
)

__all__ = [
    'ManufacturingCostConfig',
    'ManufacturingCostResult',
    'MaskManufacturingCostEvaluator',
    'MaskWriterType',
    'ShotFracturingStrategy',
    'RectangleShot',
    'estimate_vertex_count',
    'manhattanize_polygon',
    'fracturing_to_shots',
    'estimate_shot_count',
    'estimate_data_volume',
    'estimate_write_time',
    'compute_complexity_score',
    'ManufacturingPenaltyConfig',
    'MaskManufacturingPenalty',
    'compute_vertex_penalty',
    'compute_shot_penalty',
    'compute_data_penalty',
    'compute_write_time_penalty',
    'compute_manufacturing_penalty',
    'compute_manufacturing_penalty_gradient',
]
