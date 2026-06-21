# -*- coding: utf-8 -*-
"""
Fab 闭环反馈校准模块
Fab Closed-Loop Feedback Calibration Module

形成 "仿真 → 量产 → 反馈 → 再仿真" 的数据闭环：
1. 定期从 Fab 导入最新 CD-SEM 量测 CSV
2. 与当前仿真预测对比，计算偏差统计
3. 偏差超阈值时自动触发 calibration 模块，更新模型参数
4. 校准完成后重新评估所有在产掩模的 PW 余量

子模块：
- schemas: 数据结构定义
- fab_importer: Fab CD-SEM CSV 数据导入（增量、归档、历史记录）
- comparator: 仿真预测 vs 量产量测对比分析
- calibration_trigger: 自动触发 calibration 模块
- pw_reassessor: 在产掩模 PW 余量重评估
- pipeline: 端到端闭环流水线
- scheduler: 定期调度器
"""

from .schemas import (
    ClosedLoopState,
    CalibrationTriggerReason,
    MaskPriority,
    FabImportConfig,
    ImportedFileRecord,
    FabImportResult,
    PerPointComparison,
    PatternGroupStats,
    ComparisonResult,
    CalibrationTriggerThresholds,
    CalibrationTriggerResult,
    ProductionMask,
    PWReassessmentResult,
    ClosedLoopConfig,
    ClosedLoopCycle,
)
from .fab_importer import (
    FabDataImporter,
    import_fab_data,
)
from .comparator import (
    PredictionComparator,
    compare_prediction_vs_measurement,
)
from .calibration_trigger import (
    CalibrationTrigger,
    evaluate_and_trigger_calibration,
)
from .pw_reassessor import (
    PWReassessor,
    reevaluate_production_masks,
)
from .pipeline import (
    ClosedLoopPipeline,
    run_closed_loop_cycle,
)
from .scheduler import (
    SchedulerState,
    SchedulerRunRecord,
    SchedulerConfig,
    ClosedLoopScheduler,
    create_scheduler,
)

__all__ = [
    'ClosedLoopState',
    'CalibrationTriggerReason',
    'MaskPriority',
    'FabImportConfig',
    'ImportedFileRecord',
    'FabImportResult',
    'PerPointComparison',
    'PatternGroupStats',
    'ComparisonResult',
    'CalibrationTriggerThresholds',
    'CalibrationTriggerResult',
    'ProductionMask',
    'PWReassessmentResult',
    'ClosedLoopConfig',
    'ClosedLoopCycle',
    'FabDataImporter',
    'import_fab_data',
    'PredictionComparator',
    'compare_prediction_vs_measurement',
    'CalibrationTrigger',
    'evaluate_and_trigger_calibration',
    'PWReassessor',
    'reevaluate_production_masks',
    'ClosedLoopPipeline',
    'run_closed_loop_cycle',
    'SchedulerState',
    'SchedulerRunRecord',
    'SchedulerConfig',
    'ClosedLoopScheduler',
    'create_scheduler',
]
