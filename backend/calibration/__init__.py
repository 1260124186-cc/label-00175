# -*- coding: utf-8 -*-
"""
Fab 模型标定模块
Fab Model Calibration Module

接收 Fab 提供的 CD-SEM 量测数据（目标 CD、实测 CD、focus/dose 条件），
通过非线性最小二乘或贝叶斯推断反演光刻胶阈值、扩散长度、NA 有效值等模型参数，
输出标定报告与更新后的 default_config.yaml 片段。

子模块：
- schemas: 数据结构定义（量测数据、标定参数、结果等）
- data_loader: CD-SEM 数据加载与预处理（支持 CSV/JSON/YAML）
- forward_model: 简化光刻前向模型（Bossung 曲线 CD 预测）
- inversion: 参数反演引擎（非线性最小二乘 + 贝叶斯 MCMC）
- report_generator: 标定报告生成（文本/Markdown/可视化）
- config_updater: default_config.yaml 片段输出
- pipeline: 端到端标定流水线
"""

from .schemas import (
    CalibrationParameter,
    CalibrationParameterSet,
    CDSEMDataPoint,
    CDSEMDataset,
    InversionMethod,
    CalibrationConfig,
    InversionResult,
    CalibrationReport,
    PatternType,
)
from .data_loader import (
    load_cd_sem_data,
    load_cd_sem_from_csv,
    load_cd_sem_from_json,
    load_cd_sem_from_yaml,
    validate_dataset,
    split_dataset,
)
from .forward_model import (
    LithoForwardModel,
    compute_bossung_cd,
    compute_cd_sensitivity,
    model_prediction,
)
from .inversion import (
    InversionEngine,
    nlls_inversion,
    bayesian_inversion,
    run_inversion,
)
from .report_generator import (
    ReportGenerator,
    generate_calibration_report,
    generate_markdown_report,
    plot_calibration_results,
    plot_bossung_curves,
    plot_parameter_convergence,
    plot_residual_analysis,
)
from .config_updater import (
    ConfigUpdater,
    build_config_snippet,
    update_default_config,
    save_config_snippet,
)
from .pipeline import (
    CalibrationPipeline,
    run_calibration_pipeline,
    calibration_pipeline_from_config,
)

__all__ = [
    'CalibrationParameter',
    'CalibrationParameterSet',
    'CDSEMDataPoint',
    'CDSEMDataset',
    'InversionMethod',
    'CalibrationConfig',
    'InversionResult',
    'CalibrationReport',
    'PatternType',
    'load_cd_sem_data',
    'load_cd_sem_from_csv',
    'load_cd_sem_from_json',
    'load_cd_sem_from_yaml',
    'validate_dataset',
    'split_dataset',
    'LithoForwardModel',
    'compute_bossung_cd',
    'compute_cd_sensitivity',
    'model_prediction',
    'InversionEngine',
    'nlls_inversion',
    'bayesian_inversion',
    'run_inversion',
    'ReportGenerator',
    'generate_calibration_report',
    'generate_markdown_report',
    'plot_calibration_results',
    'plot_bossung_curves',
    'plot_parameter_convergence',
    'plot_residual_analysis',
    'ConfigUpdater',
    'build_config_snippet',
    'update_default_config',
    'save_config_snippet',
    'CalibrationPipeline',
    'run_calibration_pipeline',
    'calibration_pipeline_from_config',
]
