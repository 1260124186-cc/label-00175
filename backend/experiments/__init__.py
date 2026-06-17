# -*- coding: utf-8 -*-
"""
配置驱动的实验编排与回归测试模块

本模块将配置 + 脚本 + 断言标准化，支撑研究中的可复现实验与 CI 回归。

主要组件:
    1. ExperimentSchema: YAML 实验定义格式，描述输入图案、光学参数、优化器、期望指标范围
    2. ExperimentExecutor: 读取实验定义，调用 MaskOptimizer / OPC / SMO 流程，写结果到结构化目录
    3. RegressionAssertions: 回归断言引擎，如 MSE < 1e-3、100步内收敛、与golden偏差 < 5%
    4. CLI 入口: run_experiments.py，支持命令行批量执行与 pytest 集成
"""

try:
    from experiments.schema import (
        ExperimentSchema,
        PatternConfig,
        OpticalConfig,
        OptimizerConfig,
        AssertionConfig,
        GoldenReference,
        load_experiment,
        validate_experiment,
    )
    from experiments.executor import ExperimentExecutor, ExperimentResult
    from experiments.assertions import (
        RegressionAssertions,
        AssertionResult,
        AssertionReport,
    )
    from experiments.hyperparam_search import (
        ParamType,
        SearchParam,
        SearchSpace,
        SamplerType,
        PrunerType,
        ObjectiveConfig,
        HyperparamSearchConfig,
        TrialResult,
        HyperparamSearcher,
        get_default_search_space,
        get_default_objectives,
    )
except ImportError:
    from .schema import (
        ExperimentSchema,
        PatternConfig,
        OpticalConfig,
        OptimizerConfig,
        AssertionConfig,
        GoldenReference,
        load_experiment,
        validate_experiment,
    )
    from .executor import ExperimentExecutor, ExperimentResult
    from .assertions import (
        RegressionAssertions,
        AssertionResult,
        AssertionReport,
    )
    from .hyperparam_search import (
        ParamType,
        SearchParam,
        SearchSpace,
        SamplerType,
        PrunerType,
        ObjectiveConfig,
        HyperparamSearchConfig,
        TrialResult,
        HyperparamSearcher,
        get_default_search_space,
        get_default_objectives,
    )

__all__ = [
    'ExperimentSchema',
    'PatternConfig',
    'OpticalConfig',
    'OptimizerConfig',
    'AssertionConfig',
    'GoldenReference',
    'load_experiment',
    'validate_experiment',
    'ExperimentExecutor',
    'ExperimentResult',
    'RegressionAssertions',
    'AssertionResult',
    'AssertionReport',
    'ParamType',
    'SearchParam',
    'SearchSpace',
    'SamplerType',
    'PrunerType',
    'ObjectiveConfig',
    'HyperparamSearchConfig',
    'TrialResult',
    'HyperparamSearcher',
    'get_default_search_space',
    'get_default_objectives',
]
