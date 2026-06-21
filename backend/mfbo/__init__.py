# -*- coding: utf-8 -*-
"""
多保真度贝叶斯优化 (MFBO) 模块导出

核心组件：
- 数据结构: FidelityLevel, SearchSpace, MFBOConfig, Observation, MFBOResult
- 代理模型: MultiFidelityGP
- 核函数: AR1Kernel, CoKrigingKernel, LCMKernel
- 获取函数: AcquisitionFunction
- 保真度选择: FidelitySelector
- 主优化器: MultiFidelityBayesianOptimizer
"""

from mfbo.schemas import (
    FidelityLevel,
    AcquisitionFunctionType,
    KernelType,
    FidelitySelectionStrategy,
    FidelityCost,
    SearchSpace,
    MFBOConfig,
    Observation,
    IterationRecord,
    MFBOResult,
)

from mfbo.kernels import (
    AR1Kernel,
    CoKrigingKernel,
    LCMKernel,
    KernelHyperparameters,
    optimize_hyperparameters,
    rbf_kernel,
    matern52_kernel,
    matern32_kernel,
)

from mfbo.mf_gp import (
    MultiFidelityGP,
    PredictionResult,
)

from mfbo.acquisition import (
    AcquisitionFunction,
    AcquisitionConfig,
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
    ei_per_unit_cost,
    multi_fidelity_expected_improvement,
    knowledge_gradient_approx,
)

from mfbo.fidelity_strategy import (
    FidelitySelector,
    FidelityDecision,
)

from mfbo.optimizer import (
    MultiFidelityBayesianOptimizer,
    MultiFidelityEvaluator,
)

__all__ = [
    # Schemas
    'FidelityLevel',
    'AcquisitionFunctionType',
    'KernelType',
    'FidelitySelectionStrategy',
    'FidelityCost',
    'SearchSpace',
    'MFBOConfig',
    'Observation',
    'IterationRecord',
    'MFBOResult',
    # Kernels
    'AR1Kernel',
    'CoKrigingKernel',
    'LCMKernel',
    'KernelHyperparameters',
    'optimize_hyperparameters',
    'rbf_kernel',
    'matern52_kernel',
    'matern32_kernel',
    # MF-GP
    'MultiFidelityGP',
    'PredictionResult',
    # Acquisition
    'AcquisitionFunction',
    'AcquisitionConfig',
    'expected_improvement',
    'upper_confidence_bound',
    'probability_of_improvement',
    'ei_per_unit_cost',
    'multi_fidelity_expected_improvement',
    'knowledge_gradient_approx',
    # Fidelity strategy
    'FidelitySelector',
    'FidelityDecision',
    # Optimizer
    'MultiFidelityBayesianOptimizer',
    'MultiFidelityEvaluator',
]

__version__ = '1.0.0'
__author__ = 'MFBO Module for Lithography Mask Optimization'
