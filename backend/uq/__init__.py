# -*- coding: utf-8 -*-
"""
不确定性量化 (Uncertainty Quantification, UQ) 模块

对优化得到的掩模在工艺参数扰动、模型参数不确定性下的成像指标
做贝叶斯推断或 Bootstrap 分析，输出置信区间与失效概率，
回答"优化结果在真实 fab 中有多可靠"的研究问题。

核心模块：
- schemas: 数据结构定义（参数不确定性配置、置信区间、UQ结果等）
- parameter_uncertainty: 工艺参数和模型参数的不确定性建模与采样
- bootstrap: Bootstrap 重采样分析（非参数 Bootstrap、残差 Bootstrap）
- bayesian_inference: 贝叶斯推断（MCMC、近似贝叶斯计算 ABC）
- reliability: 可靠性分析：失效概率、可靠性指标
- uq_analyzer: UQ 分析器，集成工艺扰动仿真和统计推断
- visualization: UQ 结果可视化
"""

from __future__ import annotations

try:
    from uq.schemas import (
        UncertaintyType,
        ParameterDistribution,
        ProcessUncertaintyConfig,
        ModelUncertaintyConfig,
        UQConfig,
        ConfidenceInterval,
        MetricUncertainty,
        FailureProbabilityResult,
        ReliabilityResult,
        UQResult,
        UQMethod,
    )
    from uq.parameter_uncertainty import (
        ParameterSampler,
        ProcessPerturbationSampler,
        ModelUncertaintySampler,
        sample_process_uncertainties,
        sample_model_uncertainties,
    )
    from uq.bootstrap import (
        BootstrapMethod,
        BootstrapConfig,
        BootstrapResult,
        BootstrapAnalyzer,
        bootstrap_ci,
    )
    from uq.bayesian_inference import (
        BayesianInferenceConfig,
        MCMCConfig,
        ABCConfig,
        PosteriorSample,
        BayesianResult,
        MCMCSampler,
        ABCSampler,
        bayesian_inference,
    )
    from uq.reliability import (
        FailureCriterion,
        ReliabilityAnalyzer,
        compute_failure_probability,
        compute_reliability_index,
        first_order_reliability,
    )
    from uq.uq_analyzer import (
        UQAnalyzer,
        run_uq_analysis,
    )
    from uq.visualization import (
        PlotConfig,
        plot_metric_distribution,
        plot_confidence_intervals,
        plot_sensitivity_indices,
        plot_parameter_contributions,
        plot_failure_probability,
        plot_posterior_distribution,
        plot_process_scatter,
        plot_summary_dashboard,
    )
except ImportError:
    from .schemas import (
        UncertaintyType,
        ParameterDistribution,
        ProcessUncertaintyConfig,
        ModelUncertaintyConfig,
        UQConfig,
        ConfidenceInterval,
        MetricUncertainty,
        FailureProbabilityResult,
        ReliabilityResult,
        UQResult,
        UQMethod,
    )
    from .parameter_uncertainty import (
        ParameterSampler,
        ProcessPerturbationSampler,
        ModelUncertaintySampler,
        sample_process_uncertainties,
        sample_model_uncertainties,
    )
    from .bootstrap import (
        BootstrapMethod,
        BootstrapConfig,
        BootstrapResult,
        BootstrapAnalyzer,
        bootstrap_ci,
    )
    from .bayesian_inference import (
        BayesianInferenceConfig,
        MCMCConfig,
        ABCConfig,
        PosteriorSample,
        BayesianResult,
        MCMCSampler,
        ABCSampler,
        bayesian_inference,
    )
    from .reliability import (
        FailureCriterion,
        ReliabilityAnalyzer,
        compute_failure_probability,
        compute_reliability_index,
        first_order_reliability,
    )
    from .uq_analyzer import (
        UQAnalyzer,
        run_uq_analysis,
    )
    from .visualization import (
        PlotConfig,
        plot_metric_distribution,
        plot_confidence_intervals,
        plot_sensitivity_indices,
        plot_parameter_contributions,
        plot_failure_probability,
        plot_posterior_distribution,
        plot_process_scatter,
        plot_summary_dashboard,
    )

__all__ = [
    'UncertaintyType',
    'ParameterDistribution',
    'ProcessUncertaintyConfig',
    'ModelUncertaintyConfig',
    'UQConfig',
    'ConfidenceInterval',
    'MetricUncertainty',
    'FailureProbabilityResult',
    'ReliabilityResult',
    'UQResult',
    'UQMethod',
    'ParameterSampler',
    'ProcessPerturbationSampler',
    'ModelUncertaintySampler',
    'sample_process_uncertainties',
    'sample_model_uncertainties',
    'BootstrapMethod',
    'BootstrapConfig',
    'BootstrapResult',
    'BootstrapAnalyzer',
    'bootstrap_ci',
    'BayesianInferenceConfig',
    'MCMCConfig',
    'ABCConfig',
    'PosteriorSample',
    'BayesianResult',
    'MCMCSampler',
    'ABCSampler',
    'bayesian_inference',
    'FailureCriterion',
    'ReliabilityAnalyzer',
    'compute_failure_probability',
    'compute_reliability_index',
    'first_order_reliability',
    'UQAnalyzer',
    'run_uq_analysis',
    'PlotConfig',
    'plot_metric_distribution',
    'plot_confidence_intervals',
    'plot_sensitivity_indices',
    'plot_parameter_contributions',
    'plot_failure_probability',
    'plot_posterior_distribution',
    'plot_process_scatter',
    'plot_summary_dashboard',
]
