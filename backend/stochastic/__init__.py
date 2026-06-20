# -*- coding: utf-8 -*-
"""
随机光刻仿真模块

引入光子散粒噪声、光酸扩散随机性、显影阈值波动等随机过程，
对同一掩模多次蒙特卡洛显影得到 CD 分布，
评估线边缘粗糙度（LER/LWR）与失效概率，
为随机效应敏感结构提供风险评估。

核心模块：
- noise_models: 随机过程噪声模型
- monte_carlo: 蒙特卡洛仿真框架
- cd_distribution: CD 分布统计分析
- edge_roughness: LER/LWR 线边缘粗糙度评估
- risk_assessment: 失效概率与风险评估
"""

try:
    from stochastic.noise_models import (
        NoiseType,
        NoiseConfig,
        NoiseRealization,
        NoiseGenerator,
        apply_stochastic_diffusion,
        apply_stochastic_lithography,
        create_euv_noise_config,
        create_duv_noise_config,
    )
    from stochastic.monte_carlo import (
        MonteCarloStochasticConfig,
        SingleRealizationResult,
        MonteCarloStochasticResult,
        StochasticMonteCarloSimulator,
        run_stochastic_monte_carlo,
    )
    from stochastic.cd_distribution import (
        DistributionType,
        CDBasicStats,
        DistributionFitResult,
        ProcessCapability,
        CDDistributionAnalysis,
        CDDistributionAnalyzer,
        analyze_cd_distribution,
    )
    from stochastic.edge_roughness import (
        EdgeDirection,
        RoughnessMetric,
        EdgeProfile,
        SingleEdgeRoughnessResult,
        LWRResult,
        SingleRealizationRoughnessResult,
        MonteCarloRoughnessResult,
        EdgeRoughnessAnalyzer,
        compute_ler_from_edges,
        compute_lwr_from_edges,
    )
    from stochastic.risk_assessment import (
        FailureMode,
        RiskLevel,
        SensitivityMetric,
        FailureCriteria,
        FailureCount,
        FailureProbabilityResult,
        SensitivityAnalysisResult,
        RiskAssessmentResult,
        FailureProbabilityEstimator,
        FailureDetector,
        SensitivityAnalyzer,
        RiskAssessor,
        assess_stochastic_risk,
    )
except ImportError:
    from .noise_models import (
        NoiseType,
        NoiseConfig,
        NoiseRealization,
        NoiseGenerator,
        apply_stochastic_diffusion,
        apply_stochastic_lithography,
        create_euv_noise_config,
        create_duv_noise_config,
    )
    from .monte_carlo import (
        MonteCarloStochasticConfig,
        SingleRealizationResult,
        MonteCarloStochasticResult,
        StochasticMonteCarloSimulator,
        run_stochastic_monte_carlo,
    )
    from .cd_distribution import (
        DistributionType,
        CDBasicStats,
        DistributionFitResult,
        ProcessCapability,
        CDDistributionAnalysis,
        CDDistributionAnalyzer,
        analyze_cd_distribution,
    )
    from .edge_roughness import (
        EdgeDirection,
        RoughnessMetric,
        EdgeProfile,
        SingleEdgeRoughnessResult,
        LWRResult,
        SingleRealizationRoughnessResult,
        MonteCarloRoughnessResult,
        EdgeRoughnessAnalyzer,
        compute_ler_from_edges,
        compute_lwr_from_edges,
    )
    from .risk_assessment import (
        FailureMode,
        RiskLevel,
        SensitivityMetric,
        FailureCriteria,
        FailureCount,
        FailureProbabilityResult,
        SensitivityAnalysisResult,
        RiskAssessmentResult,
        FailureProbabilityEstimator,
        FailureDetector,
        SensitivityAnalyzer,
        RiskAssessor,
        assess_stochastic_risk,
    )

__all__ = [
    'NoiseType',
    'NoiseConfig',
    'NoiseRealization',
    'NoiseGenerator',
    'apply_stochastic_diffusion',
    'apply_stochastic_lithography',
    'create_euv_noise_config',
    'create_duv_noise_config',
    'MonteCarloStochasticConfig',
    'SingleRealizationResult',
    'MonteCarloStochasticResult',
    'StochasticMonteCarloSimulator',
    'run_stochastic_monte_carlo',
    'DistributionType',
    'CDBasicStats',
    'DistributionFitResult',
    'ProcessCapability',
    'CDDistributionAnalysis',
    'CDDistributionAnalyzer',
    'analyze_cd_distribution',
    'EdgeDirection',
    'RoughnessMetric',
    'EdgeProfile',
    'SingleEdgeRoughnessResult',
    'LWRResult',
    'SingleRealizationRoughnessResult',
    'MonteCarloRoughnessResult',
    'EdgeRoughnessAnalyzer',
    'compute_ler_from_edges',
    'compute_lwr_from_edges',
    'FailureMode',
    'RiskLevel',
    'SensitivityMetric',
    'FailureCriteria',
    'FailureCount',
    'FailureProbabilityResult',
    'SensitivityAnalysisResult',
    'RiskAssessmentResult',
    'FailureProbabilityEstimator',
    'FailureDetector',
    'SensitivityAnalyzer',
    'RiskAssessor',
    'assess_stochastic_risk',
]
