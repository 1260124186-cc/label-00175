# -*- coding: utf-8 -*-
"""
工作流模块：封装高级光刻优化流程

本模块提供完整的光刻优化工作流，包括：
1. OPC (Optical Proximity Correction) - 光学邻近校正工作流
2. SMO (Source-Mask Optimization) - 光源掩模协同优化工作流
3. ILT (Inverse Lithography Technology) - 反演光刻工作流
4. Hybrid OPC+ILT - OPC 与 ILT 混合精修工作流
"""

try:
    from workflows.opc import (
        OPCConfig,
        HotspotDetector,
        SRAFRuleEngine,
        OPCTransformType,
        OPCTransform,
        OPCIterationController,
        OPCWorkflow,
        HotspotRegion,
        SRAFFeature,
        OPCIterationResult,
        OPCWorkflowResult,
        run_opc_workflow,
    )

    from workflows.smo import (
        SMOptimizationStrategy,
        SourceInitializationType,
        SourceConstraintsConfig,
        SMOConfig,
        SMOIterationResult,
        SMOWorkflowResult,
        PixelatedSource,
        SMOImagingModel,
        SourceOptimizer,
        MaskOptimizerForSMO,
        JointGradientOptimizer,
        SMOWorkflow,
        run_smo_workflow,
    )

    from workflows.ilt import (
        TransmissionLevel,
        ILTOptimizerType,
        ILTLossComponent,
        ILTComplexityConfig,
        ILTConfig,
        ILTIterationResult,
        ILTWorkflowResult,
        DifferentiableImagingChain,
        GradientProjector,
        MaskComplexityPenalty,
        MultiObjectiveILT,
        ILTWorkflow,
        run_ilt_workflow,
    )

    from workflows.hybrid_opc_ilt import (
        HybridOPCILTConfig,
        LocalILTResult,
        HybridOPCILTWorkflowResult,
        HotspotBBoxManager,
        LocalILTOptimizer,
        HybridOPCILTWorkflow,
        run_hybrid_opc_ilt_workflow,
    )

    from workflows.mpc_opc import (
        MPCConfig,
        PredictionResult,
        MPCOptimizationResult,
        MPCOPCIterationResult,
        MPCOPCWorkflowResult,
        EPEPredictor,
        ProcessDriftEstimator,
        MPCOptimizer,
        MPCOPCWorkflow,
        run_mpc_opc_workflow,
    )
except ImportError:
    from .opc import (
        OPCConfig,
        HotspotDetector,
        SRAFRuleEngine,
        OPCTransformType,
        OPCTransform,
        OPCIterationController,
        OPCWorkflow,
        HotspotRegion,
        SRAFFeature,
        OPCIterationResult,
        OPCWorkflowResult,
        run_opc_workflow,
    )

    from .smo import (
        SMOptimizationStrategy,
        SourceInitializationType,
        SourceConstraintsConfig,
        SMOConfig,
        SMOIterationResult,
        SMOWorkflowResult,
        PixelatedSource,
        SMOImagingModel,
        SourceOptimizer,
        MaskOptimizerForSMO,
        JointGradientOptimizer,
        SMOWorkflow,
        run_smo_workflow,
    )

    from .ilt import (
        TransmissionLevel,
        ILTOptimizerType,
        ILTLossComponent,
        ILTComplexityConfig,
        ILTConfig,
        ILTIterationResult,
        ILTWorkflowResult,
        DifferentiableImagingChain,
        GradientProjector,
        MaskComplexityPenalty,
        MultiObjectiveILT,
        ILTWorkflow,
        run_ilt_workflow,
    )

    from .hybrid_opc_ilt import (
        HybridOPCILTConfig,
        LocalILTResult,
        HybridOPCILTWorkflowResult,
        HotspotBBoxManager,
        LocalILTOptimizer,
        HybridOPCILTWorkflow,
        run_hybrid_opc_ilt_workflow,
    )

    from .mpc_opc import (
        MPCConfig,
        PredictionResult,
        MPCOptimizationResult,
        MPCOPCIterationResult,
        MPCOPCWorkflowResult,
        EPEPredictor,
        ProcessDriftEstimator,
        MPCOptimizer,
        MPCOPCWorkflow,
        run_mpc_opc_workflow,
    )

__all__ = [
    'OPCConfig',
    'HotspotDetector',
    'SRAFRuleEngine',
    'OPCTransformType',
    'OPCTransform',
    'OPCIterationController',
    'OPCWorkflow',
    'HotspotRegion',
    'SRAFFeature',
    'OPCIterationResult',
    'OPCWorkflowResult',
    'run_opc_workflow',
    'SMOptimizationStrategy',
    'SourceInitializationType',
    'SourceConstraintsConfig',
    'SMOConfig',
    'SMOIterationResult',
    'SMOWorkflowResult',
    'PixelatedSource',
    'SMOImagingModel',
    'SourceOptimizer',
    'MaskOptimizerForSMO',
    'JointGradientOptimizer',
    'SMOWorkflow',
    'run_smo_workflow',
    'TransmissionLevel',
    'ILTOptimizerType',
    'ILTLossComponent',
    'ILTComplexityConfig',
    'ILTConfig',
    'ILTIterationResult',
    'ILTWorkflowResult',
    'DifferentiableImagingChain',
    'GradientProjector',
    'MaskComplexityPenalty',
    'MultiObjectiveILT',
    'ILTWorkflow',
    'run_ilt_workflow',
    'HybridOPCILTConfig',
    'LocalILTResult',
    'HybridOPCILTWorkflowResult',
    'HotspotBBoxManager',
    'LocalILTOptimizer',
    'HybridOPCILTWorkflow',
    'run_hybrid_opc_ilt_workflow',
    'MPCConfig',
    'PredictionResult',
    'MPCOptimizationResult',
    'MPCOPCIterationResult',
    'MPCOPCWorkflowResult',
    'EPEPredictor',
    'ProcessDriftEstimator',
    'MPCOptimizer',
    'MPCOPCWorkflow',
    'run_mpc_opc_workflow',
]
