# -*- coding: utf-8 -*-
"""
光刻-刻蚀协同仿真模块 (Lithography-Etch Co-Simulation)

在晶圆成像之后串联简化刻蚀模型，预测最终硅槽深度与 CD，
使优化目标从"光刻胶图形匹配"升级为"刻蚀后形貌匹配"，
支撑 Design-Technology Co-Optimization (DTCO) 研究。

核心模块：
- bias_model: 偏置刻蚀模型，根据局部图形密度/间距/负载效应计算 CD 偏置
- anisotropic_etch: 各向异性刻蚀核卷积模型，模拟刻蚀方向性对形貌的影响
- morphology_predictor: 刻蚀后形貌预测器，预测硅槽深度与最终 CD
- dtco_objective: DTCO 优化目标函数，以刻蚀后形貌匹配为优化目标
- co_simulation_pipeline: 光刻-刻蚀协同仿真管线，串联成像与刻蚀流程
"""

try:
    from holistic.bias_model import (
        BiasModelType,
        BiasModelConfig,
        BiasModelResult,
        EtchBiasModel,
        compute_local_density,
        compute_bias_constant,
        compute_bias_density_dependent,
        compute_bias_pitch_dependent,
        compute_bias_loading_effect,
        apply_bias_to_image,
    )
    from holistic.anisotropic_etch import (
        EtchAnisotropyMode,
        AnisotropicEtchConfig,
        AnisotropicEtchResult,
        AnisotropicEtchModel,
        build_etch_kernel,
        apply_anisotropic_etch,
    )
    from holistic.morphology_predictor import (
        TrenchProfileModel,
        EtchProcessConfig,
        TrenchDepthResult,
        PostEtchCDResult,
        MorphologyPredictionResult,
        MorphologyPredictor,
        predict_trench_depth,
        predict_post_etch_cd,
    )
    from holistic.dtco_objective import (
        DTCOObjectiveMode,
        DTCOObjectiveConfig,
        DTCOObjectiveResult,
        DTCOObjective,
        compute_post_etch_mse_objective,
        compute_post_etch_epe_objective,
        compute_post_etch_cd_objective,
        compute_composite_objective,
    )
    from holistic.co_simulation_pipeline import (
        CoSimPipelineMode,
        CoSimConfig,
        CoSimStepResult,
        CoSimResult,
        LithoEtchCoSimPipeline,
        run_litho_etch_cosim,
        create_dtco_aware_simulate_fn,
    )
except ImportError:
    from .bias_model import (
        BiasModelType,
        BiasModelConfig,
        BiasModelResult,
        EtchBiasModel,
        compute_local_density,
        compute_bias_constant,
        compute_bias_density_dependent,
        compute_bias_pitch_dependent,
        compute_bias_loading_effect,
        apply_bias_to_image,
    )
    from .anisotropic_etch import (
        EtchAnisotropyMode,
        AnisotropicEtchConfig,
        AnisotropicEtchResult,
        AnisotropicEtchModel,
        build_etch_kernel,
        apply_anisotropic_etch,
    )
    from .morphology_predictor import (
        TrenchProfileModel,
        EtchProcessConfig,
        TrenchDepthResult,
        PostEtchCDResult,
        MorphologyPredictionResult,
        MorphologyPredictor,
        predict_trench_depth,
        predict_post_etch_cd,
    )
    from .dtco_objective import (
        DTCOObjectiveMode,
        DTCOObjectiveConfig,
        DTCOObjectiveResult,
        DTCOObjective,
        compute_post_etch_mse_objective,
        compute_post_etch_epe_objective,
        compute_post_etch_cd_objective,
        compute_composite_objective,
    )
    from .co_simulation_pipeline import (
        CoSimPipelineMode,
        CoSimConfig,
        CoSimStepResult,
        CoSimResult,
        LithoEtchCoSimPipeline,
        run_litho_etch_cosim,
        create_dtco_aware_simulate_fn,
    )

__all__ = [
    'BiasModelType',
    'BiasModelConfig',
    'BiasModelResult',
    'EtchBiasModel',
    'compute_local_density',
    'compute_bias_constant',
    'compute_bias_density_dependent',
    'compute_bias_pitch_dependent',
    'compute_bias_loading_effect',
    'apply_bias_to_image',
    'EtchAnisotropyMode',
    'AnisotropicEtchConfig',
    'AnisotropicEtchResult',
    'AnisotropicEtchModel',
    'build_etch_kernel',
    'apply_anisotropic_etch',
    'TrenchProfileModel',
    'EtchProcessConfig',
    'TrenchDepthResult',
    'PostEtchCDResult',
    'MorphologyPredictionResult',
    'MorphologyPredictor',
    'predict_trench_depth',
    'predict_post_etch_cd',
    'DTCOObjectiveMode',
    'DTCOObjectiveConfig',
    'DTCOObjectiveResult',
    'DTCOObjective',
    'compute_post_etch_mse_objective',
    'compute_post_etch_epe_objective',
    'compute_post_etch_cd_objective',
    'compute_composite_objective',
    'CoSimPipelineMode',
    'CoSimConfig',
    'CoSimStepResult',
    'CoSimResult',
    'LithoEtchCoSimPipeline',
    'run_litho_etch_cosim',
    'create_dtco_aware_simulate_fn',
]
