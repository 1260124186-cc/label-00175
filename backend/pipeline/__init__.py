# -*- coding: utf-8 -*-
"""
批处理流水线模块

包含：
1. BatchRunner: 多 cell 并行优化调度
2. Orchestrator: OPC → ILT → SMO → PW 全流程编排
3. Hybrid OPC+ILT: OPC 与 ILT 混合精修模式
"""

try:
    from pipeline.batch_runner import (
        TaskStatus,
        TaskResult,
        BatchConfig,
        ResourceConfig,
        LocalBatchRunner,
        DistributedBatchRunner,
        BatchSummary,
        run_batch_optimization,
        save_batch_summary,
    )

    from pipeline.orchestrator import (
        PipelineStage,
        PWVerifyConfig,
        SurrogateIntegrationConfig,
        PipelineConfig,
        StageMetrics,
        PipelineResult,
        PipelineOrchestrator,
        run_pipeline,
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
except ImportError:
    from .batch_runner import (
        TaskStatus,
        TaskResult,
        BatchConfig,
        ResourceConfig,
        LocalBatchRunner,
        DistributedBatchRunner,
        BatchSummary,
        run_batch_optimization,
        save_batch_summary,
    )

    from .orchestrator import (
        PipelineStage,
        PWVerifyConfig,
        SurrogateIntegrationConfig,
        PipelineConfig,
        StageMetrics,
        PipelineResult,
        PipelineOrchestrator,
        run_pipeline,
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

__all__ = [
    'TaskStatus',
    'TaskResult',
    'BatchConfig',
    'ResourceConfig',
    'LocalBatchRunner',
    'DistributedBatchRunner',
    'BatchSummary',
    'run_batch_optimization',
    'save_batch_summary',
    'PipelineStage',
    'PWVerifyConfig',
    'SurrogateIntegrationConfig',
    'PipelineConfig',
    'StageMetrics',
    'PipelineResult',
    'PipelineOrchestrator',
    'run_pipeline',
    'HybridOPCILTConfig',
    'LocalILTResult',
    'HybridOPCILTWorkflowResult',
    'HotspotBBoxManager',
    'LocalILTOptimizer',
    'HybridOPCILTWorkflow',
    'run_hybrid_opc_ilt_workflow',
]
