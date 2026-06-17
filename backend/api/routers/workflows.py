import logging

from fastapi import APIRouter

from schemas import (
    TaskSubmitResponse,
    OPCRunRequest,
    SMORunRequest,
    ILTRunRequest,
    ProcessWindowRunRequest,
    BatchOptimizationRequest,
)
from services import (
    run_opc,
    run_smo,
    run_ilt,
    run_process_window,
    run_batch,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflows", tags=["RET工作流"])


@router.post("/opc", response_model=TaskSubmitResponse, summary="运行OPC工作流（光学邻近校正）")
async def submit_opc(req: OPCRunRequest):
    payload = {
        "optical_system": req.optical_system.model_dump(),
        "opc_config": req.opc_config.model_dump(),
        "pattern_type": req.pattern_type,
        "pattern_params": req.pattern_params,
        "gds_file_id": req.gds_file_id,
        "gds_layer": req.gds_layer,
        "gds_datatype": req.gds_datatype,
        "gds_pixel_size": req.gds_pixel_size,
        "gds_target_size": req.gds_target_size,
    }
    task_id = run_opc(payload)
    return TaskSubmitResponse(
        success=True,
        message="OPC工作流任务已提交",
        task_id=task_id,
        task_type="opc",
        status="pending",
    )


@router.post("/smo", response_model=TaskSubmitResponse, summary="运行SMO工作流（光源掩模协同优化）")
async def submit_smo(req: SMORunRequest):
    payload = {
        "optical_system": req.optical_system.model_dump(),
        "smo_config": req.smo_config.model_dump(),
        "pattern_type": req.pattern_type,
        "pattern_params": req.pattern_params,
        "gds_file_id": req.gds_file_id,
        "gds_layer": req.gds_layer,
        "gds_datatype": req.gds_datatype,
        "gds_pixel_size": req.gds_pixel_size,
        "gds_target_size": req.gds_target_size,
    }
    task_id = run_smo(payload)
    return TaskSubmitResponse(
        success=True,
        message="SMO工作流任务已提交",
        task_id=task_id,
        task_type="smo",
        status="pending",
    )


@router.post("/ilt", response_model=TaskSubmitResponse, summary="运行ILT工作流（反演光刻技术）")
async def submit_ilt(req: ILTRunRequest):
    payload = {
        "optical_system": req.optical_system.model_dump(),
        "ilt_config": req.ilt_config.model_dump(),
        "pattern_type": req.pattern_type,
        "pattern_params": req.pattern_params,
        "gds_file_id": req.gds_file_id,
        "gds_layer": req.gds_layer,
        "gds_datatype": req.gds_datatype,
        "gds_pixel_size": req.gds_pixel_size,
        "gds_target_size": req.gds_target_size,
    }
    task_id = run_ilt(payload)
    return TaskSubmitResponse(
        success=True,
        message="ILT工作流任务已提交",
        task_id=task_id,
        task_type="ilt",
        status="pending",
    )


@router.post("/process-window", response_model=TaskSubmitResponse, summary="运行工艺窗口分析")
async def submit_process_window(req: ProcessWindowRunRequest):
    payload = {
        "optical_system": req.optical_system.model_dump(),
        "pattern_type": req.pattern_type,
        "pattern_params": req.pattern_params,
        "gds_file_id": req.gds_file_id,
        "gds_layer": req.gds_layer,
        "gds_datatype": req.gds_datatype,
        "gds_pixel_size": req.gds_pixel_size,
        "gds_target_size": req.gds_target_size,
        "focus_range": req.focus_range,
        "dose_range": req.dose_range,
        "cd_tolerance": req.cd_tolerance,
        "epe_tolerance": req.epe_tolerance,
        "threshold": req.threshold,
        "save_visualizations": req.save_visualizations,
    }
    task_id = run_process_window(payload)
    return TaskSubmitResponse(
        success=True,
        message="工艺窗口分析任务已提交",
        task_id=task_id,
        task_type="process_window",
        status="pending",
    )


@router.post("/batch", response_model=TaskSubmitResponse, summary="运行批处理优化")
async def submit_batch(req: BatchOptimizationRequest):
    payload = {
        "source": req.source,
        "layer": req.layer,
        "optical_system": req.optical_system.model_dump(),
        "optimization": req.optimization.model_dump(),
        "max_workers": req.max_workers,
        "max_retries": req.max_retries,
        "save_optimized_masks": req.save_optimized_masks,
        "output_dir": req.output_dir,
        "stop_on_first_failure": req.stop_on_first_failure,
    }
    task_id = run_batch(payload)
    return TaskSubmitResponse(
        success=True,
        message="批处理优化任务已提交",
        task_id=task_id,
        task_type="batch",
        status="pending",
    )
