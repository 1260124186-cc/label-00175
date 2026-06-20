import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends

from schemas import (
    SimulationRunRequest,
    SimulationRunResponse,
)
from services import (
    run_simulation,
    get_task_status,
    list_tasks,
)
from auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/simulation", tags=["仿真运行"])


@router.post("/run", response_model=SimulationRunResponse, summary="运行光刻仿真")
async def run_sim(req: SimulationRunRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    config_dict = req.config.model_dump()
    task_id = run_simulation(config_dict, req.pattern_type, req.pattern_params, user_id=user_id)
    return SimulationRunResponse(
        success=True,
        message="仿真任务已提交",
        task_id=task_id,
        status="starting"
    )


@router.get("/tasks", summary="列出所有仿真任务")
async def get_all_tasks(current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    return list_tasks(user_id=user_id)


@router.get("/tasks/{task_id}", summary="查询仿真任务状态")
async def get_task(task_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    return get_task_status(task_id, user_id=user_id)
