import logging
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from schemas import (
    TaskStatusResponse,
    TaskListResponse,
)
from services import (
    get_task_status,
    get_task_result,
    get_task_download_path,
    list_tasks,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tasks", tags=["任务管理"])


@router.get("", response_model=TaskListResponse, summary="列出所有任务（支持按类型/状态筛选）")
async def list_all_tasks(
    task_type: Optional[str] = Query(None, description="按任务类型筛选: opc, smo, ilt, process_window, batch, simulation"),
    status: Optional[str] = Query(None, description="按状态筛选: pending, running, completed, failed"),
):
    data = list_tasks(task_type=task_type, status=status)
    tasks = [TaskStatusResponse(**t) for t in data.get("tasks", [])]
    return TaskListResponse(count=len(tasks), tasks=tasks)


@router.get("/{task_id}", response_model=TaskStatusResponse, summary="查询单个任务状态")
async def get_single_task(task_id: str):
    data = get_task_status(task_id)
    return TaskStatusResponse(**data)


@router.get("/{task_id}/result", summary="获取任务详细结果（仅当任务已完成时可用）")
async def get_single_task_result(task_id: str):
    return get_task_result(task_id)


@router.get("/{task_id}/download", summary="下载任务结果 JSON 文件")
async def download_task_result(task_id: str):
    file_path = get_task_download_path(task_id)
    return FileResponse(
        path=str(file_path),
        filename=f"task_{task_id}.json",
        media_type="application/json",
    )
