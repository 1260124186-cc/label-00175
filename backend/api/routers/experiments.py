import sys
import os
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel

_API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND_ROOT = os.path.dirname(_API_DIR)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from schemas import (
    ExperimentListResponse,
    ExperimentRunListResponse,
    ExperimentRunSummary,
    ExperimentRunDetail,
    ExperimentCompareRequest,
    ExperimentCompareResponse,
    ExperimentCompareRun,
    MetricCompareItem,
    MetricCurveResponse,
    MetricHistoryPoint,
)
from utils.experiment_tracking import (
    list_experiments,
    create_tracker,
    get_run_summary,
    filter_runs,
)
from auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/experiments", tags=["实验追踪"])

DEFAULT_TRACKING_DIR = os.path.join(_BACKEND_ROOT, "mlruns")


def _get_user_tracking_dir(user_id: Optional[str]) -> str:
    if user_id:
        from auth import get_user_dir
        user_dir = get_user_dir(user_id, "mlruns")
        return str(user_dir)
    return DEFAULT_TRACKING_DIR


def _get_tracker(experiment_name: str = "default", user_id: Optional[str] = None):
    tracking_dir = _get_user_tracking_dir(user_id)
    return create_tracker(
        "local",
        experiment_name=experiment_name,
        tracking_dir=tracking_dir,
    )


def _run_to_summary(run) -> ExperimentRunSummary:
    summary = get_run_summary(run)
    metrics_summary = {}
    if run.metrics:
        for name, history in run.metrics.items():
            if history:
                values = [h["value"] for h in history]
                metrics_summary[name] = {
                    "final": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "num_steps": len(values),
                }
    return ExperimentRunSummary(
        run_id=run.run_id,
        experiment_name=run.experiment_name,
        status=run.status,
        start_time=run.start_time,
        end_time=run.end_time,
        duration_seconds=round(run.duration, 2),
        tags=run.tags,
        params=run.params,
        metrics_summary=metrics_summary,
    )


def _run_to_detail(run) -> ExperimentRunDetail:
    summary = _run_to_summary(run)
    metrics = {}
    for name, history in run.metrics.items():
        points = []
        for h in history:
            points.append(MetricHistoryPoint(
                step=h.get("step"),
                value=h["value"],
                timestamp=h.get("timestamp", 0.0),
            ))
        metrics[name] = points
    return ExperimentRunDetail(
        **summary.model_dump(),
        metrics=metrics,
        artifacts=run.artifacts,
    )


@router.get("", response_model=ExperimentListResponse, summary="获取所有实验名称列表")
async def list_all_experiments(current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    tracking_dir = _get_user_tracking_dir(user_id)
    experiments = list_experiments(tracking_dir)
    return ExperimentListResponse(
        count=len(experiments),
        experiments=experiments,
    )


@router.get(
    "/{experiment_name}/runs",
    response_model=ExperimentRunListResponse,
    summary="列出指定实验的所有运行（支持筛选）",
)
async def list_experiment_runs(
    experiment_name: str,
    status: Optional[str] = Query(None, description="按状态筛选: running, completed, failed"),
    tag_key: Optional[str] = Query(None, description="按标签键筛选"),
    tag_value: Optional[str] = Query(None, description="按标签值筛选（需配合 tag_key 使用）"),
    metric_name: Optional[str] = Query(None, description="按指标名称筛选（存在该指标的运行）"),
    metric_min: Optional[float] = Query(None, description="指标最小值过滤（最终值）"),
    metric_max: Optional[float] = Query(None, description="指标最大值过滤（最终值）"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    tracker = _get_tracker(experiment_name, user_id=user_id)
    runs = tracker.list_runs()

    if status:
        runs = [r for r in runs if r.status == status]

    if tag_key:
        runs = [r for r in runs if r.tags.get(tag_key) == tag_value]

    if metric_name:
        filtered = []
        for r in runs:
            history = r.metrics.get(metric_name, [])
            if not history:
                continue
            final_val = history[-1]["value"]
            if metric_min is not None and final_val < metric_min:
                continue
            if metric_max is not None and final_val > metric_max:
                continue
            filtered.append(r)
        runs = filtered

    runs_sorted = sorted(runs, key=lambda r: r.start_time, reverse=True)
    runs_paged = runs_sorted[offset:offset + limit]

    summaries = [_run_to_summary(r) for r in runs_paged]
    return ExperimentRunListResponse(
        count=len(runs),
        runs=summaries,
    )


@router.get(
    "/runs/{run_id}",
    response_model=ExperimentRunDetail,
    summary="获取单个实验运行的详细信息",
)
async def get_run_detail(run_id: str, experiment_name: Optional[str] = None, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    if experiment_name:
        tracker = _get_tracker(experiment_name, user_id=user_id)
        run = tracker.get_run(run_id)
    else:
        run = None
        tracking_dir = _get_user_tracking_dir(user_id)
        for exp_name in list_experiments(tracking_dir):
            tracker = _get_tracker(exp_name, user_id=user_id)
            run = tracker.get_run(run_id)
            if run:
                break

    if run is None:
        raise HTTPException(status_code=404, detail=f"运行不存在: {run_id}")

    return _run_to_detail(run)


@router.post(
    "/compare",
    response_model=ExperimentCompareResponse,
    summary="并排对比多次实验运行的参数与指标",
)
async def compare_runs(request: ExperimentCompareRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    if not request.run_ids:
        raise HTTPException(status_code=400, detail="run_ids 不能为空")

    tracking_dir = _get_user_tracking_dir(user_id)
    all_experiments = list_experiments(tracking_dir)

    runs = []
    experiment_name = None
    for run_id in request.run_ids:
        run = None
        for exp_name in all_experiments:
            tracker = _get_tracker(exp_name, user_id=user_id)
            run = tracker.get_run(run_id)
            if run:
                experiment_name = exp_name
                break
        if run:
            runs.append(run)

    if not runs:
        raise HTTPException(status_code=404, detail="未找到任何运行记录")

    all_metric_names = set()
    all_param_names = set()
    for run in runs:
        all_metric_names.update(run.metrics.keys())
        all_param_names.update(run.params.keys())

    metric_names = sorted(request.metrics or all_metric_names)
    param_names = sorted(request.params or all_param_names)

    compare_runs = []
    for run in runs:
        metrics_data = {}
        for m in metric_names:
            history = run.metrics.get(m, [])
            if history:
                values = [h["value"] for h in history]
                metrics_data[m] = MetricCompareItem(
                    final=values[-1],
                    min=min(values),
                    max=max(values),
                    first=values[0],
                )
            else:
                metrics_data[m] = MetricCompareItem()

        params_data = {}
        for p in param_names:
            if p in run.params:
                params_data[p] = run.params[p]

        compare_runs.append(ExperimentCompareRun(
            run_id=run.run_id,
            status=run.status,
            duration_seconds=round(run.duration, 2),
            tags=run.tags,
            params=params_data,
            metrics=metrics_data,
        ))

    return ExperimentCompareResponse(
        experiment_name=experiment_name,
        compared_run_ids=request.run_ids,
        runs=compare_runs,
        all_metric_names=sorted(all_metric_names),
        all_param_names=sorted(all_param_names),
    )


@router.get(
    "/runs/{run_id}/metrics/{metric_name}",
    response_model=MetricCurveResponse,
    summary="获取指定运行的单个指标历史曲线数据",
)
async def get_metric_curve(
    run_id: str,
    metric_name: str,
    experiment_name: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    if experiment_name:
        tracker = _get_tracker(experiment_name, user_id=user_id)
        run = tracker.get_run(run_id)
    else:
        run = None
        tracking_dir = _get_user_tracking_dir(user_id)
        for exp_name in list_experiments(tracking_dir):
            tracker = _get_tracker(exp_name, user_id=user_id)
            run = tracker.get_run(run_id)
            if run:
                break

    if run is None:
        raise HTTPException(status_code=404, detail=f"运行不存在: {run_id}")

    history = run.metrics.get(metric_name, [])
    if not history:
        raise HTTPException(status_code=404, detail=f"指标不存在: {metric_name}")

    points = [
        MetricHistoryPoint(
            step=h.get("step"),
            value=h["value"],
            timestamp=h.get("timestamp", 0.0),
        )
        for h in history
    ]

    return MetricCurveResponse(
        run_id=run_id,
        metric_name=metric_name,
        points=points,
    )
