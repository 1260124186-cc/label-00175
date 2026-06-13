# -*- coding: utf-8 -*-
"""
实验追踪模块

提供统一的实验追踪接口，支持 MLflow、Weights & Biases (WandB) 和本地文件后端。
记录每次运行的配置、指标、耗时，支持按实验 ID 回溯与对比。
"""

import os
import json
import time
import uuid
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """实验运行记录"""
    run_id: str
    experiment_name: str
    start_time: float
    end_time: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    status: str = "running"

    @property
    def duration(self) -> float:
        """运行时长（秒）"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class BaseExperimentTracker(ABC):
    """实验追踪器基类"""

    def __init__(self, experiment_name: str = "default", **kwargs):
        self.experiment_name = experiment_name
        self.active_run: Optional[ExperimentRun] = None
        self._run_start_time: float = 0.0

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        开始一次实验运行

        Args:
            run_name: 运行名称
            tags: 标签

        Returns:
            run_id
        """
        pass

    @abstractmethod
    def end_run(self, status: str = "completed"):
        """结束当前运行"""
        pass

    @abstractmethod
    def log_param(self, key: str, value: Any):
        """记录单个参数"""
        pass

    def log_params(self, params: Dict[str, Any]):
        """批量记录参数"""
        for key, value in params.items():
            self.log_param(key, value)

    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """记录单个指标"""
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """批量记录指标"""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    @abstractmethod
    def set_tag(self, key: str, value: str):
        """设置标签"""
        pass

    def set_tags(self, tags: Dict[str, str]):
        """批量设置标签"""
        for key, value in tags.items():
            self.set_tag(key, value)

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """记录工件（文件）"""
        pass

    def log_config(self, config: Dict[str, Any]):
        """
        记录配置（作为 params 和 artifact）

        Args:
            config: 配置字典
        """
        flat_config = _flatten_dict(config)
        self.log_params(flat_config)

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """根据 run_id 获取运行记录"""
        pass

    @abstractmethod
    def list_runs(self, experiment_name: Optional[str] = None) -> List[ExperimentRun]:
        """列出指定实验的所有运行"""
        pass

    @abstractmethod
    def compare_runs(self, run_ids: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """对比多次运行的指标"""
        pass


class LocalFileTracker(BaseExperimentTracker):
    """本地文件实验追踪器（无外部依赖）"""

    def __init__(self, experiment_name: str = "default",
                 tracking_dir: str = "./mlruns", **kwargs):
        super().__init__(experiment_name)
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_dir = self.tracking_dir / experiment_name
        self._experiment_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        return self._experiment_dir / run_id

    def _save_run_info(self, run: ExperimentRun):
        """保存运行信息到 JSON"""
        run_dir = self._run_dir(run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        info = {
            "run_id": run.run_id,
            "experiment_name": run.experiment_name,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "params": run.params,
            "metrics": run.metrics,
            "tags": run.tags,
            "artifacts": run.artifacts,
            "status": run.status,
            "duration": run.duration,
        }

        info_path = run_dir / "info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(_convert_numpy_types(info), f, indent=2, ensure_ascii=False)

    def _load_run_info(self, run_id: str) -> Optional[ExperimentRun]:
        """从 JSON 加载运行信息"""
        info_path = self._run_dir(run_id) / "info.json"
        if not info_path.exists():
            return None

        with open(info_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return ExperimentRun(
            run_id=data["run_id"],
            experiment_name=data["experiment_name"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            params=data.get("params", {}),
            metrics=data.get("metrics", {}),
            tags=data.get("tags", {}),
            artifacts=data.get("artifacts", []),
            status=data.get("status", "unknown"),
        )

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        run_id = str(uuid.uuid4())[:8]
        if run_name:
            run_id = f"{run_name}_{run_id}"

        self.active_run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            start_time=time.time(),
        )

        if tags:
            self.active_run.tags.update(tags)
        if run_name:
            self.active_run.tags["run_name"] = run_name

        self._run_start_time = time.time()
        self._save_run_info(self.active_run)

        logger.info(f"实验运行开始: {run_id} (实验: {self.experiment_name})")
        return run_id

    def end_run(self, status: str = "completed"):
        if self.active_run is None:
            return

        self.active_run.end_time = time.time()
        self.active_run.status = status
        self._save_run_info(self.active_run)

        duration = self.active_run.duration
        logger.info(f"实验运行结束: {self.active_run.run_id}, "
                    f"状态: {status}, 耗时: {duration:.2f}s")

        self.active_run = None

    def log_param(self, key: str, value: Any):
        if self.active_run is None:
            return
        self.active_run.params[key] = _convert_numpy_types(value)
        self._save_run_info(self.active_run)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if self.active_run is None:
            return

        if key not in self.active_run.metrics:
            self.active_run.metrics[key] = []

        metric_entry = {
            "value": float(value),
            "timestamp": time.time(),
        }
        if step is not None:
            metric_entry["step"] = step

        self.active_run.metrics[key].append(metric_entry)
        self._save_run_info(self.active_run)

    def set_tag(self, key: str, value: str):
        if self.active_run is None:
            return
        self.active_run.tags[key] = str(value)
        self._save_run_info(self.active_run)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if self.active_run is None:
            return

        run_dir = self._run_dir(self.active_run.run_id)
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        src = Path(local_path)
        if artifact_path:
            dst = artifacts_dir / artifact_path
            dst.parent.mkdir(parents=True, exist_ok=True)
        else:
            dst = artifacts_dir / src.name

        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

        self.active_run.artifacts.append(str(dst.relative_to(run_dir)))
        self._save_run_info(self.active_run)

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        return self._load_run_info(run_id)

    def list_runs(self, experiment_name: Optional[str] = None) -> List[ExperimentRun]:
        exp_dir = self.tracking_dir / (experiment_name or self.experiment_name)
        if not exp_dir.exists():
            return []

        runs = []
        for run_dir in exp_dir.iterdir():
            if run_dir.is_dir():
                run = self._load_run_info(run_dir.name)
                if run:
                    runs.append(run)

        runs.sort(key=lambda r: r.start_time, reverse=True)
        return runs

    def compare_runs(self, run_ids: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        runs_data = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is None:
                continue

            run_data = {
                "run_id": run.run_id,
                "duration": run.duration,
                "status": run.status,
                "tags": run.tags,
            }

            metric_values = {}
            for metric_name, metric_history in run.metrics.items():
                if metrics and metric_name not in metrics:
                    continue
                if metric_history:
                    metric_values[metric_name] = {
                        "final": metric_history[-1]["value"],
                        "min": min(m["value"] for m in metric_history),
                        "max": max(m["value"] for m in metric_history),
                        "first": metric_history[0]["value"],
                    }
            run_data["metrics"] = metric_values
            runs_data.append(run_data)

        return {
            "experiment_name": self.experiment_name,
            "runs": runs_data,
            "compared_run_ids": run_ids,
        }


class MLflowTracker(BaseExperimentTracker):
    """MLflow 实验追踪器"""

    def __init__(self, experiment_name: str = "default",
                 tracking_uri: Optional[str] = None, **kwargs):
        super().__init__(experiment_name)
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            raise ImportError("MLflow 未安装，请运行: pip install mlflow")

        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)

        self._mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        run = self._mlflow.start_run(run_name=run_name, tags=tags)
        self.active_run = ExperimentRun(
            run_id=run.info.run_id,
            experiment_name=self.experiment_name,
            start_time=time.time(),
        )
        if tags:
            self.active_run.tags.update(tags)
        self._run_start_time = time.time()
        logger.info(f"MLflow 实验运行开始: {run.info.run_id}")
        return run.info.run_id

    def end_run(self, status: str = "completed"):
        if self.active_run:
            self.active_run.end_time = time.time()
            self.active_run.status = status
            logger.info(f"MLflow 实验运行结束: {self.active_run.run_id}, 状态: {status}")
        self._mlflow.end_run(status)
        self.active_run = None

    def log_param(self, key: str, value: Any):
        self._mlflow.log_param(key, value)
        if self.active_run:
            self.active_run.params[key] = value

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        self._mlflow.log_metric(key, value, step=step)
        if self.active_run:
            if key not in self.active_run.metrics:
                self.active_run.metrics[key] = []
            entry = {"value": float(value), "timestamp": time.time()}
            if step is not None:
                entry["step"] = step
            self.active_run.metrics[key].append(entry)

    def set_tag(self, key: str, value: str):
        self._mlflow.set_tag(key, value)
        if self.active_run:
            self.active_run.tags[key] = str(value)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        self._mlflow.log_artifact(local_path, artifact_path)
        if self.active_run:
            self.active_run.artifacts.append(local_path)

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        try:
            run = self._mlflow.get_run(run_id)
            return ExperimentRun(
                run_id=run.info.run_id,
                experiment_name=run.info.experiment_id,
                start_time=run.info.start_time / 1000.0,
                end_time=run.info.end_time / 1000.0 if run.info.end_time else None,
                params=dict(run.data.params),
                tags=dict(run.data.tags),
                status=run.info.status,
            )
        except Exception:
            return None

    def list_runs(self, experiment_name: Optional[str] = None) -> List[ExperimentRun]:
        exp = self._mlflow.get_experiment_by_name(
            experiment_name or self.experiment_name
        )
        if exp is None:
            return []

        runs = self._mlflow.search_runs(experiment_ids=[exp.experiment_id])
        result = []
        for _, row in runs.iterrows():
            result.append(ExperimentRun(
                run_id=row["run_id"],
                experiment_name=experiment_name or self.experiment_name,
                start_time=row["start_time"].timestamp() if hasattr(row["start_time"], 'timestamp') else time.time(),
                status=row["status"],
                params={k: v for k, v in row.items() if k.startswith("params.")},
                tags={k: v for k, v in row.items() if k.startswith("tags.")},
            ))
        return result

    def compare_runs(self, run_ids: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        runs_data = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is None:
                continue
            runs_data.append({
                "run_id": run.run_id,
                "duration": run.duration,
                "status": run.status,
                "params": run.params,
                "tags": run.tags,
            })

        return {
            "experiment_name": self.experiment_name,
            "runs": runs_data,
            "compared_run_ids": run_ids,
        }


class WandBTracker(BaseExperimentTracker):
    """Weights & Biases 实验追踪器"""

    def __init__(self, experiment_name: str = "default",
                 project: Optional[str] = None,
                 entity: Optional[str] = None, **kwargs):
        super().__init__(experiment_name)
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError("WandB 未安装，请运行: pip install wandb")

        self._project = project or experiment_name
        self._entity = entity
        self._run = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        config = tags or {}
        self._run = self._wandb.init(
            project=self._project,
            entity=self._entity,
            name=run_name,
            tags=list(tags.keys()) if tags else None,
            config=config,
            reinit=True,
        )

        run_id = self._run.id
        self.active_run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            start_time=time.time(),
        )
        if tags:
            self.active_run.tags.update(tags)

        self._run_start_time = time.time()
        logger.info(f"WandB 实验运行开始: {run_id}")
        return run_id

    def end_run(self, status: str = "completed"):
        if self._run:
            if self.active_run:
                self.active_run.end_time = time.time()
                self.active_run.status = status
                logger.info(f"WandB 实验运行结束: {self.active_run.run_id}, 状态: {status}")
            self._run.finish()
            self._run = None
        self.active_run = None

    def log_param(self, key: str, value: Any):
        if self._run:
            self._run.config[key] = value
        if self.active_run:
            self.active_run.params[key] = value

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if self._run:
            if step is not None:
                self._run.log({key: value, "step": step})
            else:
                self._run.log({key: value})
        if self.active_run:
            if key not in self.active_run.metrics:
                self.active_run.metrics[key] = []
            entry = {"value": float(value), "timestamp": time.time()}
            if step is not None:
                entry["step"] = step
            self.active_run.metrics[key].append(entry)

    def set_tag(self, key: str, value: str):
        if self._run:
            self._run.tags = list(self._run.tags) + [f"{key}:{value}"] if self._run.tags else [f"{key}:{value}"]
        if self.active_run:
            self.active_run.tags[key] = str(value)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if self._run:
            artifact = self._wandb.Artifact(name=artifact_path or Path(local_path).stem, type="dataset")
            if Path(local_path).is_dir():
                artifact.add_dir(local_path)
            else:
                artifact.add_file(local_path)
            self._run.log_artifact(artifact)
        if self.active_run:
            self.active_run.artifacts.append(local_path)

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        try:
            api = self._wandb.Api()
            run = api.run(f"{self._entity}/{self._project}/{run_id}")
            return ExperimentRun(
                run_id=run.id,
                experiment_name=self.experiment_name,
                start_time=run.created_at.timestamp() if hasattr(run.created_at, 'timestamp') else time.time(),
                status=run.state,
                params=dict(run.config),
            )
        except Exception:
            return None

    def list_runs(self, experiment_name: Optional[str] = None) -> List[ExperimentRun]:
        try:
            api = self._wandb.Api()
            runs = api.runs(f"{self._entity}/{self._project or experiment_name}")
            result = []
            for run in runs:
                result.append(ExperimentRun(
                    run_id=run.id,
                    experiment_name=self.experiment_name,
                    start_time=run.created_at.timestamp() if hasattr(run.created_at, 'timestamp') else time.time(),
                    status=run.state,
                    params=dict(run.config),
                ))
            return result
        except Exception:
            return []

    def compare_runs(self, run_ids: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        runs_data = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is None:
                continue
            runs_data.append({
                "run_id": run.run_id,
                "duration": run.duration,
                "status": run.status,
                "params": run.params,
                "tags": run.tags,
            })

        return {
            "experiment_name": self.experiment_name,
            "runs": runs_data,
            "compared_run_ids": run_ids,
        }


def create_tracker(backend: str = "local", **kwargs) -> BaseExperimentTracker:
    """
    创建实验追踪器

    Args:
        backend: 追踪后端: 'local', 'mlflow', 'wandb'
        **kwargs: 传递给追踪器的额外参数

    Returns:
        实验追踪器实例
    """
    backend = backend.lower()

    if backend == "local":
        return LocalFileTracker(**kwargs)
    elif backend == "mlflow":
        return MLflowTracker(**kwargs)
    elif backend == "wandb":
        return WandBTracker(**kwargs)
    else:
        raise ValueError(f"未知的实验追踪后端: {backend}，支持: local, mlflow, wandb")


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """将嵌套字典展平"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def _convert_numpy_types(obj: Any) -> Any:
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def list_experiments(tracking_dir: str = "./mlruns") -> List[str]:
    """
    列出所有实验名称（仅本地后端）

    Args:
        tracking_dir: 追踪目录

    Returns:
        实验名称列表
    """
    tracking_path = Path(tracking_dir)
    if not tracking_path.exists():
        return []

    experiments = []
    for exp_dir in tracking_path.iterdir():
        if exp_dir.is_dir():
            experiments.append(exp_dir.name)

    return sorted(experiments)


def get_run_summary(run: ExperimentRun) -> Dict[str, Any]:
    """
    获取运行摘要信息

    Args:
        run: 实验运行

    Returns:
        摘要字典
    """
    summary = {
        "run_id": run.run_id,
        "experiment_name": run.experiment_name,
        "status": run.status,
        "duration_seconds": round(run.duration, 2),
        "start_time": datetime.fromtimestamp(run.start_time).strftime('%Y-%m-%d %H:%M:%S'),
    }

    if run.end_time:
        summary["end_time"] = datetime.fromtimestamp(run.end_time).strftime('%Y-%m-%d %H:%M:%S')

    if run.tags:
        summary["tags"] = run.tags

    if run.params:
        summary["params"] = run.params

    if run.metrics:
        metric_summary = {}
        for name, history in run.metrics.items():
            if history:
                values = [h["value"] for h in history]
                metric_summary[name] = {
                    "final": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "num_steps": len(values),
                }
        summary["metrics"] = metric_summary

    return summary


def print_run_summary(run: ExperimentRun):
    """
    打印运行摘要

    Args:
        run: 实验运行
    """
    summary = get_run_summary(run)

    print("=" * 60)
    print(f"实验运行: {summary['run_id']}")
    print(f"实验名称: {summary['experiment_name']}")
    print(f"状态: {summary['status']}")
    print(f"开始时间: {summary['start_time']}")
    if 'end_time' in summary:
        print(f"结束时间: {summary['end_time']}")
    print(f"耗时: {summary['duration_seconds']:.2f} 秒")
    print("-" * 60)

    if 'params' in summary and summary['params']:
        print("参数:")
        for k, v in sorted(summary['params'].items()):
            print(f"  {k}: {v}")
        print("-" * 60)

    if 'tags' in summary and summary['tags']:
        print("标签:")
        for k, v in sorted(summary['tags'].items()):
            print(f"  {k}: {v}")
        print("-" * 60)

    if 'metrics' in summary and summary['metrics']:
        print("指标:")
        for name, m in sorted(summary['metrics'].items()):
            print(f"  {name}:")
            print(f"    最终值: {m['final']:.6e}")
            print(f"    最小值: {m['min']:.6e}")
            print(f"    最大值: {m['max']:.6e}")
            print(f"    步数: {m['num_steps']}")
        print("-" * 60)

    print("=" * 60)


def compare_runs_table(runs: List[ExperimentRun],
                       metrics: Optional[List[str]] = None,
                       params: Optional[List[str]] = None) -> str:
    """
    生成多次运行的对比表格（Markdown 格式）

    Args:
        runs: 运行列表
        metrics: 要对比的指标名称列表，None 表示全部
        params: 要对比的参数名称列表，None 表示全部

    Returns:
        Markdown 表格字符串
    """
    if not runs:
        return "没有可对比的运行"

    all_metric_names = set()
    all_param_names = set()
    for run in runs:
        all_metric_names.update(run.metrics.keys())
        all_param_names.update(run.params.keys())

    metric_names = sorted(metrics or all_metric_names)
    param_names = sorted(params or all_param_names)

    headers = ["run_id", "duration(s)", "status"]
    if param_names:
        headers += [f"p:{p}" for p in param_names]
    if metric_names:
        headers += [f"m:{m} (final)" for m in metric_names]
        headers += [f"m:{m} (min)" for m in metric_names]

    rows = []
    for run in runs:
        row = [
            run.run_id,
            f"{run.duration:.2f}",
            run.status,
        ]
        if param_names:
            for p in param_names:
                row.append(str(run.params.get(p, "-")))
        if metric_names:
            for m in metric_names:
                history = run.metrics.get(m, [])
                if history:
                    row.append(f"{history[-1]['value']:.6e}")
                else:
                    row.append("-")
            for m in metric_names:
                history = run.metrics.get(m, [])
                if history:
                    values = [h['value'] for h in history]
                    row.append(f"{min(values):.6e}")
                else:
                    row.append("-")
        rows.append(row)

    def format_table(header, data):
        col_widths = [len(h) for h in header]
        for row in data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        def format_row(cells):
            return "| " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

        lines = [
            format_row(header),
            "|" + "|".join("-" * (w + 2) for w in col_widths) + "|",
        ]
        for row in data:
            lines.append(format_row(row))
        return "\n".join(lines)

    return format_table(headers, rows)


def export_comparison_to_csv(runs: List[ExperimentRun],
                             output_path: str,
                             metrics: Optional[List[str]] = None,
                             params: Optional[List[str]] = None):
    """
    导出对比结果为 CSV 文件

    Args:
        runs: 运行列表
        output_path: 输出文件路径
        metrics: 要对比的指标名称列表
        params: 要对比的参数名称列表
    """
    import csv

    all_metric_names = set()
    all_param_names = set()
    for run in runs:
        all_metric_names.update(run.metrics.keys())
        all_param_names.update(run.params.keys())

    metric_names = sorted(metrics or all_metric_names)
    param_names = sorted(params or all_param_names)

    headers = ["run_id", "start_time", "duration_seconds", "status"]
    headers += [f"param_{p}" for p in param_names]
    for m in metric_names:
        headers.append(f"metric_{m}_final")
        headers.append(f"metric_{m}_min")
        headers.append(f"metric_{m}_max")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for run in runs:
            row = [
                run.run_id,
                datetime.fromtimestamp(run.start_time).isoformat(),
                f"{run.duration:.4f}",
                run.status,
            ]
            for p in param_names:
                row.append(run.params.get(p, ""))
            for m in metric_names:
                history = run.metrics.get(m, [])
                if history:
                    values = [h['value'] for h in history]
                    row.append(f"{values[-1]:.10e}")
                    row.append(f"{min(values):.10e}")
                    row.append(f"{max(values):.10e}")
                else:
                    row.append("")
                    row.append("")
                    row.append("")
            writer.writerow(row)

    logger.info(f"对比结果已导出到: {output_path}")


def filter_runs(runs: List[ExperimentRun],
                tags: Optional[Dict[str, str]] = None,
                params: Optional[Dict[str, Any]] = None,
                status: Optional[str] = None) -> List[ExperimentRun]:
    """
    按条件过滤运行

    Args:
        runs: 运行列表
        tags: 按标签过滤
        params: 按参数过滤
        status: 按状态过滤

    Returns:
        过滤后的运行列表
    """
    filtered = []

    for run in runs:
        if status and run.status != status:
            continue

        if tags:
            match = True
            for k, v in tags.items():
                if run.tags.get(k) != v:
                    match = False
                    break
            if not match:
                continue

        if params:
            match = True
            for k, v in params.items():
                if str(run.params.get(k)) != str(v):
                    match = False
                    break
            if not match:
                continue

        filtered.append(run)

    return filtered


def find_best_run(runs: List[ExperimentRun],
                  metric_name: str,
                  mode: str = 'min') -> Optional[ExperimentRun]:
    """
    找到指标最优的运行

    Args:
        runs: 运行列表
        metric_name: 指标名称
        mode: 'min' 或 'max'

    Returns:
        最优的运行
    """
    best_run = None
    best_value = float('inf') if mode == 'min' else float('-inf')

    for run in runs:
        history = run.metrics.get(metric_name, [])
        if not history:
            continue

        if mode == 'min':
            current_value = min(h['value'] for h in history)
            if current_value < best_value:
                best_value = current_value
                best_run = run
        else:
            current_value = max(h['value'] for h in history)
            if current_value > best_value:
                best_value = current_value
                best_run = run

    return best_run
