import sys
import os
import uuid
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = BACKEND_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.yaml"

API_CONFIG_DIR = Path(__file__).resolve().parent / "saved_configs"
API_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

RUNNING_TASKS: Dict[str, Dict[str, Any]] = {}


def add_backend_to_path():
    backend_path = str(BACKEND_ROOT)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)


def load_default_config() -> Dict[str, Any]:
    add_backend_to_path()
    if not DEFAULT_CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="默认配置文件不存在")
    try:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"加载默认配置: {DEFAULT_CONFIG_PATH}")
        return config
    except yaml.YAMLError as e:
        raise HTTPException(status_code=500, detail=f"YAML解析错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载配置失败: {str(e)}")


def list_saved_configs() -> Dict[str, Any]:
    try:
        files = []
        for f in API_CONFIG_DIR.glob("*.yaml"):
            files.append({
                "filename": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            })
        for f in API_CONFIG_DIR.glob("*.yml"):
            files.append({
                "filename": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            })
        files.sort(key=lambda x: x["modified"], reverse=True)
        return {"count": len(files), "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"列出配置文件失败: {str(e)}")


def load_saved_config(filename: str) -> Dict[str, Any]:
    filepath = API_CONFIG_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"配置文件不存在: {filename}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return {"config": config, "filename": filename}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=500, detail=f"YAML解析错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载配置失败: {str(e)}")


def save_config_to_file(config: Dict[str, Any], filename: Optional[str] = None) -> str:
    if filename is None:
        filename = f"config_{uuid.uuid4().hex[:8]}.yaml"
    if not (filename.endswith(".yaml") or filename.endswith(".yml")):
        filename += ".yaml"
    filepath = API_CONFIG_DIR / filename
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"保存配置到: {filepath}")
        return str(filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")


def delete_saved_config(filename: str) -> None:
    filepath = API_CONFIG_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"配置文件不存在: {filename}")
    try:
        filepath.unlink()
        logger.info(f"删除配置文件: {filepath}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除配置失败: {str(e)}")


def run_simulation(config: Dict[str, Any], pattern_type: str, pattern_params: Dict[str, Any]) -> str:
    task_id = uuid.uuid4().hex[:12]
    RUNNING_TASKS[task_id] = {
        "status": "starting",
        "progress": 0,
        "config": config,
        "pattern_type": pattern_type,
        "pattern_params": pattern_params,
        "result": None,
        "error": None
    }
    import threading
    thread = threading.Thread(target=_execute_simulation, args=(task_id,), daemon=True)
    thread.start()
    return task_id


def _execute_simulation(task_id: str):
    task = RUNNING_TASKS.get(task_id)
    if not task:
        return
    try:
        add_backend_to_path()
        task["status"] = "running"
        task["progress"] = 10

        import numpy as np
        from utils.data_io import create_test_pattern
        from core.imaging import OpticalSystem, PartialCoherentImaging
        from core.metrics import evaluate_all, mse

        opt_sys_cfg = task["config"]["optical_system"]
        opt_cfg = task["config"].get("optimization", {})

        task["status"] = "running"
        task["progress"] = 20

        optical_system = OpticalSystem(
            wavelength=opt_sys_cfg["wavelength"],
            na=opt_sys_cfg["na"],
            sigma=opt_sys_cfg["sigma"],
            pixel_size=opt_sys_cfg.get("pixel_size", 1.0),
            defocus=opt_sys_cfg.get("defocus", 0.0)
        )

        task["progress"] = 35
        size = tuple(pattern_params.get("size", [64, 64]))
        target_pattern = create_test_pattern(
            pattern_type,
            size=size,
            x_start=pattern_params.get("x_start"),
            x_end=pattern_params.get("x_end"),
            y_start=pattern_params.get("y_start"),
            y_end=pattern_params.get("y_end")
        )

        task["progress"] = 50
        imaging_model = PartialCoherentImaging(optical_system, size)
        initial_wafer_image = imaging_model.compute_aerial_image(target_pattern)
        initial_metrics = evaluate_all(initial_wafer_image, target_pattern)

        task["progress"] = 80

        result = {
            "task_id": task_id,
            "initial_metrics": {
                "mse": float(initial_metrics.mse),
                "ssim": float(initial_metrics.ssim),
                "mae": float(initial_metrics.mae) if hasattr(initial_metrics, 'mae') else None,
            },
            "target_pattern_shape": list(target_pattern.shape),
            "wafer_image_shape": list(initial_wafer_image.shape),
        }

        task["progress"] = 100
        task["status"] = "completed"
        task["result"] = result

    except Exception as e:
        logger.exception(f"仿真任务失败: {task_id}")
        task["status"] = "failed"
        task["error"] = str(e)
        task["progress"] = 0


def get_task_status(task_id: str) -> Dict[str, Any]:
    task = RUNNING_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "result": task["result"],
        "error": task["error"]
    }


def list_tasks() -> Dict[str, Any]:
    tasks = []
    for tid, task in RUNNING_TASKS.items():
        tasks.append({
            "task_id": tid,
            "status": task["status"],
            "progress": task["progress"],
            "error": task["error"]
        })
    return {"count": len(tasks), "tasks": tasks}
