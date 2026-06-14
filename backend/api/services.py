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
        task["progress"] = 5

        # ============================================================
        # 0. 参数准备与兜底（避免 NameError / TypeError）
        # ============================================================
        config = task["config"] or {}
        opt_sys_cfg = config.get("optical_system") or {}
        pattern_type = task.get("pattern_type") or "rectangle"
        pattern_params = task.get("pattern_params") or {}

        raw_size = pattern_params.get("size", [64, 64])
        if not isinstance(raw_size, (list, tuple)) or len(raw_size) < 2:
            raw_size = [64, 64]
        try:
            size_h = max(8, int(raw_size[0]))
            size_w = max(8, int(raw_size[1]))
        except (TypeError, ValueError):
            size_h, size_w = 64, 64
        size = (size_h, size_w)

        # 兜底：如果 x_end 超出图像尺寸，则 clip 以避免 create_test_pattern 内部异常
        def _int_or_none(v, max_val):
            if v is None:
                return None
            try:
                iv = int(v)
                return max(0, min(iv, max_val))
            except (TypeError, ValueError):
                return None

        x_start = _int_or_none(pattern_params.get("x_start"), size_w)
        x_end = _int_or_none(pattern_params.get("x_end"), size_w)
        y_start = _int_or_none(pattern_params.get("y_start"), size_h)
        y_end = _int_or_none(pattern_params.get("y_end"), size_h)

        task["progress"] = 15

        # ============================================================
        # 1. 导入核心模块（缺少 numpy/scipy 时会抛 ImportError）
        # ============================================================
        try:
            import numpy as np  # noqa: F401
            from utils.data_io import create_test_pattern
            from core.imaging import OpticalSystem, PartialCoherentImaging
            from core.metrics import evaluate_all
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(f"缺少依赖或后端模块导入失败: {e}") from e

        task["progress"] = 30

        # ============================================================
        # 2. 构造光学系统
        # ============================================================
        def _f(cfg, key, default):
            v = cfg.get(key, default)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        optical_system = OpticalSystem(
            wavelength=_f(opt_sys_cfg, "wavelength", 193.0),
            na=_f(opt_sys_cfg, "na", 1.35),
            sigma=_f(opt_sys_cfg, "sigma", 0.75),
            pixel_size=_f(opt_sys_cfg, "pixel_size", 1.0),
            defocus=_f(opt_sys_cfg, "defocus", 0.0)
        )

        task["progress"] = 45

        # ============================================================
        # 3. 生成测试图案
        # ============================================================
        target_pattern = create_test_pattern(
            pattern_type,
            size=size,
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end
        )
        # 兜底：任何异常都返回 0 数组
        if target_pattern is None or not hasattr(target_pattern, "shape"):
            import numpy as np
            target_pattern = np.zeros(size, dtype=np.float32)

        task["progress"] = 60

        # ============================================================
        # 4. 成像模拟
        # ============================================================
        imaging_model = PartialCoherentImaging(optical_system, size)
        initial_wafer_image = imaging_model.compute_aerial_image(target_pattern)
        initial_metrics = evaluate_all(initial_wafer_image, target_pattern)

        task["progress"] = 85

        # ============================================================
        # 5. 组装结果
        # ============================================================
        def _metric(name, default=None):
            try:
                v = getattr(initial_metrics, name, default)
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        result = {
            "task_id": task_id,
            "pattern_type": pattern_type,
            "pattern_size": [size_h, size_w],
            "initial_metrics": {
                "mse": _metric("mse", 0.0),
                "ssim": _metric("ssim", 0.0),
                "mae": _metric("mae"),
                "psnr": _metric("psnr"),
            },
            "target_pattern_shape": list(target_pattern.shape),
            "wafer_image_shape": list(initial_wafer_image.shape),
        }

        task["progress"] = 100
        task["status"] = "completed"
        task["result"] = result

    except Exception as e:
        logger.exception(f"仿真任务失败: {task_id}")
        if task is not None:
            task["status"] = "failed"
            task["error"] = f"{type(e).__name__}: {e}"
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
