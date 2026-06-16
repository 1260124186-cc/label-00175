import sys
import os
import uuid
import yaml
import json
import time
import logging
import asyncio
import base64
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from io import BytesIO

from fastapi import HTTPException

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = BACKEND_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.yaml"

API_CONFIG_DIR = Path(__file__).resolve().parent / "saved_configs"
API_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

TASK_RESULTS_DIR = Path(__file__).resolve().parent / "task_results"
TASK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUNNING_TASKS: Dict[str, Dict[str, Any]] = {}

_ws_event_loop: Optional[asyncio.AbstractEventLoop] = None
_ws_loop_lock = threading.Lock()


def _get_ws_event_loop() -> asyncio.AbstractEventLoop:
    """
    获取或创建用于 WebSocket 推送的事件循环

    在单独的线程中运行事件循环，以便从同步任务线程中推送消息。
    """
    global _ws_event_loop

    with _ws_loop_lock:
        if _ws_event_loop is None:
            loop = asyncio.new_event_loop()
            _ws_event_loop = loop

            def run_loop():
                asyncio.set_event_loop(loop)
                loop.run_forever()

            thread = threading.Thread(target=run_loop, daemon=True, name="ws-push-loop")
            thread.start()
            logger.info("WebSocket 推送事件循环已启动")

        return _ws_event_loop


def _run_async(coro):
    """
    在 WebSocket 事件循环中异步执行协程（从同步线程调用）

    Args:
        coro: 要执行的协程
    """
    loop = _get_ws_event_loop()
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as e:
        logger.debug(f"WebSocket 推送失败: {e}")


def _generate_mask_thumbnail(mask: Any, max_size: int = 64) -> Optional[str]:
    """
    生成掩模缩略图的 base64 编码

    Args:
        mask: 掩模数组
        max_size: 最大尺寸

    Returns:
        base64 编码的 PNG 图像字符串，失败则返回 None
    """
    try:
        import numpy as np
        from PIL import Image

        if not hasattr(mask, 'shape') or mask.ndim != 2:
            return None

        # 缩放到缩略图尺寸
        h, w = mask.shape
        scale = min(max_size / h, max_size / w, 1.0)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

        # 归一化到 0-255
        mask_norm = np.clip(mask, 0.0, 1.0)
        mask_uint8 = (mask_norm * 255).astype(np.uint8)

        # 缩放
        from scipy.ndimage import zoom
        mask_small = zoom(mask_uint8, (new_h / h, new_w / w), order=1)
        mask_small = np.clip(mask_small, 0, 255).astype(np.uint8)

        # 生成 PNG
        img = Image.fromarray(mask_small, mode='L')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()

        return base64.b64encode(img_bytes).decode('ascii')
    except Exception as e:
        logger.debug(f"生成掩模缩略图失败: {e}")
        return None


def _push_progress_ws(
    task_id: str,
    progress: float,
    message: Optional[str] = None,
    stage: Optional[str] = None,
    loss: Optional[float] = None,
    iteration: Optional[int] = None,
    mask_thumbnail: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
):
    """
    通过 WebSocket 推送进度更新（同步接口，内部异步执行）

    Args:
        task_id: 任务 ID
        progress: 进度百分比
        message: 消息
        stage: 阶段
        loss: 损失值
        iteration: 迭代次数
        mask_thumbnail: 掩模缩略图 base64
        extra: 额外数据
    """
    from websocket_manager import broadcast_progress

    _run_async(broadcast_progress(
        task_id=task_id,
        progress=progress,
        message=message,
        stage=stage,
        loss=loss,
        iteration=iteration,
        mask_thumbnail=mask_thumbnail,
        extra=extra,
    ))


def _push_stage_change_ws(task_id: str, stage: str, message: Optional[str] = None):
    """
    通过 WebSocket 推送阶段变化

    Args:
        task_id: 任务 ID
        stage: 新阶段
        message: 消息
    """
    from websocket_manager import broadcast_stage_change

    _run_async(broadcast_stage_change(task_id, stage, message))


def _push_task_complete_ws(task_id: str, result: Optional[Dict[str, Any]] = None):
    """
    通过 WebSocket 推送任务完成

    Args:
        task_id: 任务 ID
        result: 结果摘要
    """
    from websocket_manager import broadcast_task_complete

    _run_async(broadcast_task_complete(task_id, result))


def _push_task_failed_ws(task_id: str, error: str):
    """
    通过 WebSocket 推送任务失败

    Args:
        task_id: 任务 ID
        error: 错误信息
    """
    from websocket_manager import broadcast_task_failed

    _run_async(broadcast_task_failed(task_id, error))


def _register_task(task_type: str, payload: Dict[str, Any]) -> str:
    task_id = uuid.uuid4().hex[:12]
    RUNNING_TASKS[task_id] = {
        "task_id": task_id,
        "task_type": task_type,
        "status": "pending",
        "progress": 0.0,
        "message": None,
        "error": None,
        "result": None,
        "result_summary": None,
        "payload": payload,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
    }
    return task_id


def _start_task(task_id: str):
    task = RUNNING_TASKS.get(task_id)
    if task:
        task["status"] = "running"
        task["started_at"] = time.time()
    _push_stage_change_ws(task_id, "running", "任务开始执行")


def _finish_task(task_id: str, result: Any = None, summary: Optional[Dict[str, Any]] = None):
    task = RUNNING_TASKS.get(task_id)
    if task:
        task["status"] = "completed"
        task["progress"] = 100.0
        task["result"] = result
        task["result_summary"] = summary
        task["finished_at"] = time.time()
        _persist_task_result(task_id, task)
    _push_task_complete_ws(task_id, summary or {})


def _fail_task(task_id: str, error: str):
    task = RUNNING_TASKS.get(task_id)
    if task:
        task["status"] = "failed"
        task["error"] = error
        task["finished_at"] = time.time()
        _persist_task_result(task_id, task)
    _push_task_failed_ws(task_id, error)


def _set_progress(task_id: str, progress: float, message: Optional[str] = None,
                  stage: Optional[str] = None, loss: Optional[float] = None,
                  iteration: Optional[int] = None, mask: Optional[Any] = None,
                  extra: Optional[Dict[str, Any]] = None):
    task = RUNNING_TASKS.get(task_id)
    if task:
        task["progress"] = max(0.0, min(100.0, float(progress)))
        if message is not None:
            task["message"] = message
        if stage is not None:
            task["stage"] = stage
        if loss is not None:
            task["current_loss"] = loss
        if iteration is not None:
            task["iteration"] = iteration

    # 生成掩模缩略图（如果有掩模且有 WebSocket 连接）
    mask_thumbnail = None
    if mask is not None:
        from websocket_manager import manager
        if manager.has_connections(task_id):
            mask_thumbnail = _generate_mask_thumbnail(mask, max_size=64)

    # 推送 WebSocket 消息
    _push_progress_ws(
        task_id=task_id,
        progress=max(0.0, min(100.0, float(progress))),
        message=message,
        stage=stage,
        loss=loss,
        iteration=iteration,
        mask_thumbnail=mask_thumbnail,
        extra=extra,
    )


def _persist_task_result(task_id: str, task: Dict[str, Any]):
    try:
        out_path = TASK_RESULTS_DIR / f"{task_id}.json"
        serializable = {
            "task_id": task.get("task_id"),
            "task_type": task.get("task_type"),
            "status": task.get("status"),
            "progress": task.get("progress"),
            "message": task.get("message"),
            "error": task.get("error"),
            "result_summary": task.get("result_summary"),
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning(f"持久化任务结果失败 {task_id}: {e}")


def _build_optical_system(opt_sys_cfg: Dict[str, Any]):
    from core.imaging import OpticalSystem

    def _f(cfg, key, default):
        v = cfg.get(key, default)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    return OpticalSystem(
        wavelength=_f(opt_sys_cfg, "wavelength", 193.0),
        na=_f(opt_sys_cfg, "na", 1.35),
        sigma=_f(opt_sys_cfg, "sigma", 0.75),
        pixel_size=_f(opt_sys_cfg, "pixel_size", 1.0),
        defocus=_f(opt_sys_cfg, "defocus", 0.0),
        magnification=_f(opt_sys_cfg, "magnification", 4.0),
        illumination_type=opt_sys_cfg.get("illumination_type", "conventional"),
        source_params=dict(opt_sys_cfg.get("source_params", {})),
        tcc_mode=opt_sys_cfg.get("tcc_mode", "socs"),
        socs_num_terms=int(opt_sys_cfg.get("socs_num_terms", 5)),
        zernike_coefficients=dict(opt_sys_cfg.get("zernike_coefficients", {})),
    )


def _create_test_pattern(pattern_type: str, pattern_params: Dict[str, Any]):
    from utils.data_io import create_test_pattern

    raw_size = pattern_params.get("size", [64, 64])
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) < 2:
        raw_size = [64, 64]
    try:
        size_h = max(8, int(raw_size[0]))
        size_w = max(8, int(raw_size[1]))
    except (TypeError, ValueError):
        size_h, size_w = 64, 64
    size = (size_h, size_w)

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

    target_pattern = create_test_pattern(
        pattern_type,
        size=size,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
    )
    import numpy as np
    if target_pattern is None or not hasattr(target_pattern, "shape"):
        target_pattern = np.zeros(size, dtype=np.float32)
    return target_pattern


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


def run_opc(payload: Dict[str, Any]) -> str:
    import threading
    task_id = _register_task("opc", payload)
    thread = threading.Thread(target=_execute_opc, args=(task_id,), daemon=True)
    thread.start()
    return task_id


def _execute_opc(task_id: str):
    task = RUNNING_TASKS.get(task_id)
    if not task:
        return
    try:
        add_backend_to_path()
        _start_task(task_id)
        _set_progress(task_id, 5, "准备 OPC 工作流参数")

        payload = task["payload"]
        opt_sys_dict = payload.get("optical_system") or {}
        opc_cfg_dict = payload.get("opc_config") or {}
        pattern_type = payload.get("pattern_type") or "rectangle"
        pattern_params = payload.get("pattern_params") or {}

        _set_progress(task_id, 15, "构造光学系统与测试图案")
        optical_system = _build_optical_system(opt_sys_dict)
        target = _create_test_pattern(pattern_type, pattern_params)
        import numpy as np
        initial_mask = np.array(target, dtype=np.float32)

        _set_progress(task_id, 30, "加载 OPC 配置")
        from workflows.opc import OPCConfig, run_opc_workflow
        opc_config = OPCConfig.from_dict(opc_cfg_dict)

        _set_progress(task_id, 45, "执行 OPC 工作流...")
        result = run_opc_workflow(
            initial_mask=initial_mask,
            target=target,
            config=opc_config,
            optical_system=optical_system,
        )

        _set_progress(task_id, 85, "整理 OPC 结果")
        summary = {}
        try:
            final_metrics = getattr(result, "final_metrics", None)
            if final_metrics is not None:
                summary["final_metrics"] = {
                    "mse": float(getattr(final_metrics, "mse", 0.0) or 0.0),
                    "ssim": float(getattr(final_metrics, "ssim", 0.0) or 0.0),
                }
            summary["iterations"] = int(getattr(result, "iterations", 0) or 0)
            summary["hotspots_detected"] = int(getattr(result, "hotspots_detected", 0) or 0)
            summary["hotspots_remaining"] = int(getattr(result, "hotspots_remaining", 0) or 0)
            optimized_mask = getattr(result, "optimized_mask", None)
            if optimized_mask is not None and hasattr(optimized_mask, "shape"):
                summary["mask_shape"] = list(optimized_mask.shape)
        except Exception as e:
            logger.warning(f"OPC 结果摘要提取失败: {e}")

        _finish_task(task_id, result={"task_id": task_id, **summary}, summary=summary)
    except Exception as e:
        logger.exception(f"OPC 任务失败: {task_id}")
        _fail_task(task_id, f"{type(e).__name__}: {e}")


def run_smo(payload: Dict[str, Any]) -> str:
    import threading
    task_id = _register_task("smo", payload)
    thread = threading.Thread(target=_execute_smo, args=(task_id,), daemon=True)
    thread.start()
    return task_id


def _execute_smo(task_id: str):
    task = RUNNING_TASKS.get(task_id)
    if not task:
        return
    try:
        add_backend_to_path()
        _start_task(task_id)
        _set_progress(task_id, 5, "准备 SMO 工作流参数")

        payload = task["payload"]
        opt_sys_dict = payload.get("optical_system") or {}
        smo_cfg_dict = payload.get("smo_config") or {}
        pattern_type = payload.get("pattern_type") or "rectangle"
        pattern_params = payload.get("pattern_params") or {}

        _set_progress(task_id, 15, "构造光学系统与测试图案")
        optical_system = _build_optical_system(opt_sys_dict)
        target = _create_test_pattern(pattern_type, pattern_params)
        import numpy as np
        initial_mask = np.array(target, dtype=np.float32)

        _set_progress(task_id, 30, "加载 SMO 配置")
        from workflows.smo import SMOConfig, run_smo_workflow
        smo_config = SMOConfig.from_dict(smo_cfg_dict)

        _set_progress(task_id, 45, "执行 SMO 工作流（光源-掩模协同优化）...")
        result = run_smo_workflow(
            initial_mask=initial_mask,
            target=target,
            config=smo_config,
            optical_system=optical_system,
        )

        _set_progress(task_id, 85, "整理 SMO 结果")
        summary = {}
        try:
            summary["iterations"] = int(getattr(result, "iterations", 0) or 0)
            final_loss = getattr(result, "final_loss", None)
            if final_loss is not None:
                summary["final_loss"] = float(final_loss)
            optimized_mask = getattr(result, "optimized_mask", None)
            if optimized_mask is not None and hasattr(optimized_mask, "shape"):
                summary["mask_shape"] = list(optimized_mask.shape)
            optimized_source = getattr(result, "optimized_source", None)
            if optimized_source is not None and hasattr(optimized_source, "shape"):
                summary["source_shape"] = list(optimized_source.shape)
        except Exception as e:
            logger.warning(f"SMO 结果摘要提取失败: {e}")

        _finish_task(task_id, result={"task_id": task_id, **summary}, summary=summary)
    except Exception as e:
        logger.exception(f"SMO 任务失败: {task_id}")
        _fail_task(task_id, f"{type(e).__name__}: {e}")


def run_ilt(payload: Dict[str, Any]) -> str:
    import threading
    task_id = _register_task("ilt", payload)
    thread = threading.Thread(target=_execute_ilt, args=(task_id,), daemon=True)
    thread.start()
    return task_id


def _execute_ilt(task_id: str):
    task = RUNNING_TASKS.get(task_id)
    if not task:
        return
    try:
        add_backend_to_path()
        _start_task(task_id)
        _set_progress(task_id, 5, "准备 ILT 工作流参数")

        payload = task["payload"]
        opt_sys_dict = payload.get("optical_system") or {}
        ilt_cfg_dict = payload.get("ilt_config") or {}
        pattern_type = payload.get("pattern_type") or "rectangle"
        pattern_params = payload.get("pattern_params") or {}

        _set_progress(task_id, 15, "构造光学系统与测试图案")
        optical_system = _build_optical_system(opt_sys_dict)
        target = _create_test_pattern(pattern_type, pattern_params)
        import numpy as np
        initial_mask = np.array(target, dtype=np.float32)

        _set_progress(task_id, 30, "加载 ILT 配置")
        from workflows.ilt import ILTConfig, run_ilt_workflow
        ilt_config = ILTConfig.from_dict(ilt_cfg_dict)

        _set_progress(task_id, 45, "执行 ILT 工作流（反演光刻技术）...")
        result = run_ilt_workflow(
            initial_mask=initial_mask,
            target=target,
            optical_system=optical_system,
            config=ilt_config,
        )

        _set_progress(task_id, 85, "整理 ILT 结果")
        summary = {}
        try:
            summary["iterations"] = int(getattr(result, "iterations", 0) or 0)
            final_loss = getattr(result, "final_loss", None)
            if final_loss is not None:
                summary["final_loss"] = float(final_loss)
            optimized_mask = getattr(result, "optimized_mask", None)
            if optimized_mask is not None and hasattr(optimized_mask, "shape"):
                summary["mask_shape"] = list(optimized_mask.shape)
            transmission_levels = getattr(result, "transmission_levels", None)
            if transmission_levels is not None:
                summary["transmission_levels"] = [float(x) for x in list(transmission_levels)]
        except Exception as e:
            logger.warning(f"ILT 结果摘要提取失败: {e}")

        _finish_task(task_id, result={"task_id": task_id, **summary}, summary=summary)
    except Exception as e:
        logger.exception(f"ILT 任务失败: {task_id}")
        _fail_task(task_id, f"{type(e).__name__}: {e}")


def run_process_window(payload: Dict[str, Any]) -> str:
    import threading
    task_id = _register_task("process_window", payload)
    thread = threading.Thread(target=_execute_process_window, args=(task_id,), daemon=True)
    thread.start()
    return task_id


def _execute_process_window(task_id: str):
    task = RUNNING_TASKS.get(task_id)
    if not task:
        return
    try:
        add_backend_to_path()
        _start_task(task_id)
        _set_progress(task_id, 5, "准备工艺窗口分析参数")

        payload = task["payload"]
        opt_sys_dict = payload.get("optical_system") or {}
        pattern_type = payload.get("pattern_type") or "rectangle"
        pattern_params = payload.get("pattern_params") or {}

        raw_focus = payload.get("focus_range", [-150.0, 150.0, 11])
        raw_dose = payload.get("dose_range", [0.85, 1.15, 11])
        cd_tolerance = float(payload.get("cd_tolerance", 0.1))
        epe_tolerance = payload.get("epe_tolerance", None)
        if epe_tolerance is not None:
            epe_tolerance = float(epe_tolerance)
        threshold = float(payload.get("threshold", 0.3))
        save_vis = bool(payload.get("save_visualizations", False))

        def _parse_range(r):
            if isinstance(r, (list, tuple)) and len(r) >= 3:
                try:
                    return (float(r[0]), float(r[1]), max(2, int(r[2])))
                except (TypeError, ValueError):
                    pass
            return (-150.0, 150.0, 11) if r is raw_focus else (0.85, 1.15, 11)

        focus_range = _parse_range(raw_focus)
        dose_range = _parse_range(raw_dose)

        _set_progress(task_id, 15, "构造光学系统与测试图案")
        optical_system = _build_optical_system(opt_sys_dict)
        target = _create_test_pattern(pattern_type, pattern_params)
        import numpy as np
        mask = np.array(target, dtype=np.float32)

        _set_progress(task_id, 35, "执行工艺窗口分析（focus-dose 扫描）...")
        from analysis.process_window import quick_process_window_analysis
        output_dir = str(TASK_RESULTS_DIR / task_id) if save_vis else None
        if save_vis:
            os.makedirs(output_dir, exist_ok=True)
        pixel_size = float(opt_sys_dict.get("pixel_size", 1.0))

        result = quick_process_window_analysis(
            mask=mask,
            target=target,
            optical_system=optical_system,
            focus_range=focus_range,
            dose_range=dose_range,
            cd_tolerance=cd_tolerance,
            pixel_size=pixel_size,
            threshold=threshold,
            output_dir=output_dir,
            show=False,
        )

        _set_progress(task_id, 85, "整理工艺窗口分析结果")
        summary = {}
        try:
            if isinstance(result, dict):
                pw_metrics = result.get("pw_metrics")
                if pw_metrics is not None:
                    for k in ["max_exposure_latitude", "depth_of_focus", "process_window_area",
                              "nominal_cd", "cd_uniformity"]:
                        v = getattr(pw_metrics, k, None)
                        if v is not None:
                            try:
                                summary[k] = float(v)
                            except (TypeError, ValueError):
                                pass
                for k in ["focus_points", "dose_points"]:
                    v = result.get(k)
                    if v is not None:
                        try:
                            summary[k] = int(len(v))
                        except TypeError:
                            pass
        except Exception as e:
            logger.warning(f"工艺窗口结果摘要提取失败: {e}")

        _finish_task(task_id, result={"task_id": task_id, **summary}, summary=summary)
    except Exception as e:
        logger.exception(f"工艺窗口任务失败: {task_id}")
        _fail_task(task_id, f"{type(e).__name__}: {e}")


def run_batch(payload: Dict[str, Any]) -> str:
    import threading
    task_id = _register_task("batch", payload)
    thread = threading.Thread(target=_execute_batch, args=(task_id,), daemon=True)
    thread.start()
    return task_id


def _execute_batch(task_id: str):
    task = RUNNING_TASKS.get(task_id)
    if not task:
        return
    try:
        add_backend_to_path()
        _start_task(task_id)
        _set_progress(task_id, 5, "准备批处理优化参数")

        payload = task["payload"]
        source = payload.get("source")
        if not source:
            raise ValueError("批处理必须提供 source 参数")
        layer = payload.get("layer", None)
        if layer is not None:
            layer = int(layer)
        opt_sys_dict = payload.get("optical_system") or {}
        opt_dict = payload.get("optimization") or {}
        max_workers = payload.get("max_workers", None)
        if max_workers is not None:
            max_workers = int(max_workers)
        max_retries = int(payload.get("max_retries", 2))
        save_optimized_masks = bool(payload.get("save_optimized_masks", True))
        output_dir = payload.get("output_dir", None)
        stop_on_first_failure = bool(payload.get("stop_on_first_failure", False))

        _set_progress(task_id, 20, "构造批处理配置")
        from pipeline.batch_runner import (
            run_batch_optimization, ResourceConfig, BatchConfig,
        )
        resource_config = ResourceConfig()
        if max_workers is not None:
            resource_config.max_workers = max_workers
        resource_config.max_retries = max_retries

        batch_config = BatchConfig()
        batch_config.stop_on_first_failure = stop_on_first_failure
        batch_config.save_optimized_masks = save_optimized_masks

        layout_options = {
            "optical_system": opt_sys_dict,
            "optimization": opt_dict,
        }

        _set_progress(task_id, 35, "执行批处理优化...")
        summary_obj, task_results, lib, queue = run_batch_optimization(
            source=source,
            layer=layer,
            layout_options=layout_options,
            resource_config=resource_config,
            batch_config=batch_config,
            mode="local",
            output_dir=output_dir,
        )

        _set_progress(task_id, 85, "整理批处理结果")
        summary = {}
        try:
            summary["total"] = int(getattr(summary_obj, "total", 0) or 0)
            summary["succeeded"] = int(getattr(summary_obj, "succeeded", 0) or 0)
            summary["failed"] = int(getattr(summary_obj, "failed", 0) or 0)
            summary["skipped"] = int(getattr(summary_obj, "skipped", 0) or 0)
            avg_mse = getattr(summary_obj, "avg_mse", None)
            if avg_mse is not None:
                summary["avg_mse"] = float(avg_mse)
            avg_ssim = getattr(summary_obj, "avg_ssim", None)
            if avg_ssim is not None:
                summary["avg_ssim"] = float(avg_ssim)
            elapsed = getattr(summary_obj, "elapsed_seconds", None)
            if elapsed is not None:
                summary["elapsed_seconds"] = float(elapsed)
        except Exception as e:
            logger.warning(f"批处理结果摘要提取失败: {e}")

        _finish_task(task_id, result={"task_id": task_id, **summary}, summary=summary)
    except Exception as e:
        logger.exception(f"批处理任务失败: {task_id}")
        _fail_task(task_id, f"{type(e).__name__}: {e}")


def get_task_status(task_id: str) -> Dict[str, Any]:
    task = RUNNING_TASKS.get(task_id)
    if not task:
        persisted = TASK_RESULTS_DIR / f"{task_id}.json"
        if persisted.exists():
            try:
                with open(persisted, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"读取持久化任务失败 {task_id}: {e}")
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return {
        "task_id": task.get("task_id", task_id),
        "task_type": task.get("task_type", "unknown"),
        "status": task.get("status", "unknown"),
        "progress": task.get("progress", 0.0),
        "message": task.get("message"),
        "error": task.get("error"),
        "stage": task.get("stage"),
        "current_loss": task.get("current_loss"),
        "iteration": task.get("iteration"),
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "result_summary": task.get("result_summary"),
    }


def get_task_result(task_id: str) -> Dict[str, Any]:
    task = RUNNING_TASKS.get(task_id)
    if not task:
        persisted = TASK_RESULTS_DIR / f"{task_id}.json"
        if persisted.exists():
            try:
                with open(persisted, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {"task_id": task_id, "status": data.get("status"), "result": data.get("result_summary")}
            except Exception as e:
                logger.warning(f"读取持久化任务结果失败 {task_id}: {e}")
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    if task.get("status") not in ("completed", "failed"):
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态: {task.get('status')}")
    return {
        "task_id": task_id,
        "task_type": task.get("task_type"),
        "status": task.get("status"),
        "result": task.get("result"),
        "result_summary": task.get("result_summary"),
        "error": task.get("error"),
    }


def get_task_download_path(task_id: str) -> Path:
    persisted = TASK_RESULTS_DIR / f"{task_id}.json"
    if not persisted.exists():
        task = RUNNING_TASKS.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        if task.get("status") not in ("completed", "failed"):
            raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态: {task.get('status')}")
        _persist_task_result(task_id, task)
    if not persisted.exists():
        raise HTTPException(status_code=404, detail=f"任务结果文件不存在")
    return persisted


def list_tasks(task_type: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
    tasks = []
    for tid, task in RUNNING_TASKS.items():
        t_type = task.get("task_type", "unknown")
        t_status = task.get("status", "unknown")
        if task_type and t_type != task_type:
            continue
        if status and t_status != status:
            continue
        tasks.append({
            "task_id": tid,
            "task_type": t_type,
            "status": t_status,
            "progress": task.get("progress", 0.0),
            "message": task.get("message"),
            "error": task.get("error"),
            "stage": task.get("stage"),
            "current_loss": task.get("current_loss"),
            "iteration": task.get("iteration"),
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
            "result_summary": task.get("result_summary"),
        })
    for pf in TASK_RESULTS_DIR.glob("*.json"):
        tid = pf.stem
        if tid in RUNNING_TASKS:
            continue
        try:
            with open(pf, "r", encoding="utf-8") as f:
                data = json.load(f)
            t_type = data.get("task_type", "unknown")
            t_status = data.get("status", "unknown")
            if task_type and t_type != task_type:
                continue
            if status and t_status != status:
                continue
            tasks.append({
                "task_id": tid,
                "task_type": t_type,
                "status": t_status,
                "progress": data.get("progress", 0.0),
                "message": data.get("message"),
                "error": data.get("error"),
                "stage": data.get("stage"),
                "current_loss": data.get("current_loss"),
                "iteration": data.get("iteration"),
                "created_at": data.get("created_at"),
                "started_at": data.get("started_at"),
                "finished_at": data.get("finished_at"),
                "result_summary": data.get("result_summary"),
            })
        except Exception as e:
            logger.warning(f"读取持久化任务列表失败 {pf.name}: {e}")
    tasks.sort(key=lambda t: t.get("created_at") or 0.0, reverse=True)
    return {"count": len(tasks), "tasks": tasks}
