# -*- coding: utf-8 -*-
"""
批处理调度模块

支持从版图队列读取多个 cell，进行排队、并行优化、汇总结果。

调度模式：
1. LocalBatchRunner: 本地多进程（基于 ProcessPoolExecutor），单机部署首选
2. DistributedBatchRunner: 可选 Celery/Redis 分布式队列，多机/集群扩展

核心特性：
- 任务状态：pending / running / done / failed / cancelled
- 实时进度上报（基于回调）
- 失败自动重试（可配置次数与退避）
- 按可用 CPU/GPU 限制并发数
- 结果聚合：CSV/JSON 汇总表
  （cell 名、初始/最终 MSE/SSIM/EPE、耗时、收敛状态等）
- 容错：单任务失败不影响整批
"""

import os
import sys
import csv
import json
import time
import uuid
import signal
import logging
import threading
import traceback
from pathlib import Path
from typing import (
    Optional, List, Dict, Any, Union, Tuple, Callable, Type,
    Iterable,
)
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

import numpy as np

from layout.layout_manager import (
    LayoutCell, LayoutQueue, LayoutLibrary, LayoutLoadOptions,
    LayoutManager, GDSLoader,
)

logger = logging.getLogger(__name__)

try:
    from concurrent.futures import (
        ProcessPoolExecutor, ThreadPoolExecutor, Future, as_completed,
    )
    HAS_CONCURRENT = True
except ImportError:
    HAS_CONCURRENT = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from celery import Celery
    from celery.result import AsyncResult, GroupResult
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False


# ============================================================================
# 数据结构
# ============================================================================

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskResult:
    """
    单个优化任务的结果

    Attributes:
        task_id: 全局唯一任务 ID
        cell_name: cell 名（与 LayoutCell.name 对应）
        cell_display_name: 展示用 cell 名（GDS 原始 cell_name）
        status: 任务状态
        initial_mse: 优化前 MSE
        final_mse: 优化后 MSE
        initial_ssim: 优化前 SSIM
        final_ssim: 优化后 SSIM
        initial_epe: 优化前 EPE 均值 (nm)
        final_epe: 优化后 EPE 均值 (nm)
        iterations: 实际迭代次数
        elapsed_sec: 耗时（秒）
        converged: 是否收敛
        error_message: 失败时的错误信息
        retries: 实际重试次数
        worker_id: 执行的 worker 标识
        started_at: 开始时间戳
        finished_at: 结束时间戳
        extra: 额外字段（可扩展）
        optimized_mask_path: 若保存到磁盘则为路径
    """
    task_id: str
    cell_name: str
    cell_display_name: str = ""
    status: TaskStatus = TaskStatus.PENDING
    initial_mse: Optional[float] = None
    final_mse: Optional[float] = None
    initial_ssim: Optional[float] = None
    final_ssim: Optional[float] = None
    initial_epe: Optional[float] = None
    final_epe: Optional[float] = None
    iterations: int = 0
    elapsed_sec: float = 0.0
    converged: bool = False
    error_message: Optional[str] = None
    retries: int = 0
    worker_id: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    optimized_mask_path: Optional[str] = None

    @property
    def mse_improvement(self) -> Optional[float]:
        if self.initial_mse is None or self.final_mse is None:
            return None
        return self.initial_mse - self.final_mse

    @property
    def mse_improvement_ratio(self) -> Optional[float]:
        if self.initial_mse is None or self.final_mse is None:
            return None
        if abs(self.initial_mse) < 1e-12:
            return 0.0
        return (self.initial_mse - self.final_mse) / abs(self.initial_mse)

    def to_row(self) -> Dict[str, Any]:
        """转换为 CSV/JSON 可序列化的行"""
        return {
            'task_id': self.task_id,
            'cell_name': self.cell_name,
            'cell_display_name': self.cell_display_name,
            'status': self.status.value,
            'initial_mse': self.initial_mse,
            'final_mse': self.final_mse,
            'mse_improvement': self.mse_improvement,
            'mse_improvement_ratio': self.mse_improvement_ratio,
            'initial_ssim': self.initial_ssim,
            'final_ssim': self.final_ssim,
            'initial_epe_nm': self.initial_epe,
            'final_epe_nm': self.final_epe,
            'iterations': self.iterations,
            'elapsed_sec': round(self.elapsed_sec, 3),
            'converged': self.converged,
            'retries': self.retries,
            'worker_id': self.worker_id,
            'started_at': datetime.fromtimestamp(self.started_at).isoformat()
            if self.started_at else None,
            'finished_at': datetime.fromtimestamp(self.finished_at).isoformat()
            if self.finished_at else None,
            'error_message': self.error_message,
            'optimized_mask_path': self.optimized_mask_path,
        }


@dataclass
class BatchSummary:
    """
    整批任务的汇总统计

    Attributes:
        batch_id: 批次 ID
        total_tasks: 总任务数
        done / failed / cancelled / running / pending: 各状态计数
        success_rate: 成功率
        avg_initial_mse / avg_final_mse: 平均 MSE
        avg_mse_improvement_ratio: 平均 MSE 改善比例
        avg_elapsed_sec: 平均耗时
        total_elapsed_sec: 整批总耗时（墙钟）
        converged_count: 收敛任务数
        start_time / end_time: 起止时间戳
    """
    batch_id: str
    total_tasks: int = 0
    done: int = 0
    failed: int = 0
    cancelled: int = 0
    running: int = 0
    pending: int = 0
    timeout: int = 0
    success_rate: float = 0.0
    avg_initial_mse: Optional[float] = None
    avg_final_mse: Optional[float] = None
    avg_mse_improvement_ratio: Optional[float] = None
    avg_elapsed_sec: float = 0.0
    median_elapsed_sec: float = 0.0
    total_elapsed_sec: float = 0.0
    converged_count: int = 0
    converged_rate: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'batch_id': self.batch_id,
            'total_tasks': self.total_tasks,
            'done': self.done,
            'failed': self.failed,
            'cancelled': self.cancelled,
            'running': self.running,
            'pending': self.pending,
            'timeout': self.timeout,
            'success_rate': self.success_rate,
            'converged_count': self.converged_count,
            'converged_rate': self.converged_rate,
            'avg_initial_mse': self.avg_initial_mse,
            'avg_final_mse': self.avg_final_mse,
            'avg_mse_improvement_ratio': self.avg_mse_improvement_ratio,
            'avg_elapsed_sec': self.avg_elapsed_sec,
            'median_elapsed_sec': self.median_elapsed_sec,
            'total_elapsed_sec': self.total_elapsed_sec,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat()
            if self.start_time else None,
            'end_time': datetime.fromtimestamp(self.end_time).isoformat()
            if self.end_time else None,
            'config_snapshot': self.config_snapshot,
        }


@dataclass
class ResourceConfig:
    """
    资源限制配置

    Attributes:
        max_workers: 最大并发 worker 数，None=自动
        cpu_per_worker: 每个 worker 占用的 CPU 核数（仅用于限制/统计）
        gpu_ids: 可用 GPU ID 列表，空列表=不使用 GPU
        memory_limit_gb: 总内存限制（GB），None=不限制
        per_task_timeout_sec: 单任务超时（秒），0=不限
        auto_detect: 是否自动检测可用资源
    """
    max_workers: Optional[int] = None
    cpu_per_worker: int = 1
    gpu_ids: List[int] = field(default_factory=list)
    memory_limit_gb: Optional[float] = None
    per_task_timeout_sec: int = 0
    auto_detect: bool = True

    def resolve(self) -> 'ResourceConfig':
        """根据 auto_detect 计算实际参数，返回新实例"""
        cfg = ResourceConfig(
            max_workers=self.max_workers,
            cpu_per_worker=max(1, int(self.cpu_per_worker)),
            gpu_ids=list(self.gpu_ids),
            memory_limit_gb=self.memory_limit_gb,
            per_task_timeout_sec=self.per_task_timeout_sec,
            auto_detect=False,
        )

        if cfg.auto_detect or cfg.max_workers is None:
            if HAS_PSUTIL:
                total_cpu = psutil.cpu_count(logical=False) or os.cpu_count() or 1
            else:
                total_cpu = os.cpu_count() or 1

            if cfg.max_workers is None:
                cfg.max_workers = max(1, total_cpu // cfg.cpu_per_worker)
            else:
                cfg.max_workers = min(
                    int(cfg.max_workers),
                    max(1, total_cpu // cfg.cpu_per_worker),
                )

            if cfg.memory_limit_gb is None and HAS_PSUTIL:
                try:
                    mem = psutil.virtual_memory()
                    cfg.memory_limit_gb = round(mem.total / (1024 ** 3), 2)
                except Exception:
                    pass

        cfg.max_workers = max(1, int(cfg.max_workers or 1))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_workers': self.max_workers,
            'cpu_per_worker': self.cpu_per_worker,
            'gpu_ids': self.gpu_ids,
            'memory_limit_gb': self.memory_limit_gb,
            'per_task_timeout_sec': self.per_task_timeout_sec,
            'auto_detect': self.auto_detect,
        }


@dataclass
class BatchConfig:
    """
    批处理配置

    Attributes:
        optimizer_config: 传给 MaskOptimizer 的 OptimizationConfig 字典
        optical_system_config: OpticalSystem 参数字典（可选）
        use_multi_layer: 是否使用 MultiLayerMaskOptimizer（SMO）
        multi_layer_config: 多层优化参数字典
        max_retries: 失败重试次数（总次数，不含首次）
        retry_backoff_sec: 重试前退避秒数（按重试次数线性叠加）
        save_optimized_masks: 是否保存每个 cell 优化后的掩模
        output_dir: 输出目录（汇总表 + 掩模）
        output_formats: 输出格式列表，支持 'csv', 'json'
        progress_callback: 进度回调 (batch_id, cell_name, status, progress, result)
        interval_sec: 状态打印间隔秒（0=不打印）
        stop_on_first_failure: 是否遇到第一个失败就停止整批
    """
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    optical_system_config: Dict[str, Any] = field(default_factory=dict)
    use_multi_layer: bool = False
    multi_layer_config: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2
    retry_backoff_sec: float = 2.0
    save_optimized_masks: bool = False
    output_dir: Optional[str] = None
    output_formats: List[str] = field(default_factory=lambda: ['csv', 'json'])
    progress_callback: Optional[Callable[[
        str, str, TaskStatus, float, Optional[TaskResult]
    ], None]] = None
    interval_sec: float = 5.0
    stop_on_first_failure: bool = False
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None
    celery_queue_name: str = "litho_batch"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimizer_config': self.optimizer_config,
            'optical_system_config': self.optical_system_config,
            'use_multi_layer': self.use_multi_layer,
            'multi_layer_config': self.multi_layer_config,
            'max_retries': self.max_retries,
            'retry_backoff_sec': self.retry_backoff_sec,
            'save_optimized_masks': self.save_optimized_masks,
            'output_dir': self.output_dir,
            'output_formats': self.output_formats,
            'interval_sec': self.interval_sec,
            'stop_on_first_failure': self.stop_on_first_failure,
        }


# ============================================================================
# 单任务执行函数（进程池入口，需可 pickle）
# ============================================================================

def _execute_single_task(task_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    进程池/worker 中执行单个 cell 的优化任务

    该函数设计为纯函数（不依赖全局状态），
    所有参数通过 task_payload 传入，结果以字典返回，便于跨进程序列化。

    Payload 字段：
      - task_id, cell_name, cell_display_name
      - mask_bytes (np.save buffer), target_bytes
      - optimizer_config, optical_system_config
      - use_multi_layer, multi_layer_config
      - save_mask: bool, output_dir: str
      - worker_id: str
    """
    import io
    import hashlib

    task_id = task_payload['task_id']
    cell_name = task_payload['cell_name']
    display_name = task_payload.get('cell_display_name', cell_name)
    worker_id = task_payload.get('worker_id', '')

    started_at = time.time()

    def _fail(msg: str) -> Dict[str, Any]:
        return {
            'task_id': task_id,
            'cell_name': cell_name,
            'cell_display_name': display_name,
            'status': TaskStatus.FAILED.value,
            'error_message': msg,
            'started_at': started_at,
            'finished_at': time.time(),
            'worker_id': worker_id,
        }

    try:
        mask = np.load(io.BytesIO(task_payload['mask_bytes']))
        target = np.load(io.BytesIO(task_payload['target_bytes']))
    except Exception as e:
        return _fail(f"反序列化掩模数据失败: {e}")

    try:
        from algorithms.mask_optimizer import (
            MaskOptimizer, OptimizationConfig,
            MultiLayerMaskOptimizer, MultiLayerOptimizationConfig,
        )
        from core.imaging import OpticalSystem
        from core.metrics import evaluate_all
    except Exception as e:
        return _fail(f"导入优化模块失败，请检查 PYTHONPATH: {e}")

    try:
        os_cfg = task_payload.get('optical_system_config', {}) or {}
        optical_system = OpticalSystem(**{
            k: v for k, v in os_cfg.items()
            if k in OpticalSystem.__dataclass_fields__
        }) if os_cfg else OpticalSystem()
    except Exception as e:
        return _fail(f"构造 OpticalSystem 失败: {e}")

    try:
        opt_cfg_dict = task_payload.get('optimizer_config', {}) or {}
        opt_config = OptimizationConfig.from_dict(opt_cfg_dict)
    except Exception as e:
        return _fail(f"构造 OptimizationConfig 失败: {e}")

    use_ml = bool(task_payload.get('use_multi_layer', False))
    try:
        if use_ml:
            ml_cfg_dict = task_payload.get('multi_layer_config', {}) or {}
            ml_config = MultiLayerOptimizationConfig.from_dict(ml_cfg_dict)
            optimizer = MultiLayerMaskOptimizer(
                optical_system=optical_system,
                config=ml_config.wafer_mask_config or opt_config,
                source_config=ml_config.source_mask_config or opt_config,
            )
        else:
            optimizer = MaskOptimizer(
                optical_system=optical_system,
                config=opt_config,
            )
    except Exception as e:
        return _fail(f"构造优化器失败: {e}")

    try:
        if use_ml:
            result = optimizer.optimize(mask, target)
            final_mask = result.optimized_wafer_mask
            initial_metrics = result.initial_metrics
            final_metrics = result.final_metrics
            iterations = result.total_iterations
            converged = result.converged
        else:
            result = optimizer.optimize(mask, target)
            final_mask = result.optimized_mask
            initial_metrics = result.initial_metrics
            final_metrics = result.final_metrics
            iterations = result.total_iterations
            converged = result.converged
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return _fail(f"优化执行异常: {e}\n{tb}")

    finished_at = time.time()

    initial_mse = float(initial_metrics.mse) if hasattr(initial_metrics, 'mse') else None
    final_mse = float(final_metrics.mse) if hasattr(final_metrics, 'mse') else None
    initial_ssim = (float(initial_metrics.ssim)
                    if hasattr(initial_metrics, 'ssim') else None)
    final_ssim = (float(final_metrics.ssim)
                  if hasattr(final_metrics, 'ssim') else None)
    initial_epe = (float(initial_metrics.epe_mean)
                   if hasattr(initial_metrics, 'epe_mean') else None)
    final_epe = (float(final_metrics.epe_mean)
                 if hasattr(final_metrics, 'epe_mean') else None)

    mask_path = None
    if task_payload.get('save_mask') and task_payload.get('output_dir'):
        try:
            out_dir = Path(task_payload['output_dir']) / "masks"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(
                c if c.isalnum() or c in '-_.' else '_'
                for c in cell_name
            )
            mask_path = str(out_dir / f"{safe_name}__{task_id[:8]}.npy")
            np.save(mask_path, final_mask)
        except Exception as e:
            logger.warning(f"保存掩模失败 {cell_name}: {e}")
            mask_path = None

    extra = {}
    try:
        extra['loss_history_last5'] = (
            [float(x) for x in list(result.loss_history[-5:])]
            if hasattr(result, 'loss_history') and result.loss_history else []
        )
        if hasattr(result, 'message'):
            extra['message'] = str(result.message)
    except Exception:
        pass

    return {
        'task_id': task_id,
        'cell_name': cell_name,
        'cell_display_name': display_name,
        'status': TaskStatus.DONE.value,
        'initial_mse': initial_mse,
        'final_mse': final_mse,
        'initial_ssim': initial_ssim,
        'final_ssim': final_ssim,
        'initial_epe': initial_epe,
        'final_epe': final_epe,
        'iterations': int(iterations),
        'elapsed_sec': round(finished_at - started_at, 3),
        'converged': bool(converged),
        'worker_id': worker_id,
        'started_at': started_at,
        'finished_at': finished_at,
        'optimized_mask_path': mask_path,
        'extra': extra,
        'error_message': None,
    }


# ============================================================================
# 本地多进程 BatchRunner
# ============================================================================

class LocalBatchRunner:
    """
    本地多进程批处理调度器

    工作模式：
    - 从 LayoutQueue 持续 pop PENDING 任务
    - 提交到 ProcessPoolExecutor 执行
    - 监听 Future 完成，更新任务状态，写入结果
    - 失败按配置重试

    使用示例:
        runner = LocalBatchRunner(resource_cfg, batch_cfg)
        summary = runner.run(queue)
        save_batch_summary(summary, results, output_dir)
    """

    def __init__(self,
                 resource_config: Optional[ResourceConfig] = None,
                 batch_config: Optional[BatchConfig] = None):
        self.resource_config = (resource_config or ResourceConfig()).resolve()
        self.batch_config = batch_config or BatchConfig()
        self.batch_id: str = f"batch_{uuid.uuid4().hex[:12]}"
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._results: Dict[str, TaskResult] = {}
        self._active_futures: Dict[Future, str] = {}
        self._celery_app: Optional['Celery'] = None

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @property
    def results(self) -> List[TaskResult]:
        return list(self._results.values())

    def stop(self) -> None:
        """通知调度器停止接受新任务"""
        self._stop_event.set()

    def run(self,
            queue: LayoutQueue,
            output_dir: Optional[Union[str, Path]] = None) -> Tuple[BatchSummary, List[TaskResult]]:
        """
        启动批处理

        Args:
            queue: 版图任务队列（会被此运行器更新任务状态）
            output_dir: 覆盖 batch_config.output_dir

        Returns:
            (BatchSummary, 所有 TaskResult 列表)
        """
        if not HAS_CONCURRENT:
            raise RuntimeError("concurrent.futures 不可用，无法启动本地批处理")

        resolved_out = Path(output_dir) if output_dir else (
            Path(self.batch_config.output_dir) if self.batch_config.output_dir
            else Path.cwd() / "results" / self.batch_id
        )
        resolved_out.mkdir(parents=True, exist_ok=True)
        self.batch_config.output_dir = str(resolved_out)

        self._results.clear()
        self._active_futures.clear()
        self._stop_event.clear()

        start_time = time.time()
        summary = BatchSummary(
            batch_id=self.batch_id,
            total_tasks=len(queue),
            start_time=start_time,
            config_snapshot={
                'resource': self.resource_config.to_dict(),
                'batch': self.batch_config.to_dict(),
            },
        )

        logger.info(
            f"批次 {self.batch_id} 启动: {summary.total_tasks} 任务, "
            f"max_workers={self.resource_config.max_workers}, "
            f"输出目录={resolved_out}"
        )

        status_thread = None
        if self.batch_config.interval_sec > 0:
            stop_flag = threading.Event()
            status_thread = threading.Thread(
                target=self._status_printer_loop,
                args=(summary, queue, stop_flag),
                daemon=True,
            )
            status_thread.start()
        else:
            stop_flag = None

        def _on_progress(cell_name: str, status: TaskStatus,
                         progress: float, res: Optional[TaskResult]):
            cb = self.batch_config.progress_callback
            if cb is not None:
                try:
                    cb(self.batch_id, cell_name, status, progress, res)
                except Exception as e:
                    logger.debug(f"progress_callback 异常: {e}")

        try:
            with ProcessPoolExecutor(
                max_workers=self.resource_config.max_workers,
                initializer=_worker_initializer,
            ) as executor:
                worker_counter = 0
                # 主循环：不断入队新任务，等待结果
                while not self._stop_event.is_set():
                    with self._lock:
                        active_count = len(self._active_futures)

                    capacity = self.resource_config.max_workers - active_count

                    submitted_any = False
                    while capacity > 0:
                        entry = queue.pop_next(worker_id=f"local-{os.getpid()}")
                        if entry is None:
                            break

                        cell = entry.cell
                        if not cell.is_mask_loaded:
                            try:
                                cell.ensure_mask_loaded()
                            except Exception as e:
                                logger.error(f"加载 cell 掩模失败 {cell.name}: {e}")
                                queue.mark_failed(cell.name, f"mask load failed: {e}", retry=False)
                                self._register_failed(cell, f"mask load failed: {e}",
                                                      worker_id=f"local-{os.getpid()}")
                                _on_progress(cell.name, TaskStatus.FAILED, 0.0,
                                             self._results.get(cell.name))
                                continue

                        task_id = f"{self.batch_id}__{uuid.uuid4().hex[:8]}"
                        worker_counter += 1
                        worker_id = f"worker-{worker_counter % self.resource_config.max_workers + 1}"

                        payload = self._build_payload(task_id, cell, worker_id)
                        try:
                            fut = executor.submit(_execute_single_task, payload)
                        except Exception as e:
                            logger.error(f"提交任务失败 {cell.name}: {e}")
                            queue.mark_failed(cell.name, f"submit failed: {e}", retry=False)
                            self._register_failed(cell, f"submit failed: {e}", worker_id=worker_id)
                            _on_progress(cell.name, TaskStatus.FAILED, 0.0,
                                         self._results.get(cell.name))
                            continue

                        with self._lock:
                            self._active_futures[fut] = cell.name

                        self._register_running(cell, task_id, worker_id, entry.started_at or time.time())
                        _on_progress(cell.name, TaskStatus.RUNNING, 0.0,
                                     self._results.get(cell.name))
                        capacity -= 1
                        submitted_any = True

                    with self._lock:
                        if not self._active_futures:
                            if queue.all_done():
                                break
                            if not submitted_any:
                                time.sleep(0.05)
                                continue

                    done_futures = []
                    with self._lock:
                        items = list(self._active_futures.keys())
                    for fut in as_completed(items, timeout=0.5):
                        done_futures.append(fut)

                    for fut in done_futures:
                        with self._lock:
                            cell_name = self._active_futures.pop(fut, None)
                        if cell_name is None:
                            continue
                        self._handle_future_result(fut, cell_name, queue, _on_progress)

                    if self.batch_config.stop_on_first_failure:
                        if queue.failed_count() > 0:
                            logger.warning("检测到任务失败，按 stop_on_first_failure 停止整批")
                            self._cancel_remaining(queue)
                            break

        except KeyboardInterrupt:
            logger.warning("收到 Ctrl+C，取消剩余任务")
            self._cancel_remaining(queue)
        finally:
            if stop_flag is not None:
                stop_flag.set()
            if status_thread is not None:
                status_thread.join(timeout=2.0)

        summary.end_time = time.time()
        summary.total_elapsed_sec = round(summary.end_time - summary.start_time, 3)
        self._fill_summary_stats(summary, queue)

        # 写入汇总
        try:
            save_batch_summary(summary, self.results, resolved_out,
                               formats=self.batch_config.output_formats)
        except Exception as e:
            logger.error(f"写入汇总文件失败: {e}")

        logger.info(
            f"批次 {self.batch_id} 完成: 成功 {summary.done}/{summary.total_tasks}, "
            f"失败 {summary.failed}, 总耗时 {summary.total_elapsed_sec}s"
        )
        return summary, self.results

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _build_payload(self, task_id: str, cell: LayoutCell, worker_id: str) -> Dict[str, Any]:
        import io
        mask_buf = io.BytesIO()
        np.save(mask_buf, cell.mask)
        tgt = cell.target if cell.target is not None else cell.mask
        tgt_buf = io.BytesIO()
        np.save(tgt_buf, tgt)
        return {
            'task_id': task_id,
            'cell_name': cell.name,
            'cell_display_name': cell.cell_name,
            'mask_bytes': mask_buf.getvalue(),
            'target_bytes': tgt_buf.getvalue(),
            'optimizer_config': self.batch_config.optimizer_config,
            'optical_system_config': self.batch_config.optical_system_config,
            'use_multi_layer': self.batch_config.use_multi_layer,
            'multi_layer_config': self.batch_config.multi_layer_config,
            'save_mask': self.batch_config.save_optimized_masks,
            'output_dir': self.batch_config.output_dir,
            'worker_id': worker_id,
        }

    def _register_running(self, cell: LayoutCell, task_id: str,
                          worker_id: str, started_at: float) -> None:
        with self._lock:
            if cell.name not in self._results:
                self._results[cell.name] = TaskResult(
                    task_id=task_id,
                    cell_name=cell.name,
                    cell_display_name=cell.cell_name,
                )
            r = self._results[cell.name]
            r.task_id = task_id
            r.status = TaskStatus.RUNNING
            r.worker_id = worker_id
            r.started_at = started_at

    def _register_failed(self, cell: LayoutCell, err: str, worker_id: str) -> None:
        with self._lock:
            if cell.name not in self._results:
                self._results[cell.name] = TaskResult(
                    task_id=f"{self.batch_id}__err_{uuid.uuid4().hex[:6]}",
                    cell_name=cell.name,
                    cell_display_name=cell.cell_name,
                )
            r = self._results[cell.name]
            r.status = TaskStatus.FAILED
            r.error_message = err
            r.worker_id = worker_id
            r.finished_at = time.time()
            if r.started_at is None:
                r.started_at = r.finished_at
            r.elapsed_sec = round(r.finished_at - r.started_at, 3)

    def _handle_future_result(self, fut: Future, cell_name: str,
                              queue: LayoutQueue, progress_cb) -> None:
        entry = queue.get_entry(cell_name)
        cell = entry.cell if entry is not None else None
        try:
            timeout = self.resource_config.per_task_timeout_sec
            timeout = timeout if timeout > 0 else None
            raw = fut.result(timeout=timeout)
        except Exception as e:
            tb = traceback.format_exc(limit=5)
            err = f"worker exception: {e}\n{tb}"
            logger.error(f"任务异常 {cell_name}: {err}")
            with self._lock:
                r = self._results.get(cell_name)
                if r is not None:
                    r.status = TaskStatus.FAILED
                    r.error_message = err
                    r.finished_at = time.time()
                    if r.started_at:
                        r.elapsed_sec = round(r.finished_at - r.started_at, 3)
            if entry is not None:
                queue.mark_failed(cell_name, err, retry=True)
                if entry.status == LayoutQueue.Status.PENDING:
                    if r is not None:
                        r.status = TaskStatus.PENDING
                        r.retries = entry.retries
            progress_cb(cell_name, TaskStatus.FAILED if (entry is None or entry.status == LayoutQueue.Status.FAILED)
                        else TaskStatus.PENDING, 0.0, self._results.get(cell_name))
            return

        status_str = raw.get('status', TaskStatus.FAILED.value)
        status_enum = TaskStatus(status_str)

        with self._lock:
            r = self._results.get(cell_name)
            if r is None:
                r = TaskResult(
                    task_id=raw.get('task_id', ''),
                    cell_name=cell_name,
                    cell_display_name=raw.get('cell_display_name', ''),
                )
                self._results[cell_name] = r

            r.status = status_enum
            r.initial_mse = raw.get('initial_mse')
            r.final_mse = raw.get('final_mse')
            r.initial_ssim = raw.get('initial_ssim')
            r.final_ssim = raw.get('final_ssim')
            r.initial_epe = raw.get('initial_epe')
            r.final_epe = raw.get('final_epe')
            r.iterations = int(raw.get('iterations', 0))
            r.elapsed_sec = float(raw.get('elapsed_sec', 0.0))
            r.converged = bool(raw.get('converged', False))
            r.worker_id = raw.get('worker_id', r.worker_id)
            r.started_at = raw.get('started_at', r.started_at)
            r.finished_at = raw.get('finished_at', time.time())
            r.optimized_mask_path = raw.get('optimized_mask_path')
            r.error_message = raw.get('error_message')
            if entry is not None:
                r.retries = entry.retries
            if raw.get('extra'):
                r.extra.update(raw['extra'])

        if status_enum == TaskStatus.DONE:
            queue.mark_done(cell_name, progress=1.0)
            progress_cb(cell_name, TaskStatus.DONE, 1.0, r)
        else:
            err = raw.get('error_message') or 'unknown error'
            queue.mark_failed(cell_name, err, retry=True)
            new_entry = queue.get_entry(cell_name)
            new_status = TaskStatus.PENDING if (new_entry is not None and
                                                new_entry.status == LayoutQueue.Status.PENDING) \
                else TaskStatus.FAILED
            if new_status == TaskStatus.PENDING:
                r.status = TaskStatus.PENDING
                r.retries = new_entry.retries if new_entry else r.retries
            progress_cb(cell_name, new_status, 0.0, r)

    def _cancel_remaining(self, queue: LayoutQueue) -> None:
        for name in list(queue._order):
            e = queue.get_entry(name)
            if e is not None and e.status == LayoutQueue.Status.PENDING:
                queue.mark_cancelled(name)
                with self._lock:
                    r = self._results.get(name)
                    if r is None:
                        self._results[name] = TaskResult(
                            task_id=f"{self.batch_id}__cancel_{uuid.uuid4().hex[:6]}",
                            cell_name=name,
                            cell_display_name=e.cell.cell_name,
                        )
                    r = self._results[name]
                    r.status = TaskStatus.CANCELLED
                    r.finished_at = time.time()

    def _status_printer_loop(self, summary: BatchSummary, queue: LayoutQueue,
                             stop_flag: threading.Event) -> None:
        interval = max(0.5, float(self.batch_config.interval_sec))
        while not stop_flag.is_set():
            counts = queue.status_counts()
            elapsed = round(time.time() - summary.start_time, 1)
            total = summary.total_tasks or 1
            done = counts.get('done', 0)
            pct = round(done / total * 100, 1)
            logger.info(
                f"[批次 {self.batch_id[:10]}] t={elapsed}s  "
                f"{done}/{total} ({pct}%)  "
                f"pend={counts.get('pending', 0)}  "
                f"run={counts.get('running', 0)}  "
                f"done={done}  "
                f"fail={counts.get('failed', 0)}  "
                f"cxl={counts.get('cancelled', 0)}"
            )
            stop_flag.wait(interval)

    def _fill_summary_stats(self, summary: BatchSummary, queue: LayoutQueue) -> None:
        counts = queue.status_counts()
        summary.done = counts.get('done', 0)
        summary.failed = counts.get('failed', 0)
        summary.cancelled = counts.get('cancelled', 0)
        summary.running = counts.get('running', 0)
        summary.pending = counts.get('pending', 0)
        summary.timeout = counts.get('timeout', 0)

        successful = [r for r in self._results.values() if r.status == TaskStatus.DONE]
        if summary.total_tasks > 0:
            summary.success_rate = round(len(successful) / summary.total_tasks, 4)

        if successful:
            imses = [r.initial_mse for r in successful if r.initial_mse is not None]
            fmsses = [r.final_mse for r in successful if r.final_mse is not None]
            ratios = [r.mse_improvement_ratio for r in successful
                      if r.mse_improvement_ratio is not None]
            elaps = [r.elapsed_sec for r in successful]
            summary.avg_initial_mse = float(np.mean(imses)) if imses else None
            summary.avg_final_mse = float(np.mean(fmsses)) if fmsses else None
            summary.avg_mse_improvement_ratio = (
                float(np.mean(ratios)) if ratios else None
            )
            summary.avg_elapsed_sec = round(float(np.mean(elaps)), 3) if elaps else 0.0
            summary.median_elapsed_sec = round(float(np.median(elaps)), 3) if elaps else 0.0
            summary.converged_count = sum(1 for r in successful if r.converged)
            summary.converged_rate = (
                round(summary.converged_count / len(successful), 4) if successful else 0.0
            )


# ============================================================================
# 可选 Celery/Redis 分布式调度器
# ============================================================================

class DistributedBatchRunner:
    """
    Celery/Redis 分布式批处理调度器（可选）

    当 HAS_CELERY=False 时，构造即抛出 ImportError。
    建议 Docker Compose 中使用：
      redis 服务 + 多个 celery worker 服务（通过 --concurrency 控制）。

    典型启动:
      celery -A pipeline.batch_runner.celery_app worker \
             -Q litho_batch --concurrency=8 --loglevel=INFO
    """

    def __init__(self,
                 broker_url: str,
                 result_backend: str,
                 queue_name: str = "litho_batch",
                 batch_config: Optional[BatchConfig] = None):
        if not HAS_CELERY:
            raise ImportError(
                "celery 未安装。请先 `pip install celery redis` "
                "或使用 LocalBatchRunner。"
            )
        self.queue_name = queue_name
        self.batch_config = batch_config or BatchConfig()
        self.batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        self._app = Celery(
            "litho_batch",
            broker=broker_url,
            backend=result_backend,
        )
        self._app.conf.update(
            task_serializer='pickle',
            accept_content=['pickle', 'json'],
            result_serializer='pickle',
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            task_default_queue=queue_name,
            task_track_started=True,
        )
        self._results: Dict[str, TaskResult] = {}
        self._submitted: Dict[str, str] = {}  # celery_id -> cell_name

    @property
    def app(self) -> 'Celery':
        return self._app

    def run(self,
            queue: LayoutQueue,
            output_dir: Optional[Union[str, Path]] = None,
            poll_interval: float = 2.0,
            ) -> Tuple[BatchSummary, List[TaskResult]]:
        resolved_out = Path(output_dir) if output_dir else (
            Path(self.batch_config.output_dir) if self.batch_config.output_dir
            else Path.cwd() / "results" / self.batch_id
        )
        resolved_out.mkdir(parents=True, exist_ok=True)
        self.batch_config.output_dir = str(resolved_out)

        start_time = time.time()
        summary = BatchSummary(
            batch_id=self.batch_id,
            total_tasks=len(queue),
            start_time=start_time,
            config_snapshot={'batch': self.batch_config.to_dict()},
        )

        self._results.clear()
        self._submitted.clear()

        celery_task = self._app.task(
            name=f"{self.queue_name}.execute",
            bind=False,
            queue=self.queue_name,
        )(_execute_single_task)

        # 1) 提交所有 PENDING 任务
        submitted_count = 0
        while True:
            entry = queue.pop_next(worker_id="celery")
            if entry is None:
                break
            cell = entry.cell
            if not cell.is_mask_loaded:
                try:
                    cell.ensure_mask_loaded()
                except Exception as e:
                    queue.mark_failed(cell.name, f"mask load failed: {e}", retry=False)
                    self._register_failed(cell, f"mask load failed: {e}")
                    continue

            task_id = f"{self.batch_id}__{uuid.uuid4().hex[:8]}"
            payload = self._build_payload(task_id, cell)
            async_res = celery_task.apply_async(
                args=[payload],
                queue=self.queue_name,
                task_id=task_id,
            )
            self._submitted[async_res.id] = cell.name
            started_at = entry.started_at or time.time()
            self._register_running(cell, task_id, "celery", started_at)
            submitted_count += 1

        logger.info(f"已向 Celery 提交 {submitted_count} 任务")

        # 2) 轮询结果
        while self._submitted:
            done_ids = []
            for cid, cell_name in list(self._submitted.items()):
                try:
                    res = AsyncResult(cid, app=self._app)
                    if res.ready():
                        try:
                            raw = res.get(timeout=1.0)
                        except Exception as e:
                            raw = {
                                'task_id': cid,
                                'cell_name': cell_name,
                                'cell_display_name': cell_name,
                                'status': TaskStatus.FAILED.value,
                                'error_message': f"celery exception: {e}",
                            }
                        self._apply_raw_result(cell_name, raw, queue)
                        done_ids.append(cid)
                except Exception as e:
                    logger.debug(f"轮询 {cid} 异常: {e}")
            for cid in done_ids:
                self._submitted.pop(cid, None)

            counts = queue.status_counts()
            elapsed = round(time.time() - summary.start_time, 1)
            total = summary.total_tasks or 1
            done = counts.get('done', 0)
            logger.info(
                f"[dist批次 {self.batch_id[:10]}] t={elapsed}s  "
                f"{done}/{total}  pend={counts.get('pending', 0)}  "
                f"run={counts.get('running', 0)}  done={done}  "
                f"fail={counts.get('failed', 0)}  remain={len(self._submitted)}"
            )

            if not self._submitted:
                break
            time.sleep(poll_interval)

        summary.end_time = time.time()
        summary.total_elapsed_sec = round(summary.end_time - summary.start_time, 3)

        self._fill_summary(summary, queue)
        try:
            save_batch_summary(summary, list(self._results.values()),
                               resolved_out, formats=self.batch_config.output_formats)
        except Exception as e:
            logger.error(f"写入汇总失败: {e}")

        return summary, list(self._results.values())

    def _build_payload(self, task_id, cell, worker_id="celery") -> Dict[str, Any]:
        import io
        mask_buf = io.BytesIO()
        np.save(mask_buf, cell.mask)
        tgt = cell.target if cell.target is not None else cell.mask
        tgt_buf = io.BytesIO()
        np.save(tgt_buf, tgt)
        return {
            'task_id': task_id,
            'cell_name': cell.name,
            'cell_display_name': cell.cell_name,
            'mask_bytes': mask_buf.getvalue(),
            'target_bytes': tgt_buf.getvalue(),
            'optimizer_config': self.batch_config.optimizer_config,
            'optical_system_config': self.batch_config.optical_system_config,
            'use_multi_layer': self.batch_config.use_multi_layer,
            'multi_layer_config': self.batch_config.multi_layer_config,
            'save_mask': self.batch_config.save_optimized_masks,
            'output_dir': self.batch_config.output_dir,
            'worker_id': worker_id,
        }

    def _register_running(self, cell, task_id, worker_id, started_at):
        if cell.name not in self._results:
            self._results[cell.name] = TaskResult(
                task_id=task_id, cell_name=cell.name,
                cell_display_name=cell.cell_name,
            )
        r = self._results[cell.name]
        r.status = TaskStatus.RUNNING
        r.worker_id = worker_id
        r.started_at = started_at

    def _register_failed(self, cell, err):
        if cell.name not in self._results:
            self._results[cell.name] = TaskResult(
                task_id=f"{self.batch_id}__err", cell_name=cell.name,
                cell_display_name=cell.cell_name,
            )
        r = self._results[cell.name]
        r.status = TaskStatus.FAILED
        r.error_message = err
        r.finished_at = time.time()

    def _apply_raw_result(self, cell_name, raw, queue):
        r = self._results.get(cell_name)
        if r is None:
            r = TaskResult(task_id=raw.get('task_id', ''), cell_name=cell_name,
                           cell_display_name=raw.get('cell_display_name', ''))
            self._results[cell_name] = r

        status = TaskStatus(raw.get('status', TaskStatus.FAILED.value))
        r.status = status
        r.initial_mse = raw.get('initial_mse')
        r.final_mse = raw.get('final_mse')
        r.initial_ssim = raw.get('initial_ssim')
        r.final_ssim = raw.get('final_ssim')
        r.initial_epe = raw.get('initial_epe')
        r.final_epe = raw.get('final_epe')
        r.iterations = int(raw.get('iterations', 0))
        r.elapsed_sec = float(raw.get('elapsed_sec', 0.0))
        r.converged = bool(raw.get('converged', False))
        r.finished_at = raw.get('finished_at', time.time())
        r.optimized_mask_path = raw.get('optimized_mask_path')
        r.error_message = raw.get('error_message')

        if status == TaskStatus.DONE:
            queue.mark_done(cell_name, 1.0)
        else:
            queue.mark_failed(cell_name, r.error_message or 'celery failed', retry=False)

    def _fill_summary(self, summary, queue):
        counts = queue.status_counts()
        summary.done = counts.get('done', 0)
        summary.failed = counts.get('failed', 0)
        summary.cancelled = counts.get('cancelled', 0)
        summary.running = counts.get('running', 0)
        summary.pending = counts.get('pending', 0)
        ok = [r for r in self._results.values() if r.status == TaskStatus.DONE]
        if summary.total_tasks:
            summary.success_rate = round(len(ok) / summary.total_tasks, 4)
        if ok:
            summary.avg_initial_mse = float(np.mean(
                [r.initial_mse for r in ok if r.initial_mse is not None])) or None
            summary.avg_final_mse = float(np.mean(
                [r.final_mse for r in ok if r.final_mse is not None])) or None
            ratios = [r.mse_improvement_ratio for r in ok
                      if r.mse_improvement_ratio is not None]
            summary.avg_mse_improvement_ratio = float(np.mean(ratios)) if ratios else None
            elaps = [r.elapsed_sec for r in ok]
            summary.avg_elapsed_sec = round(float(np.mean(elaps)), 3)
            summary.median_elapsed_sec = round(float(np.median(elaps)), 3)
            summary.converged_count = sum(1 for r in ok if r.converged)
            summary.converged_rate = round(summary.converged_count / len(ok), 4) if ok else 0.0


# ============================================================================
# Worker 初始化器（进程池）
# ============================================================================

def _worker_initializer():
    """子进程初始化：忽略 Ctrl+C（由父进程统一处理），设置随机种子"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import random
        random.seed()
        np.random.seed()
    except Exception:
        pass


# ============================================================================
# 汇总结果持久化
# ============================================================================

def save_batch_summary(summary: BatchSummary,
                       results: Iterable[TaskResult],
                       output_dir: Union[str, Path],
                       formats: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    保存批处理汇总到文件

    Args:
        summary: 批次统计汇总
        results: TaskResult 列表
        output_dir: 输出目录
        formats: ['csv', 'json']，默认两者都写

    Returns:
        格式 -> 文件路径 的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ['csv', 'json']
    written: Dict[str, Path] = {}

    rows = [r.to_row() for r in results]

    if 'csv' in formats:
        csv_path = output_dir / f"batch_{summary.batch_id}_results.csv"
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        else:
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write('')
        written['csv'] = csv_path

    if 'json' in formats:
        json_path = output_dir / f"batch_{summary.batch_id}_results.json"
        payload = {
            'summary': summary.to_dict(),
            'results': rows,
            'generated_at': datetime.now().isoformat(),
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        written['json'] = json_path

    summary_json = output_dir / f"batch_{summary.batch_id}_summary.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    written['summary_json'] = summary_json

    logger.info(f"已写入批处理汇总: {list(written.values())}")
    return written


# ============================================================================
# 顶层便捷函数
# ============================================================================

def run_batch_optimization(
    source: Union[str, Path, List[Union[str, Path]], LayoutQueue, LayoutLibrary],
    layer: Optional[int] = None,
    layout_options: Optional[Dict[str, Any]] = None,
    resource_config: Optional[ResourceConfig] = None,
    batch_config: Optional[BatchConfig] = None,
    mode: str = "local",
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[BatchSummary, List[TaskResult], Optional[LayoutLibrary], LayoutQueue]:
    """
    一键完成"加载→建队→优化→汇总"流程

    Args:
        source: 目录 / GDS 文件 / 路径列表 / LayoutLibrary / LayoutQueue
        layer: GDS 层号（仅当 source 为路径时需要）
        layout_options: LayoutLoadOptions 参数字典（source 为路径时）
        resource_config: 资源配置
        batch_config: 批处理配置
        mode: 'local'（本地多进程）或 'distributed'（Celery）
        output_dir: 输出目录

    Returns:
        (BatchSummary, TaskResult 列表, LayoutLibrary（可为None）, LayoutQueue)
    """
    resource_config = resource_config or ResourceConfig()
    batch_config = batch_config or BatchConfig()

    queue: Optional[LayoutQueue] = None
    lib: Optional[LayoutLibrary] = None

    if isinstance(source, LayoutQueue):
        queue = source
    elif isinstance(source, LayoutLibrary):
        lib = source
        mgr = LayoutManager()
        queue = mgr.build_queue(lib)
    else:
        if layer is None:
            raise ValueError("source 为路径时必须指定 layer")
        mgr = LayoutManager()
        opt = LayoutLoadOptions(**(layout_options or {}), layer=layer)
        lib, queue = mgr.load_and_queue(source, layer=layer, **(layout_options or {}))

    if mode == "local":
        runner = LocalBatchRunner(resource_config, batch_config)
        summary, results = runner.run(queue, output_dir=output_dir)
        return summary, results, lib, queue
    elif mode == "distributed":
        if not (batch_config.celery_broker_url and batch_config.celery_result_backend):
            raise ValueError(
                "distributed 模式需设置 batch_config.celery_broker_url "
                "和 celery_result_backend"
            )
        runner = DistributedBatchRunner(
            broker_url=batch_config.celery_broker_url,
            result_backend=batch_config.celery_result_backend,
            queue_name=batch_config.celery_queue_name,
            batch_config=batch_config,
        )
        summary, results = runner.run(queue, output_dir=output_dir)
        return summary, results, lib, queue
    else:
        raise ValueError(f"未知 mode: {mode}，可选 'local' / 'distributed'")


# ============================================================================
# 模块级 Celery App（供 `celery -A pipeline.batch_runner.celery_app worker` 使用）
# ============================================================================

def _build_module_celery_app() -> Optional['Celery']:
    """
    基于环境变量构造 Celery 应用（单例），并注册 execute 任务。

    环境变量:
      CELERY_BROKER_URL  (默认 redis://localhost:6379/0)
      CELERY_RESULT_BACKEND (默认 redis://localhost:6379/1)
      CELERY_QUEUE (默认 litho_batch)
    """
    if not HAS_CELERY:
        return None
    broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    queue = os.environ.get("CELERY_QUEUE", "litho_batch")
    try:
        app = Celery("litho_batch", broker=broker, backend=backend)
        app.conf.update(
            task_serializer='pickle',
            accept_content=['pickle', 'json'],
            result_serializer='pickle',
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            task_default_queue=queue,
            task_track_started=True,
            task_routes={
                'pipeline.batch_runner.execute_task': {'queue': queue},
            },
        )
        _task = app.task(
            name="pipeline.batch_runner.execute_task",
            bind=False,
            queue=queue,
        )(_execute_single_task)
        globals().setdefault('_celery_module_app', app)
        return app
    except Exception as e:
        logger.debug(f"构造模块级 Celery App 失败（worker 模式才需要）: {e}")
        return None


celery_app: Optional['Celery'] = _build_module_celery_app()
"""模块级 Celery 应用，供 celery worker 命令行启动。"""
