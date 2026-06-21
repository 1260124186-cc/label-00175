# -*- coding: utf-8 -*-
"""
Fab 闭环反馈校准：定期调度器

支持定期（按时间间隔或 Cron 表达式）自动执行闭环周期。
- 可配置执行间隔（小时）
- 支持后台线程/多进程运行
- 支持暂停、恢复、立即执行
- 运行状态跟踪与历史记录
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .pipeline import ClosedLoopPipeline
from .schemas import ClosedLoopCycle, ClosedLoopState, ClosedLoopConfig

logger = logging.getLogger(__name__)


class SchedulerState(Enum):
    """调度器状态"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTDOWN = "shutdown"


@dataclass
class SchedulerRunRecord:
    """单次调度运行记录"""
    cycle_id: str
    start_time: str
    end_time: str
    state: str
    duration_sec: float
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cycle_id': self.cycle_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'state': self.state,
            'duration_sec': self.duration_sec,
            'error_message': self.error_message,
        }


@dataclass
class SchedulerConfig:
    """
    调度器配置

    Attributes:
        interval_hours: 执行间隔（小时）
        run_on_start: 启动时是否立即执行一次
        max_cycles: 最多执行多少个周期（0 表示无限）
        stop_on_error: 遇到错误是否停止调度
        cooldown_seconds: 两次执行间的最短冷却秒数（防止间隔配置过小）
        history_file: 调度历史记录 JSON 路径
    """
    interval_hours: float = 24.0
    run_on_start: bool = False
    max_cycles: int = 0
    stop_on_error: bool = False
    cooldown_seconds: float = 60.0
    history_file: str = "./closed_loop/scheduler_history.json"


class ClosedLoopScheduler:
    """
    Fab 闭环反馈校准定期调度器

    典型用法::

        scheduler = ClosedLoopScheduler(pipeline, config)
        scheduler.start()  # 后台线程运行
        # ...
        scheduler.trigger_now()  # 立即执行一次
        scheduler.pause()
        scheduler.resume()
        scheduler.stop()
    """

    def __init__(self,
                 pipeline: ClosedLoopPipeline,
                 config: Optional[SchedulerConfig] = None,
                 on_cycle_complete: Optional[
                     Callable[[ClosedLoopCycle], None]
                 ] = None,
                 on_cycle_error: Optional[
                     Callable[[ClosedLoopCycle, Exception], None]
                 ] = None,
                 ):
        """
        Args:
            pipeline: 闭环流水线实例
            config: 调度器配置；None 则使用默认
            on_cycle_complete: 周期完成回调
            on_cycle_error: 周期错误回调
        """
        self.pipeline = pipeline
        self.config = config or SchedulerConfig()
        self.on_cycle_complete = on_cycle_complete
        self.on_cycle_error = on_cycle_error

        self._state = SchedulerState.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._trigger_event = threading.Event()
        self._lock = threading.Lock()

        self._run_history: List[SchedulerRunRecord] = []
        self._cycles_completed: int = 0
        self._last_run_time: Optional[datetime] = None
        self._next_run_time: Optional[datetime] = None

        self._load_history()

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    @property
    def state(self) -> SchedulerState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == SchedulerState.RUNNING

    @property
    def is_paused(self) -> bool:
        return self._state == SchedulerState.PAUSED

    @property
    def cycles_completed(self) -> int:
        return self._cycles_completed

    @property
    def last_run_time(self) -> Optional[datetime]:
        return self._last_run_time

    @property
    def next_run_time(self) -> Optional[datetime]:
        return self._next_run_time

    # ------------------------------------------------------------------
    # 历史记录
    # ------------------------------------------------------------------
    def _load_history(self) -> None:
        path = Path(self.config.history_file)
        if path.exists():
            try:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                self._run_history = [
                    SchedulerRunRecord(**r) for r in raw
                ]
                logger.info(
                    f"已加载调度历史: {len(self._run_history)} 条记录"
                )
            except Exception as e:
                logger.warning(f"调度历史加载失败，将重建: {e}")
                self._run_history = []

    def _save_history(self) -> None:
        path = Path(self.config.history_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(
                    [r.to_dict() for r in self._run_history],
                    f, ensure_ascii=False, indent=2,
                )
        except Exception as e:
            logger.warning(f"调度历史保存失败: {e}")

    def get_run_history(self,
                        limit: Optional[int] = None,
                        ) -> List[SchedulerRunRecord]:
        """获取运行历史"""
        if limit is not None:
            return self._run_history[-limit:]
        return list(self._run_history)

    # ------------------------------------------------------------------
    # 下一次执行时间计算
    # ------------------------------------------------------------------
    def _compute_next_run(self) -> datetime:
        """计算下一次执行时间"""
        interval = timedelta(hours=max(self.config.interval_hours,
                                       self.config.cooldown_seconds / 3600))
        if self._last_run_time is None:
            return datetime.now() + timedelta(seconds=1)
        return self._last_run_time + interval

    # ------------------------------------------------------------------
    # 单次执行
    # ------------------------------------------------------------------
    def _run_single_cycle(self) -> ClosedLoopCycle:
        """执行单个闭环周期，返回结果"""
        t0 = time.time()
        start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cycle: Optional[ClosedLoopCycle] = None
        error_msg = ""

        try:
            cycle = self.pipeline.run_cycle()

            if self.on_cycle_complete is not None:
                try:
                    self.on_cycle_complete(cycle)
                except Exception as cb_e:
                    logger.warning(f"周期完成回调异常: {cb_e}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"闭环周期执行异常: {e}", exc_info=True)
            if self.on_cycle_error is not None and cycle is not None:
                try:
                    self.on_cycle_error(cycle, e)
                except Exception as cb_e:
                    logger.warning(f"周期错误回调异常: {cb_e}")

        end_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration = time.time() - t0

        state = cycle.state.value if cycle else ClosedLoopState.FAILED.value
        record = SchedulerRunRecord(
            cycle_id=cycle.cycle_id if cycle else f"failed_{start_ts}",
            start_time=start_ts,
            end_time=end_ts,
            state=state,
            duration_sec=duration,
            error_message=error_msg,
        )
        self._run_history.append(record)
        self._save_history()

        self._last_run_time = datetime.now()
        self._cycles_completed += 1

        if cycle is None:
            cycle = ClosedLoopCycle(
                cycle_id=record.cycle_id,
                start_time=start_ts,
                end_time=end_ts,
                state=ClosedLoopState.FAILED,
                error_message=error_msg,
            )
        return cycle

    # ------------------------------------------------------------------
    # 后台线程主循环
    # ------------------------------------------------------------------
    def _scheduler_loop(self) -> None:
        """后台线程主循环"""
        logger.info("调度器后台线程启动")
        self._state = SchedulerState.RUNNING

        if self.config.run_on_start:
            logger.info("启动时立即执行第一次周期")
            self._trigger_event.set()

        while not self._stop_event.is_set():
            try:
                if self._pause_event.is_set():
                    self._state = SchedulerState.PAUSED
                    self._next_run_time = None
                    if self._pause_event.wait(timeout=5.0):
                        self._pause_event.clear()
                        self._state = SchedulerState.RUNNING
                        logger.info("调度器已恢复")
                    continue

                self._state = SchedulerState.RUNNING
                self._next_run_time = self._compute_next_run()
                now = datetime.now()
                wait_seconds = max(
                    (self._next_run_time - now).total_seconds(),
                    1.0,
                )

                triggered = self._trigger_event.wait(timeout=wait_seconds)
                if self._stop_event.is_set():
                    break

                if triggered or datetime.now() >= self._next_run_time:
                    self._trigger_event.clear()
                    logger.info(
                        f"触发执行闭环周期 (手动={triggered})"
                    )
                    cycle = self._run_single_cycle()

                    if (self.config.stop_on_error
                            and cycle.state == ClosedLoopState.FAILED):
                        logger.error(
                            "检测到错误，根据配置停止调度器"
                        )
                        break

                    if (self.config.max_cycles > 0
                            and self._cycles_completed
                            >= self.config.max_cycles):
                        logger.info(
                            f"已达到最大周期数 {self.config.max_cycles}，停止调度器"
                        )
                        break

            except Exception as e:
                logger.error(f"调度器主循环异常: {e}", exc_info=True)
                if self.config.stop_on_error:
                    break
                time.sleep(5.0)

        self._state = SchedulerState.STOPPED
        logger.info("调度器后台线程已停止")

    # ------------------------------------------------------------------
    # 控制接口
    # ------------------------------------------------------------------
    def start(self, daemon: bool = True) -> None:
        """
        启动后台调度器线程

        Args:
            daemon: 是否以守护线程运行
        """
        if self._state in (SchedulerState.RUNNING, SchedulerState.PAUSED):
            logger.warning("调度器已在运行")
            return

        self._stop_event.clear()
        self._pause_event.clear()
        self._trigger_event.clear()

        self._thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=daemon,
            name="ClosedLoopScheduler",
        )
        self._thread.start()
        logger.info(
            f"调度器已启动 (interval={self.config.interval_hours}h, "
            f"run_on_start={self.config.run_on_start})"
        )

    def stop(self, timeout: float = 30.0) -> None:
        """
        停止调度器

        Args:
            timeout: 等待线程结束的超时时间（秒）
        """
        if self._state == SchedulerState.STOPPED:
            return

        logger.info("正在停止调度器...")
        self._stop_event.set()
        self._trigger_event.set()
        self._pause_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("调度器线程未在超时时间内停止")

        self._state = SchedulerState.STOPPED
        self._thread = None
        logger.info("调度器已停止")

    def pause(self) -> None:
        """暂停调度（不影响正在执行的周期）"""
        if self._state != SchedulerState.RUNNING:
            logger.warning(f"调度器当前状态 {self._state.value}，无法暂停")
            return
        self._pause_event.set()
        logger.info("调度器已暂停")

    def resume(self) -> None:
        """恢复调度"""
        if self._state != SchedulerState.PAUSED:
            logger.warning(f"调度器当前状态 {self._state.value}，无法恢复")
            return
        self._pause_event.clear()
        self._trigger_event.set()
        logger.info("调度器已恢复")

    def trigger_now(self, wait: bool = False,
                    timeout: float = 600.0) -> Optional[ClosedLoopCycle]:
        """
        立即触发一次执行

        Args:
            wait: 是否同步等待执行完成
            timeout: wait=True 时的等待超时（秒）

        Returns:
            wait=True 时返回 ClosedLoopCycle，否则 None
        """
        if self._state == SchedulerState.STOPPED:
            logger.info("调度器未启动，直接同步执行")
            return self._run_single_cycle()

        if self.is_paused:
            self.resume()

        self._trigger_event.set()
        logger.info("已触发立即执行")

        if wait:
            start_count = self._cycles_completed
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self._cycles_completed > start_count:
                    return self.pipeline.last_cycle
                time.sleep(1.0)
            logger.warning("等待执行超时")
        return None

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------
    def __enter__(self) -> 'ClosedLoopScheduler':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def create_scheduler(
    pipeline: ClosedLoopPipeline,
    interval_hours: float = 24.0,
    run_on_start: bool = False,
    **kwargs,
) -> ClosedLoopScheduler:
    """
    便捷函数：创建调度器

    Args:
        pipeline: 闭环流水线
        interval_hours: 执行间隔小时数
        run_on_start: 启动时是否立即执行
        **kwargs: 其他 SchedulerConfig 参数

    Returns:
        ClosedLoopScheduler
    """
    config = SchedulerConfig(
        interval_hours=interval_hours,
        run_on_start=run_on_start,
        **{k: v for k, v in kwargs.items()
           if k in SchedulerConfig.__dataclass_fields__},
    )
    return ClosedLoopScheduler(pipeline, config)
