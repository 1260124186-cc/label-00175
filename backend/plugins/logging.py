# -*- coding: utf-8 -*-
"""
插件日志适配

提供与框架统一风格的日志接口，自动注入插件上下文信息。
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)


def get_plugin_logger(
    plugin_type: str,
    plugin_name: str,
    *,
    parent: Optional[str] = None,
) -> logging.Logger:
    """获取插件专属 logger

    命名规范：litho_sim.plugins.<plugin_type>.<plugin_name>
    """
    base = parent or "litho_sim.plugins"
    name = f"{base}.{plugin_type}.{plugin_name}"
    return logging.getLogger(name)


class PluginLogger:
    """插件专用日志记录器包装

    在标准 logging 基础上附加：
        - 自动注入 plugin 上下文（type / name / version）
        - 结构化事件记录（用于后续分析）
        - 计时装饰器
    """

    def __init__(
        self,
        plugin_type: str,
        plugin_name: str,
        version: str = "",
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.plugin_type = plugin_type
        self.plugin_name = plugin_name
        self.version = version
        self._logger = logger or get_plugin_logger(plugin_type, plugin_name)
        self._events: List[Dict[str, Any]] = []
        self._extra_context: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 上下文
    # ------------------------------------------------------------------
    def bind(self, **kwargs: Any) -> "PluginLogger":
        """增加附加上下文字段，所有后续日志都将携带"""
        self._extra_context.update(kwargs)
        return self

    def unbind(self, *keys: str) -> "PluginLogger":
        for k in keys:
            self._extra_context.pop(k, None)
        return self

    def _prefix(self) -> str:
        parts = [f"[{self.plugin_type}:{self.plugin_name}"]
        if self.version:
            parts.append(f"v{self.version}")
        for k, v in self._extra_context.items():
            parts.append(f"{k}={v}")
        return " ".join(parts) + "]"

    # ------------------------------------------------------------------
    # 标准日志方法
    # ------------------------------------------------------------------
    def debug(self, msg: str, **extra: Any) -> None:
        self._logger.debug("%s %s %s", self._prefix(), msg, _format_extra(extra))

    def info(self, msg: str, **extra: Any) -> None:
        self._logger.info("%s %s %s", self._prefix(), msg, _format_extra(extra))

    def warning(self, msg: str, **extra: Any) -> None:
        self._logger.warning("%s %s %s", self._prefix(), msg, _format_extra(extra))

    def error(self, msg: str, **extra: Any) -> None:
        self._logger.error("%s %s %s", self._prefix(), msg, _format_extra(extra))

    def exception(self, msg: str, **extra: Any) -> None:
        self._logger.exception("%s %s %s", self._prefix(), msg, _format_extra(extra))

    # ------------------------------------------------------------------
    # 结构化事件
    # ------------------------------------------------------------------
    def event(self, event_type: str, **payload: Any) -> None:
        """记录结构化事件，同时保留在内存事件列表中"""
        evt = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "plugin_type": self.plugin_type,
            "plugin_name": self.plugin_name,
            "version": self.version,
            "event": event_type,
            "payload": dict(payload),
        }
        self._events.append(evt)
        self.info(f"event:{event_type}", **payload)

    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def clear_events(self) -> None:
        self._events.clear()

    # ------------------------------------------------------------------
    # 计时
    # ------------------------------------------------------------------
    def timed(self, name: Optional[str] = None, **extra: Any) -> "_TimedContext":
        """上下文管理器，记录耗时"""
        return _TimedContext(self, name=name, extra=extra)

    def timeit(self, func: Optional[Callable] = None, *, name: Optional[str] = None):
        """装饰器版 timed"""
        def decorator(fn: Callable) -> Callable:
            label = name or fn.__name__

            def wrapper(*args, **kwargs):
                with self.timed(name=label):
                    return fn(*args, **kwargs)
            wrapper.__name__ = fn.__name__
            return wrapper

        if func is None:
            return decorator
        return decorator(func)


@dataclass
class _TimedContext:
    logger: PluginLogger
    name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    start: float = field(default=0.0)

    def __enter__(self) -> "_TimedContext":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ms = (time.perf_counter() - self.start) * 1000.0
        payload = {"duration_ms": round(elapsed_ms, 3), **self.extra}
        payload["success"] = exc_type is None
        if exc_type is not None:
            payload["error"] = f"{exc_type.__name__}: {exc_val}"
        self.logger.event(self.name or "timed_block", **payload)


def _format_extra(extra: Dict[str, Any]) -> str:
    if not extra:
        return ""
    parts = [f"{k}={v}" for k, v in extra.items()]
    return " | " + ", ".join(parts)
