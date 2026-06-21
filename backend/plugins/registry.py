# -*- coding: utf-8 -*-
"""
插件注册表

存储已发现并注册的插件条目，支持按类型、名称查询，以及按优先级排序。
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Type,
)

from plugins.base import (
    BasePlugin,
    ImagingBackend,
    LossFunction,
    Optimizer,
    PluginMetadata,
    PluginType,
    Workflow,
)

logger = logging.getLogger(__name__)

_PLUGIN_TYPE_CLASS_MAP: Dict[PluginType, Type[BasePlugin]] = {
    PluginType.IMAGING_BACKEND: ImagingBackend,
    PluginType.OPTIMIZER: Optimizer,
    PluginType.LOSS_FUNCTION: LossFunction,
    PluginType.WORKFLOW: Workflow,
}


@dataclass
class PluginEntry:
    """注册表中的单个插件条目

    Attributes:
        metadata:       插件元数据
        plugin_class:   插件类（懒加载时可能为 None，需通过 factory 获取）
        factory:        插件类的工厂 callable，支持延迟导入
        source:         来源描述（"entry_point:xxx" / "config:xxx.yaml" / "builtin"）
        config:         该插件在注册时绑定的默认配置覆盖（instance 级别）
    """
    metadata: PluginMetadata
    plugin_class: Optional[Type[BasePlugin]] = None
    factory: Optional[Callable[[], Type[BasePlugin]]] = None
    source: str = "unknown"
    config: Dict[str, Any] = field(default_factory=dict)

    def get_class(self) -> Type[BasePlugin]:
        """获取插件类，触发懒加载"""
        if self.plugin_class is None:
            if self.factory is None:
                raise RuntimeError(
                    f"插件 {self.metadata.name!r} 未提供 plugin_class 也未提供 factory"
                )
            self.plugin_class = self.factory()
            self._validate_class(self.plugin_class)
        return self.plugin_class

    def _validate_class(self, cls: Type[BasePlugin]) -> None:
        """校验插件类与声明类型匹配"""
        expected = _PLUGIN_TYPE_CLASS_MAP.get(self.metadata.plugin_type)
        if expected is not None and not issubclass(cls, expected):
            raise TypeError(
                f"插件 {self.metadata.name!r} 声明类型为 {self.metadata.plugin_type.value}, "
                f"但其类 {cls.__name__} 并未继承 {expected.__name__}"
            )


class PluginRegistry:
    """插件注册表

    线程安全的插件条目存储，支持：
        - 按插件类型 + 名称查找
        - 按类型枚举全部插件
        - 按优先级返回排序结果
        - 重复注册检测 / 覆盖策略
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 结构: { plugin_type: { name: PluginEntry } }
        self._entries: Dict[PluginType, Dict[str, PluginEntry]] = {
            t: {} for t in PluginType
        }
        self._register_history: List[PluginEntry] = []

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(
        self,
        plugin_class: Optional[Type[BasePlugin]] = None,
        *,
        metadata: Optional[PluginMetadata] = None,
        factory: Optional[Callable[[], Type[BasePlugin]]] = None,
        source: str = "unknown",
        config: Optional[Dict[str, Any]] = None,
        replace: bool = False,
    ) -> PluginEntry:
        """注册一个插件

        可以直接传入 plugin_class，或传入 factory 实现懒加载。
        """
        if plugin_class is None and factory is None:
            raise ValueError("必须提供 plugin_class 或 factory 之一")

        # 优先使用传入的 metadata；否则尝试从类获取
        if metadata is None:
            if plugin_class is None:
                # 懒加载模式下必须提供 metadata
                raise ValueError(
                    "使用 factory 懒加载注册时必须显式传入 metadata"
                )
            if not hasattr(plugin_class, "get_metadata"):
                raise TypeError(
                    f"插件类 {plugin_class.__name__} 未实现 get_metadata() 类方法"
                )
            metadata = plugin_class.get_metadata()

        self._validate_metadata(metadata)

        with self._lock:
            existing = self._entries[metadata.plugin_type].get(metadata.name)
            if existing is not None and not replace:
                if existing.metadata.priority >= metadata.priority:
                    logger.info(
                        "插件 %s(%s) 已存在（优先级 %d），跳过优先级更低的注册（来自 %s）",
                        metadata.name, metadata.plugin_type.value,
                        existing.metadata.priority, source,
                    )
                    return existing
                logger.info(
                    "高优先级插件 %s(%s) 覆盖既有注册（%d > %d）",
                    metadata.name, metadata.plugin_type.value,
                    metadata.priority, existing.metadata.priority,
                )

            entry = PluginEntry(
                metadata=metadata,
                plugin_class=plugin_class,
                factory=factory,
                source=source,
                config=dict(config) if config else {},
            )
            self._entries[metadata.plugin_type][metadata.name] = entry
            self._register_history.append(entry)
            logger.debug(
                "已注册插件: %s (%s) 来源=%s",
                metadata.name, metadata.plugin_type.value, source,
            )
            return entry

    # ------------------------------------------------------------------
    # 注销
    # ------------------------------------------------------------------
    def unregister(
        self,
        plugin_type: PluginType,
        name: str,
    ) -> bool:
        """从注册表移除一个插件，返回是否实际移除"""
        with self._lock:
            bucket = self._entries.get(plugin_type, {})
            if name in bucket:
                del bucket[name]
                logger.info("已注销插件: %s (%s)", name, plugin_type.value)
                return True
            return False

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def has(self, plugin_type: PluginType, name: str) -> bool:
        with self._lock:
            return name in self._entries.get(plugin_type, {})

    def get(
        self,
        plugin_type: PluginType,
        name: str,
    ) -> Optional[PluginEntry]:
        """获取指定条目（不触发加载）"""
        with self._lock:
            return self._entries.get(plugin_type, {}).get(name)

    def list(
        self,
        plugin_type: Optional[PluginType] = None,
        *,
        sort_by_priority: bool = True,
    ) -> List[PluginEntry]:
        """枚举插件条目

        Args:
            plugin_type:        过滤类型，None 表示全部
            sort_by_priority:   是否按优先级降序排列
        """
        with self._lock:
            if plugin_type is None:
                items: Iterable[PluginEntry] = (
                    e for bucket in self._entries.values() for e in bucket.values()
                )
            else:
                items = iter(self._entries.get(plugin_type, {}).values())

            result = list(items)

        if sort_by_priority:
            result.sort(
                key=lambda e: (e.metadata.priority, e.metadata.name),
                reverse=True,
            )
        return result

    def list_names(
        self,
        plugin_type: Optional[PluginType] = None,
    ) -> Set[str]:
        return {e.metadata.name for e in self.list(plugin_type, sort_by_priority=False)}

    def list_by_tag(
        self,
        tag: str,
        plugin_type: Optional[PluginType] = None,
    ) -> List[PluginEntry]:
        """按标签筛选插件"""
        result: List[PluginEntry] = []
        for e in self.list(plugin_type):
            if tag in e.metadata.tags:
                result.append(e)
        return result

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """清空注册表（主要用于测试）"""
        with self._lock:
            for bucket in self._entries.values():
                bucket.clear()
            self._register_history.clear()

    def summary(self) -> Dict[str, List[Dict[str, Any]]]:
        """生成注册表摘要（用于 API / UI 展示）"""
        result: Dict[str, List[Dict[str, Any]]] = {}
        for pt in PluginType:
            entries = self.list(pt)
            result[pt.value] = [
                {
                    "name": e.metadata.name,
                    "version": e.metadata.version,
                    "description": e.metadata.description,
                    "author": e.metadata.author,
                    "tags": list(e.metadata.tags),
                    "priority": e.metadata.priority,
                    "source": e.source,
                    "requires": list(e.metadata.requires),
                }
                for e in entries
            ]
        return result

    @staticmethod
    def _validate_metadata(metadata: PluginMetadata) -> None:
        if not metadata.name:
            raise ValueError("插件名称不能为空")
        if not metadata.name.replace("_", "").isalnum():
            raise ValueError(
                f"插件名称只能包含字母数字下划线，非法名称: {metadata.name!r}"
            )
        if not metadata.version:
            raise ValueError(f"插件 {metadata.name!r} 未声明版本")
        if not isinstance(metadata.plugin_type, PluginType):
            raise TypeError(
                f"插件 {metadata.name!r} 的 plugin_type 必须是 PluginType 枚举"
            )
