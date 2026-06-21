# -*- coding: utf-8 -*-
"""
插件管理器：对外主接口

统一封装注册、加载、创建、生命周期管理等功能。
"""

import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
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
from plugins.loader import PluginLoader
from plugins.registry import PluginEntry, PluginRegistry

logger = logging.getLogger(__name__)


_TYPE_CLASS_MAP: Dict[PluginType, Type[BasePlugin]] = {
    PluginType.IMAGING_BACKEND: ImagingBackend,
    PluginType.OPTIMIZER: Optimizer,
    PluginType.LOSS_FUNCTION: LossFunction,
    PluginType.WORKFLOW: Workflow,
}


def _resolve_type(plugin_type: Union[str, PluginType]) -> PluginType:
    if isinstance(plugin_type, PluginType):
        return plugin_type
    return PluginType.from_string(plugin_type)


class PluginManager:
    """插件管理器（对外主接口）

    典型使用:

        >>> manager = PluginManager(auto_discover=True)
        >>> manager.load_from_config("plugins.yaml")
        >>> opt = manager.create("optimizer", "adam", {"learning_rate": 0.01})
        >>> manager.shutdown_all()
    """

    def __init__(
        self,
        registry: Optional[PluginRegistry] = None,
        loader: Optional[PluginLoader] = None,
        *,
        auto_discover: bool = False,
        register_builtin: bool = True,
        global_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.registry: PluginRegistry = registry or PluginRegistry()
        self.loader: PluginLoader = loader or PluginLoader(self.registry)
        self._instances: Dict[str, BasePlugin] = {}  # key: f"{type_value}:{name}"
        self._global_context: Dict[str, Any] = dict(global_context) if global_context else {}

        if register_builtin:
            try:
                from plugins.builtin import register_builtin_plugins
                register_builtin_plugins(self.registry)
            except Exception as e:
                logger.warning("内置插件注册失败: %s", e)

        if auto_discover:
            try:
                self.loader.discover_all()
            except Exception as e:
                logger.warning("自动发现 entry_points 失败: %s", e)

    # ------------------------------------------------------------------
    # 注册 / 加载
    # ------------------------------------------------------------------
    def register(
        self,
        plugin_class: Type[BasePlugin],
        *,
        config: Optional[Dict[str, Any]] = None,
        source: str = "manual",
    ) -> PluginEntry:
        """手动注册一个插件类"""
        return self.registry.register(
            plugin_class=plugin_class,
            source=source,
            config=config,
        )

    def discover(self) -> Dict[str, int]:
        """通过 entry_points 自动发现第三方插件"""
        return self.loader.discover_all()

    def load_from_file(self, config_path) -> int:
        """从 YAML/JSON 配置文件加载插件声明"""
        return self.loader.load_from_file(config_path)

    def load_from_dict(self, data: Dict[str, Any], source: str = "dict") -> int:
        """从字典加载插件声明"""
        return self.loader.load_from_dict(data, source)

    def load_dir(self, directory) -> int:
        """扫描一个目录，自动注册其中的插件"""
        return self.loader.load_dir(directory)

    # ------------------------------------------------------------------
    # 元数据查询
    # ------------------------------------------------------------------
    def has(
        self,
        plugin_type: Union[str, PluginType],
        name: str,
    ) -> bool:
        return self.registry.has(_resolve_type(plugin_type), name)

    def get_metadata(
        self,
        plugin_type: Union[str, PluginType],
        name: str,
    ) -> Optional[PluginMetadata]:
        entry = self.registry.get(_resolve_type(plugin_type), name)
        return entry.metadata if entry else None

    def list_plugins(
        self,
        plugin_type: Optional[Union[str, PluginType]] = None,
    ) -> List[PluginEntry]:
        t = _resolve_type(plugin_type) if plugin_type else None
        return self.registry.list(t)

    def summary(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.registry.summary()

    # ------------------------------------------------------------------
    # 实例创建
    # ------------------------------------------------------------------
    def create(
        self,
        plugin_type: Union[str, PluginType],
        name: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        initialize: bool = True,
        singleton: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> BasePlugin:
        """创建插件实例

        Args:
            plugin_type:    插件类型
            name:           插件名称
            config:         用户配置（与默认配置、注册时配置三层合并）
            initialize:     是否立即执行 initialize()
            singleton:      是否复用已创建的单例
            context:        初始化上下文

        Returns:
            初始化后的插件实例
        """
        pt = _resolve_type(plugin_type)
        entry = self.registry.get(pt, name)
        if entry is None:
            raise KeyError(
                f"插件未注册: type={pt.value}, name={name!r}. "
                f"可用列表: {sorted(self.registry.list_names(pt))}"
            )

        key = f"{pt.value}:{name}"
        if singleton and key in self._instances:
            instance = self._instances[key]
            if config:
                instance.update_config(**config)
            return instance

        # 三层配置合并: class_defaults < register_config < user_config
        plugin_cls = entry.get_class()
        merged_cfg: Dict[str, Any] = {}
        try:
            merged_cfg.update(plugin_cls.get_default_config())
        except Exception:
            pass
        merged_cfg.update(entry.config)
        if config:
            merged_cfg.update(config)

        instance = plugin_cls(merged_cfg)
        expected_cls = _TYPE_CLASS_MAP[pt]
        if not isinstance(instance, expected_cls):
            raise TypeError(
                f"插件 {name!r} 类型不匹配：期望 {expected_cls.__name__} 实例"
            )

        if singleton:
            self._instances[key] = instance

        if initialize:
            ctx = dict(self._global_context)
            if context:
                ctx.update(context)
            ctx.setdefault("plugin_name", name)
            ctx.setdefault("plugin_type", pt.value)
            ctx.setdefault("logger", logging.getLogger(f"plugins.{pt.value}.{name}"))
            instance.initialize(ctx)

        return instance

    def create_imaging(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ImagingBackend:
        return self.create("imaging_backend", name, config, **kwargs)  # type: ignore

    def create_optimizer(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optimizer:
        return self.create("optimizer", name, config, **kwargs)  # type: ignore

    def create_loss(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LossFunction:
        return self.create("loss_function", name, config, **kwargs)  # type: ignore

    def create_workflow(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Workflow:
        return self.create("workflow", name, config, **kwargs)  # type: ignore

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def set_global_context(self, **kwargs: Any) -> None:
        """设置全局上下文，会合并到每个插件的 initialize 调用中"""
        self._global_context.update(kwargs)

    def shutdown(self, plugin_type: Union[str, PluginType], name: str) -> bool:
        pt = _resolve_type(plugin_type)
        key = f"{pt.value}:{name}"
        instance = self._instances.pop(key, None)
        if instance is None:
            return False
        instance.shutdown()
        return True

    def shutdown_all(self) -> None:
        """关闭所有已创建的插件实例"""
        for instance in list(self._instances.values()):
            try:
                instance.shutdown()
            except Exception as e:
                logger.error("关闭插件实例失败: %s", e)
        self._instances.clear()

    def __enter__(self) -> "PluginManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown_all()
