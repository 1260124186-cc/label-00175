# -*- coding: utf-8 -*-
"""
插件化算法 SDK 模块

提供可扩展的算法插件框架，支持第三方开发者通过 entry_points 或配置文件
挂载自定义算法实现。框架负责统一的配置管理、日志记录与结果序列化。

四类核心插件接口：
    - ImagingBackend: 光学成像后端插件
    - Optimizer:      优化算法插件
    - LossFunction:   损失函数插件
    - Workflow:       工作流插件

使用示例：
    >>> from plugins import PluginManager
    >>> manager = PluginManager()
    >>> manager.discover()  # 自动发现 entry_points 中的插件
    >>> manager.load_from_config("plugins_config.yaml")  # 从配置文件加载
    >>>
    >>> # 获取插件实例
    >>> optimizer = manager.create("optimizer", "adam", {"learning_rate": 0.001})
    >>> loss_fn = manager.create("loss", "mse_weighted", {"weights": [1.0, 2.0]})
"""

try:
    from plugins.base import (
        PluginType,
        PluginMetadata,
        BasePlugin,
        ImagingBackend,
        Optimizer,
        LossFunction,
        Workflow,
    )
    from plugins.registry import PluginRegistry, PluginEntry
    from plugins.loader import PluginLoader, EntryPointLoader, ConfigLoader
    from plugins.manager import PluginManager
    from plugins.config import PluginConfig, merge_plugin_config
    from plugins.logging import PluginLogger, get_plugin_logger
    from plugins.serializer import (
        PluginResult,
        ResultSerializer,
        serialize_result,
        deserialize_result,
    )
    from plugins.builtin import register_builtin_plugins
except ImportError:
    from .base import (
        PluginType,
        PluginMetadata,
        BasePlugin,
        ImagingBackend,
        Optimizer,
        LossFunction,
        Workflow,
    )
    from .registry import PluginRegistry, PluginEntry
    from .loader import PluginLoader, EntryPointLoader, ConfigLoader
    from .manager import PluginManager
    from .config import PluginConfig, merge_plugin_config
    from .logging import PluginLogger, get_plugin_logger
    from .serializer import (
        PluginResult,
        ResultSerializer,
        serialize_result,
        deserialize_result,
    )
    from .builtin import register_builtin_plugins

__version__ = "1.0.0"
__author__ = "Lithography Simulation Team"

__all__ = [
    # 插件类型
    "PluginType",
    "PluginMetadata",
    "BasePlugin",
    "ImagingBackend",
    "Optimizer",
    "LossFunction",
    "Workflow",
    # 注册表
    "PluginRegistry",
    "PluginEntry",
    # 加载器
    "PluginLoader",
    "EntryPointLoader",
    "ConfigLoader",
    # 管理器（对外主接口）
    "PluginManager",
    # 配置
    "PluginConfig",
    "merge_plugin_config",
    # 日志
    "PluginLogger",
    "get_plugin_logger",
    # 序列化
    "PluginResult",
    "ResultSerializer",
    "serialize_result",
    "deserialize_result",
    # 内置注册
    "register_builtin_plugins",
]
