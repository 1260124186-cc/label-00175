# -*- coding: utf-8 -*-
"""
插件配置管理

负责插件配置的三层合并（默认 ← 注册 ← 用户）、校验与序列化。
"""

import copy
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
)

from plugins.base import BasePlugin, PluginType

logger = logging.getLogger(__name__)


class PluginConfig:
    """单个插件的配置管理器

    合并策略（低优先级 → 高优先级）：
        1. Class.get_default_config()   —— 插件作者写死的默认值
        2. register(config=...)         —— 注册时绑定的配置
        3. manager.create(config=...)   —— 用户创建实例时传入的配置
    """

    def __init__(
        self,
        plugin_class: Type[BasePlugin],
        register_config: Optional[Dict[str, Any]] = None,
        user_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._plugin_class = plugin_class
        self._register_config: Dict[str, Any] = dict(register_config) if register_config else {}
        self._user_config: Dict[str, Any] = dict(user_config) if user_config else {}
        self._merged: Dict[str, Any] = self._merge()

    # ------------------------------------------------------------------
    # 合并逻辑
    # ------------------------------------------------------------------
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(base)
        for k, v in override.items():
            if (
                k in result
                and isinstance(result[k], dict)
                and isinstance(v, dict)
            ):
                result[k] = PluginConfig._deep_merge(result[k], v)
            else:
                result[k] = copy.deepcopy(v)
        return result

    def _merge(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        try:
            result = self._deep_merge(result, self._plugin_class.get_default_config())
        except Exception:
            pass
        result = self._deep_merge(result, self._register_config)
        result = self._deep_merge(result, self._user_config)
        return result

    # ------------------------------------------------------------------
    # 访问 API
    # ------------------------------------------------------------------
    @property
    def merged(self) -> Dict[str, Any]:
        """返回合并后的配置字典（深拷贝快照）"""
        return copy.deepcopy(self._merged)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号分隔路径 a.b.c"""
        node: Any = self._merged
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def require(self, key: str) -> Any:
        """必选配置，缺失则抛异常"""
        val = self.get(key)
        if val is None:
            raise KeyError(f"缺少必填配置项: {key}")
        return val

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(self) -> List[str]:
        """使用插件类提供的 JSON Schema 做基础校验，返回错误列表"""
        errors: List[str] = []
        try:
            schema = self._plugin_class.get_config_schema()
        except Exception:
            return errors
        if not schema:
            return errors

        # 简单的必填项校验（不引入 jsonschema 避免硬依赖）
        required = schema.get("required", [])
        for req in required:
            if self.get(req) is None:
                errors.append(f"缺少必填配置: {req}")

        # 类型基本检查
        props = schema.get("properties", {})
        for key, prop_def in props.items():
            val = self.get(key)
            if val is None:
                continue
            expected_type = prop_def.get("type")
            if expected_type and not _check_type(val, expected_type):
                errors.append(
                    f"配置 {key} 类型不匹配: 期望 {expected_type}, 实际 {type(val).__name__}"
                )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default": copy.deepcopy(self._plugin_class.get_default_config()),
            "register": copy.deepcopy(self._register_config),
            "user": copy.deepcopy(self._user_config),
            "merged": copy.deepcopy(self._merged),
        }


def _check_type(val: Any, type_name: str) -> bool:
    """简单的 JSON 类型检查"""
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    t = mapping.get(type_name)
    if t is None:
        return True
    # 注意 bool 是 int 的子类，这里单独处理
    if type_name == "integer" and isinstance(val, bool):
        return False
    if type_name == "number" and isinstance(val, bool):
        return False
    return isinstance(val, t)


def merge_plugin_config(
    plugin_class: Type[BasePlugin],
    *config_layers: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """便捷函数：顺序合并多层配置

    后面的层覆盖前面的。
    """
    merged: Dict[str, Any] = {}
    try:
        merged = PluginConfig._deep_merge(merged, plugin_class.get_default_config())
    except Exception:
        pass
    for cfg in config_layers:
        if cfg:
            merged = PluginConfig._deep_merge(merged, cfg)
    return merged
