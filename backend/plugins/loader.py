# -*- coding: utf-8 -*-
"""
插件加载器

负责从以下渠道发现并加载插件：
    1. setuptools entry_points（第三方 pip 包注册）
    2. YAML/JSON 配置文件（动态指定 module:class 路径）
    3. 本地目录扫描（可选的扩展点）
"""

import importlib
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
)

from plugins.base import BasePlugin, PluginMetadata, PluginType
from plugins.registry import PluginRegistry, PluginEntry

logger = logging.getLogger(__name__)

# 框架支持的 entry_point group 名称
ENTRY_POINT_GROUPS: Dict[PluginType, str] = {
    PluginType.IMAGING_BACKEND: "litho_sim.imaging_backends",
    PluginType.OPTIMIZER: "litho_sim.optimizers",
    PluginType.LOSS_FUNCTION: "litho_sim.loss_functions",
    PluginType.WORKFLOW: "litho_sim.workflows",
}


class EntryPointLoader:
    """基于 setuptools entry_points 的插件加载器

    典型的 pyproject.toml 声明示例（第三方包）：

        [project.entry-points."litho_sim.optimizers"]
        adam = "my_package.optim:AdamOptimizer"
    """

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # PEP 621 / setuptools 方式（importlib.metadata）
    # ------------------------------------------------------------------
    def discover_all(self, group: Optional[str] = None) -> int:
        """通过 importlib.metadata 发现并注册 entry_points 插件

        Returns:
            成功注册数量
        """
        groups = [group] if group else list(ENTRY_POINT_GROUPS.values())
        count = 0
        try:
            from importlib.metadata import entry_points as _iter_eps

            for g in groups:
                try:
                    eps = _iter_eps(group=g)
                except TypeError:
                    eps = _iter_eps().select(group=g)
                for ep in list(eps):
                    count += self._register_entry_point(g, ep)
        except Exception as e:
            logger.warning("通过 importlib.metadata 扫描插件失败: %s", e)
            count = self._fallback_pkg_resources(groups)
        return count

    def _fallback_pkg_resources(self, groups: List[str]) -> int:
        """旧版兼容：尝试使用 pkg_resources"""
        count = 0
        try:
            import pkg_resources  # type: ignore
            for g in groups:
                for ep in pkg_resources.iter_entry_points(g):
                    count += self._register_entry_point(g, ep)
        except Exception as e:
            logger.warning("pkg_resources 也不可用，跳过 entry_points 发现")
        return count

    def _register_entry_point(self, group: str, ep: Any) -> int:
        pt = _resolve_plugin_type_from_group(group)
        if pt is None:
            return 0
        name = ep.name

        def _factory() -> Type[BasePlugin]:
            obj = ep.load()
            if isinstance(obj, type) and issubclass(obj, BasePlugin):
                return obj
            if callable(obj):
                cls = obj()
                if isinstance(cls, type) and issubclass(cls, BasePlugin):
                    return cls
            raise TypeError(
                f"entry_point {group}:{name} 解析失败：期望返回 BasePlugin 子类"
            )

        try:
            # 尝试立即加载并获取 metadata
            loaded_cls: Optional[Type[BasePlugin]] = None
            metadata: Optional[PluginMetadata] = None
            try:
                obj = ep.load()
                if isinstance(obj, type) and issubclass(obj, BasePlugin):
                    loaded_cls = obj
                    if hasattr(obj, "get_metadata"):
                        metadata = obj.get_metadata()
                elif callable(obj):
                    maybe_cls = obj()
                    if isinstance(maybe_cls, type) and issubclass(maybe_cls, BasePlugin):
                        loaded_cls = maybe_cls
                        if hasattr(maybe_cls, "get_metadata"):
                            metadata = maybe_cls.get_metadata()
            except Exception:
                loaded_cls = None
                metadata = None

            if loaded_cls is not None and metadata is not None:
                self._registry.register(
                    plugin_class=loaded_cls,
                    source=f"entry_point:{group}:{name}",
                    replace=True,
                )
                return 1

            # 懒加载回退
            if metadata is None:
                metadata = PluginMetadata(
                    name=name,
                    version="0.0.0",
                    plugin_type=pt,
                    description=f"entry_point:{group}:{name}",
                )
            self._registry.register(
                factory=_factory,
                metadata=metadata,
                source=f"entry_point:{group}:{name}",
                replace=True,
            )
            return 1
        except Exception as e:
            logger.error("注册 entry_point 插件 %s 失败: %s", name, e)
            return 0


def _resolve_plugin_type_from_group(group: str) -> Optional[PluginType]:
    for pt, g in ENTRY_POINT_GROUPS.items():
        if g == group:
            return pt
    return None


# ============================================================================
# 配置文件加载器
# ============================================================================

@dataclass
class _PluginConfig:
    """配置文件中单个插件的配置结构"""
    name: str
    plugin_type: PluginType
    module_path: str          # "package.module:ClassName
    config: Dict[str, Any]
    enabled: bool = True
    source: str = "config"


class ConfigLoader:
    """从 YAML/JSON 配置文件加载插件

    配置文件示例：

        plugins:
          imaging_backends:
            fast_rcwa:
              enabled: true
              class: "my_package.rcwa:RCWAImaging"
              config:
                max_diffraction_order: 10

          optimizers:
            custom_adam:
              enabled: true
              class: "my_opt.adam:AdamOptimizer"

          loss_functions:
            weighted_epe:
              enabled: true
              class: "my_losses.epe:WeightedEPELoss"

          workflows:
            my_calibration:
              enabled: false
              class: "my_flows.calib:CalibrationWorkflow"
    """

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def load_from_file(
        self,
        config_path,
    ) -> int:
        """从 YAML/JSON 配置文件批量注册插件

        Returns:
            成功注册数量
        """
        import yaml
        import json

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"插件配置文件不存在: {path}")

        suffix = path.suffix.lower()
        with open(path, "r", encoding="utf-8") as f:
            if suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f) or {}
            elif suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"不支持的插件配置格式: {suffix}")

        return self.load_from_dict(data, source=str(path))

    def load_from_dict(
        self,
        data: Dict[str, Any],
        source: str = "dict",
    ) -> int:
        """从字典批量注册插件"""
        plugins_root: Dict[str, Any] = data.get("plugins", data)
        count = 0

        type_keys: Dict[PluginType, str] = {
            PluginType.IMAGING_BACKEND: "imaging_backends",
            PluginType.OPTIMIZER: "optimizers",
            PluginType.LOSS_FUNCTION: "loss_functions",
            PluginType.WORKFLOW: "workflows",
        }

        for pt, key in type_keys.items():
            bucket = plugins_root.get(key, {})
            if not isinstance(bucket, dict):
                continue
            for name, spec in bucket.items():
                try:
                    if not spec.get("enabled", True):
                        logger.info("插件 %s 已在配置中禁用", name)
                        continue
                    cfg = _PluginConfig(
                        name=name,
                        plugin_type=pt,
                        module_path=spec["class"],
                        config=spec.get("config", {}),
                        source=f"config:{source}",
                    )
                    self._register_single(cfg)
                    count += 1
                except Exception as e:
                    logger.error("加载配置插件 %s 失败: %s", name, e)
        return count

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _register_single(self, cfg: _PluginConfig) -> None:
        module_path, _, class_name = cfg.module_path.partition(":")
        if class_name:
            cls = self._import_attr(module_path, class_name)
        else:
            last_dot = module_path.rfind(".")
            if last_dot < 0:
                raise ValueError(
                    f"模块路径缺少分隔符，应为 module:ClassName 或 module.ClassName: {cfg.module_path}"
                )
            cls = self._import_attr(
                module_path[:last_dot], module_path[last_dot + 1 :]
            )

        if not isinstance(cls, type) or not issubclass(cls, BasePlugin):
            raise TypeError(
                f"{cfg.module_path!r} 不是 BasePlugin 子类"
            )

        # 优先使用类本身的 metadata，若类型不匹配会在 registry.register 里会被覆盖/校验
        metadata = cls.get_metadata()
        if metadata.plugin_type != cfg.plugin_type:
            logger.warning(
                "插件 %s 类声明类型 %s 与配置指定类型 %s 不一致，以类声明为准",
                cfg.name, metadata.plugin_type.value, cfg.plugin_type.value
            )

        self._registry.register(
            plugin_class=cls,
            source=cfg.source,
            config=cfg.config,
            replace=True,
        )

    @staticmethod
    def _import_attr(module_path: str, attr: str):
        module = importlib.import_module(module_path)
        return getattr(module, attr)


# ============================================================================
# 目录扫描加载器（可选）
# ============================================================================

class DirectoryLoader:
    """从一个目录扫描 Python 扩展。"""

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    def load_dir(self, directory) -> int:
        """
        扫描目录下的 *plugin*.py 文件并从中提取插件类。"""
        count = 0
        d = Path(directory)
        if not d.is_dir():
            return 0
        for py in sorted(d.glob("**/*plugin*.py")):
            try:
                count += self._load_file(py)
            except Exception as e:
                logger.warning("扫描文件 %s 失败: %s", py, e)
        return count

    def _load_file(self, py_file: Path) -> int:
        count = 0
        spec = importlib.util.spec_from_file_location(
            f"_litho_plugin_{py_file.stem}", str(py_file)
        )
        if spec is None or spec.loader is None:
            return 0
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BasePlugin)
                and obj is not BasePlugin
                and getattr(obj, "__module__", "") == module.__name__
                and not getattr(obj, "_abstract", False)
            ):
                try:
                    self._registry.register(
                        plugin_class=obj,
                        source=f"dir:{py_file}",
                    )
                    count += 1
                except Exception:
                    pass
        return count


# ============================================================================
# 聚合加载器
# ============================================================================

class PluginLoader:
    """统一封装所有加载策略的聚合入口"""

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry
        self.entry_point = EntryPointLoader(registry)
        self.config = ConfigLoader(registry)
        self.directory = DirectoryLoader(registry)

    def discover_all(self) -> Dict[str, int]:
        """执行默认发现：entry_points"""
        return {
            "entry_points": self.entry_point.discover_all(),
        }

    def load_from_file(self, config_path) -> int:
        return self.config.load_from_file(config_path)

    def load_from_dict(self, data: Dict[str, Any], source: str = "dict") -> int:
        return self.config.load_from_dict(data, source)

    def load_dir(self, directory) -> int:
        return self.directory.load_dir(directory)
