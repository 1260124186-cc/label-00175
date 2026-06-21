# -*- coding: utf-8 -*-
"""
插件结果序列化器

统一处理插件算法输出的序列化 / 反序列化，支持 JSON、YAML 导出。
"""

import base64
import copy
import io
import json
import logging
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np

from plugins.base import PluginType

logger = logging.getLogger(__name__)


@dataclass
class PluginResult:
    """插件算法执行结果的统一封装

    Attributes:
        plugin_type:    插件类型
        plugin_name:    插件名称
        plugin_version: 插件版本
        success:        是否成功
        started_at:     开始时间 ISO 字符串
        finished_at:    结束时间 ISO 字符串
        duration_ms:    耗时
        inputs:         输入快照（尽可能可序列化）
        outputs:        输出数据（包含 numpy 数组等）
        metrics:        关键指标（标量，便于索引/绘图）
        output_files:   写出的文件路径列表
        logs:           运行日志（可选）
        error:          错误信息（成功时为 None）
        extra:          扩展字段
    """
    plugin_type: str
    plugin_name: str
    plugin_version: str = ""
    success: bool = True
    started_at: str = ""
    finished_at: str = ""
    duration_ms: float = 0.0
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典"""
        data = asdict(self)
        data["outputs"] = _to_serializable(self.outputs)
        data["inputs"] = _to_serializable(self.inputs)
        data["extra"] = _to_serializable(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginResult":
        """从反序列化后的字典重建"""
        d = dict(data)
        d["outputs"] = _from_serializable(d.get("outputs", {}))
        d["inputs"] = _from_serializable(d.get("inputs", {}))
        d["extra"] = _from_serializable(d.get("extra", {}))
        return cls(**d)


# ============================================================================
# 序列化工具
# ============================================================================

def _convert_numpy(obj: Any) -> Any:
    """递归转换 numpy 类型"""
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "data": base64.b64encode(np.ascontiguousarray(obj).tobytes()).decode("ascii"),
        }
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _restore_numpy(obj: Dict[str, Any]) -> Any:
    if obj.get("__ndarray__"):
        shape = tuple(obj["shape"])
        dtype = np.dtype(obj["dtype"])
        buf = base64.b64decode(obj["data"].encode("ascii"))
        return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
    return obj


def _convert_pathlike(obj: Any) -> Any:
    if isinstance(obj, Path):
        return {"__path__": str(obj)}
    if isinstance(obj, datetime):
        return {"__datetime__": obj.isoformat()}
    return obj


def _restore_pathlike(obj: Dict[str, Any]) -> Any:
    if "__path__" in obj:
        return Path(obj["__path__"])
    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["__datetime__"])
    return obj


def _to_serializable(obj: Any) -> Any:
    """递归转换对象为可 JSON 序列化结构"""
    if obj is None:
        return None
    conv = _convert_numpy(obj)
    if conv is not obj:
        return conv
    conv = _convert_pathlike(obj)
    if conv is not obj:
        return conv
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_serializable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _from_serializable(obj: Any) -> Any:
    """反向还原"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return _restore_numpy(obj)
        if "__path__" in obj or "__datetime__" in obj:
            return _restore_pathlike(obj)
        return {k: _from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_serializable(v) for v in obj]
    return obj


# ============================================================================
# 序列化器主类
# ============================================================================

class ResultSerializer:
    """插件结果序列化器"""

    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    def to_json(self, result: PluginResult) -> str:
        return json.dumps(result.to_dict(), indent=self._indent, ensure_ascii=False)

    def from_json(self, s: str) -> PluginResult:
        data = json.loads(s)
        return PluginResult.from_dict(data)

    def dump_json(self, result: PluginResult, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(result), encoding="utf-8")

    def load_json(self, path: Union[str, Path]) -> PluginResult:
        return self.from_json(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # YAML
    # ------------------------------------------------------------------
    def to_yaml(self, result: PluginResult) -> str:
        import yaml
        return yaml.safe_dump(result.to_dict(), allow_unicode=True, sort_keys=False)

    def from_yaml(self, s: str) -> PluginResult:
        import yaml
        data = yaml.safe_load(s) or {}
        return PluginResult.from_dict(data)

    def dump_yaml(self, result: PluginResult, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(result), encoding="utf-8")

    def load_yaml(self, path: Union[str, Path]) -> PluginResult:
        return self.from_yaml(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # NPZ（对于重数组输出）
    # ------------------------------------------------------------------
    def dump_npz(
        self,
        result: PluginResult,
        path: Union[str, Path],
    ) -> None:
        """将 outputs 中的 ndarray 存储为 .npz，其他字段放 JSON"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = result.to_dict()
        arrays: Dict[str, np.ndarray] = {}
        _extract_arrays(d["outputs"], arrays, "outputs")
        _extract_arrays(d["inputs"], arrays, "inputs")
        # 去除 d 中的 base64，节省空间
        _strip_ndarray(d["outputs"])
        _strip_ndarray(d["inputs"])
        np.savez_compressed(p, **{"__meta__": json.dumps(d), **arrays})

    def load_npz(self, path: Union[str, Path]) -> PluginResult:
        data = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(data["__meta__"]))
        # 把 npz 中的数组填回
        for key in data.files:
            if key == "__meta__":
                continue
            arr = np.asarray(data[key])
            _inject_array(meta, key, arr)
        return PluginResult.from_dict(meta)


def _extract_arrays(obj: Any, arrays: Dict[str, np.ndarray], prefix: str) -> None:
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            name = f"{prefix}"
            arr = _restore_numpy(obj)
            arrays[name] = arr
            obj["__npz_ref__"] = name
            del obj["data"]
            return
        for k, v in list(obj.items()):
            _extract_arrays(v, arrays, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _extract_arrays(v, arrays, f"{prefix}[{i}]")


def _strip_ndarray(obj: Any) -> None:
    """将 __ndarray__ 但没 __npz_ref__ 的标记为占位（npz 里已经有）"""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            _strip_ndarray(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_ndarray(v)


def _inject_array(meta: Dict[str, Any], ref_key: str, arr: np.ndarray) -> None:
    """遍历 meta，把 __npz_ref__ == ref_key 的地方替换为已还原的 ndarray dict"""
    if isinstance(meta, dict):
        if meta.get("__npz_ref__") == ref_key:
            meta.update(_convert_numpy(arr))
            del meta["__npz_ref__"]
            return
        for v in meta.values():
            _inject_array(v if isinstance(v, (dict, list)) else {}, ref_key, arr)
    if isinstance(meta, list):
        for item in meta:
            _inject_array(item if isinstance(item, (dict, list)) else {}, ref_key, arr)


# ============================================================================
# 便捷函数
# ============================================================================

_DEFAULT_SERIALIZER = ResultSerializer()


def serialize_result(
    result: PluginResult,
    format: str = "json",
    path: Optional[Union[str, Path]] = None,
) -> str:
    """
    便捷函数：按格式序列化，同时写入文件（如果指定 path）。
    """
    fmt = format.lower()
    if fmt == "json":
        s = _DEFAULT_SERIALIZER.to_json(result)
        if path:
            _DEFAULT_SERIALIZER.dump_json(result, path)
    elif fmt == "yaml":
        s = _DEFAULT_SERIALIZER.to_yaml(result)
        if path:
            _DEFAULT_SERIALIZER.dump_yaml(result, path)
    else:
        raise ValueError(f"不支持的序列化格式: {format}")
    return s


def deserialize_result(
    source: Union[str, Path],
    format: Optional[str] = None,
) -> PluginResult:
    """从文件或字符串反序列化"""
    # 如果是路径且存在
    if isinstance(source, Path) or (
        isinstance(source, str) and len(source) < 4096 and Path(source).exists()
    ):
        p = Path(source)
        fmt = (format or p.suffix.lstrip(".")).lower()
        if fmt in ("json",):
            return _DEFAULT_SERIALIZER.load_json(p)
        if fmt in ("yaml", "yml"):
            return _DEFAULT_SERIALIZER.load_yaml(p)
        if fmt == "npz":
            return _DEFAULT_SERIALIZER.load_npz(p)
        raise ValueError(f"无法从文件后缀推断格式: {p.suffix}")
    # 否则按字符串解析
    fmt = (format or "json").lower()
    if fmt == "json":
        return _DEFAULT_SERIALIZER.from_json(source)
    if fmt in ("yaml", "yml"):
        return _DEFAULT_SERIALIZER.from_yaml(source)
    raise ValueError(f"不支持的反序列化格式: {format}")
