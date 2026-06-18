# -*- coding: utf-8 -*-
"""
版图布局管理模块

提供从 GDS/OASIS 文件或目录批量加载版图 cell，
构建版图队列，并为每个 cell 创建独立的优化任务。

主要功能：
1. GDSLoader: 单/多 GDS 文件解析，提取 cell 列表和掩模数据
2. LayoutCell: 单个版图 cell 的数据结构（名称、掩模、元数据）
3. LayoutLibrary: 版图库集合，管理多个 GDS 来源的 cell
4. LayoutQueue: 版图任务队列，支持优先级、去重、分批
5. LayoutManager: 统一入口，封装加载、过滤、排序、队列化流程
"""

import os
import logging
import hashlib
import time
import re
from pathlib import Path
from typing import (
    Optional, List, Dict, Any, Union, Tuple, Iterator,
    Callable, Set,
)
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

import numpy as np

from utils.data_io import load_gds_layer, load_gds_layer_by_cell_name

logger = logging.getLogger(__name__)

try:
    import gdstk
    HAS_GDSTK = True
except ImportError:
    HAS_GDSTK = False

try:
    import gdspy
    HAS_GDSPY = True
except ImportError:
    HAS_GDSPY = False


# ============================================================================
# 数据结构定义
# ============================================================================

class LayoutSourceType(Enum):
    """版图来源类型"""
    GDS_FILE = "gds_file"
    GDS_DIRECTORY = "gds_directory"
    IMAGE_FILE = "image_file"
    NUMPY_ARRAY = "numpy_array"
    SYNTHETIC = "synthetic"


@dataclass
class LayoutLoadOptions:
    """
    版图加载选项

    Attributes:
        layer: GDS 层号（加载 GDS 时必填）
        datatype: GDS 数据类型号，默认 0
        pixel_size: 栅格化像素尺寸 (nm)，默认 1.0
        target_size: 目标栅格尺寸 (height, width)，None 自动计算
        bounds: 版图范围 (xmin, ymin, xmax, ymax)，None 自动计算包围盒
        flatten_references: 是否展平 cell 引用，默认 True
        include_subcells: 是否包含子 cell，默认 False
        skip_empty_cells: 跳过无几何图形的 cell，默认 True
        cell_name_pattern: cell 名过滤正则（re.match），None 不过滤
        cell_name_blacklist: 排除的 cell 名集合
        cell_name_whitelist: 仅包含的 cell 名集合（优先级高于黑名单）
        max_cells_per_file: 每个 GDS 文件最多加载的 cell 数，0 不限制
        load_masks_on_init: 是否在加载时立即栅格化，False 则延迟加载
    """
    layer: Optional[int] = None
    datatype: int = 0
    pixel_size: float = 1.0
    target_size: Optional[Tuple[int, int]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    flatten_references: bool = True
    include_subcells: bool = False
    skip_empty_cells: bool = True
    cell_name_pattern: Optional[str] = None
    cell_name_blacklist: Set[str] = field(default_factory=set)
    cell_name_whitelist: Optional[Set[str]] = None
    max_cells_per_file: int = 0
    load_masks_on_init: bool = True


@dataclass
class LayoutCellMetadata:
    """版图 cell 元数据"""
    cell_name: str
    source_type: LayoutSourceType
    source_path: Optional[str] = None
    layer: Optional[int] = None
    datatype: Optional[int] = None
    pixel_size: float = 1.0
    bounds: Optional[Tuple[float, float, float, float]] = None
    polygon_count: int = 0
    reference_count: int = 0
    load_timestamp: float = 0.0
    checksum: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cell_name': self.cell_name,
            'source_type': self.source_type.value,
            'source_path': self.source_path,
            'layer': self.layer,
            'datatype': self.datatype,
            'pixel_size': self.pixel_size,
            'bounds': list(self.bounds) if self.bounds else None,
            'polygon_count': self.polygon_count,
            'reference_count': self.reference_count,
            'load_timestamp': self.load_timestamp,
            'checksum': self.checksum,
            'extra': self.extra,
        }


@dataclass
class LayoutCell:
    """
    单个版图 cell 数据结构

    Attributes:
        name: cell 唯一标识名（通常为 GDS cell name，可能带前缀去重）
        mask: 二值掩模数组 (H, W)，float64，值域 [0, 1]；延迟加载时为 None
        target: 目标图像（通常等于 mask，OPC 场景下可单独设置）
        metadata: 元数据
        priority: 调度优先级（0-100，越大越先执行）
        tags: 标签集合，用于过滤/分组
    """
    name: str
    metadata: LayoutCellMetadata
    mask: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None
    priority: int = 50
    tags: Set[str] = field(default_factory=set)

    @property
    def is_mask_loaded(self) -> bool:
        return self.mask is not None

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        return self.mask.shape if self.mask is not None else None

    @property
    def cell_name(self) -> str:
        return self.metadata.cell_name

    def ensure_mask_loaded(self,
                           loader: Optional['GDSLoader'] = None,
                           options: Optional[LayoutLoadOptions] = None) -> None:
        """
        确保掩模数据已加载（延迟加载入口）

        层次化加载的 cell 会在 metadata.extra['hierarchy'] 中存储
        is_leaf 标志，延迟加载时会根据该标志决定是否展平引用：
        - is_leaf=True → flatten_references=True，用于真实仿真
        - is_leaf=False → flatten_references=False，只取自身多边形
        """
        if self.mask is not None:
            return

        if self.metadata.source_type != LayoutSourceType.GDS_FILE:
            raise ValueError(
                f"无法为来源 {self.metadata.source_type} 执行延迟加载"
            )

        src = self.metadata.source_path
        if src is None:
            raise ValueError("source_path 为空，无法加载掩模")

        if loader is None:
            loader = GDSLoader()

        # 如果没有提供 options，或需要按层次信息覆盖 flatten_references
        if options is None:
            opts = LayoutLoadOptions(
                layer=self.metadata.layer,
                datatype=self.metadata.datatype or 0,
                pixel_size=self.metadata.pixel_size,
                target_size=None,
                bounds=self.metadata.bounds,
            )
        else:
            opts = options

        # 根据 hierarchy 信息自动决定 flatten_references
        hier_info = (self.metadata.extra or {}).get('hierarchy', {})
        if 'flatten_refs' in hier_info:
            opts.flatten_references = bool(hier_info['flatten_refs'])
        elif 'is_leaf' in hier_info:
            opts.flatten_references = bool(hier_info['is_leaf'])

        loaded = loader.load_cell_mask(src, self.metadata.cell_name, opts)
        if loaded is None:
            raise RuntimeError(f"延迟加载掩模失败: {self.name}")
        self.mask = loaded
        if self.target is None:
            self.target = loaded.copy()

    def summary(self) -> Dict[str, Any]:
        """生成 cell 摘要信息"""
        return {
            'name': self.name,
            'cell_name': self.cell_name,
            'source': self.metadata.source_path,
            'layer': self.metadata.layer,
            'datatype': self.metadata.datatype,
            'pixel_size': self.metadata.pixel_size,
            'shape': list(self.shape) if self.shape else None,
            'polygon_count': self.metadata.polygon_count,
            'priority': self.priority,
            'tags': list(self.tags),
            'is_mask_loaded': self.is_mask_loaded,
            'checksum': self.metadata.checksum,
        }


# ============================================================================
# GDS 加载器
# ============================================================================

class GDSLoader:
    """
    GDS/OASIS 文件加载器

    封装 gdstk/gdspy 的差异，提供统一接口：
    - list_cells: 列出文件中所有 cell 名
    - load_cell_mask: 加载单个 cell 的指定层为二值掩模
    - load_file: 加载整个文件的所有/指定 cell
    """

    SUPPORTED_SUFFIXES = {'.gds', '.gdsii', '.oas', '.oasis'}

    def __init__(self, backend: Optional[str] = None):
        """
        初始化加载器

        Args:
            backend: 'gdstk' | 'gdspy' | None（自动选择）
        """
        if backend is None:
            if HAS_GDSTK:
                backend = 'gdstk'
            elif HAS_GDSPY:
                backend = 'gdspy'
            else:
                backend = None
        self.backend = backend
        if self.backend == 'gdstk' and not HAS_GDSTK:
            raise ImportError("gdstk 未安装")
        if self.backend == 'gdspy' and not HAS_GDSPY:
            raise ImportError("gdspy 未安装")

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def is_supported_file(self, path: Union[str, Path]) -> bool:
        return Path(path).suffix.lower() in self.SUPPORTED_SUFFIXES

    def list_cells(self, filepath: Union[str, Path]) -> List[str]:
        """
        列出 GDS 文件中所有顶层 cell 名

        Args:
            filepath: GDS/OASIS 文件路径

        Returns:
            cell 名列表
        """
        if self.backend is None:
            raise ImportError("需要安装 gdstk 或 gdspy")

        filepath = str(filepath)
        names: List[str] = []

        if self.backend == 'gdstk':
            lib = gdstk.read_gds(filepath)
            top_cells = lib.top_level()
            names = [c.name for c in top_cells]
            if not names:
                names = [c.name for c in lib.cells]
        elif self.backend == 'gdspy':
            lib = gdspy.GdsLibrary(infile=filepath)
            top_cells = lib.top_level()
            names = list(top_cells.keys()) if isinstance(top_cells, dict) else [c.name for c in top_cells]
            if not names:
                names = list(lib.cells.keys())

        return names

    def list_all_cells(self, filepath: Union[str, Path]) -> List[str]:
        """列出文件中所有 cell 名（包括子 cell）"""
        if self.backend is None:
            raise ImportError("需要安装 gdstk 或 gdspy")
        filepath = str(filepath)
        if self.backend == 'gdstk':
            lib = gdstk.read_gds(filepath)
            return [c.name for c in lib.cells]
        else:
            lib = gdspy.GdsLibrary(infile=filepath)
            return list(lib.cells.keys())

    def load_cell_mask(self,
                       filepath: Union[str, Path],
                       cell_name: str,
                       options: LayoutLoadOptions) -> Optional[np.ndarray]:
        """
        加载指定 cell 的指定层为二值掩模

        使用 utils.data_io.load_gds_layer_by_cell_name，
        按 cell_name 精确提取该 cell 的多边形：
        - flatten_references=True: 展平该 cell 内部所有引用，用于叶节点
        - flatten_references=False: 只提取该 cell 自身直接多边形，用于复合节点

        Args:
            filepath: GDS 文件路径
            cell_name: cell 名（原始名，用于精确查找）
            options: 加载选项（含 flatten_references 控制）

        Returns:
            二值掩模 (H, W) float64；若 cell 为空且 skip_empty 则返回 None
        """
        filepath = str(filepath)

        if options.layer is None:
            raise ValueError("LayoutLoadOptions.layer 不能为空")

        try:
            mask = load_gds_layer_by_cell_name(
                filepath=filepath,
                cell_name=cell_name,
                layer=options.layer,
                datatype=options.datatype,
                pixel_size=options.pixel_size,
                target_size=options.target_size,
                bounds=options.bounds,
                flatten_references=options.flatten_references,
            )
        except ValueError as e:
            if options.skip_empty_cells and "无多边形" in str(e):
                return None
            raise

        if options.skip_empty_cells and np.sum(mask) < 1e-6:
            return None

        return mask

    def load_file(self,
                  filepath: Union[str, Path],
                  options: LayoutLoadOptions) -> List[LayoutCell]:
        """
        加载单个 GDS 文件，返回 LayoutCell 列表

        Args:
            filepath: GDS/OASIS 文件路径
            options: 加载选项

        Returns:
            LayoutCell 列表
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GDS 文件不存在: {filepath}")
        if not self.is_supported_file(filepath):
            raise ValueError(f"不支持的文件类型: {filepath.suffix}")
        if self.backend is None:
            raise ImportError("需要安装 gdstk 或 gdspy 以解析 cell 列表")

        cell_names = (
            self.list_all_cells(filepath)
            if options.include_subcells
            else self.list_cells(filepath)
        )

        import re
        pattern_re = re.compile(options.cell_name_pattern) if options.cell_name_pattern else None

        filtered = []
        for name in cell_names:
            if options.cell_name_whitelist is not None:
                if name not in options.cell_name_whitelist:
                    continue
            elif name in options.cell_name_blacklist:
                continue
            if pattern_re is not None and not pattern_re.match(name):
                continue
            filtered.append(name)

        if options.max_cells_per_file > 0:
            filtered = filtered[:options.max_cells_per_file]

        cells: List[LayoutCell] = []
        file_key = filepath.stem
        seen_names: Set[str] = set()

        for cell_name in filtered:
            unique_name = self._unique_name(seen_names, file_key, cell_name)
            seen_names.add(unique_name)

            meta = LayoutCellMetadata(
                cell_name=cell_name,
                source_type=LayoutSourceType.GDS_FILE,
                source_path=str(filepath.resolve()),
                layer=options.layer,
                datatype=options.datatype,
                pixel_size=options.pixel_size,
                bounds=options.bounds,
                load_timestamp=0.0,
            )
            self._fill_polygon_counts(filepath, cell_name, options, meta)

            cell = LayoutCell(
                name=unique_name,
                metadata=meta,
                priority=50,
                tags={'source:gds', f'file:{file_key}'},
            )

            if options.load_masks_on_init:
                import time
                t0 = time.time()
                mask = None
                try:
                    mask = self.load_cell_mask(filepath, cell_name, options)
                except ValueError as e:
                    # "无多边形" 等预期内的异常
                    if not options.skip_empty_cells and '无多边形' in str(e):
                        # 空 cell（只有引用没有自身多边形），后续处理
                        mask = None
                    else:
                        logger.warning(f"加载 cell {cell_name} 失败: {e}")
                        continue
                except Exception as e:
                    # 其他异常
                    logger.warning(f"加载 cell {cell_name} 失败: {e}")
                    continue
                meta.load_timestamp = t0
                if mask is None:
                    if options.skip_empty_cells:
                        logger.debug(f"跳过空 cell: {cell_name}")
                        continue
                    # 创建空掩模占位符，后续在层次化处理中根据 bounds 重新设置
                    mask = np.zeros((1, 1), dtype=np.float64)
                cell.mask = mask
                cell.target = mask.copy()
                meta.checksum = self._mask_checksum(mask)

            cells.append(cell)

        logger.info(f"从 {filepath.name} 加载了 {len(cells)} 个 cell")
        return cells

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _unique_name(seen: Set[str], file_key: str, cell_name: str) -> str:
        base = f"{file_key}::{cell_name}"
        if base not in seen:
            return base
        i = 2
        while f"{base}_{i}" in seen:
            i += 1
        return f"{base}_{i}"

    @staticmethod
    def _mask_checksum(mask: np.ndarray) -> str:
        arr = np.ascontiguousarray(mask.astype(np.float32))
        return hashlib.md5(arr.tobytes()).hexdigest()[:16]

    def _fill_polygon_counts(self,
                             filepath: Path,
                             cell_name: str,
                             options: LayoutLoadOptions,
                             meta: LayoutCellMetadata) -> None:
        """填充多边形/引用计数元数据"""
        try:
            fp = str(filepath)
            layer = options.layer
            datatype = options.datatype
            poly_cnt = 0
            ref_cnt = 0

            if self.backend == 'gdstk':
                lib = gdstk.read_gds(fp)
                cell_map = {c.name: c for c in lib.cells}
                cell = cell_map.get(cell_name)
                if cell is not None:
                    for p in cell.polygons:
                        if p.layer == layer and p.datatype == datatype:
                            poly_cnt += 1
                    ref_cnt = len(cell.references)
            elif self.backend == 'gdspy':
                lib = gdspy.GdsLibrary(infile=fp)
                cell = lib.cells.get(cell_name)
                if cell is not None:
                    for ps in cell.polygons:
                        if (getattr(ps, 'layers', None) and ps.layers[0] == layer
                                and getattr(ps, 'datatypes', None) and ps.datatypes[0] == datatype):
                            poly_cnt += len(ps.polygons)
                    ref_cnt = len(cell.references)

            meta.polygon_count = poly_cnt
            meta.reference_count = ref_cnt
        except Exception as e:
            logger.debug(f"统计多边形信息失败 {cell_name}: {e}")


# ============================================================================
# 版图库
# ============================================================================

class LayoutLibrary:
    """
    版图 cell 集合（版图库）

    管理从多个来源加载的 cell，支持：
    - 添加/删除/查询 cell
    - 按标签/名称/来源过滤
    - 去重（基于 checksum）
    - 导出为 CSV/JSON 清单
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._cells: Dict[str, LayoutCell] = {}
        self._checksum_index: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 基础 CRUD
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self) -> Iterator[LayoutCell]:
        return iter(self._cells.values())

    def __getitem__(self, key: str) -> LayoutCell:
        return self._cells[key]

    def __contains__(self, key: str) -> bool:
        return key in self._cells

    def add(self, cell: LayoutCell, dedup: bool = True) -> bool:
        """
        添加 cell 到库

        Args:
            cell: 要添加的 cell
            dedup: 是否按 checksum 去重

        Returns:
            True 表示成功添加，False 表示被去重
        """
        if dedup and cell.is_mask_loaded:
            cs = cell.metadata.checksum
            if cs and cs in self._checksum_index:
                existing_name = self._checksum_index[cs]
                logger.debug(
                    f"cell {cell.name} 与 {existing_name} 内容重复，跳过"
                )
                return False
            if cs:
                self._checksum_index[cs] = cell.name

        self._cells[cell.name] = cell
        return True

    def add_many(self, cells: List[LayoutCell], dedup: bool = True) -> int:
        added = 0
        for c in cells:
            if self.add(c, dedup=dedup):
                added += 1
        return added

    def remove(self, name: str) -> bool:
        cell = self._cells.pop(name, None)
        if cell is None:
            return False
        if cell.metadata.checksum:
            self._checksum_index.pop(cell.metadata.checksum, None)
        return True

    def get(self, name: str, default: Optional[LayoutCell] = None) -> Optional[LayoutCell]:
        return self._cells.get(name, default)

    def names(self) -> List[str]:
        return list(self._cells.keys())

    def cells(self) -> List[LayoutCell]:
        return list(self._cells.values())

    # ------------------------------------------------------------------
    # 过滤与查询
    # ------------------------------------------------------------------

    def filter(self,
               name_contains: Optional[str] = None,
               name_pattern: Optional[str] = None,
               tags: Optional[Set[str]] = None,
               tags_all: Optional[Set[str]] = None,
               source_path: Optional[str] = None,
               min_polygons: int = 0,
               mask_loaded_only: bool = False,
               ) -> 'LayoutLibrary':
        """
        按条件过滤，返回新的 LayoutLibrary

        Args:
            name_contains: 名称包含子串
            name_pattern: 正则匹配名称
            tags: 命中任一标签即保留
            tags_all: 必须包含所有标签
            source_path: 来源路径包含此字符串
            min_polygons: 最少多边形数
            mask_loaded_only: 仅保留已加载掩模的 cell

        Returns:
            新的 LayoutLibrary 实例
        """
        import re
        pat = re.compile(name_pattern) if name_pattern else None

        result = LayoutLibrary(name=f"{self.name}(filtered)")

        for cell in self:
            if name_contains and name_contains not in cell.name:
                continue
            if pat and not pat.search(cell.name):
                continue
            if tags and not (cell.tags & tags):
                continue
            if tags_all and not (cell.tags >= tags_all):
                continue
            if source_path and (cell.metadata.source_path is None
                                or source_path not in cell.metadata.source_path):
                continue
            if cell.metadata.polygon_count < min_polygons:
                continue
            if mask_loaded_only and not cell.is_mask_loaded:
                continue
            result.add(cell, dedup=False)

        return result

    def group_by(self, key_fn: Callable[[LayoutCell], str]) -> Dict[str, 'LayoutLibrary']:
        """按键函数分组"""
        groups: Dict[str, LayoutLibrary] = {}
        for cell in self:
            key = key_fn(cell)
            if key not in groups:
                groups[key] = LayoutLibrary(name=f"{self.name}:{key}")
            groups[key].add(cell, dedup=False)
        return groups

    # ------------------------------------------------------------------
    # 统计与导出
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        total = len(self._cells)
        loaded = sum(1 for c in self if c.is_mask_loaded)
        total_polys = sum(c.metadata.polygon_count for c in self)
        shapes = [c.shape for c in self if c.shape]
        avg_h = int(np.mean([s[0] for s in shapes])) if shapes else 0
        avg_w = int(np.mean([s[1] for s in shapes])) if shapes else 0
        return {
            'library_name': self.name,
            'total_cells': total,
            'mask_loaded': loaded,
            'total_polygons': total_polys,
            'avg_shape': [avg_h, avg_w] if shapes else None,
            'unique_masks': len(self._checksum_index),
        }

    def to_records(self) -> List[Dict[str, Any]]:
        return [c.summary() for c in self]

    def to_csv(self, filepath: Union[str, Path]) -> Path:
        """导出 cell 清单为 CSV"""
        import csv
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        records = self.to_records()
        if not records:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('')
            return filepath

        fieldnames = list(records[0].keys())
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({k: ';'.join(str(x) for x in v) if isinstance(v, list) else v
                                 for k, v in r.items()})
        return filepath

    def to_json(self, filepath: Union[str, Path]) -> Path:
        """导出 cell 清单为 JSON"""
        import json
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'summary': self.summary(), 'cells': self.to_records()},
                      f, indent=2, ensure_ascii=False)
        return filepath


# ============================================================================
# 版图任务队列
# ============================================================================

class LayoutQueue:
    """
    版图优化任务队列

    特性：
    - 基于优先级 + FIFO 的出队顺序（高优先级先出，同优先级先进先出）
    - 支持任务状态标记（待入队/排队/运行中/完成/失败）
    - 支持 peek、requeue、批量操作
    """

    class Status(Enum):
        PENDING = "pending"       # 已入队，等待调度
        RUNNING = "running"       # 正在执行
        DONE = "done"             # 执行成功
        FAILED = "failed"         # 执行失败
        CANCELLED = "cancelled"   # 已取消

    @dataclass
    class QueueEntry:
        cell: LayoutCell
        status: 'LayoutQueue.Status'
        retries: int = 0
        max_retries: int = 3
        last_error: Optional[str] = None
        worker_id: Optional[str] = None
        submitted_at: float = 0.0
        started_at: Optional[float] = None
        finished_at: Optional[float] = None
        progress: float = 0.0

        def to_dict(self) -> Dict[str, Any]:
            return {
                'cell_name': self.cell.name,
                'status': self.status.value,
                'retries': self.retries,
                'max_retries': self.max_retries,
                'last_error': self.last_error,
                'worker_id': self.worker_id,
                'submitted_at': self.submitted_at,
                'started_at': self.started_at,
                'finished_at': self.finished_at,
                'progress': self.progress,
            }

    def __init__(self):
        self._entries: Dict[str, 'LayoutQueue.QueueEntry'] = {}
        self._order: deque = deque()

    # ------------------------------------------------------------------
    # 队列操作
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, cell_name: str) -> bool:
        return cell_name in self._entries

    def add(self,
            cell: LayoutCell,
            max_retries: int = 3,
            priority: Optional[int] = None) -> None:
        """添加 cell 到队列尾部"""
        if cell.name in self._entries:
            logger.debug(f"cell {cell.name} 已在队列中，跳过")
            return
        import time
        if priority is not None:
            cell.priority = priority
        entry = LayoutQueue.QueueEntry(
            cell=cell,
            status=LayoutQueue.Status.PENDING,
            max_retries=max_retries,
            submitted_at=time.time(),
        )
        self._entries[cell.name] = entry
        self._order.append(cell.name)
        self._reorder_by_priority()

    def add_many(self,
                 cells: List[LayoutCell],
                 max_retries: int = 3,
                 priority: Optional[int] = None) -> None:
        for c in cells:
            self.add(c, max_retries=max_retries, priority=priority)

    def add_from_library(self,
                         lib: LayoutLibrary,
                         max_retries: int = 3,
                         priority: Optional[int] = None) -> None:
        self.add_many(lib.cells(), max_retries=max_retries, priority=priority)

    def pop_next(self, worker_id: Optional[str] = None) -> Optional['LayoutQueue.QueueEntry']:
        """
        取出下一个 PENDING 任务并标记为 RUNNING

        Returns:
            QueueEntry 或 None（队列为空或无 PENDING 任务）
        """
        import time
        while self._order:
            name = self._order.popleft()
            entry = self._entries.get(name)
            if entry is None:
                continue
            if entry.status != LayoutQueue.Status.PENDING:
                continue
            entry.status = LayoutQueue.Status.RUNNING
            entry.worker_id = worker_id
            entry.started_at = time.time()
            return entry
        return None

    def peek(self) -> Optional['LayoutQueue.QueueEntry']:
        """查看下一个 PENDING 任务，不移除"""
        for name in self._order:
            entry = self._entries.get(name)
            if entry and entry.status == LayoutQueue.Status.PENDING:
                return entry
        return None

    def mark_done(self, cell_name: str, progress: float = 1.0) -> None:
        import time
        entry = self._entries.get(cell_name)
        if entry is None:
            return
        entry.status = LayoutQueue.Status.DONE
        entry.progress = progress
        entry.finished_at = time.time()

    def mark_failed(self, cell_name: str, error: str, retry: bool = True) -> None:
        """
        标记失败，根据重试策略决定是否重新入队

        Args:
            cell_name: cell 名
            error: 错误信息
            retry: 是否允许重试（max_retries 内）
        """
        import time
        entry = self._entries.get(cell_name)
        if entry is None:
            return
        entry.last_error = str(error)
        entry.progress = 0.0
        entry.worker_id = None
        entry.started_at = None
        entry.finished_at = None

        if retry and entry.retries < entry.max_retries:
            entry.retries += 1
            entry.status = LayoutQueue.Status.PENDING
            entry.submitted_at = time.time()
            self._order.append(cell_name)
            self._reorder_by_priority()
            logger.warning(
                f"cell {cell_name} 失败（第{entry.retries}次），已重新入队: {error[:100]}"
            )
        else:
            entry.status = LayoutQueue.Status.FAILED
            entry.finished_at = time.time()
            logger.error(
                f"cell {cell_name} 最终失败（重试{entry.retries}次）: {error[:200]}"
            )

    def mark_cancelled(self, cell_name: str) -> None:
        entry = self._entries.get(cell_name)
        if entry is None:
            return
        import time
        entry.status = LayoutQueue.Status.CANCELLED
        entry.finished_at = time.time()

    def update_progress(self, cell_name: str, progress: float) -> None:
        entry = self._entries.get(cell_name)
        if entry is not None:
            entry.progress = max(0.0, min(1.0, progress))

    def retry_all_failed(self) -> int:
        """将所有 FAILED 任务重置为 PENDING 并重新入队（不增加重试计数）"""
        import time
        count = 0
        for name, entry in self._entries.items():
            if entry.status == LayoutQueue.Status.FAILED:
                entry.status = LayoutQueue.Status.PENDING
                entry.last_error = None
                entry.worker_id = None
                entry.started_at = None
                entry.finished_at = None
                entry.submitted_at = time.time()
                self._order.append(name)
                count += 1
        self._reorder_by_priority()
        if count:
            logger.info(f"已将 {count} 个失败任务重新入队")
        return count

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_entry(self, cell_name: str) -> Optional['LayoutQueue.QueueEntry']:
        return self._entries.get(cell_name)

    def entries(self) -> List['LayoutQueue.QueueEntry']:
        return [self._entries[n] for n in self._order if n in self._entries]

    def status_counts(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in LayoutQueue.Status}
        for e in self._entries.values():
            counts[e.status.value] += 1
        return counts

    def pending_count(self) -> int:
        return self.status_counts().get(LayoutQueue.Status.PENDING.value, 0)

    def running_count(self) -> int:
        return self.status_counts().get(LayoutQueue.Status.RUNNING.value, 0)

    def done_count(self) -> int:
        return self.status_counts().get(LayoutQueue.Status.DONE.value, 0)

    def failed_count(self) -> int:
        return self.status_counts().get(LayoutQueue.Status.FAILED.value, 0)

    def all_done(self) -> bool:
        return self.pending_count() == 0 and self.running_count() == 0

    def to_records(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries()]

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _reorder_by_priority(self) -> None:
        """按优先级降序重排 _order（稳定排序，保持同优先级先进先出）"""
        indexed = list(self._order)
        indexed.sort(
            key=lambda n: (
                -self._entries[n].cell.priority,
                self._entries[n].submitted_at,
            )
        )
        self._order = deque(indexed)


# ============================================================================
# LayoutManager: 统一入口
# ============================================================================

class LayoutManager:
    """
    版图管理统一入口

    封装从加载→入库→建队的完整流程：
        mgr = LayoutManager()
        lib = mgr.load_directory("/path/to/gds/dir", options=LayoutLoadOptions(layer=0))
        queue = mgr.build_queue(lib, priority_by_size=True)

    或直接：
        queue = mgr.load_and_queue("/path/to/gds/dir", layer=0)
    """

    def __init__(self, loader: Optional[GDSLoader] = None):
        self.loader = loader or GDSLoader()

    # ------------------------------------------------------------------
    # 加载入口
    # ------------------------------------------------------------------

    def load_gds_file(self,
                      filepath: Union[str, Path],
                      options: Optional[LayoutLoadOptions] = None,
                      library_name: Optional[str] = None) -> LayoutLibrary:
        """从单个 GDS 文件加载"""
        opts = options or LayoutLoadOptions()
        if library_name is None:
            library_name = Path(filepath).stem
        lib = LayoutLibrary(name=library_name)
        cells = self.loader.load_file(filepath, opts)
        lib.add_many(cells, dedup=True)
        return lib

    def load_directory(self,
                       dirpath: Union[str, Path],
                       options: Optional[LayoutLoadOptions] = None,
                       library_name: str = "directory_library",
                       recursive: bool = True,
                       suffixes: Optional[Set[str]] = None) -> LayoutLibrary:
        """
        从目录递归加载所有 GDS/OASIS 文件

        Args:
            dirpath: 目录路径
            options: 加载选项
            library_name: 版图库名
            recursive: 是否递归子目录
            suffixes: 文件后缀集合，None 则使用默认支持列表

        Returns:
            LayoutLibrary
        """
        dirpath = Path(dirpath)
        if not dirpath.exists():
            raise FileNotFoundError(f"目录不存在: {dirpath}")
        if not dirpath.is_dir():
            raise NotADirectoryError(f"不是目录: {dirpath}")

        suffixes = suffixes or GDSLoader.SUPPORTED_SUFFIXES
        glob_pattern = "**/*" if recursive else "*"

        files: List[Path] = []
        for p in dirpath.glob(glob_pattern):
            if p.is_file() and p.suffix.lower() in suffixes:
                files.append(p)

        files.sort()

        lib = LayoutLibrary(name=library_name)
        if not files:
            logger.warning(f"目录 {dirpath} 中未找到 GDS/OASIS 文件")
            return lib

        logger.info(f"在 {dirpath} 中发现 {len(files)} 个版图文件")

        opts = options or LayoutLoadOptions()
        for fp in files:
            try:
                cells = self.loader.load_file(fp, opts)
                added = lib.add_many(cells, dedup=True)
                logger.info(f"  {fp.name}: 新增 {added}/{len(cells)} 个 cell")
            except Exception as e:
                logger.error(f"加载文件 {fp} 失败: {e}")

        return lib

    def load_file_list(self,
                       filepaths: List[Union[str, Path]],
                       options: Optional[LayoutLoadOptions] = None,
                       library_name: str = "filelist_library") -> LayoutLibrary:
        """从文件列表加载"""
        lib = LayoutLibrary(name=library_name)
        opts = options or LayoutLoadOptions()
        for fp in filepaths:
            try:
                cells = self.loader.load_file(Path(fp), opts)
                lib.add_many(cells, dedup=True)
            except Exception as e:
                logger.error(f"加载文件 {fp} 失败: {e}")
        return lib

    def add_synthetic_cell(self,
                           lib: LayoutLibrary,
                           mask: np.ndarray,
                           name: str,
                           target: Optional[np.ndarray] = None,
                           priority: int = 50) -> str:
        """
        向库中添加合成/自定义掩模 cell

        Args:
            lib: 目标版图库
            mask: 二值掩模
            name: cell 名
            target: 目标图像，None 则使用 mask
            priority: 优先级

        Returns:
            实际入库的 cell 唯一名
        """
        import time
        meta = LayoutCellMetadata(
            cell_name=name,
            source_type=LayoutSourceType.SYNTHETIC,
            source_path=None,
            pixel_size=1.0,
            load_timestamp=time.time(),
            polygon_count=-1,
        )
        seen = set(lib.names())
        unique = GDSLoader._unique_name(seen, "synth", name)
        cell = LayoutCell(
            name=unique,
            metadata=meta,
            mask=mask.astype(np.float64),
            target=(target.astype(np.float64) if target is not None else mask.astype(np.float64).copy()),
            priority=priority,
            tags={'source:synthetic'},
        )
        meta.checksum = GDSLoader._mask_checksum(cell.mask)
        lib.add(cell, dedup=True)
        return unique

    # ------------------------------------------------------------------
    # 构建队列
    # ------------------------------------------------------------------

    def build_queue(self,
                    lib: LayoutLibrary,
                    priority_by_size: bool = False,
                    priority_by_polygons: bool = False,
                    size_priority_reverse: bool = True,
                    max_retries: int = 3,
                    require_mask_loaded: bool = True) -> LayoutQueue:
        """
        从版图库构建任务队列

        Args:
            lib: 版图库
            priority_by_size: 是否按掩模尺寸分配优先级（越大越高/越低）
            priority_by_polygons: 是否按多边形数分配优先级
            size_priority_reverse: True=越大越高，False=越小越高
            max_retries: 每个任务最大重试次数
            require_mask_loaded: 仅包含已加载掩模的 cell

        Returns:
            LayoutQueue
        """
        q = LayoutQueue()

        cells = [c for c in lib if (not require_mask_loaded) or c.is_mask_loaded]

        if priority_by_size or priority_by_polygons:
            sizes = []
            polys = []
            for c in cells:
                if c.shape:
                    sizes.append(c.shape[0] * c.shape[1])
                else:
                    sizes.append(0)
                polys.append(max(0, c.metadata.polygon_count))

            if sizes:
                max_s = max(sizes) or 1
            else:
                max_s = 1
            if polys:
                max_p = max(polys) or 1
            else:
                max_p = 1

            for c, s, p in zip(cells, sizes, polys):
                score = 0.0
                if priority_by_size:
                    ratio = s / max_s
                    score += ratio if size_priority_reverse else (1 - ratio)
                if priority_by_polygons:
                    score += 0.5 * (p / max_p)
                if priority_by_size or priority_by_polygons:
                    c.priority = max(0, min(100, int(score * 100)))

        q.add_from_library(lib, max_retries=max_retries)

        logger.info(
            f"构建队列: {len(q)} 个任务 "
            f"(库大小 {len(lib)}, 已加载掩模 {sum(1 for c in lib if c.is_mask_loaded)})"
        )
        return q

    # ------------------------------------------------------------------
    # 快捷组合
    # ------------------------------------------------------------------

    def load_and_queue(self,
                       source: Union[str, Path, List[Union[str, Path]]],
                       layer: int,
                       **load_kwargs) -> Tuple[LayoutLibrary, LayoutQueue]:
        """
        加载并构建队列的快捷方法

        Args:
            source: 目录路径 / GDS 文件路径 / 路径列表
            layer: GDS 层号（必填）
            **load_kwargs: 其他 LayoutLoadOptions 参数

        Returns:
            (LayoutLibrary, LayoutQueue)
        """
        opts_kwargs = {'layer': layer, **load_kwargs}
        opts = LayoutLoadOptions(**{
            k: v for k, v in opts_kwargs.items()
            if k in LayoutLoadOptions.__dataclass_fields__
        })

        if isinstance(source, list):
            lib = self.load_file_list(source, options=opts)
        elif isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                lib = self.load_directory(p, options=opts)
            elif p.is_file():
                lib = self.load_gds_file(p, options=opts)
            else:
                raise FileNotFoundError(f"源路径不存在: {source}")
        else:
            raise TypeError(f"不支持的 source 类型: {type(source)}")

        q_kwargs = {k: v for k, v in load_kwargs.items()
                    if k in ('priority_by_size', 'priority_by_polygons',
                             'size_priority_reverse', 'max_retries',
                             'require_mask_loaded')}
        queue = self.build_queue(lib, **q_kwargs)
        return lib, queue


# ============================================================================
# 层次化数据结构
# ============================================================================


@dataclass
class CellInstance:
    """
    单个 cell 引用实例

    表示父 cell 中对子 cell 的一次引用（SRef 或 ARef 中的一个实例）。
    包含完整的变换信息，用于从子 cell 结果重建父 cell。

    Attributes:
        child_cell_name: 被引用的子 cell 名（GDS 原始名）
        origin: 引用原点 (x, y)，单位与 GDS 一致
        rotation: 旋转角度（度）
        magnification: 缩放比例
        x_reflection: 是否 X 轴镜像
        transform: 3x3 齐次变换矩阵（累积了以上所有变换）
        is_array_member: 是否为阵列引用的成员
        array_index: 若为阵列成员则为 (row, col)，否则 None
    """
    child_cell_name: str
    origin: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    magnification: float = 1.0
    x_reflection: bool = False
    transform: Optional[np.ndarray] = None
    is_array_member: bool = False
    array_index: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.transform is None:
            self.transform = self._build_transform()

    def _build_transform(self) -> np.ndarray:
        """构建 3x3 齐次变换矩阵"""
        mat = np.eye(3, dtype=np.float64)
        ox, oy = self.origin
        mat[0, 2] = ox
        mat[1, 2] = oy
        if self.rotation != 0.0:
            rad = np.radians(self.rotation)
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            rot = np.eye(3, dtype=np.float64)
            rot[0, 0] = cos_r
            rot[0, 1] = -sin_r
            rot[1, 0] = sin_r
            rot[1, 1] = cos_r
            mat = rot @ mat
        if self.magnification != 1.0:
            scale = np.eye(3, dtype=np.float64)
            scale[0, 0] = self.magnification
            scale[1, 1] = self.magnification
            mat = scale @ mat
        if self.x_reflection:
            mirror = np.eye(3, dtype=np.float64)
            mirror[1, 1] = -1.0
            mat = mirror @ mat
        return mat

    def to_dict(self) -> Dict[str, Any]:
        return {
            'child_cell_name': self.child_cell_name,
            'origin': list(self.origin),
            'rotation': self.rotation,
            'magnification': self.magnification,
            'x_reflection': self.x_reflection,
            'is_array_member': self.is_array_member,
            'array_index': list(self.array_index) if self.array_index else None,
        }


@dataclass
class ArrayReferenceInfo:
    """
    阵列引用（ARef）信息

    Attributes:
        child_cell_name: 被引用的子 cell 名
        rows: 阵列行数
        cols: 阵列列数
        spacing_x: 列间距
        spacing_y: 行间距
        base_origin: 阵列第一个实例的原点
        rotation: 整体旋转角度
        magnification: 整体缩放
        x_reflection: 是否整体 X 轴镜像
    """
    child_cell_name: str
    rows: int = 1
    cols: int = 1
    spacing_x: float = 0.0
    spacing_y: float = 0.0
    base_origin: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    magnification: float = 1.0
    x_reflection: bool = False

    @property
    def total_instances(self) -> int:
        return self.rows * self.cols

    def expand_instances(self) -> List[CellInstance]:
        """将阵列展开为单个 CellInstance 列表"""
        instances = []
        for row in range(self.rows):
            for col in range(self.cols):
                ox = self.base_origin[0] + col * self.spacing_x
                oy = self.base_origin[1] + row * self.spacing_y
                instances.append(CellInstance(
                    child_cell_name=self.child_cell_name,
                    origin=(ox, oy),
                    rotation=self.rotation,
                    magnification=self.magnification,
                    x_reflection=self.x_reflection,
                    is_array_member=True,
                    array_index=(row, col),
                ))
        return instances

    def to_dict(self) -> Dict[str, Any]:
        return {
            'child_cell_name': self.child_cell_name,
            'rows': self.rows,
            'cols': self.cols,
            'spacing': [self.spacing_x, self.spacing_y],
            'base_origin': list(self.base_origin),
            'rotation': self.rotation,
            'magnification': self.magnification,
            'x_reflection': self.x_reflection,
            'total_instances': self.total_instances,
        }


@dataclass
class HierarchyNode:
    """
    层次树中的单个 cell 节点

    Attributes:
        cell_name: cell 名（GDS 原始名）
        depth: 在层次树中的深度（顶层 = 0）
        is_leaf: 是否为叶节点（无任何子引用）
        is_top: 是否为顶层 cell（无父引用）
        children: 子 cell 引用列表（直接子节点）
        parents: 父 cell 名列表
        single_refs: 非阵列的单引用列表
        array_refs: 阵列引用列表
        total_child_instances: 所有子实例总数（含阵列展开）
        polygon_count: 本 cell 自身的多边形数（不含子引用）
        bounds: 本 cell 自身的包围盒（不含子引用）
    """
    cell_name: str
    depth: int = 0
    is_leaf: bool = True
    is_top: bool = True
    children: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    single_refs: List[CellInstance] = field(default_factory=list)
    array_refs: List[ArrayReferenceInfo] = field(default_factory=list)
    total_child_instances: int = 0
    polygon_count: int = 0
    bounds: Optional[Tuple[float, float, float, float]] = None

    @property
    def all_child_instances(self) -> List[CellInstance]:
        """获取所有子引用实例（包括阵列展开）"""
        instances = list(self.single_refs)
        for aref in self.array_refs:
            instances.extend(aref.expand_instances())
        return instances

    @property
    def unique_children(self) -> Set[str]:
        """本 cell 直接引用的唯一子 cell 集合"""
        children = {r.child_cell_name for r in self.single_refs}
        for aref in self.array_refs:
            children.add(aref.child_cell_name)
        return children

    def summary(self) -> Dict[str, Any]:
        return {
            'cell_name': self.cell_name,
            'depth': self.depth,
            'is_leaf': self.is_leaf,
            'is_top': self.is_top,
            'parent_count': len(self.parents),
            'unique_children': sorted(self.unique_children),
            'single_ref_count': len(self.single_refs),
            'array_ref_count': len(self.array_refs),
            'total_child_instances': self.total_child_instances,
            'polygon_count': self.polygon_count,
        }


@dataclass
class HierarchyGraph:
    """
    GDS 完整层次图

    提供拓扑排序、层次遍历、重复检测等功能。

    Attributes:
        nodes: cell_name -> HierarchyNode
        top_cells: 顶层 cell 名列表
        leaf_cells: 叶节点 cell 名列表
    """
    nodes: Dict[str, HierarchyNode] = field(default_factory=dict)
    top_cells: List[str] = field(default_factory=list)
    leaf_cells: List[str] = field(default_factory=list)

    def __contains__(self, cell_name: str) -> bool:
        return cell_name in self.nodes

    def __getitem__(self, cell_name: str) -> HierarchyNode:
        return self.nodes[cell_name]

    def __len__(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------
    # 拓扑排序
    # ------------------------------------------------------------------

    def topological_order(self, bottom_up: bool = True) -> List[str]:
        """
        拓扑排序

        Args:
            bottom_up: True=叶节点优先（自底向上，先处理子 cell）
                       False=顶层优先（自顶向下）

        Returns:
            cell 名的排序列表
        """
        in_degree = {name: len(node.parents) for name, node in self.nodes.items()}
        queue: deque = deque()
        for name, deg in in_degree.items():
            if deg == 0:
                queue.append(name)

        top_down: List[str] = []
        while queue:
            name = queue.popleft()
            top_down.append(name)
            for child in self.nodes[name].unique_children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(top_down) != len(self.nodes):
            remaining = set(self.nodes.keys()) - set(top_down)
            logger.warning(
                f"层次图存在循环引用: {remaining}. 忽略环。"
            )
            for name in remaining:
                top_down.append(name)

        return list(reversed(top_down)) if bottom_up else top_down

    # ------------------------------------------------------------------
    # 层次遍历
    # ------------------------------------------------------------------

    def bfs_from_top(self) -> Iterator[Tuple[int, str]]:
        """从顶层 cell 开始 BFS，产生 (depth, cell_name)"""
        visited: Set[str] = set()
        queue: deque = deque()
        for t in self.top_cells:
            if t in self.nodes:
                queue.append((0, t))
                visited.add(t)
        while queue:
            depth, name = queue.popleft()
            yield depth, name
            for child in self.nodes[name].unique_children:
                if child not in visited and child in self.nodes:
                    visited.add(child)
                    queue.append((depth + 1, child))

    def compute_depths(self) -> None:
        """
        基于最长路径计算每个节点的深度（顶层为 0）

        使用拓扑排序进行动态规划，确保获得从顶层到该节点的最大深度。
        """
        topo = self.topological_order(bottom_up=False)

        for name in topo:
            self.nodes[name].depth = 0

        for name in topo:
            current_depth = self.nodes[name].depth
            for child in self.nodes[name].unique_children:
                if child in self.nodes:
                    if self.nodes[child].depth < current_depth + 1:
                        self.nodes[child].depth = current_depth + 1

    # ------------------------------------------------------------------
    # 重复检测
    # ------------------------------------------------------------------

    def find_duplicate_leaves(self) -> Dict[str, List[str]]:
        """
        找出重复的叶节点（被多次引用的相同几何）

        Returns:
            child_cell_name -> 引用它的父 cell 名列表
        """
        result: Dict[str, List[str]] = {}
        for name in self.leaf_cells:
            if name in self.nodes:
                parents = self.nodes[name].parents
                if len(parents) > 1:
                    result[name] = list(parents)
        return result

    def find_array_cells(self) -> Dict[str, int]:
        """
        找出被阵列引用的 cell 及其总实例数

        Returns:
            cell_name -> 总阵列实例数
        """
        counts: Dict[str, int] = defaultdict(int)
        for name, node in self.nodes.items():
            for aref in node.array_refs:
                counts[aref.child_cell_name] += aref.total_instances
        return dict(counts)

    def compute_reference_counts(self) -> Dict[str, int]:
        """
        统计每个 cell 被引用的总次数（含阵列展开）

        Returns:
            cell_name -> 总引用次数
        """
        counts: Dict[str, int] = defaultdict(int)
        for name, node in self.nodes.items():
            counts[name] += len(node.single_refs)
            for aref in node.array_refs:
                counts[aref.child_cell_name] += aref.total_instances
        return dict(counts)

    def compute_bounds(self) -> None:
        """
        自底向上计算每个 node 的完整包围盒（包括所有子引用展开后的 bounds）

        对于有自身多边形的 cell，使用已有的 bounds；
        对于只有引用没有自身多边形的 cell，通过子引用的 bounds + 变换矩阵计算。

        结果直接写入每个 node.bounds 属性。
        """
        # 自底向上：先处理叶节点，再处理父节点
        order = self.topological_order(bottom_up=True)  # 叶节点在前，父节点在后

        for name in order:
            node = self.nodes[name]

            # 收集所有 bounds：自身多边形的 bounds + 所有子引用展开后的 bounds
            all_bounds = []
            if node.bounds is not None:
                all_bounds.append(node.bounds)

            # 处理单引用
            for ref in node.single_refs:
                child_node = self.nodes.get(ref.child_cell_name)
                if child_node is None or child_node.bounds is None:
                    continue
                child_xmin, child_ymin, child_xmax, child_ymax = child_node.bounds
                # 4 个角点
                corners = np.array([
                    [child_xmin, child_ymin, 1.0],
                    [child_xmax, child_ymin, 1.0],
                    [child_xmax, child_ymax, 1.0],
                    [child_xmin, child_ymax, 1.0],
                ])
                transformed = (ref.transform @ corners.T).T[:, :2]
                all_bounds.append((
                    float(transformed[:, 0].min()),
                    float(transformed[:, 1].min()),
                    float(transformed[:, 0].max()),
                    float(transformed[:, 1].max()),
                ))

            # 处理阵列引用
            for aref in node.array_refs:
                child_node = self.nodes.get(aref.child_cell_name)
                if child_node is None or child_node.bounds is None:
                    continue
                child_xmin, child_ymin, child_xmax, child_ymax = child_node.bounds
                # 展开所有阵列实例
                for inst in aref.expand_instances():
                    corners = np.array([
                        [child_xmin, child_ymin, 1.0],
                        [child_xmax, child_ymin, 1.0],
                        [child_xmax, child_ymax, 1.0],
                        [child_xmin, child_ymax, 1.0],
                    ])
                    transformed = (inst.transform @ corners.T).T[:, :2]
                    all_bounds.append((
                        float(transformed[:, 0].min()),
                        float(transformed[:, 1].min()),
                        float(transformed[:, 0].max()),
                        float(transformed[:, 1].max()),
                    ))

            if all_bounds:
                all_xmin = min(b[0] for b in all_bounds)
                all_ymin = min(b[1] for b in all_bounds)
                all_xmax = max(b[2] for b in all_bounds)
                all_ymax = max(b[3] for b in all_bounds)
                node.bounds = (all_xmin, all_ymin, all_xmax, all_ymax)

    # ------------------------------------------------------------------
    # 统计与导出
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        ref_counts = self.compute_reference_counts()
        array_cells = self.find_array_cells()
        dup_leaves = self.find_duplicate_leaves()
        return {
            'total_cells': len(self.nodes),
            'top_cells': self.top_cells,
            'leaf_cells_count': len(self.leaf_cells),
            'array_cells_count': len(array_cells),
            'array_total_instances': sum(array_cells.values()),
            'duplicate_leaves_count': len(dup_leaves),
            'max_depth': max((n.depth for n in self.nodes.values()), default=0),
            'most_referenced': sorted(
                ref_counts.items(), key=lambda x: -x[1]
            )[:10],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': self.summary(),
            'top_cells': self.top_cells,
            'leaf_cells': self.leaf_cells,
            'nodes': {name: node.summary() for name, node in self.nodes.items()},
        }


# ============================================================================
# 层次化分析器
# ============================================================================


class LayoutHierarchyAnalyzer:
    """
    GDS 层次结构分析器

    解析 GDS/OASIS 文件中的 cell 引用关系，构建 HierarchyGraph。
    识别：
    - 父子 cell 关系
    - 单引用 (SRef) 与阵列引用 (ARef)
    - 顶层 / 叶节点 cell
    - 重复引用模式

    使用示例:
        analyzer = LayoutHierarchyAnalyzer()
        graph = analyzer.analyze_file("chip.gds", layer=0)
        order = graph.topological_order(bottom_up=True)  # 叶子优先
        array_cells = graph.find_array_cells()
    """

    def __init__(self, backend: Optional[str] = None):
        if backend is None:
            if HAS_GDSTK:
                backend = 'gdstk'
            elif HAS_GDSPY:
                backend = 'gdspy'
            else:
                backend = None
        self.backend = backend
        if self.backend == 'gdstk' and not HAS_GDSTK:
            raise ImportError("gdstk 未安装")
        if self.backend == 'gdspy' and not HAS_GDSPY:
            raise ImportError("gdspy 未安装")

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def analyze_file(self,
                     filepath: Union[str, Path],
                     layer: Optional[int] = None,
                     datatype: int = 0) -> HierarchyGraph:
        """
        分析单个 GDS 文件的层次结构

        Args:
            filepath: GDS/OASIS 文件路径
            layer: 指定层号（用于统计多边形数和包围盒），None 则跳过
            datatype: 数据类型号

        Returns:
            HierarchyGraph
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GDS 文件不存在: {filepath}")

        if self.backend == 'gdstk':
            return self._analyze_gdstk(str(filepath), layer, datatype)
        elif self.backend == 'gdspy':
            return self._analyze_gdspy(str(filepath), layer, datatype)
        else:
            raise ImportError("需要安装 gdstk 或 gdspy")

    # ------------------------------------------------------------------
    # gdstk 后端
    # ------------------------------------------------------------------

    def _analyze_gdstk(self,
                       filepath: str,
                       layer: Optional[int],
                       datatype: int) -> HierarchyGraph:
        lib = gdstk.read_gds(filepath)
        cells = lib.cells
        cell_map = {c.name: c for c in cells}

        graph = HierarchyGraph()
        for c in cells:
            graph.nodes[c.name] = HierarchyNode(cell_name=c.name)

        top_cell_names = [c.name for c in lib.top_level()] or [c.name for c in cells]
        graph.top_cells = top_cell_names

        for c in cells:
            node = graph.nodes[c.name]
            node.is_top = c.name in top_cell_names

            if layer is not None:
                poly_cnt = 0
                for p in c.polygons:
                    if p.layer == layer and p.datatype == datatype:
                        poly_cnt += 1
                node.polygon_count = poly_cnt
                if poly_cnt > 0:
                    try:
                        bb = c.bounding_box()
                        if bb is not None:
                            node.bounds = (float(bb[0][0]), float(bb[0][1]),
                                           float(bb[1][0]), float(bb[1][1]))
                    except Exception:
                        pass

            for ref in c.references:
                child_name = self._get_ref_cell_name_gdstk(ref)
                if child_name is None or child_name not in graph.nodes:
                    continue

                if self._is_array_ref_gdstk(ref):
                    aref = self._parse_array_ref_gdstk(ref, child_name)
                    node.array_refs.append(aref)
                    node.total_child_instances += aref.total_instances
                else:
                    inst = self._parse_single_ref_gdstk(ref, child_name)
                    node.single_refs.append(inst)
                    node.total_child_instances += 1

                if child_name not in node.children:
                    node.children.append(child_name)
                if c.name not in graph.nodes[child_name].parents:
                    graph.nodes[child_name].parents.append(c.name)
                graph.nodes[child_name].is_top = False

            node.is_leaf = len(node.single_refs) == 0 and len(node.array_refs) == 0

        graph.leaf_cells = [name for name, n in graph.nodes.items() if n.is_leaf]
        graph.compute_depths()
        graph.compute_bounds()  # 自底向上计算所有 cell 的完整包围盒

        logger.info(
            f"层次分析完成: {len(graph.nodes)} cells, "
            f"{len(graph.top_cells)} 顶层, "
            f"{len(graph.leaf_cells)} 叶节点, "
            f"最大深度 {graph.summary()['max_depth']}"
        )
        return graph

    @staticmethod
    def _get_ref_cell_name_gdstk(ref) -> Optional[str]:
        cell = getattr(ref, 'cell', None)
        if cell is None:
            return None
        return getattr(cell, 'name', None)

    @staticmethod
    def _is_array_ref_gdstk(ref) -> bool:
        rep = getattr(ref, 'repetition', None)
        if rep is None:
            return False
        cols = getattr(rep, 'columns', None) or 1
        rows = getattr(rep, 'rows', None) or 1
        return cols > 1 or rows > 1

    @staticmethod
    def _parse_single_ref_gdstk(ref, child_name: str) -> CellInstance:
        origin = tuple(getattr(ref, 'origin', (0.0, 0.0))) or (0.0, 0.0)
        origin = (float(origin[0]), float(origin[1]))
        return CellInstance(
            child_cell_name=child_name,
            origin=origin,
            rotation=float(getattr(ref, 'rotation', 0.0) or 0.0),
            magnification=float(getattr(ref, 'magnification', 1.0) or 1.0),
            x_reflection=bool(getattr(ref, 'x_reflection', False)),
            is_array_member=False,
            array_index=None,
        )

    @staticmethod
    def _parse_array_ref_gdstk(ref, child_name: str) -> ArrayReferenceInfo:
        origin = tuple(getattr(ref, 'origin', (0.0, 0.0))) or (0.0, 0.0)
        origin = (float(origin[0]), float(origin[1]))
        rep = getattr(ref, 'repetition', None)
        cols = 1
        rows = 1
        sx, sy = 0.0, 0.0
        if rep is not None:
            cols = int(getattr(rep, 'columns', None) or 1)
            rows = int(getattr(rep, 'rows', None) or 1)
            spacing = getattr(rep, 'spacing', None)
            if spacing is not None and len(spacing) >= 2:
                sx, sy = float(spacing[0]), float(spacing[1])
            else:
                v1 = getattr(rep, 'v1', None)
                v2 = getattr(rep, 'v2', None)
                if v1 is not None and len(v1) >= 2:
                    sx = float(v1[0]) / max(1, cols - 1) if cols > 1 else 0.0
                if v2 is not None and len(v2) >= 2:
                    sy = float(v2[1]) / max(1, rows - 1) if rows > 1 else 0.0
        return ArrayReferenceInfo(
            child_cell_name=child_name,
            rows=rows,
            cols=cols,
            spacing_x=sx,
            spacing_y=sy,
            base_origin=origin,
            rotation=float(getattr(ref, 'rotation', 0.0) or 0.0),
            magnification=float(getattr(ref, 'magnification', 1.0) or 1.0),
            x_reflection=bool(getattr(ref, 'x_reflection', False)),
        )

    # ------------------------------------------------------------------
    # gdspy 后端
    # ------------------------------------------------------------------

    def _analyze_gdspy(self,
                       filepath: str,
                       layer: Optional[int],
                       datatype: int) -> HierarchyGraph:
        lib = gdspy.GdsLibrary(infile=filepath)
        cells = lib.cells

        graph = HierarchyGraph()
        for name in cells.keys():
            graph.nodes[name] = HierarchyNode(cell_name=name)

        top_cells = lib.top_level()
        top_names = list(top_cells.keys()) if isinstance(top_cells, dict) else [c.name for c in top_cells]
        if not top_names:
            top_names = list(cells.keys())
        graph.top_cells = top_names

        for name, cell in cells.items():
            node = graph.nodes[name]
            node.is_top = name in top_names

            if layer is not None:
                poly_cnt = 0
                for ps in getattr(cell, 'polygons', []):
                    if (getattr(ps, 'layers', None) and ps.layers[0] == layer
                            and getattr(ps, 'datatypes', None)
                            and ps.datatypes[0] == datatype):
                        poly_cnt += len(ps.polygons)
                node.polygon_count = poly_cnt

            for ref in getattr(cell, 'references', []):
                child_name = self._get_ref_cell_name_gdspy(ref, lib)
                if child_name is None or child_name not in graph.nodes:
                    continue

                if self._is_array_ref_gdspy(ref):
                    aref = self._parse_array_ref_gdspy(ref, child_name)
                    node.array_refs.append(aref)
                    node.total_child_instances += aref.total_instances
                else:
                    inst = self._parse_single_ref_gdspy(ref, child_name)
                    node.single_refs.append(inst)
                    node.total_child_instances += 1

                if child_name not in node.children:
                    node.children.append(child_name)
                if name not in graph.nodes[child_name].parents:
                    graph.nodes[child_name].parents.append(name)
                graph.nodes[child_name].is_top = False

            node.is_leaf = len(node.single_refs) == 0 and len(node.array_refs) == 0

        graph.leaf_cells = [name for name, n in graph.nodes.items() if n.is_leaf]
        graph.compute_depths()
        graph.compute_bounds()  # 自底向上计算所有 cell 的完整包围盒

        logger.info(
            f"层次分析完成: {len(graph.nodes)} cells, "
            f"{len(graph.top_cells)} 顶层, "
            f"{len(graph.leaf_cells)} 叶节点"
        )
        return graph

    @staticmethod
    def _get_ref_cell_name_gdspy(ref, lib) -> Optional[str]:
        ref_cell = getattr(ref, 'ref_cell', None)
        if ref_cell is None:
            return None
        if isinstance(ref_cell, str):
            return ref_cell
        return getattr(ref_cell, 'name', None)

    @staticmethod
    def _is_array_ref_gdspy(ref) -> bool:
        rows = getattr(ref, 'rows', 1) or 1
        cols = getattr(ref, 'cols', 1) or 1
        return rows > 1 or cols > 1

    @staticmethod
    def _parse_single_ref_gdspy(ref, child_name: str) -> CellInstance:
        origin = tuple(getattr(ref, 'origin', (0.0, 0.0))) or (0.0, 0.0)
        origin = (float(origin[0]), float(origin[1]))
        return CellInstance(
            child_cell_name=child_name,
            origin=origin,
            rotation=float(getattr(ref, 'rotation', 0.0) or 0.0),
            magnification=float(getattr(ref, 'magnification', 1.0) or 1.0),
            x_reflection=bool(getattr(ref, 'x_reflection', False)),
            is_array_member=False,
            array_index=None,
        )

    @staticmethod
    def _parse_array_ref_gdspy(ref, child_name: str) -> ArrayReferenceInfo:
        origin = tuple(getattr(ref, 'origin', (0.0, 0.0))) or (0.0, 0.0)
        origin = (float(origin[0]), float(origin[1]))
        spacing = getattr(ref, 'spacing', None)
        sx, sy = 0.0, 0.0
        if spacing is not None and len(spacing) >= 2:
            sx, sy = float(spacing[0]), float(spacing[1])
        rows = getattr(ref, 'rows', 1) or 1
        cols = getattr(ref, 'cols', 1) or 1
        return ArrayReferenceInfo(
            child_cell_name=child_name,
            rows=int(rows),
            cols=int(cols),
            spacing_x=sx,
            spacing_y=sy,
            base_origin=origin,
            rotation=float(getattr(ref, 'rotation', 0.0) or 0.0),
            magnification=float(getattr(ref, 'magnification', 1.0) or 1.0),
            x_reflection=bool(getattr(ref, 'x_reflection', False)),
        )


# ============================================================================
# 层次化任务分解
# ============================================================================


@dataclass
class HierarchicalTask:
    """
    层次化优化任务

    Attributes:
        task_type: 'leaf' | 'composite'
            leaf: 叶节点，直接仿真该 cell
            composite: 复合节点，复用子 cell 结果 + 自身几何
        cell_name: 目标 cell 名
        unique_cell_key: 唯一标识（相同几何的 cell 共享结果）
        parent_tasks: 依赖此任务结果的父任务 cell 名列表
        child_results_needed: 需要从哪些子 cell 获取结果
            child_name -> list of (instance_transform, bounds)
        priority: 调度优先级
        needs_full_simulation: 是否需要完整仿真（无法复用子结果时）
        estimated_size: 预估掩模尺寸（像素数），用于优先级排序
    """
    task_type: str
    cell_name: str
    unique_cell_key: str
    parent_tasks: List[str] = field(default_factory=list)
    child_results_needed: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    priority: int = 50
    needs_full_simulation: bool = False
    estimated_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_type': self.task_type,
            'cell_name': self.cell_name,
            'unique_cell_key': self.unique_cell_key,
            'parent_count': len(self.parent_tasks),
            'child_count': len(self.child_results_needed),
            'total_child_instances': sum(
                len(v) for v in self.child_results_needed.values()
            ),
            'priority': self.priority,
            'needs_full_simulation': self.needs_full_simulation,
            'estimated_size': self.estimated_size,
        }


@dataclass
class HierarchyTaskPlan:
    """
    层次化任务分解计划

    由 HierarchyTaskPlanner 生成，描述了哪些 cell 需要独立仿真，
    哪些可以复用子 cell 结果，以及任务的执行顺序。

    Attributes:
        tasks: cell_name -> HierarchicalTask
        execution_order: 建议的执行顺序（cell_name 列表，叶节点优先）
        unique_tasks: 需要独立仿真的唯一 cell 数（按 unique_cell_key 去重）
        potential_savings: 预估节省的仿真次数
            = (原始扁平 cell 数) - (实际需要仿真的唯一 cell 数)
        graph: 关联的 HierarchyGraph
        raw_to_unique_name: GDS 原始 cell 名 → LayoutCell 唯一名称的映射
            LayoutCell.name 通常带有 library 前缀（如 "chip::TOP"），
            而 HierarchyGraph 中使用的是原始 cell 名（如 "TOP"）。
            此映射确保两者可以正确关联。
        unique_to_raw_name: LayoutCell 唯一名称 → GDS 原始 cell 名的反向映射
    """
    tasks: Dict[str, HierarchicalTask] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    unique_tasks: int = 0
    potential_savings: int = 0
    graph: Optional[HierarchyGraph] = None
    raw_to_unique_name: Dict[str, str] = field(default_factory=dict)
    unique_to_raw_name: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        leaf_tasks = sum(1 for t in self.tasks.values() if t.task_type == 'leaf')
        composite_tasks = len(self.tasks) - leaf_tasks
        full_sim = sum(1 for t in self.tasks.values() if t.needs_full_simulation)
        return {
            'total_tasks': len(self.tasks),
            'leaf_tasks': leaf_tasks,
            'composite_tasks': composite_tasks,
            'unique_tasks': self.unique_tasks,
            'full_simulation_required': full_sim,
            'potential_savings': self.potential_savings,
            'execution_order_length': len(self.execution_order),
            'name_mapping_count': len(self.raw_to_unique_name),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': self.summary(),
            'execution_order': self.execution_order,
            'tasks': {name: t.to_dict() for name, t in self.tasks.items()},
            'graph_summary': self.graph.summary() if self.graph else None,
            'raw_to_unique_name': self.raw_to_unique_name,
        }

    def get_unique_name(self, raw_name: str) -> Optional[str]:
        """通过原始 cell 名获取 LayoutCell 唯一名称"""
        return self.raw_to_unique_name.get(raw_name)

    def get_raw_name(self, unique_name: str) -> Optional[str]:
        """通过 LayoutCell 唯一名称获取原始 cell 名"""
        return self.unique_to_raw_name.get(unique_name)


class HierarchyTaskPlanner:
    """
    层次化任务规划器

    基于 HierarchyGraph 分析哪些 cell 需要独立仿真，
    哪些可以通过复用子 cell 的仿真结果来加速。

    优化策略：
    1. 叶节点必须独立仿真
    2. 被多个父 cell 引用的 cell，只需仿真一次，结果被所有父复用
    3. 阵列引用的 cell，只需仿真一次，然后按阵列实例变换复制结果
    4. 如果复合 cell 自身有大量多边形且子 cell 少，则退化到完整仿真

    使用示例:
        planner = HierarchyTaskPlanner()
        plan = planner.plan(graph, options=HierarchyPlanOptions())
        for cell_name in plan.execution_order:
            task = plan.tasks[cell_name]
            # 执行仿真或复用
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def plan(self,
             graph: HierarchyGraph,
             options: Optional['HierarchyPlanOptions'] = None) -> HierarchyTaskPlan:
        """
        根据层次图生成任务计划

        Args:
            graph: HierarchyGraph
            options: 规划选项

        Returns:
            HierarchyTaskPlan
        """
        opts = options or HierarchyPlanOptions()

        plan = HierarchyTaskPlan(graph=graph)
        plan.execution_order = graph.topological_order(bottom_up=True)

        cell_to_unique: Dict[str, str] = {}
        seen_unique: Set[str] = set()

        for cell_name in plan.execution_order:
            node = graph[cell_name]

            unique_key = self._compute_unique_key(cell_name, node, opts)
            cell_to_unique[cell_name] = unique_key

            task_type = 'leaf' if node.is_leaf else 'composite'

            task = HierarchicalTask(
                task_type=task_type,
                cell_name=cell_name,
                unique_cell_key=unique_key,
                parent_tasks=list(node.parents),
            )

            if not node.is_leaf:
                task.child_results_needed = self._collect_child_needs(node, opts)
                task.needs_full_simulation = self._should_full_simulate(node, opts, graph)

            est_size = self._estimate_size(node, opts)
            task.estimated_size = est_size
            task.priority = self._compute_priority(node, est_size, opts)

            plan.tasks[cell_name] = task

            if unique_key not in seen_unique:
                seen_unique.add(unique_key)
                plan.unique_tasks += 1

        plan.potential_savings = max(0, len(graph.nodes) - plan.unique_tasks)

        logger.info(
            f"层次化任务规划: {len(plan.tasks)} 个任务, "
            f"{plan.unique_tasks} 个唯一仿真, "
            f"预估节省 {plan.potential_savings} 次冗余仿真"
        )
        return plan

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_unique_key(cell_name: str,
                            node: HierarchyNode,
                            options: 'HierarchyPlanOptions') -> str:
        """
        计算 cell 的唯一标识 key

        相同几何的 cell（同名单例、内容 checksum 相同）共享 key。
        当前基于 cell_name 唯一（GDS 中 cell_name 本身就是唯一的），
        未来可扩展为基于几何内容的 hash。
        """
        return cell_name

    @staticmethod
    def _collect_child_needs(
        node: HierarchyNode,
        options: 'HierarchyPlanOptions',
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        收集本 cell 需要从子 cell 获取的结果列表

        Returns:
            child_name -> list of {
                'transform': 3x3 齐次矩阵,
                'is_array_member': bool,
                'array_index': (row, col) or None,
            }
        """
        needs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for inst in node.all_child_instances:
            needs[inst.child_cell_name].append({
                'transform': inst.transform,
                'is_array_member': inst.is_array_member,
                'array_index': inst.array_index,
            })
        return dict(needs)

    @staticmethod
    def _should_full_simulate(node: HierarchyNode,
                              options: 'HierarchyPlanOptions',
                              graph: 'HierarchyGraph' = None) -> bool:
        """
        判断复合 cell 是否应该退化到完整仿真

        当自身多边形占比过高、或子 cell 数量过少时，
        复用的收益低于合成开销，直接完整仿真更高效。
        """
        if node.is_leaf:
            return False

        def _recursive_total_instances(n: HierarchyNode, visited: set) -> int:
            """递归计算所有子实例总数（含间接子节点的展开）"""
            if n.cell_name in visited:
                return 0
            visited.add(n.cell_name)

            total = n.total_child_instances
            if graph:
                for child_name in n.children:
                    child_node = graph.nodes.get(child_name)
                    if child_node and not child_node.is_leaf:
                        total += _recursive_total_instances(child_node, visited)
            return total

        self_polys = max(1, node.polygon_count)
        total_child = _recursive_total_instances(node, set())
        ratio = self_polys / max(1, self_polys + total_child * options.child_polygon_estimate)

        if ratio > options.self_polygon_ratio_threshold:
            return True
        if total_child < options.min_child_instances_for_reuse:
            return True

        return False

    @staticmethod
    def _estimate_size(node: HierarchyNode,
                       options: 'HierarchyPlanOptions') -> int:
        """预估 cell 掩模的像素尺寸（用于优先级排序）"""
        if node.bounds:
            xmin, ymin, xmax, ymax = node.bounds
            w = (xmax - xmin) / max(options.pixel_size, 1e-6)
            h = (ymax - ymin) / max(options.pixel_size, 1e-6)
            return max(1, int(w * h))
        return max(1, node.polygon_count * options.polygon_pixel_estimate)

    @staticmethod
    def _compute_priority(node: HierarchyNode,
                          est_size: int,
                          options: 'HierarchyPlanOptions') -> int:
        """
        计算任务优先级

        - 叶节点优先（子任务先完成，父任务才能开始）
        - 被多次引用的 cell 优先（尽早解锁多个父任务）
        - 大面积 cell 优先（更容易成为瓶颈）
        """
        base = 50
        if node.is_leaf:
            base += 30
        parent_boost = min(20, len(node.parents) * 5)
        size_boost = min(10, est_size // 10000)
        return max(0, min(100, base + parent_boost + size_boost))


@dataclass
class HierarchyPlanOptions:
    """
    层次化任务规划选项

    Attributes:
        pixel_size: 栅格化像素尺寸（用于预估掩模尺寸）
        self_polygon_ratio_threshold: 自身多边形占比阈值，超过则完整仿真
        min_child_instances_for_reuse: 最少子实例数，低于则不值得复用
        child_polygon_estimate: 每个子 cell 预估的多边形数（用于阈值计算）
        polygon_pixel_estimate: 每个多边形预估占用像素数
        prioritize_by_reference_count: 是否按被引用次数提升优先级
    """
    pixel_size: float = 1.0
    self_polygon_ratio_threshold: float = 0.7
    min_child_instances_for_reuse: int = 3
    child_polygon_estimate: int = 50
    polygon_pixel_estimate: int = 100
    prioritize_by_reference_count: bool = True


# ============================================================================
# 层次化结果合并器
# ============================================================================


class HierarchyResultMerger:
    """
    层次化仿真结果合并器

    将子 cell 的仿真/优化结果按引用变换合成到父 cell，
    避免对大芯片中重复单元做冗余全图仿真。

    核心功能：
    1. 结果缓存：按 unique_cell_key 缓存已完成的仿真结果
    2. 掩模合成：将子 cell 结果通过仿射变换拼接到父 cell 掩模上
    3. 指标聚合：将子 cell 的 MSE/EPE 等指标聚合为父 cell 的指标估计

    使用示例:
        merger = HierarchyResultMerger()
        merger.cache_result("SUB_CELL", optimized_mask, metrics)
        # 当需要合成父 cell 时:
        parent_mask = merger.compose_parent_mask("TOP", task, child_masks, parent_bounds)
    """

    def __init__(self, pixel_size: float = 1.0):
        self.pixel_size = pixel_size
        self._mask_cache: Dict[str, np.ndarray] = {}
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 缓存管理
    # ------------------------------------------------------------------

    def cache_result(self,
                     unique_key: str,
                     optimized_mask: np.ndarray,
                     metrics: Optional[Dict[str, Any]] = None) -> None:
        """缓存 cell 的仿真/优化结果"""
        self._mask_cache[unique_key] = optimized_mask.astype(np.float64)
        if metrics is not None:
            self._metrics_cache[unique_key] = dict(metrics)

    def has_cached(self, unique_key: str) -> bool:
        return unique_key in self._mask_cache

    def get_cached_mask(self, unique_key: str) -> Optional[np.ndarray]:
        return self._mask_cache.get(unique_key)

    def get_cached_metrics(self, unique_key: str) -> Optional[Dict[str, Any]]:
        return self._metrics_cache.get(unique_key)

    # ------------------------------------------------------------------
    # 掩模合成
    # ------------------------------------------------------------------

    def compose_parent_mask(self,
                            parent_task: HierarchicalTask,
                            parent_self_mask: Optional[np.ndarray],
                            parent_bounds: Tuple[float, float, float, float],
                            target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        将子 cell 结果与父 cell 自身几何合成为完整掩模

        Args:
            parent_task: 父 cell 的 HierarchicalTask
            parent_self_mask: 父 cell 自身几何的栅格化掩模（不含子引用）
            parent_bounds: 父 cell 的完整包围盒 (xmin, ymin, xmax, ymax)，世界坐标
            target_size: 目标尺寸 (H, W)，None 则根据 bounds 和 pixel_size 计算

        Returns:
            合成后的完整掩模 (H, W) float64，值域 [0, 1]
        """
        xmin, ymin, xmax, ymax = parent_bounds
        if target_size is not None:
            ny, nx = target_size
        else:
            nx = max(1, int(np.ceil((xmax - xmin) / self.pixel_size)))
            ny = max(1, int(np.ceil((ymax - ymin) / self.pixel_size)))

        if parent_self_mask is not None:
            result = parent_self_mask.astype(np.float64).copy()
            if result.shape != (ny, nx):
                result = self._resize_mask(result, ny, nx)
        else:
            result = np.zeros((ny, nx), dtype=np.float64)

        for child_name, instances in parent_task.child_results_needed.items():
            child_mask = self.get_cached_mask(child_name)
            if child_mask is None:
                logger.warning(
                    f"合并父 cell {parent_task.cell_name}: "
                    f"子 cell {child_name} 结果未缓存，跳过"
                )
                continue

            for inst_info in instances:
                transform = inst_info.get('transform')
                if transform is None:
                    continue
                self._paste_child_mask(
                    result, child_mask, transform,
                    xmin, ymin, xmax, ymax, nx, ny
                )

        result = np.clip(result, 0.0, 1.0)
        return result

    def _paste_child_mask(self,
                          parent_mask: np.ndarray,
                          child_mask: np.ndarray,
                          transform: np.ndarray,
                          parent_xmin: float,
                          parent_ymin: float,
                          parent_xmax: float,
                          parent_ymax: float,
                          parent_nx: int,
                          parent_ny: int) -> None:
        """
        将子 cell 掩模通过仿射变换粘贴到父 cell 掩模上

        坐标系统说明（严格区分像素坐标与世界坐标）：
        - 像素坐标：以像素为单位，(0,0) 在左上角，y 向下增长
        - 世界坐标：以物理长度为单位（如 nm），y 向上增长
        - pixel_size：每个像素对应的物理长度
        - transform：子世界坐标 → 父世界坐标 的 3×3 齐次变换矩阵

        变换链路：
        子像素 → 子世界 → transform → 父世界 → 父像素

        原地修改 parent_mask。
        """
        ch, cw = child_mask.shape
        ps = self.pixel_size

        # ------------------------------------------------------------
        # Step 1: 计算子 cell 四个角在父世界坐标系中的位置
        # ------------------------------------------------------------
        # 子 cell 世界坐标的 4 个角（子 cell 本地坐标系，原点在左下角）
        # 注意：子掩模图像坐标系 (0,0) 在左上角，世界坐标系 (0,0) 在左下角
        child_corners_world = np.array([
            [0.0,              (ch - 1) * ps],  # 左上角 (世界: 左, 上)
            [(cw - 1) * ps,    (ch - 1) * ps],  # 右上角
            [(cw - 1) * ps,    0.0],             # 右下角
            [0.0,              0.0],             # 左下角
        ], dtype=np.float64)

        # 子世界 → 父世界
        corners_h = np.column_stack([
            child_corners_world[:, 0],
            child_corners_world[:, 1],
            np.ones(4),
        ])
        parent_corners_world = (transform @ corners_h.T).T[:, :2]

        # 父世界 → 父像素
        # 父像素 px = (wx - parent_xmin) / pixel_size
        # 父像素 py = (parent_ymax - wy) / pixel_size  （y 轴翻转）
        parent_corners_px = np.zeros_like(parent_corners_world)
        parent_corners_px[:, 0] = (parent_corners_world[:, 0] - parent_xmin) / ps
        parent_corners_px[:, 1] = (parent_ymax - parent_corners_world[:, 1]) / ps

        # ------------------------------------------------------------
        # Step 2: 计算父掩模中需要处理的 ROI 范围
        # ------------------------------------------------------------
        x_coords = parent_corners_px[:, 0]
        y_coords = parent_corners_px[:, 1]
        x0 = int(np.floor(x_coords.min()))
        x1 = int(np.ceil(x_coords.max()))
        y0 = int(np.floor(y_coords.min()))
        y1 = int(np.ceil(y_coords.max()))

        # 裁剪到父掩模边界内
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(parent_nx, x1)
        y1 = min(parent_ny, y1)
        if x1 <= x0 or y1 <= y0:
            return

        # ------------------------------------------------------------
        # Step 3: 对 ROI 内每个父像素，逆变换到子世界，再到子像素做双线性插值
        # ------------------------------------------------------------
        try:
            inv_transform = np.linalg.inv(transform)
        except np.linalg.LinAlgError:
            return

        yy, xx = np.mgrid[y0:y1, x0:x1]
        px_flat = xx.ravel().astype(np.float64)
        py_flat = yy.ravel().astype(np.float64)

        # 父像素 → 父世界
        parent_wx = parent_xmin + px_flat * ps
        parent_wy = parent_ymax - py_flat * ps

        # 父世界 → 子世界（逆变换）
        parent_w_h = np.column_stack([parent_wx, parent_wy, np.ones_like(parent_wx)])
        child_world = (inv_transform @ parent_w_h.T).T[:, :2]

        # 子世界 → 子像素
        # 子像素 px = wx / pixel_size
        # 子像素 py = (ch - 1) - wy / pixel_size  （y 轴翻转）
        child_px = child_world[:, 0] / ps
        child_py = (ch - 1) - child_world[:, 1] / ps

        # ------------------------------------------------------------
        # Step 4: 双线性插值采样
        # ------------------------------------------------------------
        valid = (
            (child_px >= 0) & (child_px < cw - 1)
            & (child_py >= 0) & (child_py < ch - 1)
        )
        if not np.any(valid):
            return

        cx_valid = child_px[valid]
        cy_valid = child_py[valid]
        px_idx = np.clip(np.floor(cx_valid).astype(int), 0, cw - 2)
        py_idx = np.clip(np.floor(cy_valid).astype(int), 0, ch - 2)
        fx = cx_valid - px_idx
        fy = cy_valid - py_idx

        v00 = child_mask[py_idx, px_idx]
        v10 = child_mask[py_idx, px_idx + 1]
        v01 = child_mask[py_idx + 1, px_idx]
        v11 = child_mask[py_idx + 1, px_idx + 1]
        interp = (
            v00 * (1 - fx) * (1 - fy)
            + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy
            + v11 * fx * fy
        )

        dest = parent_mask[y0:y1, x0:x1]
        dest_flat = dest.ravel()
        valid_idx = np.where(valid)[0]
        dest_flat[valid_idx] = np.maximum(dest_flat[valid_idx], interp)
        parent_mask[y0:y1, x0:x1] = dest_flat.reshape(dest.shape)

    @staticmethod
    def _resize_mask(mask: np.ndarray, ny: int, nx: int) -> np.ndarray:
        """将掩模调整到目标尺寸（使用最近邻，保持二值性）"""
        h, w = mask.shape
        if h == ny and w == nx:
            return mask
        try:
            import cv2
            resized = cv2.resize(
                mask, (nx, ny),
                interpolation=cv2.INTER_NEAREST
            )
            return resized.astype(np.float64)
        except ImportError:
            yy_idx = np.clip(
                (np.arange(ny) * h / max(1, ny)).astype(int), 0, h - 1
            )
            xx_idx = np.clip(
                (np.arange(nx) * w / max(1, nx)).astype(int), 0, w - 1
            )
            return mask[np.ix_(yy_idx, xx_idx)]

    # ------------------------------------------------------------------
    # 指标聚合
    # ------------------------------------------------------------------

    def aggregate_parent_metrics(self,
                                 parent_task: HierarchicalTask,
                                 parent_self_metrics: Optional[Dict[str, Any]] = None,
                                 ) -> Dict[str, Any]:
        """
        从子 cell 指标聚合估算父 cell 指标

        这是一个近似估计，用于快速预览。精确指标需要对合成后的完整掩模
        做一次光刻仿真。

        Args:
            parent_task: 父任务
            parent_self_metrics: 父 cell 自身几何（不含子引用）的指标

        Returns:
            聚合后的指标字典
        """
        child_metrics_list = []
        weights = []
        for child_name, instances in parent_task.child_results_needed.items():
            m = self.get_cached_metrics(child_name)
            if m is None:
                continue
            n_inst = len(instances)
            child_metrics_list.append(m)
            weights.append(n_inst)

        if not child_metrics_list and parent_self_metrics is None:
            return {}

        aggregated: Dict[str, Any] = {}
        for metric_key in ['mse', 'epe_mean', 'ssim']:
            values = []
            ws = []
            for m, w in zip(child_metrics_list, weights):
                if metric_key in m and m[metric_key] is not None:
                    values.append(float(m[metric_key]))
                    ws.append(w)
            if parent_self_metrics and metric_key in parent_self_metrics:
                if parent_self_metrics[metric_key] is not None:
                    values.append(float(parent_self_metrics[metric_key]))
                    ws.append(1)
            if values:
                if metric_key == 'ssim':
                    aggregated[metric_key] = float(np.mean(values))
                else:
                    total_w = sum(ws) or 1
                    aggregated[metric_key] = float(
                        sum(v * w for v, w in zip(values, ws)) / total_w
                    )

        if parent_self_metrics:
            for k, v in parent_self_metrics.items():
                if k not in aggregated:
                    aggregated[k] = v

        aggregated['_note'] = (
            '指标由子 cell 聚合估算，非精确值。'
            '精确值请对合成掩模做完整仿真。'
        )
        return aggregated


# ============================================================================
# LayoutManager 层次化扩展
# ============================================================================


def _extend_layout_manager():
    """为 LayoutManager 动态添加层次化相关方法（保持向后兼容）"""

    def analyze_hierarchy(self,
                          filepath: Union[str, Path],
                          layer: Optional[int] = None,
                          datatype: int = 0) -> HierarchyGraph:
        """
        分析 GDS 文件的层次结构

        Args:
            filepath: GDS/OASIS 文件路径
            layer: 用于统计多边形的层号，None 则跳过统计
            datatype: 数据类型号

        Returns:
            HierarchyGraph
        """
        analyzer = LayoutHierarchyAnalyzer(backend=self.loader.backend)
        return analyzer.analyze_file(filepath, layer=layer, datatype=datatype)

    def plan_hierarchical_tasks(self,
                                graph: HierarchyGraph,
                                options: Optional[HierarchyPlanOptions] = None,
                                ) -> HierarchyTaskPlan:
        """
        根据层次图生成优化任务计划

        Args:
            graph: HierarchyGraph
            options: 规划选项

        Returns:
            HierarchyTaskPlan
        """
        planner = HierarchyTaskPlanner()
        return planner.plan(graph, options=options)

    def create_merger(self, pixel_size: float = 1.0) -> HierarchyResultMerger:
        """创建层次化结果合并器"""
        return HierarchyResultMerger(pixel_size=pixel_size)

    def load_and_queue_hierarchical(
        self,
        filepath: Union[str, Path],
        layer: int,
        **kwargs,
    ) -> Tuple[LayoutLibrary, LayoutQueue, HierarchyGraph, HierarchyTaskPlan]:
        """
        层次化加载与建队（推荐入口）

        相比 load_and_queue 的额外返回：
        - HierarchyGraph: 层次图，供分析
        - HierarchyTaskPlan: 任务计划，含执行顺序与复用信息

        队列中的 LayoutCell 会附带层次相关 tags：
        - 'hier:leaf' / 'hier:composite'
        - 'hier:top'（顶层 cell）
        - 'hier:depth_N'

        关键的掩模加载策略：
        - 叶节点 (is_leaf=True): flatten_references=True，展平所有子引用，
          获得真实完整的掩模用于仿真
        - 复合节点 (is_leaf=False): flatten_references=False，只取自身直接
          多边形（通常为空），作为子引用合成的基底

        Args:
            filepath: GDS/OASIS 文件路径
            layer: GDS 层号
            **kwargs: 其他 LayoutLoadOptions / HierarchyPlanOptions 参数

        Returns:
            (LayoutLibrary, LayoutQueue, HierarchyGraph, HierarchyTaskPlan)
        """
        opts_kwargs = {'layer': layer, **{
            k: v for k, v in kwargs.items()
            if k in LayoutLoadOptions.__dataclass_fields__
        }}
        load_opts_base = LayoutLoadOptions(**opts_kwargs)
        load_opts_base.include_subcells = True
        load_opts_base.skip_empty_cells = False  # 层次化模式下，即使没有自身多边形也要加载（如只有引用的父 cell）

        plan_kwargs = {k: v for k, v in kwargs.items()
                       if k in HierarchyPlanOptions.__dataclass_fields__}
        # pixel_size 同时在 LayoutLoadOptions 和 HierarchyPlanOptions 中，
        # 优先使用 kwargs 中显式传入的值，否则从 load_opts 继承
        if 'pixel_size' not in plan_kwargs:
            plan_kwargs['pixel_size'] = load_opts_base.pixel_size
        plan_opts = HierarchyPlanOptions(**plan_kwargs)

        # 1. 先分析层次结构（在加载掩模之前，需要知道哪些是叶节点）
        graph = self.analyze_hierarchy(filepath, layer=layer, datatype=load_opts_base.datatype)
        plan = self.plan_hierarchical_tasks(graph, options=plan_opts)

        # 2. 构建所有 cell 列表：在加载时按 is_leaf 决定是否展平子引用
        cell_names = self.loader.list_all_cells(filepath)
        file_key = Path(filepath).stem
        seen_names: Set[str] = set()
        cells: List[LayoutCell] = []

        for raw_name in cell_names:
            unique_name = self.loader._unique_name(
                seen_names, file_key, raw_name
            )
            seen_names.add(unique_name)

            node = graph.nodes.get(raw_name)
            is_leaf = node.is_leaf if node else True

            # 核心差异：
            # - 叶节点 (is_leaf=True):
            #   · flatten_references=True（展平所有子引用，得到真实掩模用于仿真）
            #   · bounds=None（自动根据自身多边形计算紧凑范围）
            # - 复合节点 (is_leaf=False):
            #   · flatten_references=False（只取自身直接多边形，子引用通过合成拼接）
            #   · bounds=node.bounds（使用完整包围盒作为栅格化坐标系，确保
            #     自身多边形落在完整坐标系的正确位置，与子引用合成坐标系一致）
            cell_load_opts = LayoutLoadOptions(
                **{
                    f: getattr(load_opts_base, f)
                    for f in LayoutLoadOptions.__dataclass_fields__
                }
            )
            cell_load_opts.flatten_references = is_leaf

            if not is_leaf and node is not None and node.bounds is not None:
                # 复合节点：使用完整包围盒作为坐标系
                # 确保自身多边形被栅格化到完整范围的正确位置
                cell_load_opts.bounds = node.bounds
                meta_bounds = node.bounds
            else:
                # 叶节点 / 没有 node 的情况
                meta_bounds = load_opts_base.bounds

            meta = LayoutCellMetadata(
                cell_name=raw_name,
                source_type=LayoutSourceType.GDS_FILE,
                source_path=str(Path(filepath).resolve()),
                layer=layer,
                datatype=load_opts_base.datatype,
                pixel_size=load_opts_base.pixel_size,
                bounds=meta_bounds,
                load_timestamp=0.0,
            )
            self.loader._fill_polygon_counts(
                filepath, raw_name, cell_load_opts, meta
            )

            cell = LayoutCell(
                name=unique_name,
                metadata=meta,
                priority=50,
                tags={'source:gds', f'file:{file_key}'},
            )

            if cell_load_opts.load_masks_on_init:
                import time
                t0 = time.time()
                mask = None
                try:
                    mask = self.loader.load_cell_mask(
                        filepath, raw_name, cell_load_opts
                    )
                except ValueError as e:
                    if (not cell_load_opts.skip_empty_cells
                            and '无多边形' in str(e)):
                        mask = None
                    else:
                        logger.warning(f"加载 cell {raw_name} 失败: {e}")
                        continue
                except Exception as e:
                    logger.warning(f"加载 cell {raw_name} 失败: {e}")
                    continue

                meta.load_timestamp = t0
                if mask is None:
                    if cell_load_opts.skip_empty_cells:
                        logger.debug(f"跳过空 cell: {raw_name}")
                        continue
                    # 创建占位符，后续根据 node.bounds 重新设置尺寸
                    mask = np.zeros((1, 1), dtype=np.float64)
                cell.mask = mask
                cell.target = mask.copy()
                meta.checksum = self.loader._mask_checksum(mask)

            cells.append(cell)

        # 3. 构建 Library，禁用去重（层次化模式下父子 cell 必须都存在）
        lib = LayoutLibrary(name=file_key)
        lib.add_many(cells, dedup=False)

        # 4. 建立原始 cell 名 → LayoutCell 唯一名称的双向映射
        raw_to_unique: Dict[str, str] = {}
        unique_to_raw: Dict[str, str] = {}
        for cell in lib.cells():
            raw_name = cell.cell_name
            unique_name = cell.name
            raw_to_unique[raw_name] = unique_name
            unique_to_raw[unique_name] = raw_name
        plan.raw_to_unique_name = raw_to_unique
        plan.unique_to_raw_name = unique_to_raw

        # 5. 为所有层次图中的 cell 打 tag + 处理只有引用的占位符掩模
        for cell in lib.cells():
            raw_name = cell.cell_name
            if raw_name in graph.nodes:
                node = graph.nodes[raw_name]
                cell.tags.add(f"hier:{'leaf' if node.is_leaf else 'composite'}")
                if node.is_top:
                    cell.tags.add('hier:top')
                cell.tags.add(f"hier:depth_{node.depth}")

                # 如果 cell mask 是占位符 (1,1) 且有 node.bounds，
                # 按 bounds 重置到真实尺寸的全 0 掩模（合成基底）
                if (cell.mask is not None and cell.mask.shape == (1, 1)
                        and node.bounds is not None):
                    xmin, ymin, xmax, ymax = node.bounds
                    nx = max(1, int(np.ceil(
                        (xmax - xmin) / plan_opts.pixel_size
                    )))
                    ny = max(1, int(np.ceil(
                        (ymax - ymin) / plan_opts.pixel_size
                    )))
                    cell.mask = np.zeros((ny, nx), dtype=np.float64)
                    cell.target = cell.mask.copy()

                if raw_name in plan.tasks:
                    task = plan.tasks[raw_name]
                    cell.priority = task.priority
                    if task.needs_full_simulation:
                        cell.tags.add('hier:full_sim')
                    cell.metadata.extra['hierarchy'] = {
                        'depth': node.depth,
                        'is_leaf': node.is_leaf,
                        'is_top': node.is_top,
                        'parent_count': len(node.parents),
                        'child_instances': node.total_child_instances,
                        'unique_key': task.unique_cell_key,
                        'raw_name': raw_name,
                        'unique_name': cell.name,
                        'flatten_refs': node.is_leaf,
                    }
                else:
                    cell.metadata.extra['hierarchy'] = {
                        'depth': node.depth,
                        'is_leaf': node.is_leaf,
                        'is_top': node.is_top,
                        'parent_count': len(node.parents),
                        'child_instances': node.total_child_instances,
                        'raw_name': raw_name,
                        'unique_name': cell.name,
                        'flatten_refs': node.is_leaf,
                    }

        q_kwargs = {k: v for k, v in kwargs.items()
                    if k in ('priority_by_size', 'priority_by_polygons',
                             'size_priority_reverse', 'max_retries',
                             'require_mask_loaded')}
        # 层次化模式下，require_mask_loaded 默认 False，允许只有引用没有自身多边形的 cell 入队
        q_kwargs.setdefault('require_mask_loaded', False)
        queue = self.build_queue(lib, **q_kwargs)

        return lib, queue, graph, plan

    LayoutManager.analyze_hierarchy = analyze_hierarchy
    LayoutManager.plan_hierarchical_tasks = plan_hierarchical_tasks
    LayoutManager.create_merger = create_merger
    LayoutManager.load_and_queue_hierarchical = load_and_queue_hierarchical


_extend_layout_manager()


__all__ = [
    'LayoutCell',
    'LayoutLibrary',
    'LayoutManager',
    'GDSLoader',
    'LayoutQueue',
    'LayoutLoadOptions',
    'LayoutCellMetadata',
    'LayoutSourceType',
    'CellInstance',
    'ArrayReferenceInfo',
    'HierarchyNode',
    'HierarchyGraph',
    'LayoutHierarchyAnalyzer',
    'HierarchicalTask',
    'HierarchyTaskPlan',
    'HierarchyTaskPlanner',
    'HierarchyPlanOptions',
    'HierarchyResultMerger',
]
