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
from pathlib import Path
from typing import (
    Optional, List, Dict, Any, Union, Tuple, Iterator,
    Callable, Set,
)
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import numpy as np

from utils.data_io import load_gds_layer

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

        Args:
            loader: GDSLoader 实例，若为 None 则尝试根据 source_path 自建
            options: 加载选项，为 None 则使用默认
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

        opts = options or LayoutLoadOptions(
            layer=self.metadata.layer,
            datatype=self.metadata.datatype or 0,
            pixel_size=self.metadata.pixel_size,
            target_size=None,
            bounds=self.metadata.bounds,
        )

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

        优先使用 utils.data_io.load_gds_layer（支持引用展平）。
        若需要指定 cell 名，则先尝试提取该 cell 的多边形。

        Args:
            filepath: GDS 文件路径
            cell_name: cell 名
            options: 加载选项

        Returns:
            二值掩模 (H, W) float64；若 cell 为空且 skip_empty 则返回 None
        """
        filepath = str(filepath)

        if options.layer is None:
            raise ValueError("LayoutLoadOptions.layer 不能为空")

        try:
            mask = load_gds_layer(
                filepath=filepath,
                layer=options.layer,
                datatype=options.datatype,
                pixel_size=options.pixel_size,
                target_size=options.target_size,
                bounds=options.bounds,
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
                try:
                    mask = self.load_cell_mask(filepath, cell_name, options)
                except Exception as e:
                    logger.warning(f"加载 cell {cell_name} 失败: {e}")
                    continue
                meta.load_timestamp = t0
                if mask is None:
                    if options.skip_empty_cells:
                        logger.debug(f"跳过空 cell: {cell_name}")
                        continue
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
