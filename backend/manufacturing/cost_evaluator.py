# -*- coding: utf-8 -*-
"""
掩模制造成本评估核心模块

基于掩模版图几何特征（多边形顶点、矩形Shot数量）、
数据存储体积以及掩模写入器物理参数，综合评估
掩模制造的复杂度与成本。

核心算法：
    1. 顶点数估算 (Vertex Count Estimation)
       - 直接从多边形提取顶点
       - 从像素掩模通过边缘轮廓检测估计
    2. 曼哈顿化 (Manhattanization)
       - 将任意斜多边形切分为轴对齐矩形 (Manhattan rectangles)
       - 经典切分策略：水平优先/垂直优先/最小矩形数
    3. Shot 数量估算 (Shot Count Estimation)
       - VSB (Variable Shaped Beam)：可变形电子束
       - 圆形/高斯束：按像素/栅格统计
    4. 数据体积估算 (Data Volume Estimation)
       - GDSII/OASIS 文件大小估算公式
       - 基于顶点数和层次结构
    5. 写入时间估算 (Write Time Estimation)
       - 电子束写入：剂量 × 面积 / 电流 + 偏转开销
       - 光学写入：扫描速度 × 面积 + 平台运动
    6. 复杂度分数 (Complexity Score)
       - 加权综合：vertex, shot, data, write_time
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.ndimage import (
    label, find_objects, sobel, binary_erosion,
    generate_binary_structure, binary_dilation,
)
from scipy.signal import convolve2d

logger = logging.getLogger(__name__)


# ============================================================================
# 枚举定义
# ============================================================================

class MaskWriterType(Enum):
    """
    掩模写入器类型

    决定写入时间和 shot 分形策略的计算公式。
    """
    VSB_EBEAM = 'vsb_ebeam'           # 可变形电子束 (Variable Shaped Beam)
    GAUSSIAN_EBEAM = 'gaussian_ebeam'  # 高斯束电子束
    DUV_OPTICAL = 'duv_optical'        # DUV 光学投影写入 (ALTA 等)
    EUV_OPTICAL = 'euv_optical'        # EUV 光学写入


class ShotFracturingStrategy(Enum):
    """
    Shot 分形策略

    将任意多边形切分为矩形 shot 时的切分策略。
    """
    MIN_RECTANGLES = 'min_rectangles'     # 最小矩形数（最优分形，计算较慢）
    HORIZONTAL_SWEEP = 'horizontal_sweep'  # 水平扫描切分（快速）
    VERTICAL_SWEEP = 'vertical_sweep'      # 垂直扫描切分（快速）
    GRID_BASED = 'grid_based'              # 栅格化近似（最快）


# ============================================================================
# 配置与结果数据结构
# ============================================================================

@dataclass
class ManufacturingCostConfig:
    """
    掩模制造成本评估配置

    Attributes:
        writer_type: 掩模写入器类型
        pixel_size_nm: 栅格化像素尺寸 (nm)
        shot_fracturing: Shot分形策略
        grid_size_nm: VSB shot 的栅格尺寸 (nm)

        # 写入器物理参数（影响写入时间估算）
        ebeam_current_nA: 电子束电流 (nA)，仅电子束类型
        dose_uC_cm2: 写入剂量 (μC/cm²)，仅电子束类型
        beam_blur_nm: 束斑模糊/束斑尺寸 (nm)
        stage_move_speed_mm_s: 平台移动速度 (mm/s)，仅光学写入
        optical_scan_speed_m_s: 光学扫描速度 (m/s)

        # Shot / 几何参数
        min_shot_size_nm: 最小 shot 尺寸 (nm)
        max_shot_size_nm: 最大 shot 尺寸 (nm)，VSB 的最大矩形
        shot_size_precision_nm: shot 尺寸精度 (nm)

        # 数据体积估算参数
        avg_bytes_per_vertex_gds: GDSII 每个顶点的平均字节数
        avg_bytes_per_vertex_oasis: OASIS 每个顶点的平均字节数
        hierarchy_factor: 层次压缩因子 (<1)，1表示无层次
        output_format: 输出格式 'gds' 或 'oasis'

        # 复杂度分数权重
        score_vertex_weight: 顶点数权重
        score_shot_weight: Shot数权重
        score_data_weight: 数据体积权重
        score_write_time_weight: 写入时间权重

        # 归一化基准值（相对复杂度时使用）
        baseline_vertex_count: 基准顶点数
        baseline_shot_count: 基准Shot数
        baseline_data_mb: 基准数据体积(MB)
        baseline_write_time_min: 基准写入时间(min)

        verbose: 是否输出详细日志
    """
    writer_type: MaskWriterType = MaskWriterType.VSB_EBEAM
    pixel_size_nm: float = 1.0
    shot_fracturing: ShotFracturingStrategy = ShotFracturingStrategy.HORIZONTAL_SWEEP
    grid_size_nm: float = 1.0

    ebeam_current_nA: float = 100.0
    dose_uC_cm2: float = 40.0
    beam_blur_nm: float = 5.0
    stage_move_speed_mm_s: float = 50.0
    optical_scan_speed_m_s: float = 10.0

    min_shot_size_nm: float = 2.0
    max_shot_size_nm: float = 2000.0
    shot_size_precision_nm: float = 1.0

    avg_bytes_per_vertex_gds: float = 16.0
    avg_bytes_per_vertex_oasis: float = 4.0
    hierarchy_factor: float = 0.6
    output_format: str = 'oasis'

    score_vertex_weight: float = 0.2
    score_shot_weight: float = 0.35
    score_data_weight: float = 0.2
    score_write_time_weight: float = 0.25

    baseline_vertex_count: float = 10000.0
    baseline_shot_count: float = 50000.0
    baseline_data_mb: float = 10.0
    baseline_write_time_min: float = 30.0

    verbose: bool = False

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'ManufacturingCostConfig':
        if d is None:
            return cls()
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                if key == 'writer_type':
                    cfg.writer_type = MaskWriterType(value) if isinstance(value, str) else value
                elif key == 'shot_fracturing':
                    cfg.shot_fracturing = ShotFracturingStrategy(value) if isinstance(value, str) else value
                else:
                    setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            'writer_type': self.writer_type.value,
            'pixel_size_nm': self.pixel_size_nm,
            'shot_fracturing': self.shot_fracturing.value,
            'grid_size_nm': self.grid_size_nm,
            'ebeam_current_nA': self.ebeam_current_nA,
            'dose_uC_cm2': self.dose_uC_cm2,
            'beam_blur_nm': self.beam_blur_nm,
            'stage_move_speed_mm_s': self.stage_move_speed_mm_s,
            'optical_scan_speed_m_s': self.optical_scan_speed_m_s,
            'min_shot_size_nm': self.min_shot_size_nm,
            'max_shot_size_nm': self.max_shot_size_nm,
            'shot_size_precision_nm': self.shot_size_precision_nm,
            'avg_bytes_per_vertex_gds': self.avg_bytes_per_vertex_gds,
            'avg_bytes_per_vertex_oasis': self.avg_bytes_per_vertex_oasis,
            'hierarchy_factor': self.hierarchy_factor,
            'output_format': self.output_format,
            'score_vertex_weight': self.score_vertex_weight,
            'score_shot_weight': self.score_shot_weight,
            'score_data_weight': self.score_data_weight,
            'score_write_time_weight': self.score_write_time_weight,
            'baseline_vertex_count': self.baseline_vertex_count,
            'baseline_shot_count': self.baseline_shot_count,
            'baseline_data_mb': self.baseline_data_mb,
            'baseline_write_time_min': self.baseline_write_time_min,
            'verbose': self.verbose,
        }

    def total_score_weight(self) -> float:
        return (self.score_vertex_weight + self.score_shot_weight +
                self.score_data_weight + self.score_write_time_weight)


@dataclass
class RectangleShot:
    """矩形 Shot 表示"""
    x: float            # 左下角 x 坐标 (nm)
    y: float            # 左下角 y 坐标 (nm)
    width: float        # 宽度 (nm)
    height: float       # 高度 (nm)

    @property
    def area_nm2(self) -> float:
        return self.width * self.height

    @property
    def perimeter_nm(self) -> float:
        return 2.0 * (self.width + self.height)


@dataclass
class ManufacturingCostResult:
    """
    掩模制造成本评估结果

    Attributes:
        vertex_count: 多边形总顶点数
        polygon_count: 多边形总数量
        shot_count: 分形后的矩形 Shot 数量
        shots: Shot 列表（按需返回，可能为 None）
        total_exposed_area_um2: 总曝光面积 (μm²)
        data_volume_mb: 预估数据体积 (MB)
        write_time_min: 预估写入时间 (min)
        write_breakdown: 写入时间明细
        complexity_score: 综合复杂度分数 (0~1, 越大越复杂/昂贵)
        relative_scores: 各分项相对分数
        cost_breakdown: 成本分项明细
    """
    vertex_count: int = 0
    polygon_count: int = 0
    shot_count: int = 0
    shots: Optional[List[RectangleShot]] = None
    total_exposed_area_um2: float = 0.0
    data_volume_mb: float = 0.0
    write_time_min: float = 0.0
    write_breakdown: Dict[str, float] = field(default_factory=dict)
    complexity_score: float = 0.0
    relative_scores: Dict[str, float] = field(default_factory=dict)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            'vertex_count': self.vertex_count,
            'polygon_count': self.polygon_count,
            'shot_count': self.shot_count,
            'total_exposed_area_um2': round(self.total_exposed_area_um2, 4),
            'data_volume_mb': round(self.data_volume_mb, 4),
            'write_time_min': round(self.write_time_min, 4),
            'write_breakdown': {k: round(v, 4) for k, v in self.write_breakdown.items()},
            'complexity_score': round(self.complexity_score, 6),
            'relative_scores': {k: round(v, 6) for k, v in self.relative_scores.items()},
            'cost_breakdown': {k: round(v, 6) for k, v in self.cost_breakdown.items()},
        }


# ============================================================================
# 辅助函数：从像素掩模提取轮廓与多边形
# ============================================================================

def _extract_contours(mask: np.ndarray,
                      pixel_size_nm: float = 1.0,
                      threshold: float = 0.5) -> List[np.ndarray]:
    """
    从二值掩模中提取轮廓多边形（Marching Squares 近似）

    使用 scipy.ndimage 的边缘检测 + 连通域分析估计顶点数。
    精确多边形提取使用 Moore 邻域追踪算法。

    Args:
        mask: 二值/连续掩模数组，(H, W)，值域[0,1]
        pixel_size_nm: 像素尺寸 (nm)
        threshold: 二值化阈值

    Returns:
        轮廓多边形列表，每个多边形为 (N, 2) 的 [x, y] 坐标数组（nm单位）
    """
    H, W = mask.shape
    binary = (mask >= threshold).astype(np.uint8)

    if np.sum(binary) == 0:
        return []

    labeled, num_features = label(binary, structure=generate_binary_structure(2, 2))

    contours: List[np.ndarray] = []
    objects = find_objects(labeled)

    for idx, obj in enumerate(objects):
        if obj is None:
            continue

        region = (labeled == (idx + 1)).astype(np.uint8)
        slices = obj
        y0, y1 = slices[0].start, slices[0].stop
        x0, x1 = slices[1].start, slices[1].stop

        region_crop = region[y0:y1, x0:x1]
        if region_crop.shape[0] < 2 or region_crop.shape[1] < 2:
            continue

        padded = np.pad(region_crop, pad_width=1, mode='constant', constant_values=0)

        edge = sobel(padded.astype(np.float64), axis=0) ** 2 + sobel(padded.astype(np.float64), axis=1) ** 2
        edge_points = np.argwhere(edge > 0.5)

        if len(edge_points) < 4:
            continue

        edge_points = edge_points[:, ::-1]  # (row,col) -> (x,y)
        edge_points = edge_points - 1 + np.array([x0, y0])
        edge_points_nm = edge_points.astype(np.float64) * pixel_size_nm

        simplified = _simplify_polygon(edge_points_nm, tolerance=pixel_size_nm * 0.5)
        if len(simplified) >= 3:
            contours.append(simplified)

    return contours


def _simplify_polygon(points: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    """
    使用距离阈值简化多边形（D-P 算法的简化近似版本）

    Args:
        points: (N, 2) 多边形顶点
        tolerance: 距离容差 (nm)

    Returns:
        简化后的多边形顶点
    """
    if len(points) < 3:
        return points

    kept = [points[0]]
    for i in range(1, len(points)):
        last = kept[-1]
        curr = points[i]
        dist = np.sqrt(np.sum((curr - last) ** 2))
        if dist >= tolerance:
            kept.append(curr)

    if len(kept) < 3:
        kept = points[::max(1, len(points) // 4)].tolist()

    result = np.array(kept, dtype=np.float64)
    if len(result) >= 3:
        first = result[0]
        last = result[-1]
        if np.sqrt(np.sum((first - last) ** 2)) < tolerance:
            result = result[:-1]

    return result


# ============================================================================
# 公共 API：顶点数估算
# ============================================================================

def estimate_vertex_count(mask_or_polygons: Union[np.ndarray, List[np.ndarray]],
                          pixel_size_nm: float = 1.0,
                          threshold: float = 0.5) -> int:
    """
    估算掩模多边形总顶点数

    Args:
        mask_or_polygons: 可以是
            - 像素掩模数组 (H, W) float/int
            - 多边形列表 [np.ndarray(N,2), ...]，坐标单位nm
        pixel_size_nm: 像素尺寸 (nm)，仅像素掩模时使用
        threshold: 二值化阈值，仅像素掩模时使用

    Returns:
        总顶点数
    """
    if isinstance(mask_or_polygons, np.ndarray):
        contours = _extract_contours(mask_or_polygons, pixel_size_nm, threshold)
        return sum(len(c) for c in contours)
    else:
        return sum(len(p) for p in mask_or_polygons)


# ============================================================================
# 公共 API：曼哈顿化
# ============================================================================

def manhattanize_polygon(polygon: np.ndarray,
                         strategy: ShotFracturingStrategy = ShotFracturingStrategy.HORIZONTAL_SWEEP,
                         min_size_nm: float = 2.0) -> List[RectangleShot]:
    """
    将任意多边形曼哈顿化，切分为轴对齐矩形（Shots）

    Args:
        polygon: (N, 2) 多边形顶点 [x, y]，单位 nm
        strategy: 分形策略
        min_size_nm: 最小矩形尺寸 (nm)

    Returns:
        矩形 Shot 列表
    """
    if len(polygon) < 3:
        return []

    if strategy == ShotFracturingStrategy.GRID_BASED:
        return _manhattanize_grid(polygon, min_size_nm)
    elif strategy == ShotFracturingStrategy.VERTICAL_SWEEP:
        return _manhattanize_sweep(polygon, min_size_nm, horizontal=False)
    else:
        return _manhattanize_sweep(polygon, min_size_nm, horizontal=True)


def _polygon_bbox(polygon: np.ndarray) -> Tuple[float, float, float, float]:
    xmin = float(np.min(polygon[:, 0]))
    ymin = float(np.min(polygon[:, 1]))
    xmax = float(np.max(polygon[:, 0]))
    ymax = float(np.max(polygon[:, 1]))
    return xmin, ymin, xmax, ymax


def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """射线法判断点是否在多边形内"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi):
            inside = not inside
        j = i
    return inside


def _manhattanize_sweep(polygon: np.ndarray,
                        min_size_nm: float,
                        horizontal: bool = True) -> List[RectangleShot]:
    """
    扫描线法曼哈顿化

    horizontal=True: 按水平线扫描，找到每行的连续区间，再纵向合并
    horizontal=False: 按垂直线扫描
    """
    xmin, ymin, xmax, ymax = _polygon_bbox(polygon)
    step = max(min_size_nm, 1.0)

    shots: List[RectangleShot] = []

    if horizontal:
        y_coords = np.arange(ymin, ymax + step, step)
        segments_by_row = []

        for y in y_coords:
            y_center = y + step / 2
            x_coords = np.arange(xmin, xmax + step, step)
            inside_mask = np.array(
                [_point_in_polygon(xc + step / 2, y_center, polygon) for xc in x_coords],
                dtype=bool
            )
            segments = _mask_to_segments(x_coords, inside_mask, step)
            segments_by_row.append((y, segments))

        shots = _merge_segments_vertical(segments_by_row, step)
    else:
        x_coords = np.arange(xmin, xmax + step, step)
        segments_by_col = []

        for x in x_coords:
            x_center = x + step / 2
            y_coords = np.arange(ymin, ymax + step, step)
            inside_mask = np.array(
                [_point_in_polygon(x_center, yc + step / 2, polygon) for yc in y_coords],
                dtype=bool
            )
            segments = _mask_to_segments(y_coords, inside_mask, step)
            segments_by_col.append((x, segments))

        shots = _merge_segments_horizontal(segments_by_col, step)

    return shots


def _mask_to_segments(coords: np.ndarray, inside_mask: np.ndarray, step: float) -> List[Tuple[float, float]]:
    """将布尔掩码转换为连续区间列表 [(start, end), ...]"""
    segments: List[Tuple[float, float]] = []
    n = len(inside_mask)
    i = 0
    while i < n:
        if inside_mask[i]:
            start = coords[i]
            j = i
            while j < n and inside_mask[j]:
                j += 1
            end = coords[j - 1] + step
            segments.append((start, end))
            i = j
        else:
            i += 1
    return segments


def _merge_segments_vertical(segments_by_row: List[Tuple[float, List[Tuple[float, float]]]],
                             step: float) -> List[RectangleShot]:
    """纵向合并相邻行的区间成矩形"""
    shots: List[RectangleShot] = []
    if not segments_by_row:
        return shots

    prev_row_y, prev_segments = segments_by_row[0]
    active_intervals: Dict[Tuple[float, float], float] = {}

    for (y, segments) in segments_by_row:
        matched_keys = set()
        for (x0, x1) in segments:
            key = None
            best_overlap = 0.0
            for (ax0, ax1) in active_intervals.keys():
                overlap_start = max(x0, ax0)
                overlap_end = min(x1, ax1)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    key = (ax0, ax1)
            if key is not None and best_overlap >= (x1 - x0) * 0.8:
                matched_keys.add(key)
            else:
                active_intervals[(x0, x1)] = y

        for key in list(active_intervals.keys()):
            if key not in matched_keys:
                start_y = active_intervals.pop(key)
                ax0, ax1 = key
                shots.append(RectangleShot(
                    x=ax0, y=start_y,
                    width=ax1 - ax0, height=y - start_y + step
                ))

    for key, start_y in active_intervals.items():
        ax0, ax1 = key
        last_y = segments_by_row[-1][0]
        shots.append(RectangleShot(
            x=ax0, y=start_y,
            width=ax1 - ax0, height=last_y - start_y + step
        ))

    return shots


def _merge_segments_horizontal(segments_by_col: List[Tuple[float, List[Tuple[float, float]]]],
                               step: float) -> List[RectangleShot]:
    """横向合并相邻列的区间成矩形"""
    shots: List[RectangleShot] = []
    if not segments_by_col:
        return shots

    active_intervals: Dict[Tuple[float, float], float] = {}

    for (x, segments) in segments_by_col:
        matched_keys = set()
        for (y0, y1) in segments:
            key = None
            best_overlap = 0.0
            for (ay0, ay1) in active_intervals.keys():
                overlap_start = max(y0, ay0)
                overlap_end = min(y1, ay1)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    key = (ay0, ay1)
            if key is not None and best_overlap >= (y1 - y0) * 0.8:
                matched_keys.add(key)
            else:
                active_intervals[(y0, y1)] = x

        for key in list(active_intervals.keys()):
            if key not in matched_keys:
                start_x = active_intervals.pop(key)
                ay0, ay1 = key
                shots.append(RectangleShot(
                    x=start_x, y=ay0,
                    width=x - start_x + step, height=ay1 - ay0
                ))

    for key, start_x in active_intervals.items():
        ay0, ay1 = key
        last_x = segments_by_col[-1][0]
        shots.append(RectangleShot(
            x=start_x, y=ay0,
            width=last_x - start_x + step, height=ay1 - ay0
        ))

    return shots


def _manhattanize_grid(polygon: np.ndarray, min_size_nm: float) -> List[RectangleShot]:
    """栅格化快速分形：直接按像素级别栅格生成矩形"""
    xmin, ymin, xmax, ymax = _polygon_bbox(polygon)
    step = max(min_size_nm, 1.0)

    x_coords = np.arange(xmin, xmax + step, step)
    y_coords = np.arange(ymin, ymax + step, step)

    nx = len(x_coords)
    ny = len(y_coords)

    if nx == 0 or ny == 0:
        return []

    mask = np.zeros((ny, nx), dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            xc = x_coords[ix] + step / 2
            yc = y_coords[iy] + step / 2
            mask[iy, ix] = _point_in_polygon(xc, yc, polygon)

    shots: List[RectangleShot] = []
    for iy in range(ny):
        ix = 0
        while ix < nx:
            if mask[iy, ix]:
                jx = ix
                while jx < nx and mask[iy, jx]:
                    jx += 1
                shots.append(RectangleShot(
                    x=x_coords[ix],
                    y=y_coords[iy],
                    width=(jx - ix) * step,
                    height=step,
                ))
                ix = jx
            else:
                ix += 1

    return shots


# ============================================================================
# 公共 API：Shot 分形与计数
# ============================================================================

def fracturing_to_shots(mask_or_polygons: Union[np.ndarray, List[np.ndarray]],
                        config: Optional[ManufacturingCostConfig] = None) -> Tuple[int, List[RectangleShot]]:
    """
    将掩模分形为矩形 Shots

    Args:
        mask_or_polygons: 像素掩模或多边形列表
        config: 成本评估配置，None 则使用默认

    Returns:
        (shot_count, shots_list)
    """
    if config is None:
        config = ManufacturingCostConfig()

    if isinstance(mask_or_polygons, np.ndarray):
        contours = _extract_contours(mask_or_polygons, config.pixel_size_nm, threshold=0.5)
    else:
        contours = mask_or_polygons

    all_shots: List[RectangleShot] = []
    for poly in contours:
        shots = manhattanize_polygon(
            poly,
            strategy=config.shot_fracturing,
            min_size_nm=config.min_shot_size_nm,
        )
        all_shots.extend(shots)

    return len(all_shots), all_shots


def estimate_shot_count(mask_or_polygons: Union[np.ndarray, List[np.ndarray]],
                        config: Optional[ManufacturingCostConfig] = None) -> int:
    """
    快速估算 Shot 数量（不保存完整列表）

    Args:
        mask_or_polygons: 像素掩模或多边形列表
        config: 成本评估配置

    Returns:
        预估 Shot 数量
    """
    if config is None:
        config = ManufacturingCostConfig()

    if isinstance(mask_or_polygons, np.ndarray):
        return _estimate_shot_count_from_mask(mask_or_polygons, config)
    else:
        count, _ = fracturing_to_shots(mask_or_polygons, config)
        return count


def _estimate_shot_count_from_mask(mask: np.ndarray,
                                   config: ManufacturingCostConfig) -> int:
    """
    从像素掩模快速估算 Shot 数量

    原理：统计边缘像素数，按 VSB 典型矩形覆盖面积估计。
    公式：N_shots ≈ (总曝光面积 / 平均shot面积) × 形状复杂度因子
    形状复杂度因子 = 周长² / (4π·面积) （类似圆度的倒数）
    """
    binary = (mask >= 0.5).astype(np.uint8)
    total_pixels = int(np.sum(binary))

    if total_pixels == 0:
        return 0

    pixel_area_nm2 = config.pixel_size_nm ** 2
    total_area_nm2 = total_pixels * pixel_area_nm2

    gy = np.zeros_like(binary, dtype=np.float64)
    gx = np.zeros_like(binary, dtype=np.float64)
    gy[:-1, :] = binary[1:, :].astype(np.float64) - binary[:-1, :].astype(np.float64)
    gx[:, :-1] = binary[:, 1:].astype(np.float64) - binary[:, :-1].astype(np.float64)
    edge_pixels = int(np.sum((np.abs(gx) + np.abs(gy)) > 0))
    perimeter_nm = edge_pixels * config.pixel_size_nm

    if total_area_nm2 < 1e-12:
        return 0

    circularity = 4.0 * np.pi * total_area_nm2 / max(perimeter_nm ** 2, 1e-12)
    complexity_factor = 1.0 / max(circularity, 0.05)

    min_shot_area = config.min_shot_size_nm ** 2
    max_shot_area = config.max_shot_size_nm ** 2
    avg_shot_area = np.sqrt(min_shot_area * max_shot_area)

    shot_count_naive = total_area_nm2 / avg_shot_area
    shot_count = int(np.ceil(shot_count_naive * complexity_factor * 0.7))

    return max(shot_count, 1)


# ============================================================================
# 公共 API：数据体积估算
# ============================================================================

def estimate_data_volume(vertex_count: int,
                         shot_count: int,
                         config: Optional[ManufacturingCostConfig] = None) -> float:
    """
    估算掩模版图文件数据体积 (MB)

    基于 GDSII/OASIS 文件格式的典型存储密度：
        GDSII: ~16 bytes / vertex
        OASIS: ~4 bytes / vertex
        + 矩形 shot 额外开销
        × 层次压缩因子

    Args:
        vertex_count: 总顶点数
        shot_count: 总 Shot 数
        config: 成本评估配置

    Returns:
        预估文件大小 (MB)
    """
    if config is None:
        config = ManufacturingCostConfig()

    if config.output_format == 'gds':
        bytes_per_vertex = config.avg_bytes_per_vertex_gds
        bytes_per_shot = 24.0
    else:
        bytes_per_vertex = config.avg_bytes_per_vertex_oasis
        bytes_per_shot = 8.0

    total_bytes = (vertex_count * bytes_per_vertex
                   + shot_count * bytes_per_shot)

    total_bytes *= config.hierarchy_factor

    header_overhead_bytes = 1024.0
    total_bytes += header_overhead_bytes

    return total_bytes / (1024.0 * 1024.0)


# ============================================================================
# 公共 API：写入时间估算
# ============================================================================

def estimate_write_time(shots: List[RectangleShot],
                        total_area_um2: float,
                        config: Optional[ManufacturingCostConfig] = None) -> Tuple[float, Dict[str, float]]:
    """
    预估掩模写入时间 (分钟)

    根据写入器类型采用不同模型：

    1. VSB E-Beam (典型 JEOL/NuFlare):
       T_write = Σ (Dose × Area_i / BeamCurrent)
              + N_shots × T_deflect_per_shot
              + T_stage_move

    2. Gaussian E-Beam:
       T_write = Total_area × Dose / BeamCurrent
              + N_pixels × T_blanking

    3. Optical (DUV/EUV ALTA / EMF):
       T_write = Total_area / (ScanSpeed × SwathWidth)
              + T_stage_settle

    Args:
        shots: 矩形 Shot 列表（可为空列表，用 total_area_um2 估算）
        total_area_um2: 总曝光面积 (μm²)
        config: 成本评估配置

    Returns:
        (total_time_min, breakdown_dict)
    """
    if config is None:
        config = ManufacturingCostConfig()

    breakdown: Dict[str, float] = {}
    shot_count = len(shots)

    if config.writer_type == MaskWriterType.VSB_EBEAM:
        dose_coul_cm2 = config.dose_uC_cm2 * 1e-6
        current_coul_s = config.ebeam_current_nA * 1e-9

        total_area_cm2 = total_area_um2 * 1e-8

        if shots and current_coul_s > 1e-18:
            exposure_time_s = 0.0
            for shot in shots:
                area_cm2 = shot.area_nm2 * 1e-14
                exposure_time_s += dose_coul_cm2 * area_cm2 / current_coul_s
        else:
            exposure_time_s = (dose_coul_cm2 * total_area_cm2 / max(current_coul_s, 1e-18))

        t_deflect_per_shot_s = 5e-7
        deflect_time_s = shot_count * t_deflect_per_shot_s

        area_cm2_sqrt = np.sqrt(max(total_area_cm2, 1e-18))
        stage_move_time_s = area_cm2_sqrt / (config.stage_move_speed_mm_s * 0.1)
        stage_move_time_s = max(stage_move_time_s, 1.0)

        total_s = exposure_time_s + deflect_time_s + stage_move_time_s

        breakdown['exposure_min'] = exposure_time_s / 60.0
        breakdown['deflection_min'] = deflect_time_s / 60.0
        breakdown['stage_move_min'] = stage_move_time_s / 60.0

    elif config.writer_type == MaskWriterType.GAUSSIAN_EBEAM:
        dose_coul_cm2 = config.dose_uC_cm2 * 1e-6
        current_coul_s = config.ebeam_current_nA * 1e-9
        total_area_cm2 = total_area_um2 * 1e-8

        exposure_time_s = dose_coul_cm2 * total_area_cm2 / max(current_coul_s, 1e-18)

        pixel_area_nm2 = config.pixel_size_nm ** 2
        total_area_nm2 = total_area_um2 * 1e6
        num_pixels = total_area_nm2 / max(pixel_area_nm2, 1e-12)
        blanking_time_s = num_pixels * 5e-9

        area_cm2_sqrt = np.sqrt(max(total_area_cm2, 1e-18))
        stage_move_time_s = area_cm2_sqrt / (config.stage_move_speed_mm_s * 0.1)
        stage_move_time_s = max(stage_move_time_s, 1.0)

        total_s = exposure_time_s + blanking_time_s + stage_move_time_s

        breakdown['exposure_min'] = exposure_time_s / 60.0
        breakdown['blanking_min'] = blanking_time_s / 60.0
        breakdown['stage_move_min'] = stage_move_time_s / 60.0

    elif config.writer_type in (MaskWriterType.DUV_OPTICAL, MaskWriterType.EUV_OPTICAL):
        total_area_m2 = total_area_um2 * 1e-12
        swath_width_m = 0.1
        scan_speed = config.optical_scan_speed_m_s

        scan_time_s = total_area_m2 / max(scan_speed * swath_width_m, 1e-18)

        area_m_sqrt = np.sqrt(max(total_area_m2, 1e-18))
        stage_settle_time_s = (area_m_sqrt / max(config.stage_move_speed_mm_s * 0.001, 1e-9)) * 2.0
        stage_settle_time_s = max(stage_settle_time_s, 10.0)

        total_s = scan_time_s + stage_settle_time_s

        breakdown['scan_min'] = scan_time_s / 60.0
        breakdown['stage_settle_min'] = stage_settle_time_s / 60.0

    else:
        total_s = 600.0
        breakdown['estimated_min'] = 10.0

    breakdown['total_min'] = total_s / 60.0
    return total_s / 60.0, breakdown


# ============================================================================
# 公共 API：复杂度分数
# ============================================================================

def compute_complexity_score(vertex_count: int,
                             shot_count: int,
                             data_volume_mb: float,
                             write_time_min: float,
                             config: Optional[ManufacturingCostConfig] = None) -> Tuple[float, Dict[str, float]]:
    """
    计算综合制造复杂度分数

    分项采用相对基准的对数缩放，压缩动态范围：
        ratio_i = max(value_i / baseline_i, 1e-6)
        rel_i = log10(1 + 9 * ratio_i)   # 映射到 [0, 1]
        score = Σ (w_i * rel_i) / Σ w_i

    Args:
        vertex_count: 顶点数
        shot_count: Shot 数
        data_volume_mb: 数据体积 (MB)
        write_time_min: 写入时间 (min)
        config: 配置，包含权重和基准值

    Returns:
        (score, relative_scores_dict)
    """
    if config is None:
        config = ManufacturingCostConfig()

    def _normalize(value: float, baseline: float) -> float:
        ratio = max(value / max(baseline, 1e-12), 1e-6)
        return np.log10(1.0 + 9.0 * ratio)

    rel_vertex = _normalize(float(vertex_count), config.baseline_vertex_count)
    rel_shot = _normalize(float(shot_count), config.baseline_shot_count)
    rel_data = _normalize(data_volume_mb, config.baseline_data_mb)
    rel_write = _normalize(write_time_min, config.baseline_write_time_min)

    w_sum = config.total_score_weight()
    if w_sum < 1e-12:
        w_sum = 1.0

    score = (config.score_vertex_weight * rel_vertex
             + config.score_shot_weight * rel_shot
             + config.score_data_weight * rel_data
             + config.score_write_time_weight * rel_write) / w_sum

    score = float(np.clip(score, 0.0, 1.0))

    relative = {
        'vertex': rel_vertex,
        'shot': rel_shot,
        'data': rel_data,
        'write_time': rel_write,
    }

    return score, relative


# ============================================================================
# 综合评估类
# ============================================================================

class MaskManufacturingCostEvaluator:
    """
    掩模制造成本综合评估器

    整合顶点数估算、Shot分形、数据体积估算、写入时间估算，
    输出综合复杂度分数。

    典型用法：
        >>> evaluator = MaskManufacturingCostEvaluator(config)
        >>> result = evaluator.evaluate(mask)
        >>> print(result.complexity_score)
    """

    def __init__(self, config: Optional[ManufacturingCostConfig] = None):
        self.config = config or ManufacturingCostConfig()

    def evaluate(self,
                 mask_or_polygons: Union[np.ndarray, List[np.ndarray]],
                 return_shots: bool = False) -> ManufacturingCostResult:
        """
        执行完整的成本评估

        Args:
            mask_or_polygons: 像素掩模 (H, W) 或多边形列表
            return_shots: 是否在结果中返回 Shot 列表（内存开销较大）

        Returns:
            ManufacturingCostResult
        """
        cfg = self.config
        result = ManufacturingCostResult()

        # 1. 提取多边形/轮廓
        if isinstance(mask_or_polygons, np.ndarray):
            contours = _extract_contours(mask_or_polygons, cfg.pixel_size_nm, threshold=0.5)
        else:
            contours = mask_or_polygons

        result.polygon_count = len(contours)

        # 2. 顶点数
        result.vertex_count = sum(len(c) for c in contours)

        # 3. Shot 分形
        if result.polygon_count > 0:
            all_shots: List[RectangleShot] = []
            for poly in contours:
                shots = manhattanize_polygon(
                    poly,
                    strategy=cfg.shot_fracturing,
                    min_size_nm=cfg.min_shot_size_nm,
                )
                all_shots.extend(shots)
            result.shot_count = len(all_shots)
            result.total_exposed_area_um2 = sum(
                s.area_nm2 for s in all_shots
            ) * 1e-6
        else:
            result.shot_count = 0
            if isinstance(mask_or_polygons, np.ndarray):
                binary = (mask_or_polygons >= 0.5)
                total_area_nm2 = float(np.sum(binary)) * (cfg.pixel_size_nm ** 2)
                result.total_exposed_area_um2 = total_area_nm2 * 1e-6
                result.shot_count = _estimate_shot_count_from_mask(mask_or_polygons, cfg)
            all_shots = []

        if return_shots:
            result.shots = all_shots

        # 4. 数据体积
        result.data_volume_mb = estimate_data_volume(
            result.vertex_count, result.shot_count, cfg
        )

        # 5. 写入时间
        result.write_time_min, result.write_breakdown = estimate_write_time(
            all_shots if (all_shots or return_shots) else [],
            result.total_exposed_area_um2, cfg
        )

        # 6. 复杂度分数
        result.complexity_score, result.relative_scores = compute_complexity_score(
            result.vertex_count,
            result.shot_count,
            result.data_volume_mb,
            result.write_time_min,
            cfg,
        )

        # 7. 成本分项（加权分数×总权重归一化）
        w_sum = cfg.total_score_weight() or 1.0
        result.cost_breakdown = {
            'vertex': result.relative_scores['vertex'] * cfg.score_vertex_weight / w_sum,
            'shot': result.relative_scores['shot'] * cfg.score_shot_weight / w_sum,
            'data': result.relative_scores['data'] * cfg.score_data_weight / w_sum,
            'write_time': result.relative_scores['write_time'] * cfg.score_write_time_weight / w_sum,
        }

        if cfg.verbose:
            logger.info(
                f"掩模制造成本评估完成: polygons={result.polygon_count}, "
                f"vertices={result.vertex_count}, shots={result.shot_count}, "
                f"area={result.total_exposed_area_um2:.2f}μm², "
                f"data={result.data_volume_mb:.3f}MB, "
                f"write_time={result.write_time_min:.2f}min, "
                f"score={result.complexity_score:.4f}"
            )

        return result

    def quick_estimate(self,
                       mask: np.ndarray) -> ManufacturingCostResult:
        """
        快速估算：不进行精确分形，使用统计公式近似

        适合在优化循环中每步调用，速度更快。
        """
        cfg = self.config
        result = ManufacturingCostResult()

        binary = (mask >= 0.5).astype(np.uint8)
        total_pixels = int(np.sum(binary))

        result.total_exposed_area_um2 = total_pixels * (cfg.pixel_size_nm ** 2) * 1e-6
        result.vertex_count = estimate_vertex_count(mask, cfg.pixel_size_nm)
        result.shot_count = _estimate_shot_count_from_mask(mask, cfg)
        result.polygon_count = max(int(label(binary)[1]), 1)

        result.data_volume_mb = estimate_data_volume(
            result.vertex_count, result.shot_count, cfg
        )

        result.write_time_min, result.write_breakdown = estimate_write_time(
            [], result.total_exposed_area_um2, cfg
        )

        result.complexity_score, result.relative_scores = compute_complexity_score(
            result.vertex_count,
            result.shot_count,
            result.data_volume_mb,
            result.write_time_min,
            cfg,
        )

        w_sum = cfg.total_score_weight() or 1.0
        result.cost_breakdown = {
            'vertex': result.relative_scores['vertex'] * cfg.score_vertex_weight / w_sum,
            'shot': result.relative_scores['shot'] * cfg.score_shot_weight / w_sum,
            'data': result.relative_scores['data'] * cfg.score_data_weight / w_sum,
            'write_time': result.relative_scores['write_time'] * cfg.score_write_time_weight / w_sum,
        }

        return result
