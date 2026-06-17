# -*- coding: utf-8 -*-
"""
OPC (Optical Proximity Correction) 工作流模块

实现标准的 OPC 流水线：
    输入原始版图 → 成像模拟 → 热点检测（高EPE区域）
    → SRAF插入/调整 → 迭代优化主特征与辅助特征 → 输出校正后掩模

主要组件：
    1. HotspotDetector: 热点检测器，基于 EPE 阈值标记需校正区域
    2. SRAFRuleEngine: SRAF 规则引擎，支持矩形/条形 SRAF 的自动生成
    3. OPCIterationController: OPC 迭代控制器，与 MaskOptimizer 解耦
    4. OPCWorkflow: 完整 OPC 工作流封装
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from scipy.ndimage import (
    distance_transform_edt, binary_dilation, binary_erosion,
    label, find_objects, generate_binary_structure
)
from scipy.spatial.distance import cdist

from core.imaging import (
    OpticalSystem, simulate_wafer_image,
    ProcessCondition
)
from core.litho_metrics import compute_epe, extract_edges
from core.metrics import edge_placement_error
from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig
from algorithms.callbacks import (
    WorkflowCheckpointManager, WorkflowCheckpointState,
)
from utils.config import load_config, save_config

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class HotspotRegion:
    """
    热点区域数据结构

    表示一个需要进行 OPC 校正的热点区域。

    Attributes:
        bbox: 边界框 (y_min, y_max, x_min, x_max)，像素坐标
        center: 区域中心 (y, x)
        epe_mean: 该区域平均 EPE (nm)
        epe_max: 该区域最大 EPE (nm)
        area: 区域面积（像素数）
        edge_type: 边缘类型 ('line_end', 'corner', 'inner_corner', 'general')
        priority: 校正优先级 (0-10，越高越紧急)
    """
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    epe_mean: float
    epe_max: float
    area: int
    edge_type: str = 'general'
    priority: float = 5.0

    @property
    def height(self) -> int:
        return self.bbox[1] - self.bbox[0]

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bbox': list(self.bbox),
            'center': list(self.center),
            'epe_mean': self.epe_mean,
            'epe_max': self.epe_max,
            'area': self.area,
            'edge_type': self.edge_type,
            'priority': self.priority,
        }


@dataclass
class SRAFFeature:
    """
    SRAF (Sub-Resolution Assist Feature) 特征数据结构

    Attributes:
        shape: 形状类型 ('rectangle', 'bar')
        position: 位置 (y_center, x_center)，像素坐标
        size: 尺寸 (height, width)，像素
        orientation: 方向角度（度），仅条形有效
        attached_to: 关联的主特征边缘类型 ('left', 'right', 'top', 'bottom', 'corner')
        is_enabled: 是否启用该 SRAF
    """
    shape: str
    position: Tuple[float, float]
    size: Tuple[float, float]
    orientation: float = 0.0
    attached_to: str = 'general'
    is_enabled: bool = True

    def get_mask_region(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        生成该 SRAF 对应的二值掩模区域

        Args:
            image_shape: 图像尺寸 (ny, nx)

        Returns:
            二值掩模，SRAF 区域为 1，其余为 0
        """
        mask = np.zeros(image_shape, dtype=np.float64)
        cy, cx = self.position
        h, w = self.size

        y_min = max(0, int(np.floor(cy - h / 2)))
        y_max = min(image_shape[0], int(np.ceil(cy + h / 2)))
        x_min = max(0, int(np.floor(cx - w / 2)))
        x_max = min(image_shape[1], int(np.ceil(cx + w / 2)))

        mask[y_min:y_max, x_min:x_max] = 1.0
        return mask

    def to_dict(self) -> Dict[str, Any]:
        return {
            'shape': self.shape,
            'position': list(self.position),
            'size': list(self.size),
            'orientation': self.orientation,
            'attached_to': self.attached_to,
            'is_enabled': self.is_enabled,
        }


class OPCTransformType(Enum):
    """OPC 变换类型枚举"""
    EDGE_OFFSET = 'edge_offset'           # 整体边缘偏移
    CORNER_BIAS = 'corner_bias'           # 拐角偏移（serif）
    LINE_END_EXTENSION = 'line_end'       # 线端延伸（hammerhead）
    SRAF_INSERT = 'sraf_insert'           # SRAF 插入
    SRAF_ADJUST = 'sraf_adjust'           # SRAF 调整


@dataclass
class OPCTransform:
    """
    OPC 变换操作

    表示对掩模的一个具体修改操作。

    Attributes:
        type: 变换类型
        region: 变换作用的区域 bbox (y_min, y_max, x_min, x_max)
        parameters: 变换参数字典
            - EDGE_OFFSET: {'offset': float, 'direction': 'inner'/'outer'}
            - CORNER_BIAS: {'bias': float, 'corner_type': 'inner'/'outer'}
            - LINE_END_EXTENSION: {'extension': float, 'width': float}
            - SRAF_INSERT: {'sraf': SRAFFeature}
            - SRAF_ADJUST: {'sraf_id': int, 'delta_size': (dy, dx), 'delta_pos': (dy, dx)}
    """
    type: OPCTransformType
    region: Tuple[int, int, int, int]
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OPCIterationResult:
    """
    OPC 单次迭代结果

    Attributes:
        iteration: 迭代次数
        mask_before: 迭代前的掩模
        mask_after: 迭代后的掩模
        wafer_before: 迭代前的晶圆成像（二值）
        wafer_after: 迭代后的晶圆成像（二值）
        epe_before: 迭代前 EPE 统计
        epe_after: 迭代后 EPE 统计
        hotspots_before: 迭代前检测到的热点
        hotspots_after: 迭代后剩余的热点
        transforms_applied: 本次应用的变换列表
        srafs_inserted: 本次插入的 SRAF 数量
        epe_improvement: EPE 改善量（epe_mean_before - epe_mean_after）
        epe_improvement_ratio: EPE 改善比例
    """
    iteration: int
    mask_before: np.ndarray
    mask_after: np.ndarray
    wafer_before: np.ndarray
    wafer_after: np.ndarray
    epe_before: Dict[str, float]
    epe_after: Dict[str, float]
    hotspots_before: List[HotspotRegion]
    hotspots_after: List[HotspotRegion]
    transforms_applied: List[OPCTransform] = field(default_factory=list)
    srafs_inserted: int = 0

    @property
    def epe_improvement(self) -> float:
        return self.epe_before['epe_mean'] - self.epe_after['epe_mean']

    @property
    def epe_improvement_ratio(self) -> float:
        if self.epe_before['epe_mean'] > 0:
            return self.epe_improvement / self.epe_before['epe_mean']
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'epe_before': self.epe_before,
            'epe_after': self.epe_after,
            'hotspots_before_count': len(self.hotspots_before),
            'hotspots_after_count': len(self.hotspots_after),
            'transforms_count': len(self.transforms_applied),
            'srafs_inserted': self.srafs_inserted,
            'epe_improvement': self.epe_improvement,
            'epe_improvement_ratio': self.epe_improvement_ratio,
        }


@dataclass
class OPCWorkflowResult:
    """
    OPC 工作流最终结果

    Attributes:
        initial_mask: 初始掩模
        corrected_mask: 校正后掩模
        initial_wafer: 初始晶圆成像
        corrected_wafer: 校正后晶圆成像
        initial_epe: 初始 EPE 统计
        final_epe: 最终 EPE 统计
        iterations: 所有迭代结果列表
        all_hotspots: 各轮热点检测结果
        all_srafs: 所有插入的 SRAF 特征
        total_epe_improvement: 总 EPE 改善量
        converged: 是否收敛
        reason: 终止原因
    """
    initial_mask: np.ndarray
    corrected_mask: np.ndarray
    initial_wafer: np.ndarray
    corrected_wafer: np.ndarray
    initial_epe: Dict[str, float]
    final_epe: Dict[str, float]
    iterations: List[OPCIterationResult] = field(default_factory=list)
    all_hotspots: List[List[HotspotRegion]] = field(default_factory=list)
    all_srafs: List[SRAFFeature] = field(default_factory=list)
    converged: bool = False
    reason: str = ''

    @property
    def total_epe_improvement(self) -> float:
        return self.initial_epe['epe_mean'] - self.final_epe['epe_mean']

    @property
    def total_epe_improvement_ratio(self) -> float:
        if self.initial_epe['epe_mean'] > 0:
            return self.total_epe_improvement / self.initial_epe['epe_mean']
        return 0.0

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    def summary(self) -> Dict[str, Any]:
        return {
            'initial_epe': self.initial_epe,
            'final_epe': self.final_epe,
            'total_epe_improvement': self.total_epe_improvement,
            'total_epe_improvement_ratio': self.total_epe_improvement_ratio,
            'num_iterations': self.num_iterations,
            'converged': self.converged,
            'reason': self.reason,
            'total_srafs_inserted': len(self.all_srafs),
            'initial_hotspot_count': len(self.all_hotspots[0]) if self.all_hotspots else 0,
            'final_hotspot_count': len(self.all_hotspots[-1]) if self.all_hotspots else 0,
        }


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class OPCConfig:
    """
    OPC 工作流配置

    Attributes:
        epe_threshold: EPE 热点判定阈值 (nm)
        epe_convergence_threshold: EPE 收敛阈值 (nm)
        max_iterations: 最大迭代次数
        min_hotspot_area: 最小热点区域面积（像素）
        hotspot_dilation: 热点区域膨胀像素数
        edge_offset_step: 每次边缘偏移步长（像素）
        max_edge_offset: 最大边缘偏移量（像素）
        corner_bias_size: 拐角 serif 尺寸（像素）
        line_end_extension: 线端延伸长度（像素）
        line_end_width: 线端延伸宽度（像素）

        sraf_enable: 是否启用 SRAF 插入
        sraf_min_distance: SRAF 与主特征最小间距（像素）
        sraf_max_distance: SRAF 与主特征最大间距（像素）
        sraf_width: SRAF 宽度（像素）
        sraf_length: SRAF 长度（像素）
        sraf_spacing: 相邻 SRAF 间距（像素）
        sraf_min_feature_size: SRAF 最小尺寸（像素）
        sraf_max_aspect_ratio: SRAF 最大长宽比

        optimizer_enable: 是否启用 MaskOptimizer 精细优化
        optimizer_max_iter: 优化器每轮最大迭代次数
        optimizer_learning_rate: 优化器学习率
        optimizer_epe_weight: 优化器 EPE 损失权重

        pixel_size: 像素尺寸 (nm)
        wafer_threshold: 晶圆成像二值化阈值
        verbose: 是否输出详细日志

        # Checkpoint 配置
        checkpoint_enable: 是否启用断点续跑功能
        checkpoint_dir: checkpoint 保存目录（None 则自动生成）
        checkpoint_save_freq: 迭代保存频率
        checkpoint_max_keep: 最多保留的 checkpoint 数量
        checkpoint_save_best_only: 是否只保存最优的 checkpoint
        checkpoint_force_restart: 是否忽略已有 checkpoint 强制重新开始
    """
    epe_threshold: float = 3.0
    epe_convergence_threshold: float = 1.0
    max_iterations: int = 10
    min_hotspot_area: int = 4
    hotspot_dilation: int = 2
    edge_offset_step: float = 0.5
    max_edge_offset: float = 3.0
    corner_bias_size: float = 1.0
    line_end_extension: float = 2.0
    line_end_width: float = 2.0

    sraf_enable: bool = True
    sraf_min_distance: float = 2.0
    sraf_max_distance: float = 5.0
    sraf_width: float = 1.0
    sraf_length: float = 4.0
    sraf_spacing: float = 2.0
    sraf_min_feature_size: float = 1.0
    sraf_max_aspect_ratio: float = 10.0

    optimizer_enable: bool = True
    optimizer_max_iter: int = 20
    optimizer_learning_rate: float = 0.05
    optimizer_epe_weight: float = 1.0

    pixel_size: float = 1.0
    wafer_threshold: float = 0.3
    verbose: bool = True

    # Checkpoint 配置
    checkpoint_enable: bool = True
    checkpoint_dir: Optional[str] = None
    checkpoint_save_freq: int = 1
    checkpoint_max_keep: int = 10
    checkpoint_save_best_only: bool = False
    checkpoint_force_restart: bool = False

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'OPCConfig':
        """从字典创建配置"""
        if d is None:
            return cls()
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'OPCConfig':
        """从 YAML 文件加载配置"""
        config_dict = load_config(config_path)
        opc_config = config_dict.get('opc', config_dict)
        return cls.from_dict(opc_config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epe_threshold': self.epe_threshold,
            'epe_convergence_threshold': self.epe_convergence_threshold,
            'max_iterations': self.max_iterations,
            'min_hotspot_area': self.min_hotspot_area,
            'hotspot_dilation': self.hotspot_dilation,
            'edge_offset_step': self.edge_offset_step,
            'max_edge_offset': self.max_edge_offset,
            'corner_bias_size': self.corner_bias_size,
            'line_end_extension': self.line_end_extension,
            'line_end_width': self.line_end_width,
            'sraf_enable': self.sraf_enable,
            'sraf_min_distance': self.sraf_min_distance,
            'sraf_max_distance': self.sraf_max_distance,
            'sraf_width': self.sraf_width,
            'sraf_length': self.sraf_length,
            'sraf_spacing': self.sraf_spacing,
            'sraf_min_feature_size': self.sraf_min_feature_size,
            'sraf_max_aspect_ratio': self.sraf_max_aspect_ratio,
            'optimizer_enable': self.optimizer_enable,
            'optimizer_max_iter': self.optimizer_max_iter,
            'optimizer_learning_rate': self.optimizer_learning_rate,
            'optimizer_epe_weight': self.optimizer_epe_weight,
            'pixel_size': self.pixel_size,
            'wafer_threshold': self.wafer_threshold,
            'verbose': self.verbose,
            'checkpoint_enable': self.checkpoint_enable,
            'checkpoint_dir': self.checkpoint_dir,
            'checkpoint_save_freq': self.checkpoint_save_freq,
            'checkpoint_max_keep': self.checkpoint_max_keep,
            'checkpoint_save_best_only': self.checkpoint_save_best_only,
            'checkpoint_force_restart': self.checkpoint_force_restart,
        }

    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """保存配置到 YAML 文件"""
        save_config({'opc': self.to_dict()}, config_path)


# ============================================================================
# 热点检测器
# ============================================================================

class HotspotDetector:
    """
    热点检测器

    对初始掩模成像后，按 EPE 阈值标记需要校正的区域。

    工作流程：
        1. 对掩模进行成像模拟，得到晶圆二值图
        2. 计算每个边缘像素的 EPE 距离场
        3. 对 EPE 超过阈值的像素进行连通域分析
        4. 标记每个连通域为一个热点区域，并计算其属性
    """

    def __init__(self, config: OPCConfig):
        """
        初始化热点检测器

        Args:
            config: OPC 配置
        """
        self.config = config

    def detect(self,
               mask: np.ndarray,
               target: np.ndarray,
               wafer_binary: Optional[np.ndarray] = None,
               optical_system: Optional[OpticalSystem] = None) -> List[HotspotRegion]:
        """
        检测热点区域

        Args:
            mask: 当前掩模图案
            target: 目标图案（原始版图）
            wafer_binary: 预计算的晶圆二值图，None 则重新模拟
            optical_system: 光学系统参数

        Returns:
            热点区域列表
        """
        if wafer_binary is None:
            wafer_continuous = simulate_wafer_image(
                mask,
                optical_system=optical_system,
                threshold=self.config.wafer_threshold,
                apply_resist=True
            )
            wafer_binary = (wafer_continuous >= self.config.wafer_threshold).astype(np.float64)

        epe_stats = compute_epe(
            wafer_binary, target,
            pixel_size=self.config.pixel_size
        )

        if self.config.verbose:
            logger.info(f"EPE 统计: mean={epe_stats['epe_mean']:.3f}nm, "
                       f"max={epe_stats['epe_max']:.3f}nm, "
                       f"std={epe_stats['epe_std']:.3f}nm")

        hotspots = self._identify_hotspots(wafer_binary, target)
        hotspots = self._prioritize_hotspots(hotspots)

        if self.config.verbose:
            logger.info(f"检测到 {len(hotspots)} 个热点区域")

        return hotspots

    def _identify_hotspots(self,
                          wafer_binary: np.ndarray,
                          target_binary: np.ndarray) -> List[HotspotRegion]:
        """
        识别热点区域

        Args:
            wafer_binary: 晶圆二值图
            target_binary: 目标二值图

        Returns:
            热点区域列表
        """
        wafer_edge = extract_edges(wafer_binary)
        target_edge = extract_edges(target_binary)

        if np.sum(target_edge) == 0 and np.sum(wafer_edge) == 0:
            return []

        dist_to_wafer = distance_transform_edt(1.0 - wafer_edge)
        dist_to_target = distance_transform_edt(1.0 - target_edge)

        epe_map = np.zeros_like(wafer_binary)
        wafer_mask = wafer_edge > 0.5
        target_mask = target_edge > 0.5
        epe_map[wafer_mask] = dist_to_target[wafer_mask]
        epe_map[target_mask] = np.maximum(epe_map[target_mask], dist_to_wafer[target_mask])

        epe_map_nm = epe_map * self.config.pixel_size
        hotspot_mask = epe_map_nm >= self.config.epe_threshold

        if self.config.hotspot_dilation > 0:
            struct = generate_binary_structure(2, 2)
            hotspot_mask = binary_dilation(hotspot_mask, struct, iterations=self.config.hotspot_dilation)

        labeled, num_features = label(hotspot_mask)

        hotspots = []
        if num_features == 0:
            return hotspots

        objects = find_objects(labeled)

        for idx, obj in enumerate(objects):
            if obj is None:
                continue

            region_mask = (labeled == (idx + 1))
            area = int(np.sum(region_mask))

            if area < self.config.min_hotspot_area:
                continue

            y_min, y_max = obj[0].start, obj[0].stop
            x_min, x_max = obj[1].start, obj[1].stop
            bbox = (y_min, y_max, x_min, x_max)

            region_epe = epe_map_nm[region_mask]
            epe_mean = float(np.mean(region_epe))
            epe_max = float(np.max(region_epe))

            cy = (y_min + y_max) / 2.0
            cx = (x_min + x_max) / 2.0
            center = (cy, cx)

            edge_type = self._classify_edge_type(target_binary, bbox)

            hotspot = HotspotRegion(
                bbox=bbox,
                center=center,
                epe_mean=epe_mean,
                epe_max=epe_max,
                area=area,
                edge_type=edge_type
            )
            hotspots.append(hotspot)

        return hotspots

    def _classify_edge_type(self,
                           target: np.ndarray,
                           bbox: Tuple[int, int, int, int]) -> str:
        """
        分类热点区域的边缘类型

        Args:
            target: 目标图案
            bbox: 热点区域边界框

        Returns:
            边缘类型字符串
        """
        y_min, y_max, x_min, x_max = bbox
        region = target[y_min:y_max, x_min:x_max]

        if region.size == 0:
            return 'general'

        target_edge = extract_edges(region)
        edge_pixels = np.argwhere(target_edge > 0.5)

        if len(edge_pixels) < 4:
            return 'general'

        center = np.mean(edge_pixels, axis=0)
        centered = edge_pixels - center
        cov = np.cov(centered.T)

        if cov.size == 1:
            eigenvalues = [cov[0, 0]]
        else:
            eigenvalues = np.linalg.eigvalsh(cov)

        if len(eigenvalues) < 2:
            return 'general'

        ratio = eigenvalues[1] / (eigenvalues[0] + 1e-8)

        region_h, region_w = region.shape
        is_line_end = False
        is_corner = False

        if region_h > 0 and region_w > 0:
            edge_on_top = np.sum(target_edge[0, :]) > 0
            edge_on_bottom = np.sum(target_edge[-1, :]) > 0
            edge_on_left = np.sum(target_edge[:, 0]) > 0
            edge_on_right = np.sum(target_edge[:, -1]) > 0

            edges_on_boundary = sum([edge_on_top, edge_on_bottom, edge_on_left, edge_on_right])
            if edges_on_boundary >= 2:
                is_corner = True
            elif edges_on_boundary == 1:
                is_line_end = True

        if is_corner:
            return 'corner'
        elif is_line_end:
            return 'line_end'
        elif ratio < 0.3:
            return 'general'
        else:
            return 'general'

    def _prioritize_hotspots(self, hotspots: List[HotspotRegion]) -> List[HotspotRegion]:
        """
        为热点分配优先级并排序

        Args:
            hotspots: 热点列表

        Returns:
            按优先级排序的热点列表
        """
        type_weight = {
            'line_end': 10.0,
            'corner': 8.0,
            'inner_corner': 9.0,
            'general': 5.0,
        }

        for hotspot in hotspots:
            epe_score = min(10.0, hotspot.epe_max / self.config.epe_threshold * 5.0)
            type_score = type_weight.get(hotspot.edge_type, 5.0)
            area_score = min(5.0, np.log10(hotspot.area + 1) * 2.0)
            hotspot.priority = 0.5 * epe_score + 0.3 * type_score + 0.2 * area_score

        hotspots.sort(key=lambda h: h.priority, reverse=True)
        return hotspots


# ============================================================================
# SRAF 规则引擎
# ============================================================================

class SRAFRuleEngine:
    """
    SRAF (Sub-Resolution Assist Feature) 规则引擎

    支持矩形/条形 SRAF 的自动生成与放置规则，包括：
    - 间距约束（与主特征最小/最大间距）
    - 尺寸约束（最小尺寸、长宽比）
    - 方向约束（与主特征边缘平行）

    典型 SRAF 放置策略：
        1. 在孤立线的两侧放置条形 SRAF
        2. 在接触孔周围放置矩形 SRAF
        3. 在线端外侧放置条形 SRAF
        4. 在拐角外侧放置 L 形 SRAF（简化为两个条形）
    """

    def __init__(self, config: OPCConfig):
        """
        初始化 SRAF 规则引擎

        Args:
            config: OPC 配置
        """
        self.config = config

    def generate_srafs(self,
                      mask: np.ndarray,
                      target: np.ndarray,
                      hotspots: List[HotspotRegion]) -> List[SRAFFeature]:
        """
        为热点区域生成 SRAF

        Args:
            mask: 当前掩模
            target: 目标图案
            hotspots: 热点区域列表

        Returns:
            生成的 SRAF 特征列表
        """
        if not self.config.sraf_enable:
            return []

        srafs = []
        used_positions = set()

        target_edge = extract_edges(target)
        distance_to_edge = distance_transform_edt(1.0 - target_edge)

        for hotspot in hotspots:
            hotspot_srafs = self._generate_for_hotspot(
                mask, target, hotspot, distance_to_edge, used_positions
            )
            srafs.extend(hotspot_srafs)

        srafs = self._resolve_conflicts(srafs, mask)
        srafs = self._apply_drc(srafs, mask)

        if self.config.verbose:
            logger.info(f"生成 {len(srafs)} 个 SRAF 特征")

        return srafs

    def _generate_for_hotspot(self,
                              mask: np.ndarray,
                              target: np.ndarray,
                              hotspot: HotspotRegion,
                              distance_to_edge: np.ndarray,
                              used_positions: set) -> List[SRAFFeature]:
        """
        为单个热点生成 SRAF

        Args:
            mask: 当前掩模
            target: 目标图案
            hotspot: 热点区域
            distance_to_edge: 到目标边缘的距离场
            used_positions: 已使用的位置集合

        Returns:
            该热点对应的 SRAF 列表
        """
        srafs = []
        y_min, y_max, x_min, x_max = hotspot.bbox

        expand = int(np.ceil(self.config.sraf_max_distance + self.config.sraf_length))
        y_min_e = max(0, y_min - expand)
        y_max_e = min(mask.shape[0], y_max + expand)
        x_min_e = max(0, x_min - expand)
        x_max_e = min(mask.shape[1], x_max + expand)

        local_target = target[y_min_e:y_max_e, x_min_e:x_max_e]
        local_edge = extract_edges(local_target)

        edge_points = np.argwhere(local_edge > 0.5)
        if len(edge_points) == 0:
            return srafs

        if len(edge_points) >= 2:
            center_local = np.mean(edge_points, axis=0)
            centered = edge_points - center_local
            cov = np.cov(centered.T)
            if cov.size > 1 and np.linalg.det(cov) > 1e-8:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                primary_axis = eigenvectors[:, np.argmax(eigenvalues)]
                orientation = np.degrees(np.arctan2(primary_axis[0], primary_axis[1]))
            else:
                orientation = 0.0
        else:
            orientation = 0.0

        edge_types = self._detect_edge_types(local_target, local_edge)

        for edge_type, edge_points_sub in edge_types.items():
            if len(edge_points_sub) == 0:
                continue

            for ep in edge_points_sub:
                global_y = ep[0] + y_min_e
                global_x = ep[1] + x_min_e

                pos_key = (int(global_y // self.config.sraf_width),
                          int(global_x // self.config.sraf_width))
                if pos_key in used_positions:
                    continue

                sraf = self._place_sraf(
                    mask, target, (global_y, global_x),
                    edge_type, orientation, distance_to_edge
                )
                if sraf is not None:
                    srafs.append(sraf)
                    used_positions.add(pos_key)

        return srafs

    def _detect_edge_types(self,
                          local_target: np.ndarray,
                          local_edge: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        检测局部边缘类型

        Args:
            local_target: 局部目标区域
            local_edge: 局部边缘图

        Returns:
            边缘类型到点列表的映射
        """
        h, w = local_target.shape
        edge_points = np.argwhere(local_edge > 0.5)

        edge_types = {
            'top': [],
            'bottom': [],
            'left': [],
            'right': [],
            'corner': [],
        }

        for (y, x) in edge_points:
            neighbors = []
            if y > 0:
                neighbors.append(local_target[y-1, x])
            if y < h-1:
                neighbors.append(local_target[y+1, x])
            if x > 0:
                neighbors.append(local_target[y, x-1])
            if x < w-1:
                neighbors.append(local_target[y, x+1])

            is_inside = local_target[y, x] >= 0.5

            if y > 0 and y < h-1 and x > 0 and x < w-1:
                diagonal_sum = (local_target[y-1, x-1] + local_target[y-1, x+1] +
                               local_target[y+1, x-1] + local_target[y+1, x+1])
                if 1 <= diagonal_sum <= 3:
                    edge_types['corner'].append((y, x))
                    continue

            if y > 0 and local_target[y-1, x] != is_inside:
                edge_types['top' if not is_inside else 'bottom'].append((y, x))
            elif y < h-1 and local_target[y+1, x] != is_inside:
                edge_types['bottom' if not is_inside else 'top'].append((y, x))
            elif x > 0 and local_target[y, x-1] != is_inside:
                edge_types['left' if not is_inside else 'right'].append((y, x))
            elif x < w-1 and local_target[y, x+1] != is_inside:
                edge_types['right' if not is_inside else 'left'].append((y, x))

        return edge_types

    def _place_sraf(self,
                   mask: np.ndarray,
                   target: np.ndarray,
                   edge_point: Tuple[int, int],
                   edge_type: str,
                   orientation: float,
                   distance_to_edge: np.ndarray) -> Optional[SRAFFeature]:
        """
        在指定边缘点放置 SRAF

        Args:
            mask: 掩模
            target: 目标图案
            edge_point: 边缘点坐标 (y, x)
            edge_type: 边缘类型
            orientation: 边缘方向（度）
            distance_to_edge: 到边缘的距离场

        Returns:
            SRAF 特征，若无法放置则返回 None
        """
        y, x = edge_point

        direction_map = {
            'top': (-1, 0),
            'bottom': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
        }

        if edge_type in direction_map:
            dy, dx = direction_map[edge_type]
        elif edge_type == 'corner':
            dy, dx = -1, -1
        else:
            return None

        min_dist_px = self.config.sraf_min_distance
        max_dist_px = self.config.sraf_max_distance

        sraf_center_y = y + dy * (min_dist_px + max_dist_px) / 2
        sraf_center_x = x + dx * (min_dist_px + max_dist_px) / 2

        sraf_center_y = int(np.round(sraf_center_y))
        sraf_center_x = int(np.round(sraf_center_x))

        if (sraf_center_y < 0 or sraf_center_y >= mask.shape[0] or
            sraf_center_x < 0 or sraf_center_x >= mask.shape[1]):
            return None

        actual_dist = np.sqrt((sraf_center_y - y)**2 + (sraf_center_x - x)**2)
        if actual_dist < min_dist_px or actual_dist > max_dist_px:
            return None

        if target[sraf_center_y, sraf_center_x] >= 0.5:
            return None

        if mask[sraf_center_y, sraf_center_x] >= 0.5:
            return None

        sraf_width = self.config.sraf_width
        sraf_length = self.config.sraf_length

        if edge_type in ('top', 'bottom'):
            size = (sraf_width, sraf_length)
            shape = 'bar'
        elif edge_type in ('left', 'right'):
            size = (sraf_length, sraf_width)
            shape = 'bar'
        else:
            size = (sraf_width, sraf_width)
            shape = 'rectangle'

        if (min(size) < self.config.sraf_min_feature_size or
            max(size) / max(min(size), 1e-8) > self.config.sraf_max_aspect_ratio):
            return None

        sraf = SRAFFeature(
            shape=shape,
            position=(sraf_center_y, sraf_center_x),
            size=size,
            orientation=orientation if shape == 'bar' else 0.0,
            attached_to=edge_type
        )

        return sraf

    def _resolve_conflicts(self,
                          srafs: List[SRAFFeature],
                          mask: np.ndarray) -> List[SRAFFeature]:
        """
        解决 SRAF 之间的冲突（重叠或过于接近）

        Args:
            srafs: SRAF 列表
            mask: 掩模

        Returns:
            去冲突后的 SRAF 列表
        """
        if len(srafs) <= 1:
            return srafs

        positions = np.array([s.position for s in srafs])
        sizes = np.array([s.size for s in srafs])

        dist_matrix = cdist(positions, positions)

        min_dist = self.config.sraf_spacing

        to_remove = set()
        for i in range(len(srafs)):
            if i in to_remove:
                continue
            for j in range(i+1, len(srafs)):
                if j in to_remove:
                    continue

                min_required = (max(sizes[i]) + max(sizes[j])) / 2 + min_dist
                if dist_matrix[i, j] < min_required:
                    to_remove.add(j)

        return [srafs[i] for i in range(len(srafs)) if i not in to_remove]

    def _apply_drc(self,
                  srafs: List[SRAFFeature],
                  mask: np.ndarray) -> List[SRAFFeature]:
        """
        应用设计规则检查（DRC）

        Args:
            srafs: SRAF 列表
            mask: 掩模

        Returns:
            通过 DRC 的 SRAF 列表
        """
        valid_srafs = []
        mask_shape = mask.shape

        for sraf in srafs:
            cy, cx = sraf.position
            h, w = sraf.size

            y_min = int(np.floor(cy - h / 2))
            y_max = int(np.ceil(cy + h / 2))
            x_min = int(np.floor(cx - w / 2))
            x_max = int(np.ceil(cx + w / 2))

            if (y_min < 0 or y_max > mask_shape[0] or
                x_min < 0 or x_max > mask_shape[1]):
                continue

            sraf_region = mask[y_min:y_max, x_min:x_max]
            if np.any(sraf_region >= 0.5):
                continue

            valid_srafs.append(sraf)

        return valid_srafs

    def insert_srafs(self, mask: np.ndarray, srafs: List[SRAFFeature]) -> np.ndarray:
        """
        将 SRAF 插入到掩模中

        Args:
            mask: 原始掩模
            srafs: SRAF 特征列表

        Returns:
            插入 SRAF 后的掩模
        """
        new_mask = mask.copy()

        for sraf in srafs:
            if not sraf.is_enabled:
                continue

            sraf_mask = sraf.get_mask_region(mask.shape)
            new_mask = np.maximum(new_mask, sraf_mask)

        if self.config.verbose:
            logger.info(f"已插入 {len([s for s in srafs if s.is_enabled])} 个 SRAF")

        return new_mask


# ============================================================================
# OPC 变换应用器
# ============================================================================

class OPCTransformApplier:
    """
    OPC 变换应用器

    将 OPCTransform 应用到掩模上，执行实际的几何修改。
    """

    def __init__(self, config: OPCConfig):
        """
        初始化变换应用器

        Args:
            config: OPC 配置
        """
        self.config = config

    def apply_transform(self,
                       mask: np.ndarray,
                       target: np.ndarray,
                       transform: OPCTransform) -> np.ndarray:
        """
        应用单个变换

        Args:
            mask: 当前掩模
            target: 目标图案
            transform: 要应用的变换

        Returns:
            修改后的掩模
        """
        y_min, y_max, x_min, x_max = transform.region

        local_mask = mask[y_min:y_max, x_min:x_max].copy()
        local_target = target[y_min:y_max, x_min:x_max].copy()

        if transform.type == OPCTransformType.EDGE_OFFSET:
            result = self._apply_edge_offset(
                local_mask, local_target, transform.parameters
            )
        elif transform.type == OPCTransformType.CORNER_BIAS:
            result = self._apply_corner_bias(
                local_mask, local_target, transform.parameters
            )
        elif transform.type == OPCTransformType.LINE_END_EXTENSION:
            result = self._apply_line_end_extension(
                local_mask, local_target, transform.parameters
            )
        else:
            result = local_mask

        new_mask = mask.copy()
        new_mask[y_min:y_max, x_min:x_max] = result
        return new_mask

    def _apply_edge_offset(self,
                          local_mask: np.ndarray,
                          local_target: np.ndarray,
                          params: Dict[str, Any]) -> np.ndarray:
        """
        应用边缘偏移变换

        Args:
            local_mask: 局部掩模
            local_target: 局部目标
            params: 参数字典

        Returns:
            修改后的局部掩模
        """
        offset = params.get('offset', self.config.edge_offset_step)
        direction = params.get('direction', 'outer')

        offset = np.clip(offset, -self.config.max_edge_offset, self.config.max_edge_offset)

        if direction == 'outer':
            struct = generate_binary_structure(2, 2)
            iterations = int(np.ceil(abs(offset)))
            if offset > 0:
                return binary_dilation(local_mask >= 0.5, struct, iterations=iterations).astype(np.float64)
            else:
                return binary_erosion(local_mask >= 0.5, struct, iterations=iterations).astype(np.float64)
        else:
            struct = generate_binary_structure(2, 2)
            iterations = int(np.ceil(abs(offset)))
            if offset > 0:
                return binary_erosion(local_mask >= 0.5, struct, iterations=iterations).astype(np.float64)
            else:
                return binary_dilation(local_mask >= 0.5, struct, iterations=iterations).astype(np.float64)

    def _apply_corner_bias(self,
                          local_mask: np.ndarray,
                          local_target: np.ndarray,
                          params: Dict[str, Any]) -> np.ndarray:
        """
        应用拐角 bias（serif）变换

        Args:
            local_mask: 局部掩模
            local_target: 局部目标
            params: 参数字典

        Returns:
            修改后的局部掩模
        """
        bias = params.get('bias', self.config.corner_bias_size)
        corner_type = params.get('corner_type', 'outer')

        result = local_mask.copy()
        h, w = local_mask.shape

        edge = extract_edges(local_target)
        edge_points = np.argwhere(edge > 0.5)

        for (y, x) in edge_points:
            is_corner = False
            if y > 0 and y < h-1 and x > 0 and x < w-1:
                diag_sum = (local_target[y-1, x-1] + local_target[y-1, x+1] +
                           local_target[y+1, x-1] + local_target[y+1, x+1])
                if 1 <= diag_sum <= 3:
                    is_corner = True

            if is_corner:
                bias_size = int(np.ceil(bias))
                if corner_type == 'outer':
                    for dy in range(-bias_size, bias_size + 1):
                        for dx in range(-bias_size, bias_size + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if dy*dy + dx*dx <= bias_size*bias_size:
                                    result[ny, nx] = 1.0
                else:
                    for dy in range(-bias_size, bias_size + 1):
                        for dx in range(-bias_size, bias_size + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if dy*dy + dx*dx <= bias_size*bias_size:
                                    result[ny, nx] = 0.0

        return result

    def _apply_line_end_extension(self,
                                   local_mask: np.ndarray,
                                   local_target: np.ndarray,
                                   params: Dict[str, Any]) -> np.ndarray:
        """
        应用线端延伸（hammerhead）变换

        Args:
            local_mask: 局部掩模
            local_target: 局部目标
            params: 参数字典

        Returns:
            修改后的局部掩模
        """
        extension = params.get('extension', self.config.line_end_extension)
        width = params.get('width', self.config.line_end_width)

        result = local_mask.copy()
        h, w = local_mask.shape

        edge = extract_edges(local_target)
        edge_points = np.argwhere(edge > 0.5)

        if len(edge_points) < 2:
            return result

        directions = []
        for (y, x) in edge_points:
            on_top = (y == 0) or (local_target[y-1, x] < 0.5 and local_target[y, x] >= 0.5)
            on_bottom = (y == h-1) or (local_target[y+1, x] < 0.5 and local_target[y, x] >= 0.5)
            on_left = (x == 0) or (local_target[y, x-1] < 0.5 and local_target[y, x] >= 0.5)
            on_right = (x == w-1) or (local_target[y, x+1] < 0.5 and local_target[y, x] >= 0.5)

            boundary_count = sum([on_top, on_bottom, on_left, on_right])
            if boundary_count == 1:
                if on_top:
                    directions.append((y, x, -1, 0))
                elif on_bottom:
                    directions.append((y, x, 1, 0))
                elif on_left:
                    directions.append((y, x, 0, -1))
                elif on_right:
                    directions.append((y, x, 0, 1))

        ext_len = int(np.ceil(extension))
        ext_width = int(np.ceil(width))

        for (y, x, dy, dx) in directions:
            for i in range(1, ext_len + 1):
                ny, nx = y + dy * i, x + dx * i
                if 0 <= ny < h and 0 <= nx < w:
                    result[ny, nx] = 1.0

                    if dx != 0:
                        for j in range(-ext_width, ext_width + 1):
                            wy = ny + j
                            if 0 <= wy < h:
                                result[wy, nx] = 1.0
                    if dy != 0:
                        for j in range(-ext_width, ext_width + 1):
                            wx = nx + j
                            if 0 <= wx < w:
                                result[ny, wx] = 1.0

        return result


# ============================================================================
# OPC 迭代控制器
# ============================================================================

class OPCIterationController:
    """
    OPC 迭代控制器

    与 MaskOptimizer 解耦，按"检测 → 修正 → 验证"循环执行，
    并记录每轮 EPE 改善量。

    迭代流程：
        1. 检测：对当前掩模成像，计算 EPE，识别热点
        2. 修正：
           a. 基于规则的修正（边缘偏移、拐角 bias、线端延伸）
           b. SRAF 插入/调整
           c. MaskOptimizer 精细优化（可选）
        3. 验证：重新成像，计算 EPE 改善
        4. 收敛检查：若 EPE 低于阈值或无改善则终止
    """

    def __init__(self,
                 config: OPCConfig,
                 optical_system: Optional[OpticalSystem] = None,
                 optimizer: Optional[MaskOptimizer] = None):
        """
        初始化迭代控制器

        Args:
            config: OPC 配置
            optical_system: 光学系统参数
            optimizer: 掩模优化器（可选）
        """
        self.config = config
        self.optical_system = optical_system or OpticalSystem()
        self.optimizer = optimizer

        self.hotspot_detector = HotspotDetector(config)
        self.sraf_engine = SRAFRuleEngine(config)
        self.transform_applier = OPCTransformApplier(config)

    def run_iteration(self,
                     current_mask: np.ndarray,
                     target: np.ndarray,
                     iteration: int,
                     existing_srafs: Optional[List[SRAFFeature]] = None) -> OPCIterationResult:
        """
        执行一次 OPC 迭代

        Args:
            current_mask: 当前掩模
            target: 目标图案
            iteration: 迭代次数
            existing_srafs: 已存在的 SRAF 列表

        Returns:
            迭代结果
        """
        if self.config.verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"OPC 迭代 {iteration}")
            logger.info(f"{'='*60}")

        wafer_before_cont = simulate_wafer_image(
            current_mask,
            optical_system=self.optical_system,
            threshold=self.config.wafer_threshold,
            apply_resist=True
        )
        wafer_before = (wafer_before_cont >= self.config.wafer_threshold).astype(np.float64)

        epe_before = compute_epe(
            wafer_before, target,
            pixel_size=self.config.pixel_size
        )

        hotspots_before = self.hotspot_detector.detect(
            current_mask, target,
            wafer_binary=wafer_before,
            optical_system=self.optical_system
        )

        mask_corrected = current_mask.copy()
        transforms_applied = []
        srafs_inserted = []

        if len(hotspots_before) > 0:
            transforms = self._generate_transforms(hotspots_before, target)
            for transform in transforms:
                mask_corrected = self.transform_applier.apply_transform(
                    mask_corrected, target, transform
                )
                transforms_applied.append(transform)

        if self.config.sraf_enable and len(hotspots_before) > 0:
            new_srafs = self.sraf_engine.generate_srafs(
                mask_corrected, target, hotspots_before
            )
            if existing_srafs:
                all_srafs = existing_srafs + new_srafs
                all_srafs = self.sraf_engine._resolve_conflicts(all_srafs, mask_corrected)
            else:
                all_srafs = new_srafs

            mask_corrected = self.sraf_engine.insert_srafs(mask_corrected, all_srafs)
            srafs_inserted = new_srafs

        if self.config.optimizer_enable and self.optimizer is not None:
            if self.config.verbose:
                logger.info("执行 MaskOptimizer 精细优化...")
            mask_corrected = self._run_optimizer(mask_corrected, target)

        wafer_after_cont = simulate_wafer_image(
            mask_corrected,
            optical_system=self.optical_system,
            threshold=self.config.wafer_threshold,
            apply_resist=True
        )
        wafer_after = (wafer_after_cont >= self.config.wafer_threshold).astype(np.float64)

        epe_after = compute_epe(
            wafer_after, target,
            pixel_size=self.config.pixel_size
        )

        hotspots_after = self.hotspot_detector.detect(
            mask_corrected, target,
            wafer_binary=wafer_after,
            optical_system=self.optical_system
        )

        result = OPCIterationResult(
            iteration=iteration,
            mask_before=current_mask,
            mask_after=mask_corrected,
            wafer_before=wafer_before,
            wafer_after=wafer_after,
            epe_before=epe_before,
            epe_after=epe_after,
            hotspots_before=hotspots_before,
            hotspots_after=hotspots_after,
            transforms_applied=transforms_applied,
            srafs_inserted=len(srafs_inserted)
        )

        if self.config.verbose:
            logger.info(f"迭代 {iteration} 完成: "
                       f"EPE {epe_before['epe_mean']:.3f} → {epe_after['epe_mean']:.3f} nm "
                       f"(改善 {result.epe_improvement:.3f} nm, {result.epe_improvement_ratio*100:.1f}%)")
            logger.info(f"热点数量: {len(hotspots_before)} → {len(hotspots_after)}")

        return result

    def _generate_transforms(self,
                            hotspots: List[HotspotRegion],
                            target: np.ndarray) -> List[OPCTransform]:
        """
        为热点生成 OPC 变换列表

        Args:
            hotspots: 热点区域列表
            target: 目标图案

        Returns:
            变换列表
        """
        transforms = []

        for hotspot in hotspots:
            bbox = hotspot.bbox

            edge_offset_transform = OPCTransform(
                type=OPCTransformType.EDGE_OFFSET,
                region=bbox,
                parameters={
                    'offset': self.config.edge_offset_step,
                    'direction': 'outer'
                }
            )
            transforms.append(edge_offset_transform)

            if hotspot.edge_type in ('corner', 'inner_corner'):
                corner_transform = OPCTransform(
                    type=OPCTransformType.CORNER_BIAS,
                    region=bbox,
                    parameters={
                        'bias': self.config.corner_bias_size,
                        'corner_type': 'outer'
                    }
                )
                transforms.append(corner_transform)

            if hotspot.edge_type == 'line_end':
                line_end_transform = OPCTransform(
                    type=OPCTransformType.LINE_END_EXTENSION,
                    region=bbox,
                    parameters={
                        'extension': self.config.line_end_extension,
                        'width': self.config.line_end_width
                    }
                )
                transforms.append(line_end_transform)

        return transforms

    def _run_optimizer(self,
                      initial_mask: np.ndarray,
                      target: np.ndarray) -> np.ndarray:
        """
        运行 MaskOptimizer 进行精细优化

        Args:
            initial_mask: 初始掩模
            target: 目标图案

        Returns:
            优化后的掩模
        """
        if self.optimizer is None:
            opt_config = OptimizationConfig(
                optimizer_type='gradient_descent',
                max_iter=self.config.optimizer_max_iter,
                learning_rate=self.config.optimizer_learning_rate,
                use_composite_loss=True,
                loss_weights=type('LossWeights', (), {
                    'mse': 0.0, 'ssim': 0.0, 'pvb': 0.0,
                    'mask_complexity': 0.0, 'binary_penalty': 0.0,
                    'tv_smooth': 0.0, 'epe': self.config.optimizer_epe_weight,
                    'min_feature': 0.0, 'weighted_mse': 0.0,
                    'weighted_mae': 0.0,
                })(),
                pixel_size=self.config.pixel_size,
                verbose=False
            )
            optimizer = MaskOptimizer(
                optical_system=self.optical_system,
                config=opt_config
            )
        else:
            optimizer = self.optimizer

        try:
            result = optimizer.optimize(initial_mask, target)
            return result.optimal_mask
        except Exception as e:
            logger.warning(f"MaskOptimizer 优化失败，跳过: {e}")
            return initial_mask

    def check_convergence(self,
                         iteration_result: OPCIterationResult,
                         prev_result: Optional[OPCIterationResult]) -> Tuple[bool, str]:
        """
        检查收敛条件

        Args:
            iteration_result: 当前迭代结果
            prev_result: 上一轮迭代结果（可选）

        Returns:
            (是否收敛, 终止原因)
        """
        epe_mean = iteration_result.epe_after['epe_mean']
        if epe_mean <= self.config.epe_convergence_threshold:
            return True, f"EPE 已收敛到阈值以下: {epe_mean:.3f} nm <= {self.config.epe_convergence_threshold} nm"

        if len(iteration_result.hotspots_after) == 0:
            return True, "所有热点已消除"

        if prev_result is not None:
            prev_epe = prev_result.epe_after['epe_mean']
            improvement = prev_epe - epe_mean
            if improvement < self.config.epe_convergence_threshold * 0.1:
                return True, f"EPE 改善停滞: {improvement:.4f} nm < 阈值"

            if iteration_result.iteration >= self.config.max_iterations:
                return True, f"达到最大迭代次数: {self.config.max_iterations}"

        return False, ""


# ============================================================================
# OPC 工作流主类
# ============================================================================

class OPCWorkflow:
    """
    OPC 完整工作流（支持断点续跑）

    封装标准 OPC 流水线：
        输入原始版图（目标图案）
        → 基于当前成像模型识别热点（高 EPE 区域）
        → 在热点周围自动插入或调整 SRAF
        → 迭代优化主特征与辅助特征
        → 输出校正后掩模

    使用示例：
        workflow = OPCWorkflow(config, optical_system)
        result = workflow.run(initial_mask, target_pattern)
    """

    def __init__(self,
                 config: Optional[OPCConfig] = None,
                 optical_system: Optional[OpticalSystem] = None,
                 optimizer: Optional[MaskOptimizer] = None):
        """
        初始化 OPC 工作流

        Args:
            config: OPC 配置，None 则使用默认配置
            optical_system: 光学系统参数，None 则使用默认参数
            optimizer: 掩模优化器，None 则根据配置自动创建
        """
        self.config = config or OPCConfig()
        self.optical_system = optical_system or OpticalSystem()

        self.controller = OPCIterationController(
            config=self.config,
            optical_system=self.optical_system,
            optimizer=optimizer
        )

        if self.config.verbose:
            logger.info("OPC 工作流已初始化")
            logger.info(f"配置: EPE 阈值={self.config.epe_threshold}nm, "
                       f"最大迭代={self.config.max_iterations}, "
                       f"SRAF={'启用' if self.config.sraf_enable else '禁用'}"
                       f", Checkpoint={'启用' if self.config.checkpoint_enable else '禁用'}")

    def _init_checkpoint_manager(self,
                                  initial_mask: np.ndarray,
                                  target: np.ndarray) -> Optional[WorkflowCheckpointManager]:
        """初始化 checkpoint 管理器"""
        if not self.config.checkpoint_enable:
            return None

        cfg = self.config
        if cfg.checkpoint_dir:
            checkpoint_dir = Path(cfg.checkpoint_dir)
        else:
            import hashlib
            mask_hash = hashlib.md5(initial_mask.tobytes()).hexdigest()[:8]
            target_hash = hashlib.md5(target.tobytes()).hexdigest()[:8]
            base_dir = Path('./checkpoints')
            checkpoint_dir = base_dir / f'opc_{mask_hash}_{target_hash}'

        return WorkflowCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            workflow_type='OPC',
            save_freq_outer=cfg.checkpoint_save_freq,
            max_checkpoints=cfg.checkpoint_max_keep,
            save_best_only=cfg.checkpoint_save_best_only,
            filename_prefix='opc',
            config=cfg,
        )

    def _try_restore_from_checkpoint(
        self,
        ckpt_mgr: WorkflowCheckpointManager,
    ) -> Optional[Tuple[int, np.ndarray, List[OPCIterationResult],
                         List[List[HotspotRegion]], List[SRAFFeature],
                         Optional[OPCIterationResult], float, float]]:
        """
        尝试从最近的 checkpoint 恢复 OPC 工作流状态

        Returns:
            (start_iter, current_mask, iterations, all_hotspots, all_srafs,
             prev_result, best_epe, best_loss)
            恢复失败返回 None
        """
        if self.config.checkpoint_force_restart:
            logger.info("checkpoint_force_restart=True，忽略已有 checkpoint，从头开始")
            return None

        latest_path = ckpt_mgr.find_latest_checkpoint(validate_config=True)
        if latest_path is None:
            logger.info("未找到可恢复的 checkpoint，从头开始运行")
            return None

        try:
            state = WorkflowCheckpointState.load(latest_path)
            state.restore_random_state()

            current_mask = state.mask
            if current_mask is None:
                logger.warning("  checkpoint 中未找到掩模，恢复失败")
                return None

            start_iter = int(state.outer_iteration)
            iterations: List[OPCIterationResult] = list(
                state.extra_data.get('iterations', [])
            )
            all_hotspots: List[List[HotspotRegion]] = list(
                state.extra_data.get('all_hotspots', [])
            )
            all_srafs: List[SRAFFeature] = list(
                state.extra_data.get('all_srafs', [])
            )
            prev_result: Optional[OPCIterationResult] = state.extra_data.get('prev_result', None)
            best_epe: float = float(state.extra_data.get('best_epe', float('inf')))
            best_loss: float = state.best_loss

            if state.config_hash and state.config_hash != ckpt_mgr.config_hash:
                logger.warning(
                    f"  配置哈希与 checkpoint 不一致，可能导致结果偏差 "
                    f"(期望 {ckpt_mgr.config_hash[:8]}..., checkpoint {state.config_hash[:8]}...)"
                )

            logger.info(
                f"成功从 checkpoint 恢复: 迭代={start_iter}, "
                f"已执行迭代={len(iterations)}, SRAF数量={len(all_srafs)}, "
                f"best_epe={best_epe:.3f}nm"
            )
            return (start_iter, current_mask, iterations, all_hotspots,
                    all_srafs, prev_result, best_epe, best_loss)

        except Exception as e:
            logger.warning(f"恢复 checkpoint 失败，将从头开始: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def run(self,
           initial_mask: np.ndarray,
           target: np.ndarray) -> OPCWorkflowResult:
        """
        运行完整的 OPC 工作流（支持断点续跑）

        Args:
            initial_mask: 初始掩模（通常与目标相同）
            target: 目标图案（原始版图）

        Returns:
            OPC 工作流结果
        """
        if self.config.verbose:
            logger.info("\n" + "="*60)
            logger.info("开始 OPC 工作流")
            logger.info("="*60)
            logger.info(f"初始掩模尺寸: {initial_mask.shape}")

        ckpt_mgr = self._init_checkpoint_manager(initial_mask, target)

        initial_wafer_cont = simulate_wafer_image(
            initial_mask,
            optical_system=self.optical_system,
            threshold=self.config.wafer_threshold,
            apply_resist=True
        )
        initial_wafer = (initial_wafer_cont >= self.config.wafer_threshold).astype(np.float64)

        initial_epe = compute_epe(
            initial_wafer, target,
            pixel_size=self.config.pixel_size
        )

        if self.config.verbose:
            logger.info(f"初始 EPE: mean={initial_epe['epe_mean']:.3f}nm, "
                       f"max={initial_epe['epe_max']:.3f}nm")

        # —— 尝试从 checkpoint 恢复 ——
        start_iter = 0
        best_epe = initial_epe['epe_mean']
        best_loss = initial_epe['epe_mean']
        restored = None
        if ckpt_mgr is not None:
            restored = self._try_restore_from_checkpoint(ckpt_mgr)

        if restored is not None:
            (start_iter, current_mask, iterations, all_hotspots,
             all_srafs, prev_result, best_epe, best_loss) = restored
        else:
            current_mask = initial_mask.copy()
            iterations: List[OPCIterationResult] = []
            all_hotspots: List[List[HotspotRegion]] = []
            all_srafs: List[SRAFFeature] = []
            prev_result = None

        converged = False
        reason = ""

        if not all_hotspots:
            initial_hotspots = self.controller.hotspot_detector.detect(
                current_mask, target,
                wafer_binary=initial_wafer if start_iter == 0 else None,
                optical_system=self.optical_system
            )
            all_hotspots.append(initial_hotspots)

        if start_iter > 0 and self.config.verbose:
            logger.info(f"\n从 checkpoint 恢复后继续: 当前迭代={start_iter}, "
                       f"best_epe={best_epe:.3f}nm")

        for iteration in range(max(1, start_iter + 1), self.config.max_iterations + 1):
            iter_result = self.controller.run_iteration(
                current_mask, target, iteration,
                existing_srafs=all_srafs
            )

            iterations.append(iter_result)
            all_hotspots.append(iter_result.hotspots_after)

            if iter_result.srafs_inserted > 0:
                new_srafs = self.controller.sraf_engine.generate_srafs(
                    iter_result.mask_after, target, iter_result.hotspots_before
                )
                all_srafs.extend(new_srafs)

            # —— 更新 best 状态 ——
            current_epe = iter_result.epe_after['epe_mean']
            if current_epe < best_epe:
                best_epe = current_epe
                best_loss = current_epe

            # —— 保存 checkpoint ——
            if ckpt_mgr is not None:
                ckpt_state = WorkflowCheckpointState(
                    workflow_type='OPC',
                    outer_iteration=iteration,
                    inner_iteration=0,
                    current_phase='opc_iteration',
                    source=None,
                    mask=iter_result.mask_after.copy(),
                    best_loss=best_loss,
                    best_mask=iter_result.mask_after.copy() if current_epe <= best_epe else None,
                    best_source=None,
                    loss_history=[float(r.epe_after.get('epe_mean', 0)) for r in iterations],
                    loss_components_history=[],
                    extra_data={
                        'iterations': iterations,
                        'all_hotspots': all_hotspots,
                        'all_srafs': all_srafs,
                        'prev_result': prev_result,
                        'best_epe': best_epe,
                    },
                )
                ckpt_mgr.save_checkpoint(
                    ckpt_state,
                    outer_iteration=iteration,
                    current_loss=current_epe,
                )

            converged, reason = self.controller.check_convergence(
                iter_result, prev_result
            )

            current_mask = iter_result.mask_after
            prev_result = iter_result

            if converged:
                if self.config.verbose:
                    logger.info(f"\n收敛: {reason}")
                break

        if not converged:
            reason = f"达到最大迭代次数 {self.config.max_iterations}"

        final_wafer_cont = simulate_wafer_image(
            current_mask,
            optical_system=self.optical_system,
            threshold=self.config.wafer_threshold,
            apply_resist=True
        )
        final_wafer = (final_wafer_cont >= self.config.wafer_threshold).astype(np.float64)

        final_epe = compute_epe(
            final_wafer, target,
            pixel_size=self.config.pixel_size
        )

        result = OPCWorkflowResult(
            initial_mask=initial_mask,
            corrected_mask=current_mask,
            initial_wafer=initial_wafer,
            corrected_wafer=final_wafer,
            initial_epe=initial_epe,
            final_epe=final_epe,
            iterations=iterations,
            all_hotspots=all_hotspots,
            all_srafs=all_srafs,
            converged=converged,
            reason=reason
        )

        if self.config.verbose:
            logger.info("\n" + "="*60)
            logger.info("OPC 工作流完成")
            logger.info("="*60)
            logger.info(f"迭代次数: {result.num_iterations}")
            logger.info(f"收敛: {'是' if result.converged else '否'}")
            logger.info(f"原因: {result.reason}")
            logger.info(f"初始 EPE: {initial_epe['epe_mean']:.3f} nm")
            logger.info(f"最终 EPE: {final_epe['epe_mean']:.3f} nm")
            logger.info(f"总改善: {result.total_epe_improvement:.3f} nm "
                       f"({result.total_epe_improvement_ratio*100:.1f}%)")
            logger.info(f"SRAF 数量: {len(all_srafs)}")
            logger.info(f"热点数量: {len(initial_hotspots)} → {len(all_hotspots[-1]) if all_hotspots else 0}")

        return result


# ============================================================================
# 便捷函数
# ============================================================================

def run_opc_workflow(initial_mask: np.ndarray,
                     target: np.ndarray,
                     config: Optional[Union[OPCConfig, str, Path]] = None,
                     optical_system: Optional[OpticalSystem] = None,
                     optimizer: Optional[MaskOptimizer] = None) -> OPCWorkflowResult:
    """
    便捷函数：运行 OPC 工作流

    Args:
        initial_mask: 初始掩模
        target: 目标图案
        config: OPC 配置，可以是 OPCConfig 对象、配置文件路径或 None
        optical_system: 光学系统参数
        optimizer: 掩模优化器

    Returns:
        OPC 工作流结果
    """
    if config is None:
        opc_config = OPCConfig()
    elif isinstance(config, (str, Path)):
        opc_config = OPCConfig.from_yaml(config)
    else:
        opc_config = config

    workflow = OPCWorkflow(
        config=opc_config,
        optical_system=optical_system,
        optimizer=optimizer
    )

    return workflow.run(initial_mask, target)
