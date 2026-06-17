# -*- coding: utf-8 -*-
"""
OPC + ILT 混合精修工作流模块

实现 OPC 与 ILT 的混合精修模式：
    OPC 完成 SRAF 放置与粗校正后，仅对热点 bbox 区域启动 ILT 可微优化，
    全局非热点保持 OPC 结果不变，兼顾计算效率与局部精度。

核心组件：
    1. HotspotBBoxManager: 热点 bbox 管理器，负责膨胀、合并、裁剪与拼合
    2. LocalILTOptimizer: 局部 ILT 优化器，对单个热点 bbox 区域进行可微优化
    3. HybridOPCILTConfig: 混合工作流配置
    4. HybridOPCILTWorkflow: 混合工作流主类
    5. run_hybrid_opc_ilt_workflow: 便捷入口函数

工作流程：
    阶段 1: 全局 OPC 粗校正（SRAF 插入 + 规则修正）
    阶段 2: 热点检测与 bbox 提取（膨胀、合并重叠区域）
    阶段 3: 逐热点局部 ILT 可微优化（带羽化边界过渡）
    阶段 4: 结果拼合（非热点保持 OPC 结果不变）
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path
from scipy.ndimage import (
    binary_dilation, binary_erosion, label, find_objects,
    generate_binary_structure, gaussian_filter
)

from core.imaging import OpticalSystem
from core.litho_metrics import compute_epe

from workflows.opc import (
    OPCConfig, OPCWorkflow, OPCWorkflowResult,
    HotspotDetector, HotspotRegion, run_opc_workflow
)
from workflows.ilt import (
    ILTConfig, ILTWorkflowResult, ILTWorkflow,
    DifferentiableImagingChain, GradientProjector,
    MaskComplexityPenalty, run_ilt_workflow
)
from utils.config import load_config, save_config

logger = logging.getLogger(__name__)


@dataclass
class HybridOPCILTConfig:
    """
    OPC + ILT 混合精修配置

    Attributes:
        opc_config: OPC 阶段配置
        ilt_config: ILT 阶段配置（用于局部优化）

        hotspot_bbox_padding: 热点 bbox 外扩像素数，确保优化区域覆盖完整
        hotspot_merge_overlap: 热点合并重叠阈值（像素），小于该距离的热点合并
        max_hotspots: 最大优化热点数量，超过则按优先级取前 N 个
        min_hotspot_size: 最小热点尺寸（像素），太小的热点跳过

        feather_width: 羽化边界宽度（像素），用于 ILT 区域与 OPC 区域平滑过渡
        use_gaussian_feather: 是否使用高斯羽化（True）或线性羽化（False）

        run_global_opc: 是否先运行全局 OPC（False 则直接从 initial_mask 开始）
        run_local_ilt: 是否运行局部 ILT（False 则仅输出 OPC 结果）

        pixel_size: 像素尺寸 (nm)
        verbose: 是否输出详细日志
    """
    opc_config: Optional[OPCConfig] = None
    ilt_config: Optional[ILTConfig] = None

    hotspot_bbox_padding: int = 8
    hotspot_merge_overlap: int = 4
    max_hotspots: int = 20
    min_hotspot_size: int = 4

    feather_width: int = 4
    use_gaussian_feather: bool = True

    run_global_opc: bool = True
    run_local_ilt: bool = True

    pixel_size: float = 1.0
    verbose: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'HybridOPCILTConfig':
        if d is None:
            return cls()
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                if key == 'opc_config':
                    cfg.opc_config = OPCConfig.from_dict(value)
                elif key == 'ilt_config':
                    cfg.ilt_config = ILTConfig.from_dict(value)
                else:
                    setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'HybridOPCILTConfig':
        config_dict = load_config(config_path)
        hybrid_config = config_dict.get('hybrid_opc_ilt', config_dict)
        return cls.from_dict(hybrid_config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'opc_config': self.opc_config.to_dict() if self.opc_config else None,
            'ilt_config': self.ilt_config.to_dict() if self.ilt_config else None,
            'hotspot_bbox_padding': self.hotspot_bbox_padding,
            'hotspot_merge_overlap': self.hotspot_merge_overlap,
            'max_hotspots': self.max_hotspots,
            'min_hotspot_size': self.min_hotspot_size,
            'feather_width': self.feather_width,
            'use_gaussian_feather': self.use_gaussian_feather,
            'run_global_opc': self.run_global_opc,
            'run_local_ilt': self.run_local_ilt,
            'pixel_size': self.pixel_size,
            'verbose': self.verbose,
        }

    def to_yaml(self, config_path: Union[str, Path]) -> None:
        save_config({'hybrid_opc_ilt': self.to_dict()}, config_path)


@dataclass
class LocalILTResult:
    """
    单个热点区域的局部 ILT 优化结果

    Attributes:
        hotspot_idx: 热点索引
        bbox: 优化区域 bbox (y_min, y_max, x_min, x_max)
        initial_mask_local: 初始局部掩模
        optimal_mask_local: 优化后局部掩模
        initial_epe: 初始 EPE 统计
        final_epe: 最终 EPE 统计
        ilt_result: 完整的 ILT 工作流结果
        feather_weight: 羽化权重图（与局部掩模同尺寸）
    """
    hotspot_idx: int
    bbox: Tuple[int, int, int, int]
    initial_mask_local: np.ndarray
    optimal_mask_local: np.ndarray
    initial_epe: Dict[str, float]
    final_epe: Dict[str, float]
    ilt_result: Optional[ILTWorkflowResult] = None
    feather_weight: Optional[np.ndarray] = None

    @property
    def epe_improvement(self) -> float:
        return self.initial_epe.get('epe_mean', 0.0) - self.final_epe.get('epe_mean', 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hotspot_idx': self.hotspot_idx,
            'bbox': list(self.bbox),
            'initial_epe': self.initial_epe,
            'final_epe': self.final_epe,
            'epe_improvement': self.epe_improvement,
        }


@dataclass
class HybridOPCILTWorkflowResult:
    """
    OPC + ILT 混合工作流最终结果

    Attributes:
        initial_mask: 初始掩模
        opc_mask: OPC 校正后掩模
        final_mask: 混合精修后最终掩模
        target: 目标图案

        initial_epe: 初始 EPE 统计
        opc_epe: OPC 后 EPE 统计
        final_epe: 最终 EPE 统计

        opc_result: OPC 工作流完整结果
        local_ilt_results: 各热点局部 ILT 优化结果列表

        merged_hotspots: 合并后的热点区域列表
        num_hotspots_optimized: 实际优化的热点数量

        converged: 是否收敛（所有热点均改善）
        reason: 终止原因

        total_time: 总耗时（秒）
        opc_time: OPC 阶段耗时（秒）
        ilt_time: ILT 阶段总耗时（秒）
    """
    initial_mask: np.ndarray
    opc_mask: np.ndarray
    final_mask: np.ndarray
    target: np.ndarray

    initial_epe: Dict[str, float]
    opc_epe: Dict[str, float]
    final_epe: Dict[str, float]

    opc_result: Optional[OPCWorkflowResult] = None
    local_ilt_results: List[LocalILTResult] = field(default_factory=list)

    merged_hotspots: List[HotspotRegion] = field(default_factory=list)
    num_hotspots_optimized: int = 0

    converged: bool = False
    reason: str = ''

    total_time: float = 0.0
    opc_time: float = 0.0
    ilt_time: float = 0.0

    @property
    def opc_epe_improvement(self) -> float:
        return self.initial_epe.get('epe_mean', 0.0) - self.opc_epe.get('epe_mean', 0.0)

    @property
    def ilt_epe_improvement(self) -> float:
        return self.opc_epe.get('epe_mean', 0.0) - self.final_epe.get('epe_mean', 0.0)

    @property
    def total_epe_improvement(self) -> float:
        return self.initial_epe.get('epe_mean', 0.0) - self.final_epe.get('epe_mean', 0.0)

    @property
    def total_epe_improvement_ratio(self) -> float:
        init_mean = self.initial_epe.get('epe_mean', 0.0)
        if init_mean > 1e-12:
            return self.total_epe_improvement / init_mean
        return 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            'initial_epe': self.initial_epe,
            'opc_epe': self.opc_epe,
            'final_epe': self.final_epe,
            'opc_epe_improvement': self.opc_epe_improvement,
            'ilt_epe_improvement': self.ilt_epe_improvement,
            'total_epe_improvement': self.total_epe_improvement,
            'total_epe_improvement_ratio': self.total_epe_improvement_ratio,
            'num_hotspots_detected': len(self.merged_hotspots),
            'num_hotspots_optimized': self.num_hotspots_optimized,
            'converged': self.converged,
            'reason': self.reason,
            'total_time': round(self.total_time, 3),
            'opc_time': round(self.opc_time, 3),
            'ilt_time': round(self.ilt_time, 3),
        }


class HotspotBBoxManager:
    """
    热点 bbox 管理器

    负责热点区域的：
    1. 膨胀（padding）：确保优化区域覆盖完整的边缘影响范围
    2. 合并：重叠或邻近的热点合并为一个大的优化区域
    3. 裁剪：从全局掩模中裁剪出局部区域
    4. 拼合：将优化后的局部区域拼合回全局掩模（带羽化过渡）
    """

    def __init__(self, config: HybridOPCILTConfig):
        """
        初始化 bbox 管理器

        Args:
            config: 混合工作流配置
        """
        self.config = config

    def expand_bbox(self,
                    bbox: Tuple[int, int, int, int],
                    image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        膨胀 bbox，外扩指定像素数

        Args:
            bbox: 原始 bbox (y_min, y_max, x_min, x_max)
            image_shape: 图像尺寸 (ny, nx)

        Returns:
            膨胀后的 bbox
        """
        y_min, y_max, x_min, x_max = bbox
        pad = self.config.hotspot_bbox_padding
        ny, nx = image_shape

        y_min = max(0, y_min - pad)
        y_max = min(ny, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(nx, x_max + pad)

        return (y_min, y_max, x_min, x_max)

    def bboxes_overlap(self,
                       bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> bool:
        """
        检查两个 bbox 是否重叠或邻近

        Args:
            bbox1: 第一个 bbox
            bbox2: 第二个 bbox

        Returns:
            True 表示重叠或邻近
        """
        y1_min, y1_max, x1_min, x1_max = bbox1
        y2_min, y2_max, x2_min, x2_max = bbox2
        tol = self.config.hotspot_merge_overlap

        return not (y1_max + tol < y2_min or y2_max + tol < y1_min or
                    x1_max + tol < x2_min or x2_max + tol < x1_min)

    def merge_bboxes(self,
                     bbox1: Tuple[int, int, int, int],
                     bbox2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        合并两个 bbox

        Args:
            bbox1: 第一个 bbox
            bbox2: 第二个 bbox

        Returns:
            合并后的 bbox
        """
        y_min = min(bbox1[0], bbox2[0])
        y_max = max(bbox1[1], bbox2[1])
        x_min = min(bbox1[2], bbox2[2])
        x_max = max(bbox1[3], bbox2[3])
        return (y_min, y_max, x_min, x_max)

    def merge_hotspots(self,
                       hotspots: List[HotspotRegion],
                       image_shape: Tuple[int, int]) -> List[HotspotRegion]:
        """
        合并重叠或邻近的热点区域

        Args:
            hotspots: 原始热点列表
            image_shape: 图像尺寸

        Returns:
            合并后的热点列表
        """
        if len(hotspots) <= 1:
            return hotspots

        expanded = [self.expand_bbox(h.bbox, image_shape) for h in hotspots]
        merged_indices = list(range(len(hotspots)))

        for i in range(len(hotspots)):
            if merged_indices[i] != i:
                continue
            for j in range(i + 1, len(hotspots)):
                if self.bboxes_overlap(expanded[i], expanded[j]):
                    root_i = self._find_root(merged_indices, i)
                    root_j = self._find_root(merged_indices, j)
                    if root_i != root_j:
                        merged_indices[root_j] = root_i
                        expanded[root_i] = self.merge_bboxes(expanded[root_i], expanded[root_j])

        groups: Dict[int, List[int]] = {}
        for i in range(len(hotspots)):
            root = self._find_root(merged_indices, i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        merged_hotspots = []
        for root, indices in groups.items():
            group_hotspots = [hotspots[i] for i in indices]
            merged_bbox = expanded[root]

            max_epe = max(h.epe_max for h in group_hotspots)
            mean_epe = float(np.mean([h.epe_mean for h in group_hotspots]))
            total_area = sum(h.area for h in group_hotspots)
            max_priority = max(h.priority for h in group_hotspots)

            cy = (merged_bbox[0] + merged_bbox[1]) / 2.0
            cx = (merged_bbox[2] + merged_bbox[3]) / 2.0

            edge_types = [h.edge_type for h in group_hotspots]
            if 'corner' in edge_types:
                edge_type = 'corner'
            elif 'line_end' in edge_types:
                edge_type = 'line_end'
            else:
                edge_type = 'general'

            merged = HotspotRegion(
                bbox=merged_bbox,
                center=(cy, cx),
                epe_mean=mean_epe,
                epe_max=max_epe,
                area=total_area,
                edge_type=edge_type,
                priority=max_priority
            )
            merged_hotspots.append(merged)

        merged_hotspots.sort(key=lambda h: h.priority, reverse=True)
        return merged_hotspots

    def _find_root(self, parent: List[int], idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def filter_hotspots(self, hotspots: List[HotspotRegion]) -> List[HotspotRegion]:
        """
        过滤热点：按尺寸过滤，按数量限制

        Args:
            hotspots: 热点列表

        Returns:
            过滤后的热点列表
        """
        filtered = []
        for h in hotspots:
            h_px = h.height
            w_px = h.width
            if h_px >= self.config.min_hotspot_size and w_px >= self.config.min_hotspot_size:
                filtered.append(h)

        if len(filtered) > self.config.max_hotspots:
            filtered = filtered[:self.config.max_hotspots]

        return filtered

    def crop_local(self,
                   mask: np.ndarray,
                   target: np.ndarray,
                   bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        裁剪局部区域

        Args:
            mask: 全局掩模
            target: 全局目标图案
            bbox: 裁剪区域

        Returns:
            (局部掩模, 局部目标)
        """
        y_min, y_max, x_min, x_max = bbox
        return mask[y_min:y_max, x_min:x_max].copy(), target[y_min:y_max, x_min:x_max].copy()

    def create_feather_weight(self, bbox_shape: Tuple[int, int]) -> np.ndarray:
        """
        创建羽化权重图

        权重从中心 1.0 向边界渐变到 0.0，用于平滑拼合。

        Args:
            bbox_shape: bbox 尺寸 (ny, nx)

        Returns:
            权重图，值范围 [0, 1]
        """
        ny, nx = bbox_shape
        fw = self.config.feather_width

        weight = np.ones((ny, nx), dtype=np.float64)

        if fw <= 0:
            return weight

        if self.config.use_gaussian_feather:
            sigma = fw / 3.0
            gauss_y = np.ones(ny, dtype=np.float64)
            gauss_x = np.ones(nx, dtype=np.float64)

            if ny > 2 * fw:
                ramp = np.linspace(0, 1, fw)
                gauss_y[:fw] = ramp
                gauss_y[-fw:] = ramp[::-1]

            if nx > 2 * fw:
                ramp = np.linspace(0, 1, fw)
                gauss_x[:fw] = ramp
                gauss_x[-fw:] = ramp[::-1]

            weight = np.outer(gauss_y, gauss_x)
            weight = gaussian_filter(weight, sigma=sigma)
            weight = np.clip(weight, 0.0, 1.0)
        else:
            y_ramp = np.ones(ny, dtype=np.float64)
            x_ramp = np.ones(nx, dtype=np.float64)

            if ny > 2 * fw:
                ramp = np.linspace(0, 1, fw)
                y_ramp[:fw] = ramp
                y_ramp[-fw:] = ramp[::-1]
            elif ny > 1:
                y_ramp = np.linspace(0, 1, ny)
                y_ramp = np.minimum(y_ramp, y_ramp[::-1])

            if nx > 2 * fw:
                ramp = np.linspace(0, 1, fw)
                x_ramp[:fw] = ramp
                x_ramp[-fw:] = ramp[::-1]
            elif nx > 1:
                x_ramp = np.linspace(0, 1, nx)
                x_ramp = np.minimum(x_ramp, x_ramp[::-1])

            weight = np.outer(y_ramp, x_ramp)

        return weight

    def blend_back(self,
                   global_mask: np.ndarray,
                   local_mask_opt: np.ndarray,
                   bbox: Tuple[int, int, int, int],
                   feather_weight: np.ndarray) -> np.ndarray:
        """
        将优化后的局部区域拼合回全局掩模

        使用羽化权重进行加权混合：
            result = w * local_opt + (1 - w) * global

        Args:
            global_mask: 全局掩模（OPC 结果）
            local_mask_opt: 优化后的局部掩模
            bbox: 局部区域在全局中的位置
            feather_weight: 羽化权重图

        Returns:
            拼合后的全局掩模
        """
        result = global_mask.copy()
        y_min, y_max, x_min, x_max = bbox

        local_global = global_mask[y_min:y_max, x_min:x_max]

        blended = (feather_weight * local_mask_opt +
                   (1.0 - feather_weight) * local_global)

        result[y_min:y_max, x_min:x_max] = blended
        return result


class LocalILTOptimizer:
    """
    局部 ILT 优化器

    对单个热点 bbox 区域进行 ILT 可微优化。
    与全局 ILT 的区别：
    - 仅优化局部区域，计算更快
    - 使用羽化边界确保与周围 OPC 区域平滑过渡
    - 可以配置更少的迭代次数（局部收敛更快）
    """

    def __init__(self,
                 config: HybridOPCILTConfig,
                 optical_system: OpticalSystem):
        """
        初始化局部 ILT 优化器

        Args:
            config: 混合工作流配置
            optical_system: 光学系统参数
        """
        self.config = config
        self.optical_system = optical_system

        ilt_cfg = config.ilt_config or ILTConfig()
        self.ilt_config = ilt_cfg

    def optimize(self,
                 local_mask: np.ndarray,
                 local_target: np.ndarray,
                 hotspot_idx: int,
                 bbox: Tuple[int, int, int, int]) -> LocalILTResult:
        """
        对局部区域进行 ILT 优化

        Args:
            local_mask: 局部掩模（来自 OPC 结果）
            local_target: 局部目标图案
            hotspot_idx: 热点索引
            bbox: 该区域在全局掩模中的 bbox

        Returns:
            局部 ILT 优化结果
        """
        if self.config.verbose:
            logger.info(f"  热点 #{hotspot_idx}: bbox={bbox}, 尺寸={local_mask.shape}")

        initial_epe = compute_epe(
            (local_mask >= 0.5).astype(np.float64),
            local_target,
            pixel_size=self.config.pixel_size
        )

        bbox_manager = HotspotBBoxManager(self.config)
        feather_weight = bbox_manager.create_feather_weight(local_mask.shape)

        ilt_config = self.ilt_config
        if self.config.verbose:
            logger.info(f"    初始 EPE: mean={initial_epe['epe_mean']:.3f} nm")

        try:
            ilt_result = run_ilt_workflow(
                local_mask, local_target,
                optical_system=self.optical_system,
                config=ilt_config
            )
            optimal_local = ilt_result.optimal_mask
            final_epe = ilt_result.final_epe
        except Exception as e:
            logger.warning(f"    局部 ILT 优化失败，跳过: {e}")
            optimal_local = local_mask.copy()
            final_epe = initial_epe
            ilt_result = None

        if self.config.verbose:
            improvement = initial_epe['epe_mean'] - final_epe.get('epe_mean', initial_epe['epe_mean'])
            logger.info(f"    最终 EPE: mean={final_epe.get('epe_mean', 0):.3f} nm "
                       f"(改善 {improvement:.3f} nm)")

        result = LocalILTResult(
            hotspot_idx=hotspot_idx,
            bbox=bbox,
            initial_mask_local=local_mask.copy(),
            optimal_mask_local=optimal_local.copy(),
            initial_epe=initial_epe,
            final_epe=final_epe,
            ilt_result=ilt_result,
            feather_weight=feather_weight
        )

        return result


class HybridOPCILTWorkflow:
    """
    OPC + ILT 混合精修工作流

    工作流程：
        阶段 1: 全局 OPC 粗校正（SRAF + 规则修正）
        阶段 2: 热点检测与 bbox 提取（膨胀、合并）
        阶段 3: 逐热点局部 ILT 可微优化
        阶段 4: 羽化拼合，输出最终掩模

    优势：
        - 全局 OPC 快速完成 SRAF 放置和整体校正
        - 仅热点区域进行高精度 ILT 优化，计算效率高
        - 羽化边界确保局部优化与全局结果平滑过渡
        - 兼顾计算效率与局部精度
    """

    def __init__(self,
                 config: Optional[HybridOPCILTConfig] = None,
                 optical_system: Optional[OpticalSystem] = None):
        """
        初始化混合工作流

        Args:
            config: 混合工作流配置
            optical_system: 光学系统参数
        """
        self.config = config or HybridOPCILTConfig()
        self.optical_system = optical_system or OpticalSystem()

        self.bbox_manager = HotspotBBoxManager(self.config)
        self.local_ilt_optimizer = LocalILTOptimizer(self.config, self.optical_system)

        if self.config.verbose:
            logger.info("OPC + ILT 混合工作流已初始化")
            logger.info(f"配置: bbox_padding={self.config.hotspot_bbox_padding}, "
                       f"max_hotspots={self.config.max_hotspots}, "
                       f"feather_width={self.config.feather_width}")

    def run(self,
            initial_mask: np.ndarray,
            target: np.ndarray) -> HybridOPCILTWorkflowResult:
        """
        运行混合精修工作流

        Args:
            initial_mask: 初始掩模
            target: 目标图案

        Returns:
            混合工作流结果
        """
        total_start = time.time()
        cfg = self.config

        initial_mask = initial_mask.astype(np.float64)
        target = target.astype(np.float64)
        image_shape = initial_mask.shape

        wafer_initial = (initial_mask >= 0.5).astype(np.float64)
        initial_epe = compute_epe(
            wafer_initial, target, pixel_size=cfg.pixel_size
        )

        if cfg.verbose:
            logger.info("\n" + "=" * 60)
            logger.info("OPC + ILT 混合精修工作流开始")
            logger.info("=" * 60)
            logger.info(f"掩模尺寸: {image_shape}")
            logger.info(f"初始 EPE: mean={initial_epe['epe_mean']:.3f} nm")

        opc_result = None
        opc_mask = initial_mask.copy()
        opc_epe = initial_epe
        opc_time = 0.0

        if cfg.run_global_opc:
            if cfg.verbose:
                logger.info("\n" + "-" * 50)
                logger.info("阶段 1/4: 全局 OPC 粗校正")
                logger.info("-" * 50)

            t0 = time.time()
            opc_config = cfg.opc_config or OPCConfig()
            opc_result = run_opc_workflow(
                initial_mask, target,
                config=opc_config,
                optical_system=self.optical_system
            )
            opc_mask = opc_result.corrected_mask.copy()
            opc_epe = opc_result.final_epe
            opc_time = time.time() - t0

            if cfg.verbose:
                logger.info(f"OPC 完成，耗时 {opc_time:.2f}s")
                logger.info(f"OPC 后 EPE: mean={opc_epe['epe_mean']:.3f} nm "
                           f"(改善 {initial_epe['epe_mean'] - opc_epe['epe_mean']:.3f} nm)")
        else:
            if cfg.verbose:
                logger.info("阶段 1 (全局 OPC): 跳过")

        if cfg.verbose:
            logger.info("\n" + "-" * 50)
            logger.info("阶段 2/4: 热点检测与 bbox 处理")
            logger.info("-" * 50)

        hotspot_detector = HotspotDetector(cfg.opc_config or OPCConfig())
        hotspots = hotspot_detector.detect(
            opc_mask, target,
            optical_system=self.optical_system
        )

        if cfg.verbose:
            logger.info(f"检测到 {len(hotspots)} 个热点")

        merged_hotspots = self.bbox_manager.merge_hotspots(hotspots, image_shape)
        if cfg.verbose:
            logger.info(f"合并后: {len(merged_hotspots)} 个热点区域")

        filtered_hotspots = self.bbox_manager.filter_hotspots(merged_hotspots)
        if cfg.verbose:
            logger.info(f"过滤后（前 {cfg.max_hotspots} 个优先级最高）: "
                       f"{len(filtered_hotspots)} 个热点待优化")

        local_ilt_results: List[LocalILTResult] = []
        final_mask = opc_mask.copy()
        ilt_time = 0.0

        if cfg.run_local_ilt and len(filtered_hotspots) > 0:
            if cfg.verbose:
                logger.info("\n" + "-" * 50)
                logger.info("阶段 3/4: 逐热点局部 ILT 优化")
                logger.info("-" * 50)

            t0 = time.time()

            for idx, hotspot in enumerate(filtered_hotspots):
                if cfg.verbose:
                    logger.info(f"\n[{idx + 1}/{len(filtered_hotspots)}] 优化热点 "
                               f"(优先级: {hotspot.priority:.1f}, "
                               f"max EPE: {hotspot.epe_max:.2f} nm)")

                bbox = hotspot.bbox
                local_mask, local_target = self.bbox_manager.crop_local(
                    opc_mask, target, bbox
                )

                local_result = self.local_ilt_optimizer.optimize(
                    local_mask, local_target, idx, bbox
                )
                local_ilt_results.append(local_result)

                if local_result.feather_weight is not None:
                    final_mask = self.bbox_manager.blend_back(
                        final_mask,
                        local_result.optimal_mask_local,
                        bbox,
                        local_result.feather_weight
                    )

            ilt_time = time.time() - t0

            if cfg.verbose:
                logger.info(f"\n局部 ILT 优化完成，总耗时 {ilt_time:.2f}s")
        else:
            if cfg.verbose:
                if not cfg.run_local_ilt:
                    logger.info("阶段 3 (局部 ILT): 跳过（配置禁用）")
                else:
                    logger.info("阶段 3 (局部 ILT): 跳过（无热点）")

        if cfg.verbose:
            logger.info("\n" + "-" * 50)
            logger.info("阶段 4/4: 最终评估")
            logger.info("-" * 50)

        wafer_final = (final_mask >= 0.5).astype(np.float64)
        final_epe = compute_epe(
            wafer_final, target, pixel_size=cfg.pixel_size
        )

        if cfg.verbose:
            logger.info(f"最终 EPE: mean={final_epe['epe_mean']:.3f} nm")
            logger.info(f"  OPC 阶段改善: {initial_epe['epe_mean'] - opc_epe['epe_mean']:.3f} nm")
            logger.info(f"  ILT 阶段改善: {opc_epe['epe_mean'] - final_epe['epe_mean']:.3f} nm")
            logger.info(f"  总改善: {initial_epe['epe_mean'] - final_epe['epe_mean']:.3f} nm")

        total_time = time.time() - total_start

        converged = all(
            r.final_epe.get('epe_mean', 0) <= r.initial_epe.get('epe_mean', 0)
            for r in local_ilt_results
        ) if len(local_ilt_results) > 0 else True

        reason = "混合精修完成"
        if len(filtered_hotspots) == 0:
            reason = "无可优化热点，直接使用 OPC 结果"
        elif not cfg.run_local_ilt:
            reason = "局部 ILT 未启用，直接使用 OPC 结果"

        result = HybridOPCILTWorkflowResult(
            initial_mask=initial_mask.copy(),
            opc_mask=opc_mask.copy(),
            final_mask=final_mask.copy(),
            target=target.copy(),
            initial_epe=initial_epe,
            opc_epe=opc_epe,
            final_epe=final_epe,
            opc_result=opc_result,
            local_ilt_results=local_ilt_results,
            merged_hotspots=filtered_hotspots,
            num_hotspots_optimized=len(local_ilt_results),
            converged=converged,
            reason=reason,
            total_time=total_time,
            opc_time=opc_time,
            ilt_time=ilt_time
        )

        if cfg.verbose:
            logger.info("\n" + "=" * 60)
            logger.info("OPC + ILT 混合精修工作流完成")
            logger.info("=" * 60)
            logger.info(f"总耗时: {total_time:.2f}s (OPC: {opc_time:.2f}s, ILT: {ilt_time:.2f}s)")
            logger.info(f"初始 EPE: {initial_epe['epe_mean']:.3f} nm")
            logger.info(f"OPC 后 EPE: {opc_epe['epe_mean']:.3f} nm")
            logger.info(f"最终 EPE: {final_epe['epe_mean']:.3f} nm")
            logger.info(f"优化热点数: {len(local_ilt_results)}")

        return result


def run_hybrid_opc_ilt_workflow(
        initial_mask: np.ndarray,
        target: np.ndarray,
        config: Optional[Union[HybridOPCILTConfig, str, Path]] = None,
        optical_system: Optional[OpticalSystem] = None
) -> HybridOPCILTWorkflowResult:
    """
    便捷函数：运行 OPC + ILT 混合精修工作流

    Args:
        initial_mask: 初始掩模
        target: 目标图案
        config: 混合工作流配置，可以是 HybridOPCILTConfig 对象、配置文件路径或 None
        optical_system: 光学系统参数

    Returns:
        混合工作流结果
    """
    if config is None:
        hybrid_config = HybridOPCILTConfig()
    elif isinstance(config, (str, Path)):
        hybrid_config = HybridOPCILTConfig.from_yaml(config)
    else:
        hybrid_config = config

    workflow = HybridOPCILTWorkflow(
        config=hybrid_config,
        optical_system=optical_system
    )

    return workflow.run(initial_mask, target)
