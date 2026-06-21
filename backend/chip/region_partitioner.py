# -*- coding: utf-8 -*-
"""
芯片区域划分模块

对完整芯片 GDS 进行区域划分，自动识别不同功能区域：
- 内存阵列区域（MEMORY_ARRAY）：规则重复结构，高周期性
- 逻辑标准单元区域（LOGIC_STDCELL）：随机逻辑，中等复杂度
- 模拟 IP 区域（ANALOG_IP）：定制电路，大尺寸器件
- 混合信号区域（MIXED_SIGNAL）：模数混合
- IO 环区域（IO_RING）：芯片外围电路

划分策略：
1. 基于 cell 名称/层次结构的启发式识别
2. 基于版图图像特征的自动分类
3. 基于频谱分析的周期性检测
4. 区域合并与边界优化
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import (
    label, find_objects, generate_binary_structure,
    distance_transform_edt, binary_dilation, binary_erosion,
    gaussian_filter, sobel,
)
from scipy.signal import fftconvolve

from layout.layout_manager import (
    GDSLoader, LayoutLoadOptions, LayoutCell, LayoutLibrary, LayoutManager,
)
from utils.data_io import load_gds_layer
from chip.schemas import (
    RegionType, ChipRegion, ChipRegionMetadata, OpticalConditionConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class RegionPartitionResult:
    """区域划分结果"""
    regions: List[ChipRegion] = field(default_factory=list)
    full_chip_mask: Optional[np.ndarray] = None
    chip_bounds_nm: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    pixel_size_nm: float = 1.0
    region_type_counts: Dict[str, int] = field(default_factory=dict)
    classification_confidence: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_regions': len(self.regions),
            'chip_bounds_nm': list(self.chip_bounds_nm),
            'pixel_size_nm': self.pixel_size_nm,
            'region_type_counts': self.region_type_counts,
            'classification_confidence': self.classification_confidence,
            'regions': [r.summary() for r in self.regions],
            'warnings': list(self.warnings),
        }


class LayoutFeatureExtractor:
    """版图特征提取器

    从掩模图像中提取用于区域分类的特征向量。
    """

    def __init__(self, pixel_size_nm: float = 1.0):
        self.pixel_size_nm = pixel_size_nm

    def extract_all(self, mask: np.ndarray) -> Dict[str, Any]:
        """提取所有特征"""
        features = {}
        features.update(self._extract_geometric_features(mask))
        features.update(self._extract_edge_features(mask))
        features.update(self._extract_spectral_features(mask))
        features.update(self._extract_topological_features(mask))
        return features

    def _extract_geometric_features(self, mask: np.ndarray) -> Dict[str, Any]:
        """提取几何特征"""
        h, w = mask.shape
        total_pixels = h * w
        fill_ratio = float(np.mean(mask))

        binary = (mask >= 0.5).astype(np.float64)
        if np.sum(binary) > 0:
            labeled, num_features = label(binary)
            sizes = []
            for obj in find_objects(labeled):
                if obj is not None:
                    region = labeled[obj] == (np.max(labeled[obj]))
                    sizes.append(int(np.sum(region)))

            if sizes:
                avg_feature_size = float(np.mean(sizes))
                min_feature_size = float(np.min(sizes)) * self.pixel_size_nm
                max_feature_size = float(np.max(sizes)) * self.pixel_size_nm
                size_std = float(np.std(sizes))
                size_cv = size_std / avg_feature_size if avg_feature_size > 0 else 0
            else:
                min_feature_size = 0.0
                max_feature_size = 0.0
                avg_feature_size = 0.0
                size_cv = 0.0
        else:
            num_features = 0
            min_feature_size = 0.0
            max_feature_size = 0.0
            avg_feature_size = 0.0
            size_cv = 0.0

        feature_density = num_features / total_pixels * 1e6 if total_pixels > 0 else 0

        return {
            'fill_ratio': fill_ratio,
            'num_features': int(num_features),
            'min_feature_size_nm': min_feature_size,
            'max_feature_size_nm': max_feature_size,
            'avg_feature_size_px': avg_feature_size,
            'feature_size_cv': size_cv,
            'feature_density_per_mm2': feature_density,
        }

    def _extract_edge_features(self, mask: np.ndarray) -> Dict[str, Any]:
        """提取边缘特征"""
        gy = np.zeros_like(mask)
        gx = np.zeros_like(mask)
        gy[:-1, :] = mask[1:, :] - mask[:-1, :]
        gx[:, :-1] = mask[:, 1:] - mask[:, :-1]

        edge_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        edge_pixels = edge_magnitude > 0.01
        edge_density = float(np.mean(edge_pixels))

        gy_sobel = sobel(mask, axis=0)
        gx_sobel = sobel(mask, axis=1)
        sobel_magnitude = np.sqrt(gx_sobel ** 2 + gy_sobel ** 2)

        corner_response = np.zeros_like(mask)
        if mask.shape[0] > 4 and mask.shape[1] > 4:
            Ixx = gx_sobel ** 2
            Iyy = gy_sobel ** 2
            Ixy = gx_sobel * gy_sobel
            det = Ixx * Iyy - Ixy ** 2
            trace = Ixx + Iyy
            corner_response = det - 0.04 * (trace ** 2)

        corner_threshold = np.max(corner_response) * 0.1
        corner_pixels = corner_response > corner_threshold
        corner_density = float(np.mean(corner_pixels))

        edge_orientation_hist = self._compute_edge_orientation_histogram(gx, gy)

        return {
            'edge_density': edge_density,
            'corner_density': corner_density,
            'edge_corner_ratio': corner_density / max(edge_density, 1e-8),
            'mean_sobel_magnitude': float(np.mean(sobel_magnitude)),
            'edge_orientation_histogram': edge_orientation_hist,
            'edge_orientation_uniformity': self._compute_uniformity(edge_orientation_hist),
        }

    def _extract_spectral_features(self, mask: np.ndarray) -> Dict[str, Any]:
        """提取频谱特征"""
        h, w = mask.shape
        if h < 16 or w < 16:
            return {
                'spectral_entropy': 0.0,
                'high_freq_energy_ratio': 0.0,
                'low_freq_energy_ratio': 0.0,
                'periodicity_score': 0.0,
                'dominant_pitch_nm': 0.0,
            }

        mask_centered = mask - np.mean(mask)
        fft = np.fft.fft2(mask_centered)
        fft_shifted = np.fft.fftshift(fft)
        power_spectrum = np.abs(fft_shifted) ** 2

        total_energy = np.sum(power_spectrum)
        if total_energy < 1e-12:
            return {
                'spectral_entropy': 0.0,
                'high_freq_energy_ratio': 0.0,
                'low_freq_energy_ratio': 0.0,
                'periodicity_score': 0.0,
                'dominant_pitch_nm': 0.0,
            }

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_dist = np.sqrt(cy ** 2 + cx ** 2)
        normalized_dist = dist_from_center / max_dist

        low_freq_mask = normalized_dist < 0.2
        high_freq_mask = normalized_dist > 0.5
        mid_freq_mask = ~(low_freq_mask | high_freq_mask)

        low_energy = np.sum(power_spectrum[low_freq_mask]) / total_energy
        mid_energy = np.sum(power_spectrum[mid_freq_mask]) / total_energy
        high_energy = np.sum(power_spectrum[high_freq_mask]) / total_energy

        prob = power_spectrum / total_energy
        prob = prob[prob > 0]
        spectral_entropy = -np.sum(prob * np.log2(prob))
        max_entropy = np.log2(prob.size) if prob.size > 0 else 1.0
        spectral_entropy_norm = spectral_entropy / max_entropy if max_entropy > 0 else 0

        periodicity, dominant_pitch = self._detect_periodicity(power_spectrum, h, w)

        return {
            'spectral_entropy': float(spectral_entropy_norm),
            'high_freq_energy_ratio': float(high_energy),
            'mid_freq_energy_ratio': float(mid_energy),
            'low_freq_energy_ratio': float(low_energy),
            'periodicity_score': float(periodicity),
            'dominant_pitch_nm': float(dominant_pitch * self.pixel_size_nm),
        }

    def _extract_topological_features(self, mask: np.ndarray) -> Dict[str, Any]:
        """提取拓扑特征"""
        binary = (mask >= 0.5).astype(bool)

        struct = generate_binary_structure(2, 2)
        eroded = binary_erosion(binary, structure=struct, iterations=2)
        skeleton = binary & (~eroded)

        if np.sum(binary) > 0:
            skeleton_density = float(np.sum(skeleton)) / float(np.sum(binary))
        else:
            skeleton_density = 0.0

        hole_mask = (1.0 - binary)
        if np.sum(hole_mask) > 0:
            labeled_holes, num_holes = label(hole_mask.astype(np.int32))
            edge_mask = np.zeros_like(hole_mask)
            edge_mask[0, :] = 1
            edge_mask[-1, :] = 1
            edge_mask[:, 0] = 1
            edge_mask[:, -1] = 1
            connected_to_edge = np.unique(labeled_holes[edge_mask > 0])
            enclosed_holes = [i for i in range(1, num_holes + 1) if i not in connected_to_edge]
            num_enclosed_holes = len(enclosed_holes)
        else:
            num_enclosed_holes = 0

        return {
            'skeleton_density': skeleton_density,
            'num_enclosed_holes': num_enclosed_holes,
            'hole_density': num_enclosed_holes / max(np.sum(binary), 1.0),
        }

    def _detect_periodicity(self, power_spectrum: np.ndarray, h: int, w: int) -> Tuple[float, float]:
        """检测周期性和主导间距"""
        cy, cx = h // 2, w // 2
        ps_half = power_spectrum[cy:, :]
        ps_half[0, :] = 0
        ps_half[:, cx] = 0

        if np.max(ps_half) < 1e-10:
            return 0.0, 0.0

        peaks = self._find_peaks_2d(ps_half, threshold_ratio=0.3)
        if len(peaks) < 2:
            return 0.0, 0.0

        distances = []
        for (py1, px1), (py2, px2) in zip(peaks[:-1], peaks[1:]):
            d = np.sqrt((py2 - py1) ** 2 + (px2 - cx) ** 2)
            distances.append(d)

        if distances:
            dominant_freq = min(distances)
            pitch = 1.0 / dominant_freq if dominant_freq > 0 else 0
        else:
            pitch = 0

        peak_energy = sum(np.max(ps_half) for _ in range(min(5, len(peaks))))
        total_energy = np.sum(ps_half)
        periodicity = min(1.0, peak_energy / max(total_energy, 1e-8) * len(peaks))

        return periodicity, pitch

    def _find_peaks_2d(self, arr: np.ndarray, threshold_ratio: float = 0.3) -> List[Tuple[int, int]]:
        """2D 峰值检测"""
        threshold = np.max(arr) * threshold_ratio
        peaks = []
        h, w = arr.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if arr[i, j] > threshold:
                    neighborhood = arr[i-1:i+2, j-1:j+2]
                    if arr[i, j] == np.max(neighborhood):
                        peaks.append((i, j))

        peaks.sort(key=lambda p: arr[p[0], p[1]], reverse=True)
        return peaks[:10]

    def _compute_edge_orientation_histogram(self, gx: np.ndarray, gy: np.ndarray) -> List[float]:
        """计算边缘方向直方图"""
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = np.arctan2(gy, gx)

        edge_mask = magnitude > 0.01
        if not np.any(edge_mask):
            return [0.0] * 8

        valid_orientations = orientation[edge_mask]
        hist, _ = np.histogram(valid_orientations, bins=8, range=(-np.pi, np.pi), density=True)
        return hist.tolist()

    def _compute_uniformity(self, hist: List[float]) -> float:
        """计算分布均匀度（1 = 完全均匀，0 = 完全集中）"""
        if not hist or sum(hist) == 0:
            return 0.0
        arr = np.array(hist)
        arr = arr / np.sum(arr)
        n = len(arr)
        uniform = np.ones(n) / n
        chi_square = np.sum((arr - uniform) ** 2 / uniform)
        uniformity = max(0.0, 1.0 - min(1.0, chi_square / (n - 1)))
        return float(uniformity)


class RegionClassifier:
    """区域分类器

    基于提取的特征向量，将区域分类为不同的 RegionType。
    """

    MEMORY_KEYWORDS = ['RAM', 'ROM', 'SRAM', 'DRAM', 'MEMORY', 'ARRAY', 'BITCELL', 'BIT_CELL']
    LOGIC_KEYWORDS = ['LOGIC', 'STDCELL', 'STD_CELL', 'CORE', 'CLUSTER', 'BLOCK']
    ANALOG_KEYWORDS = ['ANALOG', 'ADC', 'DAC', 'AMP', 'OPAMP', 'PLL', 'VCO', 'LDO', 'MIXER', 'RF']
    IO_KEYWORDS = ['IO', 'PAD', 'GPIO', 'ESD', 'DRIVER', 'RECEIVER']

    def __init__(self):
        self.feature_extractor = LayoutFeatureExtractor()

    def classify(self,
                 mask: np.ndarray,
                 cell_name_hints: Optional[List[str]] = None,
                 pixel_size_nm: float = 1.0) -> Tuple[RegionType, float, Dict[str, Any]]:
        """
        分类区域类型

        Args:
            mask: 掩模图像
            cell_name_hints: cell 名称提示列表
            pixel_size_nm: 像素尺寸 (nm)

        Returns:
            (region_type, confidence, features)
        """
        self.feature_extractor.pixel_size_nm = pixel_size_nm
        features = self.feature_extractor.extract_all(mask)

        keyword_score = self._keyword_based_score(cell_name_hints or [])
        feature_score = self._feature_based_score(features)

        combined_score = self._combine_scores(keyword_score, feature_score)

        best_type = max(combined_score.keys(), key=lambda k: combined_score[k])
        confidence = combined_score[best_type]

        return best_type, confidence, features

    def _keyword_based_score(self, cell_names: List[str]) -> Dict[RegionType, float]:
        """基于 cell 名称关键词的评分"""
        scores = {t: 0.0 for t in RegionType}

        if not cell_names:
            return scores

        combined_name = ' '.join(cell_names).upper()

        for kw in self.MEMORY_KEYWORDS:
            if kw in combined_name:
                scores[RegionType.MEMORY_ARRAY] += 0.3

        for kw in self.LOGIC_KEYWORDS:
            if kw in combined_name:
                scores[RegionType.LOGIC_STDCELL] += 0.3

        for kw in self.ANALOG_KEYWORDS:
            if kw in combined_name:
                scores[RegionType.ANALOG_IP] += 0.3

        for kw in self.IO_KEYWORDS:
            if kw in combined_name:
                scores[RegionType.IO_RING] += 0.3

        return scores

    def _feature_based_score(self, features: Dict[str, Any]) -> Dict[RegionType, float]:
        """基于图像特征的评分"""
        scores = {t: 0.0 for t in RegionType}

        periodicity = features.get('periodicity_score', 0.0)
        min_cd = features.get('min_feature_size_nm', 0.0)
        fill_ratio = features.get('fill_ratio', 0.0)
        edge_density = features.get('edge_density', 0.0)
        corner_density = features.get('corner_density', 0.0)
        size_cv = features.get('feature_size_cv', 0.0)
        high_freq_ratio = features.get('high_freq_energy_ratio', 0.0)
        orientation_uniformity = features.get('edge_orientation_uniformity', 0.0)
        num_holes = features.get('num_enclosed_holes', 0)

        if periodicity > 0.6 and size_cv < 0.3 and fill_ratio > 0.3:
            scores[RegionType.MEMORY_ARRAY] += 0.6 * periodicity
            scores[RegionType.MEMORY_ARRAY] += 0.2 * (1 - size_cv)
            if min_cd < 60:
                scores[RegionType.MEMORY_ARRAY] += 0.2

        if 0.2 < fill_ratio < 0.7 and edge_density > 0.1:
            scores[RegionType.LOGIC_STDCELL] += 0.3 * fill_ratio / 0.5
            if corner_density > 0.01:
                scores[RegionType.LOGIC_STDCELL] += 0.3
            if 0.3 < orientation_uniformity < 0.7:
                scores[RegionType.LOGIC_STDCELL] += 0.2
            if 0.3 < size_cv < 0.8:
                scores[RegionType.LOGIC_STDCELL] += 0.2

        if min_cd > 100 and fill_ratio < 0.3 and edge_density < 0.1:
            scores[RegionType.ANALOG_IP] += 0.4
            scores[RegionType.ANALOG_IP] += 0.2 * (min_cd / 500.0)
            if num_holes > 5:
                scores[RegionType.ANALOG_IP] += 0.2
            if orientation_uniformity > 0.6:
                scores[RegionType.ANALOG_IP] += 0.2

        if high_freq_ratio > 0.4 and size_cv > 0.5:
            scores[RegionType.MIXED_SIGNAL] += 0.5
            if 0.3 < fill_ratio < 0.5:
                scores[RegionType.MIXED_SIGNAL] += 0.3

        scores[RegionType.UNKNOWN] = 0.1

        return scores

    def _combine_scores(self,
                        keyword_scores: Dict[RegionType, float],
                        feature_scores: Dict[RegionType, float]) -> Dict[RegionType, float]:
        """组合关键词评分和特征评分"""
        combined = {}
        for t in RegionType:
            kw_score = keyword_scores.get(t, 0.0)
            feat_score = feature_scores.get(t, 0.0)

            if kw_score > 0:
                combined[t] = 0.6 * kw_score + 0.4 * feat_score
            else:
                combined[t] = feat_score

            combined[t] = min(1.0, max(0.0, combined[t]))

        if max(combined.values()) < 0.2:
            combined[RegionType.UNKNOWN] = 0.5

        return combined


class RegionPartitioner:
    """芯片区域划分器

    对完整芯片 GDS 进行区域划分，将其分为不同功能区域。
    """

    def __init__(self,
                 pixel_size_nm: float = 1.0,
                 min_region_size_um2: float = 100.0,
                 merge_distance_um: float = 5.0,
                 use_hierarchy: bool = True):
        """
        初始化区域划分器

        Args:
            pixel_size_nm: 像素尺寸 (nm)
            min_region_size_um2: 最小区域面积 (μm²)
            merge_distance_um: 区域合并距离 (μm)
            use_hierarchy: 是否使用 GDS 层次结构信息
        """
        self.pixel_size_nm = pixel_size_nm
        self.min_region_size_um2 = min_region_size_um2
        self.merge_distance_um = merge_distance_um
        self.use_hierarchy = use_hierarchy

        self.feature_extractor = LayoutFeatureExtractor(pixel_size_nm=pixel_size_nm)
        self.classifier = RegionClassifier()
        self.gds_loader = GDSLoader()

    def partition_gds(self,
                      gds_path: Union[str, Path],
                      layer: int,
                      datatype: int = 0,
                      cell_name: Optional[str] = None,
                      top_cell_only: bool = True) -> RegionPartitionResult:
        """
        从 GDS 文件进行区域划分

        Args:
            gds_path: GDS 文件路径
            layer: 层号
            datatype: 数据类型
            cell_name: 指定顶层 cell 名，None 则自动选择
            top_cell_only: 是否只处理顶层 cell

        Returns:
            RegionPartitionResult
        """
        gds_path = Path(gds_path)
        if not gds_path.exists():
            raise FileNotFoundError(f"GDS 文件不存在: {gds_path}")

        logger.info(f"开始划分芯片区域: {gds_path.name}, 层 {layer}/{datatype}")

        if cell_name is None:
            top_cells = self.gds_loader.list_cells(gds_path)
            if not top_cells:
                raise ValueError(f"GDS 文件中无顶层 cell: {gds_path}")
            cell_name = top_cells[0]
            logger.info(f"自动选择顶层 cell: {cell_name}")

        load_options = LayoutLoadOptions(
            layer=layer,
            datatype=datatype,
            pixel_size=self.pixel_size_nm,
            flatten_references=True,
            load_masks_on_init=True,
        )

        full_mask = self.gds_loader.load_cell_mask(gds_path, cell_name, load_options)
        if full_mask is None:
            raise RuntimeError(f"加载掩模失败: {cell_name}")

        full_chip_bounds = self._get_chip_bounds(gds_path, cell_name, layer, datatype)

        if self.use_hierarchy and not top_cell_only:
            regions = self._partition_by_hierarchy(
                gds_path, cell_name, layer, datatype, full_mask, full_chip_bounds
            )
        else:
            regions = self._partition_by_segmentation(
                full_mask, full_chip_bounds, cell_name
            )

        regions = self._merge_small_regions(regions)
        regions = self._resolve_overlaps(regions)

        result = self._build_partition_result(regions, full_mask, full_chip_bounds)

        logger.info(f"区域划分完成，共 {len(regions)} 个区域")
        for rtype, count in result.region_type_counts.items():
            logger.info(f"  {rtype}: {count} 个")

        return result

    def partition_mask(self,
                       full_mask: np.ndarray,
                       chip_bounds_nm: Tuple[float, float, float, float],
                       chip_name: str = "chip",
                       pixel_size_nm: Optional[float] = None) -> RegionPartitionResult:
        """
        从掩模图像进行区域划分

        Args:
            full_mask: 完整芯片掩模
            chip_bounds_nm: 芯片物理边界 (xmin, ymin, xmax, ymax) nm
            chip_name: 芯片名称
            pixel_size_nm: 像素尺寸 (nm)，None 则使用默认值

        Returns:
            RegionPartitionResult
        """
        if pixel_size_nm is not None:
            self.pixel_size_nm = pixel_size_nm
            self.feature_extractor = LayoutFeatureExtractor(pixel_size_nm=pixel_size_nm)

        logger.info(f"开始从掩模划分芯片区域: {chip_name}")

        regions = self._partition_by_segmentation(
            full_mask, chip_bounds_nm, chip_name
        )

        regions = self._merge_small_regions(regions)
        regions = self._resolve_overlaps(regions)

        result = self._build_partition_result(regions, full_mask, chip_bounds_nm)

        logger.info(f"区域划分完成，共 {len(regions)} 个区域")
        return result

    def _get_chip_bounds(self,
                         gds_path: Path,
                         cell_name: str,
                         layer: int,
                         datatype: int) -> Tuple[float, float, float, float]:
        """获取芯片物理边界"""
        try:
            import gdstk
            lib = gdstk.read_gds(str(gds_path))
            cell_map = {c.name: c for c in lib.cells}
            cell = cell_map.get(cell_name)
            if cell is not None:
                bounding_box = cell.bounding_box()
                if bounding_box is not None:
                    return (
                        bounding_box[0][0], bounding_box[0][1],
                        bounding_box[1][0], bounding_box[1][1],
                    )
        except Exception as e:
            logger.debug(f"获取芯片边界失败，使用默认: {e}")

        h, w = (0, 0)
        return (0.0, 0.0, w * self.pixel_size_nm, h * self.pixel_size_nm)

    def _partition_by_hierarchy(self,
                                gds_path: Path,
                                top_cell_name: str,
                                layer: int,
                                datatype: int,
                                full_mask: np.ndarray,
                                chip_bounds: Tuple[float, float, float, float]
                                ) -> List[ChipRegion]:
        """基于 GDS 层次结构进行区域划分"""
        try:
            import gdstk
            lib = gdstk.read_gds(str(gds_path))
            cell_map = {c.name: c for c in lib.cells}
            top_cell = cell_map.get(top_cell_name)
            if top_cell is None:
                return self._partition_by_segmentation(full_mask, chip_bounds, top_cell_name)

            regions = []
            processed_cells = set()

            def process_references(cell, transform=None, depth=0):
                if cell.name in processed_cells:
                    return
                processed_cells.add(cell.name)

                if depth > 0 and self._is_leaf_cell(cell, layer, datatype):
                    region = self._create_region_from_cell(
                        cell, gds_path, layer, datatype,
                        transform, chip_bounds, full_mask.shape
                    )
                    if region is not None:
                        regions.append(region)
                    return

                for ref in cell.references:
                    ref_cell = ref.ref_cell
                    if isinstance(ref_cell, str):
                        ref_cell = cell_map.get(ref_cell)
                    if ref_cell is None:
                        continue

                    ref_transform = self._compose_transforms(
                        transform,
                        self._get_reference_transform(ref)
                    )

                    if ref.repetition is not None:
                        offsets = ref.repetition.offsets()
                        for offset in offsets:
                            offset_transform = np.eye(3)
                            offset_transform[0, 2] = offset[0]
                            offset_transform[1, 2] = offset[1]
                            combined = ref_transform @ offset_transform
                            process_references(ref_cell, combined, depth + 1)
                    else:
                        process_references(ref_cell, ref_transform, depth + 1)

            process_references(top_cell, np.eye(3), 0)

            if not regions:
                logger.warning("基于层次的划分未找到区域，回退到图像分割方法")
                return self._partition_by_segmentation(full_mask, chip_bounds, top_cell_name)

            return regions

        except Exception as e:
            logger.warning(f"基于层次的划分失败: {e}，回退到图像分割方法")
            return self._partition_by_segmentation(full_mask, chip_bounds, top_cell_name)

    def _is_leaf_cell(self, cell, layer: int, datatype: int) -> bool:
        """判断是否为叶节点 cell（包含实际多边形）"""
        if not cell.references:
            return True
        for poly in cell.polygons:
            if poly.layer == layer and poly.datatype == datatype:
                return True
        return False

    def _get_reference_transform(self, ref) -> np.ndarray:
        """获取引用的变换矩阵"""
        transform = np.eye(3)
        origin = ref.origin or (0.0, 0.0)
        transform[0, 2] = origin[0]
        transform[1, 2] = origin[1]

        if ref.rotation:
            rad = np.radians(ref.rotation)
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            rot = np.eye(3)
            rot[0, 0] = cos_r
            rot[0, 1] = -sin_r
            rot[1, 0] = sin_r
            rot[1, 1] = cos_r
            transform = rot @ transform

        if ref.magnification != 1.0:
            scale = np.eye(3)
            scale[0, 0] = ref.magnification
            scale[1, 1] = ref.magnification
            transform = scale @ transform

        if ref.x_reflection:
            mirror = np.eye(3)
            mirror[1, 1] = -1.0
            transform = mirror @ transform

        return transform

    def _compose_transforms(self, t1: Optional[np.ndarray], t2: np.ndarray) -> np.ndarray:
        """组合变换矩阵"""
        if t1 is None:
            return t2
        return t2 @ t1

    def _create_region_from_cell(self,
                                 cell,
                                 gds_path: Path,
                                 layer: int,
                                 datatype: int,
                                 transform: np.ndarray,
                                 chip_bounds: Tuple[float, float, float, float],
                                 full_mask_shape: Tuple[int, int]
                                 ) -> Optional[ChipRegion]:
        """从 cell 创建区域"""
        try:
            bounding_box = cell.bounding_box()
            if bounding_box is None:
                return None

            bb_min = np.array([bounding_box[0][0], bounding_box[0][1], 1.0])
            bb_max = np.array([bounding_box[1][0], bounding_box[1][1], 1.0])

            transformed_min = transform @ bb_min
            transformed_max = transform @ bb_max

            xmin = min(transformed_min[0], transformed_max[0])
            ymin = min(transformed_min[1], transformed_max[1])
            xmax = max(transformed_min[0], transformed_max[0])
            ymax = max(transformed_min[1], transformed_max[1])

            area_um2 = (xmax - xmin) * (ymax - ymin) / 1e6
            min_area = self.min_region_size_um2 * 1e6
            if (xmax - xmin) * (ymax - ymin) < min_area:
                return None

            px_xmin = int(np.floor(xmin / self.pixel_size_nm))
            px_ymin = int(np.floor(ymin / self.pixel_size_nm))
            px_xmax = int(np.ceil(xmax / self.pixel_size_nm))
            px_ymax = int(np.ceil(ymax / self.pixel_size_nm))

            px_xmin = max(0, min(px_xmin, full_mask_shape[1] - 1))
            px_ymin = max(0, min(px_ymin, full_mask_shape[0] - 1))
            px_xmax = max(px_xmin + 1, min(px_xmax, full_mask_shape[1]))
            px_ymax = max(px_ymin + 1, min(px_ymax, full_mask_shape[0]))

            if px_xmax - px_xmin < 16 or px_ymax - px_ymin < 16:
                return None

            load_options = LayoutLoadOptions(
                layer=layer,
                datatype=datatype,
                pixel_size=self.pixel_size_nm,
                flatten_references=True,
                bounds=(xmin, ymin, xmax, ymax),
                load_masks_on_init=True,
            )

            mask = self.gds_loader.load_cell_mask(gds_path, cell.name, load_options)
            if mask is None:
                return None

            target_shape = (px_ymax - px_ymin, px_xmax - px_xmin)
            if mask.shape != target_shape:
                mask = self._resize_mask(mask, target_shape)

            region_type, confidence, features = self.classifier.classify(
                mask, [cell.name], self.pixel_size_nm
            )

            k1 = self._compute_k1(features.get('min_feature_size_nm', 0.0))
            complexity = self._compute_complexity_score(features)

            metadata = ChipRegionMetadata(
                region_id=f"region_{cell.name}_{len(region_type.value)}",
                region_type=region_type,
                bounds_nm=(xmin, ymin, xmax, ymax),
                bounds_px=(px_xmin, px_ymin, px_xmax, px_ymax),
                pixel_size_nm=self.pixel_size_nm,
                area_um2=area_um2,
                edge_density=features.get('edge_density', 0.0),
                corner_density=features.get('corner_density', 0.0),
                fill_ratio=features.get('fill_ratio', 0.0),
                min_cd_nm=features.get('min_feature_size_nm', 0.0),
                periodicity_score=features.get('periodicity_score', 0.0),
                dominant_pitch_nm=features.get('dominant_pitch_nm', 0.0),
                spectral_high_freq_ratio=features.get('high_freq_energy_ratio', 0.0),
                k1_factor=k1,
                complexity_score=complexity,
                cell_name_hints=[cell.name],
                layer_hints=[layer],
                extra={'classification_confidence': confidence, 'features': features},
            )

            region = ChipRegion(
                region_id=metadata.region_id,
                metadata=metadata,
                mask=mask,
                target=mask.copy(),
                is_optimized=False,
            )

            return region

        except Exception as e:
            logger.debug(f"从 cell 创建区域失败 {cell.name}: {e}")
            return None

    def _partition_by_segmentation(self,
                                   full_mask: np.ndarray,
                                   chip_bounds: Tuple[float, float, float, float],
                                   chip_name: str
                                   ) -> List[ChipRegion]:
        """基于图像分割进行区域划分"""
        logger.info("使用图像分割方法进行区域划分")

        h, w = full_mask.shape
        block_size_px = 512
        overlap_px = 32

        regions = []
        region_idx = 0

        xmin_nm, ymin_nm, xmax_nm, ymax_nm = chip_bounds
        chip_width_nm = xmax_nm - xmin_nm
        chip_height_nm = ymax_nm - ymin_nm

        step_px = block_size_px - overlap_px

        for y_start in range(0, h, step_px):
            for x_start in range(0, w, step_px):
                y_end = min(y_start + block_size_px, h)
                x_end = min(x_start + block_size_px, w)

                if y_end - y_start < 64 or x_end - x_start < 64:
                    continue

                block_mask = full_mask[y_start:y_end, x_start:x_end].copy()

                if np.mean(block_mask) < 0.01:
                    continue

                block_xmin_nm = xmin_nm + x_start * self.pixel_size_nm
                block_ymin_nm = ymin_nm + y_start * self.pixel_size_nm
                block_xmax_nm = xmin_nm + x_end * self.pixel_size_nm
                block_ymax_nm = ymin_nm + y_end * self.pixel_size_nm

                area_um2 = (block_xmax_nm - block_xmin_nm) * (block_ymax_nm - block_ymin_nm) / 1e6

                region_type, confidence, features = self.classifier.classify(
                    block_mask, [], self.pixel_size_nm
                )

                k1 = self._compute_k1(features.get('min_feature_size_nm', 0.0))
                complexity = self._compute_complexity_score(features)

                region_id = f"region_{chip_name}_{region_idx:04d}"
                region_idx += 1

                metadata = ChipRegionMetadata(
                    region_id=region_id,
                    region_type=region_type,
                    bounds_nm=(block_xmin_nm, block_ymin_nm, block_xmax_nm, block_ymax_nm),
                    bounds_px=(x_start, y_start, x_end, y_end),
                    pixel_size_nm=self.pixel_size_nm,
                    area_um2=area_um2,
                    edge_density=features.get('edge_density', 0.0),
                    corner_density=features.get('corner_density', 0.0),
                    fill_ratio=features.get('fill_ratio', 0.0),
                    min_cd_nm=features.get('min_feature_size_nm', 0.0),
                    periodicity_score=features.get('periodicity_score', 0.0),
                    dominant_pitch_nm=features.get('dominant_pitch_nm', 0.0),
                    spectral_high_freq_ratio=features.get('high_freq_energy_ratio', 0.0),
                    k1_factor=k1,
                    complexity_score=complexity,
                    layer_hints=[],
                    extra={'classification_confidence': confidence, 'features': features},
                )

                region = ChipRegion(
                    region_id=region_id,
                    metadata=metadata,
                    mask=block_mask,
                    target=block_mask.copy(),
                    is_optimized=False,
                )

                regions.append(region)

        regions = self._merge_adjacent_regions(regions)

        return regions

    def _merge_adjacent_regions(self, regions: List[ChipRegion]) -> List[ChipRegion]:
        """合并相邻且类型相同的区域"""
        if len(regions) < 2:
            return regions

        merge_dist_px = int(self.merge_distance_um * 1000 / self.pixel_size_nm)

        merged = list(regions)
        changed = True
        iterations = 0

        while changed and iterations < 10:
            changed = False
            iterations += 1

            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if merged[i] is None or merged[j] is None:
                        continue

                    r1, r2 = merged[i], merged[j]

                    if r1.metadata.region_type != r2.metadata.region_type:
                        continue

                    if self._are_adjacent(r1, r2, merge_dist_px):
                        merged_region = self._merge_two_regions(r1, r2)
                        merged[i] = merged_region
                        merged[j] = None
                        changed = True

            merged = [r for r in merged if r is not None]

        return merged

    def _are_adjacent(self, r1: ChipRegion, r2: ChipRegion, max_dist_px: int) -> bool:
        """判断两个区域是否相邻"""
        if r1.metadata.bounds_px is None or r2.metadata.bounds_px is None:
            return False

        x1min, y1min, x1max, y1max = r1.metadata.bounds_px
        x2min, y2min, x2max, y2max = r2.metadata.bounds_px

        x_overlap = not (x1max < x2min - max_dist_px or x2max < x1min - max_dist_px)
        y_overlap = not (y1max < y2min - max_dist_px or y2max < y1min - max_dist_px)

        return x_overlap and y_overlap

    def _merge_two_regions(self, r1: ChipRegion, r2: ChipRegion) -> ChipRegion:
        """合并两个区域"""
        b1 = r1.metadata.bounds_nm
        b2 = r2.metadata.bounds_nm
        xmin = min(b1[0], b2[0])
        ymin = min(b1[1], b2[1])
        xmax = max(b1[2], b2[2])
        ymax = max(b1[3], b2[3])

        pb1 = r1.metadata.bounds_px or (0, 0, 0, 0)
        pb2 = r2.metadata.bounds_px or (0, 0, 0, 0)
        px_xmin = min(pb1[0], pb2[0])
        px_ymin = min(pb1[1], pb2[1])
        px_xmax = max(pb1[2], pb2[2])
        px_ymax = max(pb1[3], pb2[3])

        width_px = px_xmax - px_xmin
        height_px = px_ymax - px_ymin

        merged_mask = np.zeros((height_px, width_px), dtype=np.float64)

        dy1 = pb1[1] - px_ymin
        dx1 = pb1[0] - px_xmin
        if r1.mask is not None:
            h1, w1 = r1.mask.shape
            merged_mask[dy1:dy1 + h1, dx1:dx1 + w1] = np.maximum(
                merged_mask[dy1:dy1 + h1, dx1:dx1 + w1], r1.mask
            )

        dy2 = pb2[1] - px_ymin
        dx2 = pb2[0] - px_xmin
        if r2.mask is not None:
            h2, w2 = r2.mask.shape
            merged_mask[dy2:dy2 + h2, dx2:dx2 + w2] = np.maximum(
                merged_mask[dy2:dy2 + h2, dx2:dx2 + w2], r2.mask
            )

        area_um2 = (xmax - xmin) * (ymax - ymin) / 1e6

        conf1 = r1.metadata.extra.get('classification_confidence', 0.0)
        conf2 = r2.metadata.extra.get('classification_confidence', 0.0)
        area1 = r1.metadata.area_um2
        area2 = r2.metadata.area_um2
        total_area = area1 + area2 or 1.0
        avg_conf = (conf1 * area1 + conf2 * area2) / total_area

        metadata = ChipRegionMetadata(
            region_id=f"{r1.region_id}_merged",
            region_type=r1.metadata.region_type,
            bounds_nm=(xmin, ymin, xmax, ymax),
            bounds_px=(px_xmin, px_ymin, px_xmax, px_ymax),
            pixel_size_nm=self.pixel_size_nm,
            area_um2=area_um2,
            edge_density=(r1.metadata.edge_density + r2.metadata.edge_density) / 2,
            corner_density=(r1.metadata.corner_density + r2.metadata.corner_density) / 2,
            fill_ratio=(r1.metadata.fill_ratio + r2.metadata.fill_ratio) / 2,
            min_cd_nm=min(r1.metadata.min_cd_nm, r2.metadata.min_cd_nm),
            periodicity_score=max(r1.metadata.periodicity_score, r2.metadata.periodicity_score),
            k1_factor=min(r1.metadata.k1_factor, r2.metadata.k1_factor),
            complexity_score=max(r1.metadata.complexity_score, r2.metadata.complexity_score),
            cell_name_hints=r1.metadata.cell_name_hints + r2.metadata.cell_name_hints,
            layer_hints=list(set(r1.metadata.layer_hints + r2.metadata.layer_hints)),
            extra={
                'classification_confidence': avg_conf,
                'merged_from': [r1.region_id, r2.region_id],
            },
        )

        return ChipRegion(
            region_id=metadata.region_id,
            metadata=metadata,
            mask=merged_mask,
            target=merged_mask.copy(),
            is_optimized=False,
        )

    def _merge_small_regions(self, regions: List[ChipRegion]) -> List[ChipRegion]:
        """合并过小的区域到相邻的大区域"""
        if len(regions) < 2:
            return regions

        min_area_px = self.min_region_size_um2 * 1e6 / (self.pixel_size_nm ** 2)

        small_regions = []
        large_regions = []
        for r in regions:
            if r.shape and r.shape[0] * r.shape[1] < min_area_px:
                small_regions.append(r)
            else:
                large_regions.append(r)

        if not small_regions:
            return regions

        for small in small_regions:
            best_large = None
            best_dist = float('inf')

            for large in large_regions:
                dist = self._region_distance(small, large)
                if dist < best_dist:
                    best_dist = dist
                    best_large = large

            if best_large is not None:
                merged = self._merge_two_regions(best_large, small)
                idx = large_regions.index(best_large)
                large_regions[idx] = merged
            else:
                large_regions.append(small)

        return large_regions

    def _region_distance(self, r1: ChipRegion, r2: ChipRegion) -> float:
        """计算两个区域之间的距离"""
        if r1.metadata.bounds_px is None or r2.metadata.bounds_px is None:
            return float('inf')

        x1min, y1min, x1max, y1max = r1.metadata.bounds_px
        x2min, y2min, x2max, y2max = r2.metadata.bounds_px

        cx1, cy1 = (x1min + x1max) / 2, (y1min + y1max) / 2
        cx2, cy2 = (x2min + x2max) / 2, (y2min + y2max) / 2

        return np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    def _resolve_overlaps(self, regions: List[ChipRegion]) -> List[ChipRegion]:
        """解决区域重叠问题"""
        if len(regions) < 2:
            return regions

        for i, r in enumerate(regions):
            r.overlap_region_ids = []
            for j, other in enumerate(regions):
                if i != j and self._regions_overlap(r, other):
                    r.overlap_region_ids.append(other.region_id)

        return regions

    def _regions_overlap(self, r1: ChipRegion, r2: ChipRegion) -> bool:
        """判断两个区域是否重叠"""
        if r1.metadata.bounds_px is None or r2.metadata.bounds_px is None:
            return False

        x1min, y1min, x1max, y1max = r1.metadata.bounds_px
        x2min, y2min, x2max, y2max = r2.metadata.bounds_px

        return not (x1max <= x2min or x2max <= x1min or
                    y1max <= y2min or y2max <= y1min)

    def _compute_k1(self, min_cd_nm: float) -> float:
        """计算 k1 因子"""
        if min_cd_nm <= 0:
            return 1.0
        na = 1.35
        wavelength = 193.0
        return min_cd_nm * na / wavelength

    def _compute_complexity_score(self, features: Dict[str, Any]) -> float:
        """计算区域复杂度评分 (0-1)"""
        score = 0.0

        k1 = self._compute_k1(features.get('min_feature_size_nm', 0.0))
        if k1 < 0.35:
            score += 0.4
        elif k1 < 0.5:
            score += 0.25
        elif k1 < 0.7:
            score += 0.1

        if features.get('corner_density', 0.0) > 0.05:
            score += 0.25
        elif features.get('corner_density', 0.0) > 0.02:
            score += 0.15

        if features.get('high_freq_energy_ratio', 0.0) > 0.3:
            score += 0.2
        elif features.get('high_freq_energy_ratio', 0.0) > 0.15:
            score += 0.1

        if features.get('periodicity_score', 0.0) < 0.3:
            score += 0.15

        return min(1.0, score)

    def _resize_mask(self, mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """调整掩模尺寸"""
        from scipy.ndimage import zoom
        if mask.shape == target_shape:
            return mask

        zoom_factors = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])
        resized = zoom(mask, zoom_factors, order=1)
        return (resized >= 0.5).astype(np.float64)

    def _build_partition_result(self,
                                regions: List[ChipRegion],
                                full_mask: np.ndarray,
                                chip_bounds: Tuple[float, float, float, float]
                                ) -> RegionPartitionResult:
        """构建划分结果"""
        type_counts: Dict[str, int] = defaultdict(int)
        confidence_map: Dict[str, float] = {}

        for r in regions:
            type_str = r.metadata.region_type.value
            type_counts[type_str] += 1
            conf = r.metadata.extra.get('classification_confidence', 0.0)
            confidence_map[r.region_id] = conf

        return RegionPartitionResult(
            regions=regions,
            full_chip_mask=full_mask,
            chip_bounds_nm=chip_bounds,
            pixel_size_nm=self.pixel_size_nm,
            region_type_counts=dict(type_counts),
            classification_confidence=confidence_map,
        )
