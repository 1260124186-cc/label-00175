# -*- coding: utf-8 -*-
"""
边界拼接与伪影处理器

将分块优化后的区域按坐标拼合成完整的芯片掩模，
并处理边界拼接伪影，确保区域间的连续性。

核心功能：
1. 按坐标拼合优化后的区域
2. 检测边界拼接伪影（不连续性、梯度跳变）
3. 边界平滑与一致性修正
4. 梯度融合与羽化处理
5. 全局一致性约束
6. 伪影量化评估
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from scipy.ndimage import (
    gaussian_filter, sobel, binary_dilation, binary_erosion,
    distance_transform_edt,
)

from chip.schemas import (
    RegionType, ChipRegion, StitchingConfig, BoundaryArtifactMetrics,
    ChipRETConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class BoundaryInfo:
    """边界信息"""
    boundary_id: str
    region1_id: str
    region2_id: str
    orientation: str
    bounds_px: Tuple[int, int, int, int]
    overlap_width_px: int
    has_artifact: bool = False
    artifact_score: float = 0.0


class BoundaryStitcher:
    """
    边界拼接与伪影处理器

    负责将优化后的区域拼合成完整的芯片掩模，并处理边界伪影。

    使用方法：
        stitcher = BoundaryStitcher(global_config)
        stitched_mask, metrics = stitcher.stitch_regions(regions, full_shape)
    """

    def __init__(
        self,
        global_config: Optional[ChipRETConfig] = None,
    ):
        """
        初始化边界拼接器

        Args:
            global_config: 芯片级 RET 全局配置
        """
        self.global_config = global_config or ChipRETConfig()
        self.config = self.global_config.stitching_config

    def stitch_regions(
        self,
        regions: List[ChipRegion],
        full_shape: Tuple[int, int],
        chip_origin_nm: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[np.ndarray, List[BoundaryArtifactMetrics]]:
        """
        拼接所有区域为完整掩模

        Args:
            regions: 优化后的区域列表
            full_shape: 完整芯片掩模形状 (ny, nx)
            chip_origin_nm: 芯片原点坐标 (x, y)，单位 nm

        Returns:
            (拼合后的完整掩模, 边界伪影指标列表)
        """
        logger.info(f"开始拼接 {len(regions)} 个区域，目标尺寸: {full_shape}")

        pixel_size = regions[0].metadata.pixel_size_nm if regions else 1.0

        stitched_mask = self._initial_stitch(
            regions, full_shape, chip_origin_nm, pixel_size
        )

        boundaries = self._find_boundaries(regions, full_shape, chip_origin_nm, pixel_size)

        artifact_metrics = []
        if self.config.enable_boundary_correction:
            for boundary in boundaries:
                metric = self._detect_and_correct_artifacts(
                    stitched_mask, boundary, regions
                )
                artifact_metrics.append(metric)

        if self.config.enable_gradient_blending:
            stitched_mask = self._apply_gradient_blending(
                stitched_mask, boundaries, regions
            )

        if self.config.enable_global_consistency:
            stitched_mask = self._apply_global_consistency(
                stitched_mask, regions
            )

        stitched_mask = self._finalize_mask(stitched_mask)

        logger.info(f"拼接完成，检测到 {len(boundaries)} 条边界，"
                   f"修正伪影 {sum(1 for m in artifact_metrics if m.corrected_count > 0)} 处")

        return stitched_mask, artifact_metrics

    def _initial_stitch(
        self,
        regions: List[ChipRegion],
        full_shape: Tuple[int, int],
        chip_origin_nm: Tuple[float, float],
        pixel_size: float,
    ) -> np.ndarray:
        """
        初始拼接：将各区域按坐标放置到完整掩模中

        Args:
            regions: 区域列表
            full_shape: 完整形状
            chip_origin_nm: 芯片原点
            pixel_size: 像素尺寸

        Returns:
            初始拼接的掩模
        """
        stitched = np.zeros(full_shape, dtype=np.float64)
        weight = np.zeros(full_shape, dtype=np.float64)

        overlap = self.config.overlap_width_px

        for region in regions:
            if region.optimized_mask is None:
                logger.warning(f"区域 {region.region_id} 无优化掩模，跳过")
                continue

            bounds_px = self._region_bounds_to_pixel(
                region, chip_origin_nm, pixel_size, full_shape
            )
            y0, y1, x0, x1 = bounds_px

            region_mask = region.optimized_mask
            region_h, region_w = region_mask.shape

            h = min(y1 - y0, region_h)
            w = min(x1 - x0, region_w)

            region_mask = region_mask[:h, :w]

            blend_weight = self._generate_region_blend_weight(
                (h, w), region, overlap
            )

            stitched[y0:y0+h, x0:x0+w] += region_mask * blend_weight
            weight[y0:y0+h, x0:x0+w] += blend_weight

        weight = np.where(weight > 0, weight, 1.0)
        stitched = stitched / weight

        return stitched

    def _region_bounds_to_pixel(
        self,
        region: ChipRegion,
        chip_origin_nm: Tuple[float, float],
        pixel_size: float,
        full_shape: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """
        将区域边界从 nm 转换为像素坐标

        Args:
            region: 芯片区域
            chip_origin_nm: 芯片原点
            pixel_size: 像素尺寸
            full_shape: 完整形状

        Returns:
            (y0, y1, x0, x1) 像素坐标
        """
        bx0, by0, bx1, by1 = region.metadata.bounds_nm
        cx0, cy0 = chip_origin_nm

        x0 = int(round((bx0 - cx0) / pixel_size))
        y0 = int(round((by0 - cy0) / pixel_size))
        x1 = int(round((bx1 - cx0) / pixel_size))
        y1 = int(round((by1 - cy0) / pixel_size))

        x0 = max(0, min(x0, full_shape[1]))
        y0 = max(0, min(y0, full_shape[0]))
        x1 = max(0, min(x1, full_shape[1]))
        y1 = max(0, min(y1, full_shape[0]))

        return (y0, y1, x0, x1)

    def _generate_region_blend_weight(
        self,
        shape: Tuple[int, int],
        region: ChipRegion,
        overlap: int,
    ) -> np.ndarray:
        """
        生成区域的融合权重矩阵

        Args:
            shape: 区域形状
            region: 芯片区域
            overlap: 重叠宽度

        Returns:
            权重矩阵
        """
        h, w = shape
        feather = self.config.feathering_width_px

        if feather <= 0 or overlap <= 0:
            return np.ones((h, w), dtype=np.float64)

        weight = np.ones((h, w), dtype=np.float64)

        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xv, yv = np.meshgrid(x, y)

        if region.overlap_region_ids:
            left_edge = min(feather / max(w, 1), 0.5)
            right_edge = min(feather / max(w, 1), 0.5)
            top_edge = min(feather / max(h, 1), 0.5)
            bottom_edge = min(feather / max(h, 1), 0.5)

            if region.overlap_width_px > 0:
                left_weight = np.clip(xv / left_edge, 0, 1)
                right_weight = np.clip((1 - xv) / right_edge, 0, 1)
                top_weight = np.clip(yv / top_edge, 0, 1)
                bottom_weight = np.clip((1 - yv) / bottom_edge, 0, 1)

                weight *= np.minimum(np.minimum(left_weight, right_weight),
                                    np.minimum(top_weight, bottom_weight))

        weight = np.clip(weight, 0.01, 1.0)
        return weight

    def _find_boundaries(
        self,
        regions: List[ChipRegion],
        full_shape: Tuple[int, int],
        chip_origin_nm: Tuple[float, float],
        pixel_size: float,
    ) -> List[BoundaryInfo]:
        """
        查找区域间的边界

        Args:
            regions: 区域列表
            full_shape: 完整形状
            chip_origin_nm: 芯片原点
            pixel_size: 像素尺寸

        Returns:
            边界信息列表
        """
        boundaries = []
        overlap_threshold = self.config.overlap_width_px * pixel_size * 0.5

        region_bounds = {}
        for region in regions:
            bounds_px = self._region_bounds_to_pixel(
                region, chip_origin_nm, pixel_size, full_shape
            )
            region_bounds[region.region_id] = {
                'bounds_nm': region.metadata.bounds_nm,
                'bounds_px': bounds_px,
                'region': region,
            }

        region_ids = list(region_bounds.keys())
        for i in range(len(region_ids)):
            for j in range(i + 1, len(region_ids)):
                id1 = region_ids[i]
                id2 = region_ids[j]

                b1 = region_bounds[id1]['bounds_nm']
                b2 = region_bounds[id2]['bounds_nm']

                x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))

                if x_overlap > overlap_threshold or y_overlap > overlap_threshold:
                    bp1 = region_bounds[id1]['bounds_px']
                    bp2 = region_bounds[id2]['bounds_px']

                    x_min = max(bp1[2], bp2[2])
                    x_max = min(bp1[3], bp2[3])
                    y_min = max(bp1[0], bp2[0])
                    y_max = min(bp1[1], bp2[1])

                    if x_max > x_min and y_max > y_min:
                        overlap_w = x_max - x_min
                        overlap_h = y_max - y_min

                        if overlap_w > overlap_h:
                            orientation = 'horizontal'
                            bounds_px = (y_min, y_max, x_min, x_max)
                        else:
                            orientation = 'vertical'
                            bounds_px = (y_min, y_max, x_min, x_max)

                        boundary = BoundaryInfo(
                            boundary_id=f"boundary_{id1}_{id2}",
                            region1_id=id1,
                            region2_id=id2,
                            orientation=orientation,
                            bounds_px=bounds_px,
                            overlap_width_px=max(overlap_w, overlap_h),
                        )
                        boundaries.append(boundary)

        return boundaries

    def _detect_and_correct_artifacts(
        self,
        stitched_mask: np.ndarray,
        boundary: BoundaryInfo,
        regions: List[ChipRegion],
    ) -> BoundaryArtifactMetrics:
        """
        检测并修正边界伪影

        Args:
            stitched_mask: 拼接后的掩模
            boundary: 边界信息
            regions: 区域列表

        Returns:
            边界伪影指标
        """
        y0, y1, x0, x1 = boundary.bounds_px
        window = self.config.artifact_window_size_px
        threshold = self.config.artifact_detection_threshold
        max_iter = self.config.max_artifact_correction_iterations

        metric = BoundaryArtifactMetrics(
            boundary_id=boundary.boundary_id,
        )

        boundary_region = stitched_mask[y0:y1, x0:x1]
        if boundary_region.size == 0:
            return metric

        gradient_y = sobel(boundary_region, axis=0)
        gradient_x = sobel(boundary_region, axis=1)
        gradient_mag = np.sqrt(gradient_y**2 + gradient_x**2)

        if boundary.orientation == 'horizontal':
            cross_section = gradient_mag[gradient_mag.shape[0]//2, :]
        else:
            cross_section = gradient_mag[:, gradient_mag.shape[1]//2]

        metric.max_discontinuity = float(np.max(gradient_mag))
        metric.mean_discontinuity = float(np.mean(gradient_mag))
        metric.std_discontinuity = float(np.std(gradient_mag))
        metric.max_gradient_jump = float(np.max(np.abs(np.diff(cross_section)))) if len(cross_section) > 1 else 0.0
        metric.mean_gradient_jump = float(np.mean(np.abs(np.diff(cross_section)))) if len(cross_section) > 1 else 0.0

        artifact_mask = gradient_mag > threshold
        metric.artifact_pixel_count = int(np.sum(artifact_mask))
        metric.artifact_density = float(metric.artifact_pixel_count / gradient_mag.size)

        boundary.has_artifact = metric.artifact_density > 0.1

        if boundary.has_artifact:
            corrected_count = 0
            for iteration in range(max_iter):
                artifact_coords = np.argwhere(artifact_mask)

                if len(artifact_coords) == 0:
                    break

                for ay, ax in artifact_coords:
                    abs_ay = y0 + ay
                    abs_ax = x0 + ax

                    corrected = self._correct_artifact_pixel(
                        stitched_mask, abs_ay, abs_ax, window, boundary
                    )
                    if corrected is not None:
                        stitched_mask[abs_ay, abs_ax] = corrected
                        corrected_count += 1

                boundary_region = stitched_mask[y0:y1, x0:x1]
                gradient_y = sobel(boundary_region, axis=0)
                gradient_x = sobel(boundary_region, axis=1)
                gradient_mag = np.sqrt(gradient_y**2 + gradient_x**2)
                artifact_mask = gradient_mag > threshold * 0.8

                if np.sum(artifact_mask) == 0:
                    break

            metric.corrected_count = corrected_count

            boundary_region = stitched_mask[y0:y1, x0:x1]
            gradient_y = sobel(boundary_region, axis=0)
            gradient_x = sobel(boundary_region, axis=1)
            gradient_mag = np.sqrt(gradient_y**2 + gradient_x**2)

            metric.post_correction_max = float(np.max(gradient_mag))
            metric.post_correction_mean = float(np.mean(gradient_mag))
            metric.correction_improvement = (
                metric.mean_discontinuity - metric.post_correction_mean
            ) / max(metric.mean_discontinuity, 1e-8)

        return metric

    def _correct_artifact_pixel(
        self,
        mask: np.ndarray,
        y: int,
        x: int,
        window: int,
        boundary: BoundaryInfo,
    ) -> Optional[float]:
        """
        修正单个伪影像素

        Args:
            mask: 完整掩模
            y: 像素 y 坐标
            x: 像素 x 坐标
            window: 窗口大小
            boundary: 边界信息

        Returns:
            修正后的像素值，或 None 表示不修正
        """
        h, w = mask.shape
        half_w = window // 2

        y0 = max(0, y - half_w)
        y1 = min(h, y + half_w + 1)
        x0 = max(0, x - half_w)
        x1 = min(w, x + half_w + 1)

        if y1 <= y0 or x1 <= x0:
            return None

        local_region = mask[y0:y1, x0:x1]

        if boundary.orientation == 'horizontal':
            upper = local_region[:half_w, :]
            lower = local_region[half_w+1:, :]

            if upper.size > 0 and lower.size > 0:
                upper_mean = np.mean(upper)
                lower_mean = np.mean(lower)
                return (upper_mean + lower_mean) / 2
        else:
            left = local_region[:, :half_w]
            right = local_region[:, half_w+1:]

            if left.size > 0 and right.size > 0:
                left_mean = np.mean(left)
                right_mean = np.mean(right)
                return (left_mean + right_mean) / 2

        return np.mean(local_region)

    def _apply_gradient_blending(
        self,
        stitched_mask: np.ndarray,
        boundaries: List[BoundaryInfo],
        regions: List[ChipRegion],
    ) -> np.ndarray:
        """
        应用梯度融合，平滑边界过渡

        Args:
            stitched_mask: 拼接后的掩模
            boundaries: 边界列表
            regions: 区域列表

        Returns:
            融合后的掩模
        """
        sigma = self.config.boundary_smooth_sigma_px
        if sigma <= 0:
            return stitched_mask

        result = stitched_mask.copy()

        for boundary in boundaries:
            y0, y1, x0, x1 = boundary.bounds_px
            feather = self.config.feathering_width_px

            if feather <= 0:
                continue

            expanded_y0 = max(0, y0 - feather)
            expanded_y1 = min(stitched_mask.shape[0], y1 + feather)
            expanded_x0 = max(0, x0 - feather)
            expanded_x1 = min(stitched_mask.shape[1], x1 + feather)

            if expanded_y1 <= expanded_y0 or expanded_x1 <= expanded_x0:
                continue

            local_region = result[expanded_y0:expanded_y1, expanded_x0:expanded_x1]
            smoothed_region = gaussian_filter(local_region, sigma=sigma)

            blend_mask = self._generate_blend_mask(
                (expanded_y1 - expanded_y0, expanded_x1 - expanded_x0),
                feather,
            )

            result[expanded_y0:expanded_y1, expanded_x0:expanded_x1] = (
                local_region * (1 - blend_mask) +
                smoothed_region * blend_mask
            )

        return result

    def _generate_blend_mask(
        self,
        shape: Tuple[int, int],
        feather: int,
    ) -> np.ndarray:
        """
        生成融合掩码，中心权重高，边缘权重低

        Args:
            shape: 形状
            feather: 羽化宽度

        Returns:
            融合掩码
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float64)

        if feather <= 0:
            return mask

        y_dist = np.minimum(np.arange(h), h - 1 - np.arange(h))
        x_dist = np.minimum(np.arange(w), w - 1 - np.arange(w))

        y_weight = np.clip(y_dist / feather, 0, 1)
        x_weight = np.clip(x_dist / feather, 0, 1)

        yw, xw = np.meshgrid(y_weight, x_weight, indexing='ij')
        mask = np.minimum(yw, xw)

        return mask

    def _apply_global_consistency(
        self,
        stitched_mask: np.ndarray,
        regions: List[ChipRegion],
    ) -> np.ndarray:
        """
        应用全局一致性约束

        Args:
            stitched_mask: 拼接后的掩模
            regions: 区域列表

        Returns:
            一致性修正后的掩模
        """
        consistency_weight = self.config.consistency_weight
        if consistency_weight <= 0:
            return stitched_mask

        result = stitched_mask.copy()

        original_mask = np.zeros_like(stitched_mask)
        weight = np.zeros_like(stitched_mask)

        for region in regions:
            if region.mask is None:
                continue

            bounds_px = region.metadata.bounds_px
            if bounds_px is None:
                continue

            y0, y1, x0, x1 = bounds_px
            h, w = region.mask.shape
            h = min(h, y1 - y0)
            w = min(w, x1 - x0)

            original_mask[y0:y0+h, x0:x0+w] += region.mask[:h, :w]
            weight[y0:y0+h, x0:x0+w] += 1.0

        weight = np.where(weight > 0, weight, 1.0)
        original_mask = original_mask / weight

        result = (
            result * (1 - consistency_weight) +
            original_mask * consistency_weight
        )

        return result

    def _finalize_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        最终化掩模，确保数值范围正确

        Args:
            mask: 拼接后的掩模

        Returns:
            最终的掩模
        """
        mask = np.clip(mask, 0, 1)

        if mask.dtype != np.float64:
            mask = mask.astype(np.float64)

        return mask

    def evaluate_boundary_quality(
        self,
        stitched_mask: np.ndarray,
        regions: List[ChipRegion],
        full_shape: Tuple[int, int],
        chip_origin_nm: Tuple[float, float] = (0.0, 0.0),
    ) -> List[BoundaryArtifactMetrics]:
        """
        评估边界质量（不进行修正）

        Args:
            stitched_mask: 拼接后的掩模
            regions: 区域列表
            full_shape: 完整形状
            chip_origin_nm: 芯片原点

        Returns:
            边界伪影指标列表
        """
        pixel_size = regions[0].metadata.pixel_size_nm if regions else 1.0

        boundaries = self._find_boundaries(
            regions, full_shape, chip_origin_nm, pixel_size
        )

        metrics = []
        for boundary in boundaries:
            metric = self._evaluate_boundary(stitched_mask, boundary)
            metrics.append(metric)

        return metrics

    def _evaluate_boundary(
        self,
        stitched_mask: np.ndarray,
        boundary: BoundaryInfo,
    ) -> BoundaryArtifactMetrics:
        """
        评估单条边界的质量

        Args:
            stitched_mask: 拼接后的掩模
            boundary: 边界信息

        Returns:
            边界伪影指标
        """
        y0, y1, x0, x1 = boundary.bounds_px
        threshold = self.config.artifact_detection_threshold

        metric = BoundaryArtifactMetrics(
            boundary_id=boundary.boundary_id,
        )

        boundary_region = stitched_mask[y0:y1, x0:x1]
        if boundary_region.size == 0:
            return metric

        gradient_y = sobel(boundary_region, axis=0)
        gradient_x = sobel(boundary_region, axis=1)
        gradient_mag = np.sqrt(gradient_y**2 + gradient_x**2)

        if boundary.orientation == 'horizontal' and gradient_mag.shape[0] > 0:
            cross_section = gradient_mag[gradient_mag.shape[0]//2, :]
        elif gradient_mag.shape[1] > 0:
            cross_section = gradient_mag[:, gradient_mag.shape[1]//2]
        else:
            cross_section = np.array([])

        metric.max_discontinuity = float(np.max(gradient_mag))
        metric.mean_discontinuity = float(np.mean(gradient_mag))
        metric.std_discontinuity = float(np.std(gradient_mag))
        metric.max_gradient_jump = float(np.max(np.abs(np.diff(cross_section)))) if len(cross_section) > 1 else 0.0
        metric.mean_gradient_jump = float(np.mean(np.abs(np.diff(cross_section)))) if len(cross_section) > 1 else 0.0

        artifact_mask = gradient_mag > threshold
        metric.artifact_pixel_count = int(np.sum(artifact_mask))
        metric.artifact_density = float(metric.artifact_pixel_count / gradient_mag.size)

        return metric

    def get_stitching_summary(
        self,
        metrics: List[BoundaryArtifactMetrics],
    ) -> Dict[str, Any]:
        """
        获取拼接质量统计摘要

        Args:
            metrics: 边界伪影指标列表

        Returns:
            统计摘要
        """
        if not metrics:
            return {
                'num_boundaries': 0,
                'total_artifacts': 0,
                'avg_artifact_density': 0.0,
                'max_discontinuity': 0.0,
                'mean_discontinuity': 0.0,
                'total_corrections': 0,
                'avg_correction_improvement': 0.0,
            }

        total_artifacts = sum(m.artifact_pixel_count for m in metrics)
        total_corrections = sum(m.corrected_count for m in metrics)
        avg_density = float(np.mean([m.artifact_density for m in metrics]))
        max_disc = float(max(m.post_correction_max for m in metrics))
        mean_disc = float(np.mean([m.post_correction_mean for m in metrics]))

        improvements = [m.correction_improvement for m in metrics if m.corrected_count > 0]
        avg_improvement = float(np.mean(improvements)) if improvements else 0.0

        return {
            'num_boundaries': len(metrics),
            'total_artifacts': total_artifacts,
            'avg_artifact_density': avg_density,
            'max_discontinuity': max_disc,
            'mean_discontinuity': mean_disc,
            'total_corrections': total_corrections,
            'avg_correction_improvement': avg_improvement,
        }
