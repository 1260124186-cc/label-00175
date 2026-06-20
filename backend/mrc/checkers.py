# -*- coding: utf-8 -*-
"""
MRC 核心规则检查引擎

实现最小线宽、最小间距、最小 SRAF 尺寸、禁止锐角、辅助特征与主特征最小距离等规则检查。
"""

import logging
import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
    find_objects,
    center_of_mass,
)
from skimage.morphology import skeletonize, medial_axis
from skimage.measure import regionprops, label as sk_label

from .rules import (
    MRCRules,
    MRCRuleType,
    MRCRuleConfig,
    MRCRuleSeverity,
)
from .violations import (
    MRCViolation,
    MRCCheckResult,
    ViolationRegion,
    ViolationType,
)

logger = logging.getLogger(__name__)


class MRCChecker:
    """
    MRC 规则检查器

    对二值掩模执行制造规则检查。

    使用方法:
        checker = MRCChecker(rules)
        result = checker.check(mask_array, pixel_size=1.0)
    """

    def __init__(self, rules: Optional[MRCRules] = None):
        self.rules = rules or MRCRules()

    def check(self,
              mask: np.ndarray,
              pixel_size: float = 1.0,
              target_mask: Optional[np.ndarray] = None,
              ) -> MRCCheckResult:
        """
        执行完整的 MRC 检查

        Args:
            mask: 二值掩模数组 (H, W)，值域 [0, 1] 或 [0, 255]
            pixel_size: 像素尺寸 (nm)
            target_mask: 目标掩模（可选，用于区分主特征和辅助特征）

        Returns:
            MRCCheckResult 检查结果
        """
        t_start = time.time()

        mask_binary = self._preprocess_mask(mask)
        result = MRCCheckResult(
            mask_shape=mask_binary.shape,
            pixel_size=pixel_size,
            timestamp=time.time(),
        )

        enabled_rules = self.rules.enabled_rules()
        result.rules_checked = [rt.value for rt in enabled_rules.keys()]

        main_features, sraf_features = self._separate_features(
            mask_binary, target_mask
        )

        for rule_type, rule_config in enabled_rules.items():
            try:
                violations = self._check_single_rule(
                    rule_type, rule_config,
                    mask_binary, main_features, sraf_features,
                    pixel_size,
                )
                if violations:
                    result.add_violations(violations)
                    logger.info(
                        f"规则 {rule_type.value}: 发现 {len(violations)} 处违规"
                    )
            except Exception as e:
                logger.error(f"检查规则 {rule_type.value} 时出错: {e}")

        result.check_duration_sec = time.time() - t_start
        return result

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_mask(mask: np.ndarray) -> np.ndarray:
        """预处理掩模为二值布尔数组"""
        if mask.ndim != 2:
            raise ValueError(f"掩模必须是 2D 数组，当前形状: {mask.shape}")

        if mask.dtype == np.bool_:
            return mask

        if np.max(mask) <= 1.0:
            return mask > 0.5
        else:
            return mask > 127

    def _separate_features(self,
                           mask: np.ndarray,
                           target_mask: Optional[np.ndarray],
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分离主特征和辅助特征 (SRAF)

        如果提供了 target_mask，则:
        - 主特征 = mask ∩ target_mask
        - 辅助特征 = mask ∩ ~target_mask
        否则根据连通域面积自动区分（小面积为 SRAF）
        """
        if target_mask is not None:
            target_binary = self._preprocess_mask(target_mask)
            main_features = mask & target_binary
            sraf_features = mask & (~target_binary)
            return main_features, sraf_features

        labeled, num_features = label(mask)
        if num_features == 0:
            return mask, np.zeros_like(mask, dtype=bool)

        areas = ndimage.sum(mask.astype(int), labeled, range(1, num_features + 1))
        if len(areas) == 0:
            return mask, np.zeros_like(mask, dtype=bool)

        median_area = np.median(areas)
        sraf_threshold = max(median_area * 0.3, 4)

        main_features = np.zeros_like(mask, dtype=bool)
        sraf_features = np.zeros_like(mask, dtype=bool)

        for i in range(1, num_features + 1):
            feature_mask = labeled == i
            area = areas[i - 1]
            if area >= sraf_threshold:
                main_features |= feature_mask
            else:
                sraf_features |= feature_mask

        return main_features, sraf_features

    # ------------------------------------------------------------------
    # 单规则调度
    # ------------------------------------------------------------------

    def _check_single_rule(self,
                           rule_type: MRCRuleType,
                           rule_config: MRCRuleConfig,
                           mask: np.ndarray,
                           main_features: np.ndarray,
                           sraf_features: np.ndarray,
                           pixel_size: float,
                           ) -> List[MRCViolation]:
        """根据规则类型分发到具体检查方法"""
        dispatch = {
            MRCRuleType.MIN_LINE_WIDTH: self._check_min_line_width,
            MRCRuleType.MIN_SPACING: self._check_min_spacing,
            MRCRuleType.MIN_SRAF_SIZE: self._check_min_sraf_size,
            MRCRuleType.NO_ACUTE_ANGLE: self._check_no_acute_angle,
            MRCRuleType.SRAF_MAIN_DISTANCE: self._check_sraf_main_distance,
        }

        checker_fn = dispatch.get(rule_type)
        if checker_fn is None:
            logger.warning(f"未实现规则检查: {rule_type.value}")
            return []

        return checker_fn(rule_config, mask, main_features, sraf_features, pixel_size)

    # ------------------------------------------------------------------
    # 最小线宽检查
    # ------------------------------------------------------------------

    def _check_min_line_width(self,
                              rule_config: MRCRuleConfig,
                              mask: np.ndarray,
                              main_features: np.ndarray,
                              sraf_features: np.ndarray,
                              pixel_size: float,
                              ) -> List[MRCViolation]:
        """
        最小线宽检查

        原理: 对掩模进行距离变换，距离变换值代表该点到最近背景的距离。
        对于掩模上的点，局部最大距离值的一半代表该处的线宽。
        """
        threshold_nm = rule_config.threshold_nm
        threshold_px = threshold_nm / pixel_size

        if not np.any(main_features):
            return []

        dist = distance_transform_edt(main_features.astype(float))
        dist_nm = dist * pixel_size
        line_width = dist * 2.0 * pixel_size

        skeleton = skeletonize(main_features)
        width_at_skeleton = line_width.copy()
        width_at_skeleton[~skeleton] = np.inf

        violation_mask = (width_at_skeleton < threshold_nm) & skeleton
        if not np.any(violation_mask):
            return []

        return self._mask_to_violations(
            violation_mask=violation_mask,
            mask=mask,
            rule_type=MRCRuleType.MIN_LINE_WIDTH,
            violation_type=ViolationType.LINE_WIDTH_TOO_SMALL,
            severity=rule_config.severity,
            pixel_size=pixel_size,
            threshold_nm=threshold_nm,
            measurement_array=width_at_skeleton,
            message_prefix="线宽过小",
        )

    # ------------------------------------------------------------------
    # 最小间距检查
    # ------------------------------------------------------------------

    def _check_min_spacing(self,
                           rule_config: MRCRuleConfig,
                           mask: np.ndarray,
                           main_features: np.ndarray,
                           sraf_features: np.ndarray,
                           pixel_size: float,
                           ) -> List[MRCViolation]:
        """
        最小间距检查

        原理: 对背景进行距离变换，背景点的距离值代表到最近掩模的距离。
        背景骨架点的距离值的两倍即代表间距。
        """
        threshold_nm = rule_config.threshold_nm
        threshold_px = threshold_nm / pixel_size

        all_features = main_features | sraf_features
        if not np.any(all_features):
            return []

        background = ~all_features
        dist_bg = distance_transform_edt(background.astype(float))
        spacing_nm = dist_bg * 2.0 * pixel_size

        bg_skeleton = skeletonize(background)
        spacing_at_skeleton = spacing_nm.copy()
        spacing_at_skeleton[~bg_skeleton] = np.inf

        violation_mask = (spacing_at_skeleton < threshold_nm) & bg_skeleton
        if not np.any(violation_mask):
            return []

        return self._mask_to_violations(
            violation_mask=violation_mask,
            mask=mask,
            rule_type=MRCRuleType.MIN_SPACING,
            violation_type=ViolationType.SPACING_TOO_SMALL,
            severity=rule_config.severity,
            pixel_size=pixel_size,
            threshold_nm=threshold_nm,
            measurement_array=spacing_at_skeleton,
            message_prefix="间距过小",
        )

    # ------------------------------------------------------------------
    # 最小 SRAF 尺寸检查
    # ------------------------------------------------------------------

    def _check_min_sraf_size(self,
                             rule_config: MRCRuleConfig,
                             mask: np.ndarray,
                             main_features: np.ndarray,
                             sraf_features: np.ndarray,
                             pixel_size: float,
                             ) -> List[MRCViolation]:
        """
        最小 SRAF 尺寸检查

        对每个辅助特征连通域检查其尺寸是否小于阈值。
        """
        threshold_nm = rule_config.threshold_nm
        threshold_area_nm2 = rule_config.extra_params.get(
            "min_area_nm2", threshold_nm * threshold_nm
        )
        min_dim_px = threshold_nm / pixel_size
        min_area_px = threshold_area_nm2 / (pixel_size * pixel_size)

        if not np.any(sraf_features):
            return []

        labeled, num_features = label(sraf_features)
        if num_features == 0:
            return []

        violations = []
        objects = find_objects(labeled)

        for i, obj_slice in enumerate(objects):
            if obj_slice is None:
                continue

            feature_mask = labeled[obj_slice] == (i + 1)
            area_px = int(np.sum(feature_mask))
            area_nm2 = area_px * (pixel_size ** 2)

            if area_px == 0:
                continue

            h, w = feature_mask.shape
            min_dim_px_actual = min(h, w)
            min_dim_nm = min_dim_px_actual * pixel_size

            if area_nm2 < threshold_area_nm2 or min_dim_nm < threshold_nm:
                bbox_full = (
                    obj_slice[0].start,
                    obj_slice[1].start,
                    obj_slice[0].stop,
                    obj_slice[1].stop,
                )
                cy, cx = center_of_mass(feature_mask)
                centroid = (
                    cy + obj_slice[0].start,
                    cx + obj_slice[1].start,
                )

                region = ViolationRegion(
                    bbox=bbox_full,
                    centroid=centroid,
                    area_pixels=area_px,
                    mask_slice=feature_mask,
                )

                measurement = min(area_nm2, min_dim_nm * pixel_size)
                message = (
                    f"SRAF 尺寸过小: 最小尺寸 {min_dim_nm:.1f}nm "
                    f"(阈值 {threshold_nm:.1f}nm), "
                    f"面积 {area_nm2:.0f}nm² (阈值 {threshold_area_nm2:.0f}nm²)"
                )

                violations.append(MRCViolation(
                    rule_type=MRCRuleType.MIN_SRAF_SIZE,
                    violation_type=ViolationType.SRAF_TOO_SMALL,
                    severity=rule_config.severity,
                    message=message,
                    region=region,
                    measurement_nm=min_dim_nm,
                    threshold_nm=threshold_nm,
                    pixel_size=pixel_size,
                    extra_info={
                        "area_nm2": area_nm2,
                        "min_dim_nm": min_dim_nm,
                        "threshold_area_nm2": threshold_area_nm2,
                    },
                ))

        return violations

    # ------------------------------------------------------------------
    # 禁止锐角检查
    # ------------------------------------------------------------------

    def _check_no_acute_angle(self,
                              rule_config: MRCRuleConfig,
                              mask: np.ndarray,
                              main_features: np.ndarray,
                              sraf_features: np.ndarray,
                              pixel_size: float,
                              ) -> List[MRCViolation]:
        """
        禁止锐角检查

        原理: 使用形态学操作检测拐角区域。
        对掩模进行角点检测，找到内角小于阈值的区域。
        """
        min_angle_deg = rule_config.extra_params.get("min_angle_deg", 90.0)

        all_features = main_features | sraf_features
        if not np.any(all_features):
            return []

        violations = []
        corner_mask = self._detect_corners(all_features, min_angle_deg)

        if not np.any(corner_mask):
            return []

        labeled, num_corners = label(corner_mask)
        if num_corners == 0:
            return []

        objects = find_objects(labeled)

        for i, obj_slice in enumerate(objects):
            if obj_slice is None:
                continue

            corner_feature = labeled[obj_slice] == (i + 1)
            area_px = int(np.sum(corner_feature))

            bbox_full = (
                obj_slice[0].start,
                obj_slice[1].start,
                obj_slice[0].stop,
                obj_slice[1].stop,
            )
            cy, cx = center_of_mass(corner_feature)
            centroid = (
                cy + obj_slice[0].start,
                cx + obj_slice[1].start,
            )

            estimated_angle = self._estimate_corner_angle(
                all_features, int(centroid[0]), int(centroid[1])
            )

            region = ViolationRegion(
                bbox=bbox_full,
                centroid=centroid,
                area_pixels=area_px,
                mask_slice=corner_feature,
            )

            message = (
                f"检测到锐角: 估计角度约 {estimated_angle:.1f}° "
                f"(最小允许 {min_angle_deg:.1f}°)"
            )

            violations.append(MRCViolation(
                rule_type=MRCRuleType.NO_ACUTE_ANGLE,
                violation_type=ViolationType.ACUTE_ANGLE,
                severity=rule_config.severity,
                message=message,
                region=region,
                measurement_nm=estimated_angle,
                threshold_nm=min_angle_deg,
                pixel_size=pixel_size,
                extra_info={
                    "estimated_angle_deg": estimated_angle,
                    "min_angle_deg": min_angle_deg,
                },
            ))

        return violations

    @staticmethod
    def _detect_corners(mask: np.ndarray, min_angle_deg: float) -> np.ndarray:
        """
        检测掩模中的锐角拐角

        使用多方向形态学操作检测角点。
        """
        from scipy.ndimage import convolve

        struct_small = np.ones((3, 3), dtype=bool)
        eroded = binary_erosion(mask, structure=struct_small)
        edge = mask.astype(int) - eroded.astype(int)
        edge = edge > 0

        k_size = 5
        response = np.zeros_like(mask, dtype=float)

        angles = [0, 45, 90, 135]
        for angle_deg in angles:
            rad = np.radians(angle_deg)
            for corner_type in ["convex", "concave"]:
                kernel = MRCChecker._build_corner_kernel(k_size, rad, corner_type)
                if kernel is None:
                    continue
                conv = convolve(edge.astype(float), kernel)
                response = np.maximum(response, conv)

        kernel_sum = k_size * k_size
        threshold = kernel_sum * 0.5
        corners = response >= threshold
        corners = corners & edge

        corners_cleaned = binary_erosion(
            binary_dilation(corners, structure=np.ones((3, 3))),
            structure=np.ones((3, 3))
        )

        return corners_cleaned

    @staticmethod
    def _build_corner_kernel(size: int, angle_rad: float, corner_type: str):
        """构建角点检测核"""
        if size % 2 == 0:
            size += 1
        half = size // 2
        kernel = np.zeros((size, size), dtype=float)
        cy, cx = half, half

        angle1 = angle_rad
        angle2 = angle_rad + np.radians(90)
        if corner_type == "concave":
            angle1 += np.pi
            angle2 += np.pi

        for y in range(size):
            for x in range(size):
                dy = y - cy
                dx = x - cx
                if dx == 0 and dy == 0:
                    kernel[y, x] = 1.0
                    continue

                point_angle = np.arctan2(dy, dx)

                def angle_between(a, b):
                    d = abs(a - b)
                    d = min(d, 2 * np.pi - d)
                    return d

                d1 = angle_between(point_angle, angle1)
                d2 = angle_between(point_angle, angle2)
                if min(d1, d2) < np.radians(22.5):
                    kernel[y, x] = 1.0

        if np.sum(kernel) == 0:
            return None
        return kernel / np.sum(kernel)

    @staticmethod
    def _estimate_corner_angle(mask: np.ndarray, cy: int, cx: int,
                               radius: int = 8) -> float:
        """估计拐角处的角度"""
        h, w = mask.shape
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)

        local = mask[y0:y1, x0:x1].astype(int)
        if local.size == 0:
            return 90.0

        edge_pixels = []
        for ly in range(local.shape[0]):
            for lx in range(local.shape[1]):
                if local[ly, lx] > 0:
                    gy = ly + y0 - cy
                    gx = lx + x0 - cx
                    dist = np.sqrt(gy * gy + gx * gx)
                    if 2 <= dist <= radius:
                        edge_pixels.append((gy, gx))

        if len(edge_pixels) < 4:
            return 90.0

        angles = []
        for gy, gx in edge_pixels:
            ang = np.degrees(np.arctan2(gy, gx))
            if ang < 0:
                ang += 360
            angles.append(ang)

        angles.sort()
        if len(angles) < 2:
            return 90.0

        max_gap = 0.0
        for i in range(len(angles)):
            a1 = angles[i]
            a2 = angles[(i + 1) % len(angles)]
            if i == len(angles) - 1:
                gap = (a2 + 360) - a1
            else:
                gap = a2 - a1
            if gap > max_gap:
                max_gap = gap

        internal_angle = 360.0 - max_gap
        if internal_angle > 180:
            internal_angle = 360.0 - internal_angle

        return max(10.0, min(170.0, internal_angle))

    # ------------------------------------------------------------------
    # SRAF 与主特征最小距离检查
    # ------------------------------------------------------------------

    def _check_sraf_main_distance(self,
                                  rule_config: MRCRuleConfig,
                                  mask: np.ndarray,
                                  main_features: np.ndarray,
                                  sraf_features: np.ndarray,
                                  pixel_size: float,
                                  ) -> List[MRCViolation]:
        """
        辅助特征 (SRAF) 与主特征最小/最大距离检查
        """
        min_dist_nm = rule_config.threshold_nm
        max_dist_nm = rule_config.extra_params.get("max_distance_nm", 150.0)

        if not np.any(main_features) or not np.any(sraf_features):
            return []

        dist_to_main = distance_transform_edt(~main_features.astype(float))
        dist_to_main_nm = dist_to_main * pixel_size

        violations = []

        labeled_sraf, num_sraf = label(sraf_features)
        if num_sraf == 0:
            return []

        objects = find_objects(labeled_sraf)

        for i, obj_slice in enumerate(objects):
            if obj_slice is None:
                continue

            sraf_mask_local = labeled_sraf[obj_slice] == (i + 1)
            if not np.any(sraf_mask_local):
                continue

            dists_local = dist_to_main_nm[obj_slice]
            sraf_dists = dists_local[sraf_mask_local]

            if len(sraf_dists) == 0:
                continue

            min_actual_nm = float(np.min(sraf_dists))
            max_actual_nm = float(np.max(sraf_dists))

            area_px = int(np.sum(sraf_mask_local))
            bbox_full = (
                obj_slice[0].start,
                obj_slice[1].start,
                obj_slice[0].stop,
                obj_slice[1].stop,
            )
            cy, cx = center_of_mass(sraf_mask_local)
            centroid = (
                cy + obj_slice[0].start,
                cx + obj_slice[1].start,
            )

            region = ViolationRegion(
                bbox=bbox_full,
                centroid=centroid,
                area_pixels=area_px,
                mask_slice=sraf_mask_local,
            )

            if min_actual_nm < min_dist_nm:
                message = (
                    f"SRAF 距主特征过近: 最小距离 {min_actual_nm:.1f}nm "
                    f"(阈值 {min_dist_nm:.1f}nm)"
                )
                violations.append(MRCViolation(
                    rule_type=MRCRuleType.SRAF_MAIN_DISTANCE,
                    violation_type=ViolationType.SRAF_TOO_CLOSE_TO_MAIN,
                    severity=rule_config.severity,
                    message=message,
                    region=region,
                    measurement_nm=min_actual_nm,
                    threshold_nm=min_dist_nm,
                    pixel_size=pixel_size,
                    extra_info={
                        "min_distance_nm": min_actual_nm,
                        "max_distance_nm": max_actual_nm,
                        "distance_type": "too_close",
                    },
                ))

            if max_actual_nm > max_dist_nm:
                message = (
                    f"SRAF 距主特征过远: 最大距离 {max_actual_nm:.1f}nm "
                    f"(阈值 {max_dist_nm:.1f}nm)"
                )
                violations.append(MRCViolation(
                    rule_type=MRCRuleType.SRAF_MAIN_DISTANCE,
                    violation_type=ViolationType.SRAF_TOO_FAR_FROM_MAIN,
                    severity=MRCRuleSeverity.WARNING,
                    message=message,
                    region=region,
                    measurement_nm=max_actual_nm,
                    threshold_nm=max_dist_nm,
                    pixel_size=pixel_size,
                    extra_info={
                        "min_distance_nm": min_actual_nm,
                        "max_distance_nm": max_actual_nm,
                        "distance_type": "too_far",
                    },
                ))

        return violations

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_to_violations(violation_mask: np.ndarray,
                            mask: np.ndarray,
                            rule_type: MRCRuleType,
                            violation_type: ViolationType,
                            severity: MRCRuleSeverity,
                            pixel_size: float,
                            threshold_nm: float,
                            measurement_array: Optional[np.ndarray] = None,
                            message_prefix: str = "违规",
                            ) -> List[MRCViolation]:
        """将违规掩模转换为违规记录列表"""
        if not np.any(violation_mask):
            return []

        labeled, num_regions = label(violation_mask)
        if num_regions == 0:
            return []

        objects = find_objects(labeled)
        violations = []

        for i, obj_slice in enumerate(objects):
            if obj_slice is None:
                continue

            local_mask = labeled[obj_slice] == (i + 1)
            area_px = int(np.sum(local_mask))

            if area_px == 0:
                continue

            bbox_full = (
                obj_slice[0].start,
                obj_slice[1].start,
                obj_slice[0].stop,
                obj_slice[1].stop,
            )

            cy, cx = center_of_mass(local_mask)
            centroid = (
                float(cy + obj_slice[0].start),
                float(cx + obj_slice[1].start),
            )

            measurement_nm = threshold_nm
            if measurement_array is not None:
                local_meas = measurement_array[obj_slice]
                valid_meas = local_meas[local_mask & np.isfinite(local_meas)]
                if len(valid_meas) > 0:
                    measurement_nm = float(np.mean(valid_meas))

            region = ViolationRegion(
                bbox=bbox_full,
                centroid=centroid,
                area_pixels=area_px,
                mask_slice=local_mask,
            )

            message = (
                f"{message_prefix}: 测量值 {measurement_nm:.1f}nm "
                f"(阈值 {threshold_nm:.1f}nm)"
            )

            violations.append(MRCViolation(
                rule_type=rule_type,
                violation_type=violation_type,
                severity=severity,
                message=message,
                region=region,
                measurement_nm=measurement_nm,
                threshold_nm=threshold_nm,
                pixel_size=pixel_size,
            ))

        return violations
