# -*- coding: utf-8 -*-
"""
版图设计规则邻近分析模块 - 孤立线检查器

检测版图中的孤立特征:
1. 面积过小的孤立特征 (small isolated feature)
2. 悬空短线段 (dangling line)
3. 孤儿像素团 (orphan pixel cluster)
"""

import logging
from typing import List

import numpy as np
from scipy.ndimage import distance_transform_edt, label, find_objects, center_of_mass

from .schemas import (
    LithoViolation,
    LithoViolationCategory,
    LithoViolationType,
    LithoSeverity,
    OPCFeasibility,
    ViolationRegion,
)
from .rules import LithoRuleType, LithoRuleConfig
from .geometry import (
    preprocess_mask,
    compute_component_properties,
    find_dangling_lines,
    find_orphan_pixels,
)

logger = logging.getLogger(__name__)


def check_isolated_small_feature(
    mask: np.ndarray,
    rule_config: LithoRuleConfig,
    pixel_size: float,
) -> List[LithoViolation]:
    min_area_nm2 = rule_config.extra_params.get(
        "min_area_nm2", rule_config.threshold_nm ** 2
    )
    max_aspect_ratio = rule_config.extra_params.get("max_aspect_ratio", 5.0)
    min_dim_nm = rule_config.threshold_nm

    mask_bin = preprocess_mask(mask)
    if not np.any(mask_bin):
        return []

    props = compute_component_properties(mask_bin, pixel_size)
    if not props:
        return []

    from scipy.ndimage import label as nd_label, find_objects as nd_find_objects, center_of_mass as nd_com
    struct = np.ones((3, 3), dtype=bool)
    labeled, num = nd_label(mask_bin, structure=struct)

    main_features = np.zeros_like(mask_bin, dtype=bool)
    all_features = np.zeros_like(mask_bin, dtype=bool)
    for i in range(1, num + 1):
        all_features |= (labeled == i)

    dist_to_main = distance_transform_edt((~main_features).astype(float))
    dist_to_all = distance_transform_edt((~all_features).astype(float))

    objects = nd_find_objects(labeled)
    violations = []

    for i, obj_slice in enumerate(objects):
        if obj_slice is None:
            continue

        local = labeled[obj_slice] == (i + 1)
        area_px = int(np.sum(local))
        if area_px == 0:
            continue

        area_nm2 = area_px * (pixel_size ** 2)
        min_axis_nm = props[i].get("minor_axis_nm", 0.0) if i < len(props) else 0.0
        eccentricity = props[i].get("eccentricity", 0.0) if i < len(props) else 0.0
        major_nm = props[i].get("major_axis_nm", 0.0) if i < len(props) else 0.0
        minor_nm = props[i].get("minor_axis_nm", 0.0) if i < len(props) else 0.0
        aspect = major_nm / minor_nm if minor_nm > 0 else 999.0

        is_near_other = False
        local_dist = dist_to_all[obj_slice]
        local_mask = local
        border_dist = local_dist[local_mask]
        if len(border_dist) > 0:
            min_border_dist_px = float(np.min(border_dist))
            if min_border_dist_px * pixel_size < min_dim_nm * 3:
                is_near_other = True

        if area_nm2 < min_area_nm2 or (min_axis_nm < min_dim_nm and aspect > max_aspect_ratio):
            if is_near_other and area_nm2 >= min_area_nm2 * 0.5:
                continue

            cy, cx = nd_com(local)
            centroid = (
                float(cy + obj_slice[0].start),
                float(cx + obj_slice[1].start),
            )
            bbox_full = (
                obj_slice[0].start,
                obj_slice[1].start,
                obj_slice[0].stop,
                obj_slice[1].stop,
            )
            region = ViolationRegion(
                bbox=bbox_full,
                centroid=centroid,
                area_pixels=area_px,
                mask_slice=local,
            )

            measurement = min(area_nm2, min_axis_nm)
            message = (
                f"孤立线风险-小特征: 面积 {area_nm2:.0f}nm² (阈值 {min_area_nm2:.0f}nm²), "
                f"最小轴 {min_axis_nm:.1f}nm (阈值 {min_dim_nm:.1f}nm), "
                f"可能无法有效成像"
            )

            violations.append(LithoViolation(
                category=LithoViolationCategory.ISOLATED,
                violation_type=LithoViolationType.ISOLATED_SMALL_FEATURE,
                severity=LithoSeverity(rule_config.severity.value),
                message=message,
                region=region,
                measurement_nm=min_axis_nm,
                threshold_nm=min_dim_nm,
                pixel_size=pixel_size,
                opc_feasibility=OPCFeasibility.NEEDS_REDESIGN,
                extra_info={
                    "area_nm2": area_nm2,
                    "min_axis_nm": min_axis_nm,
                    "aspect_ratio": aspect,
                    "min_area_nm2": min_area_nm2,
                },
            ))

    return violations


def check_isolated_dangling_line(
    mask: np.ndarray,
    rule_config: LithoRuleConfig,
    pixel_size: float,
) -> List[LithoViolation]:
    min_length_nm = rule_config.extra_params.get(
        "min_length_nm", rule_config.threshold_nm
    )
    max_width_nm = rule_config.extra_params.get("max_width_nm", 3.0)

    min_length_px = min_length_nm / pixel_size
    max_width_px = max_width_nm / pixel_size

    mask_bin = preprocess_mask(mask)
    if not np.any(mask_bin):
        return []

    dangling_list = find_dangling_lines(mask_bin, min_length_px, max_width_px)
    violations = []

    for dangling in dangling_list:
        centroid = dangling["centroid"]
        bbox = dangling["bbox"]
        length_px = dangling["length_px"]
        mean_width_px = dangling["mean_width_px"]
        max_width_px_val = dangling["max_width_px"]

        length_nm = length_px * pixel_size
        mean_width_nm = mean_width_px * pixel_size

        region = ViolationRegion(
            bbox=bbox,
            centroid=centroid,
            area_pixels=int(length_px * mean_width_px),
        )

        message = (
            f"孤立线风险-悬空线: 长度 {length_nm:.1f}nm, "
            f"平均线宽 {mean_width_nm:.1f}nm, "
            f"端点数 {dangling['num_endpoints']}，"
            f"至少一端无连接"
        )

        violations.append(LithoViolation(
            category=LithoViolationCategory.ISOLATED,
            violation_type=LithoViolationType.ISOLATED_DANGLING_LINE,
            severity=LithoSeverity(rule_config.severity.value),
            message=message,
            region=region,
            measurement_nm=length_nm,
            threshold_nm=min_length_nm,
            pixel_size=pixel_size,
            opc_feasibility=OPCFeasibility.FIXABLE,
            extra_info={
                "length_nm": length_nm,
                "mean_width_nm": mean_width_nm,
                "max_width_nm": max_width_px_val * pixel_size,
                "num_endpoints": dangling["num_endpoints"],
            },
        ))

    return violations


def check_isolated_orphan_pixel(
    mask: np.ndarray,
    rule_config: LithoRuleConfig,
    pixel_size: float,
) -> List[LithoViolation]:
    max_area_px = rule_config.extra_params.get("max_area_px", 4)

    mask_bin = preprocess_mask(mask)
    if not np.any(mask_bin):
        return []

    orphans = find_orphan_pixels(mask_bin, max_area_px)
    violations = []

    for orphan in orphans:
        centroid = orphan["centroid"]
        bbox = orphan["bbox"]
        area_px = orphan["area_pixels"]
        area_nm2 = area_px * (pixel_size ** 2)

        region = ViolationRegion(
            bbox=bbox,
            centroid=centroid,
            area_pixels=area_px,
        )

        message = (
            f"孤立线风险-孤儿像素: 面积 {area_px}px ({area_nm2:.1f}nm²)，"
            f"可能是版图噪声"
        )

        violations.append(LithoViolation(
            category=LithoViolationCategory.ISOLATED,
            violation_type=LithoViolationType.ISOLATED_ORPHAN_PIXEL,
            severity=LithoSeverity(rule_config.severity.value),
            message=message,
            region=region,
            measurement_nm=float(area_px),
            threshold_nm=float(max_area_px),
            pixel_size=pixel_size,
            opc_feasibility=OPCFeasibility.FIXABLE,
            extra_info={
                "area_pixels": area_px,
                "area_nm2": area_nm2,
                "max_area_px": max_area_px,
            },
        ))

    return violations
