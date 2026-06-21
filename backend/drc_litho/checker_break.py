# -*- coding: utf-8 -*-
"""
版图设计规则邻近分析模块 - 断线风险检查器

检测版图中可能导致断线 (break/open) 的拓扑模式:
1. 线宽过窄的颈部区域 (thin neck)
2. 急转弯/锐角 (sharp turn)
3. 线端缩短效应 (line end thinning)
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
    detect_neck_regions,
    detect_sharp_turns,
    find_line_ends,
    compute_local_line_width,
    mask_to_regions,
)

logger = logging.getLogger(__name__)


def check_break_thin_neck(
    mask: np.ndarray,
    rule_config: LithoRuleConfig,
    pixel_size: float,
) -> List[LithoViolation]:
    min_width_nm = rule_config.extra_params.get(
        "min_width_nm", rule_config.threshold_nm
    )
    min_width_px = min_width_nm / pixel_size
    min_length_nm = rule_config.extra_params.get("length_nm", 20.0)

    mask_bin = preprocess_mask(mask)
    if not np.any(mask_bin):
        return []

    neck_mask = detect_neck_regions(mask_bin, min_width_px)
    if not np.any(neck_mask):
        return []

    local_width_map = distance_transform_edt(mask_bin.astype(float)) * 2.0
    local_width_nm = local_width_map * pixel_size

    from skimage.morphology import skeletonize
    skel = skeletonize(mask_bin)
    width_on_skel = local_width_nm.copy()
    width_on_skel[~skel] = np.inf

    regions = mask_to_regions(neck_mask, pixel_size, width_on_skel, min_width_nm)
    violations = []

    for item in regions:
        region = item["region"]
        meas = item["measurement_nm"]
        neck_length_nm = region.area_pixels * pixel_size

        if neck_length_nm < min_length_nm:
            continue

        if meas < min_width_nm * 0.5:
            opc_feas = OPCFeasibility.NEEDS_REDESIGN
        elif meas < min_width_nm * 0.8:
            opc_feas = OPCFeasibility.UNFIXABLE
        else:
            opc_feas = OPCFeasibility.PARTIAL

        message = (
            f"断线风险-线宽过窄: 颈部线宽 {meas:.1f}nm < 阈值 {min_width_nm:.1f}nm，"
            f"颈部长度 {neck_length_nm:.1f}nm，光刻后可能断裂"
        )
        violations.append(LithoViolation(
            category=LithoViolationCategory.BREAK,
            violation_type=LithoViolationType.BREAK_RISK_THIN_NECK,
            severity=LithoSeverity(rule_config.severity.value),
            message=message,
            region=region,
            measurement_nm=meas,
            threshold_nm=min_width_nm,
            pixel_size=pixel_size,
            opc_feasibility=opc_feas,
            extra_info={
                "neck_width_nm": meas,
                "neck_length_nm": neck_length_nm,
                "min_width_nm": min_width_nm,
                "min_length_nm": min_length_nm,
            },
        ))

    return violations


def check_break_sharp_turn(
    mask: np.ndarray,
    rule_config: LithoRuleConfig,
    pixel_size: float,
) -> List[LithoViolation]:
    min_angle_deg = rule_config.extra_params.get(
        "min_angle_deg", rule_config.threshold_nm
    )

    mask_bin = preprocess_mask(mask)
    if not np.any(mask_bin):
        return []

    sharp_mask = detect_sharp_turns(mask_bin, min_angle_deg)
    if not np.any(sharp_mask):
        return []

    labeled, num = label(sharp_mask)
    if num == 0:
        return []

    objects = find_objects(labeled)
    violations = []

    for i, obj_slice in enumerate(objects):
        if obj_slice is None:
            continue
        local = labeled[obj_slice] == (i + 1)
        area_px = int(np.sum(local))
        if area_px == 0:
            continue

        bbox_full = (
            obj_slice[0].start,
            obj_slice[1].start,
            obj_slice[0].stop,
            obj_slice[1].stop,
        )
        cy, cx = center_of_mass(local)
        centroid = (float(cy + obj_slice[0].start), float(cx + obj_slice[1].start))

        from .geometry import _estimate_local_angle
        estimated_angle = _estimate_local_angle(
            mask_bin, int(centroid[0]), int(centroid[1])
        )

        region = ViolationRegion(
            bbox=bbox_full,
            centroid=centroid,
            area_pixels=area_px,
            mask_slice=local,
        )

        if estimated_angle < min_angle_deg * 0.5:
            opc_feas = OPCFeasibility.UNFIXABLE
        else:
            opc_feas = OPCFeasibility.PARTIAL

        message = (
            f"断线风险-急转弯: 角度 {estimated_angle:.1f}° < 阈值 "
            f"{min_angle_deg:.1f}°，光刻后内侧可能断开"
        )
        violations.append(LithoViolation(
            category=LithoViolationCategory.BREAK,
            violation_type=LithoViolationType.BREAK_RISK_SHARP_TURN,
            severity=LithoSeverity(rule_config.severity.value),
            message=message,
            region=region,
            measurement_nm=estimated_angle,
            threshold_nm=min_angle_deg,
            pixel_size=pixel_size,
            opc_feasibility=opc_feas,
            extra_info={
                "estimated_angle_deg": estimated_angle,
                "min_angle_deg": min_angle_deg,
            },
        ))

    return violations


def check_break_line_end(
    mask: np.ndarray,
    rule_config: LithoRuleConfig,
    pixel_size: float,
) -> List[LithoViolation]:
    search_radius_nm = rule_config.extra_params.get(
        "search_radius_nm", rule_config.threshold_nm
    )
    search_radius_px = search_radius_nm / pixel_size

    mask_bin = preprocess_mask(mask)
    if not np.any(mask_bin):
        return []

    line_end_mask = find_line_ends(mask_bin, int(max(1, search_radius_px)))
    if not np.any(line_end_mask):
        return []

    local_width_map = distance_transform_edt(mask_bin.astype(float)) * 2.0
    local_width_nm = local_width_map * pixel_size

    line_end_region = line_end_mask & mask_bin
    thin_at_end = line_end_region & (local_width_nm < rule_config.threshold_nm)

    if not np.any(thin_at_end):
        return []

    regions = mask_to_regions(thin_at_end, pixel_size, local_width_nm, rule_config.threshold_nm)
    violations = []

    for item in regions:
        region = item["region"]
        meas = item["measurement_nm"]
        message = (
            f"断线风险-线端: 线端附近线宽 {meas:.1f}nm < 阈值 "
            f"{rule_config.threshold_nm:.1f}nm，线端缩短效应可能导致断线"
        )
        violations.append(LithoViolation(
            category=LithoViolationCategory.BREAK,
            violation_type=LithoViolationType.BREAK_RISK_LINE_END,
            severity=LithoSeverity(rule_config.severity.value),
            message=message,
            region=region,
            measurement_nm=meas,
            threshold_nm=rule_config.threshold_nm,
            pixel_size=pixel_size,
            opc_feasibility=OPCFeasibility.FIXABLE,
            extra_info={
                "line_end_width_nm": meas,
                "search_radius_nm": search_radius_nm,
            },
        ))

    return violations
