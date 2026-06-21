# -*- coding: utf-8 -*-
"""
版图设计规则邻近分析模块 - 桥连风险检查器

检测版图中可能导致桥连 (bridge/short) 的拓扑模式:
1. 相邻特征间距过窄 (narrow gap)
2. 局部线宽收窄/颈部 (necking)
3. 拐角密度过高 (dense corner)
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
    find_narrow_gaps,
    detect_neck_regions,
    estimate_corner_density,
    mask_to_regions,
)

logger = logging.getLogger(__name__)


def check_bridge_narrow_gap(
    mask: np.ndarray,
    rule_config: LithoRuleConfig,
    pixel_size: float,
) -> List[LithoViolation]:
    threshold_nm = rule_config.extra_params.get("min_gap_nm", rule_config.threshold_nm)
    threshold_px = threshold_nm / pixel_size

    mask_bin = preprocess_mask(mask)
    if not np.any(mask_bin):
        return []

    gap_mask = find_narrow_gaps(mask_bin, threshold_px)
    if not np.any(gap_mask):
        return []

    spacing_map = distance_transform_edt((~mask_bin).astype(float))
    spacing_nm = spacing_map * pixel_size * 2.0

    regions = mask_to_regions(gap_mask, pixel_size, spacing_nm, threshold_nm)
    violations = []

    for item in regions:
        region = item["region"]
        meas = item["measurement_nm"]
        message = (
            f"桥连风险-间距过窄: 间距 {meas:.1f}nm < 阈值 {threshold_nm:.1f}nm，"
            f"光刻后相邻特征可能粘连"
        )
        violations.append(LithoViolation(
            category=LithoViolationCategory.BRIDGE,
            violation_type=LithoViolationType.BRIDGE_RISK_NARROW_GAP,
            severity=LithoSeverity(rule_config.severity.value),
            message=message,
            region=region,
            measurement_nm=meas,
            threshold_nm=threshold_nm,
            pixel_size=pixel_size,
            opc_feasibility=OPCFeasibility.FIXABLE,
            extra_info={"gap_nm": meas, "min_gap_nm": threshold_nm},
        ))

    return violations


def check_bridge_necking(
    mask: np.ndarray,
    rule_config: LithoRuleConfig,
    pixel_size: float,
) -> List[LithoViolation]:
    threshold_nm = rule_config.threshold_nm
    threshold_px = threshold_nm / pixel_size
    neck_ratio = rule_config.extra_params.get("neck_ratio", 0.5)

    mask_bin = preprocess_mask(mask)
    if not np.any(mask_bin):
        return []

    neck_mask = detect_neck_regions(mask_bin, threshold_px)
    if not np.any(neck_mask):
        return []

    local_width_map = distance_transform_edt(mask_bin.astype(float)) * 2.0
    local_width_nm = local_width_map * pixel_size

    from skimage.morphology import skeletonize
    skel = skeletonize(mask_bin)
    width_on_skel = local_width_nm.copy()
    width_on_skel[~skel] = np.inf

    regions = mask_to_regions(neck_mask, pixel_size, width_on_skel, threshold_nm)
    violations = []

    dist = distance_transform_edt(mask_bin.astype(float))
    max_width_px = float(np.max(dist)) * 2.0
    max_width_nm = max_width_px * pixel_size

    for item in regions:
        region = item["region"]
        meas = item["measurement_nm"]
        ratio = meas / max_width_nm if max_width_nm > 0 else 1.0

        opc_feas = OPCFeasibility.FIXABLE if ratio >= neck_ratio else OPCFeasibility.PARTIAL
        message = (
            f"桥连风险-颈部收窄: 局部线宽 {meas:.1f}nm < 阈值 {threshold_nm:.1f}nm "
            f"(占全局线宽比 {ratio:.1%})"
        )
        violations.append(LithoViolation(
            category=LithoViolationCategory.BRIDGE,
            violation_type=LithoViolationType.BRIDGE_RISK_NECKING,
            severity=LithoSeverity(rule_config.severity.value),
            message=message,
            region=region,
            measurement_nm=meas,
            threshold_nm=threshold_nm,
            pixel_size=pixel_size,
            opc_feasibility=opc_feas,
            extra_info={
                "neck_width_nm": meas,
                "max_width_nm": max_width_nm,
                "neck_ratio": ratio,
            },
        ))

    return violations


def check_bridge_dense_corner(
    mask: np.ndarray,
    rule_config: LithoRuleConfig,
    pixel_size: float,
) -> List[LithoViolation]:
    density_threshold = rule_config.extra_params.get(
        "density_threshold", rule_config.threshold_nm
    )
    block_size = rule_config.extra_params.get("block_size", 32)

    mask_bin = preprocess_mask(mask)
    if not np.any(mask_bin):
        return []

    density_map = estimate_corner_density(mask_bin, block_size)

    high_density = density_map >= density_threshold
    if not np.any(high_density):
        return []

    violations = []
    coords = np.argwhere(high_density)

    for by, bx in coords:
        y0 = by * block_size
        y1 = min(y0 + block_size, mask_bin.shape[0])
        x0 = bx * block_size
        x1 = min(x0 + block_size, mask_bin.shape[1])

        if not np.any(mask_bin[y0:y1, x0:x1]):
            continue

        density_val = float(density_map[by, bx])
        centroid = (
            float((y0 + y1) / 2.0),
            float((x0 + x1) / 2.0),
        )
        region = ViolationRegion(
            bbox=(y0, x0, y1, x1),
            centroid=centroid,
            area_pixels=int(np.sum(mask_bin[y0:y1, x0:x1])),
        )

        message = (
            f"桥连风险-拐角密度过高: 密度 {density_val:.2f} >= 阈值 "
            f"{density_threshold:.2f}，邻近拐角衍射叠加可能桥连"
        )

        violations.append(LithoViolation(
            category=LithoViolationCategory.BRIDGE,
            violation_type=LithoViolationType.BRIDGE_RISK_DENSE_CORNER,
            severity=LithoSeverity(rule_config.severity.value),
            message=message,
            region=region,
            measurement_nm=density_val,
            threshold_nm=density_threshold,
            pixel_size=pixel_size,
            opc_feasibility=OPCFeasibility.PARTIAL,
            extra_info={
                "corner_density": density_val,
                "block_size": block_size,
                "block_position": (by, bx),
            },
        ))

    return violations
