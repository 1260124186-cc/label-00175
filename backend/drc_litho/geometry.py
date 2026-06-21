# -*- coding: utf-8 -*-
"""
版图设计规则邻近分析模块 - 几何工具集

提供光刻导向 DRC 分析所需的底层几何计算：
1. 距离变换与邻近距离计算
2. 骨架提取与局部线宽分析
3. 连通域分析与拓扑关系
4. 颈部检测（变窄区域）
5. 拐角密度估计
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
    find_objects,
    center_of_mass,
    generate_binary_structure,
)
from skimage.morphology import skeletonize, medial_axis
from skimage.measure import regionprops, label as sk_label

logger = logging.getLogger(__name__)


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"掩模必须是 2D 数组，当前形状: {mask.shape}")
    if mask.dtype == np.bool_:
        return mask
    if np.max(mask) <= 1.0:
        return mask > 0.5
    return mask > 127


def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    return distance_transform_edt(mask.astype(float))


def compute_spacing_map(mask: np.ndarray) -> np.ndarray:
    background = ~mask
    return distance_transform_edt(background.astype(float))


def compute_local_line_width(mask: np.ndarray) -> np.ndarray:
    dist = distance_transform_edt(mask.astype(float))
    return dist * 2.0


def detect_neck_regions(
    mask: np.ndarray,
    neck_threshold_px: float,
) -> np.ndarray:
    mask_bool = mask.astype(bool) if mask.dtype != np.bool_ else mask
    dist = distance_transform_edt(mask_bool.astype(float))
    local_width = dist * 2.0
    skel = skeletonize(mask_bool)
    neck_mask = np.zeros_like(mask_bool, dtype=bool)
    width_on_skel = local_width.copy()
    width_on_skel[~skel] = np.inf
    neck_mask = (width_on_skel < neck_threshold_px) & skel
    return neck_mask


def find_narrow_gaps(
    mask: np.ndarray,
    gap_threshold_px: float,
) -> np.ndarray:
    mask_bool = mask.astype(bool) if mask.dtype != np.bool_ else mask
    background = ~mask_bool
    spacing = distance_transform_edt(background.astype(float))
    bg_skel = skeletonize(background)
    gap_mask = np.zeros_like(mask_bool, dtype=bool)
    spacing_on_skel = spacing.copy()
    spacing_on_skel[~bg_skel] = np.inf
    gap_mask = (spacing_on_skel < gap_threshold_px) & bg_skel
    return gap_mask


def label_connected_components(
    mask: np.ndarray,
    connectivity: int = 2,
) -> Tuple[np.ndarray, int]:
    if connectivity == 1:
        struct = generate_binary_structure(2, 1)
    else:
        struct = generate_binary_structure(2, 2)
    return label(mask, structure=struct)


def compute_component_properties(
    mask: np.ndarray,
    pixel_size: float = 1.0,
) -> List[Dict]:
    labeled, num = label_connected_components(mask)
    if num == 0:
        return []
    props = regionprops(labeled)
    results = []
    for i, prop in enumerate(props):
        area_px = prop.area
        results.append({
            "label_id": i + 1,
            "centroid": prop.centroid,
            "bbox": prop.bbox,
            "area_pixels": area_px,
            "area_nm2": area_px * (pixel_size ** 2),
            "major_axis_nm": (prop.axis_major_length if hasattr(prop, 'axis_major_length') else prop.major_axis_length) * pixel_size,
            "minor_axis_nm": (prop.axis_minor_length if hasattr(prop, 'axis_minor_length') else prop.minor_axis_length) * pixel_size,
            "eccentricity": prop.eccentricity,
            "solidity": prop.solidity,
        })
    return results


def detect_sharp_turns(
    mask: np.ndarray,
    min_angle_deg: float = 60.0,
    turn_radius: int = 6,
) -> np.ndarray:
    from scipy.ndimage import convolve

    mask_bool = mask.astype(bool) if mask.dtype != np.bool_ else mask
    struct = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask_bool, structure=struct)
    edge = mask_bool & ~eroded

    skeleton = skeletonize(mask_bool)

    k_size = 5
    response = np.zeros_like(mask_bool, dtype=float)

    for angle_deg in [0, 45, 90, 135]:
        rad = np.radians(angle_deg)
        for corner_type in ["convex", "concave"]:
            kernel = _build_corner_kernel(k_size, rad, corner_type)
            if kernel is None:
                continue
            conv = convolve(edge.astype(float), kernel)
            response = np.maximum(response, conv)

    threshold = k_size * k_size * 0.4
    corners = response >= threshold
    corners = corners & skeleton

    if not np.any(corners):
        return np.zeros_like(mask_bool, dtype=bool)

    sharp_mask = np.zeros_like(mask_bool, dtype=bool)
    corner_coords = np.argwhere(corners)
    for cy, cx in corner_coords:
        angle = _estimate_local_angle(mask_bool, cy, cx, turn_radius)
        if angle < min_angle_deg:
            sharp_mask[cy, cx] = True

    return sharp_mask


def estimate_corner_density(
    mask: np.ndarray,
    block_size: int = 32,
) -> np.ndarray:
    from scipy.ndimage import convolve

    mask_bool = mask.astype(bool) if mask.dtype != np.bool_ else mask
    struct = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask_bool, structure=struct)
    edge = mask_bool & ~eroded

    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    corner_response = np.abs(convolve(mask.astype(np.float32), laplacian))
    corner_map = (corner_response >= 0.5).astype(np.float32)

    h, w = mask.shape
    density_h = (h + block_size - 1) // block_size
    density_w = (w + block_size - 1) // block_size
    density = np.zeros((density_h, density_w), dtype=np.float32)

    for by in range(density_h):
        for bx in range(density_w):
            y0 = by * block_size
            y1 = min(y0 + block_size, h)
            x0 = bx * block_size
            x1 = min(x0 + block_size, w)
            block = corner_map[y0:y1, x0:x1]
            edge_block = edge[y0:y1, x0:x1]
            n_corners = np.sum(block > 0)
            n_edge = np.sum(edge_block)
            if n_edge > 0:
                density[by, bx] = n_corners / n_edge
            else:
                density[by, bx] = 0.0

    return density


def find_dangling_lines(
    mask: np.ndarray,
    min_length_px: float = 5.0,
    max_width_px: float = 3.0,
) -> List[Dict]:
    mask_bool = mask.astype(bool) if mask.dtype != np.bool_ else mask
    dist = distance_transform_edt(mask_bool.astype(float))
    local_width = dist * 2.0

    skeleton = skeletonize(mask_bool)
    if not np.any(skeleton):
        return []

    struct_8 = generate_binary_structure(2, 2)
    endpoint_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)

    padded = np.pad(skeleton.astype(int), 1, mode='constant', constant_values=0)
    neighbor_count = np.zeros_like(skeleton, dtype=int)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            neighbor_count += padded[1 + dy:1 + dy + skeleton.shape[0],
                                      1 + dx:1 + dx + skeleton.shape[1]]

    endpoints = (neighbor_count == 1) & skeleton
    branch_points = (neighbor_count >= 3) & skeleton

    labeled_skel, num_lines = label(skeleton.astype(int), structure=struct_8)

    dangling_list = []
    objects = find_objects(labeled_skel)
    for i, obj_slice in enumerate(objects):
        if obj_slice is None:
            continue
        local_skel = labeled_skel[obj_slice] == (i + 1)
        local_endpoints = endpoints[obj_slice] & local_skel
        local_branchpoints = branch_points[obj_slice] & local_skel

        n_ep = int(np.sum(local_endpoints))
        if n_ep < 1:
            continue

        local_width_crop = local_width[obj_slice]
        skel_widths = local_width_crop[local_skel]
        if len(skel_widths) == 0:
            continue

        mean_width = float(np.mean(skel_widths))
        max_width = float(np.max(skel_widths))
        length = float(np.sum(local_skel))

        if max_width > max_width_px:
            continue
        if length < min_length_px:
            continue

        n_ep_in_full = n_ep
        n_bp_in_full = int(np.sum(local_branchpoints))

        is_dangling = n_ep_in_full >= 1 and n_bp_in_full <= 1

        if is_dangling:
            cy, cx = center_of_mass(local_skel)
            centroid = (
                float(cy + obj_slice[0].start),
                float(cx + obj_slice[1].start),
            )
            dangling_list.append({
                "centroid": centroid,
                "bbox": (
                    obj_slice[0].start,
                    obj_slice[1].start,
                    obj_slice[0].stop,
                    obj_slice[1].stop,
                ),
                "length_px": length,
                "mean_width_px": mean_width,
                "max_width_px": max_width,
                "num_endpoints": n_ep_in_full,
            })

    return dangling_list


def find_orphan_pixels(
    mask: np.ndarray,
    max_area_px: int = 4,
) -> List[Dict]:
    labeled, num = label_connected_components(mask)
    if num == 0:
        return []

    objects = find_objects(labeled)
    orphans = []

    for i, obj_slice in enumerate(objects):
        if obj_slice is None:
            continue
        local = labeled[obj_slice] == (i + 1)
        area_px = int(np.sum(local))
        if area_px <= max_area_px:
            cy, cx = center_of_mass(local)
            centroid = (
                float(cy + obj_slice[0].start),
                float(cx + obj_slice[1].start),
            )
            orphans.append({
                "centroid": centroid,
                "bbox": (
                    obj_slice[0].start,
                    obj_slice[1].start,
                    obj_slice[0].stop,
                    obj_slice[1].stop,
                ),
                "area_pixels": area_px,
            })

    return orphans


def find_line_ends(
    mask: np.ndarray,
    search_radius_px: int = 3,
) -> np.ndarray:
    mask_bool = mask.astype(bool) if mask.dtype != np.bool_ else mask
    skeleton = skeletonize(mask_bool)
    if not np.any(skeleton):
        return np.zeros_like(mask_bool, dtype=bool)

    padded = np.pad(skeleton.astype(int), 1, mode='constant', constant_values=0)
    neighbor_count = np.zeros_like(skeleton, dtype=int)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            neighbor_count += padded[1 + dy:1 + dy + skeleton.shape[0],
                                      1 + dx:1 + dx + skeleton.shape[1]]

    endpoints = (neighbor_count == 1) & skeleton

    line_end_mask = np.zeros_like(mask_bool, dtype=bool)
    if not np.any(endpoints):
        return line_end_mask

    endpoint_coords = np.argwhere(endpoints)
    for ey, ex in endpoint_coords:
        y0 = max(0, ey - search_radius_px)
        y1 = min(mask_bool.shape[0], ey + search_radius_px + 1)
        x0 = max(0, ex - search_radius_px)
        x1 = min(mask_bool.shape[1], ex + search_radius_px + 1)
        line_end_mask[y0:y1, x0:x1] |= mask_bool[y0:y1, x0:x1]

    return line_end_mask


def mask_to_regions(
    violation_mask: np.ndarray,
    pixel_size: float = 1.0,
    measurement_array: Optional[np.ndarray] = None,
    threshold_nm: float = 0.0,
) -> List[Dict]:
    from .schemas import ViolationRegion

    if not np.any(violation_mask):
        return []

    labeled, num = label(violation_mask)
    if num == 0:
        return []

    objects = find_objects(labeled)
    regions = []

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

        measurement_nm = threshold_nm
        if measurement_array is not None:
            local_meas = measurement_array[obj_slice]
            valid = local_meas[local & np.isfinite(local_meas)]
            if len(valid) > 0:
                measurement_nm = float(np.mean(valid))

        region = ViolationRegion(
            bbox=bbox_full,
            centroid=centroid,
            area_pixels=area_px,
            mask_slice=local,
        )
        regions.append({"region": region, "measurement_nm": measurement_nm})

    return regions


def _build_corner_kernel(size: int, angle_rad: float, corner_type: str):
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


def _estimate_local_angle(mask: np.ndarray, cy: int, cx: int,
                          radius: int = 6) -> float:
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
