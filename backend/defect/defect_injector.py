# -*- coding: utf-8 -*-
"""
缺陷注入器模块

在掩模指定位置注入各类缺陷：
1. 点缺陷：圆形/方形不透明或透明缺陷
2. 线缺陷：矩形线条缺陷，支持任意角度
3. 污染斑：模拟表面颗粒污染，含衰减和粗糙边缘
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from scipy.ndimage import distance_transform_edt
import logging

from defect.schemas import (
    DefectType,
    DefectPolarity,
    PointDefect,
    LineDefect,
    ContaminationDefect,
    DefectLocation,
    DefectInjectionConfig,
)

logger = logging.getLogger(__name__)


class DefectInjector:
    """
    掩模缺陷注入器

    在掩模图案的指定位置注入各种类型的缺陷，返回注入缺陷后的掩模。

    使用方式::

        injector = DefectInjector(pixel_size=1.0)
        mask_with_defect = injector.inject(mask_nominal, point_defect)
    """

    def __init__(self, config: Optional[DefectInjectionConfig] = None,
                 pixel_size: float = 1.0,
                 random_seed: Optional[int] = None):
        """
        初始化缺陷注入器

        Args:
            config: 缺陷注入配置，None 则使用默认
            pixel_size: 像素尺寸 (nm/pixel)，仅当 config 为 None 时使用
            random_seed: 随机种子，仅当 config 为 None 时使用
        """
        if config is not None:
            self.config = config
        else:
            self.config = DefectInjectionConfig(
                pixel_size=pixel_size,
                random_seed=random_seed,
            )
        self._rng = np.random.default_rng(self.config.random_seed)

    @property
    def pixel_size(self) -> float:
        return self.config.pixel_size

    def inject(
        self,
        mask: np.ndarray,
        defect: Union[PointDefect, LineDefect, ContaminationDefect],
    ) -> np.ndarray:
        """
        在掩模中注入单个缺陷

        Args:
            mask: 标称掩模 (2D数组，值范围 [0, 1])
            defect: 缺陷参数对象

        Returns:
            注入缺陷后的掩模
        """
        if isinstance(defect, PointDefect):
            return self.inject_point_defect(mask, defect)
        elif isinstance(defect, LineDefect):
            return self.inject_line_defect(mask, defect)
        elif isinstance(defect, ContaminationDefect):
            return self.inject_contamination(mask, defect)
        else:
            raise TypeError(f"未知的缺陷类型: {type(defect)}")

    def inject_multiple(
        self,
        mask: np.ndarray,
        defects: List[Union[PointDefect, LineDefect, ContaminationDefect]],
    ) -> np.ndarray:
        """
        在掩模中注入多个缺陷

        Args:
            mask: 标称掩模
            defects: 缺陷参数对象列表

        Returns:
            注入所有缺陷后的掩模
        """
        result = mask.astype(np.float64).copy()
        for defect in defects:
            result = self.inject(result, defect)
        return result

    def inject_point_defect(
        self,
        mask: np.ndarray,
        defect: PointDefect,
    ) -> np.ndarray:
        """
        注入点缺陷

        Args:
            mask: 标称掩模
            defect: 点缺陷参数

        Returns:
            注入点缺陷后的掩模
        """
        result = mask.astype(np.float64).copy()
        ny, nx = result.shape
        ps = self.pixel_size

        if defect.location is None:
            cy, cx = ny / 2.0, nx / 2.0
        else:
            cy, cx = defect.location.y, defect.location.x

        radius_pix = defect.size_nm / (2.0 * ps)

        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        dy = yy - cy
        dx = xx - cx

        if defect.shape == 'circle':
            dist = np.sqrt(dy ** 2 + dx ** 2)
            defect_mask = dist <= radius_pix
        else:
            defect_mask = (np.abs(dy) <= radius_pix) & (np.abs(dx) <= radius_pix)

        self._apply_defect_mask(result, defect_mask, defect.polarity)
        return result

    def inject_line_defect(
        self,
        mask: np.ndarray,
        defect: LineDefect,
    ) -> np.ndarray:
        """
        注入线缺陷

        Args:
            mask: 标称掩模
            defect: 线缺陷参数

        Returns:
            注入线缺陷后的掩模
        """
        result = mask.astype(np.float64).copy()
        ny, nx = result.shape
        ps = self.pixel_size

        if defect.location is None:
            cy, cx = ny / 2.0, nx / 2.0
        else:
            cy, cx = defect.location.y, defect.location.x

        half_len_pix = defect.length_nm / (2.0 * ps)
        half_wid_pix = defect.width_nm / (2.0 * ps)
        angle_rad = np.deg2rad(defect.angle_deg)

        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        dy = yy - cy
        dx = xx - cx

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        along = dx * cos_a + dy * sin_a
        across = -dx * sin_a + dy * cos_a

        defect_mask = (np.abs(along) <= half_len_pix) & (np.abs(across) <= half_wid_pix)

        self._apply_defect_mask(result, defect_mask, defect.polarity)
        return result

    def inject_contamination(
        self,
        mask: np.ndarray,
        defect: ContaminationDefect,
    ) -> np.ndarray:
        """
        注入污染斑缺陷

        污染斑具有衰减透射特性和不规则粗糙边缘。

        Args:
            mask: 标称掩模
            defect: 污染斑参数

        Returns:
            注入污染斑后的掩模
        """
        result = mask.astype(np.float64).copy()
        ny, nx = result.shape
        ps = self.pixel_size

        if defect.location is None:
            cy, cx = ny / 2.0, nx / 2.0
        else:
            cy, cx = defect.location.y, defect.location.x

        radius_pix = defect.size_nm / (2.0 * ps)

        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        dy = yy - cy
        dx = xx - cx
        dist = np.sqrt(dy ** 2 + dx ** 2)

        if defect.roughness <= 1e-6:
            defect_mask = dist <= radius_pix
            attenuation_profile = defect_mask.astype(np.float64)
        else:
            base_profile = np.clip(1.0 - dist / (radius_pix * (1.0 + defect.roughness)), 0.0, 1.0)

            noise_grid = 8
            noise_ny = max(3, int(ny / noise_grid))
            noise_nx = max(3, int(nx / noise_grid))
            noise_small = self._rng.normal(0.0, 1.0, (noise_ny, noise_nx))

            from scipy.ndimage import zoom
            scale_y = ny / noise_ny
            scale_x = nx / noise_nx
            noise = zoom(noise_small, (scale_y, scale_x), order=3)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-12)
            noise = 2.0 * noise - 1.0

            rough_profile = base_profile + defect.roughness * 0.5 * noise * base_profile
            attenuation_profile = np.clip(rough_profile, 0.0, 1.0)
            defect_mask = attenuation_profile > 0.05

        effective_attenuation = defect.attenuation
        if defect.polarity == DefectPolarity.OPAQUE:
            defect_values = 1.0 - effective_attenuation * attenuation_profile
            result[defect_mask] = np.minimum(result[defect_mask], defect_values[defect_mask])
        else:
            defect_values = effective_attenuation * attenuation_profile
            result[defect_mask] = np.maximum(result[defect_mask], defect_values[defect_mask])

        result = np.clip(result, 0.0, 1.0)
        return result

    @staticmethod
    def _apply_defect_mask(
        mask: np.ndarray,
        defect_mask: np.ndarray,
        polarity: DefectPolarity,
    ) -> None:
        """
        根据极性将缺陷掩模应用到掩模上（原地修改）

        Args:
            mask: 待修改的掩模
            defect_mask: 缺陷区域的布尔掩码
            polarity: 缺陷极性
        """
        if polarity == DefectPolarity.OPAQUE:
            mask[defect_mask] = 0.0
        else:
            mask[defect_mask] = 1.0

    def compute_distance_to_edge(
        self,
        mask: np.ndarray,
        location: DefectLocation,
    ) -> float:
        """
        计算缺陷位置到最近掩模图案边缘的距离

        Args:
            mask: 标称掩模 (二值化)
            location: 缺陷位置

        Returns:
            到最近边缘的距离 (nm)
        """
        mask_bin = (mask >= 0.5).astype(np.float64)
        edges = self._extract_edges(mask_bin)

        if np.sum(edges) == 0:
            return float('inf')

        dist_map = distance_transform_edt(1.0 - edges)
        yi = int(round(location.y))
        xi = int(round(location.x))
        ny, nx = mask.shape
        yi = np.clip(yi, 0, ny - 1)
        xi = np.clip(xi, 0, nx - 1)

        return float(dist_map[yi, xi] * self.pixel_size)

    @staticmethod
    def _extract_edges(binary_mask: np.ndarray) -> np.ndarray:
        """从二值掩模中提取边缘"""
        from scipy.ndimage import binary_erosion
        struct = np.ones((3, 3), dtype=bool)
        eroded = binary_erosion(binary_mask > 0.5, structure=struct)
        edges = (binary_mask > 0.5) & (~eroded)
        return edges.astype(np.float64)

    def generate_edge_proximity_locations(
        self,
        mask: np.ndarray,
        n_locations: int = 5,
        min_distance_nm: float = 0.0,
        max_distance_nm: Optional[float] = None,
    ) -> List[DefectLocation]:
        """
        生成若干靠近图案边缘的缺陷位置，用于系统测试缺陷敏感度

        Args:
            mask: 标称掩模
            n_locations: 生成的位置数量
            min_distance_nm: 到边缘的最小距离 (nm)
            max_distance_nm: 到边缘的最大距离 (nm)，None 则使用图像范围

        Returns:
            缺陷位置列表
        """
        mask_bin = (mask >= 0.5).astype(np.float64)
        ny, nx = mask.shape
        ps = self.pixel_size

        edges = self._extract_edges(mask_bin)
        edge_pixels = np.argwhere(edges > 0.5)

        if len(edge_pixels) == 0:
            cy, cx = ny / 2.0, nx / 2.0
            return [DefectLocation(y=cy, x=cx, distance_to_edge=float('inf'))]

        min_dist_pix = min_distance_nm / ps if min_distance_nm > 0 else 0
        if max_distance_nm is not None:
            max_dist_pix = max_distance_nm / ps
        else:
            max_dist_pix = max(ny, nx)

        chosen = self._rng.choice(len(edge_pixels), size=min(n_locations, len(edge_pixels)), replace=False)

        locations = []
        for idx in chosen:
            ey, ex = edge_pixels[idx]

            angle = self._rng.uniform(0, 2 * np.pi)
            offset = self._rng.uniform(min_dist_pix, min(max_dist_pix, 5.0))
            dy = offset * np.sin(angle)
            dx = offset * np.cos(angle)

            y = float(ey + dy)
            x = float(ex + dx)
            y = np.clip(y, 0, ny - 1)
            x = np.clip(x, 0, nx - 1)

            loc = DefectLocation(y=y, x=x)
            loc.distance_to_edge = self.compute_distance_to_edge(mask, loc)
            locations.append(loc)

        return locations
