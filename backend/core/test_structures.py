# -*- coding: utf-8 -*-
"""
半导体标准测试结构生成模块

该模块实现了光刻工艺中常用的标准测试结构生成器，用于光刻分辨率、
工艺窗口、OPC校正等测试。所有结构支持参数化配置，包括：
- CD (Critical Dimension, 关键尺寸)
- pitch (间距)
- corner rounding (拐角圆滑度)

典型测试结构包括：
1. Line/Space (线/空间) - 用于分辨率和线宽均匀性测试
2. Contact Hole (接触孔) - 用于接触孔CD和圆形度测试
3. L-shaped Corner (L形拐角) - 用于拐角圆滑度测试
4. T-junction (T形结) - 用于线端缩短和连接测试
5. SRAM Bitcell (SRAM位单元) - 用于高密度存储单元测试
"""

import numpy as np
from numba import jit
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class TestStructureType(Enum):
    """测试结构类型枚举"""
    LINE_SPACE = "line_space"
    CONTACT_HOLE = "contact_hole"
    L_SHAPED_CORNER = "l_shaped_corner"
    T_JUNCTION = "t_junction"
    SRAM_BITCELL = "sram_bitcell"


class LineOrientation(Enum):
    """线/空间结构的线方向"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class HolePattern(Enum):
    """接触孔排列方式"""
    SQUARE_GRID = "square_grid"
    HEXAGONAL = "hexagonal"


@dataclass
class TestStructureParams:
    """
    测试结构基础参数

    所有测试结构的共享参数配置。

    Attributes:
        grid_size: 生成掩模的网格尺寸 (ny, nx)，单位为像素
        pixel_size: 像素尺寸 (nm/pixel)
        cd: 关键尺寸 (nm)，目标线宽或孔直径
        pitch: 间距 (nm)，相邻结构中心之间的距离
        corner_rounding: 拐角圆滑度 (nm)，拐角处的圆角半径
        structure_type: 测试结构类型
        name: 结构名称标识
    """
    grid_size: Tuple[int, int] = (256, 256)
    pixel_size: float = 1.0
    cd: float = 45.0
    pitch: float = 90.0
    corner_rounding: float = 0.0
    structure_type: TestStructureType = TestStructureType.LINE_SPACE
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.structure_type.value}_cd{self.cd:.0f}_pitch{self.pitch:.0f}"
        self._validate()

    def _validate(self):
        """验证参数合法性"""
        if self.grid_size[0] <= 0 or self.grid_size[1] <= 0:
            raise ValueError(f"grid_size 必须为正整数，当前: {self.grid_size}")
        if self.pixel_size <= 0:
            raise ValueError(f"pixel_size 必须为正数，当前: {self.pixel_size}")
        if self.cd <= 0:
            raise ValueError(f"cd 必须为正数，当前: {self.cd}")
        if self.pitch <= self.cd:
            raise ValueError(f"pitch ({self.pitch}) 必须大于 cd ({self.cd})")
        if self.corner_rounding < 0:
            raise ValueError(f"corner_rounding 不能为负数，当前: {self.corner_rounding}")

    @property
    def cd_pixels(self) -> float:
        """以像素为单位的CD"""
        return self.cd / self.pixel_size

    @property
    def pitch_pixels(self) -> float:
        """以像素为单位的pitch"""
        return self.pitch / self.pixel_size

    @property
    def corner_rounding_pixels(self) -> float:
        """以像素为单位的corner_rounding"""
        return self.corner_rounding / self.pixel_size

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'grid_size': list(self.grid_size),
            'pixel_size': self.pixel_size,
            'cd': self.cd,
            'pitch': self.pitch,
            'corner_rounding': self.corner_rounding,
            'structure_type': self.structure_type.value,
            'name': self.name
        }


@dataclass
class LineSpaceParams(TestStructureParams):
    """
    Line/Space 测试结构参数

    Attributes:
        orientation: 线的方向（水平或垂直）
        duty_cycle: 占空比，线宽与间距之比 cd/(pitch-cd)
        num_lines: 线的数量，None 表示填满整个网格
    """
    structure_type: TestStructureType = TestStructureType.LINE_SPACE
    orientation: LineOrientation = LineOrientation.HORIZONTAL
    duty_cycle: float = 1.0
    num_lines: Optional[int] = None

    def _validate(self):
        super()._validate()
        if self.duty_cycle <= 0:
            raise ValueError(f"duty_cycle 必须为正数，当前: {self.duty_cycle}")
        if self.num_lines is not None and self.num_lines <= 0:
            raise ValueError(f"num_lines 必须为正整数，当前: {self.num_lines}")


@dataclass
class ContactHoleParams(TestStructureParams):
    """
    Contact Hole 测试结构参数

    Attributes:
        pattern: 接触孔排列方式
        hole_shape: 孔的形状 ('circle' 或 'square')
        aspect_ratio: 孔的纵横比 (长轴/短轴)
        rotation: 孔的旋转角度 (度)
    """
    structure_type: TestStructureType = TestStructureType.CONTACT_HOLE
    pattern: HolePattern = HolePattern.SQUARE_GRID
    hole_shape: str = "circle"
    aspect_ratio: float = 1.0
    rotation: float = 0.0

    def _validate(self):
        super()._validate()
        if self.hole_shape not in ['circle', 'square']:
            raise ValueError(f"hole_shape 必须为 'circle' 或 'square'，当前: {self.hole_shape}")
        if self.aspect_ratio <= 0:
            raise ValueError(f"aspect_ratio 必须为正数，当前: {self.aspect_ratio}")


@dataclass
class LShapedCornerParams(TestStructureParams):
    """
    L-shaped Corner 测试结构参数

    Attributes:
        arm_length: L形臂的长度 (nm)
        corner_type: 拐角类型 ('inner' 或 'outer')
    """
    structure_type: TestStructureType = TestStructureType.L_SHAPED_CORNER
    arm_length: float = 200.0
    corner_type: str = "inner"

    def _validate(self):
        super()._validate()
        if self.arm_length <= 0:
            raise ValueError(f"arm_length 必须为正数，当前: {self.arm_length}")
        if self.corner_type not in ['inner', 'outer']:
            raise ValueError(f"corner_type 必须为 'inner' 或 'outer'，当前: {self.corner_type}")

    @property
    def arm_length_pixels(self) -> float:
        """以像素为单位的臂长"""
        return self.arm_length / self.pixel_size


@dataclass
class TJunctionParams(TestStructureParams):
    """
    T-junction 测试结构参数

    Attributes:
        stem_length: 主干长度 (nm)
        branch_length: 分支长度 (nm)
    """
    structure_type: TestStructureType = TestStructureType.T_JUNCTION
    stem_length: float = 200.0
    branch_length: float = 100.0

    def _validate(self):
        super()._validate()
        if self.stem_length <= 0:
            raise ValueError(f"stem_length 必须为正数，当前: {self.stem_length}")
        if self.branch_length <= 0:
            raise ValueError(f"branch_length 必须为正数，当前: {self.branch_length}")

    @property
    def stem_length_pixels(self) -> float:
        return self.stem_length / self.pixel_size

    @property
    def branch_length_pixels(self) -> float:
        return self.branch_length / self.pixel_size


@dataclass
class SRAMBitcellParams(TestStructureParams):
    """
    SRAM Bitcell 测试结构参数

    Attributes:
        bitcell_type: SRAM位单元类型 ('6T' 或 'thin-film')
        metal_layer: 金属层编号
    """
    structure_type: TestStructureType = TestStructureType.SRAM_BITCELL
    bitcell_type: str = "6T"
    metal_layer: int = 1

    def _validate(self):
        super()._validate()
        if self.bitcell_type not in ['6T', 'thin-film']:
            raise ValueError(f"bitcell_type 必须为 '6T' 或 'thin-film'，当前: {self.bitcell_type}")
        if self.metal_layer < 1:
            raise ValueError(f"metal_layer 必须 >= 1，当前: {self.metal_layer}")


class TestStructureGenerator:
    """
    测试结构生成器基类

    所有具体测试结构生成器的基类，提供通用的辅助方法。
    """

    @staticmethod
    def _get_coordinates(grid_size: Tuple[int, int],
                          pixel_size: float = 1.0,
                          center: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成以中心为原点的坐标网格

        Args:
            grid_size: 网格尺寸 (ny, nx)
            pixel_size: 像素尺寸
            center: 中心坐标 (y, x)，None 表示网格中心

        Returns:
            (yy, xx): 坐标网格
        """
        ny, nx = grid_size
        if center is None:
            center = (ny / 2.0, nx / 2.0)

        y = (np.arange(ny) - center[0]) * pixel_size
        x = (np.arange(nx) - center[1]) * pixel_size
        xx, yy = np.meshgrid(x, y)
        return yy, xx

    @staticmethod
    def _apply_corner_rounding(mask: np.ndarray,
                            corner_radius: float,
                            pixel_size: float) -> np.ndarray:
        """
        对掩模应用拐角圆滑化

        Args:
            mask: 输入二元掩模 (0或1)
            corner_radius: 拐角圆滑半径 (nm)
            pixel_size: 像素尺寸 (nm/pixel)

        Returns:
            圆滑后的掩模
        """
        if corner_radius <= 0:
            return mask

        radius_pixels = corner_radius / pixel_size
        if radius_pixels < 1.5:
            warnings.warn(f"corner_rounding ({corner_radius}nm) 小于 1.5 像素，效果不明显")
            return mask

        kernel_size = int(np.ceil(radius_pixels * 2 + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1

        y = np.arange(kernel_size) - kernel_size // 2
        xx, yy = np.meshgrid(y, y)
        dist = np.sqrt(yy ** 2 + xx ** 2)
        kernel = (dist <= radius_pixels).astype(np.float64)

        from scipy.ndimage import binary_erosion

        inner_mask = binary_erosion(mask > 0.5, structure=kernel > 0.5, iterations=1)

        border = (mask > 0.5) & (~inner_mask)

        yy_border, xx_border = np.where(border)
        result = mask.copy().astype(np.float64)

        half_k = kernel_size // 2
        for yi, xi in zip(yy_border, xx_border):
            y_min = max(0, yi - half_k)
            y_max = min(mask.shape[0], yi + half_k + 1)
            x_min = max(0, xi - half_k)
            x_max = min(mask.shape[1], xi + half_k + 1)

            ky_min = y_min - yi + half_k
            ky_max = y_max - yi + half_k
            kx_min = x_min - xi + half_k
            kx_max = x_max - xi + half_k

            region = mask[y_min:y_max, x_min:x_max]

            has_outer = np.any(region < 0.5)
            has_inner = np.any(region > 0.5)

            if has_outer and has_inner:
                result[yi, xi] = 0.0

        return result

    @staticmethod
    def _smooth_edges(mask: np.ndarray,
                    sigma: float = 0.5) -> np.ndarray:
        """
        对掩模边缘进行平滑处理

        Args:
            mask: 输入掩模
            sigma: 高斯平滑的sigma

        Returns:
            平滑后的掩模
        """
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(mask.astype(np.float64), sigma=sigma)
        return np.clip(smoothed, 0.0, 1.0)


class LineSpaceGenerator(TestStructureGenerator):
    """
    Line/Space 测试结构生成器

    生成周期性的线/空间结构，用于测试光刻分辨率、
    线宽均匀性(LWR)、边缘粗糙度(LER)等性能。
    """

    @classmethod
    def generate(cls, params: LineSpaceParams) -> np.ndarray:
        """
        生成 Line/Space 结构

        Args:
            params: Line/Space 结构参数

        Returns:
            二元掩模数组，1表示铬层（不透明），0表示石英（透明）
        """
        ny, nx = params.grid_size
        mask = np.zeros((ny, nx), dtype=np.float64)

        cd_pix = params.cd_pixels
        pitch_pix = params.pitch_pixels

        if params.orientation == LineOrientation.HORIZONTAL:
            period = pitch_pix
            line_width = cd_pix

            y_indices = np.arange(ny)
            y_mod = np.mod(y_indices + period, period)

            line_mask = (y_mod < line_width)[:, np.newaxis]
            mask = np.broadcast_to(line_mask, (ny, nx)).astype(np.float64)

            if params.num_lines is not None:
                total_lines = int(np.ceil(ny / period))
                if params.num_lines < total_lines:
                    keep = params.num_lines
                    start_y = (ny - keep * period) / 2
                    keep_mask = np.zeros(ny, dtype=bool)
                    for i in range(keep):
                        s = int(start_y + i * period)
                        e = int(s + period)
                        if 0 <= s < ny:
                            keep_mask[s:min(e, ny)] = True
                    mask = mask * keep_mask[:, np.newaxis].astype(np.float64)

        else:
            period = pitch_pix
            line_width = cd_pix

            x_indices = np.arange(nx)
            x_mod = np.mod(x_indices + period, period)

            line_mask = (x_mod < line_width)[np.newaxis, :]
            mask = np.broadcast_to(line_mask, (ny, nx)).astype(np.float64)

            if params.num_lines is not None:
                total_lines = int(np.ceil(nx / period))
                if params.num_lines < total_lines:
                    keep = params.num_lines
                    start_x = (nx - keep * period) / 2
                    keep_mask = np.zeros(nx, dtype=bool)
                    for i in range(keep):
                        s = int(start_x + i * period)
                        e = int(s + period)
                        if 0 <= s < nx:
                            keep_mask[s:min(e, nx)] = True
                    mask = mask * keep_mask[np.newaxis, :].astype(np.float64)

        if params.corner_rounding > 0:
            mask = cls._apply_corner_rounding(mask, params.corner_rounding, params.pixel_size)

        return mask


class ContactHoleGenerator(TestStructureGenerator):
    """
    Contact Hole 测试结构生成器

    生成接触孔阵列结构，用于测试接触孔CD、
    圆形度、阵列均匀性等性能。
    """

    @classmethod
    def generate(cls, params: ContactHoleParams) -> np.ndarray:
        ny, nx = params.grid_size
        mask = np.ones((ny, nx), dtype=np.float64)

        cd_pix = params.cd_pixels
        radius_pix = cd_pix / 2.0

        yy, xx = cls._get_coordinates(params.grid_size, params.pixel_size)

        if params.pattern == HolePattern.SQUARE_GRID:
            centers = cls._generate_square_grid_centers(params.grid_size, params.pitch, params.pixel_size)
        else:
            centers = cls._generate_hexagonal_centers(params.grid_size, params.pitch, params.pixel_size)

        for (yc, xc) in centers:
            if params.hole_shape == 'circle':
                mask = cls._draw_circular_hole(mask, yy, xx, yc, xc, radius_pix,
                                         params.aspect_ratio, params.rotation)
            else:
                mask = cls._draw_square_hole(mask, yy, xx, yc, xc, cd_pix,
                                              params.aspect_ratio, params.rotation)

        if params.corner_rounding > 0:
            mask = cls._apply_corner_rounding(mask, params.corner_rounding, params.pixel_size)

        return mask

    @staticmethod
    def _generate_square_grid_centers(grid_size: Tuple[int, int],
                                      pitch: float,
                                      pixel_size: float) -> list:
        """生成正方形网格的孔中心坐标（物理坐标，与_get_coordinates一致）"""
        ny, nx = grid_size
        cy, cx = ny / 2.0, nx / 2.0
        pitch_pix = pitch / pixel_size

        centers = []
        y_start_pix = pitch_pix / 2
        while y_start_pix < ny:
            x_start_pix = pitch_pix / 2
            while x_start_pix < nx:
                y_phys = (y_start_pix - cy) * pixel_size
                x_phys = (x_start_pix - cx) * pixel_size
                centers.append((y_phys, x_phys))
                x_start_pix += pitch_pix
            y_start_pix += pitch_pix

        return centers

    @staticmethod
    def _generate_hexagonal_centers(grid_size: Tuple[int, int],
                                    pitch: float,
                                    pixel_size: float) -> list:
        """生成六边形排列的孔中心坐标（物理坐标，与_get_coordinates一致）"""
        ny, nx = grid_size
        cy, cx = ny / 2.0, nx / 2.0
        pitch_pix = pitch / pixel_size
        row_spacing = pitch_pix * np.sqrt(3) / 2

        centers = []
        row = 0
        y_start_pix = pitch_pix / 2
        while y_start_pix < ny:
            x_offset = pitch_pix / 2 if row % 2 == 1 else 0
            x_start_pix = pitch_pix / 2 + x_offset
            while x_start_pix < nx:
                y_phys = (y_start_pix - cy) * pixel_size
                x_phys = (x_start_pix - cx) * pixel_size
                centers.append((y_phys, x_phys))
                x_start_pix += pitch_pix
            y_start_pix += row_spacing
            row += 1

        return centers

    @staticmethod
    def _draw_circular_hole(mask: np.ndarray,
                           yy: np.ndarray,
                           xx: np.ndarray,
                           yc: float,
                           xc: float,
                           radius: float,
                           aspect_ratio: float,
                           rotation: float) -> np.ndarray:
        """绘制圆形接触孔"""
        if aspect_ratio != 1.0 or rotation != 0.0:
            theta = np.radians(rotation)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            dx = xx - xc
            dy = yy - yc
            x_rot = dx * cos_t + dy * sin_t
            y_rot = -dx * sin_t + dy * cos_t

            a = radius
            b = radius / aspect_ratio
            ellipse = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0
        else:
            dist = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
            ellipse = dist <= radius

        result = mask.copy()
        result[ellipse] = 0.0
        return result

    @staticmethod
    def _draw_square_hole(mask: np.ndarray,
                           yy: np.ndarray,
                           xx: np.ndarray,
                           yc: float,
                           xc: float,
                           size: float,
                           aspect_ratio: float,
                           rotation: float) -> np.ndarray:
        """绘制方形接触孔"""
        theta = np.radians(rotation)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx = xx - xc
        dy = yy - yc

        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        half_w = size / 2
        half_h = (size / aspect_ratio) / 2

        square = (np.abs(x_rot) <= half_w) & (np.abs(y_rot) <= half_h)

        result = mask.copy()
        result[square] = 0.0
        return result


class LShapedCornerGenerator(TestStructureGenerator):
    """
    L-shaped Corner 测试结构生成器

    生成L形拐角结构，用于测试拐角圆滑度、
    线宽偏差等性能。
    """

    @classmethod
    def generate(cls, params: LShapedCornerParams) -> np.ndarray:
        ny, nx = params.grid_size
        mask = np.zeros((ny, nx), dtype=np.float64)

        cd_pix = params.cd_pixels
        arm_pix = params.arm_length_pixels

        yy, xx = cls._get_coordinates(params.grid_size, params.pixel_size)

        if params.corner_type == 'inner':
            arm1 = (yy >= -arm_pix / 2) & (yy <= arm_pix / 2 + cd_pix) & (xx >= -arm_pix / 2)
            arm2 = (xx >= -arm_pix / 2) & (xx <= arm_pix / 2 + cd_pix) & (yy >= -arm_pix / 2)
        else:
            arm1 = (yy >= -arm_pix / 2 - cd_pix) & (yy <= arm_pix / 2) & (xx >= -arm_pix / 2)
            arm2 = (xx >= -arm_pix / 2 - cd_pix) & (xx <= arm_pix / 2) & (yy >= -arm_pix / 2)

        l_shape = arm1 | arm2
        mask[l_shape] = 1.0

        if params.corner_rounding > 0:
            mask = cls._apply_corner_rounding(mask, params.corner_rounding, params.pixel_size)

        return mask


class TJunctionGenerator(TestStructureGenerator):
    """
    T-junction 测试结构生成器

    生成T形结结构，用于测试线端缩短、
    连接区域的光刻性能。
    """

    @classmethod
    def generate(cls, params: TJunctionParams) -> np.ndarray:
        ny, nx = params.grid_size
        mask = np.zeros((ny, nx), dtype=np.float64)

        cd_pix = params.cd_pixels
        stem_pix = params.stem_length_pixels
        branch_pix = params.branch_length_pixels

        yy, xx = cls._get_coordinates(params.grid_size, params.pixel_size)

        stem = (np.abs(yy) <= cd_pix / 2) & (np.abs(xx) <= stem_pix / 2)

        branch = (np.abs(xx) <= branch_pix) & (yy >= cd_pix / 2) & (yy <= cd_pix / 2 + cd_pix)

        t_shape = stem | branch
        mask[t_shape] = 1.0

        if params.corner_rounding > 0:
            mask = cls._apply_corner_rounding(mask, params.corner_rounding, params.pixel_size)

        return mask


class SRAMBitcellGenerator(TestStructureGenerator):
    """
    SRAM Bitcell 测试结构生成器

    生成SRAM位单元结构，用于测试高密度
    存储单元的光刻性能。
    """

    @classmethod
    def generate(cls, params: SRAMBitcellParams) -> np.ndarray:
        ny, nx = params.grid_size
        mask = np.zeros((ny, nx), dtype=np.float64)

        cd_pix = params.cd_pixels

        yy, xx = cls._get_coordinates(params.grid_size, params.pixel_size)

        if params.bitcell_type == '6T':
            mask = cls._generate_6t_bitcell(mask, yy, xx, cd_pix)
        else:
            mask = cls._generate_thin_film_bitcell(mask, yy, xx, cd_pix)

        if params.corner_rounding > 0:
            mask = cls._apply_corner_rounding(mask, params.corner_rounding, params.pixel_size)

        return mask

    @staticmethod
    def _generate_6t_bitcell(mask: np.ndarray,
                         yy: np.ndarray,
                         xx: np.ndarray,
                         cd_pix: float) -> np.ndarray:
        """生成6T SRAM位单元结构"""
        pull_up1 = (np.abs(yy - cd_pix * 1.5) <= cd_pix / 2) & (np.abs(xx - cd_pix) <= cd_pix / 2)
        pull_up2 = (np.abs(yy - cd_pix * 1.5) <= cd_pix / 2) & (np.abs(xx + cd_pix) <= cd_pix / 2)

        pull_down1 = (np.abs(yy + cd_pix * 1.5) <= cd_pix / 2) & (np.abs(xx - cd_pix) <= cd_pix / 2)
        pull_down2 = (np.abs(yy + cd_pix * 1.5) <= cd_pix / 2) & (np.abs(xx + cd_pix) <= cd_pix / 2)

        pass_gate1 = (np.abs(yy) <= cd_pix / 2) & (np.abs(xx - cd_pix * 2.5) <= cd_pix / 2)
        pass_gate2 = (np.abs(yy) <= cd_pix / 2) & (np.abs(xx + cd_pix * 2.5) <= cd_pix / 2)

        vdd_line = (np.abs(yy - cd_pix * 2.5) <= cd_pix / 2) & (np.abs(xx) <= cd_pix * 1.5)
        vss_line = (np.abs(yy + cd_pix * 2.5) <= cd_pix / 2) & (np.abs(xx) <= cd_pix * 1.5)

        bit_line1 = (np.abs(xx - cd_pix * 2.5) <= cd_pix / 2) & (np.abs(yy) <= cd_pix * 2.5)
        bit_line2 = (np.abs(xx + cd_pix * 2.5) <= cd_pix / 2) & (np.abs(yy) <= cd_pix * 2.5)

        word_line = (np.abs(yy) <= cd_pix / 2) & (np.abs(xx) <= cd_pix * 3.0)

        all_features = pull_up1 | pull_up2 | pull_down1 | pull_down2 | \
                       pass_gate1 | pass_gate2 | vdd_line | vss_line | \
                       bit_line1 | bit_line2 | word_line

        result = mask.copy()
        result[all_features] = 1.0
        return result

    @staticmethod
    def _generate_thin_film_bitcell(mask: np.ndarray,
                                   yy: np.ndarray,
                                   xx: np.ndarray,
                                   cd_pix: float) -> np.ndarray:
        """生成薄膜SRAM位单元结构"""
        transistor1 = (np.abs(yy) <= cd_pix / 2) & (np.abs(xx - cd_pix) <= cd_pix / 2)
        transistor2 = (np.abs(yy) <= cd_pix / 2) & (np.abs(xx + cd_pix) <= cd_pix / 2)

        storage_node = (np.abs(yy - cd_pix * 1.5) <= cd_pix) & (np.abs(xx) <= cd_pix)

        access_line = (np.abs(xx) <= cd_pix / 2) & (np.abs(yy) <= cd_pix * 2.5)

        bit_line = (np.abs(yy + cd_pix * 1.5) <= cd_pix / 2) & (np.abs(xx) <= cd_pix * 2.0)

        all_features = transistor1 | transistor2 | storage_node | access_line | bit_line

        result = mask.copy()
        result[all_features] = 1.0
        return result


def generate_test_structure(params: Union[TestStructureParams, dict]) -> np.ndarray:
    """
    根据参数生成测试结构

    统一的测试结构生成入口函数。

    Args:
        params: 测试结构参数，可以是TestStructureParams子类实例或字典

    Returns:
        生成的二元掩模数组

    Examples:
        >>> params = LineSpaceParams(grid_size=(256, 256), cd=45, pitch=90)
        >>> mask = generate_test_structure(params)
    """
    if isinstance(params, dict):
        params_copy = params.copy()
        structure_type = params_copy.get('structure_type', TestStructureType.LINE_SPACE)
        if isinstance(structure_type, str):
            structure_type = TestStructureType(structure_type)
            params_copy['structure_type'] = structure_type

        param_classes = {
            TestStructureType.LINE_SPACE: LineSpaceParams,
            TestStructureType.CONTACT_HOLE: ContactHoleParams,
            TestStructureType.L_SHAPED_CORNER: LShapedCornerParams,
            TestStructureType.T_JUNCTION: TJunctionParams,
            TestStructureType.SRAM_BITCELL: SRAMBitcellParams,
        }
        param_class = param_classes.get(structure_type, LineSpaceParams)
        params = param_class(**params_copy)

    generators = {
        TestStructureType.LINE_SPACE: LineSpaceGenerator,
        TestStructureType.CONTACT_HOLE: ContactHoleGenerator,
        TestStructureType.L_SHAPED_CORNER: LShapedCornerGenerator,
        TestStructureType.T_JUNCTION: TJunctionGenerator,
        TestStructureType.SRAM_BITCELL: SRAMBitcellGenerator,
    }

    generator = generators.get(params.structure_type)
    if generator is None:
        raise ValueError(f"不支持的测试结构类型: {params.structure_type}")

    return generator.generate(params)


__all__ = [
    'TestStructureType',
    'LineOrientation',
    'HolePattern',
    'TestStructureParams',
    'LineSpaceParams',
    'ContactHoleParams',
    'LShapedCornerParams',
    'TJunctionParams',
    'SRAMBitcellParams',
    'TestStructureGenerator',
    'LineSpaceGenerator',
    'ContactHoleGenerator',
    'LShapedCornerGenerator',
    'TJunctionGenerator',
    'SRAMBitcellGenerator',
    'generate_test_structure',
]
