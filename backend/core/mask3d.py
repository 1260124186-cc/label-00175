# -*- coding: utf-8 -*-
"""
三维掩模电磁邻近效应模块 (Mask3D EPE Module)

建模高数值孔径 (High-NA) 光刻中掩模的三维近场效应，包括：
1. 掩模表面形貌建模 —— 铬层表面粗糙度、起伏形貌
2. 侧壁角度建模 —— 吸收层侧壁的梯形/倾斜形貌
3. 薄膜堆栈建模 —— ARC、Mo/Si 多层膜、覆盖层的分层结构
4. 简化边界元 (S-BEM) 近场散射 —— 基于等效边缘电流的快速近场计算
5. RCWA 耦合修正 Hopkins 成像 —— 将 3D 效应转化为等效掩模透射率修正
6. CD 偏差分析 —— 量化 Mask3D 效应导致的线宽偏差

参考:
    - P. B. Fischer et al., "Mask 3D effects for 193nm immersion lithography,"
      Proc. SPIE 5754 (2005)
    - K. Adam et al., "Investigation of mask induced polarization and 3D effects
      using rigorous simulation," Proc. SPIE 6520 (2007)
    - A. Erdmann et al., "Fast mask 3D modeling for optical lithography,"
      Proc. SPIE 7640 (2010)
    - L. Pang et al., "EMF mask 3D effect modeling for 22nm and beyond,"
      Proc. SPIE 7973 (2011)
    - Moharam & Gaylord, "Rigorous coupled-wave analysis of planar-grating
      diffraction," J. Opt. Soc. Am. 72, 1385 (1982)
"""

from __future__ import annotations

import logging
import warnings
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numba import jit, prange

from core.array_backend import get_backend, DeviceType
from core.polarization import (
    ThinFilmLayer,
    ThinFilmStack,
    MaterialDispersion,
    STANDARD_MATERIALS,
    JonesVector,
)
from core.rigorous_sim import (
    RCWASolver1D,
    RCWASolver2D,
    RCWAConfig,
    Polarization,
    _toeplitz_epsilon_1d,
    _toeplitz_inverse_epsilon_1d,
    _safe_kz,
)


logger = logging.getLogger(__name__)


# =============================================================================
# 基础枚举与配置类
# =============================================================================
class MaskType(str, Enum):
    """掩模类型"""
    BINARY_COG = "binary_cog"           # 传统二元铬掩模 (Cr on Glass)
    ATTENUATED_PSM = "att_psm"         # 衰减相移掩模 (MoSi 等)
    ALTERNATING_PSM = "alt_psm"        # 交替相移掩模
    EUV_REFLECTIVE = "euv_reflective"   # EUV 反射掩模 (Mo/Si 多层)


class SidewallProfile(str, Enum):
    """侧壁形貌类型"""
    RECTANGULAR = "rectangular"         # 理想矩形 (90度)
    TRAPEZOIDAL = "trapezoidal"        # 梯形 (正/负侧壁角)
    ROUNDED_TOP = "rounded_top"        # 顶部圆角
    ROUNDED_BOTTOM = "rounded_bottom"  # 底部圆角
    REENTRANT = "reentrant"            # 倒梯形 (负侧壁角, 常出现在深宽比较大的刻蚀中)


class RoughnessModel(str, Enum):
    """表面粗糙度模型"""
    GAUSSIAN = "gaussian"              # 高斯相关粗糙面
    FRACTAL = "fractal"                # 分形粗糙面
    EXPONENTIAL = "exponential"        # 指数相关粗糙面
    NONE = "none"                      # 无粗糙度


@dataclass
class SidewallParams:
    """
    侧壁形貌参数

    Attributes:
        profile_type: 侧壁形貌类型
        sidewall_angle_deg: 侧壁与法线夹角 (度)。0=理想垂直, 正值=向外倾斜(正梯形),
                            负值=向内倾斜(倒梯形)。典型范围: -5 ~ +10 deg
        top_rounding_nm: 顶部圆角半径 (nm), 用于 ROUNDED_TOP 模式
        bottom_rounding_nm: 底部圆角半径 (nm), 用于 ROUNDED_BOTTOM 模式
        top_cd_bias_nm: 顶部相对标称 CD 的偏差 (nm)。正值=顶部线宽增大
        bottom_cd_bias_nm: 底部相对标称 CD 的偏差 (nm)。负值=底部线宽减小(undercut)
    """
    profile_type: SidewallProfile = SidewallProfile.RECTANGULAR
    sidewall_angle_deg: float = 0.0
    top_rounding_nm: float = 0.0
    bottom_rounding_nm: float = 0.0
    top_cd_bias_nm: float = 0.0
    bottom_cd_bias_nm: float = 0.0

    def __post_init__(self):
        if isinstance(self.profile_type, str):
            self.profile_type = SidewallProfile(self.profile_type)


@dataclass
class RoughnessParams:
    """
    表面粗糙度参数

    Attributes:
        model: 粗糙度统计模型
        rms_height_nm: 均方根高度 (nm), 典型值 0.5 ~ 5 nm
        correlation_length_nm: 相关长度 (nm), 典型值 10 ~ 100 nm
        rms_slope: 均方根斜率 (仅 FRACTAL 模型), 控制粗糙度频谱
        exponent_alpha: 指数相关模型的衰减指数 (指数模型), 典型值 1~3
        seed: 随机种子 (用于可重复性)
    """
    model: RoughnessModel = RoughnessModel.NONE
    rms_height_nm: float = 0.0
    correlation_length_nm: float = 20.0
    rms_slope: float = 0.05
    exponent_alpha: float = 2.0
    seed: int = 42


@dataclass
class AbsorberLayer:
    """
    掩模吸收层 (或反射层) 结构描述

    Attributes:
        material_name: 材料名称, 对应 STANDARD_MATERIALS 键
        thickness_nm: 层厚度 (nm)。典型 Cr 层: 60~80 nm
        custom_n: 自定义复折射率 (若材料不在标准库中)
        sidewall: 侧壁形貌参数
        surface_roughness: 表面粗糙度参数
        is_stack: 是否为多层膜 (如 Mo/Si 多层反射镜)
        sublayers: 子层列表 (当 is_stack=True 时)
    """
    material_name: str = "cr"
    thickness_nm: float = 70.0
    custom_n: Optional[complex] = None
    sidewall: SidewallParams = field(default_factory=SidewallParams)
    surface_roughness: RoughnessParams = field(default_factory=RoughnessParams)
    is_stack: bool = False
    sublayers: Optional[List[ThinFilmLayer]] = None

    def get_refractive_index(self, wavelength_nm: float) -> complex:
        """获取指定波长下的复折射率"""
        if self.custom_n is not None:
            return complex(self.custom_n)
        if self.material_name in STANDARD_MATERIALS:
            return STANDARD_MATERIALS[self.material_name].get_n(wavelength_nm)
        return 1.0 + 0.0j


@dataclass
class Mask3DConfig:
    """
    三维掩模完整配置

    封装掩模的完整三维结构描述，包括基底、多层薄膜堆栈、吸收层结构、
    形貌参数及求解器配置。

    Attributes:
        mask_type: 掩模类型
        wavelength_nm: 工作波长 (nm)
        absorber: 吸收层/反射层参数
        substrate_material: 基底材料 (熔融石英等)
        substrate_n: 基底复折射率 (覆盖 material 设置)
        arc_layer: 抗反射涂层参数 (可选)
        buffer_layer: 缓冲层参数 (可选, 如 Ta 层)
        capping_layer: 覆盖层参数 (可选, EUV Ru 层)
        multilayer_stack: 多层反射膜堆栈 (EUV 使用, 如 Mo/Si)
        enable_sidewall: 是否启用侧壁效应修正
        enable_roughness: 是否启用表面粗糙度修正
        enable_multilayer: 是否启用多层薄膜干涉效应
        sbem_n_segments: S-BEM 边缘分段数, 越大精度越高但速度越慢
        sbem_k_sampling: S-BEM 空间波数采样密度
        rcwa_correction_enabled: 是否使用 RCWA 进行精确修正
        rcwa_n_orders: RCWA 衍射级次截断 (单侧)
        hopkins_correction_mode: Hopkins 修正模式
    """
    mask_type: MaskType = MaskType.BINARY_COG
    wavelength_nm: float = 193.0

    absorber: AbsorberLayer = field(default_factory=AbsorberLayer)
    substrate_material: str = "sio2"
    substrate_n: Optional[complex] = None
    arc_layer: Optional[AbsorberLayer] = None
    buffer_layer: Optional[AbsorberLayer] = None
    capping_layer: Optional[AbsorberLayer] = None
    multilayer_stack: Optional[ThinFilmStack] = None

    enable_sidewall: bool = True
    enable_roughness: bool = False
    enable_multilayer: bool = False

    sbem_n_segments: int = 128
    sbem_k_sampling: int = 32
    rcwa_correction_enabled: bool = True
    rcwa_n_orders: int = 7

    hopkins_correction_mode: str = "effective_transmission"

    def __post_init__(self):
        if isinstance(self.mask_type, str):
            self.mask_type = MaskType(self.mask_type)

    def get_substrate_n(self) -> complex:
        """获取基底折射率"""
        if self.substrate_n is not None:
            return complex(self.substrate_n)
        if self.substrate_material in STANDARD_MATERIALS:
            return STANDARD_MATERIALS[self.substrate_material].get_n(self.wavelength_nm)
        return 1.56 + 0.0j

    def get_superstrate_n(self) -> complex:
        """获取上层介质 (空气/浸没液) 折射率"""
        if self.mask_type == MaskType.EUV_REFLECTIVE:
            return 1.0 + 0.0j
        if self.wavelength_nm < 200:
            return 1.437 + 0.0j
        return 1.0002 + 0.0j


# =============================================================================
# 掩模表面形貌生成
# =============================================================================
class MaskTopography:
    """
    掩模三维形貌生成器

    根据配置参数生成掩模的实际表面几何形状，包括：
    - 吸收层侧壁倾斜
    - 顶部/底部圆角
    - 表面粗糙度起伏

    输出为离散的高度场 (height field) 或分段解析描述，供 S-BEM 使用。
    """

    def __init__(self, config: Mask3DConfig):
        self.cfg = config

    def generate_cross_section(
        self,
        nominal_cd_nm: float,
        num_points_x: int = 512,
        x_range_nm: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        生成一维线结构的侧截面轮廓 (XZ 平面)

        Args:
            nominal_cd_nm: 标称线宽 (nm)
            num_points_x: X 方向采样点数
            x_range_nm: X 扫描范围 (nm), None 时自动取 3*CD

        Returns:
            dict 包含 x, z_top, z_bottom, absorber_mask, height_profile, sidewall_indices 等
        """
        absorber = self.cfg.absorber
        thickness = absorber.thickness_nm
        sw = absorber.sidewall

        if x_range_nm is None:
            half_span = max(3.0 * nominal_cd_nm, 400.0)
            x_range_nm = (-half_span, half_span)

        x = np.linspace(x_range_nm[0], x_range_nm[1], num_points_x, dtype=np.float64)
        dx = x[1] - x[0]

        z_bottom = np.zeros_like(x)
        z_top = np.zeros_like(x)

        cd_top = nominal_cd_nm + sw.top_cd_bias_nm
        cd_bot = nominal_cd_nm + sw.bottom_cd_bias_nm
        half_cd_top = cd_top / 2.0
        half_cd_bot = cd_bot / 2.0

        if sw.profile_type == SidewallProfile.RECTANGULAR:
            within = np.abs(x) <= half_cd_top
            z_top[within] = thickness

        elif sw.profile_type == SidewallProfile.TRAPEZOIDAL:
            sidewall_dx = abs(half_cd_top - half_cd_bot)
            if sidewall_dx < 1e-6 and abs(sw.sidewall_angle_deg) > 1e-6:
                sidewall_dx = thickness * np.tan(np.deg2rad(sw.sidewall_angle_deg))
                half_cd_top = half_cd_bot + sidewall_dx
            for i, xi in enumerate(x):
                ax = abs(xi)
                if ax <= min(half_cd_top, half_cd_bot):
                    z_top[i] = thickness
                elif ax <= max(half_cd_top, half_cd_bot):
                    frac = (ax - min(half_cd_top, half_cd_bot)) / (sidewall_dx + 1e-30)
                    z_top[i] = thickness * (1.0 - np.clip(frac, 0.0, 1.0))

        elif sw.profile_type == SidewallProfile.ROUNDED_TOP:
            within = np.abs(x) <= half_cd_top
            z_top[within] = thickness
            r_nm = max(sw.top_rounding_nm, 1e-6)
            left_start_round = half_cd_top - r_nm
            for i, xi in enumerate(x):
                ax = abs(xi)
                if left_start_round < ax <= half_cd_top and r_nm > 0:
                    d = ax - left_start_round
                    if d <= r_nm:
                        arc_offset = r_nm - np.sqrt(max(0.0, r_nm ** 2 - d ** 2))
                        z_top[i] = max(0.0, thickness - arc_offset)

        elif sw.profile_type == SidewallProfile.ROUNDED_BOTTOM:
            r_nm = max(sw.bottom_rounding_nm, 1e-6)
            for i, xi in enumerate(x):
                ax = abs(xi)
                if ax <= half_cd_bot:
                    z_top[i] = thickness
                elif ax <= half_cd_bot + r_nm and r_nm > 0:
                    d = ax - half_cd_bot
                    z_top[i] = r_nm - np.sqrt(max(0.0, r_nm ** 2 - d ** 2))
                elif ax <= half_cd_top:
                    frac = (ax - half_cd_bot - r_nm) / max(half_cd_top - half_cd_bot - r_nm, 1e-6)
                    z_top[i] = max(0.0, thickness * np.clip(frac, 0.0, 1.0))

        elif sw.profile_type == SidewallProfile.REENTRANT:
            if half_cd_top > half_cd_bot:
                half_cd_top, half_cd_bot = half_cd_bot, half_cd_top
            for i, xi in enumerate(x):
                ax = abs(xi)
                if ax <= half_cd_top:
                    z_top[i] = thickness
                elif ax <= half_cd_bot:
                    frac = (ax - half_cd_top) / max(half_cd_bot - half_cd_top, 1e-6)
                    z_top[i] = thickness * (1.0 - np.clip(frac, 0.0, 1.0))

        height_profile = z_top - z_bottom
        dx_val = x[1] - x[0]

        h_max = np.max(height_profile)
        sidewall_regions = []
        if h_max > 1e-6:
            # 使用高度梯度检测侧壁过渡区域 (|dh/dx| 大于阈值即为侧壁)
            if len(height_profile) > 2:
                dh_dx = np.abs(np.gradient(height_profile, dx_val))
                grad_threshold = h_max / (max(1.0, 2.0 * thickness)) if thickness > 0 else 0.1
                sw_mask = dh_dx > grad_threshold
                # 找连通区域
                transition = np.diff(sw_mask.astype(np.int8))
                starts = np.where(transition == 1)[0] + 1
                ends = np.where(transition == -1)[0] + 1
                # 处理边界
                if sw_mask[0]:
                    starts = np.concatenate(([0], starts))
                if sw_mask[-1]:
                    ends = np.concatenate((ends, [len(sw_mask) - 1]))
                for s, e in zip(starts, ends):
                    sidewall_regions.append((s, e))
            # 回退: 若未检测到侧壁 (理想矩形), 标记边缘像素附近为侧壁
            if len(sidewall_regions) == 0:
                transition2 = np.diff((height_profile > 0.05 * h_max).astype(np.int8))
                for idx in np.where(np.abs(transition2) > 0)[0]:
                    s = max(0, idx - 1)
                    e = min(len(height_profile) - 1, idx + 2)
                    sidewall_regions.append((s, e))
                # 去重合并
                if len(sidewall_regions) > 2:
                    merged = []
                    sidewall_regions.sort()
                    for s, e in sidewall_regions:
                        if merged and s <= merged[-1][1]:
                            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                        else:
                            merged.append((s, e))
                    sidewall_regions = merged

        absorber_mask = height_profile > (0.05 * max(thickness, 1e-6))

        return {
            "x": x,
            "z_top": z_top,
            "z_bottom": z_bottom,
            "height_profile": height_profile,
            "absorber_mask": absorber_mask,
            "sidewall_indices": sidewall_regions,
            "dx_nm": dx_val,
        }

    def generate_roughness(
        self,
        grid_shape: Tuple[int, int],
        pixel_size_nm: float,
    ) -> Optional[np.ndarray]:
        """
        生成表面粗糙度高度起伏图

        Args:
            grid_shape: (Ny, Nx) 网格尺寸
            pixel_size_nm: 像素大小 (nm)

        Returns:
            高度起伏数组 shape=(Ny, Nx), 单位 nm; None 表示无粗糙度
        """
        rp = self.cfg.absorber.surface_roughness
        if rp.model == RoughnessModel.NONE or rp.rms_height_nm <= 1e-6:
            return None

        rng = np.random.RandomState(rp.seed)
        Ny, Nx = grid_shape

        if rp.model == RoughnessModel.GAUSSIAN:
            raw = rng.randn(Ny, Nx)
            ky = np.fft.fftfreq(Ny, pixel_size_nm)
            kx = np.fft.fftfreq(Nx, pixel_size_nm)
            KY, KX = np.meshgrid(ky, kx, indexing="ij")
            K_sq = KX ** 2 + KY ** 2
            sigma_k = 1.0 / (rp.correlation_length_nm * np.sqrt(2.0))
            gaussian_filter = np.exp(-2.0 * (np.pi ** 2) * K_sq / (sigma_k ** 2 + 1e-30))
            f_raw = np.fft.fft2(raw)
            f_rough = f_raw * gaussian_filter
            rough = np.real(np.fft.ifft2(f_rough))
            current_rms = np.sqrt(np.mean(rough ** 2))
            if current_rms > 1e-10:
                rough = rough * (rp.rms_height_nm / current_rms)

        elif rp.model == RoughnessModel.EXPONENTIAL:
            raw = rng.randn(Ny, Nx)
            ky = np.fft.fftfreq(Ny, pixel_size_nm)
            kx = np.fft.fftfreq(Nx, pixel_size_nm)
            KY, KX = np.meshgrid(ky, kx, indexing="ij")
            K = np.sqrt(KX ** 2 + KY ** 2)
            xi = rp.correlation_length_nm
            alpha = rp.exponent_alpha
            exp_filter = 1.0 / (1.0 + (2.0 * np.pi * K * xi / np.sqrt(max(alpha, 0.1))) ** alpha + 1e-30)
            f_raw = np.fft.fft2(raw)
            f_rough = f_raw * exp_filter
            rough = np.real(np.fft.ifft2(f_rough))
            current_rms = np.sqrt(np.mean(rough ** 2))
            if current_rms > 1e-10:
                rough = rough * (rp.rms_height_nm / current_rms)

        elif rp.model == RoughnessModel.FRACTAL:
            rough = np.zeros((Ny, Nx), dtype=np.float64)
            octaves = max(3, int(np.log2(min(Ny, Nx))) - 2)
            amplitude = rp.rms_height_nm * (2.0 ** (-rp.rms_slope * 0.5))
            freq_mult = 2.0
            try:
                from scipy import ndimage
                for _ in range(octaves):
                    coarse_shape = (max(2, int(Ny / freq_mult)), max(2, int(Nx / freq_mult)))
                    noise_coarse = rng.randn(*coarse_shape)
                    noise_fine = ndimage.zoom(
                        noise_coarse,
                        (Ny / coarse_shape[0], Nx / coarse_shape[1]),
                        order=1,
                    )
                    rough += amplitude * noise_fine
                    amplitude *= (2.0 ** (-rp.rms_slope))
                    freq_mult *= 2.0
            except ImportError:
                rough = rng.randn(Ny, Nx) * rp.rms_height_nm
            current_rms = np.sqrt(np.mean(rough ** 2))
            if current_rms > 1e-10:
                rough = rough * (rp.rms_height_nm / current_rms)

        else:
            return None

        return rough


# =============================================================================
# 简化边界元 (S-BEM) 近场散射求解
# =============================================================================
class SimplifiedBEMScattering:
    """
    简化边界元 (Simplified BEM) 近场散射求解器

    核心思想：
    1. 将掩模吸收层边缘等效为线电流/磁流分布
    2. 利用物理光学近似计算边缘绕射场
    3. 与 Kirchhoff 口径场叠加，获得修正后的近场分布

    优势:
        - 速度远快于完整 FDTD/RCWA
        - 可处理任意边缘几何 (斜侧壁、圆角等)
        - 适合 OPC/ILT 循环中的快速 3D 修正
    """

    def __init__(self, config: Mask3DConfig):
        self.cfg = config
        self.k0 = 2.0 * np.pi / config.wavelength_nm
        self.n_sub = config.get_substrate_n()
        self.n_sup = config.get_superstrate_n()
        self.n_abs = config.absorber.get_refractive_index(config.wavelength_nm)
        self.topography = MaskTopography(config)

    def compute_edge_current_1d(
        self,
        nominal_cd_nm: float,
        theta_deg: float = 0.0,
        polarization: Polarization = Polarization.UNPOLARIZED,
    ) -> Dict[str, Any]:
        """
        计算一维线/间隙结构边缘的等效电磁流
        """
        theta_rad = np.deg2rad(theta_deg)
        section = self.topography.generate_cross_section(
            nominal_cd_nm, num_points_x=2 * self.cfg.sbem_n_segments + 1
        )
        x = section["x"]
        h = section["height_profile"]
        dx_nm = section["dx_nm"]
        N = len(x)

        kx_inc = self.k0 * self.n_sub * np.sin(theta_rad)
        kz_inc = np.lib.scimath.sqrt((self.k0 * self.n_sub) ** 2 - kx_inc ** 2)
        if np.imag(kz_inc) < 0:
            kz_inc = -kz_inc

        sw_regions = section["sidewall_indices"]
        half_cd = nominal_cd_nm / 2.0
        idx_center = N // 2
        sw_width_nm = max(
            10.0,
            abs(self.cfg.absorber.sidewall.top_cd_bias_nm -
                self.cfg.absorber.sidewall.bottom_cd_bias_nm) + 5.0,
        )
        sw_pix = int(sw_width_nm / dx_nm) + 1

        if len(sw_regions) >= 2:
            left_sw = slice(sw_regions[0][0], sw_regions[0][1] + 1)
            right_sw = slice(sw_regions[1][0], sw_regions[1][1] + 1)
        else:
            left_start = max(0, int(idx_center - half_cd / dx_nm - sw_pix))
            left_end = min(N, int(idx_center - half_cd / dx_nm + sw_pix) + 1)
            right_start = max(0, int(idx_center + half_cd / dx_nm - sw_pix))
            right_end = min(N, int(idx_center + half_cd / dx_nm + sw_pix) + 1)
            left_sw = slice(left_start, left_end)
            right_sw = slice(right_start, right_end)

        result = {
            "section": section,
            "left_edge_slice": left_sw,
            "right_edge_slice": right_sw,
        }

        for pol_label in self._polarization_iterator(polarization):
            J_left, M_left = self._edge_current_po(
                x[left_sw], h[left_sw], kx_inc, kz_inc, pol_label, side="left"
            )
            J_right, M_right = self._edge_current_po(
                x[right_sw], h[right_sw], kx_inc, kz_inc, pol_label, side="right"
            )
            result[f"J_left_{pol_label}"] = J_left
            result[f"M_left_{pol_label}"] = M_left
            result[f"J_right_{pol_label}"] = J_right
            result[f"M_right_{pol_label}"] = M_right

        return result

    @staticmethod
    def _polarization_iterator(pol: Polarization) -> List[str]:
        if pol == Polarization.TE:
            return ["TE"]
        elif pol == Polarization.TM:
            return ["TM"]
        return ["TE", "TM"]

    def _edge_current_po(
        self,
        x_edge: np.ndarray,
        h_edge: np.ndarray,
        kx_inc: complex,
        kz_inc: complex,
        polarization: str,
        side: str = "left",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        物理光学近似: 在边缘表面上计算等效电流 J = n × H, M = -n × E
        """
        N = len(x_edge)
        J = np.zeros(N, dtype=np.complex128)
        M = np.zeros(N, dtype=np.complex128)

        sign_side = -1.0 if side == "left" else 1.0
        n_abs = self.n_abs
        k0 = self.k0

        dh_dx = np.gradient(h_edge, x_edge[1] - x_edge[0]) if N > 1 else np.zeros_like(x_edge)

        for i in range(N):
            si = -sign_side
            sz = 1.0
            if abs(dh_dx[i]) > 1e-6:
                si = -sign_side * dh_dx[i]
                sz = 1.0
            norm = np.sqrt(si ** 2 + sz ** 2) + 1e-12
            nx = si / norm
            nz = sz / norm

            xi = x_edge[i]
            phase_inc = np.exp(1j * (kx_inc * xi + kz_inc * h_edge[i]))

            n_ratio = n_abs / self.n_sub
            cos_theta = np.real(kz_inc) / (k0 * abs(self.n_sub) + 1e-30)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            sin_theta_sq = max(0.0, 1.0 - cos_theta ** 2)
            cos_theta_t = np.lib.scimath.sqrt(
                1.0 - (self.n_sub / n_abs) ** 2 * sin_theta_sq
            )

            if polarization == "TE":
                rs = (cos_theta - n_ratio * cos_theta_t) / (
                    cos_theta + n_ratio * cos_theta_t + 1e-30
                )
                E_y = (1.0 + rs) * phase_inc
                J[i] = E_y * (1.0 - abs(rs)) * 0.5
                M[i] = (1.0 + rs) * cos_theta * E_y * 0.3
            else:
                rp = (n_ratio * cos_theta - cos_theta_t) / (
                    n_ratio * cos_theta + cos_theta_t + 1e-30
                )
                H_y = (1.0 + rp) * phase_inc
                J[i] = (1.0 + rp) * cos_theta * H_y * 0.4
                M[i] = (1.0 - abs(rp)) * H_y * 0.5

        return J, M

    def compute_near_field_correction_2d(
        self,
        mask_pattern: np.ndarray,
        pixel_size_nm: float,
        polarization: Polarization = Polarization.UNPOLARIZED,
        z_observation_nm: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        计算掩模出射面的近场修正 (2D 版图情形)
        """
        Ny, Nx = mask_pattern.shape
        mask = np.asarray(mask_pattern)
        if hasattr(mask_pattern, 'device'):
            mask = get_backend().to_numpy(mask_pattern)

        k0 = self.k0
        n_sub = self.n_sub
        n_abs = self.n_abs

        t_sub = 1.0
        thickness = self.cfg.absorber.thickness_nm
        # 薄掩模透射率: t = exp(-j k0 (n_abs - n_sub) t)
        # exp(+jωt) 约定下传播项为 exp(-j k z), 因此此处用负号
        # n_abs = n' - jk 约定 (STANDARD_MATERIALS), 展开后:
        #   t = exp(-k0 k t) * exp(-j k0 (n'_abs - n'_sub) t) = 吸收 * 相位延迟 ✓
        t_abs = np.exp(-1j * k0 * (n_abs - n_sub) * thickness)
        ideal_field = t_sub * mask + t_abs * (1.0 - mask)

        edge_mask = self._detect_edges(mask)

        cor_amp = np.ones((Ny, Nx), dtype=np.float64)
        cor_phase = np.zeros((Ny, Nx), dtype=np.float64)

        for pol in self._polarization_iterator(polarization):
            ca, cp = self._sbem_correction_field(
                mask, edge_mask, pixel_size_nm, pol
            )
            if polarization == Polarization.UNPOLARIZED:
                if pol == "TE":
                    cor_amp = cor_amp * 0.0 + ca * 0.0 + 1.0
                    cor_amp_ca = ca
                    cor_phase_ca = cp
                else:
                    cor_amp = np.sqrt(np.clip(cor_amp_ca * ca, 0.01, 10.0))
                    cor_phase = 0.5 * (cor_phase_ca + cp)
            else:
                cor_amp = ca
                cor_phase = cp

        if polarization == Polarization.UNPOLARIZED and 'cor_amp_ca' not in dir():
            pass

        correction_complex = cor_amp * np.exp(1j * cor_phase)
        corrected_field = ideal_field * correction_complex

        edge_field = corrected_field - ideal_field

        if z_observation_nm != 0.0:
            corrected_field = self._angular_spectrum_propagate(
                corrected_field, pixel_size_nm, z_observation_nm
            )

        return {
            "ideal_field": ideal_field,
            "corrected_field": corrected_field,
            "correction_amplitude": cor_amp,
            "correction_phase": cor_phase,
            "edge_field": edge_field,
            "edge_mask": edge_mask,
        }

    @staticmethod
    def _detect_edges(mask: np.ndarray) -> np.ndarray:
        """简化边缘检测"""
        gx = np.zeros_like(mask, dtype=np.int8)
        gy = np.zeros_like(mask, dtype=np.int8)
        gx[:, 1:] = np.diff(mask.astype(np.int8), axis=1)
        gy[1:, :] = np.diff(mask.astype(np.int8), axis=0)
        edge_strength = np.abs(gx) + np.abs(gy)
        return (edge_strength > 0).astype(np.float64)

    def _sbem_correction_field(
        self,
        mask: np.ndarray,
        edge_mask: np.ndarray,
        pixel_size_nm: float,
        polarization: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于物理模型的 S-BEM 快速修正场

        核心: 对于每个像素, 基于局部 CD + 侧壁参数 + 偏振,
        从物理近似模型获取振幅/相位修正因子
        """
        Ny, Nx = mask.shape
        cor_amp = np.ones((Ny, Nx), dtype=np.float64)
        cor_phase = np.zeros((Ny, Nx), dtype=np.float64)

        try:
            from scipy.ndimage import distance_transform_edt
            dist_to_edge = distance_transform_edt(1.0 - edge_mask) * pixel_size_nm

            if np.mean(mask) > 0.5:
                cd_estimate = 2.0 * distance_transform_edt(mask) * pixel_size_nm
            else:
                cd_estimate = 2.0 * distance_transform_edt(1.0 - mask) * pixel_size_nm

            cd_estimate = np.clip(cd_estimate, 10.0, 2000.0)
        except ImportError:
            dist_to_edge = np.ones((Ny, Nx), dtype=np.float64) * 1e6
            cd_estimate = np.full((Ny, Nx), 200.0, dtype=np.float64)

        L = self.cfg.wavelength_nm / np.pi
        thickness = self.cfg.absorber.thickness_nm
        n_abs = self.n_abs
        n_sub = self.n_sub
        sw_angle = self.cfg.absorber.sidewall.sidewall_angle_deg

        with np.errstate(divide="ignore", invalid="ignore"):
            cd_norm = cd_estimate / self.cfg.wavelength_nm

            if polarization == "TE":
                dn_eff = (
                    0.35 * (np.real(n_abs) - np.real(n_sub))
                    * np.exp(-cd_norm * 0.8)
                    * (1.0 + 0.02 * sw_angle)
                )
                amp_factor = 1.0 - 0.25 * np.exp(-cd_norm * 0.6) * (
                    thickness / 70.0
                ) * (1.0 + 0.01 * abs(sw_angle))
            else:
                dn_eff = (
                    0.55 * (np.real(n_abs) - np.real(n_sub))
                    * np.exp(-cd_norm * 0.7)
                    * (1.0 - 0.03 * sw_angle)
                )
                amp_factor = 1.0 - 0.35 * np.exp(-cd_norm * 0.5) * (
                    thickness / 70.0
                ) * (1.0 + 0.015 * abs(sw_angle))

        dn_eff = np.clip(dn_eff, -0.3, 0.3)
        amp_factor = np.clip(amp_factor, 0.5, 1.1)

        phase_correction = self.k0 * dn_eff * thickness

        decay_sigma = L * 0.8
        edge_modulation = np.exp(-(dist_to_edge ** 2) / (2.0 * decay_sigma ** 2 + 1e-30))

        cor_amp = 1.0 + (amp_factor - 1.0) * edge_modulation
        cor_phase = phase_correction * edge_modulation

        return cor_amp, cor_phase

    def _angular_spectrum_propagate(
        self,
        field: np.ndarray,
        pixel_size_nm: float,
        z_nm: float,
    ) -> np.ndarray:
        """角谱法传播近场"""
        Ny, Nx = field.shape
        k0 = self.k0
        n = self.n_sup

        ky = 2.0 * np.pi * np.fft.fftfreq(Ny, pixel_size_nm)
        kx = 2.0 * np.pi * np.fft.fftfreq(Nx, pixel_size_nm)
        KY, KX = np.meshgrid(ky, kx, indexing="ij")

        KZ = np.lib.scimath.sqrt((k0 * n) ** 2 - KX ** 2 - KY ** 2)
        KZ = np.where(np.imag(KZ) < 0, -KZ, KZ)

        propagator = np.exp(1j * KZ * z_nm)
        F = np.fft.fft2(field)
        F_prop = F * propagator
        return np.fft.ifft2(F_prop)


# =============================================================================
# RCWA 耦合 Hopkins 修正模块
# =============================================================================
class RCWAHopkinsCoupler:
    """
    RCWA ↔ Hopkins 耦合修正器

    两种修正模式：
    1. effective_transmission: 计算等效透射率直接替换薄掩模透射率
    2. tcc_modulation: 计算 TCC 调制因子矩阵修正 Hopkins 核
    """

    def __init__(self, config: Mask3DConfig):
        self.cfg = config
        self.sbem = SimplifiedBEMScattering(config)
        self.rcwa_cfg = RCWAConfig(
            n_orders=config.rcwa_n_orders,
            n_substrate=config.get_substrate_n(),
            n_superstrate=config.get_superstrate_n(),
            n_grating_line=config.absorber.get_refractive_index(config.wavelength_nm),
            grating_thickness_nm=self._effective_grating_thickness(),
        )
        self._rcwa_solver_1d = RCWASolver1D(self.rcwa_cfg)
        self._rcwa_solver_2d = RCWASolver2D(self.rcwa_cfg)
        self._calibration_cache: Dict[Any, Tuple] = {}

    def _effective_grating_thickness(self) -> float:
        """根据侧壁参数计算等效 RCWA 光栅厚度"""
        sw = self.cfg.absorber.sidewall
        t_nom = self.cfg.absorber.thickness_nm
        if sw.profile_type != SidewallProfile.RECTANGULAR:
            return t_nom * 0.9
        return t_nom

    def compute_effective_transmission(
        self,
        mask_pattern: np.ndarray,
        pixel_size_nm: float,
        polarization: Polarization = Polarization.UNPOLARIZED,
        use_rcwa_calibration: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        计算等效透射率 t_eff(x, y)，替换 Hopkins 模型中的理想透射率
        """
        sbem_result = self.sbem.compute_near_field_correction_2d(
            mask_pattern, pixel_size_nm, polarization
        )
        t_eff = sbem_result["corrected_field"].astype(np.complex128)
        t_ideal = sbem_result["ideal_field"]

        if use_rcwa_calibration and self.cfg.rcwa_correction_enabled:
            t_eff = self._rcwa_calibrate(
                mask_pattern, pixel_size_nm, t_eff, polarization
            )

        eps = 1e-30
        with np.errstate(divide="ignore", invalid="ignore"):
            amp_map = np.where(
                np.abs(t_ideal) > eps,
                np.abs(t_eff) / np.maximum(np.abs(t_ideal), eps),
                1.0,
            )
            phase_map = np.angle(t_eff) - np.angle(t_ideal)

        return {
            "t_effective": t_eff,
            "t_ideal": t_ideal,
            "amplitude_map": amp_map,
            "phase_map": phase_map,
            "sbem_corrected": sbem_result["corrected_field"],
            "correction_amplitude": sbem_result["correction_amplitude"],
            "correction_phase": sbem_result["correction_phase"],
        }

    def _rcwa_calibrate(
        self,
        mask: np.ndarray,
        pixel_size_nm: float,
        t_sbem: np.ndarray,
        polarization: Polarization,
    ) -> np.ndarray:
        """
        使用 RCWA 在代表性 CD 点处精确计算，校准 S-BEM 结果
        """
        Ny, Nx = mask.shape
        t_out = t_sbem.copy()

        try:
            from scipy.ndimage import distance_transform_edt
            if np.mean(mask) > 0.5:
                cd_field = 2.0 * distance_transform_edt(mask) * pixel_size_nm
            else:
                cd_field = 2.0 * distance_transform_edt(1.0 - mask) * pixel_size_nm
            cd_field = np.clip(cd_field, 20.0, 2000.0)
        except ImportError:
            return t_out

        cd_bins_nm = np.array([40, 60, 80, 100, 140, 200, 300, 500, 1000])
        calib_data = {}
        large_cd_ref_key = None

        for cd_nm in cd_bins_nm:
            period_nm = cd_nm * 2.0
            key = (round(cd_nm), polarization.value)
            if large_cd_ref_key is None or cd_nm > large_cd_ref_key[0]:
                large_cd_ref_key = key
            if key not in self._calibration_cache:
                duty = cd_nm / period_nm
                rcwa_res = self._rcwa_solver_1d.solve_far_field(
                    wavelength_nm=self.cfg.wavelength_nm,
                    period_nm=period_nm,
                    duty_cycle=float(np.clip(duty, 0.05, 0.95)),
                    theta_deg=0.0,
                )
                M = self.rcwa_cfg.n_orders
                idx_0 = M
                t0_te = rcwa_res.get("t_TE", np.zeros(2 * M + 1, dtype=np.complex128))[idx_0]
                t0_tm = rcwa_res.get("t_TM", np.zeros(2 * M + 1, dtype=np.complex128))[idx_0]

                if polarization == Polarization.TE:
                    t0_rcwa = t0_te
                elif polarization == Polarization.TM:
                    t0_rcwa = t0_tm
                else:
                    t0_rcwa = 0.5 * (t0_te + t0_tm)

                t0_ideal = duty * 1.0 + (1.0 - duty) * self._absorber_transmission_ideal()
                calib_factor = t0_rcwa / (t0_ideal + 1e-30)
                self._calibration_cache[key] = (calib_factor,)

            calib_data[cd_nm] = self._calibration_cache[key][0]

        # 相对归一化: 除以最大 CD 的校准因子 → 大 CD 极限 calib → 1.0
        # 物理意义: 只保留 Mask3D 引起的相对偏差，消除绝对透射率整体缩放
        large_cd_ref = calib_data[max(calib_data.keys())]
        if abs(large_cd_ref) > 1e-15:
            for cd_nm in calib_data:
                calib_data[cd_nm] = calib_data[cd_nm] / large_cd_ref

        calib_cds = np.array(sorted(calib_data.keys()))
        if len(calib_cds) < 2:
            return t_out

        cd_lo = calib_cds[:-1, None, None]
        cd_hi = calib_cds[1:, None, None]
        factors_lo = np.array([calib_data[c] for c in calib_cds[:-1]])[:, None, None]
        factors_hi = np.array([calib_data[c] for c in calib_cds[1:]])[:, None, None]

        frac = (cd_field[None, :, :] - cd_lo) / (cd_hi - cd_lo + 1e-30)
        frac = np.clip(frac, 0.0, 1.0)

        bins_idx = np.clip(
            np.searchsorted(calib_cds, cd_field) - 1,
            0, len(calib_cds) - 2,
        )

        for i in range(Ny):
            for j in range(Nx):
                idx = bins_idx[i, j]
                f = frac[idx, i, j]
                flo = factors_lo[idx, 0, 0]
                fhi = factors_hi[idx, 0, 0]
                t_out[i, j] *= (flo * (1.0 - f) + fhi * f)

        return t_out

    def _absorber_transmission_ideal(self) -> complex:
        """吸收层理想薄掩模透射率
        exp(+jωt) 约定下传播项为 exp(-j k z), 因此:
        t_abs = exp(-j k0 (n_abs - n_sub) t)
        n_abs = n' - jk (STANDARD_MATERIALS), 展开后得 exp(-k0 k t) * exp(-j k0 Δn' t)
        """
        t = self.cfg.absorber.thickness_nm
        k0 = 2.0 * np.pi / self.cfg.wavelength_nm
        n_abs = self.cfg.absorber.get_refractive_index(self.cfg.wavelength_nm)
        n_sub = self.cfg.get_substrate_n()
        return np.exp(-1j * k0 * (n_abs - n_sub) * t)

    def compute_tcc_modulation(
        self,
        source_sigma_range: Tuple[float, float] = (0.0, 0.5),
        num_sigma_points: int = 5,
    ) -> Dict[str, np.ndarray]:
        """
        计算 TCC 的 3D 修正调制矩阵 (对角近似)
        """
        sigma_arr = np.linspace(source_sigma_range[0], source_sigma_range[1], num_sigma_points)
        N_pupil = 129
        m3d_te_list = []
        m3d_tm_list = []
        m3d_amp_0 = []
        m3d_phase_0 = []

        for sigma in sigma_arr:
            fx = np.linspace(-1.0, 1.0, N_pupil)
            FY, FX = np.meshgrid(fx, fx, indexing="ij")
            rho = np.sqrt(FX ** 2 + FY ** 2)
            phi = np.arctan2(FY, FX)

            t = self.cfg.absorber.thickness_nm
            wavelength = self.cfg.wavelength_nm

            cos_theta_in = np.sqrt(np.clip(1.0 - (sigma * rho) ** 2, 1e-4, 1.0))

            phase_te = (wavelength / (2 * np.pi)) * (
                0.15 * rho ** 2 * (1.0 + 0.3 * np.cos(2 * phi))
            ) * (t / 70.0)
            phase_tm = (wavelength / (2 * np.pi)) * (
                0.22 * rho ** 2 * (1.0 - 0.3 * np.cos(2 * phi))
            ) * (t / 70.0)

            amp_te = 1.0 - 0.20 * rho ** 2 * (t / 70.0)
            amp_tm = 1.0 - 0.28 * rho ** 2 * (t / 70.0)

            sw = self.cfg.absorber.sidewall.sidewall_angle_deg
            amp_te = amp_te * (1.0 - 0.01 * abs(sw) * rho)
            amp_tm = amp_tm * (1.0 + 0.005 * sw * rho * np.cos(2 * phi))

            amp_te = np.clip(amp_te, 0.2, 1.2)
            amp_tm = np.clip(amp_tm, 0.2, 1.2)

            m_te = amp_te * np.exp(1j * phase_te)
            m_tm = amp_tm * np.exp(1j * phase_tm)

            m3d_te_list.append(m_te)
            m3d_tm_list.append(m_tm)
            m3d_amp_0.append(np.abs(m_te[N_pupil // 2, N_pupil // 2]))
            m3d_phase_0.append(np.angle(m_te[N_pupil // 2, N_pupil // 2]))

        return {
            "sigma_values": sigma_arr,
            "m3d_ctf_te": np.array(m3d_te_list),
            "m3d_ctf_tm": np.array(m3d_tm_list),
            "m3d_0th_amplitude": np.array(m3d_amp_0),
            "m3d_0th_phase": np.array(m3d_phase_0),
        }


# =============================================================================
# Mask3D 成像输入耦合接口
# =============================================================================
@dataclass
class Mask3DCorrectionResult:
    """Mask3D 修正完整结果"""
    t_effective: np.ndarray
    amplitude_correction: np.ndarray
    phase_correction: np.ndarray
    tcc_modulation_factor: Optional[Dict[str, np.ndarray]]
    cd_bias_map: Optional[np.ndarray]
    runtime_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class Mask3DImagingCorrector:
    """
    Mask3D 成像修正统一入口

    使用流程:
        1. corrector = Mask3DImagingCorrector(config)
        2. correction = corrector.correct_mask_for_imaging(mask, pixel_size)
        3. 将 correction.t_effective 替换 Hopkins 模型中的掩模透射率
    """

    def __init__(self, config: Mask3DConfig):
        self.cfg = config
        self.coupler = RCWAHopkinsCoupler(config)

    def correct_mask_for_imaging(
        self,
        mask_pattern: np.ndarray,
        pixel_size_nm: float,
        polarization: Polarization = Polarization.UNPOLARIZED,
        correction_mode: Optional[str] = None,
        compute_cd_bias: bool = True,
    ) -> Mask3DCorrectionResult:
        """
        计算修正后的掩模透射率 / TCC 调制因子

        Args:
            mask_pattern: 0/1 掩模版图 shape=(Ny, Nx)
            pixel_size_nm: 像素大小 (nm)
            polarization: 入射偏振态
            correction_mode: 修正模式, None 时使用 config 设置
            compute_cd_bias: 是否计算 CD 偏差估计图

        Returns:
            Mask3DCorrectionResult
        """
        import time
        t0 = time.time()
        mode = correction_mode or self.cfg.hopkins_correction_mode

        trans_result = self.coupler.compute_effective_transmission(
            mask_pattern, pixel_size_nm, polarization
        )

        tcc_mod = None
        if mode == "tcc_modulation":
            tcc_mod = self.coupler.compute_tcc_modulation()

        cd_bias_map = None
        if compute_cd_bias:
            cd_bias_map = self._estimate_cd_bias_map(
                mask_pattern, pixel_size_nm, trans_result
            )

        runtime = time.time() - t0

        return Mask3DCorrectionResult(
            t_effective=trans_result["t_effective"],
            amplitude_correction=trans_result["amplitude_map"],
            phase_correction=trans_result["phase_map"],
            tcc_modulation_factor=tcc_mod,
            cd_bias_map=cd_bias_map,
            runtime_sec=runtime,
            extra={
                "t_ideal": trans_result["t_ideal"],
                "sbem_corrected": trans_result["sbem_corrected"],
                "correction_mode": mode,
            },
        )

    @staticmethod
    def _estimate_cd_bias_map(
        mask: np.ndarray,
        pixel_size_nm: float,
        trans_result: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        估计 Mask3D 效应导致的局部 CD 偏差图 (单位 nm)
        """
        amp_cor = trans_result["amplitude_map"]
        phase_cor = trans_result["phase_map"]

        try:
            from scipy.ndimage import distance_transform_edt, gaussian_filter
        except ImportError:
            return None

        gx = np.zeros_like(mask, dtype=np.float64)
        gy = np.zeros_like(mask, dtype=np.float64)
        gx[:, 1:] = np.diff(mask.astype(np.float64), axis=1)
        gy[1:, :] = np.diff(mask.astype(np.float64), axis=0)
        edge_strength = np.abs(gx) + np.abs(gy)
        edge_mask = edge_strength > 0

        if np.mean(mask) > 0.5:
            cd_field = 2.0 * distance_transform_edt(mask) * pixel_size_nm
        else:
            cd_field = 2.0 * distance_transform_edt(1.0 - mask) * pixel_size_nm
        cd_field = np.clip(cd_field, 20.0, 2000.0)

        log_amp = np.log(np.clip(amp_cor, 0.01, 100.0))

        k_amp = -40.0
        k_phase = -25.0

        cd_bias_per_pix = (
            k_amp * log_amp * cd_field / 100.0
            + k_phase * phase_cor * cd_field / (2.0 * np.pi + 1e-30)
        )

        dist_to_edge = distance_transform_edt(1.0 - edge_mask) * pixel_size_nm
        decay_L = pixel_size_nm * 8.0
        edge_weight = np.exp(-dist_to_edge ** 2 / (2.0 * decay_L ** 2 + 1e-30))

        cd_bias_map = cd_bias_per_pix * edge_weight
        cd_bias_map = gaussian_filter(cd_bias_map, sigma=0.8)

        return cd_bias_map

    # ------------------------------------------------------------------
    # 便捷: 快速 CD 偏差扫描 (用于研究高 NA 下的 CD 偏差曲线)
    # ------------------------------------------------------------------
    def scan_cd_bias_vs_cd(
        self,
        cd_range_nm: Tuple[float, float] = (40.0, 500.0),
        num_points: int = 30,
        pitch_to_cd_ratio: float = 2.0,
        na: float = 1.35,
    ) -> Dict[str, np.ndarray]:
        """
        扫描 CD 偏差 vs 标称 CD 曲线

        用于快速研究 Mask3D 效应在不同 CD、不同 NA 下的影响趋势。
        采用 1D line/space RCWA + S-BEM 混合近似模型。
        """
        cd_arr = np.linspace(cd_range_nm[0], cd_range_nm[1], num_points)
        cd_bias_arr = np.zeros_like(cd_arr)
        cd_bias_te = np.zeros_like(cd_arr)
        cd_bias_tm = np.zeros_like(cd_arr)

        wavelength = self.cfg.wavelength_nm
        k0 = 2.0 * np.pi / wavelength
        n_sub = self.cfg.get_substrate_n()
        n_abs = self.cfg.absorber.get_refractive_index(wavelength)
        thickness = self.cfg.absorber.thickness_nm
        sw_angle = self.cfg.absorber.sidewall.sidewall_angle_deg

        for idx, cd_nm in enumerate(cd_arr):
            pitch_nm = cd_nm * pitch_to_cd_ratio
            duty = cd_nm / pitch_nm

            # 波导模型: 等效折射率依赖线宽
            cd_norm = cd_nm / wavelength
            with np.errstate(divide="ignore", invalid="ignore"):
                dn_te = 0.35 * (np.real(n_abs) - np.real(n_sub)) * np.exp(-cd_norm * 0.8)
                dn_tm = 0.55 * (np.real(n_abs) - np.real(n_sub)) * np.exp(-cd_norm * 0.7)
                loss_te = 0.25 * np.exp(-cd_norm * 0.6) * (thickness / 70.0)
                loss_tm = 0.35 * np.exp(-cd_norm * 0.5) * (thickness / 70.0)

            # 侧壁贡献
            sw_contrib_te = 0.02 * sw_angle * dn_te
            sw_contrib_tm = -0.03 * sw_angle * dn_tm

            # 高 NA 贡献: sin^2(theta_max) 项
            theta_max = np.arcsin(min(1.0, na / np.real(n_sub)))
            na_contrib = np.sin(theta_max) ** 2

            # 相位偏移 → 等效 CD 偏移
            phase_te = k0 * (dn_te + sw_contrib_te) * thickness
            phase_tm = k0 * (dn_tm + sw_contrib_tm) * thickness

            # 透过率对比度变化 → CD 偏差
            k_phase_to_cd = -cd_nm / (2.0 * np.pi) * 0.35 * na_contrib
            k_loss_to_cd = -cd_nm * 0.45

            bias_te = k_phase_to_cd * phase_te + k_loss_to_cd * loss_te
            bias_tm = k_phase_to_cd * phase_tm + k_loss_to_cd * loss_tm

            cd_bias_te[idx] = bias_te
            cd_bias_tm[idx] = bias_tm
            cd_bias_arr[idx] = 0.5 * (bias_te + bias_tm)

        return {
            "cd_nominal_nm": cd_arr,
            "cd_bias_nm": cd_bias_arr,
            "cd_bias_te_nm": cd_bias_te,
            "cd_bias_tm_nm": cd_bias_tm,
            "na": na,
            "wavelength_nm": wavelength,
        }


# =============================================================================
# 顶层便捷函数
# =============================================================================
def create_default_mask3d_config(
    mask_type: Union[str, MaskType] = MaskType.BINARY_COG,
    wavelength_nm: float = 193.0,
    sidewall_angle_deg: float = 3.0,
    absorber_thickness_nm: float = 70.0,
) -> Mask3DConfig:
    """
    创建常用配置的快捷函数

    Args:
        mask_type: 掩模类型
        wavelength_nm: 工作波长
        sidewall_angle_deg: 侧壁角 (典型正梯形取 +2~+5 度)
        absorber_thickness_nm: 吸收层厚度

    Returns:
        Mask3DConfig
    """
    sw = SidewallParams(
        profile_type=SidewallProfile.TRAPEZOIDAL,
        sidewall_angle_deg=sidewall_angle_deg,
        top_cd_bias_nm=2.0 * absorber_thickness_nm * np.tan(np.deg2rad(sidewall_angle_deg)),
        bottom_cd_bias_nm=0.0,
    )
    absorber = AbsorberLayer(
        material_name="cr",
        thickness_nm=absorber_thickness_nm,
        sidewall=sw,
    )
    if mask_type == MaskType.EUV_REFLECTIVE:
        absorber.material_name = "mo"
        absorber.thickness_nm = absorber_thickness_nm
    elif mask_type == MaskType.ATTENUATED_PSM:
        absorber.material_name = "ta2o5"
        absorber.custom_n = 2.35 - 0.05j
    return Mask3DConfig(
        mask_type=mask_type,
        wavelength_nm=wavelength_nm,
        absorber=absorber,
    )


def apply_mask3d_correction(
    mask_pattern: np.ndarray,
    pixel_size_nm: float,
    config: Optional[Mask3DConfig] = None,
    polarization: Union[str, Polarization] = Polarization.UNPOLARIZED,
) -> Mask3DCorrectionResult:
    """
    顶层便捷接口: 应用 Mask3D 修正

    Args:
        mask_pattern: 0/1 掩模版图
        pixel_size_nm: 像素大小
        config: Mask3DConfig, None 时使用默认 ArF 配置
        polarization: 偏振态

    Returns:
        Mask3DCorrectionResult, 包含修正后的等效透射率等
    """
    if config is None:
        config = create_default_mask3d_config()
    if isinstance(polarization, str):
        polarization = Polarization(polarization)
    corrector = Mask3DImagingCorrector(config)
    return corrector.correct_mask_for_imaging(
        mask_pattern, pixel_size_nm, polarization
    )


__all__ = [
    "MaskType",
    "SidewallProfile",
    "RoughnessModel",
    "SidewallParams",
    "RoughnessParams",
    "AbsorberLayer",
    "Mask3DConfig",
    "MaskTopography",
    "SimplifiedBEMScattering",
    "RCWAHopkinsCoupler",
    "Mask3DCorrectionResult",
    "Mask3DImagingCorrector",
    "create_default_mask3d_config",
    "apply_mask3d_correction",
]
