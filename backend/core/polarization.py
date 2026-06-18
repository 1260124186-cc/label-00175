# -*- coding: utf-8 -*-
"""
偏振与薄膜效应建模模块

实现光刻光学系统中的偏振效应与薄膜涂层调制，包括：
1. Jones矩阵形式的偏振传递模型
2. 薄膜堆栈（Thin Film Stack）的相位/振幅调制计算
3. 扩展光瞳函数计算，集成偏振与薄膜效应
4. 高NA浸没式与EUV反射掩模的矢量成像模型

参考:
    - A. K. Pouret, "Polarization effects in optical lithography," Proc. SPIE, 2002
    - Y. Borodovsky, "Thin film optimization for EUV masks," Proc. SPIE, 2008
    - Flagello et al., "Theory of high-NA imaging in immersion lithography,"
      J. Microlith. Microfab. Microsyst. 1, 41 (2002)
"""

import numpy as np
from numba import jit, prange
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum


# =============================================================================
# 枚举与基础类型
# =============================================================================
class PolarizationComponent(str, Enum):
    """偏振分量枚举"""
    EX = "Ex"          # x方向电场分量
    EY = "Ey"          # y方向电场分量
    EZ = "Ez"          # z方向电场分量
    S_POL = "s"        # s偏振（TE）
    P_POL = "p"        # p偏振（TM）


class CoatingMaterial(str, Enum):
    """常见涂层材料折射率（特定波长）"""
    ARF_MULTILAYER = "arf_multilayer"       # ArF 抗反射涂层
    EUV_MULTILAYER = "euv_multilayer"       # EUV 多层反射膜 (Mo/Si)
    TA2O5 = "ta2o5"                         # 五氧化二钽
    SIO2 = "sio2"                           # 二氧化硅
    HF = "hf"                               # 氟化氢
    MGF2 = "mgf2"                           # 氟化镁
    CR = "cr"                               # 铬吸收层
    MO = "mo"                               # 钼（EUV）
    SI = "si"                               # 硅（EUV）
    RU = "ru"                               # 钌（EUV覆盖层）
    CUSTOM = "custom"                       # 自定义材料


@dataclass
class MaterialDispersion:
    """材料色散关系：复折射率与波长"""
    name: str
    n_data: Dict[float, complex] = field(default_factory=dict)

    def get_n(self, wavelength_nm: float) -> complex:
        """获取指定波长下的复折射率"""
        if wavelength_nm in self.n_data:
            return self.n_data[wavelength_nm]
        wavelengths = sorted(self.n_data.keys())
        if not wavelengths:
            return 1.0 + 0.0j
        if wavelength_nm <= wavelengths[0]:
            return self.n_data[wavelengths[0]]
        if wavelength_nm >= wavelengths[-1]:
            return self.n_data[wavelengths[-1]]
        for i in range(len(wavelengths) - 1):
            wl1, wl2 = wavelengths[i], wavelengths[i + 1]
            if wl1 <= wavelength_nm <= wl2:
                n1, n2 = self.n_data[wl1], self.n_data[wl2]
                t = (wavelength_nm - wl1) / (wl2 - wl1)
                return n1 * (1 - t) + n2 * t
        return self.n_data[wavelengths[-1]]


STANDARD_MATERIALS: Dict[str, MaterialDispersion] = {
    "sio2": MaterialDispersion(
        "SiO2 (Silica)",
        {193.0: 1.567 + 0.0j, 248.0: 1.515 + 0.0j, 13.5: 0.999 - 0.001j}
    ),
    "cr": MaterialDispersion(
        "Cr (Chromium)",
        {193.0: 3.28 - 4.32j, 248.0: 3.05 - 3.95j, 13.5: 0.96 - 9.82j}
    ),
    "ta2o5": MaterialDispersion(
        "Ta2O5",
        {193.0: 2.15 + 0.001j, 248.0: 2.10 + 0.0005j}
    ),
    "mgf2": MaterialDispersion(
        "MgF2 (Magnesium Fluoride)",
        {193.0: 1.42 + 0.0j, 248.0: 1.39 + 0.0j}
    ),
    "mo": MaterialDispersion(
        "Mo (Molybdenum)",
        {13.5: 0.92 - 0.004j}
    ),
    "si": MaterialDispersion(
        "Si (Silicon)",
        {13.5: 0.99 - 0.006j}
    ),
    "ru": MaterialDispersion(
        "Ru (Ruthenium)",
        {13.5: 0.95 - 0.003j}
    ),
    "water": MaterialDispersion(
        "H2O (Water, immersion)",
        {193.0: 1.437 + 0.0j}
    ),
    "air": MaterialDispersion(
        "Air",
        {193.0: 1.0002 + 0.0j, 13.5: 1.0 + 0.0j}
    ),
}


# =============================================================================
# Jones 矩阵光学元件
# =============================================================================
@dataclass
class JonesMatrix:
    """2x2 Jones矩阵，描述偏振态变换"""

    matrix: np.ndarray

    def __post_init__(self):
        if self.matrix.shape != (2, 2):
            raise ValueError("Jones矩阵必须是2x2矩阵")
        if self.matrix.dtype not in [np.complex64, np.complex128]:
            self.matrix = self.matrix.astype(np.complex128)

    @classmethod
    def identity(cls) -> "JonesMatrix":
        """单位矩阵（无偏振变化）"""
        return cls(np.eye(2, dtype=np.complex128))

    @classmethod
    def polarizer(cls, angle_deg: float = 0.0) -> "JonesMatrix":
        """理想偏振片，沿指定角度"""
        theta = np.deg2rad(angle_deg)
        cos2 = np.cos(theta) ** 2
        sin2 = np.sin(theta) ** 2
        sincos = np.sin(theta) * np.cos(theta)
        mat = np.array([
            [cos2, sincos],
            [sincos, sin2]
        ], dtype=np.complex128)
        return cls(mat)

    @classmethod
    def waveplate(cls, phase_shift_rad: float, angle_deg: float = 0.0) -> "JonesMatrix":
        """波片（相位延迟器）"""
        theta = np.deg2rad(angle_deg)
        cos2 = np.cos(2 * theta)
        sin2 = np.sin(2 * theta)
        phi = phase_shift_rad / 2.0
        mat = np.array([
            [np.cos(phi) - 1j * cos2 * np.sin(phi), -1j * sin2 * np.sin(phi)],
            [-1j * sin2 * np.sin(phi), np.cos(phi) + 1j * cos2 * np.sin(phi)]
        ], dtype=np.complex128)
        return cls(mat)

    @classmethod
    def quarter_waveplate(cls, angle_deg: float = 0.0) -> "JonesMatrix":
        """1/4波片"""
        return cls.waveplate(np.pi / 2, angle_deg)

    @classmethod
    def half_waveplate(cls, angle_deg: float = 0.0) -> "JonesMatrix":
        """1/2波片"""
        return cls.waveplate(np.pi, angle_deg)

    @classmethod
    def rotation(cls, angle_deg: float) -> "JonesMatrix":
        """坐标旋转矩阵"""
        theta = np.deg2rad(angle_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        mat = np.array([
            [c, -s],
            [s, c]
        ], dtype=np.complex128)
        return cls(mat)

    @classmethod
    def diattenuator(cls, tx: complex, ty: complex, angle_deg: float = 0.0) -> "JonesMatrix":
        """
        二向色衰减器（偏振相关传输/反射）
        用于模拟高NA系统中的偏振依赖性
        """
        mat = np.array([
            [tx, 0],
            [0, ty]
        ], dtype=np.complex128)
        if angle_deg != 0.0:
            rot = cls.rotation(angle_deg)
            rot_inv = cls.rotation(-angle_deg)
            mat = rot.matrix @ mat @ rot_inv.matrix
        return cls(mat)

    def __matmul__(self, other: "JonesMatrix") -> "JonesMatrix":
        """矩阵串联（注意：右侧先作用）"""
        return JonesMatrix(self.matrix @ other.matrix)

    def apply(self, e_x: complex, e_y: complex) -> Tuple[complex, complex]:
        """应用Jones矩阵到偏振态 (Ex, Ey)"""
        ein = np.array([e_x, e_y], dtype=np.complex128)
        eout = self.matrix @ ein
        return eout[0], eout[1]

    def to_hermitian(self) -> np.ndarray:
        """计算Hermitian矩阵 J† J，用于光强计算"""
        return self.matrix.conj().T @ self.matrix

    def intensity_transmission(self, ein: np.ndarray) -> float:
        """计算通过后的光强"""
        eout = self.matrix @ ein
        return float(np.real(eout.conj() @ eout))


# =============================================================================
# Jones 矢量（偏振态）
# =============================================================================
@dataclass
class JonesVector:
    """Jones矢量，描述偏振态"""
    e_x: complex
    e_y: complex

    @classmethod
    def linear_polarization(cls, angle_deg: float = 0.0) -> "JonesVector":
        """线偏振态"""
        theta = np.deg2rad(angle_deg)
        return cls(np.cos(theta), np.sin(theta))

    @classmethod
    def left_circular(cls) -> "JonesVector":
        """左旋圆偏振"""
        return cls(1.0 / np.sqrt(2), -1j / np.sqrt(2))

    @classmethod
    def right_circular(cls) -> "JonesVector":
        """右旋圆偏振"""
        return cls(1.0 / np.sqrt(2), 1j / np.sqrt(2))

    @classmethod
    def elliptical(cls, amp_ratio: float, phase_deg: float,
                   orientation_deg: float = 0.0) -> "JonesVector":
        """椭圆偏振"""
        amp1 = 1.0
        amp2 = amp_ratio
        phase = np.deg2rad(phase_deg)
        vec = cls(amp1, amp2 * np.exp(1j * phase))
        if orientation_deg != 0.0:
            rot = JonesMatrix.rotation(orientation_deg)
            ex, ey = rot.apply(vec.e_x, vec.e_y)
            return cls(ex, ey)
        return vec

    @property
    def intensity(self) -> float:
        """光强"""
        return float(np.real(np.conj(self.e_x) * self.e_x +
                              np.conj(self.e_y) * self.e_y))

    def normalize(self) -> "JonesVector":
        """归一化光强为1"""
        I = self.intensity
        if I > 0:
            return JonesVector(self.e_x / np.sqrt(I), self.e_y / np.sqrt(I))
        return self

    def apply_matrix(self, jm: JonesMatrix) -> "JonesVector":
        """应用Jones矩阵"""
        ex, ey = jm.apply(self.e_x, self.e_y)
        return JonesVector(ex, ey)

    def as_array(self) -> np.ndarray:
        """转换为2元素复数数组"""
        return np.array([self.e_x, self.e_y], dtype=np.complex128)


# =============================================================================
# 薄膜堆栈与多层膜计算
# =============================================================================
@dataclass
class ThinFilmLayer:
    """薄膜堆栈中的单层"""
    thickness_nm: float
    material: Union[str, MaterialDispersion]
    name: str = ""

    def get_n(self, wavelength_nm: float) -> complex:
        """获取指定波长下的复折射率"""
        if isinstance(self.material, str):
            if self.material in STANDARD_MATERIALS:
                return STANDARD_MATERIALS[self.material].get_n(wavelength_nm)
            return 1.0 + 0.0j
        return self.material.get_n(wavelength_nm)


@dataclass
class ThinFilmStack:
    """
    薄膜堆栈

    用于模拟:
    - 掩模表面的抗反射涂层 (AR coating)
    - EUV 掩模的多层反射膜 (Mo/Si)
    - 透镜表面的增透膜
    """

    layers: List[ThinFilmLayer] = field(default_factory=list)
    n_superstrate: Union[complex, str] = 1.0 + 0.0j
    n_substrate: Union[complex, str] = 1.56 + 0.0j

    @classmethod
    def arf_antireflective(cls, wavelength_nm: float = 193.0) -> "ThinFilmStack":
        """
        创建ArF典型抗反射涂层堆栈
        典型: Cr absorber -> AR coating -> glass
        """
        return cls(
            layers=[
                ThinFilmLayer(10.0, "ta2o5", "Ta2O5_HighIndex"),
                ThinFilmLayer(35.0, "mgf2", "MgF2_LowIndex"),
            ],
            n_superstrate="water",
            n_substrate="sio2"
        )

    @classmethod
    def euv_multilayer(cls, num_pairs: int = 40, wavelength_nm: float = 13.5) -> "ThinFilmStack":
        """
        创建EUV典型多层反射膜（Mo/Si周期结构）

        标准 Mo/Si 多层膜在 13.5nm 波长下:
        - 周期厚度 ~ 6.7-7.0 nm
        - 厚度比 Γ = d_Mo / d_period ~ 0.4-0.45
        """
        period_nm = 6.78
        gamma = 0.42
        d_mo = period_nm * gamma
        d_si = period_nm * (1 - gamma)

        layers = []
        for i in range(num_pairs):
            layers.append(ThinFilmLayer(d_mo, "mo", f"Mo_{i}"))
            layers.append(ThinFilmLayer(d_si, "si", f"Si_{i}"))
        layers.append(ThinFilmLayer(2.0, "ru", "Ru_Cap"))

        return cls(
            layers=layers,
            n_superstrate="air",
            n_substrate="sio2"
        )

    def _resolve_n(self, n_val: Union[complex, str],
                    wavelength_nm: float) -> complex:
        """解析折射率值"""
        if isinstance(n_val, str):
            if n_val in STANDARD_MATERIALS:
                return STANDARD_MATERIALS[n_val].get_n(wavelength_nm)
            return 1.0 + 0.0j
        return complex(n_val)

    def _characteristic_matrix(self, n: complex, d: float,
                               wavelength_nm: float, theta_rad: float) -> np.ndarray:
        """
        计算单层薄膜的特征矩阵

        对 s 偏振 (TE):
            M = [[cos(δ), -j sin(δ)/η], [-j η sin(δ), cos(δ)]]
            其中 η = n cos(θ)

        对 p 偏振 (TM):
            η = n / cos(θ)

        其中 δ = (2π/λ) * n * d * cos(θ_t)
              θ_t 由 Snell 定律: n0 sin(θ0) = n1 sin(θ1)
        """
        k0 = 2.0 * np.pi / wavelength_nm
        n0 = self._resolve_n(self.n_superstrate, wavelength_nm)

        sin_theta_t = n0.real * np.sin(theta_rad) / n.real if n.real != 0 else 0.0
        sin_theta_t = np.clip(abs(sin_theta_t), 0, 1) * np.sign(sin_theta_t)
        cos_theta_t = np.sqrt(1.0 - sin_theta_t ** 2 + 0j)
        if np.imag(cos_theta_t) < 0:
            cos_theta_t = -cos_theta_t

        delta = k0 * n * d * cos_theta_t
        eta_s = n * cos_theta_t
        eta_p = n / cos_theta_t if cos_theta_t != 0 else 1e10 + 0j

        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)

        Ms = np.array([
            [cos_delta, -1j * sin_delta / eta_s],
            [-1j * eta_s * sin_delta, cos_delta]
        ], dtype=np.complex128)

        Mp = np.array([
            [cos_delta, -1j * sin_delta / eta_p],
            [-1j * eta_p * sin_delta, cos_delta]
        ], dtype=np.complex128)

        return Ms, Mp

    def compute_reflection_transmission(
        self,
        wavelength_nm: float,
        theta_rad: float = 0.0,
        polarization: str = "unpolarized",
    ) -> Dict[str, complex]:
        """
        计算薄膜堆栈的反射/透射系数（振幅形式）

        Args:
            wavelength_nm: 波长 (nm)
            theta_rad: 入射角 (弧度)
            polarization: 's', 'p', 'unpolarized'

        Returns:
            {'rs', 'rp', 'ts', 'tp', 'Rs', 'Rp', 'Ts', 'Tp'}
        """
        n0 = self._resolve_n(self.n_superstrate, wavelength_nm)
        n_sub = self._resolve_n(self.n_substrate, wavelength_nm)

        sin_theta_sub = n0.real * np.sin(theta_rad) / n_sub.real if n_sub.real != 0 else 0.0
        sin_theta_sub = np.clip(abs(sin_theta_sub), 0, 1) * np.sign(sin_theta_sub)
        cos_theta_sub = np.sqrt(1.0 - sin_theta_sub ** 2 + 0j)
        if np.imag(cos_theta_sub) < 0:
            cos_theta_sub = -cos_theta_sub

        eta0_s = n0 * np.cos(theta_rad) if n0.real > 0 else 0.0
        eta0_p = n0 / np.cos(theta_rad) if np.cos(theta_rad) != 0 else 1e10 + 0j
        eta_sub_s = n_sub * cos_theta_sub
        eta_sub_p = n_sub / cos_theta_sub if cos_theta_sub != 0 else 1e10 + 0j

        M_total_s = np.eye(2, dtype=np.complex128)
        M_total_p = np.eye(2, dtype=np.complex128)

        for layer in self.layers:
            n_layer = layer.get_n(wavelength_nm)
            Ms, Mp = self._characteristic_matrix(
                n_layer, layer.thickness_nm, wavelength_nm, theta_rad
            )
            M_total_s = M_total_s @ Ms
            M_total_p = M_total_p @ Mp

        M11_s, M12_s = M_total_s[0, 0], M_total_s[0, 1]
        M21_s, M22_s = M_total_s[1, 0], M_total_s[1, 1]
        M11_p, M12_p = M_total_p[0, 0], M_total_p[0, 1]
        M21_p, M22_p = M_total_p[1, 0], M_total_p[1, 1]

        rs = (M11_s * eta0_s + M12_s * eta0_s * eta_sub_s - M21_s - M22_s * eta_sub_s) / \
             (M11_s * eta0_s + M12_s * eta0_s * eta_sub_s + M21_s + M22_s * eta_sub_s)
        ts = 2.0 * eta0_s / \
             (M11_s * eta0_s + M12_s * eta0_s * eta_sub_s + M21_s + M22_s * eta_sub_s)

        rp = (M11_p * eta0_p + M12_p * eta0_p * eta_sub_p - M21_p - M22_p * eta_sub_p) / \
             (M11_p * eta0_p + M12_p * eta0_p * eta_sub_p + M21_p + M22_p * eta_sub_p)
        tp = 2.0 * eta0_p / \
             (M11_p * eta0_p + M12_p * eta0_p * eta_sub_p + M21_p + M22_p * eta_sub_p)

        Rs = float(np.abs(rs) ** 2)
        Rp = float(np.abs(rp) ** 2)
        Ts = float(np.abs(ts) ** 2 * np.real(eta_sub_s / eta0_s))
        Tp = float(np.abs(tp) ** 2 * np.real(eta_sub_p / eta0_p))

        if polarization == "s":
            return {"r": rs, "t": ts, "R": Rs, "T": Ts}
        elif polarization == "p":
            return {"r": rp, "t": tp, "R": Rp, "T": Tp}
        else:
            return {
                "rs": rs, "rp": rp, "ts": ts, "tp": tp,
                "Rs": Rs, "Rp": Rp, "Ts": Ts, "Tp": Tp,
                "R_unpol": 0.5 * (Rs + Rp),
                "T_unpol": 0.5 * (Ts + Tp)
            }

    def compute_spectrum(
        self,
        wavelengths_nm: np.ndarray,
        theta_rad: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        计算宽带光谱响应

        Args:
            wavelengths_nm: 波长数组 (nm)
            theta_rad: 入射角 (弧度)

        Returns:
            {'wavelengths', 'Rs', 'Rp', 'Ts', 'Tp'}
        """
        n_wl = len(wavelengths_nm)
        Rs = np.zeros(n_wl, dtype=np.float64)
        Rp = np.zeros(n_wl, dtype=np.float64)
        Ts = np.zeros(n_wl, dtype=np.float64)
        Tp = np.zeros(n_wl, dtype=np.float64)

        for i, wl in enumerate(wavelengths_nm):
            result = self.compute_reflection_transmission(wl, theta_rad)
            Rs[i] = result["Rs"]
            Rp[i] = result["Rp"]
            Ts[i] = result["Ts"]
            Tp[i] = result["Tp"]

        return {
            "wavelengths": wavelengths_nm,
            "Rs": Rs, "Rp": Rp, "Ts": Ts, "Tp": Tp,
            "R_unpol": 0.5 * (Rs + Rp),
            "T_unpol": 0.5 * (Ts + Tp)
        }

    def compute_angular_response(
        self,
        wavelength_nm: float,
        thetas_rad: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        计算角向响应

        Args:
            wavelength_nm: 波长 (nm)
            thetas_rad: 入射角数组 (弧度)

        Returns:
            {'thetas', 'Rs', 'Rp', 'Ts', 'Tp'}
        """
        n_theta = len(thetas_rad)
        Rs = np.zeros(n_theta, dtype=np.float64)
        Rp = np.zeros(n_theta, dtype=np.float64)
        Ts = np.zeros(n_theta, dtype=np.float64)
        Tp = np.zeros(n_theta, dtype=np.float64)

        for i, theta in enumerate(thetas_rad):
            result = self.compute_reflection_transmission(wavelength_nm, theta)
            Rs[i] = result["Rs"]
            Rp[i] = result["Rp"]
            Ts[i] = result["Ts"]
            Tp[i] = result["Tp"]

        return {
            "thetas": thetas_rad,
            "Rs": Rs, "Rp": Rp, "Ts": Ts, "Tp": Tp,
            "R_unpol": 0.5 * (Rs + Rp),
            "T_unpol": 0.5 * (Ts + Tp)
        }


# =============================================================================
# 矢量光瞳函数（高NA浸没式与EUV反射掩模）
# =============================================================================
@dataclass
class VectorPupil:
    """
    矢量光瞳函数

    对于高NA系统，光瞳函数需要考虑:
    1. 偏振态在光瞳平面上的变化（s/p分解）
    2. 薄膜涂层引起的振幅和相位调制（随入射角变化）
    3. 电场三个分量的独立贡献 (Ex, Ey, Ez)
    4. 反射掩模的斜入射效应（EUV）
    """

    wavelength_nm: float
    na: float
    n_immersion: complex = 1.0 + 0.0j
    grid_size: Tuple[int, int] = (256, 256)
    pixel_size_nm: float = 1.0
    mask_stack: Optional[ThinFilmStack] = None
    incident_polarization: JonesVector = field(
        default_factory=lambda: JonesVector.linear_polarization(0.0)
    )

    def __post_init__(self):
        if self.mask_stack is not None:
            n_super = self.mask_stack._resolve_n(
                self.mask_stack.n_superstrate, self.wavelength_nm
            )
            if np.real(n_super) > 0 and not np.isclose(np.real(n_super), np.real(self.n_immersion), rtol=1e-3):
                self.n_immersion = n_super
        self._build_coordinates()

    def _build_coordinates(self):
        """构建光瞳坐标系统

        坐标定义（与计算物理一致）：
        - 像方空间频率 fx, fy (1/nm)，由 pixel_size_nm 和 grid_size 决定
        - FX = λ·fx, FY = λ·fy  (无量纲，λ 归一化空间频率)
        - ρ = √(FX² + FY²)
        - 截止频率 f_c = NA/λ → ρ_c = λ·f_c = NA (与介质无关！因为 NA 是系统参数)
        - 介质 n 中：sin(θ) = λ·f / n = ρ / n  (由 kx = k·sinθ，k = 2πn/λ)
        - 因此 sin(θ_max) = NA / n (NA 定义)
        """
        ny, nx = self.grid_size
        self.k0 = 2.0 * np.pi / self.wavelength_nm

        fx = np.fft.fftfreq(nx, self.pixel_size_nm) * self.wavelength_nm
        fy = np.fft.fftfreq(ny, self.pixel_size_nm) * self.wavelength_nm
        self.FX, self.FY = np.meshgrid(fx, fy)

        self.rho = np.sqrt(self.FX ** 2 + self.FY ** 2)
        self.pupil_mask = self.rho <= (self.na + 1e-12)
        self.phi_pupil = np.arctan2(self.FY, self.FX)

        self.sin_theta = self.rho / abs(self.n_immersion)
        self.sin_theta = np.clip(self.sin_theta, 0, 1)
        self.cos_theta = np.sqrt(np.maximum(1.0 - self.sin_theta ** 2, 0.0))

    def s_polarization_basis(self) -> Dict[str, np.ndarray]:
        """
        s-偏振（TE）单位电场矢量在光瞳处的分布
        s-偏振: E 垂直于入射面 (k, z)，沿方位角方向
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_phi = np.cos(self.phi_pupil)
            sin_phi = np.sin(self.phi_pupil)
            Es_x = -sin_phi
            Es_y = cos_phi
            Es_z = np.zeros_like(self.rho)
        return {"Ex": Es_x, "Ey": Es_y, "Ez": Es_z}

    def p_polarization_basis(self) -> Dict[str, np.ndarray]:
        """
        p-偏振（TM）单位电场矢量在光瞳处的分布
        p-偏振: E 在入射面 (k, z) 内
        """
        cos_theta = self.cos_theta
        cos_phi = np.cos(self.phi_pupil)
        sin_phi = np.sin(self.phi_pupil)
        n_safe = abs(self.n_immersion)

        Ep_x = cos_theta * cos_phi
        Ep_y = cos_theta * sin_phi
        Ep_z = self.sin_theta

        return {"Ex": Ep_x, "Ey": Ep_y, "Ez": Ep_z}

    def decompose_incident_polarization(self) -> Dict[str, np.ndarray]:
        """
        将入射偏振态分解到光瞳平面的s/p基矢上

        Returns:
            {'s_amp': s偏振复振幅分布, 'p_amp': p偏振复振幅分布}
        """
        ein_x, ein_y = self.incident_polarization.e_x, self.incident_polarization.e_y

        s_basis = self.s_polarization_basis()
        p_basis = self.p_polarization_basis()

        s_amp = ein_x * s_basis["Ex"] + ein_y * s_basis["Ey"]
        p_amp = ein_x * p_basis["Ex"] + ein_y * p_basis["Ey"]

        s_amp = s_amp * self.pupil_mask
        p_amp = p_amp * self.pupil_mask

        return {"s_amp": s_amp, "p_amp": p_amp}

    def compute_thin_film_modulation(self, is_reflection: bool = True) -> Dict[str, np.ndarray]:
        """
        计算薄膜堆栈引起的振幅/相位调制（随入射角变化）

        Args:
            is_reflection: True 计算反射调制，False 计算透射调制

        Returns:
            {'s_mod': s偏振调制因子, 'p_mod': p偏振调制因子}
        """
        if self.mask_stack is None:
            return {
                "s_mod": np.ones_like(self.FX, dtype=np.complex128) * self.pupil_mask,
                "p_mod": np.ones_like(self.FX, dtype=np.complex128) * self.pupil_mask
            }

        ny, nx = self.grid_size
        s_mod = np.zeros((ny, nx), dtype=np.complex128)
        p_mod = np.zeros((ny, nx), dtype=np.complex128)

        for i in range(ny):
            for j in range(nx):
                if not self.pupil_mask[i, j]:
                    continue
                theta = np.arcsin(self.sin_theta[i, j])

                result = self.mask_stack.compute_reflection_transmission(
                    self.wavelength_nm, theta
                )

                if is_reflection:
                    s_mod[i, j] = result["rs"]
                    p_mod[i, j] = result["rp"]
                else:
                    s_mod[i, j] = result["ts"]
                    p_mod[i, j] = result["tp"]

        return {"s_mod": s_mod, "p_mod": p_mod}

    def compute_vector_pupil(
        self,
        defocus_nm: float = 0.0,
        zernike_phase: Optional[np.ndarray] = None,
        is_reflection: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        计算完整的矢量光瞳函数

        对于每个光瞳位置 (fx, fy):
        E(fx, fy) = M_s(fx, fy) * s_amp * s_basis + M_p(fx, fy) * p_amp * p_basis

        其中 M_s, M_p 是薄膜调制因子，包含:
        - 薄膜振幅/相位响应
        - 离焦相位
        - 像差相位

        Args:
            defocus_nm: 离焦量 (nm)
            zernike_phase: Zernike像差相位 (弧度)
            is_reflection: 是否为反射式系统（EUV）

        Returns:
            {'Ex', 'Ey', 'Ez'} 三个分量的矢量光瞳
        """
        polarization = self.decompose_incident_polarization()
        film_mod = self.compute_thin_film_modulation(is_reflection=is_reflection)
        s_basis = self.s_polarization_basis()
        p_basis = self.p_polarization_basis()

        kz_sq = (self.k0 * self.n_immersion) ** 2 - (
            (self.k0 * self.FX) ** 2 + (self.k0 * self.FY) ** 2
        )
        kz = np.lib.scimath.sqrt(kz_sq)
        kz = np.where(np.imag(kz) < 0, -kz, kz)

        defocus_phase = np.ones_like(self.FX, dtype=np.complex128)
        if abs(defocus_nm) > 1e-10:
            defocus_sign = -1.0 if is_reflection else 1.0
            defocus_phase = np.exp(1j * defocus_sign * kz * defocus_nm)

        aberration_phase = np.ones_like(self.FX, dtype=np.complex128)
        if zernike_phase is not None:
            aberration_phase = np.exp(1j * zernike_phase)

        total_phase = defocus_phase * aberration_phase * self.pupil_mask

        s_total = film_mod["s_mod"] * polarization["s_amp"] * total_phase
        p_total = film_mod["p_mod"] * polarization["p_amp"] * total_phase

        Ex = s_total * s_basis["Ex"] + p_total * p_basis["Ex"]
        Ey = s_total * s_basis["Ey"] + p_total * p_basis["Ey"]
        Ez = s_total * s_basis["Ez"] + p_total * p_basis["Ez"]

        return {"Ex": Ex, "Ey": Ey, "Ez": Ez}

    def propagate_to_image(
        self,
        mask_spectrum: np.ndarray,
        vector_pupil: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        将掩模频谱通过矢量光瞳传播到像面，计算光强分布

        I(x, y) = |IFFT{Ex(f) * M(f)}|^2 + |IFFT{Ey(f) * M(f)}|^2 + |IFFT{Ez(f) * M(f)}|^2

        Args:
            mask_spectrum: 掩模的傅里叶频谱
            vector_pupil: 矢量光瞳函数 {'Ex', 'Ey', 'Ez'}

        Returns:
            归一化光强分布
        """
        ny, nx = self.grid_size

        Ex_pupil = mask_spectrum * vector_pupil["Ex"]
        Ey_pupil = mask_spectrum * vector_pupil["Ey"]
        Ez_pupil = mask_spectrum * vector_pupil["Ez"]

        Ex_xy = np.fft.ifft2(np.fft.ifftshift(Ex_pupil))
        Ey_xy = np.fft.ifft2(np.fft.ifftshift(Ey_pupil))
        Ez_xy = np.fft.ifft2(np.fft.ifftshift(Ez_pupil))

        intensity = np.abs(Ex_xy) ** 2 + np.abs(Ey_xy) ** 2 + np.abs(Ez_xy) ** 2

        max_I = float(np.nanmax(intensity))
        if max_I > 0:
            intensity = intensity / max_I

        return np.clip(intensity, 0.0, 1.0).astype(np.float64)

    def get_polarization_state(self, fx: float, fy: float) -> JonesVector:
        """获取指定光瞳位置的偏振态"""
        mask = (np.abs(self.FX - fx) < 1e-6) & (np.abs(self.FY - fy) < 1e-6)
        if not np.any(mask):
            raise ValueError(f"频率点 ({fx}, {fy}) 不在网格上")

        polarization = self.decompose_incident_polarization()
        s_basis = self.s_polarization_basis()
        p_basis = self.p_polarization_basis()

        s_amp = float(polarization["s_amp"][mask][0])
        p_amp = float(polarization["p_amp"][mask][0])

        ex = s_amp * s_basis["Ex"][mask][0] + p_amp * p_basis["Ex"][mask][0]
        ey = s_amp * s_basis["Ey"][mask][0] + p_amp * p_basis["Ey"][mask][0]

        return JonesVector(ex, ey)


# =============================================================================
# 偏振光瞳扩展计算 - 与现有成像模型集成
# =============================================================================
def compute_polarized_pupil(
    fx: np.ndarray,
    fy: np.ndarray,
    wavelength_nm: float,
    na: float,
    cutoff: float,
    defocus_nm: float,
    zernike_phase: np.ndarray,
    incident_polarization: JonesVector,
    n_immersion: complex = 1.0 + 0.0j,
    mask_stack: Optional[ThinFilmStack] = None,
    is_reflection: bool = False,
) -> Dict[str, np.ndarray]:
    """
    扩展的偏振光瞳函数计算（与现有 _compute_pupil_with_aberrations 兼容）

    这是标量光瞳函数的矢量扩展版本，返回三个电场分量的光瞳函数。

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        wavelength_nm: 波长 (nm)
        na: 数值孔径
        cutoff: 截止频率
        defocus_nm: 离焦量 (nm)
        zernike_phase: Zernike像差相位 (弧度)
        incident_polarization: 入射偏振态
        n_immersion: 浸没介质折射率
        mask_stack: 薄膜堆栈（可选）
        is_reflection: 是否为反射式系统（EUV）

    Returns:
        {'pupil_scalar': 标量光瞳（向后兼容）,
         'Ex', 'Ey', 'Ez': 三个分量的矢量光瞳,
         's_mod', 'p_mod': s/p偏振的薄膜调制因子}
    """
    ny, nx = fx.shape
    k0 = 2.0 * np.pi / wavelength_nm

    if mask_stack is not None:
        n_super = mask_stack._resolve_n(mask_stack.n_superstrate, wavelength_nm)
        if np.real(n_super) > 0 and not np.isclose(np.real(n_super), np.real(n_immersion), rtol=1e-3):
            n_immersion = n_super

    rho_sq = (fx ** 2 + fy ** 2) / (cutoff ** 2)
    pupil_mask = rho_sq <= 1.0

    defocus_phase_scalar = np.pi * defocus_nm / wavelength_nm * rho_sq
    total_phase_scalar = defocus_phase_scalar + zernike_phase
    pupil_scalar = np.zeros((ny, nx), dtype=np.complex128)
    pupil_scalar[pupil_mask] = np.exp(1j * total_phase_scalar[pupil_mask])

    rho = np.sqrt(fx ** 2 + fy ** 2) * wavelength_nm
    n_immersion_real = float(np.real(n_immersion)) if np.real(n_immersion) != 0 else 1.0
    sin_theta = rho / n_immersion_real
    sin_theta = np.clip(sin_theta, 0, 1)
    cos_theta = np.sqrt(np.maximum(1.0 - sin_theta ** 2, 0.0))
    phi = np.arctan2(fy, fx)

    with np.errstate(divide="ignore", invalid="ignore"):
        Es_x = -np.sin(phi)
        Es_y = np.cos(phi)
        Es_z = np.zeros_like(sin_theta)

        Ep_x = cos_theta * np.cos(phi)
        Ep_y = cos_theta * np.sin(phi)
        Ep_z = sin_theta

    ein_x, ein_y = incident_polarization.e_x, incident_polarization.e_y
    s_amp = ein_x * Es_x + ein_y * Es_y
    p_amp = ein_x * Ep_x + ein_y * Ep_y

    s_mod = np.ones((ny, nx), dtype=np.complex128)
    p_mod = np.ones((ny, nx), dtype=np.complex128)

    if mask_stack is not None:
        for i in range(ny):
            for j in range(nx):
                if not pupil_mask[i, j]:
                    continue
                theta = np.arcsin(sin_theta[i, j])
                result = mask_stack.compute_reflection_transmission(
                    wavelength_nm, theta
                )
                if is_reflection:
                    s_mod[i, j] = result["rs"]
                    p_mod[i, j] = result["rp"]
                else:
                    s_mod[i, j] = result["ts"]
                    p_mod[i, j] = result["tp"]

    kz_sq = (k0 * n_immersion) ** 2 - (
        (k0 * fx * wavelength_nm) ** 2 + (k0 * fy * wavelength_nm) ** 2
    )
    kz = np.lib.scimath.sqrt(kz_sq)
    kz = np.where(np.imag(kz) < 0, -kz, kz)

    defocus_phase_vec = np.ones((ny, nx), dtype=np.complex128)
    if abs(defocus_nm) > 1e-10:
        defocus_sign = -1.0 if is_reflection else 1.0
        defocus_phase_vec = np.exp(1j * defocus_sign * kz * defocus_nm)

    aberration_phase = np.exp(1j * zernike_phase)
    total_phase = defocus_phase_vec * aberration_phase * pupil_mask

    s_total = s_mod * s_amp * total_phase
    p_total = p_mod * p_amp * total_phase

    Ex = s_total * Es_x + p_total * Ep_x
    Ey = s_total * Es_y + p_total * Ep_y
    Ez = s_total * Es_z + p_total * Ep_z

    return {
        "pupil_scalar": pupil_scalar,
        "Ex": Ex, "Ey": Ey, "Ez": Ez,
        "s_mod": s_mod, "p_mod": p_mod,
        "pupil_mask": pupil_mask
    }


def scalar_from_vector_pupil(vector_pupil: Dict[str, np.ndarray]) -> np.ndarray:
    """
    从矢量光瞳合成标量光瞳（用于向后兼容）
    取各分量的幅度平方和开根号
    """
    Ex = vector_pupil.get("Ex", 0)
    Ey = vector_pupil.get("Ey", 0)
    Ez = vector_pupil.get("Ez", 0)
    amplitude = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)

    pupil_scalar = vector_pupil.get("pupil_scalar", None)
    if pupil_scalar is not None:
        phase = np.angle(pupil_scalar)
        return amplitude * np.exp(1j * phase)
    return amplitude


def compute_partial_coherent_vectorial(
    mask: np.ndarray,
    source: np.ndarray,
    vector_pupils: Dict[str, np.ndarray],
    dfx: float,
    dfy: float,
) -> np.ndarray:
    """
    部分相干矢量成像计算（Abbe 方法）

    对每个光源点 fs，计算:
    I(x) = Σ S(fs) * |Σ_i IFFT{Ex(f) Px(f - fs) M(f)}|^2

    这里使用与现有标量模型兼容的实现。

    Args:
        mask: 掩模图案
        source: 光源分布
        vector_pupils: 矢量光瞳 {'Ex', 'Ey', 'Ez'}
        dfx, dfy: 频率间隔

    Returns:
        归一化光强分布
    """
    from core.fft import fft2d, ifft2d

    ny, nx = mask.shape
    backend = np if not hasattr(mask, 'device') else type(mask)

    mask_c = mask.astype(np.complex128)
    mask_spectrum = np.fft.fft2(mask_c)

    intensity = np.zeros((ny, nx), dtype=np.float64)

    source_flat = source.ravel()
    nonzero = source_flat > 1e-12
    source_indices = np.where(nonzero)

    fx = np.fft.fftfreq(nx, 1.0 / (nx * dfx))
    fy = np.fft.fftfreq(ny, 1.0 / (ny * dfy))
    fx_grid, fy_grid = np.meshgrid(fx, fy)

    for idx in range(len(source_indices[0])):
        sy = int(source_indices[0][idx] // nx)
        sx = int(source_indices[0][idx] % nx)
        src_val = source_flat[source_indices[0][idx]]

        fs_x = fx_grid[sy, sx]
        fs_y = fy_grid[sy, sx]

        shift_x = int(round(fs_x / dfx))
        shift_y = int(round(fs_y / dfy))

        Ex_shifted = np.roll(vector_pupils["Ex"], shift=shift_y, axis=0)
        Ex_shifted = np.roll(Ex_shifted, shift=shift_x, axis=1)
        Ey_shifted = np.roll(vector_pupils["Ey"], shift=shift_y, axis=0)
        Ey_shifted = np.roll(Ey_shifted, shift=shift_x, axis=1)
        Ez_shifted = np.roll(vector_pupils["Ez"], shift=shift_y, axis=0)
        Ez_shifted = np.roll(Ez_shifted, shift=shift_x, axis=1)

        Ex_f = mask_spectrum * Ex_shifted
        Ey_f = mask_spectrum * Ey_shifted
        Ez_f = mask_spectrum * Ez_shifted

        Ex_xy = np.fft.ifft2(Ex_f)
        Ey_xy = np.fft.ifft2(Ey_f)
        Ez_xy = np.fft.ifft2(Ez_f)

        intensity_i = np.abs(Ex_xy) ** 2 + np.abs(Ey_xy) ** 2 + np.abs(Ez_xy) ** 2
        intensity += src_val * intensity_i

    max_I = float(np.nanmax(intensity))
    if max_I > 0:
        intensity = intensity / max_I

    return np.clip(intensity, 0.0, 1.0).astype(np.float64)


# =============================================================================
# 简化预配置 - 高NA浸没式和EUV系统
# =============================================================================
def create_high_na_immersion_system(
    wavelength_nm: float = 193.0,
    na: float = 1.35,
    pixel_size_nm: float = 1.0,
    grid_size: Tuple[int, int] = (256, 256),
    incident_polarization: JonesVector = None,
) -> VectorPupil:
    """
    创建高NA浸没式（ArF）光刻系统的矢量光瞳模型

    Args:
        wavelength_nm: 波长 (nm), 默认 193nm (ArF)
        na: 数值孔径, 默认 1.35 (浸没式)
        pixel_size_nm: 像素尺寸 (nm)
        grid_size: 网格大小
        incident_polarization: 入射偏振, 默认 TE线偏振 (s-偏振)

    Returns:
        VectorPupil 实例
    """
    if incident_polarization is None:
        incident_polarization = JonesVector.linear_polarization(0.0)

    ar_stack = ThinFilmStack.arf_antireflective(wavelength_nm)

    return VectorPupil(
        wavelength_nm=wavelength_nm,
        na=na,
        n_immersion=STANDARD_MATERIALS["water"].get_n(wavelength_nm),
        pixel_size_nm=pixel_size_nm,
        grid_size=grid_size,
        mask_stack=ar_stack,
        incident_polarization=incident_polarization
    )


def create_euv_reflective_system(
    wavelength_nm: float = 13.5,
    na: float = 0.33,
    pixel_size_nm: float = 0.5,
    grid_size: Tuple[int, int] = (256, 256),
    num_multilayer_pairs: int = 40,
    incident_polarization: JonesVector = None,
) -> VectorPupil:
    """
    创建EUV反射式光刻系统的矢量光瞳模型

    Args:
        wavelength_nm: 波长 (nm), 默认 13.5nm
        na: 数值孔径, 默认 0.33
        pixel_size_nm: 像素尺寸 (nm)
        grid_size: 网格大小
        num_multilayer_pairs: Mo/Si 多层膜对数, 默认 40
        incident_polarization: 入射偏振, 默认 TE线偏振

    Returns:
        VectorPupil 实例
    """
    if incident_polarization is None:
        incident_polarization = JonesVector.linear_polarization(0.0)

    euv_stack = ThinFilmStack.euv_multilayer(num_multilayer_pairs, wavelength_nm)

    return VectorPupil(
        wavelength_nm=wavelength_nm,
        na=na,
        n_immersion=STANDARD_MATERIALS["air"].get_n(wavelength_nm),
        pixel_size_nm=pixel_size_nm,
        grid_size=grid_size,
        mask_stack=euv_stack,
        incident_polarization=incident_polarization
    )


# =============================================================================
# 偏振对比度分析工具
# =============================================================================
def compute_polarization_contribution(
    vector_pupil: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    计算各偏振分量对总光强的贡献比例

    Args:
        vector_pupil: 矢量光瞳 {'Ex', 'Ey', 'Ez'}

    Returns:
        {'Ex_contrib', 'Ey_contrib', 'Ez_contrib', 'total'}
    """
    Ex = vector_pupil.get("Ex", np.zeros_like(vector_pupil.get("pupil_scalar", 0)))
    Ey = vector_pupil.get("Ey", np.zeros_like(Ex))
    Ez = vector_pupil.get("Ez", np.zeros_like(Ex))

    Ix = float(np.sum(np.abs(Ex) ** 2))
    Iy = float(np.sum(np.abs(Ey) ** 2))
    Iz = float(np.sum(np.abs(Ez) ** 2))
    total = Ix + Iy + Iz

    if total <= 0:
        return {"Ex_contrib": 0.0, "Ey_contrib": 0.0, "Ez_contrib": 0.0, "total": 0.0}

    return {
        "Ex_contrib": Ix / total,
        "Ey_contrib": Iy / total,
        "Ez_contrib": Iz / total,
        "total": total
    }


def compute_polarization_degree(
    vector_pupil: Dict[str, np.ndarray],
) -> float:
    """
    计算偏振度 (Degree of Polarization, DOP)

    DOP = sqrt(S1^2 + S2^2 + S3^2) / S0

    其中 S0, S1, S2, S3 为 Stokes 参数

    Returns:
        DOP 值，范围 [0, 1]，1 为完全偏振，0 为完全非偏振
    """
    Ex = vector_pupil.get("Ex", 0)
    Ey = vector_pupil.get("Ey", 0)

    if np.isscalar(Ex):
        Ex = np.array([Ex])
        Ey = np.array([Ey])

    S0 = np.mean(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)
    S1 = np.mean(np.abs(Ex) ** 2 - np.abs(Ey) ** 2)
    S2 = 2 * np.mean(np.real(Ex * np.conj(Ey)))
    S3 = 2 * np.mean(np.imag(Ex * np.conj(Ey)))

    if S0 <= 0:
        return 0.0

    dop = np.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / S0
    return float(np.clip(dop, 0.0, 1.0))
