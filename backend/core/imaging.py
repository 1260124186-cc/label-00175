# -*- coding: utf-8 -*-
"""
光学成像模块：光刻光学系统建模、部分相干成像模型、晶圆成像模拟

该模块实现了光刻系统中的光学成像仿真，包括：
1. 光学系统参数配置
2. 部分相干成像模型（Hopkins模型）
3. 光强分布计算
4. 晶圆成像模拟
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class IlluminationType(Enum):
    """照明模式类型"""
    CONVENTIONAL = "conventional"      # 传统圆形照明
    DIPOLE = "dipole"                  # 偶极照明
    ANNULAR = "annular"                # 环形照明
    QUASAR = "quasar"                  # 四极照明
    CUSTOM = "custom"                 # 自定义照明


@dataclass
class OpticalSystem:
    """
    光学系统参数配置类

    Attributes:
        wavelength: 光源波长 (nm)
        na: 数值孔径 (Numerical Aperture)
        sigma: 部分相干因子 (0~1, 0为完全相干, 1为完全非相干)
        pixel_size: 像素尺寸 (nm)
        defocus: 离焦量 (nm)
        magnification: 放大倍率
        illumination_type: 照明模式类型
        source_params: 光源形状参数
            - conventional: {'sigma_inner': 0.0, 'sigma_outer': sigma}
            - dipole: {'sigma_inner': 0.5, 'sigma_outer': 0.8, 'angle': 0.0, 'opening_angle': 60.0}
            - annular: {'sigma_inner': 0.6, 'sigma_outer': 0.9}
            - quasar: {'sigma_inner': 0.5, 'sigma_outer': 0.8, 'angle': 45.0, 'opening_angle': 30.0}
        use_socs: 是否使用SOCS低秩分解近似
        socs_num_terms: SOCS分解项数
        custom_source: 自定义光源分布（当illumination_type=CUSTOM时使用）
    """
    wavelength: float = 193.0  # ArF光源波长
    na: float = 1.35  # 高NA浸没式光刻
    sigma: float = 0.75  # 部分相干因子
    pixel_size: float = 1.0  # 像素尺寸
    defocus: float = 0.0  # 离焦量
    magnification: float = 4.0  # 放大倍率
    illumination_type: IlluminationType = IlluminationType.CONVENTIONAL
    source_params: Dict[str, float] = field(default_factory=dict)
    use_socs: bool = True
    socs_num_terms: int = 5
    custom_source: Optional[np.ndarray] = None

    def __post_init__(self):
        """初始化后处理，设置默认光源参数"""
        if not self.source_params:
            self._set_default_source_params()

    def _set_default_source_params(self):
        """设置默认光源参数"""
        if self.illumination_type == IlluminationType.CONVENTIONAL:
            self.source_params = {
                'sigma_inner': 0.0,
                'sigma_outer': self.sigma
            }
        elif self.illumination_type == IlluminationType.DIPOLE:
            self.source_params = {
                'sigma_inner': 0.5,
                'sigma_outer': 0.8,
                'angle': 0.0,
                'opening_angle': 60.0
            }
        elif self.illumination_type == IlluminationType.ANNULAR:
            self.source_params = {
                'sigma_inner': 0.6,
                'sigma_outer': 0.9
            }
        elif self.illumination_type == IlluminationType.QUASAR:
            self.source_params = {
                'sigma_inner': 0.5,
                'sigma_outer': 0.8,
                'angle': 45.0,
                'opening_angle': 30.0
            }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OpticalSystem':
        """
        从配置字典创建OpticalSystem实例

        Args:
            config: 配置字典

        Returns:
            OpticalSystem实例
        """
        optics_config = config.get('optical_system', config)

        illumination_type_str = optics_config.get('illumination_type', 'conventional')
        try:
            illumination_type = IlluminationType(illumination_type_str)
        except ValueError:
            illumination_type = IlluminationType.CONVENTIONAL

        source_params = optics_config.get('source_params', {})

        return cls(
            wavelength=optics_config.get('wavelength', 193.0),
            na=optics_config.get('na', 1.35),
            sigma=optics_config.get('sigma', 0.75),
            pixel_size=optics_config.get('pixel_size', 1.0),
            defocus=optics_config.get('defocus', 0.0),
            magnification=optics_config.get('magnification', 4.0),
            illumination_type=illumination_type,
            source_params=source_params,
            use_socs=optics_config.get('use_socs', True),
            socs_num_terms=optics_config.get('socs_num_terms', 5)
        )

    @property
    def k1(self) -> float:
        """计算分辨率因子k1"""
        return self.wavelength / (2 * self.na)

    @property
    def cutoff_frequency(self) -> float:
        """计算截止频率"""
        return self.na / self.wavelength

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'wavelength': self.wavelength,
            'na': self.na,
            'sigma': self.sigma,
            'pixel_size': self.pixel_size,
            'defocus': self.defocus,
            'magnification': self.magnification,
            'illumination_type': self.illumination_type.value,
            'source_params': self.source_params,
            'use_socs': self.use_socs,
            'socs_num_terms': self.socs_num_terms
        }


def generate_source(fx: np.ndarray, fy: np.ndarray,
                    illumination_type: IlluminationType,
                    source_params: Dict[str, float],
                    cutoff: float,
                    custom_source: Optional[np.ndarray] = None) -> np.ndarray:
    """
    生成光源分布

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        illumination_type: 照明模式类型
        source_params: 光源参数
        cutoff: 截止频率
        custom_source: 自定义光源分布

    Returns:
        归一化的光源分布
    """
    ny, nx = fx.shape
    source = np.zeros((ny, nx), dtype=np.float64)

    if illumination_type == IlluminationType.CUSTOM and custom_source is not None:
        if custom_source.shape == (ny, nx):
            source = custom_source.astype(np.float64)
        else:
            raise ValueError(f"自定义光源形状 {custom_source.shape} 与频率网格形状 {(ny, nx)} 不匹配")
    else:
        sigma_inner = source_params.get('sigma_inner', 0.0)
        sigma_outer = source_params.get('sigma_outer', source_params.get('sigma', 0.75))

        rho = np.sqrt(fx**2 + fy**2) / cutoff
        theta = np.arctan2(fy, fx)

        if illumination_type == IlluminationType.CONVENTIONAL:
            mask = (rho >= sigma_inner) & (rho <= sigma_outer)
            source[mask] = 1.0

        elif illumination_type == IlluminationType.ANNULAR:
            mask = (rho >= sigma_inner) & (rho <= sigma_outer)
            source[mask] = 1.0

        elif illumination_type == IlluminationType.DIPOLE:
            angle = np.deg2rad(source_params.get('angle', 0.0))
            opening_angle = np.deg2rad(source_params.get('opening_angle', 60.0))

            radial_mask = (rho >= sigma_inner) & (rho <= sigma_outer)
            angle_diff = np.abs(np.mod(theta - angle + np.pi, 2 * np.pi) - np.pi)
            angle_mask1 = angle_diff <= opening_angle / 2
            angle_mask2 = angle_diff >= (np.pi - opening_angle / 2)

            mask = radial_mask & (angle_mask1 | angle_mask2)
            source[mask] = 1.0

        elif illumination_type == IlluminationType.QUASAR:
            angle = np.deg2rad(source_params.get('angle', 45.0))
            opening_angle = np.deg2rad(source_params.get('opening_angle', 30.0))

            radial_mask = (rho >= sigma_inner) & (rho <= sigma_outer)

            pole_angles = [angle, angle + np.pi/2, angle + np.pi, angle + 3*np.pi/2]
            angle_mask = np.zeros_like(rho, dtype=bool)

            for pole_angle in pole_angles:
                angle_diff = np.abs(np.mod(theta - pole_angle + np.pi, 2 * np.pi) - np.pi)
                angle_mask |= (angle_diff <= opening_angle / 2)

            mask = radial_mask & angle_mask
            source[mask] = 1.0

    total = np.sum(source)
    if total > 0:
        source = source / total

    return source


@jit(nopython=True, parallel=True, cache=True)
def _compute_pupil_function(fx: np.ndarray, fy: np.ndarray,
                            cutoff: float, defocus: float,
                            wavelength: float) -> np.ndarray:
    """
    计算光瞳函数（含离焦相位）

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        cutoff: 截止频率
        defocus: 离焦量
        wavelength: 波长

    Returns:
        复数光瞳函数
    """
    ny, nx = fx.shape
    pupil = np.zeros((ny, nx), dtype=np.complex128)

    for i in prange(ny):
        for j in range(nx):
            rho = np.sqrt(fx[i, j]**2 + fy[i, j]**2)
            if rho <= cutoff:
                phase = np.pi * defocus * wavelength * rho**2
                pupil[i, j] = np.exp(1j * phase)

    return pupil


def _shift_pupil(pupil: np.ndarray, shift_fx: float,
                 shift_fy: float, dfx: float, dfy: float) -> np.ndarray:
    """
    对光瞳函数进行频移

    Args:
        pupil: 光瞳函数
        shift_fx: x方向频移量
        shift_fy: y方向频移量
        dfx: x方向频率间隔
        dfy: y方向频率间隔

    Returns:
        频移后的光瞳函数
    """
    ny, nx = pupil.shape

    shift_x = int(round(shift_fx / dfx))
    shift_y = int(round(shift_fy / dfy))

    if shift_x == 0 and shift_y == 0:
        return pupil.copy()

    shifted = np.roll(pupil, shift=shift_y, axis=0)
    shifted = np.roll(shifted, shift=shift_x, axis=1)

    return shifted


def compute_tcc_full(fx: np.ndarray, fy: np.ndarray,
                     pupil: np.ndarray, source: np.ndarray,
                     cutoff: float, dfx: float, dfy: float) -> np.ndarray:
    """
    基于光源积分计算完整的 TCC 矩阵 (四维)

    TCC(f1, f2) = ∫ S(fs) * P(f1 - fs) * P*(f2 - fs) dfs

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        pupil: 光瞳函数
        source: 归一化光源分布
        cutoff: 截止频率
        dfx: x方向频率间隔
        dfy: y方向频率间隔

    Returns:
        TCC矩阵，形状为 (ny, nx, ny, nx)
    """
    ny, nx = pupil.shape
    tcc = np.zeros((ny, nx, ny, nx), dtype=np.complex128)

    source_indices = np.where(source > 1e-10)
    source_values = source[source_indices]

    for idx in range(len(source_indices[0])):
        sy, sx = source_indices[0][idx], source_indices[1][idx]
        src_val = source_values[idx]

        fs_x = fx[sy, sx]
        fs_y = fy[sy, sx]

        if np.sqrt(fs_x**2 + fs_y**2) > cutoff:
            continue

        pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, dfx, dfy)
        pupil_conj_shifted = np.conj(pupil_shifted)

        for i in range(ny):
            for j in range(nx):
                p1 = pupil_shifted[i, j]
                if abs(p1) < 1e-10:
                    continue
                for k in range(ny):
                    for l in range(nx):
                        p2 = pupil_conj_shifted[k, l]
                        if abs(p2) < 1e-10:
                            continue
                        tcc[i, j, k, l] += src_val * p1 * p2

    return tcc


def compute_tcc_kernel_2d(fx: np.ndarray, fy: np.ndarray,
                          pupil: np.ndarray, source: np.ndarray,
                          cutoff: float, dfx: float, dfy: float) -> np.ndarray:
    """
    计算二维 TCC 核（对角近似，用于传统成像）

    当 f1 = f2 时，TCC(f, f) = ∫ S(fs) * |P(f - fs)|^2 dfs

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        pupil: 光瞳函数
        source: 归一化光源分布
        cutoff: 截止频率
        dfx: x方向频率间隔
        dfy: y方向频率间隔

    Returns:
        二维 TCC 核
    """
    ny, nx = pupil.shape
    tcc_kernel = np.zeros((ny, nx), dtype=np.float64)

    source_indices = np.where(source > 1e-10)
    source_values = source[source_indices]

    for idx in range(len(source_indices[0])):
        sy, sx = source_indices[0][idx], source_indices[1][idx]
        src_val = source_values[idx]

        fs_x = fx[sy, sx]
        fs_y = fy[sy, sx]

        if np.sqrt(fs_x**2 + fs_y**2) > cutoff:
            continue

        pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, dfx, dfy)
        tcc_kernel += src_val * np.abs(pupil_shifted)**2

    total = np.sum(tcc_kernel)
    if total > 0:
        tcc_kernel = tcc_kernel / total

    return tcc_kernel


def socs_decomposition(fx: np.ndarray, fy: np.ndarray,
                       pupil: np.ndarray, source: np.ndarray,
                       cutoff: float, dfx: float, dfy: float,
                       num_terms: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    SOCS (Sum of Coherent Systems) 低秩分解

    将 TCC 矩阵分解为: TCC ≈ Σ λ_i |φ_i><φ_i|

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        pupil: 光瞳函数
        source: 归一化光源分布
        cutoff: 截止频率
        dfx: x方向频率间隔
        dfy: y方向频率间隔
        num_terms: 分解项数

    Returns:
        (eigenvalues, eigenfunctions) - 特征值和特征函数
            eigenvalues: 形状为 (num_terms,)
            eigenfunctions: 形状为 (num_terms, ny, nx)
    """
    ny, nx = pupil.shape
    N = ny * nx

    source_indices = np.where(source > 1e-10)
    source_values = source[source_indices]

    M = len(source_indices[0])
    V = np.zeros((M, N), dtype=np.complex128)

    for idx in range(M):
        sy, sx = source_indices[0][idx], source_indices[1][idx]
        src_val = np.sqrt(source_values[idx])

        fs_x = fx[sy, sx]
        fs_y = fy[sy, sx]

        pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, dfx, dfy)
        V[idx, :] = src_val * pupil_shifted.flatten()

    if M <= N:
        VVh = V @ V.conj().T
        eigenvalues, eigenvectors = np.linalg.eigh(VVh)

        idx_sorted = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sorted]
        eigenvectors = eigenvectors[:, idx_sorted]

        num_terms = min(num_terms, M)
        eigenvalues = eigenvalues[:num_terms]
        eigenvectors = eigenvectors[:, :num_terms]

        eigenfunctions = np.zeros((num_terms, ny, nx), dtype=np.complex128)
        for i in range(num_terms):
            phi_flat = V.conj().T @ eigenvectors[:, i]
            norm = np.sqrt(np.sum(np.abs(phi_flat)**2))
            if norm > 1e-10:
                phi_flat = phi_flat / norm
            eigenfunctions[i, :, :] = phi_flat.reshape(ny, nx)
    else:
        VhV = V.conj().T @ V
        eigenvalues, eigenvectors = np.linalg.eigh(VhV)

        idx_sorted = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sorted]
        eigenvectors = eigenvectors[:, idx_sorted]

        num_terms = min(num_terms, N)
        eigenvalues = eigenvalues[:num_terms]
        eigenvectors = eigenvectors[:, :num_terms]

        eigenfunctions = np.zeros((num_terms, ny, nx), dtype=np.complex128)
        for i in range(num_terms):
            eigenfunctions[i, :, :] = eigenvectors[:, i].reshape(ny, nx)

    eigenvalues = np.real(eigenvalues)
    total_energy = np.sum(eigenvalues)
    if total_energy > 0:
        eigenvalues = eigenvalues / total_energy

    return eigenvalues, eigenfunctions


@jit(nopython=True, parallel=True, cache=True)
def _compute_tcc_kernel_simple(fx: np.ndarray, fy: np.ndarray,
                               pupil: np.ndarray, sigma: float,
                               cutoff: float) -> np.ndarray:
    """
    简化的 TCC 核计算（向后兼容）

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        pupil: 光瞳函数
        sigma: 部分相干因子
        cutoff: 截止频率

    Returns:
        TCC核
    """
    ny, nx = fx.shape
    tcc = np.zeros((ny, nx), dtype=np.float64)
    source_radius = sigma * cutoff

    for i in prange(ny):
        for j in range(nx):
            rho = np.sqrt(fx[i, j]**2 + fy[i, j]**2)
            if rho <= source_radius:
                tcc[i, j] = np.abs(pupil[i, j])**2

    total = np.sum(tcc)
    if total > 0:
        tcc = tcc / total

    return tcc


class PartialCoherentImaging:
    """
    部分相干成像模型类

    实现Hopkins部分相干成像理论，用于计算掩模图案在晶圆上的成像结果。
    支持完整TCC矩阵计算和SOCS低秩分解近似。
    """

    def __init__(self, optical_system: OpticalSystem, image_size: Tuple[int, int]):
        """
        初始化部分相干成像模型

        Args:
            optical_system: 光学系统参数
            image_size: 图像尺寸 (height, width)
        """
        self.optics = optical_system
        self.image_size = image_size
        self._setup_frequency_grid()
        self._compute_source_and_pupil()
        self._compute_transfer_functions()

    def _setup_frequency_grid(self):
        """设置频率网格"""
        ny, nx = self.image_size

        self.dfx = 1.0 / (nx * self.optics.pixel_size)
        self.dfy = 1.0 / (ny * self.optics.pixel_size)

        fx = np.fft.fftfreq(nx, self.optics.pixel_size)
        fy = np.fft.fftfreq(ny, self.optics.pixel_size)
        self.fx, self.fy = np.meshgrid(fx, fy)

    def _compute_source_and_pupil(self):
        """计算光源分布和光瞳函数"""
        cutoff = self.optics.cutoff_frequency

        self.pupil = _compute_pupil_function(
            self.fx, self.fy, cutoff,
            self.optics.defocus, self.optics.wavelength
        )

        self.source = generate_source(
            self.fx, self.fy,
            self.optics.illumination_type,
            self.optics.source_params,
            cutoff,
            self.optics.custom_source
        )

    def _compute_transfer_functions(self):
        """计算传递函数（TCC或SOCS分解）"""
        cutoff = self.optics.cutoff_frequency

        if self.optics.use_socs:
            self.socs_eigenvalues, self.socs_eigenfunctions = socs_decomposition(
                self.fx, self.fy,
                self.pupil, self.source,
                cutoff, self.dfx, self.dfy,
                self.optics.socs_num_terms
            )
            self.tcc = None
        else:
            self.tcc = compute_tcc_kernel_2d(
                self.fx, self.fy,
                self.pupil, self.source,
                cutoff, self.dfx, self.dfy
            )
            self.socs_eigenvalues = None
            self.socs_eigenfunctions = None

    def compute_aerial_image(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像（晶圆上的光强分布）

        使用Hopkins公式: I = ∫ S(fs) * |FFT^{-1}[M(f) * P(f - fs)]|^2 dfs
        或通过SOCS分解加速计算

        Args:
            mask: 掩模图案 (2D numpy数组, 0-1值)

        Returns:
            空间像光强分布
        """
        mask_c = mask.astype(np.complex128)
        ny, nx = mask.shape

        if self.optics.use_socs and self.socs_eigenvalues is not None:
            intensity = np.zeros((ny, nx), dtype=np.float64)
            mask_spectrum = np.fft.fft2(mask_c)

            for i, (lam, phi) in enumerate(zip(self.socs_eigenvalues, self.socs_eigenfunctions)):
                if lam < 1e-10:
                    continue
                filtered = mask_spectrum * phi
                field_i = np.fft.ifft2(filtered)
                intensity += lam * np.abs(field_i)**2
        else:
            intensity = np.zeros((ny, nx), dtype=np.float64)
            mask_spectrum = np.fft.fft2(mask_c)
            cutoff = self.optics.cutoff_frequency

            source_indices = np.where(self.source > 1e-10)
            source_values = self.source[source_indices]

            for idx in range(len(source_indices[0])):
                sy, sx = source_indices[0][idx], source_indices[1][idx]
                src_val = source_values[idx]

                fs_x = self.fx[sy, sx]
                fs_y = self.fy[sy, sx]

                if np.sqrt(fs_x**2 + fs_y**2) > cutoff:
                    continue

                pupil_shifted = _shift_pupil(
                    self.pupil, fs_x, fs_y, self.dfx, self.dfy
                )
                filtered = mask_spectrum * pupil_shifted
                field_i = np.fft.ifft2(filtered)
                intensity += src_val * np.abs(field_i)**2

        if intensity.max() > 0:
            intensity = intensity / intensity.max()

        return intensity.astype(np.float64)

    def compute_image_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像对掩模的梯度（用于优化）

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        mask_c = mask.astype(np.complex128)
        ny, nx = mask.shape
        gradient = np.zeros((ny, nx), dtype=np.float64)
        mask_spectrum = np.fft.fft2(mask_c)
        cutoff = self.optics.cutoff_frequency

        if self.optics.use_socs and self.socs_eigenvalues is not None:
            for lam, phi in zip(self.socs_eigenvalues, self.socs_eigenfunctions):
                if lam < 1e-10:
                    continue
                filtered = mask_spectrum * phi
                field_i = np.fft.ifft2(filtered)
                grad_field_i = np.fft.ifft2(phi)
                gradient += 2 * lam * np.real(np.conj(field_i) * grad_field_i)
        else:
            source_indices = np.where(self.source > 1e-10)
            source_values = self.source[source_indices]

            for idx in range(len(source_indices[0])):
                sy, sx = source_indices[0][idx], source_indices[1][idx]
                src_val = source_values[idx]

                fs_x = self.fx[sy, sx]
                fs_y = self.fy[sy, sx]

                if np.sqrt(fs_x**2 + fs_y**2) > cutoff:
                    continue

                pupil_shifted = _shift_pupil(
                    self.pupil, fs_x, fs_y, self.dfx, self.dfy
                )
                filtered = mask_spectrum * pupil_shifted
                field_i = np.fft.ifft2(filtered)
                grad_field_i = np.fft.ifft2(pupil_shifted)
                gradient += 2 * src_val * np.real(np.conj(field_i) * grad_field_i)

        return gradient.astype(np.float64)

    def get_source_image(self) -> np.ndarray:
        """获取光源分布图像（fftshift后便于可视化）"""
        return np.fft.fftshift(self.source)

    def get_pupil_image(self) -> np.ndarray:
        """获取光瞳函数图像（fftshift后便于可视化）"""
        return np.fft.fftshift(np.abs(self.pupil))

    def get_tcc_image(self) -> Optional[np.ndarray]:
        """获取TCC核图像（fftshift后便于可视化）"""
        if self.tcc is not None:
            return np.fft.fftshift(self.tcc)
        return None


@jit(nopython=True, cache=True)
def _apply_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    应用阈值处理（模拟光刻胶响应）

    Args:
        image: 输入图像
        threshold: 阈值

    Returns:
        二值化图像
    """
    ny, nx = image.shape
    result = np.zeros((ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            if image[i, j] >= threshold:
                result[i, j] = 1.0

    return result


def simulate_wafer_image(mask: np.ndarray,
                         optical_system: Optional[OpticalSystem] = None,
                         threshold: float = 0.3,
                         apply_resist: bool = True) -> np.ndarray:
    """
    模拟晶圆成像

    完整的成像流程：掩模 -> 光学成像 -> 光刻胶响应

    Args:
        mask: 掩模图案 (2D numpy数组)
        optical_system: 光学系统参数，None则使用默认参数
        threshold: 光刻胶阈值
        apply_resist: 是否应用光刻胶阈值处理

    Returns:
        晶圆成像结果
    """
    if optical_system is None:
        optical_system = OpticalSystem()

    # 创建成像模型
    imaging_model = PartialCoherentImaging(optical_system, mask.shape)

    # 计算空间像
    aerial_image = imaging_model.compute_aerial_image(mask)

    # 应用光刻胶响应
    if apply_resist:
        wafer_image = _apply_threshold(aerial_image, threshold)
    else:
        wafer_image = aerial_image

    return wafer_image
