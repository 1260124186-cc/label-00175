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
from typing import Tuple, Optional
from dataclasses import dataclass


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
    """
    wavelength: float = 193.0  # ArF光源波长
    na: float = 1.35  # 高NA浸没式光刻
    sigma: float = 0.75  # 部分相干因子
    pixel_size: float = 1.0  # 像素尺寸
    defocus: float = 0.0  # 离焦量
    magnification: float = 4.0  # 放大倍率
    
    @property
    def k1(self) -> float:
        """计算分辨率因子k1"""
        return self.wavelength / (2 * self.na)
    
    @property
    def cutoff_frequency(self) -> float:
        """计算截止频率"""
        return self.na / self.wavelength


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
                # 离焦相位
                phase = np.pi * defocus * wavelength * rho**2
                pupil[i, j] = np.exp(1j * phase)
    
    return pupil


@jit(nopython=True, parallel=True, cache=True)
def _compute_tcc_kernel(fx: np.ndarray, fy: np.ndarray,
                        pupil: np.ndarray, sigma: float,
                        cutoff: float) -> np.ndarray:
    """
    计算传输交叉系数(TCC)核心
    
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
    
    # 简化的TCC计算（圆形光源）
    for i in prange(ny):
        for j in range(nx):
            rho = np.sqrt(fx[i, j]**2 + fy[i, j]**2)
            if rho <= source_radius:
                tcc[i, j] = np.abs(pupil[i, j])**2
    
    # 归一化
    total = np.sum(tcc)
    if total > 0:
        tcc = tcc / total
    
    return tcc


class PartialCoherentImaging:
    """
    部分相干成像模型类
    
    实现Hopkins部分相干成像理论，用于计算掩模图案在晶圆上的成像结果。
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
        self._compute_transfer_functions()
    
    def _setup_frequency_grid(self):
        """设置频率网格"""
        ny, nx = self.image_size
        
        # 频率采样间隔
        dfx = 1.0 / (nx * self.optics.pixel_size)
        dfy = 1.0 / (ny * self.optics.pixel_size)
        
        # 创建频率网格
        fx = np.fft.fftfreq(nx, self.optics.pixel_size)
        fy = np.fft.fftfreq(ny, self.optics.pixel_size)
        self.fx, self.fy = np.meshgrid(fx, fy)
    
    def _compute_transfer_functions(self):
        """计算传递函数"""
        cutoff = self.optics.cutoff_frequency
        
        # 计算光瞳函数
        self.pupil = _compute_pupil_function(
            self.fx, self.fy, cutoff,
            self.optics.defocus, self.optics.wavelength
        )
        
        # 计算TCC核
        self.tcc = _compute_tcc_kernel(
            self.fx, self.fy, self.pupil,
            self.optics.sigma, cutoff
        )
    
    def compute_aerial_image(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像（晶圆上的光强分布）
        
        Args:
            mask: 掩模图案 (2D numpy数组, 0-1值)
            
        Returns:
            空间像光强分布
        """
        # 掩模频谱
        mask_spectrum = np.fft.fft2(mask.astype(np.complex128))
        
        # 通过光瞳滤波
        filtered_spectrum = mask_spectrum * self.pupil
        
        # 计算电场
        field = np.fft.ifft2(filtered_spectrum)
        
        # 光强 = |电场|^2
        intensity = np.abs(field)**2
        
        # 归一化到0-1
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
        mask_spectrum = np.fft.fft2(mask_c)
        filtered_spectrum = mask_spectrum * self.pupil
        field = np.fft.ifft2(filtered_spectrum)
        
        # 梯度计算: d(|E|^2)/d(mask) = 2 * Re(E* * dE/d(mask))
        # dE/d(mask) 通过链式法则计算
        grad_field = np.fft.ifft2(self.pupil)
        gradient = 2 * np.real(np.conj(field) * grad_field)
        
        return gradient.astype(np.float64)


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
