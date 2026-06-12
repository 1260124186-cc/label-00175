# -*- coding: utf-8 -*-
"""
傅里叶变换模块：FFT/IFFT封装、频域滤波、相位调制

该模块提供掩模图案频域分析所需的傅里叶变换相关函数，包括：
1. 1D/2D快速傅里叶变换及逆变换
2. 频域滤波器（低通、高通、带通）
3. 相位调制函数

核心计算函数使用numba加速。
"""

import numpy as np
from numba import jit, prange, complex128, float64
from typing import Tuple, Optional, Union
from scipy import fft as scipy_fft
from scipy.signal.windows import tukey as _scipy_tukey
from enum import Enum


class WindowType(Enum):
    HANN = "hann"
    HAMMING = "hamming"
    TUKEY = "tukey"


@jit(nopython=True, cache=True)
def _fftshift_1d(spectrum: np.ndarray) -> np.ndarray:
    """
    一维FFT shift（numba加速）
    
    Args:
        spectrum: 输入频谱
        
    Returns:
        shift后的频谱
    """
    n = len(spectrum)
    mid = n // 2
    result = np.empty_like(spectrum)
    
    for i in range(n):
        new_idx = (i + mid) % n
        result[new_idx] = spectrum[i]
    
    return result


@jit(nopython=True, cache=True)
def _ifftshift_1d(spectrum: np.ndarray) -> np.ndarray:
    """
    一维IFFT shift（numba加速）
    
    Args:
        spectrum: 输入频谱
        
    Returns:
        shift后的频谱
    """
    n = len(spectrum)
    mid = (n + 1) // 2
    result = np.empty_like(spectrum)
    
    for i in range(n):
        new_idx = (i + mid) % n
        result[new_idx] = spectrum[i]
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def _fftshift_2d(spectrum: np.ndarray) -> np.ndarray:
    """
    二维FFT shift（numba加速）
    
    Args:
        spectrum: 输入二维频谱
        
    Returns:
        shift后的频谱
    """
    ny, nx = spectrum.shape
    mid_y, mid_x = ny // 2, nx // 2
    result = np.empty_like(spectrum)
    
    for i in prange(ny):
        for j in range(nx):
            new_i = (i + mid_y) % ny
            new_j = (j + mid_x) % nx
            result[new_i, new_j] = spectrum[i, j]
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def _ifftshift_2d(spectrum: np.ndarray) -> np.ndarray:
    """
    二维IFFT shift（numba加速）
    
    Args:
        spectrum: 输入二维频谱
        
    Returns:
        shift后的频谱
    """
    ny, nx = spectrum.shape
    mid_y, mid_x = (ny + 1) // 2, (nx + 1) // 2
    result = np.empty_like(spectrum)
    
    for i in prange(ny):
        for j in range(nx):
            new_i = (i + mid_y) % ny
            new_j = (j + mid_x) % nx
            result[new_i, new_j] = spectrum[i, j]
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def _normalize_spectrum(spectrum: np.ndarray, factor: float) -> np.ndarray:
    """
    频谱归一化（numba加速）
    
    Args:
        spectrum: 输入频谱
        factor: 归一化因子
        
    Returns:
        归一化后的频谱
    """
    ny, nx = spectrum.shape
    result = np.empty_like(spectrum)
    
    for i in prange(ny):
        for j in range(nx):
            result[i, j] = spectrum[i, j] / factor
    
    return result


def fft1d(signal: np.ndarray, 
          shift: bool = True,
          normalize: bool = True) -> np.ndarray:
    """
    一维快速傅里叶变换
    
    Args:
        signal: 输入一维信号
        shift: 是否将零频移到中心
        normalize: 是否归一化
        
    Returns:
        频域信号（复数数组）
    """
    spectrum = scipy_fft.fft(signal)
    
    if shift:
        spectrum = _fftshift_1d(spectrum.astype(np.complex128))
    
    if normalize:
        spectrum = spectrum / len(signal)
    
    return spectrum


def ifft1d(spectrum: np.ndarray,
           shifted: bool = True,
           was_normalized: bool = True) -> np.ndarray:
    """
    一维逆傅里叶变换
    
    Args:
        spectrum: 输入频域信号
        shifted: 输入是否已经shift过
        was_normalized: 输入是否已归一化
        
    Returns:
        时域信号
    """
    if shifted:
        spectrum = _ifftshift_1d(spectrum.astype(np.complex128))
    
    signal = scipy_fft.ifft(spectrum)
    
    if was_normalized:
        signal = signal * len(spectrum)
    
    return signal


def fft2d(image: np.ndarray,
          shift: bool = True,
          normalize: bool = True) -> np.ndarray:
    """
    二维快速傅里叶变换
    
    Args:
        image: 输入二维图像
        shift: 是否将零频移到中心
        normalize: 是否归一化
        
    Returns:
        频域图像（复数数组）
    """
    spectrum = scipy_fft.fft2(image)
    
    if shift:
        spectrum = _fftshift_2d(spectrum.astype(np.complex128))
    
    if normalize:
        spectrum = _normalize_spectrum(spectrum.astype(np.complex128), float(image.size))
    
    return spectrum


def ifft2d(spectrum: np.ndarray,
           shifted: bool = True,
           was_normalized: bool = True) -> np.ndarray:
    """
    二维逆傅里叶变换
    
    Args:
        spectrum: 输入频域图像
        shifted: 输入是否已经shift过
        was_normalized: 输入是否已归一化
        
    Returns:
        空域图像
    """
    if shifted:
        spectrum = _ifftshift_2d(spectrum.astype(np.complex128))
    
    image = scipy_fft.ifft2(spectrum)
    
    if was_normalized:
        image = image * spectrum.size
    
    return image


@jit(nopython=True, parallel=True, cache=True)
def _create_circular_mask(shape: Tuple[int, int], 
                          radius: float,
                          center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    创建圆形掩模
    
    Args:
        shape: 图像尺寸 (height, width)
        radius: 圆形半径（像素）
        center: 圆心位置，None则为图像中心
        
    Returns:
        圆形掩模（0-1值）
    """
    ny, nx = shape
    
    if center is None:
        cy, cx = ny // 2, nx // 2
    else:
        cy, cx = center
    
    mask = np.zeros((ny, nx), dtype=np.float64)
    
    for i in prange(ny):
        for j in range(nx):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            if dist <= radius:
                mask[i, j] = 1.0
    
    return mask


def frequency_filter(spectrum: np.ndarray,
                     filter_type: str = 'lowpass',
                     cutoff: float = 0.5,
                     bandwidth: Optional[float] = None,
                     order: int = 2) -> np.ndarray:
    """
    频域滤波器
    
    Args:
        spectrum: 输入频谱（已shift到中心）
        filter_type: 滤波器类型 ('lowpass', 'highpass', 'bandpass', 'bandstop')
        cutoff: 截止频率（归一化，0-1）
        bandwidth: 带宽（仅用于带通/带阻滤波器）
        order: 巴特沃斯滤波器阶数
        
    Returns:
        滤波后的频谱
    """
    ny, nx = spectrum.shape
    cy, cx = ny // 2, nx // 2
    
    # 创建频率网格
    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    X, Y = np.meshgrid(x, y)
    
    # 归一化距离
    max_dist = np.sqrt(cy**2 + cx**2)
    D = np.sqrt(X**2 + Y**2) / max_dist
    
    # 截止频率对应的距离
    D0 = cutoff
    
    # 创建滤波器
    if filter_type == 'lowpass':
        # 巴特沃斯低通滤波器
        H = 1.0 / (1.0 + (D / D0)**(2 * order))
    
    elif filter_type == 'highpass':
        # 巴特沃斯高通滤波器
        H = 1.0 / (1.0 + (D0 / (D + 1e-10))**(2 * order))
    
    elif filter_type == 'bandpass':
        if bandwidth is None:
            bandwidth = 0.1
        W = bandwidth
        H = 1.0 / (1.0 + ((D**2 - D0**2) / (D * W + 1e-10))**(2 * order))
    
    elif filter_type == 'bandstop':
        if bandwidth is None:
            bandwidth = 0.1
        W = bandwidth
        H = 1.0 / (1.0 + ((D * W) / (D**2 - D0**2 + 1e-10))**(2 * order))
    
    else:
        raise ValueError(f"未知的滤波器类型: {filter_type}")
    
    return spectrum * H


@jit(nopython=True, parallel=True, cache=True)
def _apply_phase_array(spectrum: np.ndarray, 
                       phase: np.ndarray) -> np.ndarray:
    """
    应用相位数组到频谱
    
    Args:
        spectrum: 输入频谱
        phase: 相位数组
        
    Returns:
        相位调制后的频谱
    """
    ny, nx = spectrum.shape
    result = np.zeros((ny, nx), dtype=np.complex128)
    
    for i in prange(ny):
        for j in range(nx):
            result[i, j] = spectrum[i, j] * np.exp(1j * phase[i, j])
    
    return result


def phase_modulation(spectrum: np.ndarray,
                     phase_type: str = 'linear',
                     params: Optional[dict] = None) -> np.ndarray:
    """
    相位调制函数
    
    Args:
        spectrum: 输入频谱
        phase_type: 相位类型 ('linear', 'quadratic', 'custom')
        params: 相位参数字典
            - linear: {'kx': float, 'ky': float} 线性相位斜率
            - quadratic: {'alpha': float} 二次相位系数
            - custom: {'phase_array': np.ndarray} 自定义相位数组
            
    Returns:
        相位调制后的频谱
    """
    if params is None:
        params = {}
    
    ny, nx = spectrum.shape
    cy, cx = ny // 2, nx // 2
    
    # 创建坐标网格
    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    X, Y = np.meshgrid(x, y)
    
    if phase_type == 'linear':
        # 线性相位（对应空域平移）
        kx = params.get('kx', 0.0)
        ky = params.get('ky', 0.0)
        phase = 2 * np.pi * (kx * X / nx + ky * Y / ny)
    
    elif phase_type == 'quadratic':
        # 二次相位（对应离焦）
        alpha = params.get('alpha', 0.01)
        phase = alpha * (X**2 + Y**2)
    
    elif phase_type == 'custom':
        # 自定义相位
        phase = params.get('phase_array', np.zeros((ny, nx)))
    
    else:
        raise ValueError(f"未知的相位类型: {phase_type}")
    
    return _apply_phase_array(spectrum, phase.astype(np.float64))


def compute_power_spectrum(image: np.ndarray) -> np.ndarray:
    """
    计算功率谱
    
    Args:
        image: 输入图像
        
    Returns:
        功率谱（对数尺度）
    """
    spectrum = fft2d(image, shift=True, normalize=True)
    power = np.abs(spectrum)**2
    
    # 对数变换以便可视化
    power_log = np.log10(power + 1e-10)
    
    return power_log


def get_frequency_coordinates(shape: Tuple[int, int],
                              pixel_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取频率坐标
    
    Args:
        shape: 图像尺寸 (height, width)
        pixel_size: 像素尺寸
        
    Returns:
        (fx, fy) 频率坐标网格
    """
    ny, nx = shape
    
    fx = scipy_fft.fftshift(scipy_fft.fftfreq(nx, pixel_size))
    fy = scipy_fft.fftshift(scipy_fft.fftfreq(ny, pixel_size))
    
    FX, FY = np.meshgrid(fx, fy)
    
    return FX, FY


def hann_window_2d(shape: Tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    return np.outer(wy, wx).astype(np.float64)


def hamming_window_2d(shape: Tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    wy = np.hamming(ny)
    wx = np.hamming(nx)
    return np.outer(wy, wx).astype(np.float64)


def tukey_window_2d(shape: Tuple[int, int], alpha: float = 0.5) -> np.ndarray:
    ny, nx = shape
    wy = _scipy_tukey(ny, alpha=alpha)
    wx = _scipy_tukey(nx, alpha=alpha)
    return np.outer(wy, wx).astype(np.float64)


def create_window(shape: Tuple[int, int],
                  window_type: Union[WindowType, str] = WindowType.HANN,
                  tukey_alpha: float = 0.5) -> np.ndarray:
    if isinstance(window_type, str):
        window_type = WindowType(window_type)

    if window_type == WindowType.HANN:
        return hann_window_2d(shape)
    elif window_type == WindowType.HAMMING:
        return hamming_window_2d(shape)
    elif window_type == WindowType.TUKEY:
        return tukey_window_2d(shape, alpha=tukey_alpha)
    else:
        raise ValueError(f"未知窗函数类型: {window_type}")


def apply_zero_padding(image: np.ndarray,
                       pad_width: Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = 32,
                       mode: str = 'constant') -> Tuple[np.ndarray, Tuple]:
    if isinstance(pad_width, int):
        pw = ((pad_width, pad_width), (pad_width, pad_width))
    elif isinstance(pad_width, tuple) and len(pad_width) == 2 and isinstance(pad_width[0], int):
        pw = ((pad_width[0], pad_width[0]), (pad_width[1], pad_width[1]))
    else:
        pw = pad_width

    padded = np.pad(image, pw, mode=mode)
    return padded, pw


def remove_padding(padded_image: np.ndarray, pad_width: Tuple) -> np.ndarray:
    top = pad_width[0][0]
    bottom = padded_image.shape[0] - pad_width[0][1]
    left = pad_width[1][0]
    right = padded_image.shape[1] - pad_width[1][1]
    return padded_image[top:bottom, left:right]


def apply_window_and_padding(image: np.ndarray,
                             window_type: Optional[Union[WindowType, str]] = None,
                             pad_width: Optional[Union[int, Tuple[int, int]]] = None,
                             tukey_alpha: float = 0.5) -> Tuple[np.ndarray, dict]:
    if window_type is None and pad_width is None:
        return image.copy(), {'original_shape': image.shape, 'pad_width': ((0, 0), (0, 0)), 'window': None}

    original_shape = image.shape
    processed = image.copy().astype(np.float64)

    if window_type is not None:
        win = create_window(image.shape, window_type, tukey_alpha)
        processed = processed * win
    else:
        win = None

    if pad_width is not None:
        processed, pw = apply_zero_padding(processed, pad_width)
    else:
        pw = ((0, 0), (0, 0))

    info = {
        'original_shape': original_shape,
        'pad_width': pw,
        'window': window_type,
    }
    return processed, info


def crop_to_original(padded_image: np.ndarray, info: dict) -> np.ndarray:
    pw = info['pad_width']
    if pw == ((0, 0), (0, 0)):
        return padded_image.copy()
    return remove_padding(padded_image, pw)
