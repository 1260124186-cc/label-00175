# -*- coding: utf-8 -*-
"""
傅里叶变换模块：FFT/IFFT封装、频域滤波、相位调制

该模块提供掩模图案频域分析所需的傅里叶变换相关函数，包括：
1. 1D/2D快速傅里叶变换及逆变换
2. 频域滤波器（低通、高通、带通）
3. 相位调制函数

核心计算函数支持 CPU (numba加速) 和 GPU (CuPy) 两种后端，
通过 ArrayBackend 统一调度，可通过 device 配置项切换。
"""

import numpy as np
from numba import jit, prange, complex128, float64
from typing import Tuple, Optional, Union
from scipy import fft as scipy_fft
from scipy.signal.windows import tukey as _scipy_tukey
from enum import Enum

from core.array_backend import get_backend, DeviceType


def _use_gpu() -> bool:
    """检查当前是否使用 GPU 后端"""
    return get_backend().device == DeviceType.CUDA


def _asarray(arr):
    """确保数组为当前后端的数组类型"""
    backend = get_backend()
    if isinstance(arr, np.ndarray) and _use_gpu():
        return backend.from_numpy(arr)
    return arr


def _tonumpy(arr):
    """确保返回 numpy 数组（用于对外 API 兼容）"""
    backend = get_backend()
    if _use_gpu():
        return backend.to_numpy(arr)
    return np.asarray(arr)


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
    backend = get_backend()
    x = _asarray(signal)

    spectrum = backend.fft(x)

    if shift:
        spectrum = backend.fftshift(spectrum)

    if normalize:
        n = x.shape[-1]
        spectrum = spectrum / n

    return _tonumpy(spectrum)


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
    backend = get_backend()
    spec = _asarray(spectrum)

    if shifted:
        spec = backend.ifftshift(spec)

    signal = backend.ifft(spec)

    if was_normalized:
        n = spec.shape[-1]
        signal = signal * n

    return _tonumpy(signal)


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
    backend = get_backend()
    x = _asarray(image)

    spectrum = backend.fft2(x)

    if shift:
        spectrum = backend.fftshift(spectrum)

    if normalize:
        spectrum = spectrum / x.size

    return _tonumpy(spectrum)


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
    backend = get_backend()
    spec = _asarray(spectrum)

    if shifted:
        spec = backend.ifftshift(spec)

    image = backend.ifft2(spec)

    if was_normalized:
        image = image * spec.size

    return _tonumpy(image)


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
    backend = get_backend()
    spec = _asarray(spectrum)

    ny, nx = spec.shape
    cy, cx = ny // 2, nx // 2

    y = backend.arange(ny) - cy
    x = backend.arange(nx) - cx
    X, Y = backend.meshgrid(x, y)

    max_dist = backend.sqrt(cy**2 + cx**2)
    D = backend.sqrt(X**2 + Y**2) / max_dist

    D0 = cutoff

    if filter_type == 'lowpass':
        H = 1.0 / (1.0 + (D / D0)**(2 * order))

    elif filter_type == 'highpass':
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

    return _tonumpy(spec * H)


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
    backend = get_backend()
    spec = _asarray(spectrum)

    if params is None:
        params = {}

    ny, nx = spec.shape
    cy, cx = ny // 2, nx // 2

    y = backend.arange(ny) - cy
    x = backend.arange(nx) - cx
    X, Y = backend.meshgrid(x, y)

    if phase_type == 'linear':
        kx = params.get('kx', 0.0)
        ky = params.get('ky', 0.0)
        phase = 2 * backend.pi * (kx * X / nx + ky * Y / ny)

    elif phase_type == 'quadratic':
        alpha = params.get('alpha', 0.01)
        phase = alpha * (X**2 + Y**2)

    elif phase_type == 'custom':
        custom_phase = params.get('phase_array', np.zeros((ny, nx)))
        phase = _asarray(custom_phase)

    else:
        raise ValueError(f"未知的相位类型: {phase_type}")

    result = spec * backend.exp(1j * phase)
    return _tonumpy(result)


def compute_power_spectrum(image: np.ndarray) -> np.ndarray:
    """
    计算功率谱

    Args:
        image: 输入图像

    Returns:
        功率谱（对数尺度）
    """
    backend = get_backend()
    img = _asarray(image)

    spectrum = backend.fftshift(backend.fft2(img)) / img.size
    power = backend.abs(spectrum) ** 2

    power_log = backend.log10(power + 1e-10)

    return _tonumpy(power_log)


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
    backend = get_backend()
    ny, nx = shape

    fx = backend.fftshift(backend.fftfreq(nx, pixel_size))
    fy = backend.fftshift(backend.fftfreq(ny, pixel_size))

    FX, FY = backend.meshgrid(fx, fy)

    return _tonumpy(FX), _tonumpy(FY)


def hann_window_2d(shape: Tuple[int, int]) -> np.ndarray:
    backend = get_backend()
    ny, nx = shape
    wy = backend.xp.hanning(ny)
    wx = backend.xp.hanning(nx)
    return _tonumpy(backend.outer(wy, wx).astype(backend.float64))


def hamming_window_2d(shape: Tuple[int, int]) -> np.ndarray:
    backend = get_backend()
    ny, nx = shape
    wy = backend.xp.hamming(ny)
    wx = backend.xp.hamming(nx)
    return _tonumpy(backend.outer(wy, wx).astype(backend.float64))


def tukey_window_2d(shape: Tuple[int, int], alpha: float = 0.5) -> np.ndarray:
    if _use_gpu():
        backend = get_backend()
        ny, nx = shape

        def _tukey_1d(n, alpha):
            if alpha <= 0:
                return backend.ones(n)
            if alpha >= 1:
                return backend.xp.hanning(n)
            n_taper = int(round(alpha * n / 2))
            w = backend.ones(n)
            t = backend.linspace(0, 1, n_taper, endpoint=False)
            taper = 0.5 * (1 - backend.cos(backend.pi * t / alpha))
            w[:n_taper] = taper
            w[-n_taper:] = taper[::-1]
            return w

        wy = _tukey_1d(ny, alpha)
        wx = _tukey_1d(nx, alpha)
        return _tonumpy(backend.outer(wy, wx).astype(backend.float64))
    else:
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
    backend = get_backend()
    img = _asarray(image)

    if isinstance(pad_width, int):
        pw = ((pad_width, pad_width), (pad_width, pad_width))
    elif isinstance(pad_width, tuple) and len(pad_width) == 2 and isinstance(pad_width[0], int):
        pw = ((pad_width[0], pad_width[0]), (pad_width[1], pad_width[1]))
    else:
        pw = pad_width

    padded = backend.pad(img, pw, mode=mode)
    return _tonumpy(padded), pw


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
    backend = get_backend()
    img = _asarray(image)

    if window_type is None and pad_width is None:
        return _tonumpy(backend.copy(img)), {'original_shape': img.shape, 'pad_width': ((0, 0), (0, 0)), 'window': None}

    original_shape = img.shape
    processed = backend.copy(img).astype(backend.float64)

    if window_type is not None:
        win = _asarray(create_window(img.shape, window_type, tukey_alpha))
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
        return np.array(padded_image).copy() if _use_gpu() else padded_image.copy()
    return remove_padding(padded_image, pw)


class BandlimitType(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    CUSTOM = "custom"
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"
    DIRECTIONAL = "directional"


@jit(nopython=True, parallel=True, cache=True)
def _create_circular_bandlimit(shape: Tuple[int, int],
                          inner_radius: float,
                          outer_radius: float) -> np.ndarray:
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    max_r = np.sqrt(cy**2 + cx**2)
    if max_r <= 0:
        max_r = 1.0
    mask = np.zeros((ny, nx), dtype=np.float64)
    for i in prange(ny):
        for j in prange(nx):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            norm_dist = dist / max_r
            if inner_radius <= norm_dist <= outer_radius:
                mask[i, j] = 1.0
    return mask


@jit(nopython=True, parallel=True, cache=True)
def _create_rectangular_bandlimit(shape: Tuple[int, int],
                                 fx_low: float,
                                 fx_high: float,
                                 fy_low: float,
                                 fy_high: float) -> np.ndarray:
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    mask = np.zeros((ny, nx), dtype=np.float64)
    for i in prange(ny):
        for j in prange(nx):
            fx_norm = (j - cx) / cx if cx > 0 else 0.0
            fy_norm = (i - cy) / cy if cy > 0 else 0.0
            if (fx_low <= abs(fx_norm) <= fx_high and
                fy_low <= abs(fy_norm) <= fy_high):
                mask[i, j] = 1.0
    return mask


@jit(nopython=True, parallel=True, cache=True)
def _create_directional_bandlimit(shape: Tuple[int, int],
                                   inner_radius: float,
                                   outer_radius: float,
                                   angle_min: float,
                                   angle_max: float) -> np.ndarray:
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    max_r = np.sqrt(cy**2 + cx**2)
    if max_r <= 0:
        max_r = 1.0
    mask = np.zeros((ny, nx), dtype=np.float64)
    for i in prange(ny):
        for j in prange(nx):
            dy = i - cy
            dx = j - cx
            dist = np.sqrt(dy**2 + dx**2)
            norm_dist = dist / max_r
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += 2 * np.pi
            if (inner_radius <= norm_dist <= outer_radius):
                angle_condition = (angle_min <= angle <= angle_max)
                if angle_condition:
                    mask[i, j] = 1.0
    return mask


@jit(nopython=True, parallel=True, cache=True)
def _create_smooth_cosine_butterworth_bandlimit(shape: Tuple[int, int],
                                       inner_radius: float,
                                       outer_radius: float,
                                       order: int) -> np.ndarray:
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    max_r = np.sqrt(cy**2 + cx**2)
    if max_r <= 0:
        max_r = 1.0
    mask = np.zeros((ny, nx), dtype=np.float64)
    for i in prange(ny):
        for j in prange(nx):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            norm_dist = dist / max_r
            if outer_radius > 0 and norm_dist <= outer_radius:
                if inner_radius <= 1e-10:
                    denom = 1.0 + (norm_dist / outer_radius)**(2 * order)
                else:
                    width = outer_radius - inner_radius
                    if width <= 1e-10:
                        denom = 1.0 + ((norm_dist - (inner_radius + outer_radius) / 2) / inner_radius) ** (2 * order)
                    else:
                        mid = (inner_radius + outer_radius) / 2
                        denom = 1.0 + ((norm_dist - mid) / (width / 2)) ** (2 * order)
                mask[i, j] = 1.0 / denom
    return mask


def _create_circular_bandlimit_gpu(shape: Tuple[int, int],
                                   inner_radius: float,
                                   outer_radius: float):
    backend = get_backend()
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    max_r = backend.sqrt(cy**2 + cx**2)
    if max_r <= 0:
        max_r = 1.0

    y = backend.arange(ny) - cy
    x = backend.arange(nx) - cx
    X, Y = backend.meshgrid(x, y)
    dist = backend.sqrt(Y**2 + X**2)
    norm_dist = dist / max_r

    mask = backend.zeros((ny, nx), dtype=backend.float64)
    mask[(norm_dist >= inner_radius) & (norm_dist <= outer_radius)] = 1.0
    return mask


def _create_rectangular_bandlimit_gpu(shape: Tuple[int, int],
                                      fx_low: float,
                                      fx_high: float,
                                      fy_low: float,
                                      fy_high: float):
    backend = get_backend()
    ny, nx = shape
    cy, cx = ny // 2, nx // 2

    y = backend.arange(ny) - cy
    x = backend.arange(nx) - cx
    X, Y = backend.meshgrid(x, y)

    fx_norm = X / cx if cx > 0 else backend.zeros_like(X)
    fy_norm = Y / cy if cy > 0 else backend.zeros_like(Y)

    mask = backend.zeros((ny, nx), dtype=backend.float64)
    mask[(backend.abs(fx_norm) >= fx_low) & (backend.abs(fx_norm) <= fx_high) &
         (backend.abs(fy_norm) >= fy_low) & (backend.abs(fy_norm) <= fy_high)] = 1.0
    return mask


def _create_directional_bandlimit_gpu(shape: Tuple[int, int],
                                      inner_radius: float,
                                      outer_radius: float,
                                      angle_min: float,
                                      angle_max: float):
    backend = get_backend()
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    max_r = backend.sqrt(cy**2 + cx**2)
    if max_r <= 0:
        max_r = 1.0

    y = backend.arange(ny) - cy
    x = backend.arange(nx) - cx
    X, Y = backend.meshgrid(x, y)
    dist = backend.sqrt(Y**2 + X**2)
    norm_dist = dist / max_r
    angle = backend.arctan2(Y, X)
    angle = angle % (2 * backend.pi)

    mask = backend.zeros((ny, nx), dtype=backend.float64)
    radial_condition = (norm_dist >= inner_radius) & (norm_dist <= outer_radius)
    if angle_min <= angle_max:
        angle_condition = (angle >= angle_min) & (angle <= angle_max)
    else:
        angle_condition = (angle >= angle_min) | (angle <= angle_max)
    mask[radial_condition & angle_condition] = 1.0
    return mask


def _create_smooth_bandlimit_gpu(shape: Tuple[int, int],
                                 inner_radius: float,
                                 outer_radius: float,
                                 order: int):
    backend = get_backend()
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    max_r = backend.sqrt(cy**2 + cx**2)
    if max_r <= 0:
        max_r = 1.0

    y = backend.arange(ny) - cy
    x = backend.arange(nx) - cx
    X, Y = backend.meshgrid(x, y)
    dist = backend.sqrt(Y**2 + X**2)
    norm_dist = dist / max_r

    mask = backend.zeros((ny, nx), dtype=backend.float64)

    valid = norm_dist <= outer_radius
    if inner_radius <= 1e-10:
        denom = 1.0 + (norm_dist[valid] / outer_radius) ** (2 * order)
    else:
        width = outer_radius - inner_radius
        if width <= 1e-10:
            mid = (inner_radius + outer_radius) / 2
            denom = 1.0 + ((norm_dist[valid] - mid) / inner_radius) ** (2 * order)
        else:
            mid = (inner_radius + outer_radius) / 2
            denom = 1.0 + ((norm_dist[valid] - mid) / (width / 2)) ** (2 * order)
    mask[valid] = 1.0 / denom

    return mask


def create_bandlimit_mask(shape: Tuple[int, int],
                       bandlimit_type: Union[BandlimitType, str] = BandlimitType.LOWPASS,
                       inner_radius: Optional[float] = 0.0,
                       outer_radius: float = 0.5,
                       fx_range: Optional[Tuple[float, float]] = (0.0, 0.5),
                       fy_range: Optional[Tuple[float, float]] = (0.0, 0.5),
                       angle_range: Optional[Tuple[float, float]] = (0.0, 2 * np.pi),
                       smooth: bool = False,
                       order: int = 4,
                       custom_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    创建频域带限约束掩模

    Args:
        shape: 图像尺寸 (height, width)
        bandlimit_type: 带限类型:
            - 'lowpass': 低通（内半径0，外半径cutoff
            - 'highpass': 高通（内半径cutoff，外半径1.0）
            - 'bandpass': 带通（内外半径之间）
            - 'bandstop': 带阻（内外半径之外）
            - 'circular': 圆形带通（等价于bandpass
            - 'rectangular': 矩形带通
            - 'directional': 方向带通
            - 'custom': 自定义掩模
        inner_radius: 内半径（归一化0-1），低通时设为0
        outer_radius: 外半径（归一化0-1）
        fx_range: 矩形带通x方向频率范围 (low, high)
        fy_range: 矩形带通y方向频率范围 (low, high)
        angle_range: 方向带通角度范围 (min, max)，单位弧度
        smooth: 是否使用平滑（巴特沃斯）过渡
        order: 巴特沃斯滤波器阶数（smooth=True时有效）
        custom_mask: 自定义频域掩模（bandlimit_type='custom'时使用）

    Returns:
        频域带限掩模（0-1值）
    """
    backend = get_backend()

    if isinstance(bandlimit_type, str):
        bandlimit_type = BandlimitType(bandlimit_type.lower())

    if bandlimit_type == BandlimitType.CUSTOM:
        if custom_mask is None:
            raise ValueError("custom_mask 不能为空")
        if custom_mask.shape != tuple(shape):
            raise ValueError(f"custom_mask 形状 {custom_mask.shape} 与目标形状 {shape} 不匹配")
        return _tonumpy(_asarray(custom_mask).astype(backend.float64))

    if bandlimit_type == BandlimitType.LOWPASS:
        inner, outer = 0.0, outer_radius
    elif bandlimit_type == BandlimitType.HIGHPASS:
        inner, outer = inner_radius, 1.0
    elif bandlimit_type in (BandlimitType.BANDPASS, BandlimitType.CIRCULAR):
        inner, outer = inner_radius, outer_radius
    elif bandlimit_type == BandlimitType.BANDSTOP:
        if smooth:
            if _use_gpu():
                lowpass = _create_smooth_bandlimit_gpu(shape, 0.0, inner_radius, order)
                highpass = _create_smooth_bandlimit_gpu(shape, outer_radius, 1.0, order)
            else:
                lowpass = _create_smooth_cosine_butterworth_bandlimit(shape, 0.0, inner_radius, order)
                highpass = _create_smooth_cosine_butterworth_bandlimit(shape, outer_radius, 1.0, order)
            return _tonumpy(backend.clip(lowpass + highpass, 0.0, 1.0))
        else:
            if _use_gpu():
                inner_part = _create_circular_bandlimit_gpu(shape, 0.0, inner_radius)
                outer_part = _create_circular_bandlimit_gpu(shape, outer_radius, 1.0)
            else:
                inner_part = _create_circular_bandlimit(shape, 0.0, inner_radius)
                outer_part = _create_circular_bandlimit(shape, outer_radius, 1.0)
            return _tonumpy(backend.clip(inner_part + outer_part, 0.0, 1.0))
    elif bandlimit_type == BandlimitType.RECTANGULAR:
        fx_low, fx_high = fx_range if fx_range is not None else (0.0, 0.5)
        fy_low, fy_high = fy_range if fy_range is not None else (0.0, 0.5)
        if _use_gpu():
            result = _create_rectangular_bandlimit_gpu(shape, fx_low, fx_high, fy_low, fy_high)
        else:
            result = _create_rectangular_bandlimit(shape, fx_low, fx_high, fy_low, fy_high)
        return _tonumpy(result)
    elif bandlimit_type == BandlimitType.DIRECTIONAL:
        a_min, a_max = angle_range if angle_range is not None else (0.0, 2 * np.pi)
        if _use_gpu():
            result = _create_directional_bandlimit_gpu(shape, inner_radius, outer_radius, a_min, a_max)
        else:
            result = _create_directional_bandlimit(shape, inner_radius, outer_radius, a_min, a_max)
        return _tonumpy(result)
    else:
        raise ValueError(f"未知的带限类型: {bandlimit_type}")

    if smooth:
        if _use_gpu():
            result = _create_smooth_bandlimit_gpu(shape, inner, outer, order)
        else:
            result = _create_smooth_cosine_butterworth_bandlimit(shape, inner, outer, order)
    else:
        if _use_gpu():
            result = _create_circular_bandlimit_gpu(shape, inner, outer)
        else:
            result = _create_circular_bandlimit(shape, inner, outer)
    return _tonumpy(result)


def bandlimit_projection(mask: np.ndarray,
                       bandlimit_mask: Optional[np.ndarray],
                       preserve_dc: bool = True) -> np.ndarray:
    """
    频域带限约束投影

    对掩模图案施加频域带限约束，只保留允许的空间频率分量，
    作为优化过程中的投影步骤。

    操作流程：
    1. FFT -> 2. 频谱乘掩模 -> 3. IFFT -> 4. 裁剪到原值范围

    Args:
        mask: 输入掩模图案（空域）
        bandlimit_mask: 频域带限掩模（0-1值）
        preserve_dc: 是否保留直流分量（零频）不受约束影响）

    Returns:
        投影后的掩模图案（空域，实数值）
    """
    backend = get_backend()

    if mask.shape != bandlimit_mask.shape:
        raise ValueError(
            f"掩模形状 {mask.shape} 与频域掩模形状 {bandlimit_mask.shape} 不匹配"
        )

    m = _asarray(mask)
    bl_mask = _asarray(bandlimit_mask)

    spectrum = backend.fftshift(backend.fft2(m)) / m.size

    if preserve_dc:
        ny, nx = m.shape
        cy, cx = ny // 2, nx // 2
        dc_value = spectrum[cy, cx]
        filtered_spectrum = spectrum * bl_mask
        filtered_spectrum[cy, cx] = dc_value
    else:
        filtered_spectrum = spectrum * bl_mask

    projected = backend.ifft2(backend.ifftshift(filtered_spectrum)) * m.size

    return _tonumpy(backend.real(projected).astype(backend.float64))


def bandlimit_projection_simple(mask: np.ndarray,
                            bandlimit_type: Union[BandlimitType, str] = BandlimitType.LOWPASS,
                            cutoff: float = 0.5,
                            bandwidth: Optional[float] = None,
                            smooth: bool = False,
                            order: int = 4,
                            preserve_dc: bool = True) -> np.ndarray:
    """
    简化版频域带限投影接口（自动创建频域掩模）

    Args:
        mask: 输入掩模图案（空域）
        bandlimit_type: 带限类型
        cutoff: 截止频率（归一化0-1）
        bandwidth: 带宽（bandpass/bandstop时使用）
        smooth: 是否平滑过渡
        order: 巴特沃斯阶数
        preserve_dc: 保留直流分量

    Returns:
        投影后的掩模
    """
    if isinstance(bandlimit_type, str):
        bandlimit_type = BandlimitType(bandlimit_type.lower())

    if bandlimit_type == BandlimitType.LOWPASS:
        inner, outer = 0.0, cutoff
    elif bandlimit_type == BandlimitType.HIGHPASS:
        inner, outer = cutoff, 1.0
    elif bandlimit_type in (BandlimitType.BANDPASS, BandlimitType.CIRCULAR):
        bw = bandwidth if bandwidth is not None else 0.1
        inner = max(0.0, cutoff - bw / 2)
        outer = min(1.0, cutoff + bw / 2)
    elif bandlimit_type == BandlimitType.BANDSTOP:
        bw = bandwidth if bandwidth is not None else 0.1
        bl_mask = create_bandlimit_mask(
            mask.shape,
            bandlimit_type=BandlimitType.BANDSTOP,
            inner_radius=max(0.0, cutoff - bw / 2),
            outer_radius=min(1.0, cutoff + bw / 2),
            smooth=smooth,
            order=order
        )
        return bandlimit_projection(mask, bl_mask, preserve_dc=preserve_dc)
    else:
        inner, outer = 0.0, cutoff

    bl_mask = create_bandlimit_mask(
        mask.shape,
        bandlimit_type=BandlimitType.CIRCULAR,
        inner_radius=inner,
        outer_radius=outer,
        smooth=smooth,
        order=order
    )
    return bandlimit_projection(mask, bl_mask, preserve_dc=preserve_dc)


def compute_spectral_energy_ratio(mask: np.ndarray,
                                    bandlimit_mask: np.ndarray) -> Tuple[float, float, float]:
    """
    计算频谱能量在通带内、外及总能量

    Args:
        mask: 输入掩模（空域）
        bandlimit_mask: 频域带限掩模

    Returns:
        (passband_energy_ratio, stopband_energy_ratio, total_energy)
        - passband_energy_ratio: 通带能量占比 (0-1)
        - stopband_energy_ratio: 阻带能量占比 (0-1)
        - total_energy: 总能量
    """
    backend = get_backend()

    m = _asarray(mask)
    bl_mask = _asarray(bandlimit_mask)

    spectrum = backend.fftshift(backend.fft2(m)) / m.size
    power = backend.abs(spectrum) ** 2
    total_energy = float(backend.sum(power))
    if total_energy < 1e-20:
        return 1.0, 0.0, 0.0
    pass_energy = float(backend.sum(power * bl_mask))
    pass_ratio = pass_energy / total_energy
    stop_ratio = 1.0 - pass_ratio
    return pass_ratio, stop_ratio, total_energy


def bandlimited_gradient_projection(gradient: np.ndarray,
                               bandlimit_mask: np.ndarray) -> np.ndarray:
    """
    梯度的频域带限投影（用于投影梯度法）

    对梯度施加同样的频域约束，确保梯度更新方向也满足频域约束。

    Args:
        gradient: 梯度（空域）
        bandlimit_mask: 频域带限掩模

    Returns:
        投影后的梯度
    """
    return bandlimit_projection(gradient, bandlimit_mask, preserve_dc=False)
