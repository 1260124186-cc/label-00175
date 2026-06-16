# -*- coding: utf-8 -*-
"""
数组计算后端抽象层：支持 NumPy (CPU) 和 CuPy (GPU) 可切换

通过统一的 ArrayBackend 接口封装数组操作，使得上层代码（FFT、成像、优化）
可以无缝切换 CPU 和 GPU 计算，无需修改调用方式。

使用方式:
    from core.array_backend import get_backend, set_backend, DeviceType

    # 切换到 GPU
    set_backend(DeviceType.CUDA)

    # 获取当前后端
    backend = get_backend()
    x = backend.zeros((64, 64))
    fft_x = backend.fft2(x)
"""

from enum import Enum
from typing import Tuple, Optional, Union, Any, List
import logging

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """设备类型枚举"""
    CPU = "cpu"
    CUDA = "cuda"


class ArrayBackend:
    """
    数组计算后端抽象基类

    定义统一的数组操作接口，具体实现由 NumpyBackend 和 CupyBackend 提供。
    上层模块应通过此接口进行数组操作，以实现 CPU/GPU 透明切换。
    """

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def device(self) -> DeviceType:
        """当前后端的设备类型"""
        raise NotImplementedError

    @property
    def xp(self):
        """底层数组模块 (numpy 或 cupy)"""
        raise NotImplementedError

    def to_numpy(self, arr) -> 'numpy.ndarray':
        """将数组转换为 NumPy 数组（CPU 模式下为自身）"""
        raise NotImplementedError

    def from_numpy(self, arr: 'numpy.ndarray'):
        """从 NumPy 数组创建后端数组"""
        raise NotImplementedError

    def synchronize(self):
        """同步设备（GPU 模式下等待计算完成，CPU 模式下空操作）"""
        pass

    # ------------------------------------------------------------------
    # 数组创建
    # ------------------------------------------------------------------
    def array(self, obj, dtype=None, copy: bool = True):
        raise NotImplementedError

    def zeros(self, shape, dtype=None):
        raise NotImplementedError

    def ones(self, shape, dtype=None):
        raise NotImplementedError

    def empty(self, shape, dtype=None):
        raise NotImplementedError

    def full(self, shape, fill_value, dtype=None):
        raise NotImplementedError

    def arange(self, start, stop=None, step=None, dtype=None):
        raise NotImplementedError

    def linspace(self, start, stop, num=50, endpoint=True, dtype=None):
        raise NotImplementedError

    def eye(self, N, M=None, k=0, dtype=None):
        raise NotImplementedError

    def zeros_like(self, a, dtype=None):
        raise NotImplementedError

    def ones_like(self, a, dtype=None):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 数组属性与类型
    # ------------------------------------------------------------------
    def shape(self, arr):
        return arr.shape

    def dtype(self, arr):
        return arr.dtype

    def ndim(self, arr):
        return arr.ndim

    def size(self, arr):
        return arr.size

    def astype(self, arr, dtype, copy=True):
        return arr.astype(dtype, copy=copy)

    def iscomplexobj(self, arr):
        raise NotImplementedError

    def isrealobj(self, arr):
        raise NotImplementedError

    def real(self, arr):
        raise NotImplementedError

    def imag(self, arr):
        raise NotImplementedError

    def conj(self, arr):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 数学运算（通用）
    # ------------------------------------------------------------------
    def sqrt(self, x):
        raise NotImplementedError

    def abs(self, x):
        raise NotImplementedError

    def exp(self, x):
        raise NotImplementedError

    def log(self, x):
        raise NotImplementedError

    def log10(self, x):
        raise NotImplementedError

    def sin(self, x):
        raise NotImplementedError

    def cos(self, x):
        raise NotImplementedError

    def tan(self, x):
        raise NotImplementedError

    def arctan2(self, y, x):
        raise NotImplementedError

    def power(self, x, y):
        raise NotImplementedError

    def square(self, x):
        raise NotImplementedError

    def clip(self, a, a_min, a_max):
        raise NotImplementedError

    def where(self, condition, x, y):
        raise NotImplementedError

    def maximum(self, x, y):
        raise NotImplementedError

    def minimum(self, x, y):
        raise NotImplementedError

    def argsort(self, a, axis=-1):
        raise NotImplementedError

    def sort(self, a, axis=-1):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 统计与归约
    # ------------------------------------------------------------------
    def sum(self, a, axis=None, keepdims=False):
        raise NotImplementedError

    def mean(self, a, axis=None, keepdims=False):
        raise NotImplementedError

    def max(self, a, axis=None, keepdims=False):
        raise NotImplementedError

    def min(self, a, axis=None, keepdims=False):
        raise NotImplementedError

    def std(self, a, axis=None, ddof=0):
        raise NotImplementedError

    def var(self, a, axis=None, ddof=0):
        raise NotImplementedError

    def argmin(self, a, axis=None):
        raise NotImplementedError

    def argmax(self, a, axis=None):
        raise NotImplementedError

    def all(self, a, axis=None):
        raise NotImplementedError

    def any(self, a, axis=None):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 数组操作
    # ------------------------------------------------------------------
    def reshape(self, a, newshape):
        raise NotImplementedError

    def transpose(self, a, axes=None):
        raise NotImplementedError

    def ravel(self, a):
        raise NotImplementedError

    def flatten(self, a):
        raise NotImplementedError

    def roll(self, a, shift, axis=None):
        raise NotImplementedError

    def pad(self, array, pad_width, mode='constant', constant_values=0):
        raise NotImplementedError

    def concatenate(self, arrays, axis=0):
        raise NotImplementedError

    def stack(self, arrays, axis=0):
        raise NotImplementedError

    def tile(self, A, reps):
        raise NotImplementedError

    def repeat(self, a, repeats, axis=None):
        raise NotImplementedError

    def flip(self, m, axis=None):
        raise NotImplementedError

    def meshgrid(self, *xi, indexing='xy'):
        raise NotImplementedError

    def outer(self, a, b):
        raise NotImplementedError

    def diag(self, v, k=0):
        raise NotImplementedError

    def take(self, a, indices, axis=None):
        raise NotImplementedError

    def copy(self, a):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 索引相关
    # ------------------------------------------------------------------
    def where_idx(self, condition):
        """类似 np.where，返回索引元组"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 线性代数
    # ------------------------------------------------------------------
    def dot(self, a, b):
        raise NotImplementedError

    def matmul(self, a, b):
        raise NotImplementedError

    def vdot(self, a, b):
        raise NotImplementedError

    def eigh(self, a):
        """Hermitian 矩阵特征值分解，返回 (eigenvalues, eigenvectors)"""
        raise NotImplementedError

    def svd(self, a, full_matrices=True):
        raise NotImplementedError

    def norm(self, x, ord=None, axis=None):
        raise NotImplementedError

    def trace(self, a, offset=0):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # FFT
    # ------------------------------------------------------------------
    def fft(self, a, n=None, axis=-1):
        raise NotImplementedError

    def ifft(self, a, n=None, axis=-1):
        raise NotImplementedError

    def fft2(self, a, s=None, axes=(-2, -1)):
        raise NotImplementedError

    def ifft2(self, a, s=None, axes=(-2, -1)):
        raise NotImplementedError

    def fftn(self, a, s=None, axes=None):
        raise NotImplementedError

    def ifftn(self, a, s=None, axes=None):
        raise NotImplementedError

    def fftshift(self, x, axes=None):
        raise NotImplementedError

    def ifftshift(self, x, axes=None):
        raise NotImplementedError

    def fftfreq(self, n, d=1.0):
        raise NotImplementedError

    def rfft(self, a, n=None, axis=-1):
        raise NotImplementedError

    def irfft(self, a, n=None, axis=-1):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 常数
    # ------------------------------------------------------------------
    @property
    def pi(self):
        raise NotImplementedError

    @property
    def e(self):
        raise NotImplementedError

    @property
    def inf(self):
        raise NotImplementedError

    @property
    def nan(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 数据类型
    # ------------------------------------------------------------------
    @property
    def float32(self):
        raise NotImplementedError

    @property
    def float64(self):
        raise NotImplementedError

    @property
    def complex64(self):
        raise NotImplementedError

    @property
    def complex128(self):
        raise NotImplementedError

    @property
    def int32(self):
        raise NotImplementedError

    @property
    def int64(self):
        raise NotImplementedError

    @property
    def bool_(self):
        raise NotImplementedError


# =====================================================================
# NumPy 后端实现
# =====================================================================
class NumpyBackend(ArrayBackend):
    """
    NumPy CPU 后端实现

    封装 NumPy + SciPy 提供的数组操作，保留 numba 加速路径。
    """

    def __init__(self):
        import numpy as np
        import scipy.fft as scipy_fft
        self._np = np
        self._fft = scipy_fft

    @property
    def device(self) -> DeviceType:
        return DeviceType.CPU

    @property
    def xp(self):
        return self._np

    def to_numpy(self, arr):
        return self._np.asarray(arr)

    def from_numpy(self, arr):
        return self._np.asarray(arr)

    # ------------------------------------------------------------------
    # 数组创建
    # ------------------------------------------------------------------
    def array(self, obj, dtype=None, copy: bool = True):
        return self._np.array(obj, dtype=dtype, copy=copy)

    def zeros(self, shape, dtype=None):
        return self._np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return self._np.ones(shape, dtype=dtype)

    def empty(self, shape, dtype=None):
        return self._np.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None):
        return self._np.full(shape, fill_value, dtype=dtype)

    def arange(self, start, stop=None, step=None, dtype=None):
        return self._np.arange(start, stop, step, dtype=dtype)

    def linspace(self, start, stop, num=50, endpoint=True, dtype=None):
        return self._np.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)

    def eye(self, N, M=None, k=0, dtype=None):
        return self._np.eye(N, M=M, k=k, dtype=dtype)

    def zeros_like(self, a, dtype=None):
        return self._np.zeros_like(a, dtype=dtype)

    def ones_like(self, a, dtype=None):
        return self._np.ones_like(a, dtype=dtype)

    # ------------------------------------------------------------------
    # 数组属性与类型
    # ------------------------------------------------------------------
    def iscomplexobj(self, arr):
        return self._np.iscomplexobj(arr)

    def isrealobj(self, arr):
        return self._np.isrealobj(arr)

    def real(self, arr):
        return self._np.real(arr)

    def imag(self, arr):
        return self._np.imag(arr)

    def conj(self, arr):
        return self._np.conj(arr)

    # ------------------------------------------------------------------
    # 数学运算
    # ------------------------------------------------------------------
    def sqrt(self, x):
        return self._np.sqrt(x)

    def abs(self, x):
        return self._np.abs(x)

    def exp(self, x):
        return self._np.exp(x)

    def log(self, x):
        return self._np.log(x)

    def log10(self, x):
        return self._np.log10(x)

    def sin(self, x):
        return self._np.sin(x)

    def cos(self, x):
        return self._np.cos(x)

    def tan(self, x):
        return self._np.tan(x)

    def arctan2(self, y, x):
        return self._np.arctan2(y, x)

    def power(self, x, y):
        return self._np.power(x, y)

    def square(self, x):
        return self._np.square(x)

    def clip(self, a, a_min, a_max):
        return self._np.clip(a, a_min, a_max)

    def where(self, condition, x, y):
        return self._np.where(condition, x, y)

    def maximum(self, x, y):
        return self._np.maximum(x, y)

    def minimum(self, x, y):
        return self._np.minimum(x, y)

    def argsort(self, a, axis=-1):
        return self._np.argsort(a, axis=axis)

    def sort(self, a, axis=-1):
        return self._np.sort(a, axis=axis)

    # ------------------------------------------------------------------
    # 统计与归约
    # ------------------------------------------------------------------
    def sum(self, a, axis=None, keepdims=False):
        return self._np.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a, axis=None, keepdims=False):
        return self._np.mean(a, axis=axis, keepdims=keepdims)

    def max(self, a, axis=None, keepdims=False):
        return self._np.max(a, axis=axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        return self._np.min(a, axis=axis, keepdims=keepdims)

    def std(self, a, axis=None, ddof=0):
        return self._np.std(a, axis=axis, ddof=ddof)

    def var(self, a, axis=None, ddof=0):
        return self._np.var(a, axis=axis, ddof=ddof)

    def argmin(self, a, axis=None):
        return self._np.argmin(a, axis=axis)

    def argmax(self, a, axis=None):
        return self._np.argmax(a, axis=axis)

    def all(self, a, axis=None):
        return self._np.all(a, axis=axis)

    def any(self, a, axis=None):
        return self._np.any(a, axis=axis)

    # ------------------------------------------------------------------
    # 数组操作
    # ------------------------------------------------------------------
    def reshape(self, a, newshape):
        return self._np.reshape(a, newshape)

    def transpose(self, a, axes=None):
        return self._np.transpose(a, axes=axes)

    def ravel(self, a):
        return self._np.ravel(a)

    def flatten(self, a):
        return a.flatten()

    def roll(self, a, shift, axis=None):
        return self._np.roll(a, shift, axis=axis)

    def pad(self, array, pad_width, mode='constant', constant_values=0):
        return self._np.pad(array, pad_width, mode=mode, constant_values=constant_values)

    def concatenate(self, arrays, axis=0):
        return self._np.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis=0):
        return self._np.stack(arrays, axis=axis)

    def tile(self, A, reps):
        return self._np.tile(A, reps)

    def repeat(self, a, repeats, axis=None):
        return self._np.repeat(a, repeats, axis=axis)

    def flip(self, m, axis=None):
        return self._np.flip(m, axis=axis)

    def meshgrid(self, *xi, indexing='xy'):
        return self._np.meshgrid(*xi, indexing=indexing)

    def outer(self, a, b):
        return self._np.outer(a, b)

    def diag(self, v, k=0):
        return self._np.diag(v, k=k)

    def take(self, a, indices, axis=None):
        return self._np.take(a, indices, axis=axis)

    def copy(self, a):
        return self._np.copy(a)

    # ------------------------------------------------------------------
    # 索引相关
    # ------------------------------------------------------------------
    def where_idx(self, condition):
        return self._np.where(condition)

    # ------------------------------------------------------------------
    # 线性代数
    # ------------------------------------------------------------------
    def dot(self, a, b):
        return self._np.dot(a, b)

    def matmul(self, a, b):
        return self._np.matmul(a, b)

    def vdot(self, a, b):
        return self._np.vdot(a, b)

    def eigh(self, a):
        return self._np.linalg.eigh(a)

    def svd(self, a, full_matrices=True):
        return self._np.linalg.svd(a, full_matrices=full_matrices)

    def norm(self, x, ord=None, axis=None):
        return self._np.linalg.norm(x, ord=ord, axis=axis)

    def trace(self, a, offset=0):
        return self._np.trace(a, offset=offset)

    # ------------------------------------------------------------------
    # FFT (使用 scipy.fft 以获得最佳性能)
    # ------------------------------------------------------------------
    def fft(self, a, n=None, axis=-1):
        return self._fft.fft(a, n=n, axis=axis)

    def ifft(self, a, n=None, axis=-1):
        return self._fft.ifft(a, n=n, axis=axis)

    def fft2(self, a, s=None, axes=(-2, -1)):
        return self._fft.fft2(a, s=s, axes=axes)

    def ifft2(self, a, s=None, axes=(-2, -1)):
        return self._fft.ifft2(a, s=s, axes=axes)

    def fftn(self, a, s=None, axes=None):
        return self._fft.fftn(a, s=s, axes=axes)

    def ifftn(self, a, s=None, axes=None):
        return self._fft.ifftn(a, s=s, axes=axes)

    def fftshift(self, x, axes=None):
        return self._fft.fftshift(x, axes=axes)

    def ifftshift(self, x, axes=None):
        return self._fft.ifftshift(x, axes=axes)

    def fftfreq(self, n, d=1.0):
        return self._fft.fftfreq(n, d=d)

    def rfft(self, a, n=None, axis=-1):
        return self._fft.rfft(a, n=n, axis=axis)

    def irfft(self, a, n=None, axis=-1):
        return self._fft.irfft(a, n=n, axis=axis)

    # ------------------------------------------------------------------
    # 常数
    # ------------------------------------------------------------------
    @property
    def pi(self):
        return self._np.pi

    @property
    def e(self):
        return self._np.e

    @property
    def inf(self):
        return self._np.inf

    @property
    def nan(self):
        return self._np.nan

    # ------------------------------------------------------------------
    # 数据类型
    # ------------------------------------------------------------------
    @property
    def float32(self):
        return self._np.float32

    @property
    def float64(self):
        return self._np.float64

    @property
    def complex64(self):
        return self._np.complex64

    @property
    def complex128(self):
        return self._np.complex128

    @property
    def int32(self):
        return self._np.int32

    @property
    def int64(self):
        return self._np.int64

    @property
    def bool_(self):
        return self._np.bool_


# =====================================================================
# CuPy 后端实现
# =====================================================================
class CupyBackend(ArrayBackend):
    """
    CuPy GPU 后端实现

    使用 CuPy 提供 GPU 加速的数组操作。CuPy API 与 NumPy 高度兼容，
    大多数情况下可以直接替换。
    """

    def __init__(self):
        try:
            import cupy as cp
            import cupyx.scipy.fft as cp_fft
            self._cp = cp
            self._fft = cp_fft
            logger.info("CuPy 后端初始化成功，GPU 计算已启用")
        except ImportError:
            raise ImportError(
                "CuPy 未安装，无法使用 CUDA 后端。"
                "请执行: pip install cupy-cuda12x (根据 CUDA 版本选择)"
            )

    @property
    def device(self) -> DeviceType:
        return DeviceType.CUDA

    @property
    def xp(self):
        return self._cp

    def to_numpy(self, arr):
        return self._cp.asnumpy(arr)

    def from_numpy(self, arr):
        return self._cp.asarray(arr)

    def synchronize(self):
        self._cp.cuda.Stream.null.synchronize()

    # ------------------------------------------------------------------
    # 数组创建
    # ------------------------------------------------------------------
    def array(self, obj, dtype=None, copy: bool = True):
        return self._cp.array(obj, dtype=dtype, copy=copy)

    def zeros(self, shape, dtype=None):
        return self._cp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return self._cp.ones(shape, dtype=dtype)

    def empty(self, shape, dtype=None):
        return self._cp.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None):
        return self._cp.full(shape, fill_value, dtype=dtype)

    def arange(self, start, stop=None, step=None, dtype=None):
        return self._cp.arange(start, stop, step, dtype=dtype)

    def linspace(self, start, stop, num=50, endpoint=True, dtype=None):
        return self._cp.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)

    def eye(self, N, M=None, k=0, dtype=None):
        return self._cp.eye(N, M=M, k=k, dtype=dtype)

    def zeros_like(self, a, dtype=None):
        return self._cp.zeros_like(a, dtype=dtype)

    def ones_like(self, a, dtype=None):
        return self._cp.ones_like(a, dtype=dtype)

    # ------------------------------------------------------------------
    # 数组属性与类型
    # ------------------------------------------------------------------
    def iscomplexobj(self, arr):
        return self._cp.iscomplexobj(arr)

    def isrealobj(self, arr):
        return self._cp.isrealobj(arr)

    def real(self, arr):
        return self._cp.real(arr)

    def imag(self, arr):
        return self._cp.imag(arr)

    def conj(self, arr):
        return self._cp.conj(arr)

    # ------------------------------------------------------------------
    # 数学运算
    # ------------------------------------------------------------------
    def sqrt(self, x):
        return self._cp.sqrt(x)

    def abs(self, x):
        return self._cp.abs(x)

    def exp(self, x):
        return self._cp.exp(x)

    def log(self, x):
        return self._cp.log(x)

    def log10(self, x):
        return self._cp.log10(x)

    def sin(self, x):
        return self._cp.sin(x)

    def cos(self, x):
        return self._cp.cos(x)

    def tan(self, x):
        return self._cp.tan(x)

    def arctan2(self, y, x):
        return self._cp.arctan2(y, x)

    def power(self, x, y):
        return self._cp.power(x, y)

    def square(self, x):
        return self._cp.square(x)

    def clip(self, a, a_min, a_max):
        return self._cp.clip(a, a_min, a_max)

    def where(self, condition, x, y):
        return self._cp.where(condition, x, y)

    def maximum(self, x, y):
        return self._cp.maximum(x, y)

    def minimum(self, x, y):
        return self._cp.minimum(x, y)

    def argsort(self, a, axis=-1):
        return self._cp.argsort(a, axis=axis)

    def sort(self, a, axis=-1):
        return self._cp.sort(a, axis=axis)

    # ------------------------------------------------------------------
    # 统计与归约
    # ------------------------------------------------------------------
    def sum(self, a, axis=None, keepdims=False):
        return self._cp.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a, axis=None, keepdims=False):
        return self._cp.mean(a, axis=axis, keepdims=keepdims)

    def max(self, a, axis=None, keepdims=False):
        return self._cp.max(a, axis=axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        return self._cp.min(a, axis=axis, keepdims=keepdims)

    def std(self, a, axis=None, ddof=0):
        return self._cp.std(a, axis=axis, ddof=ddof)

    def var(self, a, axis=None, ddof=0):
        return self._cp.var(a, axis=axis, ddof=ddof)

    def argmin(self, a, axis=None):
        return self._cp.argmin(a, axis=axis)

    def argmax(self, a, axis=None):
        return self._cp.argmax(a, axis=axis)

    def all(self, a, axis=None):
        return self._cp.all(a, axis=axis)

    def any(self, a, axis=None):
        return self._cp.any(a, axis=axis)

    # ------------------------------------------------------------------
    # 数组操作
    # ------------------------------------------------------------------
    def reshape(self, a, newshape):
        return self._cp.reshape(a, newshape)

    def transpose(self, a, axes=None):
        return self._cp.transpose(a, axes=axes)

    def ravel(self, a):
        return self._cp.ravel(a)

    def flatten(self, a):
        return a.flatten()

    def roll(self, a, shift, axis=None):
        return self._cp.roll(a, shift, axis=axis)

    def pad(self, array, pad_width, mode='constant', constant_values=0):
        return self._cp.pad(array, pad_width, mode=mode, constant_values=constant_values)

    def concatenate(self, arrays, axis=0):
        return self._cp.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis=0):
        return self._cp.stack(arrays, axis=axis)

    def tile(self, A, reps):
        return self._cp.tile(A, reps)

    def repeat(self, a, repeats, axis=None):
        return self._cp.repeat(a, repeats, axis=axis)

    def flip(self, m, axis=None):
        return self._cp.flip(m, axis=axis)

    def meshgrid(self, *xi, indexing='xy'):
        return self._cp.meshgrid(*xi, indexing=indexing)

    def outer(self, a, b):
        return self._cp.outer(a, b)

    def diag(self, v, k=0):
        return self._cp.diag(v, k=k)

    def take(self, a, indices, axis=None):
        return self._cp.take(a, indices, axis=axis)

    def copy(self, a):
        return self._cp.copy(a)

    # ------------------------------------------------------------------
    # 索引相关
    # ------------------------------------------------------------------
    def where_idx(self, condition):
        return self._cp.where(condition)

    # ------------------------------------------------------------------
    # 线性代数
    # ------------------------------------------------------------------
    def dot(self, a, b):
        return self._cp.dot(a, b)

    def matmul(self, a, b):
        return self._cp.matmul(a, b)

    def vdot(self, a, b):
        return self._cp.vdot(a, b)

    def eigh(self, a):
        return self._cp.linalg.eigh(a)

    def svd(self, a, full_matrices=True):
        return self._cp.linalg.svd(a, full_matrices=full_matrices)

    def norm(self, x, ord=None, axis=None):
        return self._cp.linalg.norm(x, ord=ord, axis=axis)

    def trace(self, a, offset=0):
        return self._cp.trace(a, offset=offset)

    # ------------------------------------------------------------------
    # FFT
    # ------------------------------------------------------------------
    def fft(self, a, n=None, axis=-1):
        return self._fft.fft(a, n=n, axis=axis)

    def ifft(self, a, n=None, axis=-1):
        return self._fft.ifft(a, n=n, axis=axis)

    def fft2(self, a, s=None, axes=(-2, -1)):
        return self._fft.fft2(a, s=s, axes=axes)

    def ifft2(self, a, s=None, axes=(-2, -1)):
        return self._fft.ifft2(a, s=s, axes=axes)

    def fftn(self, a, s=None, axes=None):
        return self._fft.fftn(a, s=s, axes=axes)

    def ifftn(self, a, s=None, axes=None):
        return self._fft.ifftn(a, s=s, axes=axes)

    def fftshift(self, x, axes=None):
        return self._fft.fftshift(x, axes=axes)

    def ifftshift(self, x, axes=None):
        return self._fft.ifftshift(x, axes=axes)

    def fftfreq(self, n, d=1.0):
        return self._fft.fftfreq(n, d=d)

    def rfft(self, a, n=None, axis=-1):
        return self._fft.rfft(a, n=n, axis=axis)

    def irfft(self, a, n=None, axis=-1):
        return self._fft.irfft(a, n=n, axis=axis)

    # ------------------------------------------------------------------
    # 常数
    # ------------------------------------------------------------------
    @property
    def pi(self):
        return self._cp.pi

    @property
    def e(self):
        return self._cp.e

    @property
    def inf(self):
        return self._cp.inf

    @property
    def nan(self):
        return self._cp.nan

    # ------------------------------------------------------------------
    # 数据类型
    # ------------------------------------------------------------------
    @property
    def float32(self):
        return self._cp.float32

    @property
    def float64(self):
        return self._cp.float64

    @property
    def complex64(self):
        return self._cp.complex64

    @property
    def complex128(self):
        return self._cp.complex128

    @property
    def int32(self):
        return self._cp.int32

    @property
    def int64(self):
        return self._cp.int64

    @property
    def bool_(self):
        return self._cp.bool_


# =====================================================================
# 全局后端管理
# =====================================================================
_backend: Optional[ArrayBackend] = None


def get_backend() -> ArrayBackend:
    """
    获取当前全局数组计算后端

    Returns:
        当前的 ArrayBackend 实例
    """
    global _backend
    if _backend is None:
        _backend = NumpyBackend()
    return _backend


def set_backend(device: Union[DeviceType, str]) -> ArrayBackend:
    """
    设置全局数组计算后端

    Args:
        device: 设备类型，可以是 DeviceType 枚举或字符串 ('cpu' / 'cuda')

    Returns:
        新的 ArrayBackend 实例

    Raises:
        ValueError: 不支持的设备类型
        ImportError: 请求 CUDA 但 CuPy 未安装
    """
    global _backend

    if isinstance(device, str):
        try:
            device = DeviceType(device.lower())
        except ValueError:
            raise ValueError(f"不支持的设备类型: {device}，支持: cpu, cuda")

    if device == DeviceType.CPU:
        _backend = NumpyBackend()
    elif device == DeviceType.CUDA:
        _backend = CupyBackend()
    else:
        raise ValueError(f"不支持的设备类型: {device}")

    logger.info(f"数组计算后端已切换为: {device.value}")
    return _backend


def get_device() -> DeviceType:
    """获取当前设备类型"""
    return get_backend().device


def is_gpu_available() -> bool:
    """
    检查 GPU (CUDA) 是否可用

    Returns:
        True 表示 CuPy 已安装且 GPU 可用
    """
    try:
        import cupy  # noqa: F401
        return True
    except ImportError:
        return False
