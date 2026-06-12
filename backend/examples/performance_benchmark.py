#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能测试：核心函数耗时分析

该示例对核心函数（FFT、成像模型、优化迭代）进行性能测试，
并给出numba加速前后的性能对比。

运行方式：
    python -m examples.performance_benchmark
"""

import sys
import os

# 添加父目录到路径（用于本地运行）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import time
from typing import Callable, Tuple
from scipy import fft as scipy_fft

from core.imaging import OpticalSystem, PartialCoherentImaging
from core.fft import fft2d, ifft2d
from core.metrics import mse, mae, ssim
from utils.logger import setup_logger


def benchmark(func: Callable, 
              args: tuple,
              n_runs: int = 10,
              warmup: int = 2) -> Tuple[float, float]:
    """
    性能基准测试
    
    Args:
        func: 待测试函数
        args: 函数参数
        n_runs: 测试次数
        warmup: 预热次数
        
    Returns:
        (平均耗时, 标准差)
    """
    # 预热（触发JIT编译）
    for _ in range(warmup):
        func(*args)
    
    # 正式测试
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)


def benchmark_fft():
    """FFT性能测试"""
    print("\n" + "=" * 60)
    print("FFT性能测试")
    print("=" * 60)
    
    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    
    print(f"\n{'尺寸':<15} {'scipy FFT (ms)':<20} {'封装FFT (ms)':<20}")
    print("-" * 55)
    
    for size in sizes:
        data = np.random.random(size)
        
        # scipy原生FFT
        def scipy_fft_func(x):
            return scipy_fft.fftshift(scipy_fft.fft2(x))
        
        scipy_time, scipy_std = benchmark(scipy_fft_func, (data,))
        
        # 封装的FFT
        wrapped_time, wrapped_std = benchmark(fft2d, (data,))
        
        print(f"{str(size):<15} {scipy_time*1000:>8.3f} ± {scipy_std*1000:.3f}    "
              f"{wrapped_time*1000:>8.3f} ± {wrapped_std*1000:.3f}")


def benchmark_imaging():
    """成像模型性能测试"""
    print("\n" + "=" * 60)
    print("成像模型性能测试")
    print("=" * 60)
    
    sizes = [(64, 64), (128, 128), (256, 256)]
    
    optical_system = OpticalSystem(
        wavelength=193.0,
        na=1.35,
        sigma=0.75
    )
    
    print(f"\n{'尺寸':<15} {'初始化 (ms)':<20} {'成像计算 (ms)':<20}")
    print("-" * 55)
    
    for size in sizes:
        mask = np.random.random(size)
        
        # 初始化耗时
        def init_model():
            return PartialCoherentImaging(optical_system, size)
        
        init_time, init_std = benchmark(init_model, (), n_runs=5, warmup=1)
        
        # 成像计算耗时
        model = PartialCoherentImaging(optical_system, size)
        
        imaging_time, imaging_std = benchmark(
            model.compute_aerial_image, (mask,)
        )
        
        print(f"{str(size):<15} {init_time*1000:>8.3f} ± {init_std*1000:.3f}    "
              f"{imaging_time*1000:>8.3f} ± {imaging_std*1000:.3f}")


def benchmark_metrics():
    """误差指标性能测试"""
    print("\n" + "=" * 60)
    print("误差指标性能测试")
    print("=" * 60)
    
    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    
    print(f"\n{'尺寸':<15} {'MSE (ms)':<15} {'MAE (ms)':<15} {'SSIM (ms)':<15}")
    print("-" * 60)
    
    for size in sizes:
        img1 = np.random.random(size)
        img2 = np.random.random(size)
        
        mse_time, _ = benchmark(mse, (img1, img2))
        mae_time, _ = benchmark(mae, (img1, img2))
        ssim_time, _ = benchmark(ssim, (img1, img2), n_runs=5)
        
        print(f"{str(size):<15} {mse_time*1000:>8.4f}       "
              f"{mae_time*1000:>8.4f}       {ssim_time*1000:>8.3f}")


def benchmark_numba_comparison():
    """Numba加速对比测试"""
    print("\n" + "=" * 60)
    print("Numba加速效果对比")
    print("=" * 60)
    
    # 纯Python实现的MSE（无numba）
    def mse_python(img1, img2):
        diff = img1.astype(np.float64) - img2.astype(np.float64)
        return np.mean(diff ** 2)
    
    # 纯Python实现的MAE（无numba）
    def mae_python(img1, img2):
        diff = img1.astype(np.float64) - img2.astype(np.float64)
        return np.mean(np.abs(diff))
    
    sizes = [(128, 128), (256, 256), (512, 512)]
    
    print(f"\n{'尺寸':<15} {'Python MSE (ms)':<20} {'Numba MSE (ms)':<20} {'加速比':<10}")
    print("-" * 65)
    
    for size in sizes:
        img1 = np.random.random(size)
        img2 = np.random.random(size)
        
        # Python版本
        py_time, _ = benchmark(mse_python, (img1, img2))
        
        # Numba版本
        numba_time, _ = benchmark(mse, (img1, img2))
        
        speedup = py_time / numba_time if numba_time > 0 else float('inf')
        
        print(f"{str(size):<15} {py_time*1000:>10.4f}           "
              f"{numba_time*1000:>10.4f}           {speedup:>6.2f}x")
    
    print("\n" + "-" * 65)
    print(f"{'尺寸':<15} {'Python MAE (ms)':<20} {'Numba MAE (ms)':<20} {'加速比':<10}")
    print("-" * 65)
    
    for size in sizes:
        img1 = np.random.random(size)
        img2 = np.random.random(size)
        
        py_time, _ = benchmark(mae_python, (img1, img2))
        numba_time, _ = benchmark(mae, (img1, img2))
        
        speedup = py_time / numba_time if numba_time > 0 else float('inf')
        
        print(f"{str(size):<15} {py_time*1000:>10.4f}           "
              f"{numba_time*1000:>10.4f}           {speedup:>6.2f}x")


def benchmark_optimization_iteration():
    """优化迭代性能测试"""
    print("\n" + "=" * 60)
    print("优化迭代性能测试")
    print("=" * 60)
    
    from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig
    
    sizes = [(32, 32), (64, 64)]
    
    print(f"\n{'尺寸':<15} {'单次迭代 (ms)':<20} {'10次迭代 (ms)':<20}")
    print("-" * 55)
    
    for size in sizes:
        mask = np.random.random(size)
        target = np.random.random(size)
        
        optical_system = OpticalSystem()
        
        # 单次迭代
        config_1 = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=1,
            verbose=False
        )
        optimizer_1 = MaskOptimizer(optical_system, config_1)
        
        time_1, std_1 = benchmark(
            optimizer_1.optimize, (mask, target),
            n_runs=5, warmup=1
        )
        
        # 10次迭代
        config_10 = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=10,
            verbose=False
        )
        optimizer_10 = MaskOptimizer(optical_system, config_10)
        
        time_10, std_10 = benchmark(
            optimizer_10.optimize, (mask, target),
            n_runs=3, warmup=1
        )
        
        print(f"{str(size):<15} {time_1*1000:>8.2f} ± {std_1*1000:.2f}      "
              f"{time_10*1000:>8.2f} ± {std_10*1000:.2f}")


def main():
    """运行所有性能测试"""
    logger = setup_logger('benchmark')
    
    print("\n" + "=" * 60)
    print("计算光刻仿真框架 - 性能基准测试")
    print("=" * 60)
    
    # 运行各项测试
    benchmark_fft()
    benchmark_imaging()
    benchmark_metrics()
    benchmark_numba_comparison()
    benchmark_optimization_iteration()
    
    print("\n" + "=" * 60)
    print("性能测试完成!")
    print("=" * 60)
    
    # 性能优化建议
    print("\n性能优化建议:")
    print("-" * 40)
    print("1. GPU加速: 使用CuPy替代NumPy进行大规模FFT计算")
    print("2. 批量处理: 对多个掩模同时进行成像计算")
    print("3. 内存复用: 预分配数组避免重复内存分配")
    print("4. 频域近似: 使用TCC分解减少计算量")
    print("5. 并行计算: 使用多进程处理独立的优化任务")


if __name__ == '__main__':
    main()
