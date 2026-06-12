# -*- coding: utf-8 -*-
"""
误差评估模块：图像误差计算函数

该模块实现了多种图像误差评估指标，用于衡量晶圆成像与目标图像的差异：
1. MSE (均方误差)
2. MAE (平均绝对误差)
3. SSIM (结构相似性)
4. 归一化相关系数
"""

import numpy as np
from numba import jit, prange
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@jit(nopython=True, cache=True)
def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    计算均方误差 (Mean Squared Error)
    
    Args:
        image1: 第一幅图像
        image2: 第二幅图像（目标图像）
        
    Returns:
        MSE值，越小表示越相似
    """
    diff = image1.astype(np.float64) - image2.astype(np.float64)
    return np.mean(diff ** 2)


@jit(nopython=True, cache=True)
def mae(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    Args:
        image1: 第一幅图像
        image2: 第二幅图像（目标图像）
        
    Returns:
        MAE值，越小表示越相似
    """
    diff = image1.astype(np.float64) - image2.astype(np.float64)
    return np.mean(np.abs(diff))


@jit(nopython=True, cache=True)
def _compute_local_stats(image: np.ndarray, 
                         window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算局部均值和方差
    
    Args:
        image: 输入图像
        window_size: 窗口大小
        
    Returns:
        (局部均值, 局部方差)
    """
    ny, nx = image.shape
    half_win = window_size // 2
    
    mean_img = np.zeros((ny, nx), dtype=np.float64)
    var_img = np.zeros((ny, nx), dtype=np.float64)
    
    for i in range(ny):
        for j in range(nx):
            # 确定窗口边界
            y_start = max(0, i - half_win)
            y_end = min(ny, i + half_win + 1)
            x_start = max(0, j - half_win)
            x_end = min(nx, j + half_win + 1)
            
            # 提取窗口
            window_sum = 0.0
            window_sq_sum = 0.0
            count = 0
            
            for yi in range(y_start, y_end):
                for xi in range(x_start, x_end):
                    val = image[yi, xi]
                    window_sum += val
                    window_sq_sum += val * val
                    count += 1
            
            mean_val = window_sum / count
            var_val = window_sq_sum / count - mean_val * mean_val
            
            mean_img[i, j] = mean_val
            var_img[i, j] = max(0.0, var_val)  # 确保非负
    
    return mean_img, var_img


@jit(nopython=True, cache=True)
def _compute_local_covariance(image1: np.ndarray, 
                              image2: np.ndarray,
                              mean1: np.ndarray,
                              mean2: np.ndarray,
                              window_size: int) -> np.ndarray:
    """
    计算局部协方差
    
    Args:
        image1: 第一幅图像
        image2: 第二幅图像
        mean1: 图像1的局部均值
        mean2: 图像2的局部均值
        window_size: 窗口大小
        
    Returns:
        局部协方差图
    """
    ny, nx = image1.shape
    half_win = window_size // 2
    
    cov_img = np.zeros((ny, nx), dtype=np.float64)
    
    for i in range(ny):
        for j in range(nx):
            y_start = max(0, i - half_win)
            y_end = min(ny, i + half_win + 1)
            x_start = max(0, j - half_win)
            x_end = min(nx, j + half_win + 1)
            
            cov_sum = 0.0
            count = 0
            
            for yi in range(y_start, y_end):
                for xi in range(x_start, x_end):
                    cov_sum += (image1[yi, xi] - mean1[i, j]) * (image2[yi, xi] - mean2[i, j])
                    count += 1
            
            cov_img[i, j] = cov_sum / count
    
    return cov_img


def ssim(image1: np.ndarray, 
         image2: np.ndarray,
         window_size: int = 11,
         k1: float = 0.01,
         k2: float = 0.03,
         data_range: float = 1.0) -> float:
    """
    计算结构相似性指数 (Structural Similarity Index)
    
    Args:
        image1: 第一幅图像
        image2: 第二幅图像（目标图像）
        window_size: 滑动窗口大小
        k1: 稳定常数1
        k2: 稳定常数2
        data_range: 数据范围（最大值-最小值）
        
    Returns:
        SSIM值，范围[-1, 1]，越接近1表示越相似
    """
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    
    # 稳定常数
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    # 计算局部统计量
    mean1, var1 = _compute_local_stats(img1, window_size)
    mean2, var2 = _compute_local_stats(img2, window_size)
    cov12 = _compute_local_covariance(img1, img2, mean1, mean2, window_size)
    
    # 计算SSIM
    numerator = (2 * mean1 * mean2 + c1) * (2 * cov12 + c2)
    denominator = (mean1**2 + mean2**2 + c1) * (var1 + var2 + c2)
    
    ssim_map = numerator / (denominator + 1e-10)
    
    return float(np.mean(ssim_map))


@jit(nopython=True, cache=True)
def normalized_correlation(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    计算归一化相关系数 (Normalized Cross-Correlation)
    
    Args:
        image1: 第一幅图像
        image2: 第二幅图像（目标图像）
        
    Returns:
        NCC值，范围[-1, 1]，越接近1表示越相似
    """
    img1 = image1.astype(np.float64).flatten()
    img2 = image2.astype(np.float64).flatten()
    
    # 去均值
    img1_centered = img1 - np.mean(img1)
    img2_centered = img2 - np.mean(img2)
    
    # 计算相关系数
    numerator = np.sum(img1_centered * img2_centered)
    denominator = np.sqrt(np.sum(img1_centered**2) * np.sum(img2_centered**2))
    
    if denominator < 1e-10:
        return 0.0
    
    return numerator / denominator


@jit(nopython=True, cache=True)
def psnr(image1: np.ndarray, image2: np.ndarray, 
         data_range: float = 1.0) -> float:
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)
    
    Args:
        image1: 第一幅图像
        image2: 第二幅图像（目标图像）
        data_range: 数据范围
        
    Returns:
        PSNR值（dB），越大表示越相似
    """
    mse_val = mse(image1, image2)
    
    if mse_val < 1e-10:
        return 100.0  # 完全相同
    
    return 10.0 * np.log10(data_range**2 / mse_val)


@dataclass
class MetricsResult:
    """误差评估结果"""
    mse: float
    mae: float
    ssim: float
    ncc: float
    psnr: float
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'mse': self.mse,
            'mae': self.mae,
            'ssim': self.ssim,
            'ncc': self.ncc,
            'psnr': self.psnr
        }


def evaluate_all(image1: np.ndarray, 
                 image2: np.ndarray,
                 data_range: float = 1.0) -> MetricsResult:
    """
    计算所有误差指标
    
    Args:
        image1: 第一幅图像（预测/优化结果）
        image2: 第二幅图像（目标图像）
        data_range: 数据范围
        
    Returns:
        MetricsResult对象，包含所有指标
    """
    return MetricsResult(
        mse=mse(image1, image2),
        mae=mae(image1, image2),
        ssim=ssim(image1, image2, data_range=data_range),
        ncc=normalized_correlation(image1, image2),
        psnr=psnr(image1, image2, data_range=data_range)
    )


def batch_evaluate(images: List[np.ndarray],
                   target: np.ndarray,
                   metrics: Optional[List[str]] = None) -> List[Dict[str, float]]:
    """
    批量评估多幅图像
    
    Args:
        images: 待评估图像列表
        target: 目标图像
        metrics: 要计算的指标列表，None则计算全部
                 可选: ['mse', 'mae', 'ssim', 'ncc', 'psnr']
        
    Returns:
        评估结果列表，每个元素为指标字典
    """
    if metrics is None:
        metrics = ['mse', 'mae', 'ssim', 'ncc', 'psnr']
    
    metric_funcs = {
        'mse': mse,
        'mae': mae,
        'ssim': ssim,
        'ncc': normalized_correlation,
        'psnr': psnr
    }
    
    results = []
    for img in images:
        result = {}
        for metric_name in metrics:
            if metric_name in metric_funcs:
                result[metric_name] = metric_funcs[metric_name](img, target)
        results.append(result)
    
    return results


def compute_error_map(image1: np.ndarray, 
                      image2: np.ndarray,
                      error_type: str = 'absolute') -> np.ndarray:
    """
    计算误差分布图
    
    Args:
        image1: 第一幅图像
        image2: 第二幅图像（目标）
        error_type: 误差类型 ('absolute', 'squared', 'signed')
        
    Returns:
        误差分布图
    """
    diff = image1.astype(np.float64) - image2.astype(np.float64)
    
    if error_type == 'absolute':
        return np.abs(diff)
    elif error_type == 'squared':
        return diff ** 2
    elif error_type == 'signed':
        return diff
    else:
        raise ValueError(f"未知的误差类型: {error_type}")
