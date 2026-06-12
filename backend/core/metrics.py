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


def total_variation(image: np.ndarray) -> float:
    """
    计算总变差 (Total Variation, TV)，用于衡量图像/掩模复杂度

    TV = Σ |I[i+1,j] - I[i,j]| + |I[i,j+1] - I[i,j]|

    TV 越小表示图像越平滑，越大表示边缘越多、图案越复杂。
    可作为 mask_complexity 损失项。

    Args:
        image: 输入图像（2D 数组）

    Returns:
        总变差标量值
    """
    img = image.astype(np.float64)
    # 行方向差分
    diff_y = np.abs(np.diff(img, axis=0))
    # 列方向差分
    diff_x = np.abs(np.diff(img, axis=1))
    return float(np.sum(diff_y) + np.sum(diff_x))


def total_variation_gradient(image: np.ndarray) -> np.ndarray:
    """
    计算总变差对图像像素的梯度（用于反向传播）

    Args:
        image: 输入图像（2D 数组）

    Returns:
        梯度数组，与 image 形状相同
    """
    img = image.astype(np.float64)
    ny, nx = img.shape
    grad = np.zeros_like(img)

    # 对每个像素，累加四个方向（上、下、左、右）差分的符号贡献
    # dTV/dI[i,j] = sign(I[i,j]-I[i-1,j]) + sign(I[i,j]-I[i+1,j])
    #             + sign(I[i,j]-I[i,j-1]) + sign(I[i,j]-I[i,j+1])
    # 边界处缺失的方向贡献为 0

    # 上邻居
    grad[1:, :] += np.sign(img[1:, :] - img[:-1, :])
    # 下邻居
    grad[:-1, :] += np.sign(img[:-1, :] - img[1:, :])
    # 左邻居
    grad[:, 1:] += np.sign(img[:, 1:] - img[:, :-1])
    # 右邻居
    grad[:, :-1] += np.sign(img[:, :-1] - img[:, 1:])

    return grad


def pvb(images: List[np.ndarray]) -> float:
    """
    计算工艺变化带宽 (Process Variation Band, PVB)

    PVB 衡量多组工艺条件下成像结果的不一致程度：
    PVB = mean_{x,y} ( max_i I_i[x,y] - min_i I_i[x,y] )

    PVB 越小表示工艺鲁棒性越好，不同工艺条件下成像差异越小。

    Args:
        images: 多组工艺条件下的成像结果列表，每个元素为 2D 数组，
                要求所有图像形状相同

    Returns:
        PVB 标量值
    """
    if not images:
        return 0.0

    stack = np.stack([img.astype(np.float64) for img in images], axis=0)
    band = np.max(stack, axis=0) - np.min(stack, axis=0)
    return float(np.mean(band))


def pvb_gradient(images: List[np.ndarray], weights: Optional[List[float]] = None) -> List[np.ndarray]:
    """
    计算 PVB 对各幅图像的梯度（平均带宽对像素的偏导）

    对于 max - min 的子梯度：
    - 等于 max 的图像：梯度为 +1/N（N为像素总数，因为mean）
    - 等于 min 的图像：梯度为 -1/N
    - 其他图像：梯度为 0

    当存在多个像素并列取 max/min 时，梯度在它们之间平均分配。

    Args:
        images: 多组成像结果列表
        weights: 可选，各条件的权重列表（与 images 等长）；
                 None 表示等权重

    Returns:
        梯度列表，每个元素形状与对应 image 相同
    """
    n = len(images)
    if n == 0:
        return []

    if weights is None:
        weights = [1.0] * n

    total_w = sum(weights)
    if total_w <= 0:
        return [np.zeros_like(img) for img in images]

    stack = np.stack([img.astype(np.float64) for img in images], axis=0)  # (n, H, W)
    ny, nx = images[0].shape
    n_pixels = ny * nx

    max_vals = np.max(stack, axis=0)  # (H, W)
    min_vals = np.min(stack, axis=0)  # (H, W)

    grads = [np.zeros((ny, nx), dtype=np.float64) for _ in range(n)]

    for i in range(n):
        w = weights[i] / total_w
        is_max = (stack[i] == max_vals)
        is_min = (stack[i] == min_vals)
        count_max = np.sum(stack == max_vals, axis=0).astype(np.float64)
        count_min = np.sum(stack == min_vals, axis=0).astype(np.float64)
        grads[i][is_max] += w / (n_pixels * count_max[is_max])
        grads[i][is_min] -= w / (n_pixels * count_min[is_min])

    return grads


def l1_regularization(x: np.ndarray) -> float:
    """
    L1 正则化（Lasso）: Σ |x|

    Args:
        x: 参数数组

    Returns:
        L1 范数值
    """
    return float(np.sum(np.abs(x)))


def l1_regularization_gradient(x: np.ndarray) -> np.ndarray:
    """
    L1 正则化梯度: sign(x)
    """
    return np.sign(x.astype(np.float64))


def l2_regularization(x: np.ndarray) -> float:
    """
    L2 正则化（Ridge）: 0.5 * Σ x²

    系数 0.5 使梯度形式更简洁。

    Args:
        x: 参数数组

    Returns:
        L2 正则项值
    """
    return float(0.5 * np.sum(x.astype(np.float64) ** 2))


def l2_regularization_gradient(x: np.ndarray) -> np.ndarray:
    """
    L2 正则化梯度: x
    """
    return x.astype(np.float64)


def tv_regularization(x: np.ndarray) -> float:
    """
    TV 正则化（与 total_variation 相同，作为正则项时使用）

    Args:
        x: 输入图像/掩模（2D 数组）

    Returns:
        TV 值
    """
    return total_variation(x)


def tv_regularization_gradient(x: np.ndarray) -> np.ndarray:
    """
    TV 正则化梯度
    """
    return total_variation_gradient(x)


@dataclass
class CompositeLossComponents:
    """
    复合损失函数各分量值，用于日志和调试
    """
    mse: float = 0.0
    ssim: float = 0.0  # 存储 (1 - SSIM)
    pvb: float = 0.0
    mask_complexity: float = 0.0
    regularization: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'mse': self.mse,
            'ssim_loss': self.ssim,
            'pvb': self.pvb,
            'mask_complexity': self.mask_complexity,
            'regularization': self.regularization,
            'total': self.total
        }
