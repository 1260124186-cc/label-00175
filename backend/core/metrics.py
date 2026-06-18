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
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging


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
def _ssim_gradient_core(img1: np.ndarray,
                        img2: np.ndarray,
                        mean1: np.ndarray,
                        var1: np.ndarray,
                        mean2: np.ndarray,
                        var2: np.ndarray,
                        cov12: np.ndarray,
                        window_size: int,
                        c1: float,
                        c2: float) -> np.ndarray:
    """
    SSIM 解析梯度核心计算（Numba JIT 编译）

    Args:
        img1: 图像1 (float64)
        img2: 图像2 (float64)
        mean1: 图像1局部均值
        var1: 图像1局部方差
        mean2: 图像2局部均值
        var2: 图像2局部方差
        cov12: 局部协方差
        window_size: 窗口大小
        c1: 稳定常数1
        c2: 稳定常数2

    Returns:
        梯度数组
    """
    ny, nx = img1.shape
    n_total = ny * nx
    half_win = window_size // 2

    A = 2.0 * mean1 * mean2 + c1
    B = 2.0 * cov12 + c2
    C = mean1 ** 2 + mean2 ** 2 + c1
    D = var1 + var2 + c2

    CD = C * D
    denom = CD ** 2 + 1e-10

    dmu_x = (2.0 * mean2 * B * CD - 2.0 * mean1 * A * B * D) / denom
    dvar_x = (-A * B * C) / denom
    dcov_xy = (2.0 * A * CD) / denom

    grad = np.zeros((ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            y_start = max(0, i - half_win)
            y_end = min(ny, i + half_win + 1)
            x_start = max(0, j - half_win)
            x_end = min(nx, j + half_win + 1)
            n_win = (y_end - y_start) * (x_end - x_start)

            alpha = dmu_x[i, j] / n_win
            beta = 2.0 * dvar_x[i, j] / n_win
            gamma = dcov_xy[i, j] / n_win

            for p in range(y_start, y_end):
                for q in range(x_start, x_end):
                    val = (alpha
                           + beta * (img1[p, q] - mean1[i, j])
                           + gamma * (img2[p, q] - mean2[i, j]))
                    grad[p, q] += val

    grad /= n_total
    return grad


def ssim_gradient(image1: np.ndarray,
                  image2: np.ndarray,
                  window_size: int = 11,
                  k1: float = 0.01,
                  k2: float = 0.03,
                  data_range: float = 1.0) -> np.ndarray:
    """
    计算 SSIM 对 image1 像素的解析梯度。

    SSIM 定义:
        SSIM(x,y) = mean_{i,j} ssim_map[i,j]
        ssim_map[i,j] = (A*B) / (C*D)
            A = 2*mu_x*mu_y + C1
            B = 2*sigma_xy + C2
            C = mu_x^2 + mu_y^2 + C1
            D = sigma_x^2 + sigma_y^2 + C2

    梯度通过"窗口反传"计算：每个像素 (p,q) 的梯度等于所有
    以 (i,j) 为中心、且 (p,q) 在窗口内的 ssim_map[i,j] 对 x[p,q]
    的偏导之和，再除以总像素数 (因为 SSIM 是均值)。

    核心循环使用 Numba JIT 编译，相比 O(N²) 数值差分提升数个数量级。

    Args:
        image1: 第一幅图像（变量，求梯度的对象）
        image2: 第二幅图像（目标图像，视为常数）
        window_size: 滑动窗口大小
        k1: 稳定常数1
        k2: 稳定常数2
        data_range: 数据范围

    Returns:
        梯度数组，形状与 image1 相同
    """
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mean1, var1 = _compute_local_stats(img1, window_size)
    mean2, var2 = _compute_local_stats(img2, window_size)
    cov12 = _compute_local_covariance(img1, img2, mean1, mean2, window_size)

    return _ssim_gradient_core(img1, img2, mean1, var1, mean2, var2, cov12,
                               window_size, c1, c2)


def ssim_loss_gradient(image1: np.ndarray,
                       image2: np.ndarray,
                       window_size: int = 11,
                       k1: float = 0.01,
                       k2: float = 0.03,
                       data_range: float = 1.0) -> np.ndarray:
    """
    计算 (1 - SSIM) 损失对 image1 像素的解析梯度。

    因为 d(1-SSIM)/dx = -d(SSIM)/dx，本函数返回 ssim_gradient 的负值。

    Args:
        image1: 第一幅图像（变量）
        image2: 第二幅图像（目标）
        window_size: 滑动窗口大小
        k1: 稳定常数1
        k2: 稳定常数2
        data_range: 数据范围

    Returns:
        (1-SSIM) 的梯度数组
    """
    return -ssim_gradient(image1, image2, window_size, k1, k2, data_range)


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
                 可选: ['mse', 'mae', 'ssim', 'ncc', 'psnr',
                        'mask_complexity', 'tv', 'tv_isotropic',
                        'binary_penalty', 'l0_norm']

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
        'psnr': psnr,
        'mask_complexity': lambda img, _tgt: total_variation(img),
        'tv': lambda img, _tgt: total_variation(img),
        'tv_isotropic': lambda img, _tgt: total_variation_isotropic(img),
        'binary_penalty': lambda img, _tgt: manhattan_distance_penalty(img),
        'l0_norm': lambda img, _tgt: float(np.sum(np.abs(img - 0.5) > 1e-6)),
    }

    results = []
    for img in images:
        result = {}
        for metric_name in metrics:
            if metric_name in metric_funcs:
                result[metric_name] = metric_funcs[metric_name](img, target)
        results.append(result)

    return results


@dataclass
class HistoryEvaluationRow:
    """优化历史中单步评估结果（用于Pareto前沿分析）"""
    step: int
    loss: float
    mse: float
    mae: float
    ssim: float
    ncc: float
    psnr: float
    mask_complexity: float
    tv: float
    binary_penalty: float
    wafer_image: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'step': self.step,
            'loss': self.loss,
            'mse': self.mse,
            'mae': self.mae,
            'ssim': self.ssim,
            'ncc': self.ncc,
            'psnr': self.psnr,
            'mask_complexity': self.mask_complexity,
            'tv': self.tv,
            'binary_penalty': self.binary_penalty,
        }
        return d


def _simulate_wafer(mask: np.ndarray,
                    optical_system: Optional[Any] = None,
                    imaging_model: Optional[Any] = None) -> np.ndarray:
    """将掩模通过光学成像模拟转换为晶圆图像"""
    if imaging_model is not None:
        return imaging_model.compute_aerial_image(mask)
    if optical_system is not None:
        from core.imaging import PartialCoherentImaging
        model = PartialCoherentImaging(optical_system, mask.shape)
        return model.compute_aerial_image(mask)
    return mask.astype(np.float64)


def batch_evaluate_history(
    masks: List[np.ndarray],
    target: np.ndarray,
    loss_history: Optional[List[float]] = None,
    optical_system: Optional[Any] = None,
    imaging_model: Optional[Any] = None,
    include_wafer_images: bool = False,
    extra_metrics: Optional[List[str]] = None,
) -> List[HistoryEvaluationRow]:
    """
    批量评估优化历史中每一步的中间掩模

    对每个中间掩模执行：掩模 → 光学成像 → 精度评估(MSE/MAE/SSIM/NCC/PSNR)
                         → 掩模复杂度评估(TV/二值化惩罚)

    结果可直接用于绘制 Pareto 前沿（精度 vs 掩模复杂度）。

    Args:
        masks: 每一步的中间掩模列表
        target: 目标图像
        loss_history: 可选的损失值历史，与 masks 一一对应
        optical_system: 光学系统参数（用于将掩模投影为晶圆图像）
        imaging_model: 已构建的成像模型实例，优先级高于 optical_system
        include_wafer_images: 是否在结果中保存模拟的晶圆图像（内存开销大）
        extra_metrics: 额外需要计算的精度指标列表

    Returns:
        HistoryEvaluationRow 列表，包含每一步的精度与复杂度指标
    """
    results: List[HistoryEvaluationRow] = []

    for i, mask in enumerate(masks):
        wafer = _simulate_wafer(mask, optical_system, imaging_model)

        prec = evaluate_all(wafer, target)
        mask_complexity = total_variation(mask)
        tv_iso = total_variation_isotropic(mask)
        bin_pen = manhattan_distance_penalty(mask)

        loss_val = loss_history[i] if (loss_history and i < len(loss_history)) else float('nan')

        row = HistoryEvaluationRow(
            step=i,
            loss=loss_val,
            mse=prec.mse,
            mae=prec.mae,
            ssim=prec.ssim,
            ncc=prec.ncc,
            psnr=prec.psnr,
            mask_complexity=mask_complexity,
            tv=tv_iso,
            binary_penalty=bin_pen,
            wafer_image=wafer if include_wafer_images else None,
        )
        results.append(row)

    return results


def export_evaluation_csv(
    rows: List[HistoryEvaluationRow],
    csv_path: Union[str, Path],
    extra_columns: Optional[Dict[str, List[Any]]] = None,
    include_wafer_images: bool = False,
) -> str:
    """
    将批量评估结果导出为 CSV 文件，便于绘制 Pareto 前沿

    导出字段包括：step, loss, mse, mae, ssim, ncc, psnr,
                 mask_complexity, tv, binary_penalty

    Args:
        rows: batch_evaluate_history 的输出
        csv_path: 输出 CSV 文件路径
        extra_columns: 额外需要写入的列，{列名: 与 rows 等长的值列表}
        include_wafer_images: 是否保存晶圆图像路径（暂不支持内嵌图像，仅预留接口）

    Returns:
        实际写入的文件绝对路径
    """
    import csv
    from pathlib import Path

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        'step', 'loss', 'mse', 'mae', 'ssim', 'ncc', 'psnr',
        'mask_complexity', 'tv', 'binary_penalty',
    ]

    all_fields = list(base_fields)
    if extra_columns:
        for col in extra_columns.keys():
            if col not in all_fields:
                all_fields.append(col)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            line = row.to_dict()
            if extra_columns:
                for col, values in extra_columns.items():
                    if i < len(values):
                        line[col] = values[i]
            writer.writerow(line)

    logger = logging.getLogger(__name__)
    logger.info(f"评估结果已导出到 CSV: {csv_path.resolve()}")
    return str(csv_path.resolve())


def compute_pareto_front(
    rows: List[HistoryEvaluationRow],
    objective_x: str = 'mask_complexity',
    objective_y: str = 'mse',
    minimize_x: bool = True,
    minimize_y: bool = True,
) -> List[HistoryEvaluationRow]:
    """
    计算 Pareto 前沿（非支配解集）

    Pareto 最优：在不降低至少一个目标的前提下，无法改进另一个目标。

    Args:
        rows: 批量评估结果
        objective_x: X 轴目标字段名（如 'mask_complexity', 'tv', 'binary_penalty'）
        objective_y: Y 轴目标字段名（如 'mse', 'mae', '1-ssim' 等）
        minimize_x: X 轴目标是否越小越好
        minimize_y: Y 轴目标是否越小越好

    Returns:
        Pareto 前沿上的评估点列表（按 X 升序排列）
    """
    def _better_x(a: float, b: float) -> bool:
        return (a < b) if minimize_x else (a > b)

    def _better_y(a: float, b: float) -> bool:
        return (a < b) if minimize_y else (a > b)

    def _dominates(r1: HistoryEvaluationRow, r2: HistoryEvaluationRow) -> bool:
        x1 = getattr(r1, objective_x)
        y1 = getattr(r1, objective_y)
        x2 = getattr(r2, objective_x)
        y2 = getattr(r2, objective_y)
        no_worse = ((not _better_x(x2, x1)) and (not _better_y(y2, y1)))
        strictly_better = _better_x(x1, x2) or _better_y(y1, y2)
        return no_worse and strictly_better

    front: List[HistoryEvaluationRow] = []
    for r in rows:
        dominated = False
        for other in rows:
            if other is r:
                continue
            if _dominates(other, r):
                dominated = True
                break
        if not dominated:
            front.append(r)

    front.sort(key=lambda r: getattr(r, objective_x))
    return front


def evaluate_and_export_pareto(
    masks: List[np.ndarray],
    target: np.ndarray,
    csv_path: Union[str, Path],
    loss_history: Optional[List[float]] = None,
    optical_system: Optional[Any] = None,
    imaging_model: Optional[Any] = None,
    objective_x: str = 'mask_complexity',
    objective_y: str = 'mse',
    minimize_x: bool = True,
    minimize_y: bool = True,
    pareto_csv_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[HistoryEvaluationRow], List[HistoryEvaluationRow]]:
    """
    一站式：批量评估优化历史 → 导出全量 CSV → 计算 Pareto 前沿 → 导出前沿 CSV

    这是绘制 Pareto 前沿（精度 vs 掩模复杂度）的便捷入口。

    Args:
        masks: 中间掩模列表
        target: 目标图像
        csv_path: 全量评估结果输出 CSV 路径
        loss_history: 可选损失历史
        optical_system: 光学系统参数
        imaging_model: 成像模型实例
        objective_x: Pareto X 轴目标字段
        objective_y: Pareto Y 轴目标字段
        minimize_x: X 轴是否越小越好
        minimize_y: Y 轴是否越小越好
        pareto_csv_path: Pareto 前沿单独输出路径，None 时自动命名

    Returns:
        (所有评估结果, Pareto 前沿结果)
    """
    rows = batch_evaluate_history(
        masks=masks,
        target=target,
        loss_history=loss_history,
        optical_system=optical_system,
        imaging_model=imaging_model,
    )

    export_evaluation_csv(rows, csv_path)

    pareto = compute_pareto_front(
        rows,
        objective_x=objective_x,
        objective_y=objective_y,
        minimize_x=minimize_x,
        minimize_y=minimize_y,
    )

    if pareto_csv_path is None:
        from pathlib import Path
        p = Path(csv_path)
        pareto_csv_path = p.parent / f"{p.stem}_pareto{p.suffix}"

    export_evaluation_csv(pareto, pareto_csv_path)

    return rows, pareto


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


def manhattan_distance_penalty(mask: np.ndarray) -> float:
    """
    曼哈顿距离惩罚（促进二值化）

    惩罚掩模像素值偏离 0 或 1 的程度：
    L1_binary = Σ min(|m_i|, |1 - m_i|) * 2
              = Σ 2 * |m_i - 0.5| - 0.5 （等价形式）
              = Σ 1 - |2 * m_i - 1|

    当 m_i = 0 或 m_i = 1 时，惩罚为 0；
    当 m_i = 0.5 时，惩罚达到最大值 1。

    Args:
        mask: 掩模图案（2D 数组，值范围 [0, 1]）

    Returns:
        平均曼哈顿距离惩罚值（归一化到像素数）
    """
    m = mask.astype(np.float64)
    penalty = 1.0 - np.abs(2.0 * m - 1.0)
    return float(np.mean(penalty))


def manhattan_distance_penalty_gradient(mask: np.ndarray) -> np.ndarray:
    """
    曼哈顿距离惩罚的梯度

    dL/dm_i = -2 * sign(2*m_i - 1)  当 m_i != 0.5
    在 m_i = 0.5 处子梯度可取 [-2, 2] 之间任意值，取 0。

    Args:
        mask: 掩模图案（2D 数组）

    Returns:
        梯度数组，与 mask 形状相同
    """
    m = mask.astype(np.float64)
    diff = 2.0 * m - 1.0
    grad = -2.0 * np.sign(diff)
    grad[np.abs(diff) < 1e-10] = 0.0
    return grad / mask.size


def binary_entropy_penalty(mask: np.ndarray, eps: float = 1e-10) -> float:
    """
    二值熵惩罚（另一种二值化促进方式）

    H(m) = -Σ [m_i * log(m_i) + (1 - m_i) * log(1 - m_i)]

    当 m_i = 0 或 m_i = 1 时，熵为 0；
    当 m_i = 0.5 时，熵达到最大值 log(2)。

    Args:
        mask: 掩模图案（2D 数组，值范围 (0, 1)）
        eps: 数值稳定性小量

    Returns:
        平均二值熵惩罚值
    """
    m = np.clip(mask.astype(np.float64), eps, 1.0 - eps)
    entropy = - (m * np.log(m) + (1.0 - m) * np.log(1.0 - m))
    return float(np.mean(entropy))


def binary_entropy_penalty_gradient(mask: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    二值熵惩罚的梯度

    dH/dm_i = -log(m_i) + log(1 - m_i) = log((1 - m_i) / m_i)

    Args:
        mask: 掩模图案（2D 数组）
        eps: 数值稳定性小量

    Returns:
        梯度数组
    """
    m = np.clip(mask.astype(np.float64), eps, 1.0 - eps)
    grad = np.log((1.0 - m) / m)
    return grad / mask.size


def total_variation_anisotropic(image: np.ndarray) -> float:
    """
    各向异性总变分 (Anisotropic TV) - L1 范数

    TV_L1 = Σ |I[i+1,j] - I[i,j]| + |I[i,j+1] - I[i,j]|

    对水平和垂直差分分别取 L1 范数，更倾向于保留水平/垂直边缘。

    Args:
        image: 输入图像（2D 数组）

    Returns:
        各向异性 TV 值
    """
    img = image.astype(np.float64)
    diff_y = np.abs(np.diff(img, axis=0))
    diff_x = np.abs(np.diff(img, axis=1))
    return float(np.sum(diff_y) + np.sum(diff_x))


def total_variation_isotropic(image: np.ndarray, eps: float = 1e-8) -> float:
    """
    各向同性总变分 (Isotropic TV) - L2 范数

    TV_L2 = Σ sqrt( (I[i+1,j]-I[i,j])² + (I[i,j+1]-I[i,j])² )

    对梯度取 L2 范数，对边缘方向不敏感，更均匀地平滑。

    Args:
        image: 输入图像（2D 数组）
        eps: 数值稳定性小量

    Returns:
        各向同性 TV 值
    """
    img = image.astype(np.float64)
    ny, nx = img.shape
    diff_y = np.zeros_like(img)
    diff_x = np.zeros_like(img)
    diff_y[:-1, :] = np.diff(img, axis=0)
    diff_x[:, :-1] = np.diff(img, axis=1)
    grad_mag = np.sqrt(diff_y**2 + diff_x**2 + eps**2)
    return float(np.sum(grad_mag))


def total_variation_isotropic_gradient(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    各向同性总变分的梯度

    dTV/dI[i,j] = (I[i,j] - I[i-1,j]) / |∇I[i-1,j]|
                + (I[i,j] - I[i+1,j]) / |∇I[i,j]|
                + (I[i,j] - I[i,j-1]) / |∇I[i,j-1]|
                + (I[i,j] - I[i,j+1]) / |∇I[i,j]|

    （边界处缺失项为 0）

    Args:
        image: 输入图像（2D 数组）
        eps: 数值稳定性小量

    Returns:
        梯度数组
    """
    img = image.astype(np.float64)
    ny, nx = img.shape
    grad = np.zeros_like(img)

    diff_y = np.zeros_like(img)
    diff_x = np.zeros_like(img)
    diff_y[:-1, :] = np.diff(img, axis=0)
    diff_x[:, :-1] = np.diff(img, axis=1)
    grad_mag = np.sqrt(diff_y**2 + diff_x**2 + eps**2)

    grad[1:, :] += (img[1:, :] - img[:-1, :]) / grad_mag[:-1, :]
    grad[:-1, :] += (img[:-1, :] - img[1:, :]) / grad_mag[:-1, :]
    grad[:, 1:] += (img[:, 1:] - img[:, :-1]) / grad_mag[:, :-1]
    grad[:, :-1] += (img[:, :-1] - img[:, 1:]) / grad_mag[:, :-1]

    return grad


def compute_edge_map(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    计算图像的二值边缘图（使用 Sobel 算子 + 阈值）

    Args:
        image: 输入图像（2D 数组，值范围 [0, 1]）
        threshold: 边缘阈值

    Returns:
        二值边缘图（1 表示边缘，0 表示非边缘）
    """
    img = image.astype(np.float64)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)

    from scipy.signal import convolve2d
    grad_y = convolve2d(img, sobel_y, mode='same', boundary='symm')
    grad_x = convolve2d(img, sobel_x, mode='same', boundary='symm')
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)

    grad_max = grad_mag.max()
    if grad_max > 0:
        grad_mag = grad_mag / grad_max

    edge_map = (grad_mag >= threshold).astype(np.float64)
    return edge_map


def edge_placement_error(image: np.ndarray,
                         target: np.ndarray,
                         threshold: float = 0.5,
                         pixel_size: float = 1.0) -> float:
    """
    边缘放置误差 (Edge Placement Error, EPE) 近似

    计算预测图像与目标图像的边缘位置之间的平均距离：
    EPE = mean( distance_transform(pred_edge) * target_edge
               + distance_transform(target_edge) * pred_edge )

    这是一个可微近似，使用距离变换衡量边缘对齐程度。

    Args:
        image: 预测图像（2D 数组）
        target: 目标图像（2D 数组）
        threshold: 边缘检测阈值
        pixel_size: 像素尺寸（用于转换为物理距离）

    Returns:
        平均 EPE 值（单位与 pixel_size 一致）
    """
    from scipy.ndimage import distance_transform_edt

    pred_edge = compute_edge_map(image, threshold)
    target_edge = compute_edge_map(target, threshold)

    if np.sum(target_edge) == 0 and np.sum(pred_edge) == 0:
        return 0.0

    dist_pred = distance_transform_edt(1.0 - pred_edge)
    dist_target = distance_transform_edt(1.0 - target_edge)

    epe_val = (np.sum(dist_pred * target_edge) + np.sum(dist_target * pred_edge))
    n_edges = np.sum(target_edge) + np.sum(pred_edge)

    if n_edges > 0:
        epe_val = epe_val / n_edges

    return float(epe_val * pixel_size)


def edge_placement_error_gradient(image: np.ndarray,
                                  target: np.ndarray,
                                  threshold: float = 0.5,
                                  eps: float = 1e-5) -> np.ndarray:
    """
    边缘放置误差（EPE）的数值梯度（近似）

    由于 EPE 涉及离散边缘检测，直接解析梯度困难，
    使用数值差分近似。对于小尺寸掩模可接受。

    Args:
        image: 预测图像（2D 数组，变量）
        target: 目标图像（2D 数组，常数）
        threshold: 边缘检测阈值
        eps: 数值差分步长

    Returns:
        梯度数组
    """
    ny, nx = image.shape
    grad = np.zeros((ny, nx), dtype=np.float64)
    base_epe = edge_placement_error(image, target, threshold)

    for i in range(ny):
        for j in range(nx):
            img_plus = image.copy()
            img_plus[i, j] = min(1.0, img_plus[i, j] + eps)
            epe_plus = edge_placement_error(img_plus, target, threshold)

            img_minus = image.copy()
            img_minus[i, j] = max(0.0, img_minus[i, j] - eps)
            epe_minus = edge_placement_error(img_minus, target, threshold)

            grad[i, j] = (epe_plus - epe_minus) / (2 * eps)

    return grad


def soft_edge_placement_error(image: np.ndarray,
                              target: np.ndarray,
                              sigma: float = 1.0,
                              pixel_size: float = 1.0) -> float:
    """
    软边缘放置误差（可微版本）

    使用高斯滤波的梯度幅值代替二值边缘，避免离散操作，
    使得损失函数处处可微。

    EPE_soft = || gσ * |∇I_pred| - gσ * |∇I_target| ||_1

    其中 gσ 是标准差为 σ 的高斯核。

    Args:
        image: 预测图像（2D 数组）
        target: 目标图像（2D 数组）
        sigma: 高斯平滑标准差（像素）
        pixel_size: 像素尺寸

    Returns:
        软 EPE 值
    """
    from scipy.ndimage import gaussian_filter

    img = image.astype(np.float64)
    tgt = target.astype(np.float64)

    grad_y_pred = np.zeros_like(img)
    grad_x_pred = np.zeros_like(img)
    grad_y_pred[:-1, :] = np.diff(img, axis=0)
    grad_x_pred[:, :-1] = np.diff(img, axis=1)
    grad_mag_pred = np.sqrt(grad_y_pred**2 + grad_x_pred**2)

    grad_y_tgt = np.zeros_like(tgt)
    grad_x_tgt = np.zeros_like(tgt)
    grad_y_tgt[:-1, :] = np.diff(tgt, axis=0)
    grad_x_tgt[:, :-1] = np.diff(tgt, axis=1)
    grad_mag_tgt = np.sqrt(grad_y_tgt**2 + grad_x_tgt**2)

    if sigma > 0:
        grad_mag_pred = gaussian_filter(grad_mag_pred, sigma=sigma)
        grad_mag_tgt = gaussian_filter(grad_mag_tgt, sigma=sigma)

    pred_max = grad_mag_pred.max()
    tgt_max = grad_mag_tgt.max()
    if pred_max > 0:
        grad_mag_pred = grad_mag_pred / pred_max
    if tgt_max > 0:
        grad_mag_tgt = grad_mag_tgt / tgt_max

    epe_soft = np.mean(np.abs(grad_mag_pred - grad_mag_tgt))
    return float(epe_soft * pixel_size)


def soft_edge_placement_error_gradient(image: np.ndarray,
                                       target: np.ndarray,
                                       sigma: float = 1.0) -> np.ndarray:
    """
    软边缘放置误差的解析梯度

    Args:
        image: 预测图像（2D 数组）
        target: 目标图像（2D 数组）
        sigma: 高斯平滑标准差

    Returns:
        梯度数组
    """
    from scipy.ndimage import gaussian_filter

    img = image.astype(np.float64)
    tgt = target.astype(np.float64)
    ny, nx = img.shape

    grad_y_pred = np.zeros_like(img)
    grad_x_pred = np.zeros_like(img)
    grad_y_pred[:-1, :] = np.diff(img, axis=0)
    grad_x_pred[:, :-1] = np.diff(img, axis=1)
    grad_mag_pred = np.sqrt(grad_y_pred**2 + grad_x_pred**2 + 1e-12)

    grad_y_tgt = np.zeros_like(tgt)
    grad_x_tgt = np.zeros_like(tgt)
    grad_y_tgt[:-1, :] = np.diff(tgt, axis=0)
    grad_x_tgt[:, :-1] = np.diff(tgt, axis=1)
    grad_mag_tgt = np.sqrt(grad_y_tgt**2 + grad_x_tgt**2)

    pred_max = grad_mag_pred.max()
    tgt_max = grad_mag_tgt.max()
    if pred_max <= 0:
        pred_max = 1.0
    if tgt_max <= 0:
        tgt_max = 1.0

    grad_mag_pred_norm = grad_mag_pred / pred_max
    grad_mag_tgt_norm = grad_mag_tgt / tgt_max

    if sigma > 0:
        grad_mag_pred_smooth = gaussian_filter(grad_mag_pred_norm, sigma=sigma)
        grad_mag_tgt_smooth = gaussian_filter(grad_mag_tgt_norm, sigma=sigma)
    else:
        grad_mag_pred_smooth = grad_mag_pred_norm
        grad_mag_tgt_smooth = grad_mag_tgt_norm

    sign_diff = np.sign(grad_mag_pred_smooth - grad_mag_tgt_smooth)

    if sigma > 0:
        from scipy.ndimage import correlate
        k_size = int(4 * sigma) + 1
        x = np.arange(-k_size, k_size + 1)
        g_kernel = np.exp(-x**2 / (2 * sigma**2))
        g_kernel = g_kernel / g_kernel.sum()
        g_2d = np.outer(g_kernel, g_kernel)
        backprop_sign = correlate(sign_diff, g_2d, mode='constant')
    else:
        backprop_sign = sign_diff

    backprop_sign = backprop_sign / pred_max

    dmag_dy = grad_y_pred / grad_mag_pred
    dmag_dx = grad_x_pred / grad_mag_pred

    grad = np.zeros_like(img)

    grad_y_contrib = backprop_sign * dmag_dy
    grad[1:, :] += grad_y_contrib[:-1, :]
    grad[:-1, :] -= grad_y_contrib[:-1, :]

    grad_x_contrib = backprop_sign * dmag_dx
    grad[:, 1:] += grad_x_contrib[:, :-1]
    grad[:, :-1] -= grad_x_contrib[:, :-1]

    return grad / (ny * nx)


def _create_structuring_element(kernel_size: int) -> np.ndarray:
    """
    创建形态学操作的结构元素（圆盘形）

    Args:
        kernel_size: 结构元素尺寸（奇数）

    Returns:
        二值结构元素数组
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    r = kernel_size // 2
    y, x = np.ogrid[-r:r+1, -r:r+1]
    selem = (x**2 + y**2) <= r**2
    return selem.astype(np.float64)


def min_feature_size_morphology(mask: np.ndarray,
                                min_size: int = 3,
                                threshold: float = 0.5) -> float:
    """
    最小特征尺寸约束（形态学方法）

    使用开运算（腐蚀+膨胀）检测小于 min_size 的特征：
    - 对掩模的亮区（>= threshold）做开运算，移除小的亮特征
    - 对掩模的暗区（< threshold）做开运算，移除小的暗特征（孔洞）
    惩罚 = || mask - open(mask) ||_1 + || (1-mask) - open(1-mask) ||_1

    这是一个可微的软近似，使用 sigmoid 代替硬阈值。

    Args:
        mask: 掩模图案（2D 数组，值范围 [0, 1]）
        min_size: 最小允许的特征尺寸（像素），对应结构元素半径
        threshold: 二值化阈值

    Returns:
        平均小特征惩罚值
    """
    from scipy.ndimage import binary_opening

    m = mask.astype(np.float64)
    ny, nx = m.shape

    selem = _create_structuring_element(min_size)

    bright_mask = (m >= threshold).astype(np.float64)
    bright_opened = binary_opening(bright_mask > 0.5, structure=selem > 0.5).astype(np.float64)
    bright_penalty = np.sum(np.abs(bright_mask - bright_opened))

    dark_mask = (m < threshold).astype(np.float64)
    dark_opened = binary_opening(dark_mask > 0.5, structure=selem > 0.5).astype(np.float64)
    dark_penalty = np.sum(np.abs(dark_mask - dark_opened))

    total_penalty = (bright_penalty + dark_penalty) / (ny * nx)
    return float(total_penalty)


def soft_min_feature_size_morphology(mask: np.ndarray,
                                     min_size: int = 3,
                                     steepness: float = 10.0) -> float:
    """
    软最小特征尺寸约束（可微形态学近似）

    使用可微的近似腐蚀和膨胀操作：
    - 软腐蚀：局部加权最小值（可用平均池化近似）
    - 软膨胀：局部加权最大值（可用平均池化近似）

    实际使用高斯加权的局部平均来近似形态学操作。

    Args:
        mask: 掩模图案（2D 数组）
        min_size: 最小特征尺寸（像素）
        steepness: sigmoid 陡度参数

    Returns:
        平均小特征惩罚值
    """
    from scipy.ndimage import gaussian_filter

    m = mask.astype(np.float64)
    ny, nx = m.shape

    sigma = min_size / 3.0

    m_sigmoid = 1.0 / (1.0 + np.exp(-steepness * (m - 0.5)))

    smoothed = gaussian_filter(m_sigmoid, sigma=sigma)

    diff = np.abs(m_sigmoid - smoothed)

    high_freq_mask = (diff > 0.1).astype(np.float64)

    penalty = np.sum(diff * high_freq_mask) / (ny * nx)

    return float(penalty)


def soft_min_feature_size_morphology_gradient(mask: np.ndarray,
                                              min_size: int = 3,
                                              steepness: float = 10.0) -> np.ndarray:
    """
    软最小特征尺寸约束的梯度（形态学方法）

    Args:
        mask: 掩模图案（2D 数组）
        min_size: 最小特征尺寸（像素）
        steepness: sigmoid 陡度参数

    Returns:
        梯度数组
    """
    from scipy.ndimage import gaussian_filter, correlate

    m = mask.astype(np.float64)
    ny, nx = m.shape

    sigma = min_size / 3.0

    sig_arg = steepness * (m - 0.5)
    m_sigmoid = 1.0 / (1.0 + np.exp(-sig_arg))
    dsigmoid_dm = steepness * m_sigmoid * (1.0 - m_sigmoid)

    smoothed = gaussian_filter(m_sigmoid, sigma=sigma)

    diff = np.abs(m_sigmoid - smoothed)
    sign_diff = np.sign(m_sigmoid - smoothed)

    high_freq_mask = (diff > 0.1).astype(np.float64)
    dmask_dm = sign_diff * high_freq_mask + diff * 0.0

    k_size = int(4 * sigma) + 1
    x = np.arange(-k_size, k_size + 1)
    g_kernel = np.exp(-x**2 / (2 * sigma**2))
    g_kernel = g_kernel / g_kernel.sum()
    g_2d = np.outer(g_kernel, g_kernel)

    backprop = correlate(dmask_dm, g_2d, mode='constant')

    grad = dsigmoid_dm * (dmask_dm - backprop)

    return grad / (ny * nx)


def min_feature_size_frequency(mask: np.ndarray,
                               min_size: int = 3,
                               pixel_size: float = 1.0) -> float:
    """
    最小特征尺寸约束（频域带限方法）

    惩罚高于截止频率的频谱分量：
    f_cutoff = 1 / (min_size * pixel_size)
    惩罚 = Σ |M(f)| * (|f| > f_cutoff)

    这通过移除过小特征的高频分量来约束最小特征尺寸。

    Args:
        mask: 掩模图案（2D 数组）
        min_size: 最小允许的特征尺寸（物理单位，与 pixel_size 一致）
        pixel_size: 像素尺寸

    Returns:
        高频分量惩罚值（归一化）
    """
    m = mask.astype(np.float64)
    ny, nx = m.shape

    cutoff_freq = 1.0 / (min_size * pixel_size)

    fx = np.fft.fftfreq(nx, pixel_size)
    fy = np.fft.fftfreq(ny, pixel_size)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing='ij')

    freq_mag = np.sqrt(fx_grid**2 + fy_grid**2)
    high_pass_mask = (freq_mag > cutoff_freq).astype(np.float64)

    spectrum = np.fft.fft2(m)
    spectrum_mag = np.abs(spectrum)

    total_high_freq = np.sum(spectrum_mag * high_pass_mask)
    total_energy = np.sum(spectrum_mag) + 1e-12

    penalty = total_high_freq / total_energy
    return float(penalty)


def min_feature_size_frequency_gradient(mask: np.ndarray,
                                        min_size: int = 3,
                                        pixel_size: float = 1.0) -> np.ndarray:
    """
    最小特征尺寸约束的梯度（频域方法）

    Args:
        mask: 掩模图案（2D 数组）
        min_size: 最小特征尺寸
        pixel_size: 像素尺寸

    Returns:
        梯度数组
    """
    m = mask.astype(np.float64)
    ny, nx = m.shape

    cutoff_freq = 1.0 / (min_size * pixel_size)

    fx = np.fft.fftfreq(nx, pixel_size)
    fy = np.fft.fftfreq(ny, pixel_size)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing='ij')

    freq_mag = np.sqrt(fx_grid**2 + fy_grid**2)
    high_pass_mask = (freq_mag > cutoff_freq).astype(np.float64)

    spectrum = np.fft.fft2(m)
    spectrum_mag = np.abs(spectrum) + 1e-12

    total_high_freq = np.sum(spectrum_mag * high_pass_mask)
    total_energy = np.sum(spectrum_mag) + 1e-12

    dtotal_df = high_pass_mask / spectrum_mag * spectrum
    dtotal_de = spectrum / spectrum_mag

    grad_spectrum = (dtotal_df * total_energy - total_high_freq * dtotal_de) / (total_energy**2)

    grad_space = np.real(np.fft.ifft2(grad_spectrum))

    return grad_space


def min_feature_size_combined(mask: np.ndarray,
                              min_size: int = 3,
                              pixel_size: float = 1.0,
                              alpha: float = 0.5) -> float:
    """
    最小特征尺寸约束（形态学+频域联合）

    结合形态学和频域两种方法的优势：
    - 形态学：准确检测空间域中的小特征
    - 频域：平滑的梯度，易于优化

    Args:
        mask: 掩模图案（2D 数组）
        min_size: 最小特征尺寸
        pixel_size: 像素尺寸
        alpha: 形态学权重，(1-alpha) 为频域权重

    Returns:
        联合惩罚值
    """
    penalty_morph = soft_min_feature_size_morphology(mask, min_size)
    penalty_freq = min_feature_size_frequency(mask, min_size, pixel_size)
    return float(alpha * penalty_morph + (1 - alpha) * penalty_freq)


def min_feature_size_combined_gradient(mask: np.ndarray,
                                       min_size: int = 3,
                                       pixel_size: float = 1.0,
                                       alpha: float = 0.5) -> np.ndarray:
    """
    最小特征尺寸约束的梯度（联合方法）

    Args:
        mask: 掩模图案（2D 数组）
        min_size: 最小特征尺寸
        pixel_size: 像素尺寸
        alpha: 形态学权重

    Returns:
        梯度数组
    """
    grad_morph = soft_min_feature_size_morphology_gradient(mask, min_size)
    grad_freq = min_feature_size_frequency_gradient(mask, min_size, pixel_size)
    return alpha * grad_morph + (1 - alpha) * grad_freq


@dataclass
class SpatialWeightConfig:
    """
    空间权重mask配置

    用于在关键区域（如线端、拐角）设置更高权重，使优化更关注热点区域。

    Attributes:
        enable: 是否启用空间加权
        edge_weight: 边缘区域权重倍率
        corner_weight: 拐角区域权重倍率
        line_end_weight: 线端区域权重倍率
        base_weight: 基础区域权重
        edge_sigma: 边缘检测的高斯平滑sigma（像素）
        corner_threshold: 拐角检测阈值（0-1，越大越严格）
        line_end_threshold: 线端检测阈值
        weight_erosion: 是否对权重做形态学腐蚀以避免边界伪影
        smooth_sigma: 权重mask的高斯平滑sigma（使权重过渡更平滑）
        normalize: 是否将权重归一化到均值为1
    """
    enable: bool = False
    edge_weight: float = 2.0
    corner_weight: float = 5.0
    line_end_weight: float = 4.0
    base_weight: float = 1.0
    edge_sigma: float = 1.0
    corner_threshold: float = 0.3
    line_end_threshold: float = 0.5
    weight_erosion: bool = True
    smooth_sigma: float = 0.5
    normalize: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'SpatialWeightConfig':
        if d is None:
            return cls()
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enable': self.enable,
            'edge_weight': self.edge_weight,
            'corner_weight': self.corner_weight,
            'line_end_weight': self.line_end_weight,
            'base_weight': self.base_weight,
            'edge_sigma': self.edge_sigma,
            'corner_threshold': self.corner_threshold,
            'line_end_threshold': self.line_end_threshold,
            'weight_erosion': self.weight_erosion,
            'smooth_sigma': self.smooth_sigma,
            'normalize': self.normalize
        }


def _detect_edges(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    使用Sobel算子检测边缘（连续值版本，可微）

    Args:
        image: 输入图像（2D数组）
        sigma: 高斯预平滑sigma

    Returns:
        边缘强度图（0-1之间）
    """
    from scipy.ndimage import gaussian_filter
    img = image.astype(np.float64)
    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64) / 8.0
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0

    from scipy.signal import convolve2d
    grad_y = convolve2d(img, sobel_y, mode='same', boundary='symm')
    grad_x = convolve2d(img, sobel_x, mode='same', boundary='symm')
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)

    grad_max = grad_mag.max()
    if grad_max > 0:
        grad_mag = grad_mag / grad_max
    return grad_mag


def _detect_corners(image: np.ndarray,
                    sigma: float = 1.0,
                    threshold: float = 0.3) -> np.ndarray:
    """
    使用Harris角点检测的简化可微版本检测拐角

    Args:
        image: 输入图像
        sigma: 高斯平滑sigma
        threshold: 拐角阈值（0-1）

    Returns:
        拐角强度图（0-1之间）
    """
    from scipy.ndimage import gaussian_filter
    img = image.astype(np.float64)
    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64) / 8.0
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0

    from scipy.signal import convolve2d
    Iy = convolve2d(img, sobel_y, mode='same', boundary='symm')
    Ix = convolve2d(img, sobel_x, mode='same', boundary='symm')

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    window = np.ones((3, 3), dtype=np.float64)
    Sxx = convolve2d(Ixx, window, mode='same', boundary='symm')
    Syy = convolve2d(Iyy, window, mode='same', boundary='symm')
    Sxy = convolve2d(Ixy, window, mode='same', boundary='symm')

    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy + 1e-12
    k = 0.04
    harris = det - k * trace * trace

    h_max = harris.max()
    if h_max > 0:
        harris_norm = np.clip(harris / h_max, 0.0, 1.0)
    else:
        harris_norm = np.zeros_like(harris)

    corner_map = np.where(harris_norm > threshold, harris_norm, 0.0)
    if sigma > 0:
        corner_map = gaussian_filter(corner_map, sigma=sigma)
        c_max = corner_map.max()
        if c_max > 0:
            corner_map = corner_map / c_max

    return corner_map


def _detect_line_ends(image: np.ndarray,
                      threshold: float = 0.5,
                      sigma: float = 1.0) -> np.ndarray:
    """
    检测线端区域

    通过分析局部邻域内亮像素的数量和分布来识别线端：
    - 线端是亮像素的"末端"，一侧有亮像素，另一侧没有
    - 使用形态学方法：端点 = (原图 - 原图腐蚀后膨胀) 与边缘的交集

    Args:
        image: 输入图像（二值或接近二值）
        threshold: 二值化阈值
        sigma: 平滑sigma

    Returns:
        线端强度图（0-1之间）
    """
    from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

    img = image.astype(np.float64)
    binary = (img >= threshold).astype(np.float64)

    struct_3x3 = np.ones((3, 3), dtype=bool)
    struct_plus = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    eroded = binary_erosion(binary > 0.5, structure=struct_3x3, iterations=1).astype(np.float64)
    dilated_eroded = binary_dilation(eroded > 0.5, structure=struct_plus, iterations=1).astype(np.float64)

    endpoints_raw = binary - dilated_eroded
    endpoints_raw = np.clip(endpoints_raw, 0.0, 1.0)

    from scipy.signal import convolve2d
    neighbor_count = convolve2d(binary, struct_plus.astype(np.float64), mode='same', boundary='symm')
    is_endpoint_candidate = (binary > 0.5) & (neighbor_count >= 2) & (neighbor_count <= 4)

    line_end_map = endpoints_raw * is_endpoint_candidate.astype(np.float64)

    if sigma > 0:
        line_end_map = gaussian_filter(line_end_map, sigma=sigma)
        le_max = line_end_map.max()
        if le_max > 0:
            line_end_map = line_end_map / le_max

    return line_end_map


def generate_spatial_weight_mask(target_image: np.ndarray,
                                 config: Optional[SpatialWeightConfig] = None) -> np.ndarray:
    """
    生成空间权重mask

    在关键区域（边缘、拐角、线端）分配更高权重，使优化更关注这些热点区域。

    权重计算公式:
        W(x,y) = base_weight
               + edge_weight * edge_map(x,y)
               + corner_weight * corner_map(x,y)
               + line_end_weight * line_end_map(x,y)

    然后可选做平滑和归一化。

    Args:
        target_image: 目标图像（用于分析关键区域）
        config: 空间权重配置，None则使用默认配置

    Returns:
        权重mask数组，与target_image形状相同
    """
    if config is None:
        config = SpatialWeightConfig()

    if not config.enable:
        return np.ones_like(target_image, dtype=np.float64)

    img = target_image.astype(np.float64)
    ny, nx = img.shape

    edge_map = _detect_edges(img, sigma=config.edge_sigma)
    corner_map = _detect_corners(img, sigma=config.edge_sigma, threshold=config.corner_threshold)
    line_end_map = _detect_line_ends(img, threshold=config.line_end_threshold, sigma=config.edge_sigma)

    weights = (config.base_weight
               + config.edge_weight * edge_map
               + config.corner_weight * corner_map
               + config.line_end_weight * line_end_map)

    if config.weight_erosion:
        from scipy.ndimage import minimum_filter
        weights = minimum_filter(weights, size=3)

    if config.smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        weights = gaussian_filter(weights, sigma=config.smooth_sigma)

    if config.normalize:
        mean_w = weights.mean()
        if mean_w > 0:
            weights = weights / mean_w

    weights = np.maximum(weights, config.base_weight)
    return weights


def weighted_mse(image1: np.ndarray,
                 image2: np.ndarray,
                 weight_mask: np.ndarray) -> float:
    """
    计算加权均方误差 (Weighted Mean Squared Error)

    WMSE = (Σ W[i,j] * (I1[i,j] - I2[i,j])²) / (Σ W[i,j])

    其中权重已归一化到总和为像素数，保证量级与普通MSE一致。

    Args:
        image1: 第一幅图像
        image2: 第二幅图像（目标图像）
        weight_mask: 权重mask（与图像形状相同）

    Returns:
        加权MSE值
    """
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    w = weight_mask.astype(np.float64)

    diff_sq = (img1 - img2) ** 2
    total_w = w.sum()
    if total_w <= 0:
        return float(np.mean(diff_sq))

    w_normalized = w * (img1.size / total_w)
    return float(np.sum(w_normalized * diff_sq) / img1.size)


def weighted_mae(image1: np.ndarray,
                 image2: np.ndarray,
                 weight_mask: np.ndarray) -> float:
    """
    计算加权平均绝对误差 (Weighted Mean Absolute Error)

    WMAE = (Σ W[i,j] * |I1[i,j] - I2[i,j]|) / (Σ W[i,j])

    Args:
        image1: 第一幅图像
        image2: 第二幅图像（目标图像）
        weight_mask: 权重mask

    Returns:
        加权MAE值
    """
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    w = weight_mask.astype(np.float64)

    diff_abs = np.abs(img1 - img2)
    total_w = w.sum()
    if total_w <= 0:
        return float(np.mean(diff_abs))

    w_normalized = w * (img1.size / total_w)
    return float(np.sum(w_normalized * diff_abs) / img1.size)


def weighted_mse_gradient(image1: np.ndarray,
                          image2: np.ndarray,
                          weight_mask: np.ndarray) -> np.ndarray:
    """
    计算加权MSE对image1像素的解析梯度

    d(WMSE)/dI1[i,j] = 2 * W[i,j] * (I1[i,j] - I2[i,j]) / (Σ W[k,l])
                      * N  （N为像素总数，保持量级与普通MSE梯度一致）

    Args:
        image1: 第一幅图像（变量，求梯度的对象）
        image2: 第二幅图像（目标图像，视为常数）
        weight_mask: 权重mask

    Returns:
        梯度数组，形状与image1相同
    """
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    w = weight_mask.astype(np.float64)

    n = img1.size
    total_w = w.sum()
    if total_w <= 0:
        w_normalized = np.ones_like(w)
    else:
        w_normalized = w * (n / total_w)

    return 2.0 * w_normalized * (img1 - img2) / n


def weighted_mae_gradient(image1: np.ndarray,
                          image2: np.ndarray,
                          weight_mask: np.ndarray) -> np.ndarray:
    """
    计算加权MAE对image1像素的解析梯度

    d(WMAE)/dI1[i,j] = W[i,j] * sign(I1[i,j] - I2[i,j]) / (Σ W[k,l])
                      * N  （归一化）

    Args:
        image1: 第一幅图像（变量）
        image2: 第二幅图像（目标）
        weight_mask: 权重mask

    Returns:
        梯度数组
    """
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    w = weight_mask.astype(np.float64)

    n = img1.size
    total_w = w.sum()
    if total_w <= 0:
        w_normalized = np.ones_like(w)
    else:
        w_normalized = w * (n / total_w)

    diff = img1 - img2
    sign_diff = np.sign(diff)
    sign_diff[np.abs(diff) < 1e-12] = 0.0

    return w_normalized * sign_diff / n


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
    binary_penalty: float = 0.0
    tv_smooth: float = 0.0
    epe: float = 0.0
    min_feature: float = 0.0
    weighted_mse: float = 0.0
    weighted_mae: float = 0.0
    cd_error: float = 0.0
    litho_epe: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'mse': self.mse,
            'ssim_loss': self.ssim,
            'pvb': self.pvb,
            'mask_complexity': self.mask_complexity,
            'regularization': self.regularization,
            'binary_penalty': self.binary_penalty,
            'tv_smooth': self.tv_smooth,
            'epe': self.epe,
            'min_feature': self.min_feature,
            'weighted_mse': self.weighted_mse,
            'weighted_mae': self.weighted_mae,
            'cd_error': self.cd_error,
            'litho_epe': self.litho_epe,
            'total': self.total
        }
