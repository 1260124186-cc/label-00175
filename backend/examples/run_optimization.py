#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
掩模优化示例：完整的优化流程演示

该示例展示如何使用框架完成：
1. 创建测试掩模图案
2. 光学成像模拟
3. 误差计算
4. 优化迭代
5. 结果可视化

运行方式：
    python -m examples.run_optimization
"""

import sys
import os

# 添加父目录到路径（用于本地运行）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from pathlib import Path

from core.imaging import OpticalSystem, PartialCoherentImaging, simulate_wafer_image
from core.fft import fft2d, compute_power_spectrum
from core.metrics import evaluate_all, mse
from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig
from utils.data_io import create_test_pattern
from utils.visualization import (
    plot_mask, plot_frequency_domain, plot_wafer_image,
    plot_error_curve, plot_comparison, plot_optimization_summary
)
from utils.logger import setup_logger, OptimizationLogger
from utils.config import save_results, create_default_config, save_config


def main():
    """主函数：运行完整的掩模优化示例"""
    
    # 设置日志
    logger = setup_logger('litho_sim', log_file='results/optimization.log')
    logger.info("=" * 60)
    logger.info("计算光刻掩模优化示例")
    logger.info("=" * 60)
    
    # 创建输出目录
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # ========== 1. 创建测试图案 ==========
    logger.info("步骤1: 创建测试图案")
    
    # 目标图案：简单矩形
    image_size = (64, 64)  # 使用较小尺寸以加快演示
    target_pattern = create_test_pattern(
        'rectangle',
        size=image_size,
        x_start=20, x_end=44,
        y_start=20, y_end=44
    )
    
    # 初始掩模：与目标相同（实际应用中可能不同）
    initial_mask = target_pattern.copy()
    
    logger.info(f"图像尺寸: {image_size}")
    logger.info(f"目标图案: 矩形 (20:44, 20:44)")
    
    # ========== 2. 配置光学系统 ==========
    logger.info("步骤2: 配置光学系统")
    
    optical_system = OpticalSystem(
        wavelength=193.0,  # ArF光源
        na=1.35,           # 高NA浸没式
        sigma=0.75,        # 部分相干
        pixel_size=1.0,
        defocus=0.0
    )
    
    logger.info(f"波长: {optical_system.wavelength} nm")
    logger.info(f"数值孔径: {optical_system.na}")
    logger.info(f"部分相干因子: {optical_system.sigma}")
    
    # ========== 3. 成像模拟 ==========
    logger.info("步骤3: 光学成像模拟")
    
    # 创建成像模型
    imaging_model = PartialCoherentImaging(optical_system, image_size)
    
    # 计算初始成像
    initial_wafer_image = imaging_model.compute_aerial_image(initial_mask)
    
    # 计算初始误差
    initial_metrics = evaluate_all(initial_wafer_image, target_pattern)
    logger.info(f"初始MSE: {initial_metrics.mse:.6e}")
    logger.info(f"初始SSIM: {initial_metrics.ssim:.4f}")
    
    # ========== 4. 频域分析 ==========
    logger.info("步骤4: 频域分析")
    
    # 计算掩模频谱
    mask_spectrum = fft2d(initial_mask, shift=True)
    power_spectrum = compute_power_spectrum(initial_mask)
    
    # ========== 5. 掩模优化 ==========
    logger.info("步骤5: 掩模优化")
    
    # 配置优化参数
    opt_config = OptimizationConfig(
        optimizer_type='gradient_descent',
        max_iter=50,
        learning_rate=0.1,
        tol=1e-7,
        early_stop_patience=10,
        metric='mse',
        bounds=(0.0, 1.0),
        verbose=True
    )
    
    # 创建优化器
    mask_optimizer = MaskOptimizer(
        optical_system=optical_system,
        config=opt_config
    )
    
    # 为了演示优化效果，我们添加一些噪声到初始掩模
    noisy_mask = initial_mask + 0.1 * np.random.randn(*image_size)
    noisy_mask = np.clip(noisy_mask, 0, 1)
    
    # 执行优化
    logger.info("开始优化迭代...")
    result = mask_optimizer.optimize(
        initial_mask=noisy_mask,
        target_image=target_pattern
    )
    
    logger.info(f"优化完成!")
    logger.info(f"最终MSE: {result.final_metrics.mse:.6e}")
    logger.info(f"最终SSIM: {result.final_metrics.ssim:.4f}")
    logger.info(f"总迭代次数: {result.total_iterations}")
    logger.info(f"总耗时: {result.total_time:.2f} 秒")
    
    # ========== 6. 结果可视化 ==========
    logger.info("步骤6: 结果可视化")
    
    # 绘制优化汇总图
    plot_optimization_summary(
        result,
        save_path=str(output_dir / 'optimization_summary.png'),
        show=False
    )
    
    # 绘制收敛曲线
    plot_error_curve(
        result.loss_history,
        title='优化收敛曲线',
        log_scale=True,
        save_path=str(output_dir / 'convergence_curve.png'),
        show=False
    )
    
    # 绘制频域分布
    plot_frequency_domain(
        mask_spectrum,
        title='掩模频谱',
        save_path=str(output_dir / 'frequency_domain.png'),
        show=False
    )
    
    # 绘制前后对比
    plot_comparison(
        images=[
            result.initial_mask,
            result.optimized_mask,
            result.target_image,
            result.initial_wafer_image,
            result.final_wafer_image
        ],
        titles=[
            '初始掩模',
            '优化后掩模',
            '目标图像',
            '初始成像',
            '最终成像'
        ],
        main_title='掩模优化前后对比',
        save_path=str(output_dir / 'comparison.png'),
        show=False
    )
    
    # ========== 7. 保存结果 ==========
    logger.info("步骤7: 保存结果")
    
    # 保存配置
    config = create_default_config()
    save_config(config, output_dir / 'config.yaml')
    
    # 保存优化结果
    results_dict = {
        'initial_mse': result.initial_metrics.mse,
        'final_mse': result.final_metrics.mse,
        'initial_ssim': result.initial_metrics.ssim,
        'final_ssim': result.final_metrics.ssim,
        'total_iterations': result.total_iterations,
        'total_time': result.total_time,
        'converged': result.converged,
        'loss_history': result.loss_history,
        'optimized_mask': result.optimized_mask,
        'final_wafer_image': result.final_wafer_image
    }
    
    saved_files = save_results(
        results_dict,
        output_dir,
        prefix='optimization'
    )
    
    logger.info("保存的文件:")
    for key, path in saved_files.items():
        logger.info(f"  {key}: {path}")
    
    # ========== 完成 ==========
    logger.info("=" * 60)
    logger.info("示例运行完成!")
    logger.info(f"结果保存在: {output_dir.absolute()}")
    logger.info("=" * 60)
    
    return result


if __name__ == '__main__':
    main()
