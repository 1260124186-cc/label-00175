# -*- coding: utf-8 -*-
"""
子命令: optimize
通用掩模优化（MaskOptimizer）

对应原 examples/run_optimization.py
"""

import json
from pathlib import Path

import click
import numpy as np

from ..common import (
    global_options, output_options, optical_system_options,
    test_pattern_options, optimizer_options,
    parse_grid_size, setup_cli_logger, build_optical_system,
    create_pattern, ensure_output_dir, merge_cli_with_yaml,
    print_banner, print_summary_block,
)

from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig
from core.metrics import evaluate_all
from utils.visualization import (
    plot_optimization_summary,
    plot_error_curve,
    plot_frequency_domain,
    plot_comparison,
)
from core.fft import fft2d


@click.command(
    name="optimize",
    help="通用掩模优化（MaskOptimizer），支持多种优化器对掩模做梯度/启发式优化"
)
@global_options
@output_options
@optical_system_options
@test_pattern_options
@optimizer_options
@click.option(
    "--tol",
    type=float,
    default=1e-7,
    show_default=True,
    help="收敛容差（损失改善低于此值视为收敛）"
)
@click.option(
    "--early-stop-patience",
    type=int,
    default=10,
    show_default=True,
    help="早停耐心值：连续 N 次未改善则停止"
)
@click.option(
    "--bounds",
    type=str,
    default="0.0,1.0",
    show_default=True,
    help="掩模值上下界 'min,max'"
)
@click.option(
    "--add-noise",
    is_flag=True,
    default=False,
    help="在初始掩模上添加噪声（模拟不理想的起点）"
)
@click.pass_context
def optimize_cmd(
    ctx,
    verbose, log_file, config_path,
    output_dir, save_masks, no_plot,
    wavelength, na, sigma, defocus, pixel_size,
    pattern, grid_size, cd, pitch,
    optimizer, max_iter, learning_rate, metric,
    tol, early_stop_patience, bounds, add_noise,
):
    """通用掩模优化流程"""

    logger = setup_cli_logger("litho_optimize", verbose, log_file)
    print_banner(logger, "通用掩模优化 (MaskOptimizer)")

    # --- 参数合并 ---
    cli_params = {
        "optimizer_type": optimizer,
        "max_iter": max_iter if max_iter is not None else 50,
        "learning_rate": learning_rate if learning_rate is not None else 0.1,
        "metric": metric or "mse",
        "tol": tol,
        "early_stop_patience": early_stop_patience,
    }
    merged = merge_cli_with_yaml(cli_params, config_path, section_key="optimizer")

    # --- 解析 ---
    gs = parse_grid_size(grid_size)
    bounds_vals = tuple(float(x) for x in bounds.split(","))
    if len(bounds_vals) != 2:
        raise click.BadParameter("--bounds 格式应为 'min,max'")

    # --- 准备输出 ---
    out = ensure_output_dir(output_dir, "optimize")
    logger.info(f"输出目录: {out.resolve()}")

    # --- 光学系统 ---
    optical_sys = build_optical_system(wavelength, na, sigma, defocus, pixel_size)
    logger.info(
        f"光学系统: λ={optical_sys.wavelength}nm, NA={optical_sys.na}, "
        f"σ={optical_sys.sigma}, defocus={optical_sys.defocus}nm"
    )

    # --- 测试图案 ---
    target, desc = create_pattern(pattern, gs, cd, pitch, pixel_size)
    initial_mask = target.copy()
    if add_noise:
        initial_mask = initial_mask + 0.1 * np.random.randn(*gs)
        initial_mask = np.clip(initial_mask, 0, 1)
        logger.info("初始掩模: 已添加高斯噪声")
    logger.info(f"测试图案: {desc}, 尺寸={target.shape}")

    # --- 配置优化器 ---
    optimizer_type_val = merged.get("optimizer_type") or "gradient_descent"
    metric_val = merged.get("metric") or "mse"
    opt_cfg = OptimizationConfig(
        optimizer_type=optimizer_type_val,
        max_iter=int(merged["max_iter"]),
        learning_rate=float(merged["learning_rate"]),
        tol=float(merged["tol"]),
        early_stop_patience=int(merged["early_stop_patience"]),
        metric=metric_val,
        bounds=bounds_vals,
        verbose=True,
    )
    mask_optimizer = MaskOptimizer(
        optical_system=optical_sys,
        config=opt_cfg,
    )
    logger.info(
        f"优化器: {optimizer_type_val}, max_iter={opt_cfg.max_iter}, "
        f"lr={opt_cfg.learning_rate}, metric={metric_val}"
    )

    # --- 初始评估 ---
    logger.info("评估初始状态...")
    from core.imaging import PartialCoherentImaging
    imaging = PartialCoherentImaging(optical_sys, gs)
    initial_wafer = imaging.compute_aerial_image(initial_mask)
    initial_metrics = evaluate_all(initial_wafer, target)
    logger.info(
        f"初始 MSE={initial_metrics.mse:.6e}, SSIM={initial_metrics.ssim:.4f}"
    )

    # --- 执行优化 ---
    logger.info("开始优化迭代...")
    result = mask_optimizer.optimize(
        initial_mask=initial_mask,
        target_image=target,
    )
    logger.info("优化完成!")

    # --- 汇总 ---
    lines = [
        "优化汇总:",
        f"  最终 MSE   : {result.final_metrics.mse:.6e}",
        f"  最终 SSIM  : {result.final_metrics.ssim:.4f}",
        f"  总迭代次数 : {result.total_iterations}",
        f"  总耗时     : {result.total_time:.2f} 秒",
        f"  收敛状态   : {'是' if result.converged else '否'}",
    ]
    print_summary_block(logger, lines)

    # --- 可视化 ---
    if not no_plot:
        logger.info("生成可视化图表...")
        plot_optimization_summary(
            result,
            save_path=str(out / "optimization_summary.png"),
            show=False,
        )
        plot_error_curve(
            result.loss_history,
            title="优化收敛曲线",
            log_scale=True,
            save_path=str(out / "convergence_curve.png"),
            show=False,
        )
        mask_spectrum = fft2d(initial_mask, shift=True)
        plot_frequency_domain(
            mask_spectrum,
            title="掩模频谱",
            save_path=str(out / "frequency_domain.png"),
            show=False,
        )
        plot_comparison(
            images=[
                result.initial_mask,
                result.optimized_mask,
                result.target_image,
                result.initial_wafer_image,
                result.final_wafer_image,
            ],
            titles=[
                "初始掩模", "优化后掩模", "目标图像",
                "初始成像", "最终成像",
            ],
            main_title="掩模优化前后对比",
            save_path=str(out / "comparison.png"),
            show=False,
        )
        logger.info(f"图表已保存到 {out}/")

    # --- 保存掩模 ---
    if save_masks:
        np.save(out / "initial_mask.npy", result.initial_mask)
        np.save(out / "optimized_mask.npy", result.optimized_mask)
        np.save(out / "target_image.npy", result.target_image)
        logger.info("掩模 npy 文件已保存")

    # --- 保存 JSON 汇总 ---
    summary_json = {
        "initial_mse": float(result.initial_metrics.mse),
        "final_mse": float(result.final_metrics.mse),
        "initial_ssim": float(result.initial_metrics.ssim),
        "final_ssim": float(result.final_metrics.ssim),
        "total_iterations": int(result.total_iterations),
        "total_time": float(result.total_time),
        "converged": bool(result.converged),
        "loss_history": [float(x) for x in result.loss_history],
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    logger.info(f"所有结果已保存到: {out.resolve()}")
    return 0 if result.converged else 1
