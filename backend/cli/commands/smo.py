# -*- coding: utf-8 -*-
"""
子命令: smo
SMO (Source-Mask Optimization) 光源-掩模协同优化

对应原 examples/run_smo.py
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

from workflows.smo import (
    SMOConfig, SMOWorkflow, run_smo_workflow,
    SMOptimizationStrategy, SourceInitializationType,
    SourceConstraintsConfig,
)
from core.litho_metrics import compute_epe
from utils.visualization import plot_comparison


@click.command(
    name="smo",
    help="SMO 光源-掩模协同优化：交替/联合梯度/先光源后掩模 三种策略"
)
@global_options
@output_options
@optical_system_options
@test_pattern_options
@optimizer_options
@click.option(
    "--strategy",
    type=click.Choice(["alternating", "joint_gradient", "source_first"]),
    default="alternating",
    show_default=True,
    help="SMO 优化策略"
)
@click.option(
    "--max-outer-iter",
    type=int,
    default=None,
    help="外层最大迭代次数（交替/先光源 策略）"
)
@click.option(
    "--source-max-iter",
    type=int,
    default=None,
    help="光源优化最大迭代数"
)
@click.option(
    "--mask-max-iter",
    type=int,
    default=None,
    help="掩模优化最大迭代数"
)
@click.option(
    "--joint-max-iter",
    type=int,
    default=None,
    help="联合梯度下降最大迭代数"
)
@click.option(
    "--source-init",
    type=click.Choice(["conventional", "annular", "quasar", "dipole"]),
    default=None,
    help="像素化光源初始化方式"
)
@click.option(
    "--wafer-threshold",
    type=float,
    default=None,
    help="晶圆阈值（soft resist 二值化阈值）"
)
@click.option(
    "--default-config/--no-default-config",
    is_flag=True,
    default=True,
    help="使用 config/smo_default.yaml 作为默认配置"
)
@click.pass_context
def smo_cmd(
    ctx,
    verbose, log_file, config_path,
    output_dir, save_masks, no_plot,
    wavelength, na, sigma, defocus, pixel_size,
    pattern, grid_size, cd, pitch,
    optimizer, max_iter, learning_rate, metric,
    strategy, max_outer_iter, source_max_iter, mask_max_iter,
    joint_max_iter, source_init, wafer_threshold,
    default_config,
):
    """SMO 工作流"""

    logger = setup_cli_logger("litho_smo", verbose, log_file)
    print_banner(logger, "SMO 光源-掩模协同优化工作流")

    # --- 默认配置 ---
    default_yaml = None
    if default_config:
        default_yaml = str(
            Path(__file__).resolve().parents[2] / "config" / "smo_default.yaml"
        )
        if not Path(default_yaml).exists():
            default_yaml = None

    # --- CLI 参数字典 ---
    strategy_map = {
        "alternating": SMOptimizationStrategy.ALTERNATING,
        "joint_gradient": SMOptimizationStrategy.JOINT_GRADIENT,
        "source_first": SMOptimizationStrategy.SOURCE_FIRST,
    }
    source_init_map = {
        "conventional": SourceInitializationType.CONVENTIONAL,
        "annular": SourceInitializationType.ANNULAR,
        "quasar": SourceInitializationType.QUASAR,
        "dipole": SourceInitializationType.DIPOLE,
    }

    cli_params: dict = {"pixel_size": pixel_size}
    cli_params["strategy"] = strategy_map[strategy]
    if max_outer_iter is not None:
        cli_params["max_outer_iterations"] = max_outer_iter
    if source_max_iter is not None:
        cli_params["source_max_iter"] = source_max_iter
    if mask_max_iter is not None:
        cli_params["mask_max_iter"] = mask_max_iter
    if joint_max_iter is not None:
        cli_params["joint_max_iter"] = joint_max_iter
    if source_init is not None:
        cli_params["source_init_type"] = source_init_map[source_init]
    if wafer_threshold is not None:
        cli_params["wafer_threshold"] = wafer_threshold
    if learning_rate is not None:
        if strategy == "joint_gradient":
            cli_params.setdefault("joint_learning_rate_source", learning_rate * 0.3)
            cli_params.setdefault("joint_learning_rate_mask", learning_rate)
        else:
            cli_params.setdefault("source_learning_rate", learning_rate * 0.3)
            cli_params.setdefault("mask_learning_rate", learning_rate)

    # --- 合并 ---
    merged = merge_cli_with_yaml(cli_params, config_path, section_key="smo")
    if default_yaml and not config_path:
        merged = merge_cli_with_yaml(merged, default_yaml, section_key="smo")

    # --- 解析 ---
    gs = parse_grid_size(grid_size)

    # --- 输出 ---
    out = ensure_output_dir(output_dir, f"smo_{strategy}")
    logger.info(f"输出目录: {out.resolve()}")

    # --- 光学系统 ---
    optical_sys = build_optical_system(
        wavelength, na, sigma, defocus, pixel_size, socs_num_terms=8,
    )
    logger.info(
        f"光学系统: λ={optical_sys.wavelength}nm, NA={optical_sys.na}, "
        f"σ={optical_sys.sigma}, SOCS terms=8"
    )

    # --- SMOConfig ---
    smo_config = SMOConfig.from_dict(merged)
    logger.info(
        f"SMO 配置: 策略={smo_config.strategy.value}, "
        f"外层迭代={smo_config.max_outer_iterations}, "
        f"光源初始化={smo_config.source_init_type.value}"
    )

    # --- 图案 ---
    target, desc = create_pattern(pattern, gs, cd, pitch, pixel_size)
    initial_mask = target.copy()
    logger.info(f"测试图案: {desc}, 尺寸={target.shape}")

    # --- 执行 ---
    logger.info("开始 SMO 优化...")
    result = run_smo_workflow(
        initial_mask, target,
        config=smo_config,
        optical_system=optical_sys,
    )
    logger.info("SMO 完成!")

    # --- EPE ---
    epe_initial = compute_epe(
        result.initial_wafer, target, pixel_size=smo_config.pixel_size
    )
    epe_optimal = compute_epe(
        result.optimal_wafer, target, pixel_size=smo_config.pixel_size
    )

    lines = [
        "SMO 汇总:",
        f"  策略          : {smo_config.strategy.value}",
        f"  初始 EPE(均值): {epe_initial['epe_mean']:.3f} nm",
        f"  最优 EPE(均值): {epe_optimal['epe_mean']:.3f} nm",
        f"  EPE 改善量    : {result.total_epe_improvement:.3f} nm "
        f"({result.total_epe_improvement_ratio * 100:.1f}%)",
        f"  外层迭代      : {result.num_iterations}",
        f"  收敛状态      : {'是' if result.converged else '否'} — {result.reason}",
    ]
    print_summary_block(logger, lines)

    # --- 可视化 ---
    if not no_plot:
        logger.info("生成图表...")
        fig1 = plot_comparison(
            result.initial_mask, result.optimal_mask,
            titles=["初始掩模", "优化后掩模"],
        )
        fig1.savefig(out / "mask_comparison.png", dpi=150)

        fig2 = plot_comparison(
            np.fft.fftshift(result.initial_source),
            np.fft.fftshift(result.optimal_source),
            titles=["初始光源分布", "优化后光源分布"],
        )
        fig2.savefig(out / "source_comparison.png", dpi=150)

        fig3 = plot_comparison(
            result.initial_wafer, result.optimal_wafer,
            titles=["初始晶圆成像", "优化后晶圆成像"],
        )
        fig3.savefig(out / "wafer_comparison.png", dpi=150)

    # --- 保存 ---
    if save_masks:
        np.save(out / "initial_mask.npy", result.initial_mask)
        np.save(out / "optimal_mask.npy", result.optimal_mask)
        np.save(out / "optimal_source.npy", result.optimal_source)

    summary = {
        "strategy": smo_config.strategy.value,
        "pattern": pattern,
        "description": desc,
        "initial_epe": {k: float(v) for k, v in epe_initial.items()},
        "final_epe": {k: float(v) for k, v in epe_optimal.items()},
        "total_epe_improvement": float(result.total_epe_improvement),
        "total_epe_improvement_ratio": float(result.total_epe_improvement_ratio),
        "num_iterations": int(result.num_iterations),
        "converged": bool(result.converged),
        "reason": result.reason,
        "total_time": float(result.total_time),
        "summary": result.summary(),
    }
    with open(out / "smo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"所有结果已保存到: {out.resolve()}")
    return 0 if result.converged else 1
