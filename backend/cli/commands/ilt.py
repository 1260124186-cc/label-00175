# -*- coding: utf-8 -*-
"""
子命令: ilt
ILT (Inverse Lithography Technology) 反演光刻技术

对应原 workflows/ilt.py 中的 run_ilt_workflow
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

from workflows.ilt import (
    ILTConfig, run_ilt_workflow,
    TransmissionLevel, ILTOptimizerType, ILTComplexityConfig,
)
from utils.visualization import plot_comparison, plot_error_curve


@click.command(
    name="ilt",
    help="ILT 反演光刻：可微成像链 + 梯度投影 + 掩模量化 + 复杂度惩罚"
)
@global_options
@output_options
@optical_system_options
@test_pattern_options
@optimizer_options
@click.option(
    "--transmission",
    type=click.Choice(["binary", "ternary", "continuous"]),
    default=None,
    help="离散透射率等级（二值/三值/连续）"
)
@click.option(
    "--quantization-start",
    type=int,
    default=None,
    help="开始量化的迭代数（延迟量化）"
)
@click.option(
    "--quantization-schedule",
    type=click.Choice(["step", "linear", "cosine"]),
    default=None,
    help="量化调度策略"
)
@click.option(
    "--resist-steepness",
    type=float,
    default=None,
    help="Soft resist sigmoid 陡度参数 k"
)
@click.option(
    "--wafer-threshold",
    type=float,
    default=None,
    help="光刻胶阈值"
)
@click.option(
    "--binary-penalty-weight",
    type=float,
    default=None,
    help="二值化惩罚权重"
)
@click.option(
    "--tv-smooth-weight",
    type=float,
    default=None,
    help="TV 平滑权重"
)
@click.option(
    "--perimeter-weight",
    type=float,
    default=None,
    help="掩模周长惩罚权重"
)
@click.option(
    "--vertex-weight",
    type=float,
    default=None,
    help="掩模顶点数惩罚权重"
)
@click.option(
    "--default-config/--no-default-config",
    is_flag=True,
    default=True,
    help="尝试从 config/ 目录加载默认 ILT 配置"
)
@click.pass_context
def ilt_cmd(
    ctx,
    verbose, log_file, config_path,
    output_dir, save_masks, no_plot,
    wavelength, na, sigma, defocus, pixel_size,
    pattern, grid_size, cd, pitch,
    optimizer, max_iter, learning_rate, metric,
    transmission, quantization_start, quantization_schedule,
    resist_steepness, wafer_threshold,
    binary_penalty_weight, tv_smooth_weight,
    perimeter_weight, vertex_weight,
    default_config,
):
    """ILT 工作流"""

    logger = setup_cli_logger("litho_ilt", verbose, log_file)
    print_banner(logger, "ILT 反演光刻技术工作流")

    # --- 默认配置 ---
    default_yaml = None
    if default_config:
        default_yaml_candidate = str(
            Path(__file__).resolve().parents[2] / "config" / "ilt_default.yaml"
        )
        if Path(default_yaml_candidate).exists():
            default_yaml = default_yaml_candidate

    # --- CLI 参数字典 ---
    optimizer_map = {
        "gradient_descent": ILTOptimizerType.GRADIENT_PROJECTION,
        "adam": ILTOptimizerType.ADAM_PROJECTION,
        "sgd": ILTOptimizerType.SGD_PROJECTION,
    }
    transmission_map = {
        "binary": TransmissionLevel.BINARY,
        "ternary": TransmissionLevel.TERNARY,
        "continuous": TransmissionLevel.CONTINUOUS,
    }

    cli_params: dict = {"pixel_size": pixel_size}
    if optimizer is not None and optimizer in optimizer_map:
        cli_params["optimizer_type"] = optimizer_map[optimizer]
    if max_iter is not None:
        cli_params["max_iter"] = max_iter
    if learning_rate is not None:
        cli_params["learning_rate"] = learning_rate
    if transmission is not None:
        cli_params["transmission_level"] = transmission_map[transmission]
    if quantization_start is not None:
        cli_params["quantization_start_iter"] = quantization_start
    if quantization_schedule is not None:
        cli_params["quantization_schedule"] = quantization_schedule
    if resist_steepness is not None:
        cli_params["resist_steepness"] = resist_steepness
    if wafer_threshold is not None:
        cli_params["wafer_threshold"] = wafer_threshold
    if binary_penalty_weight is not None:
        cli_params["binary_penalty_weight"] = binary_penalty_weight
    if tv_smooth_weight is not None:
        cli_params["tv_smooth_weight"] = tv_smooth_weight

    # 复杂度惩罚子配置
    complexity_dict: dict = {}
    if perimeter_weight is not None:
        complexity_dict["perimeter_weight"] = perimeter_weight
    if vertex_weight is not None:
        complexity_dict["vertex_weight"] = vertex_weight
    if complexity_dict:
        cli_params["complexity"] = complexity_dict

    # --- 合并 ---
    merged = merge_cli_with_yaml(cli_params, config_path, section_key="ilt")
    if default_yaml and not config_path:
        merged = merge_cli_with_yaml(merged, default_yaml, section_key="ilt")

    # --- 解析 ---
    gs = parse_grid_size(grid_size)

    # --- 输出 ---
    out = ensure_output_dir(output_dir, "ilt")
    logger.info(f"输出目录: {out.resolve()}")

    # --- 光学系统 ---
    optical_sys = build_optical_system(
        wavelength, na, sigma, defocus, pixel_size,
    )
    logger.info(
        f"光学系统: λ={optical_sys.wavelength}nm, NA={optical_sys.na}, "
        f"σ={optical_sys.sigma}, defocus={optical_sys.defocus}nm"
    )

    # --- ILTConfig ---
    ilt_config = ILTConfig.from_dict(merged)
    logger.info(
        f"ILT 配置: 优化器={ilt_config.optimizer_type.value}, "
        f"max_iter={ilt_config.max_iter}, lr={ilt_config.learning_rate}, "
        f"透射率={ilt_config.transmission_level.value}"
    )
    if ilt_config.complexity:
        c = ilt_config.complexity
        logger.info(
            f"  复杂度惩罚: 周长权重={c.perimeter_weight}, "
            f"顶点权重={c.vertex_weight}, 辅助特征权重={c.sub_feature_weight}"
        )

    # --- 图案 ---
    target, desc = create_pattern(pattern, gs, cd, pitch, pixel_size)
    initial_mask = target.copy()
    logger.info(f"测试图案: {desc}, 尺寸={target.shape}")

    # --- 执行 ---
    logger.info("开始 ILT 优化...")
    result = run_ilt_workflow(
        initial_mask, target,
        optical_system=optical_sys,
        config=ilt_config,
    )
    logger.info("ILT 完成!")

    # --- 汇总 ---
    lines = [
        "ILT 汇总:",
        f"  优化器           : {ilt_config.optimizer_type.value}",
        f"  透射率等级       : {ilt_config.transmission_level.value}",
        f"  初始 EPE(均值)   : {result.initial_epe['epe_mean']:.3f} nm",
        f"  最终 EPE(均值)   : {result.final_epe['epe_mean']:.3f} nm",
        f"  EPE 改善量       : {result.total_epe_improvement:.3f} nm",
        f"  初始损失         : {result.initial_loss:.6f}",
        f"  最终损失         : {result.final_loss:.6f}",
        f"  损失改善比例     : {result.total_loss_improvement_ratio * 100:.1f}%",
        f"  总迭代次数       : {result.num_iterations}",
        f"  收敛状态         : {'是' if result.converged else '否'} — {result.reason}",
        f"  最终量化强度     : {result.final_quantization_strength:.3f}",
        f"  总耗时           : {result.total_time:.2f}s",
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
            result.initial_wafer, result.optimal_wafer,
            titles=["初始晶圆成像", "优化后晶圆成像"],
        )
        fig2.savefig(out / "wafer_comparison.png", dpi=150)

        if result.loss_history:
            plot_error_curve(
                result.loss_history,
                title="ILT 损失收敛曲线",
                log_scale=True,
                save_path=str(out / "loss_curve.png"),
                show=False,
            )

    # --- 保存 ---
    if save_masks:
        np.save(out / "initial_mask.npy", result.initial_mask)
        np.save(out / "optimal_mask.npy", result.optimal_mask)

    summary = {
        "pattern": pattern,
        "description": desc,
        "optimizer": ilt_config.optimizer_type.value,
        "transmission_level": ilt_config.transmission_level.value,
        "initial_epe": {k: float(v) for k, v in result.initial_epe.items()},
        "final_epe": {k: float(v) for k, v in result.final_epe.items()},
        "total_epe_improvement": float(result.total_epe_improvement),
        "initial_loss": float(result.initial_loss),
        "final_loss": float(result.final_loss),
        "total_loss_improvement": float(result.total_loss_improvement),
        "total_loss_improvement_ratio": float(result.total_loss_improvement_ratio),
        "num_iterations": int(result.num_iterations),
        "converged": bool(result.converged),
        "reason": result.reason,
        "total_time": float(result.total_time),
        "final_quantization_strength": float(result.final_quantization_strength),
        "summary": result.summary(),
    }
    with open(out / "ilt_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"所有结果已保存到: {out.resolve()}")
    return 0 if result.converged else 1
