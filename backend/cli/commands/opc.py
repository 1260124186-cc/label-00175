# -*- coding: utf-8 -*-
"""
子命令: opc
OPC (Optical Proximity Correction) 光学邻近校正工作流

对应原 examples/run_opc.py
"""

import json
from pathlib import Path

import click
import numpy as np

from ..common import (
    global_options, output_options, optical_system_options,
    test_pattern_options,
    parse_grid_size, setup_cli_logger, build_optical_system,
    create_pattern, ensure_output_dir, merge_cli_with_yaml,
    print_banner, print_summary_block,
)

from workflows.opc import (
    OPCConfig, run_opc_workflow,
)
from core.imaging import simulate_wafer_image
from core.litho_metrics import compute_epe
from utils.visualization import plot_comparison


@click.command(
    name="opc",
    help="OPC 光学邻近校正：热点检测 + SRAF 插入 + 迭代掩模校正"
)
@global_options
@output_options
@optical_system_options
@test_pattern_options
@click.option(
    "--epe-threshold",
    type=float,
    default=None,
    help="EPE 阈值 (nm)，超过该值视为热点"
)
@click.option(
    "--max-iterations",
    type=int,
    default=None,
    help="OPC 最大迭代轮次"
)
@click.option(
    "--sraf/--no-sraf",
    default=None,
    help="是否启用 SRAF（Sub-Resolution Assist Feature）插入"
)
@click.option(
    "--optimizer/--no-optimizer",
    default=None,
    help="是否启用 MaskOptimizer 梯度细化"
)
@click.option(
    "--default-config/--no-default-config",
    is_flag=True,
    default=True,
    help="使用 config/opc_default.yaml 作为默认配置（CLI 参数会覆盖之）"
)
@click.pass_context
def opc_cmd(
    ctx,
    verbose, log_file, config_path,
    output_dir, save_masks, no_plot,
    wavelength, na, sigma, defocus, pixel_size,
    pattern, grid_size, cd, pitch,
    epe_threshold, max_iterations, sraf, optimizer,
    default_config,
):
    """OPC 工作流"""

    logger = setup_cli_logger("litho_opc", verbose, log_file)
    print_banner(logger, "OPC 光学邻近校正工作流")

    # --- 默认配置 ---
    default_yaml = None
    if default_config:
        default_yaml = str(
            Path(__file__).resolve().parents[2] / "config" / "opc_default.yaml"
        )
        if not Path(default_yaml).exists():
            default_yaml = None

    # --- 合并配置 ---
    cli_params = {}
    if epe_threshold is not None:
        cli_params["epe_threshold"] = epe_threshold
    if max_iterations is not None:
        cli_params["max_iterations"] = max_iterations
    if sraf is not None:
        cli_params["sraf_enable"] = sraf
    if optimizer is not None:
        cli_params["optimizer_enable"] = optimizer
    cli_params["pixel_size"] = pixel_size

    merged = merge_cli_with_yaml(cli_params, config_path, section_key="opc")
    if default_yaml and not config_path:
        merged = merge_cli_with_yaml(merged, default_yaml, section_key="opc")

    # --- 解析 ---
    gs = parse_grid_size(grid_size)

    # --- 输出 ---
    out = ensure_output_dir(output_dir, "opc")
    logger.info(f"输出目录: {out.resolve()}")

    # --- 光学系统 ---
    optical_sys = build_optical_system(wavelength, na, sigma, defocus, pixel_size)
    logger.info(
        f"光学系统: λ={optical_sys.wavelength}nm, NA={optical_sys.na}, "
        f"σ={optical_sys.sigma}, defocus={optical_sys.defocus}nm"
    )

    # --- 构建 OPCConfig ---
    opc_config = OPCConfig.from_dict(merged)
    logger.info(
        f"OPC 配置: EPE阈值={opc_config.epe_threshold}nm, "
        f"最大迭代={opc_config.max_iterations}, "
        f"SRAF={'启用' if opc_config.sraf_enable else '禁用'}, "
        f"Optimizer={'启用' if opc_config.optimizer_enable else '禁用'}"
    )

    # --- 处理图案 ---
    if pattern == "all":
        pattern_list = ["line_space", "l_shaped", "contact_hole"]
    else:
        pattern_list = [pattern]

    all_ok = True

    for pat in pattern_list:
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"处理图案: {pat}")
        logger.info("-" * 60)

        target, desc = create_pattern(pat, gs, cd, pitch, pixel_size)
        initial_mask = target.copy()
        logger.info(f"图案: {desc}, 尺寸={target.shape}")

        pat_out = ensure_output_dir(str(out), pat)

        # --- 执行 OPC ---
        logger.info("开始 OPC 工作流...")
        result = run_opc_workflow(
            initial_mask, target,
            config=opc_config,
            optical_system=optical_sys,
        )
        logger.info("OPC 完成!")

        # --- EPE 改善 ---
        epe_initial = compute_epe(
            result.initial_wafer, target, pixel_size=opc_config.pixel_size
        )
        epe_corrected = compute_epe(
            result.corrected_wafer, target, pixel_size=opc_config.pixel_size
        )

        lines = [
            f"图案: {pat}",
            f"  初始 EPE(均值): {epe_initial['epe_mean']:.3f} nm",
            f"  校正 EPE(均值): {epe_corrected['epe_mean']:.3f} nm",
            f"  EPE 改善量   : {result.total_epe_improvement:.3f} nm "
            f"({result.total_epe_improvement_ratio * 100:.1f}%)",
            f"  总迭代次数   : {result.num_iterations}",
            f"  SRAF 数量    : {len(result.all_srafs)}",
        ]
        print_summary_block(logger, lines)

        if epe_corrected["epe_mean"] >= epe_initial["epe_mean"]:
            all_ok = False

        # --- 可视化 ---
        if not no_plot:
            logger.info("生成图表...")
            fig1 = plot_comparison(
                result.initial_mask, target,
                titles=["初始掩模", "目标图案"],
            )
            fig1.savefig(pat_out / "mask_comparison.png", dpi=150)

            fig2 = plot_comparison(
                result.initial_wafer, result.corrected_wafer,
                titles=["初始晶圆成像", "校正后晶圆成像"],
            )
            fig2.savefig(pat_out / "wafer_comparison.png", dpi=150)

        # --- 保存 ---
        if save_masks:
            np.save(pat_out / "initial_mask.npy", result.initial_mask)
            np.save(pat_out / "corrected_mask.npy", result.corrected_mask)

        # --- JSON ---
        summary = {
            "pattern": pat,
            "description": desc,
            "initial_epe": {k: float(v) for k, v in epe_initial.items()},
            "final_epe": {k: float(v) for k, v in epe_corrected.items()},
            "total_epe_improvement": float(result.total_epe_improvement),
            "total_epe_improvement_ratio": float(result.total_epe_improvement_ratio),
            "num_iterations": int(result.num_iterations),
            "sraf_count": len(result.all_srafs),
            "summary": result.summary(),
        }
        with open(pat_out / "opc_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"子结果已保存到 {pat_out.resolve()}")

    logger.info("")
    print_banner(logger, f"OPC 全部完成，结果总目录: {out.resolve()}")
    return 0 if all_ok else 1
