#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OPC (Optical Proximity Correction) 工作流示例：完整的 OPC 流程演示

该示例展示如何使用 OPC 工作流模块完成：
1. 创建测试掩模图案（线/空间、L形拐角、接触孔等）
2. 配置 OPC 工作流参数
3. 热点检测（高 EPE 区域识别）
4. SRAF 自动插入与放置
5. 迭代优化主特征与辅助特征
6. 结果可视化与分析

运行方式：
    # 从 backend 目录下运行：
    python -m examples.run_opc
    # 或直接运行：
    python examples/run_opc.py
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import json

from core.imaging import OpticalSystem, simulate_wafer_image
from core.litho_metrics import compute_epe
from core.test_structures import (
    TestStructureParams, TestStructureType, LineOrientation,
    create_test_structure, create_line_space, create_l_shaped_corner,
    create_contact_hole
)
from workflows.opc import (
    OPCConfig, OPCWorkflow, run_opc_workflow,
    HotspotDetector, SRAFRuleEngine,
    OPCWorkflowResult
)
from utils.config import load_config, save_config
from utils.logger import setup_logger
from utils.visualization import (
    plot_mask, plot_wafer_image, plot_comparison
)


def create_demo_pattern(pattern_type: str = 'line_space',
                      grid_size: Tuple[int, int] = (64, 64),
                      cd: float = 45.0,
                      pixel_size: float = 1.0) -> Tuple[np.ndarray, str]:
    """
    创建演示用测试图案

    Args:
        pattern_type: 图案类型
        grid_size: 网格尺寸
        cd: 关键尺寸 (nm)
        pixel_size: 像素尺寸 (nm)

    Returns:
        (掩模图案, 图案描述)
    """
    pitch = cd * 2

    if pattern_type == 'line_space':
        params = TestStructureParams(
            grid_size=grid_size,
            pixel_size=pixel_size,
            cd=cd,
            pitch=pitch,
            structure_type=TestStructureType.LINE_SPACE,
        )
        target = create_line_space(params)
        desc = f"线/空间结构 CD={cd}nm Pitch={pitch}nm"

    elif pattern_type == 'l_shaped':
        params = TestStructureParams(
            grid_size=grid_size,
            pixel_size=pixel_size,
            cd=cd,
            pitch=pitch,
            structure_type=TestStructureType.L_SHAPED_CORNER,
        )
        target = create_l_shaped_corner(params)
        desc = f"L形拐角结构 CD={cd}nm"

    elif pattern_type == 'contact_hole':
        params = TestStructureParams(
            grid_size=grid_size,
            pixel_size=pixel_size,
            cd=cd,
            pitch=pitch,
            structure_type=TestStructureType.CONTACT_HOLE,
        )
        target = create_contact_hole(params)
        desc = f"接触孔阵列 CD={cd}nm Pitch={pitch}nm"

    else:
        target = np.zeros(grid_size, dtype=np.float64)
        ny, nx = grid_size
        cy, cx = ny // 2, nx // 2
        target[cy-10:cy+10, cx-20:cx+20] = 1.0
        desc = f"简单矩形结构"

    return target, desc


def setup_opc_config_from_yaml(config_path: Optional[str] = None) -> Tuple[OPCConfig, OpticalSystem]:
    """
    从 YAML 配置文件设置 OPC 配置

    Args:
        config_path: 配置文件路径

    Returns:
        (OPC配置, 光学系统配置)
    """
    if config_path is None:
        config_path = Path(parent_dir) / 'config' / 'opc_default.yaml'

    config_dict = load_config(config_path)

    opc_config = OPCConfig.from_dict(config_dict.get('opc', {}))

    opt_sys_params = config_dict.get('optical_system', {})
    optical_system = OpticalSystem(
        wavelength=opt_sys_params.get('wavelength', 193.0),
        na=opt_sys_params.get('na', 1.35),
        sigma=opt_sys_params.get('sigma', 0.75),
        pixel_size=opt_sys_params.get('pixel_size', 1.0),
        defocus=opt_sys_params.get('defocus', 0.0),
        magnification=opt_sys_params.get('magnification', 4.0),
    )

    return opc_config, optical_system


def analyze_results(result: OPCWorkflowResult,
                    output_dir: Path) -> Dict[str, Any]:
    """
    分析 OPC 结果并生成报告

    Args:
        result: OPC 工作流结果
        output_dir: 输出目录

    Returns:
        分析结果字典
    """
    analysis = {
        'summary': result.summary(),
        'iterations': [],
    }

    for i, iter_result in enumerate(result.iterations):
        iter_info = {
            'iteration': i + 1,
            'epe_before': iter_result.epe_before['epe_mean'],
            'epe_after': iter_result.epe_after['epe_mean'],
            'epe_improvement': iter_result.epe_improvement,
            'epe_improvement_ratio': iter_result.epe_improvement_ratio,
            'hotspots_before': len(iter_result.hotspots_before),
            'hotspots_after': len(iter_result.hotspots_after),
            'transforms_count': len(iter_result.transforms_applied),
            'srafs_inserted': iter_result.srafs_inserted,
        }
        analysis['iterations'].append(iter_info)

    if result.all_srafs:
        analysis['srafs'] = [s.to_dict() for s in result.all_srafs]

    if result.all_hotspots:
        analysis['hotspots_evolution'] = [
            [h.to_dict() for h in round_hotspots]
            for round_hotspots in result.all_hotspots
        ]

    report_path = output_dir / 'opc_analysis.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"分析报告已保存: {report_path}")

    return analysis


def visualize_results(result: OPCWorkflowResult,
                      target: np.ndarray,
                      output_dir: Path,
                      optical_system: OpticalSystem,
                      pixel_size: float = 1.0):
    """
    可视化 OPC 结果

    Args:
        result: OPC 结果
        target: 目标图案
        output_dir: 输出目录
        optical_system: 光学系统
        pixel_size: 像素尺寸
    """
    initial_wafer_cont = simulate_wafer_image(
        result.initial_mask, optical_system=optical_system)
    corrected_wafer_cont = simulate_wafer_image(
        result.corrected_mask, optical_system=optical_system)

    fig_initial = plot_comparison(
        result.initial_mask, target,
        titles=['初始掩模', '目标图案'])
    fig_initial.savefig(output_dir / 'mask_comparison.png', dpi=150)

    fig_wafer = plot_comparison(
        result.initial_wafer, result.corrected_wafer,
        titles=['初始晶圆成像', '校正后晶圆成像'])
    fig_wafer.savefig(output_dir / 'wafer_comparison.png', dpi=150)

    epe_initial = compute_epe(result.initial_wafer, target, pixel_size=pixel_size)
    epe_corrected = compute_epe(result.corrected_wafer, target, pixel_size=pixel_size)

    logger.info("\n" + "="*60)
    logger.info("EPE 改善详情")
    logger.info("="*60)
    logger.info(f"初始 EPE:")
    logger.info(f"  平均值: {epe_initial['epe_mean']:.3f} nm")
    logger.info(f"  最大值: {epe_initial['epe_max']:.3f} nm")
    logger.info(f"校正后 EPE:")
    logger.info(f"  平均值: {epe_corrected['epe_mean']:.3f} nm")
    logger.info(f"  最大值: {epe_corrected['epe_max']:.3f} nm")
    logger.info(f"改善量: {result.total_epe_improvement:.3f} nm ({result.total_epe_improvement_ratio*100:.1f}%)")
    logger.info(f"总迭代次数: {result.num_iterations}")
    logger.info(f"SRAF 数量: {len(result.all_srafs)}")


def main():
    """主函数：运行完整的 OPC 工作流示例"""

    logger = setup_logger('opc_demo', log_file='results/opc_workflow.log')
    logger.info("=" * 60)
    logger.info("OPC (Optical Proximity Correction) 工作流示例")
    logger.info("=" * 60)

    output_dir = Path('results') / 'opc_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 1. 配置加载 ==========
    logger.info("\n步骤 1: 加载配置")

    config_path = Path(parent_dir) / 'config' / 'opc_default.yaml'
    opc_config, optical_system = setup_opc_config_from_yaml(str(config_path))

    logger.info(f"OPC 配置:")
    logger.info(f"  EPE 阈值: {opc_config.epe_threshold} nm")
    logger.info(f"  最大迭代: {opc_config.max_iterations}")
    logger.info(f"  SRAF: {'启用' if opc_config.sraf_enable else '禁用'}")
    logger.info(f"  MaskOptimizer: {'启用' if opc_config.optimizer_enable else '禁用'}")

    # ========== 2. 创建测试图案 ==========
    logger.info("\n步骤 2: 创建测试图案")

    pattern_types = ['line_space', 'l_shaped', 'contact_hole']

    for pattern_type in pattern_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理图案: {pattern_type}")
        logger.info(f"{'='*60}")

        target, desc = create_demo_pattern(
            pattern_type=pattern_type)
        initial_mask = target.copy()

        logger.info(f"图案: {desc}")
        logger.info(f"尺寸: {target.shape}")
        logger.info(f"目标图案中非零像素: {np.sum(target > 0.5)}")

        # ========== 3. 运行 OPC 工作流 ==========
        logger.info(f"\n步骤 3: 运行 OPC 工作流")

        result = run_opc_workflow(
            initial_mask, target,
            config=opc_config,
            optical_system=optical_system
        )

        # ========== 4. 结果分析 ==========
        logger.info(f"\n步骤 4: 结果分析")

        pattern_output_dir = output_dir / pattern_type
        pattern_output_dir.mkdir(parents=True, exist_ok=True)

        analysis = analyze_results(result, pattern_output_dir)

        # ========== 5. 可视化 ==========
        logger.info(f"\n步骤 5: 结果可视化")

        visualize_results(
            result, target, pattern_output_dir, optical_system,
            pixel_size=opc_config.pixel_size
        )

        # 保存掩模
        np.save(pattern_output_dir / 'initial_mask.npy', result.initial_mask)
        np.save(pattern_output_dir / 'corrected_mask.npy', result.corrected_mask)

        logger.info(f"结果已保存到: {pattern_output_dir}")

    # ========== 6. 批量处理完成 ==========
    logger.info("\n" + "="*60)
    logger.info("所有图案处理完成！")
    logger.info("="*60)
    logger.info(f"结果目录: {output_dir}")
    logger.info("\n使用说明:")
    logger.info("  - mask_comparison.png: 初始掩模 vs 目标图案对比")
    logger.info("  - wafer_comparison.png: 初始 vs 校正后晶圆成像对比")
    logger.info("  - opc_analysis.json: 详细分析报告")
    logger.info("  - initial_mask.npy: 初始掩模数据")
    logger.info("  - corrected_mask.npy: 校正后掩模数据")


if __name__ == '__main__':
    main()
