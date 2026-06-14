#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SMO (Source-Mask Optimization) 工作流示例：完整的光源-掩模协同优化流程演示

该示例展示如何使用 SMO 工作流模块完成：
1. 创建测试掩模图案（线/空间、L形拐角、接触孔等）
2. 配置 SMO 工作流参数（策略、光源约束、损失权重等）
3. 像素化光源初始化与约束投影
4. 交替优化 / 联合梯度下降
5. 结果可视化与分析（光源演化、掩模变化、EPE 改善）

运行方式：
    # 从 backend 目录下运行：
    python -m examples.run_smo
    # 或直接运行：
    python examples/run_smo.py
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json

from core.imaging import OpticalSystem, simulate_wafer_image, ProcessCondition
from core.litho_metrics import compute_epe
from core.test_structures import (
    TestStructureParams, TestStructureType, LineOrientation,
    create_test_structure, create_line_space, create_l_shaped_corner,
    create_contact_hole
)
from workflows.smo import (
    SMOConfig, SMOWorkflow, run_smo_workflow,
    SMOptimizationStrategy, SourceInitializationType,
    SourceConstraintsConfig,
    PixelatedSource, SMOImagingModel,
    SMOWorkflowResult, SMOIterationResult,
)
from utils.config import load_config, save_config
from utils.logger import setup_logger
from utils.visualization import (
    plot_mask, plot_wafer_image, plot_comparison
)

logger = setup_logger('smo_demo', log_file='results/smo_workflow.log')


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
        desc = "简单矩形结构"

    return target, desc


def setup_smo_config_from_yaml(
    config_path: Optional[str] = None
) -> Tuple[SMOConfig, OpticalSystem]:
    """
    从 YAML 配置文件设置 SMO 配置

    Args:
        config_path: 配置文件路径

    Returns:
        (SMO 配置, 光学系统配置)
    """
    if config_path is None:
        config_path = Path(parent_dir) / 'config' / 'smo_default.yaml'

    config_dict = load_config(config_path)

    smo_config = SMOConfig.from_dict(config_dict.get('smo', {}))

    opt_sys_params = config_dict.get('optical_system', {})
    optical_system = OpticalSystem(
        wavelength=opt_sys_params.get('wavelength', 193.0),
        na=opt_sys_params.get('na', 1.35),
        sigma=opt_sys_params.get('sigma', 0.75),
        pixel_size=opt_sys_params.get('pixel_size', 1.0),
        defocus=opt_sys_params.get('defocus', 0.0),
        magnification=opt_sys_params.get('magnification', 4.0),
        socs_num_terms=opt_sys_params.get('socs_num_terms', 8),
    )

    return smo_config, optical_system


def analyze_results(result: SMOWorkflowResult,
                    output_dir: Path) -> Dict[str, Any]:
    """
    分析 SMO 结果并生成报告

    Args:
        result: SMO 工作流结果
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
            'phase': iter_result.phase,
            'loss_before': iter_result.loss_before,
            'loss_after': iter_result.loss_after,
            'loss_improvement': iter_result.loss_improvement,
            'loss_improvement_ratio': iter_result.loss_improvement_ratio,
            'epe_before_mean': iter_result.epe_before.get('epe_mean', 0.0),
            'epe_after_mean': iter_result.epe_after.get('epe_mean', 0.0),
            'source_effective_sigma': iter_result.source_effective_sigma,
        }
        analysis['iterations'].append(iter_info)

    report_path = output_dir / 'smo_analysis.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"分析报告已保存: {report_path}")

    return analysis


def visualize_results(result: SMOWorkflowResult,
                      target: np.ndarray,
                      output_dir: Path,
                      optical_system: OpticalSystem,
                      pixel_size: float = 1.0):
    """
    可视化 SMO 结果

    Args:
        result: SMO 结果
        target: 目标图案
        output_dir: 输出目录
        optical_system: 光学系统
        pixel_size: 像素尺寸
    """
    fig_mask = plot_comparison(
        result.initial_mask, result.optimal_mask,
        titles=['初始掩模', '优化后掩模'])
    fig_mask.savefig(output_dir / 'mask_comparison.png', dpi=150)

    fig_source = plot_comparison(
        np.fft.fftshift(result.initial_source),
        np.fft.fftshift(result.optimal_source),
        titles=['初始光源分布', '优化后光源分布'])
    fig_source.savefig(output_dir / 'source_comparison.png', dpi=150)

    fig_wafer = plot_comparison(
        result.initial_wafer, result.optimal_wafer,
        titles=['初始晶圆成像', '优化后晶圆成像'])
    fig_wafer.savefig(output_dir / 'wafer_comparison.png', dpi=150)

    epe_initial = compute_epe(result.initial_wafer, target, pixel_size=pixel_size)
    epe_optimal = compute_epe(result.optimal_wafer, target, pixel_size=pixel_size)

    logger.info("\n" + "=" * 60)
    logger.info("EPE 改善详情")
    logger.info("=" * 60)
    logger.info(f"初始 EPE:")
    logger.info(f"  平均值: {epe_initial['epe_mean']:.3f} nm")
    logger.info(f"  最大值: {epe_initial['epe_max']:.3f} nm")
    logger.info(f"优化后 EPE:")
    logger.info(f"  平均值: {epe_optimal['epe_mean']:.3f} nm")
    logger.info(f"  最大值: {epe_optimal['epe_max']:.3f} nm")
    logger.info(f"改善量: {result.total_epe_improvement:.3f} nm "
                f"({result.total_epe_improvement_ratio * 100:.1f}%)")
    logger.info(f"外层迭代: {result.num_iterations}")
    logger.info(f"收敛: {'是' if result.converged else '否'} — {result.reason}")


def run_demo_alternating():
    """演示交替优化策略"""
    logger.info("\n" + "=" * 60)
    logger.info("演示 1: 交替优化策略 (Alternating)")
    logger.info("=" * 60)

    output_dir = Path('results') / 'smo_alternating'
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_size = (64, 64)
    target, desc = create_demo_pattern('line_space', grid_size=grid_size)
    initial_mask = target.copy()

    logger.info(f"图案: {desc}")
    logger.info(f"尺寸: {target.shape}")

    config = SMOConfig(
        strategy=SMOptimizationStrategy.ALTERNATING,
        max_outer_iterations=10,
        source_max_iter=30,
        mask_max_iter=50,
        source_init_type=SourceInitializationType.CONVENTIONAL,
        source_constraints=SourceConstraintsConfig(
            energy_conservation=True,
            smoothness_weight=0.01,
            non_negative=True,
        ),
        wafer_threshold=0.3,
        pixel_size=1.0,
        use_wafer_image_loss=True,
        verbose=True,
    )

    optical_system = OpticalSystem(
        wavelength=193.0,
        na=1.35,
        sigma=0.75,
        pixel_size=1.0,
        socs_num_terms=8,
    )

    result = run_smo_workflow(
        initial_mask, target,
        config=config,
        optical_system=optical_system,
    )

    analyze_results(result, output_dir)
    visualize_results(result, target, output_dir, optical_system,
                      pixel_size=config.pixel_size)

    np.save(output_dir / 'initial_mask.npy', result.initial_mask)
    np.save(output_dir / 'optimal_mask.npy', result.optimal_mask)
    np.save(output_dir / 'optimal_source.npy', result.optimal_source)

    return result


def run_demo_joint_gradient():
    """演示联合梯度下降策略"""
    logger.info("\n" + "=" * 60)
    logger.info("演示 2: 联合梯度下降策略 (Joint Gradient)")
    logger.info("=" * 60)

    output_dir = Path('results') / 'smo_joint_gradient'
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_size = (64, 64)
    target, desc = create_demo_pattern('contact_hole', grid_size=grid_size)
    initial_mask = target.copy()

    logger.info(f"图案: {desc}")
    logger.info(f"尺寸: {target.shape}")

    config = SMOConfig(
        strategy=SMOptimizationStrategy.JOINT_GRADIENT,
        max_outer_iterations=15,
        joint_max_iter=150,
        joint_learning_rate_source=0.003,
        joint_learning_rate_mask=0.008,
        source_init_type=SourceInitializationType.ANNULAR,
        source_constraints=SourceConstraintsConfig(
            energy_conservation=True,
            smoothness_weight=0.02,
            smoothness_type='gaussian',
            gaussian_sigma=1.5,
            non_negative=True,
        ),
        wafer_threshold=0.3,
        pixel_size=1.0,
        use_wafer_image_loss=True,
        verbose=True,
    )

    optical_system = OpticalSystem(
        wavelength=193.0,
        na=1.35,
        sigma=0.75,
        pixel_size=1.0,
        socs_num_terms=8,
    )

    result = run_smo_workflow(
        initial_mask, target,
        config=config,
        optical_system=optical_system,
    )

    analyze_results(result, output_dir)
    visualize_results(result, target, output_dir, optical_system,
                      pixel_size=config.pixel_size)

    np.save(output_dir / 'initial_mask.npy', result.initial_mask)
    np.save(output_dir / 'optimal_mask.npy', result.optimal_mask)
    np.save(output_dir / 'optimal_source.npy', result.optimal_source)

    return result


def run_demo_source_first():
    """演示先优化光源策略"""
    logger.info("\n" + "=" * 60)
    logger.info("演示 3: 先优化光源策略 (Source First)")
    logger.info("=" * 60)

    output_dir = Path('results') / 'smo_source_first'
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_size = (64, 64)
    target, desc = create_demo_pattern('l_shaped', grid_size=grid_size)
    initial_mask = target.copy()

    logger.info(f"图案: {desc}")
    logger.info(f"尺寸: {target.shape}")

    config = SMOConfig(
        strategy=SMOptimizationStrategy.SOURCE_FIRST,
        max_outer_iterations=10,
        source_max_iter=80,
        mask_max_iter=80,
        source_init_type=SourceInitializationType.QUASAR,
        source_constraints=SourceConstraintsConfig(
            energy_conservation=True,
            smoothness_weight=0.01,
            non_negative=True,
        ),
        wafer_threshold=0.3,
        pixel_size=1.0,
        use_wafer_image_loss=True,
        verbose=True,
    )

    optical_system = OpticalSystem(
        wavelength=193.0,
        na=1.35,
        sigma=0.75,
        pixel_size=1.0,
        socs_num_terms=8,
    )

    result = run_smo_workflow(
        initial_mask, target,
        config=config,
        optical_system=optical_system,
    )

    analyze_results(result, output_dir)
    visualize_results(result, target, output_dir, optical_system,
                      pixel_size=config.pixel_size)

    np.save(output_dir / 'initial_mask.npy', result.initial_mask)
    np.save(output_dir / 'optimal_mask.npy', result.optimal_mask)
    np.save(output_dir / 'optimal_source.npy', result.optimal_source)

    return result


def run_demo_from_yaml():
    """演示从 YAML 配置文件运行"""
    logger.info("\n" + "=" * 60)
    logger.info("演示 4: 从 YAML 配置文件运行")
    logger.info("=" * 60)

    config_path = Path(parent_dir) / 'config' / 'smo_default.yaml'
    if not config_path.exists():
        logger.warning(f"配置文件不存在: {config_path}，跳过此演示")
        return None

    output_dir = Path('results') / 'smo_from_yaml'
    output_dir.mkdir(parents=True, exist_ok=True)

    smo_config, optical_system = setup_smo_config_from_yaml(str(config_path))

    grid_size = (64, 64)
    target, desc = create_demo_pattern('line_space', grid_size=grid_size)
    initial_mask = target.copy()

    logger.info(f"图案: {desc}")
    logger.info(f"策略: {smo_config.strategy.value}")
    logger.info(f"最大外层迭代: {smo_config.max_outer_iterations}")

    result = run_smo_workflow(
        initial_mask, target,
        config=smo_config,
        optical_system=optical_system,
    )

    analyze_results(result, output_dir)
    visualize_results(result, target, output_dir, optical_system,
                      pixel_size=smo_config.pixel_size)

    return result


def main():
    """主函数：运行完整的 SMO 工作流示例"""

    logger.info("=" * 70)
    logger.info("SMO (Source-Mask Optimization) 工作流示例")
    logger.info("=" * 70)

    result1 = run_demo_alternating()

    result2 = run_demo_joint_gradient()

    result3 = run_demo_source_first()

    result4 = run_demo_from_yaml()

    logger.info("\n" + "=" * 70)
    logger.info("所有 SMO 演示完成！")
    logger.info("=" * 70)
    logger.info("结果目录:")
    logger.info("  results/smo_alternating/    - 交替优化策略")
    logger.info("  results/smo_joint_gradient/ - 联合梯度下降策略")
    logger.info("  results/smo_source_first/   - 先优化光源策略")
    logger.info("  results/smo_from_yaml/      - 从YAML配置运行")
    logger.info("\n输出文件说明:")
    logger.info("  mask_comparison.png    - 初始 vs 优化后掩模对比")
    logger.info("  source_comparison.png  - 初始 vs 优化后光源分布对比")
    logger.info("  wafer_comparison.png   - 初始 vs 优化后晶圆成像对比")
    logger.info("  smo_analysis.json      - 详细分析报告")
    logger.info("  *.npy                  - 数值数据（掩模、光源）")


if __name__ == '__main__':
    main()
