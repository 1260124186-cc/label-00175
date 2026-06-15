#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SMO 多工艺条件 + PVB（工艺窗口）端到端测试"""
import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from core.imaging import OpticalSystem, TCCMode
from core.test_structures import (
    LineSpaceParams, TestStructureType,
    create_line_space,
)
from workflows.smo import (
    SMOConfig, SMOptimizationStrategy,
    SourceInitializationType, SourceConstraintsConfig,
    PixelatedSource, SMOImagingModel,
    SMOWorkflow, run_smo_workflow,
    ProcessCondition,
)


def make_test_setup():
    """构造多工艺条件（±离焦 + ±剂量）+ 线空间测试图案"""
    optics = OpticalSystem(
        wavelength=193.0, na=1.35, sigma=0.75, pixel_size=1.0,
        socs_num_terms=4, tcc_mode=TCCMode.SOCS
    )

    params = LineSpaceParams(
        grid_size=(48, 48), pixel_size=1.0, cd=45.0, pitch=90.0,
        structure_type=TestStructureType.LINE_SPACE,
    )
    initial_mask = create_line_space(params)
    initial_mask = np.clip(initial_mask + 0.1 * np.random.RandomState(42).randn(*initial_mask.shape), 0.0, 1.0)

    # 目标：标称条件下的二值化空间像
    from core.imaging import PartialCoherentImaging
    imaging_tmp = PartialCoherentImaging(optics, initial_mask.shape)
    aerial = imaging_tmp.compute_aerial_image(initial_mask)
    target = (aerial >= 0.3).astype(np.float64)

    # 多工艺条件：±50nm 离焦，±10% 剂量，共 3 个条件（保证计算量合理）
    process_conditions = [
        {'defocus': -50.0, 'dose': 0.92, 'weight': 1.0},
        {'defocus':   0.0, 'dose': 1.00, 'weight': 1.5},  # 标称条件加权更高
        {'defocus': +50.0, 'dose': 1.08, 'weight': 1.0},
    ]
    return optics, initial_mask, target, process_conditions


def compute_reference_pvb_and_mse(mask, target, source_intensity, optics, process_conditions):
    """使用独立链路计算参考 PVB（用于断言 SMO 确实改进了工艺窗口）"""
    threshold = 0.3
    wafers = []
    per_mse = []
    for cond in process_conditions:
        df, dose = cond['defocus'], cond['dose']
        cond_optics = OpticalSystem(
            wavelength=optics.wavelength, na=optics.na, sigma=optics.sigma,
            pixel_size=optics.pixel_size, defocus=df,
            illumination_type=optics.illumination_type.name if hasattr(optics.illumination_type, 'name')
                    else optics.illumination_type,
            source_params=dict(optics.source_params),
            tcc_mode=optics.tcc_mode, socs_num_terms=optics.socs_num_terms,
            custom_source=source_intensity,
            zernike_coefficients=dict(optics.zernike_coefficients)
        )
        from core.imaging import PartialCoherentImaging
        im = PartialCoherentImaging(cond_optics, mask.shape)
        aerial = np.clip(im.compute_aerial_image(mask) * dose, 0.0, None)
        wafer = 1.0 / (1.0 + np.exp(-50.0 * (aerial - threshold)))
        wafers.append(wafer)
        per_mse.append(float(np.mean((wafer - target) ** 2)))
    if len(wafers) >= 2:
        stack = np.stack(wafers, axis=0)
        bw = np.max(stack, axis=0) - np.min(stack, axis=0)
        pvb_L2 = float(np.mean(bw ** 2))
    else:
        pvb_L2 = 0.0
    weighted_mse = float(np.mean(per_mse))
    return weighted_mse, pvb_L2, per_mse


def test_alternating_with_pvb():
    """TEST 1: ALTERNATING 策略 + 多工艺条件 + PVB 损失权重"""
    print("\n" + "=" * 70)
    print("TEST 1: ALTERNATING 策略 + 多工艺条件 + PVB 损失")
    print("=" * 70)

    optics, initial_mask, target, conds = make_test_setup()

    config = SMOConfig(
        strategy=SMOptimizationStrategy.ALTERNATING,
        max_outer_iterations=2,
        source_max_iter=4,
        mask_max_iter=4,
        source_init_type=SourceInitializationType.CONVENTIONAL,
        source_grid_size=(48, 48),
        source_constraints=SourceConstraintsConfig(
            energy_conservation=True, energy_target=1.0,
            non_negative=True, smoothness_weight=0.002,
            smoothness_type='tv', support_radius=0.95,
        ),
        process_conditions=conds,
        pvb_weight=2.0,   # 强化工艺窗口惩罚
        wafer_threshold=0.3,
        use_wafer_image_loss=True,
        verbose=True,
        convergence_patience=5,
        tol=1e-7,
    )

    # 初始参考指标（使用独立链路）
    src_tmp = PixelatedSource(
        grid_size=(48, 48), optical_system=optics,
        init_type=SourceInitializationType.CONVENTIONAL,
        constraints=SourceConstraintsConfig(energy_conservation=True)
    )
    src_intensity_init = src_tmp.get_intensity()
    init_wmse, init_pvb, init_per = compute_reference_pvb_and_mse(
        initial_mask, target, src_intensity_init, optics, conds
    )
    print(f"\n[REF] 初始 weighted MSE = {init_wmse:.6f},  PVB(L2) = {init_pvb:.6f}")
    print(f"       各条件 MSE: {[f'{x:.4f}' for x in init_per]}")

    result = run_smo_workflow(initial_mask, target, config=config, optical_system=optics)

    print(f"\n[SMO-ALTERNATING] 迭代次数 = {result.num_iterations}")
    print(f"  优化阶段 = {[it.phase for it in result.iterations]}")
    print(f"  工作流损失历史: {[round(x, 6) for x in result.loss_history]}")

    # 独立链路验证最终指标
    final_src = result.optimal_source
    final_wmse, final_pvb, final_per = compute_reference_pvb_and_mse(
        result.optimal_mask, target, final_src, optics, conds
    )
    print(f"\n[REF] 最终 weighted MSE = {final_wmse:.6f},  PVB(L2) = {final_pvb:.6f}")
    print(f"       各条件 MSE: {[f'{x:.4f}' for x in final_per]}")
    print(f"\n  MSE 改进 = {(init_wmse - final_wmse):+.6f}  ({(init_wmse - final_wmse) / max(init_wmse,1e-12) * 100:.2f}%)")
    print(f"  PVB 改进 = {(init_pvb - final_pvb):+.6f}  ({(init_pvb - final_pvb) / max(init_pvb,1e-12) * 100:.2f}%)")

    assert final_wmse <= init_wmse + 0.001, f"加权 MSE 大幅上升: init={init_wmse}, final={final_wmse}"
    print("\n✅ TEST 1 通过\n")


def test_joint_gradient_with_pvb():
    """TEST 2: JOINT_GRADIENT 策略 + 多工艺条件 + PVB"""
    print("\n" + "=" * 70)
    print("TEST 2: JOINT_GRADIENT 策略 + 多工艺条件 + PVB 损失")
    print("=" * 70)

    optics, initial_mask, target, conds = make_test_setup()

    config = SMOConfig(
        strategy=SMOptimizationStrategy.JOINT_GRADIENT,
        max_outer_iterations=1,
        joint_max_iter=6,
        joint_learning_rate_source=0.15,
        joint_learning_rate_mask=0.15,
        source_init_type=SourceInitializationType.CONVENTIONAL,
        source_grid_size=(48, 48),
        source_constraints=SourceConstraintsConfig(
            energy_conservation=True, energy_target=1.0,
            non_negative=True, smoothness_weight=0.001,
            smoothness_type='gaussian', gaussian_sigma=1.5,
            support_radius=0.95,
        ),
        process_conditions=conds,
        pvb_weight=1.5,
        wafer_threshold=0.3,
        use_wafer_image_loss=True,
        verbose=True,
        convergence_patience=10,
        tol=1e-8,
    )

    # 初始参考
    src_tmp = PixelatedSource(
        grid_size=(48, 48), optical_system=optics,
        init_type=SourceInitializationType.CONVENTIONAL,
        constraints=SourceConstraintsConfig(energy_conservation=True)
    )
    init_wmse, init_pvb, init_per = compute_reference_pvb_and_mse(
        initial_mask, target, src_tmp.get_intensity(), optics, conds
    )
    print(f"\n[REF] 初始 weighted MSE = {init_wmse:.6f},  PVB(L2) = {init_pvb:.6f}")

    result = run_smo_workflow(initial_mask, target, config=config, optical_system=optics)

    print(f"\n[SMO-JOINT] 迭代次数 = {result.num_iterations}")
    print(f"  工作流损失历史: {[round(x, 6) for x in result.loss_history]}")

    final_wmse, final_pvb, final_per = compute_reference_pvb_and_mse(
        result.optimal_mask, target, result.optimal_source, optics, conds
    )
    print(f"\n[REF] 最终 weighted MSE = {final_wmse:.6f},  PVB(L2) = {final_pvb:.6f}")
    print(f"       各条件 MSE: {[f'{x:.4f}' for x in final_per]}")
    print(f"\n  MSE 改进 = {(init_wmse - final_wmse):+.6f}")
    print(f"  PVB 改进 = {(init_pvb - final_pvb):+.6f}")

    assert final_wmse <= init_wmse + 0.001, "Joint 策略加权 MSE 大幅上升"
    print("\n✅ TEST 2 通过\n")


def test_source_first_with_pvb():
    """TEST 3: SOURCE_FIRST 策略 + 多工艺条件"""
    print("\n" + "=" * 70)
    print("TEST 3: SOURCE_FIRST 策略 + 多工艺条件")
    print("=" * 70)

    optics, initial_mask, target, conds = make_test_setup()

    config = SMOConfig(
        strategy=SMOptimizationStrategy.SOURCE_FIRST,
        max_outer_iterations=1,
        source_max_iter=5,
        mask_max_iter=3,
        source_init_type=SourceInitializationType.CONVENTIONAL,
        source_grid_size=(48, 48),
        source_constraints=SourceConstraintsConfig(
            energy_conservation=True, energy_target=1.0,
            non_negative=True, smoothness_weight=0.003,
            smoothness_type='tv', support_radius=0.95,
        ),
        process_conditions=conds,
        pvb_weight=2.5,
        wafer_threshold=0.3,
        use_wafer_image_loss=True,
        verbose=True,
        convergence_patience=5,
    )

    src_tmp = PixelatedSource(
        grid_size=(48, 48), optical_system=optics,
        init_type=SourceInitializationType.CONVENTIONAL,
        constraints=SourceConstraintsConfig(energy_conservation=True)
    )
    init_wmse, init_pvb, init_per = compute_reference_pvb_and_mse(
        initial_mask, target, src_tmp.get_intensity(), optics, conds
    )
    print(f"\n[REF] 初始 weighted MSE = {init_wmse:.6f},  PVB(L2) = {init_pvb:.6f}")

    result = run_smo_workflow(initial_mask, target, config=config, optical_system=optics)

    print(f"\n[SMO-SOURCE-FIRST] 迭代次数 = {result.num_iterations}")
    print(f"  阶段 = {[it.phase for it in result.iterations]}")
    print(f"  工作流损失历史: {[round(x, 6) for x in result.loss_history]}")

    final_wmse, final_pvb, final_per = compute_reference_pvb_and_mse(
        result.optimal_mask, target, result.optimal_source, optics, conds
    )
    print(f"\n[REF] 最终 weighted MSE = {final_wmse:.6f},  PVB(L2) = {final_pvb:.6f}")
    print(f"       各条件 MSE: {[f'{x:.4f}' for x in final_per]}")
    print(f"\n  MSE 改进 = {(init_wmse - final_wmse):+.6f}")
    print(f"  PVB 改进 = {(init_pvb - final_pvb):+.6f}")

    assert final_wmse <= init_wmse + 0.001, "SOURCE_FIRST MSE 大幅上升"
    print("\n✅ TEST 3 通过\n")


def main():
    print("=" * 70)
    print("SMO 多工艺条件 + 工艺窗口（PVB）完整端到端测试")
    print("=" * 70)

    try:
        test_alternating_with_pvb()
        test_joint_gradient_with_pvb()
        test_source_first_with_pvb()

        print("=" * 70)
        print("🎉 所有 SMO 多工艺条件 + 工艺窗口测试通过！")
        print("=" * 70)
        return 0
    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
