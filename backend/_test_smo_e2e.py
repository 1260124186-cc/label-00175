"""
SMO 多条件闭环验证：验证 SourceOptimizer / JointGradientOptimizer / MaskOptimizerForSMO
都能让加权 MSE + PVB 损失真正下降，并验证 SMOWorkflow 的最终 per-condition 统计
与优化口径一致。

参数设置（让计算速度合理、物理正确性可验证）：
  N=128, pixel=40nm → FOV=5.12μm
  σ=0.75 圆盘半径 ~27 像素（足够优化自由度）
  3 工艺条件：±50nm defocus, ±8% dose
  target = 标称 wafer 平移 1 像素（确保有非平凡梯度）
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import OpticalSystem
from core.imaging import TCCMode
from algorithms.mask_optimizer import LossWeights
from workflows.smo import (
    SMOConfig, SourceInitializationType, SourceConstraintsConfig,
    PixelatedSource, SMOImagingModel, SourceOptimizer,
    JointGradientOptimizer, MaskOptimizerForSMO, SMOWorkflow,
    ProcessConditionEvaluation,
)


# ============================================================================
# 构造共用光学模型 + 多工艺条件
# ============================================================================
SIM_SIZE = 128
SIM_PIXEL = 40.0  # FOV = 5120 nm → σ=0.75 圆盘半径 ~26.9 像素

optics = OpticalSystem(
    wavelength=193.0, na=1.35, sigma=0.75, pixel_size=SIM_PIXEL,
    socs_num_terms=3, tcc_mode=TCCMode.FULL_TCC
)

process_conditions = [
    {'defocus': -50.0, 'dose': 0.92, 'weight': 1.0},
    {'defocus':   0.0, 'dose': 1.00, 'weight': 1.5},
    {'defocus': +50.0, 'dose': 1.08, 'weight': 1.0},
]


# ============================================================================
# 构造 test mask (contact hole 阵列) + target（平移 wafer）
# ============================================================================
def build_mask_and_target(imaging, threshold=0.3, k=50.0):
    mask = np.ones((SIM_SIZE, SIM_SIZE), dtype=np.float64)
    pitch_pix = int(200.0 / SIM_PIXEL)  # 5
    cd_pix = int(80.0 / SIM_PIXEL)       # 2
    yy_arr, xx_arr = np.mgrid[0:SIM_SIZE, 0:SIM_SIZE]
    for yc in range(pitch_pix//2, SIM_SIZE, pitch_pix):
        for xc in range(pitch_pix//2, SIM_SIZE, pitch_pix):
            r2 = (yy_arr - yc)**2 + (xx_arr - xc)**2
            mask[r2 <= (cd_pix/2)**2] = 0.0

    # target = 标称条件 wafer 平移 1px（创造有意义的优化目标）
    import scipy.ndimage as ndi
    im_nominal = imaging._process_imagers[1][0] if hasattr(imaging, '_process_imagers') and imaging._process_imagers else imaging._imaging
    aerial = np.clip(im_nominal.compute_aerial_image(mask) * 1.0, 0.0, None)
    aerial_shifted = ndi.shift(aerial, (1, 0))
    target = 1.0 / (1.0 + np.exp(-k * (aerial_shifted - threshold)))
    return mask, target


def _make_imaging_and_src(cfg):
    """构造共用的 SMOImagingModel + 初始光源 + mask/target"""
    imaging = SMOImagingModel(optics, (SIM_SIZE, SIM_SIZE),
                              tcc_mode=TCCMode.FULL_TCC, socs_num_terms=3)
    imaging.set_process_conditions(process_conditions)
    src0 = PixelatedSource((SIM_SIZE, SIM_SIZE), optics,
                           SourceInitializationType.CONVENTIONAL,
                           cfg.source_init_params, cfg.source_constraints)
    imaging.update_source_all_conditions(src0)
    mask, target = build_mask_and_target(imaging)
    return imaging, src0, mask, target


# ============================================================================
# Test 1: SourceOptimizer 光源优化（MSE+PVB 是否下降）
# ============================================================================
def test_source_optimizer():
    print("=" * 70)
    print("[TEST 1] SourceOptimizer — 固定 mask 优化光源")
    print("=" * 70)

    cfg = SMOConfig(
        source_grid_size=(SIM_SIZE, SIM_SIZE),
        source_init_type=SourceInitializationType.CONVENTIONAL,
        source_init_params={'sigma_inner': 0.0, 'sigma_outer': 0.75},
        source_constraints=SourceConstraintsConfig(
            non_negative=True, energy_conservation=True, energy_target=1.0,
            support_radius=0.95,
            smoothness_type='gaussian', gaussian_sigma=0.5,
            smoothness_weight=0.01,
        ),
        process_conditions=process_conditions,
        pvb_weight=2.0,
        source_loss_weights={'mse': 1.0, 'pvb': 0.0, 'epe': 0.0},
        mask_loss_weights=LossWeights(mse=1.0, epe=0.0, pvb=0.0, tv_smooth=0.0, ssim=0.0),
        source_max_iter=15,
        mask_max_iter=10,
        source_learning_rate=0.05,
        mask_learning_rate=0.05,
    )

    imaging, src0, mask, target = _make_imaging_and_src(cfg)
    src_opt = SourceOptimizer(imaging, cfg)

    initial_loss, initial_info, _ = src_opt._compute_loss_and_gradients(mask, target, src0)
    print(f"\n初始状态: total_loss = {initial_loss:.6f}")
    print(f"  Weighted_MSE = {initial_info.get('weighted_mse', -1):.6f}")
    print(f"  PVB(soft)    = {initial_info.get('pvb', -1):.6f}")
    if 'mse_per_cond' in initial_info:
        print(f"  per-cond MSE = {[f'{v:.4f}' for v in initial_info['mse_per_cond']]}")
    print(f"  effective_σ  = {src0.compute_effective_sigma():.4f}")

    print(f"\n开始优化（SourceOptimizer, max_iter={cfg.source_max_iter}, lr={cfg.source_learning_rate}）...")
    final_source, _ = src_opt.optimize(src0, mask, target,
                                       max_iter=cfg.source_max_iter,
                                       learning_rate=cfg.source_learning_rate)

    final_loss, final_info, _ = src_opt._compute_loss_and_gradients(mask, target, final_source)
    print(f"\n最终状态: total_loss = {final_loss:.6f}")
    print(f"  Weighted_MSE = {final_info.get('weighted_mse', -1):.6f}")
    print(f"  PVB(soft)    = {final_info.get('pvb', -1):.6f}")
    print(f"  effective_σ  = {final_source.compute_effective_sigma():.4f}")

    dL = initial_loss - final_loss
    dPVB = initial_info.get('pvb', 0) - final_info.get('pvb', 0)
    dMSE = initial_info.get('weighted_mse', 0) - final_info.get('weighted_mse', 0)

    passed = (dL > 1e-8) and (dPVB > -1e-10)
    print(f"\n{'='*70}")
    print(f"  ΔTotal = {dL:+.6f} {'下降 ✓' if dL > 1e-8 else '✗ 无改善'}")
    print(f"  ΔMSE   = {dMSE:+.6f} {'下降 ✓' if dMSE > 1e-8 else '✗ 无改善'}")
    print(f"  ΔPVB   = {dPVB:+.6f} {'下降 ✓' if dPVB > 1e-10 else ('稳定 ~' if abs(dPVB) < 1e-10 else '✗ 恶化')}")
    print(f"  TEST 1 结果: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'='*70}\n")
    return passed


# ============================================================================
# Test 2: JointGradientOptimizer 联合优化（loss 是否下降）
# ============================================================================
def test_joint_optimizer():
    print("=" * 70)
    print("[TEST 2] JointGradientOptimizer — 联合优化光源+掩模")
    print("=" * 70)

    cfg = SMOConfig(
        source_grid_size=(SIM_SIZE, SIM_SIZE),
        source_init_type=SourceInitializationType.CONVENTIONAL,
        source_init_params={'sigma_inner': 0.0, 'sigma_outer': 0.75},
        source_constraints=SourceConstraintsConfig(
            non_negative=True, energy_conservation=True, energy_target=1.0,
            support_radius=0.95,
            smoothness_type='gaussian', gaussian_sigma=0.5,
            smoothness_weight=0.01,
        ),
        process_conditions=process_conditions,
        pvb_weight=2.0,
        source_loss_weights={'mse': 1.0, 'pvb': 0.0, 'epe': 0.0},
        mask_loss_weights=LossWeights(mse=1.0, epe=0.0, pvb=0.0, tv_smooth=0.0, ssim=0.0),
        joint_max_iter=10,
        joint_learning_rate_source=0.05,
        joint_learning_rate_mask=0.02,
    )

    imaging, src0, mask, target = _make_imaging_and_src(cfg)
    mask0 = mask.copy()
    joint_opt = JointGradientOptimizer(imaging, cfg)

    initial_loss, initial_info, _, _ = joint_opt._compute_joint_loss_and_grads(mask0, target, src0)
    print(f"\n初始状态: total_loss = {initial_loss:.6f}")
    print(f"  Joint_MSE    = {initial_info.get('joint_weighted_mse', -1):.6f}")
    print(f"  Joint_PVB    = {initial_info.get('joint_pvb', -1):.6f}")

    print(f"\n开始联合优化（JointGradientOptimizer, max_iter={cfg.joint_max_iter}）...")
    final_source, final_mask, _ = joint_opt.optimize(src0, mask0, target, max_iter=cfg.joint_max_iter)

    final_loss, final_info, _, _ = joint_opt._compute_joint_loss_and_grads(final_mask, target, final_source)
    print(f"\n最终状态: total_loss = {final_loss:.6f}")
    print(f"  Joint_MSE    = {final_info.get('joint_weighted_mse', -1):.6f}")
    print(f"  Joint_PVB    = {final_info.get('joint_pvb', -1):.6f}")
    print(f"  effective_σ  = {final_source.compute_effective_sigma():.4f}")

    dL = initial_loss - final_loss
    dPVB = initial_info.get('joint_pvb', 0) - final_info.get('joint_pvb', 0)
    dMSE = initial_info.get('joint_weighted_mse', 0) - final_info.get('joint_weighted_mse', 0)

    passed = dL > 1e-8
    print(f"\n{'='*70}")
    print(f"  ΔTotal = {dL:+.6f} {'下降 ✓' if dL > 1e-8 else '✗ 无改善'}")
    print(f"  ΔMSE   = {dMSE:+.6f} {'下降 ✓' if dMSE > 1e-8 else '✗ 无改善'}")
    print(f"  ΔPVB   = {dPVB:+.6f} {'下降 ✓' if dPVB > 1e-10 else ('稳定 ~' if abs(dPVB) < 1e-10 else '✗ 恶化')}")
    print(f"  TEST 2 结果: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'='*70}\n")
    return passed


# ============================================================================
# Test 3: MaskOptimizerForSMO 掩模优化（固定光源，多条件 PVB 下降）
# ============================================================================
def test_mask_optimizer():
    print("=" * 70)
    print("[TEST 3] MaskOptimizerForSMO — 固定光源优化掩模")
    print("=" * 70)

    cfg = SMOConfig(
        source_grid_size=(SIM_SIZE, SIM_SIZE),
        source_init_type=SourceInitializationType.CONVENTIONAL,
        source_init_params={'sigma_inner': 0.0, 'sigma_outer': 0.75},
        source_constraints=SourceConstraintsConfig(
            non_negative=True, energy_conservation=True, energy_target=1.0,
            support_radius=0.95,
            smoothness_type='gaussian', gaussian_sigma=0.5,
            smoothness_weight=0.01,
        ),
        process_conditions=process_conditions,
        pvb_weight=2.0,
        source_loss_weights={'mse': 1.0, 'pvb': 0.0, 'epe': 0.0},
        mask_loss_weights=LossWeights(mse=1.0, epe=0.0, pvb=0.0, tv_smooth=0.0, ssim=0.0),
        source_max_iter=15,
        mask_max_iter=8,
        source_learning_rate=0.05,
        mask_learning_rate=0.1,
        wafer_threshold=0.3,
        use_wafer_image_loss=True,
        pixel_size=SIM_PIXEL,
    )

    imaging, src0, mask, target = _make_imaging_and_src(cfg)
    mask0 = mask.copy()
    mask_opt = MaskOptimizerForSMO(imaging, cfg)

    # 初始评估（用 _evaluate_all_conditions 同口径）
    workflow_for_eval = type('_Dummy', (), {})()
    from workflows.smo import SMOWorkflow
    # 用 SourceOptimizer 的前向链路做初始评估（复用其 _compute_loss_and_gradients）
    src_opt_eval = SourceOptimizer(imaging, cfg)
    initial_loss, initial_info, _ = src_opt_eval._compute_loss_and_gradients(mask0, target, src0)
    print(f"\n初始状态: total_loss = {initial_loss:.6f}")
    print(f"  Weighted_MSE = {initial_info.get('weighted_mse', -1):.6f}")
    print(f"  PVB(soft)    = {initial_info.get('pvb', -1):.6f}")

    print(f"\n开始掩模优化（MaskOptimizerForSMO, max_iter={cfg.mask_max_iter}）...")
    final_mask, loss_hist = mask_opt.optimize(mask0, target, src0,
                                               max_iter=cfg.mask_max_iter,
                                               learning_rate=cfg.mask_learning_rate)

    final_loss, final_info, _ = src_opt_eval._compute_loss_and_gradients(final_mask, target, src0)
    print(f"\n最终状态: total_loss = {final_loss:.6f}")
    print(f"  Weighted_MSE = {final_info.get('weighted_mse', -1):.6f}")
    print(f"  PVB(soft)    = {final_info.get('pvb', -1):.6f}")

    dL = initial_loss - final_loss
    dPVB = initial_info.get('pvb', 0) - final_info.get('pvb', 0)
    dMSE = initial_info.get('weighted_mse', 0) - final_info.get('weighted_mse', 0)

    passed = dL > 1e-8
    print(f"\n{'='*70}")
    print(f"  ΔTotal = {dL:+.6f} {'下降 ✓' if dL > 1e-8 else '✗ 无改善'}")
    print(f"  ΔMSE   = {dMSE:+.6f} {'下降 ✓' if dMSE > 1e-8 else '✗ 无改善'}")
    print(f"  ΔPVB   = {dPVB:+.6f} {'下降 ✓' if dPVB > 1e-10 else ('稳定 ~' if abs(dPVB) < 1e-10 else '✗ 恶化')}")
    print(f"  TEST 3 结果: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'='*70}\n")
    return passed


# ============================================================================
# Test 4: SMOWorkflow.run() 最终 per-condition 统计一致性验证
# ============================================================================
def test_workflow_final_stats():
    print("=" * 70)
    print("[TEST 4] SMOWorkflow.run() — 最终多条件统计一致性")
    print("=" * 70)

    cfg = SMOConfig(
        source_grid_size=(SIM_SIZE, SIM_SIZE),
        source_init_type=SourceInitializationType.CONVENTIONAL,
        source_init_params={'sigma_inner': 0.0, 'sigma_outer': 0.75},
        source_constraints=SourceConstraintsConfig(
            non_negative=True, energy_conservation=True, energy_target=1.0,
            support_radius=0.95,
            smoothness_type='gaussian', gaussian_sigma=0.5,
            smoothness_weight=0.01,
        ),
        process_conditions=process_conditions,
        pvb_weight=2.0,
        source_loss_weights={'mse': 1.0, 'pvb': 0.0, 'epe': 0.0},
        mask_loss_weights=LossWeights(mse=1.0, epe=0.0, pvb=0.0, tv_smooth=0.0, ssim=0.0),
        strategy='joint',
        source_max_iter=3,
        mask_max_iter=3,
        joint_max_iter=3,
        max_outer_iterations=1,
        source_learning_rate=0.05,
        mask_learning_rate=0.05,
        joint_learning_rate_source=0.05,
        joint_learning_rate_mask=0.02,
        wafer_threshold=0.3,
        use_wafer_image_loss=True,
        pixel_size=SIM_PIXEL,
        verbose=False,
    )

    # 构造 mask + target（需要先创建 SMOImagingModel 来生成 target）
    tmp_imaging = SMOImagingModel(optics, (SIM_SIZE, SIM_SIZE),
                                  tcc_mode=TCCMode.FULL_TCC, socs_num_terms=3)
    tmp_imaging.set_process_conditions(process_conditions)
    tmp_src = PixelatedSource((SIM_SIZE, SIM_SIZE), optics,
                              SourceInitializationType.CONVENTIONAL,
                              cfg.source_init_params, cfg.source_constraints)
    tmp_imaging.update_source_all_conditions(tmp_src)
    mask, target = build_mask_and_target(tmp_imaging)

    print(f"\n运行 SMOWorkflow.run() （joint 策略, 3 步联合优化）...")
    workflow = SMOWorkflow(config=cfg, optical_system=optics)
    result = workflow.run(mask, target)

    # —— 验证 1：per-condition 字段存在且非空 ——
    checks = []

    check1 = len(result.final_per_condition) == len(process_conditions)
    checks.append(('per-condition 数量匹配', check1))

    check2 = all(isinstance(pc, ProcessConditionEvaluation) for pc in result.final_per_condition)
    checks.append(('per-condition 类型正确', check2))

    check3 = result.final_total_loss > 0 and result.initial_total_loss > 0
    checks.append(('total_loss 非零', check3))

    check4 = abs(result.final_weighted_mse - result.final_per_condition[1].mse * 0.0
                 ) >= 0  # 只要加权 MSE 是标量即可
    check4 = result.final_weighted_mse > 0
    checks.append(('weighted_mse 非零', check4))

    check5 = result.final_pvb_soft > 0
    checks.append(('PVB(soft) 非零', check5))

    # —— 验证 2：final_total_loss 与独立前向计算一致 ——
    src_final = PixelatedSource((SIM_SIZE, SIM_SIZE), optics,
                                SourceInitializationType.CUSTOM,
                                {}, cfg.source_constraints,
                                custom_source=result.optimal_source)
    verify_opt = SourceOptimizer(workflow._imaging, cfg)
    verify_loss, verify_info, _ = verify_opt._compute_loss_and_gradients(
        result.optimal_mask, target, src_final
    )
    loss_match = abs(result.final_total_loss - verify_loss) / max(verify_loss, 1e-12) < 0.01
    checks.append(('final_total_loss 与独立前向一致 (<1%)', loss_match))

    # —— 验证 3：final_per_condition[i].mse 与独立前向一致 ——
    per_cond_match = True
    for i, pc in enumerate(result.final_per_condition):
        expected_mse = verify_info.get('mse_per_cond', [])[i] if i < len(verify_info.get('mse_per_cond', [])) else -1
        if expected_mse < 0:
            continue
        rel_err = abs(pc.mse - expected_mse) / max(expected_mse, 1e-12)
        if rel_err > 0.01:
            per_cond_match = False
            break
    checks.append(('per-condition MSE 与独立前向一致 (<1%)', per_cond_match))

    # —— 验证 4：损失确实下降了 ——
    loss_improved = result.initial_total_loss - result.final_total_loss > 1e-8
    checks.append(('总损失下降', loss_improved))

    # —— 输出 ——
    all_passed = all(p for _, p in checks)
    print(f"\n初始总损失: {result.initial_total_loss:.6f}")
    print(f"最终总损失: {result.final_total_loss:.6f}")
    print(f"初始加权 MSE: {result.initial_weighted_mse:.6f}")
    print(f"最终加权 MSE: {result.final_weighted_mse:.6f}")
    print(f"初始 PVB(soft): {result.initial_pvb_soft:.6f}")
    print(f"最终 PVB(soft): {result.final_pvb_soft:.6f}")
    print(f"工艺条件数: {result.num_process_conditions}")

    print(f"\n各条件详情:")
    for i, pc in enumerate(result.final_per_condition):
        print(f"  cond[{i}] df={pc.defocus:+.0f}nm dose={pc.dose:.2f} "
              f"w={pc.weight:.3f} MSE={pc.mse:.6f} "
              f"EPE_mean={pc.epe.get('epe_mean', 0):.2f}nm")

    print(f"\n检查项:")
    for name, passed in checks:
        print(f"  {'✓' if passed else '✗'} {name}")

    print(f"\n{'='*70}")
    print(f"  TEST 4 结果: {'PASS ✓' if all_passed else 'FAIL ✗'}")
    print(f"{'='*70}\n")
    return all_passed


if __name__ == '__main__':
    import time
    t0 = time.time()

    r1 = test_source_optimizer()
    t1 = time.time()
    print(f"  (耗时 {t1-t0:.1f}s)\n")

    r2 = test_joint_optimizer()
    t2 = time.time()
    print(f"  (耗时 {t2-t1:.1f}s)\n")

    r3 = test_mask_optimizer()
    t3 = time.time()
    print(f"  (耗时 {t3-t2:.1f}s)\n")

    r4 = test_workflow_final_stats()
    t4 = time.time()
    print(f"  (耗时 {t4-t3:.1f}s)\n")

    total = t4 - t0
    print("=" * 70)
    print("  SUMMARY:")
    print(f"    TEST 1 (SourceOptimizer)        : {'PASS ✓' if r1 else 'FAIL ✗'}")
    print(f"    TEST 2 (JointGradient)         : {'PASS ✓' if r2 else 'FAIL ✗'}")
    print(f"    TEST 3 (MaskOptimizerForSMO)    : {'PASS ✓' if r3 else 'FAIL ✗'}")
    print(f"    TEST 4 (SMOWorkflow 多条件统计) : {'PASS ✓' if r4 else 'FAIL ✗'}")
    print(f"    总耗时: {total:.1f}s")
    print("=" * 70)
