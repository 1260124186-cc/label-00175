#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SMO 调试脚本：检查光源和成像链路"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.imaging import OpticalSystem, TCCMode, generate_source, IlluminationType
from workflows.smo import (
    PixelatedSource, SourceInitializationType, SourceConstraintsConfig,
    SMOImagingModel,
)
from core.test_structures import (
    LineSpaceParams, TestStructureType, create_line_space,
)


def debug():
    optics = OpticalSystem(
        wavelength=193.0, na=1.35, sigma=0.75, pixel_size=1.0,
        socs_num_terms=4, tcc_mode=TCCMode.SOCS
    )
    print(f"[Optics] cutoff={optics.cutoff_frequency}, pixel_size={optics.pixel_size}")
    print(f"         wavelength={optics.wavelength}, NA={optics.na}, sigma={optics.sigma}")

    # ============== PixelatedSource 调试 (使用大尺寸模拟网格) ==============
    SIM_SIZE = 128
    SIM_PIXEL = 0.5  # 0.5nm/pixel → 64nm 模拟范围
    optics2 = OpticalSystem(
        wavelength=193.0, na=1.35, sigma=0.75, pixel_size=SIM_PIXEL,
        socs_num_terms=4, tcc_mode=TCCMode.SOCS
    )
    print(f"\n[Optics2] cutoff={optics2.cutoff_frequency}, pixel={optics2.pixel_size}")
    src = PixelatedSource(
        grid_size=(SIM_SIZE, SIM_SIZE),
        optical_system=optics2,
        init_type=SourceInitializationType.CONVENTIONAL,
        constraints=SourceConstraintsConfig(
            energy_conservation=True, energy_target=1.0,
            non_negative=True, support_radius=0.95,
        )
    )
    print(f"  光源形状: {src.intensity.shape}")
    print(f"  总能量:   {src.intensity.sum():.6f}")
    print(f"  最大值:   {src.intensity.max():.6f}")
    print(f"  非零像素数: {np.sum(src.intensity > 1e-12)}")
    print(f"  等效 sigma: {src.compute_effective_sigma():.4f}")
    print(f"  rho_norm 范围: [{src.rho_norm.min():.4f}, {src.rho_norm.max():.4f}]")

    # ============== PixelatedSource 调试 (使用大尺寸模拟网格) ==============
    # ★ 关键：FOV = N × pixel 至少 5μm，让 σ=0.75 圆盘有 ~27 像素
    # m_pix = 0.75 · (NA/λ) · FOV = 0.75 × 0.00699 × 5120 ≈ 26.88
    SIM_SIZE = 128
    SIM_PIXEL = 40.0  # 40nm/pixel → 128 × 40 = 5120 nm = 5.12 μm
    optics2 = OpticalSystem(
        wavelength=193.0, na=1.35, sigma=0.75, pixel_size=SIM_PIXEL,
        socs_num_terms=3, tcc_mode=TCCMode.FULL_TCC
    )
    expected_mpix = 0.75 * optics2.cutoff_frequency * (SIM_SIZE * SIM_PIXEL)
    print(f"\n[Optics2] cutoff={optics2.cutoff_frequency:.6f}, pixel={SIM_PIXEL}nm")
    print(f"  模拟 FOV = {SIM_SIZE*SIM_PIXEL:.0f}nm = {SIM_SIZE*SIM_PIXEL/1000:.2f}μm")
    print(f"  预计 σ=0.75 圆盘半径像素数: {expected_mpix:.1f}")
    src = PixelatedSource(
        grid_size=(SIM_SIZE, SIM_SIZE),
        optical_system=optics2,
        init_type=SourceInitializationType.CONVENTIONAL,
        constraints=SourceConstraintsConfig(
            energy_conservation=True, energy_target=1.0,
            non_negative=True, support_radius=0.95,
        )
    )
    print(f"  光源形状: {src.intensity.shape}")
    print(f"  总能量:   {src.intensity.sum():.6f}")
    print(f"  最大值:   {src.intensity.max():.6f}")
    print(f"  非零像素数: {np.sum(src.intensity > 1e-12)}")
    print(f"  等效 sigma: {src.compute_effective_sigma():.4f}")
    print(f"  rho_norm 范围: [{src.rho_norm.min():.4f}, {src.rho_norm.max():.4f}]")

    # ============== 与 generate_source 直接对比 ==============
    print(f"\n[与 generate_source 直接结果对比 (size={SIM_SIZE})]")
    fx_arr = np.fft.fftfreq(SIM_SIZE, SIM_PIXEL)
    fy_arr = np.fft.fftfreq(SIM_SIZE, SIM_PIXEL)
    fxs, fys = np.meshgrid(fx_arr, fy_arr)
    src_direct = generate_source(
        fxs, fys, IlluminationType.CONVENTIONAL,
        {'sigma_inner': 0.0, 'sigma_outer': 0.75},
        optics2.cutoff_frequency
    )
    print(f"  generate_source 直接结果: shape={src_direct.shape}, max={src_direct.max():.4f}, "
          f"非零={np.sum(src_direct > 1e-8)}")
    print(f"  PixelatedSource: 非零={np.sum(src.intensity > 1e-12)}")

    # ============== SMOImagingModel 调试 ==============
    print(f"\n[SMOImagingModel 调试 (mask={SIM_SIZE}×{SIM_SIZE})]")
    imaging = SMOImagingModel(optics2, (SIM_SIZE, SIM_SIZE), tcc_mode=TCCMode.FULL_TCC, socs_num_terms=3)
    imaging.update_source(src)
    im_source = imaging.get_source()
    print(f"  成像模型内光源: shape={im_source.shape}, sum={im_source.sum():.4f}, "
          f"max={im_source.max():.6f}, 非零={np.sum(im_source > 1e-10)}")

    # —— 成像测试：构造一个简单的多周期 contact hole mask（直接按像素画）——
    mask = np.ones((SIM_SIZE, SIM_SIZE), dtype=np.float64)
    # 生成 pitch=200nm 的方形网格，cd=80nm
    pitch_pix = int(200.0 / SIM_PIXEL)  # 200/40 = 5
    cd_pix = int(80.0 / SIM_PIXEL)       # 80/40 = 2
    yy_arr, xx_arr = np.mgrid[0:SIM_SIZE, 0:SIM_SIZE]
    for yc in range(pitch_pix//2, SIM_SIZE, pitch_pix):
        for xc in range(pitch_pix//2, SIM_SIZE, pitch_pix):
            r2 = (yy_arr - yc)**2 + (xx_arr - xc)**2
            mask[r2 <= (cd_pix/2)**2] = 0.0  # 画 hole
    print(f"  Mask shape: {mask.shape}, mask 1-pixels(暗场)={int(mask.sum())}, 0-pixels(hole)={int(mask.size-mask.sum())}")
    aerial = imaging.compute_aerial_image(mask)
    print(f"\n  Aerial image: shape={aerial.shape}")
    print(f"    range=[{aerial.min():.4f}, {aerial.max():.4f}], mean={aerial.mean():.4f}")
    print(f"    空间变异(标准差): {aerial.std():.4f}")

    # ============== 多工艺条件测试 ==============
    print("\n[多工艺条件测试]")
    conds = [
        {'defocus': -50.0, 'dose': 0.92, 'weight': 1.0},
        {'defocus':   0.0, 'dose': 1.00, 'weight': 1.5},
        {'defocus': +50.0, 'dose': 1.08, 'weight': 1.0},
    ]
    imaging.set_process_conditions(conds)
    imaging.update_source_all_conditions(src)

    threshold = 0.3
    k = 50.0
    # ★ 使用真正有差异的 target：把标称条件下的 wafer 空间平移，模拟 OPC 目标
    import scipy.ndimage as ndi
    cond0_im = imaging._process_imagers[1][0]
    target_aerial = np.clip(cond0_im.compute_aerial_image(mask) * 1.0, 0.0, None)
    # 平移 1 像素让它和 wafer 错位，保证有梯度
    target_aerial_shifted = ndi.shift(target_aerial, (1, 0))  # 水平平移 1 像素
    target = 1.0 / (1.0 + np.exp(-k * (target_aerial_shifted - threshold)))
    print(f"  Target (平移 1px 的 wafer): nonzeros(>0.5)={np.sum(target>0.5)}")
    baseline_wafer = 1.0 / (1.0 + np.exp(-k * (target_aerial - threshold)))
    print(f"  理想错位 MSE(baseline vs target) = {float(np.mean((baseline_wafer - target)**2)):.6f}")

    pvb_wafers = []
    for i, (imgr, df, dose, wt) in enumerate(imaging._process_imagers):
        a_i = np.clip(imgr.compute_aerial_image(mask) * dose, 0.0, None)
        wafer_i = 1.0 / (1.0 + np.exp(-k * (a_i - threshold)))
        pvb_wafers.append(wafer_i)
        mse = float(np.mean((wafer_i - target) ** 2))
        print(f"  cond[{i}] df={df:+.0f}nm dose={dose:.2f}: "
              f"aerial*dose=[{a_i.min():.3f},{a_i.max():.3f}], "
              f"MSE(wafer vs target)={mse:.6f}")
    if len(pvb_wafers) >= 2:
        stk = np.stack(pvb_wafers, axis=0)
        bw = np.max(stk, axis=0) - np.min(stk, axis=0)
        print(f"  PVB (hard L2) = {np.mean(bw**2):.6f}")

    # —— 测试：随机扰动光源，dLoss/dSource 是否为 0？——
    print("\n[小测试：光源扰动是否导致损失变化]")
    # 用标称 cond[1] 做单条件
    imgr0 = imaging._process_imagers[1][0]
    a0 = np.clip(imgr0.compute_aerial_image(mask) * 1.0, 0.0, None)
    w0 = 1.0 / (1.0 + np.exp(-k * (a0 - threshold)))
    L0 = float(np.mean((w0 - target) ** 2))
    print(f"  初始损失 L0 = {L0:.6f}")

    # 手动让光源随机平移（保持能量）
    import scipy.ndimage as ndi
    src_shift = ndi.shift(src.intensity, (2, 2))
    src_shift_sum = src_shift.sum() or 1.0
    src_shift = src_shift / src_shift_sum
    imgr0.update_source(src_shift)
    a1 = np.clip(imgr0.compute_aerial_image(mask) * 1.0, 0.0, None)
    w1 = 1.0 / (1.0 + np.exp(-k * (a1 - threshold)))
    L1 = float(np.mean((w1 - target) ** 2))
    print(f"  平移光源后 L1 = {L1:.6f}  (ΔL = {L1-L0:+.8f})")

    # 恢复光源
    imaging.update_source_all_conditions(src)
    a2 = np.clip(imgr0.compute_aerial_image(mask) * 1.0, 0.0, None)
    L_restored = float(np.mean((1.0/(1.0+np.exp(-k*(a2-threshold))) - target)**2))
    print(f"  恢复光源 L_restored = {L_restored:.6f}")

    # —— 测试 compute_source_gradient 的数值正确性 ——
    print("\n[数值梯度验证：Hopkins 解析 dL/dS vs 有限差分]")
    # 计算解析梯度
    # 链式：dL/dWafer * dWafer/dAerial * dAerial/dSource
    dLdW = 2.0 * (w0 - target) / w0.size
    dWdA = w0 * (1.0 - w0) * k
    dLdA = dLdW * dWdA
    dLdS_analytic = imgr0.compute_source_gradient(mask, dLdA)

    # 随机选 2 个非零光源点做有限差分验证
    nz_ys, nz_xs = np.where(src.intensity > 1e-10)
    if len(nz_ys) > 0:
        indices = np.random.choice(len(nz_ys), min(3, len(nz_ys)), replace=False)
        eps = 1e-6
        for idx in indices:
            py, px = int(nz_ys[idx]), int(nz_xs[idx])
            src_perturb = src.intensity.copy()
            src_perturb[py, px] += eps
            tot = src_perturb.sum()
            src_perturb = src_perturb / tot  # 保持能量
            imgr0.update_source(src_perturb)
            a_p = np.clip(imgr0.compute_aerial_image(mask) * 1.0, 0.0, None)
            w_p = 1.0 / (1.0 + np.exp(-k * (a_p - threshold)))
            Lp = float(np.mean((w_p - target) ** 2))
            # 反向扰动
            src_perturb2 = src.intensity.copy()
            if src_perturb2[py, px] > eps:
                src_perturb2[py, px] -= eps
            else:
                src_perturb2[py, px] = 0.0
            tot2 = src_perturb2.sum()
            if tot2 < 1e-15:
                continue
            src_perturb2 = src_perturb2 / tot2
            imgr0.update_source(src_perturb2)
            a_m = np.clip(imgr0.compute_aerial_image(mask) * 1.0, 0.0, None)
            w_m = 1.0 / (1.0 + np.exp(-k * (a_m - threshold)))
            Lm = float(np.mean((w_m - target) ** 2))
            dLdS_fd = (Lp - Lm) / (2 * eps)
            analytic_val = dLdS_analytic[py, px]
            print(f"  光源点 ({py},{px}): "
                  f"数值 dL/dS = {dLdS_fd:+.6e}, "
                  f"解析 dL/dS = {analytic_val:+.6e}, "
                  f"|误差| = {abs(dLdS_fd - analytic_val):.2e}")
        # 恢复
        imaging.update_source_all_conditions(src)


if __name__ == '__main__':
    debug()
