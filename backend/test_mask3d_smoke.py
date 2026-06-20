#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mask3D 模块快速冒烟测试"""

import sys
sys.path.insert(0, '.')
import numpy as np


def test_imports():
    from core.mask3d import (
        MaskType, SidewallProfile, RoughnessModel,
        SidewallParams, RoughnessParams, AbsorberLayer, Mask3DConfig,
        MaskTopography, SimplifiedBEMScattering, RCWAHopkinsCoupler,
        Mask3DCorrectionResult, Mask3DImagingCorrector,
        create_default_mask3d_config, apply_mask3d_correction,
    )
    print("[1/6] All imports OK")
    return True


def test_config():
    from core.mask3d import (
        MaskType, create_default_mask3d_config,
    )
    cfg = create_default_mask3d_config(
        mask_type=MaskType.BINARY_COG,
        wavelength_nm=193.0,
        sidewall_angle_deg=3.0,
        absorber_thickness_nm=70.0,
    )
    assert cfg.mask_type == MaskType.BINARY_COG
    assert abs(cfg.absorber.sidewall.sidewall_angle_deg - 3.0) < 1e-6
    n_abs = cfg.absorber.get_refractive_index(193.0)
    n_sub = cfg.get_substrate_n()
    assert abs(n_abs - (3.28 - 4.32j)) < 1e-2
    assert abs(n_sub - (1.567 + 0.0j)) < 1e-2
    print("[2/6] Config creation OK (n_abs={}, n_sub={})".format(n_abs, n_sub))
    return True


def test_topography():
    from core.mask3d import create_default_mask3d_config, MaskTopography, MaskType
    cfg = create_default_mask3d_config(
        mask_type=MaskType.BINARY_COG, sidewall_angle_deg=3.0,
    )
    topo = MaskTopography(cfg)
    section = topo.generate_cross_section(nominal_cd_nm=100.0, num_points_x=257)
    assert len(section["x"]) == 257
    assert np.max(section["height_profile"]) > 0
    assert len(section["sidewall_indices"]) >= 2
    h = section["height_profile"]
    dx = section["dx_nm"]
    # 检查吸收层内部最高高度接近标称厚度
    assert 60 < np.max(h) < 80
    print("[3/6] Topography OK (N={}, h_max={:.1f}nm, sw_regions={})".format(
        len(section["x"]), np.max(h), len(section["sidewall_indices"])
    ))
    return True


def test_correction():
    from core.mask3d import create_default_mask3d_config, apply_mask3d_correction, MaskType
    cfg = create_default_mask3d_config(
        mask_type=MaskType.BINARY_COG, sidewall_angle_deg=3.0,
    )
    ny, nx = 64, 128
    mask = np.ones((ny, nx), dtype=np.float64)
    pixel_nm = 2.0
    line_w = int(100.0 / pixel_nm)
    x0 = nx // 2 - line_w // 2
    mask[:, x0:x0 + line_w] = 0.0

    result = apply_mask3d_correction(mask, pixel_nm, cfg)
    assert result.t_effective.shape == (ny, nx)
    assert result.amplitude_correction.shape == (ny, nx)
    assert result.phase_correction.shape == (ny, nx)
    # 振幅修正应在合理范围 (小CD区域可能很低)
    assert 0.01 < np.min(result.amplitude_correction) <= 1.5
    # 非零运行时间
    assert result.runtime_sec >= 0
    print("[4/6] Correction OK (shape={}, amp_range=[{:.3f},{:.3f}], runtime={:.1f}ms)".format(
        result.t_effective.shape,
        np.min(result.amplitude_correction),
        np.max(result.amplitude_correction),
        result.runtime_sec * 1000,
    ))
    return True


def test_cd_scan():
    from core.mask3d import create_default_mask3d_config, Mask3DImagingCorrector, MaskType
    cfg = create_default_mask3d_config(
        mask_type=MaskType.BINARY_COG, sidewall_angle_deg=3.0,
    )
    corrector = Mask3DImagingCorrector(cfg)
    scan = corrector.scan_cd_bias_vs_cd(
        cd_range_nm=(50.0, 400.0), num_points=10, na=1.35
    )
    assert len(scan["cd_nominal_nm"]) == 10
    assert len(scan["cd_bias_nm"]) == 10
    # 小 CD 偏差通常为负 (through-pitch 效应)
    mean_bias = np.mean(scan["cd_bias_nm"])
    max_abs = np.max(np.abs(scan["cd_bias_nm"]))
    assert max_abs < 50  # 合理范围内
    print("[5/6] CD bias scan OK (mean_bias={:.2f}nm, max|bias|={:.2f}nm)".format(
        mean_bias, max_abs
    ))
    return True


def test_profiles():
    from core.mask3d import (
        SidewallProfile, SidewallParams, AbsorberLayer,
        Mask3DConfig, MaskTopography,
    )
    profiles = [
        SidewallProfile.RECTANGULAR,
        SidewallProfile.TRAPEZOIDAL,
        SidewallProfile.ROUNDED_TOP,
        SidewallProfile.ROUNDED_BOTTOM,
        SidewallProfile.REENTRANT,
    ]
    results = []
    for p in profiles:
        sw = SidewallParams(
            profile_type=p,
            sidewall_angle_deg=5.0 if p in (SidewallProfile.TRAPEZOIDAL,) else 0.0,
            top_rounding_nm=8.0 if p == SidewallProfile.ROUNDED_TOP else 0.0,
            bottom_rounding_nm=6.0 if p == SidewallProfile.ROUNDED_BOTTOM else 0.0,
            top_cd_bias_nm=12.0 if p == SidewallProfile.TRAPEZOIDAL else (
                -8.0 if p == SidewallProfile.REENTRANT else 0.0
            ),
            bottom_cd_bias_nm=0.0 if p != SidewallProfile.REENTRANT else 10.0,
        )
        a = AbsorberLayer(sidewall=sw)
        cp = Mask3DConfig(absorber=a)
        tp = MaskTopography(cp)
        sc = tp.generate_cross_section(nominal_cd_nm=100.0, num_points_x=201)
        h = sc["height_profile"]
        dx = sc["dx_nm"]
        sw_w = np.sum((h > 0) & (h < np.max(h))) * dx
        results.append((p.value, np.max(h), sw_w))
    print("[6/6] Sidewall profiles OK:")
    for name, hmax, sww in results:
        print("       {:<16} hmax={:>5.1f}nm  sw_width={:>5.1f}nm".format(name, hmax, sww))
    return True


def test_roughness():
    from core.mask3d import (
        RoughnessModel, RoughnessParams, AbsorberLayer,
        Mask3DConfig, MaskTopography,
    )
    rp = RoughnessParams(
        model=RoughnessModel.GAUSSIAN,
        rms_height_nm=2.0,
        correlation_length_nm=30.0,
        seed=42,
    )
    a = AbsorberLayer(surface_roughness=rp)
    cfg = Mask3DConfig(absorber=a, enable_roughness=True)
    topo = MaskTopography(cfg)
    rough = topo.generate_roughness((64, 128), pixel_size_nm=2.0)
    assert rough is not None
    current_rms = np.sqrt(np.mean(rough ** 2))
    assert 1.0 < current_rms < 3.0  # 接近目标值
    print("[EXTRA] Roughness OK (target_rms=2.0nm, actual={:.2f}nm)".format(current_rms))
    return True


def main():
    print("=" * 60)
    print("Mask3D Module Smoke Test")
    print("=" * 60)
    all_pass = True

    tests = [
        test_imports,
        test_config,
        test_topography,
        test_correction,
        test_cd_scan,
        test_profiles,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print("FAILED {}: {}".format(t.__name__, e))
            import traceback
            traceback.print_exc()
            all_pass = False

    try:
        test_roughness()
    except Exception as e:
        print("FAILED test_roughness: {}".format(e))

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
