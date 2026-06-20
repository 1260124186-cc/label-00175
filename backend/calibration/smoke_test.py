# -*- coding: utf-8 -*-
"""
Fab 模型标定模块：冒烟测试
Smoke test for the calibration pipeline.

使用方式（在 backend/ 目录下）:

    PYTHONPATH=. python3 calibration/smoke_test.py

测试内容：
1. 加载合成 CD-SEM 数据
2. 运行 LMFIT 反演
3. 生成报告与配置片段
4. 输出与真值对比
"""
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    from calibration.schemas import (
        CalibrationConfig, InversionMethod, CalibrationParameterSet,
    )
    from calibration.data_loader import (
        load_cd_sem_data, validate_dataset, split_dataset,
    )
    from calibration.pipeline import CalibrationPipeline, run_calibration_pipeline

    print("=" * 70)
    print("  Fab 模型标定模块 - 冒烟测试")
    print("=" * 70)

    sample_data = BACKEND_DIR / 'calibration' / 'sample_data' / 'sample_cd_sem_data.csv'
    ref_config = BACKEND_DIR / 'config' / 'default_config.yaml'
    output_dir = BACKEND_DIR / 'calibration_results' / 'smoke_test'

    # --------------------------------------------------------------
    # 1. 数据加载测试
    # --------------------------------------------------------------
    print("\n[1/6] 加载 CD-SEM 合成数据...")
    t0 = time.time()
    dataset = load_cd_sem_data(sample_data)
    print(f"      加载完成：{len(dataset)} 个量测点，耗时 {time.time()-t0:.2f}s")
    print(f"      Focus 范围: {dataset.focus_range()} nm")
    print(f"      Dose 范围:  {dataset.dose_range()}")
    print(f"      图形类型:   {sorted(set(p.value for p in dataset.pattern_types()))}")

    # --------------------------------------------------------------
    # 2. 校验测试
    # --------------------------------------------------------------
    print("\n[2/6] 数据校验与清洗...")
    cleaned, report = validate_dataset(dataset)
    print(f"      原始: {report['original_count']} → 清洗后: {report['final_count']} "
          f"（移除 {report['removed_count']}）")
    if report.get('warnings'):
        for w in report['warnings']:
            print(f"      ⚠️  {w}")

    # --------------------------------------------------------------
    # 3. 构建配置
    # --------------------------------------------------------------
    print("\n[3/6] 构建标定配置...")
    cfg = CalibrationConfig(
        method=InversionMethod.LMFIT,
        output_dir=str(output_dir),
        reference_config_path=str(ref_config),
        use_measurement_weights=True,
        forward_model_complexity='standard',
        generate_plots=True,
        update_config=True,
        nlls_max_iter=10000,
        nlls_method='trf',
    )
    # 固定波长（不参与反演）
    cfg.parameters.wavelength_effective.vary = False
    varying = cfg.parameters.get_varying_parameters()
    print(f"      参与反演参数: {len(varying)} 个")
    for p in varying:
        print(f"        - {p.name}: init={p.initial_value}, "
              f"bounds=[{p.lower_bound}, {p.upper_bound}]")

    # --------------------------------------------------------------
    # 4. 执行流水线
    # --------------------------------------------------------------
    print("\n[4/6] 执行标定流水线 (LMFIT)...")
    pipeline = CalibrationPipeline(cfg)
    t0 = time.time()
    result = pipeline.run(cleaned)
    total = time.time() - t0
    print(f"      总耗时 {total:.2f} s")
    print(f"      反演成功: {result.inversion_result.success}")
    print(f"      消息:     {result.inversion_result.message}")

    # --------------------------------------------------------------
    # 5. 反演结果 vs 真值
    # --------------------------------------------------------------
    TRUE_PARAMS = {
        'resist_threshold': 0.285,
        'diffusion_length': 12.5,
        'na_effective': 1.335,
        'sigma_effective': 0.78,
        'wavelength_effective': 193.2,
        'dose_to_clear': 0.48,
        'resist_contrast': 3.2,
    }
    print("\n[5/6] 参数反演结果 vs 真值:")
    print(f"      {'参数名':<24} {'反演值':>12} {'真值':>12} {'偏差':>10} {'±1σ':>10}")
    print("      " + "-" * 70)
    all_ok = True
    for name, true_val in TRUE_PARAMS.items():
        fitted = result.inversion_result.calibrated_values.get(name, float('nan'))
        unc = result.inversion_result.uncertainties.get(name, 0.0)
        diff = fitted - true_val
        ok_mark = "✓" if abs(diff) < max(0.05 * abs(true_val), 3 * unc) else "✗"
        if ok_mark != "✓":
            all_ok = False
        print(f"      {name:<24} {fitted:>12.6f} {true_val:>12.6f} "
              f"{diff:>+10.4f} {unc:>10.4f} {ok_mark}")

    # --------------------------------------------------------------
    # 6. 拟合质量
    # --------------------------------------------------------------
    print("\n[6/6] 拟合质量指标:")
    for k, v in result.metrics.items():
        print(f"      {k:<22}: {v:.6f}")

    # 配置输出
    print(f"\n✅ 标定结果输出目录: {output_dir.resolve()}")
    for k, v in pipeline.output_paths.items():
        print(f"     - {k}: {v}")

    print("\n" + "=" * 70)
    if all_ok:
        print("🎉  冒烟测试通过！所有参数与真值偏差在允许范围内。")
    else:
        print("⚠️  部分参数偏差稍大，但整体流程正常运行。")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
