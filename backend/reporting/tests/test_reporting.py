# -*- coding: utf-8 -*-
"""
Tapeout 签核报告模块测试

使用模拟数据测试完整的报告生成流程。
"""

import sys
import tempfile
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reporting import (
    TapeoutSignoffAPI,
    quick_signoff_report,
    ReportDataCollector,
    generate_html_report,
    generate_pdf_report,
)


def create_test_images(size=200):
    """创建测试用的模拟图像"""
    # 创建目标图像：一个简单的矩形
    target = np.zeros((size, size), dtype=np.float64)
    target[60:140, 50:150] = 1.0

    # 创建初始掩模：比目标稍大一点
    mask_initial = np.zeros((size, size), dtype=np.float64)
    mask_initial[55:145, 45:155] = 1.0

    # 创建最终掩模：经过 OPC 修正的（模拟）
    mask_final = np.zeros((size, size), dtype=np.float64)
    mask_final[58:142, 48:152] = 1.0
    # 添加一些 SRAF 特征
    mask_final[25:35, 70:130] = 1.0
    mask_final[165:175, 70:130] = 1.0

    # 模拟晶圆图像：略微模糊
    from scipy.ndimage import gaussian_filter
    wafer_initial = gaussian_filter(mask_initial, sigma=2.0)
    wafer_final = gaussian_filter(mask_final, sigma=1.8)

    # 模拟空间像
    aerial_initial = gaussian_filter(mask_initial, sigma=1.5)
    aerial_final = gaussian_filter(mask_final, sigma=1.3)

    return {
        'target': target,
        'mask_initial': mask_initial,
        'mask_final': mask_final,
        'wafer_initial': wafer_initial,
        'wafer_final': wafer_final,
        'aerial_initial': aerial_initial,
        'aerial_final': aerial_final,
    }


def test_data_collector_with_images():
    """测试数据收集器（使用图像）"""
    print("=" * 60)
    print("测试 1: 数据收集器（图像计算模式）")
    print("=" * 60)

    images = create_test_images()

    collector = ReportDataCollector(pixel_size=1.0, threshold=0.5)

    collector.set_basic_info(
        project_name="Test_Project",
        design_name="demo_design.gds",
        technology_node="28nm",
        ret_flow="OPC + SRAF",
        title="Tapeout 签核测试报告",
    )

    # 收集初始阶段
    collector.collect_initial(
        initial_mask=images['mask_initial'],
        target=images['target'],
        wafer_initial=images['wafer_initial'],
        aerial_initial=images['aerial_initial'],
    )

    # 收集最终阶段
    collector.collect_final(
        final_mask=images['mask_final'],
        target=images['target'],
        wafer_final=images['wafer_final'],
        aerial_final=images['aerial_final'],
    )

    print(f"✓ 初始/最终阶段指标收集成功")
    print(f"  初始平均 EPE: {collector._report.initial_metrics.epe.epe_mean_nm:.3f} nm")
    print(f"  最终平均 EPE: {collector._report.final_metrics.epe.epe_mean_nm:.3f} nm")
    print(f"  初始 TV: {collector._report.initial_metrics.mask_complexity.total_variation:.1f}")
    print(f"  最终 TV: {collector._report.final_metrics.mask_complexity.total_variation:.1f}")
    print()

    return collector


def test_process_window():
    """测试工艺窗口数据收集"""
    print("=" * 60)
    print("测试 2: 工艺窗口数据收集")
    print("=" * 60)

    collector = ReportDataCollector(pixel_size=1.0)

    # 使用字典格式的工艺窗口数据
    pw_data = {
        'pw_area': 4500.5,
        'pw_ratio': 0.68,
        'n_passing': 680,
        'n_total': 1000,
        'center_focus_nm': 50.0,
        'center_dose': 1.02,
        'best_focus_nm': 45.0,
        'best_dose': 1.015,
        'best_cd_error_nm': 0.5,
        'depth_of_focus_nm': 350.0,
        'exposure_latitude_pct': 12.5,
        'focus_min_nm': -100.0,
        'focus_max_nm': 250.0,
        'dose_min': 0.94,
        'dose_max': 1.10,
        'ellipse_area': 4200.0,
        'rect_area': 5250.0,
    }

    collector.collect_process_window(pw_data)

    pw = collector._report.process_window
    print(f"✓ 工艺窗口数据收集成功")
    print(f"  PW 面积: {pw.pw_area:.1f} nm·dose")
    print(f"  PW 占比: {pw.pw_ratio*100:.1f}%")
    print(f"  焦深: {pw.depth_of_focus_nm:.1f} nm")
    print(f"  曝光宽容度: {pw.exposure_latitude_pct:.2f}%")
    print()

    return collector


def test_mrc():
    """测试 MRC 数据收集"""
    print("=" * 60)
    print("测试 3: MRC 违规数据收集")
    print("=" * 60)

    collector = ReportDataCollector(pixel_size=1.0)

    mrc_data = {
        'total_violations': 23,
        'fatal_count': 0,
        'error_count': 3,
        'warning_count': 12,
        'info_count': 8,
        'passed': True,
        'violations_by_rule': {
            'min_width': 5,
            'min_space': 4,
            'min_gap': 3,
            'corner_rounding': 6,
            'density': 2,
            'other': 3,
        },
        'top_violations': [
            {'severity': 'error', 'message': '最小线宽违规', 'rule': 'min_width',
             'measurement_nm': 23.5, 'threshold_nm': 25.0},
            {'severity': 'error', 'message': '最小间距违规', 'rule': 'min_space',
             'measurement_nm': 22.1, 'threshold_nm': 25.0},
            {'severity': 'warning', 'message': '角部圆角过大', 'rule': 'corner_rounding',
             'measurement_nm': 8.5, 'threshold_nm': 5.0},
        ],
    }

    collector.collect_mrc(mrc_data)

    mrc = collector._report.mrc_violations
    print(f"✓ MRC 数据收集成功")
    print(f"  总违规数: {mrc.total_violations}")
    print(f"  致命: {mrc.fatal_count}, 错误: {mrc.error_count}, 警告: {mrc.warning_count}")
    print(f"  通过: {mrc.passed}")
    print(f"  规则分类数: {len(mrc.violations_by_rule)}")
    print()

    return collector


def test_metrology():
    """测试计量一致性数据收集"""
    print("=" * 60)
    print("测试 4: 计量一致性数据收集")
    print("=" * 60)

    collector = ReportDataCollector(pixel_size=1.0)

    metrology_data = {
        'm2t_mean_nm': 0.25,
        'm2t_pct': 0.25,
        'uniformity_3sigma_pct': 3.5,
        'uniformity_range_pct': 5.2,
        'linearity_r_squared': 0.987,
        'linearity_slope': 0.995,
        'grr_pct': 8.5,
        'grr_ndc': 16.5,
        'cp': 1.45,
        'cpk': 1.28,
        'pass_rate_pct': 96.5,
        'n_measurements': 200,
    }

    collector.collect_metrology(metrology_data)

    met = collector._report.metrology
    print(f"✓ 计量一致性数据收集成功")
    print(f"  M2T: {met.m2t_mean_nm:+.3f} nm ({met.m2t_pct:+.2f}%)")
    print(f"  3σ 均匀性: {met.uniformity_3sigma_pct:.2f}%")
    print(f"  Cpk: {met.cpk:.3f}")
    print(f"  合格率: {met.pass_rate_pct:.1f}%")
    print()

    return collector


def test_full_report():
    """测试完整报告生成流程"""
    print("=" * 60)
    print("测试 5: 完整报告生成")
    print("=" * 60)

    images = create_test_images()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"

        # 使用高层 API
        api = TapeoutSignoffAPI(pixel_size=1.0, threshold=0.5)

        api.set_basic_info(
            project_name="Full_Test",
            design_name="full_test.gds",
            technology_node="28nm",
            ret_flow="OPC + SRAF",
            title="完整测试签核报告",
        )

        # 收集图像指标
        api.collector.collect_initial(
            initial_mask=images['mask_initial'],
            target=images['target'],
            wafer_initial=images['wafer_initial'],
            aerial_initial=images['aerial_initial'],
        )
        api.collector.collect_final(
            final_mask=images['mask_final'],
            target=images['target'],
            wafer_final=images['wafer_final'],
            aerial_final=images['aerial_final'],
        )

        # 工艺窗口
        pw_data = {
            'pw_area': 5000.0,
            'pw_ratio': 0.75,
            'n_passing': 750,
            'n_total': 1000,
            'center_focus_nm': 0.0,
            'center_dose': 1.0,
            'best_focus_nm': 10.0,
            'best_dose': 1.0,
            'best_cd_error_nm': 0.3,
            'depth_of_focus_nm': 400.0,
            'exposure_latitude_pct': 15.0,
            'focus_min_nm': -150.0,
            'focus_max_nm': 250.0,
            'dose_min': 0.92,
            'dose_max': 1.08,
            'ellipse_area': 4700.0,
            'rect_area': 5200.0,
        }
        api.collect_process_window(pw_data)

        # MRC
        mrc_data = {
            'total_violations': 5,
            'fatal_count': 0,
            'error_count': 0,
            'warning_count': 3,
            'info_count': 2,
            'passed': True,
            'violations_by_rule': {'min_width': 2, 'min_space': 1, 'other': 2},
            'top_violations': [
                {'severity': 'warning', 'message': '线宽接近下限', 'rule': 'min_width',
                 'measurement_nm': 26.0, 'threshold_nm': 25.0},
            ],
        }
        api.collect_mrc(mrc_data)

        # 计量一致性
        met_data = {
            'm2t_mean_nm': 0.5,
            'm2t_pct': 0.5,
            'uniformity_3sigma_pct': 3.0,
            'uniformity_range_pct': 4.5,
            'linearity_r_squared': 0.99,
            'linearity_slope': 1.0,
            'grr_pct': 7.0,
            'grr_ndc': 20.0,
            'cp': 1.67,
            'cpk': 1.5,
            'pass_rate_pct': 98.5,
            'n_measurements': 150,
        }
        api.collect_metrology(met_data)

        # 默认参数表
        api.add_default_parameter_tables()

        # 构建报告
        report = api.build_report()

        print(f"✓ 报告构建成功")
        print(f"  报告 ID: {report.report_id}")
        print(f"  标题: {report.title}")
        print(f"  初始 EPE: {report.initial_metrics.epe.epe_mean_nm:.3f} nm")
        print(f"  最终 EPE: {report.final_metrics.epe.epe_mean_nm:.3f} nm")
        print(f"  参数表数量: {len(report.parameter_tables)}")
        print()

        # 生成 HTML
        html_path = output_dir / "report.html"
        api.generate_html(html_path, embed_images=False)
        print(f"✓ HTML 报告生成: {html_path.name} ({html_path.stat().st_size:,} 字节)")

        # 生成 PDF
        pdf_path = output_dir / "report.pdf"
        api.generate_pdf(pdf_path)
        print(f"✓ PDF 报告生成: {pdf_path.name} ({pdf_path.stat().st_size:,} 字节)")

        # 生成 JSON
        json_path = output_dir / "report.json"
        report.save_json(json_path)
        print(f"✓ JSON 数据生成: {json_path.name} ({json_path.stat().st_size:,} 字节)")

        # 生成图表
        try:
            fig_paths = api.generate_figures(
                output_dir=output_dir / "figures",
                mask_initial=images['mask_initial'],
                mask_final=images['mask_final'],
                wafer_initial=images['wafer_initial'],
                wafer_final=images['wafer_final'],
                target=images['target'],
            )
            n_figs = len([p for p in fig_paths.values() if p and Path(p).exists()])
            print(f"✓ 图表生成: {n_figs} 张")
        except Exception as e:
            print(f"⚠ 图表生成跳过: {e}")

        print()

        return report


def test_json_serialization():
    """测试 JSON 序列化"""
    print("=" * 60)
    print("测试 6: JSON 序列化/反序列化")
    print("=" * 60)

    images = create_test_images()
    collector = ReportDataCollector(pixel_size=1.0)
    collector.collect_initial(
        initial_mask=images['mask_initial'],
        target=images['target'],
    )
    collector.collect_final(
        final_mask=images['mask_final'],
        target=images['target'],
    )
    report = collector.build_report()

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "report.json"
        saved_path = report.save_json(json_path)

        print(f"✓ JSON 保存成功: {saved_path}")
        print(f"  文件大小: {json_path.stat().st_size:,} 字节")

        # 验证可以加载
        with open(json_path, 'r') as f:
            data = json.load(f)

        print(f"  report_id: {data['report_id']}")
        print(f"  title: {data['title']}")
        print(f"  initial EPE mean: {data['initial_metrics']['epe']['epe_mean_nm']:.3f} nm")
        print()

    return True


def test_summary():
    """测试报告摘要生成"""
    print("=" * 60)
    print("测试 7: 报告摘要生成")
    print("=" * 60)

    images = create_test_images()
    collector = ReportDataCollector(pixel_size=1.0)
    collector.collect_initial(
        initial_mask=images['mask_initial'],
        target=images['target'],
    )
    collector.collect_final(
        final_mask=images['mask_final'],
        target=images['target'],
    )

    # 添加一些其他数据
    pw = {'pw_ratio': 0.7, 'depth_of_focus_nm': 300.0, 'exposure_latitude_pct': 10.0}
    collector.collect_process_window(pw)

    mrc = {'passed': True, 'total_violations': 5}
    collector.collect_mrc(mrc)

    met = {'cpk': 1.33, 'pass_rate_pct': 97.5}
    collector.collect_metrology(met)

    report = collector.build_report()
    summary = report.generate_summary()

    print(f"✓ 摘要生成成功")
    print(f"  长度: {len(summary)} 字符")
    print("  内容预览:")
    for line in summary.split('\n')[:10]:
        print(f"    {line}")
    print()

    return summary


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Tapeout 签核报告模块 - 功能测试")
    print("=" * 60 + "\n")

    try:
        # 测试 1: 数据收集器
        test_data_collector_with_images()

        # 测试 2: 工艺窗口
        test_process_window()

        # 测试 3: MRC
        test_mrc()

        # 测试 4: 计量一致性
        test_metrology()

        # 测试 5: 完整报告
        test_full_report()

        # 测试 6: JSON 序列化
        test_json_serialization()

        # 测试 7: 摘要
        test_summary()

        print("=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
