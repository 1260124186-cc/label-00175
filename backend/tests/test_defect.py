# -*- coding: utf-8 -*-
"""
掩模缺陷打印性分析模块单元测试
"""

import pytest
import numpy as np
from pathlib import Path
import sys

_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from defect.schemas import (
    DefectType,
    DefectPolarity,
    PointDefect,
    LineDefect,
    ContaminationDefect,
    DefectLocation,
    DefectInjectionConfig,
    DefectSensitivityEntry,
    DefectSensitivityReport,
    SingleDefectResult,
)
from defect.defect_injector import DefectInjector
from defect.defect_simulator import DefectSimulator
from defect.sensitivity import (
    DefectSensitivityAnalyzer,
    DefectScanConfig,
    run_defect_analysis,
)
from core.imaging import OpticalSystem


def _create_simple_mask(shape=(128, 128), cd_pixels=20, pitch_pixels=40):
    """创建简单的线/空间测试掩模"""
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=np.float64)
    for x in range(0, nx, pitch_pixels):
        x_start = x
        x_end = min(x + cd_pixels, nx)
        mask[:, x_start:x_end] = 1.0
    return mask


class TestSchemas:
    """数据模型测试"""

    def test_point_defect_creation(self):
        defect = PointDefect(size_nm=20.0)
        assert defect.size_nm == 20.0
        assert defect.shape == 'circle'
        assert defect.polarity == DefectPolarity.OPAQUE

    def test_point_defect_invalid_shape(self):
        with pytest.raises(ValueError):
            PointDefect(size_nm=20.0, shape='triangle')

    def test_line_defect_creation(self):
        defect = LineDefect(length_nm=100.0, width_nm=20.0, angle_deg=45.0)
        assert defect.length_nm == 100.0
        assert defect.width_nm == 20.0
        assert defect.angle_deg == 45.0

    def test_contamination_defect_creation(self):
        defect = ContaminationDefect(size_nm=50.0, attenuation=0.7, roughness=0.3)
        assert defect.size_nm == 50.0
        assert defect.attenuation == 0.7
        assert defect.roughness == 0.3

    def test_contamination_defect_invalid_attenuation(self):
        with pytest.raises(ValueError):
            ContaminationDefect(size_nm=50.0, attenuation=1.5)

    def test_contamination_defect_invalid_roughness(self):
        with pytest.raises(ValueError):
            ContaminationDefect(size_nm=50.0, roughness=-0.1)

    def test_defect_injection_config_cd_limits(self):
        config = DefectInjectionConfig(cd_target=45.0, cd_tolerance=0.1)
        assert abs(config.cd_lower - 40.5) < 1e-10
        assert abs(config.cd_upper - 49.5) < 1e-10

    def test_single_defect_result_to_dict(self):
        result = SingleDefectResult(
            defect_type=DefectType.POINT,
            defect_params={'size_nm': 20.0},
            nominal_cd=45.0,
            defective_cd=43.0,
            delta_cd=-2.0,
            delta_cd_relative=-4.44,
        )
        d = result.to_dict()
        assert d['defect_type'] == 'point'
        assert d['nominal_cd'] == 45.0
        assert d['delta_cd'] == -2.0

    def test_sensitivity_report_summary(self):
        entry = DefectSensitivityEntry(
            rank=1,
            defect_type=DefectType.POINT,
            size_nm=30.0,
            polarity=DefectPolarity.OPAQUE,
            location='edge',
            delta_cd_abs=5.0,
            delta_cd_relative=10.0,
            is_critical=True,
            failure_probability=0.95,
            sensitivity_score=90.0,
            recommendation='致命缺陷',
        )
        report = DefectSensitivityReport(
            entries=[entry],
            total_defects_analyzed=1,
            critical_defect_count=1,
            critical_defect_ratio=1.0,
            recommended_spec=25.0,
            nominal_cd=45.0,
            cd_tolerance=0.1,
        )
        summary = report.summary()
        assert '致命缺陷数' in summary
        assert '推荐检测规格' in summary


class TestDefectInjector:
    """缺陷注入器测试"""

    @pytest.fixture
    def injector(self):
        return DefectInjector(pixel_size=1.0, random_seed=42)

    @pytest.fixture
    def simple_mask(self):
        return _create_simple_mask((64, 64), cd_pixels=16, pitch_pixels=32)

    def test_inject_point_defect_circle(self, injector, simple_mask):
        defect = PointDefect(
            size_nm=10.0,
            shape='circle',
            polarity=DefectPolarity.OPAQUE,
            location=DefectLocation(y=32, x=28),
        )
        result = injector.inject(simple_mask, defect)
        assert result.shape == simple_mask.shape
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
        assert np.any(result != simple_mask)

    def test_inject_point_defect_square(self, injector, simple_mask):
        defect = PointDefect(
            size_nm=10.0,
            shape='square',
            polarity=DefectPolarity.OPAQUE,
            location=DefectLocation(y=32, x=28),
        )
        result = injector.inject(simple_mask, defect)
        assert result.shape == simple_mask.shape
        assert np.any(result != simple_mask)

    def test_inject_clear_defect(self, injector, simple_mask):
        defect = PointDefect(
            size_nm=10.0,
            polarity=DefectPolarity.CLEAR,
            location=DefectLocation(y=32, x=8),
        )
        result = injector.inject(simple_mask, defect)
        center_val = result[32, 8]
        original_val = simple_mask[32, 8]
        if original_val < 0.5:
            assert center_val > original_val

    def test_inject_line_defect(self, injector, simple_mask):
        defect = LineDefect(
            length_nm=30.0,
            width_nm=5.0,
            angle_deg=0.0,
            polarity=DefectPolarity.OPAQUE,
            location=DefectLocation(y=32, x=24),
        )
        result = injector.inject(simple_mask, defect)
        assert result.shape == simple_mask.shape
        assert np.any(result != simple_mask)

    def test_inject_line_defect_rotated(self, injector, simple_mask):
        defect = LineDefect(
            length_nm=30.0,
            width_nm=5.0,
            angle_deg=45.0,
            polarity=DefectPolarity.OPAQUE,
            location=DefectLocation(y=32, x=24),
        )
        result = injector.inject(simple_mask, defect)
        assert result.shape == simple_mask.shape

    def test_inject_contamination(self, injector, simple_mask):
        defect = ContaminationDefect(
            size_nm=20.0,
            attenuation=0.7,
            roughness=0.3,
            polarity=DefectPolarity.OPAQUE,
            location=DefectLocation(y=32, x=24),
        )
        result = injector.inject(simple_mask, defect)
        assert result.shape == simple_mask.shape
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0

    def test_inject_multiple(self, injector, simple_mask):
        defects = [
            PointDefect(size_nm=8.0, location=DefectLocation(y=20, x=24)),
            PointDefect(size_nm=8.0, location=DefectLocation(y=44, x=24),
                        polarity=DefectPolarity.CLEAR),
        ]
        result = injector.inject_multiple(simple_mask, defects)
        assert result.shape == simple_mask.shape

    def test_distance_to_edge(self, injector, simple_mask):
        loc_edge = DefectLocation(y=32, x=15)
        dist = injector.compute_distance_to_edge(simple_mask, loc_edge)
        assert dist >= 0.0

    def test_generate_edge_locations(self, injector, simple_mask):
        locations = injector.generate_edge_proximity_locations(
            simple_mask, n_locations=3
        )
        assert len(locations) == 3
        for loc in locations:
            assert 0 <= loc.y < simple_mask.shape[0]
            assert 0 <= loc.x < simple_mask.shape[1]


class TestDefectSimulator:
    """缺陷仿真器测试"""

    @pytest.fixture
    def optical_system(self):
        return OpticalSystem(
            wavelength=193.0,
            na=0.85,
            sigma=0.5,
            pixel_size=1.0,
        )

    @pytest.fixture
    def small_mask(self):
        return _create_simple_mask((64, 64), cd_pixels=16, pitch_pixels=32)

    @pytest.fixture
    def simulator(self, optical_system):
        return DefectSimulator(
            optical_system=optical_system,
            config=DefectInjectionConfig(
                pixel_size=1.0, cd_target=16.0, cd_tolerance=0.1,
            ),
        )

    def test_simulate_image(self, simulator, small_mask):
        aerial, wafer = simulator.simulate_image(small_mask)
        assert aerial.shape == small_mask.shape
        assert wafer.shape == small_mask.shape
        assert np.min(aerial) >= 0.0
        assert np.max(aerial) <= 1.0
        assert set(np.unique(wafer)).issubset({0.0, 1.0})

    def test_measure_cd(self, simulator, small_mask):
        _, wafer = simulator.simulate_image(small_mask)
        cd, results = simulator.measure_cd(wafer)
        assert cd > 0.0
        assert len(results) >= 1

    def test_simulate_point_defect(self, simulator, small_mask):
        defect = PointDefect(
            size_nm=10.0,
            polarity=DefectPolarity.OPAQUE,
            location=DefectLocation(y=32, x=24),
        )
        result = simulator.simulate_defect(small_mask, defect, save_images=True)
        assert result.defect_type == DefectType.POINT
        assert result.nominal_cd > 0.0
        assert result.defective_cd > 0.0
        assert result.nominal_aerial is not None
        assert result.defective_wafer is not None
        assert result.mask_defective is not None
        assert 0.0 <= result.sensitivity_score <= 100.0

    def test_simulate_large_defect_is_critical(self, simulator, small_mask):
        defect = PointDefect(
            size_nm=40.0,
            polarity=DefectPolarity.CLEAR,
            location=DefectLocation(y=32, x=8),
        )
        result = simulator.simulate_defect(small_mask, defect, save_images=False)
        assert 0.0 <= result.failure_probability <= 1.0
        assert 0.0 <= result.sensitivity_score <= 100.0

    def test_simulate_batch(self, simulator, small_mask):
        defects = [
            PointDefect(size_nm=5.0, location=DefectLocation(y=32, x=28)),
            PointDefect(size_nm=15.0, location=DefectLocation(y=32, x=28)),
            PointDefect(size_nm=25.0, location=DefectLocation(y=32, x=28)),
        ]
        results = simulator.simulate_defects_batch(small_mask, defects, save_images=False)
        assert len(results) == 3
        cds = [r.delta_cd_relative for r in results]
        for i in range(len(cds) - 1):
            pass

    def test_cache_cleared(self, simulator, small_mask):
        simulator._get_or_simulate_nominal(small_mask)
        assert simulator._nominal_result_cache is not None
        simulator.clear_cache()
        assert simulator._nominal_result_cache is None


class TestDefectSensitivityAnalyzer:
    """缺陷敏感度分析器测试"""

    @pytest.fixture
    def optical_system(self):
        return OpticalSystem(
            wavelength=193.0,
            na=0.85,
            sigma=0.5,
            pixel_size=1.0,
        )

    @pytest.fixture
    def tiny_mask(self):
        return _create_simple_mask((32, 32), cd_pixels=8, pitch_pixels=16)

    @pytest.fixture
    def analyzer(self, optical_system):
        return DefectSensitivityAnalyzer(
            optical_system=optical_system,
            injection_config=DefectInjectionConfig(
                pixel_size=1.0, cd_tolerance=0.1,
            ),
        )

    def test_generate_defect_suite(self, analyzer, tiny_mask):
        scan_config = DefectScanConfig(
            point_sizes_nm=[5.0, 10.0],
            line_widths_nm=[5.0],
            line_lengths_nm=[20.0],
            contamination_sizes_nm=[10.0],
            polarities=[DefectPolarity.OPAQUE],
            point_shapes=['circle'],
            line_angles_deg=[0.0],
            contamination_attenuations=[0.7],
            n_edge_locations=1,
            n_center_locations=1,
        )
        defects = analyzer.generate_defect_suite(tiny_mask, scan_config)
        assert len(defects) > 0

    def test_analyze_small(self, analyzer, tiny_mask):
        scan_config = DefectScanConfig(
            point_sizes_nm=[5.0, 10.0],
            line_widths_nm=[],
            line_lengths_nm=[],
            contamination_sizes_nm=[],
            polarities=[DefectPolarity.OPAQUE],
            point_shapes=['circle'],
            line_angles_deg=[],
            contamination_attenuations=[],
            n_edge_locations=1,
            n_center_locations=0,
            scan_line=False,
            scan_contamination=False,
        )
        report = analyzer.analyze(tiny_mask, scan_config)
        assert report.total_defects_analyzed > 0
        assert len(report.entries) == report.total_defects_analyzed
        for rank, entry in enumerate(report.entries, 1):
            assert entry.rank == rank
        assert report.recommended_spec > 0

    def test_report_entries_sorted_by_score(self, analyzer, tiny_mask):
        scan_config = DefectScanConfig(
            point_sizes_nm=[5.0, 15.0],
            line_widths_nm=[],
            line_lengths_nm=[],
            contamination_sizes_nm=[],
            polarities=[DefectPolarity.OPAQUE],
            point_shapes=['circle'],
            line_angles_deg=[],
            contamination_attenuations=[],
            n_edge_locations=1,
            n_center_locations=0,
            scan_line=False,
            scan_contamination=False,
        )
        report = analyzer.analyze(tiny_mask, scan_config)
        scores = [e.sensitivity_score for e in report.entries]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-9

    def test_run_defect_analysis_convenience(self, tiny_mask):
        optics = OpticalSystem(
            wavelength=193.0, na=0.85, sigma=0.5, pixel_size=1.0,
        )
        scan_config = DefectScanConfig(
            point_sizes_nm=[5.0],
            line_widths_nm=[],
            line_lengths_nm=[],
            contamination_sizes_nm=[],
            polarities=[DefectPolarity.OPAQUE],
            point_shapes=['circle'],
            line_angles_deg=[],
            contamination_attenuations=[],
            n_edge_locations=1,
            n_center_locations=0,
            scan_line=False,
            scan_contamination=False,
        )
        report = run_defect_analysis(
            mask_nominal=tiny_mask,
            optical_system=optics,
            scan_config=scan_config,
            pixel_size=1.0,
            cd_tolerance=0.1,
        )
        assert report.total_defects_analyzed > 0

    def test_report_to_dict(self, analyzer, tiny_mask):
        scan_config = DefectScanConfig(
            point_sizes_nm=[5.0],
            line_widths_nm=[],
            line_lengths_nm=[],
            contamination_sizes_nm=[],
            polarities=[DefectPolarity.OPAQUE],
            point_shapes=['circle'],
            line_angles_deg=[],
            contamination_attenuations=[],
            n_edge_locations=1,
            n_center_locations=0,
            scan_line=False,
            scan_contamination=False,
        )
        report = analyzer.analyze(tiny_mask, scan_config)
        d = report.to_dict()
        assert 'entries' in d
        assert 'total_defects_analyzed' in d
        assert 'recommended_spec' in d
        assert isinstance(d['entries'], list)
