# -*- coding: utf-8 -*-
"""
掩模检测图像仿真模块单元测试
"""

import pytest
import numpy as np
from pathlib import Path
import sys

_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from inspection.schemas import (
    InspectionMode,
    DefectClass,
    InspectionOptics,
    InspectionConfig,
    InspectionImageResult,
    DifferenceMapResult,
    DefectCandidate,
    DetectabilityResult,
    InspectionAnalysisConfig,
    FullInspectionResult,
)
from inspection.inspection_simulator import (
    simulate_inspection_image,
    simulate_multi_mode_inspection,
    compute_defect_contrast,
)
from inspection.die_to_database import (
    compute_difference_map,
    compute_detection_threshold,
    threshold_difference_map,
    extract_candidate_regions,
    compute_die_to_database,
    compute_aligned_difference,
)
from inspection.detectability_analysis import (
    analyze_detectability,
    run_full_inspection_analysis,
    compute_false_defect_rate,
    evaluate_detection_performance,
)


def _create_test_mask(shape=(128, 128), cd_pixels=16, pitch_pixels=32):
    """创建简单的线/空间测试掩模"""
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=np.float64)
    for x in range(0, nx, pitch_pixels):
        x_start = x
        x_end = min(x + cd_pixels, nx)
        mask[:, x_start:x_end] = 1.0
    return mask


def _inject_defect(mask, center_y, center_x, size_pixels=5, polarity='opaque'):
    """在掩模中注入简单缺陷"""
    mask_copy = mask.copy()
    half = size_pixels // 2
    y1 = max(0, center_y - half)
    y2 = min(mask.shape[0], center_y + half + 1)
    x1 = max(0, center_x - half)
    x2 = min(mask.shape[1], center_x + half + 1)

    if polarity == 'opaque':
        mask_copy[y1:y2, x1:x2] = 0.0
    else:
        mask_copy[y1:y2, x1:x2] = 1.0

    return mask_copy


class TestSchemas:
    """数据模型测试"""

    def test_inspection_mode_enum(self):
        assert InspectionMode.BRIGHT_FIELD.value == "bright_field"
        assert InspectionMode.DARK_FIELD.value == "dark_field"
        assert InspectionMode.PHASE_CONTRAST.value == "phase_contrast"
        assert InspectionMode.POLARIZATION.value == "polarization"

    def test_defect_class_enum(self):
        assert DefectClass.REAL_DEFECT.value == "real_defect"
        assert DefectClass.NUISANCE_DEFECT.value == "nuisance_defect"
        assert DefectClass.NO_DEFECT.value == "no_defect"

    def test_inspection_optics_defaults(self):
        optics = InspectionOptics()
        assert optics.wavelength_nm == 266.0
        assert optics.numerical_aperture == 0.9
        assert optics.pixel_size_nm == 1.0

    def test_inspection_config_defaults(self):
        config = InspectionConfig()
        assert config.mode == InspectionMode.BRIGHT_FIELD
        assert config.noise_level == 0.03
        assert config.defect_boost == 1.5
        assert config.threshold_abs == 0.15

    def test_inspection_config_to_dict(self):
        config = InspectionConfig(mode=InspectionMode.DARK_FIELD)
        d = config.to_dict()
        assert d['mode'] == 'dark_field'
        assert 'optics' in d

    def test_defect_candidate_to_dict(self):
        candidate = DefectCandidate(
            center_y=50,
            center_x=100,
            size_nm=25.0,
            area_pixels=20,
            contrast=0.3,
            intensity=0.4,
            score=0.8,
            defect_class=DefectClass.REAL_DEFECT,
            is_printable=True,
            bounding_box=(45, 55, 95, 105),
        )
        d = candidate.to_dict()
        assert d['center_y'] == 50
        assert d['center_x'] == 100
        assert d['defect_class'] == 'real_defect'
        assert d['is_printable'] is True

    def test_detectability_result_summary(self):
        result = DetectabilityResult(
            true_positives=8,
            false_positives=2,
            false_negatives=1,
            detection_rate=0.889,
            false_alarm_rate=0.2,
            precision=0.8,
            f1_score=0.842,
            auc_score=0.95,
        )
        summary = result.summary()
        assert '检测率' in summary
        assert '假警报率' in summary
        assert 'AUC' in summary
        assert '88.9%' in summary

    def test_full_inspection_result_to_dict(self):
        config = InspectionAnalysisConfig()
        result = FullInspectionResult(
            config=config,
            total_analysis_time_s=1.5,
        )
        d = result.to_dict()
        assert d['total_analysis_time_s'] == 1.5
        assert 'config' in d


class TestInspectionSimulator:
    """检测图像仿真测试"""

    def test_bright_field_simulation(self):
        mask = _create_test_mask()
        config = InspectionConfig(mode=InspectionMode.BRIGHT_FIELD, noise_level=0.0)
        result = simulate_inspection_image(mask, mask, config, seed=42)

        assert result.inspection_image.shape == mask.shape
        assert result.mode == InspectionMode.BRIGHT_FIELD
        assert np.all(result.inspection_image >= 0)
        assert np.all(result.inspection_image <= 1)

    def test_dark_field_simulation(self):
        mask = _create_test_mask()
        config = InspectionConfig(mode=InspectionMode.DARK_FIELD, noise_level=0.0)
        result = simulate_inspection_image(mask, mask, config, seed=42)

        assert result.inspection_image.shape == mask.shape
        assert result.mode == InspectionMode.DARK_FIELD
        assert np.mean(result.inspection_image) < 0.5

    def test_phase_contrast_simulation(self):
        mask = _create_test_mask()
        config = InspectionConfig(mode=InspectionMode.PHASE_CONTRAST, noise_level=0.0)
        result = simulate_inspection_image(mask, mask, config, seed=42)

        assert result.inspection_image.shape == mask.shape
        assert result.mode == InspectionMode.PHASE_CONTRAST

    def test_polarization_simulation(self):
        mask = _create_test_mask()
        config = InspectionConfig(mode=InspectionMode.POLARIZATION, noise_level=0.0)
        result = simulate_inspection_image(mask, mask, config, seed=42)

        assert result.inspection_image.shape == mask.shape
        assert result.mode == InspectionMode.POLARIZATION

    def test_inspection_with_defect(self):
        mask_ref = _create_test_mask()
        mask_def = _inject_defect(mask_ref, 64, 64, size_pixels=6, polarity='opaque')

        config = InspectionConfig(
            mode=InspectionMode.DARK_FIELD,
            noise_level=0.0,
            defect_boost=2.0,
        )
        result = simulate_inspection_image(mask_def, mask_ref, config, seed=42)

        assert result.inspection_image.shape == mask_def.shape
        assert np.any(result.defect_mask > 0)

    def test_multi_mode_inspection(self):
        mask_ref = _create_test_mask()
        mask_def = _inject_defect(mask_ref, 64, 64, size_pixels=6, polarity='opaque')

        results = simulate_multi_mode_inspection(mask_def, mask_ref, seed=42)

        assert len(results) == 4
        assert InspectionMode.BRIGHT_FIELD in results
        assert InspectionMode.DARK_FIELD in results
        assert InspectionMode.PHASE_CONTRAST in results
        assert InspectionMode.POLARIZATION in results

    def test_compute_defect_contrast(self):
        mask_ref = _create_test_mask()
        mask_def = _inject_defect(mask_ref, 64, 64, size_pixels=8, polarity='opaque')

        config = InspectionConfig(mode=InspectionMode.DARK_FIELD, noise_level=0.0)
        result = simulate_inspection_image(mask_def, mask_ref, config, seed=42)

        contrast_metrics = compute_defect_contrast(result)

        assert 'mean_contrast' in contrast_metrics
        assert 'max_contrast' in contrast_metrics
        assert 'snr' in contrast_metrics
        assert 'cnr' in contrast_metrics
        assert contrast_metrics['mean_contrast'] >= 0
        assert contrast_metrics['snr'] >= 0

    def test_invalid_mode_raises(self):
        mask = _create_test_mask()
        config = InspectionConfig()
        config.mode = "invalid_mode"

        with pytest.raises(ValueError):
            simulate_inspection_image(mask, mask, config)


class TestDieToDatabase:
    """Die-to-Database 差异图计算测试"""

    def test_compute_difference_map(self):
        mask_ref = _create_test_mask()
        mask_def = _inject_defect(mask_ref, 64, 64, size_pixels=8, polarity='opaque')

        config = InspectionConfig(mode=InspectionMode.BRIGHT_FIELD, noise_level=0.0)
        insp_result = simulate_inspection_image(mask_def, mask_ref, config, seed=42)

        diff_map, signed_diff = compute_difference_map(
            insp_result.inspection_image,
            insp_result.reference_image,
            align=False,
            smooth_sigma=0,
        )

        assert diff_map.shape == mask_ref.shape
        assert signed_diff.shape == mask_ref.shape
        assert np.all(diff_map >= 0)

        assert diff_map[60:68, 60:68].max() > 0.1

    def test_compute_detection_threshold(self):
        np.random.seed(42)
        diff_map = np.random.normal(0.05, 0.02, (100, 100))
        diff_map[40:50, 40:50] += 0.2

        threshold = compute_detection_threshold(
            diff_map,
            threshold_abs=0.1,
            threshold_rel=3.0,
            adaptive=True,
        )

        assert threshold >= 0.1
        assert threshold < 0.5

    def test_threshold_difference_map(self):
        diff_map = np.zeros((100, 100))
        diff_map[40:50, 40:50] = 0.5
        diff_map[70:75, 70:75] = 0.3

        binary_map, labeled_map = threshold_difference_map(
            diff_map,
            threshold=0.2,
            min_area_pixels=5,
            connectivity=8,
        )

        assert binary_map.shape == diff_map.shape
        assert labeled_map.max() >= 1
        assert np.any(binary_map[40:50, 40:50])

    def test_extract_candidate_regions(self):
        diff_map = np.zeros((100, 100))
        signed_diff = np.zeros((100, 100))
        diff_map[40:50, 40:50] = 0.5
        signed_diff[40:50, 40:50] = 0.5

        binary_map, labeled_map = threshold_difference_map(diff_map, threshold=0.2)

        regions = extract_candidate_regions(
            binary_map,
            labeled_map,
            diff_map,
            signed_diff,
            pixel_size_nm=1.0,
        )

        assert len(regions) >= 1
        assert 'center_y' in regions[0]
        assert 'center_x' in regions[0]
        assert 'size_nm' in regions[0]
        assert 'confidence' in regions[0]

    def test_compute_die_to_database(self):
        mask_ref = _create_test_mask()
        mask_def = _inject_defect(mask_ref, 64, 64, size_pixels=8, polarity='opaque')

        config = InspectionConfig(mode=InspectionMode.DARK_FIELD, noise_level=0.0)
        insp_result = simulate_inspection_image(mask_def, mask_ref, config, seed=42)

        analysis_config = InspectionAnalysisConfig(
            inspection_config=config,
            diff_threshold_abs=0.05,
        )

        diff_result = compute_die_to_database(
            insp_result.inspection_image,
            insp_result.reference_image,
            config=analysis_config,
            align=False,
        )

        assert isinstance(diff_result, DifferenceMapResult)
        assert diff_result.difference_map.shape == mask_ref.shape
        assert diff_result.threshold_used > 0
        assert len(diff_result.candidate_regions) >= 1

    def test_aligned_difference_methods(self):
        img1 = np.random.rand(50, 50)
        img2 = img1.copy()
        img2[20:30, 20:30] += 0.3

        for method in ['absolute', 'squared', 'relative']:
            diff = compute_aligned_difference(img1, img2, method=method)
            assert diff.shape == img1.shape
            assert np.all(diff >= 0)
            assert diff[20:30, 20:30].max() > diff.max() * 0.5


class TestDetectabilityAnalysis:
    """可检测性分析测试"""

    def test_evaluate_detection_performance(self):
        np.random.seed(42)
        diff_map = np.random.rand(100, 100) * 0.1
        gt_mask = np.zeros((100, 100), dtype=bool)
        diff_map[40:50, 40:50] = 0.8
        gt_mask[40:50, 40:50] = True

        metrics = evaluate_detection_performance(diff_map, gt_mask, threshold=0.5)

        assert metrics['tp'] > 0
        assert 'tpr' in metrics
        assert 'fpr' in metrics
        assert 'precision' in metrics
        assert 'f1' in metrics
        assert 'mcc' in metrics
        assert 0 <= metrics['tpr'] <= 1
        assert 0 <= metrics['fpr'] <= 1

    def test_compute_false_defect_rate(self):
        candidates = [
            DefectCandidate(
                center_y=10, center_x=10, size_nm=20, area_pixels=10,
                contrast=0.3, intensity=0.4, score=0.8,
                defect_class=DefectClass.REAL_DEFECT, is_printable=True,
                bounding_box=(5, 15, 5, 15),
            ),
            DefectCandidate(
                center_y=20, center_x=20, size_nm=15, area_pixels=8,
                contrast=0.2, intensity=0.3, score=0.6,
                defect_class=DefectClass.NUISANCE_DEFECT, is_printable=False,
                bounding_box=(15, 25, 15, 25),
            ),
        ]

        result = DetectabilityResult(detected_defects=candidates)
        fdr = compute_false_defect_rate(result, area_per_die_mm2=1.0)

        assert fdr['nuisance_per_die'] == 1.0
        assert fdr['real_per_die'] == 1.0
        assert fdr['nuisance_ratio'] == 0.5
        assert fdr['nuisance_per_cm2'] > 0


class TestFullInspection:
    """完整检测流程测试"""

    def test_full_inspection_analysis(self):
        mask_ref = _create_test_mask()
        mask_def = _inject_defect(mask_ref, 64, 64, size_pixels=8, polarity='opaque')

        gt_defects = [{
            'center_y': 64,
            'center_x': 64,
            'size_nm': 8.0,
            'is_printable': True,
        }]

        gt_mask = np.zeros_like(mask_ref, dtype=bool)
        gt_mask[60:68, 60:68] = True

        config = InspectionAnalysisConfig(
            inspection_config=InspectionConfig(
                mode=InspectionMode.DARK_FIELD,
                noise_level=0.01,
                defect_boost=1.5,
            ),
            diff_threshold_abs=0.05,
            min_area_pixels=3,
        )

        result = run_full_inspection_analysis(
            mask_def,
            mask_ref,
            config=config,
            ground_truth_defects=gt_defects,
            ground_truth_mask=gt_mask,
            seed=42,
        )

        assert isinstance(result, FullInspectionResult)
        assert result.inspection_result is not None
        assert result.difference_result is not None
        assert result.detectability_result is not None
        assert result.total_analysis_time_s > 0

        assert result.detectability_result.detection_rate >= 0
        assert result.detectability_result.auc_score >= 0

        summary = result.summary()
        assert '检测率' in summary
        assert '总分析时间' in summary

    def test_full_inspection_without_ground_truth(self):
        mask_ref = _create_test_mask()
        mask_def = _inject_defect(mask_ref, 64, 64, size_pixels=10, polarity='opaque')

        config = InspectionAnalysisConfig(
            inspection_config=InspectionConfig(
                mode=InspectionMode.BRIGHT_FIELD,
                noise_level=0.0,
            ),
            diff_threshold_abs=0.05,
        )

        result = run_full_inspection_analysis(
            mask_def,
            mask_ref,
            config=config,
            seed=42,
        )

        assert result.detectability_result is not None
        assert len(result.detectability_result.detected_defects) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
