# -*- coding: utf-8 -*-
"""
光刻专用指标单元测试
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend')

from core.litho_metrics import (
    extract_edges,
    compute_epe,
    compute_cd,
    compute_cd_error,
    compute_ils,
    compute_nils,
    compute_process_window_area,
    compute_meef_simple,
    evaluate_litho_metrics,
    LithoMetricsResult,
)


def _make_square_target(size: int = 32, side: int = 12) -> np.ndarray:
    target = np.zeros((size, size), dtype=np.float64)
    margin = (size - side) // 2
    target[margin:margin + side, margin:margin + side] = 1.0
    return target


def _make_shifted_wafer(size: int = 32, side: int = 12, shift: int = 1) -> np.ndarray:
    wafer = np.zeros((size, size), dtype=np.float64)
    margin = (size - side) // 2 + shift
    wafer[margin:margin + side, margin:margin + side] = 1.0
    return wafer


def _make_aerial_from_binary(binary: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(binary.astype(np.float64), sigma=sigma)


class TestExtractEdges:
    """边缘提取测试"""

    def test_morphological_edges_shape(self):
        img = _make_square_target(16, 8)
        edges = extract_edges(img, method='morphological')
        assert edges.shape == img.shape

    def test_morphological_edges_nonempty(self):
        img = _make_square_target(16, 8)
        edges = extract_edges(img, method='morphological')
        assert np.sum(edges) > 0

    def test_morphological_edges_binary(self):
        img = _make_square_target(16, 8)
        edges = extract_edges(img, method='morphological')
        assert set(np.unique(edges)).issubset({0.0, 1.0})

    def test_sobel_edges_shape(self):
        img = _make_square_target(16, 8)
        edges = extract_edges(img, method='sobel')
        assert edges.shape == img.shape

    def test_sobel_edges_nonempty(self):
        img = _make_square_target(16, 8)
        edges = extract_edges(img, method='sobel')
        assert np.sum(edges) > 0

    def test_blank_image_no_edges(self):
        img = np.zeros((16, 16))
        edges_morph = extract_edges(img, method='morphological')
        edges_sobel = extract_edges(img, method='sobel')
        assert np.sum(edges_morph) == 0
        assert np.sum(edges_sobel) == 0

    def test_full_image_boundary_edges(self):
        img = np.ones((16, 16))
        edges_morph = extract_edges(img, method='morphological')
        assert np.sum(edges_morph) >= 0

    def test_invalid_method_raises(self):
        img = _make_square_target(16, 8)
        with pytest.raises(ValueError):
            extract_edges(img, method='invalid')


class TestEPE:
    """EPE 测试"""

    def test_identical_images_epe_zero(self):
        target = _make_square_target(32, 12)
        result = compute_epe(target, target)
        assert result['epe_mean'] < 1e-10

    def test_shifted_image_epe_positive(self):
        target = _make_square_target(32, 12)
        wafer = _make_shifted_wafer(32, 12, shift=2)
        result = compute_epe(wafer, target)
        assert result['epe_mean'] > 0

    def test_epe_max_ge_mean(self):
        target = _make_square_target(32, 12)
        wafer = _make_shifted_wafer(32, 12, shift=2)
        result = compute_epe(wafer, target)
        assert result['epe_max'] >= result['epe_mean']

    def test_epe_pixel_size_scaling(self):
        target = _make_square_target(32, 12)
        wafer = _make_shifted_wafer(32, 12, shift=1)
        r1 = compute_epe(wafer, target, pixel_size=1.0)
        r2 = compute_epe(wafer, target, pixel_size=10.0)
        assert abs(r2['epe_mean'] / r1['epe_mean'] - 10.0) < 0.1

    def test_epe_both_blank(self):
        blank = np.zeros((16, 16))
        result = compute_epe(blank, blank)
        assert result['epe_mean'] == 0.0

    def test_epe_result_keys(self):
        target = _make_square_target(32, 12)
        result = compute_epe(target, target)
        for key in ('epe_mean', 'epe_max', 'epe_std', 'epe_median'):
            assert key in result


class TestCD:
    """CD 测量测试"""

    def test_square_cd_horizontal(self):
        side = 10
        target = _make_square_target(32, side)
        cd = compute_cd(target, direction='horizontal', pixel_size=1.0)
        assert abs(cd['cd_mean'] - side) < 0.5

    def test_square_cd_vertical(self):
        side = 10
        target = _make_square_target(32, side)
        cd = compute_cd(target, direction='vertical', pixel_size=1.0)
        assert abs(cd['cd_mean'] - side) < 0.5

    def test_cd_pixel_size_scaling(self):
        side = 10
        target = _make_square_target(32, side)
        cd1 = compute_cd(target, pixel_size=1.0)
        cd5 = compute_cd(target, pixel_size=5.0)
        assert abs(cd5['cd_mean'] / cd1['cd_mean'] - 5.0) < 0.1

    def test_cd_blank_image(self):
        blank = np.zeros((16, 16))
        cd = compute_cd(blank)
        assert cd['cd_mean'] == 0.0
        assert cd['n_features'] == 0

    def test_cd_both_directions(self):
        target = _make_square_target(32, 10)
        cd = compute_cd(target, direction='both')
        assert cd['n_features'] > 0

    def test_cd_result_keys(self):
        target = _make_square_target(32, 10)
        cd = compute_cd(target)
        for key in ('cd_mean', 'cd_min', 'cd_max', 'cd_std', 'n_features'):
            assert key in cd


class TestCDError:
    """CD误差测试"""

    def test_identical_images_cd_error_zero(self):
        target = _make_square_target(32, 10)
        result = compute_cd_error(target, target)
        assert abs(result['cd_error_mean']) < 0.5

    def test_larger_wafer_positive_error(self):
        target = _make_square_target(32, 10)
        wafer = _make_square_target(32, 14)
        result = compute_cd_error(wafer, target)
        assert result['cd_error_mean'] > 0

    def test_smaller_wafer_negative_error(self):
        target = _make_square_target(32, 14)
        wafer = _make_square_target(32, 10)
        result = compute_cd_error(wafer, target)
        assert result['cd_error_mean'] < 0

    def test_cd_error_relative(self):
        target = _make_square_target(32, 10)
        wafer = _make_square_target(32, 11)
        result = compute_cd_error(wafer, target)
        assert result['cd_error_relative'] > 0

    def test_cd_error_result_keys(self):
        target = _make_square_target(32, 10)
        result = compute_cd_error(target, target)
        for key in ('cd_error_mean', 'cd_error_relative',
                    'cd_wafer_mean', 'cd_target_mean',
                    'cd_wafer', 'cd_target'):
            assert key in result


class TestILS:
    """ILS 测试"""

    def test_ils_positive(self):
        binary = _make_square_target(32, 12)
        aerial = _make_aerial_from_binary(binary, sigma=1.0)
        result = compute_ils(aerial, threshold=0.3, pixel_size=1.0)
        if result['n_sample_points'] > 0:
            assert result['ils_mean'] > 0

    def test_ils_sharper_edge_higher_ils(self):
        binary = _make_square_target(32, 12)
        aerial_sharp = _make_aerial_from_binary(binary, sigma=0.5)
        aerial_blur = _make_aerial_from_binary(binary, sigma=2.0)
        ils_sharp = compute_ils(aerial_sharp, threshold=0.3, pixel_size=1.0)
        ils_blur = compute_ils(aerial_blur, threshold=0.3, pixel_size=1.0)
        if ils_sharp['n_sample_points'] > 0 and ils_blur['n_sample_points'] > 0:
            assert ils_sharp['ils_mean'] >= ils_blur['ils_mean']

    def test_ils_uniform_image(self):
        uniform = np.ones((16, 16)) * 0.5
        result = compute_ils(uniform, threshold=0.5)
        assert abs(result['ils_mean']) < 1e-10

    def test_ils_result_keys(self):
        binary = _make_square_target(32, 12)
        aerial = _make_aerial_from_binary(binary, sigma=1.0)
        result = compute_ils(aerial, threshold=0.3)
        for key in ('ils_mean', 'ils_min', 'ils_max', 'ils_std', 'n_sample_points'):
            assert key in result


class TestNILS:
    """NILS 测试"""

    def test_nils_positive(self):
        binary = _make_square_target(32, 12)
        aerial = _make_aerial_from_binary(binary, sigma=1.0)
        cd_target = 12.0
        result = compute_nils(aerial, cd_target=cd_target, threshold=0.3, pixel_size=1.0)
        if result['n_sample_points'] > 0:
            assert result['nils_mean'] > 0

    def test_nils_equals_ils_times_cd(self):
        binary = _make_square_target(32, 12)
        aerial = _make_aerial_from_binary(binary, sigma=1.0)
        cd_target = 10.0
        ils = compute_ils(aerial, threshold=0.3, pixel_size=1.0)
        nils = compute_nils(aerial, cd_target=cd_target, threshold=0.3, pixel_size=1.0)
        if ils['n_sample_points'] > 0:
            expected = ils['ils_mean'] * cd_target
            assert abs(nils['nils_mean'] - expected) < 1e-10

    def test_nils_result_contains_ils(self):
        binary = _make_square_target(32, 12)
        aerial = _make_aerial_from_binary(binary, sigma=1.0)
        result = compute_nils(aerial, cd_target=12.0, threshold=0.3)
        assert 'ils' in result


class TestProcessWindowArea:
    """工艺窗口面积测试"""

    def _make_simple_conditions(self):
        from core.imaging import ProcessCondition
        conditions = []
        for df in [-50, 0, 50]:
            for dose in [0.9, 1.0, 1.1]:
                conditions.append(ProcessCondition(defocus=df, dose=dose))
        return conditions

    def test_all_passing(self):
        conditions = self._make_simple_conditions()
        cd_target = 40.0
        cd_values = np.full(len(conditions), cd_target)
        result = compute_process_window_area(conditions, cd_values, cd_target, cd_tolerance=0.1)
        assert result['n_passing'] == len(conditions)
        assert result['pw_ratio'] > 0

    def test_none_passing(self):
        conditions = self._make_simple_conditions()
        cd_target = 40.0
        cd_values = np.full(len(conditions), 100.0)
        result = compute_process_window_area(conditions, cd_values, cd_target, cd_tolerance=0.1)
        assert result['n_passing'] == 0
        assert result['pw_area'] == 0.0

    def test_empty_conditions(self):
        result = compute_process_window_area([], np.array([]), 40.0)
        assert result['pw_area'] == 0.0
        assert result['n_total'] == 0

    def test_partial_passing(self):
        conditions = self._make_simple_conditions()
        cd_target = 40.0
        cd_values = np.full(len(conditions), cd_target)
        cd_values[0] = 100.0
        cd_values[-1] = 100.0
        result = compute_process_window_area(conditions, cd_values, cd_target, cd_tolerance=0.1)
        assert result['n_passing'] < len(conditions)
        assert result['n_passing'] > 0

    def test_result_keys(self):
        conditions = self._make_simple_conditions()
        cd_values = np.full(len(conditions), 40.0)
        result = compute_process_window_area(conditions, cd_values, 40.0)
        for key in ('pw_area', 'pw_ratio', 'n_passing', 'n_total',
                    'focus_range', 'dose_range'):
            assert key in result


class TestMEEFSimple:
    """简化版 MEEF 测试"""

    def test_meef_positive(self):
        mask = _make_square_target(32, 10)
        result = compute_meef_simple(mask, threshold=0.5)
        assert result['meef'] > 0

    def test_meef_result_keys(self):
        mask = _make_square_target(32, 10)
        result = compute_meef_simple(mask)
        for key in ('meef', 'cd_nominal', 'cd_dilated', 'cd_eroded', 'delta_cd'):
            assert key in result

    def test_dilated_cd_larger(self):
        mask = _make_square_target(32, 10)
        result = compute_meef_simple(mask)
        assert result['cd_dilated'] >= result['cd_nominal']

    def test_eroded_cd_smaller(self):
        mask = _make_square_target(32, 10)
        result = compute_meef_simple(mask)
        assert result['cd_eroded'] <= result['cd_nominal']


class TestEvaluateLithoMetrics:
    """综合评估测试"""

    def test_basic_evaluation(self):
        target = _make_square_target(32, 10)
        wafer = _make_shifted_wafer(32, 10, shift=1)
        result = evaluate_litho_metrics(wafer, target)
        assert isinstance(result, LithoMetricsResult)
        assert 'epe_mean' in result.epe
        assert 'cd_error_mean' in result.cd_error

    def test_with_aerial_image(self):
        target = _make_square_target(32, 10)
        wafer = _make_shifted_wafer(32, 10, shift=1)
        aerial = _make_aerial_from_binary(wafer, sigma=1.0)
        result = evaluate_litho_metrics(wafer, target, aerial_image=aerial)
        assert 'ils_mean' in result.ils
        assert 'nils_mean' in result.nils

    def test_without_aerial_image(self):
        target = _make_square_target(32, 10)
        wafer = _make_shifted_wafer(32, 10, shift=1)
        result = evaluate_litho_metrics(wafer, target)
        assert result.ils == {}
        assert result.nils == {}

    def test_to_dict(self):
        target = _make_square_target(32, 10)
        result = evaluate_litho_metrics(target, target)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'epe' in d
        assert 'cd_error' in d

    def test_summary(self):
        target = _make_square_target(32, 10)
        wafer = _make_shifted_wafer(32, 10, shift=1)
        result = evaluate_litho_metrics(wafer, target)
        s = result.summary()
        assert isinstance(s, str)
        assert 'EPE' in s


class TestEdgeCases:
    """边界情况测试"""

    def test_single_pixel_feature(self):
        img = np.zeros((16, 16))
        img[8, 8] = 1.0
        edges = extract_edges(img, method='morphological')
        assert np.sum(edges) > 0

    def test_alternating_lines(self):
        img = np.zeros((32, 32))
        for i in range(4, 28, 4):
            img[4:28, i:i + 2] = 1.0
        cd = compute_cd(img, direction='horizontal', pixel_size=1.0)
        assert cd['n_features'] > 0

    def test_very_small_image(self):
        img = np.array([[0, 1], [1, 0]], dtype=np.float64)
        cd = compute_cd(img, direction='both')
        assert cd['n_features'] > 0

    def test_epe_sobel_method(self):
        target = _make_square_target(32, 12)
        wafer = _make_shifted_wafer(32, 12, shift=1)
        result = compute_epe(wafer, target, edge_method='sobel')
        assert result['epe_mean'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
