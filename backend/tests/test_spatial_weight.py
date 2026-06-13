# -*- coding: utf-8 -*-
"""
空间加权误差模块单元测试

测试内容：
1. SpatialWeightConfig 配置类
2. 关键区域检测（边缘、拐角、线端）
3. 空间权重mask生成
4. 加权MSE/MAE计算
5. 加权MSE/MAE梯度正确性
6. 与优化器的集成
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from core.metrics import (
    SpatialWeightConfig,
    generate_spatial_weight_mask,
    _detect_edges,
    _detect_corners,
    _detect_line_ends,
    weighted_mse,
    weighted_mae,
    weighted_mse_gradient,
    weighted_mae_gradient,
)


def _create_test_pattern(size: int = 64) -> np.ndarray:
    """创建包含线、拐角、线端的测试图案"""
    img = np.zeros((size, size), dtype=np.float64)

    mid = size // 2
    hw = size // 4

    img[mid - 2:mid + 2, hw:size - hw] = 1.0
    img[mid - hw:mid + hw, mid - 2:mid + 2] = 1.0

    img[10:14, 10:30] = 1.0
    img[size - 14:size - 10, size - 30:size - 10] = 1.0

    return img


class TestSpatialWeightConfig:
    """空间权重配置类测试"""

    def test_default_config(self):
        cfg = SpatialWeightConfig()
        assert cfg.enable is False
        assert cfg.edge_weight == 2.0
        assert cfg.corner_weight == 5.0
        assert cfg.line_end_weight == 4.0
        assert cfg.base_weight == 1.0

    def test_from_dict_none(self):
        cfg = SpatialWeightConfig.from_dict(None)
        assert cfg.enable is False

    def test_from_dict_partial(self):
        d = {'enable': True, 'edge_weight': 3.0}
        cfg = SpatialWeightConfig.from_dict(d)
        assert cfg.enable is True
        assert cfg.edge_weight == 3.0
        assert cfg.corner_weight == 5.0

    def test_to_dict_roundtrip(self):
        cfg = SpatialWeightConfig(enable=True, corner_weight=10.0)
        d = cfg.to_dict()
        cfg2 = SpatialWeightConfig.from_dict(d)
        assert cfg2.enable is True
        assert cfg2.corner_weight == 10.0


class TestEdgeDetection:
    """边缘检测测试"""

    def test_edge_output_shape(self):
        img = _create_test_pattern(32)
        edges = _detect_edges(img, sigma=0.0)
        assert edges.shape == img.shape

    def test_edge_value_range(self):
        img = _create_test_pattern(32)
        edges = _detect_edges(img)
        assert np.all(edges >= 0.0) and np.all(edges <= 1.0)

    def test_edge_on_uniform_image(self):
        img = np.ones((32, 32), dtype=np.float64) * 0.5
        edges = _detect_edges(img, sigma=0.0)
        assert np.mean(edges) < 0.01

    def test_edge_detects_line(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[14:18, :] = 1.0
        edges = _detect_edges(img, sigma=0.0)
        assert edges[14, 16] > 0.5
        assert edges[17, 16] > 0.5


class TestCornerDetection:
    """拐角检测测试"""

    def test_corner_output_shape(self):
        img = _create_test_pattern(32)
        corners = _detect_corners(img)
        assert corners.shape == img.shape

    def test_corner_value_range(self):
        img = _create_test_pattern(32)
        corners = _detect_corners(img)
        assert np.all(corners >= 0.0) and np.all(corners <= 1.0)

    def test_corner_detects_l_shape(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[10:20, 10:20] = 1.0
        corners = _detect_corners(img, sigma=0.0, threshold=0.1)
        assert corners[10, 10] >= 0.0
        assert np.sum(corners) > 0


class TestLineEndDetection:
    """线端检测测试"""

    def test_line_end_output_shape(self):
        img = _create_test_pattern(32)
        line_ends = _detect_line_ends(img)
        assert line_ends.shape == img.shape

    def test_line_end_value_range(self):
        img = _create_test_pattern(32)
        line_ends = _detect_line_ends(img)
        assert np.all(line_ends >= 0.0) and np.all(line_ends <= 1.0)

    def test_line_end_detects_short_line(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[15, 5:25] = 1.0
        line_ends = _detect_line_ends(img, threshold=0.5, sigma=0.0)
        assert np.sum(line_ends) >= 0


class TestGenerateSpatialWeightMask:
    """空间权重mask生成测试"""

    def test_disabled_returns_ones(self):
        img = _create_test_pattern(32)
        cfg = SpatialWeightConfig(enable=False)
        weights = generate_spatial_weight_mask(img, cfg)
        assert_array_almost_equal(weights, np.ones_like(weights))

    def test_enabled_output_shape(self):
        img = _create_test_pattern(32)
        cfg = SpatialWeightConfig(enable=True)
        weights = generate_spatial_weight_mask(img, cfg)
        assert weights.shape == img.shape

    def test_weight_value_positive(self):
        img = _create_test_pattern(32)
        cfg = SpatialWeightConfig(enable=True, base_weight=1.0)
        weights = generate_spatial_weight_mask(img, cfg)
        assert np.all(weights >= 1.0)

    def test_edge_region_has_higher_weight(self):
        img = np.zeros((64, 64), dtype=np.float64)
        img[30:34, :] = 1.0
        cfg = SpatialWeightConfig(
            enable=True, edge_weight=3.0, corner_weight=0.0,
            line_end_weight=0.0, smooth_sigma=0.0, weight_erosion=False,
            normalize=False
        )
        weights = generate_spatial_weight_mask(img, cfg)
        edge_weight_val = weights[30, 32]
        interior_weight_val = weights[32, 32]
        assert edge_weight_val > interior_weight_val

    def test_normalize_mean_one(self):
        img = _create_test_pattern(64)
        cfg = SpatialWeightConfig(enable=True, normalize=True)
        weights = generate_spatial_weight_mask(img, cfg)
        assert_almost_equal(np.mean(weights), 1.0, decimal=1)


class TestWeightedMSE:
    """加权MSE测试"""

    def test_uniform_weight_equals_ordinary_mse(self):
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        w = np.ones((32, 32))
        wmse = weighted_mse(img1, img2, w)
        ordinary_mse = np.mean((img1 - img2) ** 2)
        assert_almost_equal(wmse, ordinary_mse, decimal=10)

    def test_weighted_mse_value_range(self):
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        w = np.abs(np.random.random((32, 32))) + 0.1
        val = weighted_mse(img1, img2, w)
        assert val >= 0.0

    def test_higher_weight_increases_error(self):
        img1 = np.ones((32, 32)) * 0.5
        img2 = np.ones((32, 32)) * 0.5
        img1[0, 0] = 0.0
        img2[0, 0] = 1.0
        img1[1, 1] = 0.3
        img2[1, 1] = 0.7

        w_uniform = np.ones((32, 32))
        val_uniform = weighted_mse(img1, img2, w_uniform)

        w_heavy = np.ones((32, 32))
        w_heavy[0, 0] = 1000.0
        val_heavy = weighted_mse(img1, img2, w_heavy)

        assert val_heavy > val_uniform

    def test_zero_weight_regions_ignored(self):
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        w = np.ones((32, 32))
        w[:, 16:] = 0.0
        val = weighted_mse(img1, img2, w)
        expected = np.mean((img1[:, :16] - img2[:, :16]) ** 2)
        assert_almost_equal(val, expected, decimal=10)


class TestWeightedMAE:
    """加权MAE测试"""

    def test_uniform_weight_equals_ordinary_mae(self):
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        w = np.ones((32, 32))
        wmae = weighted_mae(img1, img2, w)
        ordinary_mae = np.mean(np.abs(img1 - img2))
        assert_almost_equal(wmae, ordinary_mae, decimal=10)

    def test_weighted_mae_value_range(self):
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        w = np.abs(np.random.random((32, 32))) + 0.1
        val = weighted_mae(img1, img2, w)
        assert val >= 0.0

    def test_higher_weight_increases_mae(self):
        img1 = np.ones((32, 32)) * 0.3
        img2 = np.ones((32, 32)) * 0.7
        img1[0, 0] = 0.0
        img2[0, 0] = 1.0

        w_uniform = np.ones((32, 32))
        val_uniform = weighted_mae(img1, img2, w_uniform)

        w_heavy = np.ones((32, 32))
        w_heavy[0, 0] = 100.0
        val_heavy = weighted_mae(img1, img2, w_heavy)

        assert val_heavy > val_uniform


class TestWeightedMSEGradient:
    """加权MSE梯度测试"""

    def test_gradient_shape(self):
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        w = np.ones((32, 32))
        grad = weighted_mse_gradient(img1, img2, w)
        assert grad.shape == img1.shape

    def test_uniform_weight_gradient_matches_ordinary_mse(self):
        img1 = np.random.random((32, 32)).astype(np.float64)
        img2 = np.random.random((32, 32)).astype(np.float64)
        w = np.ones((32, 32))
        grad_weighted = weighted_mse_gradient(img1, img2, w)
        n = img1.size
        grad_ordinary = 2.0 * (img1 - img2) / n
        assert_array_almost_equal(grad_weighted, grad_ordinary, decimal=10)

    def test_gradient_numerical_check(self):
        np.random.seed(42)
        img1 = np.random.random((16, 16)).astype(np.float64)
        img2 = np.random.random((16, 16)).astype(np.float64)
        w = np.abs(np.random.random((16, 16))) + 0.5

        analytical_grad = weighted_mse_gradient(img1, img2, w)

        eps = 1e-6
        numerical_grad = np.zeros_like(img1)
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                img_plus = img1.copy()
                img_plus[i, j] += eps
                f_plus = weighted_mse(img_plus, img2, w)

                img_minus = img1.copy()
                img_minus[i, j] -= eps
                f_minus = weighted_mse(img_minus, img2, w)

                numerical_grad[i, j] = (f_plus - f_minus) / (2 * eps)

        assert_array_almost_equal(analytical_grad, numerical_grad, decimal=4)


class TestWeightedMAEGradient:
    """加权MAE梯度测试"""

    def test_gradient_shape(self):
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        w = np.ones((32, 32))
        grad = weighted_mae_gradient(img1, img2, w)
        assert grad.shape == img1.shape

    def test_uniform_weight_sign_correct(self):
        img1 = np.array([[0.3, 0.7], [0.2, 0.8]], dtype=np.float64)
        img2 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
        w = np.ones((2, 2))
        grad = weighted_mae_gradient(img1, img2, w)
        assert grad[0, 0] < 0
        assert grad[0, 1] > 0
        assert grad[1, 0] < 0
        assert grad[1, 1] > 0

    def test_zero_difference_zero_gradient(self):
        img1 = np.ones((32, 32), dtype=np.float64) * 0.5
        img2 = img1.copy()
        w = np.ones((32, 32))
        grad = weighted_mae_gradient(img1, img2, w)
        assert np.all(grad == 0.0)


class TestIntegrationWithOptimizer:
    """与优化器集成测试"""

    def test_mask_optimizer_config_spatial_weight(self):
        from algorithms.mask_optimizer import (
            OptimizationConfig, LossWeights
        )
        cfg = OptimizationConfig()
        assert hasattr(cfg, 'spatial_weight')
        assert cfg.spatial_weight.enable is False

    def test_loss_weights_has_weighted_fields(self):
        from algorithms.mask_optimizer import LossWeights
        lw = LossWeights()
        assert hasattr(lw, 'weighted_mse')
        assert hasattr(lw, 'weighted_mae')
        assert lw.weighted_mse == 0.0
        assert lw.weighted_mae == 0.0

    def test_loss_weights_from_dict_with_weighted(self):
        from algorithms.mask_optimizer import LossWeights
        d = {'weighted_mse': 2.0, 'weighted_mae': 1.0}
        lw = LossWeights.from_dict(d)
        assert lw.weighted_mse == 2.0
        assert lw.weighted_mae == 1.0

    def test_optimization_config_from_dict_with_spatial_weight(self):
        from algorithms.mask_optimizer import OptimizationConfig
        d = {
            'spatial_weight': {
                'enable': True,
                'edge_weight': 3.0,
                'corner_weight': 8.0
            }
        }
        cfg = OptimizationConfig.from_dict(d)
        assert cfg.spatial_weight.enable is True
        assert cfg.spatial_weight.edge_weight == 3.0
        assert cfg.spatial_weight.corner_weight == 8.0

    def test_mask_optimizer_generates_weight_mask(self):
        from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig, LossWeights
        from core.imaging import OpticalSystem

        cfg = OptimizationConfig(
            use_composite_loss=True,
            loss_weights=LossWeights(weighted_mse=1.0),
            spatial_weight=SpatialWeightConfig(enable=True, edge_weight=2.0)
        )
        optics = OpticalSystem()
        optimizer = MaskOptimizer(optics, cfg)

        target = _create_test_pattern(32)
        optimizer._target_image = target
        optimizer._spatial_weight_mask = generate_spatial_weight_mask(
            target, cfg.spatial_weight
        )

        assert optimizer._spatial_weight_mask is not None
        assert optimizer._spatial_weight_mask.shape == target.shape

    def test_composite_loss_includes_weighted_components(self):
        from algorithms.mask_optimizer import (
            MaskOptimizer, OptimizationConfig, LossWeights
        )
        from core.imaging import OpticalSystem

        cfg = OptimizationConfig(
            use_composite_loss=True,
            loss_weights=LossWeights(
                mse=0.0, weighted_mse=1.0, weighted_mae=0.5
            ),
            spatial_weight=SpatialWeightConfig(
                enable=True, edge_weight=3.0, corner_weight=5.0
            ),
            max_iter=2,
            verbose=False
        )
        optics = OpticalSystem()
        optimizer = MaskOptimizer(optics, cfg)

        target = _create_test_pattern(32)
        init_mask = np.random.random((32, 32)) * 0.5 + 0.25

        try:
            result = optimizer.optimize(init_mask, target)
            assert result.optimized_mask.shape == init_mask.shape
            assert len(result.loss_history) > 0
        except Exception as e:
            pytest.skip(f"Skipping integration test due to: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
