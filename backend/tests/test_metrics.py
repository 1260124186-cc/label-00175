# -*- coding: utf-8 -*-
"""
误差评估模块单元测试
"""

import pytest
import numpy as np
import time
from core.metrics import (
    mse, mae, ssim, normalized_correlation, psnr,
    evaluate_all, batch_evaluate, compute_error_map,
    MetricsResult,
    ssim_gradient, ssim_loss_gradient
)


class TestMSE:
    """MSE测试"""
    
    def test_identical_images(self):
        """测试相同图像MSE为0"""
        img = np.random.random((32, 32))
        result = mse(img, img)
        
        assert abs(result) < 1e-10
    
    def test_mse_positive(self):
        """测试MSE非负"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        result = mse(img1, img2)
        
        assert result >= 0
    
    def test_mse_symmetric(self):
        """测试MSE对称性"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        assert abs(mse(img1, img2) - mse(img2, img1)) < 1e-10
    
    def test_mse_known_value(self):
        """测试已知MSE值"""
        img1 = np.zeros((2, 2))
        img2 = np.ones((2, 2))
        
        # MSE = mean((0-1)^2) = 1
        assert abs(mse(img1, img2) - 1.0) < 1e-10


class TestMAE:
    """MAE测试"""
    
    def test_identical_images(self):
        """测试相同图像MAE为0"""
        img = np.random.random((32, 32))
        result = mae(img, img)
        
        assert abs(result) < 1e-10
    
    def test_mae_positive(self):
        """测试MAE非负"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        result = mae(img1, img2)
        
        assert result >= 0
    
    def test_mae_known_value(self):
        """测试已知MAE值"""
        img1 = np.zeros((2, 2))
        img2 = np.ones((2, 2))
        
        # MAE = mean(|0-1|) = 1
        assert abs(mae(img1, img2) - 1.0) < 1e-10


class TestSSIM:
    """SSIM测试"""
    
    def test_identical_images(self):
        """测试相同图像SSIM为1"""
        img = np.random.random((32, 32))
        result = ssim(img, img)
        
        assert abs(result - 1.0) < 0.01  # 允许小误差
    
    def test_ssim_range(self):
        """测试SSIM范围"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        result = ssim(img1, img2)
        
        assert -1 <= result <= 1
    
    def test_ssim_symmetric(self):
        """测试SSIM对称性"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        assert abs(ssim(img1, img2) - ssim(img2, img1)) < 0.01


class TestNormalizedCorrelation:
    """归一化相关系数测试"""
    
    def test_identical_images(self):
        """测试相同图像NCC为1"""
        img = np.random.random((32, 32))
        result = normalized_correlation(img, img)
        
        assert abs(result - 1.0) < 1e-10
    
    def test_ncc_range(self):
        """测试NCC范围"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        result = normalized_correlation(img1, img2)
        
        assert -1 <= result <= 1
    
    def test_negative_correlation(self):
        """测试负相关"""
        img1 = np.array([[1, 0], [0, 1]], dtype=float)
        img2 = np.array([[0, 1], [1, 0]], dtype=float)
        
        result = normalized_correlation(img1, img2)
        
        assert result < 0


class TestPSNR:
    """PSNR测试"""
    
    def test_identical_images(self):
        """测试相同图像PSNR很高"""
        img = np.random.random((32, 32))
        result = psnr(img, img)
        
        assert result >= 100  # 应该返回最大值
    
    def test_psnr_positive(self):
        """测试PSNR为正"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        result = psnr(img1, img2)
        
        assert result > 0


class TestEvaluateAll:
    """综合评估测试"""
    
    def test_evaluate_all_returns_metrics_result(self):
        """测试返回MetricsResult对象"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        result = evaluate_all(img1, img2)
        
        assert isinstance(result, MetricsResult)
    
    def test_evaluate_all_has_all_metrics(self):
        """测试包含所有指标"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        result = evaluate_all(img1, img2)
        
        assert hasattr(result, 'mse')
        assert hasattr(result, 'mae')
        assert hasattr(result, 'ssim')
        assert hasattr(result, 'ncc')
        assert hasattr(result, 'psnr')
    
    def test_to_dict(self):
        """测试转换为字典"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        result = evaluate_all(img1, img2)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'mse' in result_dict
        assert 'ssim' in result_dict


class TestBatchEvaluate:
    """批量评估测试"""
    
    def test_batch_evaluate_length(self):
        """测试批量评估结果长度"""
        images = [np.random.random((32, 32)) for _ in range(5)]
        target = np.random.random((32, 32))
        
        results = batch_evaluate(images, target)
        
        assert len(results) == 5
    
    def test_batch_evaluate_selected_metrics(self):
        """测试选择性指标评估"""
        images = [np.random.random((32, 32)) for _ in range(3)]
        target = np.random.random((32, 32))
        
        results = batch_evaluate(images, target, metrics=['mse', 'mae'])
        
        for result in results:
            assert 'mse' in result
            assert 'mae' in result
            assert 'ssim' not in result


class TestErrorMap:
    """误差分布图测试"""
    
    def test_absolute_error_map(self):
        """测试绝对误差图"""
        img1 = np.zeros((32, 32))
        img2 = np.ones((32, 32))
        
        error_map = compute_error_map(img1, img2, 'absolute')
        
        np.testing.assert_array_almost_equal(error_map, np.ones((32, 32)))
    
    def test_squared_error_map(self):
        """测试平方误差图"""
        img1 = np.zeros((32, 32))
        img2 = np.ones((32, 32)) * 2
        
        error_map = compute_error_map(img1, img2, 'squared')
        
        np.testing.assert_array_almost_equal(error_map, np.ones((32, 32)) * 4)
    
    def test_signed_error_map(self):
        """测试有符号误差图"""
        img1 = np.ones((32, 32))
        img2 = np.zeros((32, 32))
        
        error_map = compute_error_map(img1, img2, 'signed')
        
        np.testing.assert_array_almost_equal(error_map, np.ones((32, 32)))
    
    def test_invalid_error_type(self):
        """测试无效误差类型"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))
        
        with pytest.raises(ValueError):
            compute_error_map(img1, img2, 'invalid')


def _numerical_ssim_gradient(image1: np.ndarray,
                             image2: np.ndarray,
                             eps: float = 1e-5,
                             window_size: int = 11,
                             k1: float = 0.01,
                             k2: float = 0.03,
                             data_range: float = 1.0) -> np.ndarray:
    """数值差分计算 SSIM 梯度（用于验证解析梯度）"""
    ny, nx = image1.shape
    grad = np.zeros((ny, nx), dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            img_plus = image1.copy()
            img_plus[i, j] += eps
            img_minus = image1.copy()
            img_minus[i, j] -= eps
            ssim_plus = ssim(img_plus, image2, window_size, k1, k2, data_range)
            ssim_minus = ssim(img_minus, image2, window_size, k1, k2, data_range)
            grad[i, j] = (ssim_plus - ssim_minus) / (2 * eps)
    return grad


class TestSSIMGradient:
    """SSIM 解析梯度测试"""

    def test_gradient_shape(self):
        """测试梯度输出形状与输入一致"""
        np.random.seed(42)
        img1 = np.random.random((16, 16))
        img2 = np.random.random((16, 16))
        grad = ssim_gradient(img1, img2)
        assert grad.shape == img1.shape

    def test_gradient_sign_vs_ssim_loss(self):
        """测试 ssim_loss_gradient = -ssim_gradient"""
        np.random.seed(42)
        img1 = np.random.random((16, 16))
        img2 = np.random.random((16, 16))
        grad_ssim = ssim_gradient(img1, img2)
        grad_loss = ssim_loss_gradient(img1, img2)
        np.testing.assert_array_almost_equal(grad_loss, -grad_ssim)

    def test_analytical_vs_numerical_small(self):
        """小图像上解析梯度与数值梯度对比（8x8）"""
        np.random.seed(42)
        img1 = np.random.random((8, 8))
        img2 = np.random.random((8, 8))
        grad_analytical = ssim_gradient(img1, img2, window_size=3)
        grad_numerical = _numerical_ssim_gradient(img1, img2, window_size=3, eps=1e-5)
        rel_error = np.max(np.abs(grad_analytical - grad_numerical)) / (
            np.max(np.abs(grad_numerical)) + 1e-12
        )
        assert rel_error < 1e-3, f"相对误差 {rel_error:.2e} 超过阈值 1e-3"

    def test_analytical_vs_numerical_medium(self):
        """中等图像上解析梯度与数值梯度对比（12x12）"""
        np.random.seed(123)
        img1 = np.random.random((12, 12))
        img2 = np.random.random((12, 12))
        grad_analytical = ssim_gradient(img1, img2, window_size=5)
        grad_numerical = _numerical_ssim_gradient(img1, img2, window_size=5, eps=1e-5)
        rel_error = np.max(np.abs(grad_analytical - grad_numerical)) / (
            np.max(np.abs(grad_numerical)) + 1e-12
        )
        assert rel_error < 1e-3, f"相对误差 {rel_error:.2e} 超过阈值 1e-3"

    def test_identical_images_gradient_finite(self):
        """相同图像的梯度应为有限值（不全为 0，因均值等仍可能变化）"""
        np.random.seed(7)
        img = np.random.random((16, 16))
        grad = ssim_gradient(img, img)
        assert np.all(np.isfinite(grad))

    def test_constant_image_gradient(self):
        """常数图像：梯度数值应很小（SSIM≈1，对小扰动不敏感）"""
        img1 = 0.5 * np.ones((10, 10), dtype=np.float64)
        img2 = 0.5 * np.ones((10, 10), dtype=np.float64)
        grad = ssim_gradient(img1, img2, window_size=3)
        assert np.max(np.abs(grad)) < 1e-6

    def test_performance_speedup(self):
        """验证解析梯度远快于数值梯度（64x64 尺寸下）"""
        np.random.seed(99)
        size = 64
        img1 = np.random.random((size, size))
        img2 = np.random.random((size, size))

        t0 = time.time()
        _ = ssim_gradient(img1, img2)
        _ = ssim_gradient(img1, img2)
        t_analytical = time.time() - t0

        n_pixels_numerical = 100
        idx = np.random.choice(size * size, n_pixels_numerical, replace=False)
        eps = 1e-5
        t0 = time.time()
        grad_num_partial = np.zeros(size * size, dtype=np.float64)
        for k in idx:
            i, j = divmod(k, size)
            img_p = img1.copy()
            img_p[i, j] += eps
            img_m = img1.copy()
            img_m[i, j] -= eps
            ssim_p = ssim(img_p, img2)
            ssim_m = ssim(img_m, img2)
            grad_num_partial[k] = (ssim_p - ssim_m) / (2 * eps)
        t_numerical_partial = time.time() - t0

        estimated_numerical_total = t_numerical_partial * (size * size) / n_pixels_numerical
        speedup = estimated_numerical_total / max(t_analytical, 1e-9)
        assert speedup >= 10.0, (
            f"解析梯度加速比 {speedup:.1f}x 低于预期 10x; "
            f"解析耗时 {t_analytical:.3f}s, 估计数值耗时 {estimated_numerical_total:.3f}s"
        )
