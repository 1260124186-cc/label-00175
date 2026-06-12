# -*- coding: utf-8 -*-
"""
误差评估模块单元测试
"""

import pytest
import numpy as np
from core.metrics import (
    mse, mae, ssim, normalized_correlation, psnr,
    evaluate_all, batch_evaluate, compute_error_map,
    MetricsResult
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
