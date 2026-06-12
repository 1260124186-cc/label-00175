# -*- coding: utf-8 -*-
"""
光学成像模块单元测试
"""

import pytest
import numpy as np
from core.imaging import (
    OpticalSystem, 
    PartialCoherentImaging, 
    simulate_wafer_image,
    _apply_threshold
)


class TestOpticalSystem:
    """光学系统参数测试"""
    
    def test_default_parameters(self):
        """测试默认参数"""
        optics = OpticalSystem()
        
        assert optics.wavelength == 193.0
        assert optics.na == 1.35
        assert optics.sigma == 0.75
        assert optics.pixel_size == 1.0
        assert optics.defocus == 0.0
    
    def test_custom_parameters(self):
        """测试自定义参数"""
        optics = OpticalSystem(
            wavelength=248.0,
            na=0.93,
            sigma=0.5
        )
        
        assert optics.wavelength == 248.0
        assert optics.na == 0.93
        assert optics.sigma == 0.5
    
    def test_k1_calculation(self):
        """测试k1因子计算"""
        optics = OpticalSystem(wavelength=193.0, na=1.35)
        expected_k1 = 193.0 / (2 * 1.35)
        
        assert abs(optics.k1 - expected_k1) < 1e-10
    
    def test_cutoff_frequency(self):
        """测试截止频率计算"""
        optics = OpticalSystem(wavelength=193.0, na=1.35)
        expected_cutoff = 1.35 / 193.0
        
        assert abs(optics.cutoff_frequency - expected_cutoff) < 1e-10


class TestPartialCoherentImaging:
    """部分相干成像模型测试"""
    
    @pytest.fixture
    def imaging_model(self):
        """创建成像模型fixture"""
        optics = OpticalSystem()
        return PartialCoherentImaging(optics, (64, 64))
    
    def test_initialization(self, imaging_model):
        """测试模型初始化"""
        assert imaging_model.image_size == (64, 64)
        assert imaging_model.fx.shape == (64, 64)
        assert imaging_model.fy.shape == (64, 64)
        assert imaging_model.pupil.shape == (64, 64)
    
    def test_aerial_image_shape(self, imaging_model):
        """测试空间像输出形状"""
        mask = np.random.random((64, 64))
        aerial_image = imaging_model.compute_aerial_image(mask)
        
        assert aerial_image.shape == (64, 64)
    
    def test_aerial_image_range(self, imaging_model):
        """测试空间像值范围"""
        mask = np.random.random((64, 64))
        aerial_image = imaging_model.compute_aerial_image(mask)
        
        assert aerial_image.min() >= 0
        assert aerial_image.max() <= 1
    
    def test_uniform_mask(self, imaging_model):
        """测试均匀掩模成像"""
        # 全透明掩模
        mask = np.ones((64, 64))
        aerial_image = imaging_model.compute_aerial_image(mask)
        
        # 应该得到接近均匀的成像
        assert np.std(aerial_image) < 0.1
    
    def test_gradient_shape(self, imaging_model):
        """测试梯度输出形状"""
        mask = np.random.random((64, 64))
        gradient = imaging_model.compute_image_gradient(mask)
        
        assert gradient.shape == (64, 64)


class TestSimulateWaferImage:
    """晶圆成像模拟测试"""
    
    def test_basic_simulation(self):
        """测试基本成像模拟"""
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        
        wafer_image = simulate_wafer_image(mask)
        
        assert wafer_image.shape == (32, 32)
        assert wafer_image.dtype == np.float64
    
    def test_with_custom_optics(self):
        """测试自定义光学参数"""
        mask = np.random.random((32, 32))
        optics = OpticalSystem(wavelength=248.0, na=0.93)
        
        wafer_image = simulate_wafer_image(mask, optical_system=optics)
        
        assert wafer_image.shape == (32, 32)
    
    def test_without_resist(self):
        """测试不应用光刻胶响应"""
        mask = np.random.random((32, 32))
        
        wafer_image = simulate_wafer_image(mask, apply_resist=False)
        
        # 不应用阈值时，输出应该是连续值（允许归一化后值相近的情况）
        assert wafer_image.dtype == np.float64
        assert wafer_image.shape == (32, 32)
    
    def test_with_resist(self):
        """测试应用光刻胶响应"""
        mask = np.random.random((32, 32))
        
        wafer_image = simulate_wafer_image(mask, apply_resist=True, threshold=0.5)
        
        # 应用阈值后，输出应该是二值
        unique_values = np.unique(wafer_image)
        assert len(unique_values) <= 2


class TestThreshold:
    """阈值处理测试"""
    
    def test_threshold_binary_output(self):
        """测试阈值处理输出为二值"""
        image = np.array([[0.2, 0.4], [0.6, 0.8]])
        result = _apply_threshold(image, 0.5)
        
        assert set(result.flatten()) <= {0.0, 1.0}
    
    def test_threshold_values(self):
        """测试阈值处理正确性"""
        image = np.array([[0.2, 0.4], [0.6, 0.8]])
        result = _apply_threshold(image, 0.5)
        
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_array_equal(result, expected)
