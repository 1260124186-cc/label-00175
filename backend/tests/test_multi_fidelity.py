# -*- coding: utf-8 -*-
"""
多保真度优化功能测试

测试 MaskOptimizer 中的多保真度调度功能，包括：
- FidelityLevel 枚举
- MultiFidelityConfig 配置
- FidelityScheduler 调度器
- 基于代理误差的自动切换
- 完整优化流程集成
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from algorithms.mask_optimizer import (
    MaskOptimizer,
    OptimizationConfig,
    MultiFidelityConfig,
    FidelityLevel,
    FidelityScheduler,
    SurrogateErrorRecord,
)


class TestFidelityLevel:
    """测试保真度层级枚举"""

    def test_enum_values(self):
        """测试枚举值定义"""
        assert FidelityLevel.LOW.value == "low"
        assert FidelityLevel.MEDIUM.value == "medium"
        assert FidelityLevel.HIGH.value == "high"

    def test_str_comparison(self):
        """测试字符串比较"""
        assert FidelityLevel.LOW == "low"
        assert FidelityLevel.MEDIUM == "medium"
        assert FidelityLevel.HIGH == "high"


class TestMultiFidelityConfig:
    """测试多保真度配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = MultiFidelityConfig()
        assert config.enabled is False
        assert config.start_level == FidelityLevel.LOW
        assert config.mse_threshold_medium == 0.005
        assert config.mse_threshold_high == 0.001
        assert config.ssim_threshold_medium == 0.95
        assert config.ssim_threshold_high == 0.98
        assert config.error_window_size == 10
        assert config.force_high_fidelity_final == 20
        assert config.calibration_interval == 20

    def test_enabled_config(self):
        """测试启用多保真度"""
        config = MultiFidelityConfig(enabled=True)
        assert config.enabled is True

    def test_custom_thresholds(self):
        """测试自定义阈值"""
        config = MultiFidelityConfig(
            enabled=True,
            mse_threshold_medium=5e-2,
            mse_threshold_high=5e-3,
            ssim_threshold_medium=0.8,
            ssim_threshold_high=0.9,
        )
        assert config.mse_threshold_medium == 5e-2
        assert config.mse_threshold_high == 5e-3
        assert config.ssim_threshold_medium == 0.8
        assert config.ssim_threshold_high == 0.9

    def test_to_dict(self):
        """测试序列化到字典"""
        config = MultiFidelityConfig(enabled=True)
        d = config.to_dict()
        assert d['enabled'] is True
        assert d['start_level'] == 'low'

    def test_from_dict(self):
        """测试从字典反序列化"""
        d = {
            'enabled': True,
            'start_level': 'medium',
            'mse_threshold_medium': 0.05,
        }
        config = MultiFidelityConfig.from_dict(d)
        assert config.enabled is True
        assert config.start_level == FidelityLevel.MEDIUM
        assert config.mse_threshold_medium == 0.05


class TestSurrogateErrorRecord:
    """测试代理误差记录"""

    def test_record_creation(self):
        """测试创建记录"""
        record = SurrogateErrorRecord(
            epoch=10,
            mse=0.01,
            ssim=0.9,
            psnr=20.0,
            mae=0.05,
            fidelity_level=FidelityLevel.LOW
        )
        assert record.epoch == 10
        assert record.mse == 0.01
        assert record.ssim == 0.9
        assert record.fidelity_level == FidelityLevel.LOW


class TestFidelityScheduler:
    """测试保真度调度器"""

    def test_initialization(self):
        """测试初始化"""
        config = MultiFidelityConfig(enabled=True)
        scheduler = FidelityScheduler(config, total_iterations=1000)
        assert scheduler.current_level == FidelityLevel.LOW
        assert scheduler.total_iterations == 1000
        assert len(scheduler.error_history) == 0

    def test_record_error(self):
        """测试记录误差"""
        config = MultiFidelityConfig(enabled=True)
        scheduler = FidelityScheduler(config, total_iterations=1000)
        scheduler.record_error(epoch=1, mse=0.1, ssim=0.5, psnr=10.0, mae=0.2)
        assert len(scheduler.error_history) == 1
        assert scheduler.error_history[0].mse == 0.1

    def test_average_error(self):
        """测试滑动窗口平均误差"""
        config = MultiFidelityConfig(enabled=True, error_window_size=3)
        scheduler = FidelityScheduler(config, total_iterations=1000)
        scheduler.record_error(epoch=1, mse=0.1, ssim=0.8, psnr=10.0, mae=0.2)
        scheduler.record_error(epoch=2, mse=0.05, ssim=0.85, psnr=13.0, mae=0.15)
        scheduler.record_error(epoch=3, mse=0.025, ssim=0.9, psnr=16.0, mae=0.1)
        avg_mse, avg_ssim = scheduler._average_error()
        assert abs(avg_mse - (0.1 + 0.05 + 0.025) / 3) < 1e-10
        assert abs(avg_ssim - (0.8 + 0.85 + 0.9) / 3) < 1e-10

    def test_check_switch_criteria(self):
        """测试切换条件检查"""
        config = MultiFidelityConfig(enabled=True, error_window_size=2)
        scheduler = FidelityScheduler(config, total_iterations=1000)
        scheduler.record_error(epoch=1, mse=0.01, ssim=0.8, psnr=10.0, mae=0.05)
        scheduler.record_error(epoch=2, mse=0.01, ssim=0.8, psnr=10.0, mae=0.05)
        assert scheduler._check_switch_criteria(0.005, 0.88, 0.01, 0.85) is True
        assert scheduler._check_switch_criteria(0.02, 0.7, 0.01, 0.85) is False

    def test_force_high_fidelity_final(self):
        """测试最后N次迭代强制切换到HIGH"""
        config = MultiFidelityConfig(enabled=True, force_high_fidelity_final=10)
        scheduler = FidelityScheduler(config, total_iterations=100)
        result = scheduler.check_and_switch(95)
        assert result == FidelityLevel.HIGH

    def test_update_iter(self):
        """测试迭代更新"""
        config = MultiFidelityConfig(
            enabled=True,
            mse_threshold_medium=0.5,
            ssim_threshold_medium=0.5,
            consecutive_failures=1,
        )
        scheduler = FidelityScheduler(config, total_iterations=100)
        scheduler.record_error(epoch=1, mse=0.1, ssim=0.9, psnr=20.0, mae=0.05)
        scheduler.record_error(epoch=2, mse=0.05, ssim=0.95, psnr=25.0, mae=0.03)
        new_level = scheduler.update_iter(2)
        assert new_level in [FidelityLevel.LOW, FidelityLevel.MEDIUM]


class TestMultiFidelityIntegration:
    """测试多保真度与 MaskOptimizer 的集成"""

    def test_config_with_multi_fidelity(self):
        """测试配置多保真度优化"""
        mf_config = MultiFidelityConfig(enabled=True)
        opt_config = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=5,
            learning_rate=0.1,
            verbose=False,
            multi_fidelity=mf_config,
        )
        assert opt_config.multi_fidelity.enabled is True

    def test_config_serialization(self):
        """测试配置序列化"""
        mf_config = MultiFidelityConfig(enabled=True)
        opt_config = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=5,
            multi_fidelity=mf_config,
        )
        d = opt_config.to_dict()
        assert 'multi_fidelity' in d
        assert d['multi_fidelity']['enabled'] is True

        opt_config2 = OptimizationConfig.from_dict(d)
        assert opt_config2.multi_fidelity.enabled is True

    def test_optimizer_initialization(self):
        """测试优化器初始化多保真度"""
        mf_config = MultiFidelityConfig(enabled=True)
        opt_config = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=5,
            learning_rate=0.1,
            verbose=False,
            multi_fidelity=mf_config,
        )
        optimizer = MaskOptimizer(config=opt_config)
        assert optimizer.config.multi_fidelity.enabled is True

    def test_basic_multi_fidelity_optimization(self):
        """测试基本多保真度优化流程"""
        mf_config = MultiFidelityConfig(
            enabled=True,
            error_window_size=2,
            consecutive_failures=1,
        )
        opt_config = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=5,
            learning_rate=0.1,
            verbose=False,
            multi_fidelity=mf_config,
        )
        optimizer = MaskOptimizer(config=opt_config)

        initial_mask = np.random.random((16, 16))
        target = np.random.random((16, 16))

        result = optimizer.optimize(initial_mask, target)

        assert result.optimized_mask.shape == (16, 16)
        assert len(result.loss_history) > 0
        assert result.fidelity_level_history is not None
        assert len(result.fidelity_level_history) > 0

    def test_optimization_result_contains_history(self):
        """测试优化结果包含保真度历史"""
        mf_config = MultiFidelityConfig(enabled=True)
        opt_config = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=3,
            learning_rate=0.1,
            verbose=False,
            multi_fidelity=mf_config,
        )
        optimizer = MaskOptimizer(config=opt_config)

        initial_mask = np.random.random((8, 8))
        target = np.random.random((8, 8))

        result = optimizer.optimize(initial_mask, target)

        assert result.fidelity_level_history is not None
        assert result.surrogate_error_history is not None
        assert isinstance(result.fidelity_level_history, list)
        assert isinstance(result.surrogate_error_history, list)

    def test_multi_fidelity_disabled(self):
        """测试多保真度禁用时历史为None"""
        opt_config = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=3,
            learning_rate=0.1,
            verbose=False,
        )
        optimizer = MaskOptimizer(config=opt_config)

        initial_mask = np.random.random((8, 8))
        target = np.random.random((8, 8))

        result = optimizer.optimize(initial_mask, target)

        assert result.fidelity_level_history is None
        assert result.surrogate_error_history is None

    def test_fidelity_levels_in_history(self):
        """测试历史记录中的保真度层级值正确"""
        mf_config = MultiFidelityConfig(enabled=True)
        opt_config = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=5,
            learning_rate=0.1,
            verbose=False,
            multi_fidelity=mf_config,
        )
        optimizer = MaskOptimizer(config=opt_config)

        initial_mask = np.random.random((16, 16))
        target = np.random.random((16, 16))

        result = optimizer.optimize(initial_mask, target)

        valid_levels = {'low', 'medium', 'high'}
        for level in result.fidelity_level_history:
            assert level in valid_levels


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
