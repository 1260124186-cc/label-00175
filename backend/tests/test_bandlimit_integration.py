"""
频域带限与制造约束联动的集成测试
"""

import numpy as np
import pytest

from core.fft import create_bandlimit_mask, bandlimit_projection
from algorithms.mask_optimizer import (
    RegularizationConfig,
    BandlimitConstraintConfig,
)


class TestBandlimitConstraintConfig:
    """测试 BandlimitConstraintConfig 配置类"""

    def test_default_config(self):
        """测试默认配置"""
        cfg = BandlimitConstraintConfig()
        assert cfg.enable is False
        assert cfg.type == 'lowpass'
        assert cfg.outer_radius == 0.5
        assert cfg.synergistic_regularization is True
        assert cfg.tv_scale_factor == 0.5
        assert cfg.manhattan_scale_factor == 1.0
        assert cfg.apply_bandlimit_before_regularization is True

    def test_config_with_manufacturing(self):
        """测试带制造参数的配置"""
        cfg = BandlimitConstraintConfig(
            enable=True,
            auto_detect=True,
            min_linewidth_nm=45.0,
            pixel_size_nm=5.0,
        )
        assert cfg.enable is True
        assert cfg.auto_detect is True
        assert cfg.min_linewidth_nm == 45.0
        assert cfg.pixel_size_nm == 5.0


class TestComputeCutoffFromManufacturing:
    """测试基于制造参数计算截止频率"""

    def test_cutoff_frequency_calculation(self):
        """测试截止频率计算"""
        cfg = BandlimitConstraintConfig(
            auto_detect=True,
            min_linewidth_nm=45.0,
            pixel_size_nm=5.0,
        )
        cutoff = cfg.compute_cutoff_from_manufacturing((64, 64))
        assert 0.0 < cutoff < 0.5
        assert isinstance(cutoff, float)

    def test_smaller_linewidth_gives_larger_cutoff(self):
        """测试更小的线宽对应更大的截止频率"""
        cfg_large = BandlimitConstraintConfig(
            auto_detect=True,
            min_linewidth_nm=100.0,
            pixel_size_nm=5.0,
        )
        cfg_small = BandlimitConstraintConfig(
            auto_detect=True,
            min_linewidth_nm=45.0,
            pixel_size_nm=5.0,
        )
        cutoff_large_lw = cfg_large.compute_cutoff_from_manufacturing((64, 64))
        cutoff_small_lw = cfg_small.compute_cutoff_from_manufacturing((64, 64))
        assert cutoff_small_lw > cutoff_large_lw

    def test_nyquist_limit(self):
        """测试奈奎斯特频率上限"""
        cfg = BandlimitConstraintConfig(
            auto_detect=True,
            min_linewidth_nm=10.0,
            pixel_size_nm=5.0,
        )
        cutoff = cfg.compute_cutoff_from_manufacturing((64, 64))
        assert cutoff <= 1.0

    def test_auto_detect_false_uses_outer_radius(self):
        """测试 auto_detect=False 时使用 outer_radius"""
        cfg = BandlimitConstraintConfig(
            auto_detect=False,
            outer_radius=0.3,
            min_linewidth_nm=45.0,
            pixel_size_nm=5.0,
        )
        cutoff = cfg.compute_cutoff_from_manufacturing((64, 64))
        assert cutoff == 0.3


class TestRegularizationConfigWithBandlimit:
    """测试 RegularizationConfig 中的 bandlimit 属性"""

    def test_regularization_config_has_bandlimit(self):
        """测试 RegularizationConfig 包含 bandlimit 属性"""
        reg_cfg = RegularizationConfig()
        assert hasattr(reg_cfg, 'bandlimit')
        assert isinstance(reg_cfg.bandlimit, BandlimitConstraintConfig)
        assert reg_cfg.bandlimit.enable is False

    def test_regularization_config_custom_bandlimit(self):
        """测试自定义 bandlimit 配置"""
        bl_cfg = BandlimitConstraintConfig(
            enable=True,
            outer_radius=0.3,
            tv_scale_factor=0.7,
        )
        reg_cfg = RegularizationConfig(bandlimit=bl_cfg)
        assert reg_cfg.bandlimit.enable is True
        assert reg_cfg.bandlimit.outer_radius == 0.3
        assert reg_cfg.bandlimit.tv_scale_factor == 0.7


class TestBandlimitProjectionSynergy:
    """测试带限投影与正则化的协同"""

    def test_bandlimit_projection_basic(self):
        """测试基础带限投影"""
        mask = np.random.rand(64, 64)
        bl_mask = create_bandlimit_mask(
            shape=mask.shape,
            bandlimit_type='lowpass',
            outer_radius=0.3,
        )
        result = bandlimit_projection(mask, bl_mask)
        assert result.shape == mask.shape
        assert np.all(np.isfinite(result))

    def test_bandlimit_projection_idempotent(self):
        """测试带限投影的幂等性"""
        mask = np.random.rand(64, 64)
        bl_mask = create_bandlimit_mask(
            shape=mask.shape,
            bandlimit_type='lowpass',
            outer_radius=0.3,
        )
        result1 = bandlimit_projection(mask, bl_mask)
        result2 = bandlimit_projection(result1, bl_mask)
        np.testing.assert_allclose(result1, result2, atol=1e-10)

    def test_lowpass_reduces_high_freq(self):
        """测试低通滤波器减少高频分量"""
        x = np.linspace(0, 1, 64, endpoint=False)
        y = np.linspace(0, 1, 64, endpoint=False)
        X, Y = np.meshgrid(x, y)
        mask = np.sin(2 * np.pi * 20 * X) + np.sin(2 * np.pi * 5 * Y)

        bl_mask = create_bandlimit_mask(
            shape=mask.shape,
            bandlimit_type='lowpass',
            outer_radius=0.15,
        )
        result = bandlimit_projection(mask, bl_mask)

        orig_energy = np.sum(mask ** 2)
        filtered_energy = np.sum(result ** 2)
        assert filtered_energy < orig_energy


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
