# -*- coding: utf-8 -*-
"""
新增正则化项的单元测试
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend')

from core.metrics import (
    manhattan_distance_penalty, manhattan_distance_penalty_gradient,
    binary_entropy_penalty, binary_entropy_penalty_gradient,
    total_variation_anisotropic, total_variation_isotropic,
    total_variation_isotropic_gradient,
    edge_placement_error, soft_edge_placement_error,
    soft_edge_placement_error_gradient,
    min_feature_size_morphology, min_feature_size_frequency,
    soft_min_feature_size_morphology, soft_min_feature_size_morphology_gradient,
    min_feature_size_frequency_gradient,
    min_feature_size_combined, min_feature_size_combined_gradient
)


class TestManhattanDistancePenalty:
    """曼哈顿距离惩罚测试"""

    def test_binary_image_penalty_zero(self):
        """二值图像的惩罚应为0"""
        img = np.array([[0, 1], [1, 0]], dtype=np.float64)
        penalty = manhattan_distance_penalty(img)
        assert abs(penalty) < 1e-10

    def test_half_image_penalty_max(self):
        """全0.5图像的惩罚应为1"""
        img = 0.5 * np.ones((4, 4), dtype=np.float64)
        penalty = manhattan_distance_penalty(img)
        assert abs(penalty - 1.0) < 1e-10

    def test_penalty_range(self):
        """惩罚值应在[0, 1]范围内"""
        np.random.seed(42)
        img = np.random.random((16, 16))
        penalty = manhattan_distance_penalty(img)
        assert 0 <= penalty <= 1.0

    def test_gradient_shape(self):
        """梯度形状应与输入一致"""
        img = np.random.random((8, 8))
        grad = manhattan_distance_penalty_gradient(img)
        assert grad.shape == img.shape

    def test_gradient_sign(self):
        """测试梯度符号：m > 0.5时梯度为负，m < 0.5时梯度为正"""
        img_above = np.array([[0.6, 0.7], [0.8, 0.9]])
        grad_above = manhattan_distance_penalty_gradient(img_above)
        assert np.all(grad_above <= 0)

        img_below = np.array([[0.1, 0.2], [0.3, 0.4]])
        grad_below = manhattan_distance_penalty_gradient(img_below)
        assert np.all(grad_below >= 0)


class TestBinaryEntropyPenalty:
    """二值熵惩罚测试"""

    def test_binary_image_entropy_zero(self):
        """二值图像的熵应为0"""
        img = np.array([[0, 1], [1, 0]], dtype=np.float64)
        penalty = binary_entropy_penalty(img)
        assert penalty < 1e-5

    def test_half_image_entropy_max(self):
        """全0.5图像的熵应为log(2)"""
        img = 0.5 * np.ones((4, 4), dtype=np.float64)
        penalty = binary_entropy_penalty(img)
        assert abs(penalty - np.log(2.0)) < 1e-5

    def test_gradient_shape(self):
        """梯度形状应与输入一致"""
        img = np.random.random((8, 8))
        grad = binary_entropy_penalty_gradient(img)
        assert grad.shape == img.shape


class TestTotalVariationAdvanced:
    """高级总变分测试"""

    def test_anisotropic_tv_vs_basic(self):
        """各向异性TV应与basic TV相同"""
        img = np.random.random((16, 16))
        tv1 = total_variation_anisotropic(img)
        from core.metrics import total_variation
        tv2 = total_variation(img)
        assert abs(tv1 - tv2) < 1e-10

    def test_isotropic_tv_positive(self):
        """各向同性TV应为正值"""
        img = np.random.random((16, 16))
        tv = total_variation_isotropic(img)
        assert tv > 0

    def test_constant_image_tv_zero(self):
        """常数图像的TV应接近0（考虑eps数值稳定性）"""
        img = np.ones((8, 8)) * 0.5
        tv_iso = total_variation_isotropic(img)
        assert abs(tv_iso) < 1e-5

    def test_isotropic_gradient_shape(self):
        """各向同性TV梯度形状应正确"""
        img = np.random.random((8, 8))
        grad = total_variation_isotropic_gradient(img)
        assert grad.shape == img.shape


class TestEdgePlacementError:
    """边缘放置误差测试"""

    def test_identical_images_epe_zero(self):
        """相同图像的EPE应接近0"""
        target = np.zeros((16, 16))
        target[4:12, 4:12] = 1.0
        pred = target.copy()
        epe = soft_edge_placement_error(pred, target, sigma=0.5)
        assert epe < 0.01

    def test_shifted_image_epe_positive(self):
        """平移图像的EPE应为正值"""
        target = np.zeros((16, 16))
        target[4:12, 4:12] = 1.0
        pred = np.zeros((16, 16))
        pred[5:13, 5:13] = 1.0
        epe = soft_edge_placement_error(pred, target, sigma=0.5)
        assert epe > 0

    def test_soft_epe_gradient_shape(self):
        """软EPE梯度形状应正确"""
        target = np.zeros((8, 8))
        target[2:6, 2:6] = 1.0
        pred = np.random.random((8, 8)) * 0.1 + target * 0.9
        grad = soft_edge_placement_error_gradient(pred, target, sigma=0.5)
        assert grad.shape == pred.shape
        assert np.all(np.isfinite(grad))


class TestMinFeatureSize:
    """最小特征尺寸约束测试"""

    def test_frequency_penalty_positive(self):
        """频域小特征惩罚应为正值"""
        img = np.random.random((16, 16))
        penalty = min_feature_size_frequency(img, min_size=3)
        assert penalty >= 0

    def test_morphology_penalty_range(self):
        """形态学惩罚应在合理范围内"""
        target = np.zeros((16, 16))
        target[4:12, 4:12] = 1.0
        penalty = soft_min_feature_size_morphology(target, min_size=3)
        assert 0 <= penalty <= 1.0

    def test_small_feature_has_higher_penalty(self):
        """小特征应有更高的惩罚"""
        big_feature = np.zeros((16, 16))
        big_feature[4:12, 4:12] = 1.0
        penalty_big = min_feature_size_frequency(big_feature, min_size=5)

        small_feature = np.zeros((16, 16))
        small_feature[7:9, 7:9] = 1.0
        penalty_small = min_feature_size_frequency(small_feature, min_size=5)

        assert penalty_small > penalty_big

    def test_combined_penalty(self):
        """联合惩罚应在形态学和频域惩罚之间"""
        img = np.random.random((16, 16))
        penalty_morph = soft_min_feature_size_morphology(img, min_size=3)
        penalty_freq = min_feature_size_frequency(img, min_size=3)
        penalty_combined = min_feature_size_combined(img, min_size=3, alpha=0.5)

        min_p = min(penalty_morph, penalty_freq)
        max_p = max(penalty_morph, penalty_freq)
        assert min_p - 1e-10 <= penalty_combined <= max_p + 1e-10

    def test_gradient_functions(self):
        """梯度函数应返回正确形状的有限值"""
        img = np.random.random((8, 8))

        grad_morph = soft_min_feature_size_morphology_gradient(img, min_size=3)
        assert grad_morph.shape == img.shape
        assert np.all(np.isfinite(grad_morph))

        grad_freq = min_feature_size_frequency_gradient(img, min_size=3)
        assert grad_freq.shape == img.shape
        assert np.all(np.isfinite(grad_freq))

        grad_combined = min_feature_size_combined_gradient(img, min_size=3, alpha=0.5)
        assert grad_combined.shape == img.shape
        assert np.all(np.isfinite(grad_combined))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
