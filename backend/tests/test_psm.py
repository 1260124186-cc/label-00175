# -*- coding: utf-8 -*-
"""
相位偏移掩模 (PSM) 模块测试

测试内容：
1. 各种掩模类型的复数透过率计算
2. 梯度计算的数值验证
3. 与成像系统的集成
4. 端到端梯度验证
"""

import numpy as np
import pytest
from mask_types.psm import (
    MaskType,
    PSMConfig,
    BinaryMask,
    AlternatingPSM,
    AttenuatedPSM,
    ContinuousPhaseMask,
    AmplitudePhaseMask,
    create_mask_model,
    verify_gradient_numerical,
    PSMImagingWrapper,
    PhaseOnlyImagingWrapper,
    AmplitudePhaseImagingWrapper,
    verify_end_to_end_gradient_numerical,
)
from core.imaging import OpticalSystem, PartialCoherentImaging, TechnologyNode, TCCMode


@pytest.fixture
def simple_mask():
    """生成简单的测试掩模"""
    mask = np.zeros((16, 16), dtype=np.float64)
    mask[4:12, 4:12] = 1.0
    return mask


@pytest.fixture
def random_mask():
    """生成随机掩模"""
    rng = np.random.RandomState(42)
    return rng.rand(16, 16).astype(np.float64)


@pytest.fixture
def imaging_model():
    """创建简单的成像模型"""
    optics = OpticalSystem(
        wavelength=193.0,
        na=0.8,
        pixel_size=2.0,
        tcc_mode=TCCMode.KERNEL_2D,
    )
    return PartialCoherentImaging(optics, (16, 16))


# ============================================================================
# 基本功能测试
# ============================================================================

class TestBinaryMask:
    """二值掩模测试"""

    def test_transmission_type(self, simple_mask):
        """测试透过率类型为复数"""
        model = BinaryMask()
        t = model.get_transmission(simple_mask)
        assert t.dtype == np.complex128
        assert t.shape == simple_mask.shape

    def test_transmission_values(self, simple_mask):
        """测试透过率值正确"""
        model = BinaryMask()
        t = model.get_transmission(simple_mask)
        assert np.all(np.imag(t) == 0)
        assert np.allclose(np.real(t), simple_mask)

    def test_amplitude(self, simple_mask):
        """测试振幅计算"""
        model = BinaryMask()
        amp = model.get_amplitude(simple_mask)
        assert np.allclose(amp, simple_mask)

    def test_phase(self, simple_mask):
        """测试相位计算"""
        model = BinaryMask()
        phase = model.get_phase(simple_mask)
        assert np.allclose(phase, 0.0)

    def test_gradient_numerical(self, random_mask):
        """数值验证梯度"""
        model = BinaryMask()
        result = verify_gradient_numerical(model, random_mask)
        assert result['correct'], f"梯度错误: max_rel_error={result['max_rel_error']}"


class TestAlternatingPSM:
    """交替 PSM 测试"""

    def test_transmission_type(self, simple_mask):
        """测试透过率类型为复数"""
        model = AlternatingPSM()
        t = model.get_transmission(simple_mask)
        assert t.dtype == np.complex128

    def test_amplitude_unity(self, simple_mask):
        """测试振幅恒为 1"""
        model = AlternatingPSM()
        amp = model.get_amplitude(simple_mask)
        assert np.allclose(amp, 1.0)

    def test_phase_zero_pi(self, simple_mask):
        """测试相位为 0 或 π"""
        model = AlternatingPSM()
        phase = model.get_phase(simple_mask)
        assert np.allclose(phase[simple_mask == 0], 0.0)
        assert np.allclose(phase[simple_mask == 1], np.pi)

    def test_custom_phase_shift(self, simple_mask):
        """测试自定义相移量"""
        config = PSMConfig(phase_shift=np.pi / 2)
        model = AlternatingPSM(config)
        phase = model.get_phase(simple_mask)
        assert np.allclose(phase[simple_mask == 1], np.pi / 2)

    def test_gradient_numerical(self, random_mask):
        """数值验证梯度"""
        model = AlternatingPSM()
        result = verify_gradient_numerical(model, random_mask, eps=1e-7)
        assert result['correct'], f"梯度错误: max_rel_error={result['max_rel_error']}"


class TestAttenuatedPSM:
    """衰减式 PSM 测试"""

    def test_transmission_type(self, simple_mask):
        """测试透过率类型为复数"""
        model = AttenuatedPSM()
        t = model.get_transmission(simple_mask)
        assert t.dtype == np.complex128

    def test_clear_region(self, simple_mask):
        """测试透明区透过率为 1"""
        model = AttenuatedPSM()
        t = model.get_transmission(simple_mask)
        assert np.allclose(t[simple_mask == 1], 1.0 + 0j)

    def test_shifter_region(self, simple_mask):
        """测试相移区透过率"""
        model = AttenuatedPSM()
        t = model.get_transmission(simple_mask)
        t_shifter = t[simple_mask == 0]
        assert np.allclose(np.abs(t_shifter), 0.06)
        assert np.allclose(np.angle(t_shifter), np.pi)

    def test_custom_attenuation(self, simple_mask):
        """测试自定义衰减系数"""
        config = PSMConfig(attenuation=0.1, phase_shift=np.pi)
        model = AttenuatedPSM(config)
        t = model.get_transmission(simple_mask)
        t_shifter = t[simple_mask == 0]
        assert np.allclose(np.abs(t_shifter), 0.1)

    def test_gradient_numerical(self, random_mask):
        """数值验证梯度"""
        model = AttenuatedPSM()
        result = verify_gradient_numerical(model, random_mask, eps=1e-7)
        assert result['correct'], f"梯度错误: max_rel_error={result['max_rel_error']}"


class TestContinuousPhaseMask:
    """连续相位掩模测试"""

    def test_transmission(self):
        """测试连续相位透过率"""
        model = ContinuousPhaseMask()
        phase = np.linspace(-np.pi, np.pi, 16).reshape(4, 4)
        t = model.get_transmission(phase)
        assert np.allclose(np.abs(t), 1.0)
        assert np.allclose(np.angle(t), phase)

    def test_gradient_wrt_phase(self):
        """测试对相位的梯度"""
        model = ContinuousPhaseMask()
        rng = np.random.RandomState(42)
        phase = rng.rand(8, 8) * 2 * np.pi - np.pi
        grad_trans_real = rng.randn(8, 8)
        grad_trans_imag = rng.randn(8, 8)
        grad_transmission = grad_trans_real + 1j * grad_trans_imag

        grad_phase = model.gradient_wrt_phase(phase, grad_transmission)

        eps = 1e-7
        grad_num = np.zeros_like(phase)
        for i in range(8):
            for j in range(8):
                p_plus = phase.copy()
                p_plus[i, j] += eps
                t_plus = model.get_transmission(p_plus)
                p_minus = phase.copy()
                p_minus[i, j] -= eps
                t_minus = model.get_transmission(p_minus)

                loss_plus = np.sum(grad_trans_real * np.real(t_plus) + grad_trans_imag * np.imag(t_plus))
                loss_minus = np.sum(grad_trans_real * np.real(t_minus) + grad_trans_imag * np.imag(t_minus))
                grad_num[i, j] = (loss_plus - loss_minus) / (2 * eps)

        rel_error = np.abs(grad_phase - grad_num) / (np.abs(grad_phase) + 1e-10)
        assert np.max(rel_error) < 1e-5, f"相位梯度错误: max_rel_error={np.max(rel_error)}"


class TestAmplitudePhaseMask:
    """幅度-相位联合掩模测试"""

    def test_transmission(self):
        """测试复透过率计算"""
        model = AmplitudePhaseMask()
        amplitude = np.array([0.5, 1.0]).reshape(1, 2)
        phase = np.array([0.0, np.pi / 2]).reshape(1, 2)
        t = model.get_transmission(amplitude, phase)
        assert np.allclose(t[0, 0], 0.5 + 0j)
        assert np.allclose(t[0, 1], 1j)

    def test_amplitude_gradient(self):
        """测试幅度梯度"""
        model = AmplitudePhaseMask()
        rng = np.random.RandomState(42)
        amplitude = rng.rand(8, 8)
        phase = rng.rand(8, 8) * 2 * np.pi
        grad_trans_real = rng.randn(8, 8)
        grad_trans_imag = rng.randn(8, 8)
        grad_transmission = grad_trans_real + 1j * grad_trans_imag

        grad_amp = model.gradient_wrt_amplitude(amplitude, phase, grad_transmission)

        eps = 1e-7
        grad_num = np.zeros_like(amplitude)
        for i in range(8):
            for j in range(8):
                a_plus = amplitude.copy()
                a_plus[i, j] += eps
                t_plus = model.get_transmission(a_plus, phase)
                a_minus = amplitude.copy()
                a_minus[i, j] -= eps
                t_minus = model.get_transmission(a_minus, phase)

                loss_plus = np.sum(grad_trans_real * np.real(t_plus) + grad_trans_imag * np.imag(t_plus))
                loss_minus = np.sum(grad_trans_real * np.real(t_minus) + grad_trans_imag * np.imag(t_minus))
                grad_num[i, j] = (loss_plus - loss_minus) / (2 * eps)

        rel_error = np.abs(grad_amp - grad_num) / (np.abs(grad_amp) + 1e-10)
        assert np.max(rel_error) < 1e-5, f"幅度梯度错误: max_rel_error={np.max(rel_error)}"

    def test_phase_gradient(self):
        """测试相位梯度"""
        model = AmplitudePhaseMask()
        rng = np.random.RandomState(42)
        amplitude = rng.rand(8, 8)
        phase = rng.rand(8, 8) * 2 * np.pi
        grad_trans_real = rng.randn(8, 8)
        grad_trans_imag = rng.randn(8, 8)
        grad_transmission = grad_trans_real + 1j * grad_trans_imag

        grad_phase = model.gradient_wrt_phase(amplitude, phase, grad_transmission)

        eps = 1e-7
        grad_num = np.zeros_like(phase)
        for i in range(8):
            for j in range(8):
                p_plus = phase.copy()
                p_plus[i, j] += eps
                t_plus = model.get_transmission(amplitude, p_plus)
                p_minus = phase.copy()
                p_minus[i, j] -= eps
                t_minus = model.get_transmission(amplitude, p_minus)

                loss_plus = np.sum(grad_trans_real * np.real(t_plus) + grad_trans_imag * np.imag(t_plus))
                loss_minus = np.sum(grad_trans_real * np.real(t_minus) + grad_trans_imag * np.imag(t_minus))
                grad_num[i, j] = (loss_plus - loss_minus) / (2 * eps)

        rel_error = np.abs(grad_phase - grad_num) / (np.abs(grad_phase) + 1e-10)
        assert np.max(rel_error) < 1e-5, f"相位梯度错误: max_rel_error={np.max(rel_error)}"


class TestFactoryFunction:
    """工厂函数测试"""

    def test_create_binary(self):
        """测试创建二值掩模"""
        model = create_mask_model(MaskType.BINARY)
        assert isinstance(model, BinaryMask)

    def test_create_alternating(self):
        """测试创建交替 PSM"""
        model = create_mask_model(MaskType.ALTERNATING_PSM)
        assert isinstance(model, AlternatingPSM)

    def test_create_attenuated(self):
        """测试创建衰减式 PSM"""
        model = create_mask_model(MaskType.ATTENUATED_PSM)
        assert isinstance(model, AttenuatedPSM)


# ============================================================================
# 成像集成测试
# ============================================================================

class TestImagingIntegration:
    """成像系统集成测试"""

    def test_binary_consistency(self, imaging_model, simple_mask):
        """测试二值掩模与原方法结果一致"""
        aerial_orig = imaging_model.compute_aerial_image(simple_mask)
        aerial_complex = imaging_model.compute_aerial_image_complex(simple_mask.astype(np.complex128))
        assert np.allclose(aerial_orig, aerial_complex, atol=1e-10)

    def test_complex_gradient_shape(self, imaging_model, simple_mask):
        """测试复梯度形状正确"""
        intensity_grad = np.ones_like(simple_mask)
        grad = imaging_model.compute_complex_gradient(
            simple_mask.astype(np.complex128), intensity_grad
        )
        assert grad.shape == simple_mask.shape
        assert grad.dtype == np.complex128

    def test_real_gradient_consistency(self, imaging_model, simple_mask):
        """测试实梯度与复梯度实部的一致性"""
        grad_orig = imaging_model.compute_image_gradient(simple_mask)
        intensity_grad = 2.0 * (np.ones_like(simple_mask) * 0.5 - simple_mask) / simple_mask.size

        grad_complex = imaging_model.compute_complex_gradient(
            simple_mask.astype(np.complex128), intensity_grad
        )
        grad_real = np.real(grad_complex)

        ratio = grad_orig / (grad_real + 1e-10)
        assert np.allclose(ratio, ratio[0, 0], atol=1e-3), "实梯度比例不一致"


# ============================================================================
# 端到端测试
# ============================================================================

class TestPSMImagingWrapper:
    """PSM 成像封装测试"""

    def test_binary_wrapper(self, imaging_model, simple_mask):
        """测试二值掩模封装"""
        model = BinaryMask()
        wrapper = PSMImagingWrapper(imaging_model, model)
        aerial = wrapper.compute_aerial_image(simple_mask)
        assert aerial.shape == simple_mask.shape
        assert np.all(aerial >= 0) and np.all(aerial <= 1)

    def test_alternating_wrapper(self, imaging_model, simple_mask):
        """测试交替 PSM 封装"""
        model = AlternatingPSM()
        wrapper = PSMImagingWrapper(imaging_model, model)
        aerial = wrapper.compute_aerial_image(simple_mask)
        assert aerial.shape == simple_mask.shape
        assert np.all(aerial >= 0) and np.all(aerial <= 1)

    def test_attenuated_wrapper(self, imaging_model, simple_mask):
        """测试衰减式 PSM 封装"""
        model = AttenuatedPSM()
        wrapper = PSMImagingWrapper(imaging_model, model)
        aerial = wrapper.compute_aerial_image(simple_mask)
        assert aerial.shape == simple_mask.shape
        assert np.all(aerial >= 0) and np.all(aerial <= 1)

    def test_end_to_end_gradient_binary(self, imaging_model, random_mask):
        """端到端梯度验证（二值掩模）"""
        model = BinaryMask()
        wrapper = PSMImagingWrapper(imaging_model, model)
        target = imaging_model.compute_aerial_image(random_mask)

        result = verify_end_to_end_gradient_numerical(
            wrapper, random_mask, target, eps=1e-5, metric='mse'
        )
        assert result['correct'], f"二值掩模端到端梯度错误: max_rel_error={result['max_rel_error']}"

    def test_end_to_end_gradient_alternating(self, imaging_model, random_mask):
        """端到端梯度验证（交替 PSM）"""
        model = AlternatingPSM()
        wrapper = PSMImagingWrapper(imaging_model, model)
        target = wrapper.compute_aerial_image(random_mask)

        result = verify_end_to_end_gradient_numerical(
            wrapper, random_mask, target, eps=1e-5, metric='mse'
        )
        assert result['correct'], f"交替PSM端到端梯度错误: max_rel_error={result['max_rel_error']}"

    def test_end_to_end_gradient_attenuated(self, imaging_model, random_mask):
        """端到端梯度验证（衰减式 PSM）"""
        model = AttenuatedPSM()
        wrapper = PSMImagingWrapper(imaging_model, model)
        target = wrapper.compute_aerial_image(random_mask)

        result = verify_end_to_end_gradient_numerical(
            wrapper, random_mask, target, eps=1e-5, metric='mse'
        )
        assert result['correct'], f"衰减式PSM端到端梯度错误: max_rel_error={result['max_rel_error']}"


class TestPhaseOnlyWrapper:
    """纯相位优化封装测试"""

    def test_compute_aerial(self, imaging_model):
        """测试纯相位成像"""
        wrapper = PhaseOnlyImagingWrapper(imaging_model)
        phase = np.zeros((16, 16))
        aerial = wrapper.compute_aerial_image(phase)
        assert aerial.shape == (16, 16)

    def test_gradient_numerical(self, imaging_model):
        """数值验证纯相位梯度"""
        wrapper = PhaseOnlyImagingWrapper(imaging_model)
        rng = np.random.RandomState(42)
        phase = rng.rand(16, 16) * np.pi * 0.5 - np.pi * 0.25
        target = wrapper.compute_aerial_image(phase)

        eps = 1e-6
        grad_num = np.zeros_like(phase)
        for i in range(16):
            for j in range(16):
                p_plus = phase.copy()
                p_plus[i, j] += eps
                aerial_plus = wrapper.compute_aerial_image(p_plus)

                p_minus = phase.copy()
                p_minus[i, j] -= eps
                aerial_minus = wrapper.compute_aerial_image(p_minus)

                loss_plus = np.mean((aerial_plus - target) ** 2)
                loss_minus = np.mean((aerial_minus - target) ** 2)
                grad_num[i, j] = (loss_plus - loss_minus) / (2 * eps)

        intensity_grad = 2.0 * (wrapper.compute_aerial_image(phase) - target) / (16 * 16)
        grad_analytical = wrapper.compute_gradient(phase, intensity_grad)

        rel_error = np.abs(grad_analytical - grad_num) / (np.abs(grad_analytical) + 1e-10)
        assert np.max(rel_error) < 1e-3, f"纯相位梯度错误: max_rel_error={np.max(rel_error)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
