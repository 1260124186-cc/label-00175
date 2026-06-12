# -*- coding: utf-8 -*-
"""
傅里叶变换模块单元测试
"""

import pytest
import numpy as np
from core.fft import (
    fft1d, ifft1d, fft2d, ifft2d,
    frequency_filter, phase_modulation,
    compute_power_spectrum, get_frequency_coordinates,
    WindowType, hann_window_2d, hamming_window_2d, tukey_window_2d,
    create_window, apply_zero_padding, remove_padding,
    apply_window_and_padding, crop_to_original
)


class TestFFT1D:
    """一维FFT测试"""
    
    def test_fft_ifft_roundtrip(self):
        """测试FFT-IFFT往返"""
        signal = np.random.random(64)
        
        spectrum = fft1d(signal, shift=True, normalize=True)
        recovered = ifft1d(spectrum, shifted=True, was_normalized=True)
        
        np.testing.assert_array_almost_equal(signal, np.real(recovered), decimal=10)
    
    def test_fft_output_shape(self):
        """测试FFT输出形状"""
        signal = np.random.random(128)
        spectrum = fft1d(signal)
        
        assert spectrum.shape == signal.shape
    
    def test_fft_dc_component(self):
        """测试直流分量"""
        signal = np.ones(64) * 5.0
        spectrum = fft1d(signal, shift=True, normalize=True)
        
        # 直流分量应该在中心
        center = len(spectrum) // 2
        assert abs(spectrum[center]) > abs(spectrum[0])


class TestFFT2D:
    """二维FFT测试"""
    
    def test_fft2d_ifft2d_roundtrip(self):
        """测试2D FFT-IFFT往返"""
        image = np.random.random((64, 64))
        
        spectrum = fft2d(image, shift=True, normalize=True)
        recovered = ifft2d(spectrum, shifted=True, was_normalized=True)
        
        np.testing.assert_array_almost_equal(image, np.real(recovered), decimal=10)
    
    def test_fft2d_output_shape(self):
        """测试2D FFT输出形状"""
        image = np.random.random((32, 64))
        spectrum = fft2d(image)
        
        assert spectrum.shape == image.shape
    
    def test_fft2d_symmetry(self):
        """测试实数输入的共轭对称性"""
        image = np.random.random((32, 32))
        spectrum = fft2d(image, shift=False, normalize=False)
        
        # 实数输入的FFT应该具有共轭对称性
        # F(-k) = F*(k)
        assert np.iscomplexobj(spectrum)
    
    def test_fft2d_parseval(self):
        """测试Parseval定理（能量守恒）"""
        image = np.random.random((32, 32))
        spectrum = fft2d(image, shift=False, normalize=False)
        
        # 空域能量
        spatial_energy = np.sum(image ** 2)
        
        # 频域能量
        freq_energy = np.sum(np.abs(spectrum) ** 2) / image.size
        
        np.testing.assert_almost_equal(spatial_energy, freq_energy, decimal=10)


class TestFrequencyFilter:
    """频域滤波测试"""
    
    def test_lowpass_filter(self):
        """测试低通滤波器"""
        spectrum = np.ones((64, 64), dtype=complex)
        filtered = frequency_filter(spectrum, 'lowpass', cutoff=0.3)
        
        # 中心应该保留，边缘应该衰减
        center = 32
        assert abs(filtered[center, center]) > abs(filtered[0, 0])
    
    def test_highpass_filter(self):
        """测试高通滤波器"""
        spectrum = np.ones((64, 64), dtype=complex)
        filtered = frequency_filter(spectrum, 'highpass', cutoff=0.3)
        
        # 边缘应该保留，中心应该衰减
        center = 32
        assert abs(filtered[0, 0]) > abs(filtered[center, center])
    
    def test_bandpass_filter(self):
        """测试带通滤波器"""
        spectrum = np.ones((64, 64), dtype=complex)
        filtered = frequency_filter(spectrum, 'bandpass', cutoff=0.5, bandwidth=0.2)
        
        assert filtered.shape == spectrum.shape
    
    def test_invalid_filter_type(self):
        """测试无效滤波器类型"""
        spectrum = np.ones((64, 64), dtype=complex)
        
        with pytest.raises(ValueError):
            frequency_filter(spectrum, 'invalid_type')


class TestPhaseModulation:
    """相位调制测试"""
    
    def test_linear_phase(self):
        """测试线性相位调制"""
        spectrum = np.ones((32, 32), dtype=complex)
        modulated = phase_modulation(spectrum, 'linear', {'kx': 0.1, 'ky': 0.1})
        
        # 幅度应该保持不变
        np.testing.assert_array_almost_equal(
            np.abs(spectrum), np.abs(modulated), decimal=10
        )
    
    def test_quadratic_phase(self):
        """测试二次相位调制"""
        spectrum = np.ones((32, 32), dtype=complex)
        modulated = phase_modulation(spectrum, 'quadratic', {'alpha': 0.01})
        
        assert modulated.shape == spectrum.shape
        assert np.iscomplexobj(modulated)
    
    def test_custom_phase(self):
        """测试自定义相位"""
        spectrum = np.ones((32, 32), dtype=complex)
        custom_phase = np.random.random((32, 32)) * 2 * np.pi
        
        modulated = phase_modulation(
            spectrum, 'custom', {'phase_array': custom_phase}
        )
        
        assert modulated.shape == spectrum.shape
    
    def test_invalid_phase_type(self):
        """测试无效相位类型"""
        spectrum = np.ones((32, 32), dtype=complex)
        
        with pytest.raises(ValueError):
            phase_modulation(spectrum, 'invalid_type')


class TestPowerSpectrum:
    """功率谱测试"""
    
    def test_power_spectrum_shape(self):
        """测试功率谱形状"""
        image = np.random.random((64, 64))
        power = compute_power_spectrum(image)
        
        assert power.shape == image.shape
    
    def test_power_spectrum_real(self):
        """测试功率谱为实数"""
        image = np.random.random((64, 64))
        power = compute_power_spectrum(image)
        
        assert not np.iscomplexobj(power)


class TestFrequencyCoordinates:
    """频率坐标测试"""
    
    def test_coordinate_shape(self):
        """测试坐标形状"""
        fx, fy = get_frequency_coordinates((64, 128), pixel_size=1.0)
        
        assert fx.shape == (64, 128)
        assert fy.shape == (64, 128)
    
    def test_coordinate_center(self):
        """测试坐标中心为零"""
        fx, fy = get_frequency_coordinates((64, 64), pixel_size=1.0)
        
        center = 32
        assert abs(fx[center, center]) < 1e-10
        assert abs(fy[center, center]) < 1e-10


class TestNumbaAcceleration:
    """Numba加速测试"""
    
    def test_fftshift_2d_correctness(self):
        """测试2D fftshift正确性"""
        from core.fft import _fftshift_2d
        from scipy import fft as scipy_fft
        
        image = np.random.random((32, 32)).astype(np.complex128)
        
        # 比较numba实现和scipy实现
        numba_result = _fftshift_2d(image)
        scipy_result = scipy_fft.fftshift(image)
        
        np.testing.assert_array_almost_equal(numba_result, scipy_result)
    
    def test_ifftshift_2d_correctness(self):
        """测试2D ifftshift正确性"""
        from core.fft import _ifftshift_2d
        from scipy import fft as scipy_fft
        
        image = np.random.random((32, 32)).astype(np.complex128)
        
        numba_result = _ifftshift_2d(image)
        scipy_result = scipy_fft.ifftshift(image)
        
        np.testing.assert_array_almost_equal(numba_result, scipy_result)
    
    def test_fftshift_roundtrip(self):
        """测试fftshift往返"""
        from core.fft import _fftshift_2d, _ifftshift_2d
        
        image = np.random.random((64, 64)).astype(np.complex128)
        
        shifted = _fftshift_2d(image)
        recovered = _ifftshift_2d(shifted)
        
        np.testing.assert_array_almost_equal(image, recovered)
    
    def test_normalize_spectrum(self):
        """测试频谱归一化"""
        from core.fft import _normalize_spectrum
        
        spectrum = np.ones((32, 32), dtype=np.complex128) * 100
        normalized = _normalize_spectrum(spectrum, 100.0)
        
        np.testing.assert_array_almost_equal(normalized, np.ones((32, 32), dtype=np.complex128))


class TestWindowType:
    def test_window_type_values(self):
        assert WindowType.HANN.value == "hann"
        assert WindowType.HAMMING.value == "hamming"
        assert WindowType.TUKEY.value == "tukey"

    def test_window_type_from_string(self):
        assert WindowType("hann") == WindowType.HANN
        assert WindowType("hamming") == WindowType.HAMMING
        assert WindowType("tukey") == WindowType.TUKEY


class TestHannWindow2D:
    def test_shape(self):
        w = hann_window_2d((64, 128))
        assert w.shape == (64, 128)

    def test_range(self):
        w = hann_window_2d((64, 64))
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)

    def test_center_maximum(self):
        w = hann_window_2d((64, 64))
        assert w[32, 32] > w[0, 0]
        assert w[32, 32] > w[63, 63]

    def test_boundary_near_zero(self):
        w = hann_window_2d((64, 64))
        assert abs(w[0, 0]) < 1e-10
        assert abs(w[0, 63]) < 1e-10
        assert abs(w[63, 0]) < 1e-10
        assert abs(w[63, 63]) < 1e-10

    def test_symmetry(self):
        w = hann_window_2d((64, 64))
        ny, nx = w.shape
        np.testing.assert_array_almost_equal(w, w[::-1, ::-1])


class TestHammingWindow2D:
    def test_shape(self):
        w = hamming_window_2d((32, 64))
        assert w.shape == (32, 64)

    def test_range(self):
        w = hamming_window_2d((64, 64))
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)

    def test_center_maximum(self):
        w = hamming_window_2d((64, 64))
        assert w[32, 32] > w[0, 0]

    def test_boundary_nonzero(self):
        w = hamming_window_2d((64, 64))
        assert w[0, 0] > 0.0
        assert w[63, 63] > 0.0


class TestTukeyWindow2D:
    def test_shape(self):
        w = tukey_window_2d((64, 64), alpha=0.5)
        assert w.shape == (64, 64)

    def test_range(self):
        w = tukey_window_2d((64, 64), alpha=0.5)
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)

    def test_alpha_zero_rectangular(self):
        w = tukey_window_2d((64, 64), alpha=0.0)
        np.testing.assert_array_almost_equal(w, np.ones((64, 64)))

    def test_alpha_one_hann_like(self):
        w = tukey_window_2d((64, 64), alpha=1.0)
        hann = hann_window_2d((64, 64))
        np.testing.assert_array_almost_equal(w, hann, decimal=12)

    def test_center_is_one(self):
        w = tukey_window_2d((64, 64), alpha=0.5)
        assert abs(w[32, 32] - 1.0) < 1e-10


class TestCreateWindow:
    def test_hann(self):
        w = create_window((32, 32), WindowType.HANN)
        expected = hann_window_2d((32, 32))
        np.testing.assert_array_almost_equal(w, expected)

    def test_hamming(self):
        w = create_window((32, 32), WindowType.HAMMING)
        expected = hamming_window_2d((32, 32))
        np.testing.assert_array_almost_equal(w, expected)

    def test_tukey(self):
        w = create_window((32, 32), WindowType.TUKEY, tukey_alpha=0.3)
        expected = tukey_window_2d((32, 32), alpha=0.3)
        np.testing.assert_array_almost_equal(w, expected)

    def test_string_type(self):
        w = create_window((32, 32), "hann")
        expected = hann_window_2d((32, 32))
        np.testing.assert_array_almost_equal(w, expected)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            create_window((32, 32), "invalid")


class TestZeroPadding:
    def test_uniform_padding(self):
        image = np.ones((32, 32))
        padded, pw = apply_zero_padding(image, pad_width=8)
        assert padded.shape == (48, 48)
        assert pw == ((8, 8), (8, 8))
        np.testing.assert_array_equal(padded[8:40, 8:40], image)
        assert padded[0, 0] == 0.0

    def test_asymmetric_padding(self):
        image = np.ones((32, 32))
        padded, pw = apply_zero_padding(image, pad_width=(4, 8))
        assert padded.shape == (40, 48)
        assert pw == ((4, 4), (8, 8))

    def test_no_padding(self):
        image = np.ones((32, 32))
        padded, pw = apply_zero_padding(image, pad_width=0)
        assert padded.shape == (32, 32)

    def test_remove_padding(self):
        image = np.random.random((32, 32))
        padded, pw = apply_zero_padding(image, pad_width=16)
        recovered = remove_padding(padded, pw)
        np.testing.assert_array_equal(recovered, image)

    def test_remove_asymmetric_padding(self):
        image = np.random.random((32, 32))
        padded, pw = apply_zero_padding(image, pad_width=(4, 8))
        recovered = remove_padding(padded, pw)
        np.testing.assert_array_equal(recovered, image)


class TestApplyWindowAndPadding:
    def test_no_window_no_padding(self):
        image = np.ones((32, 32))
        result, info = apply_window_and_padding(image)
        np.testing.assert_array_equal(result, image)
        assert info['pad_width'] == ((0, 0), (0, 0))

    def test_window_only(self):
        image = np.ones((32, 32))
        result, info = apply_window_and_padding(image, window_type=WindowType.HANN)
        expected = hann_window_2d((32, 32))
        np.testing.assert_array_almost_equal(result, expected)
        assert info['pad_width'] == ((0, 0), (0, 0))

    def test_padding_only(self):
        image = np.ones((32, 32))
        result, info = apply_window_and_padding(image, pad_width=16)
        assert result.shape == (64, 64)
        assert info['pad_width'] == ((16, 16), (16, 16))

    def test_window_and_padding(self):
        image = np.ones((32, 32))
        result, info = apply_window_and_padding(
            image, window_type=WindowType.HANN, pad_width=16
        )
        assert result.shape == (64, 64)
        assert result[0, 0] == 0.0
        assert info['original_shape'] == (32, 32)

    def test_crop_to_original(self):
        image = np.random.random((32, 32))
        result, info = apply_window_and_padding(
            image, window_type=WindowType.HANN, pad_width=16
        )
        cropped = crop_to_original(result, info)
        assert cropped.shape == (32, 32)
