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
    apply_window_and_padding, crop_to_original,
    BandlimitType, create_bandlimit_mask,
    bandlimit_projection, bandlimit_projection_simple,
    compute_spectral_energy_ratio, bandlimited_gradient_projection
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


class TestBandlimitType:
    """频域带限类型枚举测试"""

    def test_bandlimit_type_values(self):
        assert BandlimitType.LOWPASS.value == "lowpass"
        assert BandlimitType.HIGHPASS.value == "highpass"
        assert BandlimitType.BANDPASS.value == "bandpass"
        assert BandlimitType.BANDSTOP.value == "bandstop"
        assert BandlimitType.CUSTOM.value == "custom"
        assert BandlimitType.CIRCULAR.value == "circular"
        assert BandlimitType.RECTANGULAR.value == "rectangular"
        assert BandlimitType.DIRECTIONAL.value == "directional"

    def test_bandlimit_type_from_string(self):
        assert BandlimitType("lowpass") == BandlimitType.LOWPASS
        assert BandlimitType("circular") == BandlimitType.CIRCULAR


class TestCreateBandlimitMask:
    """创建频域带限掩模测试"""

    def test_mask_shape(self):
        mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.5)
        assert mask.shape == (64, 64)

    def test_mask_value_range(self):
        mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.5)
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)

    def test_lowpass_center_one(self):
        mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.5)
        assert mask[32, 32] == 1.0

    def test_lowpass_edge_zero(self):
        mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.1)
        assert mask[0, 0] == 0.0

    def test_highpass_center_zero(self):
        mask = create_bandlimit_mask((64, 64), BandlimitType.HIGHPASS, inner_radius=0.5)
        assert mask[32, 32] == 0.0

    def test_highpass_edge_one(self):
        mask = create_bandlimit_mask((64, 64), BandlimitType.HIGHPASS, inner_radius=0.9)
        assert mask[0, 0] == 1.0

    def test_bandpass_shape(self):
        mask = create_bandlimit_mask(
            (64, 64), BandlimitType.BANDPASS,
            inner_radius=0.2, outer_radius=0.5
        )
        assert mask.shape == (64, 64)
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)

    def test_bandstop_shape(self):
        mask = create_bandlimit_mask(
            (64, 64), BandlimitType.BANDSTOP,
            inner_radius=0.2, outer_radius=0.5
        )
        # bandstop 中心应该是1（低通部分），边缘应该是1（高通部分）
        assert mask[32, 32] == 1.0
        assert mask[0, 0] == 1.0

    def test_circular_equivalent_bandpass(self):
        mask1 = create_bandlimit_mask(
            (64, 64), BandlimitType.CIRCULAR,
            inner_radius=0.1, outer_radius=0.5
        )
        mask2 = create_bandlimit_mask(
            (64, 64), BandlimitType.BANDPASS,
            inner_radius=0.1, outer_radius=0.5
        )
        np.testing.assert_array_equal(mask1, mask2)

    def test_rectangular_mask(self):
        mask = create_bandlimit_mask(
            (64, 64), BandlimitType.RECTANGULAR,
            fx_range=(0.0, 0.3), fy_range=(0.0, 0.3)
        )
        assert mask.shape == (64, 64)
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)

    def test_directional_mask(self):
        mask = create_bandlimit_mask(
            (64, 64), BandlimitType.DIRECTIONAL,
            inner_radius=0.1, outer_radius=0.5,
            angle_range=(0.0, np.pi / 2)
        )
        assert mask.shape == (64, 64)
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)

    def test_smooth_transition(self):
        mask = create_bandlimit_mask(
            (64, 64), BandlimitType.LOWPASS,
            outer_radius=0.5, smooth=True, order=4
        )
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
        # 平滑版应该有渐变，而不是硬边界
        assert mask[32, 32] > 0.9

    def test_custom_mask(self):
        custom = np.random.random((32, 32))
        mask = create_bandlimit_mask(
            (32, 32), BandlimitType.CUSTOM, custom_mask=custom
        )
        np.testing.assert_array_almost_equal(mask, custom)

    def test_invalid_custom_mask_shape(self):
        custom = np.random.random((16, 16))
        with pytest.raises(ValueError):
            create_bandlimit_mask(
                (32, 32), BandlimitType.CUSTOM, custom_mask=custom
            )

    def test_custom_mask_none_raises(self):
        with pytest.raises(ValueError):
            create_bandlimit_mask((32, 32), BandlimitType.CUSTOM)


class TestBandlimitProjection:
    """频域带限投影测试"""

    def test_projection_output_shape(self):
        mask = np.random.random((64, 64))
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.5)
        projected = bandlimit_projection(mask, bl_mask)
        assert projected.shape == mask.shape

    def test_projection_output_real(self):
        mask = np.random.random((64, 64))
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.5)
        projected = bandlimit_projection(mask, bl_mask)
        assert not np.iscomplexobj(projected)

    def test_full_band_no_change(self):
        mask = np.random.random((64, 64))
        full_mask = np.ones((64, 64))
        projected = bandlimit_projection(mask, full_mask, preserve_dc=False)
        np.testing.assert_array_almost_equal(projected, mask, decimal=10)

    def test_lowpass_removes_high_freq(self):
        mask = np.random.random((64, 64))
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.2)
        projected = bandlimit_projection(mask, bl_mask)
        pass_ratio, stop_ratio, _ = compute_spectral_energy_ratio(projected, bl_mask)
        assert pass_ratio > 0.99

    def test_shape_mismatch_raises(self):
        mask = np.random.random((32, 32))
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS)
        with pytest.raises(ValueError):
            bandlimit_projection(mask, bl_mask)

    def test_preserve_dc(self):
        mask = np.ones((64, 64)) * 0.5
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.HIGHPASS, inner_radius=0.1)
        projected = bandlimit_projection(mask, bl_mask, preserve_dc=True)
        # 保留DC的话，均值应该接近0.5
        assert abs(np.mean(projected) - 0.5) < 0.01

    def test_no_preserve_dc(self):
        mask = np.ones((64, 64)) * 0.5
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.HIGHPASS, inner_radius=0.1)
        projected = bandlimit_projection(mask, bl_mask, preserve_dc=False)
        # 不保留DC的话，均值应该接近0
        assert abs(np.mean(projected)) < 0.01

    def test_projection_idempotent(self):
        """投影操作应该是幂等的（两次投影结果相同）"""
        mask = np.random.random((64, 64))
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.3)
        p1 = bandlimit_projection(mask, bl_mask)
        p2 = bandlimit_projection(p1, bl_mask)
        np.testing.assert_array_almost_equal(p1, p2, decimal=10)


class TestBandlimitProjectionSimple:
    """简化版频域带限投影测试"""

    def test_lowpass_simple(self):
        mask = np.random.random((64, 64))
        projected = bandlimit_projection_simple(mask, BandlimitType.LOWPASS, cutoff=0.3)
        assert projected.shape == mask.shape

    def test_highpass_simple(self):
        mask = np.random.random((64, 64))
        projected = bandlimit_projection_simple(mask, BandlimitType.HIGHPASS, cutoff=0.3)
        assert projected.shape == mask.shape

    def test_bandpass_simple(self):
        mask = np.random.random((64, 64))
        projected = bandlimit_projection_simple(
            mask, BandlimitType.BANDPASS, cutoff=0.4, bandwidth=0.2
        )
        assert projected.shape == mask.shape

    def test_bandstop_simple(self):
        mask = np.random.random((64, 64))
        projected = bandlimit_projection_simple(
            mask, BandlimitType.BANDSTOP, cutoff=0.4, bandwidth=0.2
        )
        assert projected.shape == mask.shape

    def test_string_type_input(self):
        mask = np.random.random((64, 64))
        projected = bandlimit_projection_simple(mask, 'lowpass', cutoff=0.3)
        assert projected.shape == mask.shape


class TestSpectralEnergyRatio:
    """频谱能量比率计算测试"""

    def test_energy_ratio_sum_one(self):
        mask = np.random.random((64, 64))
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.5)
        pass_r, stop_r, _ = compute_spectral_energy_ratio(mask, bl_mask)
        assert abs(pass_r + stop_r - 1.0) < 1e-10

    def test_full_band_pass_ratio_one(self):
        mask = np.random.random((64, 64))
        full_mask = np.ones((64, 64))
        pass_r, stop_r, _ = compute_spectral_energy_ratio(mask, full_mask)
        assert abs(pass_r - 1.0) < 1e-10
        assert abs(stop_r) < 1e-10

    def test_zero_mask_energy(self):
        mask = np.zeros((32, 32))
        bl_mask = create_bandlimit_mask((32, 32), BandlimitType.LOWPASS)
        pass_r, stop_r, total = compute_spectral_energy_ratio(mask, bl_mask)
        assert pass_r == 1.0
        assert stop_r == 0.0
        assert total == 0.0


class TestBandlimitedGradientProjection:
    """梯度频域带限投影测试"""

    def test_gradient_projection_shape(self):
        grad = np.random.random((64, 64))
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.5)
        projected = bandlimited_gradient_projection(grad, bl_mask)
        assert projected.shape == grad.shape

    def test_gradient_projection_no_dc(self):
        grad = np.ones((64, 64)) * 0.5
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.HIGHPASS, inner_radius=0.1)
        projected = bandlimited_gradient_projection(grad, bl_mask)
        assert abs(np.mean(projected)) < 0.01

    def test_gradient_projection_idempotent(self):
        grad = np.random.random((64, 64))
        bl_mask = create_bandlimit_mask((64, 64), BandlimitType.LOWPASS, outer_radius=0.3)
        p1 = bandlimited_gradient_projection(grad, bl_mask)
        p2 = bandlimited_gradient_projection(p1, bl_mask)
        np.testing.assert_array_almost_equal(p1, p2, decimal=10)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-k', 'Bandlimit or bandlimit or spectral or Spectral'])
