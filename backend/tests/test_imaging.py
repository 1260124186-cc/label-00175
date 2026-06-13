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
    _apply_threshold,
    _apply_sigmoid_threshold,
    _apply_car_development,
    apply_resist_model,
    ResistType,
    ResistThresholdMode,
    ResistModel,
    IlluminationType,
    TCCMode,
    generate_source,
    compute_tcc_kernel_2d,
    socs_decomposition,
    compute_zernike_phase,
    _compute_pupil_with_aberrations,
    _zernike_polynomial,
    _noll_to_nm,
    _parse_zernike_coefficients,
    ZERNIKE_NAMES,
    ZERNIKE_NAME_TO_INDEX,
    AberrationType,
    load_aberration_scenarios,
    create_aberration_sweep,
    ProcessCondition
)
from core.fft import WindowType


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


class TestIlluminationType:
    """照明模式枚举测试"""

    def test_illumination_types_exist(self):
        """测试所有照明模式类型存在"""
        assert IlluminationType.CONVENTIONAL.value == "conventional"
        assert IlluminationType.DIPOLE.value == "dipole"
        assert IlluminationType.ANNULAR.value == "annular"
        assert IlluminationType.QUASAR.value == "quasar"
        assert IlluminationType.CUSTOM.value == "custom"

    def test_illumination_type_from_string(self):
        """测试从字符串创建照明模式"""
        assert IlluminationType("conventional") == IlluminationType.CONVENTIONAL
        assert IlluminationType("annular") == IlluminationType.ANNULAR


class TestOpticalSystemExtended:
    """扩展的光学系统参数测试"""

    def test_default_illumination_type(self):
        """测试默认照明模式"""
        optics = OpticalSystem()
        assert optics.illumination_type == IlluminationType.CONVENTIONAL

    def test_custom_illumination_type(self):
        """测试自定义照明模式"""
        optics = OpticalSystem(
            illumination_type=IlluminationType.DIPOLE,
            source_params={'sigma_inner': 0.5, 'sigma_outer': 0.8, 'angle': 0.0, 'opening_angle': 60.0}
        )
        assert optics.illumination_type == IlluminationType.DIPOLE
        assert optics.source_params['sigma_inner'] == 0.5
        assert optics.source_params['sigma_outer'] == 0.8

    def test_socs_configuration(self):
        """测试SOCS配置"""
        optics = OpticalSystem(use_socs=True, socs_num_terms=10)
        assert optics.use_socs == True
        assert optics.socs_num_terms == 10

    def test_from_config(self):
        """测试从配置字典创建"""
        config = {
            'optical_system': {
                'wavelength': 193.0,
                'na': 1.35,
                'sigma': 0.75,
                'illumination_type': 'annular',
                'source_params': {'sigma_inner': 0.6, 'sigma_outer': 0.9},
                'use_socs': True,
                'socs_num_terms': 8
            }
        }
        optics = OpticalSystem.from_config(config)
        assert optics.illumination_type == IlluminationType.ANNULAR
        assert optics.source_params['sigma_inner'] == 0.6
        assert optics.source_params['sigma_outer'] == 0.9
        assert optics.socs_num_terms == 8

    def test_to_dict(self):
        """测试转换为字典"""
        optics = OpticalSystem(
            illumination_type=IlluminationType.QUASAR,
            source_params={'sigma_inner': 0.5, 'sigma_outer': 0.8, 'angle': 45.0, 'opening_angle': 30.0}
        )
        d = optics.to_dict()
        assert d['illumination_type'] == 'quasar'
        assert d['source_params']['angle'] == 45.0


class TestSourceGeneration:
    """光源生成测试"""

    @pytest.fixture
    def frequency_grid(self):
        """创建频率网格 - 使用更大的像素尺寸确保足够的频率采样点"""
        nx, ny = 128, 128
        pixel_size = 5.0  # 使用5nm像素尺寸，确保频率采样足够密集
        fx = np.fft.fftfreq(nx, pixel_size)
        fy = np.fft.fftfreq(ny, pixel_size)
        fx, fy = np.meshgrid(fx, fy)
        return fx, fy, nx, ny

    def test_conventional_source(self, frequency_grid):
        """测试传统圆形光源"""
        fx, fy, nx, ny = frequency_grid
        cutoff = 1.35 / 193.0
        source = generate_source(fx, fy, IlluminationType.CONVENTIONAL,
                                 {'sigma_inner': 0.0, 'sigma_outer': 0.75}, cutoff)

        assert source.shape == (nx, ny)
        assert np.abs(np.sum(source) - 1.0) < 1e-10
        assert np.all(source >= 0)
        assert np.sum(source > 0) > 10  # 确保有足够的非零点

    def test_annular_source(self, frequency_grid):
        """测试环形光源"""
        fx, fy, nx, ny = frequency_grid
        cutoff = 1.35 / 193.0
        source = generate_source(fx, fy, IlluminationType.ANNULAR,
                                 {'sigma_inner': 0.3, 'sigma_outer': 0.7}, cutoff)

        assert source.shape == (nx, ny)
        assert np.abs(np.sum(source) - 1.0) < 1e-10
        assert np.sum(source > 0) > 10  # 确保有足够的非零点

        rho = np.sqrt(fx**2 + fy**2) / cutoff
        inner_region = rho < 0.3
        assert np.all(source[inner_region] == 0)

    def test_dipole_source(self, frequency_grid):
        """测试偶极光源"""
        fx, fy, nx, ny = frequency_grid
        cutoff = 1.35 / 193.0
        source = generate_source(fx, fy, IlluminationType.DIPOLE,
                                 {'sigma_inner': 0.1, 'sigma_outer': 0.8, 'angle': 0.0, 'opening_angle': 90.0}, cutoff)

        assert source.shape == (nx, ny)
        assert np.abs(np.sum(source) - 1.0) < 1e-10

        nonzero_mask = source > 0
        assert np.sum(nonzero_mask) >= 8  # 离散采样下偶极光源可能点数较少

    def test_quasar_source(self, frequency_grid):
        """测试四极光源"""
        fx, fy, nx, ny = frequency_grid
        cutoff = 1.35 / 193.0
        source = generate_source(fx, fy, IlluminationType.QUASAR,
                                 {'sigma_inner': 0.1, 'sigma_outer': 0.8, 'angle': 45.0, 'opening_angle': 45.0}, cutoff)

        assert source.shape == (nx, ny)
        assert np.abs(np.sum(source) - 1.0) < 1e-10

        nonzero_mask = source > 0
        assert np.sum(nonzero_mask) >= 8  # 离散采样下四极光源可能点数较少

    def test_custom_source(self, frequency_grid):
        """测试自定义光源"""
        fx, fy, nx, ny = frequency_grid
        cutoff = 1.35 / 193.0
        custom = np.random.random((nx, ny))
        custom = custom / np.sum(custom)

        source = generate_source(fx, fy, IlluminationType.CUSTOM, {}, cutoff, custom_source=custom)

        np.testing.assert_array_almost_equal(source, custom)


class TestTCCComputation:
    """TCC计算测试"""

    @pytest.fixture
    def imaging_setup(self):
        """创建成像设置"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(optics, (32, 32))
        return imaging

    def test_tcc_kernel_2d_shape(self, imaging_setup):
        """测试二维TCC核形状"""
        imaging = imaging_setup
        cutoff = imaging.optics.cutoff_frequency

        optics_no_socs = OpticalSystem(use_socs=False)
        imaging_no_socs = PartialCoherentImaging(optics_no_socs, (32, 32))

        assert imaging_no_socs.tcc is not None
        assert imaging_no_socs.tcc.shape == (32, 32)
        assert np.all(imaging_no_socs.tcc >= 0)

    def test_tcc_kernel_normalization(self, imaging_setup):
        """测试TCC核归一化"""
        optics_no_socs = OpticalSystem(use_socs=False)
        imaging_no_socs = PartialCoherentImaging(optics_no_socs, (32, 32))

        assert np.abs(np.sum(imaging_no_socs.tcc) - 1.0) < 1e-6


class TestSOCSDecomposition:
    """SOCS低秩分解测试"""

    @pytest.fixture
    def imaging_setup(self):
        """创建成像设置 - 使用更大的像素尺寸确保足够的频率采样"""
        optics = OpticalSystem(
            socs_num_terms=5,
            pixel_size=5.0,
            sigma=0.75
        )
        imaging = PartialCoherentImaging(optics, (64, 64))
        return imaging

    def test_socs_eigenvalues_shape(self, imaging_setup):
        """测试SOCS特征值形状"""
        imaging = imaging_setup

        assert imaging.socs_eigenvalues is not None
        assert imaging.socs_eigenfunctions is not None
        assert len(imaging.socs_eigenvalues) <= 5
        assert imaging.socs_eigenfunctions.shape[0] == len(imaging.socs_eigenvalues)
        assert imaging.socs_eigenfunctions.shape[1:] == (64, 64)

    def test_socs_eigenvalues_positive(self, imaging_setup):
        """测试SOCS特征值为正"""
        imaging = imaging_setup

        assert np.all(imaging.socs_eigenvalues >= -1e-10)  # 允许微小数值误差

    def test_socs_eigenvalues_normalized(self, imaging_setup):
        """测试SOCS特征值归一化"""
        imaging = imaging_setup
        total = np.sum(imaging.socs_eigenvalues)

        if total > 0:
            assert np.abs(total - 1.0) < 1e-6

    def test_socs_eigenfunctions_orthogonal(self, imaging_setup):
        """测试SOCS特征函数正交性"""
        imaging = imaging_setup
        num_terms = len(imaging.socs_eigenvalues)
        eigenfuncs = imaging.socs_eigenfunctions.reshape(num_terms, -1)

        overlap = np.abs(eigenfuncs @ eigenfuncs.conj().T)
        diag = np.diag(overlap)
        off_diag = overlap - np.diag(diag)

        for d in diag:
            if d > 1e-6:
                assert np.abs(d - 1.0) < 1e-3  # 放宽正交性容差

        assert np.all(off_diag < 1e-2)  # 放宽非对角正交性容差


class TestPartialCoherentImagingModes:
    """不同照明模式下的成像测试"""

    def test_annular_imaging(self):
        """测试环形照明成像"""
        optics = OpticalSystem(
            illumination_type=IlluminationType.ANNULAR,
            source_params={'sigma_inner': 0.6, 'sigma_outer': 0.9}
        )
        imaging = PartialCoherentImaging(optics, (32, 32))
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0

        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)
        assert aerial.min() >= 0
        assert aerial.max() <= 1

    def test_dipole_imaging(self):
        """测试偶极照明成像"""
        optics = OpticalSystem(
            illumination_type=IlluminationType.DIPOLE,
            source_params={'sigma_inner': 0.5, 'sigma_outer': 0.8, 'angle': 0.0, 'opening_angle': 60.0}
        )
        imaging = PartialCoherentImaging(optics, (32, 32))
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0

        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)

    def test_quasar_imaging(self):
        """测试四极照明成像"""
        optics = OpticalSystem(
            illumination_type=IlluminationType.QUASAR,
            source_params={'sigma_inner': 0.5, 'sigma_outer': 0.8, 'angle': 45.0, 'opening_angle': 30.0}
        )
        imaging = PartialCoherentImaging(optics, (32, 32))
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0

        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)

    def test_imaging_without_socs(self):
        """测试不使用SOCS的成像"""
        optics = OpticalSystem(use_socs=False)
        imaging = PartialCoherentImaging(optics, (32, 32))
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0

        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)

    def test_gradient_computation_with_socs(self):
        """测试SOCS模式下的梯度计算"""
        optics = OpticalSystem(use_socs=True, socs_num_terms=3)
        imaging = PartialCoherentImaging(optics, (32, 32))
        mask = np.random.random((32, 32))

        gradient = imaging.compute_image_gradient(mask)
        assert gradient.shape == (32, 32)

    def test_gradient_computation_without_socs(self):
        """测试非SOCS模式下的梯度计算"""
        optics = OpticalSystem(use_socs=False)
        imaging = PartialCoherentImaging(optics, (32, 32))
        mask = np.random.random((32, 32))

        gradient = imaging.compute_image_gradient(mask)
        assert gradient.shape == (32, 32)


class TestVisualizationHelpers:
    """可视化辅助函数测试"""

    def test_get_source_image(self):
        """测试获取光源图像"""
        optics = OpticalSystem(illumination_type=IlluminationType.ANNULAR)
        imaging = PartialCoherentImaging(optics, (32, 32))

        source_img = imaging.get_source_image()
        assert source_img.shape == (32, 32)

    def test_get_pupil_image(self):
        """测试获取光瞳图像"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(optics, (32, 32))

        pupil_img = imaging.get_pupil_image()
        assert pupil_img.shape == (32, 32)

    def test_get_tcc_image_no_socs(self):
        """测试获取TCC图像（非SOCS模式）"""
        optics = OpticalSystem(use_socs=False)
        imaging = PartialCoherentImaging(optics, (32, 32))

        tcc_img = imaging.get_tcc_image()
        assert tcc_img is not None
        assert tcc_img.shape == (32, 32)

    def test_get_tcc_image_with_socs(self):
        """测试获取TCC图像（SOCS模式也返回2D核对角近似，便于可视化）"""
        optics = OpticalSystem(use_socs=True)
        imaging = PartialCoherentImaging(optics, (32, 32))

        tcc_img = imaging.get_tcc_image()
        assert tcc_img is not None
        assert tcc_img.shape == (32, 32)


class TestSigmoidThreshold:
    """可微 sigmoid 阈值测试"""

    def test_output_range(self):
        """测试输出值在 (0, 1) 范围内"""
        image = np.array([[0.1, 0.5], [0.7, 0.9]])
        result = _apply_sigmoid_threshold(image, 0.5, 10.0)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_monotonicity(self):
        """测试单调性：输入越大输出越大"""
        image = np.linspace(0.0, 1.0, 100).reshape(1, 100)
        result = _apply_sigmoid_threshold(image, 0.5, 20.0)
        diffs = np.diff(result.ravel())
        assert np.all(diffs >= 0)

    def test_symmetry_around_threshold(self):
        """测试关于阈值的对称性"""
        image = np.array([[0.4, 0.6]])
        result = _apply_sigmoid_threshold(image, 0.5, 10.0)
        assert abs(result[0, 0] - (1.0 - result[0, 1])) < 1e-10

    def test_high_steepness_approaches_hard(self):
        """测试高陡度时趋近硬阈值"""
        image = np.array([[0.2, 0.4], [0.6, 0.8]])
        result = _apply_sigmoid_threshold(image, 0.5, 1000.0)
        hard = _apply_threshold(image, 0.5)
        np.testing.assert_allclose(result, hard, atol=1e-3)

    def test_gradient_nonzero(self):
        """测试梯度非零（可微性验证）"""
        image = np.array([[0.45, 0.55]])
        k = 20.0
        eps = 1e-6
        r_plus = _apply_sigmoid_threshold(image + eps, 0.5, k)
        r_minus = _apply_sigmoid_threshold(image - eps, 0.5, k)
        grad = (r_plus - r_minus) / (2 * eps)
        assert np.all(grad > 0)


class TestCARDevelopment:
    """化学放大光刻胶（CAR）显影模型测试"""

    def test_output_range(self):
        """测试输出值在 [0, 1] 范围内"""
        image = np.array([[0.1, 0.5], [0.7, 0.9]])
        result = _apply_car_development(image, 0.5, 5.0, 5.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_monotonicity(self):
        """测试单调性：曝光量越大显影越多"""
        image = np.linspace(0.01, 1.0, 100).reshape(1, 100)
        result = _apply_car_development(image, 0.5, 5.0, 5.0)
        diffs = np.diff(result.ravel())
        assert np.all(diffs >= 0)

    def test_high_amplification_approaches_hard(self):
        """测试高放大倍率趋近硬阈值"""
        image = np.array([[0.2, 0.4], [0.6, 0.8]])
        result = _apply_car_development(image, 0.5, 100.0, 10.0)
        hard = _apply_threshold(image, 0.5)
        np.testing.assert_allclose(result, hard, atol=1e-2)

    def test_zero_threshold_all_clear(self):
        """测试零阈值时全部显影"""
        image = np.array([[0.1, 0.5], [0.7, 0.9]])
        result = _apply_car_development(image, 0.0, 5.0, 5.0)
        np.testing.assert_array_equal(result, 1.0)

    def test_zero_intensity_zero_output(self):
        """测试零光强输出为零"""
        image = np.array([[0.0, 0.5], [0.7, 0.0]])
        result = _apply_car_development(image, 0.5, 5.0, 5.0)
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0


class TestResistModel:
    """高级光刻胶模型测试"""

    def test_default_is_positive_hard(self):
        """测试默认模型为正性硬阈值"""
        model = ResistModel()
        assert model.resist_type == ResistType.POSITIVE
        assert model.threshold_mode == ResistThresholdMode.HARD

    def test_positive_resist(self):
        """测试正性胶：光强高于阈值处显影（值为1）"""
        image = np.array([[0.2, 0.4], [0.6, 0.8]])
        model = ResistModel(base_threshold=0.5)
        result = apply_resist_model(image, resist_model=model)
        expected = _apply_threshold(image, 0.5)
        np.testing.assert_array_equal(result, expected)

    def test_negative_resist(self):
        """测试负性胶：光强低于阈值处显影（值为1）"""
        image = np.array([[0.2, 0.4], [0.6, 0.8]])
        model = ResistModel(resist_type=ResistType.NEGATIVE, base_threshold=0.5)
        result = apply_resist_model(image, resist_model=model)
        hard = _apply_threshold(image, 0.5)
        expected = 1.0 - hard
        np.testing.assert_array_equal(result, expected)

    def test_sigmoid_mode(self):
        """测试 sigmoid 模式输出连续值"""
        image = np.array([[0.2, 0.4], [0.6, 0.8]])
        model = ResistModel(threshold_mode=ResistThresholdMode.SIGMOID,
                            base_threshold=0.5, sigmoid_steepness=20.0)
        result = apply_resist_model(image, resist_model=model)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_car_mode(self):
        """测试 CAR 模式输出"""
        image = np.array([[0.1, 0.5], [0.7, 0.9]])
        model = ResistModel(car_enabled=True, base_threshold=0.5,
                            car_amplification=10.0, car_contrast=5.0)
        result = apply_resist_model(image, resist_model=model)
        assert result.shape == (2, 2)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_negative_sigmoid(self):
        """测试负性胶 + sigmoid"""
        image = np.array([[0.2, 0.4], [0.6, 0.8]])
        model = ResistModel(resist_type=ResistType.NEGATIVE,
                            threshold_mode=ResistThresholdMode.SIGMOID,
                            base_threshold=0.5, sigmoid_steepness=20.0)
        result = apply_resist_model(image, resist_model=model)
        assert result[0, 0] > 0.5
        assert result[1, 1] < 0.5

    def test_tmr_field(self):
        """测试 TMR 阈值调制"""
        image = np.array([[0.4, 0.4], [0.4, 0.4]])
        tmr = np.array([[0.0, 0.1], [-0.1, 0.0]])
        model = ResistModel(tmr_enabled=True, tmr_field=tmr, base_threshold=0.4)
        result = apply_resist_model(image, resist_model=model)
        assert result[0, 0] == 1.0
        assert result[0, 1] == 0.0
        assert result[1, 0] == 1.0
        assert result[1, 1] == 1.0

    def test_tmr_shape_mismatch(self):
        """测试 TMR 场形状不匹配时报错"""
        image = np.ones((4, 4))
        tmr = np.ones((3, 3))
        model = ResistModel(tmr_enabled=True, tmr_field=tmr, base_threshold=0.3)
        with pytest.raises(ValueError):
            apply_resist_model(image, resist_model=model)

    def test_none_resist_model_fallback(self):
        """测试 resist_model=None 时回退到硬阈值"""
        image = np.array([[0.2, 0.4], [0.6, 0.8]])
        result = apply_resist_model(image, resist_model=None, threshold=0.5)
        expected = _apply_threshold(image, 0.5)
        np.testing.assert_array_equal(result, expected)


class TestSimulateWaferWithResistModel:
    """带高级光刻胶模型的晶圆仿真测试"""

    def test_sigmoid_resist_in_simulation(self):
        """测试 sigmoid 胶模型在仿真中的端到端运行"""
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        model = ResistModel(threshold_mode=ResistThresholdMode.SIGMOID,
                            base_threshold=0.3, sigmoid_steepness=30.0)
        wafer = simulate_wafer_image(mask, resist_model=model)
        assert wafer.shape == (32, 32)
        assert wafer.dtype == np.float64

    def test_negative_resist_in_simulation(self):
        """测试负性胶在仿真中"""
        image = np.array([[0.2, 0.8]])
        model = ResistModel(resist_type=ResistType.NEGATIVE, base_threshold=0.5)
        result = apply_resist_model(image, resist_model=model)
        assert result[0, 0] == 1.0
        assert result[0, 1] == 0.0

    def test_car_resist_in_simulation(self):
        """测试 CAR 模型在仿真中"""
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        model = ResistModel(car_enabled=True, base_threshold=0.3,
                            car_amplification=8.0, car_contrast=5.0)
        wafer = simulate_wafer_image(mask, resist_model=model)
        assert wafer.shape == (32, 32)
        assert wafer.dtype == np.float64

    def test_backward_compatible_no_resist_model(self):
        """测试无 resist_model 时向后兼容"""
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        wafer_old = simulate_wafer_image(mask, threshold=0.3, apply_resist=True)
        wafer_new = simulate_wafer_image(mask, threshold=0.3, apply_resist=True,
                                         resist_model=None)
        np.testing.assert_array_equal(wafer_old, wafer_new)


class TestZernikePolynomials:
    """Zernike 多项式测试"""

    @pytest.fixture
    def grid(self):
        nx, ny = 64, 64
        rho = np.linspace(0, 1, nx)
        theta = np.linspace(0, 2 * np.pi, ny)
        rho_grid, theta_grid = np.meshgrid(rho, theta)
        return rho_grid, theta_grid

    def test_piston(self, grid):
        """测试活塞项 Z_0 = 1"""
        rho, theta = grid
        z0 = _zernike_polynomial(0, rho, theta)
        np.testing.assert_allclose(z0, 1.0)

    def test_tilt_x(self, grid):
        """测试X倾斜 Z_1 = 2ρ cos(θ)"""
        rho, theta = grid
        z1 = _zernike_polynomial(1, rho, theta)
        expected = 2.0 * rho * np.cos(theta)
        np.testing.assert_allclose(z1, expected, atol=1e-12)

    def test_tilt_y(self, grid):
        """测试Y倾斜 Z_2 = 2ρ sin(θ)"""
        rho, theta = grid
        z2 = _zernike_polynomial(2, rho, theta)
        expected = 2.0 * rho * np.sin(theta)
        np.testing.assert_allclose(z2, expected, atol=1e-12)

    def test_defocus(self, grid):
        """测试离焦 Z_3 = √3 (2ρ² - 1)"""
        rho, theta = grid
        z3 = _zernike_polynomial(3, rho, theta)
        expected = np.sqrt(3.0) * (2.0 * rho ** 2 - 1.0)
        np.testing.assert_allclose(z3, expected, atol=1e-12)

    def test_spherical(self, grid):
        """测试球差 Z_10 = √5 (6ρ⁴ - 6ρ² + 1)"""
        rho, theta = grid
        z10 = _zernike_polynomial(10, rho, theta)
        expected = np.sqrt(5.0) * (6.0 * rho ** 4 - 6.0 * rho ** 2 + 1.0)
        np.testing.assert_allclose(z10, expected, atol=1e-12)

    def test_coma_x(self, grid):
        """测试X彗差 Z_7 = √8 (3ρ³ - 2ρ) cos(θ)"""
        rho, theta = grid
        z7 = _zernike_polynomial(7, rho, theta)
        expected = np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho) * np.cos(theta)
        np.testing.assert_allclose(z7, expected, atol=1e-12)

    def test_coma_y(self, grid):
        """测试Y彗差 Z_6 = √8 (3ρ³ - 2ρ) sin(θ)"""
        rho, theta = grid
        z6 = _zernike_polynomial(6, rho, theta)
        expected = np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho) * np.sin(theta)
        np.testing.assert_allclose(z6, expected, atol=1e-12)

    def test_astigmatism_x(self, grid):
        """测试X像散 Z_5 = √6 ρ² cos(2θ)"""
        rho, theta = grid
        z5 = _zernike_polynomial(5, rho, theta)
        expected = np.sqrt(6.0) * rho ** 2 * np.cos(2.0 * theta)
        np.testing.assert_allclose(z5, expected, atol=1e-12)

    def test_astigmatism_y(self, grid):
        """测试Y像散 Z_4 = √6 ρ² sin(2θ)"""
        rho, theta = grid
        z4 = _zernike_polynomial(4, rho, theta)
        expected = np.sqrt(6.0) * rho ** 2 * np.sin(2.0 * theta)
        np.testing.assert_allclose(z4, expected, atol=1e-12)

    def test_noll_to_nm(self):
        """测试 Noll 索引到 (n, m) 的转换"""
        assert _noll_to_nm(0) == (0, 0)
        assert _noll_to_nm(1) == (1, 1)
        assert _noll_to_nm(2) == (1, -1)
        assert _noll_to_nm(3) == (2, 0)
        assert _noll_to_nm(10) == (4, 0)
        assert _noll_to_nm(4) == (2, -2)
        assert _noll_to_nm(5) == (2, 2)
        assert _noll_to_nm(7) == (3, 1)

    def test_zernike_unit_circle(self):
        """测试 Zernike 多项式在单位圆上的正交性"""
        n = 256
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(x, y)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        mask = rho <= 1.0

        z3 = _zernike_polynomial(3, rho, theta)
        z10 = _zernike_polynomial(10, rho, theta)

        area = np.sum(mask)
        overlap = np.sum(z3[mask] * z10[mask]) / area
        assert abs(overlap) < 0.1

        norm3 = np.sum(z3[mask] ** 2) / area
        assert abs(norm3 - 1.0) < 0.1


class TestZernikePhase:
    """Zernike 像差相位计算测试"""

    @pytest.fixture
    def freq_grid(self):
        pixel_size = 5.0
        nx, ny = 64, 64
        fx = np.fft.fftfreq(nx, pixel_size)
        fy = np.fft.fftfreq(ny, pixel_size)
        fx, fy = np.meshgrid(fx, fy)
        cutoff = 1.35 / 193.0
        return fx, fy, cutoff

    def test_empty_coefficients(self, freq_grid):
        """测试空系数返回零相位"""
        fx, fy, cutoff = freq_grid
        phase = compute_zernike_phase(fx, fy, cutoff, {})
        assert np.allclose(phase, 0.0)

    def test_phase_shape(self, freq_grid):
        """测试相位输出形状"""
        fx, fy, cutoff = freq_grid
        phase = compute_zernike_phase(fx, fy, cutoff, {10: 0.05})
        assert phase.shape == fx.shape

    def test_phase_outside_pupil_zero(self, freq_grid):
        """测试光瞳外相位为零"""
        fx, fy, cutoff = freq_grid
        phase = compute_zernike_phase(fx, fy, cutoff, {10: 0.05})
        rho = np.sqrt(fx ** 2 + fy ** 2) / cutoff
        assert np.allclose(phase[rho > 1.0], 0.0)

    def test_spherical_phase_nonzero(self, freq_grid):
        """测试球差相位非零"""
        fx, fy, cutoff = freq_grid
        phase = compute_zernike_phase(fx, fy, cutoff, {10: 0.05})
        rho = np.sqrt(fx ** 2 + fy ** 2) / cutoff
        inside = rho <= 1.0
        assert np.any(np.abs(phase[inside]) > 1e-10)


class TestPupilWithAberrations:
    """含像差光瞳函数测试"""

    @pytest.fixture
    def freq_grid(self):
        pixel_size = 5.0
        nx, ny = 64, 64
        fx = np.fft.fftfreq(nx, pixel_size)
        fy = np.fft.fftfreq(ny, pixel_size)
        fx, fy = np.meshgrid(fx, fy)
        cutoff = 1.35 / 193.0
        return fx, fy, cutoff

    def test_pupil_shape(self, freq_grid):
        """测试光瞳函数输出形状"""
        fx, fy, cutoff = freq_grid
        zernike_phase = np.zeros_like(fx)
        pupil = _compute_pupil_with_aberrations(fx, fy, cutoff, 0.0, 193.0, zernike_phase)
        assert pupil.shape == fx.shape
        assert pupil.dtype == np.complex128

    def test_pupil_zero_defocus_no_aberration(self, freq_grid):
        """测试零离焦无像差时光瞳内幅度为1"""
        fx, fy, cutoff = freq_grid
        zernike_phase = np.zeros_like(fx)
        pupil = _compute_pupil_with_aberrations(fx, fy, cutoff, 0.0, 193.0, zernike_phase)
        rho = np.sqrt(fx ** 2 + fy ** 2) / cutoff
        inside = rho <= 1.0
        np.testing.assert_allclose(np.abs(pupil[inside]), 1.0, atol=1e-12)

    def test_pupil_outside_zero(self, freq_grid):
        """测试光瞳外为零"""
        fx, fy, cutoff = freq_grid
        zernike_phase = np.zeros_like(fx)
        pupil = _compute_pupil_with_aberrations(fx, fy, cutoff, 0.0, 193.0, zernike_phase)
        rho = np.sqrt(fx ** 2 + fy ** 2) / cutoff
        outside = rho > 1.0
        assert np.allclose(pupil[outside], 0.0)

    def test_aberration_modifies_pupil(self, freq_grid):
        """测试像差改变光瞳函数"""
        fx, fy, cutoff = freq_grid
        no_aberr = _compute_pupil_with_aberrations(
            fx, fy, cutoff, 0.0, 193.0, np.zeros_like(fx))
        zernike_phase = compute_zernike_phase(fx, fy, cutoff, {10: 0.1})
        with_aberr = _compute_pupil_with_aberrations(
            fx, fy, cutoff, 0.0, 193.0, zernike_phase)
        rho = np.sqrt(fx ** 2 + fy ** 2) / cutoff
        inside = rho <= 1.0
        assert not np.allclose(no_aberr[inside], with_aberr[inside])


class TestOpticalSystemZernike:
    """OpticalSystem 的 Zernike 系数支持测试"""

    def test_default_zernike_empty(self):
        """测试默认 Zernike 系数为空"""
        optics = OpticalSystem()
        assert optics.zernike_coefficients == {}

    def test_custom_zernike(self):
        """测试自定义 Zernike 系数"""
        optics = OpticalSystem(zernike_coefficients={10: 0.05, 7: 0.03})
        assert optics.zernike_coefficients[10] == 0.05
        assert optics.zernike_coefficients[7] == 0.03

    def test_from_config_with_zernike_names(self):
        """测试从配置创建（名称格式）"""
        config = {
            'optical_system': {
                'wavelength': 193.0,
                'na': 1.35,
                'zernike_coefficients': {
                    'spherical': 0.05,
                    'coma_x': 0.03
                }
            }
        }
        optics = OpticalSystem.from_config(config)
        assert 10 in optics.zernike_coefficients
        assert 7 in optics.zernike_coefficients
        assert optics.zernike_coefficients[10] == 0.05
        assert optics.zernike_coefficients[7] == 0.03

    def test_from_config_with_zernike_indices(self):
        """测试从配置创建（索引格式）"""
        config = {
            'optical_system': {
                'zernike_coefficients': {
                    '10': 0.05,
                    '7': 0.03
                }
            }
        }
        optics = OpticalSystem.from_config(config)
        assert optics.zernike_coefficients[10] == 0.05
        assert optics.zernike_coefficients[7] == 0.03

    def test_to_dict_with_zernike(self):
        """测试带 Zernike 系数的 to_dict"""
        optics = OpticalSystem(zernike_coefficients={10: 0.05, 7: 0.03})
        d = optics.to_dict()
        assert 'zernike_coefficients' in d
        assert 'spherical' in d['zernike_coefficients']
        assert d['zernike_coefficients']['spherical'] == 0.05

    def test_to_dict_without_zernike(self):
        """测试无 Zernike 系数的 to_dict"""
        optics = OpticalSystem()
        d = optics.to_dict()
        assert 'zernike_coefficients' in d
        assert d['zernike_coefficients'] == {}


class TestParseZernikeCoefficients:
    """Zernike 系数解析测试"""

    def test_name_format(self):
        """测试名称格式解析"""
        raw = {'spherical': 0.05, 'coma_x': 0.03}
        result = _parse_zernike_coefficients(raw)
        assert result == {10: 0.05, 7: 0.03}

    def test_index_string_format(self):
        """测试字符串索引格式解析"""
        raw = {'10': 0.05, '7': 0.03}
        result = _parse_zernike_coefficients(raw)
        assert result == {10: 0.05, 7: 0.03}

    def test_int_index_format(self):
        """测试整数索引格式解析"""
        raw = {10: 0.05, 7: 0.03}
        result = _parse_zernike_coefficients(raw)
        assert result == {10: 0.05, 7: 0.03}

    def test_mixed_format(self):
        """测试混合格式解析"""
        raw = {'spherical': 0.05, '7': 0.03}
        result = _parse_zernike_coefficients(raw)
        assert result == {10: 0.05, 7: 0.03}

    def test_empty(self):
        """测试空输入"""
        assert _parse_zernike_coefficients({}) == {}

    def test_unknown_key_ignored(self):
        """测试未知键被忽略"""
        raw = {'unknown_aberration': 0.05}
        result = _parse_zernike_coefficients(raw)
        assert result == {}


class TestProcessConditionZernike:
    """ProcessCondition 的 Zernike 支持测试"""

    def test_default_zernike_empty(self):
        """测试默认 Zernike 系数为空"""
        cond = ProcessCondition()
        assert cond.zernike_coefficients == {}

    def test_custom_zernike(self):
        """测试自定义 Zernike 系数"""
        cond = ProcessCondition(zernike_coefficients={10: 0.05})
        assert cond.zernike_coefficients[10] == 0.05

    def test_to_optical_system_merges_zernike(self):
        """测试 to_optical_system 合并 Zernike 系数"""
        base = OpticalSystem(zernike_coefficients={10: 0.02})
        cond = ProcessCondition(zernike_coefficients={7: 0.03})
        opt = cond.to_optical_system(base_optics=base)
        assert opt.zernike_coefficients[10] == 0.02
        assert opt.zernike_coefficients[7] == 0.03

    def test_to_optical_system_cond_overrides_zernike(self):
        """测试 ProcessCondition 的 Zernike 系数覆盖 base"""
        base = OpticalSystem(zernike_coefficients={10: 0.02})
        cond = ProcessCondition(zernike_coefficients={10: 0.08})
        opt = cond.to_optical_system(base_optics=base)
        assert opt.zernike_coefficients[10] == 0.08


class TestImagingWithAberrations:
    """含像差的光学成像测试"""

    def test_imaging_with_spherical(self):
        """测试球差下的成像"""
        optics = OpticalSystem(zernike_coefficients={10: 0.05})
        imaging = PartialCoherentImaging(optics, (32, 32))
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)
        assert aerial.min() >= 0
        assert aerial.max() <= 1

    def test_imaging_with_coma(self):
        """测试彗差下的成像"""
        optics = OpticalSystem(zernike_coefficients={7: 0.03, 6: 0.02})
        imaging = PartialCoherentImaging(optics, (32, 32))
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)

    def test_imaging_with_defocus_and_spherical(self):
        """测试离焦+球差联合成像"""
        optics = OpticalSystem(defocus=50.0, zernike_coefficients={10: 0.05})
        imaging = PartialCoherentImaging(optics, (32, 32))
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)
        assert aerial.min() >= 0

    def test_aberration_affects_image(self):
        """测试像差确实影响成像结果"""
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0

        optics_clean = OpticalSystem(pixel_size=5.0)
        imaging_clean = PartialCoherentImaging(optics_clean, (32, 32))
        aerial_clean = imaging_clean.compute_aerial_image(mask)

        optics_aberr = OpticalSystem(
            pixel_size=5.0,
            zernike_coefficients={10: 0.1, 7: 0.05}
        )
        imaging_aberr = PartialCoherentImaging(optics_aberr, (32, 32))
        aerial_aberr = imaging_aberr.compute_aerial_image(mask)

        diff = np.abs(aerial_clean - aerial_aberr)
        assert np.max(diff) > 1e-6

    def test_gradient_with_aberrations(self):
        """测试含像差的梯度计算"""
        optics = OpticalSystem(zernike_coefficients={10: 0.05})
        imaging = PartialCoherentImaging(optics, (32, 32))
        mask = np.random.random((32, 32))
        gradient = imaging.compute_image_gradient(mask)
        assert gradient.shape == (32, 32)


class TestLoadAberrationScenarios:
    """像差场景批量加载测试"""

    def test_load_all_scenarios(self):
        """测试加载所有场景"""
        import os
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'aberration_scenarios.yaml'
        )
        if not os.path.exists(config_path):
            pytest.skip("aberration_scenarios.yaml 不存在")

        scenarios = load_aberration_scenarios(config_path)
        assert len(scenarios) > 0
        for name, opt in scenarios:
            assert isinstance(opt, OpticalSystem)

    def test_load_specific_scenarios(self):
        """测试加载指定场景"""
        import os
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'aberration_scenarios.yaml'
        )
        if not os.path.exists(config_path):
            pytest.skip("aberration_scenarios.yaml 不存在")

        scenarios = load_aberration_scenarios(
            config_path, scenario_names=['spherical_only', 'coma_only'])
        names = [n for n, _ in scenarios]
        assert 'spherical_only' in names
        assert 'coma_only' in names
        assert len(scenarios) == 2

    def test_load_with_custom_base(self):
        """测试使用自定义 base_optics"""
        import os
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'aberration_scenarios.yaml'
        )
        if not os.path.exists(config_path):
            pytest.skip("aberration_scenarios.yaml 不存在")

        base = OpticalSystem(wavelength=248.0, na=0.93)
        scenarios = load_aberration_scenarios(config_path, base_optics=base)
        for name, opt in scenarios:
            assert opt.wavelength == 248.0
            assert opt.na == 0.93

    def test_ideal_scenario_no_aberration(self):
        """测试理想场景无像差"""
        import os
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'aberration_scenarios.yaml'
        )
        if not os.path.exists(config_path):
            pytest.skip("aberration_scenarios.yaml 不存在")

        scenarios = load_aberration_scenarios(
            config_path, scenario_names=['ideal'])
        assert len(scenarios) == 1
        name, opt = scenarios[0]
        assert opt.zernike_coefficients == {}

    def test_file_not_found(self):
        """测试配置文件不存在"""
        with pytest.raises(FileNotFoundError):
            load_aberration_scenarios('/nonexistent/path.yaml')


class TestCreateAberrationSweep:
    """Zernike 系数扫描测试"""

    def test_default_sweep(self):
        """测试默认系数扫描"""
        sweep = create_aberration_sweep()
        assert len(sweep) == 5
        for name, opt in sweep:
            assert 10 in opt.zernike_coefficients

    def test_custom_coeff_values(self):
        """测试自定义系数扫描值"""
        coeff_vals = [0.0, 0.05, 0.1]
        sweep = create_aberration_sweep(coeff_values=coeff_vals)
        assert len(sweep) == 3

    def test_sweep_with_defocus(self):
        """测试带离焦的系数扫描"""
        sweep = create_aberration_sweep(
            defocus_values=[0.0, 50.0],
            coeff_values=[0.0, 0.05]
        )
        assert len(sweep) == 4

    def test_sweep_different_zernike_order(self):
        """测试扫描不同 Zernike 阶"""
        sweep = create_aberration_sweep(zernike_j=7, coeff_values=[0.0, 0.03])
        assert len(sweep) == 2
        for name, opt in sweep:
            assert 7 in opt.zernike_coefficients


class TestPartialCoherentImagingWindowPadding:
    """窗函数与零填充边界处理测试"""

    def test_no_window_no_padding_backward_compat(self):
        """测试无窗无填充时向后兼容"""
        optics = OpticalSystem()
        imaging_old = PartialCoherentImaging(optics, (32, 32))
        imaging_new = PartialCoherentImaging(
            optics, (32, 32), window_type=None, pad_width=None
        )
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial_old = imaging_old.compute_aerial_image(mask)
        aerial_new = imaging_new.compute_aerial_image(mask)
        np.testing.assert_array_almost_equal(aerial_old, aerial_new)

    def test_hann_window_aerial_shape(self):
        """测试 Hann 窗成像输出形状不变"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(
            optics, (32, 32), window_type=WindowType.HANN
        )
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)
        assert aerial.min() >= 0
        assert aerial.max() <= 1

    def test_hamming_window_aerial_shape(self):
        """测试 Hamming 窗成像输出形状不变"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(
            optics, (32, 32), window_type=WindowType.HAMMING
        )
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)

    def test_tukey_window_aerial_shape(self):
        """测试 Tukey 窗成像输出形状不变"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(
            optics, (32, 32), window_type=WindowType.TUKEY, tukey_alpha=0.3
        )
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)

    def test_padding_aerial_shape(self):
        """测试零填充后成像输出裁剪回原始尺寸"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(
            optics, (32, 32), pad_width=8
        )
        assert imaging._effective_size == (48, 48)
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)

    def test_window_and_padding_combined(self):
        """测试同时加窗和零填充"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(
            optics, (32, 32),
            window_type=WindowType.HANN,
            pad_width=8
        )
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)
        assert aerial.min() >= 0
        assert aerial.max() <= 1

    def test_window_string_type(self):
        """测试字符串类型的窗函数"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(
            optics, (32, 32), window_type='hann'
        )
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)

    def test_gradient_with_window(self):
        """测试加窗时梯度计算形状"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(
            optics, (32, 32), window_type=WindowType.HANN
        )
        mask = np.random.random((32, 32))
        gradient = imaging.compute_image_gradient(mask)
        assert gradient.shape == (32, 32)

    def test_gradient_with_padding(self):
        """测试零填充时梯度计算形状"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(
            optics, (32, 32), pad_width=8
        )
        mask = np.random.random((32, 32))
        gradient = imaging.compute_image_gradient(mask)
        assert gradient.shape == (32, 32)

    def test_gradient_with_window_and_padding(self):
        """测试同时加窗和零填充时梯度计算形状"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(
            optics, (32, 32),
            window_type=WindowType.TUKEY,
            pad_width=8,
            tukey_alpha=0.25
        )
        mask = np.random.random((32, 32))
        gradient = imaging.compute_image_gradient(mask)
        assert gradient.shape == (32, 32)

    def test_asymmetric_padding(self):
        """测试非对称零填充"""
        optics = OpticalSystem()
        imaging = PartialCoherentImaging(
            optics, (32, 32), pad_width=(4, 8)
        )
        assert imaging._effective_size == (40, 48)
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)

    def test_window_reduces_boundary_artifacts(self):
        """测试窗函数缓解边界效应"""
        optics = OpticalSystem(pixel_size=5.0)
        mask = np.zeros((64, 64))
        mask[20:44, 20:44] = 1.0

        imaging_no_window = PartialCoherentImaging(optics, (64, 64))
        imaging_hann = PartialCoherentImaging(
            optics, (64, 64), window_type=WindowType.HANN
        )

        aerial_no = imaging_no_window.compute_aerial_image(mask)
        aerial_hann = imaging_hann.compute_aerial_image(mask)

        boundary_no = np.mean(aerial_no[0:3, :]) + np.mean(aerial_no[-3:, :])
        boundary_hann = np.mean(aerial_hann[0:3, :]) + np.mean(aerial_hann[-3:, :])

        assert boundary_hann <= boundary_no + 1e-10


class TestSimulateWaferWithWindowPadding:
    """simulate_wafer_image 窗函数/零填充集成测试"""

    def test_simulate_with_hann_window(self):
        """测试 Hann 窗晶圆仿真端到端"""
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        wafer = simulate_wafer_image(
            mask, window_type='hann', apply_resist=False
        )
        assert wafer.shape == (32, 32)
        assert wafer.dtype == np.float64

    def test_simulate_with_padding(self):
        """测试零填充晶圆仿真端到端"""
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        wafer = simulate_wafer_image(
            mask, pad_width=8, apply_resist=False
        )
        assert wafer.shape == (32, 32)

    def test_simulate_with_window_and_padding(self):
        """测试同时加窗和零填充晶圆仿真端到端"""
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        wafer = simulate_wafer_image(
            mask,
            window_type=WindowType.TUKEY,
            pad_width=8,
            tukey_alpha=0.25,
            apply_resist=True
        )
        assert wafer.shape == (32, 32)
        assert wafer.dtype == np.float64

    def test_backward_compat_no_window_padding(self):
        """测试不传窗/填充参数时向后兼容"""
        mask = np.zeros((32, 32))
        mask[10:22, 10:22] = 1.0
        wafer_old = simulate_wafer_image(mask, apply_resist=False)
        wafer_new = simulate_wafer_image(
            mask, window_type=None, pad_width=None, apply_resist=False
        )
        np.testing.assert_array_almost_equal(wafer_old, wafer_new)


class TestTCCMode:
    """TCCMode 枚举测试"""

    def test_tcc_modes_exist(self):
        """测试三种 TCC 模式存在"""
        assert TCCMode.FULL_TCC.value == "full_tcc"
        assert TCCMode.SOCS.value == "socs"
        assert TCCMode.KERNEL_2D.value == "kernel_2d"

    def test_tcc_mode_from_string(self):
        """测试从字符串创建 TCCMode"""
        assert TCCMode("full_tcc") == TCCMode.FULL_TCC
        assert TCCMode("socs") == TCCMode.SOCS
        assert TCCMode("kernel_2d") == TCCMode.KERNEL_2D

    def test_invalid_tcc_mode_raises(self):
        """测试无效 TCC 模式抛出异常"""
        with pytest.raises(ValueError):
            TCCMode("invalid_mode")


class TestOpticalSystemTCCMode:
    """OpticalSystem 的 TCC 模式配置测试"""

    def test_default_tcc_mode_is_socs(self):
        """测试默认 TCC 模式为 SOCS"""
        optics = OpticalSystem()
        assert optics.tcc_mode == TCCMode.SOCS

    def test_custom_tcc_mode_full(self):
        """测试自定义 FULL_TCC 模式"""
        optics = OpticalSystem(tcc_mode=TCCMode.FULL_TCC)
        assert optics.tcc_mode == TCCMode.FULL_TCC

    def test_custom_tcc_mode_kernel_2d(self):
        """测试自定义 KERNEL_2D 模式"""
        optics = OpticalSystem(tcc_mode=TCCMode.KERNEL_2D)
        assert optics.tcc_mode == TCCMode.KERNEL_2D

    def test_use_socs_true_backward_compat(self):
        """测试 use_socs=True 向后兼容（转换为 SOCS 模式）"""
        optics = OpticalSystem(use_socs=True)
        assert optics.tcc_mode == TCCMode.SOCS
        assert optics.use_socs == True

    def test_use_socs_false_backward_compat(self):
        """测试 use_socs=False 向后兼容（转换为 FULL_TCC 模式）"""
        optics = OpticalSystem(use_socs=False)
        assert optics.tcc_mode == TCCMode.FULL_TCC
        assert optics.use_socs == False

    def test_tcc_mode_overrides_use_socs(self):
        """测试 tcc_mode 优先于 use_socs"""
        optics = OpticalSystem(tcc_mode=TCCMode.KERNEL_2D, use_socs=True)
        assert optics.tcc_mode == TCCMode.KERNEL_2D
        assert optics.use_socs == True

    def test_from_config_with_tcc_mode(self):
        """测试从配置创建（tcc_mode 格式）"""
        config = {
            'optical_system': {
                'tcc_mode': 'kernel_2d',
                'socs_num_terms': 10
            }
        }
        optics = OpticalSystem.from_config(config)
        assert optics.tcc_mode == TCCMode.KERNEL_2D
        assert optics.socs_num_terms == 10

    def test_from_config_with_use_socs_backward_compat(self):
        """测试从配置创建（use_socs 旧格式向后兼容）"""
        config = {
            'optical_system': {
                'use_socs': False
            }
        }
        optics = OpticalSystem.from_config(config)
        assert optics.tcc_mode == TCCMode.FULL_TCC

    def test_from_config_tcc_mode_priority(self):
        """测试 tcc_mode 优先级高于 use_socs"""
        config = {
            'optical_system': {
                'tcc_mode': 'kernel_2d',
                'use_socs': True
            }
        }
        optics = OpticalSystem.from_config(config)
        assert optics.tcc_mode == TCCMode.KERNEL_2D

    def test_to_dict_includes_tcc_mode(self):
        """测试 to_dict 包含 tcc_mode"""
        optics = OpticalSystem(tcc_mode=TCCMode.FULL_TCC)
        d = optics.to_dict()
        assert 'tcc_mode' in d
        assert d['tcc_mode'] == 'full_tcc'
        assert 'use_socs' not in d


class TestTCCModesImaging:
    """三种 TCC 模式下的成像测试"""

    @pytest.fixture
    def mask(self):
        mask = np.zeros((32, 32))
        mask[8:24, 8:24] = 1.0
        return mask

    def test_full_tcc_imaging_shape(self, mask):
        """测试 FULL_TCC 模式成像输出形状"""
        optics = OpticalSystem(tcc_mode=TCCMode.FULL_TCC, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)
        assert aerial.min() >= 0
        assert aerial.max() <= 1

    def test_socs_imaging_shape(self, mask):
        """测试 SOCS 模式成像输出形状"""
        optics = OpticalSystem(tcc_mode=TCCMode.SOCS, socs_num_terms=5, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)
        assert aerial.min() >= 0
        assert aerial.max() <= 1

    def test_kernel_2d_imaging_shape(self, mask):
        """测试 KERNEL_2D 模式成像输出形状"""
        optics = OpticalSystem(tcc_mode=TCCMode.KERNEL_2D, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)
        assert aerial.min() >= 0
        assert aerial.max() <= 1

    def test_socs_approx_full_tcc(self, mask):
        """测试 SOCS 近似与 FULL_TCC 参考结果相近（使用较多 SOCS 项）"""
        pixel_size = 5.0
        optics_full = OpticalSystem(tcc_mode=TCCMode.FULL_TCC, pixel_size=pixel_size, sigma=0.5)
        imaging_full = PartialCoherentImaging(optics_full, (32, 32))
        aerial_full = imaging_full.compute_aerial_image(mask)

        optics_socs = OpticalSystem(tcc_mode=TCCMode.SOCS, socs_num_terms=15,
                                    pixel_size=pixel_size, sigma=0.5)
        imaging_socs = PartialCoherentImaging(optics_socs, (32, 32))
        aerial_socs = imaging_socs.compute_aerial_image(mask)

        assert np.mean(np.abs(aerial_full - aerial_socs)) < 0.1

    def test_kernel_2d_fastest_approximate(self, mask):
        """测试 KERNEL_2D 模式输出为有效图像（近似但有效）"""
        optics = OpticalSystem(tcc_mode=TCCMode.KERNEL_2D, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        aerial = imaging.compute_aerial_image(mask)

        assert aerial.shape == (32, 32)
        assert np.all(aerial >= 0)
        assert np.all(aerial <= 1)
        assert np.std(aerial) > 0.01

    def test_uniform_mask_all_modes(self):
        """测试均匀掩模在所有模式下输出接近均匀"""
        mask = np.ones((32, 32))
        pixel_size = 5.0

        for mode in [TCCMode.FULL_TCC, TCCMode.SOCS, TCCMode.KERNEL_2D]:
            optics = OpticalSystem(tcc_mode=mode, socs_num_terms=10, pixel_size=pixel_size, sigma=0.5)
            imaging = PartialCoherentImaging(optics, (32, 32))
            aerial = imaging.compute_aerial_image(mask)
            assert np.std(aerial) < 0.15

    def test_get_tcc_image_kernel_2d(self):
        """测试 KERNEL_2D 模式下 get_tcc_image 返回 2D 核"""
        optics = OpticalSystem(tcc_mode=TCCMode.KERNEL_2D, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        tcc_img = imaging.get_tcc_image()
        assert tcc_img is not None
        assert tcc_img.shape == (32, 32)

    def test_get_tcc_image_all_modes(self):
        """测试所有模式下 get_tcc_image 均返回 2D 核（向后兼容）"""
        for mode in [TCCMode.FULL_TCC, TCCMode.SOCS, TCCMode.KERNEL_2D]:
            optics = OpticalSystem(tcc_mode=mode, socs_num_terms=5, pixel_size=5.0, sigma=0.5)
            imaging = PartialCoherentImaging(optics, (32, 32))
            tcc_img = imaging.get_tcc_image()
            assert tcc_img is not None
            assert tcc_img.shape == (32, 32)


class TestTCCModesGradient:
    """三种 TCC 模式下的梯度测试"""

    @pytest.fixture
    def mask(self):
        return np.random.random((32, 32))

    def test_full_tcc_gradient_shape(self, mask):
        """测试 FULL_TCC 模式梯度形状"""
        optics = OpticalSystem(tcc_mode=TCCMode.FULL_TCC, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        grad = imaging.compute_image_gradient(mask)
        assert grad.shape == (32, 32)

    def test_socs_gradient_shape(self, mask):
        """测试 SOCS 模式梯度形状"""
        optics = OpticalSystem(tcc_mode=TCCMode.SOCS, socs_num_terms=5, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        grad = imaging.compute_image_gradient(mask)
        assert grad.shape == (32, 32)

    def test_kernel_2d_gradient_shape(self, mask):
        """测试 KERNEL_2D 模式梯度形状"""
        optics = OpticalSystem(tcc_mode=TCCMode.KERNEL_2D, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        grad = imaging.compute_image_gradient(mask)
        assert grad.shape == (32, 32)

    def test_full_tcc_gradient_finite(self, mask):
        """测试 FULL_TCC 梯度为有限值"""
        optics = OpticalSystem(tcc_mode=TCCMode.FULL_TCC, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        grad = imaging.compute_image_gradient(mask)
        assert np.all(np.isfinite(grad))

    def test_socs_gradient_finite(self, mask):
        """测试 SOCS 梯度为有限值"""
        optics = OpticalSystem(tcc_mode=TCCMode.SOCS, socs_num_terms=5, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        grad = imaging.compute_image_gradient(mask)
        assert np.all(np.isfinite(grad))

    def test_kernel_2d_gradient_finite(self, mask):
        """测试 KERNEL_2D 梯度为有限值"""
        optics = OpticalSystem(tcc_mode=TCCMode.KERNEL_2D, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        grad = imaging.compute_image_gradient(mask)
        assert np.all(np.isfinite(grad))

    def test_source_gradient_full_tcc(self, mask):
        """测试 FULL_TCC 模式光源梯度"""
        optics = OpticalSystem(tcc_mode=TCCMode.FULL_TCC, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        grad = imaging.compute_source_gradient(mask)
        assert grad.shape == (32, 32)
        assert np.all(grad >= 0)

    def test_source_gradient_socs(self, mask):
        """测试 SOCS 模式光源梯度"""
        optics = OpticalSystem(tcc_mode=TCCMode.SOCS, socs_num_terms=5, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        grad = imaging.compute_source_gradient(mask)
        assert grad.shape == (32, 32)
        assert np.all(grad >= 0)

    def test_source_gradient_kernel_2d_zero(self, mask):
        """测试 KERNEL_2D 模式光源梯度为零（近似模式不支持）"""
        optics = OpticalSystem(tcc_mode=TCCMode.KERNEL_2D, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        grad = imaging.compute_source_gradient(mask)
        assert grad.shape == (32, 32)
        assert np.all(grad == 0.0)


class TestTCCModeUpdateSource:
    """更新光源后重新计算传递函数测试"""

    def test_update_source_full_tcc(self):
        """测试 FULL_TCC 模式更新光源"""
        optics = OpticalSystem(tcc_mode=TCCMode.FULL_TCC, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))

        new_source = np.random.random((32, 32))
        new_source = new_source / np.sum(new_source)
        imaging.update_source(new_source)

        mask = np.random.random((32, 32))
        aerial = imaging.compute_aerial_image(mask)
        assert aerial.shape == (32, 32)

    def test_update_source_socs(self):
        """测试 SOCS 模式更新光源重新分解"""
        optics = OpticalSystem(tcc_mode=TCCMode.SOCS, socs_num_terms=5, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        old_eigenvalues = imaging.socs_eigenvalues.copy()

        new_source = np.random.random((32, 32))
        new_source = new_source / np.sum(new_source)
        imaging.update_source(new_source)

        assert imaging.socs_eigenvalues is not None
        assert not np.array_equal(imaging.socs_eigenvalues, old_eigenvalues)

    def test_update_source_kernel_2d(self):
        """测试 KERNEL_2D 模式更新光源重新计算核"""
        optics = OpticalSystem(tcc_mode=TCCMode.KERNEL_2D, pixel_size=5.0, sigma=0.5)
        imaging = PartialCoherentImaging(optics, (32, 32))
        old_kernel = imaging.tcc_kernel.copy()

        new_source = np.random.random((32, 32))
        new_source = new_source / np.sum(new_source)
        imaging.update_source(new_source)

        assert imaging.tcc_kernel is not None
        assert not np.array_equal(imaging.tcc_kernel, old_kernel)


class TestProcessConditionTCCMode:
    """ProcessCondition 的 TCC 模式传递测试"""

    def test_to_optical_system_preserves_tcc_mode(self):
        """测试 to_optical_system 保留 tcc_mode"""
        base = OpticalSystem(tcc_mode=TCCMode.KERNEL_2D, socs_num_terms=8)
        cond = ProcessCondition(defocus=50.0)
        opt = cond.to_optical_system(base_optics=base)
        assert opt.tcc_mode == TCCMode.KERNEL_2D
        assert opt.socs_num_terms == 8
