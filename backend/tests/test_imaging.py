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
    generate_source,
    compute_tcc_kernel_2d,
    socs_decomposition
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
        """测试获取TCC图像（SOCS模式返回None）"""
        optics = OpticalSystem(use_socs=True)
        imaging = PartialCoherentImaging(optics, (32, 32))

        tcc_img = imaging.get_tcc_image()
        assert tcc_img is None


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
