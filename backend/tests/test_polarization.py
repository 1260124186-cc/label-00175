# -*- coding: utf-8 -*-
"""
偏振与薄膜效应建模单元测试

测试内容包括：
1. Jones 矩阵与 Jones 向量
2. 薄膜堆栈计算
3. 矢量光瞳计算
4. 与现有成像系统的集成
5. 高 NA 浸没式与 EUV 反射系统
"""

import pytest
import numpy as np
from core.polarization import (
    JonesMatrix,
    JonesVector,
    ThinFilmLayer,
    ThinFilmStack,
    VectorPupil,
    compute_polarized_pupil,
    compute_partial_coherent_vectorial,
    create_high_na_immersion_system,
    create_euv_reflective_system,
    compute_polarization_contribution,
    compute_polarization_degree,
    scalar_from_vector_pupil,
    PolarizationComponent,
    STANDARD_MATERIALS,
)
from core.imaging import (
    OpticalSystem,
    PartialCoherentImaging,
    TechnologyNode,
    IlluminationType,
    TCCMode,
)


class TestJonesMatrix:
    """Jones 矩阵测试"""

    def test_identity_matrix(self):
        """测试单位矩阵"""
        jm = JonesMatrix.identity()
        assert jm.matrix.shape == (2, 2)
        assert np.allclose(jm.matrix, np.eye(2))

    def test_polarizer_horizontal(self):
        """测试水平偏振片"""
        jm = JonesMatrix.polarizer(angle_deg=0.0)
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        assert np.allclose(jm.matrix, expected)

    def test_polarizer_vertical(self):
        """测试垂直偏振片"""
        jm = JonesMatrix.polarizer(angle_deg=90.0)
        expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        assert np.allclose(jm.matrix, expected)

    def test_quarter_waveplate(self):
        """测试四分之一波片 - 验证将线偏振转换为圆偏振"""
        jm = JonesMatrix.quarter_waveplate(angle_deg=0.0)
        jv_in = JonesVector.linear_polarization(45.0)
        jv_out = jv_in.apply_matrix(jm)
        assert np.isclose(np.abs(jv_out.e_x), np.abs(jv_out.e_y), atol=1e-10)
        phase_diff = np.angle(jv_out.e_y) - np.angle(jv_out.e_x)
        assert np.isclose(np.abs(phase_diff), np.pi/2, atol=1e-10)

    def test_half_waveplate(self):
        """测试二分之一波片 - 验证偏振方向旋转90度"""
        jm = JonesMatrix.half_waveplate(angle_deg=45.0)
        jv_in = JonesVector.linear_polarization(0.0)
        jv_out = jv_in.apply_matrix(jm)
        assert np.isclose(np.abs(jv_out.e_x), 0.0, atol=1e-10)
        assert np.isclose(np.abs(jv_out.e_y), 1.0, atol=1e-10)

    def test_rotation_matrix(self):
        """测试旋转矩阵"""
        jm = JonesMatrix.rotation(angle_deg=90.0)
        expected = np.array([[0, -1], [1, 0]], dtype=np.complex128)
        assert np.allclose(jm.matrix, expected)

    def test_matrix_multiplication(self):
        """测试矩阵乘法"""
        jm1 = JonesMatrix.polarizer(0.0)
        jm2 = JonesMatrix.polarizer(90.0)
        jm_combined = jm1 @ jm2
        assert np.allclose(jm_combined.matrix, np.zeros((2, 2), dtype=np.complex128))

    def test_apply_to_vector(self):
        """测试应用到 Jones 向量"""
        jm = JonesMatrix.polarizer(0.0)
        jv = JonesVector.linear_polarization(0.0)
        ex, ey = jm.apply(jv.e_x, jv.e_y)
        assert np.isclose(ex, 1.0)
        assert np.isclose(ey, 0.0)

    def test_crossed_polarizers(self):
        """测试正交偏振片消光"""
        horizontal = JonesMatrix.polarizer(0.0)
        vertical = JonesMatrix.polarizer(90.0)
        incident = JonesVector.linear_polarization(0.0)
        
        after_horizontal = incident.apply_matrix(horizontal)
        after_vertical = after_horizontal.apply_matrix(vertical)
        
        assert after_vertical.intensity < 1e-10


class TestJonesVector:
    """Jones 向量测试"""

    def test_linear_polarization_0(self):
        """测试 0 度线偏振"""
        jv = JonesVector.linear_polarization(0.0)
        assert np.isclose(jv.e_x, 1.0)
        assert np.isclose(jv.e_y, 0.0)
        assert np.isclose(jv.intensity, 1.0)

    def test_linear_polarization_45(self):
        """测试 45 度线偏振"""
        jv = JonesVector.linear_polarization(45.0)
        expected = 1.0 / np.sqrt(2)
        assert np.isclose(jv.e_x, expected)
        assert np.isclose(jv.e_y, expected)
        assert np.isclose(jv.intensity, 1.0)

    def test_left_circular(self):
        """测试左旋圆偏振"""
        jv = JonesVector.left_circular()
        expected_amp = 1.0 / np.sqrt(2)
        assert np.isclose(np.abs(jv.e_x), expected_amp)
        assert np.isclose(np.abs(jv.e_y), expected_amp)
        phase_diff = np.angle(jv.e_y) - np.angle(jv.e_x)
        assert np.isclose(np.abs(phase_diff), np.pi/2, atol=1e-10)
        assert np.isclose(jv.intensity, 1.0)

    def test_right_circular(self):
        """测试右旋圆偏振"""
        jv = JonesVector.right_circular()
        expected_amp = 1.0 / np.sqrt(2)
        assert np.isclose(np.abs(jv.e_x), expected_amp)
        assert np.isclose(np.abs(jv.e_y), expected_amp)
        phase_diff = np.angle(jv.e_y) - np.angle(jv.e_x)
        assert np.isclose(np.abs(phase_diff), np.pi/2, atol=1e-10)
        assert np.isclose(jv.intensity, 1.0)

    def test_left_right_circular_orthogonal(self):
        """测试左右旋圆偏振正交"""
        jv_l = JonesVector.left_circular()
        jv_r = JonesVector.right_circular()
        inner_product = np.conj(jv_l.e_x) * jv_r.e_x + np.conj(jv_l.e_y) * jv_r.e_y
        assert np.isclose(np.abs(inner_product), 0.0, atol=1e-10)

    def test_normalize(self):
        """测试归一化"""
        jv = JonesVector(2.0 + 0.0j, 0.0 + 0.0j)
        normalized = jv.normalize()
        assert np.isclose(normalized.intensity, 1.0)
        assert np.isclose(normalized.e_x, 1.0)

    def test_apply_matrix(self):
        """测试应用矩阵"""
        jv = JonesVector.linear_polarization(0.0)
        jm = JonesMatrix.rotation(90.0)
        result = jv.apply_matrix(jm)
        assert np.isclose(result.e_x, 0.0)
        assert np.isclose(result.e_y, 1.0)


class TestThinFilmStack:
    """薄膜堆栈测试"""

    def test_single_layer(self):
        """测试单层薄膜"""
        layer = ThinFilmLayer(thickness_nm=10.0, material="sio2")
        assert layer.thickness_nm == 10.0
        assert layer.material == "sio2"

    def test_arf_antireflective(self):
        """测试 ArF 抗反射涂层"""
        stack = ThinFilmStack.arf_antireflective(wavelength_nm=193.0)
        assert len(stack.layers) == 2
        assert stack.n_superstrate is not None
        assert stack.n_substrate is not None

    def test_euv_multilayer(self):
        """测试 EUV 多层反射膜（包含 Ru 覆盖层）"""
        stack = ThinFilmStack.euv_multilayer(num_pairs=10, wavelength_nm=13.5)
        assert len(stack.layers) >= 20
        assert stack.n_superstrate is not None
        assert stack.n_substrate is not None

    def test_normal_incidence_reflection(self):
        """测试正入射反射率"""
        stack = ThinFilmStack.arf_antireflective(wavelength_nm=193.0)
        result = stack.compute_reflection_transmission(
            wavelength_nm=193.0, theta_rad=0.0, polarization="s"
        )
        assert "r" in result
        assert "t" in result
        assert "R" in result
        assert "T" in result
        assert np.abs(result["r"]) <= 1.0 + 1e-10
        assert 0.0 <= result["R"] <= 1.0 + 1e-10
        assert 0.0 <= result["T"] <= 1.0 + 1e-10

    def test_energy_conservation(self):
        """测试能量守恒（R + T = 1 对于无吸收介质）"""
        stack = ThinFilmStack(
            layers=[
                ThinFilmLayer(thickness_nm=25.0, material="mgf2"),
            ],
            n_superstrate="air",
            n_substrate="sio2"
        )
        result = stack.compute_reflection_transmission(
            wavelength_nm=500.0, theta_rad=0.0, polarization="s"
        )
        R = result["R"]
        T = result["T"]
        assert np.abs(R + T - 1.0) < 0.01

    def test_s_polarization_independence_at_normal(self):
        """测试正入射时 s/p 偏振等价"""
        stack = ThinFilmStack.arf_antireflective(wavelength_nm=193.0)
        result = stack.compute_reflection_transmission(
            wavelength_nm=193.0, theta_rad=0.0, polarization="unpolarized"
        )
        assert np.isclose(result["rs"], result["rp"], atol=1e-10)
        assert np.isclose(result["ts"], result["tp"], atol=1e-10)

    def test_spectrum_computation(self):
        """测试光谱计算"""
        stack = ThinFilmStack.arf_antireflective(wavelength_nm=193.0)
        wavelengths = np.linspace(180.0, 210.0, 31)
        spectrum = stack.compute_spectrum(wavelengths, theta_rad=0.0)
        assert "Rs" in spectrum
        assert spectrum["Rs"].shape == wavelengths.shape

    def test_angular_response(self):
        """测试角度响应计算"""
        stack = ThinFilmStack.euv_multilayer(num_pairs=20, wavelength_nm=13.5)
        thetas = np.linspace(0.0, np.pi/6, 19)
        response = stack.compute_angular_response(13.5, thetas)
        assert "Rs" in response
        assert response["Rs"].shape == thetas.shape


class TestVectorPupil:
    """矢量光瞳测试"""

    def test_create_high_na_immersion(self):
        """测试创建高 NA 浸没式系统"""
        vp = create_high_na_immersion_system(
            wavelength_nm=193.0,
            na=1.35,
            grid_size=(64, 64),
        )
        assert vp.wavelength_nm == 193.0
        assert vp.na == 1.35
        assert vp.grid_size == (64, 64)
        assert vp.n_immersion.real > 1.0

    def test_create_euv_reflective(self):
        """测试创建 EUV 反射系统"""
        vp = create_euv_reflective_system(
            wavelength_nm=13.5,
            na=0.33,
            grid_size=(64, 64),
        )
        assert vp.wavelength_nm == 13.5
        assert vp.na == 0.33
        assert vp.mask_stack is not None

    def test_s_polarization_basis(self):
        """测试 s 偏振基矢"""
        vp = create_high_na_immersion_system(
            wavelength_nm=193.0, na=1.35, grid_size=(64, 64)
        )
        basis = vp.s_polarization_basis()
        assert "Ex" in basis
        assert "Ey" in basis
        assert "Ez" in basis
        assert basis["Ex"].shape == (64, 64)

    def test_p_polarization_basis(self):
        """测试 p 偏振基矢"""
        vp = create_high_na_immersion_system(
            wavelength_nm=193.0, na=1.35, grid_size=(64, 64)
        )
        basis = vp.p_polarization_basis()
        assert "Ex" in basis
        assert "Ey" in basis
        assert "Ez" in basis
        assert basis["Ez"].shape == (64, 64)

    def test_compute_vector_pupil(self):
        """测试计算矢量光瞳"""
        vp = create_high_na_immersion_system(
            wavelength_nm=193.0, na=1.35, grid_size=(64, 64)
        )
        pupil = vp.compute_vector_pupil(defocus_nm=50.0)
        assert "Ex" in pupil
        assert "Ey" in pupil
        assert "Ez" in pupil
        assert pupil["Ex"].shape == (64, 64)

    def test_thin_film_modulation(self):
        """测试薄膜调制"""
        vp = create_euv_reflective_system(
            wavelength_nm=13.5, na=0.33, grid_size=(64, 64)
        )
        modulation = vp.compute_thin_film_modulation(is_reflection=True)
        assert "s_mod" in modulation
        assert "p_mod" in modulation
        assert modulation["s_mod"].shape == (64, 64)

    def test_propagate_to_image(self):
        """测试传播到像面"""
        vp = create_high_na_immersion_system(
            wavelength_nm=193.0, na=1.35, grid_size=(64, 64)
        )
        mask = np.random.rand(64, 64)
        pupil = vp.compute_vector_pupil()
        intensity = vp.propagate_to_image(mask, pupil)
        assert intensity.shape == (64, 64)
        assert np.all(intensity >= 0.0)


class TestPolarizedPupilComputation:
    """偏振光瞳计算测试"""

    def test_compute_polarized_pupil(self):
        """测试计算偏振光瞳"""
        grid_size = 64
        fx = np.fft.fftfreq(grid_size, 1.0)
        fy = np.fft.fftfreq(grid_size, 1.0)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        wavelength = 193.0
        na = 1.35
        cutoff = na / wavelength
        
        result = compute_polarized_pupil(
            fx=fx_grid,
            fy=fy_grid,
            wavelength_nm=wavelength,
            na=na,
            cutoff=cutoff,
            defocus_nm=0.0,
            zernike_phase=np.zeros_like(fx_grid),
            incident_polarization=JonesVector.linear_polarization(0.0),
        )
        
        assert "Ex" in result
        assert "Ey" in result
        assert "Ez" in result
        assert "pupil_scalar" in result
        assert result["pupil_scalar"].shape == (grid_size, grid_size)

    def test_scalar_from_vector_pupil(self):
        """测试从矢量光瞳获取标量光瞳"""
        grid_size = 64
        fx = np.fft.fftfreq(grid_size, 1.0)
        fy = np.fft.fftfreq(grid_size, 1.0)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        wavelength = 193.0
        na = 1.35
        cutoff = na / wavelength
        
        result = compute_polarized_pupil(
            fx=fx_grid,
            fy=fy_grid,
            wavelength_nm=wavelength,
            na=na,
            cutoff=cutoff,
            defocus_nm=0.0,
            zernike_phase=np.zeros_like(fx_grid),
            incident_polarization=JonesVector.linear_polarization(0.0),
        )
        
        scalar = scalar_from_vector_pupil(result)
        assert scalar.shape == (grid_size, grid_size)
        assert np.all(np.abs(scalar) >= 0.0)

    def test_compute_polarization_contribution(self):
        """测试偏振分量贡献"""
        grid_size = 64
        fx = np.fft.fftfreq(grid_size, 1.0)
        fy = np.fft.fftfreq(grid_size, 1.0)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        wavelength = 193.0
        na = 1.35
        cutoff = na / wavelength
        
        result = compute_polarized_pupil(
            fx=fx_grid,
            fy=fy_grid,
            wavelength_nm=wavelength,
            na=na,
            cutoff=cutoff,
            defocus_nm=0.0,
            zernike_phase=np.zeros_like(fx_grid),
            incident_polarization=JonesVector.linear_polarization(0.0),
        )
        
        contribution = compute_polarization_contribution(result)
        assert "Ex_contrib" in contribution
        assert "Ey_contrib" in contribution
        assert "Ez_contrib" in contribution
        assert "total" in contribution

    def test_compute_polarization_degree(self):
        """测试偏振度计算"""
        grid_size = 64
        fx = np.fft.fftfreq(grid_size, 1.0)
        fy = np.fft.fftfreq(grid_size, 1.0)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        wavelength = 193.0
        na = 1.35
        cutoff = na / wavelength
        
        result = compute_polarized_pupil(
            fx=fx_grid,
            fy=fy_grid,
            wavelength_nm=wavelength,
            na=na,
            cutoff=cutoff,
            defocus_nm=0.0,
            zernike_phase=np.zeros_like(fx_grid),
            incident_polarization=JonesVector.linear_polarization(0.0),
        )
        
        dop = compute_polarization_degree(result)
        assert isinstance(dop, (float, np.floating))
        assert 0.0 <= dop <= 1.0


class TestPartialCoherentVectorial:
    """部分相干矢量成像测试"""

    def test_compute_partial_coherent_vectorial(self):
        """测试部分相干矢量成像"""
        grid_size = 64
        mask = np.zeros((grid_size, grid_size))
        mask[20:44, 20:44] = 1.0
        
        fx = np.fft.fftfreq(grid_size, 1.0)
        fy = np.fft.fftfreq(grid_size, 1.0)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        wavelength = 193.0
        na = 1.35
        cutoff = na / wavelength
        
        source = np.zeros_like(fx_grid)
        rho = np.sqrt(fx_grid**2 + fy_grid**2) / cutoff
        source[rho <= 0.3] = 1.0
        source = source / np.sum(source)
        
        vector_pupils = compute_polarized_pupil(
            fx=fx_grid,
            fy=fy_grid,
            wavelength_nm=wavelength,
            na=na,
            cutoff=cutoff,
            defocus_nm=0.0,
            zernike_phase=np.zeros_like(fx_grid),
            incident_polarization=JonesVector.linear_polarization(0.0),
        )
        
        dfx = 1.0 / (grid_size * 1.0)
        dfy = 1.0 / (grid_size * 1.0)
        
        intensity = compute_partial_coherent_vectorial(
            mask=mask,
            source=source,
            vector_pupils=vector_pupils,
            dfx=dfx,
            dfy=dfy,
        )
        
        assert intensity.shape == (grid_size, grid_size)
        assert np.all(intensity >= 0.0)
        assert np.sum(intensity) > 0.0


class TestImagingIntegration:
    """与现有成像系统的集成测试"""

    def test_optical_system_vector_pupil_flag(self):
        """测试光学系统矢量光瞳标志"""
        optics = OpticalSystem(
            wavelength=193.0,
            na=1.35,
            use_vector_pupil=True,
            use_mask_coating=True,
            n_immersion=1.437,
        )
        assert optics.use_vector_pupil is True
        assert optics.use_mask_coating is True
        assert optics.n_immersion == 1.437

    def test_optical_system_auto_config_immersion(self):
        """测试高 NA 系统自动配置浸没折射率"""
        optics = OpticalSystem(
            technology_node=TechnologyNode.DUV_ARF,
            na=1.35,
            use_vector_pupil=True,
        )
        assert np.isclose(optics.n_immersion, 1.437)

    def test_optical_system_euv_config(self):
        """测试 EUV 系统自动配置"""
        optics = OpticalSystem(
            technology_node=TechnologyNode.EUV,
            use_vector_pupil=True,
            use_mask_coating=True,
        )
        assert optics.wavelength == 13.5
        assert optics.na == 0.33
        assert optics.mask_stack is not None
        assert optics.vector_pupil is not None

    def test_optical_system_incident_polarization(self):
        """测试入射偏振态配置"""
        optics = OpticalSystem(
            use_vector_pupil=True,
            incident_polarization_angle=45.0,
        )
        assert optics.incident_polarization is not None
        expected = 1.0 / np.sqrt(2)
        assert np.isclose(optics.incident_polarization.e_x, expected)
        assert np.isclose(optics.incident_polarization.e_y, expected)

    def test_partial_coherent_imaging_scalar(self):
        """测试标量成像（向后兼容）"""
        optics = OpticalSystem(
            wavelength=193.0,
            na=1.35,
            sigma=0.3,
            use_vector_pupil=False,
            tcc_mode=TCCMode.KERNEL_2D,
        )
        
        image_size = (64, 64)
        mask = np.zeros(image_size)
        mask[24:40, 24:40] = 1.0
        
        imaging = PartialCoherentImaging(optics, image_size)
        aerial = imaging.compute_aerial_image(mask)
        
        assert aerial.shape == image_size
        assert np.all(aerial >= 0.0)
        assert np.max(aerial) <= 1.0 + 1e-10

    def test_partial_coherent_imaging_vectorial(self):
        """测试矢量成像"""
        optics = OpticalSystem(
            wavelength=193.0,
            na=1.35,
            sigma=0.3,
            use_vector_pupil=True,
            use_mask_coating=True,
            n_immersion=1.437,
            tcc_mode=TCCMode.KERNEL_2D,
        )
        
        image_size = (64, 64)
        mask = np.zeros(image_size)
        mask[24:40, 24:40] = 1.0
        
        imaging = PartialCoherentImaging(optics, image_size)
        aerial = imaging.compute_aerial_image(mask)
        
        assert aerial.shape == image_size
        assert np.all(aerial >= 0.0)
        assert np.max(aerial) <= 1.0 + 1e-10

    def test_scalar_vector_consistency(self):
        """测试标量和矢量模型的基本一致性（低 NA 下两者应相似）"""
        image_size = (64, 64)
        mask = np.zeros(image_size)
        mask[20:44, 20:44] = 1.0
        mask = mask / np.max(mask)
        
        optics_scalar = OpticalSystem(
            wavelength=193.0,
            na=1.2,
            sigma=0.3,
            use_vector_pupil=False,
            tcc_mode=TCCMode.KERNEL_2D,
        )
        imaging_scalar = PartialCoherentImaging(optics_scalar, image_size)
        aerial_scalar = imaging_scalar.compute_aerial_image(mask)
        
        optics_vector = OpticalSystem(
            wavelength=193.0,
            na=1.2,
            sigma=0.3,
            use_vector_pupil=True,
            use_mask_coating=False,
            tcc_mode=TCCMode.KERNEL_2D,
        )
        imaging_vector = PartialCoherentImaging(optics_vector, image_size)
        aerial_vector = imaging_vector.compute_aerial_image(mask)
        
        assert aerial_scalar.shape == aerial_vector.shape
        assert np.all(aerial_scalar >= 0.0)
        assert np.all(aerial_vector >= 0.0)
        
        if np.std(aerial_scalar) > 1e-10 and np.std(aerial_vector) > 1e-10:
            correlation = np.corrcoef(aerial_scalar.flatten(), aerial_vector.flatten())[0, 1]
            if not np.isnan(correlation):
                assert correlation > 0.90
        else:
            mean_scalar = np.mean(aerial_scalar)
            mean_vector = np.mean(aerial_vector)
            assert np.isclose(mean_scalar, mean_vector, rtol=0.1)


class TestStandardMaterials:
    """标准材料库测试"""

    def test_materials_exist(self):
        """测试材料库包含必要材料"""
        assert "sio2" in STANDARD_MATERIALS
        assert "cr" in STANDARD_MATERIALS
        assert "ta2o5" in STANDARD_MATERIALS
        assert "mgf2" in STANDARD_MATERIALS
        assert "mo" in STANDARD_MATERIALS
        assert "si" in STANDARD_MATERIALS
        assert "water" in STANDARD_MATERIALS
        assert "air" in STANDARD_MATERIALS
        assert "ru" in STANDARD_MATERIALS

    def test_water_refractive_index_193nm(self):
        """测试 193nm 下水的折射率"""
        n_water = STANDARD_MATERIALS["water"].get_n(193.0)
        assert np.isclose(n_water.real, 1.437, atol=0.01)

    def test_silicon_euv_13_5nm(self):
        """测试 13.5nm 下硅的折射率"""
        n_si = STANDARD_MATERIALS["si"].get_n(13.5)
        assert n_si.imag < 0.0

    def test_material_dispersion_interface(self):
        """测试材料色散接口"""
        for mat_name, mat_disp in STANDARD_MATERIALS.items():
            n = mat_disp.get_n(193.0)
            assert isinstance(n, (complex, np.complex128))
            n2 = mat_disp.get_n(13.5)
            assert isinstance(n2, (complex, np.complex128))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
