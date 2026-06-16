# -*- coding: utf-8 -*-
"""
FDTD 求解器单元测试

测试 MeepFDTDSolver 的完整功能，包括：
1. 配置参数验证
2. 回退模式（无 meep 时）
3. 近场到远场变换
4. 远场到空间像变换
5. 3D 掩模几何结构
6. 偏振处理
7. 与光学系统参数的集成
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from core.rigorous_sim import (
    FDTDConfig,
    MeepFDTDSolver,
    Polarization,
    SimulationBackend,
    simulate,
)
from core.imaging import OpticalSystem
from core.test_structures import (
    LineSpaceParams,
    create_line_space,
    ContactHoleParams,
    create_contact_hole,
)


class TestFDTDConfig:
    """FDTD 配置参数测试"""

    def test_default_parameters(self):
        """测试默认参数"""
        cfg = FDTDConfig()

        assert cfg.grid_resolution_nm == 0.5
        assert cfg.pml_thickness_nm == 200.0
        assert cfg.total_time_steps == 2000
        assert cfg.courant_factor == 0.9
        assert cfg.use_meep_if_available is True

        assert cfg.n_substrate == 1.56 + 0.0j
        assert cfg.n_absorber == 3.28 - 4.32j
        assert cfg.n_superstrate == 1.44 + 0.0j

        assert cfg.mask_thickness_nm == 70.0
        assert cfg.substrate_thickness_nm == 500.0
        assert cfg.superstrate_thickness_nm == 500.0

        assert cfg.illumination_theta_deg == 0.0
        assert cfg.illumination_phi_deg == 0.0
        assert cfg.polarization == Polarization.UNPOLARIZED

        assert cfg.pupil_filter is True
        assert cfg.max_far_field_orders == 50

    def test_custom_parameters(self):
        """测试自定义参数"""
        cfg = FDTDConfig(
            grid_resolution_nm=0.25,
            mask_thickness_nm=100.0,
            illumination_theta_deg=15.0,
            polarization=Polarization.TE,
            pupil_filter=False,
        )

        assert cfg.grid_resolution_nm == 0.25
        assert cfg.mask_thickness_nm == 100.0
        assert cfg.illumination_theta_deg == 15.0
        assert cfg.polarization == Polarization.TE
        assert cfg.pupil_filter is False

    def test_material_epsilon_conversion(self):
        """测试材料折射率到介电常数的转换"""
        cfg = FDTDConfig(
            n_substrate=1.5 + 0.1j,
            n_absorber=3.0 - 4.0j,
            n_superstrate=1.44 + 0.0j,
        )

        eps_sub = complex(cfg.n_substrate) ** 2
        eps_abs = complex(cfg.n_absorber) ** 2
        eps_sup = complex(cfg.n_superstrate) ** 2

        assert abs(eps_sub.real - 2.24) < 1e-10
        assert abs(eps_sub.imag - 0.3) < 1e-10
        assert abs(eps_abs.real - (-7.0)) < 1e-10
        assert abs(eps_abs.imag - (-24.0)) < 1e-10
        assert abs(eps_sup.real - 2.0736) < 1e-10


class TestMeepFDTDSolverFallback:
    """无 meep 时的回退模式测试"""

    def test_meep_not_available(self):
        """测试 meep 不可用时的检测"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)

        assert solver.meep_available is False

    def test_fallback_simulation_shape(self):
        """测试回退仿真输出形状"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem()

        mask = np.random.random((32, 32))
        aerial, extra = solver.simulate_aerial(mask, optics)

        assert aerial.shape == (32, 32)
        assert aerial.dtype == np.float64
        assert extra["fdtd_fallback"] is True

    def test_fallback_simulation_range(self):
        """测试回退仿真值范围 [0, 1]"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem()

        mask = np.ones((32, 32))
        mask[8:24, 8:24] = 0.0
        aerial, extra = solver.simulate_aerial(mask, optics)

        assert np.all(aerial >= 0.0)
        assert np.all(aerial <= 1.0)
        assert np.nanmax(aerial) > 0.0

    def test_fallback_high_na_correction(self):
        """测试高 NA 时的矢量修正"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)

        optics_low_na = OpticalSystem(na=0.5)
        optics_high_na = OpticalSystem(na=1.35)

        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90, pixel_size=2.0
        ))

        aerial_low, _ = solver.simulate_aerial(mask, optics_low_na)
        aerial_high, _ = solver.simulate_aerial(mask, optics_high_na)

        diff_low = np.abs(aerial_low - np.mean(aerial_low))
        diff_high = np.abs(aerial_high - np.mean(aerial_high))

        assert np.mean(diff_high) >= np.mean(diff_low) * 0.9

    def test_fallback_uniform_mask(self):
        """测试均匀掩模回退仿真"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem()

        mask = np.ones((32, 32))
        aerial, extra = solver.simulate_aerial(mask, optics)

        assert np.std(aerial) < 0.1
        assert extra["fdtd_fallback"] is True


class TestNearToFarFieldTransform:
    """近场到远场变换测试"""

    def test_ntff_shape_preservation(self):
        """测试 NTFF 形状保持"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)

        nx, ny = 32, 32
        Ex = np.random.random((ny, nx)).astype(np.complex128)
        Ey = np.random.random((ny, nx)).astype(np.complex128)
        Ez = np.random.random((ny, nx)).astype(np.complex128)

        result = solver._near_to_far_field(
            Ex, Ey, Ez, sx_nm=1000, sy_nm=1000,
            wavelength_nm=193, cfg=cfg
        )

        assert result["Efar"].shape == (ny, nx)
        assert result["KX"].shape == (ny, nx)
        assert result["KY"].shape == (ny, nx)
        assert result["kz"].shape == (ny, nx)

    def test_ntff_evanescent_filtering(self):
        """测试倏逝波过滤"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)

        nx, ny = 64, 64
        x = np.linspace(-500, 500, nx)
        y = np.linspace(-500, 500, ny)
        X, Y = np.meshgrid(x, y)

        wavelength = 193.0
        k0 = 2 * np.pi / wavelength
        kx_prop = 0.3 * k0
        ky_prop = 0.4 * k0

        Ex = np.exp(1j * (kx_prop * X + ky_prop * Y)).astype(np.complex128)
        Ey = np.zeros_like(Ex)
        Ez = np.zeros_like(Ex)

        result = solver._near_to_far_field(
            Ex, Ey, Ez, sx_nm=1000, sy_nm=1000,
            wavelength_nm=wavelength, cfg=cfg
        )

        Efar = result["Efar"]
        kz = result["kz"]

        propagating = np.real(kz) > 0
        assert np.sum(propagating) > 0
        assert np.any(Efar[propagating] > 0)

    def test_ntff_normalization(self):
        """测试 NTFF 归一化"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)

        nx, ny = 32, 32
        Ex = np.ones((ny, nx), dtype=np.complex128)
        Ey = np.ones((ny, nx), dtype=np.complex128)
        Ez = np.ones((ny, nx), dtype=np.complex128)

        result = solver._near_to_far_field(
            Ex, Ey, Ez, sx_nm=1000, sy_nm=1000,
            wavelength_nm=193, cfg=cfg
        )

        Efar = result["Efar"]
        assert np.nanmax(Efar) <= 1.0 + 1e-10
        assert np.nanmax(Efar) > 0.0

    def test_ntff_windowing(self):
        """测试 NTFF 加窗减少频谱泄露"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)

        nx, ny = 64, 64
        Ex = np.zeros((ny, nx), dtype=np.complex128)
        Ex[20:44, 20:44] = 1.0
        Ey = np.zeros_like(Ex)
        Ez = np.zeros_like(Ex)

        result = solver._near_to_far_field(
            Ex, Ey, Ez, sx_nm=1000, sy_nm=1000,
            wavelength_nm=193, cfg=cfg
        )

        Efar = result["Efar"]
        Efar_db = 20 * np.log10(np.abs(Efar) + 1e-30)
        dynamic_range = np.nanmax(Efar_db) - np.nanmin(Efar_db)

        assert dynamic_range > 30.0


class TestFarFieldToAerialImage:
    """远场到空间像变换测试"""

    def test_aerial_image_shape(self):
        """测试空间像输出形状"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem()

        nx, ny = 32, 32
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, 2.0))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, 2.0))
        KX, KY = np.meshgrid(kx, ky)

        far_field = {
            "Efar": np.ones((ny, nx), dtype=np.float64),
            "KX": KX,
            "KY": KY,
        }

        aerial = solver._far_field_to_aerial(far_field, optics, cfg)

        assert aerial.shape == (ny, nx)
        assert aerial.dtype == np.float64

    def test_aerial_image_normalization(self):
        """测试空间像归一化"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem()

        nx, ny = 32, 32
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, 2.0))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, 2.0))
        KX, KY = np.meshgrid(kx, ky)

        cutoff = 2 * np.pi * optics.na / optics.wavelength
        pupil_mask = (KX ** 2 + KY ** 2) <= cutoff ** 2

        far_field = {
            "Efar": pupil_mask.astype(np.float64),
            "KX": KX,
            "KY": KY,
        }

        aerial = solver._far_field_to_aerial(far_field, optics, cfg)

        assert np.nanmax(aerial) <= 1.0 + 1e-10
        assert np.nanmax(aerial) > 0.0
        assert np.all(aerial >= 0.0)

    def test_pupil_filtering(self):
        """测试光瞳滤波"""
        cfg = FDTDConfig(pupil_filter=True)
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem(na=0.5)

        nx, ny = 64, 64
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, 1.0))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, 1.0))
        KX, KY = np.meshgrid(kx, ky)

        Efar = np.ones((ny, nx), dtype=np.float64)

        far_field = {
            "Efar": Efar,
            "KX": KX,
            "KY": KY,
        }

        aerial_filtered = solver._far_field_to_aerial(far_field, optics, cfg)

        cfg_no_filter = FDTDConfig(pupil_filter=False)
        solver_no_filter = MeepFDTDSolver(cfg_no_filter)
        aerial_unfiltered = solver_no_filter._far_field_to_aerial(
            far_field, optics, cfg_no_filter
        )

        assert np.std(aerial_filtered) < np.std(aerial_unfiltered)

    def test_defocus_phase(self):
        """测试离焦相位应用"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)
        optics_in_focus = OpticalSystem(defocus=0.0, na=1.35, wavelength=193.0)
        optics_out_focus = OpticalSystem(defocus=50.0, na=1.35, wavelength=193.0)

        nx, ny = 128, 128
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, 10.0))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, 10.0))
        KX, KY = np.meshgrid(kx, ky)

        cutoff = 2 * np.pi * optics_in_focus.na / optics_in_focus.wavelength

        Efar = np.zeros((ny, nx), dtype=np.float64)
        amplitudes = [1.0, 0.8, 0.6, 0.9]
        for idx, (kx_val, ky_val) in enumerate([
            (0.2 * cutoff, 0.1 * cutoff),
            (-0.3 * cutoff, 0.15 * cutoff),
            (0.1 * cutoff, -0.25 * cutoff),
            (-0.15 * cutoff, -0.2 * cutoff),
        ]):
            i = np.argmin(np.abs(kx - kx_val))
            j = np.argmin(np.abs(ky - ky_val))
            Efar[j, i] = amplitudes[idx]

        far_field = {
            "Efar": Efar,
            "KX": KX,
            "KY": KY,
        }

        aerial_in = solver._far_field_to_aerial(far_field, optics_in_focus, cfg)
        aerial_out = solver._far_field_to_aerial(far_field, optics_out_focus, cfg)

        diff = np.abs(aerial_in - aerial_out)
        assert np.max(diff) > 0.001

    def test_zernike_aberrations(self):
        """测试 Zernike 像差应用"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)

        optics_no_aberration = OpticalSystem(na=1.35, wavelength=193.0)
        optics_with_aberration = OpticalSystem(
            na=1.35, wavelength=193.0,
            zernike_coefficients={4: 0.5}
        )

        nx, ny = 128, 128
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, 10.0))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, 10.0))
        KX, KY = np.meshgrid(kx, ky)

        cutoff = 2 * np.pi * optics_no_aberration.na / optics_no_aberration.wavelength

        Efar = np.zeros((ny, nx), dtype=np.float64)
        amplitudes = [1.0, 0.7, 0.85, 0.9]
        for idx, (kx_val, ky_val) in enumerate([
            (0.2 * cutoff, 0.1 * cutoff),
            (-0.3 * cutoff, 0.15 * cutoff),
            (0.1 * cutoff, -0.25 * cutoff),
            (-0.15 * cutoff, -0.2 * cutoff),
        ]):
            i = np.argmin(np.abs(kx - kx_val))
            j = np.argmin(np.abs(ky - ky_val))
            Efar[j, i] = amplitudes[idx]

        far_field = {
            "Efar": Efar,
            "KX": KX,
            "KY": KY,
        }

        aerial_no_ab = solver._far_field_to_aerial(
            far_field, optics_no_aberration, cfg
        )
        aerial_with_ab = solver._far_field_to_aerial(
            far_field, optics_with_aberration, cfg
        )

        diff = np.abs(aerial_no_ab - aerial_with_ab)
        assert np.max(diff) > 0.001


class Test3DMaskGeometry:
    """3D 掩模几何结构测试"""

    @pytest.mark.skipif(not hasattr(MeepFDTDSolver(FDTDConfig()), '_meep') or
                       MeepFDTDSolver(FDTDConfig())._meep is None,
                    reason="meep not available")
    def test_geometry_creation(self):
        """测试 3D 几何结构创建"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)

        mp = MagicMock()
        mp.Vector3 = MagicMock(side_effect=lambda x, y, z: (x, y, z))
        mp.inf = float('inf')
        mp.Block = MagicMock()
        mp.Medium = MagicMock()

        mask = np.zeros((8, 8))
        mask[2:6, 2:6] = 1.0

        geometry = solver._build_3d_mask_geometry(
            mask, pixel_size_nm=10.0,
            z_substrate_nm=500.0, z_mask_nm=70.0, mp=mp
        )

        assert isinstance(geometry, list)
        assert len(geometry) >= 3

        assert mp.Block.call_count >= 3
        assert mp.Medium.call_count >= 3

    @pytest.mark.skipif(not hasattr(MeepFDTDSolver(FDTDConfig()), '_meep') or
                       MeepFDTDSolver(FDTDConfig())._meep is None,
                    reason="meep not available")
    def test_geometry_material_assignment(self):
        """测试材料分配正确性"""
        cfg = FDTDConfig(
            n_substrate=1.5,
            n_absorber=3.0 - 4.0j,
            n_superstrate=1.44,
        )
        solver = MeepFDTDSolver(cfg)

        mp = MagicMock()
        mp.Vector3 = MagicMock(side_effect=lambda x, y, z: (x, y, z))
        mp.inf = float('inf')
        mp.Block = MagicMock()
        mp.Medium = MagicMock()

        mask = np.zeros((4, 4))
        mask[1:3, 1:3] = 1.0

        geometry = solver._build_3d_mask_geometry(
            mask, pixel_size_nm=10.0,
            z_substrate_nm=500.0, z_mask_nm=70.0, mp=mp
        )

        expected_eps_sup = 1.44 ** 2
        expected_eps_sub = 1.5 ** 2
        expected_eps_abs_real = (3.0 ** 2 - (-4.0) ** 2)
        expected_eps_abs_imag = 2 * 3.0 * (-4.0)

        calls = mp.Medium.call_args_list
        assert len(calls) >= 3


class TestPolarizationHandling:
    """偏振处理测试"""

    def test_te_polarization_config(self):
        """测试 TE 偏振配置"""
        cfg = FDTDConfig(polarization=Polarization.TE)
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem()

        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        aerial, extra = solver.simulate_aerial(mask, optics)

        assert extra["fdtd_fallback"] is True
        assert aerial.shape == (32, 32)

    def test_tm_polarization_config(self):
        """测试 TM 偏振配置"""
        cfg = FDTDConfig(polarization=Polarization.TM)
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem()

        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        aerial, extra = solver.simulate_aerial(mask, optics)

        assert extra["fdtd_fallback"] is True
        assert aerial.shape == (32, 32)

    def test_unpolarized_config(self):
        """测试非偏振配置"""
        cfg = FDTDConfig(polarization=Polarization.UNPOLARIZED)
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem()

        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        aerial, extra = solver.simulate_aerial(mask, optics)

        assert extra["fdtd_fallback"] is True
        assert aerial.shape == (32, 32)

    def test_polarization_difference(self):
        """测试不同偏振的结果差异"""
        optics = OpticalSystem(na=1.35)
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90,
            orientation='horizontal'
        ))

        cfg_te = FDTDConfig(polarization=Polarization.TE)
        cfg_tm = FDTDConfig(polarization=Polarization.TM)

        solver_te = MeepFDTDSolver(cfg_te)
        solver_tm = MeepFDTDSolver(cfg_tm)

        aerial_te, _ = solver_te.simulate_aerial(mask, optics)
        aerial_tm, _ = solver_tm.simulate_aerial(mask, optics)

        diff = np.abs(aerial_te - aerial_tm)
        assert diff.shape == (32, 32)


class TestSimulateFDTDBackend:
    """simulate() 函数 FDTD 后端集成测试"""

    def test_simulate_fdtd_backend_call(self):
        """测试 FDTD 后端调用"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        result = simulate(
            mask, backend=SimulationBackend.FDTD,
            apply_resist=False
        )

        assert result.backend == SimulationBackend.FDTD
        assert result.aerial_image.shape == (32, 32)
        assert result.wafer_image.shape == (32, 32)
        assert result.extra["fdtd_fallback"] is True

    def test_simulate_fdtd_with_custom_config(self):
        """测试使用自定义 FDTD 配置"""
        mask = create_contact_hole(ContactHoleParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        fdtd_cfg = FDTDConfig(
            grid_resolution_nm=1.0,
            mask_thickness_nm=80.0,
            illumination_theta_deg=10.0,
            polarization=Polarization.TE,
        )

        result = simulate(
            mask, backend='fdtd',
            fdtd_config=fdtd_cfg,
            apply_resist=True,
            threshold=0.3
        )

        assert result.backend == SimulationBackend.FDTD
        assert result.aerial_image.shape == (32, 32)
        assert np.all(np.logical_or(result.wafer_image == 0, result.wafer_image == 1))

    def test_simulate_fdtd_auto_theta(self):
        """测试自动计算照明角度"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        optics = OpticalSystem(na=1.35, sigma=0.75)

        result = simulate(
            mask, backend='fdtd',
            optical_system=optics,
            apply_resist=False
        )

        assert result.extra["fdtd_fallback"] is True
        assert "theta_deg" in result.extra
        assert result.extra["theta_deg"] > 0.0

    def test_simulate_fdtd_with_defocus(self):
        """测试 FDTD 后端带离焦"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        optics_focused = OpticalSystem(defocus=0.0)
        optics_defocused = OpticalSystem(defocus=100.0)

        result_focused = simulate(
            mask, backend='fdtd',
            optical_system=optics_focused,
            apply_resist=False
        )

        result_defocused = simulate(
            mask, backend='fdtd',
            optical_system=optics_defocused,
            apply_resist=False
        )

        contrast_focused = (
            np.max(result_focused.aerial_image) -
            np.min(result_focused.aerial_image)
        )
        contrast_defocused = (
            np.max(result_defocused.aerial_image) -
            np.min(result_defocused.aerial_image)
        )

        assert contrast_focused >= contrast_defocused * 0.9

    def test_simulate_fdtd_dose_modulation(self):
        """测试 FDTD 后端剂量调制"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        result_normal = simulate(
            mask, backend='fdtd',
            dose=1.0, apply_resist=False
        )

        result_high_dose = simulate(
            mask, backend='fdtd',
            dose=1.5, apply_resist=False
        )

        assert np.mean(result_high_dose.aerial_image) >= np.mean(result_normal.aerial_image) * 0.9

    def test_simulate_fdtd_vs_hopkins(self):
        """对比 FDTD 回退和 Hopkins 结果"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        result_hopkins = simulate(
            mask, backend='hopkins', apply_resist=False
        )

        result_fdtd = simulate(
            mask, backend='fdtd', apply_resist=False
        )

        assert result_hopkins.aerial_image.shape == result_fdtd.aerial_image.shape

        mse = np.mean((result_hopkins.aerial_image - result_fdtd.aerial_image) ** 2)
        assert mse < 0.1

    def test_simulate_fdtd_runtime_recording(self):
        """测试运行时间记录"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        result = simulate(
            mask, backend='fdtd', apply_resist=False
        )

        assert result.runtime_sec > 0.0
        assert isinstance(result.runtime_sec, float)

    def test_simulate_fdtd_result_type(self):
        """测试返回类型"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        result = simulate(
            mask, backend='fdtd', apply_resist=False
        )

        assert result.aerial_image.dtype == np.float64
        assert result.wafer_image.dtype == np.float64


class TestFDTDNumericalProperties:
    """FDTD 数值特性测试"""

    def test_grid_resolution_effect(self):
        """测试网格分辨率影响"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(16, 16), cd=45, pitch=90
        ))

        cfg_coarse = FDTDConfig(grid_resolution_nm=2.0)
        cfg_fine = FDTDConfig(grid_resolution_nm=0.5)

        solver_coarse = MeepFDTDSolver(cfg_coarse)
        solver_fine = MeepFDTDSolver(cfg_fine)

        optics = OpticalSystem()

        aerial_coarse, _ = solver_coarse.simulate_aerial(mask, optics)
        aerial_fine, _ = solver_fine.simulate_aerial(mask, optics)

        assert aerial_coarse.shape == aerial_fine.shape
        assert np.all(aerial_coarse >= 0)
        assert np.all(aerial_fine >= 0)

    def test_pml_thickness_config(self):
        """测试 PML 厚度配置"""
        cfg_thin = FDTDConfig(pml_thickness_nm=100.0)
        cfg_thick = FDTDConfig(pml_thickness_nm=300.0)

        assert cfg_thin.pml_thickness_nm == 100.0
        assert cfg_thick.pml_thickness_nm == 300.0

    def test_courant_factor_stability(self):
        """测试 Courant 因子稳定性"""
        cfg = FDTDConfig(courant_factor=0.9)
        assert 0 < cfg.courant_factor <= 1.0

        cfg_unstable = FDTDConfig(courant_factor=1.5)
        assert cfg_unstable.courant_factor > 1.0

    def test_ntff_kspace_sampling(self):
        """测试 k 空间采样"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)

        nx, ny = 32, 32
        dx, dy = 2.0, 2.0

        kx_expected = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, dx))
        ky_expected = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, dy))

        Ex = np.zeros((ny, nx), dtype=np.complex128)
        Ey = np.zeros_like(Ex)
        Ez = np.zeros_like(Ex)

        result = solver._near_to_far_field(
            Ex, Ey, Ez,
            sx_nm=nx * dx, sy_nm=ny * dy,
            wavelength_nm=193, cfg=cfg
        )

        assert np.allclose(result["KX"][0, :], kx_expected)
        assert np.allclose(result["KY"][:, 0], ky_expected)


class TestFDTDErrorHandling:
    """FDTD 错误处理测试"""

    def test_invalid_polarization(self):
        """测试无效偏振值处理"""
        with pytest.raises(ValueError):
            FDTDConfig(polarization="invalid")

    def test_negative_resolution(self):
        """测试负分辨率"""
        cfg = FDTDConfig(grid_resolution_nm=-0.5)
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem()

        mask = np.ones((16, 16))
        aerial, extra = solver.simulate_aerial(mask, optics)

        assert extra["fdtd_fallback"] is True
        assert aerial.shape == (16, 16)

    def test_mask_shape_preservation(self):
        """测试掩模形状保持"""
        cfg = FDTDConfig()
        solver = MeepFDTDSolver(cfg)
        optics = OpticalSystem()

        for shape in [(16, 16), (32, 64), (64, 32), (128, 128)]:
            mask = np.random.random(shape)
            aerial, _ = solver.simulate_aerial(mask, optics)
            assert aerial.shape == shape

    def test_simulate_string_backend(self):
        """测试字符串后端参数"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90
        ))

        result = simulate(mask, backend='fdtd', apply_resist=False)
        assert result.backend == SimulationBackend.FDTD

        result_enum = simulate(mask, backend=SimulationBackend.FDTD, apply_resist=False)
        assert result_enum.backend == SimulationBackend.FDTD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
