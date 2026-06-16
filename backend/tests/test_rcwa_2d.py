# -*- coding: utf-8 -*-
"""
二维 RCWA 与矢量传递函数单元测试

测试 RCWASolver2D、VectorTransferFunction 以及标量 Hopkins vs 矢量 RCWA 对比框架。
"""

import pytest
import numpy as np

from core.rigorous_sim import (
    RCWAConfig,
    RCWA2DResult,
    RCWASolver2D,
    VectorTransferFunction,
    Polarization,
    SimulationBackend,
    simulate,
    compare_backends,
    _estimate_period_2d,
    _estimate_duty_cycle_2d,
    _detect_hole_shape,
)
from core.imaging import OpticalSystem
from core.test_structures import (
    create_contact_hole,
    ContactHoleParams,
    create_line_space,
    LineSpaceParams,
)


class TestRCWAConfig2D:
    """2D RCWA 配置参数测试"""

    def test_default_2d_parameters(self):
        """测试默认 2D 参数"""
        cfg = RCWAConfig()
        assert cfg.use_2d_rcwa is False
        assert cfg.vector_transfer is False
        assert cfg.n_orders_y is None
        assert cfg.period_y_nm is None
        assert cfg.hole_diameter_nm is None
        assert cfg.illumination_theta_deg == 0.0
        assert cfg.illumination_phi_deg == 0.0

    def test_custom_2d_parameters(self):
        """测试自定义 2D 参数"""
        cfg = RCWAConfig(
            use_2d_rcwa=True,
            vector_transfer=True,
            n_orders=4,
            n_orders_y=3,
            period_nm=180.0,
            period_y_nm=200.0,
            hole_diameter_nm=90.0,
            illumination_theta_deg=5.0,
            illumination_phi_deg=45.0,
        )
        assert cfg.use_2d_rcwa is True
        assert cfg.vector_transfer is True
        assert cfg.n_orders == 4
        assert cfg.n_orders_y == 3
        assert cfg.period_nm == 180.0
        assert cfg.period_y_nm == 200.0
        assert cfg.hole_diameter_nm == 90.0
        assert cfg.illumination_theta_deg == 5.0
        assert cfg.illumination_phi_deg == 45.0


class TestRCWASolver2D:
    """二维 RCWA 求解器测试"""

    def test_solver_creation(self):
        """测试求解器创建"""
        cfg = RCWAConfig(n_orders=2)
        solver = RCWASolver2D(cfg)
        assert solver.cfg is cfg

    def test_solve_far_field_square_hole(self):
        """测试方形接触孔远场衍射求解"""
        cfg = RCWAConfig(n_orders=2)
        solver = RCWASolver2D(cfg)
        result = solver.solve_far_field(
            wavelength_nm=193.0,
            period_x_nm=180.0,
            period_y_nm=180.0,
            duty_cycle_x=0.5,
            duty_cycle_y=0.5,
            theta_deg=0.0,
            phi_deg=0.0,
            hole_shape="square",
        )

        assert isinstance(result, RCWA2DResult)
        assert result.orders_x.shape == (5,)
        assert result.orders_y.shape == (5,)
        assert result.t_TE.shape == (5, 5)
        assert result.t_TM.shape == (5, 5)
        assert result.eff_trans_TE.shape == (5, 5)
        assert result.eff_trans_TM.shape == (5, 5)
        assert np.all(np.isfinite(result.t_TE))
        assert np.all(np.isfinite(result.t_TM))

        eff_sum_TE = float(np.sum(result.eff_trans_TE) + np.sum(result.eff_reflect_TE))
        eff_sum_TM = float(np.sum(result.eff_trans_TM) + np.sum(result.eff_reflect_TM))
        assert eff_sum_TE > 0.0
        assert eff_sum_TM > 0.0

    def test_solve_far_field_circular_hole(self):
        """测试圆形接触孔远场衍射求解"""
        cfg = RCWAConfig(n_orders=2)
        solver = RCWASolver2D(cfg)
        result = solver.solve_far_field(
            wavelength_nm=193.0,
            period_x_nm=200.0,
            period_y_nm=200.0,
            duty_cycle_x=0.45,
            duty_cycle_y=0.45,
            theta_deg=0.0,
            phi_deg=0.0,
            hole_shape="circle",
        )

        assert result.t_TE.shape == (5, 5)
        assert np.all(np.isfinite(result.eff_trans_TE))

    def test_solve_with_oblique_incidence(self):
        """测试斜入射求解"""
        cfg = RCWAConfig(n_orders=2)
        solver = RCWASolver2D(cfg)
        result = solver.solve_far_field(
            wavelength_nm=193.0,
            period_x_nm=180.0,
            period_y_nm=180.0,
            duty_cycle_x=0.5,
            duty_cycle_y=0.5,
            theta_deg=5.0,
            phi_deg=30.0,
            hole_shape="square",
        )

        assert np.all(np.isfinite(result.t_TE))
        assert np.all(np.isfinite(result.t_TM))

    def test_different_orders(self):
        """测试不同截断级次"""
        for n_ord in [1, 2, 3]:
            cfg = RCWAConfig(n_orders=n_ord)
            solver = RCWASolver2D(cfg)
            result = solver.solve_far_field(
                wavelength_nm=193.0,
                period_x_nm=180.0,
                period_y_nm=180.0,
                duty_cycle_x=0.5,
                duty_cycle_y=0.5,
            )
            expected = 2 * n_ord + 1
            assert result.t_TE.shape == (expected, expected)

    def test_result_shape_orders_property(self):
        """测试 shape_orders 属性"""
        cfg = RCWAConfig(n_orders=2, n_orders_y=3)
        solver = RCWASolver2D(cfg)
        result = solver.solve_far_field(
            wavelength_nm=193.0,
            period_x_nm=180.0,
            period_y_nm=180.0,
            duty_cycle_x=0.5,
            duty_cycle_y=0.5,
        )
        assert result.shape_orders == (7, 5)


class TestVectorTransferFunction:
    """矢量传递函数测试"""

    def test_vtf_creation(self):
        """测试 VTF 创建"""
        vtf = VectorTransferFunction(
            wavelength_nm=193.0,
            na=1.35,
            n_immersion=1.44,
            pixel_size_nm=2.0,
            grid_size=(64, 64),
        )
        assert vtf.wavelength == 193.0
        assert vtf.na == 1.35
        assert vtf.nx == 64
        assert vtf.ny == 64

    def test_s_polarization_vector(self):
        """测试 s-偏振矢量"""
        vtf = VectorTransferFunction(193.0, 1.35, grid_size=(32, 32))
        s_vec = vtf.s_polarization_vector()
        assert "Ex" in s_vec and "Ey" in s_vec and "Ez" in s_vec
        assert s_vec["Ex"].shape == (32, 32)
        assert np.allclose(s_vec["Ez"], 0.0, atol=1e-10)

        mag_sq = np.abs(s_vec["Ex"]) ** 2 + np.abs(s_vec["Ey"]) ** 2 + np.abs(s_vec["Ez"]) ** 2
        valid = vtf.pupil_mask & (vtf.rho > 1e-6)
        assert np.allclose(mag_sq[valid], 1.0, atol=1e-6)

    def test_p_polarization_vector(self):
        """测试 p-偏振矢量"""
        vtf = VectorTransferFunction(193.0, 1.35, grid_size=(32, 32))
        p_vec = vtf.p_polarization_vector()
        assert p_vec["Ex"].shape == (32, 32)

        mag_sq = np.abs(p_vec["Ex"]) ** 2 + np.abs(p_vec["Ey"]) ** 2 + np.abs(p_vec["Ez"]) ** 2
        valid = vtf.pupil_mask
        assert np.allclose(mag_sq[valid], 1.0, atol=1e-6)

    def test_polarization_orthogonality(self):
        """测试 s/p 偏振正交性"""
        vtf = VectorTransferFunction(193.0, 1.35, grid_size=(32, 32))
        s_vec = vtf.s_polarization_vector()
        p_vec = vtf.p_polarization_vector()

        dot = (
            s_vec["Ex"] * p_vec["Ex"]
            + s_vec["Ey"] * p_vec["Ey"]
            + s_vec["Ez"] * p_vec["Ez"]
        )
        valid = vtf.pupil_mask & (vtf.rho > 1e-6)
        assert np.allclose(dot[valid], 0.0, atol=1e-6)

    def test_decompose_incident_field_te(self):
        """测试 TE 偏振分解"""
        vtf = VectorTransferFunction(193.0, 1.35, grid_size=(32, 32))
        decomp = vtf.decompose_incident_field(Polarization.TE)
        assert np.all(decomp["p_weight"] == 0.0)
        assert np.any(decomp["s_weight"] > 0)

    def test_decompose_incident_field_tm(self):
        """测试 TM 偏振分解"""
        vtf = VectorTransferFunction(193.0, 1.35, grid_size=(32, 32))
        decomp = vtf.decompose_incident_field(Polarization.TM)
        assert np.all(decomp["s_weight"] == 0.0)
        assert np.any(decomp["p_weight"] > 0)

    def test_decompose_incident_field_unpolarized(self):
        """测试非偏振分解"""
        vtf = VectorTransferFunction(193.0, 1.35, grid_size=(32, 32))
        decomp = vtf.decompose_incident_field(Polarization.UNPOLARIZED)
        assert np.any(decomp["s_weight"] > 0)
        assert np.any(decomp["p_weight"] > 0)

    def test_apply_vector_transfer(self):
        """测试矢量传递函数应用"""
        vtf = VectorTransferFunction(
            wavelength_nm=193.0,
            na=1.35,
            n_immersion=1.44,
            pixel_size_nm=2.0,
            grid_size=(64, 64),
        )
        optics = OpticalSystem()

        far_TE = np.zeros((5, 5), dtype=np.complex128)
        far_TM = np.zeros((5, 5), dtype=np.complex128)
        far_TE[2, 2] = 1.0
        far_TM[2, 2] = 1.0

        aerial = vtf.apply_vector_transfer(far_TE, far_TM, optics)
        assert aerial.shape == (64, 64)
        assert aerial.dtype == np.float64
        assert np.all(aerial >= 0.0)
        assert np.all(aerial <= 1.0 + 1e-10)
        assert np.nanmax(aerial) > 0.0


class TestPeriodEstimation2D:
    """2D 周期估计辅助函数测试"""

    def test_estimate_period_2d_line_space(self):
        """测试 line/space 结构周期估计"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(64, 64), cd=45, pitch=90, pixel_size=2.0
        ))
        px, py = _estimate_period_2d(mask, 2.0)
        assert px > 0.0 or py > 0.0

    def test_estimate_period_2d_contact_hole(self):
        """测试接触孔周期估计"""
        mask = create_contact_hole(ContactHoleParams(
            grid_size=(64, 64), cd=90, pitch=180, pixel_size=2.0
        ))
        px, py = _estimate_period_2d(mask, 2.0)
        assert px > 0.0 and py > 0.0

    def test_estimate_duty_cycle(self):
        """测试占空比估计"""
        mask = np.ones((32, 32)) * 0.5
        fx, fy = _estimate_duty_cycle_2d(mask)
        assert 0.05 <= fx <= 0.95
        assert 0.05 <= fy <= 0.95

    def test_detect_hole_shape_square(self):
        """测试方形孔检测"""
        mask = create_contact_hole(ContactHoleParams(
            grid_size=(64, 64), cd=90, pitch=180, pixel_size=2.0,
            hole_shape="square",
        ))
        shape = _detect_hole_shape(mask)
        assert shape in ["square", "circle"]

    def test_detect_hole_shape_circle(self):
        """测试圆形孔检测"""
        mask = create_contact_hole(ContactHoleParams(
            grid_size=(64, 64), cd=90, pitch=180, pixel_size=2.0,
            hole_shape="circle",
        ))
        shape = _detect_hole_shape(mask)
        assert shape in ["square", "circle"]


class TestSimulateRCWA2D:
    """simulate() 函数 2D RCWA 集成测试"""

    def test_simulate_1d_rcwa_line_space(self):
        """测试 1D RCWA 后端 line/space（明确指定周期）"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(64, 64), cd=45, pitch=90, pixel_size=2.0
        ))
        result = simulate(
            mask, backend=SimulationBackend.RCWA,
            rcwa_config=RCWAConfig(
                n_orders=2, use_2d_rcwa=False,
                period_nm=90.0, line_width_nm=45.0
            ),
            apply_resist=False,
        )
        assert result.backend == SimulationBackend.RCWA
        assert result.aerial_image.shape == (64, 64)
        assert "rcwa_mode" in result.extra
        assert result.extra["rcwa_mode"] == "1d"
        assert np.all(result.aerial_image >= 0.0)
        assert np.nanmax(result.aerial_image) > 0.0

    def test_simulate_2d_rcwa_contact_hole(self):
        """测试 2D RCWA 后端接触孔（明确指定周期）"""
        mask = create_contact_hole(ContactHoleParams(
            grid_size=(64, 64), cd=45, pitch=90, pixel_size=2.0
        ))
        result = simulate(
            mask, backend=SimulationBackend.RCWA,
            rcwa_config=RCWAConfig(
                n_orders=2, use_2d_rcwa=True, vector_transfer=False,
                period_nm=90.0, period_y_nm=90.0, hole_diameter_nm=45.0
            ),
            apply_resist=False,
        )
        assert result.backend == SimulationBackend.RCWA
        assert result.aerial_image.shape == (64, 64)
        assert result.extra["rcwa_mode"] == "2d"
        assert "rcwa_hole_shape" in result.extra
        assert "diffraction_orders_2d" in result.extra
        assert np.nanmax(result.aerial_image) > 0.0

    def test_simulate_2d_rcwa_vector_transfer(self):
        """测试 2D RCWA + 矢量传递函数（明确指定周期）"""
        mask = create_contact_hole(ContactHoleParams(
            grid_size=(64, 64), cd=45, pitch=90, pixel_size=2.0
        ))
        result = simulate(
            mask, backend=SimulationBackend.RCWA,
            rcwa_config=RCWAConfig(
                n_orders=2, use_2d_rcwa=True, vector_transfer=True,
                period_nm=90.0, period_y_nm=90.0, hole_diameter_nm=45.0
            ),
            apply_resist=False,
        )
        assert result.aerial_image.shape == (64, 64)
        assert result.extra["rcwa_vector_transfer"] is True
        assert np.all(result.aerial_image >= 0.0)
        assert np.nanmax(result.aerial_image) > 0.0

    def test_simulate_2d_rcwa_with_threshold(self):
        """测试 2D RCWA 带光刻胶阈值"""
        mask = create_contact_hole(ContactHoleParams(
            grid_size=(32, 32), cd=45, pitch=90, pixel_size=2.0
        ))
        result = simulate(
            mask, backend="rcwa",
            rcwa_config=RCWAConfig(n_orders=2, use_2d_rcwa=True),
            apply_resist=True,
            threshold=0.3,
        )
        assert result.wafer_image.shape == (32, 32)
        assert set(np.unique(result.wafer_image)).issubset({0.0, 1.0})

    def test_simulate_rcwa_polarization_effect(self):
        """测试 TE vs TM 偏振差异"""
        mask = create_contact_hole(ContactHoleParams(
            grid_size=(32, 32), cd=45, pitch=90, pixel_size=2.0
        ))

        cfg_te = RCWAConfig(n_orders=2, use_2d_rcwa=True, polarization=Polarization.TE)
        cfg_tm = RCWAConfig(n_orders=2, use_2d_rcwa=True, polarization=Polarization.TM)

        res_te = simulate(mask, backend="rcwa", rcwa_config=cfg_te, apply_resist=False)
        res_tm = simulate(mask, backend="rcwa", rcwa_config=cfg_tm, apply_resist=False)

        assert res_te.aerial_image.shape == res_tm.aerial_image.shape
        diff = np.abs(res_te.aerial_image - res_tm.aerial_image)
        assert diff.shape == (32, 32)


class TestBackendComparison:
    """标量 vs 矢量后端对比框架测试"""

    def test_compare_hopkins_vs_rcwa2d(self):
        """测试 Hopkins 与 2D RCWA 对比（明确指定周期）"""
        mask = create_contact_hole(ContactHoleParams(
            grid_size=(64, 64), cd=45, pitch=90, pixel_size=2.0
        ))

        report = compare_backends(
            mask,
            mask_name="test_contact_hole",
            backend_scalar="hopkins",
            backend_vector="rcwa",
            rcwa_config=RCWAConfig(
                n_orders=2, use_2d_rcwa=True, vector_transfer=False,
                period_nm=90.0, period_y_nm=90.0, hole_diameter_nm=45.0
            ),
            apply_resist=False,
        )

        assert report.mask_name == "test_contact_hole"
        assert report.image_mse >= 0.0
        assert report.wafer_jaccard >= 0.0
        assert report.runtime_scalar_sec > 0.0
        assert report.runtime_vector_sec > 0.0
        assert "diffraction_orders_2d" in report.vector_extra

    def test_compare_report_summary(self):
        """测试对比报告 summary() 输出"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90, pixel_size=2.0
        ))

        report = compare_backends(
            mask, mask_name="test_ls",
            rcwa_config=RCWAConfig(n_orders=2, use_2d_rcwa=False),
            apply_resist=False,
        )
        summary = report.summary()
        assert "标量" in summary
        assert "矢量" in summary
        assert "test_ls" in summary

    def test_compare_report_to_dict(self):
        """测试对比报告序列化"""
        mask = create_line_space(LineSpaceParams(
            grid_size=(32, 32), cd=45, pitch=90, pixel_size=2.0
        ))
        report = compare_backends(
            mask, mask_name="test",
            rcwa_config=RCWAConfig(n_orders=2),
            apply_resist=False,
        )
        d = report.to_dict()
        assert "mask" in d
        assert "backends" in d
        assert "cd" in d
        assert "runtime_sec" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
