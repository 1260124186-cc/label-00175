# -*- coding: utf-8 -*-
"""
Pipeline Orchestrator 回归测试

覆盖：
- PW 验签光源复用 SMO optimal_source 的一致性验证
- sign-off 摘要包含验证结果
- 各阶段跳过逻辑
- PipelineConfig YAML 加载
"""

import pytest
import numpy as np
from unittest.mock import patch

from core.imaging import OpticalSystem, IlluminationType, TCCMode
from core.test_structures import create_line_space, TestStructureParams, TestStructureType

from pipeline.orchestrator import (
    PipelineConfig,
    PipelineOrchestrator,
    PipelineResult,
    StageMetrics,
    PWVerifyConfig,
    run_pipeline,
)


SIM_SIZE = 64
SIM_PIXEL = 40.0


@pytest.fixture
def optics():
    return OpticalSystem(
        wavelength=193.0, na=1.35, sigma=0.75, pixel_size=SIM_PIXEL,
        socs_num_terms=3, tcc_mode=TCCMode.SOCS,
    )


@pytest.fixture
def target():
    params = TestStructureParams(
        grid_size=(SIM_SIZE, SIM_SIZE),
        pixel_size=SIM_PIXEL,
        cd=90.0,
        pitch=180.0,
        structure_type=TestStructureType.LINE_SPACE,
    )
    return create_line_space(params)


@pytest.fixture
def initial_mask(target):
    return target.copy().astype(np.float64)


class TestPWSourceConsistency:
    """回归验证：PW 验签必须复用 SMO 产出的 optimal_source"""

    def test_pw_uses_smo_source_true_when_both_enabled_and_consistent(self, optics, target):
        """SMO 和 PW 都启用，且 PW 使用了 SMO 光源 → pw_uses_smo_source = True"""
        fake_source = np.random.rand(SIM_SIZE, SIM_SIZE)
        fake_source = fake_source / np.sum(fake_source)

        pw_optics = OpticalSystem(
            wavelength=optics.wavelength,
            na=optics.na,
            sigma=optics.sigma,
            pixel_size=optics.pixel_size,
            illumination_type=IlluminationType.CUSTOM,
            custom_source=fake_source.copy(),
            tcc_mode=optics.tcc_mode,
        )

        result = PipelineResult(
            initial_mask=target,
            final_mask=target,
            target=target,
            optical_system=optics,
            optimal_source=fake_source.copy(),
            pw_optical_system=pw_optics,
            pw_metrics=_make_fake_pw_metrics(),
        )

        assert result.pw_uses_smo_source is True

    def test_pw_uses_smo_source_false_when_illumination_not_custom(self, optics, target):
        """Bug 回归：PW 光学系统 illumination_type 不是 CUSTOM → 验证失败"""
        fake_source = np.random.rand(SIM_SIZE, SIM_SIZE)
        fake_source = fake_source / np.sum(fake_source)

        pw_optics = OpticalSystem(
            wavelength=optics.wavelength,
            na=optics.na,
            sigma=optics.sigma,
            pixel_size=optics.pixel_size,
            illumination_type=IlluminationType.CONVENTIONAL,
            custom_source=fake_source.copy(),
            tcc_mode=optics.tcc_mode,
        )

        result = PipelineResult(
            initial_mask=target,
            final_mask=target,
            target=target,
            optical_system=optics,
            optimal_source=fake_source.copy(),
            pw_optical_system=pw_optics,
            pw_metrics=_make_fake_pw_metrics(),
        )

        assert result.pw_uses_smo_source is False

    def test_pw_uses_smo_source_false_when_custom_source_none(self, optics, target):
        """Bug 回归：PW 光学系统 custom_source 为 None → 验证失败"""
        fake_source = np.random.rand(SIM_SIZE, SIM_SIZE)
        fake_source = fake_source / np.sum(fake_source)

        pw_optics = OpticalSystem(
            wavelength=optics.wavelength,
            na=optics.na,
            sigma=optics.sigma,
            pixel_size=optics.pixel_size,
            illumination_type=IlluminationType.CUSTOM,
            custom_source=None,
            tcc_mode=optics.tcc_mode,
        )

        result = PipelineResult(
            initial_mask=target,
            final_mask=target,
            target=target,
            optical_system=optics,
            optimal_source=fake_source.copy(),
            pw_optical_system=pw_optics,
            pw_metrics=_make_fake_pw_metrics(),
        )

        assert result.pw_uses_smo_source is False

    def test_pw_uses_smo_source_true_when_smo_disabled(self, optics, target):
        """SMO 未启用 → pw_uses_smo_source 默认为 True（无验证必要）"""
        result = PipelineResult(
            initial_mask=target,
            final_mask=target,
            target=target,
            optical_system=optics,
            optimal_source=None,
            pw_metrics=_make_fake_pw_metrics(),
        )
        assert result.pw_uses_smo_source is True

    def test_pw_uses_smo_source_true_when_pw_disabled(self, optics, target):
        """PW 未启用 → pw_uses_smo_source 默认为 True（无验证必要）"""
        fake_source = np.random.rand(SIM_SIZE, SIM_SIZE)
        result = PipelineResult(
            initial_mask=target,
            final_mask=target,
            target=target,
            optical_system=optics,
            optimal_source=fake_source,
            pw_metrics=None,
        )
        assert result.pw_uses_smo_source is True

    def test_validate_pw_source_consistency_details(self, optics, target):
        """详细验证：validate_pw_source_consistency 返回完整检查项"""
        fake_source = np.random.rand(SIM_SIZE, SIM_SIZE)
        fake_source = fake_source / np.sum(fake_source)

        pw_optics = OpticalSystem(
            wavelength=optics.wavelength,
            na=optics.na,
            sigma=optics.sigma,
            pixel_size=optics.pixel_size,
            illumination_type=IlluminationType.CUSTOM,
            custom_source=fake_source.copy(),
            tcc_mode=optics.tcc_mode,
        )

        result = PipelineResult(
            initial_mask=target,
            final_mask=target,
            target=target,
            optical_system=optics,
            optimal_source=fake_source.copy(),
            pw_optical_system=pw_optics,
            pw_metrics=_make_fake_pw_metrics(),
        )

        validation = result.validate_pw_source_consistency()
        assert validation['passed'] is True
        assert 'checks' in validation
        assert validation['checks']['illumination_type_is_custom'] is True
        assert validation['checks']['custom_source_not_none'] is True
        assert validation['checks']['shape_match'] is True
        assert validation['checks']['values_match'] is True
        assert 'details' in validation
        assert 'max_source_diff' in validation['details']
        assert validation['details']['max_source_diff'] < 1e-10

    def test_validate_pw_source_consistency_shape_mismatch(self, optics, target):
        """形状不匹配 → 验证失败"""
        fake_source = np.random.rand(SIM_SIZE, SIM_SIZE)
        wrong_source = np.random.rand(SIM_SIZE + 2, SIM_SIZE + 2)

        pw_optics = OpticalSystem(
            wavelength=optics.wavelength,
            na=optics.na,
            sigma=optics.sigma,
            pixel_size=optics.pixel_size,
            illumination_type=IlluminationType.CUSTOM,
            custom_source=wrong_source,
            tcc_mode=optics.tcc_mode,
        )

        result = PipelineResult(
            initial_mask=target,
            final_mask=target,
            target=target,
            optical_system=optics,
            optimal_source=fake_source,
            pw_optical_system=pw_optics,
            pw_metrics=_make_fake_pw_metrics(),
        )

        validation = result.validate_pw_source_consistency()
        assert validation['passed'] is False
        assert validation['checks']['shape_match'] is False


class TestSignOffSummary:
    """sign-off 摘要回归测试"""

    def test_sign_off_summary_contains_validation(self, optics, target):
        """sign-off 摘要必须包含 validation 字段"""
        fake_source = np.random.rand(SIM_SIZE, SIM_SIZE)
        fake_source = fake_source / np.sum(fake_source)

        pw_optics = OpticalSystem(
            wavelength=optics.wavelength,
            na=optics.na,
            sigma=optics.sigma,
            pixel_size=optics.pixel_size,
            illumination_type=IlluminationType.CUSTOM,
            custom_source=fake_source.copy(),
            tcc_mode=optics.tcc_mode,
        )

        result = PipelineResult(
            initial_mask=target,
            final_mask=target,
            target=target,
            optical_system=optics,
            optimal_source=fake_source.copy(),
            pw_optical_system=pw_optics,
            pw_metrics=_make_fake_pw_metrics(),
        )

        summary = result.sign_off_summary()
        assert 'validation' in summary
        assert 'pw_uses_smo_source' in summary['validation']
        assert summary['validation']['pw_uses_smo_source'] is True
        assert 'pw_source_consistency' in summary['validation']

    def test_sign_off_text_contains_validation(self, optics, target):
        """sign-off 文本必须包含 Validation 段落"""
        fake_source = np.random.rand(SIM_SIZE, SIM_SIZE)
        fake_source = fake_source / np.sum(fake_source)

        pw_optics = OpticalSystem(
            wavelength=optics.wavelength,
            na=optics.na,
            sigma=optics.sigma,
            pixel_size=optics.pixel_size,
            illumination_type=IlluminationType.CUSTOM,
            custom_source=fake_source.copy(),
            tcc_mode=optics.tcc_mode,
        )

        result = PipelineResult(
            initial_mask=target,
            final_mask=target,
            target=target,
            optical_system=optics,
            optimal_source=fake_source.copy(),
            pw_optical_system=pw_optics,
            pw_metrics=_make_fake_pw_metrics(),
        )

        text = result.sign_off_text()
        assert '[Validation]' in text
        assert 'PW uses SMO source: PASS' in text


class TestPipelineConfig:
    """PipelineConfig 配置测试"""

    def test_pw_verify_config_from_dict(self):
        d = {
            'focus_range': [-100, 100, 7],
            'dose_range': [0.9, 1.1, 5],
            'cd_tolerance': 0.15,
        }
        cfg = PWVerifyConfig.from_dict(d)
        assert cfg.focus_range == (-100, 100, 7)
        assert cfg.dose_range == (0.9, 1.1, 5)
        assert cfg.cd_tolerance == 0.15

    def test_pw_verify_config_to_dict(self):
        cfg = PWVerifyConfig()
        d = cfg.to_dict()
        assert 'focus_range' in d
        assert 'dose_range' in d
        assert 'cd_tolerance' in d

    def test_pipeline_config_defaults(self):
        cfg = PipelineConfig()
        assert cfg.enable_opc is True
        assert cfg.enable_ilt is True
        assert cfg.enable_smo is True
        assert cfg.enable_pw_verify is True

    def test_pipeline_config_from_dict_stages(self):
        d = {
            'enable_opc': False,
            'enable_ilt': True,
            'enable_smo': False,
            'enable_pw_verify': True,
        }
        cfg = PipelineConfig.from_dict(d)
        assert cfg.enable_opc is False
        assert cfg.enable_ilt is True
        assert cfg.enable_smo is False
        assert cfg.enable_pw_verify is True

    def test_pipeline_config_to_dict(self):
        cfg = PipelineConfig(
            enable_opc=False,
            output_dir='results/test',
        )
        d = cfg.to_dict()
        assert d['enable_opc'] is False
        assert d['output_dir'] == 'results/test'


class TestStageMetrics:
    """StageMetrics 测试"""

    def test_epe_improvement(self):
        m = StageMetrics(
            stage_name='OPC',
            epe_before={'epe_mean': 10.0, 'epe_max': 20.0},
            epe_after={'epe_mean': 6.0, 'epe_max': 15.0},
        )
        assert m.epe_improvement == pytest.approx(4.0)

    def test_to_dict(self):
        m = StageMetrics(
            stage_name='ILT',
            elapsed_sec=12.5,
            epe_before={'epe_mean': 8.0},
            epe_after={'epe_mean': 3.0},
            extra={'converged': True},
        )
        d = m.to_dict()
        assert d['stage_name'] == 'ILT'
        assert d['elapsed_sec'] == 12.5
        assert d['epe_improvement'] == 5.0
        assert d['extra']['converged'] is True


def _make_fake_pw_metrics():
    """构造一个假的 PWMetrics 对象用于测试"""
    from analysis.process_window import PWMetrics
    return PWMetrics(
        pw_area=100.0,
        pw_ratio=0.5,
        n_passing=50,
        n_total=100,
        center_focus=0.0,
        center_dose=1.0,
        best_focus=0.0,
        best_dose=1.0,
        best_cd_error=0.5,
        focus_range=(-50.0, 50.0),
        dose_range=(0.95, 1.05),
        depth_of_focus=100.0,
        exposure_latitude=10.0,
    )
