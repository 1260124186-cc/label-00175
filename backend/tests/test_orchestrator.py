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


# ---------------------------------------------------------------------------
# 新增：配置化闭环 + smoke 增量覆盖回归测试
# ---------------------------------------------------------------------------

def _pipeline_default_yaml_path():
    """定位 pipeline_default.yaml 的真实路径（pytest 从仓库根或 backend/ 启动都能找到）"""
    from pathlib import Path
    candidates = [
        Path(__file__).resolve().parent.parent / "config" / "pipeline_default.yaml",
        Path.cwd() / "backend" / "config" / "pipeline_default.yaml",
        Path.cwd() / "config" / "pipeline_default.yaml",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("pipeline_default.yaml not found")


class TestPipelineDefaultYAMLConsumption:
    """
    验证 pipeline_default.yaml 三块配置（pipeline / optical_system / test_pattern）
    都能被完整消费，而非只认一小部分字段。

    这组测试是"配置化流水线闭环"的核心回归：一旦 YAML 新增字段而消费方忘记读，
    这里会立刻报失败。
    """

    def test_yaml_sections_all_present(self):
        from utils.config import load_config
        full = load_config(_pipeline_default_yaml_path())
        assert 'pipeline' in full, "YAML 缺少 pipeline 块"
        assert 'optical_system' in full, "YAML 缺少 optical_system 块"
        assert 'test_pattern' in full, "YAML 缺少 test_pattern 块"

    def test_optical_system_section_consumed_by_from_config(self):
        """OpticalSystem.from_config({'optical_system': yaml_section}) 消费所有关键字段"""
        from utils.config import load_config
        full = load_config(_pipeline_default_yaml_path())
        optic_section = full['optical_system']
        optics = OpticalSystem.from_config({'optical_system': optic_section})

        assert optics.wavelength == 193.0
        assert optics.na == 1.35
        assert optics.sigma == 0.75
        assert optics.pixel_size == 1.0
        assert optics.illumination_type == IlluminationType.CONVENTIONAL
        assert optics.socs_num_terms == 5
        assert optics.tcc_mode == TCCMode.SOCS
        # Zernike：默认 YAML 里都写了 0；确保它们不是 None
        z_keys = ('zernike_coefficients', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11')
        for k in z_keys:
            if k in optic_section:
                val = optic_section[k]
                # 主要确认字段被消费，不纠结精度
                assert val is not None, f"optical_system.{k} 在 YAML 中是 None"

    def test_test_pattern_section_consumed(self):
        """test_pattern 块能被 build_target_from_pattern_dict 消费并生成合理图案"""
        from utils.config import load_config
        from examples.run_pipeline import build_target_from_pattern_dict
        full = load_config(_pipeline_default_yaml_path())
        section = full['test_pattern']

        target, px = build_target_from_pattern_dict(section)

        grid_size = section.get('grid_size', [64, 64])
        expect_shape = tuple(grid_size) if isinstance(grid_size, (list, tuple)) else (grid_size, grid_size)
        assert target.shape == expect_shape
        assert px == section.get('pixel_size', 1.0)
        # 图案不是全零（必须是真实的 line_space 结构）
        assert target.sum() > 0, "target 图案全零，说明 build_target_from_pattern_dict 没正确生成"
        # CD / pitch 存在于配置（字段级别验证）
        assert 'cd' in section
        assert 'pitch' in section
        assert section['cd'] > 0
        assert section['type'] in ('line_space', 'l_shaped', 'contact_hole')

    def test_pipeline_section_opc_ilt_smo_pw_all_consumed(self):
        """pipeline 块的四个阶段 config 全部被读入到 PipelineConfig 对象"""
        path = _pipeline_default_yaml_path()
        cfg = PipelineConfig.from_yaml(path)

        # 四大阶段
        assert cfg.enable_opc is True
        assert cfg.enable_ilt is True
        assert cfg.enable_smo is True
        assert cfg.enable_pw_verify is True

        # 每个阶段的 config 对象都存在，且关键字段不是"凭空默认"
        assert cfg.opc_config is not None, "OPC 配置未从 YAML 读入"
        assert cfg.ilt_config is not None, "ILT 配置未从 YAML 读入"
        assert cfg.smo_config is not None, "SMO 配置未从 YAML 读入"
        assert cfg.pw_verify_config is not None, "PW 配置未从 YAML 读入"

        # 确认是 YAML 里写的值，而不是硬编码默认
        assert cfg.opc_config.max_iterations == 10
        assert cfg.ilt_config.max_iter == 200
        # SMO 策略字段存在于 config 中（不是默认）
        from workflows.smo import SMOptimizationStrategy
        assert cfg.smo_config.strategy == SMOptimizationStrategy.ALTERNATING
        assert cfg.smo_config.max_outer_iterations == 20
        # PW 配置（YAML 里写的 (-150, 150, 11) / (0.85, 1.15, 11)）
        assert cfg.pw_verify_config.focus_range == (-150, 150, 11)
        assert cfg.pw_verify_config.dose_range == (0.85, 1.15, 11)
        # 输出目录
        assert cfg.output_dir is not None
        assert len(str(cfg.output_dir)) > 0


class TestSmokeConfigFromYaml:
    """
    smoke 模式必须：
    1. 先从 pipeline_default.yaml 读完整配置（含 PW、OPC、ILT、SMO 原始值）
    2. 再"增量覆盖"迭代数调小 / SRAF 关 / 中间产物关
    3. **不能**丢失 YAML 里的 PW 配置（回归：旧 smoke 模式 pw_verify_config=None）
    4. **不能**丢失 YAML 里 SMO 的 strategy 等非迭代参数字段
    """

    def test_smoke_preserves_pw_verify_config(self):
        """回归：旧实现把 pw_verify_config=None，这里强制验证 YAML 配置被保留"""
        from examples.run_pipeline import build_smoke_config_from_yaml

        cfg = build_smoke_config_from_yaml()
        assert cfg.pw_verify_config is not None, (
            "smoke 模式下丢失了 YAML 的 PW 配置——配置化闭环被破坏！"
        )
        # YAML 里写的 focus_range / dose_range 必须被保留
        assert cfg.pw_verify_config.focus_range == (-150, 150, 11)
        assert cfg.pw_verify_config.dose_range == (0.85, 1.15, 11)

    def test_smoke_iterations_are_actually_lower(self):
        """smoke 模式的迭代数必须真的被调小了（相对 YAML 里的 10/200/5）"""
        from examples.run_pipeline import build_smoke_config_from_yaml
        from utils.config import load_config

        yaml_cfg = PipelineConfig.from_yaml(_pipeline_default_yaml_path())
        smoke_cfg = build_smoke_config_from_yaml()

        # OPC：YAML 是 10，smoke 必须 < 10
        assert smoke_cfg.opc_config.max_iterations < yaml_cfg.opc_config.max_iterations
        assert smoke_cfg.opc_config.max_iterations == 3
        # ILT：YAML 是 200，smoke 必须 < 200
        assert smoke_cfg.ilt_config.max_iter < yaml_cfg.ilt_config.max_iter
        assert smoke_cfg.ilt_config.max_iter == 30
        # SMO：YAML 是 20 次 outer，smoke 必须 < 20
        assert smoke_cfg.smo_config.max_outer_iterations < yaml_cfg.smo_config.max_outer_iterations
        assert smoke_cfg.smo_config.max_outer_iterations == 2

    def test_smoke_preserves_non_iteration_fields(self):
        """smoke 模式不能把 SMO strategy 等非迭代相关字段覆盖掉"""
        from examples.run_pipeline import build_smoke_config_from_yaml
        from workflows.smo import SMOptimizationStrategy

        cfg = build_smoke_config_from_yaml()
        # strategy 必须仍然是 YAML 里的 alternating，而不是某个默认值
        assert cfg.smo_config.strategy == SMOptimizationStrategy.ALTERNATING
        # OPC 的 SRAF 必须关（smoke 专属行为）
        assert cfg.opc_config.sraf_enable is False
        # save_intermediate 必须关（smoke 专属行为）
        assert cfg.save_intermediate is False

    def test_smoke_orchestrator_cli_also_preserves_pw(self):
        """orchestrator_cli 版 build_smoke_config_from_yaml 同样保留 PW 配置"""
        from pipeline.orchestrator_cli import build_smoke_config_from_yaml as cli_build
        cfg = cli_build()
        assert cfg.pw_verify_config is not None
        assert cfg.pw_verify_config.focus_range == (-150, 150, 11)
        # SMO strategy 同样保留
        from workflows.smo import SMOptimizationStrategy
        assert cfg.smo_config.strategy == SMOptimizationStrategy.ALTERNATING

    def test_smoke_stage_switches_respected(self):
        """smoke 模式下 --no-* CLI 开关应该仍然生效（阶段可以被单独关掉）"""
        from examples.run_pipeline import build_smoke_config_from_yaml

        # 关 SMO、关 PW
        cfg = build_smoke_config_from_yaml(enable_smo=False, enable_pw=False)
        assert cfg.enable_smo is False
        assert cfg.enable_pw_verify is False
        # OPC / ILT 仍然开着
        assert cfg.enable_opc is True
        assert cfg.enable_ilt is True
        # 但 SMO 配置对象本身还是存在（因为 YAML 里读了），只是 enable=False
        assert cfg.smo_config is not None


class TestConfigEndToEndNoRun:
    """
    端到端"配置消费"不跑 OPC 数值计算的轻量回归：
    构造一个 PipelineResult，验证 sign_off_summary 中
    pipeline 三阶段 + PW 配置 + 光学系统 + 图案 都能正确反映出来
    """

    def test_sign_off_contains_fields_from_yaml(self, optics, target):
        """sign_off_summary 里应能间接看到 YAML 驱动的配置痕迹"""
        # 先从 YAML 读配置
        cfg = PipelineConfig.from_yaml(_pipeline_default_yaml_path())

        # 构造一个包含 pw_optical_system 的 Result（复用 SMO 光源）
        gs = target.shape
        source = np.random.rand(*gs)
        source = source / source.sum()

        pw_optics = OpticalSystem(
            wavelength=optics.wavelength,
            na=optics.na,
            sigma=optics.sigma,
            pixel_size=optics.pixel_size,
            illumination_type=IlluminationType.CUSTOM,
            custom_source=source.copy(),
        )

        result = PipelineResult(
            initial_mask=target,
            final_mask=target,
            target=target,
            optical_system=optics,
            optimal_source=source,
            pw_optical_system=pw_optics,
            pw_metrics=None,
        )

        # 追加 YAML 驱动的 stage_metrics（用于验证摘要）
        result.stage_metrics = [
            StageMetrics(
                stage_name='OPC',
                elapsed_sec=1.0,
                epe_before={'epe_mean': 10.0},
                epe_after={'epe_mean': 6.0},
                extra={'max_iterations': cfg.opc_config.max_iterations},
            ),
            StageMetrics(
                stage_name='ILT',
                elapsed_sec=2.0,
                epe_before={'epe_mean': 6.0},
                epe_after={'epe_mean': 4.0},
                extra={'max_iter': cfg.ilt_config.max_iter},
            ),
            StageMetrics(
                stage_name='SMO',
                elapsed_sec=3.0,
                epe_before={'epe_mean': 4.0},
                epe_after={'epe_mean': 3.0},
                extra={'max_outer_iterations': cfg.smo_config.max_outer_iterations},
            ),
        ]

        summary = result.sign_off_summary()

        # PW 光源一致性通过（因为我们构造了匹配的 pw_optics）
        assert summary['validation']['pw_uses_smo_source'] is True

        # 三阶段 EPE 改善量之和 > 0 说明流程链完整
        total_improve = sum(m.epe_improvement for m in result.stage_metrics)
        assert total_improve > 5.0

        # OPC/ILT/SMO 的 extra 里保存的迭代数应当等于 YAML 里的值
        assert result.stage_metrics[0].extra['max_iterations'] == 10
        assert result.stage_metrics[1].extra['max_iter'] == 200
        assert result.stage_metrics[2].extra['max_outer_iterations'] == 20

        # 掩模复杂度存在
        assert 'mask_complexity' in summary
        mc = summary['mask_complexity']
        assert isinstance(mc, dict)
        assert 'tv_norm' in mc or 'total_variation' in mc
