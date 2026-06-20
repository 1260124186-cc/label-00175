# -*- coding: utf-8 -*-
"""
MRC (Mask Rule Check) 模块单元测试

测试掩模制造规则检查的所有核心功能。
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from mrc.rules import (
    MRCRules,
    MRCRuleType,
    MRCRuleSeverity,
    MRCRuleConfig,
    load_default_rules,
)
from mrc.violations import (
    MRCViolation,
    MRCCheckResult,
    ViolationRegion,
    ViolationType,
)
from mrc.checkers import MRCChecker
from mrc.highlight import ViolationHighlighter, HighlightStyle
from mrc.repair import (
    RepairAdvisor,
    RepairAction,
    RepairReport,
    RepairSuggestion,
)
from mrc.gate import (
    MRCGate,
    GateCheckResult,
    GateStatus,
    GatePolicyMode,
    GatePolicyConfig,
)


# ===================================================================
# 测试辅助函数：生成各类测试掩模
# ===================================================================

def make_line_mask(width_px: int, length_px: int, canvas: int = 100,
                   offset: int = 0) -> np.ndarray:
    """生成单条水平线掩模"""
    mask = np.zeros((canvas, canvas), dtype=bool)
    y0 = canvas // 2 - width_px // 2 + offset
    y1 = y0 + width_px
    mask[y0:y1, 10:10 + length_px] = True
    return mask


def make_two_lines(width_px: int, gap_px: int, canvas: int = 100) -> np.ndarray:
    """生成两条平行线，中间有间距"""
    mask = np.zeros((canvas, canvas), dtype=bool)
    center = canvas // 2
    y0_top = center - gap_px // 2 - width_px
    y1_top = y0_top + width_px
    y0_bot = center + gap_px // 2
    y1_bot = y0_bot + width_px
    mask[y0_top:y1_top, 10:90] = True
    mask[y0_bot:y1_bot, 10:90] = True
    return mask


def make_sraf_mask(main_width: int = 20, sraf_width: int = 3,
                   gap: int = 10, canvas: int = 100) -> np.ndarray:
    """生成带 SRAF（辅助特征）的掩模"""
    mask = np.zeros((canvas, canvas), dtype=bool)
    center = canvas // 2
    y_main0 = center - main_width // 2
    y_main1 = y_main0 + main_width
    mask[y_main0:y_main1, 20:80] = True
    y_sraf0 = y_main0 - gap - sraf_width
    y_sraf1 = y_sraf0 + sraf_width
    mask[y_sraf0:y_sraf1, 20:80] = True
    return mask


def make_acute_angle_mask(canvas: int = 100) -> np.ndarray:
    """生成含锐角三角形的掩模"""
    mask = np.zeros((canvas, canvas), dtype=bool)
    cy, cx = canvas // 2, canvas // 2
    for y in range(canvas):
        for x in range(canvas):
            dy = y - cy
            dx = x - cx
            if dy >= 0 and dx >= 0 and (dx + dy) < 30 and dx > 2 and dy > 2:
                mask[y, x] = True
    return mask


def make_clean_mask(canvas: int = 100, feature_size: int = 20) -> np.ndarray:
    """生成合规掩模（线宽和间距都足够大）"""
    mask = np.zeros((canvas, canvas), dtype=bool)
    mask[20:40, 20:80] = True
    mask[60:80, 20:80] = True
    return mask


# ===================================================================
# 规则模块测试
# ===================================================================

class TestMRCRules:
    """MRC 规则定义测试"""

    def test_default_rules_creation(self):
        """测试默认规则创建"""
        rules = MRCRules()
        assert len(rules.rules) > 0
        assert MRCRuleType.MIN_LINE_WIDTH in rules.rules
        assert MRCRuleType.MIN_SPACING in rules.rules

    def test_load_default_rules_duv(self):
        """测试加载 DUV/ArF 默认规则"""
        rules = load_default_rules("duv_arf")
        assert rules.technology_node == "duv_arf"
        mw_rule = rules.get_rule(MRCRuleType.MIN_LINE_WIDTH)
        assert mw_rule.threshold_nm == 45.0

    def test_load_default_rules_euv(self):
        """测试加载 EUV 默认规则"""
        rules = load_default_rules("euv")
        assert rules.technology_node == "euv"
        mw_rule = rules.get_rule(MRCRuleType.MIN_LINE_WIDTH)
        assert mw_rule.threshold_nm == 16.0

    def test_enable_disable_rules(self):
        """测试规则启用/禁用"""
        rules = MRCRules()
        rules.disable_rule(MRCRuleType.MIN_LINE_WIDTH)
        assert not rules.rules[MRCRuleType.MIN_LINE_WIDTH].enabled
        rules.enable_rule(MRCRuleType.MIN_LINE_WIDTH)
        assert rules.rules[MRCRuleType.MIN_LINE_WIDTH].enabled

    def test_enabled_rules(self):
        """测试获取启用的规则"""
        rules = MRCRules()
        enabled = rules.enabled_rules()
        assert len(enabled) > 0
        for rt, rc in enabled.items():
            assert rc.enabled is True

    def test_rule_config_to_dict_from_dict(self):
        """测试规则配置序列化/反序列化"""
        config = MRCRuleConfig(
            rule_type=MRCRuleType.MIN_LINE_WIDTH,
            enabled=True,
            threshold_nm=50.0,
            severity=MRCRuleSeverity.ERROR,
            description="Test rule",
        )
        d = config.to_dict()
        restored = MRCRuleConfig.from_dict(d)
        assert restored.rule_type == config.rule_type
        assert restored.threshold_nm == config.threshold_nm
        assert restored.severity == config.severity

    def test_rules_to_dict_from_dict(self):
        """测试规则集合序列化/反序列化"""
        rules = load_default_rules("duv_arf")
        d = rules.to_dict()
        restored = MRCRules.from_dict(d)
        assert restored.technology_node == rules.technology_node
        assert len(restored.rules) == len(rules.rules)


# ===================================================================
# 违规数据结构测试
# ===================================================================

class TestViolationStructures:
    """违规数据结构测试"""

    def test_violation_region(self):
        """测试违规区域"""
        region = ViolationRegion(
            bbox=(10, 20, 30, 40),
            centroid=(20.0, 30.0),
            area_pixels=100,
        )
        assert region.bbox_size == (20, 20)
        d = region.to_dict()
        assert d["bbox"] == [10, 20, 30, 40]
        assert d["area_pixels"] == 100

    def test_mrc_violation(self):
        """测试单条违规"""
        region = ViolationRegion(
            bbox=(0, 0, 10, 10),
            centroid=(5.0, 5.0),
            area_pixels=50,
        )
        violation = MRCViolation(
            rule_type=MRCRuleType.MIN_LINE_WIDTH,
            violation_type=ViolationType.LINE_WIDTH_TOO_SMALL,
            severity=MRCRuleSeverity.ERROR,
            message="线宽过小",
            region=region,
            measurement_nm=40.0,
            threshold_nm=45.0,
            pixel_size=1.0,
        )
        assert not violation.is_fatal
        assert violation.violation_area_nm2 == 50.0

    def test_check_result_empty(self):
        """测试空检查结果"""
        result = MRCCheckResult()
        assert result.passed
        assert result.total_violations == 0

    def test_check_result_with_violations(self):
        """测试含违规的检查结果"""
        region = ViolationRegion(
            bbox=(0, 0, 10, 10),
            centroid=(5.0, 5.0),
            area_pixels=50,
        )
        v1 = MRCViolation(
            rule_type=MRCRuleType.MIN_LINE_WIDTH,
            violation_type=ViolationType.LINE_WIDTH_TOO_SMALL,
            severity=MRCRuleSeverity.FATAL,
            message="fatal",
            region=region,
        )
        v2 = MRCViolation(
            rule_type=MRCRuleType.MIN_SPACING,
            violation_type=ViolationType.SPACING_TOO_SMALL,
            severity=MRCRuleSeverity.WARNING,
            message="warning",
            region=region,
        )
        result = MRCCheckResult(violations=[v1, v2])
        assert not result.passed
        assert result.has_fatal
        assert result.fatal_count == 1
        assert result.warning_count == 1
        assert result.total_violations == 2

    def test_check_result_violations_by_rule(self):
        """测试按规则分组违规"""
        region = ViolationRegion(
            bbox=(0, 0, 10, 10),
            centroid=(5.0, 5.0),
            area_pixels=50,
        )
        violations = []
        for i in range(3):
            violations.append(MRCViolation(
                rule_type=MRCRuleType.MIN_LINE_WIDTH,
                violation_type=ViolationType.LINE_WIDTH_TOO_SMALL,
                severity=MRCRuleSeverity.ERROR,
                message=f"v{i}",
                region=region,
            ))
        result = MRCCheckResult(violations=violations)
        grouped = result.violations_by_rule()
        assert len(grouped[MRCRuleType.MIN_LINE_WIDTH]) == 3

    def test_check_result_save_json(self):
        """测试检查结果保存为 JSON"""
        result = MRCCheckResult(
            violations=[],
            mask_shape=(100, 100),
            pixel_size=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "result.json"
            result.save_json(str(fp))
            assert fp.exists()


# ===================================================================
# 核心检查引擎测试
# ===================================================================

class TestMRCChecker:
    """MRC 规则检查器测试"""

    def setup_method(self):
        """为每个测试创建检查器"""
        self.rules = MRCRules()
        self.checker = MRCChecker(self.rules)

    def test_preprocess_mask_bool(self):
        """测试二值掩模预处理（布尔型）"""
        mask = np.array([[True, False], [False, True]])
        result = MRCChecker._preprocess_mask(mask)
        assert result.dtype == bool
        assert result.shape == (2, 2)

    def test_preprocess_mask_float01(self):
        """测试二值掩模预处理（0-1 float）"""
        mask = np.array([[1.0, 0.0], [0.3, 0.8]])
        result = MRCChecker._preprocess_mask(mask)
        assert result[0, 0] == True
        assert result[0, 1] == False
        assert result[1, 0] == False
        assert result[1, 1] == True

    def test_preprocess_mask_uint8(self):
        """测试二值掩模预处理（0-255 uint8）"""
        mask = np.array([[255, 0], [100, 200]], dtype=np.uint8)
        result = MRCChecker._preprocess_mask(mask)
        assert result[0, 0] == True
        assert result[1, 1] == True

    def test_check_empty_mask(self):
        """测试空掩模检查"""
        mask = np.zeros((50, 50), dtype=bool)
        result = self.checker.check(mask, pixel_size=1.0)
        assert result.total_violations == 0
        assert result.passed

    def test_check_clean_mask_no_violations(self):
        """测试合规掩模无违规"""
        mask = make_clean_mask()
        result = self.checker.check(mask, pixel_size=3.0)
        assert result.passed

    def test_min_line_width_violation_detected(self):
        """测试最小线宽违规检测"""
        mask = make_line_mask(width_px=5, length_px=80)
        rules = MRCRules()
        rules.set_rule(MRCRuleType.MIN_LINE_WIDTH, MRCRuleConfig(
            rule_type=MRCRuleType.MIN_LINE_WIDTH,
            enabled=True,
            threshold_nm=20.0,
            severity=MRCRuleSeverity.FATAL,
        ))
        checker = MRCChecker(rules)
        result = checker.check(mask, pixel_size=2.0)
        lw_violations = result.get_violations_by_type(MRCRuleType.MIN_LINE_WIDTH)
        assert len(lw_violations) > 0

    def test_min_spacing_violation_detected(self):
        """测试最小间距违规检测"""
        mask = make_two_lines(width_px=15, gap_px=5)
        rules = MRCRules()
        rules.set_rule(MRCRuleType.MIN_SPACING, MRCRuleConfig(
            rule_type=MRCRuleType.MIN_SPACING,
            enabled=True,
            threshold_nm=20.0,
            severity=MRCRuleSeverity.FATAL,
        ))
        checker = MRCChecker(rules)
        result = checker.check(mask, pixel_size=2.0)
        sp_violations = result.get_violations_by_type(MRCRuleType.MIN_SPACING)
        assert len(sp_violations) > 0

    def test_min_sraf_size_violation_detected(self):
        """测试最小 SRAF 尺寸违规检测"""
        mask = make_sraf_mask(main_width=30, sraf_width=2, gap=10)
        rules = MRCRules()
        rules.set_rule(MRCRuleType.MIN_SRAF_SIZE, MRCRuleConfig(
            rule_type=MRCRuleType.MIN_SRAF_SIZE,
            enabled=True,
            threshold_nm=10.0,
            severity=MRCRuleSeverity.ERROR,
        ))
        checker = MRCChecker(rules)
        result = checker.check(mask, pixel_size=2.0)
        sraf_violations = result.get_violations_by_type(MRCRuleType.MIN_SRAF_SIZE)
        assert len(sraf_violations) > 0

    def test_separate_features_with_target_mask(self):
        """测试使用 target_mask 区分主特征和辅助特征"""
        main = np.zeros((50, 50), dtype=bool)
        main[20:30, 10:40] = True
        sraf = np.zeros((50, 50), dtype=bool)
        sraf[10:12, 10:40] = True
        mask = main | sraf
        target = main.copy()
        checker = MRCChecker()
        m, s = checker._separate_features(mask, target)
        assert np.array_equal(m, main)
        assert np.array_equal(s, sraf)

    def test_separate_features_auto(self):
        """测试自动区分主特征和辅助特征"""
        mask = make_sraf_mask(main_width=20, sraf_width=2, gap=5, canvas=80)
        checker = MRCChecker()
        main, sraf = checker._separate_features(mask, None)
        assert np.any(main)
        assert np.any(sraf)
        assert not np.any(main & sraf)

    def test_check_result_summary(self):
        """测试检查结果摘要输出"""
        mask = make_clean_mask()
        result = self.checker.check(mask, pixel_size=2.0)
        summary = result.summary()
        assert "MRC 检查结果" in summary
        assert "通过" in summary


# ===================================================================
# 违规高亮测试
# ===================================================================

class TestViolationHighlighter:
    """违规高亮标注测试"""

    def setup_method(self):
        self.highlighter = ViolationHighlighter()

    def test_generate_overlay_empty(self):
        """测试空违规生成叠加图"""
        mask = make_clean_mask()
        overlay = self.highlighter.generate_overlay_mask(mask, [])
        assert overlay.shape == (mask.shape[0], mask.shape[1], 3)
        assert overlay.min() >= 0.0
        assert overlay.max() <= 1.0

    def test_generate_overlay_with_violations(self):
        """测试含违规生成叠加图"""
        mask = make_clean_mask()
        region = ViolationRegion(
            bbox=(20, 20, 40, 40),
            centroid=(30.0, 30.0),
            area_pixels=100,
            mask_slice=np.ones((20, 20), dtype=bool),
        )
        violation = MRCViolation(
            rule_type=MRCRuleType.MIN_LINE_WIDTH,
            violation_type=ViolationType.LINE_WIDTH_TOO_SMALL,
            severity=MRCRuleSeverity.ERROR,
            message="test",
            region=region,
        )
        overlay = self.highlighter.generate_overlay_mask(mask, [violation])
        assert overlay.shape == (mask.shape[0], mask.shape[1], 3)

    def test_generate_heatmap(self):
        """测试违规热力图生成"""
        mask = make_clean_mask()
        region = ViolationRegion(
            bbox=(20, 20, 40, 40),
            centroid=(30.0, 30.0),
            area_pixels=100,
            mask_slice=np.ones((20, 20), dtype=bool),
        )
        violation = MRCViolation(
            rule_type=MRCRuleType.MIN_LINE_WIDTH,
            violation_type=ViolationType.LINE_WIDTH_TOO_SMALL,
            severity=MRCRuleSeverity.ERROR,
            message="test",
            region=region,
        )
        heatmap = self.highlighter.generate_heatmap(mask, [violation])
        assert heatmap.shape == mask.shape
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_highlight_style_color_mapping(self):
        """测试高亮样式颜色映射"""
        style = HighlightStyle()
        assert len(style.get_color(MRCRuleSeverity.FATAL)) == 3
        assert len(style.get_color(MRCRuleSeverity.ERROR)) == 3

    def test_save_visualization(self):
        """测试保存可视化图片"""
        mask = make_clean_mask()
        result = MRCCheckResult(
            violations=[],
            mask_shape=mask.shape,
            pixel_size=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "viz.png"
            self.highlighter.save_visualization(mask, result, str(fp), dpi=50)
            assert fp.exists()


# ===================================================================
# 修复建议测试
# ===================================================================

class TestRepairAdvisor:
    """修复建议生成测试"""

    def setup_method(self):
        self.advisor = RepairAdvisor()

    def _make_violation(self, vtype: ViolationType,
                        rtype: MRCRuleType,
                        severity: MRCRuleSeverity = MRCRuleSeverity.ERROR,
                        measurement: float = 30.0,
                        threshold: float = 45.0) -> MRCViolation:
        region = ViolationRegion(
            bbox=(10, 10, 30, 30),
            centroid=(20.0, 20.0),
            area_pixels=100,
        )
        return MRCViolation(
            rule_type=rtype,
            violation_type=vtype,
            severity=severity,
            message=f"test {vtype.value}",
            region=region,
            measurement_nm=measurement,
            threshold_nm=threshold,
            pixel_size=1.0,
        )

    def test_generate_report_empty(self):
        """测试空违规生成修复报告"""
        result = MRCCheckResult(violations=[])
        report = self.advisor.generate_report(result)
        assert report.total_suggestions == 0

    def test_advise_line_width(self):
        """测试线宽违规修复建议"""
        violation = self._make_violation(
            ViolationType.LINE_WIDTH_TOO_SMALL,
            MRCRuleType.MIN_LINE_WIDTH,
            measurement=30.0, threshold=45.0,
        )
        suggestion = RepairAdvisor._advise_line_width(violation)
        assert suggestion.action == RepairAction.WIDEN_LINE
        assert suggestion.priority > 0
        assert suggestion.auto_fixable is True
        assert len(suggestion.steps) > 0

    def test_advise_spacing(self):
        """测试间距违规修复建议"""
        violation = self._make_violation(
            ViolationType.SPACING_TOO_SMALL,
            MRCRuleType.MIN_SPACING,
            measurement=30.0, threshold=45.0,
        )
        suggestion = RepairAdvisor._advise_spacing(violation)
        assert suggestion.action == RepairAction.INCREASE_SPACING
        assert suggestion.auto_fixable is False

    def test_advise_acute_angle(self):
        """测试锐角违规修复建议"""
        violation = self._make_violation(
            ViolationType.ACUTE_ANGLE,
            MRCRuleType.NO_ACUTE_ANGLE,
            measurement=45.0, threshold=90.0,
        )
        violation.extra_info["estimated_angle_deg"] = 45.0
        suggestion = RepairAdvisor._advise_acute_angle(violation)
        assert suggestion.action == RepairAction.ROUND_CORNER

    def test_advise_sraf_too_close(self):
        """测试 SRAF 过近修复建议"""
        violation = self._make_violation(
            ViolationType.SRAF_TOO_CLOSE_TO_MAIN,
            MRCRuleType.SRAF_MAIN_DISTANCE,
            measurement=30.0, threshold=60.0,
        )
        suggestion = RepairAdvisor._advise_sraf_too_close(violation)
        assert suggestion.action == RepairAction.MOVE_SRAF

    def test_full_report_with_multiple_violations(self):
        """测试生成含多种违规的完整报告"""
        violations = [
            self._make_violation(
                ViolationType.LINE_WIDTH_TOO_SMALL,
                MRCRuleType.MIN_LINE_WIDTH,
                MRCRuleSeverity.FATAL,
            ),
            self._make_violation(
                ViolationType.SPACING_TOO_SMALL,
                MRCRuleType.MIN_SPACING,
            ),
            self._make_violation(
                ViolationType.ACUTE_ANGLE,
                MRCRuleType.NO_ACUTE_ANGLE,
                MRCRuleSeverity.WARNING,
            ),
        ]
        result = MRCCheckResult(violations=violations)
        report = self.advisor.generate_report(result)
        assert report.total_suggestions == 3
        assert report.auto_fixable_count >= 1
        sorted_suggestions = report.get_sorted_by_priority()
        assert sorted_suggestions[0].priority >= sorted_suggestions[-1].priority

    def test_report_summary(self):
        """测试修复报告摘要"""
        violations = [
            self._make_violation(
                ViolationType.LINE_WIDTH_TOO_SMALL,
                MRCRuleType.MIN_LINE_WIDTH,
            ),
        ]
        result = MRCCheckResult(violations=violations)
        report = self.advisor.generate_report(result)
        summary = report.summary()
        assert "MRC 修复建议报告" in summary
        assert "总建议数" in summary


# ===================================================================
# Tapeout 门禁测试
# ===================================================================

class TestMRCGate:
    """Tapeout 门禁检查测试"""

    def _make_violation(self, severity: MRCRuleSeverity,
                        rtype: MRCRuleType = MRCRuleType.MIN_LINE_WIDTH
                        ) -> MRCViolation:
        region = ViolationRegion(
            bbox=(0, 0, 10, 10),
            centroid=(5.0, 5.0),
            area_pixels=50,
        )
        return MRCViolation(
            rule_type=rtype,
            violation_type=ViolationType.LINE_WIDTH_TOO_SMALL,
            severity=severity,
            message=f"test {severity.value}",
            region=region,
        )

    def test_policy_mode_strict(self):
        """测试 STRICT 策略模式"""
        gate = MRCGate(policy_mode="strict")
        assert gate.policy_config.mode == GatePolicyMode.STRICT
        assert gate.policy_config.block_on_warning is True
        assert gate.policy_config.allow_waive is False

    def test_policy_mode_relaxed(self):
        """测试 RELAXED 策略模式"""
        gate = MRCGate(policy_mode="relaxed")
        assert gate.policy_config.mode == GatePolicyMode.RELAXED
        assert gate.policy_config.block_on_error is False

    def test_policy_mode_normal(self):
        """测试 NORMAL 策略模式"""
        gate = MRCGate(policy_mode="normal")
        assert gate.policy_config.mode == GatePolicyMode.NORMAL
        assert gate.policy_config.block_on_fatal is True
        assert gate.policy_config.block_on_error is True

    def test_validate_passing(self):
        """测试合规结果通过门禁"""
        mrc_result = MRCCheckResult(violations=[])
        gate = MRCGate()
        gate_result = gate.validate(mrc_result)
        assert gate_result.passed
        assert gate_result.status == GateStatus.PASS

    def test_validate_fail_on_fatal(self):
        """测试致命违规阻止流片"""
        violations = [
            self._make_violation(MRCRuleSeverity.FATAL),
        ]
        mrc_result = MRCCheckResult(violations=violations)
        gate = MRCGate()
        gate_result = gate.validate(mrc_result)
        assert not gate_result.passed
        assert gate_result.status == GateStatus.FAIL
        assert len(gate_result.blocking_violations) >= 1

    def test_validate_fail_on_error_normal_mode(self):
        """测试 NORMAL 模式下错误违规阻止流片"""
        violations = [
            self._make_violation(MRCRuleSeverity.ERROR),
        ]
        mrc_result = MRCCheckResult(violations=violations)
        gate = MRCGate(policy_mode="normal")
        gate_result = gate.validate(mrc_result)
        assert not gate_result.passed

    def test_validate_pass_error_in_relaxed_mode(self):
        """测试 RELAXED 模式下错误违规不阻止流片"""
        violations = [
            self._make_violation(MRCRuleSeverity.ERROR, MRCRuleType.MIN_SRAF_SIZE),
        ]
        mrc_result = MRCCheckResult(violations=violations)
        gate = MRCGate(policy_mode="relaxed")
        gate_result = gate.validate(mrc_result)
        assert gate_result.passed
        assert gate_result.status == GateStatus.PASS

    def test_waive_rule(self):
        """测试规则豁免"""
        violations = [
            self._make_violation(
                MRCRuleSeverity.ERROR, MRCRuleType.MIN_LINE_WIDTH
            ),
        ]
        mrc_result = MRCCheckResult(violations=violations)
        gate = MRCGate()
        gate.waive_rule(MRCRuleType.MIN_LINE_WIDTH)
        gate_result = gate.validate(mrc_result)
        assert len(gate_result.waived_violations_list) == 1

    def test_required_rules(self):
        """测试必需规则检查"""
        violations = [
            self._make_violation(
                MRCRuleSeverity.WARNING, MRCRuleType.MIN_SPACING
            ),
        ]
        mrc_result = MRCCheckResult(violations=violations)
        gate = MRCGate(policy_mode="relaxed")
        gate.add_required_rule(MRCRuleType.MIN_SPACING)
        gate_result = gate.validate(mrc_result)
        assert not gate_result.passed

    def test_violation_area_ratio(self):
        """测试违规面积占比检查"""
        violations = []
        for _ in range(50):
            region = ViolationRegion(
                bbox=(0, 0, 10, 10),
                centroid=(5.0, 5.0),
                area_pixels=100,
            )
            violations.append(MRCViolation(
                rule_type=MRCRuleType.MIN_LINE_WIDTH,
                violation_type=ViolationType.LINE_WIDTH_TOO_SMALL,
                severity=MRCRuleSeverity.WARNING,
                message="test",
                region=region,
            ))
        mrc_result = MRCCheckResult(violations=violations)
        mask = np.zeros((100, 100), dtype=bool)
        gate = MRCGate()
        gate.policy_config.max_violation_area_ratio = 0.01
        gate_result = gate.validate(mrc_result, mask=mask)
        area_item = [i for i in gate_result.check_items if i.name == "违规面积占比"]
        assert len(area_item) == 1
        assert area_item[0].passed is False

    def test_set_policy_mode(self):
        """测试切换策略模式"""
        gate = MRCGate(policy_mode="normal")
        assert gate.policy_config.mode == GatePolicyMode.NORMAL
        gate.set_policy_mode("strict")
        assert gate.policy_config.mode == GatePolicyMode.STRICT

    def test_gate_result_summary(self):
        """测试门禁结果摘要"""
        mrc_result = MRCCheckResult(violations=[])
        gate = MRCGate()
        gate_result = gate.validate(mrc_result)
        summary = gate_result.summary()
        assert "Tapeout 门禁" in summary
        assert "通过" in summary

    def test_gate_result_save_json(self):
        """测试门禁结果保存 JSON"""
        mrc_result = MRCCheckResult(violations=[])
        gate = MRCGate()
        gate_result = gate.validate(mrc_result)
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "gate.json"
            gate_result.save_json(str(fp))
            assert fp.exists()


# ===================================================================
# 端到端集成测试
# ===================================================================

class TestMRCEndToEnd:
    """MRC 模块端到端测试"""

    def test_full_pipeline_with_violations(self):
        """测试含违规的完整 MRC 流程"""
        mask = make_line_mask(width_px=3, length_px=80)
        rules = load_default_rules("duv_arf")
        rules.set_rule(MRCRuleType.MIN_LINE_WIDTH, MRCRuleConfig(
            rule_type=MRCRuleType.MIN_LINE_WIDTH,
            enabled=True,
            threshold_nm=15.0,
            severity=MRCRuleSeverity.FATAL,
        ))
        checker = MRCChecker(rules)
        mrc_result = checker.check(mask, pixel_size=2.0)
        assert not mrc_result.passed
        advisor = RepairAdvisor()
        repair_report = advisor.generate_report(mrc_result)
        assert repair_report.total_suggestions > 0
        highlighter = ViolationHighlighter()
        overlay = highlighter.generate_overlay_mask(mask, mrc_result.violations)
        assert overlay.ndim == 3
        gate = MRCGate()
        gate_result = gate.validate(mrc_result, mask=mask)
        assert not gate_result.passed

    def test_full_pipeline_clean(self):
        """测试合规掩模的完整 MRC 流程"""
        mask = make_clean_mask()
        rules = load_default_rules("duv_arf")
        checker = MRCChecker(rules)
        mrc_result = checker.check(mask, pixel_size=5.0)
        assert mrc_result.passed
        advisor = RepairAdvisor()
        repair_report = advisor.generate_report(mrc_result)
        assert repair_report.total_suggestions == 0
        gate = MRCGate()
        gate_result = gate.validate(mrc_result, mask=mask)
        assert gate_result.passed
        assert gate_result.status == GateStatus.PASS

    def test_euv_rules_pipeline(self):
        """测试 EUV 规则下的流程"""
        mask = make_clean_mask(feature_size=10, canvas=60)
        rules = load_default_rules("euv")
        checker = MRCChecker(rules)
        mrc_result = checker.check(mask, pixel_size=1.0)
        gate = MRCGate(policy_mode="strict")
        gate_result = gate.validate(mrc_result)
        summary = gate_result.summary()
        assert "Tapeout 门禁" in summary
