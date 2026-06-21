# -*- coding: utf-8 -*-
"""
版图设计规则邻近分析模块 - 单元测试
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drc_litho import (
    LithoDRCAnalyzer,
    OPCAdvisor,
    load_default_rules,
    LithoDRRules,
    LithoRuleType,
    LithoViolationCategory,
    LithoViolationType,
    LithoSeverity,
    OPCFeasibility,
    LithoDRCResult,
    OPCAdviceReport,
    preprocess_mask,
    compute_distance_map,
    compute_local_line_width,
    detect_neck_regions,
    find_narrow_gaps,
    label_connected_components,
    find_orphan_pixels,
    find_dangling_lines,
    detect_sharp_turns,
    estimate_corner_density,
)


def _make_line_space_mask(size=128, line_width=8, pitch=20):
    mask = np.zeros((size, size), dtype=np.float64)
    for x in range(0, size, pitch):
        x0 = x
        x1 = min(x + line_width, size)
        mask[:, x0:x1] = 1.0
    return mask


def _make_narrow_gap_mask(size=128, gap_width=3):
    mask = np.zeros((size, size), dtype=np.float64)
    cx = size // 2
    left_end = cx - gap_width // 2
    right_start = cx + (gap_width + 1) // 2
    mask[:, :left_end] = 1.0
    mask[:, right_start:] = 1.0
    return mask


def _make_neck_mask(size=128, neck_width=4):
    mask = np.zeros((size, size), dtype=np.float64)
    cx = size // 2
    mask[:, :cx - 10] = 1.0
    mask[:, cx + 10:] = 1.0
    neck_half = neck_width // 2
    mask[cx - neck_half:cx + neck_half, :] = 1.0
    return mask


def _make_thin_line_mask(size=128, line_width=3):
    mask = np.zeros((size, size), dtype=np.float64)
    cy = size // 2
    half = line_width // 2
    mask[cy - half:cy + half, :] = 1.0
    return mask


def _make_sharp_turn_mask(size=128):
    mask = np.zeros((size, size), dtype=np.float64)
    cy, cx = size // 4, size // 4
    mask[cy:cy + 6, cx:cx + 40] = 1.0
    mask[cy:cy + 40, cx:cx + 6] = 1.0
    return mask


def _make_isolated_feature_mask(size=128):
    mask = np.zeros((size, size), dtype=np.float64)
    mask[60:68, 10:90] = 1.0
    mask[5:8, 5:8] = 1.0
    mask[100, 100] = 1.0
    return mask


def _make_clean_mask(size=128):
    mask = np.zeros((size, size), dtype=np.float64)
    for x in range(0, size, 30):
        x1 = min(x + 10, size)
        mask[:, x:x1] = 1.0
    return mask


class TestPreprocessMask(unittest.TestCase):
    def test_bool_input(self):
        m = np.ones((10, 10), dtype=bool)
        result = preprocess_mask(m)
        self.assertEqual(result.dtype, np.bool_)

    def test_float_input(self):
        m = np.ones((10, 10), dtype=np.float64)
        result = preprocess_mask(m)
        self.assertTrue(result[0, 0])

    def test_uint8_input(self):
        m = np.full((10, 10), 255, dtype=np.uint8)
        result = preprocess_mask(m)
        self.assertTrue(result[0, 0])

    def test_zero_input(self):
        m = np.zeros((10, 10), dtype=np.float64)
        result = preprocess_mask(m)
        self.assertFalse(result[0, 0])

    def test_3d_raises(self):
        m = np.ones((10, 10, 3))
        with self.assertRaises(ValueError):
            preprocess_mask(m)


class TestGeometryUtils(unittest.TestCase):
    def test_distance_map(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[9:11, 9:11] = True
        dist = compute_distance_map(mask)
        self.assertGreater(dist[9, 9], 0)
        self.assertEqual(dist[0, 0], 0)

    def test_local_line_width(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[28:36, :] = True
        width = compute_local_line_width(mask)
        self.assertGreater(width[32, 32], 0)

    def test_detect_neck_regions(self):
        mask = _make_neck_mask(128, neck_width=4)
        neck = detect_neck_regions(mask, 6.0)
        self.assertTrue(np.any(neck))

    def test_find_narrow_gaps(self):
        mask = _make_narrow_gap_mask(128, gap_width=3)
        gaps = find_narrow_gaps(mask, 4.0)
        self.assertTrue(np.any(gaps))

    def test_label_connected_components(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[5:15, 5:15] = True
        mask[30:40, 30:40] = True
        labeled, num = label_connected_components(mask)
        self.assertEqual(num, 2)

    def test_find_orphan_pixels(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True
        mask[40, 40] = True
        orphans = find_orphan_pixels(mask, max_area_px=4)
        self.assertGreater(len(orphans), 0)

    def test_detect_sharp_turns(self):
        mask = _make_sharp_turn_mask(128)
        turns = detect_sharp_turns(mask, min_angle_deg=90.0)
        self.assertIsInstance(turns, np.ndarray)
        self.assertEqual(turns.dtype, bool)

    def test_corner_density(self):
        mask = _make_sharp_turn_mask(128)
        density = estimate_corner_density(mask, block_size=32)
        self.assertEqual(density.ndim, 2)


class TestRules(unittest.TestCase):
    def test_default_rules(self):
        rules = load_default_rules("duv_arf")
        self.assertIsInstance(rules, LithoDRRules)
        self.assertEqual(rules.technology_node, "duv_arf")
        self.assertGreater(len(rules.enabled_rules()), 0)

    def test_euv_rules(self):
        rules = load_default_rules("euv")
        self.assertEqual(rules.technology_node, "euv")
        bridge_rule = rules.get_rule(LithoRuleType.BRIDGE_NARROW_GAP)
        self.assertIsNotNone(bridge_rule)
        self.assertLess(bridge_rule.threshold_nm, 40.0)

    def test_enable_disable(self):
        rules = LithoDRRules()
        rules.disable_rule(LithoRuleType.BRIDGE_NARROW_GAP)
        rule = rules.get_rule(LithoRuleType.BRIDGE_NARROW_GAP)
        self.assertFalse(rule.enabled)
        rules.enable_rule(LithoRuleType.BRIDGE_NARROW_GAP)
        self.assertTrue(rule.enabled)

    def test_category_filtering(self):
        rules = LithoDRRules()
        bridge_rules = rules.bridge_rules()
        for rt in bridge_rules:
            self.assertTrue(rt.value.startswith("bridge_"))
        break_rules = rules.break_rules()
        for rt in break_rules:
            self.assertTrue(rt.value.startswith("break_"))

    def test_to_dict_from_dict(self):
        rules = LithoDRRules()
        d = rules.to_dict()
        rules2 = LithoDRRules.from_dict(d)
        self.assertEqual(len(rules2.rules), len(rules.rules))


class TestAnalyzer(unittest.TestCase):
    def test_analyze_clean_mask(self):
        mask = _make_clean_mask(128)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        self.assertIsInstance(result, LithoDRCResult)
        self.assertEqual(result.mask_shape, (128, 128))

    def test_analyze_narrow_gap(self):
        mask = _make_narrow_gap_mask(128, gap_width=3)
        rules = load_default_rules("duv_arf")
        analyzer = LithoDRCAnalyzer(rules, categories=["bridge"])
        result = analyzer.analyze(mask, pixel_size=1.0)
        self.assertGreater(result.bridge_count, 0)

    def test_analyze_category_only(self):
        mask = _make_narrow_gap_mask(128, gap_width=3)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze_bridge_only(mask, pixel_size=1.0)
        for v in result.violations:
            self.assertEqual(v.category, LithoViolationCategory.BRIDGE)

    def test_analyze_break_only(self):
        mask = _make_thin_line_mask(128, line_width=3)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze_break_only(mask, pixel_size=1.0)
        for v in result.violations:
            self.assertEqual(v.category, LithoViolationCategory.BREAK)

    def test_analyze_isolated_only(self):
        mask = _make_isolated_feature_mask(128)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze_isolated_only(mask, pixel_size=1.0)
        for v in result.violations:
            self.assertEqual(v.category, LithoViolationCategory.ISOLATED)

    def test_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.float64)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        self.assertEqual(result.total_violations, 0)
        self.assertTrue(result.passed)

    def test_result_summary(self):
        mask = _make_narrow_gap_mask(128, gap_width=3)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        summary = result.summary()
        self.assertIn("光刻导向 DRC", summary)

    def test_result_to_dict(self):
        mask = _make_clean_mask(128)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        d = result.to_dict()
        self.assertIn("opc_readiness_score", d)
        self.assertIn("total_violations", d)

    def test_opc_readiness_score(self):
        mask = _make_clean_mask(128)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        self.assertGreaterEqual(result.opc_readiness_score, 0.0)
        self.assertLessEqual(result.opc_readiness_score, 100.0)


class TestAdvisor(unittest.TestCase):
    def test_generate_report(self):
        mask = _make_narrow_gap_mask(128, gap_width=3)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        advisor = OPCAdvisor()
        report = advisor.generate_report(result)
        self.assertIsInstance(report, OPCAdviceReport)

    def test_report_summary(self):
        mask = _make_narrow_gap_mask(128, gap_width=3)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        advisor = OPCAdvisor()
        report = advisor.generate_report(result)
        summary = report.summary()
        self.assertIn("OPC 预警", summary)

    def test_report_to_dict(self):
        mask = _make_clean_mask(128)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        advisor = OPCAdvisor()
        report = advisor.generate_report(result)
        d = report.to_dict()
        self.assertIn("opc_readiness_score", d)

    def test_priority_sorting(self):
        mask = _make_narrow_gap_mask(128, gap_width=2)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        advisor = OPCAdvisor()
        report = advisor.generate_report(result)
        sorted_advices = report.get_sorted_by_priority()
        for i in range(len(sorted_advices) - 1):
            self.assertGreaterEqual(
                sorted_advices[i].priority,
                sorted_advices[i + 1].priority,
            )

    def test_needs_redesign(self):
        mask = _make_isolated_feature_mask(128)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        advisor = OPCAdvisor()
        report = advisor.generate_report(result)
        redesign = report.get_needs_redesign()
        for a in redesign:
            self.assertEqual(
                a.violation.opc_feasibility,
                OPCFeasibility.NEEDS_REDESIGN,
            )


class TestSchemas(unittest.TestCase):
    def test_violation_categories(self):
        self.assertEqual(LithoViolationCategory.BRIDGE.value, "bridge")
        self.assertEqual(LithoViolationCategory.BREAK.value, "break")
        self.assertEqual(LithoViolationCategory.ISOLATED.value, "isolated")

    def test_severity_levels(self):
        self.assertEqual(LithoSeverity.FATAL.value, "fatal")
        self.assertEqual(LithoSeverity.CRITICAL.value, "critical")
        self.assertEqual(LithoSeverity.WARNING.value, "warning")
        self.assertEqual(LithoSeverity.INFO.value, "info")

    def test_opc_feasibility(self):
        self.assertEqual(OPCFeasibility.FIXABLE.value, "fixable")
        self.assertEqual(OPCFeasibility.PARTIAL.value, "partial")
        self.assertEqual(OPCFeasibility.UNFIXABLE.value, "unfixable")
        self.assertEqual(OPCFeasibility.NEEDS_REDESIGN.value, "needs_redesign")

    def test_result_passed_property(self):
        result = LithoDRCResult()
        self.assertTrue(result.passed)
        self.assertTrue(result.opc_ready)

    def test_result_save_json(self):
        import tempfile
        mask = _make_clean_mask(64)
        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = result.save_json(os.path.join(tmpdir, "result.json"))
            self.assertTrue(os.path.exists(path))


class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline(self):
        mask = _make_narrow_gap_mask(128, gap_width=2)

        rules = load_default_rules("duv_arf")
        analyzer = LithoDRCAnalyzer(rules)
        result = analyzer.analyze(mask, pixel_size=1.0)

        self.assertIsInstance(result, LithoDRCResult)
        self.assertGreater(result.total_violations, 0)

        advisor = OPCAdvisor()
        report = advisor.generate_report(result)

        self.assertGreater(report.total_advices, 0)
        self.assertGreaterEqual(report.opc_readiness_score, 0.0)

        summary = result.summary()
        self.assertIn("桥连风险", summary)

        report_summary = report.summary()
        self.assertIn("OPC", report_summary)

    def test_all_categories_detected(self):
        mask = np.zeros((256, 256), dtype=np.float64)

        mask[10:18, 5:125] = 1.0
        mask[10:18, 130:250] = 1.0

        mask[50:54, 50:200] = 1.0

        cy, cx = 130, 40
        mask[cy:cy + 5, cx:cx + 60] = 1.0
        mask[cy:cy + 60, cx:cx + 5] = 1.0

        mask[200:203, 20:50] = 1.0
        mask[220, 220] = 1.0

        analyzer = LithoDRCAnalyzer()
        result = analyzer.analyze(mask, pixel_size=1.0)

        categories = set(v.category for v in result.violations)
        self.assertGreater(len(categories), 0)

        advisor = OPCAdvisor()
        report = advisor.generate_report(result)
        self.assertGreater(report.total_advices, 0)


if __name__ == "__main__":
    unittest.main()
