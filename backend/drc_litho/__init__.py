# -*- coding: utf-8 -*-
"""
版图设计规则邻近分析模块 (DRC-Litho)

对版图掩模执行光刻导向的轻量 DRC 分析（非完整 EDA DRC），
检测易导致桥连、断线、孤立线的版图拓扑模式，
在 OPC 之前预警并给出修改建议，减少后续仿真的无效迭代。

主要功能：
1. 桥连风险检测 - 间距过窄、颈部收窄、拐角密度过高
2. 断线风险检测 - 线宽过窄、急转弯、线端缩短
3. 孤立线检测 - 小特征、悬空线段、孤儿像素
4. OPC 就绪度评分 - 量化版图对 OPC 的适配程度
5. 修改建议生成 - 按优先级排序的可操作修改方案

使用方式：
    from drc_litho import LithoDRCAnalyzer, OPCAdvisor, load_default_rules

    # 加载规则
    rules = load_default_rules("duv_arf")

    # 执行分析
    analyzer = LithoDRCAnalyzer(rules)
    result = analyzer.analyze(mask_array, pixel_size=1.0)
    print(result.summary())

    # 生成修改建议
    advisor = OPCAdvisor()
    report = advisor.generate_report(result)
    print(report.summary())
"""

from .schemas import (
    LithoViolationCategory,
    LithoViolationType,
    LithoSeverity,
    OPCFeasibility,
    ViolationRegion,
    LithoViolation,
    LithoDRCResult,
)
from .rules import (
    LithoRuleType,
    LithoRuleSeverity,
    LithoRuleConfig,
    LithoDRRules,
    load_default_rules,
    load_rules_from_yaml,
)
from .geometry import (
    preprocess_mask,
    compute_distance_map,
    compute_spacing_map,
    compute_local_line_width,
    detect_neck_regions,
    find_narrow_gaps,
    label_connected_components,
    compute_component_properties,
    detect_sharp_turns,
    estimate_corner_density,
    find_dangling_lines,
    find_orphan_pixels,
    find_line_ends,
    mask_to_regions,
)
from .checker_bridge import (
    check_bridge_narrow_gap,
    check_bridge_necking,
    check_bridge_dense_corner,
)
from .checker_break import (
    check_break_thin_neck,
    check_break_sharp_turn,
    check_break_line_end,
)
from .checker_isolated import (
    check_isolated_small_feature,
    check_isolated_dangling_line,
    check_isolated_orphan_pixel,
)
from .analyzer import LithoDRCAnalyzer
from .advisor import (
    OPCAdviceAction,
    OPCAdvice,
    OPCAdviceReport,
    OPCAdvisor,
)

__all__ = [
    "LithoViolationCategory",
    "LithoViolationType",
    "LithoSeverity",
    "OPCFeasibility",
    "ViolationRegion",
    "LithoViolation",
    "LithoDRCResult",
    "LithoRuleType",
    "LithoRuleSeverity",
    "LithoRuleConfig",
    "LithoDRRules",
    "load_default_rules",
    "load_rules_from_yaml",
    "preprocess_mask",
    "compute_distance_map",
    "compute_spacing_map",
    "compute_local_line_width",
    "detect_neck_regions",
    "find_narrow_gaps",
    "label_connected_components",
    "compute_component_properties",
    "detect_sharp_turns",
    "estimate_corner_density",
    "find_dangling_lines",
    "find_orphan_pixels",
    "find_line_ends",
    "mask_to_regions",
    "check_bridge_narrow_gap",
    "check_bridge_necking",
    "check_bridge_dense_corner",
    "check_break_thin_neck",
    "check_break_sharp_turn",
    "check_break_line_end",
    "check_isolated_small_feature",
    "check_isolated_dangling_line",
    "check_isolated_orphan_pixel",
    "LithoDRCAnalyzer",
    "OPCAdviceAction",
    "OPCAdvice",
    "OPCAdviceReport",
    "OPCAdvisor",
]
