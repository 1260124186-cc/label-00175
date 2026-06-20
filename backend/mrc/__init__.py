# -*- coding: utf-8 -*-
"""
掩模制造规则检查 (Mask Rule Check, MRC) 模块

对 OPC/ILT/SMO 输出的掩模执行制造规则校验，作为 tapeout 前的硬性门禁。

主要功能：
1. 最小线宽检查 (Minimum Line Width)
2. 最小间距检查 (Minimum Spacing)
3. 最小 SRAF 尺寸检查 (Minimum SRAF Size)
4. 禁止锐角检查 (Acute Angle Prohibition)
5. 辅助特征与主特征最小距离检查 (SRAF-Main Feature Distance)
6. 违规区域高亮标注
7. 修复建议生成
8. Tapeout 门禁检查

使用方式：
    from mrc import MRCRules, MRCChecker, MRCGate
    from mrc.rules import load_default_rules

    # 加载规则
    rules = load_default_rules()

    # 执行检查
    checker = MRCChecker(rules)
    result = checker.check(mask_array, pixel_size=1.0)

    # Tapeout 门禁
    gate = MRCGate()
    passed = gate.validate(result)
"""

from .rules import (
    MRCRules,
    MRCRuleSeverity,
    MRCRuleType,
    MRCRuleConfig,
    load_default_rules,
    load_rules_from_yaml,
)
from .violations import (
    MRCViolation,
    MRCCheckResult,
    ViolationRegion,
)
from .checkers import MRCChecker
from .highlight import ViolationHighlighter
from .repair import RepairAdvisor, RepairSuggestion
from .gate import MRCGate, GateCheckResult

__all__ = [
    "MRCRules",
    "MRCRuleSeverity",
    "MRCRuleType",
    "MRCRuleConfig",
    "load_default_rules",
    "load_rules_from_yaml",
    "MRCViolation",
    "MRCCheckResult",
    "ViolationRegion",
    "MRCChecker",
    "ViolationHighlighter",
    "RepairAdvisor",
    "RepairSuggestion",
    "MRCGate",
    "GateCheckResult",
]
