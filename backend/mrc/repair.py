# -*- coding: utf-8 -*-
"""
修复建议生成模块

根据 MRC 检查结果，为各类违规生成自动化修复建议。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import numpy as np

from .violations import MRCViolation, MRCCheckResult, ViolationType
from .rules import MRCRuleType, MRCRuleSeverity

logger = logging.getLogger(__name__)


class RepairAction(Enum):
    """修复操作类型"""
    WIDEN_LINE = "widen_line"
    INCREASE_SPACING = "increase_spacing"
    REMOVE_SRAF = "remove_sraf"
    RESIZE_SRAF = "resize_sraf"
    MOVE_SRAF = "move_sraf"
    ROUND_CORNER = "round_corner"
    ADD_SERIF = "add_serif"
    NOTIFY_MANUAL = "notify_manual"


@dataclass
class RepairSuggestion:
    """
    单条修复建议

    Attributes:
        violation: 关联的违规记录
        action: 建议的修复操作
        priority: 优先级 (0-100, 越高越优先)
        description: 修复建议描述
        estimated_effect: 预估修复效果说明
        steps: 具体修复步骤
        auto_fixable: 是否可以自动修复
    """
    violation: MRCViolation
    action: RepairAction
    priority: int
    description: str
    estimated_effect: str = ""
    steps: List[str] = field(default_factory=list)
    auto_fixable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation.violation_type.value,
            "rule_type": self.violation.rule_type.value,
            "severity": self.violation.severity.value,
            "action": self.action.value,
            "priority": self.priority,
            "description": self.description,
            "estimated_effect": self.estimated_effect,
            "steps": self.steps,
            "auto_fixable": self.auto_fixable,
            "location": {
                "centroid": list(self.violation.region.centroid),
                "bbox": list(self.violation.region.bbox),
            },
        }


@dataclass
class RepairReport:
    """
    修复建议报告

    Attributes:
        suggestions: 所有修复建议列表
        total_suggestions: 建议总数
        auto_fixable_count: 可自动修复的数量
        manual_fix_needed: 需要人工修复的数量
    """
    suggestions: List[RepairSuggestion] = field(default_factory=list)
    total_suggestions: int = 0
    auto_fixable_count: int = 0
    manual_fix_needed: int = 0

    def __post_init__(self):
        self._update_counts()

    def _update_counts(self):
        self.total_suggestions = len(self.suggestions)
        self.auto_fixable_count = sum(1 for s in self.suggestions if s.auto_fixable)
        self.manual_fix_needed = sum(1 for s in self.suggestions if not s.auto_fixable)

    def add_suggestion(self, suggestion: RepairSuggestion) -> None:
        self.suggestions.append(suggestion)
        self._update_counts()

    def get_sorted_by_priority(self) -> List[RepairSuggestion]:
        """按优先级降序排列建议"""
        return sorted(self.suggestions, key=lambda s: -s.priority)

    def get_auto_fixable(self) -> List[RepairSuggestion]:
        """获取所有可自动修复的建议"""
        return [s for s in self.suggestions if s.auto_fixable]

    def get_by_action(self, action: RepairAction) -> List[RepairSuggestion]:
        """按修复操作类型筛选"""
        return [s for s in self.suggestions if s.action == action]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_suggestions": self.total_suggestions,
            "auto_fixable_count": self.auto_fixable_count,
            "manual_fix_needed": self.manual_fix_needed,
            "suggestions": [s.to_dict() for s in self.get_sorted_by_priority()],
        }

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            "MRC 修复建议报告",
            f"  总建议数: {self.total_suggestions}",
            f"  可自动修复: {self.auto_fixable_count}",
            f"  需人工修复: {self.manual_fix_needed}",
        ]
        if self.suggestions:
            lines.append("  按优先级排序的修复建议:")
            for s in self.get_sorted_by_priority()[:10]:
                lines.append(
                    f"    [P{s.priority}] {s.violation.violation_type.value}: "
                    f"{s.description[:60]}"
                )
            if len(self.suggestions) > 10:
                lines.append(f"    ... 还有 {len(self.suggestions) - 10} 条建议")
        return "\n".join(lines)


class RepairAdvisor:
    """
    修复建议生成器

    根据 MRC 违规记录生成针对性的修复建议。
    """

    def __init__(self):
        self._action_dispatch = {
            ViolationType.LINE_WIDTH_TOO_SMALL: self._advise_line_width,
            ViolationType.SPACING_TOO_SMALL: self._advise_spacing,
            ViolationType.SRAF_TOO_SMALL: self._advise_sraf_size,
            ViolationType.ACUTE_ANGLE: self._advise_acute_angle,
            ViolationType.SRAF_TOO_CLOSE_TO_MAIN: self._advise_sraf_too_close,
            ViolationType.SRAF_TOO_FAR_FROM_MAIN: self._advise_sraf_too_far,
        }

    def generate_report(self, result: MRCCheckResult) -> RepairReport:
        """
        生成完整的修复建议报告

        Args:
            result: MRC 检查结果

        Returns:
            RepairReport 修复建议报告
        """
        report = RepairReport()

        for violation in result.violations:
            suggestion = self._generate_single(violation)
            if suggestion is not None:
                report.add_suggestion(suggestion)

        return report

    def _generate_single(self, violation: MRCViolation) -> Optional[RepairSuggestion]:
        """为单条违规生成建议"""
        advisor_fn = self._action_dispatch.get(violation.violation_type)
        if advisor_fn is None:
            logger.warning(f"未实现建议生成器: {violation.violation_type.value}")
            return None
        return advisor_fn(violation)

    # ------------------------------------------------------------------
    # 具体违规类型的建议生成
    # ------------------------------------------------------------------

    @staticmethod
    def _advise_line_width(violation: MRCViolation) -> RepairSuggestion:
        """线宽过小的修复建议"""
        deficit = violation.threshold_nm - violation.measurement_nm
        priority = 90 if violation.is_fatal else 70

        steps = [
            f"确认该区域线宽测量值: {violation.measurement_nm:.1f}nm",
            f"目标最小线宽: {violation.threshold_nm:.1f}nm",
            f"需要增加线宽约: {deficit:.1f}nm",
            "使用 OPC 边缘偏置技术向外扩展该区域边缘",
            "检查扩展后是否影响相邻特征的间距规则",
        ]

        return RepairSuggestion(
            violation=violation,
            action=RepairAction.WIDEN_LINE,
            priority=priority,
            description=(
                f"线宽过小 ({violation.measurement_nm:.1f}nm < "
                f"{violation.threshold_nm:.1f}nm)，建议加宽 {deficit:.1f}nm"
            ),
            estimated_effect=(
                f"将该区域线宽从 {violation.measurement_nm:.1f}nm "
                f"增加到至少 {violation.threshold_nm:.1f}nm，可满足制造要求"
            ),
            steps=steps,
            auto_fixable=True,
        )

    @staticmethod
    def _advise_spacing(violation: MRCViolation) -> RepairSuggestion:
        """间距过小的修复建议"""
        deficit = violation.threshold_nm - violation.measurement_nm
        priority = 95 if violation.is_fatal else 75

        steps = [
            f"确认该区域间距测量值: {violation.measurement_nm:.1f}nm",
            f"目标最小间距: {violation.threshold_nm:.1f}nm",
            f"需要增加间距约: {deficit:.1f}nm",
            "分析两侧特征，选择可收缩的一侧进行边缘内移",
            "优先调整非关键路径上的特征",
            "检查调整后特征自身的线宽规则是否仍满足",
        ]

        return RepairSuggestion(
            violation=violation,
            action=RepairAction.INCREASE_SPACING,
            priority=priority,
            description=(
                f"间距过小 ({violation.measurement_nm:.1f}nm < "
                f"{violation.threshold_nm:.1f}nm)，建议增大约 {deficit:.1f}nm"
            ),
            estimated_effect=(
                f"将相邻特征间距从 {violation.measurement_nm:.1f}nm "
                f"增加到至少 {violation.threshold_nm:.1f}nm"
            ),
            steps=steps,
            auto_fixable=False,
        )

    @staticmethod
    def _advise_sraf_size(violation: MRCViolation) -> RepairSuggestion:
        """SRAF 尺寸过小的修复建议"""
        area_nm2 = violation.extra_info.get("area_nm2", 0)
        min_dim = violation.measurement_nm
        threshold = violation.threshold_nm

        priority = 60
        steps = [
            f"SRAF 当前最小尺寸: {min_dim:.1f}nm, 面积: {area_nm2:.0f}nm²",
            f"规则要求最小尺寸: {threshold:.1f}nm",
            "方案一: 增大该 SRAF 尺寸到规则允许的最小值",
            "方案二: 移除该 SRAF（如果对工艺窗口影响可接受）",
            "评估增大/移除对周围主特征成像质量的影响",
        ]

        auto_fixable = area_nm2 < threshold * threshold * 0.5

        return RepairSuggestion(
            violation=violation,
            action=RepairAction.REMOVE_SRAF if auto_fixable else RepairAction.RESIZE_SRAF,
            priority=priority,
            description=(
                f"SRAF 尺寸过小 ({min_dim:.1f}nm < {threshold:.1f}nm)，"
                f"建议增大或移除"
            ),
            estimated_effect="消除 SRAF 尺寸违规，可能轻微影响局部工艺窗口",
            steps=steps,
            auto_fixable=auto_fixable,
        )

    @staticmethod
    def _advise_acute_angle(violation: MRCViolation) -> RepairSuggestion:
        """锐角的修复建议"""
        estimated_angle = violation.extra_info.get("estimated_angle_deg", 45.0)
        min_angle = violation.threshold_nm

        priority = 70
        steps = [
            f"估计拐角角度: {estimated_angle:.1f}°",
            f"规则要求最小角度: {min_angle:.1f}°",
            "方案一: 添加倒角 (round corner) 将锐角改为钝角",
            "方案二: 添加衬线 (serif) 改善该角点的成像",
            "检查修改后是否引入其他规则违规",
        ]

        return RepairSuggestion(
            violation=violation,
            action=RepairAction.ROUND_CORNER,
            priority=priority,
            description=(
                f"检测到锐角 (约 {estimated_angle:.1f}° < {min_angle:.1f}°)，"
                f"建议添加倒角或衬线"
            ),
            estimated_effect="消除锐角违规，改善角点处的光刻成像质量",
            steps=steps,
            auto_fixable=True,
        )

    @staticmethod
    def _advise_sraf_too_close(violation: MRCViolation) -> RepairSuggestion:
        """SRAF 距主特征过近的修复建议"""
        actual = violation.measurement_nm
        threshold = violation.threshold_nm
        deficit = threshold - actual

        priority = 65
        steps = [
            f"SRAF 到主特征的最小距离: {actual:.1f}nm",
            f"规则要求最小距离: {threshold:.1f}nm",
            f"需要将 SRAF 外移至少 {deficit:.1f}nm",
            "外移 SRAF 后检查与其他 SRAF 的间距",
            "评估移动后对主特征成像辅助效果的影响",
        ]

        return RepairSuggestion(
            violation=violation,
            action=RepairAction.MOVE_SRAF,
            priority=priority,
            description=(
                f"SRAF 距主特征过近 ({actual:.1f}nm < {threshold:.1f}nm)，"
                f"建议外移约 {deficit:.1f}nm"
            ),
            estimated_effect="消除 SRAF 位置违规，避免 SRAF 与主特征粘连",
            steps=steps,
            auto_fixable=True,
        )

    @staticmethod
    def _advise_sraf_too_far(violation: MRCViolation) -> RepairSuggestion:
        """SRAF 距主特征过远的修复建议"""
        actual = violation.measurement_nm
        threshold = violation.threshold_nm

        priority = 30
        steps = [
            f"SRAF 到主特征的最大距离: {actual:.1f}nm",
            f"建议最大距离: {threshold:.1f}nm",
            "评估该 SRAF 是否仍能有效辅助主特征成像",
            "方案一: 向内移动 SRAF 靠近主特征",
            "方案二: 移除该 SRAF（效率太低时）",
        ]

        return RepairSuggestion(
            violation=violation,
            action=RepairAction.MOVE_SRAF,
            priority=priority,
            description=(
                f"SRAF 距主特征过远 ({actual:.1f}nm > {threshold:.1f}nm)，"
                f"辅助效率可能不足"
            ),
            estimated_effect="优化 SRAF 位置，提升辅助成像效率",
            steps=steps,
            auto_fixable=False,
        )
