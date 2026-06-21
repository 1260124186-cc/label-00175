# -*- coding: utf-8 -*-
"""
版图设计规则邻近分析模块 - OPC 预警与修改建议

根据 DRC-Litho 检查结果，为各类违规生成 OPC 感知的修改建议:
1. 评估 OPC 对违规的可修复程度
2. 给出具体的版图修改方案
3. 预估修改效果与对后续 OPC 的影响
4. 生成优先级排序的修复报告
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .schemas import (
    LithoViolation,
    LithoDRCResult,
    LithoViolationCategory,
    LithoViolationType,
    LithoSeverity,
    OPCFeasibility,
)

logger = logging.getLogger(__name__)


class OPCAdviceAction(Enum):
    INCREASE_SPACING = "increase_spacing"
    WIDEN_LINE = "widen_line"
    ADD_HAMMERHEAD = "add_hammerhead"
    ADD_SERIF = "add_serif"
    ROUND_CORNER = "round_corner"
    REMOVE_FEATURE = "remove_feature"
    RESIZE_FEATURE = "resize_feature"
    REDISTRIBUTE_CORNERS = "redistribute_corners"
    REDESIGN_LAYOUT = "redesign_layout"
    RUN_OPC_WITH_CAUTION = "run_opc_with_caution"


@dataclass
class OPCAdvice:
    violation: LithoViolation
    action: OPCAdviceAction
    priority: int
    description: str
    opc_impact: str = ""
    steps: List[str] = field(default_factory=list)
    auto_fixable: bool = False
    estimated_opc_iterations_saved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation.violation_type.value,
            "category": self.violation.category.value,
            "severity": self.violation.severity.value,
            "action": self.action.value,
            "priority": self.priority,
            "description": self.description,
            "opc_impact": self.opc_impact,
            "steps": self.steps,
            "auto_fixable": self.auto_fixable,
            "estimated_opc_iterations_saved": self.estimated_opc_iterations_saved,
            "location": {
                "centroid": list(self.violation.region.centroid),
                "bbox": list(self.violation.region.bbox),
            },
            "opc_feasibility": self.violation.opc_feasibility.value,
        }


@dataclass
class OPCAdviceReport:
    advices: List[OPCAdvice] = field(default_factory=list)
    total_advices: int = 0
    auto_fixable_count: int = 0
    manual_fix_needed: int = 0
    redesign_needed: int = 0
    estimated_opc_iterations_saved: int = 0
    opc_readiness_score: float = 0.0

    def __post_init__(self):
        self._update_counts()

    def _update_counts(self):
        self.total_advices = len(self.advices)
        self.auto_fixable_count = sum(1 for a in self.advices if a.auto_fixable)
        self.manual_fix_needed = sum(
            1 for a in self.advices if not a.auto_fixable
            and a.violation.opc_feasibility != OPCFeasibility.NEEDS_REDESIGN
        )
        self.redesign_needed = sum(
            1 for a in self.advices
            if a.violation.opc_feasibility == OPCFeasibility.NEEDS_REDESIGN
        )
        self.estimated_opc_iterations_saved = sum(
            a.estimated_opc_iterations_saved for a in self.advices
        )
        if self.total_advices > 0:
            self.opc_readiness_score = max(
                0.0,
                100.0
                - self.redesign_needed * 40.0
                - self.manual_fix_needed * 15.0
                - (self.total_advices - self.redesign_needed - self.manual_fix_needed) * 3.0
            )
        else:
            self.opc_readiness_score = 100.0

    def add_advice(self, advice: OPCAdvice) -> None:
        self.advices.append(advice)
        self._update_counts()

    def get_sorted_by_priority(self) -> List[OPCAdvice]:
        return sorted(self.advices, key=lambda a: -a.priority)

    def get_auto_fixable(self) -> List[OPCAdvice]:
        return [a for a in self.advices if a.auto_fixable]

    def get_by_category(self, category: LithoViolationCategory) -> List[OPCAdvice]:
        return [a for a in self.advices if a.violation.category == category]

    def get_needs_redesign(self) -> List[OPCAdvice]:
        return [
            a for a in self.advices
            if a.violation.opc_feasibility == OPCFeasibility.NEEDS_REDESIGN
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_advices": self.total_advices,
            "auto_fixable_count": self.auto_fixable_count,
            "manual_fix_needed": self.manual_fix_needed,
            "redesign_needed": self.redesign_needed,
            "estimated_opc_iterations_saved": self.estimated_opc_iterations_saved,
            "opc_readiness_score": self.opc_readiness_score,
            "advices": [a.to_dict() for a in self.get_sorted_by_priority()],
        }

    def summary(self) -> str:
        lines = [
            "OPC 预警修改建议报告",
            f"  OPC 就绪度评分: {self.opc_readiness_score:.1f}/100",
            f"  总建议数: {self.total_advices}",
            f"  可自动修复: {self.auto_fixable_count}",
            f"  需人工修复: {self.manual_fix_needed}",
            f"  需版图重设计: {self.redesign_needed}",
            f"  预计节省 OPC 迭代次数: {self.estimated_opc_iterations_saved}",
        ]
        if self.advices:
            lines.append("  按优先级排序的修改建议 (Top 10):")
            for a in self.get_sorted_by_priority()[:10]:
                lines.append(
                    f"    [P{a.priority:02d}] {a.violation.category.value}/{a.violation.violation_type.value}: "
                    f"{a.description[:70]}"
                )
            if len(self.advices) > 10:
                lines.append(f"    ... 还有 {len(self.advices) - 10} 条建议")
        return "\n".join(lines)


class OPCAdvisor:
    """
    OPC 预警修改建议生成器

    根据光刻导向 DRC 违规记录，评估 OPC 可修复程度并给出修改建议。
    核心原则: OPC 只能在有限范围内调整边缘位置，无法弥补根本性的
    版图拓扑缺陷。因此:
    - 桥连风险: OPC 可通过收窄边缘缓解，但间距严重不足需版图重设计
    - 断线风险: OPC 可通过锤头/衬线补偿，但线宽过低仍需加宽
    - 孤立线: OPC 通常无法帮助，需要删除或放大特征
    """

    def __init__(self):
        self._dispatch = {
            LithoViolationType.BRIDGE_RISK_NARROW_GAP: self._advise_bridge_narrow_gap,
            LithoViolationType.BRIDGE_RISK_NECKING: self._advise_bridge_necking,
            LithoViolationType.BRIDGE_RISK_DENSE_CORNER: self._advise_bridge_dense_corner,
            LithoViolationType.BREAK_RISK_THIN_NECK: self._advise_break_thin_neck,
            LithoViolationType.BREAK_RISK_SHARP_TURN: self._advise_break_sharp_turn,
            LithoViolationType.BREAK_RISK_LINE_END: self._advise_break_line_end,
            LithoViolationType.ISOLATED_SMALL_FEATURE: self._advise_isolated_small_feature,
            LithoViolationType.ISOLATED_DANGLING_LINE: self._advise_isolated_dangling_line,
            LithoViolationType.ISOLATED_ORPHAN_PIXEL: self._advise_isolated_orphan_pixel,
        }

    def generate_report(self, result: LithoDRCResult) -> OPCAdviceReport:
        report = OPCAdviceReport()

        for violation in result.violations:
            advice = self._generate_single(violation)
            if advice is not None:
                report.add_advice(advice)

        return report

    def _generate_single(self, violation: LithoViolation) -> Optional[OPCAdvice]:
        advisor_fn = self._dispatch.get(violation.violation_type)
        if advisor_fn is None:
            logger.warning(f"未实现建议生成器: {violation.violation_type.value}")
            return None
        return advisor_fn(violation)

    @staticmethod
    def _advise_bridge_narrow_gap(violation: LithoViolation) -> OPCAdvice:
        gap_nm = violation.measurement_nm
        threshold_nm = violation.threshold_nm
        deficit = threshold_nm - gap_nm

        if deficit > threshold_nm * 0.5:
            priority = 95
            action = OPCAdviceAction.REDESIGN_LAYOUT
            auto_fixable = False
            iters_saved = 5
        elif deficit > threshold_nm * 0.2:
            priority = 85
            action = OPCAdviceAction.INCREASE_SPACING
            auto_fixable = False
            iters_saved = 3
        else:
            priority = 70
            action = OPCAdviceAction.RUN_OPC_WITH_CAUTION
            auto_fixable = True
            iters_saved = 2

        steps = [
            f"当前间距: {gap_nm:.1f}nm, 需要至少: {threshold_nm:.1f}nm",
            f"间距不足: {deficit:.1f}nm",
            "分析两侧特征，确定哪侧可内缩",
            "若可内缩: 收缩一侧特征边缘以增大间距",
            "若间距严重不足: 重新设计局部版图布局",
            "OPC 阶段可对边缘做微调，但无法弥补大幅间距缺陷",
        ]

        return OPCAdvice(
            violation=violation,
            action=action,
            priority=priority,
            description=(
                f"桥连风险-间距过窄 ({gap_nm:.1f}nm < {threshold_nm:.1f}nm)，"
                f"需增大约 {deficit:.1f}nm"
            ),
            opc_impact=(
                f"OPC 可通过边缘收窄缓解约 {threshold_nm * 0.2:.1f}nm，"
                f"剩余 {max(0, deficit - threshold_nm * 0.2):.1f}nm 需版图修改"
            ),
            steps=steps,
            auto_fixable=auto_fixable,
            estimated_opc_iterations_saved=iters_saved,
        )

    @staticmethod
    def _advise_bridge_necking(violation: LithoViolation) -> OPCAdvice:
        neck_nm = violation.measurement_nm
        threshold_nm = violation.threshold_nm
        neck_ratio = violation.extra_info.get("neck_ratio", 1.0)

        if neck_ratio < 0.3:
            priority = 90
            action = OPCAdviceAction.WIDEN_LINE
            auto_fixable = False
            iters_saved = 4
        else:
            priority = 75
            action = OPCAdviceAction.RUN_OPC_WITH_CAUTION
            auto_fixable = True
            iters_saved = 2

        steps = [
            f"颈部线宽: {neck_nm:.1f}nm, 阈值: {threshold_nm:.1f}nm",
            f"颈部占全局线宽比: {neck_ratio:.1%}",
            "若颈部比 > 50%: OPC 可通过边缘偏置补偿",
            "若颈部比 < 30%: 需加宽该区域线条",
            "检查加宽后是否影响相邻特征间距",
        ]

        return OPCAdvice(
            violation=violation,
            action=action,
            priority=priority,
            description=(
                f"桥连风险-颈部收窄 ({neck_nm:.1f}nm < {threshold_nm:.1f}nm)，"
                f"颈部比 {neck_ratio:.1%}"
            ),
            opc_impact=(
                f"颈部比 {neck_ratio:.1%}，"
                f"{'OPC 可部分补偿' if neck_ratio >= 0.5 else '需先加宽后再做 OPC'}"
            ),
            steps=steps,
            auto_fixable=auto_fixable,
            estimated_opc_iterations_saved=iters_saved,
        )

    @staticmethod
    def _advise_bridge_dense_corner(violation: LithoViolation) -> OPCAdvice:
        density = violation.measurement_nm
        threshold = violation.threshold_nm

        priority = 60
        steps = [
            f"拐角密度: {density:.2f}, 阈值: {threshold:.2f}",
            "方案一: 增大拐角间距，分散拐角布局",
            "方案二: 合并相近拐角为平滑曲线",
            "方案三: 对高密度拐角区域优先分配 OPC 迭代资源",
            "OPC 处理: 多拐角区域的 OPC 收敛较慢，需增加迭代次数",
        ]

        return OPCAdvice(
            violation=violation,
            action=OPCAdviceAction.REDISTRIBUTE_CORNERS,
            priority=priority,
            description=(
                f"桥连风险-拐角密度过高 ({density:.2f} >= {threshold:.2f})，"
                f"建议优化拐角布局"
            ),
            opc_impact="高密度拐角区域 OPC 收敛慢，可能需要 2-3 次额外迭代",
            steps=steps,
            auto_fixable=False,
            estimated_opc_iterations_saved=2,
        )

    @staticmethod
    def _advise_break_thin_neck(violation: LithoViolation) -> OPCAdvice:
        neck_nm = violation.measurement_nm
        threshold_nm = violation.threshold_nm
        neck_length_nm = violation.extra_info.get("neck_length_nm", 0.0)

        if neck_nm < threshold_nm * 0.5:
            priority = 98
            action = OPCAdviceAction.REDESIGN_LAYOUT
            auto_fixable = False
            iters_saved = 5
        elif neck_nm < threshold_nm * 0.8:
            priority = 90
            action = OPCAdviceAction.WIDEN_LINE
            auto_fixable = False
            iters_saved = 3
        else:
            priority = 80
            action = OPCAdviceAction.ADD_HAMMERHEAD
            auto_fixable = True
            iters_saved = 2

        steps = [
            f"颈部线宽: {neck_nm:.1f}nm, 最低要求: {threshold_nm:.1f}nm",
            f"颈部长度: {neck_length_nm:.1f}nm",
            "若线宽严重不足: 重新设计局部版图，加宽线条",
            "若线宽接近阈值: OPC 可通过锤头/衬线补偿",
            "OPC 阶段应优先处理此区域，防止过度腐蚀导致断线",
        ]

        return OPCAdvice(
            violation=violation,
            action=action,
            priority=priority,
            description=(
                f"断线风险-线宽过窄 ({neck_nm:.1f}nm < {threshold_nm:.1f}nm)，"
                f"长度 {neck_length_nm:.1f}nm"
            ),
            opc_impact=(
                f"线宽仅为阈值的 {neck_nm / threshold_nm:.0%}，"
                f"{'OPC 无法修复，需重设计' if neck_nm < threshold_nm * 0.5 else 'OPC 可通过锤头补偿'}"
            ),
            steps=steps,
            auto_fixable=auto_fixable,
            estimated_opc_iterations_saved=iters_saved,
        )

    @staticmethod
    def _advise_break_sharp_turn(violation: LithoViolation) -> OPCAdvice:
        angle = violation.measurement_nm
        threshold = violation.threshold_nm

        if angle < threshold * 0.5:
            priority = 80
            action = OPCAdviceAction.ROUND_CORNER
            iters_saved = 3
        else:
            priority = 65
            action = OPCAdviceAction.ADD_SERIF
            iters_saved = 2

        steps = [
            f"估计角度: {angle:.1f}°, 最小允许: {threshold:.1f}°",
            "方案一: 添加倒角，将锐角改为钝角/圆弧",
            "方案二: 添加衬线 (serif) 改善内侧成像",
            "OPC 处理: 急转弯处 OPC 补偿效果有限，建议版图层面优化",
        ]

        return OPCAdvice(
            violation=violation,
            action=action,
            priority=priority,
            description=(
                f"断线风险-急转弯 ({angle:.1f}° < {threshold:.1f}°)，"
                f"建议{'添加倒角' if angle < threshold * 0.5 else '添加衬线'}"
            ),
            opc_impact="急转弯内侧 OPC 难以补足，倒角后 OPC 效果更佳",
            steps=steps,
            auto_fixable=True,
            estimated_opc_iterations_saved=iters_saved,
        )

    @staticmethod
    def _advise_break_line_end(violation: LithoViolation) -> OPCAdvice:
        width_nm = violation.measurement_nm
        threshold_nm = violation.threshold_nm

        priority = 70
        steps = [
            f"线端线宽: {width_nm:.1f}nm, 阈值: {threshold_nm:.1f}nm",
            "方案一: 添加锤头 (hammerhead) 增大线端面积",
            "方案二: 适当加宽线端附近的线段",
            "OPC 处理: 锤头是线端缩短的标准补偿手段，OPC 兼容性好",
        ]

        return OPCAdvice(
            violation=violation,
            action=OPCAdviceAction.ADD_HAMMERHEAD,
            priority=priority,
            description=(
                f"断线风险-线端 ({width_nm:.1f}nm < {threshold_nm:.1f}nm)，"
                f"建议添加锤头补偿"
            ),
            opc_impact="锤头补偿与 OPC 高度兼容，OPC 可进一步微调锤头尺寸",
            steps=steps,
            auto_fixable=True,
            estimated_opc_iterations_saved=2,
        )

    @staticmethod
    def _advise_isolated_small_feature(violation: LithoViolation) -> OPCAdvice:
        area_nm2 = violation.extra_info.get("area_nm2", 0)
        min_axis_nm = violation.measurement_nm
        threshold_nm = violation.threshold_nm

        priority = 55
        steps = [
            f"特征面积: {area_nm2:.0f}nm², 最小轴: {min_axis_nm:.1f}nm",
            f"规则要求最小轴: {threshold_nm:.1f}nm",
            "方案一: 放大该特征到规则允许的最小尺寸",
            "方案二: 删除该特征（如果非功能性）",
            "OPC 处理: 小特征 OPC 补偿效果差，建议从版图层面解决",
        ]

        return OPCAdvice(
            violation=violation,
            action=OPCAdviceAction.RESIZE_FEATURE,
            priority=priority,
            description=(
                f"孤立线风险-小特征 (面积 {area_nm2:.0f}nm², "
                f"最小轴 {min_axis_nm:.1f}nm < {threshold_nm:.1f}nm)"
            ),
            opc_impact="小特征 OPC 成像困难，建议版图修改后再进入 OPC 流程",
            steps=steps,
            auto_fixable=False,
            estimated_opc_iterations_saved=2,
        )

    @staticmethod
    def _advise_isolated_dangling_line(violation: LithoViolation) -> OPCAdvice:
        length_nm = violation.measurement_nm
        width_nm = violation.extra_info.get("mean_width_nm", 0)

        priority = 50
        steps = [
            f"悬空线长度: {length_nm:.1f}nm, 平均线宽: {width_nm:.1f}nm",
            "方案一: 删除该悬空线段（如果非功能性）",
            "方案二: 将悬空线连接到最近的主特征",
            "方案三: 如果是 SRAF 辅助特征，检查其位置是否合理",
            "OPC 处理: 悬空线 OPC 收益低，建议版图层面清理",
        ]

        return OPCAdvice(
            violation=violation,
            action=OPCAdviceAction.REMOVE_FEATURE,
            priority=priority,
            description=(
                f"孤立线风险-悬空线 (长度 {length_nm:.1f}nm, "
                f"线宽 {width_nm:.1f}nm)"
            ),
            opc_impact="悬空线对 OPC 几乎无正面贡献，建议清理",
            steps=steps,
            auto_fixable=True,
            estimated_opc_iterations_saved=1,
        )

    @staticmethod
    def _advise_isolated_orphan_pixel(violation: LithoViolation) -> OPCAdvice:
        area_px = violation.extra_info.get("area_pixels", 0)
        area_nm2 = violation.extra_info.get("area_nm2", 0)

        priority = 30
        steps = [
            f"孤儿像素面积: {area_px}px ({area_nm2:.1f}nm²)",
            "方案: 直接删除该像素团，很可能是版图噪声",
            "OPC 处理: 孤儿像素 OPC 不会处理，应在版图层面清理",
        ]

        return OPCAdvice(
            violation=violation,
            action=OPCAdviceAction.REMOVE_FEATURE,
            priority=priority,
            description=f"孤立线风险-孤儿像素 ({area_px}px, {area_nm2:.1f}nm²)",
            opc_impact="孤儿像素对 OPC 无影响，建议清理",
            steps=steps,
            auto_fixable=True,
            estimated_opc_iterations_saved=1,
        )
