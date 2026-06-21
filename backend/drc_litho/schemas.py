# -*- coding: utf-8 -*-
"""
版图设计规则邻近分析模块 - 数据结构定义

定义光刻导向 DRC 分析所需的违规类型、区域、违规记录和检查结果。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LithoViolationCategory(Enum):
    BRIDGE = "bridge"
    BREAK = "break"
    ISOLATED = "isolated"
    PROXIMITY = "proximity"


class LithoViolationType(Enum):
    BRIDGE_RISK_NARROW_GAP = "bridge_risk_narrow_gap"
    BRIDGE_RISK_NECKING = "bridge_risk_necking"
    BRIDGE_RISK_DENSE_CORNER = "bridge_risk_dense_corner"
    BREAK_RISK_THIN_NECK = "break_risk_thin_neck"
    BREAK_RISK_SHARP_TURN = "break_risk_sharp_turn"
    BREAK_RISK_LINE_END = "break_risk_line_end"
    ISOLATED_SMALL_FEATURE = "isolated_small_feature"
    ISOLATED_DANGLING_LINE = "isolated_dangling_line"
    ISOLATED_ORPHAN_PIXEL = "isolated_orphan_pixel"


class LithoSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class OPCFeasibility(Enum):
    FIXABLE = "fixable"
    PARTIAL = "partial"
    UNFIXABLE = "unfixable"
    NEEDS_REDESIGN = "needs_redesign"


@dataclass
class ViolationRegion:
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area_pixels: int
    mask_slice: Optional[np.ndarray] = None
    polygon_points: Optional[List[Tuple[int, int]]] = None

    @property
    def bbox_size(self) -> Tuple[int, int]:
        return (self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])

    def to_dict(self, include_masks: bool = False) -> Dict[str, Any]:
        result = {
            "bbox": [int(x) for x in self.bbox],
            "centroid": [float(x) for x in self.centroid],
            "area_pixels": int(self.area_pixels),
            "bbox_size": [int(x) for x in self.bbox_size],
        }
        if include_masks and self.mask_slice is not None:
            result["mask_slice_shape"] = list(self.mask_slice.shape)
        if self.polygon_points is not None:
            result["polygon_points"] = [list(p) for p in self.polygon_points]
        return result


@dataclass
class LithoViolation:
    category: LithoViolationCategory
    violation_type: LithoViolationType
    severity: LithoSeverity
    message: str
    region: ViolationRegion
    measurement_nm: float = 0.0
    threshold_nm: float = 0.0
    pixel_size: float = 1.0
    opc_feasibility: OPCFeasibility = OPCFeasibility.PARTIAL
    extra_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fatal(self) -> bool:
        return self.severity == LithoSeverity.FATAL

    @property
    def violation_area_nm2(self) -> float:
        return self.region.area_pixels * (self.pixel_size ** 2)

    def to_dict(self, include_masks: bool = False) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "region": self.region.to_dict(include_masks=include_masks),
            "measurement_nm": self.measurement_nm,
            "threshold_nm": self.threshold_nm,
            "pixel_size": self.pixel_size,
            "opc_feasibility": self.opc_feasibility.value,
            "violation_area_nm2": self.violation_area_nm2,
            "is_fatal": self.is_fatal,
            "extra_info": self.extra_info,
        }


@dataclass
class LithoDRCResult:
    violations: List[LithoViolation] = field(default_factory=list)
    mask_shape: Optional[Tuple[int, int]] = None
    pixel_size: float = 1.0
    total_violations: int = 0
    bridge_count: int = 0
    break_count: int = 0
    isolated_count: int = 0
    proximity_count: int = 0
    fatal_count: int = 0
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    rules_checked: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    check_duration_sec: float = 0.0
    opc_readiness_score: float = 0.0

    def __post_init__(self):
        self._count_violations()

    def _count_violations(self):
        self.total_violations = len(self.violations)
        self.bridge_count = sum(
            1 for v in self.violations if v.category == LithoViolationCategory.BRIDGE
        )
        self.break_count = sum(
            1 for v in self.violations if v.category == LithoViolationCategory.BREAK
        )
        self.isolated_count = sum(
            1 for v in self.violations if v.category == LithoViolationCategory.ISOLATED
        )
        self.proximity_count = sum(
            1 for v in self.violations if v.category == LithoViolationCategory.PROXIMITY
        )
        self.fatal_count = sum(
            1 for v in self.violations if v.severity == LithoSeverity.FATAL
        )
        self.critical_count = sum(
            1 for v in self.violations if v.severity == LithoSeverity.CRITICAL
        )
        self.warning_count = sum(
            1 for v in self.violations if v.severity == LithoSeverity.WARNING
        )
        self.info_count = sum(
            1 for v in self.violations if v.severity == LithoSeverity.INFO
        )
        self._compute_opc_readiness()

    def _compute_opc_readiness(self):
        if self.total_violations == 0:
            self.opc_readiness_score = 100.0
            return
        penalty = (
            self.fatal_count * 40.0
            + self.critical_count * 20.0
            + self.warning_count * 5.0
            + self.info_count * 1.0
        )
        self.opc_readiness_score = max(0.0, 100.0 - penalty)

    def add_violation(self, violation: LithoViolation) -> None:
        self.violations.append(violation)
        self._count_violations()

    def add_violations(self, violations: List[LithoViolation]) -> None:
        self.violations.extend(violations)
        self._count_violations()

    @property
    def passed(self) -> bool:
        return self.fatal_count == 0 and self.critical_count == 0

    @property
    def opc_ready(self) -> bool:
        return self.opc_readiness_score >= 60.0

    def violations_by_category(self) -> Dict[LithoViolationCategory, List[LithoViolation]]:
        result: Dict[LithoViolationCategory, List[LithoViolation]] = {}
        for v in self.violations:
            if v.category not in result:
                result[v.category] = []
            result[v.category].append(v)
        return result

    def violations_by_severity(self) -> Dict[LithoSeverity, List[LithoViolation]]:
        result: Dict[LithoSeverity, List[LithoViolation]] = {}
        for v in self.violations:
            if v.severity not in result:
                result[v.severity] = []
            result[v.severity].append(v)
        return result

    def to_dict(self, include_masks: bool = False) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "opc_ready": self.opc_ready,
            "opc_readiness_score": self.opc_readiness_score,
            "mask_shape": [int(x) for x in self.mask_shape] if self.mask_shape else None,
            "pixel_size": self.pixel_size,
            "total_violations": self.total_violations,
            "bridge_count": self.bridge_count,
            "break_count": self.break_count,
            "isolated_count": self.isolated_count,
            "proximity_count": self.proximity_count,
            "fatal_count": self.fatal_count,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "rules_checked": self.rules_checked,
            "timestamp": self.timestamp,
            "check_duration_sec": self.check_duration_sec,
            "violations": [
                v.to_dict(include_masks=include_masks) for v in self.violations
            ],
        }

    def summary(self) -> str:
        lines = [
            f"光刻导向 DRC 检查结果: {'通过' if self.passed else '未通过'}",
            f"  OPC 就绪度: {self.opc_readiness_score:.1f}/100 "
            f"({'就绪' if self.opc_ready else '未就绪'})",
            f"  掩模尺寸: {self.mask_shape}",
            f"  像素尺寸: {self.pixel_size} nm",
            f"  总违规数: {self.total_violations}",
            f"    桥连风险 (Bridge): {self.bridge_count}",
            f"    断线风险 (Break):   {self.break_count}",
            f"    孤立线 (Isolated):  {self.isolated_count}",
            f"    邻近效应 (Proximity): {self.proximity_count}",
            f"  严重级别分布:",
            f"    致命 (FATAL):    {self.fatal_count}",
            f"    严重 (CRITICAL): {self.critical_count}",
            f"    警告 (WARNING):  {self.warning_count}",
            f"    信息 (INFO):     {self.info_count}",
            f"  检查耗时: {self.check_duration_sec:.3f}s",
        ]
        if self.violations:
            lines.append("  违规类别详情:")
            for cat, vios in self.violations_by_category().items():
                lines.append(f"    {cat.value}: {len(vios)} 处")
        return "\n".join(lines)

    def save_json(self, filepath: str, include_masks: bool = False) -> Path:
        import json
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict(include_masks=include_masks)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
        logger.info(f"DRC-Litho 结果已保存: {filepath}")
        return filepath


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
