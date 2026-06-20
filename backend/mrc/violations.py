# -*- coding: utf-8 -*-
"""
MRC 违规数据结构定义

定义违规区域、违规记录和检查结果的数据结构。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np

from .rules import MRCRuleType, MRCRuleSeverity

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """违规类型（与规则类型对应，用于更细粒度分类）"""
    LINE_WIDTH_TOO_SMALL = "line_width_too_small"
    SPACING_TOO_SMALL = "spacing_too_small"
    SRAF_TOO_SMALL = "sraf_too_small"
    ACUTE_ANGLE = "acute_angle"
    SRAF_TOO_CLOSE_TO_MAIN = "sraf_too_close_to_main"
    SRAF_TOO_FAR_FROM_MAIN = "sraf_too_far_from_main"


@dataclass
class ViolationRegion:
    """
    违规区域定义

    Attributes:
        bbox: 边界框 (ymin, xmin, ymax, xmax)，像素坐标
        centroid: 质心坐标 (y, x)，像素坐标
        area_pixels: 违规区域面积（像素数）
        mask_slice: 违规区域对应的掩模切片（可选）
        polygon_points: 违规轮廓多边形点列表（可选）
    """
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area_pixels: int
    mask_slice: Optional[np.ndarray] = None
    polygon_points: Optional[List[Tuple[int, int]]] = None

    @property
    def bbox_size(self) -> Tuple[int, int]:
        """边界框尺寸 (height, width)"""
        return (self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])

    def to_dict(self, include_masks: bool = False) -> Dict[str, Any]:
        result = {
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "area_pixels": self.area_pixels,
            "bbox_size": list(self.bbox_size),
        }
        if include_masks and self.mask_slice is not None:
            result["mask_slice_shape"] = list(self.mask_slice.shape)
        if self.polygon_points is not None:
            result["polygon_points"] = [list(p) for p in self.polygon_points]
        return result


@dataclass
class MRCViolation:
    """
    单条 MRC 违规记录

    Attributes:
        rule_type: 触发的规则类型
        violation_type: 具体违规类型
        severity: 严重级别
        message: 违规描述信息
        region: 违规区域
        measurement_nm: 测量值 (nm)
        threshold_nm: 规则阈值 (nm)
        pixel_size: 像素尺寸 (nm/pixel)
        extra_info: 额外信息
    """
    rule_type: MRCRuleType
    violation_type: ViolationType
    severity: MRCRuleSeverity
    message: str
    region: ViolationRegion
    measurement_nm: float = 0.0
    threshold_nm: float = 0.0
    pixel_size: float = 1.0
    extra_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fatal(self) -> bool:
        """是否为致命违规"""
        return self.severity == MRCRuleSeverity.FATAL

    @property
    def violation_area_nm2(self) -> float:
        """违规区域面积 (nm^2)"""
        return self.region.area_pixels * (self.pixel_size ** 2)

    def to_dict(self, include_masks: bool = False) -> Dict[str, Any]:
        return {
            "rule_type": self.rule_type.value,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "region": self.region.to_dict(include_masks=include_masks),
            "measurement_nm": self.measurement_nm,
            "threshold_nm": self.threshold_nm,
            "pixel_size": self.pixel_size,
            "violation_area_nm2": self.violation_area_nm2,
            "is_fatal": self.is_fatal,
            "extra_info": self.extra_info,
        }


@dataclass
class MRCCheckResult:
    """
    MRC 完整检查结果

    Attributes:
        violations: 所有违规记录列表
        mask_shape: 检查的掩模形状
        pixel_size: 像素尺寸 (nm/pixel)
        total_violations: 总违规数
        fatal_count: 致命违规数
        error_count: 错误违规数
        warning_count: 警告违规数
        info_count: 信息级违规数
        rules_checked: 已检查的规则列表
        timestamp: 检查时间戳
        check_duration_sec: 检查耗时 (秒)
    """
    violations: List[MRCViolation] = field(default_factory=list)
    mask_shape: Optional[Tuple[int, int]] = None
    pixel_size: float = 1.0
    total_violations: int = 0
    fatal_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    rules_checked: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    check_duration_sec: float = 0.0

    def __post_init__(self):
        self._count_violations()

    def _count_violations(self):
        """统计各类违规数量"""
        self.total_violations = len(self.violations)
        self.fatal_count = sum(1 for v in self.violations if v.severity == MRCRuleSeverity.FATAL)
        self.error_count = sum(1 for v in self.violations if v.severity == MRCRuleSeverity.ERROR)
        self.warning_count = sum(1 for v in self.violations if v.severity == MRCRuleSeverity.WARNING)
        self.info_count = sum(1 for v in self.violations if v.severity == MRCRuleSeverity.INFO)

    def add_violation(self, violation: MRCViolation) -> None:
        """添加违规记录"""
        self.violations.append(violation)
        self._count_violations()

    def add_violations(self, violations: List[MRCViolation]) -> None:
        """批量添加违规记录"""
        self.violations.extend(violations)
        self._count_violations()

    @property
    def passed(self) -> bool:
        """是否通过检查（无致命和错误违规）"""
        return self.fatal_count == 0 and self.error_count == 0

    @property
    def has_fatal(self) -> bool:
        """是否存在致命违规"""
        return self.fatal_count > 0

    def violations_by_rule(self) -> Dict[MRCRuleType, List[MRCViolation]]:
        """按规则类型分组违规"""
        result: Dict[MRCRuleType, List[MRCViolation]] = {}
        for v in self.violations:
            if v.rule_type not in result:
                result[v.rule_type] = []
            result[v.rule_type].append(v)
        return result

    def violations_by_severity(self) -> Dict[MRCRuleSeverity, List[MRCViolation]]:
        """按严重级别分组违规"""
        result: Dict[MRCRuleSeverity, List[MRCViolation]] = {}
        for v in self.violations:
            if v.severity not in result:
                result[v.severity] = []
            result[v.severity].append(v)
        return result

    def get_fatal_violations(self) -> List[MRCViolation]:
        """获取所有致命违规"""
        return [v for v in self.violations if v.is_fatal]

    def get_violations_by_type(self, rule_type: MRCRuleType) -> List[MRCViolation]:
        """获取指定规则类型的违规"""
        return [v for v in self.violations if v.rule_type == rule_type]

    def to_dict(self, include_masks: bool = False) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "has_fatal": self.has_fatal,
            "mask_shape": list(self.mask_shape) if self.mask_shape else None,
            "pixel_size": self.pixel_size,
            "total_violations": self.total_violations,
            "fatal_count": self.fatal_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "rules_checked": self.rules_checked,
            "timestamp": self.timestamp,
            "check_duration_sec": self.check_duration_sec,
            "violations": [v.to_dict(include_masks=include_masks) for v in self.violations],
            "violations_by_rule": {
                k.value: [vi.to_dict(include_masks=include_masks) for vi in v]
                for k, v in self.violations_by_rule().items()
            },
        }

    def summary(self) -> str:
        """生成简要摘要"""
        lines = [
            f"MRC 检查结果: {'通过' if self.passed else '未通过'}",
            f"  掩模尺寸: {self.mask_shape}",
            f"  像素尺寸: {self.pixel_size} nm",
            f"  总违规数: {self.total_violations}",
            f"    致命 (FATAL): {self.fatal_count}",
            f"    错误 (ERROR): {self.error_count}",
            f"    警告 (WARNING): {self.warning_count}",
            f"    信息 (INFO): {self.info_count}",
            f"  检查耗时: {self.check_duration_sec:.3f}s",
        ]
        if self.violations:
            lines.append("  违规详情:")
            for rule_type, vios in self.violations_by_rule().items():
                lines.append(f"    {rule_type.value}: {len(vios)} 处")
        return "\n".join(lines)

    def save_json(self, filepath: str, include_masks: bool = False) -> Path:
        """保存结果为 JSON 文件"""
        import json
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_masks=include_masks), f, indent=2, ensure_ascii=False)
        logger.info(f"MRC 结果已保存: {filepath}")
        return filepath
