# -*- coding: utf-8 -*-
"""
MRC 规则定义模块

定义掩模制造规则的类型、严重级别和配置结构。
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class MRCRuleSeverity(Enum):
    """规则严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class MRCRuleType(Enum):
    """规则类型枚举"""
    MIN_LINE_WIDTH = "min_line_width"
    MIN_SPACING = "min_spacing"
    MIN_SRAF_SIZE = "min_sraf_size"
    NO_ACUTE_ANGLE = "no_acute_angle"
    SRAF_MAIN_DISTANCE = "sraf_main_distance"
    MIN_ENCLOSURE = "min_enclosure"
    MAX_DENSITY = "max_density"
    MIN_DENSITY = "min_density"


@dataclass
class MRCRuleConfig:
    """
    单条规则配置

    Attributes:
        rule_type: 规则类型
        enabled: 是否启用该规则
        threshold_nm: 规则阈值 (nm)
        severity: 违规严重级别
        description: 规则描述
        extra_params: 额外参数
    """
    rule_type: MRCRuleType
    enabled: bool = True
    threshold_nm: float = 0.0
    severity: MRCRuleSeverity = MRCRuleSeverity.ERROR
    description: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_type": self.rule_type.value,
            "enabled": self.enabled,
            "threshold_nm": self.threshold_nm,
            "severity": self.severity.value,
            "description": self.description,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MRCRuleConfig":
        return cls(
            rule_type=MRCRuleType(data["rule_type"]),
            enabled=data.get("enabled", True),
            threshold_nm=float(data.get("threshold_nm", 0.0)),
            severity=MRCRuleSeverity(data.get("severity", "error")),
            description=data.get("description", ""),
            extra_params=data.get("extra_params", {}),
        )


@dataclass
class MRCRules:
    """
    MRC 规则集合

    包含所有制造规则的配置，支持按技术节点预设。
    """
    rules: Dict[MRCRuleType, MRCRuleConfig] = field(default_factory=dict)
    technology_node: str = "custom"
    pixel_size: float = 1.0

    def __post_init__(self):
        if not self.rules:
            self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认规则"""
        self.rules = {
            MRCRuleType.MIN_LINE_WIDTH: MRCRuleConfig(
                rule_type=MRCRuleType.MIN_LINE_WIDTH,
                enabled=True,
                threshold_nm=45.0,
                severity=MRCRuleSeverity.FATAL,
                description="最小线宽规则：掩模上所有主特征线宽不得小于此值",
            ),
            MRCRuleType.MIN_SPACING: MRCRuleConfig(
                rule_type=MRCRuleType.MIN_SPACING,
                enabled=True,
                threshold_nm=45.0,
                severity=MRCRuleSeverity.FATAL,
                description="最小间距规则：相邻特征之间的间距不得小于此值",
            ),
            MRCRuleType.MIN_SRAF_SIZE: MRCRuleConfig(
                rule_type=MRCRuleType.MIN_SRAF_SIZE,
                enabled=True,
                threshold_nm=30.0,
                severity=MRCRuleSeverity.ERROR,
                description="最小 SRAF 尺寸：辅助特征的尺寸不得小于此值",
                extra_params={"min_area_nm2": 900.0},
            ),
            MRCRuleType.NO_ACUTE_ANGLE: MRCRuleConfig(
                rule_type=MRCRuleType.NO_ACUTE_ANGLE,
                enabled=True,
                threshold_nm=45.0,
                severity=MRCRuleSeverity.ERROR,
                description="禁止锐角规则：多边形内角不得小于 90 度（或指定阈值角度）",
                extra_params={"min_angle_deg": 90.0},
            ),
            MRCRuleType.SRAF_MAIN_DISTANCE: MRCRuleConfig(
                rule_type=MRCRuleType.SRAF_MAIN_DISTANCE,
                enabled=True,
                threshold_nm=60.0,
                severity=MRCRuleSeverity.ERROR,
                description="辅助特征与主特征最小距离",
                extra_params={"max_distance_nm": 150.0},
            ),
        }

    def get_rule(self, rule_type: MRCRuleType) -> Optional[MRCRuleConfig]:
        """获取指定规则"""
        return self.rules.get(rule_type)

    def set_rule(self, rule_type: MRCRuleType, config: MRCRuleConfig) -> None:
        """设置规则配置"""
        self.rules[rule_type] = config

    def enable_rule(self, rule_type: MRCRuleType) -> None:
        """启用规则"""
        if rule_type in self.rules:
            self.rules[rule_type].enabled = True

    def disable_rule(self, rule_type: MRCRuleType) -> None:
        """禁用规则"""
        if rule_type in self.rules:
            self.rules[rule_type].enabled = False

    def enabled_rules(self) -> Dict[MRCRuleType, MRCRuleConfig]:
        """获取所有启用的规则"""
        return {k: v for k, v in self.rules.items() if v.enabled}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "technology_node": self.technology_node,
            "pixel_size": self.pixel_size,
            "rules": {k.value: v.to_dict() for k, v in self.rules.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MRCRules":
        rules = {}
        for k, v in data.get("rules", {}).items():
            rules[MRCRuleType(k)] = MRCRuleConfig.from_dict(v)
        return cls(
            rules=rules,
            technology_node=data.get("technology_node", "custom"),
            pixel_size=float(data.get("pixel_size", 1.0)),
        )


def load_default_rules(technology_node: str = "duv_arf") -> MRCRules:
    """
    加载指定技术节点的默认规则

    Args:
        technology_node: 技术节点类型 'duv_arf' (ArF 193nm) 或 'euv' (13.5nm)

    Returns:
        MRCRules 实例
    """
    rules = MRCRules()
    rules.technology_node = technology_node

    if technology_node == "duv_arf":
        rules.set_rule(MRCRuleType.MIN_LINE_WIDTH, MRCRuleConfig(
            rule_type=MRCRuleType.MIN_LINE_WIDTH,
            enabled=True,
            threshold_nm=45.0,
            severity=MRCRuleSeverity.FATAL,
            description="ArF 最小线宽 45nm",
        ))
        rules.set_rule(MRCRuleType.MIN_SPACING, MRCRuleConfig(
            rule_type=MRCRuleType.MIN_SPACING,
            enabled=True,
            threshold_nm=45.0,
            severity=MRCRuleSeverity.FATAL,
            description="ArF 最小间距 45nm",
        ))
        rules.set_rule(MRCRuleType.MIN_SRAF_SIZE, MRCRuleConfig(
            rule_type=MRCRuleType.MIN_SRAF_SIZE,
            enabled=True,
            threshold_nm=30.0,
            severity=MRCRuleSeverity.ERROR,
            description="ArF SRAF 最小尺寸 30nm",
            extra_params={"min_area_nm2": 900.0},
        ))
        rules.set_rule(MRCRuleType.SRAF_MAIN_DISTANCE, MRCRuleConfig(
            rule_type=MRCRuleType.SRAF_MAIN_DISTANCE,
            enabled=True,
            threshold_nm=60.0,
            severity=MRCRuleSeverity.ERROR,
            description="ArF SRAF 与主特征最小距离 60nm",
            extra_params={"max_distance_nm": 150.0},
        ))
    elif technology_node == "euv":
        rules.set_rule(MRCRuleType.MIN_LINE_WIDTH, MRCRuleConfig(
            rule_type=MRCRuleType.MIN_LINE_WIDTH,
            enabled=True,
            threshold_nm=16.0,
            severity=MRCRuleSeverity.FATAL,
            description="EUV 最小线宽 16nm",
        ))
        rules.set_rule(MRCRuleType.MIN_SPACING, MRCRuleConfig(
            rule_type=MRCRuleType.MIN_SPACING,
            enabled=True,
            threshold_nm=16.0,
            severity=MRCRuleSeverity.FATAL,
            description="EUV 最小间距 16nm",
        ))
        rules.set_rule(MRCRuleType.MIN_SRAF_SIZE, MRCRuleConfig(
            rule_type=MRCRuleType.MIN_SRAF_SIZE,
            enabled=True,
            threshold_nm=10.0,
            severity=MRCRuleSeverity.ERROR,
            description="EUV SRAF 最小尺寸 10nm",
            extra_params={"min_area_nm2": 100.0},
        ))
        rules.set_rule(MRCRuleType.SRAF_MAIN_DISTANCE, MRCRuleConfig(
            rule_type=MRCRuleType.SRAF_MAIN_DISTANCE,
            enabled=True,
            threshold_nm=20.0,
            severity=MRCRuleSeverity.ERROR,
            description="EUV SRAF 与主特征最小距离 20nm",
            extra_params={"max_distance_nm": 50.0},
        ))

    return rules


def load_rules_from_yaml(filepath: Union[str, Path]) -> MRCRules:
    """
    从 YAML 文件加载规则配置

    Args:
        filepath: YAML 配置文件路径

    Returns:
        MRCRules 实例
    """
    import yaml

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"规则配置文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return MRCRules.from_dict(data)
