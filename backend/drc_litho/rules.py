# -*- coding: utf-8 -*-
"""
版图设计规则邻近分析模块 - 光刻导向规则定义

定义桥连、断线、孤立线三类光刻风险的检查规则及其阈值配置。
支持按技术节点 (duv_arf / euv) 预设，支持 YAML 加载。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class LithoRuleType(Enum):
    BRIDGE_NARROW_GAP = "bridge_narrow_gap"
    BRIDGE_NECKING = "bridge_necking"
    BRIDGE_DENSE_CORNER = "bridge_dense_corner"
    BREAK_THIN_NECK = "break_thin_neck"
    BREAK_SHARP_TURN = "break_sharp_turn"
    BREAK_LINE_END = "break_line_end"
    ISOLATED_SMALL_FEATURE = "isolated_small_feature"
    ISOLATED_DANGLING_LINE = "isolated_dangling_line"
    ISOLATED_ORPHAN_PIXEL = "isolated_orphan_pixel"


class LithoRuleSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class LithoRuleConfig:
    rule_type: LithoRuleType
    enabled: bool = True
    threshold_nm: float = 0.0
    severity: LithoRuleSeverity = LithoRuleSeverity.WARNING
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
    def from_dict(cls, data: Dict[str, Any]) -> "LithoRuleConfig":
        return cls(
            rule_type=LithoRuleType(data["rule_type"]),
            enabled=data.get("enabled", True),
            threshold_nm=float(data.get("threshold_nm", 0.0)),
            severity=LithoRuleSeverity(data.get("severity", "warning")),
            description=data.get("description", ""),
            extra_params=data.get("extra_params", {}),
        )


@dataclass
class LithoDRRules:
    rules: Dict[LithoRuleType, LithoRuleConfig] = field(default_factory=dict)
    technology_node: str = "custom"
    pixel_size: float = 1.0

    def __post_init__(self):
        if not self.rules:
            self._init_default_rules()

    def _init_default_rules(self):
        self.rules = {
            LithoRuleType.BRIDGE_NARROW_GAP: LithoRuleConfig(
                rule_type=LithoRuleType.BRIDGE_NARROW_GAP,
                enabled=True,
                threshold_nm=40.0,
                severity=LithoRuleSeverity.CRITICAL,
                description="桥连风险: 相邻特征间距过窄，光刻后可能粘连",
                extra_params={"min_gap_nm": 40.0},
            ),
            LithoRuleType.BRIDGE_NECKING: LithoRuleConfig(
                rule_type=LithoRuleType.BRIDGE_NECKING,
                enabled=True,
                threshold_nm=30.0,
                severity=LithoRuleSeverity.CRITICAL,
                description="桥连风险: 局部线宽收窄（颈部），光刻后可能横向桥连",
                extra_params={"neck_ratio": 0.5},
            ),
            LithoRuleType.BRIDGE_DENSE_CORNER: LithoRuleConfig(
                rule_type=LithoRuleType.BRIDGE_DENSE_CORNER,
                enabled=True,
                threshold_nm=0.3,
                severity=LithoRuleSeverity.WARNING,
                description="桥连风险: 拐角密度过高，邻近拐角衍射叠加增加桥连概率",
                extra_params={"density_threshold": 0.3, "block_size": 32},
            ),
            LithoRuleType.BREAK_THIN_NECK: LithoRuleConfig(
                rule_type=LithoRuleType.BREAK_THIN_NECK,
                enabled=True,
                threshold_nm=25.0,
                severity=LithoRuleSeverity.FATAL,
                description="断线风险: 线宽过窄的颈部区域，光刻后可能断裂",
                extra_params={"min_width_nm": 25.0, "length_nm": 20.0},
            ),
            LithoRuleType.BREAK_SHARP_TURN: LithoRuleConfig(
                rule_type=LithoRuleType.BREAK_SHARP_TURN,
                enabled=True,
                threshold_nm=60.0,
                severity=LithoRuleSeverity.WARNING,
                description="断线风险: 急转弯角度过小，光刻后内侧可能断开",
                extra_params={"min_angle_deg": 60.0},
            ),
            LithoRuleType.BREAK_LINE_END: LithoRuleConfig(
                rule_type=LithoRuleType.BREAK_LINE_END,
                enabled=True,
                threshold_nm=20.0,
                severity=LithoRuleSeverity.WARNING,
                description="断线风险: 线端附近线宽不足，线端缩短效应可能导致断线",
                extra_params={"search_radius_nm": 20.0},
            ),
            LithoRuleType.ISOLATED_SMALL_FEATURE: LithoRuleConfig(
                rule_type=LithoRuleType.ISOLATED_SMALL_FEATURE,
                enabled=True,
                threshold_nm=20.0,
                severity=LithoRuleSeverity.WARNING,
                description="孤立线风险: 面积过小的孤立特征，可能无法有效成像",
                extra_params={"min_area_nm2": 400.0, "max_aspect_ratio": 5.0},
            ),
            LithoRuleType.ISOLATED_DANGLING_LINE: LithoRuleConfig(
                rule_type=LithoRuleType.ISOLATED_DANGLING_LINE,
                enabled=True,
                threshold_nm=5.0,
                severity=LithoRuleSeverity.WARNING,
                description="孤立线风险: 悬空短线段，至少一端无连接",
                extra_params={"min_length_nm": 5.0, "max_width_nm": 3.0},
            ),
            LithoRuleType.ISOLATED_ORPHAN_PIXEL: LithoRuleConfig(
                rule_type=LithoRuleType.ISOLATED_ORPHAN_PIXEL,
                enabled=True,
                threshold_nm=0.0,
                severity=LithoRuleSeverity.INFO,
                description="孤立线风险: 极小像素团（孤儿像素），可能是噪声",
                extra_params={"max_area_px": 4},
            ),
        }

    def get_rule(self, rule_type: LithoRuleType) -> Optional[LithoRuleConfig]:
        return self.rules.get(rule_type)

    def set_rule(self, rule_type: LithoRuleType, config: LithoRuleConfig) -> None:
        self.rules[rule_type] = config

    def enable_rule(self, rule_type: LithoRuleType) -> None:
        if rule_type in self.rules:
            self.rules[rule_type].enabled = True

    def disable_rule(self, rule_type: LithoRuleType) -> None:
        if rule_type in self.rules:
            self.rules[rule_type].enabled = False

    def enabled_rules(self) -> Dict[LithoRuleType, LithoRuleConfig]:
        return {k: v for k, v in self.rules.items() if v.enabled}

    def bridge_rules(self) -> Dict[LithoRuleType, LithoRuleConfig]:
        return {
            k: v for k, v in self.enabled_rules().items()
            if k.value.startswith("bridge_")
        }

    def break_rules(self) -> Dict[LithoRuleType, LithoRuleConfig]:
        return {
            k: v for k, v in self.enabled_rules().items()
            if k.value.startswith("break_")
        }

    def isolated_rules(self) -> Dict[LithoRuleType, LithoRuleConfig]:
        return {
            k: v for k, v in self.enabled_rules().items()
            if k.value.startswith("isolated_")
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "technology_node": self.technology_node,
            "pixel_size": self.pixel_size,
            "rules": {k.value: v.to_dict() for k, v in self.rules.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LithoDRRules":
        rules = {}
        for k, v in data.get("rules", {}).items():
            rules[LithoRuleType(k)] = LithoRuleConfig.from_dict(v)
        return cls(
            rules=rules,
            technology_node=data.get("technology_node", "custom"),
            pixel_size=float(data.get("pixel_size", 1.0)),
        )


def load_default_rules(technology_node: str = "duv_arf") -> "LithoDRRules":
    rules = LithoDRRules()
    rules.technology_node = technology_node

    if technology_node == "duv_arf":
        rules.set_rule(LithoRuleType.BRIDGE_NARROW_GAP, LithoRuleConfig(
            rule_type=LithoRuleType.BRIDGE_NARROW_GAP,
            enabled=True, threshold_nm=40.0,
            severity=LithoRuleSeverity.CRITICAL,
            description="ArF 桥连: 最小间距 40nm",
            extra_params={"min_gap_nm": 40.0},
        ))
        rules.set_rule(LithoRuleType.BRIDGE_NECKING, LithoRuleConfig(
            rule_type=LithoRuleType.BRIDGE_NECKING,
            enabled=True, threshold_nm=30.0,
            severity=LithoRuleSeverity.CRITICAL,
            description="ArF 桥连颈部: 最小局部线宽 30nm",
            extra_params={"neck_ratio": 0.5},
        ))
        rules.set_rule(LithoRuleType.BREAK_THIN_NECK, LithoRuleConfig(
            rule_type=LithoRuleType.BREAK_THIN_NECK,
            enabled=True, threshold_nm=25.0,
            severity=LithoRuleSeverity.FATAL,
            description="ArF 断线: 线宽低于 25nm 的颈部可能断裂",
            extra_params={"min_width_nm": 25.0, "length_nm": 20.0},
        ))
        rules.set_rule(LithoRuleType.BREAK_SHARP_TURN, LithoRuleConfig(
            rule_type=LithoRuleType.BREAK_SHARP_TURN,
            enabled=True, threshold_nm=60.0,
            severity=LithoRuleSeverity.WARNING,
            description="ArF 断线: 急转弯角度 < 60° 风险",
            extra_params={"min_angle_deg": 60.0},
        ))
        rules.set_rule(LithoRuleType.ISOLATED_SMALL_FEATURE, LithoRuleConfig(
            rule_type=LithoRuleType.ISOLATED_SMALL_FEATURE,
            enabled=True, threshold_nm=20.0,
            severity=LithoRuleSeverity.WARNING,
            description="ArF 孤立特征: 尺寸 < 20nm 可能无法成像",
            extra_params={"min_area_nm2": 400.0, "max_aspect_ratio": 5.0},
        ))
    elif technology_node == "euv":
        rules.set_rule(LithoRuleType.BRIDGE_NARROW_GAP, LithoRuleConfig(
            rule_type=LithoRuleType.BRIDGE_NARROW_GAP,
            enabled=True, threshold_nm=16.0,
            severity=LithoRuleSeverity.CRITICAL,
            description="EUV 桥连: 最小间距 16nm",
            extra_params={"min_gap_nm": 16.0},
        ))
        rules.set_rule(LithoRuleType.BRIDGE_NECKING, LithoRuleConfig(
            rule_type=LithoRuleType.BRIDGE_NECKING,
            enabled=True, threshold_nm=12.0,
            severity=LithoRuleSeverity.CRITICAL,
            description="EUV 桥连颈部: 最小局部线宽 12nm",
            extra_params={"neck_ratio": 0.5},
        ))
        rules.set_rule(LithoRuleType.BREAK_THIN_NECK, LithoRuleConfig(
            rule_type=LithoRuleType.BREAK_THIN_NECK,
            enabled=True, threshold_nm=10.0,
            severity=LithoRuleSeverity.FATAL,
            description="EUV 断线: 线宽低于 10nm 的颈部可能断裂",
            extra_params={"min_width_nm": 10.0, "length_nm": 8.0},
        ))
        rules.set_rule(LithoRuleType.BREAK_SHARP_TURN, LithoRuleConfig(
            rule_type=LithoRuleType.BREAK_SHARP_TURN,
            enabled=True, threshold_nm=45.0,
            severity=LithoRuleSeverity.WARNING,
            description="EUV 断线: 急转弯角度 < 45° 风险",
            extra_params={"min_angle_deg": 45.0},
        ))
        rules.set_rule(LithoRuleType.ISOLATED_SMALL_FEATURE, LithoRuleConfig(
            rule_type=LithoRuleType.ISOLATED_SMALL_FEATURE,
            enabled=True, threshold_nm=8.0,
            severity=LithoRuleSeverity.WARNING,
            description="EUV 孤立特征: 尺寸 < 8nm 可能无法成像",
            extra_params={"min_area_nm2": 64.0, "max_aspect_ratio": 5.0},
        ))

    return rules


def load_rules_from_yaml(filepath: Union[str, Path]) -> "LithoDRRules":
    import yaml

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"规则配置文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return LithoDRRules.from_dict(data)
