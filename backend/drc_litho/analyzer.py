# -*- coding: utf-8 -*-
"""
版图设计规则邻近分析模块 - 主分析器入口

LithoDRCAnalyzer 是本模块的核心入口，整合桥连、断线、孤立线三类检查器，
对版图掩模执行光刻导向的轻量 DRC 分析，输出违规列表和 OPC 就绪度评分。

典型用法:
    from drc_litho import LithoDRCAnalyzer, load_default_rules

    rules = load_default_rules("duv_arf")
    analyzer = LithoDRCAnalyzer(rules)
    result = analyzer.analyze(mask_array, pixel_size=1.0)
    print(result.summary())
"""

import logging
import time
from typing import Optional, Dict, List, Any

import numpy as np

from .schemas import LithoDRCResult, LithoViolation, LithoViolationCategory
from .rules import LithoDRRules, LithoRuleType, LithoRuleConfig, load_default_rules
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

logger = logging.getLogger(__name__)

_BRIDGE_DISPATCH = {
    LithoRuleType.BRIDGE_NARROW_GAP: check_bridge_narrow_gap,
    LithoRuleType.BRIDGE_NECKING: check_bridge_necking,
    LithoRuleType.BRIDGE_DENSE_CORNER: check_bridge_dense_corner,
}

_BREAK_DISPATCH = {
    LithoRuleType.BREAK_THIN_NECK: check_break_thin_neck,
    LithoRuleType.BREAK_SHARP_TURN: check_break_sharp_turn,
    LithoRuleType.BREAK_LINE_END: check_break_line_end,
}

_ISOLATED_DISPATCH = {
    LithoRuleType.ISOLATED_SMALL_FEATURE: check_isolated_small_feature,
    LithoRuleType.ISOLATED_DANGLING_LINE: check_isolated_dangling_line,
    LithoRuleType.ISOLATED_ORPHAN_PIXEL: check_isolated_orphan_pixel,
}

_ALL_DISPATCH = {}
_ALL_DISPATCH.update(_BRIDGE_DISPATCH)
_ALL_DISPATCH.update(_BREAK_DISPATCH)
_ALL_DISPATCH.update(_ISOLATED_DISPATCH)


class LithoDRCAnalyzer:
    """
    光刻导向版图设计规则邻近分析器

    对版图掩模执行轻量级 DRC 分析，检测桥连、断线、孤立线风险，
    在 OPC 之前预警并给出修改建议，减少后续仿真的无效迭代。

    Args:
        rules: LithoDRRules 规则配置，None 使用默认规则
        categories: 要检查的类别列表，None 检查全部
            可选: "bridge", "break", "isolated"
    """

    def __init__(
        self,
        rules: Optional[LithoDRRules] = None,
        categories: Optional[List[str]] = None,
    ):
        self.rules = rules or LithoDRRules()
        self.categories = categories or ["bridge", "break", "isolated"]

    def analyze(
        self,
        mask: np.ndarray,
        pixel_size: float = 1.0,
        target_mask: Optional[np.ndarray] = None,
    ) -> LithoDRCResult:
        """
        执行完整的光刻导向 DRC 分析

        Args:
            mask: 二值掩模数组 (H, W)，值域 [0, 1] 或 [0, 255]
            pixel_size: 像素尺寸 (nm/pixel)
            target_mask: 目标掩模（可选，保留接口用于未来扩展）

        Returns:
            LithoDRCResult 分析结果
        """
        t_start = time.time()

        from .geometry import preprocess_mask
        mask_binary = preprocess_mask(mask)

        result = LithoDRCResult(
            mask_shape=mask_binary.shape,
            pixel_size=pixel_size,
            timestamp=time.time(),
        )

        rules_to_check = self._collect_rules()
        result.rules_checked = [rt.value for rt in rules_to_check.keys()]

        for rule_type, rule_config in rules_to_check.items():
            try:
                checker_fn = _ALL_DISPATCH.get(rule_type)
                if checker_fn is None:
                    logger.warning(f"未实现规则检查: {rule_type.value}")
                    continue

                violations = checker_fn(mask_binary, rule_config, pixel_size)
                if violations:
                    result.add_violations(violations)
                    logger.info(
                        f"规则 {rule_type.value}: 发现 {len(violations)} 处违规"
                    )
            except Exception as e:
                logger.error(f"检查规则 {rule_type.value} 时出错: {e}")

        result.check_duration_sec = time.time() - t_start
        return result

    def analyze_bridge_only(
        self,
        mask: np.ndarray,
        pixel_size: float = 1.0,
    ) -> LithoDRCResult:
        """仅执行桥连风险检查"""
        return self._analyze_category(mask, pixel_size, "bridge")

    def analyze_break_only(
        self,
        mask: np.ndarray,
        pixel_size: float = 1.0,
    ) -> LithoDRCResult:
        """仅执行断线风险检查"""
        return self._analyze_category(mask, pixel_size, "break")

    def analyze_isolated_only(
        self,
        mask: np.ndarray,
        pixel_size: float = 1.0,
    ) -> LithoDRCResult:
        """仅执行孤立线检查"""
        return self._analyze_category(mask, pixel_size, "isolated")

    def _analyze_category(
        self,
        mask: np.ndarray,
        pixel_size: float,
        category: str,
    ) -> LithoDRCResult:
        t_start = time.time()

        from .geometry import preprocess_mask
        mask_binary = preprocess_mask(mask)

        result = LithoDRCResult(
            mask_shape=mask_binary.shape,
            pixel_size=pixel_size,
            timestamp=time.time(),
        )

        rules_map = self._collect_rules_for_category(category)
        result.rules_checked = [rt.value for rt in rules_map.keys()]

        for rule_type, rule_config in rules_map.items():
            try:
                checker_fn = _ALL_DISPATCH.get(rule_type)
                if checker_fn is None:
                    continue
                violations = checker_fn(mask_binary, rule_config, pixel_size)
                if violations:
                    result.add_violations(violations)
            except Exception as e:
                logger.error(f"检查规则 {rule_type.value} 时出错: {e}")

        result.check_duration_sec = time.time() - t_start
        return result

    def _collect_rules(self) -> Dict[LithoRuleType, LithoRuleConfig]:
        rules = {}
        for cat in self.categories:
            rules.update(self._collect_rules_for_category(cat))
        return rules

    def _collect_rules_for_category(
        self, category: str
    ) -> Dict[LithoRuleType, LithoRuleConfig]:
        enabled = self.rules.enabled_rules()
        if category == "bridge":
            return {k: v for k, v in enabled.items() if k.value.startswith("bridge_")}
        elif category == "break":
            return {k: v for k, v in enabled.items() if k.value.startswith("break_")}
        elif category == "isolated":
            return {k: v for k, v in enabled.items() if k.value.startswith("isolated_")}
        return {}
