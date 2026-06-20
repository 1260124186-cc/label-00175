# -*- coding: utf-8 -*-
"""
Tapeout 门禁检查模块

在流片 (Tapeout) 前执行硬性 MRC 门禁检查，确保掩模符合所有制造规范。
支持配置化的门禁策略，包括严重级别阈值、违规数量限制、特定规则豁免等。
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Union
from pathlib import Path

import numpy as np

from .rules import MRCRules, MRCRuleType, MRCRuleSeverity, MRCRuleConfig
from .violations import MRCViolation, MRCCheckResult, ViolationType

logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """门禁状态"""
    PASS = "pass"
    FAIL = "fail"
    WAIVED = "waived"
    REVIEW_REQUIRED = "review_required"


class GatePolicyMode(Enum):
    """门禁策略模式"""
    STRICT = "strict"
    NORMAL = "normal"
    RELAXED = "relaxed"
    CUSTOM = "custom"


@dataclass
class GateViolationLimit:
    """各类违规的数量限制"""
    max_fatal: int = 0
    max_error: int = 0
    max_warning: int = 50
    max_info: int = 500
    max_total: int = 1000

    def to_dict(self) -> Dict[str, int]:
        return {
            "max_fatal": self.max_fatal,
            "max_error": self.max_error,
            "max_warning": self.max_warning,
            "max_info": self.max_info,
            "max_total": self.max_total,
        }


@dataclass
class GatePolicyConfig:
    """
    门禁策略配置

    Attributes:
        mode: 策略模式
        block_on_fatal: 致命违规是否阻止流片
        block_on_error: 错误违规是否阻止流片
        block_on_warning: 警告违规是否阻止流片
        block_on_info: 信息违规是否阻止流片
        violation_limits: 违规数量限制
        waived_rules: 豁免的规则类型列表
        waived_violations: 豁免的具体违规 ID 列表
        required_rules: 必须通过的规则类型（即使其他配置允许，这些规则也必须通过）
        max_violation_area_ratio: 最大违规面积占比 (0.0 - 1.0)
        allow_waive: 是否允许人工豁免
    """
    mode: GatePolicyMode = GatePolicyMode.NORMAL
    block_on_fatal: bool = True
    block_on_error: bool = True
    block_on_warning: bool = False
    block_on_info: bool = False
    violation_limits: GateViolationLimit = field(default_factory=GateViolationLimit)
    waived_rules: List[str] = field(default_factory=list)
    waived_violations: List[str] = field(default_factory=list)
    required_rules: List[str] = field(default_factory=list)
    max_violation_area_ratio: float = 0.05
    allow_waive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "block_on_fatal": self.block_on_fatal,
            "block_on_error": self.block_on_error,
            "block_on_warning": self.block_on_warning,
            "block_on_info": self.block_on_info,
            "violation_limits": self.violation_limits.to_dict(),
            "waived_rules": self.waived_rules,
            "waived_violations": self.waived_violations,
            "required_rules": self.required_rules,
            "max_violation_area_ratio": self.max_violation_area_ratio,
            "allow_waive": self.allow_waive,
        }


@dataclass
class GateCheckItem:
    """单个检查项的结果"""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class GateCheckResult:
    """
    Tapeout 门禁检查结果

    Attributes:
        status: 门禁状态 (PASS/FAIL/WAIVED/REVIEW_REQUIRED)
        passed: 是否通过门禁
        check_items: 各检查项的详细结果
        mrc_result: 关联的 MRC 检查结果
        blocking_violations: 阻止流片的违规列表
        waived_violations: 已豁免的违规列表
        needs_review: 需要人工审核的违规列表
        policy_config: 使用的门禁策略配置
        timestamp: 检查时间戳
        check_duration_sec: 检查耗时 (秒)
    """
    status: GateStatus = GateStatus.FAIL
    passed: bool = False
    check_items: List[GateCheckItem] = field(default_factory=list)
    mrc_result: Optional[MRCCheckResult] = None
    blocking_violations: List[MRCViolation] = field(default_factory=list)
    waived_violations_list: List[MRCViolation] = field(default_factory=list)
    needs_review: List[MRCViolation] = field(default_factory=list)
    policy_config: Optional[GatePolicyConfig] = None
    timestamp: float = 0.0
    check_duration_sec: float = 0.0

    def __post_init__(self):
        if self.mrc_result is not None:
            self.passed = self.status == GateStatus.PASS

    def to_dict(self, include_masks: bool = False) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "check_items": [item.to_dict() for item in self.check_items],
            "mrc_result": self.mrc_result.to_dict(include_masks=include_masks)
            if self.mrc_result else None,
            "blocking_violations_count": len(self.blocking_violations),
            "waived_violations_count": len(self.waived_violations_list),
            "needs_review_count": len(self.needs_review),
            "policy_config": self.policy_config.to_dict() if self.policy_config else None,
            "timestamp": self.timestamp,
            "check_duration_sec": self.check_duration_sec,
            "blocking_violations": [
                v.to_dict(include_masks=include_masks) for v in self.blocking_violations
            ],
        }

    def summary(self) -> str:
        """生成简要摘要"""
        status_text = {
            GateStatus.PASS: "通过",
            GateStatus.FAIL: "未通过",
            GateStatus.WAIVED: "已豁免",
            GateStatus.REVIEW_REQUIRED: "需人工审核",
        }
        lines = [
            f"Tapeout 门禁: {status_text.get(self.status, '未知')}",
            f"  状态: {self.status.value}",
            f"  是否通过: {'是' if self.passed else '否'}",
        ]
        if self.mrc_result:
            lines.extend([
                f"  MRC 总违规: {self.mrc_result.total_violations}",
                f"    致命 (FATAL): {self.mrc_result.fatal_count}",
                f"    错误 (ERROR): {self.mrc_result.error_count}",
                f"    警告 (WARNING): {self.mrc_result.warning_count}",
                f"    信息 (INFO): {self.mrc_result.info_count}",
            ])
        lines.extend([
            f"  阻止流片违规: {len(self.blocking_violations)}",
            f"  已豁免违规: {len(self.waived_violations_list)}",
            f"  需人工审核: {len(self.needs_review)}",
            f"  检查耗时: {self.check_duration_sec:.3f}s",
        ])
        if self.check_items:
            lines.append("  检查项:")
            for item in self.check_items:
                mark = "✓" if item.passed else "✗"
                lines.append(f"    {mark} {item.name}: {item.message}")
        return "\n".join(lines)

    def save_json(self, filepath: str, include_masks: bool = False) -> Path:
        """保存结果为 JSON 文件"""
        import json
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_masks=include_masks), f, indent=2, ensure_ascii=False)
        logger.info(f"门禁结果已保存: {filepath}")
        return filepath


class MRCGate:
    """
    MRC Tapeout 门禁检查器

    在流片前验证掩模是否满足所有制造规则要求，作为硬性门禁。

    使用方法:
        gate = MRCGate(policy_mode="normal")
        result = gate.validate(mrc_check_result)
        if result.passed:
            print("可以流片")
        else:
            print("存在违规，需要修复")
    """

    def __init__(self,
                 policy_config: Optional[GatePolicyConfig] = None,
                 policy_mode: Optional[Union[str, GatePolicyMode]] = None):
        """
        初始化门禁检查器

        Args:
            policy_config: 自定义策略配置
            policy_mode: 预设策略模式 ('strict', 'normal', 'relaxed')
        """
        if policy_config is not None:
            self.policy_config = policy_config
        else:
            mode = self._resolve_policy_mode(policy_mode)
            self.policy_config = self._build_policy_from_mode(mode)

    @staticmethod
    def _resolve_policy_mode(mode: Optional[Union[str, GatePolicyMode]]) -> GatePolicyMode:
        if mode is None:
            return GatePolicyMode.NORMAL
        if isinstance(mode, GatePolicyMode):
            return mode
        try:
            return GatePolicyMode(str(mode).lower())
        except ValueError:
            logger.warning(f"未知策略模式: {mode}, 使用默认 NORMAL")
            return GatePolicyMode.NORMAL

    @staticmethod
    def _build_policy_from_mode(mode: GatePolicyMode) -> GatePolicyConfig:
        """根据模式构建策略配置"""
        if mode == GatePolicyMode.STRICT:
            return GatePolicyConfig(
                mode=GatePolicyMode.STRICT,
                block_on_fatal=True,
                block_on_error=True,
                block_on_warning=True,
                block_on_info=False,
                violation_limits=GateViolationLimit(
                    max_fatal=0,
                    max_error=0,
                    max_warning=0,
                    max_info=100,
                    max_total=100,
                ),
                max_violation_area_ratio=0.001,
                allow_waive=False,
                required_rules=[rt.value for rt in MRCRuleType],
            )
        elif mode == GatePolicyMode.RELAXED:
            return GatePolicyConfig(
                mode=GatePolicyMode.RELAXED,
                block_on_fatal=True,
                block_on_error=False,
                block_on_warning=False,
                block_on_info=False,
                violation_limits=GateViolationLimit(
                    max_fatal=0,
                    max_error=10,
                    max_warning=500,
                    max_info=5000,
                    max_total=10000,
                ),
                max_violation_area_ratio=0.1,
                allow_waive=True,
            )
        else:
            return GatePolicyConfig(
                mode=GatePolicyMode.NORMAL,
                block_on_fatal=True,
                block_on_error=True,
                block_on_warning=False,
                block_on_info=False,
                violation_limits=GateViolationLimit(
                    max_fatal=0,
                    max_error=0,
                    max_warning=50,
                    max_info=500,
                    max_total=1000,
                ),
                max_violation_area_ratio=0.05,
                allow_waive=True,
            )

    # ------------------------------------------------------------------
    # 主验证入口
    # ------------------------------------------------------------------

    def validate(self,
                 mrc_result: MRCCheckResult,
                 mask: Optional[np.ndarray] = None,
                 ) -> GateCheckResult:
        """
        执行 Tapeout 门禁验证

        Args:
            mrc_result: MRC 检查结果
            mask: 原始掩模数组（可选，用于面积比例计算）

        Returns:
            GateCheckResult 门禁检查结果
        """
        t_start = time.time()
        result = GateCheckResult(
            mrc_result=mrc_result,
            policy_config=self.policy_config,
            timestamp=time.time(),
        )

        check_items: List[GateCheckItem] = []
        blocking: List[MRCViolation] = []
        waived: List[MRCViolation] = []
        needs_review: List[MRCViolation] = []

        all_violations = list(mrc_result.violations)

        # 1. 应用规则豁免
        active_violations = self._apply_rule_waives(all_violations, waived)

        # 2. 检查严重级别阻止策略
        item_severity = self._check_severity_policy(active_violations, blocking)
        check_items.append(item_severity)

        # 3. 检查违规数量限制
        item_limits = self._check_violation_limits(active_violations)
        check_items.append(item_limits)

        # 4. 检查必需规则
        item_required = self._check_required_rules(active_violations, blocking)
        check_items.append(item_required)

        # 5. 检查违规面积占比
        if mask is not None:
            item_area = self._check_violation_area_ratio(active_violations, mask)
            check_items.append(item_area)

        # 6. 识别需人工审核的违规
        self._identify_review_violations(active_violations, blocking, needs_review)

        # 汇总判断最终状态
        result.check_items = check_items
        result.blocking_violations = blocking
        result.waived_violations_list = waived
        result.needs_review = needs_review

        all_passed = all(item.passed for item in check_items)

        if all_passed and not blocking:
            if needs_review and self.policy_config.allow_waive:
                result.status = GateStatus.REVIEW_REQUIRED
                result.passed = False
            else:
                result.status = GateStatus.PASS
                result.passed = True
        elif blocking and not self.policy_config.allow_waive:
            result.status = GateStatus.FAIL
            result.passed = False
        elif blocking and self.policy_config.allow_waive:
            result.status = GateStatus.FAIL
            result.passed = False
        else:
            result.status = GateStatus.FAIL
            result.passed = False

        result.check_duration_sec = time.time() - t_start

        logger.info(
            f"Tapeout 门禁完成: 状态={result.status.value}, "
            f"阻止违规={len(blocking)}, 需审核={len(needs_review)}, "
            f"已豁免={len(waived)}"
        )

        return result

    # ------------------------------------------------------------------
    # 各项检查逻辑
    # ------------------------------------------------------------------

    def _apply_rule_waives(self,
                           violations: List[MRCViolation],
                           waived_list: List[MRCViolation],
                           ) -> List[MRCViolation]:
        """应用规则豁免，将豁免的违规从活跃列表移到豁免列表"""
        if not self.policy_config.waived_rules:
            return list(violations)

        waived_rule_types = set()
        for rule_str in self.policy_config.waived_rules:
            try:
                waived_rule_types.add(MRCRuleType(rule_str))
            except ValueError:
                logger.warning(f"未知的豁免规则类型: {rule_str}")

        active: List[MRCViolation] = []
        for v in violations:
            if v.rule_type in waived_rule_types:
                waived_list.append(v)
                logger.debug(f"违规已豁免: {v.rule_type.value} - {v.message}")
            else:
                active.append(v)

        return active

    def _check_severity_policy(self,
                               violations: List[MRCViolation],
                               blocking: List[MRCViolation],
                               ) -> GateCheckItem:
        """检查严重级别阻止策略"""
        policy = self.policy_config
        severity_checks = [
            (MRCRuleSeverity.FATAL, policy.block_on_fatal),
            (MRCRuleSeverity.ERROR, policy.block_on_error),
            (MRCRuleSeverity.WARNING, policy.block_on_warning),
            (MRCRuleSeverity.INFO, policy.block_on_info),
        ]

        failed_severities = []
        for severity, should_block in severity_checks:
            if not should_block:
                continue
            sev_violations = [v for v in violations if v.severity == severity]
            if sev_violations:
                failed_severities.append(f"{severity.value}({len(sev_violations)})")
                blocking.extend(sev_violations)

        if failed_severities:
            return GateCheckItem(
                name="严重级别检查",
                passed=False,
                message=f"存在阻止级别的违规: {', '.join(failed_severities)}",
                details={"failed_severities": failed_severities},
            )
        else:
            return GateCheckItem(
                name="严重级别检查",
                passed=True,
                message="无阻止级别的违规",
            )

    def _check_violation_limits(self,
                                violations: List[MRCViolation],
                                ) -> GateCheckItem:
        """检查违规数量限制"""
        limits = self.policy_config.violation_limits
        counts = {
            "fatal": sum(1 for v in violations if v.severity == MRCRuleSeverity.FATAL),
            "error": sum(1 for v in violations if v.severity == MRCRuleSeverity.ERROR),
            "warning": sum(1 for v in violations if v.severity == MRCRuleSeverity.WARNING),
            "info": sum(1 for v in violations if v.severity == MRCRuleSeverity.INFO),
            "total": len(violations),
        }

        failed_limits = []
        if counts["fatal"] > limits.max_fatal:
            failed_limits.append(f"致命违规 {counts['fatal']} > {limits.max_fatal}")
        if counts["error"] > limits.max_error:
            failed_limits.append(f"错误违规 {counts['error']} > {limits.max_error}")
        if counts["warning"] > limits.max_warning:
            failed_limits.append(f"警告违规 {counts['warning']} > {limits.max_warning}")
        if counts["info"] > limits.max_info:
            failed_limits.append(f"信息违规 {counts['info']} > {limits.max_info}")
        if counts["total"] > limits.max_total:
            failed_limits.append(f"总违规 {counts['total']} > {limits.max_total}")

        if failed_limits:
            return GateCheckItem(
                name="违规数量限制",
                passed=False,
                message=f"违规数量超限: {'; '.join(failed_limits)}",
                details={"counts": counts, "limits": limits.to_dict()},
            )
        else:
            return GateCheckItem(
                name="违规数量限制",
                passed=True,
                message=(
                    f"违规数量在限制内: 致命={counts['fatal']}, 错误={counts['error']}, "
                    f"警告={counts['warning']}, 信息={counts['info']}, 总计={counts['total']}"
                ),
                details={"counts": counts},
            )

    def _check_required_rules(self,
                              violations: List[MRCViolation],
                              blocking: List[MRCViolation],
                              ) -> GateCheckItem:
        """检查必需规则是否都通过"""
        if not self.policy_config.required_rules:
            return GateCheckItem(
                name="必需规则检查",
                passed=True,
                message="无必需规则配置",
            )

        required_types = set()
        for rule_str in self.policy_config.required_rules:
            try:
                required_types.add(MRCRuleType(rule_str))
            except ValueError:
                logger.warning(f"未知的必需规则类型: {rule_str}")

        violated_required = set()
        for v in violations:
            if v.rule_type in required_types:
                violated_required.add(v.rule_type)
                blocking.append(v)

        if violated_required:
            violated_names = [rt.value for rt in violated_required]
            return GateCheckItem(
                name="必需规则检查",
                passed=False,
                message=f"必需规则违规: {', '.join(violated_names)}",
                details={"violated_required_rules": violated_names},
            )
        else:
            return GateCheckItem(
                name="必需规则检查",
                passed=True,
                message=f"所有 {len(required_types)} 条必需规则均通过",
                details={"required_rules": [rt.value for rt in required_types]},
            )

    def _check_violation_area_ratio(self,
                                    violations: List[MRCViolation],
                                    mask: np.ndarray,
                                    ) -> GateCheckItem:
        """检查违规面积占比"""
        if mask.size == 0:
            return GateCheckItem(
                name="违规面积占比",
                passed=True,
                message="掩模为空，跳过面积检查",
            )

        total_area = float(mask.size)
        violation_area = sum(v.region.area_pixels for v in violations)
        ratio = violation_area / total_area if total_area > 0 else 0.0
        max_ratio = self.policy_config.max_violation_area_ratio

        if ratio > max_ratio:
            return GateCheckItem(
                name="违规面积占比",
                passed=False,
                message=(
                    f"违规面积占比 {ratio:.4%} 超过阈值 {max_ratio:.4%}"
                ),
                details={
                    "violation_area_pixels": violation_area,
                    "total_area_pixels": int(total_area),
                    "ratio": ratio,
                    "max_ratio": max_ratio,
                },
            )
        else:
            return GateCheckItem(
                name="违规面积占比",
                passed=True,
                message=(
                    f"违规面积占比 {ratio:.4%} 在阈值 {max_ratio:.4%} 内"
                ),
                details={
                    "violation_area_pixels": violation_area,
                    "total_area_pixels": int(total_area),
                    "ratio": ratio,
                },
            )

    @staticmethod
    def _identify_review_violations(violations: List[MRCViolation],
                                    blocking: List[MRCViolation],
                                    needs_review: List[MRCViolation]) -> None:
        """识别需要人工审核的违规"""
        blocking_ids = set(id(v) for v in blocking)
        for v in violations:
            if id(v) in blocking_ids:
                continue
            if v.severity in (MRCRuleSeverity.ERROR, MRCRuleSeverity.WARNING):
                if v.rule_type in (
                    MRCRuleType.MIN_LINE_WIDTH,
                    MRCRuleType.MIN_SPACING,
                    MRCRuleType.NO_ACUTE_ANGLE,
                ):
                    needs_review.append(v)

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def set_policy_mode(self, mode: Union[str, GatePolicyMode]) -> None:
        """切换预设策略模式"""
        self.policy_config = self._build_policy_from_mode(
            self._resolve_policy_mode(mode)
        )

    def waive_rule(self, rule_type: Union[str, MRCRuleType]) -> None:
        """添加规则豁免"""
        if isinstance(rule_type, str):
            rule_type = MRCRuleType(rule_type)
        if rule_type.value not in self.policy_config.waived_rules:
            self.policy_config.waived_rules.append(rule_type.value)

    def unwaive_rule(self, rule_type: Union[str, MRCRuleType]) -> None:
        """移除规则豁免"""
        if isinstance(rule_type, str):
            rule_type = MRCRuleType(rule_type)
        if rule_type.value in self.policy_config.waived_rules:
            self.policy_config.waived_rules.remove(rule_type.value)

    def add_required_rule(self, rule_type: Union[str, MRCRuleType]) -> None:
        """添加必需规则"""
        if isinstance(rule_type, str):
            rule_type = MRCRuleType(rule_type)
        if rule_type.value not in self.policy_config.required_rules:
            self.policy_config.required_rules.append(rule_type.value)

    def remove_required_rule(self, rule_type: Union[str, MRCRuleType]) -> None:
        """移除必需规则"""
        if isinstance(rule_type, str):
            rule_type = MRCRuleType(rule_type)
        if rule_type.value in self.policy_config.required_rules:
            self.policy_config.required_rules.remove(rule_type.value)
