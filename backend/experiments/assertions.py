# -*- coding: utf-8 -*-
"""
回归断言引擎模块

对实验运行结果进行回归验证，确保算法改动不会导致结果漂移。

支持的断言类型:
    - mse_threshold:     最终 MSE 应低于阈值（如 < 1e-3）
    - convergence_steps: 在指定步数内收敛（如 100 步内）
    - golden_deviation:  与 golden 结果偏差不超过指定百分比（如 < 5%）
    - ssim_threshold:    SSIM 高于阈值
    - epe_threshold:     EPE 低于阈值（OPC 工作流）
    - loss_improvement:  损失改善比例不低于阈值（SMO 工作流）
"""

import json
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

from experiments.schema import AssertionConfig, GoldenReference

logger = logging.getLogger(__name__)


@dataclass
class AssertionResult:
    """
    单个断言的验证结果

    Attributes:
        assertion_type: 断言类型
        passed: 是否通过
        actual_value: 实际值
        expected_value: 期望值（阈值）
        message: 详细信息
    """
    assertion_type: str = ''
    passed: bool = False
    actual_value: Optional[float] = None
    expected_value: Optional[float] = None
    message: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'assertion_type': self.assertion_type,
            'passed': self.passed,
            'actual_value': self.actual_value,
            'expected_value': self.expected_value,
            'message': self.message,
        }


@dataclass
class AssertionReport:
    """
    完整的断言验证报告

    Attributes:
        experiment_name: 实验名称
        results: 各断言结果列表
        all_passed: 是否全部通过
        summary: 摘要文本
    """
    experiment_name: str = ''
    results: List[AssertionResult] = field(default_factory=list)
    all_passed: bool = False
    summary: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'all_passed': self.all_passed,
            'summary': self.summary,
            'results': [r.to_dict() for r in self.results],
        }


class RegressionAssertions:
    """
    回归断言引擎

    根据实验定义中的 assertions 列表，对运行结果进行验证。
    """

    def __init__(self, experiments_dir: Optional[str] = None):
        """
        Args:
            experiments_dir: 实验定义所在目录，用于解析相对路径的 golden 文件
        """
        self.experiments_dir = Path(experiments_dir) if experiments_dir else None

    def evaluate(self, experiment_name: str,
                 assertions: List[AssertionConfig],
                 result,
                 result_dir: Optional[str] = None) -> AssertionReport:
        """
        对实验结果执行所有回归断言

        Args:
            experiment_name: 实验名称
            assertions: 断言配置列表
            result: ExperimentResult 实例
            result_dir: 结果输出目录（用于查找 golden 文件）

        Returns:
            AssertionReport
        """
        report = AssertionReport(experiment_name=experiment_name)

        for assertion in assertions:
            ar = self._evaluate_single(assertion, result, result_dir)
            report.results.append(ar)

        report.all_passed = all(r.passed for r in report.results)

        passed_count = sum(1 for r in report.results if r.passed)
        total_count = len(report.results)
        report.summary = (
            f"实验 '{experiment_name}' 回归验证: "
            f"{passed_count}/{total_count} 断言通过"
        )

        if report.all_passed:
            logger.info(report.summary)
        else:
            failed = [r for r in report.results if not r.passed]
            details = "; ".join(r.message for r in failed)
            logger.warning(f"{report.summary} - 失败项: {details}")

        return report

    def _evaluate_single(self, assertion: AssertionConfig,
                         result,
                         result_dir: Optional[str] = None) -> AssertionResult:
        """执行单个断言"""
        handler_map = {
            'mse_threshold': self._check_mse_threshold,
            'convergence_steps': self._check_convergence_steps,
            'golden_deviation': self._check_golden_deviation,
            'ssim_threshold': self._check_ssim_threshold,
            'epe_threshold': self._check_epe_threshold,
            'loss_improvement': self._check_loss_improvement,
        }

        handler = handler_map.get(assertion.type)
        if handler is None:
            return AssertionResult(
                assertion_type=assertion.type,
                passed=False,
                message=f"不支持的断言类型: {assertion.type}",
            )

        return handler(assertion, result, result_dir)

    @staticmethod
    def _check_mse_threshold(assertion: AssertionConfig,
                             result,
                             result_dir: Optional[str]) -> AssertionResult:
        """MSE 阈值断言"""
        actual = result.final_mse
        if actual is None:
            return AssertionResult(
                assertion_type='mse_threshold',
                passed=False,
                actual_value=None,
                expected_value=assertion.threshold,
                message="final_mse 为 None，无法检查",
            )

        passed = actual < assertion.threshold
        return AssertionResult(
            assertion_type='mse_threshold',
            passed=passed,
            actual_value=actual,
            expected_value=assertion.threshold,
            message=(
                f"MSE={actual:.6e} {'<' if passed else '>='} "
                f"阈值 {assertion.threshold:.6e}"
            ),
        )

    @staticmethod
    def _check_convergence_steps(assertion: AssertionConfig,
                                 result,
                                 result_dir: Optional[str]) -> AssertionResult:
        """收敛步数断言"""
        converged = result.converged
        convergence_step = result.convergence_step

        if converged is None:
            convergence_step = result.total_iterations

        max_steps = assertion.max_steps

        if convergence_step is not None:
            passed = convergence_step <= max_steps
            actual = convergence_step
        else:
            passed = result.total_iterations <= max_steps
            actual = result.total_iterations

        return AssertionResult(
            assertion_type='convergence_steps',
            passed=passed,
            actual_value=float(actual),
            expected_value=float(max_steps),
            message=(
                f"收敛步数={actual} {'<=' if passed else '>'} "
                f"最大步数 {max_steps}"
            ),
        )

    def _check_golden_deviation(self, assertion: AssertionConfig,
                                result,
                                result_dir: Optional[str]) -> AssertionResult:
        """Golden 偏差断言"""
        golden = self._load_golden(assertion.golden_path, result_dir)
        if golden is None:
            return AssertionResult(
                assertion_type='golden_deviation',
                passed=False,
                message=f"无法加载 golden 参考文件: {assertion.golden_path}",
            )

        tolerance = assertion.tolerance if assertion.tolerance is not None else 0.05

        metrics_to_check = []
        if result.final_mse is not None and golden.final_mse is not None:
            metrics_to_check.append(('final_mse', result.final_mse, golden.final_mse))
        if result.final_ssim is not None and golden.final_ssim is not None:
            metrics_to_check.append(('final_ssim', result.final_ssim, golden.final_ssim))
        if result.final_loss is not None and golden.final_loss is not None:
            metrics_to_check.append(('final_loss', result.final_loss, golden.final_loss))

        for metric_name, actual_val, golden_val in metrics_to_check:
            if abs(golden_val) < 1e-15:
                continue
            deviation = abs(actual_val - golden_val) / abs(golden_val)
            if deviation > tolerance:
                return AssertionResult(
                    assertion_type='golden_deviation',
                    passed=False,
                    actual_value=deviation,
                    expected_value=tolerance,
                    message=(
                        f"{metric_name} 偏差={deviation:.4%} > "
                        f"容差 {tolerance:.4%} "
                        f"(实际={actual_val:.6e}, golden={golden_val:.6e})"
                    ),
                )

        max_deviation = 0.0
        for _, actual_val, golden_val in metrics_to_check:
            if abs(golden_val) >= 1e-15:
                d = abs(actual_val - golden_val) / abs(golden_val)
                max_deviation = max(max_deviation, d)

        return AssertionResult(
            assertion_type='golden_deviation',
            passed=True,
            actual_value=max_deviation,
            expected_value=tolerance,
            message=f"所有指标偏差 <= 容差 {tolerance:.4%}",
        )

    def _load_golden(self, golden_path: Optional[str],
                     result_dir: Optional[str]) -> Optional[GoldenReference]:
        """加载 golden 参考文件"""
        if golden_path is None:
            if result_dir is not None:
                golden_file = Path(result_dir) / 'golden.json'
                if golden_file.exists():
                    golden_path = str(golden_file)

        if golden_path is None:
            return None

        path = Path(golden_path)
        if not path.is_absolute() and self.experiments_dir is not None:
            path = self.experiments_dir / path

        if not path.exists():
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return GoldenReference.from_dict(data)
        except Exception as e:
            logger.warning(f"加载 golden 文件失败: {path} - {e}")
            return None

    @staticmethod
    def _check_ssim_threshold(assertion: AssertionConfig,
                              result,
                              result_dir: Optional[str]) -> AssertionResult:
        """SSIM 阈值断言"""
        actual = result.final_ssim
        if actual is None:
            return AssertionResult(
                assertion_type='ssim_threshold',
                passed=False,
                actual_value=None,
                expected_value=assertion.threshold,
                message="final_ssim 为 None，无法检查",
            )

        passed = actual >= assertion.threshold
        return AssertionResult(
            assertion_type='ssim_threshold',
            passed=passed,
            actual_value=actual,
            expected_value=assertion.threshold,
            message=(
                f"SSIM={actual:.6f} {'>=' if passed else '<'} "
                f"阈值 {assertion.threshold:.6f}"
            ),
        )

    @staticmethod
    def _check_epe_threshold(assertion: AssertionConfig,
                             result,
                             result_dir: Optional[str]) -> AssertionResult:
        """EPE 阈值断言（OPC 工作流）"""
        actual = result.custom_metrics.get('epe_mean')
        if actual is None:
            return AssertionResult(
                assertion_type='epe_threshold',
                passed=False,
                actual_value=None,
                expected_value=assertion.threshold,
                message="epe_mean 不在 custom_metrics 中，无法检查",
            )

        passed = actual < assertion.threshold
        return AssertionResult(
            assertion_type='epe_threshold',
            passed=passed,
            actual_value=actual,
            expected_value=assertion.threshold,
            message=(
                f"EPE_mean={actual:.4f}nm {'<' if passed else '>='} "
                f"阈值 {assertion.threshold:.4f}nm"
            ),
        )

    @staticmethod
    def _check_loss_improvement(assertion: AssertionConfig,
                                result,
                                result_dir: Optional[str]) -> AssertionResult:
        """损失改善比例断言（SMO 工作流）"""
        actual = result.custom_metrics.get('total_loss_improvement_ratio')
        if actual is None:
            if result.initial_mse is not None and result.final_mse is not None and result.initial_mse > 1e-15:
                actual = (result.initial_mse - result.final_mse) / result.initial_mse

        if actual is None:
            return AssertionResult(
                assertion_type='loss_improvement',
                passed=False,
                actual_value=None,
                expected_value=assertion.threshold,
                message="无法计算损失改善比例",
            )

        passed = actual >= assertion.threshold
        return AssertionResult(
            assertion_type='loss_improvement',
            passed=passed,
            actual_value=actual,
            expected_value=assertion.threshold,
            message=(
                f"损失改善比例={actual:.4%} {'>=' if passed else '<'} "
                f"阈值 {assertion.threshold:.4%}"
            ),
        )
