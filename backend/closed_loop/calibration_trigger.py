# -*- coding: utf-8 -*-
"""
自动触发 calibration 模块

基于对比分析结果判断是否需要重新校准模型参数。
若满足触发条件，则自动调用 calibration.pipeline 执行完整标定流程，
输出标定报告与更新后的配置文件。
"""

import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

from backend.calibration.schemas import (
    CalibrationConfig,
    CalibrationReport,
    CDSEMDataset,
    CalibrationParameterSet,
    InversionMethod,
)
from backend.calibration.pipeline import (
    CalibrationPipeline,
    run_calibration_pipeline,
)

from .schemas import (
    ComparisonResult,
    CalibrationTriggerThresholds,
    CalibrationTriggerResult,
    CalibrationTriggerReason,
)

logger = logging.getLogger(__name__)


class CalibrationTrigger:
    """
    校准触发器

    判定是否需要触发 calibration，并执行实际的标定流程。

    典型用法::

        trigger = CalibrationTrigger(config, thresholds)
        result = trigger.evaluate_and_run(comparison_result, dataset)
    """

    def __init__(self,
                 calibration_config: Optional[CalibrationConfig] = None,
                 thresholds: Optional[CalibrationTriggerThresholds] = None,
                 reference_config_path: Optional[Union[str, Path]] = None,
                 last_calibration_time: Optional[str] = None,
                 ):
        """
        Args:
            calibration_config: CalibrationConfig；None 则使用默认配置
            thresholds: 触发阈值；None 则使用默认
            reference_config_path: 参考 default_config.yaml 路径
            last_calibration_time: 上次校准时间戳，用于冷却期判断
        """
        self.config = calibration_config or CalibrationConfig()
        self.thresholds = thresholds or CalibrationTriggerThresholds()
        self.reference_config_path = (
            str(reference_config_path) if reference_config_path else None
        )
        self.last_calibration_time = last_calibration_time
        self._last_result: Optional[CalibrationTriggerResult] = None

    # ------------------------------------------------------------------
    # 触发判定
    # ------------------------------------------------------------------
    def should_trigger(self,
                       comparison: ComparisonResult,
                       dataset: Optional[CDSEMDataset] = None,
                       force: bool = False,
                       ) -> tuple[bool, List[str]]:
        """
        判断是否需要触发校准

        Args:
            comparison: 对比分析结果
            dataset: CD-SEM 数据集（用于检查点数）
            force: 强制触发（跳过阈值检查）

        Returns:
            (是否触发, 触发原因列表)
        """
        if force:
            return True, [CalibrationTriggerReason.MANUAL.value]

        reasons: List[str] = []

        if dataset is not None and len(dataset) < self.thresholds.min_points_required:
            reasons.append(
                f"数据点不足 ({len(dataset)} < {self.thresholds.min_points_required})"
            )
            return False, reasons

        if comparison.rmse > self.thresholds.rmse_threshold_nm:
            reasons.append(
                f"{CalibrationTriggerReason.RMSE_EXCEEDED.value}: "
                f"RMSE={comparison.rmse:.2f} nm > {self.thresholds.rmse_threshold_nm:.2f} nm"
            )

        if abs(comparison.mean_residual) > self.thresholds.bias_threshold_nm:
            reasons.append(
                f"{CalibrationTriggerReason.BIAS_DRIFT.value}: "
                f"|bias|={abs(comparison.mean_residual):.2f} nm > {self.thresholds.bias_threshold_nm:.2f} nm"
            )

        if comparison.max_abs_residual > self.thresholds.max_residual_threshold_nm:
            reasons.append(
                f"{CalibrationTriggerReason.MAX_RESIDUAL_EXCEEDED.value}: "
                f"max|residual|={comparison.max_abs_residual:.2f} nm > {self.thresholds.max_residual_threshold_nm:.2f} nm"
            )

        for gs in comparison.pattern_groups:
            if abs(gs.mean_residual) > self.thresholds.group_bias_threshold_nm:
                reasons.append(
                    f"{CalibrationTriggerReason.PATTERN_GROUP_DEVIATION.value}: "
                    f"[{gs.pattern_type}] bias={gs.mean_residual:+.2f} nm > "
                    f"±{self.thresholds.group_bias_threshold_nm:.2f} nm"
                )

        if not reasons and comparison.needs_calibration:
            reasons.append(comparison.calibration_reasons[0]
                           if comparison.calibration_reasons
                           else "comparison 模块判定需校准")

        cooldown_ok = self._check_cooldown()
        if not cooldown_ok and reasons:
            logger.info(
                f"校准触发条件满足，但处于冷却期 "
                f"(上次 {self.last_calibration_time})，暂不执行"
            )
            return False, reasons

        return len(reasons) > 0, reasons

    def _check_cooldown(self) -> bool:
        """检查是否在冷却期内"""
        if not self.last_calibration_time:
            return True
        try:
            last = datetime.strptime(
                self.last_calibration_time, '%Y-%m-%d %H:%M:%S'
            )
        except (ValueError, TypeError):
            return True
        elapsed = datetime.now() - last
        cooldown = timedelta(hours=self.thresholds.cooldown_hours)
        return elapsed >= cooldown

    # ------------------------------------------------------------------
    # 执行校准
    # ------------------------------------------------------------------
    def _build_pipeline_config(self,
                               dataset: CDSEMDataset,
                               output_dir: Union[str, Path],
                               ) -> CalibrationConfig:
        """构造 CalibrationPipeline 的配置"""
        cfg = CalibrationConfig()
        cfg.method = self.config.method
        cfg.parameters = CalibrationParameterSet()
        cfg.parameters.resist_threshold.vary = True
        cfg.parameters.diffusion_length.vary = True
        cfg.parameters.na_effective.vary = True
        cfg.parameters.sigma_effective.vary = True
        cfg.output_dir = str(output_dir)
        cfg.reference_config_path = self.reference_config_path
        cfg.use_measurement_weights = self.config.use_measurement_weights
        cfg.forward_model_complexity = self.config.forward_model_complexity
        cfg.nlls_max_iter = self.config.nlls_max_iter
        cfg.nlls_method = self.config.nlls_method
        cfg.generate_plots = self.config.generate_plots
        cfg.update_config = True
        cfg.dataset_path = None
        return cfg

    def run_calibration(self,
                        dataset: CDSEMDataset,
                        output_dir: Optional[Union[str, Path]] = None,
                        ) -> tuple[Optional[CalibrationReport], str, float]:
        """
        执行校准流程

        Args:
            dataset: CD-SEM 数据集
            output_dir: 校准结果输出目录；None 则使用 config.output_dir

        Returns:
            (CalibrationReport or None, 输出目录路径, 耗时秒数)
        """
        out_dir = Path(output_dir) if output_dir else Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pipeline_cfg = self._build_pipeline_config(dataset, out_dir)

        logger.info(
            f"开始执行模型校准: 数据点={len(dataset)}, "
            f"方法={pipeline_cfg.method.value}, 输出={out_dir}"
        )
        t0 = time.time()
        try:
            pipeline = CalibrationPipeline(pipeline_cfg)
            report = pipeline.run(dataset=dataset)
            duration = time.time() - t0
            logger.info(
                f"校准完成: 成功={report.inversion_result.success}, "
                f"耗时={duration:.1f}s, χ²/dof={report.inversion_result.reduced_chi2:.4f}"
            )
            return report, str(out_dir), duration
        except Exception as e:
            duration = time.time() - t0
            logger.error(f"校准执行失败: {e}", exc_info=True)
            return None, str(out_dir), duration

    # ------------------------------------------------------------------
    # 主入口：评估并执行
    # ------------------------------------------------------------------
    def evaluate_and_run(self,
                         comparison: ComparisonResult,
                         dataset: CDSEMDataset,
                         output_dir: Optional[Union[str, Path]] = None,
                         force: bool = False,
                         ) -> CalibrationTriggerResult:
        """
        评估触发条件并按需执行校准

        Args:
            comparison: 对比分析结果
            dataset: CD-SEM 数据集
            output_dir: 校准输出目录
            force: 强制触发

        Returns:
            CalibrationTriggerResult
        """
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        should, reasons = self.should_trigger(comparison, dataset, force=force)

        if not should:
            skipped = (
                reasons[0] if reasons
                else "偏差在阈值范围内，无需校准"
            )
            result = CalibrationTriggerResult(
                triggered=False,
                trigger_reasons=[],
                skipped_reason=skipped,
                timestamp=ts,
            )
            logger.info(f"跳过校准: {skipped}")
            self._last_result = result
            return result

        logger.info(f"触发校准，原因: {', '.join(reasons)}")
        report, out_path, duration = self.run_calibration(dataset, output_dir)

        if report is not None:
            self.last_calibration_time = ts

        result = CalibrationTriggerResult(
            triggered=report is not None,
            trigger_reasons=reasons,
            skipped_reason="" if report is not None else "校准执行异常",
            calibration_report=report,
            output_dir=out_path,
            duration_sec=duration,
            timestamp=ts,
        )
        self._last_result = result
        return result

    @property
    def last_result(self) -> Optional[CalibrationTriggerResult]:
        return self._last_result


def evaluate_and_trigger_calibration(
    comparison: ComparisonResult,
    dataset: CDSEMDataset,
    calibration_config: Optional[CalibrationConfig] = None,
    thresholds: Optional[CalibrationTriggerThresholds] = None,
    reference_config_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> CalibrationTriggerResult:
    """
    便捷函数：评估并触发校准

    Args:
        comparison: 对比结果
        dataset: CD-SEM 数据集
        calibration_config: 标定配置
        thresholds: 触发阈值
        reference_config_path: 参考配置路径
        output_dir: 输出目录
        force: 强制触发

    Returns:
        CalibrationTriggerResult
    """
    trigger = CalibrationTrigger(
        calibration_config=calibration_config,
        thresholds=thresholds,
        reference_config_path=reference_config_path,
    )
    return trigger.evaluate_and_run(
        comparison, dataset, output_dir=output_dir, force=force,
    )
