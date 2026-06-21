# -*- coding: utf-8 -*-
"""
在产掩模 PW 余量重评估模块

校准模型参数更新后，重新评估所有在产掩模的工艺窗口（Process Window）余量，
与校准前的 PW 指标对比，识别 PW 余量显著下降、需要重新 OPC 的掩模。
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np

from backend.analysis.process_window import (
    ProcessWindowAnalyzer,
    PWMetrics,
)
from backend.core.imaging import OpticalSystem
from backend.calibration.schemas import (
    CalibrationReport,
    InversionResult,
)

from .schemas import (
    ProductionMask,
    PWReassessmentResult,
    MaskPriority,
)

logger = logging.getLogger(__name__)


def _build_optical_system_from_calibration(
    inversion_result: Optional[InversionResult],
    reference_config_path: Optional[Union[str, Path]] = None,
) -> OpticalSystem:
    """
    从校准结果构建 OpticalSystem。

    优先使用校准后的参数，fallback 到参考配置或默认值。
    """
    optics = OpticalSystem()

    cal_vals: Dict[str, float] = {}
    if inversion_result is not None:
        cal_vals = dict(inversion_result.calibrated_values)

    if 'na_effective' in cal_vals:
        optics.na = float(cal_vals['na_effective'])
    if 'sigma_effective' in cal_vals:
        optics.sigma = float(cal_vals['sigma_effective'])
    if 'wavelength_effective' in cal_vals:
        optics.wavelength = float(cal_vals['wavelength_effective'])

    return optics


def _get_threshold_from_calibration(
    inversion_result: Optional[InversionResult],
) -> float:
    """从校准结果读取光刻胶阈值"""
    default = 0.3
    if inversion_result is None:
        return default
    return float(
        inversion_result.calibrated_values.get(
            'resist_threshold', default
        )
    )


def _load_mask_and_target(
    mask: ProductionMask,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    加载掩模图案和目标图案。

    支持 .npy 格式；若无法加载则返回 (None, None)。
    """
    try:
        from backend.utils.data_io import load_npy
    except ImportError:
        from utils.data_io import load_npy

    mask_arr = None
    target_arr = None

    if mask.mask_path:
        mp = Path(mask.mask_path)
        if mp.exists():
            try:
                mask_arr = load_npy(str(mp))
            except Exception as e:
                logger.warning(f"加载掩模图案失败 {mask.mask_id}: {e}")

    if mask.target_path:
        tp = Path(mask.target_path)
        if tp.exists():
            try:
                target_arr = load_npy(str(tp))
            except Exception as e:
                logger.warning(f"加载目标图案失败 {mask.mask_id}: {e}")

    if mask_arr is not None and target_arr is None:
        target_arr = (mask_arr > 0.5).astype(np.float64)

    return mask_arr, target_arr


class PWReassessor:
    """
    在产掩模 PW 余量重评估器

    典型用法::

        reassessor = PWReassessor(production_masks, config)
        result = reassessor.reevaluate_all(calibration_report)
    """

    def __init__(self,
                 production_masks: Optional[List[ProductionMask]] = None,
                 focus_range: Tuple[float, float, int] = (-150, 150, 11),
                 dose_range: Tuple[float, float, int] = (0.85, 1.15, 11),
                 cd_tolerance: float = 0.1,
                 pw_drop_threshold: float = 0.15,
                 reference_config_path: Optional[Union[str, Path]] = None,
                 ):
        """
        Args:
            production_masks: 在产掩模列表；None 则需后续通过 add_mask 添加
            focus_range: PW 扫描 focus 范围 (min, max, n_points)
            dose_range: PW 扫描 dose 范围 (min, max, n_points)
            cd_tolerance: CD 相对容差 (0~1)
            pw_drop_threshold: PW 面积下降超过此比例标记需重 OPC (0~1)
            reference_config_path: 参考配置路径
        """
        self.production_masks: List[ProductionMask] = production_masks or []
        self.focus_range = focus_range
        self.dose_range = dose_range
        self.cd_tolerance = cd_tolerance
        self.pw_drop_threshold = pw_drop_threshold
        self.reference_config_path = (
            str(reference_config_path) if reference_config_path else None
        )

    # ------------------------------------------------------------------
    # 掩模管理
    # ------------------------------------------------------------------
    def add_mask(self, mask: ProductionMask) -> None:
        """添加在产掩模"""
        self.production_masks.append(mask)

    def add_masks(self, masks: List[ProductionMask]) -> None:
        """批量添加在产掩模"""
        self.production_masks.extend(masks)

    def get_mask_by_id(self, mask_id: str) -> Optional[ProductionMask]:
        """按 ID 查找掩模"""
        for m in self.production_masks:
            if m.mask_id == mask_id:
                return m
        return None

    def list_critical_masks(self) -> List[ProductionMask]:
        """列出所有 CRITICAL 优先级的掩模"""
        return [
            m for m in self.production_masks
            if m.priority == MaskPriority.CRITICAL
        ]

    # ------------------------------------------------------------------
    # 单掩模 PW 评估
    # ------------------------------------------------------------------
    def _evaluate_single_mask(
        self,
        mask: ProductionMask,
        optics: OpticalSystem,
        threshold: float,
        store_as_last: bool = False,
        store_as_updated: bool = False,
    ) -> Optional[PWMetrics]:
        """
        评估单个掩模的 PW 指标

        Args:
            mask: 掩模信息
            optics: 光学系统参数（含校准后的 NA/sigma/波长）
            threshold: 光刻胶阈值
            store_as_last: 是否存入 mask.last_pw_metrics
            store_as_updated: 是否存入 mask.updated_pw_metrics

        Returns:
            PWMetrics 或 None（评估失败）
        """
        mask_arr, target_arr = _load_mask_and_target(mask)
        if mask_arr is None or target_arr is None:
            logger.warning(
                f"掩模 {mask.mask_id} 图案加载失败，跳过 PW 评估"
            )
            return None

        try:
            analyzer = ProcessWindowAnalyzer(
                mask=mask_arr,
                target=target_arr,
                optical_system=optics,
                threshold=threshold,
            )
            analyzer.scan(
                focus_range=self.focus_range,
                dose_range=self.dose_range,
                cd_tolerance=self.cd_tolerance,
            )
            metrics = analyzer.compute_pw_metrics(
                cd_tolerance=self.cd_tolerance,
            )

            if store_as_last:
                mask.last_pw_metrics = metrics
            if store_as_updated:
                mask.updated_pw_metrics = metrics

            logger.info(
                f"掩模 {mask.mask_id} PW 评估完成: "
                f"面积={metrics.pw_area:.1f}, DOF={metrics.depth_of_focus:.1f} nm, "
                f"EL={metrics.exposure_latitude:.2f}%"
            )
            return metrics

        except Exception as e:
            logger.error(
                f"掩模 {mask.mask_id} PW 评估失败: {e}", exc_info=True
            )
            return None

    # ------------------------------------------------------------------
    # 批量重评估
    # ------------------------------------------------------------------
    def reevaluate_all(
        self,
        calibration_report: Optional[CalibrationReport] = None,
        calibration_baseline: Optional[CalibrationReport] = None,
        only_calibrated: bool = True,
    ) -> PWReassessmentResult:
        """
        重新评估所有在产掩模的 PW 余量

        Args:
            calibration_report: 本次校准报告（含更新后的参数）；
                               None 则使用默认参数
            calibration_baseline: 校准前的基线报告（用于对比 last_pw_metrics）；
                                 None 则使用现有 last_pw_metrics
            only_calibrated: 仅当 calibration_report 非空（即校准已执行）时才重评估

        Returns:
            PWReassessmentResult
        """
        logger.info(
            f"开始 PW 重评估: {len(self.production_masks)} 个在产掩模"
        )
        t0 = time.time()

        if only_calibrated and calibration_report is None:
            logger.info("未执行校准，跳过 PW 重评估")
            return PWReassessmentResult(
                n_masks_total=len(self.production_masks),
            )

        inv_result = (
            calibration_report.inversion_result
            if calibration_report else None
        )

        updated_optics = _build_optical_system_from_calibration(
            inv_result, self.reference_config_path
        )
        updated_threshold = _get_threshold_from_calibration(inv_result)

        n_reevaluated = 0
        n_needs_ropc = 0
        pw_area_changes: List[float] = []
        critical_affected: List[str] = []

        for mask in self.production_masks:

            if mask.last_pw_metrics is None and calibration_baseline is None:
                baseline_inv = (
                    calibration_baseline.inversion_result
                    if calibration_baseline else None
                )
                baseline_optics = _build_optical_system_from_calibration(
                    baseline_inv, self.reference_config_path
                )
                baseline_th = _get_threshold_from_calibration(baseline_inv)
                self._evaluate_single_mask(
                    mask, baseline_optics, baseline_th,
                    store_as_last=True,
                )

            updated_metrics = self._evaluate_single_mask(
                mask, updated_optics, updated_threshold,
                store_as_updated=True,
            )

            if updated_metrics is None:
                continue

            n_reevaluated += 1
            delta = mask.compute_pw_delta()

            if delta is not None and mask.last_pw_metrics is not None:
                area_ratio = delta.get('pw_area_ratio', 0.0)
                pw_area_changes.append(area_ratio)

                if area_ratio < -self.pw_drop_threshold:
                    mask.needs_ropc = True
                    n_needs_ropc += 1
                    if mask.priority == MaskPriority.CRITICAL:
                        critical_affected.append(mask.mask_id)
                    logger.warning(
                        f"掩模 {mask.mask_id} PW 面积下降 "
                        f"{area_ratio * 100:.1f}%，标记需重 OPC"
                    )

        avg_change = float(np.mean(pw_area_changes)) if pw_area_changes else 0.0

        result = PWReassessmentResult(
            n_masks_total=len(self.production_masks),
            n_masks_reevaluated=n_reevaluated,
            n_masks_needs_ropc=n_needs_ropc,
            masks=list(self.production_masks),
            average_pw_area_change=avg_change,
            critical_masks_affected=critical_affected,
        )

        elapsed = time.time() - t0
        logger.info(
            f"PW 重评估完成: {n_reevaluated}/{len(self.production_masks)}, "
            f"需重 OPC={n_needs_ropc}, 平均 PW 变化={avg_change * 100:+.2f}%, "
            f"耗时 {elapsed:.1f}s"
        )
        return result

    # ------------------------------------------------------------------
    # 单掩模便捷方法
    # ------------------------------------------------------------------
    def reevaluate_mask(self,
                        mask_id: str,
                        calibration_report: Optional[CalibrationReport] = None,
                        ) -> Optional[ProductionMask]:
        """重评估单个掩模"""
        mask = self.get_mask_by_id(mask_id)
        if mask is None:
            logger.warning(f"掩模不存在: {mask_id}")
            return None

        inv_result = (
            calibration_report.inversion_result
            if calibration_report else None
        )
        optics = _build_optical_system_from_calibration(
            inv_result, self.reference_config_path
        )
        threshold = _get_threshold_from_calibration(inv_result)

        self._evaluate_single_mask(
            mask, optics, threshold, store_as_updated=True
        )
        mask.compute_pw_delta()
        if mask.pw_delta and mask.last_pw_metrics:
            ratio = mask.pw_delta.get('pw_area_ratio', 0.0)
            if ratio < -self.pw_drop_threshold:
                mask.needs_ropc = True
        return mask


def reevaluate_production_masks(
    production_masks: List[ProductionMask],
    calibration_report: Optional[CalibrationReport] = None,
    focus_range: Tuple[float, float, int] = (-150, 150, 11),
    dose_range: Tuple[float, float, int] = (0.85, 1.15, 11),
    cd_tolerance: float = 0.1,
    pw_drop_threshold: float = 0.15,
    reference_config_path: Optional[Union[str, Path]] = None,
) -> PWReassessmentResult:
    """
    便捷函数：批量重评估在产掩模 PW 余量

    Args:
        production_masks: 在产掩模列表
        calibration_report: 校准报告（含更新后的参数）
        focus_range: PW 扫描 focus 范围
        dose_range: PW 扫描 dose 范围
        cd_tolerance: CD 相对容差
        pw_drop_threshold: PW 下降阈值
        reference_config_path: 参考配置路径

    Returns:
        PWReassessmentResult
    """
    reassessor = PWReassessor(
        production_masks=production_masks,
        focus_range=focus_range,
        dose_range=dose_range,
        cd_tolerance=cd_tolerance,
        pw_drop_threshold=pw_drop_threshold,
        reference_config_path=reference_config_path,
    )
    return reassessor.reevaluate_all(calibration_report)
