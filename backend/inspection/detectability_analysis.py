# -*- coding: utf-8 -*-
"""
可检测性与假缺陷率分析模块

评估 OPC 后掩模在检测机上的可检测性，包括：
1. 缺陷检测与分类（真实缺陷 vs 假缺陷）
2. 印刷性评估（判断缺陷是否影响晶圆成像）
3. ROC 曲线与 AUC 分析
4. 检测率、假警报率、准确率等指标计算
5. 最优检测阈值选择

用于研究不同检测模式、不同阈值下的检测性能，
优化掩模检测规格设定。
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
from scipy.ndimage import binary_dilation

from inspection.schemas import (
    DefectCandidate,
    DefectClass,
    DetectabilityResult,
    DifferenceMapResult,
    InspectionImageResult,
    InspectionAnalysisConfig,
    FullInspectionResult,
)
from core.imaging import OpticalSystem, PartialCoherentImaging, _apply_threshold
from core.litho_metrics import compute_cd

logger = logging.getLogger(__name__)


def _classify_defect_printability(
    mask_defective: np.ndarray,
    mask_reference: np.ndarray,
    optical_system: Optional[OpticalSystem] = None,
    threshold: float = 0.3,
    cd_tolerance: float = 0.1,
    pixel_size_nm: float = 1.0,
) -> bool:
    """
    评估缺陷是否影响印刷（是否为致命缺陷）

    通过光刻成像仿真，比较缺陷前后的晶圆 CD 变化，
    判断缺陷是否会导致印刷失效。

    Args:
        mask_defective: 含缺陷的掩模
        mask_reference: 标称掩模（无缺陷）
        optical_system: 光学系统参数，None 则使用默认
        threshold: 光刻胶阈值
        cd_tolerance: CD 相对容差 (0~1)
        pixel_size_nm: 像素尺寸 (nm/pixel)

    Returns:
        True 表示为印刷致命缺陷，False 表示不影响印刷
    """
    if optical_system is None:
        optical_system = OpticalSystem()

    if np.allclose(mask_defective, mask_reference):
        return False

    try:
        imaging = PartialCoherentImaging(optical_system, mask_defective.shape)

        aerial_ref = imaging.compute_aerial_image(mask_reference)
        wafer_ref = _apply_threshold(aerial_ref, threshold)

        aerial_def = imaging.compute_aerial_image(mask_defective)
        wafer_def = _apply_threshold(aerial_def, threshold)

        cd_ref_stats = compute_cd(wafer_ref, pixel_size=pixel_size_nm)
        cd_def_stats = compute_cd(wafer_def, pixel_size=pixel_size_nm)

        cd_ref = cd_ref_stats.get('cd_mean', 0)
        cd_def = cd_def_stats.get('cd_mean', 0)

        if cd_ref <= 0:
            return False

        cd_change_abs = abs(cd_def - cd_ref)
        cd_change_rel = cd_change_abs / cd_ref

        return cd_change_rel > cd_tolerance

    except Exception as e:
        logger.warning(f"印刷性评估失败，默认返回不致命: {e}")
        return False


def _match_detection_to_ground_truth(
    detected_candidates: List[DefectCandidate],
    ground_truth_defects: List[Dict[str, Any]],
    tolerance_nm: float = 50.0,
    pixel_size_nm: float = 1.0,
) -> Tuple[int, int, int, List[bool]]:
    """
    将检测结果与真实缺陷进行匹配

    基于距离的匹配策略：检测到的候选缺陷与真实缺陷
    中心距离在容差范围内则视为匹配成功。

    Args:
        detected_candidates: 检测到的缺陷候选列表
        ground_truth_defects: 真实缺陷列表（包含 center_y, center_x）
        tolerance_nm: 匹配容差距离 (nm)
        pixel_size_nm: 像素尺寸 (nm/pixel)

    Returns:
        (TP, FP, FN, is_matched_flags)
        - TP: 正确检测的真实缺陷数
        - FP: 误检数（没有对应真实缺陷）
        - FN: 漏检数（真实缺陷未被检测到）
        - is_matched_flags: 每个检测候选是否匹配到真实缺陷
    """
    tolerance_pixels = tolerance_nm / pixel_size_nm

    matched_gt = set()
    is_matched = [False] * len(detected_candidates)

    for i, candidate in enumerate(detected_candidates):
        min_dist = float('inf')
        matched_gt_idx = -1

        for j, gt in enumerate(ground_truth_defects):
            if j in matched_gt:
                continue

            gt_y = gt.get('center_y', gt.get('y', 0))
            gt_x = gt.get('center_x', gt.get('x', 0))

            dist = np.sqrt(
                (candidate.center_y - gt_y) ** 2 +
                (candidate.center_x - gt_x) ** 2
            )

            if dist < min_dist and dist < tolerance_pixels:
                min_dist = dist
                matched_gt_idx = j

        if matched_gt_idx >= 0:
            is_matched[i] = True
            matched_gt.add(matched_gt_idx)

    tp = sum(is_matched)
    fp = len(detected_candidates) - tp
    fn = len(ground_truth_defects) - len(matched_gt)

    return tp, fp, fn, is_matched


def _classify_defect(
    candidate: Dict[str, Any],
    is_matched: bool,
    is_printable: bool,
) -> DefectClass:
    """
    对检测到的缺陷候选进行分类

    Args:
        candidate: 候选缺陷特征
        is_matched: 是否匹配到真实缺陷
        is_printable: 是否为印刷致命缺陷

    Returns:
        缺陷分类
    """
    if is_matched and is_printable:
        return DefectClass.REAL_DEFECT
    elif is_matched and not is_printable:
        return DefectClass.NUISANCE_DEFECT
    elif not is_matched:
        return DefectClass.NUISANCE_DEFECT
    else:
        return DefectClass.NO_DEFECT


def _compute_defect_candidates(
    difference_result: DifferenceMapResult,
    pixel_size_nm: float = 1.0,
) -> List[DefectCandidate]:
    """
    从差异图结果创建缺陷候选列表

    Args:
        difference_result: 差异图计算结果
        pixel_size_nm: 像素尺寸 (nm/pixel)

    Returns:
        缺陷候选列表
    """
    candidates = []

    for region in difference_result.candidate_regions:
        bbox = tuple(region['bbox'])

        candidate = DefectCandidate(
            center_y=int(region['center_y']),
            center_x=int(region['center_x']),
            size_nm=float(region['size_nm']),
            area_pixels=int(region['area_pixels']),
            contrast=float(region['mean_difference']),
            intensity=float(region['mean_difference']),
            score=float(region['confidence']),
            defect_class=DefectClass.NO_DEFECT,
            is_printable=False,
            bounding_box=bbox,
        )
        candidates.append(candidate)

    return candidates


def _compute_roc_curve(
    difference_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    num_thresholds: int = 50,
) -> Dict[str, np.ndarray]:
    """
    计算 ROC 曲线数据

    Args:
        difference_map: 差异图（置信度图）
        ground_truth_mask: 真实缺陷二值掩模
        num_thresholds: 阈值采样数量

    Returns:
        字典，包含 fpr, tpr, thresholds
    """
    diff_flat = difference_map.flatten()
    gt_flat = ground_truth_mask.flatten().astype(bool)

    if not np.any(gt_flat):
        return {
            'fpr': np.array([0.0, 1.0]),
            'tpr': np.array([0.0, 1.0]),
            'thresholds': np.array([1.0, 0.0]),
        }

    max_diff = diff_flat.max()
    min_diff = diff_flat.min()
    thresholds = np.linspace(max_diff, min_diff, num_thresholds)

    tpr_list = []
    fpr_list = []

    total_positive = np.sum(gt_flat)
    total_negative = len(gt_flat) - total_positive

    for thresh in thresholds:
        predicted = diff_flat >= thresh

        tp = np.sum(predicted & gt_flat)
        fp = np.sum(predicted & (~gt_flat))

        tpr = tp / total_positive if total_positive > 0 else 0
        fpr = fp / total_negative if total_negative > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }


def _compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    计算 AUC (Area Under Curve)

    使用梯形法则积分。

    Args:
        fpr: 假正率数组
        tpr: 真正率数组

    Returns:
        AUC 值
    """
    if len(fpr) < 2:
        return 0.0

    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(tpr_sorted, fpr_sorted)
    else:
        auc = np.trapz(tpr_sorted, fpr_sorted)
    return float(max(0.0, min(1.0, auc)))


def _find_optimal_threshold(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    """
    寻找最优检测阈值（约登指数最大点）

    Youden's J statistic = Sensitivity + Specificity - 1
                        = TPR - FPR

    Args:
        fpr: 假正率数组
        tpr: 真正率数组
        thresholds: 阈值数组

    Returns:
        最优阈值
    """
    if len(fpr) == 0:
        return 0.0

    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)

    return float(thresholds[optimal_idx])


def analyze_detectability(
    difference_result: DifferenceMapResult,
    mask_test: np.ndarray,
    mask_reference: np.ndarray,
    ground_truth_defects: Optional[List[Dict[str, Any]]] = None,
    ground_truth_mask: Optional[np.ndarray] = None,
    config: Optional[InspectionAnalysisConfig] = None,
    optical_system: Optional[OpticalSystem] = None,
    matching_tolerance_nm: float = 50.0,
) -> DetectabilityResult:
    """
    分析掩模缺陷的可检测性

    综合评估检测系统的性能，包括检测率、假警报率、
    准确率等指标，并进行 ROC 和 AUC 分析。

    Args:
        difference_result: Die-to-Database 差异图结果
        mask_test: 待测掩模（可能含缺陷）
        mask_reference: 标称掩模（无缺陷）
        ground_truth_defects: 真实缺陷列表（可选，用于定量评估）
        ground_truth_mask: 真实缺陷二值掩模（可选，用于 ROC 计算）
        config: 分析配置
        optical_system: 光学系统参数（用于印刷性评估）
        matching_tolerance_nm: 检测匹配容差 (nm)

    Returns:
        DetectabilityResult，包含可检测性分析结果

    使用示例::

        result = analyze_detectability(
            diff_result, mask_def, mask_ref,
            ground_truth_defects=gt_defects,
            ground_truth_mask=gt_mask,
        )
        print(result.summary())
    """
    if config is None:
        config = InspectionAnalysisConfig()

    pixel_size_nm = config.inspection_config.optics.pixel_size_nm

    candidates = _compute_defect_candidates(difference_result, pixel_size_nm)

    if ground_truth_mask is None and ground_truth_defects is not None:
        ground_truth_mask = np.zeros_like(mask_test, dtype=bool)
        for gt in ground_truth_defects:
            y = int(gt.get('center_y', gt.get('y', 0)))
            x = int(gt.get('center_x', gt.get('x', 0)))
            size_nm = gt.get('size_nm', 10.0)
            size_pix = int(np.ceil(size_nm / pixel_size_nm))
            half = max(1, size_pix // 2)
            y1 = max(0, y - half)
            y2 = min(mask_test.shape[0], y + half + 1)
            x1 = max(0, x - half)
            x2 = min(mask_test.shape[1], x + half + 1)
            ground_truth_mask[y1:y2, x1:x2] = True

    tp = 0
    fp = 0
    fn = 0
    classified_candidates = []

    if ground_truth_defects is not None and len(ground_truth_defects) > 0:
        tp, fp, fn, is_matched_flags = _match_detection_to_ground_truth(
            candidates,
            ground_truth_defects,
            tolerance_nm=matching_tolerance_nm,
            pixel_size_nm=pixel_size_nm,
        )

        for candidate, is_matched in zip(candidates, is_matched_flags):
            y1, y2, x1, x2 = candidate.bounding_box
            y1, y2 = max(0, y1 - 2), min(mask_test.shape[0], y2 + 2)
            x1, x2 = max(0, x1 - 2), min(mask_test.shape[1], x2 + 2)

            mask_def_patch = mask_test[y1:y2, x1:x2]
            mask_ref_patch = mask_reference[y1:y2, x1:x2]

            is_printable = _classify_defect_printability(
                mask_def_patch,
                mask_ref_patch,
                optical_system=optical_system,
                cd_tolerance=config.printability_cd_tolerance,
                pixel_size_nm=pixel_size_nm,
            )

            defect_class = _classify_defect(
                {}, is_matched, is_printable
            )

            candidate.defect_class = defect_class
            candidate.is_printable = is_printable
            classified_candidates.append(candidate)
    else:
        for candidate in candidates:
            y1, y2, x1, x2 = candidate.bounding_box
            y1, y2 = max(0, y1 - 2), min(mask_test.shape[0], y2 + 2)
            x1, x2 = max(0, x1 - 2), min(mask_test.shape[1], x2 + 2)

            mask_def_patch = mask_test[y1:y2, x1:x2]
            mask_ref_patch = mask_reference[y1:y2, x1:x2]

            is_printable = _classify_defect_printability(
                mask_def_patch,
                mask_ref_patch,
                optical_system=optical_system,
                cd_tolerance=config.printability_cd_tolerance,
                pixel_size_nm=pixel_size_nm,
            )

            candidate.defect_class = (
                DefectClass.REAL_DEFECT if is_printable
                else DefectClass.NUISANCE_DEFECT
            )
            candidate.is_printable = is_printable
            classified_candidates.append(candidate)

    total_detected = len(classified_candidates)
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / total_detected if total_detected > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    if (detection_rate + precision) > 0:
        f1 = 2 * (detection_rate * precision) / (detection_rate + precision)
    else:
        f1 = 0.0

    roc_data = {}
    auc_score = 0.0
    optimal_threshold = difference_result.threshold_used

    if ground_truth_mask is not None and np.any(ground_truth_mask):
        roc_data = _compute_roc_curve(
            difference_result.difference_map,
            ground_truth_mask,
        )
        auc_score = _compute_auc(roc_data.get('fpr', np.array([])), roc_data.get('tpr', np.array([])))
        optimal_threshold = _find_optimal_threshold(
            roc_data.get('fpr', np.array([])),
            roc_data.get('tpr', np.array([])),
            roc_data.get('thresholds', np.array([difference_result.threshold_used])),
        )

    return DetectabilityResult(
        detected_defects=classified_candidates,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        detection_rate=float(detection_rate),
        false_alarm_rate=float(false_alarm_rate),
        precision=float(precision),
        f1_score=float(f1),
        roc_data=roc_data,
        auc_score=float(auc_score),
        optimal_threshold=float(optimal_threshold),
        pixel_size_nm=pixel_size_nm,
    )


def run_full_inspection_analysis(
    mask_test: np.ndarray,
    mask_reference: np.ndarray,
    config: Optional[InspectionAnalysisConfig] = None,
    ground_truth_defects: Optional[List[Dict[str, Any]]] = None,
    ground_truth_mask: Optional[np.ndarray] = None,
    optical_system: Optional[OpticalSystem] = None,
    seed: Optional[int] = None,
) -> FullInspectionResult:
    """
    执行完整的掩模检测分析流程

    从检测图像仿真 → Die-to-Database 差异计算 →
    可检测性与假缺陷率分析的完整流水线。

    Args:
        mask_test: 待测掩模（可能含缺陷）
        mask_reference: 参考掩模（标称设计）
        config: 完整分析配置
        ground_truth_defects: 真实缺陷列表（可选）
        ground_truth_mask: 真实缺陷二值掩模（可选）
        optical_system: 光学系统参数（用于印刷性评估）
        seed: 随机数种子（用于检测图像仿真）

    Returns:
        FullInspectionResult，包含完整分析结果

    使用示例::

        from inspection import run_full_inspection_analysis
        from inspection.schemas import InspectionConfig, InspectionMode

        config = InspectionAnalysisConfig(
            inspection_config=InspectionConfig(mode=InspectionMode.DARK_FIELD)
        )
        result = run_full_inspection_analysis(
            mask_defective, mask_nominal, config,
            ground_truth_defects=gt_list,
        )
        print(result.summary())
    """
    import time
    t_start = time.time()

    if config is None:
        config = InspectionAnalysisConfig()

    from inspection.inspection_simulator import simulate_inspection_image

    inspection_result = simulate_inspection_image(
        mask_test,
        mask_reference,
        config=config.inspection_config,
        seed=seed,
    )

    from inspection.die_to_database import compute_die_to_database_from_result

    difference_result = compute_die_to_database_from_result(
        inspection_result,
        config=config,
        align=True,
    )

    detectability_result = analyze_detectability(
        difference_result,
        mask_test,
        mask_reference,
        ground_truth_defects=ground_truth_defects,
        ground_truth_mask=ground_truth_mask,
        config=config,
        optical_system=optical_system,
    )

    total_time = time.time() - t_start

    gt_defects = ground_truth_defects if ground_truth_defects is not None else []

    return FullInspectionResult(
        inspection_result=inspection_result,
        difference_result=difference_result,
        detectability_result=detectability_result,
        config=config,
        ground_truth_defects=gt_defects,
        total_analysis_time_s=float(total_time),
    )


def compute_false_defect_rate(
    detectability_result: DetectabilityResult,
    area_per_die_mm2: float = 1.0,
) -> Dict[str, float]:
    """
    计算假缺陷率（每平方厘米的假缺陷数）

    用于评估检测系统的假警报性能。

    Args:
        detectability_result: 可检测性分析结果
        area_per_die_mm2: 每个 Die 的面积 (mm²)

    Returns:
        字典，包含：
            - 'nuisance_per_die': 每个 Die 的假缺陷数
            - 'nuisance_per_cm2': 每平方厘米的假缺陷数
            - 'real_per_die': 每个 Die 的真实致命缺陷数
            - 'total_defects_per_die': 每个 Die 的总检测缺陷数
    """
    total_detected = len(detectability_result.detected_defects)

    nuisance_count = sum(
        1 for d in detectability_result.detected_defects
        if d.defect_class == DefectClass.NUISANCE_DEFECT
    )

    real_count = sum(
        1 for d in detectability_result.detected_defects
        if d.defect_class == DefectClass.REAL_DEFECT
    )

    area_cm2 = area_per_die_mm2 / 100.0  # mm² → cm²

    return {
        'nuisance_per_die': float(nuisance_count),
        'nuisance_per_cm2': float(nuisance_count / area_cm2) if area_cm2 > 0 else 0.0,
        'real_per_die': float(real_count),
        'total_defects_per_die': float(total_detected),
        'nuisance_ratio': float(nuisance_count / total_detected) if total_detected > 0 else 0.0,
    }


def evaluate_detection_performance(
    difference_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    在指定阈值下评估检测性能

    Args:
        difference_map: 差异图
        ground_truth_mask: 真实缺陷二值掩模
        threshold: 检测阈值

    Returns:
        字典，包含各项性能指标
    """
    predicted = difference_map >= threshold
    gt = ground_truth_mask.astype(bool)

    tp = int(np.sum(predicted & gt))
    fp = int(np.sum(predicted & (~gt)))
    fn = int(np.sum((~predicted) & gt))
    tn = int(np.sum((~predicted) & (~gt)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    if (tpr + precision) > 0:
        f1 = 2 * tpr * precision / (tpr + precision)
    else:
        f1 = 0.0

    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if mcc_denom > 0:
        mcc = (tp * tn - fp * fn) / mcc_denom
    else:
        mcc = 0.0

    return {
        'threshold': float(threshold),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'tpr': float(tpr),
        'fpr': float(fpr),
        'precision': float(precision),
        'specificity': float(specificity),
        'accuracy': float(accuracy),
        'f1': float(f1),
        'mcc': float(mcc),
    }
