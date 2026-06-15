# -*- coding: utf-8 -*-
"""
计量一致性报告模块

对比目标 CD 与仿真 CD，输出全面的计量评估指标，包括：
1. Uniformity (均匀性): Within-Wafer, Wafer-to-Wafer, Lot-to-Lot
2. Linearity (线性度): CD 响应的线性偏差
3. Mean-to-Target (M2T): 平均偏差
4. Precision (精度): 重复性与再现性 (Gauge R&R)
5. Process Capability (工艺能力): Cp, Cpk 指数
"""

import numpy as np
from numba import jit
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import csv

from metrology.cd_extraction import (
    CDExtractionResult,
    MeasurementLine,
    extract_cd_multiline,
    CDExtractionMethod,
)

logger = logging.getLogger(__name__)


@dataclass
class CDTarget:
    """
    目标 CD 定义

    Attributes:
        line_name: 对应测量线名称
        target_cd_nm: 目标 CD 值 (nm)
        tolerance_nm: CD 容差 ± (nm)
        lower_spec_limit: 下规格限 (nm)
        upper_spec_limit: 上规格限 (nm)
    """
    line_name: str
    target_cd_nm: float
    tolerance_nm: float = 3.0

    @property
    def lower_spec_limit(self) -> float:
        return self.target_cd_nm - self.tolerance_nm

    @property
    def upper_spec_limit(self) -> float:
        return self.target_cd_nm + self.tolerance_nm


@dataclass
class UniformityMetrics:
    """
    CD 均匀性指标

    Attributes:
        mean_cd: 平均 CD (nm)
        std_cd: CD 标准差 (nm)
        range_cd: CD 极差 (max - min) (nm)
        uniformity_3sigma: 3σ 均匀性 (%) = 3σ / mean * 100
        uniformity_range: 极差均匀性 (%) = range / mean * 100
        max_deviation: 最大偏离均值的绝对值 (nm)
        min_cd: 最小 CD (nm)
        max_cd: 最大 CD (nm)
        n_measurements: 测量点数
    """
    mean_cd: float
    std_cd: float
    range_cd: float
    uniformity_3sigma: float
    uniformity_range: float
    max_deviation: float
    min_cd: float
    max_cd: float
    n_measurements: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_cd_nm': self.mean_cd,
            'std_cd_nm': self.std_cd,
            'range_cd_nm': self.range_cd,
            'uniformity_3sigma_pct': self.uniformity_3sigma,
            'uniformity_range_pct': self.uniformity_range,
            'max_deviation_nm': self.max_deviation,
            'min_cd_nm': self.min_cd,
            'max_cd_nm': self.max_cd,
            'n_measurements': self.n_measurements,
        }


@dataclass
class LinearityMetrics:
    """
    CD 线性度指标

    通过对一组不同目标 CD 的测量结果做线性回归，
    评估测量系统的线性响应。

    Attributes:
        slope: 拟合斜率 (理想为 1.0)
        intercept: 拟合截距 (nm, 理想为 0)
        r_squared: R² 决定系数 (理想为 1.0)
        max_deviation_nm: 最大线性偏差 (nm)
        mean_deviation_nm: 平均线性偏差 (nm)
        linearity_error_pct: 线性度误差 (%)
        target_cds: 目标 CD 列表 (nm)
        measured_cds: 实测 CD 列表 (nm)
    """
    slope: float
    intercept: float
    r_squared: float
    max_deviation_nm: float
    mean_deviation_nm: float
    linearity_error_pct: float
    target_cds: List[float]
    measured_cds: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'slope': self.slope,
            'intercept_nm': self.intercept,
            'r_squared': self.r_squared,
            'max_deviation_nm': self.max_deviation_nm,
            'mean_deviation_nm': self.mean_deviation_nm,
            'linearity_error_pct': self.linearity_error_pct,
        }


@dataclass
class PrecisionMetrics:
    """
    计量精度指标 (Gauge R&R 简化版)

    Attributes:
        repeatability_std: 重复性标准差 (nm) - 同条件多次测量
        repeatability_pct: 重复性占容差比例 (%)
        reproducibility_std: 再现性标准差 (nm) - 不同条件/操作者
        grr_std: Gauge R&R 合成标准差 (nm)
        grr_pct: Gauge R&R 占容差比例 (%，<10% 优秀，<30% 可接受)
        ndc: 可区分类别数 (Number of Distinct Categories, >5 合格)
        n_repeat: 重复测量次数
    """
    repeatability_std: float
    repeatability_pct: float
    reproducibility_std: float
    grr_std: float
    grr_pct: float
    ndc: float
    n_repeat: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'repeatability_std_nm': self.repeatability_std,
            'repeatability_pct_tolerance': self.repeatability_pct,
            'reproducibility_std_nm': self.reproducibility_std,
            'grr_std_nm': self.grr_std,
            'grr_pct_tolerance': self.grr_pct,
            'ndc': self.ndc,
            'n_repeat': self.n_repeat,
        }


@dataclass
class ProcessCapabilityMetrics:
    """
    工艺能力指标

    Attributes:
        cp: Cp = (USL - LSL) / (6σ)，工艺潜在能力
        cpk: Cpk = min( (USL - μ)/(3σ), (μ - LSL)/(3σ) )，实际工艺能力
        cpl: Cpl = (μ - LSL) / (3σ)，下限能力
        cpu: Cpu = (USL - μ) / (3σ)，上限能力
        pp: Pp = (USL - LSL) / (6σ_total)，过程性能
        ppk: Ppk，实际过程性能指数
        usl: 上规格限 (nm)
        lsl: 下规格限 (nm)
        mean_within_spec: 均值是否在规格内
        fraction_out_of_spec: 超出规格比例 (%)
    """
    cp: float
    cpk: float
    cpl: float
    cpu: float
    pp: float
    ppk: float
    usl: float
    lsl: float
    mean_within_spec: bool
    fraction_out_of_spec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'Cp': self.cp,
            'Cpk': self.cpk,
            'Cpl': self.cpl,
            'Cpu': self.cpu,
            'Pp': self.pp,
            'Ppk': self.ppk,
            'USL_nm': self.usl,
            'LSL_nm': self.lsl,
            'mean_within_spec': self.mean_within_spec,
            'fraction_out_of_spec_pct': self.fraction_out_of_spec,
        }


@dataclass
class CDMeasurementPoint:
    """
    单点 CD 测量记录

    Attributes:
        line_name: 测量线名称
        target_cd_nm: 目标 CD (nm)
        measured_cd_nm: 实测 CD (nm)
        cd_error_nm: CD 误差 = measured - target (nm)
        cd_error_pct: CD 误差百分比 (%)
        within_tolerance: 是否在容差范围内
        confidence: 提取置信度 (0~1)
        method: 使用的提取方法
    """
    line_name: str
    target_cd_nm: float
    measured_cd_nm: float
    cd_error_nm: float
    cd_error_pct: float
    within_tolerance: bool
    confidence: float
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'line_name': self.line_name,
            'target_cd_nm': self.target_cd_nm,
            'measured_cd_nm': self.measured_cd_nm,
            'cd_error_nm': self.cd_error_nm,
            'cd_error_pct': self.cd_error_pct,
            'within_tolerance': self.within_tolerance,
            'confidence': self.confidence,
            'method': self.method,
        }


@dataclass
class MetrologyReport:
    """
    完整计量一致性报告

    Attributes:
        report_id: 报告唯一标识
        timestamp: 生成时间戳
        method: 使用的 CD 提取方法
        measurements: 所有测量点记录
        uniformity: CD 均匀性指标
        linearity: CD 线性度指标 (可选)
        precision: 计量精度指标 (可选)
        process_capability: 工艺能力指标
        m2t: Mean-to-Target (nm)
        m2t_pct: Mean-to-Target (%)
        pass_rate: 合格率 (%)
        n_pass: 合格测量点数
        n_total: 总测量点数
        summary_text: 文本摘要
    """
    report_id: str
    timestamp: float
    method: str
    measurements: List[CDMeasurementPoint]
    uniformity: UniformityMetrics
    linearity: Optional[LinearityMetrics]
    precision: Optional[PrecisionMetrics]
    process_capability: ProcessCapabilityMetrics
    m2t: float
    m2t_pct: float
    pass_rate: float
    n_pass: int
    n_total: int
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp,
            'method': self.method,
            'm2t_nm': self.m2t,
            'm2t_pct': self.m2t_pct,
            'pass_rate_pct': self.pass_rate,
            'n_pass': self.n_pass,
            'n_total': self.n_total,
            'uniformity': self.uniformity.to_dict(),
            'linearity': self.linearity.to_dict() if self.linearity else None,
            'precision': self.precision.to_dict() if self.precision else None,
            'process_capability': self.process_capability.to_dict(),
            'measurements': [m.to_dict() for m in self.measurements],
            'summary': self.summary_text,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def export_csv(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'line_name', 'target_cd_nm', 'measured_cd_nm',
                'cd_error_nm', 'cd_error_pct', 'within_tolerance',
                'confidence', 'method'
            ])
            for m in self.measurements:
                writer.writerow([
                    m.line_name, m.target_cd_nm, m.measured_cd_nm,
                    m.cd_error_nm, m.cd_error_pct, m.within_tolerance,
                    m.confidence, m.method
                ])


def compute_uniformity(cd_values: List[float]) -> UniformityMetrics:
    """
    计算 CD 均匀性指标

    Args:
        cd_values: CD 测量值列表 (nm)

    Returns:
        UniformityMetrics
    """
    cds = np.array(cd_values, dtype=np.float64)
    n = len(cds)

    if n == 0:
        return UniformityMetrics(
            mean_cd=0.0, std_cd=0.0, range_cd=0.0,
            uniformity_3sigma=0.0, uniformity_range=0.0,
            max_deviation=0.0, min_cd=0.0, max_cd=0.0,
            n_measurements=0,
        )

    mean_cd = float(np.mean(cds))
    std_cd = float(np.std(cds, ddof=1)) if n > 1 else 0.0
    min_cd = float(np.min(cds))
    max_cd = float(np.max(cds))
    range_cd = max_cd - min_cd
    max_deviation = float(np.max(np.abs(cds - mean_cd)))

    uniformity_3sigma = (3.0 * std_cd / mean_cd * 100.0) if mean_cd > 0 else 0.0
    uniformity_range = (range_cd / mean_cd * 100.0) if mean_cd > 0 else 0.0

    return UniformityMetrics(
        mean_cd=mean_cd,
        std_cd=std_cd,
        range_cd=range_cd,
        uniformity_3sigma=uniformity_3sigma,
        uniformity_range=uniformity_range,
        max_deviation=max_deviation,
        min_cd=min_cd,
        max_cd=max_cd,
        n_measurements=n,
    )


def compute_linearity(target_cds: List[float],
                      measured_cds: List[float]) -> LinearityMetrics:
    """
    计算 CD 线性度指标

    Args:
        target_cds: 目标 CD 列表 (nm)
        measured_cds: 对应实测 CD 列表 (nm)

    Returns:
        LinearityMetrics
    """
    targets = np.array(target_cds, dtype=np.float64)
    measured = np.array(measured_cds, dtype=np.float64)

    if len(targets) < 2 or len(measured) < 2:
        return LinearityMetrics(
            slope=1.0, intercept=0.0, r_squared=1.0,
            max_deviation_nm=0.0, mean_deviation_nm=0.0,
            linearity_error_pct=0.0,
            target_cds=list(target_cds),
            measured_cds=list(measured_cds),
        )

    coeffs = np.polyfit(targets, measured, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    measured_pred = np.polyval(coeffs, targets)
    ss_res = np.sum((measured - measured_pred) ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    r_squared = float(1.0 - ss_res / (ss_tot + 1e-10))

    ideal_pred = targets
    deviations = np.abs(measured - ideal_pred)
    max_deviation = float(np.max(deviations))
    mean_deviation = float(np.mean(deviations))

    cd_range = float(np.max(targets) - np.min(targets)) if len(targets) > 1 else 1.0
    linearity_error = (max_deviation / cd_range * 100.0) if cd_range > 0 else 0.0

    return LinearityMetrics(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        max_deviation_nm=max_deviation,
        mean_deviation_nm=mean_deviation,
        linearity_error_pct=linearity_error,
        target_cds=list(target_cds),
        measured_cds=list(measured_cds),
    )


def compute_precision(repeat_measurements: List[List[float]],
                      tolerance_nm: float = 6.0) -> PrecisionMetrics:
    """
    计算计量精度 (Gauge R&R)

    Args:
        repeat_measurements: 重复测量数据，格式为 [[操作员1第1次, 操作员1第2次, ...], [操作员2第1次, ...]]
            或简化为 [[第1次测量所有点], [第2次测量所有点], ...]
        tolerance_nm: 总容差 (USL - LSL) (nm)

    Returns:
        PrecisionMetrics
    """
    if not repeat_measurements or len(repeat_measurements) < 2:
        return PrecisionMetrics(
            repeatability_std=0.0,
            repeatability_pct=0.0,
            reproducibility_std=0.0,
            grr_std=0.0,
            grr_pct=0.0,
            ndc=0.0,
            n_repeat=len(repeat_measurements),
        )

    repeats = [np.array(r, dtype=np.float64) for r in repeat_measurements]
    n_repeat = len(repeats)
    n_parts = len(repeats[0])

    part_means = np.zeros(n_parts, dtype=np.float64)
    repeat_std_list = []

    for i in range(n_parts):
        part_vals = np.array([repeats[r][i] for r in range(n_repeat)])
        part_means[i] = np.mean(part_vals)
        if n_repeat > 1:
            repeat_std_list.append(np.std(part_vals, ddof=1))

    repeatability_std = float(np.mean(repeat_std_list)) if repeat_std_list else 0.0
    part_variation_std = float(np.std(part_means, ddof=1)) if n_parts > 1 else 0.0

    reproducibility_std = float(max(0.0, np.sqrt(max(0.0, part_variation_std ** 2 - repeatability_std ** 2 / n_repeat))))

    grr_std = float(np.sqrt(repeatability_std ** 2 + reproducibility_std ** 2))
    tolerance_total = tolerance_nm

    repeatability_pct = (6.0 * repeatability_std / tolerance_total * 100.0) if tolerance_total > 0 else 0.0
    grr_pct = (6.0 * grr_std / tolerance_total * 100.0) if tolerance_total > 0 else 0.0

    pv = part_variation_std * 6.0
    ndc = float((pv / (grr_std * 2.0) * 1.41)) if grr_std > 0 else 0.0

    return PrecisionMetrics(
        repeatability_std=repeatability_std,
        repeatability_pct=repeatability_pct,
        reproducibility_std=reproducibility_std,
        grr_std=grr_std,
        grr_pct=grr_pct,
        ndc=ndc,
        n_repeat=n_repeat,
    )


def compute_process_capability(cd_values: List[float],
                                lsl: float,
                                usl: float) -> ProcessCapabilityMetrics:
    """
    计算工艺能力指标 (Cp, Cpk 等)

    Args:
        cd_values: CD 测量值列表 (nm)
        lsl: 下规格限 (Lower Specification Limit) (nm)
        usl: 上规格限 (Upper Specification Limit) (nm)

    Returns:
        ProcessCapabilityMetrics
    """
    cds = np.array(cd_values, dtype=np.float64)
    n = len(cds)

    if n == 0:
        return ProcessCapabilityMetrics(
            cp=0.0, cpk=0.0, cpl=0.0, cpu=0.0, pp=0.0, ppk=0.0,
            usl=usl, lsl=lsl, mean_within_spec=False,
            fraction_out_of_spec=100.0,
        )

    mean = float(np.mean(cds))
    std_within = float(np.std(cds, ddof=1)) if n > 1 else 0.0
    std_total = std_within

    cp = float((usl - lsl) / (6.0 * std_within)) if std_within > 0 else 0.0
    cpl = float((mean - lsl) / (3.0 * std_within)) if std_within > 0 else 0.0
    cpu = float((usl - mean) / (3.0 * std_within)) if std_within > 0 else 0.0
    cpk = float(min(cpl, cpu))

    pp = float((usl - lsl) / (6.0 * std_total)) if std_total > 0 else 0.0
    ppl = float((mean - lsl) / (3.0 * std_total)) if std_total > 0 else 0.0
    ppu = float((usl - mean) / (3.0 * std_total)) if std_total > 0 else 0.0
    ppk = float(min(ppl, ppu))

    mean_within_spec = bool(lsl <= mean <= usl)
    n_out = int(np.sum((cds < lsl) | (cds > usl)))
    fraction_out = float(n_out / n * 100.0)

    return ProcessCapabilityMetrics(
        cp=cp, cpk=cpk, cpl=cpl, cpu=cpu,
        pp=pp, ppk=ppk,
        usl=usl, lsl=lsl,
        mean_within_spec=mean_within_spec,
        fraction_out_of_spec=fraction_out,
    )


def generate_metrology_report(
    wafer_image: np.ndarray,
    measurement_lines: List[MeasurementLine],
    targets: List[CDTarget],
    method: Union[str, CDExtractionMethod] = CDExtractionMethod.THRESHOLD_CROSSING,
    pixel_size: float = 1.0,
    repeat_images: Optional[List[np.ndarray]] = None,
    **kwargs,
) -> MetrologyReport:
    """
    生成完整的计量一致性报告

    集成 CD 提取、均匀性、线性度、精度、工艺能力等所有指标计算。

    Args:
        wafer_image: 仿真晶圆图像 (2D)
        measurement_lines: 测量线列表
        targets: 对应目标 CD 列表 (与测量线一一对应)
        method: CD 提取方法
        pixel_size: 像素尺寸 (nm)
        repeat_images: 重复测量图像列表 (用于精度计算)
        **kwargs: 传递给 CD 提取的额外参数

    Returns:
        MetrologyReport
    """
    import time
    import uuid

    timestamp = time.time()
    report_id = str(uuid.uuid4())[:8]

    if isinstance(method, CDExtractionMethod):
        method_str = method.value
    else:
        method_str = str(method)

    cd_results = extract_cd_multiline(
        wafer_image, measurement_lines, method, pixel_size, **kwargs
    )

    measurements = []
    measured_cds = []
    target_cds = []
    tolerances = []

    for line, target, cd_res in zip(measurement_lines, targets, cd_results):
        target_val = target.target_cd_nm
        measured_val = cd_res.cd_value
        error = measured_val - target_val
        error_pct = (error / target_val * 100.0) if target_val > 0 else 0.0
        within_tol = bool(abs(error) <= target.tolerance_nm)

        mp = CDMeasurementPoint(
            line_name=line.name,
            target_cd_nm=target_val,
            measured_cd_nm=measured_val,
            cd_error_nm=float(error),
            cd_error_pct=float(error_pct),
            within_tolerance=within_tol,
            confidence=cd_res.confidence,
            method=method_str,
        )
        measurements.append(mp)
        measured_cds.append(measured_val)
        target_cds.append(target_val)
        tolerances.append(target.tolerance_nm)

    uniformity = compute_uniformity(measured_cds)

    if len(set(target_cds)) >= 2:
        linearity = compute_linearity(target_cds, measured_cds)
    else:
        linearity = None

    precision = None
    if repeat_images and len(repeat_images) >= 2:
        repeat_measurements = []
        for rep_img in repeat_images:
            rep_cd_res = extract_cd_multiline(
                rep_img, measurement_lines, method, pixel_size, **kwargs
            )
            repeat_measurements.append([r.cd_value for r in rep_cd_res])
        avg_tol = 2.0 * float(np.mean(tolerances)) if tolerances else 6.0
        precision = compute_precision(repeat_measurements, avg_tol)

    avg_lsl = float(np.mean([t.lower_spec_limit for t in targets]))
    avg_usl = float(np.mean([t.upper_spec_limit for t in targets]))
    process_capability = compute_process_capability(measured_cds, avg_lsl, avg_usl)

    m2t = float(np.mean([m.cd_error_nm for m in measurements]))
    m2t_pct = float(np.mean([m.cd_error_pct for m in measurements]))
    n_pass = int(sum(1 for m in measurements if m.within_tolerance))
    n_total = len(measurements)
    pass_rate = float(n_pass / n_total * 100.0) if n_total > 0 else 0.0

    cpk_status = "优秀" if process_capability.cpk >= 1.33 else ("可接受" if process_capability.cpk >= 1.0 else "不合格")
    grr_status = ""
    if precision:
        if precision.grr_pct < 10:
            grr_status = "优秀"
        elif precision.grr_pct < 30:
            grr_status = "可接受"
        else:
            grr_status = "不合格"

    summary = (
        f"计量报告 [{report_id}] 方法={method_str}\n"
        f"  合格率: {pass_rate:.1f}% ({n_pass}/{n_total})\n"
        f"  Mean-to-Target: {m2t:+.2f} nm ({m2t_pct:+.2f}%)\n"
        f"  均匀性 (3σ): {uniformity.uniformity_3sigma:.2f}% (范围: {uniformity.range_cd:.2f} nm)\n"
        f"  工艺能力 Cpk={process_capability.cpk:.2f} ({cpk_status})"
    )
    if precision:
        summary += f"\n  计量精度 GRR={precision.grr_pct:.1f}% ({grr_status}), NDC={precision.ndc:.1f}"
    if linearity:
        summary += f"\n  线性度 R²={linearity.r_squared:.4f}, 斜率={linearity.slope:.4f}"

    return MetrologyReport(
        report_id=report_id,
        timestamp=timestamp,
        method=method_str,
        measurements=measurements,
        uniformity=uniformity,
        linearity=linearity,
        precision=precision,
        process_capability=process_capability,
        m2t=m2t,
        m2t_pct=m2t_pct,
        pass_rate=pass_rate,
        n_pass=n_pass,
        n_total=n_total,
        summary_text=summary,
    )
