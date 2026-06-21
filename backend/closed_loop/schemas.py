# -*- coding: utf-8 -*-
"""
Fab 闭环反馈校准：数据结构定义

定义闭环系统的核心数据结构：
- Fab 数据导入配置与结果
- 仿真预测与量产对比结果
- 校准触发配置与结果
- 在产掩模 PW 重评估结果
- 完整闭环周期记录
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime

from backend.calibration.schemas import (
    CalibrationConfig,
    CalibrationReport,
    CDSEMDataset,
)
from backend.analysis.process_window import PWMetrics


class ClosedLoopState(Enum):
    """闭环周期状态枚举"""
    IDLE = "idle"
    IMPORTING = "importing"
    COMPARING = "comparing"
    CALIBRATING = "calibrating"
    REASSESSING_PW = "reassessing_pw"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CalibrationTriggerReason(Enum):
    """触发校准的原因"""
    MANUAL = "manual"
    RMSE_EXCEEDED = "rmse_exceeded"
    BIAS_DRIFT = "bias_drift"
    MAX_RESIDUAL_EXCEEDED = "max_residual_exceeded"
    PATTERN_GROUP_DEVIATION = "pattern_group_deviation"
    SCHEDULED = "scheduled"


class MaskPriority(Enum):
    """在产掩模优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FabImportConfig:
    """
    Fab 数据导入配置

    Attributes:
        watch_dir: 监控的 Fab 数据目录路径
        file_pattern: 文件名匹配模式（glob 风格，如 'cd_sem_*.csv'）
        archive_dir: 已处理文件归档目录（None 则不归档）
        history_file: 导入历史记录 JSON 路径
        encoding: CSV 文件编码
        delimiter: CSV 分隔符
        lookback_days: 回溯多少天内的文件（0 表示全部）
        auto_archive: 导入成功后是否自动归档
    """
    watch_dir: str
    file_pattern: str = "cd_sem_*.csv"
    archive_dir: Optional[str] = None
    history_file: str = "./closed_loop/import_history.json"
    encoding: str = "utf-8-sig"
    delimiter: str = ","
    lookback_days: int = 0
    auto_archive: bool = True


@dataclass
class ImportedFileRecord:
    """单个已导入文件的记录"""
    file_path: str
    file_hash: str
    file_size: int
    import_timestamp: str
    n_points: int
    fab_name: str
    lot_id: str
    wafer_id: str
    mask_set_id: str
    process_node: str
    success: bool
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'file_size': self.file_size,
            'import_timestamp': self.import_timestamp,
            'n_points': self.n_points,
            'fab_name': self.fab_name,
            'lot_id': self.lot_id,
            'wafer_id': self.wafer_id,
            'mask_set_id': self.mask_set_id,
            'process_node': self.process_node,
            'success': self.success,
            'error_message': self.error_message,
        }


@dataclass
class FabImportResult:
    """
    Fab 数据导入结果

    Attributes:
        imported_files: 本次导入的文件记录列表
        merged_dataset: 合并后的 CD-SEM 数据集
        new_files_count: 新导入文件数量
        skipped_files_count: 跳过（已导入过）的文件数量
        failed_files_count: 导入失败的文件数量
    """
    imported_files: List[ImportedFileRecord] = field(default_factory=list)
    merged_dataset: Optional[CDSEMDataset] = None
    new_files_count: int = 0
    skipped_files_count: int = 0
    failed_files_count: int = 0

    @property
    def total_points(self) -> int:
        return len(self.merged_dataset) if self.merged_dataset else 0

    def summary(self) -> str:
        lines = [
            "=== Fab 数据导入结果 ===",
            f"  新导入文件: {self.new_files_count}",
            f"  跳过(已处理): {self.skipped_files_count}",
            f"  导入失败:   {self.failed_files_count}",
            f"  总量测点:   {self.total_points}",
        ]
        for rec in self.imported_files:
            status = "✓" if rec.success else "✗"
            lines.append(
                f"    {status} {Path(rec.file_path).name}: "
                f"{rec.n_points} 点, lot={rec.lot_id}, wafer={rec.wafer_id}"
            )
        return "\n".join(lines)


@dataclass
class PerPointComparison:
    """单测点对比结果"""
    measurement_id: str
    target_cd: float
    measured_cd: float
    predicted_cd: float
    residual: float
    relative_error: float
    focus: float
    dose: float
    pattern_type: str
    site_name: str = ""
    layer: str = ""


@dataclass
class PatternGroupStats:
    """按图形类型分组的统计"""
    pattern_type: str
    n_points: int
    mean_residual: float
    std_residual: float
    rmse: float
    max_abs_residual: float
    bias_95ci: Tuple[float, float]


@dataclass
class ComparisonResult:
    """
    仿真预测与量产量测对比结果

    Attributes:
        n_points: 参与对比的量测点数量
        mean_residual: 平均残差 (measured - predicted) nm
        std_residual: 残差标准差 nm
        rmse: 均方根误差 nm
        max_abs_residual: 最大绝对残差 nm
        median_abs_residual: 绝对残差中位数 nm
        relative_rmse: 相对 RMSE (RMSE / 平均 target_cd)
        per_point: 每个测点的详细对比
        pattern_groups: 按图形类型分组统计
        trend_detected: 是否检测到显著趋势
        needs_calibration: 是否建议触发校准
        calibration_reasons: 建议校准的原因列表
    """
    n_points: int = 0
    mean_residual: float = 0.0
    std_residual: float = 0.0
    rmse: float = 0.0
    max_abs_residual: float = 0.0
    median_abs_residual: float = 0.0
    relative_rmse: float = 0.0
    per_point: List[PerPointComparison] = field(default_factory=list)
    pattern_groups: List[PatternGroupStats] = field(default_factory=list)
    trend_detected: bool = False
    needs_calibration: bool = False
    calibration_reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== 仿真 vs 量产 对比结果 ===",
            f"  量测点数量:     {self.n_points}",
            f"  平均残差:       {self.mean_residual:+.3f} nm",
            f"  残差标准差:     {self.std_residual:.3f} nm",
            f"  RMSE:           {self.rmse:.3f} nm",
            f"  相对 RMSE:      {self.relative_rmse * 100:.2f}%",
            f"  最大|残差|:     {self.max_abs_residual:.3f} nm",
            f"  |残差|中位数:   {self.median_abs_residual:.3f} nm",
            f"  检测到趋势:     {'是' if self.trend_detected else '否'}",
            f"  建议校准:       {'是' if self.needs_calibration else '否'}",
        ]
        if self.calibration_reasons:
            lines.append("  校准原因:")
            for r in self.calibration_reasons:
                lines.append(f"    - {r}")
        if self.pattern_groups:
            lines.append("  按图形类型统计:")
            for g in self.pattern_groups:
                lines.append(
                    f"    {g.pattern_type:<18s} N={g.n_points:<4d} "
                    f"bias={g.mean_residual:+.3f}±{g.std_residual:.3f} nm "
                    f"RMSE={g.rmse:.3f} nm"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_points': self.n_points,
            'mean_residual': float(self.mean_residual),
            'std_residual': float(self.std_residual),
            'rmse': float(self.rmse),
            'max_abs_residual': float(self.max_abs_residual),
            'median_abs_residual': float(self.median_abs_residual),
            'relative_rmse': float(self.relative_rmse),
            'trend_detected': self.trend_detected,
            'needs_calibration': self.needs_calibration,
            'calibration_reasons': list(self.calibration_reasons),
            'pattern_groups': [
                {
                    'pattern_type': g.pattern_type,
                    'n_points': g.n_points,
                    'mean_residual': float(g.mean_residual),
                    'std_residual': float(g.std_residual),
                    'rmse': float(g.rmse),
                    'max_abs_residual': float(g.max_abs_residual),
                }
                for g in self.pattern_groups
            ],
        }


@dataclass
class CalibrationTriggerThresholds:
    """
    校准触发阈值配置

    Attributes:
        rmse_threshold_nm: RMSE 超过此值触发校准 (nm)
        bias_threshold_nm: 平均偏差绝对值超过此值触发 (nm)
        max_residual_threshold_nm: 任何单点残差绝对值超过此值触发 (nm)
        relative_rmse_threshold: 相对 RMSE 超过此比例触发 (0~1)
        group_bias_threshold_nm: 任一图形组平均偏差超过此值触发 (nm)
        min_points_required: 触发校准所需最少数据点
        cooldown_hours: 两次校准之间的最小间隔小时数
    """
    rmse_threshold_nm: float = 2.0
    bias_threshold_nm: float = 1.0
    max_residual_threshold_nm: float = 5.0
    relative_rmse_threshold: float = 0.05
    group_bias_threshold_nm: float = 1.5
    min_points_required: int = 15
    cooldown_hours: float = 24.0


@dataclass
class CalibrationTriggerResult:
    """
    校准触发结果

    Attributes:
        triggered: 是否实际执行了校准
        trigger_reasons: 触发原因列表
        skipped_reason: 未触发时的跳过原因
        calibration_report: 校准报告（若执行了）
        output_dir: 校准输出目录
        duration_sec: 校准耗时（秒）
        timestamp: 执行时间戳
    """
    triggered: bool = False
    trigger_reasons: List[str] = field(default_factory=list)
    skipped_reason: str = ""
    calibration_report: Optional[CalibrationReport] = None
    output_dir: str = ""
    duration_sec: float = 0.0
    timestamp: str = ""

    def summary(self) -> str:
        if self.triggered:
            lines = [
                "=== 模型校准已执行 ===",
                f"  触发原因: {', '.join(self.trigger_reasons)}",
                f"  耗时:     {self.duration_sec:.1f} s",
                f"  输出目录: {self.output_dir}",
            ]
            if self.calibration_report is not None:
                r = self.calibration_report.inversion_result
                lines.append(f"  收敛状态: {'成功' if r.success else '失败'}")
                lines.append(f"  χ²/dof:   {r.reduced_chi2:.4f}")
                lines.append("  标定参数:")
                for name, val in r.calibrated_values.items():
                    unc = r.uncertainties.get(name, 0.0)
                    lines.append(f"    {name:<22s} = {val:+.6f} ± {unc:.6f}")
        else:
            lines = [
                "=== 跳过校准 ===",
                f"  原因: {self.skipped_reason}",
            ]
        return "\n".join(lines)


@dataclass
class ProductionMask:
    """
    在产掩模信息

    Attributes:
        mask_id: 掩模唯一标识
        mask_set_id: 所属掩模组
        layer: 工艺层
        priority: 优先级
        mask_path: 掩模图案文件路径 (GDS/npy)
        target_path: 目标图案文件路径
        last_pw_metrics: 上次 PW 评估指标（校准前）
        updated_pw_metrics: 更新后的 PW 评估指标（校准后）
        pw_delta: PW 指标变化量
        needs_ropc: 是否需要重新 OPC（PW 余量显著下降）
        comments: 备注
    """
    mask_id: str
    mask_set_id: str = ""
    layer: str = ""
    priority: MaskPriority = MaskPriority.MEDIUM
    mask_path: Optional[str] = None
    target_path: Optional[str] = None
    last_pw_metrics: Optional[PWMetrics] = None
    updated_pw_metrics: Optional[PWMetrics] = None
    pw_delta: Optional[Dict[str, float]] = None
    needs_ropc: bool = False
    comments: str = ""

    def compute_pw_delta(self) -> Optional[Dict[str, float]]:
        """计算校准前后 PW 指标的变化量"""
        if self.last_pw_metrics is None or self.updated_pw_metrics is None:
            return None
        delta = {
            'pw_area': (self.updated_pw_metrics.pw_area
                        - self.last_pw_metrics.pw_area),
            'pw_area_ratio': (
                (self.updated_pw_metrics.pw_area
                 - self.last_pw_metrics.pw_area)
                / max(self.last_pw_metrics.pw_area, 1e-9)
            ),
            'depth_of_focus': (self.updated_pw_metrics.depth_of_focus
                               - self.last_pw_metrics.depth_of_focus),
            'exposure_latitude': (
                self.updated_pw_metrics.exposure_latitude
                - self.last_pw_metrics.exposure_latitude
            ),
            'best_cd_error': (self.updated_pw_metrics.best_cd_error
                              - self.last_pw_metrics.best_cd_error),
        }
        self.pw_delta = delta
        return delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mask_id': self.mask_id,
            'mask_set_id': self.mask_set_id,
            'layer': self.layer,
            'priority': self.priority.value,
            'mask_path': self.mask_path,
            'target_path': self.target_path,
            'needs_ropc': self.needs_ropc,
            'comments': self.comments,
            'pw_delta': self.pw_delta,
        }


@dataclass
class PWReassessmentResult:
    """
    PW 余量重评估结果

    Attributes:
        n_masks_total: 在产掩模总数
        n_masks_reevaluated: 已完成重评估的数量
        n_masks_needs_ropc: 需要重新 OPC 的数量
        masks: 各掩模的详细评估结果
        average_pw_area_change: 平均 PW 面积变化比例
        critical_masks_affected: 受影响的关键掩模列表
    """
    n_masks_total: int = 0
    n_masks_reevaluated: int = 0
    n_masks_needs_ropc: int = 0
    masks: List[ProductionMask] = field(default_factory=list)
    average_pw_area_change: float = 0.0
    critical_masks_affected: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== 在产掩模 PW 余量重评估 ===",
            f"  在产掩模总数:   {self.n_masks_total}",
            f"  已评估数量:     {self.n_masks_reevaluated}",
            f"  需重 OPC 数量:  {self.n_masks_needs_ropc}",
            f"  平均 PW 面积变化: {self.average_pw_area_change * 100:+.2f}%",
        ]
        if self.critical_masks_affected:
            lines.append("  受影响的关键掩模:")
            for mid in self.critical_masks_affected:
                lines.append(f"    - {mid}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_masks_total': self.n_masks_total,
            'n_masks_reevaluated': self.n_masks_reevaluated,
            'n_masks_needs_ropc': self.n_masks_needs_ropc,
            'average_pw_area_change': float(self.average_pw_area_change),
            'critical_masks_affected': list(self.critical_masks_affected),
            'masks': [m.to_dict() for m in self.masks],
        }


@dataclass
class ClosedLoopConfig:
    """
    闭环反馈系统完整配置

    Attributes:
        import_config: Fab 数据导入配置
        trigger_thresholds: 校准触发阈值
        calibration_config: 标定模块配置（可选，None 则用默认）
        output_dir: 闭环运行输出目录
        reference_config_path: 参考配置文件 (default_config.yaml) 路径
        reevaluate_pw: 是否执行 PW 重评估
        pw_scan_focus: PW 扫描 focus 范围 (min, max, n)
        pw_scan_dose: PW 扫描 dose 范围 (min, max, n)
        pw_cd_tolerance: PW 判定 CD 容差
        pw_drop_threshold: PW 面积下降超过此比例标记需重 OPC (0~1)
    """
    import_config: FabImportConfig
    trigger_thresholds: CalibrationTriggerThresholds = field(
        default_factory=CalibrationTriggerThresholds
    )
    calibration_config: Optional[CalibrationConfig] = None
    output_dir: str = "./closed_loop/output"
    reference_config_path: Optional[str] = None
    reevaluate_pw: bool = True
    pw_scan_focus: Tuple[float, float, int] = (-150, 150, 11)
    pw_scan_dose: Tuple[float, float, int] = (0.85, 1.15, 11)
    pw_cd_tolerance: float = 0.1
    pw_drop_threshold: float = 0.15


@dataclass
class ClosedLoopCycle:
    """
    单个完整闭环周期记录

    Attributes:
        cycle_id: 周期唯一标识
        start_time: 开始时间
        end_time: 结束时间
        state: 执行状态
        import_result: Fab 数据导入结果
        comparison_result: 仿真预测 vs 量产对比结果
        calibration_result: 校准执行结果
        pw_result: PW 重评估结果
        error_message: 失败时的错误信息
    """
    cycle_id: str
    start_time: str = field(
        default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    end_time: str = ""
    state: ClosedLoopState = ClosedLoopState.IDLE
    import_result: Optional[FabImportResult] = None
    comparison_result: Optional[ComparisonResult] = None
    calibration_result: Optional[CalibrationTriggerResult] = None
    pw_result: Optional[PWReassessmentResult] = None
    error_message: str = ""

    def mark_completed(self) -> None:
        self.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.state != ClosedLoopState.FAILED:
            self.state = ClosedLoopState.COMPLETED

    def mark_failed(self, error: str) -> None:
        self.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.state = ClosedLoopState.FAILED
        self.error_message = error

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f" Fab 闭环反馈校准报告 [周期 {self.cycle_id}]",
            "=" * 60,
            f"  状态:     {self.state.value}",
            f"  开始时间: {self.start_time}",
            f"  结束时间: {self.end_time}",
            "",
        ]
        if self.import_result is not None:
            lines.append(self.import_result.summary())
            lines.append("")
        if self.comparison_result is not None:
            lines.append(self.comparison_result.summary())
            lines.append("")
        if self.calibration_result is not None:
            lines.append(self.calibration_result.summary())
            lines.append("")
        if self.pw_result is not None:
            lines.append(self.pw_result.summary())
            lines.append("")
        if self.state == ClosedLoopState.FAILED:
            lines.append(f"  错误: {self.error_message}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cycle_id': self.cycle_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'state': self.state.value,
            'error_message': self.error_message,
            'import_result': (
                {
                    'new_files_count': self.import_result.new_files_count,
                    'skipped_files_count': self.import_result.skipped_files_count,
                    'failed_files_count': self.import_result.failed_files_count,
                    'total_points': self.import_result.total_points,
                }
                if self.import_result else None
            ),
            'comparison_result': (
                self.comparison_result.to_dict()
                if self.comparison_result else None
            ),
            'calibration_result': (
                {
                    'triggered': self.calibration_result.triggered,
                    'trigger_reasons': self.calibration_result.trigger_reasons,
                    'duration_sec': self.calibration_result.duration_sec,
                    'output_dir': self.calibration_result.output_dir,
                }
                if self.calibration_result else None
            ),
            'pw_result': (
                self.pw_result.to_dict() if self.pw_result else None
            ),
        }
