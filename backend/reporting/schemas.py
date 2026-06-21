# -*- coding: utf-8 -*-
"""
Tapeout 签核报告数据模型

定义完整 RET 流程签核报告的数据结构，包括：
1. 初始/最终 EPE 指标
2. 工艺窗口 (PW) 面积
3. MEEF (掩模误差增强因子)
4. 掩模复杂度
5. MRC 违规统计
6. 计量一致性
7. 关键截图与参数表
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReportStatus(Enum):
    """报告状态"""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class RETStage(Enum):
    """RET 流程阶段"""
    INITIAL = "initial"
    OPC = "opc"
    ILT = "ilt"
    SMO = "smo"
    FINAL = "final"


@dataclass
class EPEMetrics:
    """EPE (Edge Placement Error) 指标"""
    epe_mean_nm: float = 0.0
    epe_max_nm: float = 0.0
    epe_min_nm: float = 0.0
    epe_std_nm: float = 0.0
    epe_median_nm: float = 0.0
    n_valid_edges: int = 0
    pixel_size_nm: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epe_mean_nm': self.epe_mean_nm,
            'epe_max_nm': self.epe_max_nm,
            'epe_min_nm': self.epe_min_nm,
            'epe_std_nm': self.epe_std_nm,
            'epe_median_nm': self.epe_median_nm,
            'n_valid_edges': self.n_valid_edges,
        }


@dataclass
class CDMetrics:
    """CD (Critical Dimension) 指标"""
    cd_mean_nm: float = 0.0
    cd_min_nm: float = 0.0
    cd_max_nm: float = 0.0
    cd_std_nm: float = 0.0
    cd_target_nm: float = 0.0
    cd_error_mean_nm: float = 0.0
    cd_error_max_nm: float = 0.0
    cd_error_relative_pct: float = 0.0
    n_features: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cd_mean_nm': self.cd_mean_nm,
            'cd_min_nm': self.cd_min_nm,
            'cd_max_nm': self.cd_max_nm,
            'cd_std_nm': self.cd_std_nm,
            'cd_target_nm': self.cd_target_nm,
            'cd_error_mean_nm': self.cd_error_mean_nm,
            'cd_error_max_nm': self.cd_error_max_nm,
            'cd_error_relative_pct': self.cd_error_relative_pct,
            'n_features': self.n_features,
        }


@dataclass
class ILSNILSMetrics:
    """ILS / NILS 指标"""
    ils_mean: float = 0.0
    ils_min: float = 0.0
    ils_max: float = 0.0
    ils_std: float = 0.0
    nils_mean: float = 0.0
    nils_min: float = 0.0
    nils_max: float = 0.0
    nils_std: float = 0.0
    n_sample_points: int = 0
    n_points: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ils_mean': self.ils_mean,
            'ils_min': self.ils_min,
            'ils_max': self.ils_max,
            'ils_std': self.ils_std,
            'nils_mean': self.nils_mean,
            'nils_min': self.nils_min,
            'nils_max': self.nils_max,
            'nils_std': self.nils_std,
            'n_sample_points': self.n_sample_points,
            'n_points': self.n_points,
        }


@dataclass
class MaskComplexityMetrics:
    """掩模复杂度指标"""
    total_variation: float = 0.0
    tv_isotropic: float = 0.0
    binary_penalty: float = 0.0
    n_edge_pixels: int = 0
    n_vertices: int = 0
    n_vertices_approx: int = 0
    sraf_count: int = 0
    sraf_avg_size_nm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_variation': self.total_variation,
            'tv_isotropic': self.tv_isotropic,
            'binary_penalty': self.binary_penalty,
            'n_edge_pixels': self.n_edge_pixels,
            'n_vertices': self.n_vertices,
            'n_vertices_approx': self.n_vertices_approx,
            'sraf_count': self.sraf_count,
            'sraf_avg_size_nm': self.sraf_avg_size_nm,
        }


@dataclass
class MEEFMetrics:
    """MEEF (Mask Error Enhancement Factor) 指标"""
    meef_mean: float = 0.0
    meef_max: float = 0.0
    meef_min: float = 0.0
    meef_std: float = 0.0
    cd_mask_original_nm: float = 0.0
    cd_wafer_original_nm: float = 0.0
    delta_cd_mask_nm: float = 0.0
    delta_cd_wafer_nm: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'meef_mean': self.meef_mean,
            'meef_max': self.meef_max,
            'meef_min': self.meef_min,
            'meef_std': self.meef_std,
            'cd_mask_original_nm': self.cd_mask_original_nm,
            'cd_wafer_original_nm': self.cd_wafer_original_nm,
            'delta_cd_mask_nm': self.delta_cd_mask_nm,
            'delta_cd_wafer_nm': self.delta_cd_wafer_nm,
            'n_samples': self.n_samples,
        }


@dataclass
class PWMetrics:
    """工艺窗口 (Process Window) 指标"""
    pw_area: float = 0.0
    pw_ratio: float = 0.0
    n_passing: int = 0
    n_total: int = 0
    center_focus_nm: float = 0.0
    center_dose: float = 0.0
    best_focus_nm: float = 0.0
    best_dose: float = 0.0
    best_cd_error_nm: float = 0.0
    focus_min_nm: float = 0.0
    focus_max_nm: float = 0.0
    dose_min: float = 0.0
    dose_max: float = 0.0
    depth_of_focus_nm: float = 0.0
    exposure_latitude_pct: float = 0.0
    ellipse_area: float = 0.0
    rect_area: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pw_area': self.pw_area,
            'pw_ratio': self.pw_ratio,
            'n_passing': self.n_passing,
            'n_total': self.n_total,
            'center_focus_nm': self.center_focus_nm,
            'center_dose': self.center_dose,
            'best_focus_nm': self.best_focus_nm,
            'best_dose': self.best_dose,
            'best_cd_error_nm': self.best_cd_error_nm,
            'focus_min_nm': self.focus_min_nm,
            'focus_max_nm': self.focus_max_nm,
            'dose_min': self.dose_min,
            'dose_max': self.dose_max,
            'depth_of_focus_nm': self.depth_of_focus_nm,
            'exposure_latitude_pct': self.exposure_latitude_pct,
            'ellipse_area': self.ellipse_area,
            'rect_area': self.rect_area,
        }


@dataclass
class MRCViolationSummary:
    """MRC 违规汇总"""
    total_violations: int = 0
    fatal_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    passed: bool = True
    violations_by_rule: Dict[str, int] = field(default_factory=dict)
    top_violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_violations': self.total_violations,
            'fatal_count': self.fatal_count,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'info_count': self.info_count,
            'passed': self.passed,
            'violations_by_rule': self.violations_by_rule,
            'top_violations': self.top_violations,
        }


@dataclass
class MetrologyConsistencyMetrics:
    """计量一致性指标"""
    m2t_mean_nm: float = 0.0
    m2t_pct: float = 0.0
    uniformity_3sigma_pct: float = 0.0
    uniformity_range_pct: float = 0.0
    linearity_r_squared: float = 0.0
    linearity_slope: float = 0.0
    linearity_max_deviation_nm: float = 0.0
    grr_pct: float = 0.0
    grr_ndc: float = 0.0
    cp: float = 0.0
    cpk: float = 0.0
    pass_rate_pct: float = 0.0
    n_measurements: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'm2t_mean_nm': self.m2t_mean_nm,
            'm2t_pct': self.m2t_pct,
            'uniformity_3sigma_pct': self.uniformity_3sigma_pct,
            'uniformity_range_pct': self.uniformity_range_pct,
            'linearity_r_squared': self.linearity_r_squared,
            'linearity_slope': self.linearity_slope,
            'linearity_max_deviation_nm': self.linearity_max_deviation_nm,
            'grr_pct': self.grr_pct,
            'grr_ndc': self.grr_ndc,
            'cp': self.cp,
            'cpk': self.cpk,
            'pass_rate_pct': self.pass_rate_pct,
            'n_measurements': self.n_measurements,
        }


@dataclass
class StageMetrics:
    """单个 RET 阶段的指标汇总"""
    stage_name: str
    epe: EPEMetrics = field(default_factory=EPEMetrics)
    cd: CDMetrics = field(default_factory=CDMetrics)
    ils_nils: ILSNILSMetrics = field(default_factory=ILSNILSMetrics)
    mask_complexity: MaskComplexityMetrics = field(default_factory=MaskComplexityMetrics)
    meef: MEEFMetrics = field(default_factory=MEEFMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stage_name': self.stage_name,
            'epe': self.epe.to_dict(),
            'cd': self.cd.to_dict(),
            'ils_nils': self.ils_nils.to_dict(),
            'mask_complexity': self.mask_complexity.to_dict(),
            'meef': self.meef.to_dict(),
        }


@dataclass
class ReportFigure:
    """报告中的图表/截图"""
    figure_id: str
    title: str
    caption: str
    file_path: str
    figure_type: str = "image"

    def to_dict(self) -> Dict[str, str]:
        return {
            'figure_id': self.figure_id,
            'title': self.title,
            'caption': self.caption,
            'file_path': self.file_path,
            'figure_type': self.figure_type,
        }


@dataclass
class ParameterTable:
    """参数表格"""
    table_id: str
    title: str
    headers: List[str]
    rows: List[List[Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'table_id': self.table_id,
            'title': self.title,
            'headers': self.headers,
            'rows': self.rows,
        }


@dataclass
class TapeoutSignoffReport:
    """Tapeout 签核完整报告"""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    status: ReportStatus = ReportStatus.PENDING
    title: str = "Tapeout 签核报告"
    project_name: str = ""
    design_name: str = ""
    technology_node: str = ""
    ret_flow: str = ""

    initial_metrics: StageMetrics = field(default_factory=lambda: StageMetrics(stage_name="initial"))
    final_metrics: StageMetrics = field(default_factory=lambda: StageMetrics(stage_name="final"))
    process_window: PWMetrics = field(default_factory=PWMetrics)
    mrc_violations: MRCViolationSummary = field(default_factory=MRCViolationSummary)
    metrology: MetrologyConsistencyMetrics = field(default_factory=MetrologyConsistencyMetrics)

    figures: List[ReportFigure] = field(default_factory=list)
    parameter_tables: List[ParameterTable] = field(default_factory=list)

    duration_sec: float = 0.0
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'title': self.title,
            'project_name': self.project_name,
            'design_name': self.design_name,
            'technology_node': self.technology_node,
            'ret_flow': self.ret_flow,
            'initial_metrics': self.initial_metrics.to_dict(),
            'final_metrics': self.final_metrics.to_dict(),
            'process_window': self.process_window.to_dict(),
            'mrc_violations': self.mrc_violations.to_dict(),
            'metrology': self.metrology.to_dict(),
            'figures': [f.to_dict() for f in self.figures],
            'parameter_tables': [t.to_dict() for t in self.parameter_tables],
            'duration_sec': self.duration_sec,
            'summary_text': self.summary_text,
        }

    def add_figure(self, figure: ReportFigure) -> None:
        self.figures.append(figure)

    def add_parameter_table(self, table: ParameterTable) -> None:
        self.parameter_tables.append(table)

    def generate_summary(self) -> str:
        """生成文本摘要"""
        lines = []
        lines.append(f"=== {self.title} ===")
        lines.append(f"报告ID: {self.report_id}")
        lines.append(f"项目: {self.project_name} | 设计: {self.design_name}")
        lines.append(f"工艺节点: {self.technology_node} | RET流程: {self.ret_flow}")
        lines.append("")

        lines.append("--- EPE 对比 (初始 → 最终) ---")
        init_epe = self.initial_metrics.epe
        final_epe = self.final_metrics.epe
        epe_improvement = ((init_epe.epe_mean_nm - final_epe.epe_mean_nm) / init_epe.epe_mean_nm * 100.0
                           if init_epe.epe_mean_nm > 0 else 0.0)
        lines.append(f"  平均EPE: {init_epe.epe_mean_nm:.2f} nm → {final_epe.epe_mean_nm:.2f} nm "
                     f"(改善 {epe_improvement:+.1f}%)")
        lines.append(f"  最大EPE: {init_epe.epe_max_nm:.2f} nm → {final_epe.epe_max_nm:.2f} nm")
        lines.append("")

        lines.append("--- CD 误差 ---")
        cd = self.final_metrics.cd
        lines.append(f"  平均CD: {cd.cd_mean_nm:.2f} nm (目标: {cd.cd_target_nm:.2f} nm)")
        lines.append(f"  CD误差: {cd.cd_error_mean_nm:+.2f} nm ({cd.cd_error_relative_pct:+.2f}%)")
        lines.append("")

        lines.append("--- 工艺窗口 ---")
        pw = self.process_window
        lines.append(f"  PW面积: {pw.pw_area:.2f} nm·dose ({pw.pw_ratio*100:.1f}%)")
        lines.append(f"  焦深 (DOF): {pw.depth_of_focus_nm:.1f} nm")
        lines.append(f"  曝光宽容度 (EL): {pw.exposure_latitude_pct:.2f}%")
        lines.append("")

        lines.append("--- 掩模复杂度 ---")
        mc = self.final_metrics.mask_complexity
        lines.append(f"  总变差 (TV): {mc.total_variation:.2f}")
        lines.append(f"  二值化惩罚: {mc.binary_penalty:.4f}")
        lines.append("")

        lines.append("--- MEEF ---")
        meef = self.final_metrics.meef
        lines.append(f"  MEEF: {meef.meef_mean:.2f}")
        lines.append("")

        lines.append("--- MRC 违规 ---")
        mrc = self.mrc_violations
        mrc_status = "通过" if mrc.passed else "未通过"
        lines.append(f"  状态: {mrc_status}")
        lines.append(f"  总违规数: {mrc.total_violations}")
        lines.append(f"    致命: {mrc.fatal_count}, 错误: {mrc.error_count}, "
                     f"警告: {mrc.warning_count}, 信息: {mrc.info_count}")
        lines.append("")

        lines.append("--- 计量一致性 ---")
        met = self.metrology
        lines.append(f"  Mean-to-Target: {met.m2t_mean_nm:+.2f} nm ({met.m2t_pct:+.2f}%)")
        lines.append(f"  均匀性 (3σ): {met.uniformity_3sigma_pct:.2f}%")
        lines.append(f"  Cpk: {met.cpk:.2f}")
        lines.append(f"  合格率: {met.pass_rate_pct:.1f}%")
        lines.append("")

        self.summary_text = "\n".join(lines)
        return self.summary_text

    def save_json(self, filepath: Union[str, Path]) -> Path:
        """保存报告为 JSON 文件"""
        import json
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"签核报告已保存为 JSON: {filepath}")
        return filepath
