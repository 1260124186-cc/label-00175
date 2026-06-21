# -*- coding: utf-8 -*-
"""
报告数据收集与聚合模块

从 RET 流程的各个阶段收集指标数据，聚合成完整的签核报告。
支持从多种数据源（掩模图像、晶圆图像、MRC结果、PW结果等）
自动计算并填充报告数据结构。
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import time

from .schemas import (
    TapeoutSignoffReport,
    StageMetrics,
    EPEMetrics,
    CDMetrics,
    ILSNILSMetrics,
    MaskComplexityMetrics,
    MEEFMetrics,
    PWMetrics,
    MRCViolationSummary,
    MetrologyConsistencyMetrics,
    ReportFigure,
    ParameterTable,
    ReportStatus,
)

logger = logging.getLogger(__name__)


class ReportDataCollector:
    """
    报告数据收集器

    从各种 RET 流程输出中收集指标，构建完整的签核报告。

    使用方式::

        collector = ReportDataCollector()
        collector.set_basic_info(project_name="Project A", design_name="chip_top")
        collector.collect_initial(initial_mask, target, wafer_initial)
        collector.collect_final(final_mask, target, wafer_final, aerial_final)
        collector.collect_mrc(mrc_result)
        collector.collect_process_window(pw_metrics)
        collector.collect_metrology(metrology_report)
        report = collector.build_report()
    """

    def __init__(self, pixel_size: float = 1.0, threshold: float = 0.3):
        """
        初始化数据收集器

        Args:
            pixel_size: 像素尺寸 (nm)
            threshold: 光刻胶阈值
        """
        self.pixel_size = pixel_size
        self.threshold = threshold
        self._report = TapeoutSignoffReport()
        self._report.status = ReportStatus.GENERATING

    def set_basic_info(self,
                       project_name: str = "",
                       design_name: str = "",
                       technology_node: str = "",
                       ret_flow: str = "",
                       title: str = "Tapeout 签核报告") -> None:
        """
        设置报告基本信息

        Args:
            project_name: 项目名称
            design_name: 设计名称
            technology_node: 工艺节点
            ret_flow: RET 流程类型
            title: 报告标题
        """
        self._report.project_name = project_name
        self._report.design_name = design_name
        self._report.technology_node = technology_node
        self._report.ret_flow = ret_flow
        self._report.title = title

    def collect_stage_metrics(self,
                              stage_name: str,
                              mask: np.ndarray,
                              target: np.ndarray,
                              wafer_image: Optional[np.ndarray] = None,
                              aerial_image: Optional[np.ndarray] = None,
                              optical_system: Optional[Any] = None,
                              compute_meef: bool = False) -> StageMetrics:
        """
        收集单个阶段的所有指标

        Args:
            stage_name: 阶段名称
            mask: 掩模图像
            target: 目标图像
            wafer_image: 晶圆图像（二值化后），None则从mask计算
            aerial_image: 空间像（用于 ILS/NILS），可选
            optical_system: 光学系统参数（用于 MEEF 计算）
            compute_meef: 是否计算 MEEF

        Returns:
            StageMetrics 阶段指标
        """
        from core.litho_metrics import (
            compute_epe,
            compute_cd,
            compute_cd_error,
            compute_ils,
            compute_nils,
            compute_meef_simple,
        )
        from core.metrics import total_variation, total_variation_isotropic, manhattan_distance_penalty

        stage = StageMetrics(stage_name=stage_name)

        target_bin = (target >= 0.5).astype(np.float64)

        if wafer_image is not None:
            wafer_bin = (wafer_image >= self.threshold).astype(np.float64)
        else:
            wafer_bin = target_bin.copy()

        epe_result = compute_epe(
            wafer_bin, target_bin,
            pixel_size=self.pixel_size,
            edge_method='morphological'
        )
        stage.epe = EPEMetrics(
            epe_mean_nm=epe_result.get('epe_mean', 0.0),
            epe_max_nm=epe_result.get('epe_max', 0.0),
            epe_std_nm=epe_result.get('epe_std', 0.0),
            epe_median_nm=epe_result.get('epe_median', 0.0),
            pixel_size_nm=self.pixel_size,
        )

        cd_result = compute_cd_error(
            wafer_bin, target_bin,
            direction='both',
            pixel_size=self.pixel_size
        )
        cd_wafer = cd_result.get('cd_wafer', {})
        stage.cd = CDMetrics(
            cd_mean_nm=cd_result.get('cd_wafer_mean', 0.0),
            cd_min_nm=cd_wafer.get('cd_min', 0.0) if isinstance(cd_wafer, dict) else 0.0,
            cd_max_nm=cd_wafer.get('cd_max', 0.0) if isinstance(cd_wafer, dict) else 0.0,
            cd_std_nm=cd_wafer.get('cd_std', 0.0) if isinstance(cd_wafer, dict) else 0.0,
            cd_target_nm=cd_result.get('cd_target_mean', 0.0),
            cd_error_mean_nm=cd_result.get('cd_error_mean', 0.0),
            cd_error_relative_pct=cd_result.get('cd_error_relative', 0.0),
            n_features=cd_wafer.get('n_features', 0) if isinstance(cd_wafer, dict) else 0,
        )

        if aerial_image is not None:
            cd_target_val = stage.cd.cd_target_nm
            ils_result = compute_ils(
                aerial_image,
                threshold=self.threshold,
                pixel_size=self.pixel_size
            )
            nils_result = compute_nils(
                aerial_image,
                cd_target=cd_target_val if cd_target_val > 0 else 100.0,
                threshold=self.threshold,
                pixel_size=self.pixel_size
            )
            stage.ils_nils = ILSNILSMetrics(
                ils_mean=ils_result.get('ils_mean', 0.0),
                ils_min=ils_result.get('ils_min', 0.0),
                ils_max=ils_result.get('ils_max', 0.0),
                nils_mean=nils_result.get('nils_mean', 0.0),
                nils_min=nils_result.get('nils_min', 0.0),
                nils_max=nils_result.get('nils_max', 0.0),
                n_sample_points=ils_result.get('n_sample_points', 0),
            )

        tv_val = total_variation(mask)
        tv_iso_val = total_variation_isotropic(mask)
        bin_pen = manhattan_distance_penalty(mask)

        edge_pixels = self._count_edge_pixels(mask)

        stage.mask_complexity = MaskComplexityMetrics(
            total_variation=tv_val,
            tv_isotropic=tv_iso_val,
            binary_penalty=bin_pen,
            n_edge_pixels=edge_pixels,
        )

        if compute_meef:
            try:
                meef_result = compute_meef_simple(
                    mask,
                    threshold=self.threshold,
                    pixel_size=self.pixel_size
                )
                stage.meef = MEEFMetrics(
                    meef_mean=meef_result.get('meef', 0.0),
                    cd_mask_original_nm=meef_result.get('cd_nominal', 0.0) if isinstance(meef_result.get('cd_nominal'), (int, float)) else 0.0,
                )
            except Exception as e:
                logger.warning(f"MEEF 计算失败: {e}")

        return stage

    def collect_initial(self,
                        initial_mask: np.ndarray,
                        target: np.ndarray,
                        wafer_initial: Optional[np.ndarray] = None,
                        aerial_initial: Optional[np.ndarray] = None) -> None:
        """
        收集初始阶段指标

        Args:
            initial_mask: 初始掩模
            target: 目标图像
            wafer_initial: 初始晶圆图像
            aerial_initial: 初始空间像
        """
        logger.info("收集初始阶段指标...")
        self._report.initial_metrics = self.collect_stage_metrics(
            stage_name="initial",
            mask=initial_mask,
            target=target,
            wafer_image=wafer_initial,
            aerial_image=aerial_initial,
            compute_meef=False,
        )
        logger.info("初始阶段指标收集完成")

    def collect_final(self,
                      final_mask: np.ndarray,
                      target: np.ndarray,
                      wafer_final: Optional[np.ndarray] = None,
                      aerial_final: Optional[np.ndarray] = None,
                      optical_system: Optional[Any] = None) -> None:
        """
        收集最终阶段指标

        Args:
            final_mask: 最终掩模
            target: 目标图像
            wafer_final: 最终晶圆图像
            aerial_final: 最终空间像
            optical_system: 光学系统（用于 MEEF 计算）
        """
        logger.info("收集最终阶段指标...")
        self._report.final_metrics = self.collect_stage_metrics(
            stage_name="final",
            mask=final_mask,
            target=target,
            wafer_image=wafer_final,
            aerial_image=aerial_final,
            optical_system=optical_system,
            compute_meef=True,
        )
        logger.info("最终阶段指标收集完成")

    def collect_process_window(self, pw_data: Union[Dict[str, Any], Any]) -> None:
        """
        收集工艺窗口数据

        Args:
            pw_data: 工艺窗口数据，可以是字典或 PWMetrics 对象
                     或来自 analysis.process_window 的 PWMetrics 对象
        """
        logger.info("收集工艺窗口数据...")

        if hasattr(pw_data, 'pw_area'):
            pw = pw_data
            self._report.process_window = PWMetrics(
                pw_area=getattr(pw, 'pw_area', 0.0),
                pw_ratio=getattr(pw, 'pw_ratio', 0.0),
                n_passing=getattr(pw, 'n_passing', 0),
                n_total=getattr(pw, 'n_total', 0),
                center_focus_nm=getattr(pw, 'center_focus', 0.0),
                center_dose=getattr(pw, 'center_dose', 0.0),
                best_focus_nm=getattr(pw, 'best_focus', 0.0),
                best_dose=getattr(pw, 'best_dose', 0.0),
                best_cd_error_nm=getattr(pw, 'best_cd_error', 0.0),
                focus_min_nm=getattr(pw, 'focus_range', (0.0, 0.0))[0],
                focus_max_nm=getattr(pw, 'focus_range', (0.0, 0.0))[1],
                dose_min=getattr(pw, 'dose_range', (0.0, 0.0))[0],
                dose_max=getattr(pw, 'dose_range', (0.0, 0.0))[1],
                depth_of_focus_nm=getattr(pw, 'depth_of_focus', 0.0),
                exposure_latitude_pct=getattr(pw, 'exposure_latitude', 0.0),
                ellipse_area=getattr(pw.ellipse_approx, 'area', 0.0)
                    if getattr(pw, 'ellipse_approx', None) else 0.0,
                rect_area=getattr(pw.rect_approx, 'area', 0.0)
                    if getattr(pw, 'rect_approx', None) else 0.0,
            )
        elif isinstance(pw_data, dict):
            pw = PWMetrics()
            for key, val in pw_data.items():
                if hasattr(pw, key):
                    setattr(pw, key, val)
            self._report.process_window = pw

        logger.info("工艺窗口数据收集完成")

    def collect_mrc(self, mrc_result: Union[Any, Dict[str, Any]]) -> None:
        """
        收集 MRC 违规检查结果

        Args:
            mrc_result: MRC 检查结果，可以是 MRCCheckResult 对象或字典
        """
        logger.info("收集 MRC 违规数据...")

        summary = MRCViolationSummary()

        if hasattr(mrc_result, 'total_violations'):
            summary.total_violations = getattr(mrc_result, 'total_violations', 0)
            summary.fatal_count = getattr(mrc_result, 'fatal_count', 0)
            summary.error_count = getattr(mrc_result, 'error_count', 0)
            summary.warning_count = getattr(mrc_result, 'warning_count', 0)
            summary.info_count = getattr(mrc_result, 'info_count', 0)
            summary.passed = getattr(mrc_result, 'passed', True)

            if hasattr(mrc_result, 'violations_by_rule'):
                vbr = mrc_result.violations_by_rule()
                summary.violations_by_rule = {
                    str(k.value if hasattr(k, 'value') else k): len(v)
                    for k, v in vbr.items()
                }

            if hasattr(mrc_result, 'violations') and mrc_result.violations:
                violations = mrc_result.violations
                sorted_vios = sorted(
                    violations,
                    key=lambda v: (0 if v.severity.value == 'fatal' else
                                   1 if v.severity.value == 'error' else
                                   2 if v.severity.value == 'warning' else 3,
                                   -v.region.area_pixels)
                )
                for v in sorted_vios[:10]:
                    summary.top_violations.append({
                        'rule_type': v.rule_type.value if hasattr(v.rule_type, 'value') else str(v.rule_type),
                        'severity': v.severity.value if hasattr(v.severity, 'value') else str(v.severity),
                        'message': v.message,
                        'area_nm2': v.violation_area_nm2,
                        'measurement_nm': v.measurement_nm,
                        'threshold_nm': v.threshold_nm,
                    })

        elif isinstance(mrc_result, dict):
            summary.total_violations = mrc_result.get('total_violations', 0)
            summary.fatal_count = mrc_result.get('fatal_count', 0)
            summary.error_count = mrc_result.get('error_count', 0)
            summary.warning_count = mrc_result.get('warning_count', 0)
            summary.info_count = mrc_result.get('info_count', 0)
            summary.passed = mrc_result.get('passed', True)
            summary.violations_by_rule = mrc_result.get('violations_by_rule', {})

        self._report.mrc_violations = summary
        logger.info(f"MRC 违规数据收集完成: {summary.total_violations} 处违规")

    def collect_metrology(self, metrology_data: Union[Any, Dict[str, Any]]) -> None:
        """
        收集计量一致性数据

        Args:
            metrology_data: 计量报告数据，可以是 MetrologyReport 对象或字典
        """
        logger.info("收集计量一致性数据...")

        met = MetrologyConsistencyMetrics()

        if hasattr(metrology_data, 'm2t'):
            met.m2t_mean_nm = getattr(metrology_data, 'm2t', 0.0)
            met.m2t_pct = getattr(metrology_data, 'm2t_pct', 0.0)
            met.pass_rate_pct = getattr(metrology_data, 'pass_rate', 0.0)
            met.n_measurements = getattr(metrology_data, 'n_total', 0)

            if hasattr(metrology_data, 'uniformity'):
                uni = metrology_data.uniformity
                met.uniformity_3sigma_pct = getattr(uni, 'uniformity_3sigma', 0.0)
                met.uniformity_range_pct = getattr(uni, 'uniformity_range', 0.0)

            if hasattr(metrology_data, 'linearity') and metrology_data.linearity:
                lin = metrology_data.linearity
                met.linearity_r_squared = getattr(lin, 'r_squared', 0.0)
                met.linearity_slope = getattr(lin, 'slope', 0.0)
                met.linearity_max_deviation_nm = getattr(lin, 'max_deviation_nm', 0.0)

            if hasattr(metrology_data, 'precision') and metrology_data.precision:
                prec = metrology_data.precision
                met.grr_pct = getattr(prec, 'grr_pct', 0.0)
                met.grr_ndc = getattr(prec, 'ndc', 0.0)

            if hasattr(metrology_data, 'process_capability'):
                pc = metrology_data.process_capability
                met.cp = getattr(pc, 'cp', 0.0)
                met.cpk = getattr(pc, 'cpk', 0.0)

        elif isinstance(metrology_data, dict):
            # 支持多种字段命名方式
            # M2T
            met.m2t_mean_nm = float(metrology_data.get(
                'm2t_mean_nm', metrology_data.get(
                    'm2t_nm', metrology_data.get(
                        'm2t', 0.0
                    )
                )
            ))
            met.m2t_pct = float(metrology_data.get('m2t_pct', 0.0))

            # 合格率
            met.pass_rate_pct = float(metrology_data.get(
                'pass_rate_pct', metrology_data.get(
                    'pass_rate', 0.0
                )
            ))
            met.n_measurements = int(metrology_data.get(
                'n_measurements', metrology_data.get(
                    'n_total', 0
                )
            ))

            # 均匀性 - 支持平级字段或嵌套在 uniformity 中
            if 'uniformity' in metrology_data and isinstance(metrology_data['uniformity'], dict):
                uni = metrology_data['uniformity']
                met.uniformity_3sigma_pct = float(uni.get(
                    'uniformity_3sigma_pct', uni.get(
                        'uniformity_3sigma', 0.0
                    )
                ))
                met.uniformity_range_pct = float(uni.get(
                    'uniformity_range_pct', uni.get(
                        'uniformity_range', 0.0
                    )
                ))
            else:
                met.uniformity_3sigma_pct = float(metrology_data.get(
                    'uniformity_3sigma_pct', metrology_data.get(
                        'uniformity_3sigma', 0.0
                    )
                ))
                met.uniformity_range_pct = float(metrology_data.get(
                    'uniformity_range_pct', metrology_data.get(
                        'uniformity_range', 0.0
                    )
                ))

            # 线性度 - 支持平级字段或嵌套在 linearity 中
            if 'linearity' in metrology_data and isinstance(metrology_data['linearity'], dict):
                lin = metrology_data['linearity']
                met.linearity_r_squared = float(lin.get(
                    'r_squared', lin.get('linearity_r_squared', 0.0)
                ))
                met.linearity_slope = float(lin.get(
                    'slope', lin.get('linearity_slope', 0.0)
                ))
                met.linearity_max_deviation_nm = float(lin.get(
                    'max_deviation_nm', 0.0
                ))
            else:
                met.linearity_r_squared = float(metrology_data.get(
                    'linearity_r_squared', metrology_data.get(
                        'r_squared', 0.0
                    )
                ))
                met.linearity_slope = float(metrology_data.get(
                    'linearity_slope', metrology_data.get(
                        'slope', 0.0
                    )
                ))
                met.linearity_max_deviation_nm = float(metrology_data.get(
                    'linearity_max_deviation_nm', 0.0
                ))

            # 精密度/GRR - 支持平级字段或嵌套在 precision 中
            if 'precision' in metrology_data and isinstance(metrology_data['precision'], dict):
                prec = metrology_data['precision']
                met.grr_pct = float(prec.get('grr_pct', 0.0))
                met.grr_ndc = float(prec.get('ndc', prec.get('grr_ndc', 0.0)))
            else:
                met.grr_pct = float(metrology_data.get('grr_pct', 0.0))
                met.grr_ndc = float(metrology_data.get(
                    'grr_ndc', metrology_data.get('ndc', 0.0)
                ))

            # 工艺能力 - 支持平级字段或嵌套在 process_capability 中
            if 'process_capability' in metrology_data and isinstance(metrology_data['process_capability'], dict):
                pc = metrology_data['process_capability']
                met.cp = float(pc.get('cp', pc.get('Cp', 0.0)))
                met.cpk = float(pc.get('cpk', pc.get('Cpk', 0.0)))
            else:
                met.cp = float(metrology_data.get('cp', 0.0))
                met.cpk = float(metrology_data.get('cpk', 0.0))

        self._report.metrology = met
        logger.info("计量一致性数据收集完成")

    def add_figure(self, figure_id: str, title: str, caption: str,
                   file_path: str, figure_type: str = "image") -> None:
        """
        添加图表到报告

        Args:
            figure_id: 图表ID
            title: 图表标题
            caption: 图表说明
            file_path: 文件路径
            figure_type: 图表类型
        """
        fig = ReportFigure(
            figure_id=figure_id,
            title=title,
            caption=caption,
            file_path=file_path,
            figure_type=figure_type,
        )
        self._report.add_figure(fig)

    def add_parameter_table(self, table_id: str, title: str,
                            headers: List[str], rows: List[List[Any]]) -> None:
        """
        添加参数表格到报告

        Args:
            table_id: 表格ID
            title: 表格标题
            headers: 表头
            rows: 数据行
        """
        table = ParameterTable(
            table_id=table_id,
            title=title,
            headers=headers,
            rows=rows,
        )
        self._report.add_parameter_table(table)

    def add_default_parameter_tables(self, optical_config: Optional[Dict[str, Any]] = None,
                                     ret_config: Optional[Dict[str, Any]] = None) -> None:
        """
        添加默认的参数表格（光学参数、RET 参数等）

        Args:
            optical_config: 光学参数字典
            ret_config: RET 参数字典
        """
        if optical_config:
            rows = [[str(k), str(v)] for k, v in optical_config.items()]
            self.add_parameter_table(
                table_id="optical_params",
                title="光学系统参数",
                headers=["参数名", "值"],
                rows=rows,
            )

        if ret_config:
            rows = [[str(k), str(v)] for k, v in ret_config.items()]
            self.add_parameter_table(
                table_id="ret_params",
                title="RET 优化参数",
                headers=["参数名", "值"],
                rows=rows,
            )

    def build_report(self) -> TapeoutSignoffReport:
        """
        构建完整的签核报告

        Returns:
            TapeoutSignoffReport 完整报告对象
        """
        logger.info("构建完整签核报告...")

        self._report.generate_summary()
        self._report.status = ReportStatus.COMPLETED
        self._report.duration_sec = time.time() - self._report.timestamp

        logger.info(f"报告构建完成，ID: {self._report.report_id}")
        return self._report

    @staticmethod
    def _count_edge_pixels(mask: np.ndarray, threshold: float = 0.5) -> int:
        """
        统计边缘像素数

        Args:
            mask: 掩模图像
            threshold: 二值化阈值

        Returns:
            边缘像素数
        """
        from scipy.ndimage import binary_erosion

        binary = mask >= threshold
        eroded = binary_erosion(binary, structure=np.ones((3, 3), dtype=bool))
        edge = binary & ~eroded
        return int(np.sum(edge))


def create_report_from_ret_flow(
    initial_mask: np.ndarray,
    final_mask: np.ndarray,
    target: np.ndarray,
    wafer_initial: Optional[np.ndarray] = None,
    wafer_final: Optional[np.ndarray] = None,
    aerial_final: Optional[np.ndarray] = None,
    mrc_result: Optional[Any] = None,
    pw_result: Optional[Any] = None,
    metrology_result: Optional[Any] = None,
    pixel_size: float = 1.0,
    threshold: float = 0.3,
    project_name: str = "",
    design_name: str = "",
    technology_node: str = "",
    ret_flow: str = "",
) -> TapeoutSignoffReport:
    """
    便捷函数：从 RET 流程结果快速创建签核报告

    Args:
        initial_mask: 初始掩模
        final_mask: 最终掩模
        target: 目标图像
        wafer_initial: 初始晶圆图像
        wafer_final: 最终晶圆图像
        aerial_final: 最终空间像
        mrc_result: MRC 检查结果
        pw_result: 工艺窗口结果
        metrology_result: 计量一致性结果
        pixel_size: 像素尺寸 (nm)
        threshold: 光刻胶阈值
        project_name: 项目名称
        design_name: 设计名称
        technology_node: 工艺节点
        ret_flow: RET 流程类型

    Returns:
        TapeoutSignoffReport 完整报告对象
    """
    collector = ReportDataCollector(pixel_size=pixel_size, threshold=threshold)

    collector.set_basic_info(
        project_name=project_name,
        design_name=design_name,
        technology_node=technology_node,
        ret_flow=ret_flow,
    )

    collector.collect_initial(
        initial_mask=initial_mask,
        target=target,
        wafer_initial=wafer_initial,
    )

    collector.collect_final(
        final_mask=final_mask,
        target=target,
        wafer_final=wafer_final,
        aerial_final=aerial_final,
    )

    if mrc_result is not None:
        collector.collect_mrc(mrc_result)

    if pw_result is not None:
        collector.collect_process_window(pw_result)

    if metrology_result is not None:
        collector.collect_metrology(metrology_result)

    return collector.build_report()
