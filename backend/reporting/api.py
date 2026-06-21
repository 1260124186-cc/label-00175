# -*- coding: utf-8 -*-
"""
Tapeout 签核报告模块 API

便捷的高层 API，用于快速生成完整的签核报告。
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from .schemas import TapeoutSignoffReport
from .collector import ReportDataCollector, create_report_from_ret_flow
from .figures import generate_all_report_figures
from .html_generator import generate_html_report, HTMLReportGenerator
from .pdf_generator import generate_pdf_report, PDFReportGenerator

logger = logging.getLogger(__name__)


class TapeoutSignoffAPI:
    """
    Tapeout 签核报告高层 API

    提供一站式的报告生成流程：数据收集 → 图表生成 → HTML/PDF 输出。
    """

    def __init__(
        self,
        pixel_size: float = 1.0,
        threshold: float = 0.5,
    ):
        """
        初始化 API

        Args:
            pixel_size: 像素尺寸 (nm)
            threshold: 灰度阈值
        """
        self.pixel_size = pixel_size
        self.threshold = threshold
        self.collector = ReportDataCollector(pixel_size, threshold)
        self._report: Optional[TapeoutSignoffReport] = None

    @property
    def report(self) -> Optional[TapeoutSignoffReport]:
        """获取当前构建的报告对象"""
        return self._report

    def set_basic_info(
        self,
        project_name: Optional[str] = None,
        design_name: Optional[str] = None,
        technology_node: Optional[str] = None,
        ret_flow: Optional[str] = None,
        title: Optional[str] = None,
    ):
        """
        设置报告基本信息

        Args:
            project_name: 项目名称
            design_name: 设计名称
            technology_node: 工艺节点
            ret_flow: RET 流程名称
            title: 报告标题
        """
        self.collector.set_basic_info(
            project_name=project_name if project_name is not None else "",
            design_name=design_name if design_name is not None else "",
            technology_node=technology_node if technology_node is not None else "",
            ret_flow=ret_flow if ret_flow is not None else "",
            title=title if title is not None else "Tapeout 签核报告",
        )

    def collect_initial(self, **kwargs):
        """收集初始阶段指标"""
        self.collector.collect_initial(**kwargs)

    def collect_final(self, **kwargs):
        """收集最终阶段指标"""
        self.collector.collect_final(**kwargs)

    def collect_process_window(self, process_window: Any):
        """收集工艺窗口数据"""
        self.collector.collect_process_window(process_window)

    def collect_mrc(self, mrc_result: Any):
        """收集 MRC 结果"""
        self.collector.collect_mrc(mrc_result)

    def collect_metrology(self, metrology_data: Any):
        """收集计量一致性数据"""
        self.collector.collect_metrology(metrology_data)

    def add_parameter_table(
        self,
        title: str,
        headers: List[str],
        rows: List[List[Any]],
        section: str = "general",
    ):
        """
        添加参数表格

        Args:
            title: 表格标题
            headers: 表头列表
            rows: 数据行
            section: 所属分类
        """
        self.collector.add_parameter_table(title, headers, rows, section)

    def add_default_parameter_tables(self):
        """添加默认参数表"""
        self.collector.add_default_parameter_tables()

    def build_report(self) -> TapeoutSignoffReport:
        """
        构建报告对象

        Returns:
            TapeoutSignoffReport 对象
        """
        self._report = self.collector.build_report()
        return self._report

    def generate_figures(
        self,
        output_dir: Union[str, Path],
        mask_initial: Optional[Any] = None,
        mask_final: Optional[Any] = None,
        wafer_initial: Optional[Any] = None,
        wafer_final: Optional[Any] = None,
        target: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        生成所有报告图表

        Args:
            output_dir: 输出目录
            mask_initial: 初始掩模
            mask_final: 最终掩模
            wafer_initial: 初始晶圆图像
            wafer_final: 最终晶圆图像
            target: 目标图形

        Returns:
            图表路径字典
        """
        if self._report is None:
            self.build_report()

        fig_paths = generate_all_report_figures(
            report=self._report,
            output_dir=output_dir,
            initial_mask=mask_initial,
            final_mask=mask_final,
            initial_wafer=wafer_initial,
            final_wafer=wafer_final,
            target=target,
        )

        return fig_paths

    def generate_html(
        self,
        output_path: Union[str, Path],
        embed_images: bool = True,
    ) -> Path:
        """
        生成 HTML 报告

        Args:
            output_path: 输出文件路径
            embed_images: 是否将图片嵌入为 base64

        Returns:
            输出文件路径
        """
        if self._report is None:
            self.build_report()

        path = generate_html_report(
            self._report,
            output_path=output_path,
            embed_images=embed_images,
        )
        logger.info(f"HTML 报告生成完成: {path}")
        return path

    def generate_pdf(
        self,
        output_path: Union[str, Path],
    ) -> Path:
        """
        生成 PDF 报告

        Args:
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        if self._report is None:
            self.build_report()

        path = generate_pdf_report(self._report, output_path)
        logger.info(f"PDF 报告生成完成: {path}")
        return path

    def generate_all(
        self,
        output_dir: Union[str, Path],
        generate_figures: bool = True,
        generate_html: bool = True,
        generate_pdf: bool = True,
        generate_json: bool = True,
        mask_initial: Optional[Any] = None,
        mask_final: Optional[Any] = None,
        wafer_initial: Optional[Any] = None,
        wafer_final: Optional[Any] = None,
        target: Optional[Any] = None,
    ) -> Dict[str, Union[str, Path]]:
        """
        一站式生成所有报告输出

        Args:
            output_dir: 输出目录
            generate_figures: 是否生成图表
            generate_html: 是否生成 HTML 报告
            generate_pdf: 是否生成 PDF 报告
            generate_json: 是否生成 JSON 数据
            mask_initial: 初始掩模（用于图表）
            mask_final: 最终掩模（用于图表）
            wafer_initial: 初始晶圆图像（用于图表）
            wafer_final: 最终晶圆图像（用于图表）
            target: 目标图形（用于图表）

        Returns:
            所有生成文件的路径字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._report is None:
            self.build_report()

        results: Dict[str, Union[str, Path]] = {}
        results['report_id'] = self._report.report_id

        # 生成图表
        if generate_figures:
            fig_paths = self.generate_figures(
                output_dir=output_dir / "figures",
                mask_initial=mask_initial,
                mask_final=mask_final,
                wafer_initial=wafer_initial,
                wafer_final=wafer_final,
                target=target,
            )
            results['figures'] = fig_paths

        # 生成 HTML
        if generate_html:
            html_path = output_dir / "tapeout_signoff_report.html"
            self.generate_html(html_path, embed_images=True)
            results['html'] = html_path

        # 生成 PDF
        if generate_pdf:
            pdf_path = output_dir / "tapeout_signoff_report.pdf"
            self.generate_pdf(pdf_path)
            results['pdf'] = pdf_path

        # 生成 JSON
        if generate_json:
            json_path = output_dir / "tapeout_signoff_data.json"
            self._report.save_json(json_path)
            results['json'] = json_path

        logger.info(f"所有报告生成完成，输出目录: {output_dir}")
        return results


def quick_signoff_report(
    output_dir: Union[str, Path],
    initial_metrics: Optional[Dict[str, Any]] = None,
    final_metrics: Optional[Dict[str, Any]] = None,
    process_window: Optional[Any] = None,
    mrc_result: Optional[Any] = None,
    metrology_data: Optional[Any] = None,
    project_name: Optional[str] = None,
    design_name: Optional[str] = None,
    technology_node: Optional[str] = None,
    ret_flow: Optional[str] = None,
    **kwargs,
) -> Dict[str, Union[str, Path]]:
    """
    快速生成签核报告（便捷函数）

    提供最简单的 API，一键生成完整的签核报告。

    Args:
        output_dir: 输出目录
        initial_metrics: 初始阶段指标字典
        final_metrics: 最终阶段指标字典
        process_window: 工艺窗口数据
        mrc_result: MRC 结果
        metrology_data: 计量一致性数据
        project_name: 项目名称
        design_name: 设计名称
        technology_node: 工艺节点
        ret_flow: RET 流程名称

    Returns:
        生成文件路径字典
    """
    api = TapeoutSignoffAPI()

    api.set_basic_info(
        project_name=project_name,
        design_name=design_name,
        technology_node=technology_node,
        ret_flow=ret_flow,
    )

    if initial_metrics:
        api.collect_initial(**initial_metrics)

    if final_metrics:
        api.collect_final(**final_metrics)

    if process_window is not None:
        api.collect_process_window(process_window)

    if mrc_result is not None:
        api.collect_mrc(mrc_result)

    if metrology_data is not None:
        api.collect_metrology(metrology_data)

    api.add_default_parameter_tables()

    return api.generate_all(output_dir, **kwargs)


__all__ = [
    'TapeoutSignoffAPI',
    'quick_signoff_report',
]
