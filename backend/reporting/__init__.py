# -*- coding: utf-8 -*-
"""
Tapeout 签核报告生成模块

汇总一次完整 RET 流程的各类指标，自动生成 PDF/HTML 签核报告，
包含关键截图与参数表，满足课题组文档化与评审需求。

模块结构：
- schemas.py: 数据模型定义
- collector.py: 数据收集与聚合
- figures.py: 图表生成
- html_generator.py: HTML 报告生成
- pdf_generator.py: PDF 报告生成
- api.py: 高层 API
"""

__version__ = "0.1.0"
__all__ = [
    # Schemas
    'TapeoutSignoffReport',
    'ReportStatus',
    'RETStage',
    'EPEMetrics',
    'CDMetrics',
    'ILSNILSMetrics',
    'MaskComplexityMetrics',
    'MEEFMetrics',
    'PWMetrics',
    'MRCViolationSummary',
    'MetrologyConsistencyMetrics',
    'StageMetrics',
    'ReportFigure',
    'ParameterTable',

    # Collector
    'ReportDataCollector',
    'create_report_from_ret_flow',

    # Figures
    'generate_all_report_figures',

    # Generators
    'HTMLReportGenerator',
    'generate_html_report',
    'PDFReportGenerator',
    'generate_pdf_report',

    # API
    'TapeoutSignoffAPI',
    'quick_signoff_report',
]

from .schemas import (
    TapeoutSignoffReport,
    ReportStatus,
    RETStage,
    EPEMetrics,
    CDMetrics,
    ILSNILSMetrics,
    MaskComplexityMetrics,
    MEEFMetrics,
    PWMetrics,
    MRCViolationSummary,
    MetrologyConsistencyMetrics,
    StageMetrics,
    ReportFigure,
    ParameterTable,
)
from .collector import ReportDataCollector, create_report_from_ret_flow
from .figures import generate_all_report_figures
from .html_generator import HTMLReportGenerator, generate_html_report
from .pdf_generator import PDFReportGenerator, generate_pdf_report
from .api import TapeoutSignoffAPI, quick_signoff_report
