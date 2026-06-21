# -*- coding: utf-8 -*-
"""
PDF 签核报告生成器

基于 matplotlib 的 PdfPages 生成多页 PDF 签核报告：
1. 封面页
2. 执行摘要
3. EPE 对比分析
4. CD 分析
5. 工艺窗口分析
6. MEEF 分析
7. 掩模复杂度分析
8. MRC 违规统计
9. 计量一致性评估
10. 关键图表
11. 参数配置表

依赖：matplotlib（已在项目 requirements.txt 中）
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

import numpy as np

from .schemas import TapeoutSignoffReport

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    PdfPages = None
    logger.warning("matplotlib 未安装，PDF 报告生成功能不可用")


# 颜色配置
COLOR_PRIMARY = '#1a237e'
COLOR_SECONDARY = '#0d47a1'
COLOR_ACCENT = '#0288d1'
COLOR_SUCCESS = '#2e7d32'
COLOR_WARNING = '#f57f17'
COLOR_DANGER = '#c62828'
COLOR_TEXT = '#333333'
COLOR_LIGHT_BG = '#f5f7fa'
COLOR_BORDER = '#e0e0e0'


class PDFReportGenerator:
    """
    PDF 签核报告生成器

    使用 matplotlib 的 PdfPages 生成专业的多页 PDF 签核报告。
    """

    def __init__(self, report: TapeoutSignoffReport):
        """
        初始化生成器

        Args:
            report: 签核报告对象
        """
        self.report = report
        self.figsize = (8.27, 11.69)  # A4 纸张尺寸 (英寸)
        self.dpi = 150

    def _add_page_header(self, ax, title: str, page_num: int, total_pages: int):
        """添加页面页眉"""
        # 顶部标题条
        ax.add_patch(plt.Rectangle(
            (0, 0.95), 1, 0.05,
            transform=ax.transAxes,
            color=COLOR_PRIMARY,
            clip_on=False,
        ))

        # 标题
        ax.text(
            0.05, 0.975, title,
            transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='white',
            va='center', ha='left',
        )

        # 页码
        ax.text(
            0.95, 0.975,
            f"{self.report.title}  |  第 {page_num} / {total_pages} 页",
            transform=ax.transAxes,
            fontsize=8, color='white',
            va='center', ha='right',
        )

        # 页脚
        timestamp = datetime.fromtimestamp(self.report.timestamp).strftime('%Y-%m-%d %H:%M')
        ax.text(
            0.5, 0.01,
            f"Tapeout Sign-off Report  |  生成时间: {timestamp}  |  报告ID: {self.report.report_id}",
            transform=ax.transAxes,
            fontsize=7, color='#999',
            va='center', ha='center',
        )

    def _draw_table(
        self,
        ax,
        headers: List[str],
        rows: List[List[Any]],
        x: float,
        y: float,
        width: float,
        height: float,
        col_widths: Optional[List[float]] = None,
    ):
        """
        在指定位置绘制表格

        Args:
            ax: matplotlib axes
            headers: 表头列表
            rows: 数据行列表
            x, y: 左上角位置 (axes坐标)
            width, height: 表格尺寸
            col_widths: 各列宽度比例，None则平均分配
        """
        n_cols = len(headers)
        n_rows = len(rows) + 1  # +1 for header

        if col_widths is None:
            col_widths = [1.0 / n_cols] * n_cols
        else:
            total = sum(col_widths)
            col_widths = [w / total for w in col_widths]

        row_height = height / n_rows

        # 绘制表头
        x_cursor = x
        for i, header in enumerate(headers):
            col_w = width * col_widths[i]
            rect = plt.Rectangle(
                (x_cursor, y - row_height), col_w, row_height,
                transform=ax.transAxes,
                facecolor=COLOR_PRIMARY,
                edgecolor='white',
                linewidth=0.5,
                clip_on=False,
            )
            ax.add_patch(rect)
            ax.text(
                x_cursor + col_w / 2, y - row_height / 2,
                str(header),
                transform=ax.transAxes,
                fontsize=9, fontweight='bold', color='white',
                va='center', ha='center',
            )
            x_cursor += col_w

        # 绘制数据行
        for row_idx, row in enumerate(rows):
            y_pos = y - row_height * (row_idx + 2)
            x_cursor = x
            bg_color = COLOR_LIGHT_BG if row_idx % 2 == 0 else 'white'

            for col_idx, cell in enumerate(row):
                col_w = width * col_widths[col_idx]
                rect = plt.Rectangle(
                    (x_cursor, y_pos - row_height), col_w, row_height,
                    transform=ax.transAxes,
                    facecolor=bg_color,
                    edgecolor=COLOR_BORDER,
                    linewidth=0.5,
                    clip_on=False,
                )
                ax.add_patch(rect)
                ax.text(
                    x_cursor + col_w / 2, y_pos - row_height / 2,
                    str(cell),
                    transform=ax.transAxes,
                    fontsize=8.5, color=COLOR_TEXT,
                    va='center', ha='center',
                )
                x_cursor += col_w

    def _draw_metric_card(
        self,
        ax,
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        value: str,
        unit: str = '',
        color: str = COLOR_PRIMARY,
    ):
        """绘制指标卡片"""
        # 卡片背景
        rect = FancyBboxPatch(
            (x, y - height), width, height,
            transform=ax.transAxes,
            boxstyle="round,pad=0.005",
            facecolor=COLOR_LIGHT_BG,
            edgecolor=color,
            linewidth=2,
            clip_on=False,
        )
        ax.add_patch(rect)

        # 左侧彩色条
        side_bar = plt.Rectangle(
            (x, y - height), 0.01, height,
            transform=ax.transAxes,
            facecolor=color,
            clip_on=False,
        )
        ax.add_patch(side_bar)

        # 标签
        ax.text(
            x + 0.03, y - 0.015,
            label,
            transform=ax.transAxes,
            fontsize=8, color='#666',
            va='top', ha='left',
        )

        # 数值
        ax.text(
            x + width / 2, y - height / 2 - 0.005,
            f"{value} {unit}" if unit else str(value),
            transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=color,
            va='center', ha='center',
        )

    def _create_cover_page(self, pdf: PdfPages):
        """创建封面页"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # 顶部蓝色渐变条
        for i in range(50):
            alpha = 0.015 + i * 0.01
            y = 0.68 + i * 0.005
            ax.axhline(y, color=COLOR_PRIMARY, alpha=min(alpha, 0.5), linewidth=1)

        # 标题
        ax.text(
            0.5, 0.82, self.report.title,
            transform=ax.transAxes,
            fontsize=22, fontweight='bold', color=COLOR_PRIMARY,
            va='center', ha='center',
        )

        # 副标题
        ax.text(
            0.5, 0.77,
            "Resolution Enhancement Technology Sign-off Report",
            transform=ax.transAxes,
            fontsize=11, color='#666',
            va='center', ha='center',
        )

        # 分隔线
        ax.plot([0.2, 0.8], [0.72, 0.72], color=COLOR_ACCENT, linewidth=2,
                transform=ax.transAxes)

        # 基本信息卡片
        info_items = [
            ("项目名称", self.report.project_name or '-'),
            ("设计名称", self.report.design_name or '-'),
            ("工艺节点", self.report.technology_node or '-'),
            ("RET 流程", self.report.ret_flow or '-'),
            ("生成时间", datetime.fromtimestamp(self.report.timestamp).strftime('%Y-%m-%d %H:%M:%S')),
            ("报告ID", self.report.report_id),
        ]

        card_w = 0.35
        card_h = 0.06
        start_x = 0.1
        start_y = 0.62
        gap_x = 0.1
        gap_y = 0.02

        for i, (label, value) in enumerate(info_items):
            col = i % 2
            row = i // 2
            x = start_x + col * (card_w + gap_x)
            y = start_y - row * (card_h + gap_y)

            self._draw_metric_card(
                ax, x, y, card_w, card_h,
                label, value,
                color=COLOR_PRIMARY,
            )

        # MRC 状态
        mrc_passed = self.report.mrc_violations.passed
        status_color = COLOR_SUCCESS if mrc_passed else COLOR_DANGER
        status_text = "✅ MRC 检查通过" if mrc_passed else "❌ MRC 检查未通过"

        ax.add_patch(FancyBboxPatch(
            (0.25, 0.28), 0.5, 0.08,
            transform=ax.transAxes,
            boxstyle="round,pad=0.01",
            facecolor=status_color,
            alpha=0.1,
            edgecolor=status_color,
            linewidth=2,
            clip_on=False,
        ))
        ax.text(
            0.5, 0.32,
            status_text,
            transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=status_color,
            va='center', ha='center',
        )

        # 底部装饰
        ax.plot([0.1, 0.9], [0.1, 0.1], color=COLOR_BORDER, linewidth=1,
                transform=ax.transAxes)
        ax.text(
            0.5, 0.06,
            f"报告 ID: {self.report.report_id}",
            transform=ax.transAxes,
            fontsize=9, color='#999',
            va='center', ha='center',
        )
        ax.text(
            0.5, 0.03,
            "本报告由 RET 签核系统自动生成",
            transform=ax.transAxes,
            fontsize=8, color='#bbb',
            va='center', ha='center',
        )

        pdf.savefig(fig)
        plt.close(fig)

    def _create_summary_page(self, pdf: PdfPages, page_num: int, total_pages: int):
        """创建执行摘要页"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        self._add_page_header(ax, "执行摘要", page_num, total_pages)

        r = self.report
        init_epe = r.initial_metrics.epe
        final_epe = r.final_metrics.epe
        epe_improvement = ((init_epe.epe_mean_nm - final_epe.epe_mean_nm) / init_epe.epe_mean_nm * 100
                           if init_epe.epe_mean_nm > 0 else 0)

        # 6 个关键指标卡片
        cards = [
            ("EPE 改善率", f"{epe_improvement:+.1f}%",
             f"{init_epe.epe_mean_nm:.1f}→{final_epe.epe_mean_nm:.1f} nm",
             COLOR_SUCCESS if epe_improvement > 0 else COLOR_DANGER),
            ("CD 误差均值", f"{r.final_metrics.cd.cd_error_mean_nm:+.2f} nm",
             f"相对误差 {r.final_metrics.cd.cd_error_relative_pct:+.2f}%",
             COLOR_PRIMARY),
            ("工艺窗口占比", f"{r.process_window.pw_ratio*100:.1f}%",
             f"面积 {r.process_window.pw_area:.1f}",
             COLOR_PRIMARY),
            ("MRC 检查", "通过" if r.mrc_violations.passed else "未通过",
             f"{r.mrc_violations.total_violations} 处违规",
             COLOR_SUCCESS if r.mrc_violations.passed else COLOR_DANGER),
            ("工艺能力 Cpk", f"{r.metrology.cpk:.2f}",
             f"合格率 {r.metrology.pass_rate_pct:.1f}%",
             COLOR_SUCCESS if r.metrology.cpk >= 1.33 else COLOR_WARNING if r.metrology.cpk >= 1.0 else COLOR_DANGER),
            ("MEEF 均值", f"{r.final_metrics.meef.meef_mean:.2f}",
             "掩模误差增强因子",
             COLOR_PRIMARY),
        ]

        card_w = 0.28
        card_h = 0.12
        start_x = 0.03
        start_y = 0.86
        gap_x = 0.03
        gap_y = 0.02

        for i, (label, value, sub, color) in enumerate(cards):
            col = i % 3
            row = i // 3
            x = start_x + col * (card_w + gap_x)
            y = start_y - row * (card_h + gap_y)

            self._draw_metric_card(ax, x, y, card_w, card_h, label, value, color=color)
            ax.text(
                x + card_w / 2, y - card_h + 0.01,
                sub,
                transform=ax.transAxes,
                fontsize=7, color='#888',
                va='top', ha='center',
            )

        # EPE 对比表
        ax.text(0.03, 0.55, "EPE 对比分析", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        epe_headers = ["指标", "初始值 (nm)", "最终值 (nm)", "变化量 (nm)", "变化率"]
        epe_rows = [
            ["平均 EPE", f"{init_epe.epe_mean_nm:.4f}", f"{final_epe.epe_mean_nm:.4f}",
             f"{final_epe.epe_mean_nm - init_epe.epe_mean_nm:+.4f}", f"{epe_improvement:+.2f}%"],
            ["最大 EPE", f"{init_epe.epe_max_nm:.4f}", f"{final_epe.epe_max_nm:.4f}",
             f"{final_epe.epe_max_nm - init_epe.epe_max_nm:+.4f}",
             f"{((final_epe.epe_max_nm - init_epe.epe_max_nm)/init_epe.epe_max_nm*100) if init_epe.epe_max_nm > 0 else 0:+.2f}%"],
            ["EPE 标准差", f"{init_epe.epe_std_nm:.4f}", f"{final_epe.epe_std_nm:.4f}",
             f"{final_epe.epe_std_nm - init_epe.epe_std_nm:+.4f}",
             f"{((final_epe.epe_std_nm - init_epe.epe_std_nm)/init_epe.epe_std_nm*100) if init_epe.epe_std_nm > 0 else 0:+.2f}%"],
        ]

        self._draw_table(
            ax, epe_headers, epe_rows,
            x=0.03, y=0.52, width=0.94, height=0.15,
            col_widths=[2, 2, 2, 2, 1.5],
        )

        # 工艺窗口参数
        ax.text(0.03, 0.32, "关键工艺参数", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        pw = r.process_window
        pw_headers = ["参数", "值", "单位", "说明"]
        pw_rows = [
            ["焦深 (DOF)", f"{pw.depth_of_focus_nm:.1f}", "nm", "可打印的离焦量范围"],
            ["曝光宽容度", f"{pw.exposure_latitude_pct:.2f}", "%", "剂量容许范围"],
            ["最佳 Focus", f"{pw.best_focus_nm:.2f}", "nm", "CD 误差最小的离焦量"],
            ["最佳 Dose", f"{pw.best_dose:.4f}", "-", "CD 误差最小的剂量"],
            ["最佳 CD 误差", f"{pw.best_cd_error_nm:+.3f}", "nm", "最佳条件下的 CD 误差"],
        ]

        self._draw_table(
            ax, pw_headers, pw_rows,
            x=0.03, y=0.29, width=0.94, height=0.22,
            col_widths=[2, 2, 1, 4],
        )

        pdf.savefig(fig)
        plt.close(fig)

    def _create_epe_page(self, pdf: PdfPages, page_num: int, total_pages: int):
        """创建 EPE 分析页"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        self._add_page_header(ax, "EPE (边缘放置误差) 分析", page_num, total_pages)

        init = self.report.initial_metrics.epe
        final = self.report.final_metrics.epe
        epe_improvement = ((init.epe_mean_nm - final.epe_mean_nm) / init.epe_mean_nm * 100
                           if init.epe_mean_nm > 0 else 0)

        # 大数值对比
        ax.text(0.15, 0.82, "初始平均 EPE", transform=ax.transAxes,
                fontsize=11, color='#666', va='center', ha='center')
        ax.text(0.15, 0.75, f"{init.epe_mean_nm:.2f}", transform=ax.transAxes,
                fontsize=28, fontweight='bold', color='#666', va='center', ha='center')
        ax.text(0.15, 0.71, "nm", transform=ax.transAxes,
                fontsize=12, color='#999', va='center', ha='center')

        ax.text(0.5, 0.78, "→", transform=ax.transAxes,
                fontsize=24, color=COLOR_SUCCESS if epe_improvement > 0 else COLOR_DANGER,
                va='center', ha='center')

        ax.text(0.85, 0.82, "最终平均 EPE", transform=ax.transAxes,
                fontsize=11, color=COLOR_PRIMARY, va='center', ha='center')
        ax.text(0.85, 0.75, f"{final.epe_mean_nm:.2f}", transform=ax.transAxes,
                fontsize=28, fontweight='bold', color=COLOR_PRIMARY, va='center', ha='center')
        ax.text(0.85, 0.71, "nm", transform=ax.transAxes,
                fontsize=12, color='#999', va='center', ha='center')

        # 改善率标签
        improve_color = COLOR_SUCCESS if epe_improvement > 0 else COLOR_DANGER
        ax.add_patch(FancyBboxPatch(
            (0.35, 0.66), 0.3, 0.05,
            transform=ax.transAxes,
            boxstyle="round,pad=0.005",
            facecolor=improve_color,
            alpha=0.1,
            edgecolor=improve_color,
            linewidth=1.5,
            clip_on=False,
        ))
        ax.text(
            0.5, 0.685,
            f"改善率: {epe_improvement:+.2f}%",
            transform=ax.transAxes,
            fontsize=12, fontweight='bold', color=improve_color,
            va='center', ha='center',
        )

        # 详细数据表
        ax.text(0.03, 0.60, "EPE 指标对比详情", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        headers = ["指标", "初始值 (nm)", "最终值 (nm)", "变化量 (nm)", "变化率"]
        rows = [
            ["平均 EPE", f"{init.epe_mean_nm:.4f}", f"{final.epe_mean_nm:.4f}",
             f"{final.epe_mean_nm - init.epe_mean_nm:+.4f}", f"{epe_improvement:+.2f}%"],
            ["最大 EPE", f"{init.epe_max_nm:.4f}", f"{final.epe_max_nm:.4f}",
             f"{final.epe_max_nm - init.epe_max_nm:+.4f}",
             f"{((final.epe_max_nm - init.epe_max_nm)/init.epe_max_nm*100) if init.epe_max_nm > 0 else 0:+.2f}%"],
            ["最小 EPE", f"{init.epe_min_nm:.4f}", f"{final.epe_min_nm:.4f}",
             f"{final.epe_min_nm - init.epe_min_nm:+.4f}",
             f"{((final.epe_min_nm - init.epe_min_nm)/init.epe_min_nm*100) if init.epe_min_nm > 0 else 0:+.2f}%"],
            ["EPE 标准差", f"{init.epe_std_nm:.4f}", f"{final.epe_std_nm:.4f}",
             f"{final.epe_std_nm - init.epe_std_nm:+.4f}",
             f"{((final.epe_std_nm - init.epe_std_nm)/init.epe_std_nm*100) if init.epe_std_nm > 0 else 0:+.2f}%"],
            ["EPE 中位数", f"{init.epe_median_nm:.4f}", f"{final.epe_median_nm:.4f}",
             f"{final.epe_median_nm - init.epe_median_nm:+.4f}",
             f"{((final.epe_median_nm - init.epe_median_nm)/init.epe_median_nm*100) if init.epe_median_nm > 0 else 0:+.2f}%"],
            ["有效边缘点数", f"{init.n_valid_edges:,}", f"{final.n_valid_edges:,}",
             f"{final.n_valid_edges - init.n_valid_edges:+,}", "-"],
        ]

        self._draw_table(
            ax, headers, rows,
            x=0.03, y=0.57, width=0.94, height=0.35,
            col_widths=[2, 2, 2, 2, 1.5],
        )

        # ILS/NILS 部分
        ax.text(0.03, 0.16, "ILS / NILS 指标（最终阶段）", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        nils = self.report.final_metrics.ils_nils
        nils_headers = ["指标", "平均值", "最小值", "最大值", "标准差"]
        nils_rows = [
            ["ILS (nm⁻¹)", f"{nils.ils_mean:.4f}", f"{nils.ils_min:.4f}",
             f"{nils.ils_max:.4f}", f"{nils.ils_std:.4f}"],
            ["NILS", f"{nils.nils_mean:.3f}", f"{nils.nils_min:.3f}",
             f"{nils.nils_max:.3f}", f"{nils.nils_std:.3f}"],
            ["评估点数", f"{nils.n_points:,}", "-", "-", "-"],
        ]

        self._draw_table(
            ax, nils_headers, nils_rows,
            x=0.03, y=0.13, width=0.94, height=0.15,
            col_widths=[2, 2, 2, 2, 2],
        )

        pdf.savefig(fig)
        plt.close(fig)

    def _create_cd_page(self, pdf: PdfPages, page_num: int, total_pages: int):
        """创建 CD 分析页"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        self._add_page_header(ax, "CD (关键尺寸) 分析", page_num, total_pages)

        cd = self.report.final_metrics.cd
        init_cd = self.report.initial_metrics.cd

        # 三个大卡片
        self._draw_metric_card(ax, 0.03, 0.85, 0.28, 0.1,
                               "目标 CD", f"{cd.cd_target_nm:.2f}", "nm", COLOR_PRIMARY)
        self._draw_metric_card(ax, 0.35, 0.85, 0.28, 0.1,
                               "最终平均 CD", f"{cd.cd_mean_nm:.2f}", "nm", COLOR_SECONDARY)
        self._draw_metric_card(ax, 0.67, 0.85, 0.28, 0.1,
                               "CD 误差均值", f"{cd.cd_error_mean_nm:+.2f}", "nm",
                               COLOR_SUCCESS if abs(cd.cd_error_relative_pct) < 3 else COLOR_WARNING)

        # 详细对比表
        ax.text(0.03, 0.70, "CD 指标对比详情", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        headers = ["指标", "初始值", "最终值", "单位"]
        rows = [
            ["平均 CD", f"{init_cd.cd_mean_nm:.2f}", f"{cd.cd_mean_nm:.2f}", "nm"],
            ["最小 CD", f"{init_cd.cd_min_nm:.2f}", f"{cd.cd_min_nm:.2f}", "nm"],
            ["最大 CD", f"{init_cd.cd_max_nm:.2f}", f"{cd.cd_max_nm:.2f}", "nm"],
            ["CD 范围 (Max-Min)", f"{init_cd.cd_max_nm - init_cd.cd_min_nm:.2f}",
             f"{cd.cd_max_nm - cd.cd_min_nm:.2f}", "nm"],
            ["CD 标准差", f"{init_cd.cd_std_nm:.2f}", f"{cd.cd_std_nm:.2f}", "nm"],
            ["CD 误差均值", f"{init_cd.cd_error_mean_nm:+.2f}", f"{cd.cd_error_mean_nm:+.2f}", "nm"],
            ["CD 误差最大值", f"{init_cd.cd_error_max_nm:+.2f}", f"{cd.cd_error_max_nm:+.2f}", "nm"],
            ["CD 相对误差", f"{init_cd.cd_error_relative_pct:+.2f}", f"{cd.cd_error_relative_pct:+.2f}", "%"],
            ["特征数量", f"{init_cd.n_features}", f"{cd.n_features}", "个"],
        ]

        self._draw_table(
            ax, headers, rows,
            x=0.03, y=0.67, width=0.94, height=0.45,
            col_widths=[2.5, 2, 2, 1],
        )

        pdf.savefig(fig)
        plt.close(fig)

    def _create_pw_page(self, pdf: PdfPages, page_num: int, total_pages: int):
        """创建工艺窗口分析页"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        self._add_page_header(ax, "工艺窗口 (PW) 分析", page_num, total_pages)

        pw = self.report.process_window

        # 4 个关键指标
        self._draw_metric_card(ax, 0.03, 0.86, 0.22, 0.1,
                               "PW 面积", f"{pw.pw_area:.2f}", "nm·dose", COLOR_PRIMARY)
        self._draw_metric_card(ax, 0.27, 0.86, 0.22, 0.1,
                               "PW 占比", f"{pw.pw_ratio*100:.1f}", "%", COLOR_SECONDARY)
        self._draw_metric_card(ax, 0.51, 0.86, 0.22, 0.1,
                               "焦深 (DOF)", f"{pw.depth_of_focus_nm:.1f}", "nm", COLOR_ACCENT)
        self._draw_metric_card(ax, 0.75, 0.86, 0.22, 0.1,
                               "曝光宽容度", f"{pw.exposure_latitude_pct:.2f}", "%", COLOR_SUCCESS)

        # 详细参数表
        ax.text(0.03, 0.70, "工艺窗口详细参数", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        headers = ["参数", "值", "单位", "说明"]
        rows = [
            ["可打印条件数", f"{pw.n_passing} / {pw.n_total}", "-", "通过 CD 容差的工艺条件数"],
            ["PW 面积 (精确)", f"{pw.pw_area:.2f}", "nm·dose", "真实可打印区域面积"],
            ["PW 面积 (椭圆近似)", f"{pw.ellipse_area:.2f}", "nm·dose", "协方差椭圆拟合面积"],
            ["PW 面积 (矩形近似)", f"{pw.rect_area:.2f}", "nm·dose", "外接矩形面积"],
            ["PW 中心 (Focus)", f"{pw.center_focus_nm:.2f}", "nm", "可打印区域中心离焦量"],
            ["PW 中心 (Dose)", f"{pw.center_dose:.4f}", "-", "可打印区域中心剂量"],
            ["最佳点 (Focus)", f"{pw.best_focus_nm:.2f}", "nm", "CD 误差最小的离焦量"],
            ["最佳点 (Dose)", f"{pw.best_dose:.4f}", "-", "CD 误差最小的剂量"],
            ["最佳点 CD 误差", f"{pw.best_cd_error_nm:+.3f}", "nm", "最佳条件下的 CD 误差"],
            ["Focus 范围", f"[{pw.focus_min_nm:.1f}, {pw.focus_max_nm:.1f}]", "nm", "可打印离焦量范围"],
            ["Dose 范围", f"[{pw.dose_min:.4f}, {pw.dose_max:.4f}]", "-", "可打印剂量范围"],
        ]

        self._draw_table(
            ax, headers, rows,
            x=0.03, y=0.67, width=0.94, height=0.50,
            col_widths=[2.5, 2.5, 1, 4],
        )

        pdf.savefig(fig)
        plt.close(fig)

    def _create_meef_page(self, pdf: PdfPages, page_num: int, total_pages: int):
        """创建 MEEF 分析页"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        self._add_page_header(ax, "MEEF (掩模误差增强因子) 分析", page_num, total_pages)

        meef = self.report.final_metrics.meef

        # 大数字展示
        ax.text(0.5, 0.82, "MEEF 均值", transform=ax.transAxes,
                fontsize=14, color='#666', va='center', ha='center')
        ax.text(0.5, 0.72, f"{meef.meef_mean:.3f}", transform=ax.transAxes,
                fontsize=48, fontweight='bold', color=COLOR_PRIMARY, va='center', ha='center')

        # 解读
        if meef.meef_mean < 2:
            level_text = "优秀"
            level_color = COLOR_SUCCESS
        elif meef.meef_mean < 3:
            level_text = "良好"
            level_color = COLOR_WARNING
        else:
            level_text = "需关注"
            level_color = COLOR_DANGER

        ax.add_patch(FancyBboxPatch(
            (0.35, 0.63), 0.3, 0.05,
            transform=ax.transAxes,
            boxstyle="round,pad=0.005",
            facecolor=level_color,
            alpha=0.1,
            edgecolor=level_color,
            linewidth=1.5,
            clip_on=False,
        ))
        ax.text(
            0.5, 0.655,
            f"MEEF 水平: {level_text}",
            transform=ax.transAxes,
            fontsize=12, fontweight='bold', color=level_color,
            va='center', ha='center',
        )

        # 详细参数表
        ax.text(0.03, 0.54, "MEEF 详细指标", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        headers = ["指标", "值", "单位"]
        rows = [
            ["MEEF 均值", f"{meef.meef_mean:.4f}", "-"],
            ["MEEF 最小值", f"{meef.meef_min:.4f}", "-"],
            ["MEEF 最大值", f"{meef.meef_max:.4f}", "-"],
            ["MEEF 标准差", f"{meef.meef_std:.4f}", "-"],
            ["原始掩模 CD", f"{meef.cd_mask_original_nm:.3f}", "nm"],
            ["原始晶圆 CD", f"{meef.cd_wafer_original_nm:.3f}", "nm"],
            ["掩模 CD 变化量", f"{meef.delta_cd_mask_nm:+.3f}", "nm"],
            ["晶圆 CD 变化量", f"{meef.delta_cd_wafer_nm:+.3f}", "nm"],
            ["放大倍率", f"{meef.meef_mean * 100:.1f}", "%"],
            ["采样点数", f"{meef.n_samples}", "个"],
        ]

        self._draw_table(
            ax, headers, rows,
            x=0.03, y=0.51, width=0.94, height=0.42,
            col_widths=[3, 2, 1],
        )

        # 物理解释
        ax.text(0.03, 0.05,
                f"物理解释: 掩模 CD 变化 1 nm 会导致晶圆 CD 变化约 {meef.meef_mean:.2f} nm。"
                f"MEEF 越大表示工艺对掩模误差越敏感。",
                transform=ax.transAxes,
                fontsize=9, color='#666', va='bottom', ha='left',
                wrap=True)

        pdf.savefig(fig)
        plt.close(fig)

    def _create_mask_complexity_page(self, pdf: PdfPages, page_num: int, total_pages: int):
        """创建掩模复杂度分析页"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        self._add_page_header(ax, "掩模复杂度分析", page_num, total_pages)

        init = self.report.initial_metrics.mask_complexity
        final = self.report.final_metrics.mask_complexity

        tv_change = ((final.total_variation - init.total_variation) / init.total_variation * 100
                     if init.total_variation > 0 else 0)

        # 4 个指标卡片
        cards = [
            ("总变差 (TV)", f"{final.total_variation:,.0f}",
             f"{tv_change:+.1f}%",
             COLOR_DANGER if tv_change > 20 else COLOR_WARNING if tv_change > 0 else COLOR_SUCCESS),
            ("二值化惩罚", f"{final.binary_penalty:.4f}",
             f"{((final.binary_penalty - init.binary_penalty)/init.binary_penalty*100) if init.binary_penalty > 0 else 0:+.1f}%",
             COLOR_PRIMARY),
            ("边缘像素数", f"{final.n_edge_pixels:,}",
             f"{final.n_edge_pixels - init.n_edge_pixels:+,} px",
             COLOR_SECONDARY),
            ("SRAF 数量", f"{final.sraf_count}",
             f"{final.sraf_count - init.sraf_count:+d}",
             COLOR_ACCENT),
        ]

        card_w = 0.22
        card_h = 0.1
        start_x = 0.03
        for i, (label, value, sub, color) in enumerate(cards):
            x = start_x + i * (card_w + 0.02)
            self._draw_metric_card(ax, x, 0.86, card_w, card_h, label, value, color=color)
            ax.text(
                x + card_w / 2, 0.86 - card_h + 0.005,
                sub,
                transform=ax.transAxes,
                fontsize=7.5, color='#888',
                va='top', ha='center',
            )

        # 详细对比表
        ax.text(0.03, 0.70, "掩模复杂度指标对比", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        headers = ["指标", "初始值", "最终值", "变化率", "物理意义"]
        rows = [
            ["总变差 (TV)", f"{init.total_variation:.2f}", f"{final.total_variation:.2f}",
             f"{tv_change:+.2f}%", "掩模图形的总梯度，反映整体复杂度"],
            ["各向同性 TV", f"{init.tv_isotropic:.2f}", f"{final.tv_isotropic:.2f}",
             f"{((final.tv_isotropic - init.tv_isotropic)/init.tv_isotropic*100) if init.tv_isotropic > 0 else 0:+.2f}%",
             "各向同性总变差"],
            ["二值化惩罚", f"{init.binary_penalty:.6f}", f"{final.binary_penalty:.6f}",
             f"{((final.binary_penalty - init.binary_penalty)/init.binary_penalty*100) if init.binary_penalty > 0 else 0:+.2f}%",
             "偏离二值的程度，越接近0越好"],
            ["边缘像素数", f"{init.n_edge_pixels:,}", f"{final.n_edge_pixels:,}",
             f"{final.n_edge_pixels - init.n_edge_pixels:+,}",
             "边缘上的像素总数，越多越复杂"],
            ["顶点数 (近似)", f"{init.n_vertices_approx:,}", f"{final.n_vertices_approx:,}",
             f"{final.n_vertices_approx - init.n_vertices_approx:+,}",
             "多边形顶点数量，影响掩模写入时间"],
            ["SRAF 数量", f"{init.sraf_count}", f"{final.sraf_count}",
             f"{final.sraf_count - init.sraf_count:+d}",
             "亚分辨率辅助图形数量"],
            ["平均 SRAF 尺寸", f"{init.sraf_avg_size_nm:.1f} nm" if init.sraf_avg_size_nm else "-",
             f"{final.sraf_avg_size_nm:.1f} nm" if final.sraf_avg_size_nm else "-",
             "-", "SRAF 的平均尺寸"],
        ]

        self._draw_table(
            ax, headers, rows,
            x=0.03, y=0.67, width=0.94, height=0.38,
            col_widths=[2, 2, 2, 1.5, 4],
        )

        pdf.savefig(fig)
        plt.close(fig)

    def _create_mrc_page(self, pdf: PdfPages, page_num: int, total_pages: int):
        """创建 MRC 违规检查页"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        self._add_page_header(ax, "MRC (掩模规则检查) 违规", page_num, total_pages)

        mrc = self.report.mrc_violations

        # 状态大字
        status_color = COLOR_SUCCESS if mrc.passed else COLOR_DANGER
        status_text = "✅ 通过" if mrc.passed else "❌ 未通过"
        ax.text(0.5, 0.85, f"MRC 检查结果: {status_text}", transform=ax.transAxes,
                fontsize=16, fontweight='bold', color=status_color, va='center', ha='center')
        ax.text(0.5, 0.80, f"总计 {mrc.total_violations} 处违规", transform=ax.transAxes,
                fontsize=11, color='#666', va='center', ha='center')

        # 4 类违规卡片
        cards = [
            ("致命 (FATAL)", mrc.fatal_count, COLOR_DANGER),
            ("错误 (ERROR)", mrc.error_count, COLOR_WARNING),
            ("警告 (WARNING)", mrc.warning_count, '#f9a825'),
            ("信息 (INFO)", mrc.info_count, COLOR_ACCENT),
        ]

        card_w = 0.22
        card_h = 0.1
        start_x = 0.03
        for i, (label, count, color) in enumerate(cards):
            x = start_x + i * (card_w + 0.02)
            self._draw_metric_card(ax, x, 0.72, card_w, card_h, label, str(count), color=color)

        # 按规则分类表
        ax.text(0.03, 0.57, "按规则类型分类", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        vbr = mrc.violations_by_rule
        if vbr:
            sorted_rules = sorted(vbr.items(), key=lambda x: -x[1])
            headers = ["规则类型", "违规数量", "占比"]
            rows = []
            total = mrc.total_violations if mrc.total_violations > 0 else 1
            for rule, count in sorted_rules[:10]:
                rows.append([rule, str(count), f"{count/total*100:.1f}%"])

            self._draw_table(
                ax, headers, rows,
                x=0.03, y=0.54, width=0.94, height=min(0.35, len(rows) * 0.035 + 0.035),
                col_widths=[3, 1.5, 1.5],
            )
        else:
            ax.text(0.5, 0.45, "暂无违规分类数据", transform=ax.transAxes,
                    fontsize=11, color='#999', va='center', ha='center')

        # TOP 违规列表
        if mrc.top_violations:
            ax.text(0.03, 0.16, "TOP 违规详情", transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

            y = 0.13
            for i, v in enumerate(mrc.top_violations[:5]):
                severity = v.get('severity', 'info')
                msg = v.get('message', '')
                sev_color = {
                    'fatal': COLOR_DANGER,
                    'error': COLOR_WARNING,
                    'warning': '#f9a825',
                    'info': COLOR_ACCENT,
                }.get(severity, COLOR_ACCENT)

                rect = plt.Rectangle(
                    (0.03, y - 0.02), 0.94, 0.022,
                    transform=ax.transAxes,
                    facecolor=COLOR_LIGHT_BG,
                    edgecolor=sev_color,
                    linewidth=1,
                    clip_on=False,
                )
                ax.add_patch(rect)

                # 左侧色条
                side = plt.Rectangle(
                    (0.03, y - 0.02), 0.008, 0.022,
                    transform=ax.transAxes,
                    facecolor=sev_color,
                    clip_on=False,
                )
                ax.add_patch(side)

                ax.text(
                    0.055, y - 0.009,
                    f"[{severity.upper()}]  {msg}",
                    transform=ax.transAxes,
                    fontsize=8, color=COLOR_TEXT,
                    va='center', ha='left',
                )

                y -= 0.028

        pdf.savefig(fig)
        plt.close(fig)

    def _create_metrology_page(self, pdf: PdfPages, page_num: int, total_pages: int):
        """创建计量一致性评估页"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        self._add_page_header(ax, "计量一致性评估", page_num, total_pages)

        met = self.report.metrology

        # 4 个关键指标
        cards = [
            ("Mean-to-Target", f"{met.m2t_mean_nm:+.2f}", "nm",
             COLOR_PRIMARY),
            ("3σ 均匀性", f"{met.uniformity_3sigma_pct:.2f}", "%",
             COLOR_SECONDARY),
            ("工艺能力 Cpk", f"{met.cpk:.2f}", "-",
             COLOR_SUCCESS if met.cpk >= 1.33 else COLOR_WARNING if met.cpk >= 1.0 else COLOR_DANGER),
            ("合格率", f"{met.pass_rate_pct:.1f}", "%",
             COLOR_ACCENT),
        ]

        card_w = 0.22
        card_h = 0.1
        start_x = 0.03
        for i, (label, value, unit, color) in enumerate(cards):
            x = start_x + i * (card_w + 0.02)
            self._draw_metric_card(ax, x, 0.86, card_w, card_h, label, value, unit, color)

        # 详细指标表
        ax.text(0.03, 0.70, "计量一致性详细指标", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        headers = ["指标类别", "指标名称", "值", "单位"]
        rows = [
            ["准确度", "Mean-to-Target (M2T)", f"{met.m2t_mean_nm:+.3f}", "nm"],
            ["准确度", "M2T 相对误差", f"{met.m2t_pct:+.3f}", "%"],
            ["均匀性", "3σ 均匀性", f"{met.uniformity_3sigma_pct:.3f}", "%"],
            ["均匀性", "极差均匀性", f"{met.uniformity_range_pct:.3f}", "%"],
            ["线性度", "R² 决定系数", f"{met.linearity_r_squared:.4f}", "-"],
            ["线性度", "线性斜率", f"{met.linearity_slope:.4f}", "-"],
            ["精密度", "GRR 占比", f"{met.grr_pct:.2f}", "%"],
            ["精密度", "可区分类别数 (NDC)", f"{met.grr_ndc:.1f}", "-"],
            ["工艺能力", "Cp", f"{met.cp:.3f}", "-"],
            ["工艺能力", "Cpk", f"{met.cpk:.3f}", "-"],
            ["工艺能力", "测量点数", f"{met.n_measurements}", "个"],
        ]

        self._draw_table(
            ax, headers, rows,
            x=0.03, y=0.67, width=0.94, height=0.48,
            col_widths=[2, 3, 2, 1],
        )

        # Cpk 评估
        cpk_status = "优秀 (≥1.33)" if met.cpk >= 1.33 else "可接受 (≥1.0)" if met.cpk >= 1.0 else "不合格 (<1.0)"
        grr_status = "优秀 (<10%)" if met.grr_pct < 10 else "可接受 (<30%)" if met.grr_pct < 30 else "不合格 (≥30%)"

        ax.text(0.03, 0.08,
                f"评估结论: Cpk={met.cpk:.2f} → {cpk_status}  |  "
                f"GRR={met.grr_pct:.1f}% → {grr_status}",
                transform=ax.transAxes,
                fontsize=10, fontweight='bold', color=COLOR_PRIMARY, va='bottom', ha='left')

        pdf.savefig(fig)
        plt.close(fig)

    def _create_figures_page(self, pdf: PdfPages, page_num: int, total_pages: int):
        """创建关键图表面（如果有图表的话）"""
        if not self.report.figures:
            return  # 没有图表则跳过

        for i, fig_data in enumerate(self.report.figures):
            fig_path = Path(fig_data.file_path)
            if not fig_path.exists():
                continue

            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            self._add_page_header(ax, f"关键图表示例 - {fig_data.title}", page_num + i, total_pages)

            # 加载图片并显示
            img = plt.imread(str(fig_path))
            ax_img = fig.add_axes([0.08, 0.15, 0.84, 0.7])
            ax_img.imshow(img)
            ax_img.axis('off')

            # 图注
            ax.text(0.5, 0.09, f"图 {i+1}: {fig_data.caption}",
                    transform=ax.transAxes,
                    fontsize=10, fontstyle='italic', color='#666',
                    va='top', ha='center')

            pdf.savefig(fig)
            plt.close(fig)

    def _get_total_pages(self) -> int:
        """估算总页数"""
        base_pages = 8  # cover, summary, EPE, CD, PW, MEEF, mask complexity, MRC, metrology = 9...
        base_pages = 9  # cover + summary + EPE + CD + PW + MEEF + mask + MRC + metrology

        n_figures = len(self.report.figures)
        # 每个图表一页（假设存在的话）
        total_figure_pages = 0
        for fig_data in self.report.figures:
            if Path(fig_data.file_path).exists():
                total_figure_pages += 1

        return base_pages + total_figure_pages

    def generate(self, output_path: Union[str, Path]) -> Path:
        """
        生成 PDF 报告

        Args:
            output_path: 输出 PDF 文件路径

        Returns:
            输出文件路径
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib 未安装，无法生成 PDF 报告")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_pages = self._get_total_pages()

        with PdfPages(str(output_path)) as pdf:
            current_page = 1

            # 封面
            self._create_cover_page(pdf)
            current_page += 1

            # 执行摘要
            self._create_summary_page(pdf, current_page, total_pages)
            current_page += 1

            # EPE 分析
            self._create_epe_page(pdf, current_page, total_pages)
            current_page += 1

            # CD 分析
            self._create_cd_page(pdf, current_page, total_pages)
            current_page += 1

            # 工艺窗口
            self._create_pw_page(pdf, current_page, total_pages)
            current_page += 1

            # MEEF
            self._create_meef_page(pdf, current_page, total_pages)
            current_page += 1

            # 掩模复杂度
            self._create_mask_complexity_page(pdf, current_page, total_pages)
            current_page += 1

            # MRC
            self._create_mrc_page(pdf, current_page, total_pages)
            current_page += 1

            # 计量一致性
            self._create_metrology_page(pdf, current_page, total_pages)
            current_page += 1

            # 图表页
            for fig_data in self.report.figures:
                if Path(fig_data.file_path).exists():
                    # 单独处理图表面
                    fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
                    ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')

                    self._add_page_header(ax, f"关键图表示例 - {fig_data.title}", current_page, total_pages)

                    img = plt.imread(str(fig_data.file_path))
                    ax_img = fig.add_axes([0.08, 0.15, 0.84, 0.7])
                    ax_img.imshow(img)
                    ax_img.axis('off')

                    ax.text(0.5, 0.09, f"图 {current_page - 9}: {fig_data.caption}",
                            transform=ax.transAxes,
                            fontsize=10, fontstyle='italic', color='#666',
                            va='top', ha='center')

                    pdf.savefig(fig)
                    plt.close(fig)
                    current_page += 1

            # PDF 元数据
            d = pdf.infodict()
            d['Title'] = self.report.title
            d['Author'] = 'RET Sign-off System'
            d['Subject'] = 'Tapeout Sign-off Report'
            d['Keywords'] = 'RET, OPC, EPE, MEEF, MRC, Sign-off'

        logger.info(f"PDF 报告已保存: {output_path}")
        return output_path


def generate_pdf_report(
    report: TapeoutSignoffReport,
    output_path: Union[str, Path],
) -> Path:
    """
    便捷函数：生成 PDF 签核报告

    Args:
        report: 签核报告对象
        output_path: 输出 PDF 文件路径

    Returns:
        输出文件路径
    """
    generator = PDFReportGenerator(report)
    return generator.generate(output_path)
