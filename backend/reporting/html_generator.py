# -*- coding: utf-8 -*-
"""
HTML 签核报告生成器

生成专业的 Tapeout 签核 HTML 报告，包含：
1. 封面与基本信息
2. EPE 对比分析
3. CD 误差分析
4. 工艺窗口分析
5. MEEF 分析
6. 掩模复杂度分析
7. MRC 违规统计
8. 计量一致性评估
9. 关键截图与参数表
"""

import base64
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
import logging

from .schemas import TapeoutSignoffReport

logger = logging.getLogger(__name__)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}

        .report-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        /* Cover Page */
        .cover-page {{
            background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
            color: white;
            padding: 80px 60px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        }}

        .cover-page h1 {{
            font-size: 36px;
            margin-bottom: 20px;
            font-weight: 700;
        }}

        .cover-page .subtitle {{
            font-size: 18px;
            opacity: 0.9;
            margin-bottom: 40px;
        }}

        .cover-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 40px;
        }}

        .cover-info-item {{
            background: rgba(255, 255, 255, 0.1);
            padding: 16px 20px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }}

        .cover-info-item .label {{
            font-size: 12px;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }}

        .cover-info-item .value {{
            font-size: 18px;
            font-weight: 600;
        }}

        /* Sections */
        .section {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 24px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        }}

        .section h2 {{
            font-size: 22px;
            color: #1a237e;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 3px solid #1a237e;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section h2::before {{
            content: '';
            width: 6px;
            height: 24px;
            background: #1a237e;
            border-radius: 3px;
        }}

        .section h3 {{
            font-size: 16px;
            color: #37474f;
            margin: 20px 0 12px 0;
        }}

        /* Summary Cards */
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .summary-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #1a237e;
        }}

        .summary-card.success {{
            border-left-color: #2e7d32;
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        }}

        .summary-card.warning {{
            border-left-color: #f57f17;
            background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        }}

        .summary-card.danger {{
            border-left-color: #c62828;
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        }}

        .summary-card .card-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}

        .summary-card .card-value {{
            font-size: 24px;
            font-weight: 700;
            color: #1a237e;
        }}

        .summary-card .card-unit {{
            font-size: 14px;
            font-weight: 400;
            color: #666;
        }}

        .summary-card .card-change {{
            font-size: 12px;
            margin-top: 6px;
        }}

        .card-change.positive {{
            color: #2e7d32;
        }}

        .card-change.negative {{
            color: #c62828;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 14px;
        }}

        th {{
            background-color: #f0f4f8;
            color: #1a237e;
            font-weight: 600;
            padding: 12px 16px;
            text-align: left;
            border-bottom: 2px solid #1a237e;
        }}

        td {{
            padding: 10px 16px;
            border-bottom: 1px solid #e0e0e0;
        }}

        tr:hover td {{
            background-color: #f8f9fa;
        }}

        .table-header {{
            margin-top: 20px;
            margin-bottom: 8px;
            font-weight: 600;
            color: #37474f;
        }}

        /* Comparison Grid */
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 20px;
        }}

        .comparison-item {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }}

        .comparison-item h4 {{
            color: #666;
            font-size: 14px;
            margin-bottom: 12px;
            font-weight: 500;
        }}

        .comparison-item .big-value {{
            font-size: 28px;
            font-weight: 700;
            color: #1a237e;
        }}

        .comparison-item .unit {{
            font-size: 14px;
            color: #666;
            font-weight: 400;
        }}

        /* Figures */
        .figure-container {{
            margin: 20px 0;
            text-align: center;
        }}

        .figure-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}

        .figure-caption {{
            margin-top: 10px;
            font-size: 13px;
            color: #666;
            font-style: italic;
        }}

        .figure-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}

        /* Status Badge */
        .status-badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }}

        .status-badge.pass {{
            background-color: #e8f5e9;
            color: #2e7d32;
        }}

        .status-badge.fail {{
            background-color: #ffebee;
            color: #c62828;
        }}

        .status-badge.warning {{
            background-color: #fff8e1;
            color: #f57f17;
        }}

        /* Meter / Gauge */
        .meter {{
            margin: 12px 0;
        }}

        .meter-label {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-bottom: 6px;
        }}

        .meter-bar {{
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
        }}

        .meter-fill {{
            height: 100%;
            border-radius: 5px;
            transition: width 0.3s ease;
        }}

        .meter-fill.good {{ background: #2e7d32; }}
        .meter-fill.ok {{ background: #f57f17; }}
        .meter-fill.bad {{ background: #c62828; }}

        /* Violation list */
        .violation-list {{
            margin: 16px 0;
        }}

        .violation-item {{
            display: flex;
            align-items: center;
            padding: 12px 16px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 8px;
            border-left: 4px solid #ccc;
        }}

        .violation-item.fatal {{ border-left-color: #c62828; }}
        .violation-item.error {{ border-left-color: #f57f17; }}
        .violation-item.warning {{ border-left-color: #f9a825; }}
        .violation-item.info {{ border-left-color: #0288d1; }}

        .violation-type {{
            font-weight: 600;
            margin-right: 12px;
            min-width: 100px;
        }}

        .violation-msg {{
            flex: 1;
            font-size: 13px;
        }}

        .violation-count {{
            font-weight: 600;
            color: #666;
        }}

        /* Footer */
        .report-footer {{
            text-align: center;
            padding: 30px;
            color: #999;
            font-size: 12px;
        }}

        /* TOC */
        .toc {{
            background: white;
            border-radius: 12px;
            padding: 24px 30px;
            margin-bottom: 24px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        }}

        .toc h3 {{
            color: #1a237e;
            margin-bottom: 16px;
            font-size: 18px;
        }}

        .toc ul {{
            list-style: none;
            columns: 2;
            column-gap: 30px;
        }}

        .toc li {{
            padding: 6px 0;
        }}

        .toc a {{
            color: #37474f;
            text-decoration: none;
            font-size: 14px;
        }}

        .toc a:hover {{
            color: #1a237e;
        }}

        @media print {{
            body {{
                background: white;
            }}
            .section {{
                break-inside: avoid;
                box-shadow: none;
                border: 1px solid #ddd;
            }}
            .cover-page {{
                break-after: page;
                height: 90vh;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        {content}
        <div class="report-footer">
            <p>报告生成时间: {timestamp} | 报告ID: {report_id}</p>
            <p>本报告由 RET 签核系统自动生成</p>
        </div>
    </div>
</body>
</html>
"""


def _get_status_color(value: float, thresholds: Dict[str, float], higher_is_better: bool = True) -> str:
    """
    根据阈值获取状态颜色

    Args:
        value: 指标值
        thresholds: 阈值字典 {'good': ..., 'ok': ...}
        higher_is_better: 值越大越好

    Returns:
        状态类名: 'good' / 'ok' / 'bad'
    """
    good = thresholds.get('good', 0)
    ok = thresholds.get('ok', 0)

    if higher_is_better:
        if value >= good:
            return 'good'
        elif value >= ok:
            return 'ok'
        else:
            return 'bad'
    else:
        if value <= good:
            return 'good'
        elif value <= ok:
            return 'ok'
        else:
            return 'bad'


def _image_to_base64(image_path: Union[str, Path]) -> str:
    """
    将图片文件转换为 base64 编码

    Args:
        image_path: 图片路径

    Returns:
        base64 编码的图片字符串
    """
    path = Path(image_path)
    if not path.exists():
        return ''

    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode('utf-8')
    suffix = path.suffix.lstrip('.')
    return f"data:image/{suffix};base64,{encoded}"


class HTMLReportGenerator:
    """
    HTML 签核报告生成器

    将 TapeoutSignoffReport 对象转换为专业的 HTML 报告。
    """

    def __init__(self, report: TapeoutSignoffReport, embed_images: bool = True):
        """
        初始化生成器

        Args:
            report: 签核报告对象
            embed_images: 是否将图片嵌入为 base64
        """
        self.report = report
        self.embed_images = embed_images

    def _build_cover(self) -> str:
        """构建封面"""
        r = self.report
        timestamp_str = datetime.fromtimestamp(r.timestamp).strftime('%Y-%m-%d %H:%M:%S')

        mrc_status = "通过" if r.mrc_violations.passed else "未通过"
        mrc_class = "status-badge pass" if r.mrc_violations.passed else "status-badge fail"

        return f"""
        <div class="cover-page">
            <h1>{r.title}</h1>
            <p class="subtitle">Resolution Enhancement Technology Sign-off Report</p>

            <div class="cover-info">
                <div class="cover-info-item">
                    <div class="label">项目名称</div>
                    <div class="value">{r.project_name or '-'}</div>
                </div>
                <div class="cover-info-item">
                    <div class="label">设计名称</div>
                    <div class="value">{r.design_name or '-'}</div>
                </div>
                <div class="cover-info-item">
                    <div class="label">工艺节点</div>
                    <div class="value">{r.technology_node or '-'}</div>
                </div>
                <div class="cover-info-item">
                    <div class="label">RET 流程</div>
                    <div class="value">{r.ret_flow or '-'}</div>
                </div>
                <div class="cover-info-item">
                    <div class="label">生成时间</div>
                    <div class="value">{timestamp_str}</div>
                </div>
                <div class="cover-info-item">
                    <div class="label">MRC 状态</div>
                    <div class="value"><span class="{mrc_class}">{mrc_status}</span></div>
                </div>
            </div>
        </div>
        """

    def _build_toc(self) -> str:
        """构建目录"""
        return f"""
        <div class="toc">
            <h3>📋 目录</h3>
            <ul>
                <li><a href="#sec-overview">1. 执行摘要</a></li>
                <li><a href="#sec-epe">2. EPE (边缘放置误差) 分析</a></li>
                <li><a href="#sec-cd">3. CD (关键尺寸) 分析</a></li>
                <li><a href="#sec-pw">4. 工艺窗口 (PW) 分析</a></li>
                <li><a href="#sec-meef">5. MEEF 分析</a></li>
                <li><a href="#sec-mask-complexity">6. 掩模复杂度分析</a></li>
                <li><a href="#sec-mrc">7. MRC 违规检查</a></li>
                <li><a href="#sec-metrology">8. 计量一致性评估</a></li>
                <li><a href="#sec-figures">9. 关键图表示例</a></li>
                <li><a href="#sec-params">10. 参数配置表</a></li>
            </ul>
        </div>
        """

    def _build_overview(self) -> str:
        """构建执行摘要"""
        r = self.report
        init_epe = r.initial_metrics.epe
        final_epe = r.final_metrics.epe
        epe_improvement = ((init_epe.epe_mean_nm - final_epe.epe_mean_nm) / init_epe.epe_mean_nm * 100.0
                           if init_epe.epe_mean_nm > 0 else 0.0)

        cd = r.final_metrics.cd

        mrc_passed = r.mrc_violations.passed
        mrc_total = r.mrc_violations.total_violations

        cpk = r.metrology.cpk
        if cpk >= 1.33:
            cpk_class = 'success'
        elif cpk >= 1.0:
            cpk_class = 'warning'
        else:
            cpk_class = 'danger'

        return f"""
        <div class="section" id="sec-overview">
            <h2>1. 执行摘要</h2>

            <div class="summary-cards">
                <div class="summary-card {'success' if epe_improvement > 0 else 'danger'}">
                    <div class="card-label">EPE 改善率</div>
                    <div class="card-value">{epe_improvement:+.1f}<span class="card-unit">%</span></div>
                    <div class="card-change positive">
                        {init_epe.epe_mean_nm:.2f} → {final_epe.epe_mean_nm:.2f} nm
                    </div>
                </div>

                <div class="summary-card">
                    <div class="card-label">CD 误差</div>
                    <div class="card-value">{cd.cd_error_mean_nm:+.2f}<span class="card-unit"> nm</span></div>
                    <div class="card-change">
                        相对误差 {cd.cd_error_relative_pct:+.2f}%
                    </div>
                </div>

                <div class="summary-card">
                    <div class="card-label">工艺窗口</div>
                    <div class="card-value">{r.process_window.pw_ratio*100:.1f}<span class="card-unit">%</span></div>
                    <div class="card-change">
                        DOF: {r.process_window.depth_of_focus_nm:.0f} nm
                    </div>
                </div>

                <div class="summary-card {'success' if mrc_passed else 'danger'}">
                    <div class="card-label">MRC 检查</div>
                    <div class="card-value">{'通过' if mrc_passed else '未通过'}</div>
                    <div class="card-change">
                        {mrc_total} 处违规
                    </div>
                </div>

                <div class="summary-card {cpk_class}">
                    <div class="card-label">工艺能力 Cpk</div>
                    <div class="card-value">{cpk:.2f}</div>
                    <div class="card-change">
                        合格率 {r.metrology.pass_rate_pct:.1f}%
                    </div>
                </div>

                <div class="summary-card">
                    <div class="card-label">MEEF</div>
                    <div class="card-value">{r.final_metrics.meef.meef_mean:.2f}</div>
                    <div class="card-change">
                        掩模误差增强因子
                    </div>
                </div>
            </div>
        </div>
        """

    def _build_epe_section(self) -> str:
        """构建 EPE 分析部分"""
        init = self.report.initial_metrics.epe
        final = self.report.final_metrics.epe

        epe_improvement = ((init.epe_mean_nm - final.epe_mean_nm) / init.epe_mean_nm * 100
                           if init.epe_mean_nm > 0 else 0)

        return f"""
        <div class="section" id="sec-epe">
            <h2>2. EPE (边缘放置误差) 分析</h2>

            <div class="comparison-grid">
                <div class="comparison-item">
                    <h4>初始平均 EPE</h4>
                    <div class="big-value">{init.epe_mean_nm:.2f}<span class="unit"> nm</span></div>
                    <p style="margin-top: 8px; font-size: 13px; color: #666;">
                        最大: {init.epe_max_nm:.2f} nm<br>
                        标准差: {init.epe_std_nm:.2f} nm
                    </p>
                </div>
                <div class="comparison-item">
                    <h4>最终平均 EPE</h4>
                    <div class="big-value">{final.epe_mean_nm:.2f}<span class="unit"> nm</span></div>
                    <p style="margin-top: 8px; font-size: 13px; color: #666;">
                        最大: {final.epe_max_nm:.2f} nm<br>
                        标准差: {final.epe_std_nm:.2f} nm
                    </p>
                </div>
            </div>

            <div class="meter">
                <div class="meter-label">
                    <span>EPE 改善率</span>
                    <span style="color: {'#2e7d32' if epe_improvement > 0 else '#c62828'}; font-weight: 600;">
                        {epe_improvement:+.2f}%
                    </span>
                </div>
                <div class="meter-bar">
                    <div class="meter-fill {'good' if epe_improvement >= 30 else 'ok' if epe_improvement >= 10 else 'bad'}"
                         style="width: {min(100, max(0, epe_improvement))}%"></div>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>指标</th>
                        <th>初始值 (nm)</th>
                        <th>最终值 (nm)</th>
                        <th>变化量 (nm)</th>
                        <th>变化率</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>平均 EPE</td>
                        <td>{init.epe_mean_nm:.4f}</td>
                        <td>{final.epe_mean_nm:.4f}</td>
                        <td style="color: {'#2e7d32' if final.epe_mean_nm < init.epe_mean_nm else '#c62828'}">
                            {final.epe_mean_nm - init.epe_mean_nm:+.4f}
                        </td>
                        <td>{epe_improvement:+.2f}%</td>
                    </tr>
                    <tr>
                        <td>最大 EPE</td>
                        <td>{init.epe_max_nm:.4f}</td>
                        <td>{final.epe_max_nm:.4f}</td>
                        <td>{final.epe_max_nm - init.epe_max_nm:+.4f}</td>
                        <td>{((final.epe_max_nm - init.epe_max_nm)/init.epe_max_nm*100) if init.epe_max_nm > 0 else 0:+.2f}%</td>
                    </tr>
                    <tr>
                        <td>EPE 标准差</td>
                        <td>{init.epe_std_nm:.4f}</td>
                        <td>{final.epe_std_nm:.4f}</td>
                        <td>{final.epe_std_nm - init.epe_std_nm:+.4f}</td>
                        <td>{((final.epe_std_nm - init.epe_std_nm)/init.epe_std_nm*100) if init.epe_std_nm > 0 else 0:+.2f}%</td>
                    </tr>
                    <tr>
                        <td>EPE 中位数</td>
                        <td>{init.epe_median_nm:.4f}</td>
                        <td>{final.epe_median_nm:.4f}</td>
                        <td>{final.epe_median_nm - init.epe_median_nm:+.4f}</td>
                        <td>{((final.epe_median_nm - init.epe_median_nm)/init.epe_median_nm*100) if init.epe_median_nm > 0 else 0:+.2f}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

    def _build_cd_section(self) -> str:
        """构建 CD 分析部分"""
        cd = self.report.final_metrics.cd
        init_cd = self.report.initial_metrics.cd

        return f"""
        <div class="section" id="sec-cd">
            <h2>3. CD (关键尺寸) 分析</h2>

            <div class="comparison-grid">
                <div class="comparison-item">
                    <h4>目标 CD</h4>
                    <div class="big-value">{cd.cd_target_nm:.2f}<span class="unit"> nm</span></div>
                </div>
                <div class="comparison-item">
                    <h4>最终平均 CD</h4>
                    <div class="big-value">{cd.cd_mean_nm:.2f}<span class="unit"> nm</span></div>
                </div>
            </div>

            <div class="meter">
                <div class="meter-label">
                    <span>CD 相对误差</span>
                    <span>{cd.cd_error_relative_pct:+.2f}%</span>
                </div>
                <div class="meter-bar">
                    <div class="meter-fill {'good' if abs(cd.cd_error_relative_pct) < 3 else 'ok' if abs(cd.cd_error_relative_pct) < 5 else 'bad'}"
                         style="width: {min(100, abs(cd.cd_error_relative_pct) * 10)}%"></div>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>指标</th>
                        <th>初始值</th>
                        <th>最终值</th>
                        <th>单位</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>平均 CD</td>
                        <td>{init_cd.cd_mean_nm:.2f}</td>
                        <td>{cd.cd_mean_nm:.2f}</td>
                        <td>nm</td>
                    </tr>
                    <tr>
                        <td>最小 CD</td>
                        <td>{init_cd.cd_min_nm:.2f}</td>
                        <td>{cd.cd_min_nm:.2f}</td>
                        <td>nm</td>
                    </tr>
                    <tr>
                        <td>最大 CD</td>
                        <td>{init_cd.cd_max_nm:.2f}</td>
                        <td>{cd.cd_max_nm:.2f}</td>
                        <td>nm</td>
                    </tr>
                    <tr>
                        <td>CD 标准差</td>
                        <td>{init_cd.cd_std_nm:.2f}</td>
                        <td>{cd.cd_std_nm:.2f}</td>
                        <td>nm</td>
                    </tr>
                    <tr>
                        <td>CD 误差均值</td>
                        <td>{init_cd.cd_error_mean_nm:+.2f}</td>
                        <td>{cd.cd_error_mean_nm:+.2f}</td>
                        <td>nm</td>
                    </tr>
                    <tr>
                        <td>特征数量</td>
                        <td>{init_cd.n_features}</td>
                        <td>{cd.n_features}</td>
                        <td>个</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

    def _build_pw_section(self) -> str:
        """构建工艺窗口分析部分"""
        pw = self.report.process_window

        return f"""
        <div class="section" id="sec-pw">
            <h2>4. 工艺窗口 (PW) 分析</h2>

            <div class="summary-cards">
                <div class="summary-card">
                    <div class="card-label">PW 面积</div>
                    <div class="card-value">{pw.pw_area:.2f}<span class="card-unit"> nm·dose</span></div>
                </div>
                <div class="summary-card">
                    <div class="card-label">PW 占比</div>
                    <div class="card-value">{pw.pw_ratio*100:.1f}<span class="card-unit"> %</span></div>
                </div>
                <div class="summary-card">
                    <div class="card-label">焦深 (DOF)</div>
                    <div class="card-value">{pw.depth_of_focus_nm:.1f}<span class="card-unit"> nm</span></div>
                </div>
                <div class="summary-card">
                    <div class="card-label">曝光宽容度</div>
                    <div class="card-value">{pw.exposure_latitude_pct:.2f}<span class="card-unit"> %</span></div>
                </div>
            </div>

            <h3>工艺窗口参数</h3>
            <table>
                <thead>
                    <tr>
                        <th>参数</th>
                        <th>值</th>
                        <th>单位</th>
                        <th>说明</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>可打印条件数</td>
                        <td>{pw.n_passing} / {pw.n_total}</td>
                        <td>-</td>
                        <td>通过 CD 容差的工艺条件</td>
                    </tr>
                    <tr>
                        <td>工艺窗口中心 (Focus)</td>
                        <td>{pw.center_focus_nm:.2f}</td>
                        <td>nm</td>
                        <td>PW 中心的离焦量</td>
                    </tr>
                    <tr>
                        <td>工艺窗口中心 (Dose)</td>
                        <td>{pw.center_dose:.4f}</td>
                        <td>-</td>
                        <td>PW 中心的剂量</td>
                    </tr>
                    <tr>
                        <td>最佳工艺点 (Focus)</td>
                        <td>{pw.best_focus_nm:.2f}</td>
                        <td>nm</td>
                        <td>CD 误差最小的离焦量</td>
                    </tr>
                    <tr>
                        <td>最佳工艺点 (Dose)</td>
                        <td>{pw.best_dose:.4f}</td>
                        <td>-</td>
                        <td>CD 误差最小的剂量</td>
                    </tr>
                    <tr>
                        <td>最佳点 CD 误差</td>
                        <td>{pw.best_cd_error_nm:+.3f}</td>
                        <td>nm</td>
                        <td>最佳条件下的 CD 误差</td>
                    </tr>
                    <tr>
                        <td>Focus 范围</td>
                        <td>[{pw.focus_min_nm:.1f}, {pw.focus_max_nm:.1f}]</td>
                        <td>nm</td>
                        <td>可打印的离焦量范围</td>
                    </tr>
                    <tr>
                        <td>Dose 范围</td>
                        <td>[{pw.dose_min:.4f}, {pw.dose_max:.4f}]</td>
                        <td>-</td>
                        <td>可打印的剂量范围</td>
                    </tr>
                    <tr>
                        <td>椭圆近似面积</td>
                        <td>{pw.ellipse_area:.2f}</td>
                        <td>nm·dose</td>
                        <td>协方差椭圆拟合的 PW 面积</td>
                    </tr>
                    <tr>
                        <td>矩形近似面积</td>
                        <td>{pw.rect_area:.2f}</td>
                        <td>nm·dose</td>
                        <td>外接矩形的 PW 面积</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

    def _build_meef_section(self) -> str:
        """构建 MEEF 分析部分"""
        meef = self.report.final_metrics.meef

        return f"""
        <div class="section" id="sec-meef">
            <h2>5. MEEF (掩模误差增强因子) 分析</h2>

            <div class="comparison-grid">
                <div class="comparison-item">
                    <h4>MEEF 均值</h4>
                    <div class="big-value">{meef.meef_mean:.3f}</div>
                    <p style="margin-top: 8px; font-size: 13px; color: #666;">
                        范围: [{meef.meef_min:.3f}, {meef.meef_max:.3f}]<br>
                        标准差: {meef.meef_std:.3f}
                    </p>
                </div>
                <div class="comparison-item">
                    <h4>CD 变化放大</h4>
                    <div class="big-value">{meef.meef_mean * 100:.1f}<span class="unit"> %</span></div>
                    <p style="margin-top: 8px; font-size: 13px; color: #666;">
                        掩模 CD 变化 1nm → 晶圆 CD 变化 {meef.meef_mean:.2f}nm
                    </p>
                </div>
            </div>

            <div class="meter">
                <div class="meter-label">
                    <span>MEEF 水平</span>
                    <span>{'优' if meef.meef_mean < 2 else '良' if meef.meef_mean < 3 else '需关注'}</span>
                </div>
                <div class="meter-bar">
                    <div class="meter-fill {'good' if meef.meef_mean < 2 else 'ok' if meef.meef_mean < 3 else 'bad'}"
                         style="width: {min(100, meef.meef_mean * 20)}%"></div>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>指标</th>
                        <th>值</th>
                        <th>单位</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>原始掩模 CD</td>
                        <td>{meef.cd_mask_original_nm:.3f}</td>
                        <td>nm</td>
                    </tr>
                    <tr>
                        <td>原始晶圆 CD</td>
                        <td>{meef.cd_wafer_original_nm:.3f}</td>
                        <td>nm</td>
                    </tr>
                    <tr>
                        <td>掩模 CD 变化量</td>
                        <td>{meef.delta_cd_mask_nm:+.3f}</td>
                        <td>nm</td>
                    </tr>
                    <tr>
                        <td>晶圆 CD 变化量</td>
                        <td>{meef.delta_cd_wafer_nm:+.3f}</td>
                        <td>nm</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

    def _build_mask_complexity_section(self) -> str:
        """构建掩模复杂度分析部分"""
        init = self.report.initial_metrics.mask_complexity
        final = self.report.final_metrics.mask_complexity

        tv_change = ((final.total_variation - init.total_variation) / init.total_variation * 100
                     if init.total_variation > 0 else 0)
        bin_change = ((final.binary_penalty - init.binary_penalty) / init.binary_penalty * 100
                      if init.binary_penalty > 0 else 0)

        return f"""
        <div class="section" id="sec-mask-complexity">
            <h2>6. 掩模复杂度分析</h2>

            <div class="summary-cards">
                <div class="summary-card">
                    <div class="card-label">总变差 (TV)</div>
                    <div class="card-value">{final.total_variation:.0f}</div>
                    <div class="card-change {'negative' if tv_change > 0 else 'positive'}">
                        {tv_change:+.1f}%
                    </div>
                </div>
                <div class="summary-card">
                    <div class="card-label">二值化惩罚</div>
                    <div class="card-value">{final.binary_penalty:.4f}</div>
                    <div class="card-change {'negative' if bin_change > 0 else 'positive'}">
                        {bin_change:+.1f}%
                    </div>
                </div>
                <div class="summary-card">
                    <div class="card-label">边缘像素数</div>
                    <div class="card-value">{final.n_edge_pixels:,}</div>
                </div>
                <div class="summary-card">
                    <div class="card-label">SRAF 数量</div>
                    <div class="card-value">{final.sraf_count}</div>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>指标</th>
                        <th>初始值</th>
                        <th>最终值</th>
                        <th>变化率</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>总变差 (Total Variation)</td>
                        <td>{init.total_variation:.2f}</td>
                        <td>{final.total_variation:.2f}</td>
                        <td class="{'negative' if tv_change > 0 else 'positive'}">{tv_change:+.2f}%</td>
                    </tr>
                    <tr>
                        <td>各向同性 TV</td>
                        <td>{init.tv_isotropic:.2f}</td>
                        <td>{final.tv_isotropic:.2f}</td>
                        <td>{((final.tv_isotropic - init.tv_isotropic)/init.tv_isotropic*100) if init.tv_isotropic > 0 else 0:+.2f}%</td>
                    </tr>
                    <tr>
                        <td>二值化惩罚</td>
                        <td>{init.binary_penalty:.6f}</td>
                        <td>{final.binary_penalty:.6f}</td>
                        <td>{bin_change:+.2f}%</td>
                    </tr>
                    <tr>
                        <td>边缘像素数</td>
                        <td>{init.n_edge_pixels:,}</td>
                        <td>{final.n_edge_pixels:,}</td>
                        <td>{final.n_edge_pixels - init.n_edge_pixels:+,} px</td>
                    </tr>
                    <tr>
                        <td>SRAF 数量</td>
                        <td>{init.sraf_count}</td>
                        <td>{final.sraf_count}</td>
                        <td>{final.sraf_count - init.sraf_count:+d}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

    def _build_mrc_section(self) -> str:
        """构建 MRC 违规检查部分"""
        mrc = self.report.mrc_violations
        status_class = 'pass' if mrc.passed else 'fail'
        status_text = '通过' if mrc.passed else '未通过'

        vbr = mrc.violations_by_rule
        vbr_rows = ''
        for rule, count in sorted(vbr.items(), key=lambda x: -x[1]):
            vbr_rows += f'<tr><td>{rule}</td><td>{count}</td></tr>\n'

        top_vios = ''
        for v in mrc.top_violations[:5]:
            severity = v.get('severity', 'info')
            msg = v.get('message', '')
            meas = v.get('measurement_nm', 0)
            thresh = v.get('threshold_nm', 0)
            top_vios += f"""
            <div class="violation-item {severity}">
                <div class="violation-type">{severity.upper()}</div>
                <div class="violation-msg">{msg}</div>
            </div>
            """

        return f"""
        <div class="section" id="sec-mrc">
            <h2>7. MRC (掩模规则检查) 违规</h2>

            <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
                <span class="status-badge {status_class}" style="font-size: 18px; padding: 10px 24px;">
                    检查结果: {status_text}
                </span>
                <span style="color: #666;">总计 {mrc.total_violations} 处违规</span>
            </div>

            <div class="summary-cards">
                <div class="summary-card danger">
                    <div class="card-label">致命 (FATAL)</div>
                    <div class="card-value">{mrc.fatal_count}</div>
                </div>
                <div class="summary-card warning">
                    <div class="card-label">错误 (ERROR)</div>
                    <div class="card-value">{mrc.error_count}</div>
                </div>
                <div class="summary-card" style="border-left-color: #f9a825;">
                    <div class="card-label">警告 (WARNING)</div>
                    <div class="card-value">{mrc.warning_count}</div>
                </div>
                <div class="summary-card" style="border-left-color: #0288d1;">
                    <div class="card-label">信息 (INFO)</div>
                    <div class="card-value">{mrc.info_count}</div>
                </div>
            </div>

            <h3>按规则类型分类</h3>
            <table>
                <thead>
                    <tr>
                        <th>规则类型</th>
                        <th>违规数量</th>
                    </tr>
                </thead>
                <tbody>
                    {vbr_rows or '<tr><td colspan="2">暂无数据</td></tr>'}
                </tbody>
            </table>

            {'<h3>TOP 违规</h3><div class="violation-list">' + top_vios + '</div>' if top_vios else ''}
        </div>
        """

    def _build_metrology_section(self) -> str:
        """构建计量一致性评估部分"""
        met = self.report.metrology

        cpk_status = '优秀' if met.cpk >= 1.33 else '可接受' if met.cpk >= 1.0 else '不合格'
        cpk_color = 'good' if met.cpk >= 1.33 else 'ok' if met.cpk >= 1.0 else 'bad'

        grr_status = '优秀' if met.grr_pct < 10 else '可接受' if met.grr_pct < 30 else '不合格'

        return f"""
        <div class="section" id="sec-metrology">
            <h2>8. 计量一致性评估</h2>

            <div class="summary-cards">
                <div class="summary-card">
                    <div class="card-label">Mean-to-Target</div>
                    <div class="card-value">{met.m2t_mean_nm:+.2f}<span class="card-unit"> nm</span></div>
                    <div class="card-change">{met.m2t_pct:+.2f}%</div>
                </div>
                <div class="summary-card">
                    <div class="card-label">均匀性 (3σ)</div>
                    <div class="card-value">{met.uniformity_3sigma_pct:.2f}<span class="card-unit"> %</span></div>
                </div>
                <div class="summary-card">
                    <div class="card-label">工艺能力 Cpk</div>
                    <div class="card-value">{met.cpk:.2f}</div>
                    <div class="card-change">{cpk_status}</div>
                </div>
                <div class="summary-card">
                    <div class="card-label">合格率</div>
                    <div class="card-value">{met.pass_rate_pct:.1f}<span class="card-unit"> %</span></div>
                </div>
            </div>

            <div class="meter">
                <div class="meter-label">
                    <span>工艺能力 Cpk</span>
                    <span>{cpk_status} ({met.cpk:.2f})</span>
                </div>
                <div class="meter-bar">
                    <div class="meter-fill {cpk_color}" style="width: {min(100, met.cpk * 40)}%"></div>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>指标类别</th>
                        <th>指标</th>
                        <th>值</th>
                        <th>单位</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td rowspan="2">准确度</td>
                        <td>Mean-to-Target</td>
                        <td>{met.m2t_mean_nm:+.3f}</td>
                        <td>nm</td>
                    </tr>
                    <tr>
                        <td>M2T 相对误差</td>
                        <td>{met.m2t_pct:+.3f}</td>
                        <td>%</td>
                    </tr>
                    <tr>
                        <td rowspan="2">均匀性</td>
                        <td>3σ 均匀性</td>
                        <td>{met.uniformity_3sigma_pct:.3f}</td>
                        <td>%</td>
                    </tr>
                    <tr>
                        <td>极差均匀性</td>
                        <td>{met.uniformity_range_pct:.3f}</td>
                        <td>%</td>
                    </tr>
                    <tr>
                        <td rowspan="2">线性度</td>
                        <td>R² 决定系数</td>
                        <td>{met.linearity_r_squared:.4f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>线性斜率</td>
                        <td>{met.linearity_slope:.4f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td rowspan="2">精密度</td>
                        <td>GRR 占比</td>
                        <td>{met.grr_pct:.2f}</td>
                        <td>%</td>
                    </tr>
                    <tr>
                        <td>可区分类别数 (NDC)</td>
                        <td>{met.grr_ndc:.1f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td rowspan="3">工艺能力</td>
                        <td>Cp</td>
                        <td>{met.cp:.3f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>Cpk</td>
                        <td>{met.cpk:.3f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>测量点数</td>
                        <td>{met.n_measurements}</td>
                        <td>个</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

    def _build_figures_section(self) -> str:
        """构建关键图表部分"""
        figures_html = ''
        for fig in self.report.figures:
            img_src = fig.file_path
            if self.embed_images and Path(fig.file_path).exists():
                img_src = _image_to_base64(fig.file_path)

            figures_html += f"""
            <div class="figure-container">
                <img src="{img_src}" alt="{fig.title}">
                <p class="figure-caption">图: {fig.caption}</p>
            </div>
            """

        return f"""
        <div class="section" id="sec-figures">
            <h2>9. 关键图表示例</h2>
            {figures_html if figures_html else '<p style="color: #999;">暂无图表数据</p>'}
        </div>
        """

    def _build_params_section(self) -> str:
        """构建参数配置表部分"""
        tables_html = ''
        for table in self.report.parameter_tables:
            header_row = ''.join(f'<th>{h}</th>' for h in table.headers)
            data_rows = ''
            for row in table.rows:
                data_rows += '<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>\n'

            tables_html += f"""
            <h3>{table.title}</h3>
            <table>
                <thead><tr>{header_row}</tr></thead>
                <tbody>{data_rows}</tbody>
            </table>
            """

        return f"""
        <div class="section" id="sec-params">
            <h2>10. 参数配置表</h2>
            {tables_html if tables_html else '<p style="color: #999;">暂无参数配置数据</p>'}
        </div>
        """

    def generate(self) -> str:
        """
        生成完整的 HTML 报告

        Returns:
            HTML 字符串
        """
        sections = [
            self._build_cover(),
            self._build_toc(),
            self._build_overview(),
            self._build_epe_section(),
            self._build_cd_section(),
            self._build_pw_section(),
            self._build_meef_section(),
            self._build_mask_complexity_section(),
            self._build_mrc_section(),
            self._build_metrology_section(),
            self._build_figures_section(),
            self._build_params_section(),
        ]

        content = '\n'.join(sections)

        timestamp_str = datetime.fromtimestamp(self.report.timestamp).strftime('%Y-%m-%d %H:%M:%S')

        html = HTML_TEMPLATE.format(
            title=self.report.title,
            content=content,
            timestamp=timestamp_str,
            report_id=self.report.report_id,
        )

        logger.info(f"HTML 报告生成完成: {self.report.report_id}")
        return html

    def save(self, output_path: Union[str, Path]) -> Path:
        """
        保存 HTML 报告到文件

        Args:
            output_path: 输出文件路径

        Returns:
            实际写入的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_content = self.generate()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML 报告已保存: {output_path}")
        return output_path


def generate_html_report(
    report: TapeoutSignoffReport,
    output_path: Optional[Union[str, Path]] = None,
    embed_images: bool = True,
) -> Union[str, Path]:
    """
    便捷函数：生成 HTML 签核报告

    Args:
        report: 签核报告对象
        output_path: 输出文件路径，None则返回HTML字符串
        embed_images: 是否将图片嵌入为base64

    Returns:
        HTML字符串或输出文件路径
    """
    generator = HTMLReportGenerator(report, embed_images=embed_images)

    if output_path is not None:
        return generator.save(output_path)
    else:
        return generator.generate()
