# -*- coding: utf-8 -*-
"""
报告图表生成模块

生成 Tapeout 签核报告所需的各类图表：
1. EPE 对比柱状图
2. 掩模复杂度对比图
3. 工艺窗口热力图
4. MEEF 柱状图
5. MRC 违规分类统计图
6. 计量一致性雷达图
7. 初始/最终掩模对比图
8. 初始/最终晶圆对比图
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any
from pathlib import Path
import logging

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib import cm
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Figure = None

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def _check_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for report figures")


def plot_epe_comparison(
    initial_epe: Dict[str, float],
    final_epe: Dict[str, float],
    title: str = "EPE 对比 (初始 vs 最终)",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (8, 6),
) -> Optional[Figure]:
    """
    绘制 EPE 对比柱状图

    Args:
        initial_epe: 初始 EPE 指标字典
        final_epe: 最终 EPE 指标字典
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图尺寸

    Returns:
        Figure 对象
    """
    _check_matplotlib()

    metrics = ['epe_mean', 'epe_max', 'epe_std', 'epe_median']
    labels = ['平均 EPE', '最大 EPE', 'EPE 标准差', 'EPE 中位数']

    initial_vals = [initial_epe.get(m + '_nm', 0.0) for m in metrics]
    final_vals = [final_epe.get(m + '_nm', 0.0) for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width/2, initial_vals, width, label='初始', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x + width/2, final_vals, width, label='最终', color='#1f77b4', alpha=0.8)

    ax.set_ylabel('EPE (nm)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    improvements = []
    for i, (init, fin) in enumerate(zip(initial_vals, final_vals)):
        if init > 0:
            imp = (init - fin) / init * 100
            improvements.append(f"{imp:+.1f}%")
        else:
            improvements.append("N/A")

    ax2 = ax.twinx()
    imp_vals = []
    for i in range(len(labels)):
        init = initial_vals[i]
        fin = final_vals[i]
        if init > 0:
            imp_vals.append((init - fin) / init * 100)
        else:
            imp_vals.append(0)
    ax2.plot(x, imp_vals, 'r--o', label='改善率', markersize=6)
    ax2.set_ylabel('改善率 (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"EPE对比图已保存: {save_path}")

    if not show:
        plt.close(fig)

    return fig


def plot_mask_comparison(
    initial_mask: np.ndarray,
    final_mask: np.ndarray,
    title: str = "掩模图案对比 (初始 vs 最终)",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (12, 5),
    cmap: str = 'gray',
) -> Optional[Figure]:
    """
    绘制初始/最终掩模对比图

    Args:
        initial_mask: 初始掩模
        final_mask: 最终掩模
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图尺寸
        cmap: 颜色映射

    Returns:
        Figure 对象
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    im1 = axes[0].imshow(initial_mask, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title('初始掩模', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X (像素)')
    axes[0].set_ylabel('Y (像素)')
    plt.colorbar(im1, ax=axes[0], label='透过率', shrink=0.8)

    im2 = axes[1].imshow(final_mask, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('最终掩模', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X (像素)')
    plt.colorbar(im2, ax=axes[1], label='透过率', shrink=0.8)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"掩模对比图已保存: {save_path}")

    if not show:
        plt.close(fig)

    return fig


def plot_wafer_comparison(
    initial_wafer: np.ndarray,
    final_wafer: np.ndarray,
    target: np.ndarray,
    title: str = "晶圆成像对比",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = 'gray',
) -> Optional[Figure]:
    """
    绘制晶圆成像对比图（初始、最终、目标）

    Args:
        initial_wafer: 初始晶圆图像
        final_wafer: 最终晶圆图像
        target: 目标图像
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图尺寸
        cmap: 颜色映射

    Returns:
        Figure 对象
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im0 = axes[0].imshow(target, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title('目标图案', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X (像素)')
    axes[0].set_ylabel('Y (像素)')

    im1 = axes[1].imshow(initial_wafer, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('初始晶圆', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X (像素)')

    im2 = axes[2].imshow(final_wafer, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title('最终晶圆', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X (像素)')

    fig.colorbar(im0, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08, label='光强')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"晶圆对比图已保存: {save_path}")

    if not show:
        plt.close(fig)

    return fig


def plot_mask_complexity_comparison(
    initial_complexity: Dict[str, Any],
    final_complexity: Dict[str, Any],
    title: str = "掩模复杂度对比",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[Figure]:
    """
    绘制掩模复杂度对比图

    Args:
        initial_complexity: 初始复杂度指标
        final_complexity: 最终复杂度指标
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图尺寸

    Returns:
        Figure 对象
    """
    _check_matplotlib()

    metrics = ['total_variation', 'tv_isotropic', 'binary_penalty', 'n_edge_pixels']
    labels = ['总变差 (TV)', '各向同性 TV', '二值化惩罚', '边缘像素数']

    initial_vals = []
    final_vals = []
    for m in metrics:
        iv = initial_complexity.get(m, 0.0)
        fv = final_complexity.get(m, 0.0)
        initial_vals.append(iv)
        final_vals.append(fv)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    colors_initial = '#ff7f0e'
    colors_final = '#1f77b4'

    for i, (label, iv, fv) in enumerate(zip(labels, initial_vals, final_vals)):
        ax = axes[i]
        bars = ax.bar(['初始', '最终'], [iv, fv],
                      color=[colors_initial, colors_final], alpha=0.8, width=0.5)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}' if height < 1000 else f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"掩模复杂度对比图已保存: {save_path}")

    if not show:
        plt.close(fig)

    return fig


def plot_mrc_violation_summary(
    mrc_summary: Dict[str, Any],
    title: str = "MRC 违规统计",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (12, 5),
) -> Optional[Figure]:
    """
    绘制 MRC 违规统计图

    Args:
        mrc_summary: MRC 违规汇总
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图尺寸

    Returns:
        Figure 对象
    """
    _check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    severity_labels = ['致命', '错误', '警告', '信息']
    severity_counts = [
        mrc_summary.get('fatal_count', 0),
        mrc_summary.get('error_count', 0),
        mrc_summary.get('warning_count', 0),
        mrc_summary.get('info_count', 0),
    ]
    severity_colors = ['#d62728', '#ff7f0e', '#ffbb78', '#1f77b4']

    wedges, texts, autotexts = ax1.pie(
        severity_counts,
        labels=severity_labels,
        colors=severity_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10},
    )
    ax1.set_title('违规严重程度分布', fontsize=12, fontweight='bold')

    violations_by_rule = mrc_summary.get('violations_by_rule', {})
    if violations_by_rule:
        rules = list(violations_by_rule.keys())
        counts = list(violations_by_rule.values())

        sorted_pairs = sorted(zip(counts, rules), reverse=True)
        counts_sorted = [p[0] for p in sorted_pairs]
        rules_sorted = [p[1] for p in sorted_pairs]

        y_pos = np.arange(len(rules_sorted))
        bars = ax2.barh(y_pos, counts_sorted, color='#1f77b4', alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(rules_sorted, fontsize=9)
        ax2.set_xlabel('违规数量')
        ax2.set_title('按规则类型分类', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax2.annotate(f'{int(width)}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=9)
    else:
        ax2.text(0.5, 0.5, '无违规分类数据', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12, alpha=0.5)
        ax2.set_title('按规则类型分类', fontsize=12, fontweight='bold')

    total = mrc_summary.get('total_violations', 0)
    passed = mrc_summary.get('passed', True)
    status = "通过 ✅" if passed else "未通过 ❌"
    fig.suptitle(f"{title} (总计 {total} 处, {status})", fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"MRC违规统计图已保存: {save_path}")

    if not show:
        plt.close(fig)

    return fig


def plot_metrology_radar(
    metrology_data: Dict[str, Any],
    title: str = "计量一致性雷达图",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (8, 8),
) -> Optional[Figure]:
    """
    绘制计量一致性雷达图

    Args:
        metrology_data: 计量一致性数据
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图尺寸

    Returns:
        Figure 对象
    """
    _check_matplotlib()

    labels = [
        'M2T 精度',
        '均匀性',
        '线性度',
        '工艺能力 Cpk',
        '合格率',
    ]

    m2t_pct = abs(metrology_data.get('m2t_pct', 0.0))
    m2t_score = max(0, min(100, 100 - m2t_pct * 10))

    uniformity = 100 - metrology_data.get('uniformity_3sigma_pct', 0.0) * 2
    uniformity_score = max(0, min(100, uniformity))

    linearity = metrology_data.get('linearity_r_squared', 0.0) * 100
    linearity_score = max(0, min(100, linearity))

    cpk = metrology_data.get('cpk', 0.0)
    cpk_score = min(100, cpk * 50)

    pass_rate = metrology_data.get('pass_rate_pct', 0.0)

    values = [
        m2t_score,
        uniformity_score,
        linearity_score,
        cpk_score,
        pass_rate,
    ]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    metric_text = (
        f"M2T: {metrology_data.get('m2t_mean_nm', 0):+.2f} nm "
        f"({metrology_data.get('m2t_pct', 0):+.2f}%)\n"
        f"均匀性 (3σ): {metrology_data.get('uniformity_3sigma_pct', 0):.2f}%\n"
        f"R²: {metrology_data.get('linearity_r_squared', 0):.4f}\n"
        f"Cpk: {metrology_data.get('cpk', 0):.2f}\n"
        f"合格率: {metrology_data.get('pass_rate_pct', 0):.1f}%"
    )
    fig.text(0.5, 0.02, metric_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"计量雷达图已保存: {save_path}")

    if not show:
        plt.close(fig)

    return fig


def plot_meef_comparison(
    meef_data: Dict[str, Any],
    title: str = "MEEF (掩模误差增强因子)",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (8, 6),
) -> Optional[Figure]:
    """
    绘制 MEEF 指标图

    Args:
        meef_data: MEEF 数据
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图尺寸

    Returns:
        Figure 对象
    """
    _check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    meef_mean = meef_data.get('meef_mean', meef_data.get('meef', 0.0))
    meef_max = meef_data.get('meef_max', meef_mean * 1.2)
    meef_min = meef_data.get('meef_min', meef_mean * 0.8)
    meef_std = meef_data.get('meef_std', 0.0)

    err_lower = max(0.0, meef_mean - meef_min) if meef_min < meef_mean else 0.0
    err_upper = max(0.0, meef_max - meef_mean) if meef_max > meef_mean else 0.0

    if err_lower == 0.0 and err_upper == 0.0 and meef_std > 0.0:
        err_lower = meef_std
        err_upper = meef_std

    x = ['MEEF']
    ax1.bar(x, [meef_mean], yerr=[[err_lower], [err_upper]],
            color='#2ca02c', alpha=0.8, capsize=10, width=0.4)
    ax1.set_ylabel('MEEF')
    ax1.set_title('MEEF 值', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='理想值 (1.0)')
    ax1.legend()

    ax1.annotate(f'{meef_mean:.2f}',
                xy=(0, meef_mean),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center', fontsize=11, fontweight='bold')

    cd_mask = meef_data.get('cd_mask_original_nm', 0)
    cd_wafer = meef_data.get('cd_wafer_original_nm', 0)
    delta_mask = meef_data.get('delta_cd_mask_nm', 0)
    delta_wafer = meef_data.get('delta_cd_wafer_nm', 0)

    categories = ['原始 CD', '扰动后 CD']
    mask_vals = [cd_mask, cd_mask + delta_mask]
    wafer_vals = [cd_wafer, cd_wafer + delta_wafer]

    x2 = np.arange(len(categories))
    width = 0.35

    ax2.bar(x2 - width/2, mask_vals, width, label='掩模 CD', color='#ff7f0e', alpha=0.8)
    ax2.bar(x2 + width/2, wafer_vals, width, label='晶圆 CD', color='#1f77b4', alpha=0.8)
    ax2.set_ylabel('CD (nm)')
    ax2.set_title('CD 变化对比', fontsize=12, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"MEEF图已保存: {save_path}")

    if not show:
        plt.close(fig)

    return fig


def plot_process_window_summary(
    pw_data: Dict[str, Any],
    title: str = "工艺窗口 (PW) 综合指标",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[Figure]:
    """
    绘制工艺窗口综合指标图

    Args:
        pw_data: 工艺窗口数据
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图尺寸

    Returns:
        Figure 对象
    """
    _check_matplotlib()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    pw_area = pw_data.get('pw_area', 0.0)
    pw_ratio = pw_data.get('pw_ratio', 0.0) * 100
    ax1.bar(['PW 面积'], [pw_area], color='#2ca02c', alpha=0.8, width=0.5)
    ax1.set_ylabel('PW 面积 (nm·dose)')
    ax1.set_title('工艺窗口面积', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.annotate(f'{pw_area:.2f}\n({pw_ratio:.1f}%)',
                xy=(0, pw_area),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', fontsize=10, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    dof = pw_data.get('depth_of_focus_nm', 0.0)
    ax2.bar(['焦深 (DOF)'], [dof], color='#1f77b4', alpha=0.8, width=0.5)
    ax2.set_ylabel('焦深 (nm)')
    ax2.set_title('焦深 (Depth of Focus)', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.annotate(f'{dof:.1f} nm',
                xy=(0, dof),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', fontsize=10, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 0])
    el = pw_data.get('exposure_latitude_pct', 0.0)
    ax3.bar(['曝光宽容度'], [el], color='#ff7f0e', alpha=0.8, width=0.5)
    ax3.set_ylabel('曝光宽容度 (%)')
    ax3.set_title('曝光宽容度 (EL)', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.annotate(f'{el:.2f}%',
                xy=(0, el),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', fontsize=10, fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 1])
    n_passing = pw_data.get('n_passing', 0)
    n_total = pw_data.get('n_total', 0)
    pass_rate = (n_passing / n_total * 100) if n_total > 0 else 0
    ax4.bar(['可打印比例'], [pass_rate], color='#9467bd', alpha=0.8, width=0.5)
    ax4.set_ylabel('可打印比例 (%)')
    ax4.set_title(f'可打印条件 ({n_passing}/{n_total})', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 100)
    ax4.annotate(f'{pass_rate:.1f}%',
                xy=(0, pass_rate),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', fontsize=10, fontweight='bold')

    fig.suptitle(title, fontsize=14, fontweight='bold')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"工艺窗口图已保存: {save_path}")

    if not show:
        plt.close(fig)

    return fig


def generate_all_report_figures(
    report: Any,
    output_dir: Union[str, Path],
    initial_mask: Optional[np.ndarray] = None,
    final_mask: Optional[np.ndarray] = None,
    initial_wafer: Optional[np.ndarray] = None,
    final_wafer: Optional[np.ndarray] = None,
    target: Optional[np.ndarray] = None,
    fmt: str = 'png',
) -> Dict[str, str]:
    """
    生成所有报告所需的图表

    Args:
        report: 签核报告对象
        output_dir: 输出目录
        initial_mask: 初始掩模（可选）
        final_mask: 最终掩模（可选）
        initial_wafer: 初始晶圆图像（可选）
        final_wafer: 最终晶圆图像（可选）
        target: 目标图像（可选）
        fmt: 图片格式

    Returns:
        图表路径字典 {figure_id: file_path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    try:
        fig_path = figures_dir / f'epe_comparison.{fmt}'
        plot_epe_comparison(
            report.initial_metrics.epe.to_dict(),
            report.final_metrics.epe.to_dict(),
            save_path=str(fig_path),
        )
        paths['epe_comparison'] = str(fig_path)
    except Exception as e:
        logger.warning(f"生成EPE对比图失败: {e}")

    try:
        fig_path = figures_dir / f'mask_complexity.{fmt}'
        plot_mask_complexity_comparison(
            report.initial_metrics.mask_complexity.to_dict(),
            report.final_metrics.mask_complexity.to_dict(),
            save_path=str(fig_path),
        )
        paths['mask_complexity'] = str(fig_path)
    except Exception as e:
        logger.warning(f"生成掩模复杂度图失败: {e}")

    try:
        fig_path = figures_dir / f'mrc_violations.{fmt}'
        plot_mrc_violation_summary(
            report.mrc_violations.to_dict(),
            save_path=str(fig_path),
        )
        paths['mrc_violations'] = str(fig_path)
    except Exception as e:
        logger.warning(f"生成MRC违规图失败: {e}")

    try:
        fig_path = figures_dir / f'metrology_radar.{fmt}'
        plot_metrology_radar(
            report.metrology.to_dict(),
            save_path=str(fig_path),
        )
        paths['metrology_radar'] = str(fig_path)
    except Exception as e:
        logger.warning(f"生成计量雷达图失败: {e}")

    try:
        fig_path = figures_dir / f'process_window.{fmt}'
        plot_process_window_summary(
            report.process_window.to_dict(),
            save_path=str(fig_path),
        )
        paths['process_window'] = str(fig_path)
    except Exception as e:
        logger.warning(f"生成工艺窗口图失败: {e}")

    try:
        fig_path = figures_dir / f'meef.{fmt}'
        plot_meef_comparison(
            report.final_metrics.meef.to_dict(),
            save_path=str(fig_path),
        )
        paths['meef'] = str(fig_path)
    except Exception as e:
        logger.warning(f"生成MEEF图失败: {e}")

    if initial_mask is not None and final_mask is not None:
        try:
            fig_path = figures_dir / f'mask_comparison.{fmt}'
            plot_mask_comparison(
                initial_mask, final_mask,
                save_path=str(fig_path),
            )
            paths['mask_comparison'] = str(fig_path)
        except Exception as e:
            logger.warning(f"生成掩模对比图失败: {e}")

    if initial_wafer is not None and final_wafer is not None and target is not None:
        try:
            fig_path = figures_dir / f'wafer_comparison.{fmt}'
            plot_wafer_comparison(
                initial_wafer, final_wafer, target,
                save_path=str(fig_path),
            )
            paths['wafer_comparison'] = str(fig_path)
        except Exception as e:
            logger.warning(f"生成晶圆对比图失败: {e}")

    logger.info(f"已生成 {len(paths)} 张图表到 {figures_dir}")
    return paths
