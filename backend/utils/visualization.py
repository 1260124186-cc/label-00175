# -*- coding: utf-8 -*-
"""
可视化模块：掩模图案、频域分布、晶圆成像结果、误差曲线绘图

该模块提供一键绑图函数，用于可视化优化过程和结果。
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib import cm

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_mask(mask: np.ndarray,
              title: str = "掩模图案",
              cmap: str = 'gray',
              save_path: Optional[str] = None,
              show: bool = True,
              figsize: Tuple[int, int] = (6, 6)) -> Figure:
    """
    绘制掩模图案

    Args:
        mask: 掩模数组
        title: 图像标题
        cmap: 颜色映射
        save_path: 保存路径，None则不保存
        show: 是否显示图像
        figsize: 图像尺寸

    Returns:
        Figure对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel('X (像素)')
    ax.set_ylabel('Y (像素)')

    plt.colorbar(im, ax=ax, label='透过率')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_frequency_domain(spectrum: np.ndarray,
                          title: str = "频域分布",
                          log_scale: bool = True,
                          save_path: Optional[str] = None,
                          show: bool = True,
                          figsize: Tuple[int, int] = (8, 6)) -> Figure:
    """
    绘制频域分布图

    Args:
        spectrum: 频谱数组（复数或幅度）
        title: 图像标题
        log_scale: 是否使用对数尺度
        save_path: 保存路径
        show: 是否显示
        figsize: 图像尺寸

    Returns:
        Figure对象
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 计算幅度和相位
    if np.iscomplexobj(spectrum):
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
    else:
        magnitude = spectrum
        phase = np.zeros_like(spectrum)

    # 幅度图
    if log_scale:
        magnitude_display = np.log10(magnitude + 1e-10)
        mag_label = '幅度 (log10)'
    else:
        magnitude_display = magnitude
        mag_label = '幅度'

    im1 = axes[0].imshow(magnitude_display, cmap='hot')
    axes[0].set_title(f'{title} - 幅度')
    axes[0].set_xlabel('频率 X')
    axes[0].set_ylabel('频率 Y')
    plt.colorbar(im1, ax=axes[0], label=mag_label)

    # 相位图
    im2 = axes[1].imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f'{title} - 相位')
    axes[1].set_xlabel('频率 X')
    axes[1].set_ylabel('频率 Y')
    plt.colorbar(im2, ax=axes[1], label='相位 (rad)')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_wafer_image(wafer_image: np.ndarray,
                     target_image: Optional[np.ndarray] = None,
                     title: str = "晶圆成像",
                     save_path: Optional[str] = None,
                     show: bool = True,
                     figsize: Tuple[int, int] = (10, 4)) -> Figure:
    """
    绘制晶圆成像结果

    Args:
        wafer_image: 晶圆成像结果
        target_image: 目标图像（可选，用于对比）
        title: 图像标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图像尺寸

    Returns:
        Figure对象
    """
    if target_image is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 晶圆成像
        im1 = axes[0].imshow(wafer_image, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('晶圆成像')
        plt.colorbar(im1, ax=axes[0])

        # 目标图像
        im2 = axes[1].imshow(target_image, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('目标图像')
        plt.colorbar(im2, ax=axes[1])

        # 误差图
        error = np.abs(wafer_image - target_image)
        im3 = axes[2].imshow(error, cmap='hot')
        axes[2].set_title('误差分布')
        plt.colorbar(im3, ax=axes[2], label='|误差|')
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(wafer_image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='光强')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_error_curve(loss_history: List[float],
                     title: str = "优化收敛曲线",
                     xlabel: str = "迭代次数",
                     ylabel: str = "损失值",
                     log_scale: bool = False,
                     save_path: Optional[str] = None,
                     show: bool = True,
                     figsize: Tuple[int, int] = (8, 5)) -> Figure:
    """
    绘制误差/损失曲线

    Args:
        loss_history: 损失值历史列表
        title: 图像标题
        xlabel: X轴标签
        ylabel: Y轴标签
        log_scale: 是否使用对数Y轴
        save_path: 保存路径
        show: 是否显示
        figsize: 图像尺寸

    Returns:
        Figure对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    iterations = range(len(loss_history))
    ax.plot(iterations, loss_history, 'b-', linewidth=2, marker='o',
            markersize=3, markevery=max(1, len(loss_history)//20))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    # 标注起始和结束值
    ax.annotate(f'起始: {loss_history[0]:.4e}',
                xy=(0, loss_history[0]),
                xytext=(10, 20), textcoords='offset points',
                fontsize=9, color='green')
    ax.annotate(f'结束: {loss_history[-1]:.4e}',
                xy=(len(loss_history)-1, loss_history[-1]),
                xytext=(-80, -20), textcoords='offset points',
                fontsize=9, color='red')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_comparison(images: List[np.ndarray],
                    titles: List[str],
                    main_title: str = "图像对比",
                    cmap: str = 'gray',
                    save_path: Optional[str] = None,
                    show: bool = True,
                    figsize: Optional[Tuple[int, int]] = None) -> Figure:
    """
    绘制多图对比

    Args:
        images: 图像列表
        titles: 标题列表
        main_title: 主标题
        cmap: 颜色映射
        save_path: 保存路径
        show: 是否显示
        figsize: 图像尺寸

    Returns:
        Figure对象
    """
    n = len(images)

    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(main_title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_optimization_summary(result,
                              save_path: Optional[str] = None,
                              show: bool = True) -> Figure:
    """
    绘制优化结果汇总图

    Args:
        result: MaskOptimizationResult对象
        save_path: 保存路径
        show: 是否显示

    Returns:
        Figure对象
    """
    fig = plt.figure(figsize=(14, 10))

    # 2x3布局
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    # 初始掩模
    im1 = ax1.imshow(result.initial_mask, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('初始掩模')
    plt.colorbar(im1, ax=ax1)

    # 优化后掩模
    im2 = ax2.imshow(result.optimized_mask, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('优化后掩模')
    plt.colorbar(im2, ax=ax2)

    # 目标图像
    im3 = ax3.imshow(result.target_image, cmap='gray', vmin=0, vmax=1)
    ax3.set_title('目标图像')
    plt.colorbar(im3, ax=ax3)

    # 初始成像
    im4 = ax4.imshow(result.initial_wafer_image, cmap='gray', vmin=0, vmax=1)
    ax4.set_title(f'初始成像 (MSE={result.initial_metrics.mse:.4e})')
    plt.colorbar(im4, ax=ax4)

    # 最终成像
    im5 = ax5.imshow(result.final_wafer_image, cmap='gray', vmin=0, vmax=1)
    ax5.set_title(f'最终成像 (MSE={result.final_metrics.mse:.4e})')
    plt.colorbar(im5, ax=ax5)

    # 收敛曲线
    ax6.plot(result.loss_history, 'b-', linewidth=2)
    ax6.set_title('收敛曲线')
    ax6.set_xlabel('迭代次数')
    ax6.set_ylabel('损失值')
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')

    # 添加统计信息
    info_text = (f"总迭代: {result.total_iterations}\n"
                f"总耗时: {result.total_time:.2f}s\n"
                f"收敛: {'是' if result.converged else '否'}")
    ax6.text(0.95, 0.95, info_text, transform=ax6.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_bossung(
        focus_values: np.ndarray,
        dose_values: np.ndarray,
        cd_matrix: np.ndarray,
        cd_target: Optional[float] = None,
        cd_tolerance: float = 0.1,
        title: str = "Bossung 图 (CD vs Focus)",
        xlabel: str = "离焦量 Focus (nm)",
        ylabel: str = "关键尺寸 CD (nm)",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (10, 7)) -> Figure:
    """
    绘制 Bossung 图（PW 曲线）：CD-Focus 曲线，以 Dose 为参数

    Bossung 图是光刻工艺窗口评估的经典工具，显示不同剂量下
    关键尺寸 (CD) 随离焦量 (Focus) 的变化曲线。

    Args:
        focus_values: 唯一 focus 值数组 (nm)，形状 (n_focus,)
        dose_values: 唯一 dose 值数组，形状 (n_dose,)
        cd_matrix: CD 值矩阵，形状 (n_focus, n_dose)，行对应 focus，列对应 dose
        cd_target: 目标 CD 值 (nm)；提供则绘制目标线与容差带
        cd_tolerance: CD 相对容差 (默认 10%)
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        save_path: 保存路径，None 不保存
        show: 是否显示
        figsize: 图像尺寸

    Returns:
        Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_focus = len(focus_values)
    n_dose = len(dose_values)

    if cd_matrix.shape != (n_focus, n_dose):
        raise ValueError(
            f"cd_matrix 形状 {cd_matrix.shape} 与 "
            f"focus_values ({n_focus}) x dose_values ({n_dose}) 不匹配"
        )

    cmap = plt.get_cmap('viridis', n_dose)
    colors = [cmap(i) for i in range(n_dose)]

    for j, dose in enumerate(dose_values):
        cd_series = cd_matrix[:, j]
        valid_mask = ~np.isnan(cd_series)
        if np.any(valid_mask):
            ax.plot(
                focus_values[valid_mask], cd_series[valid_mask],
                marker='o', markersize=5, linewidth=2,
                color=colors[j], label=f'Dose = {dose:.3f}'
            )

    if cd_target is not None:
        ax.axhline(y=cd_target, color='black', linestyle='--', linewidth=2,
                   label=f'目标 CD = {cd_target:.1f} nm')
        cd_lower = cd_target * (1.0 - cd_tolerance)
        cd_upper = cd_target * (1.0 + cd_tolerance)
        ax.axhspan(cd_lower, cd_upper, alpha=0.15, color='green',
                   label=f'容差带 ±{cd_tolerance * 100:.0f}%')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, ncol=2)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_process_window_heatmap(
        focus_values: np.ndarray,
        dose_values: np.ndarray,
        metric_matrix: np.ndarray,
        metric_name: str = "CD 误差 (nm)",
        passing_mask: Optional[np.ndarray] = None,
        cd_target: Optional[float] = None,
        title: str = "工艺窗口热力图",
        xlabel: str = "离焦量 Focus (nm)",
        ylabel: str = "曝光剂量 Dose",
        cmap: str = 'RdYlGn_r',
        show_contours: bool = True,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    绘制工艺窗口可打印区域热力图

    在 Focus-Dose 二维平面上，用颜色表示评估指标（如 CD 误差、MSE、EPE 等），
    并叠加可打印区域（Pass/Fail）的轮廓。

    Args:
        focus_values: 唯一 focus 值数组 (nm)，形状 (n_focus,)
        dose_values: 唯一 dose 值数组，形状 (n_dose,)
        metric_matrix: 评估指标矩阵，形状 (n_focus, n_dose)
        metric_name: 指标名称（用于颜色条标注）
        passing_mask: 布尔矩阵，True 表示该工艺条件满足规格；None 则不显示可打印区域
        cd_target: 目标 CD，用于在图上标注名义工作点
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        cmap: 颜色映射
        show_contours: 是否显示可打印区域轮廓线
        save_path: 保存路径
        show: 是否显示
        figsize: 图像尺寸

    Returns:
        Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_focus = len(focus_values)
    n_dose = len(dose_values)

    if metric_matrix.shape != (n_focus, n_dose):
        raise ValueError(
            f"metric_matrix 形状 {metric_matrix.shape} 与 "
            f"focus_values ({n_focus}) x dose_values ({n_dose}) 不匹配"
        )

    X, Y = np.meshgrid(focus_values, dose_values, indexing='ij')

    valid_mask = ~np.isnan(metric_matrix)
    if not np.any(valid_mask):
        logger.warning("热力图指标矩阵全为 NaN，无法绘制")
        return fig

    Z = np.ma.masked_where(~valid_mask, metric_matrix)

    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto', alpha=0.9)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, fontsize=11)

    if passing_mask is not None:
        pw_passed = np.ma.masked_where(~passing_mask, np.ones_like(passing_mask, dtype=float))
        ax.contourf(X, Y, pw_passed, levels=[0.5, 1.5],
                    colors=['none'], hatches=['....'], alpha=0.0)

        if show_contours:
            ax.contour(X, Y, passing_mask.astype(float), levels=[0.5],
                       colors='black', linewidths=2.5, linestyles='--')

        legend_elements = [
            Patch(facecolor='none', edgecolor='black', linestyle='--', linewidth=2,
                  label='工艺窗口边界'),
            Patch(facecolor='none', edgecolor='gray', hatch='....',
                  label='可打印区域')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')

    if focus_values.size >= 2 and dose_values.size >= 2:
        mid_f = (focus_values.min() + focus_values.max()) / 2
        mid_d = (dose_values.min() + dose_values.max()) / 2
        ax.axvline(x=mid_f, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=mid_d, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.plot(mid_f, mid_d, 'k*', markersize=12, markeredgecolor='white',
                markeredgewidth=1.5, label='标称工作点')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_process_window_summary(
        scan_result,
        cd_target: Optional[float] = None,
        cd_tolerance: float = 0.1,
        title: str = "工艺窗口综合分析",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 12)) -> Figure:
    """
    绘制工艺窗口综合分析图（多子图）

    包含：
    1. Bossung 图 (CD vs Focus)
    2. CD 误差热力图
    3. EPE 热力图
    4. MSE 热力图 + 可打印区域

    Args:
        scan_result: ProcessWindowScanResult 实例
        cd_target: 目标 CD (nm)；None 则根据 passing_mask 自动推断
        cd_tolerance: CD 相对容差
        title: 主标题
        save_path: 保存路径
        show: 是否显示
        figsize: 图像尺寸

    Returns:
        Figure 对象
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    focus = scan_result.unique_focus
    dose = scan_result.unique_dose

    ax1 = fig.add_subplot(gs[0, 0])
    n_dose = len(dose)
    cmap_bossung = plt.get_cmap('viridis', n_dose)
    colors = [cmap_bossung(i) for i in range(n_dose)]

    for j, d in enumerate(dose):
        cd_series = scan_result.cd_matrix[:, j]
        valid = ~np.isnan(cd_series)
        if np.any(valid):
            ax1.plot(focus[valid], cd_series[valid],
                     marker='o', markersize=4, linewidth=2,
                     color=colors[j], label=f'Dose={d:.3f}')

    if cd_target is not None:
        ax1.axhline(y=cd_target, color='black', linestyle='--', linewidth=2,
                    label=f'Target CD={cd_target:.1f}nm')
        cd_lower = cd_target * (1 - cd_tolerance)
        cd_upper = cd_target * (1 + cd_tolerance)
        ax1.axhspan(cd_lower, cd_upper, alpha=0.12, color='green',
                    label=f'±{cd_tolerance * 100:.0f}%')

    ax1.set_xlabel('Focus (nm)', fontsize=11)
    ax1.set_ylabel('CD (nm)', fontsize=11)
    ax1.set_title('Bossung 图: CD vs Focus', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8, ncol=2)

    ax2 = fig.add_subplot(gs[0, 1])
    X, Y = np.meshgrid(focus, dose, indexing='ij')
    Z2 = np.ma.masked_where(np.isnan(scan_result.cd_error_matrix), scan_result.cd_error_matrix)
    im2 = ax2.pcolormesh(X, Y, Z2, cmap='bwr', shading='auto')
    plt.colorbar(im2, ax=ax2, label='CD Error (nm)')

    if scan_result.passing_mask is not None:
        ax2.contour(X, Y, scan_result.passing_mask.astype(float),
                    levels=[0.5], colors='black', linewidths=2, linestyles='--')
    ax2.set_xlabel('Focus (nm)', fontsize=11)
    ax2.set_ylabel('Dose', fontsize=11)
    ax2.set_title('CD 误差分布', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')

    ax3 = fig.add_subplot(gs[1, 0])
    Z3 = np.ma.masked_where(np.isnan(scan_result.epe_matrix), scan_result.epe_matrix)
    im3 = ax3.pcolormesh(X, Y, Z3, cmap='YlOrRd', shading='auto')
    plt.colorbar(im3, ax=ax3, label='EPE Mean (nm)')
    if scan_result.passing_mask is not None:
        ax3.contour(X, Y, scan_result.passing_mask.astype(float),
                    levels=[0.5], colors='black', linewidths=2, linestyles='--')
    ax3.set_xlabel('Focus (nm)', fontsize=11)
    ax3.set_ylabel('Dose', fontsize=11)
    ax3.set_title('边缘放置误差 EPE', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':')

    ax4 = fig.add_subplot(gs[1, 1])
    Z4 = np.ma.masked_where(np.isnan(scan_result.mse_matrix), scan_result.mse_matrix)
    im4 = ax4.pcolormesh(X, Y, Z4, cmap='hot_r', shading='auto')
    plt.colorbar(im4, ax=ax4, label='MSE')

    if scan_result.passing_mask is not None:
        pw_passed = np.ma.masked_where(~scan_result.passing_mask,
                                       np.ones_like(scan_result.passing_mask, dtype=float))
        ax4.contourf(X, Y, pw_passed, levels=[0.5, 1.5],
                     colors=['none'], hatches=['....'], alpha=0.0)
        ax4.contour(X, Y, scan_result.passing_mask.astype(float),
                    levels=[0.5], colors='lime', linewidths=2.5, linestyles='-')

        legend_elements = [
            Patch(facecolor='none', edgecolor='lime', linewidth=2.5,
                  label='可打印区域边界'),
            Patch(facecolor='none', edgecolor='gray', hatch='....',
                  label='可打印区域')
        ]
        ax4.legend(handles=legend_elements, loc='upper left', fontsize=9)

    if len(focus) >= 2 and len(dose) >= 2:
        mid_f = (focus.min() + focus.max()) / 2
        mid_d = (dose.min() + dose.max()) / 2
        ax4.plot(mid_f, mid_d, 'w*', markersize=14, markeredgecolor='black',
                 markeredgewidth=1.5, label='Nominal')

    ax4.set_xlabel('Focus (nm)', fontsize=11)
    ax4.set_ylabel('Dose', fontsize=11)
    ax4.set_title('MSE 分布与可打印区域', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle=':')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_multi_metric_heatmaps(
        scan_result,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[int, int]] = None) -> Figure:
    """
    并排绘制多个工艺指标的热力图

    Args:
        scan_result: ProcessWindowScanResult 实例
        metrics: 要绘制的指标列表，可选值:
                 ['cd', 'cd_error', 'epe', 'mse', 'ssim', 'ils', 'nils']
                 None 则绘制所有可用指标
        save_path: 保存路径
        show: 是否显示
        figsize: 图像尺寸；None 则根据子图数自动计算

    Returns:
        Figure 对象
    """
    all_metrics_info = {
        'cd': ('CD (nm)', scan_result.cd_matrix, 'viridis'),
        'cd_error': ('CD Error (nm)', scan_result.cd_error_matrix, 'bwr'),
        'epe': ('EPE Mean (nm)', scan_result.epe_matrix, 'YlOrRd'),
        'mse': ('MSE', scan_result.mse_matrix, 'hot_r'),
        'ssim': ('SSIM', scan_result.ssim_matrix, 'viridis'),
        'ils': ('ILS (1/nm)', scan_result.ils_matrix, 'plasma'),
        'nils': ('NILS', scan_result.nils_matrix, 'plasma'),
    }

    if metrics is None:
        metrics = [k for k, v in all_metrics_info.items() if not np.all(np.isnan(v[1]))]

    n_plots = len(metrics)
    if n_plots == 0:
        raise ValueError("没有可绘制的有效指标")

    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5 * n_cols, 4.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle('工艺窗口多指标热力图', fontsize=15, fontweight='bold', y=1.0)

    focus = scan_result.unique_focus
    dose = scan_result.unique_dose
    X, Y = np.meshgrid(focus, dose, indexing='ij')

    for idx, metric_key in enumerate(metrics):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r, c]

        if metric_key not in all_metrics_info:
            logger.warning(f"未知指标: {metric_key}，跳过")
            ax.set_visible(False)
            continue

        label, data, cmap = all_metrics_info[metric_key]

        if np.all(np.isnan(data)):
            ax.text(0.5, 0.5, f'{metric_key}: 无数据',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric_key.upper()}', fontsize=12, fontweight='bold')
            continue

        Z = np.ma.masked_where(np.isnan(data), data)
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')
        plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

        if scan_result.passing_mask is not None:
            ax.contour(X, Y, scan_result.passing_mask.astype(float),
                       levels=[0.5], colors='black', linewidths=1.5, linestyles='--')

        ax.set_xlabel('Focus (nm)', fontsize=10)
        ax.set_ylabel('Dose', fontsize=10)
        ax.set_title(f'{metric_key.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle=':')

    for idx in range(n_plots, n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r, c].set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
