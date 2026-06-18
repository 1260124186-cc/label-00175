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


# ============================================================================
# Plotly 交互式可视化模块
# ============================================================================

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from plotly.io import to_html
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    logger.warning("Plotly 未安装，交互式可视化功能不可用。请运行: pip install plotly")


def _check_plotly_available():
    """检查 Plotly 是否可用"""
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly 未安装，交互式可视化功能不可用。"
            "请运行: pip install plotly"
        )


def plot_bossung_interactive(
        focus_values: np.ndarray,
        dose_values: np.ndarray,
        cd_matrix: np.ndarray,
        cd_target: Optional[float] = None,
        cd_tolerance: float = 0.1,
        title: str = "Bossung 图 (CD vs Focus) - 交互式",
        xlabel: str = "离焦量 Focus (nm)",
        ylabel: str = "关键尺寸 CD (nm)",
        height: int = 600,
        width: Optional[int] = None) -> go.Figure:
    """
    绘制交互式 Bossung 图（可缩放、可悬停查看数据）

    Args:
        focus_values: 唯一 focus 值数组 (nm)，形状 (n_focus,)
        dose_values: 唯一 dose 值数组，形状 (n_dose,)
        cd_matrix: CD 值矩阵，形状 (n_focus, n_dose)
        cd_target: 目标 CD 值 (nm)；提供则绘制目标线与容差带
        cd_tolerance: CD 相对容差 (默认 10%)
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        height: 图表高度 (px)
        width: 图表宽度 (px)，None 则自适应

    Returns:
        Plotly Figure 对象（支持 .show() 显示，或 .write_html() 导出）
    """
    _check_plotly_available()

    n_focus = len(focus_values)
    n_dose = len(dose_values)

    if cd_matrix.shape != (n_focus, n_dose):
        raise ValueError(
            f"cd_matrix 形状 {cd_matrix.shape} 与 "
            f"focus_values ({n_focus}) x dose_values ({n_dose}) 不匹配"
        )

    fig = go.Figure()

    colors = px.colors.sequential.Viridis
    color_step = max(1, len(colors) // max(n_dose, 1))

    for j, dose in enumerate(dose_values):
        cd_series = cd_matrix[:, j]
        valid_mask = ~np.isnan(cd_series)
        if np.any(valid_mask):
            color_idx = min(j * color_step, len(colors) - 1)
            fig.add_trace(go.Scatter(
                x=focus_values[valid_mask],
                y=cd_series[valid_mask],
                mode='lines+markers',
                name=f'Dose = {dose:.3f}',
                line=dict(color=colors[color_idx], width=2),
                marker=dict(size=6, color=colors[color_idx]),
                hovertemplate=(
                    f'Focus: %{{x:.1f}} nm<br>'
                    f'CD: %{{y:.2f}} nm<br>'
                    f'Dose: {dose:.3f}<extra></extra>'
                ),
            ))

    if cd_target is not None:
        fig.add_hline(
            y=cd_target,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"目标 CD = {cd_target:.1f} nm",
            annotation_position="bottom right",
        )
        cd_lower = cd_target * (1.0 - cd_tolerance)
        cd_upper = cd_target * (1.0 + cd_tolerance)
        fig.add_hrect(
            y0=cd_lower,
            y1=cd_upper,
            line_width=0,
            fillcolor="green",
            opacity=0.15,
            annotation_text=f"容差带 ±{cd_tolerance * 100:.0f}%",
            annotation_position="bottom left",
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Arial, sans-serif"),
            x=0.5,
        ),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=height,
        width=width,
        margin=dict(l=60, r=40, t=80, b=60),
        template='plotly_white',
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
        zeroline=False,
    )

    return fig


def plot_epe_heatmap_interactive(
        focus_values: np.ndarray,
        dose_values: np.ndarray,
        epe_matrix: np.ndarray,
        passing_mask: Optional[np.ndarray] = None,
        title: str = "EPE 热力图（可点击热点）",
        xlabel: str = "离焦量 Focus (nm)",
        ylabel: str = "曝光剂量 Dose",
        height: int = 600,
        width: Optional[int] = None,
        hotspot_threshold: Optional[float] = None) -> go.Figure:
    """
    绘制交互式 EPE 热力图（可点击热点、可缩放）

    Args:
        focus_values: 唯一 focus 值数组 (nm)，形状 (n_focus,)
        dose_values: 唯一 dose 值数组，形状 (n_dose,)
        epe_matrix: EPE 值矩阵，形状 (n_focus, n_dose)，单位 nm
        passing_mask: 布尔矩阵，True 表示通过规格
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        height: 图表高度
        width: 图表宽度
        hotspot_threshold: 热点阈值 (nm)，超过此值标记为热点；None 则自动取 75% 分位数

    Returns:
        Plotly Figure 对象
    """
    _check_plotly_available()

    n_focus = len(focus_values)
    n_dose = len(dose_values)

    if epe_matrix.shape != (n_focus, n_dose):
        raise ValueError(
            f"epe_matrix 形状 {epe_matrix.shape} 与 "
            f"focus_values ({n_focus}) x dose_values ({n_dose}) 不匹配"
        )

    valid_epe = epe_matrix[~np.isnan(epe_matrix)]
    if len(valid_epe) == 0:
        logger.warning("EPE 矩阵全为 NaN，无法绘制热力图")
        return go.Figure()

    if hotspot_threshold is None:
        hotspot_threshold = float(np.percentile(valid_epe, 75))

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=epe_matrix,
        x=dose_values,
        y=focus_values,
        colorscale='YlOrRd',
        colorbar=dict(
            title="EPE (nm)",
            titleside="right",
        ),
        hovertemplate=(
            'Focus: %{y:.1f} nm<br>'
            'Dose: %{x:.3f}<br>'
            'EPE: %{z:.3f} nm<extra></extra>'
        ),
        name='EPE 热力图',
    ))

    if passing_mask is not None:
        fig.add_trace(go.Contour(
            z=passing_mask.astype(float),
            x=dose_values,
            y=focus_values,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring='none',
            ),
            line=dict(color='black', width=2, dash='dash'),
            name='工艺窗口边界',
            showscale=False,
            hoverinfo='skip',
        ))

    hotspot_data = []
    for i in range(n_focus):
        for j in range(n_dose):
            if not np.isnan(epe_matrix[i, j]) and epe_matrix[i, j] >= hotspot_threshold:
                hotspot_data.append(dict(
                    focus=focus_values[i],
                    dose=dose_values[j],
                    epe=float(epe_matrix[i, j]),
                    is_passing=bool(passing_mask[i, j]) if passing_mask is not None else True,
                ))

    if hotspot_data:
        fig.add_trace(go.Scatter(
            x=[d['dose'] for d in hotspot_data],
            y=[d['focus'] for d in hotspot_data],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=10,
                color='red',
                line=dict(color='white', width=2),
            ),
            name='热点 (High EPE)',
            text=[
                f"Focus: {d['focus']:.1f} nm<br>Dose: {d['dose']:.3f}<br>"
                f"EPE: {d['epe']:.3f} nm<br>"
                f"状态: {'通过' if d['is_passing'] else '失败'}"
                for d in hotspot_data
            ],
            hoverinfo='text',
            visible='legendonly',
        ))

    if len(focus_values) >= 2 and len(dose_values) >= 2:
        mid_f = (focus_values.min() + focus_values.max()) / 2
        mid_d = (dose_values.min() + dose_values.max()) / 2
        fig.add_trace(go.Scatter(
            x=[mid_d],
            y=[mid_f],
            mode='markers',
            marker=dict(
                symbol='star',
                size=15,
                color='white',
                line=dict(color='black', width=2),
            ),
            name='标称工作点',
            hovertemplate=f'标称点<br>Focus: {mid_f:.1f} nm<br>Dose: {mid_d:.3f}<extra></extra>',
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Arial, sans-serif"),
            x=0.5,
        ),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=height,
        width=width,
        margin=dict(l=60, r=40, t=80, b=60),
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        clickmode='event+select',
    )

    return fig


def plot_experiment_comparison_interactive(
        experiments: List[Dict[str, Any]],
        metric_names: Optional[List[str]] = None,
        title: str = "多实验对比（可拖拽排序）",
        height: int = 500,
        width: Optional[int] = None) -> go.Figure:
    """
    绘制交互式多组实验对比图（柱状图，支持图例拖拽排序）

    Args:
        experiments: 实验列表，每个实验为字典，需包含:
                     - 'name': 实验名称
                     - 'metrics': 指标字典 {metric_name: value}
        metric_names: 要显示的指标名称列表；None 则使用所有实验的指标并集
        title: 图表标题
        height: 图表高度
        width: 图表宽度

    Returns:
        Plotly Figure 对象
    """
    _check_plotly_available()

    if not experiments:
        logger.warning("没有实验数据可对比")
        return go.Figure()

    if metric_names is None:
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.get('metrics', {}).keys())
        metric_names = sorted(all_metrics)

    if not metric_names:
        logger.warning("没有可对比的指标")
        return go.Figure()

    colors = px.colors.qualitative.Plotly

    fig = go.Figure()

    for i, exp in enumerate(experiments):
        exp_name = exp.get('name', f'实验 {i+1}')
        metrics = exp.get('metrics', {})
        values = []
        for m in metric_names:
            v = metrics.get(m)
            values.append(v if v is not None else None)

        color = colors[i % len(colors)]
        fig.add_trace(go.Bar(
            x=metric_names,
            y=values,
            name=exp_name,
            marker_color=color,
            text=[f'{v:.4e}' if isinstance(v, (int, float)) and v is not None else 'N/A' for v in values],
            textposition='outside',
            hovertemplate=(
                f'<b>{exp_name}</b><br>'
                '指标: %{x}<br>'
                '数值: %{y:.4e}<extra></extra>'
            ),
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Arial, sans-serif"),
            x=0.5,
        ),
        barmode='group',
        height=height,
        width=width,
        margin=dict(l=60, r=40, t=80, b=100),
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemclick='toggleothers',
            itemdoubleclick='toggle',
        ),
        xaxis=dict(
            tickangle=-45,
            title=None,
        ),
        yaxis=dict(
            title="指标数值",
        ),
    )

    return fig


def plot_loss_curves_interactive(
        experiments: List[Dict[str, Any]],
        title: str = "多实验收敛曲线对比",
        xlabel: str = "迭代次数",
        ylabel: str = "损失值",
        log_y: bool = True,
        height: int = 500,
        width: Optional[int] = None) -> go.Figure:
    """
    绘制多组实验的损失收敛曲线对比（交互式，可缩放）

    Args:
        experiments: 实验列表，每个实验为字典，需包含:
                     - 'name': 实验名称
                     - 'loss_history': 损失值列表
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        log_y: Y 轴是否使用对数刻度
        height: 图表高度
        width: 图表宽度

    Returns:
        Plotly Figure 对象
    """
    _check_plotly_available()

    if not experiments:
        logger.warning("没有实验数据可对比")
        return go.Figure()

    colors = px.colors.qualitative.Plotly

    fig = go.Figure()

    for i, exp in enumerate(experiments):
        exp_name = exp.get('name', f'实验 {i+1}')
        loss_history = exp.get('loss_history', [])
        if not loss_history:
            continue

        iterations = list(range(len(loss_history)))
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=iterations,
            y=loss_history,
            mode='lines',
            name=exp_name,
            line=dict(color=color, width=2),
            hovertemplate=(
                f'<b>{exp_name}</b><br>'
                f'迭代: %{{x}}<br>'
                f'损失: %{{y:.6e}}<extra></extra>'
            ),
        ))

        if loss_history:
            fig.add_annotation(
                x=len(loss_history) - 1,
                y=loss_history[-1],
                text=f"{loss_history[-1]:.4e}",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=0,
                font=dict(color=color, size=10),
            )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Arial, sans-serif"),
            x=0.5,
        ),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=height,
        width=width,
        margin=dict(l=60, r=80, t=80, b=60),
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode='x unified',
    )

    if log_y:
        fig.update_yaxes(type='log')

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
    )

    return fig


def plot_process_window_dashboard(
        scan_result,
        cd_target: Optional[float] = None,
        cd_tolerance: float = 0.1,
        title: str = "工艺窗口综合分析面板",
        height: int = 800,
        width: Optional[int] = None) -> go.Figure:
    """
    绘制工艺窗口综合分析面板（多子图交互式）

    包含：
    1. Bossung 曲线 (CD vs Focus)
    2. CD 误差热力图
    3. EPE 热力图
    4. MSE 热力图

    Args:
        scan_result: ProcessWindowScanResult 实例
        cd_target: 目标 CD (nm)；None 则自动推断
        cd_tolerance: CD 相对容差
        title: 主标题
        height: 图表高度
        width: 图表宽度

    Returns:
        Plotly Figure 对象
    """
    _check_plotly_available()

    focus = scan_result.unique_focus
    dose = scan_result.unique_dose

    if cd_target is None:
        mid_f_idx = len(focus) // 2
        mid_d_idx = len(dose) // 2
        cd_target = float(scan_result.cd_matrix[mid_f_idx, mid_d_idx]) if not np.isnan(
            scan_result.cd_matrix[mid_f_idx, mid_d_idx]
        ) else None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Bossung 图: CD vs Focus",
            "CD 误差分布",
            "EPE 分布",
            "MSE 分布与工艺窗口",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    n_dose = len(dose)
    colors = px.colors.sequential.Viridis
    color_step = max(1, len(colors) // max(n_dose, 1))

    for j, d in enumerate(dose):
        cd_series = scan_result.cd_matrix[:, j]
        valid = ~np.isnan(cd_series)
        if np.any(valid):
            color_idx = min(j * color_step, len(colors) - 1)
            fig.add_trace(
                go.Scatter(
                    x=focus[valid],
                    y=cd_series[valid],
                    mode='lines+markers',
                    name=f'Dose={d:.3f}',
                    line=dict(color=colors[color_idx], width=2),
                    marker=dict(size=4),
                    legendgroup='bossung',
                    showlegend=(j == 0),
                    hovertemplate=(
                        'Focus: %{x:.1f} nm<br>'
                        f'Dose: {d:.3f}<br>'
                        'CD: %{y:.2f} nm<extra></extra>'
                    ),
                ),
                row=1,
                col=1,
            )

    if cd_target is not None:
        fig.add_hline(
            y=cd_target,
            line_dash="dash",
            line_color="black",
            line_width=2,
            row=1,
            col=1,
        )
        cd_lower = cd_target * (1.0 - cd_tolerance)
        cd_upper = cd_target * (1.0 + cd_tolerance)
        fig.add_hrect(
            y0=cd_lower,
            y1=cd_upper,
            line_width=0,
            fillcolor="green",
            opacity=0.15,
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Heatmap(
            z=scan_result.cd_error_matrix,
            x=dose,
            y=focus,
            colorscale='RdBu_r',
            colorbar=dict(title="CD Error (nm)", len=0.4, y=0.8),
            hovertemplate=(
                'Focus: %{y:.1f} nm<br>'
                'Dose: %{x:.3f}<br>'
                'CD Error: %{z:.3f} nm<extra></extra>'
            ),
            showscale=True,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=scan_result.epe_matrix,
            x=dose,
            y=focus,
            colorscale='YlOrRd',
            colorbar=dict(title="EPE (nm)", len=0.4, y=0.2),
            hovertemplate=(
                'Focus: %{y:.1f} nm<br>'
                'Dose: %{x:.3f}<br>'
                'EPE: %{z:.3f} nm<extra></extra>'
            ),
            showscale=True,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=scan_result.mse_matrix,
            x=dose,
            y=focus,
            colorscale='hot_r',
            colorbar=dict(title="MSE", len=0.4, y=0.2),
            hovertemplate=(
                'Focus: %{y:.1f} nm<br>'
                'Dose: %{x:.3f}<br>'
                'MSE: %{z:.4e}<extra></extra>'
            ),
            showscale=True,
        ),
        row=2,
        col=2,
    )

    if scan_result.passing_mask is not None:
        fig.add_trace(
            go.Contour(
                z=scan_result.passing_mask.astype(float),
                x=dose,
                y=focus,
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=1,
                    coloring='none',
                ),
                line=dict(color='lime', width=3, dash='solid'),
                name='可打印区域',
                showscale=False,
            ),
            row=2,
            col=2,
        )

    if len(focus) >= 2 and len(dose) >= 2:
        mid_f = (focus.min() + focus.max()) / 2
        mid_d = (dose.min() + dose.max()) / 2
        for r in range(1, 3):
            for c in range(1, 3):
                fig.add_trace(
                    go.Scatter(
                        x=[mid_d],
                        y=[mid_f],
                        mode='markers',
                        marker=dict(
                            symbol='star',
                            size=12,
                            color='white',
                            line=dict(color='black', width=2),
                        ),
                        name='标称点',
                        showlegend=(r == 1 and c == 1),
                        legendgroup='nominal',
                    ),
                    row=r,
                    col=c,
                )

    fig.update_xaxes(title_text="Focus (nm)", row=1, col=1)
    fig.update_yaxes(title_text="CD (nm)", row=1, col=1)
    fig.update_xaxes(title_text="Dose", row=1, col=2)
    fig.update_yaxes(title_text="Focus (nm)", row=1, col=2)
    fig.update_xaxes(title_text="Dose", row=2, col=1)
    fig.update_yaxes(title_text="Focus (nm)", row=2, col=1)
    fig.update_xaxes(title_text="Dose", row=2, col=2)
    fig.update_yaxes(title_text="Focus (nm)", row=2, col=2)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Arial, sans-serif"),
            x=0.5,
        ),
        height=height,
        width=width,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def generate_html_report(
        figures: List,
        title: str = "光刻仿真分析报告",
        output_path: Optional[str] = None,
        include_summary: bool = True,
        summary_data: Optional[Dict[str, Any]] = None,
        custom_css: Optional[str] = None) -> str:
    """
    生成交互式 HTML 分析报告

    Args:
        figures: Plotly Figure 对象列表
        title: 报告标题
        output_path: 输出 HTML 文件路径；None 则仅返回 HTML 字符串
        include_summary: 是否包含摘要卡片
        summary_data: 摘要数据字典，用于生成数据卡片
        custom_css: 自定义 CSS 样式字符串

    Returns:
        HTML 字符串内容
    """
    _check_plotly_available()

    default_css = """
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            background: #f5f7fa;
            color: #303133;
            padding: 20px;
        }
        .report-container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }
        .report-header h1 {
            font-size: 28px;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .report-header .subtitle {
            font-size: 14px;
            opacity: 0.9;
        }
        .summary-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            transition: transform 0.2s;
        }
        .summary-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        }
        .summary-card .label {
            font-size: 13px;
            color: #909399;
            margin-bottom: 8px;
        }
        .summary-card .value {
            font-size: 24px;
            font-weight: 600;
            color: #303133;
            font-family: 'SF Mono', Consolas, monospace;
        }
        .summary-card .unit {
            font-size: 12px;
            color: #909399;
            font-weight: 400;
            margin-left: 4px;
        }
        .chart-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            margin-bottom: 24px;
        }
        .chart-section h2 {
            font-size: 18px;
            margin-bottom: 16px;
            color: #303133;
            font-weight: 600;
            padding-bottom: 12px;
            border-bottom: 2px solid #ebeef5;
        }
        .chart-container {
            width: 100%;
        }
        .report-footer {
            text-align: center;
            padding: 20px;
            color: #909399;
            font-size: 12px;
        }
    </style>
    """

    css = custom_css if custom_css is not None else default_css

    summary_html = ""
    if include_summary and summary_data:
        cards_html = ""
        for key, data in summary_data.items():
            label = data.get('label', key)
            value = data.get('value', '—')
            unit = data.get('unit', '')
            cards_html += f"""
            <div class="summary-card">
                <div class="label">{label}</div>
                <div class="value">{value}<span class="unit">{unit}</span></div>
            </div>
            """
        summary_html = f"""
        <div class="summary-section">
            {cards_html}
        </div>
        """

    figures_html = ""
    for i, fig in enumerate(figures):
        fig_html = to_html(
            fig,
            include_plotlyjs='cdn',
            full_html=False,
            config={
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            },
        )
        fig_title = fig.layout.title.text if fig.layout.title else f"图表 {i+1}"
        figures_html += f"""
        <div class="chart-section">
            <h2>{fig_title}</h2>
            <div class="chart-container">
                {fig_html}
            </div>
        </div>
        """

    from datetime import datetime
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {css}
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <h1>{title}</h1>
            <div class="subtitle">生成时间: {current_time} | 交互式分析报告</div>
        </div>
        {summary_html}
        {figures_html}
        <div class="report-footer">
            本报告由光刻仿真分析工具自动生成 · 所有图表支持缩放、悬停、下载等交互操作
        </div>
    </div>
</body>
</html>
"""

    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML 报告已导出: {output_path}")

    return html_content


def export_process_window_report(
        scan_result,
        output_path: str,
        cd_target: Optional[float] = None,
        cd_tolerance: float = 0.1,
        title: str = "工艺窗口分析报告") -> str:
    """
    导出工艺窗口完整分析报告（HTML 格式）

    一键生成包含 Bossung 图、各指标热力图、摘要统计的完整交互式报告，
    便于课题组内部分享。

    Args:
        scan_result: ProcessWindowScanResult 实例
        output_path: 输出 HTML 文件路径
        cd_target: 目标 CD (nm)
        cd_tolerance: CD 相对容差
        title: 报告标题

    Returns:
        HTML 字符串内容
    """
    _check_plotly_available()

    focus = scan_result.unique_focus
    dose = scan_result.unique_dose

    bossung_fig = plot_bossung_interactive(
        focus_values=focus,
        dose_values=dose,
        cd_matrix=scan_result.cd_matrix,
        cd_target=cd_target,
        cd_tolerance=cd_tolerance,
        title="Bossung 曲线",
    )

    epe_fig = plot_epe_heatmap_interactive(
        focus_values=focus,
        dose_values=dose,
        epe_matrix=scan_result.epe_matrix,
        passing_mask=scan_result.passing_mask,
        title="EPE 热力图",
    )

    dashboard_fig = plot_process_window_dashboard(
        scan_result=scan_result,
        cd_target=cd_target,
        cd_tolerance=cd_tolerance,
        title="综合分析面板",
    )

    valid_cd = scan_result.cd_matrix[~np.isnan(scan_result.cd_matrix)]
    valid_epe = scan_result.epe_matrix[~np.isnan(scan_result.epe_matrix)]
    valid_mse = scan_result.mse_matrix[~np.isnan(scan_result.mse_matrix)]

    pw_area_pct = 0.0
    if scan_result.passing_mask is not None:
        total = scan_result.passing_mask.size
        passed = np.sum(scan_result.passing_mask)
        pw_area_pct = (passed / total * 100) if total > 0 else 0.0

    summary_data = {
        'nominal_cd': {
            'label': '标称 CD',
            'value': f"{np.median(valid_cd):.2f}" if len(valid_cd) > 0 else '—',
            'unit': ' nm',
        },
        'cd_uniformity': {
            'label': 'CD 均匀性',
            'value': f"{(np.std(valid_cd) / np.mean(valid_cd) * 100) if len(valid_cd) > 0 and np.mean(valid_cd) > 0 else 0:.2f}",
            'unit': ' %',
        },
        'epe_mean': {
            'label': '平均 EPE',
            'value': f"{np.mean(valid_epe):.3f}" if len(valid_epe) > 0 else '—',
            'unit': ' nm',
        },
        'pw_ratio': {
            'label': '工艺窗口比例',
            'value': f"{pw_area_pct:.1f}",
            'unit': ' %',
        },
        'scan_points': {
            'label': '扫描点数',
            'value': f"{len(focus)} × {len(dose)}",
            'unit': '',
        },
        'mse_avg': {
            'label': '平均 MSE',
            'value': f"{np.mean(valid_mse):.4e}" if len(valid_mse) > 0 else '—',
            'unit': '',
        },
    }

    figures = [dashboard_fig, bossung_fig, epe_fig]

    html = generate_html_report(
        figures=figures,
        title=title,
        output_path=output_path,
        include_summary=True,
        summary_data=summary_data,
    )

    return html


def export_experiment_comparison_report(
        experiments: List[Dict[str, Any]],
        output_path: str,
        title: str = "多实验对比分析报告",
        metric_names: Optional[List[str]] = None) -> str:
    """
    导出多实验对比分析报告（HTML 格式）

    Args:
        experiments: 实验列表，每个实验为字典，需包含:
                     - 'name': 实验名称
                     - 'metrics': 指标字典 {metric_name: value}
                     - 'loss_history': 损失历史列表（可选）
        output_path: 输出 HTML 文件路径
        title: 报告标题
        metric_names: 要显示的指标名称列表

    Returns:
        HTML 字符串内容
    """
    _check_plotly_available()

    figures = []

    has_loss_history = any('loss_history' in exp and exp['loss_history'] for exp in experiments)

    if has_loss_history:
        loss_fig = plot_loss_curves_interactive(
            experiments=experiments,
            title="收敛曲线对比",
        )
        figures.append(loss_fig)

    compare_fig = plot_experiment_comparison_interactive(
        experiments=experiments,
        metric_names=metric_names,
        title="指标对比",
    )
    figures.append(compare_fig)

    summary_data = {}
    for i, exp in enumerate(experiments):
        exp_name = exp.get('name', f'实验 {i+1}')
        metrics = exp.get('metrics', {})
        loss_hist = exp.get('loss_history', [])

        final_loss = loss_hist[-1] if loss_hist else None
        if final_loss is not None:
            summary_data[f'exp{i}_final_loss'] = {
                'label': f'{exp_name} - 最终损失',
                'value': f"{final_loss:.4e}",
                'unit': '',
            }

        best_mse = metrics.get('mse') or metrics.get('final_mse')
        if best_mse is not None:
            summary_data[f'exp{i}_mse'] = {
                'label': f'{exp_name} - MSE',
                'value': f"{best_mse:.4e}",
                'unit': '',
            }

    html = generate_html_report(
        figures=figures,
        title=title,
        output_path=output_path,
        include_summary=bool(summary_data),
        summary_data=summary_data if summary_data else None,
    )

    return html
