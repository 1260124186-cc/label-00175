# -*- coding: utf-8 -*-
"""
可视化模块：掩模图案、频域分布、晶圆成像结果、误差曲线绘图

该模块提供一键绑图函数，用于可视化优化过程和结果。
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# 设置中文字体
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
