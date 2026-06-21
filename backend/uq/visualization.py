# -*- coding: utf-8 -*-
"""
UQ 可视化模块

提供不确定性量化结果的可视化：
- 后验分布 / Bootstrap 重采样直方图
- 置信区间与误差棒图
- 参数敏感性柱状图
- 失效概率 PPM 与可靠性指标图
- 工艺参数散点图与边际分布
- 风险矩阵
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

from uq.schemas import (
    UQResult,
    MetricUncertainty,
    ReliabilityResult,
    FailureProbabilityResult,
)
try:
    from uq.bayesian_inference import PosteriorSample
except ImportError:
    try:
        from .bayesian_inference import PosteriorSample
    except Exception:
        PosteriorSample = Any

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    """
    绘图配置

    Attributes:
        style: 绘图风格 ('default', 'seaborn', 'ggplot')
        dpi: 图像分辨率
        figure_size: 图大小 (width, height) inches
        font_size: 字号
        colormap: 色板名
        save_path: 保存路径模板，如 'uq_{name}.png'
        show: 是否 plt.show()
    """
    style: str = "default"
    dpi: int = 150
    figure_size: Tuple[float, float] = (8.0, 6.0)
    font_size: int = 12
    colormap: str = "viridis"
    save_path: Optional[str] = None
    show: bool = False


def _try_import_matplotlib():
    """延迟导入 matplotlib 以避免无 GUI 环境报错"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        logger.warning(f"matplotlib 不可用: {e}")
        return None


def plot_metric_distribution(
    metric_uncertainty: MetricUncertainty,
    config: Optional[PlotConfig] = None,
    bins: int = 30,
):
    """
    绘制单个指标的样本分布 + 置信区间

    Args:
        metric_uncertainty: 指标不确定性结果
        config: 绘图配置
        bins: 直方图 bin 数

    Returns:
        matplotlib Figure 或 None
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    cfg = config or PlotConfig()
    if cfg.style != "default":
        try:
            plt.style.use(cfg.style)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=cfg.figure_size, dpi=cfg.dpi)

    samples = metric_uncertainty.samples
    ci = metric_uncertainty.confidence_interval

    ax.hist(samples, bins=bins, density=True, alpha=0.7, color="steelblue",
            edgecolor="white", label="Empirical distribution")

    mu = metric_uncertainty.nominal_value
    ax.axvline(mu, color="red", linestyle="--", linewidth=2,
               label=f"Nominal = {mu:.4g}")

    ax.axvline(ci.lower, color="orange", linestyle=":", linewidth=2,
               label=f"CI [{ci.lower:.4g}, {ci.upper:.4g}]")
    ax.axvline(ci.upper, color="orange", linestyle=":", linewidth=2)

    ax.axvspan(ci.lower, ci.upper, alpha=0.2, color="orange",
               label=f"{int(ci.level * 100)}% {ci.method.upper()} CI")

    ax.set_xlabel(metric_uncertainty.metric_name, fontsize=cfg.font_size)
    ax.set_ylabel("Density", fontsize=cfg.font_size)
    ax.set_title(f"Distribution: {metric_uncertainty.metric_name}",
                 fontsize=cfg.font_size + 2)
    ax.legend(fontsize=cfg.font_size - 2)
    ax.grid(True, alpha=0.3)

    if cfg.save_path:
        try:
            fig.savefig(cfg.save_path.format(name="dist_" + metric_uncertainty.metric_name),
                        bbox_inches="tight")
        except Exception as e:
            logger.warning(f"保存图像失败: {e}")

    if cfg.show:
        plt.show()

    return fig


def plot_confidence_intervals(
    metric_uncertainties: Dict[str, MetricUncertainty],
    config: Optional[PlotConfig] = None,
    max_metrics: int = 15,
):
    """
    绘制多指标置信区间对比图（森林图）

    Args:
        metric_uncertainties: {metric_name: MetricUncertainty}
        config: 绘图配置
        max_metrics: 最多显示的指标数

    Returns:
        matplotlib Figure 或 None
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    cfg = config or PlotConfig()
    if cfg.style != "default":
        try:
            plt.style.use(cfg.style)
        except Exception:
            pass

    items = list(metric_uncertainties.items())[:max_metrics]
    names = [k for k, _ in items]
    means = [v.nominal_value for _, v in items]
    lowers = [v.confidence_interval.lower for _, v in items]
    uppers = [v.confidence_interval.upper for _, v in items]

    errors_low = [m - l for m, l in zip(means, lowers)]
    errors_high = [u - m for u, m in zip(uppers, means)]
    errors = [errors_low, errors_high]

    fig, ax = plt.subplots(figsize=(max(cfg.figure_size[0], len(names) * 0.4 + 3),
                                    cfg.figure_size[1]), dpi=cfg.dpi)

    y_pos = np.arange(len(names))
    ax.errorbar(means, y_pos, xerr=errors, fmt="o", capsize=5,
                color="steelblue", ecolor="gray", elinewidth=2,
                markersize=7, markeredgecolor="white")

    ax.axvline(0, color="red", linestyle=":", alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=cfg.font_size)
    ax.set_xlabel("Metric value", fontsize=cfg.font_size)
    ax.set_title(
        f"{int(items[0][1].confidence_interval.level * 100)}% Confidence Intervals",
        fontsize=cfg.font_size + 2,
    )
    ax.grid(True, alpha=0.3, axis="x")

    if cfg.save_path:
        try:
            fig.savefig(cfg.save_path.format(name="ci_forest"), bbox_inches="tight")
        except Exception as e:
            logger.warning(f"保存图像失败: {e}")

    if cfg.show:
        plt.show()

    return fig


def plot_sensitivity_indices(
    sensitivity_indices: Dict[str, Dict[str, Dict[str, float]]],
    config: Optional[PlotConfig] = None,
    top_k: int = 10,
):
    """
    绘制参数敏感度柱状图（Sobol 指数）

    Args:
        sensitivity_indices: {metric: {param: {first_order, total_order}}}
        config: 绘图配置
        top_k: 每个图最多显示参数数量

    Returns:
        Dict[str, Figure] 或 None
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    cfg = config or PlotConfig()
    if cfg.style != "default":
        try:
            plt.style.use(cfg.style)
        except Exception:
            pass

    figures = {}

    for metric_name, param_dict in sensitivity_indices.items():
        items = sorted(param_dict.items(),
                       key=lambda x: -x[1].get("total_order", 0))[:top_k]
        params = [k for k, _ in items]
        first = [v.get("first_order", 0) for _, v in items]
        total = [v.get("total_order", 0) for _, v in items]

        fig, ax = plt.subplots(figsize=(max(cfg.figure_size[0], len(params) * 0.35 + 3),
                                        cfg.figure_size[1]), dpi=cfg.dpi)

        x = np.arange(len(params))
        width = 0.35

        ax.bar(x - width / 2, first, width, label="First-order",
               color="steelblue", alpha=0.85)
        ax.bar(x + width / 2, total, width, label="Total-order",
               color="coral", alpha=0.85)

        ax.set_xlabel("Parameter", fontsize=cfg.font_size)
        ax.set_ylabel("Sensitivity Index", fontsize=cfg.font_size)
        ax.set_title(f"Sensitivity Analysis: {metric_name}",
                     fontsize=cfg.font_size + 2)
        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=45, ha="right", fontsize=cfg.font_size - 1)
        ax.legend(fontsize=cfg.font_size - 1)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(bottom=0)

        if cfg.save_path:
            try:
                fig.savefig(cfg.save_path.format(name=f"sens_{metric_name}"),
                            bbox_inches="tight")
            except Exception as e:
                logger.warning(f"保存图像失败: {e}")

        if cfg.show:
            plt.show()

        figures[metric_name] = fig

    return figures


def plot_parameter_contributions(
    parameter_contributions: Dict[str, float],
    config: Optional[PlotConfig] = None,
):
    """
    绘制参数对失效概率的贡献度饼图或条形图

    Args:
        parameter_contributions: {param_name: contribution (0~1)}
        config: 绘图配置

    Returns:
        matplotlib Figure 或 None
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    cfg = config or PlotConfig()
    if cfg.style != "default":
        try:
            plt.style.use(cfg.style)
        except Exception:
            pass

    items = sorted(parameter_contributions.items(), key=lambda x: -x[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]

    if len(names) > 8:
        others = sum(values[7:])
        names = names[:7] + ["Others"]
        values = values[:7] + [others]

    fig, axes = plt.subplots(1, 2, figsize=(cfg.figure_size[0] * 1.6,
                                           cfg.figure_size[1]),
                             dpi=cfg.dpi)

    cmap = plt.get_cmap(cfg.colormap)
    colors = [cmap(i / max(len(names) - 1, 1)) for i in range(len(names))]

    axes[0].pie(values, labels=names, autopct="%1.1f%%", colors=colors,
                startangle=90, textprops={"fontsize": cfg.font_size - 2})
    axes[0].set_title("Contribution to Failure Risk", fontsize=cfg.font_size)

    x = np.arange(len(names))
    axes[1].bar(x, values, color=colors, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right",
                            fontsize=cfg.font_size - 1)
    axes[1].set_ylabel("Contribution", fontsize=cfg.font_size)
    axes[1].set_title("Parameter Contribution", fontsize=cfg.font_size)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(bottom=0)

    fig.tight_layout()

    if cfg.save_path:
        try:
            fig.savefig(cfg.save_path.format(name="param_contribution"),
                        bbox_inches="tight")
        except Exception as e:
            logger.warning(f"保存图像失败: {e}")

    if cfg.show:
        plt.show()

    return fig


def plot_failure_probability(
    reliability_result: ReliabilityResult,
    config: Optional[PlotConfig] = None,
):
    """
    绘制各失效模式的失效概率（对数刻度 PPM）

    Args:
        reliability_result: 可靠性分析结果
        config: 绘图配置

    Returns:
        matplotlib Figure 或 None
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    cfg = config or PlotConfig()
    if cfg.style != "default":
        try:
            plt.style.use(cfg.style)
        except Exception:
            pass

    fps = reliability_result.failure_probabilities
    names = list(fps.keys())
    pps = [fps[k].probability for k in names]
    ppms = [fps[k].ppm for k in names]
    betas = [fps[k].reliability_index or 0 for k in names]

    fig, axes = plt.subplots(1, 3, figsize=(cfg.figure_size[0] * 2,
                                           cfg.figure_size[1]),
                             dpi=cfg.dpi)

    x = np.arange(len(names))

    axes[0].bar(x, pps, color="indianred", edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right",
                            fontsize=cfg.font_size - 1)
    axes[0].set_ylabel("Failure Probability P_f", fontsize=cfg.font_size)
    axes[0].set_title("Failure Probability", fontsize=cfg.font_size)
    axes[0].grid(True, alpha=0.3, which="both", axis="y")

    axes[1].bar(x, ppms, color="darkorange", edgecolor="white")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right",
                            fontsize=cfg.font_size - 1)
    axes[1].set_ylabel("Defect Rate (PPM)", fontsize=cfg.font_size)
    axes[1].set_title("Defect Rate", fontsize=cfg.font_size)
    axes[1].grid(True, alpha=0.3, which="both", axis="y")

    axes[2].bar(x, betas, color="seagreen", edgecolor="white")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right",
                            fontsize=cfg.font_size - 1)
    axes[2].set_ylabel("Reliability Index β", fontsize=cfg.font_size)
    axes[2].set_title("Reliability Index", fontsize=cfg.font_size)
    axes[2].axhline(3, color="red", linestyle="--", label="β=3 (0.135%)")
    axes[2].axhline(4, color="orange", linestyle="--", label="β=4 (31.7 ppm)")
    axes[2].legend(fontsize=cfg.font_size - 2)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Overall P_f={reliability_result.overall_failure_probability:.2e}, "
                 f"β={reliability_result.reliability_index:.2f}, "
                 f"Risk={reliability_result.risk_level.upper()}",
                 fontsize=cfg.font_size + 2, y=1.02)
    fig.tight_layout()

    if cfg.save_path:
        try:
            fig.savefig(cfg.save_path.format(name="failure_probability"),
                        bbox_inches="tight")
        except Exception as e:
            logger.warning(f"保存图像失败: {e}")

    if cfg.show:
        plt.show()

    return fig


def plot_posterior_distribution(
    posterior: PosteriorSample,
    config: Optional[PlotConfig] = None,
    params: Optional[List[str]] = None,
):
    """
    绘制贝叶斯后验样本的边际分布（直方图 + KDE）

    Args:
        posterior: 后验样本
        config: 绘图配置
        params: 要绘制的参数名列表，None 则绘制全部

    Returns:
        matplotlib Figure 或 None
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    cfg = config or PlotConfig()
    if cfg.style != "default":
        try:
            plt.style.use(cfg.style)
        except Exception:
            pass

    if params is None:
        params = posterior.parameter_names[:8]

    n = len(params)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cfg.figure_size[0] * ncols,
                                      cfg.figure_size[1] * nrows),
                             dpi=cfg.dpi)
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(params):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        samples = posterior.get_param_samples(name)
        hpd = posterior.compute_hpd(name, confidence_level=cfg.get("confidence_level", 0.95)
                                    if isinstance(cfg, dict) else 0.95)

        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(samples)
            xs = np.linspace(samples.min(), samples.max(), 200)
            ax.plot(xs, kde(xs), color="steelblue", linewidth=2, label="KDE")
        except Exception:
            pass

        ax.hist(samples, bins=30, density=True, alpha=0.6, color="steelblue",
                edgecolor="white")

        ax.axvline(np.median(samples), color="red", linestyle="--", linewidth=2,
                   label=f"Median={np.median(samples):.3g}")
        ax.axvspan(hpd.lower, hpd.upper, alpha=0.25, color="orange",
                   label=f"95% HPD")
        ax.axvline(hpd.lower, color="orange", linestyle=":")
        ax.axvline(hpd.upper, color="orange", linestyle=":")

        ax.set_title(f"{name}", fontsize=cfg.font_size)
        ax.legend(fontsize=cfg.font_size - 2)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle("Posterior Distributions", fontsize=cfg.font_size + 2)
    fig.tight_layout()

    if cfg.save_path:
        try:
            fig.savefig(cfg.save_path.format(name="posterior"), bbox_inches="tight")
        except Exception as e:
            logger.warning(f"保存图像失败: {e}")

    if cfg.show:
        plt.show()

    return fig


def plot_process_scatter(
    runs: List[Any],
    x_metric: str = "defocus",
    y_metric: str = "dose",
    color_metric: Optional[str] = "epe_mean",
    failure_mask: Optional[np.ndarray] = None,
    config: Optional[PlotConfig] = None,
):
    """
    绘制工艺参数空间散点图（focus-dose 图）

    Args:
        runs: 仿真结果列表（含 process_condition 和 metrics）
        x_metric: x 轴参数名
        y_metric: y 轴参数名
        color_metric: 颜色映射的指标名
        failure_mask: 是否失效的布尔数组
        config: 绘图配置

    Returns:
        matplotlib Figure 或 None
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    cfg = config or PlotConfig()
    if cfg.style != "default":
        try:
            plt.style.use(cfg.style)
        except Exception:
            pass

    x = []
    y = []
    c = []
    for r in runs:
        pc = r.process_condition if hasattr(r, "process_condition") else {}
        m = r.metrics if hasattr(r, "metrics") else {}
        x.append(pc.get(x_metric, m.get(x_metric, np.nan)))
        y.append(pc.get(y_metric, m.get(y_metric, np.nan)))
        if color_metric:
            c.append(m.get(color_metric, np.nan))

    x = np.array(x)
    y = np.array(y)
    valid = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=cfg.figure_size, dpi=cfg.dpi)

    if color_metric and len(c) > 0:
        c_arr = np.array(c)
        mask = valid & np.isfinite(c_arr)
        sc = ax.scatter(x[mask], y[mask], c=c_arr[mask],
                        cmap=cfg.colormap, alpha=0.8, edgecolors="none", s=40)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_metric, fontsize=cfg.font_size)
    else:
        ax.scatter(x[valid], y[valid], alpha=0.7, s=40, color="steelblue")

    if failure_mask is not None:
        fail_mask = valid & failure_mask
        if np.any(fail_mask):
            ax.scatter(x[fail_mask], y[fail_mask], s=80, facecolors="none",
                       edgecolors="red", linewidths=1.5, label="Failure")
            ax.legend(fontsize=cfg.font_size - 1)

    ax.set_xlabel(x_metric, fontsize=cfg.font_size)
    ax.set_ylabel(y_metric, fontsize=cfg.font_size)
    ax.set_title(f"Process Parameter Space: {x_metric} vs {y_metric}",
                 fontsize=cfg.font_size + 2)
    ax.grid(True, alpha=0.3)

    if cfg.save_path:
        try:
            fig.savefig(cfg.save_path.format(name=f"scatter_{x_metric}_{y_metric}"),
                        bbox_inches="tight")
        except Exception as e:
            logger.warning(f"保存图像失败: {e}")

    if cfg.show:
        plt.show()

    return fig


def plot_summary_dashboard(
    uq_result: UQResult,
    config: Optional[PlotConfig] = None,
):
    """
    绘制 UQ 结果总览仪表盘

    Args:
        uq_result: UQ 分析完整结果
        config: 绘图配置

    Returns:
        matplotlib Figure 或 None
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    cfg = config or PlotConfig()
    if cfg.style != "default":
        try:
            plt.style.use(cfg.style)
        except Exception:
            pass

    fig = plt.figure(figsize=(cfg.figure_size[0] * 2.2,
                              cfg.figure_size[1] * 2.0), dpi=cfg.dpi)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    rel = uq_result.reliability

    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis("off")
    risk_colors = {
        "very_low": "#2ecc71", "low": "#27ae60",
        "medium": "#f39c12", "high": "#e67e22", "very_high": "#e74c3c",
    }
    risk_color = risk_colors.get(rel.risk_level, "#7f8c8d")
    ax_title.text(0.5, 0.85, "UQ Reliability Dashboard",
                  ha="center", va="center", fontsize=cfg.font_size + 8,
                  fontweight="bold")
    ax_title.text(0.5, 0.6, f"Overall P_f = {rel.overall_failure_probability:.2e}",
                  ha="center", va="center", fontsize=cfg.font_size + 3)
    ax_title.text(0.5, 0.42, f"Reliability Index β = {rel.reliability_index:.2f}",
                  ha="center", va="center", fontsize=cfg.font_size + 3)
    ax_title.add_patch(plt.matplotlib.patches.Rectangle(
        (0.2, 0.18), 0.6, 0.15, facecolor=risk_color, edgecolor="none",
        transform=ax_title.transAxes, alpha=0.8,
    ))
    ax_title.text(0.5, 0.255, f"Risk Level: {rel.risk_level.upper()}",
                  ha="center", va="center",
                  fontsize=cfg.font_size + 4, fontweight="bold", color="white")

    ax_bar = fig.add_subplot(gs[0, 1])
    fps = rel.failure_probabilities
    names = list(fps.keys())[:6]
    betas = [fps[k].reliability_index or 0 for k in names]
    colors = ["#e74c3c" if b < 3 else "#f39c12" if b < 4 else "#27ae60" for b in betas]
    ax_bar.bar(names, betas, color=colors, edgecolor="white")
    ax_bar.axhline(3, color="gray", linestyle="--", alpha=0.6)
    ax_bar.axhline(4, color="gray", linestyle=":", alpha=0.6)
    ax_bar.set_ylabel("Reliability Index β", fontsize=cfg.font_size)
    ax_bar.set_title("Failure Modes", fontsize=cfg.font_size + 1)
    ax_bar.tick_params(axis="x", rotation=30, labelsize=cfg.font_size - 2)
    ax_bar.grid(True, alpha=0.3, axis="y")

    ax_pie = fig.add_subplot(gs[0, 2])
    if rel.parameter_contributions:
        items = sorted(rel.parameter_contributions.items(), key=lambda x: -x[1])[:6]
        labels = [k for k, _ in items]
        sizes = [v for _, v in items]
        cmap = plt.get_cmap(cfg.colormap)
        pcolors = [cmap(i / max(len(labels) - 1, 1)) for i in range(len(labels))]
        ax_pie.pie(sizes, labels=labels, autopct="%1.0f%%", colors=pcolors,
                   textprops={"fontsize": cfg.font_size - 2})
        ax_pie.set_title("Failure Risk Drivers", fontsize=cfg.font_size + 1)
    else:
        ax_pie.text(0.5, 0.5, "No sensitivity data", ha="center", va="center",
                    fontsize=cfg.font_size)

    ax_ci = fig.add_subplot(gs[1, :2])
    items = list(uq_result.metric_uncertainties.items())[:8]
    mnames = [k for k, _ in items]
    means = [v.nominal_value for _, v in items]
    errs_low = [v.nominal_value - v.confidence_interval.lower for _, v in items]
    errs_high = [v.confidence_interval.upper - v.nominal_value for _, v in items]
    y_pos = np.arange(len(mnames))
    ax_ci.errorbar(means, y_pos, xerr=[errs_low, errs_high], fmt="o",
                   capsize=4, color="steelblue", ecolor="gray", elinewidth=1.5)
    ax_ci.set_yticks(y_pos)
    ax_ci.set_yticklabels(mnames, fontsize=cfg.font_size - 1)
    ax_ci.set_xlabel("Metric value", fontsize=cfg.font_size)
    ax_ci.set_title(f"Metric Uncertainty ({int(uq_result.confidence_level * 100)}% CI)",
                    fontsize=cfg.font_size + 1)
    ax_ci.grid(True, alpha=0.3, axis="x")

    ax_rec = fig.add_subplot(gs[1, 2])
    ax_rec.axis("off")
    ax_rec.text(0, 1, "Recommendations:", ha="left", va="top",
                fontsize=cfg.font_size + 2, fontweight="bold")
    recs = rel.recommendations[:5]
    y_cursor = 0.88
    for i, rec in enumerate(recs):
        ax_rec.text(0.05, y_cursor, f"• {rec}",
                    ha="left", va="top", fontsize=cfg.font_size - 2,
                    wrap=True, transform=ax_rec.transAxes)
        y_cursor -= 0.18

    fig.suptitle(
        f"UQ Analysis Summary — Method: {uq_result.method.value.upper()}, "
        f"Samples: {uq_result.n_samples}, "
        f"Time: {uq_result.total_time:.1f}s",
        fontsize=cfg.font_size + 4, y=1.01,
    )

    if cfg.save_path:
        try:
            fig.savefig(cfg.save_path.format(name="dashboard"), bbox_inches="tight")
        except Exception as e:
            logger.warning(f"保存图像失败: {e}")

    if cfg.show:
        plt.show()

    return fig
