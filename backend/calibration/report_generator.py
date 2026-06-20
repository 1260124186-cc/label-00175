# -*- coding: utf-8 -*-
"""
标定报告生成器

生成 Fab 模型标定的完整报告：
1. 文本报告（stdout / 文件）
2. Markdown 报告
3. 可视化图表（测量 vs 预测、Bossung 曲线、参数相关、残差、MCMC 迹）

依赖：matplotlib（已在项目 requirements.txt 中）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端，服务器环境兼容
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from .schemas import (
    CalibrationConfig,
    CDSEMDataset,
    InversionResult,
    CalibrationReport,
    CalibrationParameterSet,
    PatternType,
)
from .forward_model import (
    LithoForwardModel,
    compute_bossung_cd,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 报告生成（文本 / Markdown）
# ---------------------------------------------------------------------------

def generate_calibration_report(config: CalibrationConfig,
                                 dataset: CDSEMDataset,
                                 inversion_result: InversionResult,
                                 test_dataset: Optional[CDSEMDataset] = None,
                                 validation_report: Optional[Dict[str, Any]] = None,
                                 duration_sec: float = 0.0,
                                 ) -> CalibrationReport:
    """
    生成完整的 CalibrationReport 对象（内存中的结构化数据）。
    """
    # 质量指标
    metrics = _compute_metrics(inversion_result, dataset, test_dataset)

    dataset_info = {
        'fab_name': dataset.fab_name,
        'process_node': dataset.process_node,
        'n_points': len(dataset),
        'focus_range_nm': dataset.focus_range(),
        'dose_range': dataset.dose_range(),
        'target_cd_range_nm': (float(dataset.target_cds().min()),
                               float(dataset.target_cds().max())),
        'measured_cd_range_nm': (float(dataset.measured_cds().min()),
                                 float(dataset.measured_cds().max())),
        'pattern_types': [pt.value for pt in sorted(set(dataset.pattern_types()),
                                                     key=lambda x: x.value)],
        'magnification': dataset.magnification,
        'mask_set_id': dataset.mask_set_id,
        'wafer_id': dataset.wafer_id,
        'lot_id': dataset.lot_id,
    }
    if validation_report is not None:
        dataset_info['validation'] = validation_report

    return CalibrationReport(
        config=config,
        dataset_info=dataset_info,
        inversion_result=inversion_result,
        metrics=metrics,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        duration_sec=duration_sec,
    )


def _compute_metrics(inversion: InversionResult,
                     train_ds: CDSEMDataset,
                     test_ds: Optional[CDSEMDataset] = None) -> Dict[str, float]:
    """计算回归质量指标。"""
    metrics: Dict[str, float] = {}

    # 训练集
    if inversion.residuals is not None and inversion.predicted_cds is not None:
        resid = inversion.residuals
        measured = train_ds.measured_cds()
        predicted = inversion.predicted_cds

        metrics['train_MAE_nm'] = float(np.mean(np.abs(resid)))
        metrics['train_RMSE_nm'] = float(np.sqrt(np.mean(resid ** 2)))
        metrics['train_max_error_nm'] = float(np.max(np.abs(resid)))
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((measured - np.mean(measured)) ** 2))
        metrics['train_R2'] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        # Mean Absolute Percentage Error
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.abs(resid) / np.maximum(np.abs(measured), 1e-9)
        metrics['train_MAPE_pct'] = float(100.0 * np.mean(mape))

    # 测试集（若提供）
    if test_ds is not None and len(test_ds) > 0:
        try:
            model = LithoForwardModel(inversion.calibrated_values)
            preds = model.predict_dataset(test_ds)
            measured = test_ds.measured_cds()
            resid = measured - preds
            metrics['test_MAE_nm'] = float(np.mean(np.abs(resid)))
            metrics['test_RMSE_nm'] = float(np.sqrt(np.mean(resid ** 2)))
            metrics['test_max_error_nm'] = float(np.max(np.abs(resid)))
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((measured - np.mean(measured)) ** 2))
            metrics['test_R2'] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.abs(resid) / np.maximum(np.abs(measured), 1e-9)
            metrics['test_MAPE_pct'] = float(100.0 * np.mean(mape))
        except Exception as e:
            logger.warning(f"测试集评估失败: {e}")

    return metrics


def generate_markdown_report(report: CalibrationReport,
                             output_dir: Union[str, Path],
                             filename: str = "calibration_report.md",
                             include_plots: bool = True,
                             ) -> str:
    """
    生成 Markdown 格式报告，写入 output_dir/filename。

    Returns:
        生成的 Markdown 文本。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inv = report.inversion_result
    cfg = report.config
    ds_info = report.dataset_info

    lines = []
    lines.append("# Fab 模型标定报告")
    lines.append("")
    lines.append(f"**生成时间**：{report.timestamp}  ")
    lines.append(f"**反演耗时**：{report.duration_sec:.2f} s  ")
    lines.append(f"**反演方法**：`{inv.method.value}`  ")
    lines.append(f"**收敛状态**：{'✅ 成功' if inv.success else '❌ 失败'}  ")
    if inv.message:
        lines.append(f"**消息**：{inv.message}  ")
    lines.append("")

    # 数据集概览
    lines.append("## 1. CD-SEM 数据集概览")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| FAB 厂 | {ds_info.get('fab_name', '-')} |")
    lines.append(f"| 工艺节点 | {ds_info.get('process_node', '-')} |")
    lines.append(f"| 掩模组 ID | {ds_info.get('mask_set_id', '-')} |")
    lines.append(f"| 晶圆 ID | {ds_info.get('wafer_id', '-')} |")
    lines.append(f"| 批次 ID | {ds_info.get('lot_id', '-')} |")
    lines.append(f"| 放大倍率 | {ds_info.get('magnification', '-')}× |")
    lines.append(f"| 数据点数量 | {ds_info.get('n_points', 0)} |")
    lines.append(f"| Focus 范围 | {ds_info.get('focus_range_nm', '-')} nm |")
    lines.append(f"| Dose 范围 | {ds_info.get('dose_range', '-')} |")
    lines.append(f"| 目标 CD 范围 | {ds_info.get('target_cd_range_nm', '-')} nm |")
    lines.append(f"| 实测 CD 范围 | {ds_info.get('measured_cd_range_nm', '-')} nm |")
    lines.append(f"| 图形类型 | {', '.join(ds_info.get('pattern_types', []))} |")
    lines.append("")

    # 标定参数
    lines.append("## 2. 标定参数结果")
    lines.append("")
    lines.append("| 参数名 | 标定值 | ±1σ 不确定度 | 单位 | 物理含义 |")
    lines.append("|--------|--------|---------------|------|----------|")
    all_params = cfg.parameters.all_parameters()
    param_by_name = {p.name: p for p in all_params}
    for name, val in inv.calibrated_values.items():
        unc = inv.uncertainties.get(name, 0.0)
        pobj = param_by_name.get(name)
        unit = pobj.unit if pobj else ""
        desc = pobj.description if pobj else ""
        vary_flag = ""
        if pobj and not pobj.vary:
            vary_flag = " 🔒(固定)"
        lines.append(f"| `{name}`{vary_flag} | {val:.6f} | ±{unc:.6f} | {unit} | {desc} |")
    lines.append("")

    # 参数相关性（展示高度相关的参数对）
    if inv.correlation_matrix is not None and len(inv.varying_names) > 1:
        lines.append("### 参数相关系数矩阵")
        lines.append("")
        corr = inv.correlation_matrix
        names = inv.varying_names
        header = "| | " + " | ".join(f"`{n}`" for n in names) + " |"
        sep = "|" + "|".join(["---"] * (len(names) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for i, n in enumerate(names):
            row = f"| `{n}` | " + " | ".join(f"{corr[i,j]:+.3f}" for j in range(len(names))) + " |"
            lines.append(row)
        lines.append("")

        # 高亮强相关
        high_corr = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                if abs(corr[i,j]) >= 0.7:
                    high_corr.append(
                        f"- `{names[i]}` ↔ `{names[j]}`: r = {corr[i,j]:+.3f}"
                    )
        if high_corr:
            lines.append("⚠️ **高相关参数对（|r|≥0.7，可能影响可辨识性）：**")
            lines.append("")
            lines.extend(high_corr)
            lines.append("")

    # 反演统计
    lines.append("## 3. 反演统计")
    lines.append("")
    lines.append(f"- 数据点数：{inv.n_data}")
    lines.append(f"- 自由参数数：{inv.n_params}")
    lines.append(f"- 自由度 (dof)：{inv.dof}")
    lines.append(f"- 迭代次数：{inv.iterations}")
    lines.append(f"- 最终代价函数：{inv.cost:.6e}")
    lines.append(f"- 卡方 χ²：{inv.chi2:.4f}")
    lines.append(f"- 约化卡方 χ²/dof：{inv.reduced_chi2:.4f}")
    if abs(inv.reduced_chi2 - 1.0) > 0.5 and inv.reduced_chi2 != 0:
        lines.append("  - ⚠️ 约化卡方显著偏离 1，可能原因：量测不确定度估计不准 / 模型偏差 / 数据不足")
    lines.append("")

    # 质量指标
    lines.append("## 4. 模型拟合质量")
    lines.append("")
    lines.append("| 指标 | 训练集 | 测试集 |")
    lines.append("|------|--------|--------|")
    train_keys = [k for k in report.metrics if k.startswith('train_')]
    for tk in sorted(train_keys):
        short = tk.replace('train_', '')
        tv = report.metrics.get(tk, float('nan'))
        kv = report.metrics.get(f'test_{short}', float('nan'))
        tv_str = f"{tv:.4f}" if np.isfinite(tv) else "-"
        kv_str = f"{kv:.4f}" if np.isfinite(kv) else "-"
        lines.append(f"| {short} | {tv_str} | {kv_str} |")
    lines.append("")

    # MCMC 诊断
    if inv.method.value == 'bayesian_mcmc' or inv.mcmc_samples is not None:
        lines.append("## 5. MCMC 诊断")
        lines.append("")
        lines.append(f"- Walker 数：{cfg.mcmc_n_walkers}")
        lines.append(f"- 每 Walker 步数：{cfg.mcmc_n_steps}")
        lines.append(f"- Burn-in：{cfg.mcmc_n_burnin}")
        lines.append(f"- 有效样本数：{len(inv.mcmc_samples) if inv.mcmc_samples is not None else '-'}")
        lines.append(f"- 平均接受率：{inv.mcmc_acceptance*100:.1f}%")
        lines.append("")

    # 图表
    if include_plots and report.plots:
        lines.append("## 6. 可视化结果")
        lines.append("")
        for p in report.plots:
            rel = Path(p).name
            lines.append(f"### {Path(p).stem.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"![{rel}]({rel})")
            lines.append("")

    # 结论
    lines.append("## 7. 结论与建议")
    lines.append("")
    if inv.success:
        lines.append("✅ 反演收敛。")
        if report.metrics.get('train_RMSE_nm', 1e9) < 2.0:
            lines.append("✅ 训练集 RMSE < 2 nm，模型拟合良好。")
        elif report.metrics.get('train_RMSE_nm', 1e9) < 5.0:
            lines.append("⚠️ 训练集 RMSE 在 2~5 nm 范围，需关注具体应用场景。")
        else:
            lines.append("❌ 训练集 RMSE 较大，考虑增加模型复杂度或检查数据质量。")
        if report.metrics.get('train_R2', -1) >= 0.95:
            lines.append("✅ R² ≥ 0.95，解释方差足够。")
        else:
            lines.append("⚠️ R² 偏低，考虑加入更多物理项。")
        if abs(inv.reduced_chi2 - 1.0) > 0.5 and inv.reduced_chi2 != 0:
            lines.append("⚠️ 约化卡方偏离 1，检查数据不确定度设置是否合理。")
    else:
        lines.append("❌ 反演未收敛。建议：")
        lines.append("- 检查输入数据质量")
        lines.append("- 调整参数初始值与边界范围")
        lines.append("- 增大 nlls_max_iter")
        lines.append("- 先运行 LMFIT 获取初值，再切换为 BOTH 方法")
    lines.append("")

    md_content = "\n".join(lines)

    output_path = output_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    logger.info(f"Markdown 报告已写入: {output_path}")
    return md_content


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_calibration_results(dataset: CDSEMDataset,
                              inversion_result: InversionResult,
                              output_dir: Union[str, Path],
                              fmt: str = 'png',
                              filename: str = 'measured_vs_predicted',
                              dpi: int = 150,
                              ) -> Optional[str]:
    """
    测量值 vs 预测值散点图（含 y=x 参考线、R²、MAE）。
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib 未安装，跳过画图")
        return None
    if inversion_result.predicted_cds is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    measured = dataset.measured_cds()
    predicted = inversion_result.predicted_cds

    fig, ax = plt.subplots(figsize=(6.5, 6))
    vmin = min(measured.min(), predicted.min()) - 2
    vmax = max(measured.max(), predicted.max()) + 2
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.2, label='y=x')

    cmap = plt.get_cmap('viridis')
    scatter = ax.scatter(measured, predicted, c=np.abs(measured - predicted),
                         cmap=cmap, s=30, alpha=0.85, edgecolors='none',
                         label='CD-SEM points')
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('|Measured - Predicted| (nm)')

    resid = measured - predicted
    mae = np.mean(np.abs(resid))
    rmse = np.sqrt(np.mean(resid ** 2))
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    txt = f"MAE = {mae:.2f} nm\nRMSE = {rmse:.2f} nm\nR² = {r2:.4f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

    ax.set_xlabel('Measured CD (nm)')
    ax.set_ylabel('Predicted CD (nm)')
    ax.set_title('Measured vs Predicted CD')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()

    out = str(output_dir / f"{filename}.{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"图表已保存: {out}")
    return out


def plot_bossung_curves(dataset: CDSEMDataset,
                         inversion_result: InversionResult,
                         output_dir: Union[str, Path],
                         fmt: str = 'png',
                         filename: str = 'bossung_curves',
                         dpi: int = 150,
                         n_dose_levels: int = 3,
                         ) -> Optional[str]:
    """
    Bossung 曲线：在 (focus, CD) 平面按剂量分层展示测量点与模型拟合曲线。
    """
    if not HAS_MATPLOTLIB:
        return None
    if inversion_result.predicted_cds is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    focuses, doses = dataset.focus_dose_grid()
    measured = dataset.measured_cds()
    target_cds = dataset.target_cds()
    pitches = dataset.pitches()
    pts_list = dataset.pattern_types()
    pts = np.array(pts_list)  # 转为数组便于布尔索引

    # 取 target_cd / pitch / pattern 最常见的组合做主图
    def _most_common_numeric(arr):
        arr = np.asarray(arr)
        vals, counts = np.unique(arr, return_counts=True)
        return vals[np.argmax(counts)]

    def _most_common_enum(arr_list):
        from collections import Counter
        return Counter(arr_list).most_common(1)[0][0]

    ref_target = _most_common_numeric(target_cds)
    ref_pitch = _most_common_numeric(pitches)
    mask_main = (np.abs(target_cds - ref_target) < 0.5) & \
                (np.abs(pitches - ref_pitch) < 0.5)

    if np.sum(mask_main) < 4:
        mask_main = np.ones_like(measured, dtype=bool)
        ref_target = _most_common_numeric(target_cds)
        ref_pitch = _most_common_numeric(pitches)

    fs_main = focuses[mask_main]
    ds_main = doses[mask_main]
    cd_main = measured[mask_main]

    # 选取 n_dose_levels 个有代表性的剂量水平（分位数）
    unique_doses = np.unique(ds_main)
    if len(unique_doses) <= n_dose_levels:
        dose_levels = unique_doses
    else:
        qs = np.linspace(0, 1, n_dose_levels)
        dose_levels = np.quantile(unique_doses, qs)

    # 常见 pattern_type
    pt_subset = pts[mask_main].tolist()
    pt_most = _most_common_enum(pt_subset)

    f_min, f_max = fs_main.min(), fs_main.max()
    f_grid = np.linspace(f_min - 10, f_max + 10, 200)
    params = inversion_result.calibrated_values

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(np.linspace(0.2, 0.8, len(dose_levels)))

    for dose, color in zip(sorted(dose_levels), colors):
        # 模型曲线
        pred_curve = compute_bossung_cd(
            f_grid, np.full_like(f_grid, dose),
            ref_target, ref_pitch, params, pt_most,
            complexity='standard',
        )
        ax.plot(f_grid, pred_curve, '-', color=color, lw=1.8,
                label=f'model dose={dose:.2f}')

        # 该剂量附近的测量点
        band = (np.abs(ds_main - dose) < 0.02)
        if np.any(band):
            ax.plot(fs_main[band], cd_main[band], 'o', ms=7,
                    markerfacecolor='none', markeredgecolor=color,
                    markeredgewidth=1.6,
                    label=f'data dose≈{dose:.2f}')

    ax.axhline(ref_target, color='gray', ls=':', lw=1.2,
               label=f'target CD = {ref_target:.1f} nm')
    ax.axvline(0, color='k', ls=':', lw=0.8)

    ax.set_xlabel('Focus (nm)')
    ax.set_ylabel('CD (nm)')
    ax.set_title(
        f'Bossung Curves (target CD={ref_target:.0f} nm, pitch={ref_pitch:.0f} nm, {pt_most.value})'
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='best')
    fig.tight_layout()

    out = str(output_dir / f"{filename}.{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"图表已保存: {out}")
    return out


def plot_parameter_convergence(inversion_result: InversionResult,
                                output_dir: Union[str, Path],
                                fmt: str = 'png',
                                filename: str = 'parameter_convergence',
                                dpi: int = 150,
                                ) -> Optional[str]:
    """
    MCMC 链迹线图 + 后验直方图，或 NLLS 收敛曲线。
    """
    if not HAS_MATPLOTLIB:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = inversion_result.varying_names
    n = len(names)
    if n == 0:
        return None

    # MCMC 有 samples 时画迹线 + 后验
    if inversion_result.mcmc_samples is not None and len(names) > 0:
        samples = inversion_result.mcmc_samples
        ncols = min(2, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for idx, name in enumerate(names):
            ax = axes[idx]
            vals = samples[:, idx]
            ax.hist(vals, bins=60, density=True, alpha=0.8,
                    color='steelblue', edgecolor='none')
            m = np.mean(vals)
            s = np.std(vals)
            ax.axvline(m, color='r', lw=1.5, label=f'Mean={m:.4f}')
            ax.axvline(m - s, color='r', ls='--', lw=1, label=f'±1σ={s:.4f}')
            ax.axvline(m + s, color='r', ls='--', lw=1)
            ax.set_title(f'{name}')
            ax.set_xlabel(f'{name} value')
            ax.set_ylabel('Density')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle('Posterior Distribution (MCMC Samples)', y=1.02)
        fig.tight_layout()
        out = str(output_dir / f"{filename}.{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"图表已保存: {out}")
        return out

    # NLLS / LMFIT 情况：参数值带误差棒
    else:
        vals = [inversion_result.calibrated_values[n] for n in names]
        uncs = [inversion_result.uncertainties[n] for n in names]
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * n), 4.5))
        x = np.arange(n)
        ax.bar(x, vals, yerr=uncs, capsize=6, color='steelblue',
               edgecolor='k', linewidth=0.8, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel('Calibrated Value ± 1σ')
        ax.set_title('Calibrated Parameters with Uncertainty Bars')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        out = str(output_dir / f"{filename}.{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"图表已保存: {out}")
        return out


def plot_residual_analysis(dataset: CDSEMDataset,
                            inversion_result: InversionResult,
                            output_dir: Union[str, Path],
                            fmt: str = 'png',
                            filename: str = 'residual_analysis',
                            dpi: int = 150,
                            ) -> Optional[str]:
    """
    残差分析四合一图：
    (a) 残差 vs 预测值；(b) 残差直方图 + 正态拟合；
    (c) 残差 vs Focus；(d) 残差 vs Dose
    """
    if not HAS_MATPLOTLIB:
        return None
    if inversion_result.residuals is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = inversion_result.residuals
    predicted = inversion_result.predicted_cds
    focuses, doses = dataset.focus_dose_grid()

    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # (a) Residual vs Predicted
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(predicted, residuals, s=30, alpha=0.7, color='steelblue',
               edgecolors='none')
    ax.axhline(0, color='r', ls='--', lw=1.2)
    ax.set_xlabel('Predicted CD (nm)')
    ax.set_ylabel('Residual (nm)')
    ax.set_title('(a) Residual vs Predicted')
    ax.grid(alpha=0.3)

    # (b) Residual histogram + normal
    ax = fig.add_subplot(gs[0, 1])
    _, bins, _ = ax.hist(residuals, bins=30, density=True, alpha=0.75,
                         color='steelblue', edgecolor='none',
                         label='Residuals')
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    x = np.linspace(bins[0], bins[-1], 200)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=1.8,
            label=f'N({mu:.2f}, {sigma:.2f}²)')
    ax.axvline(0, color='k', ls=':', lw=0.8, label='Zero')
    ax.set_xlabel('Residual (nm)')
    ax.set_ylabel('Density')
    ax.set_title('(b) Residual Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Residual vs Focus
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(focuses, residuals, s=30, alpha=0.7, color='steelblue',
               edgecolors='none')
    ax.axhline(0, color='r', ls='--', lw=1.2)
    if len(focuses) >= 4:
        coeffs = np.polyfit(focuses, residuals, 2)
        f_grid = np.linspace(focuses.min(), focuses.max(), 100)
        ax.plot(f_grid, np.polyval(coeffs, f_grid), 'm-', lw=1.5,
                label='quadratic fit')
        ax.legend(fontsize=8)
    ax.set_xlabel('Focus (nm)')
    ax.set_ylabel('Residual (nm)')
    ax.set_title('(c) Residual vs Focus')
    ax.grid(alpha=0.3)

    # (d) Residual vs Dose
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(doses, residuals, s=30, alpha=0.7, color='steelblue',
               edgecolors='none')
    ax.axhline(0, color='r', ls='--', lw=1.2)
    if len(doses) >= 4:
        coeffs = np.polyfit(doses, residuals, 2)
        d_grid = np.linspace(doses.min(), doses.max(), 100)
        ax.plot(d_grid, np.polyval(coeffs, d_grid), 'm-', lw=1.5,
                label='quadratic fit')
        ax.legend(fontsize=8)
    ax.set_xlabel('Dose (relative)')
    ax.set_ylabel('Residual (nm)')
    ax.set_title('(d) Residual vs Dose')
    ax.grid(alpha=0.3)

    fig.suptitle('Residual Analysis', fontsize=13, y=0.995)

    out = str(output_dir / f"{filename}.{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"图表已保存: {out}")
    return out


class ReportGenerator:
    """
    报告生成器：一次性生成所有输出（文本 + Markdown + 图表）。
    """

    def __init__(self, report: CalibrationReport):
        self.report = report

    def generate_all(self,
                     output_dir: Union[str, Path],
                     dataset: Optional[CDSEMDataset] = None,
                     generate_plots: bool = True,
                     plot_format: str = 'png',
                     ) -> Dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        # 文本摘要
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self.report.summary())
        paths['summary'] = str(summary_path)

        # Markdown 报告（先不插图）
        md_path = generate_markdown_report(
            self.report, output_dir, include_plots=False
        )
        paths['markdown'] = str(output_dir / "calibration_report.md")

        # JSON 结构化数据
        json_path = output_dir / "calibration_result.json"
        json_data = {
            'timestamp': self.report.timestamp,
            'duration_sec': self.report.duration_sec,
            'method': self.report.inversion_result.method.value,
            'success': self.report.inversion_result.success,
            'message': self.report.inversion_result.message,
            'calibrated_values': self.report.inversion_result.calibrated_values,
            'uncertainties': self.report.inversion_result.uncertainties,
            'varying_names': self.report.inversion_result.varying_names,
            'covariance_matrix': (self.report.inversion_result.covariance_matrix.tolist()
                                  if self.report.inversion_result.covariance_matrix is not None
                                  else None),
            'correlation_matrix': (self.report.inversion_result.correlation_matrix.tolist()
                                   if self.report.inversion_result.correlation_matrix is not None
                                   else None),
            'cost': self.report.inversion_result.cost,
            'chi2': self.report.inversion_result.chi2,
            'reduced_chi2': self.report.inversion_result.reduced_chi2,
            'metrics': self.report.metrics,
            'dataset_info': self.report.dataset_info,
        }
        import json as _json
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(json_data, f, indent=2, ensure_ascii=False)
        paths['json'] = str(json_path)

        # 图表
        if generate_plots and dataset is not None:
            plots: List[str] = []
            p = plot_calibration_results(
                dataset, self.report.inversion_result,
                output_dir, fmt=plot_format,
            )
            if p:
                plots.append(p)
            p = plot_bossung_curves(
                dataset, self.report.inversion_result,
                output_dir, fmt=plot_format,
            )
            if p:
                plots.append(p)
            p = plot_parameter_convergence(
                self.report.inversion_result,
                output_dir, fmt=plot_format,
            )
            if p:
                plots.append(p)
            p = plot_residual_analysis(
                dataset, self.report.inversion_result,
                output_dir, fmt=plot_format,
            )
            if p:
                plots.append(p)

            self.report.plots = plots
            paths['plots'] = "\n".join(plots)

            # 重新生成包含图引用的 Markdown
            generate_markdown_report(
                self.report, output_dir, include_plots=True
            )

        return paths
