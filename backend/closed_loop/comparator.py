# -*- coding: utf-8 -*-
"""
仿真预测 vs 量产实测 对比分析模块

对导入的 Fab CD-SEM 数据，使用当前模型参数进行仿真预测，
计算残差统计、按图形类型分组分析、趋势检测，
并基于阈值判断是否需要触发模型重新校准。
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from collections import defaultdict
import logging

from backend.calibration.schemas import (
    CDSEMDataset,
    CDSEMDataPoint,
    CalibrationParameterSet,
    PatternType,
)
from backend.calibration.forward_model import LithoForwardModel

from .schemas import (
    ComparisonResult,
    PerPointComparison,
    PatternGroupStats,
    CalibrationTriggerThresholds,
)

logger = logging.getLogger(__name__)


def _compute_group_stats(residuals: np.ndarray,
                         pattern_type: str,
                         ) -> PatternGroupStats:
    """计算一组残差的统计量"""
    n = len(residuals)
    if n == 0:
        return PatternGroupStats(
            pattern_type=pattern_type,
            n_points=0,
            mean_residual=0.0,
            std_residual=0.0,
            rmse=0.0,
            max_abs_residual=0.0,
            bias_95ci=(0.0, 0.0),
        )

    mean = float(np.mean(residuals))
    std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    max_abs = float(np.max(np.abs(residuals)))

    if n > 1:
        se = std / np.sqrt(n)
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
    else:
        ci_low = ci_high = mean

    return PatternGroupStats(
        pattern_type=pattern_type,
        n_points=n,
        mean_residual=mean,
        std_residual=std,
        rmse=rmse,
        max_abs_residual=max_abs,
        bias_95ci=(float(ci_low), float(ci_high)),
    )


def _detect_trend(residuals: np.ndarray,
                  timestamps: Optional[List[str]] = None,
                  ) -> bool:
    """
    检测残差是否存在显著趋势（单调递增/递减）。

    使用 Mann-Kendall 趋势检验的简化版本：
    计算符号对 (sign(x_j - x_i), j > i) 的一致性。
    """
    n = len(residuals)
    if n < 8:
        return False

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = residuals[j] - residuals[i]
            if diff > 1e-9:
                s += 1
            elif diff < -1e-9:
                s -= 1

    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s < 1e-9:
        return False

    z = (s - 1) / np.sqrt(var_s) if s > 0 else (s + 1) / np.sqrt(var_s)
    return abs(z) > 1.96


class PredictionComparator:
    """
    仿真预测与量产量测对比分析器

    典型用法::

        comparator = PredictionComparator(model_params)
        result = comparator.compare(dataset, thresholds)
        if result.needs_calibration:
            # 触发校准
    """

    def __init__(self,
                 params: Optional[Union[Dict[str, float],
                                       CalibrationParameterSet]] = None,
                 complexity: str = "standard"):
        """
        Args:
            params: 模型参数字典或 CalibrationParameterSet；
                   None 则使用默认参数
            complexity: 前向模型复杂度 ('simple' / 'standard' / 'detailed')
        """
        if params is None:
            params = CalibrationParameterSet()
        self.model = LithoForwardModel(params, complexity=complexity)

    def update_params(self,
                      params: Union[Dict[str, float],
                                    CalibrationParameterSet]) -> None:
        """更新模型参数"""
        if isinstance(params, CalibrationParameterSet):
            self.model = LithoForwardModel(params, complexity=self.model.complexity)
        else:
            self.model._params.update(params)

    @property
    def current_params(self) -> Dict[str, float]:
        return self.model.params

    # ------------------------------------------------------------------
    # 核心对比方法
    # ------------------------------------------------------------------
    def compare(self,
                dataset: CDSEMDataset,
                thresholds: Optional[CalibrationTriggerThresholds] = None,
                ) -> ComparisonResult:
        """
        对整个数据集执行对比分析

        Args:
            dataset: CD-SEM 量测数据集
            thresholds: 校准触发阈值；None 则使用默认值，且不自动判定 needs_calibration

        Returns:
            ComparisonResult
        """
        if len(dataset) == 0:
            logger.warning("数据集为空，跳过对比分析")
            return ComparisonResult()

        logger.info(f"开始对比分析: {len(dataset)} 个量测点")

        focuses, doses = dataset.focus_dose_grid()
        target_cds = dataset.target_cds()
        pitches = dataset.pitches()
        measured_cds = dataset.measured_cds()
        pattern_types = dataset.pattern_types()

        predicted_cds = np.zeros(len(dataset), dtype=np.float64)
        unique_pt = list(set(pattern_types))
        for pt in unique_pt:
            mask = np.array([p == pt for p in pattern_types], dtype=bool)
            predicted_cds[mask] = self.model.predict(
                focuses[mask], doses[mask], target_cds[mask],
                pitches[mask], pt,
            )

        residuals = measured_cds - predicted_cds
        abs_residuals = np.abs(residuals)

        per_point: List[PerPointComparison] = []
        for i, pt_data in enumerate(dataset.points):
            rel_err = abs_residuals[i] / max(abs(pt_data.target_cd), 1e-6)
            per_point.append(PerPointComparison(
                measurement_id=pt_data.measurement_id,
                target_cd=float(pt_data.target_cd),
                measured_cd=float(pt_data.measured_cd),
                predicted_cd=float(predicted_cds[i]),
                residual=float(residuals[i]),
                relative_error=float(rel_err),
                focus=float(pt_data.focus),
                dose=float(pt_data.dose),
                pattern_type=pt_data.pattern_type.value,
                site_name=pt_data.site_name,
                layer=pt_data.layer,
            ))

        group_stats: List[PatternGroupStats] = []
        groups: Dict[str, List[float]] = defaultdict(list)
        for i, pt in enumerate(pattern_types):
            groups[pt.value].append(float(residuals[i]))
        for pt_name, res_list in groups.items():
            group_stats.append(_compute_group_stats(
                np.array(res_list), pt_name
            ))

        mean_residual = float(np.mean(residuals))
        std_residual = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        max_abs = float(np.max(abs_residuals))
        median_abs = float(np.median(abs_residuals))
        mean_target = float(np.mean(target_cds))
        relative_rmse = rmse / max(mean_target, 1e-6)

        timestamps = [p.timestamp for p in dataset.points if p.timestamp]
        has_trend = _detect_trend(residuals, timestamps if timestamps else None)

        reasons: List[str] = []
        needs_calib = False
        if thresholds is not None:
            if len(dataset) >= thresholds.min_points_required:
                if rmse > thresholds.rmse_threshold_nm:
                    reasons.append(
                        f"RMSE={rmse:.2f} nm > 阈值 {thresholds.rmse_threshold_nm:.2f} nm"
                    )
                if abs(mean_residual) > thresholds.bias_threshold_nm:
                    reasons.append(
                        f"平均偏差 |{mean_residual:.2f}| nm > 阈值 {thresholds.bias_threshold_nm:.2f} nm"
                    )
                if max_abs > thresholds.max_residual_threshold_nm:
                    reasons.append(
                        f"最大|残差|={max_abs:.2f} nm > 阈值 {thresholds.max_residual_threshold_nm:.2f} nm"
                    )
                if relative_rmse > thresholds.relative_rmse_threshold:
                    reasons.append(
                        f"相对 RMSE={relative_rmse * 100:.2f}% > 阈值 {thresholds.relative_rmse_threshold * 100:.1f}%"
                    )
                for gs in group_stats:
                    if abs(gs.mean_residual) > thresholds.group_bias_threshold_nm:
                        reasons.append(
                            f"图形组 [{gs.pattern_type}] 偏差 {gs.mean_residual:+.2f} nm > ±{thresholds.group_bias_threshold_nm:.2f} nm"
                        )
                needs_calib = len(reasons) > 0
            else:
                reasons.append(
                    f"数据点不足 ({len(dataset)} < {thresholds.min_points_required})"
                )

        result = ComparisonResult(
            n_points=len(dataset),
            mean_residual=mean_residual,
            std_residual=std_residual,
            rmse=rmse,
            max_abs_residual=max_abs,
            median_abs_residual=median_abs,
            relative_rmse=relative_rmse,
            per_point=per_point,
            pattern_groups=group_stats,
            trend_detected=has_trend,
            needs_calibration=needs_calib,
            calibration_reasons=reasons,
        )

        logger.info(
            f"对比分析完成: RMSE={rmse:.3f} nm, "
            f"bias={mean_residual:+.3f}±{std_residual:.3f} nm, "
            f"建议校准={'是' if needs_calib else '否'}"
        )
        return result

    # ------------------------------------------------------------------
    # 可视化辅助
    # ------------------------------------------------------------------
    def plot_comparison(self,
                        result: ComparisonResult,
                        save_path: Optional[Union[str, Path]] = None,
                        show: bool = True,
                        ) -> Any:
        """
        绘制对比结果可视化（残差分布、Q-Q 图、分组柱状图）。
        需安装 matplotlib。
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib 未安装，跳过绘图")
            return None

        if result.n_points == 0:
            return None

        residuals = np.array([p.residual for p in result.per_point])
        measured = np.array([p.measured_cd for p in result.per_point])
        predicted = np.array([p.predicted_cd for p in result.per_point])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(x=result.mean_residual, color='blue', linestyle='-',
                   linewidth=1.5, label=f'mean={result.mean_residual:+.2f}')
        ax.set_xlabel('Residual (measured - predicted) [nm]')
        ax.set_ylabel('Count')
        ax.set_title('Residual Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        min_cd = min(measured.min(), predicted.min())
        max_cd = max(measured.max(), predicted.max())
        ax.scatter(measured, predicted, alpha=0.6, s=20, edgecolors='white')
        ax.plot([min_cd, max_cd], [min_cd, max_cd], 'r--',
                linewidth=1.5, label='y = x')
        ax.set_xlabel('Measured CD [nm]')
        ax.set_ylabel('Predicted CD [nm]')
        ax.set_title(f'Parity Plot (RMSE={result.rmse:.2f} nm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        ax = axes[1, 0]
        if result.pattern_groups:
            names = [g.pattern_type for g in result.pattern_groups]
            means = [g.mean_residual for g in result.pattern_groups]
            stds = [g.std_residual for g in result.pattern_groups]
            x_pos = np.arange(len(names))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                   edgecolor='black')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(names, rotation=30, ha='right')
            ax.set_ylabel('Mean Residual ± σ [nm]')
            ax.set_title('Residuals by Pattern Type')
            ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1, 1]
        sorted_res = np.sort(residuals)
        n = len(sorted_res)
        theoretical = np.array([
            np.percentile(np.random.standard_normal(100000),
                          (i + 0.5) / n * 100)
            for i in range(n)
        ])
        ax.scatter(theoretical, sorted_res, alpha=0.6, s=20)
        ax.plot(theoretical, theoretical, 'r--', linewidth=1.5)
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title('Q-Q Plot of Residuals')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"对比图已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


def compare_prediction_vs_measurement(
    dataset: CDSEMDataset,
    params: Optional[Union[Dict[str, float], CalibrationParameterSet]] = None,
    thresholds: Optional[CalibrationTriggerThresholds] = None,
    complexity: str = "standard",
) -> ComparisonResult:
    """
    便捷函数：执行仿真预测 vs 量产量测对比

    Args:
        dataset: CD-SEM 数据集
        params: 模型参数；None 则使用默认
        thresholds: 校准触发阈值；None 则用默认值
        complexity: 前向模型复杂度

    Returns:
        ComparisonResult
    """
    comparator = PredictionComparator(params, complexity=complexity)
    return comparator.compare(dataset, thresholds=thresholds)
