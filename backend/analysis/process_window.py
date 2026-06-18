# -*- coding: utf-8 -*-
"""
工艺窗口 (Process Window) 分析模块

在 focus-dose 二维参数空间内批量仿真，计算每个 (focus, dose) 条件下的
CD 误差、EPE 或是否可打印，输出工艺窗口面积与边界。

功能包括：
1. 参数扫描器：接收 focus 范围/步长、dose 范围/步长，调用现有成像模块批量计算
2. 可打印性判定：基于 EPE/CD 容差判断每个点是否可打印
3. PW 指标计算：PW 面积、中心点、椭圆/矩形近似 PW、最佳 focus-dose 点
4. 可视化：Bossung 曲线、PW 等高线图、与 visualization.py 集成
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time

from core.imaging import (
    OpticalSystem,
    ProcessCondition,
    ProcessWindow as ImagingProcessWindow,
    simulate_multi_process,
    MultiProcessSimulationResult,
    create_focus_dose_window,
)
from core.litho_metrics import (
    compute_cd,
    compute_cd_error,
    compute_epe,
    compute_ils,
    compute_nils,
    compute_process_window_area,
    extract_process_window_scan,
    ProcessWindowScanResult,
)
from core.metrics import mse, ssim

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """
    蒙特卡洛采样配置

    Attributes:
        n_samples: 采样数量
        focus_std: 离焦量标准差 (nm)
        dose_std: 曝光剂量相对标准差（相对于标称剂量1.0的比例）
        aberration_std: 像差系数标准差（单位: 波长λ），可以是：
            - float: 所有 Zernike 系数使用相同标准差
            - Dict[int, float]: 按 Zernike 索引指定标准差
            - None: 不考虑像差随机扰动
        zernike_indices: 要考虑随机扰动的 Zernike 系数索引列表，
                         None 则使用光学系统中已有的所有非零系数
        nominal_focus: 标称离焦量 (nm)
        nominal_dose: 标称剂量
        random_seed: 随机种子，用于结果复现
        distribution: 分布类型，'normal' 或 'uniform'
    """
    n_samples: int = 100
    focus_std: float = 30.0
    dose_std: float = 0.05
    aberration_std: Optional[Union[float, Dict[int, float]]] = None
    zernike_indices: Optional[List[int]] = None
    nominal_focus: float = 0.0
    nominal_dose: float = 1.0
    random_seed: Optional[int] = None
    distribution: str = 'normal'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_samples': self.n_samples,
            'focus_std': self.focus_std,
            'dose_std': self.dose_std,
            'aberration_std': self.aberration_std,
            'zernike_indices': self.zernike_indices,
            'nominal_focus': self.nominal_focus,
            'nominal_dose': self.nominal_dose,
            'random_seed': self.random_seed,
            'distribution': self.distribution,
        }


@dataclass
class MonteCarloMetricStats:
    """
    蒙特卡洛仿真的单个指标统计结果

    Attributes:
        mean: 均值
        std: 标准差
        variance: 方差
        min: 最小值
        max: 最大值
        median: 中位数
        percentile_5: 5% 分位数（悲观估计）
        percentile_95: 95% 分位数（乐观估计）
        worst_case: 最坏情况值（基于指标类型，越大越差或越小越差）
        samples: 所有采样值（可选保存）
    """
    mean: float
    std: float
    variance: float
    min: float
    max: float
    median: float
    percentile_5: float
    percentile_95: float
    worst_case: float
    samples: Optional[np.ndarray] = None

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        result = {
            'mean': self.mean,
            'std': self.std,
            'variance': self.variance,
            'min': self.min,
            'max': self.max,
            'median': self.median,
            'percentile_5': self.percentile_5,
            'percentile_95': self.percentile_95,
            'worst_case': self.worst_case,
        }
        if include_samples and self.samples is not None:
            result['samples'] = self.samples.tolist()
        return result


@dataclass
class MonteCarloResult:
    """
    蒙特卡洛工艺鲁棒性评估结果

    Attributes:
        n_samples: 采样数量
        config: 蒙特卡洛配置
        cd_stats: CD 统计 (nm)
        cd_error_stats: CD 误差统计 (nm)
        epe_stats: EPE 统计 (nm)
        mse_stats: MSE 统计
        ssim_stats: SSIM 统计
        ils_stats: ILS 统计
        nils_stats: NILS 统计
        sampled_conditions: 采样的工艺条件列表
        passing_ratio: 可打印条件比例（基于 CD 容差）
        worst_condition: 最坏工艺条件（CD 误差最大的条件）
    """
    n_samples: int
    config: MonteCarloConfig
    cd_stats: MonteCarloMetricStats
    cd_error_stats: MonteCarloMetricStats
    epe_stats: MonteCarloMetricStats
    mse_stats: MonteCarloMetricStats
    ssim_stats: MonteCarloMetricStats
    ils_stats: MonteCarloMetricStats
    nils_stats: MonteCarloMetricStats
    sampled_conditions: List[ProcessCondition]
    passing_ratio: float
    worst_condition: ProcessCondition

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        return {
            'n_samples': self.n_samples,
            'config': self.config.to_dict(),
            'cd_stats': self.cd_stats.to_dict(include_samples=include_samples),
            'cd_error_stats': self.cd_error_stats.to_dict(include_samples=include_samples),
            'epe_stats': self.epe_stats.to_dict(include_samples=include_samples),
            'mse_stats': self.mse_stats.to_dict(include_samples=include_samples),
            'ssim_stats': self.ssim_stats.to_dict(include_samples=include_samples),
            'ils_stats': self.ils_stats.to_dict(include_samples=include_samples),
            'nils_stats': self.nils_stats.to_dict(include_samples=include_samples),
            'passing_ratio': self.passing_ratio,
            'worst_condition': self.worst_condition.to_dict(),
        }

    def summary(self) -> str:
        lines = [
            "=== 蒙特卡洛工艺鲁棒性评估结果 ===",
            f"  采样数量: {self.n_samples}",
            f"  分布类型: {self.config.distribution}",
            f"  Focus 标准差: {self.config.focus_std:.1f} nm",
            f"  Dose 相对标准差: {self.config.dose_std * 100:.1f}%",
            "",
            "  CD (nm):",
            f"    均值: {self.cd_stats.mean:.2f}, 标准差: {self.cd_stats.std:.2f}",
            f"    范围: [{self.cd_stats.min:.2f}, {self.cd_stats.max:.2f}]",
            f"    最坏情况: {self.cd_stats.worst_case:.2f}",
            "",
            "  CD 误差 (nm):",
            f"    均值: {self.cd_error_stats.mean:.2f}, 标准差: {self.cd_error_stats.std:.2f}",
            f"    范围: [{self.cd_error_stats.min:.2f}, {self.cd_error_stats.max:.2f}]",
            f"    最坏情况: {self.cd_error_stats.worst_case:.2f}",
            "",
            "  EPE (nm):",
            f"    均值: {self.epe_stats.mean:.2f}, 标准差: {self.epe_stats.std:.2f}",
            f"    最坏情况: {self.epe_stats.worst_case:.2f}",
            "",
            "  MSE:",
            f"    均值: {self.mse_stats.mean:.6f}, 标准差: {self.mse_stats.std:.6f}",
            f"    最坏情况: {self.mse_stats.worst_case:.6f}",
            "",
            "  SSIM:",
            f"    均值: {self.ssim_stats.mean:.4f}, 标准差: {self.ssim_stats.std:.4f}",
            f"    最坏情况: {self.ssim_stats.worst_case:.4f}",
            "",
            f"  可打印比例: {self.passing_ratio * 100:.1f}%",
        ]
        return "\n".join(lines)


@dataclass
class PrintabilityResult:
    """
    可打印性判定结果

    Attributes:
        passing_mask: 布尔矩阵 (n_focus, n_dose)，True 表示该工艺条件可打印
        n_passing: 可打印条件数量
        n_total: 总条件数量
        passing_ratio: 可打印比例
        focus_range_passing: 可打印条件的离焦范围 (min, max) nm
        dose_range_passing: 可打印条件的剂量范围 (min, max)
    """
    passing_mask: np.ndarray
    n_passing: int
    n_total: int
    passing_ratio: float
    focus_range_passing: Tuple[float, float]
    dose_range_passing: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_passing': self.n_passing,
            'n_total': self.n_total,
            'passing_ratio': self.passing_ratio,
            'focus_range_passing': self.focus_range_passing,
            'dose_range_passing': self.dose_range_passing,
        }


@dataclass
class EllipseApprox:
    """
    工艺窗口椭圆近似

    Attributes:
        center_focus: 椭圆中心 focus (nm)
        center_dose: 椭圆中心 dose
        semi_axis_focus: focus 方向半轴 (nm)
        semi_axis_dose: dose 方向半轴
        angle: 旋转角度 (度)
        area: 椭圆面积 (nm * dose)
    """
    center_focus: float
    center_dose: float
    semi_axis_focus: float
    semi_axis_dose: float
    angle: float
    area: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'center_focus': self.center_focus,
            'center_dose': self.center_dose,
            'semi_axis_focus': self.semi_axis_focus,
            'semi_axis_dose': self.semi_axis_dose,
            'angle': self.angle,
            'area': self.area,
        }


@dataclass
class RectApprox:
    """
    工艺窗口矩形近似

    Attributes:
        center_focus: 矩形中心 focus (nm)
        center_dose: 矩形中心 dose
        half_width_focus: focus 方向半宽 (nm)
        half_width_dose: dose 方向半宽
        area: 矩形面积 (nm * dose)
        focus_min: focus 下界 (nm)
        focus_max: focus 上界 (nm)
        dose_min: dose 下界
        dose_max: dose 上界
    """
    center_focus: float
    center_dose: float
    half_width_focus: float
    half_width_dose: float
    area: float
    focus_min: float
    focus_max: float
    dose_min: float
    dose_max: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'center_focus': self.center_focus,
            'center_dose': self.center_dose,
            'half_width_focus': self.half_width_focus,
            'half_width_dose': self.half_width_dose,
            'area': self.area,
            'focus_min': self.focus_min,
            'focus_max': self.focus_max,
            'dose_min': self.dose_min,
            'dose_max': self.dose_max,
        }


@dataclass
class PWMetrics:
    """
    工艺窗口综合指标

    Attributes:
        pw_area: 工艺窗口面积 (nm * dose)
        pw_ratio: 工艺窗口面积占扫描总面积的比例
        n_passing: 通过条件的数量
        n_total: 总条件数量
        center_focus: 工艺窗口中心 focus (nm)
        center_dose: 工艺窗口中心 dose
        best_focus: 最佳 focus (nm)，CD 误差最小时的 focus
        best_dose: 最佳 dose，CD 误差最小时的 dose
        best_cd_error: 最佳条件下的 CD 误差 (nm)
        focus_range: 通过条件的离焦范围 (min, max) nm
        dose_range: 通过条件的剂量范围 (min, max)
        ellipse_approx: 椭圆近似结果
        rect_approx: 矩形近似结果
        depth_of_focus: 焦深 (nm)，dose=nominal 时的可接受 focus 范围
        exposure_latitude: 曝光宽容度 (%)，focus=0 时的可接受剂量范围
    """
    pw_area: float
    pw_ratio: float
    n_passing: int
    n_total: int
    center_focus: float
    center_dose: float
    best_focus: float
    best_dose: float
    best_cd_error: float
    focus_range: Tuple[float, float]
    dose_range: Tuple[float, float]
    ellipse_approx: Optional[EllipseApprox] = None
    rect_approx: Optional[RectApprox] = None
    depth_of_focus: float = 0.0
    exposure_latitude: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'pw_area': self.pw_area,
            'pw_ratio': self.pw_ratio,
            'n_passing': self.n_passing,
            'n_total': self.n_total,
            'center_focus': self.center_focus,
            'center_dose': self.center_dose,
            'best_focus': self.best_focus,
            'best_dose': self.best_dose,
            'best_cd_error': self.best_cd_error,
            'focus_range': self.focus_range,
            'dose_range': self.dose_range,
            'depth_of_focus': self.depth_of_focus,
            'exposure_latitude': self.exposure_latitude,
        }
        if self.ellipse_approx is not None:
            result['ellipse_approx'] = self.ellipse_approx.to_dict()
        if self.rect_approx is not None:
            result['rect_approx'] = self.rect_approx.to_dict()
        return result

    def summary(self) -> str:
        lines = [
            "=== 工艺窗口 (Process Window) 分析结果 ===",
            f"  PW 面积: {self.pw_area:.2f} nm·dose",
            f"  PW 比例: {self.pw_ratio * 100:.1f}%",
            f"  通过条件: {self.n_passing}/{self.n_total}",
            f"  PW 中心: focus={self.center_focus:.1f} nm, dose={self.center_dose:.4f}",
            f"  最佳条件: focus={self.best_focus:.1f} nm, dose={self.best_dose:.4f} "
            f"(CD误差={self.best_cd_error:.2f} nm)",
            f"  Focus 范围: [{self.focus_range[0]:.1f}, {self.focus_range[1]:.1f}] nm",
            f"  Dose 范围: [{self.dose_range[0]:.4f}, {self.dose_range[1]:.4f}]",
            f"  焦深 (DOF): {self.depth_of_focus:.1f} nm",
            f"  曝光宽容度 (EL): {self.exposure_latitude:.2f}%",
        ]
        if self.ellipse_approx is not None:
            lines.append(
                f"  椭圆近似: center=({self.ellipse_approx.center_focus:.1f}, "
                f"{self.ellipse_approx.center_dose:.4f}), "
                f"半轴=({self.ellipse_approx.semi_axis_focus:.1f}, "
                f"{self.ellipse_approx.semi_axis_dose:.4f}), "
                f"面积={self.ellipse_approx.area:.2f}"
            )
        if self.rect_approx is not None:
            lines.append(
                f"  矩形近似: center=({self.rect_approx.center_focus:.1f}, "
                f"{self.rect_approx.center_dose:.4f}), "
                f"半宽=({self.rect_approx.half_width_focus:.1f}, "
                f"{self.rect_approx.half_width_dose:.4f}), "
                f"面积={self.rect_approx.area:.2f}"
            )
        return "\n".join(lines)


class ProcessWindowAnalyzer:
    """
    工艺窗口分析器

    对给定掩模在 focus-dose 二维参数空间内批量仿真，计算每个
    (focus, dose) 条件下的 CD 误差、EPE 或是否可打印，
    输出工艺窗口面积与边界。

    使用方式::

        analyzer = ProcessWindowAnalyzer(mask, target, optics)
        scan_result = analyzer.scan(focus_range=(-150, 150, 11),
                                     dose_range=(0.85, 1.15, 11))
        printability = analyzer.judge_printability(cd_tolerance=0.1)
        metrics = analyzer.compute_pw_metrics()
        fig_bossung = analyzer.plot_bossung(save_path='bossung.png', show=False)
        fig_pw = analyzer.plot_pw_contour(save_path='pw_contour.png', show=False)
    """

    def __init__(
        self,
        mask: np.ndarray,
        target: np.ndarray,
        optical_system: Optional[OpticalSystem] = None,
        threshold: float = 0.3,
        pixel_size: float = 1.0,
        resist_model: Optional[Any] = None,
        window_type: Optional[Any] = None,
        pad_width: Optional[Union[int, Tuple[int, int]]] = None,
        tukey_alpha: float = 0.5,
    ):
        """
        初始化工艺窗口分析器

        Args:
            mask: 掩模图案 (2D numpy 数组, 值范围 [0, 1])
            target: 目标图案 (2D numpy 数组, 二值化)
            optical_system: 光学系统参数，None 则使用默认参数
            threshold: 光刻胶阈值
            pixel_size: 像素尺寸 (nm)
            resist_model: 光刻胶模型配置
            window_type: 窗函数类型
            pad_width: 零填充宽度
            tukey_alpha: Tukey 窗渐变比例因子
        """
        self.mask = mask.astype(np.float64)
        self.target = target.astype(np.float64)
        self.optical_system = optical_system if optical_system is not None else OpticalSystem()
        self.threshold = threshold
        self.pixel_size = pixel_size
        self.resist_model = resist_model
        self.window_type = window_type
        self.pad_width = pad_width
        self.tukey_alpha = tukey_alpha

        self._scan_result: Optional[ProcessWindowScanResult] = None
        self._multi_result: Optional[MultiProcessSimulationResult] = None
        self._cd_target: Optional[float] = None

    @property
    def scan_result(self) -> Optional[ProcessWindowScanResult]:
        """获取最近一次扫描结果"""
        return self._scan_result

    @property
    def cd_target(self) -> float:
        """获取目标 CD"""
        if self._cd_target is None:
            self._cd_target = compute_cd(
                self.target, pixel_size=self.pixel_size
            )['cd_mean']
        return self._cd_target

    def scan(
        self,
        focus_range: Union[Tuple[float, float, int], List[float], np.ndarray] = (-150, 150, 11),
        dose_range: Union[Tuple[float, float, int], List[float], np.ndarray] = (0.85, 1.15, 11),
        cd_target: Optional[float] = None,
        cd_tolerance: float = 0.1,
        progress_callback: Optional[Any] = None,
    ) -> ProcessWindowScanResult:
        """
        参数扫描：在 focus-dose 二维空间内批量仿真

        Args:
            focus_range: 离焦量扫描范围。
                - 元组 (start, stop, num): 使用 np.linspace 生成
                - 列表/数组: 直接使用
            dose_range: 曝光剂量扫描范围，格式同上
            cd_target: 目标 CD (nm)；None 则从目标图自动测量
            cd_tolerance: CD 相对容差，用于判定可打印区域
            progress_callback: 进度回调函数 callback(current, total)

        Returns:
            ProcessWindowScanResult，包含各指标的二维矩阵
        """
        if cd_target is not None:
            self._cd_target = cd_target

        conditions = create_focus_dose_window(
            focus_range=focus_range,
            dose_range=dose_range,
            na=self.optical_system.na,
            sigma=self.optical_system.sigma,
            wavelength=self.optical_system.wavelength,
            center_weight=None,
        )

        logger.info(
            f"开始工艺窗口扫描: {len(conditions)} 个条件, "
            f"focus_range={focus_range}, dose_range={dose_range}"
        )

        t_start = time.time()

        self._multi_result = simulate_multi_process(
            mask=self.mask,
            conditions=conditions,
            base_optics=self.optical_system,
            threshold=self.threshold,
            apply_resist=True,
            resist_model=self.resist_model,
            window_type=self.window_type,
            pad_width=self.pad_width,
            tukey_alpha=self.tukey_alpha,
        )

        self._scan_result = extract_process_window_scan(
            multi_result=self._multi_result,
            target_binary=self.target,
            cd_target=self.cd_target,
            cd_tolerance=cd_tolerance,
            pixel_size=self.pixel_size,
            threshold=self.threshold,
        )

        elapsed = time.time() - t_start
        logger.info(f"工艺窗口扫描完成, 耗时 {elapsed:.1f}s")

        return self._scan_result

    def judge_printability(
        self,
        cd_tolerance: float = 0.1,
        epe_tolerance: Optional[float] = None,
        cd_target: Optional[float] = None,
    ) -> PrintabilityResult:
        """
        可打印性判定

        基于 CD 容差和可选的 EPE 容差判断每个 (focus, dose) 点是否可打印。

        Args:
            cd_tolerance: CD 相对容差，|CD - CD_target| <= cd_tolerance * CD_target 为通过
            epe_tolerance: EPE 绝对容差 (nm)，EPE <= epe_tolerance 为通过；None 则不检查 EPE
            cd_target: 目标 CD (nm)；None 则使用 scan() 中自动测量的值

        Returns:
            PrintabilityResult，包含可打印性判定结果
        """
        if self._scan_result is None:
            raise RuntimeError("请先调用 scan() 方法进行参数扫描")

        cd_tgt = cd_target if cd_target is not None else self.cd_target
        cd_lower = cd_tgt * (1.0 - cd_tolerance)
        cd_upper = cd_tgt * (1.0 + cd_tolerance)

        cd_passing = (self._scan_result.cd_matrix >= cd_lower) & \
                     (self._scan_result.cd_matrix <= cd_upper)

        if epe_tolerance is not None:
            epe_passing = self._scan_result.epe_matrix <= epe_tolerance
            valid_epe = ~np.isnan(self._scan_result.epe_matrix)
            epe_passing = epe_passing & valid_epe
            passing_mask = cd_passing & epe_passing
        else:
            passing_mask = cd_passing

        n_passing = int(np.sum(passing_mask))
        n_total = int(np.prod(passing_mask.shape))
        passing_ratio = n_passing / n_total if n_total > 0 else 0.0

        focus = self._scan_result.unique_focus
        dose = self._scan_result.unique_dose

        if n_passing > 0:
            passing_indices = np.argwhere(passing_mask)
            fi_min, fi_max = passing_indices[:, 0].min(), passing_indices[:, 0].max()
            di_min, di_max = passing_indices[:, 1].min(), passing_indices[:, 1].max()
            focus_range_passing = (float(focus[fi_min]), float(focus[fi_max]))
            dose_range_passing = (float(dose[di_min]), float(dose[di_max]))
        else:
            focus_range_passing = (0.0, 0.0)
            dose_range_passing = (0.0, 0.0)

        return PrintabilityResult(
            passing_mask=passing_mask,
            n_passing=n_passing,
            n_total=n_total,
            passing_ratio=passing_ratio,
            focus_range_passing=focus_range_passing,
            dose_range_passing=dose_range_passing,
        )

    def compute_pw_metrics(
        self,
        cd_tolerance: float = 0.1,
        epe_tolerance: Optional[float] = None,
        cd_target: Optional[float] = None,
    ) -> PWMetrics:
        """
        计算 PW 综合指标

        包括 PW 面积、中心点、椭圆/矩形近似、最佳 focus-dose 点等。

        Args:
            cd_tolerance: CD 相对容差
            epe_tolerance: EPE 绝对容差 (nm)
            cd_target: 目标 CD (nm)

        Returns:
            PWMetrics，包含工艺窗口综合指标
        """
        if self._scan_result is None:
            raise RuntimeError("请先调用 scan() 方法进行参数扫描")

        cd_tgt = cd_target if cd_target is not None else self.cd_target

        printability = self.judge_printability(
            cd_tolerance=cd_tolerance,
            epe_tolerance=epe_tolerance,
            cd_target=cd_tgt,
        )

        conditions = self._multi_result.conditions
        pw_area_result = compute_process_window_area(
            conditions=conditions,
            cd_values=self._scan_result.cd_matrix.flatten(),
            cd_target=cd_tgt,
            cd_tolerance=cd_tolerance,
        )

        focus = self._scan_result.unique_focus
        dose = self._scan_result.unique_dose

        passing_mask = printability.passing_mask

        if printability.n_passing > 0:
            passing_focus = []
            passing_dose = []
            for i in range(len(focus)):
                for j in range(len(dose)):
                    if passing_mask[i, j]:
                        passing_focus.append(focus[i])
                        passing_dose.append(dose[j])

            center_focus = float(np.mean(passing_focus))
            center_dose = float(np.mean(passing_dose))
            focus_range = (float(min(passing_focus)), float(max(passing_focus)))
            dose_range = (float(min(passing_dose)), float(max(passing_dose)))
        else:
            center_focus = float(np.mean(focus))
            center_dose = float(np.mean(dose))
            focus_range = (0.0, 0.0)
            dose_range = (0.0, 0.0)

        cd_error_matrix = self._scan_result.cd_error_matrix
        valid_mask = ~np.isnan(cd_error_matrix)
        if np.any(valid_mask):
            abs_cd_error = np.abs(cd_error_matrix)
            best_idx = np.unravel_index(
                np.nanargmin(abs_cd_error), abs_cd_error.shape
            )
            best_focus = float(focus[best_idx[0]])
            best_dose = float(dose[best_idx[1]])
            best_cd_error = float(cd_error_matrix[best_idx[0], best_idx[1]])
        else:
            best_focus = center_focus
            best_dose = center_dose
            best_cd_error = 0.0

        rect_approx = self._compute_rect_approx(
            focus, dose, passing_mask, center_focus, center_dose
        )

        ellipse_approx = self._compute_ellipse_approx(
            focus, dose, passing_mask, center_focus, center_dose
        )

        depth_of_focus = self._compute_depth_of_focus(
            focus, dose, passing_mask
        )
        exposure_latitude = self._compute_exposure_latitude(
            focus, dose, passing_mask
        )

        return PWMetrics(
            pw_area=pw_area_result['pw_area'],
            pw_ratio=pw_area_result['pw_ratio'],
            n_passing=printability.n_passing,
            n_total=printability.n_total,
            center_focus=center_focus,
            center_dose=center_dose,
            best_focus=best_focus,
            best_dose=best_dose,
            best_cd_error=best_cd_error,
            focus_range=focus_range,
            dose_range=dose_range,
            ellipse_approx=ellipse_approx,
            rect_approx=rect_approx,
            depth_of_focus=depth_of_focus,
            exposure_latitude=exposure_latitude,
        )

    def _compute_rect_approx(
        self,
        focus: np.ndarray,
        dose: np.ndarray,
        passing_mask: np.ndarray,
        center_focus: float,
        center_dose: float,
    ) -> Optional[RectApprox]:
        """
        计算工艺窗口矩形近似

        使用通过条件的 bounding box 作为矩形近似。
        """
        if not np.any(passing_mask):
            return None

        passing_indices = np.argwhere(passing_mask)
        fi_min, fi_max = passing_indices[:, 0].min(), passing_indices[:, 0].max()
        di_min, di_max = passing_indices[:, 1].min(), passing_indices[:, 1].max()

        focus_min = float(focus[fi_min])
        focus_max = float(focus[fi_max])
        dose_min = float(dose[di_min])
        dose_max = float(dose[di_max])

        half_width_focus = (focus_max - focus_min) / 2.0
        half_width_dose = (dose_max - dose_min) / 2.0

        area = (focus_max - focus_min) * (dose_max - dose_min)

        return RectApprox(
            center_focus=center_focus,
            center_dose=center_dose,
            half_width_focus=half_width_focus,
            half_width_dose=half_width_dose,
            area=area,
            focus_min=focus_min,
            focus_max=focus_max,
            dose_min=dose_min,
            dose_max=dose_max,
        )

    def _compute_ellipse_approx(
        self,
        focus: np.ndarray,
        dose: np.ndarray,
        passing_mask: np.ndarray,
        center_focus: float,
        center_dose: float,
    ) -> Optional[EllipseApprox]:
        """
        计算工艺窗口椭圆近似

        使用通过条件点的二阶矩（协方差矩阵）拟合椭圆。
        椭圆半轴由协方差矩阵的特征值决定，使 95% 的通过点落在椭圆内
        （使用 chi2(0.95, df=2) ≈ 5.991 的缩放因子）。
        """
        if not np.any(passing_mask):
            return None

        passing_indices = np.argwhere(passing_mask)
        pf = focus[passing_indices[:, 0]].astype(np.float64)
        pd = dose[passing_indices[:, 1]].astype(np.float64)

        if len(pf) < 3:
            return None

        focus_span = focus.max() - focus.min()
        dose_span = dose.max() - dose.min()
        if focus_span < 1e-12 or dose_span < 1e-12:
            return None

        pf_norm = (pf - center_focus) / focus_span
        pd_norm = (pd - center_dose) / dose_span

        cov = np.cov(pf_norm, pd_norm)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        eigenvalues = np.maximum(eigenvalues, 1e-12)

        chi2_scale = 5.991
        semi_axes_norm = np.sqrt(eigenvalues * chi2_scale)

        semi_axis_focus = semi_axes_norm[1] * focus_span
        semi_axis_dose = semi_axes_norm[0] * dose_span

        angle_rad = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
        angle_deg = float(np.degrees(angle_rad))

        area = np.pi * semi_axis_focus * semi_axis_dose

        return EllipseApprox(
            center_focus=center_focus,
            center_dose=center_dose,
            semi_axis_focus=semi_axis_focus,
            semi_axis_dose=semi_axis_dose,
            angle=angle_deg,
            area=area,
        )

    def _compute_depth_of_focus(
        self,
        focus: np.ndarray,
        dose: np.ndarray,
        passing_mask: np.ndarray,
    ) -> float:
        """
        计算焦深 (Depth of Focus, DOF)

        DOF 定义为：在标称剂量（最接近 1.0 的剂量列）处，
        可打印条件的 focus 范围宽度。
        """
        if not np.any(passing_mask):
            return 0.0

        nominal_dose_idx = np.argmin(np.abs(dose - 1.0))
        column_passing = passing_mask[:, nominal_dose_idx]

        if not np.any(column_passing):
            return 0.0

        passing_focus = focus[column_passing]
        return float(passing_focus.max() - passing_focus.min())

    def _compute_exposure_latitude(
        self,
        focus: np.ndarray,
        dose: np.ndarray,
        passing_mask: np.ndarray,
    ) -> float:
        """
        计算曝光宽容度 (Exposure Latitude, EL)

        EL 定义为：在最佳焦点（最接近 0 的 focus 行）处，
        可打印条件的剂量范围占标称剂量的百分比。
        """
        if not np.any(passing_mask):
            return 0.0

        best_focus_idx = np.argmin(np.abs(focus))
        row_passing = passing_mask[best_focus_idx, :]

        if not np.any(row_passing):
            return 0.0

        passing_dose = dose[row_passing]
        dose_range = passing_dose.max() - passing_dose.min()
        el = dose_range / 1.0 * 100.0
        return float(el)

    def plot_bossung(
        self,
        cd_target: Optional[float] = None,
        cd_tolerance: float = 0.1,
        title: str = "Bossung 图 (CD vs Focus)",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (10, 7),
    ) -> Any:
        """
        绘制 Bossung 曲线

        Args:
            cd_target: 目标 CD (nm)
            cd_tolerance: CD 相对容差
            title: 图表标题
            save_path: 保存路径
            show: 是否显示
            figsize: 图像尺寸

        Returns:
            Figure 对象
        """
        if self._scan_result is None:
            raise RuntimeError("请先调用 scan() 方法进行参数扫描")

        from utils.visualization import plot_bossung as _plot_bossung

        cd_tgt = cd_target if cd_target is not None else self.cd_target

        return _plot_bossung(
            focus_values=self._scan_result.unique_focus,
            dose_values=self._scan_result.unique_dose,
            cd_matrix=self._scan_result.cd_matrix,
            cd_target=cd_tgt,
            cd_tolerance=cd_tolerance,
            title=title,
            save_path=save_path,
            show=show,
            figsize=figsize,
        )

    def plot_pw_contour(
        self,
        metric: str = 'cd_error',
        cd_target: Optional[float] = None,
        cd_tolerance: float = 0.1,
        epe_tolerance: Optional[float] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Any:
        """
        绘制工艺窗口等高线图（热力图）

        Args:
            metric: 显示的指标，可选值:
                'cd_error' - CD 误差
                'cd' - CD 值
                'epe' - EPE
                'mse' - MSE
                'ssim' - SSIM
            cd_target: 目标 CD (nm)
            cd_tolerance: CD 相对容差
            epe_tolerance: EPE 容差 (nm)
            title: 图表标题
            save_path: 保存路径
            show: 是否显示
            figsize: 图像尺寸

        Returns:
            Figure 对象
        """
        if self._scan_result is None:
            raise RuntimeError("请先调用 scan() 方法进行参数扫描")

        from utils.visualization import plot_process_window_heatmap

        metric_map = {
            'cd_error': (self._scan_result.cd_error_matrix, 'CD 误差 (nm)'),
            'cd': (self._scan_result.cd_matrix, 'CD (nm)'),
            'epe': (self._scan_result.epe_matrix, 'EPE Mean (nm)'),
            'mse': (self._scan_result.mse_matrix, 'MSE'),
            'ssim': (self._scan_result.ssim_matrix, 'SSIM'),
        }

        if metric not in metric_map:
            raise ValueError(
                f"未知指标 '{metric}'，可选值: {list(metric_map.keys())}"
            )

        metric_matrix, metric_name = metric_map[metric]

        printability = self.judge_printability(
            cd_tolerance=cd_tolerance,
            epe_tolerance=epe_tolerance,
            cd_target=cd_target,
        )

        cd_tgt = cd_target if cd_target is not None else self.cd_target

        if title is None:
            title = f"工艺窗口 - {metric_name}"

        return plot_process_window_heatmap(
            focus_values=self._scan_result.unique_focus,
            dose_values=self._scan_result.unique_dose,
            metric_matrix=metric_matrix,
            metric_name=metric_name,
            passing_mask=printability.passing_mask,
            cd_target=cd_tgt,
            title=title,
            save_path=save_path,
            show=show,
            figsize=figsize,
        )

    def plot_pw_summary(
        self,
        cd_target: Optional[float] = None,
        cd_tolerance: float = 0.1,
        title: str = "工艺窗口综合分析",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 12),
    ) -> Any:
        """
        绘制工艺窗口综合分析图（多子图）

        包含 Bossung 图、CD 误差热力图、EPE 热力图、MSE 分布与可打印区域。

        Args:
            cd_target: 目标 CD (nm)
            cd_tolerance: CD 相对容差
            title: 主标题
            save_path: 保存路径
            show: 是否显示
            figsize: 图像尺寸

        Returns:
            Figure 对象
        """
        if self._scan_result is None:
            raise RuntimeError("请先调用 scan() 方法进行参数扫描")

        from utils.visualization import plot_process_window_summary

        cd_tgt = cd_target if cd_target is not None else self.cd_target

        return plot_process_window_summary(
            scan_result=self._scan_result,
            cd_target=cd_tgt,
            cd_tolerance=cd_tolerance,
            title=title,
            save_path=save_path,
            show=show,
            figsize=figsize,
        )

    def plot_multi_metric_heatmaps(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Any:
        """
        并排绘制多个工艺指标的热力图

        Args:
            metrics: 要绘制的指标列表，可选值:
                ['cd', 'cd_error', 'epe', 'mse', 'ssim', 'ils', 'nils']
            save_path: 保存路径
            show: 是否显示
            figsize: 图像尺寸

        Returns:
            Figure 对象
        """
        if self._scan_result is None:
            raise RuntimeError("请先调用 scan() 方法进行参数扫描")

        from utils.visualization import plot_multi_metric_heatmaps

        return plot_multi_metric_heatmaps(
            scan_result=self._scan_result,
            metrics=metrics,
            save_path=save_path,
            show=show,
            figsize=figsize,
        )

    def plot_ellipse_overlay(
        self,
        cd_tolerance: float = 0.1,
        epe_tolerance: Optional[float] = None,
        cd_target: Optional[float] = None,
        title: str = "工艺窗口椭圆/矩形近似",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Any:
        """
        绘制工艺窗口椭圆和矩形近似叠加图

        在 passing 区域热力图上叠加椭圆和矩形近似轮廓。

        Args:
            cd_tolerance: CD 相对容差
            epe_tolerance: EPE 容差 (nm)
            cd_target: 目标 CD (nm)
            title: 图表标题
            save_path: 保存路径
            show: 是否显示
            figsize: 图像尺寸

        Returns:
            Figure 对象
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse as EllipsePatch, Rectangle as RectPatch

        if self._scan_result is None:
            raise RuntimeError("请先调用 scan() 方法进行参数扫描")

        pw_metrics = self.compute_pw_metrics(
            cd_tolerance=cd_tolerance,
            epe_tolerance=epe_tolerance,
            cd_target=cd_target,
        )
        printability = self.judge_printability(
            cd_tolerance=cd_tolerance,
            epe_tolerance=epe_tolerance,
            cd_target=cd_target,
        )

        focus = self._scan_result.unique_focus
        dose = self._scan_result.unique_dose
        X, Y = np.meshgrid(focus, dose, indexing='ij')

        fig, ax = plt.subplots(figsize=figsize)

        pass_count = printability.passing_mask.astype(np.float64)
        im = ax.pcolormesh(X, Y, pass_count, cmap='RdYlGn', shading='auto',
                           vmin=0, vmax=1, alpha=0.8)
        plt.colorbar(im, ax=ax, label='可打印 (1=通过, 0=失败)')

        if pw_metrics.rect_approx is not None:
            r = pw_metrics.rect_approx
            rect = RectPatch(
                (r.focus_min, r.dose_min),
                r.focus_max - r.focus_min,
                r.dose_max - r.dose_min,
                linewidth=2.5,
                edgecolor='blue',
                facecolor='none',
                linestyle='--',
                label=f'矩形近似 (面积={r.area:.1f})',
            )
            ax.add_patch(rect)

        if pw_metrics.ellipse_approx is not None:
            e = pw_metrics.ellipse_approx
            ellipse = EllipsePatch(
                (e.center_focus, e.center_dose),
                width=2 * e.semi_axis_focus,
                height=2 * e.semi_axis_dose,
                angle=e.angle,
                linewidth=2.5,
                edgecolor='red',
                facecolor='none',
                linestyle='-',
                label=f'椭圆近似 (面积={e.area:.1f})',
            )
            ax.add_patch(ellipse)

        ax.plot(pw_metrics.best_focus, pw_metrics.best_dose, 'k*',
                markersize=14, markeredgecolor='white', markeredgewidth=1.5,
                label=f'最佳点 ({pw_metrics.best_focus:.0f}nm, {pw_metrics.best_dose:.3f})')

        ax.set_xlabel('Focus (nm)', fontsize=12)
        ax.set_ylabel('Dose', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(loc='best', fontsize=10)

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

    def run_full_analysis(
        self,
        focus_range: Union[Tuple[float, float, int], List[float], np.ndarray] = (-150, 150, 11),
        dose_range: Union[Tuple[float, float, int], List[float], np.ndarray] = (0.85, 1.15, 11),
        cd_tolerance: float = 0.1,
        epe_tolerance: Optional[float] = None,
        cd_target: Optional[float] = None,
        output_dir: Optional[str] = None,
        show: bool = False,
    ) -> Dict[str, Any]:
        """
        一站式完整工艺窗口分析

        依次执行：参数扫描 → 可打印性判定 → PW 指标计算 → 可视化输出

        Args:
            focus_range: 离焦量扫描范围
            dose_range: 曝光剂量扫描范围
            cd_tolerance: CD 相对容差
            epe_tolerance: EPE 绝对容差 (nm)
            cd_target: 目标 CD (nm)
            output_dir: 输出目录，None 则不保存图片
            show: 是否显示图片

        Returns:
            包含 scan_result, printability, metrics, figures 的字典
        """
        save_prefix = None
        if output_dir is not None:
            save_prefix = Path(output_dir)
            save_prefix.mkdir(parents=True, exist_ok=True)

        scan_result = self.scan(
            focus_range=focus_range,
            dose_range=dose_range,
            cd_target=cd_target,
            cd_tolerance=cd_tolerance,
        )

        printability = self.judge_printability(
            cd_tolerance=cd_tolerance,
            epe_tolerance=epe_tolerance,
            cd_target=cd_target,
        )

        metrics = self.compute_pw_metrics(
            cd_tolerance=cd_tolerance,
            epe_tolerance=epe_tolerance,
            cd_target=cd_target,
        )

        cd_tgt = cd_target if cd_target is not None else self.cd_target

        figures = {}

        if save_prefix is not None:
            fig_bossung = self.plot_bossung(
                cd_target=cd_tgt,
                cd_tolerance=cd_tolerance,
                save_path=str(save_prefix / 'bossung.png'),
                show=show,
            )
            figures['bossung'] = fig_bossung

            fig_contour = self.plot_pw_contour(
                metric='cd_error',
                cd_target=cd_tgt,
                cd_tolerance=cd_tolerance,
                save_path=str(save_prefix / 'pw_contour.png'),
                show=show,
            )
            figures['pw_contour'] = fig_contour

            fig_summary = self.plot_pw_summary(
                cd_target=cd_tgt,
                cd_tolerance=cd_tolerance,
                save_path=str(save_prefix / 'pw_summary.png'),
                show=show,
            )
            figures['pw_summary'] = fig_summary

            fig_ellipse = self.plot_ellipse_overlay(
                cd_tolerance=cd_tolerance,
                epe_tolerance=epe_tolerance,
                cd_target=cd_tgt,
                save_path=str(save_prefix / 'pw_ellipse_overlay.png'),
                show=show,
            )
            figures['ellipse_overlay'] = fig_ellipse

            fig_multi = self.plot_multi_metric_heatmaps(
                save_path=str(save_prefix / 'pw_multi_metric.png'),
                show=show,
            )
            figures['multi_metric'] = fig_multi
        else:
            fig_bossung = self.plot_bossung(
                cd_target=cd_tgt, cd_tolerance=cd_tolerance, show=show
            )
            figures['bossung'] = fig_bossung

            fig_contour = self.plot_pw_contour(
                metric='cd_error',
                cd_target=cd_tgt,
                cd_tolerance=cd_tolerance,
                show=show,
            )
            figures['pw_contour'] = fig_contour

        print(metrics.summary())

        return {
            'scan_result': scan_result,
            'printability': printability,
            'metrics': metrics,
            'figures': figures,
        }

    def _generate_monte_carlo_conditions(
        self,
        config: MonteCarloConfig,
    ) -> List[ProcessCondition]:
        """
        生成蒙特卡洛采样的工艺条件列表

        Args:
            config: 蒙特卡洛配置

        Returns:
            ProcessCondition 列表
        """
        rng = np.random.default_rng(config.random_seed)
        n = config.n_samples

        nominal = ProcessCondition.from_optical_system(
            self.optical_system,
            dose=config.nominal_dose,
        )

        if config.distribution == 'normal':
            focus_samples = rng.normal(
                loc=config.nominal_focus,
                scale=config.focus_std,
                size=n,
            )
            dose_samples = rng.normal(
                loc=config.nominal_dose,
                scale=config.dose_std * config.nominal_dose,
                size=n,
            )
        elif config.distribution == 'uniform':
            half_focus = config.focus_std * np.sqrt(3)
            half_dose = config.dose_std * config.nominal_dose * np.sqrt(3)
            focus_samples = rng.uniform(
                low=config.nominal_focus - half_focus,
                high=config.nominal_focus + half_focus,
                size=n,
            )
            dose_samples = rng.uniform(
                low=config.nominal_dose - half_dose,
                high=config.nominal_dose + half_dose,
                size=n,
            )
        else:
            raise ValueError(f"未知分布类型: {config.distribution}")

        dose_samples = np.clip(dose_samples, 0.1, 5.0)

        if config.aberration_std is not None:
            if config.zernike_indices is not None:
                zernike_indices = list(config.zernike_indices)
            else:
                zernike_indices = list(
                    self.optical_system.zernike_coefficients.keys()
                )
                if not zernike_indices:
                    zernike_indices = [3, 4, 5, 6, 7, 8, 9, 10]

            if isinstance(config.aberration_std, (int, float)):
                aberration_std_dict = {
                    j: float(config.aberration_std) for j in zernike_indices
                }
            else:
                aberration_std_dict = dict(config.aberration_std)

            if config.distribution == 'normal':
                aberration_samples = {}
                for j in zernike_indices:
                    std = aberration_std_dict.get(j, 0.0)
                    base_val = self.optical_system.zernike_coefficients.get(j, 0.0)
                    aberration_samples[j] = rng.normal(
                        loc=base_val, scale=std, size=n
                    )
            else:
                aberration_samples = {}
                for j in zernike_indices:
                    std = aberration_std_dict.get(j, 0.0)
                    half = std * np.sqrt(3)
                    base_val = self.optical_system.zernike_coefficients.get(j, 0.0)
                    aberration_samples[j] = rng.uniform(
                        low=base_val - half, high=base_val + half, size=n
                    )
        else:
            aberration_samples = None

        conditions = []
        for i in range(n):
            zernike_dict = dict(nominal.zernike_coefficients)
            if aberration_samples is not None:
                for j, samples in aberration_samples.items():
                    zernike_dict[j] = float(samples[i])

            cond = ProcessCondition(
                defocus=float(focus_samples[i]),
                dose=float(dose_samples[i]),
                na=nominal.na,
                sigma=nominal.sigma,
                wavelength=nominal.wavelength,
                flare=nominal.flare,
                shadowing_model=nominal.shadowing_model,
                reflective_mask_attenuation=nominal.reflective_mask_attenuation,
                technology_node=nominal.technology_node,
                zernike_coefficients=zernike_dict,
                use_vector_pupil=nominal.use_vector_pupil,
                incident_polarization_angle=nominal.incident_polarization_angle,
                n_immersion=nominal.n_immersion,
                use_mask_coating=nominal.use_mask_coating,
                name=f"mc_sample_{i:04d}",
                weight=1.0,
            )
            conditions.append(cond)

        return conditions

    @staticmethod
    def _compute_metric_stats(
        values: np.ndarray,
        higher_is_worse: bool = True,
        save_samples: bool = False,
    ) -> MonteCarloMetricStats:
        """
        计算单个指标的统计量

        Args:
            values: 采样值数组
            higher_is_worse: True 表示值越大越差（如 MSE、CD 误差），
                            False 表示值越小越差（如 SSIM）
            save_samples: 是否保存原始采样值

        Returns:
            MonteCarloMetricStats 统计结果
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return MonteCarloMetricStats(
                mean=0.0, std=0.0, variance=0.0,
                min=0.0, max=0.0, median=0.0,
                percentile_5=0.0, percentile_95=0.0,
                worst_case=0.0,
                samples=arr if save_samples else None,
            )

        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        var_val = float(np.var(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        median_val = float(np.median(arr))
        p5 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))

        worst_case = max_val if higher_is_worse else min_val

        return MonteCarloMetricStats(
            mean=mean_val,
            std=std_val,
            variance=var_val,
            min=min_val,
            max=max_val,
            median=median_val,
            percentile_5=p5,
            percentile_95=p95,
            worst_case=worst_case,
            samples=arr if save_samples else None,
        )

    def monte_carlo_analysis(
        self,
        config: Optional[MonteCarloConfig] = None,
        cd_tolerance: float = 0.1,
        cd_target: Optional[float] = None,
        save_samples: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> MonteCarloResult:
        """
        蒙特卡洛工艺鲁棒性分析

        对 focus、dose、像差系数做 Monte Carlo 采样，
        统计成像指标分布（均值、方差、最差情况）。

        Args:
            config: 蒙特卡洛采样配置，None 则使用默认配置
            cd_tolerance: CD 相对容差，用于计算可打印比例
            cd_target: 目标 CD (nm)，None 则从目标图自动测量
            save_samples: 是否保存所有采样的指标值
            progress_callback: 进度回调函数 callback(current, total)

        Returns:
            MonteCarloResult，包含各指标的统计分布
        """
        if config is None:
            config = MonteCarloConfig()

        if cd_target is not None:
            self._cd_target = cd_target

        cd_tgt = self.cd_target

        logger.info(
            f"开始蒙特卡洛工艺鲁棒性分析: {config.n_samples} 个采样, "
            f"focus_std={config.focus_std}nm, dose_std={config.dose_std * 100:.1f}%"
        )

        t_start = time.time()

        conditions = self._generate_monte_carlo_conditions(config)

        multi_result = simulate_multi_process(
            mask=self.mask,
            conditions=conditions,
            base_optics=self.optical_system,
            threshold=self.threshold,
            apply_resist=True,
            resist_model=self.resist_model,
            window_type=self.window_type,
            pad_width=self.pad_width,
            tukey_alpha=self.tukey_alpha,
        )

        cd_values = []
        cd_error_values = []
        epe_values = []
        mse_values = []
        ssim_values = []
        ils_values = []
        nils_values = []
        passing_count = 0
        worst_cd_error = -float('inf')
        worst_idx = 0

        target_binary = self.target >= 0.5

        for i in range(len(conditions)):
            wafer_img = multi_result.wafer_images[i]
            aerial_img = multi_result.aerial_images[i]

            wafer_binary = wafer_img >= self.threshold

            cd_res = compute_cd(wafer_binary, pixel_size=self.pixel_size)
            cd_val = cd_res['cd_mean']
            cd_values.append(cd_val)

            cd_err_val = abs(cd_val - cd_tgt)
            cd_error_values.append(cd_err_val)

            epe_res = compute_epe(wafer_binary, target_binary, pixel_size=self.pixel_size)
            epe_values.append(epe_res['epe_mean'])

            mse_values.append(mse(wafer_img, self.target))
            ssim_values.append(ssim(wafer_img, self.target))

            ils_res = compute_ils(aerial_img, threshold=self.threshold,
                                   pixel_size=self.pixel_size)
            ils_values.append(ils_res.get('ils_mean', 0.0))

            nils_res = compute_nils(aerial_img, cd_target=cd_tgt,
                                     threshold=self.threshold,
                                     pixel_size=self.pixel_size)
            nils_values.append(nils_res.get('nils_mean', 0.0))

            cd_lower = cd_tgt * (1.0 - cd_tolerance)
            cd_upper = cd_tgt * (1.0 + cd_tolerance)
            if cd_lower <= cd_val <= cd_upper:
                passing_count += 1

            if cd_err_val > worst_cd_error:
                worst_cd_error = cd_err_val
                worst_idx = i

            if progress_callback is not None:
                progress_callback(i + 1, len(conditions))

        cd_arr = np.array(cd_values)
        cd_err_arr = np.array(cd_error_values)
        epe_arr = np.array(epe_values)
        mse_arr = np.array(mse_values)
        ssim_arr = np.array(ssim_values)
        ils_arr = np.array(ils_values)
        nils_arr = np.array(nils_values)

        cd_stats = self._compute_metric_stats(cd_arr, higher_is_worse=False,
                                               save_samples=save_samples)
        cd_error_stats = self._compute_metric_stats(cd_err_arr, higher_is_worse=True,
                                                     save_samples=save_samples)
        epe_stats = self._compute_metric_stats(epe_arr, higher_is_worse=True,
                                                save_samples=save_samples)
        mse_stats = self._compute_metric_stats(mse_arr, higher_is_worse=True,
                                                save_samples=save_samples)
        ssim_stats = self._compute_metric_stats(ssim_arr, higher_is_worse=False,
                                                 save_samples=save_samples)
        ils_stats = self._compute_metric_stats(ils_arr, higher_is_worse=False,
                                                save_samples=save_samples)
        nils_stats = self._compute_metric_stats(nils_arr, higher_is_worse=False,
                                                 save_samples=save_samples)

        passing_ratio = passing_count / len(conditions) if conditions else 0.0
        worst_condition = conditions[worst_idx]

        elapsed = time.time() - t_start
        logger.info(f"蒙特卡洛分析完成, 耗时 {elapsed:.1f}s, 可打印比例: {passing_ratio * 100:.1f}%")

        return MonteCarloResult(
            n_samples=config.n_samples,
            config=config,
            cd_stats=cd_stats,
            cd_error_stats=cd_error_stats,
            epe_stats=epe_stats,
            mse_stats=mse_stats,
            ssim_stats=ssim_stats,
            ils_stats=ils_stats,
            nils_stats=nils_stats,
            sampled_conditions=conditions,
            passing_ratio=passing_ratio,
            worst_condition=worst_condition,
        )

    def plot_monte_carlo_histogram(
        self,
        mc_result: MonteCarloResult,
        metric: str = 'cd_error',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        bins: int = 30,
    ) -> Any:
        """
        绘制蒙特卡洛采样指标的直方图

        Args:
            mc_result: 蒙特卡洛分析结果
            metric: 要绘制的指标: 'cd', 'cd_error', 'epe', 'mse', 'ssim', 'ils', 'nils'
            title: 图表标题
            save_path: 保存路径
            show: 是否显示
            figsize: 图像尺寸
            bins: 直方图分箱数

        Returns:
            Figure 对象
        """
        import matplotlib.pyplot as plt

        metric_map = {
            'cd': (mc_result.cd_stats, 'CD (nm)'),
            'cd_error': (mc_result.cd_error_stats, 'CD 误差 (nm)'),
            'epe': (mc_result.epe_stats, 'EPE (nm)'),
            'mse': (mc_result.mse_stats, 'MSE'),
            'ssim': (mc_result.ssim_stats, 'SSIM'),
            'ils': (mc_result.ils_stats, 'ILS'),
            'nils': (mc_result.nils_stats, 'NILS'),
        }

        if metric not in metric_map:
            raise ValueError(
                f"未知指标 '{metric}'，可选值: {list(metric_map.keys())}"
            )

        stats, metric_name = metric_map[metric]

        if stats.samples is None:
            raise ValueError(
                "采样值未保存，请在 monte_carlo_analysis 中设置 save_samples=True"
            )

        fig, ax = plt.subplots(figsize=figsize)

        samples = stats.samples
        ax.hist(samples, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

        ax.axvline(stats.mean, color='red', linestyle='--', linewidth=2,
                   label=f'均值: {stats.mean:.3f}')
        ax.axvline(stats.worst_case, color='orange', linestyle='-.', linewidth=2,
                   label=f'最坏情况: {stats.worst_case:.3f}')
        ax.axvline(stats.percentile_5, color='green', linestyle=':', linewidth=1.5,
                   label=f'5% 分位: {stats.percentile_5:.3f}')
        ax.axvline(stats.percentile_95, color='purple', linestyle=':', linewidth=1.5,
                   label=f'95% 分位: {stats.percentile_95:.3f}')

        ax.set_xlabel(metric_name, fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        if title is None:
            title = f"蒙特卡洛分布 - {metric_name} (N={mc_result.n_samples})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')

        info_text = (
            f"均值: {stats.mean:.3f}\n"
            f"标准差: {stats.std:.3f}\n"
            f"范围: [{stats.min:.3f}, {stats.max:.3f}]\n"
            f"最坏情况: {stats.worst_case:.3f}"
        )
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
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

    def plot_monte_carlo_scatter(
        self,
        mc_result: MonteCarloResult,
        metric: str = 'cd_error',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        cd_tolerance: Optional[float] = None,
    ) -> Any:
        """
        绘制蒙特卡洛采样在 focus-dose 平面上的散点图

        每个点代表一个蒙特卡洛采样，颜色表示对应指标的大小，
        并用不同符号区分可打印/不可打印。

        Args:
            mc_result: 蒙特卡洛分析结果
            metric: 用于颜色编码的指标: 'cd', 'cd_error', 'epe', 'mse', 'ssim', 'ils', 'nils'
            title: 图表标题
            save_path: 保存路径
            show: 是否显示
            figsize: 图像尺寸
            cd_tolerance: 用于判断可打印性的 CD 相对容差，None 则使用 mc_result 中的值

        Returns:
            Figure 对象
        """
        import matplotlib.pyplot as plt

        metric_map = {
            'cd': (mc_result.cd_stats, 'CD (nm)'),
            'cd_error': (mc_result.cd_error_stats, 'CD 误差 (nm)'),
            'epe': (mc_result.epe_stats, 'EPE (nm)'),
            'mse': (mc_result.mse_stats, 'MSE'),
            'ssim': (mc_result.ssim_stats, 'SSIM'),
            'ils': (mc_result.ils_stats, 'ILS'),
            'nils': (mc_result.nils_stats, 'NILS'),
        }

        if metric not in metric_map:
            raise ValueError(
                f"未知指标 '{metric}'，可选值: {list(metric_map.keys())}"
            )

        stats, metric_name = metric_map[metric]
        if stats.samples is None:
            raise ValueError(
                "采样值未保存，请在 monte_carlo_analysis 中设置 save_samples=True"
            )

        conditions = mc_result.sampled_conditions
        focus_vals = np.array([c.defocus for c in conditions])
        dose_vals = np.array([c.dose for c in conditions])
        metric_vals = stats.samples

        if cd_tolerance is not None:
            cd_tgt = self.cd_target
            cd_samples = mc_result.cd_stats.samples
            if cd_samples is not None:
                cd_lower = cd_tgt * (1.0 - cd_tolerance)
                cd_upper = cd_tgt * (1.0 + cd_tolerance)
                passing = (cd_samples >= cd_lower) & (cd_samples <= cd_upper)
            else:
                passing = np.ones(len(focus_vals), dtype=bool)
        else:
            passing = np.ones(len(focus_vals), dtype=bool)

        fig, ax = plt.subplots(figsize=figsize)

        sc_pass = ax.scatter(
            focus_vals[passing], dose_vals[passing],
            c=metric_vals[passing], cmap='viridis',
            marker='o', s=60, alpha=0.8, edgecolors='white', linewidths=0.5,
            label='可打印'
        )

        if np.any(~passing):
            sc_fail = ax.scatter(
                focus_vals[~passing], dose_vals[~passing],
                c=metric_vals[~passing], cmap='viridis',
                marker='x', s=80, alpha=0.9, linewidths=1.5,
                label='不可打印'
            )

        worst = mc_result.worst_condition
        ax.scatter(
            worst.defocus, worst.dose,
            c='red', marker='*', s=300, edgecolors='black', linewidths=1,
            label=f'最坏情况 (focus={worst.defocus:.0f}nm, dose={worst.dose:.3f})'
        )

        ax.axvline(0.0, color='gray', linestyle='--', alpha=0.5, label='标称 focus')
        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='标称 dose')

        cbar = plt.colorbar(sc_pass, ax=ax)
        cbar.set_label(metric_name, fontsize=12)

        ax.set_xlabel('Focus (nm)', fontsize=12)
        ax.set_ylabel('Dose', fontsize=12)
        if title is None:
            title = f"蒙特卡洛采样散点图 (N={mc_result.n_samples})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')

        info_text = (
            f"可打印比例: {mc_result.passing_ratio * 100:.1f}%\n"
            f"{metric_name} 均值: {stats.mean:.3f}\n"
            f"{metric_name} 最坏: {stats.worst_case:.3f}"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
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

    def run_monte_carlo_full_analysis(
        self,
        config: Optional[MonteCarloConfig] = None,
        cd_tolerance: float = 0.1,
        cd_target: Optional[float] = None,
        output_dir: Optional[str] = None,
        show: bool = False,
    ) -> Dict[str, Any]:
        """
        一站式蒙特卡洛工艺鲁棒性完整分析

        依次执行：蒙特卡洛采样 → 统计分析 → 可视化输出

        Args:
            config: 蒙特卡洛采样配置，None 则使用默认配置
            cd_tolerance: CD 相对容差
            cd_target: 目标 CD (nm)，None 则从目标图自动测量
            output_dir: 输出目录，None 则不保存图片
            show: 是否显示图片

        Returns:
            包含 mc_result, figures 的字典
        """
        save_prefix = None
        if output_dir is not None:
            save_prefix = Path(output_dir)
            save_prefix.mkdir(parents=True, exist_ok=True)

        mc_result = self.monte_carlo_analysis(
            config=config,
            cd_tolerance=cd_tolerance,
            cd_target=cd_target,
            save_samples=True,
        )

        print(mc_result.summary())

        figures = {}

        if save_prefix is not None:
            for metric in ['cd_error', 'cd', 'epe', 'mse', 'ssim', 'ils', 'nils']:
                try:
                    fig_hist = self.plot_monte_carlo_histogram(
                        mc_result, metric=metric,
                        save_path=str(save_prefix / f'mc_hist_{metric}.png'),
                        show=show,
                    )
                    figures[f'hist_{metric}'] = fig_hist
                except Exception as e:
                    logger.warning(f"绘制 {metric} 直方图失败: {e}")

            try:
                fig_scatter = self.plot_monte_carlo_scatter(
                    mc_result, metric='cd_error',
                    save_path=str(save_prefix / 'mc_scatter_cd_error.png'),
                    show=show,
                    cd_tolerance=cd_tolerance,
                )
                figures['scatter_cd_error'] = fig_scatter
            except Exception as e:
                logger.warning(f"绘制散点图失败: {e}")
        else:
            fig_hist = self.plot_monte_carlo_histogram(
                mc_result, metric='cd_error', show=show
            )
            figures['hist_cd_error'] = fig_hist

            fig_scatter = self.plot_monte_carlo_scatter(
                mc_result, metric='cd_error', show=show,
                cd_tolerance=cd_tolerance,
            )
            figures['scatter_cd_error'] = fig_scatter

        return {
            'mc_result': mc_result,
            'figures': figures,
        }


def quick_monte_carlo_analysis(
    mask: np.ndarray,
    target: np.ndarray,
    optical_system: Optional[OpticalSystem] = None,
    n_samples: int = 100,
    focus_std: float = 30.0,
    dose_std: float = 0.05,
    aberration_std: Optional[Union[float, Dict[int, float]]] = None,
    cd_tolerance: float = 0.1,
    cd_target: Optional[float] = None,
    pixel_size: float = 1.0,
    threshold: float = 0.3,
    random_seed: Optional[int] = None,
    distribution: str = 'normal',
    output_dir: Optional[str] = None,
    show: bool = False,
) -> Dict[str, Any]:
    """
    快速蒙特卡洛工艺鲁棒性分析（便捷入口函数）

    Args:
        mask: 掩模图案
        target: 目标图案
        optical_system: 光学系统参数
        n_samples: 蒙特卡洛采样数量
        focus_std: 离焦量标准差 (nm)
        dose_std: 剂量相对标准差
        aberration_std: 像差系数标准差（波长λ），float 或 Dict[int, float]
        cd_tolerance: CD 相对容差
        cd_target: 目标 CD (nm)
        pixel_size: 像素尺寸 (nm)
        threshold: 光刻胶阈值
        random_seed: 随机种子
        distribution: 分布类型: 'normal' 或 'uniform'
        output_dir: 输出目录
        show: 是否显示图片

    Returns:
        包含 mc_result, figures 的字典
    """
    analyzer = ProcessWindowAnalyzer(
        mask=mask,
        target=target,
        optical_system=optical_system,
        threshold=threshold,
        pixel_size=pixel_size,
    )

    config = MonteCarloConfig(
        n_samples=n_samples,
        focus_std=focus_std,
        dose_std=dose_std,
        aberration_std=aberration_std,
        random_seed=random_seed,
        distribution=distribution,
    )

    return analyzer.run_monte_carlo_full_analysis(
        config=config,
        cd_tolerance=cd_tolerance,
        cd_target=cd_target,
        output_dir=output_dir,
        show=show,
    )


def quick_process_window_analysis(
    mask: np.ndarray,
    target: np.ndarray,
    optical_system: Optional[OpticalSystem] = None,
    focus_range: Union[Tuple[float, float, int], List[float], np.ndarray] = (-150, 150, 11),
    dose_range: Union[Tuple[float, float, int], List[float], np.ndarray] = (0.85, 1.15, 11),
    cd_tolerance: float = 0.1,
    pixel_size: float = 1.0,
    threshold: float = 0.3,
    output_dir: Optional[str] = None,
    show: bool = False,
) -> Dict[str, Any]:
    """
    快速工艺窗口分析（便捷入口函数）

    Args:
        mask: 掩模图案
        target: 目标图案
        optical_system: 光学系统参数
        focus_range: 离焦量扫描范围
        dose_range: 曝光剂量扫描范围
        cd_tolerance: CD 相对容差
        pixel_size: 像素尺寸 (nm)
        threshold: 光刻胶阈值
        output_dir: 输出目录
        show: 是否显示图片

    Returns:
        包含 scan_result, printability, metrics, figures 的字典
    """
    analyzer = ProcessWindowAnalyzer(
        mask=mask,
        target=target,
        optical_system=optical_system,
        threshold=threshold,
        pixel_size=pixel_size,
    )

    return analyzer.run_full_analysis(
        focus_range=focus_range,
        dose_range=dose_range,
        cd_tolerance=cd_tolerance,
        output_dir=output_dir,
        show=show,
    )
