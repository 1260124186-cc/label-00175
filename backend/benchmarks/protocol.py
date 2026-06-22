# -*- coding: utf-8 -*-
"""
逆向光刻基准测试套件 - 统一评价协议

定义四维评价体系，确保不同优化算法之间的公平对比:
    1. 收敛速度 (Convergence Speed): 达到指定 EPE/MSE 阈值所需的迭代数
    2. 最终 EPE (Final Edge Placement Error): 优化后边缘放置误差
    3. PW 面积 (Process Window Area): 工艺窗口面积/比例
    4. 计算时间 (Compute Time): 单次完整优化的挂钟耗时

所有评价均在相同的随机种子、初始条件、光学系统下执行，
结果以 BenchmarkResult 数据结构返回，支持序列化与对比。
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetric:
    """
    收敛速度指标

    Attributes:
        iterations_to_threshold: 达到目标阈值所需的迭代数 (None 表示未达阈值)
        threshold_value: 所使用的阈值
        total_iterations: 总迭代数
        converged: 是否收敛
        convergence_step: 首次满足 tol 的迭代步
        loss_history: 损失历史
        epe_history: EPE 历史 (若可用)
    """
    iterations_to_threshold: Optional[int] = None
    threshold_value: Optional[float] = None
    total_iterations: int = 0
    converged: Optional[bool] = None
    convergence_step: Optional[int] = None
    loss_history: List[float] = field(default_factory=list)
    epe_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iterations_to_threshold': self.iterations_to_threshold,
            'threshold_value': self.threshold_value,
            'total_iterations': self.total_iterations,
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'epe_history': self.epe_history,
        }


@dataclass
class EPEMetric:
    """
    边缘放置误差指标

    Attributes:
        epe_mean: 平均 EPE (nm)
        epe_max: 最大 EPE (nm)
        epe_std: EPE 标准差 (nm)
        epe_median: EPE 中位数 (nm)
        epe_improvement_ratio: EPE 改善比例 (初始 -> 最终)
    """
    epe_mean: float = 0.0
    epe_max: float = 0.0
    epe_std: float = 0.0
    epe_median: float = 0.0
    epe_improvement_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epe_mean': self.epe_mean,
            'epe_max': self.epe_max,
            'epe_std': self.epe_std,
            'epe_median': self.epe_median,
            'epe_improvement_ratio': self.epe_improvement_ratio,
        }


@dataclass
class PWMetric:
    """
    工艺窗口指标

    Attributes:
        pw_area: 工艺窗口面积 (nm * dose)
        pw_ratio: 工艺窗口比例
        depth_of_focus: 焦深 (nm)
        exposure_latitude: 曝光宽容度 (%)
        n_passing: 通过条件数
        n_total: 总条件数
    """
    pw_area: float = 0.0
    pw_ratio: float = 0.0
    depth_of_focus: float = 0.0
    exposure_latitude: float = 0.0
    n_passing: int = 0
    n_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pw_area': self.pw_area,
            'pw_ratio': self.pw_ratio,
            'depth_of_focus': self.depth_of_focus,
            'exposure_latitude': self.exposure_latitude,
            'n_passing': self.n_passing,
            'n_total': self.n_total,
        }


@dataclass
class TimeMetric:
    """
    计算时间指标

    Attributes:
        total_wall_time: 总挂钟时间 (s)
        per_iteration_time: 每次迭代平均时间 (s)
        imaging_time: 成像仿真时间 (s)
        optimization_time: 优化迭代时间 (s)
        pw_scan_time: 工艺窗口扫描时间 (s), 0 表示未执行
    """
    total_wall_time: float = 0.0
    per_iteration_time: float = 0.0
    imaging_time: float = 0.0
    optimization_time: float = 0.0
    pw_scan_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_wall_time': self.total_wall_time,
            'per_iteration_time': self.per_iteration_time,
            'imaging_time': self.imaging_time,
            'optimization_time': self.optimization_time,
            'pw_scan_time': self.pw_scan_time,
        }


@dataclass
class BenchmarkResult:
    """
    基准测试完整结果

    Attributes:
        test_case_name: 测试用例名
        algorithm_name: 算法名
        convergence: 收敛速度指标
        epe: EPE 指标
        pw: PW 指标
        time: 时间指标
        custom_metrics: 附加自定义指标
        success: 是否成功执行
        error_message: 错误信息
    """
    test_case_name: str = ''
    algorithm_name: str = ''
    convergence: ConvergenceMetric = field(default_factory=ConvergenceMetric)
    epe: EPEMetric = field(default_factory=EPEMetric)
    pw: PWMetric = field(default_factory=PWMetric)
    time: TimeMetric = field(default_factory=TimeMetric)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    error_message: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_case_name': self.test_case_name,
            'algorithm_name': self.algorithm_name,
            'convergence': self.convergence.to_dict(),
            'epe': self.epe.to_dict(),
            'pw': self.pw.to_dict(),
            'time': self.time.to_dict(),
            'custom_metrics': self.custom_metrics,
            'success': self.success,
            'error_message': self.error_message,
        }

    def summary(self) -> str:
        lines = [
            f"--- {self.test_case_name} / {self.algorithm_name} ---",
            f"  收敛: {'是' if self.convergence.converged else '否'}, "
            f"步数={self.convergence.convergence_step or 'N/A'}"
            f"{'  (阈值步=' + str(self.convergence.iterations_to_threshold) + ')' if self.convergence.iterations_to_threshold is not None else ''}",
            f"  EPE: mean={self.epe.epe_mean:.2f}nm, max={self.epe.epe_max:.2f}nm, "
            f"改善={self.epe.epe_improvement_ratio:.1%}",
            f"  PW: 面积={self.pw.pw_area:.1f}, 比例={self.pw.pw_ratio:.1%}, "
            f"DOF={self.pw.depth_of_focus:.1f}nm, EL={self.pw.exposure_latitude:.1f}%",
            f"  时间: 总={self.time.total_wall_time:.3f}s, "
            f"每迭代={self.time.per_iteration_time:.4f}s",
        ]
        return "\n".join(lines)


class EvaluationProtocol:
    """
    统一评价协议

    按照四维评价体系，对优化算法的结果进行标准化评估。
    每个维度都有明确的计算方法，保证跨算法公平对比。
    """

    def __init__(
        self,
        epe_threshold: float = 5.0,
        mse_threshold: float = 0.01,
        convergence_tol: float = 1e-6,
        compute_pw: bool = True,
        pw_focus_range: Tuple[float, float, int] = (-150, 150, 11),
        pw_dose_range: Tuple[float, float, int] = (0.85, 1.15, 11),
        pw_cd_tolerance: float = 0.1,
        pixel_size: float = 1.0,
        wafer_threshold: float = 0.3,
    ):
        """
        Args:
            epe_threshold: EPE 收敛阈值 (nm), 达到此阈值视为收敛
            mse_threshold: MSE 收敛阈值
            convergence_tol: 损失变化收敛容差
            compute_pw: 是否计算工艺窗口 (较耗时)
            pw_focus_range: PW 扫描 focus 范围
            pw_dose_range: PW 扫描 dose 范围
            pw_cd_tolerance: PW CD 容差
            pixel_size: 像素尺寸 (nm)
            wafer_threshold: 晶圆图二值化阈值
        """
        self.epe_threshold = epe_threshold
        self.mse_threshold = mse_threshold
        self.convergence_tol = convergence_tol
        self.compute_pw = compute_pw
        self.pw_focus_range = pw_focus_range
        self.pw_dose_range = pw_dose_range
        self.pw_cd_tolerance = pw_cd_tolerance
        self.pixel_size = pixel_size
        self.wafer_threshold = wafer_threshold

    def evaluate(
        self,
        test_case_name: str,
        algorithm_name: str,
        optimized_mask: np.ndarray,
        target: np.ndarray,
        optical_system,
        loss_history: List[float],
        total_iterations: int,
        converged: Optional[bool],
        convergence_step: Optional[int],
        initial_epe: Optional[Dict[str, float]] = None,
        epe_history: Optional[List[float]] = None,
        wall_time: float = 0.0,
        imaging_time: float = 0.0,
        optimization_time: float = 0.0,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> BenchmarkResult:
        """
        执行统一评价

        Args:
            test_case_name: 测试用例名
            algorithm_name: 算法名
            optimized_mask: 优化后掩模
            target: 目标图案
            optical_system: 光学系统
            loss_history: 损失历史
            total_iterations: 总迭代数
            converged: 是否收敛
            convergence_step: 收敛步数
            initial_epe: 初始 EPE 字典
            epe_history: EPE 历史
            wall_time: 总挂钟时间
            imaging_time: 成像仿真时间
            optimization_time: 优化迭代时间
            custom_metrics: 自定义指标

        Returns:
            BenchmarkResult
        """
        result = BenchmarkResult(
            test_case_name=test_case_name,
            algorithm_name=algorithm_name,
            success=True,
        )

        try:
            result.convergence = self._evaluate_convergence(
                loss_history=loss_history,
                total_iterations=total_iterations,
                converged=converged,
                convergence_step=convergence_step,
                epe_history=epe_history,
            )

            result.epe = self._evaluate_epe(
                optimized_mask=optimized_mask,
                target=target,
                optical_system=optical_system,
                initial_epe=initial_epe,
            )

            if self.compute_pw:
                pw_start = time.time()
                result.pw = self._evaluate_pw(
                    optimized_mask=optimized_mask,
                    target=target,
                    optical_system=optical_system,
                )
                result.time.pw_scan_time = time.time() - pw_start

            result.time = self._evaluate_time(
                total_iterations=total_iterations,
                wall_time=wall_time,
                imaging_time=imaging_time,
                optimization_time=optimization_time,
                pw_scan_time=result.time.pw_scan_time,
            )

            result.custom_metrics = custom_metrics or {}

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"评价失败 [{test_case_name}/{algorithm_name}]: {e}", exc_info=True)

        return result

    def _evaluate_convergence(
        self,
        loss_history: List[float],
        total_iterations: int,
        converged: Optional[bool],
        convergence_step: Optional[int],
        epe_history: Optional[List[float]] = None,
    ) -> ConvergenceMetric:
        metric = ConvergenceMetric(
            total_iterations=total_iterations,
            converged=converged,
            convergence_step=convergence_step,
            loss_history=list(loss_history) if loss_history else [],
            epe_history=list(epe_history) if epe_history else [],
        )

        if epe_history:
            for i, epe_val in enumerate(epe_history):
                if epe_val <= self.epe_threshold:
                    metric.iterations_to_threshold = i + 1
                    metric.threshold_value = self.epe_threshold
                    break
        elif loss_history:
            for i, loss_val in enumerate(loss_history):
                if loss_val <= self.mse_threshold:
                    metric.iterations_to_threshold = i + 1
                    metric.threshold_value = self.mse_threshold
                    break

        return metric

    def _evaluate_epe(
        self,
        optimized_mask: np.ndarray,
        target: np.ndarray,
        optical_system,
        initial_epe: Optional[Dict[str, float]] = None,
    ) -> EPEMetric:
        from core.imaging import PartialCoherentImaging
        from core.litho_metrics import compute_epe

        imaging = PartialCoherentImaging(optical_system, optimized_mask.shape)
        aerial = imaging.compute_aerial_image(optimized_mask)
        wafer = (aerial >= self.wafer_threshold).astype(np.float64)
        target_bin = (target >= 0.5).astype(np.float64)

        epe_dict = compute_epe(wafer, target_bin, pixel_size=self.pixel_size)

        improvement = 0.0
        if initial_epe and initial_epe.get('epe_mean', 0) > 1e-10:
            improvement = (initial_epe['epe_mean'] - epe_dict['epe_mean']) / initial_epe['epe_mean']

        return EPEMetric(
            epe_mean=epe_dict['epe_mean'],
            epe_max=epe_dict['epe_max'],
            epe_std=epe_dict['epe_std'],
            epe_median=epe_dict['epe_median'],
            epe_improvement_ratio=improvement,
        )

    def _evaluate_pw(
        self,
        optimized_mask: np.ndarray,
        target: np.ndarray,
        optical_system,
    ) -> PWMetric:
        from analysis.process_window import ProcessWindowAnalyzer

        analyzer = ProcessWindowAnalyzer(
            mask=optimized_mask,
            target=target,
            optical_system=optical_system,
            pixel_size=self.pixel_size,
            threshold=self.wafer_threshold,
        )

        scan_result = analyzer.scan(
            focus_range=self.pw_focus_range,
            dose_range=self.pw_dose_range,
            cd_tolerance=self.pw_cd_tolerance,
        )

        pw_metrics = analyzer.compute_pw_metrics(
            cd_tolerance=self.pw_cd_tolerance,
        )

        return PWMetric(
            pw_area=pw_metrics.pw_area,
            pw_ratio=pw_metrics.pw_ratio,
            depth_of_focus=pw_metrics.depth_of_focus,
            exposure_latitude=pw_metrics.exposure_latitude,
            n_passing=pw_metrics.n_passing,
            n_total=pw_metrics.n_total,
        )

    def _evaluate_time(
        self,
        total_iterations: int,
        wall_time: float,
        imaging_time: float,
        optimization_time: float,
        pw_scan_time: float,
    ) -> TimeMetric:
        per_iter = wall_time / total_iterations if total_iterations > 0 else 0.0
        return TimeMetric(
            total_wall_time=wall_time,
            per_iteration_time=per_iter,
            imaging_time=imaging_time,
            optimization_time=optimization_time,
            pw_scan_time=pw_scan_time,
        )


def compare_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """
    对比多个算法的基准测试结果

    Args:
        results: 同一测试用例下多个算法的 BenchmarkResult 列表

    Returns:
        对比摘要字典
    """
    if not results:
        return {}

    case_name = results[0].test_case_name
    rows = []
    for r in results:
        if not r.success:
            rows.append({
                'algorithm': r.algorithm_name,
                'status': 'FAILED',
                'epe_mean': float('nan'),
                'pw_ratio': float('nan'),
                'iterations': 0,
                'wall_time': float('nan'),
            })
            continue
        rows.append({
            'algorithm': r.algorithm_name,
            'status': 'OK',
            'epe_mean': r.epe.epe_mean,
            'pw_ratio': r.pw.pw_ratio,
            'iterations': r.convergence.total_iterations,
            'convergence_step': r.convergence.convergence_step,
            'wall_time': r.time.total_wall_time,
            'per_iter_time': r.time.per_iteration_time,
        })

    successful = [row for row in rows if row['status'] == 'OK']
    best = {}
    if successful:
        best_epe = min(successful, key=lambda x: x['epe_mean'])
        best_pw = max(successful, key=lambda x: x['pw_ratio'])
        best_speed = min(successful, key=lambda x: x['wall_time'])
        best = {
            'best_epe': best_epe['algorithm'],
            'best_pw': best_pw['algorithm'],
            'best_speed': best_speed['algorithm'],
        }

    return {
        'test_case': case_name,
        'rows': rows,
        'best': best,
    }
