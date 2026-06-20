# -*- coding: utf-8 -*-
"""
缺陷敏感度分析与失效概率评估模块

系统化扫描各类缺陷（类型、尺寸、位置、极性），
计算缺陷诱导 CD 变化与失效概率，
输出缺陷敏感度排序表，供掩模检测规格制定参考。
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from defect.schemas import (
    DefectType,
    DefectPolarity,
    PointDefect,
    LineDefect,
    ContaminationDefect,
    DefectLocation,
    DefectInjectionConfig,
    DefectSensitivityEntry,
    DefectSensitivityReport,
    SingleDefectResult,
)
from defect.defect_injector import DefectInjector
from defect.defect_simulator import DefectSimulator

from core.imaging import OpticalSystem, ProcessCondition
from core.litho_metrics import compute_cd

logger = logging.getLogger(__name__)


@dataclass
class DefectScanConfig:
    """
    缺陷扫描配置

    控制系统化缺陷扫描的参数范围。

    Attributes:
        point_sizes_nm: 点缺陷尺寸扫描列表 (nm)
        line_widths_nm: 线缺陷宽度扫描列表 (nm)
        line_lengths_nm: 线缺陷长度扫描列表 (nm)
        contamination_sizes_nm: 污染斑尺寸扫描列表 (nm)
        polarities: 缺陷极性扫描列表
        point_shapes: 点缺陷形状扫描列表
        line_angles_deg: 线缺陷角度扫描列表 (度)
        contamination_attenuations: 污染斑衰减系数扫描列表
        n_edge_locations: 每个尺寸/类型组合在边缘附近采样的位置数
        n_center_locations: 在图案中心采样的位置数
        scan_opaque: 是否扫描不透明缺陷
        scan_clear: 是否扫描透明缺陷
    """
    point_sizes_nm: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0])
    line_widths_nm: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0, 30.0])
    line_lengths_nm: List[float] = field(default_factory=lambda: [30.0, 60.0, 90.0, 120.0])
    contamination_sizes_nm: List[float] = field(default_factory=lambda: [10.0, 20.0, 30.0, 45.0, 60.0, 90.0])
    polarities: List[DefectPolarity] = field(default_factory=lambda: [
        DefectPolarity.OPAQUE, DefectPolarity.CLEAR
    ])
    point_shapes: List[str] = field(default_factory=lambda: ['circle', 'square'])
    line_angles_deg: List[float] = field(default_factory=lambda: [0.0, 45.0, 90.0])
    contamination_attenuations: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.9])
    n_edge_locations: int = 3
    n_center_locations: int = 1
    scan_opaque: bool = True
    scan_clear: bool = True
    scan_point: bool = True
    scan_line: bool = True
    scan_contamination: bool = True

    def active_polarities(self) -> List[DefectPolarity]:
        result = []
        if self.scan_opaque and DefectPolarity.OPAQUE in self.polarities:
            result.append(DefectPolarity.OPAQUE)
        if self.scan_clear and DefectPolarity.CLEAR in self.polarities:
            result.append(DefectPolarity.CLEAR)
        return result


class DefectSensitivityAnalyzer:
    """
    缺陷敏感度分析器

    系统化扫描各类缺陷参数组合，仿真其对晶圆成像的影响，
    生成敏感度排序表和掩模检测规格建议。

    使用方式::

        analyzer = DefectSensitivityAnalyzer(optical_system)
        report = analyzer.analyze(mask_nominal, scan_config)
        print(report.summary())
    """

    def __init__(
        self,
        optical_system: Optional[OpticalSystem] = None,
        injection_config: Optional[DefectInjectionConfig] = None,
        simulator: Optional[DefectSimulator] = None,
    ):
        """
        初始化分析器

        Args:
            optical_system: 光学系统参数
            injection_config: 缺陷注入配置
            simulator: 可复用的仿真器实例；None 则新建
        """
        self.optical_system = optical_system if optical_system is not None else OpticalSystem()
        self.injection_config = injection_config if injection_config is not None else DefectInjectionConfig()
        self.injector = DefectInjector(self.injection_config)

        if simulator is not None:
            self.simulator = simulator
        else:
            self.simulator = DefectSimulator(
                optical_system=self.optical_system,
                config=self.injection_config,
            )

    def generate_defect_suite(
        self,
        mask_nominal: np.ndarray,
        scan_config: Optional[DefectScanConfig] = None,
    ) -> List[Union[PointDefect, LineDefect, ContaminationDefect]]:
        """
        生成系统化的缺陷测试套件

        Args:
            mask_nominal: 标称掩模（用于确定边缘附近的缺陷位置）
            scan_config: 扫描配置，None 则使用默认

        Returns:
            缺陷参数对象列表
        """
        if scan_config is None:
            scan_config = DefectScanConfig()

        defects: List[Union[PointDefect, LineDefect, ContaminationDefect]] = []

        edge_locations = self.injector.generate_edge_proximity_locations(
            mask_nominal,
            n_locations=scan_config.n_edge_locations,
            min_distance_nm=0.0,
            max_distance_nm=self.injection_config.pixel_size * 10.0,
        )

        ny, nx = mask_nominal.shape
        center_location = DefectLocation(
            y=ny / 2.0, x=nx / 2.0,
            distance_to_edge=self.injector.compute_distance_to_edge(
                mask_nominal, DefectLocation(y=ny / 2.0, x=nx / 2.0)
            ),
        )
        center_locations = [center_location] * scan_config.n_center_locations

        all_locations = edge_locations + center_locations

        polarities = scan_config.active_polarities()

        if scan_config.scan_point:
            for size in scan_config.point_sizes_nm:
                for shape in scan_config.point_shapes:
                    for polarity in polarities:
                        for loc in all_locations:
                            defects.append(PointDefect(
                                size_nm=size,
                                shape=shape,
                                polarity=polarity,
                                location=DefectLocation(
                                    y=loc.y, x=loc.x,
                                    distance_to_edge=loc.distance_to_edge,
                                ),
                            ))

        if scan_config.scan_line:
            for width in scan_config.line_widths_nm:
                for length in scan_config.line_lengths_nm:
                    for angle in scan_config.line_angles_deg:
                        for polarity in polarities:
                            for loc in all_locations:
                                defects.append(LineDefect(
                                    length_nm=length,
                                    width_nm=width,
                                    angle_deg=angle,
                                    polarity=polarity,
                                    location=DefectLocation(
                                        y=loc.y, x=loc.x,
                                        distance_to_edge=loc.distance_to_edge,
                                    ),
                                ))

        if scan_config.scan_contamination:
            for size in scan_config.contamination_sizes_nm:
                for attenuation in scan_config.contamination_attenuations:
                    for polarity in polarities:
                        for loc in all_locations:
                            defects.append(ContaminationDefect(
                                size_nm=size,
                                attenuation=attenuation,
                                roughness=0.3,
                                polarity=polarity,
                                location=DefectLocation(
                                    y=loc.y, x=loc.x,
                                    distance_to_edge=loc.distance_to_edge,
                                ),
                            ))

        logger.info(f"生成缺陷测试套件共 {len(defects)} 个缺陷")
        return defects

    def analyze(
        self,
        mask_nominal: np.ndarray,
        scan_config: Optional[DefectScanConfig] = None,
        process_condition: Optional[ProcessCondition] = None,
        progress_callback: Optional[Any] = None,
    ) -> DefectSensitivityReport:
        """
        执行完整的缺陷敏感度分析

        Args:
            mask_nominal: 标称掩模
            scan_config: 扫描配置
            process_condition: 工艺条件，None 使用标称条件
            progress_callback: 进度回调 callback(current, total)

        Returns:
            DefectSensitivityReport 完整分析报告
        """
        t_start = time.time()

        if scan_config is None:
            scan_config = DefectScanConfig()

        if self.injection_config.cd_target is None:
            nominal_cd_stats = compute_cd(
                mask_nominal, pixel_size=self.injection_config.pixel_size
            )
            self.injection_config.cd_target = nominal_cd_stats['cd_mean']
            logger.info(f"从掩模自动测量标称 CD: {self.injection_config.cd_target:.2f} nm")

        defects = self.generate_defect_suite(mask_nominal, scan_config)

        self.simulator.clear_cache()
        simulation_results = self.simulator.simulate_defects_batch(
            mask_nominal, defects, process_condition,
            save_images=False,
            progress_callback=progress_callback,
        )

        report = self._build_report(simulation_results, defects)

        elapsed = time.time() - t_start
        logger.info(f"缺陷敏感度分析完成，耗时 {elapsed:.1f}s，"
                     f"分析 {len(defects)} 个缺陷")
        return report

    def _build_report(
        self,
        results: List[SingleDefectResult],
        defects: List[Union[PointDefect, LineDefect, ContaminationDefect]],
    ) -> DefectSensitivityReport:
        """
        根据仿真结果构建敏感度分析报告

        Args:
            results: 仿真结果列表
            defects: 对应的缺陷参数列表

        Returns:
            DefectSensitivityReport
        """
        entries: List[DefectSensitivityEntry] = []

        for i, (res, defect) in enumerate(zip(results, defects)):
            size_nm = DefectSimulator.get_defect_size(defect)
            polarity = DefectSimulator.get_defect_polarity(defect)

            loc_str = self._format_location(defect.location)
            recommendation = self._generate_recommendation(
                res, defect, size_nm,
            )

            entry = DefectSensitivityEntry(
                rank=0,
                defect_type=res.defect_type,
                size_nm=size_nm,
                polarity=polarity,
                location=loc_str,
                delta_cd_abs=abs(res.delta_cd),
                delta_cd_relative=abs(res.delta_cd_relative),
                is_critical=res.is_critical,
                failure_probability=res.failure_probability,
                sensitivity_score=res.sensitivity_score,
                recommendation=recommendation,
            )
            entries.append(entry)

        entries.sort(key=lambda e: (-e.sensitivity_score, -e.delta_cd_abs))
        for rank, entry in enumerate(entries, 1):
            entry.rank = rank

        total = len(entries)
        critical_count = sum(1 for e in entries if e.is_critical)
        critical_ratio = critical_count / total if total > 0 else 0.0

        nominal_cd = self.injection_config.cd_target or 0.0
        cd_tolerance = self.injection_config.cd_tolerance

        summary_stats = self._compute_summary_stats(entries)

        recommended_spec = self._recommend_detection_spec(entries, nominal_cd, cd_tolerance)

        return DefectSensitivityReport(
            entries=entries,
            total_defects_analyzed=total,
            critical_defect_count=critical_count,
            critical_defect_ratio=critical_ratio,
            recommended_spec=recommended_spec,
            summary_stats=summary_stats,
            nominal_cd=float(nominal_cd),
            cd_tolerance=float(cd_tolerance),
        )

    @staticmethod
    def _format_location(location: Optional[DefectLocation]) -> str:
        if location is None:
            return "unknown"
        if location.distance_to_edge is not None:
            if location.distance_to_edge < 5.0:
                return "edge"
            elif location.distance_to_edge < 20.0:
                return "near_edge"
            else:
                return "interior"
        return f"({location.y:.0f}, {location.x:.0f})"

    @staticmethod
    def _generate_recommendation(
        result: SingleDefectResult,
        defect,
        size_nm: float,
    ) -> str:
        if not result.is_critical:
            if abs(result.delta_cd_relative) < 1.0:
                return f"尺寸 {size_nm:.0f}nm 可忽略"
            elif abs(result.delta_cd_relative) < 3.0:
                return f"尺寸 {size_nm:.0f}nm 建议常规检测"
            else:
                return f"尺寸 {size_nm:.0f}nm 需重点检测"
        else:
            return f"尺寸 {size_nm:.0f}nm 致命缺陷，必须检出"

    @staticmethod
    def _compute_summary_stats(
        entries: List[DefectSensitivityEntry],
    ) -> Dict[str, float]:
        if not entries:
            return {}

        delta_abs_values = [e.delta_cd_abs for e in entries]
        delta_rel_values = [e.delta_cd_relative for e in entries]
        scores = [e.sensitivity_score for e in entries]
        probs = [e.failure_probability for e in entries]

        return {
            'delta_cd_abs_mean': float(np.mean(delta_abs_values)),
            'delta_cd_abs_median': float(np.median(delta_abs_values)),
            'delta_cd_abs_max': float(np.max(delta_abs_values)),
            'delta_cd_rel_mean_pct': float(np.mean(delta_rel_values)),
            'delta_cd_rel_max_pct': float(np.max(delta_rel_values)),
            'sensitivity_score_mean': float(np.mean(scores)),
            'sensitivity_score_max': float(np.max(scores)),
            'failure_probability_mean': float(np.mean(probs)),
            'failure_probability_max': float(np.max(probs)),
        }

    @staticmethod
    def _recommend_detection_spec(
        entries: List[DefectSensitivityEntry],
        nominal_cd: float,
        cd_tolerance: float,
    ) -> float:
        """
        根据敏感度分析结果推荐掩模检测规格（最小可检出缺陷尺寸）

        策略：找到不致命缺陷的最大尺寸和致命缺陷的最小尺寸，
        在两者之间设置检测阈值。

        Args:
            entries: 敏感度排序条目
            nominal_cd: 标称 CD
            cd_tolerance: CD 容差

        Returns:
            推荐的检测规格 (nm)
        """
        if not entries:
            return nominal_cd * 0.2 if nominal_cd > 0 else 10.0

        non_critical_sizes = [e.size_nm for e in entries if not e.is_critical]
        critical_sizes = [e.size_nm for e in entries if e.is_critical]

        if not critical_sizes:
            return max(non_critical_sizes) * 1.5 if non_critical_sizes else nominal_cd * 0.3

        if not non_critical_sizes:
            return min(critical_sizes) * 0.7

        max_safe = max(non_critical_sizes)
        min_dangerous = min(critical_sizes)

        if min_dangerous <= max_safe:
            return nominal_cd * cd_tolerance * 0.5 if nominal_cd > 0 else 15.0

        return (max_safe + min_dangerous) / 2.0


def run_defect_analysis(
    mask_nominal: np.ndarray,
    optical_system: Optional[OpticalSystem] = None,
    scan_config: Optional[DefectScanConfig] = None,
    cd_target: Optional[float] = None,
    cd_tolerance: float = 0.1,
    pixel_size: float = 1.0,
    threshold: float = 0.3,
    process_condition: Optional[ProcessCondition] = None,
    progress_callback: Optional[Any] = None,
) -> DefectSensitivityReport:
    """
    便捷函数：一站式运行缺陷敏感度分析

    Args:
        mask_nominal: 标称掩模 (2D数组，值范围 [0, 1])
        optical_system: 光学系统参数，None 使用默认
        scan_config: 缺陷扫描配置，None 使用默认
        cd_target: 目标 CD (nm)，None 则从掩模自动测量
        cd_tolerance: CD 相对容差 (0~1)
        pixel_size: 像素尺寸 (nm/pixel)
        threshold: 光刻胶阈值
        process_condition: 工艺条件
        progress_callback: 进度回调 callback(current, total)

    Returns:
        DefectSensitivityReport 完整分析报告
    """
    injection_config = DefectInjectionConfig(
        pixel_size=pixel_size,
        cd_target=cd_target,
        cd_tolerance=cd_tolerance,
        threshold=threshold,
    )

    analyzer = DefectSensitivityAnalyzer(
        optical_system=optical_system,
        injection_config=injection_config,
    )

    return analyzer.analyze(
        mask_nominal=mask_nominal,
        scan_config=scan_config,
        process_condition=process_condition,
        progress_callback=progress_callback,
    )
