# -*- coding: utf-8 -*-
"""
缺陷分析数据模型模块

定义缺陷类型、缺陷参数、仿真结果、敏感度分析等数据结构。
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class DefectType(Enum):
    """缺陷类型枚举"""
    POINT = "point"                     # 点缺陷
    LINE = "line"                       # 线缺陷
    CONTAMINATION = "contamination"     # 污染斑


class DefectPolarity(Enum):
    """缺陷极性枚举

    - OPAQUE: 不透明缺陷（多余铬，在掩模透明区域增加遮光材料）
    - CLEAR: 透明缺陷（缺少铬，在掩模不透明区域去除遮光材料）
    """
    OPAQUE = "opaque"
    CLEAR = "clear"


@dataclass
class DefectLocation:
    """
    缺陷位置定义

    Attributes:
        y: 纵向像素坐标（掩模坐标系）
        x: 横向像素坐标（掩模坐标系）
        distance_to_edge: 到最近图案边缘的距离 (nm)，用于敏感度分析
    """
    y: float
    x: float
    distance_to_edge: Optional[float] = None


@dataclass
class PointDefect:
    """
    点缺陷参数

    Attributes:
        size_nm: 缺陷直径 (nm)
        shape: 缺陷形状，'circle' 圆形 或 'square' 方形
        polarity: 缺陷极性
        location: 缺陷位置
    """
    size_nm: float
    shape: str = "circle"
    polarity: DefectPolarity = DefectPolarity.OPAQUE
    location: Optional[DefectLocation] = None

    def __post_init__(self):
        if self.shape not in ['circle', 'square']:
            raise ValueError(f"不支持的点缺陷形状: {self.shape}，可选 'circle' 或 'square'")


@dataclass
class LineDefect:
    """
    线缺陷参数

    Attributes:
        length_nm: 缺陷长度 (nm)
        width_nm: 缺陷宽度 (nm)
        angle_deg: 缺陷方向角度（度，0为水平，90为垂直）
        polarity: 缺陷极性
        location: 缺陷中心位置
    """
    length_nm: float
    width_nm: float
    angle_deg: float = 0.0
    polarity: DefectPolarity = DefectPolarity.OPAQUE
    location: Optional[DefectLocation] = None


@dataclass
class ContaminationDefect:
    """
    污染斑缺陷参数

    模拟掩模表面的颗粒污染，具有不规则边缘和衰减透射特性。

    Attributes:
        size_nm: 污染斑等效直径 (nm)
        attenuation: 透射衰减系数 (0~1)，0表示完全不透明，1表示完全透明
        roughness: 边缘粗糙度 (0~1)，0为光滑，1为高度不规则
        polarity: 缺陷极性
        location: 缺陷中心位置
    """
    size_nm: float
    attenuation: float = 0.7
    roughness: float = 0.3
    polarity: DefectPolarity = DefectPolarity.OPAQUE
    location: Optional[DefectLocation] = None

    def __post_init__(self):
        if not 0 <= self.attenuation <= 1:
            raise ValueError(f"attenuation 必须在 [0, 1] 范围内，当前: {self.attenuation}")
        if not 0 <= self.roughness <= 1:
            raise ValueError(f"roughness 必须在 [0, 1] 范围内，当前: {self.roughness}")


@dataclass
class DefectInjectionConfig:
    """
    缺陷注入配置

    Attributes:
        pixel_size: 掩模像素尺寸 (nm/pixel)
        cd_target: 目标关键尺寸 (nm)，用于失效判定
        cd_tolerance: CD 相对容差 (0~1)，默认 10%
        threshold: 光刻胶阈值
        random_seed: 随机种子，用于污染斑等随机缺陷的复现
    """
    pixel_size: float = 1.0
    cd_target: Optional[float] = None
    cd_tolerance: float = 0.1
    threshold: float = 0.3
    random_seed: Optional[int] = None

    @property
    def cd_lower(self) -> Optional[float]:
        if self.cd_target is not None:
            return self.cd_target * (1.0 - self.cd_tolerance)
        return None

    @property
    def cd_upper(self) -> Optional[float]:
        if self.cd_target is not None:
            return self.cd_target * (1.0 + self.cd_tolerance)
        return None


@dataclass
class SingleDefectResult:
    """
    单个缺陷的仿真结果

    Attributes:
        defect_type: 缺陷类型
        defect_params: 缺陷参数字典
        nominal_cd: 标称 CD (nm)
        defective_cd: 有缺陷时的 CD (nm)
        delta_cd: CD 变化量 (nm)
        delta_cd_relative: 相对 CD 变化 (%)
        nominal_wafer: 标称晶圆图像
        defective_wafer: 含缺陷晶圆图像
        nominal_aerial: 标称空间像
        defective_aerial: 含缺陷空间像
        mask_defective: 注入缺陷后的掩模
        is_critical: 是否为致命缺陷（CD超出容差）
        failure_probability: 失效概率估计
        sensitivity_score: 缺陷敏感度评分
        measurement_lines: 测量线定义（用于复现 CD 提取）
    """
    defect_type: DefectType
    defect_params: Dict[str, Any]
    nominal_cd: float
    defective_cd: float
    delta_cd: float
    delta_cd_relative: float
    nominal_wafer: Optional[np.ndarray] = None
    defective_wafer: Optional[np.ndarray] = None
    nominal_aerial: Optional[np.ndarray] = None
    defective_aerial: Optional[np.ndarray] = None
    mask_defective: Optional[np.ndarray] = None
    is_critical: bool = False
    failure_probability: float = 0.0
    sensitivity_score: float = 0.0
    measurement_lines: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'defect_type': self.defect_type.value,
            'defect_params': self.defect_params,
            'nominal_cd': float(self.nominal_cd),
            'defective_cd': float(self.defective_cd),
            'delta_cd': float(self.delta_cd),
            'delta_cd_relative': float(self.delta_cd_relative),
            'is_critical': bool(self.is_critical),
            'failure_probability': float(self.failure_probability),
            'sensitivity_score': float(self.sensitivity_score),
        }
        return result


@dataclass
class DefectSensitivityEntry:
    """
    缺陷敏感度排序表条目

    Attributes:
        rank: 敏感度排名（1为最敏感）
        defect_type: 缺陷类型
        size_nm: 缺陷特征尺寸 (nm)
        polarity: 缺陷极性
        location: 缺陷位置描述
        delta_cd_abs: 绝对 CD 变化量 (nm)
        delta_cd_relative: 相对 CD 变化 (%)
        is_critical: 是否为致命缺陷
        failure_probability: 失效概率
        sensitivity_score: 综合敏感度评分
        recommendation: 掩模检测规格建议
    """
    rank: int
    defect_type: DefectType
    size_nm: float
    polarity: DefectPolarity
    location: str
    delta_cd_abs: float
    delta_cd_relative: float
    is_critical: bool
    failure_probability: float
    sensitivity_score: float
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rank': self.rank,
            'defect_type': self.defect_type.value,
            'size_nm': float(self.size_nm),
            'polarity': self.polarity.value,
            'location': self.location,
            'delta_cd_abs': float(self.delta_cd_abs),
            'delta_cd_relative': float(self.delta_cd_relative),
            'is_critical': bool(self.is_critical),
            'failure_probability': float(self.failure_probability),
            'sensitivity_score': float(self.sensitivity_score),
            'recommendation': self.recommendation,
        }


@dataclass
class DefectSensitivityReport:
    """
    缺陷敏感度完整报告

    Attributes:
        entries: 敏感度排序表条目
        total_defects_analyzed: 分析的缺陷总数
        critical_defect_count: 致命缺陷数量
        critical_defect_ratio: 致命缺陷比例
        recommended_spec: 推荐的掩模检测规格 (nm)
        summary_stats: 汇总统计数据
        nominal_cd: 标称 CD (nm)
        cd_tolerance: CD 相对容差
    """
    entries: List[DefectSensitivityEntry] = field(default_factory=list)
    total_defects_analyzed: int = 0
    critical_defect_count: int = 0
    critical_defect_ratio: float = 0.0
    recommended_spec: float = 0.0
    summary_stats: Dict[str, float] = field(default_factory=dict)
    nominal_cd: float = 0.0
    cd_tolerance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entries': [e.to_dict() for e in self.entries],
            'total_defects_analyzed': self.total_defects_analyzed,
            'critical_defect_count': self.critical_defect_count,
            'critical_defect_ratio': float(self.critical_defect_ratio),
            'recommended_spec': float(self.recommended_spec),
            'summary_stats': self.summary_stats,
            'nominal_cd': float(self.nominal_cd),
            'cd_tolerance': float(self.cd_tolerance),
        }

    def summary(self) -> str:
        lines = [
            "=== 掩模缺陷打印性分析报告 ===",
            f"  分析缺陷总数: {self.total_defects_analyzed}",
            f"  致命缺陷数: {self.critical_defect_count} "
            f"({self.critical_defect_ratio * 100:.1f}%)",
            f"  标称 CD: {self.nominal_cd:.1f} nm (容差 ±{self.cd_tolerance * 100:.0f}%)",
            f"  推荐检测规格: {self.recommended_spec:.1f} nm",
            "",
        ]
        if self.summary_stats:
            lines.append("  汇总统计:")
            for k, v in self.summary_stats.items():
                lines.append(f"    {k}: {v:.3f}")
            lines.append("")
        lines.append("  敏感度排名 (Top 10):")
        lines.append(
            f"    {'#':>3}  {'类型':<14}  {'尺寸(nm)':>8}  {'极性':<8}  "
            f"{'|ΔCD|(nm)':>9}  {'ΔCD(%)':>7}  {'失效概率':>8}  {'致命':<4}"
        )
        for e in self.entries[:10]:
            lines.append(
                f"    {e.rank:>3d}  {e.defect_type.value:<14}  {e.size_nm:>8.1f}  "
                f"{e.polarity.value:<8}  {e.delta_cd_abs:>9.2f}  "
                f"{e.delta_cd_relative:>6.1f}%  {e.failure_probability:>7.3f}  "
                f"{'YES' if e.is_critical else 'NO':<4}"
            )
        return "\n".join(lines)
