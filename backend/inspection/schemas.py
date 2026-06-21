# -*- coding: utf-8 -*-
"""
掩模检测数据模型模块

定义掩模检测图像仿真、die-to-database 差异分析、
可检测性与假缺陷率分析等数据结构。
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class InspectionMode(Enum):
    """检测模式枚举"""
    BRIGHT_FIELD = "bright_field"      # 明场检测
    DARK_FIELD = "dark_field"          # 暗场检测
    PHASE_CONTRAST = "phase_contrast"  # 相位对比检测
    POLARIZATION = "polarization"      # 偏振检测


class DefectClass(Enum):
    """缺陷分类"""
    REAL_DEFECT = "real_defect"           # 真实缺陷（印刷致命）
    NUISANCE_DEFECT = "nuisance_defect"   # 假缺陷（不影响印刷）
    NO_DEFECT = "no_defect"               # 无缺陷


class DieType(Enum):
    """Die 类型"""
    TEST_DIE = "test_die"       # 待测 Die
    REFERENCE_DIE = "reference_die"  # 参考 Die（数据库）


@dataclass
class InspectionOptics:
    """
    检测光学系统参数

    Attributes:
        wavelength_nm: 照明波长 (nm)
        numerical_aperture: 数值孔径 (NA)
        magnification: 光学放大倍率
        pixel_size_nm: 图像像素尺寸 (nm/pixel)
        illumination_na: 照明 NA (用于暗场检测)
        collection_na: 收集 NA (用于暗场检测)
        polarization_state: 偏振态 's', 'p', 'circular', 或 'unpolarized'
    """
    wavelength_nm: float = 266.0
    numerical_aperture: float = 0.9
    magnification: float = 200.0
    pixel_size_nm: float = 1.0
    illumination_na: float = 0.85
    collection_na: float = 0.9
    polarization_state: str = "unpolarized"


@dataclass
class InspectionConfig:
    """
    掩模检测配置

    Attributes:
        mode: 检测模式（明场/暗场等）
        optics: 检测光学系统参数
        noise_level: 检测器噪声水平 (0~1)
        contrast_enhancement: 对比度增强强度 (0~3)
        defect_boost: 缺陷信号增强倍数 (1~5)
        threshold_abs: 绝对检测阈值 (0~1)
        threshold_rel: 相对检测阈值（标准差倍数）
        blur_sigma: 光学模糊 sigma (像素)
        gamma: Gamma 校正指数
        adaptive_threshold: 是否使用自适应阈值
        min_defect_size_nm: 最小可检测缺陷尺寸 (nm)
    """
    mode: InspectionMode = InspectionMode.BRIGHT_FIELD
    optics: InspectionOptics = field(default_factory=InspectionOptics)
    noise_level: float = 0.03
    contrast_enhancement: float = 1.0
    defect_boost: float = 1.5
    threshold_abs: float = 0.15
    threshold_rel: float = 3.0
    blur_sigma: float = 0.5
    gamma: float = 1.0
    adaptive_threshold: bool = True
    min_defect_size_nm: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d['mode'] = self.mode.value
        d['optics'] = self.optics.__dict__.copy()
        return d


@dataclass
class InspectionImageResult:
    """
    检测图像仿真结果

    Attributes:
        inspection_image: 生成的检测图像 (0~1 归一化)
        reference_image: 参考图像（无缺陷）
        defect_mask: 缺陷位置二值掩模
        edge_map: 边缘强度图
        noise_map: 添加的噪声图
        config: 使用的检测配置
        mode: 检测模式
    """
    inspection_image: np.ndarray
    reference_image: np.ndarray
    defect_mask: np.ndarray
    edge_map: np.ndarray
    noise_map: np.ndarray
    config: InspectionConfig
    mode: InspectionMode

    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_shape': list(self.inspection_image.shape),
            'mode': self.mode.value,
            'config': self.config.to_dict(),
        }


@dataclass
class DifferenceMapResult:
    """
    Die-to-Database 差异图计算结果

    Attributes:
        difference_map: 差异图（已取绝对值）
        signed_difference: 带符号差异图
        thresholded_map: 二值化差异图（超阈值区域）
        threshold_used: 使用的检测阈值
        candidate_regions: 候选缺陷区域列表
        difference_histogram: 差异值直方图统计
        mean_difference: 平均绝对差异
        max_difference: 最大绝对差异
        std_difference: 差异标准差
    """
    difference_map: np.ndarray
    signed_difference: np.ndarray
    thresholded_map: np.ndarray
    threshold_used: float
    candidate_regions: List[Dict[str, Any]]
    difference_histogram: Dict[str, np.ndarray]
    mean_difference: float
    max_difference: float
    std_difference: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'map_shape': list(self.difference_map.shape),
            'threshold_used': float(self.threshold_used),
            'mean_difference': float(self.mean_difference),
            'max_difference': float(self.max_difference),
            'std_difference': float(self.std_difference),
            'num_candidate_regions': int(len(self.candidate_regions)),
        }


@dataclass
class DefectCandidate:
    """
    缺陷候选检测结果

    Attributes:
        center_y: 中心 Y 坐标 (像素)
        center_x: 中心 X 坐标 (像素)
        size_nm: 缺陷等效尺寸 (nm)
        area_pixels: 缺陷面积 (像素)
        contrast: 缺陷对比度 (0~1)
        intensity: 缺陷平均强度
        score: 检测置信度分数 (0~1)
        defect_class: 缺陷分类结果
        is_printable: 是否为印刷致命缺陷
        bounding_box: 边界框 (y1, y2, x1, x2)
    """
    center_y: int
    center_x: int
    size_nm: float
    area_pixels: int
    contrast: float
    intensity: float
    score: float
    defect_class: DefectClass
    is_printable: bool
    bounding_box: Tuple[int, int, int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'center_y': int(self.center_y),
            'center_x': int(self.center_x),
            'size_nm': float(self.size_nm),
            'area_pixels': int(self.area_pixels),
            'contrast': float(self.contrast),
            'intensity': float(self.intensity),
            'score': float(self.score),
            'defect_class': self.defect_class.value,
            'is_printable': bool(self.is_printable),
            'bounding_box': list(self.bounding_box),
        }


@dataclass
class DetectabilityResult:
    """
    可检测性分析结果

    Attributes:
        detected_defects: 检测到的缺陷候选列表
        true_positives: 真实缺陷被正确检测的数量
        false_positives: 假缺陷（误检）数量
        false_negatives: 漏检真实缺陷数量
        detection_rate: 检测率 (TP / (TP + FN))
        false_alarm_rate: 假警报率 (FP / 总检测数)
        precision: 准确率 (TP / (TP + FP))
        f1_score: F1 分数
        roc_data: ROC 曲线数据 (fpr, tpr, thresholds)
        auc_score: AUC 分数
        optimal_threshold: 最优检测阈值
        pixel_size_nm: 像素尺寸 (nm/pixel)
    """
    detected_defects: List[DefectCandidate] = field(default_factory=list)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    detection_rate: float = 0.0
    false_alarm_rate: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    roc_data: Dict[str, np.ndarray] = field(default_factory=dict)
    auc_score: float = 0.0
    optimal_threshold: float = 0.0
    pixel_size_nm: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'true_positives': int(self.true_positives),
            'false_positives': int(self.false_positives),
            'false_negatives': int(self.false_negatives),
            'detection_rate': float(self.detection_rate),
            'false_alarm_rate': float(self.false_alarm_rate),
            'precision': float(self.precision),
            'f1_score': float(self.f1_score),
            'auc_score': float(self.auc_score),
            'optimal_threshold': float(self.optimal_threshold),
            'num_detected': int(len(self.detected_defects)),
        }

    def summary(self) -> str:
        lines = [
            "=== 掩模可检测性分析报告 ===",
            f"  真实缺陷数: {self.true_positives + self.false_negatives}",
            f"  检测到缺陷数: {len(self.detected_defects)}",
            f"  正确检测 (TP): {self.true_positives}",
            f"  误检 (FP): {self.false_positives}",
            f"  漏检 (FN): {self.false_negatives}",
            f"  检测率: {self.detection_rate * 100:.1f}%",
            f"  假警报率: {self.false_alarm_rate * 100:.1f}%",
            f"  准确率 (Precision): {self.precision * 100:.1f}%",
            f"  F1 分数: {self.f1_score:.3f}",
            f"  AUC: {self.auc_score:.3f}",
            f"  最优阈值: {self.optimal_threshold:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class InspectionAnalysisConfig:
    """
    完整检测分析配置

    Attributes:
        inspection_config: 检测图像仿真配置
        diff_threshold_abs: 差异图绝对阈值
        diff_threshold_rel: 差异图相对阈值（标准差倍数）
        min_area_pixels: 最小缺陷面积 (像素)
        max_area_pixels: 最大缺陷面积 (像素)
        connectivity: 连通域分析连接性 (4 或 8)
        printability_cd_tolerance: 印刷性 CD 容差 (0~1)
        save_intermediate: 是否保存中间结果
    """
    inspection_config: InspectionConfig = field(default_factory=InspectionConfig)
    diff_threshold_abs: float = 0.1
    diff_threshold_rel: float = 3.0
    min_area_pixels: int = 3
    max_area_pixels: Optional[int] = None
    connectivity: int = 8
    printability_cd_tolerance: float = 0.1
    save_intermediate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d['inspection_config'] = self.inspection_config.to_dict()
        return d


@dataclass
class FullInspectionResult:
    """
    完整掩模检测分析结果

    Attributes:
        inspection_result: 检测图像仿真结果
        difference_result: Die-to-Database 差异图结果
        detectability_result: 可检测性分析结果
        config: 使用的完整配置
        ground_truth_defects: 真实缺陷信息（如果有）
        total_analysis_time_s: 总分析时间 (秒)
    """
    inspection_result: Optional[InspectionImageResult] = None
    difference_result: Optional[DifferenceMapResult] = None
    detectability_result: Optional[DetectabilityResult] = None
    config: InspectionAnalysisConfig = field(default_factory=InspectionAnalysisConfig)
    ground_truth_defects: List[Dict[str, Any]] = field(default_factory=list)
    total_analysis_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'inspection_result': self.inspection_result.to_dict() if self.inspection_result else None,
            'difference_result': self.difference_result.to_dict() if self.difference_result else None,
            'detectability_result': self.detectability_result.to_dict() if self.detectability_result else None,
            'config': self.config.to_dict(),
            'num_ground_truth_defects': int(len(self.ground_truth_defects)),
            'total_analysis_time_s': float(self.total_analysis_time_s),
        }

    def summary(self) -> str:
        lines = []
        if self.detectability_result:
            lines.append(self.detectability_result.summary())
        lines.append(f"\n  总分析时间: {self.total_analysis_time_s:.2f}s")
        return "\n".join(lines)
