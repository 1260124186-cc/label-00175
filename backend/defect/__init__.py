# -*- coding: utf-8 -*-
"""
掩模缺陷打印性分析模块

支持在掩模指定位置注入点缺陷、线缺陷或污染斑，
仿真其对晶圆成像的影响，计算缺陷诱导 CD 变化与失效概率，
输出缺陷敏感度排序表，供掩模检测规格制定参考。

核心功能:
1. 缺陷注入: 点缺陷、线缺陷、污染斑
2. 成像仿真: 基于现有光刻成像模型计算缺陷对晶圆的影响
3. CD 变化分析: 缺陷诱导的关键尺寸变化计算
4. 失效概率评估: 基于 CD 容差的失效概率估计
5. 敏感度排序: 按缺陷类型/尺寸/位置输出敏感度排序表
"""

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
from defect.sensitivity import DefectSensitivityAnalyzer, DefectScanConfig, run_defect_analysis

__all__ = [
    'DefectType',
    'DefectPolarity',
    'PointDefect',
    'LineDefect',
    'ContaminationDefect',
    'DefectLocation',
    'DefectInjectionConfig',
    'DefectSensitivityEntry',
    'DefectSensitivityReport',
    'SingleDefectResult',
    'DefectInjector',
    'DefectSimulator',
    'DefectSensitivityAnalyzer',
    'DefectScanConfig',
    'run_defect_analysis',
]
