# -*- coding: utf-8 -*-
"""
标定数据结构定义

定义 CD-SEM 量测数据、待标定参数、标定配置与结果等数据结构。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PatternType(Enum):
    """图形类型枚举"""
    LINE_SPACE = "line_space"
    CONTACT_HOLE = "contact_hole"
    ISOLATED_LINE = "isolated_line"
    ISOLATED_SPACE = "isolated_space"
    LINE_END = "line_end"
    CORNER = "corner"
    CUSTOM = "custom"


class InversionMethod(Enum):
    """参数反演方法枚举

    - NLLS: 非线性最小二乘法 (scipy.optimize.least_squares)
    - LMFIT: Levenberg-Marquardt (scipy.optimize.curve_fit)
    - BAYESIAN_MCMC: 贝叶斯 MCMC (emcee/pymc 可选)
    - BOTH: 先 NLLS 得到初值，再用 MCMC 采样后验
    """
    NLLS = "nlls"
    LMFIT = "lmfit"
    BAYESIAN_MCMC = "bayesian_mcmc"
    BOTH = "both"


@dataclass
class CalibrationParameter:
    """
    单个待标定参数定义

    Attributes:
        name: 参数名称（需匹配 forward_model 中的键名）
        initial_value: 初始猜测值
        lower_bound: 下界（None 表示无约束）
        upper_bound: 上界（None 表示无约束）
        unit: 参数单位描述
        description: 参数物理含义
        vary: 是否参与反演（False 表示固定为 initial_value）
        prior_mean: 贝叶斯先验均值（仅 BAYESIAN 使用）
        prior_std: 贝叶斯先验标准差（仅 BAYESIAN 使用，None 表示均匀先验）
        config_path: 对应 default_config.yaml 中的路径，如 'optical_system.na'
    """
    name: str
    initial_value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    unit: str = ""
    description: str = ""
    vary: bool = True
    prior_mean: Optional[float] = None
    prior_std: Optional[float] = None
    config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'initial_value': self.initial_value,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'unit': self.unit,
            'description': self.description,
            'vary': self.vary,
            'prior_mean': self.prior_mean,
            'prior_std': self.prior_std,
            'config_path': self.config_path,
        }

    @property
    def bounds_tuple(self) -> Tuple[float, float]:
        lb = -np.inf if self.lower_bound is None else self.lower_bound
        ub = np.inf if self.upper_bound is None else self.upper_bound
        return (lb, ub)


@dataclass
class CalibrationParameterSet:
    """
    待标定参数集合

    包含光刻系统的关键可校准参数：
    - resist_threshold: 光刻胶阈值
    - diffusion_length: PEB 酸扩散长度 (nm)
    - na_effective: 有效数值孔径（考虑偏振效应等的修正值）
    - dose_to_clear: 光刻胶清零剂量（相对单位）
    - resist_contrast: 光刻胶对比度 (γ 值)
    - sigma_effective: 有效部分相干因子
    - wavelength_effective: 有效波长 (nm)
    """
    resist_threshold: CalibrationParameter = field(default_factory=lambda: CalibrationParameter(
        name="resist_threshold",
        initial_value=0.30,
        lower_bound=0.05,
        upper_bound=0.80,
        unit="norm.intensity",
        description="光刻胶显影阈值（归一化光强）",
        config_path="imaging.resist_threshold",
    ))
    diffusion_length: CalibrationParameter = field(default_factory=lambda: CalibrationParameter(
        name="diffusion_length",
        initial_value=10.0,
        lower_bound=0.0,
        upper_bound=50.0,
        unit="nm",
        description="PEB 后酸扩散长度的 RMS 值",
        config_path="imaging.diffusion_length",
    ))
    na_effective: CalibrationParameter = field(default_factory=lambda: CalibrationParameter(
        name="na_effective",
        initial_value=1.35,
        lower_bound=0.80,
        upper_bound=1.60,
        unit="-",
        description="有效数值孔径（偏振/填充因子修正）",
        config_path="optical_system.na",
    ))
    dose_to_clear: CalibrationParameter = field(default_factory=lambda: CalibrationParameter(
        name="dose_to_clear",
        initial_value=0.5,
        lower_bound=0.05,
        upper_bound=2.0,
        unit="relative",
        description="大开阔区域刚好完全曝光的剂量（相对标称剂量）",
    ))
    resist_contrast: CalibrationParameter = field(default_factory=lambda: CalibrationParameter(
        name="resist_contrast",
        initial_value=3.0,
        lower_bound=1.0,
        upper_bound=10.0,
        unit="-",
        description="光刻胶对比度 γ (H-D 曲线斜率)",
    ))
    sigma_effective: CalibrationParameter = field(default_factory=lambda: CalibrationParameter(
        name="sigma_effective",
        initial_value=0.75,
        lower_bound=0.10,
        upper_bound=0.99,
        unit="-",
        description="有效部分相干因子",
        config_path="optical_system.sigma",
    ))
    wavelength_effective: CalibrationParameter = field(default_factory=lambda: CalibrationParameter(
        name="wavelength_effective",
        initial_value=193.0,
        lower_bound=190.0,
        upper_bound=196.0,
        unit="nm",
        description="有效光源波长（考虑吸收偏移等）",
        vary=False,
        config_path="optical_system.wavelength",
    ))

    def get_varying_parameters(self) -> List[CalibrationParameter]:
        return [p for p in self.all_parameters() if p.vary]

    def get_fixed_parameters(self) -> List[CalibrationParameter]:
        return [p for p in self.all_parameters() if not p.vary]

    def all_parameters(self) -> List[CalibrationParameter]:
        return [
            self.resist_threshold,
            self.diffusion_length,
            self.na_effective,
            self.dose_to_clear,
            self.resist_contrast,
            self.sigma_effective,
            self.wavelength_effective,
        ]

    def param_dict(self) -> Dict[str, float]:
        return {p.name: p.initial_value for p in self.all_parameters()}

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        varying = self.get_varying_parameters()
        lbs = np.array([p.bounds_tuple[0] for p in varying])
        ubs = np.array([p.bounds_tuple[1] for p in varying])
        return lbs, ubs

    def varying_names(self) -> List[str]:
        return [p.name for p in self.get_varying_parameters()]


@dataclass
class CDSEMDataPoint:
    """
    单个 CD-SEM 量测点

    Attributes:
        target_cd: 目标 CD (nm)，即设计值
        measured_cd: CD-SEM 实测值 (nm)
        focus: 离焦量条件 (nm)，正值为过焦，负值为欠焦
        dose: 曝光剂量（相对标称剂量，1.0 = 标称剂量）
        pattern_type: 图形类型
        pitch: 图形节距 (nm)，line-space 为周期；contact 为节距
        measurement_id: 量测编号
        site_name: 量测站点名称（如 "Center", "TopLeft"）
        measurement_uncertainty: CD-SEM 量测不确定度 (nm, 1σ)，默认 1.0 nm
        mask_cd: 掩模 CD (nm)，可选；不提供则用 target_cd * 放大倍率
        layer: 工艺层标签
        timestamp: 量测时间戳
        extra: 额外的关键字段（保存任意附加信息）
    """
    target_cd: float
    measured_cd: float
    focus: float
    dose: float
    pattern_type: PatternType = PatternType.LINE_SPACE
    pitch: Optional[float] = None
    measurement_id: str = ""
    site_name: str = ""
    measurement_uncertainty: float = 1.0
    mask_cd: Optional[float] = None
    layer: str = ""
    timestamp: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'measurement_id': self.measurement_id,
            'site_name': self.site_name,
            'target_cd': self.target_cd,
            'measured_cd': self.measured_cd,
            'focus': self.focus,
            'dose': self.dose,
            'pattern_type': self.pattern_type.value,
            'pitch': self.pitch,
            'measurement_uncertainty': self.measurement_uncertainty,
            'mask_cd': self.mask_cd,
            'layer': self.layer,
            'timestamp': self.timestamp,
            **self.extra,
        }


@dataclass
class CDSEMDataset:
    """
    CD-SEM 量测数据集

    Attributes:
        points: 所有量测点
        magnification: 投影物镜放大倍率（用于 mask_cd 推断）
        fab_name: FAB 厂名称
        process_node: 工艺节点（如 "14nm", "7nm", "5nm"）
        mask_set_id: 掩模组编号
        wafer_id: 晶圆编号
        lot_id: 批次编号
        comments: 备注说明
    """
    points: List[CDSEMDataPoint] = field(default_factory=list)
    magnification: float = 4.0
    fab_name: str = ""
    process_node: str = ""
    mask_set_id: str = ""
    wafer_id: str = ""
    lot_id: str = ""
    comments: str = ""

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> CDSEMDataPoint:
        return self.points[idx]

    def __iter__(self):
        return iter(self.points)

    def add_point(self, point: CDSEMDataPoint) -> None:
        self.points.append(point)

    def focus_dose_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        focuses = np.array([p.focus for p in self.points])
        doses = np.array([p.dose for p in self.points])
        return focuses, doses

    def measured_cds(self) -> np.ndarray:
        return np.array([p.measured_cd for p in self.points])

    def target_cds(self) -> np.ndarray:
        return np.array([p.target_cd for p in self.points])

    def pitches(self) -> np.ndarray:
        return np.array([p.pitch if p.pitch is not None else 2.0 * p.target_cd for p in self.points])

    def pattern_types(self) -> List[PatternType]:
        return [p.pattern_type for p in self.points]

    def uncertainties(self) -> np.ndarray:
        return np.array([p.measurement_uncertainty for p in self.points])

    def focus_range(self) -> Tuple[float, float]:
        fs = [p.focus for p in self.points]
        return (min(fs), max(fs))

    def dose_range(self) -> Tuple[float, float]:
        ds = [p.dose for p in self.points]
        return (min(ds), max(ds))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'fab_name': self.fab_name,
            'process_node': self.process_node,
            'mask_set_id': self.mask_set_id,
            'wafer_id': self.wafer_id,
            'lot_id': self.lot_id,
            'magnification': self.magnification,
            'comments': self.comments,
            'n_points': len(self.points),
            'points': [p.to_dict() for p in self.points],
        }

    def filter_by_pattern(self, pattern_type: PatternType) -> 'CDSEMDataset':
        filtered = [p for p in self.points if p.pattern_type == pattern_type]
        return CDSEMDataset(
            points=filtered,
            magnification=self.magnification,
            fab_name=self.fab_name,
            process_node=self.process_node,
            mask_set_id=self.mask_set_id,
            wafer_id=self.wafer_id,
            lot_id=self.lot_id,
            comments=f"{self.comments}; filtered by {pattern_type.value}",
        )


@dataclass
class CalibrationConfig:
    """
    标定任务完整配置

    Attributes:
        method: 反演方法
        parameters: 待标定参数集合
        dataset_path: CD-SEM 数据文件路径
        output_dir: 输出目录
        random_seed: 随机种子
        use_measurement_weights: 是否用量测不确定度做加权
        forward_model_complexity: 前向模型复杂度 ('simple', 'standard', 'detailed')
        nlls_max_iter: NLLS 最大迭代次数
        nlls_method: NLLS 求解方法 ('trf', 'dogbox', 'lm')
        mcmc_n_walkers: MCMC walker 数量
        mcmc_n_steps: MCMC 每 walker 步数
        mcmc_n_burnin: MCMC burn-in 步数
        mcmc_progress: 是否显示 MCMC 进度
        generate_plots: 是否生成图表
        plot_format: 图表格式 ('png', 'svg', 'pdf')
        update_config: 是否生成更新后的配置片段
    """
    method: InversionMethod = InversionMethod.LMFIT
    parameters: CalibrationParameterSet = field(default_factory=CalibrationParameterSet)
    dataset_path: Optional[str] = None
    output_dir: str = "./calibration_results"
    random_seed: Optional[int] = 42
    use_measurement_weights: bool = True
    forward_model_complexity: str = "standard"
    nlls_max_iter: int = 10000
    nlls_method: str = "trf"
    nlls_ftol: float = 1e-12
    nlls_xtol: float = 1e-12
    mcmc_n_walkers: int = 32
    mcmc_n_steps: int = 5000
    mcmc_n_burnin: int = 1000
    mcmc_progress: bool = True
    generate_plots: bool = True
    plot_format: str = "png"
    update_config: bool = True
    reference_config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method.value,
            'output_dir': self.output_dir,
            'random_seed': self.random_seed,
            'use_measurement_weights': self.use_measurement_weights,
            'forward_model_complexity': self.forward_model_complexity,
            'nlls_max_iter': self.nlls_max_iter,
            'nlls_method': self.nlls_method,
            'nlls_ftol': self.nlls_ftol,
            'nlls_xtol': self.nlls_xtol,
            'mcmc_n_walkers': self.mcmc_n_walkers,
            'mcmc_n_steps': self.mcmc_n_steps,
            'mcmc_n_burnin': self.mcmc_n_burnin,
            'generate_plots': self.generate_plots,
            'plot_format': self.plot_format,
            'update_config': self.update_config,
            'parameters': [p.to_dict() for p in self.parameters.all_parameters()],
        }


@dataclass
class InversionResult:
    """
    参数反演结果

    Attributes:
        method: 使用的反演方法
        success: 是否收敛成功
        message: 收敛/失败消息
        calibrated_values: 反演得到的参数值 {name: value}
        uncertainties: 参数 1σ 不确定度 {name: ±value}
        covariance_matrix: 参数协方差矩阵 (n_vary × n_vary)
        correlation_matrix: 参数相关系数矩阵
        varying_names: 参与反演的参数名列表（协方差矩阵行列顺序）
        cost: 最终残差平方和（或对数似然）
        chi2: 卡方值（加权时）
        reduced_chi2: 约化卡方 (chi2 / dof)
        n_data: 数据点数量
        n_params: 自由参数数量
        dof: 自由度 (n_data - n_params)
        residuals: 每个数据点的残差 (measured - predicted)
        predicted_cds: 每个数据点的模型预测 CD
        iterations: 实际迭代次数
        mcmc_samples: MCMC 采样结果 (n_steps, n_walkers, n_params)，仅 BAYESIAN
        mcmc_acceptance: 平均接受率
        cost_history: 每次迭代的代价函数历史（用于调试）
    """
    method: InversionMethod
    success: bool
    message: str
    calibrated_values: Dict[str, float]
    uncertainties: Dict[str, float]
    covariance_matrix: Optional[np.ndarray] = None
    correlation_matrix: Optional[np.ndarray] = None
    varying_names: List[str] = field(default_factory=list)
    cost: float = 0.0
    chi2: float = 0.0
    reduced_chi2: float = 0.0
    n_data: int = 0
    n_params: int = 0
    dof: int = 0
    residuals: Optional[np.ndarray] = None
    predicted_cds: Optional[np.ndarray] = None
    iterations: int = 0
    mcmc_samples: Optional[np.ndarray] = None
    mcmc_acceptance: float = 0.0
    cost_history: Optional[np.ndarray] = None

    def summary_table(self) -> str:
        lines = [
            f"{'参数名称':<24s} {'标定值':>14s} {'±1σ':>12s} {'单位':<16s}",
            "-" * 70,
        ]
        for name, val in self.calibrated_values.items():
            unc = self.uncertainties.get(name, 0.0)
            lines.append(f"{name:<24s} {val:>14.6f} ±{unc:<11.6f}")
        return "\n".join(lines)


@dataclass
class CalibrationReport:
    """
    完整标定报告

    Attributes:
        config: 标定配置副本
        dataset_info: 数据集基本信息
        inversion_result: 参数反演结果
        metrics: 标定质量指标
        config_snippet: default_config.yaml 片段（已更新参数）
        plots: 生成的图表文件路径列表
        timestamp: 报告生成时间戳
        duration_sec: 反演耗时 (秒)
    """
    config: CalibrationConfig
    dataset_info: Dict[str, Any]
    inversion_result: InversionResult
    metrics: Dict[str, float] = field(default_factory=dict)
    config_snippet: Optional[str] = None
    plots: List[str] = field(default_factory=list)
    timestamp: str = ""
    duration_sec: float = 0.0

    def summary(self) -> str:
        r = self.inversion_result
        lines = [
            "=" * 70,
            " Fab 模型标定报告 ",
            "=" * 70,
            "",
            f"  生成时间:    {self.timestamp}",
            f"  反演耗时:    {self.duration_sec:.2f} s",
            f"  反演方法:    {r.method.value}",
            f"  收敛状态:    {'成功' if r.success else '失败'}",
            f"  数据点数:    {r.n_data}",
            f"  自由参数:    {r.n_params}",
            f"  自由度:      {r.dof}",
            f"  最终代价:    {r.cost:.6e}",
            f"  卡方 χ²:     {r.chi2:.4f}",
            f"  约化卡方:    {r.reduced_chi2:.4f}",
            "",
            "  标定参数:",
            r.summary_table(),
            "",
            "  质量指标:",
        ]
        for k, v in self.metrics.items():
            lines.append(f"    {k:<24s}: {v:.6e}")
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)
