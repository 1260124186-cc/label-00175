# -*- coding: utf-8 -*-
"""
实验定义格式与校验模块

YAML 描述输入图案、光学参数、优化器、期望指标范围。
每个实验定义是一个独立的 YAML 文件，可被执行器读取并运行。

实验 YAML 结构:
    experiment:
      name: string           # 实验名称（唯一标识）
      description: string    # 实验描述
      tags: [string]         # 标签列表
      workflow: string       # 工作流类型: mask_optimization | opc | smo

    pattern:                 # 输入图案配置
      type: string           # line_space | contact_hole | l_shaped_corner | t_junction | sram_bitcell
      grid_size: [int, int]  # (ny, nx)
      pixel_size: float      # nm/pixel
      cd: float              # 关键尺寸 (nm)
      pitch: float           # 间距 (nm)
      ...                    # 各类型的额外参数

    optical:                 # 光学系统参数
      wavelength: float
      na: float
      sigma: float
      ...

    optimizer:               # 优化器配置
      type: string           # gradient_descent | adam | bfgs | ...
      max_iter: int
      learning_rate: float
      ...

    assertions:              # 回归断言
      - type: string         # mse_threshold | convergence_steps | golden_deviation | ssim_threshold
        ...
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PatternConfig:
    """
    输入图案配置

    Attributes:
        type: 图案类型
        grid_size: 网格尺寸 (ny, nx)
        pixel_size: 像素尺寸 (nm/pixel)
        cd: 关键尺寸 (nm)
        pitch: 间距 (nm)
        corner_rounding: 拐角圆滑度 (nm)
        extra: 各类型的额外参数
    """
    type: str = 'line_space'
    grid_size: Tuple[int, int] = (64, 64)
    pixel_size: float = 1.0
    cd: float = 45.0
    pitch: float = 90.0
    corner_rounding: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    VALID_TYPES = {
        'line_space', 'contact_hole', 'l_shaped_corner',
        't_junction', 'sram_bitcell',
    }

    def __post_init__(self):
        if self.type not in self.VALID_TYPES:
            raise ValueError(f"无效图案类型: {self.type}, 有效值: {self.VALID_TYPES}")
        if isinstance(self.grid_size, list):
            self.grid_size = tuple(self.grid_size)
        if self.cd <= 0:
            raise ValueError(f"cd 必须为正数: {self.cd}")
        if self.pitch <= self.cd:
            raise ValueError(f"pitch ({self.pitch}) 必须大于 cd ({self.cd})")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PatternConfig':
        reserved = {'type', 'grid_size', 'pixel_size', 'cd', 'pitch', 'corner_rounding'}
        extra = {k: v for k, v in d.items() if k not in reserved}
        grid_size = d.get('grid_size', (64, 64))
        if isinstance(grid_size, list):
            grid_size = tuple(grid_size)
        return cls(
            type=d.get('type', 'line_space'),
            grid_size=grid_size,
            pixel_size=float(d.get('pixel_size', 1.0)),
            cd=float(d.get('cd', 45.0)),
            pitch=float(d.get('pitch', 90.0)),
            corner_rounding=float(d.get('corner_rounding', 0.0)),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'type': self.type,
            'grid_size': list(self.grid_size),
            'pixel_size': self.pixel_size,
            'cd': self.cd,
            'pitch': self.pitch,
            'corner_rounding': self.corner_rounding,
        }
        result.update(self.extra)
        return result


@dataclass
class OpticalConfig:
    """
    光学系统参数配置

    Attributes:
        wavelength: 波长 (nm)
        na: 数值孔径
        sigma: 部分相干因子
        pixel_size: 像素尺寸 (nm)
        defocus: 离焦量 (nm)
        magnification: 放大倍率
        illumination_type: 照明类型
        source_params: 光源参数
        tcc_mode: TCC 计算模式
        socs_num_terms: SOCS 分解项数
        zernike_coefficients: Zernike 像差系数
    """
    wavelength: float = 193.0
    na: float = 1.35
    sigma: float = 0.75
    pixel_size: float = 1.0
    defocus: float = 0.0
    magnification: float = 4.0
    illumination_type: str = 'conventional'
    source_params: Dict[str, Any] = field(default_factory=lambda: {
        'sigma_inner': 0.0, 'sigma_outer': 0.75
    })
    tcc_mode: str = 'socs'
    socs_num_terms: int = 5
    zernike_coefficients: Dict[str, float] = field(default_factory=dict)

    VALID_ILLUMINATION = {'conventional', 'annular', 'dipole', 'quasar', 'custom'}
    VALID_TCC_MODE = {'full_tcc', 'socs', 'kernel_2d'}

    def __post_init__(self):
        if self.wavelength <= 0:
            raise ValueError(f"wavelength 必须为正数: {self.wavelength}")
        if not 0 < self.na <= 2:
            raise ValueError(f"na 必须在 (0, 2] 范围内: {self.na}")
        if not 0 <= self.sigma <= 1:
            raise ValueError(f"sigma 必须在 [0, 1] 范围内: {self.sigma}")
        if self.illumination_type not in self.VALID_ILLUMINATION:
            raise ValueError(f"无效照明类型: {self.illumination_type}")
        if self.tcc_mode not in self.VALID_TCC_MODE:
            raise ValueError(f"无效 TCC 模式: {self.tcc_mode}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OpticalConfig':
        return cls(
            wavelength=float(d.get('wavelength', 193.0)),
            na=float(d.get('na', 1.35)),
            sigma=float(d.get('sigma', 0.75)),
            pixel_size=float(d.get('pixel_size', 1.0)),
            defocus=float(d.get('defocus', 0.0)),
            magnification=float(d.get('magnification', 4.0)),
            illumination_type=d.get('illumination_type', 'conventional'),
            source_params=d.get('source_params', {'sigma_inner': 0.0, 'sigma_outer': 0.75}),
            tcc_mode=d.get('tcc_mode', 'socs'),
            socs_num_terms=int(d.get('socs_num_terms', 5)),
            zernike_coefficients=d.get('zernike_coefficients', {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'wavelength': self.wavelength,
            'na': self.na,
            'sigma': self.sigma,
            'pixel_size': self.pixel_size,
            'defocus': self.defocus,
            'magnification': self.magnification,
            'illumination_type': self.illumination_type,
            'source_params': dict(self.source_params),
            'tcc_mode': self.tcc_mode,
            'socs_num_terms': self.socs_num_terms,
            'zernike_coefficients': dict(self.zernike_coefficients),
        }


@dataclass
class OptimizerConfig:
    """
    优化器配置

    Attributes:
        type: 优化器类型
        max_iter: 最大迭代次数
        learning_rate: 学习率
        tol: 收敛容差
        early_stop_patience: 早停耐心值
        random_seed: 随机种子
        loss_weights: 复合损失权重
        extra: 额外优化器参数
    """
    type: str = 'gradient_descent'
    max_iter: int = 100
    learning_rate: float = 0.01
    tol: float = 1e-6
    early_stop_patience: int = 10
    random_seed: Optional[int] = 42
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'mse': 1.0})
    extra: Dict[str, Any] = field(default_factory=dict)

    VALID_TYPES = {
        'gradient_descent', 'adam', 'rmsprop', 'bfgs', 'newton',
        'genetic', 'pso', 'sa', 'de', 'cmaes',
    }

    def __post_init__(self):
        if self.type not in self.VALID_TYPES:
            raise ValueError(f"无效优化器类型: {self.type}, 有效值: {self.VALID_TYPES}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter 必须为正整数: {self.max_iter}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate 必须为正数: {self.learning_rate}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OptimizerConfig':
        reserved = {
            'type', 'max_iter', 'learning_rate', 'tol',
            'early_stop_patience', 'random_seed', 'loss_weights',
        }
        extra = {k: v for k, v in d.items() if k not in reserved}
        return cls(
            type=d.get('type', 'gradient_descent'),
            max_iter=int(d.get('max_iter', 100)),
            learning_rate=float(d.get('learning_rate', 0.01)),
            tol=float(d.get('tol', 1e-6)),
            early_stop_patience=int(d.get('early_stop_patience', 10)),
            random_seed=d.get('random_seed', 42),
            loss_weights=d.get('loss_weights', {'mse': 1.0}),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'type': self.type,
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate,
            'tol': self.tol,
            'early_stop_patience': self.early_stop_patience,
            'random_seed': self.random_seed,
            'loss_weights': dict(self.loss_weights),
        }
        result.update(self.extra)
        return result


@dataclass
class AssertionConfig:
    """
    回归断言配置

    每个断言描述一个期望指标条件，类型包括:
    - mse_threshold: 最终 MSE 应低于阈值
    - convergence_steps: 在指定步数内收敛
    - golden_deviation: 与 golden 结果偏差不超过指定百分比
    - ssim_threshold: SSIM 高于阈值
    - epe_threshold: EPE 低于阈值 (OPC)
    - loss_improvement: 损失改善比例不低于阈值 (SMO)

    Attributes:
        type: 断言类型
        threshold: 阈值
        golden_path: golden 结果路径 (golden_deviation 类型使用)
        tolerance: 容差 (golden_deviation 使用，偏差百分比)
        max_steps: 最大步数 (convergence_steps 使用)
        description: 断言描述
    """
    type: str = 'mse_threshold'
    threshold: Optional[float] = None
    golden_path: Optional[str] = None
    tolerance: Optional[float] = None
    max_steps: Optional[int] = None
    description: str = ''

    VALID_TYPES = {
        'mse_threshold', 'convergence_steps', 'golden_deviation',
        'ssim_threshold', 'epe_threshold', 'loss_improvement',
    }

    def __post_init__(self):
        if self.type not in self.VALID_TYPES:
            raise ValueError(f"无效断言类型: {self.type}, 有效值: {self.VALID_TYPES}")
        if self.type == 'mse_threshold' and self.threshold is None:
            raise ValueError("mse_threshold 断言必须指定 threshold")
        if self.type == 'convergence_steps' and self.max_steps is None:
            raise ValueError("convergence_steps 断言必须指定 max_steps")
        if self.type == 'golden_deviation' and self.golden_path is None:
            raise ValueError("golden_deviation 断言必须指定 golden_path")
        if self.type == 'ssim_threshold' and self.threshold is None:
            raise ValueError("ssim_threshold 断言必须指定 threshold")
        if self.type == 'epe_threshold' and self.threshold is None:
            raise ValueError("epe_threshold 断言必须指定 threshold")
        if self.type == 'loss_improvement' and self.threshold is None:
            raise ValueError("loss_improvement 断言必须指定 threshold")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AssertionConfig':
        return cls(
            type=d.get('type', 'mse_threshold'),
            threshold=d.get('threshold'),
            golden_path=d.get('golden_path'),
            tolerance=d.get('tolerance'),
            max_steps=d.get('max_steps'),
            description=d.get('description', ''),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {'type': self.type}
        if self.threshold is not None:
            result['threshold'] = self.threshold
        if self.golden_path is not None:
            result['golden_path'] = self.golden_path
        if self.tolerance is not None:
            result['tolerance'] = self.tolerance
        if self.max_steps is not None:
            result['max_steps'] = self.max_steps
        if self.description:
            result['description'] = self.description
        return result


@dataclass
class GoldenReference:
    """
    Golden 参考结果

    存储已验证的基准结果，用于回归测试中比较偏差。

    Attributes:
        experiment_name: 实验名称
        final_mse: 基准 MSE
        final_ssim: 基准 SSIM
        converged: 是否收敛
        convergence_step: 收敛步数
        final_loss: 最终损失
        total_iterations: 总迭代次数
        custom_metrics: 自定义指标
    """
    experiment_name: str = ''
    final_mse: Optional[float] = None
    final_ssim: Optional[float] = None
    converged: Optional[bool] = None
    convergence_step: Optional[int] = None
    final_loss: Optional[float] = None
    total_iterations: Optional[int] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GoldenReference':
        reserved = {
            'experiment_name', 'final_mse', 'final_ssim', 'converged',
            'convergence_step', 'final_loss', 'total_iterations',
        }
        custom = {k: v for k, v in d.items() if k not in reserved}
        return cls(
            experiment_name=d.get('experiment_name', ''),
            final_mse=d.get('final_mse'),
            final_ssim=d.get('final_ssim'),
            converged=d.get('converged'),
            convergence_step=d.get('convergence_step'),
            final_loss=d.get('final_loss'),
            total_iterations=d.get('total_iterations'),
            custom_metrics=custom,
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {'experiment_name': self.experiment_name}
        if self.final_mse is not None:
            result['final_mse'] = self.final_mse
        if self.final_ssim is not None:
            result['final_ssim'] = self.final_ssim
        if self.converged is not None:
            result['converged'] = self.converged
        if self.convergence_step is not None:
            result['convergence_step'] = self.convergence_step
        if self.final_loss is not None:
            result['final_loss'] = self.final_loss
        if self.total_iterations is not None:
            result['total_iterations'] = self.total_iterations
        result.update(self.custom_metrics)
        return result


@dataclass
class ExperimentSchema:
    """
    完整实验定义

    Attributes:
        name: 实验名称（唯一标识）
        description: 实验描述
        tags: 标签列表
        workflow: 工作流类型
        pattern: 输入图案配置
        optical: 光学系统配置
        optimizer: 优化器配置
        assertions: 回归断言列表
        workflow_extra: 工作流额外配置 (OPC/SMO 特有参数)
    """
    name: str = 'unnamed_experiment'
    description: str = ''
    tags: List[str] = field(default_factory=list)
    workflow: str = 'mask_optimization'
    pattern: PatternConfig = field(default_factory=PatternConfig)
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    assertions: List[AssertionConfig] = field(default_factory=list)
    workflow_extra: Dict[str, Any] = field(default_factory=dict)

    VALID_WORKFLOWS = {'mask_optimization', 'opc', 'smo'}

    def __post_init__(self):
        if self.workflow not in self.VALID_WORKFLOWS:
            raise ValueError(f"无效工作流类型: {self.workflow}, 有效值: {self.VALID_WORKFLOWS}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentSchema':
        pattern = PatternConfig.from_dict(d.get('pattern', {}))
        optical = OpticalConfig.from_dict(d.get('optical', {}))
        optimizer = OptimizerConfig.from_dict(d.get('optimizer', {}))

        assertions = [
            AssertionConfig.from_dict(a) for a in d.get('assertions', [])
        ]

        exp_section = d.get('experiment', d)
        workflow = exp_section.get('workflow', 'mask_optimization')

        reserved_experiment = {'name', 'description', 'tags', 'workflow'}
        reserved_top = {'experiment', 'pattern', 'optical', 'optimizer', 'assertions'}
        workflow_extra_keys = set(d.keys()) - reserved_top
        workflow_extra = {k: d[k] for k in workflow_extra_keys if k not in reserved_experiment}

        return cls(
            name=exp_section.get('name', 'unnamed_experiment'),
            description=exp_section.get('description', ''),
            tags=exp_section.get('tags', []),
            workflow=workflow,
            pattern=pattern,
            optical=optical,
            optimizer=optimizer,
            assertions=assertions,
            workflow_extra=workflow_extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment': {
                'name': self.name,
                'description': self.description,
                'tags': list(self.tags),
                'workflow': self.workflow,
            },
            'pattern': self.pattern.to_dict(),
            'optical': self.optical.to_dict(),
            'optimizer': self.optimizer.to_dict(),
            'assertions': [a.to_dict() for a in self.assertions],
            **self.workflow_extra,
        }


def load_experiment(yaml_path: Union[str, Path]) -> ExperimentSchema:
    """
    从 YAML 文件加载实验定义

    Args:
        yaml_path: YAML 文件路径

    Returns:
        ExperimentSchema 实例
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"实验定义文件不存在: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"实验定义文件为空: {yaml_path}")

    experiment = ExperimentSchema.from_dict(raw)
    logger.info(f"加载实验定义: {experiment.name} (workflow={experiment.workflow})")
    return experiment


def validate_experiment(experiment: ExperimentSchema) -> List[str]:
    """
    验证实验定义的完整性与合理性

    Args:
        experiment: 实验定义

    Returns:
        错误信息列表（空列表表示验证通过）
    """
    errors = []

    if not experiment.name or experiment.name == 'unnamed_experiment':
        errors.append("实验名称未指定或为默认值")

    if experiment.workflow == 'opc':
        opc = experiment.workflow_extra.get('opc', {})
        if not opc:
            errors.append("OPC 工作流需要 opc 配置段")

    if experiment.workflow == 'smo':
        smo = experiment.workflow_extra.get('smo', {})
        if not smo:
            errors.append("SMO 工作流需要 smo 配置段")

    if not experiment.assertions:
        errors.append("未定义任何回归断言，实验无法进行回归验证")

    pattern = experiment.pattern
    if pattern.pixel_size != experiment.optical.pixel_size:
        errors.append(
            f"图案 pixel_size ({pattern.pixel_size}) 与光学系统 pixel_size "
            f"({experiment.optical.pixel_size}) 不一致"
        )

    for i, assertion in enumerate(experiment.assertions):
        if assertion.type == 'golden_deviation' and not assertion.golden_path:
            errors.append(f"断言 {i}: golden_deviation 类型需要指定 golden_path")

    if errors:
        logger.warning(f"实验 '{experiment.name}' 验证发现 {len(errors)} 个问题")
    else:
        logger.info(f"实验 '{experiment.name}' 验证通过")

    return errors
