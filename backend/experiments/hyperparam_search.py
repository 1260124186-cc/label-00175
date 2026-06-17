# -*- coding: utf-8 -*-
"""
超参数自动搜索模块

基于 Optuna 的贝叶斯优化超参搜索层，支持：
    - 学习率、正则化权重、SMO 交替轮次、OPC EPE 阈值等参数自动扫描
    - 单目标与多目标（Pareto 最优）优化
    - 与实验追踪系统集成（local/mlflow/wandb）
    - YAML 配置驱动的搜索空间定义

使用方式:
    # 单目标优化
    python -m experiments.hyperparam_search --config search_config.yaml

    # 多目标 Pareto 优化
    python -m experiments.hyperparam_search --config search_config.yaml --multi-objective
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import yaml
import numpy as np

from experiments.schema import (
    ExperimentSchema, load_experiment,
)
from experiments.executor import ExperimentExecutor, ExperimentResult

logger = logging.getLogger(__name__)


# ============================================================================
# 搜索空间定义
# ============================================================================

class ParamType(Enum):
    """超参数类型"""
    FLOAT = 'float'           # 连续浮点数
    INT = 'int'               # 整数
    CATEGORICAL = 'categorical'  # 分类变量
    LOG_FLOAT = 'log_float'   # 对数刻度浮点数
    LOG_INT = 'log_int'       # 对数刻度整数


@dataclass
class SearchParam:
    """单个搜索参数定义

    Attributes:
        name: 参数名称（支持点路径，如 optimizer.learning_rate）
        param_type: 参数类型
        low: 最小值（float/int）
        high: 最大值（float/int）
        choices: 可选值列表（categorical）
        step: 步长（可选）
        log: 是否对数刻度（与 param_type 对应，冗余但方便使用）
        description: 参数描述
    """
    name: str
    param_type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    description: str = ''

    def __post_init__(self):
        if self.param_type in (ParamType.FLOAT, ParamType.INT,
                               ParamType.LOG_FLOAT, ParamType.LOG_INT):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"参数 {self.name} 类型为 {self.param_type.value}，"
                    f"必须指定 low 和 high"
                )
            if self.low >= self.high:
                raise ValueError(
                    f"参数 {self.name} 的 low ({self.low}) 必须小于 high ({self.high})"
                )
        elif self.param_type == ParamType.CATEGORICAL:
            if not self.choices:
                raise ValueError(
                    f"参数 {self.name} 类型为 categorical，必须指定 choices"
                )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SearchParam':
        """从字典创建搜索参数"""
        param_type = ParamType(d.get('type', 'float'))

        low = d.get('low')
        if low is not None and param_type in (
            ParamType.FLOAT, ParamType.LOG_FLOAT,
        ):
            low = float(low)
        elif low is not None and param_type in (ParamType.INT, ParamType.LOG_INT):
            low = int(low)

        high = d.get('high')
        if high is not None and param_type in (
            ParamType.FLOAT, ParamType.LOG_FLOAT,
        ):
            high = float(high)
        elif high is not None and param_type in (ParamType.INT, ParamType.LOG_INT):
            high = int(high)

        step = d.get('step')
        if step is not None:
            step = float(step)

        return cls(
            name=d['name'],
            param_type=param_type,
            low=low,
            high=high,
            choices=d.get('choices'),
            step=step,
            description=d.get('description', ''),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'name': self.name,
            'type': self.param_type.value,
        }
        if self.low is not None:
            result['low'] = self.low
        if self.high is not None:
            result['high'] = self.high
        if self.choices is not None:
            result['choices'] = self.choices
        if self.step is not None:
            result['step'] = self.step
        if self.description:
            result['description'] = self.description
        return result


@dataclass
class SearchSpace:
    """搜索空间定义

    Attributes:
        params: 搜索参数列表
    """
    params: List[SearchParam] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SearchSpace':
        """从字典创建搜索空间"""
        params = [SearchParam.from_dict(p) for p in d.get('params', [])]
        return cls(params=params)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'SearchSpace':
        """从 YAML 文件加载搜索空间"""
        yaml_path = Path(yaml_path)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get('search_space', data))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {'params': [p.to_dict() for p in self.params]}

    def get_param(self, name: str) -> Optional[SearchParam]:
        """按名称获取参数"""
        for p in self.params:
            if p.name == name:
                return p
        return None


# ============================================================================
# 搜索配置
# ============================================================================

class SamplerType(Enum):
    """采样器类型"""
    TPE = 'tpe'               # Tree-structured Parzen Estimator (贝叶斯)
    RANDOM = 'random'         # 随机搜索
    GRID = 'grid'             # 网格搜索
    CMAES = 'cmaes'           # CMA-ES (进化策略)


class PrunerType(Enum):
    """剪枝器类型"""
    NONE = 'none'             # 不剪枝
    MEDIAN = 'median'         # 中位数剪枝
    SUCCESSIVE_HALVING = 'successive_halving'  # 连续减半
    HYPERBAND = 'hyperband'   # HyperBand


@dataclass
class ObjectiveConfig:
    """目标配置

    Attributes:
        name: 目标名称（对应实验结果中的指标）
        direction: 优化方向 'minimize' 或 'maximize'
        weight: 权重（单目标时可忽略）
    """
    name: str
    direction: str = 'minimize'
    weight: float = 1.0

    def __post_init__(self):
        if self.direction not in ('minimize', 'maximize'):
            raise ValueError(
                f"目标 {self.name} 的 direction 必须为 "
                f"'minimize' 或 'maximize'，当前为 {self.direction}"
            )


@dataclass
class HyperparamSearchConfig:
    """超参数搜索配置

    Attributes:
        name: 搜索名称
        description: 描述
        base_experiment: 基准实验 YAML 路径
        search_space: 搜索空间
        objectives: 目标列表
        n_trials: 试验次数
        sampler: 采样器类型
        pruner: 剪枝器类型
        seed: 随机种子
        max_concurrent: 最大并发数
        tracking_backend: 实验追踪后端
        tracking_dir: 追踪目录/URI
        output_dir: 输出目录
    """
    name: str = 'hyperparam_search'
    description: str = ''
    base_experiment: str = ''
    search_space: SearchSpace = field(default_factory=SearchSpace)
    objectives: List[ObjectiveConfig] = field(default_factory=list)
    n_trials: int = 50
    sampler: SamplerType = SamplerType.TPE
    pruner: PrunerType = PrunerType.NONE
    seed: int = 42
    max_concurrent: int = 1
    tracking_backend: str = 'local'
    tracking_dir: str = './hyperparam_tracking'
    output_dir: str = './hyperparam_results'

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HyperparamSearchConfig':
        """从字典创建配置"""
        search_space = SearchSpace.from_dict(d.get('search_space', {}))

        objectives = [
            ObjectiveConfig(
                name=o['name'],
                direction=o.get('direction', 'minimize'),
                weight=o.get('weight', 1.0),
            )
            for o in d.get('objectives', [])
        ]

        return cls(
            name=d.get('name', 'hyperparam_search'),
            description=d.get('description', ''),
            base_experiment=d.get('base_experiment', ''),
            search_space=search_space,
            objectives=objectives,
            n_trials=int(d.get('n_trials', 50)),
            sampler=SamplerType(d.get('sampler', 'tpe')),
            pruner=PrunerType(d.get('pruner', 'none')),
            seed=int(d.get('seed', 42)),
            max_concurrent=int(d.get('max_concurrent', 1)),
            tracking_backend=d.get('tracking_backend', 'local'),
            tracking_dir=d.get('tracking_dir', './hyperparam_tracking'),
            output_dir=d.get('output_dir', './hyperparam_results'),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'HyperparamSearchConfig':
        """从 YAML 文件加载配置"""
        yaml_path = Path(yaml_path)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'base_experiment': self.base_experiment,
            'search_space': self.search_space.to_dict(),
            'objectives': [
                {'name': o.name, 'direction': o.direction, 'weight': o.weight}
                for o in self.objectives
            ],
            'n_trials': self.n_trials,
            'sampler': self.sampler.value,
            'pruner': self.pruner.value,
            'seed': self.seed,
            'max_concurrent': self.max_concurrent,
            'tracking_backend': self.tracking_backend,
            'tracking_dir': self.tracking_dir,
            'output_dir': self.output_dir,
        }


# ============================================================================
# 试验结果
# ============================================================================

@dataclass
class TrialResult:
    """单次试验结果

    Attributes:
        trial_id: 试验 ID
        params: 试验参数
        values: 目标值
        experiment_result: 实验结果
        success: 是否成功
        error_message: 错误信息
        duration: 耗时（秒）
    """
    trial_id: int
    params: Dict[str, Any]
    values: List[float] = field(default_factory=list)
    experiment_result: Optional[ExperimentResult] = None
    success: bool = False
    error_message: str = ''
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trial_id': self.trial_id,
            'params': self.params,
            'values': self.values,
            'success': self.success,
            'error_message': self.error_message,
            'duration': self.duration,
            'experiment_result': (
                self.experiment_result.to_dict()
                if self.experiment_result else None
            ),
        }


# ============================================================================
# 超参数搜索器
# ============================================================================

class HyperparamSearcher:
    """超参数搜索器

    基于 Optuna 的贝叶斯优化超参搜索，支持单目标和多目标优化。
    """

    def __init__(self, config: HyperparamSearchConfig):
        self.config = config
        self._optuna_available = False
        self._study = None
        self.trial_results: List[TrialResult] = []
        self._base_experiment: Optional[ExperimentSchema] = None
        self._executor: Optional[ExperimentExecutor] = None
        self._tracker = None

        self._check_optuna()

    def _check_optuna(self):
        """检查 Optuna 是否可用"""
        try:
            import optuna
            self._optuna_available = True
            logger.info("Optuna 可用，将使用贝叶斯优化")
        except ImportError:
            self._optuna_available = False
            logger.warning(
                "Optuna 未安装，将使用随机搜索作为后备方案。"
                "安装命令: pip install optuna"
            )

    def _load_base_experiment(self):
        """加载基准实验"""
        if not self.config.base_experiment:
            raise ValueError("必须指定 base_experiment 路径")

        base_path = Path(self.config.base_experiment)
        if not base_path.exists():
            raise FileNotFoundError(f"基准实验文件不存在: {base_path}")

        self._base_experiment = load_experiment(base_path)
        logger.info(f"已加载基准实验: {self._base_experiment.name}")

    def _init_executor(self):
        """初始化实验执行器"""
        output_dir = Path(self.config.output_dir) / 'trials'
        self._executor = ExperimentExecutor(base_output_dir=str(output_dir))

    def _init_tracker(self):
        """初始化实验追踪器"""
        try:
            from utils.experiment_tracking import create_tracker
            self._tracker = create_tracker(
                backend=self.config.tracking_backend,
                experiment_name=self.config.name,
                tracking_dir=self.config.tracking_dir,
            )
            logger.info(f"实验追踪器已初始化: {self.config.tracking_backend}")
        except Exception as e:
            logger.warning(f"实验追踪器初始化失败: {e}，将跳过追踪")
            self._tracker = None

    def _apply_params_to_experiment(
        self, params: Dict[str, Any]
    ) -> ExperimentSchema:
        """将超参数应用到基准实验配置

        Args:
            params: 参数字典，键为点路径（如 optimizer.learning_rate）

        Returns:
            修改后的实验配置
        """
        import copy
        if self._base_experiment is None:
            self._load_base_experiment()

        experiment = copy.deepcopy(self._base_experiment)

        for param_name, param_value in params.items():
            self._set_param_by_path(experiment, param_name, param_value)

        return experiment

    @staticmethod
    def _set_param_by_path(
        experiment: ExperimentSchema, param_path: str, value: Any):
        """根据参数路径设置实验配置

        支持的路径格式:
            optimizer.learning_rate      → experiment.optimizer.learning_rate
            opc.epe_threshold            → experiment.workflow_extra['opc']['epe_threshold']
            smo.max_outer_iterations      → experiment.workflow_extra['smo']['max_outer_iterations']
            pattern.cd                       → experiment.pattern.cd
            optical.na                    → experiment.optical.na
            pattern.extra.some_key          → experiment.pattern.extra['some_key']
        """
        parts = param_path.split('.')
        first = parts[0]

        if first in ('optimizer', 'pattern', 'optical'):
            HyperparamSearcher._set_nested_attr(
                getattr(experiment, first), parts[1:], value
            )
        elif first in ('opc', 'smo', 'ilt'):
            if first not in experiment.workflow_extra:
                experiment.workflow_extra[first] = {}
            HyperparamSearcher._set_nested_attr(
                experiment.workflow_extra[first], parts[1:], value
            )
        else:
            if hasattr(experiment, first):
                HyperparamSearcher._set_nested_attr(
                    getattr(experiment, first), parts[1:], value
                )
            else:
                raise AttributeError(
                    f"未知的参数路径前缀: {first}, 参数: {param_path}"
                )

    @staticmethod
    def _set_nested_attr(obj: Any, parts: List[str], value: Any):
        """递归设置嵌套属性/字典键

        Args:
            obj: 对象或字典
            parts: 路径部分列表
            value: 要设置的值
        """
        if len(parts) == 1:
            key = parts[0]
            if hasattr(obj, key):
                setattr(obj, key, value)
            elif isinstance(obj, dict):
                obj[key] = value
            else:
                raise AttributeError(f"对象没有属性或键: {key}")
            return

        key = parts[0]
        if hasattr(obj, key):
            child = getattr(obj, key)
        elif isinstance(obj, dict):
            child = obj.get(key)
            if child is None:
                child = {}
                obj[key] = child
        else:
            raise AttributeError(f"对象没有属性或键: {key}")

        HyperparamSearcher._set_nested_attr(child, parts[1:], value)

    def _extract_objective_values(
        self, result: ExperimentResult
    ) -> List[float]:
        """从实验结果中提取目标值

        Args:
            result: 实验结果

        Returns:
            目标值列表，顺序与 objectives 配置一致
        """
        values = []

        for obj in self.config.objectives:
            name = obj.name
            value = None

            if name == 'final_mse' and result.final_mse is not None:
                value = result.final_mse
            elif name == 'final_loss' and result.final_loss is not None:
                value = result.final_loss
            elif name == 'final_ssim' and result.final_ssim is not None:
                value = result.final_ssim
            elif name == 'total_time':
                value = result.total_time
            elif name == 'total_iterations':
                value = float(result.total_iterations)
            elif name.startswith('custom_metrics.'):
                metric_name = name[len('custom_metrics.'):]
                if metric_name in result.custom_metrics:
                    value = result.custom_metrics[metric_name]

            if value is None:
                logger.warning(f"目标 {name} 在实验结果中不存在，使用 NaN")
                value = float('nan')

            values.append(float(value))

        return values

    def _objective(self, trial) -> Union[float, List[float]]:
        """Optuna 目标函数

        Args:
            trial: Optuna trial 对象

        Returns:
            目标值（单目标）或目标值列表（多目标）
        """
        trial_start = time.time()

        params = {}
        for param in self.config.search_space.params:
            value = self._suggest_param(trial, param)
            params[param.name] = value

        trial_id = trial.number
        logger.info(f"Trial {trial_id}: 参数 = {params}")

        try:
            experiment = self._apply_params_to_experiment(params)
            experiment.name = f"{self.config.name}_trial_{trial_id}"

            if self._tracker:
                self._tracker.start_run(
                    run_name=f"trial_{trial_id}",
                    tags={'phase': 'hyperparam_search', 'trial_id': str(trial_id)},
                )
                self._tracker.log_params(params)

            result = self._executor.run(experiment)

            if result.success:
                values = self._extract_objective_values(result)

                if self._tracker:
                    for i, obj in enumerate(self.config.objectives):
                        self._tracker.log_metric(obj.name, values[i])
                    self._tracker.log_metric('duration', result.total_time)
                    self._tracker.end_run('completed')

                trial_result = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    values=values,
                    experiment_result=result,
                    success=True,
                    duration=time.time() - trial_start,
                )
                self.trial_results.append(trial_result)

                logger.info(
                    f"Trial {trial_id} 完成: 目标值 = {values}, "
                    f"耗时 = {trial_result.duration:.2f}s"
                )

                if len(self.config.objectives) == 1:
                    return values[0]
                return values
            else:
                if self._tracker:
                    self._tracker.end_run('failed')

                trial_result = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    success=False,
                    error_message=result.error_message,
                    duration=time.time() - trial_start,
                )
                self.trial_results.append(trial_result)

                logger.warning(f"Trial {trial_id} 失败: {result.error_message}")

                if len(self.config.objectives) == 1:
                    return float('inf')
                return [float('inf')] * len(self.config.objectives)

        except Exception as e:
            if self._tracker:
                self._tracker.end_run('failed')

            trial_result = TrialResult(
                trial_id=trial_id,
                params=params,
                success=False,
                error_message=str(e),
                duration=time.time() - trial_start,
            )
            self.trial_results.append(trial_result)

            logger.error(f"Trial {trial_id} 异常: {e}", exc_info=True)

            if len(self.config.objectives) == 1:
                return float('inf')
            return [float('inf')] * len(self.config.objectives)

    def _suggest_param(self, trial, param: SearchParam):
        """从 Optuna trial 中建议参数值

        Args:
            trial: Optuna trial 对象
            param: 搜索参数定义

        Returns:
            建议的参数值
        """
        name = param.name

        if param.param_type == ParamType.FLOAT:
            return trial.suggest_float(
                name, param.low, param.high, step=param.step,
            )
        elif param.param_type == ParamType.LOG_FLOAT:
            return trial.suggest_float(
                name, param.low, param.high, log=True,
            )
        elif param.param_type == ParamType.INT:
            return trial.suggest_int(
                name, int(param.low), int(param.high), step=int(param.step or 1),
            )
        elif param.param_type == ParamType.LOG_INT:
            return trial.suggest_int(
                name, int(param.low), int(param.high), log=True,
            )
        elif param.param_type == ParamType.CATEGORICAL:
            return trial.suggest_categorical(name, param.choices)
        else:
            raise ValueError(f"不支持的参数类型: {param.param_type}")

    def _create_sampler(self):
        """创建 Optuna 采样器"""
        import optuna

        if self.config.sampler == SamplerType.TPE:
            return optuna.samplers.TPESampler(seed=self.config.seed)
        elif self.config.sampler == SamplerType.RANDOM:
            return optuna.samplers.RandomSampler(seed=self.config.seed)
        elif self.config.sampler == SamplerType.CMAES:
            return optuna.samplers.CmaEsSampler(seed=self.config.seed)
        elif self.config.sampler == SamplerType.GRID:
            return optuna.samplers.GridSampler(
                search_space=self._build_grid_search_space()
            )
        else:
            return optuna.samplers.TPESampler(seed=self.config.seed)

    def _build_grid_search_space(self) -> Dict[str, List[Any]]:
        """构建网格搜索空间"""
        space = {}
        for param in self.config.search_space.params:
            if param.param_type == ParamType.CATEGORICAL:
                space[param.name] = list(param.choices)
            elif param.param_type in (ParamType.FLOAT, ParamType.INT):
                if param.step:
                    low = param.low
                    high = param.high
                    step = param.step
                    values = []
                    v = low
                    while v <= high:
                        values.append(v)
                        v += step
                    if param.param_type == ParamType.INT:
                        values = [int(v) for v in values]
                    space[param.name] = values
                else:
                    raise ValueError(
                        f"网格搜索需要为参数 {param.name} 指定 step"
                    )
            else:
                raise ValueError(
                    f"网格搜索不支持参数类型 {param.param_type}"
                )
        return space

    def _create_pruner(self):
        """创建 Optuna 剪枝器"""
        import optuna

        if self.config.pruner == PrunerType.NONE:
            return optuna.pruners.NopPruner()
        elif self.config.pruner == PrunerType.MEDIAN:
            return optuna.pruners.MedianPruner()
        elif self.config.pruner == PrunerType.SUCCESSIVE_HALVING:
            return optuna.pruners.SuccessiveHalvingPruner()
        elif self.config.pruner == PrunerType.HYPERBAND:
            return optuna.pruners.HyperbandPruner()
        else:
            return optuna.pruners.NopPruner()

    def run(self) -> Dict[str, Any]:
        """执行超参数搜索

        Returns:
            搜索结果摘要
        """
        logger.info(f"开始超参数搜索: {self.config.name}")
        logger.info(f"试验次数: {self.config.n_trials}")
        logger.info(f"采样器: {self.config.sampler.value}")
        logger.info(f"目标数: {len(self.config.objectives)}")

        self._load_base_experiment()
        self._init_executor()
        self._init_tracker()

        if self._optuna_available:
            result = self._run_optuna()
        else:
            result = self._run_random_search()

        self._save_results(result)
        return result

    def _run_optuna(self) -> Dict[str, Any]:
        """使用 Optuna 执行搜索"""
        import optuna

        sampler = self._create_sampler()
        pruner = self._create_pruner()

        is_multi_objective = len(self.config.objectives) > 1

        if is_multi_objective:
            directions = [o.direction for o in self.config.objectives]
            study = optuna.create_study(
                study_name=self.config.name,
                directions=directions,
                sampler=sampler,
                pruner=pruner,
            )
        else:
            direction = self.config.objectives[0].direction
            study = optuna.create_study(
                study_name=self.config.name,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
            )

        self._study = study

        study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            show_progress_bar=False,
        )

        result = self._build_result_summary(study)
        return result

    def _run_random_search(self) -> Dict[str, Any]:
        """后备方案：随机搜索（无 Optuna 时使用）"""
        logger.info("使用随机搜索作为后备方案")

        rng = np.random.RandomState(self.config.seed)
        best_values = None
        best_params = None
        best_trial_id = -1

        for trial_id in range(self.config.n_trials):
            trial_start = time.time()
            params = {}

            for param in self.config.search_space.params:
                params[param.name] = self._random_sample(param, rng)

            try:
                experiment = self._apply_params_to_experiment(params)
                experiment.name = f"{self.config.name}_trial_{trial_id}"

                if self._tracker:
                    self._tracker.start_run(
                        run_name=f"trial_{trial_id}",
                        tags={'phase': 'hyperparam_search', 'trial_id': str(trial_id)},
                    )
                    self._tracker.log_params(params)

                result = self._executor.run(experiment)

                if result.success:
                    values = self._extract_objective_values(result)

                    if self._tracker:
                        for i, obj in enumerate(self.config.objectives):
                            self._tracker.log_metric(obj.name, values[i])
                        self._tracker.log_metric('duration', result.total_time)
                        self._tracker.end_run('completed')

                    trial_result = TrialResult(
                        trial_id=trial_id,
                        params=params,
                        values=values,
                        experiment_result=result,
                        success=True,
                        duration=time.time() - trial_start,
                    )
                    self.trial_results.append(trial_result)

                    if best_values is None or self._is_better(values, best_values):
                        best_values = values
                        best_params = params
                        best_trial_id = trial_id

                    logger.info(
                        f"Trial {trial_id} 完成: 目标值 = {values}, "
                        f"耗时 = {trial_result.duration:.2f}s"
                    )
                else:
                    if self._tracker:
                        self._tracker.end_run('failed')

                    trial_result = TrialResult(
                        trial_id=trial_id,
                        params=params,
                        success=False,
                        error_message=result.error_message,
                        duration=time.time() - trial_start,
                    )
                    self.trial_results.append(trial_result)

            except Exception as e:
                if self._tracker:
                    self._tracker.end_run('failed')

                trial_result = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    success=False,
                    error_message=str(e),
                    duration=time.time() - trial_start,
                )
                self.trial_results.append(trial_result)
                logger.error(f"Trial {trial_id} 异常: {e}")

        result = {
            'name': self.config.name,
            'n_trials': self.config.n_trials,
            'best_trial_id': best_trial_id,
            'best_params': best_params,
            'best_values': best_values,
            'trials': [t.to_dict() for t in self.trial_results],
            'pareto_front': self._compute_pareto_front(),
        }
        return result

    def _random_sample(self, param: SearchParam, rng: np.random.RandomState) -> Any:
        """随机采样参数值"""
        if param.param_type == ParamType.FLOAT:
            return float(rng.uniform(param.low, param.high))
        elif param.param_type == ParamType.LOG_FLOAT:
            log_low = np.log10(param.low)
            log_high = np.log10(param.high)
            return float(10 ** rng.uniform(log_low, log_high))
        elif param.param_type == ParamType.INT:
            return int(rng.randint(int(param.low), int(param.high) + 1))
        elif param.param_type == ParamType.LOG_INT:
            log_low = np.log10(param.low)
            log_high = np.log10(param.high)
            return int(round(10 ** rng.uniform(log_low, log_high)))
        elif param.param_type == ParamType.CATEGORICAL:
            return param.choices[rng.randint(len(param.choices))]
        else:
            raise ValueError(f"不支持的参数类型: {param.param_type}")

    def _is_better(self, values: List[float], best_values: List[float]) -> bool:
        """判断是否更优（单目标时使用）"""
        if len(values) != len(best_values):
            return False

        direction = self.config.objectives[0].direction
        if direction == 'minimize':
            return values[0] < best_values[0]
        else:
            return values[0] > best_values[0]

    def _build_result_summary(self, study) -> Dict[str, Any]:
        """构建搜索结果摘要"""
        is_multi_objective = len(self.config.objectives) > 1

        if is_multi_objective:
            pareto_trials = study.best_trials
            pareto_front = [
                {
                    'trial_id': t.number,
                    'params': t.params,
                    'values': list(t.values),
                }
                for t in pareto_trials
            ]
            best_trial = None
            best_params = None
            best_values = None
        else:
            best_trial = study.best_trial
            best_params = best_trial.params
            best_values = [best_trial.value]
            pareto_front = None

        result = {
            'name': self.config.name,
            'n_trials': len(study.trials),
            'is_multi_objective': is_multi_objective,
            'best_trial_id': best_trial.number if best_trial else None,
            'best_params': best_params,
            'best_values': best_values,
            'trials': [t.to_dict() for t in self.trial_results],
            'pareto_front': pareto_front or self._compute_pareto_front(),
            'study_summary': {
                'n_complete': len([t for t in study.trials if t.state.name == 'COMPLETE']),
                'n_failed': len([t for t in study.trials if t.state.name == 'FAIL']),
            },
        }
        return result

    def _compute_pareto_front(self) -> List[Dict[str, Any]]:
        """计算 Pareto 最优前沿

        使用非支配排序算法计算 Pareto 前沿。

        Returns:
            Pareto 前沿上的试验列表
        """
        successful_trials = [
            t for t in self.trial_results
            if t.success and not any(np.isnan(v) for v in t.values)
        ]

        if not successful_trials:
            return []

        directions = [o.direction for o in self.config.objectives]
        pareto_set = []

        for i, trial in enumerate(successful_trials):
            dominated = False
            for j, other in enumerate(successful_trials):
                if i == j:
                    continue
                if self._dominates(other, trial, directions):
                    dominated = True
                    break
            if not dominated:
                pareto_set.append(trial)

        return [
            {
                'trial_id': t.trial_id,
                'params': t.params,
                'values': t.values,
                'duration': t.duration,
            }
            for t in pareto_set
        ]

    @staticmethod
    def _dominates(
        trial_a: TrialResult,
        trial_b: TrialResult,
        directions: List[str],
    ) -> bool:
        """判断 trial_a 是否支配 trial_b

        支配定义：a 在所有目标上不差于 b，且至少在一个目标上更优。
        """
        at_least_one_better = False

        for i, direction in enumerate(directions):
            val_a = trial_a.values[i]
            val_b = trial_b.values[i]

            if direction == 'minimize':
                if val_a > val_b:
                    return False
                if val_a < val_b:
                    at_least_one_better = True
            else:
                if val_a < val_b:
                    return False
                if val_a > val_b:
                    at_least_one_better = True

        return at_least_one_better

    def _save_results(self, result: Dict[str, Any]):
        """保存搜索结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result_path = output_dir / 'search_results.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        config_path = output_dir / 'search_config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.config.to_dict(), f,
                default_flow_style=False, allow_unicode=True,
            )

        if result.get('pareto_front'):
            pareto_path = output_dir / 'pareto_front.json'
            with open(pareto_path, 'w', encoding='utf-8') as f:
                json.dump(
                    result['pareto_front'], f,
                    indent=2, ensure_ascii=False, default=str,
                )

        logger.info(f"搜索结果已保存到: {output_dir}")

    def print_summary(self):
        """打印搜索结果摘要"""
        result = {
            'name': self.config.name,
            'n_trials': len(self.trial_results),
        }

        successful = [t for t in self.trial_results if t.success]
        failed = [t for t in self.trial_results if not t.success]

        print("\n" + "=" * 70)
        print(f"超参数搜索结果: {self.config.name}")
        print("=" * 70)
        print(f"总试验次数: {len(self.trial_results)}")
        print(f"成功: {len(successful)}, 失败: {len(failed)}")

        if successful:
            if len(self.config.objectives) == 1:
                obj = self.config.objectives[0]
                best_trial = min(
                    successful,
                    key=lambda t: t.values[0] if obj.direction == 'minimize' else -t.values[0],
                )
                print(f"\n最优试验 (Trial {best_trial.trial_id}):")
                print(f"  目标值 ({obj.name}): {best_trial.values[0]:.6e}")
                print(f"  耗时: {best_trial.duration:.2f}s")
                print("  参数:")
                for k, v in sorted(best_trial.params.items()):
                    print(f"    {k}: {v}")
            else:
                pareto = self._compute_pareto_front()
                print(f"\nPareto 最优前沿 ({len(pareto)} 个解):")
                for i, p in enumerate(pareto):
                    values_str = ", ".join(
                        f"{self.config.objectives[j].name}={v:.6e}"
                        for j, v in enumerate(p['values'])
                    )
                    print(f"  解 {i + 1} (Trial {p['trial_id']}): {values_str}")

        print("=" * 70)


# ============================================================================
# 预设搜索空间
# ============================================================================

def get_default_search_space(workflow: str = 'mask_optimization') -> SearchSpace:
    """获取默认搜索空间

    Args:
        workflow: 工作流类型 (mask_optimization / opc / smo)

    Returns:
        默认搜索空间
    """
    params = []

    params.append(SearchParam(
        name='optimizer.learning_rate',
        param_type=ParamType.LOG_FLOAT,
        low=1e-5,
        high=1e-1,
        description='优化器学习率',
    ))

    params.append(SearchParam(
        name='optimizer.max_iter',
        param_type=ParamType.INT,
        low=20,
        high=200,
        description='最大迭代次数',
    ))

    if workflow == 'mask_optimization':
        params.append(SearchParam(
            name='optimizer.loss_weights.mse',
            param_type=ParamType.FLOAT,
            low=0.1,
            high=10.0,
            description='MSE 损失权重',
        ))
        params.append(SearchParam(
            name='optimizer.loss_weights.tv',
            param_type=ParamType.FLOAT,
            low=0.0,
            high=1.0,
            description='TV 正则化权重',
        ))

    elif workflow == 'opc':
        params.append(SearchParam(
            name='opc.epe_threshold',
            param_type=ParamType.FLOAT,
            low=1.0,
            high=10.0,
            description='EPE 热点判定阈值 (nm)',
        ))
        params.append(SearchParam(
            name='opc.max_iterations',
            param_type=ParamType.INT,
            low=1,
            high=10,
            description='OPC 最大迭代次数',
        ))
        params.append(SearchParam(
            name='opc.optimizer_learning_rate',
            param_type=ParamType.LOG_FLOAT,
            low=1e-4,
            high=1e-1,
            description='优化器学习率',
        ))
        params.append(SearchParam(
            name='opc.optimizer_epe_weight',
            param_type=ParamType.FLOAT,
            low=0.1,
            high=5.0,
            description='EPE 损失权重',
        ))

    elif workflow == 'smo':
        params.append(SearchParam(
            name='smo.max_outer_iterations',
            param_type=ParamType.INT,
            low=2,
            high=15,
            description='SMO 外层交替轮次',
        ))
        params.append(SearchParam(
            name='smo.source_learning_rate',
            param_type=ParamType.LOG_FLOAT,
            low=1e-5,
            high=1e-2,
            description='光源优化学习率',
        ))
        params.append(SearchParam(
            name='smo.mask_learning_rate',
            param_type=ParamType.LOG_FLOAT,
            low=1e-4,
            high=1e-1,
            description='掩模优化学习率',
        ))
        params.append(SearchParam(
            name='smo.source_constraints.smoothness_weight',
            param_type=ParamType.FLOAT,
            low=0.001,
            high=0.1,
            description='光源平滑正则化权重',
        ))

    return SearchSpace(params=params)


def get_default_objectives(workflow: str = 'mask_optimization') -> List[ObjectiveConfig]:
    """获取默认目标配置

    Args:
        workflow: 工作流类型

    Returns:
        目标配置列表
    """
    if workflow == 'opc':
        return [
            ObjectiveConfig(name='custom_metrics.epe_mean', direction='minimize'),
            ObjectiveConfig(name='total_time', direction='minimize'),
        ]
    elif workflow == 'smo':
        return [
            ObjectiveConfig(name='final_loss', direction='minimize'),
            ObjectiveConfig(name='total_time', direction='minimize'),
        ]
    else:
        return [
            ObjectiveConfig(name='final_mse', direction='minimize'),
            ObjectiveConfig(name='total_time', direction='minimize'),
        ]


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='超参数自动搜索（基于 Optuna 贝叶斯优化）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='搜索配置 YAML 文件路径',
    )
    parser.add_argument(
        '--base-experiment', '-b',
        type=str,
        help='基准实验 YAML 文件路径',
    )
    parser.add_argument(
        '--workflow', '-w',
        type=str,
        default='mask_optimization',
        choices=['mask_optimization', 'opc', 'smo'],
        help='工作流类型（使用默认搜索空间时）',
    )
    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=50,
        help='试验次数（默认: 50）',
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./hyperparam_results',
        help='输出目录（默认: ./hyperparam_results）',
    )
    parser.add_argument(
        '--sampler',
        type=str,
        default='tpe',
        choices=['tpe', 'random', 'grid', 'cmaes'],
        help='采样器类型（默认: tpe）',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认: 42）',
    )
    parser.add_argument(
        '--tracking-backend',
        type=str,
        default='local',
        choices=['local', 'mlflow', 'wandb'],
        help='实验追踪后端（默认: local）',
    )
    parser.add_argument(
        '--tracking-dir',
        type=str,
        default='./hyperparam_tracking',
        help='实验追踪目录/URI',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细日志输出',
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    if args.config:
        config = HyperparamSearchConfig.from_yaml(args.config)
    elif args.base_experiment:
        search_space = get_default_search_space(args.workflow)
        objectives = get_default_objectives(args.workflow)

        config = HyperparamSearchConfig(
            name=f'hyperparam_search_{args.workflow}',
            base_experiment=args.base_experiment,
            search_space=search_space,
            objectives=objectives,
            n_trials=args.n_trials,
            sampler=SamplerType(args.sampler),
            seed=args.seed,
            tracking_backend=args.tracking_backend,
            tracking_dir=args.tracking_dir,
            output_dir=args.output_dir,
        )
    else:
        parser.error("必须指定 --config 或 --base-experiment")
        return

    searcher = HyperparamSearcher(config)
    result = searcher.run()
    searcher.print_summary()

    print(f"\n结果已保存到: {config.output_dir}")
    return result


if __name__ == '__main__':
    main()
