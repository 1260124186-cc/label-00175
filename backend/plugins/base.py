# -*- coding: utf-8 -*-
"""
插件基础抽象接口

定义四类核心插件的抽象基类和通用元数据结构。
第三方插件必须继承对应的基类并实现其抽象方法。
"""

import abc
import enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np


class PluginType(str, enum.Enum):
    """插件类型枚举

    定义框架支持的四种扩展点。
    """
    IMAGING_BACKEND = "imaging_backend"
    OPTIMIZER = "optimizer"
    LOSS_FUNCTION = "loss_function"
    WORKFLOW = "workflow"

    @classmethod
    def from_string(cls, value: str) -> "PluginType":
        """从字符串解析插件类型"""
        mapping = {
            "imaging": cls.IMAGING_BACKEND,
            "imaging_backend": cls.IMAGING_BACKEND,
            "optimizer": cls.OPTIMIZER,
            "optim": cls.OPTIMIZER,
            "loss": cls.LOSS_FUNCTION,
            "loss_function": cls.LOSS_FUNCTION,
            "workflow": cls.WORKFLOW,
            "pipeline": cls.WORKFLOW,
        }
        normalized = value.lower().strip()
        if normalized in mapping:
            return mapping[normalized]
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"未知插件类型: '{value}'。支持类型: {[t.value for t in cls]}"
        )


@dataclass
class PluginMetadata:
    """插件元数据

    描述插件的基本信息，用于注册、展示和版本管理。

    Attributes:
        name:           插件唯一名称（字母数字下划线）
        version:        语义化版本号，如 "1.0.0"
        plugin_type:    插件类型
        description:    插件功能简述
        author:         作者名称
        email:          联系方式（可选）
        homepage:       项目主页（可选）
        tags:           标签列表，用于分类搜索
        priority:       加载优先级，数值越大越优先，默认 0
        requires:       依赖的其他插件名称列表
        min_framework_version:  要求的最低框架版本（可选）
    """
    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    author: str = ""
    email: str = ""
    homepage: str = ""
    tags: List[str] = field(default_factory=list)
    priority: int = 0
    requires: List[str] = field(default_factory=list)
    min_framework_version: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["plugin_type"] = self.plugin_type.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        """从字典构造"""
        data = dict(data)
        if "plugin_type" in data:
            data["plugin_type"] = PluginType.from_string(data["plugin_type"])
        return cls(**data)


T = TypeVar("T", bound="BasePlugin")


class BasePlugin(abc.ABC):
    """插件基类

    所有插件类型的公共基类，提供统一的生命周期管理钩子。

    生命周期顺序:
        1. __init__         : 构造，保存配置
        2. initialize()     : 初始化，加载资源
        3. (业务调用)       : 执行具体算法
        4. shutdown()       : 释放资源
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 插件实例化时传入的配置字典
        """
        self._config: Dict[str, Any] = dict(config) if config else {}
        self._initialized: bool = False
        self._shutdown: bool = False

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """返回插件的元数据

        类方法，每个插件类必须提供其元信息。
        """
        ...

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """返回插件的默认配置

        插件类可以重写此方法提供默认配置值，框架会将其与
        用户传入的配置进行合并。
        """
        return {}

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """返回插件的配置 JSON Schema

        可选重写，用于配置校验和 UI 自动生成表单。
        """
        return {}

    # ------------------------------------------------------------------
    # 生命周期钩子
    # ------------------------------------------------------------------
    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown

    @property
    def config(self) -> Dict[str, Any]:
        """获取（只读的）配置快照"""
        return dict(self._config)

    def initialize(self, context: Optional[Dict[str, Any]] = None) -> None:
        """初始化插件

        在插件被创建后、首次使用前调用。
        子类可在此加载模型、分配资源。

        Args:
            context: 运行时上下文，可能包含全局配置、logger、工作目录等
        """
        if self._initialized:
            return
        self._context = dict(context) if context else {}
        self._on_initialize()
        self._initialized = True

    def _on_initialize(self) -> None:
        """子类可重写的初始化钩子"""
        pass

    def shutdown(self) -> None:
        """关闭插件，释放资源"""
        if self._shutdown:
            return
        self._on_shutdown()
        self._shutdown = True
        self._initialized = False

    def _on_shutdown(self) -> None:
        """子类可重写的关闭钩子"""
        pass

    def __enter__(self: T) -> T:
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def update_config(self, **kwargs: Any) -> None:
        """运行时动态更新部分配置"""
        self._config.update(kwargs)


# ============================================================================
# 成像后端插件
# ============================================================================

@dataclass
class ImagingInput:
    """成像输入

    Attributes:
        mask:           掩模图像（2D numpy 数组，0-1 归一化）
        wavelength:     波长（nm）
        numerical_aperture:  数值孔径
        pixel_size:     像素大小（nm）
        process_condition: 工艺条件字典（focus, dose 等）
    """
    mask: np.ndarray
    wavelength: float
    numerical_aperture: float
    pixel_size: float
    process_condition: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImagingOutput:
    """成像输出

    Attributes:
        wafer_image:    晶圆上光强分布（2D 数组）
        aerial_image:   空域像（可选）
        extra:          附加信息（pupil, TCC kernel 等）
    """
    wafer_image: np.ndarray
    aerial_image: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ImagingBackend(BasePlugin):
    """成像后端插件基类

    扩展点：替换/新增光学成像仿真算法，例如：
        - 标量 Hopkings 模型（内置）
        - 矢量成像模型
        - 严格电磁仿真（RCWA / FDTD 封装）
        - 神经网络代理模型
    """

    @abc.abstractmethod
    def simulate(self, inp: ImagingInput) -> ImagingOutput:
        """执行光学成像仿真

        Args:
            inp: 成像输入参数

        Returns:
            ImagingOutput 仿真结果
        """
        ...


# ============================================================================
# 优化器插件
# ============================================================================

@dataclass
class OptimizationOutput:
    """优化结果

    Attributes:
        x:              最优解（扁平化或原始形状均可）
        fun:            最优目标值
        nit:            迭代次数
        nfev:           函数调用次数
        success:        是否成功收敛
        message:        状态描述
        history:        目标值历史
    """
    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str
    history: List[float] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class Optimizer(BasePlugin):
    """优化器插件基类

    扩展点：新增掩模优化算法，例如：
        - 梯度下降 / BFGS / Adam（内置）
        - 遗传算法 / PSO（启发式）
        - 强化学习优化器
        - 基于模型的离线优化器
    """

    @abc.abstractmethod
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        bounds: Optional[Tuple[float, float]] = None,
        callbacks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
        **kwargs: Any,
    ) -> OptimizationOutput:
        """执行优化

        Args:
            objective:  目标函数 f(x) -> float
            x0:         初始解
            gradient:   梯度函数（可选，None 则内部自动求数值梯度）
            bounds:     变量上下界 (min, max)
            callbacks:  迭代回调列表，每次迭代调用 cb(state)
            **kwargs:   子类扩展参数

        Returns:
            OptimizationOutput
        """
        ...


# ============================================================================
# 损失函数插件
# ============================================================================

class LossFunction(BasePlugin):
    """损失函数插件基类

    扩展点：新增优化目标函数，例如：
        - MSE / MAE / SSIM（内置）
        - EPE (Edge Placement Error)
        - 多目标加权损失
        - 定制化工艺感知损失
    """

    @abc.abstractmethod
    def __call__(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """计算损失值

        Args:
            predicted:  模型输出图像（2D 或批量 3D）
            target:     目标图像
            **kwargs:   附加输入，例如权重、掩模、梯度标志

        Returns:
            标量损失值
        """
        ...

    def gradient(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """返回损失对 predicted 的梯度

        可选实现，默认使用数值差分。
        """
        eps = float(self._config.get("grad_eps", 1e-5))
        grad = np.zeros_like(predicted, dtype=np.float64)
        base = self.__call__(predicted, target, **kwargs)
        it = np.nditer(predicted, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = predicted[idx]
            perturbed = predicted.copy()
            perturbed[idx] = orig + eps
            grad[idx] = (self.__call__(perturbed, target, **kwargs) - base) / eps
            it.iternext()
        return grad


# ============================================================================
# 工作流插件
# ============================================================================

@dataclass
class WorkflowInput:
    """工作流输入

    Attributes:
        config:         工作流级配置（从 YAML/JSON 加载）
        data_dir:       数据目录
        output_dir:     输出目录
        overrides:      命令行覆盖参数
    """
    config: Dict[str, Any]
    data_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowOutput:
    """工作流输出

    Attributes:
        success:        工作流是否成功
        message:        状态描述
        output_files:   生成的输出文件相对路径列表
        metrics:        关键指标字典
        artifacts:      额外产出物
    """
    success: bool
    message: str = ""
    output_files: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


class Workflow(BasePlugin):
    """工作流插件基类

    扩展点：新增高级流程，例如：
        - OPC / SMO / ILT（内置）
        - Hybrid OPC+ILT（内置）
        - 定制化校准流程
        - 多芯片批处理流水线
    """

    @abc.abstractmethod
    def run(self, inp: WorkflowInput) -> WorkflowOutput:
        """执行工作流

        Args:
            inp: 工作流输入参数

        Returns:
            WorkflowOutput
        """
        ...
