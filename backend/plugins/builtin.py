# -*- coding: utf-8 -*-
"""
内置插件适配层

将项目现有的核心模块（core / algorithms / workflows 等）
适配为插件系统的标准插件，使得它们可通过 PluginManager 统一访问。
"""

import logging
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from plugins.base import (
    ImagingBackend,
    ImagingInput,
    ImagingOutput,
    LossFunction,
    OptimizationOutput,
    Optimizer,
    PluginMetadata,
    PluginType,
    Workflow,
    WorkflowInput,
    WorkflowOutput,
)
from plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# 成像后端插件：标量 Hopkins 模型（即核心模块 PartialCoherentImaging）
# ============================================================================

class BuiltinHopkinsImaging(ImagingBackend):
    """内置标量 Hopkins 成像模型"""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="hopkins_scalar",
            version="1.0.0",
            plugin_type=PluginType.IMAGING_BACKEND,
            description="内置标量 Hopkins 部分相干成像模型",
            author="Lithography Simulation Team",
            tags=["built-in", "scalar", "hopkins", "partial_coherent"],
            priority=100,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "tcc_mode": "kernel_2d",
            "socs_rank": 16,
            "use_resist_model": True,
            "resist_threshold": 0.5,
        }

    def _on_initialize(self) -> None:
        try:
            from core.imaging import (
                OpticalSystem,
                PartialCoherentImaging,
                apply_resist_model,
            )
            self._OpticalSystem = OpticalSystem
            self._PartialCoherentImaging = PartialCoherentImaging
            self._apply_resist = apply_resist_model
        except ImportError as e:
            raise RuntimeError("core.imaging 模块不可用: %s" % e)

    def simulate(self, inp: ImagingInput) -> ImagingOutput:
        cfg = self._config
        pc = dict(inp.process_condition)
        focus = pc.get("focus", 0.0)
        dose = pc.get("dose", 1.0)

        opt_sys = self._OpticalSystem(
            wavelength=inp.wavelength,
            numerical_aperture=inp.numerical_aperture,
        )
        imaging = self._PartialCoherentImaging(
            opt_sys,
            pixel_size=inp.pixel_size,
            tcc_mode=cfg.get("tcc_mode", "kernel_2d"),
        )
        aerial = imaging.simulate(inp.mask, focus=focus)
        wafer = aerial * dose
        if cfg.get("use_resist_model", True):
            wafer = self._apply_resist(
                wafer,
                threshold=cfg.get("resist_threshold", 0.5),
            )
        return ImagingOutput(
            wafer_image=wafer,
            aerial_image=aerial,
            extra={"dose": dose, "focus": focus},
        )


class BuiltinVectorImaging(ImagingBackend):
    """内置矢量成像模型"""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="vector_hopkins",
            version="1.0.0",
            plugin_type=PluginType.IMAGING_BACKEND,
            description="内置矢量 Hopkins 成像（含偏振）",
            author="Lithography Simulation Team",
            tags=["built-in", "vector", "polarization"],
            priority=99,
        )

    def simulate(self, inp: ImagingInput) -> ImagingOutput:
        try:
            from core.imaging import OpticalSystem, simulate_wafer_image
        except ImportError as e:
            raise RuntimeError("core.imaging 不可用: %s" % e)
        opt_sys = OpticalSystem(
            wavelength=inp.wavelength,
            numerical_aperture=inp.numerical_aperture,
        )
        wafer = simulate_wafer_image(
            mask=inp.mask,
            optical_system=opt_sys,
            pixel_size=inp.pixel_size,
            **inp.process_condition,
        )
        return ImagingOutput(wafer_image=wafer, aerial_image=wafer)


# ============================================================================
# 优化器插件：梯度下降 / BFGS / 粒子群
# ============================================================================

class _BuiltinOptimizerBase(Optimizer):
    def _make_numerical_gradient(
        self,
        objective: Callable[[np.ndarray], float],
        x0_shape: Tuple[int, ...],
    ) -> Callable[[np.ndarray], np.ndarray]:
        eps = float(self._config.get("grad_eps", 1e-6))

        def grad(x: np.ndarray) -> np.ndarray:
            x_ = x.reshape(x0_shape)
            g = np.zeros_like(x_)
            base = objective(x_)
            it = np.nditer(x_, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                orig = x_[idx]
                x_[idx] = orig + eps
                fp = objective(x_)
                g[idx] = (fp - base) / eps
                x_[idx] = orig
                it.iternext()
            return g.flatten()
        return grad


class BuiltinGradientDescentOptimizer(_BuiltinOptimizerBase):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="gradient_descent",
            version="1.0.0",
            plugin_type=PluginType.OPTIMIZER,
            description="内置梯度下降优化器（带动量、线搜索）",
            author="Lithography Simulation Team",
            tags=["built-in", "gradient", "first_order"],
            priority=100,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "max_iter": 100,
            "tol": 1e-6,
            "learning_rate": 0.01,
            "momentum": 0.0,
            "use_line_search": False,
            "grad_eps": 1e-6,
        }

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        bounds: Optional[Tuple[float, float]] = None,
        callbacks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
        **kwargs: Any,
    ) -> OptimizationOutput:
        try:
            from algorithms.optimizer import GradientDescentOptimizer as _Inner
        except ImportError as e:
            raise RuntimeError("algorithms.optimizer 不可用: %s" % e)

        cfg = self._config
        inner = _Inner(
            learning_rate=cfg.get("learning_rate", 0.01),
            momentum=cfg.get("momentum", 0.0),
            use_line_search=cfg.get("use_line_search", False),
            max_iter=cfg.get("max_iter", 100),
            tol=cfg.get("tol", 1e-6),
            verbose=cfg.get("verbose", False),
        )
        if gradient is None:
            gradient = self._make_numerical_gradient(objective, x0.shape)
        orig_shape = x0.shape

        def obj_flat(x_flat: np.ndarray) -> float:
            return float(objective(x_flat.reshape(orig_shape)))

        def grad_flat(x_flat: np.ndarray) -> np.ndarray:
            return np.asarray(gradient(x_flat.reshape(orig_shape)), dtype=np.float64).flatten()

        result = inner.optimize(obj_flat, x0.flatten(), grad_flat, bounds, **kwargs)

        history = getattr(result, "history", [])
        return OptimizationOutput(
            x=np.asarray(result.x).reshape(orig_shape),
            fun=float(result.fun),
            nit=int(result.nit),
            nfev=int(result.nfev),
            success=bool(result.success),
            message=result.message,
            history=[float(h) for h in history],
        )


class BuiltinBFGSOptimizer(_BuiltinOptimizerBase):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="bfgs",
            version="1.0.0",
            plugin_type=PluginType.OPTIMIZER,
            description="内置 BFGS 拟牛顿优化器",
            author="Lithography Simulation Team",
            tags=["built-in", "quasi_newton", "bfgs"],
            priority=100,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "max_iter": 200,
            "tol": 1e-7,
            "grad_eps": 1e-6,
        }

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        bounds: Optional[Tuple[float, float]] = None,
        callbacks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
        **kwargs: Any,
    ) -> OptimizationOutput:
        try:
            from algorithms.optimizer import BFGSOptimizer as _Inner
        except ImportError as e:
            raise RuntimeError("algorithms.optimizer 不可用: %s" % e)

        cfg = self._config
        inner = _Inner(
            max_iter=cfg.get("max_iter", 200),
            tol=cfg.get("tol", 1e-7),
            verbose=cfg.get("verbose", False),
        )
        if gradient is None:
            gradient = self._make_numerical_gradient(objective, x0.shape)
        orig_shape = x0.shape
        result = inner.optimize(
            lambda x: objective(x.reshape(orig_shape)),
            x0.flatten(),
            lambda x: np.asarray(gradient(x.reshape(orig_shape)), dtype=np.float64).flatten(),
            bounds,
            **kwargs,
        )
        return OptimizationOutput(
            x=np.asarray(result.x).reshape(orig_shape),
            fun=float(result.fun),
            nit=int(result.nit),
            nfev=int(result.nfev),
            success=bool(result.success),
            message=result.message,
            history=[float(h) for h in getattr(result, "history", [])],
        )


class BuiltinPSOOptimizer(Optimizer):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="pso",
            version="1.0.0",
            plugin_type=PluginType.OPTIMIZER,
            description="内置粒子群优化（启发式）",
            author="Lithography Simulation Team",
            tags=["built-in", "heuristic", "pso", "derivative_free"],
            priority=95,
        )

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        bounds: Optional[Tuple[float, float]] = None,
        callbacks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
        **kwargs: Any,
    ) -> OptimizationOutput:
        try:
            from algorithms.advanced_optimizer import ParticleSwarmOptimizer as _Inner
        except ImportError as e:
            raise RuntimeError("algorithms.advanced_optimizer 不可用: %s" % e)
        cfg = self._config
        inner = _Inner(
            population_size=cfg.get("population_size", 30),
            max_iter=cfg.get("max_iter", 50),
            w=cfg.get("inertia", 0.7),
            c1=cfg.get("c1", 1.5),
            c2=cfg.get("c2", 1.5),
            bounds=bounds or (0.0, 1.0),
            verbose=cfg.get("verbose", False),
        )
        orig_shape = x0.shape
        result = inner.optimize(
            lambda x: objective(x.reshape(orig_shape)),
            x0.flatten(),
            **kwargs,
        )
        return OptimizationOutput(
            x=np.asarray(result.x).reshape(orig_shape),
            fun=float(result.fun),
            nit=int(result.nit),
            nfev=int(result.nfev),
            success=bool(result.success),
            message=result.message,
            history=[float(h) for h in getattr(result, "history", [])],
        )


# ============================================================================
# 损失函数插件：MSE / MAE / SSIM / EPE
# ============================================================================

class _BuiltinLossBase(LossFunction):
    pass


class BuiltinMSELoss(_BuiltinLossBase):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="mse",
            version="1.0.0",
            plugin_type=PluginType.LOSS_FUNCTION,
            description="均方误差（MSE）损失",
            author="Lithography Simulation Team",
            tags=["built-in", "pixel", "mse"],
            priority=100,
        )

    def __call__(self, predicted: np.ndarray, target: np.ndarray, **kwargs: Any) -> float:
        from core.metrics import mse as _mse
        return float(_mse(predicted, target))

    def gradient(self, predicted: np.ndarray, target: np.ndarray, **kwargs: Any) -> np.ndarray:
        return 2.0 * (predicted.astype(np.float64) - target.astype(np.float64)) / predicted.size


class BuiltinMAELoss(_BuiltinLossBase):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="mae",
            version="1.0.0",
            plugin_type=PluginType.LOSS_FUNCTION,
            description="平均绝对误差（MAE）损失",
            author="Lithography Simulation Team",
            tags=["built-in", "pixel", "mae"],
            priority=99,
        )

    def __call__(self, predicted: np.ndarray, target: np.ndarray, **kwargs: Any) -> float:
        from core.metrics import mae as _mae
        return float(_mae(predicted, target))


class BuiltinSSIMLoss(_BuiltinLossBase):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="ssim",
            version="1.0.0",
            plugin_type=PluginType.LOSS_FUNCTION,
            description="1 - SSIM 结构相似性损失",
            author="Lithography Simulation Team",
            tags=["built-in", "perceptual", "ssim"],
            priority=98,
        )

    def __call__(self, predicted: np.ndarray, target: np.ndarray, **kwargs: Any) -> float:
        from core.metrics import ssim as _ssim
        return float(1.0 - _ssim(predicted, target))


class BuiltinWeightedCombinedLoss(_BuiltinLossBase):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="weighted_combined",
            version="1.0.0",
            plugin_type=PluginType.LOSS_FUNCTION,
            description="多损失加权组合 (a*MSE + b*MAE + c*TV)",
            author="Lithography Simulation Team",
            tags=["built-in", "combined"],
            priority=97,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "mse_weight": 1.0,
            "mae_weight": 0.0,
            "tv_weight": 0.01,
        }

    def __call__(self, predicted: np.ndarray, target: np.ndarray, **kwargs: Any) -> float:
        from core.metrics import mse as _mse, mae as _mae, total_variation as _tv
        cfg = self._config
        val = (
            cfg.get("mse_weight", 1.0) * _mse(predicted, target)
            + cfg.get("mae_weight", 0.0) * _mae(predicted, target)
            + cfg.get("tv_weight", 0.0) * _tv(predicted)
        )
        return float(val)


# ============================================================================
# 工作流插件：OPC / SMO / ILT / Hybrid
# ============================================================================

class _BuiltinWorkflowBase(Workflow):
    """工作流插件基类：通用的输入/输出适配逻辑"""

    def _prepare_dirs(self, inp: WorkflowInput) -> None:
        Path(inp.output_dir).mkdir(parents=True, exist_ok=True)

    def _merge_config(self, inp: WorkflowInput) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        cfg.update(self._config)
        cfg.update(dict(inp.config))
        cfg.update(dict(inp.overrides))
        return cfg


class BuiltinOPCWorkflow(_BuiltinWorkflowBase):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="opc",
            version="1.0.0",
            plugin_type=PluginType.WORKFLOW,
            description="内置 OPC（光学邻近校正）工作流",
            author="Lithography Simulation Team",
            tags=["built-in", "opc", "proximity_correction"],
            priority=100,
        )

    def run(self, inp: WorkflowInput) -> WorkflowOutput:
        try:
            from workflows.opc import run_opc_workflow
        except ImportError as e:
            raise RuntimeError("workflows.opc 不可用: %s" % e)
        self._prepare_dirs(inp)
        cfg = self._merge_config(inp)
        t0 = time.time()
        try:
            result = run_opc_workflow(
                config=cfg,
                data_dir=str(inp.data_dir),
                output_dir=str(inp.output_dir),
            )
            return WorkflowOutput(
                success=True,
                message="OPC 工作流完成",
                output_files=list(getattr(result, "output_files", [])),
                metrics=dict(getattr(result, "metrics", {})),
                artifacts={"raw_result": result},
            )
        except Exception as e:
            return WorkflowOutput(
                success=False,
                message=f"OPC 工作流失败: {e}",
                error=str(e),
            )


class BuiltinSMOWorkflow(_BuiltinWorkflowBase):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="smo",
            version="1.0.0",
            plugin_type=PluginType.WORKFLOW,
            description="内置 SMO（光源掩模协同优化）工作流",
            author="Lithography Simulation Team",
            tags=["built-in", "smo", "source_mask"],
            priority=99,
        )

    def run(self, inp: WorkflowInput) -> WorkflowOutput:
        try:
            from workflows.smo import run_smo_workflow
        except ImportError as e:
            raise RuntimeError("workflows.smo 不可用: %s" % e)
        self._prepare_dirs(inp)
        cfg = self._merge_config(inp)
        try:
            result = run_smo_workflow(
                config=cfg,
                data_dir=str(inp.data_dir),
                output_dir=str(inp.output_dir),
            )
            return WorkflowOutput(
                success=True,
                message="SMO 工作流完成",
                output_files=list(getattr(result, "output_files", [])),
                metrics=dict(getattr(result, "metrics", {})),
                artifacts={"raw_result": result},
            )
        except Exception as e:
            return WorkflowOutput(success=False, message=f"SMO 工作流失败: {e}")


class BuiltinILTWorkflow(_BuiltinWorkflowBase):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="ilt",
            version="1.0.0",
            plugin_type=PluginType.WORKFLOW,
            description="内置 ILT（反演光刻技术）工作流",
            author="Lithography Simulation Team",
            tags=["built-in", "ilt", "inverse_lithography"],
            priority=98,
        )

    def run(self, inp: WorkflowInput) -> WorkflowOutput:
        try:
            from workflows.ilt import run_ilt_workflow
        except ImportError as e:
            raise RuntimeError("workflows.ilt 不可用: %s" % e)
        self._prepare_dirs(inp)
        cfg = self._merge_config(inp)
        try:
            result = run_ilt_workflow(
                config=cfg,
                data_dir=str(inp.data_dir),
                output_dir=str(inp.output_dir),
            )
            return WorkflowOutput(
                success=True,
                message="ILT 工作流完成",
                output_files=list(getattr(result, "output_files", [])),
                metrics=dict(getattr(result, "metrics", {})),
                artifacts={"raw_result": result},
            )
        except Exception as e:
            return WorkflowOutput(success=False, message=f"ILT 工作流失败: {e}")


class BuiltinHybridOPCILTWorkflow(_BuiltinWorkflowBase):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="hybrid_opc_ilt",
            version="1.0.0",
            plugin_type=PluginType.WORKFLOW,
            description="内置 Hybrid OPC+ILT 混合工作流",
            author="Lithography Simulation Team",
            tags=["built-in", "hybrid", "opc", "ilt"],
            priority=97,
        )

    def run(self, inp: WorkflowInput) -> WorkflowOutput:
        try:
            from workflows.hybrid_opc_ilt import run_hybrid_opc_ilt_workflow
        except ImportError as e:
            raise RuntimeError("workflows.hybrid_opc_ilt 不可用: %s" % e)
        self._prepare_dirs(inp)
        cfg = self._merge_config(inp)
        try:
            result = run_hybrid_opc_ilt_workflow(
                config=cfg,
                data_dir=str(inp.data_dir),
                output_dir=str(inp.output_dir),
            )
            return WorkflowOutput(
                success=True,
                message="Hybrid OPC+ILT 工作流完成",
                output_files=list(getattr(result, "output_files", [])),
                metrics=dict(getattr(result, "metrics", {})),
                artifacts={"raw_result": result},
            )
        except Exception as e:
            return WorkflowOutput(success=False, message=f"Hybrid 工作流失败: {e}")


# ============================================================================
# 注册入口
# ============================================================================

_BUILTIN_PLUGINS: List[type] = [
    # Imaging
    BuiltinHopkinsImaging,
    BuiltinVectorImaging,
    # Optimizer
    BuiltinGradientDescentOptimizer,
    BuiltinBFGSOptimizer,
    BuiltinPSOOptimizer,
    # Loss
    BuiltinMSELoss,
    BuiltinMAELoss,
    BuiltinSSIMLoss,
    BuiltinWeightedCombinedLoss,
    # Workflow
    BuiltinOPCWorkflow,
    BuiltinSMOWorkflow,
    BuiltinILTWorkflow,
    BuiltinHybridOPCILTWorkflow,
]


def register_builtin_plugins(registry: PluginRegistry) -> int:
    """向注册表注册全部内置插件，返回成功数量"""
    count = 0
    for cls in _BUILTIN_PLUGINS:
        try:
            registry.register(plugin_class=cls, source="builtin")
            count += 1
        except Exception as e:
            logger.warning("注册内置插件 %s 失败: %s", cls.__name__, e)
    logger.info("已注册 %d 个内置插件", count)
    return count
