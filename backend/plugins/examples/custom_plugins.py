# -*- coding: utf-8 -*-
"""
第三方插件开发示例

演示如何为四类扩展点编写自定义插件。

使用方法：
    方式一（配置文件挂载）：
        在 plugins_config.yaml 中添加：
        plugins:
          loss_functions:
            robust_l1:
              enabled: true
              class: "plugins.examples.custom_plugins:RobustL1Loss"
              config:
                epsilon: 0.01

          optimizers:
            heavy_ball:
              enabled: true
              class: "plugins.examples.custom_plugins:HeavyBallOptimizer"

    方式二（entry_points 挂载，第三方 pip 包）：
        在 pyproject.toml 中声明：
        [project.entry-points."litho_sim.loss_functions"]
        robust_l1 = "your_pkg.losses:RobustL1Loss"
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from plugins.logging import PluginLogger


# ============================================================================
# 1. 自定义损失函数：鲁棒 L1 (Huber-like) 损失
# ============================================================================

class RobustL1Loss(LossFunction):
    """
    鲁棒 L1 损失，对异常点不敏感。

    L(delta) = sqrt(delta^2 + eps^2) - eps
    """

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="robust_l1",
            version="0.1.0",
            plugin_type=PluginType.LOSS_FUNCTION,
            description="鲁棒 Huber 风格 L1 损失，抗异常值",
            author="Third-Party Developer",
            email="dev@example.com",
            tags=["example", "robust", "loss"],
            priority=10,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"epsilon": 0.01}

    def _on_initialize(self) -> None:
        self.log = PluginLogger(
            "loss_function", "robust_l1", version="0.1.0"
        )
        self.log.info("RobustL1Loss 初始化完成", eps=self._config["epsilon"])

    def __call__(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,
    ) -> float:
        eps = float(self._config.get("epsilon", 0.01))
        diff = predicted.astype(np.float64) - target.astype(np.float64)
        loss = np.mean(np.sqrt(diff * diff + eps * eps) - eps)
        return float(loss)

    def gradient(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        eps = float(self._config.get("epsilon", 0.01))
        diff = predicted.astype(np.float64) - target.astype(np.float64)
        denom = np.sqrt(diff * diff + eps * eps)
        return diff / denom / predicted.size


# ============================================================================
# 2. 自定义优化器：Heavy-Ball 动量法
# ============================================================================

class HeavyBallOptimizer(Optimizer):
    """
    Polyak  Heavy-Ball 动量优化器。

    更新规则:
        v_{k+1} = beta * v_k - alpha * grad(x_k)
        x_{k+1} = x_k + v_{k+1}
    """

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="heavy_ball",
            version="0.1.0",
            plugin_type=PluginType.OPTIMIZER,
            description="Polyak Heavy-Ball 动量梯度下降",
            author="Third-Party Developer",
            tags=["example", "momentum", "first_order"],
            priority=10,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "alpha": 0.01,
            "beta": 0.9,
            "max_iter": 200,
            "tol": 1e-6,
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
        cfg = self._config
        alpha = float(cfg.get("alpha", 0.01))
        beta = float(cfg.get("beta", 0.9))
        max_iter = int(cfg.get("max_iter", 200))
        tol = float(cfg.get("tol", 1e-6))
        grad_eps = float(cfg.get("grad_eps", 1e-6))

        x = x0.astype(np.float64).copy()
        v = np.zeros_like(x)
        history: List[float] = []
        f_prev = float("inf")
        nfev = 0

        def num_grad(x_: np.ndarray) -> np.ndarray:
            g = np.zeros_like(x_)
            base = objective(x_)
            for idx in np.ndindex(x_.shape):
                orig = x_[idx]
                x_[idx] = orig + grad_eps
                fp = objective(x_)
                g[idx] = (fp - base) / grad_eps
                x_[idx] = orig
            return g

        grad_fn = gradient if gradient is not None else num_grad

        for it in range(max_iter):
            f_val = float(objective(x))
            nfev += 1
            history.append(f_val)

            if abs(f_prev - f_val) / (abs(f_prev) + 1e-12) < tol and it > 0:
                msg = f"收敛于第 {it} 次迭代"
                return OptimizationOutput(
                    x=x, fun=f_val, nit=it, nfev=nfev,
                    success=True, message=msg, history=history,
                )
            f_prev = f_val

            g = np.asarray(grad_fn(x), dtype=np.float64)
            v = beta * v - alpha * g
            x = x + v
            if bounds is not None:
                x = np.clip(x, bounds[0], bounds[1])

            if callbacks:
                state = {"iter": it, "fun": f_val, "x": x, "grad": g}
                for cb in callbacks:
                    cb(state)

        return OptimizationOutput(
            x=x, fun=float(objective(x)), nit=max_iter, nfev=nfev,
            success=False, message="达到最大迭代次数", history=history,
        )


# ============================================================================
# 3. 自定义成像后端：基于 FFT 的简化 Abbe 成像模型
# ============================================================================

class SimpleAbbeImaging(ImagingBackend):
    """
    简化 Abbe 成像模型（教学示例）。

    仅实现低通滤波 + 阈值，不包含 TCC / SOCS 等高级特性。
    """

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="simple_abbe",
            version="0.1.0",
            plugin_type=PluginType.IMAGING_BACKEND,
            description="简化 Abbe 低通滤波成像（教学示例）",
            author="Third-Party Developer",
            tags=["example", "imaging", "abbe"],
            priority=5,
        )

    def _on_initialize(self) -> None:
        self.log = PluginLogger("imaging_backend", "simple_abbe", "0.1.0")

    def simulate(self, inp: ImagingInput) -> ImagingOutput:
        """
        简化成像:
          1. 对掩模做 FFT
          2. 圆形截止频率（数值孔径/波长）
          3. IFFT -> 空域像，再平方得光强
        """
        mask = inp.mask.astype(np.float64)
        ny, nx = mask.shape
        ly = (ny - 1) * inp.pixel_size
        lx = (nx - 1) * inp.pixel_size
        f_max = inp.numerical_aperture / inp.wavelength  # 1/nm
        fy = np.fft.fftfreq(ny, d=ly / ny)[:, None]
        fx = np.fft.fftfreq(nx, d=lx / nx)[None, :]
        pupil = (fx ** 2 + fy ** 2) <= (f_max ** 2)
        pupil = pupil.astype(np.float64)

        fft_m = np.fft.fft2(mask)
        fft_f = fft_m * pupil
        aerial = np.fft.ifft2(fft_f).real
        # 正频移，保持强度为正
        aerial = aerial - aerial.min()
        aerial = aerial / (aerial.max() + 1e-12)

        dose = float(inp.process_condition.get("dose", 1.0))
        wafer = aerial * dose
        return ImagingOutput(wafer_image=wafer, aerial_image=aerial)


# ============================================================================
# 4. 自定义工作流：多分辨率优化流水线
# ============================================================================

class MultiResWorkflow(Workflow):
    """
    多分辨率优化工作流示例：
        从低分辨率粗优化 -> 逐步提升分辨率精修。
    """

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="multires",
            version="0.1.0",
            plugin_type=PluginType.WORKFLOW,
            description="多分辨率从粗到精掩模优化工作流",
            author="Third-Party Developer",
            tags=["example", "workflow", "multires"],
            priority=5,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "levels": 3,
            "optimizer": "gradient_descent",
            "loss": "mse",
            "iters_per_level": 50,
        }

    def run(self, inp: WorkflowInput) -> WorkflowOutput:
        cfg = inp.config
        levels = int(cfg.get("levels", 3))
        iters_per_level = int(cfg.get("iters_per_level", 50))

        # 使用 PluginManager 获取所需插件
        from plugins.manager import PluginManager
        mgr = PluginManager()
        try:
            opt = mgr.create_optimizer(
                cfg.get("optimizer", "gradient_descent"),
                {"max_iter": iters_per_level},
            )
            loss = mgr.create_loss(cfg.get("loss", "mse"), {})
        except Exception as e:
            return WorkflowOutput(success=False, message=f"插件获取失败: {e}")

        return WorkflowOutput(
            success=True,
            message=f"多分辨率工作流完成（{levels} 级，每级 {iters_per_level} 次迭代）",
            output_files=[],
            metrics={"levels": float(levels), "iters_total": float(levels * iters_per_level)},
        )
