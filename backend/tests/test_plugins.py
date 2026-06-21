# -*- coding: utf-8 -*-
"""
插件系统单元测试
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# 路径兼容：确保 backend/ 在 sys.path 中，和 __init__.py 逻辑一致
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from plugins.base import (
    ImagingInput,
    ImagingOutput,
    LossFunction,
    OptimizationOutput,
    PluginMetadata,
    PluginType,
    WorkflowInput,
    WorkflowOutput,
)
from plugins.builtin import register_builtin_plugins
from plugins.config import PluginConfig, merge_plugin_config
from plugins.manager import PluginManager
from plugins.registry import PluginRegistry
from plugins.serializer import (
    PluginResult,
    ResultSerializer,
    deserialize_result,
    serialize_result,
)
from plugins.logging import PluginLogger


# ============================================================================
# 1. 基础抽象接口 / 元数据测试
# ============================================================================

class TestBase:
    def test_plugin_type_from_string(self):
        assert PluginType.from_string("optimizer") is PluginType.OPTIMIZER
        assert PluginType.from_string("imaging") is PluginType.IMAGING_BACKEND
        assert PluginType.from_string("loss") is PluginType.LOSS_FUNCTION
        assert PluginType.from_string("workflow") is PluginType.WORKFLOW
        with pytest.raises(ValueError):
            PluginType.from_string("invalid_type")

    def test_plugin_metadata_roundtrip(self):
        md = PluginMetadata(
            name="test_plugin",
            version="0.0.1",
            plugin_type=PluginType.OPTIMIZER,
            description="测试插件",
            tags=["unit", "test"],
            priority=5,
        )
        d = md.to_dict()
        assert d["name"] == "test_plugin"
        assert d["plugin_type"] == "optimizer"
        md2 = PluginMetadata.from_dict(d)
        assert md2.name == md.name
        assert md2.plugin_type == PluginType.OPTIMIZER


# ============================================================================
# 2. 注册表测试
# ============================================================================

class _DummyOptimizer:
    """最小可注册的最小测试插件"""

    def __init__(self, config=None):
        self._config = config or {}

    @classmethod
    def get_metadata(cls):
        return PluginMetadata(
            name="dummy_opt",
            version="1.0.0",
            plugin_type=PluginType.OPTIMIZER,
            description="Dummy",
            priority=7,
        )

    def optimize(self, objective, x0, gradient=None, bounds=None, callbacks=None, **kw):
        return OptimizationOutput(
            x=x0.copy(), fun=0.0, nit=0, nfev=0,
            success=True, message="ok", history=[0.0]
        )


class TestRegistry:
    def test_register_and_query(self):
        reg = PluginRegistry()
        reg.register(plugin_class=_DummyOptimizer, source="test")
        assert reg.has(PluginType.OPTIMIZER, "dummy_opt")

        entry = reg.get(PluginType.OPTIMIZER, "dummy_opt")
        assert entry is not None
        assert entry.get_class() is _DummyOptimizer

    def test_priority_override(self):
        reg = PluginRegistry()

        class LowPri(_DummyOptimizer):
            @classmethod
            def get_metadata(cls):
                md = super().get_metadata()
                md.priority = 1
                return md

        class HiPri(_DummyOptimizer):
            @classmethod
            def get_metadata(cls):
                md = super().get_metadata()
                md.priority = 99
                return md

        reg.register(plugin_class=LowPri, source="low")
        # 更低优先级不应覆盖
        reg.register(plugin_class=HiPri, source="hi")
        entry = reg.get(PluginType.OPTIMIZER, "dummy_opt")
        assert entry.source == "hi"
        assert entry.metadata.priority == 99

    def test_invalid_metadata_rejected(self):
        reg = PluginRegistry()
        # 空名称
        with pytest.raises(ValueError):
            md = PluginMetadata(name="", version="1.0", plugin_type=PluginType.OPTIMIZER)
            reg.register(plugin_class=_DummyOptimizer, metadata=md)


# ============================================================================
# 3. 管理器 + 内置插件注册
# ============================================================================

class TestManager:
    def test_builtin_plugins_registered(self):
        mgr = PluginManager(register_builtin=True, auto_discover=False)
        summary = mgr.summary()
        # 至少应有若干内置插件存在
        assert len(summary["imaging_backend"]) >= 1
        assert len(summary["optimizer"]) >= 1
        assert len(summary["loss_function"]) >= 1
        assert len(summary["workflow"]) >= 1

    def test_create_builtin_loss(self):
        mgr = PluginManager(register_builtin=True, auto_discover=False)
        loss = mgr.create_loss("mse")
        assert loss is not None
        # 验证 MSE 计算正确
        pred = np.array([[0.0, 1.0], [1.0, 0.0]])
        target = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert abs(float(loss(pred, target))) < 1e-9
        noisy = pred + 0.1
        val = float(loss(noisy, target))
        assert val == pytest.approx(0.01, abs=1e-6)

    def test_create_optimizer_heavy_ball(self):
        """通过 load_from_dict 注册示例插件并运行"""
        mgr = PluginManager(register_builtin=False, auto_discover=False)
        # 直接手动注册示例
        from plugins.examples.custom_plugins import HeavyBallOptimizer
        mgr.register(HeavyBallOptimizer)

        opt = mgr.create_optimizer("heavy_ball", {"max_iter": 30})
        # 对简单二次函数 x^2 求最小值
        def obj(x):
            return float(np.sum(x ** 2))
        x0 = np.array([3.0, -2.0])
        out = opt.optimize(obj, x0, gradient=lambda x: 2 * x)
        assert out.success or out.nit > 0
        # 应明显下降（从初始范数 ~3.6 下降到 <1.5 视为有效）
        assert np.linalg.norm(out.x) < 1.5

    def test_singleton_reuse(self):
        mgr = PluginManager(register_builtin=True, auto_discover=False)
        a = mgr.create_loss("mse")
        b = mgr.create_loss("mse")
        assert a is b
        c = mgr.create_loss("mse", singleton=False)
        assert a is not c

    def test_config_merge(self):
        """三层配置合并：默认 < 注册 < 用户"""
        from plugins.examples.custom_plugins import HeavyBallOptimizer
        # 默认: alpha=0.01
        # 注册覆盖: alpha=0.005
        # 用户覆盖: alpha=0.001, max_iter=5
        mgr = PluginManager(register_builtin=False, auto_discover=False)
        mgr.register(HeavyBallOptimizer, config={"alpha": 0.005})
        inst = mgr.create_optimizer("heavy_ball", {"alpha": 0.001, "max_iter": 5})
        assert inst.config["alpha"] == 0.001
        assert inst.config["max_iter"] == 5
        # beta 未被覆盖，保留默认值 0.9
        assert inst.config["beta"] == 0.9

    def test_lifecycle(self):
        mgr = PluginManager(register_builtin=False, auto_discover=False)
        from plugins.examples.custom_plugins import RobustL1Loss
        mgr.register(RobustL1Loss)
        loss = mgr.create_loss("robust_l1")
        assert loss.initialized
        mgr.shutdown_all()
        assert loss.is_shutdown


# ============================================================================
# 4. 配置管理测试
# ============================================================================

class TestConfig:
    def test_deep_merge(self):
        from plugins.examples.custom_plugins import HeavyBallOptimizer
        pc = PluginConfig(
            HeavyBallOptimizer,
            register_config={"alpha": 0.005, "nested": {"a": 1}},
            user_config={"max_iter": 10, "nested": {"b": 2}},
        )
        merged = pc.merged
        assert merged["alpha"] == 0.005
        assert merged["max_iter"] == 10
        assert merged["nested"]["a"] == 1
        assert merged["nested"]["b"] == 2
        # 默认值保留
        assert merged["beta"] == 0.9

    def test_dot_get(self):
        from plugins.examples.custom_plugins import RobustL1Loss
        pc = PluginConfig(RobustL1Loss, user_config={"epsilon": 0.05})
        assert pc.get("epsilon") == 0.05
        assert pc.get("nope.missing.key", "def") == "def"
        pc.require("epsilon")  # 不应抛
        with pytest.raises(KeyError):
            pc.require("does_not_exist")


# ============================================================================
# 5. 日志测试
# ============================================================================

class TestLogging:
    def test_event_tracking(self):
        log = PluginLogger("optimizer", "test", "1.0.0")
        log.bind(iteration=5)
        log.event("step", loss=0.12, grad_norm=0.03)
        events = log.events()
        assert len(events) == 1
        assert events[0]["event"] == "step"
        assert events[0]["payload"]["loss"] == 0.12

    def test_timed(self):
        log = PluginLogger("loss", "test")
        with log.timed("op"):
            _ = np.ones((10, 10)).sum()
        events = log.events()
        assert len(events) == 1
        assert events[0]["payload"]["success"] is True
        assert events[0]["payload"]["duration_ms"] >= 0


# ============================================================================
# 6. 序列化测试
# ============================================================================

class TestSerializer:
    def test_json_roundtrip(self):
        mask = np.random.rand(8, 8).astype(np.float32)
        res = PluginResult(
            plugin_type="optimizer",
            plugin_name="test",
            plugin_version="1.0",
            success=True,
            started_at="2025-01-01T00:00:00",
            finished_at="2025-01-01T00:00:05",
            duration_ms=5000.0,
            inputs={"mask": mask},
            outputs={"optimized_mask": mask * 0.9, "history": [0.5, 0.2, 0.1]},
            metrics={"final_loss": 0.1},
        )
        s = serialize_result(res, format="json")
        restored = deserialize_result(s, format="json")
        assert restored.success is True
        assert restored.metrics["final_loss"] == 0.1
        assert np.allclose(restored.outputs["optimized_mask"], mask * 0.9)

    def test_npz_roundtrip(self, tmp_path):
        arr = np.random.rand(32, 32)
        res = PluginResult(
            plugin_type="imaging_backend",
            plugin_name="test",
            outputs={"wafer": arr},
            metrics={"mean": float(arr.mean())},
        )
        p = tmp_path / "out.npz"
        # 使用 serializer 实例直接 dump_npz
        ResultSerializer().dump_npz(res, p)
        loaded = ResultSerializer().load_npz(p)
        assert np.allclose(loaded.outputs["wafer"], arr)

    def test_yaml_roundtrip(self, tmp_path):
        res = PluginResult(
            plugin_type="workflow",
            plugin_name="test_wf",
            outputs={"msg": "你好，中文"},
            metrics={"x": 1.5},
        )
        s = serialize_result(res, format="yaml")
        restored = deserialize_result(s, format="yaml")
        assert restored.outputs["msg"] == "你好，中文"


# ============================================================================
# 7. 配置文件加载器测试
# ============================================================================

class TestConfigLoader:
    def test_load_from_dict(self, tmp_path):
        data = {
            "plugins": {
                "optimizers": {
                    "heavy_ball": {
                        "enabled": True,
                        "class": "plugins.examples.custom_plugins:HeavyBallOptimizer",
                        "config": {"alpha": 0.007}},
                },
                "loss_functions": {
                    "robust_l1": {
                        "enabled": True,
                        "class": "plugins.examples.custom_plugins.RobustL1Loss",
                    }
                }
            }
        }
        mgr = PluginManager(register_builtin=False, auto_discover=False)
        n = mgr.load_from_dict(data, source="inline_test")
        assert n == 2
        assert mgr.has("optimizer", "heavy_ball")
        assert mgr.has("loss_function", "robust_l1")

    def test_load_from_file(self, tmp_path):
        # 写入模板文件
        p = tmp_path / "plugins.yaml"
        content = """
plugins:
  imaging_backends:
    simple_abbe:
      enabled: true
      class: "plugins.examples.custom_plugins:SimpleAbbeImaging"
  workflows:
    multires:
      enabled: true
      class: plugins.examples.custom_plugins.MultiResWorkflow
      config:
        levels: 4
"""
        p.write_text(content, encoding="utf-8")
        mgr = PluginManager(register_builtin=False, auto_discover=False)
        n = mgr.load_from_file(str(p))
        assert n == 2
        assert mgr.has("imaging_backend", "simple_abbe")
        assert mgr.has("workflow", "multires")


# ============================================================================
# 8. 端到端：成像插件 + 优化器插件 + 损失插件
# ============================================================================

class TestEndToEnd:
    def test_imaging_plus_loss(self):
        """完整链路：简单掩模 -> 成像仿真 -> 损失计算"""
        mgr = PluginManager(register_builtin=False, auto_discover=False)
        from plugins.examples.custom_plugins import SimpleAbbeImaging, RobustL1Loss
        mgr.register(SimpleAbbeImaging)
        mgr.register(RobustL1Loss)

        imaging = mgr.create_imaging("simple_abbe")
        target = np.zeros((64, 64), dtype=np.float64)
        target[16:48, 16:48] = 1.0
        mask = target.copy()
        out = imaging.simulate(ImagingInput(
            mask=mask,
            wavelength=193.0,
            numerical_aperture=0.85,
            pixel_size=1.0,
            process_condition={"dose": 1.0},
        ))
        assert isinstance(out, ImagingOutput)
        assert out.wafer_image.shape == mask.shape
        # 损失应是一个标量
        loss = mgr.create_loss("robust_l1")
        l_val = float(loss(out.wafer_image, target))
        assert isinstance(l_val, float)
        assert l_val >= 0.0
