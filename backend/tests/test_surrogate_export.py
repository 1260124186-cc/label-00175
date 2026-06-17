# -*- coding: utf-8 -*-
"""
代理模型生产化导出功能单元测试
"""

import pytest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from surrogate.model import SurrogateModelConfig, build_model
from surrogate.train import (
    TrainingConfig,
    ExportConfig,
    ExportPaths,
    export_to_onnx,
    export_to_torchscript,
    export_trained_model,
    export_metadata,
)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch 未安装")
class TestModelExport:
    """模型导出功能测试"""

    def _create_test_model(self):
        """创建测试用模型"""
        cfg = SurrogateModelConfig(
            model_type='unet',
            in_channels=1,
            out_channels=1,
            base_channels=16,
            num_levels=3,
            use_batch_norm=True,
            final_activation='sigmoid',
        )
        model = build_model(cfg)
        model.eval()
        return model, cfg

    def test_export_config_defaults(self):
        """测试导出配置默认值"""
        cfg = ExportConfig()
        assert cfg.export_onnx == True
        assert cfg.export_torchscript == True
        assert cfg.onnx_opset_version == 17
        assert cfg.dynamic_batch == True
        assert cfg.optimize == True
        assert cfg.validate_export == True
        assert cfg.simplify_onnx == True

    def test_export_paths_defaults(self):
        """测试导出路径默认值"""
        paths = ExportPaths()
        assert paths.onnx_path is None
        assert paths.torchscript_path is None
        assert paths.metadata_path is None

    def test_training_config_with_export(self):
        """测试训练配置包含导出配置"""
        train_cfg = TrainingConfig()
        assert hasattr(train_cfg, 'export')
        assert isinstance(train_cfg.export, ExportConfig)
        assert train_cfg.export.export_onnx == True

    def test_training_config_to_dict_with_export(self):
        """测试训练配置序列化包含导出配置"""
        train_cfg = TrainingConfig()
        cfg_dict = train_cfg.to_dict()
        assert 'export' in cfg_dict
        assert isinstance(cfg_dict['export'], dict)
        assert cfg_dict['export']['export_onnx'] == True

    def test_training_config_from_dict_with_export(self):
        """测试训练配置反序列化包含导出配置"""
        original = TrainingConfig()
        original.export.export_onnx = False
        original.export.onnx_opset_version = 16

        cfg_dict = original.to_dict()
        restored = TrainingConfig.from_dict(cfg_dict)

        assert restored.export.export_onnx == False
        assert restored.export.onnx_opset_version == 16
        assert restored.export.export_torchscript == True

    def test_export_metadata(self, tmp_path):
        """测试元数据导出"""
        model, model_cfg = self._create_test_model()
        train_cfg = TrainingConfig(grid_size=(64, 64))
        export_paths = ExportPaths(
            onnx_path="/tmp/model.onnx",
            torchscript_path="/tmp/model.pt",
        )

        metadata_path = export_metadata(
            output_path=str(tmp_path / "metadata.json"),
            model_config=model_cfg,
            training_config=train_cfg,
            grid_size=(64, 64),
            export_paths=export_paths,
            extra_info={"mse": 0.001, "ssim": 0.99},
        )

        assert os.path.exists(metadata_path)

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        assert 'model' in metadata
        assert metadata['model']['type'] == 'unet'
        assert metadata['model']['input_shape'] == [-1, 1, 64, 64]
        assert metadata['model']['output_shape'] == [-1, 1, 64, 64]
        assert metadata['model']['input_range'] == [0.0, 1.0]
        assert metadata['model']['output_range'] == [0.0, 1.0]

        assert 'export' in metadata
        assert metadata['export']['onnx_path'] == "/tmp/model.onnx"
        assert metadata['export']['torchscript_path'] == "/tmp/model.pt"
        assert metadata['export']['dynamic_batch'] == True

        assert 'preprocessing' in metadata
        assert 'postprocessing' in metadata
        assert 'performance_hints' in metadata

        assert 'extra' in metadata
        assert metadata['extra']['mse'] == 0.001
        assert metadata['extra']['ssim'] == 0.99

    def test_export_to_onnx_basic(self, tmp_path):
        """测试基础 ONNX 导出（不验证，避免依赖问题）"""
        model, model_cfg = self._create_test_model()
        output_path = str(tmp_path / "model.onnx")

        # 不验证，因为可能没有安装 onnxruntime
        result = export_to_onnx(
            model=model,
            output_path=output_path,
            input_shape=(1, 1, 64, 64),
            opset_version=17,
            dynamic_batch=True,
            optimize=True,
            validate=False,  # 跳过验证
            simplify=False,  # 跳过简化
            device=torch.device('cpu'),
        )

        assert result is not None
        assert os.path.exists(output_path)
        assert result == output_path

    def test_export_to_torchscript_basic(self, tmp_path):
        """测试基础 TorchScript 导出（不验证）"""
        model, model_cfg = self._create_test_model()
        output_path = str(tmp_path / "model.pt")

        result = export_to_torchscript(
            model=model,
            output_path=output_path,
            input_shape=(1, 1, 64, 64),
            method='trace',
            optimize=True,
            validate=False,  # 跳过验证
            device=torch.device('cpu'),
        )

        assert result is not None
        assert os.path.exists(output_path)
        assert result == output_path

    def test_export_trained_model_full(self, tmp_path):
        """测试完整导出流程"""
        model, model_cfg = self._create_test_model()
        train_cfg = TrainingConfig(
            grid_size=(64, 64),
            export=ExportConfig(
                export_onnx=True,
                export_torchscript=True,
                validate_export=False,
                simplify_onnx=False,
            ),
        )

        export_paths = export_trained_model(
            model=model,
            output_dir=str(tmp_path),
            model_config=model_cfg,
            training_config=train_cfg,
            grid_size=(64, 64),
            device=torch.device('cpu'),
            extra_metrics={"mse": 0.001, "ssim": 0.99},
        )

        assert export_paths.onnx_path is not None
        assert export_paths.torchscript_path is not None
        assert export_paths.metadata_path is not None

        assert os.path.exists(export_paths.onnx_path)
        assert os.path.exists(export_paths.torchscript_path)
        assert os.path.exists(export_paths.metadata_path)

    def test_export_trained_model_onnx_only(self, tmp_path):
        """测试仅导出 ONNX"""
        model, model_cfg = self._create_test_model()
        train_cfg = TrainingConfig(
            grid_size=(64, 64),
            export=ExportConfig(
                export_onnx=True,
                export_torchscript=False,
                validate_export=False,
                simplify_onnx=False,
            ),
        )

        export_paths = export_trained_model(
            model=model,
            output_dir=str(tmp_path),
            model_config=model_cfg,
            training_config=train_cfg,
            grid_size=(64, 64),
            device=torch.device('cpu'),
        )

        assert export_paths.onnx_path is not None
        assert export_paths.torchscript_path is None
        assert export_paths.metadata_path is not None

        assert os.path.exists(export_paths.onnx_path)
        assert os.path.exists(export_paths.metadata_path)

    def test_export_trained_model_torchscript_only(self, tmp_path):
        """测试仅导出 TorchScript"""
        model, model_cfg = self._create_test_model()
        train_cfg = TrainingConfig(
            grid_size=(64, 64),
            export=ExportConfig(
                export_onnx=False,
                export_torchscript=True,
                validate_export=False,
                simplify_onnx=False,
            ),
        )

        export_paths = export_trained_model(
            model=model,
            output_dir=str(tmp_path),
            model_config=model_cfg,
            training_config=train_cfg,
            grid_size=(64, 64),
            device=torch.device('cpu'),
        )

        assert export_paths.onnx_path is None
        assert export_paths.torchscript_path is not None
        assert export_paths.metadata_path is not None

        assert os.path.exists(export_paths.torchscript_path)
        assert os.path.exists(export_paths.metadata_path)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch 未安装")
class TestInferenceEngine:
    """推理引擎测试"""

    def _create_exported_model(self, tmp_path):
        """创建导出的模型用于测试"""
        model_cfg = SurrogateModelConfig(
            model_type='unet',
            in_channels=1,
            out_channels=1,
            base_channels=16,
            num_levels=3,
            use_batch_norm=True,
            final_activation='sigmoid',
        )
        model = build_model(model_cfg)
        model.eval()

        train_cfg = TrainingConfig(
            grid_size=(64, 64),
            export=ExportConfig(
                export_onnx=True,
                export_torchscript=True,
                validate_export=False,
                simplify_onnx=False,
            ),
        )

        export_trained_model(
            model=model,
            output_dir=str(tmp_path),
            model_config=model_cfg,
            training_config=train_cfg,
            grid_size=(64, 64),
            device=torch.device('cpu'),
        )

        return str(tmp_path)

    def test_inference_engine_init(self, tmp_path):
        """测试推理引擎初始化"""
        model_dir = self._create_exported_model(tmp_path)

        from surrogate.inference_server import SurrogateInferenceEngine

        engine = SurrogateInferenceEngine(
            model_dir=model_dir,
            prefer_onnx=True,
            device='cpu',
        )

        assert engine.backend is not None
        assert engine.metadata is not None
        assert engine.input_shape == (-1, 1, 64, 64)
        assert engine.output_shape == (-1, 1, 64, 64)

    def test_inference_engine_predict_single(self, tmp_path):
        """测试单张掩模推理"""
        model_dir = self._create_exported_model(tmp_path)

        from surrogate.inference_server import SurrogateInferenceEngine

        engine = SurrogateInferenceEngine(model_dir, device='cpu')

        mask = np.random.rand(64, 64).astype(np.float32)
        aerial = engine.predict(mask)

        assert aerial.shape == (64, 64)
        assert aerial.dtype == np.float32
        assert np.all(aerial >= 0.0) and np.all(aerial <= 1.0)

    def test_inference_engine_predict_batch(self, tmp_path):
        """测试批量掩模推理"""
        model_dir = self._create_exported_model(tmp_path)

        from surrogate.inference_server import SurrogateInferenceEngine

        engine = SurrogateInferenceEngine(model_dir, device='cpu')

        masks = np.random.rand(4, 64, 64).astype(np.float32)
        aerials = engine.predict(masks)

        assert aerials.shape == (4, 64, 64)
        assert aerials.dtype == np.float32
        assert np.all(aerials >= 0.0) and np.all(aerials <= 1.0)

    def test_inference_engine_stats(self, tmp_path):
        """测试推理统计"""
        model_dir = self._create_exported_model(tmp_path)

        from surrogate.inference_server import SurrogateInferenceEngine

        engine = SurrogateInferenceEngine(model_dir, device='cpu')

        mask = np.random.rand(64, 64).astype(np.float32)
        for _ in range(5):
            engine.predict(mask)

        stats = engine.get_stats()
        assert stats['total_requests'] == 5
        assert stats['total_masks'] == 5
        assert stats['avg_latency_ms_per_mask'] > 0
        assert stats['throughput_masks_per_second'] > 0

    def test_inference_engine_reload(self, tmp_path):
        """测试模型热重载"""
        model_dir = self._create_exported_model(tmp_path)

        from surrogate.inference_server import SurrogateInferenceEngine

        engine = SurrogateInferenceEngine(model_dir, device='cpu')

        # 先做一些推理
        mask = np.random.rand(64, 64).astype(np.float32)
        engine.predict(mask)

        stats_before = engine.get_stats()
        assert stats_before['total_requests'] == 1

        # 重载
        engine.reload()

        stats_after = engine.get_stats()
        assert stats_after['total_requests'] == 0  # 统计被重置


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
