# -*- coding: utf-8 -*-
"""
代理模型独立推理服务

提供低延迟 REST API，供外部 OPC 工具或产线系统调用。
支持 ONNX 和 TorchScript 两种部署格式，不依赖完整 Python 仿真环境。

主要特性:
- 自动检测并加载 ONNX 或 TorchScript 模型
- 支持单张和批量掩模推理
- 健康检查和模型元数据接口
- 内置性能统计（延迟、吞吐量）
- 支持 CORS，可直接被前端调用
- 优雅关闭和热重载支持

启动方式:
    python -m surrogate.inference_server --model-dir ./surrogate_checkpoints --host 0.0.0.0 --port 8000

或者使用 uvicorn:
    uvicorn surrogate.inference_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@dataclass
class InferenceStats:
    """推理统计信息"""
    total_requests: int = 0
    total_masks: int = 0
    total_time_ms: float = 0.0
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, num_masks: int, latency_ms: float):
        with self.lock:
            self.total_requests += 1
            self.total_masks += num_masks
            self.total_time_ms += latency_ms
            self.recent_latencies.append(latency_ms / max(num_masks, 1))

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            if self.total_masks == 0:
                return {
                    'total_requests': 0,
                    'total_masks': 0,
                    'avg_latency_ms_per_mask': 0.0,
                    'throughput_masks_per_second': 0.0,
                    'recent_p50_latency_ms': 0.0,
                    'recent_p95_latency_ms': 0.0,
                    'recent_p99_latency_ms': 0.0,
                }

            recent = list(self.recent_latencies)
            recent_sorted = sorted(recent)
            n = len(recent_sorted)

            return {
                'total_requests': self.total_requests,
                'total_masks': self.total_masks,
                'avg_latency_ms_per_mask': self.total_time_ms / self.total_masks,
                'throughput_masks_per_second': self.total_masks / (self.total_time_ms / 1000.0) if self.total_time_ms > 0 else 0.0,
                'recent_p50_latency_ms': recent_sorted[int(n * 0.5)] if n > 0 else 0.0,
                'recent_p95_latency_ms': recent_sorted[int(n * 0.95)] if n > 0 else 0.0,
                'recent_p99_latency_ms': recent_sorted[int(n * 0.99)] if n > 0 else 0.0,
                'recent_window_size': n,
            }


class SurrogateInferenceEngine:
    """
    代理模型推理引擎

    自动选择最优后端（ONNX Runtime 优先，其次 TorchScript），
    提供统一的推理接口。
    """

    def __init__(
        self,
        model_dir: str,
        prefer_onnx: bool = True,
        device: str = 'auto',
    ):
        """
        Args:
            model_dir: 模型目录，包含 model.onnx 或 model.pt 和 metadata.json
            prefer_onnx: 优先使用 ONNX Runtime
            device: 推理设备: 'auto', 'cpu', 'cuda'
        """
        self.model_dir = model_dir
        self.prefer_onnx = prefer_onnx
        self.device = device
        self.metadata: Optional[Dict[str, Any]] = None
        self.session = None
        self.torchscript_model = None
        self.backend: Optional[str] = None
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.stats = InferenceStats()

        self._load_metadata()
        self._load_model()

    def _load_metadata(self):
        """加载模型元数据"""
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"加载元数据: {metadata_path}")

            self.input_shape = tuple(self.metadata['model']['input_shape'])
            self.output_shape = tuple(self.metadata['model']['output_shape'])
        else:
            logger.warning(f"未找到元数据文件: {metadata_path}")

    def _load_model(self):
        """加载模型，优先 ONNX Runtime"""
        onnx_path = os.path.join(self.model_dir, 'model.onnx')
        torchscript_path = os.path.join(self.model_dir, 'model.pt')

        if self.prefer_onnx and os.path.exists(onnx_path):
            self._load_onnx(onnx_path)
        elif os.path.exists(torchscript_path):
            self._load_torchscript(torchscript_path)
        elif os.path.exists(onnx_path):
            self._load_onnx(onnx_path)
        else:
            raise FileNotFoundError(
                f"未找到可加载的模型文件 (ONNX: {onnx_path}, TorchScript: {torchscript_path})"
            )

    def _load_onnx(self, onnx_path: str):
        """加载 ONNX 模型"""
        try:
            import onnxruntime as ort

            providers = []
            if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 0
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=providers,
            )

            self.backend = 'onnxruntime'
            provider = self.session.get_providers()[0]
            logger.info(f"ONNX 模型已加载，使用 provider: {provider}")

        except ImportError:
            logger.warning("onnxruntime 未安装，尝试使用 TorchScript")
            torchscript_path = os.path.join(self.model_dir, 'model.pt')
            if os.path.exists(torchscript_path):
                self._load_torchscript(torchscript_path)
            else:
                raise ImportError(
                    "onnxruntime 未安装且 TorchScript 模型不存在，"
                    "请安装: pip install onnxruntime"
                )

    def _load_torchscript(self, torchscript_path: str):
        """加载 TorchScript 模型"""
        try:
            import torch

            if self.device == 'cuda' and torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            self.torchscript_model = torch.jit.load(
                torchscript_path, map_location=device
            )
            self.torchscript_model.eval()
            self.backend = 'torchscript'
            logger.info(f"TorchScript 模型已加载，设备: {device}")

        except ImportError:
            raise ImportError(
                "PyTorch 未安装，请安装: pip install torch"
            )

    def _preprocess(self, masks: np.ndarray) -> np.ndarray:
        """预处理输入掩模"""
        if masks.ndim == 2:
            masks = masks[np.newaxis, np.newaxis, :, :]
        elif masks.ndim == 3:
            masks = masks[:, np.newaxis, :, :]
        elif masks.ndim == 4:
            if masks.shape[1] != 1:
                masks = masks[:, 0:1, :, :]
        else:
            raise ValueError(f"不支持的输入维度: {masks.ndim}")

        masks = masks.astype(np.float32)
        masks = np.clip(masks, 0.0, 1.0)

        return masks

    def _postprocess(self, outputs: np.ndarray) -> np.ndarray:
        """后处理输出空间像"""
        if outputs.ndim == 4 and outputs.shape[1] == 1:
            outputs = outputs[:, 0]

        outputs = np.clip(outputs, 0.0, 1.0)
        return outputs

    def predict(self, masks: np.ndarray) -> np.ndarray:
        """
        预测空间像

        Args:
            masks: 输入掩模，支持:
                - 单张: (H, W) float32/64
                - 批量: (N, H, W) float32/64 或 (N, 1, H, W)

        Returns:
            空间像数组:
                - 单张输入: (H, W) float32
                - 批量输入: (N, H, W) float32
        """
        was_single = masks.ndim == 2

        t0 = time.time()
        input_data = self._preprocess(masks)
        num_masks = input_data.shape[0]

        if self.backend == 'onnxruntime':
            outputs = self._predict_onnx(input_data)
        elif self.backend == 'torchscript':
            outputs = self._predict_torchscript(input_data)
        else:
            raise RuntimeError("模型未加载")

        outputs = self._postprocess(outputs)

        latency_ms = (time.time() - t0) * 1000
        self.stats.record(num_masks, latency_ms)

        if was_single:
            return outputs[0]
        return outputs

    def _predict_onnx(self, input_data: np.ndarray) -> np.ndarray:
        """ONNX Runtime 推理"""
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data})
        return outputs[0]

    def _predict_torchscript(self, input_data: np.ndarray) -> np.ndarray:
        """TorchScript 推理"""
        import torch

        with torch.no_grad():
            input_tensor = torch.from_numpy(input_data)
            if next(self.torchscript_model.parameters()).is_cuda:
                input_tensor = input_tensor.cuda()
            output_tensor = self.torchscript_model(input_tensor)
            return output_tensor.cpu().numpy()

    def predict_batch(
        self,
        masks_list: List[np.ndarray],
        max_batch_size: int = 32,
    ) -> List[np.ndarray]:
        """
        批量预测，自动分批处理

        Args:
            masks_list: 掩模列表，每个元素 (H, W)
            max_batch_size: 最大批大小

        Returns:
            空间像列表，与输入一一对应
        """
        if not masks_list:
            return []

        if self.metadata and 'model' in self.metadata:
            h, w = self.metadata['model']['input_shape'][2:]
        else:
            h, w = masks_list[0].shape

        batch_input = np.stack(masks_list, axis=0)

        results = []
        for start in range(0, len(masks_list), max_batch_size):
            end = min(start + max_batch_size, len(masks_list))
            batch = batch_input[start:end]
            batch_result = self.predict(batch)
            for i in range(len(batch_result)):
                results.append(batch_result[i])

        return results

    def get_metadata(self) -> Dict[str, Any]:
        """获取模型元数据"""
        if self.metadata is None:
            return {
                'backend': self.backend,
                'model_dir': self.model_dir,
            }
        return {
            'backend': self.backend,
            'model_dir': self.model_dir,
            **self.metadata,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计"""
        return self.stats.get_stats()

    def reload(self):
        """热重载模型"""
        logger.info("热重载模型...")
        self._load_metadata()
        self._load_model()
        self.stats = InferenceStats()
        logger.info("模型重载完成")


# ======================================================================
# FastAPI 服务
# ======================================================================

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class PredictRequest(BaseModel):
    """推理请求体"""
    masks: List[List[List[float]]] = Field(
        ...,
        description="掩模数组列表，每个掩模为二维数组 (H, W)，值域 [0, 1]",
        example=[[[0.0, 1.0], [1.0, 0.0]]]
    )
    return_masks: Optional[bool] = Field(
        False,
        description="是否在响应中返回原始掩模"
    )


class PredictResponse(BaseModel):
    """推理响应体"""
    aerial_images: List[List[List[float]]] = Field(
        ...,
        description="空间像数组列表，与输入掩模一一对应"
    )
    masks: Optional[List[List[List[float]]]] = Field(
        None,
        description="原始掩模（如果请求中 return_masks=true）"
    )
    latency_ms: float = Field(
        ...,
        description="本次推理总耗时（毫秒）"
    )
    num_masks: int = Field(
        ...,
        description="处理的掩模数量"
    )


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态: 'healthy' 或 'unhealthy'")
    backend: Optional[str] = Field(None, description="推理后端")
    model_loaded: bool = Field(..., description="模型是否已加载")
    uptime_seconds: float = Field(..., description="服务运行时间（秒）")


class MetadataResponse(BaseModel):
    """元数据响应"""
    backend: str
    model_dir: str
    model: Optional[Dict[str, Any]] = None
    export: Optional[Dict[str, Any]] = None
    preprocessing: Optional[Dict[str, Any]] = None
    postprocessing: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    """统计信息响应"""
    total_requests: int
    total_masks: int
    avg_latency_ms_per_mask: float
    throughput_masks_per_second: float
    recent_p50_latency_ms: float
    recent_p95_latency_ms: float
    recent_p99_latency_ms: float


_engine: Optional[SurrogateInferenceEngine] = None
_start_time: float = time.time()


def create_app(model_dir: str, prefer_onnx: bool = True, device: str = 'auto') -> FastAPI:
    """
    创建 FastAPI 应用实例

    Args:
        model_dir: 模型目录
        prefer_onnx: 优先使用 ONNX Runtime
        device: 推理设备

    Returns:
        FastAPI 应用实例
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI 未安装，请安装: pip install fastapi uvicorn pydantic"
        )

    app = FastAPI(
        title="代理模型推理服务",
        description="光刻掩模 → 空间像映射的低延迟推理服务，支持 ONNX 和 TorchScript 模型",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    global _engine
    _engine = SurrogateInferenceEngine(model_dir, prefer_onnx, device)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "代理模型推理服务",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "metadata": "/metadata",
                "predict": "/predict",
                "predict_batch": "/predict/batch",
                "stats": "/stats",
                "reload": "/reload",
            }
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """健康检查接口"""
        return {
            "status": "healthy" if _engine is not None else "unhealthy",
            "backend": _engine.backend if _engine else None,
            "model_loaded": _engine is not None,
            "uptime_seconds": time.time() - _start_time,
        }

    @app.get("/metadata", response_model=MetadataResponse)
    async def metadata():
        """获取模型元数据"""
        if _engine is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        return _engine.get_metadata()

    @app.get("/stats", response_model=StatsResponse)
    async def stats():
        """获取推理统计信息"""
        if _engine is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        return _engine.get_stats()

    @app.post("/reload")
    async def reload_model():
        """热重载模型"""
        if _engine is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        try:
            _engine.reload()
            return {"status": "success", "message": "模型重载完成"}
        except Exception as e:
            logger.error(f"模型重载失败: {e}")
            raise HTTPException(status_code=500, detail=f"模型重载失败: {str(e)}")

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        """
        单张或批量掩模推理

        - **masks**: 掩模数组列表，每个为 (H, W) 二维数组
        - **return_masks**: 是否返回原始掩模

        返回空间像数组，与输入一一对应
        """
        if _engine is None:
            raise HTTPException(status_code=503, detail="模型未加载")

        try:
            masks_np = [np.array(m, dtype=np.float32) for m in request.masks]

            if not masks_np:
                raise HTTPException(status_code=400, detail="masks 不能为空")

            expected_shape = masks_np[0].shape
            for i, m in enumerate(masks_np):
                if m.shape != expected_shape:
                    raise HTTPException(
                        status_code=400,
                        detail=f"第 {i} 个掩模形状 {m.shape} 与第一个 {expected_shape} 不一致"
                    )

            batch_input = np.stack(masks_np, axis=0)

            t0 = time.time()
            outputs = _engine.predict(batch_input)
            latency_ms = (time.time() - t0) * 1000

            response = {
                "aerial_images": outputs.tolist(),
                "latency_ms": latency_ms,
                "num_masks": len(masks_np),
            }

            if request.return_masks:
                response["masks"] = request.masks

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

    @app.post("/predict/batch", response_model=PredictResponse)
    async def predict_batch(
        max_batch_size: int = Query(32, ge=1, le=256, description="最大批大小"),
        file: UploadFile = File(..., description="NPY 或 NPZ 文件，包含 masks 数组"),
    ):
        """
        批量文件上传推理

        上传包含 masks 数组的 .npy 或 .npz 文件，支持大量掩模批量处理。

        - **文件格式**:
          - .npy: 形状为 (N, H, W) 的 float32 数组
          - .npz: 包含 'masks' 键，值为 (N, H, W) float32 数组
        """
        if _engine is None:
            raise HTTPException(status_code=503, detail="模型未加载")

        try:
            content = await file.read()
            filename = file.filename or ''

            import io
            data = np.load(io.BytesIO(content))

            if filename.endswith('.npz'):
                if 'masks' not in data.files:
                    raise HTTPException(
                        status_code=400,
                        detail="NPZ 文件必须包含 'masks' 键"
                    )
                masks = data['masks']
            else:
                masks = data

            if masks.ndim != 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"输入形状应为 (N, H, W)，实际为 {masks.shape}"
                )

            t0 = time.time()
            outputs = _engine.predict(masks)
            latency_ms = (time.time() - t0) * 1000

            return {
                "aerial_images": outputs.tolist(),
                "latency_ms": latency_ms,
                "num_masks": len(masks),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"批量推理失败: {e}")
            raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")

    return app


# ======================================================================
# 命令行入口
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="代理模型独立推理服务"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="模型目录，包含 model.onnx 或 model.pt 和 metadata.json",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="监听端口",
    )
    parser.add_argument(
        "--prefer-torchscript",
        action="store_true",
        help="优先使用 TorchScript 而非 ONNX Runtime",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推理设备",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数 (仅生产环境建议 >1)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not HAS_FASTAPI:
        logger.error("FastAPI 未安装，请先安装依赖:")
        logger.error("  pip install fastapi uvicorn pydantic python-multipart")
        logger.error("  pip install onnxruntime  # 或 onnxruntime-gpu")
        sys.exit(1)

    app = create_app(
        model_dir=args.model_dir,
        prefer_onnx=not args.prefer_torchscript,
        device=args.device,
    )

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
