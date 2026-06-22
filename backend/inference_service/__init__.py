# -*- coding: utf-8 -*-
"""
产线推理微服务模块 (Production Inference Microservice)

将训练好的代理模型或轻量化 Hopkins 近似部署为独立 gRPC/HTTP 微服务，
接收掩模片段返回空间像或 EPE 估计，供产线 OPC 引擎或 MES 系统低延迟调用，
与主研究框架解耦部署。

核心特性:
- 双协议支持: gRPC (高性能) + HTTP/REST (易用性)
- 双推理引擎: 
  1. SurrogateEngine: 神经网络代理模型 (ONNX/TorchScript)
  2. HopkinsLiteEngine: 轻量化 Hopkins 近似 (无训练依赖)
- 双输出模式: 空间像 (Aerial Image) + EPE 直接估计
- 产线级特性: 健康检查、性能监控、限流、优雅关闭、热重载
- 低延迟优化: 连接池、批量推理、模型预热、内存池

典型用法:
    # 1. 启动服务 (命令行)
    python -m inference_service --config config.yaml
    
    # 2. gRPC 客户端调用
    from inference_service.grpc_client import InferenceGrpcClient
    client = InferenceGrpcClient("localhost:50051")
    aerial = client.predict_aerial(mask_array)
    
    # 3. HTTP 客户端调用
    from inference_service.http_client import InferenceHttpClient
    client = InferenceHttpClient("http://localhost:8080")
    epe_result = client.estimate_epe(mask_array, target_array)

模块结构:
    inference_service/
    ├── __init__.py              # 本文件
    ├── config.py                # 服务配置管理
    ├── schemas.py               # 数据结构定义 (Pydantic)
    ├── inference_engine.py      # 推理引擎抽象 + 双引擎实现
    ├── epe_estimator.py         # EPE 快速估计模块
    ├── proto/
    │   └── inference.proto      # gRPC Protocol Buffers 定义
    ├── grpc_server.py           # gRPC 服务端实现
    ├── grpc_client.py           # gRPC 客户端 SDK
    ├── http_server.py           # FastAPI HTTP/REST 服务端
    ├── http_client.py           # HTTP 客户端 SDK
    ├── requirements.txt         # Python 依赖清单
    ├── Dockerfile               # 容器化部署
    ├── docker-compose.yml       # 容器编排 (含 Prometheus/Grafana)
    └── start_service.py         # 统一启动入口
"""

from __future__ import annotations

from .config import (
    ServiceConfig,
    InferenceEngineConfig,
    GrpcConfig,
    HttpConfig,
    MetricsConfig,
    load_config,
    default_config,
)
from .schemas import (
    InferenceMode,
    OutputType,
    AerialImageRequest,
    AerialImageResponse,
    EpeEstimateRequest,
    EpeEstimateResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    ServiceInfo,
    HealthStatus,
    PerformanceMetrics,
)
from .inference_engine import (
    BaseInferenceEngine,
    SurrogateEngine,
    HopkinsLiteEngine,
    EngineFactory,
    create_engine,
)
from .epe_estimator import (
    EpeEstimator,
    EpeResult,
    FastEpeEstimator,
)

__version__ = "1.0.0"
__author__ = "Lithography Research Team"

__all__ = [
    # Config
    "ServiceConfig",
    "InferenceEngineConfig",
    "GrpcConfig",
    "HttpConfig",
    "MetricsConfig",
    "load_config",
    "default_config",
    # Schemas
    "InferenceMode",
    "OutputType",
    "AerialImageRequest",
    "AerialImageResponse",
    "EpeEstimateRequest",
    "EpeEstimateResponse",
    "BatchInferenceRequest",
    "BatchInferenceResponse",
    "ServiceInfo",
    "HealthStatus",
    "PerformanceMetrics",
    # Engines
    "BaseInferenceEngine",
    "SurrogateEngine",
    "HopkinsLiteEngine",
    "EngineFactory",
    "create_engine",
    # EPE
    "EpeEstimator",
    "EpeResult",
    "FastEpeEstimator",
]
