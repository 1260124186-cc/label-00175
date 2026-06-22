# -*- coding: utf-8 -*-
"""
服务配置管理模块

支持 YAML/JSON 配置文件、环境变量覆盖、命令行参数三级配置优先级。
配置按功能模块化，支持热重载。
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class InferenceEngineConfig:
    """推理引擎配置"""
    engine_type: str = "auto"
    model_dir: str = "./models"
    model_filename: Optional[str] = None
    prefer_onnx: bool = True
    device: str = "auto"
    intra_op_threads: int = 0
    inter_op_threads: int = 0
    enable_graph_optimization: bool = True
    warmup_iterations: int = 10
    max_batch_size: int = 64
    memory_pool_mb: int = 2048
    hopkins_kernel_dir: Optional[str] = None
    hopkins_approximation_order: int = 2


@dataclass
class GrpcConfig:
    """gRPC 服务配置"""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 8
    max_message_size_mb: int = 64
    enable_reflection: bool = True
    keepalive_time_ms: int = 30000
    keepalive_timeout_ms: int = 10000
    keepalive_permit_without_calls: bool = True
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class HttpConfig:
    """HTTP/REST 服务配置"""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_docs: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 600
    max_upload_size_mb: int = 128
    enable_gzip: bool = True


@dataclass
class MetricsConfig:
    """监控指标配置"""
    enabled: bool = True
    prometheus_port: int = 9090
    enable_prometheus: bool = True
    collect_latency_histogram: bool = True
    latency_histogram_buckets_ms: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    )
    log_level: str = "INFO"
    log_format: str = "json"
    enable_file_logging: bool = False
    log_file_path: str = "./logs/inference_service.log"
    log_rotation_max_mb: int = 100
    log_rotation_backup_count: int = 10


@dataclass
class ServiceConfig:
    """服务主配置"""
    service_name: str = "litho-inference-service"
    service_id: str = field(default_factory=lambda: f"inf-{os.getpid()}-{os.urandom(4).hex()}")
    environment: str = "production"
    timezone: str = "Asia/Shanghai"
    graceful_shutdown_timeout_sec: int = 30
    pid_file: Optional[str] = None
    engine: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    grpc: GrpcConfig = field(default_factory=GrpcConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_yaml(self) -> str:
        try:
            import yaml
            return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)
        except ImportError:
            logger.warning("PyYAML 未安装，使用 JSON 格式替代")
            return self.to_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceConfig":
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if isinstance(attr, (InferenceEngineConfig, GrpcConfig, HttpConfig, MetricsConfig)):
                    sub_cfg = type(attr)()
                    for k, v in (value or {}).items():
                        if hasattr(sub_cfg, k):
                            setattr(sub_cfg, k, v)
                    setattr(cfg, key, sub_cfg)
                else:
                    setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_json(cls, json_str: str) -> "ServiceConfig":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ServiceConfig":
        try:
            import yaml
            return cls.from_dict(yaml.safe_load(yaml_str))
        except ImportError:
            raise ImportError("加载 YAML 配置需要 PyYAML: pip install pyyaml")


def default_config() -> ServiceConfig:
    """获取默认配置"""
    return ServiceConfig()


def load_config(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    env_prefix: str = "LITHO_INF_",
) -> ServiceConfig:
    """
    加载配置，优先级:
    1. 环境变量 (env_prefix + 路径)
    2. config_dict 传入的字典
    3. config_path 指定的配置文件
    4. 默认配置
    """
    cfg = default_config()

    if config_path and os.path.exists(config_path):
        ext = Path(config_path).suffix.lower()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            if ext in (".yaml", ".yml"):
                cfg = ServiceConfig.from_yaml(content)
            elif ext == ".json":
                cfg = ServiceConfig.from_json(content)
            else:
                logger.warning(f"未知配置文件格式 {ext}，尝试 YAML 解析")
                try:
                    cfg = ServiceConfig.from_yaml(content)
                except Exception:
                    cfg = ServiceConfig.from_json(content)
            logger.info(f"已加载配置文件: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败 {config_path}: {e}，使用默认配置")

    if config_dict:
        file_cfg = ServiceConfig.from_dict(config_dict)
        for field_name in cfg.to_dict():
            if hasattr(file_cfg, field_name):
                setattr(cfg, field_name, getattr(file_cfg, field_name))

    cfg = _apply_env_overrides(cfg, env_prefix)

    return cfg


def _apply_env_overrides(cfg: ServiceConfig, env_prefix: str) -> ServiceConfig:
    """应用环境变量覆盖配置"""
    env_mappings = {
        "SERVICE_NAME": ("service_name", None),
        "ENVIRONMENT": ("environment", None),
        "ENGINE_TYPE": ("engine", "engine_type"),
        "MODEL_DIR": ("engine", "model_dir"),
        "PREFER_ONNX": ("engine", "prefer_onnx"),
        "DEVICE": ("engine", "device"),
        "MAX_BATCH_SIZE": ("engine", "max_batch_size"),
        "GRPC_ENABLED": ("grpc", "enabled"),
        "GRPC_HOST": ("grpc", "host"),
        "GRPC_PORT": ("grpc", "port"),
        "GRPC_API_KEY": ("grpc", "api_key"),
        "HTTP_ENABLED": ("http", "enabled"),
        "HTTP_HOST": ("http", "host"),
        "HTTP_PORT": ("http", "port"),
        "HTTP_API_KEY": ("http", "api_key"),
        "HTTP_RATE_LIMIT": ("http", "rate_limit_per_minute"),
        "METRICS_ENABLED": ("metrics", "enabled"),
        "PROMETHEUS_PORT": ("metrics", "prometheus_port"),
        "LOG_LEVEL": ("metrics", "log_level"),
    }

    for env_key, (section, field_name) in env_mappings.items():
        full_env_key = f"{env_prefix}{env_key}"
        value = os.environ.get(full_env_key)
        if value is not None:
            try:
                target = cfg if section is None else getattr(cfg, section)
                if hasattr(target, field_name):
                    current_val = getattr(target, field_name)
                    if isinstance(current_val, bool):
                        parsed = value.lower() in ("1", "true", "yes", "on")
                    elif isinstance(current_val, int):
                        parsed = int(value)
                    elif isinstance(current_val, float):
                        parsed = float(value)
                    elif isinstance(current_val, list):
                        parsed = [v.strip() for v in value.split(",") if v.strip()]
                    else:
                        parsed = value
                    setattr(target, field_name, parsed)
                    logger.debug(f"环境变量覆盖: {full_env_key} = {value}")
            except Exception as e:
                logger.warning(f"解析环境变量 {full_env_key}={value} 失败: {e}")

    return cfg
