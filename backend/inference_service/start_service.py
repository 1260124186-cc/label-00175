#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
产线推理微服务 - 统一启动入口

同时启动 gRPC 和 HTTP 服务，支持:
- 配置文件/环境变量/命令行三级参数覆盖
- gRPC + HTTP 双协议同时运行
- 优雅关闭 (SIGINT/SIGTERM)
- Prometheus 监控指标端口

用法:
    # 默认配置
    python -m inference_service.start_service

    # 指定配置文件
    python -m inference_service.start_service --config config.yaml

    # 命令行参数覆盖
    python -m inference_service.start_service \
        --engine-type hopkins_lite \
        --grpc-port 50051 --http-port 8080 \
        --log-level DEBUG
"""

from __future__ import annotations

import os
import sys
import time
import signal
import logging
import argparse
import threading
from typing import Optional

logger = logging.getLogger("inference_service")

_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SERVICE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

from .config import ServiceConfig, load_config, default_config
from .inference_engine import BaseInferenceEngine, EngineFactory
from .epe_estimator import EpeEstimator
from .grpc_server import GrpcServer
from .http_server import HttpServer


def _setup_logging(log_level: str = "INFO", log_format: str = "text"):
    """配置日志"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = (
        "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
        if log_format == "text"
        else None
    )
    handlers = [logging.StreamHandler(sys.stdout)]
    if fmt:
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        handlers[0].setFormatter(formatter)
    logging.basicConfig(level=level, handlers=handlers, force=True)


def _parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="光刻产线推理微服务 (Litho Inference Microservice)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="配置文件路径 (YAML/JSON)",
    )
    parser.add_argument(
        "--engine-type",
        type=str,
        default="auto",
        choices=["auto", "surrogate", "hopkins_lite"],
        help="推理引擎类型",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="代理模型目录 (含 model.onnx / model.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推理设备",
    )

    grpc_group = parser.add_argument_group("gRPC")
    grpc_group.add_argument("--no-grpc", action="store_true", help="禁用 gRPC")
    grpc_group.add_argument("--grpc-host", type=str, default="0.0.0.0", help="gRPC 监听地址")
    grpc_group.add_argument("--grpc-port", type=int, default=50051, help="gRPC 监听端口")
    grpc_group.add_argument("--grpc-workers", type=int, default=8, help="gRPC 工作线程数")

    http_group = parser.add_argument_group("HTTP")
    http_group.add_argument("--no-http", action="store_true", help="禁用 HTTP")
    http_group.add_argument("--http-host", type=str, default="0.0.0.0", help="HTTP 监听地址")
    http_group.add_argument("--http-port", type=int, default=8080, help="HTTP 监听端口")
    http_group.add_argument("--http-workers", type=int, default=1, help="HTTP 工作进程数")

    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument("--log-level", type=str, default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    misc_group.add_argument("--no-docs", action="store_true", help="禁用 Swagger 文档")
    misc_group.add_argument("--api-key", type=str, default=None, help="HTTP API Key")

    return parser.parse_args()


def _apply_cli_overrides(cfg: ServiceConfig, args: argparse.Namespace) -> ServiceConfig:
    """应用命令行参数覆盖"""
    if args.engine_type:
        cfg.engine.engine_type = args.engine_type
    if args.model_dir:
        cfg.engine.model_dir = args.model_dir
    if args.device:
        cfg.engine.device = args.device

    cfg.grpc.enabled = not args.no_grpc
    if args.grpc_host:
        cfg.grpc.host = args.grpc_host
    if args.grpc_port:
        cfg.grpc.port = args.grpc_port
    if args.grpc_workers:
        cfg.grpc.max_workers = args.grpc_workers

    cfg.http.enabled = not args.no_http
    if args.http_host:
        cfg.http.host = args.http_host
    if args.http_port:
        cfg.http.port = args.http_port
    if args.http_workers:
        cfg.http.workers = args.http_workers

    cfg.metrics.log_level = args.log_level
    if args.no_docs:
        cfg.http.enable_docs = False
    if args.api_key:
        cfg.http.api_key = args.api_key

    return cfg


class InferenceServiceRunner:
    """推理服务运行器，管理 gRPC/HTTP 服务生命周期"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self._config = config or default_config()
        self._engine: Optional[BaseInferenceEngine] = None
        self._epe_estimator: Optional[EpeEstimator] = None
        self._grpc_server: Optional[GrpcServer] = None
        self._http_server: Optional[HttpServer] = None
        self._shutdown_event = threading.Event()
        self._http_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def initialize(self):
        """初始化引擎"""
        logger.info("=" * 60)
        logger.info("光刻产线推理微服务启动中...")
        logger.info("=" * 60)

        engine_cfg = {
            "model_dir": self._config.engine.model_dir,
            "prefer_onnx": self._config.engine.prefer_onnx,
            "device": self._config.engine.device,
            "warmup_iterations": self._config.engine.warmup_iterations,
            "hopkins_kernel_dir": self._config.engine.hopkins_kernel_dir,
        }
        self._engine = EngineFactory.create(
            engine_type=self._config.engine.engine_type,
            config=engine_cfg,
            initialize=True,
        )
        self._epe_estimator = EpeEstimator()

        logger.info(
            f"引擎初始化完成: type={self._engine.engine_type.value}, "
            f"ready={self._engine.is_ready}"
        )

    def start(self):
        """启动所有服务"""
        self.initialize()

        if self._config.grpc.enabled:
            self._start_grpc()

        if self._config.http.enabled:
            self._start_http()

        logger.info("-" * 60)
        if self._config.grpc.enabled:
            logger.info(f"gRPC 服务:    {self._config.grpc.host}:{self._config.grpc.port}")
        if self._config.http.enabled:
            logger.info(f"HTTP 服务:    http://{self._config.http.host}:{self._config.http.port}")
            if self._config.http.enable_docs:
                logger.info(f"  Swagger UI: http://{self._config.http.host}:{self._config.http.port}{self._config.http.docs_url}")
        logger.info("=" * 60)
        logger.info("服务启动完成，等待请求...")

    def _start_grpc(self):
        try:
            self._grpc_server = GrpcServer(self._config, self._engine)
            self._grpc_server.start()
        except ImportError as e:
            logger.warning(f"gRPC 不可用，跳过: {e}")
            self._config.grpc.enabled = False
            self._grpc_server = None

    def _start_http(self):
        try:
            self._http_server = HttpServer(self._config, self._engine, self._epe_estimator)

            def _run_http():
                try:
                    self._http_server.run()
                except Exception as e:
                    logger.error(f"HTTP 服务异常: {e}")
                    self._shutdown_event.set()

            self._http_thread = threading.Thread(target=_run_http, daemon=True, name="http-server")
            self._http_thread.start()
        except ImportError as e:
            logger.warning(f"HTTP 不可用，跳过: {e}")
            self._config.http.enabled = False
            self._http_server = None

    def wait(self):
        """阻塞等待直到收到关闭信号"""
        try:
            while not self._shutdown_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C, 准备关闭...")
        finally:
            self.shutdown()

    def shutdown(self):
        """优雅关闭"""
        logger.info("正在关闭服务...")
        self._shutdown_event.set()

        if self._grpc_server:
            try:
                self._grpc_server.stop(grace=self._config.graceful_shutdown_timeout_sec)
            except Exception as e:
                logger.warning(f"关闭 gRPC 出错: {e}")

        if self._engine:
            try:
                self._engine.shutdown()
            except Exception as e:
                logger.warning(f"关闭引擎出错: {e}")

        logger.info("服务已关闭")

    def handle_signal(self, signum, frame):
        """信号处理器"""
        sig_name = signal.Signals(signum).name
        logger.info(f"收到信号 {sig_name}, 准备优雅关闭...")
        self._shutdown_event.set()


def main():
    args = _parse_args()
    _setup_logging(args.log_level)

    try:
        cfg = load_config(config_path=args.config)
    except Exception as e:
        logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        cfg = default_config()

    cfg = _apply_cli_overrides(cfg, args)

    if not cfg.grpc.enabled and not cfg.http.enabled:
        logger.error("gRPC 和 HTTP 均被禁用，没有可用服务！")
        sys.exit(1)

    runner = InferenceServiceRunner(cfg)

    signal.signal(signal.SIGINT, runner.handle_signal)
    signal.signal(signal.SIGTERM, runner.handle_signal)

    try:
        runner.start()
        runner.wait()
    except Exception as e:
        logger.error(f"服务运行失败: {e}", exc_info=True)
        runner.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
