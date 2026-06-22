# -*- coding: utf-8 -*-
"""
gRPC 服务端实现

封装推理引擎和 EPE 估计器，通过 gRPC 对外提供低延迟服务。
支持流式推理、健康检查、模型热重载。
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple, Iterator
from concurrent import futures
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .config import ServiceConfig, GrpcConfig
from .schemas import InferenceMode, OpticalParams as SchemasOpticalParams
from .inference_engine import BaseInferenceEngine, EngineFactory
from .epe_estimator import EpeEstimator, EpeResult

try:
    import grpc
    from .proto import ensure_grpc, get_pb2, get_pb2_grpc
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False


def _mode_enum_to_str(mode_value: int) -> str:
    mapping = {0: "auto", 1: "surrogate", 2: "hopkins_lite"}
    return mapping.get(mode_value, "auto")


def _array2d_to_numpy(arr) -> np.ndarray:
    """gRPC Array2D -> numpy (H, W)"""
    h = arr.height
    w = arr.width
    return np.array(arr.data, dtype=np.float32).reshape(h, w)


def _numpy_to_array2d(arr: np.ndarray, pb2):
    """numpy (H, W) -> gRPC Array2D"""
    h, w = arr.shape
    return pb2.Array2D(
        height=int(h),
        width=int(w),
        data=arr.astype(np.float32).flatten().tolist(),
    )


def _optical_params_to_dict(op) -> Optional[Dict[str, Any]]:
    """gRPC OpticalParams -> dict"""
    if op is None:
        return None
    result = {
        "wavelength_nm": op.wavelength_nm,
        "na": op.na,
        "sigma": op.sigma,
        "defocus_nm": op.defocus_nm,
        "pixel_size_nm": op.pixel_size_nm,
        "illumination_type": op.illumination_type or "conventional",
    }
    if op.annular_sigma_inner is not None:
        result["annular_sigma_inner"] = op.annular_sigma_inner
    if op.annular_sigma_outer is not None:
        result["annular_sigma_outer"] = op.annular_sigma_outer
    if op.dipole_angle_deg is not None:
        result["dipole_angle_deg"] = op.dipole_angle_deg
    return result


class _LithoInferenceServiceServicer:
    """gRPC Servicer 实现"""

    def __init__(
        self,
        engine: BaseInferenceEngine,
        config: ServiceConfig,
        epe_estimator: Optional[EpeEstimator] = None,
    ):
        self._engine = engine
        self._config = config
        self._epe = epe_estimator or EpeEstimator()
        self._start_time = time.time()
        self._lock = threading.RLock()
        self._request_counter = 0

    # ------------------------------------------------------------------
    # PredictAerial
    # ------------------------------------------------------------------

    def PredictAerial(self, request, context):
        pb2 = get_pb2()
        req_id = request.request_id or f"grpc-{uuid.uuid4().hex[:12]}"
        t_total = time.time()

        try:
            if len(request.masks) == 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("masks 不能为空")
                return pb2.PredictAerialResponse()

            masks_np = [_array2d_to_numpy(m) for m in request.masks]
            batch = np.stack(masks_np, axis=0)

            mode = _mode_enum_to_str(request.inference_mode)
            opt_params = _optical_params_to_dict(request.optical_params) or {}

            t0 = time.time()
            aerial_batch = self._engine.predict(
                batch, optical_params=opt_params
            )
            latency_ms = (time.time() - t0) * 1000

            response = pb2.PredictAerialResponse(
                inference_mode_used=mode,
                engine_type=self._engine.engine_type.value,
                latency_ms_total=latency_ms,
                latency_ms_per_mask_avg=latency_ms / max(len(masks_np), 1),
                num_masks=len(masks_np),
                request_id=req_id,
            )

            for i in range(len(aerial_batch)):
                response.aerial_images.append(_numpy_to_array2d(aerial_batch[i], pb2))

            if request.return_input_mask:
                for m in request.masks:
                    response.masks.append(m)

            return response

        except Exception as e:
            logger.error(f"[gRPC] PredictAerial 失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.PredictAerialResponse(request_id=req_id)

    # ------------------------------------------------------------------
    # PredictAerialStream
    # ------------------------------------------------------------------

    def PredictAerialStream(self, request_iterator, context):
        pb2 = get_pb2()
        for request in request_iterator:
            yield self.PredictAerial(request, context)

    # ------------------------------------------------------------------
    # EstimateEpe
    # ------------------------------------------------------------------

    def EstimateEpe(self, request, context):
        pb2 = get_pb2()
        req_id = request.request_id or f"grpc-{uuid.uuid4().hex[:12]}"

        try:
            if len(request.masks) == 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("masks 不能为空")
                return pb2.EstimateEpeResponse(request_id=req_id)

            if len(request.targets) != len(request.masks):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"targets 数量 ({len(request.targets)}) 与 masks 不一致")
                return pb2.EstimateEpeResponse(request_id=req_id)

            masks_np = [_array2d_to_numpy(m) for m in request.masks]
            targets_np = [_array2d_to_numpy(t) for t in request.targets]
            batch = np.stack(masks_np, axis=0)

            mode = _mode_enum_to_str(request.inference_mode)
            opt_params = _optical_params_to_dict(request.optical_params) or {}

            t0 = time.time()
            aerial_batch = self._engine.predict(batch, optical_params=opt_params)
            latency_ms = (time.time() - t0) * 1000

            ps_nm = request.pixel_size_nm or 1.0
            th = request.threshold or 0.5

            epe_list: List[EpeResult] = []
            for i in range(len(masks_np)):
                epe_result = self._epe.compute(
                    aerial_batch[i], targets_np[i],
                    pixel_size_nm=ps_nm, threshold=th,
                )
                epe_list.append(epe_result)

            response = pb2.EstimateEpeResponse(
                inference_mode_used=mode,
                engine_type=self._engine.engine_type.value,
                latency_ms_total=latency_ms,
                latency_ms_per_mask_avg=latency_ms / max(len(masks_np), 1),
                num_masks=len(masks_np),
                request_id=req_id,
            )

            for e in epe_list:
                response.epe_results.append(pb2.EpeMetrics(
                    epe_mean_nm=e.epe_mean_nm,
                    epe_max_nm=e.epe_max_nm,
                    epe_std_nm=e.epe_std_nm,
                    epe_median_nm=e.epe_median_nm,
                    num_edge_pixels_wafer=e.num_edge_pixels_wafer,
                    num_edge_pixels_target=e.num_edge_pixels_target,
                ))

            if request.return_aerial_images:
                for i in range(len(aerial_batch)):
                    response.aerial_images.append(_numpy_to_array2d(aerial_batch[i], pb2))

            return response

        except Exception as e:
            logger.error(f"[gRPC] EstimateEpe 失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.EstimateEpeResponse(request_id=req_id)

    # ------------------------------------------------------------------
    # BatchInference
    # ------------------------------------------------------------------

    def BatchInference(self, request, context):
        pb2 = get_pb2()
        req_id = request.request_id or f"grpc-{uuid.uuid4().hex[:12]}"

        try:
            if len(request.masks) == 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("masks 不能为空")
                return pb2.BatchInferenceResponse(request_id=req_id)

            output_type = request.output_type
            masks_np = [_array2d_to_numpy(m) for m in request.masks]
            targets_np = [_array2d_to_numpy(t) for t in request.targets] if request.targets else None

            if targets_np is not None and len(targets_np) != len(masks_np):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("targets 与 masks 数量不一致")
                return pb2.BatchInferenceResponse(request_id=req_id)

            batch_size = min(request.max_batch_size or 32, 256)
            num_batches = (len(masks_np) + batch_size - 1) // batch_size

            mode = _mode_enum_to_str(request.inference_mode)
            opt_params = _optical_params_to_dict(request.optical_params) or {}

            all_aerials: List[np.ndarray] = []
            all_epes: List[EpeResult] = []
            t_total = time.time()

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, len(masks_np))
                chunk = np.stack(masks_np[start:end], axis=0)
                aerials = self._engine.predict(chunk, optical_params=opt_params)
                for i in range(len(aerials)):
                    all_aerials.append(aerials[i])

                if output_type in (1, 2) and targets_np is not None:
                    ps_nm = request.pixel_size_nm or 1.0
                    th = request.threshold or 0.5
                    for i in range(len(aerials)):
                        idx = start + i
                        epe = self._epe.compute(
                            aerials[i], targets_np[idx],
                            pixel_size_nm=ps_nm, threshold=th,
                        )
                        all_epes.append(epe)

            total_latency = (time.time() - t_total) * 1000

            response = pb2.BatchInferenceResponse(
                inference_mode_used=mode,
                engine_type=self._engine.engine_type.value,
                latency_ms_total=total_latency,
                latency_ms_per_mask_avg=total_latency / max(len(masks_np), 1),
                num_masks=len(masks_np),
                num_batches=num_batches,
                request_id=req_id,
            )

            if output_type in (0, 2):
                for a in all_aerials:
                    response.aerial_images.append(_numpy_to_array2d(a, pb2))

            if output_type in (1, 2):
                for e in all_epes:
                    response.epe_results.append(pb2.EpeMetrics(
                        epe_mean_nm=e.epe_mean_nm,
                        epe_max_nm=e.epe_max_nm,
                        epe_std_nm=e.epe_std_nm,
                        epe_median_nm=e.epe_median_nm,
                        num_edge_pixels_wafer=e.num_edge_pixels_wafer,
                        num_edge_pixels_target=e.num_edge_pixels_target,
                    ))

            return response

        except Exception as e:
            logger.error(f"[gRPC] BatchInference 失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.BatchInferenceResponse(request_id=req_id)

    # ------------------------------------------------------------------
    # 健康检查 / 监控
    # ------------------------------------------------------------------

    def HealthCheck(self, request, context):
        pb2 = get_pb2()
        engine_stats = self._engine.stats
        engine_ready = self._engine.is_ready

        status = 1 if engine_ready else 3

        checks = {
            "engine_ready": str(engine_ready),
            "engine_type": self._engine.engine_type.value,
            "total_masks": str(engine_stats.get("total_masks", 0)),
            "avg_latency_ms": f"{engine_stats.get('avg_latency_ms_per_mask', 0):.2f}",
        }

        return pb2.HealthCheckResponse(
            status=status,
            engine_status=str(engine_stats),
            grpc_enabled=True,
            http_enabled=self._config.http.enabled,
            uptime_seconds=time.time() - self._start_time,
            service_name=self._config.service_name,
            service_id=self._config.service_id,
            checks=checks,
        )

    def GetServiceInfo(self, request, context):
        pb2 = get_pb2()
        from . import __version__

        engine_stats = self._engine.stats
        return pb2.ServiceInfoResponse(
            name=self._config.service_name,
            version=__version__,
            environment=self._config.environment,
            engine={
                "type": self._engine.engine_type.value,
                "total_calls": str(engine_stats.get("total_calls", 0)),
                "total_masks": str(engine_stats.get("total_masks", 0)),
                "errors": str(engine_stats.get("errors", 0)),
            },
            endpoints={
                "grpc": f"{self._config.grpc.host}:{self._config.grpc.port}",
                "http": f"http://{self._config.http.host}:{self._config.http.port}",
            },
        )

    def GetMetrics(self, request, context):
        pb2 = get_pb2()
        stats = self._engine.stats
        return pb2.MetricsResponse(
            total_requests=stats.get("total_calls", 0),
            total_masks_processed=stats.get("total_masks", 0),
            total_errors=stats.get("errors", 0),
            avg_latency_ms=stats.get("avg_latency_ms_per_mask", 0),
            p50_latency_ms=stats.get("p50_latency_ms", 0),
            p95_latency_ms=stats.get("p95_latency_ms", 0),
            p99_latency_ms=stats.get("p99_latency_ms", 0),
            throughput_masks_per_sec=stats.get("throughput_masks_per_sec", 0),
        )

    def ReloadModel(self, request, context):
        pb2 = get_pb2()
        t0 = time.time()
        try:
            with self._lock:
                self._engine.reload({})
            return pb2.ReloadResponse(
                success=True,
                message="模型重载成功",
                reload_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return pb2.ReloadResponse(
                success=False,
                message=f"重载失败: {e}",
                reload_time_ms=(time.time() - t0) * 1000,
            )


class GrpcServer:
    """gRPC 服务器封装"""

    def __init__(self, config: ServiceConfig, engine: BaseInferenceEngine):
        if not HAS_GRPC:
            raise ImportError("grpcio 未安装")
        ensure_grpc()

        self._config = config
        self._engine = engine
        self._grpc_cfg: GrpcConfig = config.grpc
        self._server = None
        self._servicer = _LithoInferenceServiceServicer(engine, config)
        self._stopped = threading.Event()

    def start(self):
        """启动 gRPC 服务器"""
        pb2_grpc = get_pb2_grpc()

        options = [
            ("grpc.max_send_message_length", self._grpc_cfg.max_message_size_mb * 1024 * 1024),
            ("grpc.max_receive_message_length", self._grpc_cfg.max_message_size_mb * 1024 * 1024),
            ("grpc.keepalive_time_ms", self._grpc_cfg.keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", self._grpc_cfg.keepalive_timeout_ms),
            ("grpc.keepalive_permit_without_calls", int(self._grpc_cfg.keepalive_permit_without_calls)),
        ]

        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._grpc_cfg.max_workers),
            options=options,
        )

        pb2_grpc.add_LithoInferenceServiceServicer_to_server(self._servicer, self._server)

        if self._grpc_cfg.enable_reflection:
            try:
                from grpc_reflection.v1alpha import reflection
                pb2 = get_pb2()
                service_names = (
                    pb2.DESCRIPTOR.services_by_name["LithoInferenceService"].full_name,
                    reflection.SERVICE_NAME,
                )
                reflection.enable_server_reflection(service_names, self._server)
                logger.info("已启用 gRPC Reflection")
            except ImportError:
                logger.info("grpcio-reflection 未安装，跳过 Reflection")

        bind_addr = f"{self._grpc_cfg.host}:{self._grpc_cfg.port}"

        if self._grpc_cfg.ssl_enabled:
            server_credentials = self._build_ssl_credentials()
            self._server.add_secure_port(bind_addr, server_credentials)
            logger.info(f"gRPC 安全服务器启动: {bind_addr} (SSL)")
        else:
            self._server.add_insecure_port(bind_addr)
            logger.info(f"gRPC 服务器启动: {bind_addr}")

        self._server.start()

    def _build_ssl_credentials(self):
        """构建 SSL 凭证"""
        cfg = self._grpc_cfg
        with open(cfg.ssl_cert_path, "rb") as f:
            cert_chain = f.read()
        with open(cfg.ssl_key_path, "rb") as f:
            private_key = f.read()

        root_certs = None
        if cfg.ssl_ca_path:
            with open(cfg.ssl_ca_path, "rb") as f:
                root_certs = f.read()

        return grpc.ssl_server_credentials(
            [(private_key, cert_chain)],
            root_certificates=root_certs,
            require_client_auth=root_certs is not None,
        )

    def wait_for_termination(self, timeout: Optional[float] = None):
        """等待服务器终止"""
        if self._server:
            self._server.wait_for_termination(timeout)

    def stop(self, grace: float = 5.0):
        """优雅关闭服务器"""
        logger.info(f"正在关闭 gRPC 服务器 (grace={grace}s)...")
        if self._server:
            self._server.stop(grace)
        self._stopped.set()
        logger.info("gRPC 服务器已关闭")
