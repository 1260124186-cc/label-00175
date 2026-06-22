# -*- coding: utf-8 -*-
"""
gRPC 客户端 SDK

提供 Python 侧对推理微服务的 gRPC 调用接口，
与 HTTPClient 功能对齐，延迟更低。
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple, Union, Iterator
from dataclasses import dataclass, field
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import grpc
    from .proto import ensure_grpc, get_pb2, get_pb2_grpc
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False


@dataclass
class GrpcClientOptions:
    """gRPC 客户端配置"""
    target: str = "localhost:50051"
    timeout_ms: int = 30000
    max_message_size_mb: int = 64
    use_ssl: bool = False
    ssl_cert_path: Optional[str] = None
    api_key: Optional[str] = None
    enable_retry: bool = True
    max_retries: int = 3
    retry_backoff_ms: int = 1000
    keepalive_time_ms: int = 30000
    keepalive_timeout_ms: int = 10000


class InferenceGrpcClient:
    """
    推理服务 gRPC 客户端

    用法:
        client = InferenceGrpcClient("localhost:50051")
        aerial = client.predict_aerial(mask_array)
        epe = client.estimate_epe(mask_array, target_array)
    """

    def __init__(
        self,
        target: str = "localhost:50051",
        options: Optional[GrpcClientOptions] = None,
    ):
        if not HAS_GRPC:
            raise ImportError("grpcio 未安装")
        ensure_grpc()

        self._opts = options or GrpcClientOptions(target=target)
        self._channel = None
        self._stub = None
        self._connect()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def _connect(self):
        """建立 gRPC 连接"""
        opts = self._opts
        channel_opts = [
            ("grpc.max_send_message_length", opts.max_message_size_mb * 1024 * 1024),
            ("grpc.max_receive_message_length", opts.max_message_size_mb * 1024 * 1024),
            ("grpc.keepalive_time_ms", opts.keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", opts.keepalive_timeout_ms),
        ]

        if opts.use_ssl:
            creds = self._build_ssl_credentials()
            self._channel = grpc.secure_channel(opts.target, creds, options=channel_opts)
        else:
            self._channel = grpc.insecure_channel(opts.target, options=channel_opts)

        pb2_grpc = get_pb2_grpc()
        self._stub = pb2_grpc.LithoInferenceServiceStub(self._channel)
        logger.debug(f"gRPC 客户端已连接: {opts.target}")

    def _build_ssl_credentials(self):
        opts = self._opts
        root_certs = None
        if opts.ssl_cert_path and os.path.exists(opts.ssl_cert_path):
            with open(opts.ssl_cert_path, "rb") as f:
                root_certs = f.read()
        return grpc.ssl_channel_credentials(root_certificates=root_certs)

    def close(self):
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------

    @staticmethod
    def _numpy_to_array2d(arr: np.ndarray, pb2):
        h, w = arr.shape
        return pb2.Array2D(
            height=int(h),
            width=int(w),
            data=arr.astype(np.float32).flatten().tolist(),
        )

    @staticmethod
    def _array2d_to_numpy(arr) -> np.ndarray:
        h, w = arr.height, arr.width
        return np.array(arr.data, dtype=np.float32).reshape(h, w)

    @staticmethod
    def _mode_str_to_enum(mode: str, pb2) -> int:
        mapping = {
            "auto": 0, "surrogate": 1, "hopkins_lite": 2,
        }
        return mapping.get(mode.lower(), 0)

    def _check_stub(self):
        if self._stub is None:
            raise RuntimeError("未连接 gRPC 服务器")

    def _call_with_retry(self, method, *args, **kwargs):
        opts = self._opts
        last_error = None
        for attempt in range(opts.max_retries if opts.enable_retry else 1):
            try:
                return method(*args, **kwargs, timeout=opts.timeout_ms / 1000.0)
            except grpc.RpcError as e:
                last_error = e
                if attempt < opts.max_retries - 1 and opts.enable_retry:
                    backoff = opts.retry_backoff_ms * (2 ** attempt) / 1000.0
                    time.sleep(backoff)
                    continue
        raise last_error

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def predict_aerial(
        self,
        masks: Union[np.ndarray, List[np.ndarray]],
        inference_mode: str = "auto",
        optical_params: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
        return_input_mask: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        预测空间像

        Args:
            masks: 单张 (H, W) 或批量 (N, H, W) 或 List[(H, W)]
            inference_mode: auto/surrogate/hopkins_lite
            optical_params: 光学参数字典
            threshold: 光刻胶阈值
            return_input_mask: 是否返回输入

        Returns:
            单张返回 (H, W)，批量返回 (N, H, W)
        """
        self._check_stub()
        pb2 = get_pb2()

        was_single = False
        if isinstance(masks, np.ndarray) and masks.ndim == 2:
            mask_list = [masks]
            was_single = True
        elif isinstance(masks, np.ndarray) and masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
        elif isinstance(masks, list):
            mask_list = masks
        else:
            raise ValueError(f"不支持的 masks 类型: {type(masks)}, shape={getattr(masks, 'shape', None)}")

        grpc_masks = [self._numpy_to_array2d(m, pb2) for m in mask_list]

        request = pb2.PredictAerialRequest(
            masks=grpc_masks,
            inference_mode=self._mode_str_to_enum(inference_mode, pb2),
            threshold=threshold,
            return_input_mask=return_input_mask,
            request_id=f"cli-{uuid.uuid4().hex[:12]}",
        )

        if optical_params:
            op = pb2.OpticalParams(**{k: v for k, v in optical_params.items()
                                      if k in ("wavelength_nm", "na", "sigma", "defocus_nm",
                                               "pixel_size_nm", "illumination_type",
                                               "annular_sigma_inner", "annular_sigma_outer",
                                               "dipole_angle_deg")})
            request.optical_params.CopyFrom(op)

        response = self._call_with_retry(self._stub.PredictAerial, request)
        aerials = [self._array2d_to_numpy(a) for a in response.aerial_images]

        if was_single:
            return aerials[0]
        return np.stack(aerials, axis=0) if len(aerials) > 1 else aerials[0][np.newaxis, ...]

    def estimate_epe(
        self,
        masks: Union[np.ndarray, List[np.ndarray]],
        targets: Union[np.ndarray, List[np.ndarray]],
        inference_mode: str = "auto",
        optical_params: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
        pixel_size_nm: float = 1.0,
        return_aerial: bool = False,
    ) -> Union[Dict[str, float], List[Dict[str, float]], Tuple]:
        """
        估计 EPE

        Args:
            masks: 掩模
            targets: 目标图
            inference_mode: 推理模式
            optical_params: 光学参数
            threshold: 二值化阈值
            pixel_size_nm: 像素尺寸
            return_aerial: 是否同时返回空间像

        Returns:
            EPE 结果字典 (或列表)，若 return_aerial=True 则返回 (epe_result, aerials)
        """
        self._check_stub()
        pb2 = get_pb2()

        was_single = False
        if isinstance(masks, np.ndarray) and masks.ndim == 2:
            mask_list = [masks]
            target_list = [targets] if isinstance(targets, np.ndarray) and targets.ndim == 2 else [targets[0]]
            was_single = True
        elif isinstance(masks, np.ndarray) and masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
            if isinstance(targets, np.ndarray) and targets.ndim == 3:
                target_list = [targets[i] for i in range(targets.shape[0])]
            else:
                target_list = targets
        elif isinstance(masks, list):
            mask_list = masks
            target_list = targets if isinstance(targets, list) else [targets]
        else:
            raise ValueError("不支持的 masks 格式")

        grpc_masks = [self._numpy_to_array2d(m, pb2) for m in mask_list]
        grpc_targets = [self._numpy_to_array2d(t, pb2) for t in target_list]

        request = pb2.EstimateEpeRequest(
            masks=grpc_masks,
            targets=grpc_targets,
            inference_mode=self._mode_str_to_enum(inference_mode, pb2),
            threshold=threshold,
            pixel_size_nm=pixel_size_nm,
            return_aerial_images=return_aerial,
            request_id=f"cli-{uuid.uuid4().hex[:12]}",
        )

        if optical_params:
            op = pb2.OpticalParams(**{k: v for k, v in optical_params.items()
                                      if k in ("wavelength_nm", "na", "sigma", "defocus_nm",
                                               "pixel_size_nm", "illumination_type",
                                               "annular_sigma_inner", "annular_sigma_outer",
                                               "dipole_angle_deg")})
            request.optical_params.CopyFrom(op)

        response = self._call_with_retry(self._stub.EstimateEpe, request)

        epe_results = []
        for e in response.epe_results:
            epe_results.append({
                "epe_mean_nm": e.epe_mean_nm,
                "epe_max_nm": e.epe_max_nm,
                "epe_std_nm": e.epe_std_nm,
                "epe_median_nm": e.epe_median_nm,
            })

        result_epe = epe_results[0] if was_single else epe_results

        if return_aerial:
            aerials = [self._array2d_to_numpy(a) for a in response.aerial_images]
            if was_single:
                return result_epe, aerials[0]
            return result_epe, np.stack(aerials, axis=0)

        return result_epe

    # ------------------------------------------------------------------
    # 健康检查 / 监控
    # ------------------------------------------------------------------

    def health_check(self, detailed: bool = False) -> Dict[str, Any]:
        self._check_stub()
        pb2 = get_pb2()
        response = self._call_with_retry(
            self._stub.HealthCheck,
            pb2.HealthCheckRequest(detailed=detailed),
        )
        return {
            "status": ["unknown", "healthy", "degraded", "unhealthy"][response.status],
            "engine_status": response.engine_status,
            "uptime_seconds": response.uptime_seconds,
            "service_name": response.service_name,
            "service_id": response.service_id,
            "checks": dict(response.checks),
        }

    def get_metrics(self) -> Dict[str, Any]:
        self._check_stub()
        pb2 = get_pb2()
        r = self._call_with_retry(self._stub.GetMetrics, pb2.MetricsRequest())
        return {
            "total_requests": r.total_requests,
            "total_masks_processed": r.total_masks_processed,
            "total_errors": r.total_errors,
            "avg_latency_ms": r.avg_latency_ms,
            "p50_latency_ms": r.p50_latency_ms,
            "p95_latency_ms": r.p95_latency_ms,
            "p99_latency_ms": r.p99_latency_ms,
            "throughput_masks_per_sec": r.throughput_masks_per_sec,
        }

    def reload_model(self) -> Dict[str, Any]:
        self._check_stub()
        pb2 = get_pb2()
        r = self._call_with_retry(self._stub.ReloadModel, pb2.ReloadRequest())
        return {"success": r.success, "message": r.message, "reload_time_ms": r.reload_time_ms}
