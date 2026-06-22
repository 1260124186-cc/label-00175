# -*- coding: utf-8 -*-
"""
HTTP/REST 服务端 (FastAPI)

与 gRPC 服务功能对齐，同时提供 Swagger/OpenAPI 文档接口，
方便前端、测试工具、OPC 引擎的 REST 调用。
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .config import ServiceConfig, HttpConfig
from .schemas import (
    AerialImageRequest,
    AerialImageResponse,
    EpeEstimateRequest,
    EpeEstimateResponse,
    EpeMetrics,
    BatchInferenceRequest,
    BatchInferenceResponse,
    HealthStatus,
    ServiceInfo,
    PerformanceMetrics,
)
from .inference_engine import BaseInferenceEngine, EngineFactory
from .epe_estimator import EpeEstimator, EpeResult

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Header, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class HttpServer:
    """HTTP 服务器封装 (FastAPI)"""

    def __init__(
        self,
        config: ServiceConfig,
        engine: BaseInferenceEngine,
        epe_estimator: Optional[EpeEstimator] = None,
    ):
        if not HAS_FASTAPI:
            raise ImportError("FastAPI 未安装，请先安装 fastapi uvicorn")

        self._config = config
        self._http_cfg: HttpConfig = config.http
        self._engine = engine
        self._epe = epe_estimator or EpeEstimator()
        self._start_time = time.time()
        self._app: Optional[FastAPI] = None
        self._lock = threading.RLock()

        self._build_app()

    @property
    def app(self) -> FastAPI:
        return self._app

    # ------------------------------------------------------------------
    # 构建 FastAPI 应用
    # ------------------------------------------------------------------

    def _build_app(self):
        cfg = self._http_cfg
        from . import __version__

        self._app = FastAPI(
            title=self._config.service_name,
            description="光刻产线推理微服务 - 空间像推理与 EPE 估计",
            version=__version__,
            docs_url=cfg.docs_url if cfg.enable_docs else None,
            redoc_url=cfg.redoc_url if cfg.enable_docs else None,
        )

        if cfg.enable_cors:
            self._app.add_middleware(
                CORSMiddleware,
                allow_origins=cfg.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        self._register_routes()
        logger.info(f"HTTP 应用构建完成: docs={cfg.docs_url}")

    # ------------------------------------------------------------------
    # 路由注册
    # ------------------------------------------------------------------

    def _register_routes(self):
        app = self._app
        engine = self._engine
        epe = self._epe
        start_time = self._start_time
        cfg = self._config
        http_cfg = self._http_cfg

        def _check_api_key(x_api_key: Optional[str] = Header(default=None)):
            if http_cfg.api_key and x_api_key != http_cfg.api_key:
                raise HTTPException(status_code=401, detail="无效的 API Key")
            return True

        def _generate_request_id() -> str:
            return f"http-{uuid.uuid4().hex[:12]}"

        def _optical_params_dict(opt) -> Dict[str, Any]:
            if opt is None:
                return {}
            d = {
                "wavelength_nm": opt.wavelength_nm,
                "na": opt.na,
                "sigma": opt.sigma,
                "defocus_nm": opt.defocus_nm,
                "pixel_size_nm": opt.pixel_size_nm,
                "illumination_type": opt.illumination_type,
            }
            if opt.annular_sigma_inner is not None:
                d["annular_sigma_inner"] = opt.annular_sigma_inner
            if opt.annular_sigma_outer is not None:
                d["annular_sigma_outer"] = opt.annular_sigma_outer
            if opt.dipole_angle_deg is not None:
                d["dipole_angle_deg"] = opt.dipole_angle_deg
            return d

        # ---- 根路由 -------------------------------------------------

        @app.get("/", tags=["Root"])
        async def root():
            return {
                "name": cfg.service_name,
                "version": __version__,
                "environment": cfg.environment,
                "endpoints": {
                    "health": "/health",
                    "info": "/info",
                    "metrics": "/metrics",
                    "predict_aerial": "/api/v1/predict/aerial",
                    "estimate_epe": "/api/v1/estimate/epe",
                    "batch": "/api/v1/batch",
                    "reload": "/api/v1/reload",
                    "docs": http_cfg.docs_url,
                },
            }

        # ---- 健康检查 -----------------------------------------------

        @app.get("/health", response_model=HealthStatus, tags=["Status"])
        async def health(detailed: bool = Query(False)):
            engine_stats = engine.stats
            engine_ready = engine.is_ready
            status = "healthy" if engine_ready else "unhealthy"
            checks = {
                "engine_ready": str(engine_ready),
                "engine_type": engine.engine_type.value,
                "total_masks": str(engine_stats.get("total_masks", 0)),
            }
            if detailed:
                checks["stats"] = str(engine_stats)
            return HealthStatus(
                status=status,
                engine_status="ready" if engine_ready else "not_ready",
                grpc_enabled=cfg.grpc.enabled,
                http_enabled=True,
                uptime_seconds=time.time() - start_time,
                service_name=cfg.service_name,
                service_id=cfg.service_id,
                checks=checks,
            )

        # ---- 服务信息 -----------------------------------------------

        @app.get("/info", response_model=ServiceInfo, tags=["Status"])
        async def get_info():
            engine_stats = engine.stats
            return ServiceInfo(
                name=cfg.service_name,
                version=__version__,
                environment=cfg.environment,
                engine={
                    "type": engine.engine_type.value,
                    "initialized": engine.is_ready,
                    "total_calls": engine_stats.get("total_calls", 0),
                    "total_masks": engine_stats.get("total_masks", 0),
                    "errors": engine_stats.get("errors", 0),
                },
                endpoints={
                    "grpc": f"{cfg.grpc.host}:{cfg.grpc.port}",
                    "http": f"http://{http_cfg.host}:{http_cfg.port}",
                },
                config_summary={
                    "engine": cfg.engine.engine_type,
                    "grpc_port": cfg.grpc.port,
                    "http_port": http_cfg.port,
                },
            )

        # ---- 性能指标 -----------------------------------------------

        @app.get("/metrics", response_model=PerformanceMetrics, tags=["Status"])
        async def get_metrics(authenticated: bool = Depends(_check_api_key)):
            stats = engine.stats
            return PerformanceMetrics(
                total_requests=stats.get("total_calls", 0),
                total_masks_processed=stats.get("total_masks", 0),
                total_errors=stats.get("errors", 0),
                avg_latency_ms=stats.get("avg_latency_ms_per_mask", 0),
                p50_latency_ms=stats.get("p50_latency_ms", 0),
                p95_latency_ms=stats.get("p95_latency_ms", 0),
                p99_latency_ms=stats.get("p99_latency_ms", 0),
                throughput_masks_per_sec=stats.get("throughput_masks_per_sec", 0),
            )

        # ---- 热重载 -------------------------------------------------

        @app.post("/api/v1/reload", tags=["Management"])
        async def reload_model(authenticated: bool = Depends(_check_api_key)):
            t0 = time.time()
            try:
                with self._lock:
                    engine.reload({})
                return {
                    "success": True,
                    "message": "模型重载成功",
                    "reload_time_ms": (time.time() - t0) * 1000,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"重载失败: {e}")

        # ---- 空间像推理 ---------------------------------------------

        @app.post("/api/v1/predict/aerial", response_model=AerialImageResponse, tags=["Inference"])
        async def predict_aerial(
            request: AerialImageRequest,
            authenticated: bool = Depends(_check_api_key),
        ):
            req_id = _generate_request_id()
            try:
                masks_np = request.to_numpy()
                opt_params = _optical_params_dict(request.optical_params)

                t0 = time.time()
                aerials = engine.predict(masks_np, optical_params=opt_params)
                latency_ms = (time.time() - t0) * 1000

                return AerialImageResponse(
                    aerial_images=aerials.tolist() if aerials.ndim == 3
                        else [aerials.tolist()],
                    masks=request.masks if request.return_input_mask else None,
                    inference_mode_used=request.inference_mode.value,
                    engine_type=engine.engine_type.value,
                    latency_ms_total=latency_ms,
                    latency_ms_per_mask_avg=latency_ms / max(len(request.masks), 1),
                    num_masks=len(request.masks),
                    request_id=req_id,
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"predict_aerial 失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ---- EPE 估计 -----------------------------------------------

        @app.post("/api/v1/estimate/epe", response_model=EpeEstimateResponse, tags=["Inference"])
        async def estimate_epe(
            request: EpeEstimateRequest,
            authenticated: bool = Depends(_check_api_key),
        ):
            req_id = _generate_request_id()
            try:
                masks_np = np.array(request.masks, dtype=np.float32)
                targets_np = np.array(request.targets, dtype=np.float32)
                opt_params = _optical_params_dict(request.optical_params)

                t0 = time.time()
                aerials = engine.predict(masks_np, optical_params=opt_params)
                aerial_latency_ms = (time.time() - t0) * 1000

                ps_nm = request.pixel_size_nm
                th = request.threshold

                epe_list = []
                aerials_3d = aerials if aerials.ndim == 3 else aerials[np.newaxis, ...]
                for i in range(len(request.masks)):
                    result = epe.compute(
                        aerials_3d[i], targets_np[i],
                        pixel_size_nm=ps_nm, threshold=th,
                    )
                    epe_list.append(EpeMetrics(
                        epe_mean_nm=result.epe_mean_nm,
                        epe_max_nm=result.epe_max_nm,
                        epe_std_nm=result.epe_std_nm,
                        epe_median_nm=result.epe_median_nm,
                    ))

                return EpeEstimateResponse(
                    epe_results=epe_list,
                    aerial_images=aerial_latency_ms >= 0 and aerials_3d.tolist(),
                    inference_mode_used=request.inference_mode.value,
                    engine_type=engine.engine_type.value,
                    latency_ms_total=(time.time() - t0) * 1000,
                    latency_ms_per_mask_avg=(time.time() - t0) * 1000 / max(len(request.masks), 1),
                    num_masks=len(request.masks),
                    request_id=req_id,
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"estimate_epe 失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ---- 批量混合推理 -------------------------------------------

        @app.post("/api/v1/batch", response_model=BatchInferenceResponse, tags=["Inference"])
        async def batch_inference(
            request: BatchInferenceRequest,
            authenticated: bool = Depends(_check_api_key),
        ):
            req_id = _generate_request_id()
            try:
                masks_np = np.array(request.masks, dtype=np.float32)
                targets_np = np.array(request.targets, dtype=np.float32) if request.targets else None
                opt_params = _optical_params_dict(request.optical_params)
                output_type = request.output_type.value

                batch_size = min(request.max_batch_size, 256)
                N = len(request.masks)
                num_batches = (N + batch_size - 1) // batch_size

                all_aerials: List[np.ndarray] = []
                all_epes: List[EpeMetrics] = []

                t_total = time.time()
                for b in range(num_batches):
                    start = b * batch_size
                    end = min(start + batch_size, N)
                    chunk = masks_np[start:end]
                    aerials = engine.predict(chunk, optical_params=opt_params)
                    for i in range(len(aerials)):
                        all_aerials.append(aerials[i])

                    if output_type in ("epe", "both") and targets_np is not None:
                        ps_nm = request.pixel_size_nm
                        th = request.threshold
                        for i in range(len(aerials)):
                            idx = start + i
                            r = epe.compute(
                                aerials[i], targets_np[idx],
                                pixel_size_nm=ps_nm, threshold=th,
                            )
                            all_epes.append(EpeMetrics(
                                epe_mean_nm=r.epe_mean_nm,
                                epe_max_nm=r.epe_max_nm,
                                epe_std_nm=r.epe_std_nm,
                                epe_median_nm=r.epe_median_nm,
                            ))

                total_latency = (time.time() - t_total) * 1000

                result = BatchInferenceResponse(
                    inference_mode_used=request.inference_mode.value,
                    engine_type=engine.engine_type.value,
                    latency_ms_total=total_latency,
                    latency_ms_per_mask_avg=total_latency / max(N, 1),
                    num_masks=N,
                    num_batches=num_batches,
                    request_id=req_id,
                )

                if output_type in ("aerial_image", "both"):
                    stacked = np.stack(all_aerials, axis=0)
                    result.aerial_images = stacked.tolist()
                if output_type in ("epe", "both"):
                    result.epe_results = all_epes

                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"batch_inference 失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # 启动/关闭
    # ------------------------------------------------------------------

    def run(self):
        """使用 uvicorn 启动 HTTP 服务（阻塞）"""
        import uvicorn
        cfg = self._http_cfg
        logger.info(f"启动 HTTP 服务: {cfg.host}:{cfg.port}")
        uvicorn.run(
            self._app,
            host=cfg.host,
            port=cfg.port,
            workers=cfg.workers,
            log_level=cfg.log_level.lower(),
        )
