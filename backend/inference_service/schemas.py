# -*- coding: utf-8 -*-
"""
数据结构定义模块

使用 Pydantic v2 定义请求/响应数据结构，
同时提供与 NumPy 数组的双向转换。
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict


class InferenceMode(str, Enum):
    """推理模式枚举"""
    SURROGATE = "surrogate"
    HOPKINS_LITE = "hopkins_lite"
    AUTO = "auto"


class OutputType(str, Enum):
    """输出类型枚举"""
    AERIAL_IMAGE = "aerial_image"
    EPE = "epe"
    BOTH = "both"


class EngineStatus(str, Enum):
    """引擎状态枚举"""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    RELOADING = "reloading"


# ---------------------------------------------------------------------------
# Pydantic 模型（用于 HTTP API 序列化
# ---------------------------------------------------------------------------

class OpticalParams(BaseModel):
    """光学系统参数（用于 Hopkins 近似）"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    wavelength_nm: float = Field(193.0, description="光源波长 (nm)", ge=10.0, le=1000.0)
    na: float = Field(1.35, description="数值孔径", ge=0.1, le=2.0)
    sigma: float = Field(0.75, description="部分相干因子", ge=0.1, le=1.0)
    defocus_nm: float = Field(0.0, description="离焦量 (nm)")
    pixel_size_nm: float = Field(1.0, description="像素尺寸 (nm)", gt=0.0)
    illumination_type: str = Field("conventional", description="照明模式: conventional/annular/dipole/quasar")
    annular_sigma_inner: Optional[float] = Field(None, description="环形照明内半径")
    annular_sigma_outer: Optional[float] = Field(None, description="环形照明外半径")
    dipole_angle_deg: Optional[float] = Field(None, description="偶极照明角度 (度)")


class AerialImageRequest(BaseModel):
    """空间像推理请求"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    masks: List[List[List[float]]] = Field(
        ...,
        description="掩模数组列表，每个为 (H, W) 二维数组，值域 [0, 1]",
        min_length=1,
    )
    inference_mode: InferenceMode = Field(
        InferenceMode.AUTO,
        description="推理模式选择"
    )
    optical_params: Optional[OpticalParams] = Field(
        None,
        description="光学系统参数（Hopkins 模式必需）"
    )
    threshold: Optional[float] = Field(
        0.5,
        description="光刻胶阈值，用于二值化（EPE 估计时使用）"
    )
    return_input_mask: bool = Field(
        False,
        description="是否在响应中返回原始掩模"
    )

    @field_validator("masks")
    @classmethod
    def validate_masks(cls, v):
        if not v:
            raise ValueError("masks 列表不能为空")
        h, w = len(v[0]), len(v[0][0]) if v[0] else 0
        for i, m in enumerate(v):
            if len(m) != h or any(len(row) != w for row in m):
                raise ValueError(f"第 {i} 个掩模形状不一致，期望 ({h}, {w})")
        return v

    def to_numpy(self) -> np.ndarray:
        return np.array(self.masks, dtype=np.float32)


class AerialImageResponse(BaseModel):
    """空间像推理响应"""
    aerial_images: List[List[List[float]]] = Field(
        ...,
        description="空间像数组列表，与输入掩模一一对应"
    )
    masks: Optional[List[List[List[float]]]] = Field(
        None,
        description="原始掩模（如请求时返回）"
    )
    inference_mode_used: str = Field(..., description="实际使用的推理模式")
    engine_type: str = Field(..., description="使用的引擎类型")
    latency_ms_total: float = Field(..., description="总耗时 (ms)")
    latency_ms_per_mask_avg: float = Field(..., description="单掩模平均耗时 (ms)")
    num_masks: int = Field(..., description="处理的掩模数量")
    request_id: str = Field(..., description="请求唯一标识")


class EpeEstimateRequest(BaseModel):
    """EPE 估计请求"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    masks: List[List[List[float]]] = Field(
        ...,
        description="掩模数组列表，每个为 (H, W)",
        min_length=1,
    )
    targets: List[List[List[float]]] = Field(
        ...,
        description="目标图案数组列表，与 masks 一一对应",
        min_length=1,
    )
    inference_mode: InferenceMode = Field(InferenceMode.AUTO)
    optical_params: Optional[OpticalParams] = Field(None)
    threshold: float = Field(0.5, description="光刻胶二值化阈值")
    pixel_size_nm: float = Field(1.0, description="像素尺寸 (nm)", gt=0.0)
    edge_method: str = Field("morphological", description="边缘提取方法")

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, v, info):
        masks = info.data.get("masks", [])
        if len(v) != len(masks):
            raise ValueError(f"targets 数量 ({len(v)}) 与 masks 数量 ({len(masks)}) 不一致")
        return v


class EpeMetrics(BaseModel):
    """单个掩模的 EPE 指标"""
    epe_mean_nm: float = Field(..., description="平均 EPE (nm)")
    epe_max_nm: float = Field(..., description="最大 EPE (nm)")
    epe_std_nm: float = Field(..., description="EPE 标准差 (nm)")
    epe_median_nm: float = Field(..., description="EPE 中位数 (nm)")


class EpeEstimateResponse(BaseModel):
    """EPE 估计响应"""
    epe_results: List[EpeMetrics] = Field(..., description="每个掩模的 EPE 结果")
    aerial_images: Optional[List[List[List[float]]]] = Field(
        None,
        description="空间像数组（可选返回"
    )
    inference_mode_used: str = Field(...)
    engine_type: str = Field(...)
    latency_ms_total: float = Field(...)
    latency_ms_per_mask_avg: float = Field(...)
    num_masks: int = Field(...)
    request_id: str = Field(...)


class BatchInferenceRequest(BaseModel):
    """批量混合推理请求（同时返回空间像和 EPE"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    masks: List[List[List[float]]] = Field(..., min_length=1)
    targets: Optional[List[List[List[float]]]] = Field(None)
    output_type: OutputType = Field(OutputType.AERIAL_IMAGE)
    inference_mode: InferenceMode = Field(InferenceMode.AUTO)
    optical_params: Optional[OpticalParams] = Field(None)
    threshold: float = Field(0.5)
    pixel_size_nm: float = Field(1.0, gt=0.0)
    max_batch_size: int = Field(32, ge=1, le=512)


class BatchInferenceResponse(BaseModel):
    """批量混合推理响应"""
    aerial_images: Optional[List[List[List[float]]]] = None
    epe_results: Optional[List[EpeMetrics]] = None
    inference_mode_used: str
    engine_type: str
    latency_ms_total: float
    latency_ms_per_mask_avg: float
    num_masks: int
    num_batches: int
    request_id: str


class HealthStatus(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态: healthy/degraded/unhealthy")
    engine_status: str = Field(..., description="引擎状态")
    grpc_enabled: bool
    http_enabled: bool
    uptime_seconds: float
    service_name: str
    service_id: str
    checks: Dict[str, Any] = Field(default_factory=dict)


class ServiceInfo(BaseModel):
    """服务信息响应"""
    name: str
    version: str
    environment: str
    engine: Dict[str, Any]
    endpoints: Dict[str, Any]
    config_summary: Dict[str, Any]


class PerformanceMetrics(BaseModel):
    """性能指标响应"""
    total_requests: int = 0
    total_masks_processed: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_masks_per_sec: float = 0.0
    recent_requests: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 内部数据类（用于服务内部，非序列化
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """内部推理结果包装"""
    aerial_images: Optional[np.ndarray] = None
    epe_results: Optional[List[Dict[str, float]]] = None
    inference_mode: str = "unknown"
    engine_type: str = "unknown"
    latency_ms: float = 0.0
    num_masks: int = 0
    error: Optional[str] = None

    def ok(self) -> bool:
        return self.error is None
