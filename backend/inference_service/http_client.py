# -*- coding: utf-8 -*-
"""
HTTP 客户端 SDK

提供 Python 侧对推理微服务的 REST API 调用接口，
与 GrpcClient 功能对齐。
"""

from __future__ import annotations

import os
import sys
import io
import time
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class HttpClientOptions:
    """HTTP 客户端配置"""
    base_url: str = "http://localhost:8080"
    timeout_sec: float = 30.0
    api_key: Optional[str] = None
    enable_retry: bool = True
    max_retries: int = 3
    retry_backoff_sec: float = 0.5


class InferenceHttpClient:
    """
    推理服务 HTTP/REST 客户端

    用法:
        client = InferenceHttpClient("http://localhost:8080")
        aerial = client.predict_aerial(mask_array)
        epe = client.estimate_epe(mask_array, target_array)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        options: Optional[HttpClientOptions] = None,
    ):
        if not HAS_REQUESTS:
            raise ImportError("requests 未安装: pip install requests")
        self._opts = options or HttpClientOptions(base_url=base_url, api_key=api_key)
        self._session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        if self._opts.api_key:
            session.headers["X-API-Key"] = self._opts.api_key
        return session

    def close(self):
        if self._session:
            self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        last_error = None
        url = f"{self._opts.base_url.rstrip('/')}/{path.lstrip('/')}"
        for attempt in range(self._opts.max_retries if self._opts.enable_retry else 1):
            try:
                if "timeout" not in kwargs:
                    kwargs["timeout"] = self._opts.timeout_sec
                response = self._session.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as e:
                last_error = e
                if e.response is not None and 400 <= e.response.status_code < 500:
                    try:
                        detail = e.response.json()
                    except Exception:
                        detail = e.response.text
                    raise RuntimeError(f"HTTP {e.response.status_code}: {detail}")
                if attempt < self._opts.max_retries - 1 and self._opts.enable_retry:
                    backoff = self._opts.retry_backoff_sec * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"HTTP 请求失败: {e}")
            except requests.RequestException as e:
                last_error = e
                if attempt < self._opts.max_retries - 1 and self._opts.enable_retry:
                    backoff = self._opts.retry_backoff_sec * (2 ** attempt)
                    time.sleep(backoff)
                    continue
        raise RuntimeError(f"HTTP 请求最终失败: {last_error}")

    @staticmethod
    def _prepare_masks(masks) -> Tuple[List[List[List[float]]], bool]:
        """将输入掩模标准化为 JSON 列表形式"""
        was_single = False
        if isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                mask_list = [masks.astype(np.float32).tolist()]
                was_single = True
            elif masks.ndim == 3:
                mask_list = [masks[i].astype(np.float32).tolist() for i in range(masks.shape[0])]
            else:
                raise ValueError(f"不支持的 ndarray 维度: {masks.ndim}")
        elif isinstance(masks, list):
            if len(masks) > 0 and isinstance(masks[0], np.ndarray):
                mask_list = [m.astype(np.float32).tolist() for m in masks]
            else:
                mask_list = masks
        else:
            raise ValueError(f"不支持的 masks 类型: {type(masks)}")
        return mask_list, was_single

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
            masks: 单张 (H, W) 或批量 (N, H, W) 或列表
            inference_mode: auto/surrogate/hopkins_lite
            optical_params: 光学参数字典
            threshold: 光刻胶阈值
            return_input_mask: 是否返回输入

        Returns:
            单张返回 (H, W)，批量返回 (N, H, W)
        """
        mask_list, was_single = self._prepare_masks(masks)

        payload: Dict[str, Any] = {
            "masks": mask_list,
            "inference_mode": inference_mode,
            "threshold": threshold,
            "return_input_mask": return_input_mask,
        }
        if optical_params:
            payload["optical_params"] = optical_params

        response = self._request_with_retry(
            "POST",
            "/api/v1/predict/aerial",
            json=payload,
        )

        aerials = [np.array(a, dtype=np.float32) for a in response["aerial_images"]]
        if was_single:
            return aerials[0]
        return np.stack(aerials, axis=0)

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
            return_aerial: 是否返回空间像

        Returns:
            EPE 结果字典 (或列表)，如果 return_aerial=True 则返回 (epe, aerials)
        """
        mask_list, was_single = self._prepare_masks(masks)
        target_list, _ = self._prepare_masks(targets)

        payload: Dict[str, Any] = {
            "masks": mask_list,
            "targets": target_list,
            "inference_mode": inference_mode,
            "threshold": threshold,
            "pixel_size_nm": pixel_size_nm,
        }
        if optical_params:
            payload["optical_params"] = optical_params

        response = self._request_with_retry(
            "POST",
            "/api/v1/estimate/epe",
            json=payload,
        )

        epe_list = [
            {
                "epe_mean_nm": e["epe_mean_nm"],
                "epe_max_nm": e["epe_max_nm"],
                "epe_std_nm": e["epe_std_nm"],
                "epe_median_nm": e["epe_median_nm"],
            }
            for e in response["epe_results"]
        ]

        epe_result = epe_list[0] if was_single else epe_list

        if return_aerial:
            aerials = [np.array(a, dtype=np.float32) for a in response["aerial_images"]]
            aerial_result = aerials[0] if was_single else np.stack(aerials, axis=0)
            return epe_result, aerial_result
        return epe_result

    # ------------------------------------------------------------------
    # 状态 / 监控
    # ------------------------------------------------------------------

    def health(self, detailed: bool = False) -> Dict[str, Any]:
        return self._request_with_retry("GET", "/health", params={"detailed": detailed})

    def info(self) -> Dict[str, Any]:
        return self._request_with_retry("GET", "/info")

    def metrics(self) -> Dict[str, Any]:
        return self._request_with_retry("GET", "/metrics")

    def reload(self) -> Dict[str, Any]:
        return self._request_with_retry("POST", "/api/v1/reload")
