# -*- coding: utf-8 -*-
"""
EPE (Edge Placement Error) 快速估计模块

提供从空间像 + 目标图直接计算 EPE 的算法，
与主研究框架解耦。
支持:
1. 精确 EPE 计算 (距离变换方法)
2. 快速 EPE 估计 (基于空间像梯度近似)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EpeMethod(str, Enum):
    """EPE 计算方法"""
    EXACT_DISTANCE = "exact_distance"
    FAST_GRADIENT = "fast_gradient"
    HYBRID = "hybrid"


@dataclass
class EpeResult:
    """EPE 计算结果"""
    epe_mean_nm: float = 0.0
    epe_max_nm: float = 0.0
    epe_std_nm: float = 0.0
    epe_median_nm: float = 0.0
    method: str = "exact_distance"
    pixel_size_nm: float = 1.0
    num_edge_pixels_wafer: int = 0
    num_edge_pixels_target: int = 0
    _raw_distances: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epe_mean_nm": self.epe_mean_nm,
            "epe_max_nm": self.epe_max_nm,
            "epe_std_nm": self.epe_std_nm,
            "epe_median_nm": self.epe_median_nm,
            "method": self.method,
            "pixel_size_nm": self.pixel_size_nm,
            "num_edge_pixels_wafer": self.num_edge_pixels_wafer,
            "num_edge_pixels_target": self.num_edge_pixels_target,
        }


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _binarize(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """图像二值化"""
    return (image >= threshold).astype(np.float64)


def _sobel_edges(binary: np.ndarray) -> np.ndarray:
    """Sobel 边缘提取"""
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)

    pad = np.pad(binary, 1, mode="symmetric")
    gy = _conv2d_valid(pad, sobel_y)
    gx = _conv2d_valid(pad, sobel_x)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return (grad >= 0.5).astype(np.float64)


def _morph_edges(binary: np.ndarray) -> np.ndarray:
    """形态学边缘提取 (原图 - 腐蚀)"""
    struct = np.ones((3, 3), dtype=bool)
    img_bool = binary >= 0.5
    eroded = _binary_erosion(img_bool, struct)
    edges = img_bool & ~eroded
    return edges.astype(np.float64)


def _conv2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """简单 2D 卷积 (valid 模式)"""
    kh, kw = kernel.shape
    ih, iw = image.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * image[i:i + oh, j:j + ow]
    return out


def _binary_erosion(img: np.ndarray, struct: np.ndarray) -> np.ndarray:
    """二值腐蚀"""
    kh, kw = struct.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(
        img,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=False,
    )
    ih, iw = img.shape
    result = np.ones_like(img, dtype=bool)
    for i in range(kh):
        for j in range(kw):
            if struct[i, j]:
                result &= padded[i:i + ih, j:j + iw]
    return result


def _distance_transform(binary: np.ndarray) -> np.ndarray:
    """Manhattan 距离变换近似"""
    h, w = binary.shape
    dist = np.where(binary > 0.5, 0.0, np.inf)

    MAX_D = float(h + w)

    for i in range(h):
        for j in range(w):
            if not np.isinf(dist[i, j]):
                if i > 0:
                    dist[i, j] = min(dist[i, j], dist[i - 1, j] + 1)
                if j > 0:
                    dist[i, j] = min(dist[i, j], dist[i, j - 1] + 1)
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            if i < h - 1:
                dist[i, j] = min(dist[i, j], dist[i + 1, j] + 1)
            if j < w - 1:
                dist[i, j] = min(dist[i, j], dist[i, j + 1] + 1)
    return np.clip(dist, 0, MAX_D)


# ---------------------------------------------------------------------------
# EPE 估计器
# ---------------------------------------------------------------------------

class EpeEstimator:
    """精确 EPE 计算 (距离变换方法)"""

    def __init__(
        self,
        method: EpeMethod = EpeMethod.EXACT_DISTANCE,
        pixel_size_nm: float = 1.0,
        threshold: float = 0.5,
        edge_method: str = "morphological",
    ):
        self.method = method
        self.pixel_size_nm = pixel_size_nm
        self.threshold = threshold
        self.edge_method = edge_method

    def compute(
        self,
        wafer_image: np.ndarray,
        target_binary: np.ndarray,
        pixel_size_nm: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> EpeResult:
        """
        计算 EPE

        Args:
            wafer_image: 空间像 (H, W) 或二值化晶圆图
            target_binary: 目标二值图 (H, W)
            pixel_size_nm: 像素尺寸 (nm)
            threshold: 二值化阈值

        Returns:
            EpeResult
        """
        ps = pixel_size_nm or self.pixel_size_nm
        th = threshold or self.threshold

        if wafer_image.dtype == bool:
            wafer_bin = wafer_image.astype(np.float64)
        else:
            wafer_bin = _binarize(wafer_image, th)

        if target_binary.dtype == bool:
            target_bin = target_binary.astype(np.float64)
        else:
            target_bin = _binarize(target_binary, 0.5)

        if self.edge_method == "sobel":
            wafer_edge = _sobel_edges(wafer_bin)
            target_edge = _sobel_edges(target_bin)
        else:
            wafer_edge = _morph_edges(wafer_bin)
            target_edge = _morph_edges(target_bin)

        wafer_edge_count = int(np.sum(wafer_edge))
        target_edge_count = int(np.sum(target_edge))

        if wafer_edge_count == 0 and target_edge_count == 0:
            return EpeResult(
                method=self.method.value,
                pixel_size_nm=ps,
                num_edge_pixels_wafer=0,
                num_edge_pixels_target=0,
            )

        if wafer_edge_count == 0 or target_edge_count == 0:
            return EpeResult(
                epe_mean_nm=float("inf"),
                epe_max_nm=float("inf"),
                epe_std_nm=0.0,
                epe_median_nm=float("inf"),
                method=self.method.value,
                pixel_size_nm=ps,
                num_edge_pixels_wafer=wafer_edge_count,
                num_edge_pixels_target=target_edge_count,
            )

        distances_wafer_to_target = self._distance_map(target_edge)
        distances_target_to_wafer = self._distance_map(wafer_edge)

        wafer_dists = distances_wafer_to_target[wafer_edge > 0.5]
        target_dists = distances_target_to_wafer[target_edge > 0.5]

        all_dists = np.concatenate([wafer_dists, target_dists])
        all_dists_nm = all_dists * ps

        return EpeResult(
            epe_mean_nm=float(np.mean(all_dists_nm)) if len(all_dists_nm) > 0 else 0.0,
            epe_max_nm=float(np.max(all_dists_nm)) if len(all_dists_nm) > 0 else 0.0,
            epe_std_nm=float(np.std(all_dists_nm)) if len(all_dists_nm) > 0 else 0.0,
            epe_median_nm=float(np.median(all_dists_nm)) if len(all_dists_nm) > 0 else 0.0,
            method=self.method.value,
            pixel_size_nm=ps,
            num_edge_pixels_wafer=wafer_edge_count,
            num_edge_pixels_target=target_edge_count,
        )

    @staticmethod
    def _distance_map(edge_map: np.ndarray) -> np.ndarray:
        """计算到最近边缘的距离图"""
        inverted = 1.0 - (edge_map > 0.5).astype(np.float64)
        return _distance_transform(inverted)


# ---------------------------------------------------------------------------
# 快速 EPE 估计器（基于梯度近似）
# ---------------------------------------------------------------------------

class FastEpeEstimator:
    """
    快速 EPE 估计 (基于空间像梯度近似)

    不做二值化和距离变换，
    直接从空间像梯度在阈值附近估计 EPE，
    速度提升 5-10x。
    """

    def __init__(
        self,
        pixel_size_nm: float = 1.0,
        threshold: float = 0.5,
    ):
        self.pixel_size_nm = pixel_size_nm
        self.threshold = threshold
        self._gauss_kernel = self._make_gauss(5, 1.0)

    def compute(
        self,
        aerial: np.ndarray,
        target: np.ndarray,
        pixel_size_nm: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> EpeResult:
        ps = pixel_size_nm or self.pixel_size_nm
        th = threshold or self.threshold

        if aerial.ndim != 2:
            raise ValueError(f"期望 2D 数组，得到 {aerial.ndim}D")

        padded = np.pad(aerial, 2, mode="symmetric")
        smooth = _conv2d_valid(padded, self._gauss_kernel)

        H, W = smooth.shape
        grad_y = np.zeros_like(smooth, dtype=np.float64)
        grad_x = np.zeros_like(smooth, dtype=np.float64)
        grad_y[1:H - 1, :] = smooth[2:, :] - smooth[:H - 2, :]
        grad_x[:, 1:W - 1] = smooth[:, 2:] - smooth[:, :W - 2]
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8

        wafer_edges = np.abs(smooth - th) < 0.05
        target_edges = _morph_edges(_binarize(target, 0.5))

        has_wafer = np.any(wafer_edges)
        has_target = np.any(target_edges)

        if not has_wafer or not has_target:
            return EpeResult(
                method="fast_gradient",
                pixel_size_nm=ps,
                num_edge_pixels_wafer=int(np.sum(wafer_edges)),
                num_edge_pixels_target=int(np.sum(target_edges)),
            )

        estimated_errors = np.abs(smooth - th) / grad_mag
        epe_approx = estimated_errors[wafer_edges] * ps

        return EpeResult(
            epe_mean_nm=float(np.mean(epe_approx)),
            epe_max_nm=float(np.max(epe_approx)),
            epe_std_nm=float(np.std(epe_approx)),
            epe_median_nm=float(np.median(epe_approx)),
            method="fast_gradient",
            pixel_size_nm=ps,
            num_edge_pixels_wafer=int(np.sum(wafer_edges)),
            num_edge_pixels_target=int(np.sum(target_edges)),
        )

    @staticmethod
    def _make_gauss(size: int, sigma: float) -> np.ndarray:
        """构造高斯核"""
        ax = np.arange(size) - size // 2
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()
