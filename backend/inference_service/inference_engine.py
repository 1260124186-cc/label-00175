# -*- coding: utf-8 -*-
"""
推理引擎模块

提供两种推理引擎实现:
1. SurrogateEngine: 神经网络代理模型 (ONNX Runtime / TorchScript)
2. HopkinsLiteEngine: 轻量化 Hopkins 近似 (纯 NumPy，无训练依赖)

两种引擎实现统一接口，可无缝切换。
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import threading
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class EngineType(str, Enum):
    SURROGATE = "surrogate"
    HOPKINS_LITE = "hopkins_lite"


@dataclass
class EngineStats:
    """引擎统计信息"""
    total_calls: int = 0
    total_masks: int = 0
    total_time_ms: float = 0.0
    errors: int = 0
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=10000))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, num_masks: int, latency_ms: float, error: bool = False):
        with self._lock:
            self.total_calls += 1
            self.total_masks += num_masks
            self.total_time_ms += latency_ms
            if error:
                self.errors += 1
            else:
                self.recent_latencies.append(latency_ms / max(num_masks, 1))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            recent = sorted(list(self.recent_latencies))
            n = len(recent)
            return {
                "total_calls": self.total_calls,
                "total_masks": self.total_masks,
                "total_time_ms": self.total_time_ms,
                "errors": self.errors,
                "avg_latency_ms_per_mask": self.total_time_ms / max(self.total_masks, 1),
                "throughput_masks_per_sec": self.total_masks / max(self.total_time_ms / 1000.0, 1e-6),
                "p50_latency_ms": recent[int(n * 0.5)] if n > 0 else 0.0,
                "p95_latency_ms": recent[int(n * 0.95)] if n > 0 else 0.0,
                "p99_latency_ms": recent[int(n * 0.99)] if n > 0 else 0.0,
                "recent_window_size": n,
            }


class BaseInferenceEngine(ABC):
    """推理引擎抽象基类"""

    engine_type: EngineType = EngineType.SURROGATE

    def __init__(self):
        self._stats = EngineStats()
        self._initialized = False
        self._lock = threading.RLock()

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "engine_type": self.engine_type.value,
            "initialized": self._initialized,
            **self._stats.snapshot(),
        }

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化引擎，加载模型/核函数"""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """关闭引擎，释放资源"""
        ...

    def predict(
        self,
        masks: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        推理入口: 掩模 → 空间像

        Args:
            masks: 输入掩模，支持:
                - 单张: (H, W) float32/64
                - 批量: (N, H, W) float32/64
            **kwargs: 引擎特定参数

        Returns:
            空间像数组，与输入维度对应
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，先调用 initialize()")

        was_single = masks.ndim == 2
        if was_single:
            masks = masks[np.newaxis, ...]

        if masks.ndim != 3:
            raise ValueError(f"不支持的输入维度: {masks.ndim}，期望 2 或 3 维")

        t0 = time.time()
        try:
            with self._lock:
                result = self._predict_impl(masks, **kwargs)
            latency_ms = (time.time() - t0) * 1000
            self._stats.record(len(masks), latency_ms, error=False)
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            self._stats.record(len(masks), latency_ms, error=True)
            logger.error(f"推理失败: {e}")
            raise

        if was_single:
            return result[0]
        return result

    @abstractmethod
    def _predict_impl(self, masks: np.ndarray, **kwargs) -> np.ndarray:
        """子类实现的实际推理逻辑"""
        ...

    def reload(self, config: Optional[Dict[str, Any]] = None) -> None:
        """热重载引擎"""
        logger.info(f"重载引擎: {self.engine_type.value}")
        self.shutdown()
        self._initialized = False
        self._stats = EngineStats()
        self.initialize(config)


# ===========================================================================
# SurrogateEngine: 神经网络代理模型推理引擎
# ===========================================================================

class SurrogateEngine(BaseInferenceEngine):
    """
    神经网络代理模型推理引擎

    优先使用 ONNX Runtime (轻量高性能)，回退到 TorchScript。
    支持模型热重载、预热推理、批量推理优化。
    """

    engine_type = EngineType.SURROGATE

    def __init__(self):
        super().__init__()
        self._model_dir: Optional[str] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._session = None
        self._torchscript_model = None
        self._backend: Optional[str] = None
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._output_shape: Optional[Tuple[int, ...]] = None

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._model_dir = cfg.get("model_dir", "./models")
        prefer_onnx = cfg.get("prefer_onnx", True)
        device = cfg.get("device", "auto")

        logger.info(f"初始化 SurrogateEngine: model_dir={self._model_dir}")

        self._load_metadata()
        self._load_model(prefer_onnx, device)

        warmup_iters = cfg.get("warmup_iterations", 10)
        if warmup_iters > 0 and self._input_shape:
            self._warmup(warmup_iters)

        self._initialized = True
        logger.info(f"SurrogateEngine 就绪: backend={self._backend}, "
                    f"input_shape={self._input_shape}")

    def _load_metadata(self):
        if not self._model_dir:
            return
        metadata_path = os.path.join(self._model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                self._input_shape = tuple(self._metadata["model"]["input_shape"])
                self._output_shape = tuple(self._metadata["model"]["output_shape"])
                logger.info(f"加载模型元数据: {metadata_path}")
            except Exception as e:
                logger.warning(f"加载元数据失败: {e}")

    def _load_model(self, prefer_onnx: bool, device: str):
        if not self._model_dir:
            raise ValueError("model_dir 未配置")

        onnx_path = os.path.join(self._model_dir, "model.onnx")
        torchscript_path = os.path.join(self._model_dir, "model.pt")

        if prefer_onnx and os.path.exists(onnx_path):
            self._try_load_onnx(onnx_path, device)
            return
        if os.path.exists(torchscript_path):
            self._try_load_torchscript(torchscript_path, device)
            return
        if os.path.exists(onnx_path):
            self._try_load_onnx(onnx_path, device)
            return

        raise FileNotFoundError(
            f"未找到可加载的模型: ONNX={onnx_path}, TorchScript={torchscript_path}"
        )

    def _try_load_onnx(self, onnx_path: str, device: str):
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime 未安装")
            raise

        providers = []
        if device == "cuda" or (device == "auto" and "CUDAExecutionProvider" in ort.get_available_providers()):
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 0
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        self._backend = "onnxruntime"
        logger.info(f"ONNX 模型加载成功: provider={self._session.get_providers()[0]}")

    def _try_load_torchscript(self, path: str, device: str):
        try:
            import torch
        except ImportError:
            raise

        dev = torch.device("cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu")
        self._torchscript_model = torch.jit.load(path, map_location=dev)
        self._torchscript_model.eval()
        self._backend = "torchscript"
        logger.info(f"TorchScript 模型加载成功: device={dev}")

    def _warmup(self, iterations: int = 10):
        if self._input_shape is None:
            return
        h, w = self._input_shape[2], self._input_shape[3]
        dummy = np.random.rand(1, h, w).astype(np.float32)
        logger.info(f"模型预热 {iterations} 次...")
        t0 = time.time()
        for _ in range(iterations):
            try:
                self._predict_impl(dummy)
            except Exception as e:
                logger.warning(f"预热失败: {e}")
                break
        logger.info(f"预热完成: {(time.time() - t0) * 1000:.1f}ms")

    def shutdown(self) -> None:
        self._session = None
        self._torchscript_model = None
        self._initialized = False

    def _preprocess(self, masks: np.ndarray) -> np.ndarray:
        x = masks.astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        x = x[:, np.newaxis, :, :]
        return x

    def _postprocess(self, outputs: np.ndarray) -> np.ndarray:
        if outputs.ndim == 4 and outputs.shape[1] == 1:
            outputs = outputs[:, 0]
        return np.clip(outputs, 0.0, 1.0)

    def _predict_impl(self, masks: np.ndarray, **kwargs) -> np.ndarray:
        if self._backend == "onnxruntime":
            return self._predict_onnx(masks)
        elif self._backend == "torchscript":
            return self._predict_torchscript(masks)
        raise RuntimeError("模型未加载")

    def _predict_onnx(self, masks: np.ndarray) -> np.ndarray:
        import onnxruntime as ort
        input_data = self._preprocess(masks)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_data})
        return self._postprocess(outputs[0])

    def _predict_torchscript(self, masks: np.ndarray) -> np.ndarray:
        import torch
        input_data = self._preprocess(masks)
        with torch.no_grad():
            t = torch.from_numpy(input_data)
            if next(self._torchscript_model.parameters()).is_cuda:
                t = t.cuda()
            out = self._torchscript_model(t)
            return self._postprocess(out.cpu().numpy())


# ===========================================================================
# HopkinsLiteEngine: 轻量化 Hopkins 近似引擎
# ===========================================================================

@dataclass
class HopkinsParams:
    """Hopkins 近似参数"""
    wavelength_nm: float = 193.0
    na: float = 1.35
    sigma: float = 0.75
    defocus_nm: float = 0.0
    pixel_size_nm: float = 1.0
    illumination_type: str = "conventional"
    annular_sigma_inner: Optional[float] = None
    annular_sigma_outer: Optional[float] = None
    dipole_angle_deg: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "HopkinsParams":
        if not d:
            return cls()
        p = cls()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p


class HopkinsLiteEngine(BaseInferenceEngine):
    """
    轻量化 Hopkins 近似引擎

    基于 SOCS (Sum of Coherent Systems) 近似的快速部分相干成像，
    使用预计算的传输交叉系数 (TCC) 特征分解核函数。
    无需训练模型，纯 NumPy 实现，适合:
    - 代理模型不可用时的回退
    - 需要可解释性的场景
    - 快速原型验证
    """

    engine_type = EngineType.HOPKINS_LITE

    def __init__(self):
        super().__init__()
        self._default_params = HopkinsParams()
        self._kernel_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_lock = threading.Lock()

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        kernel_dir = cfg.get("hopkins_kernel_dir")
        if kernel_dir and os.path.exists(kernel_dir):
            self._preload_kernels(kernel_dir)

        self._default_params = HopkinsParams.from_dict({
            k: v for k, v in cfg.items()
            if k in ("wavelength_nm", "na", "sigma", "defocus_nm", "pixel_size_nm",
                     "illumination_type", "annular_sigma_inner", "annular_sigma_outer")
        })
        self._initialized = True
        logger.info("HopkinsLiteEngine 就绪 (纯 NumPy 近似)")

    def shutdown(self) -> None:
        self._kernel_cache.clear()
        self._initialized = False

    def _preload_kernels(self, kernel_dir: str):
        for fname in os.listdir(kernel_dir):
            if fname.endswith(".npz"):
                try:
                    key = fname[:-4]
                    data = np.load(os.path.join(kernel_dir, fname))
                    self._kernel_cache[key] = (data["kernels"], data["coeffs"])
                except Exception as e:
                    logger.warning(f"加载核函数失败 {fname}: {e}")
        logger.info(f"预加载 {len(self._kernel_cache)} 组核函数")

    # ------------------------------------------------------------------
    # 核心 Hopkins 近似实现
    # ------------------------------------------------------------------

    def _build_illumination_pupil(
        self, freq_x: np.ndarray, freq_y: np.ndarray, params: HopkinsParams
    ) -> np.ndarray:
        """构造照明光瞳 J(fx, fy)"""
        sigma = params.sigma
        rho_max = sigma
        freq_rho = np.sqrt(freq_x ** 2 + freq_y ** 2) / rho_max
        freq_rho = np.clip(freq_rho, 0, 1e6)

        illum_type = params.illumination_type.lower()

        if illum_type == "conventional":
            pupil = (freq_rho <= 1.0).astype(np.complex128)

        elif illum_type == "annular":
            s_in = params.annular_sigma_inner or (sigma * 0.7)
            s_out = params.annular_sigma_outer or sigma
            rho_in = s_in / sigma
            rho_out = s_out / sigma
            pupil = ((freq_rho >= rho_in) & (freq_rho <= rho_out)).astype(np.complex128)

        elif illum_type in ("dipole", "quasar"):
            angle = np.deg2rad(params.dipole_angle_deg or 0)
            sigma_open = 0.3
            n_poles = 2 if illum_type == "dipole" else 4
            pupil = np.zeros_like(freq_rho, dtype=np.complex128)
            for i in range(n_poles):
                theta = angle + i * np.pi / (n_poles // 2)
                pole_x = np.cos(theta) * (sigma * 0.7) / rho_max if rho_max > 0 else 0
                pole_y = np.sin(theta) * (sigma * 0.7) / rho_max if rho_max > 0 else 0
                dist = np.sqrt((freq_x / rho_max - pole_x) ** 2 + (freq_y / rho_max - pole_y) ** 2) if rho_max > 0 else 1e6
                pupil = np.maximum(pupil, (dist <= sigma_open).astype(np.complex128))
        else:
            pupil = (freq_rho <= 1.0).astype(np.complex128)

        return pupil

    def _build_projection_pupil(
        self, freq_x: np.ndarray, freq_y: np.ndarray, params: HopkinsParams
    ) -> np.ndarray:
        """构造投影光瞳 P(fx, fy)，含离焦像差"""
        na = params.na
        wavelength = params.wavelength_nm * 1e-9
        f_max = na / wavelength
        freq_rho = np.sqrt(freq_x ** 2 + freq_y ** 2)
        pupil = (freq_rho <= f_max).astype(np.complex128)

        if abs(params.defocus_nm) > 1e-6:
            defocus_m = params.defocus_nm * 1e-9
            radial_norm = freq_rho / max(f_max, 1e-20)
            radial_norm = np.clip(radial_norm, 0, 1)
            w_defocus = np.pi * defocus_m * (na ** 2) / wavelength * (1 - radial_norm ** 2)
            phase = np.exp(1j * w_defocus)
            pupil = pupil * phase

        return pupil

    def _compute_socs_kernels(
        self, image_shape: Tuple[int, int], params: HopkinsParams, num_kernels: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SOCS 分解: 计算 TCC 的前 K 个特征核

        返回:
            kernels: (K, H, W) 复值核函数 (空域)
            coeffs:  (K,) 特征值系数
        """
        cache_key = f"{image_shape[0]}x{image_shape[1]}_{params.wavelength_nm}_{params.na}_{params.sigma}_{params.defocus_nm}_{params.illumination_type}"
        with self._cache_lock:
            if cache_key in self._kernel_cache:
                return self._kernel_cache[cache_key]

        H, W = image_shape
        pixel = params.pixel_size_nm * 1e-9
        wavelength = params.wavelength_nm * 1e-9

        fx = np.fft.fftfreq(W, d=pixel).astype(np.float64)
        fy = np.fft.fftfreq(H, d=pixel).astype(np.float64)
        FX, FY = np.meshgrid(fx, fy)

        J = self._build_illumination_pupil(FX, FY, params)
        P = self._build_projection_pupil(FX, FY, params)
        P_conj = np.conj(P)

        k = num_kernels
        coeffs = np.zeros(k, dtype=np.float64)
        kernels = np.zeros((k, H, W), dtype=np.complex128)

        P_size = int(np.sum(np.abs(P) > 0))
        if P_size == 0:
            coeffs[0] = 1.0
            kernels[0] = np.ones((H, W), dtype=np.complex128) / np.sqrt(H * W)
            return kernels, coeffs

        approx_method = "diagonal"
        if approx_method == "diagonal":
            TCC_diag = np.real(J) * np.abs(P) ** 2
            flat = TCC_diag.flatten()
            top_indices = np.argsort(-np.abs(flat))[:k]

            for i, idx in enumerate(top_indices):
                h_idx, w_idx = divmod(idx, W)
                v = np.zeros((H, W), dtype=np.complex128)
                v[h_idx, w_idx] = 1.0
                kernel_freq = v * P
                kernel = np.fft.ifft2(kernel_freq) * np.sqrt(H * W)
                coeff = float(np.abs(flat[idx]))
                kernels[i] = kernel
                coeffs[i] = coeff

        total = np.sum(coeffs)
        if total > 0:
            coeffs = coeffs / total * np.sum(np.abs(J) * np.abs(P) ** 2) / (H * W)

        with self._cache_lock:
            self._kernel_cache[cache_key] = (kernels, coeffs)

        return kernels, coeffs

    def _predict_impl(self, masks: np.ndarray, **kwargs) -> np.ndarray:
        optical_params = kwargs.get("optical_params")
        params = HopkinsParams.from_dict(optical_params) if optical_params else self._default_params
        num_kernels = kwargs.get("hopkins_kernels", 8)

        N, H, W = masks.shape
        kernels, coeffs = self._compute_socs_kernels((H, W), params, num_kernels)

        aerial_accum = np.zeros((N, H, W), dtype=np.float64)
        masks_c64 = masks.astype(np.complex128)

        for i in range(len(kernels)):
            if coeffs[i] <= 0:
                continue
            kernel = kernels[i]
            K_norm = kernel / (np.sqrt(np.sum(np.abs(kernel) ** 2)) + 1e-20)

            for n in range(N):
                M_fft = np.fft.fft2(masks_c64[n])
                K_fft = np.fft.fft2(K_norm)
                conv_fft = M_fft * K_fft
                I_part = np.abs(np.fft.ifft2(conv_fft)) ** 2
                aerial_accum[n] += coeffs[i] * I_part

        max_val = aerial_accum.max()
        if max_val > 0:
            aerial_accum = aerial_accum / max_val

        return np.clip(aerial_accum.astype(np.float32), 0.0, 1.0)


# ===========================================================================
# 引擎工厂
# ===========================================================================

class EngineFactory:
    """引擎工厂，自动选择或创建推理引擎"""

    @staticmethod
    def create(
        engine_type: Union[EngineType, str] = "auto",
        config: Optional[Dict[str, Any]] = None,
        initialize: bool = True,
    ) -> BaseInferenceEngine:
        """
        创建推理引擎实例

        Args:
            engine_type: "surrogate", "hopkins_lite", "auto"
            config: 引擎配置字典
            initialize: 是否立即初始化

        Returns:
            初始化后的引擎实例
        """
        cfg = config or {}
        engine_type_str = engine_type.value if isinstance(engine_type, EngineType) else engine_type

        selected_type = None
        if engine_type_str == "auto":
            model_dir = cfg.get("model_dir", "./models")
            has_surrogate = (
                os.path.exists(os.path.join(model_dir, "model.onnx"))
                or os.path.exists(os.path.join(model_dir, "model.pt"))
            )
            selected_type = EngineType.SURROGATE if has_surrogate else EngineType.HOPKINS_LITE
            logger.info(f"自动选择引擎: {selected_type.value} (surrogate_available={has_surrogate})")
        else:
            selected_type = EngineType(engine_type_str)

        if selected_type == EngineType.SURROGATE:
            engine = SurrogateEngine()
        else:
            engine = HopkinsLiteEngine()

        if initialize:
            engine.initialize(cfg)

        return engine


def create_engine(
    engine_type: str = "auto",
    config: Optional[Dict[str, Any]] = None,
) -> BaseInferenceEngine:
    """便捷函数: 创建并初始化引擎"""
    return EngineFactory.create(engine_type, config, initialize=True)
