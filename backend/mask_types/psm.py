# -*- coding: utf-8 -*-
"""
相位偏移掩模 (PSM) 建模模块

支持多种光刻掩模类型的复数透过率建模：
- 二值掩模 (Binary Mask)：传统幅度掩模，作为基准
- 交替 PSM (Alt-PSM)：0/π 相位层，通过相消干涉提高分辨率
- 衰减式 PSM (Att-PSM)：同时具有幅度衰减和相位偏移

该模块提供：
1. 掩模变量 → 复数透过率的正向映射
2. 损失对掩模变量的梯度反向传播（链式法则）
3. 支持直接优化相位分布而非仅幅度
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


class MaskType(Enum):
    """
    掩模类型枚举

    Attributes:
        BINARY: 传统二值幅度掩模
        ALTERNATING_PSM: 交替相位偏移掩模 (0/π)
        ATTENUATED_PSM: 衰减式相位偏移掩模
    """
    BINARY = "binary"
    ALTERNATING_PSM = "alternating_psm"
    ATTENUATED_PSM = "attenuated_psm"


@dataclass
class PSMConfig:
    """
    PSM 配置参数

    Attributes:
        mask_type: 掩模类型
        phase_shift: 相移量 (弧度)，默认 π
        attenuation: 衰减系数 (0~1)，Att-PSM 专用
        phase_range: 相位优化范围 (弧度)，用于连续相位优化
    """
    mask_type: MaskType = MaskType.BINARY
    phase_shift: float = np.pi
    attenuation: float = 0.06
    phase_range: float = np.pi


class PhaseShiftMask:
    """
    相位偏移掩模基类

    定义掩模类型的通用接口，支持：
    - 正向：掩模变量 → 复数透过率
    - 反向：复透过率梯度 → 掩模变量梯度（链式法则）

    掩模变量 m 的取值范围为 [0, 1]，与现有优化器兼容。
    具体掩模类型定义 m 到复数透过率 t = A * exp(iφ) 的映射。
    """

    def __init__(self, config: Optional[PSMConfig] = None):
        """
        初始化 PSM 模型

        Args:
            config: PSM 配置参数
        """
        self.config = config or PSMConfig()

    def get_transmission(self, mask: np.ndarray) -> np.ndarray:
        """
        计算复数透过率

        Args:
            mask: 掩模变量数组，取值范围 [0, 1]

        Returns:
            复数透过率数组 (complex128)
        """
        raise NotImplementedError

    def get_amplitude(self, mask: np.ndarray) -> np.ndarray:
        """
        计算振幅透过率

        Args:
            mask: 掩模变量数组

        Returns:
            振幅透过率数组
        """
        return np.abs(self.get_transmission(mask))

    def get_phase(self, mask: np.ndarray) -> np.ndarray:
        """
        计算相位分布

        Args:
            mask: 掩模变量数组

        Returns:
            相位分布数组 (弧度)
        """
        return np.angle(self.get_transmission(mask))

    def gradient_wrt_mask(self, mask: np.ndarray, grad_transmission: np.ndarray) -> np.ndarray:
        """
        计算损失对掩模变量的梯度（链式法则）

        给定损失对复透过率的梯度 dL/dt，
        计算损失对掩模变量的梯度 dL/dm。

        数学推导：
            L 为实值损失函数
            t(m) = A(m) + i*B(m) 为复透过率
            dL/dm = Re{ dL/dt * dt/dm }
                  = Re(dL/dt) * dA/dm + Im(dL/dt) * dB/dm

        Args:
            mask: 掩模变量数组
            grad_transmission: 损失对复透过率的梯度 (dL/dt)，复数组

        Returns:
            损失对掩模变量的梯度 (dL/dm)，实数组
        """
        raise NotImplementedError

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """便捷调用：返回复数透过率"""
        return self.get_transmission(mask)


class BinaryMask(PhaseShiftMask):
    """
    传统二值幅度掩模

    作为基准模型，掩模变量直接对应振幅透过率，相位恒为 0。

    映射关系：
        t(m) = m + i*0
        A(m) = m
        φ(m) = 0

    梯度：
        dt/dm = 1
        dL/dm = Re(dL/dt)
    """

    def __init__(self, config: Optional[PSMConfig] = None):
        super().__init__(config)

    def get_transmission(self, mask: np.ndarray) -> np.ndarray:
        """
        二值掩模的复数透过率

        Args:
            mask: 掩模变量 [0, 1]

        Returns:
            复数透过率，实部等于掩模值，虚部为 0
        """
        return mask.astype(np.complex128)

    def gradient_wrt_mask(self, mask: np.ndarray, grad_transmission: np.ndarray) -> np.ndarray:
        """
        二值掩模梯度

        dL/dm = Re(dL/dt) （因为 t(m) = m，dt/dm = 1，且为实数）

        Args:
            mask: 掩模变量数组
            grad_transmission: 损失对复透过率的梯度

        Returns:
            损失对掩模变量的梯度
        """
        return np.real(grad_transmission).astype(np.float64)


class AlternatingPSM(PhaseShiftMask):
    """
    交替相位偏移掩模 (Alt-PSM)

    掩模完全透明（振幅=1），仅通过相位差实现成像增强。
    掩模变量 m 控制相位：φ = m * phase_shift

    典型应用：
    - phase_shift = π 时，相邻区域相位差 180°，产生相消干涉
    - 可提高分辨率和对比度，但需要额外的相位层制造工艺

    映射关系：
        t(m) = exp(i * m * φ_0)
        A(m) = 1
        φ(m) = m * φ_0
        其中 φ_0 = phase_shift (默认 π)

    梯度推导：
        dt/dm = i * φ_0 * exp(i * m * φ_0) = i * φ_0 * t(m)
        dL/dm = Re{ dL/dt * dt/dm }
              = Re{ dL/dt * i * φ_0 * t }
              = φ_0 * Re{ i * dL/dt * t }

        利用恒等式：Re{i * z} = -Im(z)
        dL/dm = -φ_0 * Im(dL/dt * t)

        另一种表达（用 dL/dt 的实部和虚部）：
        t = cosφ + i*sinφ
        dt/dm = i*φ_0 * (cosφ + i*sinφ)
              = -φ_0 * sinφ + i*φ_0 * cosφ
        dL/dm = Re(dL/dt) * (-φ_0*sinφ) + Im(dL/dt) * (φ_0*cosφ)
              = φ_0 * (-Re(dL/dt)*sinφ + Im(dL/dt)*cosφ)
    """

    def __init__(self, config: Optional[PSMConfig] = None):
        super().__init__(config)
        self._phase_shift = self.config.phase_shift

    def get_transmission(self, mask: np.ndarray) -> np.ndarray:
        """
        Alt-PSM 的复数透过率

        t(m) = exp(i * m * phase_shift)

        Args:
            mask: 掩模变量 [0, 1]，控制相位

        Returns:
            复数透过率，振幅恒为 1，相位随 m 线性变化
        """
        phase = mask.astype(np.float64) * self._phase_shift
        return np.exp(1j * phase).astype(np.complex128)

    def gradient_wrt_mask(self, mask: np.ndarray, grad_transmission: np.ndarray) -> np.ndarray:
        """
        Alt-PSM 梯度

        dL/dm = φ_0 * (-Re(dL/dt)*sinφ + Im(dL/dt)*cosφ)

        Args:
            mask: 掩模变量数组
            grad_transmission: 损失对复透过率的梯度 (dL/dt)

        Returns:
            损失对掩模变量的梯度 (dL/dm)
        """
        phase = mask.astype(np.float64) * self._phase_shift
        sin_phi = np.sin(phase)
        cos_phi = np.cos(phase)

        grad_real = np.real(grad_transmission)
        grad_imag = np.imag(grad_transmission)

        grad = self._phase_shift * (-grad_real * sin_phi + grad_imag * cos_phi)

        return grad.astype(np.float64)

    def gradient_wrt_phase(self, phase: np.ndarray, grad_transmission: np.ndarray) -> np.ndarray:
        """
        直接计算对相位的梯度（当相位作为直接优化变量时使用）

        dL/dφ = -Im(dL/dt * t)
              = -Re(dL/dt)*sinφ + Im(dL/dt)*cosφ

        Args:
            phase: 相位分布 (弧度)
            grad_transmission: 损失对复透过率的梯度

        Returns:
            损失对相位的梯度
        """
        sin_phi = np.sin(phase)
        cos_phi = np.cos(phase)

        grad_real = np.real(grad_transmission)
        grad_imag = np.imag(grad_transmission)

        grad = -grad_real * sin_phi + grad_imag * cos_phi

        return grad.astype(np.float64)


class AttenuatedPSM(PhaseShiftMask):
    """
    衰减式相位偏移掩模 (Att-PSM)

    相移层同时具有幅度衰减和相位偏移。
    掩模变量 m 在"完全透明区"和"相移衰减区"之间线性插值。

    典型参数（MoSi 相移掩模）：
    - 衰减系数：~6% (attenuation = 0.06)
    - 相移量：~180° (phase_shift = π)

    映射关系：
        t_clear = 1 + i*0  (m = 1)
        t_shifter = attenuation * exp(i * phase_shift)  (m = 0)
        t(m) = m * t_clear + (1 - m) * t_shifter
             = t_shifter + m * (t_clear - t_shifter)

    梯度推导：
        dt/dm = t_clear - t_shifter = 1 - attenuation * exp(i*φ_0)  (常数)
        dL/dm = Re{ dL/dt * dt/dm }
              = Re(dL/dt) * Re(dt/dm) + Im(dL/dt) * Im(dt/dm)

        其中：
        Re(dt/dm) = 1 - attenuation * cos(φ_0)
        Im(dt/dm) = -attenuation * sin(φ_0)
    """

    def __init__(self, config: Optional[PSMConfig] = None):
        super().__init__(config)
        self._phase_shift = self.config.phase_shift
        self._attenuation = self.config.attenuation

        t_shifter = self._attenuation * np.exp(1j * self._phase_shift)
        self._t_clear = 1.0 + 0j
        self._t_shifter = t_shifter

        self._dt_dm = self._t_clear - self._t_shifter
        self._dt_dm_real = np.real(self._dt_dm)
        self._dt_dm_imag = np.imag(self._dt_dm)

    def get_transmission(self, mask: np.ndarray) -> np.ndarray:
        """
        Att-PSM 的复数透过率

        t(m) = m * 1 + (1 - m) * attenuation * exp(i * phase_shift)

        Args:
            mask: 掩模变量 [0, 1]

        Returns:
            复数透过率
        """
        mask_f = mask.astype(np.float64)
        t = self._t_shifter + mask_f * self._dt_dm
        return t.astype(np.complex128)

    def gradient_wrt_mask(self, mask: np.ndarray, grad_transmission: np.ndarray) -> np.ndarray:
        """
        Att-PSM 梯度

        dL/dm = Re(dL/dt) * Re(dt/dm) + Im(dL/dt) * Im(dt/dm)

        由于 dt/dm 是常数，梯度就是 dL/dt 在 dt/dm 方向上的实内积。

        Args:
            mask: 掩模变量数组（未直接使用，保持接口一致性）
            grad_transmission: 损失对复透过率的梯度

        Returns:
            损失对掩模变量的梯度
        """
        grad_real = np.real(grad_transmission)
        grad_imag = np.imag(grad_transmission)

        grad = grad_real * self._dt_dm_real + grad_imag * self._dt_dm_imag

        return grad.astype(np.float64)

    def get_shifter_transmission(self) -> complex:
        """获取相移层的复数透过率（固定值）"""
        return self._t_shifter

    def get_clear_transmission(self) -> complex:
        """获取透明区的复数透过率（固定值 = 1）"""
        return self._t_clear


class ContinuousPhaseMask(PhaseShiftMask):
    """
    连续相位掩模

    相位作为连续优化变量，振幅固定为 1。
    可用于更一般的相位优化问题。

    映射关系：
        t(φ) = exp(i * φ)
        A(φ) = 1
        φ 直接为优化变量（不再是 [0,1] 归一化）

    梯度：
        dt/dφ = i * exp(iφ) = i * t
        dL/dφ = Re{ dL/dt * i * t } = -Im(dL/dt * t)
    """

    def __init__(self, config: Optional[PSMConfig] = None):
        super().__init__(config)

    def get_transmission(self, phase: np.ndarray) -> np.ndarray:
        """
        连续相位掩模的复数透过率

        Args:
            phase: 相位分布 (弧度)

        Returns:
            复数透过率
        """
        return np.exp(1j * phase.astype(np.float64)).astype(np.complex128)

    def gradient_wrt_phase(self, phase: np.ndarray, grad_transmission: np.ndarray) -> np.ndarray:
        """
        对相位的梯度

        dL/dφ = -Im(dL/dt * t)
              = -Re(dL/dt)*sinφ + Im(dL/dt)*cosφ

        Args:
            phase: 相位分布
            grad_transmission: 损失对复透过率的梯度

        Returns:
            损失对相位的梯度
        """
        sin_phi = np.sin(phase)
        cos_phi = np.cos(phase)

        grad_real = np.real(grad_transmission)
        grad_imag = np.imag(grad_transmission)

        grad = -grad_real * sin_phi + grad_imag * cos_phi

        return grad.astype(np.float64)

    def gradient_wrt_mask(self, mask: np.ndarray, grad_transmission: np.ndarray) -> np.ndarray:
        """
        兼容接口：将 [0,1] 掩模变量映射到相位范围

        φ(m) = m * phase_range - phase_range/2 （以 0 为中心）

        Args:
            mask: 掩模变量 [0, 1]
            grad_transmission: 损失对复透过率的梯度

        Returns:
            损失对掩模变量的梯度
        """
        phase_range = self.config.phase_range
        phase = mask.astype(np.float64) * phase_range - phase_range / 2.0

        grad_phase = self.gradient_wrt_phase(phase, grad_transmission)
        grad_mask = grad_phase * phase_range

        return grad_mask.astype(np.float64)


class AmplitudePhaseMask(PhaseShiftMask):
    """
    幅度-相位联合掩模

    同时优化幅度和相位，需要两个独立的掩模变量。
    支持最一般的复振幅掩模优化。

    映射关系：
        t(A, φ) = A * exp(i * φ)
        A ∈ [0, 1], φ ∈ [-π, π]

    梯度（分别对幅度和相位）：
        dt/dA = exp(iφ) = t / A
        dL/dA = Re{ dL/dt * exp(iφ) }
              = Re(dL/dt) * cosφ + Im(dL/dt) * sinφ

        dt/dφ = i * A * exp(iφ) = i * t
        dL/dφ = Re{ dL/dt * i * t }
              = -Im(dL/dt * t)
              = A * (-Re(dL/dt)*sinφ + Im(dL/dt)*cosφ)
    """

    def __init__(self, config: Optional[PSMConfig] = None):
        super().__init__(config)

    def get_transmission(self, amplitude: np.ndarray, phase: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算复数透过率

        Args:
            amplitude: 振幅分布 [0, 1]
            phase: 相位分布 (弧度)，如果为 None 则假设相位为 0

        Returns:
            复数透过率
        """
        amp = amplitude.astype(np.float64)
        if phase is None:
            return amp.astype(np.complex128)
        return (amp * np.exp(1j * phase.astype(np.float64))).astype(np.complex128)

    def gradient_wrt_amplitude(
        self, amplitude: np.ndarray, phase: np.ndarray, grad_transmission: np.ndarray
    ) -> np.ndarray:
        """
        对幅度的梯度

        dL/dA = Re(dL/dt) * cosφ + Im(dL/dt) * sinφ

        Args:
            amplitude: 振幅分布
            phase: 相位分布
            grad_transmission: 损失对复透过率的梯度

        Returns:
            损失对幅度的梯度
        """
        cos_phi = np.cos(phase)
        sin_phi = np.sin(phase)

        grad_real = np.real(grad_transmission)
        grad_imag = np.imag(grad_transmission)

        grad = grad_real * cos_phi + grad_imag * sin_phi

        return grad.astype(np.float64)

    def gradient_wrt_phase(
        self, amplitude: np.ndarray, phase: np.ndarray, grad_transmission: np.ndarray
    ) -> np.ndarray:
        """
        对相位的梯度

        dL/dφ = A * (-Re(dL/dt)*sinφ + Im(dL/dt)*cosφ)

        Args:
            amplitude: 振幅分布
            phase: 相位分布
            grad_transmission: 损失对复透过率的梯度

        Returns:
            损失对相位的梯度
        """
        amp = amplitude.astype(np.float64)
        cos_phi = np.cos(phase)
        sin_phi = np.sin(phase)

        grad_real = np.real(grad_transmission)
        grad_imag = np.imag(grad_transmission)

        grad = amp * (-grad_real * sin_phi + grad_imag * cos_phi)

        return grad.astype(np.float64)

    def gradient_wrt_mask(self, mask: np.ndarray, grad_transmission: np.ndarray) -> np.ndarray:
        """
        兼容接口：不支持单变量模式

        对于双变量掩模，需要分别使用 gradient_wrt_amplitude 和 gradient_wrt_phase。
        """
        raise NotImplementedError(
            "AmplitudePhaseMask 需要两个独立变量，"
            "请使用 gradient_wrt_amplitude 和 gradient_wrt_phase 分别计算梯度。"
        )


def create_mask_model(mask_type: MaskType, **kwargs) -> PhaseShiftMask:
    """
    工厂函数：创建掩模模型

    Args:
        mask_type: 掩模类型
        **kwargs: 传递给 PSMConfig 的参数

    Returns:
        掩模模型实例
    """
    config = PSMConfig(mask_type=mask_type, **kwargs)

    if mask_type == MaskType.BINARY:
        return BinaryMask(config)
    elif mask_type == MaskType.ALTERNATING_PSM:
        return AlternatingPSM(config)
    elif mask_type == MaskType.ATTENUATED_PSM:
        return AttenuatedPSM(config)
    else:
        raise ValueError(f"未知的掩模类型: {mask_type}")


def compute_complex_gradient(
    field: np.ndarray,
    intensity_grad: np.ndarray,
) -> np.ndarray:
    """
    从光强梯度计算复场梯度

    给定光强 I = |E|^2 = E * E*，以及损失对光强的梯度 dL/dI，
    计算损失对复电场的梯度 dL/dE。

    数学推导：
        dL/dE = dL/dI * dI/dE
        dI/dE = E* （共轭）
        dL/dE = dL/dI * E*

    注意：这里是 Wirtinger 导数意义下的梯度，
    即把 E 和 E* 视为独立变量。
    对于实值损失函数 L，实际梯度（沿实部/虚部分解）为：
        ∂L/∂Re(E) = 2 * Re(dL/dE)
        ∂L/∂Im(E) = 2 * Im(dL/dE)
    但在链式法则中直接使用 dL/dE 即可。

    Args:
        field: 复电场分布 E
        intensity_grad: 损失对光强的梯度 dL/dI

    Returns:
        损失对复电场的梯度 dL/dE
    """
    return intensity_grad.astype(np.float64) * np.conj(field)


def verify_gradient_numerical(
    mask_model: PhaseShiftMask,
    mask: np.ndarray,
    eps: float = 1e-6,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    数值验证梯度正确性

    使用有限差分法验证解析梯度的正确性。

    Args:
        mask_model: 掩模模型
        mask: 掩模变量
        eps: 有限差分步长
        seed: 随机种子（用于生成随机损失梯度）

    Returns:
        包含解析梯度、数值梯度和相对误差的字典
    """
    rng = np.random.RandomState(seed)

    transmission = mask_model.get_transmission(mask)

    grad_trans_real = rng.randn(*mask.shape)
    grad_trans_imag = rng.randn(*mask.shape)
    grad_transmission = grad_trans_real + 1j * grad_trans_imag

    grad_analytical = mask_model.gradient_wrt_mask(mask, grad_transmission)

    grad_numerical = np.zeros_like(mask, dtype=np.float64)
    ny, nx = mask.shape

    for i in range(ny):
        for j in range(nx):
            mask_plus = mask.copy()
            mask_plus[i, j] += eps
            mask_minus = mask.copy()
            mask_minus[i, j] -= eps

            t_plus = mask_model.get_transmission(mask_plus)
            t_minus = mask_model.get_transmission(mask_minus)

            loss_plus = np.sum(grad_trans_real * np.real(t_plus) + grad_trans_imag * np.imag(t_plus))
            loss_minus = np.sum(grad_trans_real * np.real(t_minus) + grad_trans_imag * np.imag(t_minus))

            grad_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

    error = np.abs(grad_analytical - grad_numerical)
    rel_error = error / (np.abs(grad_analytical) + 1e-10)

    return {
        'analytical': grad_analytical,
        'numerical': grad_numerical,
        'abs_error': error,
        'rel_error': rel_error,
        'max_rel_error': float(np.max(rel_error)),
        'mean_rel_error': float(np.mean(rel_error)),
        'correct': bool(np.max(rel_error) < 1e-4),
    }


class PSMImagingWrapper:
    """
    PSM 掩模 + 成像系统的端到端封装

    将相位偏移掩模模型与部分相干成像系统结合起来，
    提供从掩模变量到空间像的前向传播，以及
    损失对掩模变量的反向梯度传播。

    使用场景：
    - 直接优化相位分布（Alt-PSM 或 Att-PSM）
    - 对比不同掩模类型的成像效果
    - 端到端梯度验证

    梯度链：
        L( I( t(m) ) )
        dL/dm = dL/dI · dI/dt · dt/dm

    其中：
        dL/dI: 损失对光强的梯度（外部提供）
        dI/dt: 光强对复透过率的梯度（由成像模型提供）
        dt/dm: 复透过率对掩模变量的梯度（由 PSM 模型提供）
    """

    def __init__(
        self,
        imaging_model,
        mask_model: PhaseShiftMask,
    ):
        """
        初始化 PSM 成像封装

        Args:
            imaging_model: 部分相干成像模型 (PartialCoherentImaging)
            mask_model: 掩模模型 (PhaseShiftMask)
        """
        self.imaging = imaging_model
        self.mask_model = mask_model

    def compute_aerial_image(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像

        Args:
            mask: 掩模变量 [0, 1]

        Returns:
            空间像光强分布
        """
        t = self.mask_model.get_transmission(mask)
        return self.imaging.compute_aerial_image_complex(t)

    def compute_gradient(
        self,
        mask: np.ndarray,
        intensity_grad: np.ndarray,
    ) -> np.ndarray:
        """
        计算损失对掩模变量的梯度（端到端）

        Args:
            mask: 掩模变量 [0, 1]
            intensity_grad: 损失对光强的梯度 dL/dI

        Returns:
            损失对掩模变量的梯度 dL/dm
        """
        t = self.mask_model.get_transmission(mask)
        grad_t = self.imaging.compute_complex_gradient(t, intensity_grad)
        grad_m = self.mask_model.gradient_wrt_mask(mask, grad_t)
        return grad_m

    def compute_image_and_gradient(
        self,
        mask: np.ndarray,
        intensity_grad: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时计算空间像和梯度（更高效，复用中间结果）

        Args:
            mask: 掩模变量
            intensity_grad: 损失对光强的梯度

        Returns:
            (空间像, 掩模梯度)
        """
        t = self.mask_model.get_transmission(mask)
        aerial = self.imaging.compute_aerial_image_complex(t)
        grad_t = self.imaging.compute_complex_gradient(t, intensity_grad)
        grad_m = self.mask_model.gradient_wrt_mask(mask, grad_t)
        return aerial, grad_m


class PhaseOnlyImagingWrapper:
    """
    纯相位优化的成像封装

    直接以相位作为优化变量，振幅固定为 1。
    适用于纯相位掩模的优化问题。

    优化变量：相位分布 φ(x, y)
    复透过率：t(φ) = exp(iφ)
    """

    def __init__(
        self,
        imaging_model,
    ):
        """
        初始化纯相位成像封装

        Args:
            imaging_model: 部分相干成像模型
        """
        self.imaging = imaging_model

    def compute_aerial_image(self, phase: np.ndarray) -> np.ndarray:
        """
        计算空间像

        Args:
            phase: 相位分布 (弧度)

        Returns:
            空间像光强分布
        """
        t = np.exp(1j * phase.astype(np.float64))
        return self.imaging.compute_aerial_image_complex(t)

    def compute_gradient(
        self,
        phase: np.ndarray,
        intensity_grad: np.ndarray,
    ) -> np.ndarray:
        """
        计算损失对相位的梯度

        dL/dφ = -Im( dL/dt * t )
              = -Re(dL/dt)*sinφ + Im(dL/dt)*cosφ

        Args:
            phase: 相位分布
            intensity_grad: 损失对光强的梯度

        Returns:
            损失对相位的梯度
        """
        t = np.exp(1j * phase.astype(np.float64))
        grad_t = self.imaging.compute_complex_gradient(t, intensity_grad)

        sin_phi = np.sin(phase)
        cos_phi = np.cos(phase)

        grad_real = np.real(grad_t)
        grad_imag = np.imag(grad_t)

        grad_phase = -grad_real * sin_phi + grad_imag * cos_phi

        return grad_phase.astype(np.float64)


class AmplitudePhaseImagingWrapper:
    """
    幅度-相位联合优化的成像封装

    同时优化幅度和相位两个独立变量。
    适用于一般复振幅掩模的优化问题。

    优化变量：幅度 A(x, y) ∈ [0, 1]，相位 φ(x, y)
    复透过率：t(A, φ) = A * exp(iφ)
    """

    def __init__(
        self,
        imaging_model,
    ):
        """
        初始化幅度-相位成像封装

        Args:
            imaging_model: 部分相干成像模型
        """
        self.imaging = imaging_model

    def compute_aerial_image(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
    ) -> np.ndarray:
        """
        计算空间像

        Args:
            amplitude: 幅度分布 [0, 1]
            phase: 相位分布 (弧度)

        Returns:
            空间像光强分布
        """
        t = amplitude.astype(np.float64) * np.exp(1j * phase.astype(np.float64))
        return self.imaging.compute_aerial_image_complex(t)

    def compute_gradients(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        intensity_grad: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算损失对幅度和相位的梯度

        dL/dA = Re(dL/dt) * cosφ + Im(dL/dt) * sinφ
        dL/dφ = A * (-Re(dL/dt)*sinφ + Im(dL/dt)*cosφ)

        Args:
            amplitude: 幅度分布
            phase: 相位分布
            intensity_grad: 损失对光强的梯度

        Returns:
            (幅度梯度, 相位梯度)
        """
        t = amplitude.astype(np.float64) * np.exp(1j * phase.astype(np.float64))
        grad_t = self.imaging.compute_complex_gradient(t, intensity_grad)

        cos_phi = np.cos(phase)
        sin_phi = np.sin(phase)
        amp = amplitude.astype(np.float64)

        grad_real = np.real(grad_t)
        grad_imag = np.imag(grad_t)

        grad_amplitude = grad_real * cos_phi + grad_imag * sin_phi
        grad_phase = amp * (-grad_real * sin_phi + grad_imag * cos_phi)

        return grad_amplitude.astype(np.float64), grad_phase.astype(np.float64)


def verify_end_to_end_gradient_numerical(
    wrapper: PSMImagingWrapper,
    mask: np.ndarray,
    target_image: np.ndarray,
    eps: float = 1e-5,
    metric: str = 'mse',
) -> Dict[str, Any]:
    """
    数值验证端到端梯度正确性

    使用有限差分法验证整个成像链路的梯度计算。

    Args:
        wrapper: PSM 成像封装
        mask: 掩模变量
        target_image: 目标图像（用于计算损失）
        eps: 有限差分步长
        metric: 损失函数类型 ('mse' 或 'mae')

    Returns:
        包含解析梯度、数值梯度和相对误差的字典
    """
    ny, nx = mask.shape

    aerial = wrapper.compute_aerial_image(mask)

    if metric.lower() == 'mse':
        error = aerial - target_image
        intensity_grad = 2.0 * error / (ny * nx)
        loss_analytical = np.mean(error ** 2)
    elif metric.lower() == 'mae':
        error = np.sign(aerial - target_image)
        intensity_grad = error / (ny * nx)
        loss_analytical = np.mean(np.abs(aerial - target_image))
    else:
        raise ValueError(f"不支持的度量: {metric}")

    grad_analytical = wrapper.compute_gradient(mask, intensity_grad)

    grad_numerical = np.zeros_like(mask, dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            mask_plus = mask.copy()
            mask_plus[i, j] += eps
            aerial_plus = wrapper.compute_aerial_image(mask_plus)

            mask_minus = mask.copy()
            mask_minus[i, j] -= eps
            aerial_minus = wrapper.compute_aerial_image(mask_minus)

            if metric.lower() == 'mse':
                loss_plus = np.mean((aerial_plus - target_image) ** 2)
                loss_minus = np.mean((aerial_minus - target_image) ** 2)
            else:
                loss_plus = np.mean(np.abs(aerial_plus - target_image))
                loss_minus = np.mean(np.abs(aerial_minus - target_image))

            grad_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

    error = np.abs(grad_analytical - grad_numerical)
    rel_error = error / (np.abs(grad_analytical) + 1e-10)

    return {
        'analytical': grad_analytical,
        'numerical': grad_numerical,
        'abs_error': error,
        'rel_error': rel_error,
        'max_rel_error': float(np.max(rel_error)),
        'mean_rel_error': float(np.mean(rel_error)),
        'correct': bool(np.max(rel_error) < 1e-3),
    }
