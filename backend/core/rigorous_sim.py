# -*- coding: utf-8 -*-
"""
严格电磁仿真与矢量衍射接口模块

为高精度场景提供与现有标量 Hopkins 模型并行的仿真路径。
包含三种后端：
    1. 'hopkins'  — 现有标量 Hopkins 模型 (PartialCoherentImaging)，默认
    2. 'rcwa'     — 1D 严格耦合波分析 (Rigorous Coupled-Wave Analysis)
                    用于周期性 line/space 等一维光栅结构的矢量衍射求解
    3. 'fdtd'     — 有限差分时域法 (Finite-Difference Time-Domain) 占位
                    预留接口，可对接 meep / MEEP 或自研求解器

统一入口：
    simulate(mask, backend='hopkins', **kwargs)
        -> {
            'aerial_image': 2D ndarray,       # 归一化空间像 [0,1]
            'wafer_image': 2D ndarray,        # 阈值后晶圆图 [0,1]
            'diffraction_orders': (可选) {
                'orders': 1D ndarray,         # 衍射级次编号 [-M..M]
                'efficiencies_TE': 1D ndarray,# TE 偏振衍射效率
                'efficiencies_TM': 1D ndarray,# TM 偏振衍射效率
                'fields': ...                 # 近/远场细节
            }
           }
"""

from __future__ import annotations

import logging
import warnings
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numba import jit, prange

from core.imaging import (
    OpticalSystem,
    PartialCoherentImaging,
    ProcessCondition,
    ResistModel,
    apply_resist_model,
    simulate_wafer_image,
    simulate_multi_process,
    MultiProcessSimulationResult,
    _apply_threshold,
)
from core.litho_metrics import (
    compute_cd,
    compute_cd_error,
    compute_epe,
    compute_ils,
    compute_nils,
    evaluate_litho_metrics,
    LithoMetricsResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举与配置类
# =============================================================================
class SimulationBackend(str, Enum):
    """仿真后端枚举"""
    HOPKINS = "hopkins"
    RCWA = "rcwa"
    FDTD = "fdtd"


class Polarization(str, Enum):
    """偏振态枚举"""
    TE = "TE"          # 电场沿光栅沟槽方向 (s-polarization, E//y)
    TM = "TM"          # 电场垂直沟槽方向 (p-polarization, E//xz 面)
    UNPOLARIZED = "unpolarized"  # TE+TM 等权平均


@dataclass
class RCWAConfig:
    """RCWA 求解器配置

    Attributes:
        n_orders: 衍射级次截断数（单侧），总级次数 = 2*n_orders+1；
                  典型收敛需要 3~10 级，高对比度需要更多。
        polarization: 入射偏振态。
        n_superstrate: 上层介质折射率（如光刻胶/浸没液）。
        n_substrate: 下层介质折射率（如掩模玻璃 / SiO2）。
        n_grating_line: 光栅线条区域折射率（如 Cr 吸收层取复数 n-ik）。
        grating_thickness_nm: 光栅厚度 (nm)。
        period_nm: 光栅周期 (nm)；None 时从掩模图案自动估计。
        line_width_nm: 线宽 (nm)；None 时从掩模图案自动估计。
        convergence_tol: S 矩阵级次收敛判据（相对能量残差）。
        max_iter: 反射/透射迭代次数（用于薄膜堆栈时）。
        use_meent_if_available: 若安装了 meent 开源库则优先使用。
    """
    n_orders: int = 5
    polarization: Polarization = Polarization.UNPOLARIZED
    n_superstrate: complex = 1.44 + 0.0j      # 典型浸没水
    n_substrate: complex = 1.56 + 0.0j         # 熔融石英
    n_grating_line: complex = 3.28 - 4.32j     # 典型 Cr @ 193nm
    grating_thickness_nm: float = 70.0
    period_nm: Optional[float] = None
    line_width_nm: Optional[float] = None
    convergence_tol: float = 1e-6
    max_iter: int = 50
    use_meent_if_available: bool = True


@dataclass
class FDTDConfig:
    """FDTD 求解器配置（占位实现，可对接 meep 等开源库）"""
    grid_resolution_nm: float = 0.5
    pml_thickness_nm: float = 200.0
    total_time_steps: int = 2000
    courant_factor: float = 0.9
    use_meep_if_available: bool = True
    extra_material_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """统一仿真返回结构"""
    aerial_image: np.ndarray
    wafer_image: np.ndarray
    backend: SimulationBackend
    runtime_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.aerial_image.shape)


# =============================================================================
# 1D RCWA 严格耦合波分析（简化参考实现）
# =============================================================================
class RCWASolver1D:
    """
    一维二进制相位/吸收光栅 RCWA 求解器

    参考 Moharam & Gaylord 经典公式体系：
        J. Opt. Soc. Am. 72, 1385 (1982)
        J. Opt. Soc. Am. A 12, 1068 (1995) —— 增强透射矩阵 (RAT) 稳定版本

    目前实现：
        * 1D 二元光栅（line/space）
        * TE / TM 两种偏振
        * 任意入射角（由照明 NA 自动换算）
        * 远场衍射级次复振幅 → 通过投影物镜光瞳 → 空间像计算
    """

    def __init__(self, config: RCWAConfig):
        self.cfg = config

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def solve_far_field(
        self,
        wavelength_nm: float,
        period_nm: float,
        duty_cycle: float,
        theta_deg: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        求解远场衍射级次（标准 RAT 增强透射矩阵法）

        参考: Moharam & Gaylord, J. Opt. Soc. Am. A 12, 1068 (1995)

        Args:
            wavelength_nm: 真空波长 (nm)
            period_nm: 光栅周期 (nm)
            duty_cycle: 占空比 = 线宽 / 周期，范围 (0,1)
            theta_deg: 入射角 (度)

        Returns:
            dict 包含:
                orders:       级次编号数组 shape=(2M+1,)
                r_TE/t_TE:    TE 反射/透射复振幅 (2M+1,)
                r_TM/t_TM:    TM 反射/透射复振幅 (2M+1,)
                eff_reflect_TE/TM: 各级反射衍射效率
                eff_trans_TE/TM:   各级透射衍射效率
        """
        k0 = 2.0 * np.pi / wavelength_nm
        d = self.cfg.grating_thickness_nm
        n1 = complex(self.cfg.n_superstrate)
        n2 = complex(self.cfg.n_substrate)
        na = complex(self.cfg.n_grating_line)
        nb = n2
        fill = float(np.clip(duty_cycle, 1e-4, 1 - 1e-4))

        eps1 = n1 * n1
        eps2 = n2 * n2
        eps_a = na * na
        eps_b = nb * nb

        M = int(self.cfg.n_orders)
        orders = np.arange(-M, M + 1)
        N = 2 * M + 1

        theta_rad = np.deg2rad(theta_deg)
        kx_inc = k0 * n1 * np.sin(theta_rad)
        kx_m = kx_inc + 2.0 * np.pi * orders / period_nm

        result: Dict[str, np.ndarray] = {"orders": orders}
        inc_idx = M

        for pol in ("TE", "TM"):
            if pol == "TE":
                # --------------------------------------------------------
                # TE 偏振: E_y 为主分量
                #   连续边界条件: E_y, H_x ∝ ∂E_y/∂z
                #   均匀层导纳: Y_m = kz_m / (ωμ₀) = kz_m / k0 (η₀归一化)
                # --------------------------------------------------------
                E_mat = _toeplitz_epsilon_1d(eps_a, eps_b, fill, N)
                Kx = np.diag(kx_m / k0)
                kz1 = _safe_kz(kx_m, k0 * n1)
                kz2 = _safe_kz(kx_m, k0 * n2)

                eigvals, W = np.linalg.eig(Kx @ Kx + E_mat)
                beta_raw = k0 * np.lib.scimath.sqrt(
                    np.atleast_1d(eigvals).astype(complex)
                )
                beta = np.where(np.imag(beta_raw) < 0, -beta_raw, beta_raw)

                Lam_diag = np.exp(-1j * beta * d)
                Lam = np.diag(Lam_diag)
                Lam_inv = np.diag(1.0 / Lam_diag)

                # V = W · diag(β/k0): 与 ∂E_y/∂z 成比例的量
                V = W @ np.diag(beta / k0)

                # 区域1 (superstrate) 与区域2 (substrate) 的导纳矩阵
                Y1 = np.diag(kz1 / k0)
                Y2 = np.diag(kz2 / k0)

                # --- 构建 RAT 全局矩阵 ---
                # 边界 z=0:  [W,   W  ] [c+]   = [I,   I ] [δ]
                #            [V,  -V  ] [c-]     [Y1, -Y1] [r]
                # 边界 z=d:  [WΛ, WΛ⁻¹] [c+]   = [I,   I ] [t]
                #            [VΛ,-VΛ⁻¹] [c-]     [Y2, -Y2] [0]
                A_bot = np.hstack([W, W])
                A_top = np.hstack([V, -V])
                B_bot = np.hstack([W @ Lam, W @ Lam_inv])
                B_top = np.hstack([V @ Lam, -V @ Lam_inv])
                T1_bot = np.hstack([np.eye(N), np.eye(N)])
                T1_top = np.hstack([Y1, -Y1])
                T2_bot = np.hstack([np.eye(N), np.eye(N)])
                T2_top = np.hstack([Y2, -Y2])

                # 消去光栅内部变量: [c+; c-]
                # A_mat = [[A_bot],[A_top]]; B_mat = [[B_bot],[B_top]]
                # A_mat·[c+;c-] = T1·[δ;r]  ==> [c+;c-] = A_mat^-1·T1·[δ;r]
                # B_mat·[c+;c-] = T2·[t;0]  ==> [c+;c-] = B_mat^-1·T2·[t;0]
                # ==>  S_mat = A_mat·B_mat^-1·T2  ;  T1·[δ;r] = S_mat·[t;0]
                A_mat = np.vstack([A_bot, A_top])
                B_mat = np.vstack([B_bot, B_top])
                T1 = np.vstack([T1_bot, T1_top])
                T2 = np.vstack([T2_bot, T2_top])

                try:
                    S_mat = A_mat @ np.linalg.solve(B_mat, T2)
                except np.linalg.LinAlgError:
                    S_mat = A_mat @ np.linalg.lstsq(B_mat, T2, rcond=None)[0]

                # T1 · [δ; r] = S_mat · [t; 0]
                # 分块 S_mat = [[S11, S12],[S21, S22]] (每块 N×N)
                S11 = S_mat[:N, :N]
                S21 = S_mat[N:, :N]

                # 入射向量 (只有 0 级入射)
                delta = np.zeros(N, dtype=complex)
                delta[inc_idx] = 1.0 + 0.0j

                # T1 = [[I, I],[Y1,-Y1]]
                # => [I·δ + I·r] = [S11·t]
                #    [Y1·δ - Y1·r]  [S21·t]
                # 消去 r: 2·Y1·δ = (Y1·S11 + S21)·t
                lhs_t = Y1 @ S11 + S21
                try:
                    t_TE = np.linalg.solve(lhs_t, 2.0 * (Y1 @ delta))
                except np.linalg.LinAlgError:
                    t_TE = np.linalg.lstsq(lhs_t, 2.0 * (Y1 @ delta), rcond=None)[0]
                r_TE = S11 @ t_TE - delta

                # 衍射效率 (Poynting 矢量 z 分量归一化)
                kz1_inc = kz1[inc_idx]
                eff_t_TE = np.abs(t_TE) ** 2 * np.real(kz2 / (kz1_inc + 1e-30))
                eff_r_TE = np.abs(r_TE) ** 2 * np.real(kz1 / (kz1_inc + 1e-30))
                result["t_TE"] = t_TE
                result["r_TE"] = r_TE
                result["eff_trans_TE"] = eff_t_TE
                result["eff_reflect_TE"] = eff_r_TE

            else:
                # --------------------------------------------------------
                # TM 偏振: H_y 为主分量
                #   连续边界条件: H_y, E_x ∝ (1/ε) ∂H_y/∂z
                #   均匀层导纳: Y_m = ωε / kz_m = k0·ε_r / kz_m
                # --------------------------------------------------------
                inv_E_mat = _toeplitz_inverse_epsilon_1d(eps_a, eps_b, fill, N)
                Kx = np.diag(kx_m / k0)
                kz1 = _safe_kz(kx_m, k0 * n1)
                kz2 = _safe_kz(kx_m, k0 * n2)

                eigvals, W = np.linalg.eig(Kx @ inv_E_mat @ Kx + np.eye(N))
                beta_raw = k0 * np.lib.scimath.sqrt(
                    np.atleast_1d(eigvals).astype(complex)
                )
                beta = np.where(np.imag(beta_raw) < 0, -beta_raw, beta_raw)

                Lam_diag = np.exp(-1j * beta * d)
                Lam = np.diag(Lam_diag)
                Lam_inv = np.diag(1.0 / Lam_diag)

                # V = inv_E · W · diag(β/k0): 与 (1/ε)∂H_y/∂z 成比例
                V = inv_E_mat @ (W @ np.diag(beta / k0))

                # TM 导纳: Y = diag(k0·ε_r / kz)
                Y1 = np.diag(k0 * eps1 / (kz1 + 1e-30))
                Y2 = np.diag(k0 * eps2 / (kz2 + 1e-30))

                A_bot = np.hstack([W, W])
                A_top = np.hstack([V, -V])
                B_bot = np.hstack([W @ Lam, W @ Lam_inv])
                B_top = np.hstack([V @ Lam, -V @ Lam_inv])
                T1_bot = np.hstack([np.eye(N), np.eye(N)])
                T1_top = np.hstack([Y1, -Y1])
                T2_bot = np.hstack([np.eye(N), np.eye(N)])
                T2_top = np.hstack([Y2, -Y2])

                A_mat = np.vstack([A_bot, A_top])
                B_mat = np.vstack([B_bot, B_top])
                T1 = np.vstack([T1_bot, T1_top])
                T2 = np.vstack([T2_bot, T2_top])

                try:
                    S_mat = A_mat @ np.linalg.solve(B_mat, T2)
                except np.linalg.LinAlgError:
                    S_mat = A_mat @ np.linalg.lstsq(B_mat, T2, rcond=None)[0]

                S11 = S_mat[:N, :N]
                S21 = S_mat[N:, :N]

                delta = np.zeros(N, dtype=complex)
                delta[inc_idx] = 1.0 + 0.0j

                lhs_t = Y1 @ S11 + S21
                try:
                    t_TM = np.linalg.solve(lhs_t, 2.0 * (Y1 @ delta))
                except np.linalg.LinAlgError:
                    t_TM = np.linalg.lstsq(lhs_t, 2.0 * (Y1 @ delta), rcond=None)[0]
                r_TM = S11 @ t_TM - delta

                kz1_inc = kz1[inc_idx]
                eff_t_TM = np.abs(t_TM) ** 2 * np.real(kz2 / (kz1_inc + 1e-30))
                eff_r_TM = np.abs(r_TM) ** 2 * np.real(kz1 / (kz1_inc + 1e-30))
                result["t_TM"] = t_TM
                result["r_TM"] = r_TM
                result["eff_trans_TM"] = eff_t_TM
                result["eff_reflect_TM"] = eff_r_TM

        return result


# =============================================================================
# RCWA 辅助: 傅里叶展开与 Toeplitz 矩阵
# =============================================================================
def _toeplitz_epsilon_1d(
    eps_a: complex, eps_b: complex, fill: float, truncation_size: int
) -> np.ndarray:
    """
    介电函数 ε(x) 的傅里叶展开 Toeplitz 矩阵

    truncation_size = 2M+1 (矩阵阶数 = 截断级次总数)
    矩阵 [F]_{i,j} = ε̂_{m_i - m_j}， 其中 m_i, m_j ∈ [-M, M]
    因此级次差 k = m_i - m_j ∈ [-2M, 2M]，需要计算 4M+1 个傅里叶系数。

    ε(x) = ε_b + (ε_a - ε_b) * rect(x / (fill·Λ))
    ε̂_k = (ε_a - ε_b) · sin(π·k·fill) / (π·k), k ≠ 0
    ε̂_0 = fill·ε_a + (1-fill)·ε_b
    """
    M = (truncation_size - 1) // 2
    # 需要所有 k ∈ [-2M, 2M] 的傅里叶系数
    k_all = np.arange(-2 * M, 2 * M + 1)
    coeffs_extended = np.zeros(4 * M + 1, dtype=complex)
    zero_mask = k_all == 0
    coeffs_extended[zero_mask] = fill * eps_a + (1 - fill) * eps_b
    non_zero = ~zero_mask
    k_nz = k_all[non_zero]
    coeffs_extended[non_zero] = (
        (eps_a - eps_b) * np.sin(np.pi * k_nz * fill) / (np.pi * k_nz)
    )
    # 首行首列：
    #   第一行 i=0 (对应 m=-M)，j∈[0,2M]：k=(-M) - m_j = (-M) - (j-M) = -j
    #   第一列 j=0 (对应 m=-M)，i∈[0,2M]：k=m_i - (-M) = (i-M) - (-M) = i
    c = coeffs_extended[2 * M : 2 * M + truncation_size]  # k=0,...,2M
    r = coeffs_extended[2 * M :: -1]  # k=0,-1,-2,...,-2M
    try:
        from scipy.linalg import toeplitz
        return toeplitz(c, r)
    except Exception:
        # 退化实现（纯 numpy）
        idx = np.arange(truncation_size)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        diff = ii - jj  # k  ∈ [-2M, 2M]
        return coeffs_extended[diff + 2 * M]


def _toeplitz_inverse_epsilon_1d(
    eps_a: complex, eps_b: complex, fill: float, truncation_size: int
) -> np.ndarray:
    """
    1/ε(x) 的傅里叶展开矩阵（Li 反演规则，TM 偏振必需）

    使用与 _toeplitz_epsilon_1d 相同的扩展级次差范围 [-2M, 2M] 的傅里叶系数。
    """
    inv_a = 1.0 / complex(eps_a) if abs(eps_a) > 1e-30 else 0.0
    inv_b = 1.0 / complex(eps_b) if abs(eps_b) > 1e-30 else 0.0
    return _toeplitz_epsilon_1d(inv_a, inv_b, fill, truncation_size)


@jit(nopython=True, cache=True)
def _safe_kz(kx: np.ndarray, nk0: complex) -> np.ndarray:
    """k_z = sqrt((n k0)^2 - kx^2)，取 Im(kz)>=0 分支"""
    out = np.empty(kx.shape, dtype=np.complex128)
    nk0_2 = complex(nk0) * complex(nk0)
    for i in range(kx.shape[0]):
        kz2 = nk0_2 - kx[i] * kx[i]
        re = np.real(kz2)
        im = np.imag(kz2)
        r = np.sqrt(re * re + im * im)
        real_part = np.sqrt(0.5 * (r + re))
        imag_part = 0.5 * im / real_part if real_part > 1e-30 else np.sqrt(r)
        if imag_part < 0:
            real_part = -real_part
            imag_part = -imag_part
        out[i] = complex(real_part, imag_part)
    return out


# =============================================================================
# 从远场衍射级次重建 2D 空间像
# =============================================================================
def _rcwa_diffraction_to_aerial(
    mask: np.ndarray,
    rcwa_result: Dict[str, np.ndarray],
    optics: OpticalSystem,
    polarization: Polarization,
    pixel_size_nm: float,
    rcwa_cfg: RCWAConfig,
) -> np.ndarray:
    """
    将 RCWA 求解得到的远场衍射级次，通过投影物镜光瞳传播到晶圆面，
    并对标量成像框架的输出做矢量修正。

    实现思路：
    1. 用标量 Hopkins 模型作为基础；
    2. 在频域对每个级次按 RCWA 计算的效率乘以矢量权重；
    3. 对 TE/TM 两种偏振按配置合成。
    """
    ny, nx = mask.shape
    imaging = PartialCoherentImaging(optics, (ny, nx))
    base_aerial = imaging.compute_aerial_image(mask)

    # --- 矢量修正因子 ---
    orders = rcwa_result["orders"]
    if polarization == Polarization.TE:
        eff = rcwa_result["eff_trans_TE"]
    elif polarization == Polarization.TM:
        eff = rcwa_result["eff_trans_TM"]
    else:
        eff = 0.5 * (rcwa_result["eff_trans_TE"] + rcwa_result["eff_trans_TM"])

    # 归一化到最大 1 作为乘性修正
    eff_max = float(np.nanmax(eff))
    if eff_max <= 0:
        return base_aerial
    norm_eff = eff / eff_max

    # 将一维级次权重扩展到 2D 频率平面，按 |sinθ_x| = m·λ/Λ 匹配
    # 这里只做一个近似的乘性修正：
    # 对基础频谱 |M(f)| 乘以对应级次权重
    period_nm = float(rcwa_cfg.period_nm or _estimate_period(mask, pixel_size_nm))
    if period_nm <= 0:
        return base_aerial

    fx = np.fft.fftfreq(nx, pixel_size_nm)
    fy = np.fft.fftfreq(ny, pixel_size_nm)
    FX, FY = np.meshgrid(fx, fy)
    # x 方向的空间频率对应级次: m = round( fx * Λ )
    m_idx = np.round(FX * period_nm).astype(np.int64)
    M = rcwa_cfg.n_orders
    m_clipped = np.clip(m_idx, -M, M) + M  # -> 0..2M

    weight_2d = norm_eff[m_clipped.ravel()].reshape(ny, nx)
    # 对低空间频率（< λ/4NA）减弱修正幅度，防止过度修正
    cutoff = optics.cutoff_frequency
    radial = np.sqrt(FX ** 2 + FY ** 2) / (cutoff + 1e-12)
    blend = np.clip(radial, 0.0, 1.0)  # 0=DC, 1=边缘
    blend = blend ** 2
    final_weight = 1.0 + blend * (weight_2d - 1.0)

    # 对 aerial 做乘性修正并再次归一化
    corrected = base_aerial * np.fft.ifftshift(final_weight)
    corrected = np.maximum(corrected, 0.0)
    m = float(np.nanmax(corrected))
    if m > 0:
        corrected = corrected / m
    return corrected.astype(np.float64)


def _estimate_period(mask: np.ndarray, pixel_size_nm: float) -> float:
    """粗略估计一维 line/space 结构的周期 (nm)"""
    try:
        prof = np.mean(mask, axis=0)
        prof = prof - prof.mean()
        acf = np.correlate(prof, prof, mode="full")[len(prof) - 1 :]
        # 找第一个峰值位置（跳过 0 位移）
        if acf.size < 3:
            return 0.0
        diff = np.diff(np.sign(np.diff(acf)))
        peaks = np.where(diff < 0)[0] + 1
        if len(peaks) == 0:
            return 0.0
        first = peaks[0] if peaks[0] > 1 else (peaks[1] if len(peaks) > 1 else 0)
        return float(first) * pixel_size_nm if first > 0 else 0.0
    except Exception:
        return 0.0


# =============================================================================
# meent / rcwa 等外部开源库封装（可选）
# =============================================================================
def _try_solve_with_meent(
    mask: np.ndarray, optics: OpticalSystem, cfg: RCWAConfig
) -> Optional[Dict[str, np.ndarray]]:
    """尝试使用 meent 开源 RCWA 库（如果已安装）求解"""
    if not cfg.use_meent_if_available:
        return None
    try:
        # 只做软导入，未安装则降级
        import meent  # type: ignore  # noqa

        warnings.warn(
            "检测到 meent 库，但当前版本尚未提供细粒度 API 封装，"
            "已回退到内置简化 RCWA 实现。可手动扩展本函数对接。",
            stacklevel=2,
        )
        return None
    except Exception:
        return None


# =============================================================================
# FDTD 占位求解器
# =============================================================================
class _FDTDPlaceholderSolver:
    """
    FDTD 占位实现。

    当用户未安装 meep 等 FDTD 仿真库时，退化为：
        1. 对标量 Hopkins 结果做一个 phenomenological 矢量修正；
        2. 在 extra 字段中标记 "fdtd_fallback=True"。
    """

    def __init__(self, config: FDTDConfig):
        self.cfg = config
        self._meep_available = False
        try:
            import meep  # type: ignore  # noqa
            self._meep_available = True
        except Exception:
            self._meep_available = False

    def simulate_aerial(
        self, mask: np.ndarray, optics: OpticalSystem
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        ny, nx = mask.shape
        imaging = PartialCoherentImaging(optics, (ny, nx))
        base = imaging.compute_aerial_image(mask)

        extra: Dict[str, Any] = {"fdtd_fallback": not self._meep_available}
        if not self._meep_available:
            logger.warning(
                "未检测到 meep 库，FDTD 后端使用标量 Hopkins 结果占位。"
                "请 pip install meep 以启用严格 FDTD 求解。"
            )
            return base, extra
        # TODO: 真正的 meep 3D 掩模结构建模、照明光源注入、光瞳滤波等
        return base, extra


# =============================================================================
# 统一仿真入口
# =============================================================================
def simulate(
    mask: np.ndarray,
    backend: Union[str, SimulationBackend] = SimulationBackend.HOPKINS,
    optical_system: Optional[OpticalSystem] = None,
    threshold: float = 0.3,
    apply_resist: bool = True,
    dose: float = 1.0,
    resist_model: Optional[ResistModel] = None,
    pixel_size_nm: Optional[float] = None,
    rcwa_config: Optional[RCWAConfig] = None,
    fdtd_config: Optional[FDTDConfig] = None,
) -> SimulationResult:
    """
    统一光刻仿真入口

    Args:
        mask: 2D 掩模图案（值范围 [0,1]）
        backend: 'hopkins' (默认) | 'rcwa' | 'fdtd'
        optical_system: 光学系统参数；None 使用默认 193nm 浸没式
        threshold: 光刻胶阈值（当 resist_model=None 时生效）
        apply_resist: 是否应用光刻胶阈值模型
        dose: 归一化曝光剂量
        resist_model: 高级光刻胶模型配置
        pixel_size_nm: 掩模像素尺寸 (nm)；None 则取自 optical_system.pixel_size
        rcwa_config: RCWA 求解器参数
        fdtd_config: FDTD 求解器参数

    Returns:
        SimulationResult: 包含 aerial_image、wafer_image、backend 等
    """
    import time

    t0 = time.perf_counter()
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim != 2:
        raise ValueError(f"mask 必须是 2D 数组，当前 shape={mask.shape}")

    optics = optical_system or OpticalSystem()
    ps_nm = float(pixel_size_nm if pixel_size_nm is not None else optics.pixel_size)

    be = SimulationBackend(backend) if isinstance(backend, str) else backend

    extra: Dict[str, Any] = {}
    aerial: Optional[np.ndarray] = None

    if be == SimulationBackend.HOPKINS:
        imaging = PartialCoherentImaging(optics, mask.shape)
        aerial = imaging.compute_aerial_image(mask)

    elif be == SimulationBackend.RCWA:
        cfg = rcwa_config or RCWAConfig()
        # 自动估计周期 / 占空比
        period = float(cfg.period_nm or _estimate_period(mask, ps_nm))
        if period <= 0:
            logger.warning(
                "RCWA 后端未能从掩模中估计出线/空周期，"
                "将退化为标量 Hopkins 结果（无矢量修正）。"
            )
            extra["rcwa_warning"] = "period_estimation_failed"
            aerial = PartialCoherentImaging(optics, mask.shape).compute_aerial_image(mask)
        else:
            # 粗估计线宽 = 周期 × mask 平均覆盖率
            fill = float(np.clip(np.mean(mask), 0.05, 0.95))
            line = float(cfg.line_width_nm or (period * fill))
            duty = float(np.clip(line / period, 0.05, 0.95))
            extra["rcwa_period_nm"] = period
            extra["rcwa_duty_cycle"] = duty

            # 优先尝试外部库
            far_field = _try_solve_with_meent(mask, optics, cfg)
            if far_field is None:
                solver = RCWASolver1D(cfg)
                far_field = solver.solve_far_field(
                    wavelength_nm=optics.wavelength,
                    period_nm=period,
                    duty_cycle=duty,
                    theta_deg=0.0,
                )
            extra["diffraction_orders"] = far_field
            aerial = _rcwa_diffraction_to_aerial(
                mask, far_field, optics, cfg.polarization, ps_nm, cfg
            )

    elif be == SimulationBackend.FDTD:
        cfg = fdtd_config or FDTDConfig()
        solver = _FDTDPlaceholderSolver(cfg)
        aerial, fdtd_extra = solver.simulate_aerial(mask, optics)
        extra.update(fdtd_extra)

    else:  # pragma: no cover
        raise ValueError(f"未知仿真后端: {backend}")

    # 剂量调制
    if dose != 1.0:
        aerial = np.clip(aerial * dose, 0.0, 1.0)

    # 光刻胶模型
    if resist_model is not None:
        wafer = apply_resist_model(aerial, resist_model=resist_model)
    elif apply_resist:
        wafer = _apply_threshold(aerial, threshold)
    else:
        wafer = aerial.copy()

    runtime = float(time.perf_counter() - t0)
    return SimulationResult(
        aerial_image=aerial.astype(np.float64),
        wafer_image=wafer.astype(np.float64),
        backend=be,
        runtime_sec=runtime,
        extra=extra,
    )


# =============================================================================
# 多工艺条件联合仿真（对标 simulate_multi_process）
# =============================================================================
def simulate_multi_process_unified(
    mask: np.ndarray,
    conditions: List[ProcessCondition],
    backend: Union[str, SimulationBackend] = SimulationBackend.HOPKINS,
    base_optics: Optional[OpticalSystem] = None,
    threshold: float = 0.3,
    apply_resist: bool = True,
    resist_model: Optional[ResistModel] = None,
    rcwa_config: Optional[RCWAConfig] = None,
    fdtd_config: Optional[FDTDConfig] = None,
) -> MultiProcessSimulationResult:
    """
    多工艺条件联合仿真（统一后端版本）

    Args:
        mask: 掩模图案
        conditions: 工艺条件列表
        backend: 仿真后端
        base_optics: 基础光学系统
        threshold: 光刻胶阈值
        apply_resist: 是否应用光刻胶
        resist_model: 高级光刻胶模型
        rcwa_config: RCWA 配置
        fdtd_config: FDTD 配置

    Returns:
        MultiProcessSimulationResult（与 imaging.simulate_multi_process 保持一致）
    """
    if backend == SimulationBackend.HOPKINS:
        return simulate_multi_process(
            mask, conditions, base_optics=base_optics,
            threshold=threshold, apply_resist=apply_resist,
            resist_model=resist_model,
        )

    base = base_optics or OpticalSystem()
    aerial_list: List[np.ndarray] = []
    wafer_list: List[np.ndarray] = []
    for cond in conditions:
        optics = cond.to_optical_system(base_optics=base)
        res = simulate(
            mask, backend=backend, optical_system=optics,
            threshold=threshold, apply_resist=apply_resist,
            dose=cond.dose, resist_model=resist_model,
            rcwa_config=rcwa_config, fdtd_config=fdtd_config,
        )
        aerial_list.append(res.aerial_image)
        wafer_list.append(res.wafer_image)

    return MultiProcessSimulationResult(
        aerial_images=aerial_list,
        wafer_images=wafer_list,
        conditions=conditions,
        threshold=threshold,
    )


# =============================================================================
# 标量 vs 矢量 精度对比工具
# =============================================================================
@dataclass
class BackendComparisonReport:
    """标量 (Hopkins) vs 矢量 (RCWA/FDTD) 精度对比报告"""
    mask_name: str
    backend_scalar: SimulationBackend
    backend_vector: SimulationBackend
    pixel_size_nm: float
    cd_target_nm: Optional[float]

    # 晶圆图指标对比
    cd_scalar: Dict[str, float] = field(default_factory=dict)
    cd_vector: Dict[str, float] = field(default_factory=dict)
    cd_error: Dict[str, float] = field(default_factory=dict)   # vector - scalar

    epe_scalar: Dict[str, float] = field(default_factory=dict)
    epe_vector: Dict[str, float] = field(default_factory=dict)
    epe_error: Dict[str, float] = field(default_factory=dict)  # vector - scalar

    # 空间像指标对比
    ils_scalar: Dict[str, float] = field(default_factory=dict)
    ils_vector: Dict[str, float] = field(default_factory=dict)
    nils_scalar: Dict[str, float] = field(default_factory=dict)
    nils_vector: Dict[str, float] = field(default_factory=dict)

    # 图像级差异
    image_mse: float = 0.0
    image_max_abs_diff: float = 0.0
    wafer_jaccard: float = 0.0

    # 运行时间
    runtime_scalar_sec: float = 0.0
    runtime_vector_sec: float = 0.0

    # 矢量后端附加信息
    vector_extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask": self.mask_name,
            "backends": {
                "scalar": self.backend_scalar.value,
                "vector": self.backend_vector.value,
            },
            "pixel_size_nm": self.pixel_size_nm,
            "cd_target_nm": self.cd_target_nm,
            "cd": {
                "scalar": self.cd_scalar,
                "vector": self.cd_vector,
                "delta_vector_minus_scalar": self.cd_error,
            },
            "epe": {
                "scalar": self.epe_scalar,
                "vector": self.epe_vector,
                "delta_vector_minus_scalar": self.epe_error,
            },
            "ils": {
                "scalar": self.ils_scalar,
                "vector": self.ils_vector,
            },
            "nils": {
                "scalar": self.nils_scalar,
                "vector": self.nils_vector,
            },
            "image_level": {
                "aerial_mse": self.image_mse,
                "aerial_max_abs_diff": self.image_max_abs_diff,
                "wafer_jaccard": self.wafer_jaccard,
            },
            "runtime_sec": {
                "scalar": self.runtime_scalar_sec,
                "vector": self.runtime_vector_sec,
                "speedup_scalar_over_vector": (
                    self.runtime_scalar_sec / self.runtime_vector_sec
                    if self.runtime_vector_sec > 0 else float("nan")
                ),
            },
            "vector_extra": _json_safe(self.vector_extra),
        }

    def summary(self) -> str:
        lines = [
            "=" * 64,
            f"  标量({self.backend_scalar.value}) vs 矢量({self.backend_vector.value}) 对比报告",
            "=" * 64,
            f"Mask: {self.mask_name}    Pixel: {self.pixel_size_nm:.2f} nm    "
            f"CD target: {self.cd_target_nm or 'N/A'} nm",
            "-" * 64,
            f"[CD]  标量={self.cd_scalar.get('cd_mean', float('nan')):7.2f} nm   "
            f"矢量={self.cd_vector.get('cd_mean', float('nan')):7.2f} nm   "
            f"Δ={self.cd_error.get('cd_error_mean', float('nan')):+.2f} nm",
            f"[EPE] 标量={self.epe_scalar.get('epe_mean', float('nan')):7.2f} nm   "
            f"矢量={self.epe_vector.get('epe_mean', float('nan')):7.2f} nm   "
            f"Δ={self.epe_error.get('epe_mean', float('nan')):+.2f} nm",
            f"[ILS] 标量={self.ils_scalar.get('ils_mean', float('nan')):7.4f} 1/nm   "
            f"矢量={self.ils_vector.get('ils_mean', float('nan')):7.4f} 1/nm",
            f"[NILS]标量={self.nils_scalar.get('nils_mean', float('nan')):7.2f}   "
            f"矢量={self.nils_vector.get('nils_mean', float('nan')):7.2f}",
            "-" * 64,
            f"图像级: MSE={self.image_mse:.3e}   max|ΔI|={self.image_max_abs_diff:.3f}   "
            f"Wafer Jaccard={self.wafer_jaccard:.3f}",
            f"运行时间: 标量 {self.runtime_scalar_sec:.3f}s   矢量 {self.runtime_vector_sec:.3f}s   "
            f"加速比(scalar/vector)={self.runtime_scalar_sec / max(self.runtime_vector_sec, 1e-9):.2f}x",
            "=" * 64,
        ]
        return "\n".join(lines)


def _json_safe(obj: Any) -> Any:
    """递归转换 numpy / complex 对象为 JSON 可序列化类型"""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, Enum):
        return obj.value
    return str(obj) if obj is not None else None


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = (a >= 0.5).astype(bool)
    b_bin = (b >= 0.5).astype(bool)
    inter = int(np.sum(a_bin & b_bin))
    union = int(np.sum(a_bin | b_bin))
    return float(inter / union) if union > 0 else 1.0


def compare_backends(
    mask: np.ndarray,
    target_binary: Optional[np.ndarray] = None,
    backend_scalar: Union[str, SimulationBackend] = SimulationBackend.HOPKINS,
    backend_vector: Union[str, SimulationBackend] = SimulationBackend.RCWA,
    optical_system: Optional[OpticalSystem] = None,
    threshold: float = 0.3,
    apply_resist: bool = True,
    dose: float = 1.0,
    resist_model: Optional[ResistModel] = None,
    pixel_size_nm: Optional[float] = None,
    cd_target_nm: Optional[float] = None,
    rcwa_config: Optional[RCWAConfig] = None,
    fdtd_config: Optional[FDTDConfig] = None,
    mask_name: str = "unnamed_mask",
    edge_method: str = "morphological",
    cd_direction: str = "both",
) -> BackendComparisonReport:
    """
    在同一掩模下对比标量 vs 矢量后端的 CD/EPE/NILS 等指标差异

    Args:
        mask: 2D 掩模图案 [0,1]
        target_binary: 目标二值图（用于 EPE/CD-Error 参考）；None 则用标量结果 wafer 作参考
        backend_scalar: 标量后端，默认 'hopkins'
        backend_vector: 矢量后端，默认 'rcwa'
        optical_system: 光学系统
        threshold: 光刻胶阈值
        apply_resist: 是否应用光刻胶
        dose: 曝光剂量
        resist_model: 高级光刻胶模型
        pixel_size_nm: 像素尺寸
        cd_target_nm: 目标 CD (nm)；None 从 target_binary 或标量晶圆图估计
        rcwa_config: RCWA 配置
        fdtd_config: FDTD 配置
        mask_name: 报告中显示的掩模名称
        edge_method: EPE 边缘提取方法
        cd_direction: CD 测量方向

    Returns:
        BackendComparisonReport：可打印 summary() 或转字典用于导出
    """
    optics = optical_system or OpticalSystem()
    ps_nm = float(pixel_size_nm if pixel_size_nm is not None else optics.pixel_size)

    # 分别仿真
    res_s = simulate(
        mask, backend=backend_scalar,
        optical_system=optics, threshold=threshold, apply_resist=apply_resist,
        dose=dose, resist_model=resist_model, pixel_size_nm=ps_nm,
        rcwa_config=rcwa_config, fdtd_config=fdtd_config,
    )
    res_v = simulate(
        mask, backend=backend_vector,
        optical_system=optics, threshold=threshold, apply_resist=apply_resist,
        dose=dose, resist_model=resist_model, pixel_size_nm=ps_nm,
        rcwa_config=rcwa_config, fdtd_config=fdtd_config,
    )

    wafer_s = (res_s.wafer_image >= threshold).astype(np.float64)
    wafer_v = (res_v.wafer_image >= threshold).astype(np.float64)
    if target_binary is None:
        target_bin = wafer_s
    else:
        target_bin = (np.asarray(target_binary) >= 0.5).astype(np.float64)

    # CD 指标
    cd_s = compute_cd(wafer_s, direction=cd_direction, pixel_size=ps_nm,
                      threshold=threshold)
    cd_v = compute_cd(wafer_v, direction=cd_direction, pixel_size=ps_nm,
                      threshold=threshold)
    cd_err_v = compute_cd_error(wafer_v, target_bin, direction=cd_direction,
                                pixel_size=ps_nm)
    cd_err_s = compute_cd_error(wafer_s, target_bin, direction=cd_direction,
                                pixel_size=ps_nm)
    cd_delta = {
        k: cd_v.get(k, float("nan")) - cd_s.get(k, float("nan"))
        for k in ("cd_mean", "cd_min", "cd_max", "cd_std")
    }
    cd_delta["cd_error_mean"] = (
        cd_err_v.get("cd_error_mean", float("nan"))
        - cd_err_s.get("cd_error_mean", float("nan"))
    )

    # EPE 指标
    epe_s = compute_epe(wafer_s, target_bin, pixel_size=ps_nm,
                        edge_method=edge_method)
    epe_v = compute_epe(wafer_v, target_bin, pixel_size=ps_nm,
                        edge_method=edge_method)
    epe_delta = {k: epe_v[k] - epe_s[k] for k in epe_v}

    # ILS / NILS
    ils_s = compute_ils(res_s.aerial_image, threshold=threshold,
                        pixel_size=ps_nm)
    ils_v = compute_ils(res_v.aerial_image, threshold=threshold,
                        pixel_size=ps_nm)
    cd_t = float(
        cd_target_nm or cd_s.get("cd_mean", 0.0)
        or compute_cd(target_bin, pixel_size=ps_nm).get("cd_mean", 0.0)
    )
    nils_s = compute_nils(res_s.aerial_image, cd_t,
                          threshold=threshold, pixel_size=ps_nm) if cd_t > 0 else {}
    nils_v = compute_nils(res_v.aerial_image, cd_t,
                          threshold=threshold, pixel_size=ps_nm) if cd_t > 0 else {}

    # 图像级差异
    diff = np.abs(res_s.aerial_image - res_v.aerial_image)
    mse_val = float(np.mean(diff ** 2))
    max_abs = float(np.max(diff))
    jacc = _jaccard(wafer_s, wafer_v)

    be_scalar = (
        SimulationBackend(backend_scalar)
        if isinstance(backend_scalar, str) else backend_scalar
    )
    be_vector = (
        SimulationBackend(backend_vector)
        if isinstance(backend_vector, str) else backend_vector
    )

    return BackendComparisonReport(
        mask_name=mask_name,
        backend_scalar=be_scalar,
        backend_vector=be_vector,
        pixel_size_nm=ps_nm,
        cd_target_nm=cd_t if cd_target_nm is not None else None,
        cd_scalar=cd_s,
        cd_vector=cd_v,
        cd_error=cd_delta,
        epe_scalar=epe_s,
        epe_vector=epe_v,
        epe_error=epe_delta,
        ils_scalar=ils_s,
        ils_vector=ils_v,
        nils_scalar=nils_s,
        nils_vector=nils_v,
        image_mse=mse_val,
        image_max_abs_diff=max_abs,
        wafer_jaccard=jacc,
        runtime_scalar_sec=res_s.runtime_sec,
        runtime_vector_sec=res_v.runtime_sec,
        vector_extra=res_v.extra,
    )


# =============================================================================
# 便捷：生成结构化对比报告（Dict 列表，便于批量转 CSV / Excel）
# =============================================================================
def batch_compare_backends(
    masks: Dict[str, np.ndarray],
    **compare_kwargs,
) -> List[BackendComparisonReport]:
    """
    批量对比多个掩模的标量/矢量后端差异

    Args:
        masks: {mask_name: mask_2d_array} 字典
        **compare_kwargs: 透传给 compare_backends 的其他参数

    Returns:
        BackendComparisonReport 列表
    """
    reports = []
    for name, mask in masks.items():
        reports.append(compare_backends(mask=mask, mask_name=name, **compare_kwargs))
    return reports


def export_comparison_csv(
    reports: List[BackendComparisonReport],
    output_path: str,
) -> None:
    """将对比报告导出为 CSV 表格（每一行一个掩模）"""
    import csv

    rows = []
    header = None
    for r in reports:
        d = r.to_dict()
        flat: Dict[str, Any] = {
            "mask": d["mask"],
            "scalar_backend": d["backends"]["scalar"],
            "vector_backend": d["backends"]["vector"],
            "pixel_size_nm": d["pixel_size_nm"],
            "cd_target_nm": d["cd_target_nm"],
            "cd_scalar_mean": d["cd"]["scalar"].get("cd_mean"),
            "cd_vector_mean": d["cd"]["vector"].get("cd_mean"),
            "cd_delta_nm": d["cd"]["delta_vector_minus_scalar"].get("cd_error_mean"),
            "epe_scalar_mean": d["epe"]["scalar"].get("epe_mean"),
            "epe_vector_mean": d["epe"]["vector"].get("epe_mean"),
            "epe_delta_nm": d["epe"]["delta_vector_minus_scalar"].get("epe_mean"),
            "ils_scalar_mean": d["ils"]["scalar"].get("ils_mean"),
            "ils_vector_mean": d["ils"]["vector"].get("ils_mean"),
            "nils_scalar_mean": d["nils"]["scalar"].get("nils_mean"),
            "nils_vector_mean": d["nils"]["vector"].get("nils_mean"),
            "aerial_mse": d["image_level"]["aerial_mse"],
            "aerial_max_abs_diff": d["image_level"]["aerial_max_abs_diff"],
            "wafer_jaccard": d["image_level"]["wafer_jaccard"],
            "runtime_scalar_s": d["runtime_sec"]["scalar"],
            "runtime_vector_s": d["runtime_sec"]["vector"],
            "speedup": d["runtime_sec"]["speedup_scalar_over_vector"],
        }
        if header is None:
            header = list(flat.keys())
        rows.append(flat)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header or [])
        w.writeheader()
        for row in rows:
            w.writerow(row)


# =============================================================================
# 与 MaskOptimizer 集成的便捷包装
# =============================================================================
def make_simulate_fn_for_optimizer(
    backend: Union[str, SimulationBackend] = SimulationBackend.HOPKINS,
    **kwargs,
):
    """
    生成可直接赋值给自定义损失/评估流程的仿真函数。

    示例（在 MaskOptimizer 中使用矢量后端评估）::

        from core.rigorous_sim import make_simulate_fn_for_optimizer

        optimizer.config.simulation_backend = 'rcwa'
        sim_fn = make_simulate_fn_for_optimizer('rcwa', rcwa_config=RCWAConfig(...))
        result = sim_fn(mask, optical_system=optimizer.optical_system)
    """
    def _sim(mask: np.ndarray, optical_system: Optional[OpticalSystem] = None,
             threshold: float = 0.3, apply_resist: bool = True,
             dose: float = 1.0, resist_model: Optional[ResistModel] = None):
        return simulate(
            mask=mask, backend=backend,
            optical_system=optical_system,
            threshold=threshold, apply_resist=apply_resist,
            dose=dose, resist_model=resist_model,
            **kwargs,
        )
    return _sim


__all__ = [
    "SimulationBackend",
    "Polarization",
    "RCWAConfig",
    "FDTDConfig",
    "SimulationResult",
    "RCWASolver1D",
    "simulate",
    "simulate_multi_process_unified",
    "BackendComparisonReport",
    "compare_backends",
    "batch_compare_backends",
    "export_comparison_csv",
    "make_simulate_fn_for_optimizer",
]
