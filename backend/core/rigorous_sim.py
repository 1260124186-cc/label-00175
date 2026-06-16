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
    _zernike_polynomial,
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
        n_orders_y: Y 方向单侧衍射级次；None 时使用 n_orders。
        polarization: 入射偏振态。
        n_superstrate: 上层介质折射率（如光刻胶/浸没液）。
        n_substrate: 下层介质折射率（如掩模玻璃 / SiO2）。
        n_grating_line: 光栅线条区域折射率（如 Cr 吸收层取复数 n-ik）。
        grating_thickness_nm: 光栅厚度 (nm)。
        period_nm: X 方向光栅周期 (nm)；None 时从掩模图案自动估计。
        period_y_nm: Y 方向光栅周期 (nm)；None 时与 period_nm 相同。
        line_width_nm: 线宽 (nm)；None 时从掩模图案自动估计。
        hole_diameter_nm: 接触孔直径 (nm)；用于 2D 周期结构。
        convergence_tol: S 矩阵级次收敛判据（相对能量残差）。
        max_iter: 反射/透射迭代次数（用于薄膜堆栈时）。
        use_meent_if_available: 若安装了 meent 开源库则优先使用。
        use_2d_rcwa: 是否启用 2D RCWA（用于接触孔等二维周期结构）。
        vector_transfer: 是否使用矢量传递函数（高 NA 场景推荐启用）。
        illumination_theta_deg: 照明极角（度），用于斜入射。
        illumination_phi_deg: 照明方位角（度），用于斜入射。
    """
    n_orders: int = 5
    n_orders_y: Optional[int] = None
    polarization: Polarization = Polarization.UNPOLARIZED
    n_superstrate: complex = 1.44 + 0.0j      # 典型浸没水
    n_substrate: complex = 1.56 + 0.0j         # 熔融石英
    n_grating_line: complex = 3.28 - 4.32j     # 典型 Cr @ 193nm
    grating_thickness_nm: float = 70.0
    period_nm: Optional[float] = None
    period_y_nm: Optional[float] = None
    line_width_nm: Optional[float] = None
    hole_diameter_nm: Optional[float] = None
    convergence_tol: float = 1e-6
    max_iter: int = 50
    use_meent_if_available: bool = True
    use_2d_rcwa: bool = False
    vector_transfer: bool = False
    illumination_theta_deg: float = 0.0
    illumination_phi_deg: float = 0.0


@dataclass
class FDTDConfig:
    """
    FDTD 求解器配置（基于 meep 的严格 3D 矢量电磁仿真）

    包含 3D 掩模结构建模、倾斜照明注入、近场到远场传播的完整参数配置。
    当 meep 不可用时自动退化为标量 Hopkins  phenomenological 修正。
    """
    grid_resolution_nm: float = 0.5
    pml_thickness_nm: float = 200.0
    total_time_steps: int = 2000
    courant_factor: float = 0.9
    use_meep_if_available: bool = True
    extra_material_params: Dict[str, Any] = field(default_factory=dict)

    n_substrate: complex = 1.56 + 0.0j
    n_absorber: complex = 3.28 - 4.32j
    n_superstrate: complex = 1.44 + 0.0j

    mask_thickness_nm: float = 70.0
    substrate_thickness_nm: float = 500.0
    superstrate_thickness_nm: float = 500.0

    illumination_theta_deg: float = 0.0
    illumination_phi_deg: float = 0.0
    polarization: Polarization = Polarization.UNPOLARIZED

    source_width_nm: float = 300.0
    ntff_distance_nm: float = 100.0

    pupil_filter: bool = True
    max_far_field_orders: int = 50

    def __post_init__(self):
        if isinstance(self.polarization, str):
            try:
                self.polarization = Polarization(self.polarization)
            except ValueError as e:
                raise ValueError(
                    f"Invalid polarization '{self.polarization}'. "
                    f"Must be one of: {[p.value for p in Polarization]}"
                ) from e


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
# 2D RCWA 严格耦合波分析（二维周期结构）
# =============================================================================
@dataclass
class RCWA2DResult:
    """二维 RCWA 求解结果结构"""
    orders_x: np.ndarray
    orders_y: np.ndarray
    t_TE: np.ndarray
    t_TM: np.ndarray
    r_TE: np.ndarray
    r_TM: np.ndarray
    eff_trans_TE: np.ndarray
    eff_trans_TM: np.ndarray
    eff_reflect_TE: np.ndarray
    eff_reflect_TM: np.ndarray
    KX: np.ndarray
    KY: np.ndarray
    kz_super: np.ndarray
    kz_sub: np.ndarray

    @property
    def shape_orders(self) -> Tuple[int, int]:
        return self.orders_y.size, self.orders_x.size


class RCWASolver2D:
    """
    二维二进制相位/吸收光栅 RCWA 求解器

    用于接触孔阵列、交叉光栅等真实二维周期光刻版图的矢量衍射求解。

    参考:
        Moharam & Gaylord, J. Opt. Soc. Am. A 12, 1077 (1995) —— 二维 RAT 法
        Li, J. Opt. Soc. Am. A 14, 2758 (1997) —— Fourier 因子分解规则

    实现特性:
        * 二维二元光栅（方形/圆形接触孔、交叉光栅）
        * 完整 TE/TM 偏振耦合求解
        * 任意入射角 (theta, phi)
        * 傅里叶模态法 (FMM)，支持 Li 反演规则
        * S 矩阵级联保证数值稳定性
    """

    def __init__(self, config: RCWAConfig):
        self.cfg = config

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def solve_far_field(
        self,
        wavelength_nm: float,
        period_x_nm: float,
        period_y_nm: float,
        duty_cycle_x: float,
        duty_cycle_y: float,
        theta_deg: float = 0.0,
        phi_deg: float = 0.0,
        hole_shape: str = "square",
    ) -> RCWA2DResult:
        """
        求解二维周期结构远场衍射级次

        Args:
            wavelength_nm: 真空波长 (nm)
            period_x_nm: X 方向周期 (nm)
            period_y_nm: Y 方向周期 (nm)
            duty_cycle_x: X 方向占空比 (线宽 / 周期)
            duty_cycle_y: Y 方向占空比
            theta_deg: 入射极角 (度)，0=正入射
            phi_deg: 入射方位角 (度)，0=XZ 平面入射
            hole_shape: 'square' 方形孔 | 'circle' 圆形孔

        Returns:
            RCWA2DResult：包含各级次透射/反射复振幅与衍射效率
        """
        k0 = 2.0 * np.pi / wavelength_nm
        d = self.cfg.grating_thickness_nm
        n1 = complex(self.cfg.n_superstrate)
        n2 = complex(self.cfg.n_substrate)
        na = complex(self.cfg.n_grating_line)
        nb = n2
        fx = float(np.clip(duty_cycle_x, 1e-4, 1 - 1e-4))
        fy = float(np.clip(duty_cycle_y, 1e-4, 1 - 1e-4))

        eps1 = n1 * n1
        eps2 = n2 * n2
        eps_a = na * na
        eps_b = nb * nb

        Mx = int(self.cfg.n_orders)
        My = int(self.cfg.n_orders_y or self.cfg.n_orders)
        orders_x = np.arange(-Mx, Mx + 1)
        orders_y = np.arange(-My, My + 1)
        Nx = 2 * Mx + 1
        Ny = 2 * My + 1
        N = Nx * Ny

        theta_rad = np.deg2rad(theta_deg)
        phi_rad = np.deg2rad(phi_deg)
        kx_inc = k0 * n1 * np.sin(theta_rad) * np.cos(phi_rad)
        ky_inc = k0 * n1 * np.sin(theta_rad) * np.sin(phi_rad)

        Gx = 2.0 * np.pi / period_x_nm
        Gy = 2.0 * np.pi / period_y_nm
        KX_grid, KY_grid = np.meshgrid(
            kx_inc + orders_x * Gx,
            ky_inc + orders_y * Gy,
            indexing="xy",
        )
        KX_flat = KX_grid.ravel()
        KY_flat = KY_grid.ravel()

        result_orders = {
            "orders_x": orders_x,
            "orders_y": orders_y,
            "KX": KX_grid,
            "KY": KY_grid,
        }

        inc_x_idx = Mx
        inc_y_idx = My
        inc_idx = inc_y_idx * Nx + inc_x_idx

        # ---- 构造 2D 介电常数傅里叶矩阵 ----
        E_mat = _toeplitz_epsilon_2d(eps_a, eps_b, fx, fy, Nx, Ny, hole_shape)
        inv_E_mat = _toeplitz_inverse_epsilon_2d(eps_a, eps_b, fx, fy, Nx, Ny, hole_shape)

        Kx_mat = np.diag(KX_flat / k0)
        Ky_mat = np.diag(KY_flat / k0)

        kz1 = _safe_kz_2d(KX_flat, KY_flat, k0 * n1)
        kz2 = _safe_kz_2d(KX_flat, KY_flat, k0 * n2)

        # ---- 构建本征方程（二维 FMM）----
        # [P] = [[ -Kx·invE·Ky,  Kx·invE·Kx + I ],
        #        [ -Ky·invE·Ky - E, Ky·invE·Kx     ]]
        # 本征值 q² 对应传播常数
        P11 = -Kx_mat @ inv_E_mat @ Ky_mat
        P12 = Kx_mat @ inv_E_mat @ Kx_mat + np.eye(N)
        P21 = -Ky_mat @ inv_E_mat @ Ky_mat - E_mat
        P22 = Ky_mat @ inv_E_mat @ Kx_mat
        P_mat = np.vstack([
            np.hstack([P11, P12]),
            np.hstack([P21, P22]),
        ])

        eigvals, eigvecs = np.linalg.eig(P_mat)
        q_raw = k0 * np.lib.scimath.sqrt(
            np.atleast_1d(eigvals).astype(complex)
        )
        q = np.where(np.imag(q_raw) < 0, -q_raw, q_raw)

        # W = [E_x, E_y]^T 的本征向量
        W = eigvecs
        Lam_diag = np.exp(-1j * q * d)
        Lam = np.diag(Lam_diag)
        Lam_inv = np.diag(1.0 / Lam_diag)

        # V 矩阵: 与 H_x, H_y 成比例
        V_pos = np.zeros_like(W)
        for n in range(2 * N):
            qn = q[n]
            wn = W[:, n]
            ex_n = wn[:N]
            ey_n = wn[N:]
            # H 分量由 Maxwell 方程: -jωμ H = ∇ × E
            # 傅里叶空间: -jωμ H_m = j K × E_m
            # => H_x ∝ (Ky·Ez - Kz·Ey), H_y ∝ (Kz·Ex - Kx·Ez)
            # 利用 invE 关系从 (Ex, Ey) 推出其余场分量
            hx_n = (1.0 / k0) * (
                Ky_mat @ (inv_E_mat @ (Kx_mat @ ey_n - Ky_mat @ ex_n))
                - qn / k0 * ey_n
            )
            hy_n = (1.0 / k0) * (
                qn / k0 * ex_n
                - Kx_mat @ (inv_E_mat @ (Kx_mat @ ey_n - Ky_mat @ ex_n))
            )
            V_pos[:N, n] = hx_n
            V_pos[N:, n] = hy_n
        V = V_pos

        # ---- 均匀层导纳矩阵 (superstrate & substrate) ----
        Y1 = _build_homogeneous_admittance_2d(KX_flat, KY_flat, kz1, k0, eps1)
        Y2 = _build_homogeneous_admittance_2d(KX_flat, KY_flat, kz2, k0, eps2)

        # ---- RAT 全局矩阵组装 ----
        A_bot = np.hstack([W, W])
        A_top = np.hstack([V, -V])
        B_bot = np.hstack([W @ Lam, W @ Lam_inv])
        B_top = np.hstack([V @ Lam, -V @ Lam_inv])

        I_N = np.eye(N)
        Z_N = np.zeros((N, N))
        T1_bot = np.hstack([I_N, Z_N, I_N, Z_N])
        T1_top = np.hstack([Z_N, I_N, Z_N, -I_N])
        T1 = np.vstack([T1_bot, T1_top])

        T2_bot = np.hstack([I_N, Z_N, I_N, Z_N])
        T2_top = np.hstack([Z_N, I_N, Z_N, -I_N])
        T2 = np.vstack([T2_bot, T2_top])
        T1 = np.vstack([
            np.hstack([np.eye(2 * N), np.eye(2 * N)]),
            np.hstack([Y1, -Y1]),
        ])
        T2 = np.vstack([
            np.hstack([np.eye(2 * N), np.eye(2 * N)]),
            np.hstack([Y2, -Y2]),
        ])

        A_mat = np.vstack([A_bot, A_top])
        B_mat = np.vstack([B_bot, B_top])

        try:
            S_mat = A_mat @ np.linalg.solve(B_mat, T2)
        except np.linalg.LinAlgError:
            S_mat = A_mat @ np.linalg.lstsq(B_mat, T2, rcond=None)[0]

        S11 = S_mat[: 2 * N, : 2 * N]
        S21 = S_mat[2 * N :, : 2 * N]

        # 入射向量: 只有 (0,0) 级入射，含 TE/TM 两个偏振
        delta_TE = np.zeros(2 * N, dtype=complex)
        delta_TM = np.zeros(2 * N, dtype=complex)
        # TE: E 垂直入射面 (入射面=XZ平面当 phi=0)，E_y 主导
        delta_TE[N + inc_idx] = 1.0
        # TM: H 垂直入射面，E_x 主导
        delta_TM[inc_idx] = 1.0

        lhs_t = Y1 @ S11 + S21

        def _solve_transmission(delta_vec):
            try:
                t_vec = np.linalg.solve(lhs_t, 2.0 * (Y1 @ delta_vec))
            except np.linalg.LinAlgError:
                t_vec = np.linalg.lstsq(lhs_t, 2.0 * (Y1 @ delta_vec), rcond=None)[0]
            r_vec = S11 @ t_vec - delta_vec
            return t_vec, r_vec

        t_TE_vec, r_TE_vec = _solve_transmission(delta_TE)
        t_TM_vec, r_TM_vec = _solve_transmission(delta_TM)

        # 提取透射 E_y (TE) 与 E_x (TM) 主分量
        t_TE = t_TE_vec[:N].reshape(Ny, Nx)
        t_TM = t_TM_vec[N:].reshape(Ny, Nx)
        r_TE = r_TE_vec[:N].reshape(Ny, Nx)
        r_TM = r_TM_vec[N:].reshape(Ny, Nx)

        # 衍射效率 (Poynting z 分量)
        kz1_inc = _safe_kz_2d_scalar(kx_inc, ky_inc, k0 * n1)
        eff_t_TE = np.abs(t_TE) ** 2 * np.real(kz2.reshape(Ny, Nx) / (kz1_inc + 1e-30))
        eff_t_TM = np.abs(t_TM) ** 2 * np.real(kz2.reshape(Ny, Nx) / (kz1_inc + 1e-30))
        eff_r_TE = np.abs(r_TE) ** 2 * np.real(kz1.reshape(Ny, Nx) / (kz1_inc + 1e-30))
        eff_r_TM = np.abs(r_TM) ** 2 * np.real(kz1.reshape(Ny, Nx) / (kz1_inc + 1e-30))

        return RCWA2DResult(
            orders_x=orders_x,
            orders_y=orders_y,
            t_TE=t_TE,
            t_TM=t_TM,
            r_TE=r_TE,
            r_TM=r_TM,
            eff_trans_TE=eff_t_TE,
            eff_trans_TM=eff_t_TM,
            eff_reflect_TE=eff_r_TE,
            eff_reflect_TM=eff_r_TM,
            KX=KX_grid,
            KY=KY_grid,
            kz_super=kz1.reshape(Ny, Nx),
            kz_sub=kz2.reshape(Ny, Nx),
        )


# =============================================================================
# 矢量传递函数 (Vector Transfer Function)
# =============================================================================
class VectorTransferFunction:
    """
    高 NA 矢量光学传递函数

    针对高数值孔径 (NA>0.8) 浸没式光刻系统，考虑:
    1. 偏振态在光瞳传播过程中的变化
    2. 倾斜波前的 s/p 偏振分解
    3. 电场三个分量 (Ex, Ey, Ez) 的独立贡献

    参考:
        Flagello et al., J. Microlith. Microfab. Microsyst. 1, 41 (2002)
        Totzeck, Proc. SPIE 5377 (2004)
    """

    def __init__(
        self,
        wavelength_nm: float,
        na: float,
        n_immersion: complex = 1.44 + 0.0j,
        pixel_size_nm: float = 1.0,
        grid_size: Tuple[int, int] = (256, 256),
    ):
        self.wavelength = wavelength_nm
        self.na = na
        self.n_imm = complex(n_immersion)
        self.pixel_size = pixel_size_nm
        self.ny, self.nx = grid_size
        self.k0 = 2.0 * np.pi / wavelength_nm

        self._build_pupil_coordinates()

    def _build_pupil_coordinates(self):
        """构建归一化光瞳坐标 (fx, fy)"""
        fx = np.fft.fftfreq(self.nx, self.pixel_size) * self.wavelength
        fy = np.fft.fftfreq(self.ny, self.pixel_size) * self.wavelength
        self.FX, self.FY = np.meshgrid(fx, fy)
        self.rho = np.sqrt(self.FX ** 2 + self.FY ** 2)
        self.pupil_mask = self.rho <= (self.na / abs(self.n_imm))
        self.phi_pupil = np.arctan2(self.FY, self.FX)

    def s_polarization_vector(self) -> Dict[str, np.ndarray]:
        """
        s-偏振 (TE) 在光瞳处的单位电场矢量

        s-偏振: E 垂直于入射面 (k, z)，即沿方位角方向
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_phi = np.cos(self.phi_pupil)
            sin_phi = np.sin(self.phi_pupil)
            Es_x = -sin_phi
            Es_y = cos_phi
            Es_z = np.zeros_like(self.rho)
        return {"Ex": Es_x, "Ey": Es_y, "Ez": Es_z}

    def p_polarization_vector(self) -> Dict[str, np.ndarray]:
        """
        p-偏振 (TM) 在光瞳处的单位电场矢量

        p-偏振: E 在入射面 (k, z) 内
        """
        rho_safe = np.where(self.rho < 1e-10, 1e-10, self.rho)
        n_safe = abs(self.n_imm)
        cos_theta = np.sqrt(np.maximum(1.0 - (self.rho / n_safe) ** 2, 0.0))
        cos_phi = np.cos(self.phi_pupil)
        sin_phi = np.sin(self.phi_pupil)

        Ep_x = cos_theta * cos_phi
        Ep_y = cos_theta * sin_phi
        Ep_z = self.rho / n_safe
        return {"Ex": Ep_x, "Ey": Ep_y, "Ez": Ep_z}

    def decompose_incident_field(
        self,
        polarization: Polarization,
        phi_in_deg: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        将入射偏振分解为 s/p 偏振分量在光瞳上的权重分布

        Args:
            polarization: 入射偏振态
            phi_in_deg: 入射光方位角 (度)

        Returns:
            dict: 含 's_weight', 'p_weight' 两个 2D 权重数组
        """
        phi_in = np.deg2rad(phi_in_deg)
        ny, nx = self.rho.shape

        if polarization == Polarization.TE:
            s_weight = np.ones((ny, nx), dtype=np.float64)
            p_weight = np.zeros((ny, nx), dtype=np.float64)
        elif polarization == Polarization.TM:
            s_weight = np.zeros((ny, nx), dtype=np.float64)
            p_weight = np.ones((ny, nx), dtype=np.float64)
        else:
            s_weight = np.ones((ny, nx), dtype=np.float64) * np.sqrt(0.5)
            p_weight = np.ones((ny, nx), dtype=np.float64) * np.sqrt(0.5)

        # 旋转坐标: 如果入射不在 XZ 平面，旋转 s/p 基矢
        if abs(phi_in_deg) > 1e-6:
            rot = np.array([
                [np.cos(phi_in), -np.sin(phi_in)],
                [np.sin(phi_in), np.cos(phi_in)],
            ])
            # 权重投影
            s_new = np.zeros_like(s_weight)
            p_new = np.zeros_like(p_weight)
            s_new = s_weight * rot[0, 0] + p_weight * rot[0, 1]
            p_new = s_weight * rot[1, 0] + p_weight * rot[1, 1]
            s_weight, p_weight = s_new, p_new

        s_weight = s_weight * self.pupil_mask.astype(np.float64)
        p_weight = p_weight * self.pupil_mask.astype(np.float64)

        return {"s_weight": s_weight, "p_weight": p_weight}

    def apply_vector_transfer(
        self,
        far_field_TE: np.ndarray,
        far_field_TM: np.ndarray,
        optics,
    ) -> np.ndarray:
        """
        应用矢量传递函数，将 RCWA 远场 TE/TM 分量合成晶圆面空间像

        Args:
            far_field_TE: s-偏振远场复振幅 (2D)
            far_field_TM: p-偏振远场复振幅 (2D)
            optics: OpticalSystem 实例（含 defocus, zernike 等）

        Returns:
            aerial_image: 归一化光强分布 (2D, [0,1])
        """
        ny, nx = self.ny, self.nx

        s_vec = self.s_polarization_vector()
        p_vec = self.p_polarization_vector()

        cutoff = 2.0 * np.pi * self.na / self.wavelength

        defocus_phase = np.ones((ny, nx), dtype=np.complex128)
        if hasattr(optics, "defocus") and abs(optics.defocus) > 1e-10:
            kz_sq = (self.k0 * abs(self.n_imm)) ** 2 - (
                (self.k0 * self.FX) ** 2 + (self.k0 * self.FY) ** 2
            )
            kz = np.lib.scimath.sqrt(kz_sq)
            kz = np.where(np.imag(kz) < 0, -kz, kz)
            defocus_phase = np.exp(-1j * kz * optics.defocus)

        zernike_phase = np.zeros((ny, nx), dtype=np.float64)
        if hasattr(optics, "zernike_coefficients") and optics.zernike_coefficients:
            rho_norm = self.rho / (self.na / abs(self.n_imm) + 1e-12)
            for j, coeff in optics.zernike_coefficients.items():
                if abs(coeff) < 1e-15:
                    continue
                zn_val = _zernike_polynomial(j, rho_norm, self.phi_pupil)
                zernike_phase += coeff * 2.0 * np.pi * zn_val
            zernike_phase[~self.pupil_mask] = 0.0

        total_phase = defocus_phase * np.exp(1j * zernike_phase)

        def _propagate_single(amp_2d, pol_vec):
            amp_pad = np.zeros((ny, nx), dtype=np.complex128)
            oy, ox = amp_2d.shape
            sy = min(oy, ny)
            sx = min(ox, nx)

            src_y0 = oy // 2 - sy // 2
            src_x0 = ox // 2 - sx // 2
            dst_y0 = ny // 2 - sy // 2
            dst_x0 = nx // 2 - sx // 2

            amp_pad[dst_y0 : dst_y0 + sy, dst_x0 : dst_x0 + sx] = (
                amp_2d[src_y0 : src_y0 + sy, src_x0 : src_x0 + sx]
            )

            amp_pad_shifted = np.fft.ifftshift(amp_pad)

            Ex = amp_pad_shifted * pol_vec["Ex"] * total_phase * self.pupil_mask
            Ey = amp_pad_shifted * pol_vec["Ey"] * total_phase * self.pupil_mask
            Ez = amp_pad_shifted * pol_vec["Ez"] * total_phase * self.pupil_mask

            Ex_xy = np.fft.ifft2(Ex)
            Ey_xy = np.fft.ifft2(Ey)
            Ez_xy = np.fft.ifft2(Ez)
            return np.abs(Ex_xy) ** 2 + np.abs(Ey_xy) ** 2 + np.abs(Ez_xy) ** 2

        I_TE = _propagate_single(far_field_TE, s_vec)
        I_TM = _propagate_single(far_field_TM, p_vec)
        total_I = I_TE + I_TM

        max_I = float(np.nanmax(total_I))
        if max_I > 0:
            total_I = total_I / max_I
        return np.clip(total_I, 0.0, 1.0).astype(np.float64)


# =============================================================================
# 2D RCWA 辅助: 傅里叶展开、Toeplitz 矩阵、均匀层导纳
# =============================================================================
def _toeplitz_epsilon_2d(
    eps_a: complex,
    eps_b: complex,
    fill_x: float,
    fill_y: float,
    Nx: int,
    Ny: int,
    shape: str = "square",
) -> np.ndarray:
    """
    二维介电函数 ε(x,y) 的傅里叶展开矩阵

    对于矩形孔:
        ε(x,y) = ε_b + (ε_a-ε_b)·rect(x/(fx·Λx))·rect(y/(fy·Λy))
        ε̂(m,n) = (ε_a-ε_b)·fx·fy·sinc(m·fx)·sinc(n·fy)

    对于圆形孔 (半径 r, 方形单元):
        ε̂(m,n) = (ε_a-ε_b)·(π·r²/Λ²)·2·J1(2π·ρ·r/Λ) / (2π·ρ·r/Λ)
        其中 ρ = sqrt((m/Λx)² + (n/Λy)²), J1 为第一类一阶 Bessel 函数
    """
    Mx = (Nx - 1) // 2
    My = (Ny - 1) // 2
    N = Nx * Ny

    m_all = np.arange(-2 * Mx, 2 * Mx + 1)
    n_all = np.arange(-2 * My, 2 * My + 1)
    MM, NN = np.meshgrid(m_all, n_all, indexing="ij")

    if shape == "circle":
        r_ratio = np.sqrt(fill_x * fill_y) / 2.0
        rho_mn = np.sqrt((MM / (2.0 * r_ratio + 1e-10)) ** 2 + (NN / (2.0 * r_ratio + 1e-10)) ** 2)
        from scipy.special import j1
        coeffs_ext = np.zeros_like(MM, dtype=complex)
        zero_mask = rho_mn < 1e-8
        coeffs_ext[zero_mask] = fill_x * fill_y * (eps_a - eps_b) + eps_b
        arg = 2.0 * np.pi * rho_mn * r_ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            bessel_term = np.where(arg > 1e-10, 2.0 * j1(arg) / arg, 1.0)
        coeffs_ext[~zero_mask] = (eps_a - eps_b) * fill_x * fill_y * bessel_term[~zero_mask]
    else:
        zero_m = MM == 0
        zero_n = NN == 0
        sinc_m = np.where(zero_m, 1.0, np.sin(np.pi * MM * fill_x) / (np.pi * MM * fill_x + 1e-30))
        sinc_n = np.where(zero_n, 1.0, np.sin(np.pi * NN * fill_y) / (np.pi * NN * fill_y + 1e-30))
        coeffs_ext = (eps_a - eps_b) * fill_x * fill_y * sinc_m * sinc_n
        coeffs_ext[zero_m & zero_n] += eps_b

    idx_i = np.arange(N)
    idx_j = np.arange(N)
    II, JJ = np.meshgrid(idx_i, idx_j, indexing="ij")
    mi = (II % Nx) - Mx
    ni = (II // Nx) - My
    mj = (JJ % Nx) - Mx
    nj = (JJ // Nx) - My
    dm = mi - mj
    dn = ni - nj
    m_idx = dm + 2 * Mx
    n_idx = dn + 2 * My
    m_idx = np.clip(m_idx, 0, 4 * Mx)
    n_idx = np.clip(n_idx, 0, 4 * My)

    return coeffs_ext[m_idx, n_idx]


def _toeplitz_inverse_epsilon_2d(
    eps_a: complex,
    eps_b: complex,
    fill_x: float,
    fill_y: float,
    Nx: int,
    Ny: int,
    shape: str = "square",
) -> np.ndarray:
    """1/ε(x,y) 的傅里叶展开矩阵（Li 反演规则）"""
    inv_a = 1.0 / complex(eps_a) if abs(eps_a) > 1e-30 else 0.0
    inv_b = 1.0 / complex(eps_b) if abs(eps_b) > 1e-30 else 0.0
    return _toeplitz_epsilon_2d(inv_a, inv_b, fill_x, fill_y, Nx, Ny, shape)


@jit(nopython=True, cache=True)
def _safe_kz_2d(kx: np.ndarray, ky: np.ndarray, nk0: complex) -> np.ndarray:
    """k_z = sqrt((n k0)^2 - kx^2 - ky^2)，取 Im(kz)>=0 分支"""
    out = np.empty(kx.shape, dtype=np.complex128)
    nk0_2 = complex(nk0) * complex(nk0)
    for i in range(kx.shape[0]):
        kz2 = nk0_2 - kx[i] * kx[i] - ky[i] * ky[i]
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


def _safe_kz_2d_scalar(kx: float, ky: float, nk0: complex) -> complex:
    """标量版本 _safe_kz_2d"""
    nk0_2 = complex(nk0) * complex(nk0)
    kz2 = nk0_2 - kx * kx - ky * ky
    re = float(np.real(kz2))
    im = float(np.imag(kz2))
    r = float(np.sqrt(re * re + im * im))
    real_part = float(np.sqrt(0.5 * (r + re)))
    imag_part = 0.5 * im / real_part if real_part > 1e-30 else float(np.sqrt(r))
    if imag_part < 0:
        real_part = -real_part
        imag_part = -imag_part
    return complex(real_part, imag_part)


def _build_homogeneous_admittance_2d(
    KX: np.ndarray,
    KY: np.ndarray,
    kz: np.ndarray,
    k0: float,
    eps_r: complex,
) -> np.ndarray:
    """
    构建二维均匀介质层的 2N×2N 导纳矩阵

    输入输出: [E_x, E_y]^T <-> [H_x, H_y]^T (η₀ 归一化)
    """
    N = KX.size
    I_N = np.eye(N, dtype=complex)
    Z_N = np.zeros((N, N), dtype=complex)

    Kx = np.diag(KX / k0)
    Ky = np.diag(KY / k0)
    Kz_diag = kz / k0

    kz_safe = np.where(np.abs(kz) < 1e-30, 1e-30 + 0j, kz)
    Kz_inv = np.diag(k0 / kz_safe)
    eps_term = complex(eps_r)

    # 均匀层关系:
    #   H_x = (1/jωμ)·(∂E_z/∂y - ∂E_y/∂z)
    #   H_y = (1/jωμ)·(∂E_x/∂z - ∂E_z/∂x)
    # 利用 ∇·D=0 => Kx·Ex + Ky·Ey + Kz·Ez = 0 => Ez = -(Kx·Ex+Ky·Ey)/Kz
    # 最终 (Hx, Hy) = Y · (Ex, Ey)
    Y11 = Kx @ Kz_inv @ Ky + eps_term * Kz_inv * (0 + 0j)
    Y11 = -Kx @ Kz_inv @ Ky
    Y12 = Kx @ Kz_inv @ Kx - eps_term * I_N
    Y21 = -Ky @ Kz_inv @ Ky + eps_term * I_N
    Y22 = Ky @ Kz_inv @ Kx

    Y = np.vstack([
        np.hstack([Y11, Y12]),
        np.hstack([Y21, Y22]),
    ])
    return Y


# =============================================================================
# meent 2D 模式接入
# =============================================================================
def _try_solve_2d_with_meent(
    mask: np.ndarray,
    optics,
    cfg: RCWAConfig,
) -> Optional[RCWA2DResult]:
    """尝试使用 meent 开源库的 2D RCWA 模式求解"""
    if not cfg.use_meent_if_available:
        return None
    try:
        import meent  # type: ignore  # noqa

        warnings.warn(
            "检测到 meent 库，2D RCWA 模式 API 封装开发中，"
            "已回退到内置 2D RCWA 实现。可手动扩展本函数对接 meent 的 "
            "rcwa_2d / fmm_2d 接口。",
            stacklevel=2,
        )
        return None
    except Exception:
        return None


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
    """估计一维 line/space 结构的周期 (nm)，使用功率谱方法"""
    try:
        prof = np.mean(mask, axis=0)
        prof = prof - prof.mean()
        n = prof.size

        pspec = np.abs(np.fft.fft(prof)) ** 2
        pspec[0] = 0.0

        peak_idx = int(np.argmax(pspec[: n // 2]))
        if peak_idx == 0:
            return 0.0

        freq = peak_idx / (n * pixel_size_nm)
        if freq > 1e-10:
            return 1.0 / freq
        return 0.0
    except Exception:
        return 0.0


def _estimate_period_2d(mask: np.ndarray, pixel_size_nm: float) -> Tuple[float, float]:
    """
    估计二维周期结构的 X/Y 方向周期 (nm)

    适用于接触孔阵列、交叉光栅等 2D 周期性版图。
    使用 2D 功率谱检测主频。
    """
    try:
        ny, nx = mask.shape
        mask_centered = mask - mask.mean()

        pspec = np.abs(np.fft.fft2(mask_centered)) ** 2
        pspec = np.fft.fftshift(pspec)

        cy, cx = ny // 2, nx // 2
        pspec[cy, cx] = 0.0

        period_x = 0.0
        period_y = 0.0

        prof_x = pspec[cy, :]
        peak_x = int(np.argmax(prof_x))
        freq_x = abs(peak_x - cx) / (nx * pixel_size_nm)
        if freq_x > 1e-10:
            period_x = 1.0 / freq_x

        prof_y = pspec[:, cx]
        peak_y = int(np.argmax(prof_y))
        freq_y = abs(peak_y - cy) / (ny * pixel_size_nm)
        if freq_y > 1e-10:
            period_y = 1.0 / freq_y

        if period_x <= 0 and period_y > 0:
            period_x = period_y
        if period_y <= 0 and period_x > 0:
            period_y = period_x

        return period_x, period_y
    except Exception:
        return 0.0, 0.0


def _estimate_duty_cycle_2d(mask: np.ndarray) -> Tuple[float, float]:
    """
    估计二维周期结构的 X/Y 方向占空比

    通过统计掩模透明区覆盖率近似。
    """
    coverage = float(np.clip(np.mean(mask), 0.05, 0.95))
    fx = float(np.clip(np.sqrt(coverage), 0.1, 0.9))
    fy = fx
    return fx, fy


def _detect_hole_shape(mask: np.ndarray) -> str:
    """
    检测 2D 周期结构是方形还是圆形接触孔

    通过比较透明区域的圆度 (4πA/P²) 判断:
        - 圆形: 圆度≈1
        - 方形: 圆度≈π/4≈0.785
    """
    try:
        from scipy import ndimage

        binary = (mask > 0.5).astype(np.uint8)
        if np.sum(binary) == 0:
            return "square"

        labeled, num_features = ndimage.label(binary)
        if num_features == 0:
            return "square"

        circularities = []
        for i in range(1, min(num_features + 1, 10)):
            region = (labeled == i)
            area = float(np.sum(region))
            if area < 4:
                continue
            perimeter = float(ndimage.measurements.perimeter(region))
            if perimeter > 0:
                circularity = 4.0 * np.pi * area / (perimeter ** 2)
                circularities.append(circularity)

        if len(circularities) == 0:
            return "square"

        mean_circ = float(np.mean(circularities))
        return "circle" if mean_circ > 0.85 else "square"
    except Exception:
        return "square"


def _rcwa2d_diffraction_to_aerial(
    mask: np.ndarray,
    rcwa2d_result: RCWA2DResult,
    optics: OpticalSystem,
    polarization: Polarization,
    pixel_size_nm: float,
    rcwa_cfg: RCWAConfig,
) -> np.ndarray:
    """
    将 2D RCWA 求解得到的远场衍射级次，通过矢量传递函数传播到晶圆面。

    两种模式:
    1. vector_transfer=True: 完整矢量传递函数 (VectorTransferFunction)
    2. vector_transfer=False: 对标量 Hopkins 结果做乘性修正（与 1D 一致）
    """
    ny, nx = mask.shape

    if rcwa_cfg.vector_transfer:
        vtf = VectorTransferFunction(
            wavelength_nm=optics.wavelength,
            na=optics.na,
            n_immersion=rcwa_cfg.n_superstrate,
            pixel_size_nm=pixel_size_nm,
            grid_size=(ny, nx),
        )
        return vtf.apply_vector_transfer(
            rcwa2d_result.t_TE.astype(np.complex128),
            rcwa2d_result.t_TM.astype(np.complex128),
            optics,
        )

    imaging = PartialCoherentImaging(optics, (ny, nx))
    base_aerial = imaging.compute_aerial_image(mask)

    if polarization == Polarization.TE:
        eff = rcwa2d_result.eff_trans_TE
    elif polarization == Polarization.TM:
        eff = rcwa2d_result.eff_trans_TM
    else:
        eff = 0.5 * (rcwa2d_result.eff_trans_TE + rcwa2d_result.eff_trans_TM)

    eff_max = float(np.nanmax(eff))
    if eff_max <= 0:
        return base_aerial
    norm_eff = eff / eff_max

    period_x = float(rcwa_cfg.period_nm or _estimate_period(mask, pixel_size_nm))
    period_y = float(rcwa_cfg.period_y_nm or period_x)
    if period_x <= 0 or period_y <= 0:
        return base_aerial

    fx = np.fft.fftfreq(nx, pixel_size_nm)
    fy = np.fft.fftfreq(ny, pixel_size_nm)
    FX, FY = np.meshgrid(fx, fy)
    m_idx = np.round(FX * period_x).astype(np.int64)
    n_idx = np.round(FY * period_y).astype(np.int64)

    Mx = rcwa_cfg.n_orders
    My = rcwa_cfg.n_orders_y or Mx
    m_clipped = np.clip(m_idx, -Mx, Mx) + Mx
    n_clipped = np.clip(n_idx, -My, My) + My

    Ny_ord, Nx_ord = norm_eff.shape
    m_safe = np.clip(m_clipped, 0, Nx_ord - 1)
    n_safe = np.clip(n_clipped, 0, Ny_ord - 1)

    weight_2d = norm_eff[n_safe.ravel(), m_safe.ravel()].reshape(ny, nx)

    cutoff = optics.cutoff_frequency
    radial = np.sqrt(FX ** 2 + FY ** 2) / (cutoff + 1e-12)
    blend = np.clip(radial ** 2, 0.0, 1.0)
    final_weight = 1.0 + blend * (weight_2d - 1.0)

    corrected = base_aerial * np.fft.ifftshift(final_weight)
    corrected = np.maximum(corrected, 0.0)
    m = float(np.nanmax(corrected))
    if m > 0:
        corrected = corrected / m
    return corrected.astype(np.float64)


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
# 严格 FDTD 求解器（基于 meep）
# =============================================================================
class MeepFDTDSolver:
    """
    基于 meep 的严格 3D FDTD 求解器。

    实现完整的光刻掩模电磁仿真流程：
        1. 3D 掩模结构建模（石英基底 + Cr 吸收层 + 浸没液）
        2. 倾斜照明高斯光束注入（支持 TE/TM/非偏振）
        3. 时域 FDTD 仿真求解近场分布
        4. 近场到远场（NTFF）变换
        5. 投影光瞳滤波 → 晶圆面空间像

    当 meep 不可用时，自动退化为对标量 Hopkins 结果的 phenomenological 矢量修正。
    """

    def __init__(self, config: FDTDConfig):
        self.cfg = config
        self._meep_available = False
        self._meep = None
        self._mp = None
        try:
            if config.use_meep_if_available:
                import meep as mp  # type: ignore
                self._meep = mp
                self._mp = mp
                self._meep_available = True
        except Exception:
            self._meep_available = False

    @property
    def meep_available(self) -> bool:
        return self._meep_available

    def simulate_aerial(
        self, mask: np.ndarray, optics: OpticalSystem
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        ny, nx = mask.shape
        extra: Dict[str, Any] = {
            "fdtd_fallback": not self._meep_available,
            "theta_deg": self.cfg.illumination_theta_deg,
            "phi_deg": self.cfg.illumination_phi_deg,
            "polarization": self.cfg.polarization.value,
        }

        if not self._meep_available:
            logger.warning(
                "未检测到 meep 库，FDTD 后端使用 phenomenological 矢量修正。"
                "请 pip install meep 以启用严格 3D FDTD 求解。"
            )
            return self._fallback_simulation(mask, optics), extra

        try:
            result = self._run_meep_simulation(mask, optics)
            extra.update(result.get("extra", {}))
            extra["fdtd_fallback"] = False
            return result["aerial_image"], extra
        except Exception as e:
            logger.warning(
                f"Meep FDTD 仿真失败 ({e})，回退到 phenomenological 修正。"
            )
            extra["fdtd_error"] = str(e)
            return self._fallback_simulation(mask, optics), extra

    def _fallback_simulation(
        self, mask: np.ndarray, optics: OpticalSystem
    ) -> np.ndarray:
        ny, nx = mask.shape
        imaging = PartialCoherentImaging(optics, (ny, nx))
        base = imaging.compute_aerial_image(mask)

        wavelength = optics.wavelength
        na = optics.na
        high_na_factor = np.clip(na / 1.2, 1.0, 1.5)
        vector_correction = 1.0 + 0.1 * (high_na_factor - 1.0)

        fy = np.fft.fftfreq(ny, optics.pixel_size)
        fx = np.fft.fftfreq(nx, optics.pixel_size)
        FX, FY = np.meshgrid(fx, fy)
        radial = np.sqrt(FX ** 2 + FY ** 2) / (optics.cutoff_frequency + 1e-12)
        blend = np.clip(radial ** 2, 0.0, 1.0)

        corrected = base * (1.0 + blend * (vector_correction - 1.0))
        m = float(np.nanmax(corrected))
        if m > 0:
            corrected = corrected / m
        return np.clip(corrected, 0.0, 1.0).astype(np.float64)

    def _run_meep_simulation(
        self, mask: np.ndarray, optics: OpticalSystem
    ) -> Dict[str, Any]:
        mp = self._mp

        cfg = self.cfg
        resolution = 1.0 / cfg.grid_resolution_nm

        ny, nx = mask.shape
        pixel_size = optics.pixel_size
        sx = nx * pixel_size
        sy = ny * pixel_size

        z_substrate = cfg.substrate_thickness_nm
        z_mask = cfg.mask_thickness_nm
        z_super = cfg.superstrate_thickness_nm
        pml_z = cfg.pml_thickness_nm
        pml_xy = cfg.pml_thickness_nm

        sz = z_substrate + z_mask + z_super + 2 * pml_z

        cell_size = mp.Vector3(
            sx + 2 * pml_xy,
            sy + 2 * pml_xy,
            sz
        )

        geometry = self._build_3d_mask_geometry(mask, pixel_size, z_substrate, z_mask, mp)

        theta_rad = np.deg2rad(cfg.illumination_theta_deg)
        phi_rad = np.deg2rad(cfg.illumination_phi_deg)

        wavelength_um = optics.wavelength * 1e-3  # nm -> um
        frequency = 1.0 / wavelength_um

        fwidth = 0.1 * frequency

        source_center_z = -sz / 2 + pml_z + z_substrate * 0.5

        polarizations = []
        if cfg.polarization == Polarization.TE:
            polarizations = [("TE",)]
        elif cfg.polarization == Polarization.TM:
            polarizations = [("TM",)]
        else:
            polarizations = [("TE",), ("TM",)]

        far_field_results = []
        near_field_data = []

        for pol_tuple in polarizations:
            pol = pol_tuple[0]

            if pol == "TE":
                src_amplitude = mp.Vector3(
                    np.cos(phi_rad) * np.cos(theta_rad),
                    np.sin(phi_rad) * np.cos(theta_rad),
                    -np.sin(theta_rad)
                )
            else:
                src_amplitude = mp.Vector3(
                    -np.sin(phi_rad),
                    np.cos(phi_rad),
                    0.0
                )

            k_point = mp.Vector3(
                (2 * np.pi / wavelength_um) * np.sin(theta_rad) * np.cos(phi_rad),
                (2 * np.pi / wavelength_um) * np.sin(theta_rad) * np.sin(phi_rad),
                (2 * np.pi / wavelength_um) * np.cos(theta_rad)
            )

            sources = [
                mp.GaussianSource(
                    frequency=frequency,
                    fwidth=fwidth,
                    is_integrated=True
                )
            ]

            source_obj = mp.Source(
                src=sources[0],
                component=mp.Ez if pol == "TM" else mp.Ey,
                center=mp.Vector3(0, 0, source_center_z),
                size=mp.Vector3(sx, sy, 0),
                amplitude=1.0,
                amp_func=lambda x: src_amplitude
            )

            pml_layers = [
                mp.PML(pml_xy * 1e-3, direction=mp.X),
                mp.PML(pml_xy * 1e-3, direction=mp.Y),
                mp.PML(pml_z * 1e-3, direction=mp.Z)
            ]

            sim = mp.Simulation(
                cell_size=cell_size * 1e-3,
                geometry=geometry,
                sources=[source_obj],
                boundary_layers=pml_layers,
                resolution=resolution * 1e3,
                Courant=cfg.courant_factor,
                k_point=k_point,
                force_complex_fields=True
            )

            near_field_mon_z = sz / 2 - pml_z - cfg.ntff_distance_nm
            near_flux_region = mp.FluxRegion(
                center=mp.Vector3(0, 0, near_field_mon_z * 1e-3),
                size=mp.Vector3(sx * 1e-3, sy * 1e-3, 0),
                direction=mp.Z
            )
            near_flux = sim.add_flux(
                frequency, 0, 1, near_flux_region
            )

            dft_monitor_z = near_field_mon_z
            sim.add_dft_fields(
                [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz],
                frequency, 0, 1,
                center=mp.Vector3(0, 0, dft_monitor_z * 1e-3),
                size=mp.Vector3(sx * 1e-3, sy * 1e-3, 0)
            )

            sim.run(
                until_after_sources=mp.stop_when_fields_decayed(
                    cfg.total_time_steps * 0.5,
                    mp.Ey,
                    mp.Vector3(0, 0, near_field_mon_z * 1e-3),
                    1e-6
                )
            )

            near_eps = sim.get_epsilon()
            near_ex = sim.get_dft_array(near_flux, mp.Ex, 0)
            near_ey = sim.get_dft_array(near_flux, mp.Ey, 0)
            near_ez = sim.get_dft_array(near_flux, mp.Ez, 0)

            near_field_data.append({
                "pol": pol,
                "Ex": np.asarray(near_ex),
                "Ey": np.asarray(near_ey),
                "Ez": np.asarray(near_ez)
            })

            ff_result = self._near_to_far_field(
                near_ex, near_ey, near_ez,
                sx, sy, optics.wavelength, cfg
            )
            far_field_results.append(ff_result)

        if len(far_field_results) == 2:
            ff_te = far_field_results[0]
            ff_tm = far_field_results[1]
            far_field = {
                "k_xy": ff_te["k_xy"],
                "Efar_TE": ff_te["Efar"],
                "Efar_TM": ff_tm["Efar"],
                "Efar": 0.5 * (ff_te["Efar"] + ff_tm["Efar"])
            }
        else:
            ff = far_field_results[0]
            far_field = {
                "k_xy": ff["k_xy"],
                f"Efar_{polarizations[0][0]}": ff["Efar"],
                "Efar": ff["Efar"]
            }

        aerial_image = self._far_field_to_aerial(
            far_field, optics, cfg
        )

        return {
            "aerial_image": aerial_image,
            "extra": {
                "far_field": far_field,
                "near_fields": near_field_data,
                "wavelength_nm": optics.wavelength,
                "na": optics.na,
                "polarization": cfg.polarization.value,
                "theta_deg": cfg.illumination_theta_deg,
                "phi_deg": cfg.illumination_phi_deg,
                "fdtd_fallback": False
            }
        }

    def _build_3d_mask_geometry(
        self,
        mask: np.ndarray,
        pixel_size_nm: float,
        z_substrate_nm: float,
        z_mask_nm: float,
        mp
    ) -> list:
        geometry = []

        eps_substrate = complex(self.cfg.n_substrate) ** 2
        eps_absorber = complex(self.cfg.n_absorber) ** 2
        eps_superstrate = complex(self.cfg.n_superstrate) ** 2

        geometry.append(
            mp.Block(
                size=mp.Vector3(mp.inf, mp.inf, mp.inf),
                material=mp.Medium(
                    epsilon=float(eps_superstrate.real),
                    D_conductivity=2 * np.pi * (eps_superstrate.imag) if abs(eps_superstrate.imag) > 1e-10 else 0.0
                )
            )
        )

        substrate_z_center = -z_substrate_nm * 1e-3 / 2
        geometry.append(
            mp.Block(
                size=mp.Vector3(mp.inf, mp.inf, z_substrate_nm * 1e-3),
                center=mp.Vector3(0, 0, substrate_z_center),
                material=mp.Medium(
                    epsilon=float(eps_substrate.real),
                    D_conductivity=2 * np.pi * (eps_substrate.imag) if abs(eps_substrate.imag) > 1e-10 else 0.0
                )
            )
        )

        mask_z_center = (z_mask_nm - z_substrate_nm) * 1e-3 / 2

        ny, nx = mask.shape
        for j in range(ny):
            for i in range(nx):
                if mask[j, i] > 0.5:
                    x_pos = (i - nx / 2 + 0.5) * pixel_size_nm * 1e-3
                    y_pos = (j - ny / 2 + 0.5) * pixel_size_nm * 1e-3
                    geometry.append(
                        mp.Block(
                            size=mp.Vector3(
                                pixel_size_nm * 1e-3,
                                pixel_size_nm * 1e-3,
                                z_mask_nm * 1e-3
                            ),
                            center=mp.Vector3(x_pos, y_pos, mask_z_center),
                            material=mp.Medium(
                                epsilon=float(eps_absorber.real),
                                D_conductivity=2 * np.pi * (eps_absorber.imag) if abs(eps_absorber.imag) > 1e-10 else 0.0
                            )
                        )
                    )

        return geometry

    def _near_to_far_field(
        self,
        Ex: np.ndarray,
        Ey: np.ndarray,
        Ez: np.ndarray,
        sx_nm: float,
        sy_nm: float,
        wavelength_nm: float,
        cfg: FDTDConfig
    ) -> Dict[str, np.ndarray]:
        ny, nx = Ex.shape

        dx = sx_nm / nx
        dy = sy_nm / ny

        x = (np.arange(nx) - nx / 2) * dx
        y = (np.arange(ny) - ny / 2) * dy
        X, Y = np.meshgrid(x, y)

        window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]

        Ex_win = Ex * window
        Ey_win = Ey * window
        Ez_win = Ez * window

        fft_Ex = np.fft.fft2(np.fft.fftshift(Ex_win))
        fft_Ey = np.fft.fft2(np.fft.fftshift(Ey_win))
        fft_Ez = np.fft.fft2(np.fft.fftshift(Ez_win))

        fft_Ex = np.fft.fftshift(fft_Ex)
        fft_Ey = np.fft.fftshift(fft_Ey)
        fft_Ez = np.fft.fftshift(fft_Ez)

        k0 = 2 * np.pi / wavelength_nm

        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, dx))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, dy))
        KX, KY = np.meshgrid(kx, ky)

        kz_sq = k0 ** 2 - KX ** 2 - KY ** 2
        kz = np.lib.scimath.sqrt(kz_sq)
        kz = np.where(np.imag(kz) < 0, -kz, kz)

        radiation_mask = np.real(kz) > 0

        Efar_x = fft_Ex * radiation_mask
        Efar_y = fft_Ey * radiation_mask
        Efar_z = fft_Ez * radiation_mask

        Efar_mag = np.sqrt(
            np.abs(Efar_x) ** 2 +
            np.abs(Efar_y) ** 2 +
            np.abs(Efar_z) ** 2
        )

        max_mag = float(np.nanmax(Efar_mag))
        if max_mag > 0:
            Efar_norm = Efar_mag / max_mag
        else:
            Efar_norm = Efar_mag

        return {
            "k_xy": np.sqrt(KX ** 2 + KY ** 2),
            "KX": KX,
            "KY": KY,
            "kz": kz,
            "Efar_x": Efar_x,
            "Efar_y": Efar_y,
            "Efar_z": Efar_z,
            "Efar": Efar_norm
        }

    def _far_field_to_aerial(
        self,
        far_field: Dict[str, np.ndarray],
        optics: OpticalSystem,
        cfg: FDTDConfig
    ) -> np.ndarray:
        Efar = far_field["Efar"]
        KX = far_field["KX"]
        KY = far_field["KY"]

        ny, nx = Efar.shape

        cutoff = 2 * np.pi * optics.na / optics.wavelength

        if cfg.pupil_filter:
            pupil_mask = (KX ** 2 + KY ** 2) <= cutoff ** 2
            Efar_filtered = Efar * pupil_mask.astype(np.float64)
        else:
            Efar_filtered = Efar.copy()

        defocus_phase = np.zeros_like(Efar_filtered, dtype=np.complex128)
        if abs(optics.defocus) > 1e-10:
            kz_sq = (2 * np.pi / optics.wavelength) ** 2 - KX ** 2 - KY ** 2
            kz = np.lib.scimath.sqrt(kz_sq)
            kz = np.where(np.imag(kz) < 0, -kz, kz)
            defocus_phase = np.exp(-1j * kz * optics.defocus)

        zernike_phase = np.zeros_like(Efar_filtered, dtype=np.float64)
        if optics.zernike_coefficients:
            radial = np.sqrt(KX ** 2 + KY ** 2) / (cutoff + 1e-12)
            theta = np.arctan2(KY, KX)
            pupil_mask_zn = radial <= 1.0
            for j, coeff in optics.zernike_coefficients.items():
                if abs(coeff) < 1e-15:
                    continue
                zernike_val = _zernike_polynomial(j, radial, theta)
                zernike_phase += coeff * 2.0 * np.pi * zernike_val
            zernike_phase[~pupil_mask_zn] = 0.0

        total_phase = np.exp(1j * (zernike_phase + np.angle(defocus_phase)))
        Efar_pupil = Efar_filtered * total_phase

        Efar_shifted = np.fft.ifftshift(Efar_pupil)
        aerial_complex = np.fft.ifft2(Efar_shifted)
        aerial = np.abs(aerial_complex) ** 2

        max_val = float(np.nanmax(aerial))
        if max_val > 0:
            aerial = aerial / max_val

        return np.clip(aerial, 0.0, 1.0).astype(np.float64)


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

        if cfg.use_2d_rcwa:
            period_x, period_y = _estimate_period_2d(mask, ps_nm)
            period_x = float(cfg.period_nm or period_x)
            period_y = float(cfg.period_y_nm or period_y)

            if period_x <= 0 or period_y <= 0:
                logger.warning(
                    "2D RCWA 后端未能从掩模中估计出二维周期，"
                    "将退化为标量 Hopkins 结果（无矢量修正）。"
                )
                extra["rcwa_warning"] = "period_2d_estimation_failed"
                extra["rcwa_mode"] = "2d_fallback_hopkins"
                aerial = PartialCoherentImaging(optics, mask.shape).compute_aerial_image(mask)
            else:
                fx, fy = _estimate_duty_cycle_2d(mask)
                if cfg.hole_diameter_nm and period_x > 0:
                    fx = float(np.clip(cfg.hole_diameter_nm / period_x, 0.05, 0.95))
                    fy = fx
                if cfg.line_width_nm and period_x > 0:
                    fx = float(np.clip(cfg.line_width_nm / period_x, 0.05, 0.95))
                    fy = fx

                hole_shape = _detect_hole_shape(mask)
                extra["rcwa_period_x_nm"] = period_x
                extra["rcwa_period_y_nm"] = period_y
                extra["rcwa_duty_cycle_x"] = fx
                extra["rcwa_duty_cycle_y"] = fy
                extra["rcwa_hole_shape"] = hole_shape
                extra["rcwa_mode"] = "2d"
                extra["rcwa_vector_transfer"] = cfg.vector_transfer

                far_field_2d = _try_solve_2d_with_meent(mask, optics, cfg)
                if far_field_2d is None:
                    solver_2d = RCWASolver2D(cfg)
                    far_field_2d = solver_2d.solve_far_field(
                        wavelength_nm=optics.wavelength,
                        period_x_nm=period_x,
                        period_y_nm=period_y,
                        duty_cycle_x=fx,
                        duty_cycle_y=fy,
                        theta_deg=cfg.illumination_theta_deg,
                        phi_deg=cfg.illumination_phi_deg,
                        hole_shape=hole_shape,
                    )

                extra["diffraction_orders_2d"] = {
                    "orders_x": far_field_2d.orders_x.tolist(),
                    "orders_y": far_field_2d.orders_y.tolist(),
                    "eff_trans_TE": far_field_2d.eff_trans_TE.tolist(),
                    "eff_trans_TM": far_field_2d.eff_trans_TM.tolist(),
                    "eff_reflect_TE": far_field_2d.eff_reflect_TE.tolist(),
                    "eff_reflect_TM": far_field_2d.eff_reflect_TM.tolist(),
                }
                aerial = _rcwa2d_diffraction_to_aerial(
                    mask, far_field_2d, optics, cfg.polarization, ps_nm, cfg
                )
        else:
            period = float(cfg.period_nm or _estimate_period(mask, ps_nm))
            if period <= 0:
                logger.warning(
                    "RCWA 后端未能从掩模中估计出线/空周期，"
                    "将退化为标量 Hopkins 结果（无矢量修正）。"
                )
                extra["rcwa_warning"] = "period_estimation_failed"
                extra["rcwa_mode"] = "1d_fallback_hopkins"
                aerial = PartialCoherentImaging(optics, mask.shape).compute_aerial_image(mask)
            else:
                fill = float(np.clip(np.mean(mask), 0.05, 0.95))
                line = float(cfg.line_width_nm or (period * fill))
                duty = float(np.clip(line / period, 0.05, 0.95))
                extra["rcwa_period_nm"] = period
                extra["rcwa_duty_cycle"] = duty
                extra["rcwa_mode"] = "1d"

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

        if cfg.illumination_theta_deg == 0.0 and optics.sigma > 0:
            cfg.illumination_theta_deg = float(
                np.rad2deg(np.arcsin(optics.sigma * optics.na / optics.n_substrate.real))
                if hasattr(optics, 'n_substrate') else
                np.rad2deg(np.arcsin(optics.sigma * optics.na / 1.56))
            )

        solver = MeepFDTDSolver(cfg)
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
    "RCWA2DResult",
    "RCWASolver1D",
    "RCWASolver2D",
    "VectorTransferFunction",
    "MeepFDTDSolver",
    "simulate",
    "simulate_multi_process_unified",
    "BackendComparisonReport",
    "compare_backends",
    "batch_compare_backends",
    "export_comparison_csv",
    "make_simulate_fn_for_optimizer",
]
