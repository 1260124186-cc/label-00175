# -*- coding: utf-8 -*-
"""
水平集函数模块：连续域上表示与演化掩模边界

Level Set Method (LSM) 将掩模边界编码为零水平集 φ=0，
φ>0 区域对应掩模实体、φ<0 对应空白区域。相比像素级 [0,1]
参数化，水平集天然保证：
  - 边界光滑性：φ 的空间连续性 ⟹ 零等值线 C¹ 光滑
  - 最小特征尺寸：曲率流/形态学滤波直接作用在 φ 上
  - 拓扑灵活性：自动处理孔洞生成/合并

核心类 LevelSetFunction 封装 φ 的存储、初始化、重初始化
与零水平集提取等操作。
"""

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    laplace,
)
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LevelSetFunction:
    """
    水平集函数 φ(x)

    将二值掩模编码为符号距离函数 (Signed Distance Function, SDF)：
      - φ > 0 : 掩模实体（铬区域）
      - φ < 0 : 空白区域
      - φ = 0 : 边界（零水平集）

    Attributes:
        phi: 水平集函数值数组，shape=(H, W)
        dx: 网格间距
    """

    def __init__(self, phi: np.ndarray, dx: float = 1.0):
        """
        Args:
            phi: 水平集函数数组
            dx: 网格间距
        """
        self.phi = phi.astype(np.float64)
        self.dx = dx

    @classmethod
    def from_binary_mask(cls,
                         mask: np.ndarray,
                         smooth_sigma: float = 1.0,
                         dx: float = 1.0) -> 'LevelSetFunction':
        """
        从二值掩模 (0/1) 构造符号距离函数

        Steps:
          1. 二值图 → 距离变换
          2. 符号赋值：实体内为正、外为负
          3. 高斯平滑使零水平集从像素阶梯过渡到光滑曲线

        Args:
            mask: 二值掩模，1=实体, 0=空白
            smooth_sigma: 初始平滑 sigma（像素），0 表示不平滑
            dx: 网格间距

        Returns:
            LevelSetFunction 实例
        """
        binary = (mask >= 0.5).astype(np.float64)
        dist_inside = distance_transform_edt(binary)
        dist_outside = distance_transform_edt(1.0 - binary)
        phi = dist_inside - dist_outside

        if smooth_sigma > 0:
            phi = gaussian_filter(phi, sigma=smooth_sigma)

        return cls(phi, dx=dx)

    @classmethod
    def from_target(cls,
                    target: np.ndarray,
                    dx: float = 1.0,
                    smooth_sigma: float = 1.0) -> 'LevelSetFunction':
        """
        从目标图案构造水平集（与 from_binary_mask 相同逻辑，语义别名）

        Args:
            target: 目标图案（0/1）
            dx: 网格间距
            smooth_sigma: 平滑 sigma

        Returns:
            LevelSetFunction 实例
        """
        return cls.from_binary_mask(target, smooth_sigma=smooth_sigma, dx=dx)

    def to_density(self,
                   epsilon: float = 1.0,
                   projection: str = 'heaviside') -> np.ndarray:
        """
        将水平集转换为密度场 ρ ∈ [0, 1]

        用正则化 Heaviside 函数实现光滑投影：

          H_ε(φ) = ½ + (1/π)·arctan(φ/ε)

        或 SIMP 风格：

          ρ = H_ε(φ)^p   （p 为惩罚指数）

        Args:
            epsilon: 正则化参数，越小越接近硬阈值
            projection: 'heaviside' 或 'simp'

        Returns:
            密度场，shape 同 phi
        """
        H = 0.5 + (1.0 / np.pi) * np.arctan(self.phi / max(epsilon, 1e-12))
        if projection == 'heaviside':
            return H
        return np.clip(H, 0.0, 1.0)

    def to_binary_mask(self, threshold: float = 0.0) -> np.ndarray:
        """
        提取二值掩模：φ > threshold → 1, 否则 0

        Args:
            threshold: 水平集阈值，默认 0（零水平集）

        Returns:
            二值掩模数组
        """
        return (self.phi > threshold).astype(np.float64)

    def extract_contour(self) -> np.ndarray:
        """
        提取零水平集轮廓点（φ 符号变化处的像素坐标）

        Returns:
            (N, 2) 数组，每行为 (row, col) 坐标
        """
        sign_change_h = self.phi[:, :-1] * self.phi[:, 1:] < 0
        sign_change_v = self.phi[:-1, :] * self.phi[1:, :] < 0

        rows_h, cols_h = np.where(sign_change_h)
        cols_h = cols_h + 0.5

        rows_v, cols_v = np.where(sign_change_v)
        rows_v = rows_v + 0.5

        rows = np.concatenate([rows_h, rows_v])
        cols = np.concatenate([cols_h, cols_v])
        return np.column_stack([rows, cols])

    def compute_curvature(self) -> np.ndarray:
        """
        计算水平集曲率 κ = div(∇φ/|∇φ|)

        展开形式：
          κ = (φ_xx·φ_y² - 2·φ_xy·φ_x·φ_y + φ_yy·φ_x²)
              / (φ_x² + φ_y²)^(3/2)

        Returns:
            曲率数组
        """
        phi = self.phi
        dx = self.dx

        phi_x = np.zeros_like(phi)
        phi_y = np.zeros_like(phi)
        phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dx)
        phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dx)

        phi_xx = np.zeros_like(phi)
        phi_yy = np.zeros_like(phi)
        phi_xx[:, 1:-1] = (phi[:, 2:] - 2 * phi[:, 1:-1] + phi[:, :-2]) / (dx ** 2)
        phi_yy[1:-1, :] = (phi[2:, :] - 2 * phi[1:-1, :] + phi[:-2, :]) / (dx ** 2)

        phi_xy = np.zeros_like(phi)
        phi_xy[1:-1, 1:-1] = (
            phi[2:, 2:] - phi[2:, :-2] - phi[:-2, 2:] + phi[:-2, :-2]
        ) / (4.0 * dx ** 2)

        grad_mag_sq = phi_x ** 2 + phi_y ** 2 + 1e-12
        grad_mag_32 = grad_mag_sq ** 1.5

        kappa = (phi_xx * phi_y ** 2 - 2 * phi_xy * phi_x * phi_y
                 + phi_yy * phi_x ** 2) / grad_mag_32

        return kappa

    def compute_gradient_magnitude(self) -> np.ndarray:
        """
        计算 |∇φ|

        Returns:
            梯度幅值数组
        """
        phi = self.phi
        dx = self.dx

        phi_x = np.zeros_like(phi)
        phi_y = np.zeros_like(phi)
        phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dx)
        phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dx)

        return np.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-12)

    def reinitialize(self,
                     n_iters: int = 5,
                     dt_reinit: Optional[float] = None) -> 'LevelSetFunction':
        """
        重新初始化为符号距离函数 (Sussman et al. 1994)

        求解 Hamilton-Jacobi 方程至稳态：
          ∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0

        Args:
            n_iters: 重初始化迭代次数
            dt_reinit: 时间步长，None 时自动取 CFL 条件的 0.5 倍

        Returns:
            重初始化后的 LevelSetFunction（原地修改并返回 self）
        """
        phi = self.phi.copy()
        dx = self.dx
        phi0 = phi.copy()

        sign_phi0 = np.sign(phi0)
        sign_phi0[np.abs(phi0) < 1e-6] = 0.0

        if dt_reinit is None:
            dt_reinit = 0.5 * dx

        for _ in range(n_iters):
            phi_xp = np.roll(phi, -1, axis=1) - phi
            phi_xm = phi - np.roll(phi, 1, axis=1)
            phi_yp = np.roll(phi, -1, axis=0) - phi
            phi_ym = phi - np.roll(phi, 1, axis=0)

            dx_plus = np.maximum(phi_xp, 0.0)
            dx_minus = np.minimum(phi_xm, 0.0)
            dy_plus = np.maximum(phi_yp, 0.0)
            dy_minus = np.minimum(phi_ym, 0.0)

            grad_sq_pos = dx_plus ** 2 + dy_plus ** 2
            grad_sq_neg = dx_minus ** 2 + dy_minus ** 2

            s = sign_phi0

            term_pos = s * (np.sqrt(grad_sq_pos + 1e-12) - 1.0)
            term_neg = s * (np.sqrt(grad_sq_neg + 1e-12) - 1.0)

            d_phi = np.where(s > 0, term_pos,
                             np.where(s < 0, term_neg,
                                      np.maximum(term_pos, term_neg)))

            phi = phi - dt_reinit * d_phi

        phi[np.abs(phi0) < 1e-6] = 0.0
        self.phi = phi
        return self

    def compute_perimeter(self) -> float:
        """
        计算零水平集周长（基于梯度幅值积分的近似）

        P ≈ ∫|∇H_ε(φ)| dΩ ≈ Σ δ_ε(φ)·|∇φ|·Δx²

        Returns:
            周长近似值
        """
        epsilon = max(self.dx, 1.0)
        delta = epsilon / (np.pi * (self.phi ** 2 + epsilon ** 2))
        grad_mag = self.compute_gradient_magnitude()
        return float(np.sum(delta * grad_mag) * self.dx ** 2)

    def compute_area(self) -> float:
        """
        计算实体区域面积

        A ≈ Σ H_ε(φ)·Δx²

        Returns:
            面积近似值
        """
        density = self.to_density(epsilon=max(self.dx, 1.0))
        return float(np.sum(density) * self.dx ** 2)

    def copy(self) -> 'LevelSetFunction':
        return LevelSetFunction(self.phi.copy(), self.dx)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.phi.shape

    def __repr__(self) -> str:
        return (f"LevelSetFunction(shape={self.shape}, "
                f"phi_range=[{self.phi.min():.2f}, {self.phi.max():.2f}], "
                f"dx={self.dx})")
