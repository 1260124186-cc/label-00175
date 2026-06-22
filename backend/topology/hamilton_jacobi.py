# -*- coding: utf-8 -*-
"""
Hamilton-Jacobi 水平集演化模块

实现水平集函数 φ(x,t) 沿法向速度 V_n 的演化：

    ∂φ/∂t + V_n·|∇φ| = 0

其中 V_n 来自目标函数的形状灵敏度。采用迎风 (upwind) 有限差分
格式保证数值稳定性，CFL 条件限制时间步长。

演化框架：
  1. 计算形状灵敏度 → 法向速度 V_n
  2. 迎风格式推进 φ
  3. 周期性重初始化为符号距离函数
  4. 可选曲率正则化（光滑边界）

参考文献：
  - Osher & Sethian, "Fronts Propagating with Curvature-Dependent
    Speed", J. Comput. Phys., 1988.
  - Sethian, "Level Set Methods and Fast Marching Methods", 1999.
"""

import numpy as np
from typing import Optional, Tuple
import logging

from topology.level_set import LevelSetFunction

logger = logging.getLogger(__name__)


def upwind_gradient(phi: np.ndarray,
                    dx: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    迎风格式计算空间导数 D⁺φ 和 D⁻φ

    对 Hamilton-Jacobi 方程的数值求解至关重要：
      - V_n > 0（边界扩张）：使用后向差分 D⁻φ
      - V_n < 0（边界收缩）：使用前向差分 D⁺φ

    Args:
        phi: 水平集函数
        dx: 网格间距

    Returns:
        (D⁺φ, D⁻φ) 各方向的前向/后向差分元组
    """
    phi_xp = np.roll(phi, -1, axis=1)
    phi_xm = np.roll(phi, 1, axis=1)
    phi_yp = np.roll(phi, -1, axis=0)
    phi_ym = np.roll(phi, 1, axis=0)

    dphi_xp = (phi_xp - phi) / dx
    dphi_xm = (phi - phi_xm) / dx
    dphi_yp = (phi_yp - phi) / dx
    dphi_ym = (phi - phi_ym) / dx

    D_plus = (dphi_xp, dphi_yp)
    D_minus = (dphi_xm, dphi_ym)
    return D_plus, D_minus


def compute_upwind_gradient_magnitude(phi: np.ndarray,
                                      velocity: np.ndarray,
                                      dx: float = 1.0) -> np.ndarray:
    """
    根据 V_n 符号选择迎风方向，计算 |∇φ|

    Godunov 熵满足格式：
      V_n > 0: |∇⁻φ| = √(max(D⁻φ_x,0)² + min(D⁺φ_x,0)² + ...)
      V_n < 0: |∇⁺φ| = √(min(D⁻φ_x,0)² + max(D⁺φ_x,0)² + ...)

    Args:
        phi: 水平集函数
        velocity: 法向速度 V_n
        dx: 网格间距

    Returns:
        迎风格式梯度幅值
    """
    D_plus, D_minus = upwind_gradient(phi, dx)
    dphi_xp, dphi_yp = D_plus
    dphi_xm, dphi_ym = D_minus

    grad_sq_expand = (
        np.maximum(dphi_xm, 0) ** 2 + np.minimum(dphi_xp, 0) ** 2
        + np.maximum(dphi_ym, 0) ** 2 + np.minimum(dphi_yp, 0) ** 2
    )
    grad_sq_shrink = (
        np.minimum(dphi_xm, 0) ** 2 + np.maximum(dphi_xp, 0) ** 2
        + np.minimum(dphi_ym, 0) ** 2 + np.maximum(dphi_yp, 0) ** 2
    )

    grad_mag = np.where(
        velocity > 0,
        np.sqrt(grad_sq_expand + 1e-12),
        np.sqrt(grad_sq_shrink + 1e-12)
    )

    return grad_mag


def compute_cfl_timestep(phi: np.ndarray,
                         max_velocity: float,
                         dx: float = 1.0,
                         cfl_number: float = 0.5) -> float:
    """
    CFL 条件计算稳定时间步长

    Δt ≤ CFL · Δx / max(|V_n|·|∇φ|)

    Args:
        phi: 水平集函数
        max_velocity: 速度幅值上界
        dx: 网格间距
        cfl_number: CFL 数（安全系数），通常 ≤ 0.5

    Returns:
        允许的最大时间步长
    """
    if max_velocity < 1e-12:
        return 1.0
    return cfl_number * dx / (max_velocity + 1e-12)


class HamiltonJacobiEvolver:
    """
    Hamilton-Jacobi 水平集演化器

    核心迭代：
      φ^{n+1} = φ^n - Δt · V_n · |∇φ|_upwind

    可选增强：
      - 曲率正则化：V_n ← V_n - μ·κ（光滑边界锯齿）
      - 周期重初始化：每 K 步重初始化为 SDF
      - 自适应时间步：CFL 条件动态计算

    Attributes:
        cfl_number: CFL 安全系数
        reinit_interval: 重初始化间隔（迭代数），0 表示不重初始化
        reinit_iters: 每次重初始化的子迭代次数
        curvature_weight: 曲率正则化权重 μ
        max_dt: 时间步长上界
    """

    def __init__(self,
                 cfl_number: float = 0.5,
                 reinit_interval: int = 5,
                 reinit_iters: int = 5,
                 curvature_weight: float = 0.0,
                 max_dt: float = 1.0,
                 dx: float = 1.0):
        self.cfl_number = cfl_number
        self.reinit_interval = reinit_interval
        self.reinit_iters = reinit_iters
        self.curvature_weight = curvature_weight
        self.max_dt = max_dt
        self.dx = dx

    def evolve(self,
               ls: LevelSetFunction,
               velocity: np.ndarray,
               n_steps: int = 1,
               fixed_dt: Optional[float] = None) -> LevelSetFunction:
        """
        演化水平集函数

        Args:
            ls: 当前水平集函数
            velocity: 法向速度 V_n（正值=边界扩张，负值=收缩）
            n_steps: 演化步数
            fixed_dt: 固定时间步长，None 时使用 CFL 自适应步长

        Returns:
            演化后的 LevelSetFunction（原地修改）
        """
        for step in range(n_steps):
            v = velocity.copy()

            if self.curvature_weight > 0:
                kappa = ls.compute_curvature()
                v = v - self.curvature_weight * kappa

            if fixed_dt is not None:
                dt = min(fixed_dt, self.max_dt)
            else:
                max_v = float(np.max(np.abs(v)))
                dt = compute_cfl_timestep(
                    ls.phi, max_v, self.dx, self.cfl_number
                )
                dt = min(dt, self.max_dt)

            grad_mag = compute_upwind_gradient_magnitude(
                ls.phi, v, self.dx
            )

            ls.phi = ls.phi - dt * v * grad_mag

            if (self.reinit_interval > 0
                    and (step + 1) % self.reinit_interval == 0):
                ls.reinitialize(n_iters=self.reinit_iters)

        return ls

    def evolve_with_velocity_field(self,
                                   ls: LevelSetFunction,
                                   velocity_field: np.ndarray,
                                   dt: Optional[float] = None) -> LevelSetFunction:
        """
        使用预计算的速度场执行一步演化

        Args:
            ls: 水平集函数
            velocity_field: 法向速度场 V_n
            dt: 时间步长

        Returns:
            演化后的 LevelSetFunction
        """
        return self.evolve(ls, velocity_field, n_steps=1, fixed_dt=dt)


class ShapeVelocityCalculator:
    """
    形状速度计算器：从目标函数灵敏度提取法向速度

    水平集拓扑优化中的速度场计算：

      V_n = -dL/dφ / |∇φ|

    其中 dL/dφ 由 SIMP 模块的链式求导得到。
    分母 |∇φ| 将形状导数归一化为边界法向速度，
    消除网格尺度依赖。

    可选增强：
      - 速度归一化：防止速度幅值差异过大
      - 速度扩展：将边界速度扩展到全计算域
    """

    def __init__(self,
                 normalize: bool = True,
                 extension_method: str = 'gradient',
                 dx: float = 1.0):
        """
        Args:
            normalize: 是否归一化速度场
            extension_method: 速度扩展方法
                - 'gradient': 使用 ∇φ 方向扩展（默认）
                - 'none': 不扩展，直接使用 dL/dφ
            dx: 网格间距
        """
        self.normalize = normalize
        self.extension_method = extension_method
        self.dx = dx

    def compute(self,
                level_set_gradient: np.ndarray,
                ls: LevelSetFunction) -> np.ndarray:
        """
        计算法向速度

        V_n = -(dL/dφ) / |∇φ|

        Args:
            level_set_gradient: dL/dφ 损失对水平集的梯度
            ls: 水平集函数

        Returns:
            法向速度 V_n
        """
        grad_mag = ls.compute_gradient_magnitude()

        if self.extension_method == 'none':
            velocity = -level_set_gradient
        else:
            velocity = -level_set_gradient / (grad_mag + 1e-8)

        if self.normalize:
            max_v = float(np.max(np.abs(velocity)))
            if max_v > 1e-10:
                velocity = velocity / max_v

        return velocity
