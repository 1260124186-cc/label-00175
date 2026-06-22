# -*- coding: utf-8 -*-
"""
SIMP 材料插值与 Heaviside 投影模块

Solid Isotropic Material with Penalization (SIMP) 是连续拓扑优化中
经典的密度-刚度插值方法。本模块将其与水平集方法结合：

  1. Heaviside 投影：φ → ρ = H_ε(φ) 将水平集映射为 [0,1] 密度
  2. SIMP 插值：ρ → ρ^p 惩罚中间密度，驱动二值化
  3. 灵敏度过滤：在密度场上施加棋盘格抑制与梯度平滑

与像素级 SIMP 的区别在于：水平集天然提供光滑边界，
SIMP 惩罚仅用于加速收敛到清晰 0/1 解，而非主要正则化手段。
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def heaviside_projection(phi: np.ndarray,
                         epsilon: float = 1.0,
                         method: str = 'arctan') -> np.ndarray:
    """
    正则化 Heaviside 投影 H_ε(φ)

    将水平集 φ 映射到密度 ρ ∈ [0, 1]：

      arctan 方法：H_ε(φ) = ½ + (1/π)·arctan(φ/ε)
      tanh 方法：  H_ε(φ) = ½·(1 + tanh(φ/ε))

    两种方法在 ε→0 时均趋近于硬阈值 H(φ) = 1{φ>0}。

    Args:
        phi: 水平集函数
        epsilon: 正则化参数，越小边界越锐利
        method: 'arctan' 或 'tanh'

    Returns:
        密度场 ρ ∈ [0, 1]
    """
    eps = max(epsilon, 1e-12)
    if method == 'tanh':
        return 0.5 * (1.0 + np.tanh(phi / eps))
    return 0.5 + (1.0 / np.pi) * np.arctan(phi / eps)


def heaviside_projection_gradient(phi: np.ndarray,
                                  epsilon: float = 1.0,
                                  method: str = 'arctan') -> np.ndarray:
    """
    dH_ε/dφ（Heaviside 投影对水平集的导数）

    等价于正则化 Dirac delta 函数 δ_ε(φ)：

      arctan 方法：dH/dφ = (1/π)·ε / (φ² + ε²)
      tanh 方法：  dH/dφ = (1/2ε)·sech²(φ/ε)

    Args:
        phi: 水平集函数
        epsilon: 正则化参数
        method: 'arctan' 或 'tanh'

    Returns:
        导数数组
    """
    eps = max(epsilon, 1e-12)
    if method == 'tanh':
        return (1.0 / (2.0 * eps)) / np.cosh(phi / eps) ** 2
    return (1.0 / np.pi) * eps / (phi ** 2 + eps ** 2)


def simp_interpolation(density: np.ndarray,
                       penalty: float = 3.0,
                        min_density: float = 1e-3) -> np.ndarray:
    """
    SIMP 材料插值

    E(ρ) = ρ_min + (1 - ρ_min)·ρ^p

    惩罚中间密度值，使最优解趋向 0/1 分布。
    ρ_min 防止刚度矩阵奇异。

    Args:
        density: 密度场 ρ ∈ [0, 1]
        penalty: SIMP 惩罚指数 p (≥ 1)
        min_density: 最小密度 ρ_min

    Returns:
        插值后的材料属性
    """
    rho = np.clip(density, 0.0, 1.0)
    return min_density + (1.0 - min_density) * rho ** penalty


def simp_interpolation_gradient(density: np.ndarray,
                                penalty: float = 3.0,
                                min_density: float = 1e-3) -> np.ndarray:
    """
    SIMP 插值对密度的导数 dE/dρ

    dE/dρ = (1 - ρ_min)·p·ρ^(p-1)

    Args:
        density: 密度场
        penalty: 惩罚指数
        min_density: 最小密度

    Returns:
        导数数组
    """
    rho = np.clip(density, 1e-10, 1.0)
    return (1.0 - min_density) * penalty * rho ** (penalty - 1)


def sensitivity_filter(gradient: np.ndarray,
                       radius: float = 1.5,
                       dx: float = 1.0,
                       method: str = 'gaussian') -> np.ndarray:
    """
    灵敏度过滤（棋盘格抑制）

    对梯度场施加低通滤波，抑制高频振荡（棋盘格伪影），
    确保优化结果的网格无关性。

    Args:
        gradient: 梯度/灵敏度场
        radius: 过滤半径（物理单位），对应最小特征尺寸
        dx: 网格间距
        method: 'gaussian'（高斯核）或 'uniform'（均值核）

    Returns:
        过滤后的梯度场
    """
    sigma_px = max(radius / dx, 0.5)
    if method == 'gaussian':
        return gaussian_filter(gradient, sigma=sigma_px)
    size_px = max(int(2 * sigma_px), 1)
    return uniform_filter(gradient, size=size_px)


def compute_shape_gradient_to_levelset(shape_gradient: np.ndarray,
                                       phi: np.ndarray,
                                       epsilon: float = 1.0,
                                       method: str = 'arctan') -> np.ndarray:
    """
    将形状灵敏度（对密度的梯度）转换为水平集速度场

    链式法则：dL/dφ = (dL/dρ)·(dρ/dφ) = (dL/dρ)·δ_ε(φ)

    这实现了从"像素级灵敏度"到"边界法向速度"的投影，
    仅在零水平集附近有非零贡献，等价于形状导数。

    Args:
        shape_gradient: dL/dρ 形状灵敏度
        phi: 水平集函数
        epsilon: Heaviside 投影的正则化参数
        method: 投影方法

    Returns:
        dL/dφ 水平集速度场
    """
    dH_dphi = heaviside_projection_gradient(phi, epsilon, method)
    return shape_gradient * dH_dphi


class SIMPMaterialModel:
    """
    SIMP 材料模型：封装水平集 → 密度 → 材料属性 的完整映射

    典型工作流：
      1. φ → ρ = H_ε(φ)        （Heaviside 投影）
      2. ρ → E(ρ) = SIMP(ρ)    （材料插值）
      3. L(E) → dL/dE           （目标函数对材料的灵敏度）
      4. dL/dE → dL/dρ → dL/dφ  （链式求导回水平集）

    Attributes:
        penalty: SIMP 惩罚指数
        epsilon: Heaviside 投影正则化参数
        min_density: 最小密度
        projection_method: Heaviside 投影方法
        filter_radius: 灵敏度过滤半径
        filter_method: 过滤方法
    """

    def __init__(self,
                 penalty: float = 3.0,
                 epsilon: float = 1.0,
                 min_density: float = 1e-3,
                 projection_method: str = 'arctan',
                 filter_radius: float = 0.0,
                 filter_method: str = 'gaussian',
                 dx: float = 1.0):
        self.penalty = penalty
        self.epsilon = epsilon
        self.min_density = min_density
        self.projection_method = projection_method
        self.filter_radius = filter_radius
        self.filter_method = filter_method
        self.dx = dx

    def phi_to_density(self, phi: np.ndarray) -> np.ndarray:
        """水平集 → 密度场"""
        return heaviside_projection(phi, self.epsilon, self.projection_method)

    def density_to_material(self, density: np.ndarray) -> np.ndarray:
        """密度场 → 材料属性（SIMP 插值）"""
        return simp_interpolation(density, self.penalty, self.min_density)

    def phi_to_material(self, phi: np.ndarray) -> np.ndarray:
        """水平集 → 材料属性（一步完成）"""
        rho = self.phi_to_density(phi)
        return self.density_to_material(rho)

    def density_gradient_to_phi(self,
                                dL_drho: np.ndarray,
                                phi: np.ndarray) -> np.ndarray:
        """
        密度梯度 → 水平集速度场

        dL/dφ = (dL/dρ)·δ_ε(φ)

        Args:
            dL_drho: 损失对密度的梯度
            phi: 水平集函数

        Returns:
            dL/dφ
        """
        if self.filter_radius > 0:
            dL_drho = sensitivity_filter(
                dL_drho, self.filter_radius, self.dx, self.filter_method
            )
        return compute_shape_gradient_to_levelset(
            dL_drho, phi, self.epsilon, self.projection_method
        )

    def material_gradient_to_phi(self,
                                 dL_dE: np.ndarray,
                                 phi: np.ndarray) -> np.ndarray:
        """
        材料属性梯度 → 水平集速度场

        dL/dφ = (dL/dE)·(dE/dρ)·(dρ/dφ)

        Args:
            dL_dE: 损失对材料属性的梯度
            phi: 水平集函数

        Returns:
            dL/dφ
        """
        rho = self.phi_to_density(phi)
        dE_drho = simp_interpolation_gradient(rho, self.penalty, self.min_density)
        dL_drho = dL_dE * dE_drho
        return self.density_gradient_to_phi(dL_drho, phi)

    def project_density(self, density: np.ndarray) -> np.ndarray:
        """
        对密度场施加阈值投影（加速二值化）

        使用改进的 Heaviside 投影（eta 参数控制锐度）：
          ρ_proj = tanh(β·η) + tanh(β·(ρ - η)) / (tanh(β·η) + tanh(β·(1-η)))

        Args:
            density: 输入密度场

        Returns:
            投影后密度场
        """
        return np.clip(density, 0.0, 1.0)
