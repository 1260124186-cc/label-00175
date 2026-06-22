# -*- coding: utf-8 -*-
"""
边界光滑与最小特征尺寸约束模块

水平集方法的天然优势：边界光滑性与特征尺寸约束可直接
在水平集函数上施加，无需像像素级方法那样依赖后处理
正则化。本模块提供：

  1. 曲率流平滑：∂φ/∂t = μ·κ·|∇φ|，消除锯齿与高频抖动
  2. 最小特征尺寸约束：基于形态学腐蚀-膨胀的距离约束
  3. 周长约束：限制边界总长度，控制掩模复杂度
  4. 半体积约束：约束实体面积比例

所有约束均以速度场形式施加到水平集演化方程中：
    ∂φ/∂t + (V_n + V_reg)·|∇φ| = 0

其中 V_reg 来自本模块的约束项。
"""

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    generate_binary_structure,
    label,
    find_objects,
)
from typing import Optional, Tuple, Dict
import logging

from topology.level_set import LevelSetFunction

logger = logging.getLogger(__name__)


def curvature_smoothing_velocity(ls: LevelSetFunction,
                                 weight: float = 1.0) -> np.ndarray:
    """
    曲率流平滑速度

    V_smooth = -μ·κ

    曲率流使边界沿曲率方向移动，凸部收缩、凹部扩张，
    实现边界光滑化。等价于最小化边界长度（周长）的
   梯度下降。

    Args:
        ls: 水平集函数
        weight: 平滑权重 μ

    Returns:
        曲率平滑速度场
    """
    kappa = ls.compute_curvature()
    return -weight * kappa


def min_feature_size_constraint_velocity(
        ls: LevelSetFunction,
        min_width: float,
        dx: float = 1.0,
        weight: float = 1.0) -> np.ndarray:
    """
    最小特征尺寸约束速度

    基于形态学腐蚀-膨胀的检测与惩罚：

      1. 腐蚀 φ > 0 的区域 r_min 像素
      2. 若腐蚀后出现新区域（被完全腐蚀的细条/小岛），
         则在对应位置施加正速度（扩张），防止特征消失
      3. 同理对孔洞：膨胀后检测过小的孔洞，施加负速度

    更高效的水平集实现：直接在 φ 上操作

      V_minfeat(x) = w · max(0, d_threshold - |φ(x)|/|∇φ|)

    其中 d_threshold = min_width/2，当某点到边界距离小于
    阈值时施加扩张力。

    Args:
        ls: 水平集函数
        min_width: 最小允许特征宽度（物理单位）
        dx: 网格间距
        weight: 约束权重

    Returns:
        最小特征尺寸约束速度场
    """
    phi = ls.phi
    grad_mag = ls.compute_gradient_magnitude()

    distance_approx = np.abs(phi) / (grad_mag + 1e-8)
    d_threshold = min_width / (2.0 * dx)

    violation = np.maximum(0, d_threshold - distance_approx)

    sign_phi = np.sign(phi)
    velocity = -weight * sign_phi * violation

    boundary_band = np.abs(phi) < (d_threshold * grad_mag * 2 + dx)
    velocity *= boundary_band.astype(np.float64)

    return velocity


def perimeter_constraint_velocity(ls: LevelSetFunction,
                                  target_perimeter: Optional[float] = None,
                                  weight: float = 0.1) -> np.ndarray:
    """
    周长约束速度

    当实际周长超过目标值时施加曲率流收缩：

      V_perimeter = -λ·(P - P_target)·κ

    P < P_target 时不惩罚。

    Args:
        ls: 水平集函数
        target_perimeter: 目标周长，None 时不惩罚
        weight: 约束权重 λ

    Returns:
        周长约束速度场
    """
    if target_perimeter is None:
        return np.zeros_like(ls.phi)

    current_perimeter = ls.compute_perimeter()
    if current_perimeter <= target_perimeter:
        return np.zeros_like(ls.phi)

    kappa = ls.compute_curvature()
    excess = current_perimeter - target_perimeter
    return -weight * excess * kappa


def volume_constraint_velocity(ls: LevelSetFunction,
                               target_volume_fraction: float = 0.5,
                               weight: float = 1.0) -> np.ndarray:
    """
    半体积（面积比例）约束速度

    驱动掩模面积比例趋近目标值：

      V_vol = -λ·(f_current - f_target)

    均匀速度场，使边界整体扩张或收缩。

    Args:
        ls: 水平集函数
        target_volume_fraction: 目标面积比例 (0,1)
        weight: 约束权重

    Returns:
        体积约束速度场
    """
    total_area = float(ls.phi.size)
    current_area = float(np.sum(ls.phi > 0))
    current_fraction = current_area / total_area

    return -weight * (current_fraction - target_volume_fraction)


class TopologyConstraints:
    """
    拓扑约束集合：整合边界光滑与特征尺寸约束

    将多个约束项的速度场叠加后注入 Hamilton-Jacobi 演化：

      V_total = V_shape + w_smooth·V_curvature + w_minfeat·V_minfeat
                + w_perimeter·V_perimeter + w_volume·V_volume

    Attributes:
        curvature_weight: 曲率平滑权重
        min_feature_width: 最小特征宽度
        min_feature_weight: 最小特征尺寸约束权重
        perimeter_target: 目标周长
        perimeter_weight: 周长约束权重
        volume_fraction: 目标面积比例
        volume_weight: 面积约束权重
    """

    def __init__(self,
                 curvature_weight: float = 0.1,
                 min_feature_width: float = 0.0,
                 min_feature_weight: float = 1.0,
                 perimeter_target: Optional[float] = None,
                 perimeter_weight: float = 0.1,
                 volume_fraction: Optional[float] = None,
                 volume_weight: float = 1.0,
                 dx: float = 1.0):
        self.curvature_weight = curvature_weight
        self.min_feature_width = min_feature_width
        self.min_feature_weight = min_feature_weight
        self.perimeter_target = perimeter_target
        self.perimeter_weight = perimeter_weight
        self.volume_fraction = volume_fraction
        self.volume_weight = volume_weight
        self.dx = dx

    def compute_constraint_velocity(self,
                                    ls: LevelSetFunction) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        计算总约束速度

        Args:
            ls: 水平集函数

        Returns:
            (约束速度场, 各约束项的贡献字典)
        """
        v_total = np.zeros_like(ls.phi)
        contributions: Dict[str, float] = {}

        if self.curvature_weight > 0:
            v_curv = curvature_smoothing_velocity(ls, self.curvature_weight)
            contributions['curvature'] = float(np.mean(np.abs(v_curv)))
            v_total += v_curv

        if self.min_feature_width > 0 and self.min_feature_weight > 0:
            v_mf = min_feature_size_constraint_velocity(
                ls, self.min_feature_width, self.dx, self.min_feature_weight
            )
            contributions['min_feature'] = float(np.mean(np.abs(v_mf)))
            v_total += v_mf

        if self.perimeter_target is not None and self.perimeter_weight > 0:
            v_peri = perimeter_constraint_velocity(
                ls, self.perimeter_target, self.perimeter_weight
            )
            contributions['perimeter'] = float(np.mean(np.abs(v_peri)))
            v_total += v_peri

        if self.volume_fraction is not None and self.volume_weight > 0:
            v_vol = volume_constraint_velocity(
                ls, self.volume_fraction, self.volume_weight
            )
            contributions['volume'] = float(np.mean(np.abs(v_vol)))
            v_total += v_vol

        contributions['total_constraint'] = float(np.mean(np.abs(v_total)))
        return v_total, contributions

    def enforce_min_feature_morphological(self,
                                          ls: LevelSetFunction,
                                          n_iters: int = 1) -> LevelSetFunction:
        """
        形态学方法强制最小特征尺寸（后处理）

        1. 开运算（腐蚀+膨胀）：去除过小的实体突起
        2. 闭运算（膨胀+腐蚀）：填充过小的孔洞
        3. 将结果转换回水平集

        Args:
            ls: 水平集函数
            n_iters: 形态学操作迭代次数

        Returns:
            约束后的 LevelSetFunction（原地修改）
        """
        if self.min_feature_width <= 0:
            return ls

        radius = max(int(self.min_feature_width / (2 * self.dx)), 1)
        struct = generate_binary_structure(2, 1)

        binary = ls.to_binary_mask()

        for _ in range(n_iters):
            opened = binary_erosion(binary, structure=struct, iterations=radius)
            opened = binary_dilation(opened, structure=struct, iterations=radius)

            closed = binary_dilation(binary, structure=struct, iterations=radius)
            closed = binary_erosion(closed, structure=struct, iterations=radius)

            binary = 0.5 * (opened + closed)
            binary = (binary >= 0.5).astype(np.float64)

        new_ls = LevelSetFunction.from_binary_mask(
            binary, smooth_sigma=0.5, dx=self.dx
        )
        ls.phi = new_ls.phi
        return ls
