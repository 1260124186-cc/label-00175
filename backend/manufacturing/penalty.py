# -*- coding: utf-8 -*-
"""
掩模制造复杂度惩罚项模块

提供可微的制造复杂度惩罚函数和梯度接口，用于 SMO/ILT 等
掩模优化流程中作为附加惩罚项，平衡成像质量与制造成本。

关键设计原则：
    - 所有函数必须可微（提供解析梯度或数值梯度）
    - 惩罚值量级与 MSE/EPE 等成像损失可比（~1e-3 ~ 1）
    - 梯度量级与成像损失梯度可比，防止梯度爆炸/消失

分项可微近似：
    1. 顶点数惩罚 (Vertex Count Penalty)
       - 基于 Harris-like 角点响应的可微版本
       - 使用对角方向的二阶差分

    2. Shot数惩罚 (Shot Count Penalty)
       - 基于周长、面积和形状复杂度的可微近似
       - 与曼哈顿化后的矩形数量正相关

    3. 数据体积惩罚 (Data Volume Penalty)
       - 顶点数与 Shot 数的加权线性组合
       - 乘以格式因子（GDS/OASIS）

    4. 写入时间惩罚 (Write Time Penalty)
       - 曝光面积 × 剂量 + Shot 偏转开销
       - 按写入器类型调整系数

综合分数：加权几何平均或线性组合
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .cost_evaluator import (
    ManufacturingCostConfig,
    ManufacturingCostResult,
    MaskManufacturingCostEvaluator,
    MaskWriterType,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 配置与封装类
# ============================================================================

@dataclass
class ManufacturingPenaltyConfig:
    """
    制造复杂度惩罚配置

    Attributes:
        enabled: 是否启用惩罚项
        total_weight: 总惩罚权重（乘以各分项之和）
        vertex_weight: 顶点数惩罚分项权重
        shot_weight: Shot数惩罚分项权重
        data_weight: 数据体积惩罚分项权重
        write_time_weight: 写入时间惩罚分项权重

        # 可微近似参数
        vertex_smoothness: 顶点数检测的平滑度（越大越宽松）
        shot_min_area_factor: Shot 最小面积因子（相对像素面积）
        curvature_threshold: 拐角曲率阈值（相对梯度）

        # 量级调整（确保与成像损失可比）
        vertex_scale: 顶点数惩罚缩放因子
        shot_scale: Shot数惩罚缩放因子
        data_scale: 数据体积惩罚缩放因子
        write_time_scale: 写入时间惩罚缩放因子

        # 归一化基准（相对复杂度惩罚时使用）
        baseline_vertex_density: 单位面积基准顶点数 (vertices/μm²)
        baseline_shot_density: 单位面积基准Shot数 (shots/μm²)
        baseline_data_density: 单位面积基准数据量 (MB/cm²)
        baseline_write_density: 单位面积基准写入时间 (min/cm²)

        # 详细输出
        return_components: 是否返回各分项明细
    """
    enabled: bool = False
    total_weight: float = 0.0
    vertex_weight: float = 0.2
    shot_weight: float = 0.35
    data_weight: float = 0.2
    write_time_weight: float = 0.25

    vertex_smoothness: float = 2.0
    shot_min_area_factor: float = 4.0
    curvature_threshold: float = 0.1

    vertex_scale: float = 1e-4
    shot_scale: float = 5e-6
    data_scale: float = 1e-2
    write_time_scale: float = 5e-3

    baseline_vertex_density: float = 100.0
    baseline_shot_density: float = 500.0
    baseline_data_density: float = 0.1
    baseline_write_density: float = 5.0

    return_components: bool = True

    cost_config: Optional[ManufacturingCostConfig] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'ManufacturingPenaltyConfig':
        if d is None:
            return cls()
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                if key == 'cost_config' and isinstance(value, dict):
                    cfg.cost_config = ManufacturingCostConfig.from_dict(value)
                else:
                    setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'enabled': self.enabled,
            'total_weight': self.total_weight,
            'vertex_weight': self.vertex_weight,
            'shot_weight': self.shot_weight,
            'data_weight': self.data_weight,
            'write_time_weight': self.write_time_weight,
            'vertex_smoothness': self.vertex_smoothness,
            'shot_min_area_factor': self.shot_min_area_factor,
            'curvature_threshold': self.curvature_threshold,
            'vertex_scale': self.vertex_scale,
            'shot_scale': self.shot_scale,
            'data_scale': self.data_scale,
            'write_time_scale': self.write_time_scale,
            'baseline_vertex_density': self.baseline_vertex_density,
            'baseline_shot_density': self.baseline_shot_density,
            'baseline_data_density': self.baseline_data_density,
            'baseline_write_density': self.baseline_write_density,
            'return_components': self.return_components,
        }
        if self.cost_config is not None:
            result['cost_config'] = self.cost_config.to_dict()
        return result

    def total_component_weight(self) -> float:
        return (self.vertex_weight + self.shot_weight
                + self.data_weight + self.write_time_weight)


# ============================================================================
# 分项惩罚与梯度（可微近似）
# ============================================================================

def _compute_perimeter_gradient(mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算掩模周长的可微近似及梯度

    P ≈ Σ sqrt((∂M/∂x)² + (∂M/∂y)² + ε)

    Returns:
        (perimeter_value, gradient_array)
    """
    H, W = mask.shape
    gy = np.zeros_like(mask)
    gx = np.zeros_like(mask)
    gy[:-1, :] = mask[1:, :] - mask[:-1, :]
    gx[:, :-1] = mask[:, 1:] - mask[:, :-1]

    eps = 1e-8
    mag = np.sqrt(gx ** 2 + gy ** 2 + eps)
    perimeter = float(np.sum(mag))

    if perimeter < 1e-12:
        return perimeter, np.zeros_like(mask)

    grad = np.zeros_like(mask)

    dx_pos = np.zeros_like(mask)
    dx_pos[:, :-1] = gx[:, :-1] / (mag[:, :-1] + eps)
    dx_neg = np.zeros_like(mask)
    dx_neg[:, 1:] = -gx[:, :-1] / (mag[:, :-1] + eps)

    dy_pos = np.zeros_like(mask)
    dy_pos[:-1, :] = gy[:-1, :] / (mag[:-1, :] + eps)
    dy_neg = np.zeros_like(mask)
    dy_neg[1:, :] = -gy[:-1, :] / (mag[:-1, :] + eps)

    grad = dx_pos + dx_neg + dy_pos + dy_neg

    return perimeter, grad


def compute_vertex_penalty(mask: np.ndarray,
                           smoothness: float = 2.0,
                           curvature_threshold: float = 0.1,
                           pixel_size_nm: float = 1.0,
                           ) -> Tuple[float, np.ndarray]:
    """
    顶点数惩罚的可微近似

    使用 Harris-like 角点响应的平滑版本：
        Corner(M) = |∇²M|² - κ·|∇M|⁴   (简化为对角方向二阶差响应)
        VertexPenalty = Σ softplus(Corner - threshold)

    Args:
        mask: 掩模图案 (H, W) float64 [0,1]
        smoothness: 平滑参数，softplus 的温度
        curvature_threshold: 响应阈值
        pixel_size_nm: 像素尺寸 (nm)

    Returns:
        (penalty_value, gradient_array)
    """
    H, W = mask.shape

    if H < 5 or W < 5:
        return 0.0, np.zeros_like(mask)

    # 一阶梯度（周长相关）
    gx = np.zeros_like(mask)
    gy = np.zeros_like(mask)
    gx[:, :-1] = mask[:, 1:] - mask[:, :-1]
    gy[:-1, :] = mask[1:, :] - mask[:-1, :]

    # 二阶梯度
    gxx = np.zeros_like(mask)
    gyy = np.zeros_like(mask)
    gxy = np.zeros_like(mask)

    gxx[:, 1:-1] = gx[:, 1:-1] - gx[:, :-2]
    gyy[1:-1, :] = gy[1:-1, :] - gy[:-2, :]

    gxy[:-1, :-1] = (mask[1:, 1:] - mask[1:, :-1]
                      - mask[:-1, 1:] + mask[:-1, :-1])

    # Harris 响应 R = det(M) - k·trace(M)²
    M00 = gxx * gxx + 1e-6
    M11 = gyy * gyy + 1e-6
    M01 = gxy * gxy

    det = M00 * M11 - M01 * M01
    trace = M00 + M11
    kappa = 0.04
    R = det - kappa * trace * trace

    # 归一化响应到合理范围
    R_norm = R / (1e-10 + np.mean(np.abs(R)) + 1e-18)

    # Softplus 近似: log(1 + exp(x))，平滑阈值处理
    temperature = 1.0 / max(smoothness, 0.01)
    shifted = temperature * (R_norm - curvature_threshold)

    # 数值稳定的 softplus
    max_shifted = np.max(shifted)
    shifted_clamped = shifted - max_shifted
    softplus = (np.log(1.0 + np.exp(shifted_clamped)) + max_shifted) / temperature

    penalty = float(np.sum(softplus)) / pixel_size_nm

    # --- 梯度反向传播 ---
    dSoftplus = 1.0 / (1.0 + np.exp(-shifted))  # σ(x) = 1/(1+exp(-x))
    dSoftplus = dSoftplus / pixel_size_nm

    dR_norm = dSoftplus * temperature

    mean_R_abs = np.mean(np.abs(R)) + 1e-18
    dR = dR_norm / mean_R_abs
    dR -= dR_norm * (R * np.sign(R) / (mean_R_abs ** 2)) * 1.0 / (H * W)

    dM00 = dR * (M11 - 2 * kappa * trace)
    dM11 = dR * (M00 - 2 * kappa * trace)
    dM01 = dR * (-2 * M01)

    dGxx = dM00 * 2 * gxx
    dGyy = dM11 * 2 * gyy
    dGxy = dM01 * 2 * gxy

    # 二阶梯度回传到一阶梯度
    dGx = np.zeros_like(mask)
    dGy = np.zeros_like(mask)

    dGx[:, 1:-1] += dGxx[:, 1:-1]
    dGx[:, :-2] -= dGxx[:, 1:-1]

    dGy[1:-1, :] += dGyy[1:-1, :]
    dGy[:-2, :] -= dGyy[1:-1, :]

    dGxy_pad = np.zeros_like(mask)
    dGxy_pad[:-1, :-1] = dGxy[:-1, :-1]

    # 一阶梯度回传到 mask
    grad = np.zeros_like(mask)
    grad[:, :-1] -= dGx[:, :-1]
    grad[:, 1:] += dGx[:, :-1]

    grad[:-1, :] -= dGy[:-1, :]
    grad[1:, :] += dGy[:-1, :]

    grad[:-1, :-1] += dGxy_pad[:-1, :-1]
    grad[1:, :-1] -= dGxy_pad[:-1, :-1]
    grad[:-1, 1:] -= dGxy_pad[:-1, :-1]
    grad[1:, 1:] += dGxy_pad[:-1, :-1]

    return penalty, grad


def compute_shot_penalty(mask: np.ndarray,
                         min_area_factor: float = 4.0,
                         pixel_size_nm: float = 1.0,
                         ) -> Tuple[float, np.ndarray]:
    """
    Shot 数惩罚的可微近似

    Shot 数近似公式：
        N_shots ≈ (Perimeter² / (4π·Area)) × (Area / AvgShotArea)
                = (Perimeter²) / (4π·AvgShotArea)

    其中 AvgShotArea = min_size_factor × pixel_area

    Args:
        mask: 掩模图案 (H, W) float64 [0,1]
        min_area_factor: 平均Shot面积因子（相对像素面积）
        pixel_size_nm: 像素尺寸 (nm)

    Returns:
        (penalty_value, gradient_array)
    """
    H, W = mask.shape
    N = H * W

    perimeter, dP_dM = _compute_perimeter_gradient(mask)

    area = float(np.sum(mask))
    if area < 1e-8:
        return 0.0, np.zeros_like(mask)
    dArea_dM = np.ones_like(mask) / N  # 对平均面积的归一化

    pixel_area = pixel_size_nm ** 2
    avg_shot_area = min_area_factor * pixel_area

    # N_shots ≈ perimeter² / (4 * pi * avg_shot_area)
    shots_est = (perimeter ** 2) / (4.0 * np.pi * avg_shot_area)
    penalty = shots_est / N  # 归一化到每个像素

    # 梯度: dP/dM = (2·P / (4π·A_avg)) · dP/dM
    coeff = (2.0 * perimeter) / (4.0 * np.pi * avg_shot_area) / N
    grad = coeff * dP_dM

    return penalty, grad


def compute_data_penalty(mask: np.ndarray,
                         vertex_penalty_value: float,
                         shot_penalty_value: float,
                         format_gds: bool = False,
                         hierarchy_factor: float = 0.6,
                         ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    数据体积惩罚的可微近似

    Data ≈ (w_vertex · VertexCount + w_shot · ShotCount) × hierarchy
           + constant_overhead

    Args:
        mask: 掩模图案（用于尺寸归一化）
        vertex_penalty_value: 已计算的顶点惩罚值
        shot_penalty_value: 已计算的 Shot 惩罚值
        format_gds: True=GDSII, False=OASIS
        hierarchy_factor: 层次压缩因子 (<1)

    Returns:
        (data_penalty, grad_wrt_vertex_penalty_input, grad_wrt_shot_penalty_input)
    """
    N = mask.shape[0] * mask.shape[1]

    if format_gds:
        w_vertex = 16.0
        w_shot = 24.0
    else:
        w_vertex = 4.0
        w_shot = 8.0

    data_bytes = (w_vertex * vertex_penalty_value * N
                  + w_shot * shot_penalty_value * N)
    data_bytes *= hierarchy_factor

    # 归一化到 MB，再除以 N
    penalty_mb = data_bytes / (1024.0 * 1024.0) / N

    dv = w_vertex * hierarchy_factor / (1024.0 * 1024.0)
    ds = w_shot * hierarchy_factor / (1024.0 * 1024.0)

    return penalty_mb, np.array(dv), np.array(ds)


def compute_write_time_penalty(mask: np.ndarray,
                               total_area_um2: float,
                               shot_penalty_value: float,
                               writer_type: MaskWriterType = MaskWriterType.VSB_EBEAM,
                               dose_uC_cm2: float = 40.0,
                               ebeam_current_nA: float = 100.0,
                               pixel_size_nm: float = 1.0,
                               ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    写入时间惩罚的可微近似

    VSB E-Beam:
        T ≈ (Dose · Area / Current) + N_shots · T_deflect

    Args:
        mask: 掩模图案（尺寸归一化）
        total_area_um2: 总曝光面积 (μm²)
        shot_penalty_value: Shot 惩罚值
        writer_type: 写入器类型
        dose_uC_cm2: 剂量 (μC/cm²)
        ebeam_current_nA: 电子束电流 (nA)
        pixel_size_nm: 像素尺寸 (nm)

    Returns:
        (write_penalty_min, grad_area_component, grad_shot_component)
    """
    N = mask.shape[0] * mask.shape[1]

    if writer_type in (MaskWriterType.VSB_EBEAM, MaskWriterType.GAUSSIAN_EBEAM):
        dose_C_cm2 = dose_uC_cm2 * 1e-6
        current_C_s = ebeam_current_nA * 1e-9
        area_cm2 = total_area_um2 * 1e-8

        if current_C_s < 1e-18:
            current_C_s = 1e-18

        exposure_min = (dose_C_cm2 * area_cm2 / current_C_s) / 60.0

        t_deflect_per_shot_s = 5e-7
        shot_count_est = shot_penalty_value * N
        deflect_min = shot_count_est * t_deflect_per_shot_s / 60.0

        total_min = exposure_min + deflect_min
        penalty = total_min / N

        # 梯度分量
        grad_area_factor = (dose_C_cm2 / current_C_s / 60.0) * 1e-8 / N
        grad_shot_factor = t_deflect_per_shot_s / 60.0

        return penalty, np.array(grad_area_factor), np.array(grad_shot_factor)

    else:
        total_area_m2 = total_area_um2 * 1e-12
        swath_width_m = 0.1
        scan_speed = 10.0

        scan_min = total_area_m2 / (scan_speed * swath_width_m) / 60.0
        stage_min = 2.0  # 近似平台移动开销
        total_min = scan_min + stage_min
        penalty = total_min / N

        grad_area_factor = (1.0 / (scan_speed * swath_width_m) / 60.0) * 1e-12 / N
        grad_shot_factor = 0.0

        return penalty, np.array(grad_area_factor), np.array(grad_shot_factor)


# ============================================================================
# 综合惩罚 API（供优化器调用）
# ============================================================================

def compute_manufacturing_penalty(
        mask: np.ndarray,
        config: Optional[ManufacturingPenaltyConfig] = None,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    计算制造复杂度综合惩罚值（无梯度版本，供损失评估使用）

    Args:
        mask: 掩模图案 (H, W) float64 [0,1]
        config: 惩罚配置

    Returns:
        若 return_components=False: 标量惩罚值
        若 return_components=True: (total_penalty, components_dict)
    """
    if config is None:
        config = ManufacturingPenaltyConfig()

    if not config.enabled or config.total_weight < 1e-12:
        if config.return_components:
            return 0.0, {'total': 0.0}
        return 0.0

    cfg = config
    cost_cfg = cfg.cost_config or ManufacturingCostConfig()
    N = mask.shape[0] * mask.shape[1]

    components: Dict[str, float] = {}

    # 1. 顶点惩罚
    vertex_p, _ = compute_vertex_penalty(
        mask,
        smoothness=cfg.vertex_smoothness,
        curvature_threshold=cfg.curvature_threshold,
        pixel_size_nm=cost_cfg.pixel_size_nm,
    )
    components['vertex_raw'] = vertex_p
    vertex_scaled = cfg.vertex_scale * vertex_p
    components['vertex_scaled'] = vertex_scaled

    # 2. Shot 惩罚
    shot_p, _ = compute_shot_penalty(
        mask,
        min_area_factor=cfg.shot_min_area_factor,
        pixel_size_nm=cost_cfg.pixel_size_nm,
    )
    components['shot_raw'] = shot_p
    shot_scaled = cfg.shot_scale * shot_p
    components['shot_scaled'] = shot_scaled

    # 3. 数据体积惩罚
    total_area_um2 = float(np.sum(mask)) * (cost_cfg.pixel_size_nm ** 2) * 1e-6
    format_gds = (cost_cfg.output_format == 'gds')
    data_p, _, _ = compute_data_penalty(
        mask, vertex_p, shot_p,
        format_gds=format_gds,
        hierarchy_factor=cost_cfg.hierarchy_factor,
    )
    components['data_raw'] = data_p
    data_scaled = cfg.data_scale * data_p
    components['data_scaled'] = data_scaled

    # 4. 写入时间惩罚
    write_p, _, _ = compute_write_time_penalty(
        mask, total_area_um2, shot_p,
        writer_type=cost_cfg.writer_type,
        dose_uC_cm2=cost_cfg.dose_uC_cm2,
        ebeam_current_nA=cost_cfg.ebeam_current_nA,
        pixel_size_nm=cost_cfg.pixel_size_nm,
    )
    components['write_time_raw'] = write_p
    write_scaled = cfg.write_time_scale * write_p
    components['write_time_scaled'] = write_scaled

    # 加权求和
    w_sum = cfg.total_component_weight() or 1.0
    weighted_sum = (
        cfg.vertex_weight * vertex_scaled
        + cfg.shot_weight * shot_scaled
        + cfg.data_weight * data_scaled
        + cfg.write_time_weight * write_scaled
    ) / w_sum

    total = cfg.total_weight * weighted_sum

    components['weighted_sum'] = weighted_sum
    components['total'] = total

    if cfg.return_components:
        return total, components
    return total


def compute_manufacturing_penalty_gradient(
        mask: np.ndarray,
        config: Optional[ManufacturingPenaltyConfig] = None,
) -> np.ndarray:
    """
    计算制造复杂度惩罚对掩模的梯度

    通过链式法则反向传播各分项梯度。

    Args:
        mask: 掩模图案 (H, W) float64 [0,1]
        config: 惩罚配置

    Returns:
        梯度数组，与 mask 同形状
    """
    if config is None:
        config = ManufacturingPenaltyConfig()

    if not config.enabled or config.total_weight < 1e-12:
        return np.zeros_like(mask)

    cfg = config
    cost_cfg = cfg.cost_config or ManufacturingCostConfig()
    N = mask.shape[0] * mask.shape[1]

    w_sum = cfg.total_component_weight() or 1.0
    total_grad = np.zeros_like(mask)

    # 1. 顶点惩罚梯度
    _, v_grad = compute_vertex_penalty(
        mask,
        smoothness=cfg.vertex_smoothness,
        curvature_threshold=cfg.curvature_threshold,
        pixel_size_nm=cost_cfg.pixel_size_nm,
    )
    coeff_v = cfg.total_weight * cfg.vertex_weight * cfg.vertex_scale / w_sum
    total_grad += coeff_v * v_grad

    # 2. Shot 惩罚梯度（同时影响 Shot、数据、写入时间）
    shot_p, s_grad = compute_shot_penalty(
        mask,
        min_area_factor=cfg.shot_min_area_factor,
        pixel_size_nm=cost_cfg.pixel_size_nm,
    )
    coeff_s = cfg.total_weight * cfg.shot_weight * cfg.shot_scale / w_sum

    # 3. 数据惩罚：通过顶点和 Shot 反向传播
    vertex_p, _ = compute_vertex_penalty(
        mask,
        smoothness=cfg.vertex_smoothness,
        curvature_threshold=cfg.curvature_threshold,
        pixel_size_nm=cost_cfg.pixel_size_nm,
    )
    format_gds = (cost_cfg.output_format == 'gds')
    _, dv_penalty, ds_penalty = compute_data_penalty(
        mask, vertex_p, shot_p,
        format_gds=format_gds,
        hierarchy_factor=cost_cfg.hierarchy_factor,
    )
    coeff_data = cfg.total_weight * cfg.data_weight * cfg.data_scale / w_sum

    # 4. 写入时间惩罚：通过面积和 Shot 反向传播
    total_area_um2 = float(np.sum(mask)) * (cost_cfg.pixel_size_nm ** 2) * 1e-6
    _, dw_area, dw_shot = compute_write_time_penalty(
        mask, total_area_um2, shot_p,
        writer_type=cost_cfg.writer_type,
        dose_uC_cm2=cost_cfg.dose_uC_cm2,
        ebeam_current_nA=cost_cfg.ebeam_current_nA,
        pixel_size_nm=cost_cfg.pixel_size_nm,
    )
    coeff_write = cfg.total_weight * cfg.write_time_weight * cfg.write_time_scale / w_sum

    # Shot 综合梯度
    shot_chain = (
        coeff_s
        + coeff_data * float(ds_penalty) * N
        + coeff_write * float(dw_shot) * N
    )
    total_grad += shot_chain * s_grad

    # 顶点对数据惩罚的影响
    total_grad += coeff_data * float(dv_penalty) * N * v_grad

    # 面积对写入时间的影响：面积梯度 = 全1 × 归一化
    area_to_mask_factor = (cost_cfg.pixel_size_nm ** 2) * 1e-6
    area_grad = np.ones_like(mask) * area_to_mask_factor / N
    total_grad += coeff_write * float(dw_area) * N * area_grad

    return total_grad


# ============================================================================
# 封装类
# ============================================================================

class MaskManufacturingPenalty:
    """
    掩模制造复杂度惩罚项封装类

    提供面向对象的接口，集成损失计算与梯度评估，
    可直接注入到 SMO/ILT/OPC 工作流的优化器中。

    典型用法：
        >>> penalty = MaskManufacturingPenalty(config)
        >>> loss = penalty(mask)           # 仅损失值
        >>> grad = penalty.gradient(mask)  # 仅梯度
        >>> loss, grad = penalty.loss_and_grad(mask)  # 同时计算
    """

    def __init__(self, config: Optional[ManufacturingPenaltyConfig] = None):
        self.config = config or ManufacturingPenaltyConfig()
        self._evaluator = MaskManufacturingCostEvaluator(self.config.cost_config)
        self._last_components: Dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.total_weight > 1e-12

    def __call__(self, mask: np.ndarray) -> float:
        """计算惩罚值"""
        result = compute_manufacturing_penalty(mask, self.config)
        if isinstance(result, tuple):
            val, comp = result
            self._last_components = comp
            return val
        return result

    def loss(self, mask: np.ndarray) -> float:
        """同义词接口，与 __call__ 一致"""
        return self(mask)

    def gradient(self, mask: np.ndarray) -> np.ndarray:
        """计算梯度"""
        return compute_manufacturing_penalty_gradient(mask, self.config)

    def loss_and_grad(self,
                      mask: np.ndarray
                      ) -> Tuple[float, Optional[Dict[str, float]], np.ndarray]:
        """
        同时计算损失和梯度（可能比分别调用更快）

        Returns:
            (loss_value, components_dict_or_None, gradient_array)
        """
        cfg = self.config
        if not cfg.enabled or cfg.total_weight < 1e-12:
            return 0.0, {'total': 0.0}, np.zeros_like(mask)

        # 一次性计算所有分项和梯度
        # 先调用 loss 获取 components
        loss_val = compute_manufacturing_penalty(mask, cfg)
        if isinstance(loss_val, tuple):
            loss_val, components = loss_val
        else:
            components = None

        # 再计算梯度
        grad = compute_manufacturing_penalty_gradient(mask, cfg)

        if components is not None:
            self._last_components = components

        return loss_val, components, grad

    def evaluate_detailed(self,
                          mask: np.ndarray,
                          ) -> Tuple[float, Dict[str, float], ManufacturingCostResult]:
        """
        详细评估：同时返回惩罚值、分项明细和完整成本评估结果

        用于每 N 次迭代或结束时的记录与可视化，不用于优化循环。
        """
        loss_val, components = compute_manufacturing_penalty(mask, self.config)
        if not isinstance(loss_val, tuple):
            loss_val, components = compute_manufacturing_penalty(mask, self.config)
        cost_result = self._evaluator.quick_estimate(mask)
        return loss_val, components, cost_result

    def get_last_components(self) -> Dict[str, float]:
        """返回最近一次计算的分项明细"""
        return dict(self._last_components)

    def update_config(self, new_config: ManufacturingPenaltyConfig):
        """更新配置"""
        self.config = new_config
        self._evaluator = MaskManufacturingCostEvaluator(new_config.cost_config)
