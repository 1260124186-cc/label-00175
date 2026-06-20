# -*- coding: utf-8 -*-
"""
简化光刻前向模型

用于 CD-SEM 标定的解析/半解析前向模型：
    CD = f(focus, dose, target_cd, pitch, pattern_type; model_params)

核心思想：
1. 用解析 Bossung 曲线描述 CD 随 focus 和 dose 的变化趋势
2. 显式依赖关键物理参数：
   - resist_threshold (光刻胶阈值)
   - diffusion_length (酸扩散长度，控制边缘模糊 / CD 偏移)
   - na_effective (有效 NA，控制焦深与分辨率)
   - wavelength_effective (有效波长)
   - sigma_effective (有效部分相干因子，影响成像对比度)
   - resist_contrast / dose_to_clear (显影响应曲线)
3. 模型是可微的闭式表达式，便于 NLLS / LM 求解

复杂度等级：
- simple:   仅考虑焦点/剂量的 2 次项耦合，适合快速初值估计
- standard: 加入酸扩散卷积、焦深 (DOF) 依赖、CD 偏置项
- detailed: 加入 pitch 依赖的 OAI/annular 成像修正、MEEF 一阶近似
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging

from .schemas import (
    CalibrationParameterSet,
    CDSEMDataset,
    CDSEMDataPoint,
    PatternType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 物理常数辅助函数
# ---------------------------------------------------------------------------

def compute_dof(na: float, wavelength: float, k2: float = 0.6) -> float:
    """
    焦深 (Depth of Focus, DOF)

    DOF = k2 * λ / NA²   (单位与 λ 相同)
    """
    return k2 * wavelength / (na ** 2)


def compute_k1(cd: float, na: float, wavelength: float) -> float:
    """k1 因子：k1 = CD * NA / λ"""
    return cd * na / wavelength


def dose_response(relative_dose: np.ndarray,
                  dose_to_clear: float,
                  resist_contrast: float) -> np.ndarray:
    """
    光刻胶剂量响应曲线（Mack 模型简化）。

    输入为相对剂量（1.0 = 标称），输出为有效光强倍率。
    当 dose >> dose_to_clear 时趋近 1；在阈值附近呈 γ 阶过渡。

    Mack 模型的反函数简化为正响应：
        R = 1 - exp( - ( (dose / dose_to_clear) ** resist_contrast ) )
    """
    x = np.clip(relative_dose / np.maximum(dose_to_clear, 1e-6), 1e-4, 1e3)
    return 1.0 - np.exp(- (x ** resist_contrast))


def gaussian_blur_cd_shift(target_cd: np.ndarray,
                           diffusion_length: float,
                           pixel_size: float = 1.0) -> np.ndarray:
    """
    酸扩散对线宽的一阶修正（高斯模糊导致的 CD 偏移近似）。

    对于 line/space 结构，扩散使窄线变粗、窄缝变细；
    一阶近似 ΔCD ≈ A * diffusion_length² / target_cd 当 target_cd >> σ。
    这里用连续过渡公式更稳健。
    """
    if diffusion_length < 1e-6:
        return np.zeros_like(target_cd, dtype=np.float64)

    sigma = diffusion_length  # nm
    # 窄特征受扩散影响更大；经验公式（与 pitch 无关的一阶修正）
    ratio = np.clip(target_cd / (2.5 * sigma + 1e-6), 1e-3, 1e3)
    # 当 ratio >> 1，ΔCD ≈ 0；当 ratio ~ 1，ΔCD ≈ + sigma * erf(1)
    delta_cd = sigma * (1.0 - np.exp(-1.0 / ratio))
    return delta_cd


def focus_blur_factor(focus: np.ndarray, dof: float) -> np.ndarray:
    """
    离焦导致的成像模糊因子（归一化，0 表示完全模糊）。

    基于高斯离焦近似：对比度 ~ exp( - (focus / DOF)² )
    """
    return np.exp(- (focus / np.maximum(dof, 1e-6)) ** 2)


# ---------------------------------------------------------------------------
# 核心 CD 预测函数
# ---------------------------------------------------------------------------

def compute_bossung_cd(focus: Union[float, np.ndarray],
                       dose: Union[float, np.ndarray],
                       target_cd: Union[float, np.ndarray],
                       pitch: Union[float, np.ndarray, None],
                       params: Dict[str, float],
                       pattern_type: PatternType = PatternType.LINE_SPACE,
                       complexity: str = "standard",
                       ) -> Union[float, np.ndarray]:
    """
    解析 Bossung 模型：预测给定 (focus, dose) 条件下的晶圆 CD (nm)。

    Args:
        focus: 离焦量 (nm)
        dose: 相对剂量 (1.0 = 标称)
        target_cd: 目标/设计 CD (nm)
        pitch: 图形节距 (nm)，None 时使用 2*target_cd（密集）
        params: 模型参数字典（参见 CalibrationParameterSet）
        pattern_type: 图形类型
        complexity: 模型复杂度 ('simple' / 'standard' / 'detailed')

    Returns:
        预测的晶圆 CD (nm)，与 focus/dose 同形状
    """
    focus = np.asarray(focus, dtype=np.float64)
    dose = np.asarray(dose, dtype=np.float64)
    target_cd = np.asarray(target_cd, dtype=np.float64)

    if pitch is None:
        pitch = 2.0 * target_cd
    pitch = np.asarray(pitch, dtype=np.float64)

    # ---- 物理参数解包 ---------------------------------------------------
    th = float(params.get('resist_threshold', 0.30))
    diff_len = float(params.get('diffusion_length', 10.0))
    na = float(params.get('na_effective', 1.35))
    lam = float(params.get('wavelength_effective', 193.0))
    sigma = float(params.get('sigma_effective', 0.75))
    dtc = float(params.get('dose_to_clear', 0.5))
    gamma = float(params.get('resist_contrast', 3.0))

    # ---- 1. 基础分辨率 / k1 因子 -----------------------------------------
    k1_nom = compute_k1(target_cd, na, lam)

    # ---- 2. 焦深 ---------------------------------------------------------
    dof = compute_dof(na, lam, k2=0.6)

    # ---- 3. 剂量响应曲线（有效曝光强度倍率） ------------------------------
    dose_gain = dose_response(dose, dtc, gamma)  # ∈ (0, 1]

    # ---- 4. 离焦模糊造成的成像对比度衰减 ----------------------------------
    blur = focus_blur_factor(focus, dof)  # ∈ (0, 1]

    # ---- 5. 成像系统对 CD 的一阶修正 --------------------------------------
    #
    # 核心思路：Bossung 曲线的 CD 偏移 =
    #   a0 * (1 - blur)                    — 离焦导致线宽膨胀/收缩
    # + a1 * (focus)                       — 轻微线性倾斜（球差一阶）
    # + a2 * (focus * (dose - 1))          — 焦距-剂量交叉耦合
    # + a3 * (dose_gain - 1) * target_cd   — 剂量直接缩放
    #
    # a0/a1/a2/a3 与 k1、sigma 相关，由物理直觉构造：
    base_cd = target_cd

    # k1 越低，离焦对 CD 影响越大
    k1_eff = np.clip(k1_nom, 0.15, 2.0)
    a0 = base_cd * (0.10 + 0.35 / k1_eff) * (1.0 + 0.5 * sigma)  # 典型值 ~ CD × 25%
    a1 = base_cd * 0.0008 / k1_eff                                 # 线性倾斜系数
    a2 = base_cd * 0.0015                                          # f×d 耦合
    a3 = base_cd * 0.55                                            # 剂量敏感度

    # 图形类型修正
    if pattern_type == PatternType.ISOLATED_LINE:
        a0 *= 1.4
        a3 *= 1.25
    elif pattern_type == PatternType.ISOLATED_SPACE:
        a0 *= 1.3
        a3 *= 1.2
    elif pattern_type == PatternType.CONTACT_HOLE:
        a0 *= 1.15
        a3 *= 1.1
    # line_space / corner / custom 使用默认

    # ---- 6. pitch 依赖修正（detailed 复杂度） -----------------------------
    if complexity == 'detailed':
        # pitch = target_cd 的 2x 为密集；越大越孤立（直至 isoline）
        density_ratio = np.clip((2.0 * target_cd) / np.maximum(pitch, 1e-6), 0.1, 1.0)
        pitch_correction = (1.0 - density_ratio) * base_cd * 0.08 * (1.0 - blur)
    else:
        pitch_correction = 0.0

    # ---- 7. 合成 Bossung 偏移 --------------------------------------------
    #
    # 离焦在正值（过焦）和负值（欠焦）下对 CD 的影响是非对称的，
    # 因此使用 sign(focus) 对 (1-blur) 分量加权：
    #   - 过焦 (+focus): 线宽略收缩
    #   - 欠焦 (-focus): 线宽略膨胀
    #
    focus_sign = np.sign(focus + 1e-12)
    cd_shift = (
        a0 * (1.0 - blur) * (-focus_sign)        # 焦斑方向相关
        + a1 * focus                             # 线性倾斜
        + a2 * focus * (dose - 1.0)              # 交叉耦合
        + a3 * (dose_gain - 1.0)                 # 剂量缩放
        + pitch_correction                       # pitch 修正
    )

    # ---- 8. 阈值对 CD 的一阶修正 ------------------------------------------
    # 阈值越高 → 线越细（CD 越小）
    # 将阈值偏差（相对 0.3）映射为相对 CD 变化
    threshold_ref = 0.3
    threshold_factor = 1.0 - (th - threshold_ref) * 0.75
    threshold_factor = np.clip(threshold_factor, 0.5, 1.5)

    # ---- 9. 酸扩散修正 ---------------------------------------------------
    diffusion_shift = gaussian_blur_cd_shift(target_cd, diff_len)

    # ---- 10. 综合预测 ----------------------------------------------------
    predicted = (base_cd + cd_shift + diffusion_shift) * threshold_factor

    # 物理合理范围：CD 必须 > 0，且不应超过 pitch 的 90%（line_space）
    max_cd = pitch * 0.9 if pattern_type == PatternType.LINE_SPACE else pitch * 1.0
    predicted = np.clip(predicted, 1.0, max_cd)

    # 标量输入输出标量
    if predicted.ndim == 0:
        return float(predicted)
    return predicted


def compute_cd_sensitivity(focus: float,
                           dose: float,
                           target_cd: float,
                           pitch: Optional[float],
                           params: Dict[str, float],
                           pattern_type: PatternType = PatternType.LINE_SPACE,
                           complexity: str = "standard",
                           eps: float = 1e-5,
                           ) -> Dict[str, float]:
    """
    计算 CD 对各模型参数的一阶灵敏度（数值差分）。

    返回: {param_name: d(CD)/d(param)}
    """
    cd0 = compute_bossung_cd(focus, dose, target_cd, pitch, params,
                             pattern_type, complexity)
    sens = {}
    for name, val in params.items():
        step = max(abs(val) * eps, eps * 1e-2)
        params_plus = dict(params)
        params_plus[name] = val + step
        cd_plus = compute_bossung_cd(focus, dose, target_cd, pitch, params_plus,
                                     pattern_type, complexity)
        sens[name] = (cd_plus - cd0) / step
    return sens


def model_prediction(params_vec: np.ndarray,
                     param_names: List[str],
                     fixed_params: Dict[str, float],
                     focuses: np.ndarray,
                     doses: np.ndarray,
                     target_cds: np.ndarray,
                     pitches: np.ndarray,
                     pattern_types: List[PatternType],
                     complexity: str = "standard",
                     ) -> np.ndarray:
    """
    NLLS / LMFIT 内部使用的批量预测包装器。

    Args:
        params_vec: 待优化参数向量 (n_vary,)
        param_names: 与 params_vec 一一对应的参数名
        fixed_params: 固定（不优化）的参数
        focuses: N 个 focus 值
        doses: N 个 dose 值
        target_cds: N 个 target_cd 值
        pitches: N 个 pitch 值
        pattern_types: N 个 PatternType
        complexity: 模型复杂度

    Returns:
        预测 CD 数组 (N,)
    """
    # 组装完整参数
    full_params = dict(fixed_params)
    for name, val in zip(param_names, params_vec):
        full_params[name] = float(val)

    # 向量化计算（按 pattern_type 分组以减小分支开销）
    n = len(focuses)
    preds = np.zeros(n, dtype=np.float64)

    unique_pt = list(set(pattern_types))
    for pt in unique_pt:
        mask = np.array([p == pt for p in pattern_types], dtype=bool)
        preds[mask] = compute_bossung_cd(
            focuses[mask], doses[mask], target_cds[mask], pitches[mask],
            full_params, pt, complexity,
        )
    return preds


# ---------------------------------------------------------------------------
# 包装类（便于状态化使用）
# ---------------------------------------------------------------------------

class LithoForwardModel:
    """
    光刻前向模型的面向对象封装。

    典型用法::

        model = LithoForwardModel(params, complexity='standard')
        cd = model.predict(focus=50, dose=1.05, target_cd=45, pitch=90)
        jac = model.jacobian([50, -30], [1.0, 0.95], [45, 45], [90, 90])
    """

    def __init__(self,
                 params: Union[Dict[str, float], CalibrationParameterSet],
                 complexity: str = "standard"):
        if isinstance(params, CalibrationParameterSet):
            self._params = params.param_dict()
        else:
            self._params = dict(params)
        self.complexity = complexity

    @property
    def params(self) -> Dict[str, float]:
        return dict(self._params)

    def update(self, **kwargs) -> None:
        self._params.update(kwargs)

    def predict(self,
                focus: Union[float, np.ndarray],
                dose: Union[float, np.ndarray],
                target_cd: Union[float, np.ndarray],
                pitch: Union[float, np.ndarray, None] = None,
                pattern_type: PatternType = PatternType.LINE_SPACE,
                ) -> Union[float, np.ndarray]:
        return compute_bossung_cd(
            focus, dose, target_cd, pitch, self._params,
            pattern_type, self.complexity,
        )

    def predict_dataset(self,
                        dataset: CDSEMDataset,
                        param_overrides: Optional[Dict[str, float]] = None,
                        ) -> np.ndarray:
        """对整个数据集批量预测。"""
        params = dict(self._params)
        if param_overrides:
            params.update(param_overrides)

        focuses, doses = dataset.focus_dose_grid()
        target_cds = dataset.target_cds()
        pitches = dataset.pitches()
        pattern_types = dataset.pattern_types()

        return model_prediction(
            params_vec=np.array([params[n] for n in list(params.keys())]),
            param_names=list(params.keys()),
            fixed_params={},
            focuses=focuses,
            doses=doses,
            target_cds=target_cds,
            pitches=pitches,
            pattern_types=pattern_types,
            complexity=self.complexity,
        )

    def jacobian(self,
                 focuses: np.ndarray,
                 doses: np.ndarray,
                 target_cds: np.ndarray,
                 pitches: np.ndarray,
                 pattern_types: Optional[List[PatternType]] = None,
                 param_names: Optional[List[str]] = None,
                 eps: float = 1e-5,
                 ) -> np.ndarray:
        """
        计算模型对指定参数的雅可比矩阵 (N × n_params)。
        """
        if pattern_types is None:
            pattern_types = [PatternType.LINE_SPACE] * len(focuses)
        if param_names is None:
            param_names = list(self._params.keys())

        n = len(focuses)
        m = len(param_names)
        J = np.zeros((n, m), dtype=np.float64)

        for j, name in enumerate(param_names):
            orig = self._params[name]
            step = max(abs(orig) * eps, eps * 1e-2)

            self._params[name] = orig - 0.5 * step
            cd_minus = self._predict_batch(
                focuses, doses, target_cds, pitches, pattern_types,
            )

            self._params[name] = orig + 0.5 * step
            cd_plus = self._predict_batch(
                focuses, doses, target_cds, pitches, pattern_types,
            )

            self._params[name] = orig
            J[:, j] = (cd_plus - cd_minus) / step
        return J

    def _predict_batch(self,
                       focuses, doses, target_cds, pitches, pattern_types):
        return model_prediction(
            params_vec=np.array([self._params[n] for n in list(self._params.keys())]),
            param_names=list(self._params.keys()),
            fixed_params={},
            focuses=focuses,
            doses=doses,
            target_cds=target_cds,
            pitches=pitches,
            pattern_types=pattern_types,
            complexity=self.complexity,
        )
