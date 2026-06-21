# -*- coding: utf-8 -*-
"""
多保真度获取函数 (Acquisition Functions)

获取函数决定下一个评估点（x, fidelity_level），平衡探索与利用：

基础获取函数：
1. EI (Expected Improvement) - 经典的改进期望
2. UCB (Upper Confidence Bound) - 置信上界
3. PI (Probability of Improvement) - 改进概率

多保真度扩展（核心）：
4. EIV (EI per unit Cost) - 单位成本的期望改进 [Snoek et al., 2012]
   - α_EIV(x, s) = EI(x, s) / cost(s)
   - 直接将获取函数除以成本，考虑性价比

5. MFES (Multi-Fidelity Expected Improvement)
   - 考虑低保真度评估对高保真度预测的信息增益

6. KG (Knowledge Gradient) - 信息论角度的知识梯度
   - 计算评估(x,s)后对最优解期望的改进

所有获取函数均支持最小化问题（贝叶斯优化中默认最小化）
"""

import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass, field
import logging

from mfbo.schemas import (
    FidelityLevel,
    AcquisitionFunctionType,
    MFBOConfig,
    FidelityCost,
    SearchSpace,
)
from mfbo.mf_gp import MultiFidelityGP, PredictionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 基础获取函数（单保真度）
# ---------------------------------------------------------------------------

def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    f_best: float,
    xi: float = 0.01,
    maximize: bool = False,
) -> np.ndarray:
    """
    Expected Improvement (EI) 获取函数

    EI(x) = E[max(0, f(x) - f_best)] （最大化问题）
    EI(x) = E[max(0, f_best - f(x))] （最小化问题，默认）

    Args:
        mean: GP预测均值 (N,)
        std: GP预测标准差 (N,)
        f_best: 当前最优函数值
        xi: 探索偏移（平衡探索-利用）
        maximize: True=最大化，False=最小化（默认）

    Returns:
        ei: EI值 (N,)
    """
    std = np.maximum(std, 1e-12)

    if maximize:
        improvement = mean - f_best - xi
    else:
        improvement = f_best - mean - xi

    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    return ei


def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    beta: float = 2.0,
    maximize: bool = False,
) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) 获取函数

    UCB(x) = μ(x) + β σ(x)   (最大化)
    LCB(x) = μ(x) - β σ(x)   (最小化，返回负的以便最大化获取函数)

    Args:
        mean: GP预测均值 (N,)
        std: GP预测标准差 (N,)
        beta: 探索系数
        maximize: 是否最大化

    Returns:
        ucb: 获取函数值（总是越大越好）
    """
    if maximize:
        return mean + beta * std
    else:
        return -(mean - beta * std)


def probability_of_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    f_best: float,
    xi: float = 0.01,
    maximize: bool = False,
) -> np.ndarray:
    """Probability of Improvement (PI) 获取函数"""
    std = np.maximum(std, 1e-12)
    if maximize:
        z = (mean - f_best - xi) / std
    else:
        z = (f_best - mean - xi) / std
    return norm.cdf(z)


# ---------------------------------------------------------------------------
# 多保真度获取函数
# ---------------------------------------------------------------------------

def ei_per_unit_cost(
    pred: PredictionResult,
    f_best: float,
    fidelity_cost: float,
    fidelity_weight: float = 1.0,
    xi: float = 0.01,
    maximize: bool = False,
) -> np.ndarray:
    """
    Expected Improvement per Unit Cost (EIV)

    α(x, s) = EI(x, s) / (cost(s) ^ fidelity_weight)

    Args:
        pred: 该保真度下的预测
        f_best: 当前最优目标值（TARGET保真度下）
        fidelity_cost: 当前保真度的相对成本
        fidelity_weight: 成本惩罚强度 (0=忽略成本, 1=标准性价比, >1=更偏好低成本)
        xi: EI探索偏移
        maximize: 是否最大化

    Returns:
        eiv: 单位成本EI值 (N_candidates,)
    """
    ei = expected_improvement(pred.mean, pred.std, f_best, xi=xi, maximize=maximize)
    cost_denom = fidelity_cost ** fidelity_weight
    cost_denom = max(cost_denom, 1e-8)
    return ei / cost_denom


def multi_fidelity_expected_improvement(
    preds_all: Dict[FidelityLevel, PredictionResult],
    target_level: FidelityLevel,
    f_best: float,
    cost_config: FidelityCost,
    info_gain_weight: float = 0.3,
    xi: float = 0.01,
) -> Dict[FidelityLevel, np.ndarray]:
    """
    Multi-Fidelity Expected Improvement (MFES)

    综合考虑：
    1. 直接改进期望：EI在该保真度下的值
    2. 信息传递价值：低保真度→高保真度的ρ系数×EI
    3. 成本惩罚：除以成本的幂次

    α_MF(x, s) = [ρ_s * EI_s(x) + w_info * IG(x,s)] / cost(s)^α

    Args:
        preds_all: 各保真度的预测结果
        target_level: 目标保真度
        f_best: TARGET保真度下的当前最优值
        cost_config: 成本配置
        info_gain_weight: 信息增益权重
        xi: EI偏移

    Returns:
        acq_values: 各保真度下的获取函数值
    """
    results = {}
    target_int = target_level.to_int()

    for level, pred in preds_all.items():
        level_int = level.to_int()
        cost_s = cost_config.get_cost(level)

        # 保真度间相关系数（AR1假设）
        if level_int <= target_int:
            # 从level到target的ρ乘积
            rho_st = 1.0
            # 简化：相邻保真度ρ=0.8的|diff|次方
            rho_st = 0.8 ** abs(target_int - level_int)
        else:
            rho_st = 0.1  # 超过target的保真度价值很低

        # 该保真度下的EI
        ei_s = expected_improvement(pred.mean, pred.std, f_best, xi=xi, maximize=False)

        # 信息增益近似（与不确定性正相关）
        info_gain = np.log(1.0 + pred.variance / max(np.mean(pred.variance), 1e-12))

        # 综合得分
        score = (rho_st * ei_s + info_gain_weight * info_gain * np.mean(ei_s))
        score = score / (cost_s ** 0.7)  # 成本惩罚略小于1

        results[level] = score

    return results


def knowledge_gradient_approx(
    pred: PredictionResult,
    f_best: float,
    n_fantasy: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Knowledge Gradient (KG) 的近似实现

    KG(x) = E[max_x' μ'(x') - max_x' μ(x)]
          = 评估(x)后期望最优值的提升

    这里使用 Fantesy 采样近似（而非精确解析形式）

    Args:
        pred: 预测结果
        f_best: 当前最优
        n_fantasy: 幻想采样数

    Returns:
        kg_approx: KG近似值
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(pred.mean)
    kg = np.zeros(N)

    for i in range(N):
        mean_i, std_i = pred.mean[i], pred.std[i]

        # 幻想采样：从N(mean_i, std_i^2)采样
        fantasies = rng.normal(mean_i, std_i, size=n_fantasy)

        # 评估后新的f_best
        new_best = np.minimum(f_best, fantasies)
        kg[i] = f_best - np.mean(new_best)

    return kg


# ---------------------------------------------------------------------------
# 获取函数计算器
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionConfig:
    """获取函数配置"""
    function_type: AcquisitionFunctionType = AcquisitionFunctionType.EIV
    ucb_beta: float = 2.0
    ei_xi: float = 0.01
    fidelity_weight: float = 1.0  # 成本惩罚强度
    info_gain_weight: float = 0.3
    maximize: bool = False  # 默认最小化


class AcquisitionFunction:
    """
    获取函数计算器

    封装各种获取函数的计算逻辑，支持：
    - 任意候选点集的评估
    - 各保真度的获取函数值计算
    - 获取函数的数值优化（找到最大值点）
    """

    def __init__(self, config: Optional[AcquisitionConfig] = None,
                 mfbo_config: Optional[MFBOConfig] = None):
        self.config = config or AcquisitionConfig()
        self.mfbo_config = mfbo_config or MFBOConfig()

    def evaluate(
        self,
        mf_gp: MultiFidelityGP,
        X_candidates: np.ndarray,
        fidelity_level: FidelityLevel,
        f_best_target: float,
    ) -> np.ndarray:
        """
        评估指定保真度下候选点的获取函数值

        Args:
            mf_gp: 已拟合的多保真度GP
            X_candidates: (N_cand, D) 候选点
            fidelity_level: 评估的保真度
            f_best_target: TARGET保真度下的当前最优值

        Returns:
            acq_values: (N_cand,) 获取函数值（越大越好）
        """
        # 预测
        pred = mf_gp.predict(X_candidates, target_fidelity=fidelity_level)

        acq_type = self.config.function_type
        cost_config = self.mfbo_config.cost_config
        cost_s = cost_config.get_cost(fidelity_level)

        if acq_type == AcquisitionFunctionType.EI:
            return expected_improvement(
                pred.mean, pred.std, f_best_target,
                xi=self.config.ei_xi, maximize=self.config.maximize
            )

        elif acq_type == AcquisitionFunctionType.UCB:
            return upper_confidence_bound(
                pred.mean, pred.std,
                beta=self.config.ucb_beta, maximize=self.config.maximize
            )

        elif acq_type == AcquisitionFunctionType.PI:
            return probability_of_improvement(
                pred.mean, pred.std, f_best_target,
                xi=self.config.ei_xi, maximize=self.config.maximize
            )

        elif acq_type == AcquisitionFunctionType.EIV:
            return ei_per_unit_cost(
                pred, f_best_target,
                fidelity_cost=cost_s,
                fidelity_weight=self.config.fidelity_weight,
                xi=self.config.ei_xi,
                maximize=self.config.maximize,
            )

        elif acq_type == AcquisitionFunctionType.KG:
            return knowledge_gradient_approx(pred, f_best_target)

        elif acq_type == AcquisitionFunctionType.MFES:
            # 需要所有保真度的预测
            preds_all = mf_gp.predict_at_all_fidelities(X_candidates)
            results = multi_fidelity_expected_improvement(
                preds_all, self.mfbo_config.target_fidelity,
                f_best_target, cost_config,
                info_gain_weight=self.config.info_gain_weight,
                xi=self.config.ei_xi,
            )
            return results.get(fidelity_level, np.zeros(len(X_candidates)))

        else:
            # 默认回退到EI
            logger.warning(f"Unknown acquisition type {acq_type}, using EI")
            return expected_improvement(
                pred.mean, pred.std, f_best_target,
                xi=self.config.ei_xi, maximize=self.config.maximize
            )

    def evaluate_all_fidelities(
        self,
        mf_gp: MultiFidelityGP,
        X_candidates: np.ndarray,
        f_best_target: float,
    ) -> Dict[FidelityLevel, np.ndarray]:
        """评估所有保真度下的获取函数值"""
        results = {}
        for level in [FidelityLevel.LOW, FidelityLevel.MEDIUM, FidelityLevel.HIGH]:
            results[level] = self.evaluate(
                mf_gp, X_candidates, level, f_best_target
            )
        return results

    # ------------------------------------------------------------------
    # 获取函数最大化（优化）
    # ------------------------------------------------------------------

    def optimize(
        self,
        mf_gp: MultiFidelityGP,
        search_space: SearchSpace,
        fidelity_level: FidelityLevel,
        f_best_target: float,
        n_candidates: int = 500,
        n_restarts: int = 5,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        通过多阶段策略最大化获取函数

        策略：
        1. 随机采样大量候选点
        2. 选出前 K 个
        3. 对前 K 个进行 L-BFGS-B 精修

        Args:
            mf_gp: 已拟合的MF-GP
            search_space: 搜索空间
            fidelity_level: 优化哪个保真度的获取函数
            f_best_target: 当前最优值
            n_candidates: 随机采样候选点数
            n_restarts: L-BFGS重启次数
            rng: 随机数生成器

        Returns:
            (x_best, acq_best, X_sampled, acq_sampled)
            - x_best: 最优候选点 (D,)
            - acq_best: 最优获取函数值
            - X_sampled: 所有采样候选点
            - acq_sampled: 对应的获取函数值
        """
        if rng is None:
            rng = np.random.default_rng()

        # 阶段1：随机采样
        X_sample = search_space.sample(n_candidates, rng=rng)
        acq_sample = self.evaluate(
            mf_gp, X_sample, fidelity_level, f_best_target
        )

        # 当前最优（从随机采样中）
        best_idx = int(np.argmax(acq_sample))
        x_best = X_sample[best_idx].copy()
        acq_best = float(acq_sample[best_idx])

        # 阶段2：使用 scipy.optimize 对前几个点精修
        try:
            from scipy.optimize import minimize

            top_k = min(n_restarts, n_candidates)
            top_indices = np.argsort(-acq_sample)[:top_k]

            def neg_acq(x_flat):
                x = search_space.clip(x_flat.reshape(1, -1))
                val = self.evaluate(mf_gp, x, fidelity_level, f_best_target)[0]
                return -float(val)

            bounds = list(search_space.bounds)

            for idx in top_indices:
                x0 = X_sample[idx].copy()
                try:
                    result = minimize(
                        neg_acq, x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50}
                    )
                    x_opt = search_space.clip(result.x.reshape(1, -1))[0]
                    acq_opt = self.evaluate(mf_gp, x_opt.reshape(1, -1),
                                            fidelity_level, f_best_target)[0]
                    acq_opt = float(acq_opt)
                    if acq_opt > acq_best:
                        x_best = x_opt
                        acq_best = acq_opt
                except Exception as e:
                    logger.debug(f"L-BFGS refinement failed: {e}")
                    continue

        except Exception as e:
            logger.debug(f"Acquisition optimization fallback to random: {e}")

        return x_best, acq_best, X_sample, acq_sample
