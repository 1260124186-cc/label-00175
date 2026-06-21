# -*- coding: utf-8 -*-
"""
保真度选择策略 (Fidelity Selection Strategies)

决定下一次评估应使用哪种保真度层级，核心创新点：
在有限计算预算下智能分配低保真度（廉价、快速）和
高保真度（昂贵、精确）的评估次数。

支持的策略：
1. COST_AWARE (成本感知) - 默认
   计算各保真度下获取函数优化结果的"单位成本改进率"，选择最高者
   本质：argmax_{s} α(x_s^*, s) / cost(s)

2. INFORMATION_GAIN (信息增益)
   基于互信息 / 知识梯度，选择能最大程度降低对最优解不确定性的保真度

3. BUDGET_PROPORTIONAL (预算比例)
   按预设比例分配各保真度的评估次数
   如：LOW:MEDIUM:HIGH = 5:2:1

4. SCHEDULED (预设调度)
   根据迭代阶段切换：前期大量低保真→中期混合→后期纯高保真

5. ADAPTIVE_THRESHOLD (自适应阈值)
   监测低保真→高保真的代理误差，当误差低于阈值时允许用低保真，
   否则强制切换到高保真
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from mfbo.schemas import (
    FidelityLevel,
    FidelitySelectionStrategy,
    MFBOConfig,
    FidelityCost,
    SearchSpace,
    Observation,
)
from mfbo.mf_gp import MultiFidelityGP
from mfbo.acquisition import AcquisitionFunction

logger = logging.getLogger(__name__)


@dataclass
class FidelityDecision:
    """
    保真度决策结果

    Attributes:
        selected_fidelity: 选择的保真度
        best_x: 对应保真度下的最优候选点 (D,)
        acq_value: 获取函数值
        scores: 各保真度的分数（用于调试）
        reason: 决策理由（用于调试和日志）
    """
    selected_fidelity: FidelityLevel
    best_x: np.ndarray
    acq_value: float
    scores: Dict[FidelityLevel, float] = field(default_factory=dict)
    best_x_per_fidelity: Dict[FidelityLevel, np.ndarray] = field(default_factory=dict)
    reason: str = ""


class FidelitySelector:
    """
    保真度选择器

    根据策略决定下一次评估应使用的保真度和对应的候选点。
    """

    def __init__(self, config: Optional[MFBOConfig] = None,
                 acquisition: Optional[AcquisitionFunction] = None):
        self.config = config or MFBOConfig()
        self.acquisition = acquisition or AcquisitionFunction(mfbo_config=self.config)
        self.rng = np.random.default_rng(self.config.random_seed)

        # 决策历史（用于自适应策略）
        self.decision_history: List[FidelityDecision] = []

        # 各保真度评估计数
        self.eval_counts: Dict[FidelityLevel, int] = defaultdict(int)

    def _get_available_fidelities(self, budget_used: float,
                                  budget_total: float) -> List[FidelityLevel]:
        """
        根据剩余预算确定可考虑的保真度

        如果剩余预算不足以支撑一次高保真评估，则只考虑低保真度
        """
        cost_config = self.config.cost_config
        available = []

        for level in [FidelityLevel.LOW, FidelityLevel.MEDIUM, FidelityLevel.HIGH]:
            c = cost_config.get_cost(level)
            if budget_used + c <= budget_total + 1e-9:
                available.append(level)

        if not available:
            # 预算不足但仍要决策，返回所有（由上层处理）
            available = [FidelityLevel.LOW, FidelityLevel.MEDIUM, FidelityLevel.HIGH]

        return available

    # ------------------------------------------------------------------
    # 策略 1: 成本感知（默认）
    # ------------------------------------------------------------------

    def _select_cost_aware(
        self,
        mf_gp: MultiFidelityGP,
        search_space: SearchSpace,
        f_best_target: float,
        available: List[FidelityLevel],
    ) -> FidelityDecision:
        """
        成本感知策略：对每个保真度优化获取函数，比较性价比

        选择:
            (x_s^*, s) = argmax_{x, s} α(x, s) / cost(s)^γ

        其中 γ 是成本惩罚权重（默认1.0）
        """
        cost_config = self.config.cost_config
        scores: Dict[FidelityLevel, float] = {}
        best_x_per: Dict[FidelityLevel, np.ndarray] = {}
        acq_per: Dict[FidelityLevel, float] = {}

        for level in available:
            x_best, acq_best, _, _ = self.acquisition.optimize(
                mf_gp=mf_gp,
                search_space=search_space,
                fidelity_level=level,
                f_best_target=f_best_target,
                n_candidates=self.config.acq_n_candidates,
                n_restarts=3,
                rng=self.rng,
            )

            cost_s = cost_config.get_cost(level)
            gamma = self._get_cost_penalty_strength()
            score = acq_best / (cost_s ** gamma)

            scores[level] = score
            best_x_per[level] = x_best
            acq_per[level] = acq_best

        # 选择分数最高的保真度
        if scores:
            selected = max(scores.keys(), key=lambda k: scores[k])
        else:
            selected = FidelityLevel.LOW

        decision = FidelityDecision(
            selected_fidelity=selected,
            best_x=best_x_per[selected],
            acq_value=acq_per[selected],
            scores=scores,
            best_x_per_fidelity=best_x_per,
            reason=f"Cost-aware: best score={scores.get(selected, 0):.4f} at {selected.value}",
        )
        return decision

    def _get_cost_penalty_strength(self) -> float:
        """
        动态调整成本惩罚强度 γ

        策略：
        - 预算充足时：γ较小（更愿用高保真）
        - 预算紧张时：γ较大（更偏好低保真）
        """
        gamma_base = self.config.acquisition_type == "eiv" and 1.0 or 0.7
        total_iters = self.config.max_iterations
        current_iter = sum(self.eval_counts.values())

        # 前30%迭代预算充足，降低成本惩罚
        progress = current_iter / max(1, total_iters)
        if progress < 0.3:
            gamma = gamma_base * 0.6
        elif progress < 0.7:
            gamma = gamma_base
        else:
            # 最后30%增加成本惩罚，但仍允许高保真
            gamma = gamma_base * 1.2
        return gamma

    # ------------------------------------------------------------------
    # 策略 2: 信息增益
    # ------------------------------------------------------------------

    def _select_info_gain(
        self,
        mf_gp: MultiFidelityGP,
        search_space: SearchSpace,
        f_best_target: float,
        available: List[FidelityLevel],
    ) -> FidelityDecision:
        """
        信息增益策略：选择能最大程度降低对TARGET不确定性的保真度

        综合考虑：
        1. 预测不确定性（高方差→高信息）
        2. 保真度间相关性（高→target 信息传递好）
        3. 成本（单位成本的信息增益）
        """
        cost_config = self.config.cost_config
        scores: Dict[FidelityLevel, float] = {}
        best_x_per: Dict[FidelityLevel, np.ndarray] = {}
        acq_per: Dict[FidelityLevel, float] = {}

        # 采样候选点
        X_cand = search_space.sample(min(self.config.acq_n_candidates, 1000),
                                      rng=self.rng)

        for level in available:
            # 获取该保真度下的预测
            pred = mf_gp.predict(X_cand, target_fidelity=level)
            info_gain = mf_gp.mutual_information_improvement(
                X_cand, level
            )

            # EI部分
            ei_vals = self.acquisition.evaluate(
                mf_gp, X_cand, level, f_best_target
            )

            # 综合：EI + λ * 信息增益
            lambda_ig = 0.5
            combined = ei_vals + lambda_ig * info_gain * max(ei_vals)

            # 除以成本
            cost_s = cost_config.get_cost(level)
            combined_scaled = combined / (cost_s ** 0.8)

            best_idx = int(np.argmax(combined_scaled))
            scores[level] = float(combined_scaled[best_idx])
            best_x_per[level] = X_cand[best_idx].copy()
            acq_per[level] = float(ei_vals[best_idx])

        if scores:
            selected = max(scores.keys(), key=lambda k: scores[k])
        else:
            selected = FidelityLevel.LOW

        return FidelityDecision(
            selected_fidelity=selected,
            best_x=best_x_per[selected],
            acq_value=acq_per[selected],
            scores=scores,
            best_x_per_fidelity=best_x_per,
            reason=f"Info-Gain: best score={scores.get(selected, 0):.4f} at {selected.value}",
        )

    # ------------------------------------------------------------------
    # 策略 3: 预算比例分配
    # ------------------------------------------------------------------

    def _select_budget_proportional(
        self,
        mf_gp: MultiFidelityGP,
        search_space: SearchSpace,
        f_best_target: float,
        available: List[FidelityLevel],
    ) -> FidelityDecision:
        """
        预算比例策略：按预设比例分配评估次数

        默认比例：LOW 60%, MEDIUM 25%, HIGH 15%
        （近似对应成本加权的均匀分配）
        """
        # 目标比例（可调）
        target_ratio = {
            FidelityLevel.LOW: 0.55,
            FidelityLevel.MEDIUM: 0.30,
            FidelityLevel.HIGH: 0.15,
        }

        # 当前比例
        total = max(1, sum(self.eval_counts.values()))
        current_ratio = {
            l: self.eval_counts[l] / total for l in available
        }

        # 选择缺口最大的保真度
        deficits = {}
        for level in available:
            target = target_ratio.get(level, 0.0)
            current = current_ratio.get(level, 0.0)
            deficits[level] = max(0.0, target - current)

        # 归一化成概率
        total_deficit = sum(deficits.values())
        if total_deficit < 1e-12:
            # 无明显缺口，回退到成本感知
            return self._select_cost_aware(mf_gp, search_space, f_best_target, available)

        probs = {l: d / total_deficit for l, d in deficits.items()}

        # 选择概率最大的（或按概率采样）
        if self.rng.random() < 0.7:
            # 贪心选择
            selected = max(probs.keys(), key=lambda k: probs[k])
        else:
            # 按概率采样（增加探索）
            levels = list(probs.keys())
            p = [probs[l] for l in levels]
            selected = levels[self.rng.choice(len(levels), p=p)]

        # 对选中的保真度优化获取函数
        x_best, acq_best, _, _ = self.acquisition.optimize(
            mf_gp, search_space, selected, f_best_target,
            n_candidates=self.config.acq_n_candidates,
            n_restarts=3,
            rng=self.rng,
        )

        scores = {l: probs[l] for l in available}
        return FidelityDecision(
            selected_fidelity=selected,
            best_x=x_best,
            acq_value=acq_best,
            scores=scores,
            best_x_per_fidelity={selected: x_best},
            reason=f"Budget-prop: selected {selected.value} "
                   f"(deficit={deficits[selected]:.3f}, prob={probs[selected]:.2f})",
        )

    # ------------------------------------------------------------------
    # 策略 4: 预设调度
    # ------------------------------------------------------------------

    def _select_scheduled(
        self,
        mf_gp: MultiFidelityGP,
        search_space: SearchSpace,
        f_best_target: float,
        available: List[FidelityLevel],
    ) -> FidelityDecision:
        """
        预设调度策略：按迭代阶段切换

        阶段划分（基于进度 p = current / total_iter）：
        - p < 0.4：纯低保真度（快速探索全局趋势）
        - 0.4 ≤ p < 0.75：低中混合（低保真探索+中保真精修）
        - p ≥ 0.75：纯高保真（最终精修，确保精度）
        """
        total_iters = self.config.max_iterations
        current_iter = sum(self.eval_counts.values())
        p = current_iter / max(1, total_iters)

        # 按阶段确定候选保真度集合
        if p < 0.4:
            stage_levels = [FidelityLevel.LOW]
            stage_name = "early-explore"
        elif p < 0.75:
            # 低中混合，概率 0.7 LOW, 0.3 MEDIUM
            if self.rng.random() < 0.7:
                stage_levels = [FidelityLevel.LOW]
            else:
                stage_levels = [FidelityLevel.MEDIUM]
            stage_name = "mid-mixed"
        else:
            # 最后阶段：纯高保真（如果预算允许）
            if FidelityLevel.HIGH in available:
                stage_levels = [FidelityLevel.HIGH]
            else:
                stage_levels = [FidelityLevel.MEDIUM]
            stage_name = "late-refine"

        # 从允许的列表中选
        allowed = [l for l in stage_levels if l in available]
        if not allowed:
            allowed = available

        # 对每个允许的保真度优化获取函数，选最佳
        best_score = -np.inf
        best_level = allowed[0]
        best_x = None
        best_acq = 0.0
        scores = {}

        for level in allowed:
            x_best, acq_best, _, _ = self.acquisition.optimize(
                mf_gp, search_space, level, f_best_target,
                n_candidates=self.config.acq_n_candidates,
                n_restarts=3,
                rng=self.rng,
            )
            cost_s = self.config.cost_config.get_cost(level)
            score = acq_best / max(cost_s, 1e-8)
            scores[level] = score
            if score > best_score:
                best_score = score
                best_level = level
                best_x = x_best
                best_acq = acq_best

        return FidelityDecision(
            selected_fidelity=best_level,
            best_x=best_x,
            acq_value=best_acq,
            scores=scores,
            best_x_per_fidelity={best_level: best_x},
            reason=f"Scheduled ({stage_name}, p={p:.2f}): "
                   f"chose {best_level.value}",
        )

    # ------------------------------------------------------------------
    # 策略 5: 自适应阈值
    # ------------------------------------------------------------------

    def _select_adaptive_threshold(
        self,
        mf_gp: MultiFidelityGP,
        search_space: SearchSpace,
        f_best_target: float,
        available: List[FidelityLevel],
        observations: List[Observation],
    ) -> FidelityDecision:
        """
        自适应阈值策略：监测代理误差决定保真度

        当低保真度→目标保真度的预测误差足够低时，使用低保真度
        否则强制提升保真度层级
        """
        # 估计代理误差
        errors = self._estimate_surrogate_errors(observations, mf_gp)

        target = self.config.target_fidelity
        low_error = errors.get((FidelityLevel.LOW, target), np.inf)
        med_error = errors.get((FidelityLevel.MEDIUM, target), np.inf)

        # 阈值（可调）
        error_threshold = self.config.min_improvement_threshold * 10

        # 决定允许的保真度
        allowed = []
        if FidelityLevel.LOW in available:
            # 如果有足够多高保真样本可以估计误差
            if mf_gp.n_by_fidelity.get(target, 0) >= 3:
                if low_error < error_threshold:
                    allowed.append(FidelityLevel.LOW)
                else:
                    logger.info(f"Adaptive: LOW error too high ({low_error:.4f}), skipping")
            else:
                allowed.append(FidelityLevel.LOW)  # 初期允许

        if FidelityLevel.MEDIUM in available:
            if mf_gp.n_by_fidelity.get(target, 0) >= 3:
                if med_error < error_threshold * 2:
                    allowed.append(FidelityLevel.MEDIUM)
            else:
                allowed.append(FidelityLevel.MEDIUM)

        if FidelityLevel.HIGH in available:
            allowed.append(FidelityLevel.HIGH)  # 总是允许HIGH

        if not allowed:
            allowed = [FidelityLevel.HIGH] if FidelityLevel.HIGH in available else available

        # 成本感知选择
        return self._select_cost_aware(mf_gp, search_space, f_best_target, allowed)

    def _estimate_surrogate_errors(self, observations: List[Observation],
                                    mf_gp: MultiFidelityGP
                                    ) -> Dict[Tuple[FidelityLevel, FidelityLevel], float]:
        """估计保真度间的代理误差"""
        return mf_gp.surrogate_fidelity_error()

    # ------------------------------------------------------------------
    # 统一决策入口
    # ------------------------------------------------------------------

    def select_next(
        self,
        mf_gp: MultiFidelityGP,
        search_space: SearchSpace,
        f_best_target: float,
        budget_used: float,
        observations: Optional[List[Observation]] = None,
    ) -> FidelityDecision:
        """
        选择下一次评估的保真度和候选点

        Args:
            mf_gp: 已拟合的多保真度GP
            search_space: 搜索空间
            f_best_target: TARGET保真度下的当前最优值
            budget_used: 已使用的预算
            observations: 历史观测（用于自适应策略）

        Returns:
            FidelityDecision
        """
        # 确定可用保真度
        available = self._get_available_fidelities(
            budget_used, self.config.max_budget
        )

        strategy = self.config.fidelity_strategy

        if strategy == FidelitySelectionStrategy.COST_AWARE:
            decision = self._select_cost_aware(
                mf_gp, search_space, f_best_target, available
            )
        elif strategy == FidelitySelectionStrategy.INFORMATION_GAIN:
            decision = self._select_info_gain(
                mf_gp, search_space, f_best_target, available
            )
        elif strategy == FidelitySelectionStrategy.BUDGET_PROPORTIONAL:
            decision = self._select_budget_proportional(
                mf_gp, search_space, f_best_target, available
            )
        elif strategy == FidelitySelectionStrategy.SCHEDULED:
            decision = self._select_scheduled(
                mf_gp, search_space, f_best_target, available
            )
        elif strategy == FidelitySelectionStrategy.ADAPTIVE_THRESHOLD:
            if observations is None:
                observations = []
            decision = self._select_adaptive_threshold(
                mf_gp, search_space, f_best_target, available, observations
            )
        else:
            logger.warning(f"Unknown strategy {strategy}, fallback to COST_AWARE")
            decision = self._select_cost_aware(
                mf_gp, search_space, f_best_target, available
            )

        # 更新计数
        self.eval_counts[decision.selected_fidelity] += 1
        self.decision_history.append(decision)

        return decision

    # ------------------------------------------------------------------
    # 统计与诊断
    # ------------------------------------------------------------------

    def get_fidelity_distribution(self) -> Dict[str, float]:
        """获取各保真度的评估次数分布"""
        total = max(1, sum(self.eval_counts.values()))
        return {
            level.value: self.eval_counts[level] / total
            for level in [FidelityLevel.LOW, FidelityLevel.MEDIUM, FidelityLevel.HIGH]
        }

    def reset(self):
        """重置状态"""
        self.decision_history.clear()
        self.eval_counts.clear()
