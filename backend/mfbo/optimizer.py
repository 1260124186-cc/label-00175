# -*- coding: utf-8 -*-
"""
多保真度贝叶斯优化 (Multi-Fidelity Bayesian Optimization, MFBO)

核心优化循环，将廉价代理仿真与昂贵全精度仿真
纳入统一的贝叶斯优化框架：

算法流程（MFBO主循环）：
-----------------------------------------------------------
1. 初始化：各保真度采集初始样本（拉丁超立方/随机）
2. 拟合：用所有观测数据训练多保真度高斯过程(MF-GP)
3. 决策：
   a. 对每个候选保真度 s ∈ {LOW, MEDIUM, HIGH}:
      i.  在该保真度下最大化获取函数 α_s(x)
      ii. 得到 (x_s^*, α_s^*)
   b.  基于策略选择 (x^*, s^*) = argmax Score(x_s^*, s)
       （Score考虑成本、信息增益、代理误差等）
4. 评估：用选定保真度 s^* 评估 x^*，得到 y^*
5. 记录：将 (x^*, y^*, s^*) 加入数据集
6. 检查终止条件：
   - 达到最大迭代次数？
   - 计算预算耗尽？
   - 连续N次无改善？
   否则回到步骤2。
7. 返回：TARGET保真度下预测最优的 x

关键特性：
- 支持任意保真度评估函数（用户自定义callable）
- 5种保真度选择策略可切换
- 4种多保真度核函数
- 进度回调、早停、检查点支持
- 适合博士课题中的算法对比研究
"""

import numpy as np
from typing import Optional, Callable, Tuple, Dict, Any, List, Union
from dataclasses import dataclass, field
import logging
import time
import copy

from mfbo.schemas import (
    FidelityLevel,
    MFBOConfig,
    SearchSpace,
    Observation,
    IterationRecord,
    MFBOResult,
    FidelityCost,
    KernelType,
    AcquisitionFunctionType,
    FidelitySelectionStrategy,
)
from mfbo.mf_gp import MultiFidelityGP
from mfbo.acquisition import AcquisitionFunction, AcquisitionConfig
from mfbo.fidelity_strategy import FidelitySelector, FidelityDecision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 多保真度评估函数类型定义
# ---------------------------------------------------------------------------

MultiFidelityEvaluator = Callable[
    [np.ndarray, FidelityLevel],
    Union[float, Tuple[float, Dict[str, Any]]]
]
"""
多保真度评估函数签名：
    f(x: np.ndarray, fidelity: FidelityLevel) -> y: float 或 (y, metadata)

参数:
    x: (D,) 输入参数向量
    fidelity: 评估的保真度层级

返回:
    y: 标量目标值（最小化方向）
    metadata: 可选的额外信息（耗时、中间结果等）
"""


# ---------------------------------------------------------------------------
# 主优化器类
# ---------------------------------------------------------------------------

class MultiFidelityBayesianOptimizer:
    """
    多保真度贝叶斯优化器

    将廉价低保真度（代理仿真）与昂贵高保真度（全精度仿真）
    纳入统一贝叶斯优化循环，智能分配计算预算。

    典型用法（最小化问题）：
        >>> from mfbo import MultiFidelityBayesianOptimizer, MFBOConfig, SearchSpace
        >>> 
        >>> def my_evaluator(x, fidelity):
        ...     # 根据fidelity调用不同精度的仿真器
        ...     if fidelity == FidelityLevel.LOW:
        ...         return cheap_surrogate(x)
        ...     elif fidelity == FidelityLevel.MEDIUM:
        ...         return medium_simulation(x)
        ...     else:
        ...         return expensive_full_simulation(x)
        >>>
        >>> search_space = SearchSpace(bounds=[(0,1)]*10)
        >>> config = MFBOConfig(max_budget=30.0)
        >>> mfbo = MultiFidelityBayesianOptimizer(config, search_space)
        >>> result = mfbo.minimize(my_evaluator)
        >>> print(result.best_x, result.best_y)
    """

    def __init__(self,
                 config: Optional[MFBOConfig] = None,
                 search_space: Optional[SearchSpace] = None):
        """
        初始化MFBO优化器

        Args:
            config: 优化配置（核类型、预算、策略等）
            search_space: 搜索空间定义
        """
        self.config = config or MFBOConfig()
        self.search_space = search_space
        self.rng = np.random.default_rng(self.config.random_seed)

        # 核心组件
        self.mf_gp = MultiFidelityGP(self.config)
        self.acquisition = AcquisitionFunction(
            config=AcquisitionConfig(
                function_type=self.config.acquisition_type,
                ucb_beta=self.config.ucb_beta,
            ),
            mfbo_config=self.config,
        )
        self.fidelity_selector = FidelitySelector(
            config=self.config,
            acquisition=self.acquisition,
        )

        # 状态
        self.observations: List[Observation] = []
        self.history: List[IterationRecord] = []
        self.total_budget_used: float = 0.0
        self.total_time: float = 0.0

        # 最优记录
        self._best_x: Optional[np.ndarray] = None
        self._best_y: float = np.inf
        self._best_fidelity: FidelityLevel = FidelityLevel.HIGH

        # 早停状态
        self._iters_without_improvement: int = 0

    # ------------------------------------------------------------------
    # 初始采样
    # ------------------------------------------------------------------

    def _initial_sampling(self, evaluator: MultiFidelityEvaluator) -> None:
        """
        初始采样阶段：各保真度采集初始样本

        使用拉丁超立方采样（LHS）或均匀采样
        """
        cfg = self.config
        target = cfg.target_fidelity
        init_configs = [
            (FidelityLevel.LOW, cfg.n_init_low),
            (FidelityLevel.MEDIUM, cfg.n_init_medium),
            (FidelityLevel.HIGH, cfg.n_init_high),
        ]

        logger.info("=== MFBO 初始采样阶段 ===")

        # 生成所有初始点（不同保真度可共享x，提高数据效率）
        all_x_for_levels: Dict[FidelityLevel, np.ndarray] = {}

        for level, n in init_configs:
            if n <= 0:
                continue

            # 采样x点
            X_samples = self._latin_hypercube(n)
            X_samples = self.search_space.clip(X_samples)

            # 如果目标保真度也有初始点，让低保真度共享这些点的x
            if level < target and cfg.n_init_high > 0 and level == FidelityLevel.LOW:
                # 叠加一些高保真点的x
                n_extra = min(cfg.n_init_high, n // 3)
                if n_extra > 0:
                    X_extra = self._latin_hypercube(n_extra)
                    X_samples = np.vstack([X_samples, X_extra])

            all_x_for_levels[level] = X_samples

        # 按保真度由低到高评估（共享点可以提前计算相关性）
        for level, n in init_configs:
            if n <= 0:
                continue
            X_samples = all_x_for_levels[level]
            logger.info(f"  采集 {level.value}: {len(X_samples)} 个初始样本")

            for i, x in enumerate(X_samples):
                obs = self._evaluate_single(evaluator, x, level)
                self.observations.append(obs)
                self.total_budget_used += obs.cost
                self.total_time += obs.time

                # 检查是否更新最优
                if obs.fidelity >= self.config.target_fidelity and obs.y < self._best_y:
                    self._update_best(obs)

        logger.info(f"  初始采样完成，预算已用: {self.total_budget_used:.2f} "
                    f"/ {cfg.max_budget:.2f}")
        logger.info(f"  初始样本总数: {len(self.observations)}")

    def _latin_hypercube(self, n_samples: int) -> np.ndarray:
        """
        拉丁超立方采样（Latin Hypercube Sampling, LHS）

        比均匀采样有更好的空间覆盖性，适合初始实验设计。
        """
        D = self.search_space.dimensions
        samples = np.zeros((n_samples, D))

        for d in range(D):
            # 每个维度分成n个区间，每个区间均匀采1个
            perm = self.rng.permutation(n_samples)
            u = self.rng.uniform(size=n_samples)
            samples[:, d] = (perm + u) / n_samples

        # 映射到搜索空间边界
        bounds = np.array(self.search_space.bounds)
        for d in range(D):
            low, high = bounds[d]
            samples[:, d] = low + samples[:, d] * (high - low)

        return samples

    # ------------------------------------------------------------------
    # 单次评估包装
    # ------------------------------------------------------------------

    def _evaluate_single(self,
                         evaluator: MultiFidelityEvaluator,
                         x: np.ndarray,
                         fidelity: FidelityLevel
                         ) -> Observation:
        """
        执行一次评估，记录成本和时间

        Args:
            evaluator: 用户评估函数
            x: 输入点
            fidelity: 保真度

        Returns:
            Observation
        """
        x = np.asarray(x).reshape(-1)
        cost_config = self.config.cost_config

        t_start = time.time()
        cost_expected = cost_config.get_cost(fidelity)

        try:
            result = evaluator(x, fidelity)
            if isinstance(result, tuple):
                y, metadata = result[0], result[1] if len(result) > 1 else {}
            else:
                y, metadata = float(result), {}
        except Exception as e:
            logger.error(f"评估失败 x={x}, fidelity={fidelity.value}: {e}")
            y = np.inf
            metadata = {"error": str(e)}

        elapsed = time.time() - t_start

        # 更新实际成本记录（优先使用真实时间/成本）
        actual_cost = metadata.get("cost", cost_expected)

        return Observation(
            x=x,
            y=float(y),
            fidelity=fidelity,
            cost=float(actual_cost),
            time=float(elapsed),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # 更新最优解
    # ------------------------------------------------------------------

    def _update_best(self, obs: Observation) -> None:
        """更新最优解记录"""
        target = self.config.target_fidelity

        # 仅考虑目标保真度或更高保真度的观测
        if obs.fidelity >= target and obs.y < self._best_y:
            self._best_x = obs.x.copy()
            self._best_y = float(obs.y)
            self._best_fidelity = obs.fidelity
            self._iters_without_improvement = 0
            logger.debug(f"  ★ 找到更好的解: y={self._best_y:.6e} @ {obs.fidelity.value}")

    def _get_target_best(self) -> float:
        """
        获取TARGET保真度下的当前最优值

        如果TARGET保真度没有观测，则用最高的可用保真度估计
        """
        target = self.config.target_fidelity

        # 首先查找 TARGET 保真度的实际观测
        target_obs = [o for o in self.observations if o.fidelity == target]
        if len(target_obs) > 0:
            return min(o.y for o in target_obs)

        # 否则用所有观测中的最小值（带惩罚）
        if len(self.observations) > 0:
            all_min = min(o.y for o in self.observations)
            return float(all_min * 1.1)  # 惩罚低精度数据
        return 0.0

    # ------------------------------------------------------------------
    # 早停检查
    # ------------------------------------------------------------------

    def _check_termination(self, iteration: int) -> Tuple[bool, str]:
        """
        检查是否终止优化

        Returns:
            (should_stop, reason)
        """
        cfg = self.config

        if iteration >= cfg.max_iterations:
            return True, f"达到最大迭代次数 ({cfg.max_iterations})"

        if self.total_budget_used >= cfg.max_budget:
            return True, f"计算预算耗尽 ({self.total_budget_used:.2f} >= {cfg.max_budget:.2f})"

        if self._iters_without_improvement >= cfg.early_stop_patience:
            return True, (f"早停: 连续 {cfg.early_stop_patience} 次迭代 "
                         f"无超过 {cfg.min_improvement_threshold:.2e} 的改善")

        return False, ""

    # ------------------------------------------------------------------
    # 主优化入口：最小化
    # ------------------------------------------------------------------

    def minimize(self,
                 evaluator: MultiFidelityEvaluator,
                 search_space: Optional[SearchSpace] = None,
                 initial_observations: Optional[List[Observation]] = None,
                 ) -> MFBOResult:
        """
        运行多保真度贝叶斯优化（最小化目标函数）

        Args:
            evaluator: 多保真度评估函数 f(x, fidelity) -> y
            search_space: 搜索空间（可选，覆盖构造函数中的）
            initial_observations: 预存在的观测数据（热启动）

        Returns:
            MFBOResult 优化结果
        """
        if search_space is not None:
            self.search_space = search_space
        if self.search_space is None:
            raise ValueError("必须提供搜索空间（search_space）")

        # 重置状态
        self.observations = []
        self.history = []
        self.total_budget_used = 0.0
        self.total_time = 0.0
        self._best_x = None
        self._best_y = np.inf
        self._iters_without_improvement = 0
        self.fidelity_selector.reset()

        # 载入热启动数据
        if initial_observations:
            self.observations.extend(copy.deepcopy(initial_observations))
            for obs in initial_observations:
                self.total_budget_used += obs.cost
                self.total_time += obs.time
                if obs.fidelity >= self.config.target_fidelity:
                    self._update_best(obs)
            logger.info(f"热启动: 载入 {len(initial_observations)} 个预存在观测")

        total_start = time.time()

        # ------------------------------------------------------------------
        # 阶段1: 初始采样
        # ------------------------------------------------------------------
        if len(self.observations) == 0:
            self._initial_sampling(evaluator)
        else:
            logger.info("跳过初始采样（已存在观测数据）")

        # ------------------------------------------------------------------
        # 阶段2: 主优化循环
        # ------------------------------------------------------------------
        logger.info("=== MFBO 主优化循环开始 ===")
        logger.info(f"  核: {self.config.kernel_type.value} | "
                     f"获取函数: {self.config.acquisition_type.value} | "
                     f"策略: {self.config.fidelity_strategy.value}")
        logger.info(f"  目标保真度: {self.config.target_fidelity.value} | "
                     f"最大预算: {self.config.max_budget:.2f}")

        iteration = 0
        while True:
            iter_start = time.time()

            # 1. 拟合 MF-GP
            prev_best = self._best_y
            try:
                self.mf_gp.fit(self.observations)
                gp_nll = self.mf_gp.last_nll
            except Exception as e:
                logger.error(f"GP拟合失败 (iter {iteration}): {e}")
                # 回退：用简单最近邻估计
                gp_nll = np.inf

            # 2. 获取当前最优值（TARGET保真度）
            f_best = self._get_target_best()

            # 3. 决策：选择下一个评估点和保真度
            try:
                decision = self.fidelity_selector.select_next(
                    mf_gp=self.mf_gp,
                    search_space=self.search_space,
                    f_best_target=f_best,
                    budget_used=self.total_budget_used,
                    observations=self.observations,
                )
                next_x = decision.best_x
                next_fidelity = decision.selected_fidelity
                acq_value = decision.acq_value
            except Exception as e:
                logger.error(f"保真度决策失败，回退到随机+低保真: {e}")
                next_x = self.search_space.sample(1, rng=self.rng)[0]
                next_fidelity = FidelityLevel.LOW
                acq_value = 0.0

            # 4. 预测该点的统计量（用于记录）
            try:
                pred = self.mf_gp.predict(
                    next_x.reshape(1, -1), target_fidelity=next_fidelity
                )
                pred_mean = float(pred.mean[0])
                pred_std = float(pred.std[0])
            except Exception:
                pred_mean, pred_std = f_best, 1.0

            # 5. 实际评估
            obs = self._evaluate_single(evaluator, next_x, next_fidelity)
            self.observations.append(obs)
            self.total_budget_used += obs.cost

            # 6. 代理误差估计（如可用）
            surr_error = None
            try:
                errors = self.mf_gp.surrogate_fidelity_error()
                if (FidelityLevel.LOW, self.config.target_fidelity) in errors:
                    surr_error = errors[(FidelityLevel.LOW, self.config.target_fidelity)]
            except Exception:
                pass

            # 7. 更新最优
            old_best = self._best_y
            self._update_best(obs)
            if self._best_y == old_best and obs.y >= old_best - self.config.min_improvement_threshold:
                self._iters_without_improvement += 1

            # 8. 记录迭代
            iter_elapsed = time.time() - iter_start
            self.total_time += iter_elapsed

            record = IterationRecord(
                iteration=iteration,
                selected_x=obs.x.copy(),
                selected_fidelity=next_fidelity,
                acquisition_value=float(acq_value),
                predicted_mean=pred_mean,
                predicted_std=pred_std,
                observed_y=obs.y,
                cost_spent=obs.cost,
                best_y_so_far=self._best_y if np.isfinite(self._best_y) else 0.0,
                total_budget_used=self.total_budget_used,
                surrogate_error=surr_error,
            )
            self.history.append(record)

            # 9. 日志
            improvement_flag = "★" if obs.y < old_best else " "
            log_line = (
                f"Iter {iteration:3d}{improvement_flag} | "
                f"fid={next_fidelity.value:6s} | "
                f"y_obs={obs.y:.4e} | "
                f"μ/σ={pred_mean:.2e}/{pred_std:.2e} | "
                f"acq={acq_value:.2e} | "
                f"cost={obs.cost:.3f}({self.total_budget_used:.2f}) | "
                f"best={self._best_y:.4e} | "
                f"t={iter_elapsed:.1f}s"
            )
            logger.info(log_line)

            # 进度回调
            if self.config.progress_callback:
                try:
                    self.config.progress_callback(
                        iteration, self.config.max_iterations,
                        {"best_y": self._best_y, "budget_used": self.total_budget_used}
                    )
                except Exception:
                    pass

            iteration += 1

            # 10. 终止检查
            stop, reason = self._check_termination(iteration)
            if stop:
                logger.info(f"=== 优化终止: {reason} ===")
                break

        total_elapsed = time.time() - total_start

        # ------------------------------------------------------------------
        # 后处理：最终GP拟合一次，预测所有观测的TARGET保真度值，选最优
        # ------------------------------------------------------------------
        try:
            self.mf_gp.fit(self.observations)

            # 如果没有TARGET保真度观测，用GP预测找最优
            target = self.config.target_fidelity
            target_obs = [o for o in self.observations if o.fidelity == target]

            if len(target_obs) == 0 or (len(self.observations) > 0 and
                                        not np.isfinite(self._best_y)):
                # 采样大量候选点，预测TARGET保真度值
                X_search = self.search_space.sample(min(5000, 500 * self.search_space.dimensions),
                                                     rng=self.rng)
                pred = self.mf_gp.predict(X_search, target_fidelity=target)
                best_idx = int(np.argmin(pred.mean))
                x_predicted = X_search[best_idx]
                y_predicted = float(pred.mean[best_idx])

                # 与已有最优比较（如果有）
                if not np.isfinite(self._best_y) or y_predicted < self._best_y:
                    self._best_x = x_predicted
                    self._best_y = y_predicted
                    self._best_fidelity = target
                    logger.info(f"后处理: GP预测最优解 y={self._best_y:.6e}")
        except Exception as e:
            logger.warning(f"最终GP后处理失败: {e}")

        # 构造结果
        result = MFBOResult(
            best_x=self._best_x if self._best_x is not None else np.zeros(self.search_space.dimensions),
            best_y=float(self._best_y) if np.isfinite(self._best_y) else 0.0,
            best_fidelity=self._best_fidelity,
            n_iterations=iteration,
            total_budget_used=self.total_budget_used,
            total_time=total_elapsed,
            observations=self.observations,
            history=self.history,
            final_gp_nll=float(self.mf_gp.last_nll),
        )

        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # 便捷接口
    # ------------------------------------------------------------------

    def maximize(self,
                 evaluator: MultiFidelityEvaluator,
                 **kwargs) -> MFBOResult:
        """
        最大化目标函数（将evaluator取负）

        Args:
            evaluator: f(x, fidelity) -> y（越大越好）
            **kwargs: 同minimize

        Returns:
            MFBOResult（best_y是原始值，非取负后的值）
        """
        def neg_evaluator(x, fidelity):
            result = evaluator(x, fidelity)
            if isinstance(result, tuple):
                y, md = result
                return -float(y), md
            return -float(result)

        result = self.minimize(neg_evaluator, **kwargs)
        # 恢复符号
        result.best_y = -result.best_y
        for obs in result.observations:
            obs.y = -obs.y
        for h in result.history:
            h.observed_y = -h.observed_y
            h.best_y_so_far = -h.best_y_so_far
        return result

    # ------------------------------------------------------------------
    # 诊断与可视化辅助
    # ------------------------------------------------------------------

    def get_convergence_curve(self) -> Dict[str, np.ndarray]:
        """获取收敛曲线数据"""
        iters = np.array([h.iteration for h in self.history])
        budgets = np.array([h.total_budget_used for h in self.history])
        bests = np.array([h.best_y_so_far for h in self.history])
        fid_int = np.array([h.selected_fidelity.to_int() for h in self.history])
        return {
            "iterations": iters,
            "budget": budgets,
            "best_y": bests,
            "fidelity_selected": fid_int,
        }

    def get_fidelity_statistics(self) -> Dict[str, Any]:
        """获取保真度使用统计"""
        stats: Dict[str, Any] = {"counts": {}, "costs": {}, "times": {}}
        for level in [FidelityLevel.LOW, FidelityLevel.MEDIUM, FidelityLevel.HIGH]:
            obs_level = [o for o in self.observations if o.fidelity == level]
            stats["counts"][level.value] = len(obs_level)
            stats["costs"][level.value] = sum(o.cost for o in obs_level)
            stats["times"][level.value] = sum(o.time for o in obs_level)
        return stats

    def predict(self, X: np.ndarray,
                target_fidelity: FidelityLevel = FidelityLevel.HIGH):
        """对外暴露GP预测接口"""
        return self.mf_gp.predict(X, target_fidelity=target_fidelity)
