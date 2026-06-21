# -*- coding: utf-8 -*-
"""
贝叶斯推断模块

提供贝叶斯不确定性量化方法：
1. MCMC (Markov Chain Monte Carlo)
   - Metropolis-Hastings 算法
   - 自适应建议分布
   - 多链并行

2. ABC (Approximate Bayesian Computation, 近似贝叶斯计算)
   - 拒绝采样 ABC
   - ABC-SMC (Sequential Monte Carlo)
   - 适用于似然函数难以计算的仿真模型

适用于：
- 模型参数的后验分布估计
- 超参数不确定性量化
- 小样本条件下的鲁棒推断
- 复杂仿真模型的参数校准
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.stats import norm, multivariate_normal

from uq.schemas import (
    ConfidenceInterval,
    ParameterDistribution,
)

logger = logging.getLogger(__name__)


@dataclass
class BayesianInferenceConfig:
    """
    贝叶斯推断基础配置

    Attributes:
        n_samples: 后验样本数量
        n_burnin: 燃烧期样本数（丢弃）
        n_chains: MCMC 链数（用于收敛诊断）
        confidence_level: 置信水平（HPD 区间）
        random_seed: 随机种子
        progress_callback: 进度回调 callback(current, total)
    """
    n_samples: int = 5000
    n_burnin: int = 1000
    n_chains: int = 3
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    progress_callback: Optional[Callable[[int, int], None]] = None


@dataclass
class MCMCConfig(BayesianInferenceConfig):
    """
    MCMC 配置

    Attributes:
        proposal_std: 建议分布标准差，可以是标量或数组（各维度不同）
        adapt_interval: 自适应建议分布的间隔（样本数）
        target_accept_rate: 目标接受率（默认 0.234 为随机游走 Metropolis 最优）
        initial_values: 各链的初始值，None 则从先验采样
    """
    proposal_std: Union[float, np.ndarray] = 0.1
    adapt_interval: int = 100
    target_accept_rate: float = 0.234
    initial_values: Optional[List[np.ndarray]] = None


@dataclass
class ABCConfig(BayesianInferenceConfig):
    """
    ABC 配置

    Attributes:
        epsilon: 接受阈值（距离小于此值才接受）
        distance_metric: 距离度量函数
        summary_statistics: 摘要统计量函数（将数据/仿真结果压缩为低维向量）
        n_populations: ABC-SMC 的种群数
        alpha: ABC-SMC 每代保留比例
    """
    epsilon: float = 0.1
    distance_metric: str = "euclidean"
    summary_statistics: Optional[Callable[[np.ndarray], np.ndarray]] = None
    n_populations: int = 5
    alpha: float = 0.5


@dataclass
class PosteriorSample:
    """
    后验样本结果

    Attributes:
        parameter_names: 参数名称列表
        samples: 后验样本数组 (n_samples, n_params)
        log_likelihoods: 每个样本的对数似然 (n_samples,)
        log_priors: 每个样本的对数先验 (n_samples,)
        acceptance_rate: 接受率
        chain_ids: 每个样本所属的链 ID (n_samples,)
        n_eff: 有效样本数（各参数）
        r_hat: Gelman-Rubin 收敛诊断（各参数）
    """
    parameter_names: List[str]
    samples: np.ndarray
    log_likelihoods: Optional[np.ndarray] = None
    log_priors: Optional[np.ndarray] = None
    acceptance_rate: float = 0.0
    chain_ids: Optional[np.ndarray] = None
    n_eff: Optional[Dict[str, float]] = None
    r_hat: Optional[Dict[str, float]] = None

    @property
    def n_params(self) -> int:
        return self.samples.shape[1]

    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]

    def get_param_samples(self, name: str) -> np.ndarray:
        """获取指定参数的后验样本"""
        idx = self.parameter_names.index(name)
        return self.samples[:, idx]

    def compute_hpd(
        self,
        param_name: str,
        confidence_level: Optional[float] = None,
    ) -> ConfidenceInterval:
        """
        计算最高后验密度 (HPD) 区间

        Args:
            param_name: 参数名称
            confidence_level: 置信水平，None 则使用默认

        Returns:
            ConfidenceInterval
        """
        if confidence_level is None:
            confidence_level = 0.95

        samples = self.get_param_samples(param_name)
        return self._hpd_interval(samples, confidence_level)

    @staticmethod
    def _hpd_interval(samples: np.ndarray, cred_mass: float = 0.95) -> ConfidenceInterval:
        """使用 Chen-Shao 方法计算 HPD 区间"""
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        interval_idx_inc = int(np.floor(cred_mass * n))
        n_intervals = n - interval_idx_inc
        interval_widths = np.zeros(n_intervals)

        for i in range(n_intervals):
            interval_widths[i] = sorted_samples[i + interval_idx_inc] - sorted_samples[i]

        min_idx = int(np.argmin(interval_widths))
        hpd_low = float(sorted_samples[min_idx])
        hpd_high = float(sorted_samples[min_idx + interval_idx_inc])

        return ConfidenceInterval(
            lower=hpd_low,
            upper=hpd_high,
            level=cred_mass,
            method="hpd",
            point_estimate=float(np.median(samples)),
            standard_error=float(np.std(samples) / np.sqrt(n)),
        )

    def summary(self) -> str:
        lines = ["=== 后验样本摘要 ==="]
        lines.append(f"  参数: {self.parameter_names}")
        lines.append(f"  样本数: {self.n_samples}")
        lines.append(f"  接受率: {self.acceptance_rate:.3f}")

        for i, name in enumerate(self.parameter_names):
            s = self.samples[:, i]
            mean = np.mean(s)
            std = np.std(s)
            hpd = self.compute_hpd(name)
            lines.append(f"  {name}:")
            lines.append(f"    均值 ± std: {mean:.4f} ± {std:.4f}")
            lines.append(f"    中位数: {np.median(s):.4f}")
            lines.append(f"    95% HPD: [{hpd.lower:.4f}, {hpd.upper:.4f}]")
            if self.n_eff and name in self.n_eff:
                lines.append(f"    有效样本数 n_eff: {self.n_eff[name]:.1f}")
            if self.r_hat and name in self.r_hat:
                lines.append(f"    R-hat: {self.r_hat[name]:.3f}")

        return "\n".join(lines)


@dataclass
class BayesianResult:
    """
    贝叶斯推断完整结果

    Attributes:
        method: 使用的方法 ('mcmc' / 'abc')
        posterior: 后验样本
        prior_predictive: 先验预测样本（可选）
        posterior_predictive: 后验预测样本（可选）
        evidence: 边际似然估计（可选）
        total_time: 总计算时间
    """
    method: str
    posterior: PosteriorSample
    prior_predictive: Optional[np.ndarray] = None
    posterior_predictive: Optional[np.ndarray] = None
    evidence: Optional[float] = None
    total_time: float = 0.0

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        result = {
            "method": self.method,
            "posterior": {
                "parameter_names": self.posterior.parameter_names,
                "acceptance_rate": float(self.posterior.acceptance_rate),
                "n_eff": self.posterior.n_eff,
                "r_hat": self.posterior.r_hat,
            },
            "evidence": float(self.evidence) if self.evidence is not None else None,
            "total_time": float(self.total_time),
        }
        if include_samples:
            result["posterior"]["samples"] = self.posterior.samples.tolist()
            if self.posterior.log_likelihoods is not None:
                result["posterior"]["log_likelihoods"] = self.posterior.log_likelihoods.tolist()
        return result


class MCMCSampler:
    """
    MCMC 采样器（Metropolis-Hastings 算法）

    实现了带有自适应建议分布的随机游走 Metropolis-Hastings 算法，
    支持多链并行和收敛诊断。
    """

    def __init__(self, config: Optional[MCMCConfig] = None):
        """
        初始化 MCMC 采样器

        Args:
            config: MCMC 配置
        """
        self.config = config if config is not None else MCMCConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def sample(
        self,
        log_posterior: Callable[[np.ndarray], float],
        parameter_names: List[str],
        prior_distributions: Optional[List[ParameterDistribution]] = None,
    ) -> BayesianResult:
        """
        运行 MCMC 采样

        Args:
            log_posterior: 对数后验密度函数 log_p(θ|y)
            parameter_names: 参数名称列表
            prior_distributions: 先验分布列表（用于采样初始值）

        Returns:
            BayesianResult
        """
        import time

        t_start = time.time()
        n_params = len(parameter_names)
        n_chains = self.config.n_chains
        n_total = self.config.n_samples + self.config.n_burnin

        proposal_std = self.config.proposal_std
        if np.isscalar(proposal_std):
            proposal_std = np.full(n_params, float(proposal_std))
        proposal_std = np.asarray(proposal_std, dtype=np.float64)

        if self.config.initial_values is not None:
            initial_values = [np.asarray(v, dtype=np.float64) for v in self.config.initial_values]
            if len(initial_values) != n_chains:
                raise ValueError(f"initial_values 数量 ({len(initial_values)}) 与 n_chains ({n_chains}) 不匹配")
        else:
            initial_values = []
            for _ in range(n_chains):
                if prior_distributions is not None:
                    theta = []
                    for prior in prior_distributions:
                        sample = self._sample_from_prior(prior)
                        theta.append(sample)
                    initial_values.append(np.array(theta, dtype=np.float64))
                else:
                    initial_values.append(self.rng.normal(0, 1, size=n_params))

        all_samples = []
        all_log_posts = []
        all_accepts = []

        for chain_idx in range(n_chains):
            samples, log_posts, n_accept = self._run_single_chain(
                log_posterior=log_posterior,
                initial_value=initial_values[chain_idx],
                proposal_std=proposal_std.copy(),
                n_total=n_total,
                n_params=n_params,
            )
            all_samples.append(samples[self.config.n_burnin :])
            all_log_posts.append(log_posts[self.config.n_burnin :])
            all_accepts.append(n_accept)

        combined_samples = np.vstack(all_samples)
        combined_log_posts = np.concatenate(all_log_posts)
        chain_ids = np.concatenate(
            [np.full(len(s), i, dtype=int) for i, s in enumerate(all_samples)]
        )

        acceptance_rate = float(np.mean(all_accepts) / n_total)

        n_eff = self._compute_effective_sample_size(all_samples, parameter_names)
        r_hat = self._compute_r_hat(all_samples, parameter_names)

        log_priors = None
        if prior_distributions is not None:
            log_priors = self._compute_log_priors(combined_samples, prior_distributions)

        posterior = PosteriorSample(
            parameter_names=parameter_names,
            samples=combined_samples,
            log_likelihoods=combined_log_posts,
            log_priors=log_priors,
            acceptance_rate=acceptance_rate,
            chain_ids=chain_ids,
            n_eff=n_eff,
            r_hat=r_hat,
        )

        total_time = time.time() - t_start
        return BayesianResult(
            method="mcmc",
            posterior=posterior,
            total_time=total_time,
        )

    def _run_single_chain(
        self,
        log_posterior: Callable[[np.ndarray], float],
        initial_value: np.ndarray,
        proposal_std: np.ndarray,
        n_total: int,
        n_params: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """运行单条 MCMC 链"""
        samples = np.zeros((n_total, n_params))
        log_posts = np.zeros(n_total)
        n_accept = 0

        current = initial_value.copy()
        current_log_post = log_posterior(current)

        samples[0] = current
        log_posts[0] = current_log_post

        for i in range(1, n_total):
            if i % self.config.adapt_interval == 0 and i < self.config.n_burnin:
                if n_accept > 0:
                    current_rate = n_accept / i
                    scaling = np.exp(
                        0.5 * (current_rate - self.config.target_accept_rate)
                    )
                    proposal_std *= np.clip(scaling, 0.5, 2.0)

            proposal = current + self.rng.normal(0, proposal_std)

            try:
                proposal_log_post = log_posterior(proposal)
            except Exception:
                proposal_log_post = -np.inf

            log_alpha = proposal_log_post - current_log_post
            if np.log(self.rng.uniform()) < log_alpha:
                current = proposal
                current_log_post = proposal_log_post
                n_accept += 1

            samples[i] = current
            log_posts[i] = current_log_post

            if self.config.progress_callback is not None and i % 100 == 0:
                self.config.progress_callback(i, n_total)

        return samples, log_posts, n_accept

    def _sample_from_prior(self, dist: ParameterDistribution) -> float:
        """从先验分布采样单个值"""
        dtype = dist.distribution_type.lower()
        if dtype == "normal":
            std = dist.params.get("std", 1.0)
            return float(self.rng.normal(dist.nominal, std))
        elif dtype == "uniform":
            low = dist.params.get("low", dist.nominal - 1)
            high = dist.params.get("high", dist.nominal + 1)
            return float(self.rng.uniform(low, high))
        elif dtype == "lognormal":
            sigma = dist.params.get("sigma", 1.0)
            mu = dist.params.get("mu", np.log(max(dist.nominal, 1e-10)))
            return float(self.rng.lognormal(mu, sigma))
        else:
            std = dist.params.get("std", 1.0)
            return float(self.rng.normal(dist.nominal, std))

    def _compute_log_priors(
        self, samples: np.ndarray, priors: List[ParameterDistribution]
    ) -> np.ndarray:
        """计算每个样本的对数先验"""
        n = samples.shape[0]
        log_priors = np.zeros(n)
        for j, prior in enumerate(priors):
            s = samples[:, j]
            dtype = prior.distribution_type.lower()
            if dtype == "normal":
                std = prior.params.get("std", 1.0)
                log_priors += norm.logpdf(s, prior.nominal, std)
            elif dtype == "uniform":
                low = prior.params.get("low", prior.nominal - 1)
                high = prior.params.get("high", prior.nominal + 1)
                log_priors += stats.uniform.logpdf(s, low, high - low)
        return log_priors

    def _compute_effective_sample_size(
        self, chains: List[np.ndarray], param_names: List[str]
    ) -> Dict[str, float]:
        """计算各参数的有效样本数"""
        n_params = len(param_names)
        n_eff = {}
        for j in range(n_params):
            all_chain_data = [c[:, j] for c in chains]
            n_eff[param_names[j]] = float(
                self._multi_chain_effective_n(all_chain_data)
            )
        return n_eff

    def _multi_chain_effective_n(self, chains: List[np.ndarray]) -> float:
        """多链有效样本数（Gelman 的 split-R 辅助方法）"""
        m = len(chains)
        if m < 2:
            return float(len(chains[0]))

        all_samples = np.concatenate(chains)
        n = len(chains[0])

        chain_means = np.array([np.mean(c) for c in chains])
        grand_mean = np.mean(all_samples)

        B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
        W = np.mean([np.var(c, ddof=1) for c in chains])

        var_hat = (1 - 1 / n) * W + (1 / n) * B
        if var_hat < 1e-15:
            return float(len(all_samples))

        rho_hat_sum = 0.0
        for lag in range(1, min(n, 100)):
            autocov_chain = []
            for c in chains:
                c_centered = c - np.mean(c)
                if n - lag > 0:
                    autocov = np.mean(c_centered[:-lag] * c_centered[lag:])
                    autocov_chain.append(autocov)
            V_t = np.mean(autocov_chain) if autocov_chain else 0.0
            rho_hat = 1.0 - (W - V_t) / (2 * var_hat) if var_hat > 1e-15 else 0.0
            if rho_hat < 0.05:
                break
            rho_hat_sum += rho_hat

        n_eff = len(all_samples) / (1 + 2 * rho_hat_sum)
        return max(1.0, n_eff)

    def _compute_r_hat(
        self, chains: List[np.ndarray], param_names: List[str]
    ) -> Dict[str, float]:
        """计算 Gelman-Rubin R-hat 收敛诊断"""
        n_params = len(param_names)
        r_hat = {}
        m = len(chains)
        if m < 2:
            for name in param_names:
                r_hat[name] = float("nan")
            return r_hat

        for j in range(n_params):
            all_chain_data = [c[:, j] for c in chains]
            n = len(all_chain_data[0])

            chain_means = np.array([np.mean(c) for c in all_chain_data])
            chain_vars = np.array([np.var(c, ddof=1) for c in all_chain_data])
            grand_mean = np.mean(np.concatenate(all_chain_data))

            B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
            W = np.mean(chain_vars)

            if W < 1e-15:
                r_hat[param_names[j]] = 1.0
            else:
                var_hat = (1 - 1 / n) * W + (1 / n) * B
                r_hat[param_names[j]] = float(np.sqrt(var_hat / W))

        return r_hat


class ABCSampler:
    """
    近似贝叶斯计算 (ABC) 采样器

    适用于似然函数难以计算但可以从模型生成仿真数据的情况。
    实现了拒绝采样 ABC 和 ABC-SMC (Sequential Monte Carlo)。
    """

    def __init__(self, config: Optional[ABCConfig] = None):
        """
        初始化 ABC 采样器

        Args:
            config: ABC 配置
        """
        self.config = config if config is not None else ABCConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def sample_rejection(
        self,
        observed_data: np.ndarray,
        simulator: Callable[[np.ndarray], np.ndarray],
        prior_sampler: Callable[[], np.ndarray],
        parameter_names: List[str],
        n_samples: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> BayesianResult:
        """
        拒绝采样 ABC

        Args:
            observed_data: 观测数据
            simulator: 仿真函数 simulator(params) -> simulated_data
            prior_sampler: 先验采样函数 prior_sampler() -> params
            parameter_names: 参数名称列表
            n_samples: 样本数，None 则使用配置值
            epsilon: 接受阈值，None 则使用配置值

        Returns:
            BayesianResult
        """
        import time

        t_start = time.time()
        if n_samples is None:
            n_samples = self.config.n_samples
        if epsilon is None:
            epsilon = self.config.epsilon

        if self.config.summary_statistics is not None:
            obs_sum = self.config.summary_statistics(observed_data)
        else:
            obs_sum = observed_data.ravel()

        if epsilon is None:
            epsilon = self.config.epsilon
        if epsilon is None or epsilon <= 0:
            epsilon = self._auto_tune_epsilon(
                obs_sum, simulator, prior_sampler,
                self.config.summary_statistics, n_samples,
            )
            logger.info(f"ABC auto-tuned epsilon = {epsilon:.4g}")

        accepted_params = []
        accepted_distances = []
        total_simulations = 0
        max_simulations = n_samples * 1000
        candidate_params = []
        candidate_distances = []

        for attempt in range(3):
            while len(accepted_params) < n_samples and total_simulations < max_simulations:
                params = prior_sampler()
                simulated = simulator(params)

                if self.config.summary_statistics is not None:
                    sim_sum = self.config.summary_statistics(simulated)
                else:
                    sim_sum = simulated.ravel()

                distance = self._compute_distance(obs_sum, sim_sum)
                total_simulations += 1
                candidate_params.append(params)
                candidate_distances.append(distance)

                if distance <= epsilon:
                    accepted_params.append(params)
                    accepted_distances.append(distance)

                if self.config.progress_callback is not None and total_simulations % 100 == 0:
                    self.config.progress_callback(len(accepted_params), n_samples)

            if len(accepted_params) > 0:
                break
            if attempt < 2:
                epsilon *= 3.0
                logger.info(f"放宽 ABC epsilon 至 {epsilon:.4g} 重试")

        if len(accepted_params) == 0:
            if len(candidate_distances) >= n_samples:
                order = np.argsort(candidate_distances)[:n_samples]
                accepted_params = [candidate_params[i] for i in order]
                accepted_distances = [candidate_distances[i] for i in order]
                logger.warning(
                    f"ABC 未在 epsilon={epsilon:.3g} 下接受任何样本，"
                    f"回退到距离最近的 {n_samples} 个样本"
                )
            else:
                raise RuntimeError(
                    f"ABC 拒绝采样未接受任何样本（epsilon={epsilon}, "
                    f"尝试了 {total_simulations} 次仿真）。建议增大 epsilon。"
                )

        samples = np.array(accepted_params)
        acceptance_rate = len(accepted_params) / total_simulations

        posterior = PosteriorSample(
            parameter_names=parameter_names,
            samples=samples,
            log_likelihoods=np.array(accepted_distances),
            acceptance_rate=acceptance_rate,
        )

        total_time = time.time() - t_start
        return BayesianResult(
            method="abc_rejection",
            posterior=posterior,
            total_time=total_time,
        )

    def sample_smc(
        self,
        observed_data: np.ndarray,
        simulator: Callable[[np.ndarray], np.ndarray],
        prior_sampler: Callable[[], np.ndarray],
        prior_density: Callable[[np.ndarray], float],
        parameter_names: List[str],
        n_samples: Optional[int] = None,
        n_populations: Optional[int] = None,
    ) -> BayesianResult:
        """
        ABC-SMC (Approximate Bayesian Computation - Sequential Monte Carlo)

        通过逐步降低阈值 epsilon 提高采样效率。

        Args:
            observed_data: 观测数据
            simulator: 仿真函数
            prior_sampler: 先验采样函数
            prior_density: 先验密度函数
            parameter_names: 参数名称列表
            n_samples: 样本数
            n_populations: 种群数

        Returns:
            BayesianResult
        """
        import time

        t_start = time.time()
        if n_samples is None:
            n_samples = self.config.n_samples
        if n_populations is None:
            n_populations = self.config.n_populations

        if self.config.summary_statistics is not None:
            obs_sum = self.config.summary_statistics(observed_data)
        else:
            obs_sum = observed_data.ravel()

        alpha = self.config.alpha
        n_keep = max(1, int(alpha * n_samples))

        all_distances = []
        for _ in range(n_samples * 2):
            params = prior_sampler()
            simulated = simulator(params)
            if self.config.summary_statistics is not None:
                sim_sum = self.config.summary_statistics(simulated)
            else:
                sim_sum = simulated.ravel()
            all_distances.append(self._compute_distance(obs_sum, sim_sum))

        all_distances.sort()
        epsilons = []
        for t in range(n_populations):
            quantile = 1.0 - (t + 1) / n_populations * 0.9
            eps_idx = min(len(all_distances) - 1, max(0, int(quantile * len(all_distances))))
            epsilons.append(all_distances[eps_idx])
        epsilons = sorted(epsilons, reverse=True)

        particles = []
        weights = []
        distances = []
        for _ in range(n_samples):
            params = prior_sampler()
            simulated = simulator(params)
            if self.config.summary_statistics is not None:
                sim_sum = self.config.summary_statistics(simulated)
            else:
                sim_sum = simulated.ravel()
            dist = self._compute_distance(obs_sum, sim_sum)
            particles.append(params)
            weights.append(1.0 / n_samples)
            distances.append(dist)

        particles = np.array(particles)
        weights = np.array(weights)
        distances = np.array(distances)

        for t, eps in enumerate(epsilons):
            accept_mask = distances <= eps
            if np.sum(accept_mask) < n_keep:
                accept_mask = distances <= np.percentile(distances, alpha * 100)

            keep_idx = np.where(accept_mask)[0]
            if len(keep_idx) == 0:
                logger.warning(f"Population {t}: 无满足条件的粒子，跳过")
                continue

            keep_particles = particles[keep_idx]
            keep_weights = weights[keep_idx]
            keep_weights = keep_weights / keep_weights.sum()

            if t < n_populations - 1:
                cov = np.cov(keep_particles.T, aweights=keep_weights) * 2.0
                cov = np.atleast_2d(cov)

                new_particles = []
                new_weights = []
                new_distances = []

                for _ in range(n_samples):
                    idx = self.rng.choice(len(keep_particles), p=keep_weights)
                    mean = keep_particles[idx]
                    if cov.shape[0] == 1:
                        proposal = self.rng.normal(mean, np.sqrt(cov[0, 0]))
                    else:
                        proposal = self.rng.multivariate_normal(mean, cov)

                    simulated = simulator(proposal)
                    if self.config.summary_statistics is not None:
                        sim_sum = self.config.summary_statistics(simulated)
                    else:
                        sim_sum = simulated.ravel()
                    dist = self._compute_distance(obs_sum, sim_sum)

                    if dist <= eps:
                        prior_prob = prior_density(proposal)
                        if prior_prob <= 0:
                            continue

                        kernel_sum = 0.0
                        for i, kp in enumerate(keep_particles):
                            if cov.shape[0] == 1:
                                kernel_val = norm.pdf(proposal, kp, np.sqrt(cov[0, 0]))
                            else:
                                kernel_val = multivariate_normal.pdf(proposal, kp, cov)
                            kernel_sum += keep_weights[i] * kernel_val

                        weight = prior_prob / max(kernel_sum, 1e-15)
                        new_particles.append(proposal)
                        new_weights.append(weight)
                        new_distances.append(dist)

                if len(new_particles) > 0:
                    particles = np.array(new_particles)
                    weights = np.array(new_weights)
                    weights = weights / weights.sum()
                    distances = np.array(new_distances)

        if len(particles) < n_samples:
            logger.warning(f"最终粒子数 ({len(particles)}) 少于目标 ({n_samples})")

        acceptance_rate = len(particles) / (n_populations * n_samples)

        posterior = PosteriorSample(
            parameter_names=parameter_names,
            samples=particles,
            log_likelihoods=-distances,
            acceptance_rate=acceptance_rate,
        )

        total_time = time.time() - t_start
        return BayesianResult(
            method="abc_smc",
            posterior=posterior,
            total_time=total_time,
        )

    def _compute_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算距离度量"""
        if self.config.distance_metric == "euclidean":
            return float(np.sqrt(np.sum((x - y) ** 2)))
        elif self.config.distance_metric == "manhattan":
            return float(np.sum(np.abs(x - y)))
        elif self.config.distance_metric == "mahalanobis":
            diff = x - y
            cov = np.cov(np.vstack([x, y]).T)
            try:
                inv_cov = np.linalg.inv(cov + 1e-10 * np.eye(cov.shape[0]))
                return float(np.sqrt(diff @ inv_cov @ diff))
            except Exception:
                return float(np.sqrt(np.sum(diff ** 2)))
        else:
            return float(np.sqrt(np.sum((x - y) ** 2)))

    def _auto_tune_epsilon(
        self,
        obs_sum: np.ndarray,
        simulator: Callable[[np.ndarray], np.ndarray],
        prior_sampler: Callable[[], np.ndarray],
        summary_statistics: Optional[Callable[[np.ndarray], np.ndarray]],
        n_samples: int,
        n_calib: int = 200,
        acceptance_quantile: float = 0.1,
    ) -> float:
        """
        通过先验仿真自动估计 epsilon 阈值

        Args:
            obs_sum: 观测数据摘要统计量
            simulator: 仿真函数
            prior_sampler: 先验采样函数
            summary_statistics: 摘要统计量函数
            n_samples: 目标样本数
            n_calib: 校准仿真次数
            acceptance_quantile: 选取距离的此分位数作为 epsilon

        Returns:
            自动校准的 epsilon
        """
        distances = []
        for _ in range(n_calib):
            params = prior_sampler()
            simulated = simulator(params)
            if summary_statistics is not None:
                sim_sum = summary_statistics(simulated)
            else:
                sim_sum = simulated.ravel()
            distances.append(self._compute_distance(obs_sum, sim_sum))

        if not distances:
            return 1.0
        distances_arr = np.array(distances)
        q = max(acceptance_quantile, n_samples / max(n_calib, 1))
        q = min(q, 0.9)
        return float(np.quantile(distances_arr, q))


def bayesian_inference(
    log_posterior_or_observed: Union[Callable[[np.ndarray], float], np.ndarray],
    parameter_names: List[str],
    method: str = "mcmc",
    config: Optional[BayesianInferenceConfig] = None,
    **kwargs,
) -> BayesianResult:
    """
    便捷函数：执行贝叶斯推断

    Args:
        log_posterior_or_observed: MCMC 时为对数后验函数，ABC 时为观测数据
        parameter_names: 参数名称列表
        method: 方法 ('mcmc', 'abc_rejection', 'abc_smc')
        config: 推断配置
        **kwargs: 额外参数

    Returns:
        BayesianResult
    """
    if method.lower() == "mcmc":
        mcmc_config = config if isinstance(config, MCMCConfig) else MCMCConfig()
        sampler = MCMCSampler(mcmc_config)
        return sampler.sample(
            log_posterior=log_posterior_or_observed,
            parameter_names=parameter_names,
            **kwargs,
        )
    elif method.lower() in ("abc", "abc_rejection"):
        abc_config = config if isinstance(config, ABCConfig) else ABCConfig()
        sampler = ABCSampler(abc_config)
        return sampler.sample_rejection(
            observed_data=log_posterior_or_observed,
            parameter_names=parameter_names,
            **kwargs,
        )
    elif method.lower() == "abc_smc":
        abc_config = config if isinstance(config, ABCConfig) else ABCConfig()
        sampler = ABCSampler(abc_config)
        return sampler.sample_smc(
            observed_data=log_posterior_or_observed,
            parameter_names=parameter_names,
            **kwargs,
        )
    else:
        raise ValueError(f"未知贝叶斯推断方法: {method}")
