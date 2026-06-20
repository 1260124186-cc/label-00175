# -*- coding: utf-8 -*-
"""
参数反演引擎

基于 CD-SEM 量测数据反演光刻模型参数，支持：
1. 非线性最小二乘法 (NLLS, scipy.optimize.least_squares)
2. Levenberg-Marquardt (scipy.optimize.curve_fit)
3. 贝叶斯 MCMC (可选使用 emcee，若未安装则采用 numpy 随机游走 Metropolis)

核心思想：
    minimize  Σ  w_i * (measured_CD_i - model(focus_i, dose_i, params))²
    其中 w_i = 1 / (measurement_uncertainty_i)²   （加权最小二乘）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging
import time

from scipy.optimize import least_squares, curve_fit

from .schemas import (
    CalibrationParameter,
    CalibrationParameterSet,
    CDSEMDataset,
    InversionMethod,
    CalibrationConfig,
    InversionResult,
)
from .forward_model import (
    LithoForwardModel,
    model_prediction,
    compute_bossung_cd,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具函数：残差、代价、加权
# ---------------------------------------------------------------------------

def _prepare_arrays(dataset: CDSEMDataset):
    focuses, doses = dataset.focus_dose_grid()
    measured = dataset.measured_cds()
    target_cds = dataset.target_cds()
    pitches = dataset.pitches()
    pattern_types = dataset.pattern_types()
    uncertainties = dataset.uncertainties()
    return focuses, doses, measured, target_cds, pitches, pattern_types, uncertainties


def _residuals(params_vec: np.ndarray,
               param_names: List[str],
               fixed_params: Dict[str, float],
               focuses: np.ndarray,
               doses: np.ndarray,
               measured: np.ndarray,
               target_cds: np.ndarray,
               pitches: np.ndarray,
               pattern_types: List,
               weights: np.ndarray,
               complexity: str,
               ) -> np.ndarray:
    """加权残差向量：w * (predicted - measured)。"""
    preds = model_prediction(
        params_vec, param_names, fixed_params,
        focuses, doses, target_cds, pitches, pattern_types, complexity,
    )
    return weights * (preds - measured)


def _log_posterior(params_vec: np.ndarray,
                   param_names: List[str],
                   fixed_params: Dict[str, float],
                   focuses: np.ndarray,
                   doses: np.ndarray,
                   measured: np.ndarray,
                   target_cds: np.ndarray,
                   pitches: np.ndarray,
                   pattern_types: List,
                   weights: np.ndarray,
                   complexity: str,
                   lbs: np.ndarray,
                   ubs: np.ndarray,
                   param_objects: List[CalibrationParameter],
                   ) -> float:
    """
    对数后验（贝叶斯 MCMC 使用）。

    log P(θ|y) ∝ log P(y|θ) + log P(θ)
    其中：
    - log P(y|θ) = -½ Σ w_i² (pred - meas)²   （高斯似然）
    - log P(θ)   = 高斯 / 均匀 先验
    """
    # 先验：边界外为 -∞
    if np.any(params_vec < lbs) or np.any(params_vec > ubs):
        return -np.inf

    log_prior = 0.0
    for j, pobj in enumerate(param_objects):
        if pobj.prior_mean is not None and pobj.prior_std is not None:
            mu, s = pobj.prior_mean, pobj.prior_std
            if s > 0:
                log_prior += -0.5 * ((params_vec[j] - mu) / s) ** 2

    # 似然
    preds = model_prediction(
        params_vec, param_names, fixed_params,
        focuses, doses, target_cds, pitches, pattern_types, complexity,
    )
    # weights 是 1/sigma_meas，平方后与似然公式一致
    residuals = weights * (preds - measured)
    log_likelihood = -0.5 * np.sum(residuals ** 2)

    return log_likelihood + log_prior


# ---------------------------------------------------------------------------
# 非线性最小二乘法 (scipy.optimize.least_squares)
# ---------------------------------------------------------------------------

def nlls_inversion(config: CalibrationConfig,
                   dataset: CDSEMDataset,
                   ) -> InversionResult:
    """
    非线性最小二乘法参数反演。

    使用 scipy.optimize.least_squares，支持边界约束与雅可比（数值差分）。
    """
    t0 = time.time()

    # 提取数据
    (focuses, doses, measured, target_cds, pitches,
     pattern_types, uncertainties) = _prepare_arrays(dataset)

    # 权重：加权最小二乘
    if config.use_measurement_weights:
        weights = 1.0 / np.maximum(uncertainties, 1e-3)
    else:
        weights = np.ones_like(measured)

    # 参数
    varying_params = config.parameters.get_varying_parameters()
    fixed_params = {p.name: p.initial_value
                    for p in config.parameters.get_fixed_parameters()}
    param_names = [p.name for p in varying_params]
    x0 = np.array([p.initial_value for p in varying_params])
    lbs, ubs = config.parameters.bounds()

    n_data = len(measured)
    n_params = len(param_names)
    dof = max(n_data - n_params, 1)

    logger.info(
        f"NLLS 反演开始：{n_data} 个数据点，{n_params} 个自由参数 "
        f"({', '.join(param_names)})"
    )

    # 残差函数（仅需 residuals 即可）
    def _residual_fun(x):
        return _residuals(
            x, param_names, fixed_params,
            focuses, doses, measured, target_cds, pitches, pattern_types,
            weights, config.forward_model_complexity,
        )

    # 求解
    nlls_method = config.nlls_method
    if nlls_method == 'lm' and any(np.isfinite(lbs)):
        logger.warning("Levenberg-Marquardt 不支持边界约束，自动切换到 'trf'")
        nlls_method = 'trf'

    try:
        result = least_squares(
            _residual_fun,
            x0=x0,
            method=nlls_method,
            bounds=(lbs, ubs),
            max_nfev=config.nlls_max_iter,
            ftol=config.nlls_ftol,
            xtol=config.nlls_xtol,
            gtol=1e-12,
            diff_step=None,
            verbose=0,
        )
    except Exception as e:
        logger.error(f"least_squares 抛出异常: {e}")
        return _make_failed_result(
            method=InversionMethod.NLLS,
            message=f"least_squares exception: {e}",
            varying=varying_params,
            fixed_params=fixed_params,
            n_data=n_data,
        )

    # 协方差矩阵估计
    x_opt = result.x
    J = np.zeros((n_data, n_params), dtype=np.float64)
    eps = 1e-5
    for j in range(n_params):
        step = max(abs(x_opt[j]) * eps, eps * 1e-2)
        r_plus = _residual_fun(x_opt + np.eye(n_params)[j] * step)
        r_minus = _residual_fun(x_opt - np.eye(n_params)[j] * step)
        J[:, j] = (r_plus - r_minus) / (2 * step)

    try:
        JtJ = J.T @ J
        # 代价 = ½ Σ r²
        cost_final = 0.5 * np.sum(result.fun ** 2)
        # 残差方差 = 2 * cost_final / dof（假设加权残差正确）
        s2 = 2 * cost_final / dof if dof > 0 else 1.0
        cov = s2 * np.linalg.pinv(JtJ)
        stds = np.sqrt(np.maximum(np.diag(cov), 0.0))
        # 相关系数
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = cov / np.outer(stds, stds)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.warning(f"协方差估计失败: {e}，使用零不确定度")
        cov = np.zeros((n_params, n_params))
        stds = np.zeros(n_params)
        corr = np.eye(n_params)

    # 最终预测、残差
    preds = model_prediction(
        x_opt, param_names, fixed_params,
        focuses, doses, target_cds, pitches, pattern_types,
        config.forward_model_complexity,
    )
    raw_residuals = measured - preds

    calibrated = {name: float(val) for name, val in zip(param_names, x_opt)}
    for name, val in fixed_params.items():
        calibrated[name] = float(val)
    uncert_dict = {name: float(std) for name, std in zip(param_names, stds)}
    for name in fixed_params:
        uncert_dict[name] = 0.0

    # 卡方
    meas_sigma = np.maximum(uncertainties, 1e-3)
    chi2 = float(np.sum(((measured - preds) / meas_sigma) ** 2)) \
        if config.use_measurement_weights else float(np.sum((measured - preds) ** 2))
    reduced_chi2 = chi2 / dof if dof > 0 else chi2

    dt = time.time() - t0
    logger.info(
        f"NLLS 完成：成功={result.success}, 迭代={result.nfev}, "
        f"代价={result.cost:.6e}, 耗时={dt:.2f}s"
    )

    return InversionResult(
        method=InversionMethod.NLLS,
        success=bool(result.success),
        message=str(result.message),
        calibrated_values=calibrated,
        uncertainties=uncert_dict,
        covariance_matrix=cov,
        correlation_matrix=corr,
        varying_names=param_names,
        cost=float(result.cost),
        chi2=chi2,
        reduced_chi2=reduced_chi2,
        n_data=n_data,
        n_params=n_params,
        dof=dof,
        residuals=raw_residuals,
        predicted_cds=preds,
        iterations=int(result.nfev),
    )


# ---------------------------------------------------------------------------
# Levenberg-Marquardt / TRF (scipy.optimize.least_squares)
# ---------------------------------------------------------------------------

def lmfit_inversion(config: CalibrationConfig,
                    dataset: CDSEMDataset,
                    ) -> InversionResult:
    """
    Levenberg-Marquardt 参数反演（基于 least_squares，支持 TRF/LM 方法）。

    - 若存在边界约束：使用 Trust Region Reflective (TRF) 方法
    - 若无边界约束：优先使用经典 Levenberg-Marquardt (lm)
    """
    t0 = time.time()

    (focuses, doses, measured, target_cds, pitches,
     pattern_types, uncertainties) = _prepare_arrays(dataset)

    varying_params = config.parameters.get_varying_parameters()
    fixed_params = {p.name: p.initial_value
                    for p in config.parameters.get_fixed_parameters()}
    param_names = [p.name for p in varying_params]
    x0 = np.array([p.initial_value for p in varying_params])
    lbs, ubs = config.parameters.bounds()

    n_data = len(measured)
    n_params = len(param_names)
    dof = max(n_data - n_params, 1)

    # 加权残差函数（与 NLLS 完全一致，确保复用性）
    if config.use_measurement_weights:
        weights = 1.0 / np.maximum(uncertainties, 1e-3)
    else:
        weights = np.ones_like(measured)

    def _residual_fun(x):
        return _residuals(
            x, param_names, fixed_params,
            focuses, doses, measured, target_cds, pitches, pattern_types,
            weights, config.forward_model_complexity,
        )

    # 求解方法：LM 适合无边界；TRF 适合有边界
    has_bounds = any(np.isfinite(lbs)) or any(np.isfinite(ubs))
    if has_bounds:
        solver_method = config.nlls_method if config.nlls_method != 'lm' else 'trf'
    else:
        solver_method = config.nlls_method

    logger.info(
        f"LMFIT 反演开始：{n_data} 个数据点，{n_params} 个自由参数，"
        f"求解器={solver_method}"
    )

    try:
        result = least_squares(
            _residual_fun,
            x0=x0,
            method=solver_method,
            bounds=(lbs, ubs) if has_bounds else (-np.inf, np.inf),
            max_nfev=config.nlls_max_iter,
            ftol=config.nlls_ftol,
            xtol=config.nlls_xtol,
            gtol=1e-12,
            verbose=0,
        )
        success = bool(result.success)
        message = str(result.message)
        popt = result.x
        nfev_approx = int(result.nfev)
    except Exception as e:
        logger.error(f"LMFIT 异常: {e}")
        return _make_failed_result(
            method=InversionMethod.LMFIT,
            message=f"least_squares exception: {e}",
            varying=varying_params,
            fixed_params=fixed_params,
            n_data=n_data,
        )

    # 协方差估计（基于最小二乘结果的雅可比）
    x_opt = popt
    try:
        eps = 1e-5
        J = np.zeros((n_data, n_params), dtype=np.float64)
        for j in range(n_params):
            step = max(abs(x_opt[j]) * eps, eps * 1e-2)
            r_plus = _residual_fun(x_opt + np.eye(n_params)[j] * step)
            r_minus = _residual_fun(x_opt - np.eye(n_params)[j] * step)
            J[:, j] = (r_plus - r_minus) / (2 * step)

        JtJ = J.T @ J
        s2 = (2.0 * result.cost) / dof if dof > 0 else 1.0
        cov = s2 * np.linalg.pinv(JtJ)
        stds = np.sqrt(np.maximum(np.diag(cov), 0.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = cov / np.outer(stds, stds)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.warning(f"协方差估计失败: {e}")
        cov = np.zeros((n_params, n_params))
        stds = np.zeros(n_params)
        corr = np.eye(n_params)

    preds = model_prediction(
        popt, param_names, fixed_params,
        focuses, doses, target_cds, pitches, pattern_types,
        config.forward_model_complexity,
    )
    raw_residuals = measured - preds

    calibrated = {name: float(val) for name, val in zip(param_names, popt)}
    for name, val in fixed_params.items():
        calibrated[name] = float(val)
    uncert_dict = {name: float(std) for name, std in zip(param_names, stds)}
    for name in fixed_params:
        uncert_dict[name] = 0.0

    cost = float(0.5 * np.sum(raw_residuals ** 2))
    meas_sigma = np.maximum(uncertainties, 1e-3)
    chi2 = float(np.sum((raw_residuals / meas_sigma) ** 2)) \
        if config.use_measurement_weights else float(np.sum(raw_residuals ** 2))
    reduced_chi2 = chi2 / dof if dof > 0 else chi2

    dt = time.time() - t0
    logger.info(
        f"LMFIT 完成：成功={success}, 耗时={dt:.2f}s, χ²/dof={reduced_chi2:.3f}"
    )

    return InversionResult(
        method=InversionMethod.LMFIT,
        success=success,
        message=message,
        calibrated_values=calibrated,
        uncertainties=uncert_dict,
        covariance_matrix=cov,
        correlation_matrix=corr,
        varying_names=param_names,
        cost=cost,
        chi2=chi2,
        reduced_chi2=reduced_chi2,
        n_data=n_data,
        n_params=n_params,
        dof=dof,
        residuals=raw_residuals,
        predicted_cds=preds,
        iterations=nfev_approx,
    )


# ---------------------------------------------------------------------------
# 贝叶斯 MCMC（Metropolis-Hastings 随机游走）
# ---------------------------------------------------------------------------

def _metropolis_hastings(log_post,
                         x0: np.ndarray,
                         lbs: np.ndarray,
                         ubs: np.ndarray,
                         n_steps: int,
                         n_walkers: int,
                         n_burnin: int,
                         proposal_std: Optional[np.ndarray] = None,
                         random_seed: Optional[int] = None,
                         ) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    多链 Metropolis-Hastings 采样（简化版，无需额外依赖）。

    Returns:
        samples: (n_steps - n_burnin) * n_walkers × n_params  的后验样本
        acceptance: 平均接受率
        chain_history: (n_walkers, n_steps, n_params) 完整链
    """
    rng = np.random.default_rng(random_seed)
    n_params = len(x0)

    if proposal_std is None:
        # 用一个合理的初始提议尺度（相对于边界宽度的几百分之一）
        span = ubs - lbs
        span = np.where(np.isfinite(span), span, np.abs(x0) * 0.05 + 1e-2)
        proposal_std = 0.01 * span

    # 初始化 walker 位置（在 x0 附近随机扰动）
    chains = np.zeros((n_walkers, n_steps, n_params), dtype=np.float64)
    for w in range(n_walkers):
        while True:
            x = x0 + rng.normal(0, proposal_std * 2, n_params)
            if np.all(x >= lbs) and np.all(x <= ubs):
                chains[w, 0] = x
                break

    log_p = np.array([log_post(chains[w, 0]) for w in range(n_walkers)])
    total_proposed = 0
    total_accepted = 0

    for step in range(1, n_steps):
        for w in range(n_walkers):
            x_cur = chains[w, step - 1]
            lp_cur = log_p[w]

            x_prop = x_cur + rng.normal(0, proposal_std)
            lp_prop = log_post(x_prop)

            if not np.isfinite(lp_prop):
                alpha = 0.0
            else:
                delta = lp_prop - lp_cur
                if delta >= 0:
                    alpha = 1.0
                elif delta < -500:
                    alpha = 0.0
                else:
                    alpha = float(np.exp(delta))
            total_proposed += 1
            if rng.random() < alpha:
                chains[w, step] = x_prop
                log_p[w] = lp_prop
                total_accepted += 1
            else:
                chains[w, step] = x_cur

    # 聚合后验样本（丢弃 burn-in）
    samples = chains[:, n_burnin:, :].reshape(-1, n_params)
    acceptance = total_accepted / max(total_proposed, 1)

    return samples, acceptance, chains


def bayesian_inversion(config: CalibrationConfig,
                       dataset: CDSEMDataset,
                       initial_params: Optional[Dict[str, float]] = None,
                       ) -> InversionResult:
    """
    贝叶斯 MCMC 参数反演。

    使用多链 Metropolis-Hastings 随机游走（避免引入额外依赖）。
    后验众数附近可用 NLLS 提供的 MAP 估计作为初值。
    """
    t0 = time.time()

    (focuses, doses, measured, target_cds, pitches,
     pattern_types, uncertainties) = _prepare_arrays(dataset)

    varying_params = config.parameters.get_varying_parameters()
    fixed_params = {p.name: p.initial_value
                    for p in config.parameters.get_fixed_parameters()}
    param_names = [p.name for p in varying_params]
    lbs, ubs = config.parameters.bounds()

    # 初始点：优先用传入，其次用参数集 initial_value
    if initial_params is not None:
        x0 = np.array([initial_params.get(p.name, p.initial_value)
                       for p in varying_params], dtype=np.float64)
    else:
        x0 = np.array([p.initial_value for p in varying_params], dtype=np.float64)

    # 边界裁剪
    x0 = np.clip(x0, lbs, ubs)

    n_data = len(measured)
    n_params = len(param_names)
    dof = max(n_data - n_params, 1)

    weights = (1.0 / np.maximum(uncertainties, 1e-3)
               if config.use_measurement_weights
               else np.ones_like(measured))

    def _log_post(x):
        return _log_posterior(
            x, param_names, fixed_params,
            focuses, doses, measured, target_cds, pitches, pattern_types,
            weights, config.forward_model_complexity,
            lbs, ubs, varying_params,
        )

    logger.info(
        f"贝叶斯 MCMC 开始：{config.mcmc_n_walkers} walkers × "
        f"{config.mcmc_n_steps} steps（burn-in {config.mcmc_n_burnin}）"
    )

    try:
        samples, acceptance, chains = _metropolis_hastings(
            _log_post,
            x0=x0,
            lbs=lbs,
            ubs=ubs,
            n_steps=config.mcmc_n_steps,
            n_walkers=config.mcmc_n_walkers,
            n_burnin=config.mcmc_n_burnin,
            random_seed=config.random_seed,
        )
    except Exception as e:
        logger.error(f"MCMC 异常: {e}")
        return _make_failed_result(
            method=InversionMethod.BAYESIAN_MCMC,
            message=f"MCMC exception: {e}",
            varying=varying_params,
            fixed_params=fixed_params,
            n_data=n_data,
        )

    if len(samples) == 0:
        return _make_failed_result(
            method=InversionMethod.BAYESIAN_MCMC,
            message="MCMC 无有效样本（burn-in 过长？）",
            varying=varying_params,
            fixed_params=fixed_params,
            n_data=n_data,
        )

    # 汇总后验：MAP 估计 vs 均值
    lp_samples = np.array([_log_post(s) for s in samples])
    map_idx = int(np.argmax(lp_samples))
    x_map = samples[map_idx]
    x_mean = np.mean(samples, axis=0)
    x_std = np.std(samples, axis=0)

    # 报告以 MAP 为估计值，以 ±1σ 为不确定度
    x_opt = x_map
    stds = x_std

    # 预测
    preds = model_prediction(
        x_opt, param_names, fixed_params,
        focuses, doses, target_cds, pitches, pattern_types,
        config.forward_model_complexity,
    )
    raw_residuals = measured - preds

    # 协方差（样本协方差）
    cov = np.cov(samples, rowvar=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / np.outer(stds, stds)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    calibrated = {name: float(val) for name, val in zip(param_names, x_opt)}
    for name, val in fixed_params.items():
        calibrated[name] = float(val)
    uncert_dict = {name: float(std) for name, std in zip(param_names, stds)}
    for name in fixed_params:
        uncert_dict[name] = 0.0

    cost = float(0.5 * np.sum(raw_residuals ** 2))
    meas_sigma = np.maximum(uncertainties, 1e-3)
    chi2 = float(np.sum((raw_residuals / meas_sigma) ** 2)) \
        if config.use_measurement_weights else float(np.sum(raw_residuals ** 2))
    reduced_chi2 = chi2 / dof if dof > 0 else chi2

    dt = time.time() - t0
    logger.info(
        f"贝叶斯 MCMC 完成：接受率 {acceptance*100:.1f}%, "
        f"样本数 {len(samples)}, 耗时 {dt:.2f}s"
    )

    # chains 形状 (n_walkers, n_steps, n_params) → 保存
    return InversionResult(
        method=InversionMethod.BAYESIAN_MCMC,
        success=True,
        message=f"MCMC done, acceptance={acceptance:.3f}, MAP logPost={lp_samples[map_idx]:.2f}",
        calibrated_values=calibrated,
        uncertainties=uncert_dict,
        covariance_matrix=cov,
        correlation_matrix=corr,
        varying_names=param_names,
        cost=cost,
        chi2=chi2,
        reduced_chi2=reduced_chi2,
        n_data=n_data,
        n_params=n_params,
        dof=dof,
        residuals=raw_residuals,
        predicted_cds=preds,
        iterations=config.mcmc_n_walkers * config.mcmc_n_steps,
        mcmc_samples=samples,
        mcmc_acceptance=acceptance,
    )


# ---------------------------------------------------------------------------
# 统一入口：run_inversion
# ---------------------------------------------------------------------------

def run_inversion(config: CalibrationConfig,
                  dataset: CDSEMDataset,
                  ) -> InversionResult:
    """
    根据配置选择反演方法并执行。

    - NLLS       → nlls_inversion
    - LMFIT      → lmfit_inversion
    - BAYESIAN   → bayesian_inversion（用参数初值）
    - BOTH       → 先 LMFIT，得到 MAP 后用其初值做 MCMC
    """
    method = config.method

    if method == InversionMethod.NLLS:
        return nlls_inversion(config, dataset)
    elif method == InversionMethod.LMFIT:
        return lmfit_inversion(config, dataset)
    elif method == InversionMethod.BAYESIAN_MCMC:
        return bayesian_inversion(config, dataset)
    elif method == InversionMethod.BOTH:
        # 先 LMFIT 找初值
        lm_result = lmfit_inversion(config, dataset)
        # 用其解作为 MCMC 初值
        return bayesian_inversion(config, dataset,
                                  initial_params=lm_result.calibrated_values)
    else:
        raise ValueError(f"未知反演方法: {method}")


class InversionEngine:
    """
    面向对象的反演引擎。

    典型用法::

        engine = InversionEngine(config)
        result = engine.run(dataset)
        print(result.summary_table())
    """

    def __init__(self, config: CalibrationConfig):
        self.config = config

    def run(self, dataset: CDSEMDataset) -> InversionResult:
        return run_inversion(self.config, dataset)


# ---------------------------------------------------------------------------
# 内部工具：失败结果构造
# ---------------------------------------------------------------------------

def _make_failed_result(method: InversionMethod,
                        message: str,
                        varying: List[CalibrationParameter],
                        fixed_params: Dict[str, float],
                        n_data: int,
                        ) -> InversionResult:
    param_names = [p.name for p in varying]
    calibrated = {p.name: p.initial_value for p in varying}
    calibrated.update(fixed_params)
    uncert = {p.name: float('nan') for p in varying}
    for n in fixed_params:
        uncert[n] = 0.0
    n_params = len(param_names)
    return InversionResult(
        method=method,
        success=False,
        message=message,
        calibrated_values=calibrated,
        uncertainties=uncert,
        covariance_matrix=np.full((n_params, n_params), np.nan),
        correlation_matrix=np.full((n_params, n_params), np.nan),
        varying_names=param_names,
        cost=np.nan,
        chi2=np.nan,
        reduced_chi2=np.nan,
        n_data=n_data,
        n_params=n_params,
        dof=max(n_data - n_params, 1),
        residuals=None,
        predicted_cds=None,
        iterations=0,
    )
