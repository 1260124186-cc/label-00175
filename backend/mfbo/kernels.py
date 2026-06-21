# -*- coding: utf-8 -*-
"""
多保真度核函数 (Multi-Fidelity Kernels)

实现文献中经典的多保真度高斯过程核函数：

1. AR1 (Auto-Regressive 1) 核 [Kennedy & O'Hagan, 2000]
   - 最经典的递归自回归多保真度模型
   - 每层保真度 = scale * 上一层 + delta

2. Co-Kriging 核
   - 通过跨保真度协方差矩阵建模相关性
   - 支持任意数量的保真度层级

3. Linear Model of Coregionalization (LCM)
   - 多输出GP的线性共区域化模型
   - 可学习各保真度间的相关矩阵

4. NARGP (Nonlinear Auto-Regressive GP) 的简化核实现
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from scipy.optimize import minimize
import logging

from mfbo.schemas import FidelityLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 基础单保真度核函数
# ---------------------------------------------------------------------------

def rbf_kernel(x1: np.ndarray, x2: np.ndarray, lengthscales: np.ndarray,
               variance: float = 1.0) -> np.ndarray:
    """
    RBF (Radial Basis Function) / Squared Exponential 核

    k(x1, x2) = variance * exp(-0.5 * sum_d ((x1_d - x2_d) / l_d)^2)

    Args:
        x1: (N1, D)
        x2: (N2, D)
        lengthscales: (D,) 各维度长度尺度
        variance: 信号方差

    Returns:
        (N1, N2) 核矩阵
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    l = np.asarray(lengthscales, dtype=np.float64).reshape(1, -1)

    diff = (x1[:, None, :] - x2[None, :, :]) / l
    sq_dist = np.sum(diff ** 2, axis=-1)
    return variance * np.exp(-0.5 * sq_dist)


def matern52_kernel(x1: np.ndarray, x2: np.ndarray, lengthscales: np.ndarray,
                    variance: float = 1.0) -> np.ndarray:
    """
    Matérn 5/2 核（更光滑的平稳核）

    k(r) = variance * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    l = np.asarray(lengthscales, dtype=np.float64).reshape(1, -1)

    diff = (x1[:, None, :] - x2[None, :, :]) / l
    sq_dist = np.sum(diff ** 2, axis=-1)
    dist = np.sqrt(sq_dist + 1e-15)
    sqrt5 = np.sqrt(5.0)

    return variance * (1.0 + sqrt5 * dist + 5.0 / 3.0 * sq_dist) * np.exp(-sqrt5 * dist)


def matern32_kernel(x1: np.ndarray, x2: np.ndarray, lengthscales: np.ndarray,
                    variance: float = 1.0) -> np.ndarray:
    """Matérn 3/2 核（一次可微）"""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    l = np.asarray(lengthscales, dtype=np.float64).reshape(1, -1)

    diff = (x1[:, None, :] - x2[None, :, :]) / l
    sq_dist = np.sum(diff ** 2, axis=-1)
    dist = np.sqrt(sq_dist + 1e-15)
    sqrt3 = np.sqrt(3.0)

    return variance * (1.0 + sqrt3 * dist) * np.exp(-sqrt3 * dist)


# ---------------------------------------------------------------------------
# 核超参数容器
# ---------------------------------------------------------------------------

@dataclass
class KernelHyperparameters:
    """
    多保真度核的超参数容器

    Attributes:
        base_kernel: 基础核类型 ('rbf', 'matern32', 'matern52')
        lengthscales: 各维度长度尺度 (D,)
        variances: 各保真度的信号方差 {level: float}
        noise_variance: 噪声方差
        rho: AR1的相关系数 ρ（相邻保真度间）{level: float}
        W: LCM的核心区域矩阵 (n_levels, rank)
        kappa: LCM的附加对角方差 (n_levels,)
    """
    base_kernel: str = 'rbf'
    lengthscales: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    variances: Dict[int, float] = field(default_factory=lambda: {0: 1.0, 1: 1.0, 2: 1.0})
    noise_variance: float = 1e-6
    rho: Dict[int, float] = field(default_factory=lambda: {1: 0.5, 2: 0.5})
    W: np.ndarray = field(default_factory=lambda: np.eye(3))
    kappa: np.ndarray = field(default_factory=lambda: np.ones(3))

    def get_raw_vector(self) -> np.ndarray:
        """将超参数展平为优化向量（对数空间）"""
        params = []
        params.extend(np.log(self.lengthscales + 1e-15))
        for i in sorted(self.variances.keys()):
            params.append(np.log(max(self.variances[i], 1e-15)))
        params.append(np.log(max(self.noise_variance, 1e-15)))
        for i in sorted(self.rho.keys()):
            rho_transformed = np.log((1.0 + self.rho[i]) / (1.0 - self.rho[i]) + 1e-15)
            params.append(rho_transformed)
        return np.array(params)

    def set_from_raw_vector(self, raw: np.ndarray, n_dims: int, n_levels: int):
        """从优化向量恢复超参数"""
        idx = 0
        self.lengthscales = np.exp(raw[idx:idx + n_dims])
        idx += n_dims
        for i in range(n_levels):
            self.variances[i] = np.exp(raw[idx])
            idx += 1
        self.noise_variance = np.exp(raw[idx])
        idx += 1
        for i in range(1, n_levels):
            if idx < len(raw):
                t = raw[idx]
                self.rho[i] = 2.0 / (1.0 + np.exp(-t)) - 1.0
                idx += 1


# ---------------------------------------------------------------------------
# AR1 (Auto-Regressive 1) 多保真度核
# ---------------------------------------------------------------------------

class AR1Kernel:
    """
    Kennedy & O'Hagan (2000) 的 AR1 多保真度核

    递归定义：
        f_0(x) = z_0(x)                  ~ GP(0, σ_0² k_0(x,x'))
        f_t(x) = ρ_t * f_{t-1}(x) + z_t(x),  z_t ~ GP(0, σ_t² k_t(x,x'))

    其中 k_t 使用相同的基础核（共享长度尺度）

    优势：
    - 参数数量与保真度数线性增长
    - 物理直觉清晰（上一层保真度缩放 + 增量修正）
    """

    def __init__(self, base_kernel: str = 'rbf', n_levels: int = 3):
        self.base_kernel = base_kernel
        self.n_levels = n_levels
        self._base_kernels = {
            'rbf': rbf_kernel,
            'matern32': matern32_kernel,
            'matern52': matern52_kernel,
        }
        if base_kernel not in self._base_kernels:
            raise ValueError(f"Unknown base kernel: {base_kernel}")

    def _k_base(self, x1: np.ndarray, x2: np.ndarray, lengthscales: np.ndarray,
                variance: float) -> np.ndarray:
        return self._base_kernels[self.base_kernel](x1, x2, lengthscales, variance)

    def build_covariance_matrix(self, X: np.ndarray, levels: np.ndarray,
                                hp: KernelHyperparameters) -> np.ndarray:
        """
        构建完整的 N×N 协方差矩阵（含噪声）

        Args:
            X: (N, D) 所有观测的输入点
            levels: (N,) 各观测对应的保真度等级（int: 0,1,2...）
            hp: 超参数

        Returns:
            K_total: (N, N) 协方差矩阵
        """
        N = len(X)
        K = np.zeros((N, N))
        noise = hp.noise_variance

        for i in range(N):
            for j in range(N):
                li, lj = int(levels[i]), int(levels[j])
                xi, xj = X[i:i+1], X[j:j+1]

                if li == lj:
                    K_ij = self._compute_same_level_cov(xi, xj, li, hp)
                else:
                    K_ij = self._compute_cross_level_cov(xi, xj, li, lj, hp)

                K[i, j] = K_ij
                if i == j:
                    K[i, j] += noise

        return K

    def _compute_same_level_cov(self, x1: np.ndarray, x2: np.ndarray,
                                level: int, hp: KernelHyperparameters) -> float:
        """计算同一保真度下两点的协方差"""
        total = 0.0
        for t in range(level + 1):
            rho_prod = 1.0
            for s in range(t + 1, level + 1):
                rho_prod *= hp.rho.get(s, 0.5)
            var_t = hp.variances.get(t, 1.0)
            k_base = self._k_base(x1, x2, hp.lengthscales, 1.0)[0, 0]
            total += (rho_prod ** 2) * var_t * k_base
        return total

    def _compute_cross_level_cov(self, x1: np.ndarray, x2: np.ndarray,
                                 l1: int, l2: int,
                                 hp: KernelHyperparameters) -> float:
        """计算不同保真度下两点的协方差"""
        l_min = min(l1, l2)
        l_max = max(l1, l2)

        total = 0.0
        for t in range(l_min + 1):
            rho_prod_l1 = 1.0
            rho_prod_l2 = 1.0
            for s in range(t + 1, l1 + 1):
                rho_prod_l1 *= hp.rho.get(s, 0.5)
            for s in range(t + 1, l2 + 1):
                rho_prod_l2 *= hp.rho.get(s, 0.5)

            var_t = hp.variances.get(t, 1.0)
            k_base = self._k_base(x1, x2, hp.lengthscales, 1.0)[0, 0]
            total += rho_prod_l1 * rho_prod_l2 * var_t * k_base
        return total

    def build_predictive_covariance(self, X_train: np.ndarray, X_test: np.ndarray,
                                    levels_train: np.ndarray,
                                    target_level: int,
                                    hp: KernelHyperparameters) -> np.ndarray:
        """
        构建训练集与测试点（目标保真度）之间的协方差矩阵

        Args:
            X_train: (N_train, D)
            X_test: (N_test, D)
            levels_train: (N_train,)
            target_level: 目标保真度

        Returns:
            K_trans: (N_train, N_test) 跨协方差矩阵
        """
        N_train = len(X_train)
        N_test = len(X_test)
        K_trans = np.zeros((N_train, N_test))

        for i in range(N_train):
            li = int(levels_train[i])
            for j in range(N_test):
                xi, xj = X_train[i:i+1], X_test[j:j+1]
                K_trans[i, j] = self._compute_cross_level_cov(
                    xi, xj, li, target_level, hp
                )
        return K_trans

    def build_test_covariance(self, X_test: np.ndarray,
                              target_level: int,
                              hp: KernelHyperparameters) -> np.ndarray:
        """构建测试集在目标保真度下的自协方差矩阵"""
        n = len(X_test)
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                xi, xj = X_test[i:i+1], X_test[j:j+1]
                K[i, j] = self._compute_same_level_cov(xi, xj, target_level, hp)
        return K

    def negative_log_likelihood(self, hp_raw: np.ndarray, X: np.ndarray,
                                levels: np.ndarray, y: np.ndarray,
                                n_dims: int) -> float:
        """
        计算负对数边际似然（用于超参数优化）

        Args:
            hp_raw: 展平的超参数向量（对数空间）
            X: (N, D)
            levels: (N,)
            y: (N,)
            n_dims: D

        Returns:
            nll: 负对数似然值
        """
        hp = KernelHyperparameters(base_kernel=self.base_kernel)
        hp.set_from_raw_vector(hp_raw, n_dims, self.n_levels)

        N = len(X)
        try:
            K = self.build_covariance_matrix(X, levels, hp)
            jitter = 1e-8 * np.eye(N)
            K_jitter = K + jitter

            L, low = cho_factor(K_jitter, lower=True)
            alpha = cho_solve((L, low), y)

            nll = 0.5 * np.dot(y, alpha)
            nll += np.sum(np.log(np.diag(L)))
            nll += 0.5 * N * np.log(2 * np.pi)
            return float(nll)
        except np.linalg.LinAlgError:
            return 1e10


# ---------------------------------------------------------------------------
# Co-Kriging 核
# ---------------------------------------------------------------------------

class CoKrigingKernel:
    """
    Co-Kriging 多保真度核（直接建模跨保真度协方差）

    对于保真度 l, m：
        k_{l,m}(x, x') = σ_l σ_m * ρ_{l,m} * k_base(x, x')

    其中 ρ_{l,m} 是保真度间的相关系数，学习得到
    """

    def __init__(self, base_kernel: str = 'rbf', n_levels: int = 3):
        self.base_kernel = base_kernel
        self.n_levels = n_levels
        self._base_kernels = {
            'rbf': rbf_kernel,
            'matern32': matern32_kernel,
            'matern52': matern52_kernel,
        }

    def _k_base(self, x1, x2, lengthscales, variance=1.0):
        return self._base_kernels[self.base_kernel](x1, x2, lengthscales, variance)

    def _build_correlation_matrix(self, hp: KernelHyperparameters) -> np.ndarray:
        """构建保真度间的相关矩阵 B (n_levels × n_levels)"""
        L = len(hp.variances)
        B = np.eye(L)
        for i in range(L):
            for j in range(L):
                if i != j:
                    key = max(i, j)
                    rho_ij = hp.rho.get(key, 0.5)
                    if i < j:
                        rho_ij = hp.rho.get(j, 0.5) ** (j - i)
                    else:
                        rho_ij = hp.rho.get(i, 0.5) ** (i - j)
                    B[i, j] = rho_ij
        return B

    def build_covariance_matrix(self, X: np.ndarray, levels: np.ndarray,
                                hp: KernelHyperparameters) -> np.ndarray:
        N = len(X)
        K = np.zeros((N, N))
        noise = hp.noise_variance
        B = self._build_correlation_matrix(hp)

        k_base_all = self._k_base(X, X, hp.lengthscales, 1.0)

        for i in range(N):
            for j in range(N):
                li, lj = int(levels[i]), int(levels[j])
                sigma_i = np.sqrt(hp.variances.get(li, 1.0))
                sigma_j = np.sqrt(hp.variances.get(lj, 1.0))
                K[i, j] = sigma_i * sigma_j * B[li, lj] * k_base_all[i, j]
                if i == j:
                    K[i, j] += noise
        return K

    def build_predictive_covariance(self, X_train: np.ndarray, X_test: np.ndarray,
                                    levels_train: np.ndarray,
                                    target_level: int,
                                    hp: KernelHyperparameters) -> np.ndarray:
        N_train = len(X_train)
        N_test = len(X_test)
        B = self._build_correlation_matrix(hp)
        k_trans = self._k_base(X_train, X_test, hp.lengthscales, 1.0)

        K_trans = np.zeros((N_train, N_test))
        sigma_target = np.sqrt(hp.variances.get(target_level, 1.0))
        for i in range(N_train):
            li = int(levels_train[i])
            sigma_i = np.sqrt(hp.variances.get(li, 1.0))
            K_trans[i, :] = sigma_i * sigma_target * B[li, target_level] * k_trans[i, :]
        return K_trans

    def build_test_covariance(self, X_test: np.ndarray, target_level: int,
                              hp: KernelHyperparameters) -> np.ndarray:
        var_target = hp.variances.get(target_level, 1.0)
        return self._k_base(X_test, X_test, hp.lengthscales, var_target)

    def negative_log_likelihood(self, hp_raw: np.ndarray, X: np.ndarray,
                                levels: np.ndarray, y: np.ndarray,
                                n_dims: int) -> float:
        hp = KernelHyperparameters(base_kernel=self.base_kernel)
        hp.set_from_raw_vector(hp_raw, n_dims, self.n_levels)

        N = len(X)
        try:
            K = self.build_covariance_matrix(X, levels, hp)
            jitter = 1e-8 * np.eye(N)
            L, low = cho_factor(K + jitter, lower=True)
            alpha = cho_solve((L, low), y)
            nll = 0.5 * np.dot(y, alpha)
            nll += np.sum(np.log(np.diag(L)))
            nll += 0.5 * N * np.log(2 * np.pi)
            return float(nll)
        except np.linalg.LinAlgError:
            return 1e10


# ---------------------------------------------------------------------------
# Linear Model of Coregionalization (LCM) 核
# ---------------------------------------------------------------------------

class LCMKernel:
    """
    Linear Model of Coregionalization (LCM) 多保真度核

    核分解：
        k({x, l}, {x', l'}) = sum_r W_{l,r} * W_{l',r} * k_r(x, x')
                              + delta(l=l') * kappa_l

    其中 W 是核心区域矩阵 (n_levels × R)，R 是秩
    """

    def __init__(self, base_kernel: str = 'rbf', n_levels: int = 3, rank: int = 2):
        self.base_kernel = base_kernel
        self.n_levels = n_levels
        self.rank = rank
        self._base_kernels = {
            'rbf': rbf_kernel,
            'matern32': matern32_kernel,
            'matern52': matern52_kernel,
        }

    def _k_base(self, x1, x2, lengthscales, variance=1.0):
        return self._base_kernels[self.base_kernel](x1, x2, lengthscales, variance)

    def _rebuild_W_and_kappa(self, hp: KernelHyperparameters) -> Tuple[np.ndarray, np.ndarray]:
        """从超参数恢复 W 和 kappa"""
        L = self.n_levels
        R = self.rank
        W = hp.W[:L, :R]
        kappa = hp.kappa[:L]
        return W, kappa

    def build_covariance_matrix(self, X: np.ndarray, levels: np.ndarray,
                                hp: KernelHyperparameters) -> np.ndarray:
        N = len(X)
        K = np.zeros((N, N))
        noise = hp.noise_variance

        W, kappa = self._rebuild_W_and_kappa(hp)
        k_base_all = self._k_base(X, X, hp.lengthscales, 1.0)

        for r in range(self.rank):
            w_r = W[:, r]
            w_outer = np.outer(w_r, w_r)

            w_ij = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    w_ij[i, j] = w_outer[int(levels[i]), int(levels[j])]

            var_r = hp.variances.get(r, 1.0)
            K += var_r * w_ij * k_base_all

        for i in range(N):
            li = int(levels[i])
            K[i, i] += kappa[li]
            K[i, i] += noise
        return K

    def build_predictive_covariance(self, X_train: np.ndarray, X_test: np.ndarray,
                                    levels_train: np.ndarray,
                                    target_level: int,
                                    hp: KernelHyperparameters) -> np.ndarray:
        N_train = len(X_train)
        N_test = len(X_test)
        W, _ = self._rebuild_W_and_kappa(hp)
        k_trans = self._k_base(X_train, X_test, hp.lengthscales, 1.0)

        K_trans = np.zeros((N_train, N_test))
        w_target = W[target_level, :]
        for r in range(self.rank):
            var_r = hp.variances.get(r, 1.0)
            w_train_r = W[levels_train.astype(int), r]
            K_trans += var_r * np.outer(w_train_r, w_target[r]) * k_trans

        return K_trans

    def build_test_covariance(self, X_test: np.ndarray, target_level: int,
                              hp: KernelHyperparameters) -> np.ndarray:
        W, kappa = self._rebuild_W_and_kappa(hp)
        w_target = W[target_level, :]
        total_var = 0.0
        for r in range(self.rank):
            var_r = hp.variances.get(r, 1.0)
            total_var += var_r * (w_target[r] ** 2)
        total_var += kappa[target_level]
        return self._k_base(X_test, X_test, hp.lengthscales, total_var)

    def negative_log_likelihood(self, hp_raw: np.ndarray, X: np.ndarray,
                                levels: np.ndarray, y: np.ndarray,
                                n_dims: int) -> float:
        hp = KernelHyperparameters(base_kernel=self.base_kernel)
        hp.set_from_raw_vector(hp_raw, n_dims, self.n_levels)

        N = len(X)
        try:
            K = self.build_covariance_matrix(X, levels, hp)
            jitter = 1e-8 * np.eye(N)
            L, low = cho_factor(K + jitter, lower=True)
            alpha = cho_solve((L, low), y)
            nll = 0.5 * np.dot(y, alpha)
            nll += np.sum(np.log(np.diag(L)))
            nll += 0.5 * N * np.log(2 * np.pi)
            return float(nll)
        except np.linalg.LinAlgError:
            return 1e10


# ---------------------------------------------------------------------------
# 超参数优化
# ---------------------------------------------------------------------------

def optimize_hyperparameters(kernel, X: np.ndarray, levels: np.ndarray,
                             y: np.ndarray, n_dims: int,
                             n_restarts: int = 5,
                             rng: Optional[np.random.Generator] = None
                             ) -> KernelHyperparameters:
    """
    通过多重启L-BFGS-B优化核超参数

    Args:
        kernel: AR1Kernel / CoKrigingKernel / LCMKernel 实例
        X: (N, D) 训练输入
        levels: (N,) 保真度等级
        y: (N,) 训练目标（已标准化）
        n_dims: D
        n_restarts: 重启次数
        rng: 随机数生成器

    Returns:
        best_hp: 最优超参数
    """
    if rng is None:
        rng = np.random.default_rng()

    L = kernel.n_levels

    # 计算参数总数
    n_params = n_dims + L + 1 + (L - 1)  # lengthscales + variances + noise + rhos

    best_nll = np.inf
    best_hp_raw = None

    bounds = []
    for _ in range(n_dims):
        bounds.append((np.log(1e-3), np.log(1e3)))
    for _ in range(L):
        bounds.append((np.log(1e-4), np.log(1e2)))
    bounds.append((np.log(1e-8), np.log(1e-1)))
    for _ in range(L - 1):
        bounds.append((-5.0, 5.0))
    bounds = np.array(bounds)

    for restart in range(n_restarts):
        if restart == 0:
            x0 = np.zeros(n_params)
            x0[:n_dims] = np.log(1.0)
            x0[n_dims:n_dims + L] = np.log(1.0)
            x0[n_dims + L] = np.log(1e-5)
            x0[n_dims + L + 1:] = 0.0
        else:
            x0 = rng.uniform(bounds[:, 0], bounds[:, 1])

        try:
            result = minimize(
                fun=lambda p: kernel.negative_log_likelihood(p, X, levels, y, n_dims),
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200}
            )
            if result.fun < best_nll and not np.isnan(result.fun):
                best_nll = result.fun
                best_hp_raw = result.x.copy()
        except Exception as e:
            logger.debug(f"HP optimization restart {restart} failed: {e}")
            continue

    if best_hp_raw is None:
        best_hp_raw = x0
        best_nll = kernel.negative_log_likelihood(x0, X, levels, y, n_dims)

    hp = KernelHyperparameters(base_kernel=kernel.base_kernel)
    hp.set_from_raw_vector(best_hp_raw, n_dims, L)
    hp._nll_value = best_nll
    return hp
