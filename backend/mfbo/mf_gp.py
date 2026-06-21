# -*- coding: utf-8 -*-
"""
多保真度高斯过程 (Multi-Fidelity Gaussian Process, MF-GP)

代理模型核心模块，建模不同保真度层级之间的相关性：
- 利用低保真度的廉价样本提供全局趋势
- 高保真度样本提供局部精确修正
- 通过核函数建模跨保真度相关性

支持的核类型：
1. AR1 (Kennedy & O'Hagan 2000) - 递归自回归，经典方法
2. Co-Kriging - 直接建模跨保真度协方差
3. LCM (Linear Coregionalization Model) - 线性共区域化模型
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from scipy.linalg import cho_factor, cho_solve
import logging
import time

from mfbo.schemas import (
    FidelityLevel,
    KernelType,
    MFBOConfig,
    Observation,
)
from mfbo.kernels import (
    AR1Kernel,
    CoKrigingKernel,
    LCMKernel,
    KernelHyperparameters,
    optimize_hyperparameters,
    rbf_kernel,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    GP预测结果

    Attributes:
        mean: 预测均值 (N_test,)
        std: 预测标准差 (N_test,)
        variance: 预测方差 (N_test,)
        covariance: 完整协方差矩阵 (N_test, N_test)
    """
    mean: np.ndarray
    std: np.ndarray
    variance: np.ndarray
    covariance: Optional[np.ndarray] = None


class MultiFidelityGP:
    """
    多保真度高斯过程代理模型

    使用不同保真度的观测数据联合建模 f_l(x)，支持：
    - 任意保真度层级的预测
    - 超参数自动优化（最大边际似然）
    - 增量更新（添加新样本后重新拟合）
    - 目标保真度的不确定性量化

    典型用法：
        mf_gp = MultiFidelityGP(config)
        mf_gp.fit(X, levels, y)
        pred = mf_gp.predict(X_test, target_fidelity=FidelityLevel.HIGH)
    """

    def __init__(self, config: Optional[MFBOConfig] = None):
        """
        初始化MF-GP

        Args:
            config: MFBO配置（包含核类型、超参优化次数等）
        """
        self.config = config or MFBOConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

        # 根据配置选择核
        n_levels = 3
        base_kernel_name = 'rbf'
        if self.config.kernel_type == KernelType.AR1:
            self.kernel = AR1Kernel(base_kernel=base_kernel_name, n_levels=n_levels)
        elif self.config.kernel_type == KernelType.COKriging:
            self.kernel = CoKrigingKernel(base_kernel=base_kernel_name, n_levels=n_levels)
        elif self.config.kernel_type == KernelType.LinearCoregional:
            self.kernel = LCMKernel(base_kernel=base_kernel_name, n_levels=n_levels, rank=2)
        else:
            logger.warning(f"Unknown kernel type {self.config.kernel_type}, using AR1")
            self.kernel = AR1Kernel(base_kernel=base_kernel_name, n_levels=n_levels)

        # 训练数据
        self._X_train: Optional[np.ndarray] = None
        self._levels_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_raw: Optional[np.ndarray] = None

        # 标准化参数
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

        # 预计算矩阵
        self._L: Optional[np.ndarray] = None  # Cholesky 因子
        self._alpha: Optional[np.ndarray] = None  # K^{-1} y
        self._K_inv: Optional[np.ndarray] = None  # K^{-1}（用于某些获取函数）

        # 超参数
        self.hyperparameters: Optional[KernelHyperparameters] = None
        self.last_nll: float = np.inf

        # 数据量统计
        self.n_train: int = 0
        self.n_by_fidelity: Dict[FidelityLevel, int] = {
            FidelityLevel.LOW: 0,
            FidelityLevel.MEDIUM: 0,
            FidelityLevel.HIGH: 0,
        }

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------

    def _prepare_training_data(self, observations: List[Observation]
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将Observation列表转换为训练数组

        Args:
            observations: 观测列表

        Returns:
            (X, levels_int, y)
        """
        X_list = []
        levels_list = []
        y_list = []

        for obs in observations:
            X_list.append(obs.x)
            levels_list.append(obs.fidelity.to_int())
            y_list.append(obs.y)

        X = np.array(X_list, dtype=np.float64)
        levels = np.array(levels_list, dtype=np.int32)
        y = np.array(y_list, dtype=np.float64)

        return X, levels, y

    def _standardize_y(self, y: np.ndarray) -> np.ndarray:
        """标准化目标值（零均值单位方差）"""
        if len(y) > 1:
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y))
            if self._y_std < 1e-12:
                self._y_std = 1.0
        return (y - self._y_mean) / self._y_std

    def _destandardize(self, z: np.ndarray) -> np.ndarray:
        """反标准化"""
        return z * self._y_std + self._y_mean

    def _destandardize_var(self, var_z: np.ndarray) -> np.ndarray:
        """反标准化方差"""
        return var_z * (self._y_std ** 2)

    # ------------------------------------------------------------------
    # 核心拟合接口
    # ------------------------------------------------------------------

    def fit(self, observations: List[Observation]) -> 'MultiFidelityGP':
        """
        用观测数据拟合MF-GP

        Args:
            observations: 所有观测列表

        Returns:
            self（支持链式调用）
        """
        if len(observations) == 0:
            raise ValueError("No observations provided for training")

        X, levels, y_raw = self._prepare_training_data(observations)
        self._X_train = X
        self._levels_train = levels
        self._y_raw = y_raw
        self.n_train = len(y_raw)

        # 统计各保真度样本数
        for level_int in range(3):
            count = int(np.sum(levels == level_int))
            self.n_by_fidelity[FidelityLevel.from_int(level_int)] = count

        n_dims = X.shape[1]

        # 标准化
        y_std = self._standardize_y(y_raw)
        self._y_train = y_std

        # 优化超参数
        logger.debug(f"Optimizing MF-GP hyperparameters (n={self.n_train}, "
                     f"dims={n_dims}, restarts={self.config.optimizer_restarts})")
        t0 = time.time()

        self.hyperparameters = optimize_hyperparameters(
            kernel=self.kernel,
            X=X,
            levels=levels,
            y=y_std,
            n_dims=n_dims,
            n_restarts=self.config.optimizer_restarts,
            rng=self.rng,
        )
        self.last_nll = getattr(self.hyperparameters, '_nll_value', np.inf)

        logger.debug(f"HP optimization done in {time.time() - t0:.2f}s, "
                     f"NLL={self.last_nll:.3f}")

        # 预计算Cholesky和alpha
        self._precompute_matrices()
        return self

    def _precompute_matrices(self):
        """预计算 K = LL^T 和 alpha = K^{-1} y"""
        N = self.n_train
        try:
            K = self.kernel.build_covariance_matrix(
                self._X_train, self._levels_train, self.hyperparameters
            )
            jitter = 1e-8 * np.eye(N)
            self._L, low = cho_factor(K + jitter, lower=True)
            self._alpha = cho_solve((self._L, low), self._y_train)
            self._K_inv = cho_solve((self._L, low), np.eye(N))
        except np.linalg.LinAlgError as e:
            logger.error(f"Cholesky decomposition failed: {e}")
            raise

    # ------------------------------------------------------------------
    # 预测接口
    # ------------------------------------------------------------------

    def predict(self, X_test: np.ndarray,
                target_fidelity: Union[FidelityLevel, int] = FidelityLevel.HIGH,
                return_cov: bool = False,
                return_grad: bool = False) -> PredictionResult:
        """
        在指定保真度下预测测试点

        Args:
            X_test: (N_test, D) 测试点
            target_fidelity: 目标保真度层级
            return_cov: 是否返回完整协方差矩阵
            return_grad: 是否返回预测均值的梯度（暂未实现）

        Returns:
            PredictionResult
        """
        if self._L is None or self.hyperparameters is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_test = np.atleast_2d(X_test)
        if isinstance(target_fidelity, FidelityLevel):
            target_int = target_fidelity.to_int()
        else:
            target_int = int(target_fidelity)

        N_test = len(X_test)

        # K_trans: (N_train, N_test)
        K_trans = self.kernel.build_predictive_covariance(
            self._X_train, X_test, self._levels_train,
            target_int, self.hyperparameters
        )

        # 预测均值：mu = K_trans^T @ K^{-1} @ y = K_trans^T @ alpha
        mean_std = K_trans.T @ self._alpha

        # 预测方差
        # Var = K_tt - K_trans^T @ K^{-1} @ K_trans
        # 使用 Cholesky 高效计算
        v = cho_solve((self._L, True), K_trans)  # L^{-1} K_trans
        var_std_diag = np.zeros(N_test)

        # 对角方差
        K_tt_diag = np.array([
            self.kernel.build_test_covariance(
                X_test[i:i+1], target_int, self.hyperparameters
            )[0, 0]
            for i in range(N_test)
        ])
        for i in range(N_test):
            var_std_diag[i] = K_tt_diag[i] - np.dot(K_trans[:, i], v[:, i])
        var_std_diag = np.maximum(var_std_diag, 1e-12)

        # 反标准化
        mean = self._destandardize(mean_std)
        variance = self._destandardize_var(var_std_diag)
        std = np.sqrt(variance)

        # 完整协方差（如需要）
        covariance = None
        if return_cov:
            K_tt = self.kernel.build_test_covariance(
                X_test, target_int, self.hyperparameters
            )
            cov_std = K_tt - K_trans.T @ self._K_inv @ K_trans
            cov_std = 0.5 * (cov_std + cov_std.T)  # 对称化
            # 加数值稳定项
            cov_std += 1e-10 * np.eye(N_test)
            covariance = self._destandardize_var(cov_std)

        return PredictionResult(
            mean=mean,
            std=std,
            variance=variance,
            covariance=covariance,
        )

    def predict_at_all_fidelities(self, X_test: np.ndarray
                                  ) -> Dict[FidelityLevel, PredictionResult]:
        """在所有保真度层级上预测"""
        results = {}
        for level in [FidelityLevel.LOW, FidelityLevel.MEDIUM, FidelityLevel.HIGH]:
            results[level] = self.predict(X_test, target_fidelity=level)
        return results

    # ------------------------------------------------------------------
    # 信息论辅助方法
    # ------------------------------------------------------------------

    def mutual_information_improvement(self, X_candidate: np.ndarray,
                                       target_fidelity: FidelityLevel
                                       ) -> np.ndarray:
        """
        近似计算候选点添加后关于最大值的互信息增益（MES近似）

        Args:
            X_candidate: (N_cand, D) 候选点
            target_fidelity: 评估的保真度

        Returns:
            info_gain: (N_cand,) 互信息增益近似
        """
        if self._X_train is None:
            return np.zeros(len(X_candidate))

        pred = self.predict(X_candidate, target_fidelity=target_fidelity)

        # 基于不确定性的简单近似（当方差高时信息增益大）
        n_train_total = self.n_train
        n_target = self.n_by_fidelity.get(target_fidelity, 0)

        # 低保真度折扣系数
        level_int = target_fidelity.to_int()
        rho = self.hyperparameters.rho.get(2, 0.5) ** max(0, 2 - level_int) if self.hyperparameters else 0.5

        info_gain = rho * np.log(1.0 + pred.variance / max(self._y_std ** 2, 1e-12))

        # 鼓励探索样本少的保真度
        target_bonus = np.sqrt(max(1, n_train_total) / max(1, n_target + 1)) * 0.1
        info_gain += target_bonus

        return info_gain

    # ------------------------------------------------------------------
    # 诊断与调试
    # ------------------------------------------------------------------

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取训练数据副本 (X, levels, y_raw)"""
        return (
            self._X_train.copy() if self._X_train is not None else None,
            self._levels_train.copy() if self._levels_train is not None else None,
            self._y_raw.copy() if self._y_raw is not None else None,
        )

    def log_likelihood(self) -> float:
        """返回当前训练数据的负对数边际似然"""
        return self.last_nll

    def cross_validate_loo(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        留一交叉验证（LOO-CV），使用高斯过程的高效公式

        Returns:
            (loo_errors, loo_predictions) - 各点的LOO误差和预测
        """
        if self._K_inv is None or self._y_train is None:
            raise RuntimeError("Model not fitted")

        K_inv_diag = np.diag(self._K_inv)
        K_inv_diag = np.maximum(K_inv_diag, 1e-12)

        loo_pred_std = self._y_train - self._alpha / K_inv_diag
        loo_var_std = 1.0 / K_inv_diag

        loo_pred = self._destandardize(loo_pred_std)
        loo_error = (loo_pred - self._y_raw) / np.sqrt(self._destandardize_var(loo_var_std))

        return loo_error, loo_pred

    def surrogate_fidelity_error(self) -> Dict[Tuple[FidelityLevel, FidelityLevel], float]:
        """
        估计保真度间的代理误差（RMSE）

        通过LOO预测估计不同保真度间的建模误差。

        Returns:
            dict: (from_level, to_level) -> RMSE
        """
        if self._X_train is None:
            return {}

        errors = {}
        levels_available = []
        for level in [FidelityLevel.LOW, FidelityLevel.MEDIUM, FidelityLevel.HIGH]:
            if self.n_by_fidelity.get(level, 0) > 0:
                levels_available.append(level)

        for i, l1 in enumerate(levels_available):
            for l2 in levels_available[i + 1:]:
                idx1 = np.where(self._levels_train == l1.to_int())[0]
                idx2 = np.where(self._levels_train == l2.to_int())[0]

                if len(idx1) == 0 or len(idx2) == 0:
                    continue

                # 在 l1 数据上训练，预测 l2 的点
                X_l1 = self._X_train[idx1]
                y_l1 = self._y_raw[idx1]

                # 简单最近邻误差估计
                errors[(l1, l2)] = float(self._estimate_fidelity_gap(X_l1, y_l1, idx2))

        return errors

    def _estimate_fidelity_gap(self, X_from: np.ndarray, y_from: np.ndarray,
                               idx_to: np.ndarray) -> float:
        """估计保真度差距的简单方法"""
        if len(X_from) == 0 or len(idx_to) == 0:
            return 0.0

        X_to = self._X_train[idx_to]
        y_to = self._y_raw[idx_to]

        # 对每个 to 点，找最近的 from 点
        sq_dists = np.sum((X_to[:, None, :] - X_from[None, :, :]) ** 2, axis=-1)
        nn_idx = np.argmin(sq_dists, axis=1)

        # 这里只是粗略估计，实际应用会用更复杂的方法
        # 简单用 y 的全局均值差近似
        mean_from = np.mean(y_from)
        mean_to = np.mean(y_to)
        return abs(mean_from - mean_to) + 0.1 * np.std(y_to)
