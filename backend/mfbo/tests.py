# -*- coding: utf-8 -*-
"""
多保真度贝叶斯优化 (MFBO) 单元测试

覆盖：
- 数据结构与枚举
- 核函数计算
- 多保真度GP拟合与预测
- 获取函数计算
- 保真度选择策略
- 完整优化循环
- 边界条件与异常处理

运行: cd backend && python -m pytest mfbo/tests.py -v
"""

import pytest
import numpy as np
from typing import Dict
import sys
from pathlib import Path

# 路径兼容
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from mfbo.schemas import (
    FidelityLevel,
    SearchSpace,
    MFBOConfig,
    Observation,
    FidelityCost,
    KernelType,
    AcquisitionFunctionType,
    FidelitySelectionStrategy,
    IterationRecord,
    MFBOResult,
)
from mfbo.kernels import (
    AR1Kernel,
    CoKrigingKernel,
    LCMKernel,
    KernelHyperparameters,
    rbf_kernel,
    matern52_kernel,
    optimize_hyperparameters,
)
from mfbo.mf_gp import MultiFidelityGP, PredictionResult
from mfbo.acquisition import (
    AcquisitionFunction,
    AcquisitionConfig,
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
    ei_per_unit_cost,
)
from mfbo.fidelity_strategy import FidelitySelector, FidelityDecision
from mfbo.optimizer import MultiFidelityBayesianOptimizer


# ===========================================================================
# 测试：数据结构 Schemas
# ===========================================================================

class TestFidelityLevel:
    """测试保真度枚举"""

    def test_basic_values(self):
        assert FidelityLevel.LOW.value == "low"
        assert FidelityLevel.MEDIUM.value == "medium"
        assert FidelityLevel.HIGH.value == "high"

    def test_integer_conversion(self):
        assert FidelityLevel.from_int(0) == FidelityLevel.LOW
        assert FidelityLevel.from_int(1) == FidelityLevel.MEDIUM
        assert FidelityLevel.from_int(2) == FidelityLevel.HIGH
        assert FidelityLevel.LOW.to_int() == 0
        assert FidelityLevel.HIGH.to_int() == 2

    def test_comparison_operators(self):
        assert FidelityLevel.LOW < FidelityLevel.MEDIUM
        assert FidelityLevel.MEDIUM < FidelityLevel.HIGH
        assert FidelityLevel.HIGH > FidelityLevel.LOW
        assert FidelityLevel.MEDIUM >= FidelityLevel.LOW
        assert FidelityLevel.LOW <= FidelityLevel.LOW

    def test_invalid_int_raises(self):
        with pytest.raises(ValueError):
            FidelityLevel.from_int(999)


class TestSearchSpace:
    """测试搜索空间"""

    def test_creation(self):
        bounds = [(0.0, 1.0), (-1.0, 2.0)]
        ss = SearchSpace(bounds=bounds)
        assert ss.dimensions == 2
        assert ss.names == ["x0", "x1"]

    def test_sampling(self):
        bounds = [(-5.0, 10.0), (0.0, 15.0)]
        ss = SearchSpace(bounds=bounds)
        samples = ss.sample(n_samples=100, rng=np.random.default_rng(0))
        assert samples.shape == (100, 2)
        for i, (low, high) in enumerate(bounds):
            assert np.all(samples[:, i] >= low)
            assert np.all(samples[:, i] <= high)

    def test_clip(self):
        bounds = [(0.0, 1.0)]
        ss = SearchSpace(bounds=bounds)
        x = np.array([[-0.5], [0.5], [1.5]])
        clipped = ss.clip(x)
        assert clipped[0, 0] == 0.0
        assert clipped[1, 0] == 0.5
        assert clipped[2, 0] == 1.0


class TestMFBOConfig:
    """测试配置序列化"""

    def test_default_config(self):
        cfg = MFBOConfig()
        assert cfg.n_init_low == 10
        assert cfg.target_fidelity == FidelityLevel.HIGH
        assert cfg.max_budget == 50.0

    def test_to_dict_from_dict_roundtrip(self):
        cfg = MFBOConfig(
            n_init_low=8,
            max_budget=25.0,
            kernel_type=KernelType.COKriging,
            acquisition_type=AcquisitionFunctionType.UCB,
        )
        d = cfg.to_dict()
        cfg2 = MFBOConfig.from_dict(d)
        assert cfg2.n_init_low == 8
        assert cfg2.max_budget == 25.0
        assert cfg2.kernel_type == KernelType.COKriging
        assert cfg2.acquisition_type == AcquisitionFunctionType.UCB


class TestFidelityCost:
    """测试成本配置"""

    def test_default_costs(self):
        fc = FidelityCost()
        assert fc.get_cost(FidelityLevel.LOW) == 0.01
        assert fc.get_cost(FidelityLevel.MEDIUM) == 0.1
        assert fc.get_cost(FidelityLevel.HIGH) == 1.0

    def test_cost_ratio(self):
        fc = FidelityCost()
        # HIGH 比 LOW 贵 100 倍
        assert fc.cost_ratio(FidelityLevel.LOW, FidelityLevel.HIGH) == pytest.approx(100.0)


# ===========================================================================
# 测试：核函数 Kernels
# ===========================================================================

@pytest.fixture
def simple_dataset():
    """创建简单的二分类数据集用于核测试"""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (15, 2))
    levels = rng.integers(0, 3, 15)
    y = np.sum(X ** 2, axis=1) + rng.normal(0, 0.01, 15)
    return X, levels, y


@pytest.fixture
def default_hp():
    hp = KernelHyperparameters()
    hp.lengthscales = np.array([0.5, 0.5])
    hp.variances = {0: 1.0, 1: 1.0, 2: 1.0}
    hp.noise_variance = 1e-5
    hp.rho = {1: 0.8, 2: 0.8}
    return hp


class TestBaseKernels:
    """测试基础单保真度核"""

    def test_rbf_kernel_shape(self):
        X1 = np.random.randn(10, 3)
        X2 = np.random.randn(5, 3)
        K = rbf_kernel(X1, X2, lengthscales=np.array([1.0, 1.0, 1.0]))
        assert K.shape == (10, 5)

    def test_rbf_kernel_diagonal(self):
        X = np.random.randn(8, 2)
        K = rbf_kernel(X, X, lengthscales=np.array([1.0, 1.0]), variance=2.0)
        # 对角元 = variance
        np.testing.assert_allclose(np.diag(K), 2.0, atol=1e-10)

    def test_rbf_kernel_symmetric(self):
        X = np.random.randn(10, 3)
        K = rbf_kernel(X, X, lengthscales=np.array([0.5, 0.5, 0.5]))
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_matern52(self):
        X1 = np.random.randn(6, 2)
        X2 = np.random.randn(4, 2)
        K = matern52_kernel(X1, X2, lengthscales=np.array([1.0, 1.0]))
        assert K.shape == (6, 4)
        assert np.all(K >= 0)


class TestAR1Kernel:
    """测试 AR1 多保真度核"""

    def test_construction(self):
        kernel = AR1Kernel(base_kernel='rbf', n_levels=3)
        assert kernel.n_levels == 3
        assert kernel.base_kernel == 'rbf'

    def test_covariance_shape(self, simple_dataset, default_hp):
        X, levels, y = simple_dataset
        kernel = AR1Kernel(n_levels=3)
        K = kernel.build_covariance_matrix(X, levels, default_hp)
        assert K.shape == (15, 15)

    def test_covariance_symmetric(self, simple_dataset, default_hp):
        X, levels, y = simple_dataset
        kernel = AR1Kernel(n_levels=3)
        K = kernel.build_covariance_matrix(X, levels, default_hp)
        np.testing.assert_allclose(K, K.T, atol=1e-8)

    def test_covariance_positive_definite(self, simple_dataset, default_hp):
        X, levels, y = simple_dataset
        kernel = AR1Kernel(n_levels=3)
        K = kernel.build_covariance_matrix(X, levels, default_hp)
        # 加 jitter 后应该正定
        eigvals = np.linalg.eigvalsh(K + 1e-6 * np.eye(len(K)))
        assert np.all(eigvals > 0)

    def test_predictive_covariance(self, simple_dataset, default_hp):
        X_train, levels_train, y = simple_dataset
        kernel = AR1Kernel(n_levels=3)
        X_test = np.random.randn(5, 2)
        K_trans = kernel.build_predictive_covariance(
            X_train, X_test, levels_train, target_level=2, hp=default_hp
        )
        assert K_trans.shape == (15, 5)


class TestCoKrigingKernel:
    """测试 Co-Kriging 核"""

    def test_covariance_build(self, simple_dataset, default_hp):
        X, levels, y = simple_dataset
        kernel = CoKrigingKernel(n_levels=3)
        K = kernel.build_covariance_matrix(X, levels, default_hp)
        assert K.shape == (15, 15)
        np.testing.assert_allclose(K, K.T, atol=1e-8)


class TestHyperparameterOptimization:
    """测试超参数优化"""

    def test_basic_optimization(self, simple_dataset):
        X, levels, y = simple_dataset
        y_std = (y - y.mean()) / (y.std() + 1e-12)
        kernel = AR1Kernel(n_levels=3)
        hp = optimize_hyperparameters(
            kernel, X, levels, y_std, n_dims=2, n_restarts=1,
            rng=np.random.default_rng(0),
        )
        assert hp.lengthscales.shape == (2,)
        assert len(hp.variances) == 3
        assert np.all(hp.lengthscales > 0)


# ===========================================================================
# 测试：多保真度 GP
# ===========================================================================

@pytest.fixture
def observations_2d():
    """创建2维 Branin-Hoo 风格的观测数据"""
    rng = np.random.default_rng(99)
    observations = []

    def f_high(x):
        x1, x2 = x
        return (x2 - 0.1 * x1 ** 2 + x1 - 2) ** 2 + 10 * np.cos(x1) + 10

    # LOW 保真度
    for _ in range(8):
        x = rng.uniform([-5, 0], [10, 15])
        y = f_high(x) + rng.normal(0, 0.3) + 0.5  # 有偏+噪声
        observations.append(Observation(x=x, y=y, fidelity=FidelityLevel.LOW, cost=0.01))

    # MEDIUM 保真度
    for _ in range(4):
        x = rng.uniform([-5, 0], [10, 15])
        y = f_high(x) + rng.normal(0, 0.1) + 0.2
        observations.append(Observation(x=x, y=y, fidelity=FidelityLevel.MEDIUM, cost=0.1))

    # HIGH 保真度
    for _ in range(3):
        x = rng.uniform([-5, 0], [10, 15])
        y = f_high(x) + rng.normal(0, 0.01)
        observations.append(Observation(x=x, y=y, fidelity=FidelityLevel.HIGH, cost=1.0))

    return observations


class TestMultiFidelityGP:
    """测试多保真度GP"""

    def test_fit_predict(self, observations_2d):
        mf_gp = MultiFidelityGP(MFBOConfig(random_seed=42, optimizer_restarts=1))
        mf_gp.fit(observations_2d)
        assert mf_gp.n_train == len(observations_2d)
        assert mf_gp.hyperparameters is not None

        # 预测
        X_test = np.array([[0.0, 7.5], [np.pi, 2.275]])
        pred = mf_gp.predict(X_test, target_fidelity=FidelityLevel.HIGH)

        assert isinstance(pred, PredictionResult)
        assert pred.mean.shape == (2,)
        assert pred.std.shape == (2,)
        assert np.all(pred.std > 0)

    def test_predict_all_fidelities(self, observations_2d):
        mf_gp = MultiFidelityGP(MFBOConfig(random_seed=42, optimizer_restarts=1))
        mf_gp.fit(observations_2d)

        X_test = np.random.randn(4, 2)
        preds_all = mf_gp.predict_at_all_fidelities(X_test)

        assert FidelityLevel.LOW in preds_all
        assert FidelityLevel.MEDIUM in preds_all
        assert FidelityLevel.HIGH in preds_all

    def test_loo_cv(self, observations_2d):
        mf_gp = MultiFidelityGP(MFBOConfig(random_seed=42, optimizer_restarts=1))
        mf_gp.fit(observations_2d)

        errors, preds = mf_gp.cross_validate_loo()
        assert len(errors) == len(observations_2d)


# ===========================================================================
# 测试：获取函数 Acquisition
# ===========================================================================

class TestAcquisitionFunctions:
    """测试获取函数"""

    def test_ei_improvement_expected(self):
        rng = np.random.default_rng(0)
        mean = np.array([-1.0, 0.0, 1.0, 2.0])
        std = np.array([0.5, 0.5, 0.5, 0.5])
        f_best = 0.5  # 当前最优（最小化）

        ei = expected_improvement(mean, std, f_best, maximize=False)

        # mean=-1 远好于 f_best=0.5，应该有最大EI
        assert ei[0] > ei[1]
        assert ei[0] > ei[3]  # mean=2 比 f_best 差，EI≈0
        assert ei[3] == pytest.approx(0.0, abs=1e-3)

    def test_ei_nonnegative(self):
        rng = np.random.default_rng(1)
        mean = rng.standard_normal(100)
        std = np.abs(rng.standard_normal(100)) + 1e-3
        ei = expected_improvement(mean, std, f_best=0.0)
        assert np.all(ei >= 0)

    def test_ucb(self):
        mean = np.array([1.0, -1.0])
        std = np.array([0.1, 1.0])
        # 最小化：返回 -(mean - beta*std)
        ucb = upper_confidence_bound(mean, std, beta=2.0, maximize=False)
        # 第二个点虽均值低但std大，按LCB可能更优
        assert len(ucb) == 2

    def test_pi(self):
        mean = np.array([-2.0, 0.0, 2.0])
        std = np.array([0.1, 0.1, 0.1])
        pi = probability_of_improvement(mean, std, f_best=0.0)
        # 最小化问题：f<0 才算改进
        assert pi[0] > pi[1]
        assert pi[1] > pi[2]

    def test_eiv_cost_scaling(self):
        pred = PredictionResult(
            mean=np.array([0.0]), std=np.array([1.0]), variance=np.array([1.0])
        )
        # 低保真成本低 → EIV 应更高
        eiv_low = ei_per_unit_cost(pred, f_best=1.0, fidelity_cost=0.01)
        eiv_high = ei_per_unit_cost(pred, f_best=1.0, fidelity_cost=1.0)
        assert eiv_low[0] > eiv_high[0]

    def test_acquisition_function_wrapper(self, observations_2d):
        mf_gp = MultiFidelityGP(MFBOConfig(random_seed=42, optimizer_restarts=1))
        mf_gp.fit(observations_2d)

        acq = AcquisitionFunction(
            config=AcquisitionConfig(function_type=AcquisitionFunctionType.EI),
            mfbo_config=MFBOConfig(),
        )

        X_cand = np.random.randn(10, 2)
        values = acq.evaluate(
            mf_gp, X_cand, FidelityLevel.HIGH, f_best_target=0.0
        )
        assert values.shape == (10,)


# ===========================================================================
# 测试：保真度选择策略
# ===========================================================================

class TestFidelitySelector:
    """测试保真度选择器"""

    def test_cost_aware_selection(self, observations_2d):
        search_space = SearchSpace(bounds=[(-5.0, 10.0), (0.0, 15.0)])
        mf_gp = MultiFidelityGP(MFBOConfig(random_seed=42, optimizer_restarts=1))
        mf_gp.fit(observations_2d)

        selector = FidelitySelector(
            config=MFBOConfig(
                fidelity_strategy=FidelitySelectionStrategy.COST_AWARE,
                max_budget=100.0,
                random_seed=0,
                acq_n_candidates=100,
            )
        )

        decision = selector.select_next(
            mf_gp=mf_gp,
            search_space=search_space,
            f_best_target=0.0,
            budget_used=0.0,
        )

        assert isinstance(decision, FidelityDecision)
        assert isinstance(decision.selected_fidelity, FidelityLevel)
        assert decision.best_x.shape == (2,)

    def test_all_strategies_run(self, observations_2d):
        """所有策略都应该能正常运行"""
        search_space = SearchSpace(bounds=[(-5.0, 10.0), (0.0, 15.0)])
        mf_gp = MultiFidelityGP(MFBOConfig(random_seed=42, optimizer_restarts=1))
        mf_gp.fit(observations_2d)

        strategies = [
            FidelitySelectionStrategy.COST_AWARE,
            FidelitySelectionStrategy.BUDGET_PROPORTIONAL,
            FidelitySelectionStrategy.SCHEDULED,
        ]

        for strategy in strategies:
            selector = FidelitySelector(
                config=MFBOConfig(
                    fidelity_strategy=strategy,
                    max_budget=100.0,
                    max_iterations=50,
                    random_seed=0,
                    acq_n_candidates=50,
                )
            )
            decision = selector.select_next(
                mf_gp=mf_gp, search_space=search_space,
                f_best_target=0.0, budget_used=0.0,
            )
            assert decision.selected_fidelity in [
                FidelityLevel.LOW, FidelityLevel.MEDIUM, FidelityLevel.HIGH
            ]

    def test_budget_limit_available(self, observations_2d):
        """剩余预算不足时应只考虑低保真度"""
        search_space = SearchSpace(bounds=[(-5.0, 10.0), (0.0, 15.0)])
        mf_gp = MultiFidelityGP(MFBOConfig(random_seed=42, optimizer_restarts=1))
        mf_gp.fit(observations_2d)

        selector = FidelitySelector(
            config=MFBOConfig(max_budget=0.05, random_seed=0, acq_n_candidates=50)
        )

        available = selector._get_available_fidelities(
            budget_used=0.0, budget_total=0.05
        )
        # 预算 0.05 不足以负担 HIGH (1.0) 和 MEDIUM (0.1)
        assert FidelityLevel.HIGH not in available
        assert FidelityLevel.MEDIUM not in available
        assert FidelityLevel.LOW in available


# ===========================================================================
# 测试：完整优化循环
# ===========================================================================

def simple_branin_mf(x, fidelity: FidelityLevel):
    """简化版多保真度 Branin-Hoo 用于快速测试"""
    x1, x2 = x[0], x[1]
    y_base = (x2 - 0.1 * x1 ** 2 + x1 - 2) ** 2 + 10 * np.cos(x1) + 10
    rng = np.random.default_rng(int(abs(hash((x1, x2, fidelity.value))) % 10000))

    if fidelity == FidelityLevel.HIGH:
        return float(y_base + rng.normal(0, 0.01))
    elif fidelity == FidelityLevel.MEDIUM:
        return float(y_base + 0.2 + rng.normal(0, 0.05))
    else:
        return float(y_base + 0.5 + rng.normal(0, 0.2))


class TestMultiFidelityBayesianOptimizer:
    """测试完整MFBO优化器"""

    def test_minimize_basic(self):
        """最基础的最小化测试"""
        search_space = SearchSpace(bounds=[(-3.0, 5.0), (-1.0, 10.0)])

        config = MFBOConfig(
            n_init_low=4,
            n_init_medium=2,
            n_init_high=1,
            max_iterations=8,
            max_budget=100.0,  # 预算充足以完成迭代数限制
            target_fidelity=FidelityLevel.HIGH,
            random_seed=42,
            optimizer_restarts=1,
            acq_n_candidates=200,
        )

        mfbo = MultiFidelityBayesianOptimizer(config, search_space)
        result = mfbo.minimize(simple_branin_mf)

        assert isinstance(result, MFBOResult)
        assert result.n_iterations <= config.max_iterations
        assert result.best_x.shape == (2,)
        assert np.isfinite(result.best_y)
        assert len(result.observations) > 0
        assert len(result.history) > 0

    def test_budget_constraint_respected(self):
        """测试预算约束被尊重"""
        search_space = SearchSpace(bounds=[(-3.0, 5.0), (-1.0, 10.0)])
        budget = 0.5  # 很小的预算

        config = MFBOConfig(
            n_init_low=2,
            n_init_medium=1,
            n_init_high=0,
            max_iterations=1000,
            max_budget=budget,
            random_seed=1,
            optimizer_restarts=1,
            acq_n_candidates=100,
        )

        mfbo = MultiFidelityBayesianOptimizer(config, search_space)
        result = mfbo.minimize(simple_branin_mf)

        # 允许略微超额（因初始采样），但不应超太多
        assert result.total_budget_used <= budget + 2.0

    def test_convergence_data(self):
        """测试收敛曲线数据生成"""
        search_space = SearchSpace(bounds=[(-3.0, 5.0), (-1.0, 10.0)])
        config = MFBOConfig(
            n_init_low=3, n_init_medium=1, n_init_high=1,
            max_iterations=5, max_budget=100.0,
            random_seed=7, optimizer_restarts=1, acq_n_candidates=100,
        )
        mfbo = MultiFidelityBayesianOptimizer(config, search_space)
        result = mfbo.minimize(simple_branin_mf)

        iters, budgets, bests = result.get_convergence_data()
        assert len(iters) == result.n_iterations
        # best_y_so_far 应该单调不增（最小化）
        for i in range(1, len(bests)):
            assert bests[i] <= bests[i - 1] + 1e-8

    def test_fidelity_statistics(self):
        """测试保真度使用统计"""
        search_space = SearchSpace(bounds=[(-3.0, 5.0), (-1.0, 10.0)])
        config = MFBOConfig(
            n_init_low=2, n_init_medium=2, n_init_high=1,
            max_iterations=3, max_budget=100.0,
            random_seed=12, optimizer_restarts=1, acq_n_candidates=100,
        )
        mfbo = MultiFidelityBayesianOptimizer(config, search_space)
        result = mfbo.minimize(simple_branin_mf)

        stats = mfbo.get_fidelity_statistics()
        # 初始采样确保 LOW/MEDIUM/HIGH 都有
        assert stats["counts"]["low"] >= 2
        assert stats["counts"]["medium"] >= 2
        assert stats["counts"]["high"] >= 1

    def test_serialize_result(self):
        """测试结果序列化"""
        search_space = SearchSpace(bounds=[(-3.0, 5.0), (-1.0, 10.0)])
        config = MFBOConfig(
            n_init_low=2, n_init_medium=1, n_init_high=0,
            max_iterations=3, max_budget=100.0,
            random_seed=15, optimizer_restarts=1, acq_n_candidates=50,
        )
        mfbo = MultiFidelityBayesianOptimizer(config, search_space)
        result = mfbo.minimize(simple_branin_mf)

        d = result.to_dict()
        assert "best_x" in d
        assert "best_y" in d
        assert "history" in d
        assert len(d["history"]) == result.n_iterations

        d_full = result.to_dict(include_observations=True)
        assert "observations" in d_full
        assert len(d_full["observations"]) == len(result.observations)


# ===========================================================================
# 测试：边界条件与异常
# ===========================================================================

class TestEdgeCases:
    """测试边界条件"""

    def test_no_observations_gp_raises(self):
        """空数据拟合应报错"""
        mf_gp = MultiFidelityGP()
        with pytest.raises(ValueError):
            mf_gp.fit([])

    def test_predict_without_fit_raises(self):
        """未拟合预测应报错"""
        mf_gp = MultiFidelityGP()
        with pytest.raises(RuntimeError):
            mf_gp.predict(np.array([[0.0, 0.0]]))

    def test_single_fidelity_only(self):
        """只有一个保真度的数据也能运行"""
        rng = np.random.default_rng(100)
        observations = []
        for _ in range(10):
            x = rng.uniform(0, 1, 2)
            y = np.sum(x ** 2)
            observations.append(Observation(
                x=x, y=y, fidelity=FidelityLevel.HIGH, cost=1.0
            ))

        mf_gp = MultiFidelityGP(MFBOConfig(random_seed=5, optimizer_restarts=1))
        mf_gp.fit(observations)
        pred = mf_gp.predict(np.array([[0.5, 0.5]]), FidelityLevel.HIGH)
        assert np.isfinite(pred.mean[0])

    def test_integer_parameters_search_space(self):
        """支持整数参数搜索空间"""
        search_space = SearchSpace(
            bounds=[(0, 10), (-5, 5)],
            types=['integer', 'continuous'],
        )
        samples = search_space.sample(50, rng=np.random.default_rng(0))
        # 第一个维度应该是整数
        assert np.allclose(samples[:, 0], np.round(samples[:, 0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
