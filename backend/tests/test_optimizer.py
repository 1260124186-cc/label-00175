# -*- coding: utf-8 -*-
"""
优化器模块单元测试
"""

import pytest
import numpy as np
from algorithms.optimizer import (
    GradientDescentOptimizer,
    BFGSOptimizer,
    NewtonOptimizer,
    AdamOptimizer,
    RMSpropOptimizer,
    OptimizationResult
)
from algorithms.advanced_optimizer import (
    GeneticAlgorithmOptimizer,
    ParticleSwarmOptimizer,
    ReinforcementLearningOptimizer,
    SimpleQLearningModel,
    SimulatedAnnealingOptimizer,
    DifferentialEvolutionOptimizer,
    CMAESOptimizer
)
from algorithms.mask_optimizer import (
    MaskOptimizer,
    OptimizationConfig,
    LearningRateScheduler,
    EarlyStopping
)


class TestGradientDescentOptimizer:
    """梯度下降优化器测试"""
    
    def test_simple_quadratic(self):
        """测试简单二次函数优化"""
        # f(x) = x^2, 最小值在x=0
        def objective(x):
            return np.sum(x ** 2)
        
        def gradient(x):
            return 2 * x
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.1,
            max_iter=100,
            tol=1e-6
        )
        
        x0 = np.array([5.0, 5.0])
        result = optimizer.optimize(objective, x0, gradient)
        
        assert result.fun < 0.01
        assert np.allclose(result.x, [0, 0], atol=0.1)
    
    def test_with_bounds(self):
        """测试带边界约束的优化"""
        def objective(x):
            return np.sum((x - 0.5) ** 2)
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.1,
            max_iter=50
        )
        
        x0 = np.array([0.0, 0.0])
        result = optimizer.optimize(objective, x0, bounds=(0.0, 1.0))
        
        assert np.all(result.x >= 0)
        assert np.all(result.x <= 1)
    
    def test_history_recorded(self):
        """测试历史记录"""
        def objective(x):
            return np.sum(x ** 2)
        
        optimizer = GradientDescentOptimizer(max_iter=10)
        x0 = np.array([1.0, 1.0])
        result = optimizer.optimize(objective, x0)
        
        assert len(result.history) > 0
        # 损失应该递减
        assert result.history[-1] <= result.history[0]


class TestBFGSOptimizer:
    """BFGS优化器测试"""
    
    def test_rosenbrock(self):
        """测试Rosenbrock函数优化"""
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        
        optimizer = BFGSOptimizer(max_iter=200, tol=1e-6)
        x0 = np.array([0.0, 0.0])
        result = optimizer.optimize(rosenbrock, x0)
        
        # Rosenbrock最小值在(1, 1)
        assert np.allclose(result.x, [1, 1], atol=0.1)
    
    def test_with_gradient(self):
        """测试提供梯度函数"""
        def objective(x):
            return np.sum(x ** 2)
        
        def gradient(x):
            return 2 * x
        
        optimizer = BFGSOptimizer(max_iter=50)
        x0 = np.array([5.0, 5.0])
        result = optimizer.optimize(objective, x0, gradient)
        
        assert result.fun < 0.01


class TestNewtonOptimizer:
    """牛顿法优化器测试"""
    
    def test_quadratic_convergence(self):
        """测试二次函数快速收敛"""
        def objective(x):
            return np.sum(x ** 2)
        
        optimizer = NewtonOptimizer(max_iter=20)
        x0 = np.array([5.0, 5.0])
        result = optimizer.optimize(objective, x0)
        
        # 牛顿法对二次函数应该快速收敛
        assert result.nit < 10


class TestGeneticAlgorithmOptimizer:
    """遗传算法优化器测试"""
    
    def test_simple_optimization(self):
        """测试简单优化问题"""
        def objective(x):
            return np.sum(x ** 2)
        
        optimizer = GeneticAlgorithmOptimizer(
            population_size=50,
            max_iter=100,
            seed=42
        )
        
        x0 = np.array([5.0, 5.0])
        result = optimizer.optimize(objective, x0, bounds=(-10, 10))
        
        # GA应该能找到比初始值更好的解
        initial_value = np.sum(x0 ** 2)
        assert result.fun < initial_value
    
    def test_reproducibility(self):
        """测试结果可复现性"""
        def objective(x):
            return np.sum(x ** 2)
        
        x0 = np.array([5.0, 5.0])
        
        optimizer1 = GeneticAlgorithmOptimizer(
            population_size=20, max_iter=30, seed=42
        )
        result1 = optimizer1.optimize(objective, x0)
        
        optimizer2 = GeneticAlgorithmOptimizer(
            population_size=20, max_iter=30, seed=42
        )
        result2 = optimizer2.optimize(objective, x0)
        
        assert abs(result1.fun - result2.fun) < 1e-10


class TestParticleSwarmOptimizer:
    """粒子群优化器测试"""
    
    def test_simple_optimization(self):
        """测试简单优化问题"""
        def objective(x):
            return np.sum(x ** 2)
        
        optimizer = ParticleSwarmOptimizer(
            population_size=20,
            max_iter=50,
            seed=42
        )
        
        x0 = np.array([5.0, 5.0])
        result = optimizer.optimize(objective, x0, bounds=(-10, 10))
        
        assert result.fun < 1.0


class TestLearningRateScheduler:
    """学习率调度器测试"""
    
    def test_step_scheduler(self):
        """测试阶梯衰减"""
        scheduler = LearningRateScheduler(
            initial_lr=0.1,
            scheduler_type='step',
            decay=0.5,
            step_size=10
        )
        
        lr_0 = scheduler.step(0)
        lr_10 = scheduler.step(10)
        lr_20 = scheduler.step(20)
        
        assert lr_0 == 0.1
        assert lr_10 == 0.05
        assert lr_20 == 0.025
    
    def test_exponential_scheduler(self):
        """测试指数衰减"""
        scheduler = LearningRateScheduler(
            initial_lr=0.1,
            scheduler_type='exponential',
            decay=0.9
        )
        
        lr_0 = scheduler.step(0)
        lr_1 = scheduler.step(1)
        
        assert lr_0 == 0.1
        assert abs(lr_1 - 0.09) < 1e-10
    
    def test_min_lr(self):
        """测试最小学习率限制"""
        scheduler = LearningRateScheduler(
            initial_lr=0.1,
            scheduler_type='exponential',
            decay=0.1,
            min_lr=0.001
        )
        
        # 多次衰减后应该不低于min_lr
        for i in range(100):
            scheduler.step(i)
        
        assert scheduler.current_lr >= 0.001


class TestEarlyStopping:
    """早停机制测试"""
    
    def test_no_improvement(self):
        """测试无改善时早停"""
        early_stop = EarlyStopping(patience=3)
        
        # 模拟损失不下降
        losses = [1.0, 1.0, 1.0, 1.0]
        
        for loss in losses:
            should_stop = early_stop(loss)
        
        assert should_stop
    
    def test_with_improvement(self):
        """测试有改善时不早停"""
        early_stop = EarlyStopping(patience=3)
        
        # 模拟损失持续下降
        losses = [1.0, 0.9, 0.8, 0.7]
        
        for loss in losses:
            should_stop = early_stop(loss)
        
        assert not should_stop
    
    def test_best_loss_tracking(self):
        """测试最佳损失跟踪"""
        early_stop = EarlyStopping(patience=5)
        
        losses = [1.0, 0.5, 0.6, 0.4, 0.5]
        
        for loss in losses:
            early_stop(loss)
        
        assert early_stop.best_loss == 0.4


class TestMaskOptimizer:
    """掩模优化器测试"""
    
    def test_basic_optimization(self):
        """测试基本优化流程"""
        config = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=10,
            learning_rate=0.1,
            verbose=False
        )
        
        optimizer = MaskOptimizer(config=config)
        
        # 创建简单测试数据
        initial_mask = np.random.random((16, 16))
        target = np.random.random((16, 16))
        
        result = optimizer.optimize(initial_mask, target)
        
        assert result.optimized_mask.shape == (16, 16)
        assert len(result.loss_history) > 0
    
    def test_different_optimizers(self):
        """测试不同优化器"""
        optimizer_types = ['gradient_descent', 'bfgs']
        
        for opt_type in optimizer_types:
            config = OptimizationConfig(
                optimizer_type=opt_type,
                max_iter=5,
                verbose=False
            )
            
            optimizer = MaskOptimizer(config=config)
            
            initial_mask = np.random.random((16, 16))
            target = np.random.random((16, 16))
            
            result = optimizer.optimize(initial_mask, target)
            
            assert result.optimized_mask is not None
    
    def test_custom_objective(self):
        """测试自定义目标函数"""
        config = OptimizationConfig(
            optimizer_type='gradient_descent',
            max_iter=10,
            verbose=False
        )
        
        optimizer = MaskOptimizer(config=config)
        
        def custom_objective(x):
            return np.sum(x ** 2)
        
        initial_mask = np.random.random((8, 8))
        result = optimizer.optimize_with_custom_objective(
            initial_mask, custom_objective
        )
        
        assert result.x is not None

    def test_with_random_seed(self):
        """测试随机种子配置"""
        config = OptimizationConfig(
            optimizer_type='genetic',
            max_iter=10,
            random_seed=42,
            verbose=False
        )
        
        optimizer1 = MaskOptimizer(config=config)
        optimizer2 = MaskOptimizer(config=config)
        
        initial_mask = np.random.RandomState(123).random((8, 8))
        target = np.random.RandomState(456).random((8, 8))
        
        result1 = optimizer1.optimize(initial_mask.copy(), target)
        result2 = optimizer2.optimize(initial_mask.copy(), target)
        
        # 相同种子应该产生相同结果
        assert abs(result1.final_metrics.mse - result2.final_metrics.mse) < 1e-10


class TestReinforcementLearningOptimizer:
    """强化学习优化器测试"""
    
    def test_basic_rl_optimization(self):
        """测试基本RL优化"""
        def objective(x):
            return np.sum(x ** 2)
        
        optimizer = ReinforcementLearningOptimizer(
            max_iter=20,
            seed=42,
            verbose=False
        )
        
        x0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        target = np.zeros_like(x0)
        
        result = optimizer.optimize(objective, x0, target=target)
        
        assert result.x is not None
        assert len(result.history) > 0
    
    def test_with_custom_model(self):
        """测试自定义RL模型"""
        def objective(x):
            return np.sum(x ** 2)
        
        x0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        state_dim = x0.size * 2  # mask + error
        action_dim = x0.size
        
        model = SimpleQLearningModel(state_dim, action_dim)
        
        optimizer = ReinforcementLearningOptimizer(
            max_iter=10,
            seed=42,
            verbose=False,
            state_encoding='simple',
        )
        optimizer.set_model(model)
        
        result = optimizer.optimize(objective, x0)
        
        assert result.x is not None
    
    def test_epsilon_decay(self):
        """测试探索率衰减"""
        optimizer = ReinforcementLearningOptimizer(
            epsilon=1.0,
            epsilon_decay=0.9,
            min_epsilon=0.1,
            max_iter=10,
            seed=42
        )
        
        def objective(x):
            return np.sum(x ** 2)
        
        x0 = np.array([[0.5, 0.5]])
        optimizer.optimize(objective, x0)
        
        # 探索率应该衰减
        assert optimizer.epsilon < 1.0
        assert optimizer.epsilon >= 0.1


class TestAdamOptimizer:
    """Adam 优化器测试"""

    def test_simple_quadratic(self):
        """测试简单二次函数优化"""
        def objective(x):
            return np.sum(x ** 2)

        def gradient(x):
            return 2 * x

        optimizer = AdamOptimizer(
            learning_rate=0.1,
            max_iter=200,
            tol=1e-6
        )

        x0 = np.array([5.0, 5.0])
        result = optimizer.optimize(objective, x0, gradient)

        assert result.fun < 0.01
        assert np.allclose(result.x, [0, 0], atol=0.1)

    def test_without_gradient(self):
        """测试不提供梯度时使用数值梯度"""
        def objective(x):
            return np.sum((x - 1.0) ** 2)

        optimizer = AdamOptimizer(
            learning_rate=0.05,
            max_iter=300
        )

        x0 = np.array([2.0, 2.0])
        result = optimizer.optimize(objective, x0)

        assert result.fun < 0.1
        assert np.allclose(result.x, [1.0, 1.0], atol=0.2)

    def test_history_recorded(self):
        """测试历史记录"""
        def objective(x):
            return np.sum(x ** 2)

        optimizer = AdamOptimizer(max_iter=10)
        x0 = np.array([1.0, 1.0])
        result = optimizer.optimize(objective, x0)

        assert len(result.history) > 0
        assert result.history[-1] <= result.history[0]


class TestRMSpropOptimizer:
    """RMSprop 优化器测试"""

    def test_simple_quadratic(self):
        """测试简单二次函数优化"""
        def objective(x):
            return np.sum(x ** 2)

        def gradient(x):
            return 2 * x

        optimizer = RMSpropOptimizer(
            learning_rate=0.05,
            max_iter=200,
            tol=1e-6
        )

        x0 = np.array([5.0, 5.0])
        result = optimizer.optimize(objective, x0, gradient)

        assert result.fun < 0.01
        assert np.allclose(result.x, [0, 0], atol=0.1)

    def test_with_momentum(self):
        """测试带动量的 RMSprop"""
        def objective(x):
            return np.sum((x - 2.0) ** 2)

        def gradient(x):
            return 2 * (x - 2.0)

        optimizer = RMSpropOptimizer(
            learning_rate=0.01,
            momentum=0.9,
            max_iter=300
        )

        x0 = np.array([0.0, 0.0])
        result = optimizer.optimize(objective, x0, gradient)

        assert result.fun < 0.1
        assert np.allclose(result.x, [2.0, 2.0], atol=0.2)

    def test_with_bounds(self):
        """测试带边界约束"""
        def objective(x):
            return np.sum((x - 0.5) ** 2)

        optimizer = RMSpropOptimizer(
            learning_rate=0.05,
            max_iter=100
        )

        x0 = np.array([0.0, 0.0])
        result = optimizer.optimize(objective, x0, bounds=(0.0, 1.0))

        assert np.all(result.x >= 0)
        assert np.all(result.x <= 1)


class TestSimulatedAnnealingOptimizer:
    """模拟退火优化器测试"""

    def test_simple_optimization(self):
        """测试简单优化问题"""
        def objective(x):
            return np.sum(x ** 2)

        optimizer = SimulatedAnnealingOptimizer(
            initial_temperature=10.0,
            cooling_rate=0.95,
            step_size=0.2,
            max_iter=200,
            seed=42
        )

        x0 = np.array([5.0, 5.0])
        result = optimizer.optimize(objective, x0, bounds=(-10, 10))

        initial_value = np.sum(x0 ** 2)
        assert result.fun < initial_value

    def test_bounds_respected(self):
        """测试边界约束被遵守"""
        def objective(x):
            return np.sum(x ** 2)

        optimizer = SimulatedAnnealingOptimizer(
            max_iter=50,
            seed=42
        )

        x0 = np.array([0.5, 0.5])
        result = optimizer.optimize(objective, x0, bounds=(0.0, 1.0))

        assert np.all(result.x >= 0.0)
        assert np.all(result.x <= 1.0)

    def test_history_recorded(self):
        """测试历史记录"""
        def objective(x):
            return np.sum(x ** 2)

        optimizer = SimulatedAnnealingOptimizer(
            max_iter=20,
            seed=42
        )
        x0 = np.array([1.0, 1.0])
        result = optimizer.optimize(objective, x0)

        assert len(result.history) > 0


class TestDifferentialEvolutionOptimizer:
    """差分进化优化器测试"""

    def test_simple_optimization(self):
        """测试简单优化问题"""
        def objective(x):
            return np.sum(x ** 2)

        optimizer = DifferentialEvolutionOptimizer(
            population_size=30,
            max_iter=100,
            seed=42
        )

        x0 = np.array([5.0, 5.0])
        result = optimizer.optimize(objective, x0, bounds=(-10, 10))

        assert result.fun < 1.0

    def test_reproducibility(self):
        """测试结果可复现性"""
        def objective(x):
            return np.sum(x ** 2)

        x0 = np.array([5.0, 5.0])

        optimizer1 = DifferentialEvolutionOptimizer(
            population_size=20, max_iter=30, seed=42
        )
        result1 = optimizer1.optimize(objective, x0)

        optimizer2 = DifferentialEvolutionOptimizer(
            population_size=20, max_iter=30, seed=42
        )
        result2 = optimizer2.optimize(objective, x0)

        assert abs(result1.fun - result2.fun) < 1e-10

    def test_different_strategies(self):
        """测试不同变异策略"""
        def objective(x):
            return np.sum((x - 1.0) ** 2)

        for strategy in ['best1bin', 'rand1bin']:
            optimizer = DifferentialEvolutionOptimizer(
                population_size=20,
                max_iter=50,
                strategy=strategy,
                seed=42
            )
            x0 = np.array([0.0, 0.0])
            result = optimizer.optimize(objective, x0, bounds=(-5, 5))
            assert result.fun < 5.0


class TestCMAESOptimizer:
    """CMA-ES 优化器测试"""

    def test_simple_optimization(self):
        """测试简单优化问题"""
        def objective(x):
            return np.sum(x ** 2)

        optimizer = CMAESOptimizer(
            population_size=20,
            max_iter=50,
            sigma=0.5,
            seed=42
        )

        x0 = np.array([2.0, 2.0])
        result = optimizer.optimize(objective, x0, bounds=(-5, 5))

        assert result.fun < 1.0

    def test_history_recorded(self):
        """测试历史记录"""
        def objective(x):
            return np.sum(x ** 2)

        optimizer = CMAESOptimizer(
            population_size=15,
            max_iter=10,
            seed=42
        )
        x0 = np.array([1.0, 1.0])
        result = optimizer.optimize(objective, x0)

        assert len(result.history) > 0

    def test_bounds_respected(self):
        """测试边界约束"""
        def objective(x):
            return np.sum(x ** 2)

        optimizer = CMAESOptimizer(
            population_size=15,
            max_iter=20,
            seed=42
        )

        x0 = np.array([0.5, 0.5])
        result = optimizer.optimize(objective, x0, bounds=(0.0, 1.0))

        assert np.all(result.x >= 0.0)
        assert np.all(result.x <= 1.0)
