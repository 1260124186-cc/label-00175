# -*- coding: utf-8 -*-
"""
进阶优化器模块：启发式算法接口

该模块预留启发式算法接口，包括：
1. 遗传算法 (Genetic Algorithm)
2. 粒子群优化 (Particle Swarm Optimization)
3. 强化学习接口（预留）
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass, field
import logging

from algorithms.optimizer import OptimizationResult

logger = logging.getLogger(__name__)


class BaseHeuristicOptimizer(ABC):
    """
    启发式优化器基类
    
    定义启发式算法的通用接口，支持自定义目标函数接入。
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iter: int = 100,
                 seed: Optional[int] = None,
                 verbose: bool = False):
        """
        初始化启发式优化器
        
        Args:
            population_size: 种群大小
            max_iter: 最大迭代次数
            seed: 随机种子（用于结果复现）
            verbose: 是否输出详细信息
        """
        self.population_size = population_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.history: List[float] = []
        
        if seed is not None:
            np.random.seed(seed)
    
    @abstractmethod
    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 **kwargs) -> OptimizationResult:
        """
        执行优化
        
        Args:
            objective: 目标函数
            x0: 初始解（用于确定问题维度）
            bounds: 变量边界
            **kwargs: 其他参数
            
        Returns:
            OptimizationResult对象
        """
        pass
    
    def set_custom_objective(self, 
                             objective_func: Callable[[np.ndarray], float]):
        """
        设置自定义目标函数
        
        Args:
            objective_func: 自定义目标函数
        """
        self.custom_objective = objective_func


class GeneticAlgorithmOptimizer(BaseHeuristicOptimizer):
    """
    遗传算法优化器
    
    实现基本的遗传算法，包括选择、交叉、变异操作。
    """
    
    def __init__(self,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_ratio: float = 0.1,
                 **kwargs):
        """
        初始化遗传算法优化器
        
        Args:
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elite_ratio: 精英保留比例
        """
        super().__init__(**kwargs)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
    
    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 **kwargs) -> OptimizationResult:
        """执行遗传算法优化"""
        shape = x0.shape
        dim = x0.size
        
        # 初始化种群
        population = self._initialize_population(dim, bounds)
        
        self.history = []
        best_x = None
        best_fitness = float('inf')
        nfev = 0
        
        for gen in range(self.max_iter):
            # 评估适应度
            fitness = np.array([objective(ind.reshape(shape)) for ind in population])
            nfev += len(population)
            
            # 记录最优
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_x = population[min_idx].copy()
            
            self.history.append(best_fitness)
            
            if self.verbose and gen % 10 == 0:
                logger.info(f"代数 {gen}: 最优适应度 = {best_fitness:.6e}")
            
            # 选择
            selected = self._selection(population, fitness)
            
            # 交叉
            offspring = self._crossover(selected, bounds)
            
            # 变异
            offspring = self._mutation(offspring, bounds)
            
            # 精英保留
            n_elite = max(1, int(self.elite_ratio * self.population_size))
            elite_indices = np.argsort(fitness)[:n_elite]
            elite = population[elite_indices]
            
            # 新种群
            population = np.vstack([elite, offspring[:self.population_size - n_elite]])
        
        return OptimizationResult(
            x=best_x.reshape(shape),
            fun=best_fitness,
            nit=self.max_iter,
            nfev=nfev,
            success=True,
            message="遗传算法完成",
            history=self.history
        )
    
    def _initialize_population(self, dim: int, bounds: Tuple[float, float]) -> np.ndarray:
        """初始化种群"""
        return np.random.uniform(
            bounds[0], bounds[1],
            size=(self.population_size, dim)
        )
    
    def _selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """轮盘赌选择"""
        # 转换为最大化问题
        max_fit = np.max(fitness)
        adjusted_fitness = max_fit - fitness + 1e-10
        probs = adjusted_fitness / np.sum(adjusted_fitness)
        
        indices = np.random.choice(
            len(population),
            size=self.population_size,
            p=probs
        )
        return population[indices]
    
    def _crossover(self, population: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        """均匀交叉"""
        offspring = []
        
        for i in range(0, len(population) - 1, 2):
            parent1, parent2 = population[i], population[i + 1]
            
            if np.random.random() < self.crossover_rate:
                # 均匀交叉
                mask = np.random.random(len(parent1)) < 0.5
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            offspring.extend([child1, child2])
        
        return np.array(offspring)
    
    def _mutation(self, population: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        """高斯变异"""
        for i in range(len(population)):
            if np.random.random() < self.mutation_rate:
                # 随机选择变异位置
                mutation_mask = np.random.random(len(population[i])) < 0.1
                noise = np.random.normal(0, 0.1, len(population[i]))
                population[i] = population[i] + mutation_mask * noise
                population[i] = np.clip(population[i], bounds[0], bounds[1])
        
        return population


class ParticleSwarmOptimizer(BaseHeuristicOptimizer):
    """
    粒子群优化器 (PSO)
    
    实现标准粒子群优化算法。
    """
    
    def __init__(self,
                 w: float = 0.7,  # 惯性权重
                 c1: float = 1.5,  # 认知系数
                 c2: float = 1.5,  # 社会系数
                 **kwargs):
        """
        初始化粒子群优化器
        
        Args:
            w: 惯性权重
            c1: 认知系数（个体学习因子）
            c2: 社会系数（群体学习因子）
        """
        super().__init__(**kwargs)
        self.w = w
        self.c1 = c1
        self.c2 = c2
    
    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 **kwargs) -> OptimizationResult:
        """执行粒子群优化"""
        shape = x0.shape
        dim = x0.size
        
        # 初始化粒子位置和速度
        positions = np.random.uniform(
            bounds[0], bounds[1],
            size=(self.population_size, dim)
        )
        velocities = np.random.uniform(
            -0.1, 0.1,
            size=(self.population_size, dim)
        )
        
        # 个体最优和全局最优
        pbest = positions.copy()
        pbest_fitness = np.array([objective(p.reshape(shape)) for p in positions])
        
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        
        self.history = [gbest_fitness]
        nfev = self.population_size
        
        for it in range(self.max_iter):
            # 更新速度和位置
            r1 = np.random.random((self.population_size, dim))
            r2 = np.random.random((self.population_size, dim))
            
            velocities = (self.w * velocities +
                         self.c1 * r1 * (pbest - positions) +
                         self.c2 * r2 * (gbest - positions))
            
            # 速度限制
            v_max = 0.2 * (bounds[1] - bounds[0])
            velocities = np.clip(velocities, -v_max, v_max)
            
            positions = positions + velocities
            positions = np.clip(positions, bounds[0], bounds[1])
            
            # 评估适应度
            fitness = np.array([objective(p.reshape(shape)) for p in positions])
            nfev += self.population_size
            
            # 更新个体最优
            improved = fitness < pbest_fitness
            pbest[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]
            
            # 更新全局最优
            min_idx = np.argmin(pbest_fitness)
            if pbest_fitness[min_idx] < gbest_fitness:
                gbest = pbest[min_idx].copy()
                gbest_fitness = pbest_fitness[min_idx]
            
            self.history.append(gbest_fitness)
            
            if self.verbose and it % 10 == 0:
                logger.info(f"迭代 {it}: 全局最优 = {gbest_fitness:.6e}")
        
        return OptimizationResult(
            x=gbest.reshape(shape),
            fun=gbest_fitness,
            nit=self.max_iter,
            nfev=nfev,
            success=True,
            message="粒子群优化完成",
            history=self.history
        )


class ReinforcementLearningOptimizer(BaseHeuristicOptimizer):
    """
    强化学习优化器接口
    
    该类提供强化学习方法的接口框架，支持自定义RL模型接入。
    可用于掩模优化的策略学习。
    """
    
    def __init__(self, 
                 state_dim: Optional[int] = None,
                 action_dim: Optional[int] = None,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01,
                 **kwargs):
        """
        初始化强化学习优化器
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            min_epsilon: 最小探索率
        """
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self._model = None
        self._replay_buffer: List[Tuple] = []
        self._buffer_size = 10000
    
    def set_model(self, model):
        """
        设置强化学习模型
        
        Args:
            model: 强化学习模型（需实现predict和update方法）
                   - predict(state) -> action
                   - update(batch) -> loss
        """
        self._model = model
    
    def _get_state(self, mask: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        从掩模和目标图像提取状态特征
        
        Args:
            mask: 当前掩模
            target: 目标图像
            
        Returns:
            状态向量
        """
        # 默认实现：展平并拼接
        mask_flat = mask.flatten()
        target_flat = target.flatten()
        
        # 可以添加更多特征：误差图、频域特征等
        error = np.abs(mask_flat - target_flat)
        
        return np.concatenate([mask_flat, error])
    
    def _apply_action(self, mask: np.ndarray, action: np.ndarray,
                      bounds: Tuple[float, float]) -> np.ndarray:
        """
        将动作应用到掩模上
        
        Args:
            mask: 当前掩模
            action: 动作（掩模调整量）
            bounds: 值边界
            
        Returns:
            更新后的掩模
        """
        new_mask = mask + action.reshape(mask.shape)
        return np.clip(new_mask, bounds[0], bounds[1])
    
    def _select_action(self, state: np.ndarray, shape: Tuple) -> np.ndarray:
        """
        选择动作（epsilon-greedy策略）
        
        Args:
            state: 当前状态
            shape: 掩模形状
            
        Returns:
            动作向量
        """
        if self._model is None or np.random.random() < self.epsilon:
            # 随机探索
            return np.random.uniform(-0.1, 0.1, np.prod(shape))
        else:
            # 利用模型
            return self._model.predict(state)
    
    def _compute_reward(self, old_loss: float, new_loss: float) -> float:
        """
        计算奖励
        
        Args:
            old_loss: 旧损失值
            new_loss: 新损失值
            
        Returns:
            奖励值
        """
        # 损失减少则正奖励
        improvement = old_loss - new_loss
        return improvement * 100  # 放大奖励信号
    
    def _store_transition(self, state: np.ndarray, action: np.ndarray,
                          reward: float, next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        if len(self._replay_buffer) >= self._buffer_size:
            self._replay_buffer.pop(0)
        self._replay_buffer.append((state, action, reward, next_state, done))
    
    def _sample_batch(self, batch_size: int = 32) -> List[Tuple]:
        """从回放缓冲区采样"""
        if len(self._replay_buffer) < batch_size:
            return self._replay_buffer
        indices = np.random.choice(len(self._replay_buffer), batch_size, replace=False)
        return [self._replay_buffer[i] for i in indices]
    
    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 target: Optional[np.ndarray] = None,
                 **kwargs) -> OptimizationResult:
        """
        执行强化学习优化
        
        Args:
            objective: 目标函数
            x0: 初始掩模
            bounds: 值边界
            target: 目标图像（用于状态构建）
            
        Returns:
            OptimizationResult对象
        """
        shape = x0.shape
        mask = x0.copy()
        
        if target is None:
            target = np.zeros_like(x0)
        
        self.history = []
        best_mask = mask.copy()
        best_loss = objective(mask)
        self.history.append(best_loss)
        
        nfev = 1
        
        for episode in range(self.max_iter):
            state = self._get_state(mask, target)
            action = self._select_action(state, shape)
            
            # 应用动作
            new_mask = self._apply_action(mask, action, bounds)
            new_loss = objective(new_mask)
            nfev += 1
            
            # 计算奖励
            reward = self._compute_reward(best_loss if best_loss < float('inf') else new_loss, new_loss)
            
            # 获取新状态
            next_state = self._get_state(new_mask, target)
            done = episode == self.max_iter - 1
            
            # 存储经验
            self._store_transition(state, action, reward, next_state, done)
            
            # 更新模型（如果有）
            if self._model is not None and len(self._replay_buffer) >= 32:
                batch = self._sample_batch(32)
                self._model.update(batch)
            
            # 更新最优解
            if new_loss < best_loss:
                best_loss = new_loss
                best_mask = new_mask.copy()
            
            mask = new_mask
            self.history.append(best_loss)
            
            # 衰减探索率
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            if self.verbose and episode % 10 == 0:
                logger.info(f"Episode {episode}: loss = {best_loss:.6e}, epsilon = {self.epsilon:.4f}")
        
        return OptimizationResult(
            x=best_mask,
            fun=best_loss,
            nit=self.max_iter,
            nfev=nfev,
            success=True,
            message="强化学习优化完成",
            history=self.history
        )


class SimpleQLearningModel:
    """
    简单Q-Learning模型示例
    
    用于演示如何接入自定义RL模型。
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.01):
        """
        初始化Q-Learning模型
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        
        # 简单线性Q函数
        self.weights = np.random.randn(state_dim, action_dim) * 0.01
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """预测动作"""
        q_values = state @ self.weights
        # 返回连续动作（基于Q值的加权）
        return np.tanh(q_values) * 0.1
    
    def update(self, batch: List[Tuple]) -> float:
        """更新模型"""
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            # 简化的TD更新
            current_q = state @ self.weights
            next_q = next_state @ self.weights
            
            target = reward
            if not done:
                target += 0.99 * np.max(next_q)
            
            # 梯度更新
            error = target - np.mean(current_q)
            self.weights += self.lr * np.outer(state, np.ones(self.action_dim)) * error
            total_loss += error ** 2
        
        return total_loss / len(batch)
