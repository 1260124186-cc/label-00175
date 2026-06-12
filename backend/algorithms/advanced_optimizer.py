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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from algorithms.optimizer import OptimizationResult
from algorithms.deep_rl_models import (
    TORCH_AVAILABLE,
    MultiChannelStateEncoder,
    StateEncoderConfig,
    DQNModel,
    DQNConfig,
    PPOModel,
    PPOConfig,
    ActorCriticModel,
    ActorCriticConfig,
    DeepRLModelFactory,
)

logger = logging.getLogger(__name__)


def _evaluate_single(args):
    """
    模块级辅助函数：评估单个个体的适应度

    必须是模块级函数才能被 multiprocessing pickle。

    Args:
        args: (objective_func, individual, shape) 元组

    Returns:
        适应度值
    """
    objective_func, individual, shape = args
    return objective_func(individual.reshape(shape))


class BaseHeuristicOptimizer(ABC):
    """
    启发式优化器基类
    
    定义启发式算法的通用接口，支持自定义目标函数接入。
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iter: int = 100,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 n_jobs: int = 1):
        """
        初始化启发式优化器
        
        Args:
            population_size: 种群大小
            max_iter: 最大迭代次数
            seed: 随机种子（用于结果复现）
            verbose: 是否输出详细信息
            n_jobs: 并行工作进程数，1表示串行，-1表示使用所有CPU核心
        """
        self.population_size = population_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.history: List[float] = []
        self.n_jobs = self._resolve_n_jobs(n_jobs)
        self._parallel_available = None

        if seed is not None:
            np.random.seed(seed)

    def _resolve_n_jobs(self, n_jobs: int) -> int:
        """解析 n_jobs 参数，-1 表示使用所有 CPU 核心"""
        if n_jobs == -1:
            return os.cpu_count() or 1
        return max(1, n_jobs)

    def _evaluate_population(self,
                             population: np.ndarray,
                             objective: Callable[[np.ndarray], float],
                             shape: Tuple[int, ...]) -> np.ndarray:
        """
        评估种群中所有个体的适应度

        当 n_jobs > 1 时使用多进程并行评估，否则串行评估。
        如果并行执行失败（如pickle错误），自动回退到串行模式。

        Args:
            population: 种群数组，形状为 (population_size, dim)
            objective: 目标函数
            shape: 个体的原始形状（用于reshape）

        Returns:
            适应度数组，形状为 (population_size,)
        """
        n = len(population)

        if self.n_jobs == 1 or n <= 1:
            return np.array([objective(ind.reshape(shape)) for ind in population])

        if self._parallel_available is False:
            return np.array([objective(ind.reshape(shape)) for ind in population])

        try:
            tasks = [(objective, population[i], shape) for i in range(n)]
            results = [None] * n

            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                future_to_idx = {
                    executor.submit(_evaluate_single, tasks[i]): i
                    for i in range(n)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()

            self._parallel_available = True
            return np.array(results)

        except Exception as e:
            logger.warning(
                f"并行适应度评估失败，回退到串行模式: {e}"
            )
            self._parallel_available = False
            return np.array([objective(ind.reshape(shape)) for ind in population])
    
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
            # 评估适应度（支持并行）
            fitness = self._evaluate_population(population, objective, shape)
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
        pbest_fitness = self._evaluate_population(positions, objective, shape)
        
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
            
            # 评估适应度（支持并行）
            fitness = self._evaluate_population(positions, objective, shape)
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
    强化学习优化器

    支持三种深度模型（DQN / PPO / Actor-Critic，PyTorch）和传统表格模型。
    状态编码默认使用多通道表示：局部 patch + 频域特征 + 历史损失。
    当 PyTorch 不可用时自动回退到传统简单状态编码。
    """

    def __init__(self,
                 state_dim: Optional[int] = None,
                 action_dim: Optional[int] = None,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01,
                 model_type: str = 'simple',
                 state_encoding: str = 'multichannel',
                 encoder_config: Optional[StateEncoderConfig] = None,
                 dqn_config: Optional[DQNConfig] = None,
                 ppo_config: Optional[PPOConfig] = None,
                 ac_config: Optional[ActorCriticConfig] = None,
                 device: str = 'cpu',
                 **kwargs):
        """
        初始化强化学习优化器

        Args:
            state_dim: 状态空间维度（仅 simple 编码使用）
            action_dim: 动作空间维度（仅 simple 编码使用）
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            min_epsilon: 最小探索率
            model_type: 模型类型 'simple' | 'dqn' | 'ppo' | 'actor_critic'
            state_encoding: 状态编码方式 'simple' | 'multichannel'
            encoder_config: 多通道编码器配置
            dqn_config: DQN 配置
            ppo_config: PPO 配置
            ac_config: Actor-Critic 配置
            device: PyTorch 设备
        """
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.model_type = model_type
        self.state_encoding = state_encoding
        self.device = device

        self._replay_buffer: List[Tuple] = []
        self._buffer_size = 10000

        if state_encoding == 'multichannel':
            self._state_encoder = MultiChannelStateEncoder(encoder_config)
        else:
            self._state_encoder = None

        self._model = None
        self._deep_model = None
        if model_type != 'simple':
            self._init_deep_model(model_type, dqn_config, ppo_config, ac_config)

    def _init_deep_model(self, model_type: str,
                         dqn_config: Optional[DQNConfig],
                         ppo_config: Optional[PPOConfig],
                         ac_config: Optional[ActorCriticConfig]):
        if not TORCH_AVAILABLE:
            logger.warning(
                "PyTorch not available, falling back to simple model. "
                "Install torch to use deep RL models."
            )
            self.model_type = 'simple'
            return

        if model_type == 'dqn':
            self._deep_model = DQNModel(dqn_config or DQNConfig(), self.device)
        elif model_type == 'ppo':
            self._deep_model = PPOModel(ppo_config or PPOConfig(), self.device)
        elif model_type == 'actor_critic':
            self._deep_model = ActorCriticModel(ac_config or ActorCriticConfig(), self.device)
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}', "
                f"available: {DeepRLModelFactory.available_models() + ['simple']}"
            )

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

        当 state_encoding='multichannel' 时返回 (3, H, W) 多通道表示：
          - 通道 1: 局部 patch 特征
          - 通道 2: 频域特征
          - 通道 3: 历史损失轨迹

        当 state_encoding='simple' 时返回展平拼接的 (mask_flat, error) 向量。

        Args:
            mask: 当前掩模
            target: 目标图像

        Returns:
            状态数组
        """
        if self._state_encoder is not None:
            return self._state_encoder.encode(mask, target)

        mask_flat = mask.flatten()
        target_flat = target.flatten()
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
        if self.model_type in ('dqn',):
            delta = action
            if isinstance(delta, np.ndarray) and delta.size <= 2:
                h, w = mask.shape
                full_delta = np.zeros_like(mask)
                full_delta[:max(1, h // 8), :max(1, w // 8)] = delta.flat[0] if delta.size == 1 else delta[0, 0] if delta.ndim >= 2 else delta[0]
                new_mask = mask + full_delta
            else:
                new_mask = mask + action.reshape(mask.shape)
        else:
            new_mask = mask + action.reshape(mask.shape)
        return np.clip(new_mask, bounds[0], bounds[1])

    def _select_action(self, state: np.ndarray, shape: Tuple) -> np.ndarray:
        """
        选择动作

        Args:
            state: 当前状态
            shape: 掩模形状

        Returns:
            动作向量
        """
        if self._deep_model is not None:
            return self._deep_model.predict(state)

        if self._model is None or np.random.random() < self.epsilon:
            return np.random.uniform(-0.1, 0.1, np.prod(shape))
        else:
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
        improvement = old_loss - new_loss
        return improvement * 100

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

        if self._state_encoder is not None:
            self._state_encoder.reset()

        self.history = []
        best_mask = mask.copy()
        best_loss = objective(mask)
        self.history.append(best_loss)

        if self._state_encoder is not None:
            self._state_encoder.record_loss(best_loss)

        nfev = 1

        for episode in range(self.max_iter):
            state = self._get_state(mask, target)
            action = self._select_action(state, shape)

            new_mask = self._apply_action(mask, action, bounds)
            new_loss = objective(new_mask)
            nfev += 1

            if self._state_encoder is not None:
                self._state_encoder.record_loss(new_loss)

            reward = self._compute_reward(best_loss if best_loss < float('inf') else new_loss, new_loss)

            next_state = self._get_state(new_mask, target)
            done = episode == self.max_iter - 1

            if self._deep_model is not None:
                if isinstance(self._deep_model, PPOModel):
                    self._deep_model.store_transition(state, action, reward, done)
                elif isinstance(self._deep_model, DQNModel):
                    action_idx = 0
                    if isinstance(action, np.ndarray):
                        action_idx = int(np.argmax(np.abs(action.flatten()))) % self._deep_model.config.num_actions
                    self._deep_model.store(state, action_idx, reward, next_state, done)
                    self._deep_model.train_step()
                else:
                    self._store_transition(state, action, reward, next_state, done)
            else:
                self._store_transition(state, action, reward, next_state, done)

            if self._model is not None and len(self._replay_buffer) >= 32:
                batch = self._sample_batch(32)
                self._model.update(batch)

            if self._deep_model is not None and isinstance(self._deep_model, ActorCriticModel):
                if len(self._replay_buffer) >= 16:
                    batch = self._sample_batch(16)
                    self._deep_model.update(batch)

            if self._deep_model is not None and isinstance(self._deep_model, PPOModel):
                if episode > 0 and (episode + 1) % 4 == 0:
                    self._deep_model.update()

            if new_loss < best_loss:
                best_loss = new_loss
                best_mask = new_mask.copy()

            mask = new_mask
            self.history.append(best_loss)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if self.verbose and episode % 10 == 0:
                model_info = self.model_type
                logger.info(
                    f"Episode {episode}: loss = {best_loss:.6e}, "
                    f"epsilon = {self.epsilon:.4f}, model = {model_info}"
                )

        return OptimizationResult(
            x=best_mask,
            fun=best_loss,
            nit=self.max_iter,
            nfev=nfev,
            success=True,
            message=f"强化学习优化完成 (model={self.model_type}, encoding={self.state_encoding})",
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


class SimulatedAnnealingOptimizer(BaseHeuristicOptimizer):
    """
    模拟退火优化器 (Simulated Annealing, SA)

    基于金属退火原理的随机优化算法，允许以一定概率接受更差的解，
    从而有能力跳出局部最优。
    """

    def __init__(self,
                 initial_temperature: float = 100.0,
                 cooling_rate: float = 0.95,
                 min_temperature: float = 1e-8,
                 step_size: float = 0.1,
                 **kwargs):
        """
        初始化模拟退火优化器

        Args:
            initial_temperature: 初始温度
            cooling_rate: 温度衰减系数 (0, 1)
            min_temperature: 最低温度
            step_size: 邻域搜索步长
        """
        super().__init__(**kwargs)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.step_size = step_size

    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 **kwargs) -> OptimizationResult:
        """执行模拟退火优化"""
        shape = x0.shape
        dim = x0.size

        current_x = x0.copy().flatten()
        current_f = objective(current_x.reshape(shape))

        best_x = current_x.copy()
        best_f = current_f

        temperature = self.initial_temperature
        self.history = [best_f]
        nfev = 1
        it = 0

        while temperature > self.min_temperature and it < self.max_iter:
            # 在邻域内生成新解
            noise = np.random.normal(0, self.step_size, dim)
            candidate_x = current_x + noise
            candidate_x = np.clip(candidate_x, bounds[0], bounds[1])

            candidate_f = objective(candidate_x.reshape(shape))
            nfev += 1

            # 计算接受概率
            delta = candidate_f - current_f
            if delta < 0:
                accept = True
            else:
                accept_prob = np.exp(-delta / temperature)
                accept = np.random.random() < accept_prob

            if accept:
                current_x = candidate_x
                current_f = candidate_f

                if current_f < best_f:
                    best_x = current_x.copy()
                    best_f = current_f

            self.history.append(best_f)

            if self.verbose and it % 10 == 0:
                logger.info(
                    f"迭代 {it}: T = {temperature:.4e}, best_f = {best_f:.6e}")

            temperature *= self.cooling_rate
            it += 1

        return OptimizationResult(
            x=best_x.reshape(shape),
            fun=best_f,
            nit=it,
            nfev=nfev,
            success=True,
            message="模拟退火优化完成",
            history=self.history
        )


class DifferentialEvolutionOptimizer(BaseHeuristicOptimizer):
    """
    差分进化优化器 (Differential Evolution, DE)

    基于种群的进化算法，通过差分变异、交叉和选择操作搜索最优解。
    """

    def __init__(self,
                 f: float = 0.8,
                 cr: float = 0.7,
                 strategy: str = 'best1bin',
                 **kwargs):
        """
        初始化差分进化优化器

        Args:
            f: 差分缩放因子 (0, 2]
            cr: 交叉概率 [0, 1]
            strategy: 变异策略 ('best1bin', 'rand1bin', 'rand2bin')
        """
        super().__init__(**kwargs)
        self.f = f
        self.cr = cr
        self.strategy = strategy

    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 **kwargs) -> OptimizationResult:
        """执行差分进化优化"""
        shape = x0.shape
        dim = x0.size

        # 初始化种群
        population = np.random.uniform(
            bounds[0], bounds[1],
            size=(self.population_size, dim)
        )

        # 评估初始种群（支持并行）
        fitness = self._evaluate_population(population, objective, shape)
        nfev = self.population_size

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        self.history = [best_f]

        for gen in range(self.max_iter):
            # 批量生成所有 trial 向量
            trials = np.empty_like(population)

            for i in range(self.population_size):
                # 选择变异基向量
                if self.strategy.startswith('best'):
                    base = population[best_idx]
                else:
                    base = population[np.random.randint(self.population_size)]

                # 选择差分向量
                candidates = [j for j in range(self.population_size) if j != i]
                if self.strategy.endswith('2bin'):
                    r1, r2, r3, r4 = np.random.choice(
                        candidates, 4, replace=False)
                    donor = base + self.f * (
                        population[r1] - population[r2] +
                        population[r3] - population[r4]
                    )
                else:
                    r1, r2, r3 = np.random.choice(
                        candidates, 3, replace=False)
                    donor = base + self.f * (population[r1] - population[r2])

                # 二项式交叉
                cross_mask = np.random.random(dim) < self.cr
                cross_mask[np.random.randint(dim)] = True
                trial = np.where(cross_mask, donor, population[i])

                # 边界处理
                trials[i] = np.clip(trial, bounds[0], bounds[1])

            # 并行评估所有 trial
            trial_fitness = self._evaluate_population(trials, objective, shape)
            nfev += self.population_size

            # 选择更新（代模式）
            improved = trial_fitness < fitness
            population[improved] = trials[improved]
            fitness[improved] = trial_fitness[improved]

            # 更新全局最优
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_f:
                best_f = fitness[current_best_idx]
                best_x = population[current_best_idx].copy()
                best_idx = current_best_idx

            self.history.append(best_f)

            if self.verbose and gen % 10 == 0:
                logger.info(f"代数 {gen}: 最优适应度 = {best_f:.6e}")

        return OptimizationResult(
            x=best_x.reshape(shape),
            fun=best_f,
            nit=self.max_iter,
            nfev=nfev,
            success=True,
            message="差分进化优化完成",
            history=self.history
        )


class CMAESOptimizer(BaseHeuristicOptimizer):
    """
    CMA-ES 协方差矩阵自适应进化策略

    基于正态分布采样的进化算法，自适应调整协方差矩阵以适应目标函数地形。
    """

    def __init__(self,
                 sigma: float = 0.3,
                 **kwargs):
        """
        初始化 CMA-ES 优化器

        Args:
            sigma: 初始步长（分布标准差）
        """
        super().__init__(**kwargs)
        self.sigma_init = sigma

    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 **kwargs) -> OptimizationResult:
        """执行 CMA-ES 优化"""
        shape = x0.shape
        dim = x0.size

        mean = x0.copy().flatten()
        sigma = self.sigma_init
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        B, D = np.eye(dim), np.ones(dim)

        # 种群大小
        lam = self.population_size
        mu = lam // 2

        # 权重计算
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / (weights ** 2).sum()

        # 学习率
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) /
                  ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        self.history = []
        best_x = mean.copy()
        best_f = float('inf')
        nfev = 0

        for gen in range(self.max_iter):
            # 采样
            try:
                Z = np.random.randn(lam, dim)
                Y = Z @ (B * D).T
                X = mean + sigma * Y
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B, D = np.eye(dim), np.ones(dim)
                Z = np.random.randn(lam, dim)
                Y = Z @ (B * D).T
                X = mean + sigma * Y

            # 边界约束
            X = np.clip(X, bounds[0], bounds[1])

            # 评估（支持并行）
            fitness = self._evaluate_population(X, objective, shape)
            nfev += lam

            # 排序
            idx = np.argsort(fitness)
            X_sorted = X[idx]
            Y_sorted = Y[idx]
            Z_sorted = Z[idx]

            # 更新历史最优
            if fitness[idx[0]] < best_f:
                best_f = fitness[idx[0]]
                best_x = X_sorted[0].copy()

            self.history.append(best_f)

            if self.verbose and gen % 10 == 0:
                logger.info(f"迭代 {gen}: 最优适应度 = {best_f:.6e}")

            # 加权均值
            y_w = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            mean_new = mean + sigma * y_w
            mean_new = np.clip(mean_new, bounds[0], bounds[1])

            # 进化路径
            z_w = np.sum(weights[:, None] * Z_sorted[:mu], axis=0)
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * z_w
            hsig = (np.linalg.norm(p_s) /
                    np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN <
                    1.4 + 2 / (dim + 1))
            p_c = (1 - cc) * p_c + hsig * \
                np.sqrt(cc * (2 - cc) * mueff) * y_w

            # 更新协方差矩阵
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - hsig)
                                            * cc * (2 - cc) * C)
            for k in range(mu):
                C += cmu * weights[k] * np.outer(Y_sorted[k], Y_sorted[k])

            # 对称化并强制正定
            C = np.triu(C) + np.triu(C, 1).T

            # 特征分解
            try:
                D2, B = np.linalg.eigh(C)
                D2 = np.maximum(D2, 1e-14)
                D = np.sqrt(D2)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B, D = np.eye(dim), np.ones(dim)

            # 更新步长
            sigma *= np.exp((cs / damps) *
                            (np.linalg.norm(p_s) / chiN - 1))
            sigma = max(sigma, 1e-10)

            mean = mean_new

        return OptimizationResult(
            x=best_x.reshape(shape),
            fun=best_f,
            nit=self.max_iter,
            nfev=nfev,
            success=True,
            message="CMA-ES 优化完成",
            history=self.history
        )
