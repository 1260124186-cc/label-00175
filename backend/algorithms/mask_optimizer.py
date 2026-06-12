# -*- coding: utf-8 -*-
"""
掩模优化模块：完整的掩模图案优化流程

该模块实现"掩模图案→光学成像→误差计算→参数更新"的迭代优化流程，
支持早停、学习率调度等功能。
"""

import numpy as np
from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field
import logging
import time

from core.imaging import OpticalSystem, PartialCoherentImaging, simulate_wafer_image
from core.metrics import mse, mae, ssim, evaluate_all, MetricsResult
from algorithms.optimizer import (
    BaseOptimizer, GradientDescentOptimizer, BFGSOptimizer, 
    NewtonOptimizer, OptimizationResult
)
from algorithms.advanced_optimizer import (
    BaseHeuristicOptimizer, GeneticAlgorithmOptimizer, ParticleSwarmOptimizer
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """
    优化配置类
    
    Attributes:
        optimizer_type: 优化器类型
        max_iter: 最大迭代次数
        learning_rate: 学习率（梯度下降）
        tol: 收敛容差
        early_stop_patience: 早停耐心值
        lr_scheduler: 学习率调度器类型
        metric: 优化目标指标
        bounds: 掩模值边界
        verbose: 是否输出详细信息
        random_seed: 随机种子（用于结果复现）
    """
    optimizer_type: str = 'gradient_descent'
    max_iter: int = 100
    learning_rate: float = 0.01
    tol: float = 1e-6
    early_stop_patience: int = 10
    lr_scheduler: Optional[str] = None  # 'step', 'exponential', 'cosine'
    lr_decay: float = 0.95
    lr_step_size: int = 20
    metric: str = 'mse'  # 'mse', 'mae', 'ssim'
    bounds: tuple = (0.0, 1.0)
    verbose: bool = True
    random_seed: Optional[int] = None  # 随机种子，用于结果复现
    
    # 启发式算法参数
    population_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, 
                 initial_lr: float,
                 scheduler_type: str,
                 decay: float = 0.95,
                 step_size: int = 20,
                 min_lr: float = 1e-6):
        """
        初始化学习率调度器
        
        Args:
            initial_lr: 初始学习率
            scheduler_type: 调度器类型
            decay: 衰减率
            step_size: 步长（用于step调度）
            min_lr: 最小学习率
        """
        self.initial_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.decay = decay
        self.step_size = step_size
        self.min_lr = min_lr
        self.current_lr = initial_lr
    
    def step(self, epoch: int) -> float:
        """
        更新学习率
        
        Args:
            epoch: 当前迭代次数
            
        Returns:
            更新后的学习率
        """
        if self.scheduler_type == 'step':
            # 阶梯衰减
            self.current_lr = self.initial_lr * (self.decay ** (epoch // self.step_size))
        
        elif self.scheduler_type == 'exponential':
            # 指数衰减
            self.current_lr = self.initial_lr * (self.decay ** epoch)
        
        elif self.scheduler_type == 'cosine':
            # 余弦退火
            self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                             (1 + np.cos(np.pi * epoch / self.step_size))
        
        self.current_lr = max(self.current_lr, self.min_lr)
        return self.current_lr


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        """
        初始化早停
        
        Args:
            patience: 耐心值（连续多少次无改善则停止）
            min_delta: 最小改善量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, loss: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            loss: 当前损失值
            
        Returns:
            是否应该停止
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


@dataclass
class MaskOptimizationResult:
    """掩模优化结果"""
    optimized_mask: np.ndarray
    initial_mask: np.ndarray
    target_image: np.ndarray
    final_wafer_image: np.ndarray
    initial_wafer_image: np.ndarray
    final_metrics: MetricsResult
    initial_metrics: MetricsResult
    loss_history: List[float]
    total_iterations: int
    total_time: float
    converged: bool
    message: str


class MaskOptimizer:
    """
    掩模优化器
    
    实现完整的掩模图案优化流程。
    """
    
    def __init__(self, 
                 optical_system: Optional[OpticalSystem] = None,
                 config: Optional[OptimizationConfig] = None):
        """
        初始化掩模优化器
        
        Args:
            optical_system: 光学系统参数
            config: 优化配置
        """
        self.optical_system = optical_system or OpticalSystem()
        self.config = config or OptimizationConfig()
        
        self._imaging_model: Optional[PartialCoherentImaging] = None
        self._target_image: Optional[np.ndarray] = None
        self._optimizer: Optional[Union[BaseOptimizer, BaseHeuristicOptimizer]] = None
        self._lr_scheduler: Optional[LearningRateScheduler] = None
        self._early_stopping: Optional[EarlyStopping] = None
    
    def _setup_imaging_model(self, image_size: tuple):
        """设置成像模型"""
        self._imaging_model = PartialCoherentImaging(
            self.optical_system, image_size
        )
    
    def _setup_optimizer(self):
        """设置优化器"""
        opt_type = self.config.optimizer_type.lower()
        seed = self.config.random_seed  # 从配置获取随机种子
        
        if opt_type == 'gradient_descent':
            self._optimizer = GradientDescentOptimizer(
                learning_rate=self.config.learning_rate,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                verbose=self.config.verbose
            )
        elif opt_type == 'bfgs':
            self._optimizer = BFGSOptimizer(
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                verbose=self.config.verbose
            )
        elif opt_type == 'newton':
            self._optimizer = NewtonOptimizer(
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                verbose=self.config.verbose
            )
        elif opt_type == 'genetic':
            self._optimizer = GeneticAlgorithmOptimizer(
                population_size=self.config.population_size,
                max_iter=self.config.max_iter,
                crossover_rate=self.config.crossover_rate,
                mutation_rate=self.config.mutation_rate,
                verbose=self.config.verbose,
                seed=seed  # 传递随机种子
            )
        elif opt_type == 'pso':
            self._optimizer = ParticleSwarmOptimizer(
                population_size=self.config.population_size,
                max_iter=self.config.max_iter,
                verbose=self.config.verbose,
                seed=seed  # 传递随机种子
            )
        else:
            raise ValueError(f"未知的优化器类型: {opt_type}")
    
    def _setup_lr_scheduler(self):
        """设置学习率调度器"""
        if self.config.lr_scheduler:
            self._lr_scheduler = LearningRateScheduler(
                initial_lr=self.config.learning_rate,
                scheduler_type=self.config.lr_scheduler,
                decay=self.config.lr_decay,
                step_size=self.config.lr_step_size
            )
    
    def _setup_early_stopping(self):
        """设置早停"""
        self._early_stopping = EarlyStopping(
            patience=self.config.early_stop_patience
        )
    
    def _compute_loss(self, mask: np.ndarray) -> float:
        """
        计算损失函数
        
        Args:
            mask: 掩模图案
            
        Returns:
            损失值
        """
        # 计算晶圆成像
        wafer_image = self._imaging_model.compute_aerial_image(mask)
        
        # 计算误差
        metric = self.config.metric.lower()
        if metric == 'mse':
            return mse(wafer_image, self._target_image)
        elif metric == 'mae':
            return mae(wafer_image, self._target_image)
        elif metric == 'ssim':
            return 1.0 - ssim(wafer_image, self._target_image)  # 转换为最小化问题
        else:
            raise ValueError(f"未知的评估指标: {metric}")
    
    def _compute_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算损失函数对掩模的梯度
        
        Args:
            mask: 掩模图案
            
        Returns:
            梯度数组
        """
        wafer_image = self._imaging_model.compute_aerial_image(mask)
        
        # 误差对成像的梯度
        if self.config.metric.lower() == 'mse':
            error_grad = 2 * (wafer_image - self._target_image) / mask.size
        elif self.config.metric.lower() == 'mae':
            error_grad = np.sign(wafer_image - self._target_image) / mask.size
        else:
            # 对于SSIM，使用数值梯度
            return self._numerical_gradient(mask)
        
        # 成像对掩模的梯度（链式法则）
        imaging_grad = self._imaging_model.compute_image_gradient(mask)
        
        # 总梯度
        gradient = error_grad * imaging_grad
        
        return gradient
    
    def _numerical_gradient(self, mask: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """数值梯度计算"""
        gradient = np.zeros_like(mask)
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask_plus = mask.copy()
                mask_plus[i, j] += eps
                mask_minus = mask.copy()
                mask_minus[i, j] -= eps
                
                gradient[i, j] = (self._compute_loss(mask_plus) - 
                                 self._compute_loss(mask_minus)) / (2 * eps)
        
        return gradient
    
    def optimize(self,
                 initial_mask: np.ndarray,
                 target_image: np.ndarray,
                 callback: Optional[Callable[[int, np.ndarray, float], None]] = None
                 ) -> MaskOptimizationResult:
        """
        执行掩模优化
        
        Args:
            initial_mask: 初始掩模图案
            target_image: 目标图像
            callback: 回调函数 callback(iteration, current_mask, current_loss)
            
        Returns:
            MaskOptimizationResult对象
        """
        start_time = time.time()
        
        # 保存目标图像
        self._target_image = target_image.astype(np.float64)
        
        # 设置各组件
        self._setup_imaging_model(initial_mask.shape)
        self._setup_optimizer()
        self._setup_lr_scheduler()
        self._setup_early_stopping()
        
        # 计算初始状态
        initial_wafer = self._imaging_model.compute_aerial_image(initial_mask)
        initial_metrics = evaluate_all(initial_wafer, target_image)
        
        logger.info(f"开始掩模优化，初始MSE: {initial_metrics.mse:.6e}")
        
        # 执行优化
        if isinstance(self._optimizer, BaseHeuristicOptimizer):
            # 启发式算法
            result = self._optimizer.optimize(
                objective=self._compute_loss,
                x0=initial_mask,
                bounds=self.config.bounds
            )
        else:
            # 传统优化算法
            result = self._optimizer.optimize(
                objective=self._compute_loss,
                x0=initial_mask,
                gradient=self._compute_gradient,
                bounds=self.config.bounds
            )
        
        # 计算最终状态
        optimized_mask = result.x
        final_wafer = self._imaging_model.compute_aerial_image(optimized_mask)
        final_metrics = evaluate_all(final_wafer, target_image)
        
        total_time = time.time() - start_time
        
        logger.info(f"优化完成，最终MSE: {final_metrics.mse:.6e}，"
                   f"耗时: {total_time:.2f}秒")
        
        return MaskOptimizationResult(
            optimized_mask=optimized_mask,
            initial_mask=initial_mask,
            target_image=target_image,
            final_wafer_image=final_wafer,
            initial_wafer_image=initial_wafer,
            final_metrics=final_metrics,
            initial_metrics=initial_metrics,
            loss_history=result.history,
            total_iterations=result.nit,
            total_time=total_time,
            converged=result.success,
            message=result.message
        )
    
    def optimize_with_custom_objective(self,
                                       initial_mask: np.ndarray,
                                       objective_func: Callable[[np.ndarray], float],
                                       gradient_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
                                       ) -> OptimizationResult:
        """
        使用自定义目标函数进行优化
        
        Args:
            initial_mask: 初始掩模
            objective_func: 自定义目标函数
            gradient_func: 自定义梯度函数（可选）
            
        Returns:
            OptimizationResult对象
        """
        self._setup_optimizer()
        
        if isinstance(self._optimizer, BaseHeuristicOptimizer):
            return self._optimizer.optimize(
                objective=objective_func,
                x0=initial_mask,
                bounds=self.config.bounds
            )
        else:
            return self._optimizer.optimize(
                objective=objective_func,
                x0=initial_mask,
                gradient=gradient_func,
                bounds=self.config.bounds
            )
