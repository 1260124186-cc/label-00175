# -*- coding: utf-8 -*-
"""
基础优化器模块：梯度下降、牛顿法、拟牛顿法(BFGS)

该模块封装了传统优化算法，适配掩模图案的像素级优化。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from scipy.optimize import minimize, line_search
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果"""
    x: np.ndarray  # 最优解
    fun: float  # 最优目标函数值
    nit: int  # 迭代次数
    nfev: int  # 函数评估次数
    success: bool  # 是否成功收敛
    message: str  # 状态信息
    history: List[float] = field(default_factory=list)  # 目标函数历史


class BaseOptimizer(ABC):
    """
    优化器基类
    
    定义优化器的通用接口和基本功能。
    """
    
    def __init__(self, 
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 verbose: bool = False):
        """
        初始化优化器
        
        Args:
            max_iter: 最大迭代次数
            tol: 收敛容差
            verbose: 是否输出详细信息
        """
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.history: List[float] = []
    
    @abstractmethod
    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 bounds: Optional[Tuple[float, float]] = None,
                 **kwargs) -> OptimizationResult:
        """
        执行优化
        
        Args:
            objective: 目标函数 f(x) -> float
            x0: 初始解
            gradient: 梯度函数 grad(x) -> ndarray，可选
            bounds: 变量边界 (min, max)
            **kwargs: 其他参数
            
        Returns:
            OptimizationResult对象
        """
        pass
    
    def _clip_to_bounds(self, x: np.ndarray, 
                        bounds: Optional[Tuple[float, float]]) -> np.ndarray:
        """将解裁剪到边界内"""
        if bounds is not None:
            return np.clip(x, bounds[0], bounds[1])
        return x
    
    def _check_convergence(self, 
                           f_old: float, 
                           f_new: float,
                           x_old: np.ndarray,
                           x_new: np.ndarray) -> bool:
        """检查是否收敛"""
        # 函数值变化
        f_change = abs(f_new - f_old) / (abs(f_old) + 1e-10)
        
        # 解的变化
        x_change = np.linalg.norm(x_new - x_old) / (np.linalg.norm(x_old) + 1e-10)
        
        return f_change < self.tol or x_change < self.tol


class GradientDescentOptimizer(BaseOptimizer):
    """
    梯度下降优化器
    
    支持固定学习率和自适应学习率（线搜索）。
    """
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 use_line_search: bool = False,
                 **kwargs):
        """
        初始化梯度下降优化器
        
        Args:
            learning_rate: 学习率
            momentum: 动量系数 (0-1)
            use_line_search: 是否使用线搜索确定步长
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_line_search = use_line_search
    
    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 bounds: Optional[Tuple[float, float]] = None,
                 **kwargs) -> OptimizationResult:
        """执行梯度下降优化"""
        x = x0.copy().flatten()
        velocity = np.zeros_like(x)
        
        self.history = []
        f_val = objective(x.reshape(x0.shape))
        self.history.append(f_val)
        
        nfev = 1
        success = False
        message = "达到最大迭代次数"
        
        for i in range(self.max_iter):
            # 计算梯度
            if gradient is not None:
                grad = gradient(x.reshape(x0.shape)).flatten()
            else:
                # 数值梯度
                grad = self._numerical_gradient(objective, x, x0.shape)
            nfev += 1
            
            # 确定步长
            if self.use_line_search:
                alpha = self._line_search(objective, x, -grad, x0.shape)
            else:
                alpha = self.learning_rate
            
            # 动量更新
            velocity = self.momentum * velocity - alpha * grad
            x_new = x + velocity
            
            # 边界约束
            x_new = self._clip_to_bounds(x_new, bounds)
            
            # 计算新的目标函数值
            f_new = objective(x_new.reshape(x0.shape))
            nfev += 1
            self.history.append(f_new)
            
            if self.verbose and i % 10 == 0:
                logger.info(f"迭代 {i}: f = {f_new:.6e}")
            
            # 检查收敛
            if self._check_convergence(f_val, f_new, x, x_new):
                success = True
                message = f"在第{i+1}次迭代收敛"
                x = x_new
                f_val = f_new
                break
            
            x = x_new
            f_val = f_new
        
        return OptimizationResult(
            x=x.reshape(x0.shape),
            fun=f_val,
            nit=i + 1,
            nfev=nfev,
            success=success,
            message=message,
            history=self.history
        )
    
    def _numerical_gradient(self, 
                            objective: Callable,
                            x: np.ndarray,
                            shape: Tuple) -> np.ndarray:
        """计算数值梯度"""
        eps = 1e-7
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            grad[i] = (objective(x_plus.reshape(shape)) - 
                      objective(x_minus.reshape(shape))) / (2 * eps)
        
        return grad
    
    def _line_search(self,
                     objective: Callable,
                     x: np.ndarray,
                     direction: np.ndarray,
                     shape: Tuple) -> float:
        """简单的回溯线搜索"""
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        
        f0 = objective(x.reshape(shape))
        
        for _ in range(20):
            x_new = x + alpha * direction
            f_new = objective(x_new.reshape(shape))
            
            if f_new < f0 - c * alpha * np.dot(direction, direction):
                return alpha
            
            alpha *= rho
        
        return alpha


class NewtonOptimizer(BaseOptimizer):
    """
    牛顿法优化器
    
    使用二阶导数信息加速收敛。
    """
    
    def __init__(self,
                 regularization: float = 1e-6,
                 **kwargs):
        """
        初始化牛顿法优化器
        
        Args:
            regularization: Hessian正则化系数（确保正定）
        """
        super().__init__(**kwargs)
        self.regularization = regularization
    
    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 bounds: Optional[Tuple[float, float]] = None,
                 hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 **kwargs) -> OptimizationResult:
        """执行牛顿法优化"""
        x = x0.copy().flatten()
        n = len(x)
        
        self.history = []
        f_val = objective(x.reshape(x0.shape))
        self.history.append(f_val)
        
        nfev = 1
        success = False
        message = "达到最大迭代次数"
        
        for i in range(self.max_iter):
            # 计算梯度
            if gradient is not None:
                grad = gradient(x.reshape(x0.shape)).flatten()
            else:
                grad = self._numerical_gradient(objective, x, x0.shape)
            
            # 计算Hessian
            if hessian is not None:
                H = hessian(x.reshape(x0.shape))
            else:
                H = self._numerical_hessian(objective, x, x0.shape)
            
            # 正则化Hessian
            H_reg = H + self.regularization * np.eye(n)
            
            # 求解牛顿方向
            try:
                direction = np.linalg.solve(H_reg, -grad)
            except np.linalg.LinAlgError:
                direction = -grad  # 退化为梯度下降
            
            # 线搜索
            alpha = self._backtracking_line_search(
                objective, x, direction, grad, x0.shape
            )
            
            x_new = x + alpha * direction
            x_new = self._clip_to_bounds(x_new, bounds)
            
            f_new = objective(x_new.reshape(x0.shape))
            nfev += 1
            self.history.append(f_new)
            
            if self.verbose and i % 5 == 0:
                logger.info(f"迭代 {i}: f = {f_new:.6e}")
            
            if self._check_convergence(f_val, f_new, x, x_new):
                success = True
                message = f"在第{i+1}次迭代收敛"
                x = x_new
                f_val = f_new
                break
            
            x = x_new
            f_val = f_new
        
        return OptimizationResult(
            x=x.reshape(x0.shape),
            fun=f_val,
            nit=i + 1,
            nfev=nfev,
            success=success,
            message=message,
            history=self.history
        )
    
    def _numerical_gradient(self, objective, x, shape):
        """数值梯度"""
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_p, x_m = x.copy(), x.copy()
            x_p[i] += eps
            x_m[i] -= eps
            grad[i] = (objective(x_p.reshape(shape)) - 
                      objective(x_m.reshape(shape))) / (2 * eps)
        return grad
    
    def _numerical_hessian(self, objective, x, shape):
        """数值Hessian"""
        eps = 1e-5
        n = len(x)
        H = np.zeros((n, n))
        f0 = objective(x.reshape(shape))
        
        for i in range(n):
            x_pi = x.copy()
            x_pi[i] += eps
            fi = objective(x_pi.reshape(shape))
            
            for j in range(i, n):
                x_pj = x.copy()
                x_pj[j] += eps
                fj = objective(x_pj.reshape(shape))
                
                x_pij = x.copy()
                x_pij[i] += eps
                x_pij[j] += eps
                fij = objective(x_pij.reshape(shape))
                
                H[i, j] = (fij - fi - fj + f0) / (eps * eps)
                H[j, i] = H[i, j]
        
        return H
    
    def _backtracking_line_search(self, objective, x, direction, grad, shape):
        """回溯线搜索"""
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        f0 = objective(x.reshape(shape))
        slope = np.dot(grad, direction)
        
        for _ in range(20):
            f_new = objective((x + alpha * direction).reshape(shape))
            if f_new <= f0 + c * alpha * slope:
                return alpha
            alpha *= rho
        
        return alpha


class BFGSOptimizer(BaseOptimizer):
    """
    BFGS拟牛顿法优化器
    
    使用scipy.optimize.minimize的BFGS实现。
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def optimize(self,
                 objective: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 bounds: Optional[Tuple[float, float]] = None,
                 **kwargs) -> OptimizationResult:
        """执行BFGS优化"""
        shape = x0.shape
        x0_flat = x0.flatten()
        
        self.history = []
        
        def obj_wrapper(x):
            val = objective(x.reshape(shape))
            self.history.append(val)
            return val
        
        def grad_wrapper(x):
            if gradient is not None:
                return gradient(x.reshape(shape)).flatten()
            return None
        
        # 设置边界
        if bounds is not None:
            scipy_bounds = [(bounds[0], bounds[1])] * len(x0_flat)
            method = 'L-BFGS-B'
        else:
            scipy_bounds = None
            method = 'BFGS'
        
        result = minimize(
            obj_wrapper,
            x0_flat,
            method=method,
            jac=grad_wrapper if gradient else None,
            bounds=scipy_bounds,
            options={
                'maxiter': self.max_iter,
                'gtol': self.tol,
                'disp': self.verbose
            }
        )
        
        return OptimizationResult(
            x=result.x.reshape(shape),
            fun=result.fun,
            nit=result.nit if hasattr(result, 'nit') else len(self.history),
            nfev=result.nfev,
            success=result.success,
            message=result.message,
            history=self.history
        )
