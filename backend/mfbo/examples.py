# -*- coding: utf-8 -*-
"""
多保真度贝叶斯优化 (MFBO) 使用示例

包含：
1. 基础使用：经典测试函数 (Branin-Hoo, Hartmann) 的 MFBO 优化
2. 保真度设计：如何为不同保真度设计评估函数
3. 算法对比：MFBO vs 单保真度贝叶斯优化 vs 随机搜索
4. 策略切换：不同保真度选择策略的效果对比
5. 掩模优化集成：与计算光刻掩模优化结合的用法

适合博士课题中的算法对比研究，所有示例均可复现。
"""

import numpy as np
from typing import Tuple, Dict, Any
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 导入 MFBO 模块
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from mfbo import (
    FidelityLevel,
    SearchSpace,
    MFBOConfig,
    MFBOResult,
    Observation,
    FidelityCost,
    KernelType,
    AcquisitionFunctionType,
    FidelitySelectionStrategy,
    MultiFidelityBayesianOptimizer,
    MultiFidelityEvaluator,
)

# ===========================================================================
# 示例 1: Branin-Hoo 函数（经典贝叶斯优化测试函数）
# ===========================================================================

def branin_hoo(x: np.ndarray,
               fidelity: FidelityLevel,
               noise_scale: float = 0.0,
               bias_low: float = 0.5,
               bias_medium: float = 0.2) -> float:
    """
    多保真度 Branin-Hoo 函数

    全局最小值 f(x*) ≈ 0.397887
        x* = (-π, 12.275), (π, 2.275), (9.42478, 2.475)

    保真度设计:
    - HIGH: 精确 Branin-Hoo（无噪声、无偏置）
    - MEDIUM: 全局偏移 + 低噪声
    - LOW: 更大偏移 + 中噪声 + 平滑（降采样）

    Args:
        x: (2,) [x1, x2]，x1 ∈ [-5, 10], x2 ∈ [0, 15]
        fidelity: 保真度层级
    """
    x1, x2 = x[0], x[1]

    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)

    # 基础函数（HIGH 保真度）
    y_high = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    rng = np.random.default_rng(int(abs(hash((x1, x2, fidelity.value))) % 1_000_000))

    if fidelity == FidelityLevel.HIGH:
        noise = rng.normal(0, noise_scale)
        return float(y_high + noise)

    elif fidelity == FidelityLevel.MEDIUM:
        # MEDIUM: 平滑 + 小偏移 + 小噪声
        # 用 sin 扰动模拟近似仿真
        smooth = 0.3 * np.sin(0.5 * x1) * np.cos(0.3 * x2)
        noise = rng.normal(0, 0.05)
        return float(y_high + bias_medium + smooth + noise)

    else:  # LOW
        # LOW: 更大偏移 + 平滑 + 中等噪声
        x1_low = x1 + 0.5 * np.sin(0.3 * x1)  # 形变
        x2_low = x2 + 0.3 * np.cos(0.2 * x2)
        y_low = a * (x2_low - b * x1_low ** 2 + c * x1_low - r) ** 2
        y_low += s * (1 - t) * np.cos(x1_low) + s
        noise = rng.normal(0, 0.2)
        return float(y_low + bias_low + noise)


def example_branin_hoo() -> MFBOResult:
    """
    示例1: 用 MFBO 优化 Branin-Hoo 函数

    展示完整流程：定义搜索空间 -> 配置 -> 定义评估器 -> 运行优化
    """
    print("\n" + "=" * 70)
    print("示例 1: 多保真度 Branin-Hoo 函数优化")
    print("=" * 70)

    # 1. 定义搜索空间
    search_space = SearchSpace(
        bounds=[(-5.0, 10.0), (0.0, 15.0)],
        names=["x1", "x2"],
    )

    # 2. 配置成本（1:10:100 比率，典型仿真场景）
    cost_config = FidelityCost(
        costs={
            FidelityLevel.LOW: 0.01,
            FidelityLevel.MEDIUM: 0.1,
            FidelityLevel.HIGH: 1.0,
        },
        absolute_times={
            FidelityLevel.LOW: 0.05,
            FidelityLevel.MEDIUM: 0.5,
            FidelityLevel.HIGH: 5.0,
        }
    )

    # 3. MFBO 配置
    config = MFBOConfig(
        n_init_low=8,
        n_init_medium=4,
        n_init_high=2,
        max_iterations=50,
        max_budget=15.0,  # 相当于 15 次 HIGH 保真度评估
        target_fidelity=FidelityLevel.HIGH,
        kernel_type=KernelType.AR1,
        acquisition_type=AcquisitionFunctionType.EIV,
        fidelity_strategy=FidelitySelectionStrategy.COST_AWARE,
        ucb_beta=2.0,
        optimizer_restarts=3,
        acq_n_candidates=1000,
        early_stop_patience=20,
        random_seed=42,
        cost_config=cost_config,
    )

    # 4. 创建优化器，运行
    mfbo = MultiFidelityBayesianOptimizer(config, search_space)
    result = mfbo.minimize(branin_hoo)

    # 5. 分析结果
    known_min = 0.397887
    error = abs(result.best_y - known_min)
    print(f"\n  已知全局最小值: {known_min:.6f}")
    print(f"  MFBO 找到最小值: {result.best_y:.6f}")
    print(f"  绝对误差:       {error:.6f}")
    print(f"  最优解位置:     x1={result.best_x[0]:.4f}, x2={result.best_x[1]:.4f}")
    print(f"  保真度分布:     {mfbo.get_fidelity_statistics()['counts']}")

    return result


# ===========================================================================
# 示例 2: 多保真度设计模式 - 自定义评估器
# ===========================================================================

class MultiFidelityObjective:
    """
    多保真度目标函数基类（面向对象模式）

    适合封装复杂仿真器，例如：
    - LOW: 神经网络代理模型 (surrogate/*.py)
    - MEDIUM: 部分相干成像近似 (core/imaging.py)
    - HIGH: 严格电磁仿真 (core/rigorous_sim.py)
    """

    def __init__(self, name: str = "custom"):
        self.name = name
        self.n_calls: Dict[FidelityLevel, int] = {
            FidelityLevel.LOW: 0,
            FidelityLevel.MEDIUM: 0,
            FidelityLevel.HIGH: 0,
        }

    def __call__(self, x: np.ndarray, fidelity: FidelityLevel) -> Tuple[float, Dict[str, Any]]:
        """可调用接口，返回 (y, metadata)"""
        self.n_calls[fidelity] += 1

        if fidelity == FidelityLevel.HIGH:
            y = self._evaluate_high(x)
        elif fidelity == FidelityLevel.MEDIUM:
            y = self._evaluate_medium(x)
        else:
            y = self._evaluate_low(x)

        metadata = {
            "level": fidelity.value,
            "call_count": self.n_calls[fidelity],
        }
        return y, metadata

    def _evaluate_low(self, x: np.ndarray) -> float:
        """低保真度评估（快速代理）"""
        raise NotImplementedError

    def _evaluate_medium(self, x: np.ndarray) -> float:
        """中保真度评估（近似仿真）"""
        raise NotImplementedError

    def _evaluate_high(self, x: np.ndarray) -> float:
        """高保真度评估（精确仿真）"""
        raise NotImplementedError


class Hartmann6MultiFidelity(MultiFidelityObjective):
    """
    6维 Hartmann 函数（经典高维贝叶斯优化测试函数）

    全局最小值 f(x*) ≈ -3.32237
    """

    def __init__(self):
        super().__init__("hartmann6")
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array([
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ])
        self.P = 1e-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1415, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ])

    def _base(self, x: np.ndarray) -> float:
        x = np.atleast_2d(x).reshape(-1, 6)
        n = x.shape[0]
        outer = np.zeros(n)
        for i in range(4):
            inner = np.zeros(n)
            for j in range(6):
                inner += self.A[i, j] * (x[:, j] - self.P[i, j]) ** 2
            outer += self.alpha[i] * np.exp(-inner)
        return float(-outer[0]) if n == 1 else -outer

    def _evaluate_low(self, x: np.ndarray) -> float:
        """低保真度：用降维近似 + 偏移"""
        # 只取前3个主维度 + 随机扰动
        x_low = x.copy()
        x_low[3:] = 0.5  # 冻结后三维
        rng = np.random.default_rng(int(abs(hash(tuple(x_low))) % 1_000_000))
        noise = rng.normal(0, 0.1)
        return self._base(x_low) + 0.8 + noise

    def _evaluate_medium(self, x: np.ndarray) -> float:
        """中保真度：加入可分离近似"""
        rng = np.random.default_rng(int(abs(hash(tuple(x))) % 1_000_000))
        noise = rng.normal(0, 0.03)
        return self._base(x) + 0.2 + noise

    def _evaluate_high(self, x: np.ndarray) -> float:
        """高保真度：精确值"""
        return self._base(x)


def example_hartmann6() -> MFBOResult:
    """示例2: 6维 Hartmann 高维优化"""
    print("\n" + "=" * 70)
    print("示例 2: 6维 Hartmann 函数优化（高维场景）")
    print("=" * 70)

    search_space = SearchSpace(
        bounds=[(0.0, 1.0)] * 6,
        names=[f"x{i}" for i in range(6)],
    )

    config = MFBOConfig(
        n_init_low=10,
        n_init_medium=5,
        n_init_high=2,
        max_iterations=40,
        max_budget=12.0,
        target_fidelity=FidelityLevel.HIGH,
        kernel_type=KernelType.AR1,
        acquisition_type=AcquisitionFunctionType.EIV,
        fidelity_strategy=FidelitySelectionStrategy.COST_AWARE,
        random_seed=123,
    )

    objective = Hartmann6MultiFidelity()
    mfbo = MultiFidelityBayesianOptimizer(config, search_space)
    result = mfbo.minimize(objective)

    known_min = -3.32237
    error = abs(result.best_y - known_min)
    print(f"\n  已知最小值: {known_min:.6f}")
    print(f"  MFBO结果:   {result.best_y:.6f}")
    print(f"  绝对误差:   {error:.6f}")
    print(f"  各保真度调用: {objective.n_calls}")

    return result


# ===========================================================================
# 示例 3: 算法对比实验（博士论文场景）
# ===========================================================================

def compare_methods() -> Dict[str, Dict[str, Any]]:
    """
    示例3: 算法对比实验

    对比方法：
    1. MFBO-AR1-EIV (默认)
    2. MFBO-CoKriging-UCB
    3. 单保真度贝叶斯优化 (只用HIGH)
    4. 纯随机搜索

    指标：
    - 最终最优目标值
    - 收敛速度（达到阈值所需预算）
    - 每保真度样本效率
    """
    print("\n" + "=" * 70)
    print("示例 3: 算法对比实验（博士课题研究场景）")
    print("=" * 70)

    search_space = SearchSpace(bounds=[(-5.0, 10.0), (0.0, 15.0)])
    budget = 10.0
    known_min = 0.397887

    methods = [
        ("MFBO-AR1-EIV", MFBOConfig(
            n_init_low=6, n_init_medium=3, n_init_high=1,
            max_iterations=200, max_budget=budget,
            target_fidelity=FidelityLevel.HIGH,
            kernel_type=KernelType.AR1,
            acquisition_type=AcquisitionFunctionType.EIV,
            fidelity_strategy=FidelitySelectionStrategy.COST_AWARE,
            random_seed=1,
            optimizer_restarts=2,
            acq_n_candidates=500,
        )),
        ("MFBO-LCM-KG", MFBOConfig(
            n_init_low=6, n_init_medium=3, n_init_high=1,
            max_iterations=200, max_budget=budget,
            target_fidelity=FidelityLevel.HIGH,
            kernel_type=KernelType.LinearCoregional,
            acquisition_type=AcquisitionFunctionType.KG,
            fidelity_strategy=FidelitySelectionStrategy.INFORMATION_GAIN,
            random_seed=2,
            optimizer_restarts=2,
            acq_n_candidates=500,
        )),
        ("Single-Fidelity (High only)", MFBOConfig(
            n_init_low=0, n_init_medium=0, n_init_high=3,  # 初始：只有HIGH
            max_iterations=200, max_budget=budget,
            target_fidelity=FidelityLevel.HIGH,
            kernel_type=KernelType.AR1,
            acquisition_type=AcquisitionFunctionType.EI,  # 单保真：标准EI
            fidelity_strategy=FidelitySelectionStrategy.SCHEDULED,
            random_seed=3,
            optimizer_restarts=2,
            acq_n_candidates=500,
        )),
    ]

    results: Dict[str, Dict[str, Any]] = {}

    for name, config in methods:
        print(f"\n  --- 运行 {name} ---")
        mfbo = MultiFidelityBayesianOptimizer(config, search_space)

        # 单保真度模式：修改evaluator，强制只用HIGH
        if "Single" in name:
            def evaluator_sf(x, fidelity, orig=branin_hoo):
                # 忽略传入的fidelity，总是用HIGH
                return orig(x, FidelityLevel.HIGH)
            evaluator = evaluator_sf
        else:
            evaluator = branin_hoo

        result = mfbo.minimize(evaluator)

        error = abs(result.best_y - known_min)
        fid_stats = mfbo.get_fidelity_statistics()

        # 计算收敛速度：达到 error<0.01 所需预算
        iters, budgets, bests = result.get_convergence_data()
        threshold = known_min + 0.05
        idx_conv = np.where(bests <= threshold)[0]
        budget_to_converge = budgets[idx_conv[0]] if len(idx_conv) > 0 else budget

        results[name] = {
            "best_y": result.best_y,
            "absolute_error": error,
            "budget_used": result.total_budget_used,
            "n_iterations": result.n_iterations,
            "fidelity_counts": fid_stats["counts"],
            "budget_to_converge": budget_to_converge,
            "total_time": result.total_time,
        }

        print(f"    最优值: {result.best_y:.6f} | 误差: {error:.6f} | "
              f"预算: {result.total_budget_used:.2f}")
        print(f"    保真度计数: {fid_stats['counts']} | 收敛预算: {budget_to_converge:.2f}")

    # 打印对比表
    print("\n" + "-" * 70)
    print("算法对比汇总表")
    print("-" * 70)
    header = f"{'方法':<30s} {'最优f(x)':>10s} {'误差':>10s} {'收敛预算':>10s}"
    print(header)
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<30s} {r['best_y']:>10.4f} {r['absolute_error']:>10.4f} "
              f"{r['budget_to_converge']:>10.2f}")
    print("-" * 70)

    return results


# ===========================================================================
# 示例 4: 策略对比（5种保真度选择策略）
# ===========================================================================

def compare_fidelity_strategies() -> Dict[str, MFBOResult]:
    """
    示例4: 保真度选择策略对比

    对比5种策略在相同预算下的性能差异
    """
    print("\n" + "=" * 70)
    print("示例 4: 保真度选择策略对比")
    print("=" * 70)

    search_space = SearchSpace(bounds=[(-5.0, 10.0), (0.0, 15.0)])
    budget = 8.0

    strategies = [
        ("Cost-Aware (EIV)", FidelitySelectionStrategy.COST_AWARE),
        ("Information-Gain", FidelitySelectionStrategy.INFORMATION_GAIN),
        ("Budget-Proportional", FidelitySelectionStrategy.BUDGET_PROPORTIONAL),
        ("Scheduled", FidelitySelectionStrategy.SCHEDULED),
        ("Adaptive-Threshold", FidelitySelectionStrategy.ADAPTIVE_THRESHOLD),
    ]

    results: Dict[str, MFBOResult] = {}

    for name, strategy in strategies:
        print(f"\n  运行策略: {name}")
        config = MFBOConfig(
            n_init_low=5, n_init_medium=2, n_init_high=1,
            max_iterations=100, max_budget=budget,
            target_fidelity=FidelityLevel.HIGH,
            kernel_type=KernelType.AR1,
            acquisition_type=AcquisitionFunctionType.EIV,
            fidelity_strategy=strategy,
            random_seed=10,
            optimizer_restarts=2,
            acq_n_candidates=400,
        )
        mfbo = MultiFidelityBayesianOptimizer(config, search_space)
        result = mfbo.minimize(branin_hoo)
        results[name] = result

        print(f"    结果: f*={result.best_y:.4f}, "
              f"分布={mfbo.get_fidelity_statistics()['counts']}")

    return results


# ===========================================================================
# 示例 5: 与掩模优化集成（假想接口示例）
# ===========================================================================

def mask_optimization_scenario() -> None:
    """
    示例5: 与计算光刻掩模优化集成的概念演示

    展示如何将 MFBO 与现有的掩模优化模块结合：
    - LOW:  使用 surrogate/imaging.py 神经网络代理模型
    - MEDIUM: 使用 core/imaging.py 部分相干成像（近似）
    - HIGH:  使用 core/rigorous_sim.py RCWA / FDTD 严格电磁仿真
    """
    print("\n" + "=" * 70)
    print("示例 5: 与计算光刻掩模优化集成（概念演示）")
    print("=" * 70)

    print("""
    典型掩模优化场景下的保真度设计：

    保真度层级     | 仿真模型                | 典型耗时    | 相对成本
    ----------------|-------------------------|------------|---------
    LOW (低保真)    | 神经网络代理 (UNet)      | ~50 ms     | 0.01
    MEDIUM (中保真) | 部分相干成像(Abbe模型)  | ~0.5 s     | 0.1
    HIGH (高保真)   | RCWA严格电磁仿真         | ~5 s       | 1.0

    用法：
        def mask_mf_evaluator(params, fidelity):
            # params: 光源参数、掩模偏置、剂量等超参数
            if fidelity == FidelityLevel.LOW:
                return surrogate_model.predict(params)
            elif fidelity == FidelityLevel.MEDIUM:
                return imaging.simulate_wafer_image(params, fast_mode=True)
            else:
                return rigorous_sim.rcwa(params, high_res=True)

        search_space = SearchSpace(bounds=[...])
        mfbo = MultiFidelityBayesianOptimizer(config, search_space)
        result = mfbo.minimize(mask_mf_evaluator)
    """)


# ===========================================================================
# 主入口：运行所有示例
# ===========================================================================

def run_all_examples():
    """运行所有示例"""
    print("=" * 70)
    print("多保真度贝叶斯优化 (MFBO) 模块示例集")
    print("=" * 70)
    print()

    # 示例1: Branin-Hoo
    r1 = example_branin_hoo()

    # 示例2: Hartmann6
    r2 = example_hartmann6()

    # 示例3: 算法对比（默认关闭，运行时间较长）
    # compare_methods()

    # 示例4: 策略对比（默认关闭）
    # compare_fidelity_strategies()

    # 示例5: 掩模场景
    mask_optimization_scenario()

    print("\n" + "=" * 70)
    print("示例运行完成！")
    print("=" * 70)

    return r1, r2


if __name__ == "__main__":
    run_all_examples()
