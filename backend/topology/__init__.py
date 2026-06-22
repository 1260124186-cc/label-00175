# -*- coding: utf-8 -*-
"""
水平集拓扑优化模块 (backend.topology)

用水平集或 SIMP 方法在连续域上优化掩模边界演化，
替代现有像素级 [0,1] 参数化，天然保证边界光滑性与
最小特征尺寸，为算法研究提供与梯度投影 ILT 不同的
方法论路径。

核心组件：
  - LevelSetFunction: 水平集函数表示（SDF 初始化、重初始化、
    零水平集提取、曲率/梯度计算）
  - SIMPMaterialModel: Heaviside 投影 + SIMP 插值 +
    灵敏度过滤，实现 φ→ρ→E 的可微映射
  - HamiltonJacobiEvolver: 迎风格式水平集演化器，
    支持 CFL 自适应步长与周期性重初始化
  - ShapeVelocityCalculator: 形状灵敏度→法向速度的转换
  - TopologyConstraints: 曲率平滑 + 最小特征尺寸 + 周长/面积约束
  - LevelSetTopologyOptimizer: 完整优化流程封装
  - TopologyOptConfig / TopologyOptimizationResult: 配置与结果

方法对比：
  ┌──────────────┬─────────────────────┬──────────────────────┐
  │              │ ILT (梯度投影)       │ Level Set (本模块)    │
  ├──────────────┼─────────────────────┼──────────────────────┤
  │ 参数化       │ 像素 [0,1]          │ 连续 SDF φ(x)        │
  │ 边界表示     │ 隐式（像素阶梯）     │ 显式（零等值线）      │
  │ 光滑性       │ 需 TV/量化后处理     │ 天然 C¹ 连续          │
  │ 特征尺寸     │ 形态学/频率约束      │ 距离约束/曲率流       │
  │ 拓扑变化     │ 需显式分裂/合并      │ 自动处理              │
  │ 优化变量     │ N² 像素值           │ N² SDF 值 + 边界速度  │
  └──────────────┴─────────────────────┴──────────────────────┘
"""

try:
    from topology.level_set import LevelSetFunction
    from topology.simp_material import (
        heaviside_projection,
        heaviside_projection_gradient,
        simp_interpolation,
        simp_interpolation_gradient,
        sensitivity_filter,
        compute_shape_gradient_to_levelset,
        SIMPMaterialModel,
    )
    from topology.hamilton_jacobi import (
        upwind_gradient,
        compute_upwind_gradient_magnitude,
        compute_cfl_timestep,
        HamiltonJacobiEvolver,
        ShapeVelocityCalculator,
    )
    from topology.constraints import (
        curvature_smoothing_velocity,
        min_feature_size_constraint_velocity,
        perimeter_constraint_velocity,
        volume_constraint_velocity,
        TopologyConstraints,
    )
    from topology.optimizer import (
        TopologyMethod,
        TopologyOptConfig,
        TopologyIterationResult,
        TopologyOptimizationResult,
        LevelSetTopologyOptimizer,
        run_topology_optimization,
    )
except ImportError:
    from .level_set import LevelSetFunction
    from .simp_material import (
        heaviside_projection,
        heaviside_projection_gradient,
        simp_interpolation,
        simp_interpolation_gradient,
        sensitivity_filter,
        compute_shape_gradient_to_levelset,
        SIMPMaterialModel,
    )
    from .hamilton_jacobi import (
        upwind_gradient,
        compute_upwind_gradient_magnitude,
        compute_cfl_timestep,
        HamiltonJacobiEvolver,
        ShapeVelocityCalculator,
    )
    from .constraints import (
        curvature_smoothing_velocity,
        min_feature_size_constraint_velocity,
        perimeter_constraint_velocity,
        volume_constraint_velocity,
        TopologyConstraints,
    )
    from .optimizer import (
        TopologyMethod,
        TopologyOptConfig,
        TopologyIterationResult,
        TopologyOptimizationResult,
        LevelSetTopologyOptimizer,
        run_topology_optimization,
    )

__all__ = [
    'LevelSetFunction',
    'heaviside_projection',
    'heaviside_projection_gradient',
    'simp_interpolation',
    'simp_interpolation_gradient',
    'sensitivity_filter',
    'compute_shape_gradient_to_levelset',
    'SIMPMaterialModel',
    'upwind_gradient',
    'compute_upwind_gradient_magnitude',
    'compute_cfl_timestep',
    'HamiltonJacobiEvolver',
    'ShapeVelocityCalculator',
    'curvature_smoothing_velocity',
    'min_feature_size_constraint_velocity',
    'perimeter_constraint_velocity',
    'volume_constraint_velocity',
    'TopologyConstraints',
    'TopologyMethod',
    'TopologyOptConfig',
    'TopologyIterationResult',
    'TopologyOptimizationResult',
    'LevelSetTopologyOptimizer',
    'run_topology_optimization',
]
