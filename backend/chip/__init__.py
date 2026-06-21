# -*- coding: utf-8 -*-
"""
芯片级层次化 RET 编排模块

提供完整芯片的层次化分辨率增强技术（RET）编排能力：
1. 芯片 GDS 区域划分（内存阵列、逻辑标准单元、模拟 IP）
2. 区域级 RET 策略匹配
3. 分块优化执行
4. 坐标拼合与边界伪影处理

主要组件：
- ChipRegion: 芯片区域数据结构
- RegionPartitioner: 区域划分器
- RETStrategyMatcher: RET 策略匹配器
- BlockOptimizer: 分块优化器
- BoundaryStitcher: 边界拼合器
- ChipRETOrchestrator: 芯片级 RET 编排器
"""

from .schemas import (
    RegionType,
    RETStrategyType,
    ChipRegion,
    ChipRegionMetadata,
    RETStrategyConfig,
    BlockOptimizationConfig,
    StitchingConfig,
    ChipRETConfig,
    BlockOptimizationResult,
    ChipRETResult,
    BoundaryArtifactMetrics,
)

from .region_partitioner import (
    RegionPartitioner,
    RegionPartitionResult,
)

from .ret_strategy_matcher import (
    RETStrategyMatcher,
)

from .block_optimizer import (
    BlockOptimizer,
)

from .stitcher import (
    BoundaryStitcher,
)

from .orchestrator import (
    ChipRETOrchestrator,
    run_chip_level_ret,
)

__all__ = [
    'RegionType',
    'RETStrategyType',
    'ChipRegion',
    'ChipRegionMetadata',
    'RETStrategyConfig',
    'BlockOptimizationConfig',
    'StitchingConfig',
    'ChipRETConfig',
    'BlockOptimizationResult',
    'ChipRETResult',
    'BoundaryArtifactMetrics',
    'RegionPartitioner',
    'RegionPartitionResult',
    'RETStrategyMatcher',
    'BlockOptimizer',
    'BoundaryStitcher',
    'ChipRETOrchestrator',
    'run_chip_level_ret',
]

__version__ = '1.0.0'
