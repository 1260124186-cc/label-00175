# -*- coding: utf-8 -*-
"""
芯片级 RET 编排数据结构定义

定义芯片区域、RET 策略、光学条件、优化结果等核心数据结构。
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class RegionType(Enum):
    """芯片区域类型枚举

    - MEMORY_ARRAY: 内存阵列区域（SRAM/DRAM 等规则重复结构）
    - LOGIC_STDCELL: 逻辑标准单元区域（随机逻辑，中等复杂度）
    - ANALOG_IP: 模拟 IP 区域（放大器、ADC 等特殊定制电路）
    - MIXED_SIGNAL: 混合信号区域（模数混合电路）
    - IO_RING: IO 环区域（芯片外围输入输出电路）
    - UNKNOWN: 未识别区域
    """
    MEMORY_ARRAY = "memory_array"
    LOGIC_STDCELL = "logic_stdcell"
    ANALOG_IP = "analog_ip"
    MIXED_SIGNAL = "mixed_signal"
    IO_RING = "io_ring"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, s: str) -> 'RegionType':
        """从字符串创建区域类型"""
        try:
            return cls(s.lower())
        except ValueError:
            return cls.UNKNOWN


class RETStrategyType(Enum):
    """RET 策略类型枚举

    - OPC_RULE_BASED: 基于规则的 OPC（简单快速，适用于规则结构）
    - OPC_MODEL_BASED: 基于模型的 OPC（精度高，适用于逻辑电路）
    - OPC_SRAF: OPC + SRAF 辅助特征（适用于 k1 较小的密集结构）
    - ILT_BINARY: 二值反演光刻（适用于复杂逻辑，高质量）
    - ILT_TERNARY: 三值反演光刻（含半透膜，适用于先进节点）
    - SMO_ILT: 光源掩模协同优化 + ILT（极端条件下的最优解）
    - INVERSE_DITHER: 反演抖动（适用于模拟/射频等特殊要求）
    - NO_RET: 不进行 RET（适用于大尺寸特征）
    """
    OPC_RULE_BASED = "opc_rule_based"
    OPC_MODEL_BASED = "opc_model_based"
    OPC_SRAF = "opc_sraf"
    ILT_BINARY = "ilt_binary"
    ILT_TERNARY = "ilt_ternary"
    SMO_ILT = "smo_ilt"
    INVERSE_DITHER = "inverse_dither"
    NO_RET = "no_ret"

    @property
    def complexity_level(self) -> int:
        """策略复杂度等级（1-8），用于估算计算成本"""
        level_map = {
            self.NO_RET: 0,
            self.OPC_RULE_BASED: 2,
            self.OPC_MODEL_BASED: 4,
            self.OPC_SRAF: 5,
            self.ILT_BINARY: 6,
            self.INVERSE_DITHER: 7,
            self.ILT_TERNARY: 7,
            self.SMO_ILT: 8,
        }
        return level_map[self]


@dataclass
class OpticalConditionConfig:
    """光学条件配置

    为特定区域定制的光学参数，可覆盖全局配置。
    """
    wavelength_nm: float = 193.0
    na: float = 1.35
    sigma: float = 0.75
    defocus_nm: float = 0.0
    illumination_type: str = "conventional"
    source_params: Dict[str, Any] = field(default_factory=dict)
    n_immersion: float = 1.437
    use_vector_pupil: bool = False
    flare: float = 0.0
    mask_attenuation: float = 0.0
    zernike_coefficients: Dict[int, float] = field(default_factory=dict)
    tcc_mode: str = "socs"
    socs_num_terms: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            'wavelength_nm': self.wavelength_nm,
            'na': self.na,
            'sigma': self.sigma,
            'defocus_nm': self.defocus_nm,
            'illumination_type': self.illumination_type,
            'source_params': dict(self.source_params),
            'n_immersion': self.n_immersion,
            'use_vector_pupil': self.use_vector_pupil,
            'flare': self.flare,
            'mask_attenuation': self.mask_attenuation,
            'zernike_coefficients': dict(self.zernike_coefficients),
            'tcc_mode': self.tcc_mode,
            'socs_num_terms': self.socs_num_terms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OpticalConditionConfig':
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


@dataclass
class RETStrategyConfig:
    """RET 策略配置

    定义特定区域的 RET 优化策略和参数。
    """
    strategy_type: RETStrategyType = RETStrategyType.OPC_MODEL_BASED
    optical_condition: OpticalConditionConfig = field(
        default_factory=OpticalConditionConfig
    )

    max_iterations: int = 100
    learning_rate: float = 0.01
    convergence_tol: float = 1e-6
    epe_threshold_nm: float = 2.0
    cd_error_threshold_nm: float = 1.5

    sraf_enable: bool = False
    sraf_min_distance_nm: float = 20.0
    sraf_width_nm: float = 10.0
    sraf_length_nm: float = 40.0

    ilt_quantization_level: str = "binary"
    ilt_resist_steepness: float = 50.0
    ilt_wafer_threshold: float = 0.3

    mask_complexity_weight: float = 0.0
    perimeter_weight: float = 0.0
    vertex_weight: float = 0.0
    tv_smooth_weight: float = 0.0

    multi_focus_conditions: Optional[List[Dict[str, float]]] = None
    multi_focus_weights: Optional[List[float]] = None

    pixel_size_nm: float = 1.0
    wafer_threshold: float = 0.3
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_type': self.strategy_type.value,
            'optical_condition': self.optical_condition.to_dict(),
            'max_iterations': self.max_iterations,
            'learning_rate': self.learning_rate,
            'convergence_tol': self.convergence_tol,
            'epe_threshold_nm': self.epe_threshold_nm,
            'cd_error_threshold_nm': self.cd_error_threshold_nm,
            'sraf_enable': self.sraf_enable,
            'sraf_min_distance_nm': self.sraf_min_distance_nm,
            'sraf_width_nm': self.sraf_width_nm,
            'sraf_length_nm': self.sraf_length_nm,
            'ilt_quantization_level': self.ilt_quantization_level,
            'ilt_resist_steepness': self.ilt_resist_steepness,
            'ilt_wafer_threshold': self.ilt_wafer_threshold,
            'mask_complexity_weight': self.mask_complexity_weight,
            'perimeter_weight': self.perimeter_weight,
            'vertex_weight': self.vertex_weight,
            'tv_smooth_weight': self.tv_smooth_weight,
            'multi_focus_conditions': self.multi_focus_conditions,
            'multi_focus_weights': self.multi_focus_weights,
            'pixel_size_nm': self.pixel_size_nm,
            'wafer_threshold': self.wafer_threshold,
            'verbose': self.verbose,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RETStrategyConfig':
        cfg = cls()
        for key, value in d.items():
            if key == 'strategy_type' and isinstance(value, str):
                cfg.strategy_type = RETStrategyType(value)
            elif key == 'optical_condition' and isinstance(value, dict):
                cfg.optical_condition = OpticalConditionConfig.from_dict(value)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    @property
    def estimated_runtime_factor(self) -> float:
        """估算运行时间因子，相对于基础 OPC"""
        base = 1.0
        strategy_factor = {
            RETStrategyType.NO_RET: 0.1,
            RETStrategyType.OPC_RULE_BASED: 0.5,
            RETStrategyType.OPC_MODEL_BASED: 1.0,
            RETStrategyType.OPC_SRAF: 1.5,
            RETStrategyType.ILT_BINARY: 3.0,
            RETStrategyType.ILT_TERNARY: 4.0,
            RETStrategyType.INVERSE_DITHER: 5.0,
            RETStrategyType.SMO_ILT: 8.0,
        }
        base *= strategy_factor.get(self.strategy_type, 1.0)

        if self.multi_focus_conditions:
            base *= len(self.multi_focus_conditions) * 0.75

        if self.optical_condition.use_vector_pupil:
            base *= 1.5

        return base


@dataclass
class ChipRegionMetadata:
    """芯片区域元数据

    记录区域的几何、拓扑和复杂度信息。
    """
    region_id: str
    region_type: RegionType
    bounds_nm: Tuple[float, float, float, float]
    bounds_px: Optional[Tuple[int, int, int, int]] = None

    pixel_size_nm: float = 1.0
    area_um2: float = 0.0
    polygon_count: int = 0
    edge_density: float = 0.0
    corner_density: float = 0.0
    fill_ratio: float = 0.0
    min_cd_nm: float = 0.0
    periodicity_score: float = 0.0
    dominant_pitch_nm: float = 0.0
    spectral_high_freq_ratio: float = 0.0

    k1_factor: float = 0.0
    complexity_score: float = 0.0

    cell_name_hints: List[str] = field(default_factory=list)
    layer_hints: List[int] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'region_id': self.region_id,
            'region_type': self.region_type.value,
            'bounds_nm': list(self.bounds_nm),
            'bounds_px': list(self.bounds_px) if self.bounds_px else None,
            'pixel_size_nm': self.pixel_size_nm,
            'area_um2': self.area_um2,
            'polygon_count': self.polygon_count,
            'edge_density': self.edge_density,
            'corner_density': self.corner_density,
            'fill_ratio': self.fill_ratio,
            'min_cd_nm': self.min_cd_nm,
            'periodicity_score': self.periodicity_score,
            'dominant_pitch_nm': self.dominant_pitch_nm,
            'spectral_high_freq_ratio': self.spectral_high_freq_ratio,
            'k1_factor': self.k1_factor,
            'complexity_score': self.complexity_score,
            'cell_name_hints': list(self.cell_name_hints),
            'layer_hints': list(self.layer_hints),
            'extra': dict(self.extra),
        }


@dataclass
class ChipRegion:
    """芯片区域数据结构

    表示芯片上的一个独立区域，包含掩模数据、元数据和 RET 策略。
    """
    region_id: str
    metadata: ChipRegionMetadata
    mask: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None
    ret_strategy: Optional[RETStrategyConfig] = None

    optimized_mask: Optional[np.ndarray] = None
    optimization_result: Optional['BlockOptimizationResult'] = None

    is_optimized: bool = False
    priority: int = 50
    dependencies: List[str] = field(default_factory=list)

    overlap_region_ids: List[str] = field(default_factory=list)
    overlap_width_px: int = 0

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        return self.mask.shape if self.mask is not None else None

    @property
    def is_mask_loaded(self) -> bool:
        return self.mask is not None

    @property
    def has_optimized(self) -> bool:
        return self.optimized_mask is not None

    @property
    def width_nm(self) -> float:
        return self.metadata.bounds_nm[2] - self.metadata.bounds_nm[0]

    @property
    def height_nm(self) -> float:
        return self.metadata.bounds_nm[3] - self.metadata.bounds_nm[1]

    @property
    def origin_nm(self) -> Tuple[float, float]:
        return (self.metadata.bounds_nm[0], self.metadata.bounds_nm[1])

    def ensure_mask_loaded(self) -> None:
        if self.mask is None:
            raise RuntimeError(f"区域 {self.region_id} 的掩模未加载")
        if self.target is None:
            self.target = self.mask.copy()

    def summary(self) -> Dict[str, Any]:
        return {
            'region_id': self.region_id,
            'region_type': self.metadata.region_type.value,
            'bounds_nm': list(self.metadata.bounds_nm),
            'shape': list(self.shape) if self.shape else None,
            'area_um2': self.metadata.area_um2,
            'polygon_count': self.metadata.polygon_count,
            'min_cd_nm': self.metadata.min_cd_nm,
            'k1_factor': self.metadata.k1_factor,
            'complexity_score': self.metadata.complexity_score,
            'ret_strategy': self.ret_strategy.strategy_type.value if self.ret_strategy else None,
            'is_optimized': self.is_optimized,
            'priority': self.priority,
        }


@dataclass
class BlockOptimizationConfig:
    """分块优化配置"""
    block_size_px: Tuple[int, int] = (512, 512)
    overlap_px: int = 32
    pixel_size_nm: float = 1.0
    max_parallel_blocks: int = 4
    enable_checkpointing: bool = True
    checkpoint_dir: Optional[str] = None
    save_intermediate_results: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'block_size_px': list(self.block_size_px),
            'overlap_px': self.overlap_px,
            'pixel_size_nm': self.pixel_size_nm,
            'max_parallel_blocks': self.max_parallel_blocks,
            'enable_checkpointing': self.enable_checkpointing,
            'checkpoint_dir': self.checkpoint_dir,
            'save_intermediate_results': self.save_intermediate_results,
        }


@dataclass
class StitchingConfig:
    """拼合配置

    控制区域拼合和边界伪影处理的参数。
    """
    overlap_width_px: int = 32
    feathering_width_px: int = 16
    boundary_smooth_sigma_px: float = 2.0
    artifact_detection_threshold: float = 0.15
    artifact_window_size_px: int = 16
    max_artifact_correction_iterations: int = 3
    enable_gradient_blending: bool = True
    enable_boundary_correction: bool = True
    enable_global_consistency: bool = True
    consistency_weight: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            'overlap_width_px': self.overlap_width_px,
            'feathering_width_px': self.feathering_width_px,
            'boundary_smooth_sigma_px': self.boundary_smooth_sigma_px,
            'artifact_detection_threshold': self.artifact_detection_threshold,
            'artifact_window_size_px': self.artifact_window_size_px,
            'max_artifact_correction_iterations': self.max_artifact_correction_iterations,
            'enable_gradient_blending': self.enable_gradient_blending,
            'enable_boundary_correction': self.enable_boundary_correction,
            'enable_global_consistency': self.enable_global_consistency,
            'consistency_weight': self.consistency_weight,
        }


@dataclass
class ChipRETConfig:
    """芯片级 RET 编排配置

    完整的芯片级 RET 优化配置。
    """
    chip_name: str = "unknown"
    layer: int = 0
    datatype: int = 0
    pixel_size_nm: float = 1.0

    global_optical_condition: OpticalConditionConfig = field(
        default_factory=OpticalConditionConfig
    )

    block_config: BlockOptimizationConfig = field(
        default_factory=BlockOptimizationConfig
    )

    stitching_config: StitchingConfig = field(
        default_factory=StitchingConfig
    )

    enable_parallel_optimization: bool = True
    max_parallel_regions: int = 4

    strategy_auto_selection: bool = True
    min_cd_for_ilt_nm: float = 40.0
    max_cd_for_opc_nm: float = 100.0

    min_region_size_um2: float = 100.0
    merge_distance_um: float = 5.0
    use_hierarchy_for_partition: bool = True

    output_dir: Optional[str] = None
    save_regions_separately: bool = True
    save_stitched_mask: bool = True
    save_report: bool = True

    verbose: bool = True
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chip_name': self.chip_name,
            'layer': self.layer,
            'datatype': self.datatype,
            'pixel_size_nm': self.pixel_size_nm,
            'global_optical_condition': self.global_optical_condition.to_dict(),
            'block_config': self.block_config.to_dict(),
            'stitching_config': self.stitching_config.to_dict(),
            'enable_parallel_optimization': self.enable_parallel_optimization,
            'max_parallel_regions': self.max_parallel_regions,
            'strategy_auto_selection': self.strategy_auto_selection,
            'min_cd_for_ilt_nm': self.min_cd_for_ilt_nm,
            'max_cd_for_opc_nm': self.max_cd_for_opc_nm,
            'min_region_size_um2': self.min_region_size_um2,
            'merge_distance_um': self.merge_distance_um,
            'use_hierarchy_for_partition': self.use_hierarchy_for_partition,
            'output_dir': self.output_dir,
            'save_regions_separately': self.save_regions_separately,
            'save_stitched_mask': self.save_stitched_mask,
            'save_report': self.save_report,
            'verbose': self.verbose,
            'log_level': self.log_level,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ChipRETConfig':
        cfg = cls()
        for key, value in d.items():
            if key == 'global_optical_condition' and isinstance(value, dict):
                cfg.global_optical_condition = OpticalConditionConfig.from_dict(value)
            elif key == 'block_config' and isinstance(value, dict):
                cfg.block_config = BlockOptimizationConfig(**value)
            elif key == 'stitching_config' and isinstance(value, dict):
                cfg.stitching_config = StitchingConfig(**value)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


@dataclass
class BlockOptimizationResult:
    """分块优化结果"""
    region_id: str
    success: bool = False
    converged: bool = False
    iterations: int = 0
    total_time_sec: float = 0.0

    initial_mask: Optional[np.ndarray] = None
    optimized_mask: Optional[np.ndarray] = None
    initial_wafer: Optional[np.ndarray] = None
    optimized_wafer: Optional[np.ndarray] = None

    initial_epe: Dict[str, float] = field(default_factory=dict)
    final_epe: Dict[str, float] = field(default_factory=dict)
    initial_cd: Dict[str, float] = field(default_factory=dict)
    final_cd: Dict[str, float] = field(default_factory=dict)
    initial_mse: float = 0.0
    final_mse: float = 0.0

    strategy_used: Optional[RETStrategyType] = None
    optical_condition_used: Optional[OpticalConditionConfig] = None

    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def epe_improvement_nm(self) -> float:
        init = self.initial_epe.get('epe_mean', 0.0)
        final = self.final_epe.get('epe_mean', 0.0)
        return init - final

    @property
    def epe_improvement_ratio(self) -> float:
        init = self.initial_epe.get('epe_mean', 0.0)
        if init <= 0:
            return 0.0
        return self.epe_improvement_nm / init

    def to_dict(self) -> Dict[str, Any]:
        return {
            'region_id': self.region_id,
            'success': self.success,
            'converged': self.converged,
            'iterations': self.iterations,
            'total_time_sec': self.total_time_sec,
            'initial_epe': dict(self.initial_epe),
            'final_epe': dict(self.final_epe),
            'initial_cd': dict(self.initial_cd),
            'final_cd': dict(self.final_cd),
            'initial_mse': self.initial_mse,
            'final_mse': self.final_mse,
            'epe_improvement_nm': self.epe_improvement_nm,
            'epe_improvement_ratio': self.epe_improvement_ratio,
            'strategy_used': self.strategy_used.value if self.strategy_used else None,
            'error_message': self.error_message,
            'warnings': list(self.warnings),
            'extra': dict(self.extra),
        }


@dataclass
class BoundaryArtifactMetrics:
    """边界伪影指标

    量化评估拼合边界的伪影程度。
    """
    boundary_id: str
    max_discontinuity: float = 0.0
    mean_discontinuity: float = 0.0
    std_discontinuity: float = 0.0
    max_gradient_jump: float = 0.0
    mean_gradient_jump: float = 0.0
    artifact_pixel_count: int = 0
    artifact_density: float = 0.0
    corrected_count: int = 0
    post_correction_max: float = 0.0
    post_correction_mean: float = 0.0
    correction_improvement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'boundary_id': self.boundary_id,
            'max_discontinuity': self.max_discontinuity,
            'mean_discontinuity': self.mean_discontinuity,
            'std_discontinuity': self.std_discontinuity,
            'max_gradient_jump': self.max_gradient_jump,
            'mean_gradient_jump': self.mean_gradient_jump,
            'artifact_pixel_count': self.artifact_pixel_count,
            'artifact_density': self.artifact_density,
            'corrected_count': self.corrected_count,
            'post_correction_max': self.post_correction_max,
            'post_correction_mean': self.post_correction_mean,
            'correction_improvement': self.correction_improvement,
        }


@dataclass
class ChipRETResult:
    """芯片级 RET 最终结果"""
    chip_name: str
    success: bool = False
    total_time_sec: float = 0.0

    original_mask: Optional[np.ndarray] = None
    stitched_mask: Optional[np.ndarray] = None
    original_wafer: Optional[np.ndarray] = None
    optimized_wafer: Optional[np.ndarray] = None

    regions: List[ChipRegion] = field(default_factory=list)
    block_results: List[BlockOptimizationResult] = field(default_factory=list)
    boundary_metrics: List[BoundaryArtifactMetrics] = field(default_factory=list)

    global_initial_epe: Dict[str, float] = field(default_factory=dict)
    global_final_epe: Dict[str, float] = field(default_factory=dict)
    global_initial_cd: Dict[str, float] = field(default_factory=dict)
    global_final_cd: Dict[str, float] = field(default_factory=dict)

    region_type_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    strategy_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    error_message: str = ""
    warnings: List[str] = field(default_factory=list)

    output_files: Dict[str, str] = field(default_factory=dict)

    @property
    def num_regions(self) -> int:
        return len(self.regions)

    @property
    def num_successful_regions(self) -> int:
        return sum(1 for r in self.block_results if r.success)

    @property
    def success_rate(self) -> float:
        if self.num_regions == 0:
            return 0.0
        return self.num_successful_regions / self.num_regions

    @property
    def global_epe_improvement(self) -> float:
        init = self.global_initial_epe.get('epe_mean', 0.0)
        final = self.global_final_epe.get('epe_mean', 0.0)
        return init - final

    def summary(self) -> Dict[str, Any]:
        return {
            'chip_name': self.chip_name,
            'success': self.success,
            'total_time_sec': self.total_time_sec,
            'num_regions': self.num_regions,
            'num_successful_regions': self.num_successful_regions,
            'success_rate': self.success_rate,
            'global_initial_epe': dict(self.global_initial_epe),
            'global_final_epe': dict(self.global_final_epe),
            'global_epe_improvement': self.global_epe_improvement,
            'region_type_summary': self.region_type_summary,
            'strategy_summary': self.strategy_summary,
            'boundary_count': len(self.boundary_metrics),
            'max_boundary_discontinuity': max(
                (m.post_correction_max for m in self.boundary_metrics),
                default=0.0
            ),
            'mean_boundary_discontinuity': float(np.mean(
                [m.post_correction_mean for m in self.boundary_metrics]
            )) if self.boundary_metrics else 0.0,
            'warnings': list(self.warnings),
            'output_files': dict(self.output_files),
        }
