# -*- coding: utf-8 -*-
"""
RET 策略匹配器

为不同类型的芯片区域自动匹配最合适的 RET 策略和光学条件。

核心功能：
1. 区域类型默认策略映射
2. 基于特征的智能策略推荐（集成 advisor 推荐引擎）
3. 区域定制化光学条件配置
4. 策略冲突检测与解决
5. 计算资源预算分配
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

from chip.schemas import (
    RegionType, RETStrategyType, ChipRegion, ChipRegionMetadata,
    RETStrategyConfig, OpticalConditionConfig, ChipRETConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyMatchResult:
    """策略匹配结果"""
    region_id: str
    strategy_config: RETStrategyConfig
    match_reason: str
    confidence: float = 0.0
    alternative_strategies: List[Tuple[RETStrategyType, float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'region_id': self.region_id,
            'strategy_config': self.strategy_config.to_dict(),
            'match_reason': self.match_reason,
            'confidence': self.confidence,
            'alternative_strategies': [(s.value, w) for s, w in self.alternative_strategies],
            'warnings': list(self.warnings),
        }


class RETStrategyMatcher:
    """
    RET 策略匹配器

    为芯片区域分配合适的 RET 策略和光学条件。

    使用方法：
        matcher = RETStrategyMatcher(global_config)
        for region in regions:
            result = matcher.match(region)
            region.ret_strategy = result.strategy_config
    """

    DEFAULT_STRATEGY_MAP: Dict[RegionType, RETStrategyType] = {
        RegionType.MEMORY_ARRAY: RETStrategyType.OPC_SRAF,
        RegionType.LOGIC_STDCELL: RETStrategyType.ILT_BINARY,
        RegionType.ANALOG_IP: RETStrategyType.OPC_MODEL_BASED,
        RegionType.MIXED_SIGNAL: RETStrategyType.ILT_BINARY,
        RegionType.IO_RING: RETStrategyType.OPC_RULE_BASED,
        RegionType.UNKNOWN: RETStrategyType.OPC_MODEL_BASED,
    }

    def __init__(
        self,
        global_config: Optional[ChipRETConfig] = None,
        enable_advisor_engine: bool = True,
        user_preference: str = "balanced",
    ):
        """
        初始化策略匹配器

        Args:
            global_config: 芯片级 RET 全局配置
            enable_advisor_engine: 是否启用 advisor 推荐引擎
            user_preference: 用户偏好 ('speed' / 'quality' / 'balanced')
        """
        self.global_config = global_config or ChipRETConfig()
        self.enable_advisor_engine = enable_advisor_engine
        self.user_preference = user_preference
        self._advisor_engine = None

        if self.enable_advisor_engine:
            try:
                from advisor.recommendation_engine import RETRecommendationEngine
                self._advisor_engine = RETRecommendationEngine()
                logger.info("成功加载 RET 推荐引擎")
            except ImportError as e:
                logger.warning(f"无法加载 RET 推荐引擎: {e}，将使用基于规则的匹配")
                self.enable_advisor_engine = False

    def match(
        self,
        region: ChipRegion,
        override_strategy: Optional[RETStrategyType] = None,
    ) -> StrategyMatchResult:
        """
        为单个区域匹配 RET 策略

        Args:
            region: 芯片区域
            override_strategy: 强制指定的策略类型（可选）

        Returns:
            StrategyMatchResult 匹配结果
        """
        metadata = region.metadata
        region_type = metadata.region_type

        if override_strategy is not None:
            strategy_type = override_strategy
            reason = f"用户强制指定策略: {override_strategy.value}"
            confidence = 1.0
        elif self.enable_advisor_engine and region.mask is not None:
            strategy_type, reason, confidence, alternatives = self._match_with_advisor(region)
        else:
            strategy_type, reason, confidence, alternatives = self._match_by_rules(metadata)

        optical_condition = self._configure_optical_condition(metadata, strategy_type)
        strategy_config = self._build_strategy_config(
            metadata, strategy_type, optical_condition
        )

        warnings = self._generate_warnings(metadata, strategy_type)

        result = StrategyMatchResult(
            region_id=region.region_id,
            strategy_config=strategy_config,
            match_reason=reason,
            confidence=confidence,
            warnings=warnings,
        )

        if not override_strategy and 'alternatives' in locals():
            result.alternative_strategies = alternatives

        return result

    def match_all(
        self,
        regions: List[ChipRegion],
        strategy_overrides: Optional[Dict[str, RETStrategyType]] = None,
    ) -> List[StrategyMatchResult]:
        """
        为所有区域批量匹配 RET 策略

        Args:
            regions: 芯片区域列表
            strategy_overrides: 区域 ID 到强制策略的映射

        Returns:
            匹配结果列表
        """
        results = []
        strategy_overrides = strategy_overrides or {}

        for region in regions:
            override = strategy_overrides.get(region.region_id)
            result = self.match(region, override_strategy=override)
            region.ret_strategy = result.strategy_config
            results.append(result)

        self._check_strategy_consistency(results)
        return results

    def _match_with_advisor(
        self,
        region: ChipRegion,
    ) -> Tuple[RETStrategyType, str, float, List[Tuple[RETStrategyType, float]]]:
        """
        使用 advisor 推荐引擎进行策略匹配

        Args:
            region: 芯片区域

        Returns:
            (策略类型, 匹配原因, 置信度, 备选策略列表)
        """
        metadata = region.metadata
        mask = region.mask

        try:
            from advisor.schemas import LayoutFeatures, SpectralFeatures, RETStrategy as AdvisorStrategy

            spectral = SpectralFeatures(
                high_freq_energy_ratio=metadata.spectral_high_freq_ratio,
                dominant_frequency=1.0 / metadata.dominant_pitch_nm if metadata.dominant_pitch_nm > 0 else 0.0,
            )

            features = LayoutFeatures(
                min_cd_nm=metadata.min_cd_nm,
                corner_density=metadata.corner_density,
                periodicity_score=metadata.periodicity_score,
                dominant_pitch_nm=metadata.dominant_pitch_nm,
                fill_ratio=metadata.fill_ratio,
                spectral=spectral,
                wavelength=self.global_config.global_optical_condition.wavelength_nm,
                na=self.global_config.global_optical_condition.na,
                pixel_size=metadata.pixel_size_nm,
                image_shape=region.shape or (0, 0),
            )

            advisor_result = self._advisor_engine.recommend_from_features(
                features=features,
                user_preference=self.user_preference,
            )

            primary = advisor_result.primary
            advisor_strategy = primary.strategy

            chip_strategy = self._map_advisor_to_chip_strategy(
                advisor_strategy, metadata.region_type
            )

            alternatives = []
            for alt in advisor_result.alternatives:
                alt_chip_strategy = self._map_advisor_to_chip_strategy(
                    alt.strategy, metadata.region_type
                )
                alternatives.append((alt_chip_strategy, alt.confidence))

            region_type_hint = self.DEFAULT_STRATEGY_MAP.get(
                metadata.region_type, RETStrategyType.OPC_MODEL_BASED
            )

            if metadata.k1_factor < 0.35 and chip_strategy.complexity_level < 6:
                chip_strategy = RETStrategyType.ILT_BINARY
                reason = f"k1={metadata.k1_factor:.2f} 极低，需要 ILT 以上强度。" + primary.reason
            elif metadata.k1_factor > 0.7 and chip_strategy.complexity_level > 4:
                chip_strategy = RETStrategyType.OPC_MODEL_BASED
                reason = f"k1={metadata.k1_factor:.2f} 较高，使用基础 OPC 即可。" + primary.reason
            else:
                reason = primary.reason

            if metadata.region_type == RegionType.MEMORY_ARRAY and metadata.periodicity_score > 0.7:
                if chip_strategy.complexity_level > 5:
                    chip_strategy = RETStrategyType.OPC_SRAF
                    reason = f"高周期性内存阵列，OPC+SRAF 性价比最优。" + reason

            if metadata.region_type == RegionType.ANALOG_IP and metadata.min_cd_nm > 100:
                chip_strategy = RETStrategyType.OPC_MODEL_BASED
                reason = f"模拟 IP 大尺寸器件，模型 OPC 足够。" + reason

            confidence = primary.confidence

            return chip_strategy, reason, confidence, alternatives

        except Exception as e:
            logger.warning(f"Advisor 引擎匹配失败，回退到规则匹配: {e}")
            return self._match_by_rules(metadata)

    def _match_by_rules(
        self,
        metadata: ChipRegionMetadata,
    ) -> Tuple[RETStrategyType, str, float, List[Tuple[RETStrategyType, float]]]:
        """
        基于规则的策略匹配

        Args:
            metadata: 区域元数据

        Returns:
            (策略类型, 匹配原因, 置信度, 备选策略列表)
        """
        region_type = metadata.region_type
        k1 = metadata.k1_factor
        min_cd = metadata.min_cd_nm
        complexity = metadata.complexity_score
        periodicity = metadata.periodicity_score

        default_strategy = self.DEFAULT_STRATEGY_MAP.get(
            region_type, RETStrategyType.OPC_MODEL_BASED
        )

        strategy_scores: Dict[RETStrategyType, float] = {
            s: 0.0 for s in RETStrategyType
        }

        if k1 < 0.35:
            strategy_scores[RETStrategyType.SMO_ILT] += 0.5
            strategy_scores[RETStrategyType.ILT_TERNARY] += 0.4
            strategy_scores[RETStrategyType.ILT_BINARY] += 0.3
        elif k1 < 0.45:
            strategy_scores[RETStrategyType.ILT_BINARY] += 0.4
            strategy_scores[RETStrategyType.ILT_TERNARY] += 0.3
            strategy_scores[RETStrategyType.OPC_SRAF] += 0.2
        elif k1 < 0.6:
            strategy_scores[RETStrategyType.OPC_SRAF] += 0.3
            strategy_scores[RETStrategyType.ILT_BINARY] += 0.2
            strategy_scores[RETStrategyType.OPC_MODEL_BASED] += 0.1
        else:
            strategy_scores[RETStrategyType.OPC_MODEL_BASED] += 0.4
            strategy_scores[RETStrategyType.OPC_RULE_BASED] += 0.3
            strategy_scores[RETStrategyType.OPC_SRAF] += 0.1

        if complexity > 0.7:
            strategy_scores[RETStrategyType.ILT_BINARY] += 0.3
            strategy_scores[RETStrategyType.ILT_TERNARY] += 0.2
        elif complexity > 0.4:
            strategy_scores[RETStrategyType.OPC_MODEL_BASED] += 0.2
            strategy_scores[RETStrategyType.OPC_SRAF] += 0.1

        if periodicity > 0.7:
            strategy_scores[RETStrategyType.OPC_SRAF] += 0.3
            strategy_scores[RETStrategyType.OPC_RULE_BASED] += 0.2
        elif periodicity < 0.3:
            strategy_scores[RETStrategyType.ILT_BINARY] += 0.2
            strategy_scores[RETStrategyType.OPC_MODEL_BASED] += 0.1

        if region_type == RegionType.MEMORY_ARRAY:
            if periodicity > 0.6:
                strategy_scores[RETStrategyType.OPC_SRAF] += 0.4
                strategy_scores[RETStrategyType.OPC_RULE_BASED] += 0.2
            else:
                strategy_scores[RETStrategyType.ILT_BINARY] += 0.2

        elif region_type == RegionType.LOGIC_STDCELL:
            strategy_scores[RETStrategyType.ILT_BINARY] += 0.3
            strategy_scores[RETStrategyType.OPC_MODEL_BASED] += 0.2
            if complexity > 0.5:
                strategy_scores[RETStrategyType.ILT_BINARY] += 0.2

        elif region_type == RegionType.ANALOG_IP:
            strategy_scores[RETStrategyType.OPC_MODEL_BASED] += 0.4
            strategy_scores[RETStrategyType.INVERSE_DITHER] += 0.2
            if min_cd > 150:
                strategy_scores[RETStrategyType.NO_RET] += 0.3

        elif region_type == RegionType.IO_RING:
            strategy_scores[RETStrategyType.OPC_RULE_BASED] += 0.4
            strategy_scores[RETStrategyType.OPC_MODEL_BASED] += 0.2
            if min_cd > 200:
                strategy_scores[RETStrategyType.NO_RET] += 0.3

        elif region_type == RegionType.MIXED_SIGNAL:
            strategy_scores[RETStrategyType.ILT_BINARY] += 0.3
            strategy_scores[RETStrategyType.OPC_MODEL_BASED] += 0.2

        if min_cd > self.global_config.max_cd_for_opc_nm:
            strategy_scores[RETStrategyType.NO_RET] += 0.3
            strategy_scores[RETStrategyType.OPC_RULE_BASED] += 0.2

        if self.user_preference == 'speed':
            strategy_scores[RETStrategyType.OPC_RULE_BASED] *= 1.3
            strategy_scores[RETStrategyType.OPC_MODEL_BASED] *= 1.2
            strategy_scores[RETStrategyType.OPC_SRAF] *= 1.1
            strategy_scores[RETStrategyType.ILT_BINARY] *= 0.8
            strategy_scores[RETStrategyType.ILT_TERNARY] *= 0.7
            strategy_scores[RETStrategyType.SMO_ILT] *= 0.6
        elif self.user_preference == 'quality':
            strategy_scores[RETStrategyType.SMO_ILT] *= 1.3
            strategy_scores[RETStrategyType.ILT_TERNARY] *= 1.2
            strategy_scores[RETStrategyType.ILT_BINARY] *= 1.1
            strategy_scores[RETStrategyType.OPC_MODEL_BASED] *= 0.9
            strategy_scores[RETStrategyType.OPC_SRAF] *= 0.85
            strategy_scores[RETStrategyType.OPC_RULE_BASED] *= 0.7

        for s in strategy_scores:
            strategy_scores[s] = max(0.0, min(1.0, strategy_scores[s]))

        ranked = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        best_strategy, best_score = ranked[0]

        if best_score < 0.1:
            best_strategy = default_strategy
            best_score = 0.5

        reason_parts = [
            f"区域类型: {region_type.value}",
            f"k1={k1:.2f}",
            f"复杂度={complexity:.2f}",
            f"周期性={periodicity:.2f}",
            f"最小CD={min_cd:.0f}nm",
        ]
        reason = "基于规则匹配: " + ", ".join(reason_parts) + f" → {best_strategy.value}"

        confidence = min(best_score, 0.9)

        alternatives = [(s, sc) for s, sc in ranked[1:] if sc > 0.1]

        return best_strategy, reason, confidence, alternatives

    def _map_advisor_to_chip_strategy(
        self,
        advisor_strategy: Any,
        region_type: RegionType,
    ) -> RETStrategyType:
        """
        将 advisor 策略类型映射到芯片级策略类型

        Args:
            advisor_strategy: advisor 模块的策略枚举
            region_type: 区域类型

        Returns:
            芯片级 RET 策略类型
        """
        from advisor.schemas import RETStrategy as AdvisorStrategy

        mapping = {
            AdvisorStrategy.OPC_ONLY: RETStrategyType.OPC_MODEL_BASED,
            AdvisorStrategy.OPC_SRAF: RETStrategyType.OPC_SRAF,
            AdvisorStrategy.ILT: RETStrategyType.ILT_BINARY,
            AdvisorStrategy.SMO_ILT: RETStrategyType.SMO_ILT,
        }

        strategy = mapping.get(advisor_strategy, RETStrategyType.OPC_MODEL_BASED)

        if region_type == RegionType.ANALOG_IP and strategy == RETStrategyType.ILT_BINARY:
            strategy = RETStrategyType.INVERSE_DITHER

        return strategy

    def _configure_optical_condition(
        self,
        metadata: ChipRegionMetadata,
        strategy_type: RETStrategyType,
    ) -> OpticalConditionConfig:
        """
        为区域配置定制化光学条件

        Args:
            metadata: 区域元数据
            strategy_type: RET 策略类型

        Returns:
            光学条件配置
        """
        global_opt = self.global_config.global_optical_condition
        opt = OpticalConditionConfig(**global_opt.to_dict())

        region_type = metadata.region_type
        k1 = metadata.k1_factor
        periodicity = metadata.periodicity_score
        min_cd = metadata.min_cd_nm

        if strategy_type in (RETStrategyType.ILT_BINARY, RETStrategyType.ILT_TERNARY,
                            RETStrategyType.SMO_ILT, RETStrategyType.INVERSE_DITHER):
            opt.tcc_mode = "socs"
            opt.socs_num_terms = max(opt.socs_num_terms, 8)

            if k1 < 0.4:
                opt.use_vector_pupil = True

        if region_type == RegionType.MEMORY_ARRAY and periodicity > 0.6:
            opt.illumination_type = "annular"
            opt.source_params = {
                'sigma_inner': 0.6,
                'sigma_outer': 0.9,
            }
        elif region_type == RegionType.LOGIC_STDCELL:
            if metadata.corner_density > 0.08:
                opt.illumination_type = "quasar"
                opt.source_params = {
                    'sigma_inner': 0.5,
                    'sigma_outer': 0.8,
                    'angle': 45.0,
                    'opening_angle': 30.0,
                }
            else:
                opt.illumination_type = "conventional"
        elif region_type == RegionType.ANALOG_IP:
            opt.illumination_type = "conventional"
            opt.sigma = min(opt.sigma, 0.6)
        elif region_type == RegionType.IO_RING:
            opt.illumination_type = "conventional"
            opt.sigma = 0.7

        if k1 < 0.35:
            opt.mask_attenuation = 0.06
        elif k1 < 0.5:
            opt.mask_attenuation = 0.06

        if min_cd < 50 and opt.n_immersion < 1.5:
            opt.n_immersion = max(opt.n_immersion, 1.437)

        return opt

    def _build_strategy_config(
        self,
        metadata: ChipRegionMetadata,
        strategy_type: RETStrategyType,
        optical_condition: OpticalConditionConfig,
    ) -> RETStrategyConfig:
        """
        构建完整的 RET 策略配置

        Args:
            metadata: 区域元数据
            strategy_type: 策略类型
            optical_condition: 光学条件

        Returns:
            RET 策略配置
        """
        config = RETStrategyConfig(
            strategy_type=strategy_type,
            optical_condition=optical_condition,
            pixel_size_nm=metadata.pixel_size_nm,
        )

        k1 = metadata.k1_factor
        min_cd = metadata.min_cd_nm
        region_type = metadata.region_type

        if k1 < 0.4:
            config.max_iterations = max(config.max_iterations, 150)
            config.learning_rate = min(config.learning_rate, 0.008)
            config.epe_threshold_nm = 2.0
            config.cd_error_threshold_nm = 1.5
        elif k1 < 0.6:
            config.max_iterations = max(config.max_iterations, 100)
            config.learning_rate = 0.01
            config.epe_threshold_nm = 2.5
            config.cd_error_threshold_nm = 2.0
        else:
            config.max_iterations = max(config.max_iterations, 50)
            config.epe_threshold_nm = 3.0
            config.cd_error_threshold_nm = 2.5

        if strategy_type in (RETStrategyType.OPC_SRAF, RETStrategyType.ILT_BINARY,
                            RETStrategyType.ILT_TERNARY, RETStrategyType.SMO_ILT):
            config.sraf_enable = True
            config.sraf_min_distance_nm = max(min_cd * 1.5, 20.0)
            config.sraf_width_nm = max(min_cd * 0.4, 10.0)
            config.sraf_length_nm = max(min_cd * 2.0, 40.0)

        if strategy_type in (RETStrategyType.ILT_BINARY, RETStrategyType.ILT_TERNARY,
                            RETStrategyType.SMO_ILT, RETStrategyType.INVERSE_DITHER):
            config.ilt_quantization_level = "binary" if strategy_type == RETStrategyType.ILT_BINARY else "ternary"
            config.ilt_resist_steepness = 50.0
            config.ilt_wafer_threshold = 0.3

            if k1 < 0.4:
                config.ilt_resist_steepness = 60.0
                config.mask_complexity_weight = 0.02
                config.tv_smooth_weight = 0.001
            else:
                config.mask_complexity_weight = 0.01

        if region_type == RegionType.MEMORY_ARRAY:
            config.multi_focus_conditions = [
                {'defocus_nm': -30},
                {'defocus_nm': 0},
                {'defocus_nm': 30},
            ]
            config.multi_focus_weights = [0.25, 0.5, 0.25]

        if k1 < 0.4 and strategy_type.complexity_level >= 6:
            config.multi_focus_conditions = [
                {'defocus_nm': -50},
                {'defocus_nm': 0},
                {'defocus_nm': 50},
            ]
            config.multi_focus_weights = [0.3, 0.4, 0.3]

        if region_type == RegionType.ANALOG_IP:
            config.tv_smooth_weight = max(config.tv_smooth_weight, 0.002)
            config.vertex_weight = 0.001

        return config

    def _generate_warnings(
        self,
        metadata: ChipRegionMetadata,
        strategy_type: RETStrategyType,
    ) -> List[str]:
        """生成策略匹配警告信息"""
        warnings = []
        k1 = metadata.k1_factor

        if k1 < 0.3 and strategy_type.complexity_level < 7:
            warnings.append(
                f"k1={k1:.2f} 极低，建议使用 SMO+ILT 或三值 ILT 策略。"
            )

        if metadata.min_cd_nm < 30 and strategy_type in (
            RETStrategyType.OPC_RULE_BASED, RETStrategyType.OPC_MODEL_BASED
        ):
            warnings.append(
                f"最小 CD={metadata.min_cd_nm:.0f}nm 极小，OPC 策略可能效果有限，建议升级至 ILT。"
            )

        if strategy_type == RETStrategyType.SMO_ILT:
            warnings.append("SMO+ILT 计算开销极大，预计运行时间较长。")

        if metadata.periodicity_score > 0.8 and strategy_type.complexity_level > 5:
            warnings.append(
                f"高周期性区域（{metadata.periodicity_score:.2f}），使用更简单的 OPC+SRAF 可能性价比更高。"
            )

        return warnings

    def _check_strategy_consistency(
        self,
        results: List[StrategyMatchResult],
    ) -> None:
        """
        检查策略一致性，检测潜在的策略冲突

        Args:
            results: 匹配结果列表
        """
        region_types = {}
        for r in results:
            rt = r.strategy_config.strategy_type
            region_types.setdefault(rt, 0)
            region_types[rt] += 1

        if len(region_types) > 4:
            logger.warning(
                f"检测到 {len(region_types)} 种不同的 RET 策略，"
                "可能增加掩模制造复杂度。建议尽量统一邻近区域的策略。"
            )

        total_complexity = sum(
            r.strategy_config.strategy_type.complexity_level
            for r in results
        )
        avg_complexity = total_complexity / len(results) if results else 0

        if avg_complexity > 5:
            logger.info(
                f"平均策略复杂度 {avg_complexity:.1f} 较高，"
                "预计总计算时间较长，请确保计算资源充足。"
            )

    def estimate_total_runtime(
        self,
        results: List[StrategyMatchResult],
        regions: List[ChipRegion],
    ) -> float:
        """
        估算总运行时间（相对单位）

        Args:
            results: 匹配结果列表
            regions: 区域列表

        Returns:
            估算的总运行时间因子
        """
        total_factor = 0.0

        for result, region in zip(results, regions):
            strategy_config = result.strategy_config
            area = region.metadata.area_um2

            runtime_factor = strategy_config.estimated_runtime_factor
            total_factor += runtime_factor * area

        return total_factor

    def get_strategy_summary(
        self,
        results: List[StrategyMatchResult],
    ) -> Dict[str, Any]:
        """
        获取策略分配统计摘要

        Args:
            results: 匹配结果列表

        Returns:
            策略分配统计
        """
        strategy_counts: Dict[str, int] = {}
        region_type_counts: Dict[str, Dict[str, int]] = {}
        avg_confidence = 0.0

        for r in results:
            st = r.strategy_config.strategy_type.value
            strategy_counts[st] = strategy_counts.get(st, 0) + 1
            avg_confidence += r.confidence

        avg_confidence = avg_confidence / len(results) if results else 0.0

        return {
            'total_regions': len(results),
            'strategy_distribution': strategy_counts,
            'average_confidence': avg_confidence,
            'num_warnings': sum(len(r.warnings) for r in results),
        }
