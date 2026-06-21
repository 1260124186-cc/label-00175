# -*- coding: utf-8 -*-
"""
RET 策略推荐引擎

基于版图特征 + 历史实验知识库 + 规则引擎，自动推荐
应采用的 RET 流程组合及初始参数。

推荐决策逻辑：
    1. 从输入版图提取特征 (LayoutFeatureExtractor)
    2. 基于特征在知识库中查找相似实验 (RETKnowledgeBase)
    3. 应用规则引擎进行策略决策：
       - k1 因子判定：k1 < 0.4 → 需要 ILT/SMO+ILT
       - 拐角密度判定：高拐角密度 → 倾向 ILT
       - 周期性判定：高周期性 → OPC/SRAF 通常足够
       - 高频能量判定：高频能量占比高 → 倾向 ILT
       - 技术节点判定：EUV → 倾向 ILT
    4. 结合知识库匹配结果和规则引擎输出，综合评分
    5. 输出推荐策略 + 初始参数 + 备选方案
"""

import numpy as np
from typing import Optional, Dict, Any, List
import logging

from advisor.schemas import (
    RETStrategy, LayoutFeatures, RETRecommendation,
    RETRecommendationResult, ExperimentRecord,
)
from advisor.feature_extractor import LayoutFeatureExtractor
from advisor.knowledge_base import RETKnowledgeBase

logger = logging.getLogger(__name__)


class _RuleScores:
    opc_only: float = 0.0
    opc_sraf: float = 0.0
    ilt: float = 0.0
    smo_ilt: float = 0.0


class RETRecommendationEngine:
    """
    RET 策略推荐引擎

    核心方法 recommend()：输入版图掩模 → 输出 RET 推荐结果

    推荐决策三要素：
    1. 规则引擎：基于 k1 因子、拐角密度、周期性、高频能量比的启发式规则
    2. 知识库匹配：查找特征相似的历史实验，参考已知效果
    3. 综合评分：加权融合规则评分和知识库评分

    使用方法：
        engine = RETRecommendationEngine()
        result = engine.recommend(mask, pixel_size=1.0)
        print(result.primary.strategy)  # RETStrategy.ILT
    """

    RULE_WEIGHT = 0.5
    KB_WEIGHT = 0.5

    def __init__(
        self,
        knowledge_base: Optional[RETKnowledgeBase] = None,
        rule_weight: float = 0.5,
        kb_weight: float = 0.5,
    ):
        """
        初始化推荐引擎

        Args:
            knowledge_base: 历史实验知识库，None 则使用默认内置数据
            rule_weight: 规则引擎评分权重
            kb_weight: 知识库匹配评分权重
        """
        self._kb = knowledge_base or RETKnowledgeBase()
        self._rule_weight = rule_weight
        self._kb_weight = kb_weight

    def recommend(
        self,
        mask: np.ndarray,
        pixel_size: float = 1.0,
        technology_node: str = 'duv_arf',
        wavelength: float = 193.0,
        na: float = 1.35,
        user_preference: Optional[str] = None,
    ) -> RETRecommendationResult:
        """
        从输入版图推荐 RET 策略

        Args:
            mask: 二值掩模图案
            pixel_size: 像素尺寸 (nm)
            technology_node: 技术节点 ('duv_arf' / 'euv')
            wavelength: 光源波长 (nm)
            na: 数值孔径
            user_preference: 用户偏好策略 ('speed' / 'quality' / 'balanced')

        Returns:
            RETRecommendationResult 推荐结果
        """
        features = LayoutFeatureExtractor.extract(
            mask, pixel_size=pixel_size,
            technology_node=technology_node,
            wavelength=wavelength, na=na,
        )

        rule_scores = self._apply_rules(features)
        kb_scores = self._match_knowledge_base(features, technology_node)

        combined = self._combine_scores(rule_scores, kb_scores, user_preference)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        primary_strategy = ranked[0][0]
        primary_score = ranked[0][1]

        best_record = self._kb.find_best_strategy(features, strategy=primary_strategy.value)

        primary = self._build_recommendation(
            primary_strategy, features, best_record, primary_score
        )

        alternatives = []
        for strategy, score in ranked[1:]:
            if score < 0.05:
                continue
            alt_record = self._kb.find_best_strategy(features, strategy=strategy.value)
            alt_rec = self._build_recommendation(strategy, features, alt_record, score)
            alternatives.append(alt_rec)

        matched_ids = []
        similar = self._kb.find_similar(features, top_k=5, technology_filter=technology_node)
        for rec, sim in similar:
            if sim > 0.5:
                matched_ids.append(rec.id)

        warnings = self._generate_warnings(features, primary_strategy)

        return RETRecommendationResult(
            primary=primary,
            alternatives=alternatives,
            features=features,
            matched_experiments=matched_ids,
            warnings=warnings,
        )

    def recommend_from_features(
        self,
        features: LayoutFeatures,
        user_preference: Optional[str] = None,
    ) -> RETRecommendationResult:
        """
        从已有版图特征推荐 RET 策略（跳过特征提取步骤）

        Args:
            features: 已提取的版图特征
            user_preference: 用户偏好策略

        Returns:
            RETRecommendationResult 推荐结果
        """
        tech = features.technology_node
        rule_scores = self._apply_rules(features)
        kb_scores = self._match_knowledge_base(features, tech)
        combined = self._combine_scores(rule_scores, kb_scores, user_preference)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        primary_strategy = ranked[0][0]
        primary_score = ranked[0][1]

        best_record = self._kb.find_best_strategy(features, strategy=primary_strategy.value)
        primary = self._build_recommendation(
            primary_strategy, features, best_record, primary_score
        )

        alternatives = []
        for strategy, score in ranked[1:]:
            if score < 0.05:
                continue
            alt_record = self._kb.find_best_strategy(features, strategy=strategy.value)
            alt_rec = self._build_recommendation(strategy, features, alt_record, score)
            alternatives.append(alt_rec)

        matched_ids = []
        similar = self._kb.find_similar(features, top_k=5, technology_filter=tech)
        for rec, sim in similar:
            if sim > 0.5:
                matched_ids.append(rec.id)

        warnings = self._generate_warnings(features, primary_strategy)

        return RETRecommendationResult(
            primary=primary,
            alternatives=alternatives,
            features=features,
            matched_experiments=matched_ids,
            warnings=warnings,
        )

    def _apply_rules(self, features: LayoutFeatures) -> Dict[RETStrategy, float]:
        """
        应用规则引擎计算各策略评分

        规则体系基于光刻领域经验知识：
        - k1 < 0.35: 极高分辨率需求，必须 SMO+ILT
        - k1 < 0.5:  高分辨率需求，至少 ILT
        - k1 < 0.7:  中等需求，OPC+SRAF 可满足
        - k1 >= 0.7: 低分辨率需求，纯 OPC 足够

        辅助规则：
        - 拐角密度 > 0.10: 拐角校正需求强，倾向 ILT
        - 高频能量比 > 0.25: 版图复杂度高，倾向 ILT
        - 周期性 > 0.7: 周期性结构，OPC/SRAF 效果好

        Args:
            features: 版图特征

        Returns:
            {strategy: rule_score} 各策略的规则评分
        """
        scores = {
            RETStrategy.OPC_ONLY: 0.5,
            RETStrategy.OPC_SRAF: 0.5,
            RETStrategy.ILT: 0.5,
            RETStrategy.SMO_ILT: 0.5,
        }

        k1 = features.k1_factor()

        if k1 < 0.35:
            scores[RETStrategy.SMO_ILT] += 0.5
            scores[RETStrategy.ILT] += 0.3
            scores[RETStrategy.OPC_SRAF] += 0.0
            scores[RETStrategy.OPC_ONLY] -= 0.4
        elif k1 < 0.45:
            scores[RETStrategy.ILT] += 0.4
            scores[RETStrategy.SMO_ILT] += 0.3
            scores[RETStrategy.OPC_SRAF] += 0.1
            scores[RETStrategy.OPC_ONLY] -= 0.3
        elif k1 < 0.55:
            scores[RETStrategy.OPC_SRAF] += 0.3
            scores[RETStrategy.ILT] += 0.2
            scores[RETStrategy.OPC_ONLY] -= 0.1
        elif k1 < 0.7:
            scores[RETStrategy.OPC_SRAF] += 0.2
            scores[RETStrategy.OPC_ONLY] += 0.1
        else:
            scores[RETStrategy.OPC_ONLY] += 0.4
            scores[RETStrategy.OPC_SRAF] += 0.1
            scores[RETStrategy.ILT] -= 0.1
            scores[RETStrategy.SMO_ILT] -= 0.2

        if features.corner_density > 0.12:
            scores[RETStrategy.ILT] += 0.3
            scores[RETStrategy.SMO_ILT] += 0.2
            scores[RETStrategy.OPC_ONLY] -= 0.2
        elif features.corner_density > 0.06:
            scores[RETStrategy.ILT] += 0.15
            scores[RETStrategy.OPC_SRAF] += 0.1
        elif features.corner_density < 0.02:
            scores[RETStrategy.OPC_ONLY] += 0.1
            scores[RETStrategy.OPC_SRAF] += 0.05

        hf_ratio = features.spectral.high_freq_energy_ratio
        if hf_ratio > 0.30:
            scores[RETStrategy.ILT] += 0.25
            scores[RETStrategy.SMO_ILT] += 0.15
            scores[RETStrategy.OPC_ONLY] -= 0.15
        elif hf_ratio > 0.18:
            scores[RETStrategy.OPC_SRAF] += 0.1
            scores[RETStrategy.ILT] += 0.1
        elif hf_ratio < 0.08:
            scores[RETStrategy.OPC_ONLY] += 0.1

        if features.periodicity_score > 0.7:
            scores[RETStrategy.OPC_ONLY] += 0.15
            scores[RETStrategy.OPC_SRAF] += 0.1
        elif features.periodicity_score < 0.3:
            scores[RETStrategy.ILT] += 0.15
            scores[RETStrategy.SMO_ILT] += 0.1

        if features.technology_node == 'euv':
            scores[RETStrategy.ILT] += 0.2
            scores[RETStrategy.SMO_ILT] += 0.15
            scores[RETStrategy.OPC_ONLY] -= 0.2
            scores[RETStrategy.OPC_SRAF] -= 0.1

        if features.duty_cycle < 0.25 or features.duty_cycle > 0.75:
            scores[RETStrategy.OPC_SRAF] += 0.1
            scores[RETStrategy.ILT] += 0.05

        for key in scores:
            scores[key] = max(0.0, min(1.0, scores[key]))

        return scores

    def _match_knowledge_base(
        self,
        features: LayoutFeatures,
        technology_node: str,
    ) -> Dict[RETStrategy, float]:
        """
        基于知识库匹配计算各策略评分

        对每种策略查找最相似的历史实验，
        用相似度 × 效果改善率作为知识库评分。

        Args:
            features: 版图特征
            technology_node: 技术节点

        Returns:
            {strategy: kb_score}
        """
        strategy_map = {
            'opc_only': RETStrategy.OPC_ONLY,
            'opc_sraf': RETStrategy.OPC_SRAF,
            'ilt': RETStrategy.ILT,
            'smo_ilt': RETStrategy.SMO_ILT,
        }

        scores = {s: 0.0 for s in RETStrategy}

        for strategy_name, strategy_enum in strategy_map.items():
            best = self._kb.find_best_strategy(features, strategy=strategy_name)
            if best is not None:
                similar = self._kb.find_similar(features, top_k=10, technology_filter=technology_node)
                sim_score = 0.0
                for rec, sim in similar:
                    if rec.strategy == strategy_name:
                        quality = sim * (best.epe_improvement_pct / 100.0)
                        if quality > sim_score:
                            sim_score = quality
                scores[strategy_enum] = sim_score

        max_score = max(scores.values()) if scores else 1.0
        if max_score > 1e-8:
            for key in scores:
                scores[key] = scores[key] / max_score

        return scores

    def _combine_scores(
        self,
        rule_scores: Dict[RETStrategy, float],
        kb_scores: Dict[RETStrategy, float],
        user_preference: Optional[str] = None,
    ) -> Dict[RETStrategy, float]:
        """
        综合规则评分和知识库评分

        综合评分 = rule_weight × rule_score + kb_weight × kb_score

        如果用户指定偏好，调整权重：
        - 'speed': 倾向 OPC 类策略（更快）
        - 'quality': 倾向 ILT/SMO 类策略（更好）
        - 'balanced': 不调整

        Args:
            rule_scores: 规则引擎评分
            kb_scores: 知识库评分
            user_preference: 用户偏好

        Returns:
            {strategy: combined_score}
        """
        combined = {}
        for strategy in RETStrategy:
            r = rule_scores.get(strategy, 0.0)
            k = kb_scores.get(strategy, 0.0)
            combined[strategy] = self._rule_weight * r + self._kb_weight * k

        if user_preference == 'speed':
            combined[RETStrategy.OPC_ONLY] *= 1.3
            combined[RETStrategy.OPC_SRAF] *= 1.2
            combined[RETStrategy.ILT] *= 0.8
            combined[RETStrategy.SMO_ILT] *= 0.7
        elif user_preference == 'quality':
            combined[RETStrategy.OPC_ONLY] *= 0.7
            combined[RETStrategy.OPC_SRAF] *= 0.8
            combined[RETStrategy.ILT] *= 1.2
            combined[RETStrategy.SMO_ILT] *= 1.3

        return combined

    def _build_recommendation(
        self,
        strategy: RETStrategy,
        features: LayoutFeatures,
        best_record: Optional[ExperimentRecord],
        score: float,
    ) -> RETRecommendation:
        """
        构建推荐结果

        Args:
            strategy: 推荐策略
            features: 版图特征
            best_record: 最佳匹配历史实验
            score: 综合评分

        Returns:
            RETRecommendation 推荐结果
        """
        opc_params = self._generate_opc_params(strategy, features, best_record)
        ilt_params = self._generate_ilt_params(strategy, features, best_record)
        smo_params = self._generate_smo_params(strategy, features, best_record)
        optical_hints = self._generate_optical_hints(strategy, features, best_record)

        reason = self._generate_reason(strategy, features, best_record)

        return RETRecommendation(
            strategy=strategy,
            confidence=min(score, 1.0),
            reason=reason,
            opc_params=opc_params,
            ilt_params=ilt_params,
            smo_params=smo_params,
            optical_system_hints=optical_hints,
        )

    def _generate_opc_params(
        self,
        strategy: RETStrategy,
        features: LayoutFeatures,
        record: Optional[ExperimentRecord],
    ) -> Dict[str, Any]:
        """生成 OPC 初始参数"""
        params: Dict[str, Any] = {}

        if record and record.opc_params:
            params.update(record.opc_params)

        k1 = features.k1_factor()

        if 'max_iterations' not in params:
            if k1 < 0.5:
                params['max_iterations'] = 15
            elif k1 < 0.7:
                params['max_iterations'] = 10
            else:
                params['max_iterations'] = 8

        if 'epe_threshold' not in params:
            if k1 < 0.5:
                params['epe_threshold'] = 2.0
            else:
                params['epe_threshold'] = 3.0

        if strategy in (RETStrategy.OPC_SRAF, RETStrategy.ILT, RETStrategy.SMO_ILT):
            params.setdefault('sraf_enable', True)
            params.setdefault('sraf_width', 1.0)
            params.setdefault('sraf_length', 4.0)
            params.setdefault('sraf_min_distance', 2.0)
            params.setdefault('sraf_max_distance', 5.0)
        else:
            params.setdefault('sraf_enable', False)

        params.setdefault('edge_offset_step', 0.5)
        params.setdefault('max_edge_offset', 3.0)
        params.setdefault('corner_bias_size', 1.0)
        params.setdefault('optimizer_enable', True)
        params.setdefault('wafer_threshold', 0.3)

        return params

    def _generate_ilt_params(
        self,
        strategy: RETStrategy,
        features: LayoutFeatures,
        record: Optional[ExperimentRecord],
    ) -> Dict[str, Any]:
        """生成 ILT 初始参数"""
        if strategy not in (RETStrategy.ILT, RETStrategy.SMO_ILT):
            return {}

        params: Dict[str, Any] = {}

        if record and record.ilt_params:
            params.update(record.ilt_params)

        params.setdefault('max_iter', 200)
        params.setdefault('learning_rate', 0.01)
        params.setdefault('optimizer_type', 'adam_projection')
        params.setdefault('transmission_level', 'continuous')
        params.setdefault('quantization_start_iter', 100)
        params.setdefault('quantization_schedule', 'linear')
        params.setdefault('quantization_strength', 1.0)
        params.setdefault('wafer_threshold', 0.3)
        params.setdefault('l2_wafer_weight', 1.0)
        params.setdefault('binary_penalty_weight', 0.01)
        params.setdefault('resist_steepness', 50.0)

        k1 = features.k1_factor()
        if k1 < 0.4:
            params['max_iter'] = max(params.get('max_iter', 200), 250)
            params['learning_rate'] = min(params.get('learning_rate', 0.01), 0.008)
            params['binary_penalty_weight'] = max(
                params.get('binary_penalty_weight', 0.01), 0.02
            )

        if features.corner_density > 0.10:
            params.setdefault('tv_smooth_weight', 0.001)

        return params

    def _generate_smo_params(
        self,
        strategy: RETStrategy,
        features: LayoutFeatures,
        record: Optional[ExperimentRecord],
    ) -> Dict[str, Any]:
        """生成 SMO 初始参数"""
        if strategy != RETStrategy.SMO_ILT:
            return {}

        params: Dict[str, Any] = {}

        if record and record.smo_params:
            params.update(record.smo_params)

        params.setdefault('strategy', 'alternating')
        params.setdefault('max_outer_iterations', 15)
        params.setdefault('source_max_iter', 50)
        params.setdefault('mask_max_iter', 100)
        params.setdefault('source_learning_rate', 0.005)
        params.setdefault('mask_learning_rate', 0.01)
        params.setdefault('wafer_threshold', 0.3)

        if features.periodicity_score > 0.7:
            params.setdefault('source_init_type', 'annular')
        elif features.corner_density > 0.08:
            params.setdefault('source_init_type', 'quasar')
        else:
            params.setdefault('source_init_type', 'conventional')

        k1 = features.k1_factor()
        if k1 < 0.4:
            params['max_outer_iterations'] = max(
                params.get('max_outer_iterations', 15), 20
            )

        return params

    def _generate_optical_hints(
        self,
        strategy: RETStrategy,
        features: LayoutFeatures,
        record: Optional[ExperimentRecord],
    ) -> Dict[str, Any]:
        """生成光学系统提示参数"""
        hints: Dict[str, Any] = {}

        if features.technology_node == 'euv':
            hints['illumination_type'] = 'annular'
            hints['source_params'] = {'sigma_inner': 0.5, 'sigma_outer': 0.8}
        else:
            if strategy in (RETStrategy.ILT, RETStrategy.SMO_ILT):
                if features.periodicity_score > 0.7:
                    hints['illumination_type'] = 'annular'
                    hints['source_params'] = {'sigma_inner': 0.5, 'sigma_outer': 0.8}
                elif features.corner_density > 0.08:
                    hints['illumination_type'] = 'quasar'
                    hints['source_params'] = {
                        'sigma_inner': 0.5, 'sigma_outer': 0.8,
                        'angle': 45.0, 'opening_angle': 30.0
                    }
                else:
                    hints['illumination_type'] = 'conventional'
            else:
                hints['illumination_type'] = 'conventional'

        if features.k1_factor() < 0.45:
            hints['tcc_mode'] = 'socs'
            hints['socs_num_terms'] = 8
        else:
            hints['tcc_mode'] = 'socs'
            hints['socs_num_terms'] = 5

        return hints

    def _generate_reason(
        self,
        strategy: RETStrategy,
        features: LayoutFeatures,
        record: Optional[ExperimentRecord],
    ) -> str:
        """生成推荐原因说明"""
        k1 = features.k1_factor()
        parts = []

        if k1 < 0.35:
            parts.append(f"k1={k1:.2f}（极低，需要最强 RET）")
        elif k1 < 0.45:
            parts.append(f"k1={k1:.2f}（低，需要高强度 RET）")
        elif k1 < 0.55:
            parts.append(f"k1={k1:.2f}（中等偏低）")
        elif k1 < 0.7:
            parts.append(f"k1={k1:.2f}（中等）")
        else:
            parts.append(f"k1={k1:.2f}（较高，常规 RET 即可）")

        if features.corner_density > 0.08:
            parts.append(f"拐角密度较高({features.corner_density:.2f})")
        if features.spectral.high_freq_energy_ratio > 0.20:
            parts.append(f"高频能量占比高({features.spectral.high_freq_energy_ratio:.2f})")
        if features.periodicity_score > 0.7:
            parts.append(f"版图周期性强({features.periodicity_score:.2f})")
        elif features.periodicity_score < 0.3:
            parts.append(f"版图非周期性({features.periodicity_score:.2f})")

        strategy_names = {
            RETStrategy.OPC_ONLY: '纯 OPC',
            RETStrategy.OPC_SRAF: 'OPC + SRAF',
            RETStrategy.ILT: 'ILT',
            RETStrategy.SMO_ILT: 'SMO + ILT',
        }

        reason = f"推荐 {strategy_names[strategy]}：" + "，".join(parts) + "。"

        if record:
            reason += (
                f" 历史实验 {record.id} 显示该策略在相似场景下"
                f" EPE 改善率 {record.epe_improvement_pct:.0f}%，"
                f"最终 EPE {record.final_epe_nm:.1f} nm。"
            )

        return reason

    def _generate_warnings(
        self,
        features: LayoutFeatures,
        strategy: RETStrategy,
    ) -> List[str]:
        """生成警告信息"""
        warnings = []
        k1 = features.k1_factor()

        if k1 < 0.3:
            warnings.append(
                f"k1={k1:.2f} 极低，当前 RET 策略可能无法完全满足要求，"
                "建议考虑多重图形技术 (MPT) 或工艺窗口增强。"
            )

        if features.min_cd_nm < 20 and features.technology_node == 'duv_arf':
            warnings.append(
                f"最小 CD={features.min_cd_nm:.0f}nm 在 DUV 下可能超出分辨能力，"
                "建议确认是否需要 EUV 或多重图形。"
            )

        if strategy == RETStrategy.OPC_ONLY and k1 < 0.5:
            warnings.append(
                "纯 OPC 在 k1<0.5 场景下效果有限，"
                "建议考虑 OPC+SRAF 或 ILT。"
            )

        if strategy == RETStrategy.SMO_ILT:
            warnings.append(
                "SMO+ILT 计算开销较大，建议预留充足计算资源。"
            )

        if features.corner_density > 0.15 and strategy in (
            RETStrategy.OPC_ONLY, RETStrategy.OPC_SRAF
        ):
            warnings.append(
                "高拐角密度版图使用规则 OPC 策略可能在拐角区域效果不理想，"
                "建议升级至 ILT。"
            )

        return warnings
