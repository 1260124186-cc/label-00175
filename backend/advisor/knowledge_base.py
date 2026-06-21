# -*- coding: utf-8 -*-
"""
RET 历史实验知识库

内置典型版图类型的历史实验数据，包含各 RET 策略的已知效果，
支持基于特征相似度的最近邻匹配。

知识库内容覆盖：
    - DUV ArF (193nm, NA=1.35) 典型场景
    - EUV (13.5nm, NA=0.33) 典型场景
    - 不同 CD / 间距 / 拐角密度 / 周期性组合
    - 四种 RET 策略的已知效果
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging

from advisor.schemas import ExperimentRecord

logger = logging.getLogger(__name__)


def _build_default_knowledge_base() -> List[ExperimentRecord]:
    """
    构建默认的历史实验知识库

    覆盖以下典型场景：
    1. 大 CD 周期性线/空间 (L/S) → 纯 OPC 足够
    2. 中等 CD 周期性 L/S → OPC + SRAF
    3. 小 CD 周期性 L/S → ILT 或 SMO+ILT
    4. 接触孔阵列 → OPC + SRAF 或 ILT
    5. L 形拐角密集 → ILT
    6. SRAM 高密度 → SMO + ILT
    7. EUV 小 CD → ILT
    8. EUV 接触孔 → SMO + ILT
    """
    records = []

    records.append(ExperimentRecord(
        id='duv_ls_cd80_pitch160', layout_type='line_space',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=80.0, corner_density=0.01, periodicity_score=0.95,
        high_freq_energy_ratio=0.05, strategy='opc_only',
        final_epe_nm=1.2, epe_improvement_pct=85.0, convergence=True,
        total_time_sec=15.0,
        opc_params={'epe_threshold': 3.0, 'max_iterations': 8, 'sraf_enable': False,
                     'edge_offset_step': 0.5, 'optimizer_enable': True},
    ))

    records.append(ExperimentRecord(
        id='duv_ls_cd60_pitch120', layout_type='line_space',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=60.0, corner_density=0.01, periodicity_score=0.93,
        high_freq_energy_ratio=0.08, strategy='opc_only',
        final_epe_nm=1.8, epe_improvement_pct=78.0, convergence=True,
        total_time_sec=18.0,
        opc_params={'epe_threshold': 3.0, 'max_iterations': 10, 'sraf_enable': False,
                     'edge_offset_step': 0.5, 'corner_bias_size': 1.0},
    ))

    records.append(ExperimentRecord(
        id='duv_ls_cd45_pitch90', layout_type='line_space',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=45.0, corner_density=0.02, periodicity_score=0.90,
        high_freq_energy_ratio=0.12, strategy='opc_sraf',
        final_epe_nm=1.5, epe_improvement_pct=88.0, convergence=True,
        total_time_sec=30.0,
        opc_params={'epe_threshold': 2.5, 'max_iterations': 12, 'sraf_enable': True,
                     'sraf_width': 1.0, 'sraf_length': 4.0, 'sraf_min_distance': 2.0,
                     'sraf_max_distance': 5.0, 'optimizer_enable': True},
    ))

    records.append(ExperimentRecord(
        id='duv_ls_cd38_pitch76', layout_type='line_space',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=38.0, corner_density=0.02, periodicity_score=0.88,
        high_freq_energy_ratio=0.18, strategy='opc_sraf',
        final_epe_nm=2.0, epe_improvement_pct=82.0, convergence=True,
        total_time_sec=40.0,
        opc_params={'epe_threshold': 2.0, 'max_iterations': 15, 'sraf_enable': True,
                     'sraf_width': 0.8, 'sraf_length': 3.0, 'sraf_min_distance': 1.5,
                     'sraf_max_distance': 4.0, 'optimizer_enable': True},
    ))

    records.append(ExperimentRecord(
        id='duv_ls_cd32_pitch64', layout_type='line_space',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=32.0, corner_density=0.03, periodicity_score=0.85,
        high_freq_energy_ratio=0.25, strategy='ilt',
        final_epe_nm=1.2, epe_improvement_pct=92.0, convergence=True,
        total_time_sec=120.0,
        ilt_params={'max_iter': 200, 'learning_rate': 0.01, 'optimizer_type': 'adam_projection',
                     'transmission_level': 'continuous', 'quantization_start_iter': 100,
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.01},
    ))

    records.append(ExperimentRecord(
        id='duv_ls_cd28_pitch56', layout_type='line_space',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=28.0, corner_density=0.03, periodicity_score=0.82,
        high_freq_energy_ratio=0.30, strategy='smo_ilt',
        final_epe_nm=0.8, epe_improvement_pct=95.0, convergence=True,
        total_time_sec=300.0,
        ilt_params={'max_iter': 200, 'learning_rate': 0.01, 'optimizer_type': 'adam_projection',
                     'transmission_level': 'continuous', 'quantization_start_iter': 100,
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.01},
        smo_params={'strategy': 'alternating', 'max_outer_iterations': 15,
                     'source_max_iter': 50, 'mask_max_iter': 100,
                     'source_init_type': 'annular', 'source_learning_rate': 0.005},
    ))

    records.append(ExperimentRecord(
        id='duv_contact_cd50_pitch100', layout_type='contact_hole',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=50.0, corner_density=0.08, periodicity_score=0.80,
        high_freq_energy_ratio=0.20, strategy='opc_sraf',
        final_epe_nm=2.5, epe_improvement_pct=75.0, convergence=True,
        total_time_sec=45.0,
        opc_params={'epe_threshold': 3.0, 'max_iterations': 12, 'sraf_enable': True,
                     'sraf_width': 1.0, 'sraf_length': 2.0, 'sraf_min_distance': 2.0,
                     'corner_bias_size': 1.5, 'optimizer_enable': True},
    ))

    records.append(ExperimentRecord(
        id='duv_contact_cd40_pitch80', layout_type='contact_hole',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=40.0, corner_density=0.10, periodicity_score=0.78,
        high_freq_energy_ratio=0.28, strategy='ilt',
        final_epe_nm=1.5, epe_improvement_pct=90.0, convergence=True,
        total_time_sec=150.0,
        ilt_params={'max_iter': 250, 'learning_rate': 0.008, 'optimizer_type': 'adam_projection',
                     'transmission_level': 'continuous', 'quantization_start_iter': 120,
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.02,
                     'tv_smooth_weight': 0.001},
    ))

    records.append(ExperimentRecord(
        id='duv_lcorner_cd50', layout_type='l_shaped_corner',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=50.0, corner_density=0.15, periodicity_score=0.10,
        high_freq_energy_ratio=0.35, strategy='ilt',
        final_epe_nm=1.8, epe_improvement_pct=88.0, convergence=True,
        total_time_sec=100.0,
        ilt_params={'max_iter': 200, 'learning_rate': 0.01, 'optimizer_type': 'adam_projection',
                     'transmission_level': 'continuous', 'quantization_start_iter': 80,
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.02},
    ))

    records.append(ExperimentRecord(
        id='duv_tjunction_cd45', layout_type='t_junction',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=45.0, corner_density=0.12, periodicity_score=0.15,
        high_freq_energy_ratio=0.30, strategy='ilt',
        final_epe_nm=1.6, epe_improvement_pct=86.0, convergence=True,
        total_time_sec=110.0,
        ilt_params={'max_iter': 200, 'learning_rate': 0.01, 'optimizer_type': 'adam_projection',
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.015},
    ))

    records.append(ExperimentRecord(
        id='duv_sram_cd30', layout_type='sram_bitcell',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=30.0, corner_density=0.12, periodicity_score=0.30,
        high_freq_energy_ratio=0.35, strategy='smo_ilt',
        final_epe_nm=1.0, epe_improvement_pct=93.0, convergence=True,
        total_time_sec=350.0,
        ilt_params={'max_iter': 250, 'learning_rate': 0.008, 'optimizer_type': 'adam_projection',
                     'transmission_level': 'continuous', 'quantization_start_iter': 100,
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.02},
        smo_params={'strategy': 'alternating', 'max_outer_iterations': 20,
                     'source_max_iter': 60, 'mask_max_iter': 120,
                     'source_init_type': 'quasar', 'source_learning_rate': 0.005},
    ))

    records.append(ExperimentRecord(
        id='euv_ls_cd20_pitch40', layout_type='line_space',
        technology_node='euv', wavelength=13.5, na=0.33,
        min_cd_nm=20.0, corner_density=0.02, periodicity_score=0.90,
        high_freq_energy_ratio=0.15, strategy='ilt',
        final_epe_nm=1.0, epe_improvement_pct=91.0, convergence=True,
        total_time_sec=130.0,
        ilt_params={'max_iter': 200, 'learning_rate': 0.01, 'optimizer_type': 'adam_projection',
                     'transmission_level': 'continuous', 'quantization_start_iter': 100,
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.01},
    ))

    records.append(ExperimentRecord(
        id='euv_ls_cd16_pitch32', layout_type='line_space',
        technology_node='euv', wavelength=13.5, na=0.33,
        min_cd_nm=16.0, corner_density=0.03, periodicity_score=0.85,
        high_freq_energy_ratio=0.22, strategy='ilt',
        final_epe_nm=0.9, epe_improvement_pct=93.0, convergence=True,
        total_time_sec=160.0,
        ilt_params={'max_iter': 250, 'learning_rate': 0.008, 'optimizer_type': 'adam_projection',
                     'transmission_level': 'continuous', 'quantization_start_iter': 120,
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.02},
    ))

    records.append(ExperimentRecord(
        id='euv_ls_cd14_pitch28', layout_type='line_space',
        technology_node='euv', wavelength=13.5, na=0.33,
        min_cd_nm=14.0, corner_density=0.03, periodicity_score=0.80,
        high_freq_energy_ratio=0.30, strategy='smo_ilt',
        final_epe_nm=0.6, epe_improvement_pct=96.0, convergence=True,
        total_time_sec=400.0,
        ilt_params={'max_iter': 250, 'learning_rate': 0.008, 'optimizer_type': 'adam_projection',
                     'transmission_level': 'continuous', 'quantization_start_iter': 120,
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.02},
        smo_params={'strategy': 'alternating', 'max_outer_iterations': 20,
                     'source_max_iter': 60, 'mask_max_iter': 120,
                     'source_init_type': 'annular', 'source_learning_rate': 0.005},
    ))

    records.append(ExperimentRecord(
        id='euv_contact_cd25_pitch50', layout_type='contact_hole',
        technology_node='euv', wavelength=13.5, na=0.33,
        min_cd_nm=25.0, corner_density=0.10, periodicity_score=0.75,
        high_freq_energy_ratio=0.25, strategy='ilt',
        final_epe_nm=1.5, epe_improvement_pct=89.0, convergence=True,
        total_time_sec=180.0,
        ilt_params={'max_iter': 250, 'learning_rate': 0.008, 'optimizer_type': 'adam_projection',
                     'transmission_level': 'continuous', 'quantization_start_iter': 100,
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.02,
                     'tv_smooth_weight': 0.001},
    ))

    records.append(ExperimentRecord(
        id='euv_contact_cd20_pitch40', layout_type='contact_hole',
        technology_node='euv', wavelength=13.5, na=0.33,
        min_cd_nm=20.0, corner_density=0.12, periodicity_score=0.70,
        high_freq_energy_ratio=0.32, strategy='smo_ilt',
        final_epe_nm=0.8, epe_improvement_pct=94.0, convergence=True,
        total_time_sec=420.0,
        ilt_params={'max_iter': 250, 'learning_rate': 0.008, 'optimizer_type': 'adam_projection',
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.02},
        smo_params={'strategy': 'alternating', 'max_outer_iterations': 20,
                     'source_max_iter': 60, 'mask_max_iter': 120,
                     'source_init_type': 'quasar', 'source_learning_rate': 0.005},
    ))

    records.append(ExperimentRecord(
        id='duv_ls_cd45_pitch90_ilt', layout_type='line_space',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=45.0, corner_density=0.02, periodicity_score=0.90,
        high_freq_energy_ratio=0.12, strategy='ilt',
        final_epe_nm=0.8, epe_improvement_pct=94.0, convergence=True,
        total_time_sec=90.0,
        ilt_params={'max_iter': 200, 'learning_rate': 0.01, 'optimizer_type': 'adam_projection',
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.01},
        notes='opc_sraf 也可达到 1.5nm EPE，但 ILT 可进一步改善到 0.8nm',
    ))

    records.append(ExperimentRecord(
        id='duv_mixed_cd40_corner', layout_type='mixed',
        technology_node='duv_arf', wavelength=193.0, na=1.35,
        min_cd_nm=40.0, corner_density=0.18, periodicity_score=0.25,
        high_freq_energy_ratio=0.38, strategy='smo_ilt',
        final_epe_nm=1.2, epe_improvement_pct=90.0, convergence=True,
        total_time_sec=380.0,
        ilt_params={'max_iter': 250, 'learning_rate': 0.008, 'optimizer_type': 'adam_projection',
                     'l2_wafer_weight': 1.0, 'binary_penalty_weight': 0.02},
        smo_params={'strategy': 'alternating', 'max_outer_iterations': 18,
                     'source_max_iter': 50, 'mask_max_iter': 100,
                     'source_init_type': 'annular', 'source_learning_rate': 0.005},
    ))

    return records


class RETKnowledgeBase:
    """
    RET 历史实验知识库

    内置典型版图类型的历史实验数据，支持：
    1. 基于特征相似度的最近邻匹配
    2. 按策略过滤最佳实验
    3. 查询特定场景的推荐参数

    使用方法：
        kb = RETKnowledgeBase()
        matches = kb.find_similar(features, top_k=5)
        best = kb.find_best_strategy(features)
    """

    def __init__(self, records: Optional[List[ExperimentRecord]] = None):
        """
        初始化知识库

        Args:
            records: 实验记录列表，None 则使用内置默认数据
        """
        self._records = records if records is not None else _build_default_knowledge_base()
        self._feature_matrix = None
        self._build_index()

    def _build_index(self):
        """构建特征向量索引"""
        if len(self._records) == 0:
            self._feature_matrix = np.zeros((0, 10), dtype=np.float64)
            return

        vectors = []
        for r in self._records:
            vectors.append(r.to_feature_vector())
        self._feature_matrix = np.stack(vectors, axis=0)

    def add_record(self, record: ExperimentRecord):
        """
        添加新的实验记录

        Args:
            record: 实验记录
        """
        self._records.append(record)
        vec = record.to_feature_vector().reshape(1, -1)
        if self._feature_matrix is None or self._feature_matrix.shape[0] == 0:
            self._feature_matrix = vec
        else:
            self._feature_matrix = np.concatenate([self._feature_matrix, vec], axis=0)

    def find_similar(
        self,
        features: 'LayoutFeatures',
        top_k: int = 5,
        technology_filter: Optional[str] = None,
    ) -> List[Tuple[ExperimentRecord, float]]:
        """
        查找特征相似的历史实验

        使用余弦相似度匹配输入特征与历史实验特征。

        Args:
            features: 输入版图特征
            top_k: 返回前 k 个最相似实验
            technology_filter: 技术节点过滤器 ('duv_arf' / 'euv')

        Returns:
            [(experiment_record, similarity_score), ...] 按相似度降序排列
        """
        from advisor.schemas import LayoutFeatures as LF
        query_vec = features.to_feature_vector()

        if self._feature_matrix.shape[0] == 0:
            return []

        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-10:
            return []

        db_norms = np.linalg.norm(self._feature_matrix, axis=1)
        valid = db_norms > 1e-10

        similarities = np.zeros(len(self._records), dtype=np.float64)
        if np.any(valid):
            cos_sim = np.dot(self._feature_matrix[valid], query_vec) / (
                db_norms[valid] * query_norm
            )
            similarities[valid] = cos_sim

        indices = np.argsort(-similarities)
        results = []

        for idx in indices:
            if len(results) >= top_k:
                break

            record = self._records[idx]
            sim = similarities[idx]

            if technology_filter is not None:
                if record.technology_node != technology_filter:
                    continue

            results.append((record, float(sim)))

        return results

    def find_best_strategy(
        self,
        features: 'LayoutFeatures',
        strategy: Optional[str] = None,
    ) -> Optional[ExperimentRecord]:
        """
        查找指定策略下最佳的历史实验

        Args:
            features: 输入版图特征
            strategy: 指定策略过滤 ('opc_only' / 'opc_sraf' / 'ilt' / 'smo_ilt')

        Returns:
            最佳匹配的实验记录，或 None
        """
        matches = self.find_similar(features, top_k=20)
        best = None
        best_score = -1.0

        for record, sim in matches:
            if strategy is not None and record.strategy != strategy:
                continue

            quality_score = sim * (record.epe_improvement_pct / 100.0)
            if quality_score > best_score:
                best_score = quality_score
                best = record

        return best

    def get_all_strategies_for_features(
        self,
        features: 'LayoutFeatures',
    ) -> Dict[str, Optional[ExperimentRecord]]:
        """
        为输入特征查找每种 RET 策略的最佳历史实验

        Args:
            features: 输入版图特征

        Returns:
            {strategy_name: best_record_or_None}
        """
        strategies = ['opc_only', 'opc_sraf', 'ilt', 'smo_ilt']
        result = {}
        for s in strategies:
            result[s] = self.find_best_strategy(features, strategy=s)
        return result

    def get_records(self) -> List[ExperimentRecord]:
        """获取所有实验记录"""
        return list(self._records)

    def size(self) -> int:
        """获取知识库大小"""
        return len(self._records)
