# -*- coding: utf-8 -*-
"""
RET 推荐引擎 API 路由

提供 REST API 端点，支持：
1. POST /api/advisor/recommend - 从版图推荐 RET 策略
2. POST /api/advisor/recommend-features - 从已有特征推荐 RET 策略
3. GET /api/advisor/knowledge-base - 查询历史实验知识库
"""

import logging
import sys
import os
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.schemas import (
    RETRecommendRequest,
    RETRecommendFromFeaturesRequest,
    RETRecommendResponse,
    OpticalSystem,
)
from advisor.recommendation_engine import RETRecommendationEngine
from advisor.knowledge_base import RETKnowledgeBase
from api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/advisor", tags=["RET推荐引擎"])

_engine_instance: Optional[RETRecommendationEngine] = None


def _get_engine() -> RETRecommendationEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RETRecommendationEngine()
    return _engine_instance


@router.post(
    "/recommend",
    response_model=RETRecommendResponse,
    summary="根据版图推荐 RET 策略",
)
async def recommend_ret(
    req: RETRecommendRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    从输入版图掩模自动推荐 RET 流程组合及初始参数。

    支持的版图类型：
    - line_space: 线/空间结构
    - contact_hole: 接触孔
    - l_shaped_corner: L 形拐角
    - t_junction: T 形结
    - sram_bitcell: SRAM 位单元
    - custom: 自定义图案（需提供 gds_file_id）
    """
    engine = _get_engine()

    from core.test_structures import generate_test_structure

    mask = _generate_mask_from_request(req)

    technology_node = req.optical_system.technology_node
    wavelength = req.optical_system.wavelength
    na = req.optical_system.na
    pixel_size = req.optical_system.pixel_size

    result = engine.recommend(
        mask=mask,
        pixel_size=pixel_size,
        technology_node=technology_node,
        wavelength=wavelength,
        na=na,
        user_preference=req.user_preference,
    )

    return RETRecommendResponse(
        success=True,
        message="RET 策略推荐完成",
        recommendation=result.to_dict(),
    )


@router.post(
    "/recommend-features",
    response_model=RETRecommendResponse,
    summary="根据版图特征推荐 RET 策略",
)
async def recommend_ret_from_features(
    req: RETRecommendFromFeaturesRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    从已有的版图特征（无需提供掩模）推荐 RET 策略。

    适用于用户已知版图特征但不想重新提取的场景。
    """
    engine = _get_engine()

    from advisor.schemas import LayoutFeatures, SpectralFeatures

    spectral = SpectralFeatures(
        dominant_frequency=req.features.get('spectral', {}).get('dominant_frequency', 0.0),
        bandwidth_3db=req.features.get('spectral', {}).get('bandwidth_3db', 0.0),
        spectral_entropy=req.features.get('spectral', {}).get('spectral_entropy', 0.0),
        high_freq_energy_ratio=req.features.get('spectral', {}).get('high_freq_energy_ratio', 0.0),
        low_freq_energy_ratio=req.features.get('spectral', {}).get('low_freq_energy_ratio', 0.0),
        peak_count=req.features.get('spectral', {}).get('peak_count', 0),
        spectral_centroid=req.features.get('spectral', {}).get('spectral_centroid', 0.0),
    )

    features = LayoutFeatures(
        min_cd_nm=req.features.get('min_cd_nm', 45.0),
        corner_density=req.features.get('corner_density', 0.0),
        periodicity_score=req.features.get('periodicity_score', 0.0),
        dominant_pitch_nm=req.features.get('dominant_pitch_nm', 0.0),
        duty_cycle=req.features.get('duty_cycle', 0.5),
        fill_ratio=req.features.get('fill_ratio', 0.5),
        spectral=spectral,
        technology_node=req.features.get('technology_node', 'duv_arf'),
        wavelength=req.features.get('wavelength', 193.0),
        na=req.features.get('na', 1.35),
        pixel_size=req.features.get('pixel_size', 1.0),
        image_shape=tuple(req.features.get('image_shape', [0, 0])),
    )

    result = engine.recommend_from_features(
        features=features,
        user_preference=req.user_preference,
    )

    return RETRecommendResponse(
        success=True,
        message="RET 策略推荐完成",
        recommendation=result.to_dict(),
    )


@router.get(
    "/knowledge-base",
    summary="查询历史实验知识库",
)
async def get_knowledge_base(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """查询内置历史实验知识库的所有记录"""
    engine = _get_engine()
    records = engine._kb.get_records()
    return {
        'success': True,
        'count': len(records),
        'records': [r.to_dict() for r in records],
    }


def _generate_mask_from_request(req: RETRecommendRequest):
    """从请求参数生成版图掩模"""
    from core.test_structures import generate_test_structure

    pattern_type = req.pattern_type
    pattern_params = dict(req.pattern_params)

    grid_size = pattern_params.get('size', [128, 128])
    if isinstance(grid_size, list):
        grid_size = tuple(grid_size)

    pixel_size = req.optical_system.pixel_size

    params: Dict[str, Any] = {
        'grid_size': grid_size,
        'pixel_size': pixel_size,
    }

    if 'cd' in pattern_params:
        params['cd'] = pattern_params['cd']
    if 'pitch' in pattern_params:
        params['pitch'] = pattern_params['pitch']

    type_map = {
        'line_space': 'line_space',
        'contact_hole': 'contact_hole',
        'l_shaped_corner': 'l_shaped_corner',
        't_junction': 't_junction',
        'sram_bitcell': 'sram_bitcell',
    }

    params['structure_type'] = type_map.get(pattern_type, 'line_space')

    if pattern_type == 'contact_hole':
        params.setdefault('cd', 50.0)
        params.setdefault('pitch', 100.0)
        params['hole_shape'] = pattern_params.get('hole_shape', 'circle')
    elif pattern_type == 'line_space':
        params.setdefault('cd', 45.0)
        params.setdefault('pitch', 90.0)
        params['orientation'] = pattern_params.get('orientation', 'horizontal')
    elif pattern_type == 'l_shaped_corner':
        params.setdefault('cd', 50.0)
        params.setdefault('pitch', 150.0)
        params['arm_length'] = pattern_params.get('arm_length', 200.0)
        params['corner_type'] = pattern_params.get('corner_type', 'inner')
    elif pattern_type == 't_junction':
        params.setdefault('cd', 45.0)
        params.setdefault('pitch', 150.0)
        params['stem_length'] = pattern_params.get('stem_length', 200.0)
        params['branch_length'] = pattern_params.get('branch_length', 100.0)
    elif pattern_type == 'sram_bitcell':
        params.setdefault('cd', 30.0)
        params.setdefault('pitch', 90.0)

    return generate_test_structure(params)
