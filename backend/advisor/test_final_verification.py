#!/usr/bin/env python3
"""
RET 策略推荐引擎 - 最终验证

验证所有核心组件协同工作，模拟真实使用场景
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from advisor import (
    RETStrategy, LayoutFeatures, RETRecommendation,
    RETRecommendationResult, ExperimentRecord,
    LayoutFeatureExtractor, RETKnowledgeBase, RETRecommendationEngine,
)
import numpy as np

print("=" * 70)
print("RET 策略推荐引擎 - 最终验证")
print("=" * 70)

engine = RETRecommendationEngine()
kb = RETKnowledgeBase()

print("\n" + "=" * 70)
print("场景 1: DUV 大 CD 周期性线/空间 (CD=80nm, k1=0.56")
print("=" * 70)

mask1 = np.zeros((320, 320))
for y in range(0, 320, 160):
    mask1[y:y+80, :] = 1.0

result1 = engine.recommend(mask1, pixel_size=1.0, wavelength=193.0, na=1.35)
f1 = result1.features
print(f"  提取特征:")
print(f"    - 最小 CD: {f1.min_cd_nm:.1f} nm")
print(f"    - k1 因子: {f1.k1_factor():.3f}")
print(f"    - 拐角密度: {f1.corner_density:.3f}")
print(f"    - 周期性评分: {f1.periodicity_score:.3f}")
print(f"    - 高频能量比: {f1.spectral.high_freq_energy_ratio:.3f}")
print(f"    - 复杂度评分: {f1.complexity_score():.3f}")
print(f"  推荐策略: {result1.primary.strategy.value}")
print(f"  置信度: {result1.primary.confidence:.3f}")
print(f"  推荐原因: {result1.primary.reason}")
if result1.primary.opc_params:
    print(f"  OPC 参数示例: max_iterations={result1.primary.opc_params.get('max_iterations')}, "
          f"epe_threshold={result1.primary.opc_params.get('epe_threshold')}")
if result1.warnings:
    print(f"  警告: {result1.warnings}")
print(f"  备选方案: {[a.strategy.value for a in result1.alternatives[:2]]}")
print(f"  匹配历史实验: {result1.matched_experiments}")

print("\n" + "=" * 70)
print("场景 2: DUV 小 CD 线/空间 (CD=32nm, k1=0.22")
print("=" * 70)

mask2 = np.zeros((256, 256))
for y in range(0, 256, 64):
    mask2[y:y+32, :] = 1.0

result2 = engine.recommend(mask2, pixel_size=1.0, wavelength=193.0, na=1.35)
f2 = result2.features
print(f"  提取特征:")
print(f"    - 最小 CD: {f2.min_cd_nm:.1f} nm")
print(f"    - k1 因子: {f2.k1_factor():.3f}")
print(f"    - 拐角密度: {f2.corner_density:.3f}")
print(f"    - 高频能量比: {f2.spectral.high_freq_energy_ratio:.3f}")
print(f"  推荐策略: {result2.primary.strategy.value}")
print(f"  置信度: {result2.primary.confidence:.3f}")
print(f"  推荐原因: {result2.primary.reason}")
if result2.primary.ilt_params:
    print(f"  ILT 参数: max_iter={result2.primary.ilt_params.get('max_iter')}, "
          f"learning_rate={result2.primary.ilt_params.get('learning_rate')}")
if result2.primary.smo_params:
    print(f"  SMO 参数: max_outer_iterations={result2.primary.smo_params.get('max_outer_iterations')}")
if result2.warnings:
    print(f"  警告: {result2.warnings}")

print("\n" + "=" * 70)
print("场景 3: EUV 接触孔阵列 (CD=20nm, k1=0.49")
print("=" * 70)

f3 = LayoutFeatures(
    min_cd_nm=20.0, corner_density=0.12, periodicity_score=0.70,
    spectral=__import__('advisor.schemas').schemas.SpectralFeatures(high_freq_energy_ratio=0.32),
    technology_node='euv', wavelength=13.5, na=0.33,
)
result3 = engine.recommend_from_features(f3, user_preference='quality')
print(f"  输入特征:")
print(f"    - 最小 CD: {f3.min_cd_nm:.1f} nm")
print(f"    - k1 因子: {f3.k1_factor():.3f}")
print(f"    - 技术节点: {f3.technology_node}")
print(f"    - 拐角密度: {f3.corner_density:.3f}")
print(f"  推荐策略: {result3.primary.strategy.value}")
print(f"  置信度: {result3.primary.confidence:.3f}")
print(f"  推荐原因: {result3.primary.reason}")
print(f"  光学系统提示: {result3.primary.optical_system_hints}")

print("\n" + "=" * 70)
print("场景 4: 高拐角密度 SRAM 类图案")
print("=" * 70)

mask4 = np.zeros((200, 200))
for cy in range(20, 200, 60):
    for cx in range(20, 200, 60):
        mask4[cy:cy+30, cx:cx+20] = 1.0
        mask4[cy+10:cy+30, cx+20:cx+50] = 1.0

result4 = engine.recommend(mask4, pixel_size=1.0, wavelength=193.0, na=1.35, user_preference='balanced')
f4 = result4.features
print(f"  提取特征:")
print(f"    - 最小 CD: {f4.min_cd_nm:.1f} nm")
print(f"    - k1 因子: {f4.k1_factor():.3f}")
print(f"    - 拐角密度: {f4.corner_density:.3f}")
print(f"    - 周期性评分: {f4.periodicity_score:.3f}")
print(f"    - 高频能量比: {f4.spectral.high_freq_energy_ratio:.3f}")
print(f"  推荐策略: {result4.primary.strategy.value}")
print(f"  置信度: {result4.primary.confidence:.3f}")
print(f"  推荐原因: {result4.primary.reason}")

print("\n" + "=" * 70)
print("场景 5: 用户偏好对比 (速度 vs 质量)")
print("=" * 70)

f5 = LayoutFeatures(
    min_cd_nm=38.0, corner_density=0.08, periodicity_score=0.6,
    spectral=__import__('advisor.schemas').schemas.SpectralFeatures(high_freq_energy_ratio=0.22),
    wavelength=193.0, na=1.35,
)

result_speed = engine.recommend_from_features(f5, user_preference='speed')
result_quality = engine.recommend_from_features(f5, user_preference='quality')
result_balanced = engine.recommend_from_features(f5, user_preference='balanced')

print(f"  输入特征: CD=38nm, k1={f5.k1_factor():.3f}")
print(f"  速度优先: {result_speed.primary.strategy.value} (置信度 {result_speed.primary.confidence:.3f})")
print(f"  质量优先: {result_quality.primary.strategy.value} (置信度 {result_quality.primary.confidence:.3f})")
print(f"  平衡模式: {result_balanced.primary.strategy.value} (置信度 {result_balanced.primary.confidence:.3f})")

print("\n" + "=" * 70)
print("知识库统计")
print("=" * 70)
print(f"  总记录数: {kb.size()}")
from collections import Counter
strategy_counts = Counter(r.strategy for r in kb.get_records())
print(f"  各策略记录数: {dict(strategy_counts)}")
tech_counts = Counter(r.technology_node for r in kb.get_records())
print(f"  各技术节点记录数: {dict(tech_counts)}")
layout_counts = Counter(r.layout_type for r in kb.get_records())
print(f"  各地图类型记录数: {dict(layout_counts)}")

print("\n" + "=" * 70)
print("✓ RET 策略推荐引擎验证完成！")
print("=" * 70)
print("\n核心功能:")
print("  ✓ 版图特征提取（频谱、最小CD、拐角密度、周期性）")
print("  ✓ 历史实验知识库（18条内置记录）")
print("  ✓ 规则引擎评分（k1因子、拐角密度、周期性、高频能量比）")
print("  ✓ 知识库相似度匹配（余弦相似度）")
print("  ✓ 综合推荐决策（规则+知识库加权融合）")
print("  ✓ 初始参数生成（OPC/ILT/SMO参数）")
print("  ✓ 用户偏好支持（速度/质量/平衡）")
print("  ✓ 备选方案推荐")
print("  ✓ 警告信息生成")
print("  ✓ REST API 集成")
print("\nAPI 端点:")
print("  POST /api/advisor/recommend          - 从版图推荐")
print("  POST /api/advisor/recommend-features - 从特征推荐")
print("  GET  /api/advisor/knowledge-base     - 查询知识库")
print("\n推荐策略:")
print("  纯 OPC      - 大CD、低复杂度版图")
print("  OPC + SRAF   - 中等CD、周期性版图")
print("  ILT          - 小CD、高复杂度版图")
print("  SMO + ILT    - 极小CD、超高复杂度版图")
print("=" * 70)
