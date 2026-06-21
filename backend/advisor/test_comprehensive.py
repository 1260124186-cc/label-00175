#!/usr/bin/env python3
"""
RET 策略推荐引擎综合测试

验证所有核心功能：
1. 版图特征提取（频谱、最小CD、拐角密度、周期性）
2. 知识库查询与相似度匹配
3. 规则引擎评分
4. 综合推荐决策
5. 参数生成
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from advisor.recommendation_engine import RETRecommendationEngine
from advisor.feature_extractor import LayoutFeatureExtractor
from advisor.knowledge_base import RETKnowledgeBase
from advisor.schemas import (
    LayoutFeatures, SpectralFeatures, RETStrategy,
    RETRecommendationResult, ExperimentRecord,
)
import numpy as np

print("=" * 70)
print("RET 策略推荐引擎 - 综合测试")
print("=" * 70)

engine = RETRecommendationEngine()
kb = RETKnowledgeBase()

all_passed = True


def test_case(name, condition, details=""):
    global all_passed
    status = "✓ PASS" if condition else "✗ FAIL"
    if not condition:
        all_passed = False
    print(f"  {status}: {name}")
    if details and not condition:
        print(f"         {details}")


print("\n【1/5】版图特征提取测试")
print("-" * 70)

# 1.1 周期性线/空间结构
mask_ls = np.zeros((180, 180))
for y in range(0, 180, 90):
    mask_ls[y:y+45, :] = 1.0

features_ls = LayoutFeatureExtractor.extract(mask_ls, pixel_size=1.0)
test_case("L/S 特征提取: min_cd ≈ 45nm",
          abs(features_ls.min_cd_nm - 45.0) < 5.0,
          f"实际: {features_ls.min_cd_nm:.1f}nm")
test_case("L/S 特征提取: 周期性评分 > 0.7",
          features_ls.periodicity_score > 0.7,
          f"实际: {features_ls.periodicity_score:.3f}")
test_case("L/S 特征提取: k1 因子计算正确",
          abs(features_ls.k1_factor() - 0.322) < 0.02,
          f"实际: {features_ls.k1_factor():.3f}")
test_case("L/S 特征提取: 复杂度评分 > 0.5",
          features_ls.complexity_score() > 0.5,
          f"实际: {features_ls.complexity_score():.3f}")
test_case("L/S 特征提取: 频谱特征完整",
          features_ls.spectral.dominant_frequency > 0,
          f"主频: {features_ls.spectral.dominant_frequency:.1f} cycles/μm")

# 1.2 接触孔阵列
mask_ch = np.zeros((200, 200))
for cy in range(20, 200, 100):
    for cx in range(20, 200, 100):
        y, x = np.ogrid[:200, :200]
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        mask_ch[dist < 25] = 1.0

features_ch = LayoutFeatureExtractor.extract(mask_ch, pixel_size=1.0)
test_case("接触孔: 拐角密度 > 0.05",
          features_ch.corner_density > 0.05,
          f"实际: {features_ch.corner_density:.3f}")
test_case("接触孔: 高频能量占比 > 0.15",
          features_ch.spectral.high_freq_energy_ratio > 0.15,
          f"实际: {features_ch.spectral.high_freq_energy_ratio:.3f}")

# 1.3 L形拐角阵列（多个拐角提高密度）
mask_l = np.zeros((256, 256))
for cy in range(30, 256, 80):
    for cx in range(30, 256, 80):
        mask_l[cy:cy+40, cx:cx+25] = 1.0
        mask_l[cy+15:cy+40, cx+25:cx+60] = 1.0

features_l = LayoutFeatureExtractor.extract(mask_l, pixel_size=1.0)
test_case("L形拐角阵列: 拐角密度 > 0.03",
          features_l.corner_density > 0.03,
          f"实际: {features_l.corner_density:.3f}")
test_case("L形拐角阵列: 高频能量占比 > 0.2",
          features_l.spectral.high_freq_energy_ratio > 0.2,
          f"实际: {features_l.spectral.high_freq_energy_ratio:.3f}")

# 1.4 超大CD简单结构（宽线条）
mask_simple = np.zeros((400, 400))
for y in range(0, 400, 200):
    mask_simple[y:y+100, :] = 1.0

features_simple = LayoutFeatureExtractor.extract(mask_simple, pixel_size=1.0)
test_case("大CD: min_cd 合理值",
          features_simple.min_cd_nm > 50.0,
          f"实际: {features_simple.min_cd_nm:.1f}nm")
test_case("大CD: 复杂度评分 < 0.5",
          features_simple.complexity_score() < 0.5,
          f"实际: {features_simple.complexity_score():.3f}")


print("\n【2/5】知识库测试")
print("-" * 70)

test_case(f"知识库记录数: {kb.size()}",
          kb.size() >= 18,
          f"实际: {kb.size()}")

# 2.1 相似度匹配
query_features = LayoutFeatures(
    min_cd_nm=45.0, corner_density=0.02, periodicity_score=0.9,
    spectral=SpectralFeatures(high_freq_energy_ratio=0.12),
    wavelength=193.0, na=1.35,
)
matches = kb.find_similar(query_features, top_k=3)
test_case("相似度匹配: 返回结果",
          len(matches) > 0,
          f"返回 {len(matches)} 条")
if matches:
    test_case("相似度匹配: 相似度 > 0.7",
              matches[0][1] > 0.7,
              f"最高相似度: {matches[0][1]:.3f}")
    test_case(f"最佳匹配返回有效记录: {matches[0][0].id}",
              matches[0][0].strategy in ['opc_only', 'opc_sraf', 'ilt', 'smo_ilt'],
              f"实际策略: {matches[0][0].strategy}")

# 2.2 各策略最佳匹配
best_opc = kb.find_best_strategy(query_features, strategy='opc_only')
best_opc_sraf = kb.find_best_strategy(query_features, strategy='opc_sraf')
best_ilt = kb.find_best_strategy(query_features, strategy='ilt')
best_smo_ilt = kb.find_best_strategy(query_features, strategy='smo_ilt')

test_case("OPC最佳匹配存在", best_opc is not None)
test_case("OPC+SRAF最佳匹配存在", best_opc_sraf is not None)
test_case("ILT最佳匹配存在", best_ilt is not None)
test_case("SMO+ILT最佳匹配存在", best_smo_ilt is not None)

# 2.3 知识库添加记录
new_record = ExperimentRecord(
    id='test_new_001', layout_type='test',
    technology_node='duv_arf', wavelength=193.0, na=1.35,
    min_cd_nm=50.0, corner_density=0.05, periodicity_score=0.5,
    high_freq_energy_ratio=0.2, strategy='opc_sraf',
    final_epe_nm=2.0, epe_improvement_pct=80.0, convergence=True,
    total_time_sec=25.0,
)
old_size = kb.size()
kb.add_record(new_record)
test_case("知识库添加记录", kb.size() == old_size + 1,
          f"旧:{old_size}, 新:{kb.size()}")


print("\n【3/5】规则引擎测试")
print("-" * 70)

rule_scores = engine._apply_rules(query_features)
test_case("规则评分: 所有策略在 [0,1] 范围",
          all(0 <= s <= 1 for s in rule_scores.values()),
          f"评分: { {k.value: f'{v:.3f}' for k, v in rule_scores.items()} }")

# 3.1 小CD场景（k1<0.4）
f_small_cd = LayoutFeatures(
    min_cd_nm=30.0, corner_density=0.02, periodicity_score=0.85,
    wavelength=193.0, na=1.35,
)
scores_small = engine._apply_rules(f_small_cd)
test_case("小CD: ILT/SMO_ILT 评分 > OPC类",
          scores_small[RETStrategy.ILT] > scores_small[RETStrategy.OPC_ONLY] and
          scores_small[RETStrategy.SMO_ILT] > scores_small[RETStrategy.OPC_ONLY],
          f"ILT:{scores_small[RETStrategy.ILT]:.3f}, OPC:{scores_small[RETStrategy.OPC_ONLY]:.3f}")

# 3.2 高拐角密度场景
f_corner = LayoutFeatures(
    min_cd_nm=50.0, corner_density=0.15, periodicity_score=0.2,
    spectral=SpectralFeatures(high_freq_energy_ratio=0.35),
    wavelength=193.0, na=1.35,
)
scores_corner = engine._apply_rules(f_corner)
test_case("高拐角: ILT 评分 > OPC类",
          scores_corner[RETStrategy.ILT] > scores_corner[RETStrategy.OPC_ONLY],
          f"ILT:{scores_corner[RETStrategy.ILT]:.3f}, OPC:{scores_corner[RETStrategy.OPC_ONLY]:.3f}")

# 3.3 EUV场景
f_euv = LayoutFeatures(
    min_cd_nm=16.0, corner_density=0.03, periodicity_score=0.85,
    technology_node='euv', wavelength=13.5, na=0.33,
)
scores_euv = engine._apply_rules(f_euv)
test_case("EUV: ILT/SMO_ILT 评分较高",
          scores_euv[RETStrategy.ILT] > 0.5 and scores_euv[RETStrategy.SMO_ILT] > 0.5,
          f"ILT:{scores_euv[RETStrategy.ILT]:.3f}, SMO_ILT:{scores_euv[RETStrategy.SMO_ILT]:.3f}")

# 3.4 大CD简单场景
f_large = LayoutFeatures(
    min_cd_nm=100.0, corner_density=0.01, periodicity_score=0.9,
    wavelength=193.0, na=1.35,
)
scores_large = engine._apply_rules(f_large)
test_case("大CD: OPC_ONLY 评分最高",
          scores_large[RETStrategy.OPC_ONLY] == max(scores_large.values()),
          f"OPC:{scores_large[RETStrategy.OPC_ONLY]:.3f}, ILT:{scores_large[RETStrategy.ILT]:.3f}")


print("\n【4/5】综合推荐测试")
print("-" * 70)

# 4.1 从版图推荐
result_ls = engine.recommend(mask_ls, pixel_size=1.0, wavelength=193.0, na=1.35)
test_case("L/S推荐: 返回完整结果",
          isinstance(result_ls, RETRecommendationResult) and
          result_ls.primary is not None,
          f"策略: {result_ls.primary.strategy.value}")
test_case("LSM推荐: k1=0.32 → 推荐 SMO_ILT 或 ILT",
          result_ls.primary.strategy in (RETStrategy.SMO_ILT, RETStrategy.ILT),
          f"实际推荐: {result_ls.primary.strategy.value}")
test_case("推荐置信度 > 0.5",
          result_ls.primary.confidence > 0.5,
          f"置信度: {result_ls.primary.confidence:.3f}")
test_case("包含备选方案",
          len(result_ls.alternatives) >= 1,
          f"备选数: {len(result_ls.alternatives)}")
test_case("包含版图特征",
          result_ls.features is not None)
test_case("推荐原因非空",
          len(result_ls.primary.reason) > 0,
          f"原因: {result_ls.primary.reason[:50]}...")

# 4.2 从特征推荐
result_feat = engine.recommend_from_features(query_features)
test_case("从特征推荐: 成功",
          result_feat.primary is not None,
          f"策略: {result_feat.primary.strategy.value}")

# 4.3 用户偏好测试
result_speed = engine.recommend_from_features(f_small_cd, user_preference='speed')
result_quality = engine.recommend_from_features(f_small_cd, user_preference='quality')
test_case("速度偏好: 倾向较简单策略",
          (result_speed.primary.strategy != result_quality.primary.strategy) or
          (result_speed.primary.confidence < result_quality.primary.confidence),
          f"速度:{result_speed.primary.strategy.value}, 质量:{result_quality.primary.strategy.value}")

# 4.4 参数生成测试
rec = result_ls.primary
if rec.strategy == RETStrategy.OPC_ONLY:
    test_case("OPC参数: 包含关键参数",
              'max_iterations' in rec.opc_params and
              'epe_threshold' in rec.opc_params and
              rec.opc_params.get('sraf_enable') == False)
elif rec.strategy == RETStrategy.OPC_SRAF:
    test_case("OPC+SRAF参数: SRAF已启用",
              rec.opc_params.get('sraf_enable') == True,
              f"SRAF: {rec.opc_params.get('sraf_enable')}")
elif rec.strategy == RETStrategy.ILT:
    test_case("ILT参数: 完整可用",
              'max_iter' in rec.ilt_params and
              'learning_rate' in rec.ilt_params and
              'optimizer_type' in rec.ilt_params)
elif rec.strategy == RETStrategy.SMO_ILT:
    test_case("SMO+ILT参数: 双配置完整",
              'max_iter' in rec.ilt_params and
              'max_outer_iterations' in rec.smo_params and
              'source_init_type' in rec.smo_params)

test_case("光学系统提示: 包含照明类型",
          'illumination_type' in rec.optical_system_hints)


print("\n【5/5】策略演进测试")
print("-" * 70)

print(f"  {'CD(nm)':<8} {'k1':<8} {'推荐策略':<15} {'置信度':<8}")
print("  " + "-" * 50)

cd_values = [100, 80, 60, 45, 38, 32, 28, 20]
last_strategy = None
strategy_transitions = []

for cd in cd_values:
    f = LayoutFeatures(
        min_cd_nm=float(cd), corner_density=0.05, periodicity_score=0.8,
        spectral=SpectralFeatures(high_freq_energy_ratio=0.15),
        wavelength=193.0, na=1.35,
    )
    r = engine.recommend_from_features(f)
    k1 = f.k1_factor()
    strat = r.primary.strategy.value
    conf = r.primary.confidence

    strategy_name = {
        'opc_only': '纯 OPC',
        'opc_sraf': 'OPC+SRAF',
        'ilt': 'ILT',
        'smo_ilt': 'SMO+ILT',
    }[strat]

    print(f"  {cd:<8} {k1:<8.3f} {strategy_name:<15} {conf:<8.3f}")

    if last_strategy and strat != last_strategy:
        strategy_transitions.append((last_strategy, strat))
    last_strategy = strat

test_case("策略演进: 随CD减小策略升级",
          len(strategy_transitions) >= 2,
          f"策略转换次数: {len(strategy_transitions)}")

expected_order = ['opc_only', 'opc_sraf', 'ilt', 'smo_ilt']
seen_strategies = set()
for cd in cd_values:
    f = LayoutFeatures(
        min_cd_nm=float(cd), corner_density=0.05, periodicity_score=0.8,
        spectral=SpectralFeatures(high_freq_energy_ratio=0.15),
        wavelength=193.0, na=1.35,
    )
    r = engine.recommend_from_features(f)
    seen_strategies.add(r.primary.strategy.value)

test_case("覆盖至少三种策略",
          len(seen_strategies) >= 3,
          f"覆盖: {seen_strategies}")
test_case("包含高级策略 ILT 和 SMO_ILT",
          'ilt' in seen_strategies and 'smo_ilt' in seen_strategies,
          f"覆盖: {seen_strategies}")


print("\n" + "=" * 70)
if all_passed:
    print("✓ 所有测试通过！RET 策略推荐引擎功能完整。")
else:
    print("✗ 部分测试失败，请检查以上失败项。")
print("=" * 70)

sys.exit(0 if all_passed else 1)
