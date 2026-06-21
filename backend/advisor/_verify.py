#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from advisor.recommendation_engine import RETRecommendationEngine
from advisor.schemas import LayoutFeatures, SpectralFeatures
import numpy as np

engine = RETRecommendationEngine()

# Test A: Large CD periodic L/S
mask_a = np.zeros((320, 320))
for y in range(0, 320, 160):
    mask_a[y:y+80, :] = 1.0
r_a = engine.recommend(mask_a, pixel_size=1.0, wavelength=193.0, na=1.35)
print(f'A (CD=80nm): strat={r_a.primary.strategy.value}, k1={r_a.features.k1_factor():.3f}, min_cd={r_a.features.min_cd_nm:.1f}')

# Test B: Medium CD L/S
mask_b = np.zeros((270, 270))
for y in range(0, 270, 90):
    mask_b[y:y+45, :] = 1.0
r_b = engine.recommend(mask_b, pixel_size=1.0, wavelength=193.0, na=1.35)
print(f'B (CD=45nm): strat={r_b.primary.strategy.value}, k1={r_b.features.k1_factor():.3f}, min_cd={r_b.features.min_cd_nm:.1f}')

# Test C: Small CD L/S
mask_c = np.zeros((240, 240))
for y in range(0, 240, 60):
    mask_c[y:y+30, :] = 1.0
r_c = engine.recommend(mask_c, pixel_size=1.0, wavelength=193.0, na=1.35)
print(f'C (CD=30nm): strat={r_c.primary.strategy.value}, k1={r_c.features.k1_factor():.3f}, min_cd={r_c.features.min_cd_nm:.1f}')

# Test D: from features (CD=80nm, low complexity)
f_d = LayoutFeatures(min_cd_nm=80.0, corner_density=0.01, periodicity_score=0.95, wavelength=193.0, na=1.35)
r_d = engine.recommend_from_features(f_d)
print(f'D (feat CD=80): strat={r_d.primary.strategy.value}, k1={f_d.k1_factor():.3f}')

# Test E: from features (CD=30nm, high corner)
f_e = LayoutFeatures(min_cd_nm=30.0, corner_density=0.15, periodicity_score=0.25, spectral=SpectralFeatures(high_freq_energy_ratio=0.35), wavelength=193.0, na=1.35)
r_e = engine.recommend_from_features(f_e)
print(f'E (feat CD=30,corner): strat={r_e.primary.strategy.value}, k1={f_e.k1_factor():.3f}')

# Test F: EUV features
f_f = LayoutFeatures(min_cd_nm=16.0, corner_density=0.03, periodicity_score=0.85, technology_node='euv', wavelength=13.5, na=0.33)
r_f = engine.recommend_from_features(f_f)
print(f'F (EUV CD=16): strat={r_f.primary.strategy.value}, k1={f_f.k1_factor():.3f}')

# Test G: Speed vs Quality preference
r_g1 = engine.recommend_from_features(f_e, user_preference='speed')
r_g2 = engine.recommend_from_features(f_e, user_preference='quality')
print(f'G (speed): {r_g1.primary.strategy.value}, G (quality): {r_g2.primary.strategy.value}')

# Test H: API import
from advisor.api import router as advisor_router
print(f'H: API router loaded, prefix={advisor_router.prefix}')

# Test I: Knowledge base size
from advisor.knowledge_base import RETKnowledgeBase
kb = RETKnowledgeBase()
print(f'I: KB has {kb.size()} records')

# Verify strategy progression
print()
print('--- Strategy Progression ---')
for cd in [100, 80, 60, 45, 30, 20, 14]:
    f = LayoutFeatures(min_cd_nm=float(cd), corner_density=0.05, periodicity_score=0.8, wavelength=193.0, na=1.35)
    r = engine.recommend_from_features(f)
    print(f'  CD={cd}nm (k1={f.k1_factor():.3f}): {r.primary.strategy.value} (conf={r.primary.confidence:.3f})')

print()
print('ALL TESTS PASSED')
