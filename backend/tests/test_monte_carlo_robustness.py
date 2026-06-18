# -*- coding: utf-8 -*-
"""
蒙特卡洛工艺鲁棒性评估单元测试

覆盖:
1. ProcessWindowAnalyzer.monte_carlo_analysis — 采样、统计、可打印比例
2. MaskOptimizer._generate_mc_conditions — 固定种子 / 无种子 / 均匀分布
3. MaskOptimizer._resample_monte_carlo_conditions — 重采样种子推进
4. MaskOptimizer._should_resample_monte_carlo — 判定逻辑
5. MaskOptimizer._compute_monte_carlo_loss / gradient — 损失与梯度集成
"""

import pytest
import numpy as np

from core.imaging import OpticalSystem, ProcessCondition
from analysis.process_window import (
    ProcessWindowAnalyzer,
    MonteCarloConfig,
    MonteCarloMetricStats,
    MonteCarloResult,
)
from algorithms.mask_optimizer import (
    MaskOptimizer,
    OptimizationConfig,
    LossWeights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def optics():
    return OpticalSystem(wavelength=193.0, na=0.85, sigma=0.5)


@pytest.fixture
def line_mask():
    mask = np.zeros((100, 100))
    mask[40:60, :] = 1.0
    return mask


@pytest.fixture
def analyzer(optics, line_mask):
    return ProcessWindowAnalyzer(
        mask=line_mask,
        target=line_mask.copy(),
        optical_system=optics,
        threshold=0.3,
        pixel_size=1.0,
    )


# ---------------------------------------------------------------------------
# 1. ProcessWindowAnalyzer — 蒙特卡洛分析
# ---------------------------------------------------------------------------

class TestMonteCarloAnalysis:

    def test_basic_run(self, analyzer):
        cfg = MonteCarloConfig(n_samples=10, focus_std=30.0, dose_std=0.03,
                               random_seed=0)
        result = analyzer.monte_carlo_analysis(cfg, cd_tolerance=0.2, cd_target=20.0)
        assert isinstance(result, MonteCarloResult)
        assert result.n_samples == 10
        assert len(result.sampled_conditions) == 10

    def test_stats_fields(self, analyzer):
        cfg = MonteCarloConfig(n_samples=5, random_seed=1)
        result = analyzer.monte_carlo_analysis(cfg, cd_tolerance=0.1, cd_target=20.0)
        for stats in (result.cd_stats, result.cd_error_stats, result.epe_stats,
                      result.mse_stats, result.ssim_stats, result.ils_stats,
                      result.nils_stats):
            assert isinstance(stats, MonteCarloMetricStats)
            assert hasattr(stats, 'mean')
            assert hasattr(stats, 'std')
            assert hasattr(stats, 'worst_case')
            assert hasattr(stats, 'variance')
            assert stats.variance == pytest.approx(stats.std ** 2, rel=1e-10)

    def test_passing_ratio_range(self, analyzer):
        cfg = MonteCarloConfig(n_samples=8, random_seed=42)
        result = analyzer.monte_carlo_analysis(cfg, cd_tolerance=0.1, cd_target=20.0)
        assert 0.0 <= result.passing_ratio <= 1.0

    def test_deterministic_with_seed(self, analyzer):
        cfg = MonteCarloConfig(n_samples=10, focus_std=50.0, dose_std=0.05,
                               random_seed=123)
        r1 = analyzer.monte_carlo_analysis(cfg, cd_tolerance=0.1, cd_target=20.0)
        r2 = analyzer.monte_carlo_analysis(cfg, cd_tolerance=0.1, cd_target=20.0)
        np.testing.assert_allclose(r1.cd_stats.mean, r2.cd_stats.mean)
        np.testing.assert_allclose(r1.mse_stats.mean, r2.mse_stats.mean)
        c1 = r1.sampled_conditions
        c2 = r2.sampled_conditions
        for a, b in zip(c1, c2):
            assert a.defocus == b.defocus
            assert a.dose == b.dose

    def test_uniform_distribution(self, analyzer):
        cfg = MonteCarloConfig(n_samples=20, focus_std=50.0, dose_std=0.05,
                               distribution='uniform', random_seed=7)
        result = analyzer.monte_carlo_analysis(cfg, cd_tolerance=0.1, cd_target=20.0)
        for cond in result.sampled_conditions:
            half_focus = 50.0 * np.sqrt(3)
            assert -half_focus <= cond.defocus <= half_focus

    def test_worst_condition_is_in_samples(self, analyzer):
        cfg = MonteCarloConfig(n_samples=15, focus_std=40.0, dose_std=0.04,
                               random_seed=9)
        result = analyzer.monte_carlo_analysis(cfg, cd_tolerance=0.1, cd_target=20.0)
        worst = result.worst_condition
        found = any(
            c.defocus == worst.defocus and c.dose == worst.dose
            for c in result.sampled_conditions
        )
        assert found

    def test_no_seed_different_conditions(self, analyzer):
        cfg = MonteCarloConfig(n_samples=10, focus_std=50.0, dose_std=0.05,
                               random_seed=None)
        r1 = analyzer.monte_carlo_analysis(cfg, cd_tolerance=0.1, cd_target=20.0)
        r2 = analyzer.monte_carlo_analysis(cfg, cd_tolerance=0.1, cd_target=20.0)
        d1 = [c.defocus for c in r1.sampled_conditions]
        d2 = [c.defocus for c in r2.sampled_conditions]
        assert d1 != d2, "无固定 seed 时采样条件应不同"


# ---------------------------------------------------------------------------
# 2. MaskOptimizer — _generate_mc_conditions
# ---------------------------------------------------------------------------

class TestGenerateMcConditions:

    def test_conditions_count(self, optics):
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=12,
            monte_carlo_seed=0,
            loss_weights=LossWeights(worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        opt._mc_resample_counter = 0
        conds = opt._generate_mc_conditions()
        assert len(conds) == 12

    def test_deterministic_same_seed(self, optics):
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=10,
            monte_carlo_seed=42,
            loss_weights=LossWeights(worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        opt._mc_resample_counter = 0
        c1 = opt._generate_mc_conditions()
        opt._mc_resample_counter = 0
        c2 = opt._generate_mc_conditions()
        for a, b in zip(c1, c2):
            assert a.defocus == b.defocus
            assert a.dose == b.dose

    def test_different_seeds_different_samples(self, optics):
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=10,
            monte_carlo_seed=42,
            loss_weights=LossWeights(worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        opt._mc_resample_counter = 0
        c1 = opt._generate_mc_conditions()
        opt._mc_resample_counter = 1
        c2 = opt._generate_mc_conditions()
        defocus_1 = [c.defocus for c in c1]
        defocus_2 = [c.defocus for c in c2]
        assert defocus_1 != defocus_2, "不同 resample_counter 应产出不同样本"

    def test_uniform_bounds(self, optics):
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=50,
            monte_carlo_seed=0,
            monte_carlo_distribution='uniform',
            monte_carlo_focus_std=30.0,
            monte_carlo_dose_std=0.03,
            loss_weights=LossWeights(worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        opt._mc_resample_counter = 0
        conds = opt._generate_mc_conditions()
        half_focus = 30.0 * np.sqrt(3)
        for c in conds:
            assert -half_focus - 1e-9 <= c.defocus <= half_focus + 1e-9


# ---------------------------------------------------------------------------
# 3. MaskOptimizer — 重采样种子推进 (核心 bug 修复验证)
# ---------------------------------------------------------------------------

class TestResampleSeedProgression:

    def test_resample_produces_different_conditions(self, optics):
        """固定 seed 时，连续重采样必须产出不同的样本"""
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=10,
            monte_carlo_seed=42,
            monte_carlo_resample_freq=1,
            loss_weights=LossWeights(worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        image_size = (100, 100)

        opt._setup_monte_carlo(image_size)
        first_conditions = list(opt._mc_conditions)

        opt._resample_monte_carlo_conditions(image_size)
        second_conditions = list(opt._mc_conditions)

        first_defocus = [c.defocus for c in first_conditions]
        second_defocus = [c.defocus for c in second_conditions]
        assert first_defocus != second_defocus, (
            "重采样后条件应与首次不同 (seed 应被推进)"
        )

    def test_counter_increments_on_resample(self, optics):
        """每次重采样 _mc_resample_counter 递增 1"""
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=5,
            monte_carlo_seed=0,
            monte_carlo_resample_freq=1,
            loss_weights=LossWeights(worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        image_size = (64, 64)

        opt._setup_monte_carlo(image_size)
        assert opt._mc_resample_counter == 1

        opt._resample_monte_carlo_conditions(image_size)
        assert opt._mc_resample_counter == 2

        opt._resample_monte_carlo_conditions(image_size)
        assert opt._mc_resample_counter == 3

    def test_three_rounds_all_different(self, optics):
        """连续三轮重采样产出三组不同的样本"""
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=20,
            monte_carlo_seed=99,
            monte_carlo_resample_freq=1,
            loss_weights=LossWeights(worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        image_size = (64, 64)

        opt._setup_monte_carlo(image_size)
        all_defocus = []
        for _ in range(3):
            all_defocus.append([c.defocus for c in opt._mc_conditions])
            opt._resample_monte_carlo_conditions(image_size)

        assert all_defocus[0] != all_defocus[1], "第1轮 != 第2轮"
        assert all_defocus[1] != all_defocus[2], "第2轮 != 第3轮"
        assert all_defocus[0] != all_defocus[2], "第1轮 != 第3轮"

    def test_deterministic_replay(self, optics):
        """用相同 seed 从同一 counter 值出发，结果一致"""
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=10,
            monte_carlo_seed=42,
            monte_carlo_resample_freq=1,
            loss_weights=LossWeights(worst_case=1.0),
        )
        image_size = (64, 64)

        opt1 = MaskOptimizer(optical_system=optics, config=cfg)
        opt1._setup_monte_carlo(image_size)
        opt1._resample_monte_carlo_conditions(image_size)
        d1 = [c.defocus for c in opt1._mc_conditions]

        opt2 = MaskOptimizer(optical_system=optics, config=cfg)
        opt2._setup_monte_carlo(image_size)
        opt2._resample_monte_carlo_conditions(image_size)
        d2 = [c.defocus for c in opt2._mc_conditions]

        assert d1 == d2, "相同 seed + 相同重采样轮次，结果应完全一致"


# ---------------------------------------------------------------------------
# 4. MaskOptimizer — _should_resample_monte_carlo
# ---------------------------------------------------------------------------

class TestShouldResample:

    def test_never_when_disabled(self, optics):
        cfg = OptimizationConfig(use_monte_carlo=False)
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        for e in range(1, 30):
            assert opt._should_resample_monte_carlo(e) is False

    def test_never_when_freq_zero(self, optics):
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_resample_freq=0,
            loss_weights=LossWeights(worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        for e in range(1, 30):
            assert opt._should_resample_monte_carlo(e) is False

    def test_triggers_at_correct_epochs(self, optics):
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_resample_freq=5,
            loss_weights=LossWeights(worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        for e in range(1, 21):
            expected = (e % 5 == 0)
            assert opt._should_resample_monte_carlo(e) is expected, f"epoch={e}"

    def test_epoch_zero_never_resamples(self, optics):
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_resample_freq=1,
            loss_weights=LossWeights(worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        assert opt._should_resample_monte_carlo(0) is False


# ---------------------------------------------------------------------------
# 5. MaskOptimizer — 蒙特卡洛损失与梯度集成
# ---------------------------------------------------------------------------

class TestMonteCarloLossIntegration:

    def test_mc_loss_returns_worst(self, optics):
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=5,
            monte_carlo_seed=0,
            use_composite_loss=True,
            loss_weights=LossWeights(mse=1.0, worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        mask = np.zeros((64, 64))
        mask[25:40, :] = 1.0
        target = mask.copy()
        opt._target_image = target
        opt._spatial_weight_mask = None
        opt._setup_imaging_model(mask.shape)
        opt._setup_monte_carlo(mask.shape)

        worst, mean, idx = opt._compute_monte_carlo_loss(mask)
        assert idx >= 0
        assert idx < 5
        assert worst >= mean - 1e-12

    def test_mc_gradient_shape(self, optics):
        cfg = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=3,
            monte_carlo_seed=0,
            use_composite_loss=True,
            loss_weights=LossWeights(mse=1.0, worst_case=1.0),
        )
        opt = MaskOptimizer(optical_system=optics, config=cfg)
        mask = np.zeros((64, 64))
        mask[25:40, :] = 1.0
        target = mask.copy()
        opt._target_image = target
        opt._spatial_weight_mask = None
        opt._setup_imaging_model(mask.shape)
        opt._setup_monte_carlo(mask.shape)

        opt._compute_monte_carlo_loss(mask)
        grad = opt._compute_monte_carlo_gradient(mask)
        assert grad.shape == mask.shape

    def test_loss_includes_worst_case_weight(self, optics):
        mask = np.zeros((64, 64))
        mask[25:40, :] = 1.0
        target = mask.copy()

        cfg_base = OptimizationConfig(
            use_composite_loss=True,
            loss_weights=LossWeights(mse=1.0, worst_case=0.0),
        )
        opt_base = MaskOptimizer(optical_system=optics, config=cfg_base)
        opt_base._target_image = target
        opt_base._spatial_weight_mask = None
        opt_base._setup_imaging_model(mask.shape)
        loss_base = opt_base._compute_loss(mask)

        cfg_mc = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=5,
            monte_carlo_seed=0,
            use_composite_loss=True,
            loss_weights=LossWeights(mse=1.0, worst_case=0.5),
        )
        opt_mc = MaskOptimizer(optical_system=optics, config=cfg_mc)
        opt_mc._target_image = target
        opt_mc._spatial_weight_mask = None
        opt_mc._setup_imaging_model(mask.shape)
        opt_mc._setup_monte_carlo(mask.shape)
        loss_mc = opt_mc._compute_loss(mask)

        assert loss_mc > loss_base, "worst_case>0 时总损失应大于基础损失"

    def test_gradient_includes_mc_contribution(self, optics):
        mask = np.zeros((64, 64))
        mask[25:40, :] = 1.0
        target = mask.copy()

        cfg_base = OptimizationConfig(
            use_composite_loss=True,
            loss_weights=LossWeights(mse=1.0, worst_case=0.0),
        )
        opt_base = MaskOptimizer(optical_system=optics, config=cfg_base)
        opt_base._target_image = target
        opt_base._spatial_weight_mask = None
        opt_base._setup_imaging_model(mask.shape)
        grad_base = opt_base._compute_gradient(mask)

        cfg_mc = OptimizationConfig(
            use_monte_carlo=True,
            monte_carlo_n_samples=3,
            monte_carlo_seed=0,
            use_composite_loss=True,
            loss_weights=LossWeights(mse=1.0, worst_case=1.0),
        )
        opt_mc = MaskOptimizer(optical_system=optics, config=cfg_mc)
        opt_mc._target_image = target
        opt_mc._spatial_weight_mask = None
        opt_mc._setup_imaging_model(mask.shape)
        opt_mc._setup_monte_carlo(mask.shape)
        grad_mc = opt_mc._compute_gradient(mask)

        diff = np.abs(grad_mc - grad_base).sum()
        assert diff > 1e-12, "worst_case>0 时梯度应有额外贡献"
