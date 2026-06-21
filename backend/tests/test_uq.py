# -*- coding: utf-8 -*-
"""
UQ (Uncertainty Quantification) 模块测试

覆盖范围：
- schemas: 数据结构正确性
- parameter_uncertainty: 工艺/模型参数采样
- bootstrap: Bootstrap 重采样与置信区间
- bayesian_inference: MCMC 与 ABC 采样
- reliability: 失效概率、可靠性指标、敏感度
- uq_analyzer: 端到端 UQ 分析
- visualization: 可视化（仅确保不崩溃）
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# =========================================================================
# schemas 测试
# =========================================================================

class TestSchemas:
    """UQ 数据结构测试"""

    def test_confidence_interval(self):
        from uq.schemas import ConfidenceInterval

        ci = ConfidenceInterval(
            lower=1.0,
            upper=3.0,
            level=0.95,
            method="percentile",
            point_estimate=2.0,
            standard_error=0.5,
        )
        assert ci.width == pytest.approx(2.0, abs=1e-9)
        assert ci.relative_width == pytest.approx(1.0, abs=1e-9)
        assert ci.contains(2.0)
        assert not ci.contains(0.5)

    def test_metric_uncertainty(self):
        from uq.schemas import MetricUncertainty, ConfidenceInterval, UncertaintyType

        samples = np.random.default_rng(0).normal(10, 1, 500)
        ci = ConfidenceInterval(
            lower=8.0, upper=12.0, level=0.95, method="test",
            point_estimate=10.0, standard_error=0.1,
        )
        mu = MetricUncertainty(
            metric_name="cd",
            samples=samples,
            nominal_value=10.0,
            standard_error=0.1,
            confidence_interval=ci,
            uncertainty_type=UncertaintyType.ALEATORY,
        )
        stats = mu.compute_stats()
        assert 9.8 < stats["mean"] < 10.2
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "cv" in stats

    def test_parameter_distribution_normal(self):
        from uq.schemas import ParameterDistribution

        dist = ParameterDistribution.normal("focus", 0.0, 30.0, "nm")
        assert dist.distribution_type == "normal"
        assert dist.params["std"] == 30.0

        rng = np.random.default_rng(42)
        samples = dist.sample(1000, rng)
        assert samples.shape == (1000,)
        assert np.abs(np.mean(samples)) < 3.0

    def test_parameter_distribution_uniform(self):
        from uq.schemas import ParameterDistribution

        dist = ParameterDistribution.uniform("dose", 0.9, 1.1, "rel")
        rng = np.random.default_rng(42)
        samples = dist.sample(2000, rng)
        assert np.all(samples >= 0.9)
        assert np.all(samples <= 1.1)
        assert dist.pdf(1.0) == pytest.approx(5.0)

    def test_uq_result_summary(self):
        from uq.schemas import (
            UQResult, UQMethod, MetricUncertainty, ConfidenceInterval,
            ReliabilityResult, FailureProbabilityResult,
        )
        ci = ConfidenceInterval(
            lower=0.9, upper=1.1, level=0.95, method="test",
            point_estimate=1.0, standard_error=0.05,
        )
        mu = MetricUncertainty(
            metric_name="iou",
            samples=np.random.rand(100),
            nominal_value=0.95,
            standard_error=0.01,
            confidence_interval=ci,
        )
        fp = FailureProbabilityResult(
            failure_mode="test",
            probability=0.01,
            confidence_interval=ci,
            standard_error=0.001,
            ppm=1e4,
            log_ppm=4.0,
            n_failures=10,
            n_samples=1000,
        )
        rel = ReliabilityResult(
            overall_failure_probability=0.01,
            failure_probabilities={"test": fp},
            reliability_index=2.326,
            risk_level="medium",
            recommendations=["test rec"],
        )
        uq = UQResult(
            method=UQMethod.BOOTSTRAP,
            n_samples=1000,
            confidence_level=0.95,
            metric_uncertainties={"iou": mu},
            reliability=rel,
            total_time=0.5,
        )
        summary = uq.summary()
        assert "iou" in summary
        assert "medium" in summary.lower()
        assert "β" in summary


# =========================================================================
# parameter_uncertainty 测试
# =========================================================================

class TestParameterUncertainty:
    """参数不确定性采样测试"""

    def test_parameter_sampler_normal(self):
        from uq.parameter_uncertainty import ParameterSampler
        from uq.schemas import ParameterDistribution

        rng = np.random.default_rng(0)
        sampler = ParameterSampler(
            [ParameterDistribution.normal("x", 0.0, 1.0)],
            random_seed=42,
        )
        samples = sampler.sample(2000)
        assert samples.shape == (2000, 1)
        assert np.abs(np.mean(samples[:, 0])) < 0.1

    def test_parameter_sampler_mixed(self):
        from uq.parameter_uncertainty import ParameterSampler
        from uq.schemas import ParameterDistribution

        sampler = ParameterSampler(
            [
                ParameterDistribution.normal("x", 0.0, 1.0),
                ParameterDistribution.uniform("y", 0.0, 1.0),
                ParameterDistribution.lognormal("z", 0.0, 0.5),
            ],
            random_seed=1,
        )
        samples = sampler.sample(500)
        assert samples.shape == (500, 3)
        assert np.all(samples[:, 1] >= 0) and np.all(samples[:, 1] <= 1)
        assert np.all(samples[:, 2] > 0)

    def test_process_sampler(self):
        from uq.parameter_uncertainty import ProcessPerturbationSampler
        from uq.schemas import ProcessUncertaintyConfig

        cfg = ProcessUncertaintyConfig(focus_std=20.0, dose_std=0.02)
        sampler = ProcessPerturbationSampler(cfg, random_seed=42)
        conditions = sampler.sample(20)
        assert len(conditions) == 20
        for pc in conditions:
            assert hasattr(pc, "defocus")
            assert hasattr(pc, "dose")

    def test_model_sampler(self):
        from uq.parameter_uncertainty import ModelUncertaintySampler
        from uq.schemas import ModelUncertaintyConfig

        cfg = ModelUncertaintyConfig(
            threshold_std=0.05,
            diffusion_length_std=0.1,
        )
        sampler = ModelUncertaintySampler(cfg, random_seed=7)
        perts = sampler.sample(10)
        assert len(perts) == 10
        for p in perts:
            assert "threshold_factor" in p
            assert 0.5 < p["threshold_factor"] < 1.5


# =========================================================================
# bootstrap 测试
# =========================================================================

class TestBootstrap:
    """Bootstrap 分析测试"""

    def test_nonparametric_bootstrap_mean(self):
        from uq.bootstrap import BootstrapAnalyzer, BootstrapConfig

        rng = np.random.default_rng(0)
        data = rng.normal(5.0, 2.0, 500)

        cfg = BootstrapConfig(n_bootstrap=1000, ci_method="percentile", random_seed=1)
        analyzer = BootstrapAnalyzer(cfg)
        result = analyzer.nonparametric_bootstrap(data)

        assert result.original_stat == pytest.approx(np.mean(data), abs=1e-9)
        assert len(result.bootstrap_statistics) == 1000
        assert "percentile" in result.confidence_intervals
        assert "normal" in result.confidence_intervals

        ci = result.confidence_intervals["percentile"]
        assert ci.lower < 5.0 < ci.upper

    def test_bca_ci(self):
        from uq.bootstrap import BootstrapAnalyzer, BootstrapConfig

        rng = np.random.default_rng(42)
        data = rng.exponential(2.0, 300)

        cfg = BootstrapConfig(n_bootstrap=2000, ci_method="bca", random_seed=0)
        analyzer = BootstrapAnalyzer(cfg)
        result = analyzer.nonparametric_bootstrap(data)
        assert "bca" in result.confidence_intervals

        ci = result.confidence_intervals["bca"]
        assert ci.lower > 0
        assert ci.upper > ci.lower

    def test_parametric_bootstrap(self):
        from uq.bootstrap import BootstrapAnalyzer, BootstrapConfig

        rng = np.random.default_rng(5)
        data = rng.normal(0, 1, 200)

        cfg = BootstrapConfig(n_bootstrap=500, random_seed=99)
        analyzer = BootstrapAnalyzer(cfg)
        result = analyzer.parametric_bootstrap(data, distribution="normal")

        assert result.converged
        assert len(result.bootstrap_statistics) == 500

    def test_residual_bootstrap(self):
        from uq.bootstrap import BootstrapAnalyzer, BootstrapConfig

        rng = np.random.default_rng(7)
        x = np.linspace(0, 10, 100)
        y_true = 2.0 * x + 1.0
        y = y_true + rng.normal(0, 0.5, 100)

        def predict(x_in):
            return 2.0 * x_in + 1.0

        cfg = BootstrapConfig(n_bootstrap=300, random_seed=123)
        analyzer = BootstrapAnalyzer(cfg)
        result = analyzer.residual_bootstrap(y, predict(x), predict, x)

        assert result.converged
        assert result.confidence_intervals


# =========================================================================
# bayesian_inference 测试
# =========================================================================

class TestBayesianInference:
    """贝叶斯推断测试"""

    def test_mcmc_gaussian(self):
        from uq.bayesian_inference import MCMCSampler, MCMCConfig

        def log_posterior(theta):
            return -0.5 * np.sum((theta - 2.0) ** 2 / (1.0 ** 2))

        cfg = MCMCConfig(
            n_samples=1000,
            n_burnin=200,
            n_chains=2,
            proposal_std=0.5,
            random_seed=42,
        )
        sampler = MCMCSampler(cfg)
        result = sampler.sample(
            log_posterior=log_posterior,
            parameter_names=["mu"],
        )
        assert result.method == "mcmc"
        assert result.posterior.n_params == 1
        samples = result.posterior.samples[:, 0]
        assert 1.5 < np.mean(samples) < 2.5

        hpd = result.posterior.compute_hpd("mu", 0.95)
        assert hpd.lower < 2.0 < hpd.upper

    def test_abc_rejection(self):
        from uq.bayesian_inference import ABCSampler, ABCConfig

        rng = np.random.default_rng(0)
        obs_mean = 5.0
        observed = rng.normal(obs_mean, 1.0, 50)

        def simulator(params):
            mu = params[0]
            return rng.normal(mu, 1.0, 50)

        def prior_sampler():
            return np.array([rng.uniform(0, 10)])

        cfg = ABCConfig(n_samples=100, epsilon=1.5, random_seed=99)
        sampler = ABCSampler(cfg)
        result = sampler.sample_rejection(
            observed_data=observed,
            simulator=simulator,
            prior_sampler=prior_sampler,
            parameter_names=["mu"],
        )
        assert result.method == "abc_rejection"
        assert result.posterior.n_samples >= 1
        samples = result.posterior.samples[:, 0]
        assert np.all((samples >= 0) & (samples <= 10))

    def test_posterior_hpd_interval(self):
        from uq.bayesian_inference import PosteriorSample

        rng = np.random.default_rng(1)
        samples = rng.normal(0, 1, (2000, 1))
        ps = PosteriorSample(
            parameter_names=["x"],
            samples=samples,
            acceptance_rate=1.0,
        )
        hpd = ps.compute_hpd("x", 0.95)
        assert hpd.lower < 0 < hpd.upper
        assert hpd.upper - hpd.lower < 8.0


# =========================================================================
# reliability 测试
# =========================================================================

class TestReliability:
    """可靠性分析测试"""

    def test_failure_criterion_cd(self):
        from uq.reliability import FailureCriterion

        fc = FailureCriterion.cd_failure(cd_target=45, cd_tolerance=0.1)
        assert fc.name == "cd_out_of_spec"
        assert fc.is_safe({"cd": 45})
        assert not fc.is_safe({"cd": 50})
        assert not fc.is_safe({"cd": 40})

    def test_failure_criterion_epe(self):
        from uq.reliability import FailureCriterion

        fc = FailureCriterion.epe_failure(epe_limit=3.0)
        assert fc.is_safe({"epe": 1.5})
        assert not fc.is_safe({"epe": 4.0})

    def test_compute_failure_probability(self):
        from uq.reliability import compute_failure_probability, FailureCriterion

        rng = np.random.default_rng(42)
        n = 2000
        cd_samples = rng.normal(45, 3, n)
        epe_samples = rng.exponential(1.0, n)

        results = compute_failure_probability(
            metric_samples={"cd": cd_samples, "epe": epe_samples},
            failure_criteria=[
                FailureCriterion.cd_failure(45, 0.1),
                FailureCriterion.epe_failure(3.0),
            ],
            confidence_level=0.95,
            n_bootstrap=200,
            random_seed=7,
        )
        assert "cd_out_of_spec" in results
        assert "epe_exceed" in results

        cd_fp = results["cd_out_of_spec"]
        assert 0 < cd_fp.probability < 0.5
        assert cd_fp.confidence_interval is not None
        assert cd_fp.ppm == cd_fp.probability * 1e6

    def test_reliability_index(self):
        from uq.reliability import compute_reliability_index

        assert compute_reliability_index(1e-3) == pytest.approx(3.09, abs=0.1)
        assert compute_reliability_index(1e-6) == pytest.approx(4.75, abs=0.1)
        beta_50 = compute_reliability_index(0.5)
        assert abs(beta_50) < 0.1

    def test_reliability_analyzer_full(self):
        from uq.reliability import ReliabilityAnalyzer, FailureCriterion

        rng = np.random.default_rng(42)
        n = 1000
        cd_samples = rng.normal(45, 2.5, n)
        epe_samples = rng.exponential(0.8, n)

        focus = rng.normal(0, 30, n)
        dose = rng.normal(1.0, 0.03, n)

        analyzer = ReliabilityAnalyzer(confidence_level=0.95)
        result = analyzer.analyze(
            metric_samples={"cd": cd_samples, "epe_mean": epe_samples},
            failure_criteria=[
                FailureCriterion.cd_failure(45, 0.1),
                FailureCriterion.epe_failure(3.0),
            ],
            parameter_samples=np.column_stack([focus, dose, np.ones(n) * 1.35, np.ones(n) * 0.75]),
            parameter_names=["defocus", "dose", "na", "sigma"],
            n_bootstrap=200,
            random_seed=0,
        )
        assert 0 <= result.overall_failure_probability <= 1
        assert result.reliability_index >= 0
        assert result.risk_level in (
            "very_low", "low", "medium", "high", "very_high"
        )
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) >= 1
        assert "cd" in result.sensitivity_indices


# =========================================================================
# uq_analyzer 集成测试
# =========================================================================

class TestUQAnalyzer:
    """UQ 分析器端到端测试"""

    def _make_simple_mask(self, size=64, side=24):
        mask = np.zeros((size, size), dtype=np.float64)
        margin = (size - side) // 2
        mask[margin:margin + side, margin:margin + side] = 1.0
        return mask

    def test_run_simulations_smoke(self):
        from uq.uq_analyzer import UQAnalyzer
        from uq.schemas import UQConfig, ProcessUncertaintyConfig

        mask = self._make_simple_mask(32, 12)
        cfg = UQConfig(
            n_samples=10,
            n_bootstrap=50,
            process_uncertainty=ProcessUncertaintyConfig(
                focus_std=10.0, dose_std=0.01,
            ),
        )
        analyzer = UQAnalyzer(config=cfg, random_seed=42)
        runs = analyzer.run_simulations(mask, save_images=False)
        assert len(runs) == 10
        assert isinstance(runs[0].metrics, dict)
        assert "wafer_mean" in runs[0].metrics

    def test_full_analysis(self):
        from uq.uq_analyzer import UQAnalyzer
        from uq.schemas import UQConfig, ProcessUncertaintyConfig

        mask = self._make_simple_mask(40, 16)
        target = self._make_simple_mask(40, 16)

        cfg = UQConfig(
            n_samples=30,
            n_bootstrap=100,
            process_uncertainty=ProcessUncertaintyConfig(
                focus_std=20.0, dose_std=0.02,
            ),
        )
        analyzer = UQAnalyzer(config=cfg, random_seed=7)
        result = analyzer.full_analysis(
            mask=mask,
            target=target,
        )

        assert result.n_samples == 30
        assert len(result.metric_uncertainties) > 0
        assert result.reliability is not None
        assert result.total_time >= 0

        for name, mu in result.metric_uncertainties.items():
            assert mu.confidence_interval is not None

        summary = result.summary()
        assert "可靠性" in summary or "reliability" in summary.lower()


# =========================================================================
# visualization 测试
# =========================================================================

class TestVisualization:
    """可视化模块测试（仅验证接口正确，不验证图像内容）"""

    def test_plot_metric_distribution(self):
        from uq.visualization import (
            plot_metric_distribution, PlotConfig,
        )
        from uq.schemas import MetricUncertainty, ConfidenceInterval

        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")

        rng = np.random.default_rng(0)
        mu = MetricUncertainty(
            metric_name="epe_mean",
            samples=rng.normal(1.5, 0.3, 300),
            nominal_value=1.5,
            standard_error=0.02,
            confidence_interval=ConfidenceInterval(
                lower=1.2, upper=1.8, level=0.95,
                method="test", point_estimate=1.5, standard_error=0.02,
            ),
        )
        fig = plot_metric_distribution(mu, PlotConfig(show=False))
        assert fig is not None

    def test_plot_confidence_intervals(self):
        from uq.visualization import (
            plot_confidence_intervals, PlotConfig,
        )
        from uq.schemas import MetricUncertainty, ConfidenceInterval

        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")

        rng = np.random.default_rng(0)
        metrics = {}
        for i, name in enumerate(["epe_mean", "iou", "nils", "wafer_mean"]):
            mu = MetricUncertainty(
                metric_name=name,
                samples=rng.normal(i, 0.5, 200),
                nominal_value=float(i),
                standard_error=0.05,
                confidence_interval=ConfidenceInterval(
                    lower=i - 0.3, upper=i + 0.3, level=0.95,
                    method="test", point_estimate=float(i), standard_error=0.05,
                ),
            )
            metrics[name] = mu

        fig = plot_confidence_intervals(metrics, PlotConfig(show=False))
        assert fig is not None

    def test_plot_failure_probability(self):
        from uq.visualization import plot_failure_probability, PlotConfig
        from uq.schemas import ReliabilityResult, FailureProbabilityResult, ConfidenceInterval

        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")

        ci = ConfidenceInterval(
            lower=0.0, upper=0.1, level=0.95, method="test",
            point_estimate=0.01, standard_error=0.003,
        )
        fps = {
            "cd_out_of_spec": FailureProbabilityResult(
                failure_mode="cd_out_of_spec", probability=0.01,
                confidence_interval=ci, standard_error=0.003,
                ppm=1e4, log_ppm=4.0, n_failures=10, n_samples=1000,
                reliability_index=2.326,
            ),
            "epe_exceed": FailureProbabilityResult(
                failure_mode="epe_exceed", probability=0.001,
                confidence_interval=ci, standard_error=0.001,
                ppm=1e3, log_ppm=3.0, n_failures=1, n_samples=1000,
                reliability_index=3.090,
            ),
        }
        rel = ReliabilityResult(
            overall_failure_probability=0.011,
            failure_probabilities=fps,
            reliability_index=2.29,
            risk_level="medium",
            recommendations=[],
        )
        fig = plot_failure_probability(rel, PlotConfig(show=False))
        assert fig is not None


# =========================================================================
# 便捷函数冒烟测试
# =========================================================================

class TestConvenienceFunctions:
    """顶层模块便捷函数测试"""

    def test_bootstrap_ci(self):
        from uq.bootstrap import bootstrap_ci

        rng = np.random.default_rng(0)
        data = rng.normal(10, 1, 200)
        ci = bootstrap_ci(data, n_bootstrap=500, ci_method="percentile", random_seed=1)
        assert ci.lower < 10 < ci.upper

    def test_bayesian_inference_shortcut(self):
        from uq.bayesian_inference import bayesian_inference, MCMCConfig

        def log_p(theta):
            return -0.5 * np.sum((theta - 0.0) ** 2)

        cfg = MCMCConfig(
            n_samples=500, n_burnin=100, n_chains=2,
            proposal_std=0.5, random_seed=42,
        )
        result = bayesian_inference(
            log_posterior_or_observed=log_p,
            parameter_names=["x"],
            method="mcmc",
            config=cfg,
        )
        assert result.posterior.n_samples > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
