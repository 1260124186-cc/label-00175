# -*- coding: utf-8 -*-
"""
配置驱动的回归测试套件

使用 pytest 参数化扫描 experiments/ 目录下的 YAML 实验定义，
逐一执行并验证回归断言。

运行方式:
    # 运行全部回归实验
    pytest experiments/test_regression.py -v

    # 仅运行 mask_optimization 工作流
    pytest experiments/test_regression.py -v -k mask_optimization

    # 仅运行特定实验
    pytest experiments/test_regression.py -v -k line_space_mse

    # 生成 golden 参考文件（首次运行或更新 baseline）
    pytest experiments/test_regression.py -v --generate-golden

    # 自定义实验目录
    pytest experiments/test_regression.py -v --experiments-dir=/path/to/experiments
"""

import pytest
import json
import numpy as np
from pathlib import Path
from typing import List, Optional

from experiments.schema import (
    load_experiment, validate_experiment, ExperimentSchema,
)
from experiments.executor import ExperimentExecutor, ExperimentResult
from experiments.assertions import RegressionAssertions, AssertionReport


EXPERIMENTS_DIR = Path(__file__).parent


def collect_experiment_files(experiments_dir: Optional[str] = None) -> List[Path]:
    """收集实验目录下所有 exp_*.yaml 文件"""
    exp_dir = Path(experiments_dir) if experiments_dir else EXPERIMENTS_DIR
    if not exp_dir.exists():
        return []
    return sorted(exp_dir.glob('exp_*.yaml'))


def pytest_addoption(parser):
    parser.addoption(
        '--experiments-dir',
        action='store',
        default=None,
        help='实验定义目录（默认为 experiments/）',
    )
    parser.addoption(
        '--generate-golden',
        action='store_true',
        default=False,
        help='生成 golden 参考文件（覆盖现有）',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'experiment: 标记为配置驱动的回归实验'
    )
    config.addinivalue_line(
        'markers', 'benchmark: 标记为逆向光刻基准测试'
    )
    config.addinivalue_line(
        'markers', 'slow: 标记为慢速测试, 需显式启用'
    )


def pytest_generate_tests(metafunc):
    if 'experiment_path' not in metafunc.fixturenames:
        return

    experiments_dir = metafunc.config.getoption('--experiments-dir', default=None)
    exp_files = collect_experiment_files(experiments_dir)

    ids = [f.stem for f in exp_files]
    metafunc.parametrize('experiment_path', exp_files, ids=ids)


@pytest.fixture(scope='session')
def executor(tmp_path_factory):
    """创建实验执行器（session 级别共享）"""
    output_dir = str(tmp_path_factory.mktemp('experiment_results'))
    return ExperimentExecutor(base_output_dir=output_dir)


@pytest.fixture(scope='session')
def assertions_checker():
    """创建回归断言检查器"""
    return RegressionAssertions(experiments_dir=str(EXPERIMENTS_DIR))


@pytest.fixture(scope='session')
def generate_golden(request):
    """是否生成 golden 参考文件"""
    return request.config.getoption('--generate-golden', default=False)


@pytest.mark.experiment
def test_experiment_regression(experiment_path: Path,
                               executor: ExperimentExecutor,
                               assertions_checker: RegressionAssertions,
                               generate_golden: bool):
    """
    参数化回归测试：对每个 exp_*.yaml 执行实验并验证断言
    """
    experiment = load_experiment(experiment_path)

    errors = validate_experiment(experiment)
    validation_warnings = [e for e in errors if 'golden' not in e.lower()]
    if validation_warnings:
        pytest.skip(f"实验定义验证失败: {'; '.join(validation_warnings)}")

    result = executor.run(experiment)

    if not result.success:
        pytest.fail(f"实验 '{experiment.name}' 执行失败: {result.error_message}")

    if generate_golden and result.output_dir:
        golden_path = Path(result.output_dir) / 'golden.json'
        golden_data = {
            'experiment_name': experiment.name,
            'final_mse': result.final_mse,
            'final_ssim': result.final_ssim,
            'converged': result.converged,
            'convergence_step': result.convergence_step,
            'final_loss': result.final_loss,
            'total_iterations': result.total_iterations,
        }
        golden_data.update(result.custom_metrics)
        with open(golden_path, 'w', encoding='utf-8') as f:
            json.dump(golden_data, f, indent=2, ensure_ascii=False)
        pytest.skip(f"Golden 文件已生成: {golden_path}")

    report = assertions_checker.evaluate(
        experiment_name=experiment.name,
        assertions=experiment.assertions,
        result=result,
        result_dir=result.output_dir,
    )

    if not report.all_passed:
        failed_details = []
        for r in report.results:
            if not r.passed:
                failed_details.append(f"[{r.assertion_type}] {r.message}")
        pytest.fail(
            f"实验 '{experiment.name}' 回归验证失败:\n" +
            "\n".join(f"  - {d}" for d in failed_details)
        )


@pytest.mark.experiment
class TestSchemaValidation:
    """实验定义格式验证测试"""

    def test_pattern_config_valid_types(self):
        from experiments.schema import PatternConfig
        for ptype in PatternConfig.VALID_TYPES:
            cfg = PatternConfig(type=ptype, cd=45.0, pitch=90.0)
            assert cfg.type == ptype

    def test_pattern_config_invalid_type(self):
        from experiments.schema import PatternConfig
        with pytest.raises(ValueError, match="无效图案类型"):
            PatternConfig(type='invalid_type')

    def test_pattern_config_cd_pitch_validation(self):
        from experiments.schema import PatternConfig
        with pytest.raises(ValueError, match="pitch"):
            PatternConfig(cd=100.0, pitch=50.0)

    def test_optical_config_valid(self):
        from experiments.schema import OpticalConfig
        cfg = OpticalConfig(wavelength=193.0, na=1.35, sigma=0.75)
        assert cfg.wavelength == 193.0

    def test_optical_config_invalid_na(self):
        from experiments.schema import OpticalConfig
        with pytest.raises(ValueError, match="na"):
            OpticalConfig(na=3.0)

    def test_optimizer_config_valid_types(self):
        from experiments.schema import OptimizerConfig
        for otype in OptimizerConfig.VALID_TYPES:
            cfg = OptimizerConfig(type=otype)
            assert cfg.type == otype

    def test_optimizer_config_invalid_type(self):
        from experiments.schema import OptimizerConfig
        with pytest.raises(ValueError, match="无效优化器类型"):
            OptimizerConfig(type='invalid_optimizer')

    def test_assertion_config_mse_requires_threshold(self):
        from experiments.schema import AssertionConfig
        with pytest.raises(ValueError, match="threshold"):
            AssertionConfig(type='mse_threshold')

    def test_assertion_config_convergence_requires_max_steps(self):
        from experiments.schema import AssertionConfig
        with pytest.raises(ValueError, match="max_steps"):
            AssertionConfig(type='convergence_steps')

    def test_assertion_config_golden_requires_path(self):
        from experiments.schema import AssertionConfig
        with pytest.raises(ValueError, match="golden_path"):
            AssertionConfig(type='golden_deviation')

    def test_experiment_schema_workflow_validation(self):
        from experiments.schema import ExperimentSchema
        with pytest.raises(ValueError, match="无效工作流类型"):
            ExperimentSchema(workflow='invalid_workflow')

    def test_load_all_experiment_yamls(self):
        exp_files = collect_experiment_files()
        for exp_file in exp_files:
            experiment = load_experiment(exp_file)
            assert experiment.name != 'unnamed_experiment', f"{exp_file.name} 缺少实验名称"
            assert experiment.workflow in ExperimentSchema.VALID_WORKFLOWS

    def test_validate_experiment_no_assertions_warning(self):
        experiment = ExperimentSchema(
            name='test_no_assertions',
            assertions=[],
        )
        errors = validate_experiment(experiment)
        assert any('断言' in e for e in errors)


@pytest.mark.experiment
class TestAssertionEngine:
    """回归断言引擎单元测试"""

    def test_mse_threshold_pass(self):
        from experiments.assertions import RegressionAssertions
        from experiments.schema import AssertionConfig
        from experiments.executor import ExperimentResult

        checker = RegressionAssertions()
        assertion = AssertionConfig(type='mse_threshold', threshold=0.01)
        result = ExperimentResult(final_mse=0.005)
        report = checker.evaluate('test', [assertion], result)
        assert report.all_passed

    def test_mse_threshold_fail(self):
        from experiments.assertions import RegressionAssertions
        from experiments.schema import AssertionConfig
        from experiments.executor import ExperimentResult

        checker = RegressionAssertions()
        assertion = AssertionConfig(type='mse_threshold', threshold=0.001)
        result = ExperimentResult(final_mse=0.01)
        report = checker.evaluate('test', [assertion], result)
        assert not report.all_passed

    def test_convergence_steps_pass(self):
        from experiments.assertions import RegressionAssertions
        from experiments.schema import AssertionConfig
        from experiments.executor import ExperimentResult

        checker = RegressionAssertions()
        assertion = AssertionConfig(type='convergence_steps', max_steps=100)
        result = ExperimentResult(converged=True, convergence_step=50)
        report = checker.evaluate('test', [assertion], result)
        assert report.all_passed

    def test_convergence_steps_fail(self):
        from experiments.assertions import RegressionAssertions
        from experiments.schema import AssertionConfig
        from experiments.executor import ExperimentResult

        checker = RegressionAssertions()
        assertion = AssertionConfig(type='convergence_steps', max_steps=10)
        result = ExperimentResult(converged=True, convergence_step=50)
        report = checker.evaluate('test', [assertion], result)
        assert not report.all_passed

    def test_ssim_threshold_pass(self):
        from experiments.assertions import RegressionAssertions
        from experiments.schema import AssertionConfig
        from experiments.executor import ExperimentResult

        checker = RegressionAssertions()
        assertion = AssertionConfig(type='ssim_threshold', threshold=0.9)
        result = ExperimentResult(final_ssim=0.95)
        report = checker.evaluate('test', [assertion], result)
        assert report.all_passed

    def test_loss_improvement_pass(self):
        from experiments.assertions import RegressionAssertions
        from experiments.schema import AssertionConfig
        from experiments.executor import ExperimentResult

        checker = RegressionAssertions()
        assertion = AssertionConfig(type='loss_improvement', threshold=0.1)
        result = ExperimentResult(
            custom_metrics={'total_loss_improvement_ratio': 0.3}
        )
        report = checker.evaluate('test', [assertion], result)
        assert report.all_passed

    def test_epe_threshold_pass(self):
        from experiments.assertions import RegressionAssertions
        from experiments.schema import AssertionConfig
        from experiments.executor import ExperimentResult

        checker = RegressionAssertions()
        assertion = AssertionConfig(type='epe_threshold', threshold=3.0)
        result = ExperimentResult(
            custom_metrics={'epe_mean': 1.5}
        )
        report = checker.evaluate('test', [assertion], result)
        assert report.all_passed

    def test_golden_deviation_with_file(self, tmp_path):
        from experiments.assertions import RegressionAssertions
        from experiments.schema import AssertionConfig
        from experiments.executor import ExperimentResult

        golden_file = tmp_path / 'golden.json'
        golden_data = {
            'experiment_name': 'test',
            'final_mse': 0.005,
            'final_ssim': 0.95,
        }
        golden_file.write_text(json.dumps(golden_data))

        checker = RegressionAssertions()
        assertion = AssertionConfig(
            type='golden_deviation',
            golden_path=str(golden_file),
            tolerance=0.05,
        )
        result = ExperimentResult(final_mse=0.0051, final_ssim=0.95)
        report = checker.evaluate('test', [assertion], result)
        assert report.all_passed

    def test_golden_deviation_fail(self, tmp_path):
        from experiments.assertions import RegressionAssertions
        from experiments.schema import AssertionConfig
        from experiments.executor import ExperimentResult

        golden_file = tmp_path / 'golden.json'
        golden_data = {
            'experiment_name': 'test',
            'final_mse': 0.005,
        }
        golden_file.write_text(json.dumps(golden_data))

        checker = RegressionAssertions()
        assertion = AssertionConfig(
            type='golden_deviation',
            golden_path=str(golden_file),
            tolerance=0.05,
        )
        result = ExperimentResult(final_mse=0.01)
        report = checker.evaluate('test', [assertion], result)
        assert not report.all_passed

    def test_multiple_assertions(self):
        from experiments.assertions import RegressionAssertions
        from experiments.schema import AssertionConfig
        from experiments.executor import ExperimentResult

        checker = RegressionAssertions()
        assertions = [
            AssertionConfig(type='mse_threshold', threshold=0.01),
            AssertionConfig(type='convergence_steps', max_steps=100),
        ]
        result = ExperimentResult(
            final_mse=0.005,
            converged=True,
            convergence_step=50,
        )
        report = checker.evaluate('test', assertions, result)
        assert report.all_passed
        assert len(report.results) == 2


def _collect_benchmark_ids():
    from benchmarks import get_all_test_cases, DEFAULT_ALGORITHMS
    ids = []
    for tc in get_all_test_cases():
        for algo in DEFAULT_ALGORITHMS:
            ids.append(f"{tc.name}__{algo.name}")
    return ids


def _collect_benchmark_params():
    from benchmarks import get_all_test_cases, DEFAULT_ALGORITHMS
    params = []
    for tc in get_all_test_cases():
        for algo in DEFAULT_ALGORITHMS:
            params.append((tc.name, algo.name))
    return params


_benchmark_params = _collect_benchmark_params()
_benchmark_ids = _collect_benchmark_ids()


@pytest.mark.experiment
@pytest.mark.benchmark
class TestBenchmarkSuite:
    """逆向光刻基准测试套件 - 回归集成"""

    def test_benchmark_test_case_definitions(self):
        from benchmarks import get_all_test_cases, TestCaseCategory, DifficultyLevel

        cases = get_all_test_cases()
        assert len(cases) > 0, "基准测试用例列表不应为空"

        categories = set(tc.category for tc in cases)
        required = {TestCaseCategory.LINE_SPACE, TestCaseCategory.CONTACT_HOLE,
                    TestCaseCategory.LOGIC_CELL, TestCaseCategory.SRAM_BITCELL}
        assert required.issubset(categories), \
            f"缺少必要分类: {required - categories}"

        for tc in cases:
            assert tc.name, "测试用例名称不能为空"
            assert tc.technology_node > 0, f"工艺节点必须为正: {tc.name}"
            assert tc.pattern_params, f"图案参数不能为空: {tc.name}"
            assert tc.optical_params, f"光学参数不能为空: {tc.name}"

    def test_benchmark_protocol_convergence_metric(self):
        from benchmarks.protocol import ConvergenceMetric

        metric = ConvergenceMetric(
            iterations_to_threshold=50,
            threshold_value=5.0,
            total_iterations=100,
            converged=True,
            convergence_step=50,
            epe_history=[10.0, 7.0, 5.0, 4.0, 3.5],
        )
        d = metric.to_dict()
        assert d['iterations_to_threshold'] == 50
        assert d['converged'] is True

    def test_benchmark_protocol_epe_metric(self):
        from benchmarks.protocol import EPEMetric

        metric = EPEMetric(
            epe_mean=2.5, epe_max=5.0,
            epe_std=1.0, epe_median=2.3,
            epe_improvement_ratio=0.5,
        )
        d = metric.to_dict()
        assert d['epe_mean'] == 2.5
        assert d['epe_improvement_ratio'] == 0.5

    def test_benchmark_protocol_pw_metric(self):
        from benchmarks.protocol import PWMetric

        metric = PWMetric(
            pw_area=1000.0, pw_ratio=0.35,
            depth_of_focus=150.0, exposure_latitude=12.0,
            n_passing=50, n_total=121,
        )
        d = metric.to_dict()
        assert d['pw_ratio'] == 0.35
        assert d['n_passing'] == 50

    def test_benchmark_protocol_time_metric(self):
        from benchmarks.protocol import TimeMetric

        metric = TimeMetric(
            total_wall_time=5.0, per_iteration_time=0.05,
            imaging_time=2.0, optimization_time=3.0,
        )
        d = metric.to_dict()
        assert d['total_wall_time'] == 5.0

    def test_benchmark_result_serialization(self):
        from benchmarks.protocol import (
            BenchmarkResult, ConvergenceMetric, EPEMetric,
            PWMetric, TimeMetric,
        )

        result = BenchmarkResult(
            test_case_name='ls_45nm_half_pitch',
            algorithm_name='gd',
            convergence=ConvergenceMetric(converged=True, convergence_step=50),
            epe=EPEMetric(epe_mean=2.5),
            pw=PWMetric(pw_ratio=0.3),
            time=TimeMetric(total_wall_time=1.0),
            success=True,
        )
        d = result.to_dict()
        assert d['test_case_name'] == 'ls_45nm_half_pitch'
        assert d['epe']['epe_mean'] == 2.5

        summary = result.summary()
        assert 'ls_45nm_half_pitch' in summary
        assert 'gd' in summary

    def test_benchmark_algorithm_spec(self):
        from benchmarks import AlgorithmSpec, DEFAULT_ALGORITHMS

        spec = AlgorithmSpec(name='test_algo', optimizer_type='adam')
        assert spec.name == 'test_algo'
        assert spec.optimizer_type == 'adam'

        assert len(DEFAULT_ALGORITHMS) > 0
        for a in DEFAULT_ALGORITHMS:
            assert a.name
            assert a.optimizer_type

    def test_benchmark_get_by_category(self):
        from benchmarks import get_test_cases_by_category, TestCaseCategory

        ls_cases = get_test_cases_by_category(TestCaseCategory.LINE_SPACE)
        assert len(ls_cases) > 0
        for tc in ls_cases:
            assert tc.category == TestCaseCategory.LINE_SPACE

    def test_benchmark_get_by_difficulty(self):
        from benchmarks import get_test_cases_by_difficulty, DifficultyLevel

        easy = get_test_cases_by_difficulty(DifficultyLevel.EASY)
        assert len(easy) > 0
        for tc in easy:
            assert tc.difficulty == DifficultyLevel.EASY

    def test_benchmark_get_by_name(self):
        from benchmarks import get_test_case_by_name

        tc = get_test_case_by_name('ls_45nm_half_pitch')
        assert tc is not None
        assert tc.name == 'ls_45nm_half_pitch'

        tc_none = get_test_case_by_name('nonexistent_case')
        assert tc_none is None

    def test_benchmark_compare_results(self):
        from benchmarks.protocol import (
            BenchmarkResult, ConvergenceMetric, EPEMetric,
            PWMetric, TimeMetric, compare_results,
        )

        results = [
            BenchmarkResult(
                test_case_name='test_case',
                algorithm_name='algo_a',
                convergence=ConvergenceMetric(converged=True, total_iterations=50),
                epe=EPEMetric(epe_mean=2.0),
                pw=PWMetric(pw_ratio=0.3),
                time=TimeMetric(total_wall_time=1.0),
                success=True,
            ),
            BenchmarkResult(
                test_case_name='test_case',
                algorithm_name='algo_b',
                convergence=ConvergenceMetric(converged=True, total_iterations=80),
                epe=EPEMetric(epe_mean=3.0),
                pw=PWMetric(pw_ratio=0.2),
                time=TimeMetric(total_wall_time=2.0),
                success=True,
            ),
        ]

        comp = compare_results(results)
        assert comp['test_case'] == 'test_case'
        assert len(comp['rows']) == 2
        assert comp['best']['best_epe'] == 'algo_a'
        assert comp['best']['best_pw'] == 'algo_a'
        assert comp['best']['best_speed'] == 'algo_a'

    def test_benchmark_report_generation(self):
        from benchmarks.protocol import (
            BenchmarkResult, ConvergenceMetric, EPEMetric,
            PWMetric, TimeMetric,
        )
        from benchmarks.report import (
            generate_comparison_table, generate_ranking,
            generate_summary,
        )

        results = [
            BenchmarkResult(
                test_case_name='ls_45nm',
                algorithm_name='gd',
                convergence=ConvergenceMetric(converged=True, total_iterations=50),
                epe=EPEMetric(epe_mean=2.0, epe_max=5.0),
                pw=PWMetric(pw_ratio=0.3, depth_of_focus=120.0),
                time=TimeMetric(total_wall_time=1.0),
                success=True,
            ),
        ]

        table = generate_comparison_table(results)
        assert 'ls_45nm' in table
        assert 'gd' in table

        ranking = generate_ranking(results)
        assert 'ls_45nm' in ranking

        summary = generate_summary(results)
        assert '逆向光刻基准测试' in summary

    def test_benchmark_runner_pattern_generation(self):
        from benchmarks import get_test_case_by_name, BenchmarkRunner

        tc = get_test_case_by_name('ls_45nm_half_pitch')
        assert tc is not None

        mask, target = BenchmarkRunner._generate_pattern(tc)
        assert mask.shape == target.shape
        assert mask.shape == (64, 64)

    def test_benchmark_runner_optical_system(self):
        from benchmarks import get_test_case_by_name, BenchmarkRunner

        tc = get_test_case_by_name('ls_45nm_half_pitch')
        assert tc is not None

        optical = BenchmarkRunner._build_optical_system(tc)
        assert optical.wavelength == 193.0
        assert optical.na == 1.35

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "case_name,algo_name",
        _benchmark_params,
        ids=_benchmark_ids,
    )
    def test_benchmark_regression(self, case_name, algo_name):
        """
        参数化基准回归测试: 对每个 (测试用例, 算法) 组合执行完整评价
        并验证关键指标满足回归断言。

        标记为 slow, 需显式启用:
            pytest experiments/test_regression.py -v -m benchmark
        """
        from benchmarks import (
            get_test_case_by_name,
            AlgorithmSpec,
            DEFAULT_ALGORITHMS,
            BenchmarkRunner,
            EvaluationProtocol,
        )

        tc = get_test_case_by_name(case_name)
        assert tc is not None, f"测试用例不存在: {case_name}"

        algo_spec = None
        for a in DEFAULT_ALGORITHMS:
            if a.name == algo_name:
                algo_spec = a
                break
        assert algo_spec is not None, f"算法规格不存在: {algo_name}"

        protocol = EvaluationProtocol(
            epe_threshold=tc.reference_metrics.get('epe_mean_nm', 5.0) * 1.5,
            mse_threshold=0.05,
            compute_pw=False,
            pixel_size=tc.optical_params.get('pixel_size', 1.0),
        )

        runner = BenchmarkRunner(protocol=protocol, algorithms=[algo_spec])
        result = runner.run_single(tc, algo_spec)

        assert result.success, f"基准测试执行失败: {result.error_message}"

        assert result.epe.epe_mean < tc.reference_metrics.get('epe_mean_nm', 10.0) * 2.0, \
            f"EPE 回归失败: {result.epe.epe_mean:.2f}nm 超过阈值"

        if result.convergence.convergence_step is not None:
            assert result.convergence.convergence_step <= tc.optimizer_defaults.get('max_iter', 200), \
                f"收敛步数回归失败: {result.convergence.convergence_step}"

    def test_benchmark_report_save(self, tmp_path):
        from benchmarks.protocol import (
            BenchmarkResult, ConvergenceMetric, EPEMetric,
            PWMetric, TimeMetric,
        )
        from benchmarks.report import save_report

        results = [
            BenchmarkResult(
                test_case_name='ls_45nm',
                algorithm_name='gd',
                convergence=ConvergenceMetric(converged=True, total_iterations=50),
                epe=EPEMetric(epe_mean=2.0),
                pw=PWMetric(pw_ratio=0.3),
                time=TimeMetric(total_wall_time=1.0),
                success=True,
            ),
        ]

        saved = save_report(results, str(tmp_path), fmt='all')
        assert len(saved) == 2
        from pathlib import Path
        assert (Path(saved[0])).exists()
        assert (Path(saved[1])).exists()
