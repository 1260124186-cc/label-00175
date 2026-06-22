# -*- coding: utf-8 -*-
"""
逆向光刻基准测试套件

提供业界标准测试用例与统一评价协议，支持不同优化算法的公平对比。

主要组件:
    1. test_cases: 标准测试用例定义 (Line/Space, ContactHole, Logic, SRAM, Dense/Isolated, ThroughFocus)
    2. protocol: 统一四维评价协议 (收敛速度, 最终EPE, PW面积, 计算时间)
    3. runner: 基准运行器, 编排测试用例 x 算法的交叉评价
    4. report: 报告生成, 对比表格与排名

使用示例:
    from benchmarks import BenchmarkRunner, AlgorithmSpec, EvaluationProtocol

    protocol = EvaluationProtocol(compute_pw=False)
    runner = BenchmarkRunner(protocol=protocol)
    results = runner.run_all()
    print(runner.generate_summary())
"""

from benchmarks.test_cases import (
    BenchmarkTestCase,
    TestCaseCategory,
    DifficultyLevel,
    get_all_test_cases,
    get_test_cases_by_category,
    get_test_cases_by_difficulty,
    get_test_case_by_name,
)
from benchmarks.protocol import (
    ConvergenceMetric,
    EPEMetric,
    PWMetric,
    TimeMetric,
    BenchmarkResult,
    EvaluationProtocol,
    compare_results,
)
from benchmarks.runner import (
    AlgorithmSpec,
    DEFAULT_ALGORITHMS,
    BenchmarkRunner,
)
from benchmarks.report import (
    generate_comparison_table,
    generate_ranking,
    generate_summary,
    save_report,
)

__all__ = [
    'BenchmarkTestCase',
    'TestCaseCategory',
    'DifficultyLevel',
    'get_all_test_cases',
    'get_test_cases_by_category',
    'get_test_cases_by_difficulty',
    'get_test_case_by_name',
    'ConvergenceMetric',
    'EPEMetric',
    'PWMetric',
    'TimeMetric',
    'BenchmarkResult',
    'EvaluationProtocol',
    'compare_results',
    'AlgorithmSpec',
    'DEFAULT_ALGORITHMS',
    'BenchmarkRunner',
    'generate_comparison_table',
    'generate_ranking',
    'generate_summary',
    'save_report',
]
