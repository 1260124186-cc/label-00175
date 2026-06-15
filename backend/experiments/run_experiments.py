# -*- coding: utf-8 -*-
"""
实验编排 CLI 入口

支持命令行批量执行实验定义、生成 golden 参考文件、运行回归验证。

用法:
    # 执行指定实验
    python -m experiments.run_experiments experiments/exp_line_space_mse.yaml

    # 执行目录下所有实验
    python -m experiments.run_experiments experiments/

    # 生成 golden 参考文件
    python -m experiments.run_experiments experiments/ --generate-golden

    # 指定输出目录
    python -m experiments.run_experiments experiments/ --output-dir ./my_results

    # 仅验证（不重新执行，使用已有结果）
    python -m experiments.run_experiments experiments/ --verify-only
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional

from experiments.schema import load_experiment, validate_experiment
from experiments.executor import ExperimentExecutor, ExperimentResult
from experiments.assertions import RegressionAssertions, AssertionReport

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def collect_experiments(path: str) -> List[Path]:
    """收集实验定义文件"""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob('exp_*.yaml'))
    raise FileNotFoundError(f"路径不存在: {path}")


def run_single(experiment_path: Path,
               executor: ExperimentExecutor,
               assertions: RegressionAssertions,
               generate_golden: bool = False) -> dict:
    """执行单个实验并返回汇总"""
    name = experiment_path.stem
    print(f"\n{'=' * 70}")
    print(f"实验: {name}")
    print(f"{'=' * 70}")

    try:
        experiment = load_experiment(experiment_path)
    except Exception as e:
        print(f"  [失败] 加载实验定义失败: {e}")
        return {'name': name, 'status': 'load_error', 'error': str(e)}

    errors = validate_experiment(experiment)
    if errors:
        print(f"  [警告] 验证问题: {'; '.join(errors)}")

    print(f"  工作流: {experiment.workflow}")
    print(f"  图案: {experiment.pattern.type} "
          f"(cd={experiment.pattern.cd}, pitch={experiment.pattern.pitch})")
    print(f"  优化器: {experiment.optimizer.type} "
          f"(max_iter={experiment.optimizer.max_iter})")
    print(f"  断言数: {len(experiment.assertions)}")

    start = time.time()
    result = executor.run(experiment)
    elapsed = time.time() - start

    if not result.success:
        print(f"  [失败] 执行错误: {result.error_message}")
        return {
            'name': name,
            'status': 'execution_error',
            'error': result.error_message,
            'time': elapsed,
        }

    print(f"  耗时: {elapsed:.2f}s")
    print(f"  迭代: {result.total_iterations}")
    if result.final_mse is not None:
        print(f"  最终 MSE: {result.final_mse:.6e}")
    if result.final_ssim is not None:
        print(f"  最终 SSIM: {result.final_ssim:.6f}")
    if result.converged is not None:
        print(f"  收敛: {result.converged}")

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
        print(f"  Golden 已生成: {golden_path}")

    report = assertions.evaluate(
        experiment_name=experiment.name,
        assertions=experiment.assertions,
        result=result,
        result_dir=result.output_dir,
    )

    for ar in report.results:
        status = "通过" if ar.passed else "失败"
        print(f"  断言 [{ar.assertion_type}]: {status} - {ar.message}")

    return {
        'name': name,
        'status': 'passed' if report.all_passed else 'assertion_failed',
        'time': elapsed,
        'report': report.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='配置驱动的实验编排与回归测试 CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='experiments/',
        help='实验定义文件或目录（默认: experiments/）',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./experiment_results',
        help='结果输出目录（默认: ./experiment_results）',
    )
    parser.add_argument(
        '--generate-golden', '-g',
        action='store_true',
        help='生成 golden 参考文件',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细日志输出',
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    experiments = collect_experiments(args.path)
    if not experiments:
        print(f"未找到实验定义文件: {args.path}")
        sys.exit(1)

    print(f"发现 {len(experiments)} 个实验定义")

    executor = ExperimentExecutor(base_output_dir=args.output_dir)
    assertions = RegressionAssertions(
        experiments_dir=str(Path(args.path).parent
                           if Path(args.path).is_file()
                           else args.path)
    )

    summaries = []
    for exp_path in experiments:
        summary = run_single(exp_path, executor, assertions, args.generate_golden)
        summaries.append(summary)

    print(f"\n{'=' * 70}")
    print("汇总报告")
    print(f"{'=' * 70}")

    passed = sum(1 for s in summaries if s['status'] == 'passed')
    failed = sum(1 for s in summaries if s['status'] != 'passed')
    total = len(summaries)

    for s in summaries:
        status_icon = "ok" if s['status'] == 'passed' else "FAIL"
        print(f"  [{status_icon}] {s['name']} ({s.get('time', 0):.2f}s)")

    print(f"\n通过: {passed}/{total}")
    if failed > 0:
        print(f"失败: {failed}/{total}")
        sys.exit(1)
    else:
        print("全部通过")


if __name__ == '__main__':
    main()
