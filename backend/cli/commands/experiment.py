# -*- coding: utf-8 -*-
"""
子命令: experiment
实验编排与回归测试 CLI

对应原 experiments/run_experiments.py
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Optional

import click

from ..common import (
    global_options, setup_cli_logger, print_banner,
)


@click.command(
    name="experiment",
    help="配置驱动的实验编排与回归测试：批量执行 YAML 定义的实验并生成断言"
)
@global_options
@click.argument(
    "path",
    type=click.Path(exists=True),
    default="experiments/",
    required=False,
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(file_okay=False),
    default="./experiment_results",
    show_default=True,
    help="结果输出目录"
)
@click.option(
    "-g", "--generate-golden",
    is_flag=True,
    default=False,
    help="生成 golden 参考文件（用于后续回归对比）"
)
@click.option(
    "--verify-only",
    is_flag=True,
    default=False,
    help="仅验证（不重新执行，使用已有结果与 golden 对比）"
)
@click.pass_context
def experiment_cmd(
    ctx,
    verbose, log_file, config_path,
    path, output_dir, generate_golden, verify_only,
):
    """实验编排 CLI"""

    logger = setup_cli_logger("experiment_runner", verbose, log_file)
    print_banner(logger, "配置驱动的实验编排与回归测试")

    # --- 延迟导入 ---
    try:
        from experiments.schema import load_experiment, validate_experiment
        from experiments.executor import ExperimentExecutor, ExperimentResult
        from experiments.assertions import (
            RegressionAssertions, AssertionReport,
        )
    except Exception as e:
        logger.error(f"导入实验模块失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

    # --- 收集实验定义 ---
    def collect(p: str) -> List[Path]:
        pp = Path(p)
        if pp.is_file():
            return [pp]
        if pp.is_dir():
            return sorted(pp.glob("exp_*.yaml"))
        raise FileNotFoundError(f"路径不存在: {p}")

    experiments = collect(path)
    if not experiments:
        logger.error(f"未找到实验定义文件: {path}")
        sys.exit(1)

    logger.info(f"发现 {len(experiments)} 个实验定义")
    for exp in experiments:
        logger.info(f"  - {exp.name}")

    # --- 执行器 ---
    executor = ExperimentExecutor(base_output_dir=output_dir)
    assertions = RegressionAssertions(
        experiments_dir=str(
            Path(path).parent if Path(path).is_file() else path
        )
    )

    # --- 运行单个实验 ---
    def run_single(
        experiment_path: Path,
        gen_golden: bool = False,
    ) -> dict:
        name = experiment_path.stem
        click.echo("")
        click.echo("=" * 70)
        click.echo(f"实验: {name}")
        click.echo("=" * 70)

        try:
            experiment = load_experiment(experiment_path)
        except Exception as e:
            click.echo(f"  [失败] 加载实验定义失败: {e}")
            return {"name": name, "status": "load_error", "error": str(e)}

        errors = validate_experiment(experiment)
        if errors:
            click.echo(f"  [警告] 验证问题: {'; '.join(errors)}")

        click.echo(f"  工作流: {experiment.workflow}")
        click.echo(
            f"  图案: {experiment.pattern.type} "
            f"(cd={experiment.pattern.cd}, pitch={experiment.pattern.pitch})"
        )
        click.echo(
            f"  优化器: {experiment.optimizer.type} "
            f"(max_iter={experiment.optimizer.max_iter})"
        )
        click.echo(f"  断言数: {len(experiment.assertions)}")

        if verify_only:
            click.echo("  [验证模式] 使用已有结果，不重新执行")
            result_dir = Path(output_dir) / name
            golden_path = result_dir / "golden.json"
            if not golden_path.exists():
                click.echo(f"  [失败] Golden 文件不存在: {golden_path}")
                return {
                    "name": name,
                    "status": "missing_golden",
                    "error": "Golden 文件不存在",
                }
            with open(golden_path, "r", encoding="utf-8") as f:
                golden = json.load(f)
            result = ExperimentResult(
                success=True,
                output_dir=str(result_dir),
                final_mse=golden.get("final_mse"),
                final_ssim=golden.get("final_ssim"),
                converged=golden.get("converged"),
                convergence_step=golden.get("convergence_step"),
                final_loss=golden.get("final_loss"),
                total_iterations=golden.get("total_iterations", 0),
                custom_metrics={
                    k: v for k, v in golden.items()
                    if k not in {
                        "experiment_name", "final_mse", "final_ssim",
                        "converged", "convergence_step", "final_loss",
                        "total_iterations",
                    }
                },
            )
            elapsed = 0.0
        else:
            start = time.time()
            result = executor.run(experiment)
            elapsed = time.time() - start

            if not result.success:
                click.echo(f"  [失败] 执行错误: {result.error_message}")
                return {
                    "name": name,
                    "status": "execution_error",
                    "error": result.error_message,
                    "time": elapsed,
                }

            click.echo(f"  耗时: {elapsed:.2f}s")
            click.echo(f"  迭代: {result.total_iterations}")
            if result.final_mse is not None:
                click.echo(f"  最终 MSE: {result.final_mse:.6e}")
            if result.final_ssim is not None:
                click.echo(f"  最终 SSIM: {result.final_ssim:.6f}")
            if result.converged is not None:
                click.echo(f"  收敛: {result.converged}")

            if gen_golden and result.output_dir:
                gp = Path(result.output_dir) / "golden.json"
                golden_data = {
                    "experiment_name": experiment.name,
                    "final_mse": result.final_mse,
                    "final_ssim": result.final_ssim,
                    "converged": result.converged,
                    "convergence_step": result.convergence_step,
                    "final_loss": result.final_loss,
                    "total_iterations": result.total_iterations,
                }
                golden_data.update(result.custom_metrics)
                with open(gp, "w", encoding="utf-8") as f:
                    json.dump(golden_data, f, indent=2, ensure_ascii=False)
                click.echo(f"  Golden 已生成: {gp}")

        report = assertions.evaluate(
            experiment_name=experiment.name,
            assertions=experiment.assertions,
            result=result,
            result_dir=result.output_dir,
        )

        for ar in report.results:
            status = "通过" if ar.passed else "失败"
            click.echo(f"  断言 [{ar.assertion_type}]: {status} - {ar.message}")

        return {
            "name": name,
            "status": "passed" if report.all_passed else "assertion_failed",
            "time": elapsed if not verify_only else 0.0,
            "report": report.to_dict(),
        }

    # --- 批量执行 ---
    summaries = []
    for exp_path in experiments:
        summary = run_single(exp_path, generate_golden)
        summaries.append(summary)

    # --- 汇总 ---
    click.echo("")
    click.echo("=" * 70)
    click.echo("汇总报告")
    click.echo("=" * 70)

    passed = sum(1 for s in summaries if s["status"] == "passed")
    failed = sum(1 for s in summaries if s["status"] != "passed")
    total = len(summaries)

    for s in summaries:
        status_icon = "ok" if s["status"] == "passed" else "FAIL"
        click.echo(
            f"  [{status_icon}] {s['name']} ({s.get('time', 0):.2f}s)"
        )

    click.echo("")
    click.echo(f"通过: {passed}/{total}")
    if failed > 0:
        click.echo(f"失败: {failed}/{total}")
        sys.exit(1)
    else:
        click.echo("全部通过")
        return 0
