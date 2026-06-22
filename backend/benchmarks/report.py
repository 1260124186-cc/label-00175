# -*- coding: utf-8 -*-
"""
逆向光刻基准测试套件 - 报告生成

生成跨算法对比表格与摘要报告:
1. 对齐表格: 按 (测试用例 x 算法) 展示四维指标
2. 排名表: 每个测试用例下各算法排名
3. 总分: 跨测试用例综合排名
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from benchmarks.protocol import BenchmarkResult


def generate_comparison_table(results: List[BenchmarkResult]) -> str:
    """
    生成对齐的文本对比表格

    Args:
        results: 基准测试结果列表

    Returns:
        格式化的表格字符串
    """
    if not results:
        return "(无结果)"

    by_case: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        by_case.setdefault(r.test_case_name, []).append(r)

    headers = [
        ('test_case', 'Test Case', 28),
        ('algorithm', 'Algorithm', 14),
        ('epe_mean', 'EPE(nm)', 10),
        ('epe_max', 'EPE_max', 10),
        ('pw_ratio', 'PW%', 8),
        ('dof', 'DOF(nm)', 10),
        ('conv_step', 'Conv', 6),
        ('iter', 'Iter', 6),
        ('time', 'Time(s)', 10),
        ('status', 'Status', 7),
    ]

    sep = '+' + '+'.join('-' * (w + 2) for _, _, w in headers) + '+'
    hdr = '|' + '|'.join(f' {h:^{w}} ' for _, h, w in headers) + '|'

    lines = [sep, hdr, sep]

    for case_name, case_results in sorted(by_case.items()):
        for r in case_results:
            if r.success:
                epe_mean = f'{r.epe.epe_mean:.2f}'
                epe_max = f'{r.epe.epe_max:.2f}'
                pw_ratio = f'{r.pw.pw_ratio * 100:.1f}'
                dof = f'{r.pw.depth_of_focus:.1f}'
                conv = str(r.convergence.convergence_step or '-')
                it = str(r.convergence.total_iterations)
                t = f'{r.time.total_wall_time:.3f}'
                status = 'OK'
            else:
                epe_mean = epe_max = pw_ratio = dof = conv = it = t = '-'
                status = 'FAIL'

            cells = [
                (case_name[:28], 28),
                (r.algorithm_name[:14], 14),
                (epe_mean, 10),
                (epe_max, 10),
                (pw_ratio, 8),
                (dof, 10),
                (conv, 6),
                (it, 6),
                (t, 10),
                (status, 7),
            ]
            row = '|' + '|'.join(f' {c:<{w}} ' for c, w in cells) + '|'
            lines.append(row)
        lines.append(sep)

    return '\n'.join(lines)


def generate_ranking(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """
    生成每个测试用例下的算法排名

    排名依据 (越小越好): epe_mean -> pw_ratio (大好) -> wall_time (小好)
    """
    by_case: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        by_case.setdefault(r.test_case_name, []).append(r)

    ranking = {}
    for case_name, case_results in sorted(by_case.items()):
        successful = [r for r in case_results if r.success]
        if not successful:
            ranking[case_name] = []
            continue

        sorted_by_epe = sorted(successful, key=lambda r: r.epe.epe_mean)
        sorted_by_pw = sorted(successful, key=lambda r: -r.pw.pw_ratio)
        sorted_by_time = sorted(successful, key=lambda r: r.time.total_wall_time)

        rank_list = []
        for i, r in enumerate(sorted_by_epe):
            epe_rank = i + 1
            pw_rank = next(j + 1 for j, rr in enumerate(sorted_by_pw) if rr.algorithm_name == r.algorithm_name)
            time_rank = next(j + 1 for j, rr in enumerate(sorted_by_time) if rr.algorithm_name == r.algorithm_name)
            rank_list.append({
                'algorithm': r.algorithm_name,
                'epe_rank': epe_rank,
                'pw_rank': pw_rank,
                'time_rank': time_rank,
                'composite_rank': epe_rank + pw_rank + time_rank,
            })

        rank_list.sort(key=lambda x: x['composite_rank'])
        ranking[case_name] = rank_list

    return ranking


def generate_summary(results: List[BenchmarkResult]) -> str:
    """
    生成完整摘要文本

    Args:
        results: 基准测试结果列表

    Returns:
        格式化摘要字符串
    """
    n_total = len(results)
    n_success = sum(1 for r in results if r.success)

    lines = [
        "=" * 70,
        "逆向光刻基准测试 - 结果摘要",
        "=" * 70,
        f"总组合数: {n_total}, 成功: {n_success}, 失败: {n_total - n_success}",
        "",
    ]

    for r in results:
        lines.append(r.summary())

    lines.append("")
    lines.append(generate_comparison_table(results))

    ranking = generate_ranking(results)
    if ranking:
        lines.append("")
        lines.append("=" * 50)
        lines.append("算法排名 (综合 EPE + PW + 时间)")
        lines.append("=" * 50)
        for case_name, rank_list in ranking.items():
            lines.append(f"\n  [{case_name}]")
            for entry in rank_list:
                lines.append(
                    f"    #{entry['composite_rank']:2d} {entry['algorithm']:<14s} "
                    f"EPE_rank={entry['epe_rank']} PW_rank={entry['pw_rank']} "
                    f"Time_rank={entry['time_rank']}"
                )

    return '\n'.join(lines)


def save_report(
    results: List[BenchmarkResult],
    output_dir: str,
    fmt: str = 'all',
) -> List[str]:
    """
    保存基准测试报告

    Args:
        results: 结果列表
        output_dir: 输出目录
        fmt: 输出格式 ('json', 'text', 'all')

    Returns:
        保存的文件路径列表
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    if fmt in ('json', 'all'):
        data = [r.to_dict() for r in results]
        path = out / 'benchmark_results.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        saved.append(str(path))

    if fmt in ('text', 'all'):
        path = out / 'benchmark_report.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(generate_summary(results))
        saved.append(str(path))

    return saved
