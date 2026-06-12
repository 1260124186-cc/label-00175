#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
算法对比实验框架：多种优化器在相同初始掩模与目标下的批量对比。

功能：
1. 固定随机种子，生成统一的初始掩模与目标图案
2. 依次运行多种优化器（梯度下降、Adam、RMSprop、BFGS、牛顿、遗传、PSO、SA、DE、CMA-ES）
3. 收集 MSE、SSIM、耗时、迭代次数、收敛状态等指标
4. 打印对齐的对比表格
5. 通过 save_results 写入 JSON（包含对比表、各优化器 loss_history、最终掩模等）

运行方式：
    python -m examples.algorithm_comparison
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.imaging import OpticalSystem
from core.metrics import evaluate_all
from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig
from utils.data_io import create_test_pattern
from utils.logger import setup_logger
from utils.config import save_results, create_default_config, save_config


DEFAULT_OPTIMIZERS: List[Dict[str, Any]] = [
    {
        'name': 'gradient_descent',
        'label': 'Gradient Descent',
        'config': {
            'optimizer_type': 'gradient_descent',
            'learning_rate': 0.1,
        },
    },
    {
        'name': 'adam',
        'label': 'Adam',
        'config': {
            'optimizer_type': 'adam',
            'learning_rate': 0.05,
        },
    },
    {
        'name': 'rmsprop',
        'label': 'RMSprop',
        'config': {
            'optimizer_type': 'rmsprop',
            'learning_rate': 0.05,
        },
    },
    {
        'name': 'bfgs',
        'label': 'BFGS (L-BFGS-B)',
        'config': {
            'optimizer_type': 'bfgs',
        },
    },
    {
        'name': 'newton',
        'label': 'Newton',
        'config': {
            'optimizer_type': 'newton',
        },
    },
    {
        'name': 'genetic',
        'label': 'Genetic Algorithm',
        'config': {
            'optimizer_type': 'genetic',
            'population_size': 30,
        },
    },
    {
        'name': 'pso',
        'label': 'Particle Swarm',
        'config': {
            'optimizer_type': 'pso',
            'population_size': 30,
        },
    },
    {
        'name': 'sa',
        'label': 'Simulated Annealing',
        'config': {
            'optimizer_type': 'sa',
        },
    },
    {
        'name': 'de',
        'label': 'Differential Evolution',
        'config': {
            'optimizer_type': 'de',
            'population_size': 30,
        },
    },
    {
        'name': 'cma_es',
        'label': 'CMA-ES',
        'config': {
            'optimizer_type': 'cma_es',
            'population_size': 30,
        },
    },
]


def build_base_config(max_iter: int = 50,
                      metric: str = 'mse',
                      tol: float = 1e-7,
                      random_seed: int = 42,
                      bounds: tuple = (0.0, 1.0),
                      verbose: bool = False) -> OptimizationConfig:
    return OptimizationConfig(
        max_iter=max_iter,
        tol=tol,
        metric=metric,
        bounds=bounds,
        verbose=verbose,
        random_seed=random_seed,
        use_callbacks=False,
        early_stopping_enable=False,
        lr_scheduler=None,
    )


def print_comparison_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        ('optimizer', 'Optimizer', 24),
        ('final_mse', 'Final MSE', 14),
        ('final_ssim', 'Final SSIM', 12),
        ('mse_improve', 'MSE Improv.%', 14),
        ('iterations', 'Iterations', 11),
        ('time_s', 'Time (s)', 11),
        ('converged', 'Converged', 10),
    ]

    separator = '+' + '+'.join('-' * (w + 2) for _, _, w in headers) + '+'
    header_line = '|' + '|'.join(
        f' {h:^{w}} ' for _, h, w in headers
    ) + '|'

    print()
    print('=' * len(separator))
    print('优化器对比结果')
    print('=' * len(separator))
    print(separator)
    print(header_line)
    print(separator)

    for row in rows:
        cells = []
        for key, _, width in headers:
            val = row.get(key, '')
            if key == 'final_mse':
                cells.append(f' {val:.4e} '.rjust(width + 2))
            elif key == 'final_ssim':
                cells.append(f' {val:.6f} '.rjust(width + 2))
            elif key == 'mse_improve':
                cells.append(f' {val:+.2f}% '.rjust(width + 2))
            elif key == 'time_s':
                cells.append(f' {val:.3f} '.rjust(width + 2))
            elif key == 'iterations':
                cells.append(f' {int(val)} '.rjust(width + 2))
            elif key == 'converged':
                cells.append(f' {str(bool(val)):^{width}} ')
            else:
                cells.append(f' {str(val):<{width}} ')
        print('|' + '|'.join(cells) + '|')

    print(separator)
    print()


def run_single_comparison(
    optimizer_spec: Dict[str, Any],
    initial_mask: np.ndarray,
    target_pattern: np.ndarray,
    optical_system: OpticalSystem,
    base_config: OptimizationConfig,
    logger,
) -> Dict[str, Any]:
    name = optimizer_spec['name']
    label = optimizer_spec['label']
    extra_cfg = optimizer_spec.get('config', {})

    cfg = OptimizationConfig(
        optimizer_type=extra_cfg.get('optimizer_type', name),
        max_iter=base_config.max_iter,
        tol=base_config.tol,
        metric=base_config.metric,
        bounds=base_config.bounds,
        verbose=False,
        random_seed=base_config.random_seed,
        use_callbacks=False,
        early_stopping_enable=False,
        lr_scheduler=None,
        learning_rate=extra_cfg.get('learning_rate', base_config.learning_rate),
        population_size=extra_cfg.get('population_size', base_config.population_size),
    )

    optimizer = MaskOptimizer(optical_system=optical_system, config=cfg)

    logger.info(f"[{label}] 开始优化...")
    t_start = time.perf_counter()
    try:
        result = optimizer.optimize(
            initial_mask=initial_mask,
            target_image=target_pattern,
        )
        elapsed = time.perf_counter() - t_start
        success = True
        err_msg = ''
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        logger.error(f"[{label}] 优化失败: {e}")
        result = None
        success = False
        err_msg = str(e)

    row: Dict[str, Any] = {
        'name': name,
        'label': label,
        'optimizer': label,
        'success': success,
        'error': err_msg,
    }

    if result is not None:
        initial_mse = float(result.initial_metrics.mse)
        final_mse = float(result.final_metrics.mse)
        mse_improve = (
            (initial_mse - final_mse) / initial_mse * 100.0
            if initial_mse > 1e-20 else 0.0
        )

        row.update({
            'initial_mse': initial_mse,
            'final_mse': final_mse,
            'initial_ssim': float(result.initial_metrics.ssim),
            'final_ssim': float(result.final_metrics.ssim),
            'initial_mae': float(result.initial_metrics.mae),
            'final_mae': float(result.final_metrics.mae),
            'initial_psnr': float(result.initial_metrics.psnr),
            'final_psnr': float(result.final_metrics.psnr),
            'mse_improve': mse_improve,
            'iterations': int(result.total_iterations),
            'time_s': float(elapsed),
            'converged': bool(result.converged),
            'message': result.message,
            'loss_history': list(result.loss_history),
            'optimized_mask': result.optimized_mask,
            'final_wafer_image': result.final_wafer_image,
        })

        logger.info(
            f"[{label}] 完成: MSE={final_mse:.4e}, "
            f"SSIM={row['final_ssim']:.4f}, "
            f"迭代={row['iterations']}, "
            f"耗时={elapsed:.3f}s, "
            f"收敛={row['converged']}"
        )
    else:
        row.update({
            'initial_mse': float('nan'),
            'final_mse': float('nan'),
            'initial_ssim': float('nan'),
            'final_ssim': float('nan'),
            'mse_improve': float('nan'),
            'iterations': 0,
            'time_s': float(elapsed),
            'converged': False,
            'loss_history': [],
        })

    return row


def main(
    optimizer_specs: Optional[List[Dict[str, Any]]] = None,
    image_size: tuple = (64, 64),
    pattern_type: str = 'rectangle',
    pattern_kwargs: Optional[Dict[str, Any]] = None,
    max_iter: int = 50,
    noise_level: float = 0.1,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    logger = setup_logger('algo_cmp', log_file=None)
    logger.info("=" * 70)
    logger.info("算法对比实验框架 - 多优化器批量对比")
    logger.info("=" * 70)

    if optimizer_specs is None:
        optimizer_specs = DEFAULT_OPTIMIZERS

    if output_dir is None:
        output_dir = Path('results') / 'algorithm_comparison'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pattern_kwargs is None:
        pattern_kwargs = {
            'x_start': 20, 'x_end': 44,
            'y_start': 20, 'y_end': 44,
        }

    np.random.seed(random_seed)

    logger.info(f"图像尺寸: {image_size}")
    logger.info(f"图案类型: {pattern_type}")
    logger.info(f"最大迭代: {max_iter}")
    logger.info(f"随机种子: {random_seed}")
    logger.info(f"参与对比的优化器: {len(optimizer_specs)}")
    for s in optimizer_specs:
        logger.info(f"  - {s['label']} ({s['name']})")

    target_pattern = create_test_pattern(
        pattern_type, size=image_size, **pattern_kwargs
    )

    initial_mask = target_pattern + noise_level * np.random.randn(*image_size)
    initial_mask = np.clip(initial_mask, 0.0, 1.0)

    optical_system = OpticalSystem(
        wavelength=193.0,
        na=1.35,
        sigma=0.75,
        pixel_size=1.0,
        defocus=0.0,
    )

    base_config = build_base_config(
        max_iter=max_iter,
        metric='mse',
        tol=1e-7,
        random_seed=random_seed,
        verbose=False,
    )

    baseline_config = OptimizationConfig(
        optimizer_type='gradient_descent',
        max_iter=1,
        tol=1e-7,
        verbose=False,
        random_seed=random_seed,
        use_callbacks=False,
    )
    baseline_opt = MaskOptimizer(optical_system, baseline_config)
    baseline_result = baseline_opt.optimize(
        initial_mask=initial_mask, target_image=target_pattern
    )
    baseline_metrics = baseline_result.initial_metrics
    logger.info(
        f"初始状态: MSE={baseline_metrics.mse:.6e}, "
        f"SSIM={baseline_metrics.ssim:.4f}"
    )

    results: List[Dict[str, Any]] = []
    for spec in optimizer_specs:
        row = run_single_comparison(
            spec, initial_mask, target_pattern, optical_system,
            base_config, logger,
        )
        results.append(row)

    successful = [r for r in results if r['success']]
    print_comparison_table(results)

    if successful:
        best_mse = min(successful, key=lambda r: r['final_mse'])
        best_ssim = max(successful, key=lambda r: r['final_ssim'])
        best_speed = min(successful, key=lambda r: r['time_s'])

        logger.info("最佳性能总结:")
        logger.info(f"  最低 MSE:  {best_mse['label']} -> {best_mse['final_mse']:.4e}")
        logger.info(f"  最高 SSIM: {best_ssim['label']} -> {best_ssim['final_ssim']:.6f}")
        logger.info(f"  最快速度:  {best_speed['label']} -> {best_speed['time_s']:.3f}s")

    comparison_table: List[Dict[str, Any]] = []
    for r in results:
        comparison_table.append({
            'name': r['name'],
            'label': r['label'],
            'success': r['success'],
            'error': r.get('error', ''),
            'initial_mse': r.get('initial_mse'),
            'final_mse': r.get('final_mse'),
            'initial_ssim': r.get('initial_ssim'),
            'final_ssim': r.get('final_ssim'),
            'mse_improve_percent': r.get('mse_improve'),
            'iterations': r.get('iterations'),
            'time_seconds': r.get('time_s'),
            'converged': r.get('converged'),
            'message': r.get('message', ''),
        })

    save_dict: Dict[str, Any] = {
        'experiment_meta': {
            'image_size': list(image_size),
            'pattern_type': pattern_type,
            'pattern_kwargs': pattern_kwargs,
            'max_iter': max_iter,
            'noise_level': noise_level,
            'random_seed': random_seed,
            'baseline_mse': float(baseline_metrics.mse),
            'baseline_ssim': float(baseline_metrics.ssim),
            'num_optimizers': len(optimizer_specs),
        },
        'comparison_table': comparison_table,
        'target_pattern': target_pattern,
        'initial_mask': initial_mask,
    }

    for r in results:
        key = r['name']
        save_dict[f'{key}_loss_history'] = np.array(r.get('loss_history', []), dtype=np.float64)
        if r.get('optimized_mask') is not None:
            save_dict[f'{key}_optimized_mask'] = r['optimized_mask']
        if r.get('final_wafer_image') is not None:
            save_dict[f'{key}_final_wafer_image'] = r['final_wafer_image']

    saved_files = save_results(
        save_dict,
        output_dir,
        prefix='algorithm_comparison',
        save_arrays=True,
    )

    logger.info("保存的文件:")
    for k, p in saved_files.items():
        logger.info(f"  {k}: {p}")

    config_file = output_dir / 'experiment_config.yaml'
    full_config = create_default_config()
    full_config['comparison'] = {
        'image_size': list(image_size),
        'pattern_type': pattern_type,
        'pattern_kwargs': pattern_kwargs,
        'max_iter': max_iter,
        'noise_level': noise_level,
        'random_seed': random_seed,
        'optimizers': [
            {'name': s['name'], 'label': s['label'], 'config': s.get('config', {})}
            for s in optimizer_specs
        ],
    }
    save_config(full_config, config_file)
    logger.info(f"实验配置已保存: {config_file}")

    logger.info("=" * 70)
    logger.info("算法对比实验完成!")
    logger.info(f"结果目录: {output_dir.absolute()}")
    logger.info("=" * 70)

    return results


if __name__ == '__main__':
    main()
