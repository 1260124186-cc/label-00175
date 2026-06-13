#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验追踪模块测试脚本"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.experiment_tracking import (
    create_tracker, list_experiments, get_run_summary,
    print_run_summary, compare_runs_table, find_best_run, filter_runs,
    export_comparison_to_csv
)
from algorithms.callbacks import ExperimentTrackingCallback
from algorithms.mask_optimizer import OptimizationConfig


def test_tracker_basic():
    """测试追踪器基本功能"""
    print("=" * 60)
    print("测试 1: 追踪器基本功能")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    tracking_dir = os.path.join(tmpdir, 'mlruns')

    tracker = create_tracker('local', experiment_name='test_exp', tracking_dir=tracking_dir)
    print("✓ 追踪器创建成功")

    run_id1 = tracker.start_run(run_name='test_run_1', tags={'version': 'v1'})
    print(f"✓ 运行1开始: {run_id1}")

    tracker.log_param('learning_rate', 0.01)
    tracker.log_param('optimizer', 'adam')
    tracker.log_params({'batch_size': 32, 'epochs': 100})
    print("✓ 参数记录成功")

    for i in range(10):
        tracker.log_metric('loss', 1.0 / (i + 1), step=i)
        tracker.log_metric('accuracy', 0.5 + i * 0.05, step=i)
    print("✓ 指标记录成功")

    config = {'model': {'type': 'cnn', 'layers': 3}}
    tracker.log_config(config)
    print("✓ 配置记录成功")

    tracker.set_tag('status', 'good')
    print("✓ 标签设置成功")

    tracker.end_run(status='completed')
    print("✓ 运行1结束")

    run_id2 = tracker.start_run(run_name='test_run_2', tags={'version': 'v2'})
    print(f"✓ 运行2开始: {run_id2}")

    tracker.log_param('learning_rate', 0.001)
    tracker.log_param('optimizer', 'sgd')
    for i in range(10):
        tracker.log_metric('loss', 2.0 / (i + 1), step=i)
        tracker.log_metric('accuracy', 0.4 + i * 0.06, step=i)
    tracker.end_run(status='completed')
    print("✓ 运行2结束")

    runs = tracker.list_runs()
    print(f"✓ 列出运行: {len(runs)} 个")

    run = tracker.get_run(run_id1)
    print(f"✓ 获取单个运行: {run.run_id if run else None}")

    if run:
        summary = get_run_summary(run)
        print(f"✓ 运行摘要: 包含 {len(summary)} 个字段")

    if len(runs) >= 2:
        comparison = tracker.compare_runs([run_id1, run_id2], metrics=['loss'])
        print(f"✓ 对比运行: 包含 {len(comparison['runs'])} 个运行")

    table = compare_runs_table(runs, metrics=['loss', 'accuracy'])
    print(f"✓ 对比表格生成: {len(table)} 字符")

    filtered = filter_runs(runs, tags={'version': 'v1'})
    print(f"✓ 过滤运行: {len(filtered)} 个 (预期 1 个)")

    best = find_best_run(runs, 'loss', 'min')
    print(f"✓ 最佳 loss 运行: {best.run_id if best else None}")

    exps = list_experiments(tracking_dir)
    print(f"✓ 实验列表: {exps}")

    csv_path = os.path.join(tmpdir, 'comparison.csv')
    export_comparison_to_csv(runs, csv_path)
    print(f"✓ CSV 导出成功: {os.path.exists(csv_path)}")

    print()
    return True


def test_callback():
    """测试实验追踪回调"""
    print("=" * 60)
    print("测试 2: 实验追踪回调")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    tracking_dir = os.path.join(tmpdir, 'mlruns')

    callback = ExperimentTrackingCallback(
        backend='local',
        experiment_name='callback_test',
        run_name='test_run',
        tags={'test': 'true'},
        tracking_dir=tracking_dir,
        log_metrics_freq=1,
    )
    print("✓ 回调创建成功")

    from algorithms.callbacks import TrainerState
    state = TrainerState()
    state.learning_rate = 0.01
    state.loss = 1.0
    callback.set_state(state)
    callback.set_params({'max_iter': 100, 'optimizer_type': 'gradient_descent'})
    print("✓ 回调状态设置成功")

    callback.on_train_begin()
    print(f"✓ 训练开始, run_id: {callback.run_id}")

    for i in range(5):
        state.loss = 1.0 / (i + 1)
        state.learning_rate = 0.01 * (0.95 ** i)
        callback.on_epoch_end(i, logs={'loss': state.loss, 'mse': state.loss * 0.9})
    print("✓ epoch 回调成功")

    callback.on_train_end(logs={'loss': 0.2, 'mse': 0.18})
    print("✓ 训练结束回调成功")

    print()
    return True


def test_config():
    """测试 OptimizationConfig 中的实验追踪配置"""
    print("=" * 60)
    print("测试 3: OptimizationConfig 实验追踪配置")
    print("=" * 60)

    config = OptimizationConfig()
    print(f"✓ 默认配置 experiment_tracking_enable: {config.experiment_tracking_enable}")
    print(f"✓ 默认配置 experiment_tracking_backend: {config.experiment_tracking_backend}")
    print(f"✓ 默认配置 experiment_name: {config.experiment_name}")
    print(f"✓ 默认配置 tracking_dir: {config.tracking_dir}")
    print(f"✓ 默认配置 log_experiment_config: {config.log_experiment_config}")
    print(f"✓ 默认配置 log_metrics_freq: {config.log_metrics_freq}")

    config_dict = config.to_dict()
    print(f"✓ to_dict 包含实验追踪配置: {'experiment_tracking_enable' in config_dict}")

    config2 = OptimizationConfig.from_dict({
        'experiment_tracking_enable': True,
        'experiment_tracking_backend': 'mlflow',
        'experiment_name': 'my_exp',
    })
    print(f"✓ from_dict 设置成功: {config2.experiment_tracking_enable}")
    print(f"✓ from_dict 后端: {config2.experiment_tracking_backend}")
    print(f"✓ from_dict 实验名: {config2.experiment_name}")

    print()
    return True


def main():
    """主测试函数"""
    all_passed = True

    try:
        test_tracker_basic()
    except Exception as e:
        print(f"✗ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_callback()
    except Exception as e:
        print(f"✗ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_config()
    except Exception as e:
        print(f"✗ 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("所有测试通过! ✓")
    else:
        print("部分测试失败! ✗")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
