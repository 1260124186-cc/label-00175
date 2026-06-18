#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层次化批处理端到端集成测试

覆盖 HierarchicalBatchRunner.run() 完整执行路径，包括：
- 层次化加载与建队
- 叶节点仿真任务提交到进程池
- worker 等待 > 0.5s 的路径（通过配置多轮迭代实现）
- 子结果缓存与父 cell 合成
- 结果写回与状态更新
"""
import sys
import os
import tempfile
import time
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from layout.layout_manager import LayoutManager
from pipeline.batch_runner import (
    run_batch_optimization,
    BatchConfig,
    ResourceConfig,
    HierarchicalBatchRunner,
    TaskStatus,
)


def test_hierarchical_batch_runner_full_flow():
    """
    端到端测试 HierarchicalBatchRunner 完整执行流程

    层次结构:
        TOP (depth 0) ─┬─ MID (depth 1) ── LEAF (depth 2, 2×2 阵列 = 4 实例)
                       └─ LEAF (depth 1, 单引用)

    预期:
    - LEAF 仿真 1 次（被 2 个父节点引用，但只需仿真 1 次）
    - MID 合成（复用 LEAF 结果，2×2 阵列 = 4 个实例）
    - TOP 合成（复用 MID 和 LEAF 结果）
    - 总仿真次数 = 1（LEAF），而不是扁平模式下的 3 次
    """
    print("\n" + "=" * 70)
    print("测试: HierarchicalBatchRunner 完整执行流程（端到端）")
    print("=" * 70)

    try:
        import gdstk
    except ImportError:
        print("  ⚠️  跳过：gdstk 不可用")
        return True

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            gds_path = os.path.join(tmpdir, "integration_test.gds")
            out_dir = os.path.join(tmpdir, "results")

            # 1. 创建层次化 GDS
            lib = gdstk.Library()
            leaf = lib.new_cell("LEAF")
            leaf.add(gdstk.rectangle((0, 0), (1000, 1000), layer=0))
            mid = lib.new_cell("MID")
            aref = gdstk.Reference("LEAF", origin=(0, 0), columns=2, rows=2,
                                   spacing=(2000, 2000))
            mid.add(aref)
            top = lib.new_cell("TOP")
            top.add(gdstk.Reference("MID", origin=(0, 0)))
            top.add(gdstk.Reference("LEAF", origin=(5000, 5000)))
            lib.write_gds(gds_path)
            print(f"  ✓ 创建测试 GDS: {gds_path}")

            # 2. 层次化加载
            mgr = LayoutManager()
            lib_obj, queue, graph, plan = mgr.load_and_queue_hierarchical(
                gds_path, layer=0, pixel_size=10.0,
            )
            print(f"  ✓ 层次加载: {len(lib_obj)} cells, 队列 {len(queue)} 任务")
            print(f"    层次节点: {sorted(graph.nodes.keys())}")
            print(f"    执行顺序: {plan.execution_order}")
            print(f"    名称映射: {plan.raw_to_unique_name}")

            # 验证数据一致性
            assert len(lib_obj) == 3
            assert len(queue) == 3
            assert len(plan.raw_to_unique_name) == 3
            for raw_name in ['LEAF', 'MID', 'TOP']:
                unique_name = plan.get_unique_name(raw_name)
                assert unique_name is not None
                assert unique_name in lib_obj
                assert queue.get_entry(unique_name) is not None
            print(f"  ✓ 数据一致性验证通过")

            # 3. 配置并运行层次化批处理
            optimizer_config = {
                'algorithm': 'gradient_descent',
                'max_iterations': 10,
                'learning_rate': 0.01,
                'convergence_threshold': 1e-6,
            }
            batch_cfg = BatchConfig(
                use_hierarchy=True,
                hierarchy_options={'pixel_size': 10.0},
                optimizer_config=optimizer_config,
                save_optimized_masks=True,
                interval_sec=0,
                max_retries=0,
            )
            res_cfg = ResourceConfig(max_workers=2, auto_detect=False,
                                     per_task_timeout_sec=30)

            callback_events = []
            def progress_cb(batch_id, cell_name, status, progress, result):
                callback_events.append({
                    'cell': cell_name, 'status': status.value,
                    'progress': progress,
                })
                print(f"    ← 回调: {cell_name[:20]:20s} "
                      f"{status.value:10s} progress={progress:.0%}")

            batch_cfg.progress_callback = progress_cb

            print(f"\n  启动层次化批处理...")
            start_time = time.time()
            runner = HierarchicalBatchRunner(res_cfg, batch_cfg)

            depths = {n.cell_name: n.depth for n in graph.nodes.values()}
            print(f"    层次深度: {depths}")
            print(f"    叶节点仿真: LEAF (深度 2)")
            print(f"    复合节点合成: MID (深度 1), TOP (深度 0)")

            summary, results = runner.run(
                queue, plan, graph, output_dir=out_dir, pixel_size=10.0,
            )

            total_time = time.time() - start_time
            print(f"  ✓ 批处理完成，总耗时 {total_time:.2f}s")

            # 4. 验证结果
            print(f"\n  结果验证:")
            print(f"    Summary: {summary.total_tasks} 任务, "
                  f"{summary.done} 完成, {summary.failed} 失败")

            assert summary.total_tasks == 3
            assert summary.done == 3
            assert summary.failed == 0
            assert len(results) == 3

            results_by_name = {r.cell_name: r for r in results}

            # LEAF 仿真
            leaf_unique = plan.get_unique_name('LEAF')
            leaf_result = results_by_name.get(leaf_unique)
            assert leaf_result is not None
            assert leaf_result.status == TaskStatus.DONE
            assert 'composed_from_children' not in leaf_result.extra
            assert leaf_result.worker_id != "composer"
            assert leaf_result.elapsed_sec > 0
            print(f"    ✓ LEAF (仿真): worker={leaf_result.worker_id}, "
                  f"耗时 {leaf_result.elapsed_sec:.2f}s")

            # MID 合成
            mid_unique = plan.get_unique_name('MID')
            mid_result = results_by_name.get(mid_unique)
            assert mid_result is not None
            assert mid_result.status == TaskStatus.DONE
            assert mid_result.extra.get('composed_from_children') is True
            assert mid_result.worker_id == "composer"
            assert mid_result.extra.get('total_child_instances') == 4
            print(f"    ✓ MID (合成): 子实例数="
                  f"{mid_result.extra['total_child_instances']}")

            # TOP 合成（MID 1个实例 + LEAF 1个实例 = 2个直接子实例）
            top_unique = plan.get_unique_name('TOP')
            top_result = results_by_name.get(top_unique)
            assert top_result is not None
            assert top_result.status == TaskStatus.DONE
            assert top_result.extra.get('composed_from_children') is True
            assert top_result.worker_id == "composer"
            assert top_result.extra.get('total_child_instances') == 2
            assert top_result.extra.get('child_count') == 2
            print(f"    ✓ TOP (合成): 直接子实例数="
                  f"{top_result.extra['total_child_instances']}, "
                  f"不重复子cell={top_result.extra['child_count']}")

            # 回调验证
            statuses = [e['status'] for e in callback_events]
            assert 'running' in statuses
            assert 'done' in statuses
            print(f"    ✓ 进度回调正常: {len(callback_events)} 次事件")

            # 运行时间验证（覆盖 >0.5s 等待路径）
            assert total_time > 0.5, \
                f"总运行时间 {total_time:.2f}s 应 > 0.5s"
            print(f"    ✓ 运行时间 {total_time:.2f}s > 0.5s，"
                  f"覆盖 worker 等待路径")

            # 掩模保存验证
            mask_files = glob.glob(os.path.join(out_dir, "masks", "*.npy"))
            print(f"    ✓ 掩模文件: {len(mask_files)} 个")
            assert len(mask_files) >= 1

            # 结果缓存验证
            assert runner._merger is not None
            for raw_name in ['LEAF', 'MID']:
                cached = runner._merger.get_cached_mask(raw_name)
                assert cached is not None
                print(f"    ✓ {raw_name} 结果已缓存: {cached.shape}")

            top_mask = runner._merger.get_cached_mask('TOP')
            assert top_mask is not None
            print(f"    ✓ TOP 合成结果已缓存: {top_mask.shape}")

            print(f"\n  ✅ 端到端测试通过！")
            print(f"    实际仿真: 1 次 (LEAF)")
            print(f"    合成复用: 2 次 (MID, TOP)")
            print(f"    节省冗余仿真: 2 次")
            return True

    except AssertionError as e:
        print(f"\n  ❌ 断言失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n  ❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hierarchical_run_batch_optimization_entry():
    """
    测试通过 run_batch_optimization 顶层函数调用层次化模式
    """
    print("\n" + "=" * 70)
    print("测试: run_batch_optimization 层次化入口")
    print("=" * 70)

    try:
        import gdstk
    except ImportError:
        print("  ⚠️  跳过：gdstk 不可用")
        return True

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            gds_path = os.path.join(tmpdir, "top_level_test.gds")
            out_dir = os.path.join(tmpdir, "results")

            lib = gdstk.Library()
            leaf = lib.new_cell("LEAF")
            leaf.add(gdstk.rectangle((0, 0), (500, 500), layer=0))
            top = lib.new_cell("TOP")
            top.add(gdstk.Reference("LEAF", origin=(0, 0)))
            top.add(gdstk.Reference("LEAF", origin=(1000, 0)))
            top.add(gdstk.Reference("LEAF", origin=(2000, 0)))
            lib.write_gds(gds_path)

            batch_cfg = BatchConfig(
                use_hierarchy=True,
                hierarchy_options={'pixel_size': 10.0},
                optimizer_config={'max_iterations': 5, 'learning_rate': 0.01},
                save_optimized_masks=False,
                interval_sec=0,
                max_retries=0,
            )

            print(f"  调用 run_batch_optimization(use_hierarchy=True)...")

            summary, results, lib_obj, queue = run_batch_optimization(
                gds_path, layer=0, batch_config=batch_cfg,
                output_dir=out_dir,
            )

            print(f"  ✓ 调用成功: {summary.total_tasks} 任务, "
                  f"{summary.done} 完成")
            print(f"    Library cells: {[c.name for c in lib_obj.cells()]}")
            print(f"    Queue size: {len(queue)}")

            assert summary.total_tasks == 2
            assert len(lib_obj) == 2
            assert len(queue) == 2

            leaf_unique = None
            top_unique = None
            for cell in lib_obj.cells():
                if cell.cell_name == 'LEAF':
                    leaf_unique = cell.name
                elif cell.cell_name == 'TOP':
                    top_unique = cell.name

            results_by_name = {r.cell_name: r for r in results}
            leaf_result = results_by_name.get(leaf_unique)
            top_result = results_by_name.get(top_unique)

            assert leaf_result is not None
            assert leaf_result.status == TaskStatus.DONE
            assert top_result is not None
            assert top_result.status == TaskStatus.DONE
            assert top_result.extra.get('composed_from_children') is True

            print(f"    ✓ LEAF 仿真: 耗时 {leaf_result.elapsed_sec:.2f}s")
            print(f"    ✓ TOP 合成: 子实例数="
                  f"{top_result.extra.get('total_child_instances')}")
            print(f"  ✅ 顶层入口测试通过！")
            return True

    except Exception as e:
        print(f"\n  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("层次化批处理端到端集成测试")
    print("=" * 70)

    all_passed = True

    if not test_hierarchical_batch_runner_full_flow():
        all_passed = False

    if not test_hierarchical_run_batch_optimization_entry():
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 所有集成测试通过！")
    else:
        print("❌ 部分测试失败")
    print("=" * 70 + "\n")

    sys.exit(0 if all_passed else 1)
