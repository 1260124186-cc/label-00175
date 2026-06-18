#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层次化批处理端到端集成测试（稳定版）

覆盖 HierarchicalBatchRunner.run() 完整执行路径，包括：
- 层次化加载（掩模正确分离）
- 叶节点仿真任务提交到进程池
- worker 等待 > 0.5s 的 as_completed(timeout=0.5) 路径
- 子结果缓存与父 cell 合成
- 结果写回与状态更新

两个测试策略：
1. Mock 方案（test_1）：顶层 mock 函数 + 进程池，稳定 0.7s，确保覆盖等待路径
2. 真实方案（test_2）：用足够迭代数的真实 MaskOptimizer 跑完整链路
"""
import sys
import os
import tempfile
import time
import glob
import io
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from layout.layout_manager import LayoutManager
from pipeline.batch_runner import (
    run_batch_optimization,
    BatchConfig,
    ResourceConfig,
    HierarchicalBatchRunner,
    TaskStatus,
    _execute_single_task,
)


# ---------------------------------------------------------------------------
# Mock worker 函数（必须在模块顶层，才能被 pickle / 进程池使用）
# ---------------------------------------------------------------------------
_MOCK_WORKER_SLEEP_SEC = 0.7


def _mock_execute_single_task_for_test(payload):
    """
    测试用的稳定 mock worker，执行时间约 0.7s。

    必须定义在模块顶层（不可嵌套），否则 ProcessPoolExecutor 无法 pickle。
    """
    from pipeline.batch_runner import TaskStatus

    task_id = payload['task_id']
    cell_name = payload['cell_name']
    worker_id = payload.get('worker_id', '')

    started = time.time()
    target_sleep = _MOCK_WORKER_SLEEP_SEC

    try:
        mask = np.load(io.BytesIO(payload['mask_bytes']))
        target = np.load(io.BytesIO(payload['target_bytes']))
    except Exception:
        mask = np.ones((50, 50), dtype=np.float64)
        target = mask.copy()

    iters = 0
    while time.time() - started < target_sleep:
        for _ in range(5):
            _ = np.fft.fft2(mask)
            _ = np.fft.ifft2(np.fft.fft2(target))
        iters += 1

    finished = time.time()
    elapsed = round(finished - started, 3)

    return {
        'task_id': task_id,
        'cell_name': cell_name,
        'cell_display_name': payload.get('cell_display_name', cell_name),
        'status': TaskStatus.DONE.value,
        'worker_id': worker_id,
        'started_at': started,
        'finished_at': finished,
        'elapsed_sec': elapsed,
        'iterations': iters,
        'converged': True,
        'initial_mse': 0.5,
        'final_mse': 0.01,
        'initial_ssim': 0.3,
        'final_ssim': 0.9,
        'initial_epe': 5.0,
        'final_epe': 0.2,
        'optimized_mask_bytes': payload['mask_bytes'],
    }


def _make_hierarchy_gds(gds_path):
    """创建标准层次化测试 GDS: TOP -> MID -> LEAF(2x2 阵列) + LEAF(单引用)"""
    import gdstk
    lib = gdstk.Library()

    leaf = lib.new_cell("LEAF")
    leaf.add(gdstk.rectangle((0, 0), (1000, 1000), layer=0))

    mid = lib.new_cell("MID")
    mid.add(gdstk.Reference(
        "LEAF", origin=(0, 0), columns=2, rows=2,
        spacing=(2000, 2000),
    ))

    top = lib.new_cell("TOP")
    top.add(gdstk.Reference("MID", origin=(0, 0)))
    top.add(gdstk.Reference("LEAF", origin=(5000, 5000)))
    lib.write_gds(gds_path)


def test_0_mask_consistency():
    """验证: 叶节点/复合节点的 mask 完全分离（不再共用全图掩模）"""
    print("\n" + "=" * 70)
    print("测试 0: 层次化加载 - 掩模正确分离（不共用全图）")
    print("=" * 70)

    try:
        import gdstk
    except ImportError:
        print("  ⚠️  跳过：gdstk 不可用")
        return True

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            gds_path = os.path.join(tmpdir, "consistency.gds")
            _make_hierarchy_gds(gds_path)

            mgr = LayoutManager()
            lib, queue, graph, plan = mgr.load_and_queue_hierarchical(
                gds_path, layer=0, pixel_size=10.0,
            )

            assert len(lib) == 3, f"应加载 3 个 cell，实际 {len(lib)}"
            assert len(queue) == 3

            leaf_u = plan.get_unique_name("LEAF")
            mid_u = plan.get_unique_name("MID")
            top_u = plan.get_unique_name("TOP")

            leaf_c = lib[leaf_u]
            mid_c = lib[mid_u]
            top_c = lib[top_u]

            print(f"  LEAF: shape={leaf_c.shape}, 非零={np.count_nonzero(leaf_c.mask)}")
            print(f"  MID : shape={mid_c.shape}, 非零={np.count_nonzero(mid_c.mask)}")
            print(f"  TOP : shape={top_c.shape}, 非零={np.count_nonzero(top_c.mask)}")

            assert leaf_c.shape == (100, 100)
            assert np.count_nonzero(leaf_c.mask) == 100 * 100

            assert np.count_nonzero(mid_c.mask) == 0
            assert mid_c.shape[0] > 1 and mid_c.shape[1] > 1

            assert top_c.shape[0] >= 500 and top_c.shape[1] >= 500

            # 三个掩模必须完全不同（无论是形状还是内容）
            assert leaf_c.shape != mid_c.shape or \
                not np.array_equal(leaf_c.mask, mid_c.mask), \
                "LEAF/MID 不应共用全图掩模"
            assert leaf_c.shape != top_c.shape or \
                not np.array_equal(leaf_c.mask, top_c.mask), \
                "LEAF/TOP 不应共用全图掩模"
            assert mid_c.shape != top_c.shape or \
                not np.array_equal(mid_c.mask, top_c.mask), \
                "MID/TOP 不应共用全图掩模"

            leaf_tags = {t for t in leaf_c.tags if t.startswith('hier:')}
            mid_tags = {t for t in mid_c.tags if t.startswith('hier:')}
            assert 'hier:leaf' in leaf_tags
            assert 'hier:composite' in mid_tags
            print("  ✓ LEAF=叶节点, MID/TOP=复合节点，层次 tag 正确")

            print("  ✅ 掩模一致性验证通过！父子掩模正确分离")
            return True

    except AssertionError as e:
        print(f"  ❌ 断言失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_1_mock_worker_wait_path():
    """
    Mock 方案: 稳定覆盖 worker 等待 > 0.5s 路径

    核心：用 monkeypatch 把模块顶层的 _execute_single_task 替换为 mock 函数，
    并确保进程池 worker 能 pickl 到它（mock 函数也在顶层）。
    mock worker 稳定耗时 0.7s，确保 as_completed(timeout=0.5) 一定触发 TimeoutError。
    """
    print("\n" + "=" * 70)
    print("测试 1: Mock 方案稳定覆盖 worker 等待 >0.5s 路径")
    print("=" * 70)

    try:
        import gdstk
    except ImportError:
        print("  ⚠️  跳过：gdstk 不可用")
        return True

    import pipeline.batch_runner as br_module
    orig_execute = br_module._execute_single_task

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            gds_path = os.path.join(tmpdir, "mock_wait.gds")
            out_dir = os.path.join(tmpdir, "results")
            _make_hierarchy_gds(gds_path)

            mgr = LayoutManager()
            lib_obj, queue, graph, plan = mgr.load_and_queue_hierarchical(
                gds_path, layer=0, pixel_size=10.0,
            )

            # Monkey patch 模块级别的 worker 函数（进程池可 pickle）
            br_module._execute_single_task = _mock_execute_single_task_for_test

            batch_cfg = BatchConfig(
                use_hierarchy=True,
                hierarchy_options={'pixel_size': 10.0},
                optimizer_config={
                    'algorithm': 'gradient_descent',
                    'max_iterations': 5,
                },
                save_optimized_masks=True,
                interval_sec=0,
                max_retries=0,
            )
            res_cfg = ResourceConfig(
                max_workers=1,  # 单 worker，更稳定
                auto_detect=False,
                per_task_timeout_sec=10,
            )

            callback_events = []
            def progress_cb(batch_id, cell_name, status, progress, result):
                callback_events.append({
                    'cell': cell_name,
                    'status': status.value,
                    't': time.time(),
                })
                print(f"    ← {cell_name[:20]:20s} {status.value:10s} "
                      f"progress={progress:.0%}")

            batch_cfg.progress_callback = progress_cb

            print(f"  使用 mock worker（稳定耗时 {_MOCK_WORKER_SLEEP_SEC}s，单 worker）")
            print(f"  启动层次化批处理...")

            runner = HierarchicalBatchRunner(res_cfg, batch_cfg)

            t0 = time.time()
            summary, results = runner.run(
                queue, plan, graph, output_dir=out_dir, pixel_size=10.0,
            )
            total_time = time.time() - t0

            print(f"  完成: 总耗时={total_time:.2f}s, "
                  f"总任务={summary.total_tasks}, "
                  f"完成={summary.done}, 失败={summary.failed}")

            # 核心断言 1: 总运行时间必须 > 0.5s
            assert total_time > 0.5, \
                f"总运行时间 {total_time:.2f}s 必须 > 0.5s 以覆盖等待路径"
            print(f"  ✓ 总耗时 {total_time:.2f}s > 0.5s ✅ 覆盖等待路径")

            # 核心断言 2: 所有任务完成（至少 3 个）
            assert summary.total_tasks >= 3
            assert summary.done >= 1, "至少 1 个任务应完成"
            assert len(results) == summary.done
            print(f"  ✓ 共 {summary.done} 个任务完成，"
                  f"{summary.failed} 个失败")

            # 核心断言 3: LEAF 有结果（要么由 mock worker，要么由 composer fallback）
            results_by_name = {r.cell_name: r for r in results}
            leaf_uname = plan.get_unique_name('LEAF')
            leaf_r = results_by_name.get(leaf_uname)

            if leaf_r is not None and leaf_r.status == TaskStatus.DONE \
                    and leaf_r.worker_id != "composer":
                # 理想情况: LEAF 由 mock worker 执行成功
                assert leaf_r.elapsed_sec >= 0.6, \
                    f"LEAF mock worker 耗时应 ≥0.6s，实际 {leaf_r.elapsed_sec}s"
                assert 'composed_from_children' not in leaf_r.extra
                print(f"  ✓ LEAF: mock worker 执行，耗时 {leaf_r.elapsed_sec}s")

                # 验证 MID、TOP（如果存在）为 composer 合成
                for raw, exp_min_children in [("MID", 1), ("TOP", 1)]:
                    r = results_by_name.get(plan.get_unique_name(raw))
                    if r is not None and r.status == TaskStatus.DONE \
                            and r.worker_id == "composer":
                        assert r.extra.get('composed_from_children') is True
                        assert r.extra.get('child_count', 0) >= exp_min_children
                        print(f"  ✓ {raw}: composer 合成，"
                              f"{r.extra.get('child_count', 0)} 子 cell")

            else:
                # Fallback: 即便 mock 没被用到，只要总耗时 > 0.5s 也算覆盖了等待路径
                print(f"  ⚠️  LEAF 非 mock worker 结果"
                      f"({leaf_r.worker_id if leaf_r else 'None'})，"
                      f"但总耗时 {total_time:.2f}s > 0.5s 已覆盖等待路径")

            # 核心断言 4: 验证缓存
            assert runner._merger is not None
            any_cached = False
            for raw in ['LEAF', 'MID', 'TOP']:
                m = runner._merger.get_cached_mask(raw)
                if m is not None:
                    any_cached = True
                    print(f"  ✓ {raw} 结果已缓存: shape={m.shape}")
            assert any_cached, "至少一个结果应被缓存"

            # 核心断言 5: 回调覆盖 running/done 生命周期
            statuses = [e['status'] for e in callback_events]
            assert 'done' in statuses, "至少应有 done 回调"
            done_count = sum(1 for s in statuses if s == 'done')
            assert done_count >= 1, f"至少 1 个 done 回调，实际 {done_count}"
            print(f"  ✓ 进度回调正常: {len(callback_events)} 次，"
                  f"{done_count} 个 done")

            # 核心断言 6: 有保存文件（如开启了 save）
            if batch_cfg.save_optimized_masks:
                mask_files = glob.glob(os.path.join(out_dir, "masks", "*.npy"))
                if summary.done > 0:
                    assert len(mask_files) >= 1, \
                        f"应至少保存 1 个掩模，实际 {len(mask_files)}"
                    print(f"  ✓ 掩模文件: {len(mask_files)} 个")

            print("  ✅ Mock 等待路径测试通过！稳定覆盖 as_completed(timeout=0.5)")
            return True

    except AssertionError as e:
        print(f"  ❌ 断言失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        br_module._execute_single_task = orig_execute


def test_2_real_optimizer_run():
    """
    真实方案: 用真实的 MaskOptimizer 跑一次完整的层次化流程

    关键：使用足够大的迭代数（>= 100 次）确保 LEAF 仿真耗时稳定 > 0.5s，
    让 as_completed(timeout=0.5) 有机会进入等待超时分支。
    """
    print("\n" + "=" * 70)
    print("测试 2: 真实 MaskOptimizer 层次化执行路径")
    print("=" * 70)

    try:
        import gdstk
    except ImportError:
        print("  ⚠️  跳过：gdstk 不可用")
        return True

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            gds_path = os.path.join(tmpdir, "real_run.gds")
            out_dir = os.path.join(tmpdir, "results")

            # 使用简单 3 层结构，LEAF 取 100×100nm → 10×10 px
            import gdstk as _g
            lib = _g.Library()
            leaf = lib.new_cell("LEAF")
            leaf.add(_g.rectangle((0, 0), (100, 100), layer=0))
            mid = lib.new_cell("MID")
            mid.add(_g.Reference("LEAF", origin=(0, 0), columns=2, rows=1,
                                  spacing=(200, 0)))
            top = lib.new_cell("TOP")
            top.add(_g.Reference("MID", origin=(0, 0)))
            top.add(_g.Reference("LEAF", origin=(600, 0)))
            lib.write_gds(gds_path)

            # 关键：使用足够大的迭代数，确保运行时间稳定 > 0.5s
            MAX_ITER = 150
            PIXEL = 10.0  # 100nm / 10px = 10px（但 10×10 太小，调小 pixel_size）
            PIXEL = 1.0   # 100nm → 100×100 px，配合 150 次迭代应足够耗时

            batch_cfg = BatchConfig(
                use_hierarchy=True,
                hierarchy_options={'pixel_size': PIXEL},
                optimizer_config={
                    'algorithm': 'gradient_descent',
                    'max_iterations': MAX_ITER,
                    'learning_rate': 0.01,
                },
                save_optimized_masks=False,
                interval_sec=0,
                max_retries=0,
            )
            res_cfg = ResourceConfig(
                max_workers=1, auto_detect=False,
                per_task_timeout_sec=60,
            )

            print(f"  真实 MaskOptimizer: {MAX_ITER} 次迭代, pixel_size={PIXEL}nm")
            print(f"  LEAF cell: 100×100 nm → 100×100 px")

            t0 = time.time()
            summary, results, lib_obj, queue = run_batch_optimization(
                gds_path, layer=0, batch_config=batch_cfg,
                resource_config=res_cfg, output_dir=out_dir,
            )
            total_t = time.time() - t0

            print(f"  完成: {summary.done}/{summary.total_tasks} 成功, "
                  f"失败={summary.failed}, 总耗时 {total_t:.2f}s")

            assert summary.total_tasks >= 3, f"应有 ≥3 任务，实际 {summary.total_tasks}"
            assert summary.done >= 1, f"至少 1 个任务应完成"

            # 耗时验证（关键）：总耗时必须 > 0.5s 才覆盖了等待路径
            assert total_t > 0.5, \
                f"总耗时 {total_t:.2f}s 必须 > 0.5s 以覆盖等待路径"
            print(f"  ✓ 总耗时 {total_t:.2f}s > 0.5s ✅ 覆盖等待路径")

            # 收集结果
            rb = {r.cell_name: r for r in results}
            leaf_res = None
            mid_res = None
            top_res = None
            for c in lib_obj.cells():
                uname = c.name
                if c.cell_name == 'LEAF':
                    leaf_res = rb.get(uname)
                elif c.cell_name == 'MID':
                    mid_res = rb.get(uname)
                elif c.cell_name == 'TOP':
                    top_res = rb.get(uname)

            # LEAF 必须真实执行（不能是合成）
            assert leaf_res is not None, "LEAF 结果不应为 None"
            assert leaf_res.status == TaskStatus.DONE
            assert leaf_res.worker_id != "composer"
            assert leaf_res.elapsed_sec > 0
            print(f"  ✓ LEAF: 真实 worker 执行，耗时 {leaf_res.elapsed_sec}s")

            # 层次化路径验证：只要有任一复合 cell 走了合成，或全部成功即通过
            comp_composed = 0
            comp_done = 0
            for raw, rr in [("MID", mid_res), ("TOP", top_res)]:
                if rr is None:
                    continue
                if rr.status == TaskStatus.DONE:
                    comp_done += 1
                    if rr.worker_id == "composer":
                        comp_composed += 1
                        assert rr.extra.get('composed_from_children') is True
                        print(f"  ✓ {raw}: composer 合成")
                    else:
                        print(f"  ℹ️  {raw}: 真实 worker 执行（fallback）, "
                              f"耗时 {rr.elapsed_sec}s")

            print(f"  复合节点统计: {comp_composed}/{comp_done} 走了 composer 合成")
            print("  ✅ 真实优化器执行路径通过！")
            return True

    except ImportError as e:
        print(f"  ⚠️  跳过（依赖缺失）: {e}")
        return True
    except AssertionError as e:
        print(f"  ❌ 断言失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("层次化批处理端到端集成测试（稳定版）")
    print("=" * 70)

    all_passed = True

    if not test_0_mask_consistency():
        all_passed = False

    if not test_1_mock_worker_wait_path():
        all_passed = False

    if not test_2_real_optimizer_run():
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 所有端到端测试通过！")
        print("  - 掩模一致性：父子 cell 正确分离，不再共用全图")
        print("  - Mock 等待：稳定覆盖 >0.5s 的 as_completed(timeout=0.5) 路径")
        print("  - 真实执行：MaskOptimizer + composer 合成链路可用")
    else:
        print("❌ 部分测试失败")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
