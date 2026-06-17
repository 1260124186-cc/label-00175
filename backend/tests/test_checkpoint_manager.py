# -*- coding: utf-8 -*-
"""
WorkflowCheckpointManager 断点续跑单元测试

核心验证：
  1. find_latest_checkpoint 严格按**墙钟时间最近**（created_at desc → mtime desc）
     返回 checkpoint，而非 outer_iteration desc。这保证复用同一目录重跑时
     正确返回第二次运行的 checkpoint（时间更近），而非第一次运行中 outer
     更大的旧文件。
  2. find_best_checkpoint 独立按 best_loss 最小返回，与 latest 语义解耦。
  3. 不被 _latest_best 快捷方式、字典序最旧文件劫持。
"""

import os
import time
import json
from pathlib import Path

import pytest
import numpy as np

from algorithms.callbacks import (
    WorkflowCheckpointState,
    WorkflowCheckpointManager,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_state(outer: int, inner: int = 0, phase: str = 'test',
                best_loss: float = 1.0, seed: int = 0) -> WorkflowCheckpointState:
    rng = np.random.RandomState(seed)
    state = WorkflowCheckpointState(
        workflow_type='TEST',
        outer_iteration=outer,
        inner_iteration=inner,
        current_phase=phase,
        mask=rng.rand(8, 8),
        best_loss=best_loss,
        loss_history=[best_loss],
    )
    state.capture_random_state()
    return state


def _forge_created_at(base_path: Path, timestamp: float) -> None:
    """篡改 .json 元数据中的 created_at，模拟不同墙钟时间的 checkpoint"""
    json_path = base_path.with_suffix('.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta['created_at'] = timestamp
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f)


def _touch_mtime(path: Path, offset_seconds: float) -> None:
    for suffix in ('.npz', '.pkl', '.json'):
        f = path.with_suffix(suffix)
        if f.exists():
            ts = time.time() + offset_seconds
            os.utime(f, (ts, ts))


# ============================================================================
# Tests
# ============================================================================

class TestFindLatestCheckpointStrictRecency:
    """验证 find_latest_checkpoint 严格按墙钟时间最近续跑"""

    def test_rerun_same_dir_returns_second_run(self, tmp_path):
        """
        核心回归用例：同目录重跑

        场景：
          第一次运行：outer=1,2,3,4,5（created_at = 1000 秒）
          第二次运行：outer=1,2       （created_at = 2000 秒，时间更近）

        旧 bug：按 outer desc 排序 → 错误返回 outer=5（第一次运行）
        修复后：按 created_at desc → 正确返回 outer=2（第二次运行，时间最近）
        """
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=20,
            save_best_only=False,
            filename_prefix='test',
        )

        # 模拟第一次运行（墙钟时间 T=1000）
        first_run_ts = 1000.0
        for outer in range(1, 6):
            s = _make_state(outer=outer, best_loss=float(outer), seed=outer)
            p = mgr.save_checkpoint(s, outer_iteration=outer,
                                    current_loss=float(outer), force=True)
            _forge_created_at(p, first_run_ts + outer)
            _touch_mtime(p, offset_seconds=first_run_ts + outer - time.time())
            time.sleep(0.005)

        time.sleep(0.01)

        # 模拟第二次运行（墙钟时间 T=2000，更近）
        second_run_ts = 2000.0
        p_run2_last = None
        for outer in range(1, 3):
            s = _make_state(outer=outer, best_loss=float(outer) + 0.5, seed=outer + 100)
            p = mgr.save_checkpoint(s, outer_iteration=outer,
                                    current_loss=float(outer) + 0.5, force=True)
            _forge_created_at(p, second_run_ts + outer)
            _touch_mtime(p, offset_seconds=second_run_ts + outer - time.time())
            if outer == 2:
                p_run2_last = p
            time.sleep(0.005)

        latest = mgr.find_latest_checkpoint(validate_config=False)
        assert latest is not None
        assert 'outer_0002' in latest.name, (
            f"同目录重跑：应返回第二次运行的 outer_0002（时间最近），"
            f"实际返回 {latest.name}"
        )
        # 绝不能返回第一次运行的 outer=5
        assert 'outer_0005' not in latest.name

    def test_latest_not_hijacked_by_best_link(self, tmp_path):
        """
        回归用例：
          - iter=1 best_loss=0.01（生成 _latest_best 快捷方式）
          - iter=5 best_loss=0.50（时间更靠后）
        find_latest_checkpoint 返回 iter=5，find_best_checkpoint 返回 iter=1。
        """
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=10,
            save_best_only=False,
            filename_prefix='test',
        )

        s_best = _make_state(outer=1, best_loss=0.01, seed=1)
        p1 = mgr.save_checkpoint(s_best, outer_iteration=1, current_loss=0.01, force=True)
        time.sleep(0.01)

        s_latest = _make_state(outer=5, best_loss=0.50, seed=5)
        p5 = mgr.save_checkpoint(s_latest, outer_iteration=5, current_loss=0.50, force=True)

        best_link = tmp_path / 'test_latest_best.npz'
        assert best_link.exists() or best_link.is_symlink()

        latest = mgr.find_latest_checkpoint(validate_config=False)
        assert latest is not None
        assert 'outer_0005' in latest.name, (
            f"应返回时间最近的 outer_0005，实际返回 {latest.name}"
        )

        best = mgr.find_best_checkpoint(validate_config=False)
        assert best is not None
        assert 'outer_0001' in best.name or 'latest_best' in best.name, (
            f"find_best 应返回 best (outer_0001)，实际返回 {best.name}"
        )

    def test_same_second_uses_outer_as_tiebreaker(self, tmp_path):
        """
        同一秒内创建的 checkpoint（created_at 相同），outer 更大的更近
        （同次运行内迭代号大的就是后保存的）
        """
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=10,
            save_best_only=False,
            filename_prefix='test',
        )

        # outer=3, inner=1（先保存）
        s_a = _make_state(outer=3, inner=1, best_loss=0.9, seed=11)
        pa = mgr.save_checkpoint(s_a, outer_iteration=3, current_loss=0.9, force=True)
        time.sleep(0.01)

        # outer=3, inner=5（后保存，inner 更大 → 同次运行内更近）
        s_b = _make_state(outer=3, inner=5, best_loss=0.8, seed=22)
        pb = mgr.save_checkpoint(s_b, outer_iteration=3, current_loss=0.8, force=True)

        # 把两个 created_at 强制设为同一秒
        shared_ts = 1500.0
        _forge_created_at(pa, shared_ts)
        _forge_created_at(pb, shared_ts)
        # mtime 也设为相同
        _touch_mtime(pa, 0)
        _touch_mtime(pb, 0)
        # 现在按排序应看 outer/inner 作为 tiebreaker

        latest = mgr.find_latest_checkpoint(validate_config=False)
        assert latest is not None
        # outer 相同、inner 5 > inner 1 → 返回 inner_0005
        assert 'outer_0003' in latest.name and 'inner_0005' in latest.name, (
            f"同秒内应按 outer/inner 辅助排序返回 inner_0005，实际是 {latest.name}"
        )

    def test_latest_ignores_lexicographically_earliest_file(self, tmp_path):
        """
        回归用例：防止用 sorted(glob) 字典序错误返回最旧文件。

        保存顺序（时间从先到后）：
            outer=7 → outer=1 → outer=3 → outer=10
        正确语义：按墙钟时间最近 → 返回最后保存的 outer_0010。
        字典序陷阱：会错误返回 outer_0001。
        """
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=20,
            save_best_only=False,
            filename_prefix='t',
        )

        order = [7, 1, 3, 10]
        saved = []
        for i, outer in enumerate(order):
            s = _make_state(outer=outer, inner=0, best_loss=0.5 + (i % 3) * 0.1,
                            seed=100 + i)
            p = mgr.save_checkpoint(s, outer_iteration=outer,
                                    current_loss=s.best_loss + 0.01, force=True)
            # 用 created_at 和 mtime 共同强化"后保存的更新"
            _forge_created_at(p, 1000.0 + i)
            _touch_mtime(p, offset_seconds=i * 10)
            saved.append(p)
            time.sleep(0.01)

        latest = mgr.find_latest_checkpoint(validate_config=False)
        assert latest is not None
        assert 'outer_0010' in latest.name, (
            f"应返回最后保存的 outer_0010，实际是 {latest.name}"
        )
        assert 'outer_0001' not in latest.name

    def test_latest_respects_config_hash_validation(self, tmp_path):
        """最近 checkpoint 若 config_hash 不匹配，应跳过并返回下一个最近的"""
        from dataclasses import dataclass, asdict

        @dataclass
        class FakeConfig:
            param_a: int = 42
            param_b: str = 'hello'

            def to_dict(self):
                return asdict(self)

        cfg = FakeConfig()
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=10,
            save_best_only=False,
            filename_prefix='test',
            config=cfg,
        )
        assert mgr.config_hash, "mgr.config_hash 应为非空"

        # iter=3（正常，时间较早）
        s3 = _make_state(outer=3, best_loss=0.3, seed=1)
        p3 = mgr.save_checkpoint(s3, outer_iteration=3, current_loss=0.3, force=True)
        time.sleep(0.01)

        # iter=5（时间更近，但篡改其 config_hash）
        s5 = _make_state(outer=5, best_loss=0.5, seed=2)
        p5 = mgr.save_checkpoint(s5, outer_iteration=5, current_loss=0.5, force=True)
        json_path = p5.with_suffix('.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        meta['config_hash'] = 'tampered_hash_12345'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f)

        latest_no_valid = mgr.find_latest_checkpoint(validate_config=False)
        assert latest_no_valid is not None and 'outer_0005' in latest_no_valid.name

        latest_valid = mgr.find_latest_checkpoint(validate_config=True)
        assert latest_valid is not None and 'outer_0003' in latest_valid.name, (
            f"应跳过哈希不一致的 outer_0005，返回合法的 outer_0003，实际是 {latest_valid}"
        )

    def test_list_all_checkpoints_sorted_by_recency(self, tmp_path):
        """list_all_checkpoints 按 created_at desc 排序"""
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=10,
            save_best_only=False,
            filename_prefix='test',
        )
        paths = []
        for outer in (1, 5, 3):
            s = _make_state(outer=outer, best_loss=float(outer), seed=outer)
            p = mgr.save_checkpoint(s, outer_iteration=outer, current_loss=float(outer), force=True)
            paths.append(p)
            time.sleep(0.01)

        listed = mgr.list_all_checkpoints()
        assert len(listed) >= 3
        # 按时间保存顺序 1→5→3，created_at 递增，所以 list 应该是 3,5,1
        created_ats = [m.get('created_at', 0) for m in listed]
        assert created_ats == sorted(created_ats, reverse=True), (
            f"list_all_checkpoints 应按 created_at desc 排序，实际时间序列={created_ats}"
        )

    def test_empty_dir_returns_none(self, tmp_path):
        """空目录返回 None"""
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path / 'not_exist',
            workflow_type='TEST',
            save_freq_outer=1,
        )
        assert mgr.find_latest_checkpoint(validate_config=False) is None
        assert mgr.find_best_checkpoint(validate_config=False) is None
        assert mgr.list_all_checkpoints() == []

    def test_rerun_with_older_outer_wins_over_newer(self, tmp_path):
        """
        回归用例：第二次运行的 outer 编号比第一次小，但墙钟时间更近。

        第一次运行：outer=8,9,10  (created_at ≈ T_old)
        第二次运行：outer=1,2     (created_at ≈ T_new, T_new > T_old)

        find_latest_checkpoint 应返回第二次运行的 outer=2（时间最近），
        而非第一次运行的 outer=10（outer 更大但时间更旧）。
        """
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=20,
            save_best_only=False,
            filename_prefix='test',
        )

        # 第一次运行：outer=8,9,10 (T=500)
        old_ts = 500.0
        for outer in (8, 9, 10):
            s = _make_state(outer=outer, best_loss=1.0 / outer, seed=outer)
            p = mgr.save_checkpoint(s, outer_iteration=outer,
                                    current_loss=s.best_loss, force=True)
            _forge_created_at(p, old_ts + outer)
            _touch_mtime(p, offset_seconds=old_ts + outer - time.time())
            time.sleep(0.005)

        time.sleep(0.01)

        # 第二次运行：outer=1,2 (T=3000)
        new_ts = 3000.0
        for outer in (1, 2):
            s = _make_state(outer=outer, best_loss=1.0 / (outer + 10), seed=outer + 200)
            p = mgr.save_checkpoint(s, outer_iteration=outer,
                                    current_loss=s.best_loss, force=True)
            _forge_created_at(p, new_ts + outer)
            _touch_mtime(p, offset_seconds=new_ts + outer - time.time())
            time.sleep(0.005)

        latest = mgr.find_latest_checkpoint(validate_config=False)
        assert latest is not None
        # 必须返回第二次运行的 outer=2（时间最近）
        assert 'outer_0002' in latest.name, (
            f"重跑 outer 更小但时间更近：应返回 outer_0002，实际是 {latest.name}"
        )
        # 绝不能返回第一次运行的 outer=10（outer 最大但时间更旧）
        assert 'outer_0010' not in latest.name
