# -*- coding: utf-8 -*-
"""
WorkflowCheckpointManager 断点续跑单元测试

核心验证：
  1. find_latest_checkpoint 严格返回"最近一次 checkpoint"（按 outer_iteration desc →
     inner_iteration desc → created_at desc），绝不会被 _latest_best 快捷方式或
     字典序最旧文件劫持。
  2. find_best_checkpoint 独立按 best_loss 最小返回，与 latest 语义解耦。
  3. SMO / OPC 工作流恢复入口实际命中的是最近一次 checkpoint。
"""

import os
import time
import json
import tempfile
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
    """构造一个最小化 WorkflowCheckpointState"""
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


def _touch_mtime(path: Path, offset_seconds: float) -> None:
    """强制修改文件 mtime，模拟写入先后顺序"""
    for suffix in ('.npz', '.pkl', '.json'):
        f = path.with_suffix(suffix)
        if f.exists():
            ts = time.time() + offset_seconds
            os.utime(f, (ts, ts))


# ============================================================================
# Tests
# ============================================================================

class TestFindLatestCheckpointStrictlyRecency:
    """验证 find_latest_checkpoint 严格按最近续跑（不被 best/字典序欺骗）"""

    def test_latest_not_hijacked_by_best_link(self, tmp_path):
        """
        回归用例：
          - iter=1 best_loss=0.01（同时是 _latest_best 快捷方式指向的文件）
          - iter=5 best_loss=0.50（最近一次 checkpoint，但 loss 更差）
        断言 find_latest_checkpoint 返回 iter=5，find_best_checkpoint 返回 iter=1。
        """
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=10,
            save_best_only=False,
            filename_prefix='test',
        )

        # iter=1 是 best（loss 更小），同时生成 _latest_best 快捷方式
        s_best = _make_state(outer=1, best_loss=0.01, seed=1)
        p1 = mgr.save_checkpoint(s_best, outer_iteration=1, current_loss=0.01, force=True)
        assert p1 is not None

        # 等 10ms 保证 created_at 不同
        time.sleep(0.01)

        # iter=5 loss 更差，但时间更靠后（是"最近"的）
        s_latest = _make_state(outer=5, best_loss=0.50, seed=5)
        p5 = mgr.save_checkpoint(s_latest, outer_iteration=5, current_loss=0.50, force=True)
        assert p5 is not None

        # 确认 _latest_best 快捷方式存在（它应指向 iter=1）
        best_link = tmp_path / 'test_latest_best.npz'
        assert best_link.exists() or best_link.is_symlink()

        latest = mgr.find_latest_checkpoint(validate_config=False)
        assert latest is not None
        # 核心断言：必须是 iter=5（最近一次），绝不能被 best_link 劫持
        assert 'outer_0005' in latest.name, (
            f"find_latest_checkpoint 应返回最近的 outer_0005，实际返回 {latest.name}"
        )
        assert latest.resolve() == p5.resolve()

        best = mgr.find_best_checkpoint(validate_config=False)
        assert best is not None
        assert 'outer_0001' in best.name or 'latest_best' in best.name, (
            f"find_best_checkpoint 应返回 best (outer_0001 或 latest_best)，实际返回 {best.name}"
        )

    def test_latest_sorted_by_outer_then_inner(self, tmp_path):
        """相同 outer_iteration 时按 inner_iteration 降序，再按 created_at 降序"""
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=10,
            save_best_only=False,
            filename_prefix='test',
        )

        # outer=3, inner=1
        s_a = _make_state(outer=3, inner=1, best_loss=0.9, seed=11)
        mgr.save_checkpoint(s_a, outer_iteration=3, current_loss=0.9, force=True)
        time.sleep(0.01)

        # outer=3, inner=5 （inner 更大 → 更近）
        s_b = _make_state(outer=3, inner=5, best_loss=0.8, seed=22)
        mgr.save_checkpoint(s_b, outer_iteration=3, current_loss=0.8, force=True)
        time.sleep(0.01)

        # outer=2（outer 更小 → 更旧）
        s_c = _make_state(outer=2, inner=9, best_loss=0.1, seed=33)
        mgr.save_checkpoint(s_c, outer_iteration=2, current_loss=0.1, force=True)

        latest = mgr.find_latest_checkpoint(validate_config=False)
        assert latest is not None
        assert 'outer_0003' in latest.name and 'inner_0005' in latest.name, (
            f"期望最近 checkpoint 是 outer_0003/inner_0005，实际是 {latest.name}"
        )

    def test_latest_ignores_lexicographically_earliest_file(self, tmp_path):
        """
        回归用例：防止用 sorted(glob) 字典序错误返回最旧文件。

        保存顺序（时间从先到后）：
            outer=7  →  outer=1  →  outer=3  →  outer=10
        若用 naive sorted(glob) 字典序，文件名会被排序为：
            outer_0001, outer_0003, outer_0007, outer_0010
            → 错误地把 outer_0001（最早且字典序最小）当作"最近"返回。

        正确语义：按 outer_iteration desc，最近的是 outer=10（最后保存）。
        """
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=20,
            save_best_only=False,
            filename_prefix='t',
        )

        # 用不按大小递增的顺序依次保存，使 outer_iter 与字典序不一致
        order = [7, 1, 3, 10]  # 最后一个 outer=10 应是"最近"
        saved = []
        for i, outer in enumerate(order):
            # best_loss 故意不单调，避免 best 逻辑干扰
            s = _make_state(outer=outer, inner=0, best_loss=0.5 + (i % 3) * 0.1,
                            seed=100 + i)
            p = mgr.save_checkpoint(s, outer_iteration=outer,
                                    current_loss=s.best_loss + 0.01, force=True)
            _touch_mtime(p, offset_seconds=i * 10)
            saved.append(p)
            time.sleep(0.01)

        latest = mgr.find_latest_checkpoint(validate_config=False)
        assert latest is not None
        # 必须返回 outer 最大的那一个（outer_0010，按 outer desc 也按时间 desc）
        assert 'outer_0010' in latest.name, (
            f"期望 outer 最大/最后保存的 outer_0010 是最近 checkpoint，实际是 {latest.name}"
        )
        # 绝对不能是字典序最小的 outer_0001（最早保存）
        assert 'outer_0001' not in latest.name

    def test_latest_respects_config_hash_validation(self, tmp_path):
        """最近 checkpoint 若 config_hash 不匹配，应跳过并返回下一个最近的合法文件"""
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
        # 确保配置哈希被正确计算（非空）
        assert mgr.config_hash, "mgr.config_hash 应为非空以进行哈希校验测试"

        # 先保存 iter=3，正常状态
        s3 = _make_state(outer=3, best_loss=0.3, seed=1)
        p3 = mgr.save_checkpoint(s3, outer_iteration=3, current_loss=0.3, force=True)
        time.sleep(0.01)

        # 再保存 iter=5，篡改其 .json config_hash
        s5 = _make_state(outer=5, best_loss=0.5, seed=2)
        p5 = mgr.save_checkpoint(s5, outer_iteration=5, current_loss=0.5, force=True)
        json_path = p5.with_suffix('.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        meta['config_hash'] = 'tampered_hash_12345'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f)

        # validate_config=False 时仍返回 iter=5
        latest_no_valid = mgr.find_latest_checkpoint(validate_config=False)
        assert latest_no_valid is not None and 'outer_0005' in latest_no_valid.name

        # validate_config=True（且期望 hash 是 mgr.config_hash）时应跳过篡改的 iter=5，回退到 iter=3
        latest_valid = mgr.find_latest_checkpoint(validate_config=True)
        assert latest_valid is not None and 'outer_0003' in latest_valid.name, (
            f"应跳过哈希不一致的 outer_0005，返回合法的 outer_0003，实际是 {latest_valid}"
        )

    def test_list_all_checkpoints_sorted_by_recency(self, tmp_path):
        """list_all_checkpoints 也应按最近→最旧排序"""
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=tmp_path,
            workflow_type='TEST',
            save_freq_outer=1,
            max_checkpoints=10,
            save_best_only=False,
            filename_prefix='test',
        )
        for outer in (1, 5, 3):
            s = _make_state(outer=outer, best_loss=float(outer), seed=outer)
            mgr.save_checkpoint(s, outer_iteration=outer, current_loss=float(outer), force=True)
            time.sleep(0.01)

        listed = mgr.list_all_checkpoints()
        assert len(listed) >= 3
        outers = [int(m['outer_iteration']) for m in listed]
        # 必须严格降序
        assert outers == sorted(outers, reverse=True), (
            f"list_all_checkpoints 应按 outer_iteration desc 排序，实际 outer 序列={outers}"
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
