# -*- coding: utf-8 -*-
"""
SMO / OPC 工作流"保存后重启继续跑"的集成测试

核心验证：
  1. OPC：prev_result 恢复后不会导致收敛判断错位
  2. SMO：SOURCE_FIRST 预优化阶段的断点不重复执行；阶段游标正确续跑
  3. 工作流最终收敛结果与不间断运行的连续运行在迭代计数上一致
     （对于随机敏感的数值只做结构性/计数上的断言，避免浮点漂移）
"""

import os
import copy
import time
import json
import shutil
from pathlib import Path

import pytest
import numpy as np

from core import OpticalSystem
from algorithms.callbacks import (
    WorkflowCheckpointState,
    WorkflowCheckpointManager,
)
from workflows.opc import (
    OPCConfig,
    OPCWorkflow,
    OPCIterationController,
)
from workflows.smo import (
    SMOConfig,
    SMOWorkflow,
    SMOptimizationStrategy,
    SourceInitializationType,
)


# ============================================================================
# Helpers / Fixtures
# ============================================================================

@pytest.fixture
def simple_mask_and_target():
    """构造一个简单的掩模/目标图案（小尺寸，保证测试速度）"""
    size = 64
    target = np.zeros((size, size), dtype=np.float64)
    # 中心十字形
    target[28:36, 20:44] = 1.0
    target[20:44, 28:36] = 1.0
    # 四周加一些小方块
    target[8:16, 8:16] = 1.0
    target[48:56, 48:56] = 1.0
    initial = target.copy()
    return initial, target


@pytest.fixture
def optics():
    return OpticalSystem(
        wavelength=193.0,
        na=1.35,
        pixel_size=1.0,
    )


# ============================================================================
# OPC Workflow checkpoint resume tests
# ============================================================================

class TestOPCResumeFromCheckpoint:
    """验证 OPC 恢复时 prev_result、convergence 判断等状态的连续性"""

    @staticmethod
    def _run_interrupted(tmp_path, mask, target, optics, cfg,
                         max_native_iters: int = 4,
                         stop_at_iteration: int = 2,
                         seed: int = 42):
        """
        在 stop_at_iteration 之前跑完（共 stop_at_iteration 次迭代）后中断。
        返回 checkpoint 目录路径。
        """
        np.random.seed(seed)
        import random
        random.seed(seed + 1)

        cfg.max_iterations = max_native_iters
        cfg.checkpoint_enable = True
        cfg.checkpoint_dir = str(tmp_path / 'opc_ckpt')
        cfg.checkpoint_save_freq = 1    # 每次迭代都保存
        cfg.checkpoint_max_keep = 20
        cfg.checkpoint_save_best_only = False
        cfg.verbose = False

        wf = OPCWorkflow(config=cfg, optical_system=optics)
        # 第一次跑：仅允许到 stop_at_iteration，通过 monkey patch controller
        original_run_iter = wf.controller.run_iteration
        counter = {'n': 0}

        def _patched_run_iteration(*args, **kwargs):
            counter['n'] += 1
            if counter['n'] > stop_at_iteration:
                # 抛异常强行中断，模拟进程崩溃
                raise RuntimeError('SIMULATED_CRASH')
            return original_run_iter(*args, **kwargs)

        wf.controller.run_iteration = _patched_run_iteration
        try:
            wf.run(mask, target)
        except RuntimeError as e:
            if str(e) != 'SIMULATED_CRASH':
                raise
        return cfg.checkpoint_dir

    def test_resume_opc_iterations_are_contiguous(self, tmp_path,
                                                  simple_mask_and_target, optics):
        """
        中断后继续的 iterations 列表与连续跑的 iterations 结构一致：
        - 迭代号连续
        - 连续跑和续跑收敛前的迭代次数相同（或相同最大次数）
        """
        initial, target = simple_mask_and_target

        cfg = OPCConfig(
            sraf_enable=False,   # 简化，先不涉及 SRAF
            optimizer_enable=True,
            optimizer_max_iter=3,
            max_iterations=4,
            epe_threshold=100.0,
            verbose=False,
        )

        # ---------- ① 连续跑一次（作为基线） ----------
        np.random.seed(42)
        import random
        random.seed(43)
        cfg_c = copy.deepcopy(cfg)
        cfg_c.checkpoint_enable = False
        wf_continuous = OPCWorkflow(config=cfg_c, optical_system=optics)
        res_c = wf_continuous.run(initial, target)

        # ---------- ② 先跑 2 次迭代模拟崩溃 ----------
        ckpt_dir = self._run_interrupted(
            tmp_path, initial, target, optics, cfg,
            max_native_iters=4, stop_at_iteration=2, seed=42,
        )

        # ---------- ③ 恢复后继续 ----------
        np.random.seed(42)
        random.seed(43)
        cfg_r = copy.deepcopy(cfg)
        cfg_r.max_iterations = 4
        cfg_r.checkpoint_enable = True
        cfg_r.checkpoint_dir = ckpt_dir
        cfg_r.checkpoint_save_freq = 1
        cfg_r.checkpoint_max_keep = 20
        cfg_r.checkpoint_save_best_only = False
        cfg_r.verbose = False

        wf_resume = OPCWorkflow(config=cfg_r, optical_system=optics)
        res_r = wf_resume.run(initial, target)

        # 结构断言：
        # - 连续跑的 iterations 长度应该等于续跑后累计的 iterations 长度
        #   （如果连续跑在某点收敛，续跑也应在同一迭代附近收敛，长度允许差 0~1）
        diff = abs(len(res_c.iterations) - len(res_r.iterations))
        assert diff <= 1, (
            f"续跑迭代数 {len(res_r.iterations)} 与连续跑 {len(res_c.iterations)} "
            f"相差 >1，说明恢复逻辑存在跳跃或重复"
        )

        # 续跑 iterations 中每个外层迭代号应该严格单调递增
        it_numbers = [getattr(r, 'iteration', None) for r in res_r.iterations]
        assert all(isinstance(n, int) for n in it_numbers), "OPCIterationResult 缺少 iteration 字段"
        assert it_numbers == sorted(set(it_numbers)), (
            f"续跑 iteration 序列不单调或有重复: {it_numbers}"
        )

    def test_resume_opc_prev_result_is_correct(self, tmp_path,
                                                simple_mask_and_target, optics):
        """
        直接恢复 checkpoint，验证 extra_data['prev_result'] 指向保存时最后一次迭代结果
        （即：下一轮 check_convergence 需要用的"上一轮"结果，不是上上一轮）
        """
        initial, target = simple_mask_and_target

        cfg = OPCConfig(
            sraf_enable=False,
            optimizer_enable=True,
            optimizer_max_iter=2,
            epe_threshold=1e9,  # 保证不收敛
            verbose=False,
        )

        ckpt_dir = self._run_interrupted(
            tmp_path, initial, target, optics, cfg,
            max_native_iters=4, stop_at_iteration=2, seed=7,
        )

        # 直接读取 checkpoint
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=ckpt_dir, workflow_type='OPC',
            save_freq_outer=1, filename_prefix='opc',
        )
        latest = mgr.find_latest_checkpoint(validate_config=False)
        assert latest is not None, "应存在可恢复的 checkpoint"

        state = WorkflowCheckpointState.load(latest)
        prev_result = state.extra_data.get('prev_result')
        assert prev_result is not None, "checkpoint 中缺少 prev_result"

        # 关键断言：saved outer_iteration == prev_result.iteration
        saved_iter = int(state.outer_iteration)
        assert prev_result.iteration == saved_iter, (
            f"prev_result.iteration={prev_result.iteration} 与 "
            f"saved outer_iteration={saved_iter} 不一致，"
            f"说明 checkpoint 保存时 prev_result 存的是上上个迭代而非上个迭代"
        )


# ============================================================================
# SMO Workflow checkpoint resume tests
# ============================================================================

class TestSMOResumeFromCheckpoint:
    """验证 SMO 的阶段游标恢复"""

    def test_smo_source_first_phase_not_duplicated(self, tmp_path,
                                                   simple_mask_and_target, optics):
        """
        SOURCE_FIRST 预优化完成后会立即保存一个 source_first_done 阶段 checkpoint。
        恢复后重新 run 时，SOURCE_FIRST 预优化不应重复执行：
        通过监控 SourceOptimizer.optimize 调用次数验证——
        连续跑（包含预优化 1 次 + 每轮 ALTERNATING source 优化）调用次数应 >
        续跑（跳过预优化 + 剩余轮次）的调用次数。
        """
        initial, target = simple_mask_and_target
        common_cfg_kwargs = dict(
            strategy=SMOptimizationStrategy.SOURCE_FIRST,
            max_outer_iterations=2,  # 两轮 ALTERNATING，加速测试
            source_max_iter=2,
            mask_max_iter=2,
            convergence_patience=10,
            source_init_type=SourceInitializationType.UNIFORM_DISK,
            verbose=False,
            checkpoint_enable=True,
            checkpoint_dir=str(tmp_path / 'smo_ckpt'),
            checkpoint_save_freq_outer=1,
            checkpoint_max_keep=20,
            checkpoint_save_best_only=False,
        )

        # ----- 第一次 run：跑 SOURCE_FIRST 预优化 + 第一轮 outer，然后强制中断 -----
        np.random.seed(88)
        import random
        random.seed(89)
        cfg1 = SMOConfig(**common_cfg_kwargs)

        from workflows.smo import SourceOptimizer as _SO
        orig_source_opt = _SO.optimize

        # 外层循环次数计数
        outer_run_counter = {'outer_count': 0}

        wf1 = SMOWorkflow(config=cfg1, optical_system=optics)

        # 在每轮外层循环"结尾"保存 checkpoint 之后触发中断
        # 直接 patch 内层的 run_iteration 做法不稳定，改用 patch save_checkpoint
        # 在 outer=1 保存后抛异常
        orig_mgr_save = type(wf1._init_checkpoint_manager(initial, target)).save_checkpoint

        def _patched_save(self_mgr, state, *args, **kwargs):
            ret = orig_mgr_save(self_mgr, state, *args, **kwargs)
            # 如果是 outer=1 的 save，则抛异常（模拟崩溃）
            outer_saved = int(getattr(state, 'outer_iteration', 0))
            if outer_saved >= 1:
                raise RuntimeError('SIMULATED_CRASH_AFTER_OUTER_1')
            return ret

        # 直接调用 wf1.run，内部创建的 ckpt_mgr 的 save_checkpoint 需要被 patch
        # 最简单做法：直接 patch WorkflowCheckpointManager.save_checkpoint
        from algorithms.callbacks import WorkflowCheckpointManager
        orig_class_save = WorkflowCheckpointManager.save_checkpoint

        def _class_patched_save(self_mgr, state, *args, **kwargs):
            ret = orig_class_save(self_mgr, state, *args, **kwargs)
            outer_saved = int(getattr(state, 'outer_iteration', 0))
            if outer_saved >= 1:
                raise RuntimeError('SIMULATED_CRASH_AFTER_OUTER_1')
            return ret

        WorkflowCheckpointManager.save_checkpoint = _class_patched_save
        try:
            wf1.run(initial, target)
        except RuntimeError as e:
            if 'SIMULATED_CRASH_AFTER_OUTER_1' not in str(e):
                raise
        finally:
            WorkflowCheckpointManager.save_checkpoint = orig_class_save

        # 确认 source_first_done 阶段 checkpoint 已生成
        mgr = WorkflowCheckpointManager(
            checkpoint_dir=cfg1.checkpoint_dir,
            workflow_type='SMO',
            save_freq_outer=1,
            filename_prefix='smo',
        )
        all_meta = mgr.list_all_checkpoints()
        phases = [str(m.get('current_phase', '')) for m in all_meta]
        assert any('source_first_done' in p for p in phases), (
            f"SOURCE_FIRST 预优化完成后应生成 source_first_done 阶段 checkpoint, "
            f"实际 phases={phases}"
        )
        # 也确认 outer=1 的 checkpoint 存在（中断前刚保存）
        outers = sorted(set(int(m.get('outer_iteration', 0)) for m in all_meta))
        assert 1 in outers, f"应存在 outer=1 的 checkpoint，实际 outers 有 {outers}"

        # ----- 第二次 run（续跑）：配置完全相同，跳过 SOURCE_FIRST 预优化 -----
        np.random.seed(88)
        random.seed(89)
        cfg2 = SMOConfig(**common_cfg_kwargs)   # 配置完全相同 → 哈希一致
        wf2 = SMOWorkflow(config=cfg2, optical_system=optics)

        resume_counter = {'source_opt_counter': 0}

        def _patched_source_opt_resume(self_obj, *args, **kwargs):
            resume_counter['source_opt_counter'] += 1
            return orig_source_opt(self_obj, *args, **kwargs)

        _SO.optimize = _patched_source_opt_resume
        try:
            wf2.run(initial, target)
        finally:
            _SO.optimize = orig_source_opt

        # ----- baseline：连续跑完整 2 轮，调用次数 -----
        np.random.seed(88)
        random.seed(89)
        cfg_base = SMOConfig(**common_cfg_kwargs)
        cfg_base.checkpoint_enable = False
        wf_base = SMOWorkflow(config=cfg_base, optical_system=optics)
        baseline_counter = {'source_opt_counter': 0}

        def _patched_source_opt_base(self_obj, *args, **kwargs):
            baseline_counter['source_opt_counter'] += 1
            return orig_source_opt(self_obj, *args, **kwargs)

        _SO.optimize = _patched_source_opt_base
        try:
            wf_base.run(initial, target)
        finally:
            _SO.optimize = orig_source_opt

        # 关键断言：续跑的 SourceOptimizer.optimize 次数 < baseline 次数
        # baseline 次数 = SOURCE_FIRST 预优化(1) + 2 轮 ALTERNATING 每轮 source 优化(2) = 3
        # 续跑 次数   = 跳过 SOURCE_FIRST + 剩 1 轮 ALTERNATING = 1
        assert resume_counter['source_opt_counter'] < baseline_counter['source_opt_counter'], (
            f"续跑 SourceOptimizer 调用次数 {resume_counter['source_opt_counter']} "
            f"应少于连续跑 {baseline_counter['source_opt_counter']}，"
            f"SOURCE_FIRST 预优化被重复执行了！"
        )

    def test_smo_resume_iterations_count(self, tmp_path,
                                          simple_mask_and_target, optics):
        """
        JOINT 策略 SMO：连续跑 max_outer=3 的 iterations 数量与
        先跑到 outer=1 后恢复继续跑的 iterations 数量结构一致。
        """
        initial, target = simple_mask_and_target

        base_cfg_kwargs = dict(
            strategy=SMOptimizationStrategy.JOINT_GRADIENT,
            joint_max_iter=2,
            max_outer_iterations=3,
            convergence_patience=10,
            source_init_type=SourceInitializationType.UNIFORM_DISK,
            verbose=False,
        )

        # ----- 连续跑 baseline -----
        np.random.seed(101)
        import random
        random.seed(102)
        cfg_c = SMOConfig(**base_cfg_kwargs)
        cfg_c.checkpoint_enable = False
        wf_c = SMOWorkflow(config=cfg_c, optical_system=optics)
        res_c = wf_c.run(initial, target)

        # ----- 中断到 outer=1，然后恢复 -----
        np.random.seed(101)
        random.seed(102)
        cfg_r1 = copy.deepcopy(SMOConfig(**base_cfg_kwargs))
        cfg_r1.max_outer_iterations = 1
        cfg_r1.checkpoint_enable = True
        cfg_r1.checkpoint_dir = str(tmp_path / 'smo_joint_ckpt')
        cfg_r1.checkpoint_save_freq_outer = 1
        cfg_r1.checkpoint_max_keep = 20
        cfg_r1.checkpoint_save_best_only = False
        wf_r1 = SMOWorkflow(config=cfg_r1, optical_system=optics)
        wf_r1.run(initial, target)

        # 恢复后继续
        np.random.seed(101)
        random.seed(102)
        cfg_r2 = copy.deepcopy(SMOConfig(**base_cfg_kwargs))
        cfg_r2.max_outer_iterations = 3
        cfg_r2.checkpoint_enable = True
        cfg_r2.checkpoint_dir = cfg_r1.checkpoint_dir
        cfg_r2.checkpoint_save_freq_outer = 1
        cfg_r2.checkpoint_max_keep = 20
        cfg_r2.checkpoint_save_best_only = False
        wf_r2 = SMOWorkflow(config=cfg_r2, optical_system=optics)
        res_r = wf_r2.run(initial, target)

        # JOINT 策略每个 outer 产生 1 次 iteration
        diff = abs(len(res_c.iterations) - len(res_r.iterations))
        assert diff <= 1, (
            f"JOINT SMO: 续跑 iterations {len(res_r.iterations)} 与 "
            f"连续跑 {len(res_c.iterations)} 相差 >1，恢复阶段游标可能有问题"
        )

    def test_smo_restored_state_contains_phase_cursor(self, tmp_path,
                                                      simple_mask_and_target, optics):
        """恢复出的 WorkflowCheckpointState 应该有 current_phase='mask'/'source'/'joint'"""
        initial, target = simple_mask_and_target
        cfg = SMOConfig(
            strategy=SMOptimizationStrategy.ALTERNATING,
            max_outer_iterations=2,
            source_max_iter=2,
            mask_max_iter=2,
            convergence_patience=10,
            source_init_type=SourceInitializationType.UNIFORM_DISK,
            verbose=False,
            checkpoint_enable=True,
            checkpoint_dir=str(tmp_path / 'smo_phase_ckpt'),
            checkpoint_save_freq_outer=1,
            checkpoint_max_keep=20,
            checkpoint_save_best_only=False,
        )

        np.random.seed(2020)
        import random
        random.seed(2021)
        wf = SMOWorkflow(config=cfg, optical_system=optics)
        wf.run(initial, target)

        mgr = WorkflowCheckpointManager(
            checkpoint_dir=cfg.checkpoint_dir,
            workflow_type='SMO',
            save_freq_outer=1,
            filename_prefix='smo',
        )
        latest = mgr.find_latest_checkpoint(validate_config=False)
        assert latest is not None
        state = WorkflowCheckpointState.load(latest)
        # ALTERNATING 每轮结尾都是 mask 阶段
        assert state.current_phase == 'mask', (
            f"ALTERNATING 策略每次 outer 结束后 current_phase 应为 'mask'，"
            f"实际是 {state.current_phase!r}"
        )
        # source_first_done 字段存在
        assert isinstance(state.extra_data.get('source_first_done'), bool)
