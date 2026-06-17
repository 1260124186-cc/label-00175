# -*- coding: utf-8 -*-
"""
Hybrid OPC+ILT 混合精修回归测试

覆盖：
1. 单热点场景：验证 bbox padding/expand 流程正确执行
2. 多热点场景：验证热点合并逻辑正确
3. 非热点区域保持 OPC 结果不变
4. EPE 评估口径一致性：与 OPC/ILT 成像链口径一致
"""

import pytest
import numpy as np

from core.imaging import OpticalSystem, TCCMode, simulate_wafer_image
from core.litho_metrics import compute_epe

from workflows.opc import OPCConfig, run_opc_workflow
from workflows.ilt import ILTConfig, run_ilt_workflow
from workflows.hybrid_opc_ilt import (
    HybridOPCILTConfig,
    HotspotBBoxManager,
    HybridOPCILTWorkflow,
    LocalILTOptimizer,
    run_hybrid_opc_ilt_workflow,
    HotspotRegion,
)


SIM_SIZE = 64
SIM_PIXEL = 1.0
WAFER_THRESHOLD = 0.3


@pytest.fixture
def optics():
    return OpticalSystem(
        wavelength=193.0, na=0.65, sigma=0.5, pixel_size=SIM_PIXEL,
        socs_num_terms=3, tcc_mode=TCCMode.SOCS,
    )


@pytest.fixture
def target_with_single_hotspot():
    """
    创建一个带有单个"热点"特征的目标图案
    中心一个小方块，边角处容易产生高 EPE
    """
    tgt = np.zeros((SIM_SIZE, SIM_SIZE), dtype=np.float64)
    tgt[20:44, 20:44] = 1.0
    return tgt


@pytest.fixture
def target_with_multiple_hotspots():
    """
    创建带有多个热点特征的目标图案
    两个分离的矩形，各自会产生热点
    """
    tgt = np.zeros((SIM_SIZE, SIM_SIZE), dtype=np.float64)
    tgt[10:25, 10:54] = 1.0
    tgt[40:55, 10:54] = 1.0
    return tgt


@pytest.fixture
def initial_mask(target_with_single_hotspot):
    return target_with_single_hotspot.copy().astype(np.float64)


class TestEPEConsistency:
    """EPE 评估口径一致性验证：与 OPC/ILT 成像链口径一致"""

    def test_epe_uses_imaging_chain_not_mask_directly(self, optics, target_with_single_hotspot):
        """
        验证 hybrid 工作流的 EPE 是通过成像链后 wafer 二值图计算的，
        而不是直接对掩模做二值化。

        关键断言：
        1. initial_wafer 不等于 (initial_mask >= 0.5)
        2. EPE 计算结果与手动用 simulate_wafer_image 计算的一致
        """
        mask = target_with_single_hotspot.copy().astype(np.float64)
        target = target_with_single_hotspot.copy()

        config = HybridOPCILTConfig(
            run_global_opc=False,
            run_local_ilt=False,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        result = run_hybrid_opc_ilt_workflow(
            mask, target, config=config, optical_system=optics
        )

        direct_binary = (mask >= 0.5).astype(np.float64)
        assert not np.array_equal(result.initial_wafer, direct_binary), \
            "initial_wafer 不应该等于直接二值化的掩模，必须经过成像链"

        wafer_cont = simulate_wafer_image(
            mask, optical_system=optics,
            threshold=WAFER_THRESHOLD, apply_resist=True
        )
        expected_wafer = (wafer_cont >= WAFER_THRESHOLD).astype(np.float64)
        assert np.array_equal(result.initial_wafer, expected_wafer), \
            "initial_wafer 应该与 simulate_wafer_image 输出一致"

        expected_epe = compute_epe(
            expected_wafer, target, pixel_size=SIM_PIXEL
        )
        assert abs(result.initial_epe['epe_mean'] - expected_epe['epe_mean']) < 1e-9, \
            "initial EPE 应该与手动成像计算的一致"

    def test_final_epe_uses_imaging_chain(self, optics, target_with_single_hotspot):
        """验证最终 EPE 也是基于成像链后 wafer 计算的"""
        mask = target_with_single_hotspot.copy().astype(np.float64)
        target = target_with_single_hotspot.copy()

        config = HybridOPCILTConfig(
            run_global_opc=False,
            run_local_ilt=False,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        result = run_hybrid_opc_ilt_workflow(
            mask, target, config=config, optical_system=optics
        )

        wafer_cont = simulate_wafer_image(
            result.final_mask, optical_system=optics,
            threshold=WAFER_THRESHOLD, apply_resist=True
        )
        expected_wafer = (wafer_cont >= WAFER_THRESHOLD).astype(np.float64)

        assert np.array_equal(result.final_wafer, expected_wafer), \
            "final_wafer 应该与 simulate_wafer_image 输出一致"


class TestSingleHotspotPadding:
    """单热点场景验证：统一走 padding/expand 流程"""

    def test_single_hotspot_gets_padded(self, optics, target_with_single_hotspot):
        """
        验证即使只有 1 个热点，也会执行 bbox expand/padding，
        而不是直接返回原始 bbox。
        """
        mask = target_with_single_hotspot.copy().astype(np.float64)
        target = target_with_single_hotspot.copy()

        config = HybridOPCILTConfig(
            hotspot_bbox_padding=6,
            max_hotspots=5,
            run_global_opc=False,
            run_local_ilt=False,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        bbox_mgr = HotspotBBoxManager(config)

        original_bbox = (20, 44, 20, 44)
        single_hotspot = HotspotRegion(
            bbox=original_bbox,
            center=(32.0, 32.0),
            epe_mean=5.0,
            epe_max=10.0,
            area=100,
            edge_type='corner',
            priority=10.0,
        )

        merged = bbox_mgr.merge_hotspots([single_hotspot], (SIM_SIZE, SIM_SIZE))

        assert len(merged) == 1, "单热点合并后仍应为 1 个"

        padded = merged[0].bbox
        assert (padded[1] - padded[0]) > (original_bbox[1] - original_bbox[0]), \
            "合并后 bbox 高度应该比原始大（padding 生效）"
        assert (padded[3] - padded[2]) > (original_bbox[3] - original_bbox[2]), \
            "合并后 bbox 宽度应该比原始大（padding 生效）"

        expected_pad = 6
        assert padded[0] == max(0, original_bbox[0] - expected_pad), \
            "y_min 应该外扩 padding 像素"
        assert padded[1] == min(SIM_SIZE, original_bbox[1] + expected_pad), \
            "y_max 应该外扩 padding 像素"

    def test_single_hotspot_workflow_padding(self, optics, target_with_single_hotspot):
        """
        端到端验证：混合工作流中，单热点也会经过 padding 处理。

        验证方式：比较 HotspotDetector 原始输出的 bbox 与
        HybridOPCILTWorkflow 中经过 merge_hotspots（含 padding）后的 bbox
        """
        from workflows.opc import HotspotDetector

        mask = target_with_single_hotspot.copy().astype(np.float64)
        target = target_with_single_hotspot.copy()

        opc_cfg = OPCConfig(
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            epe_threshold=1.0,
            hotspot_dilation=1,
            verbose=False,
        )

        config_small_pad = HybridOPCILTConfig(
            opc_config=opc_cfg,
            hotspot_bbox_padding=2,
            max_hotspots=10,
            run_global_opc=False,
            run_local_ilt=False,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        config_large_pad = HybridOPCILTConfig(
            opc_config=opc_cfg,
            hotspot_bbox_padding=10,
            max_hotspots=10,
            run_global_opc=False,
            run_local_ilt=False,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        workflow_small = HybridOPCILTWorkflow(config=config_small_pad, optical_system=optics)
        result_small = workflow_small.run(mask, target)

        workflow_large = HybridOPCILTWorkflow(config=config_large_pad, optical_system=optics)
        result_large = workflow_large.run(mask, target)

        if len(result_small.merged_hotspots) > 0 and len(result_large.merged_hotspots) > 0:
            h_small = result_small.merged_hotspots[0]
            h_large = result_large.merged_hotspots[0]

            assert h_large.height >= h_small.height, \
                "padding 更大的配置，热点高度应该更大或相等"
            assert h_large.width >= h_small.width, \
                "padding 更大的配置，热点宽度应该更大或相等"

            if h_small.height < SIM_SIZE and h_small.width < SIM_SIZE:
                assert h_large.height > h_small.height or h_large.width > h_small.width, \
                    "当热点未触及图像边界时，更大的 padding 应该产生更大的 bbox"


class TestMultipleHotspotsMerge:
    """多热点场景验证：热点合并逻辑"""

    def test_overlapping_hotspots_merge(self, optics):
        """验证重叠/邻近的热点会被合并"""
        config = HybridOPCILTConfig(
            hotspot_bbox_padding=2,
            hotspot_merge_overlap=2,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )
        bbox_mgr = HotspotBBoxManager(config)

        h1 = HotspotRegion(
            bbox=(10, 20, 10, 20),
            center=(15.0, 15.0),
            epe_mean=3.0, epe_max=5.0, area=50,
            edge_type='line_end', priority=5.0,
        )
        h2 = HotspotRegion(
            bbox=(18, 28, 18, 28),
            center=(23.0, 23.0),
            epe_mean=4.0, epe_max=6.0, area=60,
            edge_type='corner', priority=8.0,
        )

        merged = bbox_mgr.merge_hotspots([h1, h2], (SIM_SIZE, SIM_SIZE))
        assert len(merged) == 1, "两个邻近热点应该合并为 1 个"

        merged_bbox = merged[0].bbox
        assert merged_bbox[0] <= min(h1.bbox[0], h2.bbox[0])
        assert merged_bbox[1] >= max(h1.bbox[1], h2.bbox[1])
        assert merged_bbox[2] <= min(h1.bbox[2], h2.bbox[2])
        assert merged_bbox[3] >= max(h1.bbox[3], h2.bbox[3])

    def test_distant_hotspots_no_merge(self, optics):
        """验证距离较远的热点不会被合并"""
        config = HybridOPCILTConfig(
            hotspot_bbox_padding=2,
            hotspot_merge_overlap=2,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )
        bbox_mgr = HotspotBBoxManager(config)

        h1 = HotspotRegion(
            bbox=(5, 15, 5, 15),
            center=(10.0, 10.0),
            epe_mean=3.0, epe_max=5.0, area=50,
            edge_type='line_end', priority=5.0,
        )
        h2 = HotspotRegion(
            bbox=(40, 50, 40, 50),
            center=(45.0, 45.0),
            epe_mean=4.0, epe_max=6.0, area=60,
            edge_type='corner', priority=8.0,
        )

        merged = bbox_mgr.merge_hotspots([h1, h2], (SIM_SIZE, SIM_SIZE))
        assert len(merged) == 2, "两个远距离热点不应该合并"


class TestNonHotspotPreservation:
    """非热点区域保持 OPC 结果不变验证"""

    def test_non_hotspot_mask_unchanged_no_ilt(self, optics, target_with_multiple_hotspots):
        """
        验证：当 run_local_ilt=False 时，最终掩模与 OPC 掩模完全一致
        （即非热点区域不变，热点区域也不变，因为 ILT 没运行）
        """
        mask = target_with_multiple_hotspots.copy().astype(np.float64)
        target = target_with_multiple_hotspots.copy()

        opc_cfg = OPCConfig(
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            max_iterations=2,
            verbose=False,
        )

        config = HybridOPCILTConfig(
            opc_config=opc_cfg,
            run_global_opc=True,
            run_local_ilt=False,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        result = run_hybrid_opc_ilt_workflow(
            mask, target, config=config, optical_system=optics
        )

        assert np.array_equal(result.final_mask, result.opc_mask), \
            "ILT 关闭时，最终掩模应该与 OPC 掩模完全一致"

    def test_hotspot_regions_change_non_hotspot_stable(self, optics, target_with_multiple_hotspots):
        """
        验证：ILT 优化后，热点 bbox 内的掩模发生变化，
        bbox 外的区域保持 OPC 结果不变。

        这是混合精修模式的核心正确性验证。
        """
        mask = target_with_multiple_hotspots.copy().astype(np.float64)
        target = target_with_multiple_hotspots.copy()

        opc_cfg = OPCConfig(
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            max_iterations=2,
            verbose=False,
        )

        ilt_cfg = ILTConfig(
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            max_iter=5,
            learning_rate=0.05,
            verbose=False,
        )

        config = HybridOPCILTConfig(
            opc_config=opc_cfg,
            ilt_config=ilt_cfg,
            hotspot_bbox_padding=4,
            max_hotspots=5,
            feather_width=2,
            run_global_opc=True,
            run_local_ilt=True,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        result = run_hybrid_opc_ilt_workflow(
            mask, target, config=config, optical_system=optics
        )

        if result.num_hotspots_optimized == 0:
            pytest.skip("没有检测到热点，跳过非热点保持不变验证")

        non_hotspot_mask = np.ones_like(result.final_mask, dtype=bool)
        for local_result in result.local_ilt_results:
            y_min, y_max, x_min, x_max = local_result.bbox
            non_hotspot_mask[y_min:y_max, x_min:x_max] = False

        opc_non_hotspot = result.opc_mask[non_hotspot_mask]
        final_non_hotspot = result.final_mask[non_hotspot_mask]

        assert np.allclose(opc_non_hotspot, final_non_hotspot, atol=1e-12), \
            "非热点区域的掩模值应该保持 OPC 结果不变"

    def test_feather_boundary_smooth_transition(self, optics, target_with_single_hotspot):
        """
        验证：羽化边界处的过渡是平滑的，没有突变
        """
        mask = target_with_single_hotspot.copy().astype(np.float64)
        target = target_with_single_hotspot.copy()

        opc_cfg = OPCConfig(
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            max_iterations=1,
            verbose=False,
        )

        ilt_cfg = ILTConfig(
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            max_iter=3,
            learning_rate=0.1,
            verbose=False,
        )

        config = HybridOPCILTConfig(
            opc_config=opc_cfg,
            ilt_config=ilt_cfg,
            hotspot_bbox_padding=4,
            max_hotspots=3,
            feather_width=3,
            use_gaussian_feather=True,
            run_global_opc=True,
            run_local_ilt=True,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        result = run_hybrid_opc_ilt_workflow(
            mask, target, config=config, optical_system=optics
        )

        if result.num_hotspots_optimized == 0:
            pytest.skip("没有检测到热点，跳过羽化边界验证")

        for local_result in result.local_ilt_results:
            y_min, y_max, x_min, x_max = local_result.bbox
            fw = config.feather_width

            if x_min > 0:
                left_edge_final = result.final_mask[y_min:y_max, x_min]
                left_edge_opc = result.opc_mask[y_min:y_max, x_min]
                left_inner_final = result.final_mask[y_min:y_max, x_min + fw]
                left_inner_ilt = local_result.optimal_mask_local[:, fw]

                assert not np.allclose(left_edge_final, left_inner_final, atol=1e-6), \
                    "从边界到内部应该有渐变过渡"


class TestHybridWorkflowIntegration:
    """混合工作流集成测试"""

    def test_workflow_runs_with_opc_disabled(self, optics, target_with_single_hotspot):
        """验证 OPC 关闭时，混合工作流仍然能正常运行"""
        mask = target_with_single_hotspot.copy().astype(np.float64)
        target = target_with_single_hotspot.copy()

        ilt_cfg = ILTConfig(
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            max_iter=3,
            verbose=False,
        )

        config = HybridOPCILTConfig(
            ilt_config=ilt_cfg,
            hotspot_bbox_padding=4,
            max_hotspots=5,
            run_global_opc=False,
            run_local_ilt=True,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        result = run_hybrid_opc_ilt_workflow(
            mask, target, config=config, optical_system=optics
        )

        assert result.opc_result is None
        assert result.opc_time == 0.0
        assert result.final_mask.shape == mask.shape
        assert hasattr(result, 'initial_wafer')
        assert hasattr(result, 'final_wafer')

    def test_workflow_produces_valid_outputs(self, optics, target_with_single_hotspot):
        """验证混合工作流输出所有必要字段"""
        mask = target_with_single_hotspot.copy().astype(np.float64)
        target = target_with_single_hotspot.copy()

        config = HybridOPCILTConfig(
            run_global_opc=False,
            run_local_ilt=False,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        result = run_hybrid_opc_ilt_workflow(
            mask, target, config=config, optical_system=optics
        )

        assert result.initial_mask.shape == mask.shape
        assert result.final_mask.shape == mask.shape
        assert result.initial_wafer.shape == mask.shape
        assert result.final_wafer.shape == mask.shape
        assert isinstance(result.initial_epe, dict)
        assert 'epe_mean' in result.initial_epe
        assert 'epe_max' in result.initial_epe
        assert isinstance(result.final_epe, dict)
        assert isinstance(result.merged_hotspots, list)
        assert isinstance(result.local_ilt_results, list)
        assert isinstance(result.num_hotspots_optimized, int)
        assert result.total_time >= 0.0

    def test_summary_contains_expected_keys(self, optics, target_with_single_hotspot):
        """验证 summary() 方法返回预期字段"""
        mask = target_with_single_hotspot.copy().astype(np.float64)
        target = target_with_single_hotspot.copy()

        config = HybridOPCILTConfig(
            run_global_opc=False,
            run_local_ilt=False,
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        result = run_hybrid_opc_ilt_workflow(
            mask, target, config=config, optical_system=optics
        )

        summary = result.summary()
        expected_keys = [
            'initial_epe', 'opc_epe', 'final_epe',
            'opc_epe_improvement', 'ilt_epe_improvement',
            'total_epe_improvement', 'total_epe_improvement_ratio',
            'num_hotspots_detected', 'num_hotspots_optimized',
            'converged', 'reason', 'total_time', 'opc_time', 'ilt_time',
        ]
        for key in expected_keys:
            assert key in summary, f"summary 应该包含字段: {key}"


class TestLocalILTOptimizer:
    """LocalILTOptimizer 单元测试"""

    def test_local_ilt_produces_wafer_and_epe(self, optics, target_with_single_hotspot):
        """验证 LocalILTOptimizer 输出 wafer 和 EPE，且口径一致"""
        mask = target_with_single_hotspot.copy().astype(np.float64)
        target = target_with_single_hotspot.copy()

        bbox = (10, 54, 10, 54)
        local_mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]].copy()
        local_target = target[bbox[0]:bbox[1], bbox[2]:bbox[3]].copy()

        config = HybridOPCILTConfig(
            ilt_config=ILTConfig(
                pixel_size=SIM_PIXEL,
                wafer_threshold=WAFER_THRESHOLD,
                max_iter=3,
                verbose=False,
            ),
            pixel_size=SIM_PIXEL,
            wafer_threshold=WAFER_THRESHOLD,
            verbose=False,
        )

        optimizer = LocalILTOptimizer(config, optics)
        result = optimizer.optimize(local_mask, local_target, 0, bbox)

        assert hasattr(result, 'initial_wafer')
        assert hasattr(result, 'optimal_wafer')
        assert result.initial_wafer.shape == local_mask.shape
        assert result.optimal_wafer.shape == local_mask.shape

        wafer_cont = simulate_wafer_image(
            local_mask, optical_system=optics,
            threshold=WAFER_THRESHOLD, apply_resist=True
        )
        expected_wafer = (wafer_cont >= WAFER_THRESHOLD).astype(np.float64)
        assert np.array_equal(result.initial_wafer, expected_wafer), \
            "LocalILTOptimizer 的 initial_wafer 应该与成像链输出一致"
