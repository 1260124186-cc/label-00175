# -*- coding: utf-8 -*-
"""
SMO (Source-Mask Optimization) 模块单元测试

覆盖：
- PixelatedSource 初始化与约束
- SMOImagingModel 多工艺条件成像
- SourceOptimizer 多条件损失下降
- MaskOptimizerForSMO 多条件损失下降
- JointGradientOptimizer 联合优化损失下降
- SMOWorkflow 最终 per-condition 统计一致性
"""

import pytest
import numpy as np
from core import OpticalSystem
from core.imaging import TCCMode
from algorithms.mask_optimizer import LossWeights
from workflows.smo import (
    SMOConfig, SourceInitializationType, SourceConstraintsConfig,
    PixelatedSource, SMOImagingModel, SourceOptimizer,
    JointGradientOptimizer, MaskOptimizerForSMO, SMOWorkflow,
    SMOWorkflowResult, ProcessConditionEvaluation,
)


# ============================================================================
# Fixtures
# ============================================================================

SIM_SIZE = 64
SIM_PIXEL = 40.0  # FOV = 2560 nm


@pytest.fixture
def optics():
    return OpticalSystem(
        wavelength=193.0, na=1.35, sigma=0.75, pixel_size=SIM_PIXEL,
        socs_num_terms=3, tcc_mode=TCCMode.FULL_TCC
    )


@pytest.fixture
def process_conditions():
    return [
        {'defocus': -30.0, 'dose': 0.95, 'weight': 1.0},
        {'defocus':   0.0, 'dose': 1.00, 'weight': 1.5},
        {'defocus': +30.0, 'dose': 1.05, 'weight': 1.0},
    ]


@pytest.fixture
def smo_cfg(process_conditions):
    return SMOConfig(
        source_grid_size=(SIM_SIZE, SIM_SIZE),
        source_init_type=SourceInitializationType.CONVENTIONAL,
        source_init_params={'sigma_inner': 0.0, 'sigma_outer': 0.75},
        source_constraints=SourceConstraintsConfig(
            non_negative=True, energy_conservation=True, energy_target=1.0,
            support_radius=0.95,
            smoothness_type='gaussian', gaussian_sigma=0.5,
            smoothness_weight=0.01,
        ),
        process_conditions=process_conditions,
        pvb_weight=1.0,
        source_loss_weights={'mse': 1.0, 'pvb': 0.0, 'epe': 0.0},
        mask_loss_weights=LossWeights(mse=1.0, epe=0.0, pvb=0.0, tv_smooth=0.0, ssim=0.0),
        source_max_iter=5,
        mask_max_iter=5,
        joint_max_iter=5,
        max_outer_iterations=1,
        source_learning_rate=0.05,
        mask_learning_rate=0.05,
        joint_learning_rate_source=0.05,
        joint_learning_rate_mask=0.02,
        wafer_threshold=0.3,
        use_wafer_image_loss=True,
        pixel_size=SIM_PIXEL,
        verbose=False,
    )


@pytest.fixture
def mask_and_target(optics, smo_cfg):
    """构造测试用的 mask 和 target（wafer 平移 1px 作为优化目标）"""
    imaging = SMOImagingModel(optics, (SIM_SIZE, SIM_SIZE),
                              tcc_mode=TCCMode.FULL_TCC, socs_num_terms=3)
    imaging.set_process_conditions(smo_cfg.process_conditions)
    src = PixelatedSource(
        (SIM_SIZE, SIM_SIZE), optics,
        SourceInitializationType.CONVENTIONAL,
        smo_cfg.source_init_params, smo_cfg.source_constraints
    )
    imaging.update_source_all_conditions(src)

    # contact hole 阵列 mask
    mask = np.ones((SIM_SIZE, SIM_SIZE), dtype=np.float64)
    pitch_pix = int(200.0 / SIM_PIXEL)
    cd_pix = int(80.0 / SIM_PIXEL)
    yy_arr, xx_arr = np.mgrid[0:SIM_SIZE, 0:SIM_SIZE]
    for yc in range(pitch_pix // 2, SIM_SIZE, pitch_pix):
        for xc in range(pitch_pix // 2, SIM_SIZE, pitch_pix):
            r2 = (yy_arr - yc) ** 2 + (xx_arr - xc) ** 2
            mask[r2 <= (cd_pix / 2) ** 2] = 0.0

    # target = 标称条件 wafer 平移 1px
    import scipy.ndimage as ndi
    im_nominal = imaging._process_imagers[1][0]
    aerial = np.clip(im_nominal.compute_aerial_image(mask), 0.0, None)
    aerial_shifted = ndi.shift(aerial, (1, 0))
    k = 50.0
    threshold = smo_cfg.wafer_threshold
    target = 1.0 / (1.0 + np.exp(-k * (aerial_shifted - threshold)))
    return mask, target


@pytest.fixture
def src0(optics, smo_cfg):
    return PixelatedSource(
        (SIM_SIZE, SIM_SIZE), optics,
        SourceInitializationType.CONVENTIONAL,
        smo_cfg.source_init_params, smo_cfg.source_constraints
    )


@pytest.fixture
def imaging(optics, smo_cfg, src0):
    img = SMOImagingModel(optics, (SIM_SIZE, SIM_SIZE),
                          tcc_mode=TCCMode.FULL_TCC, socs_num_terms=3)
    img.set_process_conditions(smo_cfg.process_conditions)
    img.update_source_all_conditions(src0)
    return img


# ============================================================================
# Test PixelatedSource
# ============================================================================

class TestPixelatedSource:

    def test_conventional_init(self, optics, smo_cfg):
        src = PixelatedSource(
            (SIM_SIZE, SIM_SIZE), optics,
            SourceInitializationType.CONVENTIONAL,
            smo_cfg.source_init_params, smo_cfg.source_constraints
        )
        assert src.source_map.shape == (SIM_SIZE, SIM_SIZE)
        assert np.all(src.source_map >= 0)
        sigma_eff = src.compute_effective_sigma()
        assert 0.7 < sigma_eff < 0.8

    def test_energy_conservation(self, optics, smo_cfg):
        src = PixelatedSource(
            (SIM_SIZE, SIM_SIZE), optics,
            SourceInitializationType.CONVENTIONAL,
            smo_cfg.source_init_params, smo_cfg.source_constraints
        )
        total_energy = np.sum(src.source_map)
        assert abs(total_energy - 1.0) < 0.01

    def test_custom_source(self, optics, smo_cfg):
        custom = np.random.rand(SIM_SIZE, SIM_SIZE)
        custom = np.clip(custom, 0, None)
        custom /= np.sum(custom)
        src = PixelatedSource(
            (SIM_SIZE, SIM_SIZE), optics,
            SourceInitializationType.CUSTOM,
            {}, smo_cfg.source_constraints,
            custom_source=custom
        )
        np.testing.assert_array_almost_equal(src.source_map, custom)


# ============================================================================
# Test SMOImagingModel
# ============================================================================

class TestSMOImagingModel:

    def test_multi_process_setup(self, imaging, process_conditions):
        assert len(imaging._process_imagers) == len(process_conditions)
        weights = [pc['weight'] for pc in process_conditions]
        doses = [pc['dose'] for pc in process_conditions]
        for i, (im, weight, dose) in enumerate(imaging._process_imagers):
            assert weight == weights[i]
            assert dose == doses[i]

    def test_update_source_all_conditions(self, imaging, src0):
        im0, _, _ = imaging._process_imagers[0]
        mask = np.ones((SIM_SIZE, SIM_SIZE), dtype=np.float64)
        aerial_before = im0.compute_aerial_image(mask)

        new_src = src0.copy()
        new_src.source_map *= 0.5
        imaging.update_source_all_conditions(new_src)

        aerial_after = im0.compute_aerial_image(mask)
        assert not np.allclose(aerial_before, aerial_after)

    def test_multi_condition_aerial(self, imaging):
        mask = np.ones((SIM_SIZE, SIM_SIZE), dtype=np.float64)
        aerials = [im.compute_aerial_image(mask) for im, _, _ in imaging._process_imagers]
        assert len(aerials) == 3
        assert np.mean(aerials[1]) > np.mean(aerials[0])
        assert np.mean(aerials[1]) > np.mean(aerials[2])


# ============================================================================
# Test SourceOptimizer
# ============================================================================

class TestSourceOptimizer:

    def test_loss_decreases(self, imaging, smo_cfg, mask_and_target, src0):
        mask, target = mask_and_target
        opt = SourceOptimizer(imaging, smo_cfg)

        initial_loss, initial_info, _ = opt._compute_loss_and_gradients(mask, target, src0)
        assert initial_loss > 0
        assert 'weighted_mse' in initial_info
        assert 'pvb' in initial_info
        assert initial_info['weighted_mse'] > 0
        assert initial_info['pvb'] > 0

        final_src, history = opt.optimize(
            src0, mask, target,
            max_iter=smo_cfg.source_max_iter,
            learning_rate=smo_cfg.source_learning_rate
        )
        final_loss, final_info, _ = opt._compute_loss_and_gradients(mask, target, final_src)

        assert final_loss < initial_loss
        assert final_info['weighted_mse'] < initial_info['weighted_mse']

    def test_per_condition_mse_consistency(self, imaging, smo_cfg, mask_and_target, src0):
        mask, target = mask_and_target
        opt = SourceOptimizer(imaging, smo_cfg)
        loss, info, _ = opt._compute_loss_and_gradients(mask, target, src0)

        assert 'mse_per_cond' in info
        assert len(info['mse_per_cond']) == len(smo_cfg.process_conditions)

        weights = [pc['weight'] for pc in smo_cfg.process_conditions]
        weighted_sum = sum(
            w * mse for w, mse in zip(weights, info['mse_per_cond'])
        ) / sum(weights)
        assert abs(weighted_sum - info['weighted_mse']) < 1e-8


# ============================================================================
# Test MaskOptimizerForSMO
# ============================================================================

class TestMaskOptimizerForSMO:

    def test_loss_decreases(self, imaging, smo_cfg, mask_and_target, src0):
        mask, target = mask_and_target
        mask0 = mask.copy()
        opt = MaskOptimizerForSMO(imaging, smo_cfg)

        src_opt = SourceOptimizer(imaging, smo_cfg)
        initial_loss, initial_info, _ = src_opt._compute_loss_and_gradients(mask0, target, src0)

        final_mask, history = opt.optimize(
            mask0, target, src0,
            max_iter=smo_cfg.mask_max_iter,
            learning_rate=smo_cfg.mask_learning_rate
        )

        final_loss, final_info, _ = src_opt._compute_loss_and_gradients(final_mask, target, src0)
        assert final_loss < initial_loss


# ============================================================================
# Test JointGradientOptimizer
# ============================================================================

class TestJointGradientOptimizer:

    def test_loss_decreases(self, imaging, smo_cfg, mask_and_target, src0):
        mask, target = mask_and_target
        mask0 = mask.copy()
        opt = JointGradientOptimizer(imaging, smo_cfg)

        initial_loss, initial_info, _, _ = opt._compute_joint_loss_and_grads(mask0, target, src0)
        assert initial_loss > 0
        assert 'joint_weighted_mse' in initial_info
        assert 'joint_pvb' in initial_info

        final_src, final_mask, history = opt.optimize(
            src0, mask0, target,
            max_iter=smo_cfg.joint_max_iter
        )

        final_loss, final_info, _, _ = opt._compute_joint_loss_and_grads(final_mask, target, final_src)
        assert final_loss < initial_loss


# ============================================================================
# Test SMOWorkflow
# ============================================================================

class TestSMOWorkflow:

    def test_result_structure(self, optics, smo_cfg, mask_and_target):
        mask, target = mask_and_target
        workflow = SMOWorkflow(config=smo_cfg, optical_system=optics)
        result = workflow.run(mask, target)

        assert isinstance(result, SMOWorkflowResult)
        assert hasattr(result, 'optimal_source')
        assert hasattr(result, 'optimal_mask')
        assert hasattr(result, 'initial_wafer')
        assert hasattr(result, 'final_wafer')
        assert hasattr(result, 'initial_epe')
        assert hasattr(result, 'final_epe')
        assert hasattr(result, 'initial_mse')
        assert hasattr(result, 'final_mse')

    def test_multi_condition_stats(self, optics, smo_cfg, mask_and_target, process_conditions):
        mask, target = mask_and_target
        workflow = SMOWorkflow(config=smo_cfg, optical_system=optics)
        result = workflow.run(mask, target)

        assert len(result.final_per_condition) == len(process_conditions)
        assert len(result.initial_per_condition) == len(process_conditions)
        assert result.num_process_conditions == len(process_conditions)

        for pc in result.final_per_condition:
            assert isinstance(pc, ProcessConditionEvaluation)
            assert hasattr(pc, 'defocus')
            assert hasattr(pc, 'dose')
            assert hasattr(pc, 'weight')
            assert hasattr(pc, 'mse')
            assert hasattr(pc, 'wafer_continuous')
            assert hasattr(pc, 'wafer_binary')
            assert hasattr(pc, 'epe')

    def test_total_loss_decreases(self, optics, smo_cfg, mask_and_target):
        mask, target = mask_and_target
        workflow = SMOWorkflow(config=smo_cfg, optical_system=optics)
        result = workflow.run(mask, target)

        assert result.final_total_loss < result.initial_total_loss
        assert result.total_loss_improvement > 0
        assert result.total_loss_improvement_ratio > 0

    def test_weighted_mse_decreases(self, optics, smo_cfg, mask_and_target):
        mask, target = mask_and_target
        workflow = SMOWorkflow(config=smo_cfg, optical_system=optics)
        result = workflow.run(mask, target)

        assert result.final_weighted_mse < result.initial_weighted_mse

    def test_pvb_values_nonnegative(self, optics, smo_cfg, mask_and_target):
        mask, target = mask_and_target
        workflow = SMOWorkflow(config=smo_cfg, optical_system=optics)
        result = workflow.run(mask, target)

        assert result.final_pvb_hard >= 0
        assert result.final_pvb_soft >= 0
        assert result.initial_pvb_hard >= 0
        assert result.initial_pvb_soft >= 0

    def test_final_stats_match_independent_forward(self, optics, smo_cfg, mask_and_target):
        mask, target = mask_and_target
        workflow = SMOWorkflow(config=smo_cfg, optical_system=optics)
        result = workflow.run(mask, target)

        src_final = PixelatedSource(
            (SIM_SIZE, SIM_SIZE), optics,
            SourceInitializationType.CUSTOM,
            {}, smo_cfg.source_constraints,
            custom_source=result.optimal_source
        )
        verify_opt = SourceOptimizer(workflow._imaging, smo_cfg)
        verify_loss, verify_info, _ = verify_opt._compute_loss_and_gradients(
            result.optimal_mask, target, src_final
        )

        rel_err = abs(result.final_total_loss - verify_loss) / max(verify_loss, 1e-12)
        assert rel_err < 0.05

        for i, pc in enumerate(result.final_per_condition):
            expected_mse = verify_info['mse_per_cond'][i]
            rel_err_pc = abs(pc.mse - expected_mse) / max(expected_mse, 1e-12)
            assert rel_err_pc < 0.05

    def test_summary_contains_all_fields(self, optics, smo_cfg, mask_and_target):
        mask, target = mask_and_target
        workflow = SMOWorkflow(config=smo_cfg, optical_system=optics)
        result = workflow.run(mask, target)
        summary = result.summary()

        assert isinstance(summary, str)
        assert 'SMO Optimization Result' in summary
        assert 'Initial MSE' in summary
        assert 'Final MSE' in summary
        assert 'Process Conditions' in summary
