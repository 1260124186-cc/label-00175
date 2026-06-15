# -*- coding: utf-8 -*-
"""
实验执行器模块

读取实验定义，调用 MaskOptimizer / OPC / SMO 流程，
将结果写入结构化目录，并收集断言所需的指标。

输出目录结构:
    results/
      <experiment_name>/
        <timestamp>/
          config.yaml           # 完整实验配置快照
          metrics.json          # 标量指标
          mask_optimized.npy    # 优化后掩模
          mask_initial.npy      # 初始掩模
          target.npy            # 目标图案
          loss_history.csv      # 损失历史
          golden.json           # golden 参考结果（首次运行时生成）
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import csv
import time
import logging

from experiments.schema import (
    ExperimentSchema, PatternConfig, OpticalConfig,
    OptimizerConfig, GoldenReference,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """
    实验运行结果

    Attributes:
        experiment_name: 实验名称
        success: 是否成功执行
        error_message: 错误信息
        final_mse: 最终 MSE
        final_ssim: 最终 SSIM
        converged: 是否收敛
        convergence_step: 收敛步数（首次满足容差的迭代）
        final_loss: 最终损失
        total_iterations: 总迭代次数
        total_time: 总耗时（秒）
        loss_history: 损失历史
        initial_mse: 初始 MSE
        custom_metrics: 自定义指标 (如 EPE、PVB 等)
        output_dir: 结果输出目录
    """
    experiment_name: str = ''
    success: bool = False
    error_message: str = ''
    final_mse: Optional[float] = None
    final_ssim: Optional[float] = None
    converged: Optional[bool] = None
    convergence_step: Optional[int] = None
    final_loss: Optional[float] = None
    total_iterations: int = 0
    total_time: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    initial_mse: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    output_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'success': self.success,
            'error_message': self.error_message,
            'final_mse': self.final_mse,
            'final_ssim': self.final_ssim,
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'final_loss': self.final_loss,
            'total_iterations': self.total_iterations,
            'total_time': self.total_time,
            'initial_mse': self.initial_mse,
            'custom_metrics': self.custom_metrics,
            'output_dir': self.output_dir,
        }


class ExperimentExecutor:
    """
    实验执行器

    读取 ExperimentSchema，调用对应工作流，
    收集指标，写结果到结构化目录。
    """

    def __init__(self, base_output_dir: str = './experiment_results'):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, experiment: ExperimentSchema,
            output_dir: Optional[str] = None) -> ExperimentResult:
        """
        执行单个实验

        Args:
            experiment: 实验定义
            output_dir: 自定义输出目录，None 则自动生成

        Returns:
            ExperimentResult
        """
        result = ExperimentResult(experiment_name=experiment.name)
        start_time = time.time()

        try:
            mask, target = self._generate_pattern(experiment.pattern)

            if experiment.workflow == 'mask_optimization':
                workflow_result = self._run_mask_optimization(
                    experiment, mask, target
                )
            elif experiment.workflow == 'opc':
                workflow_result = self._run_opc(
                    experiment, mask, target
                )
            elif experiment.workflow == 'smo':
                workflow_result = self._run_smo(
                    experiment, mask, target
                )
            else:
                raise ValueError(f"不支持的工作流: {experiment.workflow}")

            result.success = True
            result.final_mse = workflow_result.get('final_mse')
            result.final_ssim = workflow_result.get('final_ssim')
            result.converged = workflow_result.get('converged')
            result.convergence_step = workflow_result.get('convergence_step')
            result.final_loss = workflow_result.get('final_loss')
            result.total_iterations = workflow_result.get('total_iterations', 0)
            result.loss_history = workflow_result.get('loss_history', [])
            result.initial_mse = workflow_result.get('initial_mse')
            result.custom_metrics = workflow_result.get('custom_metrics', {})

            if output_dir is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = str(
                    self.base_output_dir / experiment.name / timestamp
                )

            self._save_results(
                output_dir, experiment, result,
                workflow_result, mask, target
            )
            result.output_dir = output_dir

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"实验 '{experiment.name}' 执行失败: {e}", exc_info=True)

        result.total_time = time.time() - start_time
        return result

    def _generate_pattern(self, pattern: PatternConfig):
        """
        根据图案配置生成掩模和目标

        Returns:
            (mask, target) - 初始掩模与目标图案
        """
        from core.test_structures import (
            generate_test_structure,
            LineSpaceParams, ContactHoleParams,
            LShapedCornerParams, TJunctionParams,
            SRAMBitcellParams, TestStructureType,
            LineOrientation, HolePattern,
        )

        params_map = {
            'line_space': (LineSpaceParams, {
                'grid_size': pattern.grid_size,
                'pixel_size': pattern.pixel_size,
                'cd': pattern.cd,
                'pitch': pattern.pitch,
                'corner_rounding': pattern.corner_rounding,
                'orientation': pattern.extra.get('orientation', 'horizontal'),
            }),
            'contact_hole': (ContactHoleParams, {
                'grid_size': pattern.grid_size,
                'pixel_size': pattern.pixel_size,
                'cd': pattern.cd,
                'pitch': pattern.pitch,
                'corner_rounding': pattern.corner_rounding,
                'hole_shape': pattern.extra.get('hole_shape', 'circle'),
            }),
            'l_shaped_corner': (LShapedCornerParams, {
                'grid_size': pattern.grid_size,
                'pixel_size': pattern.pixel_size,
                'cd': pattern.cd,
                'pitch': pattern.pitch,
                'corner_rounding': pattern.corner_rounding,
                'arm_length': pattern.extra.get('arm_length', 200.0),
                'corner_type': pattern.extra.get('corner_type', 'inner'),
            }),
            't_junction': (TJunctionParams, {
                'grid_size': pattern.grid_size,
                'pixel_size': pattern.pixel_size,
                'cd': pattern.cd,
                'pitch': pattern.pitch,
                'corner_rounding': pattern.corner_rounding,
                'stem_length': pattern.extra.get('stem_length', 200.0),
                'branch_length': pattern.extra.get('branch_length', 100.0),
            }),
            'sram_bitcell': (SRAMBitcellParams, {
                'grid_size': pattern.grid_size,
                'pixel_size': pattern.pixel_size,
                'cd': pattern.cd,
                'pitch': pattern.pitch,
                'corner_rounding': pattern.corner_rounding,
                'bitcell_type': pattern.extra.get('bitcell_type', '6T'),
            }),
        }

        param_cls, param_kwargs = params_map.get(
            pattern.type, (LineSpaceParams, {
                'grid_size': pattern.grid_size,
                'pixel_size': pattern.pixel_size,
                'cd': pattern.cd,
                'pitch': pattern.pitch,
            })
        )

        params = param_cls(**param_kwargs)
        target = generate_test_structure(params)
        mask = target.copy()

        return mask, target

    def _build_optical_system(self, optical: OpticalConfig):
        """根据光学配置构建 OpticalSystem"""
        from core.imaging import OpticalSystem

        return OpticalSystem(
            wavelength=optical.wavelength,
            na=optical.na,
            sigma=optical.sigma,
            pixel_size=optical.pixel_size,
            defocus=optical.defocus,
            magnification=optical.magnification,
            illumination_type=optical.illumination_type,
            source_params=dict(optical.source_params),
            tcc_mode=optical.tcc_mode,
            socs_num_terms=optical.socs_num_terms,
            zernike_coefficients=dict(optical.zernike_coefficients),
        )

    def _run_mask_optimization(self, experiment: ExperimentSchema,
                               mask: np.ndarray,
                               target: np.ndarray) -> Dict[str, Any]:
        """
        运行掩模优化工作流

        Returns:
            包含各项指标的字典
        """
        from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig, LossWeights
        from core.metrics import mse, ssim, evaluate_all

        optical_system = self._build_optical_system(experiment.optical)
        opt_cfg = experiment.optimizer

        loss_weights = LossWeights.from_dict(opt_cfg.loss_weights)

        config = OptimizationConfig(
            optimizer_type=opt_cfg.type,
            max_iter=opt_cfg.max_iter,
            learning_rate=opt_cfg.learning_rate,
            tol=opt_cfg.tol,
            early_stop_patience=opt_cfg.early_stop_patience,
            random_seed=opt_cfg.random_seed,
            use_composite_loss=bool(opt_cfg.loss_weights),
            loss_weights=loss_weights,
            verbose=False,
        )

        for key, value in opt_cfg.extra.items():
            if hasattr(config, key):
                setattr(config, key, value)

        optimizer = MaskOptimizer(
            optical_system=optical_system,
            config=config,
        )

        initial_metrics = evaluate_all(mask, target)
        opt_result = optimizer.optimize(mask, target)

        final_metrics = evaluate_all(
            opt_result.optimal_mask if hasattr(opt_result, 'optimal_mask') else opt_result.optimized_mask,
            target,
        )

        convergence_step = self._find_convergence_step(
            opt_result.loss_history, opt_cfg.tol
        )

        return {
            'final_mse': float(final_metrics.mse),
            'final_ssim': float(final_metrics.ssim),
            'converged': opt_result.converged,
            'convergence_step': convergence_step,
            'final_loss': float(opt_result.loss_history[-1]) if opt_result.loss_history else None,
            'total_iterations': opt_result.total_iterations,
            'loss_history': [float(v) for v in opt_result.loss_history],
            'initial_mse': float(initial_metrics.mse),
            'optimized_mask': opt_result.optimized_mask,
            'custom_metrics': {},
        }

    def _run_opc(self, experiment: ExperimentSchema,
                 mask: np.ndarray,
                 target: np.ndarray) -> Dict[str, Any]:
        """
        运行 OPC 工作流

        Returns:
            包含各项指标的字典
        """
        from workflows.opc import OPCConfig, OPCWorkflow
        from core.metrics import mse, ssim, evaluate_all
        from core.litho_metrics import compute_epe

        optical_system = self._build_optical_system(experiment.optical)
        opc_dict = experiment.workflow_extra.get('opc', {})
        opc_dict.update(experiment.optimizer.extra.get('opc', {}))

        if 'pixel_size' not in opc_dict:
            opc_dict['pixel_size'] = experiment.optical.pixel_size
        if 'verbose' not in opc_dict:
            opc_dict['verbose'] = False

        opc_config = OPCConfig.from_dict(opc_dict)

        workflow = OPCWorkflow(
            config=opc_config,
            optical_system=optical_system,
        )

        initial_metrics = evaluate_all(mask, target)
        wafer_initial = (workflow._simulate_wafer(mask) >= opc_config.wafer_threshold).astype(np.float64)
        epe_initial = compute_epe(wafer_initial, target, pixel_size=opc_config.pixel_size)

        opc_result = workflow.run(mask, target)

        custom_metrics = {}
        if opc_result.final_epe:
            custom_metrics['epe_mean'] = opc_result.final_epe.get('epe_mean', 0.0)
            custom_metrics['epe_max'] = opc_result.final_epe.get('epe_max', 0.0)
            custom_metrics['epe_improvement_ratio'] = opc_result.total_epe_improvement_ratio

        final_mse = float(evaluate_all(opc_result.corrected_mask, target).mse)
        final_ssim = float(evaluate_all(opc_result.corrected_mask, target).ssim)

        loss_history = []
        for it in opc_result.iterations:
            loss_history.append(it.epe_after.get('epe_mean', 0.0))

        return {
            'final_mse': final_mse,
            'final_ssim': final_ssim,
            'converged': opc_result.converged,
            'convergence_step': None,
            'final_loss': opc_result.final_epe.get('epe_mean', 0.0) if opc_result.final_epe else None,
            'total_iterations': opc_result.num_iterations,
            'loss_history': loss_history,
            'initial_mse': float(initial_metrics.mse),
            'optimized_mask': opc_result.corrected_mask,
            'custom_metrics': custom_metrics,
        }

    def _run_smo(self, experiment: ExperimentSchema,
                 mask: np.ndarray,
                 target: np.ndarray) -> Dict[str, Any]:
        """
        运行 SMO 工作流

        Returns:
            包含各项指标的字典
        """
        from workflows.smo import SMOConfig, run_smo_workflow
        from core.metrics import mse, ssim, evaluate_all

        optical_system = self._build_optical_system(experiment.optical)
        smo_dict = experiment.workflow_extra.get('smo', {})
        smo_dict.update(experiment.optimizer.extra.get('smo', {}))

        if 'verbose' not in smo_dict:
            smo_dict['verbose'] = False

        smo_config = SMOConfig.from_dict(smo_dict)

        initial_metrics = evaluate_all(mask, target)

        smo_result = run_smo_workflow(
            mask=mask,
            target=target,
            optical_system=optical_system,
            config=smo_config,
        )

        custom_metrics = {}
        if smo_result.final_epe:
            custom_metrics['epe_mean'] = smo_result.final_epe.get('epe_mean', 0.0)
        custom_metrics['total_loss_improvement_ratio'] = smo_result.total_loss_improvement_ratio
        custom_metrics['initial_total_loss'] = smo_result.initial_total_loss
        custom_metrics['final_total_loss'] = smo_result.final_total_loss

        final_mse = float(evaluate_all(smo_result.optimal_mask, target).mse)
        final_ssim = float(evaluate_all(smo_result.optimal_mask, target).ssim)

        return {
            'final_mse': final_mse,
            'final_ssim': final_ssim,
            'converged': smo_result.converged,
            'convergence_step': None,
            'final_loss': smo_result.final_total_loss,
            'total_iterations': len(smo_result.iterations),
            'loss_history': [float(v) for v in smo_result.loss_history],
            'initial_mse': float(initial_metrics.mse),
            'optimized_mask': smo_result.optimal_mask,
            'custom_metrics': custom_metrics,
        }

    @staticmethod
    def _find_convergence_step(loss_history: List[float],
                               tol: float) -> Optional[int]:
        """
        在损失历史中找到首次满足收敛容差的步数

        Args:
            loss_history: 损失历史列表
            tol: 收敛容差

        Returns:
            收敛步数（0-indexed），若未收敛返回 None
        """
        if not loss_history or len(loss_history) < 2:
            return None

        for i in range(1, len(loss_history)):
            if abs(loss_history[i] - loss_history[i - 1]) < tol:
                return i

        if loss_history[-1] < tol:
            return len(loss_history) - 1

        return None

    def _save_results(self, output_dir: str,
                      experiment: ExperimentSchema,
                      result: ExperimentResult,
                      workflow_result: Dict[str, Any],
                      mask: np.ndarray,
                      target: np.ndarray):
        """将实验结果保存到结构化目录"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        import yaml as _yaml

        config_path = out / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            _yaml.dump(experiment.to_dict(), f,
                       default_flow_style=False, allow_unicode=True)

        metrics = result.to_dict()
        metrics_path = out / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        np.save(out / 'mask_initial.npy', mask)
        np.save(out / 'target.npy', target)

        optimized_mask = workflow_result.get('optimized_mask')
        if optimized_mask is not None:
            np.save(out / 'mask_optimized.npy', optimized_mask)

        if result.loss_history:
            csv_path = out / 'loss_history.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'loss'])
                for i, loss in enumerate(result.loss_history):
                    writer.writerow([i, loss])

        golden_path = out / 'golden.json'
        if not golden_path.exists():
            golden = GoldenReference(
                experiment_name=experiment.name,
                final_mse=result.final_mse,
                final_ssim=result.final_ssim,
                converged=result.converged,
                convergence_step=result.convergence_step,
                final_loss=result.final_loss,
                total_iterations=result.total_iterations,
                custom_metrics=result.custom_metrics,
            )
            with open(golden_path, 'w', encoding='utf-8') as f:
                json.dump(golden.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Golden 参考已生成: {golden_path}")

        logger.info(f"实验结果已保存到: {out}")
