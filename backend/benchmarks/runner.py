# -*- coding: utf-8 -*-
"""
逆向光刻基准测试套件 - 基准运行器

编排测试用例在不同优化算法上的执行流程:
1. 根据测试用例生成目标图案与初始掩模
2. 构建统一的光学系统
3. 调用指定算法执行优化
4. 通过 EvaluationProtocol 收集四维评价
5. 汇总结果供对比与回归检测
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import logging

from benchmarks.test_cases import (
    BenchmarkTestCase,
    TestCaseCategory,
    DifficultyLevel,
    get_all_test_cases,
    get_test_case_by_name,
    get_test_cases_by_category,
)
from benchmarks.protocol import (
    EvaluationProtocol,
    BenchmarkResult,
    ConvergenceMetric,
    EPEMetric,
    PWMetric,
    TimeMetric,
)

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmSpec:
    """
    算法规格说明

    Attributes:
        name: 算法名称
        optimizer_type: 优化器类型 (对应 OptimizerConfig.VALID_TYPES)
        config_overrides: 覆盖优化器默认参数
        workflow: 工作流类型 ('mask_optimization' | 'opc' | 'smo')
        workflow_extra: 工作流额外参数
    """
    name: str
    optimizer_type: str = 'gradient_descent'
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    workflow: str = 'mask_optimization'
    workflow_extra: Dict[str, Any] = field(default_factory=dict)


DEFAULT_ALGORITHMS: List[AlgorithmSpec] = [
    AlgorithmSpec(name='gd', optimizer_type='gradient_descent',
                  config_overrides=dict(learning_rate=0.01)),
    AlgorithmSpec(name='adam', optimizer_type='adam',
                  config_overrides=dict(learning_rate=0.005)),
    AlgorithmSpec(name='rmsprop', optimizer_type='rmsprop',
                  config_overrides=dict(learning_rate=0.005)),
    AlgorithmSpec(name='bfgs', optimizer_type='bfgs'),
]


class BenchmarkRunner:
    """
    基准测试运行器

    对一组测试用例和一组算法规格，执行交叉评价并汇总结果。
    """

    def __init__(
        self,
        protocol: Optional[EvaluationProtocol] = None,
        algorithms: Optional[List[AlgorithmSpec]] = None,
        output_dir: Optional[str] = None,
    ):
        self.protocol = protocol or EvaluationProtocol()
        self.algorithms = algorithms or DEFAULT_ALGORITHMS
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: List[BenchmarkResult] = []

    @property
    def results(self) -> List[BenchmarkResult]:
        return list(self._results)

    def run_single(
        self,
        test_case: BenchmarkTestCase,
        algorithm: AlgorithmSpec,
    ) -> BenchmarkResult:
        """
        执行单个 (测试用例, 算法) 组合的基准测试

        Args:
            test_case: 基准测试用例
            algorithm: 算法规格

        Returns:
            BenchmarkResult
        """
        logger.info(
            f"基准测试开始: {test_case.name} / {algorithm.name}"
        )

        try:
            mask, target = self._generate_pattern(test_case)
            optical_system = self._build_optical_system(test_case)

            initial_epe = self._compute_initial_epe(
                mask, target, optical_system, test_case
            )

            merged_config = dict(test_case.optimizer_defaults)
            merged_config['type'] = algorithm.optimizer_type
            merged_config.update(algorithm.config_overrides)

            wall_start = time.time()
            opt_result = self._run_workflow(
                workflow=algorithm.workflow,
                mask=mask,
                target=target,
                optical_system=optical_system,
                optimizer_config=merged_config,
                workflow_extra=algorithm.workflow_extra,
                test_case=test_case,
            )
            wall_time = time.time() - wall_start

            optimized_mask = opt_result.get('optimized_mask', mask)

            result = self.protocol.evaluate(
                test_case_name=test_case.name,
                algorithm_name=algorithm.name,
                optimized_mask=optimized_mask,
                target=target,
                optical_system=optical_system,
                loss_history=opt_result.get('loss_history', []),
                total_iterations=opt_result.get('total_iterations', 0),
                converged=opt_result.get('converged'),
                convergence_step=opt_result.get('convergence_step'),
                initial_epe=initial_epe,
                epe_history=opt_result.get('epe_history'),
                wall_time=wall_time,
                imaging_time=0.0,
                optimization_time=wall_time,
                custom_metrics=opt_result.get('custom_metrics', {}),
            )

            result.success = opt_result.get('success', True)
            if not result.success:
                result.error_message = opt_result.get('error_message', 'Unknown error')

        except Exception as e:
            result = BenchmarkResult(
                test_case_name=test_case.name,
                algorithm_name=algorithm.name,
                success=False,
                error_message=str(e),
            )
            logger.error(
                f"基准测试失败: {test_case.name} / {algorithm.name}: {e}",
                exc_info=True,
            )

        self._results.append(result)
        logger.info(result.summary())
        return result

    def run_all(
        self,
        test_cases: Optional[List[BenchmarkTestCase]] = None,
        algorithms: Optional[List[AlgorithmSpec]] = None,
    ) -> List[BenchmarkResult]:
        """
        对所有 (测试用例, 算法) 组合执行基准测试

        Args:
            test_cases: 测试用例列表, None 则使用全部
            algorithms: 算法规格列表, None 则使用 runner 的默认列表

        Returns:
            所有 BenchmarkResult 列表
        """
        if test_cases is None:
            test_cases = get_all_test_cases()
        if algorithms is None:
            algorithms = self.algorithms

        total = len(test_cases) * len(algorithms)
        logger.info(f"基准测试套件启动: {len(test_cases)} 用例 x {len(algorithms)} 算法 = {total} 组合")

        results = []
        for i, tc in enumerate(test_cases):
            for j, algo in enumerate(algorithms):
                logger.info(f"[{i * len(algorithms) + j + 1}/{total}] {tc.name} / {algo.name}")
                r = self.run_single(tc, algo)
                results.append(r)

        logger.info(f"基准测试套件完成: {len(results)} 组合已评估")
        return results

    def save_results(self, path: Optional[str] = None) -> str:
        """
        将所有结果序列化为 JSON

        Args:
            path: 保存路径, None 则使用 output_dir/benchmark_results.json

        Returns:
            保存的文件路径
        """
        if path is None:
            if self.output_dir is None:
                raise ValueError("未设置 output_dir 且未指定 path")
            path = str(self.output_dir / 'benchmark_results.json')

        data = [r.to_dict() for r in self._results]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"基准测试结果已保存: {path}")
        return path

    @staticmethod
    def _generate_pattern(test_case: BenchmarkTestCase):
        from core.test_structures import (
            generate_test_structure,
            LineSpaceParams, ContactHoleParams,
            LShapedCornerParams, TJunctionParams,
            SRAMBitcellParams, TestStructureType,
        )

        pp = test_case.pattern_params
        ptype = pp.get('type', 'line_space')

        params_map = {
            'line_space': (LineSpaceParams, {
                'grid_size': tuple(pp.get('grid_size', [64, 64])),
                'pixel_size': pp.get('pixel_size', 1.0),
                'cd': pp.get('cd', 45.0),
                'pitch': pp.get('pitch', 90.0),
                'corner_rounding': pp.get('corner_rounding', 0.0),
                'orientation': pp.get('orientation', 'horizontal'),
            }),
            'contact_hole': (ContactHoleParams, {
                'grid_size': tuple(pp.get('grid_size', [64, 64])),
                'pixel_size': pp.get('pixel_size', 1.0),
                'cd': pp.get('cd', 45.0),
                'pitch': pp.get('pitch', 90.0),
                'corner_rounding': pp.get('corner_rounding', 0.0),
                'hole_shape': pp.get('hole_shape', 'circle'),
            }),
            'l_shaped_corner': (LShapedCornerParams, {
                'grid_size': tuple(pp.get('grid_size', [64, 64])),
                'pixel_size': pp.get('pixel_size', 1.0),
                'cd': pp.get('cd', 45.0),
                'pitch': pp.get('pitch', 90.0),
                'corner_rounding': pp.get('corner_rounding', 0.0),
                'arm_length': pp.get('arm_length', 200.0),
                'corner_type': pp.get('corner_type', 'inner'),
            }),
            't_junction': (TJunctionParams, {
                'grid_size': tuple(pp.get('grid_size', [64, 64])),
                'pixel_size': pp.get('pixel_size', 1.0),
                'cd': pp.get('cd', 45.0),
                'pitch': pp.get('pitch', 90.0),
                'corner_rounding': pp.get('corner_rounding', 0.0),
                'stem_length': pp.get('stem_length', 200.0),
                'branch_length': pp.get('branch_length', 100.0),
            }),
            'sram_bitcell': (SRAMBitcellParams, {
                'grid_size': tuple(pp.get('grid_size', [128, 128])),
                'pixel_size': pp.get('pixel_size', 0.5),
                'cd': pp.get('cd', 25.0),
                'pitch': pp.get('pitch', 50.0),
                'corner_rounding': pp.get('corner_rounding', 0.0),
                'bitcell_type': pp.get('bitcell_type', '6T'),
            }),
        }

        param_cls, param_kwargs = params_map.get(ptype, (LineSpaceParams, {
            'grid_size': tuple(pp.get('grid_size', [64, 64])),
            'pixel_size': pp.get('pixel_size', 1.0),
            'cd': pp.get('cd', 45.0),
            'pitch': pp.get('pitch', 90.0),
        }))

        params = param_cls(**param_kwargs)
        target = generate_test_structure(params)
        mask = target.copy()

        return mask, target

    @staticmethod
    def _build_optical_system(test_case: BenchmarkTestCase):
        from core.imaging import OpticalSystem

        op = test_case.optical_params
        return OpticalSystem(
            wavelength=op.get('wavelength', 193.0),
            na=op.get('na', 1.35),
            sigma=op.get('sigma', 0.75),
            pixel_size=op.get('pixel_size', 1.0),
            defocus=op.get('defocus', 0.0),
            magnification=op.get('magnification', 4.0),
            illumination_type=op.get('illumination_type', 'conventional'),
            source_params=op.get('source_params', {'sigma_inner': 0.0, 'sigma_outer': 0.75}),
            tcc_mode=op.get('tcc_mode', 'socs'),
            socs_num_terms=op.get('socs_num_terms', 5),
            zernike_coefficients=op.get('zernike_coefficients', {}),
        )

    @staticmethod
    def _compute_initial_epe(
        mask: np.ndarray,
        target: np.ndarray,
        optical_system,
        test_case: BenchmarkTestCase,
    ) -> Dict[str, float]:
        from core.imaging import PartialCoherentImaging
        from core.litho_metrics import compute_epe

        pixel_size = test_case.optical_params.get('pixel_size', 1.0)
        threshold = 0.3

        imaging = PartialCoherentImaging(optical_system, mask.shape)
        aerial = imaging.compute_aerial_image(mask)
        wafer = (aerial >= threshold).astype(np.float64)
        target_bin = (target >= 0.5).astype(np.float64)

        return compute_epe(wafer, target_bin, pixel_size=pixel_size)

    @staticmethod
    def _run_workflow(
        workflow: str,
        mask: np.ndarray,
        target: np.ndarray,
        optical_system,
        optimizer_config: Dict[str, Any],
        workflow_extra: Dict[str, Any],
        test_case: BenchmarkTestCase,
    ) -> Dict[str, Any]:
        if workflow == 'mask_optimization':
            return BenchmarkRunner._run_mask_optimization(
                mask, target, optical_system, optimizer_config
            )
        elif workflow == 'opc':
            return BenchmarkRunner._run_opc(
                mask, target, optical_system, optimizer_config, workflow_extra, test_case
            )
        elif workflow == 'smo':
            return BenchmarkRunner._run_smo(
                mask, target, optical_system, optimizer_config, workflow_extra, test_case
            )
        else:
            raise ValueError(f"不支持的工作流: {workflow}")

    @staticmethod
    def _run_mask_optimization(
        mask: np.ndarray,
        target: np.ndarray,
        optical_system,
        optimizer_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig, LossWeights

        loss_weights_dict = optimizer_config.get('loss_weights', {'mse': 1.0})
        loss_weights = LossWeights.from_dict(loss_weights_dict)

        config = OptimizationConfig(
            optimizer_type=optimizer_config.get('type', 'gradient_descent'),
            max_iter=optimizer_config.get('max_iter', 100),
            learning_rate=optimizer_config.get('learning_rate', 0.01),
            tol=optimizer_config.get('tol', 1e-6),
            early_stop_patience=optimizer_config.get('early_stop_patience', 10),
            random_seed=optimizer_config.get('random_seed', 42),
            use_composite_loss=bool(loss_weights_dict),
            loss_weights=loss_weights,
            verbose=False,
        )

        optimizer = MaskOptimizer(optical_system=optical_system, config=config)
        opt_result = optimizer.optimize(mask, target)

        convergence_step = None
        if opt_result.loss_history and len(opt_result.loss_history) >= 2:
            tol = config.tol
            for i in range(1, len(opt_result.loss_history)):
                if abs(opt_result.loss_history[i] - opt_result.loss_history[i - 1]) < tol:
                    convergence_step = i
                    break
            if convergence_step is None and opt_result.loss_history[-1] < tol:
                convergence_step = len(opt_result.loss_history) - 1

        optimized = (
            opt_result.optimal_mask
            if hasattr(opt_result, 'optimal_mask')
            else opt_result.optimized_mask
        )

        return {
            'optimized_mask': optimized,
            'loss_history': [float(v) for v in opt_result.loss_history],
            'total_iterations': opt_result.total_iterations,
            'converged': opt_result.converged,
            'convergence_step': convergence_step,
            'success': True,
            'custom_metrics': {},
        }

    @staticmethod
    def _run_opc(
        mask: np.ndarray,
        target: np.ndarray,
        optical_system,
        optimizer_config: Dict[str, Any],
        workflow_extra: Dict[str, Any],
        test_case: BenchmarkTestCase,
    ) -> Dict[str, Any]:
        from workflows.opc import OPCConfig, OPCWorkflow
        from core.litho_metrics import compute_epe

        pixel_size = test_case.optical_params.get('pixel_size', 1.0)
        opc_dict = dict(workflow_extra)
        opc_dict.setdefault('pixel_size', pixel_size)
        opc_dict.setdefault('verbose', False)

        opc_config = OPCConfig.from_dict(opc_dict)
        workflow = OPCWorkflow(config=opc_config, optical_system=optical_system)

        opc_result = workflow.run(mask, target)

        epe_history = []
        for it in opc_result.iterations:
            epe_history.append(it.epe_after.get('epe_mean', 0.0))

        custom_metrics = {}
        if opc_result.final_epe:
            custom_metrics['epe_mean'] = opc_result.final_epe.get('epe_mean', 0.0)
            custom_metrics['epe_max'] = opc_result.final_epe.get('epe_max', 0.0)
            custom_metrics['epe_improvement_ratio'] = opc_result.total_epe_improvement_ratio

        return {
            'optimized_mask': opc_result.corrected_mask,
            'loss_history': epe_history,
            'total_iterations': opc_result.num_iterations,
            'converged': opc_result.converged,
            'convergence_step': None,
            'epe_history': epe_history,
            'success': True,
            'custom_metrics': custom_metrics,
        }

    @staticmethod
    def _run_smo(
        mask: np.ndarray,
        target: np.ndarray,
        optical_system,
        optimizer_config: Dict[str, Any],
        workflow_extra: Dict[str, Any],
        test_case: BenchmarkTestCase,
    ) -> Dict[str, Any]:
        from workflows.smo import SMOConfig, run_smo_workflow

        smo_dict = dict(workflow_extra)
        smo_dict.setdefault('verbose', False)
        smo_config = SMOConfig.from_dict(smo_dict)

        smo_result = run_smo_workflow(
            mask=mask, target=target,
            optical_system=optical_system,
            config=smo_config,
        )

        custom_metrics = {}
        if smo_result.final_epe:
            custom_metrics['epe_mean'] = smo_result.final_epe.get('epe_mean', 0.0)
        custom_metrics['total_loss_improvement_ratio'] = smo_result.total_loss_improvement_ratio

        return {
            'optimized_mask': smo_result.optimal_mask,
            'loss_history': [float(v) for v in smo_result.loss_history],
            'total_iterations': len(smo_result.iterations),
            'converged': smo_result.converged,
            'convergence_step': None,
            'success': True,
            'custom_metrics': custom_metrics,
        }
