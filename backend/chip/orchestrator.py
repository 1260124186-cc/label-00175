# -*- coding: utf-8 -*-
"""
芯片级 RET 编排主控制器

协调芯片级 RET 优化的完整流程：
1. 芯片 GDS 加载与区域划分
2. 各区域 RET 策略与光学条件匹配
3. 分块并行优化执行
4. 坐标拼合与边界伪影处理
5. 结果输出与质量评估

使用方法：
    result = run_chip_level_ret(gds_path, config)
"""

import numpy as np
import logging
import time
import json
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

from chip.schemas import (
    RegionType, RETStrategyType, ChipRegion, ChipRETConfig,
    ChipRETResult, BlockOptimizationResult, BoundaryArtifactMetrics,
    ChipRegionMetadata,
)
from chip.region_partitioner import RegionPartitioner, RegionPartitionResult
from chip.ret_strategy_matcher import RETStrategyMatcher
from chip.block_optimizer import BlockOptimizer
from chip.stitcher import BoundaryStitcher

logger = logging.getLogger(__name__)


class ChipRETOrchestrator:
    """
    芯片级 RET 编排主控制器

    协调整个芯片级 RET 优化流程，从 GDS 加载到最终拼合输出。

    使用方法：
        orchestrator = ChipRETOrchestrator(config)
        result = orchestrator.run(gds_path)
    """

    def __init__(
        self,
        config: Optional[ChipRETConfig] = None,
    ):
        """
        初始化编排器

        Args:
            config: 芯片级 RET 配置
        """
        self.config = config or ChipRETConfig()

        self.partitioner = RegionPartitioner(
            pixel_size_nm=self.config.pixel_size_nm,
            min_region_size_um2=self.config.min_region_size_um2,
            merge_distance_um=self.config.merge_distance_um,
            use_hierarchy=self.config.use_hierarchy_for_partition,
        )
        self.strategy_matcher = RETStrategyMatcher(
            global_config=self.config,
            user_preference="balanced",
        )
        self.block_optimizer = BlockOptimizer(
            global_config=self.config,
            enable_parallel=self.config.enable_parallel_optimization,
            max_workers=self.config.max_parallel_regions,
        )
        self.stitcher = BoundaryStitcher(global_config=self.config)

    def run(
        self,
        gds_path: Optional[Union[str, Path]] = None,
        full_mask: Optional[np.ndarray] = None,
        chip_bounds_nm: Optional[Tuple[float, float, float, float]] = None,
        strategy_overrides: Optional[Dict[str, RETStrategyType]] = None,
    ) -> ChipRETResult:
        """
        执行完整的芯片级 RET 优化流程

        Args:
            gds_path: GDS 文件路径（与 full_mask 二选一）
            full_mask: 完整芯片掩模（与 gds_path 二选一）
            chip_bounds_nm: 芯片边界 (x0, y0, x1, y1)，单位 nm
            strategy_overrides: 区域 ID 到强制策略的映射

        Returns:
            ChipRETResult 完整的优化结果
        """
        start_time = time.time()

        result = ChipRETResult(
            chip_name=self.config.chip_name,
        )

        try:
            logger.info("=" * 60)
            logger.info(f"开始芯片级 RET 优化: {self.config.chip_name}")
            logger.info("=" * 60)

            logger.info("步骤 1/5: 芯片区域划分...")
            regions, partition_result = self._partition_chip(
                gds_path, full_mask, chip_bounds_nm
            )
            result.regions = regions

            if not regions:
                raise RuntimeError("未识别到任何芯片区域")

            logger.info(f"识别到 {len(regions)} 个区域")
            for region in regions:
                logger.info(f"  - {region.region_id}: {region.metadata.region_type.value}, "
                           f"{region.metadata.area_um2:.1f} um², "
                           f"k1={region.metadata.k1_factor:.2f}")

            logger.info("\n步骤 2/5: RET 策略匹配...")
            strategy_results = self.strategy_matcher.match_all(
                regions, strategy_overrides=strategy_overrides
            )

            strategy_summary = self.strategy_matcher.get_strategy_summary(strategy_results)
            logger.info(f"策略分配完成: {json.dumps(strategy_summary, indent=2, ensure_ascii=False)}")

            runtime_estimate = self.strategy_matcher.estimate_total_runtime(
                strategy_results, regions
            )
            logger.info(f"预计总运行时间因子: {runtime_estimate:.1f}")

            for r in strategy_results:
                if r.warnings:
                    for w in r.warnings:
                        logger.warning(f"  [{r.region_id}] {w}")
                        result.warnings.append(f"[{r.region_id}] {w}")

            logger.info("\n步骤 3/5: 分块优化...")
            block_results = self.block_optimizer.optimize_all(
                regions,
                save_checkpoint=self.config.block_config.enable_checkpointing,
            )
            result.block_results = block_results

            opt_summary = self.block_optimizer.get_optimization_summary(block_results)
            logger.info(f"优化完成: {json.dumps(opt_summary, indent=2, ensure_ascii=False)}")

            successful_regions = [r for r in regions if r.is_optimized]
            failed_regions = [r for r in regions if not r.is_optimized]

            if failed_regions:
                logger.warning(f"{len(failed_regions)} 个区域优化失败:")
                for r in failed_regions:
                    logger.warning(f"  - {r.region_id}: {r.optimization_result.error_message if r.optimization_result else '未知错误'}")
                    result.warnings.append(f"区域 {r.region_id} 优化失败: {r.optimization_result.error_message if r.optimization_result else '未知错误'}")

            if not successful_regions:
                raise RuntimeError("所有区域优化均失败")

            logger.info("\n步骤 4/5: 边界拼合与伪影处理...")
            full_shape, origin_nm = self._get_full_shape_and_origin(
                regions, full_mask, gds_path
            )

            if full_mask is not None:
                result.original_mask = full_mask.copy()

            stitched_mask, boundary_metrics = self.stitcher.stitch_regions(
                successful_regions, full_shape, origin_nm
            )
            result.stitched_mask = stitched_mask
            result.boundary_metrics = boundary_metrics

            stitch_summary = self.stitcher.get_stitching_summary(boundary_metrics)
            logger.info(f"拼接完成: {json.dumps(stitch_summary, indent=2, ensure_ascii=False)}")

            logger.info("\n步骤 5/5: 质量评估与结果输出...")
            self._compute_global_metrics(result, regions, full_mask)

            self._generate_summaries(result, regions, strategy_results, block_results)

            if self.config.output_dir and (
                self.config.save_regions_separately or
                self.config.save_stitched_mask or
                self.config.save_report
            ):
                output_files = self._save_results(
                    result, regions, self.config.output_dir
                )
                result.output_files = output_files

            result.success = True
            logger.info("\n" + "=" * 60)
            logger.info(f"芯片级 RET 优化完成，总耗时: {time.time() - start_time:.2f}s")
            logger.info(f"成功率: {result.success_rate:.1%}")
            logger.info(f"全局 EPE 改善: {result.global_epe_improvement:.2f} nm")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"芯片级 RET 优化失败: {e}", exc_info=True)
            result.success = False
            result.error_message = str(e)
            result.warnings.append(f"优化失败: {e}")

        result.total_time_sec = time.time() - start_time
        return result

    def _partition_chip(
        self,
        gds_path: Optional[Union[str, Path]],
        full_mask: Optional[np.ndarray],
        chip_bounds_nm: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[List[ChipRegion], Optional[RegionPartitionResult]]:
        """
        芯片区域划分

        Args:
            gds_path: GDS 文件路径
            full_mask: 完整掩模
            chip_bounds_nm: 芯片边界

        Returns:
            (区域列表, 划分结果)
        """
        if gds_path is not None:
            partition_result = self.partitioner.partition_gds(
                gds_path=str(gds_path),
                layer=self.config.layer,
                datatype=self.config.datatype,
                pixel_size_nm=self.config.pixel_size_nm,
            )
        elif full_mask is not None and chip_bounds_nm is not None:
            partition_result = self.partitioner.partition_mask(
                full_mask=full_mask,
                chip_bounds_nm=chip_bounds_nm,
                chip_name=self.config.chip_name,
                pixel_size_nm=self.config.pixel_size_nm,
            )
        else:
            raise ValueError("必须提供 gds_path 或 (full_mask + chip_bounds_nm)")

        regions = partition_result.regions

        for region in regions:
            if region.mask is not None and region.target is None:
                region.target = region.mask.copy()

        return regions, partition_result

    def _get_full_shape_and_origin(
        self,
        regions: List[ChipRegion],
        full_mask: Optional[np.ndarray],
        gds_path: Optional[Union[str, Path]],
    ) -> Tuple[Tuple[int, int], Tuple[float, float]]:
        """
        获取完整掩模的形状和原点

        Args:
            regions: 区域列表
            full_mask: 完整掩模（如果提供）
            gds_path: GDS 文件路径

        Returns:
            (完整形状 (ny, nx), 原点坐标 (x0, y0) nm)
        """
        if full_mask is not None:
            return full_mask.shape, (0.0, 0.0)

        if not regions:
            raise ValueError("没有区域信息")

        all_x0 = [r.metadata.bounds_nm[0] for r in regions]
        all_y0 = [r.metadata.bounds_nm[1] for r in regions]
        all_x1 = [r.metadata.bounds_nm[2] for r in regions]
        all_y1 = [r.metadata.bounds_nm[3] for r in regions]

        x0 = min(all_x0)
        y0 = min(all_y0)
        x1 = max(all_x1)
        y1 = max(all_y1)

        pixel_size = regions[0].metadata.pixel_size_nm

        nx = int(round((x1 - x0) / pixel_size))
        ny = int(round((y1 - y0) / pixel_size))

        return (ny, nx), (x0, y0)

    def _compute_global_metrics(
        self,
        result: ChipRETResult,
        regions: List[ChipRegion],
        full_mask: Optional[np.ndarray],
    ) -> None:
        """
        计算全局指标

        Args:
            result: 结果对象
            regions: 区域列表
            full_mask: 完整掩模
        """
        all_initial_epe_mean = []
        all_final_epe_mean = []
        all_initial_cd_mean = []
        all_final_cd_mean = []
        all_initial_mse = []
        all_final_mse = []

        for br in result.block_results:
            if not br.success:
                continue

            if br.initial_epe:
                all_initial_epe_mean.append(br.initial_epe.get('epe_mean', 0.0))
            if br.final_epe:
                all_final_epe_mean.append(br.final_epe.get('epe_mean', 0.0))
            if br.initial_cd:
                all_initial_cd_mean.append(br.initial_cd.get('cd_mean', 0.0))
            if br.final_cd:
                all_final_cd_mean.append(br.final_cd.get('cd_mean', 0.0))
            if br.initial_mse > 0:
                all_initial_mse.append(br.initial_mse)
            if br.final_mse > 0:
                all_final_mse.append(br.final_mse)

        if all_initial_epe_mean:
            result.global_initial_epe['epe_mean'] = float(np.mean(all_initial_epe_mean))
            result.global_initial_epe['epe_max'] = float(np.max([
                br.initial_epe.get('epe_max', 0.0)
                for br in result.block_results if br.success and br.initial_epe
            ])) if any(br.initial_epe for br in result.block_results if br.success) else 0.0

        if all_final_epe_mean:
            result.global_final_epe['epe_mean'] = float(np.mean(all_final_epe_mean))
            result.global_final_epe['epe_max'] = float(np.max([
                br.final_epe.get('epe_max', 0.0)
                for br in result.block_results if br.success and br.final_epe
            ])) if any(br.final_epe for br in result.block_results if br.success) else 0.0

        if all_initial_cd_mean:
            result.global_initial_cd['cd_mean'] = float(np.mean(all_initial_cd_mean))

        if all_final_cd_mean:
            result.global_final_cd['cd_mean'] = float(np.mean(all_final_cd_mean))

        if full_mask is not None and result.stitched_mask is not None:
            try:
                from core.imaging import simulate_wafer_image
                from core.litho_metrics import compute_epe

                optical_config = self.config.global_optical_condition
                from core.imaging import OpticalSystem
                optical_system = OpticalSystem(
                    wavelength=optical_config.wavelength_nm,
                    na=optical_config.na,
                    sigma=optical_config.sigma,
                    pixel_size=self.config.pixel_size_nm,
                )

                result.original_wafer = simulate_wafer_image(
                    full_mask, optical_system=optical_system,
                    threshold=self.config.global_optical_condition.defocus_nm == 0,
                )
                result.optimized_wafer = simulate_wafer_image(
                    result.stitched_mask, optical_system=optical_system,
                    threshold=self.config.global_optical_condition.defocus_nm == 0,
                )

                target = (full_mask > 0.5).astype(np.float64)
                init_epe = compute_epe(
                    (result.original_wafer > 0.5).astype(np.float64),
                    target,
                    pixel_size=self.config.pixel_size_nm,
                )
                final_epe = compute_epe(
                    (result.optimized_wafer > 0.5).astype(np.float64),
                    target,
                    pixel_size=self.config.pixel_size_nm,
                )

                if isinstance(init_epe, dict):
                    result.global_initial_epe = init_epe
                if isinstance(final_epe, dict):
                    result.global_final_epe = final_epe

            except Exception as e:
                logger.warning(f"计算全局晶圆图像失败: {e}")

    def _generate_summaries(
        self,
        result: ChipRETResult,
        regions: List[ChipRegion],
        strategy_results: List[Any],
        block_results: List[BlockOptimizationResult],
    ) -> None:
        """
        生成各类汇总统计

        Args:
            result: 结果对象
            regions: 区域列表
            strategy_results: 策略匹配结果
            block_results: 块优化结果
        """
        region_type_stats: Dict[str, Dict[str, Any]] = {}
        strategy_stats: Dict[str, Dict[str, Any]] = {}

        for region, sr, br in zip(regions, strategy_results, block_results):
            rt = region.metadata.region_type.value
            st = sr.strategy_config.strategy_type.value

            if rt not in region_type_stats:
                region_type_stats[rt] = {
                    'count': 0,
                    'total_area_um2': 0.0,
                    'success_count': 0,
                    'avg_epe_improvement_nm': 0.0,
                    'epe_improvements': [],
                }

            if st not in strategy_stats:
                strategy_stats[st] = {
                    'count': 0,
                    'success_count': 0,
                    'avg_runtime_sec': 0.0,
                    'avg_epe_improvement_nm': 0.0,
                    'runtimes': [],
                    'epe_improvements': [],
                }

            region_type_stats[rt]['count'] += 1
            region_type_stats[rt]['total_area_um2'] += region.metadata.area_um2
            if br.success:
                region_type_stats[rt]['success_count'] += 1
                region_type_stats[rt]['epe_improvements'].append(br.epe_improvement_nm)

            strategy_stats[st]['count'] += 1
            strategy_stats[st]['runtimes'].append(br.total_time_sec)
            if br.success:
                strategy_stats[st]['success_count'] += 1
                strategy_stats[st]['epe_improvements'].append(br.epe_improvement_nm)

        for rt in region_type_stats:
            stats = region_type_stats[rt]
            if stats['epe_improvements']:
                stats['avg_epe_improvement_nm'] = float(np.mean(stats['epe_improvements']))
            stats['success_rate'] = stats['success_count'] / stats['count'] if stats['count'] > 0 else 0.0
            del stats['epe_improvements']

        for st in strategy_stats:
            stats = strategy_stats[st]
            if stats['runtimes']:
                stats['avg_runtime_sec'] = float(np.mean(stats['runtimes']))
            if stats['epe_improvements']:
                stats['avg_epe_improvement_nm'] = float(np.mean(stats['epe_improvements']))
            stats['success_rate'] = stats['success_count'] / stats['count'] if stats['count'] > 0 else 0.0
            del stats['runtimes']
            del stats['epe_improvements']

        result.region_type_summary = region_type_stats
        result.strategy_summary = strategy_stats

    def _save_results(
        self,
        result: ChipRETResult,
        regions: List[ChipRegion],
        output_dir: Union[str, Path],
    ) -> Dict[str, str]:
        """
        保存结果到文件

        Args:
            result: 结果对象
            regions: 区域列表
            output_dir: 输出目录

        Returns:
            输出文件路径映射
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_files: Dict[str, str] = {}

        if self.config.save_stitched_mask and result.stitched_mask is not None:
            stitched_path = output_path / f"{self.config.chip_name}_optimized_mask.npy"
            np.save(stitched_path, result.stitched_mask)
            output_files['stitched_mask'] = str(stitched_path)

            if result.original_mask is not None:
                original_path = output_path / f"{self.config.chip_name}_original_mask.npy"
                np.save(original_path, result.original_mask)
                output_files['original_mask'] = str(original_path)

        if self.config.save_regions_separately:
            regions_dir = output_path / "regions"
            regions_dir.mkdir(exist_ok=True)

            for region in regions:
                if region.optimized_mask is not None:
                    region_path = regions_dir / f"{region.region_id}_optimized.npy"
                    np.save(region_path, region.optimized_mask)
                    output_files[f"region_{region.region_id}"] = str(region_path)

        if self.config.save_report:
            report_path = output_path / f"{self.config.chip_name}_ret_report.json"
            report = {
                'chip_name': self.config.chip_name,
                'success': result.success,
                'total_time_sec': result.total_time_sec,
                'num_regions': result.num_regions,
                'success_rate': result.success_rate,
                'global_initial_epe': result.global_initial_epe,
                'global_final_epe': result.global_final_epe,
                'global_epe_improvement_nm': result.global_epe_improvement,
                'region_type_summary': result.region_type_summary,
                'strategy_summary': result.strategy_summary,
                'boundary_metrics': [m.to_dict() for m in result.boundary_metrics],
                'warnings': result.warnings,
                'config': self.config.to_dict(),
            }

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            output_files['report'] = str(report_path)

        logger.info(f"结果已保存到 {output_path}")
        return output_files


def run_chip_level_ret(
    gds_path: Optional[Union[str, Path]] = None,
    full_mask: Optional[np.ndarray] = None,
    chip_bounds_nm: Optional[Tuple[float, float, float, float]] = None,
    config: Optional[ChipRETConfig] = None,
    strategy_overrides: Optional[Dict[str, RETStrategyType]] = None,
) -> ChipRETResult:
    """
    便捷函数：执行芯片级 RET 优化

    Args:
        gds_path: GDS 文件路径（与 full_mask 二选一）
        full_mask: 完整芯片掩模（与 gds_path 二选一）
        chip_bounds_nm: 芯片边界 (x0, y0, x1, y1)，单位 nm
        config: 芯片级 RET 配置
        strategy_overrides: 区域 ID 到强制策略的映射

    Returns:
        ChipRETResult 完整的优化结果

    Examples:
        # 从 GDS 文件运行
        result = run_chip_level_ret(
            gds_path="chip.gds",
            config=ChipRETConfig(chip_name="my_chip", layer=1)
        )

        # 从掩模数组运行
        result = run_chip_level_ret(
            full_mask=mask_array,
            chip_bounds_nm=(0, 0, 10000, 10000),
            config=ChipRETConfig(chip_name="my_chip")
        )
    """
    orchestrator = ChipRETOrchestrator(config=config)
    return orchestrator.run(
        gds_path=gds_path,
        full_mask=full_mask,
        chip_bounds_nm=chip_bounds_nm,
        strategy_overrides=strategy_overrides,
    )
