# -*- coding: utf-8 -*-
"""
Fab 闭环反馈校准：端到端流水线

串联完整闭环流程：
  1. 从 Fab 导入最新 CD-SEM CSV
  2. 使用当前模型参数做仿真预测
  3. 对比预测与实测，计算偏差
  4. 偏差超阈值时自动触发 calibration 重新校准模型
  5. 校准完成后重新评估所有在产掩模的 PW 余量
  6. 输出完整闭环周期报告

形成 "仿真 → 量产 → 反馈 → 再仿真" 的数据闭环。
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from backend.calibration.schemas import (
    CalibrationConfig,
    CDSEMDataset,
)

from .schemas import (
    ClosedLoopConfig,
    ClosedLoopCycle,
    ClosedLoopState,
    FabImportResult,
    ComparisonResult,
    CalibrationTriggerResult,
    PWReassessmentResult,
    ProductionMask,
    FabImportConfig,
    CalibrationTriggerThresholds,
)
from .fab_importer import FabDataImporter
from .comparator import PredictionComparator
from .calibration_trigger import CalibrationTrigger
from .pw_reassessor import PWReassessor

logger = logging.getLogger(__name__)


class ClosedLoopPipeline:
    """
    Fab 闭环反馈校准流水线

    典型用法::

        # 方式1: 从配置文件启动
        pipeline = ClosedLoopPipeline.from_config(config)
        cycle = pipeline.run_cycle()

        # 方式2: 手动组装
        pipeline = ClosedLoopPipeline(
            closed_loop_config,
            production_masks=[...],
        )
        cycle = pipeline.run_cycle()
        print(cycle.summary())
    """

    def __init__(self,
                 config: ClosedLoopConfig,
                 production_masks: Optional[List[ProductionMask]] = None,
                 ):
        """
        Args:
            config: 闭环系统配置
            production_masks: 在产掩模列表（用于 PW 重评估）
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._importer = FabDataImporter(config.import_config)
        self._comparator = PredictionComparator()
        self._trigger = CalibrationTrigger(
            calibration_config=config.calibration_config,
            thresholds=config.trigger_thresholds,
            reference_config_path=config.reference_config_path,
        )
        self._reassessor = PWReassessor(
            production_masks=production_masks or [],
            focus_range=config.pw_scan_focus,
            dose_range=config.pw_scan_dose,
            cd_tolerance=config.pw_cd_tolerance,
            pw_drop_threshold=config.pw_drop_threshold,
            reference_config_path=config.reference_config_path,
        )

        self._cycle_history: List[ClosedLoopCycle] = []
        self._current_cycle: Optional[ClosedLoopCycle] = None

    # ------------------------------------------------------------------
    # 掩模管理
    # ------------------------------------------------------------------
    def add_production_mask(self, mask: ProductionMask) -> None:
        """添加在产掩模"""
        self._reassessor.add_mask(mask)

    def add_production_masks(self, masks: List[ProductionMask]) -> None:
        """批量添加在产掩模"""
        self._reassessor.add_masks(masks)

    @property
    def production_masks(self) -> List[ProductionMask]:
        return self._reassessor.production_masks

    # ------------------------------------------------------------------
    # 子步骤
    # ------------------------------------------------------------------
    def _step_import(self, cycle: ClosedLoopCycle,
                     force_import: bool = False) -> FabImportResult:
        """步骤1: 导入 Fab 新数据"""
        cycle.state = ClosedLoopState.IMPORTING
        logger.info("[1/4] 导入 Fab CD-SEM 数据...")
        result = self._importer.import_new_data(force=force_import)
        cycle.import_result = result
        logger.info(
            f"      → 新文件 {result.new_files_count}, "
            f"总量测点 {result.total_points}"
        )
        return result

    def _step_compare(self, cycle: ClosedLoopCycle,
                      dataset: CDSEMDataset) -> ComparisonResult:
        """步骤2: 仿真预测 vs 量产量测对比"""
        cycle.state = ClosedLoopState.COMPARING
        logger.info("[2/4] 对比分析: 仿真预测 vs 量产量测...")
        result = self._comparator.compare(
            dataset, thresholds=self.config.trigger_thresholds
        )
        cycle.comparison_result = result
        logger.info(
            f"      → RMSE={result.rmse:.3f} nm, "
            f"bias={result.mean_residual:+.3f} nm, "
            f"建议校准={'是' if result.needs_calibration else '否'}"
        )
        return result

    def _step_calibrate(self, cycle: ClosedLoopCycle,
                        comparison: ComparisonResult,
                        dataset: CDSEMDataset,
                        force_calibrate: bool = False,
                        ) -> CalibrationTriggerResult:
        """步骤3: 按需触发校准"""
        cycle.state = ClosedLoopState.CALIBRATING
        logger.info("[3/4] 评估并触发模型校准...")

        calib_output = self.output_dir / f"cycle_{cycle.cycle_id}" / "calibration"
        result = self._trigger.evaluate_and_run(
            comparison, dataset,
            output_dir=calib_output,
            force=force_calibrate,
        )
        cycle.calibration_result = result

        if result.triggered:
            logger.info(f"      → 校准已执行, 耗时 {result.duration_sec:.1f}s")
            self._comparator.update_params(
                self._extract_calibrated_params(result)
            )
        else:
            logger.info(f"      → {result.skipped_reason}")
        return result

    def _step_reevaluate_pw(self, cycle: ClosedLoopCycle,
                            calibration_result: CalibrationTriggerResult,
                            ) -> PWReassessmentResult:
        """步骤4: 重新评估在产掩模 PW 余量"""
        cycle.state = ClosedLoopState.REASSESSING_PW
        if not self.config.reevaluate_pw:
            logger.info("[4/4] 跳过 PW 重评估 (config.reevaluate_pw=False)")
            result = PWReassessmentResult(
                n_masks_total=len(self.production_masks),
            )
            cycle.pw_result = result
            return result

        logger.info(
            f"[4/4] 重新评估 {len(self.production_masks)} "
            f"个在产掩模的 PW 余量..."
        )
        result = self._reassessor.reevaluate_all(
            calibration_report=calibration_result.calibration_report,
        )
        cycle.pw_result = result
        logger.info(
            f"      → 评估 {result.n_masks_reevaluated} 个, "
            f"需重 OPC={result.n_masks_needs_ropc}"
        )
        return result

    @staticmethod
    def _extract_calibrated_params(
        trigger_result: CalibrationTriggerResult,
    ) -> Dict[str, float]:
        """从校准结果提取参数字典"""
        if not trigger_result.calibration_report:
            return {}
        return dict(
            trigger_result.calibration_report.inversion_result.calibrated_values
        )

    # ------------------------------------------------------------------
    # 主入口：运行单个完整周期
    # ------------------------------------------------------------------
    def run_cycle(self,
                  force_import: bool = False,
                  force_calibrate: bool = False,
                  skip_if_no_new_data: bool = True,
                  ) -> ClosedLoopCycle:
        """
        运行完整的一个闭环周期

        Args:
            force_import: 强制重新导入（即使文件已处理过）
            force_calibrate: 强制触发校准（即使偏差在阈值内）
            skip_if_no_new_data: 没有新数据时直接返回，跳过后续步骤

        Returns:
            ClosedLoopCycle，包含各步骤结果
        """
        cycle_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        cycle = ClosedLoopCycle(cycle_id=cycle_id)
        self._current_cycle = cycle

        logger.info("=" * 60)
        logger.info(f" 开始闭环周期 {cycle_id}")
        logger.info("=" * 60)

        try:
            import_result = self._step_import(cycle, force_import=force_import)

            if (skip_if_no_new_data
                    and import_result.merged_dataset is None):
                logger.info("无新数据，跳过本周期剩余步骤")
                cycle.state = ClosedLoopState.SKIPPED
                cycle.mark_completed()
                self._cycle_history.append(cycle)
                return cycle

            dataset = import_result.merged_dataset
            if dataset is None or len(dataset) == 0:
                raise ValueError("导入后数据集为空")

            comparison = self._step_compare(cycle, dataset)
            calibration = self._step_calibrate(
                cycle, comparison, dataset,
                force_calibrate=force_calibrate,
            )
            self._step_reevaluate_pw(cycle, calibration)

            cycle.mark_completed()
            self._save_cycle_report(cycle)

        except Exception as e:
            logger.error(f"闭环周期执行失败: {e}", exc_info=True)
            cycle.mark_failed(str(e))
            self._save_cycle_report(cycle)

        self._cycle_history.append(cycle)

        logger.info("")
        logger.info(cycle.summary())
        logger.info("")
        return cycle

    # ------------------------------------------------------------------
    # 报告持久化
    # ------------------------------------------------------------------
    def _save_cycle_report(self, cycle: ClosedLoopCycle) -> None:
        """保存周期报告为 JSON 和 Markdown"""
        cycle_dir = self.output_dir / f"cycle_{cycle.cycle_id}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        json_path = cycle_dir / "cycle_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cycle.to_dict(), f, ensure_ascii=False, indent=2)

        md_path = cycle_dir / "cycle_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(cycle.summary())

        logger.info(f"周期报告已保存: {cycle_dir}")

    # ------------------------------------------------------------------
    # 历史记录
    # ------------------------------------------------------------------
    def get_history(self, limit: Optional[int] = None) -> List[ClosedLoopCycle]:
        """获取历史周期记录"""
        if limit is not None:
            return self._cycle_history[-limit:]
        return list(self._cycle_history)

    @property
    def last_cycle(self) -> Optional[ClosedLoopCycle]:
        return self._cycle_history[-1] if self._cycle_history else None

    # ------------------------------------------------------------------
    # 配置构造
    # ------------------------------------------------------------------
    @staticmethod
    def from_config_dict(config_dict: Dict[str, Any]) -> 'ClosedLoopPipeline':
        """从字典构造闭环流水线配置"""
        import_cfg_dict = config_dict.get('import_config', {})
        import_cfg = FabImportConfig(
            watch_dir=import_cfg_dict.get('watch_dir', './fab_data/incoming'),
            file_pattern=import_cfg_dict.get('file_pattern', 'cd_sem_*.csv'),
            archive_dir=import_cfg_dict.get('archive_dir'),
            history_file=import_cfg_dict.get(
                'history_file', './closed_loop/import_history.json'
            ),
        )

        thresh_dict = config_dict.get('trigger_thresholds', {})
        thresholds = CalibrationTriggerThresholds(
            rmse_threshold_nm=thresh_dict.get('rmse_threshold_nm', 2.0),
            bias_threshold_nm=thresh_dict.get('bias_threshold_nm', 1.0),
            max_residual_threshold_nm=thresh_dict.get(
                'max_residual_threshold_nm', 5.0
            ),
        )

        loop_cfg = ClosedLoopConfig(
            import_config=import_cfg,
            trigger_thresholds=thresholds,
            output_dir=config_dict.get('output_dir', './closed_loop/output'),
            reference_config_path=config_dict.get('reference_config_path'),
            reevaluate_pw=config_dict.get('reevaluate_pw', True),
        )
        return ClosedLoopPipeline(loop_cfg)


def run_closed_loop_cycle(
    config: Union[ClosedLoopConfig, Dict[str, Any]],
    production_masks: Optional[List[ProductionMask]] = None,
    force_import: bool = False,
    force_calibrate: bool = False,
) -> ClosedLoopCycle:
    """
    便捷函数：执行单个完整闭环周期

    Args:
        config: ClosedLoopConfig 或字典
        production_masks: 在产掩模列表
        force_import: 强制重新导入
        force_calibrate: 强制触发校准

    Returns:
        ClosedLoopCycle
    """
    if isinstance(config, dict):
        pipeline = ClosedLoopPipeline.from_config_dict(config)
    else:
        pipeline = ClosedLoopPipeline(config, production_masks=production_masks)
    if production_masks and isinstance(config, ClosedLoopConfig):
        pipeline.add_production_masks(production_masks)
    return pipeline.run_cycle(
        force_import=force_import,
        force_calibrate=force_calibrate,
    )
