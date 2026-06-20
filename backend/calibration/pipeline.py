# -*- coding: utf-8 -*-
"""
Fab 模型标定：端到端流水线

串联：数据加载 → 校验清洗 → 数据集划分 → 参数反演 → 报告生成 → 配置输出。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import time
from datetime import datetime

import yaml

from .schemas import (
    CalibrationConfig,
    CalibrationParameterSet,
    CDSEMDataset,
    InversionMethod,
    CalibrationReport,
)
from .data_loader import (
    load_cd_sem_data,
    validate_dataset,
    split_dataset,
)
from .inversion import (
    run_inversion,
    InversionEngine,
)
from .report_generator import (
    generate_calibration_report,
    generate_markdown_report,
    ReportGenerator,
    plot_calibration_results,
    plot_bossung_curves,
    plot_parameter_convergence,
    plot_residual_analysis,
)
from .config_updater import (
    build_config_snippet,
    update_default_config,
    save_config_snippet,
    ConfigUpdater,
)

logger = logging.getLogger(__name__)


class CalibrationPipeline:
    """
    标定流水线。

    典型用法::

        pipeline = CalibrationPipeline(config)
        result = pipeline.run(data_path)
        print(result.summary())

    或者从 YAML 配置文件启动::

        pipeline = calibration_pipeline_from_config('calib_config.yaml')
        result = pipeline.run()
    """

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.dataset: Optional[CDSEMDataset] = None
        self.train_ds: Optional[CDSEMDataset] = None
        self.test_ds: Optional[CDSEMDataset] = None
        self.validation_report: Optional[Dict[str, Any]] = None
        self.inversion_result = None
        self.report: Optional[CalibrationReport] = None
        self.output_paths: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self,
            dataset: Union[CDSEMDataset, str, Path, None] = None,
            dataset_path: Optional[Union[str, Path]] = None,
            output_dir: Optional[Union[str, Path]] = None,
            ) -> CalibrationReport:
        """
        运行完整标定流水线。

        Args:
            dataset: 已加载的数据集，或数据文件路径
            dataset_path: （兼容）数据文件路径，优先级低于 dataset
            output_dir: 输出目录，覆盖 config 中的设置

        Returns:
            CalibrationReport 对象
        """
        t_total = time.time()
        if output_dir is not None:
            self.config.output_dir = str(output_dir)
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) 数据加载
        if isinstance(dataset, CDSEMDataset):
            self.dataset = dataset
        elif isinstance(dataset, (str, Path)):
            self.dataset = load_cd_sem_data(dataset)
        elif dataset_path is not None:
            self.dataset = load_cd_sem_data(dataset_path)
        elif self.config.dataset_path is not None:
            self.dataset = load_cd_sem_data(self.config.dataset_path)
        else:
            raise ValueError("未提供数据集 (dataset / dataset_path / config.dataset_path)")

        logger.info(f"[1/5] 数据加载完成，共 {len(self.dataset)} 个量测点")

        # 2) 校验与清洗
        self.dataset, self.validation_report = validate_dataset(
            self.dataset, remove_outliers=True, outlier_sigma=4.0,
        )
        if self.validation_report.get('warnings'):
            for w in self.validation_report['warnings']:
                logger.warning(f"  数据校验警告: {w}")
        if self.validation_report.get('removed_count', 0) > 0:
            logger.info(
                f"[2/5] 数据清洗完成，移除 {self.validation_report['removed_count']} 点，"
                f"剩余 {self.validation_report['final_count']} 点"
            )
        else:
            logger.info("[2/5] 数据校验通过，无需清洗")

        # 3) 划分训练/测试集（数据量足够时）
        if len(self.dataset) >= 20:
            self.train_ds, self.test_ds = split_dataset(
                self.dataset, train_frac=0.8,
                random_seed=self.config.random_seed,
            )
            logger.info(
                f"[3/5] 数据集划分：训练 {len(self.train_ds)} / 测试 {len(self.test_ds)}"
            )
        else:
            self.train_ds = self.dataset
            self.test_ds = None
            logger.info("[3/5] 数据量 < 20，跳过训练/测试划分，全部用于训练")

        # 4) 参数反演
        logger.info("[4/5] 执行参数反演...")
        t_inv = time.time()
        engine = InversionEngine(self.config)
        self.inversion_result = engine.run(self.train_ds)
        inv_duration = time.time() - t_inv
        logger.info(
            f"[4/5] 反演完成：成功={self.inversion_result.success}，"
            f"耗时 {inv_duration:.2f} s"
        )

        # 5) 报告与配置输出
        logger.info("[5/5] 生成报告与配置...")
        self.report = generate_calibration_report(
            self.config, self.train_ds, self.inversion_result,
            test_dataset=self.test_ds,
            validation_report=self.validation_report,
            duration_sec=inv_duration,
        )

        # 生成输出文件
        report_gen = ReportGenerator(self.report)
        self.output_paths = report_gen.generate_all(
            out_dir,
            dataset=self.train_ds,
            generate_plots=self.config.generate_plots,
            plot_format=self.config.plot_format,
        )

        # YAML 配置片段
        if self.config.update_config:
            ref_path = self.config.reference_config_path
            cfg_paths = save_config_snippet(
                self.inversion_result, self.config.parameters,
                out_dir,
                reference_config_path=ref_path,
                update_full_config=(ref_path is not None),
            )
            self.output_paths.update(cfg_paths)

            # 同时把 snippet 写入 CalibrationReport
            snippet_path = Path(cfg_paths['snippet'])
            if snippet_path.exists():
                with open(snippet_path, 'r', encoding='utf-8') as f:
                    self.report.config_snippet = f.read()

        total_duration = time.time() - t_total
        logger.info(f"标定流水线全部完成，总耗时 {total_duration:.2f} s")
        logger.info(f"输出目录: {out_dir.resolve()}")
        return self.report


# ---------------------------------------------------------------------------
# 从配置文件启动流水线
# ---------------------------------------------------------------------------

def _dict_to_config(config_dict: Dict[str, Any]) -> CalibrationConfig:
    """从 YAML 字典构造 CalibrationConfig。"""
    cfg = CalibrationConfig()

    # 顶层字段
    if 'method' in config_dict:
        cfg.method = InversionMethod(str(config_dict['method']).lower())
    if 'dataset_path' in config_dict:
        cfg.dataset_path = str(config_dict['dataset_path'])
    if 'output_dir' in config_dict:
        cfg.output_dir = str(config_dict['output_dir'])
    if 'random_seed' in config_dict:
        cfg.random_seed = (int(config_dict['random_seed'])
                           if config_dict['random_seed'] is not None else None)
    if 'use_measurement_weights' in config_dict:
        cfg.use_measurement_weights = bool(config_dict['use_measurement_weights'])
    if 'forward_model_complexity' in config_dict:
        cfg.forward_model_complexity = str(config_dict['forward_model_complexity'])
    if 'reference_config_path' in config_dict:
        cfg.reference_config_path = str(config_dict['reference_config_path'])
    if 'generate_plots' in config_dict:
        cfg.generate_plots = bool(config_dict['generate_plots'])
    if 'plot_format' in config_dict:
        cfg.plot_format = str(config_dict['plot_format'])
    if 'update_config' in config_dict:
        cfg.update_config = bool(config_dict['update_config'])

    # NLLS
    nlls = config_dict.get('nlls', {})
    if isinstance(nlls, dict):
        if 'max_iter' in nlls:
            cfg.nlls_max_iter = int(nlls['max_iter'])
        if 'method' in nlls:
            cfg.nlls_method = str(nlls['method'])
        if 'ftol' in nlls:
            cfg.nlls_ftol = float(nlls['ftol'])
        if 'xtol' in nlls:
            cfg.nlls_xtol = float(nlls['xtol'])

    # MCMC
    mcmc = config_dict.get('mcmc', {})
    if isinstance(mcmc, dict):
        if 'n_walkers' in mcmc:
            cfg.mcmc_n_walkers = int(mcmc['n_walkers'])
        if 'n_steps' in mcmc:
            cfg.mcmc_n_steps = int(mcmc['n_steps'])
        if 'n_burnin' in mcmc:
            cfg.mcmc_n_burnin = int(mcmc['n_burnin'])

    # 参数集合
    params_section = config_dict.get('parameters', [])
    if isinstance(params_section, list):
        ps = CalibrationParameterSet()
        existing = {p.name: p for p in ps.all_parameters()}
        for item in params_section:
            if not isinstance(item, dict) or 'name' not in item:
                continue
            name = item['name']
            if name in existing:
                pobj = existing[name]
            else:
                pobj = ps.resist_threshold  # placeholder
            if 'initial_value' in item:
                pobj.initial_value = float(item['initial_value'])
            if 'lower_bound' in item:
                pobj.lower_bound = (float(item['lower_bound'])
                                    if item['lower_bound'] is not None else None)
            if 'upper_bound' in item:
                pobj.upper_bound = (float(item['upper_bound'])
                                    if item['upper_bound'] is not None else None)
            if 'vary' in item:
                pobj.vary = bool(item['vary'])
            if 'prior_mean' in item:
                pobj.prior_mean = (float(item['prior_mean'])
                                   if item['prior_mean'] is not None else None)
            if 'prior_std' in item:
                pobj.prior_std = (float(item['prior_std'])
                                  if item['prior_std'] is not None else None)
            if 'config_path' in item:
                pobj.config_path = (str(item['config_path'])
                                    if item['config_path'] is not None else None)
            if 'unit' in item:
                pobj.unit = str(item['unit'])
            if 'description' in item:
                pobj.description = str(item['description'])
        cfg.parameters = ps

    return cfg


def calibration_pipeline_from_config(config_path: Union[str, Path]) -> CalibrationPipeline:
    """
    从 YAML 配置文件构造 CalibrationPipeline。

    配置示例::

        method: lmfit
        dataset_path: ./data/fab_cd_sem.csv
        output_dir: ./results/calib_20260120
        reference_config_path: ../config/default_config.yaml
        use_measurement_weights: true

        parameters:
          - name: resist_threshold
            initial_value: 0.30
            lower_bound: 0.05
            upper_bound: 0.80
            vary: true
          - name: na_effective
            initial_value: 1.35
            vary: true

        nlls:
          max_iter: 10000
          method: trf

        mcmc:
          n_walkers: 32
          n_steps: 5000
          n_burnin: 1000
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"标定配置文件不存在: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    config = _dict_to_config(raw)
    return CalibrationPipeline(config)


def run_calibration_pipeline(config: Union[CalibrationConfig, str, Path],
                             dataset: Union[CDSEMDataset, str, Path, None] = None,
                             output_dir: Optional[Union[str, Path]] = None,
                             ) -> CalibrationReport:
    """
    便捷函数：一站式执行标定流水线。

    Args:
        config: CalibrationConfig 对象或 YAML 配置文件路径
        dataset: CDSEMDataset 或数据文件路径（若 config 中已指定 dataset_path 可省略）
        output_dir: 输出目录，覆盖配置中的设置

    Returns:
        CalibrationReport 对象
    """
    if isinstance(config, CalibrationConfig):
        pipeline = CalibrationPipeline(config)
    else:
        pipeline = calibration_pipeline_from_config(config)
    return pipeline.run(dataset=dataset, output_dir=output_dir)
