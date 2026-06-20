# -*- coding: utf-8 -*-
"""
CD-SEM 数据加载与预处理模块

支持从 CSV、JSON、YAML 文件加载 Fab 提供的 CD-SEM 量测数据，
并执行数据校验、异常值剔除、数据集划分等预处理操作。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import csv
import json

import yaml

from .schemas import (
    CDSEMDataPoint,
    CDSEMDataset,
    PatternType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV 格式加载
# ---------------------------------------------------------------------------

# CSV 列名映射（支持多种列名约定）
_CSV_COLUMN_ALIASES: Dict[str, List[str]] = {
    'measurement_id': ['measurement_id', 'meas_id', 'id', 'point_id', 'index'],
    'site_name': ['site_name', 'site', 'location', 'die_site', 'wafer_site'],
    'target_cd': ['target_cd', 'target', 'design_cd', 'nominal_cd', 'cd_target', 'cd_design'],
    'measured_cd': ['measured_cd', 'meas_cd', 'cd', 'cd_meas', 'actual_cd', 'sem_cd'],
    'focus': ['focus', 'defocus', 'focus_offset', 'focus_nm', 'z_offset'],
    'dose': ['dose', 'exposure_dose', 'dose_factor', 'relative_dose', 'dose_rel'],
    'pattern_type': ['pattern_type', 'pattern', 'feature_type', 'feature'],
    'pitch': ['pitch', 'period', 'line_space_pitch', 'hpitch'],
    'measurement_uncertainty': [
        'measurement_uncertainty', 'uncertainty', 'cd_error', 'sem_error',
        'uncertainty_1s', 'sigma_cd',
    ],
    'mask_cd': ['mask_cd', 'reticle_cd', 'mask_linewidth', 'reticle_cd_um'],
    'layer': ['layer', 'process_layer', 'mask_layer', 'metal_layer'],
    'timestamp': ['timestamp', 'measurement_time', 'time', 'date'],
}


def _resolve_csv_column(header: List[str], aliases: List[str]) -> Optional[str]:
    """在表头中匹配别名，返回实际列名；不区分大小写。"""
    header_lower = {h.strip().lower(): h for h in header}
    for alias in aliases:
        if alias.lower() in header_lower:
            return header_lower[alias.lower()]
    return None


def load_cd_sem_from_csv(filepath: Union[str, Path],
                         encoding: str = 'utf-8-sig',
                         delimiter: str = ',') -> CDSEMDataset:
    """
    从 CSV 文件加载 CD-SEM 数据。

    CSV 必需列（大小写不敏感）：
        target_cd, measured_cd, focus, dose
    可选列：
        measurement_id, site_name, pattern_type, pitch, measurement_uncertainty,
        mask_cd, layer, timestamp

    Args:
        filepath: CSV 文件路径
        encoding: 文件编码（默认带 BOM 的 UTF-8，兼容 Excel 导出）
        delimiter: 列分隔符（逗号或制表符等）

    Returns:
        CDSEMDataset 对象
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CD-SEM 数据文件不存在: {filepath}")

    dataset = CDSEMDataset()
    with open(filepath, 'r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"CSV 文件没有表头: {filepath}")

        header = list(reader.fieldnames)

        required = ['target_cd', 'measured_cd', 'focus', 'dose']
        for req in required:
            if _resolve_csv_column(header, _CSV_COLUMN_ALIASES[req]) is None:
                raise ValueError(
                    f"CSV 缺少必需列 '{req}'（或别名），实际表头: {header}"
                )

        for i, row in enumerate(reader):
            try:
                point = _parse_csv_row(row, header, i)
                dataset.add_point(point)
            except (ValueError, TypeError) as e:
                logger.warning(f"跳过 CSV 第 {i+2} 行，解析失败: {e}")
                continue

    logger.info(f"从 CSV 加载 {len(dataset)} 个 CD-SEM 量测点: {filepath}")
    return dataset


def _parse_csv_row(row: Dict[str, str],
                   header: List[str],
                   index: int) -> CDSEMDataPoint:
    """解析单个 CSV 行。"""

    def col(aliases_key: str) -> Optional[str]:
        name = _resolve_csv_column(header, _CSV_COLUMN_ALIASES[aliases_key])
        if name is None:
            return None
        return row.get(name, None)

    def to_float(value: Optional[str], field: str, default: float = None) -> Optional[float]:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"字段 '{field}' 无法转为 float: '{value}'")

    target_cd = to_float(col('target_cd'), 'target_cd')
    measured_cd = to_float(col('measured_cd'), 'measured_cd')
    focus = to_float(col('focus'), 'focus')
    dose = to_float(col('dose'), 'dose')

    if target_cd is None or measured_cd is None or focus is None or dose is None:
        raise ValueError(f"存在空值的必需字段 (target_cd/measured_cd/focus/dose)")

    pattern_str = col('pattern_type')
    try:
        pattern_type = (PatternType(pattern_str.strip().lower())
                        if pattern_str and pattern_str.strip()
                        else PatternType.LINE_SPACE)
    except ValueError:
        pattern_type = PatternType.LINE_SPACE

    pitch = to_float(col('pitch'), 'pitch')
    uncertainty = to_float(col('measurement_uncertainty'),
                           'measurement_uncertainty', default=1.0)
    mask_cd = to_float(col('mask_cd'), 'mask_cd')
    meas_id = col('measurement_id') or f"csv_{index+1:05d}"
    site = col('site_name') or ""
    layer = col('layer') or ""
    timestamp = col('timestamp') or ""

    return CDSEMDataPoint(
        measurement_id=meas_id,
        site_name=site,
        target_cd=target_cd,
        measured_cd=measured_cd,
        focus=focus,
        dose=dose,
        pattern_type=pattern_type,
        pitch=pitch,
        measurement_uncertainty=uncertainty,
        mask_cd=mask_cd,
        layer=layer,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# JSON 格式加载
# ---------------------------------------------------------------------------

def load_cd_sem_from_json(filepath: Union[str, Path]) -> CDSEMDataset:
    """
    从 JSON 文件加载 CD-SEM 数据。

    JSON 结构支持两种：
    1. 顶层为数组，每个元素是一个量测点对象
    2. 顶层为对象，包含 'points' 数组以及元信息字段
       ('fab_name', 'process_node', 'magnification', ...)

    Returns:
        CDSEMDataset 对象
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CD-SEM 数据文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = CDSEMDataset()

    if isinstance(data, list):
        points_raw = data
    elif isinstance(data, dict):
        points_raw = data.get('points', data.get('measurements', []))
        dataset.fab_name = data.get('fab_name', '')
        dataset.process_node = data.get('process_node', '')
        dataset.mask_set_id = data.get('mask_set_id', '')
        dataset.wafer_id = data.get('wafer_id', '')
        dataset.lot_id = data.get('lot_id', '')
        dataset.magnification = data.get('magnification', 4.0)
        dataset.comments = data.get('comments', '')
    else:
        raise ValueError(f"JSON 顶层必须是 list 或 dict: {filepath}")

    for i, raw in enumerate(points_raw):
        try:
            point = _dict_to_datapoint(raw, fallback_id=f"json_{i+1:05d}")
            dataset.add_point(point)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"跳过 JSON 第 {i+1} 个点，解析失败: {e}")
            continue

    logger.info(f"从 JSON 加载 {len(dataset)} 个 CD-SEM 量测点: {filepath}")
    return dataset


# ---------------------------------------------------------------------------
# YAML 格式加载
# ---------------------------------------------------------------------------

def load_cd_sem_from_yaml(filepath: Union[str, Path]) -> CDSEMDataset:
    """
    从 YAML 文件加载 CD-SEM 数据。格式与 JSON 一致。
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CD-SEM 数据文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    dataset = CDSEMDataset()

    if isinstance(data, list):
        points_raw = data
    elif isinstance(data, dict):
        points_raw = data.get('points', data.get('measurements', []))
        dataset.fab_name = data.get('fab_name', '')
        dataset.process_node = data.get('process_node', '')
        dataset.mask_set_id = data.get('mask_set_id', '')
        dataset.wafer_id = data.get('wafer_id', '')
        dataset.lot_id = data.get('lot_id', '')
        dataset.magnification = data.get('magnification', 4.0)
        dataset.comments = data.get('comments', '')
    else:
        raise ValueError(f"YAML 顶层必须是 list 或 dict: {filepath}")

    for i, raw in enumerate(points_raw):
        try:
            point = _dict_to_datapoint(raw, fallback_id=f"yaml_{i+1:05d}")
            dataset.add_point(point)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"跳过 YAML 第 {i+1} 个点，解析失败: {e}")
            continue

    logger.info(f"从 YAML 加载 {len(dataset)} 个 CD-SEM 量测点: {filepath}")
    return dataset


def _dict_to_datapoint(raw: Dict[str, Any],
                       fallback_id: str = "") -> CDSEMDataPoint:
    """把 dict 转为 CDSEMDataPoint（兼容多种字段命名）。"""

    def g(*keys, default=None):
        for k in keys:
            if k in raw and raw[k] is not None:
                return raw[k]
        return default

    target_cd = g('target_cd', 'target', 'design_cd', 'nominal_cd')
    measured_cd = g('measured_cd', 'meas_cd', 'cd', 'actual_cd', 'sem_cd')
    focus = g('focus', 'defocus', 'focus_offset', 'focus_nm')
    dose = g('dose', 'exposure_dose', 'dose_factor', 'relative_dose')

    if target_cd is None or measured_cd is None or focus is None or dose is None:
        raise ValueError(
            f"缺少必需字段 (target_cd/measured_cd/focus/dose): {raw.keys()}"
        )

    pattern_str = g('pattern_type', 'pattern', 'feature_type')
    try:
        pattern_type = (PatternType(str(pattern_str).strip().lower())
                        if pattern_str else PatternType.LINE_SPACE)
    except ValueError:
        pattern_type = PatternType.LINE_SPACE

    return CDSEMDataPoint(
        measurement_id=str(g('measurement_id', 'meas_id', 'id', default=fallback_id)),
        site_name=str(g('site_name', 'site', 'location', default='')),
        target_cd=float(target_cd),
        measured_cd=float(measured_cd),
        focus=float(focus),
        dose=float(dose),
        pattern_type=pattern_type,
        pitch=(float(p) if (p := g('pitch', 'period')) is not None else None),
        measurement_uncertainty=float(g('measurement_uncertainty', 'uncertainty',
                                         'cd_error', default=1.0)),
        mask_cd=(float(m) if (m := g('mask_cd', 'reticle_cd')) is not None else None),
        layer=str(g('layer', 'process_layer', default='')),
        timestamp=str(g('timestamp', 'measurement_time', default='')),
        extra={k: v for k, v in raw.items()
               if k not in {'target_cd', 'target', 'design_cd', 'nominal_cd',
                            'measured_cd', 'meas_cd', 'cd', 'actual_cd', 'sem_cd',
                            'focus', 'defocus', 'focus_offset', 'focus_nm',
                            'dose', 'exposure_dose', 'dose_factor', 'relative_dose',
                            'pattern_type', 'pattern', 'feature_type',
                            'pitch', 'period',
                            'measurement_uncertainty', 'uncertainty', 'cd_error',
                            'mask_cd', 'reticle_cd',
                            'measurement_id', 'meas_id', 'id',
                            'site_name', 'site', 'location',
                            'layer', 'process_layer',
                            'timestamp', 'measurement_time'}},
    )


# ---------------------------------------------------------------------------
# 通用入口 / 校验 / 划分
# ---------------------------------------------------------------------------

def load_cd_sem_data(filepath: Union[str, Path]) -> CDSEMDataset:
    """
    根据文件扩展名自动选择加载器。

    支持 .csv / .json / .yaml / .yml
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CD-SEM 数据文件不存在: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix == '.csv':
        return load_cd_sem_from_csv(filepath)
    elif suffix == '.json':
        return load_cd_sem_from_json(filepath)
    elif suffix in ('.yaml', '.yml'):
        return load_cd_sem_from_yaml(filepath)
    else:
        raise ValueError(
            f"不支持的文件格式: {suffix}（支持 .csv, .json, .yaml, .yml）"
        )


def validate_dataset(dataset: CDSEMDataset,
                     remove_outliers: bool = True,
                     outlier_sigma: float = 4.0,
                     min_points: int = 10,
                     ) -> Tuple[CDSEMDataset, Dict[str, Any]]:
    """
    校验并清洗数据集。

    检查项：
    - 数据点数量
    - target_cd / measured_cd 为正
    - focus/dose 合理范围
    - 重复测量条件处理
    - (可选) 基于残差的离群点剔除（4σ 规则）

    Args:
        dataset: 原始数据集
        remove_outliers: 是否剔除离群点
        outlier_sigma: 离群点阈值 (σ)
        min_points: 最少数据点数

    Returns:
        (清洗后数据集, 校验报告字典)
    """
    report: Dict[str, Any] = {
        'original_count': len(dataset),
        'removed_count': 0,
        'removed_reasons': [],
        'warnings': [],
    }

    if len(dataset) < min_points:
        report['warnings'].append(
            f"数据点过少 ({len(dataset)} < {min_points})，反演可能不稳定"
        )

    good: List[CDSEMDataPoint] = []
    for p in dataset.points:
        reasons = []
        if p.target_cd <= 0:
            reasons.append(f"target_cd={p.target_cd:.2f} <= 0")
        if p.measured_cd <= 0:
            reasons.append(f"measured_cd={p.measured_cd:.2f} <= 0")
        if abs(p.focus) > 500:
            reasons.append(f"focus={p.focus:.2f} 超出合理范围 ±500 nm")
        if p.dose <= 0 or p.dose > 5.0:
            reasons.append(f"dose={p.dose:.3f} 超出合理范围 (0, 5.0]")
        if abs(p.measured_cd - p.target_cd) / max(p.target_cd, 1e-6) > 1.0:
            reasons.append(
                f"|measured-target|/target > 100%: "
                f"target={p.target_cd:.2f}, measured={p.measured_cd:.2f}"
            )

        if reasons:
            report['removed_count'] += 1
            report['removed_reasons'].append({
                'id': p.measurement_id,
                'reasons': reasons,
            })
        else:
            good.append(p)

    cleaned = CDSEMDataset(
        points=good,
        magnification=dataset.magnification,
        fab_name=dataset.fab_name,
        process_node=dataset.process_node,
        mask_set_id=dataset.mask_set_id,
        wafer_id=dataset.wafer_id,
        lot_id=dataset.lot_id,
        comments=f"{dataset.comments}; validated (removed {report['removed_count']} invalid)",
    )

    if remove_outliers and len(cleaned) >= min_points:
        cleaned, outlier_report = _remove_outliers_by_residual(
            cleaned, outlier_sigma
        )
        report['removed_count'] += outlier_report['removed']
        report['removed_reasons'].extend(outlier_report['details'])

    report['final_count'] = len(cleaned)
    logger.info(
        f"数据集校验完成：原始 {report['original_count']} → "
        f"清洗后 {report['final_count']}，移除 {report['removed_count']}"
    )
    return cleaned, report


def _remove_outliers_by_residual(dataset: CDSEMDataset,
                                 sigma: float) -> Tuple[CDSEMDataset, Dict[str, Any]]:
    """用稳健线性拟合 (focus, dose → cd) 估计残差，剔除显著离群点。"""
    if len(dataset) < 8:
        return dataset, {'removed': 0, 'details': []}

    focuses = np.array([p.focus for p in dataset.points])
    doses = np.array([p.dose for p in dataset.points])
    cds = np.array([p.measured_cd for p in dataset.points])

    A = np.column_stack([np.ones_like(focuses), focuses, doses,
                         focuses * doses, focuses ** 2, doses ** 2])
    try:
        beta, *_ = np.linalg.lstsq(A, cds, rcond=None)
        predicted = A @ beta
        residuals = cds - predicted
    except np.linalg.LinAlgError:
        return dataset, {'removed': 0, 'details': []}

    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med)) * 1.4826
    if mad < 1e-9:
        return dataset, {'removed': 0, 'details': []}

    threshold = sigma * mad
    good_points: List[CDSEMDataPoint] = []
    details = []
    for p, r in zip(dataset.points, residuals):
        if abs(r - med) > threshold:
            details.append({
                'id': p.measurement_id,
                'reasons': [f"残差 {r:.2f} nm 超过 {sigma}σ (±{threshold:.2f} nm)"],
            })
        else:
            good_points.append(p)

    cleaned = CDSEMDataset(
        points=good_points,
        magnification=dataset.magnification,
        fab_name=dataset.fab_name,
        process_node=dataset.process_node,
        mask_set_id=dataset.mask_set_id,
        wafer_id=dataset.wafer_id,
        lot_id=dataset.lot_id,
        comments=f"{dataset.comments}; outlier removed (threshold={sigma}σ)",
    )
    return cleaned, {'removed': len(details), 'details': details}


def split_dataset(dataset: CDSEMDataset,
                  train_frac: float = 0.8,
                  random_seed: Optional[int] = 42,
                  stratify_by_pattern: bool = True,
                  ) -> Tuple[CDSEMDataset, CDSEMDataset]:
    """
    按比例随机划分训练集 / 测试集。

    Args:
        dataset: 完整数据集
        train_frac: 训练集比例 (0, 1)
        random_seed: 随机种子
        stratify_by_pattern: 是否按图形类型分层抽样

    Returns:
        (train_dataset, test_dataset)
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac 必须在 (0,1) 之间，实际: {train_frac}")

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(dataset))

    if stratify_by_pattern:
        pattern_types = dataset.pattern_types()
        unique_pt = list(set(pattern_types))
        train_idx = []
        test_idx = []
        for pt in unique_pt:
            sub = indices[np.array([p == pt for p in pattern_types])]
            rng.shuffle(sub)
            n_train = max(1, int(round(len(sub) * train_frac)))
            if n_train >= len(sub):
                n_train = len(sub) - 1 if len(sub) > 1 else 1
            train_idx.extend(sub[:n_train])
            test_idx.extend(sub[n_train:])
    else:
        rng.shuffle(indices)
        n_train = max(1, int(round(len(indices) * train_frac)))
        if n_train >= len(indices):
            n_train = len(indices) - 1 if len(indices) > 1 else 1
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

    def _build(idxs: List[int]) -> CDSEMDataset:
        return CDSEMDataset(
            points=[dataset.points[i] for i in sorted(idxs)],
            magnification=dataset.magnification,
            fab_name=dataset.fab_name,
            process_node=dataset.process_node,
            mask_set_id=dataset.mask_set_id,
            wafer_id=dataset.wafer_id,
            lot_id=dataset.lot_id,
            comments=dataset.comments,
        )

    return _build(train_idx), _build(test_idx)
