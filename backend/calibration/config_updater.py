# -*- coding: utf-8 -*-
"""
default_config.yaml 片段生成器

根据标定结果自动生成 / 更新 default_config.yaml 中对应的参数字段：
- optical_system.na          ← na_effective
- optical_system.sigma       ← sigma_effective
- optical_system.wavelength  ← wavelength_effective
- imaging.resist_threshold   ← resist_threshold
- imaging.diffusion_length   ← diffusion_length  (若不存在则新增)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from copy import deepcopy

import yaml

from .schemas import (
    CalibrationParameterSet,
    CalibrationParameter,
    InversionResult,
    CalibrationConfig,
    CalibrationReport,
)
from utils.config import _convert_numpy_types

logger = logging.getLogger(__name__)


# 参数名 → YAML 配置路径（点号分隔，如 'optical_system.na'）
# 如果 CalibrationParameter.config_path 已设置，优先使用它
_DEFAULT_CONFIG_PATHS: Dict[str, str] = {
    'na_effective': 'optical_system.na',
    'sigma_effective': 'optical_system.sigma',
    'wavelength_effective': 'optical_system.wavelength',
    'resist_threshold': 'imaging.resist_threshold',
    'diffusion_length': 'imaging.diffusion_length',
    'dose_to_clear': 'imaging.dose_to_clear',
    'resist_contrast': 'imaging.resist_contrast',
}


def _resolve_path(param_name: str,
                  param_obj: Optional[CalibrationParameter]) -> Optional[str]:
    """解析参数对应的 YAML 路径。"""
    if param_obj and param_obj.config_path:
        return param_obj.config_path
    return _DEFAULT_CONFIG_PATHS.get(param_name)


def _set_nested(d: Dict[str, Any], path: str, value: Any) -> None:
    """按路径设置嵌套字典。"""
    parts = path.split('.')
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _get_nested(d: Dict[str, Any], path: str, default=None):
    """按路径读取嵌套字典。"""
    parts = path.split('.')
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def build_config_snippet(inversion_result: InversionResult,
                         parameter_set: CalibrationParameterSet,
                         include_uncertainties: bool = True,
                         include_comments: bool = True,
                         ) -> Dict[str, Any]:
    """
    根据反演结果构建 default_config.yaml 片段（嵌套字典结构）。

    仅处理 config_path 不为空的参数。

    Args:
        inversion_result: 反演结果
        parameter_set: 参数集合（用于读取 config_path）
        include_uncertainties: 是否在输出字典中额外加入 _uncertainties 段
        include_comments: 是否加入 _calibration_meta 元信息段

    Returns:
        可直接 yaml.dump 的嵌套字典
    """
    snippet: Dict[str, Any] = {}
    param_objs = {p.name: p for p in parameter_set.all_parameters()}

    for name, value in inversion_result.calibrated_values.items():
        pobj = param_objs.get(name)
        path = _resolve_path(name, pobj)
        if path is None:
            continue
        # 保留合理精度（浮点参数 6 位足够）
        rounded = float(np.round(float(value), 6))
        _set_nested(snippet, path, rounded)

    if include_uncertainties:
        unc_section: Dict[str, Any] = {}
        for name, unc in inversion_result.uncertainties.items():
            pobj = param_objs.get(name)
            path = _resolve_path(name, pobj)
            if path is None:
                continue
            unc_path = f"_calibration_uncertainties.{path}"
            rounded = float(np.round(float(unc), 6))
            _set_nested(unc_section, unc_path, rounded)
        if unc_section.get('_calibration_uncertainties'):
            snippet['_calibration_uncertainties'] = unc_section['_calibration_uncertainties']

    if include_comments:
        snippet['_calibration_meta'] = {
            'method': inversion_result.method.value,
            'success': bool(inversion_result.success),
            'message': inversion_result.message,
            'n_data_points': inversion_result.n_data,
            'n_free_params': inversion_result.n_params,
            'degrees_of_freedom': inversion_result.dof,
            'final_cost': float(np.round(inversion_result.cost, 6)),
            'chi_squared': float(np.round(inversion_result.chi2, 6)),
            'reduced_chi_squared': float(np.round(inversion_result.reduced_chi2, 6)),
            'varying_parameters': list(inversion_result.varying_names),
            'timestamp': None,  # 会在 save 时填充
        }
        # 相关系数矩阵（只保留变化参数）
        if inversion_result.correlation_matrix is not None \
                and len(inversion_result.varying_names) > 1:
            corr = inversion_result.correlation_matrix
            names = inversion_result.varying_names
            corr_dict = {}
            for i, ni in enumerate(names):
                corr_dict[ni] = {
                    nj: float(np.round(corr[i, j], 4))
                    for j, nj in enumerate(names)
                }
            snippet['_calibration_meta']['correlation_matrix'] = corr_dict

    return _convert_numpy_types(snippet)


def update_default_config(inversion_result: InversionResult,
                          parameter_set: CalibrationParameterSet,
                          reference_config: Union[Dict[str, Any], str, Path, None] = None,
                          ) -> Dict[str, Any]:
    """
    用标定结果更新一份完整的配置字典。

    Args:
        inversion_result: 反演结果
        parameter_set: 参数集合
        reference_config: 参考配置（dict 或 YAML 文件路径）；None 时返回仅含被更新字段的最小配置

    Returns:
        更新后的完整配置字典
    """
    # 参考配置加载
    if reference_config is None:
        base: Dict[str, Any] = {}
    elif isinstance(reference_config, dict):
        base = deepcopy(reference_config)
    else:
        path = Path(reference_config)
        if not path.exists():
            raise FileNotFoundError(f"参考配置文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            base = yaml.safe_load(f) or {}

    # 应用标定更新
    snippet = build_config_snippet(
        inversion_result, parameter_set,
        include_uncertainties=False,
        include_comments=False,
    )

    def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
                _merge(dst[k], v)
            else:
                dst[k] = v

    _merge(base, snippet)

    # 在配置末尾加入校准元信息
    meta = build_config_snippet(
        inversion_result, parameter_set,
        include_uncertainties=True,
        include_comments=True,
    ).get('_calibration_meta', {})
    unc = build_config_snippet(
        inversion_result, parameter_set,
        include_uncertainties=True,
        include_comments=False,
    ).get('_calibration_uncertainties', {})
    if meta:
        base['_calibration_meta'] = meta
    if unc:
        base['_calibration_uncertainties'] = unc

    return _convert_numpy_types(base)


def save_config_snippet(inversion_result: InversionResult,
                        parameter_set: CalibrationParameterSet,
                        output_dir: Union[str, Path],
                        filename: str = "calibrated_config_snippet.yaml",
                        reference_config_path: Optional[Union[str, Path]] = None,
                        update_full_config: bool = True,
                        full_config_filename: str = "calibrated_default_config.yaml",
                        ) -> Dict[str, str]:
    """
    将标定结果保存为 YAML 片段与更新后的完整配置。

    Returns:
        {'snippet': 片段路径, 'full_config': 完整配置路径（如生成）}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # 1) 仅含被更新字段的最小片段
    snippet = build_config_snippet(inversion_result, parameter_set)
    snippet_path = output_dir / filename
    with open(snippet_path, 'w', encoding='utf-8') as f:
        f.write("# =====================================================\n")
        f.write("# Fab 模型标定结果：default_config.yaml 更新片段\n")
        f.write("# Calibrated YAML Snippet for default_config.yaml\n")
        f.write("#\n")
        f.write(f"# 方法: {inversion_result.method.value}\n")
        f.write(f"# 状态: {'成功' if inversion_result.success else '失败'}\n")
        f.write(f"# χ²/dof: {inversion_result.reduced_chi2:.4f}\n")
        f.write("#\n")
        f.write("# 使用方法：\n")
        f.write("#   1. 将本文件中 optical_system / imaging 段\n")
        f.write("#      直接合并到 default_config.yaml 对应段\n")
        f.write("#   2. 或使用 update_default_config() 自动合并\n")
        f.write("# =====================================================\n\n")
        yaml.dump(snippet, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)
    paths['snippet'] = str(snippet_path)
    logger.info(f"标定配置片段已写入: {snippet_path}")

    # 2) 完整配置（如果提供了参考）
    if update_full_config and reference_config_path is not None:
        ref = Path(reference_config_path)
        if ref.exists():
            full = update_default_config(
                inversion_result, parameter_set, reference_config=ref
            )
            # 补充时间戳
            if '_calibration_meta' in full:
                from datetime import datetime
                full['_calibration_meta']['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            full_path = output_dir / full_config_filename
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write("# =====================================================\n")
                f.write("# Calibrated Default Configuration\n")
                f.write(f"# 参考配置来源: {ref.name}\n")
                f.write("# =====================================================\n\n")
                yaml.dump(_convert_numpy_types(full), f,
                          default_flow_style=False, allow_unicode=True,
                          sort_keys=False)
            paths['full_config'] = str(full_path)
            logger.info(f"完整标定配置已写入: {full_path}")
        else:
            logger.warning(f"参考配置不存在，跳过完整配置生成: {ref}")

    return paths


class ConfigUpdater:
    """
    面向对象的配置更新器。

    典型用法::

        updater = ConfigUpdater(config, parameter_set)
        updater.apply(inversion_result)
        paths = updater.save(output_dir)
    """

    def __init__(self,
                 reference_config: Optional[Union[Dict[str, Any], str, Path]] = None,
                 parameter_set: Optional[CalibrationParameterSet] = None):
        self.reference_config = reference_config
        self.parameter_set = parameter_set or CalibrationParameterSet()
        self.current_snippet: Optional[Dict[str, Any]] = None
        self.current_full: Optional[Dict[str, Any]] = None

    def apply(self, inversion_result: InversionResult) -> Dict[str, Any]:
        """应用反演结果，返回更新后的完整配置。"""
        self.current_snippet = build_config_snippet(
            inversion_result, self.parameter_set
        )
        self.current_full = update_default_config(
            inversion_result, self.parameter_set, self.reference_config
        )
        return self.current_full

    def save(self,
             output_dir: Union[str, Path],
             snippet_filename: str = "calibrated_config_snippet.yaml",
             full_filename: str = "calibrated_default_config.yaml",
             ) -> Dict[str, str]:
        if self.current_snippet is None:
            raise RuntimeError("请先调用 apply(inversion_result)")
        return save_config_snippet(
            inversion_result=None,  # not used if we have snippet — actually used below
            parameter_set=self.parameter_set,
            output_dir=output_dir,
            filename=snippet_filename,
            reference_config_path=(self.reference_config
                                   if isinstance(self.reference_config, (str, Path))
                                   else None),
            update_full_config=True,
            full_config_filename=full_filename,
        ) if False else self._manual_save(output_dir, snippet_filename, full_filename)

    def _manual_save(self, output_dir, snippet_filename, full_filename):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        snippet_path = output_dir / snippet_filename
        with open(snippet_path, 'w', encoding='utf-8') as f:
            f.write("# =====================================================\n")
            f.write("# Fab 模型标定结果：default_config.yaml 更新片段\n")
            f.write("# =====================================================\n\n")
            yaml.dump(self.current_snippet, f,
                      default_flow_style=False, allow_unicode=True,
                      sort_keys=False)
        paths['snippet'] = str(snippet_path)

        if self.current_full is not None:
            full_path = output_dir / full_filename
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write("# =====================================================\n")
                f.write("# Calibrated Default Configuration\n")
                f.write("# =====================================================\n\n")
                yaml.dump(_convert_numpy_types(self.current_full), f,
                          default_flow_style=False, allow_unicode=True,
                          sort_keys=False)
            paths['full_config'] = str(full_path)
        return paths
