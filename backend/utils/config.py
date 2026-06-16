# -*- coding: utf-8 -*-
"""
配置模块：参数配置文件解析、结果保存

该模块提供YAML配置文件解析和结果保存功能。
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
import logging
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path], apply_device: bool = True) -> Dict[str, Any]:
    """
    加载配置文件

    支持YAML和JSON格式。如果配置中指定了 device，会自动设置计算后端。

    Args:
        config_path: 配置文件路径
        apply_device: 是否应用 device 配置设置计算后端

    Returns:
        配置字典
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    suffix = config_path.suffix.lower()

    with open(config_path, 'r', encoding='utf-8') as f:
        if suffix in ('.yaml', '.yml'):
            config = yaml.safe_load(f)
        elif suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {suffix}")

    logger.info(f"加载配置文件: {config_path}")

    if apply_device:
        system_config = config.get('system', {})
        device = system_config.get('device', 'cpu')
        try:
            from core.array_backend import set_backend
            set_backend(device)
            logger.info(f"已设置计算后端: {device}")
        except Exception as e:
            logger.warning(f"设置计算后端失败: {e}")

    return config


def save_config(config: Dict[str, Any],
                config_path: Union[str, Path],
                format: str = 'yaml') -> None:
    """
    保存配置文件

    Args:
        config: 配置字典
        config_path: 保存路径
        format: 文件格式 ('yaml' 或 'json')
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换numpy类型为Python原生类型
    config = _convert_numpy_types(config)

    with open(config_path, 'w', encoding='utf-8') as f:
        if format == 'yaml':
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif format == 'json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")

    logger.info(f"保存配置文件: {config_path}")


def _convert_numpy_types(obj: Any) -> Any:
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_results(results: Dict[str, Any],
                 output_dir: Union[str, Path],
                 prefix: str = 'result',
                 save_arrays: bool = True) -> Dict[str, str]:
    """
    保存优化结果

    Args:
        results: 结果字典，可包含numpy数组
        output_dir: 输出目录
        prefix: 文件名前缀
        save_arrays: 是否单独保存numpy数组为.npy文件

    Returns:
        保存的文件路径字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_files = {}

    # 分离数组和标量数据
    scalar_data = {}
    array_data = {}

    for key, value in results.items():
        if isinstance(value, np.ndarray):
            array_data[key] = value
        else:
            scalar_data[key] = value

    # 保存标量数据为JSON
    if scalar_data:
        json_path = output_dir / f'{prefix}_{timestamp}.json'
        scalar_data = _convert_numpy_types(scalar_data)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scalar_data, f, indent=2, ensure_ascii=False)

        saved_files['json'] = str(json_path)
        logger.info(f"保存标量结果: {json_path}")

    # 保存数组数据
    if save_arrays and array_data:
        for key, arr in array_data.items():
            npy_path = output_dir / f'{prefix}_{key}_{timestamp}.npy'
            np.save(npy_path, arr)
            saved_files[f'npy_{key}'] = str(npy_path)
            logger.info(f"保存数组 {key}: {npy_path}")

    # 保存CSV格式的历史数据（如果有）
    if 'loss_history' in results:
        csv_path = output_dir / f'{prefix}_history_{timestamp}.csv'
        history = results['loss_history']

        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('iteration,loss\n')
            for i, loss in enumerate(history):
                f.write(f'{i},{loss}\n')

        saved_files['csv'] = str(csv_path)
        logger.info(f"保存历史数据: {csv_path}")

    return saved_files


def load_results(result_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载保存的结果

    Args:
        result_path: 结果文件路径（JSON或NPY）

    Returns:
        结果字典
    """
    result_path = Path(result_path)

    if not result_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {result_path}")

    suffix = result_path.suffix.lower()

    if suffix == '.json':
        with open(result_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif suffix == '.npy':
        return {'data': np.load(result_path)}
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置

    Returns:
        默认配置字典
    """
    return {
        'system': {
            'device': 'cpu',
        },
        'optical_system': {
            'wavelength': 193.0,
            'na': 1.35,
            'sigma': 0.75,
            'pixel_size': 1.0,
            'defocus': 0.0,
            'magnification': 4.0,
            'illumination_type': 'conventional',
            'source_params': {
                'sigma_inner': 0.0,
                'sigma_outer': 0.75
            },
            'use_socs': True,
            'socs_num_terms': 5,
            'tcc_mode': 'socs'
        },
        'optimization': {
            'optimizer_type': 'gradient_descent',
            'max_iter': 100,
            'learning_rate': 0.01,
            'tol': 1e-6,
            'early_stop_patience': 10,
            'lr_scheduler': None,
            'metric': 'mse',
            'use_composite_loss': False,
            'loss_weights': {
                'mse': 1.0,
                'ssim': 0.0,
                'pvb': 0.0,
                'mask_complexity': 0.0
            },
            'regularization': {
                'type': None,
                'strength': 0.0
            },
            'bounds': [0.0, 1.0],
            'random_seed': 42,  # 随机种子用于结果复现
            'population_size': 50,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'n_jobs': 1
        },
        'output': {
            'save_dir': './results',
            'save_images': True,
            'save_history': True
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置有效性

    Args:
        config: 配置字典

    Returns:
        是否有效
    """
    required_keys = ['optical_system', 'optimization']

    for key in required_keys:
        if key not in config:
            logger.error(f"配置缺少必要字段: {key}")
            return False

    # 验证系统配置（device）
    system_config = config.get('system', {})
    device = system_config.get('device', 'cpu')
    valid_devices = ['cpu', 'cuda']
    if device not in valid_devices:
        logger.error(f"device 必须为以下之一: {valid_devices}")
        return False

    # 验证光学系统参数
    optics = config.get('optical_system', {})
    if optics.get('wavelength', 0) <= 0:
        logger.error("波长必须为正数")
        return False

    if not 0 < optics.get('na', 0) <= 2:
        logger.error("数值孔径NA必须在(0, 2]范围内")
        return False

    if not 0 <= optics.get('sigma', 0) <= 1:
        logger.error("部分相干因子sigma必须在[0, 1]范围内")
        return False

    # 验证照明模式
    valid_illumination_types = ['conventional', 'annular', 'dipole', 'quasar', 'custom']
    illu_type = optics.get('illumination_type', 'conventional')
    if illu_type not in valid_illumination_types:
        logger.error(f"照明模式必须为以下之一: {valid_illumination_types}")
        return False

    # 验证SOCS参数
    if optics.get('socs_num_terms', 5) <= 0:
        logger.error("SOCS分解项数必须为正整数")
        return False

    # 验证 TCC 模式
    valid_tcc_modes = ['full_tcc', 'socs', 'kernel_2d']
    tcc_mode = optics.get('tcc_mode', None)
    if tcc_mode is not None and tcc_mode not in valid_tcc_modes:
        logger.error(f"tcc_mode 必须为以下之一: {valid_tcc_modes}")
        return False

    # 验证光源参数
    source_params = optics.get('source_params', {})
    sigma_inner = source_params.get('sigma_inner', 0.0)
    sigma_outer = source_params.get('sigma_outer', optics.get('sigma', 0.75))

    if not 0 <= sigma_inner <= 1:
        logger.error("sigma_inner必须在[0, 1]范围内")
        return False

    if not 0 < sigma_outer <= 1:
        logger.error("sigma_outer必须在(0, 1]范围内")
        return False

    if sigma_inner >= sigma_outer:
        logger.error("sigma_inner必须小于sigma_outer")
        return False

    # 验证dipole/quasar特有参数
    if illu_type in ['dipole', 'quasar']:
        opening_angle = source_params.get('opening_angle', 60.0)
        if not 0 < opening_angle <= 180:
            logger.error("opening_angle必须在(0, 180]度范围内")
            return False

    # 验证Zernike像差系数
    zernike = optics.get('zernike_coefficients', {})
    if zernike:
        valid_names = {
            'piston', 'tilt_x', 'tilt_y', 'defocus',
            'astigmatism_x', 'astigmatism_y', 'coma_x', 'coma_y',
            'trefoil_x', 'trefoil_y', 'spherical',
            'secondary_astigmatism_x', 'secondary_astigmatism_y',
            'secondary_coma_x', 'secondary_coma_y', 'secondary_spherical'
        }
        for key, value in zernike.items():
            if isinstance(key, str) and not key.isdigit() and key not in valid_names:
                logger.error(f"未知的Zernike像差名称: {key}")
                return False
            try:
                float(value)
            except (ValueError, TypeError):
                logger.error(f"Zernike系数值必须为数值: {key}={value}")
                return False

    # 验证优化参数
    opt = config.get('optimization', {})
    if opt.get('max_iter', 0) <= 0:
        logger.error("最大迭代次数必须为正整数")
        return False

    if opt.get('learning_rate', 0) <= 0:
        logger.error("学习率必须为正数")
        return False

    # 验证复合损失权重
    loss_weights = opt.get('loss_weights', {})
    if isinstance(loss_weights, dict):
        valid_loss_keys = {
            'mse', 'ssim', 'pvb', 'mask_complexity',
            'weighted_mse', 'weighted_mae'
        }
        for key in loss_weights:
            if key not in valid_loss_keys:
                logger.error(f"未知的损失权重键: {key}，有效键: {valid_loss_keys}")
                return False
            try:
                float(loss_weights[key])
            except (ValueError, TypeError):
                logger.error(f"损失权重值必须为数值: {key}={loss_weights[key]}")
                return False

    # 验证正则化配置
    regularization = opt.get('regularization', {})
    if isinstance(regularization, dict):
        reg_type = regularization.get('type', None)
        valid_reg_types = {None, 'l1', 'l2', 'tv', 'none', 'None'}
        if reg_type is not None and reg_type not in valid_reg_types:
            logger.error(f"未知的正则化类型: {reg_type}，有效类型: None, 'l1', 'l2', 'tv'")
            return False
        if 'strength' in regularization:
            try:
                float(regularization['strength'])
            except (ValueError, TypeError):
                logger.error(f"正则化强度必须为数值: {regularization['strength']}")
                return False

    logger.info("配置验证通过")
    return True
