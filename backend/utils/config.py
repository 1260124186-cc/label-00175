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


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载配置文件
    
    支持YAML和JSON格式。
    
    Args:
        config_path: 配置文件路径
        
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
        'optical_system': {
            'wavelength': 193.0,
            'na': 1.35,
            'sigma': 0.75,
            'pixel_size': 1.0,
            'defocus': 0.0,
            'magnification': 4.0
        },
        'optimization': {
            'optimizer_type': 'gradient_descent',
            'max_iter': 100,
            'learning_rate': 0.01,
            'tol': 1e-6,
            'early_stop_patience': 10,
            'lr_scheduler': None,
            'metric': 'mse',
            'bounds': [0.0, 1.0],
            'random_seed': 42,  # 随机种子用于结果复现
            'population_size': 50,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1
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
    
    # 验证优化参数
    opt = config.get('optimization', {})
    if opt.get('max_iter', 0) <= 0:
        logger.error("最大迭代次数必须为正整数")
        return False
    
    if opt.get('learning_rate', 0) <= 0:
        logger.error("学习率必须为正数")
        return False
    
    logger.info("配置验证通过")
    return True
