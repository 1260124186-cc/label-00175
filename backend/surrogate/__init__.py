# -*- coding: utf-8 -*-
"""
神经网络代理模型（Surrogate Model）模块

使用深度神经网络近似掩模 → 空间像映射，
加速 OPC/SMO 中数万次前向仿真。

主要组件:
- dataset: 训练数据生成器，使用 PartialCoherentImaging 生成 (mask, aerial_image) 对
- model: U-Net / CNN encoder-decoder 模型架构
- train: PyTorch 训练流水线，含验证、模型保存加载
- imaging: SurrogateImaging 推理接口，与真实成像对比精度
"""

try:
    from surrogate.model import UNet, SurrogateModelConfig
    from surrogate.imaging import SurrogateImaging
    from surrogate.dataset import (
        SurrogateDataset,
        DatasetConfig,
        generate_training_data,
        save_dataset_hdf5,
        load_dataset_hdf5,
    )
    from surrogate.train import (
        TrainingConfig,
        train_surrogate_model,
        evaluate_surrogate_model,
        load_trained_model,
    )
except ImportError:
    from .model import UNet, SurrogateModelConfig
    from .imaging import SurrogateImaging
    from .dataset import (
        SurrogateDataset,
        DatasetConfig,
        generate_training_data,
        save_dataset_hdf5,
        load_dataset_hdf5,
    )
    from .train import (
        TrainingConfig,
        train_surrogate_model,
        evaluate_surrogate_model,
        load_trained_model,
    )

__all__ = [
    'UNet',
    'SurrogateModelConfig',
    'SurrogateImaging',
    'SurrogateDataset',
    'DatasetConfig',
    'generate_training_data',
    'save_dataset_hdf5',
    'load_dataset_hdf5',
    'TrainingConfig',
    'train_surrogate_model',
    'evaluate_surrogate_model',
    'load_trained_model',
]
