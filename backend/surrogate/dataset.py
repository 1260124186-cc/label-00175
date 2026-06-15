# -*- coding: utf-8 -*-
"""
代理模型训练数据生成器

使用现有 PartialCoherentImaging 批量生成 (mask, aerial_image) 训练对，
覆盖多种测试图案与光学参数（离焦、NA、sigma、波长、像差、照明模式等）。

主要功能:
- generate_training_data: 生成 Numpy 数组形式的数据集
- save_dataset_hdf5 / load_dataset_hdf5: HDF5 格式读写
- SurrogateDataset: PyTorch Dataset 封装，支持 DataLoader
"""

import os
import sys
import logging
import numpy as np
import h5py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from itertools import product
import random
import time

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.imaging import (
    OpticalSystem,
    PartialCoherentImaging,
    IlluminationType,
    ProcessCondition,
)
from core.test_structures import (
    TestStructureType,
    LineOrientation,
    HolePattern,
    LineSpaceParams,
    ContactHoleParams,
    LShapedCornerParams,
    TJunctionParams,
    SRAMBitcellParams,
    generate_test_structure,
)


@dataclass
class DatasetConfig:
    """
    训练数据集生成配置

    Attributes:
        grid_size: 掩模/空间像尺寸 (ny, nx)
        num_samples: 总样本数（每个样本 = 1个掩模 + 1组光学参数 + 1个空间像）
        structure_types: 包含的测试结构类型列表
        optical_param_sweep: 是否对光学参数做随机扫描（True）或只使用标称值（False）
        defocus_range: 离焦量范围 (nm)，(min, max)
        na_range: NA 范围，(min, max)
        sigma_range: sigma 范围，(min, max)
        wavelength_range: 波长范围 (nm)，(min, max)
        illumination_types: 包含的照明模式列表
        include_aberrations: 是否包含随机 Zernike 像差
        aberration_max_strength: 像差最大强度（单位：波长 λ）
        seed: 随机种子
        train_val_split: 训练集比例 (0-1)，剩余为验证集
        pixel_size: 像素尺寸 (nm)
        cd_range: 关键尺寸 CD 范围 (nm)，(min, max)
        pitch_range: pitch 范围 (nm)，(min, max)
        random_structure_noise: 是否给掩模添加随机扰动（模拟优化过程中的连续值掩模）
        noise_level: 扰动强度（最大偏离 0/1 的幅度）
        noise_probability: 添加噪声的样本比例
    """
    grid_size: Tuple[int, int] = (128, 128)
    num_samples: int = 5000
    structure_types: List[str] = field(default_factory=lambda: [
        'line_space', 'contact_hole', 'l_shaped_corner',
        't_junction', 'sram_bitcell'
    ])
    optical_param_sweep: bool = True
    defocus_range: Tuple[float, float] = (-100.0, 100.0)
    na_range: Tuple[float, float] = (1.20, 1.35)
    sigma_range: Tuple[float, float] = (0.5, 0.9)
    wavelength_range: Tuple[float, float] = (193.0, 193.0)
    illumination_types: List[str] = field(default_factory=lambda: [
        'conventional', 'annular', 'dipole', 'quasar'
    ])
    include_aberrations: bool = True
    aberration_max_strength: float = 0.05
    seed: Optional[int] = 42
    train_val_split: float = 0.8
    pixel_size: float = 1.0
    cd_range: Tuple[float, float] = (32.0, 90.0)
    pitch_range: Tuple[float, float] = (80.0, 200.0)
    random_structure_noise: bool = True
    noise_level: float = 0.3
    noise_probability: float = 0.5

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            elif isinstance(v, list):
                d[k] = v
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'DatasetConfig':
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                if isinstance(getattr(cfg, k), tuple) and isinstance(v, list):
                    setattr(cfg, k, tuple(v))
                else:
                    setattr(cfg, k, v)
        return cfg


def _random_structure_params(
    rng: np.random.Generator,
    cfg: DatasetConfig,
) -> Dict[str, Any]:
    """随机采样一组测试结构参数"""
    structure_type = rng.choice(cfg.structure_types)
    cd = float(rng.uniform(*cfg.cd_range))
    pitch = float(rng.uniform(
        max(cfg.pitch_range[0], cd * 1.5),
        cfg.pitch_range[1]
    ))
    corner_rounding = float(rng.uniform(0.0, 5.0))

    params_base = {
        'structure_type': structure_type,
        'grid_size': list(cfg.grid_size),
        'pixel_size': cfg.pixel_size,
        'cd': cd,
        'pitch': pitch,
        'corner_rounding': corner_rounding,
    }

    if structure_type == 'line_space':
        orientation = rng.choice(['horizontal', 'vertical'])
        params_base['orientation'] = orientation
        params_base['duty_cycle'] = float(rng.uniform(0.5, 1.5))
        if rng.random() < 0.3:
            params_base['num_lines'] = int(rng.integers(3, 15))

    elif structure_type == 'contact_hole':
        pattern = rng.choice(['square_grid', 'hexagonal'])
        params_base['pattern'] = pattern
        params_base['hole_shape'] = rng.choice(['circle', 'square'])
        params_base['aspect_ratio'] = float(rng.uniform(0.8, 1.2))
        params_base['rotation'] = float(rng.uniform(0.0, 90.0))

    elif structure_type == 'l_shaped_corner':
        params_base['arm_length'] = float(rng.uniform(cd * 2, cd * 8))
        params_base['corner_type'] = rng.choice(['inner', 'outer'])

    elif structure_type == 't_junction':
        params_base['stem_length'] = float(rng.uniform(cd * 3, cd * 10))
        params_base['branch_length'] = float(rng.uniform(cd * 2, cd * 6))

    elif structure_type == 'sram_bitcell':
        params_base['bitcell_type'] = rng.choice(['6T', 'thin-film'])
        params_base['metal_layer'] = int(rng.integers(1, 4))

    return params_base


def _random_optical_system(
    rng: np.random.Generator,
    cfg: DatasetConfig,
) -> OpticalSystem:
    """随机采样一组光学系统参数"""
    if cfg.optical_param_sweep:
        defocus = float(rng.uniform(*cfg.defocus_range))
        na = float(rng.uniform(*cfg.na_range))
        sigma = float(rng.uniform(*cfg.sigma_range))
        wavelength = float(rng.uniform(*cfg.wavelength_range))
        illum_type_str = rng.choice(cfg.illumination_types)
    else:
        defocus = 0.0
        na = 1.35
        sigma = 0.75
        wavelength = 193.0
        illum_type_str = 'conventional'

    try:
        illum_type = IlluminationType(illum_type_str)
    except ValueError:
        illum_type = IlluminationType.CONVENTIONAL

    source_params = {}
    if illum_type == IlluminationType.DIPOLE:
        source_params = {
            'sigma_inner': float(rng.uniform(0.4, 0.7)),
            'sigma_outer': float(rng.uniform(0.7, 0.95)),
            'angle': float(rng.uniform(0.0, 180.0)),
            'opening_angle': float(rng.uniform(30.0, 90.0)),
        }
    elif illum_type == IlluminationType.QUASAR:
        source_params = {
            'sigma_inner': float(rng.uniform(0.4, 0.7)),
            'sigma_outer': float(rng.uniform(0.7, 0.95)),
            'angle': float(rng.uniform(0.0, 90.0)),
            'opening_angle': float(rng.uniform(20.0, 60.0)),
        }
    elif illum_type == IlluminationType.ANNULAR:
        source_params = {
            'sigma_inner': float(rng.uniform(0.5, 0.75)),
            'sigma_outer': float(rng.uniform(0.75, 0.95)),
        }

    zernike_coefficients: Dict[int, float] = {}
    if cfg.include_aberrations and rng.random() < 0.7:
        num_terms = int(rng.integers(1, 5))
        zernike_indices = rng.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            size=num_terms, replace=False
        )
        for j in zernike_indices:
            strength = float(rng.uniform(
                -cfg.aberration_max_strength,
                cfg.aberration_max_strength
            ))
            if abs(strength) > 1e-4:
                zernike_coefficients[int(j)] = strength

    return OpticalSystem(
        wavelength=wavelength,
        na=na,
        sigma=sigma,
        pixel_size=cfg.pixel_size,
        defocus=defocus,
        illumination_type=illum_type,
        source_params=source_params,
        zernike_coefficients=zernike_coefficients,
    )


def _apply_mask_noise(
    mask: np.ndarray,
    rng: np.random.Generator,
    cfg: DatasetConfig,
) -> np.ndarray:
    """对二值掩模添加平滑的随机扰动，模拟优化中的连续值掩模"""
    if not cfg.random_structure_noise:
        return mask
    if rng.random() > cfg.noise_probability:
        return mask

    from scipy.ndimage import gaussian_filter

    noise = rng.standard_normal(mask.shape)
    sigma = float(rng.uniform(0.5, 3.0))
    smoothed_noise = gaussian_filter(noise, sigma=sigma)

    max_amp = cfg.noise_level
    smoothed_noise = smoothed_noise / (np.abs(smoothed_noise).max() + 1e-12) * max_amp
    result = mask + smoothed_noise
    return np.clip(result, 0.0, 1.0)


def generate_training_data(
    config: Optional[DatasetConfig] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], DatasetConfig]:
    """
    生成训练数据集 (masks, aerial_images, metadata)

    Args:
        config: 数据集配置，None 则使用默认
        verbose: 是否打印进度

    Returns:
        (masks, aerial_images, metadata_list, config)
        - masks: (N, H, W) float32 数组，掩模
        - aerial_images: (N, H, W) float32 数组，空间像
        - metadata_list: 每个样本的参数字典
        - config: 实际使用的配置
    """
    cfg = config or DatasetConfig()
    if cfg.seed is not None:
        rng = np.random.default_rng(cfg.seed)
    else:
        rng = np.random.default_rng()

    ny, nx = cfg.grid_size
    N = cfg.num_samples

    masks = np.zeros((N, ny, nx), dtype=np.float32)
    aerial_images = np.zeros((N, ny, nx), dtype=np.float32)
    metadata: List[Dict[str, Any]] = []

    t0 = time.time()
    log_interval = max(1, N // 20)

    for i in range(N):
        structure_params = _random_structure_params(rng, cfg)

        try:
            mask = generate_test_structure(structure_params)
        except Exception as e:
            logger.warning(f"样本 {i} 生成结构失败: {e}，回退到 LineSpace")
            fallback = LineSpaceParams(
                grid_size=cfg.grid_size,
                pixel_size=cfg.pixel_size,
                cd=45.0, pitch=90.0,
            )
            mask = generate_test_structure(fallback)
            structure_params = fallback.to_dict()

        mask = _apply_mask_noise(mask, rng, cfg)

        optics = _random_optical_system(rng, cfg)

        try:
            imaging = PartialCoherentImaging(optics, cfg.grid_size)
            aerial = imaging.compute_aerial_image(mask)
        except Exception as e:
            logger.error(f"样本 {i} 成像计算失败: {e}，使用零图像")
            aerial = np.zeros(cfg.grid_size, dtype=np.float64)

        masks[i] = mask.astype(np.float32)
        aerial_images[i] = aerial.astype(np.float32)

        meta = {
            'sample_idx': i,
            'structure_params': structure_params,
            'optical_params': optics.to_dict(),
            'structure_type': structure_params.get('structure_type', 'unknown'),
        }
        metadata.append(meta)

        if verbose and (i + 1) % log_interval == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            logger.info(
                f"生成进度: {i + 1}/{N} ({100 * (i + 1) / N:.1f}%) "
                f"耗时 {elapsed:.1f}s, 速度 {rate:.1f} 样本/s, 预计剩余 {eta:.1f}s"
            )

    elapsed = time.time() - t0
    if verbose:
        logger.info(
            f"数据集生成完成: {N} 个样本, 总耗时 {elapsed:.1f}s, "
            f"平均 {elapsed / N * 1000:.1f} ms/样本"
        )
        aerial_min, aerial_max = aerial_images.min(), aerial_images.max()
        logger.info(f"空间像数值范围: [{aerial_min:.4f}, {aerial_max:.4f}]")

    return masks, aerial_images, metadata, cfg


def save_dataset_hdf5(
    filepath: str,
    masks: np.ndarray,
    aerial_images: np.ndarray,
    metadata: Optional[List[Dict[str, Any]]] = None,
    config: Optional[DatasetConfig] = None,
    train_indices: Optional[np.ndarray] = None,
    val_indices: Optional[np.ndarray] = None,
    compression: str = 'gzip',
):
    """
    将数据集保存为 HDF5 格式

    Args:
        filepath: 输出文件路径
        masks: (N, H, W) 掩模数组
        aerial_images: (N, H, W) 空间像数组
        metadata: 样本元数据列表
        config: 数据集配置
        train_indices: 训练集索引
        val_indices: 验证集索引
        compression: HDF5 压缩方式
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or '.', exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('masks', data=masks, compression=compression)
        f.create_dataset('aerial_images', data=aerial_images, compression=compression)

        if metadata is not None:
            import json
            meta_json = json.dumps(metadata, ensure_ascii=False, default=str)
            f.create_dataset('metadata', data=np.bytes_(meta_json))

        if config is not None:
            import json
            cfg_json = json.dumps(config.to_dict(), ensure_ascii=False)
            f.create_dataset('config', data=np.bytes_(cfg_json))

        if train_indices is not None:
            f.create_dataset('train_indices', data=np.asarray(train_indices, dtype=np.int64))
        if val_indices is not None:
            f.create_dataset('val_indices', data=np.asarray(val_indices, dtype=np.int64))

        f.attrs['num_samples'] = len(masks)
        f.attrs['grid_size_h'] = masks.shape[1]
        f.attrs['grid_size_w'] = masks.shape[2]
        f.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    logger.info(f"数据集已保存到 {filepath} ({os.path.getsize(filepath) / 1e6:.2f} MB)")


def load_dataset_hdf5(
    filepath: str,
    load_metadata: bool = True,
    load_config: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict[str, Any]]], Optional[DatasetConfig],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从 HDF5 文件加载数据集

    Returns:
        (masks, aerial_images, metadata, config, train_indices, val_indices)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据集文件不存在: {filepath}")

    with h5py.File(filepath, 'r') as f:
        masks = f['masks'][:].astype(np.float32)
        aerial_images = f['aerial_images'][:].astype(np.float32)

        metadata = None
        if load_metadata and 'metadata' in f:
            import json
            meta_json = bytes(f['metadata'][()]).decode('utf-8')
            metadata = json.loads(meta_json)

        config = None
        if load_config and 'config' in f:
            import json
            cfg_json = bytes(f['config'][()]).decode('utf-8')
            config = DatasetConfig.from_dict(json.loads(cfg_json))

        train_indices = None
        if 'train_indices' in f:
            train_indices = f['train_indices'][:]
        val_indices = None
        if 'val_indices' in f:
            val_indices = f['val_indices'][:]

    logger.info(
        f"从 {filepath} 加载数据集: {len(masks)} 个样本, "
        f"尺寸 {masks.shape[1]}x{masks.shape[2]}"
    )
    return masks, aerial_images, metadata, config, train_indices, val_indices


def split_train_val(
    num_samples: int,
    train_ratio: float = 0.8,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """划分训练集/验证集索引"""
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    split = int(num_samples * train_ratio)
    return indices[:split], indices[split:]


try:
    import torch
    from torch.utils.data import Dataset as TorchDataset

    class SurrogateDataset(TorchDataset):
        """
        PyTorch Dataset 封装

        用法:
            dataset = SurrogateDataset(masks, aerial_images)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
        """

        def __init__(
            self,
            masks: np.ndarray,
            aerial_images: np.ndarray,
            transform=None,
            add_channel_dim: bool = True,
        ):
            """
            Args:
                masks: (N, H, W) 掩模数组
                aerial_images: (N, H, W) 空间像数组
                transform: 可选的 torchvision 变换
                add_channel_dim: 是否在第1维插入通道维度，变为 (N, 1, H, W)
            """
            if len(masks) != len(aerial_images):
                raise ValueError(
                    f"masks({len(masks)}) 和 aerial_images({len(aerial_images)}) 数量不一致"
                )

            self.masks = masks.astype(np.float32)
            self.aerial_images = aerial_images.astype(np.float32)
            self.transform = transform
            self.add_channel_dim = add_channel_dim

        def __len__(self) -> int:
            return len(self.masks)

        def __getitem__(self, idx: int):
            mask = self.masks[idx]
            aerial = self.aerial_images[idx]

            if self.add_channel_dim:
                mask = mask[np.newaxis, ...]
                aerial = aerial[np.newaxis, ...]

            mask_t = torch.from_numpy(mask)
            aerial_t = torch.from_numpy(aerial)

            if self.transform is not None:
                cat = torch.cat([mask_t, aerial_t], dim=0)
                cat = self.transform(cat)
                mask_t = cat[0:1]
                aerial_t = cat[1:2]

            return mask_t, aerial_t

        @classmethod
        def from_hdf5(cls, filepath: str, split: str = 'train', **kwargs):
            """从 HDF5 文件直接构建 Dataset"""
            masks, aerials, _, _, train_idx, val_idx = load_dataset_hdf5(filepath)

            if split == 'train':
                if train_idx is None:
                    train_idx, _ = split_train_val(len(masks))
                idx = train_idx
            elif split == 'val' or split == 'valid' or split == 'validation':
                if val_idx is None:
                    _, val_idx = split_train_val(len(masks))
                idx = val_idx
            elif split == 'all':
                idx = np.arange(len(masks))
            else:
                raise ValueError(f"未知 split: {split}，支持 train/val/all")

            return cls(masks[idx], aerials[idx], **kwargs)

except ImportError:
    class SurrogateDataset:
        """PyTorch 未安装时的占位类"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch 未安装，无法使用 SurrogateDataset。"
                "请安装 torch: pip install torch"
            )
