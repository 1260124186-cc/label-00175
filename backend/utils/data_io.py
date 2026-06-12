# -*- coding: utf-8 -*-
"""
数据处理模块：图像读取、保存、格式转换

该模块提供掩模/目标图像的读取、归一化、像素格式转换等功能。
"""

import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 尝试导入图像处理库
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("opencv-python未安装，部分图像处理功能不可用")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_image(filepath: Union[str, Path],
               grayscale: bool = True,
               normalize: bool = True,
               target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    加载图像文件
    
    支持格式：png, tiff, jpg, bmp等常见格式
    
    Args:
        filepath: 图像文件路径
        grayscale: 是否转换为灰度图
        normalize: 是否归一化到[0, 1]
        target_size: 目标尺寸 (height, width)，None则保持原尺寸
        
    Returns:
        图像数组
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"图像文件不存在: {filepath}")
    
    # 使用OpenCV加载
    if HAS_CV2:
        if grayscale:
            image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise ValueError(f"无法读取图像: {filepath}")
        
        # 调整尺寸
        if target_size is not None:
            image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # 使用PIL加载
    elif HAS_PIL:
        pil_image = Image.open(filepath)
        
        if grayscale:
            pil_image = pil_image.convert('L')
        else:
            pil_image = pil_image.convert('RGB')
        
        if target_size is not None:
            pil_image = pil_image.resize((target_size[1], target_size[0]))
        
        image = np.array(pil_image)
    
    else:
        raise ImportError("需要安装opencv-python或Pillow来加载图像")
    
    # 归一化
    if normalize:
        image = normalize_image(image)
    
    logger.debug(f"加载图像: {filepath}, 尺寸: {image.shape}")
    
    return image.astype(np.float64)


def save_image(image: np.ndarray,
               filepath: Union[str, Path],
               normalize_output: bool = True) -> None:
    """
    保存图像到文件
    
    Args:
        image: 图像数组
        filepath: 保存路径
        normalize_output: 是否将[0,1]范围转换为[0,255]
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备输出图像
    if normalize_output and image.max() <= 1.0:
        output = (image * 255).astype(np.uint8)
    else:
        output = image.astype(np.uint8)
    
    if HAS_CV2:
        cv2.imwrite(str(filepath), output)
    elif HAS_PIL:
        pil_image = Image.fromarray(output)
        pil_image.save(filepath)
    else:
        raise ImportError("需要安装opencv-python或Pillow来保存图像")
    
    logger.debug(f"保存图像: {filepath}")


def normalize_image(image: np.ndarray,
                    method: str = 'minmax',
                    target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    图像归一化
    
    Args:
        image: 输入图像
        method: 归一化方法
            - 'minmax': 最小-最大归一化
            - 'zscore': Z-score标准化
            - 'fixed': 固定范围归一化（假设输入为0-255）
        target_range: 目标范围
        
    Returns:
        归一化后的图像
    """
    image = image.astype(np.float64)
    
    if method == 'minmax':
        min_val = image.min()
        max_val = image.max()
        
        if max_val - min_val > 1e-10:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image)
        
        # 映射到目标范围
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    elif method == 'zscore':
        mean = image.mean()
        std = image.std()
        
        if std > 1e-10:
            normalized = (image - mean) / std
        else:
            normalized = image - mean
    
    elif method == 'fixed':
        normalized = image / 255.0
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    else:
        raise ValueError(f"未知的归一化方法: {method}")
    
    return normalized


def convert_pixel_format(image: np.ndarray,
                         source_format: str,
                         target_format: str) -> np.ndarray:
    """
    像素格式转换
    
    Args:
        image: 输入图像
        source_format: 源格式 ('uint8', 'uint16', 'float32', 'float64', 'binary')
        target_format: 目标格式
        
    Returns:
        转换后的图像
    """
    # 首先转换为float64
    if source_format == 'uint8':
        temp = image.astype(np.float64) / 255.0
    elif source_format == 'uint16':
        temp = image.astype(np.float64) / 65535.0
    elif source_format in ('float32', 'float64'):
        temp = image.astype(np.float64)
    elif source_format == 'binary':
        temp = image.astype(np.float64)
    else:
        raise ValueError(f"未知的源格式: {source_format}")
    
    # 转换到目标格式
    if target_format == 'uint8':
        result = (np.clip(temp, 0, 1) * 255).astype(np.uint8)
    elif target_format == 'uint16':
        result = (np.clip(temp, 0, 1) * 65535).astype(np.uint16)
    elif target_format == 'float32':
        result = temp.astype(np.float32)
    elif target_format == 'float64':
        result = temp.astype(np.float64)
    elif target_format == 'binary':
        result = (temp > 0.5).astype(np.float64)
    else:
        raise ValueError(f"未知的目标格式: {target_format}")
    
    return result


def create_test_pattern(pattern_type: str,
                        size: Tuple[int, int] = (128, 128),
                        **kwargs) -> np.ndarray:
    """
    创建测试图案
    
    Args:
        pattern_type: 图案类型
            - 'rectangle': 矩形
            - 'circle': 圆形
            - 'line': 线条
            - 'checkerboard': 棋盘格
            - 'random': 随机图案
        size: 图像尺寸 (height, width)
        **kwargs: 图案参数
        
    Returns:
        测试图案数组
    """
    ny, nx = size
    pattern = np.zeros((ny, nx), dtype=np.float64)
    
    if pattern_type == 'rectangle':
        # 矩形参数
        x_start = kwargs.get('x_start', nx // 4)
        x_end = kwargs.get('x_end', 3 * nx // 4)
        y_start = kwargs.get('y_start', ny // 4)
        y_end = kwargs.get('y_end', 3 * ny // 4)
        
        pattern[y_start:y_end, x_start:x_end] = 1.0
    
    elif pattern_type == 'circle':
        # 圆形参数
        cx = kwargs.get('cx', nx // 2)
        cy = kwargs.get('cy', ny // 2)
        radius = kwargs.get('radius', min(nx, ny) // 4)
        
        y, x = np.ogrid[:ny, :nx]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        pattern[mask] = 1.0
    
    elif pattern_type == 'line':
        # 线条参数
        orientation = kwargs.get('orientation', 'horizontal')
        width = kwargs.get('width', 10)
        spacing = kwargs.get('spacing', 20)
        
        if orientation == 'horizontal':
            for i in range(0, ny, spacing):
                pattern[i:min(i+width, ny), :] = 1.0
        else:
            for j in range(0, nx, spacing):
                pattern[:, j:min(j+width, nx)] = 1.0
    
    elif pattern_type == 'checkerboard':
        # 棋盘格参数
        block_size = kwargs.get('block_size', 16)
        
        for i in range(0, ny, block_size):
            for j in range(0, nx, block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    pattern[i:i+block_size, j:j+block_size] = 1.0
    
    elif pattern_type == 'random':
        # 随机图案
        threshold = kwargs.get('threshold', 0.5)
        seed = kwargs.get('seed', None)
        
        if seed is not None:
            np.random.seed(seed)
        
        pattern = (np.random.random((ny, nx)) > threshold).astype(np.float64)
    
    else:
        raise ValueError(f"未知的图案类型: {pattern_type}")
    
    return pattern


def batch_load_images(filepaths: list,
                      grayscale: bool = True,
                      normalize: bool = True,
                      target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    批量加载图像
    
    Args:
        filepaths: 图像文件路径列表
        grayscale: 是否转换为灰度图
        normalize: 是否归一化
        target_size: 目标尺寸
        
    Returns:
        图像数组 (N, H, W) 或 (N, H, W, C)
    """
    images = []
    
    for fp in filepaths:
        img = load_image(fp, grayscale, normalize, target_size)
        images.append(img)
    
    return np.array(images)
