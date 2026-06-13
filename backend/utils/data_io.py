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
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import gdstk
    HAS_GDSTK = True
except ImportError:
    HAS_GDSTK = False

try:
    import gdspy
    HAS_GDSPY = True
except ImportError:
    HAS_GDSPY = False


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


def load_gds_layer(filepath: Union[str, Path],
                   layer: int,
                   datatype: int = 0,
                   pixel_size: float = 1.0,
                   target_size: Optional[Tuple[int, int]] = None,
                   bounds: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """
    加载GDS/OASIS版图文件中指定层的几何图形，栅格化为numpy掩模数组

    支持通过gdstk或gdspy读取GDS/OASIS文件，将指定(layer, datatype)的
    多边形栅格化为二值掩模（0.0/1.0）。

    Args:
        filepath: GDS/OASIS文件路径
        layer: GDS层号
        datatype: GDS数据类型号，默认为0
        pixel_size: 每像素对应的GDS单位长度（GDS单位通常为nm），默认1.0
        target_size: 目标尺寸 (height, width)，None则由bounds自动计算
        bounds: 版图范围 (xmin, ymin, xmax, ymax)，GDS单位；
                None则自动计算指定层所有多边形的包围盒

    Returns:
        二值掩模数组，形状为 (H, W)，dtype为float64，值为0.0或1.0

    Raises:
        FileNotFoundError: 文件不存在
        ImportError: 未安装gdstk或gdspy
        ValueError: 指定层无多边形或其他参数错误
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"GDS文件不存在: {filepath}")

    if not HAS_GDSTK and not HAS_GDSPY:
        raise ImportError("需要安装gdstk或gdspy来加载GDS/OASIS文件")

    polygons = _read_gds_polygons(str(filepath), layer, datatype)

    if not polygons:
        logger.warning(f"GDS文件 {filepath} 中 (layer={layer}, datatype={datatype}) 无多边形")

        if target_size is not None:
            return np.zeros(target_size, dtype=np.float64)
        raise ValueError(
            f"GDS文件中 (layer={layer}, datatype={datatype}) 无多边形，"
            f"且未指定target_size"
        )

    if bounds is None:
        all_pts = np.vstack(polygons)
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
    else:
        xmin, ymin, xmax, ymax = bounds

    if target_size is not None:
        ny, nx = target_size
    else:
        nx = max(1, int(np.ceil((xmax - xmin) / pixel_size)))
        ny = max(1, int(np.ceil((ymax - ymin) / pixel_size)))

    pixel_polygons = []
    for poly in polygons:
        px = (poly[:, 0] - xmin) / pixel_size
        py = (ymax - poly[:, 1]) / pixel_size
        pixel_polygons.append(np.column_stack([px, py]).astype(np.float64))

    mask = _rasterize_polygons(pixel_polygons, ny, nx)

    logger.debug(
        f"加载GDS层: {filepath}, layer={layer}, datatype={datatype}, "
        f"掩模尺寸: {mask.shape}, 多边形数: {len(polygons)}"
    )

    return mask


def _read_gds_polygons(filepath: str,
                        layer: int,
                        datatype: int) -> list:
    """
    从GDS/OASIS文件中读取指定层的多边形顶点列表

    Args:
        filepath: GDS文件路径
        layer: GDS层号
        datatype: GDS数据类型号

    Returns:
        多边形顶点列表，每个元素为 (N, 2) numpy数组
    """
    polygons = []

    if HAS_GDSTK:
        lib = gdstk.read_gds(filepath)
        for cell in lib.cells:
            for poly in cell.polygons:
                if poly.layer == layer and poly.datatype == datatype:
                    polygons.append(np.array(poly.points))
            for ref in cell.references:
                if not hasattr(ref, 'cell'):
                    continue
                ref_polygons = _flatten_reference(
                    ref, layer, datatype, np.eye(3)
                )
                polygons.extend(ref_polygons)

    elif HAS_GDSPY:
        lib = gdspy.GdsLibrary(infile=filepath)
        for cell in lib.cells.values():
            for polyset in cell.polygons:
                if (polyset.layers[0] == layer
                        and polyset.datatypes[0] == datatype):
                    for pts in polyset.polygons:
                        polygons.append(np.array(pts))
            for ref in cell.references:
                ref_polygons = _flatten_reference_gdspy(
                    ref, lib, layer, datatype, np.eye(3)
                )
                polygons.extend(ref_polygons)

    return polygons


def _flatten_reference(ref, layer: int, datatype: int,
                        transform: np.ndarray) -> list:
    """
    递归展平gdstk引用（SRef/ARef），收集变换后的多边形

    Args:
        ref: gdstk引用对象
        layer: 目标层号
        datatype: 目标数据类型号
        transform: 累积3x3齐次变换矩阵

    Returns:
        变换后的多边形顶点列表
    """
    polygons = []

    local_transform = _ref_transform_gdstk(ref)
    combined = local_transform @ transform

    cell = ref.cell
    if cell is None:
        return polygons

    for poly in cell.polygons:
        if poly.layer == layer and poly.datatype == datatype:
            pts = np.array(poly.points)
            pts_h = np.column_stack([pts, np.ones(len(pts))])
            transformed = (combined @ pts_h.T).T[:, :2]
            polygons.append(transformed)

    for sub_ref in cell.references:
        polygons.extend(
            _flatten_reference(sub_ref, layer, datatype, combined)
        )

    return polygons


def _ref_transform_gdstk(ref) -> np.ndarray:
    """
    从gdstk引用对象构建3x3齐次变换矩阵

    Args:
        ref: gdstk SRef或ARef对象

    Returns:
        3x3齐次变换矩阵
    """
    mat = np.eye(3)

    origin = getattr(ref, 'origin', None)
    if origin is not None:
        origin = np.asarray(origin)
        if origin.ndim == 0:
            origin = np.array([float(origin), 0.0])
        mat[0, 2] = origin[0]
        mat[1, 2] = origin[1]

    rotation = getattr(ref, 'rotation', None)
    if rotation is not None:
        rotation = float(rotation)
        cos_r = np.cos(np.radians(rotation))
        sin_r = np.sin(np.radians(rotation))
        rot = np.eye(3)
        rot[0, 0] = cos_r
        rot[0, 1] = -sin_r
        rot[1, 0] = sin_r
        rot[1, 1] = cos_r
        mat = rot @ mat

    magnification = getattr(ref, 'magnification', None)
    if magnification is not None and magnification != 1.0:
        scale = np.eye(3)
        scale[0, 0] = float(magnification)
        scale[1, 1] = float(magnification)
        mat = scale @ mat

    x_reflection = getattr(ref, 'x_reflection', False)
    if x_reflection:
        mirror = np.eye(3)
        mirror[1, 1] = -1.0
        mat = mirror @ mat

    return mat


def _flatten_reference_gdspy(ref, lib, layer: int, datatype: int,
                              transform: np.ndarray) -> list:
    """
    递归展平gdspy引用，收集变换后的多边形

    Args:
        ref: gdspy引用对象
        lib: gdspy GdsLibrary
        layer: 目标层号
        datatype: 目标数据类型号
        transform: 累积3x3齐次变换矩阵

    Returns:
        变换后的多边形顶点列表
    """
    polygons = []

    local_transform = _ref_transform_gdspy(ref)
    combined = local_transform @ transform

    ref_cell = None
    ref_cell_name = getattr(ref, 'ref_cell', None)
    if ref_cell_name is not None:
        if isinstance(ref_cell_name, str):
            ref_cell = lib.cells.get(ref_cell_name)
        else:
            ref_cell = ref_cell_name

    if ref_cell is None:
        return polygons

    for polyset in ref_cell.polygons:
        if (polyset.layers[0] == layer
                and polyset.datatypes[0] == datatype):
            for pts in polyset.polygons:
                pts_arr = np.array(pts)
                pts_h = np.column_stack([pts_arr, np.ones(len(pts_arr))])
                transformed = (combined @ pts_h.T).T[:, :2]
                polygons.append(transformed)

    for sub_ref in ref_cell.references:
        polygons.extend(
            _flatten_reference_gdspy(sub_ref, lib, layer, datatype, combined)
        )

    return polygons


def _ref_transform_gdspy(ref) -> np.ndarray:
    """
    从gdspy引用对象构建3x3齐次变换矩阵

    Args:
        ref: gdspy CellReference或CellArray对象

    Returns:
        3x3齐次变换矩阵
    """
    mat = np.eye(3)

    origin = getattr(ref, 'origin', (0, 0))
    if origin is not None:
        origin = np.asarray(origin, dtype=float).ravel()
        if len(origin) >= 2:
            mat[0, 2] = origin[0]
            mat[1, 2] = origin[1]

    rotation = getattr(ref, 'rotation', None)
    if rotation is not None and rotation != 0:
        rotation = float(rotation)
        cos_r = np.cos(np.radians(rotation))
        sin_r = np.sin(np.radians(rotation))
        rot = np.eye(3)
        rot[0, 0] = cos_r
        rot[0, 1] = -sin_r
        rot[1, 0] = sin_r
        rot[1, 1] = cos_r
        mat = rot @ mat

    magnification = getattr(ref, 'magnification', None)
    if magnification is not None and magnification != 1.0:
        scale = np.eye(3)
        scale[0, 0] = float(magnification)
        scale[1, 1] = float(magnification)
        mat = scale @ mat

    x_reflection = getattr(ref, 'x_reflection', False)
    if x_reflection:
        mirror = np.eye(3)
        mirror[1, 1] = -1.0
        mat = mirror @ mat

    return mat


def _rasterize_polygons(pixel_polygons: list,
                         height: int,
                         width: int) -> np.ndarray:
    """
    将像素坐标多边形栅格化为二值掩模

    Args:
        pixel_polygons: 多边形顶点列表（像素坐标）
        height: 掩模高度
        width: 掩模宽度

    Returns:
        二值掩模数组 (H, W)，dtype=float64
    """
    if HAS_CV2:
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in pixel_polygons:
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask.astype(np.float64)

    elif HAS_PIL:
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        for poly in pixel_polygons:
            xy = [(float(p[0]), float(p[1])) for p in poly]
            draw.polygon(xy, fill=255)
        mask = np.array(img, dtype=np.float64) / 255.0
        mask = (mask > 0.5).astype(np.float64)
        return mask

    else:
        raise ImportError("需要安装opencv-python或Pillow来栅格化多边形")


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
