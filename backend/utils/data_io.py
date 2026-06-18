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

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


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


def load_gds_layer_by_cell_name(filepath: Union[str, Path],
                                cell_name: str,
                                layer: int,
                                datatype: int = 0,
                                pixel_size: float = 1.0,
                                target_size: Optional[Tuple[int, int]] = None,
                                bounds: Optional[Tuple[float, float, float, float]] = None,
                                flatten_references: bool = False) -> np.ndarray:
    """
    加载 GDS 中指定 cell 的指定层，栅格化为 numpy 掩模

    **与 load_gds_layer 的区别**：
    - load_gds_layer: 遍历整个 GDS 所有 cell，展平所有引用，生成全图掩模
    - 本函数: 只处理指定 cell，由 flatten_references 控制是否展平子引用

    Args:
        filepath: GDS/OASIS 文件路径
        cell_name: 目标 cell 名
        layer: GDS 层号
        datatype: GDS 数据类型号
        pixel_size: 每像素对应的 GDS 单位长度
        target_size: 目标尺寸 (height, width)，None 则由 bounds 自动计算
        bounds: 版图范围 (xmin, ymin, xmax, ymax)
        flatten_references: 是否递归展平该 cell 内的子引用。
            - False（默认）: 只取该 cell 自身的直接多边形，用于层次化处理
            - True: 等价于对该 cell 展平，用于层次化加载的叶节点

    Returns:
        二值掩模 (H, W) float64

    Raises:
        ValueError: 指定 cell 不存在或无多边形且未指定 target_size
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GDS文件不存在: {filepath}")
    if not HAS_GDSTK and not HAS_GDSPY:
        raise ImportError("需要安装 gdstk 或 gdspy")

    polygons = _read_gds_polygons_by_cell(
        str(filepath), cell_name, layer, datatype,
        flatten_references=flatten_references,
    )

    if not polygons:
        logger.warning(
            f"GDS {filepath.name} cell={cell_name} (layer={layer}, dt={datatype}) "
            f"无多边形 (flatten_references={flatten_references})"
        )
        if target_size is not None:
            return np.zeros(target_size, dtype=np.float64)
        raise ValueError(
            f"cell={cell_name} 在 (layer={layer}, datatype={datatype}) "
            f"无多边形，且未指定 target_size"
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
        f"按cell加载GDS层: {filepath.name}#{cell_name}, "
        f"flatten_refs={flatten_references}, layer={layer}, "
        f"掩模尺寸: {mask.shape}, 多边形数: {len(polygons)}"
    )

    return mask


def _read_gds_polygons_by_cell(filepath: str,
                               cell_name: str,
                               layer: int,
                               datatype: int,
                               flatten_references: bool = False) -> list:
    """
    从 GDS 中提取指定 cell 的多边形顶点列表

    Args:
        filepath: GDS 文件路径
        cell_name: 目标 cell 名
        layer: 层号
        datatype: 数据类型
        flatten_references: 是否递归展平子引用

    Returns:
        多边形顶点列表
    """
    polygons = []

    if HAS_GDSTK:
        lib = gdstk.read_gds(filepath)
        cell_map = {c.name: c for c in lib.cells}
        cell = cell_map.get(cell_name)
        if cell is None:
            logger.warning(f"GDS 中找不到 cell: {cell_name}, "
                           f"可用: {list(cell_map.keys())}")
            return []

        if flatten_references:
            # 使用 gdstk 内置展平功能（正确处理 ARef 阵列展开）
            # depth=None 表示完全展平，返回的多边形已经过变换
            for poly in cell.get_polygons(depth=None):
                if poly.layer == layer and poly.datatype == datatype:
                    polygons.append(np.array(poly.points))
        else:
            # 只取自身直接多边形，不展平子引用
            for poly in cell.polygons:
                if poly.layer == layer and poly.datatype == datatype:
                    polygons.append(np.array(poly.points))

    elif HAS_GDSPY:
        lib = gdspy.GdsLibrary(infile=filepath)
        cell_map = lib.cells
        cell = cell_map.get(cell_name)
        if cell is None:
            logger.warning(f"GDS 中找不到 cell: {cell_name}, "
                           f"可用: {list(cell_map.keys())}")
            return []

        if flatten_references:
            # gdspy: 用 flatten() 方法展平 cell
            try:
                flat_cell = cell.flatten()
                for polyset in flat_cell.polygons:
                    if (polyset.layers[0] == layer
                            and polyset.datatypes[0] == datatype):
                        for pts in polyset.polygons:
                            polygons.append(np.array(pts))
            except Exception as e:
                logger.warning(f"gdspy flatten 失败，回退手动展平: {e}")
                identity = np.eye(3)
                # 自身直接多边形
                for polyset in cell.polygons:
                    if (polyset.layers[0] == layer
                            and polyset.datatypes[0] == datatype):
                        for pts in polyset.polygons:
                            polygons.append(np.array(pts))
                # 手动展平子引用
                for ref in cell.references:
                    ref_polys = _flatten_reference_gdspy(
                        ref, lib, layer, datatype, identity
                    )
                    polygons.extend(ref_polys)
        else:
            # 只取自身直接多边形
            for polyset in cell.polygons:
                if (polyset.layers[0] == layer
                        and polyset.datatypes[0] == datatype):
                    for pts in polyset.polygons:
                        polygons.append(np.array(pts))

    return polygons


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


def save_npy(array: np.ndarray,
             filepath: Union[str, Path]) -> None:
    """
    保存 numpy 数组为 .npy 格式文件

    Args:
        array: numpy 数组
        filepath: 保存路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(filepath), array)
    logger.debug(f"保存 numpy 数组: {filepath}, shape={array.shape}, dtype={array.dtype}")


def load_npy(filepath: Union[str, Path]) -> np.ndarray:
    """
    从 .npy 文件加载 numpy 数组

    Args:
        filepath: 文件路径

    Returns:
        numpy 数组
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    array = np.load(str(filepath))
    logger.debug(f"加载 numpy 数组: {filepath}, shape={array.shape}, dtype={array.dtype}")
    return array


def save_gds_layer(mask: np.ndarray,
                   filepath: Union[str, Path],
                   layer: int = 0,
                   datatype: int = 0,
                   pixel_size: float = 1.0,
                   origin: Tuple[float, float] = (0.0, 0.0),
                   cell_name: str = 'TOP',
                   library_name: str = 'MASK',
                   threshold: float = 0.5,
                   merge_polygons: bool = True) -> None:
    """
    将二值掩模数组导出为 GDS/OASIS 版图文件

    通过轮廓提取将二值掩模转换为多边形，写入 GDS 指定层。
    支持 gdstk（优先）和 gdspy 两种后端。

    Args:
        mask: 二值掩模数组 (H, W)，值范围 [0, 1]
        filepath: 输出 GDS 文件路径 (.gds 或 .oas)
        layer: GDS 层号
        datatype: GDS 数据类型号
        pixel_size: 每像素对应的物理长度（GDS单位，通常为nm）
        origin: 版图左下角物理坐标 (x, y)
        cell_name: 顶层单元格名称
        library_name: GDS库名称
        threshold: 二值化阈值，> threshold 视为掩模区域
        merge_polygons: 是否对相邻区域做多边形合并（减少矩形数量）

    Raises:
        ImportError: 未安装 gdstk 或 gdspy
        ValueError: 掩模不是二维数组
    """
    if not HAS_GDSTK and not HAS_GDSPY:
        raise ImportError("需要安装 gdstk 或 gdspy 来导出 GDS/OASIS 文件")

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"掩模必须是二维数组，当前维度: {mask.ndim}")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    binary_mask = (mask > threshold).astype(np.uint8)

    polygons = _mask_to_polygons(binary_mask, pixel_size, origin, merge_polygons)

    if HAS_GDSTK:
        lib = gdstk.Library(name=library_name)
        cell = lib.new_cell(cell_name)
        for poly_pts in polygons:
            poly = gdstk.Polygon(poly_pts, layer=layer, datatype=datatype)
            cell.add(poly)
        if filepath.suffix.lower() in ('.oas', '.oasis'):
            lib.write_oas(str(filepath))
        else:
            lib.write_gds(str(filepath))
    else:
        lib = gdspy.GdsLibrary()
        cell = lib.new_cell(cell_name)
        for poly_pts in polygons:
            poly = gdspy.Polygon(poly_pts, layer=layer, datatype=datatype)
            cell.add(poly)
        lib.write_gds(str(filepath))

    logger.info(
        f"导出 GDS: {filepath}, layer={layer}, datatype={datatype}, "
        f"像素尺寸={pixel_size}, 多边形数={len(polygons)}"
    )


def _mask_to_polygons(binary_mask: np.ndarray,
                      pixel_size: float,
                      origin: Tuple[float, float],
                      merge_polygons: bool) -> List[np.ndarray]:
    """
    将二值掩模转换为物理坐标多边形列表

    使用连通域分析提取每个独立区域，再转换为矩形多边形集合。
    对于每个连通域，将其分解为轴对齐矩形的并集。

    Args:
        binary_mask: 二值掩模 (uint8)，值为 0 或 1
        pixel_size: 每像素物理长度
        origin: 版图左下角原点坐标
        merge_polygons: 是否尝试合并相邻矩形

    Returns:
        多边形顶点列表，每个多边形为 (N, 2) 的 numpy 数组
    """
    try:
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary_mask)
    except ImportError:
        logger.warning("scipy 未安装，退化为逐像素矩形方法")
        labeled = binary_mask
        num_features = int(binary_mask.sum()) if binary_mask.size > 0 else 0

    polygons: List[np.ndarray] = []
    ox, oy = origin
    ny, nx = binary_mask.shape

    if merge_polygons and num_features > 0:
        for label_idx in range(1, num_features + 1):
            region_mask = (labeled == label_idx)
            rects = _extract_rectangles(region_mask)
            for (r_start, r_end, c_start, c_end) in rects:
                x0 = ox + c_start * pixel_size
                y0 = oy + (ny - r_end) * pixel_size
                x1 = ox + c_end * pixel_size
                y1 = oy + (ny - r_start) * pixel_size
                poly_pts = np.array([
                    [x0, y0], [x1, y0], [x1, y1], [x0, y1]
                ], dtype=np.float64)
                polygons.append(poly_pts)
    else:
        rows, cols = np.where(binary_mask > 0)
        for r, c in zip(rows, cols):
            x0 = ox + c * pixel_size
            y0 = oy + (ny - r - 1) * pixel_size
            x1 = ox + (c + 1) * pixel_size
            y1 = oy + (ny - r) * pixel_size
            poly_pts = np.array([
                [x0, y0], [x1, y0], [x1, y1], [x0, y1]
            ], dtype=np.float64)
            polygons.append(poly_pts)

    return polygons


def _extract_rectangles(region_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    从单个连通域中提取轴对齐矩形集合（行扫描法）

    Args:
        region_mask: 单个连通域的二值掩模

    Returns:
        矩形列表，每个矩形为 (row_start, row_end, col_start, col_end)
        区间为 [start, end)
    """
    rects: List[Tuple[int, int, int, int]] = []
    ny, nx = region_mask.shape

    for r in range(ny):
        c = 0
        while c < nx:
            if region_mask[r, c]:
                c_start = c
                while c < nx and region_mask[r, c]:
                    c += 1
                c_end = c
                rects.append((r, r + 1, c_start, c_end))
            else:
                c += 1

    merged: List[Tuple[int, int, int, int]] = []
    for rect in rects:
        rs, re, cs, ce = rect
        merged_flag = False
        for i, (mrs, mre, mcs, mce) in enumerate(merged):
            if cs == mcs and ce == mce and re == mrs:
                merged[i] = (mrs, re, mcs, mce)
                merged_flag = True
                break
        if not merged_flag:
            merged.append(rect)

    return merged


def save_hdf5_results(filepath: Union[str, Path],
                      mask_sequence: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                      loss_history: Optional[Union[List[float], np.ndarray]] = None,
                      optical_params: Optional[Dict[str, Any]] = None,
                      extra_data: Optional[Dict[str, Any]] = None,
                      compression: str = 'gzip',
                      compression_opts: int = 4) -> None:
    """
    将优化结果批量存储为 HDF5 文件

    存储内容包括：
    - 掩模序列（优化过程中的中间掩模）
    - 损失历史（每步损失值）
    - 光学系统参数（波长、NA、sigma等）
    - 额外自定义数据

    Args:
        filepath: HDF5 文件输出路径 (.h5 或 .hdf5)
        mask_sequence: 掩模序列，形状 (N, H, W) 或掩模列表
        loss_history: 损失历史列表/数组
        optical_params: 光学参数字典（如 OpticalSystem.to_dict()）
        extra_data: 额外要存储的数据字典（支持 numpy 数组、标量、字符串）
        compression: 压缩算法 ('gzip', 'lzf', None)
        compression_opts: 压缩级别 (gzip: 0-9)

    Raises:
        ImportError: 未安装 h5py
    """
    if not HAS_H5PY:
        raise ImportError("需要安装 h5py 来导出 HDF5 文件")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(filepath), 'w') as f:
        if mask_sequence is not None:
            if isinstance(mask_sequence, list):
                mask_array = np.stack(mask_sequence, axis=0)
            else:
                mask_array = np.asarray(mask_sequence)
            f.create_dataset(
                'mask_sequence',
                data=mask_array,
                compression=compression,
                compression_opts=compression_opts if compression == 'gzip' else None
            )
            f['mask_sequence'].attrs['num_masks'] = mask_array.shape[0]
            f['mask_sequence'].attrs['height'] = mask_array.shape[1]
            f['mask_sequence'].attrs['width'] = mask_array.shape[2]
            logger.debug(
                f"HDF5: 存储掩模序列 {mask_array.shape}, "
                f"压缩={compression}"
            )

        if loss_history is not None:
            loss_array = np.asarray(loss_history, dtype=np.float64)
            f.create_dataset(
                'loss_history',
                data=loss_array,
                compression=compression,
                compression_opts=compression_opts if compression == 'gzip' else None
            )
            f['loss_history'].attrs['num_iterations'] = len(loss_array)
            f['loss_history'].attrs['final_loss'] = float(loss_array[-1]) if len(loss_array) > 0 else 0.0
            f['loss_history'].attrs['min_loss'] = float(loss_array.min()) if len(loss_array) > 0 else 0.0
            logger.debug(
                f"HDF5: 存储损失历史 ({len(loss_array)} 步), "
                f"最终损失={loss_array[-1] if len(loss_array) > 0 else 'N/A'}"
            )

        if optical_params is not None:
            optics_group = f.create_group('optical_params')
            _write_dict_to_hdf5(optics_group, optical_params)
            logger.debug(f"HDF5: 存储光学参数 {list(optical_params.keys())}")

        if extra_data is not None:
            extra_group = f.create_group('extra_data')
            _write_dict_to_hdf5(extra_group, extra_data)
            logger.debug(f"HDF5: 存储额外数据 {list(extra_data.keys())}")

    logger.info(f"HDF5 结果已保存: {filepath}")


def load_hdf5_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    从 HDF5 文件加载优化结果

    Args:
        filepath: HDF5 文件路径

    Returns:
        包含所有存储数据的字典：
        - 'mask_sequence': 掩模序列数组 (N, H, W)，若不存在则为 None
        - 'loss_history': 损失历史数组，若不存在则为 None
        - 'optical_params': 光学参数字典，若不存在则为 None
        - 'extra_data': 额外数据字典，若不存在则为 None

    Raises:
        ImportError: 未安装 h5py
        FileNotFoundError: 文件不存在
    """
    if not HAS_H5PY:
        raise ImportError("需要安装 h5py 来读取 HDF5 文件")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 文件不存在: {filepath}")

    result: Dict[str, Any] = {
        'mask_sequence': None,
        'loss_history': None,
        'optical_params': None,
        'extra_data': None
    }

    with h5py.File(str(filepath), 'r') as f:
        if 'mask_sequence' in f:
            result['mask_sequence'] = f['mask_sequence'][:]

        if 'loss_history' in f:
            result['loss_history'] = f['loss_history'][:]

        if 'optical_params' in f:
            result['optical_params'] = _read_hdf5_to_dict(f['optical_params'])

        if 'extra_data' in f:
            result['extra_data'] = _read_hdf5_to_dict(f['extra_data'])

    logger.info(f"从 HDF5 加载结果: {filepath}")
    return result


def _write_dict_to_hdf5(group: 'h5py.Group',
                         data_dict: Dict[str, Any]) -> None:
    """
    递归将字典写入 HDF5 Group

    Args:
        group: HDF5 Group 对象
        data_dict: 要写入的数据字典
    """
    for key, value in data_dict.items():
        key_str = str(key)
        if isinstance(value, dict):
            sub_group = group.create_group(key_str)
            _write_dict_to_hdf5(sub_group, value)
        elif isinstance(value, np.ndarray):
            group.create_dataset(key_str, data=value)
        elif isinstance(value, (list, tuple)):
            if len(value) > 0 and all(isinstance(v, (int, float, np.number)) for v in value):
                group.create_dataset(key_str, data=np.asarray(value))
            else:
                group.attrs[key_str] = str(value)
        elif isinstance(value, (int, float, np.number, bool)):
            group.attrs[key_str] = value
        elif value is None:
            group.attrs[key_str] = 'None'
        else:
            group.attrs[key_str] = str(value)


def _read_hdf5_to_dict(group: 'h5py.Group') -> Dict[str, Any]:
    """
    递归将 HDF5 Group 读取为字典

    Args:
        group: HDF5 Group 对象

    Returns:
        数据字典
    """
    result: Dict[str, Any] = {}

    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = _read_hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[:]

    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if value == 'None':
            result[key] = None
        else:
            result[key] = value

    return result


def save_optimization_result(result: Any,
                             output_dir: Union[str, Path],
                             prefix: str = 'result',
                             formats: Optional[List[str]] = None,
                             optical_params: Optional[Dict[str, Any]] = None,
                             gds_layer: int = 0,
                             gds_pixel_size: float = 1.0,
                             hdf5_compression: str = 'gzip') -> Dict[str, str]:
    """
    保存 MaskOptimizationResult 到多种格式

    支持的格式：
    - 'png': 保存优化后掩模为 PNG 图像
    - 'npy': 保存优化后掩模为 numpy .npy
    - 'gds': 保存优化后掩模为 GDS 版图
    - 'hdf5': 批量保存（掩模序列+损失历史+光学参数）

    Args:
        result: MaskOptimizationResult 对象
        output_dir: 输出目录
        prefix: 文件名前缀
        formats: 要保存的格式列表，默认 ['png', 'npy', 'gds', 'hdf5']
        optical_params: 光学参数字典（用于 HDF5）
        gds_layer: GDS 导出层号
        gds_pixel_size: GDS 导出像素尺寸
        hdf5_compression: HDF5 压缩算法

    Returns:
        保存文件路径字典 {format: filepath}
    """
    if formats is None:
        formats = ['png', 'npy', 'gds', 'hdf5']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: Dict[str, str] = {}

    optimized_mask = getattr(result, 'optimized_mask', None)
    loss_history = getattr(result, 'loss_history', None)
    mask_history = getattr(result, 'mask_history', None)

    if 'png' in formats and optimized_mask is not None:
        png_path = output_dir / f'{prefix}_mask.png'
        save_image(optimized_mask, str(png_path))
        saved_files['png'] = str(png_path)
        logger.info(f"保存 PNG: {png_path}")

    if 'npy' in formats and optimized_mask is not None:
        npy_path = output_dir / f'{prefix}_mask.npy'
        save_npy(optimized_mask, str(npy_path))
        saved_files['npy'] = str(npy_path)
        logger.info(f"保存 NPY: {npy_path}")

    if 'gds' in formats and optimized_mask is not None:
        gds_path = output_dir / f'{prefix}_mask.gds'
        try:
            save_gds_layer(
                optimized_mask,
                str(gds_path),
                layer=gds_layer,
                pixel_size=gds_pixel_size
            )
            saved_files['gds'] = str(gds_path)
            logger.info(f"保存 GDS: {gds_path}")
        except ImportError as e:
            logger.warning(f"跳过 GDS 导出: {e}")

    if 'hdf5' in formats:
        hdf5_path = output_dir / f'{prefix}_results.h5'
        try:
            extra = {}
            for attr in ['initial_mask', 'target_image', 'final_wafer_image',
                         'initial_wafer_image', 'total_iterations', 'total_time',
                         'converged', 'message']:
                val = getattr(result, attr, None)
                if val is not None:
                    if isinstance(val, (int, float, str, bool, np.ndarray)):
                        extra[attr] = val

            final_metrics = getattr(result, 'final_metrics', None)
            if final_metrics is not None and hasattr(final_metrics, '__dict__'):
                metrics_dict = {
                    k: v for k, v in final_metrics.__dict__.items()
                    if isinstance(v, (int, float))
                }
                if metrics_dict:
                    extra['final_metrics'] = metrics_dict

            initial_metrics = getattr(result, 'initial_metrics', None)
            if initial_metrics is not None and hasattr(initial_metrics, '__dict__'):
                metrics_dict = {
                    k: v for k, v in initial_metrics.__dict__.items()
                    if isinstance(v, (int, float))
                }
                if metrics_dict:
                    extra['initial_metrics'] = metrics_dict

            save_hdf5_results(
                str(hdf5_path),
                mask_sequence=mask_history if mask_history else (
                    np.stack([optimized_mask]) if optimized_mask is not None else None
                ),
                loss_history=loss_history,
                optical_params=optical_params,
                extra_data=extra if extra else None,
                compression=hdf5_compression
            )
            saved_files['hdf5'] = str(hdf5_path)
            logger.info(f"保存 HDF5: {hdf5_path}")
        except ImportError as e:
            logger.warning(f"跳过 HDF5 导出: {e}")

    return saved_files
