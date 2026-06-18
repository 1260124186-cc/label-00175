# -*- coding: utf-8 -*-
"""
CLI 公共工具模块

提供各子命令共享的：
  1. Click 选项装饰器（光学系统、测试图案、输出目录等）
  2. 通用工具函数（创建测试图案、加载 YAML、设置日志、构建光学系统等）
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from functools import wraps

import click
import numpy as np

from utils.logger import setup_logger as _setup_logger
from utils.config import load_config
from core.imaging import OpticalSystem
from core.test_structures import (
    TestStructureParams,
    TestStructureType,
    LineOrientation,
    create_line_space,
    create_l_shaped_corner,
    create_contact_hole,
)


# ---------------------------------------------------------------------------
# 共享选项装饰器
# ---------------------------------------------------------------------------

def global_options(f: Callable) -> Callable:
    """全局选项：--verbose / --log-file / --config"""
    @click.option(
        "-v", "--verbose",
        is_flag=True,
        default=False,
        help="启用详细日志（DEBUG 级别）"
    )
    @click.option(
        "--log-file",
        type=click.Path(dir_okay=False),
        default=None,
        help="日志输出文件路径"
    )
    @click.option(
        "-c", "--config",
        "config_path",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="YAML 配置文件路径（全局参数会被子命令参数覆盖）"
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def output_options(f: Callable) -> Callable:
    """输出目录选项：--output-dir / --save-masks / --no-plot"""
    @click.option(
        "-o", "--output-dir",
        type=click.Path(file_okay=False),
        default="results",
        show_default=True,
        help="结果输出目录"
    )
    @click.option(
        "--save-masks",
        is_flag=True,
        default=False,
        help="保存优化前后掩模的 .npy 文件"
    )
    @click.option(
        "--no-plot",
        is_flag=True,
        default=False,
        help="不生成可视化图表（仅保存数值数据）"
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def optical_system_options(f: Callable) -> Callable:
    """光学系统选项：--wavelength / --na / --sigma / --defocus / --pixel-size"""
    @click.option(
        "--wavelength",
        type=float,
        default=193.0,
        show_default=True,
        help="光源波长 (nm)，如 ArF=193, EUV=13.5"
    )
    @click.option(
        "--na",
        type=float,
        default=1.35,
        show_default=True,
        help="数值孔径 Numerical Aperture"
    )
    @click.option(
        "--sigma",
        type=float,
        default=0.75,
        show_default=True,
        help="部分相干因子 (0~1)"
    )
    @click.option(
        "--defocus",
        type=float,
        default=0.0,
        show_default=True,
        help="离焦量 (nm)，正值过焦，负值欠焦"
    )
    @click.option(
        "--pixel-size",
        type=float,
        default=1.0,
        show_default=True,
        help="栅格化像素尺寸 (nm)"
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def test_pattern_options(f: Callable) -> Callable:
    """测试图案选项：--pattern / --grid-size / --cd / --pitch"""
    @click.option(
        "-p", "--pattern",
        type=click.Choice(["line_space", "l_shaped", "contact_hole", "rectangle"]),
        default="line_space",
        show_default=True,
        help="测试图案类型"
    )
    @click.option(
        "--grid-size",
        type=str,
        default="64x64",
        show_default=True,
        help="栅格尺寸 HxW，例如 128x128"
    )
    @click.option(
        "--cd",
        type=float,
        default=45.0,
        show_default=True,
        help="关键尺寸 Critical Dimension (nm)"
    )
    @click.option(
        "--pitch",
        type=float,
        default=None,
        help="间距 Pitch (nm)，默认 = 2×CD"
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def optimizer_options(f: Callable) -> Callable:
    """通用优化器选项：--optimizer / --max-iter / --learning-rate / --metric"""
    @click.option(
        "--optimizer",
        type=click.Choice([
            "gradient_descent", "adam", "bfgs", "sgd",
            "rmsprop", "lbfgs", "newton",
            "genetic", "pso", "simulated_annealing", "de", "cmaes"
        ]),
        default=None,
        help="优化器类型（默认视子命令而定）"
    )
    @click.option(
        "--max-iter",
        type=int,
        default=None,
        help="最大迭代次数"
    )
    @click.option(
        "--learning-rate",
        type=float,
        default=None,
        help="学习率"
    )
    @click.option(
        "--metric",
        type=click.Choice(["mse", "mae", "ssim", "epe"]),
        default=None,
        help="优化目标度量（部分子命令有效）"
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# 通用工具函数
# ---------------------------------------------------------------------------

def parse_grid_size(grid_size_str: str) -> Tuple[int, int]:
    """解析 'HxW' 格式字符串为 (h, w) 元组"""
    try:
        h, w = [int(x) for x in grid_size_str.lower().split("x")]
        return (h, w)
    except Exception:
        raise click.BadParameter(
            f"--grid-size 格式应为 HxW，例如 512x512，当前: {grid_size_str}"
        )


def setup_cli_logger(
    name: str = "litho_sim_cli",
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """配置并返回 CLI 日志器"""
    level = logging.DEBUG if verbose else logging.INFO
    logger = _setup_logger(
        name,
        log_file=log_file,
        level=level,
    )
    return logger


def build_optical_system(
    wavelength: float,
    na: float,
    sigma: float,
    defocus: float,
    pixel_size: float,
    magnification: float = 4.0,
    socs_num_terms: int = 8,
) -> OpticalSystem:
    """根据 CLI 参数构造 OpticalSystem"""
    return OpticalSystem(
        wavelength=wavelength,
        na=na,
        sigma=sigma,
        pixel_size=pixel_size,
        defocus=defocus,
        magnification=magnification,
        socs_num_terms=socs_num_terms,
    )


def create_pattern(
    pattern_type: str,
    grid_size: Tuple[int, int],
    cd: float,
    pitch: Optional[float] = None,
    pixel_size: float = 1.0,
) -> Tuple[np.ndarray, str]:
    """
    根据 CLI 参数创建测试图案

    Returns:
        (mask_array, description)
    """
    if pitch is None:
        pitch = cd * 2

    if pattern_type == "line_space":
        params = TestStructureParams(
            grid_size=grid_size,
            pixel_size=pixel_size,
            cd=cd,
            pitch=pitch,
            structure_type=TestStructureType.LINE_SPACE,
        )
        target = create_line_space(params)
        desc = f"线/空间结构 CD={cd}nm Pitch={pitch}nm"
    elif pattern_type == "l_shaped":
        params = TestStructureParams(
            grid_size=grid_size,
            pixel_size=pixel_size,
            cd=cd,
            pitch=pitch,
            structure_type=TestStructureType.L_SHAPED_CORNER,
        )
        target = create_l_shaped_corner(params)
        desc = f"L形拐角结构 CD={cd}nm"
    elif pattern_type == "contact_hole":
        params = TestStructureParams(
            grid_size=grid_size,
            pixel_size=pixel_size,
            cd=cd,
            pitch=pitch,
            structure_type=TestStructureType.CONTACT_HOLE,
        )
        target = create_contact_hole(params)
        desc = f"接触孔阵列 CD={cd}nm Pitch={pitch}nm"
    elif pattern_type == "rectangle":
        target = np.zeros(grid_size, dtype=np.float64)
        ny, nx = grid_size
        cy, cx = ny // 2, nx // 2
        y_r = max(4, int(cd / pixel_size // 2))
        x_r = max(8, int(cd / pixel_size))
        target[cy - y_r:cy + y_r, cx - x_r:cx + x_r] = 1.0
        desc = f"简单矩形结构 CD={cd}nm"
    else:
        raise click.BadParameter(f"不支持的图案类型: {pattern_type}")

    return target, desc


def ensure_output_dir(output_dir: str, subdir: Optional[str] = None) -> Path:
    """确保输出目录存在，可选追加子目录"""
    path = Path(output_dir)
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge_cli_with_yaml(
    cli_params: Dict[str, Any],
    config_path: Optional[str],
    section_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    合并 YAML 配置与 CLI 参数。
    CLI 参数优先级高于 YAML 配置（非 None 的 CLI 值覆盖 YAML）。

    Args:
        cli_params:   从 click 解析得到的参数字典
        config_path:  YAML 配置文件路径（可为 None）
        section_key:  取 YAML 中的哪个子 section，None 取顶层

    Returns:
        合并后的配置字典
    """
    if not config_path:
        return cli_params

    yaml_cfg = load_config(config_path)
    if section_key and section_key in yaml_cfg:
        yaml_section = yaml_cfg[section_key]
    else:
        yaml_section = yaml_cfg

    merged = dict(yaml_section)
    for k, v in cli_params.items():
        if v is not None:
            merged[k] = v
    return merged


def print_banner(logger: logging.Logger, title: str) -> None:
    """打印醒目的日志标题横幅"""
    sep = "=" * 70
    logger.info(sep)
    logger.info(title)
    logger.info(sep)


def print_summary_block(logger: logging.Logger, lines: list) -> None:
    """打印汇总信息块"""
    sep = "-" * 60
    logger.info("")
    logger.info(sep)
    for line in lines:
        logger.info(line)
    logger.info(sep)
    logger.info("")
