#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline 编排器端到端冒烟脚本

演示如何用一条命令跑通 OPC → ILT → SMO → PW 验签全流程，
并生成统一的 sign-off 摘要（EPE、PW 面积、掩模复杂度）。

运行方式（支持从仓库根目录或 backend/ 目录启动）：

    # 方式 1：从仓库根目录运行
    cd label-00175
    python -m backend.examples.run_pipeline --smoke
    python -m backend.examples.run_pipeline --config backend/config/pipeline_default.yaml

    # 方式 2：从 backend/ 目录运行
    cd label-00175/backend
    python -m examples.run_pipeline --smoke
    python -m examples.run_pipeline --config config/pipeline_default.yaml

    # 指定图案
    python -m backend.examples.run_pipeline --pattern l_shaped --grid-size 128 --cd 32.0
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# ---------------------------------------------------------------------------
# 路径兼容：无论从仓库根目录还是 backend/ 目录启动，都能正确发现 backend 子包
# ---------------------------------------------------------------------------
_CURRENT_FILE = Path(__file__).resolve()
_BACKEND_DIR = _CURRENT_FILE.parent.parent   # backend/
_REPO_ROOT = _BACKEND_DIR.parent             # label-00175/

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import numpy as np

from core.imaging import OpticalSystem
from core.test_structures import (
    TestStructureParams, TestStructureType,
    create_line_space, create_l_shaped_corner, create_contact_hole,
)
from pipeline.orchestrator import PipelineConfig, run_pipeline
from utils.logger import setup_logger
from utils.config import load_config


def resolve_config_path(path: Optional[str]) -> Optional[str]:
    """解析配置文件路径：优先相对启动 cwd，其次相对 backend/config/"""
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p.resolve())
    candidate = _BACKEND_DIR / p
    if candidate.exists():
        return str(candidate.resolve())
    candidate = _BACKEND_DIR / "config" / p
    if candidate.exists():
        return str(candidate.resolve())
    raise FileNotFoundError(f"Config file not found: {path}. "
                            f"Tried: {p.resolve()}, {_BACKEND_DIR / p}, {_BACKEND_DIR / 'config' / p}")


def build_optical_system_from_dict(optics_dict: Optional[Dict[str, Any]] = None) -> OpticalSystem:
    """从 YAML 配置构造 OpticalSystem；缺省时返回默认"""
    if optics_dict:
        return OpticalSystem.from_config({'optical_system': optics_dict})
    return OpticalSystem()


def build_target_from_pattern_dict(pattern_dict: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, Optional[float]]:
    """从 YAML test_pattern 配置构造 target 图案；返回 (target, pixel_size_or_None)"""
    if pattern_dict is None:
        pattern = create_line_space(TestStructureParams(
            grid_size=(64, 64), pixel_size=1.0, cd=45.0, pitch=90.0,
            structure_type=TestStructureType.LINE_SPACE,
        ))
        return pattern, 1.0

    grid_size = pattern_dict.get('grid_size', [64, 64])
    if isinstance(grid_size, int):
        gs = (grid_size, grid_size)
    elif isinstance(grid_size, (list, tuple)) and len(grid_size) == 1:
        gs = (grid_size[0], grid_size[0])
    else:
        gs = (int(grid_size[0]), int(grid_size[1]))

    pixel_size = float(pattern_dict.get('pixel_size', 1.0))
    cd = float(pattern_dict.get('cd', 45.0))
    pitch = float(pattern_dict.get('pitch', cd * 2))
    ptype = str(pattern_dict.get('type', 'line_space')).lower()

    if ptype == 'line_space':
        params = TestStructureParams(
            grid_size=gs, pixel_size=pixel_size, cd=cd, pitch=pitch,
            structure_type=TestStructureType.LINE_SPACE,
        )
        return create_line_space(params), pixel_size
    elif ptype == 'contact_hole':
        params = TestStructureParams(
            grid_size=gs, pixel_size=pixel_size, cd=cd, pitch=pitch,
            structure_type=TestStructureType.CONTACT_HOLE,
        )
        return create_contact_hole(params), pixel_size
    elif ptype == 'l_shaped':
        params = TestStructureParams(
            grid_size=gs, pixel_size=pixel_size, cd=cd, pitch=pitch,
            structure_type=TestStructureType.L_SHAPED_CORNER,
        )
        return create_l_shaped_corner(params), pixel_size
    else:
        target = np.zeros(gs, dtype=np.float64)
        cy, cx = gs[0] // 2, gs[1] // 2
        h = max(1, int(cd / 2 / max(pixel_size, 1e-6)))
        w = max(1, int(pitch / pixel_size))
        target[cy - h:cy + h, cx - w:cx + w] = 1.0
        return target, pixel_size


def create_demo_pattern(pattern_type: str = 'line_space',
                        grid_size: int = 64,
                        cd: float = 45.0,
                        pixel_size: float = 1.0) -> Tuple[np.ndarray, float]:
    pitch = cd * 2
    gs = (grid_size, grid_size)
    params = TestStructureParams(
        grid_size=gs,
        pixel_size=pixel_size,
        cd=cd,
        pitch=pitch,
        structure_type=TestStructureType.LINE_SPACE,
    )

    if pattern_type == 'line_space':
        target = create_line_space(params)
    elif pattern_type == 'l_shaped':
        params.structure_type = TestStructureType.L_SHAPED_CORNER
        target = create_l_shaped_corner(params)
    elif pattern_type == 'contact_hole':
        params.structure_type = TestStructureType.CONTACT_HOLE
        target = create_contact_hole(params)
    else:
        target = np.zeros(gs, dtype=np.float64)
        cy, cx = grid_size // 2, grid_size // 2
        target[cy - 10:cy + 10, cx - 20:cx + 20] = 1.0

    return target, pixel_size


def _quick_opc_config():
    from workflows.opc import OPCConfig
    return OPCConfig(
        max_iterations=3,
        optimizer_max_iter=5,
        sraf_enable=False,
        verbose=True,
    )


def _quick_ilt_config():
    from workflows.ilt import ILTConfig
    return ILTConfig(
        max_iter=30,
        convergence_patience=5,
        verbose=True,
    )


def _quick_smo_config():
    from workflows.smo import SMOConfig
    return SMOConfig(
        strategy='alternating',
        max_outer_iterations=2,
        source_max_iter=10,
        mask_max_iter=20,
        verbose=True,
    )


def make_smoke_config():
    return PipelineConfig(
        enable_opc=True,
        enable_ilt=True,
        enable_smo=True,
        enable_pw_verify=True,
        opc_config=_quick_opc_config(),
        ilt_config=_quick_ilt_config(),
        smo_config=_quick_smo_config(),
        pw_verify_config=None,
        save_intermediate=False,
        verbose=True,
    )


def build_pipeline_config_from_yaml(yaml_path: str,
                                    enable_opc: bool = True,
                                    enable_ilt: bool = True,
                                    enable_smo: bool = True,
                                    enable_pw: bool = True,
                                    output_dir: Optional[str] = None,
                                    save_intermediate: bool = True) -> PipelineConfig:
    full_cfg = load_config(yaml_path)
    cfg = PipelineConfig.from_yaml(yaml_path)

    cfg.enable_opc = enable_opc
    cfg.enable_ilt = enable_ilt
    cfg.enable_smo = enable_smo
    cfg.enable_pw_verify = enable_pw
    if output_dir is not None:
        cfg.output_dir = output_dir
    cfg.save_intermediate = save_intermediate
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='Litho Pipeline: OPC -> ILT -> SMO -> PW Verify  (end-to-end smoke test)'
    )
    parser.add_argument('--pattern', type=str, default=None,
                        choices=['line_space', 'l_shaped', 'contact_hole'],
                        help='Override test pattern type')
    parser.add_argument('--grid-size', type=int, default=None,
                        help='Override grid size (NxN)')
    parser.add_argument('--cd', type=float, default=None,
                        help='Override critical dimension (nm)')
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Override pixel size (nm)')
    parser.add_argument('--wavelength', type=float, default=None,
                        help='Override wavelength (nm)')
    parser.add_argument('--na', type=float, default=None,
                        help='Override numerical aperture')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Override partial coherence factor')

    parser.add_argument('--no-opc', action='store_true', help='Skip OPC stage')
    parser.add_argument('--no-ilt', action='store_true', help='Skip ILT stage')
    parser.add_argument('--no-smo', action='store_true', help='Skip SMO stage')
    parser.add_argument('--no-pw', action='store_true', help='Skip PW verification')

    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file path (e.g. config/pipeline_default.yaml)')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--no-intermediate', action='store_true',
                        help='Do not save intermediate masks')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick smoke test with minimal iterations')

    args = parser.parse_args()

    # --- 解析配置路径 ---
    cfg_path = resolve_config_path(args.config)
    if args.smoke and cfg_path is None:
        cfg_path = resolve_config_path('pipeline_default.yaml')

    # --- 从 YAML 加载全部配置块 ---
    optics_dict: Optional[Dict[str, Any]] = None
    pattern_dict: Optional[Dict[str, Any]] = None
    if cfg_path is not None:
        full_cfg = load_config(cfg_path)
        optics_dict = full_cfg.get('optical_system', None)
        pattern_dict = full_cfg.get('test_pattern', None)

    logger = setup_logger('pipeline_smoke', log_file='results/pipeline_smoke.log')
    logger.info("Litho Pipeline Orchestrator — Smoke Test")
    logger.info("=" * 60)
    if cfg_path is not None:
        logger.info(f"Config   : {cfg_path}")
    else:
        logger.info("Config   : (defaults, no YAML)")

    # --- Optical System：优先 YAML，其次 CLI 覆盖，最终默认值 ---
    optical_system = build_optical_system_from_dict(optics_dict)
    if args.wavelength is not None:
        optical_system.wavelength = args.wavelength
    if args.na is not None:
        optical_system.na = args.na
    if args.sigma is not None:
        optical_system.sigma = args.sigma
    if args.pixel_size is not None:
        optical_system.pixel_size = args.pixel_size

    # --- Pattern：优先 YAML，其次 CLI 参数，最终默认 ---
    if args.pattern is None and args.grid_size is None and args.cd is None and pattern_dict is not None:
        target, px = build_target_from_pattern_dict(pattern_dict)
        pattern_name = pattern_dict.get('type', 'line_space')
        gs_hint = pattern_dict.get('grid_size', [64, 64])
        grid_log = f"{gs_hint[0]}x{gs_hint[1]}" if isinstance(gs_hint, (list, tuple)) else f"{gs_hint}x{gs_hint}"
        cd_log = pattern_dict.get('cd', 45.0)
    else:
        ptype = args.pattern or 'line_space'
        gs = args.grid_size or 64
        cd_val = args.cd or 45.0
        px = args.pixel_size or optical_system.pixel_size or 1.0
        target, px = create_demo_pattern(ptype, gs, cd_val, px)
        pattern_name = ptype
        grid_log = f"{gs}x{gs}"
        cd_log = cd_val
        optical_system.pixel_size = px

    # --- Pipeline 配置 ---
    if args.smoke:
        pipeline_config = make_smoke_config()
    elif cfg_path is not None:
        pipeline_config = build_pipeline_config_from_yaml(
            cfg_path,
            enable_opc=not args.no_opc,
            enable_ilt=not args.no_ilt,
            enable_smo=not args.no_smo,
            enable_pw=not args.no_pw,
            output_dir=args.output,
            save_intermediate=not args.no_intermediate,
        )
    else:
        pipeline_config = PipelineConfig(
            enable_opc=not args.no_opc,
            enable_ilt=not args.no_ilt,
            enable_smo=not args.no_smo,
            enable_pw_verify=not args.no_pw,
            save_intermediate=not args.no_intermediate,
            output_dir=args.output,
            verbose=True,
        )

    initial_mask = target.copy()

    logger.info(f"Pattern : {pattern_name}  Grid: {grid_log}  CD: {cd_log}nm  Pixel: {px}nm")
    logger.info(f"Optics  : lambda={optical_system.wavelength}nm  NA={optical_system.na}  "
                f"sigma={optical_system.sigma}  type={optical_system.illumination_type.value}")
    logger.info(f"Stages  : OPC={pipeline_config.enable_opc}  ILT={pipeline_config.enable_ilt}  "
                f"SMO={pipeline_config.enable_smo}  PW={pipeline_config.enable_pw_verify}")
    if pipeline_config.output_dir:
        logger.info(f"Output  : {pipeline_config.output_dir}")

    t0 = time.time()
    result = run_pipeline(
        target=target,
        initial_mask=initial_mask,
        optical_system=optical_system,
        config=pipeline_config,
    )
    elapsed = time.time() - t0

    print("\n" + result.sign_off_text())

    validation = result.validate_pw_source_consistency()
    if not validation['passed']:
        print("\n*** WARNING: PW source consistency check FAILED! ***")
        print(f"    Details: {validation}")
        print("    The sign-off PW metrics may NOT be based on the SMO-optimized source.")
    else:
        print(f"\n  PW source consistency: PASSED  (elapsed {elapsed:.1f}s)")

    return 0 if validation['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
