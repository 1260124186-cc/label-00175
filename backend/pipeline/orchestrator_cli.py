#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主流程编排器 CLI 入口

用法：
    # 从 backend 目录运行：
    python -m pipeline.orchestrator_cli --pattern line_space --output results/pipeline_run

    # 指定配置文件：
    python -m pipeline.orchestrator_cli --config config/pipeline_default.yaml

    # 只运行 OPC + ILT，跳过 SMO 和 PW：
    python -m pipeline.orchestrator_cli --no-smo --no-pw --pattern l_shaped

    # 自定义网格和 CD：
    python -m pipeline.orchestrator_cli --grid-size 64 --cd 45.0
"""

import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from pathlib import Path

from core.imaging import OpticalSystem
from core.test_structures import (
    TestStructureParams, TestStructureType,
    create_line_space, create_l_shaped_corner, create_contact_hole,
)
from pipeline.orchestrator import PipelineConfig, PipelineOrchestrator, run_pipeline
from utils.logger import setup_logger


def create_demo_pattern(pattern_type: str = 'line_space',
                        grid_size: int = 64,
                        cd: float = 45.0,
                        pixel_size: float = 1.0):
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
        ny, nx = gs
        cy, cx = ny // 2, nx // 2
        target[cy - 10:cy + 10, cx - 20:cx + 20] = 1.0

    return target


def main():
    parser = argparse.ArgumentParser(
        description='Litho Pipeline Orchestrator: OPC → ILT → SMO → PW Verify'
    )
    parser.add_argument('--pattern', type=str, default='line_space',
                        choices=['line_space', 'l_shaped', 'contact_hole'],
                        help='Test pattern type')
    parser.add_argument('--grid-size', type=int, default=64,
                        help='Grid size (NxN)')
    parser.add_argument('--cd', type=float, default=45.0,
                        help='Critical dimension (nm)')
    parser.add_argument('--pixel-size', type=float, default=1.0,
                        help='Pixel size (nm)')
    parser.add_argument('--wavelength', type=float, default=193.0,
                        help='Wavelength (nm)')
    parser.add_argument('--na', type=float, default=1.35,
                        help='Numerical aperture')
    parser.add_argument('--sigma', type=float, default=0.75,
                        help='Partial coherence factor')

    parser.add_argument('--no-opc', action='store_true', help='Skip OPC stage')
    parser.add_argument('--no-ilt', action='store_true', help='Skip ILT stage')
    parser.add_argument('--no-smo', action='store_true', help='Skip SMO stage')
    parser.add_argument('--no-pw', action='store_true', help='Skip PW verification')

    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--no-intermediate', action='store_true',
                        help='Do not save intermediate masks')

    args = parser.parse_args()

    logger = setup_logger('pipeline_cli', log_file='results/pipeline_orchestrator.log')
    logger.info("Litho Pipeline Orchestrator")
    logger.info("=" * 50)

    if args.config:
        pipeline_config = PipelineConfig.from_yaml(args.config)
    else:
        pipeline_config = PipelineConfig(
            enable_opc=not args.no_opc,
            enable_ilt=not args.no_ilt,
            enable_smo=not args.no_smo,
            enable_pw_verify=not args.no_pw,
            save_intermediate=not args.no_intermediate,
            verbose=True,
        )

    if args.output:
        pipeline_config.output_dir = args.output

    target = create_demo_pattern(
        pattern_type=args.pattern,
        grid_size=args.grid_size,
        cd=args.cd,
        pixel_size=args.pixel_size,
    )
    initial_mask = target.copy()

    optical_system = OpticalSystem(
        wavelength=args.wavelength,
        na=args.na,
        sigma=args.sigma,
        pixel_size=args.pixel_size,
    )

    logger.info(f"Pattern: {args.pattern}  Grid: {args.grid_size}x{args.grid_size}  CD: {args.cd}nm")
    logger.info(f"Stages: OPC={pipeline_config.enable_opc}  ILT={pipeline_config.enable_ilt}  "
                f"SMO={pipeline_config.enable_smo}  PW={pipeline_config.enable_pw_verify}")

    result = run_pipeline(
        target=target,
        initial_mask=initial_mask,
        optical_system=optical_system,
        config=pipeline_config,
    )

    print("\n" + result.sign_off_text())


if __name__ == '__main__':
    main()
