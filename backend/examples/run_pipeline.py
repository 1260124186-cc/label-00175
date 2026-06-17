#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline 编排器端到端冒烟脚本

演示如何用一条命令跑通 OPC → ILT → SMO → PW 验签全流程，
并生成统一的 sign-off 摘要（EPE、PW 面积、掩模复杂度）。

运行方式：
    # 从 backend 目录运行（默认配置）：
    python -m examples.run_pipeline

    # 指定配置文件：
    python -m examples.run_pipeline --config config/pipeline_default.yaml

    # 只跑 OPC + ILT，跳过 SMO 和 PW：
    python -m examples.run_pipeline --no-smo --no-pw

    # 自定义图案和参数：
    python -m examples.run_pipeline --pattern l_shaped --grid-size 128 --cd 32.0

    # 快速冒烟（少量迭代）：
    python -m examples.run_pipeline --smoke
"""

import sys
import os
import argparse
import time

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
    from workflows.smo import SMOConfig, SourceConstraintsConfig
    return SMOConfig(
        strategy='alternating',
        max_outer_iterations=2,
        source_max_iter=10,
        mask_max_iter=20,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Litho Pipeline: OPC -> ILT -> SMO -> PW Verify  (end-to-end smoke test)'
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
    parser.add_argument('--smoke', action='store_true',
                        help='Quick smoke test with minimal iterations')

    args = parser.parse_args()

    logger = setup_logger('pipeline_smoke', log_file='results/pipeline_smoke.log')
    logger.info("Litho Pipeline Orchestrator — Smoke Test")
    logger.info("=" * 60)

    if args.smoke:
        pipeline_config = make_smoke_config()
        pipeline_config.enable_opc = not args.no_opc
        pipeline_config.enable_ilt = not args.no_ilt
        pipeline_config.enable_smo = not args.no_smo
        pipeline_config.enable_pw_verify = not args.no_pw
        logger.info("Smoke mode: minimal iterations for quick verification")
    elif args.config:
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

    logger.info(f"Pattern : {args.pattern}  Grid: {args.grid_size}x{args.grid_size}  CD: {args.cd}nm")
    logger.info(f"Optics  : lambda={args.wavelength}nm  NA={args.na}  sigma={args.sigma}")
    logger.info(f"Stages  : OPC={pipeline_config.enable_opc}  ILT={pipeline_config.enable_ilt}  "
                f"SMO={pipeline_config.enable_smo}  PW={pipeline_config.enable_pw_verify}")

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
