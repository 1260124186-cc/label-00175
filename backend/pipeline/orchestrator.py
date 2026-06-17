# -*- coding: utf-8 -*-
"""
主流程编排器 (Pipeline Orchestrator)

将 OPC → ILT 精修 → SMO 光源优化 → 工艺窗口验签串联为可配置流水线，
各阶段输出自动作为下一阶段输入，并生成统一的 sign-off 摘要。

典型用法::

    from pipeline.orchestrator import PipelineOrchestrator, PipelineConfig

    config = PipelineConfig(
        enable_opc=True,
        enable_ilt=True,
        enable_smo=True,
        enable_pw_verify=True,
    )
    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run(target=target, initial_mask=mask, optical_system=opt)
    print(result.sign_off_summary())

流水线阶段：
    1. OPC  → 热点检测 + SRAF 插入 + 规则修正 → corrected_mask
    2. ILT  → 梯度投影精修 + 量化 + 复杂度控制 → optimal_mask
    3. SMO  → 光源-掩模协同优化 → optimal_mask + optimal_source
    4. PW   → focus-dose 扫描 + 可打印性判定 → PW 面积 / DOF / EL

每阶段可独立使能/禁用，跳过的阶段将透传掩模。
"""

import time
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from core.imaging import OpticalSystem, IlluminationType
from core.litho_metrics import compute_epe
from core.metrics import total_variation_isotropic

from workflows.opc import OPCConfig, OPCWorkflowResult, run_opc_workflow
from workflows.ilt import ILTConfig, ILTWorkflowResult, run_ilt_workflow
from workflows.smo import SMOConfig, SMOWorkflowResult, run_smo_workflow
from workflows.hybrid_opc_ilt import (
    HybridOPCILTConfig, HybridOPCILTWorkflowResult, run_hybrid_opc_ilt_workflow
)
from analysis.process_window import (
    ProcessWindowAnalyzer, PWMetrics, PrintabilityResult,
)

from utils.config import load_config, save_config

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    OPC = "opc"
    ILT = "ilt"
    SMO = "smo"
    PW_VERIFY = "pw_verify"


@dataclass
class PWVerifyConfig:
    focus_range: Tuple[float, float, int] = (-150, 150, 11)
    dose_range: Tuple[float, float, int] = (0.85, 1.15, 11)
    cd_tolerance: float = 0.1
    epe_tolerance: Optional[float] = None
    cd_target: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'PWVerifyConfig':
        if d is None:
            return cls()
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                if key in ('focus_range', 'dose_range') and isinstance(value, list):
                    value = tuple(value)
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            'focus_range': list(self.focus_range),
            'dose_range': list(self.dose_range),
            'cd_tolerance': self.cd_tolerance,
            'epe_tolerance': self.epe_tolerance,
            'cd_target': self.cd_target,
        }


@dataclass
class SurrogateIntegrationConfig:
    """代理模型集成配置"""
    enabled: bool = False
    checkpoint_path: Optional[str] = None
    use_adaptive: bool = True
    adaptive_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'SurrogateIntegrationConfig':
        if d is None:
            return cls()
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'checkpoint_path': self.checkpoint_path,
            'use_adaptive': self.use_adaptive,
            'adaptive_config': self.adaptive_config,
        }


@dataclass
class PipelineConfig:
    enable_opc: bool = True
    enable_ilt: bool = True
    enable_smo: bool = True
    enable_pw_verify: bool = True

    use_hybrid_opc_ilt: bool = False

    opc_config: Optional[OPCConfig] = None
    ilt_config: Optional[ILTConfig] = None
    hybrid_config: Optional[HybridOPCILTConfig] = None
    smo_config: Optional[SMOConfig] = None
    pw_verify_config: Optional[PWVerifyConfig] = None
    surrogate_config: Optional[SurrogateIntegrationConfig] = None

    output_dir: Optional[str] = None
    save_intermediate: bool = True
    verbose: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'PipelineConfig':
        if d is None:
            return cls()
        cfg = cls()
        stage_keys = {
            'enable_opc', 'enable_ilt', 'enable_smo', 'enable_pw_verify',
            'use_hybrid_opc_ilt',
            'output_dir', 'save_intermediate', 'verbose',
        }
        for key, value in d.items():
            if key in stage_keys and hasattr(cfg, key):
                setattr(cfg, key, value)
            elif key == 'opc_config':
                cfg.opc_config = OPCConfig.from_dict(value)
            elif key == 'ilt_config':
                cfg.ilt_config = ILTConfig.from_dict(value)
            elif key == 'hybrid_config':
                cfg.hybrid_config = HybridOPCILTConfig.from_dict(value)
            elif key == 'smo_config':
                cfg.smo_config = SMOConfig.from_dict(value)
            elif key == 'pw_verify_config':
                cfg.pw_verify_config = PWVerifyConfig.from_dict(value)
            elif key == 'surrogate_config':
                cfg.surrogate_config = SurrogateIntegrationConfig.from_dict(value)
        return cfg

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'PipelineConfig':
        d = load_config(config_path)
        return cls.from_dict(d.get('pipeline', d))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enable_opc': self.enable_opc,
            'enable_ilt': self.enable_ilt,
            'enable_smo': self.enable_smo,
            'enable_pw_verify': self.enable_pw_verify,
            'use_hybrid_opc_ilt': self.use_hybrid_opc_ilt,
            'opc_config': self.opc_config.to_dict() if self.opc_config else None,
            'ilt_config': self.ilt_config.to_dict() if self.ilt_config else None,
            'hybrid_config': self.hybrid_config.to_dict() if self.hybrid_config else None,
            'smo_config': self.smo_config.to_dict() if self.smo_config else None,
            'pw_verify_config': self.pw_verify_config.to_dict() if self.pw_verify_config else None,
            'surrogate_config': self.surrogate_config.to_dict() if self.surrogate_config else None,
            'output_dir': self.output_dir,
            'save_intermediate': self.save_intermediate,
            'verbose': self.verbose,
        }


@dataclass
class StageMetrics:
    stage_name: str
    elapsed_sec: float = 0.0
    skipped: bool = False
    epe_before: Optional[Dict[str, float]] = None
    epe_after: Optional[Dict[str, float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def epe_improvement(self) -> Optional[float]:
        if self.epe_before and self.epe_after:
            return self.epe_before.get('epe_mean', 0.0) - self.epe_after.get('epe_mean', 0.0)
        return None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            'stage_name': self.stage_name,
            'elapsed_sec': round(self.elapsed_sec, 3),
            'skipped': self.skipped,
        }
        if self.epe_before is not None:
            d['epe_before'] = self.epe_before
        if self.epe_after is not None:
            d['epe_after'] = self.epe_after
        if self.epe_improvement is not None:
            d['epe_improvement'] = round(self.epe_improvement, 4)
        d['extra'] = self.extra
        return d


@dataclass
class PipelineResult:
    initial_mask: np.ndarray
    final_mask: np.ndarray
    target: np.ndarray
    optical_system: OpticalSystem

    opc_result: Optional[OPCWorkflowResult] = None
    ilt_result: Optional[ILTWorkflowResult] = None
    hybrid_result: Optional[HybridOPCILTWorkflowResult] = None
    smo_result: Optional[SMOWorkflowResult] = None
    pw_metrics: Optional[PWMetrics] = None
    pw_printability: Optional[PrintabilityResult] = None

    optimal_source: Optional[np.ndarray] = None
    pw_optical_system: Optional[OpticalSystem] = None

    surrogate_used: bool = False
    surrogate_adaptive_mode: bool = False
    surrogate_stats: Optional[Dict[str, Any]] = None

    stage_metrics: List[StageMetrics] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def total_epe_improvement(self) -> float:
        first = self.stage_metrics[0] if self.stage_metrics else None
        last_non_skip = None
        for m in reversed(self.stage_metrics):
            if not m.skipped and m.epe_after is not None:
                last_non_skip = m
                break
        if first and last_non_skip and first.epe_before and last_non_skip.epe_after:
            return first.epe_before.get('epe_mean', 0.0) - last_non_skip.epe_after.get('epe_mean', 0.0)
        return 0.0

    @property
    def final_epe(self) -> Optional[Dict[str, float]]:
        for m in reversed(self.stage_metrics):
            if not m.skipped and m.epe_after is not None:
                return m.epe_after
        return None

    @property
    def initial_epe(self) -> Optional[Dict[str, float]]:
        for m in self.stage_metrics:
            if not m.skipped and m.epe_before is not None:
                return m.epe_before
        return None

    def compute_mask_complexity(self) -> Dict[str, float]:
        mask = self.final_mask
        tv = float(total_variation_isotropic(mask))
        binary_penalty = float(np.mean(4.0 * mask * (1.0 - mask)))
        grad_y = np.zeros_like(mask)
        grad_x = np.zeros_like(mask)
        grad_y[:-1, :] = mask[1:, :] - mask[:-1, :]
        grad_x[:, :-1] = mask[:, 1:] - mask[:, :-1]
        perimeter = float(np.sum(np.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)))
        return {
            'total_variation': round(tv, 4),
            'binary_penalty': round(binary_penalty, 6),
            'perimeter': round(perimeter, 4),
        }

    def sign_off_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        summary['epe'] = {}
        if self.initial_epe:
            summary['epe']['initial_mean_nm'] = round(self.initial_epe.get('epe_mean', 0.0), 4)
            summary['epe']['initial_max_nm'] = round(self.initial_epe.get('epe_max', 0.0), 4)
        if self.final_epe:
            summary['epe']['final_mean_nm'] = round(self.final_epe.get('epe_mean', 0.0), 4)
            summary['epe']['final_max_nm'] = round(self.final_epe.get('epe_max', 0.0), 4)
        summary['epe']['total_improvement_nm'] = round(self.total_epe_improvement, 4)
        init_mean = self.initial_epe.get('epe_mean', 0.0) if self.initial_epe else 0.0
        if init_mean > 1e-12:
            summary['epe']['improvement_ratio_pct'] = round(
                self.total_epe_improvement / init_mean * 100, 2)
        else:
            summary['epe']['improvement_ratio_pct'] = 0.0

        if self.pw_metrics is not None:
            summary['process_window'] = {
                'pw_area': round(self.pw_metrics.pw_area, 2),
                'pw_ratio_pct': round(self.pw_metrics.pw_ratio * 100, 2),
                'n_passing': self.pw_metrics.n_passing,
                'n_total': self.pw_metrics.n_total,
                'depth_of_focus_nm': round(self.pw_metrics.depth_of_focus, 2),
                'exposure_latitude_pct': round(self.pw_metrics.exposure_latitude, 2),
                'best_focus_nm': round(self.pw_metrics.best_focus, 2),
                'best_dose': round(self.pw_metrics.best_dose, 4),
                'best_cd_error_nm': round(self.pw_metrics.best_cd_error, 4),
            }
        else:
            summary['process_window'] = None

        summary['mask_complexity'] = self.compute_mask_complexity()

        summary['pipeline'] = {
            'stages': [m.to_dict() for m in self.stage_metrics],
            'total_time_sec': round(self.total_time, 3),
            'final_mask_shape': list(self.final_mask.shape),
        }

        if self.smo_result is not None and self.optimal_source is not None:
            src = self.optimal_source
            src_total = float(np.sum(src))
            src_nonzero = int(np.sum(src > 1e-6))
            summary['source'] = {
                'total_energy': round(src_total, 6),
                'nonzero_pixels': src_nonzero,
                'grid_size': list(src.shape),
            }
        else:
            summary['source'] = None

        summary['validation'] = {
            'pw_uses_smo_source': self.pw_uses_smo_source,
        }
        if self.pw_metrics is not None and self.optimal_source is not None:
            summary['validation']['pw_source_consistency'] = self.validate_pw_source_consistency()

        summary['surrogate'] = {
            'used': self.surrogate_used,
            'adaptive_mode': self.surrogate_adaptive_mode,
            'stats': self.surrogate_stats,
        }

        return summary

    def sign_off_text(self) -> str:
        s = self.sign_off_summary()
        lines = [
            "=" * 64,
            "  Sign-Off Summary  (OPC → ILT → SMO → PW)",
            "=" * 64,
        ]

        epe = s.get('epe', {})
        lines.append("")
        lines.append("[EPE]")
        if 'initial_mean_nm' in epe:
            lines.append(f"  Initial EPE mean : {epe['initial_mean_nm']:.3f} nm")
        if 'final_mean_nm' in epe:
            lines.append(f"  Final   EPE mean : {epe['final_mean_nm']:.3f} nm")
        lines.append(f"  Improvement      : {epe.get('total_improvement_nm', 0.0):.3f} nm  "
                      f"({epe.get('improvement_ratio_pct', 0.0):.1f}%)")

        pw = s.get('process_window')
        lines.append("")
        lines.append("[Process Window]")
        if pw is not None:
            lines.append(f"  PW area          : {pw['pw_area']:.2f} nm·dose")
            lines.append(f"  PW ratio         : {pw['pw_ratio_pct']:.1f}%")
            lines.append(f"  Passing          : {pw['n_passing']}/{pw['n_total']}")
            lines.append(f"  DOF              : {pw['depth_of_focus_nm']:.1f} nm")
            lines.append(f"  EL               : {pw['exposure_latitude_pct']:.2f}%")
            lines.append(f"  Best focus/dose  : {pw['best_focus_nm']:.1f} nm / {pw['best_dose']:.4f}")
        else:
            lines.append("  (skipped)")

        mc = s.get('mask_complexity', {})
        lines.append("")
        lines.append("[Mask Complexity]")
        lines.append(f"  Total Variation  : {mc.get('total_variation', 0.0):.4f}")
        lines.append(f"  Binary Penalty   : {mc.get('binary_penalty', 0.0):.6f}")
        lines.append(f"  Perimeter        : {mc.get('perimeter', 0.0):.2f}")

        src = s.get('source')
        lines.append("")
        lines.append("[Source]")
        if src is not None:
            lines.append(f"  Total Energy     : {src['total_energy']:.6f}")
            lines.append(f"  Nonzero Pixels   : {src['nonzero_pixels']}")
        else:
            lines.append("  (no SMO)")

        lines.append("")
        lines.append("[Pipeline]")
        for m in s.get('pipeline', {}).get('stages', []):
            tag = "(skipped)" if m['skipped'] else f"{m['elapsed_sec']:.2f}s"
            lines.append(f"  {m['stage_name']:12s} : {tag}")
        lines.append(f"  {'Total':12s} : {s.get('pipeline', {}).get('total_time_sec', 0):.2f}s")

        validation = s.get('validation', {})
        if validation:
            lines.append("")
            lines.append("[Validation]")
            pw_src_ok = validation.get('pw_uses_smo_source', None)
            if pw_src_ok is True:
                lines.append(f"  PW uses SMO source: PASS")
            elif pw_src_ok is False:
                lines.append(f"  PW uses SMO source: FAIL  (sign-off PW metrics NOT based on SMO result!)")
            else:
                lines.append(f"  PW uses SMO source: N/A")

        lines.append("=" * 64)

        return "\n".join(lines)

    @property
    def pw_uses_smo_source(self) -> bool:
        """
        回归验证：检查 PW 验签是否使用了 SMO 优化后的光源。

        当 SMO 和 PW 都启用时，PW 必须复用 SMO 产出的 optimal_source，
        否则 sign-off 的 PW 指标不代表最终联合优化结果。

        Returns:
            True 表示 PW 使用了 SMO 光源（或 SMO/PW 未同时启用），
            False 表示存在 Bug：SMO 启用但 PW 未使用其光源。
        """
        if self.optimal_source is None:
            return True
        if self.pw_metrics is None:
            return True
        if self.pw_optical_system is None:
            return False
        if self.pw_optical_system.illumination_type != IlluminationType.CUSTOM:
            return False
        if self.pw_optical_system.custom_source is None:
            return False
        smo_src = self.optimal_source
        pw_src = self.pw_optical_system.custom_source
        if smo_src.shape != pw_src.shape:
            return False
        return bool(np.allclose(smo_src, pw_src, atol=1e-10))

    def validate_pw_source_consistency(self) -> Dict[str, Any]:
        """
        回归验证：详细检查 PW 验签光源与 SMO 最优光源的一致性。

        用于确保 sign-off 的 PW 指标基于最终 SMO 联合优化结果，
        而不是默认光源或中间态光源。

        Returns:
            包含验证结果的字典：passed、checks、details
        """
        checks: Dict[str, Any] = {}
        details: Dict[str, Any] = {}

        smo_enabled = self.optimal_source is not None
        pw_enabled = self.pw_metrics is not None

        checks['smo_enabled'] = smo_enabled
        checks['pw_enabled'] = pw_enabled

        if not smo_enabled or not pw_enabled:
            return {
                'passed': True,
                'reason': 'SMO and PW are not both enabled, no consistency check needed',
                'checks': checks,
                'details': details,
            }

        checks['pw_optical_system_present'] = self.pw_optical_system is not None
        if self.pw_optical_system is None:
            return {
                'passed': False,
                'reason': 'PW optical system not saved in result',
                'checks': checks,
                'details': details,
            }

        pw_opt = self.pw_optical_system
        checks['illumination_type_is_custom'] = pw_opt.illumination_type == IlluminationType.CUSTOM
        checks['custom_source_not_none'] = pw_opt.custom_source is not None

        if pw_opt.custom_source is not None:
            smo_src = self.optimal_source
            pw_src = pw_opt.custom_source
            checks['shape_match'] = smo_src.shape == pw_src.shape
            if smo_src.shape == pw_src.shape:
                max_diff = float(np.max(np.abs(smo_src - pw_src)))
                mean_diff = float(np.mean(np.abs(smo_src - pw_src)))
                total_energy_smo = float(np.sum(smo_src))
                total_energy_pw = float(np.sum(pw_src))
                checks['values_match'] = bool(np.allclose(smo_src, pw_src, atol=1e-10))
                details['max_source_diff'] = round(max_diff, 10)
                details['mean_source_diff'] = round(mean_diff, 10)
                details['smo_source_total_energy'] = round(total_energy_smo, 8)
                details['pw_source_total_energy'] = round(total_energy_pw, 8)
            else:
                details['smo_source_shape'] = list(smo_src.shape)
                details['pw_source_shape'] = list(pw_src.shape)

        details['smo_illumination_type'] = str(pw_opt.illumination_type)

        all_passed = all(v for k, v in checks.items() if k not in ('smo_enabled', 'pw_enabled'))
        return {
            'passed': all_passed,
            'checks': checks,
            'details': details,
        }


class PipelineOrchestrator:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def run(self,
            target: np.ndarray,
            initial_mask: Optional[np.ndarray] = None,
            optical_system: Optional[OpticalSystem] = None) -> PipelineResult:
        cfg = self.config
        mask = initial_mask.copy().astype(np.float64) if initial_mask is not None else target.copy().astype(np.float64)
        tgt = target.astype(np.float64)
        optics = optical_system or OpticalSystem()

        output_dir = None
        if cfg.output_dir:
            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        stage_metrics: List[StageMetrics] = []
        t_total_start = time.time()

        opc_result = None
        ilt_result = None
        hybrid_result = None
        smo_result = None
        pw_metrics = None
        pw_printability = None
        optimal_source = None

        surrogate_imaging = None
        surrogate_stats = None

        if cfg.surrogate_config and cfg.surrogate_config.enabled:
            try:
                from surrogate import (
                    AdaptiveSurrogateImaging,
                    SurrogateImaging,
                    AdaptiveSurrogateConfig,
                )

                surr_cfg = cfg.surrogate_config
                if surr_cfg.checkpoint_path and os.path.exists(surr_cfg.checkpoint_path):
                    if surr_cfg.use_adaptive:
                        adaptive_cfg = None
                        if surr_cfg.adaptive_config:
                            adaptive_cfg = AdaptiveSurrogateConfig.from_dict(
                                surr_cfg.adaptive_config
                            )
                        surrogate_imaging = AdaptiveSurrogateImaging.from_checkpoint(
                            surr_cfg.checkpoint_path,
                            optical_system=optics,
                            adaptive_config=adaptive_cfg,
                        )
                    else:
                        surrogate_imaging = SurrogateImaging.from_checkpoint(
                            surr_cfg.checkpoint_path,
                            optical_system=optics,
                        )
                    logger.info(
                        f"代理模型加载成功: "
                        f"{'自适应' if surr_cfg.use_adaptive else '静态'}模式"
                    )
                else:
                    logger.warning(
                        f"代理模型 checkpoint 不存在: {surr_cfg.checkpoint_path}, "
                        f"将使用全精度仿真"
                    )
            except Exception as e:
                logger.warning(f"加载代理模型失败: {e}，将使用全精度仿真")
                surrogate_imaging = None

        if cfg.use_hybrid_opc_ilt and cfg.enable_opc and cfg.enable_ilt:
            if cfg.verbose:
                logger.info("=" * 50)
                logger.info("Stage 1-2/4: Hybrid OPC + ILT (混合精修模式)")
                logger.info("=" * 50)
            t0 = time.time()
            hybrid_config = cfg.hybrid_config or HybridOPCILTConfig()
            if surrogate_imaging is not None:
                if hasattr(hybrid_config, 'ilt_config') and hybrid_config.ilt_config:
                    if hybrid_config.ilt_config.imaging_model is None:
                        hybrid_config.ilt_config.imaging_model = surrogate_imaging
                else:
                    from workflows.ilt import ILTConfig
                    ilt_cfg = ILTConfig()
                    ilt_cfg.imaging_model = surrogate_imaging
                    hybrid_config.ilt_config = ilt_cfg
            hybrid_result = run_hybrid_opc_ilt_workflow(
                mask, tgt, config=hybrid_config, optical_system=optics
            )
            mask = hybrid_result.final_mask.copy()
            elapsed = time.time() - t0
            sm = StageMetrics(
                stage_name="HYBRID_OPC_ILT",
                elapsed_sec=elapsed,
                epe_before=hybrid_result.initial_epe,
                epe_after=hybrid_result.final_epe,
                extra={
                    'opc_epe_improvement': round(hybrid_result.opc_epe_improvement, 4),
                    'ilt_epe_improvement': round(hybrid_result.ilt_epe_improvement, 4),
                    'num_hotspots_optimized': hybrid_result.num_hotspots_optimized,
                    'opc_time': round(hybrid_result.opc_time, 3),
                    'ilt_time': round(hybrid_result.ilt_time, 3),
                    'converged': hybrid_result.converged,
                },
            )
            stage_metrics.append(sm)
            if cfg.verbose:
                logger.info(f"Hybrid OPC+ILT done in {elapsed:.2f}s  EPE: "
                            f"{hybrid_result.initial_epe.get('epe_mean', 0):.3f} → "
                            f"{hybrid_result.final_epe.get('epe_mean', 0):.3f} nm  "
                            f"({hybrid_result.num_hotspots_optimized} hotspots optimized)")
            if cfg.save_intermediate and output_dir:
                np.save(output_dir / 'mask_after_hybrid.npy', mask)
        else:
            if cfg.enable_opc:
                if cfg.verbose:
                    logger.info("=" * 50)
                    logger.info("Stage 1/4: OPC (Optical Proximity Correction)")
                    logger.info("=" * 50)
                t0 = time.time()
                opc_config = cfg.opc_config or OPCConfig()
                opc_result = run_opc_workflow(mask, tgt, config=opc_config, optical_system=optics)
                mask = opc_result.corrected_mask.copy()
                elapsed = time.time() - t0
                sm = StageMetrics(
                    stage_name="OPC",
                    elapsed_sec=elapsed,
                    epe_before=opc_result.initial_epe,
                    epe_after=opc_result.final_epe,
                )
                stage_metrics.append(sm)
                if cfg.verbose:
                    logger.info(f"OPC done in {elapsed:.2f}s  EPE: "
                                f"{opc_result.initial_epe.get('epe_mean', 0):.3f} → "
                                f"{opc_result.final_epe.get('epe_mean', 0):.3f} nm")
                if cfg.save_intermediate and output_dir:
                    np.save(output_dir / 'mask_after_opc.npy', mask)
            else:
                stage_metrics.append(StageMetrics(stage_name="OPC", skipped=True))
                if cfg.verbose:
                    logger.info("Stage OPC: skipped")

            if cfg.enable_ilt:
                if cfg.verbose:
                    logger.info("=" * 50)
                    logger.info("Stage 2/4: ILT (Inverse Lithography Technology)")
                    logger.info("=" * 50)
                t0 = time.time()
                ilt_config = cfg.ilt_config or ILTConfig()
                if surrogate_imaging is not None and ilt_config.imaging_model is None:
                    ilt_config.imaging_model = surrogate_imaging
                ilt_result = run_ilt_workflow(mask, tgt, optical_system=optics, config=ilt_config)
                mask = ilt_result.optimal_mask.copy()
                elapsed = time.time() - t0
                sm = StageMetrics(
                    stage_name="ILT",
                    elapsed_sec=elapsed,
                    epe_before=ilt_result.initial_epe,
                    epe_after=ilt_result.final_epe,
                    extra={
                        'initial_loss': round(ilt_result.initial_loss, 6),
                        'final_loss': round(ilt_result.final_loss, 6),
                        'converged': ilt_result.converged,
                        'iterations': ilt_result.num_iterations,
                    },
                )
                stage_metrics.append(sm)
                if cfg.verbose:
                    logger.info(f"ILT done in {elapsed:.2f}s  EPE: "
                                f"{ilt_result.initial_epe.get('epe_mean', 0):.3f} → "
                                f"{ilt_result.final_epe.get('epe_mean', 0):.3f} nm")
                if cfg.save_intermediate and output_dir:
                    np.save(output_dir / 'mask_after_ilt.npy', mask)
            else:
                stage_metrics.append(StageMetrics(stage_name="ILT", skipped=True))
                if cfg.verbose:
                    logger.info("Stage ILT: skipped")

        if cfg.enable_smo:
            if cfg.verbose:
                logger.info("=" * 50)
                logger.info("Stage 3/4: SMO (Source-Mask Optimization)")
                logger.info("=" * 50)
            t0 = time.time()
            smo_config = cfg.smo_config or SMOConfig()
            smo_result = run_smo_workflow(mask, tgt, config=smo_config, optical_system=optics)
            mask = smo_result.optimal_mask.copy()
            optimal_source = smo_result.optimal_source.copy()
            elapsed = time.time() - t0
            sm = StageMetrics(
                stage_name="SMO",
                elapsed_sec=elapsed,
                epe_before=smo_result.initial_epe,
                epe_after=smo_result.final_epe,
                extra={
                    'initial_weighted_mse': round(smo_result.initial_weighted_mse, 6),
                    'final_weighted_mse': round(smo_result.final_weighted_mse, 6),
                    'initial_pvb_hard': round(smo_result.initial_pvb_hard, 6),
                    'final_pvb_hard': round(smo_result.final_pvb_hard, 6),
                    'converged': smo_result.converged,
                    'iterations': len(smo_result.iterations),
                },
            )
            stage_metrics.append(sm)
            if cfg.verbose:
                logger.info(f"SMO done in {elapsed:.2f}s  EPE: "
                            f"{smo_result.initial_epe.get('epe_mean', 0):.3f} → "
                            f"{smo_result.final_epe.get('epe_mean', 0):.3f} nm")
            if cfg.save_intermediate and output_dir:
                np.save(output_dir / 'mask_after_smo.npy', mask)
                np.save(output_dir / 'optimal_source.npy', optimal_source)
        else:
            stage_metrics.append(StageMetrics(stage_name="SMO", skipped=True))
            if cfg.verbose:
                logger.info("Stage SMO: skipped")

        if cfg.enable_pw_verify:
            if cfg.verbose:
                logger.info("=" * 50)
                logger.info("Stage 4/4: PW Verify (Process Window Verification)")
                logger.info("=" * 50)
            t0 = time.time()
            pw_cfg = cfg.pw_verify_config or PWVerifyConfig()

            pw_optics = optics
            if optimal_source is not None:
                pw_optics = OpticalSystem(
                    wavelength=optics.wavelength,
                    na=optics.na,
                    sigma=optics.sigma,
                    pixel_size=optics.pixel_size,
                    defocus=optics.defocus,
                    magnification=optics.magnification,
                    illumination_type=IlluminationType.CUSTOM,
                    source_params=dict(optics.source_params),
                    tcc_mode=optics.tcc_mode,
                    socs_num_terms=optics.socs_num_terms,
                    custom_source=optimal_source,
                    zernike_coefficients=dict(optics.zernike_coefficients),
                )

            pixel_size = 1.0
            if smo_result is not None and cfg.smo_config is not None:
                pixel_size = cfg.smo_config.pixel_size
            elif ilt_result is not None and cfg.ilt_config is not None:
                pixel_size = cfg.ilt_config.pixel_size
            elif opc_result is not None and cfg.opc_config is not None:
                pixel_size = cfg.opc_config.pixel_size

            wafer_threshold = 0.3
            if smo_result is not None and cfg.smo_config is not None:
                wafer_threshold = cfg.smo_config.wafer_threshold
            elif ilt_result is not None and cfg.ilt_config is not None:
                wafer_threshold = cfg.ilt_config.wafer_threshold
            elif opc_result is not None and cfg.opc_config is not None:
                wafer_threshold = cfg.opc_config.wafer_threshold

            analyzer = ProcessWindowAnalyzer(
                mask=mask,
                target=tgt,
                optical_system=pw_optics,
                threshold=wafer_threshold,
                pixel_size=pixel_size,
            )

            try:
                analyzer.scan(
                    focus_range=pw_cfg.focus_range,
                    dose_range=pw_cfg.dose_range,
                    cd_target=pw_cfg.cd_target,
                    cd_tolerance=pw_cfg.cd_tolerance,
                )
                pw_printability = analyzer.judge_printability(
                    cd_tolerance=pw_cfg.cd_tolerance,
                    epe_tolerance=pw_cfg.epe_tolerance,
                    cd_target=pw_cfg.cd_target,
                )
                pw_metrics = analyzer.compute_pw_metrics(
                    cd_tolerance=pw_cfg.cd_tolerance,
                    epe_tolerance=pw_cfg.epe_tolerance,
                    cd_target=pw_cfg.cd_target,
                )
            except Exception as e:
                logger.warning(f"PW verification failed: {e}")
                pw_metrics = None
                pw_printability = None

            elapsed = time.time() - t0

            epe_after = None
            if pw_metrics is None:
                wafer_binary = (mask >= wafer_threshold).astype(np.float64)
                epe_after = compute_epe(wafer_binary, tgt, pixel_size=pixel_size)

            sm = StageMetrics(
                stage_name="PW_VERIFY",
                elapsed_sec=elapsed,
                epe_after=epe_after,
                extra={
                    'pw_area': round(pw_metrics.pw_area, 2) if pw_metrics else None,
                    'depth_of_focus': round(pw_metrics.depth_of_focus, 2) if pw_metrics else None,
                    'exposure_latitude': round(pw_metrics.exposure_latitude, 2) if pw_metrics else None,
                    'n_passing': pw_printability.n_passing if pw_printability else None,
                    'n_total': pw_printability.n_total if pw_printability else None,
                },
            )
            stage_metrics.append(sm)
            if cfg.verbose:
                if pw_metrics:
                    logger.info(f"PW done in {elapsed:.2f}s  area={pw_metrics.pw_area:.2f}  "
                                f"DOF={pw_metrics.depth_of_focus:.1f}nm  "
                                f"EL={pw_metrics.exposure_latitude:.2f}%")
                else:
                    logger.info(f"PW done in {elapsed:.2f}s  (analysis failed)")
            if cfg.save_intermediate and output_dir and pw_metrics:
                self._save_pw_report(pw_metrics, pw_printability, output_dir)
        else:
            stage_metrics.append(StageMetrics(stage_name="PW_VERIFY", skipped=True))
            if cfg.verbose:
                logger.info("Stage PW_VERIFY: skipped")

        total_time = time.time() - t_total_start

        surrogate_used = surrogate_imaging is not None
        surrogate_adaptive_mode = False
        surrogate_stats = None
        if surrogate_imaging is not None:
            surrogate_adaptive_mode = hasattr(
                surrogate_imaging, 'get_adaptive_stats'
            )
            if surrogate_adaptive_mode:
                surrogate_stats = surrogate_imaging.get_adaptive_stats()
            else:
                surrogate_stats = surrogate_imaging.get_stats()

        result = PipelineResult(
            initial_mask=initial_mask.copy() if initial_mask is not None else target.copy(),
            final_mask=mask,
            target=tgt,
            optical_system=optics,
            opc_result=opc_result,
            ilt_result=ilt_result,
            hybrid_result=hybrid_result,
            smo_result=smo_result,
            pw_metrics=pw_metrics,
            pw_printability=pw_printability,
            optimal_source=optimal_source,
            pw_optical_system=pw_optics if cfg.enable_pw_verify else None,
            surrogate_used=surrogate_used,
            surrogate_adaptive_mode=surrogate_adaptive_mode,
            surrogate_stats=surrogate_stats,
            stage_metrics=stage_metrics,
            total_time=total_time,
        )

        if output_dir:
            self._save_sign_off(result, output_dir)
            np.save(output_dir / 'final_mask.npy', mask)
            if initial_mask is not None:
                np.save(output_dir / 'initial_mask.npy', initial_mask)

        if cfg.verbose:
            logger.info("\n" + result.sign_off_text())

        return result

    def _save_sign_off(self, result: PipelineResult, output_dir: Path):
        summary = result.sign_off_summary()
        path = output_dir / 'sign_off_summary.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Sign-off summary saved: {path}")

    def _save_pw_report(self,
                        pw_metrics: PWMetrics,
                        pw_printability: Optional[PrintabilityResult],
                        output_dir: Path):
        report: Dict[str, Any] = {
            'pw_metrics': pw_metrics.to_dict(),
        }
        if pw_printability is not None:
            report['printability'] = pw_printability.to_dict()
        path = output_dir / 'pw_report.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"PW report saved: {path}")


def run_pipeline(target: np.ndarray,
                 initial_mask: Optional[np.ndarray] = None,
                 optical_system: Optional[OpticalSystem] = None,
                 config: Optional[PipelineConfig] = None,
                 config_path: Optional[Union[str, Path]] = None) -> PipelineResult:
    if config_path is not None:
        config = PipelineConfig.from_yaml(config_path)
    orchestrator = PipelineOrchestrator(config)
    return orchestrator.run(target=target, initial_mask=initial_mask, optical_system=optical_system)
