# -*- coding: utf-8 -*-
"""
子命令: batch
批处理调度器（版图库批量优化）

对应原 pipeline/batch_runner_cli.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import click

from ..common import (
    global_options, output_options, optimizer_options,
    setup_cli_logger, ensure_output_dir, merge_cli_with_yaml,
    print_banner,
)


def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    suffix = p.suffix.lower()
    with open(p, "r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f) or {}
        elif suffix == ".json":
            import json
            return json.load(f)
        else:
            raise ValueError(f"不支持的配置格式: {suffix}")


@click.command(
    name="batch",
    help="版图掩模优化批处理调度器：支持本地多进程 / Redis-Celery 分布式"
)
@global_options
@output_options
@optimizer_options
# --- 输入源 ---
@click.option(
    "-s", "--source",
    type=click.Path(),
    default=None,
    help="GDS 文件/目录/文件列表路径（逗号分隔多文件）"
)
@click.option(
    "-l", "--layer",
    type=int,
    default=None,
    help="GDS 层号（source 为路径时必填）"
)
@click.option(
    "--datatype",
    type=int,
    default=0,
    show_default=True,
    help="GDS datatype"
)
@click.option(
    "--pixel-size",
    type=float,
    default=1.0,
    show_default=True,
    help="栅格化像素尺寸 (nm)"
)
@click.option(
    "--target-size",
    type=str,
    default=None,
    help="目标栅格尺寸 HxW，例如 512x512；留空自动计算"
)
@click.option(
    "--cell-name-pattern",
    type=str,
    default=None,
    help="cell 名过滤正则（re.match）"
)
@click.option(
    "--include-subcells",
    is_flag=True,
    default=False,
    help="包含非顶层 cell"
)
@click.option(
    "--lazy-load",
    is_flag=True,
    default=False,
    help="延迟加载掩模（仅建元数据，实际运行时加载）"
)
# --- 调度 ---
@click.option(
    "--mode",
    type=click.Choice(["local", "distributed"]),
    default="local",
    show_default=True,
    help="调度模式"
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="本地最大并发 worker 数（默认=CPU核数）"
)
@click.option(
    "--cpu-per-worker",
    type=int,
    default=1,
    show_default=True,
    help="每个 worker 预留 CPU 核数"
)
@click.option(
    "--per-task-timeout",
    type=int,
    default=0,
    show_default=True,
    help="单任务超时秒数（0=不限）"
)
@click.option(
    "--max-retries",
    type=int,
    default=2,
    show_default=True,
    help="失败最大重试次数"
)
# --- 优化配置 ---
@click.option(
    "--optimizer-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="OptimizationConfig YAML/JSON 文件路径"
)
@click.option(
    "--use-multi-layer",
    is_flag=True,
    default=False,
    help="使用 SMO（多层掩模联合优化）"
)
# --- 输出 ---
@click.option(
    "--format",
    "out_format",
    type=click.Choice(["csv", "json", "both"]),
    default="both",
    show_default=True,
    help="汇总输出格式"
)
@click.option(
    "--save-cell-list",
    is_flag=True,
    default=False,
    help="保存 cell 清单 CSV/JSON 到输出目录"
)
# --- 分布式 ---
@click.option(
    "--broker",
    type=str,
    default=os.environ.get("CELERY_BROKER_URL"),
    help="Celery broker URL (redis://...), 默认取环境变量 CELERY_BROKER_URL"
)
@click.option(
    "--result-backend",
    type=str,
    default=os.environ.get("CELERY_RESULT_BACKEND"),
    help="Celery 结果后端 URL, 默认取环境变量 CELERY_RESULT_BACKEND"
)
@click.option(
    "--queue",
    type=str,
    default=os.environ.get("CELERY_QUEUE", "litho_batch"),
    show_default=True,
    help="Celery 队列名"
)
# --- 杂项 ---
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="仅加载和建队，不执行优化"
)
@click.option(
    "--stop-on-failure",
    is_flag=True,
    default=False,
    help="遇到第一个失败即停止整批"
)
@click.option(
    "--priority-by-size",
    is_flag=True,
    default=False,
    help="掩模尺寸越大优先级越高"
)
@click.pass_context
def batch_cmd(
    ctx,
    verbose, log_file, config_path,
    output_dir, save_masks, no_plot,
    optimizer, max_iter, learning_rate, metric,
    # 输入源
    source, layer, datatype, pixel_size, target_size,
    cell_name_pattern, include_subcells, lazy_load,
    # 调度
    mode, max_workers, cpu_per_worker, per_task_timeout, max_retries,
    # 优化配置
    optimizer_config, use_multi_layer,
    # 输出
    out_format, save_cell_list,
    # 分布式
    broker, result_backend, queue,
    # 杂项
    dry_run, stop_on_failure, priority_by_size,
):
    """批处理调度器"""

    logger = setup_cli_logger("batch_runner_cli", verbose, log_file)
    print_banner(logger, "版图掩模优化批处理调度器")

    # --- 依赖检查 ---
    if mode == "distributed":
        try:
            import celery  # noqa: F401
        except ImportError:
            logger.error("分布式模式需安装 celery: pip install celery redis")
            sys.exit(2)
        if not broker or not result_backend:
            logger.error(
                "分布式模式需设置 --broker 和 --result-backend "
                "（或通过 CELERY_BROKER_URL / CELERY_RESULT_BACKEND 环境变量）"
            )
            sys.exit(2)

    # --- 延迟导入核心模块 ---
    try:
        from layout.layout_manager import (
            LayoutManager, LayoutLoadOptions,
        )
        from pipeline.batch_runner import (
            ResourceConfig, BatchConfig, run_batch_optimization,
            save_batch_summary,
        )
    except Exception as e:
        logger.error(f"导入模块失败，请确认 PYTHONPATH 设置正确: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

    # --- 解析 target-size ---
    resolved_target_size: Optional[tuple] = None
    if target_size:
        try:
            h, w = [int(x) for x in target_size.lower().split("x")]
            resolved_target_size = (h, w)
        except Exception:
            logger.error("--target-size 格式应为 HxW，例如 512x512")
            sys.exit(2)

    # --- LayoutLoadOptions ---
    layout_opts_kwargs: Dict[str, Any] = dict(
        datatype=datatype,
        pixel_size=pixel_size,
        target_size=resolved_target_size,
        include_subcells=include_subcells,
        cell_name_pattern=cell_name_pattern,
        load_masks_on_init=not lazy_load,
    )

    # --- 优化配置合并 ---
    optimizer_cfg_dict: Dict[str, Any] = {}
    if optimizer_config:
        try:
            raw = _load_yaml_or_json(optimizer_config)
            if isinstance(raw, dict) and "optimizer" in raw:
                optimizer_cfg_dict = dict(raw["optimizer"])
            elif isinstance(raw, dict):
                optimizer_cfg_dict = dict(raw)
        except Exception as e:
            logger.error(f"加载优化配置失败: {e}")
            sys.exit(2)
    if optimizer:
        optimizer_cfg_dict.setdefault("optimizer_type", optimizer)
    if max_iter is not None:
        optimizer_cfg_dict["max_iter"] = int(max_iter)
    if learning_rate is not None:
        optimizer_cfg_dict["learning_rate"] = float(learning_rate)
    if metric is not None:
        optimizer_cfg_dict["metric"] = metric

    # --- 合并全局 YAML ---
    if config_path:
        global_cfg = merge_cli_with_yaml({}, config_path, section_key="batch")
        if "optimizer_config" in global_cfg:
            for k, v in global_cfg["optimizer_config"].items():
                if k not in optimizer_cfg_dict:
                    optimizer_cfg_dict[k] = v

    # --- 资源配置 ---
    resource_cfg = ResourceConfig(
        max_workers=max_workers,
        cpu_per_worker=cpu_per_worker,
        per_task_timeout_sec=per_task_timeout,
        auto_detect=(max_workers is None),
    )

    # --- 输出目录 ---
    out = ensure_output_dir(output_dir) if output_dir else None
    out_str = str(out.resolve()) if out else None

    # --- 批处理配置 ---
    formats = ["csv", "json"] if out_format == "both" else [out_format]
    batch_cfg = BatchConfig(
        optimizer_config=optimizer_cfg_dict,
        use_multi_layer=use_multi_layer,
        max_retries=max_retries,
        save_optimized_masks=save_masks,
        output_dir=out_str,
        output_formats=formats,
        stop_on_first_failure=stop_on_failure,
        celery_broker_url=broker,
        celery_result_backend=result_backend,
        celery_queue_name=queue,
    )

    # --- 处理源 ---
    src_obj: Any
    if not source:
        default_dir = Path.cwd() / "layout_library"
        if default_dir.is_dir() and layer is not None:
            src_obj = str(default_dir)
            logger.info(f"未指定 --source，使用默认目录: {default_dir}")
        else:
            logger.error(
                "必须通过 --source 指定 GDS 文件/目录，"
                "或在当前目录下存在 layout_library/ 且提供 --layer"
            )
            click.echo(ctx.get_help())
            sys.exit(2)
    else:
        if "," in source and not Path(source).exists():
            src_obj = [s.strip() for s in source.split(",") if s.strip()]
        else:
            src_obj = source

    if layer is None and not (
        hasattr(src_obj, "__class__")
        and type(src_obj).__name__ in ("LayoutQueue", "LayoutLibrary")
    ):
        logger.error("当 --source 为路径时，必须指定 --layer")
        sys.exit(2)

    # --- Dry Run ---
    if dry_run:
        logger.info("=== Dry-Run 模式：仅加载与建队，不执行优化 ===")
        mgr = LayoutManager()
        lo = LayoutLoadOptions(layer=layer, **layout_opts_kwargs)
        if isinstance(src_obj, list):
            lib = mgr.load_file_list(src_obj, options=lo)
        elif Path(str(src_obj)).is_dir():
            lib = mgr.load_directory(Path(src_obj), options=lo)
        else:
            lib = mgr.load_gds_file(Path(src_obj), options=lo)
        logger.info(f"加载完成: {lib.summary()}")
        q = mgr.build_queue(
            lib, priority_by_size=priority_by_size,
            require_mask_loaded=not lazy_load,
        )
        logger.info(f"队列长度: {len(q)}，状态: {q.status_counts()}")
        if out_str:
            out_dir = Path(out_str)
            out_dir.mkdir(parents=True, exist_ok=True)
            lib.to_csv(out_dir / "cell_list.csv")
            lib.to_json(out_dir / "cell_list.json")
            logger.info(f"已导出 cell 清单到 {out_dir}")
        return 0

    # --- 执行 ---
    try:
        layout_options = (
            LayoutLoadOptions(layer=layer, **layout_opts_kwargs).__dict__
            if layer is not None
            else layout_opts_kwargs
        )
        layout_options = {
            k: v for k, v in layout_options.items()
            if k in LayoutLoadOptions.__dataclass_fields__
        }
        if layer is not None:
            layout_options["layer"] = layer

        logger.info(f"调度模式: {mode}")
        logger.info(f"总 worker 数: {resource_cfg.max_workers or 'auto'}")
        logger.info(f"输出目录: {out_str or '临时目录'}")

        summary, results, lib, queue = run_batch_optimization(
            source=src_obj,
            layer=layer,
            layout_options=(
                layout_options if layer is not None else None
            ),
            resource_config=resource_cfg,
            batch_config=batch_cfg,
            mode=mode,
            output_dir=output_dir,
        )

        if save_cell_list and lib is not None and out_str:
            out_dir = Path(out_str)
            lib.to_csv(out_dir / "cell_list.csv")
            lib.to_json(out_dir / "cell_list.json")

        s = summary
        sep = "=" * 60
        logger.info("")
        logger.info(sep)
        logger.info(f"批次 {s.batch_id} 完成")
        logger.info(
            f"总耗时: {s.total_elapsed_sec:.1f}s  "
            f"任务数: {s.total_tasks}  "
            f"成功率: {s.success_rate * 100:.1f}%"
        )
        logger.info(
            f"状态分布: done={s.done}  fail={s.failed}  "
            f"cancel={s.cancelled}  timeout={s.timeout}"
        )
        if s.avg_initial_mse is not None:
            logger.info(
                f"MSE: {s.avg_initial_mse:.3e} → {s.avg_final_mse:.3e}  "
                f"平均改善: {(s.avg_mse_improvement_ratio or 0) * 100:.2f}%"
            )
        logger.info(
            f"收敛: {s.converged_count}/{s.done}  "
            f"({s.converged_rate * 100:.1f}%)  "
            f"平均耗时 {s.avg_elapsed_sec:.2f}s "
            f"(中位数 {s.median_elapsed_sec:.2f}s)"
        )
        logger.info(sep)
        return 0 if s.failed == 0 else 1
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130
    except Exception as e:
        logger.error(f"批处理执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 4
