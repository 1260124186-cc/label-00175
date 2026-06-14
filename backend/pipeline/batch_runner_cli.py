# -*- coding: utf-8 -*-
"""
批处理调度命令行入口（供 Docker Compose 服务与本地脚本使用）

用法示例：

  # 本地多进程模式（默认）
  python -m pipeline.batch_runner_cli \
      --source ./layout_library --layer 0 \
      --output-dir ./results/batch_run \
      --max-workers 4

  # 分布式模式（提交到 Redis/Celery 集群）
  CELERY_BROKER_URL=redis://redis:6379/0 \
  CELERY_RESULT_BACKEND=redis://redis:6379/1 \
  python -m pipeline.batch_runner_cli \
      --mode distributed \
      --source ./layout_library --layer 0

  # 使用 YAML 优化配置
  python -m pipeline.batch_runner_cli \
      --source ./layout_library --layer 0 \
      --optimizer-config ./backend/config/opc_default.yaml
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    try:
        import colorlog  # type: ignore
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s" + fmt,
            log_colors={
                'DEBUG': 'cyan', 'INFO': 'green',
                'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red',
            }
        ))
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
    except ImportError:
        logging.basicConfig(level=level, format=fmt)


def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    suffix = p.suffix.lower()
    with open(p, 'r', encoding='utf-8') as f:
        if suffix in ('.yaml', '.yml'):
            import yaml
            return yaml.safe_load(f) or {}
        elif suffix == '.json':
            import json
            return json.load(f)
        else:
            raise ValueError(f"不支持的配置格式: {suffix}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="版图掩模优化批处理调度器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入源
    src = parser.add_argument_group("输入源")
    src.add_argument("--source", "-s", required=False,
                     help="GDS 文件/目录/文件列表路径（逗号分隔多文件）")
    src.add_argument("--layer", type=int, default=None,
                     help="GDS 层号（source 为路径时必填）")
    src.add_argument("--datatype", type=int, default=0, help="GDS datatype")
    src.add_argument("--pixel-size", type=float, default=1.0,
                     help="栅格化像素尺寸 (nm)")
    src.add_argument("--target-size", default=None,
                     help="目标栅格尺寸 HxW，例如 512x512；留空自动计算")
    src.add_argument("--cell-name-pattern", default=None,
                     help="cell 名过滤正则（re.match）")
    src.add_argument("--include-subcells", action="store_true",
                     help="包含非顶层 cell")
    src.add_argument("--lazy-load", action="store_true",
                     help="延迟加载掩模（仅建元数据，实际运行时加载）")

    # 调度模式与资源
    sched = parser.add_argument_group("调度")
    sched.add_argument("--mode", choices=["local", "distributed"], default="local",
                       help="调度模式")
    sched.add_argument("--max-workers", type=int, default=None,
                       help="本地最大并发 worker 数（默认=CPU核数）")
    sched.add_argument("--cpu-per-worker", type=int, default=1,
                       help="每个 worker 预留 CPU 核数")
    sched.add_argument("--per-task-timeout", type=int, default=0,
                       help="单任务超时秒数（0=不限）")
    sched.add_argument("--max-retries", type=int, default=2,
                       help="失败最大重试次数")

    # 优化配置
    opt = parser.add_argument_group("优化配置")
    opt.add_argument("--optimizer-config", default=None,
                     help="OptimizationConfig YAML/JSON 文件路径")
    opt.add_argument("--optimizer", default=None,
                     help="优化器类型，如 gradient_descent / adam / lbfgs")
    opt.add_argument("--max-iter", type=int, default=None,
                     help="每任务最大迭代次数")
    opt.add_argument("--learning-rate", type=float, default=None,
                     help="学习率")
    opt.add_argument("--use-multi-layer", action="store_true",
                     help="使用 SMO（多层掩模联合优化）")

    # 输出
    out = parser.add_argument_group("输出")
    out.add_argument("--output-dir", "-o", default=None,
                     help="输出目录（汇总表+掩模）")
    out.add_argument("--format", choices=["csv", "json", "both"], default="both",
                     help="汇总输出格式")
    out.add_argument("--save-masks", action="store_true",
                     help="保存每个 cell 优化后的掩模 npy")
    out.add_argument("--save-cell-list", action="store_true",
                     help="保存 cell 清单 CSV/JSON 到输出目录")

    # 分布式
    dist = parser.add_argument_group("分布式")
    dist.add_argument("--broker", default=os.environ.get("CELERY_BROKER_URL"),
                      help="Celery broker URL (redis://...)")
    dist.add_argument("--result-backend",
                      default=os.environ.get("CELERY_RESULT_BACKEND"),
                      help="Celery 结果后端 URL")
    dist.add_argument("--queue", default=os.environ.get("CELERY_QUEUE", "litho_batch"),
                      help="Celery 队列名")

    # 杂项
    misc = parser.add_argument_group("杂项")
    misc.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    misc.add_argument("--dry-run", action="store_true",
                      help="仅加载和建队，不执行优化")
    misc.add_argument("--stop-on-failure", action="store_true",
                      help="遇到第一个失败即停止整批")
    misc.add_argument("--priority-by-size", action="store_true",
                      help="掩模尺寸越大优先级越高")

    return parser


def _main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    log = logging.getLogger("batch_runner_cli")

    # --- 依赖检查 ---
    if args.mode == "distributed":
        try:
            import celery  # noqa: F401
        except ImportError:
            log.error("分布式模式需安装 celery: pip install celery redis")
            return 2
        if not args.broker or not args.result_backend:
            log.error("分布式模式需设置 --broker 和 --result-backend "
                      "（或通过 CELERY_BROKER_URL / CELERY_RESULT_BACKEND 环境变量）")
            return 2

    # --- 延迟导入核心模块（避免子进程重复初始化开销） ---
    try:
        from layout.layout_manager import (
            LayoutManager, LayoutLoadOptions,
        )
        from pipeline.batch_runner import (
            ResourceConfig, BatchConfig, run_batch_optimization,
            save_batch_summary,
        )
    except Exception as e:
        log.error(f"导入模块失败，请确认 PYTHONPATH 设置正确: {e}")
        import traceback
        traceback.print_exc()
        return 3

    # --- 构建 LayoutLoadOptions ---
    target_size = None
    if args.target_size:
        try:
            h, w = [int(x) for x in args.target_size.lower().split('x')]
            target_size = (h, w)
        except Exception:
            log.error("--target-size 格式应为 HxW，例如 512x512")
            return 2

    layout_opts_kwargs: Dict[str, Any] = dict(
        datatype=args.datatype,
        pixel_size=args.pixel_size,
        target_size=target_size,
        include_subcells=args.include_subcells,
        cell_name_pattern=args.cell_name_pattern,
        load_masks_on_init=not args.lazy_load,
    )

    # --- 构建优化配置 ---
    optimizer_cfg_dict: Dict[str, Any] = {}
    if args.optimizer_config:
        try:
            raw = _load_yaml_or_json(args.optimizer_config)
            if isinstance(raw, dict) and 'optimizer' in raw:
                optimizer_cfg_dict = dict(raw['optimizer'])
            elif isinstance(raw, dict):
                optimizer_cfg_dict = dict(raw)
        except Exception as e:
            log.error(f"加载优化配置失败: {e}")
            return 2
    if args.optimizer:
        optimizer_cfg_dict.setdefault('optimizer_type', args.optimizer)
    if args.max_iter is not None:
        optimizer_cfg_dict['max_iter'] = int(args.max_iter)
    if args.learning_rate is not None:
        optimizer_cfg_dict['learning_rate'] = float(args.learning_rate)

    # --- 资源配置 ---
    resource_cfg = ResourceConfig(
        max_workers=args.max_workers,
        cpu_per_worker=args.cpu_per_worker,
        per_task_timeout_sec=args.per_task_timeout,
        auto_detect=(args.max_workers is None),
    )

    # --- 批处理配置 ---
    formats = (['csv', 'json'] if args.format == 'both'
               else [args.format])
    batch_cfg = BatchConfig(
        optimizer_config=optimizer_cfg_dict,
        use_multi_layer=args.use_multi_layer,
        max_retries=args.max_retries,
        save_optimized_masks=args.save_masks,
        output_dir=str(Path(args.output_dir).resolve()) if args.output_dir else None,
        output_formats=formats,
        stop_on_first_failure=args.stop_on_failure,
        celery_broker_url=args.broker,
        celery_result_backend=args.result_backend,
        celery_queue_name=args.queue,
    )

    # --- 处理源 ---
    source: Any
    if not args.source:
        # 交互式：尝试读当前目录下的 layout_library
        default_dir = Path.cwd() / "layout_library"
        if default_dir.is_dir() and args.layer is not None:
            source = default_dir
            log.info(f"未指定 --source，使用默认目录: {default_dir}")
        else:
            log.error("必须通过 --source 指定 GDS 文件/目录，或在当前目录下存在 layout_library/ 且提供 --layer")
            parser.print_help()
            return 2
    else:
        # 支持逗号分隔的多文件
        if ',' in args.source and not Path(args.source).exists():
            source = [s.strip() for s in args.source.split(',') if s.strip()]
        else:
            source = args.source

    layer = args.layer
    if layer is None and not (
        hasattr(source, '__class__')
        and type(source).__name__ in ('LayoutQueue', 'LayoutLibrary')
    ):
        log.error("当 --source 为路径时，必须指定 --layer")
        return 2

    # --- 仅加载不执行 ---
    if args.dry_run:
        log.info("=== Dry-Run 模式：仅加载与建队，不执行优化 ===")
        mgr = LayoutManager()
        lo = LayoutLoadOptions(layer=layer, **layout_opts_kwargs)
        if isinstance(source, list):
            lib = mgr.load_file_list(source, options=lo)
        elif Path(str(source)).is_dir():
            lib = mgr.load_directory(Path(source), options=lo)
        else:
            lib = mgr.load_gds_file(Path(source), options=lo)
        log.info(f"加载完成: {lib.summary()}")
        q = mgr.build_queue(
            lib, priority_by_size=args.priority_by_size,
            require_mask_loaded=not args.lazy_load,
        )
        log.info(f"队列长度: {len(q)}，状态: {q.status_counts()}")
        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            lib.to_csv(out_dir / "cell_list.csv")
            lib.to_json(out_dir / "cell_list.json")
            log.info(f"已导出 cell 清单到 {out_dir}")
        return 0

    # --- 执行 ---
    try:
        layout_options = LayoutLoadOptions(
            layer=layer, **layout_opts_kwargs
        ).__dict__ if layer is not None else layout_opts_kwargs
        layout_options = {
            k: v for k, v in layout_options.items()
            if k in LayoutLoadOptions.__dataclass_fields__
        }
        if layer is not None:
            layout_options['layer'] = layer

        summary, results, lib, queue = run_batch_optimization(
            source=source,
            layer=layer,
            layout_options=layout_options if layer is not None else None,
            resource_config=resource_cfg,
            batch_config=batch_cfg,
            mode=args.mode,
            output_dir=args.output_dir,
        )

        if args.save_cell_list and lib is not None and args.output_dir:
            out_dir = Path(args.output_dir)
            lib.to_csv(out_dir / "cell_list.csv")
            lib.to_json(out_dir / "cell_list.json")

        # 打印摘要
        s = summary
        log.info("\n" + "=" * 60)
        log.info(f"批次 {s.batch_id} 完成")
        log.info(f"总耗时: {s.total_elapsed_sec:.1f}s  "
                 f"任务数: {s.total_tasks}  "
                 f"成功率: {s.success_rate*100:.1f}%")
        log.info(f"状态分布: done={s.done}  fail={s.failed}  "
                 f"cancel={s.cancelled}  timeout={s.timeout}")
        if s.avg_initial_mse is not None:
            log.info(
                f"MSE: {s.avg_initial_mse:.3e} → {s.avg_final_mse:.3e}  "
                f"平均改善: {(s.avg_mse_improvement_ratio or 0)*100:.2f}%"
            )
        log.info(f"收敛: {s.converged_count}/{s.done}  "
                 f"({s.converged_rate*100:.1f}%)  "
                 f"平均耗时 {s.avg_elapsed_sec:.2f}s "
                 f"(中位数 {s.median_elapsed_sec:.2f}s)")
        log.info("=" * 60)
        return 0 if s.failed == 0 else 1
    except KeyboardInterrupt:
        log.warning("用户中断")
        return 130
    except Exception as e:
        log.error(f"批处理执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 4


if __name__ == "__main__":
    sys.exit(_main())
