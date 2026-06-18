# -*- coding: utf-8 -*-
"""
Litho-Sim 统一命令行主入口

将分散的 run_optimization.py、run_opc.py、run_smo.py、batch_runner_cli.py、
run_experiments.py 聚合为单一 litho-sim CLI（基于 Click）。

子命令：
  - optimize:   通用掩模优化（MaskOptimizer）
  - opc:        OPC 光学邻近校正工作流
  - smo:        SMO 光源-掩模协同优化工作流
  - ilt:        ILT 反演光刻技术工作流
  - batch:      版图库批量优化调度器
  - experiment: 实验编排与回归测试

使用方式：
  python -m backend.cli --help
  python -m backend.cli optimize --help
  python -m backend.cli opc --pattern line_space --cd 45 --grid-size 64x64
"""

import sys
from pathlib import Path

import click

# 确保 backend 目录在 sys.path 中（兼容多种启动方式）
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from .commands.optimize import optimize_cmd
from .commands.opc import opc_cmd
from .commands.smo import smo_cmd
from .commands.ilt import ilt_cmd
from .commands.batch import batch_cmd
from .commands.experiment import experiment_cmd


# ---------------------------------------------------------------------------
# 主 CLI Group
# ---------------------------------------------------------------------------

class LithoSimCLI(click.Group):
    """自定义 Group：在 help 中展示更精美的排版"""

    def format_help(self, ctx, formatter):
        """重写 help 格式，增加 ASCII 横幅"""
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)
        self.format_commands(ctx, formatter)
        self.format_epilog(ctx, formatter)


CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    max_content_width=120,
)


@click.group(
    cls=LithoSimCLI,
    context_settings=CONTEXT_SETTINGS,
    invoke_without_command=True,
    no_args_is_help=True,
    epilog="""
\b
示例：
  # 运行一个简单的掩模优化演示
  litho-sim optimize --pattern rectangle --grid-size 64x64 --max-iter 50

  # OPC 校正一个线/空间结构
  litho-sim opc --pattern line_space --cd 45 --epe-threshold 3.0

  # SMO 交替优化策略
  litho-sim smo --strategy alternating --pattern contact_hole --cd 50

  # ILT 反演光刻（二值掩模）
  litho-sim ilt --transmission binary --pattern l_shaped --perimeter-weight 1e-4

  # 批量处理一个 GDS 目录
  litho-sim batch --source ./layout_library --layer 0 --max-workers 4

  # 运行所有实验并生成 golden
  litho-sim experiment experiments/ --generate-golden

\b
项目主页： https://github.com/lithography/litho-sim
文档：     https://litho-sim.readthedocs.io
    """,
)
@click.version_option(
    version="1.0.0",
    prog_name="litho-sim",
    message="%(prog)s %(version)s — 计算光刻与版图优化仿真框架",
)
@click.option(
    "--show-config-paths",
    is_flag=True,
    default=False,
    help="打印默认配置文件搜索路径后退出",
)
@click.pass_context
def cli(ctx: click.Context, show_config_paths: bool):
    """
    Litho-Sim 计算光刻与版图优化仿真框架

    统一命令行入口，覆盖以下工作流：

    \b
    • 掩模优化 (optimize)      — 通用梯度/启发式掩模优化
    • 光学邻近校正 (opc)        — OPC + SRAF + 热点检测
    • 光源-掩模协同 (smo)       — 交替/联合/先光源 三策略
    • 反演光刻 (ilt)            — 可微成像链 + 梯度投影 + 量化
    • 批处理 (batch)            — GDS 版图库多进程/分布式调度
    • 实验编排 (experiment)     — YAML 驱动实验 + 回归断言
    """
    if show_config_paths:
        config_dir = Path(__file__).resolve().parent.parent / "config"
        click.echo("默认配置文件搜索路径：")
        for name in [
            "default_config.yaml",
            "opc_default.yaml",
            "smo_default.yaml",
            "pipeline_default.yaml",
            "euv_default.yaml",
            "aberration_scenarios.yaml",
        ]:
            p = config_dir / name
            status = "✓" if p.exists() else "✗"
            click.echo(f"  [{status}] {p}")
        ctx.exit(0)


# ---------------------------------------------------------------------------
# 注册子命令
# ---------------------------------------------------------------------------

cli.add_command(optimize_cmd)
cli.add_command(opc_cmd)
cli.add_command(smo_cmd)
cli.add_command(ilt_cmd)
cli.add_command(batch_cmd)
cli.add_command(experiment_cmd)


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------

def run_cli(argv=None):
    """
    编程式调用 CLI 的便捷函数

    Args:
        argv: 命令行参数列表，None 则使用 sys.argv[1:]

    Returns:
        int: 退出码，0 表示成功
    """
    try:
        return cli.main(
            args=argv,
            prog_name="litho-sim",
            standalone_mode=False,
        )
    except click.exceptions.Exit as e:
        return e.exit_code
    except click.ClickException as e:
        e.show()
        return e.exit_code
    except click.Abort:
        click.echo("\n中断", err=True)
        return 130
    except Exception as e:
        click.echo(f"\n未捕获异常: {e}", err=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())
