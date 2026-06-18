# -*- coding: utf-8 -*-
"""
Litho-Sim 统一命令行入口模块

提供基于 Click 的统一 CLI，子命令包括：
  - optimize: 通用掩模优化（MaskOptimizer）
  - opc:      OPC 光学邻近校正工作流
  - smo:      SMO 光源-掩模协同优化工作流
  - ilt:      ILT 反演光刻技术工作流
  - batch:    批处理调度器（版图库批量优化）
  - experiment: 实验编排与回归测试

使用方式：
  python -m backend.cli --help
  python -m backend.cli optimize --help
  litho-sim optimize --help          # 安装为 console script 后
"""

from .main import cli, run_cli

__version__ = "1.0.0"
__all__ = ["cli", "run_cli"]
