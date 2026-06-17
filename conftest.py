# -*- coding: utf-8 -*-
"""
仓库根目录 Pytest 配置

确保从仓库根目录运行 pytest 时:
1. backend/ 在 sys.path 中（裸导入 core / pipeline / workflows 生效）
2. 不会把 backend 当作 Python 包触发 backend/__init__.py 的相对导入
"""

import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent / "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)


def pytest_collection_modifyitems(items):
    pass
