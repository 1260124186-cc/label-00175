# -*- coding: utf-8 -*-
"""
Pytest 根配置

确保从仓库任意目录运行 pytest 时，backend/ 在 sys.path 中，
使得 core / pipeline / workflows 等裸导入正常工作。
"""

import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
