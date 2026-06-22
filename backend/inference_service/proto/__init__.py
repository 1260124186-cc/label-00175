# -*- coding: utf-8 -*-
"""
gRPC Protocol Buffers 定义

使用说明:
1. 安装编译工具: pip install grpcio-tools
2. 编译 (在 inference_service 目录下):
   python -m grpc_tools.protoc -I./proto \
     --python_out=. --grpc_python_out=. \
     --pyi_out=. proto/inference.proto

或者使用 services 提供的动态编译功能自动生成。
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROTO_DIR = Path(__file__).parent.resolve()
PROTO_FILE = PROTO_DIR / "inference.proto"

GRPC_AVAILABLE = False
_inference_pb2 = None
_inference_pb2_grpc = None


def _try_import_compiled():
    """尝试导入已编译的 pb2 模块"""
    global GRPC_AVAILABLE, _inference_pb2, _inference_pb2_grpc
    try:
        import grpc

        parent_dir = str(PROTO_DIR.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        proto_parent = str(PROTO_DIR)
        if proto_parent not in sys.path:
            sys.path.insert(0, proto_parent)

        try:
            import inference_pb2
            import inference_pb2_grpc
            _inference_pb2 = inference_pb2
            _inference_pb2_grpc = inference_pb2_grpc
            GRPC_AVAILABLE = True
            logger.debug("使用预编译的 gRPC 模块")
            return True
        except ImportError:
            pass

        try:
            from . import inference_pb2
            from . import inference_pb2_grpc
            _inference_pb2 = inference_pb2
            _inference_pb2_grpc = inference_pb2_grpc
            GRPC_AVAILABLE = True
            logger.debug("使用包内预编译的 gRPC 模块")
            return True
        except ImportError:
            pass

    except ImportError:
        logger.debug("grpcio 未安装")
    return False


def compile_proto(force: bool = False) -> bool:
    """
    动态编译 proto 文件 (运行时)

    Args:
        force: 是否强制重新编译

    Returns:
        是否编译成功
    """
    global GRPC_AVAILABLE, _inference_pb2, _inference_pb2_grpc

    if not force and GRPC_AVAILABLE and _inference_pb2 is not None:
        return True

    try:
        from grpc_tools import protoc
    except ImportError:
        logger.warning("grpcio-tools 未安装，无法编译 proto")
        return False

    if not PROTO_FILE.exists():
        logger.error(f"proto 文件不存在: {PROTO_FILE}")
        return False

    output_dir = str(PROTO_DIR)
    proto_path = str(PROTO_DIR)

    original_dir = os.getcwd()
    try:
        os.chdir(proto_path)
        protoc.main(
            (
                "grpc_tools.protoc",
                f"--proto_path={proto_path}",
                f"--python_out={output_dir}",
                f"--grpc_python_out={output_dir}",
                f"--pyi_out={output_dir}",
                "inference.proto",
            )
        )
        logger.info("proto 文件编译成功")
    except Exception as e:
        logger.error(f"编译 proto 失败: {e}")
        return False
    finally:
        os.chdir(original_dir)

    return _try_import_compiled()


def ensure_grpc() -> bool:
    """确保 gRPC 模块可用"""
    if _try_import_compiled():
        return True
    return compile_proto()


def get_pb2():
    """获取编译后的 pb2 模块"""
    ensure_grpc()
    return _inference_pb2


def get_pb2_grpc():
    """获取编译后的 grpc pb2_grpc 模块"""
    ensure_grpc()
    return _inference_pb2_grpc


__all__ = [
    "PROTO_DIR",
    "PROTO_FILE",
    "GRPC_AVAILABLE",
    "compile_proto",
    "ensure_grpc",
    "get_pb2",
    "get_pb2_grpc",
]
