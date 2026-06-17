# -*- coding: utf-8 -*-
"""
代理模型推理客户端示例

提供多种调用方式，供 OPC 工具或产线系统集成使用:
1. Python HTTP 客户端 (requests)
2. Python 本地直接调用 (无需网络)
3. cURL 命令行示例
4. C++/C# 调用参考

使用方式:
    # 方式 1: HTTP 客户端
    from surrogate.inference_client import SurrogateHttpClient
    client = SurrogateHttpClient("http://localhost:8000")
    aerial = client.predict(mask)

    # 方式 2: 本地直接调用
    from surrogate.inference_client import SurrogateLocalClient
    client = SurrogateLocalClient("./surrogate_checkpoints")
    aerial = client.predict(mask)
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ======================================================================
# HTTP 客户端 (用于远程服务调用)
# ======================================================================

class SurrogateHttpClient:
    """
    HTTP 客户端，通过 REST API 调用远程推理服务

    适用于:
    - 分布式部署，服务运行在独立服务器
    - 多语言环境，不同语言的客户端都可以调用
    - 产线系统集成，通过标准 HTTP 协议通信
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            base_url: 推理服务基础 URL
            timeout: 请求超时时间（秒）
            api_key: 可选的 API 密钥
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_key = api_key
        self.session = self._create_session()

    def _create_session(self):
        """创建 requests Session"""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests 未安装，请安装: pip install requests"
            )

        session = requests.Session()
        if self.api_key:
            session.headers.update({"X-API-Key": self.api_key})
        return session

    def health(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def metadata(self) -> Dict[str, Any]:
        """获取模型元数据"""
        response = self.session.get(
            f"{self.base_url}/metadata",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        response = self.session.get(
            f"{self.base_url}/stats",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        masks: Union[np.ndarray, List[np.ndarray]],
        return_masks: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        预测空间像

        Args:
            masks: 输入掩模，支持:
                - 单张: (H, W) float32/64
                - 批量: (N, H, W) float32/64 或 列表形式
            return_masks: 是否返回原始掩模

        Returns:
            空间像数组:
                - 单张输入: (H, W) float32
                - 批量输入: (N, H, W) float32
        """
        was_single = False
        if isinstance(masks, np.ndarray) and masks.ndim == 2:
            masks_list = [masks.tolist()]
            was_single = True
        elif isinstance(masks, np.ndarray) and masks.ndim == 3:
            masks_list = [masks[i].tolist() for i in range(masks.shape[0])]
        elif isinstance(masks, list):
            masks_list = [m.tolist() if isinstance(m, np.ndarray) else m for m in masks]
        else:
            raise ValueError(f"不支持的输入类型: {type(masks)}, shape={getattr(masks, 'shape', 'N/A')}")

        payload = {
            "masks": masks_list,
            "return_masks": return_masks,
        }

        t0 = time.time()
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()

        logger.debug(
            f"HTTP 推理完成: {result['num_masks']} 张掩模, "
            f"耗时 {result['latency_ms']:.2f}ms, "
            f"网络开销 {(time.time() - t0) * 1000 - result['latency_ms']:.2f}ms"
        )

        outputs = np.array(result['aerial_images'], dtype=np.float32)

        if was_single:
            return outputs[0]
        return outputs

    def predict_batch_file(
        self,
        file_path: str,
        max_batch_size: int = 32,
    ) -> np.ndarray:
        """
        上传 NPY/NPZ 文件进行批量推理

        Args:
            file_path: .npy 或 .npz 文件路径
            max_batch_size: 服务端最大批大小

        Returns:
            (N, H, W) 空间像数组
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}

            response = self.session.post(
                f"{self.base_url}/predict/batch",
                files=files,
                params={'max_batch_size': max_batch_size},
                timeout=self.timeout * 10,
            )
            response.raise_for_status()
            result = response.json()

        logger.info(
            f"批量文件推理完成: {result['num_masks']} 张掩模, "
            f"耗时 {result['latency_ms']:.2f}ms"
        )

        return np.array(result['aerial_images'], dtype=np.float32)

    def reload(self) -> Dict[str, Any]:
        """触发服务端模型热重载"""
        response = self.session.post(
            f"{self.base_url}/reload",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """关闭会话"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ======================================================================
# 本地客户端 (直接加载模型，无需网络)
# ======================================================================

class SurrogateLocalClient:
    """
    本地推理客户端，直接加载 ONNX 或 TorchScript 模型

    适用于:
    - 单机部署，追求最低延迟
    - 离线批量处理
    - 没有网络环境的场景
    """

    def __init__(
        self,
        model_dir: str,
        prefer_onnx: bool = True,
        device: str = 'auto',
    ):
        """
        Args:
            model_dir: 模型目录，包含 model.onnx 或 model.pt 和 metadata.json
            prefer_onnx: 优先使用 ONNX Runtime
            device: 推理设备: 'auto', 'cpu', 'cuda'
        """
        from surrogate.inference_server import SurrogateInferenceEngine

        self.engine = SurrogateInferenceEngine(model_dir, prefer_onnx, device)
        self.model_dir = model_dir

    def predict(
        self,
        masks: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        预测空间像

        Args:
            masks: 输入掩模，支持:
                - 单张: (H, W) float32/64
                - 批量: (N, H, W) float32/64 或 列表形式

        Returns:
            空间像数组:
                - 单张输入: (H, W) float32
                - 批量输入: (N, H, W) float32
        """
        if isinstance(masks, list):
            masks = np.stack(masks, axis=0)

        return self.engine.predict(masks)

    def predict_batch(
        self,
        masks_list: List[np.ndarray],
        max_batch_size: int = 32,
    ) -> List[np.ndarray]:
        """
        批量预测，自动分批处理

        Args:
            masks_list: 掩模列表，每个元素 (H, W)
            max_batch_size: 最大批大小

        Returns:
            空间像列表，与输入一一对应
        """
        return self.engine.predict_batch(masks_list, max_batch_size)

    def metadata(self) -> Dict[str, Any]:
        """获取模型元数据"""
        return self.engine.get_metadata()

    def stats(self) -> Dict[str, Any]:
        """获取推理统计"""
        return self.engine.get_stats()

    def reload(self):
        """热重载模型"""
        self.engine.reload()


# ======================================================================
# 示例用法
# ======================================================================

def example_http_client():
    """HTTP 客户端使用示例"""
    print("=" * 60)
    print("示例 1: HTTP 客户端调用")
    print("=" * 60)

    mask = np.random.rand(128, 128).astype(np.float32)

    try:
        with SurrogateHttpClient("http://localhost:8000") as client:
            health = client.health()
            print(f"服务状态: {health['status']}")
            print(f"推理后端: {health.get('backend')}")

            metadata = client.metadata()
            print(f"模型类型: {metadata.get('model', {}).get('type')}")

            t0 = time.time()
            aerial = client.predict(mask)
            elapsed = (time.time() - t0) * 1000

            print(f"单张推理完成: 形状={aerial.shape}, 耗时={elapsed:.2f}ms")

            batch_masks = np.random.rand(4, 128, 128).astype(np.float32)
            t0 = time.time()
            aerials = client.predict(batch_masks)
            elapsed = (time.time() - t0) * 1000

            print(f"批量推理完成: 形状={aerials.shape}, 耗时={elapsed:.2f}ms")

            stats = client.stats()
            print(f"服务统计: 总请求={stats['total_requests']}, "
                  f"吞吐量={stats['throughput_masks_per_second']:.1f} 张/秒")

    except Exception as e:
        print(f"HTTP 客户端示例失败: {e}")
        print("请先启动推理服务:")
        print("  python -m surrogate.inference_server --model-dir ./surrogate_checkpoints")


def example_local_client():
    """本地客户端使用示例"""
    print("\n" + "=" * 60)
    print("示例 2: 本地客户端调用")
    print("=" * 60)

    model_dir = "./surrogate_checkpoints"
    mask = np.random.rand(128, 128).astype(np.float32)

    try:
        client = SurrogateLocalClient(model_dir)

        metadata = client.metadata()
        print(f"推理后端: {metadata['backend']}")
        print(f"模型类型: {metadata.get('model', {}).get('type')}")

        t0 = time.time()
        aerial = client.predict(mask)
        elapsed = (time.time() - t0) * 1000

        print(f"单张推理完成: 形状={aerial.shape}, 耗时={elapsed:.2f}ms")

        batch_masks = np.random.rand(8, 128, 128).astype(np.float32)
        t0 = time.time()
        aerials = client.predict(batch_masks)
        elapsed = (time.time() - t0) * 1000

        print(f"批量推理完成: 形状={aerials.shape}, 耗时={elapsed:.2f}ms, "
              f"{elapsed / 8:.2f}ms/张")

        stats = client.stats()
        print(f"本地统计: 总请求={stats['total_requests']}, "
              f"吞吐量={stats['throughput_masks_per_second']:.1f} 张/秒")

    except Exception as e:
        print(f"本地客户端示例失败: {e}")
        print("请先训练并导出模型:")
        print("  python -c \"from surrogate.train import train_surrogate_model; train_surrogate_model()\"")


def example_curl_commands():
    """cURL 命令行示例"""
    print("\n" + "=" * 60)
    print("示例 3: cURL 命令行调用")
    print("=" * 60)

    print("""
# 健康检查
curl http://localhost:8000/health

# 获取元数据
curl http://localhost:8000/metadata

# 获取统计信息
curl http://localhost:8000/stats

# 单张掩模推理（创建测试数据）
python3 -c "
import json
import numpy as np
mask = np.random.rand(128, 128).tolist()
payload = {'masks': [mask]}
print(json.dumps(payload))
" > /tmp/payload.json

curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d @/tmp/payload.json

# 批量文件推理
curl -X POST "http://localhost:8000/predict/batch?max_batch_size=32" \\
  -F "file=@masks.npy"

# 热重载模型
curl -X POST http://localhost:8000/reload
    """)


def example_opc_integration():
    """
    OPC 工具集成示例

    展示如何在 OPC 优化循环中使用代理模型替换传统仿真
    """
    print("\n" + "=" * 60)
    print("示例 4: OPC 工具集成")
    print("=" * 60)

    print("""
在 OPC 优化中，传统流程:
    for iteration in range(max_iterations):
        aerial = rigorous_simulation(mask)  # 慢！
        loss = compute_loss(aerial, target)
        gradient = compute_gradient(loss, mask)
        mask = update_mask(mask, gradient)

使用代理模型后:
    from surrogate.inference_client import SurrogateLocalClient

    surrogate = SurrogateLocalClient("./surrogate_checkpoints")

    for iteration in range(max_iterations):
        # 快速推理，加速 10-100x
        aerial = surrogate.predict(mask)
        loss = compute_loss(aerial, target)

        # 梯度计算也可以使用代理模型
        # gradient = surrogate.compute_gradient(mask, target)
        gradient = compute_gradient(loss, mask)
        mask = update_mask(mask, gradient)

        # 定期验证精度
        if iteration % 100 == 0:
            real_aerial = rigorous_simulation(mask)
            error = np.mean((aerial - real_aerial) ** 2)
            if error > threshold:
                print(f\"Warning: surrogate error too high: {error}\")
    """)


def main():
    """运行所有示例"""
    logging.basicConfig(level=logging.INFO)

    example_http_client()
    example_local_client()
    example_curl_commands()
    example_opc_integration()


if __name__ == "__main__":
    main()
