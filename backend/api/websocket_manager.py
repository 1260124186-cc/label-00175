import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    WebSocket 连接管理器

    管理所有活动的 WebSocket 连接，支持按 task_id 分组推送消息。
    """

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, task_id: str, websocket: WebSocket):
        """
        连接到指定任务的 WebSocket

        Args:
            task_id: 任务 ID
            websocket: WebSocket 连接
        """
        await websocket.accept()
        async with self._lock:
            if task_id not in self.active_connections:
                self.active_connections[task_id] = []
            self.active_connections[task_id].append(websocket)
        logger.info(f"WebSocket 连接已建立: task_id={task_id}, 连接数={len(self.active_connections[task_id])}")

    async def disconnect(self, task_id: str, websocket: WebSocket):
        """
        断开 WebSocket 连接

        Args:
            task_id: 任务 ID
            websocket: WebSocket 连接
        """
        async with self._lock:
            if task_id in self.active_connections:
                if websocket in self.active_connections[task_id]:
                    self.active_connections[task_id].remove(websocket)
                if not self.active_connections[task_id]:
                    del self.active_connections[task_id]
        logger.info(f"WebSocket 连接已断开: task_id={task_id}")

    async def send_personal_message(self, task_id: str, message: Dict[str, Any]):
        """
        向指定任务的所有连接发送消息

        Args:
            task_id: 任务 ID
            message: 消息字典
        """
        if task_id not in self.active_connections:
            return

        connections = list(self.active_connections.get(task_id, []))
        dead_connections = []

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.debug(f"发送 WebSocket 消息失败: {e}")
                dead_connections.append(connection)

        if dead_connections:
            async with self._lock:
                if task_id in self.active_connections:
                    for conn in dead_connections:
                        if conn in self.active_connections[task_id]:
                            self.active_connections[task_id].remove(conn)
                    if not self.active_connections[task_id]:
                        del self.active_connections[task_id]

    def has_connections(self, task_id: str) -> bool:
        """检查指定任务是否有活动连接"""
        return task_id in self.active_connections and len(self.active_connections[task_id]) > 0

    def get_connection_count(self, task_id: str) -> int:
        """获取指定任务的连接数"""
        return len(self.active_connections.get(task_id, []))


manager = ConnectionManager()


async def broadcast_progress(
    task_id: str,
    progress: float,
    message: Optional[str] = None,
    stage: Optional[str] = None,
    loss: Optional[float] = None,
    iteration: Optional[int] = None,
    mask_thumbnail: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
):
    """
    广播进度更新消息

    Args:
        task_id: 任务 ID
        progress: 进度百分比 (0-100)
        message: 进度消息
        stage: 当前阶段
        loss: 当前损失值
        iteration: 当前迭代次数
        mask_thumbnail: 掩模缩略图 (base64)
        extra: 额外数据
    """
    msg = {
        "type": "progress",
        "task_id": task_id,
        "progress": progress,
        "message": message,
        "stage": stage,
        "loss": loss,
        "iteration": iteration,
        "mask_thumbnail": mask_thumbnail,
    }
    if extra:
        msg.update(extra)

    await manager.send_personal_message(task_id, msg)


async def broadcast_stage_change(task_id: str, stage: str, message: Optional[str] = None):
    """
    广播阶段变化消息

    Args:
        task_id: 任务 ID
        stage:新阶段
        message: 描述消息
    """
    msg = {
        "type": "stage_change",
        "task_id": task_id,
        "stage": stage,
        "message": message,
    }
    await manager.send_personal_message(task_id, msg)


async def broadcast_task_complete(task_id: str, result: Optional[Dict[str, Any]] = None):
    """
    广播任务完成消息

    Args:
        task_id: 任务 ID
        result: 任务结果摘要
    """
    msg = {
        "type": "task_complete",
        "task_id": task_id,
        "result": result or {},
    }
    await manager.send_personal_message(task_id, msg)


async def broadcast_task_failed(task_id: str, error: str):
    """
    广播任务失败消息

    Args:
        task_id: 任务 ID
        error: 错误信息
    """
    msg = {
        "type": "task_failed",
        "task_id": task_id,
        "error": error,
    }
    await manager.send_personal_message(task_id, msg)


async def broadcast_heartbeat(task_id: str):
    """广播心跳消息"""
    msg = {
        "type": "heartbeat",
        "task_id": task_id,
    }
    await manager.send_personal_message(task_id, msg)
