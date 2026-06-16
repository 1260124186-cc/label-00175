import logging
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from websocket_manager import manager, broadcast_heartbeat

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["WebSocket"])


@router.websocket("/tasks/{task_id}")
async def websocket_task_progress(
    websocket: WebSocket,
    task_id: str,
):
    """
    WebSocket 端点：实时获取任务进度

    连接到指定任务的 WebSocket 通道，实时接收进度更新、阶段变化、
    损失值、掩模缩略图等信息。

    消息类型：
        - progress: 进度更新
        - stage_change: 阶段变化
        - task_complete: 任务完成
        - task_failed: 任务失败
        - heartbeat: 心跳

    Args:
        task_id: 任务 ID
    """
    await manager.connect(task_id, websocket)

    try:
        # 发送初始连接确认
        await websocket.send_json({
            "type": "connected",
            "task_id": task_id,
            "message": "WebSocket 连接已建立",
        })

        # 心跳循环
        heartbeat_task = asyncio.create_task(_heartbeat_loop(task_id, websocket))

        # 监听客户端消息
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # 处理客户端消息（如 ping）
                try:
                    import json
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "task_id": task_id,
                        })
                except (json.JSONDecodeError, ValueError):
                    pass
            except asyncio.TimeoutError:
                # 超时检查连接是否还活着
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "task_id": task_id,
                    })
                except Exception:
                    break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.debug(f"WebSocket 接收消息异常: {e}")
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 客户端断开连接: task_id={task_id}")
    except Exception as e:
        logger.error(f"WebSocket 连接异常: task_id={task_id}, error={e}")
    finally:
        await manager.disconnect(task_id, websocket)


async def _heartbeat_loop(task_id: str, websocket: WebSocket):
    """心跳循环 - 定期发送心跳保持连接"""
    try:
        while True:
            await asyncio.sleep(15)
            if manager.has_connections(task_id):
                await broadcast_heartbeat(task_id)
            else:
                break
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.debug(f"心跳循环异常: {e}")
