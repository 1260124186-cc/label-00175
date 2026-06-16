"""
WebSocket 进度回调

集成到优化器的回调系统中，在每一步迭代时通过 WebSocket 推送
损失值、当前掩模缩略图和阶段状态等实时信息。
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class WebSocketProgressCallback:
    """
    WebSocket 进度回调
    
    在优化迭代过程中通过 WebSocket 推送实时进度信息。
    支持与 MaskOptimizer 的 Callback 系统以及 OPC/SMO 工作流集成。
    """
    
    def __init__(
        self,
        task_id: str,
        total_iterations: int = 100,
        progress_range: tuple = (0.0, 100.0),
        stage: str = "optimization",
        push_frequency: int = 1,
        include_mask_thumbnail: bool = True,
    ):
        """
        初始化 WebSocket 进度回调
        
        Args:
            task_id: 任务 ID
            total_iterations: 总迭代次数
            progress_range: 进度范围 (起始百分比, 结束百分比)
            stage: 阶段名称
            push_frequency: 推送频率（每 N 次迭代推送一次）
            include_mask_thumbnail: 是否包含掩模缩略图
        """
        self.task_id = task_id
        self.total_iterations = max(1, total_iterations)
        self.progress_start, self.progress_end = progress_range
        self.stage = stage
        self.push_frequency = max(1, push_frequency)
        self.include_mask_thumbnail = include_mask_thumbnail
        
        self.current_iteration = 0
        self.current_loss: Optional[float] = None
        self._ws_manager_checked = False
        self._ws_available = False
    
    def _check_ws_available(self) -> bool:
        """检查 WebSocket 是否可用（有客户端连接）"""
        try:
            from websocket_manager import manager
            self._ws_available = manager.has_connections(self.task_id)
        except Exception as e:
            logger.debug(f"检查 WebSocket 连接失败: {e}")
            self._ws_available = False
        self._ws_manager_checked = True
        return self._ws_available
    
    def _compute_progress(self, iteration: int) -> float:
        """计算当前进度百分比"""
        ratio = min(1.0, max(0.0, iteration / self.total_iterations))
        return self.progress_start + ratio * (self.progress_end - self.progress_start)
    
    def _push_progress(
        self,
        iteration: int,
        loss: Optional[float] = None,
        mask: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """推送进度更新"""
        # 每 N 次检查一次 WebSocket 连接状态
        if iteration % self.push_frequency != 0 and iteration > 0:
            return
        
        # 定期重新检查连接状态
        if iteration % 10 == 0:
            self._ws_manager_checked = False
        
        if not self._ws_manager_checked:
            if not self._check_ws_available():
                return
        elif not self._ws_available:
            return
        
        progress = self._compute_progress(iteration)
        
        try:
            from services import _set_progress
            _set_progress(
                task_id=self.task_id,
                progress=progress,
                message=f"{self.stage} - 第 {iteration} 次迭代",
                stage=self.stage,
                loss=loss,
                iteration=iteration,
                mask=mask if self.include_mask_thumbnail else None,
                extra=extra,
            )
        except Exception as e:
            logger.debug(f"推送 WebSocket 进度失败: {e}")
    
    # ------------------------------------------------------------
    # 优化器回调接口 (Callback 兼容)
    # ------------------------------------------------------------
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始"""
        self.current_iteration = 0
        self._push_progress(0, mask=getattr(logs, 'mask', None) if logs else None)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束"""
        loss = logs.get('loss') if logs else None
        self._push_progress(
            self.total_iterations,
            loss=loss,
            mask=getattr(logs, 'mask', None) if logs else None,
        )
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 epoch 开始"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 epoch 结束"""
        self.current_iteration = epoch + 1
        loss = logs.get('loss') if logs else None
        self.current_loss = loss
        
        # 从 state 中获取 mask
        mask = None
        if hasattr(self, 'state') and self.state is not None:
            mask = getattr(self.state, 'mask', None)
        elif logs:
            mask = logs.get('mask')
        
        self._push_progress(epoch + 1, loss=loss, mask=mask, extra=logs)
    
    def set_state(self, state):
        """设置训练状态引用（Callback 接口）"""
        self.state = state
    
    def set_params(self, params: Dict[str, Any]):
        """设置参数（Callback 接口）"""
        self.params = params
    
    # ------------------------------------------------------------
    # OPC 工作流接口
    # ------------------------------------------------------------
    
    def on_opc_iteration_begin(self, iteration: int):
        """OPC 迭代开始"""
        pass
    
    def on_opc_iteration_end(
        self,
        iteration: int,
        epe_before: Optional[Dict[str, float]] = None,
        epe_after: Optional[Dict[str, float]] = None,
        mask_before: Optional[Any] = None,
        mask_after: Optional[Any] = None,
        hotspots_before_count: int = 0,
        hotspots_after_count: int = 0,
    ):
        """OPC 迭代结束"""
        self.current_iteration = iteration
        
        loss = None
        if epe_after and 'epe_mean' in epe_after:
            loss = epe_after['epe_mean']
        
        extra = {
            'epe_before': epe_before,
            'epe_after': epe_after,
            'hotspots_before': hotspots_before_count,
            'hotspots_after': hotspots_after_count,
        }
        
        self._push_progress(iteration, loss=loss, mask=mask_after, extra=extra)
    
    # ------------------------------------------------------------
    # SMO 工作流接口
    # ------------------------------------------------------------
    
    def on_smo_outer_iteration_begin(self, iteration: int, phase: str):
        """SMO 外层迭代开始"""
        self.stage = f"smo_{phase}"
    
    def on_smo_outer_iteration_end(
        self,
        iteration: int,
        phase: str,
        loss_before: float,
        loss_after: float,
        mask_before: Optional[Any] = None,
        mask_after: Optional[Any] = None,
        source_before: Optional[Any] = None,
        source_after: Optional[Any] = None,
    ):
        """SMO 外层迭代结束"""
        self.current_iteration = iteration
        
        extra = {
            'phase': phase,
            'loss_before': loss_before,
            'loss_after': loss_after,
        }
        
        self._push_progress(iteration, loss=loss_after, mask=mask_after, extra=extra)
    
    # ------------------------------------------------------------
    # 通用阶段接口
    # ------------------------------------------------------------
    
    def set_stage(self, stage: str, message: Optional[str] = None):
        """设置当前阶段"""
        self.stage = stage
        try:
            from services import _push_stage_change_ws
            _push_stage_change_ws(self.task_id, stage, message)
        except Exception as e:
            logger.debug(f"推送阶段变化失败: {e}")
