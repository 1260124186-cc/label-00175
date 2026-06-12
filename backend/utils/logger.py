# -*- coding: utf-8 -*-
"""
日志模块：日志配置与管理

该模块提供统一的日志配置，用于记录优化迭代过程。
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(name: str = 'litho_sim',
                 level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 console: bool = True,
                 file_level: int = logging.DEBUG) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 控制台日志级别
        log_file: 日志文件路径，None则不写入文件
        console: 是否输出到控制台
        file_level: 文件日志级别
        
    Returns:
        配置好的Logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置最低级别
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 日志格式
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'litho_sim') -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        Logger对象
    """
    return logging.getLogger(name)


class OptimizationLogger:
    """
    优化过程专用日志记录器
    
    提供结构化的优化迭代日志记录。
    """
    
    def __init__(self, 
                 name: str = 'optimization',
                 log_dir: Optional[str] = None):
        """
        初始化优化日志记录器
        
        Args:
            name: 日志名称
            log_dir: 日志目录
        """
        self.name = name
        
        if log_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = Path(log_dir) / f'{name}_{timestamp}.log'
        else:
            log_file = None
        
        self.logger = setup_logger(
            name=f'litho_sim.{name}',
            log_file=str(log_file) if log_file else None
        )
        
        self.iteration_count = 0
        self.start_time = None
    
    def start(self, config: dict = None):
        """
        记录优化开始
        
        Args:
            config: 优化配置字典
        """
        self.start_time = datetime.now()
        self.iteration_count = 0
        
        self.logger.info("=" * 60)
        self.logger.info("优化开始")
        self.logger.info(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if config:
            self.logger.info("配置参数:")
            for key, value in config.items():
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=" * 60)
    
    def log_iteration(self, 
                      iteration: int,
                      loss: float,
                      metrics: dict = None,
                      extra_info: str = None):
        """
        记录迭代信息
        
        Args:
            iteration: 迭代次数
            loss: 当前损失值
            metrics: 其他指标字典
            extra_info: 额外信息
        """
        self.iteration_count = iteration
        
        msg = f"迭代 {iteration:4d} | 损失: {loss:.6e}"
        
        if metrics:
            for key, value in metrics.items():
                msg += f" | {key}: {value:.6e}"
        
        if extra_info:
            msg += f" | {extra_info}"
        
        self.logger.info(msg)
    
    def end(self, 
            final_loss: float,
            final_metrics: dict = None,
            success: bool = True,
            message: str = None):
        """
        记录优化结束
        
        Args:
            final_loss: 最终损失值
            final_metrics: 最终指标
            success: 是否成功
            message: 结束消息
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.logger.info("=" * 60)
        self.logger.info("优化结束")
        self.logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"总耗时: {duration:.2f} 秒")
        self.logger.info(f"总迭代: {self.iteration_count}")
        self.logger.info(f"最终损失: {final_loss:.6e}")
        
        if final_metrics:
            self.logger.info("最终指标:")
            for key, value in final_metrics.items():
                self.logger.info(f"  {key}: {value:.6e}")
        
        self.logger.info(f"状态: {'成功' if success else '失败'}")
        
        if message:
            self.logger.info(f"消息: {message}")
        
        self.logger.info("=" * 60)
    
    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误"""
        self.logger.error(message)
