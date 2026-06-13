# -*- coding: utf-8 -*-
"""
算法模块：优化器、掩模优化
"""

from algorithms.optimizer import GradientDescentOptimizer, BFGSOptimizer, NewtonOptimizer
from algorithms.advanced_optimizer import (
    BaseHeuristicOptimizer, GeneticAlgorithmOptimizer, ParticleSwarmOptimizer,
    ReinforcementLearningOptimizer, SimpleQLearningModel
)
from algorithms.deep_rl_models import (
    TORCH_AVAILABLE,
    MultiChannelStateEncoder,
    StateEncoderConfig,
    DQNModel,
    DQNConfig,
    PPOModel,
    PPOConfig,
    ActorCriticModel,
    ActorCriticConfig,
    DeepRLModelFactory,
)
from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig, MaskOptimizationResult
from algorithms.callbacks import (
    Callback, CallbackList, TrainerState,
    LearningRateSchedulerCallback, EarlyStoppingCallback,
    ModelCheckpointCallback, MaskSnapshotCallback,
    ConvergencePlotCallback, LoggerCallback, HistoryCallback,
    LambdaCallback, AnimationCallback, ExperimentTrackingCallback
)

__all__ = [
    'GradientDescentOptimizer',
    'BFGSOptimizer',
    'NewtonOptimizer',
    'BaseHeuristicOptimizer',
    'GeneticAlgorithmOptimizer',
    'ParticleSwarmOptimizer',
    'ReinforcementLearningOptimizer',
    'SimpleQLearningModel',
    'TORCH_AVAILABLE',
    'MultiChannelStateEncoder',
    'StateEncoderConfig',
    'DQNModel',
    'DQNConfig',
    'PPOModel',
    'PPOConfig',
    'ActorCriticModel',
    'ActorCriticConfig',
    'DeepRLModelFactory',
    'MaskOptimizer',
    'OptimizationConfig',
    'MaskOptimizationResult',
    'Callback',
    'CallbackList',
    'TrainerState',
    'LearningRateSchedulerCallback',
    'EarlyStoppingCallback',
    'ModelCheckpointCallback',
    'MaskSnapshotCallback',
    'ConvergencePlotCallback',
    'LoggerCallback',
    'HistoryCallback',
    'LambdaCallback',
    'AnimationCallback',
    'ExperimentTrackingCallback',
]
