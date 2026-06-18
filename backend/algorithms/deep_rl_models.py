# -*- coding: utf-8 -*-
"""
深度强化学习模型模块：DQN / PPO / Actor-Critic 及多通道状态编码器

依赖 PyTorch，若未安装则相关类在实例化时抛出 ImportError。

新增功能：
1. 合成测试结构库环境 (SyntheticTestStructureEnv) - 用于预训练
2. 预训练方法 pretrain_on_synthetic() - 在合成结构库上预训练策略网络
3. 权重保存/加载 save_weights()/load_weights() - 迁移学习支持
4. 微调方法 fine_tune() - 对特定目标版图微调
5. 预训练配置 PretrainConfig - 预训练超参数配置
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque
import logging
import os
import copy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep RL models. "
            "Install it via: pip install torch"
        )


@dataclass
class StateEncoderConfig:
    patch_size: int = 8
    history_length: int = 16
    freq_bins: int = 32
    normalize: bool = True


class MultiChannelStateEncoder:
    """
    多通道状态编码器

    将掩模和目标图像编码为三通道表示：
      - 通道 1: 局部 patch 特征（误差图的局部窗口均值/方差）
      - 通道 2: 频域特征（误差图 FFT 幅度谱的低频部分）
      - 通道 3: 历史损失轨迹（滑动窗口）
    """

    def __init__(self, config: Optional[StateEncoderConfig] = None):
        self.config = config or StateEncoderConfig()
        self._loss_history: deque = deque(maxlen=self.config.history_length)

    def reset(self):
        self._loss_history.clear()

    def record_loss(self, loss: float):
        self._loss_history.append(loss)

    def encode(self,
               mask: np.ndarray,
               target: np.ndarray,
               error_map: Optional[np.ndarray] = None) -> np.ndarray:
        if error_map is None:
            error_map = np.abs(mask.astype(np.float64) - target.astype(np.float64))

        ch_patch = self._encode_patch_channel(error_map)
        ch_freq = self._encode_freq_channel(error_map)
        ch_hist = self._encode_history_channel(mask.shape)

        state = np.stack([ch_patch, ch_freq, ch_hist], axis=0).astype(np.float32)

        if self.config.normalize:
            for c in range(3):
                ch = state[c]
                std = ch.std()
                if std > 1e-8:
                    state[c] = (ch - ch.mean()) / std

        return state

    def _encode_patch_channel(self, error_map: np.ndarray) -> np.ndarray:
        p = self.config.patch_size
        h, w = error_map.shape
        padded = np.pad(error_map, ((p // 2, p // 2), (p // 2, p // 2)), mode='reflect')

        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, (p, p))
        local_mean = windows.mean(axis=(-2, -1))
        local_var = windows.var(axis=(-2, -1))

        ch = 0.5 * local_mean + 0.5 * local_var

        if ch.shape != (h, w):
            ch = ch[:h, :w]

        return ch.astype(np.float32)

    def _encode_freq_channel(self, error_map: np.ndarray) -> np.ndarray:
        fft = np.fft.fft2(error_map)
        magnitude = np.abs(np.fft.fftshift(fft))
        log_mag = np.log1p(magnitude)

        h, w = error_map.shape
        n_bins = self.config.freq_bins
        radius = np.sqrt(
            (np.arange(h)[:, None] - h // 2) ** 2 +
            (np.arange(w)[None, :] - w // 2) ** 2
        )
        max_r = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
        bin_edges = np.linspace(0, max_r, n_bins + 1)

        radial_profile = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_bins):
            mask_r = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
            if mask_r.any():
                radial_profile[i] = log_mag[mask_r].mean()

        profile_norm = radial_profile / (radial_profile.max() + 1e-8)
        bin_idx = np.clip((radius / max_r * n_bins).astype(int), 0, n_bins - 1)
        ch = profile_norm[bin_idx]

        return ch.astype(np.float32)

    def _encode_history_channel(self, shape: Tuple[int, ...]) -> np.ndarray:
        h, w = shape
        if len(self._loss_history) == 0:
            return np.zeros((h, w), dtype=np.float32)

        losses = np.array(self._loss_history, dtype=np.float64)
        losses_norm = losses / (losses.max() + 1e-8)

        row = np.zeros(w, dtype=np.float32)
        n = min(len(losses_norm), w)
        step = max(1, w // n)
        for i in range(n):
            idx = min(i * step, w - 1)
            row[idx] = losses_norm[i]

        ch = np.tile(row, (h, 1))
        return ch

    @property
    def state_channels(self) -> int:
        return 3

    @property
    def encoded_shape(self) -> Tuple[int, int, int]:
        return (3, 0, 0)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class RolloutBuffer:
    """PPO 的 on-policy 轨迹缓冲"""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def push(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma: float, lam: float = 0.95) -> Tuple:
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)


# ─── PyTorch 依赖组件 ──────────────────────────────────

if TORCH_AVAILABLE:

    class _ConvEncoder(nn.Module):
        """小型卷积特征提取器，输入 (B, 3, H, W) → 输出 (B, feat_dim)"""

        def __init__(self, feat_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, feat_dim),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class DQNNetwork(nn.Module):
        """Deep Q-Network：状态 → Q 值（离散动作空间）"""

        def __init__(self, num_actions: int, feat_dim: int = 128):
            super().__init__()
            self.encoder = _ConvEncoder(feat_dim)
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.encoder(x)
            return self.head(feat)

    class ActorCriticNetwork(nn.Module):
        """Actor-Critic 网络：共享卷积编码器 + 双头输出"""

        def __init__(self, action_dim: int, feat_dim: int = 128):
            super().__init__()
            self.encoder = _ConvEncoder(feat_dim)
            self.actor = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            self._log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, x: torch.Tensor):
            feat = self.encoder(x)
            mean = torch.tanh(self.actor(feat))
            std = F.softplus(self._log_std).expand_as(mean)
            value = self.critic(feat)
            return mean, std, value

    class PPOActorCritic(nn.Module):
        """PPO 使用的 Actor-Critic 网络"""

        def __init__(self, action_dim: int, feat_dim: int = 128):
            super().__init__()
            self.encoder = _ConvEncoder(feat_dim)
            self.actor = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            self._log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, x: torch.Tensor):
            feat = self.encoder(x)
            mean = torch.tanh(self.actor(feat))
            std = F.softplus(self._log_std).expand_as(mean)
            value = self.critic(feat)
            return mean, std, value

        def get_action(self, x: torch.Tensor):
            mean, std, value = self.forward(x)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            return action, log_prob, value

        def evaluate(self, x: torch.Tensor, action: torch.Tensor):
            mean, std, value = self.forward(x)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1)
            return log_prob, value, entropy


# ─── Config dataclasses（不依赖 PyTorch） ──────────────


@dataclass
class DQNConfig:
    num_actions: int = 9
    feat_dim: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 500
    target_update_freq: int = 10
    buffer_capacity: int = 10000
    batch_size: int = 32


@dataclass
class ActorCriticConfig:
    action_dim: int = 4
    feat_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5


@dataclass
class PPOConfig:
    action_dim: int = 4
    feat_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 16


@dataclass
class PretrainConfig:
    """
    深度强化学习迁移预训练配置

    用于控制在合成测试结构库上的预训练过程。

    Attributes:
        total_episodes: 预训练总回合数
        max_steps_per_episode: 每回合最大步数
        grid_size_range: 版图网格尺寸范围 [(ny_min, nx_min), (ny_max, nx_max)]
        cd_range: 关键尺寸范围 (nm)
        pitch_range: 间距范围 (nm)
        structure_types: 使用的测试结构类型列表
        noise_level: 初始掩模噪声水平 (0~1)
        reward_scale: 奖励缩放因子
        save_freq: 保存检查点的频率（回合数）
        checkpoint_dir: 检查点保存目录
        log_freq: 日志输出频率（回合数）
        use_curriculum: 是否使用课程学习（从简单到复杂）
    """
    total_episodes: int = 1000
    max_steps_per_episode: int = 50
    grid_size_range: Tuple[Tuple[int, int], Tuple[int, int]] = ((32, 32), (128, 128))
    cd_range: Tuple[float, float] = (32.0, 90.0)
    pitch_range: Tuple[float, float] = (64.0, 180.0)
    structure_types: List[str] = field(default_factory=lambda: [
        "line_space", "contact_hole", "l_shaped_corner", "t_junction", "sram_bitcell"
    ])
    noise_level: float = 0.2
    reward_scale: float = 100.0
    save_freq: int = 100
    checkpoint_dir: str = "./pretrained_checkpoints"
    log_freq: int = 50
    use_curriculum: bool = True


@dataclass
class FinetuneConfig:
    """
    微调配置

    Attributes:
        total_episodes: 微调总回合数
        max_steps_per_episode: 每回合最大步数
        lr_factor: 学习率缩放因子（相对于预训练）
        freeze_encoder: 是否冻结编码器层（只微调头部）
    """
    total_episodes: int = 200
    max_steps_per_episode: int = 100
    lr_factor: float = 0.1
    freeze_encoder: bool = False


class SyntheticTestStructureEnv:
    """
    合成测试结构环境 - 用于 RL 预训练

    在每次重置时，随机生成一种测试结构作为目标版图，
    并在目标版图上添加噪声作为初始掩模。Agent 的目标是
    通过一系列动作（掩模调整）将初始掩模调整为接近目标。

    支持课程学习：前期使用大尺寸、简单结构，后期逐渐增加难度。
    """

    def __init__(self, pretrain_config: Optional[PretrainConfig] = None):
        self.config = pretrain_config or PretrainConfig()
        self._current_episode = 0

        try:
            from core.test_structures import (
                TestStructureType,
                LineSpaceParams,
                ContactHoleParams,
                LShapedCornerParams,
                TJunctionParams,
                SRAMBitcellParams,
                LineOrientation,
                HolePattern,
                generate_test_structure,
            )
            self._ts_modules = {
                'generate': generate_test_structure,
                'TestStructureType': TestStructureType,
                'LineSpaceParams': LineSpaceParams,
                'ContactHoleParams': ContactHoleParams,
                'LShapedCornerParams': LShapedCornerParams,
                'TJunctionParams': TJunctionParams,
                'SRAMBitcellParams': SRAMBitcellParams,
                'LineOrientation': LineOrientation,
                'HolePattern': HolePattern,
            }
        except ImportError as e:
            logger.warning(f"无法导入 test_structures 模块: {e}，将使用备用生成器")
            self._ts_modules = None

        self._state_encoder = MultiChannelStateEncoder()

        self.target: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.best_loss: float = float('inf')
        self.current_loss: float = float('inf')
        self._step_count: int = 0

    def _sample_grid_size(self) -> Tuple[int, int]:
        """根据课程学习进度采样网格尺寸"""
        (ny_min, nx_min), (ny_max, nx_max) = self.config.grid_size_range
        if self.config.use_curriculum:
            progress = min(self._current_episode / max(self.config.total_episodes, 1), 1.0)
        else:
            progress = np.random.random()

        ny = int(ny_min + progress * (ny_max - ny_min) + np.random.randint(-8, 9))
        nx = int(nx_min + progress * (nx_max - nx_min) + np.random.randint(-8, 9))
        ny = max(16, min(ny, ny_max))
        nx = max(16, min(nx, nx_max))
        ny = ny // 8 * 8
        nx = nx // 8 * 8
        return (ny, nx)

    def _sample_structure_params(self, grid_size: Tuple[int, int]) -> Any:
        """随机采样测试结构参数"""
        cd = np.random.uniform(*self.config.cd_range)
        pitch = cd * np.random.uniform(1.5, 3.0)
        pitch = max(pitch, self.config.pitch_range[0])
        pitch = min(pitch, self.config.pitch_range[1])

        available_types = [t for t in self.config.structure_types]
        if self.config.use_curriculum:
            progress = min(self._current_episode / max(self.config.total_episodes, 1), 1.0)
            easy = ["line_space", "contact_hole"]
            hard = ["l_shaped_corner", "t_junction", "sram_bitcell"]
            if progress < 0.3:
                pool = easy
            elif progress < 0.7:
                pool = easy + hard
            else:
                pool = available_types
            available_types = [t for t in pool if t in self.config.structure_types]
            if not available_types:
                available_types = ["line_space"]

        structure_type = np.random.choice(available_types)

        pixel_size = 1.0

        if self._ts_modules is not None:
            TSType = self._ts_modules['TestStructureType']
            type_map = {
                'line_space': TSType.LINE_SPACE,
                'contact_hole': TSType.CONTACT_HOLE,
                'l_shaped_corner': TSType.L_SHAPED_CORNER,
                't_junction': TSType.T_JUNCTION,
                'sram_bitcell': TSType.SRAM_BITCELL,
            }

            base_params = {
                'grid_size': grid_size,
                'pixel_size': pixel_size,
                'cd': cd,
                'pitch': pitch,
                'structure_type': type_map.get(structure_type, TSType.LINE_SPACE),
            }

            if structure_type == 'line_space':
                orientation = np.random.choice(['horizontal', 'vertical'])
                LineOrientation = self._ts_modules['LineOrientation']
                orient_map = {
                    'horizontal': LineOrientation.HORIZONTAL,
                    'vertical': LineOrientation.VERTICAL,
                }
                base_params['orientation'] = orient_map[orientation]
                base_params['duty_cycle'] = np.random.uniform(0.8, 1.2)
                return self._ts_modules['LineSpaceParams'](**base_params)

            elif structure_type == 'contact_hole':
                pattern = np.random.choice(['square_grid', 'hexagonal'])
                HolePattern = self._ts_modules['HolePattern']
                pattern_map = {
                    'square_grid': HolePattern.SQUARE_GRID,
                    'hexagonal': HolePattern.HEXAGONAL,
                }
                base_params['pattern'] = pattern_map[pattern]
                base_params['hole_shape'] = np.random.choice(['circle', 'square'])
                base_params['aspect_ratio'] = np.random.uniform(0.8, 1.2)
                base_params['rotation'] = np.random.uniform(-10, 10)
                return self._ts_modules['ContactHoleParams'](**base_params)

            elif structure_type == 'l_shaped_corner':
                base_params['arm_length'] = np.random.uniform(cd * 3, cd * 6)
                base_params['corner_type'] = np.random.choice(['inner', 'outer'])
                return self._ts_modules['LShapedCornerParams'](**base_params)

            elif structure_type == 't_junction':
                base_params['stem_length'] = np.random.uniform(cd * 4, cd * 8)
                base_params['branch_length'] = np.random.uniform(cd * 2, cd * 5)
                return self._ts_modules['TJunctionParams'](**base_params)

            elif structure_type == 'sram_bitcell':
                base_params['bitcell_type'] = np.random.choice(['6T', 'thin-film'])
                return self._ts_modules['SRAMBitcellParams'](**base_params)

        return None

    def _generate_target(self, params: Any) -> np.ndarray:
        """生成目标版图"""
        if self._ts_modules is not None and params is not None:
            try:
                return self._ts_modules['generate'](params)
            except Exception as e:
                logger.warning(f"测试结构生成失败，使用备用: {e}")

        return self._fallback_generator(params)

    def _fallback_generator(self, params: Any) -> np.ndarray:
        """备用版图生成器（当 test_structures 不可用时）"""
        if params is not None and hasattr(params, 'grid_size'):
            grid_size = params.grid_size
        else:
            grid_size = self.config.grid_size_range[1]

        ny, nx = grid_size
        target = np.zeros((ny, nx), dtype=np.float64)

        structure = np.random.choice(['lines_h', 'lines_v', 'blocks', 'holes', 'random'])

        if structure == 'lines_h':
            period = max(8, int(np.random.uniform(8, 32)))
            width = max(2, period // 2)
            for y in range(0, ny, period):
                target[y:min(y + width, ny), :] = 1.0
        elif structure == 'lines_v':
            period = max(8, int(np.random.uniform(8, 32)))
            width = max(2, period // 2)
            for x in range(0, nx, period):
                target[:, x:min(x + width, nx)] = 1.0
        elif structure == 'blocks':
            n_blocks = np.random.randint(3, 10)
            for _ in range(n_blocks):
                bh = np.random.randint(4, ny // 4)
                bw = np.random.randint(4, nx // 4)
                by = np.random.randint(0, ny - bh)
                bx = np.random.randint(0, nx - bw)
                target[by:by + bh, bx:bx + bw] = 1.0
        elif structure == 'holes':
            target = np.ones((ny, nx), dtype=np.float64)
            n_holes = np.random.randint(5, 20)
            for _ in range(n_holes):
                r = np.random.randint(2, min(ny, nx) // 8)
                cy = np.random.randint(r, ny - r)
                cx = np.random.randint(r, nx - r)
                yy, xx = np.ogrid[-cy:ny - cy, -cx:nx - cx]
                mask = yy ** 2 + xx ** 2 <= r ** 2
                target[mask] = 0.0
        else:
            target = np.random.rand(ny, nx) > 0.6
            target = target.astype(np.float64)

        return target

    def _compute_loss(self, mask: np.ndarray) -> float:
        """计算当前掩模与目标的 MSE 损失"""
        return float(np.mean((mask - self.target) ** 2))

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        重置环境：随机生成新的目标版图和带噪声的初始掩模

        Returns:
            (state, target)
        """
        self._current_episode += 1
        self._step_count = 0
        self._state_encoder.reset()

        grid_size = self._sample_grid_size()
        params = self._sample_structure_params(grid_size)
        self.target = self._generate_target(params)

        initial_mask = self.target.copy()
        noise = np.random.uniform(
            -self.config.noise_level,
            self.config.noise_level,
            self.target.shape
        )
        initial_mask = np.clip(initial_mask + noise, 0.0, 1.0)
        self.mask = initial_mask

        self.current_loss = self._compute_loss(self.mask)
        self.best_loss = self.current_loss
        self._state_encoder.record_loss(self.current_loss)

        state = self._state_encoder.encode(self.mask, self.target)
        return state, self.target

    def _apply_action_dqn(self, action_idx: int, bounds: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """将 DQN 离散动作应用到掩模"""
        deltas = [
            np.array([0.0, 0.0]),
            np.array([0.05, 0.0]),
            np.array([-0.05, 0.0]),
            np.array([0.0, 0.05]),
            np.array([0.0, -0.05]),
            np.array([0.05, 0.05]),
            np.array([-0.05, 0.05]),
            np.array([0.05, -0.05]),
            np.array([-0.05, -0.05]),
        ]
        delta = deltas[action_idx % len(deltas)]

        h, w = self.mask.shape
        full_delta = np.zeros_like(self.mask)

        regions = [
            (0, h // 2, 0, w // 2),
            (0, h // 2, w // 2, w),
            (h // 2, h, 0, w // 2),
            (h // 2, h, w // 2, w),
        ]
        region_idx = action_idx % len(regions)
        y1, y2, x1, x2 = regions[region_idx]

        if delta.size == 2:
            full_delta[y1:y2, x1:x2] = delta[0]
            full_delta[:, :] += np.random.normal(0, abs(delta[1]) * 0.1, full_delta.shape)
        else:
            full_delta[y1:y2, x1:x2] = delta.flat[0]

        new_mask = self.mask + full_delta
        return np.clip(new_mask, bounds[0], bounds[1])

    def _apply_action_ppo(self, action: np.ndarray, bounds: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """将 PPO 连续动作应用到掩模"""
        h, w = self.mask.shape
        action_dim = action.size

        if action_dim == 4:
            regions = [
                (0, h // 2, 0, w // 2),
                (0, h // 2, w // 2, w),
                (h // 2, h, 0, w // 2),
                (h // 2, h, w // 2, w),
            ]
            full_delta = np.zeros_like(self.mask)
            for i, (y1, y2, x1, x2) in enumerate(regions):
                full_delta[y1:y2, x1:x2] = np.tanh(action[i]) * 0.1
        elif action_dim >= h * w:
            full_delta = np.tanh(action[:h * w].reshape(h, w)) * 0.1
        else:
            scale = int(np.ceil(np.sqrt(h * w / action_dim)))
            full_delta = np.zeros((h, w), dtype=np.float64)
            for i in range(action_dim):
                row = (i * scale) // w
                col = (i * scale) % w
                if row < h and col < w:
                    val = np.tanh(action[i]) * 0.1
                    full_delta[row:min(row + scale, h), col:min(col + scale, w)] = val

        new_mask = self.mask + full_delta
        return np.clip(new_mask, bounds[0], bounds[1])

    def step_dqn(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步 DQN 离散动作

        Returns:
            (next_state, reward, done, info)
        """
        self._step_count += 1
        new_mask = self._apply_action_dqn(action_idx)
        new_loss = self._compute_loss(new_mask)
        reward = (self.current_loss - new_loss) * self.config.reward_scale

        self.mask = new_mask
        self.current_loss = new_loss
        if new_loss < self.best_loss:
            self.best_loss = new_loss

        self._state_encoder.record_loss(new_loss)
        next_state = self._state_encoder.encode(self.mask, self.target)

        done = (self._step_count >= self.config.max_steps_per_episode) or (new_loss < 1e-6)

        info = {
            'loss': new_loss,
            'best_loss': self.best_loss,
            'step': self._step_count,
            'improvement': reward > 0,
        }

        return next_state, reward, done, info

    def step_ppo(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步 PPO 连续动作

        Returns:
            (next_state, reward, done, info)
        """
        self._step_count += 1
        new_mask = self._apply_action_ppo(action)
        new_loss = self._compute_loss(new_mask)
        reward = (self.current_loss - new_loss) * self.config.reward_scale

        self.mask = new_mask
        self.current_loss = new_loss
        if new_loss < self.best_loss:
            self.best_loss = new_loss

        self._state_encoder.record_loss(new_loss)
        next_state = self._state_encoder.encode(self.mask, self.target)

        done = (self._step_count >= self.config.max_steps_per_episode) or (new_loss < 1e-6)

        info = {
            'loss': new_loss,
            'best_loss': self.best_loss,
            'step': self._step_count,
            'improvement': reward > 0,
        }

        return next_state, reward, done, info

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (3,) + self.mask.shape if self.mask is not None else (3, 0, 0)


# ─── 模型接口类 ────────────────────────────────────────


class DQNModel:
    """
    DQN 模型接口

    与 ReinforcementLearningOptimizer 对接：
      - predict(state) → action
      - update(batch)  → loss
    """

    def __init__(self, config: Optional[DQNConfig] = None, device: str = 'cpu'):
        _check_torch()
        self.config = config or DQNConfig()
        self.device = torch.device(device)

        self.policy_net = DQNNetwork(
            num_actions=self.config.num_actions,
            feat_dim=self.config.feat_dim,
        ).to(self.device)
        self.target_net = DQNNetwork(
            num_actions=self.config.num_actions,
            feat_dim=self.config.feat_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_capacity)
        self._steps = 0

    def _epsilon(self) -> float:
        eps = self.config.epsilon_end + \
              (self.config.epsilon_start - self.config.epsilon_end) * \
              np.exp(-self._steps / self.config.epsilon_decay)
        return eps

    def select_action(self, state: np.ndarray) -> int:
        self._steps += 1
        if np.random.random() < self._epsilon():
            return np.random.randint(self.config.num_actions)
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            return q.argmax(dim=1).item()

    def predict(self, state: np.ndarray) -> np.ndarray:
        action_idx = self.select_action(state)
        return self._action_idx_to_delta(action_idx)

    def _action_idx_to_delta(self, idx: int) -> np.ndarray:
        deltas = [
            np.array([0.0, 0.0]),
            np.array([0.05, 0.0]),
            np.array([-0.05, 0.0]),
            np.array([0.0, 0.05]),
            np.array([0.0, -0.05]),
            np.array([0.05, 0.05]),
            np.array([-0.05, 0.05]),
            np.array([0.05, -0.05]),
            np.array([-0.05, -0.05]),
        ]
        return deltas[idx % len(deltas)]

    def store(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        if len(self.buffer) < self.config.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.config.batch_size)

        s = torch.from_numpy(states).to(self.device)
        a = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        r = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        ns = torch.from_numpy(next_states).to(self.device)
        d = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1, keepdim=True)[0]
            target_q = r + self.config.gamma * next_q * (1 - d)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update(self, batch: List[Tuple]) -> float:
        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            action_idx = 0
            if isinstance(action, (int, np.integer)):
                action_idx = int(action)
            elif isinstance(action, np.ndarray):
                action_idx = int(np.argmax(np.abs(action.flatten()))) % self.config.num_actions
            self.store(state, action_idx, reward, next_state, done)
            loss = self.train_step()
            total_loss += loss
            self._steps += 1
            if self._steps % self.config.target_update_freq == 0:
                self.update_target()
        return total_loss / max(len(batch), 1)

    def save_weights(self, path: str) -> str:
        """
        保存模型权重到文件

        Args:
            path: 保存路径（目录或文件路径）

        Returns:
            实际保存的文件路径
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        if not path.endswith('.pt') and not path.endswith('.pth'):
            path = path + '.pt'

        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'num_actions': self.config.num_actions,
                'feat_dim': self.config.feat_dim,
                'lr': self.config.lr,
                'gamma': self.config.gamma,
                'epsilon_start': self.config.epsilon_start,
                'epsilon_end': self.config.epsilon_end,
                'epsilon_decay': self.config.epsilon_decay,
                'target_update_freq': self.config.target_update_freq,
                'buffer_capacity': self.config.buffer_capacity,
                'batch_size': self.config.batch_size,
            },
            'steps': self._steps,
            'model_type': 'dqn',
        }
        torch.save(checkpoint, path)
        logger.info(f"DQN 权重已保存到: {path}")
        return path

    def load_weights(self, path: str, strict: bool = True) -> None:
        """
        从文件加载模型权重

        Args:
            path: 权重文件路径
            strict: 是否严格匹配层名称
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"权重文件不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        if 'policy_net_state_dict' in checkpoint:
            self.policy_net.load_state_dict(
                checkpoint['policy_net_state_dict'], strict=strict
            )
            self.target_net.load_state_dict(
                checkpoint['target_net_state_dict'], strict=strict
            )
            if 'optimizer_state_dict' in checkpoint and not strict:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    logger.warning(f"优化器状态加载失败，使用新优化器: {e}")
            if 'steps' in checkpoint:
                self._steps = checkpoint['steps']
        else:
            self.policy_net.load_state_dict(checkpoint, strict=strict)
            self.target_net.load_state_dict(checkpoint, strict=strict)

        self.target_net.eval()
        logger.info(f"DQN 权重已从 {path} 加载")

    def freeze_encoder(self) -> None:
        """冻结卷积编码器层（只训练头部）"""
        for param in self.policy_net.encoder.parameters():
            param.requires_grad = False
        logger.info("DQN 编码器已冻结")

    def unfreeze_encoder(self) -> None:
        """解冻卷积编码器层"""
        for param in self.policy_net.encoder.parameters():
            param.requires_grad = True
        logger.info("DQN 编码器已解冻")

    def pretrain_on_synthetic(self,
                               pretrain_config: Optional[PretrainConfig] = None,
                               callback: Optional[Callable[[Dict], None]] = None
                               ) -> Dict[str, Any]:
        """
        在合成测试结构库上预训练策略网络

        Args:
            pretrain_config: 预训练配置
            callback: 训练进度回调函数，接收包含训练信息的字典

        Returns:
            预训练统计信息字典
        """
        config = pretrain_config or PretrainConfig()
        env = SyntheticTestStructureEnv(config)

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        episode_rewards: List[float] = []
        episode_losses: List[float] = []
        episode_best_losses: List[float] = []
        best_avg_reward = float('-inf')
        best_checkpoint_path: Optional[str] = None

        logger.info(
            f"开始 DQN 预训练: total_episodes={config.total_episodes}, "
            f"max_steps={config.max_steps_per_episode}, "
            f"curriculum={config.use_curriculum}"
        )

        for episode in range(1, config.total_episodes + 1):
            state, target = env.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            n_steps = 0
            n_updates = 0

            for step in range(config.max_steps_per_episode):
                action_idx = self.select_action(state)
                next_state, reward, done, info = env.step_dqn(action_idx)

                self.store(state, action_idx, reward, next_state, done)
                loss = self.train_step()
                n_updates += 1 if loss > 0 else 0
                episode_loss += loss

                episode_reward += reward
                n_steps += 1
                state = next_state

                if self._steps % self.config.target_update_freq == 0:
                    self.update_target()

                if done:
                    break

            avg_loss = episode_loss / max(n_updates, 1)
            episode_rewards.append(episode_reward)
            episode_losses.append(avg_loss)
            episode_best_losses.append(env.best_loss)

            if episode % config.log_freq == 0 or episode == 1:
                window = min(10, len(episode_rewards))
                avg_recent_reward = np.mean(episode_rewards[-window:])
                avg_recent_best = np.mean(episode_best_losses[-window:])
                logger.info(
                    f"[DQN Pretrain] Episode {episode}/{config.total_episodes} | "
                    f"Avg Reward (last {window}): {avg_recent_reward:.3f} | "
                    f"Ep Reward: {episode_reward:.3f} | "
                    f"Avg Loss: {avg_loss:.6f} | "
                    f"Best Loss: {env.best_loss:.6e} | "
                    f"Avg Best (last {window}): {avg_recent_best:.6e} | "
                    f"Steps: {n_steps} | Epsilon: {self._epsilon():.4f}"
                )

                if callback is not None:
                    callback({
                        'episode': episode,
                        'total_episodes': config.total_episodes,
                        'episode_reward': episode_reward,
                        'avg_reward_window': avg_recent_reward,
                        'avg_loss': avg_loss,
                        'best_loss': env.best_loss,
                        'avg_best_loss_window': avg_recent_best,
                        'epsilon': self._epsilon(),
                        'steps': n_steps,
                        'model_type': 'dqn',
                    })

            if episode % config.save_freq == 0:
                checkpoint_path = os.path.join(
                    config.checkpoint_dir,
                    f"dqn_pretrain_ep{episode:06d}.pt"
                )
                self.save_weights(checkpoint_path)

                if len(episode_rewards) >= 10:
                    current_avg = np.mean(episode_rewards[-10:])
                    if current_avg > best_avg_reward:
                        best_avg_reward = current_avg
                        best_path = os.path.join(
                            config.checkpoint_dir, "dqn_pretrain_best.pt"
                        )
                        self.save_weights(best_path)
                        best_checkpoint_path = best_path

        final_path = os.path.join(config.checkpoint_dir, "dqn_pretrain_final.pt")
        self.save_weights(final_path)

        stats = {
            'model_type': 'dqn',
            'total_episodes': config.total_episodes,
            'final_path': final_path,
            'best_checkpoint_path': best_checkpoint_path,
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'episode_best_losses': episode_best_losses,
            'avg_final_reward': float(np.mean(episode_rewards[-50:])) if len(episode_rewards) >= 50 else float(np.mean(episode_rewards)),
            'best_avg_reward': best_avg_reward,
        }

        logger.info(
            f"DQN 预训练完成! 最终权重: {final_path}, "
            f"最佳权重: {best_checkpoint_path}"
        )
        return stats

    def fine_tune(self,
                   target: np.ndarray,
                   objective: Optional[Callable[[np.ndarray], float]] = None,
                   finetune_config: Optional[FinetuneConfig] = None,
                   initial_mask: Optional[np.ndarray] = None,
                   ) -> Dict[str, Any]:
        """
        对特定目标版图微调

        Args:
            target: 目标版图
            objective: 自定义目标函数，若为 None 则使用 MSE
            finetune_config: 微调配置
            initial_mask: 初始掩模，若为 None 则使用 target + 噪声

        Returns:
            微调统计信息和最佳掩模
        """
        config = finetune_config or FinetuneConfig()

        if config.freeze_encoder:
            self.freeze_encoder()

        original_lr = self.optimizer.param_groups[0]['lr']
        for pg in self.optimizer.param_groups:
            pg['lr'] = original_lr * config.lr_factor

        if objective is None:
            def objective(mask):
                return float(np.mean((mask - target) ** 2))

        if initial_mask is None:
            initial_mask = target.copy() + np.random.uniform(-0.2, 0.2, target.shape)
            initial_mask = np.clip(initial_mask, 0.0, 1.0)

        state_encoder = MultiChannelStateEncoder()
        mask = initial_mask.copy()
        best_mask = mask.copy()
        best_loss = objective(mask)
        current_loss = best_loss

        episode_rewards: List[float] = []
        episode_best_losses: List[float] = []

        logger.info(
            f"开始 DQN 微调: episodes={config.total_episodes}, "
            f"max_steps={config.max_steps_per_episode}, "
            f"freeze_encoder={config.freeze_encoder}, lr_factor={config.lr_factor}"
        )

        for episode in range(1, config.total_episodes + 1):
            mask = initial_mask.copy() + np.random.uniform(-0.05, 0.05, target.shape)
            mask = np.clip(mask, 0.0, 1.0)
            state_encoder.reset()
            current_loss = objective(mask)
            state_encoder.record_loss(current_loss)
            state = state_encoder.encode(mask, target)
            episode_reward = 0.0

            for step in range(config.max_steps_per_episode):
                action_idx = self.select_action(state)
                delta = self._action_idx_to_delta(action_idx)

                h, w = mask.shape
                full_delta = np.zeros_like(mask)
                regions = [
                    (0, h // 2, 0, w // 2),
                    (0, h // 2, w // 2, w),
                    (h // 2, h, 0, w // 2),
                    (h // 2, h, w // 2, w),
                ]
                region_idx = action_idx % len(regions)
                y1, y2, x1, x2 = regions[region_idx]
                full_delta[y1:y2, x1:x2] = delta[0] if delta.size >= 1 else delta.flat[0]

                new_mask = np.clip(mask + full_delta, 0.0, 1.0)
                new_loss = objective(new_mask)
                reward = (current_loss - new_loss) * 100.0

                state_encoder.record_loss(new_loss)
                next_state = state_encoder.encode(new_mask, target)
                done = step == config.max_steps_per_episode - 1

                self.store(state, action_idx, reward, next_state, done)
                loss = self.train_step()
                if self._steps % self.config.target_update_freq == 0:
                    self.update_target()

                episode_reward += reward
                mask = new_mask
                current_loss = new_loss
                state = next_state

                if new_loss < best_loss:
                    best_loss = new_loss
                    best_mask = mask.copy()

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_best_losses.append(best_loss)

            if episode % 20 == 0 or episode == 1:
                window = min(10, len(episode_rewards))
                logger.info(
                    f"[DQN Finetune] Episode {episode}/{config.total_episodes} | "
                    f"Avg Reward (last {window}): {np.mean(episode_rewards[-window:]):.3f} | "
                    f"Best Loss: {best_loss:.6e} | Epsilon: {self._epsilon():.4f}"
                )

        for pg in self.optimizer.param_groups:
            pg['lr'] = original_lr

        if config.freeze_encoder:
            self.unfreeze_encoder()

        stats = {
            'best_mask': best_mask,
            'best_loss': best_loss,
            'episode_rewards': episode_rewards,
            'episode_best_losses': episode_best_losses,
        }
        logger.info(f"DQN 微调完成! Best Loss: {best_loss:.6e}")
        return stats


class ActorCriticModel:
    """Actor-Critic (A2C) 模型接口"""

    def __init__(self, config: Optional[ActorCriticConfig] = None,
                 device: str = 'cpu'):
        _check_torch()
        self.config = config or ActorCriticConfig()
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(
            action_dim=self.config.action_dim,
            feat_dim=self.config.feat_dim,
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)

        self._transition_cache: Dict[str, Any] = {}

    def predict(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(self.device)
            mean, std, value = self.network(s)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            self._transition_cache = {
                'log_prob': log_prob.item(),
                'value': value.item(),
            }
            return action.squeeze(0).cpu().numpy()

    def update(self, batch: List[Tuple]) -> float:
        if not batch:
            return 0.0

        states, actions_list, rewards, next_states, dones = [], [], [], [], []
        for s, a, r, ns, d in batch:
            states.append(s)
            actions_list.append(a if isinstance(a, np.ndarray) else np.array([a]))
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        s_t = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        a_t = torch.from_numpy(np.array(actions_list, dtype=np.float32)).to(self.device)
        r_t = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(1).to(self.device)
        ns_t = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        d_t = torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            _, _, next_value = self.network(ns_t)
            target = r_t + self.config.gamma * next_value * (1 - d_t)

        mean, std, value = self.network(s_t)
        advantage = target - value

        dist = Normal(mean, std)
        log_prob = dist.log_prob(a_t).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()

        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = F.mse_loss(value, target.detach())
        loss = actor_loss + self.config.value_coef * critic_loss - \
               self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()


class PPOModel:
    """PPO 模型接口"""

    def __init__(self, config: Optional[PPOConfig] = None, device: str = 'cpu'):
        _check_torch()
        self.config = config or PPOConfig()
        self.device = torch.device(device)

        self.network = PPOActorCritic(
            action_dim=self.config.action_dim,
            feat_dim=self.config.feat_dim,
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer()

    def predict(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action(s)
            self._last_log_prob = log_prob.item()
            self._last_value = value.item()
            return action.squeeze(0).cpu().numpy()

    def store_transition(self, state, action, reward, done):
        self.buffer.push(
            state, action,
            getattr(self, '_last_log_prob', 0.0),
            reward,
            getattr(self, '_last_value', 0.0),
            done,
        )

    def update(self, batch: List[Tuple] = None) -> float:
        if len(self.buffer) < 2:
            return 0.0

        advantages, returns = self.buffer.compute_returns(
            self.config.gamma, self.config.gae_lambda
        )

        states = np.array(self.buffer.states, dtype=np.float32)
        actions = np.array(self.buffer.actions, dtype=np.float32)
        old_log_probs = np.array(self.buffer.log_probs, dtype=np.float32)

        adv_t = torch.from_numpy(advantages).unsqueeze(1).to(self.device)
        ret_t = torch.from_numpy(returns).unsqueeze(1).to(self.device)
        old_lp_t = torch.from_numpy(old_log_probs).unsqueeze(1).to(self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        n_updates = 0
        dataset_size = len(states)
        mb_size = min(self.config.minibatch_size, dataset_size)

        for _ in range(self.config.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, mb_size):
                idx = indices[start:start + mb_size]
                s_b = torch.from_numpy(states[idx]).to(self.device)
                a_b = torch.from_numpy(actions[idx]).to(self.device)
                old_lp_b = old_lp_t[idx]
                adv_b = adv_t[idx]
                ret_b = ret_t[idx]

                log_prob, value, entropy = self.network.evaluate(s_b, a_b)

                ratio = torch.exp(log_prob - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio, 1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(value, ret_b)
                entropy_bonus = entropy.mean()

                loss = (actor_loss
                        + self.config.value_coef * critic_loss
                        - self.config.entropy_coef * entropy_bonus)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        self.buffer.clear()
        return total_loss / max(n_updates, 1)

    def save_weights(self, path: str) -> str:
        """
        保存模型权重到文件

        Args:
            path: 保存路径（目录或文件路径）

        Returns:
            实际保存的文件路径
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        if not path.endswith('.pt') and not path.endswith('.pth'):
            path = path + '.pt'

        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'action_dim': self.config.action_dim,
                'feat_dim': self.config.feat_dim,
                'lr': self.config.lr,
                'gamma': self.config.gamma,
                'gae_lambda': self.config.gae_lambda,
                'clip_epsilon': self.config.clip_epsilon,
                'entropy_coef': self.config.entropy_coef,
                'value_coef': self.config.value_coef,
                'ppo_epochs': self.config.ppo_epochs,
                'minibatch_size': self.config.minibatch_size,
            },
            'model_type': 'ppo',
        }
        torch.save(checkpoint, path)
        logger.info(f"PPO 权重已保存到: {path}")
        return path

    def load_weights(self, path: str, strict: bool = True) -> None:
        """
        从文件加载模型权重

        Args:
            path: 权重文件路径
            strict: 是否严格匹配层名称
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"权重文件不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        if 'network_state_dict' in checkpoint:
            self.network.load_state_dict(
                checkpoint['network_state_dict'], strict=strict
            )
            if 'optimizer_state_dict' in checkpoint and not strict:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    logger.warning(f"优化器状态加载失败，使用新优化器: {e}")
        else:
            self.network.load_state_dict(checkpoint, strict=strict)

        logger.info(f"PPO 权重已从 {path} 加载")

    def freeze_encoder(self) -> None:
        """冻结卷积编码器层（只训练头部）"""
        for param in self.network.encoder.parameters():
            param.requires_grad = False
        logger.info("PPO 编码器已冻结")

    def unfreeze_encoder(self) -> None:
        """解冻卷积编码器层"""
        for param in self.network.encoder.parameters():
            param.requires_grad = True
        logger.info("PPO 编码器已解冻")

    def pretrain_on_synthetic(self,
                               pretrain_config: Optional[PretrainConfig] = None,
                               callback: Optional[Callable[[Dict], None]] = None
                               ) -> Dict[str, Any]:
        """
        在合成测试结构库上预训练策略网络

        Args:
            pretrain_config: 预训练配置
            callback: 训练进度回调函数，接收包含训练信息的字典

        Returns:
            预训练统计信息字典
        """
        config = pretrain_config or PretrainConfig()
        env = SyntheticTestStructureEnv(config)

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        episode_rewards: List[float] = []
        episode_losses: List[float] = []
        episode_best_losses: List[float] = []
        best_avg_reward = float('-inf')
        best_checkpoint_path: Optional[str] = None
        update_interval = 4

        logger.info(
            f"开始 PPO 预训练: total_episodes={config.total_episodes}, "
            f"max_steps={config.max_steps_per_episode}, "
            f"curriculum={config.use_curriculum}"
        )

        for episode in range(1, config.total_episodes + 1):
            state, target = env.reset()
            episode_reward = 0.0
            n_steps = 0

            for step in range(config.max_steps_per_episode):
                action = self.predict(state)
                next_state, reward, done, info = env.step_ppo(action)
                self.store_transition(state, action, reward, done)

                episode_reward += reward
                n_steps += 1
                state = next_state

                if done:
                    break

            loss = 0.0
            if (episode % update_interval == 0) and len(self.buffer) >= 2:
                loss = self.update()

            episode_rewards.append(episode_reward)
            episode_losses.append(loss)
            episode_best_losses.append(env.best_loss)

            if episode % config.log_freq == 0 or episode == 1:
                window = min(10, len(episode_rewards))
                avg_recent_reward = np.mean(episode_rewards[-window:])
                avg_recent_best = np.mean(episode_best_losses[-window:])
                logger.info(
                    f"[PPO Pretrain] Episode {episode}/{config.total_episodes} | "
                    f"Avg Reward (last {window}): {avg_recent_reward:.3f} | "
                    f"Ep Reward: {episode_reward:.3f} | "
                    f"Loss: {loss:.6f} | "
                    f"Best Loss: {env.best_loss:.6e} | "
                    f"Avg Best (last {window}): {avg_recent_best:.6e} | "
                    f"Steps: {n_steps}"
                )

                if callback is not None:
                    callback({
                        'episode': episode,
                        'total_episodes': config.total_episodes,
                        'episode_reward': episode_reward,
                        'avg_reward_window': avg_recent_reward,
                        'loss': loss,
                        'best_loss': env.best_loss,
                        'avg_best_loss_window': avg_recent_best,
                        'steps': n_steps,
                        'model_type': 'ppo',
                    })

            if episode % config.save_freq == 0:
                checkpoint_path = os.path.join(
                    config.checkpoint_dir,
                    f"ppo_pretrain_ep{episode:06d}.pt"
                )
                self.save_weights(checkpoint_path)

                if len(episode_rewards) >= 10:
                    current_avg = np.mean(episode_rewards[-10:])
                    if current_avg > best_avg_reward:
                        best_avg_reward = current_avg
                        best_path = os.path.join(
                            config.checkpoint_dir, "ppo_pretrain_best.pt"
                        )
                        self.save_weights(best_path)
                        best_checkpoint_path = best_path

        final_path = os.path.join(config.checkpoint_dir, "ppo_pretrain_final.pt")
        self.save_weights(final_path)

        stats = {
            'model_type': 'ppo',
            'total_episodes': config.total_episodes,
            'final_path': final_path,
            'best_checkpoint_path': best_checkpoint_path,
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'episode_best_losses': episode_best_losses,
            'avg_final_reward': float(np.mean(episode_rewards[-50:])) if len(episode_rewards) >= 50 else float(np.mean(episode_rewards)),
            'best_avg_reward': best_avg_reward,
        }

        logger.info(
            f"PPO 预训练完成! 最终权重: {final_path}, "
            f"最佳权重: {best_checkpoint_path}"
        )
        return stats

    def fine_tune(self,
                   target: np.ndarray,
                   objective: Optional[Callable[[np.ndarray], float]] = None,
                   finetune_config: Optional[FinetuneConfig] = None,
                   initial_mask: Optional[np.ndarray] = None,
                   ) -> Dict[str, Any]:
        """
        对特定目标版图微调

        Args:
            target: 目标版图
            objective: 自定义目标函数，若为 None 则使用 MSE
            finetune_config: 微调配置
            initial_mask: 初始掩模，若为 None 则使用 target + 噪声

        Returns:
            微调统计信息和最佳掩模
        """
        config = finetune_config or FinetuneConfig()

        if config.freeze_encoder:
            self.freeze_encoder()

        original_lr = self.optimizer.param_groups[0]['lr']
        for pg in self.optimizer.param_groups:
            pg['lr'] = original_lr * config.lr_factor

        if objective is None:
            def objective(mask):
                return float(np.mean((mask - target) ** 2))

        if initial_mask is None:
            initial_mask = target.copy() + np.random.uniform(-0.2, 0.2, target.shape)
            initial_mask = np.clip(initial_mask, 0.0, 1.0)

        state_encoder = MultiChannelStateEncoder()
        best_mask = initial_mask.copy()
        best_loss = objective(best_mask)
        current_loss = best_loss

        episode_rewards: List[float] = []
        episode_best_losses: List[float] = []

        logger.info(
            f"开始 PPO 微调: episodes={config.total_episodes}, "
            f"max_steps={config.max_steps_per_episode}, "
            f"freeze_encoder={config.freeze_encoder}, lr_factor={config.lr_factor}"
        )

        update_interval = 4
        for episode in range(1, config.total_episodes + 1):
            mask = initial_mask.copy() + np.random.uniform(-0.05, 0.05, target.shape)
            mask = np.clip(mask, 0.0, 1.0)
            state_encoder.reset()
            current_loss = objective(mask)
            state_encoder.record_loss(current_loss)
            state = state_encoder.encode(mask, target)
            episode_reward = 0.0

            for step in range(config.max_steps_per_episode):
                action = self.predict(state)

                h, w = mask.shape
                action_dim = action.size
                if action_dim == 4:
                    regions = [
                        (0, h // 2, 0, w // 2),
                        (0, h // 2, w // 2, w),
                        (h // 2, h, 0, w // 2),
                        (h // 2, h, w // 2, w),
                    ]
                    full_delta = np.zeros_like(mask)
                    for i, (y1, y2, x1, x2) in enumerate(regions):
                        full_delta[y1:y2, x1:x2] = np.tanh(action[i]) * 0.1
                else:
                    full_delta = np.tanh(action[:h * w].reshape(h, w)) * 0.1 if action.size >= h * w else np.zeros_like(mask)

                new_mask = np.clip(mask + full_delta, 0.0, 1.0)
                new_loss = objective(new_mask)
                reward = (current_loss - new_loss) * 100.0

                state_encoder.record_loss(new_loss)
                next_state = state_encoder.encode(new_mask, target)
                done = step == config.max_steps_per_episode - 1

                self.store_transition(state, action, reward, done)

                episode_reward += reward
                mask = new_mask
                current_loss = new_loss
                state = next_state

                if new_loss < best_loss:
                    best_loss = new_loss
                    best_mask = mask.copy()

                if done:
                    break

            if (episode % update_interval == 0) and len(self.buffer) >= 2:
                self.update()

            episode_rewards.append(episode_reward)
            episode_best_losses.append(best_loss)

            if episode % 20 == 0 or episode == 1:
                window = min(10, len(episode_rewards))
                logger.info(
                    f"[PPO Finetune] Episode {episode}/{config.total_episodes} | "
                    f"Avg Reward (last {window}): {np.mean(episode_rewards[-window:]):.3f} | "
                    f"Best Loss: {best_loss:.6e}"
                )

        for pg in self.optimizer.param_groups:
            pg['lr'] = original_lr

        if config.freeze_encoder:
            self.unfreeze_encoder()

        stats = {
            'best_mask': best_mask,
            'best_loss': best_loss,
            'episode_rewards': episode_rewards,
            'episode_best_losses': episode_best_losses,
        }
        logger.info(f"PPO 微调完成! Best Loss: {best_loss:.6e}")
        return stats


class DeepRLModelFactory:
    """深度 RL 模型工厂"""

    _registry: Dict[str, type] = {
        'dqn': DQNModel,
        'ppo': PPOModel,
        'actor_critic': ActorCriticModel,
    }

    @classmethod
    def create(cls, model_type: str, **kwargs) -> Any:
        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[model_type](**kwargs)

    @classmethod
    def available_models(cls) -> List[str]:
        return list(cls._registry.keys())


# ─── 预训练权重管理器 ──────────────────────────────────

class PretrainedWeightManager:
    """
    预训练权重管理器

    负责预训练权重的查找、加载和作为 ReinforcementLearningOptimizer
    默认初始化的管理。支持以下查找顺序：
    1. 显式指定的路径
    2. 项目内预训练检查点目录 (pretrained_checkpoints/)
    3. 用户家目录缓存 (~/.litho_rl_pretrained/)
    """

    DEFAULT_CACHE_DIR = os.path.expanduser("~/.litho_rl_pretrained")
    PROJECT_CHECKPOINT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pretrained_checkpoints"
    )

    _default_weights_paths: Dict[str, str] = {}

    @classmethod
    def search_paths(cls) -> List[str]:
        """返回预训练权重搜索路径列表（按优先级排序）"""
        paths = []
        env_path = os.environ.get("LITHO_RL_PRETRAINED_DIR")
        if env_path:
            paths.append(env_path)
        paths.append(cls.PROJECT_CHECKPOINT_DIR)
        paths.append(cls.DEFAULT_CACHE_DIR)
        return paths

    @classmethod
    def find_pretrained_weights(cls,
                                 model_type: str,
                                 prefer: str = "best"
                                 ) -> Optional[str]:
        """
        查找预训练权重文件

        Args:
            model_type: 模型类型 ('dqn' 或 'ppo')
            prefer: 偏好类型 'best' | 'final' | 'latest'

        Returns:
            找到的权重文件路径，未找到返回 None
        """
        search_dirs = cls.search_paths()

        filenames = []
        if prefer == "best":
            filenames = [f"{model_type}_pretrain_best.pt"]
        elif prefer == "final":
            filenames = [f"{model_type}_pretrain_final.pt"]
        else:
            filenames = [
                f"{model_type}_pretrain_best.pt",
                f"{model_type}_pretrain_final.pt",
            ]

        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            for fname in filenames:
                fpath = os.path.join(search_dir, fname)
                if os.path.exists(fpath):
                    return fpath

        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            pattern = f"{model_type}_pretrain_ep*.pt"
            try:
                import glob
                matches = sorted(glob.glob(os.path.join(search_dir, pattern)))
                if matches:
                    return matches[-1]
            except Exception:
                pass

        if model_type in cls._default_weights_paths:
            fpath = cls._default_weights_paths[model_type]
            if os.path.exists(fpath):
                return fpath

        return None

    @classmethod
    def register_default_weights(cls, model_type: str, path: str) -> None:
        """
        注册默认预训练权重路径

        Args:
            model_type: 模型类型 ('dqn' 或 'ppo')
            path: 权重文件路径
        """
        if not os.path.exists(path):
            logger.warning(f"注册的预训练权重路径不存在: {path}")
        cls._default_weights_paths[model_type] = path
        logger.info(f"已注册 {model_type} 默认预训练权重: {path}")

    @classmethod
    def load_pretrained_if_available(cls,
                                      model: Any,
                                      model_type: str,
                                      strict: bool = False,
                                      custom_path: Optional[str] = None
                                      ) -> bool:
        """
        如果存在预训练权重则加载到模型

        Args:
            model: 模型实例 (DQNModel 或 PPOModel)
            model_type: 模型类型 ('dqn' 或 'ppo')
            strict: 是否严格加载
            custom_path: 自定义权重路径，优先于自动搜索

        Returns:
            是否成功加载了预训练权重
        """
        weight_path = None

        if custom_path and os.path.exists(custom_path):
            weight_path = custom_path
        elif cls._default_weights_paths.get(model_type):
            registered = cls._default_weights_paths[model_type]
            if os.path.exists(registered):
                weight_path = registered
        else:
            weight_path = cls.find_pretrained_weights(model_type)

        if weight_path is not None:
            try:
                model.load_weights(weight_path, strict=strict)
                logger.info(
                    f"成功加载 {model_type} 预训练权重: {weight_path}"
                )
                return True
            except Exception as e:
                logger.warning(
                    f"加载 {model_type} 预训练权重失败 ({weight_path}): {e}"
                )
                return False
        else:
            logger.info(
                f"未找到 {model_type} 预训练权重，使用随机初始化"
            )
            return False


def run_dqn_pretrain(
    total_episodes: int = 200,
    checkpoint_dir: str = "./pretrained_checkpoints",
    device: str = "cpu",
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    便捷函数：运行 DQN 在合成测试结构库上的预训练

    Args:
        total_episodes: 预训练总回合数
        checkpoint_dir: 检查点保存目录
        device: 计算设备
        seed: 随机种子

    Returns:
        预训练统计信息
    """
    if seed is not None:
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)

    pretrain_cfg = PretrainConfig(
        total_episodes=total_episodes,
        checkpoint_dir=checkpoint_dir,
        log_freq=max(1, total_episodes // 20),
        save_freq=max(1, total_episodes // 10),
    )

    dqn_cfg = DQNConfig(
        num_actions=9,
        feat_dim=128,
        buffer_capacity=20000,
        batch_size=64,
    )

    model = DQNModel(config=dqn_cfg, device=device)
    stats = model.pretrain_on_synthetic(pretrain_cfg)

    PretrainedWeightManager.register_default_weights(
        "dqn", stats.get("best_checkpoint_path") or stats.get("final_path")
    )

    return stats


def run_ppo_pretrain(
    total_episodes: int = 200,
    checkpoint_dir: str = "./pretrained_checkpoints",
    device: str = "cpu",
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    便捷函数：运行 PPO 在合成测试结构库上的预训练

    Args:
        total_episodes: 预训练总回合数
        checkpoint_dir: 检查点保存目录
        device: 计算设备
        seed: 随机种子

    Returns:
        预训练统计信息
    """
    if seed is not None:
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)

    pretrain_cfg = PretrainConfig(
        total_episodes=total_episodes,
        checkpoint_dir=checkpoint_dir,
        log_freq=max(1, total_episodes // 20),
        save_freq=max(1, total_episodes // 10),
    )

    ppo_cfg = PPOConfig(
        action_dim=4,
        feat_dim=128,
        minibatch_size=32,
    )

    model = PPOModel(config=ppo_cfg, device=device)
    stats = model.pretrain_on_synthetic(pretrain_cfg)

    PretrainedWeightManager.register_default_weights(
        "ppo", stats.get("best_checkpoint_path") or stats.get("final_path")
    )

    return stats
