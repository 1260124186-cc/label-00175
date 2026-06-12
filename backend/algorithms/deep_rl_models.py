# -*- coding: utf-8 -*-
"""
深度强化学习模型模块：DQN / PPO / Actor-Critic 及多通道状态编码器

依赖 PyTorch，若未安装则相关类在实例化时抛出 ImportError。
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import logging

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
