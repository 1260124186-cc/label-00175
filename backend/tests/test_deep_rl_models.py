# -*- coding: utf-8 -*-
"""
深度强化学习模型单元测试
"""

import pytest
import numpy as np
from algorithms.deep_rl_models import (
    TORCH_AVAILABLE,
    MultiChannelStateEncoder,
    StateEncoderConfig,
    DQNConfig,
    PPOConfig,
    ActorCriticConfig,
    DeepRLModelFactory,
    ReplayBuffer,
    RolloutBuffer,
)


class TestMultiChannelStateEncoder:
    """多通道状态编码器测试"""

    def test_encode_output_shape(self):
        encoder = MultiChannelStateEncoder()
        mask = np.random.rand(16, 16)
        target = np.random.rand(16, 16)
        state = encoder.encode(mask, target)
        assert state.shape == (3, 16, 16)
        assert state.dtype == np.float32

    def test_encode_different_sizes(self):
        for size in [(8, 8), (16, 32), (32, 32)]:
            encoder = MultiChannelStateEncoder()
            mask = np.random.rand(*size)
            target = np.random.rand(*size)
            state = encoder.encode(mask, target)
            assert state.shape == (3, *size)

    def test_encode_with_error_map(self):
        encoder = MultiChannelStateEncoder()
        mask = np.random.rand(16, 16)
        target = np.random.rand(16, 16)
        error_map = np.abs(mask - target)
        state1 = encoder.encode(mask, target)
        state2 = encoder.encode(mask, target, error_map=error_map)
        np.testing.assert_allclose(state1, state2, atol=1e-5)

    def test_encode_channels_content(self):
        encoder = MultiChannelStateEncoder(StateEncoderConfig(normalize=False))
        mask = np.ones((8, 8)) * 0.5
        target = np.zeros((8, 8))
        state = encoder.encode(mask, target)
        ch_patch = state[0]
        ch_freq = state[1]
        ch_hist = state[2]
        assert ch_patch.shape == (8, 8)
        assert ch_freq.shape == (8, 8)
        assert ch_hist.shape == (8, 8)

    def test_history_channel_with_losses(self):
        encoder = MultiChannelStateEncoder(
            StateEncoderConfig(history_length=8, normalize=False)
        )
        for loss in [1.0, 0.8, 0.6, 0.4]:
            encoder.record_loss(loss)
        mask = np.random.rand(8, 8)
        target = np.zeros((8, 8))
        state = encoder.encode(mask, target)
        ch_hist = state[2]
        assert ch_hist.shape == (8, 8)
        assert not np.all(ch_hist == 0)

    def test_history_channel_empty(self):
        encoder = MultiChannelStateEncoder(
            StateEncoderConfig(history_length=8, normalize=False)
        )
        mask = np.random.rand(8, 8)
        target = np.zeros((8, 8))
        state = encoder.encode(mask, target)
        ch_hist = state[2]
        np.testing.assert_array_equal(ch_hist, np.zeros((8, 8)))

    def test_reset(self):
        encoder = MultiChannelStateEncoder()
        encoder.record_loss(1.0)
        encoder.record_loss(0.5)
        encoder.reset()
        assert len(encoder._loss_history) == 0

    def test_state_channels_property(self):
        encoder = MultiChannelStateEncoder()
        assert encoder.state_channels == 3

    def test_normalize_vs_no_normalize(self):
        cfg_norm = StateEncoderConfig(normalize=True)
        cfg_no = StateEncoderConfig(normalize=False)
        mask = np.random.rand(16, 16)
        target = np.zeros((16, 16))

        enc_norm = MultiChannelStateEncoder(cfg_norm)
        enc_no = MultiChannelStateEncoder(cfg_no)

        s_norm = enc_norm.encode(mask, target)
        s_no = enc_no.encode(mask, target)

        for c in range(3):
            std_n = s_norm[c].std()
            std_no = s_no[c].std()
            if std_no > 1e-8:
                assert abs(std_n - 1.0) < 0.5 or std_n < std_no

    def test_custom_config(self):
        cfg = StateEncoderConfig(patch_size=4, history_length=32, freq_bins=16)
        encoder = MultiChannelStateEncoder(cfg)
        mask = np.random.rand(8, 8)
        target = np.zeros((8, 8))
        state = encoder.encode(mask, target)
        assert state.shape == (3, 8, 8)


class TestReplayBuffer:
    """经验回放缓冲区测试"""

    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(
                np.random.rand(3, 8, 8).astype(np.float32),
                i % 9,
                float(i),
                np.random.rand(3, 8, 8).astype(np.float32),
                i == 49,
            )
        assert len(buf) == 50
        batch = buf.sample(16)
        assert len(batch) == 5
        assert batch[0].shape[0] == 16

    def test_capacity(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(np.zeros((3, 4, 4), dtype=np.float32), 0, 0.0,
                     np.zeros((3, 4, 4), dtype=np.float32), False)
        assert len(buf) == 10


class TestRolloutBuffer:
    """PPO 轨迹缓冲区测试"""

    def test_push_and_compute_returns(self):
        buf = RolloutBuffer()
        for i in range(10):
            buf.push(
                np.random.rand(3, 8, 8).astype(np.float32),
                np.random.rand(4).astype(np.float32),
                -0.5,
                0.0,
                0.0,
                i == 9,
            )
        assert len(buf) == 10
        advantages, returns = buf.compute_returns(gamma=0.99)
        assert advantages.shape == (10,)
        assert returns.shape == (10,)

    def test_clear(self):
        buf = RolloutBuffer()
        buf.push(np.zeros((3, 4, 4)), np.zeros(4), 0.0, 0.0, 0.0, False)
        buf.clear()
        assert len(buf) == 0


class TestDeepRLModelFactory:
    """深度 RL 模型工厂测试"""

    def test_available_models(self):
        models = DeepRLModelFactory.available_models()
        assert 'dqn' in models
        assert 'ppo' in models
        assert 'actor_critic' in models

    def test_unknown_model_type(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            DeepRLModelFactory.create('nonexistent')


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestDQNModel:
    """DQN 模型测试"""

    def test_creation(self):
        from algorithms.deep_rl_models import DQNModel
        model = DQNModel(DQNConfig(num_actions=9), device='cpu')
        assert model.config.num_actions == 9

    def test_predict(self):
        from algorithms.deep_rl_models import DQNModel
        model = DQNModel(DQNConfig(num_actions=9), device='cpu')
        state = np.random.rand(3, 16, 16).astype(np.float32)
        action = model.predict(state)
        assert isinstance(action, np.ndarray)

    def test_store_and_train(self):
        from algorithms.deep_rl_models import DQNModel
        model = DQNModel(DQNConfig(num_actions=9, buffer_capacity=100, batch_size=4), device='cpu')
        for _ in range(10):
            s = np.random.rand(3, 8, 8).astype(np.float32)
            ns = np.random.rand(3, 8, 8).astype(np.float32)
            model.store(s, 0, 1.0, ns, False)
        loss = model.train_step()
        assert isinstance(loss, float)

    def test_update_batch(self):
        from algorithms.deep_rl_models import DQNModel
        model = DQNModel(DQNConfig(num_actions=9), device='cpu')
        batch = []
        for _ in range(5):
            s = np.random.rand(3, 8, 8).astype(np.float32)
            a = np.random.rand(2)
            ns = np.random.rand(3, 8, 8).astype(np.float32)
            batch.append((s, a, 1.0, ns, False))
        loss = model.update(batch)
        assert isinstance(loss, float)

    def test_factory_create(self):
        model = DeepRLModelFactory.create('dqn', config=DQNConfig(num_actions=9), device='cpu')
        assert model is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestActorCriticModel:
    """Actor-Critic 模型测试"""

    def test_creation(self):
        from algorithms.deep_rl_models import ActorCriticModel
        model = ActorCriticModel(ActorCriticConfig(action_dim=4), device='cpu')
        assert model.config.action_dim == 4

    def test_predict(self):
        from algorithms.deep_rl_models import ActorCriticModel
        model = ActorCriticModel(ActorCriticConfig(action_dim=4), device='cpu')
        state = np.random.rand(3, 16, 16).astype(np.float32)
        action = model.predict(state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (4,)

    def test_update(self):
        from algorithms.deep_rl_models import ActorCriticModel
        model = ActorCriticModel(ActorCriticConfig(action_dim=4), device='cpu')
        batch = []
        for _ in range(5):
            s = np.random.rand(3, 8, 8).astype(np.float32)
            ns = np.random.rand(3, 8, 8).astype(np.float32)
            batch.append((s, np.random.rand(4), 1.0, ns, False))
        loss = model.update(batch)
        assert isinstance(loss, float)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPPOModel:
    """PPO 模型测试"""

    def test_creation(self):
        from algorithms.deep_rl_models import PPOModel
        model = PPOModel(PPOConfig(action_dim=4), device='cpu')
        assert model.config.action_dim == 4

    def test_predict(self):
        from algorithms.deep_rl_models import PPOModel
        model = PPOModel(PPOConfig(action_dim=4), device='cpu')
        state = np.random.rand(3, 16, 16).astype(np.float32)
        action = model.predict(state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (4,)

    def test_store_and_update(self):
        from algorithms.deep_rl_models import PPOModel
        model = PPOModel(PPOConfig(action_dim=4, minibatch_size=2), device='cpu')
        for _ in range(5):
            s = np.random.rand(3, 8, 8).astype(np.float32)
            action = model.predict(s)
            model.store_transition(s, action, reward=1.0, done=False)
        loss = model.update()
        assert isinstance(loss, float)


class TestReinforcementLearningOptimizerDeep:
    """强化学习优化器深度模型集成测试"""

    def test_multichannel_encoding_default(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer
        opt = ReinforcementLearningOptimizer(
            max_iter=5, seed=42, state_encoding='multichannel'
        )
        assert opt._state_encoder is not None
        assert opt.state_encoding == 'multichannel'

    def test_simple_encoding(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer
        opt = ReinforcementLearningOptimizer(
            max_iter=5, seed=42, state_encoding='simple'
        )
        assert opt._state_encoder is None

    def test_model_type_simple(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer
        opt = ReinforcementLearningOptimizer(
            max_iter=5, seed=42, model_type='simple'
        )
        assert opt.model_type == 'simple'
        assert opt._deep_model is None

    def test_optimize_with_multichannel(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer

        def objective(x):
            return np.sum(x ** 2)

        opt = ReinforcementLearningOptimizer(
            max_iter=10, seed=42,
            state_encoding='multichannel',
            model_type='simple',
        )
        x0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = opt.optimize(objective, x0, target=np.zeros_like(x0))
        assert result.x is not None
        assert len(result.history) > 0
        assert 'multichannel' in result.message

    def test_optimize_with_simple_encoding(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer

        def objective(x):
            return np.sum(x ** 2)

        opt = ReinforcementLearningOptimizer(
            max_iter=10, seed=42,
            state_encoding='simple',
            model_type='simple',
        )
        x0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = opt.optimize(objective, x0, target=np.zeros_like(x0))
        assert result.x is not None
        assert 'simple' in result.message

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_dqn_model_type(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer

        opt = ReinforcementLearningOptimizer(
            max_iter=5, seed=42,
            model_type='dqn',
            state_encoding='multichannel',
        )
        assert opt._deep_model is not None
        assert opt.model_type == 'dqn'

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_ppo_model_type(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer

        opt = ReinforcementLearningOptimizer(
            max_iter=5, seed=42,
            model_type='ppo',
            state_encoding='multichannel',
        )
        assert opt._deep_model is not None
        assert opt.model_type == 'ppo'

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_actor_critic_model_type(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer

        opt = ReinforcementLearningOptimizer(
            max_iter=5, seed=42,
            model_type='actor_critic',
            state_encoding='multichannel',
        )
        assert opt._deep_model is not None
        assert opt.model_type == 'actor_critic'

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_optimize_with_dqn(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer

        def objective(x):
            return np.sum(x ** 2)

        opt = ReinforcementLearningOptimizer(
            max_iter=10, seed=42,
            model_type='dqn',
            state_encoding='multichannel',
        )
        x0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = opt.optimize(objective, x0, target=np.zeros_like(x0))
        assert result.x is not None
        assert result.success

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_optimize_with_ppo(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer

        def objective(x):
            return np.sum(x ** 2)

        opt = ReinforcementLearningOptimizer(
            max_iter=10, seed=42,
            model_type='ppo',
            state_encoding='multichannel',
        )
        x0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = opt.optimize(objective, x0, target=np.zeros_like(x0))
        assert result.x is not None
        assert result.success

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_optimize_with_actor_critic(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer

        def objective(x):
            return np.sum(x ** 2)

        opt = ReinforcementLearningOptimizer(
            max_iter=10, seed=42,
            model_type='actor_critic',
            state_encoding='multichannel',
        )
        x0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = opt.optimize(objective, x0, target=np.zeros_like(x0))
        assert result.x is not None
        assert result.success

    def test_backward_compatible(self):
        from algorithms.advanced_optimizer import (
            ReinforcementLearningOptimizer, SimpleQLearningModel
        )

        def objective(x):
            return np.sum(x ** 2)

        x0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        state_dim = x0.size * 2
        action_dim = x0.size
        model = SimpleQLearningModel(state_dim, action_dim)

        opt = ReinforcementLearningOptimizer(
            max_iter=10, seed=42,
            state_encoding='simple',
            model_type='simple',
        )
        opt.set_model(model)
        result = opt.optimize(objective, x0)
        assert result.x is not None

    def test_custom_encoder_config(self):
        from algorithms.advanced_optimizer import ReinforcementLearningOptimizer
        from algorithms.deep_rl_models import StateEncoderConfig

        cfg = StateEncoderConfig(patch_size=4, history_length=32, freq_bins=16)
        opt = ReinforcementLearningOptimizer(
            max_iter=5, seed=42,
            state_encoding='multichannel',
            encoder_config=cfg,
        )
        assert opt._state_encoder.config.patch_size == 4
        assert opt._state_encoder.config.history_length == 32
