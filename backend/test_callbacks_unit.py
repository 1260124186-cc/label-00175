#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, tempfile, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.callbacks import (
    TrainerState,
    Callback,
    CallbackList,
    LambdaCallback,
    LearningRateSchedulerCallback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    MaskSnapshotCallback,
    ConvergencePlotCallback,
    HistoryCallback,
    LoggerCallback,
)


def test_trainer_state():
    """测试 TrainerState"""
    print("1. 测试 TrainerState...")
    state = TrainerState()
    assert state.epoch == 0
    assert state.loss == float('inf')
    assert state.stop_training == False
    state.loss_history.append(1.0)
    state.loss_history.append(0.5)
    assert len(state.loss_history) == 2
    print("   ✓ 通过")


def test_callback_list():
    """测试 CallbackList"""
    print("\n2. 测试 CallbackList...")
    
    state = TrainerState()
    callbacks = CallbackList()
    callbacks.set_state(state)
    
    called = {'train_begin': 0, 'train_end': 0, 'epoch_begin': 0, 'epoch_end': 0}
    
    def on_train_begin(logs=None):
        called['train_begin'] += 1
    def on_train_end(logs=None):
        called['train_end'] += 1
    def on_epoch_begin(epoch, logs=None):
        called['epoch_begin'] += 1
    def on_epoch_end(epoch, logs=None):
        called['epoch_end'] += 1
    
    cb = LambdaCallback(
        on_train_begin=on_train_begin,
        on_train_end=on_train_end,
        on_epoch_begin=on_epoch_begin,
        on_epoch_end=on_epoch_end
    )
    callbacks.append(cb)
    
    callbacks.on_train_begin()
    assert called['train_begin'] == 1
    
    for i in range(5):
        callbacks.on_epoch_begin(i)
        callbacks.on_epoch_end(i, {'loss': 1.0 - i * 0.1})
    
    assert called['epoch_begin'] == 5
    assert called['epoch_end'] == 5
    
    callbacks.on_train_end()
    assert called['train_end'] == 1
    
    print("   ✓ 通过")


def test_lr_scheduler():
    """测试学习率调度器回调"""
    print("\n3. 测试 LearningRateSchedulerCallback...")
    state = TrainerState()
    
    cb = LearningRateSchedulerCallback(
        initial_lr=0.1,
        scheduler_type='step',
        decay=0.5,
        step_size=3,
        min_lr=0.001
    )
    cb.set_state(state)
    
    cb.on_train_begin()
    assert state.learning_rate == 0.1
    
    for epoch in range(1, 11):
        cb.on_epoch_begin(epoch)
        cb.on_epoch_end(epoch, {'loss': 1.0 - epoch * 0.05})
    
    print(f"   最终学习率: {state.learning_rate:.6f}")
    assert state.learning_rate < 0.1
    assert state.learning_rate >= 0.001
    print("   ✓ 通过")


def test_early_stopping():
    """测试早停回调"""
    print("\n4. 测试 EarlyStoppingCallback...")
    state = TrainerState()
    
    cb = EarlyStoppingCallback(
        patience=3,
        min_delta=0.01,
        monitor='loss',
        mode='min',
        restore_best=True
    )
    cb.set_state(state)
    
    cb.on_train_begin({'loss': 1.0})
    
    losses = [0.9, 0.8, 0.79, 0.785, 0.783, 0.782, 0.781, 0.780]
    stop = False
    
    for i, loss in enumerate(losses):
        state.epoch = i + 1
        state.loss = loss
        state.mask = np.array([loss, loss])
        cb.on_epoch_end(i + 1, {'loss': loss})
        if state.stop_training:
            stop = True
            print(f"   在第 {i+1} 次迭代早停")
            break
    
    assert stop == True, "应该触发早停"
    assert state.best_loss < 1.0
    print("   ✓ 通过")


def test_model_checkpoint():
    """测试 checkpoint 回调"""
    print("\n5. 测试 ModelCheckpointCallback...")
    tmpdir = tempfile.mkdtemp()
    try:
        state = TrainerState()
        state.mask = np.random.rand(8, 8)
        state.loss = 0.5
        state.epoch = 0
        state.loss_history = [1.0, 0.8, 0.6, 0.5]
        state.lr_history = [0.1, 0.1, 0.1, 0.1]
        state.best_loss = 0.5
        state.best_mask = state.mask.copy()
        
        cb = ModelCheckpointCallback(
            checkpoint_dir=os.path.join(tmpdir, "ckpt"),
            save_freq=2,
            save_best_only=False,
            max_checkpoints=3,
            prefix='ckpt'
        )
        cb.set_state(state)
        
        cb.on_train_begin({'loss': 1.0})
        
        for epoch in range(1, 11):
            state.epoch = epoch
            state.loss = 1.0 - epoch * 0.08
            state.mask = np.random.rand(8, 8)
            state.loss_history.append(state.loss)
            state.lr_history.append(0.1)
            cb.on_epoch_end(epoch, {'loss': state.loss})
        
        ckpt_dir = os.path.join(tmpdir, "ckpt")
        files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.npz')])
        print(f"   生成了 {len(files)} 个 checkpoint 文件")
        
        if files:
            latest = os.path.join(ckpt_dir, files[-1])
            data = ModelCheckpointCallback.load_checkpoint(latest)
            print(f"   加载 checkpoint: epoch={data.get('epoch')}, loss={data.get('loss'):.4f}")
            assert 'mask' in data
            assert 'loss_history' in data
        print("   ✓ 通过")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mask_snapshot():
    """测试掩模快照回调"""
    print("\n6. 测试 MaskSnapshotCallback...")
    tmpdir = tempfile.mkdtemp()
    try:
        state = TrainerState()
        state.mask = np.random.rand(8, 8)
        state.loss = 0.5
        state.epoch = 0
        state.best_loss = 0.5
        state.best_mask = state.mask.copy()
        
        cb = MaskSnapshotCallback(
            save_dir=os.path.join(tmpdir, "snapshots"),
            save_freq=2,
            save_best=True,
            save_npy=True
        )
        cb.set_state(state)
        
        cb.on_train_begin()
        
        for epoch in range(1, 11):
            state.epoch = epoch
            state.loss = 1.0 - epoch * 0.08
            state.mask = np.random.rand(8, 8)
            if state.loss < state.best_loss:
                state.best_loss = state.loss
                state.best_mask = state.mask.copy()
            cb.on_epoch_end(epoch, {'loss': state.loss})
        
        snap_dir = os.path.join(tmpdir, "snapshots")
        if os.path.exists(snap_dir):
            files = os.listdir(snap_dir)
            print(f"   生成了 {len(files)} 个快照文件")
            if files:
                print(f"   文件示例: {files[0]}")
        print("   ✓ 通过")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_convergence_plot():
    """测试收敛曲线绘制回调"""
    print("\n7. 测试 ConvergencePlotCallback...")
    tmpdir = tempfile.mkdtemp()
    try:
        state = TrainerState()
        state.epoch = 0
        state.loss = 1.0
        state.learning_rate = 0.1
        state.loss_history = []
        state.lr_history = []
        
        cb = ConvergencePlotCallback(
            save_dir=os.path.join(tmpdir, "plots"),
            plot_freq=5,
            log_scale=False,
            plot_lr=True
        )
        cb.set_state(state)
        
        cb.on_train_begin()
        
        np.random.seed(42)
        for epoch in range(1, 21):
            state.epoch = epoch
            loss = 1.0 * np.exp(-0.1 * epoch) + 0.01 * np.random.randn()
            state.loss = max(0.001, loss)
            state.learning_rate = 0.1 * (0.95 ** epoch)
            state.loss_history.append(state.loss)
            state.lr_history.append(state.learning_rate)
            cb.on_epoch_end(epoch, {'loss': state.loss})
        
        cb.on_train_end()
        
        plot_file = os.path.join(tmpdir, "plots", 'convergence_curve.png')
        if os.path.exists(plot_file):
            print(f"   生成曲线图: {os.path.getsize(plot_file)} bytes")
        print("   ✓ 通过")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_history_callback():
    """测试历史记录回调"""
    print("\n8. 测试 HistoryCallback...")
    state = TrainerState()
    state.learning_rate = 0.1
    
    cb = HistoryCallback()
    cb.set_state(state)
    
    cb.on_train_begin({'loss': 1.0})
    
    for i in range(10):
        state.epoch = i
        state.loss = 1.0 - i * 0.1
        state.loss_history.append(state.loss)
        cb.on_epoch_end(i, {'loss': state.loss})
    
    history = cb.get_history()
    assert len(history['loss']) == 10
    print(f"   记录了 {len(history['loss'])} 个损失值")
    print("   ✓ 通过")


def test_callback_integration():
    """测试多个回调协同工作"""
    print("\n9. 测试多回调协同工作...")
    
    state = TrainerState()
    state.learning_rate = 0.1
    
    callbacks = CallbackList()
    callbacks.set_state(state)
    
    lr_cb = LearningRateSchedulerCallback(
        initial_lr=0.1,
        scheduler_type='exponential',
        decay=0.95,
        min_lr=0.001
    )
    
    es_cb = EarlyStoppingCallback(
        patience=5,
        min_delta=0.001,
        restore_best=True
    )
    
    hist_cb = HistoryCallback()
    
    callbacks.append(lr_cb)
    callbacks.append(es_cb)
    callbacks.append(hist_cb)
    
    callbacks.on_train_begin({'loss': 1.0})
    
    np.random.seed(42)
    for epoch in range(1, 50):
        callbacks.on_epoch_begin(epoch)
        
        loss = 1.0 * np.exp(-0.05 * epoch) + 0.005 * np.random.randn()
        state.loss = max(0.001, loss)
        state.mask = np.array([loss])
        state.epoch = epoch
        state.loss_history.append(state.loss)
        
        stop = callbacks.on_epoch_end(epoch, {'loss': state.loss})
        
        if stop:
            print(f"   在 epoch {epoch} 停止")
            break
    
    history = hist_cb.get_history()
    print(f"   共运行 {len(history['loss'])} 个 epoch")
    print(f"   最终学习率: {state.learning_rate:.6f}")
    print("   ✓ 通过")


def main():
    print("=" * 60)
    print("Callback 系统单元测试")
    print("=" * 60)
    
    tests = [
        test_trainer_state,
        test_callback_list,
        test_lr_scheduler,
        test_early_stopping,
        test_model_checkpoint,
        test_mask_snapshot,
        test_convergence_plot,
        test_history_callback,
        test_callback_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"   ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"结果: {passed} 个通过, {failed} 个失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
