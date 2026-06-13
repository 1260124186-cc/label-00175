# -*- coding: utf-8 -*-
"""
快速端到端验证：测试优化过程动画生成（GIF）
验证 AnimationCallback 能够正确生成包含掩模、空间像、误差图和收敛曲线的动画。
"""

import os
import sys
import tempfile
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from core.imaging import OpticalSystem
from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig


def create_test_patterns(shape=(64, 64)):
    h, w = shape
    target = np.zeros((h, w), dtype=np.float64)
    cx, cy = w // 2, h // 2
    r = min(h, w) // 4
    for i in range(h):
        for j in range(w):
            if (i - cy) ** 2 + (j - cx) ** 2 <= r * r:
                target[i, j] = 1.0
    target[max(0, h // 2 - 3):min(h, h // 2 + 3), :] = 1.0
    target[:, max(0, w // 2 - 3):min(w, w // 2 + 3)] = 1.0

    rng = np.random.RandomState(42)
    initial = rng.rand(h, w)
    initial = np.clip(0.5 + 0.3 * (initial - 0.5), 0.0, 1.0)
    return initial, target


def main():
    tmp_dir = tempfile.mkdtemp(prefix='anim_test_')
    anim_dir = os.path.join(tmp_dir, 'animations')
    os.makedirs(anim_dir, exist_ok=True)
    print(f"[INFO] 测试输出目录: {tmp_dir}")

    initial_mask, target_img = create_test_patterns((64, 64))
    optics = OpticalSystem(na=0.75, sigma=0.5, wavelength=193.0)

    optimizers_to_test = [
        ('adam (step-training)', 'adam', 30),
        ('bfgs (callback-driven)', 'bfgs', 15),
    ]

    all_ok = True
    for label, opt_type, max_iter in optimizers_to_test:
        cfg = OptimizationConfig(
            optimizer_type=opt_type,
            max_iter=max_iter,
            learning_rate=0.1,
            verbose=True,
            random_seed=42,
            use_callbacks=True,
            animation_enable=True,
            animation_dir=anim_dir,
            animation_freq=1,  # 每步都记录，确保即使迭代少也有帧
            animation_format='gif',
            animation_fps=5,
            animation_dpi=80,
            animation_show_info=True,
            animation_show_convergence=True,
            animation_consistent_error=True,
            animation_show_wafer=False,
            use_wafer_image_loss=False,
            callback_log_freq=10,
            early_stopping_enable=False,
            plot_enable=False,
            checkpoint_enable=False,
            snapshot_enable=False,
        )

        print(f"\n{'='*60}")
        print(f"[TEST] {label}")
        print(f"{'='*60}")

        try:
            optimizer = MaskOptimizer(optical_system=optics, config=cfg)
            result = optimizer.optimize(initial_mask, target_img)
            print(f"  -> 完成 {result.total_iterations} 次迭代, "
                  f"初始MSE={result.initial_metrics.mse:.4e}, "
                  f"最终MSE={result.final_metrics.mse:.4e}")

            gif_file = os.path.join(anim_dir, 'optimization_animation.gif')
            if os.path.exists(gif_file):
                sz = os.path.getsize(gif_file)
                print(f"  -> 动画已生成: {gif_file} ({sz} bytes)")
                if sz < 1024:
                    print(f"  !! 警告: 文件过小，可能生成有问题")
                    all_ok = False
            else:
                print(f"  !! 错误: 动画文件未找到: {gif_file}")
                all_ok = False
        except Exception as e:
            print(f"  !! 失败: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    print(f"\n{'='*60}")
    print(f"[SUMMARY] 结果: {'全部通过' if all_ok else '存在问题'}")
    print(f"[SUMMARY] 输出目录可手动查看: {anim_dir}")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
