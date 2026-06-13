# -*- coding: utf-8 -*-
"""
误差评估模块单元测试
"""

import pytest
import numpy as np
import time
from core.metrics import (
    mse, mae, ssim, normalized_correlation, psnr,
    evaluate_all, batch_evaluate, compute_error_map,
    MetricsResult,
    ssim_gradient, ssim_loss_gradient,
    HistoryEvaluationRow,
    batch_evaluate_history,
    export_evaluation_csv,
    compute_pareto_front,
    evaluate_and_export_pareto,
    total_variation,
    total_variation_isotropic,
    manhattan_distance_penalty,
)


class TestMSE:
    """MSE测试"""

    def test_identical_images(self):
        """测试相同图像MSE为0"""
        img = np.random.random((32, 32))
        result = mse(img, img)

        assert abs(result) < 1e-10

    def test_mse_positive(self):
        """测试MSE非负"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        result = mse(img1, img2)

        assert result >= 0

    def test_mse_symmetric(self):
        """测试MSE对称性"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        assert abs(mse(img1, img2) - mse(img2, img1)) < 1e-10

    def test_mse_known_value(self):
        """测试已知MSE值"""
        img1 = np.zeros((2, 2))
        img2 = np.ones((2, 2))

        # MSE = mean((0-1)^2) = 1
        assert abs(mse(img1, img2) - 1.0) < 1e-10


class TestMAE:
    """MAE测试"""

    def test_identical_images(self):
        """测试相同图像MAE为0"""
        img = np.random.random((32, 32))
        result = mae(img, img)

        assert abs(result) < 1e-10

    def test_mae_positive(self):
        """测试MAE非负"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        result = mae(img1, img2)

        assert result >= 0

    def test_mae_known_value(self):
        """测试已知MAE值"""
        img1 = np.zeros((2, 2))
        img2 = np.ones((2, 2))

        # MAE = mean(|0-1|) = 1
        assert abs(mae(img1, img2) - 1.0) < 1e-10


class TestSSIM:
    """SSIM测试"""

    def test_identical_images(self):
        """测试相同图像SSIM为1"""
        img = np.random.random((32, 32))
        result = ssim(img, img)

        assert abs(result - 1.0) < 0.01  # 允许小误差

    def test_ssim_range(self):
        """测试SSIM范围"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        result = ssim(img1, img2)

        assert -1 <= result <= 1

    def test_ssim_symmetric(self):
        """测试SSIM对称性"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        assert abs(ssim(img1, img2) - ssim(img2, img1)) < 0.01


class TestNormalizedCorrelation:
    """归一化相关系数测试"""

    def test_identical_images(self):
        """测试相同图像NCC为1"""
        img = np.random.random((32, 32))
        result = normalized_correlation(img, img)

        assert abs(result - 1.0) < 1e-10

    def test_ncc_range(self):
        """测试NCC范围"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        result = normalized_correlation(img1, img2)

        assert -1 <= result <= 1

    def test_negative_correlation(self):
        """测试负相关"""
        img1 = np.array([[1, 0], [0, 1]], dtype=float)
        img2 = np.array([[0, 1], [1, 0]], dtype=float)

        result = normalized_correlation(img1, img2)

        assert result < 0


class TestPSNR:
    """PSNR测试"""

    def test_identical_images(self):
        """测试相同图像PSNR很高"""
        img = np.random.random((32, 32))
        result = psnr(img, img)

        assert result >= 100  # 应该返回最大值

    def test_psnr_positive(self):
        """测试PSNR为正"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        result = psnr(img1, img2)

        assert result > 0


class TestEvaluateAll:
    """综合评估测试"""

    def test_evaluate_all_returns_metrics_result(self):
        """测试返回MetricsResult对象"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        result = evaluate_all(img1, img2)

        assert isinstance(result, MetricsResult)

    def test_evaluate_all_has_all_metrics(self):
        """测试包含所有指标"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        result = evaluate_all(img1, img2)

        assert hasattr(result, 'mse')
        assert hasattr(result, 'mae')
        assert hasattr(result, 'ssim')
        assert hasattr(result, 'ncc')
        assert hasattr(result, 'psnr')

    def test_to_dict(self):
        """测试转换为字典"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        result = evaluate_all(img1, img2)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'mse' in result_dict
        assert 'ssim' in result_dict


class TestBatchEvaluate:
    """批量评估测试"""

    def test_batch_evaluate_length(self):
        """测试批量评估结果长度"""
        images = [np.random.random((32, 32)) for _ in range(5)]
        target = np.random.random((32, 32))

        results = batch_evaluate(images, target)

        assert len(results) == 5

    def test_batch_evaluate_selected_metrics(self):
        """测试选择性指标评估"""
        images = [np.random.random((32, 32)) for _ in range(3)]
        target = np.random.random((32, 32))

        results = batch_evaluate(images, target, metrics=['mse', 'mae'])

        for result in results:
            assert 'mse' in result
            assert 'mae' in result
            assert 'ssim' not in result


class TestErrorMap:
    """误差分布图测试"""

    def test_absolute_error_map(self):
        """测试绝对误差图"""
        img1 = np.zeros((32, 32))
        img2 = np.ones((32, 32))

        error_map = compute_error_map(img1, img2, 'absolute')

        np.testing.assert_array_almost_equal(error_map, np.ones((32, 32)))

    def test_squared_error_map(self):
        """测试平方误差图"""
        img1 = np.zeros((32, 32))
        img2 = np.ones((32, 32)) * 2

        error_map = compute_error_map(img1, img2, 'squared')

        np.testing.assert_array_almost_equal(error_map, np.ones((32, 32)) * 4)

    def test_signed_error_map(self):
        """测试有符号误差图"""
        img1 = np.ones((32, 32))
        img2 = np.zeros((32, 32))

        error_map = compute_error_map(img1, img2, 'signed')

        np.testing.assert_array_almost_equal(error_map, np.ones((32, 32)))

    def test_invalid_error_type(self):
        """测试无效误差类型"""
        img1 = np.random.random((32, 32))
        img2 = np.random.random((32, 32))

        with pytest.raises(ValueError):
            compute_error_map(img1, img2, 'invalid')


def _numerical_ssim_gradient(image1: np.ndarray,
                             image2: np.ndarray,
                             eps: float = 1e-5,
                             window_size: int = 11,
                             k1: float = 0.01,
                             k2: float = 0.03,
                             data_range: float = 1.0) -> np.ndarray:
    """数值差分计算 SSIM 梯度（用于验证解析梯度）"""
    ny, nx = image1.shape
    grad = np.zeros((ny, nx), dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            img_plus = image1.copy()
            img_plus[i, j] += eps
            img_minus = image1.copy()
            img_minus[i, j] -= eps
            ssim_plus = ssim(img_plus, image2, window_size, k1, k2, data_range)
            ssim_minus = ssim(img_minus, image2, window_size, k1, k2, data_range)
            grad[i, j] = (ssim_plus - ssim_minus) / (2 * eps)
    return grad


class TestSSIMGradient:
    """SSIM 解析梯度测试"""

    def test_gradient_shape(self):
        """测试梯度输出形状与输入一致"""
        np.random.seed(42)
        img1 = np.random.random((16, 16))
        img2 = np.random.random((16, 16))
        grad = ssim_gradient(img1, img2)
        assert grad.shape == img1.shape

    def test_gradient_sign_vs_ssim_loss(self):
        """测试 ssim_loss_gradient = -ssim_gradient"""
        np.random.seed(42)
        img1 = np.random.random((16, 16))
        img2 = np.random.random((16, 16))
        grad_ssim = ssim_gradient(img1, img2)
        grad_loss = ssim_loss_gradient(img1, img2)
        np.testing.assert_array_almost_equal(grad_loss, -grad_ssim)

    def test_analytical_vs_numerical_small(self):
        """小图像上解析梯度与数值梯度对比（8x8）"""
        np.random.seed(42)
        img1 = np.random.random((8, 8))
        img2 = np.random.random((8, 8))
        grad_analytical = ssim_gradient(img1, img2, window_size=3)
        grad_numerical = _numerical_ssim_gradient(img1, img2, window_size=3, eps=1e-5)
        rel_error = np.max(np.abs(grad_analytical - grad_numerical)) / (
            np.max(np.abs(grad_numerical)) + 1e-12
        )
        assert rel_error < 1e-3, f"相对误差 {rel_error:.2e} 超过阈值 1e-3"

    def test_analytical_vs_numerical_medium(self):
        """中等图像上解析梯度与数值梯度对比（12x12）"""
        np.random.seed(123)
        img1 = np.random.random((12, 12))
        img2 = np.random.random((12, 12))
        grad_analytical = ssim_gradient(img1, img2, window_size=5)
        grad_numerical = _numerical_ssim_gradient(img1, img2, window_size=5, eps=1e-5)
        rel_error = np.max(np.abs(grad_analytical - grad_numerical)) / (
            np.max(np.abs(grad_numerical)) + 1e-12
        )
        assert rel_error < 1e-3, f"相对误差 {rel_error:.2e} 超过阈值 1e-3"

    def test_identical_images_gradient_finite(self):
        """相同图像的梯度应为有限值（不全为 0，因均值等仍可能变化）"""
        np.random.seed(7)
        img = np.random.random((16, 16))
        grad = ssim_gradient(img, img)
        assert np.all(np.isfinite(grad))

    def test_constant_image_gradient(self):
        """常数图像：梯度数值应很小（SSIM≈1，对小扰动不敏感）"""
        img1 = 0.5 * np.ones((10, 10), dtype=np.float64)
        img2 = 0.5 * np.ones((10, 10), dtype=np.float64)
        grad = ssim_gradient(img1, img2, window_size=3)
        assert np.max(np.abs(grad)) < 1e-6

    def test_performance_speedup(self):
        """验证解析梯度远快于数值梯度（64x64 尺寸下）"""
        np.random.seed(99)
        size = 64
        img1 = np.random.random((size, size))
        img2 = np.random.random((size, size))

        t0 = time.time()
        _ = ssim_gradient(img1, img2)
        _ = ssim_gradient(img1, img2)
        t_analytical = time.time() - t0

        n_pixels_numerical = 100
        idx = np.random.choice(size * size, n_pixels_numerical, replace=False)
        eps = 1e-5
        t0 = time.time()
        grad_num_partial = np.zeros(size * size, dtype=np.float64)
        for k in idx:
            i, j = divmod(k, size)
            img_p = img1.copy()
            img_p[i, j] += eps
            img_m = img1.copy()
            img_m[i, j] -= eps
            ssim_p = ssim(img_p, img2)
            ssim_m = ssim(img_m, img2)
            grad_num_partial[k] = (ssim_p - ssim_m) / (2 * eps)
        t_numerical_partial = time.time() - t0

        estimated_numerical_total = t_numerical_partial * (size * size) / n_pixels_numerical
        speedup = estimated_numerical_total / max(t_analytical, 1e-9)
        assert speedup >= 10.0, (
            f"解析梯度加速比 {speedup:.1f}x 低于预期 10x; "
            f"解析耗时 {t_analytical:.3f}s, 估计数值耗时 {estimated_numerical_total:.3f}s"
        )


class TestBatchEvaluateExtended:
    """batch_evaluate 扩展掩模复杂度指标测试"""

    def test_batch_evaluate_with_mask_complexity(self):
        """测试 batch_evaluate 支持 mask_complexity 指标"""
        images = [np.random.random((16, 16)) for _ in range(3)]
        target = np.random.random((16, 16))
        results = batch_evaluate(
            images, target,
            metrics=['mse', 'mask_complexity', 'tv', 'binary_penalty']
        )
        assert len(results) == 3
        for r in results:
            assert 'mse' in r
            assert 'mask_complexity' in r
            assert 'tv' in r
            assert 'binary_penalty' in r
            assert r['mask_complexity'] >= 0
            assert r['tv'] >= 0
            assert 0 <= r['binary_penalty'] <= 1

    def test_batch_evaluate_mask_complexity_equals_total_variation(self):
        """mask_complexity 与 tv 指标应等价于 total_variation"""
        img = np.random.random((16, 16))
        target = np.random.random((16, 16))
        results = batch_evaluate([img], target, metrics=['mask_complexity', 'tv'])
        expected_tv = total_variation(img)
        assert abs(results[0]['mask_complexity'] - expected_tv) < 1e-10
        assert abs(results[0]['tv'] - expected_tv) < 1e-10


class TestBatchEvaluateHistory:
    """batch_evaluate_history 批量评估优化历史测试"""

    def test_evaluate_history_basic(self):
        """基本功能：给定掩模列表，返回评估行列表"""
        np.random.seed(0)
        masks = [np.random.random((16, 16)) for _ in range(5)]
        target = np.random.random((16, 16))
        rows = batch_evaluate_history(masks, target)
        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert row.step == i
            assert isinstance(row.mse, float) and row.mse >= 0
            assert isinstance(row.ssim, float) and -1 <= row.ssim <= 1
            assert isinstance(row.mask_complexity, float) and row.mask_complexity >= 0
            assert isinstance(row.tv, float) and row.tv >= 0
            assert isinstance(row.binary_penalty, float)

    def test_evaluate_history_with_loss_history(self):
        """测试传入 loss_history"""
        masks = [np.random.random((16, 16)) for _ in range(3)]
        target = np.random.random((16, 16))
        losses = [0.5, 0.3, 0.2]
        rows = batch_evaluate_history(masks, target, loss_history=losses)
        assert rows[0].loss == 0.5
        assert rows[1].loss == 0.3
        assert rows[2].loss == 0.2

    def test_history_evaluation_row_to_dict(self):
        """测试 HistoryEvaluationRow.to_dict"""
        row = HistoryEvaluationRow(
            step=2, loss=0.1, mse=0.01, mae=0.05, ssim=0.95,
            ncc=0.9, psnr=30.0, mask_complexity=100.0,
            tv=100.0, binary_penalty=0.1,
        )
        d = row.to_dict()
        assert d['step'] == 2
        assert d['loss'] == 0.1
        assert d['mse'] == 0.01
        assert d['ssim'] == 0.95
        assert d['mask_complexity'] == 100.0
        assert 'wafer_image' not in d


class TestExportEvaluationCSV:
    """export_evaluation_csv 导出功能测试"""

    def test_export_csv_creates_file(self, tmp_path):
        """测试 CSV 文件被正确创建并包含正确字段"""
        rows = [
            HistoryEvaluationRow(
                step=i, loss=1.0 / (i + 1),
                mse=0.01 * i, mae=0.02 * i, ssim=0.9 - 0.01 * i,
                ncc=0.9 - 0.02 * i, psnr=40.0 - i,
                mask_complexity=100.0 + i * 10,
                tv=100.0 + i * 10,
                binary_penalty=0.5 - 0.01 * i,
            )
            for i in range(5)
        ]
        csv_path = tmp_path / "evaluation.csv"
        result_path = export_evaluation_csv(rows, str(csv_path))
        assert csv_path.exists()
        assert result_path == str(csv_path.resolve())

        with open(csv_path, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split(',')
        expected_cols = [
            'step', 'loss', 'mse', 'mae', 'ssim', 'ncc', 'psnr',
            'mask_complexity', 'tv', 'binary_penalty'
        ]
        for col in expected_cols:
            assert col in header
        assert len(lines) == 6  # header + 5 rows

    def test_export_csv_with_extra_columns(self, tmp_path):
        """测试额外列写入"""
        rows = [
            HistoryEvaluationRow(
                step=i, loss=1.0, mse=0.01, mae=0.02, ssim=0.9,
                ncc=0.8, psnr=30.0, mask_complexity=100.0,
                tv=100.0, binary_penalty=0.1,
            )
            for i in range(3)
        ]
        extra = {'custom_col': ['a', 'b', 'c']}
        csv_path = tmp_path / "eval_extra.csv"
        export_evaluation_csv(rows, str(csv_path), extra_columns=extra)
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split(',')
        assert 'custom_col' in header
        for i in range(1, 4):
            assert lines[i].strip().split(',')[-1] == ['a', 'b', 'c'][i - 1]


class TestComputeParetoFront:
    """compute_pareto_front Pareto 前沿计算测试"""

    def test_pareto_front_simple(self):
        """简单场景：三个点，其中一个被支配"""
        rows = [
            HistoryEvaluationRow(
                step=0, loss=1.0, mse=0.10, mae=0, ssim=0, ncc=0, psnr=0,
                mask_complexity=100.0, tv=100.0, binary_penalty=0,
            ),
            HistoryEvaluationRow(
                step=1, loss=0.5, mse=0.05, mae=0, ssim=0, ncc=0, psnr=0,
                mask_complexity=150.0, tv=150.0, binary_penalty=0,
            ),
            HistoryEvaluationRow(
                step=2, loss=0.2, mse=0.02, mae=0, ssim=0, ncc=0, psnr=0,
                mask_complexity=200.0, tv=200.0, binary_penalty=0,
            ),
        ]
        front = compute_pareto_front(rows, 'mask_complexity', 'mse')
        assert len(front) == 3  # 三点各自在X或Y上有优势，均非支配

    def test_pareto_front_dominated_point(self):
        """存在完全被支配的点应被剔除"""
        rows = [
            HistoryEvaluationRow(
                step=0, loss=1.0, mse=0.10, mae=0, ssim=0, ncc=0, psnr=0,
                mask_complexity=100.0, tv=100.0, binary_penalty=0,
            ),
            HistoryEvaluationRow(
                step=1, loss=0.5, mse=0.15, mae=0, ssim=0, ncc=0, psnr=0,
                mask_complexity=200.0, tv=200.0, binary_penalty=0,
            ),
            HistoryEvaluationRow(
                step=2, loss=0.2, mse=0.02, mae=0, ssim=0, ncc=0, psnr=0,
                mask_complexity=300.0, tv=300.0, binary_penalty=0,
            ),
        ]
        # step=1 在 mse 和 mask_complexity 上都比 step=0 差，被支配
        front = compute_pareto_front(rows, 'mask_complexity', 'mse')
        steps = [r.step for r in front]
        assert 1 not in steps
        assert 0 in steps
        assert 2 in steps

    def test_pareto_front_sorted_by_x(self):
        """返回的前沿应按 objective_x 升序排列"""
        rows = [
            HistoryEvaluationRow(
                step=i, loss=1.0, mse=0.1 - i * 0.02, mae=0, ssim=0, ncc=0, psnr=0,
                mask_complexity=float(200 - i * 50),
                tv=0, binary_penalty=0,
            )
            for i in range(4)
        ]
        front = compute_pareto_front(rows, 'mask_complexity', 'mse')
        xs = [r.mask_complexity for r in front]
        assert xs == sorted(xs)

    def test_pareto_front_maximize(self):
        """测试最大化目标方向"""
        rows = [
            HistoryEvaluationRow(
                step=0, loss=1.0, mse=0, mae=0, ssim=0.90, ncc=0, psnr=0,
                mask_complexity=100.0, tv=100.0, binary_penalty=0,
            ),
            HistoryEvaluationRow(
                step=1, loss=0.5, mse=0, mae=0, ssim=0.95, ncc=0, psnr=0,
                mask_complexity=150.0, tv=150.0, binary_penalty=0,
            ),
        ]
        # 最大化 SSIM（越大越好），最小化复杂度
        front = compute_pareto_front(
            rows, 'mask_complexity', 'ssim',
            minimize_x=True, minimize_y=False,
        )
        assert len(front) == 2  # 两者互不支配


class TestEvaluateAndExportPareto:
    """evaluate_and_export_pareto 一站式流水线测试"""

    def test_end_to_end_pipeline(self, tmp_path):
        """端到端：评估 → 导出 CSV → Pareto 前沿 → 导出前沿 CSV"""
        np.random.seed(42)
        n_steps = 10
        masks = []
        for i in range(n_steps):
            base = np.full((16, 16), 0.3 + 0.05 * i)
            noise = np.random.random((16, 16)) * 0.1
            masks.append(np.clip(base + noise, 0, 1))
        target = np.full((16, 16), 0.5)

        csv_path = tmp_path / "full_eval.csv"
        rows, pareto = evaluate_and_export_pareto(
            masks=masks,
            target=target,
            csv_path=str(csv_path),
            objective_x='mask_complexity',
            objective_y='mse',
        )
        assert len(rows) == n_steps
        assert csv_path.exists()
        pareto_path = tmp_path / "full_eval_pareto.csv"
        assert pareto_path.exists()
        assert len(pareto) >= 1
        for p in pareto:
            assert p.step < n_steps

    def test_custom_pareto_csv_path(self, tmp_path):
        """测试自定义 pareto CSV 输出路径"""
        masks = [np.random.random((8, 8)) for _ in range(3)]
        target = np.random.random((8, 8))
        csv_path = tmp_path / "main.csv"
        custom_pareto = tmp_path / "my_pareto.csv"
        rows, pareto = evaluate_and_export_pareto(
            masks, target, str(csv_path),
            pareto_csv_path=str(custom_pareto),
        )
        assert custom_pareto.exists()
        assert len(pareto) >= 1
