# -*- coding: utf-8 -*-
"""
结果导出功能单元测试：GDS层、NPY、HDF5批量存储
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from utils.data_io import (
    save_npy, load_npy,
    save_gds_layer, load_gds_layer,
    save_hdf5_results, load_hdf5_results,
    save_optimization_result
)

try:
    import gdstk
    HAS_GDSTK = True
except ImportError:
    HAS_GDSTK = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _create_test_mask(size=(64, 64)):
    """创建测试用二值掩模"""
    mask = np.zeros(size, dtype=np.float64)
    mask[10:30, 10:30] = 1.0
    mask[40:55, 20:50] = 1.0
    mask[5:15, 45:60] = 1.0
    return mask


class TestNpyIO:
    """NPY格式读写测试"""

    def test_save_and_load_npy(self, tmp_path):
        """测试保存和加载 npy 文件"""
        mask = _create_test_mask()
        npy_path = tmp_path / "test_mask.npy"

        save_npy(mask, str(npy_path))

        assert npy_path.exists()

        loaded = load_npy(str(npy_path))
        np.testing.assert_array_equal(mask, loaded)
        assert loaded.dtype == mask.dtype
        assert loaded.shape == mask.shape

    def test_save_npy_creates_parent_dir(self, tmp_path):
        """测试 save_npy 自动创建父目录"""
        mask = _create_test_mask()
        npy_path = tmp_path / "subdir" / "deep" / "mask.npy"

        save_npy(mask, str(npy_path))
        assert npy_path.exists()

    def test_load_npy_nonexistent_raises(self, tmp_path):
        """测试加载不存在的文件抛出异常"""
        with pytest.raises(FileNotFoundError):
            load_npy(str(tmp_path / "nonexistent.npy"))


@pytest.mark.skipif(not HAS_GDSTK, reason="gdstk未安装")
class TestGdsExport:
    """GDS层导出测试"""

    def test_save_and_reload_gds_layer(self, tmp_path):
        """测试保存 GDS 后重新加载，掩模内容一致"""
        mask = _create_test_mask(size=(32, 32))
        gds_path = tmp_path / "test_mask.gds"

        save_gds_layer(
            mask, str(gds_path),
            layer=0, datatype=0,
            pixel_size=1.0,
            origin=(0.0, 0.0)
        )

        assert gds_path.exists()

        reloaded = load_gds_layer(
            str(gds_path), layer=0, datatype=0,
            pixel_size=1.0,
            target_size=mask.shape,
            bounds=(0, 0, mask.shape[1], mask.shape[0])
        )

        assert reloaded.shape == mask.shape
        overlap = np.sum((mask > 0.5) & (reloaded > 0.5))
        union = np.sum((mask > 0.5) | (reloaded > 0.5))
        iou = overlap / union if union > 0 else 0.0
        assert iou > 0.9

    def test_save_gds_custom_layer_datatype(self, tmp_path):
        """测试自定义 layer 和 datatype"""
        mask = _create_test_mask(size=(32, 32))
        gds_path = tmp_path / "custom_layer.gds"

        save_gds_layer(mask, str(gds_path), layer=5, datatype=2)

        reloaded = load_gds_layer(
            str(gds_path), layer=5, datatype=2,
            pixel_size=1.0,
            target_size=mask.shape,
            bounds=(0, 0, mask.shape[1], mask.shape[0])
        )

        assert np.any(reloaded > 0)

    def test_save_gds_pixel_size_scaling(self, tmp_path):
        """测试像素尺寸缩放"""
        mask = _create_test_mask(size=(16, 16))
        gds_path = tmp_path / "scaled.gds"
        pixel_size = 10.0

        save_gds_layer(mask, str(gds_path), pixel_size=pixel_size)

        ny, nx = mask.shape
        reloaded = load_gds_layer(
            str(gds_path), layer=0, datatype=0,
            pixel_size=pixel_size,
            target_size=mask.shape,
            bounds=(0, 0, nx * pixel_size, ny * pixel_size)
        )

        assert reloaded.shape == mask.shape

    def test_save_gds_invalid_mask_raises(self, tmp_path):
        """测试非二维掩模抛出异常"""
        mask_3d = np.zeros((8, 8, 3))
        gds_path = tmp_path / "invalid.gds"

        with pytest.raises(ValueError, match="二维数组"):
            save_gds_layer(mask_3d, str(gds_path))

    def test_save_gds_creates_parent_dir(self, tmp_path):
        """测试 save_gds_layer 自动创建父目录"""
        mask = _create_test_mask(size=(16, 16))
        gds_path = tmp_path / "nested" / "dir" / "mask.gds"

        save_gds_layer(mask, str(gds_path))
        assert gds_path.exists()

    def test_save_gds_threshold(self, tmp_path):
        """测试二值化阈值"""
        mask = np.array([[0.0, 0.3, 0.6, 1.0],
                         [0.2, 0.4, 0.7, 0.9],
                         [0.1, 0.5, 0.8, 0.95],
                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        gds_path = tmp_path / "threshold.gds"

        save_gds_layer(mask, str(gds_path), threshold=0.5, pixel_size=1.0)

        expected = (mask > 0.5).astype(np.float64)
        reloaded = load_gds_layer(
            str(gds_path), layer=0, datatype=0,
            pixel_size=1.0,
            target_size=mask.shape,
            bounds=(0, 0, mask.shape[1], mask.shape[0])
        )

        overlap = np.sum((expected > 0.5) & (reloaded > 0.5))
        union = np.sum((expected > 0.5) | (reloaded > 0.5))
        iou = overlap / union if union > 0 else 0.0
        assert iou > 0.8


@pytest.mark.skipif(not HAS_H5PY, reason="h5py未安装")
class TestHdf5Export:
    """HDF5批量存储测试"""

    def test_save_and_load_mask_sequence(self, tmp_path):
        """测试保存和加载掩模序列"""
        masks = [_create_test_mask(size=(32, 32)) for _ in range(5)]
        hdf5_path = tmp_path / "masks.h5"

        save_hdf5_results(str(hdf5_path), mask_sequence=masks)

        assert hdf5_path.exists()
        result = load_hdf5_results(str(hdf5_path))

        assert result['mask_sequence'] is not None
        assert result['mask_sequence'].shape == (5, 32, 32)
        for i in range(5):
            np.testing.assert_array_almost_equal(
                result['mask_sequence'][i], masks[i], decimal=10
            )

    def test_save_and_load_loss_history(self, tmp_path):
        """测试保存和加载损失历史"""
        loss_history = [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]
        hdf5_path = tmp_path / "loss.h5"

        save_hdf5_results(str(hdf5_path), loss_history=loss_history)

        result = load_hdf5_results(str(hdf5_path))
        assert result['loss_history'] is not None
        np.testing.assert_array_almost_equal(
            result['loss_history'], np.array(loss_history), decimal=10
        )

    def test_save_and_load_optical_params(self, tmp_path):
        """测试保存和加载光学参数"""
        optical_params = {
            'wavelength': 193.0,
            'na': 1.35,
            'sigma': 0.75,
            'pixel_size': 1.0,
            'defocus': 0.0,
            'magnification': 4.0,
            'illumination_type': 'conventional',
            'source_params': {'sigma_inner': 0.0, 'sigma_outer': 0.75},
            'zernike_coefficients': {'spherical': 0.05, 'defocus': 0.02}
        }
        hdf5_path = tmp_path / "optics.h5"

        save_hdf5_results(str(hdf5_path), optical_params=optical_params)

        result = load_hdf5_results(str(hdf5_path))
        assert result['optical_params'] is not None
        assert result['optical_params']['wavelength'] == 193.0
        assert result['optical_params']['na'] == 1.35
        assert result['optical_params']['illumination_type'] == 'conventional'

    def test_save_complete_results(self, tmp_path):
        """测试完整结果（掩模序列+损失历史+光学参数）"""
        masks = [_create_test_mask(size=(16, 16)) for _ in range(3)]
        loss_history = [0.5, 0.3, 0.1]
        optical_params = {
            'wavelength': 193.0,
            'na': 1.35,
            'sigma': 0.75
        }
        extra_data = {
            'total_iterations': 100,
            'converged': True,
            'message': '优化完成'
        }
        hdf5_path = tmp_path / "complete.h5"

        save_hdf5_results(
            str(hdf5_path),
            mask_sequence=masks,
            loss_history=loss_history,
            optical_params=optical_params,
            extra_data=extra_data
        )

        result = load_hdf5_results(str(hdf5_path))
        assert result['mask_sequence'].shape == (3, 16, 16)
        assert len(result['loss_history']) == 3
        assert result['optical_params']['wavelength'] == 193.0
        assert result['extra_data']['total_iterations'] == 100
        assert result['extra_data']['converged'] == True

    def test_hdf5_compression(self, tmp_path):
        """测试压缩选项（不抛出异常即可）"""
        masks = [_create_test_mask(size=(16, 16)) for _ in range(3)]
        hdf5_path = tmp_path / "compressed.h5"

        save_hdf5_results(
            str(hdf5_path),
            mask_sequence=masks,
            compression='gzip',
            compression_opts=6
        )

        assert hdf5_path.exists()
        result = load_hdf5_results(str(hdf5_path))
        assert result['mask_sequence'].shape == (3, 16, 16)

    def test_load_hdf5_nonexistent_raises(self, tmp_path):
        """测试加载不存在的 HDF5 文件抛出异常"""
        with pytest.raises(FileNotFoundError):
            load_hdf5_results(str(tmp_path / "nonexistent.h5"))

    def test_save_hdf5_creates_parent_dir(self, tmp_path):
        """测试自动创建父目录"""
        masks = [_create_test_mask(size=(8, 8))]
        hdf5_path = tmp_path / "a" / "b" / "c" / "result.h5"

        save_hdf5_results(str(hdf5_path), mask_sequence=masks)
        assert hdf5_path.exists()


class TestSaveOptimizationResult:
    """优化结果整体保存测试"""

    def test_save_all_formats(self, tmp_path):
        """测试保存所有格式（PNG/NPY/GDS/HDF5）"""
        from dataclasses import dataclass, field
        from typing import List, Optional

        @dataclass
        class MockMetrics:
            mse: float = 0.001
            mae: float = 0.01
            ssim: float = 0.99

        @dataclass
        class MockResult:
            optimized_mask: np.ndarray = None
            initial_mask: np.ndarray = None
            target_image: np.ndarray = None
            final_wafer_image: np.ndarray = None
            initial_wafer_image: np.ndarray = None
            final_metrics: MockMetrics = field(default_factory=MockMetrics)
            initial_metrics: MockMetrics = field(default_factory=MockMetrics)
            loss_history: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.1])
            total_iterations: int = 100
            total_time: float = 10.5
            converged: bool = True
            message: str = "优化完成"
            mask_history: Optional[List[np.ndarray]] = None

        mask = _create_test_mask(size=(32, 32))
        result = MockResult(
            optimized_mask=mask,
            initial_mask=mask * 0.8,
            target_image=mask,
            final_wafer_image=mask * 0.95,
            initial_wafer_image=mask * 0.7,
            mask_history=[mask, mask * 0.9, mask]
        )

        optical_params = {
            'wavelength': 193.0,
            'na': 1.35,
            'sigma': 0.75
        }

        saved = save_optimization_result(
            result,
            str(tmp_path),
            prefix='test',
            formats=['png', 'npy', 'gds', 'hdf5'],
            optical_params=optical_params
        )

        assert 'png' in saved
        assert 'npy' in saved
        assert 'gds' in saved
        assert 'hdf5' in saved

        for fmt, path in saved.items():
            assert Path(path).exists()

    def test_save_only_selected_formats(self, tmp_path):
        """测试只保存选定格式"""
        from dataclasses import dataclass

        @dataclass
        class MockResult:
            optimized_mask: np.ndarray = None
            loss_history: list = None

        mask = _create_test_mask(size=(16, 16))
        result = MockResult(optimized_mask=mask, loss_history=[0.5, 0.3])

        saved = save_optimization_result(
            result,
            str(tmp_path),
            prefix='partial',
            formats=['png', 'npy']
        )

        assert 'png' in saved
        assert 'npy' in saved
        assert 'gds' not in saved
        assert 'hdf5' not in saved


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
