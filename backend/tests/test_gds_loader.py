# -*- coding: utf-8 -*-
"""
GDS/OASIS版图导入模块单元测试
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from utils.data_io import load_gds_layer, _rasterize_polygons

try:
    import gdstk
    HAS_GDSTK = True
except ImportError:
    HAS_GDSTK = False

try:
    import gdspy
    HAS_GDSPY = True
except ImportError:
    HAS_GDSPY = False


def _create_test_gds(filepath, layer=0, datatype=0):
    """创建包含简单矩形和圆形多边形的测试GDS文件"""
    lib = gdstk.Library(name='test')
    cell = lib.new_cell('TOP')

    rect = gdstk.rectangle((0, 0), (100, 50), layer=layer, datatype=datatype)
    cell.add(rect)

    circle = gdstk.ellipse((200, 200), 30, layer=layer, datatype=datatype)
    cell.add(circle)

    lib.write_gds(filepath)
    return filepath


def _create_test_gds_with_reference(filepath, layer=1, datatype=0):
    """创建包含单元格引用的测试GDS文件"""
    lib = gdstk.Library(name='test_ref')
    sub_cell = lib.new_cell('SUB')
    rect = gdstk.rectangle((0, 0), (50, 50), layer=layer, datatype=datatype)
    sub_cell.add(rect)

    top_cell = lib.new_cell('TOP')
    ref = gdstk.Reference(sub_cell, origin=(100, 100))
    top_cell.add(ref)

    lib.write_gds(filepath)
    return filepath


def _create_test_gds_multi_layer(filepath):
    """创建包含多个层的测试GDS文件"""
    lib = gdstk.Library(name='multi_layer')
    cell = lib.new_cell('TOP')

    rect0 = gdstk.rectangle((0, 0), (100, 100), layer=0, datatype=0)
    cell.add(rect0)

    rect1 = gdstk.rectangle((50, 50), (150, 150), layer=1, datatype=0)
    cell.add(rect1)

    rect2 = gdstk.rectangle((10, 10), (90, 90), layer=0, datatype=1)
    cell.add(rect2)

    lib.write_gds(filepath)
    return filepath


@pytest.mark.skipif(not HAS_GDSTK, reason="gdstk未安装")
class TestLoadGdsLayer:
    """GDS层加载测试"""

    def test_load_single_layer(self, tmp_path):
        """测试加载单层GDS"""
        gds_path = str(tmp_path / "test.gds")
        _create_test_gds(gds_path, layer=0, datatype=0)

        mask = load_gds_layer(gds_path, layer=0, datatype=0, pixel_size=1.0)

        assert mask.dtype == np.float64
        assert mask.ndim == 2
        assert np.any(mask > 0)
        unique_vals = np.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_load_layer_returns_binary_mask(self, tmp_path):
        """测试返回值为二值掩模"""
        gds_path = str(tmp_path / "test.gds")
        _create_test_gds(gds_path, layer=0, datatype=0)

        mask = load_gds_layer(gds_path, layer=0, datatype=0, pixel_size=1.0)

        unique = set(np.unique(mask))
        assert unique.issubset({0.0, 1.0})

    def test_pixel_size_affects_resolution(self, tmp_path):
        """测试pixel_size影响分辨率"""
        gds_path = str(tmp_path / "test.gds")
        _create_test_gds(gds_path, layer=0, datatype=0)

        mask_coarse = load_gds_layer(gds_path, layer=0, datatype=0, pixel_size=10.0)
        mask_fine = load_gds_layer(gds_path, layer=0, datatype=0, pixel_size=1.0)

        assert mask_coarse.shape[0] < mask_fine.shape[0]
        assert mask_coarse.shape[1] < mask_fine.shape[1]

    def test_target_size_override(self, tmp_path):
        """测试target_size覆盖自动计算"""
        gds_path = str(tmp_path / "test.gds")
        _create_test_gds(gds_path, layer=0, datatype=0)

        mask = load_gds_layer(gds_path, layer=0, datatype=0, target_size=(256, 256))

        assert mask.shape == (256, 256)

    def test_bounds_parameter(self, tmp_path):
        """测试bounds参数指定范围"""
        gds_path = str(tmp_path / "test.gds")
        _create_test_gds(gds_path, layer=0, datatype=0)

        mask = load_gds_layer(
            gds_path, layer=0, datatype=0,
            bounds=(-50, -50, 350, 350), pixel_size=1.0
        )

        assert mask.shape == (400, 400)

    def test_specific_layer_extraction(self, tmp_path):
        """测试指定层提取"""
        gds_path = str(tmp_path / "multi.gds")
        _create_test_gds_multi_layer(gds_path)

        mask_l0 = load_gds_layer(gds_path, layer=0, datatype=0, pixel_size=1.0)
        mask_l1 = load_gds_layer(gds_path, layer=1, datatype=0, pixel_size=1.0)

        assert np.any(mask_l0 > 0)
        assert np.any(mask_l1 > 0)

    def test_datatype_filtering(self, tmp_path):
        """测试数据类型过滤"""
        gds_path = str(tmp_path / "multi.gds")
        _create_test_gds_multi_layer(gds_path)

        mask_dt0 = load_gds_layer(gds_path, layer=0, datatype=0, pixel_size=1.0)
        mask_dt1 = load_gds_layer(gds_path, layer=0, datatype=1, pixel_size=1.0)

        assert np.any(mask_dt0 > 0)
        assert np.any(mask_dt1 > 0)

    def test_nonexistent_layer_empty_with_target_size(self, tmp_path):
        """测试不存在的层返回零掩模（指定target_size时）"""
        gds_path = str(tmp_path / "test.gds")
        _create_test_gds(gds_path, layer=0, datatype=0)

        mask = load_gds_layer(
            gds_path, layer=99, datatype=99, target_size=(64, 64)
        )

        assert mask.shape == (64, 64)
        assert np.all(mask == 0.0)

    def test_nonexistent_layer_raises_without_target_size(self, tmp_path):
        """测试不存在的层且无target_size时抛出异常"""
        gds_path = str(tmp_path / "test.gds")
        _create_test_gds(gds_path, layer=0, datatype=0)

        with pytest.raises(ValueError, match="无多边形"):
            load_gds_layer(gds_path, layer=99, datatype=99)

    def test_file_not_found(self):
        """测试文件不存在时抛出异常"""
        with pytest.raises(FileNotFoundError):
            load_gds_layer("/nonexistent/path/test.gds", layer=0)

    def test_pathlib_input(self, tmp_path):
        """测试Path对象输入"""
        gds_path = tmp_path / "test.gds"
        _create_test_gds(str(gds_path), layer=0, datatype=0)

        mask = load_gds_layer(gds_path, layer=0, datatype=0, pixel_size=1.0)

        assert mask.ndim == 2
        assert np.any(mask > 0)

    def test_rectangle_mask_coverage(self, tmp_path):
        """测试矩形多边形的掩模覆盖"""
        lib = gdstk.Library(name='rect_test')
        cell = lib.new_cell('TOP')
        rect = gdstk.rectangle((0, 0), (100, 100), layer=0, datatype=0)
        cell.add(rect)
        gds_path = str(tmp_path / "rect.gds")
        lib.write_gds(gds_path)

        mask = load_gds_layer(
            gds_path, layer=0, datatype=0,
            pixel_size=1.0, bounds=(0, 0, 100, 100)
        )

        assert mask.shape == (100, 100)
        assert np.all(mask == 1.0)

    def test_with_reference(self, tmp_path):
        """测试包含单元格引用的GDS"""
        gds_path = str(tmp_path / "ref_test.gds")
        _create_test_gds_with_reference(gds_path, layer=1, datatype=0)

        mask = load_gds_layer(
            gds_path, layer=1, datatype=0,
            pixel_size=1.0,
            bounds=(0, 0, 200, 200)
        )

        assert np.any(mask > 0)

        assert mask[100, 100] == 1.0


class TestRasterizePolygons:
    """多边形栅格化测试"""

    def test_single_rectangle(self):
        """测试单个矩形栅格化"""
        rect = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float64)
        mask = _rasterize_polygons([rect], 100, 100)

        assert mask.shape == (100, 100)
        assert mask[50, 50] == 1.0
        assert mask[0, 0] == 0.0

    def test_empty_polygon_list(self):
        """测试空多边形列表"""
        mask = _rasterize_polygons([], 64, 64)

        assert mask.shape == (64, 64)
        assert np.all(mask == 0.0)

    def test_multiple_polygons(self):
        """测试多个多边形叠加"""
        rect1 = np.array([[0, 0], [30, 0], [30, 30], [0, 30]], dtype=np.float64)
        rect2 = np.array([[20, 20], [60, 20], [60, 60], [20, 60]], dtype=np.float64)
        mask = _rasterize_polygons([rect1, rect2], 64, 64)

        assert mask[15, 15] == 1.0
        assert mask[40, 40] == 1.0
        assert mask[0, 50] == 0.0

    def test_output_dtype(self):
        """测试输出数据类型"""
        rect = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        mask = _rasterize_polygons([rect], 20, 20)

        assert mask.dtype == np.float64


@pytest.mark.skipif(not HAS_GDSTK, reason="gdstk未安装")
class TestLoadGdsLayerIntegration:
    """GDS层加载集成测试"""

    def test_mask_compatible_with_fft(self, tmp_path):
        """测试掩模可与FFT模块配合使用"""
        from core.fft import fft2d, ifft2d

        gds_path = str(tmp_path / "test.gds")
        _create_test_gds(gds_path, layer=0, datatype=0)

        mask = load_gds_layer(
            gds_path, layer=0, datatype=0,
            target_size=(64, 64), bounds=(0, 0, 230, 230)
        )

        spectrum = fft2d(mask, shift=True, normalize=True)
        recovered = ifft2d(spectrum, shifted=True, was_normalized=True)

        np.testing.assert_array_almost_equal(mask, np.real(recovered), decimal=10)

    def test_load_gds_layer_parallel_to_load_image(self, tmp_path):
        """测试load_gds_layer与load_image返回格式一致"""
        from utils.data_io import load_image

        gds_path = str(tmp_path / "test.gds")
        _create_test_gds(gds_path, layer=0, datatype=0)

        gds_mask = load_gds_layer(
            gds_path, layer=0, datatype=0,
            target_size=(64, 64), bounds=(0, 0, 230, 230)
        )

        assert gds_mask.dtype == np.float64
        assert gds_mask.ndim == 2
        unique_vals = np.unique(gds_mask)
        assert all(0.0 <= v <= 1.0 for v in unique_vals)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
