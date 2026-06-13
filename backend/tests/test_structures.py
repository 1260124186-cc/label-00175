# -*- coding: utf-8 -*-
"""
半导体标准测试结构生成模块单元测试
"""

import pytest
import numpy as np
import warnings
from core.test_structures import (
    TestStructureType,
    LineOrientation,
    HolePattern,
    TestStructureParams,
    LineSpaceParams,
    ContactHoleParams,
    LShapedCornerParams,
    TJunctionParams,
    SRAMBitcellParams,
    TestStructureGenerator,
    LineSpaceGenerator,
    ContactHoleGenerator,
    LShapedCornerGenerator,
    TJunctionGenerator,
    SRAMBitcellGenerator,
    generate_test_structure,
)


class TestEnumerations:
    """枚举类型测试"""

    def test_test_structure_type_values(self):
        """测试测试结构类型枚举值"""
        assert TestStructureType.LINE_SPACE.value == "line_space"
        assert TestStructureType.CONTACT_HOLE.value == "contact_hole"
        assert TestStructureType.L_SHAPED_CORNER.value == "l_shaped_corner"
        assert TestStructureType.T_JUNCTION.value == "t_junction"
        assert TestStructureType.SRAM_BITCELL.value == "sram_bitcell"

    def test_test_structure_type_from_string(self):
        """测试从字符串创建测试结构类型"""
        assert TestStructureType("line_space") == TestStructureType.LINE_SPACE
        assert TestStructureType("contact_hole") == TestStructureType.CONTACT_HOLE

    def test_line_orientation_values(self):
        """测试线方向枚举值"""
        assert LineOrientation.HORIZONTAL.value == "horizontal"
        assert LineOrientation.VERTICAL.value == "vertical"

    def test_hole_pattern_values(self):
        """测试孔排列方式枚举值"""
        assert HolePattern.SQUARE_GRID.value == "square_grid"
        assert HolePattern.HEXAGONAL.value == "hexagonal"


class TestTestStructureParams:
    """测试结构基础参数测试"""

    def test_default_parameters(self):
        """测试默认参数"""
        params = TestStructureParams()
        assert params.grid_size == (256, 256)
        assert params.pixel_size == 1.0
        assert params.cd == 45.0
        assert params.pitch == 90.0
        assert params.corner_rounding == 0.0

    def test_custom_parameters(self):
        """测试自定义参数"""
        params = TestStructureParams(
            grid_size=(128, 256),
            pixel_size=2.0,
            cd=32,
            pitch=64,
            corner_rounding=5
        )
        assert params.grid_size == (128, 256)
        assert params.pixel_size == 2.0
        assert params.cd == 32.0
        assert params.pitch == 64.0
        assert params.corner_rounding == 5.0

    def test_pixel_conversion_properties(self):
        """测试像素转换属性"""
        params = TestStructureParams(pixel_size=1.0, cd=45, pitch=90, corner_rounding=10)
        assert params.cd_pixels == 45.0
        assert params.pitch_pixels == 90.0
        assert params.corner_rounding_pixels == 10.0

        params2 = TestStructureParams(pixel_size=2.0, cd=45, pitch=90, corner_rounding=10)
        assert params2.cd_pixels == 22.5
        assert params2.pitch_pixels == 45.0
        assert params2.corner_rounding_pixels == 5.0

    def test_auto_name_generation(self):
        """测试自动名称生成"""
        params = TestStructureParams(
            structure_type=TestStructureType.LINE_SPACE,
            cd=45,
            pitch=90
        )
        assert "line_space" in params.name
        assert "cd45" in params.name
        assert "pitch90" in params.name

    def test_to_dict(self):
        """测试转换为字典"""
        params = TestStructureParams(grid_size=(128, 128), cd=32, pitch=64)
        d = params.to_dict()
        assert d['grid_size'] == [128, 128]
        assert d['cd'] == 32.0
        assert d['pitch'] == 64.0
        assert d['structure_type'] == 'line_space'

    def test_invalid_grid_size(self):
        """测试无效的网格尺寸"""
        with pytest.raises(ValueError):
            TestStructureParams(grid_size=(0, 256))

    def test_invalid_pixel_size(self):
        """测试无效的像素尺寸"""
        with pytest.raises(ValueError):
            TestStructureParams(pixel_size=-1.0)

    def test_invalid_cd(self):
        """测试无效的CD"""
        with pytest.raises(ValueError):
            TestStructureParams(cd=0)

    def test_pitch_less_than_cd(self):
        """测试pitch小于cd"""
        with pytest.raises(ValueError):
            TestStructureParams(cd=90, pitch=45)

    def test_negative_corner_rounding(self):
        """测试负的拐角圆滑度"""
        with pytest.raises(ValueError):
            TestStructureParams(corner_rounding=-5)


class TestLineSpaceParams:
    """Line/Space 参数测试"""

    def test_default_parameters(self):
        """测试默认参数"""
        params = LineSpaceParams()
        assert params.orientation == LineOrientation.HORIZONTAL
        assert params.duty_cycle == 1.0
        assert params.num_lines is None

    def test_custom_parameters(self):
        """测试自定义参数"""
        params = LineSpaceParams(
            orientation=LineOrientation.VERTICAL,
            duty_cycle=0.5,
            num_lines=5
        )
        assert params.orientation == LineOrientation.VERTICAL
        assert params.duty_cycle == 0.5
        assert params.num_lines == 5

    def test_invalid_duty_cycle(self):
        """测试无效的占空比"""
        with pytest.raises(ValueError):
            LineSpaceParams(duty_cycle=0)

    def test_invalid_num_lines(self):
        """测试无效的线数量"""
        with pytest.raises(ValueError):
            LineSpaceParams(num_lines=0)


class TestContactHoleParams:
    """Contact Hole 参数测试"""

    def test_default_parameters(self):
        """测试默认参数"""
        params = ContactHoleParams()
        assert params.pattern == HolePattern.SQUARE_GRID
        assert params.hole_shape == "circle"
        assert params.aspect_ratio == 1.0
        assert params.rotation == 0.0

    def test_custom_parameters(self):
        """测试自定义参数"""
        params = ContactHoleParams(
            pattern=HolePattern.HEXAGONAL,
            hole_shape="square",
            aspect_ratio=1.5,
            rotation=45.0
        )
        assert params.pattern == HolePattern.HEXAGONAL
        assert params.hole_shape == "square"
        assert params.aspect_ratio == 1.5
        assert params.rotation == 45.0

    def test_invalid_hole_shape(self):
        """测试无效的孔形状"""
        with pytest.raises(ValueError):
            ContactHoleParams(hole_shape="triangle")

    def test_invalid_aspect_ratio(self):
        """测试无效的纵横比"""
        with pytest.raises(ValueError):
            ContactHoleParams(aspect_ratio=0)


class TestLShapedCornerParams:
    """L-shaped Corner 参数测试"""

    def test_default_parameters(self):
        """测试默认参数"""
        params = LShapedCornerParams()
        assert params.arm_length == 200.0
        assert params.corner_type == "inner"

    def test_custom_parameters(self):
        """测试自定义参数"""
        params = LShapedCornerParams(
            arm_length=300,
            corner_type="outer"
        )
        assert params.arm_length == 300.0
        assert params.corner_type == "outer"
        assert params.arm_length_pixels == 300.0

    def test_invalid_arm_length(self):
        """测试无效的臂长"""
        with pytest.raises(ValueError):
            LShapedCornerParams(arm_length=0)

    def test_invalid_corner_type(self):
        """测试无效的拐角类型"""
        with pytest.raises(ValueError):
            LShapedCornerParams(corner_type="invalid")


class TestTJunctionParams:
    """T-junction 参数测试"""

    def test_default_parameters(self):
        """测试默认参数"""
        params = TJunctionParams()
        assert params.stem_length == 200.0
        assert params.branch_length == 100.0

    def test_custom_parameters(self):
        """测试自定义参数"""
        params = TJunctionParams(
            stem_length=300,
            branch_length=150
        )
        assert params.stem_length == 300.0
        assert params.branch_length == 150.0
        assert params.stem_length_pixels == 300.0
        assert params.branch_length_pixels == 150.0

    def test_invalid_stem_length(self):
        """测试无效的主干长度"""
        with pytest.raises(ValueError):
            TJunctionParams(stem_length=0)

    def test_invalid_branch_length(self):
        """测试无效的分支长度"""
        with pytest.raises(ValueError):
            TJunctionParams(branch_length=0)


class TestSRAMBitcellParams:
    """SRAM Bitcell 参数测试"""

    def test_default_parameters(self):
        """测试默认参数"""
        params = SRAMBitcellParams()
        assert params.bitcell_type == "6T"
        assert params.metal_layer == 1

    def test_custom_parameters(self):
        """测试自定义参数"""
        params = SRAMBitcellParams(
            bitcell_type="thin-film",
            metal_layer=3
        )
        assert params.bitcell_type == "thin-film"
        assert params.metal_layer == 3

    def test_invalid_bitcell_type(self):
        """测试无效的位单元类型"""
        with pytest.raises(ValueError):
            SRAMBitcellParams(bitcell_type="invalid")

    def test_invalid_metal_layer(self):
        """测试无效的金属层编号"""
        with pytest.raises(ValueError):
            SRAMBitcellParams(metal_layer=0)


class TestTestStructureGenerator:
    """测试结构生成器基类测试"""

    def test_get_coordinates_centered(self):
        """测试获取以中心为原点的坐标"""
        yy, xx = TestStructureGenerator._get_coordinates((4, 4))
        assert yy.shape == (4, 4)
        assert xx.shape == (4, 4)
        assert yy[1, 1] < 0
        assert yy[2, 2] >= 0
        assert xx[1, 1] < 0
        assert xx[2, 2] >= 0

    def test_get_coordinates_custom_center(self):
        """测试自定义中心坐标"""
        yy, xx = TestStructureGenerator._get_coordinates((4, 4), center=(0, 0))
        assert yy[0, 0] == 0.0
        assert xx[0, 0] == 0.0

    def test_get_coordinates_pixel_size(self):
        """测试像素尺寸缩放"""
        yy, xx = TestStructureGenerator._get_coordinates((4, 4), pixel_size=2.0)
        assert abs(yy[1, 1]) == 2.0

    def test_no_corner_rounding(self):
        """测试corner_rounding为0时不做处理"""
        mask = np.random.rand(10, 10)
        result = TestStructureGenerator._apply_corner_rounding(mask, 0, 1.0)
        np.testing.assert_array_equal(result, mask)

    def test_small_corner_rounding_warning(self):
        """测试过小的corner_rounding发出警告"""
        mask = np.ones((10, 10))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = TestStructureGenerator._apply_corner_rounding(mask, 1.0, 1.0)
            assert len(w) == 1
            assert "小于 1.5 像素" in str(w[0].message)

    def test_smooth_edges(self):
        """测试边缘平滑"""
        mask = np.zeros((10, 10))
        mask[3:7, 3:7] = 1.0
        smoothed = TestStructureGenerator._smooth_edges(mask, sigma=1.0)
        assert smoothed.shape == mask.shape
        assert np.all(smoothed >= 0.0) and np.all(smoothed <= 1.0)
        assert smoothed[0, 0] < 1e-4


class TestLineSpaceGenerator:
    """Line/Space 生成器测试"""

    def test_horizontal_lines(self):
        """测试水平线生成"""
        params = LineSpaceParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            orientation=LineOrientation.HORIZONTAL
        )
        mask = LineSpaceGenerator.generate(params)
        assert mask.shape == (64, 64)
        assert np.min(mask) == 0.0
        assert np.max(mask) == 1.0

        assert np.all(mask[0:8, :] == 1.0)
        assert np.all(mask[8:16, :] == 0.0)

    def test_vertical_lines(self):
        """测试垂直线生成"""
        params = LineSpaceParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            orientation=LineOrientation.VERTICAL
        )
        mask = LineSpaceGenerator.generate(params)
        assert mask.shape == (64, 64)

        assert np.all(mask[:, 0:8] == 1.0)
        assert np.all(mask[:, 8:16] == 0.0)

    def test_limited_num_lines(self):
        """测试限制线数量"""
        params = LineSpaceParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            num_lines=3
        )
        mask = LineSpaceGenerator.generate(params)

        total_lines = np.sum(np.any(mask == 1.0, axis=1)) // 8
        assert total_lines <= 3 * 2

    def test_duty_cycle(self):
        """测试占空比"""
        params = LineSpaceParams(
            grid_size=(32, 32),
            cd=8,
            pitch=16,
            duty_cycle=1.0
        )
        mask = LineSpaceGenerator.generate(params)

        line_area = np.sum(mask)
        total_area = mask.size
        expected_ratio = 8 / 16
        actual_ratio = line_area / total_area
        assert abs(actual_ratio - expected_ratio) < 0.1

    def test_with_corner_rounding(self):
        """测试带拐角圆滑的线结构"""
        params = LineSpaceParams(
            grid_size=(64, 64),
            cd=16,
            pitch=32,
            corner_rounding=3,
            num_lines=3
        )
        mask = LineSpaceGenerator.generate(params)
        assert mask.shape == (64, 64)


class TestContactHoleGenerator:
    """Contact Hole 生成器测试"""

    def test_circular_holes_square_grid(self):
        """测试正方形网格圆形孔"""
        params = ContactHoleParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            pattern=HolePattern.SQUARE_GRID,
            hole_shape="circle"
        )
        mask = ContactHoleGenerator.generate(params)
        assert mask.shape == (64, 64)
        assert np.min(mask) == 0.0
        assert np.max(mask) == 1.0

        assert mask[0, 0] == 1.0
        assert mask[40, 40] == 0.0
        assert np.sum(mask == 0.0) > 0

    def test_circular_holes_hexagonal(self):
        """测试六边形排列圆形孔"""
        params = ContactHoleParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            pattern=HolePattern.HEXAGONAL,
            hole_shape="circle"
        )
        mask = ContactHoleGenerator.generate(params)
        assert mask.shape == (64, 64)
        assert np.sum(mask == 0.0) > 0

        mask_square = ContactHoleGenerator.generate(
            ContactHoleParams(
                grid_size=(64, 64),
                cd=8,
                pitch=16,
                pattern=HolePattern.SQUARE_GRID,
                hole_shape="circle"
            )
        )

        assert not np.array_equal(mask, mask_square)

    def test_square_holes(self):
        """测试方形孔"""
        params = ContactHoleParams(
            grid_size=(32, 32),
            cd=8,
            pitch=16,
            hole_shape="square"
        )
        mask = ContactHoleGenerator.generate(params)
        assert mask.shape == (32, 32)

    def test_elliptical_holes(self):
        """测试椭圆孔"""
        params = ContactHoleParams(
            grid_size=(32, 32),
            cd=8,
            pitch=16,
            aspect_ratio=2.0,
            rotation=0
        )
        mask = ContactHoleGenerator.generate(params)
        assert mask.shape == (32, 32)

    def test_rotated_square_holes(self):
        """测试旋转方形孔"""
        params = ContactHoleParams(
            grid_size=(32, 32),
            cd=8,
            pitch=16,
            hole_shape="square",
            rotation=45.0
        )
        mask = ContactHoleGenerator.generate(params)
        assert mask.shape == (32, 32)


class TestLShapedCornerGenerator:
    """L-shaped Corner 生成器测试"""

    def test_inner_corner(self):
        """测试内拐角"""
        params = LShapedCornerParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            arm_length=32,
            corner_type="inner"
        )
        mask = LShapedCornerGenerator.generate(params)
        assert mask.shape == (64, 64)
        assert np.min(mask) == 0.0
        assert np.max(mask) == 1.0

        center = 32
        assert mask[center, center] == 1.0

    def test_outer_corner(self):
        """测试外拐角"""
        params = LShapedCornerParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            arm_length=32,
            corner_type="outer"
        )
        mask = LShapedCornerGenerator.generate(params)
        assert mask.shape == (64, 64)

        center = 32
        assert mask[center, center] == 1.0

    def test_arm_length(self):
        """测试臂长"""
        params = LShapedCornerParams(
            grid_size=(128, 128),
            cd=8,
            pitch=16,
            arm_length=64
        )
        mask = LShapedCornerGenerator.generate(params)

        assert np.sum(mask) > 0

    def test_with_corner_rounding(self):
        """测试带拐角圆滑"""
        params = LShapedCornerParams(
            grid_size=(64, 64),
            cd=16,
            pitch=32,
            arm_length=32,
            corner_rounding=5
        )
        mask = LShapedCornerGenerator.generate(params)
        assert mask.shape == (64, 64)


class TestTJunctionGenerator:
    """T-junction 生成器测试"""

    def test_t_junction_shape(self):
        """测试T形结形状"""
        params = TJunctionParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            stem_length=32,
            branch_length=24
        )
        mask = TJunctionGenerator.generate(params)
        assert mask.shape == (64, 64)
        assert np.min(mask) == 0.0
        assert np.max(mask) == 1.0

        center = 32
        assert mask[center, center] == 1.0

    def test_stem_and_branch(self):
        """测试主干和分支"""
        params = TJunctionParams(
            grid_size=(128, 128),
            cd=16,
            pitch=32,
            stem_length=64,
            branch_length=48
        )
        mask = TJunctionGenerator.generate(params)

        center = 64
        assert mask[center, center] == 1.0

        assert mask[center, center - 30] == 1.0
        assert mask[center + 8, center] == 1.0

    def test_with_corner_rounding(self):
        """测试带拐角圆滑"""
        params = TJunctionParams(
            grid_size=(64, 64),
            cd=16,
            pitch=32,
            stem_length=32,
            branch_length=24,
            corner_rounding=4
        )
        mask = TJunctionGenerator.generate(params)
        assert mask.shape == (64, 64)


class TestSRAMBitcellGenerator:
    """SRAM Bitcell 生成器测试"""

    def test_6t_bitcell(self):
        """测试6T SRAM位单元"""
        params = SRAMBitcellParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            bitcell_type="6T"
        )
        mask = SRAMBitcellGenerator.generate(params)
        assert mask.shape == (64, 64)
        assert np.min(mask) == 0.0
        assert np.max(mask) == 1.0

        assert np.sum(mask) > 0

    def test_thin_film_bitcell(self):
        """测试薄膜SRAM位单元"""
        params = SRAMBitcellParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            bitcell_type="thin-film"
        )
        mask = SRAMBitcellGenerator.generate(params)
        assert mask.shape == (64, 64)
        assert np.sum(mask) > 0

    def test_metal_layer(self):
        """测试金属层参数"""
        params = SRAMBitcellParams(
            grid_size=(64, 64),
            cd=8,
            pitch=16,
            bitcell_type="6T",
            metal_layer=2
        )
        mask = SRAMBitcellGenerator.generate(params)
        assert mask.shape == (64, 64)

    def test_with_corner_rounding(self):
        """测试带拐角圆滑"""
        params = SRAMBitcellParams(
            grid_size=(64, 64),
            cd=12,
            pitch=24,
            bitcell_type="6T",
            corner_rounding=3
        )
        mask = SRAMBitcellGenerator.generate(params)
        assert mask.shape == (64, 64)


class TestGenerateTestStructure:
    """统一入口函数测试"""

    def test_generate_with_params_object(self):
        """测试使用参数对象生成"""
        params = LineSpaceParams(
            grid_size=(32, 32),
            cd=8,
            pitch=16
        )
        mask = generate_test_structure(params)
        assert mask.shape == (32, 32)

    def test_generate_with_dict(self):
        """测试使用字典生成"""
        params_dict = {
            'structure_type': 'line_space',
            'grid_size': (32, 32),
            'cd': 8,
            'pitch': 16
        }
        mask = generate_test_structure(params_dict)
        assert mask.shape == (32, 32)

    def test_generate_contact_hole_from_dict(self):
        """测试从字典生成接触孔"""
        params_dict = {
            'structure_type': TestStructureType.CONTACT_HOLE,
            'grid_size': (32, 32),
            'cd': 8,
            'pitch': 16,
            'hole_shape': 'circle'
        }
        mask = generate_test_structure(params_dict)
        assert mask.shape == (32, 32)

    def test_generate_all_structure_types(self):
        """测试生成所有类型的结构"""
        structure_types = [
            (TestStructureType.LINE_SPACE, LineSpaceParams),
            (TestStructureType.CONTACT_HOLE, ContactHoleParams),
            (TestStructureType.L_SHAPED_CORNER, LShapedCornerParams),
            (TestStructureType.T_JUNCTION, TJunctionParams),
            (TestStructureType.SRAM_BITCELL, SRAMBitcellParams),
        ]

        for struct_type, param_class in structure_types:
            params = param_class(
                grid_size=(32, 32),
                cd=8,
                pitch=16
            )
            mask = generate_test_structure(params)
            assert mask.shape == (32, 32)
            assert np.min(mask) >= 0.0
            assert np.max(mask) <= 1.0

    def test_invalid_structure_type(self):
        """测试无效的结构类型"""
        class InvalidParams(TestStructureParams):
            pass

        params = InvalidParams()
        params.structure_type = "invalid_type"

        with pytest.raises(ValueError):
            generate_test_structure(params)


class TestIntegration:
    """集成测试"""

    def test_generate_and_analyze_line_space(self):
        """测试生成并分析线/空间结构"""
        params = LineSpaceParams(
            grid_size=(128, 128),
            cd=16,
            pitch=32,
            orientation=LineOrientation.HORIZONTAL
        )
        mask = generate_test_structure(params)

        line_profile = mask[:, 64]
        transitions = np.where(np.diff(line_profile) != 0)[0]
        assert len(transitions) > 0

    def test_generate_contact_hole_density(self):
        """测试接触孔密度"""
        params = ContactHoleParams(
            grid_size=(256, 256),
            cd=16,
            pitch=32
        )
        mask = generate_test_structure(params)

        hole_area = np.sum(mask == 0.0)
        total_area = mask.size
        density = hole_area / total_area

        expected_density = (np.pi * (8 ** 2)) / (32 ** 2)
        assert abs(density - expected_density) < 0.05

    def test_all_structures_corner_rounding(self):
        """测试所有结构的拐角圆滑效果"""
        structure_configs = [
            (LineSpaceParams, {'cd': 16, 'pitch': 32, 'num_lines': 3}),
            (LShapedCornerParams, {'cd': 16, 'pitch': 32, 'arm_length': 48}),
            (TJunctionParams, {'cd': 16, 'pitch': 32, 'stem_length': 48, 'branch_length': 32}),
            (SRAMBitcellParams, {'cd': 12, 'pitch': 24}),
        ]

        for param_class, config in structure_configs:
            params_no_rounding = param_class(
                grid_size=(64, 64),
                corner_rounding=0,
                **config
            )
            mask_no_rounding = generate_test_structure(params_no_rounding)

            params_with_rounding = param_class(
                grid_size=(64, 64),
                corner_rounding=5,
                **config
            )
            mask_with_rounding = generate_test_structure(params_with_rounding)

            assert mask_with_rounding.shape == mask_no_rounding.shape

            diff = np.abs(mask_with_rounding - mask_no_rounding)
            assert np.sum(diff) >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
