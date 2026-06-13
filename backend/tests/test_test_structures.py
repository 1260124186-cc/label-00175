# -*- coding: utf-8 -*-
"""
测试结构生成模块单元测试
"""

import pytest
import numpy as np
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
    LineSpaceGenerator,
    ContactHoleGenerator,
    LShapedCornerGenerator,
    TJunctionGenerator,
    SRAMBitcellGenerator,
    generate_test_structure,
)


class TestEnums:
    """枚举类型测试"""

    def test_structure_type_values(self):
        assert TestStructureType.LINE_SPACE.value == "line_space"
        assert TestStructureType.CONTACT_HOLE.value == "contact_hole"
        assert TestStructureType.L_SHAPED_CORNER.value == "l_shaped_corner"
        assert TestStructureType.T_JUNCTION.value == "t_junction"
        assert TestStructureType.SRAM_BITCELL.value == "sram_bitcell"

    def test_structure_type_from_string(self):
        assert TestStructureType("line_space") == TestStructureType.LINE_SPACE
        assert TestStructureType("contact_hole") == TestStructureType.CONTACT_HOLE

    def test_line_orientation_values(self):
        assert LineOrientation.HORIZONTAL.value == "horizontal"
        assert LineOrientation.VERTICAL.value == "vertical"

    def test_hole_pattern_values(self):
        assert HolePattern.SQUARE_GRID.value == "square_grid"
        assert HolePattern.HEXAGONAL.value == "hexagonal"


class TestBaseParams:
    """基础参数类测试"""

    def test_default_params(self):
        params = TestStructureParams()
        assert params.grid_size == (256, 256)
        assert params.pixel_size == 1.0
        assert params.cd == 45.0
        assert params.pitch == 90.0
        assert params.corner_rounding == 0.0

    def test_pixel_conversion(self):
        params = TestStructureParams(cd=45, pitch=90, pixel_size=1.0)
        assert params.cd_pixels == 45.0
        assert params.pitch_pixels == 90.0
        assert params.corner_rounding_pixels == 0.0

    def test_custom_pixel_size(self):
        params = TestStructureParams(cd=90, pitch=180, pixel_size=2.0)
        assert params.cd_pixels == 45.0
        assert params.pitch_pixels == 90.0

    def test_auto_name(self):
        params = LineSpaceParams(cd=45, pitch=90)
        assert "line_space" in params.name
        assert "cd45" in params.name
        assert "pitch90" in params.name

    def test_custom_name(self):
        params = TestStructureParams(name="custom_test")
        assert params.name == "custom_test"

    def test_to_dict(self):
        params = TestStructureParams(cd=50, pitch=100)
        d = params.to_dict()
        assert d['cd'] == 50.0
        assert d['pitch'] == 100.0
        assert d['structure_type'] == 'line_space'

    def test_invalid_grid_size(self):
        with pytest.raises(ValueError):
            TestStructureParams(grid_size=(0, 100))

    def test_invalid_cd(self):
        with pytest.raises(ValueError):
            TestStructureParams(cd=-10)

    def test_invalid_pitch(self):
        with pytest.raises(ValueError):
            TestStructureParams(cd=50, pitch=40)

    def test_invalid_corner_rounding(self):
        with pytest.raises(ValueError):
            TestStructureParams(corner_rounding=-5)


class TestLineSpaceParams:
    """Line/Space 参数测试"""

    def test_default_orientation(self):
        params = LineSpaceParams()
        assert params.orientation == LineOrientation.HORIZONTAL

    def test_vertical_orientation(self):
        params = LineSpaceParams(orientation=LineOrientation.VERTICAL)
        assert params.orientation == LineOrientation.VERTICAL

    def test_num_lines(self):
        params = LineSpaceParams(num_lines=5)
        assert params.num_lines == 5

    def test_invalid_duty_cycle(self):
        with pytest.raises(ValueError):
            LineSpaceParams(duty_cycle=-1)

    def test_invalid_num_lines(self):
        with pytest.raises(ValueError):
            LineSpaceParams(num_lines=-3)


class TestContactHoleParams:
    """Contact Hole 参数测试"""

    def test_default_pattern(self):
        params = ContactHoleParams()
        assert params.pattern == HolePattern.SQUARE_GRID

    def test_hexagonal_pattern(self):
        params = ContactHoleParams(pattern=HolePattern.HEXAGONAL)
        assert params.pattern == HolePattern.HEXAGONAL

    def test_circle_shape(self):
        params = ContactHoleParams(hole_shape='circle')
        assert params.hole_shape == 'circle'

    def test_square_shape(self):
        params = ContactHoleParams(hole_shape='square')
        assert params.hole_shape == 'square'

    def test_invalid_hole_shape(self):
        with pytest.raises(ValueError):
            ContactHoleParams(hole_shape='triangle')

    def test_aspect_ratio(self):
        params = ContactHoleParams(aspect_ratio=2.0)
        assert params.aspect_ratio == 2.0

    def test_invalid_aspect_ratio(self):
        with pytest.raises(ValueError):
            ContactHoleParams(aspect_ratio=-1)


class TestLShapedCornerParams:
    """L-shaped Corner 参数测试"""

    def test_default_corner_type(self):
        params = LShapedCornerParams()
        assert params.corner_type == 'inner'

    def test_outer_corner(self):
        params = LShapedCornerParams(corner_type='outer')
        assert params.corner_type == 'outer'

    def test_arm_length(self):
        params = LShapedCornerParams(arm_length=300)
        assert params.arm_length == 300
        assert params.arm_length_pixels == 300.0

    def test_invalid_corner_type(self):
        with pytest.raises(ValueError):
            LShapedCornerParams(corner_type='invalid')

    def test_invalid_arm_length(self):
        with pytest.raises(ValueError):
            LShapedCornerParams(arm_length=-100)


class TestTJunctionParams:
    """T-junction 参数测试"""

    def test_default_lengths(self):
        params = TJunctionParams()
        assert params.stem_length == 200.0
        assert params.branch_length == 100.0

    def test_custom_lengths(self):
        params = TJunctionParams(stem_length=300, branch_length=150)
        assert params.stem_length_pixels == 300.0
        assert params.branch_length_pixels == 150.0

    def test_invalid_stem_length(self):
        with pytest.raises(ValueError):
            TJunctionParams(stem_length=-50)

    def test_invalid_branch_length(self):
        with pytest.raises(ValueError):
            TJunctionParams(branch_length=-50)


class TestSRAMBitcellParams:
    """SRAM Bitcell 参数测试"""

    def test_default_type(self):
        params = SRAMBitcellParams()
        assert params.bitcell_type == '6T'

    def test_thin_film_type(self):
        params = SRAMBitcellParams(bitcell_type='thin-film')
        assert params.bitcell_type == 'thin-film'

    def test_metal_layer(self):
        params = SRAMBitcellParams(metal_layer=3)
        assert params.metal_layer == 3

    def test_invalid_bitcell_type(self):
        with pytest.raises(ValueError):
            SRAMBitcellParams(bitcell_type='invalid')

    def test_invalid_metal_layer(self):
        with pytest.raises(ValueError):
            SRAMBitcellParams(metal_layer=0)


class TestLineSpaceGenerator:
    """Line/Space 生成器测试"""

    def test_output_shape(self):
        params = LineSpaceParams(grid_size=(64, 128))
        mask = LineSpaceGenerator.generate(params)
        assert mask.shape == (64, 128)

    def test_horizontal_lines(self):
        params = LineSpaceParams(
            grid_size=(64, 64), cd=4, pitch=8, orientation=LineOrientation.HORIZONTAL
        )
        mask = LineSpaceGenerator.generate(params)
        for i in range(0, 64, 8):
            assert np.all(mask[i:i+4, :] == 1.0)
            assert np.all(mask[i+4:i+8, :] == 0.0)

    def test_vertical_lines(self):
        params = LineSpaceParams(
            grid_size=(64, 64), cd=4, pitch=8, orientation=LineOrientation.VERTICAL
        )
        mask = LineSpaceGenerator.generate(params)
        for j in range(0, 64, 8):
            assert np.all(mask[:, j:j+4] == 1.0)
            assert np.all(mask[:, j+4:j+8] == 0.0)

    def test_limited_lines(self):
        params = LineSpaceParams(
            grid_size=(64, 64), cd=4, pitch=8, num_lines=3
        )
        mask = LineSpaceGenerator.generate(params)
        line_count = np.sum(np.mean(mask, axis=1) > 0.5)
        assert line_count == 12

    def test_binary_values(self):
        params = LineSpaceParams(grid_size=(32, 32))
        mask = LineSpaceGenerator.generate(params)
        assert np.all(np.logical_or(mask == 0.0, mask == 1.0))

    def test_duty_cycle_effect(self):
        params = LineSpaceParams(
            grid_size=(32, 32), cd=2, pitch=6
        )
        mask = LineSpaceGenerator.generate(params)
        fill_ratio = np.mean(mask)
        expected_ratio = params.cd / params.pitch
        assert abs(fill_ratio - expected_ratio) < 0.1


class TestContactHoleGenerator:
    """Contact Hole 生成器测试"""

    def test_output_shape(self):
        params = ContactHoleParams(grid_size=(64, 128))
        mask = ContactHoleGenerator.generate(params)
        assert mask.shape == (64, 128)

    def test_square_grid(self):
        params = ContactHoleParams(
            grid_size=(64, 64), cd=8, pitch=16, pattern=HolePattern.SQUARE_GRID
        )
        mask = ContactHoleGenerator.generate(params)
        assert mask[0, 0] == 1.0
        hole_count = np.sum(mask == 0.0)
        expected_holes = (64 // 16) * (64 // 16)
        assert hole_count > 0

    def test_hexagonal_pattern(self):
        params = ContactHoleParams(
            grid_size=(64, 64), cd=8, pitch=16, pattern=HolePattern.HEXAGONAL
        )
        mask = ContactHoleGenerator.generate(params)
        hole_count = np.sum(mask == 0.0)
        assert hole_count > 0

    def test_square_holes(self):
        params = ContactHoleParams(
            grid_size=(32, 32), cd=8, pitch=16, hole_shape='square'
        )
        mask = ContactHoleGenerator.generate(params)
        hole_count = np.sum(mask == 0.0)
        assert hole_count > 0

    def test_elliptical_holes(self):
        params = ContactHoleParams(
            grid_size=(32, 32), cd=8, pitch=16, aspect_ratio=2.0, rotation=45
        )
        mask = ContactHoleGenerator.generate(params)
        assert mask.shape == (32, 32)

    def test_binary_values(self):
        params = ContactHoleParams(grid_size=(32, 32))
        mask = ContactHoleGenerator.generate(params)
        assert np.all(np.logical_or(mask == 0.0, mask == 1.0))


class TestLShapedCornerGenerator:
    """L-shaped Corner 生成器测试"""

    def test_output_shape(self):
        params = LShapedCornerParams(grid_size=(64, 128))
        mask = LShapedCornerGenerator.generate(params)
        assert mask.shape == (64, 128)

    def test_inner_corner(self):
        params = LShapedCornerParams(
            grid_size=(64, 64), cd=8, arm_length=32, corner_type='inner'
        )
        mask = LShapedCornerGenerator.generate(params)
        assert np.sum(mask > 0.5) > 0
        cy, cx = 32, 32
        assert mask[cy, cx] == 1.0

    def test_outer_corner(self):
        params = LShapedCornerParams(
            grid_size=(64, 64), cd=8, arm_length=32, corner_type='outer'
        )
        mask = LShapedCornerGenerator.generate(params)
        assert np.sum(mask > 0.5) > 0

    def test_binary_values(self):
        params = LShapedCornerParams(grid_size=(32, 32))
        mask = LShapedCornerGenerator.generate(params)
        assert np.all(np.logical_or(mask == 0.0, mask == 1.0))


class TestTJunctionGenerator:
    """T-junction 生成器测试"""

    def test_output_shape(self):
        params = TJunctionParams(grid_size=(64, 128))
        mask =