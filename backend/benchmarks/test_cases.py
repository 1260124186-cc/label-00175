# -*- coding: utf-8 -*-
"""
逆向光刻基准测试套件 - 标准测试用例定义

收录业界公开测试场景，涵盖从简单周期结构到复杂逻辑/存储单元的典型光刻图案。
每个测试用例包含：目标图案生成参数、光学系统参数、工艺节点参考、难度等级。

测试用例分类:
    1. 标准线/空间 (Line/Space) - 分辨率与线宽均匀性
    2. 接触孔 (Contact Hole) - 孔径圆形度与阵列均匀性
    3. 逻辑标准单元 (Logic Standard Cell) - 拐角/线端/密集-稀疏过渡
    4. SRAM 存储单元 (SRAM Bitcell) - 高密度金属层布线
    5. 孤立/密集过渡 (Dense/Isolated Transition) - 邻近效应补偿
    6. 通过焦点密集阵列 (Through-Focus Dense Array) - 工艺窗口评估
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class TestCaseCategory(Enum):
    LINE_SPACE = "line_space"
    CONTACT_HOLE = "contact_hole"
    LOGIC_CELL = "logic_cell"
    SRAM_BITCELL = "sram_bitcell"
    DENSE_ISOLATED = "dense_isolated"
    THROUGH_FOCUS = "through_focus"


@dataclass
class BenchmarkTestCase:
    """
    基准测试用例定义

    Attributes:
        name: 用例唯一标识名
        category: 测试用例分类
        difficulty: 难度等级
        description: 用例描述
        technology_node: 参考工艺节点 (nm)
        pattern_params: 图案生成参数 (传入 core.test_structures)
        optical_params: 光学系统参数
        optimizer_defaults: 默认优化器参数
        pw_scan_defaults: 工艺窗口扫描默认参数
        reference_metrics: 业界公开参考指标 (可选)
        tags: 附加标签
    """
    name: str
    category: TestCaseCategory
    difficulty: DifficultyLevel
    description: str
    technology_node: float
    pattern_params: Dict[str, Any] = field(default_factory=dict)
    optical_params: Dict[str, Any] = field(default_factory=dict)
    optimizer_defaults: Dict[str, Any] = field(default_factory=dict)
    pw_scan_defaults: Dict[str, Any] = field(default_factory=dict)
    reference_metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.pattern_params.get('pixel_size') and self.optical_params.get('pixel_size'):
            self.pattern_params['pixel_size'] = self.optical_params['pixel_size']


def _make_line_space_cases() -> List[BenchmarkTestCase]:
    return [
        BenchmarkTestCase(
            name="ls_45nm_half_pitch",
            category=TestCaseCategory.LINE_SPACE,
            difficulty=DifficultyLevel.EASY,
            description="标准 1:1 线/空间, CD=45nm, pitch=90nm, 水平方向",
            technology_node=45,
            pattern_params=dict(
                type='line_space', grid_size=[64, 64], pixel_size=1.0,
                cd=45.0, pitch=90.0, corner_rounding=0.0,
                orientation='horizontal',
            ),
            optical_params=dict(
                wavelength=193.0, na=1.35, sigma=0.75, pixel_size=1.0,
                defocus=0.0, magnification=4.0,
                illumination_type='conventional',
                source_params=dict(sigma_inner=0.0, sigma_outer=0.75),
                tcc_mode='socs', socs_num_terms=5,
                zernike_coefficients={},
            ),
            optimizer_defaults=dict(
                type='gradient_descent', max_iter=100, learning_rate=0.01,
                tol=1e-6, early_stop_patience=10, random_seed=42,
                loss_weights=dict(mse=1.0),
            ),
            pw_scan_defaults=dict(
                focus_range=(-150, 150, 11), dose_range=(0.85, 1.15, 11),
                cd_tolerance=0.1,
            ),
            reference_metrics=dict(epe_mean_nm=2.0, pw_ratio=0.3),
            tags=['regression', 'resolution', 'lwr'],
        ),
        BenchmarkTestCase(
            name="ls_38nm_half_pitch",
            category=TestCaseCategory.LINE_SPACE,
            difficulty=DifficultyLevel.MEDIUM,
            description="亚半间距 线/空间, CD=38nm, pitch=76nm",
            technology_node=38,
            pattern_params=dict(
                type='line_space', grid_size=[64, 64], pixel_size=1.0,
                cd=38.0, pitch=76.0, corner_rounding=0.0,
                orientation='horizontal',
            ),
            optical_params=dict(
                wavelength=193.0, na=1.35, sigma=0.75, pixel_size=1.0,
                defocus=0.0, magnification=4.0,
                illumination_type='conventional',
                source_params=dict(sigma_inner=0.0, sigma_outer=0.75),
                tcc_mode='socs', socs_num_terms=5,
                zernike_coefficients={},
            ),
            optimizer_defaults=dict(
                type='adam', max_iter=150, learning_rate=0.005,
                tol=1e-6, early_stop_patience=15, random_seed=42,
                loss_weights=dict(mse=1.0),
            ),
            pw_scan_defaults=dict(
                focus_range=(-120, 120, 11), dose_range=(0.88, 1.12, 11),
                cd_tolerance=0.1,
            ),
            reference_metrics=dict(epe_mean_nm=3.0, pw_ratio=0.2),
            tags=['regression', 'sub-resolution'],
        ),
        BenchmarkTestCase(
            name="ls_28nm_vertical",
            category=TestCaseCategory.LINE_SPACE,
            difficulty=DifficultyLevel.HARD,
            description="垂直线/空间, CD=28nm, pitch=56nm, EUV",
            technology_node=28,
            pattern_params=dict(
                type='line_space', grid_size=[128, 128], pixel_size=0.5,
                cd=28.0, pitch=56.0, corner_rounding=0.0,
                orientation='vertical',
            ),
            optical_params=dict(
                wavelength=13.5, na=0.33, sigma=0.8, pixel_size=0.5,
                defocus=0.0, magnification=4.0,
                illumination_type='conventional',
                source_params=dict(sigma_inner=0.0, sigma_outer=0.8),
                tcc_mode='socs', socs_num_terms=7,
                zernike_coefficients={},
            ),
            optimizer_defaults=dict(
                type='adam', max_iter=200, learning_rate=0.003,
                tol=1e-7, early_stop_patience=20, random_seed=42,
                loss_weights=dict(mse=0.8, perimeter=0.2),
            ),
            pw_scan_defaults=dict(
                focus_range=(-80, 80, 9), dose_range=(0.9, 1.1, 9),
                cd_tolerance=0.1,
            ),
            reference_metrics=dict(epe_mean_nm=2.5, pw_ratio=0.15),
            tags=['euv', 'vertical', 'high-NA'],
        ),
    ]


def _make_contact_hole_cases() -> List[BenchmarkTestCase]:
    return [
        BenchmarkTestCase(
            name="ch_50nm_square_grid",
            category=TestCaseCategory.CONTACT_HOLE,
            difficulty=DifficultyLevel.MEDIUM,
            description="正方形阵列接触孔, CD=50nm, pitch=100nm, 圆形孔",
            technology_node=45,
            pattern_params=dict(
                type='contact_hole', grid_size=[64, 64], pixel_size=1.0,
                cd=50.0, pitch=100.0, corner_rounding=0.0,
                hole_shape='circle',
            ),
            optical_params=dict(
                wavelength=193.0, na=1.35, sigma=0.75, pixel_size=1.0,
                defocus=0.0, magnification=4.0,
                illumination_type='conventional',
                source_params=dict(sigma_inner=0.0, sigma_outer=0.75),
                tcc_mode='socs', socs_num_terms=5,
                zernike_coefficients={},
            ),
            optimizer_defaults=dict(
                type='adam', max_iter=150, learning_rate=0.005,
                tol=1e-6, early_stop_patience=15, random_seed=42,
                loss_weights=dict(mse=1.0),
            ),
            pw_scan_defaults=dict(
                focus_range=(-120, 120, 9), dose_range=(0.88, 1.12, 9),
                cd_tolerance=0.1,
            ),
            reference_metrics=dict(epe_mean_nm=3.5, pw_ratio=0.2),
            tags=['contact', 'circularity'],
        ),
        BenchmarkTestCase(
            name="ch_40nm_hex",
            category=TestCaseCategory.CONTACT_HOLE,
            difficulty=DifficultyLevel.HARD,
            description="六角排列接触孔, CD=40nm, pitch=80nm",
            technology_node=28,
            pattern_params=dict(
                type='contact_hole', grid_size=[128, 128], pixel_size=0.5,
                cd=40.0, pitch=80.0, corner_rounding=0.0,
                hole_shape='circle',
            ),
            optical_params=dict(
                wavelength=13.5, na=0.33, sigma=0.8, pixel_size=0.5,
                defocus=0.0, magnification=4.0,
                illumination_type='conventional',
                source_params=dict(sigma_inner=0.0, sigma_outer=0.8),
                tcc_mode='socs', socs_num_terms=7,
                zernike_coefficients={},
            ),
            optimizer_defaults=dict(
                type='adam', max_iter=200, learning_rate=0.003,
                tol=1e-7, early_stop_patience=20, random_seed=42,
                loss_weights=dict(mse=0.8, perimeter=0.2),
            ),
            pw_scan_defaults=dict(
                focus_range=(-60, 60, 7), dose_range=(0.92, 1.08, 7),
                cd_tolerance=0.1,
            ),
            reference_metrics=dict(epe_mean_nm=3.0, pw_ratio=0.12),
            tags=['euv', 'hexagonal', 'high-density'],
        ),
    ]


def _make_logic_cell_cases() -> List[BenchmarkTestCase]:
    return [
        BenchmarkTestCase(
            name="logic_l_corner_inner",
            category=TestCaseCategory.LOGIC_CELL,
            difficulty=DifficultyLevel.MEDIUM,
            description="L形内拐角结构, 拐角圆滑度与EPE评估",
            technology_node=45,
            pattern_params=dict(
                type='l_shaped_corner', grid_size=[64, 64], pixel_size=1.0,
                cd=45.0, pitch=90.0, corner_rounding=0.0,
                arm_length=200.0, corner_type='inner',
            ),
            optical_params=dict(
                wavelength=193.0, na=1.35, sigma=0.75, pixel_size=1.0,
                defocus=0.0, magnification=4.0,
                illumination_type='conventional',
                source_params=dict(sigma_inner=0.0, sigma_outer=0.75),
                tcc_mode='socs', socs_num_terms=5,
                zernike_coefficients={},
            ),
            optimizer_defaults=dict(
                type='gradient_descent', max_iter=150, learning_rate=0.01,
                tol=1e-6, early_stop_patience=15, random_seed=42,
                loss_weights=dict(mse=0.7, epe=0.3),
            ),
            pw_scan_defaults=dict(
                focus_range=(-100, 100, 9), dose_range=(0.88, 1.12, 9),
                cd_tolerance=0.1,
            ),
            reference_metrics=dict(epe_mean_nm=4.0, pw_ratio=0.18),
            tags=['corner', 'inner', 'opc-critical'],
        ),
        BenchmarkTestCase(
            name="logic_t_junction",
            category=TestCaseCategory.LOGIC_CELL,
            difficulty=DifficultyLevel.MEDIUM,
            description="T形结结构, 线端缩短与连接区域评估",
            technology_node=45,
            pattern_params=dict(
                type='t_junction', grid_size=[64, 64], pixel_size=1.0,
                cd=45.0, pitch=90.0, corner_rounding=0.0,
                stem_length=200.0, branch_length=100.0,
            ),
            optical_params=dict(
                wavelength=193.0, na=1.35, sigma=0.75, pixel_size=1.0,
                defocus=0.0, magnification=4.0,
                illumination_type='conventional',
                source_params=dict(sigma_inner=0.0, sigma_outer=0.75),
                tcc_mode='socs', socs_num_terms=5,
                zernike_coefficients={},
            ),
            optimizer_defaults=dict(
                type='adam', max_iter=150, learning_rate=0.008,
                tol=1e-6, early_stop_patience=15, random_seed=42,
                loss_weights=dict(mse=0.7, epe=0.3),
            ),
            pw_scan_defaults=dict(
                focus_range=(-100, 100, 9), dose_range=(0.88, 1.12, 9),
                cd_tolerance=0.1,
            ),
            reference_metrics=dict(epe_mean_nm=4.5, pw_ratio=0.16),
            tags=['line-end', 'bridge', 'opc-critical'],
        ),
    ]


def _make_sram_cases() -> List[BenchmarkTestCase]:
    return [
        BenchmarkTestCase(
            name="sram_6t_bitcell",
            category=TestCaseCategory.SRAM_BITCELL,
            difficulty=DifficultyLevel.HARD,
            description="6T SRAM位单元, 高密度金属层布线",
            technology_node=28,
            pattern_params=dict(
                type='sram_bitcell', grid_size=[128, 128], pixel_size=0.5,
                cd=25.0, pitch=50.0, corner_rounding=0.0,
                bitcell_type='6T',
            ),
            optical_params=dict(
                wavelength=13.5, na=0.33, sigma=0.8, pixel_size=0.5,
                defocus=0.0, magnification=4.0,
                illumination_type='quasar',
                source_params=dict(sigma_inner=0.4, sigma_outer=0.8),
                tcc_mode='socs', socs_num_terms=7,
                zernike_coefficients={},
            ),
            optimizer_defaults=dict(
                type='adam', max_iter=300, learning_rate=0.002,
                tol=1e-7, early_stop_patience=25, random_seed=42,
                loss_weights=dict(mse=0.6, epe=0.3, perimeter=0.1),
            ),
            pw_scan_defaults=dict(
                focus_range=(-60, 60, 7), dose_range=(0.93, 1.07, 7),
                cd_tolerance=0.08,
            ),
            reference_metrics=dict(epe_mean_nm=3.0, pw_ratio=0.1),
            tags=['euv', 'sram', 'high-density', 'quasar'],
        ),
    ]


def _make_dense_isolated_cases() -> List[BenchmarkTestCase]:
    return [
        BenchmarkTestCase(
            name="di_dense_to_isolated",
            category=TestCaseCategory.DENSE_ISOLATED,
            difficulty=DifficultyLevel.HARD,
            description="密集到孤立线宽过渡, 邻近效应补偿测试",
            technology_node=45,
            pattern_params=dict(
                type='line_space', grid_size=[128, 64], pixel_size=1.0,
                cd=45.0, pitch=90.0, corner_rounding=0.0,
                orientation='horizontal', num_lines=3,
            ),
            optical_params=dict(
                wavelength=193.0, na=1.35, sigma=0.75, pixel_size=1.0,
                defocus=0.0, magnification=4.0,
                illumination_type='annular',
                source_params=dict(sigma_inner=0.5, sigma_outer=0.8),
                tcc_mode='socs', socs_num_terms=5,
                zernike_coefficients={},
            ),
            optimizer_defaults=dict(
                type='adam', max_iter=200, learning_rate=0.005,
                tol=1e-7, early_stop_patience=20, random_seed=42,
                loss_weights=dict(mse=0.6, epe=0.3, tone=0.1),
            ),
            pw_scan_defaults=dict(
                focus_range=(-100, 100, 9), dose_range=(0.88, 1.12, 9),
                cd_tolerance=0.1,
            ),
            reference_metrics=dict(epe_mean_nm=5.0, pw_ratio=0.15),
            tags=['proximity', 'annular', 'dense-isolated'],
        ),
    ]


def _make_through_focus_cases() -> List[BenchmarkTestCase]:
    return [
        BenchmarkTestCase(
            name="tf_ls_45nm_pw",
            category=TestCaseCategory.THROUGH_FOCUS,
            difficulty=DifficultyLevel.MEDIUM,
            description="通过焦点工艺窗口评估, 1:1 线/空间",
            technology_node=45,
            pattern_params=dict(
                type='line_space', grid_size=[64, 64], pixel_size=1.0,
                cd=45.0, pitch=90.0, corner_rounding=0.0,
                orientation='horizontal',
            ),
            optical_params=dict(
                wavelength=193.0, na=1.35, sigma=0.75, pixel_size=1.0,
                defocus=0.0, magnification=4.0,
                illumination_type='conventional',
                source_params=dict(sigma_inner=0.0, sigma_outer=0.75),
                tcc_mode='socs', socs_num_terms=5,
                zernike_coefficients={},
            ),
            optimizer_defaults=dict(
                type='gradient_descent', max_iter=100, learning_rate=0.01,
                tol=1e-6, early_stop_patience=10, random_seed=42,
                loss_weights=dict(mse=1.0),
            ),
            pw_scan_defaults=dict(
                focus_range=(-200, 200, 15), dose_range=(0.8, 1.2, 15),
                cd_tolerance=0.1,
            ),
            reference_metrics=dict(pw_ratio=0.35, dof_nm=150.0, el_pct=12.0),
            tags=['process-window', 'dof', 'el'],
        ),
    ]


def get_all_test_cases() -> List[BenchmarkTestCase]:
    return (
        _make_line_space_cases()
        + _make_contact_hole_cases()
        + _make_logic_cell_cases()
        + _make_sram_cases()
        + _make_dense_isolated_cases()
        + _make_through_focus_cases()
    )


def get_test_cases_by_category(category: TestCaseCategory) -> List[BenchmarkTestCase]:
    return [tc for tc in get_all_test_cases() if tc.category == category]


def get_test_cases_by_difficulty(difficulty: DifficultyLevel) -> List[BenchmarkTestCase]:
    return [tc for tc in get_all_test_cases() if tc.difficulty == difficulty]


def get_test_case_by_name(name: str) -> Optional[BenchmarkTestCase]:
    for tc in get_all_test_cases():
        if tc.name == name:
            return tc
    return None
