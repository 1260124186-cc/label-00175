# -*- coding: utf-8 -*-
"""
光学成像模块：光刻光学系统建模、部分相干成像模型、晶圆成像模拟

该模块实现了光刻系统中的光学成像仿真，包括：
1. 光学系统参数配置
2. 部分相干成像模型（Hopkins模型）
3. 光强分布计算
4. 晶圆成像模拟
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
from itertools import product


class IlluminationType(Enum):
    """照明模式类型"""
    CONVENTIONAL = "conventional"      # 传统圆形照明
    DIPOLE = "dipole"                  # 偶极照明
    ANNULAR = "annular"                # 环形照明
    QUASAR = "quasar"                  # 四极照明
    CUSTOM = "custom"                 # 自定义照明


@dataclass
class ProcessCondition:
    """
    单组工艺条件

    封装一次光刻仿真所需的完整工艺参数集合。

    Attributes:
        defocus: 离焦量 (nm)，正值表示过聚焦，负值表示欠聚焦
        dose: 曝光剂量 (mJ/cm²)，归一化相对剂量，1.0为标称剂量
        na: 数值孔径 (Numerical Aperture)
        sigma: 部分相干因子 (0~1)
        wavelength: 光源波长 (nm)
        name: 工艺条件名称，用于日志和结果标识
        weight: 该工艺条件在优化中的权重
    """
    defocus: float = 0.0
    dose: float = 1.0
    na: float = 1.35
    sigma: float = 0.75
    wavelength: float = 193.0
    name: str = ""
    weight: float = 1.0

    def __post_init__(self):
        if not self.name:
            self.name = f"df={self.defocus:.0f}nm_dose={self.dose:.2f}_NA={self.na:.2f}_σ={self.sigma:.2f}"

    @classmethod
    def from_optical_system(cls, optics: 'OpticalSystem',
                            dose: float = 1.0,
                            weight: float = 1.0,
                            name: str = "") -> 'ProcessCondition':
        """从OpticalSystem创建工艺条件"""
        return cls(
            defocus=optics.defocus,
            dose=dose,
            na=optics.na,
            sigma=optics.sigma,
            wavelength=optics.wavelength,
            name=name,
            weight=weight
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'defocus': self.defocus,
            'dose': self.dose,
            'na': self.na,
            'sigma': self.sigma,
            'wavelength': self.wavelength,
            'name': self.name,
            'weight': self.weight
        }

    def to_optical_system(self, base_optics: Optional['OpticalSystem'] = None) -> 'OpticalSystem':
        """
        转换为OpticalSystem实例

        Args:
            base_optics: 基础光学系统，提供未在此工艺条件中指定的参数

        Returns:
            OpticalSystem实例
        """
        if base_optics is not None:
            return OpticalSystem(
                wavelength=self.wavelength,
                na=self.na,
                sigma=self.sigma,
                pixel_size=base_optics.pixel_size,
                defocus=self.defocus,
                magnification=base_optics.magnification,
                illumination_type=base_optics.illumination_type,
                source_params=dict(base_optics.source_params),
                use_socs=base_optics.use_socs,
                socs_num_terms=base_optics.socs_num_terms,
                custom_source=base_optics.custom_source
            )
        else:
            return OpticalSystem(
                wavelength=self.wavelength,
                na=self.na,
                sigma=self.sigma,
                defocus=self.defocus
            )


@dataclass
class ProcessWindow:
    """
    工艺窗口参数扫描范围定义

    用于生成多组工艺条件的笛卡尔积。每个参数可以是：
    - 单个标量值：该参数固定不变
    - 列表/数组：取其中每个值参与笛卡尔积
    - (start, stop, num)元组：使用np.linspace生成等间隔采样

    Attributes:
        defocus_values: 离焦量扫描值 (nm)
        dose_values: 曝光剂量扫描值 (相对剂量)
        na_values: 数值孔径扫描值
        sigma_values: 部分相干因子扫描值
        wavelength_values: 波长扫描值 (nm)
        default_weight: 默认权重
    """
    defocus_values: Any = 0.0
    dose_values: Any = 1.0
    na_values: Any = 1.35
    sigma_values: Any = 0.75
    wavelength_values: Any = 193.0
    default_weight: float = 1.0

    @staticmethod
    def _normalize_scan_values(values: Any) -> List[float]:
        """规范化扫描值输入为列表"""
        if isinstance(values, (list, np.ndarray)):
            return [float(v) for v in values]
        elif isinstance(values, tuple) and len(values) == 3:
            start, stop, num = values
            return list(np.linspace(start, stop, int(num)))
        elif isinstance(values, tuple) and len(values) != 3:
            return [float(v) for v in values]
        elif np.isscalar(values):
            return [float(values)]
        else:
            return [float(values)]

    def generate_conditions(self,
                            weights: Optional[Union[float, List[float], np.ndarray]] = None,
                            center_weight_boost: Optional[float] = None) -> List[ProcessCondition]:
        """
        生成工艺条件组合（笛卡尔积）

        Args:
            weights: 每个条件的权重。可以是单个值（所有条件相同权重）、
                     或与条件数相同的列表。None则使用default_weight。
            center_weight_boost: 可选，对最接近参数空间中心的工艺条件额外乘以该权重系数。
                                用于在工艺窗口中心设置更强的约束。

        Returns:
            ProcessCondition列表
        """
        defocus_list = self._normalize_scan_values(self.defocus_values)
        dose_list = self._normalize_scan_values(self.dose_values)
        na_list = self._normalize_scan_values(self.na_values)
        sigma_list = self._normalize_scan_values(self.sigma_values)
        wavelength_list = self._normalize_scan_values(self.wavelength_values)

        all_combos = list(product(defocus_list, dose_list, na_list, sigma_list, wavelength_list))

        n_conditions = len(all_combos)

        if weights is None:
            weight_list = [self.default_weight] * n_conditions
        elif np.isscalar(weights):
            weight_list = [float(weights)] * n_conditions
        else:
            weight_list = list(weights)
            if len(weight_list) != n_conditions:
                raise ValueError(
                    f"权重数量 ({len(weight_list)}) 与工艺条件数量 ({n_conditions}) 不匹配"
                )

        if center_weight_boost is not None:
            df_center = (min(defocus_list) + max(defocus_list)) / 2
            dose_center = (min(dose_list) + max(dose_list)) / 2
            na_center = (min(na_list) + max(na_list)) / 2
            sigma_center = (min(sigma_list) + max(sigma_list)) / 2
            wl_center = (min(wavelength_list) + max(wavelength_list)) / 2

            distances = []
            for df, d, na, sg, wl in all_combos:
                dist = np.sqrt(
                    ((df - df_center) / (max(defocus_list) - min(defocus_list) + 1e-12))**2 +
                    ((d - dose_center) / (max(dose_list) - min(dose_list) + 1e-12))**2 +
                    ((na - na_center) / (max(na_list) - min(na_list) + 1e-12))**2 +
                    ((sg - sigma_center) / (max(sigma_list) - min(sigma_list) + 1e-12))**2 +
                    ((wl - wl_center) / (max(wavelength_list) - min(wavelength_list) + 1e-12))**2
                )
                distances.append(dist)

            if distances:
                min_dist_idx = np.argmin(distances)
                weight_list[min_dist_idx] *= float(center_weight_boost)

        conditions = []
        for idx, (df, d, na, sg, wl) in enumerate(all_combos):
            w = weight_list[idx]
            cond = ProcessCondition(
                defocus=df,
                dose=d,
                na=na,
                sigma=sg,
                wavelength=wl,
                weight=w,
                name=f"cond_{idx:03d}_df={df:.0f}_dose={d:.2f}_NA={na:.2f}_σ={sg:.2f}"
            )
            conditions.append(cond)

        return conditions


@dataclass
class MultiProcessSimulationResult:
    """多工艺条件联合仿真结果"""
    aerial_images: List[np.ndarray]
    wafer_images: List[np.ndarray]
    conditions: List[ProcessCondition]
    dose: Optional[float] = None
    threshold: float = 0.3

    @property
    def n_conditions(self) -> int:
        """工艺条件数量"""
        return len(self.conditions)

    def get_condition_result(self, idx: int) -> Tuple[ProcessCondition, np.ndarray, np.ndarray]:
        """获取指定索引的条件和结果"""
        return self.conditions[idx], self.aerial_images[idx], self.wafer_images[idx]

    def get_by_name(self, name: str) -> Optional[Tuple[ProcessCondition, np.ndarray, np.ndarray]]:
        """按名称查找结果"""
        for i, c in enumerate(self.conditions):
            if c.name == name:
                return c, self.aerial_images[i], self.wafer_images[i]
        return None

    def weighted_combined_image(self, use_wafer: bool = True) -> np.ndarray:
        """按权重加权合成所有条件的图像"""
        total_weight = sum(c.weight for c in self.conditions)
        if total_weight <= 0:
            raise ValueError("工艺条件权重总和必须大于0")

        images = self.wafer_images if use_wafer else self.aerial_images
        combined = np.zeros_like(images[0], dtype=np.float64)
        for c, img in zip(self.conditions, images):
            combined += (c.weight / total_weight) * img
        return combined


@dataclass
class OpticalSystem:
    """
    光学系统参数配置类

    Attributes:
        wavelength: 光源波长 (nm)
        na: 数值孔径 (Numerical Aperture)
        sigma: 部分相干因子 (0~1, 0为完全相干, 1为完全非相干)
        pixel_size: 像素尺寸 (nm)
        defocus: 离焦量 (nm)
        magnification: 放大倍率
        illumination_type: 照明模式类型
        source_params: 光源形状参数
            - conventional: {'sigma_inner': 0.0, 'sigma_outer': sigma}
            - dipole: {'sigma_inner': 0.5, 'sigma_outer': 0.8, 'angle': 0.0, 'opening_angle': 60.0}
            - annular: {'sigma_inner': 0.6, 'sigma_outer': 0.9}
            - quasar: {'sigma_inner': 0.5, 'sigma_outer': 0.8, 'angle': 45.0, 'opening_angle': 30.0}
        use_socs: 是否使用SOCS低秩分解近似
        socs_num_terms: SOCS分解项数
        custom_source: 自定义光源分布（当illumination_type=CUSTOM时使用）
    """
    wavelength: float = 193.0  # ArF光源波长
    na: float = 1.35  # 高NA浸没式光刻
    sigma: float = 0.75  # 部分相干因子
    pixel_size: float = 1.0  # 像素尺寸
    defocus: float = 0.0  # 离焦量
    magnification: float = 4.0  # 放大倍率
    illumination_type: IlluminationType = IlluminationType.CONVENTIONAL
    source_params: Dict[str, float] = field(default_factory=dict)
    use_socs: bool = True
    socs_num_terms: int = 5
    custom_source: Optional[np.ndarray] = None

    def __post_init__(self):
        """初始化后处理，设置默认光源参数"""
        if not self.source_params:
            self._set_default_source_params()

    def _set_default_source_params(self):
        """设置默认光源参数"""
        if self.illumination_type == IlluminationType.CONVENTIONAL:
            self.source_params = {
                'sigma_inner': 0.0,
                'sigma_outer': self.sigma
            }
        elif self.illumination_type == IlluminationType.DIPOLE:
            self.source_params = {
                'sigma_inner': 0.5,
                'sigma_outer': 0.8,
                'angle': 0.0,
                'opening_angle': 60.0
            }
        elif self.illumination_type == IlluminationType.ANNULAR:
            self.source_params = {
                'sigma_inner': 0.6,
                'sigma_outer': 0.9
            }
        elif self.illumination_type == IlluminationType.QUASAR:
            self.source_params = {
                'sigma_inner': 0.5,
                'sigma_outer': 0.8,
                'angle': 45.0,
                'opening_angle': 30.0
            }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OpticalSystem':
        """
        从配置字典创建OpticalSystem实例

        Args:
            config: 配置字典

        Returns:
            OpticalSystem实例
        """
        optics_config = config.get('optical_system', config)

        illumination_type_str = optics_config.get('illumination_type', 'conventional')
        try:
            illumination_type = IlluminationType(illumination_type_str)
        except ValueError:
            illumination_type = IlluminationType.CONVENTIONAL

        source_params = optics_config.get('source_params', {})

        return cls(
            wavelength=optics_config.get('wavelength', 193.0),
            na=optics_config.get('na', 1.35),
            sigma=optics_config.get('sigma', 0.75),
            pixel_size=optics_config.get('pixel_size', 1.0),
            defocus=optics_config.get('defocus', 0.0),
            magnification=optics_config.get('magnification', 4.0),
            illumination_type=illumination_type,
            source_params=source_params,
            use_socs=optics_config.get('use_socs', True),
            socs_num_terms=optics_config.get('socs_num_terms', 5)
        )

    @property
    def k1(self) -> float:
        """计算分辨率因子k1"""
        return self.wavelength / (2 * self.na)

    @property
    def cutoff_frequency(self) -> float:
        """计算截止频率"""
        return self.na / self.wavelength

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'wavelength': self.wavelength,
            'na': self.na,
            'sigma': self.sigma,
            'pixel_size': self.pixel_size,
            'defocus': self.defocus,
            'magnification': self.magnification,
            'illumination_type': self.illumination_type.value,
            'source_params': self.source_params,
            'use_socs': self.use_socs,
            'socs_num_terms': self.socs_num_terms
        }

    def parameter_sweep(
        self,
        defocus_values: Any = None,
        dose_values: Any = None,
        na_values: Any = None,
        sigma_values: Any = None,
        center_weight_boost: Optional[float] = None,
        default_weight: float = 1.0
    ) -> Tuple[List['OpticalSystem'], List[float], List[ProcessCondition]]:
        """
        参数扫描：生成多组光学系统变体

        基于当前光学系统的基准参数，对指定维度进行扫描，
        生成笛卡尔积组合的光学系统实例列表。

        Args:
            defocus_values: 离焦量扫描值。可以是：
                - None: 使用当前 defocus 固定不变
                - 标量: 固定为该值
                - 列表/数组: 取其中每个值
                - (start, stop, num) 元组: np.linspace 等间隔采样
            dose_values: 曝光剂量扫描值（归一化），格式同上，None 则固定为 1.0
            na_values: 数值孔径扫描值，格式同上，None 则使用当前 NA
            sigma_values: 部分相干因子扫描值，格式同上，None 则使用当前 sigma
            center_weight_boost: 中心条件额外权重倍率，None 则不区分
            default_weight: 所有条件的默认基础权重

        Returns:
            (optical_systems, weights, conditions) 三元组：
            - optical_systems: 扫描产生的 OpticalSystem 列表
            - weights: 每个系统对应的权重列表
            - conditions: 对应的 ProcessCondition 列表
        """
        df_vals = ProcessWindow._normalize_scan_values(
            defocus_values if defocus_values is not None else self.defocus
        )
        dose_vals = ProcessWindow._normalize_scan_values(
            dose_values if dose_values is not None else 1.0
        )
        na_vals = ProcessWindow._normalize_scan_values(
            na_values if na_values is not None else self.na
        )
        sigma_vals = ProcessWindow._normalize_scan_values(
            sigma_values if sigma_values is not None else self.sigma
        )

        pw = ProcessWindow(
            defocus_values=df_vals,
            dose_values=dose_vals,
            na_values=na_vals,
            sigma_values=sigma_vals,
            wavelength_values=self.wavelength,
            default_weight=default_weight
        )
        conditions = pw.generate_conditions(center_weight_boost=center_weight_boost)

        optical_systems = []
        weights = []
        for cond in conditions:
            opt_sys = cond.to_optical_system(base_optics=self)
            optical_systems.append(opt_sys)
            weights.append(cond.weight)

        return optical_systems, weights, conditions


def generate_source(fx: np.ndarray, fy: np.ndarray,
                    illumination_type: IlluminationType,
                    source_params: Dict[str, float],
                    cutoff: float,
                    custom_source: Optional[np.ndarray] = None) -> np.ndarray:
    """
    生成光源分布

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        illumination_type: 照明模式类型
        source_params: 光源参数
        cutoff: 截止频率
        custom_source: 自定义光源分布

    Returns:
        归一化的光源分布
    """
    ny, nx = fx.shape
    source = np.zeros((ny, nx), dtype=np.float64)

    if illumination_type == IlluminationType.CUSTOM and custom_source is not None:
        if custom_source.shape == (ny, nx):
            source = custom_source.astype(np.float64)
        else:
            raise ValueError(f"自定义光源形状 {custom_source.shape} 与频率网格形状 {(ny, nx)} 不匹配")
    else:
        sigma_inner = source_params.get('sigma_inner', 0.0)
        sigma_outer = source_params.get('sigma_outer', source_params.get('sigma', 0.75))

        rho = np.sqrt(fx**2 + fy**2) / cutoff
        theta = np.arctan2(fy, fx)

        if illumination_type == IlluminationType.CONVENTIONAL:
            mask = (rho >= sigma_inner) & (rho <= sigma_outer)
            source[mask] = 1.0

        elif illumination_type == IlluminationType.ANNULAR:
            mask = (rho >= sigma_inner) & (rho <= sigma_outer)
            source[mask] = 1.0

        elif illumination_type == IlluminationType.DIPOLE:
            angle = np.deg2rad(source_params.get('angle', 0.0))
            opening_angle = np.deg2rad(source_params.get('opening_angle', 60.0))

            radial_mask = (rho >= sigma_inner) & (rho <= sigma_outer)
            angle_diff = np.abs(np.mod(theta - angle + np.pi, 2 * np.pi) - np.pi)
            angle_mask1 = angle_diff <= opening_angle / 2
            angle_mask2 = angle_diff >= (np.pi - opening_angle / 2)

            mask = radial_mask & (angle_mask1 | angle_mask2)
            source[mask] = 1.0

        elif illumination_type == IlluminationType.QUASAR:
            angle = np.deg2rad(source_params.get('angle', 45.0))
            opening_angle = np.deg2rad(source_params.get('opening_angle', 30.0))

            radial_mask = (rho >= sigma_inner) & (rho <= sigma_outer)

            pole_angles = [angle, angle + np.pi/2, angle + np.pi, angle + 3*np.pi/2]
            angle_mask = np.zeros_like(rho, dtype=bool)

            for pole_angle in pole_angles:
                angle_diff = np.abs(np.mod(theta - pole_angle + np.pi, 2 * np.pi) - np.pi)
                angle_mask |= (angle_diff <= opening_angle / 2)

            mask = radial_mask & angle_mask
            source[mask] = 1.0

    total = np.sum(source)
    if total > 0:
        source = source / total

    return source


@jit(nopython=True, parallel=True, cache=True)
def _compute_pupil_function(fx: np.ndarray, fy: np.ndarray,
                            cutoff: float, defocus: float,
                            wavelength: float) -> np.ndarray:
    """
    计算光瞳函数（含离焦相位）

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        cutoff: 截止频率
        defocus: 离焦量
        wavelength: 波长

    Returns:
        复数光瞳函数
    """
    ny, nx = fx.shape
    pupil = np.zeros((ny, nx), dtype=np.complex128)

    for i in prange(ny):
        for j in range(nx):
            rho = np.sqrt(fx[i, j]**2 + fy[i, j]**2)
            if rho <= cutoff:
                phase = np.pi * defocus * wavelength * rho**2
                pupil[i, j] = np.exp(1j * phase)

    return pupil


def _shift_pupil(pupil: np.ndarray, shift_fx: float,
                 shift_fy: float, dfx: float, dfy: float) -> np.ndarray:
    """
    对光瞳函数进行频移

    Args:
        pupil: 光瞳函数
        shift_fx: x方向频移量
        shift_fy: y方向频移量
        dfx: x方向频率间隔
        dfy: y方向频率间隔

    Returns:
        频移后的光瞳函数
    """
    ny, nx = pupil.shape

    shift_x = int(round(shift_fx / dfx))
    shift_y = int(round(shift_fy / dfy))

    if shift_x == 0 and shift_y == 0:
        return pupil.copy()

    shifted = np.roll(pupil, shift=shift_y, axis=0)
    shifted = np.roll(shifted, shift=shift_x, axis=1)

    return shifted


def compute_tcc_full(fx: np.ndarray, fy: np.ndarray,
                     pupil: np.ndarray, source: np.ndarray,
                     cutoff: float, dfx: float, dfy: float) -> np.ndarray:
    """
    基于光源积分计算完整的 TCC 矩阵 (四维)

    TCC(f1, f2) = ∫ S(fs) * P(f1 - fs) * P*(f2 - fs) dfs

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        pupil: 光瞳函数
        source: 归一化光源分布
        cutoff: 截止频率
        dfx: x方向频率间隔
        dfy: y方向频率间隔

    Returns:
        TCC矩阵，形状为 (ny, nx, ny, nx)
    """
    ny, nx = pupil.shape
    tcc = np.zeros((ny, nx, ny, nx), dtype=np.complex128)

    source_indices = np.where(source > 1e-10)
    source_values = source[source_indices]

    for idx in range(len(source_indices[0])):
        sy, sx = source_indices[0][idx], source_indices[1][idx]
        src_val = source_values[idx]

        fs_x = fx[sy, sx]
        fs_y = fy[sy, sx]

        if np.sqrt(fs_x**2 + fs_y**2) > cutoff:
            continue

        pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, dfx, dfy)
        pupil_conj_shifted = np.conj(pupil_shifted)

        for i in range(ny):
            for j in range(nx):
                p1 = pupil_shifted[i, j]
                if abs(p1) < 1e-10:
                    continue
                for k in range(ny):
                    for l in range(nx):
                        p2 = pupil_conj_shifted[k, l]
                        if abs(p2) < 1e-10:
                            continue
                        tcc[i, j, k, l] += src_val * p1 * p2

    return tcc


def compute_tcc_kernel_2d(fx: np.ndarray, fy: np.ndarray,
                          pupil: np.ndarray, source: np.ndarray,
                          cutoff: float, dfx: float, dfy: float) -> np.ndarray:
    """
    计算二维 TCC 核（对角近似，用于传统成像）

    当 f1 = f2 时，TCC(f, f) = ∫ S(fs) * |P(f - fs)|^2 dfs

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        pupil: 光瞳函数
        source: 归一化光源分布
        cutoff: 截止频率
        dfx: x方向频率间隔
        dfy: y方向频率间隔

    Returns:
        二维 TCC 核
    """
    ny, nx = pupil.shape
    tcc_kernel = np.zeros((ny, nx), dtype=np.float64)

    source_indices = np.where(source > 1e-10)
    source_values = source[source_indices]

    for idx in range(len(source_indices[0])):
        sy, sx = source_indices[0][idx], source_indices[1][idx]
        src_val = source_values[idx]

        fs_x = fx[sy, sx]
        fs_y = fy[sy, sx]

        if np.sqrt(fs_x**2 + fs_y**2) > cutoff:
            continue

        pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, dfx, dfy)
        tcc_kernel += src_val * np.abs(pupil_shifted)**2

    total = np.sum(tcc_kernel)
    if total > 0:
        tcc_kernel = tcc_kernel / total

    return tcc_kernel


def socs_decomposition(fx: np.ndarray, fy: np.ndarray,
                       pupil: np.ndarray, source: np.ndarray,
                       cutoff: float, dfx: float, dfy: float,
                       num_terms: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    SOCS (Sum of Coherent Systems) 低秩分解

    将 TCC 矩阵分解为: TCC ≈ Σ λ_i |φ_i><φ_i|

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        pupil: 光瞳函数
        source: 归一化光源分布
        cutoff: 截止频率
        dfx: x方向频率间隔
        dfy: y方向频率间隔
        num_terms: 分解项数

    Returns:
        (eigenvalues, eigenfunctions) - 特征值和特征函数
            eigenvalues: 形状为 (num_terms,)
            eigenfunctions: 形状为 (num_terms, ny, nx)
    """
    ny, nx = pupil.shape
    N = ny * nx

    source_indices = np.where(source > 1e-10)
    source_values = source[source_indices]

    M = len(source_indices[0])
    V = np.zeros((M, N), dtype=np.complex128)

    for idx in range(M):
        sy, sx = source_indices[0][idx], source_indices[1][idx]
        src_val = np.sqrt(source_values[idx])

        fs_x = fx[sy, sx]
        fs_y = fy[sy, sx]

        pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, dfx, dfy)
        V[idx, :] = src_val * pupil_shifted.flatten()

    if M <= N:
        VVh = V @ V.conj().T
        eigenvalues, eigenvectors = np.linalg.eigh(VVh)

        idx_sorted = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sorted]
        eigenvectors = eigenvectors[:, idx_sorted]

        num_terms = min(num_terms, M)
        eigenvalues = eigenvalues[:num_terms]
        eigenvectors = eigenvectors[:, :num_terms]

        eigenfunctions = np.zeros((num_terms, ny, nx), dtype=np.complex128)
        for i in range(num_terms):
            phi_flat = V.conj().T @ eigenvectors[:, i]
            norm = np.sqrt(np.sum(np.abs(phi_flat)**2))
            if norm > 1e-10:
                phi_flat = phi_flat / norm
            eigenfunctions[i, :, :] = phi_flat.reshape(ny, nx)
    else:
        VhV = V.conj().T @ V
        eigenvalues, eigenvectors = np.linalg.eigh(VhV)

        idx_sorted = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sorted]
        eigenvectors = eigenvectors[:, idx_sorted]

        num_terms = min(num_terms, N)
        eigenvalues = eigenvalues[:num_terms]
        eigenvectors = eigenvectors[:, :num_terms]

        eigenfunctions = np.zeros((num_terms, ny, nx), dtype=np.complex128)
        for i in range(num_terms):
            eigenfunctions[i, :, :] = eigenvectors[:, i].reshape(ny, nx)

    eigenvalues = np.real(eigenvalues)
    total_energy = np.sum(eigenvalues)
    if total_energy > 0:
        eigenvalues = eigenvalues / total_energy

    return eigenvalues, eigenfunctions


@jit(nopython=True, parallel=True, cache=True)
def _compute_tcc_kernel_simple(fx: np.ndarray, fy: np.ndarray,
                               pupil: np.ndarray, sigma: float,
                               cutoff: float) -> np.ndarray:
    """
    简化的 TCC 核计算（向后兼容）

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        pupil: 光瞳函数
        sigma: 部分相干因子
        cutoff: 截止频率

    Returns:
        TCC核
    """
    ny, nx = fx.shape
    tcc = np.zeros((ny, nx), dtype=np.float64)
    source_radius = sigma * cutoff

    for i in prange(ny):
        for j in range(nx):
            rho = np.sqrt(fx[i, j]**2 + fy[i, j]**2)
            if rho <= source_radius:
                tcc[i, j] = np.abs(pupil[i, j])**2

    total = np.sum(tcc)
    if total > 0:
        tcc = tcc / total

    return tcc


class PartialCoherentImaging:
    """
    部分相干成像模型类

    实现Hopkins部分相干成像理论，用于计算掩模图案在晶圆上的成像结果。
    支持完整TCC矩阵计算和SOCS低秩分解近似。
    """

    def __init__(self, optical_system: OpticalSystem, image_size: Tuple[int, int]):
        """
        初始化部分相干成像模型

        Args:
            optical_system: 光学系统参数
            image_size: 图像尺寸 (height, width)
        """
        self.optics = optical_system
        self.image_size = image_size
        self._setup_frequency_grid()
        self._compute_source_and_pupil()
        self._compute_transfer_functions()

    def _setup_frequency_grid(self):
        """设置频率网格"""
        ny, nx = self.image_size

        self.dfx = 1.0 / (nx * self.optics.pixel_size)
        self.dfy = 1.0 / (ny * self.optics.pixel_size)

        fx = np.fft.fftfreq(nx, self.optics.pixel_size)
        fy = np.fft.fftfreq(ny, self.optics.pixel_size)
        self.fx, self.fy = np.meshgrid(fx, fy)

    def _compute_source_and_pupil(self):
        """计算光源分布和光瞳函数"""
        cutoff = self.optics.cutoff_frequency

        self.pupil = _compute_pupil_function(
            self.fx, self.fy, cutoff,
            self.optics.defocus, self.optics.wavelength
        )

        self.source = generate_source(
            self.fx, self.fy,
            self.optics.illumination_type,
            self.optics.source_params,
            cutoff,
            self.optics.custom_source
        )

    def _compute_transfer_functions(self):
        """计算传递函数（TCC或SOCS分解）"""
        cutoff = self.optics.cutoff_frequency

        if self.optics.use_socs:
            self.socs_eigenvalues, self.socs_eigenfunctions = socs_decomposition(
                self.fx, self.fy,
                self.pupil, self.source,
                cutoff, self.dfx, self.dfy,
                self.optics.socs_num_terms
            )
            self.tcc = None
        else:
            self.tcc = compute_tcc_kernel_2d(
                self.fx, self.fy,
                self.pupil, self.source,
                cutoff, self.dfx, self.dfy
            )
            self.socs_eigenvalues = None
            self.socs_eigenfunctions = None

    def compute_aerial_image(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像（晶圆上的光强分布）

        使用Hopkins公式: I = ∫ S(fs) * |FFT^{-1}[M(f) * P(f - fs)]|^2 dfs
        或通过SOCS分解加速计算

        Args:
            mask: 掩模图案 (2D numpy数组, 0-1值)

        Returns:
            空间像光强分布
        """
        mask_c = mask.astype(np.complex128)
        ny, nx = mask.shape

        if self.optics.use_socs and self.socs_eigenvalues is not None:
            intensity = np.zeros((ny, nx), dtype=np.float64)
            mask_spectrum = np.fft.fft2(mask_c)

            for i, (lam, phi) in enumerate(zip(self.socs_eigenvalues, self.socs_eigenfunctions)):
                if lam < 1e-10:
                    continue
                filtered = mask_spectrum * phi
                field_i = np.fft.ifft2(filtered)
                intensity += lam * np.abs(field_i)**2
        else:
            intensity = np.zeros((ny, nx), dtype=np.float64)
            mask_spectrum = np.fft.fft2(mask_c)
            cutoff = self.optics.cutoff_frequency

            source_indices = np.where(self.source > 1e-10)
            source_values = self.source[source_indices]

            for idx in range(len(source_indices[0])):
                sy, sx = source_indices[0][idx], source_indices[1][idx]
                src_val = source_values[idx]

                fs_x = self.fx[sy, sx]
                fs_y = self.fy[sy, sx]

                if np.sqrt(fs_x**2 + fs_y**2) > cutoff:
                    continue

                pupil_shifted = _shift_pupil(
                    self.pupil, fs_x, fs_y, self.dfx, self.dfy
                )
                filtered = mask_spectrum * pupil_shifted
                field_i = np.fft.ifft2(filtered)
                intensity += src_val * np.abs(field_i)**2

        if intensity.max() > 0:
            intensity = intensity / intensity.max()

        return intensity.astype(np.float64)

    def compute_image_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像对掩模的梯度（用于优化）

        Args:
            mask: 掩模图案

        Returns:
            梯度数组
        """
        mask_c = mask.astype(np.complex128)
        ny, nx = mask.shape
        gradient = np.zeros((ny, nx), dtype=np.float64)
        mask_spectrum = np.fft.fft2(mask_c)
        cutoff = self.optics.cutoff_frequency

        if self.optics.use_socs and self.socs_eigenvalues is not None:
            for lam, phi in zip(self.socs_eigenvalues, self.socs_eigenfunctions):
                if lam < 1e-10:
                    continue
                filtered = mask_spectrum * phi
                field_i = np.fft.ifft2(filtered)
                grad_field_i = np.fft.ifft2(phi)
                gradient += 2 * lam * np.real(np.conj(field_i) * grad_field_i)
        else:
            source_indices = np.where(self.source > 1e-10)
            source_values = self.source[source_indices]

            for idx in range(len(source_indices[0])):
                sy, sx = source_indices[0][idx], source_indices[1][idx]
                src_val = source_values[idx]

                fs_x = self.fx[sy, sx]
                fs_y = self.fy[sy, sx]

                if np.sqrt(fs_x**2 + fs_y**2) > cutoff:
                    continue

                pupil_shifted = _shift_pupil(
                    self.pupil, fs_x, fs_y, self.dfx, self.dfy
                )
                filtered = mask_spectrum * pupil_shifted
                field_i = np.fft.ifft2(filtered)
                grad_field_i = np.fft.ifft2(pupil_shifted)
                gradient += 2 * src_val * np.real(np.conj(field_i) * grad_field_i)

        return gradient.astype(np.float64)

    def get_source_image(self) -> np.ndarray:
        """获取光源分布图像（fftshift后便于可视化）"""
        return np.fft.fftshift(self.source)

    def get_pupil_image(self) -> np.ndarray:
        """获取光瞳函数图像（fftshift后便于可视化）"""
        return np.fft.fftshift(np.abs(self.pupil))

    def get_tcc_image(self) -> Optional[np.ndarray]:
        """获取TCC核图像（fftshift后便于可视化）"""
        if self.tcc is not None:
            return np.fft.fftshift(self.tcc)
        return None


@jit(nopython=True, cache=True)
def _apply_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    应用阈值处理（模拟光刻胶响应）

    Args:
        image: 输入图像
        threshold: 阈值

    Returns:
        二值化图像
    """
    ny, nx = image.shape
    result = np.zeros((ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            if image[i, j] >= threshold:
                result[i, j] = 1.0

    return result


def simulate_wafer_image(mask: np.ndarray,
                         optical_system: Optional[OpticalSystem] = None,
                         threshold: float = 0.3,
                         apply_resist: bool = True,
                         dose: float = 1.0) -> np.ndarray:
    """
    模拟晶圆成像

    完整的成像流程：掩模 -> 光学成像 -> 剂量缩放 -> 光刻胶响应

    Args:
        mask: 掩模图案 (2D numpy数组)
        optical_system: 光学系统参数，None则使用默认参数
        threshold: 光刻胶阈值
        apply_resist: 是否应用光刻胶阈值处理
        dose: 曝光相对剂量，1.0为标称剂量，大于1为过曝，小于1为欠曝

    Returns:
        晶圆成像结果
    """
    if optical_system is None:
        optical_system = OpticalSystem()

    imaging_model = PartialCoherentImaging(optical_system, mask.shape)

    aerial_image = imaging_model.compute_aerial_image(mask)

    if dose != 1.0:
        aerial_image = np.clip(aerial_image * dose, 0.0, 1.0)

    if apply_resist:
        wafer_image = _apply_threshold(aerial_image, threshold)
    else:
        wafer_image = aerial_image

    return wafer_image


def simulate_multi_process(
    mask: np.ndarray,
    conditions: List[ProcessCondition],
    base_optics: Optional[OpticalSystem] = None,
    threshold: float = 0.3,
    apply_resist: bool = True
) -> MultiProcessSimulationResult:
    """
    多工艺条件联合仿真

    对给定的多组工艺条件，依次进行光学成像仿真。
    支持 focus（defocus）、dose、NA、sigma 等参数扫描。

    Args:
        mask: 掩模图案 (2D numpy数组)
        conditions: 工艺条件列表
        base_optics: 基础光学系统（提供未在ProcessCondition中定义的参数）
        threshold: 光刻胶阈值
        apply_resist: 是否应用光刻胶阈值处理

    Returns:
        MultiProcessSimulationResult，包含所有工艺条件下的仿真结果
    """
    if base_optics is None:
        base_optics = OpticalSystem()

    aerial_images = []
    wafer_images = []

    for cond in conditions:
        optics = cond.to_optical_system(base_optics=base_optics)
        imaging_model = PartialCoherentImaging(optics, mask.shape)
        aerial = imaging_model.compute_aerial_image(mask)

        if cond.dose != 1.0:
            aerial_dosed = np.clip(aerial * cond.dose, 0.0, 1.0)
        else:
            aerial_dosed = aerial.copy()

        if apply_resist:
            wafer = _apply_threshold(aerial_dosed, threshold)
        else:
            wafer = aerial_dosed.copy()

        aerial_images.append(aerial_dosed)
        wafer_images.append(wafer)

    return MultiProcessSimulationResult(
        aerial_images=aerial_images,
        wafer_images=wafer_images,
        conditions=conditions,
        threshold=threshold
    )


def create_focus_dose_window(
    focus_range: Tuple[float, float, int] = (-100, 100, 5),
    dose_range: Tuple[float, float, int] = (0.8, 1.2, 5),
    na: float = 1.35,
    sigma: float = 0.75,
    wavelength: float = 193.0,
    center_weight: Optional[float] = 2.0,
    edge_weight: float = 1.0
) -> List[ProcessCondition]:
    """
    便捷函数：创建经典的 focus-dose 工艺窗口

    生成 focus × dose 的二维工艺窗口扫描条件，用于工艺鲁棒性优化。
    中心条件（最佳焦点、标称剂量）权重更高，四角条件权重较低。

    Args:
        focus_range: (start_nm, stop_nm, n_points) 离焦量扫描范围
        dose_range: (start, stop, n_points) 曝光剂量扫描范围
        na: 固定的数值孔径
        sigma: 固定的部分相干因子
        wavelength: 固定的波长 (nm)
        center_weight: 中心条件额外权重倍率；None则不做区分
        edge_weight: 边界条件的基础权重

    Returns:
        ProcessCondition列表，中心条件权重被center_weight放大
    """
    pw = ProcessWindow(
        defocus_values=focus_range,
        dose_values=dose_range,
        na_values=na,
        sigma_values=sigma,
        wavelength_values=wavelength,
        default_weight=edge_weight
    )

    return pw.generate_conditions(center_weight_boost=center_weight)


def create_full_process_window(
    focus_values: Any = (-100, 0, 100, 3),
    dose_values: Any = (0.85, 1.0, 1.15, 3),
    na_values: Any = (1.30, 1.35, 3),
    sigma_values: Any = (0.65, 0.75, 0.85, 3),
    wavelength: float = 193.0,
    center_weight_boost: Optional[float] = 3.0
) -> List[ProcessCondition]:
    """
    便捷函数：创建完整四维工艺窗口

    包含 focus、dose、NA、sigma 四维参数扫描。
    用于对工艺窗口进行全面的鲁棒性验证和优化。

    Args:
        focus_values: 离焦量扫描范围（标量/列表/三元组）
        dose_values: 剂量扫描范围
        na_values: 数值孔径扫描范围
        sigma_values: 部分相干因子扫描范围
        wavelength: 固定波长
        center_weight_boost: 中心条件权重倍率

    Returns:
        ProcessCondition列表
    """
    pw = ProcessWindow(
        defocus_values=focus_values,
        dose_values=dose_values,
        na_values=na_values,
        sigma_values=sigma_values,
        wavelength_values=wavelength,
        default_weight=1.0
    )

    return pw.generate_conditions(center_weight_boost=center_weight_boost)
