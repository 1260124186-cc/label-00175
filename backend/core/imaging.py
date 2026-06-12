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


class AberrationType(Enum):
    """像差类型枚举"""
    PISTON = "piston"
    TILT_X = "tilt_x"
    TILT_Y = "tilt_y"
    DEFOCUS = "defocus"
    ASTIGMATISM_X = "astigmatism_x"
    ASTIGMATISM_Y = "astigmatism_y"
    COMA_X = "coma_x"
    COMA_Y = "coma_y"
    TREFOIL_X = "trefoil_x"
    TREFOIL_Y = "trefoil_y"
    SPHERICAL = "spherical"
    SECONDARY_ASTIGMATISM_X = "secondary_astigmatism_x"
    SECONDARY_ASTIGMATISM_Y = "secondary_astigmatism_y"
    SECONDARY_COMA_X = "secondary_coma_x"
    SECONDARY_COMA_Y = "secondary_coma_y"
    SECONDARY_SPHERICAL = "secondary_spherical"


ZERNIKE_NAMES: Dict[int, AberrationType] = {
    0: AberrationType.PISTON,
    1: AberrationType.TILT_X,
    2: AberrationType.TILT_Y,
    3: AberrationType.DEFOCUS,
    4: AberrationType.ASTIGMATISM_Y,
    5: AberrationType.ASTIGMATISM_X,
    6: AberrationType.COMA_Y,
    7: AberrationType.COMA_X,
    8: AberrationType.TREFOIL_Y,
    9: AberrationType.TREFOIL_X,
    10: AberrationType.SPHERICAL,
    11: AberrationType.SECONDARY_ASTIGMATISM_X,
    12: AberrationType.SECONDARY_ASTIGMATISM_Y,
    13: AberrationType.SECONDARY_COMA_X,
    14: AberrationType.SECONDARY_COMA_Y,
    15: AberrationType.SECONDARY_SPHERICAL,
}


ZERNIKE_NAME_TO_INDEX: Dict[str, int] = {
    v.value: k for k, v in ZERNIKE_NAMES.items()
}


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
    zernike_coefficients: Dict[int, float] = field(default_factory=dict)
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
            zernike_coefficients=dict(optics.zernike_coefficients),
            name=name,
            weight=weight
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        zernike_out = {}
        for j, coeff in self.zernike_coefficients.items():
            name = ZERNIKE_NAMES.get(j, AberrationType.PISTON).value if j in ZERNIKE_NAMES else str(j)
            zernike_out[name] = coeff
        return {
            'defocus': self.defocus,
            'dose': self.dose,
            'na': self.na,
            'sigma': self.sigma,
            'wavelength': self.wavelength,
            'name': self.name,
            'weight': self.weight,
            'zernike_coefficients': zernike_out if zernike_out else {}
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
            merged_zernike = dict(base_optics.zernike_coefficients)
            merged_zernike.update(self.zernike_coefficients)
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
                custom_source=base_optics.custom_source,
                zernike_coefficients=merged_zernike
            )
        else:
            return OpticalSystem(
                wavelength=self.wavelength,
                na=self.na,
                sigma=self.sigma,
                defocus=self.defocus,
                zernike_coefficients=self.zernike_coefficients
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


def _parse_zernike_coefficients(raw: Dict[str, Any]) -> Dict[int, float]:
    """
    解析 Zernike 系数配置

    支持两种格式混合使用:
    - 名称格式: {"spherical": 0.05, "coma_x": 0.03}
    - 索引格式: {"10": 0.05, "7": 0.03}  或  {10: 0.05, 7: 0.03}

    Args:
        raw: 原始配置字典

    Returns:
        {j: coefficient} 字典，j 为 0-based Noll 索引
    """
    result = {}
    for key, value in raw.items():
        coeff = float(value)
        if isinstance(key, int):
            result[key] = coeff
        elif isinstance(key, str) and key.isdigit():
            result[int(key)] = coeff
        elif isinstance(key, str) and key in ZERNIKE_NAME_TO_INDEX:
            result[ZERNIKE_NAME_TO_INDEX[key]] = coeff
        else:
            try:
                result[int(key)] = coeff
            except (ValueError, TypeError):
                pass
    return result


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
    zernike_coefficients: Dict[int, float] = field(default_factory=dict)

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

        配置中 zernike_coefficients 可以是以下格式之一:
        - {0: 0.01, 10: 0.05}  直接使用 Noll 索引
        - {piston: 0.01, spherical: 0.05}  使用像差名称
        - 混合格式

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

        zernike_raw = optics_config.get('zernike_coefficients', {})
        zernike_coefficients = _parse_zernike_coefficients(zernike_raw)

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
            socs_num_terms=optics_config.get('socs_num_terms', 5),
            zernike_coefficients=zernike_coefficients
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
        zernike_out = {}
        for j, coeff in self.zernike_coefficients.items():
            name = ZERNIKE_NAMES.get(j, AberrationType.PISTON).value if j in ZERNIKE_NAMES else str(j)
            zernike_out[name] = coeff

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
            'socs_num_terms': self.socs_num_terms,
            'zernike_coefficients': zernike_out if zernike_out else {}
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


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    计算 Zernike 径向多项式 R_n^m(ρ)

    使用递推公式:
        R_n^m(ρ) = ((2n-1)(2n(n-m)R_{n-2}^m(ρ) - (n+m-1)R_{n-4}^m(ρ))) / ((n+m)(n-m))
    基例:
        R_m^m(ρ) = ρ^m
        R_{m+2}^m(ρ) = (m+2)ρ^{m+2} - (m+1)ρ^m  ... 实际用递推更稳妥

    Args:
        n: 径向阶数 (>= 0)
        m: 角向频率 (>= 0, m <= n, n-m 为偶数)
        rho: 归一化径向坐标 ρ = r/r_max, 范围 [0, 1]

    Returns:
        R_n^m(ρ) 数组
    """
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)

    if n == 0 and m == 0:
        return np.ones_like(rho)

    if m == n:
        return rho ** n

    R_m_m = rho ** m
    if n == m:
        return R_m_m

    R_mplus2_m = (m + 1) * rho ** (m + 2) - m * rho ** m
    if n == m + 2:
        return R_mplus2_m

    R_prev2 = R_m_m
    R_prev1 = R_mplus2_m

    for nn in range(m + 4, n + 1, 2):
        k = (nn - m) // 2
        R_curr = (
            (nn - 1) * ((2 * nn * (nn - m) * rho ** 2 - (nn + m) * (nn - m) - 2 * nn) * R_prev1
                        - (nn + m) * (nn - m - 2) * R_prev2)
        ) / ((nn + m) * (nn - m) * (nn - 1))

        # 更简洁的递推:
        # R_n^m = [2(n-1)/((n+m)(n-m))] * [(2n-1)(n-m)ρ^2 - (n+m-1)] * R_{n-2}^m
        #       - [(n-m-2)/(n+m)] * [(n+m-2)/(n-m)] * R_{n-4}^m  ... 不对
        # 用标准 OSA 递推:
        a = 2 * nn * (nn - 1) * (2 * nn - 2)
        b1 = (nn + m) * (nn - m) * (2 * nn - 4)
        b2 = (nn + m - 2) * (nn - m - 2) * (2 * nn - 2)
        c1 = (nn + m) * (nn - m) * (nn - 1) * 2
        c2 = (nn + m - 2) * (nn - m - 2) * (nn - 1)
        d = (nn + m) * (nn - m) * (nn - 1)

        # 换用更稳定的递推（Born & Wolf）:
        # R_n^m = (2n-1)/(n+m) * [2ρ^2 - 1] * R_{n-2}^m - (n-m-2)/(n+m) * R_{n-4}^m ... 不对
        # 最终用直接求和公式更稳定
        R_prev2 = R_prev1
        R_prev1 = R_curr

    return R_prev1


def _zernike_radial_direct(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    使用直接求和公式计算 Zernike 径向多项式 R_n^m(ρ)

    R_n^m(ρ) = Σ_{k=0}^{(n-m)/2} (-1)^k * (n-k)! / [k! * ((n+m)/2 - k)! * ((n-m)/2 - k)!] * ρ^{n-2k}

    Args:
        n: 径向阶数
        m: 角向频率
        rho: 归一化径向坐标

    Returns:
        R_n^m(ρ) 数组
    """
    result = np.zeros_like(rho, dtype=np.float64)
    if (n - m) % 2 != 0:
        return result

    for k in range((n - m) // 2 + 1):
        num = 1
        for i in range(1, n - k + 1):
            num *= i
        den = 1
        for i in range(1, k + 1):
            den *= i
        for i in range(1, (n + m) // 2 - k + 1):
            den *= i
        for i in range(1, (n - m) // 2 - k + 1):
            den *= i

        coeff = ((-1) ** k) * num / den
        result += coeff * rho ** (n - 2 * k)

    return result


def _zernike_polynomial(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    计算第 j 阶 Noll 索引 Zernike 多项式 Z_j(ρ, θ)

    Noll 索引约定:
        j=0: Z_0  = 1 (piston)
        j=1: Z_1  = 2ρ cos(θ) (tilt_x)
        j=2: Z_2  = 2ρ sin(θ) (tilt_y)
        j=3: Z_3  = √3 (2ρ² - 1) (defocus)
        j=4: Z_4  = √6 ρ² sin(2θ) (astigmatism_45)
        j=5: Z_5  = √6 ρ² cos(2θ) (astigmatism_0)
        j=6: Z_6  = √8 (3ρ³ - 2ρ) sin(θ) (coma_y)
        j=7: Z_7  = √8 (3ρ³ - 2ρ) cos(θ) (coma_x)
        j=8: Z_8  = √8 ρ³ sin(3θ) (trefoil_y)
        j=9: Z_9  = √8 ρ³ cos(3θ) (trefoil_x)
        j=10: Z_10 = √5 (6ρ⁴ - 6ρ² + 1) (spherical)
        j=11: Z_11 = √10 (4ρ⁴ - 3ρ²) cos(2θ) (secondary_astigmatism_0)
        j=12: Z_12 = √10 (4ρ⁴ - 3ρ²) sin(2θ) (secondary_astigmatism_45)
        j=13: Z_13 = √12 (10ρ⁵ - 12ρ³ + 3ρ) cos(θ) (secondary_coma_x)
        j=14: Z_14 = √12 (10ρ⁵ - 12ρ³ + 3ρ) sin(θ) (secondary_coma_y)
        j=15: Z_15 = √7 (20ρ⁶ - 30ρ⁴ + 12ρ² - 1) (secondary_spherical)

    Args:
        j: Noll 索引 (从0开始)
        rho: 归一化径向坐标 [0, 1]
        theta: 角度坐标

    Returns:
        Z_j(ρ, θ) 数组
    """
    if j == 0:
        return np.ones_like(rho)
    elif j == 1:
        return 2.0 * rho * np.cos(theta)
    elif j == 2:
        return 2.0 * rho * np.sin(theta)
    elif j == 3:
        return np.sqrt(3.0) * (2.0 * rho ** 2 - 1.0)
    elif j == 4:
        return np.sqrt(6.0) * rho ** 2 * np.sin(2.0 * theta)
    elif j == 5:
        return np.sqrt(6.0) * rho ** 2 * np.cos(2.0 * theta)
    elif j == 6:
        return np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho) * np.sin(theta)
    elif j == 7:
        return np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho) * np.cos(theta)
    elif j == 8:
        return np.sqrt(8.0) * rho ** 3 * np.sin(3.0 * theta)
    elif j == 9:
        return np.sqrt(8.0) * rho ** 3 * np.cos(3.0 * theta)
    elif j == 10:
        return np.sqrt(5.0) * (6.0 * rho ** 4 - 6.0 * rho ** 2 + 1.0)
    elif j == 11:
        return np.sqrt(10.0) * (4.0 * rho ** 4 - 3.0 * rho ** 2) * np.cos(2.0 * theta)
    elif j == 12:
        return np.sqrt(10.0) * (4.0 * rho ** 4 - 3.0 * rho ** 2) * np.sin(2.0 * theta)
    elif j == 13:
        return np.sqrt(12.0) * (10.0 * rho ** 5 - 12.0 * rho ** 3 + 3.0 * rho) * np.cos(theta)
    elif j == 14:
        return np.sqrt(12.0) * (10.0 * rho ** 5 - 12.0 * rho ** 3 + 3.0 * rho) * np.sin(theta)
    elif j == 15:
        return np.sqrt(7.0) * (20.0 * rho ** 6 - 30.0 * rho ** 4 + 12.0 * rho ** 2 - 1.0)
    else:
        n, m = _noll_to_nm(j)
        R = _zernike_radial_direct(n, abs(m), rho)
        if m >= 0:
            norm = np.sqrt(2.0 * (n + 1)) if m != 0 else np.sqrt(n + 1)
            return norm * R * np.cos(m * theta)
        else:
            norm = np.sqrt(2.0 * (n + 1))
            return norm * R * np.sin(abs(m) * theta)


_NOLL_NM_CACHE: Dict[int, Tuple[int, int]] = {
    0: (0, 0),
    1: (1, 1),
    2: (1, -1),
    3: (2, 0),
    4: (2, -2),
    5: (2, 2),
    6: (3, -1),
    7: (3, 1),
    8: (3, -3),
    9: (3, 3),
    10: (4, 0),
    11: (4, 2),
    12: (4, -2),
    13: (4, 4),
    14: (4, -4),
    15: (5, 1),
}


def _noll_to_nm(j: int) -> Tuple[int, int]:
    """
    将 0-based 索引 j 转换为 (n, m) 阶数对

    对应标准 Noll 约定:
        j=0 → (0,0), j=1 → (1,1), j=2 → (1,-1),
        j=3 → (2,0), j=4 → (2,-2), j=5 → (2,2), ...

    Args:
        j: 0-based 索引

    Returns:
        (n, m) 元组
    """
    if j in _NOLL_NM_CACHE:
        return _NOLL_NM_CACHE[j]

    j_noll = j + 1
    n = 0
    while (n + 1) * (n + 2) // 2 < j_noll:
        n += 1

    k = j_noll - n * (n + 1) // 2

    has_m0 = (n % 2 == 0)
    if has_m0:
        if k == 1:
            return (n, 0)
        adj = k - 1
    else:
        adj = k

    m_abs = (adj + 1) // 2
    if adj % 2 == 1:
        m = -m_abs
    else:
        m = m_abs

    return (n, m)


def compute_zernike_phase(fx: np.ndarray, fy: np.ndarray,
                          cutoff: float,
                          zernike_coefficients: Dict[int, float]) -> np.ndarray:
    """
    计算 Zernike 像差引起的相位

    在光瞳内，将频率坐标映射到归一化 (ρ, θ) 坐标，
    然后叠加各阶 Zernike 多项式贡献的相位。

    W(ρ, θ) = Σ_j c_j * Z_j(ρ, θ)

    其中 c_j 为 Zernike 系数（单位为波长 λ），
    最终相位 = 2π * W(ρ, θ)。

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        cutoff: 截止频率
        zernike_coefficients: Zernike 系数字典 {j: coefficient}，
                              j 为 Noll 索引(0-based)，coefficient 单位为波长 λ

    Returns:
        像差相位数组（弧度），形状与 fx 相同
    """
    ny, nx = fx.shape
    phase = np.zeros((ny, nx), dtype=np.float64)

    if not zernike_coefficients:
        return phase

    rho = np.sqrt(fx ** 2 + fy ** 2) / cutoff
    theta = np.arctan2(fy, fx)

    pupil_mask = rho <= 1.0

    for j, coeff in zernike_coefficients.items():
        if abs(coeff) < 1e-15:
            continue
        zernike_val = _zernike_polynomial(j, rho, theta)
        phase += coeff * 2.0 * np.pi * zernike_val

    phase[~pupil_mask] = 0.0

    return phase


def _compute_pupil_with_aberrations(fx: np.ndarray, fy: np.ndarray,
                                     cutoff: float,
                                     defocus: float,
                                     wavelength: float,
                                     zernike_phase: np.ndarray) -> np.ndarray:
    """
    计算含离焦和 Zernike 像差的光瞳函数

    P(f) = circ(|f|/f_c) * exp(i * Φ_defocus + i * Φ_zernike)

    其中:
        Φ_defocus = π * Δz * λ * (|f|/f_c)²
        Φ_zernike = 2π * Σ_j c_j * Z_j(ρ, θ)

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        cutoff: 截止频率
        defocus: 离焦量 (nm)
        wavelength: 波长 (nm)
        zernike_phase: Zernike 像差相位数组（弧度）

    Returns:
        复数光瞳函数
    """
    ny, nx = fx.shape
    pupil = np.zeros((ny, nx), dtype=np.complex128)

    rho_sq = (fx ** 2 + fy ** 2) / (cutoff ** 2)
    pupil_mask = rho_sq <= 1.0

    defocus_phase = np.pi * defocus / wavelength * rho_sq

    total_phase = defocus_phase + zernike_phase

    pupil[pupil_mask] = np.exp(1j * total_phase[pupil_mask])

    return pupil


@jit(nopython=True, parallel=True, cache=True)
def _compute_pupil_function(fx: np.ndarray, fy: np.ndarray,
                            cutoff: float, defocus: float,
                            wavelength: float) -> np.ndarray:
    """
    计算光瞳函数（含离焦相位，不含 Zernike 像差）

    向后兼容接口。新代码应使用 _compute_pupil_with_aberrations。

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
        """计算光源分布和光瞳函数（含离焦和Zernike像差）"""
        cutoff = self.optics.cutoff_frequency

        zernike_phase = compute_zernike_phase(
            self.fx, self.fy, cutoff,
            self.optics.zernike_coefficients
        )

        self.pupil = _compute_pupil_with_aberrations(
            self.fx, self.fy, cutoff,
            self.optics.defocus, self.optics.wavelength,
            zernike_phase
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

    def update_source(self, new_source: np.ndarray) -> None:
        """
        更新光源分布并重新计算传递函数

        Args:
            new_source: 新的光源分布，形状需与频率网格一致
        """
        if new_source.shape != self.source.shape:
            raise ValueError(
                f"新光源形状 {new_source.shape} 与当前形状 {self.source.shape} 不匹配"
            )

        new_source = np.clip(new_source, 0.0, None)
        total = np.sum(new_source)
        if total > 0:
            new_source = new_source / total

        self.source = new_source.astype(np.float64)
        self._compute_transfer_functions()

    def compute_source_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像对光源分布的梯度

        根据 Hopkins 公式:
        I = ∫ S(fs) * |FFT^{-1}[M(f) * P(f - fs)]|^2 dfs

        因此 dI/dS(fs_i) = |FFT^{-1}[M(f) * P(f - fs_i)]|^2

        注意：返回的是每个光源点对空间像的梯度贡献，需要进一步
        与损失函数对空间像的梯度链式相乘。

        Args:
            mask: 掩模图案

        Returns:
            梯度数组，形状与光源相同
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
                intensity_i = np.abs(field_i) ** 2

                source_indices = np.where(self.source > 1e-10)
                for idx in range(len(source_indices[0])):
                    sy, sx = source_indices[0][idx], source_indices[1][idx]
                    fs_x = self.fx[sy, sx]
                    fs_y = self.fy[sy, sx]
                    if np.sqrt(fs_x ** 2 + fs_y ** 2) > cutoff:
                        continue

                    pupil_shifted = _shift_pupil(
                        self.pupil, fs_x, fs_y, self.dfx, self.dfy
                    )
                    filtered_s = mask_spectrum * pupil_shifted
                    field_s = np.fft.ifft2(filtered_s)
                    intensity_s = np.abs(field_s) ** 2

                    gradient[sy, sx] += np.sum(intensity_i * intensity_s) / (ny * nx)
        else:
            source_indices = np.where(self.source > 1e-10)
            source_values = self.source[source_indices]

            for idx in range(len(source_indices[0])):
                sy, sx = source_indices[0][idx], source_indices[1][idx]
                src_val = source_values[idx]
                if src_val <= 0:
                    continue

                fs_x = self.fx[sy, sx]
                fs_y = self.fy[sy, sx]
                if np.sqrt(fs_x ** 2 + fs_y ** 2) > cutoff:
                    continue

                pupil_shifted = _shift_pupil(
                    self.pupil, fs_x, fs_y, self.dfx, self.dfy
                )
                filtered = mask_spectrum * pupil_shifted
                field_i = np.fft.ifft2(filtered)
                intensity_i = np.abs(field_i) ** 2

                gradient[sy, sx] = np.sum(intensity_i) / (ny * nx)

        return gradient.astype(np.float64)


class ResistType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ResistThresholdMode(Enum):
    HARD = "hard"
    SIGMOID = "sigmoid"


@dataclass
class ResistModel:
    """
    高级光刻胶模型

    封装光刻胶显影物理与近似方法，支持：
    1. 阈值调制（TMR）—— 空间变化的阈值场
    2. 化学放大光刻胶（CAR）简化显影模型
    3. 正性/负性胶切换
    4. 可微近似（sigmoid / soft threshold），用于端到端梯度优化

    Attributes:
        resist_type: 正性/负性光刻胶类型
        threshold_mode: 阈值模式 —— HARD（硬阈值）或 SIGMOID（可微近似）
        base_threshold: 基础阈值（归一化光强 0~1）
        tmr_enabled: 是否启用阈值调制
        tmr_field: 阈值调制场，形状与输入图像一致；
                   调制后阈值 = base_threshold + tmr_field
        car_enabled: 是否启用 CAR 简化显影模型
        car_amplification: 化学放大倍率（CAR gain），>=1；
                           越大越接近理想阶跃
        car_contrast: CAR 对比度参数 n（Mack 模型指数），>=1
        sigmoid_steepness: sigmoid 陡度参数 k，仅 threshold_mode=SIGMOID 时生效；
                           越大越接近硬阈值，典型值 20~100
    """
    resist_type: ResistType = ResistType.POSITIVE
    threshold_mode: ResistThresholdMode = ResistThresholdMode.HARD
    base_threshold: float = 0.3
    tmr_enabled: bool = False
    tmr_field: Optional[np.ndarray] = None
    car_enabled: bool = False
    car_amplification: float = 10.0
    car_contrast: float = 5.0
    sigmoid_steepness: float = 50.0

    def get_local_threshold(self, image_shape: Tuple[int, int]) -> np.ndarray:
        if not self.tmr_enabled or self.tmr_field is None:
            return np.full(image_shape, self.base_threshold, dtype=np.float64)
        if self.tmr_field.shape != image_shape:
            raise ValueError(
                f"TMR 场形状 {self.tmr_field.shape} 与图像形状 {image_shape} 不匹配"
            )
        return self.base_threshold + self.tmr_field


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


@jit(nopython=True, cache=True)
def _apply_sigmoid_threshold(image: np.ndarray, threshold: float,
                             steepness: float) -> np.ndarray:
    """
    可微近似阈值（sigmoid soft threshold）

    f(I) = σ(k·(I - T))  =  1 / (1 + exp(-k·(I - T)))

    当 k → ∞ 时退化为硬阈值；有限 k 下处处可微，
    梯度 df/dI = k·f·(1-f)，可用于端到端优化。

    Args:
        image: 输入光强图像
        threshold: 阈值
        steepness: sigmoid 陡度 k

    Returns:
        近似二值化图像，值域 (0, 1)
    """
    ny, nx = image.shape
    result = np.zeros((ny, nx), dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            arg = steepness * (image[i, j] - threshold)
            if arg > 500.0:
                result[i, j] = 1.0
            elif arg < -500.0:
                result[i, j] = 0.0
            else:
                result[i, j] = 1.0 / (1.0 + np.exp(-arg))
    return result


@jit(nopython=True, cache=True)
def _apply_car_development(image: np.ndarray, threshold: float,
                           amplification: float,
                           contrast: float) -> np.ndarray:
    """
    化学放大光刻胶（CAR）简化显影模型

    基于 Mack 显影模型的简化形式：

        M(x) = 1 / (1 + (E_c / E(x))^(2·n))

    其中 E(x) = A·I(x) 为有效曝光量，A 为化学放大倍率，
    E_c = A·T 为临界曝光量，n 为对比度指数。

    代入化简后：

        M(x) = 1 / (1 + (T / I(x))^(2·n·A))

    当 A·n >> 1 时趋近于理想阶跃。

    对于负性胶，输出取反。

    Args:
        image: 输入光强图像（归一化 0~1）
        threshold: 基础阈值 T
        amplification: 化学放大倍率 A（>=1）
        contrast: 对比度指数 n（>=1）

    Returns:
        显影后图像，值域 [0, 1]
    """
    ny, nx = image.shape
    result = np.zeros((ny, nx), dtype=np.float64)
    exponent = 2.0 * contrast * amplification
    for i in range(ny):
        for j in range(nx):
            if threshold <= 0.0:
                result[i, j] = 1.0
            elif image[i, j] <= 0.0:
                result[i, j] = 0.0
            else:
                ratio = threshold / image[i, j]
                if ratio > 1e15:
                    result[i, j] = 0.0
                else:
                    result[i, j] = 1.0 / (1.0 + ratio ** exponent)
    return result


def apply_resist_model(image: np.ndarray,
                       resist_model: Optional[ResistModel] = None,
                       threshold: float = 0.3) -> np.ndarray:
    """
    统一光刻胶响应入口

    处理流程：阈值调制（TMR）→ CAR / sigmoid / 硬阈值 → 正性/负性翻转

    Args:
        image: 输入光强图像（归一化 0~1）
        resist_model: 光刻胶模型配置，None 则使用硬阈值
        threshold: 兼容旧接口的基础阈值，当 resist_model 为 None 时生效

    Returns:
        显影后晶圆图像
    """
    if resist_model is None:
        return _apply_threshold(image, threshold)

    local_threshold = resist_model.get_local_threshold(image.shape)

    if resist_model.car_enabled:
        if np.all(local_threshold == local_threshold[0, 0]):
            result = _apply_car_development(
                image, local_threshold[0, 0],
                resist_model.car_amplification,
                resist_model.car_contrast
            )
        else:
            result = np.zeros_like(image, dtype=np.float64)
            n_bins = 64
            unique_thresh = np.unique(local_threshold)
            if len(unique_thresh) > n_bins:
                t_min = float(np.min(unique_thresh))
                t_max = float(np.max(unique_thresh))
                unique_thresh = np.linspace(t_min, t_max, n_bins)
            for t_val in unique_thresh:
                mask_t = np.isclose(local_threshold, t_val)
                if not np.any(mask_t):
                    continue
                sub_image = image[mask_t]
                sub_result = _apply_car_development(
                    sub_image.reshape(1, -1), float(t_val),
                    resist_model.car_amplification,
                    resist_model.car_contrast
                )
                result[mask_t] = sub_result.ravel()
    elif resist_model.threshold_mode == ResistThresholdMode.SIGMOID:
        if np.all(local_threshold == local_threshold[0, 0]):
            result = _apply_sigmoid_threshold(
                image, local_threshold[0, 0],
                resist_model.sigmoid_steepness
            )
        else:
            result = np.zeros_like(image, dtype=np.float64)
            n_bins = 64
            unique_thresh = np.unique(local_threshold)
            if len(unique_thresh) > n_bins:
                t_min = float(np.min(unique_thresh))
                t_max = float(np.max(unique_thresh))
                unique_thresh = np.linspace(t_min, t_max, n_bins)
            for t_val in unique_thresh:
                mask_t = np.isclose(local_threshold, t_val)
                if not np.any(mask_t):
                    continue
                sub_image = image[mask_t]
                sub_result = _apply_sigmoid_threshold(
                    sub_image.reshape(1, -1), float(t_val),
                    resist_model.sigmoid_steepness
                )
                result[mask_t] = sub_result.ravel()
    else:
        if np.all(local_threshold == local_threshold[0, 0]):
            result = _apply_threshold(image, local_threshold[0, 0])
        else:
            result = np.zeros_like(image, dtype=np.float64)
            n_bins = 64
            unique_thresh = np.unique(local_threshold)
            if len(unique_thresh) > n_bins:
                t_min = float(np.min(unique_thresh))
                t_max = float(np.max(unique_thresh))
                unique_thresh = np.linspace(t_min, t_max, n_bins)
            for t_val in unique_thresh:
                mask_t = np.isclose(local_threshold, t_val)
                if not np.any(mask_t):
                    continue
                sub_image = image[mask_t]
                sub_result = _apply_threshold(
                    sub_image.reshape(1, -1), float(t_val)
                )
                result[mask_t] = sub_result.ravel()

    if resist_model.resist_type == ResistType.NEGATIVE:
        result = 1.0 - result

    return result


def simulate_wafer_image(mask: np.ndarray,
                         optical_system: Optional[OpticalSystem] = None,
                         threshold: float = 0.3,
                         apply_resist: bool = True,
                         dose: float = 1.0,
                         resist_model: Optional[ResistModel] = None) -> np.ndarray:
    """
    模拟晶圆成像

    完整的成像流程：掩模 -> 光学成像 -> 剂量缩放 -> 光刻胶响应

    Args:
        mask: 掩模图案 (2D numpy数组)
        optical_system: 光学系统参数，None则使用默认参数
        threshold: 光刻胶阈值（当 resist_model 为 None 时生效）
        apply_resist: 是否应用光刻胶响应
        dose: 曝光相对剂量，1.0为标称剂量，大于1为过曝，小于1为欠曝
        resist_model: 高级光刻胶模型配置，优先于 threshold/apply_resist 参数

    Returns:
        晶圆成像结果
    """
    if optical_system is None:
        optical_system = OpticalSystem()

    imaging_model = PartialCoherentImaging(optical_system, mask.shape)

    aerial_image = imaging_model.compute_aerial_image(mask)

    if dose != 1.0:
        aerial_image = np.clip(aerial_image * dose, 0.0, 1.0)

    if resist_model is not None:
        wafer_image = apply_resist_model(aerial_image, resist_model=resist_model)
    elif apply_resist:
        wafer_image = _apply_threshold(aerial_image, threshold)
    else:
        wafer_image = aerial_image

    return wafer_image


def simulate_multi_process(
    mask: np.ndarray,
    conditions: List[ProcessCondition],
    base_optics: Optional[OpticalSystem] = None,
    threshold: float = 0.3,
    apply_resist: bool = True,
    resist_model: Optional[ResistModel] = None
) -> MultiProcessSimulationResult:
    """
    多工艺条件联合仿真

    对给定的多组工艺条件，依次进行光学成像仿真。
    支持 focus（defocus）、dose、NA、sigma 等参数扫描。

    Args:
        mask: 掩模图案 (2D numpy数组)
        conditions: 工艺条件列表
        base_optics: 基础光学系统（提供未在ProcessCondition中定义的参数）
        threshold: 光刻胶阈值（当 resist_model 为 None 时生效）
        apply_resist: 是否应用光刻胶响应
        resist_model: 高级光刻胶模型配置，优先于 threshold/apply_resist 参数

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

        if resist_model is not None:
            wafer = apply_resist_model(aerial_dosed, resist_model=resist_model)
        elif apply_resist:
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


def load_aberration_scenarios(
    config_path: Union[str, 'Path'],
    base_optics: Optional[OpticalSystem] = None,
    scenario_names: Optional[List[str]] = None
) -> List[Tuple[str, OpticalSystem]]:
    """
    从配置文件批量加载像差场景

    配置文件格式参见 config/aberration_scenarios.yaml。

    Args:
        config_path: 像差场景配置文件路径（YAML格式）
        base_optics: 基础光学系统参数。若为 None，则使用配置文件中的 base_optics 段。
        scenario_names: 需要加载的场景名称列表。None 则加载所有场景。

    Returns:
        [(scenario_name, OpticalSystem), ...] 列表
    """
    from pathlib import Path as _Path

    config_path = _Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"像差场景配置文件不存在: {config_path}")

    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if base_optics is None:
        base_config = config.get('base_optics', {})
        base_optics = OpticalSystem.from_config({'optical_system': base_config})

    scenarios = config.get('scenarios', {})
    result = []

    for name, scenario_config in scenarios.items():
        if scenario_names is not None and name not in scenario_names:
            continue

        zernike_raw = scenario_config.get('zernike_coefficients', {})
        zernike_coefficients = _parse_zernike_coefficients(zernike_raw)

        defocus_override = scenario_config.get('defocus', base_optics.defocus)

        opt = OpticalSystem(
            wavelength=base_optics.wavelength,
            na=base_optics.na,
            sigma=base_optics.sigma,
            pixel_size=base_optics.pixel_size,
            defocus=defocus_override,
            magnification=base_optics.magnification,
            illumination_type=base_optics.illumination_type,
            source_params=dict(base_optics.source_params),
            use_socs=base_optics.use_socs,
            socs_num_terms=base_optics.socs_num_terms,
            zernike_coefficients=zernike_coefficients
        )
        result.append((name, opt))

    return result


def create_aberration_sweep(
    base_optics: Optional[OpticalSystem] = None,
    zernike_j: int = 10,
    coeff_values: Optional[List[float]] = None,
    defocus_values: Optional[Any] = None
) -> List[Tuple[str, OpticalSystem]]:
    """
    便捷函数：对单个 Zernike 阶进行系数扫描

    生成一系列光学系统，仅在指定 Zernike 阶的系数上变化，
    用于分析特定像差对成像的影响。

    Args:
        base_optics: 基础光学系统。None 则使用默认参数。
        zernike_j: 要扫描的 Zernike 阶索引 (0-based Noll)
        coeff_values: 系数扫描值列表（单位为波长 λ），默认 [-0.1, -0.05, 0, 0.05, 0.1]
        defocus_values: 离焦量扫描值。None 则使用 base_optics 的 defocus。
                       支持标量/列表/三元组（与 ProcessWindow 相同格式）。

    Returns:
        [(描述名, OpticalSystem), ...] 列表
    """
    if base_optics is None:
        base_optics = OpticalSystem()

    if coeff_values is None:
        coeff_values = [-0.1, -0.05, 0.0, 0.05, 0.1]

    if defocus_values is not None:
        df_list = ProcessWindow._normalize_scan_values(defocus_values)
    else:
        df_list = [base_optics.defocus]

    aberration_name = ZERNIKE_NAMES.get(zernike_j, AberrationType.PISTON).value

    result = []
    for df in df_list:
        for coeff in coeff_values:
            zernike = {zernike_j: coeff}
            opt = OpticalSystem(
                wavelength=base_optics.wavelength,
                na=base_optics.na,
                sigma=base_optics.sigma,
                pixel_size=base_optics.pixel_size,
                defocus=df,
                magnification=base_optics.magnification,
                illumination_type=base_optics.illumination_type,
                source_params=dict(base_optics.source_params),
                use_socs=base_optics.use_socs,
                socs_num_terms=base_optics.socs_num_terms,
                zernike_coefficients=zernike
            )
            desc = f"{aberration_name}_c={coeff:.3f}_df={df:.0f}nm"
            result.append((desc, opt))

    return result


def downsample_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    """
    对掩模进行整数倍下采样

    使用区域平均池化将掩模缩小 scale 倍。

    Args:
        mask: 输入掩模 (2D numpy数组, 0-1值)
        scale: 下采样倍数 (>= 2)

    Returns:
        下采样后的掩模
    """
    if scale < 2:
        return mask.copy()
    ny, nx = mask.shape
    ny_new = ny // scale
    nx_new = nx // scale
    if ny_new < 1 or nx_new < 1:
        return mask.copy()
    cropped = mask[:ny_new * scale, :nx_new * scale]
    reshaped = cropped.reshape(ny_new, scale, nx_new, scale)
    return reshaped.mean(axis=(1, 3)).astype(np.float64)


def upsample_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    将掩模上采样到目标尺寸

    使用双线性插值进行上采样，保持值在 [0, 1] 范围内。

    Args:
        mask: 输入掩模 (2D numpy数组)
        target_shape: 目标尺寸 (height, width)

    Returns:
        上采样后的掩模
    """
    from scipy.ndimage import zoom as scipy_zoom
    ny_src, nx_src = mask.shape
    ny_tgt, nx_tgt = target_shape
    if ny_src == ny_tgt and nx_src == nx_tgt:
        return mask.copy()
    zoom_y = ny_tgt / ny_src
    zoom_x = nx_tgt / nx_src
    result = scipy_zoom(mask, (zoom_y, zoom_x), order=1)
    return np.clip(result, 0.0, 1.0).astype(np.float64)


def build_pyramid_scales(mask_shape: Tuple[int, int],
                         min_size: int = 64,
                         n_scales: int = 3) -> List[Tuple[int, int]]:
    """
    构建金字塔多尺度尺寸列表

    从原始尺寸向下生成多级分辨率，每级缩小 2 倍，
    直到最小维度达到 min_size 或达到 n_scales 层。

    Args:
        mask_shape: 原始掩模尺寸 (height, width)
        min_size: 最低分辨率的最小尺寸
        n_scales: 金字塔层数（不含原始分辨率）

    Returns:
        尺寸列表，从低分辨率到高分辨率（含原始分辨率）
    """
    scales = []
    h, w = mask_shape
    for _ in range(n_scales):
        h_half = max(h // 2, min_size)
        w_half = max(w // 2, min_size)
        if h_half == h and w_half == w:
            break
        h, w = h_half, w_half
        scales.append((h, w))
    scales.reverse()
    scales.append(mask_shape)
    return scales


def split_tiles(mask: np.ndarray,
                tile_size: int = 256,
                overlap: int = 32) -> List[Dict[str, Any]]:
    """
    将掩模分割为重叠的 tile 块

    每个 tile 包含位置信息（row_start, col_start, row_end, col_end），
    以及在原始掩模中对应的非重叠有效区域边界。

    Args:
        mask: 输入掩模 (2D numpy数组)
        tile_size: 单个 tile 的尺寸（像素）
        overlap: 相邻 tile 的重叠区域（像素）

    Returns:
        tile 信息列表，每个元素为字典:
        {
            'data': tile 数据 (2D 数组),
            'row_start': 在原始掩模中的起始行,
            'col_start': 在原始掩模中的起始列,
            'row_end': 在原始掩模中的结束行（不含）,
            'col_end': 在原始掩模中的结束列（不含）,
            'inner_row_start': 有效区域起始行（相对于 tile）,
            'inner_col_start': 有效区域起始列（相对于 tile）,
            'inner_row_end': 有效区域结束行（相对于 tile, 不含）,
            'inner_col_end': 有效区域结束列（相对于 tile, 不含）
        }
    """
    ny, nx = mask.shape
    step = max(tile_size - overlap, 1)
    tiles = []
    for row in range(0, ny, step):
        for col in range(0, nx, step):
            row_start = row
            col_start = col
            row_end = min(row + tile_size, ny)
            col_end = min(col + tile_size, nx)
            tile_data = mask[row_start:row_end, col_start:col_end].copy()
            inner_row_start = overlap // 2 if row_start > 0 else 0
            inner_col_start = overlap // 2 if col_start > 0 else 0
            inner_row_end = tile_data.shape[0] - (overlap // 2 if row_end < ny else 0)
            inner_col_end = tile_data.shape[1] - (overlap // 2 if col_end < nx else 0)
            tiles.append({
                'data': tile_data,
                'row_start': row_start,
                'col_start': col_start,
                'row_end': row_end,
                'col_end': col_end,
                'inner_row_start': inner_row_start,
                'inner_col_start': inner_col_start,
                'inner_row_end': inner_row_end,
                'inner_col_end': inner_col_end,
            })
    return tiles


def merge_tiles_with_blend(tiles: List[Dict[str, Any]],
                           target_shape: Tuple[int, int],
                           overlap: int = 32,
                           blend_sigma: float = 8.0) -> np.ndarray:
    """
    将优化后的 tile 块拼合并进行边界融合

    使用高斯加权融合处理重叠区域，避免拼接缝隙。

    Args:
        tiles: tile 信息列表（与 split_tiles 输出格式一致，data 字段已被更新）
        target_shape: 原始掩模尺寸 (height, width)
        overlap: 重叠区域大小（像素）
        blend_sigma: 融合权重的高斯 sigma

    Returns:
        拼合后的掩模 (2D numpy数组)
    """
    ny, nx = target_shape
    result = np.zeros((ny, nx), dtype=np.float64)
    weight = np.zeros((ny, nx), dtype=np.float64)
    for t in tiles:
        tile_data = t['data']
        rs = t['row_start']
        cs = t['col_start']
        re = t['row_end']
        ce = t['col_end']
        th, tw = tile_data.shape
        w_tile = np.ones((th, tw), dtype=np.float64)
        if blend_sigma > 0 and overlap > 0:
            half = overlap // 2
            if rs > 0:
                ramp = np.linspace(0, 1, min(half, th)).reshape(-1, 1)
                w_tile[:ramp.shape[0], :] *= ramp
            if re < ny:
                ramp = np.linspace(1, 0, min(half, th)).reshape(-1, 1)
                w_tile[-ramp.shape[0]:, :] *= ramp
            if cs > 0:
                ramp = np.linspace(0, 1, min(half, tw)).reshape(1, -1)
                w_tile[:, :ramp.shape[1]] *= ramp
            if ce < nx:
                ramp = np.linspace(1, 0, min(half, tw)).reshape(1, -1)
                w_tile[:, -ramp.shape[1]:] *= ramp
        result[rs:re, cs:ce] += w_tile * tile_data
        weight[rs:re, cs:ce] += w_tile
    valid = weight > 1e-12
    result[valid] /= weight[valid]
    return np.clip(result, 0.0, 1.0).astype(np.float64)
