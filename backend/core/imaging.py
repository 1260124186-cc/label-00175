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
from typing import Tuple, Optional, Dict, Any, List, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from core.fft import (
    WindowType,
    create_window,
    apply_zero_padding,
    remove_padding,
)
from core.array_backend import get_backend, DeviceType
from core.polarization import (
    JonesVector,
    ThinFilmStack,
    VectorPupil,
    compute_polarized_pupil,
    compute_partial_coherent_vectorial,
    create_high_na_immersion_system,
    create_euv_reflective_system,
)


def _use_gpu() -> bool:
    """检查当前是否使用 GPU 后端"""
    return get_backend().device == DeviceType.CUDA


def _asarray(arr):
    """确保数组为当前后端的数组类型"""
    backend = get_backend()
    if isinstance(arr, np.ndarray) and _use_gpu():
        return backend.from_numpy(arr)
    return arr


def _tonumpy(arr):
    """确保返回 numpy 数组（用于对外 API 兼容）"""
    backend = get_backend()
    if _use_gpu():
        return backend.to_numpy(arr)
    return np.asarray(arr)


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


class TCCMode(Enum):
    """TCC 计算模式类型

    - FULL_TCC: 完整 TCC 矩阵（光源积分法），精度最高，复杂度 O(M·N² log N)
    - SOCS: SOCS (Sum of Coherent Systems) 低秩近似，复杂度 O(K·N² log N)
    - KERNEL_2D: 二维 TCC 核对角近似，最快，复杂度 O(N² log N)，精度最低
    """
    FULL_TCC = "full_tcc"
    SOCS = "socs"
    KERNEL_2D = "kernel_2d"


class TechnologyNode(Enum):
    """光刻技术节点类型

    - DUV_ARF: ArF 深紫外光刻 (193nm)，适用于 130nm ~ 7nm 技术节点
    - EUV: 极紫外光刻 (13.5nm)，适用于 7nm 及以下技术节点
    """
    DUV_ARF = "duv_arf"
    EUV = "euv"


class ShadowingEffectModel(Enum):
    """EUV 阴影效应近似模型

    - NONE: 不考虑阴影效应
    - APPROXIMATE: 近似几何阴影模型，计算速度快
    - RIGOROUS: 严格电磁阴影模型，精度高但计算量大
    """
    NONE = "none"
    APPROXIMATE = "approximate"
    RIGOROUS = "rigorous"


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
        flare: Flare 系数 (0~1)，EUV 系统中杂散光的比例
        shadowing_model: 阴影效应近似模型（EUV 特有）
        reflective_mask_attenuation: 反射式掩模衰减因子 (0~1)，EUV 特有
        technology_node: 技术节点类型
        zernike_coefficients: Zernike 像差系数（单位: 波长λ）
        use_vector_pupil: 是否使用矢量光瞳模型（考虑偏振效应）
        incident_polarization_angle: 入射偏振方向（度），仅线偏振时有效
        n_immersion: 浸没介质折射率，1.0为干式，1.437为ArF水浸没
        use_mask_coating: 是否启用掩模涂层（多层膜/抗反射膜）效应
        name: 工艺条件名称，用于日志和结果标识
        weight: 该工艺条件在优化中的权重
    """
    defocus: float = 0.0
    dose: float = 1.0
    na: float = 1.35
    sigma: float = 0.75
    wavelength: float = 193.0
    flare: float = 0.0
    shadowing_model: ShadowingEffectModel = ShadowingEffectModel.NONE
    reflective_mask_attenuation: float = 0.0
    technology_node: TechnologyNode = TechnologyNode.DUV_ARF
    zernike_coefficients: Dict[int, float] = field(default_factory=dict)
    use_vector_pupil: bool = False
    incident_polarization_angle: float = 0.0
    n_immersion: float = 1.0
    use_mask_coating: bool = False
    name: str = ""
    weight: float = 1.0

    def __post_init__(self):
        if not self.name:
            tech_str = f"_tech={self.technology_node.value}" if self.technology_node != TechnologyNode.DUV_ARF else ""
            self.name = f"df={self.defocus:.0f}nm_dose={self.dose:.2f}_NA={self.na:.2f}_σ={self.sigma:.2f}{tech_str}"

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
            flare=optics.flare,
            shadowing_model=optics.shadowing_model,
            reflective_mask_attenuation=optics.reflective_mask_attenuation,
            technology_node=optics.technology_node,
            zernike_coefficients=dict(optics.zernike_coefficients),
            use_vector_pupil=optics.use_vector_pupil,
            incident_polarization_angle=optics.incident_polarization_angle,
            n_immersion=optics.n_immersion,
            use_mask_coating=optics.use_mask_coating,
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
            'flare': self.flare,
            'shadowing_model': self.shadowing_model.value,
            'reflective_mask_attenuation': self.reflective_mask_attenuation,
            'technology_node': self.technology_node.value,
            'use_vector_pupil': self.use_vector_pupil,
            'incident_polarization_angle': self.incident_polarization_angle,
            'n_immersion': self.n_immersion,
            'use_mask_coating': self.use_mask_coating,
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
                tcc_mode=base_optics.tcc_mode,
                socs_num_terms=base_optics.socs_num_terms,
                custom_source=base_optics.custom_source,
                flare=self.flare,
                shadowing_model=self.shadowing_model,
                reflective_mask_attenuation=self.reflective_mask_attenuation,
                technology_node=self.technology_node,
                zernike_coefficients=merged_zernike,
                use_vector_pupil=self.use_vector_pupil,
                incident_polarization_angle=self.incident_polarization_angle,
                n_immersion=self.n_immersion,
                use_mask_coating=self.use_mask_coating
            )
        else:
            return OpticalSystem(
                wavelength=self.wavelength,
                na=self.na,
                sigma=self.sigma,
                defocus=self.defocus,
                flare=self.flare,
                shadowing_model=self.shadowing_model,
                reflective_mask_attenuation=self.reflective_mask_attenuation,
                technology_node=self.technology_node,
                zernike_coefficients=self.zernike_coefficients,
                use_vector_pupil=self.use_vector_pupil,
                incident_polarization_angle=self.incident_polarization_angle,
                n_immersion=self.n_immersion,
                use_mask_coating=self.use_mask_coating
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
        flare_values: Flare 系数扫描值 (0~1)
        shadowing_model_values: 阴影效应模型扫描值
        reflective_mask_attenuation_values: 反射式掩模衰减因子扫描值
        technology_node_values: 技术节点扫描值
        use_vector_pupil_values: 是否使用矢量光瞳模型扫描值
        incident_polarization_angle_values: 入射偏振角度扫描值（度）
        n_immersion_values: 浸没介质折射率扫描值
        use_mask_coating_values: 是否使用掩模涂层效应扫描值
        default_weight: 默认权重
    """
    defocus_values: Any = 0.0
    dose_values: Any = 1.0
    na_values: Any = 1.35
    sigma_values: Any = 0.75
    wavelength_values: Any = 193.0
    flare_values: Any = 0.0
    shadowing_model_values: Any = ShadowingEffectModel.NONE
    reflective_mask_attenuation_values: Any = 0.0
    technology_node_values: Any = TechnologyNode.DUV_ARF
    use_vector_pupil_values: Any = False
    incident_polarization_angle_values: Any = 0.0
    n_immersion_values: Any = 1.0
    use_mask_coating_values: Any = False
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

    @staticmethod
    def _normalize_enum_values(values: Any, enum_cls: Type[Enum]) -> List[Enum]:
        """规范化枚举类型扫描值输入为列表"""
        if isinstance(values, list):
            result = []
            for v in values:
                if isinstance(v, enum_cls):
                    result.append(v)
                elif isinstance(v, str):
                    result.append(enum_cls(v))
                else:
                    result.append(enum_cls(str(v)))
            return result
        elif isinstance(values, enum_cls):
            return [values]
        elif isinstance(values, str):
            return [enum_cls(values)]
        else:
            return [enum_cls(str(values))]

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
        flare_list = self._normalize_scan_values(self.flare_values)
        shadowing_list = self._normalize_enum_values(self.shadowing_model_values, ShadowingEffectModel)
        attenuation_list = self._normalize_scan_values(self.reflective_mask_attenuation_values)
        tech_node_list = self._normalize_enum_values(self.technology_node_values, TechnologyNode)
        use_vec_pupil_list = self._normalize_scan_values(self.use_vector_pupil_values)
        pol_angle_list = self._normalize_scan_values(self.incident_polarization_angle_values)
        n_immersion_list = self._normalize_scan_values(self.n_immersion_values)
        use_mask_coating_list = self._normalize_scan_values(self.use_mask_coating_values)

        all_combos = list(product(
            defocus_list, dose_list, na_list, sigma_list, wavelength_list,
            flare_list, shadowing_list, attenuation_list, tech_node_list,
            use_vec_pupil_list, pol_angle_list, n_immersion_list, use_mask_coating_list
        ))

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
            flare_center = (min(flare_list) + max(flare_list)) / 2
            atten_center = (min(attenuation_list) + max(attenuation_list)) / 2
            pol_angle_center = (min(pol_angle_list) + max(pol_angle_list)) / 2
            n_immersion_center = (min(n_immersion_list) + max(n_immersion_list)) / 2

            distances = []
            for df, d, na, sg, wl, fl, sh, at, tn, uvp, pa, ni, umc in all_combos:
                dist = np.sqrt(
                    ((df - df_center) / (max(defocus_list) - min(defocus_list) + 1e-12))**2 +
                    ((d - dose_center) / (max(dose_list) - min(dose_list) + 1e-12))**2 +
                    ((na - na_center) / (max(na_list) - min(na_list) + 1e-12))**2 +
                    ((sg - sigma_center) / (max(sigma_list) - min(sigma_list) + 1e-12))**2 +
                    ((wl - wl_center) / (max(wavelength_list) - min(wavelength_list) + 1e-12))**2 +
                    ((fl - flare_center) / (max(flare_list) - min(flare_list) + 1e-12))**2 +
                    ((at - atten_center) / (max(attenuation_list) - min(attenuation_list) + 1e-12))**2 +
                    ((pa - pol_angle_center) / (max(pol_angle_list) - min(pol_angle_list) + 1e-12))**2 +
                    ((ni - n_immersion_center) / (max(n_immersion_list) - min(n_immersion_list) + 1e-12))**2
                )
                distances.append(dist)

            if distances:
                min_dist_idx = np.argmin(distances)
                weight_list[min_dist_idx] *= float(center_weight_boost)

        conditions = []
        for idx, (df, d, na, sg, wl, fl, sh, at, tn, uvp, pa, ni, umc) in enumerate(all_combos):
            w = weight_list[idx]
            tech_str = f"_tech={tn.value}" if tn != TechnologyNode.DUV_ARF else ""
            vec_str = f"_vec={uvp}" if uvp else ""
            coat_str = f"_coat={umc}" if umc else ""
            cond = ProcessCondition(
                defocus=df,
                dose=d,
                na=na,
                sigma=sg,
                wavelength=wl,
                flare=fl,
                shadowing_model=sh,
                reflective_mask_attenuation=at,
                technology_node=tn,
                use_vector_pupil=bool(uvp),
                incident_polarization_angle=pa,
                n_immersion=ni,
                use_mask_coating=bool(umc),
                weight=w,
                name=f"cond_{idx:03d}_df={df:.0f}_dose={d:.2f}_NA={na:.2f}_σ={sg:.2f}{tech_str}{vec_str}{coat_str}"
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
        tcc_mode: TCC 计算模式 (full_tcc / socs / kernel_2d)
        socs_num_terms: SOCS 分解项数（仅 SOCS 模式生效）
        use_socs: [已弃用] 是否使用 SOCS 低秩分解近似，向后兼容
        custom_source: 自定义光源分布（当illumination_type=CUSTOM时使用）
        technology_node: 技术节点类型 (DUV_ARF / EUV)
        flare: Flare 系数 (0~1)，EUV 系统中杂散光的比例
        shadowing_model: 阴影效应近似模型（EUV 特有）
        reflective_mask_attenuation: 反射式掩模衰减因子 (0~1)，EUV 特有
        use_vector_pupil: 是否使用矢量光瞳模型（考虑偏振效应）
        incident_polarization_angle: 入射偏振方向（度），仅线偏振时有效
        n_immersion: 浸没介质折射率，1.0为干式，1.437为ArF水浸没
        use_mask_coating: 是否启用掩模涂层（多层膜/抗反射膜）效应
        vector_pupil: 矢量光瞳实例（内部使用，由参数自动构建）
        mask_stack: 薄膜堆栈实例（内部使用，由参数自动构建）
        incident_polarization: 入射偏振态（内部使用，由参数自动构建）
    """
    wavelength: float = 193.0  # ArF光源波长
    na: float = 1.35  # 高NA浸没式光刻
    sigma: float = 0.75  # 部分相干因子
    pixel_size: float = 1.0  # 像素尺寸
    defocus: float = 0.0  # 离焦量
    magnification: float = 4.0  # 放大倍率
    illumination_type: IlluminationType = IlluminationType.CONVENTIONAL
    source_params: Dict[str, float] = field(default_factory=dict)
    tcc_mode: Optional[TCCMode] = None
    socs_num_terms: int = 5
    use_socs: Optional[bool] = None
    custom_source: Optional[np.ndarray] = None
    technology_node: TechnologyNode = TechnologyNode.DUV_ARF
    flare: float = 0.0  # Flare 系数
    shadowing_model: ShadowingEffectModel = ShadowingEffectModel.NONE
    reflective_mask_attenuation: float = 0.0  # 反射式掩模衰减因子
    zernike_coefficients: Dict[int, float] = field(default_factory=dict)
    use_vector_pupil: bool = False
    incident_polarization_angle: float = 0.0
    n_immersion: float = 1.0
    use_mask_coating: bool = False
    vector_pupil: Optional[VectorPupil] = None
    mask_stack: Optional[ThinFilmStack] = None
    incident_polarization: Optional[JonesVector] = None

    def __post_init__(self):
        """初始化后处理：设置默认光源参数，处理 use_socs 向后兼容

        tcc_mode 优先级高于 use_socs：
        - 若显式指定 tcc_mode，直接使用
        - 否则若指定 use_socs，从 use_socs 推导
        - 否则默认为 SOCS 模式

        技术节点自动配置：
        - EUV 节点自动设置波长 13.5nm、NA 0.33、典型参数
        - DUV ArF 节点保持默认 193nm、NA 1.35

        偏振与薄膜效应自动配置：
        - 根据技术节点自动创建矢量光瞳和薄膜堆栈
        - 配置入射偏振态
        """
        if self.technology_node == TechnologyNode.EUV:
            if self.wavelength == 193.0:
                self.wavelength = 13.5
            if self.na == 1.35:
                self.na = 0.33
            if self.magnification == 4.0:
                self.magnification = 4.0
            if self.n_immersion == 1.0 and self.use_vector_pupil:
                self.n_immersion = 1.0
        if self.technology_node == TechnologyNode.DUV_ARF and self.na > 1.0:
            if self.n_immersion == 1.0 and self.use_vector_pupil:
                self.n_immersion = 1.437
        if not self.source_params:
            self._set_default_source_params()
        if self.tcc_mode is None:
            if self.use_socs is not None:
                if self.use_socs:
                    self.tcc_mode = TCCMode.SOCS
                else:
                    self.tcc_mode = TCCMode.FULL_TCC
            else:
                self.tcc_mode = TCCMode.SOCS
        self._setup_polarization_and_coating()

    def _setup_polarization_and_coating(self):
        """配置偏振态和薄膜涂层效应"""
        if self.incident_polarization is None:
            self.incident_polarization = JonesVector.linear_polarization(
                self.incident_polarization_angle
            )
        if self.use_mask_coating and self.mask_stack is None:
            if self.technology_node == TechnologyNode.EUV:
                self.mask_stack = ThinFilmStack.euv_multilayer(
                    num_pairs=40, wavelength_nm=self.wavelength
                )
            else:
                self.mask_stack = ThinFilmStack.arf_antireflective(
                    wavelength_nm=self.wavelength
                )
        if self.use_vector_pupil and self.vector_pupil is None:
            if self.technology_node == TechnologyNode.EUV:
                self.vector_pupil = create_euv_reflective_system(
                    wavelength_nm=self.wavelength,
                    na=self.na,
                    num_multilayer_pairs=40,
                    grid_size=(256, 256),
                    incident_polarization=self.incident_polarization,
                )
            else:
                self.vector_pupil = create_high_na_immersion_system(
                    wavelength_nm=self.wavelength,
                    na=self.na,
                    grid_size=(256, 256),
                    incident_polarization=self.incident_polarization,
                )

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

        tcc_mode_str = optics_config.get('tcc_mode', None)
        if tcc_mode_str is not None:
            try:
                tcc_mode = TCCMode(tcc_mode_str)
            except ValueError:
                tcc_mode = TCCMode.SOCS
        else:
            use_socs = optics_config.get('use_socs', None)
            if use_socs is not None:
                tcc_mode = TCCMode.SOCS if use_socs else TCCMode.FULL_TCC
            else:
                tcc_mode = TCCMode.SOCS

        tech_node_str = optics_config.get('technology_node', 'duv_arf')
        try:
            technology_node = TechnologyNode(tech_node_str)
        except ValueError:
            technology_node = TechnologyNode.DUV_ARF

        shadowing_str = optics_config.get('shadowing_model', 'none')
        try:
            shadowing_model = ShadowingEffectModel(shadowing_str)
        except ValueError:
            shadowing_model = ShadowingEffectModel.NONE

        source_params = optics_config.get('source_params', {})

        zernike_raw = optics_config.get('zernike_coefficients', {})
        zernike_coefficients = _parse_zernike_coefficients(zernike_raw)

        use_vector_pupil = optics_config.get('use_vector_pupil', False)
        incident_polarization_angle = optics_config.get('incident_polarization_angle', 0.0)
        n_immersion = optics_config.get('n_immersion', 1.0)
        use_mask_coating = optics_config.get('use_mask_coating', False)

        return cls(
            wavelength=optics_config.get('wavelength', 193.0),
            na=optics_config.get('na', 1.35),
            sigma=optics_config.get('sigma', 0.75),
            pixel_size=optics_config.get('pixel_size', 1.0),
            defocus=optics_config.get('defocus', 0.0),
            magnification=optics_config.get('magnification', 4.0),
            illumination_type=illumination_type,
            source_params=source_params,
            tcc_mode=tcc_mode,
            socs_num_terms=optics_config.get('socs_num_terms', 5),
            technology_node=technology_node,
            flare=optics_config.get('flare', 0.0),
            shadowing_model=shadowing_model,
            reflective_mask_attenuation=optics_config.get('reflective_mask_attenuation', 0.0),
            zernike_coefficients=zernike_coefficients,
            use_vector_pupil=use_vector_pupil,
            incident_polarization_angle=incident_polarization_angle,
            n_immersion=n_immersion,
            use_mask_coating=use_mask_coating
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
            'tcc_mode': self.tcc_mode.value,
            'socs_num_terms': self.socs_num_terms,
            'technology_node': self.technology_node.value,
            'flare': self.flare,
            'shadowing_model': self.shadowing_model.value,
            'reflective_mask_attenuation': self.reflective_mask_attenuation,
            'use_vector_pupil': self.use_vector_pupil,
            'incident_polarization_angle': self.incident_polarization_angle,
            'n_immersion': self.n_immersion,
            'use_mask_coating': self.use_mask_coating,
            'zernike_coefficients': zernike_out if zernike_out else {}
        }

    def parameter_sweep(
        self,
        defocus_values: Any = None,
        dose_values: Any = None,
        na_values: Any = None,
        sigma_values: Any = None,
        flare_values: Any = None,
        shadowing_model_values: Any = None,
        reflective_mask_attenuation_values: Any = None,
        technology_node_values: Any = None,
        use_vector_pupil_values: Any = None,
        incident_polarization_angle_values: Any = None,
        n_immersion_values: Any = None,
        use_mask_coating_values: Any = None,
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
            flare_values: Flare 系数扫描值，格式同上，None 则使用当前 flare
            shadowing_model_values: 阴影效应模型扫描值，None 则使用当前模型
            reflective_mask_attenuation_values: 反射式掩模衰减因子扫描值，格式同上
            technology_node_values: 技术节点扫描值，None 则使用当前节点
            use_vector_pupil_values: 是否使用矢量光瞳模型扫描值
            incident_polarization_angle_values: 入射偏振角度扫描值（度）
            n_immersion_values: 浸没介质折射率扫描值
            use_mask_coating_values: 是否使用掩模涂层效应扫描值
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
        flare_vals = ProcessWindow._normalize_scan_values(
            flare_values if flare_values is not None else self.flare
        )
        shadowing_vals = ProcessWindow._normalize_enum_values(
            shadowing_model_values if shadowing_model_values is not None else self.shadowing_model,
            ShadowingEffectModel
        )
        attenuation_vals = ProcessWindow._normalize_scan_values(
            reflective_mask_attenuation_values if reflective_mask_attenuation_values is not None
            else self.reflective_mask_attenuation
        )
        tech_node_vals = ProcessWindow._normalize_enum_values(
            technology_node_values if technology_node_values is not None else self.technology_node,
            TechnologyNode
        )
        use_vec_pupil_vals = ProcessWindow._normalize_scan_values(
            use_vector_pupil_values if use_vector_pupil_values is not None else self.use_vector_pupil
        )
        pol_angle_vals = ProcessWindow._normalize_scan_values(
            incident_polarization_angle_values if incident_polarization_angle_values is not None
            else self.incident_polarization_angle
        )
        n_immersion_vals = ProcessWindow._normalize_scan_values(
            n_immersion_values if n_immersion_values is not None else self.n_immersion
        )
        use_mask_coat_vals = ProcessWindow._normalize_scan_values(
            use_mask_coating_values if use_mask_coating_values is not None else self.use_mask_coating
        )

        pw = ProcessWindow(
            defocus_values=df_vals,
            dose_values=dose_vals,
            na_values=na_vals,
            sigma_values=sigma_vals,
            wavelength_values=self.wavelength,
            flare_values=flare_vals,
            shadowing_model_values=shadowing_vals,
            reflective_mask_attenuation_values=attenuation_vals,
            technology_node_values=tech_node_vals,
            use_vector_pupil_values=use_vec_pupil_vals,
            incident_polarization_angle_values=pol_angle_vals,
            n_immersion_values=n_immersion_vals,
            use_mask_coating_values=use_mask_coat_vals,
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
    backend = get_backend()
    fx_b = _asarray(fx)
    fy_b = _asarray(fy)

    ny, nx = fx_b.shape
    source = backend.zeros((ny, nx), dtype=backend.float64)

    if illumination_type == IlluminationType.CUSTOM and custom_source is not None:
        if custom_source.shape == (ny, nx):
            source = _asarray(custom_source).astype(backend.float64)
        else:
            raise ValueError(f"自定义光源形状 {custom_source.shape} 与频率网格形状 {(ny, nx)} 不匹配")
    else:
        sigma_inner = source_params.get('sigma_inner', 0.0)
        sigma_outer = source_params.get('sigma_outer', source_params.get('sigma', 0.75))

        rho = backend.sqrt(fx_b**2 + fy_b**2) / cutoff
        theta = backend.arctan2(fy_b, fx_b)

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
            angle_diff = backend.abs((theta - angle + backend.pi) % (2 * backend.pi) - backend.pi)
            angle_mask1 = angle_diff <= opening_angle / 2
            angle_mask2 = angle_diff >= (backend.pi - opening_angle / 2)

            mask = radial_mask & (angle_mask1 | angle_mask2)
            source[mask] = 1.0

        elif illumination_type == IlluminationType.QUASAR:
            angle = np.deg2rad(source_params.get('angle', 45.0))
            opening_angle = np.deg2rad(source_params.get('opening_angle', 30.0))

            radial_mask = (rho >= sigma_inner) & (rho <= sigma_outer)

            pole_angles = [angle, angle + np.pi/2, angle + np.pi, angle + 3*np.pi/2]
            angle_mask = backend.zeros_like(rho, dtype=bool)

            for pole_angle in pole_angles:
                angle_diff = backend.abs((theta - pole_angle + backend.pi) % (2 * backend.pi) - backend.pi)
                angle_mask = angle_mask | (angle_diff <= opening_angle / 2)

            mask = radial_mask & angle_mask
            source[mask] = 1.0

    total = backend.sum(source)
    if total > 0:
        source = source / total

    return _tonumpy(source)


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

    注意：Zernike 多项式计算在 CPU 上进行（仅初始化时执行一次），
         结果会自动转换为当前后端的数组类型。

    Args:
        fx: x方向频率网格
        fy: y方向频率网格
        cutoff: 截止频率
        zernike_coefficients: Zernike 系数字典 {j: coefficient}，
                              j 为 Noll 索引(0-based)，coefficient 单位为波长 λ

    Returns:
        像差相位数组（弧度），形状与 fx 相同
    """
    fx_np = np.asarray(fx)
    fy_np = np.asarray(fy)

    ny, nx = fx_np.shape
    phase = np.zeros((ny, nx), dtype=np.float64)

    if not zernike_coefficients:
        return phase

    rho = np.sqrt(fx_np ** 2 + fy_np ** 2) / cutoff
    theta = np.arctan2(fy_np, fx_np)

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
    backend = get_backend()
    fx_b = _asarray(fx)
    fy_b = _asarray(fy)
    zernike_phase_b = _asarray(zernike_phase)

    ny, nx = fx_b.shape
    pupil = backend.zeros((ny, nx), dtype=backend.complex128)

    rho_sq = (fx_b ** 2 + fy_b ** 2) / (cutoff ** 2)
    pupil_mask = rho_sq <= 1.0

    defocus_phase = backend.pi * defocus / wavelength * rho_sq

    total_phase = defocus_phase + zernike_phase_b

    pupil[pupil_mask] = backend.exp(1j * total_phase[pupil_mask])

    return _tonumpy(pupil)


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


def _shift_pupil(pupil, shift_fx: float,
                 shift_fy: float, dfx: float, dfy: float):
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
    backend = get_backend()
    ny, nx = pupil.shape

    shift_x = int(round(shift_fx / dfx))
    shift_y = int(round(shift_fy / dfy))

    if shift_x == 0 and shift_y == 0:
        return backend.copy(pupil)

    shifted = backend.roll(pupil, shift=shift_y, axis=0)
    shifted = backend.roll(shifted, shift=shift_x, axis=1)

    return shifted


def compute_tcc_full(fx: np.ndarray, fy: np.ndarray,
                     pupil: np.ndarray, source: np.ndarray,
                     cutoff: float, dfx: float, dfy: float) -> np.ndarray:
    """
    基于光源积分计算完整的 TCC 矩阵 (四维)

    TCC(f1, f2) = ∫ S(fs) * P(f1 - fs) * P*(f2 - fs) dfs

    注意：完整 TCC 矩阵计算量很大，建议仅用于小型系统验证。
         对于大规模计算，请使用 SOCS 或 KERNEL_2D 模式。

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
    backend = get_backend()
    fx_b = _asarray(fx)
    fy_b = _asarray(fy)
    pupil_b = _asarray(pupil)
    source_b = _asarray(source)

    ny, nx = pupil_b.shape
    tcc = backend.zeros((ny, nx, ny, nx), dtype=backend.complex128)

    source_indices = backend.where_idx(source_b > 1e-10)
    source_values = source_b[source_indices]

    for idx in range(len(source_indices[0])):
        sy, sx = int(source_indices[0][idx]), int(source_indices[1][idx])
        src_val = source_values[idx]

        fs_x = fx_b[sy, sx]
        fs_y = fy_b[sy, sx]

        if backend.sqrt(fs_x**2 + fs_y**2) > cutoff:
            continue

        pupil_shifted = _shift_pupil(pupil_b, fs_x, fs_y, dfx, dfy)
        pupil_conj_shifted = backend.conj(pupil_shifted)

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

    return _tonumpy(tcc)


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
    backend = get_backend()
    fx_b = _asarray(fx)
    fy_b = _asarray(fy)
    pupil_b = _asarray(pupil)
    source_b = _asarray(source)

    ny, nx = pupil_b.shape
    tcc_kernel = backend.zeros((ny, nx), dtype=backend.float64)

    source_indices = backend.where_idx(source_b > 1e-10)
    source_values = source_b[source_indices]

    for idx in range(len(source_indices[0])):
        sy, sx = int(source_indices[0][idx]), int(source_indices[1][idx])
        src_val = source_values[idx]

        fs_x = fx_b[sy, sx]
        fs_y = fy_b[sy, sx]

        if backend.sqrt(fs_x**2 + fs_y**2) > cutoff:
            continue

        pupil_shifted = _shift_pupil(pupil_b, fs_x, fs_y, dfx, dfy)
        tcc_kernel = tcc_kernel + src_val * backend.abs(pupil_shifted)**2

    total = backend.sum(tcc_kernel)
    if total > 0:
        tcc_kernel = tcc_kernel / total

    return _tonumpy(tcc_kernel)


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
    backend = get_backend()
    fx_b = _asarray(fx)
    fy_b = _asarray(fy)
    pupil_b = _asarray(pupil)
    source_b = _asarray(source)

    ny, nx = pupil_b.shape
    N = ny * nx

    source_indices = backend.where_idx(source_b > 1e-10)
    source_values = source_b[source_indices]

    M = len(source_indices[0])
    V = backend.zeros((M, N), dtype=backend.complex128)

    for idx in range(M):
        sy, sx = int(source_indices[0][idx]), int(source_indices[1][idx])
        src_val = backend.sqrt(source_values[idx])

        fs_x = fx_b[sy, sx]
        fs_y = fy_b[sy, sx]

        pupil_shifted = _shift_pupil(pupil_b, fs_x, fs_y, dfx, dfy)
        V[idx, :] = src_val * pupil_shifted.flatten()

    if M <= N:
        VVh = V @ V.conj().T
        eigenvalues, eigenvectors = backend.eigh(VVh)

        idx_sorted = backend.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sorted]
        eigenvectors = eigenvectors[:, idx_sorted]

        num_terms = min(num_terms, M)
        eigenvalues = eigenvalues[:num_terms]
        eigenvectors = eigenvectors[:, :num_terms]

        eigenfunctions = backend.zeros((num_terms, ny, nx), dtype=backend.complex128)
        for i in range(num_terms):
            phi_flat = V.conj().T @ eigenvectors[:, i]
            norm = backend.sqrt(backend.sum(backend.abs(phi_flat)**2))
            if norm > 1e-10:
                phi_flat = phi_flat / norm
            eigenfunctions[i, :, :] = phi_flat.reshape(ny, nx)
    else:
        VhV = V.conj().T @ V
        eigenvalues, eigenvectors = backend.eigh(VhV)

        idx_sorted = backend.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sorted]
        eigenvectors = eigenvectors[:, idx_sorted]

        num_terms = min(num_terms, N)
        eigenvalues = eigenvalues[:num_terms]
        eigenvectors = eigenvectors[:, :num_terms]

        eigenfunctions = backend.zeros((num_terms, ny, nx), dtype=backend.complex128)
        for i in range(num_terms):
            eigenfunctions[i, :, :] = eigenvectors[:, i].reshape(ny, nx)

    eigenvalues = backend.real(eigenvalues)
    total_energy = backend.sum(eigenvalues)
    if total_energy > 0:
        eigenvalues = eigenvalues / total_energy

    return _tonumpy(eigenvalues), _tonumpy(eigenfunctions)


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

    支持在成像前对掩模做零填充（zero-padding）与加窗（windowing），
    缓解 FFT 边界效应引起的频谱伪影。

    Attributes:
        optics: 光学系统参数
        image_size: 原始掩模尺寸 (height, width)
        window_type: 窗函数类型，None 表示不加窗
        pad_width: 零填充宽度（像素），None 表示不填充；
                   整数表示各方向均匀填充，元组 (py, px) 表示各方向不同
        tukey_alpha: Tukey 窗的渐变比例因子 (0~1)，仅 window_type=Tukey 时生效
    """

    def __init__(self, optical_system: OpticalSystem,
                 image_size: Tuple[int, int],
                 window_type: Optional[Union[WindowType, str]] = None,
                 pad_width: Optional[Union[int, Tuple[int, int]]] = None,
                 tukey_alpha: float = 0.5):
        """
        初始化部分相干成像模型

        Args:
            optical_system: 光学系统参数
            image_size: 图像尺寸 (height, width)
            window_type: 窗函数类型 ('hann', 'hamming', 'tukey')，None 不加窗
            pad_width: 零填充宽度，None 不填充
            tukey_alpha: Tukey 窗渐变比例因子
        """
        self.optics = optical_system
        self.image_size = image_size
        self.window_type = self._normalize_window_type(window_type)
        self.pad_width = pad_width
        self.tukey_alpha = tukey_alpha

        self._effective_size = self._compute_effective_size()
        self._window_2d = self._create_window_for_original()

        self._setup_frequency_grid()
        self._compute_source_and_pupil()
        self._compute_transfer_functions()

    @staticmethod
    def _normalize_window_type(window_type: Optional[Union[WindowType, str]]) -> Optional[WindowType]:
        if window_type is None:
            return None
        if isinstance(window_type, str):
            return WindowType(window_type)
        return window_type

    def _compute_effective_size(self) -> Tuple[int, int]:
        ny, nx = self.image_size
        if self.pad_width is not None:
            if isinstance(self.pad_width, int):
                py, px = self.pad_width, self.pad_width
            else:
                py, px = self.pad_width
            return (ny + 2 * py, nx + 2 * px)
        return (ny, nx)

    def _create_window_for_original(self) -> Optional[np.ndarray]:
        if self.window_type is None:
            return None
        return create_window(self.image_size, self.window_type, self.tukey_alpha)

    def _apply_reflective_mask_attenuation(self, mask: np.ndarray) -> np.ndarray:
        """
        应用反射式掩模衰减效应（EUV 特有）。

        DUV 为透射式掩模，掩模值 0/1 直接对应不透明/透明。
        EUV 为反射式掩模，吸收层并非完全吸收，存在反射衰减。
        将 [0, 1] 线性映射到 [attenuation, 1]。

        Args:
            mask: 原始掩模 [0, 1]

        Returns:
            应用衰减后的有效掩模
        """
        atten = self.optics.reflective_mask_attenuation
        if atten <= 0.0:
            return mask
        return atten + (1.0 - atten) * mask

    def _apply_shadowing_effect(self, mask: np.ndarray) -> np.ndarray:
        """
        应用阴影效应近似模型（EUV 反射式掩模特有）。

        EUV 采用掠入射照明，掩模吸收层侧壁对衍射光产生几何遮挡。
        近似模型：
        - NONE: 不处理
        - APPROXIMATE: 沿入射方向（y轴）做不对称一维卷积，
          模拟边缘有效宽度的非对称偏移
        - RIGOROUS: 进一步加强的近似模型（更宽的非对称核）

        Args:
            mask: 输入掩模

        Returns:
            应用阴影效应后的掩模
        """
        model = self.optics.shadowing_model
        if model == ShadowingEffectModel.NONE:
            return mask

        pixel_size_nm = self.optics.pixel_size
        shadow_width_nm = 2.0 if model == ShadowingEffectModel.APPROXIMATE else 4.0
        k = max(1, int(round(shadow_width_nm / pixel_size_nm)))

        if k <= 1:
            return mask

        kernel = np.zeros((2 * k + 1,), dtype=np.float64)
        kernel[k:] = 1.0
        kernel = kernel / kernel.sum()

        padded = np.pad(mask, ((0, 0), (k, k)), mode='edge')
        shadowed = np.zeros_like(mask)
        ny, nx = mask.shape
        for y in range(ny):
            for x in range(nx):
                shadowed[y, x] = np.sum(padded[y, x:x + 2 * k + 1] * kernel)

        return shadowed

    def _apply_flare(self, intensity: np.ndarray) -> np.ndarray:
        """
        应用 Flare（杂散光）效应。

        Flare 来自光学系统散射，表现为均匀背景叠加：
            I_final = (1 - flare) * I_ideal + flare * mean(I_ideal)

        Args:
            intensity: 理想空间像光强

        Returns:
            叠加 flare 后的光强
        """
        flare = self.optics.flare
        if flare <= 0.0:
            return intensity
        bg = float(np.mean(intensity))
        return (1.0 - flare) * intensity + flare * bg

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        processed = mask.astype(np.float64)
        processed = self._apply_reflective_mask_attenuation(processed)
        processed = self._apply_shadowing_effect(processed)
        if self._window_2d is not None:
            processed = processed * self._window_2d
        if self.pad_width is not None:
            processed, _ = apply_zero_padding(processed, self.pad_width)
        return processed

    def _setup_frequency_grid(self):
        """设置频率网格"""
        backend = get_backend()
        ny, nx = self._effective_size

        self.dfx = 1.0 / (nx * self.optics.pixel_size)
        self.dfy = 1.0 / (ny * self.optics.pixel_size)

        fx = backend.fftfreq(nx, self.optics.pixel_size)
        fy = backend.fftfreq(ny, self.optics.pixel_size)
        fx_grid, fy_grid = backend.meshgrid(fx, fy)
        self.fx = _tonumpy(fx_grid)
        self.fy = _tonumpy(fy_grid)

    def _compute_source_and_pupil(self):
        """计算光源分布和光瞳函数（含离焦和Zernike像差）

        支持两种光瞳计算模式：
        - 标量模式（默认）：使用传统标量光瞳函数，不考虑偏振效应
        - 矢量模式：考虑偏振效应和薄膜涂层的矢量光瞳函数
        """
        cutoff = self.optics.cutoff_frequency

        zernike_phase = compute_zernike_phase(
            self.fx, self.fy, cutoff,
            self.optics.zernike_coefficients
        )

        self.vector_pupil_dict = None

        if self.optics.use_vector_pupil:
            is_reflection = self.optics.technology_node == TechnologyNode.EUV
            polarized_result = compute_polarized_pupil(
                fx=self.fx,
                fy=self.fy,
                wavelength_nm=self.optics.wavelength,
                na=self.optics.na,
                cutoff=cutoff,
                defocus_nm=self.optics.defocus,
                zernike_phase=zernike_phase,
                incident_polarization=self.optics.incident_polarization,
                n_immersion=complex(self.optics.n_immersion),
                mask_stack=self.optics.mask_stack,
                is_reflection=is_reflection,
            )
            self.pupil = polarized_result['pupil_scalar']
            self.vector_pupil_dict = polarized_result
        else:
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
        """计算传递函数（根据 tcc_mode 选择 FULL_TCC / SOCS / KERNEL_2D）"""
        cutoff = self.optics.cutoff_frequency
        mode = self.optics.tcc_mode

        self.tcc_kernel = compute_tcc_kernel_2d(
            self.fx, self.fy,
            self.pupil, self.source,
            cutoff, self.dfx, self.dfy
        )
        self.tcc_full = None

        if mode == TCCMode.SOCS:
            self.socs_eigenvalues, self.socs_eigenfunctions = socs_decomposition(
                self.fx, self.fy,
                self.pupil, self.source,
                cutoff, self.dfx, self.dfy,
                self.optics.socs_num_terms
            )
        else:
            self.socs_eigenvalues = None
            self.socs_eigenfunctions = None

    @property
    def tcc(self) -> Optional[np.ndarray]:
        """向后兼容：返回 2D TCC 核对角近似"""
        return self.tcc_kernel

    def compute_aerial_image(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像（晶圆上的光强分布）

        支持两种成像模式：
        - 标量模式（默认）：使用传统标量光瞳函数，支持三种 TCC 计算模式
            * FULL_TCC: 光源积分法（Hopkins公式），精度最高，复杂度 O(M·N² log N)
            * SOCS: SOCS 低秩近似，复杂度 O(K·N² log N)
            * KERNEL_2D: 二维 TCC 核对角近似，最快，复杂度 O(N² log N)
        - 矢量模式：考虑偏振效应和薄膜涂层的矢量成像模型，使用 Abbe 方法

        如果初始化时指定了窗函数和/或零填充，会在FFT前对掩模
        做加窗和零填充处理，计算完成后裁剪回原始尺寸。

        支持 CPU/GPU 透明切换，通过 ArrayBackend 统一调度。

        Args:
            mask: 掩模图案 (2D numpy数组, 0-1值)

        Returns:
            空间像光强分布（原始掩模尺寸）
        """
        backend = get_backend()
        processed_mask = self._preprocess_mask(mask)

        if self.optics.use_vector_pupil and self.vector_pupil_dict is not None:
            intensity = compute_partial_coherent_vectorial(
                mask=processed_mask,
                source=self.source,
                vector_pupils=self.vector_pupil_dict,
                dfx=self.dfx,
                dfy=self.dfy,
            )
            intensity_np = intensity
        else:
            mask_c = _asarray(processed_mask).astype(backend.complex128)
            ny_eff, nx_eff = processed_mask.shape
            mode = self.optics.tcc_mode

            if mode == TCCMode.SOCS and self.socs_eigenvalues is not None:
                intensity = backend.zeros((ny_eff, nx_eff), dtype=backend.float64)
                mask_spectrum = backend.fft2(mask_c)
                eigenvalues = _asarray(self.socs_eigenvalues)
                eigenfunctions = _asarray(self.socs_eigenfunctions)

                for i in range(len(eigenvalues)):
                    lam = eigenvalues[i]
                    if lam < 1e-10:
                        continue
                    phi = eigenfunctions[i]
                    filtered = mask_spectrum * phi
                    field_i = backend.ifft2(filtered)
                    intensity = intensity + lam * backend.abs(field_i)**2
            elif mode == TCCMode.KERNEL_2D and self.tcc_kernel is not None:
                mask_spectrum = backend.fft2(mask_c)
                tcc_kernel = _asarray(self.tcc_kernel)
                effective_pupil = backend.sqrt(backend.maximum(tcc_kernel, 0.0))
                filtered = mask_spectrum * effective_pupil
                field = backend.ifft2(filtered)
                intensity = backend.abs(field)**2
            else:
                intensity = backend.zeros((ny_eff, nx_eff), dtype=backend.float64)
                mask_spectrum = backend.fft2(mask_c)
                cutoff = self.optics.cutoff_frequency

                source = _asarray(self.source)
                fx = _asarray(self.fx)
                fy = _asarray(self.fy)
                pupil = _asarray(self.pupil)

                source_indices = backend.where_idx(source > 1e-10)
                source_values = source[source_indices]

                for idx in range(len(source_indices[0])):
                    sy, sx = int(source_indices[0][idx]), int(source_indices[1][idx])
                    src_val = source_values[idx]

                    fs_x = fx[sy, sx]
                    fs_y = fy[sy, sx]

                    if backend.sqrt(fs_x**2 + fs_y**2) > cutoff:
                        continue

                    pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, self.dfx, self.dfy)
                    filtered = mask_spectrum * pupil_shifted
                    field_i = backend.ifft2(filtered)
                    intensity = intensity + src_val * backend.abs(field_i)**2

            if self.pad_width is not None:
                if isinstance(self.pad_width, int):
                    py, px = self.pad_width, self.pad_width
                else:
                    py, px = self.pad_width
                intensity = intensity[py:py + self.image_size[0], px:px + self.image_size[1]]

            intensity_np = _tonumpy(intensity.astype(backend.float64))

        intensity_np = self._apply_flare(intensity_np)

        if intensity_np.max() > 0:
            intensity_np = intensity_np / intensity_np.max()

        return intensity_np

    def _apply_shadowing_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        阴影效应近似的梯度反向传播。

        阴影效应正向是沿 x 方向的非对称一维卷积，
        梯度反向传播使用转置（反向）卷积核。
        """
        model = self.optics.shadowing_model
        if model == ShadowingEffectModel.NONE:
            return grad

        pixel_size_nm = self.optics.pixel_size
        shadow_width_nm = 2.0 if model == ShadowingEffectModel.APPROXIMATE else 4.0
        k = max(1, int(round(shadow_width_nm / pixel_size_nm)))

        if k <= 1:
            return grad

        kernel = np.zeros((2 * k + 1,), dtype=np.float64)
        kernel[k:] = 1.0
        kernel = kernel / kernel.sum()
        kernel_rev = kernel[::-1]

        padded = np.pad(grad, ((0, 0), (k, k)), mode='edge')
        backprop = np.zeros_like(grad)
        ny, nx = grad.shape
        for y in range(ny):
            for x in range(nx):
                backprop[y, x] = np.sum(padded[y, x:x + 2 * k + 1] * kernel_rev)

        return backprop

    def _apply_attenuation_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        反射式掩模衰减的梯度传播。
        正向: m_eff = atten + (1 - atten) * m
        反向: dL/dm = dL/dm_eff * (1 - atten)
        """
        atten = self.optics.reflective_mask_attenuation
        if atten <= 0.0:
            return grad
        return grad * (1.0 - atten)

    def _apply_flare_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Flare 的梯度传播。
        正向: I_out = (1-flare) * I_in + flare * mean(I_in)
        反向: dL/dI_in = (1-flare) * dL/dI_out + flare * mean(dL/dI_out)
        """
        flare = self.optics.flare
        if flare <= 0.0:
            return grad
        mean_g = float(np.mean(grad))
        return (1.0 - flare) * grad + flare * mean_g

    def _preprocess_complex_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        预处理复数掩模（加窗、零填充）

        注意：反射式掩模衰减和阴影效应是幅度效应，
        对于复振幅掩模，应在外部将衰减和阴影效应
        合并到复振幅中。这里只做加窗和零填充。

        Args:
            mask: 复数掩模

        Returns:
            预处理后的复数掩模
        """
        processed = mask.astype(np.complex128)
        if self._window_2d is not None:
            processed = processed * self._window_2d.astype(np.complex128)
        if self.pad_width is not None:
            processed, _ = apply_zero_padding(processed, self.pad_width)
        return processed

    def compute_aerial_image_complex(self, mask_complex: np.ndarray) -> np.ndarray:
        """
        使用复振幅掩模计算空间像（晶圆上的光强分布）

        支持复数透过率掩模（如 PSM 相位掩模），掩模可以同时包含
        幅度和相位信息。

        数学原理：
            I(x,y) = Σ_i λ_i · | IFFT{ M(f) · φ_i(f) } |^2
            其中 M(f) = FFT{ t(x,y) } 是复掩模的频谱，
                  φ_i(f) 是第 i 个 SOCS 本征函数。

        Args:
            mask_complex: 复数掩模图案（复振幅透过率）

        Returns:
            空间像光强分布（原始掩模尺寸）
        """
        backend = get_backend()
        processed_mask = self._preprocess_complex_mask(mask_complex)

        mask_c = _asarray(processed_mask).astype(backend.complex128)
        ny_eff, nx_eff = processed_mask.shape
        mode = self.optics.tcc_mode

        if mode == TCCMode.SOCS and self.socs_eigenvalues is not None:
            intensity = backend.zeros((ny_eff, nx_eff), dtype=backend.float64)
            mask_spectrum = backend.fft2(mask_c)
            eigenvalues = _asarray(self.socs_eigenvalues)
            eigenfunctions = _asarray(self.socs_eigenfunctions)

            for i in range(len(eigenvalues)):
                lam = eigenvalues[i]
                if lam < 1e-10:
                    continue
                phi = eigenfunctions[i]
                filtered = mask_spectrum * phi
                field_i = backend.ifft2(filtered)
                intensity = intensity + lam * backend.abs(field_i)**2
        elif mode == TCCMode.KERNEL_2D and self.tcc_kernel is not None:
            mask_spectrum = backend.fft2(mask_c)
            tcc_kernel = _asarray(self.tcc_kernel)
            effective_pupil = backend.sqrt(backend.maximum(tcc_kernel, 0.0))
            filtered = mask_spectrum * effective_pupil
            field = backend.ifft2(filtered)
            intensity = backend.abs(field)**2
        else:
            intensity = backend.zeros((ny_eff, nx_eff), dtype=backend.float64)
            mask_spectrum = backend.fft2(mask_c)
            cutoff = self.optics.cutoff_frequency

            source = _asarray(self.source)
            fx = _asarray(self.fx)
            fy = _asarray(self.fy)
            pupil = _asarray(self.pupil)

            source_indices = backend.where_idx(source > 1e-10)
            source_values = source[source_indices]

            for idx in range(len(source_indices[0])):
                sy, sx = int(source_indices[0][idx]), int(source_indices[1][idx])
                src_val = source_values[idx]

                fs_x = fx[sy, sx]
                fs_y = fy[sy, sx]

                if backend.sqrt(fs_x**2 + fs_y**2) > cutoff:
                    continue

                pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, self.dfx, self.dfy)
                filtered = mask_spectrum * pupil_shifted
                field_i = backend.ifft2(filtered)
                intensity = intensity + src_val * backend.abs(field_i)**2

        if self.pad_width is not None:
            if isinstance(self.pad_width, int):
                py, px = self.pad_width, self.pad_width
            else:
                py, px = self.pad_width
            intensity = intensity[py:py + self.image_size[0], px:px + self.image_size[1]]

        intensity_np = _tonumpy(intensity.astype(backend.float64))
        intensity_np = self._apply_flare(intensity_np)

        if intensity_np.max() > 0:
            intensity_np = intensity_np / intensity_np.max()

        return intensity_np

    def compute_complex_gradient(
        self,
        mask_complex: np.ndarray,
        intensity_grad: np.ndarray,
        return_spectral: bool = False,
    ) -> np.ndarray:
        """
        计算损失对复振幅掩模的梯度

        使用"分量梯度"约定：返回的梯度 g = ∂L/∂a + i ∂L/∂b，
        其中 a = Re(t), b = Im(t) 分别是复透过率的实部和虚部。

        数学推导：
            光强 I = Σ_i λ_i · |E_i|^2
            其中 E_i = IFFT{ M(f) · φ_i(f) }

            像面电场梯度（分量约定）：
                g_Ei = ∂L/∂Re(E_i) + i ∂L/∂Im(E_i) = 2 · dL/dI · E_i

            频域梯度：
                G_Mi(f) = φ_i*(f) · G_Ei(f)

            空间域梯度（所有 SOCS 项求和）：
                g_t = Σ_i λ_i · IFFT{ G_Mi }

        Args:
            mask_complex: 复数掩模
            intensity_grad: 损失对光强的梯度 dL/dI
            return_spectral: 是否返回频域梯度（调试用）

        Returns:
            损失对复掩模的梯度（分量约定）。
            如果 return_spectral=True，返回 (空间域梯度, 频域梯度)
        """
        backend = get_backend()
        processed_mask = self._preprocess_complex_mask(mask_complex)

        mask_c = _asarray(processed_mask).astype(backend.complex128)
        ny_eff, nx_eff = processed_mask.shape
        mode = self.optics.tcc_mode
        mask_spectrum = backend.fft2(mask_c)

        dLdI = _asarray(intensity_grad).astype(backend.float64)
        if dLdI.shape != self.image_size:
            raise ValueError(
                f"intensity_grad 形状 {dLdI.shape} 与 image_size {self.image_size} 不匹配"
            )

        if self.pad_width is not None:
            if isinstance(self.pad_width, int):
                py, px = self.pad_width, self.pad_width
            else:
                py, px = self.pad_width
            dLdI_padded = backend.zeros((ny_eff, nx_eff), dtype=backend.float64)
            dLdI_padded[py:py + self.image_size[0], px:px + self.image_size[1]] = dLdI
            dLdI = dLdI_padded

        if self.optics.flare > 0.0:
            dLdI = _asarray(self._apply_flare_gradient(_tonumpy(dLdI)))

        grad_spectrum = backend.zeros((ny_eff, nx_eff), dtype=backend.complex128)

        if mode == TCCMode.SOCS and self.socs_eigenvalues is not None:
            eigenvalues = _asarray(self.socs_eigenvalues)
            eigenfunctions = _asarray(self.socs_eigenfunctions)

            for i in range(len(eigenvalues)):
                lam = eigenvalues[i]
                if lam < 1e-10:
                    continue
                phi = eigenfunctions[i]
                filtered = mask_spectrum * phi
                field_i = backend.ifft2(filtered)

                g_field = 2.0 * dLdI * field_i
                g_field_spectrum = backend.fft2(g_field)
                grad_spectrum = grad_spectrum + lam * backend.conj(phi) * g_field_spectrum
        elif mode == TCCMode.KERNEL_2D and self.tcc_kernel is not None:
            tcc_kernel = _asarray(self.tcc_kernel)
            effective_pupil = backend.sqrt(backend.maximum(tcc_kernel, 0.0))
            filtered = mask_spectrum * effective_pupil
            field = backend.ifft2(filtered)

            g_field = 2.0 * dLdI * field
            g_field_spectrum = backend.fft2(g_field)
            grad_spectrum = backend.conj(effective_pupil) * g_field_spectrum
        else:
            source = _asarray(self.source)
            fx = _asarray(self.fx)
            fy = _asarray(self.fy)
            pupil = _asarray(self.pupil)
            cutoff = self.optics.cutoff_frequency

            source_indices = backend.where_idx(source > 1e-10)
            source_values = source[source_indices]

            for idx in range(len(source_indices[0])):
                sy, sx = int(source_indices[0][idx]), int(source_indices[1][idx])
                src_val = source_values[idx]

                fs_x = fx[sy, sx]
                fs_y = fy[sy, sx]

                if backend.sqrt(fs_x**2 + fs_y**2) > cutoff:
                    continue

                pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, self.dfx, self.dfy)
                filtered = mask_spectrum * pupil_shifted
                field_i = backend.ifft2(filtered)

                g_field = 2.0 * dLdI * field_i
                g_field_spectrum = backend.fft2(g_field)
                grad_spectrum = grad_spectrum + src_val * backend.conj(pupil_shifted) * g_field_spectrum

        grad_spatial = backend.ifft2(grad_spectrum)

        if self.pad_width is not None:
            if isinstance(self.pad_width, int):
                py, px = self.pad_width, self.pad_width
            else:
                py, px = self.pad_width
            grad_spatial = grad_spatial[py:py + self.image_size[0], px:px + self.image_size[1]]
            grad_spectrum_out = grad_spectrum  # 频域梯度保持填充后尺寸
        else:
            grad_spectrum_out = grad_spectrum

        if self._window_2d is not None:
            window = _asarray(self._window_2d).astype(backend.complex128)
            grad_spatial = grad_spatial * window

        grad_spatial_np = _tonumpy(grad_spatial).astype(np.complex128)
        grad_spectrum_np = _tonumpy(grad_spectrum_out).astype(np.complex128)

        if return_spectral:
            return grad_spatial_np, grad_spectrum_np
        return grad_spatial_np

    def compute_image_gradient(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间像对掩模的梯度（用于优化）

        支持三种 TCC 计算模式。

        如果启用了窗函数，梯度会自动乘以窗函数以保持链式法则一致性。

        支持 CPU/GPU 透明切换，通过 ArrayBackend 统一调度。

        Args:
            mask: 掩模图案

        Returns:
            梯度数组（原始掩模尺寸）
        """
        backend = get_backend()
        processed_mask = self._preprocess_mask(mask)
        mask_c = _asarray(processed_mask).astype(backend.complex128)
        ny_eff, nx_eff = processed_mask.shape
        gradient = backend.zeros((ny_eff, nx_eff), dtype=backend.float64)
        mask_spectrum = backend.fft2(mask_c)
        cutoff = self.optics.cutoff_frequency
        mode = self.optics.tcc_mode

        if mode == TCCMode.SOCS and self.socs_eigenvalues is not None:
            eigenvalues = _asarray(self.socs_eigenvalues)
            eigenfunctions = _asarray(self.socs_eigenfunctions)
            for i in range(len(eigenvalues)):
                lam = eigenvalues[i]
                if lam < 1e-10:
                    continue
                phi = eigenfunctions[i]
                filtered = mask_spectrum * phi
                field_i = backend.ifft2(filtered)
                grad_field_i = backend.ifft2(phi)
                gradient = gradient + 2 * lam * backend.real(backend.conj(field_i) * grad_field_i)
        elif mode == TCCMode.KERNEL_2D and self.tcc_kernel is not None:
            tcc_kernel = _asarray(self.tcc_kernel)
            effective_pupil = backend.sqrt(backend.maximum(tcc_kernel, 0.0))
            filtered = mask_spectrum * effective_pupil
            field = backend.ifft2(filtered)
            grad_field = backend.ifft2(effective_pupil)
            gradient = 2.0 * backend.real(backend.conj(field) * grad_field)
        else:
            source = _asarray(self.source)
            fx = _asarray(self.fx)
            fy = _asarray(self.fy)
            pupil = _asarray(self.pupil)

            source_indices = backend.where_idx(source > 1e-10)
            source_values = source[source_indices]

            for idx in range(len(source_indices[0])):
                sy, sx = int(source_indices[0][idx]), int(source_indices[1][idx])
                src_val = source_values[idx]

                fs_x = fx[sy, sx]
                fs_y = fy[sy, sx]

                if backend.sqrt(fs_x**2 + fs_y**2) > cutoff:
                    continue

                pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, self.dfx, self.dfy)
                filtered = mask_spectrum * pupil_shifted
                field_i = backend.ifft2(filtered)
                grad_field_i = backend.ifft2(pupil_shifted)
                gradient = gradient + 2 * src_val * backend.real(backend.conj(field_i) * grad_field_i)

        if self.pad_width is not None:
            if isinstance(self.pad_width, int):
                py, px = self.pad_width, self.pad_width
            else:
                py, px = self.pad_width
            gradient = gradient[py:py + self.image_size[0], px:px + self.image_size[1]]

        if self._window_2d is not None:
            window = _asarray(self._window_2d)
            gradient = gradient * window

        grad_np = _tonumpy(gradient.astype(backend.float64))

        grad_np = self._apply_flare_gradient(grad_np)
        grad_np = self._apply_shadowing_gradient(grad_np)
        grad_np = self._apply_attenuation_gradient(grad_np)

        return grad_np

    def get_source_image(self) -> np.ndarray:
        """获取光源分布图像（fftshift后便于可视化）"""
        backend = get_backend()
        source = _asarray(self.source)
        return _tonumpy(backend.fftshift(source))

    def get_pupil_image(self) -> np.ndarray:
        """获取光瞳函数图像（fftshift后便于可视化）"""
        backend = get_backend()
        pupil = _asarray(self.pupil)
        return _tonumpy(backend.fftshift(backend.abs(pupil)))

    def get_tcc_image(self) -> Optional[np.ndarray]:
        """获取TCC核图像（fftshift后便于可视化）

        返回 2D TCC 核对角近似（fftshift 后）。所有模式下均可用。
        """
        if self.tcc_kernel is not None:
            backend = get_backend()
            tcc = _asarray(self.tcc_kernel)
            return _tonumpy(backend.fftshift(tcc))
        return None

    def update_source(self, new_source: np.ndarray) -> None:
        """
        更新光源分布并重新计算传递函数

        Args:
            new_source: 新的光源分布，形状需与频率网格一致
        """
        backend = get_backend()
        if new_source.shape != self.source.shape:
            raise ValueError(
                f"新光源形状 {new_source.shape} 与当前形状 {self.source.shape} 不匹配"
            )

        new_src = _asarray(new_source)
        new_src = backend.clip(new_src, 0.0, None)
        total = backend.sum(new_src)
        if total > 0:
            new_src = new_src / total

        self.source = _tonumpy(new_src.astype(backend.float64))
        self._compute_transfer_functions()

    def compute_source_gradient(self, mask: np.ndarray,
                                dLoss_dAerial: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算损失对光源分布的梯度 dLoss/dS。

        根据 Hopkins 公式:
            I(x, y) = ∫ S(fs) · |FFT^{-1}[M(f) · P(f - fs)]|^2 dfs

        链式法则：
            dLoss/dS(fs_i) = Σ_{x,y} dLoss/dI(x,y) · dI/dS(fs_i)(x,y)
            其中 dI/dS(fs_i)(x,y) = |FFT^{-1}[M(f) · P(f - fs_i)]|^2

        若不提供 dLoss_dAerial，则默认 dLoss/dI = 1（返回每个光源点贡献
        的空间像总能量，主要用于调试或用户自己实现链式相乘）。

        KERNEL_2D 模式下不支持光源梯度计算（2D核对角近似下光源信息已被积分掉），
        将返回全零数组。

        支持 CPU/GPU 透明切换，通过 ArrayBackend 统一调度。

        Args:
            mask: 掩模图案 (H, W)
            dLoss_dAerial: 损失对空间像的梯度 (H, W)，可选

        Returns:
            梯度数组，形状与 self.source 相同（频率网格尺寸）
        """
        backend = get_backend()
        processed_mask = self._preprocess_mask(mask)
        mask_c = _asarray(processed_mask).astype(backend.complex128)
        mask_spectrum = backend.fft2(mask_c)
        ny_eff, nx_eff = mask_spectrum.shape
        cutoff = self.optics.cutoff_frequency
        mode = self.optics.tcc_mode

        gradient = backend.zeros_like(_asarray(self.source), dtype=backend.float64)

        if mode == TCCMode.KERNEL_2D:
            return _tonumpy(gradient)

        if dLoss_dAerial is None:
            dLoss_dI = backend.ones(self.image_size, dtype=backend.float64)
        else:
            dLoss_dI = _asarray(dLoss_dAerial).astype(backend.float64)
            if dLoss_dI.shape != self.image_size:
                raise ValueError(
                    f"dLoss_dAerial 形状 {dLoss_dI.shape} 与 image_size {self.image_size} 不匹配"
                )

        if self.pad_width is not None:
            if isinstance(self.pad_width, int):
                py, px = self.pad_width, self.pad_width
            else:
                py, px = self.pad_width
            dLoss_dI_padded = backend.zeros((ny_eff, nx_eff), dtype=backend.float64)
            dLoss_dI_padded[py:py + self.image_size[0], px:px + self.image_size[1]] = dLoss_dI
            dLoss_dI = dLoss_dI_padded

        source = _asarray(self.source)
        fx = _asarray(self.fx)
        fy = _asarray(self.fy)
        pupil = _asarray(self.pupil)

        source_indices = backend.where_idx(source > 1e-12)
        num_points = len(source_indices[0])
        if num_points == 0:
            return _tonumpy(gradient)

        max_points_per_batch = 64
        for batch_start in range(0, num_points, max_points_per_batch):
            batch_end = min(batch_start + max_points_per_batch, num_points)

            for bi in range(batch_start, batch_end):
                sy = int(source_indices[0][bi])
                sx = int(source_indices[1][bi])

                fs_x = fx[sy, sx]
                fs_y = fy[sy, sx]
                if cutoff is not None and backend.sqrt(fs_x ** 2 + fs_y ** 2) > cutoff:
                    continue

                pupil_shifted = _shift_pupil(pupil, fs_x, fs_y, self.dfx, self.dfy)
                filtered = mask_spectrum * pupil_shifted
                field_i = backend.ifft2(filtered)
                intensity_i = backend.abs(field_i) ** 2

                gradient[sy, sx] = float(backend.sum(dLoss_dI * intensity_i))

        return _tonumpy(gradient.astype(backend.float64))


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
                         resist_model: Optional[ResistModel] = None,
                         window_type: Optional[Union[WindowType, str]] = None,
                         pad_width: Optional[Union[int, Tuple[int, int]]] = None,
                         tukey_alpha: float = 0.5) -> np.ndarray:
    """
    模拟晶圆成像

    完整的成像流程：掩模 -> 加窗/零填充 -> 光学成像 -> 剂量缩放 -> 光刻胶响应

    Args:
        mask: 掩模图案 (2D numpy数组)
        optical_system: 光学系统参数，None则使用默认参数
        threshold: 光刻胶阈值（当 resist_model 为 None 时生效）
        apply_resist: 是否应用光刻胶响应
        dose: 曝光相对剂量，1.0为标称剂量，大于1为过曝，小于1为欠曝
        resist_model: 高级光刻胶模型配置，优先于 threshold/apply_resist 参数
        window_type: 窗函数类型 ('hann', 'hamming', 'tukey')，None 不加窗
        pad_width: 零填充宽度（像素），None 不填充；整数表示各方向均匀填充
        tukey_alpha: Tukey 窗渐变比例因子 (0~1)

    Returns:
        晶圆成像结果
    """
    if optical_system is None:
        optical_system = OpticalSystem()

    imaging_model = PartialCoherentImaging(
        optical_system, mask.shape,
        window_type=window_type,
        pad_width=pad_width,
        tukey_alpha=tukey_alpha
    )

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
    resist_model: Optional[ResistModel] = None,
    window_type: Optional[Union[WindowType, str]] = None,
    pad_width: Optional[Union[int, Tuple[int, int]]] = None,
    tukey_alpha: float = 0.5
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
        window_type: 窗函数类型 ('hann', 'hamming', 'tukey')，None 不加窗
        pad_width: 零填充宽度（像素），None 不填充
        tukey_alpha: Tukey 窗渐变比例因子 (0~1)

    Returns:
        MultiProcessSimulationResult，包含所有工艺条件下的仿真结果
    """
    if base_optics is None:
        base_optics = OpticalSystem()

    aerial_images = []
    wafer_images = []

    for cond in conditions:
        optics = cond.to_optical_system(base_optics=base_optics)
        imaging_model = PartialCoherentImaging(
            optics, mask.shape,
            window_type=window_type,
            pad_width=pad_width,
            tukey_alpha=tukey_alpha
        )
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
            tcc_mode=base_optics.tcc_mode,
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
                tcc_mode=base_optics.tcc_mode,
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
