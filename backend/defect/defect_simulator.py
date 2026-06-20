# -*- coding: utf-8 -*-
"""
缺陷仿真与 CD 变化计算模块

对注入缺陷前后的掩模分别进行光刻成像仿真，
提取晶圆 CD 并计算缺陷诱导的 CD 变化量。
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass
import logging
import time

from defect.schemas import (
    DefectType,
    DefectPolarity,
    PointDefect,
    LineDefect,
    ContaminationDefect,
    DefectLocation,
    DefectInjectionConfig,
    SingleDefectResult,
)
from defect.defect_injector import DefectInjector

from core.imaging import (
    OpticalSystem,
    ProcessCondition,
    PartialCoherentImaging,
    apply_resist_model,
    _apply_threshold,
)
from core.litho_metrics import compute_cd, compute_cd_error
from metrology.cd_extraction import (
    MeasurementLine,
    extract_cd,
    CDExtractionMethod,
    CDExtractionResult,
)

logger = logging.getLogger(__name__)


class DefectSimulator:
    """
    缺陷仿真器

    对注入缺陷前后的掩模进行成像仿真，计算 CD 变化。

    使用方式::

        simulator = DefectSimulator(optical_system, config)
        result = simulator.simulate_defect(mask_nominal, defect)
        print(f"CD 变化: {result.delta_cd:.2f} nm")
    """

    def __init__(
        self,
        optical_system: Optional[OpticalSystem] = None,
        config: Optional[DefectInjectionConfig] = None,
        pixel_size: float = 1.0,
        threshold: float = 0.3,
        measurement_lines: Optional[List[MeasurementLine]] = None,
        cd_extraction_method: Union[str, CDExtractionMethod] = CDExtractionMethod.THRESHOLD_CROSSING,
        window_type: Optional[str] = None,
        pad_width: Optional[Union[int, Tuple[int, int]]] = None,
        tukey_alpha: float = 0.5,
    ):
        """
        初始化缺陷仿真器

        Args:
            optical_system: 光学系统参数，None 则使用默认
            config: 缺陷注入配置，None 则使用默认
            pixel_size: 像素尺寸 (nm/pixel)，仅当 config 为 None 时使用
            threshold: 光刻胶阈值，仅当 config 为 None 时使用
            measurement_lines: CD 测量线定义；None 则自动生成穿过中心的测量线
            cd_extraction_method: CD 提取方法
            window_type: 成像仿真的窗函数类型
            pad_width: 成像仿真的零填充宽度
            tukey_alpha: Tukey 窗渐变比例
        """
        if config is not None:
            self.config = config
        else:
            self.config = DefectInjectionConfig(
                pixel_size=pixel_size,
                threshold=threshold,
            )

        self.optical_system = optical_system if optical_system is not None else OpticalSystem()
        self.injector = DefectInjector(self.config)
        self.measurement_lines = measurement_lines
        self.cd_extraction_method = cd_extraction_method
        self.window_type = window_type
        self.pad_width = pad_width
        self.tukey_alpha = tukey_alpha

        self._nominal_result_cache: Optional[Dict[str, Any]] = None

    def _auto_measurement_lines(
        self,
        image_shape: Tuple[int, int],
    ) -> List[MeasurementLine]:
        """
        自动生成穿过图像中心的水平和垂直测量线

        Args:
            image_shape: (ny, nx) 图像尺寸

        Returns:
            测量线列表
        """
        ny, nx = image_shape
        cy, cx = ny / 2.0, nx / 2.0
        margin = min(ny, nx) * 0.1

        return [
            MeasurementLine(
                start=(cy, margin),
                end=(cy, nx - margin),
                direction='horizontal',
                name='ML_horizontal',
            ),
            MeasurementLine(
                start=(margin, cx),
                end=(ny - margin, cx),
                direction='vertical',
                name='ML_vertical',
            ),
        ]

    def _get_measurement_lines(
        self,
        image_shape: Tuple[int, int],
    ) -> List[MeasurementLine]:
        if self.measurement_lines is None:
            return self._auto_measurement_lines(image_shape)
        return self.measurement_lines

    def simulate_image(
        self,
        mask: np.ndarray,
        process_condition: Optional[ProcessCondition] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单个掩模进行成像仿真

        Args:
            mask: 掩模图案
            process_condition: 工艺条件，None 则使用标称条件

        Returns:
            (aerial_image, wafer_image) 空间像和晶圆图
        """
        if process_condition is not None:
            optics = process_condition.to_optical_system(base_optics=self.optical_system)
        else:
            optics = self.optical_system

        imaging_model = PartialCoherentImaging(
            optics, mask.shape,
            window_type=self.window_type,
            pad_width=self.pad_width,
            tukey_alpha=self.tukey_alpha,
        )
        aerial = imaging_model.compute_aerial_image(mask)

        if process_condition is not None and process_condition.dose != 1.0:
            aerial_dosed = np.clip(aerial * process_condition.dose, 0.0, 1.0)
        else:
            aerial_dosed = aerial.copy()

        wafer = _apply_threshold(aerial_dosed, self.config.threshold)
        return aerial_dosed, wafer

    def measure_cd(
        self,
        wafer_image: np.ndarray,
    ) -> Tuple[float, List[CDExtractionResult]]:
        """
        从晶圆图中测量平均 CD

        Args:
            wafer_image: 二值化晶圆图像

        Returns:
            (mean_cd_nm, extraction_results) 平均 CD 和各测量线提取结果
        """
        lines = self._get_measurement_lines(wafer_image.shape)
        results = []
        cd_values = []

        for line in lines:
            result = extract_cd(
                wafer_image, line,
                method=self.cd_extraction_method,
                pixel_size=self.config.pixel_size,
            )
            results.append(result)
            if result.confidence > 0.1 and result.cd_value > 0:
                cd_values.append(result.cd_value)

        if cd_values:
            mean_cd = float(np.mean(cd_values))
        else:
            cd_stats = compute_cd(wafer_image, pixel_size=self.config.pixel_size)
            mean_cd = cd_stats['cd_mean']

        return mean_cd, results

    def _get_or_simulate_nominal(
        self,
        mask_nominal: np.ndarray,
    ) -> Dict[str, Any]:
        """
        获取标称掩模的仿真结果（带缓存）

        Args:
            mask_nominal: 标称掩模

        Returns:
            包含 nominal_aerial, nominal_wafer, nominal_cd, cd_results 的字典
        """
        if self._nominal_result_cache is not None:
            return self._nominal_result_cache

        t0 = time.time()
        aerial, wafer = self.simulate_image(mask_nominal)
        cd, cd_results = self.measure_cd(wafer)
        elapsed = time.time() - t0
        logger.debug(f"标称仿真完成，CD={cd:.2f}nm，耗时 {elapsed:.2f}s")

        result = {
            'aerial': aerial,
            'wafer': wafer,
            'cd': cd,
            'cd_results': cd_results,
        }
        self._nominal_result_cache = result
        return result

    def clear_cache(self):
        """清除标称仿真缓存"""
        self._nominal_result_cache = None

    def simulate_defect(
        self,
        mask_nominal: np.ndarray,
        defect: Union[PointDefect, LineDefect, ContaminationDefect],
        process_condition: Optional[ProcessCondition] = None,
        save_images: bool = True,
    ) -> SingleDefectResult:
        """
        仿真单个缺陷对晶圆成像的影响

        Args:
            mask_nominal: 标称掩模
            defect: 缺陷参数
            process_condition: 工艺条件，None 使用标称条件
            save_images: 是否在结果中保存仿真图像

        Returns:
            SingleDefectResult，包含 CD 变化、失效判定等
        """
        nominal = self._get_or_simulate_nominal(mask_nominal)
        nominal_cd = nominal['cd']

        mask_defective = self.injector.inject(mask_nominal, defect)

        t0 = time.time()
        def_aerial, def_wafer = self.simulate_image(mask_defective, process_condition)
        def_cd, def_cd_results = self.measure_cd(def_wafer)
        elapsed = time.time() - t0
        logger.debug(f"缺陷仿真完成，类型={defect.__class__.__name__}，"
                     f"缺陷CD={def_cd:.2f}nm，耗时 {elapsed:.2f}s")

        delta_cd = def_cd - nominal_cd
        if nominal_cd > 1e-10:
            delta_cd_relative = delta_cd / nominal_cd * 100.0
        else:
            delta_cd_relative = 0.0

        is_critical = False
        failure_probability = 0.0
        if self.config.cd_target is not None or nominal_cd > 0:
            cd_ref = self.config.cd_target if self.config.cd_target is not None else nominal_cd
            cd_lower = cd_ref * (1.0 - self.config.cd_tolerance)
            cd_upper = cd_ref * (1.0 + self.config.cd_tolerance)
            is_critical = (def_cd < cd_lower) or (def_cd > cd_upper)

            if nominal_cd > 1e-10 and self.config.cd_tolerance > 0:
                from scipy.stats import norm
                sigma_est = max(abs(delta_cd) / 3.0, cd_ref * self.config.cd_tolerance / 6.0)
                if delta_cd >= 0:
                    z = (def_cd - cd_upper) / sigma_est if sigma_est > 0 else 0.0
                    failure_probability = float(1.0 - norm.cdf(z)) if is_critical else 0.0
                else:
                    z = (def_cd - cd_lower) / sigma_est if sigma_est > 0 else 0.0
                    failure_probability = float(norm.cdf(z)) if is_critical else 0.0

        sensitivity_score = self._compute_sensitivity_score(
            delta_cd, delta_cd_relative, is_critical, failure_probability
        )

        defect_params = self._defect_to_params_dict(defect)

        return SingleDefectResult(
            defect_type=self._defect_to_type(defect),
            defect_params=defect_params,
            nominal_cd=float(nominal_cd),
            defective_cd=float(def_cd),
            delta_cd=float(delta_cd),
            delta_cd_relative=float(delta_cd_relative),
            nominal_aerial=nominal['aerial'] if save_images else None,
            nominal_wafer=nominal['wafer'] if save_images else None,
            defective_aerial=def_aerial if save_images else None,
            defective_wafer=def_wafer if save_images else None,
            mask_defective=mask_defective if save_images else None,
            is_critical=bool(is_critical),
            failure_probability=float(failure_probability),
            sensitivity_score=float(sensitivity_score),
            measurement_lines=self._get_measurement_lines(mask_nominal.shape),
        )

    def simulate_defects_batch(
        self,
        mask_nominal: np.ndarray,
        defects: List[Union[PointDefect, LineDefect, ContaminationDefect]],
        process_condition: Optional[ProcessCondition] = None,
        save_images: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> List[SingleDefectResult]:
        """
        批量仿真多个缺陷

        Args:
            mask_nominal: 标称掩模
            defects: 缺陷参数列表
            process_condition: 工艺条件
            save_images: 是否保存图像（批量模式默认不保存以节省内存）
            progress_callback: 进度回调 callback(current, total)

        Returns:
            仿真结果列表
        """
        results = []
        total = len(defects)
        for i, defect in enumerate(defects, 1):
            result = self.simulate_defect(
                mask_nominal, defect, process_condition,
                save_images=save_images,
            )
            results.append(result)
            if progress_callback is not None:
                progress_callback(i, total)
        return results

    @staticmethod
    def _compute_sensitivity_score(
        delta_cd: float,
        delta_cd_relative: float,
        is_critical: bool,
        failure_probability: float,
    ) -> float:
        """
        计算综合敏感度评分

        综合考虑绝对 CD 变化、相对 CD 变化、是否致命、失效概率。
        分数越高表示越敏感。

        Args:
            delta_cd: 绝对 CD 变化 (nm)
            delta_cd_relative: 相对 CD 变化 (%)
            is_critical: 是否为致命缺陷
            failure_probability: 失效概率

        Returns:
            敏感度评分 (0~100)
        """
        abs_rel = abs(delta_cd_relative)
        base_score = min(abs_rel * 2.0, 80.0)

        if is_critical:
            base_score = max(base_score, 60.0)
            base_score += failure_probability * 20.0

        return float(min(max(base_score, 0.0), 100.0))

    @staticmethod
    def _defect_to_type(defect) -> DefectType:
        if isinstance(defect, PointDefect):
            return DefectType.POINT
        elif isinstance(defect, LineDefect):
            return DefectType.LINE
        elif isinstance(defect, ContaminationDefect):
            return DefectType.CONTAMINATION
        else:
            raise TypeError(f"未知缺陷类型: {type(defect)}")

    @staticmethod
    def _defect_to_params_dict(defect) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if isinstance(defect, PointDefect):
            params = {
                'size_nm': defect.size_nm,
                'shape': defect.shape,
                'polarity': defect.polarity.value,
                'location_y': defect.location.y if defect.location else None,
                'location_x': defect.location.x if defect.location else None,
                'distance_to_edge': defect.location.distance_to_edge if defect.location else None,
            }
        elif isinstance(defect, LineDefect):
            params = {
                'length_nm': defect.length_nm,
                'width_nm': defect.width_nm,
                'angle_deg': defect.angle_deg,
                'polarity': defect.polarity.value,
                'location_y': defect.location.y if defect.location else None,
                'location_x': defect.location.x if defect.location else None,
                'distance_to_edge': defect.location.distance_to_edge if defect.location else None,
            }
        elif isinstance(defect, ContaminationDefect):
            params = {
                'size_nm': defect.size_nm,
                'attenuation': defect.attenuation,
                'roughness': defect.roughness,
                'polarity': defect.polarity.value,
                'location_y': defect.location.y if defect.location else None,
                'location_x': defect.location.x if defect.location else None,
                'distance_to_edge': defect.location.distance_to_edge if defect.location else None,
            }
        return params

    @staticmethod
    def get_defect_size(defect) -> float:
        """获取缺陷特征尺寸 (nm)"""
        if isinstance(defect, PointDefect):
            return defect.size_nm
        elif isinstance(defect, LineDefect):
            return max(defect.length_nm, defect.width_nm)
        elif isinstance(defect, ContaminationDefect):
            return defect.size_nm
        else:
            return 0.0

    @staticmethod
    def get_defect_polarity(defect) -> DefectPolarity:
        if isinstance(defect, (PointDefect, LineDefect, ContaminationDefect)):
            return defect.polarity
        return DefectPolarity.OPAQUE
