# -*- coding: utf-8 -*-
"""
参数不确定性建模与采样模块

对光刻工艺参数（focus, dose, NA, sigma, 像差等）和模型参数
（光刻胶阈值、扩散长度等）的不确定性进行建模，并提供采样功能。

支持的概率分布：
- 正态分布 (Normal)
- 均匀分布 (Uniform)
- 三角分布 (Triangular)
- 对数正态分布 (LogNormal)
- Gamma 分布
- Beta 分布
- 自定义分布 (基于样本或 PDF)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass
from scipy import stats
import logging

from core.imaging import (
    OpticalSystem,
    ProcessCondition,
)
from uq.schemas import (
    ParameterDistribution,
    ProcessUncertaintyConfig,
    ModelUncertaintyConfig,
)

logger = logging.getLogger(__name__)


class ParameterSampler:
    """
    通用参数采样器

    根据参数分布描述，生成随机样本。
    支持传入分布列表同时采样多个参数。
    """

    def __init__(
        self,
        distributions: Optional[List[ParameterDistribution]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        初始化采样器

        Args:
            distributions: 参数分布列表（可选，单分布采样时可为 None）
            random_seed: 随机种子
        """
        self.rng = np.random.default_rng(random_seed)
        self.distributions = distributions or []

    def sample(
        self,
        dist_or_n: Union[ParameterDistribution, int, None] = None,
        n: Optional[int] = None,
    ) -> np.ndarray:
        """
        采样

        调用方式：
        - sampler.sample(dist, n): 从单个分布采 n 个
        - sampler.sample(n): 从 distributions 列表采 n 组 (n, n_params)
        - sampler.sample(): 从 distributions 列表采 1 组 (1, n_params)

        Returns:
            样本数组 (n_samples, n_params) 或 (n_samples,)
        """
        if isinstance(dist_or_n, ParameterDistribution):
            single_dist = dist_or_n
            count = n if n is not None else 1
            return self._sample_single(single_dist, count)

        if isinstance(dist_or_n, int):
            count = dist_or_n
            return self._sample_multi(count)

        if dist_or_n is None:
            count = n if n is not None else 1
            return self._sample_multi(count)

        raise TypeError(f"Unsupported arg: {type(dist_or_n)}")

    def _sample_single(self, dist: ParameterDistribution, n: int) -> np.ndarray:
        return dist.sample(n, self.rng)

    def _sample_multi(self, n: int) -> np.ndarray:
        if not self.distributions:
            return np.zeros((n, 0))
        cols = [d.sample(n, self.rng) for d in self.distributions]
        return np.column_stack(cols)

    def _rejection_sampling(
        self, pdf: callable, n: int, dist: ParameterDistribution
    ) -> np.ndarray:
        """拒绝采样实现"""
        samples = []
        low = dist.bounds[0] if dist.bounds is not None else dist.nominal - 10 * dist.params.get("std", 1.0)
        high = dist.bounds[1] if dist.bounds is not None else dist.nominal + 10 * dist.params.get("std", 1.0)
        max_pdf = pdf(dist.nominal) * 1.5
        max_iter = n * 100
        iterations = 0

        while len(samples) < n and iterations < max_iter:
            x = self.rng.uniform(low, high)
            u = self.rng.uniform(0, max_pdf)
            if u <= pdf(x):
                samples.append(x)
            iterations += 1

        if len(samples) < n:
            padding = self.rng.normal(loc=dist.nominal, scale=dist.params.get("std", 1.0), size=n - len(samples))
            samples.extend(padding.tolist())

        return np.array(samples[:n])


class ProcessPerturbationSampler:
    """
    工艺参数扰动采样器

    基于 ProcessUncertaintyConfig 生成多组工艺参数扰动样本，
    每组样本包含一组 ProcessCondition。
    """

    def __init__(
        self,
        config: ProcessUncertaintyConfig,
        base_optics: Optional[OpticalSystem] = None,
        random_seed: Optional[int] = None,
    ):
        """
        初始化工艺参数采样器

        Args:
            config: 工艺不确定性配置
            base_optics: 基准光学系统（用于获取标称参数）
            random_seed: 随机种子（覆盖 config.random_seed）
        """
        self.config = config
        self.base_optics = base_optics if base_optics is not None else OpticalSystem()
        seed = random_seed if random_seed is not None else config.random_seed
        self.sampler = ParameterSampler(random_seed=seed)

    def _get_default_zernike_indices(self) -> List[int]:
        """获取默认的 Zernike 系数索引（常见像差）"""
        if self.config.zernike_indices is not None:
            return list(self.config.zernike_indices)
        existing = list(self.base_optics.zernike_coefficients.keys())
        if existing:
            return existing
        return [3, 4, 5, 6, 7, 8, 9, 10]

    def sample(self, n: int) -> List[ProcessCondition]:
        """
        生成 n 组工艺条件样本

        Args:
            n: 样本数量

        Returns:
            ProcessCondition 列表，长度 n
        """
        rng = self.sampler.rng
        dist_type = self.config.distribution

        nominal = ProcessCondition.from_optical_system(self.base_optics)

        if dist_type == "normal":
            focus_samples = rng.normal(
                loc=nominal.defocus, scale=self.config.focus_std, size=n
            )
            dose_samples = rng.normal(
                loc=nominal.dose,
                scale=self.config.dose_std * nominal.dose,
                size=n,
            )
            na_samples = rng.normal(
                loc=nominal.na, scale=self.config.na_std * nominal.na, size=n
            )
            sigma_samples = rng.normal(
                loc=nominal.sigma,
                scale=self.config.sigma_std * nominal.sigma,
                size=n,
            )
            wavelength_samples = rng.normal(
                loc=nominal.wavelength, scale=self.config.wavelength_std, size=n
            )
            flare_samples = rng.normal(
                loc=nominal.flare,
                scale=self.config.flare_std * max(nominal.flare, 0.01),
                size=n,
            )
        elif dist_type == "uniform":
            half_focus = self.config.focus_std * np.sqrt(3)
            half_dose = self.config.dose_std * nominal.dose * np.sqrt(3)
            half_na = self.config.na_std * nominal.na * np.sqrt(3)
            half_sigma = self.config.sigma_std * nominal.sigma * np.sqrt(3)
            half_wavelength = self.config.wavelength_std * np.sqrt(3)
            half_flare = self.config.flare_std * max(nominal.flare, 0.01) * np.sqrt(3)

            focus_samples = rng.uniform(
                nominal.defocus - half_focus, nominal.defocus + half_focus, size=n
            )
            dose_samples = rng.uniform(
                nominal.dose - half_dose, nominal.dose + half_dose, size=n
            )
            na_samples = rng.uniform(
                nominal.na - half_na, nominal.na + half_na, size=n
            )
            sigma_samples = rng.uniform(
                nominal.sigma - half_sigma, nominal.sigma + half_sigma, size=n
            )
            wavelength_samples = rng.uniform(
                nominal.wavelength - half_wavelength,
                nominal.wavelength + half_wavelength,
                size=n,
            )
            flare_samples = rng.uniform(
                nominal.flare - half_flare, nominal.flare + half_flare, size=n
            )
        elif dist_type == "triangular":
            half_focus = self.config.focus_std * np.sqrt(6)
            half_dose = self.config.dose_std * nominal.dose * np.sqrt(6)
            half_na = self.config.na_std * nominal.na * np.sqrt(6)
            half_sigma = self.config.sigma_std * nominal.sigma * np.sqrt(6)
            half_wavelength = self.config.wavelength_std * np.sqrt(6)
            half_flare = self.config.flare_std * max(nominal.flare, 0.01) * np.sqrt(6)

            focus_samples = rng.triangular(
                nominal.defocus - half_focus,
                nominal.defocus,
                nominal.defocus + half_focus,
                size=n,
            )
            dose_samples = rng.triangular(
                nominal.dose - half_dose, nominal.dose, nominal.dose + half_dose, size=n
            )
            na_samples = rng.triangular(
                nominal.na - half_na, nominal.na, nominal.na + half_na, size=n
            )
            sigma_samples = rng.triangular(
                nominal.sigma - half_sigma,
                nominal.sigma,
                nominal.sigma + half_sigma,
                size=n,
            )
            wavelength_samples = rng.triangular(
                nominal.wavelength - half_wavelength,
                nominal.wavelength,
                nominal.wavelength + half_wavelength,
                size=n,
            )
            flare_samples = rng.triangular(
                nominal.flare - half_flare,
                nominal.flare,
                nominal.flare + half_flare,
                size=n,
            )
        else:
            raise ValueError(f"未知分布类型: {dist_type}")

        dose_samples = np.clip(dose_samples, 0.1, 5.0)
        na_samples = np.clip(na_samples, 0.1, 2.0)
        sigma_samples = np.clip(sigma_samples, 0.05, 1.0)
        wavelength_samples = np.clip(wavelength_samples, 1.0, 500.0)
        flare_samples = np.clip(flare_samples, 0.0, 1.0)

        zernike_indices = self._get_default_zernike_indices()
        aberration_samples: Dict[int, np.ndarray] = {}

        if self.config.aberration_std is not None:
            if isinstance(self.config.aberration_std, (int, float)):
                aber_std_dict = {j: float(self.config.aberration_std) for j in zernike_indices}
            else:
                aber_std_dict = dict(self.config.aberration_std)

            for j in zernike_indices:
                std = aber_std_dict.get(j, 0.0)
                base_val = self.base_optics.zernike_coefficients.get(j, 0.0)
                if std > 0:
                    if dist_type == "normal":
                        aberration_samples[j] = rng.normal(loc=base_val, scale=std, size=n)
                    elif dist_type == "uniform":
                        half = std * np.sqrt(3)
                        aberration_samples[j] = rng.uniform(base_val - half, base_val + half, size=n)
                    elif dist_type == "triangular":
                        half = std * np.sqrt(6)
                        aberration_samples[j] = rng.triangular(
                            base_val - half, base_val, base_val + half, size=n
                        )
                else:
                    aberration_samples[j] = np.full(n, base_val)

        conditions = []
        for i in range(n):
            zernike_dict = dict(nominal.zernike_coefficients)
            for j, samples in aberration_samples.items():
                zernike_dict[j] = float(samples[i])

            cond = ProcessCondition(
                defocus=float(focus_samples[i]),
                dose=float(dose_samples[i]),
                na=float(na_samples[i]),
                sigma=float(sigma_samples[i]),
                wavelength=float(wavelength_samples[i]),
                flare=float(flare_samples[i]),
                shadowing_model=nominal.shadowing_model,
                reflective_mask_attenuation=nominal.reflective_mask_attenuation,
                technology_node=nominal.technology_node,
                zernike_coefficients=zernike_dict,
                use_vector_pupil=nominal.use_vector_pupil,
                incident_polarization_angle=nominal.incident_polarization_angle,
                n_immersion=nominal.n_immersion,
                use_mask_coating=nominal.use_mask_coating,
                name=f"uq_sample_{i:04d}",
                weight=1.0,
            )
            conditions.append(cond)

        for param_dist in self.config.custom_parameters:
            samples = self.sampler.sample(param_dist, n)
            for i in range(n):
                setattr(conditions[i], param_dist.name, float(samples[i]))

        return conditions


class ModelUncertaintySampler:
    """
    模型参数不确定性采样器

    对光刻仿真模型中的参数（阈值、扩散长度等）进行采样，
    用于评估模型不确定性对成像指标的影响。
    """

    def __init__(
        self,
        config: ModelUncertaintyConfig,
        random_seed: Optional[int] = None,
    ):
        """
        初始化模型参数采样器

        Args:
            config: 模型不确定性配置
            random_seed: 随机种子（覆盖 config.random_seed）
        """
        self.config = config
        seed = random_seed if random_seed is not None else config.random_seed
        self.sampler = ParameterSampler(random_seed=seed)

    def sample(self, n: int) -> List[Dict[str, float]]:
        """
        生成 n 组模型参数样本

        Args:
            n: 样本数量

        Returns:
            模型参数字典列表，每个字典包含:
                - threshold: 光刻胶阈值扰动比例
                - diffusion_length: 扩散长度扰动比例
                - resist_params: 光刻胶参数扰动比例
                - meef_factor: MEEF 扰动比例
                - aberration_calibration: 像差校准误差（波长λ）
                - model_form_noise: 模型形式噪声标准差
        """
        rng = self.sampler.rng
        dist_type = self.config.distribution

        results = []
        for i in range(n):
            params: Dict[str, float] = {}

            if dist_type == "normal":
                params["threshold"] = float(rng.normal(1.0, self.config.threshold_std))
                params["diffusion_length"] = float(
                    rng.normal(1.0, self.config.diffusion_length_std)
                )
                params["resist_params"] = float(
                    rng.normal(1.0, self.config.resist_model_params_std)
                )
                params["meef_factor"] = float(rng.normal(1.0, self.config.meef_std))
                params["aberration_calibration"] = float(
                    rng.normal(0.0, self.config.aberration_calibration_std)
                )
                params["model_form_noise"] = float(
                    rng.normal(0.0, self.config.model_form_uncertainty)
                )
            elif dist_type == "uniform":
                half_thr = self.config.threshold_std * np.sqrt(3)
                half_diff = self.config.diffusion_length_std * np.sqrt(3)
                half_resist = self.config.resist_model_params_std * np.sqrt(3)
                half_meef = self.config.meef_std * np.sqrt(3)
                half_aber = self.config.aberration_calibration_std * np.sqrt(3)
                half_form = self.config.model_form_uncertainty * np.sqrt(3)

                params["threshold"] = float(rng.uniform(1.0 - half_thr, 1.0 + half_thr))
                params["diffusion_length"] = float(
                    rng.uniform(1.0 - half_diff, 1.0 + half_diff)
                )
                params["resist_params"] = float(
                    rng.uniform(1.0 - half_resist, 1.0 + half_resist)
                )
                params["meef_factor"] = float(rng.uniform(1.0 - half_meef, 1.0 + half_meef))
                params["aberration_calibration"] = float(
                    rng.uniform(-half_aber, half_aber)
                )
                params["model_form_noise"] = float(rng.uniform(-half_form, half_form))
            else:
                params["threshold"] = float(rng.normal(1.0, self.config.threshold_std))
                params["diffusion_length"] = float(
                    rng.normal(1.0, self.config.diffusion_length_std)
                )
                params["resist_params"] = float(
                    rng.normal(1.0, self.config.resist_model_params_std)
                )
                params["meef_factor"] = float(rng.normal(1.0, self.config.meef_std))
                params["aberration_calibration"] = float(
                    rng.normal(0.0, self.config.aberration_calibration_std)
                )
                params["model_form_noise"] = float(
                    rng.normal(0.0, self.config.model_form_uncertainty)
                )

            params["threshold"] = max(0.5, min(2.0, params["threshold"]))
            params["diffusion_length"] = max(0.1, min(3.0, params["diffusion_length"]))

            params["threshold_factor"] = params["threshold"]
            params["diffusion_length_factor"] = params["diffusion_length"]

            results.append(params)

        for param_dist in self.config.custom_parameters:
            samples = self.sampler.sample(param_dist, n)
            for i in range(n):
                results[i][param_dist.name] = float(samples[i])

        return results


def sample_process_uncertainties(
    config: ProcessUncertaintyConfig,
    base_optics: Optional[OpticalSystem] = None,
    n_samples: int = 100,
) -> List[ProcessCondition]:
    """
    便捷函数：生成工艺参数扰动样本

    Args:
        config: 工艺不确定性配置
        base_optics: 基准光学系统
        n_samples: 样本数量

    Returns:
        ProcessCondition 列表
    """
    sampler = ProcessPerturbationSampler(config, base_optics)
    return sampler.sample(n_samples)


def sample_model_uncertainties(
    config: ModelUncertaintyConfig,
    n_samples: int = 100,
) -> List[Dict[str, float]]:
    """
    便捷函数：生成模型参数不确定性样本

    Args:
        config: 模型不确定性配置
        n_samples: 样本数量

    Returns:
        模型参数字典列表
    """
    sampler = ModelUncertaintySampler(config)
    return sampler.sample(n_samples)
