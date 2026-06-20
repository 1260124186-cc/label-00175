# -*- coding: utf-8 -*-
"""
蒙特卡洛仿真框架模块

对同一掩模进行多次随机光刻仿真，收集CD分布、边缘粗糙度等统计数据。

核心功能：
1. 集成光学成像模型与随机噪声模型
2. 批量蒙特卡洛仿真调度
3. 中间结果收集与存储
4. 进度回调支持
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import h5py

from core.imaging import (
    OpticalSystem,
    ProcessCondition,
    PartialCoherentImaging,
    simulate_multi_process,
)
from core.litho_metrics import compute_cd, compute_epe

from .noise_models import (
    NoiseConfig,
    NoiseGenerator,
    NoiseRealization,
    NoiseType,
    apply_stochastic_lithography,
)

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloStochasticConfig:
    """
    随机蒙特卡洛仿真配置

    Attributes:
        n_realizations: 蒙特卡洛实现次数
        noise_config: 噪声配置
        process_condition: 工艺条件
        pixel_size: 像素尺寸 (nm)
        base_threshold: 基础显影阈值
        save_intermediate: 是否保存中间结果（每一次的晶圆图）
        save_path: 中间结果保存路径（HDF5文件）
        batch_size: 批处理大小，用于内存优化
        progress_callback: 进度回调函数 callback(current, total, elapsed)
        random_seed: 全局随机种子
    """
    n_realizations: int = 100
    noise_config: Optional[NoiseConfig] = None
    process_condition: Optional[ProcessCondition] = None
    pixel_size: float = 1.0
    base_threshold: float = 0.5
    save_intermediate: bool = False
    save_path: Optional[str] = None
    batch_size: int = 10
    progress_callback: Optional[Callable[[int, int, float], None]] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.noise_config is None:
            self.noise_config = NoiseConfig(random_seed=self.random_seed)
        elif self.random_seed is not None:
            self.noise_config.random_seed = self.random_seed


@dataclass
class SingleRealizationResult:
    """
    单次蒙特卡洛实现结果

    Attributes:
        realization_id: 实现ID
        seed: 使用的随机种子
        aerial_image: 空间像光强分布
        latent_image: 扩散后的潜像
        resist_image: 显影后的光刻胶图像（连续值）
        wafer_binary: 二值化晶圆图
        noise: 噪声实现
        cd_horizontal: 水平方向CD统计
        cd_vertical: 垂直方向CD统计
        epe: EPE统计（如果有目标图）
        edge_positions: 边缘位置（用于LER计算）
        processing_time: 处理时间 (秒)
    """
    realization_id: int
    seed: int
    aerial_image: Optional[np.ndarray] = None
    latent_image: Optional[np.ndarray] = None
    resist_image: Optional[np.ndarray] = None
    wafer_binary: Optional[np.ndarray] = None
    noise: Optional[NoiseRealization] = None
    cd_horizontal: Optional[Dict[str, float]] = None
    cd_vertical: Optional[Dict[str, float]] = None
    epe: Optional[Dict[str, float]] = None
    edge_positions: Optional[Dict[str, np.ndarray]] = None
    processing_time: float = 0.0

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        result = {
            'realization_id': self.realization_id,
            'seed': self.seed,
            'cd_horizontal': self.cd_horizontal,
            'cd_vertical': self.cd_vertical,
            'epe': self.epe,
            'processing_time': self.processing_time,
        }
        if include_images:
            result['aerial_image'] = self.aerial_image
            result['latent_image'] = self.latent_image
            result['resist_image'] = self.resist_image
            result['wafer_binary'] = self.wafer_binary
        return result


@dataclass
class MonteCarloStochasticResult:
    """
    随机蒙特卡洛仿真完整结果

    Attributes:
        config: 仿真配置
        n_realizations: 完成的实现次数
        nominal_aerial_image: 标称空间像（无噪声）
        nominal_wafer_binary: 标称晶圆图（无噪声）
        all_cd_values: 所有CD值数组 (n_realizations, n_features)
        all_cd_horizontal: 水平CD统计数组
        all_cd_vertical: 垂直CD统计数组
        all_epe_values: 所有EPE值数组
        all_edge_positions: 所有边缘位置列表
        all_noise_seeds: 所有噪声种子列表
        realization_results: 单次实现结果列表
        total_time: 总处理时间 (秒)
        intermediate_file: 中间结果文件路径
    """
    config: MonteCarloStochasticConfig
    n_realizations: int = 0
    nominal_aerial_image: Optional[np.ndarray] = None
    nominal_wafer_binary: Optional[np.ndarray] = None
    all_cd_values: Optional[np.ndarray] = None
    all_cd_horizontal: Optional[np.ndarray] = None
    all_cd_vertical: Optional[np.ndarray] = None
    all_epe_values: Optional[np.ndarray] = None
    all_edge_positions: List[Dict[str, np.ndarray]] = field(default_factory=list)
    all_noise_seeds: List[int] = field(default_factory=list)
    realization_results: List[SingleRealizationResult] = field(default_factory=list)
    total_time: float = 0.0
    intermediate_file: Optional[str] = None

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        result = {
            'n_realizations': self.n_realizations,
            'total_time': self.total_time,
            'nominal_cd': None,
            'intermediate_file': self.intermediate_file,
        }
        if self.nominal_wafer_binary is not None:
            nominal_cd = compute_cd(
                self.nominal_wafer_binary, pixel_size=self.config.pixel_size
            )
            result['nominal_cd'] = nominal_cd
        if include_samples:
            result['all_cd_values'] = (
                self.all_cd_values.tolist() if self.all_cd_values is not None else None
            )
            result['all_noise_seeds'] = self.all_noise_seeds
        return result

    def summary(self) -> str:
        lines = [
            "=== 随机蒙特卡洛仿真结果 ===",
            f"  实现次数: {self.n_realizations}",
            f"  总耗时: {self.total_time:.1f}s",
            f"  平均每次: {self.total_time / max(1, self.n_realizations):.2f}s",
        ]
        if self.all_cd_horizontal is not None and len(self.all_cd_horizontal) > 0:
            cd_mean = np.mean(self.all_cd_horizontal)
            cd_std = np.std(self.all_cd_horizontal)
            lines.append("")
            lines.append("  水平CD统计:")
            lines.append(f"    均值: {cd_mean:.2f} nm")
            lines.append(f"    标准差: {cd_std:.2f} nm")
            lines.append(f"    3σ范围: [{cd_mean - 3*cd_std:.2f}, {cd_mean + 3*cd_std:.2f}] nm")
        if self.all_cd_vertical is not None and len(self.all_cd_vertical) > 0:
            cd_mean = np.mean(self.all_cd_vertical)
            cd_std = np.std(self.all_cd_vertical)
            lines.append("")
            lines.append("  垂直CD统计:")
            lines.append(f"    均值: {cd_mean:.2f} nm")
            lines.append(f"    标准差: {cd_std:.2f} nm")
            lines.append(f"    范围: [{np.min(self.all_cd_vertical):.2f}, {np.max(self.all_cd_vertical):.2f}] nm")
        return "\n".join(lines)


class StochasticMonteCarloSimulator:
    """
    随机光刻蒙特卡洛仿真器

    对同一掩模在相同工艺条件下进行多次随机仿真，
    收集CD分布、边缘粗糙度等统计数据。

    使用方式::

        simulator = StochasticMonteCarloSimulator(optical_system)
        result = simulator.run_simulation(
            mask=mask,
            target=target,
            config=MonteCarloStochasticConfig(n_realizations=100)
        )
    """

    def __init__(
        self,
        optical_system: OpticalSystem,
        window_type: Optional[Any] = None,
        pad_width: Optional[Union[int, Tuple[int, int]]] = None,
        tukey_alpha: float = 0.5,
    ):
        """
        初始化仿真器

        Args:
            optical_system: 光学系统参数
            window_type: 窗函数类型
            pad_width: 零填充宽度
            tukey_alpha: Tukey 窗渐变比例因子
        """
        self.optical_system = optical_system
        self.window_type = window_type
        self.pad_width = pad_width
        self.tukey_alpha = tukey_alpha
        self._imaging_model: Optional[PartialCoherentImaging] = None

    def _setup_imaging_model(self, image_size: Tuple[int, int]):
        """设置成像模型"""
        if self._imaging_model is None or self._imaging_model.image_size != image_size:
            self._imaging_model = PartialCoherentImaging(
                optical_system=self.optical_system,
                image_size=image_size,
                window_type=self.window_type,
                pad_width=self.pad_width,
                tukey_alpha=self.tukey_alpha,
            )

    def _compute_nominal_aerial_image(self, mask: np.ndarray) -> np.ndarray:
        """计算标称空间像（无噪声）"""
        self._setup_imaging_model(mask.shape)
        return self._imaging_model.compute_aerial_image(mask)

    def _extract_edge_positions(
        self,
        wafer_binary: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        提取边缘位置（用于LER计算）

        Args:
            wafer_binary: 二值化晶圆图

        Returns:
            包含水平和垂直边缘位置的字典
        """
        ny, nx = wafer_binary.shape
        result = {}

        horizontal_edges = []
        for y in range(ny):
            row = wafer_binary[y, :]
            edges = []
            for x in range(1, nx):
                if row[x] != row[x - 1]:
                    edges.append(float(x))
            if edges:
                horizontal_edges.append(np.array(edges))
        if horizontal_edges:
            max_len = max(len(e) for e in horizontal_edges)
            edges_array = np.full((len(horizontal_edges), max_len), np.nan)
            for i, e in enumerate(horizontal_edges):
                edges_array[i, :len(e)] = e
            result['horizontal'] = edges_array

        vertical_edges = []
        for x in range(nx):
            col = wafer_binary[:, x]
            edges = []
            for y in range(1, ny):
                if col[y] != col[y - 1]:
                    edges.append(float(y))
            if edges:
                vertical_edges.append(np.array(edges))
        if vertical_edges:
            max_len = max(len(e) for e in vertical_edges)
            edges_array = np.full((len(vertical_edges), max_len), np.nan)
            for i, e in enumerate(vertical_edges):
                edges_array[i, :len(e)] = e
            result['vertical'] = edges_array

        return result

    def _process_single_realization(
        self,
        realization_id: int,
        nominal_aerial: np.ndarray,
        noise_generator: NoiseGenerator,
        target_binary: Optional[np.ndarray],
        config: MonteCarloStochasticConfig,
    ) -> SingleRealizationResult:
        """
        处理单次蒙特卡洛实现

        Args:
            realization_id: 实现ID
            nominal_aerial: 标称空间像
            noise_generator: 噪声生成器
            target_binary: 目标二值图（用于EPE计算）
            config: 仿真配置

        Returns:
            单次实现结果
        """
        t_start = time.time()

        noise = noise_generator.generate(
            shape=nominal_aerial.shape,
            pixel_size=config.pixel_size,
            nominal_intensity=nominal_aerial,
        )

        latent, resist = apply_stochastic_lithography(
            aerial_image=nominal_aerial,
            noise=noise,
            pixel_size=config.pixel_size,
            base_threshold=config.base_threshold,
        )

        wafer_binary = (resist >= 0.0).astype(np.float64)

        cd_h = compute_cd(
            wafer_binary,
            direction='horizontal',
            pixel_size=config.pixel_size,
        )
        cd_v = compute_cd(
            wafer_binary,
            direction='vertical',
            pixel_size=config.pixel_size,
        )

        epe_result = None
        if target_binary is not None:
            epe_result = compute_epe(
                wafer_binary,
                target_binary,
                pixel_size=config.pixel_size,
            )

        edge_positions = self._extract_edge_positions(wafer_binary)

        t_elapsed = time.time() - t_start

        return SingleRealizationResult(
            realization_id=realization_id,
            seed=noise.seed if noise.seed is not None else 0,
            aerial_image=nominal_aerial if config.save_intermediate else None,
            latent_image=latent if config.save_intermediate else None,
            resist_image=resist if config.save_intermediate else None,
            wafer_binary=wafer_binary if config.save_intermediate else None,
            noise=noise if config.save_intermediate else None,
            cd_horizontal=cd_h,
            cd_vertical=cd_v,
            epe=epe_result,
            edge_positions=edge_positions,
            processing_time=t_elapsed,
        )

    def _save_intermediate_result(
        self,
        result: SingleRealizationResult,
        h5file: h5py.File,
    ):
        """保存单次实现结果到HDF5文件"""
        grp = h5file.create_group(f"realization_{result.realization_id:06d}")
        grp.attrs['seed'] = result.seed
        grp.attrs['processing_time'] = result.processing_time

        if result.wafer_binary is not None:
            grp.create_dataset('wafer_binary', data=result.wafer_binary, compression='gzip')
        if result.latent_image is not None:
            grp.create_dataset('latent_image', data=result.latent_image, compression='gzip')
        if result.resist_image is not None:
            grp.create_dataset('resist_image', data=result.resist_image, compression='gzip')

        if result.cd_horizontal is not None:
            cd_h_grp = grp.create_group('cd_horizontal')
            for k, v in result.cd_horizontal.items():
                cd_h_grp.attrs[k] = v
        if result.cd_vertical is not None:
            cd_v_grp = grp.create_group('cd_vertical')
            for k, v in result.cd_vertical.items():
                cd_v_grp.attrs[k] = v

    def run_simulation(
        self,
        mask: np.ndarray,
        target: Optional[np.ndarray] = None,
        config: Optional[MonteCarloStochasticConfig] = None,
    ) -> MonteCarloStochasticResult:
        """
        运行蒙特卡洛仿真

        Args:
            mask: 掩模图案
            target: 目标图案（用于EPE计算）
            config: 仿真配置，None 则使用默认配置

        Returns:
            MonteCarloStochasticResult 完整仿真结果
        """
        if config is None:
            config = MonteCarloStochasticConfig()

        t_total_start = time.time()

        mask = mask.astype(np.float64)
        target_binary = None
        if target is not None:
            target_binary = (target.astype(np.float64) >= 0.5).astype(np.float64)

        logger.info(
            f"开始随机蒙特卡洛仿真: {config.n_realizations} 次实现, "
            f"掩模尺寸={mask.shape}"
        )

        nominal_aerial = self._compute_nominal_aerial_image(mask)

        _, nominal_resist = apply_stochastic_lithography(
            aerial_image=nominal_aerial,
            noise=NoiseRealization(
                effective_threshold=np.full_like(nominal_aerial, config.base_threshold)
            ),
            pixel_size=config.pixel_size,
            base_threshold=config.base_threshold,
        )
        nominal_wafer = (nominal_resist >= 0.0).astype(np.float64)

        noise_generator = NoiseGenerator(config.noise_config)

        h5file = None
        if config.save_intermediate and config.save_path:
            Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)
            h5file = h5py.File(config.save_path, 'w')
            h5file.attrs['n_realizations'] = config.n_realizations
            h5file.attrs['pixel_size'] = config.pixel_size
            h5file.create_dataset('nominal_aerial', data=nominal_aerial, compression='gzip')
            h5file.create_dataset('nominal_wafer', data=nominal_wafer, compression='gzip')
            h5file.create_dataset('mask', data=mask, compression='gzip')
            if target_binary is not None:
                h5file.create_dataset('target', data=target_binary, compression='gzip')

        all_cd_h = []
        all_cd_v = []
        all_cd_values = []
        all_epe = []
        all_edge_positions = []
        all_seeds = []
        all_results = []

        for batch_start in range(0, config.n_realizations, config.batch_size):
            batch_end = min(batch_start + config.batch_size, config.n_realizations)
            batch_size = batch_end - batch_start

            for i in range(batch_size):
                realization_id = batch_start + i

                result = self._process_single_realization(
                    realization_id=realization_id,
                    nominal_aerial=nominal_aerial,
                    noise_generator=noise_generator,
                    target_binary=target_binary,
                    config=config,
                )

                all_cd_h.append(result.cd_horizontal['cd_mean'])
                all_cd_v.append(result.cd_vertical['cd_mean'])
                all_cd_values.append(result.cd_horizontal.get('cd_mean', 0.0))
                all_seeds.append(result.seed)
                all_edge_positions.append(result.edge_positions)
                all_results.append(result)

                if target_binary is not None and result.epe is not None:
                    all_epe.append(result.epe['epe_mean'])

                if h5file is not None:
                    self._save_intermediate_result(result, h5file)

                if config.progress_callback is not None:
                    elapsed = time.time() - t_total_start
                    config.progress_callback(realization_id + 1, config.n_realizations, elapsed)

            logger.debug(
                f"批次完成: {batch_end}/{config.n_realizations}, "
                f"已耗时 {time.time() - t_total_start:.1f}s"
            )

        total_time = time.time() - t_total_start

        if h5file is not None:
            h5file.close()

        result = MonteCarloStochasticResult(
            config=config,
            n_realizations=config.n_realizations,
            nominal_aerial_image=nominal_aerial,
            nominal_wafer_binary=nominal_wafer,
            all_cd_values=np.array(all_cd_values) if all_cd_values else None,
            all_cd_horizontal=np.array(all_cd_h) if all_cd_h else None,
            all_cd_vertical=np.array(all_cd_v) if all_cd_v else None,
            all_epe_values=np.array(all_epe) if all_epe else None,
            all_edge_positions=all_edge_positions,
            all_noise_seeds=all_seeds,
            realization_results=all_results,
            total_time=total_time,
            intermediate_file=config.save_path if config.save_intermediate else None,
        )

        logger.info(
            f"随机蒙特卡洛仿真完成: {config.n_realizations} 次实现, "
            f"总耗时 {total_time:.1f}s"
        )

        return result


def run_stochastic_monte_carlo(
    mask: np.ndarray,
    optical_system: OpticalSystem,
    n_realizations: int = 100,
    noise_config: Optional[NoiseConfig] = None,
    target: Optional[np.ndarray] = None,
    pixel_size: float = 1.0,
    base_threshold: float = 0.5,
    random_seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    **kwargs,
) -> MonteCarloStochasticResult:
    """
    便捷函数：运行随机蒙特卡洛仿真

    Args:
        mask: 掩模图案
        optical_system: 光学系统参数
        n_realizations: 实现次数
        noise_config: 噪声配置
        target: 目标图案
        pixel_size: 像素尺寸 (nm)
        base_threshold: 基础显影阈值
        random_seed: 随机种子
        progress_callback: 进度回调
        **kwargs: 传递给 StochasticMonteCarloSimulator 的额外参数

    Returns:
        MonteCarloStochasticResult 仿真结果
    """
    config = MonteCarloStochasticConfig(
        n_realizations=n_realizations,
        noise_config=noise_config,
        pixel_size=pixel_size,
        base_threshold=base_threshold,
        random_seed=random_seed,
        progress_callback=progress_callback,
    )

    simulator = StochasticMonteCarloSimulator(
        optical_system=optical_system,
        **kwargs,
    )

    return simulator.run_simulation(mask=mask, target=target, config=config)
