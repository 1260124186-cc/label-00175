from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator


class SourceParams(BaseModel):
    sigma_inner: float = Field(0.0, ge=0, le=1, description="内环sigma")
    sigma_outer: float = Field(0.75, gt=0, le=1, description="外环sigma")
    angle: Optional[float] = Field(None, description="极角(度)")
    opening_angle: Optional[float] = Field(None, gt=0, le=180, description="开口角(度)")


class OpticalSystem(BaseModel):
    wavelength: float = Field(193.0, gt=0, description="光源波长(nm)")
    na: float = Field(1.35, gt=0, le=2, description="数值孔径")
    sigma: float = Field(0.75, ge=0, le=1, description="部分相干因子")
    pixel_size: float = Field(1.0, gt=0, description="像素尺寸(nm)")
    defocus: float = Field(0.0, description="离焦量(nm)")
    magnification: float = Field(4.0, gt=0, description="放大倍率")
    illumination_type: str = Field(
        "conventional",
        description="照明模式: conventional, annular, dipole, quasar, custom"
    )
    source_params: SourceParams = Field(default_factory=SourceParams)
    tcc_mode: str = Field("socs", description="TCC模式: full_tcc, socs, kernel_2d")
    socs_num_terms: int = Field(5, gt=0, description="SOCS分解项数")
    use_socs: bool = Field(True, description="[已弃用]是否使用SOCS")
    zernike_coefficients: Dict[str, float] = Field(
        default_factory=dict,
        description="Zernike像差系数(单位:波长λ)"
    )

    @field_validator("illumination_type")
    @classmethod
    def validate_illumination_type(cls, v):
        valid = ["conventional", "annular", "dipole", "quasar", "custom"]
        if v not in valid:
            raise ValueError(f"照明模式必须为以下之一: {valid}")
        return v

    @field_validator("tcc_mode")
    @classmethod
    def validate_tcc_mode(cls, v):
        valid = ["full_tcc", "socs", "kernel_2d"]
        if v not in valid:
            raise ValueError(f"TCC模式必须为以下之一: {valid}")
        return v


class SpatialWeight(BaseModel):
    enable: bool = Field(False, description="是否启用空间加权")
    edge_weight: float = Field(2.0, gt=0, description="边缘区域权重倍率")
    corner_weight: float = Field(5.0, gt=0, description="拐角区域权重倍率")
    line_end_weight: float = Field(4.0, gt=0, description="线端区域权重倍率")
    base_weight: float = Field(1.0, gt=0, description="基础区域权重")
    edge_sigma: float = Field(1.0, gt=0, description="边缘检测高斯sigma")
    corner_threshold: float = Field(0.3, ge=0, le=1, description="拐角检测阈值")
    line_end_threshold: float = Field(0.5, ge=0, le=1, description="线端检测阈值")
    weight_erosion: bool = Field(True, description="是否形态学腐蚀权重")
    smooth_sigma: float = Field(0.5, ge=0, description="权重mask高斯平滑sigma")
    normalize: bool = Field(True, description="是否归一化权重到均值为1")


class LossWeights(BaseModel):
    mse: float = Field(1.0, ge=0, description="MSE权重")
    ssim: float = Field(0.0, ge=0, description="(1-SSIM)权重")
    pvb: float = Field(0.0, ge=0, description="PVB权重")
    mask_complexity: float = Field(0.0, ge=0, description="掩模复杂度TV权重")
    weighted_mse: float = Field(0.0, ge=0, description="空间加权MSE权重")
    weighted_mae: float = Field(0.0, ge=0, description="空间加权MAE权重")


class Regularization(BaseModel):
    type: Optional[str] = Field(None, description="正则化类型: null, l1, l2, tv")
    strength: float = Field(0.0, ge=0, description="正则化强度")

    @field_validator("type")
    @classmethod
    def validate_reg_type(cls, v):
        if v is None:
            return v
        valid = ["l1", "l2", "tv", "none", "None"]
        if v not in valid:
            raise ValueError(f"正则化类型必须为以下之一: None, {valid}")
        return v if v not in ["none", "None"] else None


class Optimization(BaseModel):
    optimizer_type: str = Field(
        "gradient_descent",
        description="优化器类型: gradient_descent, bfgs, newton, genetic, pso, rl"
    )
    max_iter: int = Field(100, gt=0, description="最大迭代次数")
    learning_rate: float = Field(0.01, gt=0, description="学习率")
    tol: float = Field(1e-6, gt=0, description="收敛容差")
    early_stop_patience: int = Field(10, ge=0, description="早停耐心值")
    lr_scheduler: Optional[str] = Field(
        None,
        description="学习率调度器: step, exponential, cosine, null"
    )
    lr_decay: float = Field(0.95, gt=0, lt=1, description="学习率衰减率")
    lr_step_size: int = Field(20, gt=0, description="学习率调度步长")
    metric: str = Field("mse", description="优化目标指标: mse, mae, ssim")
    use_composite_loss: bool = Field(False, description="是否启用复合损失")
    loss_weights: LossWeights = Field(default_factory=LossWeights)
    spatial_weight: SpatialWeight = Field(default_factory=SpatialWeight)
    regularization: Regularization = Field(default_factory=Regularization)
    bounds: List[float] = Field([0.0, 1.0], description="掩模值边界[min, max]")
    verbose: bool = Field(True, description="是否输出详细信息")
    random_seed: Optional[int] = Field(42, description="随机种子")
    population_size: int = Field(50, gt=0, description="种群大小")
    crossover_rate: float = Field(0.8, ge=0, le=1, description="交叉概率")
    mutation_rate: float = Field(0.1, ge=0, le=1, description="变异概率")
    n_jobs: int = Field(1, description="并行工作进程数")
    rl_gamma: float = Field(0.99, ge=0, le=1, description="RL折扣因子")
    rl_epsilon: float = Field(0.1, ge=0, le=1, description="RL初始探索率")
    rl_epsilon_decay: float = Field(0.995, ge=0, le=1, description="RL探索率衰减")

    @field_validator("optimizer_type")
    @classmethod
    def validate_optimizer(cls, v):
        valid = ["gradient_descent", "bfgs", "newton", "genetic", "pso", "rl"]
        if v not in valid:
            raise ValueError(f"优化器类型必须为以下之一: {valid}")
        return v

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, v):
        if len(v) != 2:
            raise ValueError("bounds必须包含两个元素[min, max]")
        if v[0] >= v[1]:
            raise ValueError("bounds[0]必须小于bounds[1]")
        return v


class OutputConfig(BaseModel):
    save_dir: str = Field("./results", description="结果保存目录")
    save_images: bool = Field(True, description="是否保存图像")
    save_history: bool = Field(True, description="是否保存历史数据")
    image_format: str = Field("png", description="图像格式: png, tiff")
    log_level: str = Field("INFO", description="日志级别: DEBUG, INFO, WARNING, ERROR")


class ImagingConfig(BaseModel):
    resist_threshold: float = Field(0.3, ge=0, le=1, description="光刻胶阈值")
    apply_resist: bool = Field(True, description="是否应用光刻胶响应")


class SimulationConfig(BaseModel):
    optical_system: OpticalSystem = Field(default_factory=OpticalSystem)
    optimization: Optimization = Field(default_factory=Optimization)
    output: OutputConfig = Field(default_factory=OutputConfig)
    imaging: ImagingConfig = Field(default_factory=ImagingConfig)


class ConfigResponse(BaseModel):
    success: bool
    config: SimulationConfig
    message: Optional[str] = None


class SaveConfigRequest(BaseModel):
    config: SimulationConfig
    filename: Optional[str] = None


class SaveConfigResponse(BaseModel):
    success: bool
    message: str
    saved_path: Optional[str] = None


class SimulationRunRequest(BaseModel):
    config: SimulationConfig
    pattern_type: str = Field("rectangle", description="测试图案类型")
    pattern_params: Dict[str, Any] = Field(
        default_factory=lambda: {"size": [64, 64], "x_start": 20, "x_end": 44, "y_start": 20, "y_end": 44}
    )


class SimulationRunResponse(BaseModel):
    success: bool
    message: str
    task_id: Optional[str] = None
    status: Optional[str] = None
