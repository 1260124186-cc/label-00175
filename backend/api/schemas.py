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
    technology_node: str = Field(
        "duv_arf",
        description="技术节点: duv_arf (ArF深紫外), euv (极紫外)"
    )
    flare: float = Field(0.0, ge=0, le=1, description="Flare系数(0~1)，EUV系统杂散光比例")
    shadowing_model: str = Field(
        "none",
        description="阴影效应模型: none, approximate, rigorous"
    )
    reflective_mask_attenuation: float = Field(
        0.0, ge=0, le=1, description="反射式掩模衰减因子(0~1)，EUV特有"
    )
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

    @field_validator("technology_node")
    @classmethod
    def validate_technology_node(cls, v):
        valid = ["duv_arf", "euv"]
        if v not in valid:
            raise ValueError(f"技术节点必须为以下之一: {valid}")
        return v

    @field_validator("shadowing_model")
    @classmethod
    def validate_shadowing_model(cls, v):
        valid = ["none", "approximate", "rigorous"]
        if v not in valid:
            raise ValueError(f"阴影效应模型必须为以下之一: {valid}")
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


class TaskSubmitResponse(BaseModel):
    success: bool
    message: str
    task_id: str
    task_type: str
    status: str = "pending"


class TaskStatusResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    progress: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result_summary: Optional[Dict[str, Any]] = None


class TaskListResponse(BaseModel):
    count: int
    tasks: List[TaskStatusResponse]


class OPCConfigParams(BaseModel):
    epe_threshold: float = Field(3.0, gt=0, description="EPE 热点判定阈值 (nm)")
    epe_convergence_threshold: float = Field(1.0, gt=0, description="EPE 收敛阈值 (nm)")
    max_iterations: int = Field(10, gt=0, description="最大迭代次数")
    min_hotspot_area: int = Field(4, gt=0, description="最小热点区域面积（像素）")
    hotspot_dilation: int = Field(2, ge=0, description="热点区域膨胀像素数")
    edge_offset_step: float = Field(0.5, gt=0, description="每次边缘偏移步长（像素）")
    max_edge_offset: float = Field(3.0, gt=0, description="最大边缘偏移量（像素）")
    corner_bias_size: float = Field(1.0, gt=0, description="拐角 serif 尺寸（像素）")
    line_end_extension: float = Field(2.0, gt=0, description="线端延伸长度（像素）")
    line_end_width: float = Field(2.0, gt=0, description="线端延伸宽度（像素）")
    sraf_enable: bool = Field(True, description="是否启用 SRAF 插入")
    sraf_min_distance: float = Field(2.0, gt=0, description="SRAF 与主特征最小间距（像素）")
    sraf_max_distance: float = Field(5.0, gt=0, description="SRAF 与主特征最大间距（像素）")
    sraf_width: float = Field(1.0, gt=0, description="SRAF 宽度（像素）")
    sraf_length: float = Field(4.0, gt=0, description="SRAF 长度（像素）")
    sraf_spacing: float = Field(2.0, gt=0, description="相邻 SRAF 间距（像素）")
    sraf_min_feature_size: float = Field(1.0, gt=0, description="SRAF 最小尺寸（像素）")
    sraf_max_aspect_ratio: float = Field(10.0, gt=0, description="SRAF 最大长宽比")
    optimizer_enable: bool = Field(True, description="是否启用 MaskOptimizer 精细优化")
    optimizer_max_iter: int = Field(20, gt=0, description="优化器每轮最大迭代次数")
    optimizer_learning_rate: float = Field(0.05, gt=0, description="优化器学习率")
    optimizer_epe_weight: float = Field(1.0, ge=0, description="优化器 EPE 损失权重")
    wafer_threshold: float = Field(0.3, ge=0, le=1, description="晶圆成像二值化阈值")
    verbose: bool = Field(True, description="是否输出详细日志")


class OPCRunRequest(BaseModel):
    optical_system: OpticalSystem = Field(default_factory=OpticalSystem)
    opc_config: OPCConfigParams = Field(default_factory=OPCConfigParams)
    pattern_type: str = Field("rectangle", description="测试图案类型")
    pattern_params: Dict[str, Any] = Field(
        default_factory=lambda: {"size": [64, 64], "x_start": 20, "x_end": 44, "y_start": 20, "y_end": 44}
    )
    gds_file_id: Optional[str] = Field(None, description="已上传 GDS 的 file_id，提供则忽略图案参数")
    gds_layer: Optional[int] = Field(None, description="GDS 层号，gds_file_id 存在时必填")
    gds_datatype: Optional[int] = Field(0, description="GDS 数据类型号，默认 0")
    gds_pixel_size: Optional[float] = Field(1.0, description="栅格化像素大小 (GDS 单位/像素)")
    gds_target_size: Optional[List[int]] = Field(None, description="强制输出尺寸 [H, W]")


class SourceConstraintsParams(BaseModel):
    energy_conservation: bool = Field(True, description="是否启用总能量守恒约束")
    energy_target: float = Field(1.0, gt=0, description="目标总能量")
    sigma_target: Optional[float] = Field(None, description="目标等效sigma值")
    sigma_tolerance: float = Field(0.02, ge=0, description="sigma约束容差")
    smoothness_weight: float = Field(0.01, ge=0, description="平滑正则化权重")
    smoothness_type: str = Field("tv", description="平滑类型 'tv' 或 'gaussian'")
    gaussian_sigma: float = Field(1.5, gt=0, description="高斯平滑 sigma")
    non_negative: bool = Field(True, description="是否强制光源强度非负")
    support_radius: Optional[float] = Field(None, description="光源最大支持半径")
    support_radius_inner: Optional[float] = Field(None, description="光源最小内半径")


class SMOConfigParams(BaseModel):
    strategy: str = Field("alternating", description="优化策略: alternating, joint_gradient, source_first")
    max_outer_iterations: int = Field(20, gt=0, description="外层交替优化最大迭代次数")
    source_max_iter: int = Field(50, gt=0, description="每轮光源优化最大迭代次数")
    mask_max_iter: int = Field(100, gt=0, description="每轮掩模优化最大迭代次数")
    joint_max_iter: int = Field(200, gt=0, description="联合梯度下降最大迭代次数")
    source_learning_rate: float = Field(0.005, gt=0, description="光源优化学习率")
    mask_learning_rate: float = Field(0.01, gt=0, description="掩模优化学习率")
    joint_learning_rate_source: float = Field(0.003, gt=0, description="联合优化时光源学习率")
    joint_learning_rate_mask: float = Field(0.008, gt=0, description="联合优化时掩模学习率")
    tol: float = Field(1e-5, gt=0, description="收敛容差")
    convergence_patience: int = Field(5, ge=0, description="收敛耐心值")
    source_init_type: str = Field("conventional", description="光源初始化类型")
    source_constraints: SourceConstraintsParams = Field(default_factory=SourceConstraintsParams)
    wafer_threshold: float = Field(0.3, ge=0, le=1, description="晶圆成像二值化阈值")
    use_wafer_image_loss: bool = Field(True, description="是否使用wafer图像计算损失")
    pvb_weight: float = Field(0.0, ge=0, description="工艺变化带宽损失权重")
    verbose: bool = Field(True, description="是否输出详细日志")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        valid = ["alternating", "joint_gradient", "source_first"]
        if v not in valid:
            raise ValueError(f"优化策略必须为以下之一: {valid}")
        return v

    @field_validator("source_init_type")
    @classmethod
    def validate_source_init(cls, v):
        valid = ["conventional", "annular", "dipole", "quasar", "uniform_disk", "random", "custom"]
        if v not in valid:
            raise ValueError(f"光源初始化类型必须为以下之一: {valid}")
        return v


class SMORunRequest(BaseModel):
    optical_system: OpticalSystem = Field(default_factory=OpticalSystem)
    smo_config: SMOConfigParams = Field(default_factory=SMOConfigParams)
    pattern_type: str = Field("rectangle", description="测试图案类型")
    pattern_params: Dict[str, Any] = Field(
        default_factory=lambda: {"size": [64, 64], "x_start": 20, "x_end": 44, "y_start": 20, "y_end": 44}
    )
    gds_file_id: Optional[str] = Field(None, description="已上传 GDS 的 file_id，提供则忽略图案参数")
    gds_layer: Optional[int] = Field(None, description="GDS 层号，gds_file_id 存在时必填")
    gds_datatype: Optional[int] = Field(0, description="GDS 数据类型号，默认 0")
    gds_pixel_size: Optional[float] = Field(1.0, description="栅格化像素大小 (GDS 单位/像素)")
    gds_target_size: Optional[List[int]] = Field(None, description="强制输出尺寸 [H, W]")


class ILTComplexityParams(BaseModel):
    perimeter_weight: float = Field(0.0, ge=0, description="掩模周长惩罚权重")
    vertex_weight: float = Field(0.0, ge=0, description="顶点数惩罚权重")
    sub_feature_weight: float = Field(0.0, ge=0, description="辅助特征数量惩罚权重")
    sub_feature_min_area: int = Field(2, gt=0, description="辅助特征最小面积阈值")
    sub_feature_max_area: int = Field(100, gt=0, description="辅助特征最大面积阈值")


class ILTConfigParams(BaseModel):
    max_iter: int = Field(200, gt=0, description="最大迭代次数")
    learning_rate: float = Field(0.01, gt=0, description="学习率")
    optimizer_type: str = Field("adam_projection", description="优化器类型")
    convergence_tol: float = Field(1e-6, gt=0, description="收敛容差")
    convergence_patience: int = Field(20, ge=0, description="收敛耐心值")
    transmission_level: str = Field("continuous", description="离散透射率等级: binary, ternary, continuous")
    quantization_start_iter: int = Field(100, ge=0, description="开始量化的迭代数")
    quantization_schedule: str = Field("linear", description="量化调度类型: step, linear, cosine")
    quantization_strength: float = Field(1.0, ge=0, le=1, description="量化强度")
    resist_steepness: float = Field(50.0, gt=0, description="soft resist sigmoid 陡度参数 k")
    wafer_threshold: float = Field(0.3, ge=0, le=1, description="光刻胶阈值")
    l2_wafer_weight: float = Field(1.0, ge=0, description="晶圆图 L2 损失权重")
    complexity: ILTComplexityParams = Field(default_factory=ILTComplexityParams)
    binary_penalty_weight: float = Field(0.0, ge=0, description="二值化惩罚权重")
    tv_smooth_weight: float = Field(0.0, ge=0, description="TV 平滑权重")
    verbose: bool = Field(True, description="是否输出详细日志")

    @field_validator("optimizer_type")
    @classmethod
    def validate_ilt_optimizer(cls, v):
        valid = ["gradient_projection", "adam_projection", "sgd_projection"]
        if v not in valid:
            raise ValueError(f"ILT优化器类型必须为以下之一: {valid}")
        return v

    @field_validator("transmission_level")
    @classmethod
    def validate_transmission(cls, v):
        valid = ["binary", "ternary", "continuous"]
        if v not in valid:
            raise ValueError(f"透射率等级必须为以下之一: {valid}")
        return v

    @field_validator("quantization_schedule")
    @classmethod
    def validate_quant_schedule(cls, v):
        valid = ["step", "linear", "cosine"]
        if v not in valid:
            raise ValueError(f"量化调度必须为以下之一: {valid}")
        return v


class ILTRunRequest(BaseModel):
    optical_system: OpticalSystem = Field(default_factory=OpticalSystem)
    ilt_config: ILTConfigParams = Field(default_factory=ILTConfigParams)
    pattern_type: str = Field("rectangle", description="测试图案类型")
    pattern_params: Dict[str, Any] = Field(
        default_factory=lambda: {"size": [64, 64], "x_start": 20, "x_end": 44, "y_start": 20, "y_end": 44}
    )
    gds_file_id: Optional[str] = Field(None, description="已上传 GDS 的 file_id，提供则忽略图案参数")
    gds_layer: Optional[int] = Field(None, description="GDS 层号，gds_file_id 存在时必填")
    gds_datatype: Optional[int] = Field(0, description="GDS 数据类型号，默认 0")
    gds_pixel_size: Optional[float] = Field(1.0, description="栅格化像素大小 (GDS 单位/像素)")
    gds_target_size: Optional[List[int]] = Field(None, description="强制输出尺寸 [H, W]")


class ProcessWindowRunRequest(BaseModel):
    optical_system: OpticalSystem = Field(default_factory=OpticalSystem)
    pattern_type: str = Field("rectangle", description="测试图案类型")
    pattern_params: Dict[str, Any] = Field(
        default_factory=lambda: {"size": [64, 64], "x_start": 20, "x_end": 44, "y_start": 20, "y_end": 44}
    )
    gds_file_id: Optional[str] = Field(None, description="已上传 GDS 的 file_id，提供则忽略图案参数")
    gds_layer: Optional[int] = Field(None, description="GDS 层号，gds_file_id 存在时必填")
    gds_datatype: Optional[int] = Field(0, description="GDS 数据类型号，默认 0")
    gds_pixel_size: Optional[float] = Field(1.0, description="栅格化像素大小 (GDS 单位/像素)")
    gds_target_size: Optional[List[int]] = Field(None, description="强制输出尺寸 [H, W]")
    focus_range: List[float] = Field(
        [-150.0, 150.0, 11],
        description="离焦量扫描范围 [start, stop, num_points]"
    )
    dose_range: List[float] = Field(
        [0.85, 1.15, 11],
        description="曝光剂量扫描范围 [start, stop, num_points]"
    )
    cd_tolerance: float = Field(0.1, gt=0, description="CD 相对容差")
    epe_tolerance: Optional[float] = Field(None, gt=0, description="EPE 绝对容差 (nm)，None 则不检查 EPE")
    threshold: float = Field(0.3, ge=0, le=1, description="光刻胶阈值")
    save_visualizations: bool = Field(False, description="是否保存可视化图片")


class BatchOptimizationRequest(BaseModel):
    source: str = Field(..., description="GDS文件路径或目录路径")
    layer: Optional[int] = Field(None, description="GDS 层号")
    optical_system: OpticalSystem = Field(default_factory=OpticalSystem)
    optimization: Optimization = Field(default_factory=Optimization)
    max_workers: Optional[int] = Field(None, description="最大并发 worker 数")
    max_retries: int = Field(2, ge=0, description="失败重试次数")
    save_optimized_masks: bool = Field(True, description="是否保存每个 cell 优化后的掩模")
    output_dir: Optional[str] = Field(None, description="输出目录")
    stop_on_first_failure: bool = Field(False, description="是否遇到第一个失败就停止整批")


class GdsLayerInfo(BaseModel):
    layer: int
    datatype: int = 0


class GdsFileInfo(BaseModel):
    file_id: str
    filename: str
    size: int
    uploaded_at: float


class GdsUploadResponse(BaseModel):
    success: bool = True
    message: str = "上传成功"
    file: Optional[GdsFileInfo] = None


class GdsListResponse(BaseModel):
    count: int
    files: List[GdsFileInfo]


class GdsLayersResponse(BaseModel):
    file_id: str
    layers: List[GdsLayerInfo]
    cells: List[str]
    layer_count: int
    cell_count: int


class BatchSubTask(BaseModel):
    task_id: Optional[str] = None
    cell_name: str
    status: str
    initial_mse: Optional[float] = None
    final_mse: Optional[float] = None
    initial_ssim: Optional[float] = None
    final_ssim: Optional[float] = None
    iterations: Optional[int] = None
    converged: Optional[bool] = None
    elapsed_sec: Optional[float] = None
    error_message: Optional[str] = None


class ProcessWindowDetail(BaseModel):
    focus_values: Optional[List[float]] = None
    dose_values: Optional[List[float]] = None
    cd_matrix: Optional[List[List[float]]] = None
    cd_error_matrix: Optional[List[List[float]]] = None
    epe_matrix: Optional[List[List[float]]] = None
    mse_matrix: Optional[List[List[float]]] = None
    ssim_matrix: Optional[List[List[float]]] = None
    printability_mask: Optional[List[List[bool]]] = None
    best_focus: Optional[float] = None
    best_dose: Optional[float] = None
    nominal_cd: Optional[float] = None
    ellipse_approx: Optional[Dict[str, Any]] = None
    rect_approx: Optional[Dict[str, Any]] = None


class BatchResultDetail(BaseModel):
    sub_tasks: Optional[List[BatchSubTask]] = None
    total_sub_tasks: Optional[int] = None


class TaskResultResponse(BaseModel):
    task_id: str
    task_type: Optional[str] = None
    status: str
    result: Optional[Dict[str, Any]] = None
    result_summary: Optional[Dict[str, Any]] = None
    result_detail: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MetricHistoryPoint(BaseModel):
    step: Optional[int] = None
    value: float
    timestamp: float


class ExperimentRunSummary(BaseModel):
    run_id: str
    experiment_name: str
    status: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: float
    tags: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    metrics_summary: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class ExperimentRunDetail(ExperimentRunSummary):
    metrics: Dict[str, List[MetricHistoryPoint]] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)


class ExperimentListResponse(BaseModel):
    count: int
    experiments: List[str]


class ExperimentRunListResponse(BaseModel):
    count: int
    runs: List[ExperimentRunSummary]


class ExperimentCompareRequest(BaseModel):
    run_ids: List[str]
    metrics: Optional[List[str]] = None
    params: Optional[List[str]] = None


class MetricCompareItem(BaseModel):
    final: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    first: Optional[float] = None


class ExperimentCompareRun(BaseModel):
    run_id: str
    status: str
    duration_seconds: float
    tags: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, MetricCompareItem] = Field(default_factory=dict)


class ExperimentCompareResponse(BaseModel):
    experiment_name: Optional[str] = None
    compared_run_ids: List[str]
    runs: List[ExperimentCompareRun]
    all_metric_names: List[str] = Field(default_factory=list)
    all_param_names: List[str] = Field(default_factory=list)


class MetricCurveResponse(BaseModel):
    run_id: str
    metric_name: str
    points: List[MetricHistoryPoint]


class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, description="用户名")
    password: str = Field(..., min_length=6, max_length=64, description="密码")
    display_name: Optional[str] = Field(None, max_length=64, description="显示名称")


class UserLoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class UserInfoResponse(BaseModel):
    user_id: str
    username: str
    display_name: str
    created_at: Optional[str] = None


class TokenResponse(BaseModel):
    success: bool = True
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token 有效期（秒）")
    user: UserInfoResponse
