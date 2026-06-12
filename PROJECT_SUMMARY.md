# 计算光刻与版图优化仿真框架 - 项目说明

## 一、项目概述

本项目是一个基于 Python 的计算光刻与版图优化仿真框架，为博士研究生提供计算光刻研究中的算法开发与验证环境。框架实现了从掩模图案光学成像到优化算法迭代的完整仿真流程，支持多种经典与启发式优化算法。

---

## 二、核心功能

### 2.1 光学成像建模

基于 **Hopkins 部分相干成像理论** 构建光刻系统的光学成像仿真：

| 功能 | 说明 | 代码位置 |
|------|------|----------|
| 光学系统配置 | 波长(193nm ArF)、数值孔径(NA=1.35高NA浸没式)、部分相干因子、离焦量、像素尺寸等参数配置 | [imaging.py:OpticalSystem](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/imaging.py#L19-L47) |
| 光瞳函数计算 | 包含离焦相位的复数光瞳函数，使用 Numba 并行加速 | [imaging.py:_compute_pupil_function](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/imaging.py#L49-L77) |
| TCC 传输交叉系数 | 简化的圆形光源 TCC 核计算，用于部分相干成像 | [imaging.py:_compute_tcc_kernel](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/imaging.py#L80-L113) |
| 空间像计算 | 掩模→频谱→光瞳滤波→逆FFT→光强分布的完整成像流程 | [imaging.py:compute_aerial_image](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/imaging.py#L165-L191) |
| 成像梯度计算 | 空间像对掩模的解析梯度，支持基于梯度的优化 | [imaging.py:compute_image_gradient](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/imaging.py#L193-L213) |
| 光刻胶响应 | 阈值二值化模拟光刻胶显影过程 | [imaging.py:_apply_threshold](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/imaging.py#L216-L236) |

### 2.2 傅里叶变换与频域处理

| 功能 | 说明 | 代码位置 |
|------|------|----------|
| 1D/2D FFT/IFFT | 基于 scipy.fft 的快速傅里叶变换封装，支持频谱中心化与归一化 | [fft.py:fft2d/ifft2d](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/fft.py#L183-L230) |
| FFT Shift | Numba JIT 加速的 1D/2D 频谱移位操作 | [fft.py:_fftshift_2d](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/fft.py#L63-L84) |
| 频域滤波器 | 巴特沃斯低通/高通/带通/带阻滤波器 | [fft.py:frequency_filter](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/fft.py#L266-L323) |
| 相位调制 | 线性相位(空域平移)、二次相位(离焦)、自定义相位 | [fft.py:phase_modulation](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/fft.py#L349-L395) |
| 功率谱分析 | 对数尺度功率谱计算与频率坐标生成 | [fft.py:compute_power_spectrum](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/fft.py#L398-L414) |

### 2.3 掩模优化算法

框架提供三类优化算法，覆盖传统梯度法、启发式算法与强化学习方法：

#### 传统优化算法

| 算法 | 特点 | 代码位置 |
|------|------|----------|
| **梯度下降 (Gradient Descent)** | 支持动量(Momentum)、回溯线搜索(Backtracking Line Search)、数值梯度/解析梯度 | [optimizer.py:GradientDescentOptimizer](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/algorithms/optimizer.py#L98-L232) |
| **BFGS 拟牛顿法** | 基于 scipy.optimize.minimize 实现，支持 L-BFGS-B 边界约束 | [optimizer.py:BFGSOptimizer](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/algorithms/optimizer.py#L386-L447) |
| **牛顿法 (Newton)** | 使用数值 Hessian 矩阵，带正则化确保正定性 | [optimizer.py:NewtonOptimizer](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/algorithms/optimizer.py#L235-L383) |

#### 启发式优化算法

| 算法 | 特点 | 代码位置 |
|------|------|----------|
| **遗传算法 (GA)** | 轮盘赌选择、均匀交叉、高斯变异、精英保留策略 | [advanced_optimizer.py:GeneticAlgorithmOptimizer](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/algorithms/advanced_optimizer.py#L82-L217) |
| **粒子群优化 (PSO)** | 标准 PSO 算法，含惯性权重 w、认知系数 c1、社会系数 c2，速度限制 | [advanced_optimizer.py:ParticleSwarmOptimizer](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/algorithms/advanced_optimizer.py#L220-L319) |

#### 强化学习优化

| 算法 | 特点 | 代码位置 |
|------|------|----------|
| **强化学习优化器** | 基于 epsilon-greedy 策略，经验回放缓冲区，支持自定义 RL 模型接入 | [advanced_optimizer.py:ReinforcementLearningOptimizer](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/algorithms/advanced_optimizer.py#L322-L531) |
| **Q-Learning 模型示例** | 简单线性 Q 函数示例，演示如何接入自定义模型 | [advanced_optimizer.py:SimpleQLearningModel](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/algorithms/advanced_optimizer.py#L534-L582) |

### 2.4 误差评估指标

| 指标 | 说明 | 代码位置 |
|------|------|----------|
| **MSE** (均方误差) | 像素级平方误差的均值 | [metrics.py:mse](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/metrics.py#L18-L31) |
| **MAE** (平均绝对误差) | 像素级绝对误差的均值 | [metrics.py:mae](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/metrics.py#L34-L47) |
| **SSIM** (结构相似性) | 基于局部统计量(均值/方差/协方差)的结构相似性指数，范围 [-1, 1] | [metrics.py:ssim](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/metrics.py#L142-L180) |
| **NCC** (归一化相关系数) | Pearson 相关系数，衡量线性相关性 | [metrics.py:normalized_correlation](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/metrics.py#L183-L209) |
| **PSNR** (峰值信噪比) | 基于 MSE 的对数信噪比指标 (dB) | [metrics.py:psnr](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/core/metrics.py#L212-L231) |

### 2.5 辅助功能

- **学习率调度器**：阶梯衰减(Step)、指数衰减(Exponential)、余弦退火(Cosine) — [mask_optimizer.py:LearningRateScheduler](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/algorithms/mask_optimizer.py#L64-L114)
- **早停机制**：连续 N 次无改善自动停止 — [mask_optimizer.py:EarlyStopping](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/algorithms/mask_optimizer.py#L117-L152)
- **配置管理**：YAML/JSON 配置文件加载、验证、保存 — [utils/config.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/utils/config.py)
- **结果可视化**：掩模图、频域图、晶圆成像、收敛曲线、对比汇总图 — [utils/visualization.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00175/backend/utils/visualization.py)

---

## 三、算法实现原理

### 3.1 光学成像：Hopkins 部分相干成像

成像流程遵循以下数学模型：

```
掩模 M(x,y) → FFT → 频谱 M̂(fx,fy)
                    ↓
            × 光瞳函数 P(fx,fy)  [含离焦相位: exp(jπ·defocus·λ·ρ²)]
                    ↓
            IFFT → 电场 E(x,y)
                    ↓
            光强 I = |E|² → 归一化 → 空间像
                    ↓
            光刻胶阈值 → 晶圆成像
```

梯度推导（链式法则）：
```
dL/dM = dL/dI × dI/dM
其中 dI/dM = 2·Re(E* · dE/dM)，dE/dM 通过光瞳逆 FFT 获得
```

### 3.2 优化算法分类体系

```
优化算法
├── 基于梯度的方法 (需目标函数可导)
│   ├── 梯度下降 (GD + Momentum + Line Search)
│   ├── 拟牛顿法 (BFGS / L-BFGS-B)
│   └── 牛顿法 (Hessian + 正则化)
│
├── 启发式方法 (无梯度需求)
│   ├── 遗传算法
│   │   ├── 选择: 轮盘赌选择 (适应度比例)
│   │   ├── 交叉: 均匀交叉 (Uniform Crossover)
│   │   ├── 变异: 高斯变异 + 边界裁剪
│   │   └── 精英保留: 最优个体直接进入下一代
│   │
│   └── 粒子群优化
│       ├── 速度更新: v = w·v + c1·r1·(pbest-x) + c2·r2·(gbest-x)
│       └── 位置更新: x = x + v
│
└── 强化学习方法
    ├── 状态: [掩模展平, 绝对误差]
    ├── 动作: 掩模像素调整量
    ├── 奖励: 损失减少量 × 100
    └── 策略: ε-greedy + 经验回放
```

### 3.3 性能优化技术

| 技术 | 应用场景 |
|------|----------|
| **Numba JIT** | 核心计算函数（光瞳函数、TCC、局部统计、FFT移位）使用 `@jit(nopython=True, parallel=True)` 加速 |
| **并行计算** | prange 显式并行循环，利用多核 CPU |
| **缓存编译** | `cache=True` 避免重复编译 |
| **解析梯度** | 避免有限差分数值梯度的 O(N) 额外开销 |
| **scipy 集成** | BFGS/线搜索等使用成熟的 scipy.optimize 实现 |

---

## 四、项目架构

### 4.1 目录结构

```
label-00175/
├── backend/
│   ├── core/                          # 【核心物理层】物理建模与数值计算
│   │   ├── imaging.py                 #   光学成像：Hopkins部分相干成像模型
│   │   ├── fft.py                     #   傅里叶变换：FFT/IFFT、频域滤波、相位调制
│   │   └── metrics.py                 #   评估指标：MSE/MAE/SSIM/NCC/PSNR
│   │
│   ├── algorithms/                    # 【算法层】优化算法实现
│   │   ├── optimizer.py               #   基础优化器：梯度下降、BFGS、牛顿法
│   │   ├── advanced_optimizer.py      #   进阶优化器：GA、PSO、强化学习
│   │   └── mask_optimizer.py          #   掩模优化编排：整合成像+优化+调度
│   │
│   ├── utils/                         # 【工具层】通用功能
│   │   ├── config.py                  #   配置管理：YAML加载/验证/保存
│   │   ├── data_io.py                 #   数据IO：测试图案生成、文件读写
│   │   ├── visualization.py           #   可视化：Matplotlib绑图函数库
│   │   └── logger.py                  #   日志：彩色日志、优化过程记录
│   │
│   ├── examples/                      # 【示例层】使用演示
│   │   ├── run_optimization.py        #   完整优化流程示例
│   │   └── performance_benchmark.py   #   性能基准测试
│   │
│   ├── tests/                         # 【测试层】单元测试
│   │   ├── test_imaging.py            #   光学成像测试
│   │   ├── test_fft.py                #   FFT模块测试
│   │   ├── test_metrics.py            #   评估指标测试
│   │   └── test_optimizer.py          #   优化器测试
│   │
│   ├── config/
│   │   └── default_config.yaml        # 默认配置文件
│   ├── requirements.txt               # Python依赖
│   ├── Dockerfile                     # Docker构建
│   └── __init__.py
│
├── docker-compose.yml                 # Docker编排
├── .gitignore
└── README.md
```

### 4.2 分层架构设计

```
┌─────────────────────────────────────────────────┐
│              应用层 (Application)                │
│  示例脚本 / 用户自定义研究代码                    │
├─────────────────────────────────────────────────┤
│              编排层 (Orchestration)              │
│  MaskOptimizer - 整合成像、优化、调度、早停       │
├─────────────────────────────────────────────────┤
│              算法层 (Algorithms)                 │
│  基础优化器(GD/BFGS/Newton)                      │
│  启发式优化器(GA/PSO)                            │
│  强化学习优化器(RL Interface)                    │
├─────────────────────────────────────────────────┤
│              核心物理层 (Core Physics)           │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ 光学成像 │  │ FFT计算  │  │  误差评估指标 │ │
│  │(Hopkins) │  │(频域处理)│  │ (MSE/SSIM等)  │ │
│  └──────────┘  └──────────┘  └───────────────┘ │
├─────────────────────────────────────────────────┤
│              工具层 (Utilities)                  │
│  配置管理 │ 数据IO │ 可视化 │ 日志              │
├─────────────────────────────────────────────────┤
│              基础设施层 (Infrastructure)         │
│  NumPy / SciPy / Numba / Matplotlib / PyYAML    │
└─────────────────────────────────────────────────┘
```

### 4.3 核心数据流（掩模优化流程）

```
初始掩模 M₀
    │
    ▼
┌────────────────────┐
│  PartialCoherent   │──── 光学系统参数配置
│  Imaging Model     │
└─────────┬──────────┘
          │  计算晶圆成像 I
          ▼
┌────────────────────┐
│   Metrics Module   │──── 目标图像 I_target
│  (MSE/MAE/SSIM)    │
└─────────┬──────────┘
          │  计算损失 L = metric(I, I_target)
          ▼
┌────────────────────┐
│   BaseOptimizer    │──── 解析梯度 dL/dM
│  / HeuristicOpt    │     或 启发式搜索
└─────────┬──────────┘
          │  更新掩模 M'
          ▼
    ┌─ 收敛？ ─否 ─┐
    │              │
    是             │
    │              ▼
    ▼         M = M'，返回迭代
优化结果 M*
```

### 4.4 关键类设计

| 类名 | 职责 | 设计模式 |
|------|------|----------|
| `OpticalSystem` | 光学系统参数数据类 | Data Class |
| `PartialCoherentImaging` | 部分相干成像计算引擎 | Facade |
| `BaseOptimizer` | 优化器抽象基类 | Template Method |
| `GradientDescentOptimizer` | 梯度下降具体实现 | Strategy |
| `BFGSOptimizer` | BFGS具体实现（适配scipy） | Adapter |
| `BaseHeuristicOptimizer` | 启发式算法基类 | Template Method |
| `GeneticAlgorithmOptimizer` | 遗传算法具体实现 | Strategy |
| `ParticleSwarmOptimizer` | PSO具体实现 | Strategy |
| `ReinforcementLearningOptimizer` | RL接口，支持自定义模型 | Strategy + Composition |
| `MaskOptimizer` | 完整优化流程编排 | Facade + Builder |
| `LearningRateScheduler` | 学习率调度 | Strategy |
| `EarlyStopping` | 早停机制 | Observer |

### 4.5 技术栈

| 类别 | 依赖 | 版本 | 用途 |
|------|------|------|------|
| 数值计算 | NumPy | ≥1.21 | 数组运算、线性代数 |
| 数值计算 | SciPy | ≥1.7 | FFT、优化算法(minimize) |
| 性能加速 | Numba | ≥0.54 | JIT编译、并行循环 |
| 图像处理 | OpenCV | ≥4.5 | 图像读写与处理 |
| 图像处理 | Pillow | ≥8.0 | 图像格式支持 |
| 可视化 | Matplotlib | ≥3.4 | 结果绘图 |
| 配置文件 | PyYAML | ≥5.4 | YAML配置解析 |
| 测试 | pytest | ≥6.2 | 单元测试框架 |
| 测试 | pytest-cov | ≥2.12 | 测试覆盖率 |
| 日志 | colorlog | ≥5.0 | 彩色日志输出 |

### 4.6 部署与运行

项目支持两种运行方式：

1. **Docker Compose**（推荐）
   - `backend` 服务：运行优化示例
   - `benchmark` 服务：性能基准测试
   - `test` 服务：执行单元测试

2. **本地 Python 环境**（Python 3.9+）
   - `pip install -r requirements.txt` 安装依赖
   - `PYTHONPATH=. python -m examples.run_optimization` 运行示例

---

## 五、优化迭代流程伪代码

```python
def optimize(initial_mask, target_image, config):
    # 1. 初始化
    imaging = PartialCoherentImaging(optical_system, mask.shape)
    optimizer = create_optimizer(config.optimizer_type)
    lr_scheduler = LearningRateScheduler(config)
    early_stopping = EarlyStopping(patience=10)
    
    # 2. 定义损失函数
    def loss(mask):
        wafer = imaging.compute_aerial_image(mask)
        return mse(wafer, target_image)
    
    # 3. 定义梯度函数（仅梯度类优化器使用）
    def gradient(mask):
        wafer = imaging.compute_aerial_image(mask)
        error_grad = 2 * (wafer - target) / mask.size
        imaging_grad = imaging.compute_image_gradient(mask)
        return error_grad * imaging_grad
    
    # 4. 迭代优化
    result = optimizer.optimize(
        objective=loss,
        x0=initial_mask,
        gradient=gradient,  # 启发式算法可省略
        bounds=(0.0, 1.0)
    )
    
    return result
```

---

## 六、可扩展方向

项目 README 已明确预留以下扩展点：

1. **GPU 加速**：使用 CuPy 替代 NumPy，将计算迁移至 GPU
2. **多层掩模优化**：扩展 `MaskOptimizer` 支持 SMO/DMD 等多掩模联合优化
3. **RL 模型扩展**：`ReinforcementLearningOptimizer` 已预留接口，可接入 DQN、PPO 等深度 RL 模型
4. **更精确成像模型**：可扩展至矢量成像、严格电磁仿真（RCWA/FDTD）
5. **并行优化**：启发式算法的种群评估可使用多进程/多线程加速
