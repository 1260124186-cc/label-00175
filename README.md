# 计算光刻与版图优化仿真框架

基于Python的掩模图案优化仿真框架，用于计算光刻研究中的算法开发与验证。

## How to Run

### 使用Docker（推荐）

```bash
# 构建并运行优化示例
docker-compose up --build -d

# 查看运行日志
docker logs -f litho-sim-backend

# 运行性能测试
docker-compose --profile benchmark up benchmark

# 运行单元测试
docker-compose --profile test up test

# 停止服务
docker-compose down
```

### 本地运行

```bash
# 安装依赖
cd backend
pip install -r requirements.txt

# 运行优化示例
PYTHONPATH=. python -m examples.run_optimization

# 运行性能测试
PYTHONPATH=. python -m examples.performance_benchmark

# 运行单元测试
PYTHONPATH=. pytest tests/ -v
```

## Services

| 服务 | 描述 | 命令 |
|------|------|------|
| backend | 主仿真服务，运行掩模优化示例 | `docker-compose up backend` |
| benchmark | 性能基准测试服务 | `docker-compose --profile benchmark up benchmark` |
| test | 单元测试服务 | `docker-compose --profile test up test` |

## 测试

### 运行全部测试

```bash
# Docker方式
docker-compose --profile test up test

# 本地方式
cd backend
pytest tests/ -v
```

### 测试覆盖率

```bash
cd backend
PYTHONPATH=. pytest tests/ -v --cov=. --cov-report=html
```

### 测试模块说明

| 测试文件 | 覆盖模块 |
|----------|----------|
| test_imaging.py | 光学成像模块（OpticalSystem, PartialCoherentImaging） |
| test_fft.py | 傅里叶变换模块（FFT, 滤波, 相位调制, Numba加速） |
| test_metrics.py | 误差评估模块（MSE, MAE, SSIM, NCC, PSNR） |
| test_optimizer.py | 优化器模块（梯度下降, BFGS, GA, PSO, RL） |

---

## 项目简介

本框架为博士研究生计算光刻与版图优化研究提供Python仿真环境，核心功能包括：

- **光学成像建模**：部分相干成像模型（Hopkins模型）
- **傅里叶变换计算**：FFT/IFFT、频域滤波、相位调制
- **优化算法**：梯度下降、BFGS、遗传算法、粒子群优化
- **误差评估**：MSE、MAE、SSIM、归一化相关系数

## 项目结构

```
.
├── backend/                    # 后端仿真代码
│   ├── core/                   # 核心模块
│   │   ├── imaging.py          # 光学成像模块
│   │   ├── fft.py              # 傅里叶变换模块
│   │   └── metrics.py          # 误差评估模块
│   ├── algorithms/             # 算法模块
│   │   ├── optimizer.py        # 基础优化器
│   │   ├── advanced_optimizer.py # 进阶优化器
│   │   └── mask_optimizer.py   # 掩模优化模块
│   ├── utils/                  # 工具模块
│   │   ├── data_io.py          # 数据处理
│   │   ├── visualization.py    # 可视化
│   │   ├── logger.py           # 日志
│   │   └── config.py           # 配置管理
│   ├── examples/               # 示例代码
│   │   ├── run_optimization.py # 优化示例
│   │   └── performance_benchmark.py # 性能测试
│   ├── tests/                  # 单元测试
│   ├── config/                 # 配置文件
│   ├── requirements.txt        # Python依赖
│   └── Dockerfile              # Docker构建文件
├── docker-compose.yml          # Docker Compose配置
├── .gitignore                  # Git忽略文件
└── README.md                   # 项目说明
```

## 核心功能

### 1. 光学成像模块 (core/imaging.py)

```python
from core.imaging import OpticalSystem, simulate_wafer_image

# 配置光学系统
optics = OpticalSystem(
    wavelength=193.0,  # ArF光源
    na=1.35,           # 高NA浸没式
    sigma=0.75         # 部分相干
)

# 模拟晶圆成像
wafer_image = simulate_wafer_image(mask, optical_system=optics)
```

### 2. 傅里叶变换模块 (core/fft.py)

```python
from core.fft import fft2d, frequency_filter

# 计算频谱
spectrum = fft2d(image, shift=True)

# 低通滤波
filtered = frequency_filter(spectrum, 'lowpass', cutoff=0.3)
```

### 3. 掩模优化 (algorithms/mask_optimizer.py)

```python
from algorithms.mask_optimizer import MaskOptimizer, OptimizationConfig

# 配置优化参数（支持随机种子用于结果复现）
config = OptimizationConfig(
    optimizer_type='gradient_descent',
    max_iter=100,
    learning_rate=0.01,
    random_seed=42  # 设置随机种子确保结果可复现
)

# 执行优化
optimizer = MaskOptimizer(optical_system=optics, config=config)
result = optimizer.optimize(initial_mask, target_image)
```

### 4. 强化学习优化器 (algorithms/advanced_optimizer.py)

```python
from algorithms.advanced_optimizer import ReinforcementLearningOptimizer, SimpleQLearningModel

# 创建RL优化器
rl_optimizer = ReinforcementLearningOptimizer(
    max_iter=100,
    epsilon=0.1,
    seed=42
)

# 可选：接入自定义RL模型
model = SimpleQLearningModel(state_dim=128, action_dim=64)
rl_optimizer.set_model(model)

# 执行优化
result = rl_optimizer.optimize(objective_func, initial_mask, target=target_image)
```

## 框架扩展建议

### GPU加速

```python
# 使用CuPy替代NumPy进行GPU加速
import cupy as cp

# 将数据转移到GPU
mask_gpu = cp.asarray(mask)
spectrum_gpu = cp.fft.fft2(mask_gpu)
```

### 多掩模层优化

```python
# 扩展MaskOptimizer支持多层掩模
class MultiLayerMaskOptimizer(MaskOptimizer):
    def optimize_layers(self, masks: List[np.ndarray], target: np.ndarray):
        # 实现多层联合优化
        pass
```

## 常见问题排查

### FFT边界效应

- **问题**：频谱出现伪影
- **解决**：使用零填充或窗函数
```python
# 零填充
padded = np.pad(image, pad_width=32, mode='constant')
spectrum = fft2d(padded)
```

### 优化算法收敛性

- **问题**：优化不收敛或收敛到局部最优
- **解决**：
  1. 调整学习率（尝试0.001-0.1范围）
  2. 使用学习率调度器
  3. 尝试不同优化器（BFGS通常更稳定）
  4. 使用启发式算法（GA/PSO）跳出局部最优

## 性能优化方向

1. **批量处理**：同时处理多个掩模图案
2. **内存复用**：预分配数组避免重复内存分配
3. **频域近似**：使用TCC分解减少计算量
4. **并行计算**：使用多进程处理独立任务
5. **JIT编译**：使用Numba加速关键计算函数

## 许可证

MIT License
