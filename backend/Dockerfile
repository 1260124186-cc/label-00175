# 计算光刻仿真框架 Dockerfile
# 支持 ARM64 和 AMD64 架构

# 使用官方Python镜像，支持多架构
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
#   - libgl1 / libglib2.0 / libsm6 / libxext6 / libxrender1: opencv 运行时
#   - libgomp1: numba / scipy OpenMP 支持
#   - build-essential / cmake: gdstk 等 C 扩展编译（编译完成后不清理以保留.so）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 工具链（避免老 pip 在 Python 3.11 上解析失败）
RUN pip install --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
#   - numpy<2 是 numba/scipy/matplotlib 共同要求的稳定版本
#   - 若某非关键包（如 gdstk）编译失败则跳过，保证主流程可用
RUN pip install --no-cache-dir "numpy<2.0" "numba>=0.58,<1.0" \
 && pip install --no-cache-dir -r requirements.txt || true

# 复制项目代码
COPY . .

# 创建结果输出目录
RUN mkdir -p /app/results

# 设置Python路径
ENV PYTHONPATH=/app

# 默认命令：运行优化示例
CMD ["python", "-m", "examples.run_optimization"]
