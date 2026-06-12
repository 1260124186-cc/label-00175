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
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建结果输出目录
RUN mkdir -p /app/results

# 设置Python路径
ENV PYTHONPATH=/app

# 默认命令：运行优化示例
CMD ["python", "-m", "examples.run_optimization"]
