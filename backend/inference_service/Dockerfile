# ========================================================================
# 产线推理微服务 - Dockerfile
# ========================================================================
# 多阶段构建: 最小化生产镜像体积
#
# 构建:
#   docker build -t litho-inference-service:latest .
#
# 运行 (Hopkins 近似模式, 无需模型):
#   docker run -p 50051:50051 -p 8080:8080 litho-inference-service:latest
#
# 运行 (代理模型模式):
#   docker run -p 50051:50051 -p 8080:8080 \
#     -v /path/to/models:/app/backend/inference_service/models:ro \
#     -e LITHO_INF_ENGINE_TYPE=surrogate \
#     litho-inference-service:latest
# ========================================================================

# ---------- Stage 1: 构建/依赖安装 ----------
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装 (独立层以利用缓存
COPY backend/inference_service/requirements.txt /tmp/requirements.txt
RUN pip install --prefix=/install --no-cache-dir -r /tmp/requirements.txt

# ---------- Stage 2: 生产镜像 ----------
FROM python:3.11-slim AS production

LABEL maintainer="Lithography Research Team" \
      description="光刻产线推理微服务 (Hopkins 近似 + 代理模型)" \
      org.opencontainers.image.title="litho-inference-service" \
      org.opencontainers.image.source="https://github.com/your-org/litho-sim"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/app/backend:/app/backend/inference_service:${PYTHONPATH}"

# 系统依赖 (仅运行时必需)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -m appuser

# 复制依赖
COPY --from=builder /install /usr/local

# 复制推理服务代码
COPY backend/inference_service/ /app/backend/inference_service/

# 复制核心共享模块 (依赖的少量核心文件)
COPY backend/core/__init__.py /app/backend/core/__init__.py
COPY backend/core/fft.py /app/backend/core/fft.py
COPY backend/core/array_backend.py /app/backend/core/array_backend.py

# 创建必要目录
RUN mkdir -p /app/backend/inference_service/models \
             /app/backend/inference_service/logs \
             /app/backend/inference_service/.cache \
    && chown -R appuser:appuser /app

WORKDIR /app/backend/inference_service

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 切换非 root 用户
USER appuser

# 暴露端口
EXPOSE 50051/tcp 8080/tcp 9090/tcp

# 默认入口 (Hopkins 近似模式)
ENTRYPOINT ["python", "-m", "start_service"]
CMD ["--engine-type", "hopkins_lite", "--log-level", "INFO"]
