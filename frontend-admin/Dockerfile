# 前端 admin Dockerfile（多阶段构建）
#
# 镜像仓库前缀：官方镜像位于 library/ 命名空间下，需使用完整路径
#   默认值: docker.io/        （直连 Docker Hub，绕过 daemon.json 中失效的镜像）
#   DaoCloud: docker.m.daocloud.io/
#   网易: hub-mirror.c.163.com/
# 用法:
#   docker build --build-arg IMAGE_REGISTRY=docker.m.daocloud.io/ .
#   IMAGE_REGISTRY=docker.m.daocloud.io/ docker compose build
ARG IMAGE_REGISTRY=docker.io/

# Stage 1: Node 构建
FROM ${IMAGE_REGISTRY}library/node:20-alpine AS builder

WORKDIR /app

# 先复制 package.json，利用 docker 缓存
COPY package.json package-lock.json* /app/
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

# 复制源码并构建
COPY . /app/
RUN npm run build

# ============================================================
# Stage 2: Nginx 托管 + 反代 /api → 后端
# ============================================================
FROM ${IMAGE_REGISTRY}library/nginx:1.27-alpine AS runner

LABEL maintainer="litho-sim-admin"

# 时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 删除 nginx 默认站
RUN rm -rf /usr/share/nginx/html/*

# 复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制 nginx 配置（反代 /api、SPA history fallback）
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 健康检查：访问 index.html
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=5 \
    CMD wget --no-verbose --tries=1 --spider http://127.0.0.1:80/ || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
