# 分布式批处理一键启动指南

## 架构概述

本框架支持两种批处理运行模式，可无缝切换：

```
┌─────────────────────────────────────────────────────────────┐
│                     单机多进程模式（默认）                   │
│  LocalBatchRunner → ProcessPoolExecutor → 多进程并发        │
│  无需额外组件，开箱即用                                     │
└─────────────────────────────────────────────────────────────┘

                              ↓  无缝扩展

┌─────────────────────────────────────────────────────────────┐
│                   多容器分布式集群模式                       │
│                                                             │
│  ┌──────────────┐     ┌──────────┐     ┌────────────────┐  │
│  │ batch_scheduler │ →  │  Redis   │ →  │ celery_worker  │  │
│  │  (任务提交)    │    │ (Broker/ │    │ (执行节点xN)    │  │
│  │                │    │  Backend) │    │  可水平扩展     │  │
│  └──────────────┘     └──────────┘     └────────────────┘  │
│                       ↑                                      │
│                ┌───────────────┐                            │
│                │ celery_flower │  →  Web监控面板            │
│                └───────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 0. 前置准备

将待处理的 GDS 文件放入 `./layout_library/` 目录：
```bash
mkdir -p layout_library
# 复制你的 GDS 文件到此目录
```

### 1. 一键启动分布式集群（最简命令）

```bash
# 启动完整分布式栈（Redis + 2个 Worker + Flower监控）
docker compose --profile distributed up -d --build
```

### 2. 查看集群状态

```bash
# 查看所有服务健康状态
docker compose ps

# 查看 Worker 日志
docker compose logs -f celery_worker

# 访问 Flower 监控面板: http://localhost:5555
# 默认账号: admin / admin123
```

### 3. 提交批处理任务

```bash
# 使用分布式模式提交任务（自动使用 Celery 集群）
docker compose run --rm batch_scheduler \
  --source /app/layout_library \
  --layer 0 \
  --output-dir /app/results/batch_run \
  --max-iter 100
```

### 4. 查看结果

```bash
# 结果会保存在 ./results/batch_run/ 目录下
ls -la results/batch_run/
```

---

## 从单机到集群的无缝迁移步骤

### 原单机命令（本地多进程）
```bash
# 方式1：直接运行 Python
python -m pipeline.batch_runner_cli \
  --source ./layout_library --layer 0 \
  --output-dir ./results/batch_run \
  --max-workers 4

# 方式2：使用 Docker Compose batch 服务
docker compose run --rm batch_runner \
  --source /app/layout_library --layer 0 \
  --output-dir /app/results/batch_run
```

### 迁移到分布式集群（只需改2处）

```diff
# 1. 启动分布式依赖
- 无需启动额外服务
+ docker compose --profile distributed up -d --build

# 2. 提交任务时指定 --mode distributed
- docker compose run --rm batch_runner ...
+ docker compose run --rm batch_scheduler --mode distributed ...
```

**代码零改动**，`batch_runner.py` 内部会自动选择 `DistributedBatchRunner`。

---

## 水平扩展 Worker

### 调整 Worker 副本数

```bash
# 扩展到 4 个 Worker 容器
docker compose --profile distributed up -d --scale celery_worker=4

# 扩展到 8 个 Worker 容器
docker compose --profile distributed up -d --scale celery_worker=8

# 缩容到 2 个 Worker
docker compose --profile distributed up -d --scale celery_worker=2
```

### 调整单容器内并发进程数

每个 Worker 容器内默认启动 2 个并发进程，可通过环境变量调整：

```bash
# 单容器内 4 个并发进程 × 4 个容器 = 16 个并发任务
CELERY_CONCURRENCY=4 docker compose --profile distributed up -d --scale celery_worker=4
```

### 资源配置参考

| 服务器配置 | Worker 副本数 | 单容器并发数 | 总并发数 |
|-----------|--------------|-------------|---------|
| 4核16G     | 2            | 2           | 4       |
| 8核32G     | 4            | 2           | 8       |
| 16核64G    | 4            | 4           | 16      |
| 32核128G   | 8            | 4           | 32      |

---

## 常用命令速查

### 集群管理

```bash
# 启动完整分布式栈（含监控）
docker compose --profile distributed up -d --build

# 仅启动核心服务（不含监控）
docker compose --profile distributed up -d --build redis celery_worker

# 停止所有分布式服务
docker compose --profile distributed down

# 重启 Worker（配置变更后）
docker compose restart celery_worker
```

### 任务提交

```bash
# 基础用法
docker compose run --rm batch_scheduler \
  --source /app/layout_library --layer 0 \
  --output-dir /app/results/batch_run

# 使用自定义优化配置
docker compose run --rm batch_scheduler \
  --source /app/layout_library --layer 0 \
  --optimizer-config /app/config/opc_default.yaml \
  --output-dir /app/results/batch_run

# 启用层次化批处理 + 保存掩模
docker compose run --rm batch_scheduler \
  --source /app/layout_library --layer 0 \
  --save-masks \
  --output-dir /app/results/batch_run

# Dry-Run 模式（仅加载建队，不执行）
docker compose run --rm batch_scheduler \
  --source /app/layout_library --layer 0 \
  --dry-run --save-cell-list \
  --output-dir /app/results/batch_preview
```

### 监控与调试

```bash
# 查看所有服务状态
docker compose ps

# 查看 Redis 日志
docker compose logs -f redis

# 查看 Worker 日志（跟踪任务执行）
docker compose logs -f celery_worker

# 查看调度器日志
docker compose logs -f batch_scheduler

# 查看 Flower 监控
# 浏览器访问: http://localhost:5555

# 查看队列深度（Redis CLI）
docker compose exec redis redis-cli -n 0 llen litho_batch

# 手动检查 Worker 存活
docker compose exec -it celery_worker celery -A pipeline.batch_runner.celery_app inspect active
```

---

## 健康检查说明

所有服务均配置了健康检查，确保服务可用性：

| 服务 | 检查方式 | 间隔 | 超时 | 重试 | 启动宽限 |
|------|---------|------|------|------|---------|
| redis | `redis-cli ping` | 5s | 3s | 5次 | 10s |
| celery_worker | `celery inspect ping` | 20s | 10s | 3次 | 60s |
| celery_flower | HTTP GET / | 15s | 3s | 3次 | 20s |
| batch_scheduler | 依赖 redis + worker 就绪 | - | - | - | - |

---

## 配置说明

### 环境变量

在 `.env` 文件中可配置以下参数：

```env
# 镜像源（国内加速可选）
IMAGE_REGISTRY=docker.m.daocloud.io/

# Celery 配置
CELERY_QUEUE=litho_batch
CELERY_CONCURRENCY=2
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

# Flower 监控认证（修改默认密码！）
FLOWER_AUTH=admin:your_secure_password

# 时区
TZ=Asia/Shanghai
```

### Redis 配置

Redis 已针对批处理场景优化：
- AOF 持久化（everysec）：防止重启后任务丢失
- 最大内存 512MB，LRU 淘汰：避免内存溢出
- TCP keepalive 60s：检测死连接
- 超时 300s：自动清理空闲连接

---

## 完整使用流程示例

### 场景：100个 GDS 文件的批量优化

```bash
# 1. 准备数据
mkdir -p layout_library
cp /path/to/your/gds/files/*.gds layout_library/

# 2. 启动分布式集群（4 Worker，每 Worker 4 并发 = 16 并发）
CELERY_CONCURRENCY=4 docker compose --profile distributed up -d --build --scale celery_worker=4

# 3. 等待服务就绪（检查健康状态）
watch docker compose ps
# 所有服务显示 healthy 后继续

# 4. Dry-Run 预览（确认数据加载正确）
docker compose run --rm batch_scheduler \
  --source /app/layout_library --layer 0 \
  --dry-run --save-cell-list \
  --output-dir /app/results/batch_preview

# 5. 提交正式任务
docker compose run --rm batch_scheduler \
  --source /app/layout_library --layer 0 \
  --optimizer-config /app/config/opc_default.yaml \
  --max-iter 200 \
  --save-masks \
  --max-retries 2 \
  --output-dir /app/results/batch_20240618

# 6. 实时监控
# 方式1：Flower 面板 http://localhost:5555
# 方式2：查看 Worker 日志
docker compose logs -f --tail=100 celery_worker
# 方式3：查看队列长度
watch 'docker compose exec redis redis-cli -n 0 llen litho_batch'

# 7. 任务完成后查看结果
ls -la results/batch_20240618/
# - batch_xxx_results.csv: 详细结果表格
# - batch_xxx_results.json: JSON 格式结果
# - batch_xxx_summary.json: 批次汇总统计
# - masks/: 优化后的掩模 npy 文件
```

---

## 故障排查

### Worker 启动失败
```bash
# 查看详细日志
docker compose logs --tail=200 celery_worker

# 常见问题：
# 1. Redis 连接失败 → 检查 redis 服务是否 healthy
# 2. 依赖缺失 → 重新构建镜像: docker compose build
# 3. 端口冲突 → 修改 docker-compose.yml 中的端口映射
```

### 任务一直 PENDING
```bash
# 检查 Worker 是否注册成功
docker compose exec -it celery_worker celery -A pipeline.batch_runner.celery_app inspect registered

# 检查队列是否有任务
docker compose exec redis redis-cli -n 0 llen litho_batch

# 检查 Worker 是否在消费
docker compose exec -it celery_worker celery -A pipeline.batch_runner.celery_app inspect active
```

### 结果丢失
- 确保 Redis 配置了 AOF 持久化（已默认配置）
- 任务结果默认保存 24 小时，可在 Celery 配置中调整
- 重要结果请在任务完成后及时从 `results/` 目录备份

---

## 最佳实践

1. **资源预留**：每个 Worker 至少预留 2 核 CPU 和 4GB 内存
2. **并发数设置**：单容器并发数建议 ≤ CPU 核数
3. **失败重试**：重要任务建议设置 `--max-retries 2`
4. **监控告警**：集成 Flower 或 Prometheus 监控任务队列长度
5. **数据持久化**：`results/` 和 `redis_data/` 目录建议挂载到高性能存储
6. **滚动更新**：更新 Worker 时使用 `--scale` 逐步替换，避免全部重启

---

## 相关文件参考

- 核心调度逻辑: [batch_runner.py](backend/pipeline/batch_runner.py)
- 命令行入口: [batch_runner_cli.py](backend/pipeline/batch_runner_cli.py)
- Docker 配置: [docker-compose.yml](docker-compose.yml)
- 后端镜像: [backend/Dockerfile](backend/Dockerfile)
