import sys
import os
import logging

_API_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_API_DIR)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from routers.config import router as config_router
from routers.simulation import router as simulation_router
from routers.workflows import router as workflows_router
from routers.tasks import router as tasks_router
from routers.websocket import router as websocket_router
from routers.gds import router as gds_router
from routers.experiments import router as experiments_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RET 光刻仿真工作台 API",
    description="计算光刻仿真框架 Web API，支持参数配置、仿真运行、OPC/SMO/ILT 工作流、工艺窗口分析、批处理优化、统一任务管理",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(config_router)
app.include_router(simulation_router)
app.include_router(workflows_router)
app.include_router(tasks_router)
app.include_router(websocket_router)
app.include_router(gds_router)
app.include_router(experiments_router)


@app.get("/", summary="根路径 - 跳转到 API 文档")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health", summary="健康检查")
async def health_check():
    return {
        "status": "ok",
        "service": "ret-litho-api",
        "version": "2.0.0",
        "features": [
            "config",
            "simulation",
            "opc",
            "smo",
            "ilt",
            "process_window",
            "batch_optimization",
            "task_management",
            "experiment_tracking",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
