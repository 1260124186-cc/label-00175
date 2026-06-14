import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .routers.config import router as config_router
from .routers.simulation import router as simulation_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="光刻仿真工作台 API",
    description="计算光刻仿真框架 Web API，支持参数配置、仿真运行、结果查看",
    version="1.0.0",
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


@app.get("/", summary="根路径 - 跳转到 API 文档")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health", summary="健康检查")
async def health_check():
    return {
        "status": "ok",
        "service": "litho-simulation-api",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
