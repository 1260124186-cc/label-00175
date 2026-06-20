import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, Query, Depends

from schemas import (
    SimulationConfig,
    ConfigResponse,
    SaveConfigRequest,
    SaveConfigResponse,
)
from services import (
    load_default_config,
    save_config_to_file,
    list_saved_configs,
    load_saved_config,
    delete_saved_config,
    add_backend_to_path,
)
from auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/config", tags=["配置管理"])


@router.get("/default", response_model=ConfigResponse, summary="获取默认配置")
async def get_default_config(current_user: Dict[str, Any] = Depends(get_current_user)):
    config_data = load_default_config()
    config = SimulationConfig.model_validate(config_data)
    return ConfigResponse(success=True, config=config, message="默认配置加载成功")


@router.get("/saved", summary="列出已保存的配置文件")
async def list_configs(current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    result = list_saved_configs(user_id=user_id)
    return {"success": True, "count": result["count"], "files": result["files"], "message": "配置列表加载成功"}


@router.get("/saved/{filename}", summary="加载指定的已保存配置")
async def get_saved_config(filename: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    data = load_saved_config(filename, user_id=user_id)
    config = SimulationConfig.model_validate(data["config"])
    return ConfigResponse(success=True, config=config, message=f"加载配置: {filename}")


@router.post("/save", response_model=SaveConfigResponse, summary="保存配置")
async def save_config(req: SaveConfigRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    config_dict = req.config.model_dump()
    saved_path = save_config_to_file(config_dict, req.filename, user_id=user_id)
    return SaveConfigResponse(
        success=True,
        message="配置保存成功",
        saved_path=saved_path
    )


@router.delete("/saved/{filename}", summary="删除已保存的配置文件")
async def remove_config(filename: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    delete_saved_config(filename, user_id=user_id)
    return {"success": True, "message": f"配置文件已删除: {filename}"}


@router.post("/validate", summary="验证配置有效性")
async def validate_config(config: SimulationConfig, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        config_dict = config.model_dump()
        add_backend_to_path()
        from utils.config import validate_config as vc
        is_valid = vc(config_dict)
        return {"success": True, "valid": is_valid, "message": "验证完成" if is_valid else "配置验证失败"}
    except Exception as e:
        return {"success": False, "valid": False, "message": f"验证异常: {str(e)}"}
