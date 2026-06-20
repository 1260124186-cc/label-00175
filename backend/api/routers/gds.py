from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Dict, Any

from .. import services
from ..schemas import (
    GdsUploadResponse,
    GdsListResponse,
    GdsLayersResponse,
)
from ..auth import get_current_user

router = APIRouter(prefix="/api/gds", tags=["GDS"])


@router.post("/upload", response_model=GdsUploadResponse)
async def upload_gds(file: UploadFile = File(...), current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    try:
        contents = await file.read()
        result = services.upload_gds_file(contents, file.filename or "unknown.gds", user_id=user_id)
        return GdsUploadResponse(
            success=True,
            message="上传成功",
            file=result,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


@router.get("/list", response_model=GdsListResponse)
def list_gds(current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    result = services.list_gds_files(user_id=user_id)
    return GdsListResponse(**result)


@router.get("/{file_id}/layers", response_model=GdsLayersResponse)
def get_gds_layers(file_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    result = services.get_gds_layers(file_id, user_id=user_id)
    return GdsLayersResponse(**result)


@router.delete("/{file_id}")
def delete_gds(file_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    result = services.delete_gds_file(file_id, user_id=user_id)
    return result
