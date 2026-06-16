from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from .. import services
from ..schemas import (
    GdsUploadResponse,
    GdsListResponse,
    GdsLayersResponse,
)

router = APIRouter(prefix="/api/gds", tags=["GDS"])


@router.post("/upload", response_model=GdsUploadResponse)
async def upload_gds(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = services.upload_gds_file(contents, file.filename or "unknown.gds")
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
def list_gds():
    result = services.list_gds_files()
    return GdsListResponse(**result)


@router.get("/{file_id}/layers", response_model=GdsLayersResponse)
def get_gds_layers(file_id: str):
    result = services.get_gds_layers(file_id)
    return GdsLayersResponse(**result)


@router.delete("/{file_id}")
def delete_gds(file_id: str):
    result = services.delete_gds_file(file_id)
    return result
