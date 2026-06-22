# -*- coding: utf-8 -*-
"""
协作式研究空间 API 路由

将 WorkspaceService 的全部功能暴露为 REST 接口。
前缀: /api/workspace
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query, HTTPException, Depends

_WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND_ROOT = os.path.dirname(_WORKSPACE_DIR)
_API_DIR = os.path.join(_BACKEND_ROOT, "api")
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from auth import get_current_user

from workspace.schemas import (
    ProjectCreate,
    ProjectUpdate,
    ProjectInfo,
    ProjectListResponse,
    ProjectMemberAdd,
    ProjectMemberUpdate,
    ProjectMemberInfo,
    ProjectVersionCreate,
    ProjectVersionInfo,
    ProjectVersionListResponse,
    HotspotCommentCreate,
    HotspotCommentInfo,
    HotspotCommentListResponse,
    ForkCreate,
    ForkInfo,
    ForkListResponse,
)
from workspace.services import WorkspaceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workspace", tags=["协作研究空间"])


def _handle_service_error(e: Exception) -> HTTPException:
    """统一转换服务层异常为 HTTP 错误"""
    msg = str(e)
    if isinstance(e, PermissionError):
        return HTTPException(status_code=403, detail=msg)
    if isinstance(e, ValueError):
        if "不存在" in msg or "not found" in msg.lower():
            return HTTPException(status_code=404, detail=msg)
        return HTTPException(status_code=400, detail=msg)
    logger.exception("Workspace API 未预期异常")
    return HTTPException(status_code=500, detail=f"服务内部错误: {msg}")


# ===========================================================================
# 项目管理
# ===========================================================================

@router.post("/projects", response_model=ProjectInfo, summary="创建新项目")
async def create_project(
    req: ProjectCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return WorkspaceService.create_project(
            user_id=current_user["user_id"],
            username=current_user["username"],
            req=req,
        )
    except Exception as e:
        raise _handle_service_error(e)


@router.get("/projects", response_model=ProjectListResponse, summary="列出项目")
async def list_projects(
    scope: str = Query(
        "all",
        description="范围: mine(我的), public(公开), shared(共享), all(所有可见)",
    ),
    visibility: Optional[str] = Query(None, description="按可见性过滤"),
    tag: Optional[str] = Query(None, description="按标签过滤"),
    keyword: Optional[str] = Query(None, description="关键词搜索(名称/描述/标签)"),
    archived: Optional[bool] = Query(False, description="是否包含归档项目"),
    limit: int = Query(100, ge=1, le=500, description="返回数量上限"),
    offset: int = Query(0, ge=0, description="偏移量"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        total, projects = WorkspaceService.list_projects(
            user_id=current_user["user_id"],
            scope=scope,
            visibility=visibility,
            tag=tag,
            keyword=keyword,
            archived=archived,
            limit=limit,
            offset=offset,
        )
        return ProjectListResponse(count=total, projects=projects)
    except Exception as e:
        raise _handle_service_error(e)


@router.get("/projects/{project_id}", response_model=ProjectInfo, summary="获取项目详情")
async def get_project(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return WorkspaceService.get_project(project_id, current_user["user_id"])
    except Exception as e:
        raise _handle_service_error(e)


@router.put("/projects/{project_id}", response_model=ProjectInfo, summary="更新项目信息")
async def update_project(
    project_id: str,
    req: ProjectUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return WorkspaceService.update_project(project_id, current_user["user_id"], req)
    except Exception as e:
        raise _handle_service_error(e)


@router.delete("/projects/{project_id}", summary="删除项目")
async def delete_project(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        WorkspaceService.delete_project(project_id, current_user["user_id"])
        return {"success": True, "message": "项目已删除"}
    except Exception as e:
        raise _handle_service_error(e)


# ===========================================================================
# 成员管理（项目共享）
# ===========================================================================

@router.get("/projects/{project_id}/members", response_model=List[ProjectMemberInfo], summary="列出项目成员")
async def list_members(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return WorkspaceService.list_members(project_id, current_user["user_id"])
    except Exception as e:
        raise _handle_service_error(e)


@router.post("/projects/{project_id}/members", response_model=ProjectMemberInfo, summary="添加项目成员")
async def add_member(
    project_id: str,
    req: ProjectMemberAdd,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return WorkspaceService.add_member(project_id, current_user["user_id"], req)
    except Exception as e:
        raise _handle_service_error(e)


@router.put("/projects/{project_id}/members/{target_user_id}", response_model=ProjectMemberInfo, summary="修改成员角色")
async def update_member_role(
    project_id: str,
    target_user_id: str,
    req: ProjectMemberUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return WorkspaceService.update_member_role(
            project_id, target_user_id, current_user["user_id"], req,
        )
    except Exception as e:
        raise _handle_service_error(e)


@router.delete("/projects/{project_id}/members/{target_user_id}", summary="移除项目成员")
async def remove_member(
    project_id: str,
    target_user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        WorkspaceService.remove_member(project_id, target_user_id, current_user["user_id"])
        return {"success": True, "message": "成员已移除"}
    except Exception as e:
        raise _handle_service_error(e)


# ===========================================================================
# 版本化掩模与配置
# ===========================================================================

@router.post("/projects/{project_id}/versions", response_model=ProjectVersionInfo, summary="创建版本快照")
async def create_version(
    project_id: str,
    req: ProjectVersionCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return WorkspaceService.create_version(project_id, current_user["user_id"], req)
    except Exception as e:
        raise _handle_service_error(e)


@router.get("/projects/{project_id}/versions", response_model=ProjectVersionListResponse, summary="列出项目所有版本")
async def list_versions(
    project_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        total, versions = WorkspaceService.list_versions(
            project_id, current_user["user_id"], limit=limit, offset=offset,
        )
        return ProjectVersionListResponse(count=total, versions=versions)
    except Exception as e:
        raise _handle_service_error(e)


@router.get("/projects/{project_id}/versions/{version_id}", response_model=ProjectVersionInfo, summary="获取版本信息")
async def get_version(
    project_id: str,
    version_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return WorkspaceService.get_version(project_id, version_id, current_user["user_id"])
    except Exception as e:
        raise _handle_service_error(e)


@router.get("/projects/{project_id}/versions/{version_id}/mask", summary="获取版本的掩模数据")
async def get_version_mask(
    project_id: str,
    version_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        mask = WorkspaceService.get_version_mask(project_id, version_id, current_user["user_id"])
        if mask is None:
            raise HTTPException(status_code=404, detail="该版本无掩模数据")
        return {"success": True, "mask": mask}
    except HTTPException:
        raise
    except Exception as e:
        raise _handle_service_error(e)


@router.get("/projects/{project_id}/versions/{version_id}/config", summary="获取版本的配置数据")
async def get_version_config(
    project_id: str,
    version_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        cfg = WorkspaceService.get_version_config(project_id, version_id, current_user["user_id"])
        if cfg is None:
            raise HTTPException(status_code=404, detail="该版本无配置数据")
        return {"success": True, "config": cfg}
    except HTTPException:
        raise
    except Exception as e:
        raise _handle_service_error(e)


@router.delete("/projects/{project_id}/versions/{version_id}", summary="删除版本")
async def delete_version(
    project_id: str,
    version_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        WorkspaceService.delete_version(project_id, version_id, current_user["user_id"])
        return {"success": True, "message": "版本已删除"}
    except Exception as e:
        raise _handle_service_error(e)


# ===========================================================================
# 热点评论标注
# ===========================================================================

@router.post("/projects/{project_id}/comments", response_model=HotspotCommentInfo, summary="添加评论/热点标注")
async def add_comment(
    project_id: str,
    req: HotspotCommentCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return WorkspaceService.add_comment(project_id, current_user["user_id"], req)
    except Exception as e:
        raise _handle_service_error(e)


@router.get("/projects/{project_id}/comments", response_model=HotspotCommentListResponse, summary="查询评论列表")
async def list_comments(
    project_id: str,
    version_id: Optional[str] = Query(None, description="按版本过滤"),
    severity: Optional[str] = Query(None, description="按严重程度过滤: info/warning/error/critical"),
    category: Optional[str] = Query(None, description="按分类过滤"),
    resolved: Optional[bool] = Query(None, description="是否已解决: true/false"),
    reply_to: Optional[str] = Query(None, description="某条评论的回复列表"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        total, comments = WorkspaceService.list_comments(
            project_id, current_user["user_id"],
            version_id=version_id,
            severity=severity,
            category=category,
            resolved=resolved,
            reply_to=reply_to,
            limit=limit,
            offset=offset,
        )
        return HotspotCommentListResponse(count=total, comments=comments)
    except Exception as e:
        raise _handle_service_error(e)


@router.post("/projects/{project_id}/comments/{comment_id}/resolve", response_model=HotspotCommentInfo, summary="标记评论已解决")
async def resolve_comment(
    project_id: str,
    comment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return WorkspaceService.resolve_comment(project_id, comment_id, current_user["user_id"])
    except Exception as e:
        raise _handle_service_error(e)


@router.delete("/projects/{project_id}/comments/{comment_id}", summary="删除评论")
async def delete_comment(
    project_id: str,
    comment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        WorkspaceService.delete_comment(project_id, comment_id, current_user["user_id"])
        return {"success": True, "message": "评论已删除"}
    except Exception as e:
        raise _handle_service_error(e)


# ===========================================================================
# Fork 他人实验参数
# ===========================================================================

@router.post("/forks", summary="Fork 项目到自己的空间")
async def fork_project(
    req: ForkCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        fork_info, new_project = WorkspaceService.fork_project(
            user_id=current_user["user_id"],
            username=current_user["username"],
            req=req,
        )
        return {
            "success": True,
            "message": "Fork 成功",
            "fork": fork_info.model_dump(),
            "project": new_project.model_dump(),
        }
    except Exception as e:
        raise _handle_service_error(e)


@router.get("/forks", response_model=ForkListResponse, summary="列出 Fork 记录")
async def list_forks(
    source_project_id: Optional[str] = Query(None, description="按源项目过滤"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        total, forks = WorkspaceService.list_forks(
            user_id=current_user["user_id"],
            source_project_id=source_project_id,
            limit=limit,
            offset=offset,
        )
        return ForkListResponse(count=total, forks=forks)
    except Exception as e:
        raise _handle_service_error(e)
