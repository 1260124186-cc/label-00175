# -*- coding: utf-8 -*-
"""
协作式研究空间模块

支持多人共享项目、版本化掩模与配置、评论标注热点区域、
fork 他人实验参数，与 JWT 多租户配合，升级为课题组协作平台。
"""

from .schemas import (
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
from .services import WorkspaceService

__all__ = [
    "ProjectCreate",
    "ProjectUpdate",
    "ProjectInfo",
    "ProjectListResponse",
    "ProjectMemberAdd",
    "ProjectMemberUpdate",
    "ProjectMemberInfo",
    "ProjectVersionCreate",
    "ProjectVersionInfo",
    "ProjectVersionListResponse",
    "HotspotCommentCreate",
    "HotspotCommentInfo",
    "HotspotCommentListResponse",
    "ForkCreate",
    "ForkInfo",
    "ForkListResponse",
    "WorkspaceService",
]
