# -*- coding: utf-8 -*-
"""
Workspace 模块数据模型 (Pydantic Schemas)
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 权限角色定义
# ---------------------------------------------------------------------------
ROLE_OWNER = "owner"
ROLE_EDITOR = "editor"
ROLE_VIEWER = "viewer"
VALID_ROLES = {ROLE_OWNER, ROLE_EDITOR, ROLE_VIEWER}


# ---------------------------------------------------------------------------
# 项目相关
# ---------------------------------------------------------------------------
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="项目名称")
    description: Optional[str] = Field(None, max_length=2048, description="项目描述")
    visibility: str = Field(
        "private",
        description="可见性: private(私有), shared(共享), public(公开)",
    )
    tags: List[str] = Field(default_factory=list, description="标签列表")
    config: Optional[Dict[str, Any]] = Field(None, description="初始配置")

    @classmethod
    def validate_visibility(cls, v: str) -> str:
        valid = ["private", "shared", "public"]
        if v not in valid:
            raise ValueError(f"visibility 必须为以下之一: {valid}")
        return v


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128, description="项目名称")
    description: Optional[str] = Field(None, max_length=2048, description="项目描述")
    visibility: Optional[str] = Field(None, description="可见性")
    tags: Optional[List[str]] = Field(None, description="标签列表")
    archived: Optional[bool] = Field(None, description="是否归档")


class ProjectMemberAdd(BaseModel):
    username: str = Field(..., description="要添加的用户名")
    role: str = Field(ROLE_VIEWER, description="角色: owner, editor, viewer")


class ProjectMemberUpdate(BaseModel):
    role: str = Field(..., description="新角色")


class ProjectMemberInfo(BaseModel):
    user_id: str
    username: str
    display_name: str
    role: str
    joined_at: str


class ProjectInfo(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    owner_id: str
    owner_username: str
    owner_display_name: str
    visibility: str
    tags: List[str] = Field(default_factory=list)
    archived: bool = False
    member_count: int = 0
    version_count: int = 0
    comment_count: int = 0
    fork_count: int = 0
    forked_from: Optional[str] = None
    created_at: str
    updated_at: str
    current_user_role: Optional[str] = None


class ProjectListResponse(BaseModel):
    count: int
    projects: List[ProjectInfo]


# ---------------------------------------------------------------------------
# 版本相关（版本化掩模与配置）
# ---------------------------------------------------------------------------
class ProjectVersionCreate(BaseModel):
    version_name: str = Field(..., min_length=1, max_length=64, description="版本名称，如 v1.0")
    change_log: Optional[str] = Field(None, max_length=2048, description="变更说明")
    mask_data: Optional[Dict[str, Any]] = Field(None, description="掩模数据 (JSON 序列化的 numpy 数组或路径)")
    config: Optional[Dict[str, Any]] = Field(None, description="仿真配置")
    experiment_run_id: Optional[str] = Field(None, description="关联的实验运行 ID")
    metrics: Optional[Dict[str, float]] = Field(None, description="关键指标快照")
    tags: List[str] = Field(default_factory=list, description="版本标签")


class ProjectVersionInfo(BaseModel):
    version_id: str
    project_id: str
    version_name: str
    change_log: Optional[str] = None
    author_id: str
    author_username: str
    author_display_name: str
    experiment_run_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    has_mask: bool = False
    has_config: bool = False
    created_at: str


class ProjectVersionListResponse(BaseModel):
    count: int
    versions: List[ProjectVersionInfo]


# ---------------------------------------------------------------------------
# 热点评论标注
# ---------------------------------------------------------------------------
class HotspotRegion(BaseModel):
    """热点区域 - 支持矩形/多边形/像素坐标标注"""
    region_type: str = Field(
        "rectangle",
        description="区域类型: rectangle, polygon, point",
    )
    x: Optional[float] = Field(None, description="左上角 x 或 单点 x (像素)")
    y: Optional[float] = Field(None, description="左上角 y 或 单点 y (像素)")
    width: Optional[float] = Field(None, description="宽度 (像素)")
    height: Optional[float] = Field(None, description="高度 (像素)")
    points: Optional[List[List[float]]] = Field(
        None,
        description="多边形顶点列表 [[x1,y1], [x2,y2], ...]",
    )
    layer: Optional[str] = Field(None, description="图层名/版本名")


class HotspotCommentCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=2048, description="评论内容")
    version_id: Optional[str] = Field(None, description="关联版本 ID")
    experiment_run_id: Optional[str] = Field(None, description="关联实验运行 ID")
    region: Optional[HotspotRegion] = Field(None, description="标注的热点区域")
    severity: str = Field(
        "info",
        description="严重程度: info, warning, error, critical",
    )
    category: Optional[str] = Field(
        None,
        description="分类: epe_violation, pvb_violation, mrc_violation, performance, suggestion, other",
    )
    reply_to: Optional[str] = Field(None, description="回复的评论 ID")


class HotspotCommentInfo(BaseModel):
    comment_id: str
    project_id: str
    author_id: str
    author_username: str
    author_display_name: str
    content: str
    version_id: Optional[str] = None
    experiment_run_id: Optional[str] = None
    region: Optional[Dict[str, Any]] = None
    severity: str = "info"
    category: Optional[str] = None
    reply_to: Optional[str] = None
    reply_count: int = 0
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    created_at: str


class HotspotCommentListResponse(BaseModel):
    count: int
    comments: List[HotspotCommentInfo]


# ---------------------------------------------------------------------------
# Fork 相关
# ---------------------------------------------------------------------------
class ForkCreate(BaseModel):
    source_project_id: str = Field(..., description="源项目 ID")
    new_name: Optional[str] = Field(None, max_length=128, description="新项目名，留空则自动生成")
    include_versions: bool = Field(True, description="是否包含所有版本历史")
    include_comments: bool = Field(False, description="是否包含评论")
    description: Optional[str] = Field(None, max_length=2048, description="新项目描述")


class ForkInfo(BaseModel):
    fork_id: str
    source_project_id: str
    source_project_name: str
    source_owner_username: str
    forked_project_id: str
    forked_project_name: str
    forked_by_user_id: str
    forked_by_username: str
    include_versions: bool
    include_comments: bool
    created_at: str


class ForkListResponse(BaseModel):
    count: int
    forks: List[ForkInfo]
