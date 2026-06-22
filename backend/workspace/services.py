# -*- coding: utf-8 -*-
"""
Workspace 服务层

实现协作式研究空间的业务逻辑，包括：
- 项目管理与权限校验
- 成员管理（共享）
- 版本化掩模与配置
- 热点评论标注
- Fork 他人实验参数
"""

import os
import sys
import copy
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from . import storage
from .schemas import (
    ProjectCreate,
    ProjectUpdate,
    ProjectInfo,
    ProjectMemberAdd,
    ProjectMemberUpdate,
    ProjectMemberInfo,
    ProjectVersionCreate,
    ProjectVersionInfo,
    HotspotCommentCreate,
    HotspotCommentInfo,
    ForkCreate,
    ForkInfo,
    ROLE_OWNER,
    ROLE_EDITOR,
    ROLE_VIEWER,
    VALID_ROLES,
)

_WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_WORKSPACE_DIR)
_API_DIR = os.path.join(_BACKEND_ROOT, "api")
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from auth import _load_users  # 导入用户信息查询（JSON 文件存储）

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 用户信息辅助
# ---------------------------------------------------------------------------
def _get_user_info(username: str) -> Optional[Dict[str, Any]]:
    users = _load_users()
    u = users.get(username)
    if not u:
        return None
    return {
        "user_id": u["user_id"],
        "username": u["username"],
        "display_name": u.get("display_name", u["username"]),
    }


def _get_user_info_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    users = _load_users()
    for u in users.values():
        if u.get("user_id") == user_id:
            return {
                "user_id": u["user_id"],
                "username": u["username"],
                "display_name": u.get("display_name", u["username"]),
            }
    return None


def _require_user_info(user_id: str) -> Dict[str, Any]:
    u = _get_user_info_by_id(user_id)
    if u is None:
        return {
            "user_id": user_id,
            "username": f"user_{user_id[:8]}",
            "display_name": f"用户 {user_id[:8]}",
        }
    return u


# ---------------------------------------------------------------------------
# 权限校验
# ---------------------------------------------------------------------------
def _get_user_role_in_project(project_id: str, user_id: str) -> Optional[str]:
    members = storage.load_project_members(project_id)
    return members.get(user_id, {}).get("role")


def _check_permission(
    project_id: str,
    user_id: str,
    allowed_roles: Optional[set] = None,
    require_owner: bool = False,
) -> Optional[str]:
    """
    校验用户对项目的访问权限。

    Returns: 用户的角色，若无权访问则返回 None。
    """
    meta = storage.load_project_meta(project_id)
    if not meta:
        return None

    role = _get_user_role_in_project(project_id, user_id)

    # Owner / 成员直接访问
    if role in VALID_ROLES:
        if require_owner and role != ROLE_OWNER:
            return None
        if allowed_roles and role not in allowed_roles:
            return None
        return role

    # 非成员情况 - 仅根据 visibility 判断
    visibility = meta.get("visibility", "private")
    if visibility == "public":
        # 公开项目所有人拥有 viewer 权限（只读）
        viewer_role = ROLE_VIEWER
        if allowed_roles and viewer_role not in allowed_roles:
            return None
        if require_owner:
            return None
        return viewer_role

    if visibility == "shared":
        # shared 项目 - 不在成员列表中则不可访问
        return None

    # private - 仅成员可访问
    return None


def _assert_permission(
    project_id: str,
    user_id: str,
    allowed_roles: Optional[set] = None,
    require_owner: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """权限校验失败抛异常"""
    meta = storage.load_project_meta(project_id)
    if not meta:
        raise ValueError(f"项目不存在: {project_id}")
    role = _check_permission(project_id, user_id, allowed_roles, require_owner)
    if role is None:
        raise PermissionError("无权访问该项目")
    return role, meta


# ---------------------------------------------------------------------------
# WorkspaceService
# ---------------------------------------------------------------------------
class WorkspaceService:
    """协作式研究空间服务"""

    # ===================================================================
    # 项目管理
    # ===================================================================
    @staticmethod
    def create_project(user_id: str, username: str, req: ProjectCreate) -> ProjectInfo:
        """创建新项目"""
        with storage._lock:
            project_id = storage._gen_id("proj")
            now = storage._now_iso()

            owner_info = _require_user_info(user_id)

            meta = {
                "project_id": project_id,
                "name": req.name,
                "description": req.description,
                "owner_id": user_id,
                "owner_username": owner_info["username"],
                "owner_display_name": owner_info["display_name"],
                "visibility": req.visibility if req.visibility in ("private", "shared", "public") else "private",
                "tags": list(req.tags or []),
                "archived": False,
                "forked_from": None,
                "created_at": now,
                "updated_at": now,
            }
            storage.save_project_meta(meta)

            # Owner 加入成员列表
            members = {
                user_id: {
                    "user_id": user_id,
                    "username": owner_info["username"],
                    "display_name": owner_info["display_name"],
                    "role": ROLE_OWNER,
                    "joined_at": now,
                }
            }
            storage.save_project_members(project_id, members)
            storage._update_user_index(user_id, project_id, ROLE_OWNER)

            # 初始化评论
            storage.save_comments(project_id, {})

            logger.info(f"项目创建成功: {project_id} 由 {username} 创建")
            return WorkspaceService._to_project_info(meta, user_id)

    @staticmethod
    def get_project(project_id: str, user_id: str) -> ProjectInfo:
        """获取单个项目详情"""
        role, meta = _assert_permission(project_id, user_id)
        return WorkspaceService._to_project_info(meta, user_id, effective_role=role)

    @staticmethod
    def list_projects(
        user_id: str,
        scope: str = "all",
        visibility: Optional[str] = None,
        tag: Optional[str] = None,
        keyword: Optional[str] = None,
        archived: Optional[bool] = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[int, List[ProjectInfo]]:
        """
        列出项目。

        scope:
          - mine: 我是成员的项目
          - public: 所有公开项目
          - shared: 共享给我的 + 公开
          - all: 我能看到的所有 (mine + public)
        """
        user_idx_path = storage._user_index_path()
        user_idx = storage._read_json(user_idx_path)
        my_projects = {e["project_id"]: e["role"] for e in user_idx.get(user_id, [])}

        all_pids = storage.list_all_project_ids()
        results: List[ProjectInfo] = []

        for pid in all_pids:
            meta = storage.load_project_meta(pid)
            if not meta:
                continue

            is_member = pid in my_projects
            vis = meta.get("visibility", "private")
            is_archived = meta.get("archived", False)

            if archived is not None and is_archived != archived:
                continue

            # 根据 scope 过滤
            if scope == "mine":
                if not is_member:
                    continue
            elif scope == "public":
                if vis != "public":
                    continue
            elif scope == "shared":
                if not (is_member or vis == "public"):
                    continue
            # "all" 包含我能看到的一切：成员 + public

            if visibility and vis != visibility:
                continue

            if tag and tag not in meta.get("tags", []):
                continue

            if keyword:
                kw = keyword.lower()
                haystack = " ".join([
                    meta.get("name", ""),
                    meta.get("description", ""),
                    " ".join(meta.get("tags", [])),
                ]).lower()
                if kw not in haystack:
                    continue

            # 权限检查（非成员只能看 public）
            effective_role: Optional[str] = my_projects.get(pid)
            if not effective_role and vis == "public":
                effective_role = ROLE_VIEWER
            if effective_role is None:
                continue

            info = WorkspaceService._to_project_info(meta, user_id, effective_role=effective_role)
            results.append(info)

        results.sort(key=lambda p: p.updated_at, reverse=True)
        total = len(results)
        paged = results[offset:offset + limit]
        return total, paged

    @staticmethod
    def update_project(project_id: str, user_id: str, req: ProjectUpdate) -> ProjectInfo:
        """更新项目信息（owner/editor 均可更新元信息；archived 仅 owner）"""
        with storage._lock:
            role, meta = _assert_permission(project_id, user_id)
            if req.archived is not None:
                if role != ROLE_OWNER:
                    raise PermissionError("仅项目所有者可归档/取消归档项目")

            if req.name is not None:
                meta["name"] = req.name
            if req.description is not None:
                meta["description"] = req.description
            if req.visibility is not None:
                if req.visibility not in ("private", "shared", "public"):
                    raise ValueError("visibility 非法")
                meta["visibility"] = req.visibility
            if req.tags is not None:
                meta["tags"] = list(req.tags)
            if req.archived is not None:
                meta["archived"] = req.archived
            meta["updated_at"] = storage._now_iso()

            storage.save_project_meta(meta)
            return WorkspaceService._to_project_info(meta, user_id, effective_role=role)

    @staticmethod
    def delete_project(project_id: str, user_id: str) -> None:
        """删除项目（仅 owner）"""
        with storage._lock:
            _, meta = _assert_permission(project_id, user_id, require_owner=True)
            # 从用户索引中移除所有成员
            members = storage.load_project_members(project_id)
            for uid in members.keys():
                storage._remove_from_user_index(uid, project_id)
            storage.delete_project_storage(project_id)
            logger.info(f"项目已删除: {project_id}")

    # ===================================================================
    # 成员管理（项目共享）
    # ===================================================================
    @staticmethod
    def add_member(project_id: str, current_user_id: str, req: ProjectMemberAdd) -> ProjectMemberInfo:
        """添加项目成员（owner 或 editor 可加人，但只有 owner 能加 owner）"""
        with storage._lock:
            role, _ = _assert_permission(project_id, current_user_id, allowed_roles={ROLE_OWNER, ROLE_EDITOR})

            target_role = req.role if req.role in VALID_ROLES else ROLE_VIEWER
            if target_role == ROLE_OWNER and role != ROLE_OWNER:
                raise PermissionError("仅所有者可指定 owner 角色")

            target_user = _get_user_info(req.username)
            if not target_user:
                raise ValueError(f"用户不存在: {req.username}")

            target_uid = target_user["user_id"]
            members = storage.load_project_members(project_id)
            if target_uid in members:
                raise ValueError(f"用户已是项目成员: {req.username}")

            now = storage._now_iso()
            members[target_uid] = {
                "user_id": target_uid,
                "username": target_user["username"],
                "display_name": target_user["display_name"],
                "role": target_role,
                "joined_at": now,
            }
            storage.save_project_members(project_id, members)
            storage._update_user_index(target_uid, project_id, target_role)

            # updated_at
            meta = storage.load_project_meta(project_id)
            if meta:
                meta["updated_at"] = now
                storage.save_project_meta(meta)

            return ProjectMemberInfo(
                user_id=target_uid,
                username=target_user["username"],
                display_name=target_user["display_name"],
                role=target_role,
                joined_at=now,
            )

    @staticmethod
    def update_member_role(
        project_id: str,
        target_user_id: str,
        current_user_id: str,
        req: ProjectMemberUpdate,
    ) -> ProjectMemberInfo:
        """修改成员角色"""
        with storage._lock:
            role, _ = _assert_permission(project_id, current_user_id, allowed_roles={ROLE_OWNER})
            new_role = req.role
            if new_role not in VALID_ROLES:
                raise ValueError(f"非法角色: {new_role}")

            members = storage.load_project_members(project_id)
            if target_user_id not in members:
                raise ValueError("目标用户不是项目成员")

            # 不能把自己从 owner 降职（防止孤立项目）
            if target_user_id == current_user_id and new_role != ROLE_OWNER:
                owners = [uid for uid, m in members.items() if m["role"] == ROLE_OWNER]
                if len(owners) <= 1:
                    raise PermissionError("项目至少需要一位所有者，无法移除自身的 owner 身份")

            members[target_user_id]["role"] = new_role
            storage.save_project_members(project_id, members)
            storage._update_user_index(target_user_id, project_id, new_role)

            info = members[target_user_id]
            return ProjectMemberInfo(
                user_id=info["user_id"],
                username=info["username"],
                display_name=info["display_name"],
                role=new_role,
                joined_at=info["joined_at"],
            )

    @staticmethod
    def remove_member(project_id: str, target_user_id: str, current_user_id: str) -> None:
        """移除成员"""
        with storage._lock:
            role, _ = _assert_permission(project_id, current_user_id, allowed_roles={ROLE_OWNER})

            members = storage.load_project_members(project_id)
            if target_user_id not in members:
                raise ValueError("目标用户不是项目成员")

            # 保护最后一个 owner
            if members[target_user_id]["role"] == ROLE_OWNER:
                owners = [uid for uid, m in members.items() if m["role"] == ROLE_OWNER]
                if len(owners) <= 1:
                    raise PermissionError("项目至少需要一位所有者")

            del members[target_user_id]
            storage.save_project_members(project_id, members)
            storage._remove_from_user_index(target_user_id, project_id)

            meta = storage.load_project_meta(project_id)
            if meta:
                meta["updated_at"] = storage._now_iso()
                storage.save_project_meta(meta)

    @staticmethod
    def list_members(project_id: str, user_id: str) -> List[ProjectMemberInfo]:
        """列出所有成员"""
        role, _ = _assert_permission(project_id, user_id)
        members = storage.load_project_members(project_id)
        result = []
        for uid, info in members.items():
            result.append(ProjectMemberInfo(
                user_id=info["user_id"],
                username=info["username"],
                display_name=info["display_name"],
                role=info["role"],
                joined_at=info["joined_at"],
            ))
        # Owner 先排
        role_order = {ROLE_OWNER: 0, ROLE_EDITOR: 1, ROLE_VIEWER: 2}
        result.sort(key=lambda m: (role_order.get(m.role, 99), m.joined_at))
        return result

    # ===================================================================
    # 版本化掩模与配置
    # ===================================================================
    @staticmethod
    def create_version(
        project_id: str,
        user_id: str,
        req: ProjectVersionCreate,
    ) -> ProjectVersionInfo:
        """创建版本快照（掩模 + 配置）"""
        with storage._lock:
            role, meta = _assert_permission(project_id, user_id, allowed_roles={ROLE_OWNER, ROLE_EDITOR})

            version_id = storage._gen_id("ver")
            author = _require_user_info(user_id)
            now = storage._now_iso()

            version_meta = {
                "version_id": version_id,
                "project_id": project_id,
                "version_name": req.version_name,
                "change_log": req.change_log,
                "author_id": user_id,
                "author_username": author["username"],
                "author_display_name": author["display_name"],
                "experiment_run_id": req.experiment_run_id,
                "metrics": dict(req.metrics or {}),
                "tags": list(req.tags or []),
                "has_mask": bool(req.mask_data),
                "has_config": bool(req.config),
                "created_at": now,
            }
            storage.save_version_meta(project_id, version_id, version_meta)

            if req.mask_data:
                storage.save_version_mask(project_id, version_id, {"mask": req.mask_data})
            if req.config:
                storage.save_version_config(project_id, version_id, {"config": req.config})

            if meta:
                meta["updated_at"] = now
                storage.save_project_meta(meta)

            logger.info(f"版本创建成功: project={project_id} version={version_id} by {author['username']}")
            return ProjectVersionInfo(**version_meta)

    @staticmethod
    def get_version(project_id: str, version_id: str, user_id: str) -> ProjectVersionInfo:
        """获取版本元信息"""
        _assert_permission(project_id, user_id)
        meta = storage.load_version_meta(project_id, version_id)
        if not meta:
            raise ValueError(f"版本不存在: {version_id}")
        return ProjectVersionInfo(**meta)

    @staticmethod
    def get_version_mask(project_id: str, version_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """获取版本的掩模数据"""
        _assert_permission(project_id, user_id)
        data = storage.load_version_mask(project_id, version_id)
        return data.get("mask") if data else None

    @staticmethod
    def get_version_config(project_id: str, version_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """获取版本的配置数据"""
        _assert_permission(project_id, user_id)
        data = storage.load_version_config(project_id, version_id)
        return data.get("config") if data else None

    @staticmethod
    def list_versions(
        project_id: str,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[int, List[ProjectVersionInfo]]:
        """列出项目所有版本（按创建时间倒序）"""
        _assert_permission(project_id, user_id)
        vids = storage.list_version_ids(project_id)
        versions = []
        for vid in vids:
            meta = storage.load_version_meta(project_id, vid)
            if meta:
                versions.append(ProjectVersionInfo(**meta))
        versions.sort(key=lambda v: v.created_at, reverse=True)
        total = len(versions)
        return total, versions[offset:offset + limit]

    @staticmethod
    def delete_version(project_id: str, version_id: str, user_id: str) -> None:
        """删除版本（仅 owner）"""
        with storage._lock:
            _assert_permission(project_id, user_id, allowed_roles={ROLE_OWNER})
            import shutil
            d = storage._project_version_dir(project_id, version_id)
            if d.exists():
                shutil.rmtree(d)
            meta = storage.load_project_meta(project_id)
            if meta:
                meta["updated_at"] = storage._now_iso()
                storage.save_project_meta(meta)

    # ===================================================================
    # 热点评论标注
    # ===================================================================
    @staticmethod
    def add_comment(
        project_id: str,
        user_id: str,
        req: HotspotCommentCreate,
    ) -> HotspotCommentInfo:
        """添加评论/热点标注"""
        with storage._lock:
            role, meta = _assert_permission(project_id, user_id, allowed_roles={ROLE_OWNER, ROLE_EDITOR, ROLE_VIEWER})

            comment_id = storage._gen_id("cmt")
            author = _require_user_info(user_id)
            now = storage._now_iso()

            if req.reply_to:
                # 校验被回复的评论存在
                all_comments = storage.load_comments(project_id)
                if req.reply_to not in all_comments:
                    raise ValueError(f"被回复的评论不存在: {req.reply_to}")

            comment = {
                "comment_id": comment_id,
                "project_id": project_id,
                "author_id": user_id,
                "author_username": author["username"],
                "author_display_name": author["display_name"],
                "content": req.content,
                "version_id": req.version_id,
                "experiment_run_id": req.experiment_run_id,
                "region": req.region.model_dump() if req.region else None,
                "severity": req.severity if req.severity in ("info", "warning", "error", "critical") else "info",
                "category": req.category,
                "reply_to": req.reply_to,
                "reply_count": 0,
                "resolved": False,
                "resolved_by": None,
                "resolved_at": None,
                "created_at": now,
            }

            all_comments = storage.load_comments(project_id)
            all_comments[comment_id] = comment

            # 回复计数
            if req.reply_to:
                parent = all_comments.get(req.reply_to)
                if parent:
                    parent["reply_count"] = parent.get("reply_count", 0) + 1

            storage.save_comments(project_id, all_comments)

            if meta:
                meta["updated_at"] = now
                storage.save_project_meta(meta)

            logger.info(f"评论已添加: project={project_id} comment={comment_id} by {author['username']}")
            return HotspotCommentInfo(**comment)

    @staticmethod
    def list_comments(
        project_id: str,
        user_id: str,
        version_id: Optional[str] = None,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        resolved: Optional[bool] = None,
        reply_to: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> Tuple[int, List[HotspotCommentInfo]]:
        """查询评论列表"""
        _assert_permission(project_id, user_id)
        all_comments = storage.load_comments(project_id)
        results = []
        for cid, c in all_comments.items():
            if version_id and c.get("version_id") != version_id:
                continue
            if severity and c.get("severity") != severity:
                continue
            if category and c.get("category") != category:
                continue
            if resolved is not None and c.get("resolved") != resolved:
                continue
            if reply_to and c.get("reply_to") != reply_to:
                continue
            results.append(HotspotCommentInfo(**c))

        results.sort(key=lambda c: c.created_at, reverse=True)
        total = len(results)
        return total, results[offset:offset + limit]

    @staticmethod
    def resolve_comment(
        project_id: str,
        comment_id: str,
        user_id: str,
    ) -> HotspotCommentInfo:
        """标记评论已解决（owner/editor/作者本人）"""
        with storage._lock:
            role, _ = _assert_permission(project_id, user_id, allowed_roles={ROLE_OWNER, ROLE_EDITOR})
            all_comments = storage.load_comments(project_id)
            comment = all_comments.get(comment_id)
            if not comment:
                raise ValueError(f"评论不存在: {comment_id}")

            # 作者本人也可以解决自己的评论
            author_allowed = (role in (ROLE_OWNER, ROLE_EDITOR)) or (comment["author_id"] == user_id)
            if not author_allowed:
                raise PermissionError("无权标记此评论已解决")

            resolver = _require_user_info(user_id)
            comment["resolved"] = True
            comment["resolved_by"] = resolver["username"]
            comment["resolved_at"] = storage._now_iso()
            all_comments[comment_id] = comment
            storage.save_comments(project_id, all_comments)
            return HotspotCommentInfo(**comment)

    @staticmethod
    def delete_comment(
        project_id: str,
        comment_id: str,
        user_id: str,
    ) -> None:
        """删除评论（owner/作者本人）"""
        with storage._lock:
            role, _ = _assert_permission(project_id, user_id, allowed_roles={ROLE_OWNER, ROLE_EDITOR})
            all_comments = storage.load_comments(project_id)
            comment = all_comments.get(comment_id)
            if not comment:
                raise ValueError(f"评论不存在: {comment_id}")

            can_delete = (role == ROLE_OWNER) or (comment["author_id"] == user_id)
            if not can_delete:
                raise PermissionError("无权删除此评论")

            # 若有回复，同步更新父评论 reply_count
            if comment.get("reply_to"):
                parent = all_comments.get(comment["reply_to"])
                if parent:
                    parent["reply_count"] = max(0, parent.get("reply_count", 0) - 1)

            del all_comments[comment_id]
            storage.save_comments(project_id, all_comments)

    # ===================================================================
    # Fork 他人实验参数
    # ===================================================================
    @staticmethod
    def fork_project(
        user_id: str,
        username: str,
        req: ForkCreate,
    ) -> Tuple[ForkInfo, ProjectInfo]:
        """Fork 一个项目到自己的空间"""
        with storage._lock:
            # 1. 对源项目至少需要 viewer 权限
            src_role, src_meta = _assert_permission(req.source_project_id, user_id, allowed_roles={ROLE_OWNER, ROLE_EDITOR, ROLE_VIEWER})
            if src_meta.get("visibility") not in ("public", "shared") and src_role not in VALID_ROLES:
                raise PermissionError("无权 Fork 该项目")

            # 2. 准备新项目
            new_name = req.new_name or f"{src_meta.get('name', 'forked')} (fork)"
            fork_id = storage._gen_id("fork")
            new_project_id = storage._gen_id("proj")
            now = storage._now_iso()

            me = _require_user_info(user_id)

            new_meta = copy.deepcopy(src_meta)
            new_meta["project_id"] = new_project_id
            new_meta["name"] = new_name
            if req.description is not None:
                new_meta["description"] = req.description
            new_meta["owner_id"] = user_id
            new_meta["owner_username"] = me["username"]
            new_meta["owner_display_name"] = me["display_name"]
            new_meta["visibility"] = "private"  # 默认为私有
            new_meta["forked_from"] = req.source_project_id
            new_meta["archived"] = False
            new_meta["created_at"] = now
            new_meta["updated_at"] = now
            storage.save_project_meta(new_meta)

            # 成员：只有我自己作为 owner
            members = {
                user_id: {
                    "user_id": user_id,
                    "username": me["username"],
                    "display_name": me["display_name"],
                    "role": ROLE_OWNER,
                    "joined_at": now,
                }
            }
            storage.save_project_members(new_project_id, members)
            storage._update_user_index(user_id, new_project_id, ROLE_OWNER)

            # 复制版本
            if req.include_versions:
                src_vids = storage.list_version_ids(req.source_project_id)
                for src_vid in src_vids:
                    src_vmeta = storage.load_version_meta(req.source_project_id, src_vid)
                    if not src_vmeta:
                        continue
                    new_vid = storage._gen_id("ver")
                    new_vmeta = copy.deepcopy(src_vmeta)
                    new_vmeta["version_id"] = new_vid
                    new_vmeta["project_id"] = new_project_id
                    new_vmeta["author_id"] = user_id
                    new_vmeta["author_username"] = me["username"]
                    new_vmeta["author_display_name"] = me["display_name"]
                    new_vmeta["created_at"] = now
                    storage.save_version_meta(new_project_id, new_vid, new_vmeta)

                    src_mask = storage.load_version_mask(req.source_project_id, src_vid)
                    if src_mask:
                        storage.save_version_mask(new_project_id, new_vid, copy.deepcopy(src_mask))
                    src_cfg = storage.load_version_config(req.source_project_id, src_vid)
                    if src_cfg:
                        storage.save_version_config(new_project_id, new_vid, copy.deepcopy(src_cfg))

            # 复制评论
            if req.include_comments:
                src_comments = storage.load_comments(req.source_project_id)
                new_comments = {}
                cid_mapping = {}
                for old_cid, c in src_comments.items():
                    new_cid = storage._gen_id("cmt")
                    cid_mapping[old_cid] = new_cid
                    nc = copy.deepcopy(c)
                    nc["comment_id"] = new_cid
                    nc["project_id"] = new_project_id
                    new_comments[new_cid] = nc
                # 修复 reply_to 引用
                for nc in new_comments.values():
                    if nc.get("reply_to") and nc["reply_to"] in cid_mapping:
                        nc["reply_to"] = cid_mapping[nc["reply_to"]]
                storage.save_comments(new_project_id, new_comments)
            else:
                storage.save_comments(new_project_id, {})

            # 记录 fork 关系
            fork_info = {
                "fork_id": fork_id,
                "source_project_id": req.source_project_id,
                "source_project_name": src_meta.get("name", ""),
                "source_owner_username": src_meta.get("owner_username", ""),
                "forked_project_id": new_project_id,
                "forked_project_name": new_name,
                "forked_by_user_id": user_id,
                "forked_by_username": me["username"],
                "include_versions": bool(req.include_versions),
                "include_comments": bool(req.include_comments),
                "created_at": now,
            }
            storage.save_fork(fork_id, fork_info)

            logger.info(f"Fork 成功: {req.source_project_id} -> {new_project_id} by {username}")

            return (
                ForkInfo(**fork_info),
                WorkspaceService._to_project_info(new_meta, user_id),
            )

    @staticmethod
    def list_forks(
        user_id: str,
        source_project_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[int, List[ForkInfo]]:
        """列出 Fork 记录"""
        all_forks = storage.list_all_forks()
        results = []
        for f in all_forks:
            # 仅显示我 fork 出去的，或者源项目我有访问权的
            src_project_id = f.get("source_project_id", "")
            is_my_fork = f.get("forked_by_user_id") == user_id
            can_see_src = bool(_check_permission(src_project_id, user_id))
            if not (is_my_fork or can_see_src):
                continue
            if source_project_id and src_project_id != source_project_id:
                continue
            results.append(ForkInfo(**f))

        results.sort(key=lambda f: f.created_at, reverse=True)
        total = len(results)
        return total, results[offset:offset + limit]

    # ===================================================================
    # 内部：转换工具
    # ===================================================================
    @staticmethod
    def _to_project_info(
        meta: Dict[str, Any],
        current_user_id: str,
        effective_role: Optional[str] = None,
    ) -> ProjectInfo:
        """将项目元数据转换为响应结构，并填充计数"""
        project_id = meta["project_id"]

        if effective_role is None:
            effective_role = _get_user_role_in_project(project_id, current_user_id)
            if effective_role is None and meta.get("visibility") == "public":
                effective_role = ROLE_VIEWER

        members = storage.load_project_members(project_id)
        member_count = len(members)

        version_ids = storage.list_version_ids(project_id)
        version_count = len(version_ids)

        comments = storage.load_comments(project_id)
        # 仅统计根评论（非回复）作为评论计数
        comment_count = sum(1 for c in comments.values() if not c.get("reply_to"))

        fork_count = 0
        for f in storage.list_all_forks():
            if f.get("source_project_id") == project_id:
                fork_count += 1

        return ProjectInfo(
            project_id=project_id,
            name=meta["name"],
            description=meta.get("description"),
            owner_id=meta["owner_id"],
            owner_username=meta.get("owner_username", ""),
            owner_display_name=meta.get("owner_display_name", ""),
            visibility=meta.get("visibility", "private"),
            tags=list(meta.get("tags", [])),
            archived=bool(meta.get("archived", False)),
            member_count=member_count,
            version_count=version_count,
            comment_count=comment_count,
            fork_count=fork_count,
            forked_from=meta.get("forked_from"),
            created_at=meta["created_at"],
            updated_at=meta["updated_at"],
            current_user_role=effective_role,
        )
