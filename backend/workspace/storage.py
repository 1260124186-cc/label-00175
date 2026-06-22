# -*- coding: utf-8 -*-
"""
Workspace 存储层

基于 JSON 文件的持久化存储，与现有认证模块保持一致的存储风格。
"""

import os
import json
import time
import uuid
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from copy import deepcopy

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _get_workspace_root() -> Path:
    """获取 workspace 根目录"""
    return Path(__file__).resolve().parent


def _get_global_store() -> Path:
    """获取全局 workspace 存储目录（跨用户共享）"""
    # workspace 目录作为多租户共享存储
    store_dir = Path(__file__).resolve().parent / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def _projects_dir() -> Path:
    d = _get_global_store() / "projects"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_dir(project_id: str) -> Path:
    d = _projects_dir() / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_meta_path(project_id: str) -> Path:
    return _project_dir(project_id) / "meta.json"


def _project_members_path(project_id: str) -> Path:
    return _project_dir(project_id) / "members.json"


def _project_versions_dir(project_id: str) -> Path:
    d = _project_dir(project_id) / "versions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_version_dir(project_id: str, version_id: str) -> Path:
    d = _project_versions_dir(project_id) / version_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_version_meta(project_id: str, version_id: str) -> Path:
    return _project_version_dir(project_id, version_id) / "meta.json"


def _project_comments_path(project_id: str) -> Path:
    return _project_dir(project_id) / "comments.json"


def _forks_dir() -> Path:
    d = _get_global_store() / "forks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _user_index_path() -> Path:
    return _get_global_store() / "user_index.json"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"读取 JSON 失败 {path}: {e}")
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"写入 JSON 失败 {path}: {e}")
        raise


def _gen_id(prefix: str = "") -> str:
    raw = f"{prefix}_{time.time()}_{uuid.uuid4().hex}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_user_index(user_id: str, project_id: str, role: str) -> None:
    """维护 user_id -> [(project_id, role] 倒排索引"""
    path = _user_index_path()
    idx = _read_json(path)
    if user_id not in idx:
        idx[user_id] = []
    entries = idx[user_id]
    found = False
    for i, entry in enumerate(entries):
        if entry.get("project_id") == project_id:
            entries[i]["role"] = role
            found = True
            break
    if not found:
        entries.append({"project_id": project_id, "role": role})
    _write_json(path, idx)


def _remove_from_user_index(user_id: str, project_id: str) -> None:
    path = _user_index_path()
    idx = _read_json(path)
    if user_id in idx:
        idx[user_id] = [e for e in idx[user_id] if e.get("project_id") != project_id]
        if not idx[user_id]:
            del idx[user_id]
        _write_json(path, idx)


# ---------------------------------------------------------------------------
# 项目元数据
# ---------------------------------------------------------------------------
def save_project_meta(meta: Dict[str, Any]) -> None:
    path = _project_meta_path(meta["project_id"])
    _write_json(path, meta)


def load_project_meta(project_id: str) -> Optional[Dict[str, Any]]:
    path = _project_meta_path(project_id)
    data = _read_json(path)
    return data if data.get("project_id") else None


def list_all_project_ids() -> List[str]:
    d = _projects_dir()
    if not d.exists():
        return []
    return [p.name for p in d.iterdir() if p.is_dir()]


def delete_project_storage(project_id: str) -> None:
    import shutil
    d = _project_dir(project_id)
    if d.exists():
        shutil.rmtree(d)


# ---------------------------------------------------------------------------
# 成员
# ---------------------------------------------------------------------------
def save_project_members(project_id: str, members: Dict[str, Any]) -> None:
    path = _project_members_path(project_id)
    _write_json(path, members)


def load_project_members(project_id: str) -> Dict[str, Any]:
    path = _project_members_path(project_id)
    return _read_json(path)


# ---------------------------------------------------------------------------
# 版本
# ---------------------------------------------------------------------------
def save_version_meta(project_id: str, version_id: str, meta: Dict[str, Any]) -> None:
    path = _project_version_meta(project_id, version_id)
    _write_json(path, meta)


def load_version_meta(project_id: str, version_id: str) -> Optional[Dict[str, Any]]:
    path = _project_version_meta(project_id, version_id)
    data = _read_json(path)
    return data if data.get("version_id") else None


def list_version_ids(project_id: str) -> List[str]:
    d = _project_versions_dir(project_id)
    if not d.exists():
        return []
    version_names = [p.name for p in d.iterdir() if p.is_dir()]
    return sorted(
        version_names,
        key=lambda vid: (
            load_version_meta(project_id, vid).get("created_at", "")
            if load_version_meta(project_id, vid) is not None
            else ""
        ),
    )


def save_version_mask(project_id: str, version_id: str, mask_data: Dict[str, Any]) -> None:
    path = _project_version_dir(project_id, version_id) / "mask.json"
    _write_json(path, mask_data)


def load_version_mask(project_id: str, version_id: str) -> Optional[Dict[str, Any]]:
    path = _project_version_dir(project_id, version_id) / "mask.json"
    return _read_json(path)


def save_version_config(project_id: str, version_id: str, config: Dict[str, Any]) -> None:
    path = _project_version_dir(project_id, version_id) / "config.json"
    _write_json(path, config)


def load_version_config(project_id: str, version_id: str) -> Optional[Dict[str, Any]]:
    path = _project_version_dir(project_id, version_id) / "config.json"
    return _read_json(path)


# ---------------------------------------------------------------------------
# 评论
# ---------------------------------------------------------------------------
def save_comments(project_id: str, comments: Dict[str, Any]) -> None:
    path = _project_comments_path(project_id)
    _write_json(path, comments)


def load_comments(project_id: str) -> Dict[str, Any]:
    path = _project_comments_path(project_id)
    return _read_json(path)


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------
def save_fork(fork_id: str, data: Dict[str, Any]) -> None:
    path = _forks_dir() / f"{fork_id}.json"
    _write_json(path, data)


def load_fork(fork_id: str) -> Optional[Dict[str, Any]]:
    path = _forks_dir() / f"{fork_id}.json"
    data = _read_json(path)
    return data if data.get("fork_id") else None


def list_all_forks() -> List[Dict[str, Any]]:
    d = _forks_dir()
    if not d.exists():
        return []
    forks = []
    for f in d.iterdir():
        if f.is_file() and f.suffix == ".json":
            data = _read_json(f)
            if data.get("fork_id"):
                forks.append(data)
    return forks
