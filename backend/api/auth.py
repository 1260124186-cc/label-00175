import os
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production-litho-sim-2024")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.environ.get("JWT_EXPIRE_HOURS", "24"))

AUTH_DATA_DIR = Path(__file__).resolve().parent / "auth_data"
AUTH_DATA_DIR.mkdir(parents=True, exist_ok=True)

USERS_FILE = AUTH_DATA_DIR / "users.json"

_lock = threading.Lock()

security = HTTPBearer(auto_error=False)


def _load_users() -> Dict[str, Any]:
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载用户数据失败: {e}")
        return {}


def _save_users(users: Dict[str, Any]) -> None:
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"保存用户数据失败: {e}")


def _hash_password(password: str) -> str:
    salt = "litho_sim_salt_v1"
    return hashlib.sha256(f"{salt}{password}{salt}".encode()).hexdigest()


def _verify_password(plain: str, hashed: str) -> bool:
    return _hash_password(plain) == hashed


def create_user(username: str, password: str, display_name: Optional[str] = None) -> Dict[str, Any]:
    with _lock:
        users = _load_users()
        if username in users:
            raise ValueError(f"用户名已存在: {username}")
        user_id = hashlib.sha256(f"u_{username}_{time.time()}".encode()).hexdigest()[:16]
        user = {
            "user_id": user_id,
            "username": username,
            "password_hash": _hash_password(password),
            "display_name": display_name or username,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        users[username] = user
        _save_users(users)

        user_dir = AUTH_DATA_DIR / user_id
        (user_dir / "configs").mkdir(parents=True, exist_ok=True)
        (user_dir / "tasks").mkdir(parents=True, exist_ok=True)
        (user_dir / "gds_uploads").mkdir(parents=True, exist_ok=True)
        (user_dir / "results").mkdir(parents=True, exist_ok=True)

        logger.info(f"用户创建成功: {username} (id={user_id})")
        return {
            "user_id": user_id,
            "username": username,
            "display_name": user["display_name"],
            "created_at": user["created_at"],
        }


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    with _lock:
        users = _load_users()
    user = users.get(username)
    if not user:
        return None
    if not _verify_password(password, user["password_hash"]):
        return None
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "display_name": user["display_name"],
    }


def create_access_token(data: Dict[str, Any]) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已过期，请重新登录",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 Token",
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证凭据",
        )
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("user_id")
    username = payload.get("username")
    if not user_id or not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 中缺少用户信息",
        )
    return {
        "user_id": user_id,
        "username": username,
        "display_name": payload.get("display_name", username),
    }


def get_user_dir(user_id: str, subdir: str = "") -> Path:
    base = AUTH_DATA_DIR / user_id
    if subdir:
        target = base / subdir
    else:
        target = base
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_user_configs_dir(user_id: str) -> Path:
    return get_user_dir(user_id, "configs")


def get_user_tasks_dir(user_id: str) -> Path:
    return get_user_dir(user_id, "tasks")


def get_user_gds_dir(user_id: str) -> Path:
    return get_user_dir(user_id, "gds_uploads")


def get_user_results_dir(user_id: str) -> Path:
    return get_user_dir(user_id, "results")
