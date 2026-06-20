import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends

from schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    UserInfoResponse,
    TokenResponse,
)
from auth import (
    create_user,
    authenticate_user,
    create_access_token,
    get_current_user,
    JWT_EXPIRE_HOURS,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post("/register", response_model=TokenResponse, summary="注册新用户")
async def register(req: UserRegisterRequest):
    try:
        user = create_user(
            username=req.username,
            password=req.password,
            display_name=req.display_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    token_data = {
        "user_id": user["user_id"],
        "username": user["username"],
        "display_name": user["display_name"],
    }
    access_token = create_access_token(token_data)

    return TokenResponse(
        access_token=access_token,
        expires_in=JWT_EXPIRE_HOURS * 3600,
        user=UserInfoResponse(
            user_id=user["user_id"],
            username=user["username"],
            display_name=user["display_name"],
            created_at=user.get("created_at"),
        ),
    )


@router.post("/login", response_model=TokenResponse, summary="用户登录")
async def login(req: UserLoginRequest):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token_data = {
        "user_id": user["user_id"],
        "username": user["username"],
        "display_name": user["display_name"],
    }
    access_token = create_access_token(token_data)

    return TokenResponse(
        access_token=access_token,
        expires_in=JWT_EXPIRE_HOURS * 3600,
        user=UserInfoResponse(
            user_id=user["user_id"],
            username=user["username"],
            display_name=user["display_name"],
        ),
    )


@router.get("/me", response_model=UserInfoResponse, summary="获取当前用户信息")
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return UserInfoResponse(
        user_id=current_user["user_id"],
        username=current_user["username"],
        display_name=current_user.get("display_name", current_user["username"]),
    )
