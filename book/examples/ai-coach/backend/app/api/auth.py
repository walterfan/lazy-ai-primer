"""认证 API — 注册、登录、刷新 Token"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import select
from loguru import logger

from app.models.schemas import UserRegister, UserLogin, TokenResponse
from app.models.db_models import User
from app.core.database import async_session
from app.core.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
)

router = APIRouter()


@router.post("/register", response_model=TokenResponse)
async def register(data: UserRegister):
    """用户注册"""
    async with async_session() as session:
        # 检查用户名是否已存在
        result = await session.execute(
            select(User).where(User.username == data.username)
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="用户名已存在")

        # 检查邮箱是否已存在
        if data.email:
            result = await session.execute(
                select(User).where(User.email == data.email)
            )
            if result.scalar_one_or_none():
                raise HTTPException(status_code=409, detail="邮箱已被注册")

        # 创建用户
        user = User(
            username=data.username,
            email=data.email or "",
            password_hash=hash_password(data.password),
            role="user",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        logger.info(f"新用户注册: {data.username}")

        return TokenResponse(
            access_token=create_access_token(user.id, user.role),
            refresh_token=create_refresh_token(user.id),
            user={"id": user.id, "username": user.username, "role": user.role},
        )


@router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin):
    """用户登录"""
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.username == data.username)
        )
        user = result.scalar_one_or_none()

        if not user or not verify_password(data.password, user.password_hash):
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        logger.info(f"用户登录: {data.username}")

        return TokenResponse(
            access_token=create_access_token(user.id, user.role),
            refresh_token=create_refresh_token(user.id),
            user={"id": user.id, "username": user.username, "role": user.role},
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """刷新 Token"""
    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="不是有效的 refresh token")

    user_id = payload.get("sub")
    async with async_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在")

    return TokenResponse(
        access_token=create_access_token(user.id, user.role),
        refresh_token=create_refresh_token(user.id),
        user={"id": user.id, "username": user.username, "role": user.role},
    )


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """获取当前用户信息"""
    return current_user
