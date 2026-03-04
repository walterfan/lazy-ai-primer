"""JWT 认证与角色授权"""
import time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import async_session

# 密码哈希
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer Token 提取
security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, role: str = "user", expires_minutes: int = 60) -> str:
    """创建 JWT access token"""
    settings = get_settings()
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(minutes=expires_minutes),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def create_refresh_token(user_id: str, expires_days: int = 7) -> str:
    """创建 JWT refresh token"""
    settings = get_settings()
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": datetime.utcnow() + timedelta(days=expires_days),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    """解码并验证 JWT token"""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token 无效: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """FastAPI 依赖：获取当前认证用户"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token 中缺少用户信息")

    # 验证用户存在
    from app.models.db_models import User
    async with async_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在")

    return {
        "id": user_id,
        "role": payload.get("role", "user"),
        "username": user.username,
    }


def require_role(required_role: str):
    """角色授权装饰器工厂"""
    async def role_checker(current_user: dict = Depends(get_current_user)):
        role_hierarchy = {"admin": 3, "editor": 2, "user": 1}
        user_level = role_hierarchy.get(current_user["role"], 0)
        required_level = role_hierarchy.get(required_role, 0)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要 {required_role} 权限",
            )
        return current_user
    return role_checker
