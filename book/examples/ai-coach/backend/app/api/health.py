"""健康检查接口"""
from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter()


@router.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }
