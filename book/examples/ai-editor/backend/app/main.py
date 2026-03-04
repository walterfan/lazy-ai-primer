"""FastAPI 应用入口"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import get_settings
from app.core.database import init_db
from app.api import chapters, editor, writer, chat, health, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"🚀 启动 {settings.APP_NAME} v{settings.APP_VERSION}")

    await init_db()
    logger.info("✅ 数据库初始化完成")

    os.makedirs(settings.MANUSCRIPTS_DIR, exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    yield
    logger.info("👋 应用关闭")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["健康检查"])
    app.include_router(auth.router, prefix="/api/auth", tags=["认证"])
    app.include_router(chapters.router, prefix="/api/chapters", tags=["章节管理"])
    app.include_router(editor.router, prefix="/api/editor", tags=["AI 编辑"])
    app.include_router(writer.router, prefix="/api/writer", tags=["AI 写作"])
    app.include_router(chat.router, prefix="/api/chat", tags=["编辑对话"])

    return app


app = create_app()
