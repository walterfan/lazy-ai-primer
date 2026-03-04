"""FastAPI 应用入口"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import get_settings
from app.core.database import init_db
from app.api import knowledge, chat, learning, health, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    settings = get_settings()
    logger.info(f"🚀 启动 {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # 初始化数据库
    await init_db()
    logger.info("✅ 数据库初始化完成")
    
    # 初始化 RAG 引擎
    from app.rag.engine import RAGEngine
    app.state.rag_engine = RAGEngine()
    await app.state.rag_engine.initialize()
    logger.info("✅ RAG 引擎初始化完成")
    
    yield
    
    logger.info("👋 应用关闭")


def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(health.router, tags=["健康检查"])
    app.include_router(auth.router, prefix="/api/auth", tags=["认证"])
    app.include_router(knowledge.router, prefix="/api/knowledge", tags=["知识库"])
    app.include_router(chat.router, prefix="/api/chat", tags=["对话"])
    app.include_router(learning.router, prefix="/api/learning", tags=["学习计划"])
    
    return app


app = create_app()
