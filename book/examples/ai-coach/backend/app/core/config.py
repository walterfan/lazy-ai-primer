"""应用配置 — 所有配置项集中管理"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置，支持 .env 文件和环境变量"""
    
    # 应用
    APP_NAME: str = "AI Coach - 虚拟教练"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # 数据库
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/coach.db"
    
    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # RAG
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    SIMILARITY_TOP_K: int = 5
    
    # 学习计划
    DAILY_REMINDER_HOUR: int = 9  # 每天 9 点提醒
    
    # JWT
    JWT_SECRET: str = "your-secret-key-change-in-production"
    JWT_EXPIRE_MINUTES: int = 60

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
