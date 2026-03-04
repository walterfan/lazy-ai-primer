"""SQLAlchemy ORM 模型 — 数据库表定义"""
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.core.database import Base
import uuid


def gen_id() -> str:
    return str(uuid.uuid4())[:8]


class Document(Base):
    """知识库文档"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=gen_id)
    title = Column(String(200), nullable=False)
    content = Column(Text, default="")
    tags = Column(JSON, default=list)
    source = Column(String(50), default="manual")
    chunk_count = Column(Integer, default=0)
    file_path = Column(String(500), default="")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class LearningGoalDB(Base):
    """学习目标"""
    __tablename__ = "learning_goals"
    
    id = Column(String, primary_key=True, default=gen_id)
    topic = Column(String(200), nullable=False)
    target = Column(Text, nullable=False)
    deadline = Column(DateTime, nullable=True)
    daily_minutes = Column(Integer, default=60)
    status = Column(String(20), default="active")  # active / completed / paused
    created_at = Column(DateTime, server_default=func.now())


class StudySessionDB(Base):
    """学习记录"""
    __tablename__ = "study_sessions"
    
    id = Column(String, primary_key=True, default=gen_id)
    goal_id = Column(String, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    notes = Column(Text, default="")
    difficulty = Column(Integer, default=3)
    created_at = Column(DateTime, server_default=func.now())


class ChatSessionDB(Base):
    """对话会话"""
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, default=gen_id)
    mode = Column(String(20), default="coach")
    messages = Column(JSON, default=list)  # [{role, content, timestamp}]
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class User(Base):
    """用户"""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_id)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(200), default="")
    password_hash = Column(String(200), nullable=False)
    role = Column(String(20), default="user")  # user / editor / admin
    created_at = Column(DateTime, server_default=func.now())
