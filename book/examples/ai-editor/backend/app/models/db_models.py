"""数据库模型"""
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.core.database import Base
import uuid


def gen_id() -> str:
    return str(uuid.uuid4())[:8]


class Book(Base):
    """书籍"""
    __tablename__ = "books"

    id = Column(String, primary_key=True, default=gen_id)
    title = Column(String(200), nullable=False)
    author = Column(String(100), default="")
    description = Column(Text, default="")
    created_at = Column(DateTime, server_default=func.now())


class Chapter(Base):
    """章节"""
    __tablename__ = "chapters"

    id = Column(String, primary_key=True, default=gen_id)
    book_id = Column(String, nullable=False, default="default")
    number = Column(Integer, nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(Text, default="")
    word_count = Column(Integer, default=0)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class EditHistory(Base):
    """编辑历史"""
    __tablename__ = "edit_history"

    id = Column(String, primary_key=True, default=gen_id)
    chapter_id = Column(String, nullable=False)
    action = Column(String(20), nullable=False)
    original_text = Column(Text, default="")
    edited_text = Column(Text, default="")
    instruction = Column(Text, default="")
    created_at = Column(DateTime, server_default=func.now())


class ChatSession(Base):
    """编辑对话会话"""
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=gen_id)
    chapter_id = Column(String, default="")
    messages = Column(JSON, default=list)
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
