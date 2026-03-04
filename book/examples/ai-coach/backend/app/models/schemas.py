"""Pydantic 数据模型 — API 请求/响应的数据结构"""
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ── 知识库相关 ──────────────────────────────

class DocumentUpload(BaseModel):
    """文档上传请求"""
    title: str = Field(..., description="文档标题")
    content: str = Field(default="", description="文档内容（纯文本）")
    tags: list[str] = Field(default_factory=list, description="标签")
    source: str = Field(default="manual", description="来源：manual/upload/url")


class DocumentResponse(BaseModel):
    """文档响应"""
    id: str
    title: str
    tags: list[str]
    source: str
    chunk_count: int
    created_at: datetime


class KnowledgeQuery(BaseModel):
    """知识库查询"""
    question: str = Field(..., description="查询问题")
    top_k: int = Field(default=5, ge=1, le=20)
    tags: list[str] = Field(default_factory=list, description="过滤标签")


class KnowledgeResponse(BaseModel):
    """知识库查询响应"""
    answer: str
    sources: list[dict]
    confidence: float


# ── 学习计划相关 ──────────────────────────────

class LearningGoal(BaseModel):
    """学习目标"""
    topic: str = Field(..., description="学习主题，如 'Python 异步编程'")
    target: str = Field(..., description="目标描述，如 '能独立编写异步爬虫'")
    deadline: datetime | None = Field(default=None, description="截止日期")
    daily_minutes: int = Field(default=60, description="每日学习时长（分钟）")


class StudySession(BaseModel):
    """学习记录"""
    goal_id: str
    duration_minutes: int = Field(..., ge=1, description="学习时长（分钟）")
    notes: str = Field(default="", description="学习笔记")
    difficulty: int = Field(default=3, ge=1, le=5, description="难度 1-5")


class ProgressReport(BaseModel):
    """学习进度报告"""
    goal_id: str
    total_hours: float
    streak_days: int
    completion_pct: float
    coach_feedback: str
    suggestions: list[str]


# ── 对话相关 ──────────────────────────────

class ChatRole(str, Enum):
    USER = "user"
    COACH = "coach"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """对话消息"""
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., description="用户消息")
    session_id: str = Field(default="", description="会话 ID，空则新建")
    mode: str = Field(default="coach", description="模式：coach/tutor/quiz")


class ChatResponse(BaseModel):
    """对话响应"""
    reply: str
    session_id: str
    mode: str
    sources: list[dict] = Field(default_factory=list)


# ── 认证相关 ──────────────────────────────

class UserRegister(BaseModel):
    """用户注册"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: str = Field(default="")


class UserLogin(BaseModel):
    """用户登录"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token 响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict
