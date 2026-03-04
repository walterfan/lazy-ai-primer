"""Pydantic 数据模型"""
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ── 书稿相关 ──────────────────────────────

class ChapterCreate(BaseModel):
    """创建章节"""
    book_id: str = Field(default="default")
    number: int = Field(..., description="章节编号")
    title: str = Field(..., description="章节标题")
    content: str = Field(default="", description="章节内容（Markdown）")


class ChapterUpdate(BaseModel):
    """更新章节"""
    title: str | None = None
    content: str | None = None


class ChapterResponse(BaseModel):
    """章节响应"""
    id: str
    book_id: str
    number: int
    title: str
    content: str
    word_count: int
    version: int
    created_at: datetime
    updated_at: datetime


# ── 编辑操作相关 ──────────────────────────────

class EditAction(str, Enum):
    PROOFREAD = "proofread"      # 校对：修正错别字、语法、标点
    POLISH = "polish"            # 润色：改善表达、提升可读性
    EXPAND = "expand"            # 扩写：补充内容、增加细节
    CONDENSE = "condense"        # 缩写：精简内容、去除冗余
    RESTRUCTURE = "restructure"  # 重构：调整结构、重新组织
    TRANSLATE = "translate"      # 翻译：中英互译
    REVIEW = "review"            # 审查：给出修改建议，不直接修改


class EditRequest(BaseModel):
    """编辑请求"""
    chapter_id: str = Field(..., description="章节 ID")
    action: EditAction = Field(..., description="编辑操作类型")
    selection: str = Field(default="", description="选中的文本（空则处理全文）")
    instruction: str = Field(default="", description="额外指令")
    target_language: str = Field(default="en", description="翻译目标语言")


class EditResponse(BaseModel):
    """编辑响应"""
    original: str
    edited: str
    action: EditAction
    diff_html: str = ""
    suggestions: list[str] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)


# ── 写作辅助相关 ──────────────────────────────

class WriteRequest(BaseModel):
    """写作请求"""
    topic: str = Field(..., description="写作主题")
    outline: str = Field(default="", description="大纲（可选）")
    style: str = Field(default="technical", description="风格：technical/casual/academic")
    word_count: int = Field(default=2000, description="目标字数")
    context: str = Field(default="", description="上下文（前后章节摘要）")


class WriteResponse(BaseModel):
    """写作响应"""
    content: str
    word_count: int
    outline_used: str


# ── 对话相关 ──────────────────────────────

class ChatRequest(BaseModel):
    """编辑对话请求"""
    message: str
    session_id: str = Field(default="")
    chapter_id: str = Field(default="", description="关联的章节 ID")


class ChatResponse(BaseModel):
    """编辑对话响应"""
    reply: str
    session_id: str


# ── 认证相关 ──────────────────────────────

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: str = Field(default="")


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict
