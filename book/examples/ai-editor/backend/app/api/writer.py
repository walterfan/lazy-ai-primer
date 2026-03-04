"""AI 写作 API — 辅助生成书稿内容"""
from fastapi import APIRouter
from loguru import logger

from app.models.schemas import WriteRequest, WriteResponse
from app.agents.editor_agent import WriterAgent

router = APIRouter()


@router.post("/generate", response_model=WriteResponse)
async def generate_content(req: WriteRequest):
    """AI 辅助生成书稿内容"""
    agent = WriterAgent()
    result = agent.write(
        topic=req.topic,
        outline=req.outline,
        style=req.style,
        word_count=req.word_count,
        context=req.context,
    )
    return WriteResponse(**result)
