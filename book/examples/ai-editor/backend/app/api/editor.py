"""AI 编辑 API — 校对、润色、扩写、缩写、重构、翻译、审查"""
from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from loguru import logger

from app.models.schemas import EditRequest, EditResponse
from app.models.db_models import Chapter, EditHistory
from app.core.database import async_session
from app.agents.editor_agent import EditorAgent

router = APIRouter()


@router.post("/edit", response_model=EditResponse)
async def edit_text(req: EditRequest):
    """执行 AI 编辑操作"""
    # 获取章节内容
    async with async_session() as session:
        result = await session.execute(
            select(Chapter).where(Chapter.id == req.chapter_id)
        )
        chapter = result.scalar_one_or_none()
        if not chapter:
            raise HTTPException(status_code=404, detail="章节不存在")

        # 确定要编辑的文本
        text = req.selection if req.selection else chapter.content
        if not text.strip():
            raise HTTPException(status_code=400, detail="没有可编辑的内容")

        # 执行编辑
        agent = EditorAgent()
        result = agent.edit(
            text=text,
            action=req.action.value,
            instruction=req.instruction,
            target_language=req.target_language,
        )

        # 保存编辑历史
        history = EditHistory(
            chapter_id=req.chapter_id,
            action=req.action.value,
            original_text=text,
            edited_text=result["edited"],
            instruction=req.instruction,
        )
        session.add(history)
        await session.commit()

        logger.info(f"编辑完成: {req.action.value} on chapter {req.chapter_id}")

        return EditResponse(**result)


@router.post("/apply/{chapter_id}")
async def apply_edit(chapter_id: str, edited_content: dict):
    """将编辑结果应用到章节"""
    content = edited_content.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="内容不能为空")

    async with async_session() as session:
        result = await session.execute(
            select(Chapter).where(Chapter.id == chapter_id)
        )
        chapter = result.scalar_one_or_none()
        if not chapter:
            raise HTTPException(status_code=404, detail="章节不存在")

        chapter.content = content
        chapter.word_count = len(content)
        chapter.version += 1
        await session.commit()

        return {"message": "已应用编辑", "version": chapter.version, "word_count": chapter.word_count}


@router.get("/history/{chapter_id}")
async def get_edit_history(chapter_id: str, limit: int = 20):
    """获取编辑历史"""
    async with async_session() as session:
        result = await session.execute(
            select(EditHistory)
            .where(EditHistory.chapter_id == chapter_id)
            .order_by(EditHistory.created_at.desc())
            .limit(limit)
        )
        records = result.scalars().all()
        return [
            {
                "id": r.id,
                "action": r.action,
                "instruction": r.instruction,
                "original_length": len(r.original_text),
                "edited_length": len(r.edited_text),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
