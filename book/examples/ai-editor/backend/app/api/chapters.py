"""章节管理 API"""
from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from loguru import logger

from app.models.schemas import ChapterCreate, ChapterUpdate, ChapterResponse
from app.models.db_models import Chapter
from app.core.database import async_session

router = APIRouter()


@router.post("/", response_model=ChapterResponse)
async def create_chapter(data: ChapterCreate):
    """创建章节"""
    async with async_session() as session:
        chapter = Chapter(
            book_id=data.book_id,
            number=data.number,
            title=data.title,
            content=data.content,
            word_count=len(data.content),
        )
        session.add(chapter)
        await session.commit()
        await session.refresh(chapter)

        return ChapterResponse(
            id=chapter.id,
            book_id=chapter.book_id,
            number=chapter.number,
            title=chapter.title,
            content=chapter.content,
            word_count=chapter.word_count,
            version=chapter.version,
            created_at=chapter.created_at,
            updated_at=chapter.updated_at,
        )


@router.get("/")
async def list_chapters(book_id: str = "default"):
    """列出所有章节"""
    async with async_session() as session:
        result = await session.execute(
            select(Chapter)
            .where(Chapter.book_id == book_id)
            .order_by(Chapter.number)
        )
        chapters = result.scalars().all()
        return [
            {
                "id": c.id,
                "number": c.number,
                "title": c.title,
                "word_count": c.word_count,
                "version": c.version,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            }
            for c in chapters
        ]


@router.get("/{chapter_id}", response_model=ChapterResponse)
async def get_chapter(chapter_id: str):
    """获取章节详情"""
    async with async_session() as session:
        result = await session.execute(
            select(Chapter).where(Chapter.id == chapter_id)
        )
        chapter = result.scalar_one_or_none()
        if not chapter:
            raise HTTPException(status_code=404, detail="章节不存在")

        return ChapterResponse(
            id=chapter.id,
            book_id=chapter.book_id,
            number=chapter.number,
            title=chapter.title,
            content=chapter.content,
            word_count=chapter.word_count,
            version=chapter.version,
            created_at=chapter.created_at,
            updated_at=chapter.updated_at,
        )


@router.put("/{chapter_id}")
async def update_chapter(chapter_id: str, data: ChapterUpdate):
    """更新章节"""
    async with async_session() as session:
        result = await session.execute(
            select(Chapter).where(Chapter.id == chapter_id)
        )
        chapter = result.scalar_one_or_none()
        if not chapter:
            raise HTTPException(status_code=404, detail="章节不存在")

        if data.title is not None:
            chapter.title = data.title
        if data.content is not None:
            chapter.content = data.content
            chapter.word_count = len(data.content)
            chapter.version += 1

        await session.commit()
        return {"message": "更新成功", "version": chapter.version}


@router.delete("/{chapter_id}")
async def delete_chapter(chapter_id: str):
    """删除章节"""
    async with async_session() as session:
        result = await session.execute(
            select(Chapter).where(Chapter.id == chapter_id)
        )
        chapter = result.scalar_one_or_none()
        if not chapter:
            raise HTTPException(status_code=404, detail="章节不存在")

        await session.delete(chapter)
        await session.commit()
        return {"message": f"章节 {chapter_id} 已删除"}
