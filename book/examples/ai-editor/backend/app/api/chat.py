"""编辑对话 API"""
from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from loguru import logger

from app.models.schemas import ChatRequest, ChatResponse
from app.models.db_models import ChatSession, Chapter
from app.core.database import async_session
from app.agents.chat_agent import ChatAgent

router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def send_message(req: ChatRequest):
    """发送消息给 AI 编辑"""
    async with async_session() as session:
        # 获取或创建会话
        if req.session_id:
            result = await session.execute(
                select(ChatSession).where(ChatSession.id == req.session_id)
            )
            chat_session = result.scalar_one_or_none()
            if not chat_session:
                raise HTTPException(status_code=404, detail="会话不存在")
        else:
            chat_session = ChatSession(chapter_id=req.chapter_id, messages=[])
            session.add(chat_session)
            await session.flush()

        history = chat_session.messages or []

        # 获取关联章节内容
        chapter_content = ""
        chapter_id = req.chapter_id or chat_session.chapter_id
        if chapter_id:
            result = await session.execute(
                select(Chapter).where(Chapter.id == chapter_id)
            )
            chapter = result.scalar_one_or_none()
            if chapter:
                chapter_content = chapter.content

        # 添加用户消息
        history.append({"role": "user", "content": req.message})

        # 调用 Agent
        agent = ChatAgent()
        reply = agent.respond(
            message=req.message,
            history=history,
            chapter_content=chapter_content,
        )

        # 保存回复
        history.append({"role": "editor", "content": reply})
        chat_session.messages = history
        await session.commit()

        return ChatResponse(
            reply=reply,
            session_id=chat_session.id,
        )


@router.get("/sessions")
async def list_sessions(limit: int = 20):
    """列出对话会话"""
    async with async_session() as session:
        result = await session.execute(
            select(ChatSession)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
        )
        sessions = result.scalars().all()
        return [
            {
                "id": s.id,
                "chapter_id": s.chapter_id,
                "message_count": len(s.messages) if s.messages else 0,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in sessions
        ]


@router.post("/message/stream")
async def send_message_stream(req: ChatRequest):
    """流式发送消息给 AI 编辑（SSE）"""
    import json
    from fastapi.responses import StreamingResponse

    async def event_generator():
        async with async_session() as session:
            if req.session_id:
                result = await session.execute(
                    select(ChatSession).where(ChatSession.id == req.session_id)
                )
                chat_session = result.scalar_one_or_none()
                if not chat_session:
                    yield f"data: {json.dumps({'error': '会话不存在'})}\n\n"
                    return
            else:
                chat_session = ChatSession(chapter_id=req.chapter_id, messages=[])
                session.add(chat_session)
                await session.flush()

            history = chat_session.messages or []

            # 获取章节内容
            chapter_content = ""
            chapter_id = req.chapter_id or chat_session.chapter_id
            if chapter_id:
                result = await session.execute(
                    select(Chapter).where(Chapter.id == chapter_id)
                )
                chapter = result.scalar_one_or_none()
                if chapter:
                    chapter_content = chapter.content

            history.append({"role": "user", "content": req.message})

            # 发送开始信号
            yield f"data: {json.dumps({'session_id': chat_session.id, 'type': 'start'})}\n\n"

            # 流式生成
            agent = ChatAgent()
            full_reply = ""
            for token in agent.respond_stream(
                message=req.message,
                history=history,
                chapter_content=chapter_content,
            ):
                full_reply += token
                yield f"data: {json.dumps({'token': token, 'type': 'token'})}\n\n"

            # 完成
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            history.append({"role": "editor", "content": full_reply})
            chat_session.messages = history
            await session.commit()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
