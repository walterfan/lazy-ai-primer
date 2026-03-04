"""对话 API — AI 教练对话系统"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
import json

from app.models.schemas import ChatRequest, ChatResponse
from app.models.db_models import ChatSessionDB
from app.core.database import async_session
from app.agents.coach import CoachAgent
from sqlalchemy import select

router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def send_message(chat_req: ChatRequest, request: Request):
    """发送消息给 AI 教练"""
    rag_engine = request.app.state.rag_engine

    # 获取或创建会话
    async with async_session() as session:
        if chat_req.session_id:
            result = await session.execute(
                select(ChatSessionDB).where(ChatSessionDB.id == chat_req.session_id)
            )
            chat_session = result.scalar_one_or_none()
            if not chat_session:
                raise HTTPException(status_code=404, detail="会话不存在")
        else:
            chat_session = ChatSessionDB(mode=chat_req.mode, messages=[])
            session.add(chat_session)
            await session.flush()

        session_id = chat_session.id
        history = chat_session.messages or []

        # 添加用户消息
        history.append({
            "role": "user",
            "content": chat_req.message,
        })

        # 调用 AI 教练
        coach = CoachAgent(rag_engine=rag_engine)
        result = await coach.respond(
            message=chat_req.message,
            history=history,
            mode=chat_req.mode,
        )

        # 添加教练回复
        history.append({
            "role": "coach",
            "content": result["reply"],
        })

        # 更新会话
        chat_session.messages = history
        await session.commit()

        return ChatResponse(
            reply=result["reply"],
            session_id=session_id,
            mode=chat_req.mode,
            sources=result.get("sources", []),
        )


@router.get("/sessions")
async def list_chat_sessions(limit: int = 20):
    """列出对话会话"""
    async with async_session() as session:
        result = await session.execute(
            select(ChatSessionDB)
            .order_by(ChatSessionDB.updated_at.desc())
            .limit(limit)
        )
        sessions = result.scalars().all()
        return [
            {
                "id": s.id,
                "mode": s.mode,
                "message_count": len(s.messages) if s.messages else 0,
                "last_message": s.messages[-1]["content"][:100] if s.messages else "",
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in sessions
        ]


@router.get("/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """获取对话历史"""
    async with async_session() as session:
        result = await session.execute(
            select(ChatSessionDB).where(ChatSessionDB.id == session_id)
        )
        chat_session = result.scalar_one_or_none()
        if not chat_session:
            raise HTTPException(status_code=404, detail="会话不存在")

        return {
            "id": chat_session.id,
            "mode": chat_session.mode,
            "messages": chat_session.messages,
            "created_at": chat_session.created_at.isoformat() if chat_session.created_at else None,
        }


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """删除对话会话"""
    async with async_session() as session:
        result = await session.execute(
            select(ChatSessionDB).where(ChatSessionDB.id == session_id)
        )
        chat_session = result.scalar_one_or_none()
        if not chat_session:
            raise HTTPException(status_code=404, detail="会话不存在")

        await session.delete(chat_session)
        await session.commit()

    return {"message": f"会话 {session_id} 已删除"}


@router.post("/message/stream")
async def send_message_stream(chat_req: ChatRequest, request: Request):
    """流式发送消息给 AI 教练（SSE）"""
    rag_engine = request.app.state.rag_engine

    async def event_generator():
        async with async_session() as session:
            # 获取或创建会话
            if chat_req.session_id:
                result = await session.execute(
                    select(ChatSessionDB).where(ChatSessionDB.id == chat_req.session_id)
                )
                chat_session = result.scalar_one_or_none()
                if not chat_session:
                    yield f"data: {json.dumps({'error': '会话不存在'})}\n\n"
                    return
            else:
                chat_session = ChatSessionDB(mode=chat_req.mode, messages=[])
                session.add(chat_session)
                await session.flush()

            session_id = chat_session.id
            history = chat_session.messages or []
            history.append({"role": "user", "content": chat_req.message})

            # 流式生成
            coach = CoachAgent(rag_engine=rag_engine)
            full_reply = ""

            # 发送 session_id
            yield f"data: {json.dumps({'session_id': session_id, 'type': 'start'})}\n\n"

            chunk = None
            async for chunk in coach.respond_stream(
                message=chat_req.message,
                history=history,
                mode=chat_req.mode,
            ):
                token = chunk.get("token", "")
                if token:
                    full_reply += token
                    yield f"data: {json.dumps({'token': token, 'type': 'token'})}\n\n"

            # 发送完成信号
            sources = chunk.get("sources", []) if chunk else []
            yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"

            # 保存完整回复
            history.append({"role": "coach", "content": full_reply})
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
