"""知识库 API — 文档管理与知识检索"""
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from loguru import logger

from app.models.schemas import (
    DocumentUpload,
    DocumentResponse,
    KnowledgeQuery,
    KnowledgeResponse,
)
from app.core.database import async_session
from app.models.db_models import Document
from sqlalchemy import select

router = APIRouter()


@router.post("/documents", response_model=DocumentResponse)
async def upload_document(doc: DocumentUpload, request: Request):
    """上传文档到知识库"""
    rag_engine = request.app.state.rag_engine

    # 保存到数据库
    async with async_session() as session:
        db_doc = Document(
            title=doc.title,
            content=doc.content,
            tags=doc.tags,
            source=doc.source,
        )
        session.add(db_doc)
        await session.flush()
        doc_id = db_doc.id

        # 添加到 RAG 索引
        chunk_count = await rag_engine.add_document(
            title=doc.title,
            content=doc.content,
            tags=doc.tags,
            doc_id=doc_id,
        )

        db_doc.chunk_count = chunk_count
        await session.commit()

        logger.info(f"文档上传成功: {doc.title} (ID: {doc_id}, {chunk_count} chunks)")

        return DocumentResponse(
            id=doc_id,
            title=doc.title,
            tags=doc.tags,
            source=doc.source,
            chunk_count=chunk_count,
            created_at=db_doc.created_at,
        )


@router.post("/documents/file")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    tags: str = "",
):
    """上传文件到知识库（PDF、TXT、MD）"""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    doc = DocumentUpload(
        title=file.filename or "未命名文件",
        content=text,
        tags=tag_list,
        source="upload",
    )
    return await upload_document(doc, request)


@router.get("/documents")
async def list_documents():
    """列出所有文档"""
    async with async_session() as session:
        result = await session.execute(
            select(Document).order_by(Document.created_at.desc())
        )
        docs = result.scalars().all()
        return [
            {
                "id": d.id,
                "title": d.title,
                "tags": d.tags,
                "source": d.source,
                "chunk_count": d.chunk_count,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    """删除文档"""
    rag_engine = request.app.state.rag_engine

    async with async_session() as session:
        result = await session.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        await rag_engine.delete_document(doc_id)
        await session.delete(doc)
        await session.commit()

    return {"message": f"文档 {doc_id} 已删除"}


@router.post("/query", response_model=KnowledgeResponse)
async def query_knowledge(query: KnowledgeQuery, request: Request):
    """查询知识库"""
    rag_engine = request.app.state.rag_engine

    result = await rag_engine.query(
        question=query.question,
        top_k=query.top_k,
        tags=query.tags,
    )

    return KnowledgeResponse(**result)


@router.get("/stats")
async def knowledge_stats(request: Request):
    """知识库统计"""
    rag_engine = request.app.state.rag_engine
    return rag_engine.get_stats()
