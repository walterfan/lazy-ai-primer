"""RAG 引擎 — 基于 LlamaIndex 的知识库检索增强生成"""
import os
from pathlib import Path
from loguru import logger

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings as LlamaSettings,
    Document as LlamaDocument,
    PromptTemplate,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from app.core.config import get_settings
from app.core.openai_client import get_openai_client_kwargs


# 自定义 QA Prompt
QA_PROMPT = PromptTemplate(
    """你是一个专业的学习教练助手。请基于以下参考资料回答用户的问题。

要求：
1. 基于参考资料回答，如果资料不足请明确说明
2. 用清晰易懂的语言解释
3. 如果适合，给出学习建议和下一步行动
4. 引用具体来源

参考资料：
{context_str}

用户问题：{query_str}

回答："""
)


class RAGEngine:
    """RAG 引擎：管理知识库的索引、检索和生成"""

    def __init__(self):
        self.settings = get_settings()
        self.index: VectorStoreIndex | None = None
        self.chroma_client = None
        self.collection = None

    async def initialize(self):
        """初始化 RAG 引擎"""
        # 配置 LlamaIndex 全局设置
        _extra = get_openai_client_kwargs(self.settings)
        LlamaSettings.llm = OpenAI(
            model=self.settings.OPENAI_MODEL,
            temperature=0.1,
            api_key=self.settings.OPENAI_API_KEY,
            **_extra,
        )
        LlamaSettings.embed_model = OpenAIEmbedding(
            model=self.settings.OPENAI_EMBEDDING_MODEL,
            api_key=self.settings.OPENAI_API_KEY,
            **_extra,
        )
        LlamaSettings.node_parser = SentenceSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )

        # 初始化 Chroma 向量数据库
        persist_dir = self.settings.CHROMA_PERSIST_DIR
        os.makedirs(persist_dir, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection("knowledge_base")

        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 加载或创建索引
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
        logger.info(f"RAG 引擎初始化完成，向量库路径: {persist_dir}")

    async def add_document(
        self,
        title: str,
        content: str,
        tags: list[str] = None,
        doc_id: str = "",
    ) -> int:
        """添加文档到知识库，返回分块数量"""
        if not self.index:
            raise RuntimeError("RAG 引擎未初始化")

        metadata = {
            "title": title,
            "tags": ",".join(tags or []),
            "doc_id": doc_id,
        }

        doc = LlamaDocument(text=content, metadata=metadata)
        
        # 分块并插入索引
        parser = SentenceSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )
        nodes = parser.get_nodes_from_documents([doc])
        
        self.index.insert_nodes(nodes)
        logger.info(f"文档 '{title}' 已添加，分为 {len(nodes)} 个块")
        return len(nodes)

    async def query(
        self,
        question: str,
        top_k: int = None,
        tags: list[str] = None,
    ) -> dict:
        """查询知识库"""
        if not self.index:
            raise RuntimeError("RAG 引擎未初始化")

        k = top_k or self.settings.SIMILARITY_TOP_K

        # 构建检索器
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=k)

        # 后处理：过滤低相关度结果
        postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.5),
        ]

        # 响应合成器
        synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=QA_PROMPT,
        )

        # 组装查询引擎
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=postprocessors,
        )

        response = query_engine.query(question)

        # 提取来源信息
        sources = []
        for node in response.source_nodes:
            sources.append({
                "title": node.metadata.get("title", "未知"),
                "score": round(node.score, 4) if node.score else 0,
                "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                "tags": node.metadata.get("tags", ""),
            })

        return {
            "answer": str(response),
            "sources": sources,
            "confidence": sources[0]["score"] if sources else 0,
        }

    async def delete_document(self, doc_id: str):
        """从知识库删除文档（按 doc_id 元数据过滤）"""
        if self.collection:
            self.collection.delete(where={"doc_id": doc_id})
            logger.info(f"文档 {doc_id} 已从向量库删除")

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        count = self.collection.count() if self.collection else 0
        return {
            "total_chunks": count,
            "persist_dir": self.settings.CHROMA_PERSIST_DIR,
            "model": self.settings.OPENAI_EMBEDDING_MODEL,
        }
