####################################
Tutorial 9: 构建知识库系统
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

.. contents:: 目录
   :local:
   :depth: 2

知识库系统概述
==============

本教程将整合前面学到的所有技术，构建一个完整的企业级知识库系统。

.. code-block:: text

   知识库系统架构：

   ┌─────────────────────────────────────────────────────────────────┐
   │                    Enterprise Knowledge Base                     │
   ├─────────────────────────────────────────────────────────────────┤
   │                                                                  │
   │   数据层                     索引层                查询层        │
   │   ┌───────────┐             ┌───────────┐        ┌───────────┐  │
   │   │ 文档加载  │────────────►│ 向量索引  │───────►│ 检索引擎  │  │
   │   │ PDF/Doc   │             │ Chroma    │        │           │  │
   │   └───────────┘             └───────────┘        │           │  │
   │   ┌───────────┐             ┌───────────┐        │           │  │
   │   │ 数据库    │────────────►│ 关键词    │───────►│ 查询引擎  │  │
   │   │ MySQL     │             │ 索引      │        │           │  │
   │   └───────────┘             └───────────┘        └───────────┘  │
   │   ┌───────────┐                                        │        │
   │   │ API/Web   │                                        ▼        │
   │   └───────────┘                                  ┌───────────┐  │
   │                                                  │   Agent   │  │
   │                                                  │  智能问答  │  │
   │                                                  └───────────┘  │
   │                                                                  │
   └─────────────────────────────────────────────────────────────────┘

系统设计
========

核心组件
--------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 组件
     - 职责
     - 技术选型
   * - 文档管理
     - 加载、解析、存储文档
     - SimpleDirectoryReader, LlamaParse
   * - 索引管理
     - 创建和维护索引
     - VectorStoreIndex, Chroma
   * - 检索引擎
     - 混合检索、重排序
     - HybridRetriever, Reranker
   * - 查询引擎
     - 生成回答
     - QueryEngine, Agent
   * - API 服务
     - 对外提供接口
     - FastAPI

完整实现
========

项目结构
--------

.. code-block:: text

   knowledge_base/
   ├── __init__.py
   ├── config.py           # 配置管理
   ├── document_manager.py # 文档管理
   ├── index_manager.py    # 索引管理
   ├── retriever.py        # 检索器
   ├── query_engine.py     # 查询引擎
   ├── agent.py            # 智能代理
   ├── api.py              # API 服务
   └── main.py             # 主程序

配置管理
--------

.. code-block:: python

   # config.py
   from pydantic_settings import BaseSettings
   from typing import Optional

   class Settings(BaseSettings):
       """系统配置"""
       # OpenAI
       openai_api_key: str
       openai_model: str = "gpt-4o-mini"
       embedding_model: str = "text-embedding-3-small"

       # 存储
       chroma_persist_dir: str = "./chroma_db"
       document_dir: str = "./documents"

       # 检索
       similarity_top_k: int = 5
       rerank_top_n: int = 3

       # 服务
       api_host: str = "0.0.0.0"
       api_port: int = 8000

       class Config:
           env_file = ".env"

   settings = Settings()

文档管理器
----------

.. code-block:: python

   # document_manager.py
   from llama_index.core import SimpleDirectoryReader, Document
   from llama_index.core.node_parser import SentenceSplitter
   from typing import List, Optional
   import os
   import hashlib

   class DocumentManager:
       """文档管理器"""

       def __init__(self, document_dir: str):
           self.document_dir = document_dir
           self.node_parser = SentenceSplitter(
               chunk_size=512,
               chunk_overlap=50
           )
           self._document_hashes = {}

       def load_documents(self, subdirectory: Optional[str] = None) -> List[Document]:
           """加载文档"""
           target_dir = self.document_dir
           if subdirectory:
               target_dir = os.path.join(self.document_dir, subdirectory)

           if not os.path.exists(target_dir):
               raise ValueError(f"目录不存在: {target_dir}")

           reader = SimpleDirectoryReader(
               input_dir=target_dir,
               recursive=True,
               required_exts=[".pdf", ".md", ".txt", ".docx"]
           )

           documents = reader.load_data()

           # 添加元数据
           for doc in documents:
               doc.metadata["source_dir"] = subdirectory or "root"
               doc.metadata["doc_hash"] = self._compute_hash(doc.text)

           return documents

       def _compute_hash(self, text: str) -> str:
           """计算文档哈希"""
           return hashlib.md5(text.encode()).hexdigest()

       def get_nodes(self, documents: List[Document]) -> List:
           """将文档解析为节点"""
           return self.node_parser.get_nodes_from_documents(documents)

       def add_document(self, text: str, metadata: dict = None) -> Document:
           """添加单个文档"""
           doc = Document(
               text=text,
               metadata=metadata or {}
           )
           doc.metadata["doc_hash"] = self._compute_hash(text)
           return doc

索引管理器
----------

.. code-block:: python

   # index_manager.py
   import chromadb
   from llama_index.core import VectorStoreIndex, StorageContext, Settings
   from llama_index.vector_stores.chroma import ChromaVectorStore
   from llama_index.embeddings.openai import OpenAIEmbedding
   from typing import List, Optional

   class IndexManager:
       """索引管理器"""

       def __init__(self, persist_dir: str, embedding_model: str):
           self.persist_dir = persist_dir
           Settings.embed_model = OpenAIEmbedding(model=embedding_model)

           # 初始化 Chroma
           self.chroma_client = chromadb.PersistentClient(path=persist_dir)
           self.indexes = {}

       def create_index(
           self,
           collection_name: str,
           nodes: List,
           overwrite: bool = False
       ) -> VectorStoreIndex:
           """创建索引"""
           # 获取或创建集合
           if overwrite:
               try:
                   self.chroma_client.delete_collection(collection_name)
               except:
                   pass

           collection = self.chroma_client.get_or_create_collection(
               name=collection_name
           )

           # 创建向量存储
           vector_store = ChromaVectorStore(chroma_collection=collection)
           storage_context = StorageContext.from_defaults(
               vector_store=vector_store
           )

           # 创建索引
           index = VectorStoreIndex(
               nodes,
               storage_context=storage_context,
               show_progress=True
           )

           self.indexes[collection_name] = index
           return index

       def load_index(self, collection_name: str) -> Optional[VectorStoreIndex]:
           """加载已有索引"""
           try:
               collection = self.chroma_client.get_collection(collection_name)
               vector_store = ChromaVectorStore(chroma_collection=collection)
               index = VectorStoreIndex.from_vector_store(vector_store)
               self.indexes[collection_name] = index
               return index
           except Exception as e:
               print(f"加载索引失败: {e}")
               return None

       def get_index(self, collection_name: str) -> Optional[VectorStoreIndex]:
           """获取索引"""
           if collection_name in self.indexes:
               return self.indexes[collection_name]
           return self.load_index(collection_name)

       def list_collections(self) -> List[str]:
           """列出所有集合"""
           collections = self.chroma_client.list_collections()
           return [c.name for c in collections]

       def delete_collection(self, collection_name: str):
           """删除集合"""
           self.chroma_client.delete_collection(collection_name)
           if collection_name in self.indexes:
               del self.indexes[collection_name]

混合检索器
----------

.. code-block:: python

   # retriever.py
   from llama_index.core.retrievers import BaseRetriever
   from llama_index.core.schema import NodeWithScore, QueryBundle
   from llama_index.retrievers.bm25 import BM25Retriever
   from typing import List, Optional

   class HybridRetriever(BaseRetriever):
       """混合检索器"""

       def __init__(
           self,
           vector_retriever,
           bm25_retriever: Optional[BM25Retriever] = None,
           vector_weight: float = 0.6,
           top_k: int = 10
       ):
           self.vector_retriever = vector_retriever
           self.bm25_retriever = bm25_retriever
           self.vector_weight = vector_weight
           self.bm25_weight = 1 - vector_weight
           self.top_k = top_k

       def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
           # 向量检索
           vector_nodes = self.vector_retriever.retrieve(query_bundle)

           if not self.bm25_retriever:
               return vector_nodes[:self.top_k]

           # BM25 检索
           bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

           # 合并结果
           return self._merge_results(vector_nodes, bm25_nodes)

       def _merge_results(
           self,
           vector_nodes: List[NodeWithScore],
           bm25_nodes: List[NodeWithScore]
       ) -> List[NodeWithScore]:
           """合并并重新评分"""
           all_nodes = {}

           # 归一化向量分数
           v_scores = self._normalize([n.score for n in vector_nodes])
           for node, score in zip(vector_nodes, v_scores):
               all_nodes[node.node.id_] = {
                   "node": node,
                   "v_score": score,
                   "b_score": 0
               }

           # 归一化 BM25 分数
           b_scores = self._normalize([n.score for n in bm25_nodes])
           for node, score in zip(bm25_nodes, b_scores):
               if node.node.id_ in all_nodes:
                   all_nodes[node.node.id_]["b_score"] = score
               else:
                   all_nodes[node.node.id_] = {
                       "node": node,
                       "v_score": 0,
                       "b_score": score
                   }

           # 计算最终分数
           results = []
           for data in all_nodes.values():
               final_score = (
                   data["v_score"] * self.vector_weight +
                   data["b_score"] * self.bm25_weight
               )
               node = data["node"]
               node.score = final_score
               results.append(node)

           results.sort(key=lambda x: x.score, reverse=True)
           return results[:self.top_k]

       def _normalize(self, scores: List[float]) -> List[float]:
           """归一化分数"""
           if not scores:
               return []
           min_s, max_s = min(scores), max(scores)
           if max_s == min_s:
               return [1.0] * len(scores)
           return [(s - min_s) / (max_s - min_s) for s in scores]

查询引擎
--------

.. code-block:: python

   # query_engine.py
   from llama_index.core import Settings
   from llama_index.core.query_engine import RetrieverQueryEngine
   from llama_index.core.response_synthesizers import get_response_synthesizer
   from llama_index.core.prompts import PromptTemplate
   from llama_index.llms.openai import OpenAI
   from typing import Optional

   class KnowledgeQueryEngine:
       """知识库查询引擎"""

       def __init__(self, retriever, llm_model: str = "gpt-4o-mini"):
           self.retriever = retriever
           Settings.llm = OpenAI(model=llm_model)

           # 自定义提示模板
           self.qa_template = PromptTemplate(
               """你是一个知识库助手。请基于以下参考内容回答用户的问题。

参考内容：
{context_str}

用户问题：{query_str}

回答要求：
1. 如果参考内容中有答案，请准确回答并标注来源
2. 如果参考内容不足以回答，请明确说明
3. 不要编造信息
4. 回答要简洁明了

回答："""
           )

           # 创建响应合成器
           self.synthesizer = get_response_synthesizer(
               response_mode="compact",
               text_qa_template=self.qa_template
           )

           # 创建查询引擎
           self.query_engine = RetrieverQueryEngine(
               retriever=self.retriever,
               response_synthesizer=self.synthesizer
           )

       def query(self, question: str) -> dict:
           """执行查询"""
           response = self.query_engine.query(question)

           # 提取来源信息
           sources = []
           for node in response.source_nodes:
               sources.append({
                   "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                   "score": float(node.score) if node.score else 0,
                   "metadata": node.metadata
               })

           return {
               "answer": str(response),
               "sources": sources
           }

       def query_stream(self, question: str):
           """流式查询"""
           streaming_response = self.query_engine.query(question)
           for text in streaming_response.response_gen:
               yield text

智能 Agent
----------

.. code-block:: python

   # agent.py
   from llama_index.core.agent import ReActAgent
   from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata
   from llama_index.llms.openai import OpenAI
   from datetime import datetime
   from typing import Dict, List

   class KnowledgeAgent:
       """知识库智能代理"""

       def __init__(self, query_engines: Dict[str, any], llm_model: str = "gpt-4o-mini"):
           self.llm = OpenAI(model=llm_model)
           self.tools = []
           self._setup_tools(query_engines)
           self._build_agent()

       def _setup_tools(self, query_engines: Dict):
           """设置工具"""
           # 添加知识库查询工具
           for name, engine in query_engines.items():
               tool = QueryEngineTool(
                   query_engine=engine.query_engine,
                   metadata=ToolMetadata(
                       name=f"query_{name}",
                       description=f"查询{name}知识库的内容"
                   )
               )
               self.tools.append(tool)

           # 添加辅助工具
           def get_current_time() -> str:
               """获取当前时间"""
               return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

           self.tools.append(FunctionTool.from_defaults(
               fn=get_current_time,
               name="get_time",
               description="获取当前日期和时间"
           ))

       def _build_agent(self):
           """构建 Agent"""
           self.agent = ReActAgent.from_tools(
               self.tools,
               llm=self.llm,
               verbose=True,
               system_prompt="""你是一个智能知识库助手。
               你可以查询多个知识库来回答用户的问题。
               请根据问题选择合适的知识库进行查询。
               回答时要准确、简洁，并说明信息来源。"""
           )

       def chat(self, message: str) -> str:
           """对话"""
           response = self.agent.chat(message)
           return str(response)

       def reset(self):
           """重置对话"""
           self.agent.reset()

API 服务
--------

.. code-block:: python

   # api.py
   from fastapi import FastAPI, HTTPException
   from fastapi.middleware.cors import CORSMiddleware
   from pydantic import BaseModel
   from typing import Optional, List

   app = FastAPI(title="Knowledge Base API")

   # CORS
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_methods=["*"],
       allow_headers=["*"],
   )

   # 全局变量（实际应用中应使用依赖注入）
   knowledge_system = None

   class QueryRequest(BaseModel):
       question: str
       collection: Optional[str] = "default"

   class QueryResponse(BaseModel):
       answer: str
       sources: List[dict]

   class ChatRequest(BaseModel):
       message: str

   class DocumentRequest(BaseModel):
       text: str
       metadata: Optional[dict] = None
       collection: str = "default"

   @app.post("/query", response_model=QueryResponse)
   async def query(request: QueryRequest):
       """查询知识库"""
       if not knowledge_system:
           raise HTTPException(status_code=500, detail="系统未初始化")

       try:
           result = knowledge_system.query(
               request.question,
               request.collection
           )
           return QueryResponse(**result)
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))

   @app.post("/chat")
   async def chat(request: ChatRequest):
       """智能对话"""
       if not knowledge_system:
           raise HTTPException(status_code=500, detail="系统未初始化")

       try:
           response = knowledge_system.chat(request.message)
           return {"response": response}
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))

   @app.post("/documents")
   async def add_document(request: DocumentRequest):
       """添加文档"""
       if not knowledge_system:
           raise HTTPException(status_code=500, detail="系统未初始化")

       try:
           knowledge_system.add_document(
               request.text,
               request.metadata,
               request.collection
           )
           return {"message": "文档添加成功"}
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))

   @app.get("/collections")
   async def list_collections():
       """列出所有集合"""
       if not knowledge_system:
           raise HTTPException(status_code=500, detail="系统未初始化")

       collections = knowledge_system.list_collections()
       return {"collections": collections}

   @app.get("/health")
   async def health_check():
       """健康检查"""
       return {"status": "healthy"}

主程序
------

.. code-block:: python

   # main.py
   from config import settings
   from document_manager import DocumentManager
   from index_manager import IndexManager
   from retriever import HybridRetriever
   from query_engine import KnowledgeQueryEngine
   from agent import KnowledgeAgent
   from llama_index.retrievers.bm25 import BM25Retriever
   import os

   class KnowledgeBaseSystem:
       """知识库系统"""

       def __init__(self):
           self.doc_manager = DocumentManager(settings.document_dir)
           self.index_manager = IndexManager(
               settings.chroma_persist_dir,
               settings.embedding_model
           )
           self.query_engines = {}
           self.agent = None

       def initialize(self, collections: list = None):
           """初始化系统"""
           if collections is None:
               collections = ["default"]

           for collection in collections:
               self._initialize_collection(collection)

           # 初始化 Agent
           if self.query_engines:
               self.agent = KnowledgeAgent(
                   self.query_engines,
                   settings.openai_model
               )

       def _initialize_collection(self, collection: str):
           """初始化单个集合"""
           # 尝试加载已有索引
           index = self.index_manager.load_index(collection)

           if index is None:
               # 创建新索引
               doc_path = os.path.join(settings.document_dir, collection)
               if os.path.exists(doc_path):
                   documents = self.doc_manager.load_documents(collection)
                   nodes = self.doc_manager.get_nodes(documents)
                   index = self.index_manager.create_index(collection, nodes)

           if index:
               # 创建检索器
               vector_retriever = index.as_retriever(
                   similarity_top_k=settings.similarity_top_k
               )

               retriever = HybridRetriever(
                   vector_retriever=vector_retriever,
                   top_k=settings.similarity_top_k
               )

               # 创建查询引擎
               self.query_engines[collection] = KnowledgeQueryEngine(
                   retriever,
                   settings.openai_model
               )

       def query(self, question: str, collection: str = "default") -> dict:
           """查询"""
           if collection not in self.query_engines:
               raise ValueError(f"集合不存在: {collection}")

           return self.query_engines[collection].query(question)

       def chat(self, message: str) -> str:
           """智能对话"""
           if not self.agent:
               raise ValueError("Agent 未初始化")
           return self.agent.chat(message)

       def add_document(self, text: str, metadata: dict, collection: str):
           """添加文档"""
           doc = self.doc_manager.add_document(text, metadata)
           nodes = self.doc_manager.get_nodes([doc])

           index = self.index_manager.get_index(collection)
           if index:
               for node in nodes:
                   index.insert_nodes([node])

       def list_collections(self) -> list:
           """列出集合"""
           return self.index_manager.list_collections()

   def main():
       # 初始化系统
       system = KnowledgeBaseSystem()
       system.initialize(["default", "products", "faq"])

       # 启动 API 服务
       import api
       api.knowledge_system = system

       import uvicorn
       uvicorn.run(api.app, host=settings.api_host, port=settings.api_port)

   if __name__ == "__main__":
       main()

使用示例
========

.. code-block:: python

   # 使用知识库系统

   # 1. 初始化
   system = KnowledgeBaseSystem()
   system.initialize()

   # 2. 查询
   result = system.query("什么是机器学习？")
   print(f"回答: {result['answer']}")
   print(f"来源: {result['sources']}")

   # 3. 智能对话
   response = system.chat("告诉我关于深度学习的知识")
   print(response)

   # 4. 添加新文档
   system.add_document(
       text="这是新添加的知识内容...",
       metadata={"source": "manual", "topic": "ai"},
       collection="default"
   )

小结
====

本教程展示了：

- 完整的知识库系统架构设计
- 模块化的组件实现
- 混合检索策略的应用
- 智能 Agent 的集成
- RESTful API 服务的构建
- 生产级代码组织方式

下一步
------

在下一个教程中，我们将学习如何将知识库系统部署到生产环境，
包括性能优化、监控、安全等方面。

练习
====

1. 为系统添加用户认证功能
2. 实现文档的更新和删除
3. 添加查询缓存机制
4. 实现多租户支持
