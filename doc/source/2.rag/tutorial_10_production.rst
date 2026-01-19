####################################
Tutorial 10: RAG 生产部署
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

生产环境考量
============

将 RAG 系统部署到生产环境需要考虑：

- **性能**: 响应时间、吞吐量
- **可靠性**: 高可用、容错
- **可扩展性**: 水平扩展、负载均衡
- **安全性**: 数据保护、访问控制
- **成本**: API 调用、存储、计算

架构设计
========

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                    RAG 生产架构                                  │
   │                                                                  │
   │  ┌──────────────────────────────────────────────────────────┐   │
   │  │                      负载均衡                              │   │
   │  └──────────────────────────┬───────────────────────────────┘   │
   │                             │                                    │
   │         ┌───────────────────┼───────────────────┐               │
   │         │                   │                   │               │
   │         ▼                   ▼                   ▼               │
   │  ┌────────────┐     ┌────────────┐     ┌────────────┐          │
   │  │  API Server │     │  API Server │     │  API Server │          │
   │  └──────┬─────┘     └──────┬─────┘     └──────┬─────┘          │
   │         │                  │                  │                 │
   │         └──────────────────┼──────────────────┘                 │
   │                            │                                    │
   │  ┌─────────────────────────┼─────────────────────────┐         │
   │  │                         ▼                         │         │
   │  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    │         │
   │  │  │ 向量数据库 │    │  Redis   │    │   LLM    │    │         │
   │  │  │ (Milvus) │    │ (缓存)   │    │  (API)   │    │         │
   │  │  └──────────┘    └──────────┘    └──────────┘    │         │
   │  │                   服务层                          │         │
   │  └───────────────────────────────────────────────────┘         │
   │                                                                  │
   └─────────────────────────────────────────────────────────────────┘

FastAPI 服务
============

.. code-block:: python

   # app/main.py
   from fastapi import FastAPI, HTTPException, BackgroundTasks
   from pydantic import BaseModel
   from typing import List, Optional
   import time

   app = FastAPI(title="RAG API", version="1.0.0")

   # 请求/响应模型
   class QueryRequest(BaseModel):
       question: str
       top_k: int = 3
       use_cache: bool = True

   class QueryResponse(BaseModel):
       answer: str
       sources: List[dict]
       latency_ms: float

   class IndexRequest(BaseModel):
       documents: List[str]
       metadatas: Optional[List[dict]] = None

   # RAG 服务
   from app.rag_service import RAGService
   rag_service = RAGService()

   @app.post("/query", response_model=QueryResponse)
   async def query(request: QueryRequest):
       """查询接口"""
       start_time = time.time()
       
       try:
           result = await rag_service.query(
               question=request.question,
               top_k=request.top_k,
               use_cache=request.use_cache
           )
           
           latency_ms = (time.time() - start_time) * 1000
           
           return QueryResponse(
               answer=result["answer"],
               sources=result["sources"],
               latency_ms=latency_ms
           )
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))

   @app.post("/index")
   async def index_documents(
       request: IndexRequest,
       background_tasks: BackgroundTasks
   ):
       """索引文档（后台任务）"""
       background_tasks.add_task(
           rag_service.index_documents,
           request.documents,
           request.metadatas
       )
       return {"message": "Indexing started", "count": len(request.documents)}

   @app.get("/health")
   async def health_check():
       """健康检查"""
       return {"status": "healthy"}

RAG 服务实现
============

.. code-block:: python

   # app/rag_service.py
   from langchain_community.vectorstores import Chroma
   from langchain_community.embeddings import HuggingFaceEmbeddings
   from langchain_openai import ChatOpenAI
   from typing import List, Dict, Optional
   import redis
   import hashlib
   import json

   class RAGService:
       """RAG 服务"""
       
       def __init__(self):
           # 初始化组件
           self.embeddings = HuggingFaceEmbeddings(
               model_name="sentence-transformers/all-MiniLM-L6-v2"
           )
           self.vectorstore = Chroma(
               persist_directory="./chroma_db",
               embedding_function=self.embeddings
           )
           self.llm = ChatOpenAI(
               model="gpt-3.5-turbo",
               temperature=0,
               request_timeout=30
           )
           
           # 缓存
           self.cache = redis.Redis(host='localhost', port=6379, db=0)
           self.cache_ttl = 3600  # 1小时
       
       async def query(
           self,
           question: str,
           top_k: int = 3,
           use_cache: bool = True
       ) -> Dict:
           """执行查询"""
           
           # 检查缓存
           if use_cache:
               cached = self._get_from_cache(question)
               if cached:
                   return cached
           
           # 检索
           docs = self.vectorstore.similarity_search(question, k=top_k)
           
           # 构建上下文
           context = "\n\n".join([doc.page_content for doc in docs])
           
           # 生成回答
           prompt = f"""基于以下信息回答问题。

   信息：
   {context}

   问题：{question}

   回答："""
           
           answer = self.llm.invoke(prompt).content
           
           result = {
               "answer": answer,
               "sources": [doc.metadata for doc in docs]
           }
           
           # 存入缓存
           if use_cache:
               self._save_to_cache(question, result)
           
           return result
       
       def index_documents(
           self,
           documents: List[str],
           metadatas: Optional[List[dict]] = None
       ):
           """索引文档"""
           self.vectorstore.add_texts(
               texts=documents,
               metadatas=metadatas
           )
       
       def _get_cache_key(self, question: str) -> str:
           return f"rag:{hashlib.md5(question.encode()).hexdigest()}"
       
       def _get_from_cache(self, question: str) -> Optional[Dict]:
           key = self._get_cache_key(question)
           cached = self.cache.get(key)
           if cached:
               return json.loads(cached)
           return None
       
       def _save_to_cache(self, question: str, result: Dict):
           key = self._get_cache_key(question)
           self.cache.setex(key, self.cache_ttl, json.dumps(result))

Docker 部署
===========

.. code-block:: dockerfile

   # Dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # 安装依赖
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # 复制代码
   COPY app/ app/

   # 环境变量
   ENV PYTHONPATH=/app
   ENV PYTHONUNBUFFERED=1

   EXPOSE 8000

   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

.. code-block:: yaml

   # docker-compose.yml
   version: '3.8'

   services:
     api:
       build: .
       ports:
         - "8000:8000"
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - REDIS_URL=redis://redis:6379
       depends_on:
         - redis
         - milvus
       volumes:
         - ./chroma_db:/app/chroma_db

     redis:
       image: redis:7-alpine
       ports:
         - "6379:6379"

     milvus:
       image: milvusdb/milvus:latest
       ports:
         - "19530:19530"
       volumes:
         - ./milvus_data:/var/lib/milvus

性能优化
========

1. 批量处理
-----------

.. code-block:: python

   from typing import List
   import asyncio

   class BatchProcessor:
       """批量处理器"""
       
       def __init__(self, rag_service, batch_size=10, max_wait_ms=100):
           self.rag_service = rag_service
           self.batch_size = batch_size
           self.max_wait_ms = max_wait_ms
           self.pending = []
           self.lock = asyncio.Lock()
       
       async def query(self, question: str) -> dict:
           """添加到批量队列"""
           future = asyncio.Future()
           
           async with self.lock:
               self.pending.append((question, future))
               
               if len(self.pending) >= self.batch_size:
                   await self._process_batch()
           
           # 等待结果
           return await future
       
       async def _process_batch(self):
           """处理批量请求"""
           batch = self.pending[:self.batch_size]
           self.pending = self.pending[self.batch_size:]
           
           questions = [q for q, _ in batch]
           
           # 批量嵌入
           embeddings = self.rag_service.embeddings.embed_documents(questions)
           
           # 批量检索和生成
           for i, (question, future) in enumerate(batch):
               result = await self.rag_service.query(question)
               future.set_result(result)

2. 异步处理
-----------

.. code-block:: python

   import asyncio
   from concurrent.futures import ThreadPoolExecutor

   class AsyncRAGService:
       """异步 RAG 服务"""
       
       def __init__(self):
           self.executor = ThreadPoolExecutor(max_workers=10)
           # ... 初始化其他组件
       
       async def query(self, question: str) -> dict:
           """异步查询"""
           loop = asyncio.get_event_loop()
           
           # 并行执行检索和其他操作
           retrieval_task = loop.run_in_executor(
               self.executor,
               self._retrieve,
               question
           )
           
           docs = await retrieval_task
           
           # 生成回答
           answer = await self._generate_async(question, docs)
           
           return {"answer": answer, "sources": docs}

监控和日志
==========

.. code-block:: python

   import logging
   import time
   from prometheus_client import Counter, Histogram, generate_latest
   from functools import wraps

   # 配置日志
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   logger = logging.getLogger("rag")

   # Prometheus 指标
   REQUEST_COUNT = Counter(
       'rag_requests_total',
       'Total RAG requests',
       ['status']
   )

   REQUEST_LATENCY = Histogram(
       'rag_request_latency_seconds',
       'RAG request latency',
       buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
   )

   def monitor(func):
       """监控装饰器"""
       @wraps(func)
       async def wrapper(*args, **kwargs):
           start_time = time.time()
           
           try:
               result = await func(*args, **kwargs)
               REQUEST_COUNT.labels(status='success').inc()
               return result
           except Exception as e:
               REQUEST_COUNT.labels(status='error').inc()
               logger.error(f"Error in {func.__name__}: {e}")
               raise
           finally:
               latency = time.time() - start_time
               REQUEST_LATENCY.observe(latency)
               logger.info(f"{func.__name__} completed in {latency:.3f}s")
       
       return wrapper

   # 使用
   @monitor
   async def query(question: str):
       # ... 查询逻辑
       pass

   # 指标端点
   @app.get("/metrics")
   async def metrics():
       from fastapi.responses import Response
       return Response(
           generate_latest(),
           media_type="text/plain"
       )

安全最佳实践
============

.. code-block:: python

   from fastapi import Depends, HTTPException, Security
   from fastapi.security import APIKeyHeader
   from pydantic import BaseModel, validator
   import re

   # API Key 认证
   api_key_header = APIKeyHeader(name="X-API-Key")

   async def verify_api_key(api_key: str = Security(api_key_header)):
       if api_key != settings.api_key:
           raise HTTPException(status_code=403, detail="Invalid API key")
       return api_key

   # 输入验证
   class QueryRequest(BaseModel):
       question: str
       
       @validator('question')
       def validate_question(cls, v):
           if len(v) > 1000:
               raise ValueError('Question too long')
           if len(v) < 3:
               raise ValueError('Question too short')
           # 防止注入
           if re.search(r'[<>{}]', v):
               raise ValueError('Invalid characters')
           return v

   # 限流
   from slowapi import Limiter
   from slowapi.util import get_remote_address

   limiter = Limiter(key_func=get_remote_address)

   @app.post("/query")
   @limiter.limit("10/minute")
   async def query(request: QueryRequest):
       # ...

关键概念总结
============

.. csv-table::
   :header: "方面", "关键点"
   :widths: 25, 75

   "性能", "缓存、批处理、异步"
   "可靠性", "重试、降级、健康检查"
   "可扩展", "水平扩展、负载均衡"
   "安全", "认证、验证、限流"
   "监控", "日志、指标、告警"

总结
====

恭喜你完成了 RAG 全部教程！

你已经学习了：

1. ✅ RAG 基本概念和原理
2. ✅ 文档加载和处理
3. ✅ 文本分块策略
4. ✅ 向量嵌入技术
5. ✅ 向量数据库使用
6. ✅ 检索策略优化
7. ✅ Prompt 工程
8. ✅ 系统评估方法
9. ✅ 高级 RAG 技术
10. ✅ 生产环境部署

🎉 祝你在 RAG 应用开发中取得成功！
