################################
Tutorial 10: 生产部署
################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

.. contents:: 目录
   :local:
   :depth: 2

生产部署概述
============

将 LlamaIndex 应用部署到生产环境需要考虑性能、可靠性、安全性等多个方面。

.. code-block:: text

   生产部署架构：

   ┌─────────────────────────────────────────────────────────────────┐
   │                       Production Stack                          │
   ├─────────────────────────────────────────────────────────────────┤
   │                                                                  │
   │   负载均衡            应用服务                 存储层           │
   │   ┌────────┐         ┌────────┐              ┌────────┐        │
   │   │ Nginx  │────────►│ App 1  │─────────────►│ Chroma │        │
   │   │        │         └────────┘              │ /Milvus│        │
   │   │        │         ┌────────┐              └────────┘        │
   │   │        │────────►│ App 2  │              ┌────────┐        │
   │   │        │         └────────┘─────────────►│ Redis  │        │
   │   │        │         ┌────────┐              │ Cache  │        │
   │   │        │────────►│ App N  │              └────────┘        │
   │   └────────┘         └────────┘              ┌────────┐        │
   │                           │                  │ PostgreSQL      │
   │                           ▼                  └────────┘        │
   │                      ┌────────┐                                │
   │                      │监控系统│                                │
   │                      │Prometheus                               │
   │                      │Grafana │                                │
   │                      └────────┘                                │
   │                                                                  │
   └─────────────────────────────────────────────────────────────────┘

Docker 部署
===========

Dockerfile
----------

.. code-block:: dockerfile

   # Dockerfile
   FROM python:3.11-slim

   # 设置工作目录
   WORKDIR /app

   # 安装系统依赖
   RUN apt-get update && apt-get install -y \
       build-essential \
       curl \
       && rm -rf /var/lib/apt/lists/*

   # 复制依赖文件
   COPY requirements.txt .

   # 安装 Python 依赖
   RUN pip install --no-cache-dir -r requirements.txt

   # 复制应用代码
   COPY . .

   # 暴露端口
   EXPOSE 8000

   # 健康检查
   HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
       CMD curl -f http://localhost:8000/health || exit 1

   # 启动命令
   CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

requirements.txt
----------------

.. code-block:: text

   llama-index>=0.10.0
   llama-index-llms-openai>=0.1.0
   llama-index-embeddings-openai>=0.1.0
   llama-index-vector-stores-chroma>=0.1.0
   chromadb>=0.4.0
   fastapi>=0.109.0
   uvicorn>=0.27.0
   pydantic-settings>=2.0.0
   redis>=5.0.0
   prometheus-client>=0.19.0
   python-dotenv>=1.0.0

Docker Compose
--------------

.. code-block:: yaml

   # docker-compose.yml
   version: '3.8'

   services:
     app:
       build: .
       ports:
         - "8000:8000"
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - REDIS_URL=redis://redis:6379
         - CHROMA_HOST=chroma
         - CHROMA_PORT=8000
       depends_on:
         - redis
         - chroma
       volumes:
         - ./documents:/app/documents
       deploy:
         replicas: 2
         resources:
           limits:
             cpus: '2'
             memory: 4G

     redis:
       image: redis:7-alpine
       ports:
         - "6379:6379"
       volumes:
         - redis_data:/data

     chroma:
       image: chromadb/chroma:latest
       ports:
         - "8001:8000"
       volumes:
         - chroma_data:/chroma/chroma

     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf:ro
       depends_on:
         - app

     prometheus:
       image: prom/prometheus
       ports:
         - "9090:9090"
       volumes:
         - ./prometheus.yml:/etc/prometheus/prometheus.yml

     grafana:
       image: grafana/grafana
       ports:
         - "3000:3000"
       environment:
         - GF_SECURITY_ADMIN_PASSWORD=admin

   volumes:
     redis_data:
     chroma_data:

性能优化
========

缓存策略
--------

.. code-block:: python

   # cache.py
   import redis
   import json
   import hashlib
   from typing import Optional, Any
   from functools import wraps

   class CacheManager:
       """缓存管理器"""

       def __init__(self, redis_url: str = "redis://localhost:6379"):
           self.redis_client = redis.from_url(redis_url)
           self.default_ttl = 3600  # 1小时

       def _make_key(self, prefix: str, *args, **kwargs) -> str:
           """生成缓存键"""
           data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
           hash_key = hashlib.md5(data.encode()).hexdigest()
           return f"{prefix}:{hash_key}"

       def get(self, key: str) -> Optional[Any]:
           """获取缓存"""
           data = self.redis_client.get(key)
           if data:
               return json.loads(data)
           return None

       def set(self, key: str, value: Any, ttl: int = None):
           """设置缓存"""
           self.redis_client.setex(
               key,
               ttl or self.default_ttl,
               json.dumps(value)
           )

       def delete(self, key: str):
           """删除缓存"""
           self.redis_client.delete(key)

       def clear_prefix(self, prefix: str):
           """清除指定前缀的所有缓存"""
           for key in self.redis_client.scan_iter(f"{prefix}:*"):
               self.redis_client.delete(key)

   # 缓存装饰器
   def cached(prefix: str, ttl: int = 3600):
       def decorator(func):
           @wraps(func)
           def wrapper(self, *args, **kwargs):
               cache = getattr(self, 'cache', None)
               if not cache:
                   return func(self, *args, **kwargs)

               key = cache._make_key(prefix, *args, **kwargs)
               result = cache.get(key)

               if result is not None:
                   return result

               result = func(self, *args, **kwargs)
               cache.set(key, result, ttl)
               return result
           return wrapper
       return decorator

   # 使用示例
   class CachedQueryEngine:
       def __init__(self, query_engine, cache_manager):
           self.query_engine = query_engine
           self.cache = cache_manager

       @cached("query", ttl=1800)
       def query(self, question: str) -> dict:
           response = self.query_engine.query(question)
           return {
               "answer": str(response),
               "sources": [
                   {"text": n.text, "score": n.score}
                   for n in response.source_nodes
               ]
           }

嵌入缓存
--------

.. code-block:: python

   # embedding_cache.py
   from llama_index.core.embeddings import BaseEmbedding
   from typing import List
   import hashlib

   class CachedEmbedding(BaseEmbedding):
       """带缓存的嵌入模型"""

       def __init__(self, embed_model: BaseEmbedding, cache_manager):
           self._embed_model = embed_model
           self._cache = cache_manager

       def _get_text_embedding(self, text: str) -> List[float]:
           key = f"embed:{hashlib.md5(text.encode()).hexdigest()}"
           cached = self._cache.get(key)

           if cached:
               return cached

           embedding = self._embed_model.get_text_embedding(text)
           self._cache.set(key, embedding, ttl=86400)  # 24小时
           return embedding

       def _get_query_embedding(self, query: str) -> List[float]:
           return self._get_text_embedding(query)

       async def _aget_text_embedding(self, text: str) -> List[float]:
           return self._get_text_embedding(text)

批量处理
--------

.. code-block:: python

   # batch_processor.py
   from typing import List
   import asyncio

   class BatchProcessor:
       """批量处理器"""

       def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
           self.batch_size = batch_size
           self.max_concurrent = max_concurrent
           self.semaphore = asyncio.Semaphore(max_concurrent)

       async def process_batch(self, items: List, processor_fn):
           """批量异步处理"""
           results = []

           for i in range(0, len(items), self.batch_size):
               batch = items[i:i + self.batch_size]
               batch_results = await asyncio.gather(*[
                   self._process_with_limit(item, processor_fn)
                   for item in batch
               ])
               results.extend(batch_results)

           return results

       async def _process_with_limit(self, item, processor_fn):
           """带并发限制的处理"""
           async with self.semaphore:
               return await processor_fn(item)

   # 使用示例
   async def index_documents(documents: List):
       processor = BatchProcessor(batch_size=5, max_concurrent=3)

       async def process_doc(doc):
           # 处理单个文档
           return await index.ainsert(doc)

       await processor.process_batch(documents, process_doc)

监控与可观测性
==============

Prometheus 指标
---------------

.. code-block:: python

   # metrics.py
   from prometheus_client import Counter, Histogram, Gauge
   import time
   from functools import wraps

   # 定义指标
   QUERY_COUNT = Counter(
       'llama_query_total',
       'Total number of queries',
       ['collection', 'status']
   )

   QUERY_LATENCY = Histogram(
       'llama_query_latency_seconds',
       'Query latency in seconds',
       ['collection'],
       buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
   )

   ACTIVE_QUERIES = Gauge(
       'llama_active_queries',
       'Number of active queries'
   )

   CACHE_HITS = Counter(
       'llama_cache_hits_total',
       'Total cache hits',
       ['cache_type']
   )

   CACHE_MISSES = Counter(
       'llama_cache_misses_total',
       'Total cache misses',
       ['cache_type']
   )

   INDEX_SIZE = Gauge(
       'llama_index_size',
       'Number of documents in index',
       ['collection']
   )

   # 装饰器
   def track_query(collection: str):
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               ACTIVE_QUERIES.inc()
               start_time = time.time()

               try:
                   result = func(*args, **kwargs)
                   QUERY_COUNT.labels(collection=collection, status='success').inc()
                   return result
               except Exception as e:
                   QUERY_COUNT.labels(collection=collection, status='error').inc()
                   raise
               finally:
                   ACTIVE_QUERIES.dec()
                   QUERY_LATENCY.labels(collection=collection).observe(
                       time.time() - start_time
                   )

           return wrapper
       return decorator

   # 在 FastAPI 中暴露指标
   from fastapi import FastAPI
   from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
   from starlette.responses import Response

   app = FastAPI()

   @app.get("/metrics")
   def metrics():
       return Response(
           generate_latest(),
           media_type=CONTENT_TYPE_LATEST
       )

日志记录
--------

.. code-block:: python

   # logging_config.py
   import logging
   import json
   from datetime import datetime

   class JSONFormatter(logging.Formatter):
       """JSON 格式的日志"""

       def format(self, record):
           log_data = {
               "timestamp": datetime.utcnow().isoformat(),
               "level": record.levelname,
               "message": record.getMessage(),
               "module": record.module,
               "function": record.funcName,
               "line": record.lineno
           }

           if hasattr(record, 'extra'):
               log_data.update(record.extra)

           if record.exc_info:
               log_data["exception"] = self.formatException(record.exc_info)

           return json.dumps(log_data)

   def setup_logging():
       """配置日志"""
       logger = logging.getLogger("llama_app")
       logger.setLevel(logging.INFO)

       # 控制台处理器
       console_handler = logging.StreamHandler()
       console_handler.setFormatter(JSONFormatter())
       logger.addHandler(console_handler)

       # 文件处理器
       file_handler = logging.FileHandler("app.log")
       file_handler.setFormatter(JSONFormatter())
       logger.addHandler(file_handler)

       return logger

   # 使用
   logger = setup_logging()

   def log_query(query: str, response: str, latency: float, sources: int):
       logger.info(
           "Query processed",
           extra={
               "extra": {
                   "query": query[:100],
                   "response_length": len(response),
                   "latency_ms": latency * 1000,
                   "source_count": sources
               }
           }
       )

分布式追踪
----------

.. code-block:: python

   # tracing.py
   from opentelemetry import trace
   from opentelemetry.exporter.jaeger.thrift import JaegerExporter
   from opentelemetry.sdk.trace import TracerProvider
   from opentelemetry.sdk.trace.export import BatchSpanProcessor
   from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

   def setup_tracing(app, service_name: str = "llama-service"):
       """配置分布式追踪"""
       # 设置 TracerProvider
       provider = TracerProvider()
       trace.set_tracer_provider(provider)

       # Jaeger 导出器
       jaeger_exporter = JaegerExporter(
           agent_host_name="localhost",
           agent_port=6831,
       )
       provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

       # 自动检测 FastAPI
       FastAPIInstrumentor.instrument_app(app)

       return trace.get_tracer(service_name)

   # 使用
   tracer = setup_tracing(app)

   @app.post("/query")
   async def query(request: QueryRequest):
       with tracer.start_as_current_span("query") as span:
           span.set_attribute("query.collection", request.collection)
           span.set_attribute("query.text", request.question[:50])

           with tracer.start_as_current_span("retrieve"):
               nodes = retriever.retrieve(request.question)

           with tracer.start_as_current_span("synthesize"):
               response = synthesizer.synthesize(request.question, nodes)

           return {"answer": str(response)}

安全性
======

API 认证
--------

.. code-block:: python

   # auth.py
   from fastapi import HTTPException, Security
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
   import jwt
   from datetime import datetime, timedelta

   security = HTTPBearer()

   SECRET_KEY = "your-secret-key"  # 从环境变量获取
   ALGORITHM = "HS256"

   def create_token(user_id: str, expires_delta: timedelta = timedelta(hours=24)):
       """创建 JWT token"""
       expire = datetime.utcnow() + expires_delta
       payload = {
           "sub": user_id,
           "exp": expire,
           "iat": datetime.utcnow()
       }
       return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

   def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
       """验证 JWT token"""
       try:
           payload = jwt.decode(
               credentials.credentials,
               SECRET_KEY,
               algorithms=[ALGORITHM]
           )
           return payload["sub"]
       except jwt.ExpiredSignatureError:
           raise HTTPException(status_code=401, detail="Token expired")
       except jwt.JWTError:
           raise HTTPException(status_code=401, detail="Invalid token")

   # 使用
   @app.post("/query")
   async def query(request: QueryRequest, user_id: str = Security(verify_token)):
       # user_id 已验证
       result = query_engine.query(request.question)
       return result

输入验证
--------

.. code-block:: python

   # validation.py
   from pydantic import BaseModel, validator, Field
   from typing import Optional
   import re

   class QueryRequest(BaseModel):
       question: str = Field(..., min_length=1, max_length=1000)
       collection: str = Field(default="default", max_length=50)
       top_k: Optional[int] = Field(default=5, ge=1, le=20)

       @validator('question')
       def validate_question(cls, v):
           # 检查恶意内容
           if re.search(r'<script|javascript:|data:', v, re.IGNORECASE):
               raise ValueError('Invalid content in question')
           return v.strip()

       @validator('collection')
       def validate_collection(cls, v):
           if not re.match(r'^[a-zA-Z0-9_-]+$', v):
               raise ValueError('Invalid collection name')
           return v

速率限制
--------

.. code-block:: python

   # rate_limit.py
   from fastapi import HTTPException, Request
   from redis import Redis
   import time

   class RateLimiter:
       def __init__(self, redis_client: Redis, requests_per_minute: int = 60):
           self.redis = redis_client
           self.limit = requests_per_minute
           self.window = 60  # 1分钟

       async def check_rate_limit(self, request: Request):
           # 获取客户端标识
           client_id = request.client.host

           key = f"rate_limit:{client_id}"
           current = self.redis.get(key)

           if current is None:
               self.redis.setex(key, self.window, 1)
           elif int(current) >= self.limit:
               raise HTTPException(
                   status_code=429,
                   detail="Rate limit exceeded"
               )
           else:
               self.redis.incr(key)

   # 中间件
   from fastapi import FastAPI
   from starlette.middleware.base import BaseHTTPMiddleware

   class RateLimitMiddleware(BaseHTTPMiddleware):
       def __init__(self, app, rate_limiter: RateLimiter):
           super().__init__(app)
           self.rate_limiter = rate_limiter

       async def dispatch(self, request: Request, call_next):
           await self.rate_limiter.check_rate_limit(request)
           return await call_next(request)

   # 使用
   app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

高可用部署
==========

Kubernetes 部署
---------------

.. code-block:: yaml

   # kubernetes/deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: llama-app
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: llama-app
     template:
       metadata:
         labels:
           app: llama-app
       spec:
         containers:
         - name: llama-app
           image: llama-app:latest
           ports:
           - containerPort: 8000
           env:
           - name: OPENAI_API_KEY
             valueFrom:
               secretKeyRef:
                 name: llama-secrets
                 key: openai-api-key
           resources:
             requests:
               memory: "2Gi"
               cpu: "1"
             limits:
               memory: "4Gi"
               cpu: "2"
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 5
             periodSeconds: 5
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: llama-service
   spec:
     selector:
       app: llama-app
     ports:
     - port: 80
       targetPort: 8000
     type: LoadBalancer
   ---
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: llama-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: llama-app
     minReplicas: 2
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70

故障恢复
--------

.. code-block:: python

   # resilience.py
   from tenacity import retry, stop_after_attempt, wait_exponential
   from circuitbreaker import circuit

   # 重试装饰器
   @retry(
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=10)
   )
   def query_with_retry(query_engine, question: str):
       """带重试的查询"""
       return query_engine.query(question)

   # 熔断器
   @circuit(failure_threshold=5, recovery_timeout=60)
   def query_with_circuit_breaker(query_engine, question: str):
       """带熔断器的查询"""
       return query_engine.query(question)

   # 组合使用
   class ResilientQueryEngine:
       def __init__(self, query_engine, fallback_response: str = "服务暂时不可用"):
           self.query_engine = query_engine
           self.fallback_response = fallback_response

       @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
       @circuit(failure_threshold=5, recovery_timeout=60)
       def query(self, question: str):
           try:
               return self.query_engine.query(question)
           except Exception as e:
               logging.error(f"Query failed: {e}")
               return self.fallback_response

小结
====

本教程介绍了：

- Docker 和 Docker Compose 部署
- 性能优化：缓存、批量处理
- 监控：Prometheus 指标、日志、分布式追踪
- 安全性：认证、输入验证、速率限制
- 高可用：Kubernetes 部署、故障恢复

最佳实践总结
------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 方面
     - 建议
   * - 性能
     - 使用缓存、批量处理、异步操作
   * - 可靠性
     - 实现重试、熔断器、优雅降级
   * - 可观测性
     - 集成监控、日志、追踪
   * - 安全性
     - 认证授权、输入验证、速率限制
   * - 可扩展性
     - 容器化、自动扩缩容

练习
====

1. 使用 Docker Compose 部署完整的知识库系统
2. 配置 Prometheus + Grafana 监控仪表板
3. 实现基于 Redis 的查询缓存
4. 部署到 Kubernetes 并配置 HPA
