####################################
Tutorial 10: 生产环境部署
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

生产环境考量
============

将 LangGraph 应用部署到生产环境需要考虑：

- **可靠性**: 错误处理、重试机制、故障恢复
- **可扩展性**: 支持高并发、水平扩展
- **可观测性**: 日志、监控、追踪
- **安全性**: 认证、授权、数据保护
- **性能**: 响应时间、资源优化

架构设计
========

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                    Production Architecture                       │
   │                                                                  │
   │  ┌──────────────────────────────────────────────────────────┐   │
   │  │                      Load Balancer                        │   │
   │  └──────────────────────────┬───────────────────────────────┘   │
   │                             │                                    │
   │         ┌───────────────────┼───────────────────┐               │
   │         │                   │                   │               │
   │         ▼                   ▼                   ▼               │
   │  ┌────────────┐     ┌────────────┐     ┌────────────┐          │
   │  │  API Server │     │  API Server │     │  API Server │          │
   │  │  (FastAPI)  │     │  (FastAPI)  │     │  (FastAPI)  │          │
   │  └──────┬─────┘     └──────┬─────┘     └──────┬─────┘          │
   │         │                  │                  │                 │
   │         └──────────────────┼──────────────────┘                 │
   │                            │                                    │
   │  ┌─────────────────────────┼─────────────────────────┐         │
   │  │                         ▼                         │         │
   │  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    │         │
   │  │  │ PostgreSQL│    │  Redis   │    │ LangGraph│    │         │
   │  │  │(Checkpoint)│    │ (Cache)  │    │ (Workflow)│    │         │
   │  │  └──────────┘    └──────────┘    └──────────┘    │         │
   │  │                   Data Layer                      │         │
   │  └───────────────────────────────────────────────────┘         │
   │                                                                  │
   │  ┌───────────────────────────────────────────────────┐         │
   │  │              Monitoring & Observability            │         │
   │  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │         │
   │  │  │Prometheus│  │ Grafana │  │ Jaeger │  │  ELK   │  │         │
   │  │  └────────┘  └────────┘  └────────┘  └────────┘  │         │
   │  └───────────────────────────────────────────────────┘         │
   │                                                                  │
   └─────────────────────────────────────────────────────────────────┘

项目结构
========

.. code-block:: text

   selfmedia_workflow/
   ├── app/
   │   ├── __init__.py
   │   ├── main.py              # FastAPI 入口
   │   ├── config.py            # 配置管理
   │   ├── models.py            # Pydantic 模型
   │   ├── workflow/
   │   │   ├── __init__.py
   │   │   ├── graph.py         # LangGraph 定义
   │   │   ├── nodes.py         # 节点实现
   │   │   └── state.py         # 状态定义
   │   ├── api/
   │   │   ├── __init__.py
   │   │   ├── routes.py        # API 路由
   │   │   └── dependencies.py  # 依赖注入
   │   └── services/
   │       ├── __init__.py
   │       ├── task_service.py  # 任务服务
   │       └── publish_service.py
   ├── tests/
   │   ├── __init__.py
   │   ├── test_workflow.py
   │   └── test_api.py
   ├── docker/
   │   ├── Dockerfile
   │   └── docker-compose.yml
   ├── requirements.txt
   └── README.md

配置管理
========

.. code-block:: python

   # app/config.py
   from pydantic_settings import BaseSettings
   from functools import lru_cache

   class Settings(BaseSettings):
       # API
       api_host: str = "0.0.0.0"
       api_port: int = 8000
       debug: bool = False
       
       # Database
       database_url: str = "postgresql://user:pass@localhost/db"
       redis_url: str = "redis://localhost:6379"
       
       # LLM
       openai_api_key: str
       openai_model: str = "gpt-4o-mini"
       
       # Workflow
       max_iterations: int = 5
       default_timeout: int = 300
       
       class Config:
           env_file = ".env"

   @lru_cache()
   def get_settings() -> Settings:
       return Settings()

FastAPI 应用
============

.. code-block:: python

   # app/main.py
   from fastapi import FastAPI, HTTPException, BackgroundTasks
   from fastapi.middleware.cors import CORSMiddleware
   from contextlib import asynccontextmanager
   from app.config import get_settings
   from app.api.routes import router
   from app.workflow.graph import create_workflow
   import logging

   # 配置日志
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   logger = logging.getLogger(__name__)

   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # 启动时初始化
       logger.info("Starting application...")
       settings = get_settings()
       app.state.workflow = create_workflow(settings)
       yield
       # 关闭时清理
       logger.info("Shutting down...")

   app = FastAPI(
       title="Self-Media Content Workflow API",
       version="1.0.0",
       lifespan=lifespan
   )

   # CORS
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )

   # 路由
   app.include_router(router, prefix="/api/v1")

   @app.get("/health")
   async def health_check():
       return {"status": "healthy"}

API 路由
========

.. code-block:: python

   # app/api/routes.py
   from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
   from pydantic import BaseModel
   from typing import List, Optional
   from datetime import datetime
   import uuid

   router = APIRouter()

   # ===== 请求/响应模型 =====

   class CreateTaskRequest(BaseModel):
       topic: str
       platforms: List[str]
       style: str = "专业但易懂"
       requirements: Optional[str] = None

   class TaskResponse(BaseModel):
       task_id: str
       status: str
       stage: str
       created_at: str

   class ReviewRequest(BaseModel):
       approved: bool
       feedback: Optional[str] = ""

   class TaskStatusResponse(BaseModel):
       task_id: str
       status: str
       stage: str
       quality_score: Optional[int]
       content_preview: Optional[str]
       logs: List[str]

   # ===== 任务存储（生产环境用数据库）=====

   tasks = {}

   # ===== API 端点 =====

   @router.post("/tasks", response_model=TaskResponse)
   async def create_task(
       request: CreateTaskRequest,
       background_tasks: BackgroundTasks
   ):
       """创建内容生产任务"""
       task_id = str(uuid.uuid4())[:8]
       
       initial_state = {
           "task_id": task_id,
           "topic": request.topic,
           "target_platforms": request.platforms,
           "style": request.style,
           "requirements": request.requirements,
           "stage": "init",
           "iteration": 0,
           "max_iterations": 3,
           "created_at": datetime.now().isoformat(),
           "logs": []
       }
       
       tasks[task_id] = {
           "state": initial_state,
           "status": "running"
       }
       
       # 后台执行工作流
       background_tasks.add_task(run_workflow, task_id, initial_state)
       
       return TaskResponse(
           task_id=task_id,
           status="running",
           stage="init",
           created_at=initial_state["created_at"]
       )

   @router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
   async def get_task_status(task_id: str):
       """获取任务状态"""
       if task_id not in tasks:
           raise HTTPException(status_code=404, detail="Task not found")
       
       task = tasks[task_id]
       state = task["state"]
       
       return TaskStatusResponse(
           task_id=task_id,
           status=task["status"],
           stage=state.get("stage", "unknown"),
           quality_score=state.get("quality_score"),
           content_preview=state.get("draft_content", "")[:200] if state.get("draft_content") else None,
           logs=state.get("logs", [])[-10:]
       )

   @router.post("/tasks/{task_id}/review")
   async def submit_review(
       task_id: str,
       request: ReviewRequest,
       background_tasks: BackgroundTasks
   ):
       """提交人工审核"""
       if task_id not in tasks:
           raise HTTPException(status_code=404, detail="Task not found")
       
       task = tasks[task_id]
       
       if task["status"] != "pending_review":
           raise HTTPException(status_code=400, detail="Task not pending review")
       
       # 更新审核结果
       task["state"]["human_review"] = {
           "approved": request.approved,
           "feedback": request.feedback,
           "timestamp": datetime.now().isoformat()
       }
       
       # 继续执行工作流
       background_tasks.add_task(continue_workflow, task_id)
       
       return {"message": "Review submitted", "task_id": task_id}

   @router.get("/tasks/{task_id}/result")
   async def get_task_result(task_id: str):
       """获取任务结果"""
       if task_id not in tasks:
           raise HTTPException(status_code=404, detail="Task not found")
       
       task = tasks[task_id]
       
       if task["status"] != "completed":
           raise HTTPException(status_code=400, detail="Task not completed")
       
       state = task["state"]
       
       return {
           "task_id": task_id,
           "title": state.get("selected_title"),
           "content": state.get("draft_content"),
           "adapted_contents": state.get("adapted_contents", {}),
           "publish_results": state.get("publish_results", [])
       }

工作流执行
==========

.. code-block:: python

   # app/workflow/executor.py
   from langgraph.checkpoint.postgres import PostgresSaver
   from app.config import get_settings
   from app.workflow.graph import create_workflow
   import logging

   logger = logging.getLogger(__name__)

   async def run_workflow(task_id: str, initial_state: dict):
       """执行工作流"""
       settings = get_settings()
       
       try:
           # 创建持久化检查点
           with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
               workflow = create_workflow(checkpointer)
               config = {"configurable": {"thread_id": task_id}}
               
               # 执行到人工审核点
               result = workflow.invoke(initial_state, config)
               
               # 更新任务状态
               tasks[task_id]["state"] = result
               
               if result.get("stage") == "human_review":
                   tasks[task_id]["status"] = "pending_review"
               elif result.get("stage") == "completed":
                   tasks[task_id]["status"] = "completed"
               else:
                   tasks[task_id]["status"] = "running"
                   
       except Exception as e:
           logger.error(f"Workflow error for task {task_id}: {e}")
           tasks[task_id]["status"] = "failed"
           tasks[task_id]["state"]["errors"] = [str(e)]

   async def continue_workflow(task_id: str):
       """继续执行工作流"""
       settings = get_settings()
       
       try:
           with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
               workflow = create_workflow(checkpointer)
               config = {"configurable": {"thread_id": task_id}}
               
               # 注入人工审核结果
               workflow.update_state(
                   config,
                   tasks[task_id]["state"]
               )
               
               # 继续执行
               result = workflow.invoke(None, config)
               
               tasks[task_id]["state"] = result
               tasks[task_id]["status"] = "completed" if result.get("stage") == "completed" else "running"
               
       except Exception as e:
           logger.error(f"Continue workflow error for task {task_id}: {e}")
           tasks[task_id]["status"] = "failed"

Docker 部署
===========

Dockerfile
----------

.. code-block:: dockerfile

   # docker/Dockerfile
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

   # 暴露端口
   EXPOSE 8000

   # 启动命令
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

docker-compose.yml
------------------

.. code-block:: yaml

   # docker/docker-compose.yml
   version: '3.8'

   services:
     api:
       build:
         context: ..
         dockerfile: docker/Dockerfile
       ports:
         - "8000:8000"
       environment:
         - DATABASE_URL=postgresql://postgres:postgres@db:5432/workflow
         - REDIS_URL=redis://redis:6379
         - OPENAI_API_KEY=${OPENAI_API_KEY}
       depends_on:
         - db
         - redis
       restart: unless-stopped

     db:
       image: postgres:15
       environment:
         - POSTGRES_USER=postgres
         - POSTGRES_PASSWORD=postgres
         - POSTGRES_DB=workflow
       volumes:
         - postgres_data:/var/lib/postgresql/data
       restart: unless-stopped

     redis:
       image: redis:7-alpine
       restart: unless-stopped

   volumes:
     postgres_data:

监控与日志
==========

结构化日志
----------

.. code-block:: python

   import structlog
   from datetime import datetime

   # 配置结构化日志
   structlog.configure(
       processors=[
           structlog.processors.TimeStamper(fmt="iso"),
           structlog.processors.JSONRenderer()
       ]
   )

   logger = structlog.get_logger()

   # 在节点中使用
   def my_node(state):
       logger.info(
           "node_executed",
           node="my_node",
           task_id=state["task_id"],
           stage=state["stage"]
       )
       # ...

Prometheus 指标
---------------

.. code-block:: python

   from prometheus_client import Counter, Histogram, generate_latest
   from fastapi import Response

   # 定义指标
   TASK_COUNTER = Counter(
       'workflow_tasks_total',
       'Total number of tasks',
       ['status']
   )

   TASK_DURATION = Histogram(
       'workflow_task_duration_seconds',
       'Task duration in seconds',
       ['stage']
   )

   # 指标端点
   @app.get("/metrics")
   async def metrics():
       return Response(
           generate_latest(),
           media_type="text/plain"
       )

   # 在工作流中记录指标
   def track_metrics(func):
       async def wrapper(state, *args, **kwargs):
           start_time = time.time()
           result = await func(state, *args, **kwargs)
           duration = time.time() - start_time
           
           TASK_DURATION.labels(stage=state["stage"]).observe(duration)
           
           return result
       return wrapper

错误处理与重试
==============

.. code-block:: python

   from tenacity import retry, stop_after_attempt, wait_exponential
   import logging

   logger = logging.getLogger(__name__)

   @retry(
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=4, max=10)
   )
   async def call_llm_with_retry(prompt: str):
       """带重试的 LLM 调用"""
       try:
           response = await llm.ainvoke(prompt)
           return response.content
       except Exception as e:
           logger.warning(f"LLM call failed, retrying: {e}")
           raise

   def safe_node(func):
       """节点错误处理装饰器"""
       async def wrapper(state):
           try:
               return await func(state)
           except Exception as e:
               logger.error(f"Node error: {e}", exc_info=True)
               return {
                   "errors": state.get("errors", []) + [str(e)],
                   "stage": "error"
               }
       return wrapper

安全最佳实践
============

1. **API 认证**

.. code-block:: python

   from fastapi import Depends, HTTPException, Security
   from fastapi.security import APIKeyHeader

   api_key_header = APIKeyHeader(name="X-API-Key")

   async def verify_api_key(api_key: str = Security(api_key_header)):
       if api_key != settings.api_key:
           raise HTTPException(status_code=403, detail="Invalid API key")
       return api_key

   @router.post("/tasks", dependencies=[Depends(verify_api_key)])
   async def create_task(...):
       ...

2. **输入验证**

.. code-block:: python

   from pydantic import BaseModel, validator

   class CreateTaskRequest(BaseModel):
       topic: str
       platforms: List[str]
       
       @validator('topic')
       def validate_topic(cls, v):
           if len(v) > 200:
               raise ValueError('Topic too long')
           return v
       
       @validator('platforms')
       def validate_platforms(cls, v):
           valid = ["微信公众号", "知乎", "小红书"]
           for p in v:
               if p not in valid:
                   raise ValueError(f'Invalid platform: {p}')
           return v

3. **速率限制**

.. code-block:: python

   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address

   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter

   @router.post("/tasks")
   @limiter.limit("10/minute")
   async def create_task(request: Request, ...):
       ...

总结
====

恭喜你完成了 LangGraph 全部教程！

你已经学会了：

1. ✅ LangGraph 核心概念（State、Node、Edge）
2. ✅ 条件路由和循环控制
3. ✅ Human-in-the-Loop 人工干预
4. ✅ 状态持久化和检查点
5. ✅ 多 Agent 协作系统
6. ✅ 完整的自媒体内容工作流
7. ✅ 生产环境部署最佳实践

下一步建议
==========

1. **深入学习**: 阅读 LangGraph 官方文档
2. **实践项目**: 构建自己的 Agent 应用
3. **社区交流**: 加入 LangChain 社区
4. **持续优化**: 根据实际需求迭代改进

🎉 祝你在 AI Agent 开发之路上越走越远！
