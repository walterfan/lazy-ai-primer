####################################
Tutorial 7: 持久化与检查点
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

为什么需要持久化？
==================

持久化让你的工作流能够：

- **断点恢复**: 系统重启后继续执行
- **状态追踪**: 查看历史状态和执行路径
- **多会话管理**: 支持多个并发工作流
- **时间旅行**: 回滚到之前的状态

LangGraph 检查点系统
====================

LangGraph 使用 Checkpointer 来保存和恢复状态：

.. code-block:: python

   from langgraph.checkpoint.memory import MemorySaver
   from langgraph.checkpoint.sqlite import SqliteSaver
   from langgraph.checkpoint.postgres import PostgresSaver

   # 内存存储（开发用）
   memory_saver = MemorySaver()

   # SQLite 存储（单机持久化）
   sqlite_saver = SqliteSaver.from_conn_string("checkpoints.db")

   # PostgreSQL 存储（生产环境）
   postgres_saver = PostgresSaver.from_conn_string(
       "postgresql://user:pass@localhost/db"
   )

基本用法
========

.. code-block:: python

   from typing import TypedDict
   from langgraph.graph import StateGraph, START, END
   from langgraph.checkpoint.memory import MemorySaver

   class State(TypedDict):
       count: int
       messages: list

   def increment(state: State) -> dict:
       return {"count": state["count"] + 1}

   # 构建图
   graph = StateGraph(State)
   graph.add_node("increment", increment)
   graph.add_edge(START, "increment")
   graph.add_edge("increment", END)

   # 编译（带检查点）
   memory = MemorySaver()
   app = graph.compile(checkpointer=memory)

   # 运行（需要 thread_id）
   config = {"configurable": {"thread_id": "session_1"}}
   result = app.invoke({"count": 0, "messages": []}, config)

   print(f"Count: {result['count']}")  # 1

   # 再次运行同一会话
   result = app.invoke({"count": 0, "messages": []}, config)
   # 注意：每次 invoke 是独立的，但状态历史会保存

Thread ID 管理
==============

Thread ID 用于区分不同的工作流会话：

.. code-block:: python

   import uuid

   # 为每个用户/任务创建唯一 ID
   user_thread_id = f"user_{user_id}_{uuid.uuid4()}"

   config = {"configurable": {"thread_id": user_thread_id}}

   # 不同 thread_id 的工作流相互独立
   config_a = {"configurable": {"thread_id": "task_a"}}
   config_b = {"configurable": {"thread_id": "task_b"}}

   result_a = app.invoke(state, config_a)
   result_b = app.invoke(state, config_b)

状态历史
========

.. code-block:: python

   # 获取当前状态
   current = app.get_state(config)
   print(f"当前值: {current.values}")
   print(f"下一步: {current.next}")
   print(f"配置: {current.config}")

   # 获取状态历史
   for state in app.get_state_history(config):
       print(f"时间: {state.created_at}")
       print(f"节点: {state.next}")
       print(f"值: {state.values}")
       print("---")

时间旅行（回滚）
================

.. code-block:: python

   # 获取历史状态
   history = list(app.get_state_history(config))

   # 选择要回滚到的状态
   target_state = history[2]  # 例如：回滚到第3个状态

   # 从该状态继续执行
   result = app.invoke(
       None,
       {"configurable": {
           "thread_id": config["configurable"]["thread_id"],
           "checkpoint_id": target_state.config["configurable"]["checkpoint_id"]
       }}
   )

使用 SQLite 持久化
==================

.. code-block:: python

   from langgraph.checkpoint.sqlite import SqliteSaver

   # 创建 SQLite 检查点
   with SqliteSaver.from_conn_string("workflow_state.db") as checkpointer:
       app = graph.compile(checkpointer=checkpointer)
       
       config = {"configurable": {"thread_id": "persistent_session"}}
       
       # 第一次运行
       result = app.invoke(initial_state, config)
       print("第一次运行完成")

   # 程序重启后...

   with SqliteSaver.from_conn_string("workflow_state.db") as checkpointer:
       app = graph.compile(checkpointer=checkpointer)
       
       # 恢复之前的会话
       config = {"configurable": {"thread_id": "persistent_session"}}
       
       # 获取之前的状态
       previous_state = app.get_state(config)
       print(f"恢复状态: {previous_state.values}")
       
       # 继续执行
       if previous_state.next:
           result = app.invoke(None, config)

实战：可恢复的内容创作流程
==========================

.. code-block:: python

   from typing import TypedDict, List, Optional, Annotated
   from operator import add
   from langgraph.graph import StateGraph, START, END
   from langgraph.checkpoint.sqlite import SqliteSaver
   from langchain_openai import ChatOpenAI
   from datetime import datetime
   import json

   # ========== 状态定义 ==========

   class ContentState(TypedDict):
       # 任务信息
       task_id: str
       topic: str
       platform: str
       
       # 工作流状态
       current_stage: str
       started_at: str
       updated_at: str
       
       # 内容
       research: Optional[str]
       outline: Optional[dict]
       draft: Optional[str]
       final: Optional[str]
       
       # 日志
       logs: Annotated[List[str], add]

   # ========== 节点定义 ==========

   llm = ChatOpenAI(model="gpt-4o-mini")

   def log_progress(stage: str, message: str) -> dict:
       """记录进度"""
       timestamp = datetime.now().isoformat()
       return {
           "current_stage": stage,
           "updated_at": timestamp,
           "logs": [f"[{timestamp}] {stage}: {message}"]
       }

   def research_node(state: ContentState) -> dict:
       """研究阶段"""
       topic = state["topic"]
       
       response = llm.invoke(f"分析话题「{topic}」的要点和角度")
       
       return {
           **log_progress("research", "研究完成"),
           "research": response.content
       }

   def outline_node(state: ContentState) -> dict:
       """大纲阶段"""
       research = state["research"]
       
       response = llm.invoke(f"""
   基于研究结果创建大纲：
   {research}

   输出JSON: {{"title": "标题", "sections": ["章节1", "章节2"]}}
   """)
       
       try:
           outline = json.loads(response.content)
       except:
           outline = {"title": state["topic"], "sections": ["介绍", "主体", "结论"]}
       
       return {
           **log_progress("outline", "大纲完成"),
           "outline": outline
       }

   def draft_node(state: ContentState) -> dict:
       """草稿阶段"""
       outline = state["outline"]
       platform = state["platform"]
       
       response = llm.invoke(f"""
   根据大纲写文章：
   {json.dumps(outline, ensure_ascii=False)}
   平台：{platform}
   """)
       
       return {
           **log_progress("draft", "草稿完成"),
           "draft": response.content
       }

   def finalize_node(state: ContentState) -> dict:
       """定稿阶段"""
       draft = state["draft"]
       
       response = llm.invoke(f"优化并定稿：\n{draft}")
       
       return {
           **log_progress("finalize", "定稿完成"),
           "final": response.content
       }

   # ========== 构建图 ==========

   def create_persistent_workflow(db_path: str = "content_workflow.db"):
       graph = StateGraph(ContentState)
       
       graph.add_node("research", research_node)
       graph.add_node("outline", outline_node)
       graph.add_node("draft", draft_node)
       graph.add_node("finalize", finalize_node)
       
       graph.add_edge(START, "research")
       graph.add_edge("research", "outline")
       graph.add_edge("outline", "draft")
       graph.add_edge("draft", "finalize")
       graph.add_edge("finalize", END)
       
       # 使用 SQLite 持久化
       checkpointer = SqliteSaver.from_conn_string(db_path)
       
       return graph.compile(checkpointer=checkpointer), checkpointer

   # ========== 任务管理器 ==========

   class ContentTaskManager:
       def __init__(self, db_path: str = "content_workflow.db"):
           self.db_path = db_path
           self.workflow, self.checkpointer = create_persistent_workflow(db_path)
       
       def create_task(self, topic: str, platform: str) -> str:
           """创建新任务"""
           import uuid
           task_id = str(uuid.uuid4())[:8]
           
           initial_state = {
               "task_id": task_id,
               "topic": topic,
               "platform": platform,
               "current_stage": "created",
               "started_at": datetime.now().isoformat(),
               "updated_at": datetime.now().isoformat(),
               "logs": [f"任务创建: {topic}"]
           }
           
           config = {"configurable": {"thread_id": task_id}}
           
           # 开始执行
           try:
               result = self.workflow.invoke(initial_state, config)
               return task_id
           except Exception as e:
               print(f"任务执行出错: {e}")
               return task_id
       
       def get_task_status(self, task_id: str) -> dict:
           """获取任务状态"""
           config = {"configurable": {"thread_id": task_id}}
           state = self.workflow.get_state(config)
           
           if state.values:
               return {
                   "task_id": task_id,
                   "stage": state.values.get("current_stage"),
                   "next": state.next,
                   "updated_at": state.values.get("updated_at"),
                   "logs": state.values.get("logs", [])[-5:]  # 最近5条日志
               }
           return {"task_id": task_id, "status": "not_found"}
       
       def resume_task(self, task_id: str) -> dict:
           """恢复任务"""
           config = {"configurable": {"thread_id": task_id}}
           state = self.workflow.get_state(config)
           
           if state.next:
               print(f"从 {state.next} 继续执行...")
               result = self.workflow.invoke(None, config)
               return result
           else:
               print("任务已完成")
               return state.values
       
       def get_task_history(self, task_id: str) -> List[dict]:
           """获取任务历史"""
           config = {"configurable": {"thread_id": task_id}}
           history = []
           
           for state in self.workflow.get_state_history(config):
               history.append({
                   "stage": state.values.get("current_stage"),
                   "next": state.next,
                   "checkpoint_id": state.config["configurable"].get("checkpoint_id")
               })
           
           return history
       
       def rollback_task(self, task_id: str, checkpoint_id: str) -> dict:
           """回滚到指定检查点"""
           config = {
               "configurable": {
                   "thread_id": task_id,
                   "checkpoint_id": checkpoint_id
               }
           }
           
           result = self.workflow.invoke(None, config)
           return result

   # ========== 使用示例 ==========

   def demo():
       manager = ContentTaskManager()
       
       # 创建任务
       print("创建任务...")
       task_id = manager.create_task("AI编程入门", "微信公众号")
       print(f"任务ID: {task_id}")
       
       # 查看状态
       print("\n任务状态:")
       status = manager.get_task_status(task_id)
       print(json.dumps(status, ensure_ascii=False, indent=2))
       
       # 查看历史
       print("\n执行历史:")
       history = manager.get_task_history(task_id)
       for h in history[:5]:
           print(f"  {h['stage']} -> {h['next']}")
       
       # 模拟程序重启后恢复
       print("\n模拟重启后恢复...")
       manager2 = ContentTaskManager()
       status2 = manager2.get_task_status(task_id)
       print(f"恢复的状态: {status2['stage']}")

   if __name__ == "__main__":
       demo()

检查点配置选项
==============

.. code-block:: python

   # 配置检查点保存频率
   app = graph.compile(
       checkpointer=memory,
       # 每个节点后保存（默认）
   )

   # 自定义配置
   config = {
       "configurable": {
           "thread_id": "my_thread",
           "checkpoint_ns": "namespace",  # 命名空间
       }
   }

最佳实践
========

1. **选择合适的存储**

.. code-block:: python

   # 开发环境
   checkpointer = MemorySaver()

   # 单机生产
   checkpointer = SqliteSaver.from_conn_string("prod.db")

   # 分布式生产
   checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)

2. **清理过期数据**

.. code-block:: python

   # 定期清理旧的检查点
   async def cleanup_old_checkpoints(days: int = 7):
       cutoff = datetime.now() - timedelta(days=days)
       # 实现清理逻辑...

3. **错误恢复**

.. code-block:: python

   def safe_invoke(app, state, config, max_retries=3):
       for attempt in range(max_retries):
           try:
               return app.invoke(state, config)
           except Exception as e:
               print(f"尝试 {attempt + 1} 失败: {e}")
               if attempt < max_retries - 1:
                   # 从最后的检查点恢复
                   state = None  # invoke(None) 从上次状态继续
       raise Exception("达到最大重试次数")

下一步
======

在下一个教程中，我们将学习如何实现多 Agent 协作。

:doc:`tutorial_08_multi_agent`
