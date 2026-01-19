####################################
Tutorial 6: Human-in-the-Loop
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是 Human-in-the-Loop？
==========================

Human-in-the-Loop（人工干预）是指在 AI 工作流中加入人工审核、确认或输入的环节。

常见场景：

- **内容审核**: 发布前人工确认
- **决策确认**: 重要操作前获取批准
- **数据校正**: 人工修正 AI 输出
- **反馈收集**: 获取用户反馈优化结果

LangGraph 的中断机制
====================

LangGraph 通过 ``interrupt_before`` 和 ``interrupt_after`` 实现人工干预：

.. code-block:: python

   from langgraph.graph import StateGraph
   from langgraph.checkpoint.memory import MemorySaver

   # 创建检查点存储
   memory = MemorySaver()

   # 编译时指定中断点
   app = graph.compile(
       checkpointer=memory,
       interrupt_before=["human_review"]  # 在此节点前中断
   )

基本用法
========

.. code-block:: python

   from typing import TypedDict
   from langgraph.graph import StateGraph, START, END
   from langgraph.checkpoint.memory import MemorySaver

   class State(TypedDict):
       content: str
       approved: bool
       feedback: str

   def generate_content(state: State) -> dict:
       return {"content": "AI生成的内容..."}

   def human_review(state: State) -> dict:
       # 这个节点会在执行前中断，等待人工输入
       return {}

   def publish(state: State) -> dict:
       return {"content": f"[已发布] {state['content']}"}

   # 构建图
   graph = StateGraph(State)
   graph.add_node("generate", generate_content)
   graph.add_node("review", human_review)
   graph.add_node("publish", publish)

   graph.add_edge(START, "generate")
   graph.add_edge("generate", "review")
   graph.add_edge("review", "publish")
   graph.add_edge("publish", END)

   # 编译（带中断点）
   memory = MemorySaver()
   app = graph.compile(
       checkpointer=memory,
       interrupt_before=["review"]
   )

   # 运行到中断点
   config = {"configurable": {"thread_id": "1"}}
   result = app.invoke({"content": "", "approved": False}, config)

   print("当前状态:", result)
   print("等待人工审核...")

   # 人工审核后，更新状态并继续
   app.update_state(
       config,
       {"approved": True, "feedback": "内容不错，可以发布"}
   )

   # 继续执行
   final_result = app.invoke(None, config)
   print("最终结果:", final_result)

中断点类型
==========

interrupt_before
----------------

在指定节点执行**之前**中断：

.. code-block:: python

   app = graph.compile(
       checkpointer=memory,
       interrupt_before=["sensitive_action"]
   )

   # 执行流程：
   # START -> node_a -> [中断] -> sensitive_action -> END

interrupt_after
---------------

在指定节点执行**之后**中断：

.. code-block:: python

   app = graph.compile(
       checkpointer=memory,
       interrupt_after=["generate_content"]
   )

   # 执行流程：
   # START -> generate_content -> [中断] -> next_node -> END

实战：自媒体内容审核流程
========================

.. code-block:: python

   from typing import TypedDict, List, Optional, Literal
   from langgraph.graph import StateGraph, START, END
   from langgraph.checkpoint.memory import MemorySaver
   from langchain_openai import ChatOpenAI
   from datetime import datetime

   # ========== 状态定义 ==========

   class ReviewState(TypedDict):
       # 内容
       topic: str
       draft_content: str
       final_content: str
       
       # 审核状态
       ai_review: dict
       human_review: Optional[dict]
       
       # 决策
       status: str  # draft, pending_review, approved, rejected, published

   # ========== 节点定义 ==========

   llm = ChatOpenAI(model="gpt-4o-mini")

   def generate_draft(state: ReviewState) -> dict:
       """生成草稿"""
       topic = state["topic"]
       
       response = llm.invoke(f"写一篇关于「{topic}」的短文，300字左右")
       
       return {
           "draft_content": response.content,
           "status": "draft"
       }

   def ai_review(state: ReviewState) -> dict:
       """AI 预审"""
       content = state["draft_content"]
       
       response = llm.invoke(f"""
   审核以下内容，给出评分和建议：

   {content}

   输出格式：
   评分：[0-100]
   建议：[具体建议]
   风险：[有/无]
   """)
       
       return {
           "ai_review": {
               "result": response.content,
               "timestamp": datetime.now().isoformat()
           },
           "status": "pending_review"
       }

   def human_review_node(state: ReviewState) -> dict:
       """人工审核节点（会在此处中断）"""
       # 人工审核的结果会通过 update_state 注入
       human_review = state.get("human_review", {})
       
       if human_review.get("approved"):
           return {"status": "approved"}
       elif human_review.get("rejected"):
           return {"status": "rejected"}
       else:
           return {}

   def apply_feedback(state: ReviewState) -> dict:
       """应用人工反馈"""
       content = state["draft_content"]
       feedback = state.get("human_review", {}).get("feedback", "")
       
       if feedback:
           response = llm.invoke(f"""
   根据反馈修改内容：

   原文：
   {content}

   反馈：{feedback}

   输出修改后的完整内容。
   """)
           return {"final_content": response.content}
       
       return {"final_content": content}

   def publish(state: ReviewState) -> dict:
       """发布"""
       return {"status": "published"}

   def reject(state: ReviewState) -> dict:
       """拒绝"""
       return {"status": "rejected"}

   # ========== 路由函数 ==========

   def review_router(state: ReviewState) -> Literal["apply_feedback", "reject"]:
       if state["status"] == "approved":
           return "apply_feedback"
       return "reject"

   # ========== 构建图 ==========

   def create_review_workflow():
       graph = StateGraph(ReviewState)
       
       graph.add_node("generate", generate_draft)
       graph.add_node("ai_review", ai_review)
       graph.add_node("human_review", human_review_node)
       graph.add_node("apply_feedback", apply_feedback)
       graph.add_node("publish", publish)
       graph.add_node("reject", reject)
       
       graph.add_edge(START, "generate")
       graph.add_edge("generate", "ai_review")
       graph.add_edge("ai_review", "human_review")
       
       graph.add_conditional_edges(
           "human_review",
           review_router,
           {
               "apply_feedback": "apply_feedback",
               "reject": "reject"
           }
       )
       
       graph.add_edge("apply_feedback", "publish")
       graph.add_edge("publish", END)
       graph.add_edge("reject", END)
       
       # 编译（在人工审核前中断）
       memory = MemorySaver()
       return graph.compile(
           checkpointer=memory,
           interrupt_before=["human_review"]
       )

   # ========== 使用示例 ==========

   def run_with_human_review():
       workflow = create_review_workflow()
       
       # 配置（用于追踪会话）
       config = {"configurable": {"thread_id": "review_001"}}
       
       # 第一阶段：生成内容并 AI 预审
       print("=" * 50)
       print("第一阶段：生成和AI预审")
       print("=" * 50)
       
       result = workflow.invoke(
           {"topic": "AI编程入门", "status": ""},
           config
       )
       
       print(f"草稿内容:\n{result['draft_content'][:200]}...")
       print(f"\nAI审核结果:\n{result['ai_review']['result']}")
       print(f"\n状态: {result['status']}")
       
       # 模拟人工审核
       print("\n" + "=" * 50)
       print("等待人工审核...")
       print("=" * 50)
       
       # 获取人工输入（实际应用中这里会是 UI 交互）
       human_decision = input("批准发布? (y/n): ").strip().lower()
       human_feedback = ""
       
       if human_decision == 'y':
           human_feedback = input("有修改建议吗? (直接回车跳过): ").strip()
           human_review_result = {
               "approved": True,
               "rejected": False,
               "feedback": human_feedback,
               "reviewer": "human",
               "timestamp": datetime.now().isoformat()
           }
       else:
           reject_reason = input("拒绝原因: ").strip()
           human_review_result = {
               "approved": False,
               "rejected": True,
               "feedback": reject_reason,
               "reviewer": "human",
               "timestamp": datetime.now().isoformat()
           }
       
       # 注入人工审核结果
       workflow.update_state(
           config,
           {"human_review": human_review_result}
       )
       
       # 第二阶段：继续执行
       print("\n" + "=" * 50)
       print("第二阶段：处理审核结果")
       print("=" * 50)
       
       final_result = workflow.invoke(None, config)
       
       print(f"最终状态: {final_result['status']}")
       if final_result.get('final_content'):
           print(f"最终内容:\n{final_result['final_content'][:300]}...")
       
       return final_result

   # 运行
   if __name__ == "__main__":
       run_with_human_review()

查看和管理状态
==============

.. code-block:: python

   # 获取当前状态
   current_state = workflow.get_state(config)
   print(f"当前状态: {current_state.values}")
   print(f"下一个节点: {current_state.next}")

   # 获取状态历史
   for state in workflow.get_state_history(config):
       print(f"节点: {state.next}, 状态: {state.values}")

   # 更新状态
   workflow.update_state(
       config,
       {"field": "new_value"},
       as_node="node_name"  # 可选：指定作为哪个节点的输出
   )

Web 应用集成示例
================

.. code-block:: python

   from fastapi import FastAPI, HTTPException
   from pydantic import BaseModel
   from typing import Optional
   import uuid

   app = FastAPI()

   # 存储工作流实例
   workflows = {}

   class StartRequest(BaseModel):
       topic: str

   class ReviewRequest(BaseModel):
       session_id: str
       approved: bool
       feedback: Optional[str] = ""

   @app.post("/start")
   async def start_workflow(request: StartRequest):
       """启动工作流"""
       session_id = str(uuid.uuid4())
       config = {"configurable": {"thread_id": session_id}}
       
       workflow = create_review_workflow()
       workflows[session_id] = workflow
       
       result = workflow.invoke(
           {"topic": request.topic, "status": ""},
           config
       )
       
       return {
           "session_id": session_id,
           "draft": result["draft_content"],
           "ai_review": result["ai_review"],
           "status": "pending_review"
       }

   @app.post("/review")
   async def submit_review(request: ReviewRequest):
       """提交人工审核"""
       workflow = workflows.get(request.session_id)
       if not workflow:
           raise HTTPException(404, "Session not found")
       
       config = {"configurable": {"thread_id": request.session_id}}
       
       # 注入审核结果
       workflow.update_state(
           config,
           {
               "human_review": {
                   "approved": request.approved,
                   "rejected": not request.approved,
                   "feedback": request.feedback
               }
           }
       )
       
       # 继续执行
       final_result = workflow.invoke(None, config)
       
       return {
           "status": final_result["status"],
           "final_content": final_result.get("final_content")
       }

最佳实践
========

1. **明确中断点**

.. code-block:: python

   # ✅ 好：在关键决策点中断
   interrupt_before=["publish", "delete", "send_email"]

   # ❌ 差：在每个节点都中断
   interrupt_before=["node1", "node2", "node3", ...]

2. **提供清晰的上下文**

.. code-block:: python

   def prepare_for_review(state):
       """准备审核所需的信息"""
       return {
           "review_context": {
               "content": state["content"],
               "ai_suggestions": state["ai_review"],
               "risk_level": state["risk_assessment"],
               "previous_versions": state.get("history", [])
           }
       }

3. **处理超时**

.. code-block:: python

   import asyncio

   async def wait_for_review(workflow, config, timeout=3600):
       """等待人工审核，带超时"""
       start_time = time.time()
       
       while time.time() - start_time < timeout:
           state = workflow.get_state(config)
           if state.values.get("human_review"):
               return True
           await asyncio.sleep(5)
       
       # 超时处理
       workflow.update_state(config, {"status": "timeout"})
       return False

下一步
======

在下一个教程中，我们将学习如何持久化状态和使用检查点。

:doc:`tutorial_07_persistence`
