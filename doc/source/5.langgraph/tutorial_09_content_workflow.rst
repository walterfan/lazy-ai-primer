####################################
Tutorial 9: 自媒体内容工作流
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

项目概述
========

本教程将整合前面所学的所有知识，构建一个完整的自媒体内容生产工作流：

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                  Complete Content Workflow                       │
   │                                                                  │
   │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
   │   │ 话题发现 │───►│ 内容策划 │───►│ 内容创作 │───►│ 质量审核 │     │
   │   └─────────┘    └─────────┘    └─────────┘    └────┬────┘     │
   │                                                      │          │
   │                    ┌─────────────────────────────────┘          │
   │                    │                                            │
   │                    ▼                                            │
   │              ┌──────────┐                                       │
   │              │ 人工审核  │  ◄── Human-in-the-Loop               │
   │              └────┬─────┘                                       │
   │                   │                                             │
   │         ┌─────────┴─────────┐                                   │
   │         ▼                   ▼                                   │
   │   ┌──────────┐        ┌──────────┐                             │
   │   │ 平台适配  │        │   修改   │                             │
   │   └────┬─────┘        └──────────┘                             │
   │        │                                                        │
   │        ▼                                                        │
   │   ┌──────────┐                                                  │
   │   │ 多平台发布 │                                                 │
   │   └────┬─────┘                                                  │
   │        │                                                        │
   │        ▼                                                        │
   │   ┌──────────┐                                                  │
   │   │ 数据追踪  │                                                  │
   │   └──────────┘                                                  │
   │                                                                  │
   └─────────────────────────────────────────────────────────────────┘

完整实现
========

状态定义
--------

.. code-block:: python

   from typing import TypedDict, List, Dict, Optional, Annotated, Literal
   from operator import add
   from datetime import datetime
   from enum import Enum

   class WorkflowStage(str, Enum):
       INIT = "init"
       DISCOVERY = "discovery"
       PLANNING = "planning"
       WRITING = "writing"
       AI_REVIEW = "ai_review"
       HUMAN_REVIEW = "human_review"
       ADAPTING = "adapting"
       PUBLISHING = "publishing"
       TRACKING = "tracking"
       COMPLETED = "completed"
       FAILED = "failed"

   class ContentWorkflowState(TypedDict):
       # ===== 任务配置 =====
       task_id: str
       topic: str
       target_platforms: List[str]
       style: str
       requirements: Optional[str]
       
       # ===== 工作流控制 =====
       stage: WorkflowStage
       iteration: int
       max_iterations: int
       created_at: str
       updated_at: str
       
       # ===== 话题发现 =====
       trending_topics: List[dict]
       selected_topic: Optional[dict]
       topic_analysis: Optional[str]
       
       # ===== 内容策划 =====
       content_angle: str
       outline: Optional[dict]
       title_candidates: List[str]
       selected_title: str
       
       # ===== 内容创作 =====
       draft_content: str
       word_count: int
       
       # ===== 质量审核 =====
       ai_review_result: Optional[dict]
       quality_score: int
       issues: List[str]
       
       # ===== 人工审核 =====
       human_review: Optional[dict]
       human_approved: bool
       human_feedback: str
       
       # ===== 平台适配 =====
       adapted_contents: Dict[str, dict]
       
       # ===== 发布结果 =====
       publish_results: List[dict]
       
       # ===== 日志 =====
       logs: Annotated[List[str], add]
       errors: List[str]

节点实现
--------

.. code-block:: python

   from langgraph.graph import StateGraph, START, END
   from langgraph.checkpoint.memory import MemorySaver
   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate
   import json
   import uuid

   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

   def log(stage: str, message: str) -> dict:
       """生成日志"""
       timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
       return {
           "stage": stage,
           "updated_at": timestamp,
           "logs": [f"[{timestamp}] [{stage}] {message}"]
       }

   # ===== 话题发现节点 =====

   def discover_topics(state: ContentWorkflowState) -> dict:
       """发现热门话题"""
       topic = state["topic"]
       
       response = llm.invoke(f"""
   分析话题领域「{topic}」，找出5个当前热门的具体话题。

   输出JSON格式：
   [{{"topic": "话题名", "heat": 85, "reason": "热门原因"}}]
   """)
       
       try:
           topics = json.loads(response.content)
       except:
           topics = [{"topic": topic, "heat": 80, "reason": "用户指定"}]
       
       return {
           **log(WorkflowStage.DISCOVERY, f"发现 {len(topics)} 个热门话题"),
           "trending_topics": topics,
           "selected_topic": topics[0] if topics else None
       }

   # ===== 内容策划节点 =====

   def plan_content(state: ContentWorkflowState) -> dict:
       """策划内容"""
       topic = state["selected_topic"]["topic"] if state["selected_topic"] else state["topic"]
       platforms = state["target_platforms"]
       style = state["style"]
       
       response = llm.invoke(f"""
   为话题「{topic}」策划内容。
   目标平台：{', '.join(platforms)}
   风格：{style}

   输出JSON格式：
   {{
       "angle": "内容角度",
       "title": "推荐标题",
       "outline": {{
           "hook": "开头钩子",
           "sections": [{{"heading": "章节标题", "points": ["要点1", "要点2"]}}],
           "cta": "行动号召"
       }}
   }}
   """)
       
       try:
           plan = json.loads(response.content)
       except:
           plan = {"angle": "综合介绍", "title": topic, "outline": {}}
       
       return {
           **log(WorkflowStage.PLANNING, f"内容策划完成: {plan.get('title', 'N/A')}"),
           "content_angle": plan.get("angle", ""),
           "selected_title": plan.get("title", topic),
           "outline": plan.get("outline", {})
       }

   # ===== 内容创作节点 =====

   def write_content(state: ContentWorkflowState) -> dict:
       """撰写内容"""
       title = state["selected_title"]
       outline = state["outline"]
       style = state["style"]
       feedback = state.get("human_feedback", "")
       
       feedback_instruction = f"\n根据反馈进行修改：{feedback}" if feedback else ""
       
       response = llm.invoke(f"""
   撰写文章。
   
   标题：{title}
   大纲：{json.dumps(outline, ensure_ascii=False)}
   风格：{style}
   {feedback_instruction}

   要求：
   1. 开头要有吸引力
   2. 内容有深度和价值
   3. 结构清晰
   4. 字数1000-2000字
   """)
       
       content = response.content
       word_count = len(content)
       
       return {
           **log(WorkflowStage.WRITING, f"内容撰写完成，字数: {word_count}"),
           "draft_content": content,
           "word_count": word_count
       }

   # ===== AI 审核节点 =====

   def ai_review(state: ContentWorkflowState) -> dict:
       """AI 质量审核"""
       content = state["draft_content"]
       
       response = llm.invoke(f"""
   审核以下内容的质量。

   内容：
   {content[:2000]}

   评估维度（每项0-25分，总分100）：
   1. 内容价值
   2. 结构清晰度
   3. 语言表达
   4. 平台适配度

   输出JSON格式：
   {{
       "total_score": 85,
       "dimensions": {{"value": 22, "structure": 20, "language": 21, "adaptation": 22}},
       "issues": ["问题1", "问题2"],
       "passed": true
   }}
   """)
       
       try:
           review = json.loads(response.content)
       except:
           review = {"total_score": 75, "issues": [], "passed": True}
       
       return {
           **log(WorkflowStage.AI_REVIEW, f"AI审核完成，分数: {review.get('total_score', 'N/A')}"),
           "ai_review_result": review,
           "quality_score": review.get("total_score", 75),
           "issues": review.get("issues", [])
       }

   # ===== 人工审核节点 =====

   def human_review_node(state: ContentWorkflowState) -> dict:
       """人工审核（会在此处中断）"""
       human_review = state.get("human_review")
       
       if human_review:
           approved = human_review.get("approved", False)
           feedback = human_review.get("feedback", "")
           
           return {
               **log(WorkflowStage.HUMAN_REVIEW, f"人工审核: {'通过' if approved else '需修改'}"),
               "human_approved": approved,
               "human_feedback": feedback
           }
       
       return {
           **log(WorkflowStage.HUMAN_REVIEW, "等待人工审核..."),
       }

   # ===== 平台适配节点 =====

   def adapt_content(state: ContentWorkflowState) -> dict:
       """适配到各平台"""
       content = state["draft_content"]
       title = state["selected_title"]
       platforms = state["target_platforms"]
       
       adapted = {}
       
       for platform in platforms:
           response = llm.invoke(f"""
   将以下内容适配到「{platform}」平台。

   原标题：{title}
   原内容：{content[:1500]}

   根据平台特点调整：
   - 标题风格
   - 内容长度
   - 语言风格
   - 格式要求

   输出JSON：
   {{"title": "适配后标题", "content": "适配后内容", "tags": ["标签1", "标签2"]}}
   """)
           
           try:
               adapted[platform] = json.loads(response.content)
           except:
               adapted[platform] = {"title": title, "content": content, "tags": []}
       
       return {
           **log(WorkflowStage.ADAPTING, f"已适配 {len(adapted)} 个平台"),
           "adapted_contents": adapted
       }

   # ===== 发布节点 =====

   def publish_content(state: ContentWorkflowState) -> dict:
       """发布到各平台"""
       adapted = state["adapted_contents"]
       results = []
       
       for platform, content in adapted.items():
           # 模拟发布
           result = {
               "platform": platform,
               "success": True,
               "post_id": str(uuid.uuid4())[:8],
               "url": f"https://{platform.lower().replace(' ', '')}.com/p/xxx",
               "published_at": datetime.now().isoformat()
           }
           results.append(result)
       
       success_count = sum(1 for r in results if r["success"])
       
       return {
           **log(WorkflowStage.PUBLISHING, f"发布完成: {success_count}/{len(results)} 成功"),
           "publish_results": results
       }

   # ===== 数据追踪节点 =====

   def track_performance(state: ContentWorkflowState) -> dict:
       """追踪发布效果"""
       results = state["publish_results"]
       
       # 模拟获取数据
       for result in results:
           result["stats"] = {
               "views": 0,
               "likes": 0,
               "comments": 0,
               "shares": 0
           }
       
       return {
           **log(WorkflowStage.TRACKING, "数据追踪已启动"),
           "publish_results": results,
           "stage": WorkflowStage.COMPLETED
       }

路由函数
--------

.. code-block:: python

   def ai_review_router(state: ContentWorkflowState) -> Literal["human_review", "rewrite"]:
       """AI审核后的路由"""
       score = state.get("quality_score", 0)
       if score >= 70:
           return "human_review"
       return "rewrite"

   def human_review_router(state: ContentWorkflowState) -> Literal["adapt", "rewrite", "wait"]:
       """人工审核后的路由"""
       human_review = state.get("human_review")
       
       if not human_review:
           return "wait"
       
       if human_review.get("approved"):
           return "adapt"
       return "rewrite"

构建完整工作流
--------------

.. code-block:: python

   def create_content_workflow():
       graph = StateGraph(ContentWorkflowState)
       
       # 添加所有节点
       graph.add_node("discover", discover_topics)
       graph.add_node("plan", plan_content)
       graph.add_node("write", write_content)
       graph.add_node("ai_review", ai_review)
       graph.add_node("human_review", human_review_node)
       graph.add_node("adapt", adapt_content)
       graph.add_node("publish", publish_content)
       graph.add_node("track", track_performance)
       
       # 主流程
       graph.add_edge(START, "discover")
       graph.add_edge("discover", "plan")
       graph.add_edge("plan", "write")
       graph.add_edge("write", "ai_review")
       
       # AI 审核后路由
       graph.add_conditional_edges(
           "ai_review",
           ai_review_router,
           {
               "human_review": "human_review",
               "rewrite": "write"
           }
       )
       
       # 人工审核后路由
       graph.add_conditional_edges(
           "human_review",
           human_review_router,
           {
               "adapt": "adapt",
               "rewrite": "write",
               "wait": END  # 等待人工输入
           }
       )
       
       # 发布流程
       graph.add_edge("adapt", "publish")
       graph.add_edge("publish", "track")
       graph.add_edge("track", END)
       
       # 编译（带检查点和人工审核中断点）
       memory = MemorySaver()
       return graph.compile(
           checkpointer=memory,
           interrupt_before=["human_review"]
       )

使用示例
--------

.. code-block:: python

   def run_workflow_demo():
       workflow = create_content_workflow()
       
       # 初始状态
       initial_state = {
           "task_id": str(uuid.uuid4())[:8],
           "topic": "AI编程",
           "target_platforms": ["微信公众号", "知乎"],
           "style": "专业但易懂",
           "requirements": None,
           "stage": WorkflowStage.INIT,
           "iteration": 0,
           "max_iterations": 3,
           "created_at": datetime.now().isoformat(),
           "updated_at": datetime.now().isoformat(),
           "logs": [],
           "errors": []
       }
       
       config = {"configurable": {"thread_id": initial_state["task_id"]}}
       
       # 第一阶段：运行到人工审核
       print("=" * 60)
       print("阶段一：自动化流程")
       print("=" * 60)
       
       result = workflow.invoke(initial_state, config)
       
       # 打印日志
       for log_entry in result.get("logs", []):
           print(log_entry)
       
       print(f"\n当前阶段: {result.get('stage')}")
       print(f"AI审核分数: {result.get('quality_score')}")
       print(f"内容预览: {result.get('draft_content', '')[:200]}...")
       
       # 第二阶段：人工审核
       print("\n" + "=" * 60)
       print("阶段二：人工审核")
       print("=" * 60)
       
       # 模拟人工审核
       print("请审核内容...")
       approval = input("批准? (y/n): ").strip().lower() == 'y'
       feedback = ""
       if not approval:
           feedback = input("修改建议: ").strip()
       
       # 注入人工审核结果
       workflow.update_state(
           config,
           {
               "human_review": {
                   "approved": approval,
                   "feedback": feedback,
                   "reviewer": "human",
                   "timestamp": datetime.now().isoformat()
               }
           }
       )
       
       # 继续执行
       final_result = workflow.invoke(None, config)
       
       # 打印最终结果
       print("\n" + "=" * 60)
       print("最终结果")
       print("=" * 60)
       
       for log_entry in final_result.get("logs", [])[-5:]:
           print(log_entry)
       
       print(f"\n最终阶段: {final_result.get('stage')}")
       
       if final_result.get("publish_results"):
           print("\n发布结果:")
           for pr in final_result["publish_results"]:
               print(f"  - {pr['platform']}: {pr['url']}")
       
       return final_result

   if __name__ == "__main__":
       run_workflow_demo()

工作流可视化
============

.. code-block:: text

   graph TD
       START --> discover[话题发现]
       discover --> plan[内容策划]
       plan --> write[内容创作]
       write --> ai_review[AI审核]
       
       ai_review -->|分数>=70| human_review[人工审核]
       ai_review -->|分数<70| write
       
       human_review -->|批准| adapt[平台适配]
       human_review -->|拒绝| write
       human_review -->|等待| END_WAIT[等待输入]
       
       adapt --> publish[多平台发布]
       publish --> track[数据追踪]
       track --> END

下一步
======

在最后一个教程中，我们将学习如何将工作流部署到生产环境。

:doc:`tutorial_10_production`
