####################################
Tutorial 8: 多 Agent 协作
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

多 Agent 系统概述
=================

多 Agent 系统让多个专业化的 Agent 协同工作，各司其职：

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    Multi-Agent System                        │
   │                                                              │
   │  ┌──────────────────────────────────────────────────────┐   │
   │  │                   Supervisor Agent                    │   │
   │  │              (协调者：分配任务、汇总结果)               │   │
   │  └──────────────────────────┬───────────────────────────┘   │
   │                             │                                │
   │         ┌───────────────────┼───────────────────┐           │
   │         │                   │                   │           │
   │         ▼                   ▼                   ▼           │
   │  ┌────────────┐     ┌────────────┐     ┌────────────┐      │
   │  │ Researcher │     │   Writer   │     │  Reviewer  │      │
   │  │  研究员    │     │   写手     │     │   审核员   │      │
   │  └────────────┘     └────────────┘     └────────────┘      │
   │                                                              │
   └─────────────────────────────────────────────────────────────┘

多 Agent 模式
=============

1. Supervisor 模式
------------------

一个主管 Agent 协调多个工作 Agent：

.. code-block:: python

   from typing import TypedDict, Literal, List
   from langgraph.graph import StateGraph, START, END
   from langchain_openai import ChatOpenAI

   class TeamState(TypedDict):
       task: str
       current_agent: str
       results: dict
       final_output: str

   # Supervisor 决定下一个执行的 Agent
   def supervisor(state: TeamState) -> dict:
       llm = ChatOpenAI(model="gpt-4o-mini")
       
       response = llm.invoke(f"""
   任务: {state['task']}
   已完成: {list(state['results'].keys())}

   决定下一步由谁执行：
   - researcher: 需要研究信息
   - writer: 需要写内容
   - reviewer: 需要审核
   - FINISH: 任务完成

   只输出一个词。
   """)
       
       return {"current_agent": response.content.strip().lower()}

   def route_to_agent(state: TeamState) -> str:
       return state["current_agent"]

2. 协作模式
-----------

Agent 之间直接传递工作：

.. code-block:: python

   # Agent A 完成后传给 Agent B
   graph.add_edge("agent_a", "agent_b")
   graph.add_edge("agent_b", "agent_c")

3. 竞争模式
-----------

多个 Agent 并行工作，选择最佳结果：

.. code-block:: python

   # 并行执行
   graph.add_edge(START, "agent_1")
   graph.add_edge(START, "agent_2")
   graph.add_edge(START, "agent_3")

   # 汇总选择
   graph.add_edge("agent_1", "selector")
   graph.add_edge("agent_2", "selector")
   graph.add_edge("agent_3", "selector")

实战：自媒体内容团队
====================

.. code-block:: python

   from typing import TypedDict, List, Literal, Optional, Annotated
   from operator import add
   from langgraph.graph import StateGraph, START, END
   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate

   # ========== 状态定义 ==========

   class TeamState(TypedDict):
       # 任务信息
       topic: str
       platform: str
       requirements: str
       
       # Agent 输出
       research_output: Optional[str]
       outline_output: Optional[dict]
       draft_output: Optional[str]
       review_output: Optional[dict]
       final_output: Optional[str]
       
       # 协调状态
       next_agent: str
       iteration: int
       max_iterations: int
       
       # 通信日志
       messages: Annotated[List[str], add]

   # ========== Agent 定义 ==========

   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

   class ResearcherAgent:
       """研究员 Agent：负责话题研究和资料收集"""
       
       name = "researcher"
       
       @staticmethod
       def run(state: TeamState) -> dict:
           topic = state["topic"]
           platform = state["platform"]
           
           prompt = ChatPromptTemplate.from_template("""
   你是专业的内容研究员。

   任务：研究话题「{topic}」
   目标平台：{platform}

   请提供：
   1. 话题背景和重要性
   2. 目标受众分析
   3. 竞品内容分析
   4. 推荐的内容角度
   5. 关键数据和案例

   输出详细的研究报告。
   """)
           
           chain = prompt | llm
           response = chain.invoke({"topic": topic, "platform": platform})
           
           return {
               "research_output": response.content,
               "messages": [f"[Researcher] 完成话题研究: {topic}"]
           }

   class PlannerAgent:
       """策划 Agent：负责内容规划和大纲设计"""
       
       name = "planner"
       
       @staticmethod
       def run(state: TeamState) -> dict:
           research = state["research_output"]
           platform = state["platform"]
           
           prompt = ChatPromptTemplate.from_template("""
   你是资深的内容策划师。

   研究报告：
   {research}

   目标平台：{platform}

   请创建详细的内容大纲：
   1. 标题（吸引人、适合平台）
   2. 开头钩子
   3. 主体章节（3-5个）
   4. 结尾和CTA

   输出JSON格式：
   {{"title": "标题", "hook": "钩子", "sections": [{{"heading": "章节", "points": ["要点"]}}], "cta": "行动号召"}}
   """)
           
           chain = prompt | llm
           response = chain.invoke({"research": research, "platform": platform})
           
           import json
           try:
               outline = json.loads(response.content)
           except:
               outline = {"title": state["topic"], "sections": []}
           
           return {
               "outline_output": outline,
               "messages": [f"[Planner] 完成内容大纲: {outline.get('title', 'N/A')}"]
           }

   class WriterAgent:
       """写手 Agent：负责内容创作"""
       
       name = "writer"
       
       @staticmethod
       def run(state: TeamState) -> dict:
           outline = state["outline_output"]
           research = state["research_output"]
           platform = state["platform"]
           
           prompt = ChatPromptTemplate.from_template("""
   你是专业的内容写手。

   大纲：
   {outline}

   研究资料：
   {research}

   目标平台：{platform}

   请根据大纲撰写完整的文章：
   - 开头要有吸引力
   - 内容要有深度和价值
   - 语言风格适合平台
   - 结尾要有互动引导
   """)
           
           chain = prompt | llm
           response = chain.invoke({
               "outline": str(outline),
               "research": research[:1000],  # 限制长度
               "platform": platform
           })
           
           return {
               "draft_output": response.content,
               "messages": [f"[Writer] 完成文章草稿"]
           }

   class ReviewerAgent:
       """审核 Agent：负责内容审核和优化建议"""
       
       name = "reviewer"
       
       @staticmethod
       def run(state: TeamState) -> dict:
           draft = state["draft_output"]
           platform = state["platform"]
           
           prompt = ChatPromptTemplate.from_template("""
   你是严格的内容审核专家。

   文章内容：
   {draft}

   目标平台：{platform}

   请审核并评分（0-100）：
   1. 内容质量
   2. 结构清晰度
   3. 平台适配度
   4. 语言表达

   输出格式：
   总分：[数字]
   通过：[是/否]
   问题：[具体问题]
   建议：[改进建议]
   """)
           
           chain = prompt | llm
           response = chain.invoke({"draft": draft, "platform": platform})
           
           result = response.content
           passed = "是" in result and "通过：是" in result or int(''.join(filter(str.isdigit, result.split('\n')[0])) or 0) >= 80
           
           return {
               "review_output": {
                   "result": result,
                   "passed": passed
               },
               "messages": [f"[Reviewer] 审核完成: {'通过' if passed else '需修改'}"]
           }

   class EditorAgent:
       """编辑 Agent：负责最终润色"""
       
       name = "editor"
       
       @staticmethod
       def run(state: TeamState) -> dict:
           draft = state["draft_output"]
           review = state["review_output"]
           
           suggestions = review.get("result", "") if review else ""
           
           prompt = ChatPromptTemplate.from_template("""
   你是资深编辑。

   原稿：
   {draft}

   审核意见：
   {suggestions}

   请进行最终润色：
   1. 修正问题
   2. 优化表达
   3. 确保质量

   输出最终版本。
   """)
           
           chain = prompt | llm
           response = chain.invoke({"draft": draft, "suggestions": suggestions})
           
           return {
               "final_output": response.content,
               "messages": [f"[Editor] 完成最终润色"]
           }

   # ========== Supervisor Agent ==========

   class SupervisorAgent:
       """主管 Agent：协调团队工作"""
       
       @staticmethod
       def decide_next(state: TeamState) -> dict:
           """决定下一个执行的 Agent"""
           
           # 检查当前进度
           has_research = state.get("research_output") is not None
           has_outline = state.get("outline_output") is not None
           has_draft = state.get("draft_output") is not None
           has_review = state.get("review_output") is not None
           iteration = state.get("iteration", 0)
           max_iter = state.get("max_iterations", 3)
           
           # 决策逻辑
           if not has_research:
               next_agent = "researcher"
           elif not has_outline:
               next_agent = "planner"
           elif not has_draft:
               next_agent = "writer"
           elif not has_review:
               next_agent = "reviewer"
           elif has_review:
               review = state["review_output"]
               if review.get("passed") or iteration >= max_iter:
                   next_agent = "editor"
               else:
                   # 需要修改，回到写手
                   next_agent = "writer"
           else:
               next_agent = "FINISH"
           
           return {
               "next_agent": next_agent,
               "iteration": iteration + 1,
               "messages": [f"[Supervisor] 下一步: {next_agent}"]
           }

   # ========== 路由函数 ==========

   def route_to_agent(state: TeamState) -> str:
       """路由到对应的 Agent"""
       next_agent = state["next_agent"]
       
       if next_agent == "FINISH":
           return "end"
       return next_agent

   # ========== 构建图 ==========

   def create_content_team():
       graph = StateGraph(TeamState)
       
       # 添加 Agent 节点
       graph.add_node("supervisor", SupervisorAgent.decide_next)
       graph.add_node("researcher", ResearcherAgent.run)
       graph.add_node("planner", PlannerAgent.run)
       graph.add_node("writer", WriterAgent.run)
       graph.add_node("reviewer", ReviewerAgent.run)
       graph.add_node("editor", EditorAgent.run)
       
       # 流程：从 Supervisor 开始
       graph.add_edge(START, "supervisor")
       
       # Supervisor 路由到各 Agent
       graph.add_conditional_edges(
           "supervisor",
           route_to_agent,
           {
               "researcher": "researcher",
               "planner": "planner",
               "writer": "writer",
               "reviewer": "reviewer",
               "editor": "editor",
               "end": END
           }
       )
       
       # 各 Agent 完成后回到 Supervisor
       graph.add_edge("researcher", "supervisor")
       graph.add_edge("planner", "supervisor")
       graph.add_edge("writer", "supervisor")
       graph.add_edge("reviewer", "supervisor")
       graph.add_edge("editor", END)
       
       return graph.compile()

   # ========== 运行 ==========

   def run_content_team():
       team = create_content_team()
       
       initial_state = {
           "topic": "AI Agent 开发入门",
           "platform": "微信公众号",
           "requirements": "深入浅出，适合初学者",
           "iteration": 0,
           "max_iterations": 2,
           "messages": []
       }
       
       print("=" * 60)
       print("自媒体内容团队开始工作")
       print("=" * 60)
       
       # 流式执行，观察每个 Agent 的工作
       for event in team.stream(initial_state):
           for node, output in event.items():
               if "messages" in output and output["messages"]:
                   for msg in output["messages"]:
                       print(msg)
       
       # 获取最终结果
       result = team.invoke(initial_state)
       
       print("\n" + "=" * 60)
       print("最终输出")
       print("=" * 60)
       print(f"\n标题: {result['outline_output'].get('title', 'N/A')}")
       print(f"\n内容预览:\n{result['final_output'][:500]}...")
       
       return result

   if __name__ == "__main__":
       run_content_team()

Agent 通信模式
==============

共享状态通信
------------

.. code-block:: python

   class State(TypedDict):
       # 共享的通信通道
       shared_memory: dict
       agent_outputs: dict

   def agent_a(state):
       # 写入共享内存
       return {"shared_memory": {"agent_a_data": "..."}}

   def agent_b(state):
       # 读取 Agent A 的数据
       a_data = state["shared_memory"].get("agent_a_data")
       # 处理...

消息传递通信
------------

.. code-block:: python

   from typing import Annotated
   from operator import add

   class State(TypedDict):
       messages: Annotated[List[dict], add]

   def agent_a(state):
       return {
           "messages": [{
               "from": "agent_a",
               "to": "agent_b",
               "content": "请处理这个任务",
               "data": {...}
           }]
       }

   def agent_b(state):
       # 找到发给自己的消息
       my_messages = [m for m in state["messages"] if m["to"] == "agent_b"]
       # 处理...

最佳实践
========

1. **明确职责分工**

.. code-block:: python

   # ✅ 好：每个 Agent 职责清晰
   ResearcherAgent  # 只负责研究
   WriterAgent      # 只负责写作
   ReviewerAgent    # 只负责审核

   # ❌ 差：职责混乱
   DoEverythingAgent  # 什么都做

2. **设计清晰的通信协议**

.. code-block:: python

   class AgentMessage(TypedDict):
       sender: str
       receiver: str
       type: str  # request, response, notification
       content: dict
       timestamp: str

3. **处理 Agent 失败**

.. code-block:: python

   def safe_agent_run(agent_func, state, max_retries=3):
       for i in range(max_retries):
           try:
               return agent_func(state)
           except Exception as e:
               if i == max_retries - 1:
                   return {"error": str(e), "status": "failed"}
               time.sleep(1)

下一步
======

在下一个教程中，我们将构建完整的自媒体内容工作流。

:doc:`tutorial_09_content_workflow`
