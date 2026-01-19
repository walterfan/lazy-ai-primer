####################################
Tutorial 2: State 与 Graph 基础
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

理解 State（状态）
==================

State 是 LangGraph 中最重要的概念。它是在节点之间传递的数据结构。

定义 State
----------

使用 TypedDict 定义状态结构：

.. code-block:: python

   from typing import TypedDict, List, Optional

   class ContentState(TypedDict):
       # 输入
       topic: str
       target_platform: str
       
       # 中间状态
       research_result: Optional[str]
       outline: Optional[str]
       
       # 输出
       title: str
       content: str
       tags: List[str]

使用 Annotated 定义状态更新方式
-------------------------------

.. code-block:: python

   from typing import Annotated
   from operator import add

   class MessageState(TypedDict):
       # 消息列表，使用 add 操作符追加新消息
       messages: Annotated[list, add]
       
       # 普通字段，直接覆盖
       current_step: str

   # 当节点返回 {"messages": [new_msg]} 时
   # 新消息会追加到列表，而不是替换

状态更新机制
============

覆盖更新（默认）
----------------

.. code-block:: python

   class State(TypedDict):
       value: str

   def node_a(state: State) -> dict:
       return {"value": "new_value"}  # 直接覆盖

   # state["value"] = "new_value"

追加更新（使用 Annotated）
--------------------------

.. code-block:: python

   from typing import Annotated
   from operator import add

   class State(TypedDict):
       items: Annotated[list, add]

   def node_a(state: State) -> dict:
       return {"items": ["new_item"]}  # 追加到列表

   # state["items"] = state["items"] + ["new_item"]

自定义 Reducer
--------------

.. code-block:: python

   def merge_dicts(existing: dict, new: dict) -> dict:
       """合并字典而不是覆盖"""
       return {**existing, **new}

   class State(TypedDict):
       metadata: Annotated[dict, merge_dicts]

   def node_a(state: State) -> dict:
       return {"metadata": {"key1": "value1"}}

   def node_b(state: State) -> dict:
       return {"metadata": {"key2": "value2"}}

   # 最终 state["metadata"] = {"key1": "value1", "key2": "value2"}

StateGraph 详解
===============

创建 StateGraph
---------------

.. code-block:: python

   from langgraph.graph import StateGraph, START, END

   # 创建图
   graph = StateGraph(ContentState)

   # 添加节点
   graph.add_node("research", research_node)
   graph.add_node("outline", outline_node)
   graph.add_node("write", write_node)

   # 设置入口点
   graph.add_edge(START, "research")

   # 添加边
   graph.add_edge("research", "outline")
   graph.add_edge("outline", "write")

   # 设置出口点
   graph.add_edge("write", END)

   # 编译
   app = graph.compile()

图的编译选项
------------

.. code-block:: python

   # 基本编译
   app = graph.compile()

   # 带中断点的编译（用于人工干预）
   app = graph.compile(interrupt_before=["write"])

   # 带检查点的编译（用于持久化）
   from langgraph.checkpoint.memory import MemorySaver
   
   memory = MemorySaver()
   app = graph.compile(checkpointer=memory)

实战：自媒体内容状态设计
========================

.. code-block:: python

   from typing import TypedDict, List, Optional, Annotated
   from operator import add
   from enum import Enum

   class WorkflowStatus(str, Enum):
       RESEARCHING = "researching"
       PLANNING = "planning"
       WRITING = "writing"
       REVIEWING = "reviewing"
       PUBLISHING = "publishing"
       COMPLETED = "completed"
       FAILED = "failed"

   class ContentState(TypedDict):
       """自媒体内容工作流状态"""
       
       # ===== 输入参数 =====
       topic: str                          # 内容主题
       target_platforms: List[str]         # 目标平台列表
       style: str                          # 写作风格
       
       # ===== 工作流状态 =====
       status: WorkflowStatus              # 当前状态
       current_step: str                   # 当前步骤
       error_message: Optional[str]        # 错误信息
       
       # ===== 研究阶段 =====
       hot_topics: List[dict]              # 热门话题
       topic_analysis: Optional[str]       # 话题分析
       competitors: List[dict]             # 竞品分析
       
       # ===== 策划阶段 =====
       content_angle: str                  # 内容角度
       outline: Optional[dict]             # 文章大纲
       title_candidates: List[str]         # 标题候选
       selected_title: str                 # 选定标题
       
       # ===== 创作阶段 =====
       draft_content: str                  # 草稿内容
       final_content: str                  # 最终内容
       summary: str                        # 摘要
       tags: List[str]                     # 标签
       
       # ===== 发布阶段 =====
       adapted_contents: dict              # 各平台适配内容
       publish_results: List[dict]         # 发布结果
       
       # ===== 追踪数据 =====
       messages: Annotated[List[str], add] # 日志消息（追加模式）

完整示例：内容策划工作流
========================

.. code-block:: python

   from typing import TypedDict, List, Optional
   from langgraph.graph import StateGraph, START, END
   from langchain_openai import ChatOpenAI

   # 1. 定义状态
   class PlanningState(TypedDict):
       topic: str
       platform: str
       hot_topics: List[str]
       selected_angle: str
       outline: dict
       title: str

   # 2. 定义节点
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

   def research_topics(state: PlanningState) -> dict:
       """研究热门话题"""
       topic = state["topic"]
       response = llm.invoke(
           f"列出与「{topic}」相关的5个热门子话题，每行一个"
       )
       topics = response.content.strip().split("\n")
       return {"hot_topics": topics}

   def select_angle(state: PlanningState) -> dict:
       """选择内容角度"""
       topic = state["topic"]
       hot_topics = state["hot_topics"]
       platform = state["platform"]
       
       response = llm.invoke(f"""
   主题：{topic}
   热门子话题：{hot_topics}
   目标平台：{platform}

   请选择一个最适合的内容角度，并说明理由。只输出角度名称。
   """)
       return {"selected_angle": response.content.strip()}

   def create_outline(state: PlanningState) -> dict:
       """创建大纲"""
       topic = state["topic"]
       angle = state["selected_angle"]
       
       response = llm.invoke(f"""
   主题：{topic}
   角度：{angle}

   请创建文章大纲，包含：
   1. 引言
   2. 3-4个主体章节
   3. 结论

   输出JSON格式：{{"sections": [{{"title": "章节标题", "points": ["要点1", "要点2"]}}]}}
   """)
       
       import json
       try:
           outline = json.loads(response.content)
       except:
           outline = {"sections": [{"title": "主体内容", "points": ["要点"]}]}
       
       return {"outline": outline}

   def generate_title(state: PlanningState) -> dict:
       """生成标题"""
       topic = state["topic"]
       angle = state["selected_angle"]
       platform = state["platform"]
       
       response = llm.invoke(f"""
   主题：{topic}
   角度：{angle}
   平台：{platform}

   请生成一个适合该平台的吸引人标题，只输出标题。
   """)
       return {"title": response.content.strip()}

   # 3. 构建图
   graph = StateGraph(PlanningState)

   graph.add_node("research", research_topics)
   graph.add_node("select_angle", select_angle)
   graph.add_node("outline", create_outline)
   graph.add_node("title", generate_title)

   graph.add_edge(START, "research")
   graph.add_edge("research", "select_angle")
   graph.add_edge("select_angle", "outline")
   graph.add_edge("outline", "title")
   graph.add_edge("title", END)

   # 4. 编译并运行
   app = graph.compile()

   result = app.invoke({
       "topic": "AI编程",
       "platform": "微信公众号"
   })

   print("=== 内容策划结果 ===")
   print(f"主题: {result['topic']}")
   print(f"热门话题: {result['hot_topics']}")
   print(f"选定角度: {result['selected_angle']}")
   print(f"大纲: {result['outline']}")
   print(f"标题: {result['title']}")

流式执行
========

.. code-block:: python

   # 流式获取每个节点的输出
   for event in app.stream({"topic": "AI编程", "platform": "知乎"}):
       for node_name, output in event.items():
           print(f"[{node_name}] 完成")
           print(f"  输出: {output}")
           print()

下一步
======

在下一个教程中，我们将深入学习节点和边的高级用法。

:doc:`tutorial_03_nodes_edges`
