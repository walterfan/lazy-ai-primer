####################################
Tutorial 1: LangGraph 入门
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是 LangGraph？
==================

LangGraph 是一个用于构建有状态、多步骤 AI 应用的框架。它的核心理念是：

**将 Agent 工作流建模为图（Graph）**

- **节点（Node）**: 执行具体操作的函数
- **边（Edge）**: 连接节点，定义执行顺序
- **状态（State）**: 在节点之间传递的数据

LangGraph vs LangChain AgentExecutor
====================================

.. csv-table::
   :header: "特性", "AgentExecutor", "LangGraph"
   :widths: 25, 35, 40

   "流程控制", "隐式，由 LLM 决定", "显式，由图结构定义"
   "状态管理", "有限", "完整的状态机"
   "循环控制", "有限", "完全支持"
   "人工干预", "困难", "原生支持"
   "多 Agent", "需要额外封装", "原生支持"
   "调试", "困难", "可视化图结构"
   "持久化", "需要额外实现", "内置支持"

安装配置
========

.. code-block:: bash

   pip install langgraph langchain-openai

核心概念
========

1. StateGraph（状态图）
-----------------------

StateGraph 是 LangGraph 的核心类，用于定义工作流：

.. code-block:: python

   from langgraph.graph import StateGraph

   # 定义状态类型
   from typing import TypedDict

   class MyState(TypedDict):
       messages: list
       current_step: str

   # 创建图
   graph = StateGraph(MyState)

2. Node（节点）
---------------

节点是执行具体操作的函数：

.. code-block:: python

   def my_node(state: MyState) -> dict:
       # 处理状态
       return {"current_step": "completed"}

   # 添加节点
   graph.add_node("my_node", my_node)

3. Edge（边）
-------------

边定义节点之间的连接：

.. code-block:: python

   # 普通边：A -> B
   graph.add_edge("node_a", "node_b")

   # 条件边：根据条件选择下一个节点
   graph.add_conditional_edges(
       "node_a",
       routing_function,
       {"option1": "node_b", "option2": "node_c"}
   )

第一个 LangGraph 程序
=====================

.. code-block:: python

   from typing import TypedDict, Annotated
   from langgraph.graph import StateGraph, START, END
   from langchain_openai import ChatOpenAI

   # 1. 定义状态
   class State(TypedDict):
       topic: str
       title: str
       outline: str

   # 2. 定义节点函数
   llm = ChatOpenAI(model="gpt-4o-mini")

   def generate_title(state: State) -> dict:
       """生成标题"""
       topic = state["topic"]
       response = llm.invoke(f"为主题「{topic}」生成一个吸引人的标题，只输出标题")
       return {"title": response.content}

   def generate_outline(state: State) -> dict:
       """生成大纲"""
       title = state["title"]
       response = llm.invoke(f"为标题「{title}」生成文章大纲，3-5个章节")
       return {"outline": response.content}

   # 3. 构建图
   graph = StateGraph(State)

   # 添加节点
   graph.add_node("generate_title", generate_title)
   graph.add_node("generate_outline", generate_outline)

   # 添加边
   graph.add_edge(START, "generate_title")
   graph.add_edge("generate_title", "generate_outline")
   graph.add_edge("generate_outline", END)

   # 4. 编译图
   app = graph.compile()

   # 5. 运行
   result = app.invoke({"topic": "AI编程入门"})

   print(f"主题: {result['topic']}")
   print(f"标题: {result['title']}")
   print(f"大纲:\n{result['outline']}")

图的可视化
==========

.. code-block:: python

   # 生成 Mermaid 图
   print(app.get_graph().draw_mermaid())

输出：

.. code-block:: text

   graph TD
       __start__ --> generate_title
       generate_title --> generate_outline
       generate_outline --> __end__

LangGraph 执行流程
==================

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    LangGraph 执行流程                        │
   │                                                              │
   │  初始状态 ──► [START] ──► [Node 1] ──► [Node 2] ──► [END]   │
   │     │            │           │            │           │      │
   │     │            │           │            │           │      │
   │     └────────────┴───────────┴────────────┴───────────┘      │
   │                         状态传递                              │
   │                                                              │
   │  每个节点：                                                   │
   │  1. 接收当前状态                                             │
   │  2. 执行操作                                                 │
   │  3. 返回状态更新                                             │
   │  4. 状态合并后传递给下一个节点                               │
   │                                                              │
   └─────────────────────────────────────────────────────────────┘

自媒体项目预览
==============

在接下来的教程中，我们将逐步构建一个完整的自媒体内容工作流：

.. code-block:: python

   # 最终我们将实现这样的工作流
   from selfmedia_workflow import create_content_workflow

   workflow = create_content_workflow()

   # 执行完整流程
   result = workflow.invoke({
       "topic": "AI Agent 开发",
       "target_platforms": ["微信公众号", "知乎"],
       "style": "专业但易懂"
   })

   # 结果包含
   # - 话题分析
   # - 内容大纲
   # - 完整文章
   # - 多平台适配版本
   # - 发布结果

下一步
======

在下一个教程中，我们将深入学习 State 和 Graph 的基础知识。

:doc:`tutorial_02_state_graph`
