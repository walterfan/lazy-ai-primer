####################################
Tutorial 3: Nodes 与 Edges
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

Node（节点）详解
================

节点是 LangGraph 中执行具体操作的单元。每个节点是一个函数，接收状态并返回状态更新。

基本节点定义
------------

.. code-block:: python

   from typing import TypedDict

   class State(TypedDict):
       input: str
       output: str

   def my_node(state: State) -> dict:
       """
       节点函数
       - 参数：当前状态
       - 返回：状态更新字典
       """
       input_text = state["input"]
       result = process(input_text)
       return {"output": result}

节点类型
--------

1. **普通函数节点**

.. code-block:: python

   def simple_node(state: State) -> dict:
       return {"field": "value"}

2. **异步节点**

.. code-block:: python

   async def async_node(state: State) -> dict:
       result = await async_operation()
       return {"field": result}

3. **类方法节点**

.. code-block:: python

   class ContentProcessor:
       def __init__(self, llm):
           self.llm = llm
       
       def process(self, state: State) -> dict:
           response = self.llm.invoke(state["input"])
           return {"output": response.content}

   processor = ContentProcessor(llm)
   graph.add_node("process", processor.process)

4. **Runnable 节点**

.. code-block:: python

   from langchain_core.prompts import ChatPromptTemplate
   from langchain_openai import ChatOpenAI

   # 直接使用 LangChain Runnable 作为节点
   prompt = ChatPromptTemplate.from_template("总结：{input}")
   chain = prompt | ChatOpenAI(model="gpt-4o-mini")

   graph.add_node("summarize", chain)

添加节点到图
------------

.. code-block:: python

   from langgraph.graph import StateGraph

   graph = StateGraph(State)

   # 方式1：函数
   graph.add_node("node_name", my_function)

   # 方式2：Lambda
   graph.add_node("transform", lambda state: {"output": state["input"].upper()})

   # 方式3：Runnable
   graph.add_node("llm_call", prompt | llm)

Edge（边）详解
==============

边定义了节点之间的连接关系，决定了工作流的执行顺序。

普通边
------

.. code-block:: python

   from langgraph.graph import START, END

   # 从 START 到第一个节点
   graph.add_edge(START, "first_node")

   # 节点之间的连接
   graph.add_edge("first_node", "second_node")

   # 从最后一个节点到 END
   graph.add_edge("second_node", END)

条件边
------

.. code-block:: python

   def route_decision(state: State) -> str:
       """路由函数：返回下一个节点的名称"""
       if state["score"] > 80:
           return "high_quality"
       elif state["score"] > 60:
           return "medium_quality"
       else:
           return "low_quality"

   graph.add_conditional_edges(
       "evaluate",           # 源节点
       route_decision,       # 路由函数
       {
           "high_quality": "publish",
           "medium_quality": "review",
           "low_quality": "rewrite"
       }
   )

使用 END 作为条件目标
---------------------

.. code-block:: python

   def should_continue(state: State) -> str:
       if state["is_complete"]:
           return "end"
       return "continue"

   graph.add_conditional_edges(
       "check",
       should_continue,
       {
           "continue": "process",
           "end": END
       }
   )

实战：内容创作节点系统
======================

.. code-block:: python

   from typing import TypedDict, List, Optional
   from langgraph.graph import StateGraph, START, END
   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate

   # ========== 状态定义 ==========

   class ContentState(TypedDict):
       topic: str
       platform: str
       research: str
       outline: dict
       draft: str
       quality_score: int
       final_content: str
       feedback: Optional[str]

   # ========== 节点定义 ==========

   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

   class ContentNodes:
       """内容创作节点集合"""
       
       @staticmethod
       def research(state: ContentState) -> dict:
           """研究节点：分析话题"""
           topic = state["topic"]
           
           prompt = ChatPromptTemplate.from_template("""
   分析话题「{topic}」：
   1. 目标受众是谁？
   2. 核心痛点是什么？
   3. 有哪些独特角度？
   
   简洁回答。
   """)
           chain = prompt | llm
           response = chain.invoke({"topic": topic})
           
           return {"research": response.content}
       
       @staticmethod
       def create_outline(state: ContentState) -> dict:
           """大纲节点：生成文章结构"""
           topic = state["topic"]
           research = state["research"]
           
           prompt = ChatPromptTemplate.from_template("""
   基于以下研究，为「{topic}」创建文章大纲：

   研究结果：
   {research}

   输出JSON格式的大纲：
   {{"title": "标题", "sections": [{{"heading": "章节标题", "points": ["要点"]}}]}}
   """)
           chain = prompt | llm
           response = chain.invoke({"topic": topic, "research": research})
           
           import json
           try:
               outline = json.loads(response.content)
           except:
               outline = {"title": topic, "sections": []}
           
           return {"outline": outline}
       
       @staticmethod
       def write_draft(state: ContentState) -> dict:
           """写作节点：撰写草稿"""
           outline = state["outline"]
           platform = state["platform"]
           
           prompt = ChatPromptTemplate.from_template("""
   根据大纲撰写文章：

   大纲：{outline}
   目标平台：{platform}

   要求：
   - 符合平台风格
   - 内容充实有价值
   - 结构清晰
   """)
           chain = prompt | llm
           response = chain.invoke({
               "outline": str(outline),
               "platform": platform
           })
           
           return {"draft": response.content}
       
       @staticmethod
       def evaluate(state: ContentState) -> dict:
           """评估节点：评价内容质量"""
           draft = state["draft"]
           
           prompt = ChatPromptTemplate.from_template("""
   评估以下文章的质量（0-100分）：

   {draft}

   评分标准：
   - 内容价值（40分）
   - 结构清晰（30分）
   - 语言表达（30分）

   只输出一个数字分数。
   """)
           chain = prompt | llm
           response = chain.invoke({"draft": draft})
           
           try:
               score = int(response.content.strip())
           except:
               score = 70
           
           return {"quality_score": score}
       
       @staticmethod
       def improve(state: ContentState) -> dict:
           """改进节点：优化内容"""
           draft = state["draft"]
           feedback = state.get("feedback", "请优化内容质量")
           
           prompt = ChatPromptTemplate.from_template("""
   请优化以下文章：

   原文：
   {draft}

   改进建议：{feedback}

   输出优化后的完整文章。
   """)
           chain = prompt | llm
           response = chain.invoke({"draft": draft, "feedback": feedback})
           
           return {"draft": response.content}
       
       @staticmethod
       def finalize(state: ContentState) -> dict:
           """定稿节点：最终处理"""
           return {"final_content": state["draft"]}

   # ========== 路由函数 ==========

   def quality_router(state: ContentState) -> str:
       """根据质量分数路由"""
       score = state["quality_score"]
       if score >= 80:
           return "pass"
       else:
           return "improve"

   # ========== 构建图 ==========

   def create_content_graph():
       graph = StateGraph(ContentState)
       
       # 添加节点
       nodes = ContentNodes()
       graph.add_node("research", nodes.research)
       graph.add_node("outline", nodes.create_outline)
       graph.add_node("write", nodes.write_draft)
       graph.add_node("evaluate", nodes.evaluate)
       graph.add_node("improve", nodes.improve)
       graph.add_node("finalize", nodes.finalize)
       
       # 添加边
       graph.add_edge(START, "research")
       graph.add_edge("research", "outline")
       graph.add_edge("outline", "write")
       graph.add_edge("write", "evaluate")
       
       # 条件边：根据质量分数决定下一步
       graph.add_conditional_edges(
           "evaluate",
           quality_router,
           {
               "pass": "finalize",
               "improve": "improve"
           }
       )
       
       graph.add_edge("improve", "evaluate")  # 改进后重新评估
       graph.add_edge("finalize", END)
       
       return graph.compile()

   # ========== 运行 ==========

   app = create_content_graph()

   result = app.invoke({
       "topic": "LangGraph入门教程",
       "platform": "知乎"
   })

   print(f"质量分数: {result['quality_score']}")
   print(f"最终内容:\n{result['final_content'][:500]}...")

图结构可视化
============

.. code-block:: text

   graph TD
       START --> research
       research --> outline
       outline --> write
       write --> evaluate
       evaluate -->|score >= 80| finalize
       evaluate -->|score < 80| improve
       improve --> evaluate
       finalize --> END

节点最佳实践
============

1. **单一职责**

.. code-block:: python

   # ✅ 好：每个节点只做一件事
   def generate_title(state): ...
   def generate_outline(state): ...
   def write_content(state): ...

   # ❌ 差：一个节点做太多事
   def do_everything(state): ...

2. **清晰的输入输出**

.. code-block:: python

   # ✅ 好：明确的状态更新
   def my_node(state: State) -> dict:
       result = process(state["input"])
       return {"output": result}  # 只返回需要更新的字段

3. **错误处理**

.. code-block:: python

   def safe_node(state: State) -> dict:
       try:
           result = risky_operation()
           return {"result": result, "error": None}
       except Exception as e:
           return {"result": None, "error": str(e)}

下一步
======

在下一个教程中，我们将学习条件路由的高级用法。

:doc:`tutorial_04_conditional_routing`
