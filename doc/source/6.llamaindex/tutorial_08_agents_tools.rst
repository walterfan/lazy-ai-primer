################################
Tutorial 8: Agents 与 Tools
################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

.. contents:: 目录
   :local:
   :depth: 2

Agent 概述
==========

Agent 是能够自主决策和使用工具的智能系统。
LlamaIndex 提供了构建 Agent 的完整框架。

.. code-block:: text

   Agent 工作流程：

   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  用户输入   │───►│   Agent     │───►│  选择工具   │
   └─────────────┘    │  (推理)     │    └─────────────┘
                      └─────────────┘           │
                             ▲                  ▼
                             │           ┌─────────────┐
                             │           │  执行工具   │
                             │           └─────────────┘
                             │                  │
                      ┌─────────────┐           │
                      │  观察结果   │◄──────────┘
                      └─────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │  生成回答   │
                      └─────────────┘

ReAct Agent
===========

基于 ReAct（Reasoning + Acting）范式的 Agent。

基本用法
--------

.. code-block:: python

   from llama_index.core.agent import ReActAgent
   from llama_index.core.tools import FunctionTool
   from llama_index.llms.openai import OpenAI

   # 定义工具函数
   def multiply(a: int, b: int) -> int:
       """将两个整数相乘"""
       return a * b

   def add(a: int, b: int) -> int:
       """将两个整数相加"""
       return a + b

   # 创建工具
   multiply_tool = FunctionTool.from_defaults(fn=multiply)
   add_tool = FunctionTool.from_defaults(fn=add)

   # 创建 Agent
   llm = OpenAI(model="gpt-4o-mini")
   agent = ReActAgent.from_tools(
       [multiply_tool, add_tool],
       llm=llm,
       verbose=True
   )

   # 使用 Agent
   response = agent.chat("计算 (3 + 5) * 2 的结果")
   print(response)

查看推理过程
------------

.. code-block:: python

   # Agent 会显示其推理过程（当 verbose=True）
   # Thought: 我需要先计算 3 + 5
   # Action: add
   # Action Input: {"a": 3, "b": 5}
   # Observation: 8
   # Thought: 现在我需要将结果乘以 2
   # Action: multiply
   # Action Input: {"a": 8, "b": 2}
   # Observation: 16
   # Thought: 我现在知道最终答案了
   # Answer: (3 + 5) * 2 = 16

FunctionTool
============

将 Python 函数转换为 Agent 可用的工具。

基本用法
--------

.. code-block:: python

   from llama_index.core.tools import FunctionTool

   def get_weather(city: str) -> str:
       """获取指定城市的天气信息

       Args:
           city: 城市名称
       """
       # 模拟天气API
       weather_data = {
           "北京": "晴天，25°C",
           "上海": "多云，28°C",
           "深圳": "小雨，30°C"
       }
       return weather_data.get(city, f"未找到{city}的天气信息")

   # 创建工具
   weather_tool = FunctionTool.from_defaults(
       fn=get_weather,
       name="get_weather",
       description="获取指定城市的当前天气"
   )

   # 使用
   agent = ReActAgent.from_tools([weather_tool], llm=llm, verbose=True)
   response = agent.chat("北京今天天气怎么样？")

异步工具
--------

.. code-block:: python

   import aiohttp
   from llama_index.core.tools import FunctionTool

   async def async_fetch_data(url: str) -> str:
       """异步获取URL内容"""
       async with aiohttp.ClientSession() as session:
           async with session.get(url) as response:
               return await response.text()

   async_tool = FunctionTool.from_defaults(
       async_fn=async_fetch_data,
       name="fetch_url",
       description="获取URL的内容"
   )

QueryEngineTool
===============

将查询引擎包装为工具。

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Document
   from llama_index.core.tools import QueryEngineTool, ToolMetadata

   # 创建知识库索引
   documents = [
       Document(text="公司成立于2020年，总部位于北京。"),
       Document(text="公司主要产品是AI解决方案，服务于企业客户。"),
       Document(text="公司2023年营收10亿元，同比增长50%。"),
   ]
   index = VectorStoreIndex.from_documents(documents)

   # 创建查询引擎工具
   query_engine = index.as_query_engine()
   query_tool = QueryEngineTool(
       query_engine=query_engine,
       metadata=ToolMetadata(
           name="company_knowledge",
           description="查询公司相关信息，包括历史、产品、财务等"
       )
   )

   # 创建 Agent
   agent = ReActAgent.from_tools([query_tool], llm=llm, verbose=True)

   response = agent.chat("公司的营收情况如何？")
   print(response)

多知识库 Agent
--------------

.. code-block:: python

   # 创建多个知识库
   product_docs = [Document(text="产品A是我们的旗舰产品...")]
   finance_docs = [Document(text="2023年财务报告显示...")]
   hr_docs = [Document(text="公司目前有500名员工...")]

   product_index = VectorStoreIndex.from_documents(product_docs)
   finance_index = VectorStoreIndex.from_documents(finance_docs)
   hr_index = VectorStoreIndex.from_documents(hr_docs)

   # 创建多个查询工具
   tools = [
       QueryEngineTool(
           query_engine=product_index.as_query_engine(),
           metadata=ToolMetadata(
               name="product_info",
               description="查询产品相关信息"
           )
       ),
       QueryEngineTool(
           query_engine=finance_index.as_query_engine(),
           metadata=ToolMetadata(
               name="finance_info",
               description="查询财务相关信息"
           )
       ),
       QueryEngineTool(
           query_engine=hr_index.as_query_engine(),
           metadata=ToolMetadata(
               name="hr_info",
               description="查询人力资源相关信息"
           )
       ),
   ]

   # 创建多知识库 Agent
   agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

   # Agent 会自动选择合适的知识库
   response = agent.chat("公司的员工人数是多少？")

自定义工具
==========

创建更复杂的自定义工具。

.. code-block:: python

   from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
   from typing import Any

   class DatabaseTool(BaseTool):
       """数据库查询工具"""

       def __init__(self, connection_string: str):
           self._connection_string = connection_string

       @property
       def metadata(self) -> ToolMetadata:
           return ToolMetadata(
               name="database_query",
               description="执行SQL查询获取数据库信息"
           )

       def call(self, query: str) -> ToolOutput:
           """执行查询"""
           # 这里是模拟，实际应该连接数据库
           result = f"执行查询: {query}"
           return ToolOutput(
               content=result,
               tool_name=self.metadata.name,
               raw_input={"query": query},
               raw_output=result
           )

   # 使用自定义工具
   db_tool = DatabaseTool("postgresql://localhost/mydb")
   agent = ReActAgent.from_tools([db_tool], llm=llm)

工具规范
--------

.. code-block:: python

   from llama_index.core.tools import ToolSpec

   class SearchToolSpec(ToolSpec):
       """搜索工具规范"""

       spec_functions = ["web_search", "image_search"]

       def web_search(self, query: str) -> str:
           """搜索网页

           Args:
               query: 搜索关键词
           """
           return f"网页搜索结果: {query}"

       def image_search(self, query: str) -> str:
           """搜索图片

           Args:
               query: 搜索关键词
           """
           return f"图片搜索结果: {query}"

   # 从规范创建工具
   search_spec = SearchToolSpec()
   tools = search_spec.to_tool_list()

   agent = ReActAgent.from_tools(tools, llm=llm)

内置工具
========

LlamaIndex 提供了一些内置工具。

代码解释器
----------

.. code-block:: python

   # pip install llama-index-tools-code-interpreter

   from llama_index.tools.code_interpreter import CodeInterpreterToolSpec

   code_spec = CodeInterpreterToolSpec()
   tools = code_spec.to_tool_list()

   agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

   response = agent.chat("计算1到100的素数之和")

维基百科工具
------------

.. code-block:: python

   # pip install llama-index-tools-wikipedia

   from llama_index.tools.wikipedia import WikipediaToolSpec

   wiki_spec = WikipediaToolSpec()
   tools = wiki_spec.to_tool_list()

   agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

   response = agent.chat("介绍一下阿尔伯特·爱因斯坦")

会话历史
========

Agent 支持多轮对话。

.. code-block:: python

   from llama_index.core.agent import ReActAgent
   from llama_index.core.memory import ChatMemoryBuffer

   # 创建带记忆的 Agent
   memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

   agent = ReActAgent.from_tools(
       tools,
       llm=llm,
       memory=memory,
       verbose=True
   )

   # 多轮对话
   response1 = agent.chat("公司的主要产品是什么？")
   print(f"回答1: {response1}")

   response2 = agent.chat("它的价格是多少？")  # 会记住上下文
   print(f"回答2: {response2}")

   # 查看对话历史
   print("\n对话历史:")
   for msg in agent.chat_history:
       print(f"{msg.role}: {msg.content[:50]}...")

   # 重置对话
   agent.reset()

OpenAI Agent
============

使用 OpenAI 的 Function Calling 能力。

.. code-block:: python

   from llama_index.agent.openai import OpenAIAgent
   from llama_index.llms.openai import OpenAI

   # 创建 OpenAI Agent
   llm = OpenAI(model="gpt-4o-mini")
   agent = OpenAIAgent.from_tools(
       tools,
       llm=llm,
       verbose=True
   )

   response = agent.chat("查询公司的财务信息")
   print(response)

流式响应
--------

.. code-block:: python

   # 流式对话
   streaming_response = agent.stream_chat("详细介绍公司的产品线")

   for token in streaming_response.response_gen:
       print(token, end="", flush=True)

并行工具调用
============

同时调用多个工具。

.. code-block:: python

   from llama_index.agent.openai import OpenAIAgent

   # OpenAI Agent 支持并行函数调用
   agent = OpenAIAgent.from_tools(
       tools,
       llm=OpenAI(model="gpt-4o"),
       verbose=True
   )

   # 当问题涉及多个领域时，Agent 可能并行调用多个工具
   response = agent.chat("告诉我公司的产品和财务状况")

Agent 执行器
============

自定义 Agent 的执行逻辑。

.. code-block:: python

   from llama_index.core.agent import AgentRunner
   from llama_index.core.agent.react.step import ReActAgentWorker

   # 创建 Agent Worker
   agent_worker = ReActAgentWorker.from_tools(
       tools,
       llm=llm,
       verbose=True
   )

   # 创建 Agent Runner
   agent = AgentRunner(agent_worker)

   # 创建任务
   task = agent.create_task("分析公司的财务状况")

   # 逐步执行
   step_output = agent.run_step(task.task_id)
   print(f"步骤完成: {step_output.is_last}")

   # 继续执行直到完成
   while not step_output.is_last:
       step_output = agent.run_step(task.task_id)
       print(f"步骤输出: {step_output.output}")

   # 获取最终响应
   response = agent.finalize_response(task.task_id)
   print(f"最终答案: {response}")

实战示例
========

构建一个智能助手 Agent。

.. code-block:: python

   from llama_index.core.agent import ReActAgent
   from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
   from llama_index.core import VectorStoreIndex, Document
   from llama_index.llms.openai import OpenAI
   from datetime import datetime
   import json

   class SmartAssistant:
       """智能助手"""

       def __init__(self):
           self.llm = OpenAI(model="gpt-4o-mini")
           self.tools = []
           self._setup_tools()
           self.agent = None
           self._build_agent()

       def _setup_tools(self):
           """设置工具"""
           # 1. 时间工具
           def get_current_time() -> str:
               """获取当前时间"""
               return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

           self.tools.append(FunctionTool.from_defaults(
               fn=get_current_time,
               name="get_time",
               description="获取当前日期和时间"
           ))

           # 2. 计算工具
           def calculate(expression: str) -> str:
               """计算数学表达式

               Args:
                   expression: 数学表达式，如 "2 + 3 * 4"
               """
               try:
                   result = eval(expression)
                   return str(result)
               except Exception as e:
                   return f"计算错误: {str(e)}"

           self.tools.append(FunctionTool.from_defaults(
               fn=calculate,
               name="calculator",
               description="计算数学表达式"
           ))

           # 3. 笔记工具
           self.notes = []

           def add_note(content: str) -> str:
               """添加笔记

               Args:
                   content: 笔记内容
               """
               self.notes.append({
                   "content": content,
                   "time": datetime.now().isoformat()
               })
               return f"笔记已添加: {content}"

           def get_notes() -> str:
               """获取所有笔记"""
               if not self.notes:
                   return "暂无笔记"
               return json.dumps(self.notes, ensure_ascii=False, indent=2)

           self.tools.append(FunctionTool.from_defaults(
               fn=add_note,
               name="add_note",
               description="添加一条笔记"
           ))

           self.tools.append(FunctionTool.from_defaults(
               fn=get_notes,
               name="get_notes",
               description="获取所有笔记"
           ))

           # 4. 知识库工具
           knowledge_docs = [
               Document(text="Python是一种解释型、面向对象的高级程序设计语言。"),
               Document(text="机器学习是人工智能的一个分支，通过算法从数据中学习。"),
               Document(text="深度学习使用多层神经网络，特别适合处理图像和语言。"),
           ]
           knowledge_index = VectorStoreIndex.from_documents(knowledge_docs)

           self.tools.append(QueryEngineTool(
               query_engine=knowledge_index.as_query_engine(),
               metadata=ToolMetadata(
                   name="knowledge_base",
                   description="查询技术知识库，包含编程、AI等内容"
               )
           ))

       def _build_agent(self):
           """构建 Agent"""
           self.agent = ReActAgent.from_tools(
               self.tools,
               llm=self.llm,
               verbose=True,
               system_prompt="""你是一个智能助手，可以：
               1. 查询当前时间
               2. 进行数学计算
               3. 管理笔记
               4. 查询技术知识

               请根据用户的问题选择合适的工具来帮助回答。
               """
           )

       def chat(self, message: str) -> str:
           """与助手对话"""
           response = self.agent.chat(message)
           return str(response)

       def reset(self):
           """重置对话历史"""
           self.agent.reset()

   # 使用示例
   assistant = SmartAssistant()

   # 时间查询
   print("=" * 50)
   print(assistant.chat("现在几点了？"))

   # 计算
   print("=" * 50)
   print(assistant.chat("帮我计算 (15 * 8 + 32) / 4"))

   # 笔记
   print("=" * 50)
   print(assistant.chat("帮我记一下：明天下午3点开会"))
   print(assistant.chat("我有哪些笔记？"))

   # 知识查询
   print("=" * 50)
   print(assistant.chat("什么是深度学习？"))

   # 复合任务
   print("=" * 50)
   print(assistant.chat("现在几点了？另外帮我查一下Python是什么"))

小结
====

本教程介绍了：

- Agent 的基本概念和工作流程
- ReAct Agent 的使用
- 各种工具的创建：FunctionTool、QueryEngineTool
- 自定义工具和工具规范
- 内置工具的使用
- 会话历史和记忆管理
- OpenAI Agent 和流式响应
- 完整的智能助手实现

下一步
------

在下一个教程中，我们将学习如何构建一个完整的知识库系统，
将前面学到的所有技术整合在一起。

练习
====

1. 创建一个能够搜索网页的 Agent
2. 实现一个多轮对话的客服 Agent
3. 构建一个能够执行代码的编程助手
4. 创建一个多知识库的企业问答 Agent
