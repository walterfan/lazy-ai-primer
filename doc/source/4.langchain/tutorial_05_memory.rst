####################################
Tutorial 5: Memory 记忆系统
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是 Memory？
===============

Memory 让 LLM 应用能够记住之前的对话内容，实现：

- 多轮对话的上下文保持
- 用户偏好记忆
- 长期知识积累

LangChain 提供了多种 Memory 类型，适用于不同场景。

Memory 类型概览
===============

.. csv-table::
   :header: "类型", "特点", "适用场景"
   :widths: 25, 40, 35

   "ConversationBufferMemory", "保存完整对话历史", "短对话"
   "ConversationBufferWindowMemory", "只保留最近N轮", "长对话，节省token"
   "ConversationSummaryMemory", "保存对话摘要", "超长对话"
   "ConversationEntityMemory", "提取并记忆实体信息", "需要记住关键信息"

基础用法
========

ConversationBufferMemory
------------------------

.. code-block:: python

   from langchain.memory import ConversationBufferMemory
   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
   from langchain_core.runnables import RunnablePassthrough

   # 创建 Memory
   memory = ConversationBufferMemory(return_messages=True)

   # 创建带 Memory 的 Prompt
   prompt = ChatPromptTemplate.from_messages([
       ("system", "你是自媒体写作助手，帮助用户创作内容"),
       MessagesPlaceholder(variable_name="history"),
       ("human", "{input}")
   ])

   model = ChatOpenAI(model="gpt-4o-mini")

   # 对话函数
   def chat(user_input: str) -> str:
       # 获取历史
       history = memory.load_memory_variables({})["history"]
       
       # 调用模型
       messages = prompt.format_messages(history=history, input=user_input)
       response = model.invoke(messages)
       
       # 保存到 Memory
       memory.save_context(
           {"input": user_input},
           {"output": response.content}
       )
       
       return response.content

   # 多轮对话示例
   print(chat("我想写一篇关于Python的文章"))
   print(chat("文章主要面向初学者"))
   print(chat("帮我生成一个大纲"))  # 模型会记住前面的上下文

ConversationBufferWindowMemory
------------------------------

.. code-block:: python

   from langchain.memory import ConversationBufferWindowMemory

   # 只保留最近5轮对话
   memory = ConversationBufferWindowMemory(k=5, return_messages=True)

ConversationSummaryMemory
-------------------------

.. code-block:: python

   from langchain.memory import ConversationSummaryMemory
   from langchain_openai import ChatOpenAI

   llm = ChatOpenAI(model="gpt-4o-mini")

   # 使用 LLM 生成对话摘要
   memory = ConversationSummaryMemory(llm=llm, return_messages=True)

实战：带记忆的自媒体写作助手
============================

.. code-block:: python

   from langchain.memory import ConversationBufferWindowMemory
   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
   from langchain_core.output_parsers import StrOutputParser
   from typing import Optional

   class WritingAssistant:
       """带记忆的自媒体写作助手"""
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
           self.memory = ConversationBufferWindowMemory(
               k=10,  # 保留最近10轮对话
               return_messages=True,
               memory_key="history"
           )
           
           self.prompt = ChatPromptTemplate.from_messages([
               ("system", """你是专业的自媒体写作助手。你的职责是：
   1. 帮助用户规划内容主题
   2. 撰写各种类型的文章
   3. 优化标题和文案
   4. 适配不同平台风格

   请记住用户之前提到的偏好和需求，保持对话连贯性。"""),
               MessagesPlaceholder(variable_name="history"),
               ("human", "{input}")
           ])
           
           self.chain = self.prompt | self.llm | StrOutputParser()
       
       def chat(self, user_input: str) -> str:
           # 获取历史消息
           history = self.memory.load_memory_variables({})["history"]
           
           # 调用 Chain
           response = self.chain.invoke({
               "history": history,
               "input": user_input
           })
           
           # 保存对话
           self.memory.save_context(
               {"input": user_input},
               {"output": response}
           )
           
           return response
       
       def get_history(self) -> list:
           """获取对话历史"""
           return self.memory.load_memory_variables({})["history"]
       
       def clear_history(self):
           """清空对话历史"""
           self.memory.clear()

   # 使用示例
   assistant = WritingAssistant()

   # 第一轮：确定主题
   print("用户: 我想写一篇关于AI编程的文章")
   print("助手:", assistant.chat("我想写一篇关于AI编程的文章"))

   # 第二轮：细化需求（助手会记住是AI编程主题）
   print("\n用户: 主要面向没有编程基础的小白")
   print("助手:", assistant.chat("主要面向没有编程基础的小白"))

   # 第三轮：生成内容（助手会记住所有上下文）
   print("\n用户: 帮我写一个吸引人的开头")
   print("助手:", assistant.chat("帮我写一个吸引人的开头"))

   # 第四轮：修改优化
   print("\n用户: 开头太长了，精简一下")
   print("助手:", assistant.chat("开头太长了，精简一下"))

持久化 Memory
=============

将 Memory 保存到数据库：

.. code-block:: python

   from langchain_community.chat_message_histories import SQLChatMessageHistory
   from langchain.memory import ConversationBufferMemory

   # 使用 SQLite 持久化
   message_history = SQLChatMessageHistory(
       session_id="user_123",
       connection_string="sqlite:///chat_history.db"
   )

   memory = ConversationBufferMemory(
       chat_memory=message_history,
       return_messages=True
   )

使用 Redis 存储
---------------

.. code-block:: python

   from langchain_community.chat_message_histories import RedisChatMessageHistory

   message_history = RedisChatMessageHistory(
       session_id="user_123",
       url="redis://localhost:6379"
   )

   memory = ConversationBufferMemory(
       chat_memory=message_history,
       return_messages=True
   )

实战：多用户写作助手
====================

.. code-block:: python

   from langchain.memory import ConversationBufferWindowMemory
   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
   from typing import Dict

   class MultiUserWritingAssistant:
       """支持多用户的写作助手，每个用户有独立的记忆"""
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-4o-mini")
           self.user_memories: Dict[str, ConversationBufferWindowMemory] = {}
           
           self.prompt = ChatPromptTemplate.from_messages([
               ("system", "你是专业的自媒体写作助手"),
               MessagesPlaceholder(variable_name="history"),
               ("human", "{input}")
           ])
       
       def _get_memory(self, user_id: str) -> ConversationBufferWindowMemory:
           """获取或创建用户的 Memory"""
           if user_id not in self.user_memories:
               self.user_memories[user_id] = ConversationBufferWindowMemory(
                   k=10,
                   return_messages=True
               )
           return self.user_memories[user_id]
       
       def chat(self, user_id: str, message: str) -> str:
           memory = self._get_memory(user_id)
           history = memory.load_memory_variables({})["history"]
           
           messages = self.prompt.format_messages(
               history=history,
               input=message
           )
           response = self.llm.invoke(messages)
           
           memory.save_context(
               {"input": message},
               {"output": response.content}
           )
           
           return response.content

   # 使用示例
   assistant = MultiUserWritingAssistant()

   # 用户A的对话
   print(assistant.chat("user_a", "我想写技术文章"))
   print(assistant.chat("user_a", "主题是Python"))

   # 用户B的对话（独立的上下文）
   print(assistant.chat("user_b", "我想写生活分享"))
   print(assistant.chat("user_b", "关于旅行的"))

   # 用户A继续对话（记得之前的上下文）
   print(assistant.chat("user_a", "帮我写个标题"))  # 会记得是Python技术文章

下一步
======

在下一个教程中，我们将学习如何使用 Agents 和 Tools 来让 AI 具备执行操作的能力。

:doc:`tutorial_06_agents_tools`
