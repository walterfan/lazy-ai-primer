####################################
Tutorial 1: LangChain 入门
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是 LangChain？
==================

LangChain 是一个用于开发由大语言模型（LLM）驱动的应用程序的框架。它提供了：

- **模块化组件**：可组合的抽象层，用于处理语言模型
- **链式调用**：将多个组件串联成复杂的工作流
- **Agent 能力**：让 LLM 能够使用工具并做出决策
- **记忆系统**：在对话中保持上下文

核心概念
========

.. csv-table::
   :header: "概念", "说明", "类比"
   :widths: 20, 50, 30

   "LLM/Chat Model", "语言模型接口，处理文本生成", "大脑"
   "Prompt Template", "动态构建提示词的模板", "问题模板"
   "Chain", "将多个组件串联的管道", "流水线"
   "Agent", "能够使用工具并决策的智能体", "员工"
   "Tool", "Agent 可以调用的外部功能", "工具箱"
   "Memory", "存储和检索对话历史", "记忆"

安装配置
========

1. 安装 LangChain
-----------------

.. code-block:: bash

   pip install langchain langchain-openai langchain-community

2. 配置 API Key
---------------

.. code-block:: python

   import os
   os.environ["OPENAI_API_KEY"] = "your-api-key-here"

   # 或者使用 .env 文件
   from dotenv import load_dotenv
   load_dotenv()

第一个 LangChain 程序
=====================

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain_core.messages import HumanMessage

   # 初始化模型
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

   # 发送消息
   messages = [HumanMessage(content="你好，请用一句话介绍自己")]
   response = llm.invoke(messages)

   print(response.content)

LangChain 架构概览
==================

.. code-block:: text

   ┌─────────────────────────────────────────────────────┐
   │                    Application                       │
   ├─────────────────────────────────────────────────────┤
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │
   │  │ Chains  │  │ Agents  │  │ Memory  │  │  RAG   │ │
   │  └────┬────┘  └────┬────┘  └────┬────┘  └───┬────┘ │
   │       │            │            │            │      │
   │  ┌────┴────────────┴────────────┴────────────┴────┐ │
   │  │              LangChain Core                     │ │
   │  │  (Prompts, LLMs, Output Parsers, Tools)        │ │
   │  └────────────────────┬───────────────────────────┘ │
   │                       │                             │
   │  ┌────────────────────┴───────────────────────────┐ │
   │  │           Model Providers & Integrations        │ │
   │  │  (OpenAI, Anthropic, HuggingFace, etc.)        │ │
   │  └────────────────────────────────────────────────┘ │
   └─────────────────────────────────────────────────────┘

自媒体 Agent 项目预览
=====================

在接下来的教程中，我们将逐步构建一个自媒体内容创作 Agent：

.. code-block:: python

   # 最终我们将实现这样的功能
   from selfmedia_agent import SelfMediaAgent

   agent = SelfMediaAgent()
   
   # 生成内容创意
   ideas = agent.generate_ideas(topic="AI编程", count=5)
   
   # 撰写文章
   article = agent.write_article(
       idea=ideas[0],
       platform="微信公众号",
       style="专业但易懂"
   )
   
   # 发布到多平台
   agent.publish(
       content=article,
       platforms=["微信公众号", "知乎", "头条"]
   )

下一步
======

在下一个教程中，我们将深入学习 LLM 和 Chat Models 的使用方法。

:doc:`tutorial_02_llm_chat_models`
