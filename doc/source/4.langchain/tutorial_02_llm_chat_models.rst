####################################
Tutorial 2: LLM 与 Chat Models
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

LLM vs Chat Models
==================

LangChain 支持两种类型的语言模型：

.. csv-table::
   :header: "类型", "输入", "输出", "适用场景"
   :widths: 15, 25, 25, 35

   "LLM", "字符串", "字符串", "文本补全、简单生成"
   "Chat Model", "消息列表", "消息对象", "对话、复杂交互"

现代应用推荐使用 **Chat Models**，因为它们更强大且支持更丰富的交互模式。

使用 Chat Models
================

基本用法
--------

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

   # 初始化 Chat Model
   chat = ChatOpenAI(
       model="gpt-4o-mini",
       temperature=0.7,  # 创造性程度 0-1
       max_tokens=1000   # 最大输出长度
   )

   # 构建消息
   messages = [
       SystemMessage(content="你是一个专业的自媒体内容创作者"),
       HumanMessage(content="帮我写一个关于AI编程的文章标题")
   ]

   # 调用模型
   response = chat.invoke(messages)
   print(response.content)

消息类型
--------

.. code-block:: python

   from langchain_core.messages import (
       SystemMessage,    # 系统指令，设定角色和行为
       HumanMessage,     # 用户输入
       AIMessage,        # AI 回复
       ToolMessage,      # 工具调用结果
   )

   # 多轮对话示例
   messages = [
       SystemMessage(content="你是自媒体写作助手"),
       HumanMessage(content="我想写一篇关于Python的文章"),
       AIMessage(content="好的！Python是个很好的主题。你想侧重哪个方面？"),
       HumanMessage(content="侧重于初学者入门")
   ]

   response = chat.invoke(messages)

流式输出
--------

.. code-block:: python

   # 流式输出，适合长文本生成
   for chunk in chat.stream(messages):
       print(chunk.content, end="", flush=True)

使用不同的模型提供商
====================

OpenAI
------

.. code-block:: python

   from langchain_openai import ChatOpenAI

   chat = ChatOpenAI(model="gpt-4o")

Anthropic (Claude)
------------------

.. code-block:: python

   from langchain_anthropic import ChatAnthropic

   chat = ChatAnthropic(model="claude-3-5-sonnet-20241022")

本地模型 (Ollama)
-----------------

.. code-block:: python

   from langchain_community.chat_models import ChatOllama

   chat = ChatOllama(model="llama3.2")

实战：自媒体标题生成器
======================

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain_core.messages import SystemMessage, HumanMessage

   class TitleGenerator:
       def __init__(self):
           self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
           self.system_prompt = """你是一个专业的自媒体标题创作专家。
           你擅长创作吸引眼球、引发好奇心的标题。
           标题要求：
           1. 简洁有力，不超过20个字
           2. 包含数字或疑问能增加点击率
           3. 避免标题党，保持内容相关性
           """
       
       def generate(self, topic: str, count: int = 5) -> list[str]:
           messages = [
               SystemMessage(content=self.system_prompt),
               HumanMessage(content=f"请为主题「{topic}」生成{count}个标题，每行一个")
           ]
           
           response = self.chat.invoke(messages)
           titles = response.content.strip().split("\n")
           return [t.strip() for t in titles if t.strip()]

   # 使用示例
   generator = TitleGenerator()
   titles = generator.generate("Python自动化办公")
   
   for i, title in enumerate(titles, 1):
       print(f"{i}. {title}")

输出示例::

   1. 5个Python技巧，让你的办公效率提升10倍
   2. 还在手动处理Excel？Python帮你一键搞定
   3. 程序员都在用的办公自动化神器，你还不知道？
   4. 从入门到精通：Python自动化办公完全指南
   5. 每天节省2小时！Python办公自动化实战

模型参数详解
============

.. csv-table::
   :header: "参数", "说明", "推荐值"
   :widths: 20, 50, 30

   "temperature", "控制输出随机性，越高越有创意", "创意写作: 0.7-0.9"
   "max_tokens", "最大输出 token 数", "根据需求设置"
   "top_p", "核采样参数", "通常保持默认"
   "frequency_penalty", "降低重复词出现频率", "0-1"
   "presence_penalty", "鼓励谈论新话题", "0-1"

下一步
======

在下一个教程中，我们将学习如何使用 Prompt Templates 来构建可复用的提示词模板。

:doc:`tutorial_03_prompt_templates`
