####################################
Tutorial 7: RAG Prompt 工程
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

Prompt 在 RAG 中的作用
======================

Prompt 是 RAG 系统中连接检索和生成的桥梁，好的 Prompt 能显著提升回答质量。

.. code-block:: text

   RAG Prompt 结构
   
   ┌─────────────────────────────────────────────────────────────┐
   │  系统指令 (System)                                          │
   │  · 定义角色和行为                                           │
   │  · 设置回答规则                                             │
   ├─────────────────────────────────────────────────────────────┤
   │  上下文 (Context)                                           │
   │  · 检索到的相关文档                                         │
   │  · 结构化呈现                                               │
   ├─────────────────────────────────────────────────────────────┤
   │  用户问题 (Question)                                        │
   │  · 原始问题                                                 │
   │  · 可能的改写                                               │
   ├─────────────────────────────────────────────────────────────┤
   │  输出格式 (Format)                                          │
   │  · 期望的回答格式                                           │
   │  · 特殊要求                                                 │
   └─────────────────────────────────────────────────────────────┘

基础 RAG Prompt
===============

.. code-block:: python

   from langchain.prompts import PromptTemplate

   # 基础 RAG Prompt
   basic_rag_prompt = PromptTemplate(
       input_variables=["context", "question"],
       template="""基于以下参考资料回答问题。

   参考资料：
   {context}

   问题：{question}

   回答："""
   )

   # 使用
   context = """
   1. Python是一种高级编程语言，以简洁易读著称。
   2. Python支持多种编程范式，包括面向对象和函数式编程。
   """
   question = "Python有什么特点？"

   prompt = basic_rag_prompt.format(context=context, question=question)
   print(prompt)

增强型 RAG Prompt
=================

1. 带角色定义的 Prompt
----------------------

.. code-block:: python

   role_based_prompt = PromptTemplate(
       input_variables=["context", "question"],
       template="""你是一个专业的技术顾问，负责回答用户的技术问题。

   回答要求：
   - 基于提供的参考资料回答
   - 使用清晰、专业的语言
   - 如果资料中没有相关信息，明确说明

   参考资料：
   {context}

   用户问题：{question}

   请提供专业、准确的回答："""
   )

2. 带引用的 Prompt
------------------

.. code-block:: python

   citation_prompt = PromptTemplate(
       input_variables=["context", "question"],
       template="""基于以下编号的参考资料回答问题，并在回答中标注引用来源。

   参考资料：
   {context}

   问题：{question}

   请回答问题，并使用 [1], [2] 等格式标注引用的资料编号："""
   )

   # 格式化上下文
   def format_context_with_numbers(documents):
       formatted = []
       for i, doc in enumerate(documents, 1):
           formatted.append(f"[{i}] {doc}")
       return "\n".join(formatted)

   # 使用
   docs = [
       "Python是解释型语言，代码可以直接运行。",
       "Python有丰富的标准库和第三方库。",
       "Python广泛应用于数据科学和AI领域。"
   ]
   context = format_context_with_numbers(docs)
   prompt = citation_prompt.format(context=context, question="Python的优势是什么？")

3. 分步骤回答的 Prompt
----------------------

.. code-block:: python

   step_by_step_prompt = PromptTemplate(
       input_variables=["context", "question"],
       template="""请按照以下步骤回答问题：

   参考资料：
   {context}

   问题：{question}

   请按以下格式回答：

   1. 理解问题：简述你对问题的理解
   2. 相关信息：列出参考资料中的相关要点
   3. 综合回答：基于相关信息给出完整回答
   4. 补充说明：如有必要，添加注意事项或局限性

   回答："""
   )

处理特殊情况
============

1. 信息不足时的处理
-------------------

.. code-block:: python

   insufficient_info_prompt = PromptTemplate(
       input_variables=["context", "question"],
       template="""基于以下参考资料回答问题。

   参考资料：
   {context}

   问题：{question}

   回答要求：
   - 只基于参考资料中的信息回答
   - 如果资料中没有足够信息，请明确说明"根据提供的资料，无法回答这个问题"
   - 不要编造或猜测答案
   - 如果只能部分回答，说明哪些部分可以回答，哪些不能

   回答："""
   )

2. 多文档综合
-------------

.. code-block:: python

   multi_doc_prompt = PromptTemplate(
       input_variables=["context", "question"],
       template="""你需要综合多个来源的信息来回答问题。

   参考资料（来自不同来源）：
   {context}

   问题：{question}

   回答要求：
   - 综合各个来源的信息
   - 如果不同来源有矛盾，指出差异
   - 优先使用多个来源都支持的信息
   - 标注信息来源

   综合回答："""
   )

3. 对话历史
-----------

.. code-block:: python

   conversational_prompt = PromptTemplate(
       input_variables=["context", "chat_history", "question"],
       template="""基于参考资料和对话历史回答问题。

   参考资料：
   {context}

   对话历史：
   {chat_history}

   当前问题：{question}

   请结合上下文和对话历史，给出连贯的回答："""
   )

   # 格式化对话历史
   def format_chat_history(messages):
       formatted = []
       for msg in messages:
           role = "用户" if msg["role"] == "user" else "助手"
           formatted.append(f"{role}: {msg['content']}")
       return "\n".join(formatted)

实战：构建 Prompt 管理器
========================

.. code-block:: python

   from langchain.prompts import PromptTemplate
   from typing import List, Dict, Optional
   from enum import Enum

   class PromptType(Enum):
       BASIC = "basic"
       CITATION = "citation"
       STEP_BY_STEP = "step_by_step"
       CONVERSATIONAL = "conversational"

   class RAGPromptManager:
       """RAG Prompt 管理器"""
       
       def __init__(self):
           self.prompts = self._init_prompts()
       
       def _init_prompts(self) -> Dict[PromptType, PromptTemplate]:
           return {
               PromptType.BASIC: PromptTemplate(
                   input_variables=["context", "question"],
                   template="""参考以下资料回答问题。

   资料：
   {context}

   问题：{question}

   回答："""
               ),
               
               PromptType.CITATION: PromptTemplate(
                   input_variables=["context", "question"],
                   template="""基于编号资料回答，标注引用。

   资料：
   {context}

   问题：{question}

   回答（使用[1][2]标注来源）："""
               ),
               
               PromptType.STEP_BY_STEP: PromptTemplate(
                   input_variables=["context", "question"],
                   template="""分步骤回答问题。

   资料：
   {context}

   问题：{question}

   步骤：
   1. 问题理解：
   2. 关键信息：
   3. 回答："""
               ),
               
               PromptType.CONVERSATIONAL: PromptTemplate(
                   input_variables=["context", "chat_history", "question"],
                   template="""结合历史对话回答。

   资料：
   {context}

   历史：
   {chat_history}

   问题：{question}

   回答："""
               )
           }
       
       def format_context(
           self,
           documents: List[str],
           with_numbers: bool = False,
           max_length: int = 2000
       ) -> str:
           """格式化上下文"""
           if with_numbers:
               formatted = [f"[{i+1}] {doc}" for i, doc in enumerate(documents)]
           else:
               formatted = documents
           
           context = "\n\n".join(formatted)
           
           # 截断过长的上下文
           if len(context) > max_length:
               context = context[:max_length] + "...[内容过长已截断]"
           
           return context
       
       def get_prompt(
           self,
           prompt_type: PromptType,
           documents: List[str],
           question: str,
           chat_history: Optional[List[Dict]] = None
       ) -> str:
           """获取格式化的 Prompt"""
           with_numbers = prompt_type == PromptType.CITATION
           context = self.format_context(documents, with_numbers)
           
           template = self.prompts[prompt_type]
           
           if prompt_type == PromptType.CONVERSATIONAL:
               history_str = self._format_history(chat_history or [])
               return template.format(
                   context=context,
                   chat_history=history_str,
                   question=question
               )
           
           return template.format(context=context, question=question)
       
       def _format_history(self, messages: List[Dict]) -> str:
           if not messages:
               return "无历史对话"
           
           formatted = []
           for msg in messages[-5:]:  # 只保留最近5轮
               role = "用户" if msg.get("role") == "user" else "助手"
               formatted.append(f"{role}: {msg.get('content', '')}")
           
           return "\n".join(formatted)

   # 使用示例
   manager = RAGPromptManager()

   documents = [
       "Python是一种解释型高级编程语言。",
       "Python支持多种编程范式。",
       "Python在数据科学领域应用广泛。"
   ]

   # 基础 Prompt
   basic_prompt = manager.get_prompt(
       PromptType.BASIC,
       documents,
       "Python是什么？"
   )
   print("基础 Prompt:")
   print(basic_prompt)

   # 带引用的 Prompt
   citation_prompt = manager.get_prompt(
       PromptType.CITATION,
       documents,
       "Python有什么特点？"
   )
   print("\n带引用 Prompt:")
   print(citation_prompt)

   # 对话式 Prompt
   chat_history = [
       {"role": "user", "content": "什么是编程语言？"},
       {"role": "assistant", "content": "编程语言是人与计算机交流的工具。"}
   ]
   conv_prompt = manager.get_prompt(
       PromptType.CONVERSATIONAL,
       documents,
       "Python属于哪种类型？",
       chat_history
   )
   print("\n对话式 Prompt:")
   print(conv_prompt)

Prompt 优化技巧
===============

1. **明确指令**

.. code-block:: python

   # ❌ 模糊
   template = "回答问题：{question}"

   # ✅ 明确
   template = """你是专业的技术顾问。
   基于以下资料回答问题，要求：
   - 使用简洁专业的语言
   - 如果不确定，说明原因
   
   资料：{context}
   问题：{question}
   回答："""

2. **结构化输出**

.. code-block:: python

   structured_prompt = """基于资料回答问题。

   资料：{context}
   问题：{question}

   请按以下JSON格式回答：
   {{
       "answer": "回答内容",
       "confidence": "high/medium/low",
       "sources": ["引用的资料编号"]
   }}"""

3. **Few-shot 示例**

.. code-block:: python

   few_shot_prompt = """基于资料回答问题，参考以下示例：

   示例1：
   资料：[1] Python是解释型语言
   问题：Python是什么类型的语言？
   回答：根据资料[1]，Python是解释型语言。

   示例2：
   资料：[1] Java是编译型语言
   问题：Python的特点是什么？
   回答：提供的资料中没有关于Python的信息，无法回答。

   现在回答：
   资料：{context}
   问题：{question}
   回答："""

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "System Prompt", "定义AI角色和行为的指令"
   "Context", "检索到的相关文档"
   "Few-shot", "提供示例来引导回答"
   "Chain of Thought", "引导分步骤思考"
   "Output Format", "指定输出格式"

下一步
======

在下一个教程中，我们将学习如何评估 RAG 系统。

:doc:`tutorial_08_evaluation`
