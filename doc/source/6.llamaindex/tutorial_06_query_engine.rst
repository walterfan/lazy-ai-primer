################################
Tutorial 6: 查询引擎
################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

.. contents:: 目录
   :local:
   :depth: 2

查询引擎概述
============

查询引擎（Query Engine）是 LlamaIndex 中处理用户查询的核心组件，
负责检索相关内容并生成回答。

.. code-block:: text

   查询引擎工作流程：

   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  用户查询   │───►│   检索器    │───►│ 相关节点    │
   └─────────────┘    │ (Retriever) │    └─────────────┘
                      └─────────────┘           │
                                                ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │   最终回答  │◄───│  响应合成器 │◄───│ 上下文构建  │
   └─────────────┘    │ (Synthesizer)│   └─────────────┘
                      └─────────────┘

基础查询引擎
============

创建查询引擎
------------

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Document

   # 创建索引
   documents = [
       Document(text="LlamaIndex 是一个数据框架，用于构建 LLM 应用。"),
       Document(text="RAG 是检索增强生成的缩写，结合检索和生成。"),
       Document(text="向量数据库存储嵌入向量，支持相似度搜索。"),
   ]
   index = VectorStoreIndex.from_documents(documents)

   # 创建基础查询引擎
   query_engine = index.as_query_engine()

   # 执行查询
   response = query_engine.query("什么是 LlamaIndex？")
   print(response)

配置参数
--------

.. code-block:: python

   # 配置查询引擎参数
   query_engine = index.as_query_engine(
       similarity_top_k=5,           # 检索前5个相似结果
       response_mode="compact",      # 响应模式
       streaming=True,               # 启用流式输出
       verbose=True                  # 显示详细信息
   )

响应模式
========

LlamaIndex 提供多种响应合成模式。

refine 模式
-----------

逐步精炼答案，适合需要综合多个来源的场景。

.. code-block:: python

   query_engine = index.as_query_engine(
       response_mode="refine"
   )

   # 工作流程：
   # 1. 用第一个节点生成初始回答
   # 2. 用后续节点逐步精炼回答
   # 3. 返回最终精炼后的答案

compact 模式
------------

合并所有上下文后一次性生成，减少 API 调用。

.. code-block:: python

   query_engine = index.as_query_engine(
       response_mode="compact"
   )

   # 工作流程：
   # 1. 将所有检索到的节点合并
   # 2. 尽可能多地放入单个提示
   # 3. 一次性生成回答

tree_summarize 模式
-------------------

层次化摘要，适合处理大量文档。

.. code-block:: python

   query_engine = index.as_query_engine(
       response_mode="tree_summarize"
   )

   # 工作流程：
   # 1. 将节点分组
   # 2. 为每组生成摘要
   # 3. 递归合并摘要直到得到最终答案

accumulate 模式
---------------

累积所有响应，适合需要多个视角的场景。

.. code-block:: python

   query_engine = index.as_query_engine(
       response_mode="accumulate"
   )

   # 为每个节点单独生成回答，然后合并

no_text 模式
------------

只返回检索到的节点，不生成回答。

.. code-block:: python

   query_engine = index.as_query_engine(
       response_mode="no_text"
   )

   response = query_engine.query("关键词")
   # response.source_nodes 包含检索到的节点

流式输出
========

实时显示生成的回答。

.. code-block:: python

   # 启用流式输出
   query_engine = index.as_query_engine(streaming=True)

   # 流式查询
   streaming_response = query_engine.query("详细解释 RAG 的工作原理")

   # 逐步输出
   for text in streaming_response.response_gen:
       print(text, end="", flush=True)

   print()  # 换行

   # 获取完整响应
   full_response = streaming_response.get_response()

异步查询
========

.. code-block:: python

   import asyncio
   from llama_index.core import VectorStoreIndex

   async def async_query_example():
       # 创建索引
       index = VectorStoreIndex.from_documents(documents)
       query_engine = index.as_query_engine()

       # 异步查询
       response = await query_engine.aquery("什么是向量数据库？")
       print(response)

       # 并发多个查询
       questions = [
           "什么是 LlamaIndex？",
           "RAG 如何工作？",
           "向量嵌入是什么？"
       ]

       tasks = [query_engine.aquery(q) for q in questions]
       responses = await asyncio.gather(*tasks)

       for q, r in zip(questions, responses):
           print(f"Q: {q}")
           print(f"A: {r}\n")

   # 运行
   asyncio.run(async_query_example())

检索器（Retriever）
===================

检索器负责从索引中获取相关节点。

基础检索器
----------

.. code-block:: python

   from llama_index.core import VectorStoreIndex

   index = VectorStoreIndex.from_documents(documents)

   # 创建检索器
   retriever = index.as_retriever(
       similarity_top_k=5  # 返回前5个相似结果
   )

   # 执行检索
   nodes = retriever.retrieve("LlamaIndex 的功能")

   for node in nodes:
       print(f"Score: {node.score:.4f}")
       print(f"Text: {node.text[:100]}...")
       print("---")

自定义检索器
------------

.. code-block:: python

   from llama_index.core.retrievers import BaseRetriever
   from llama_index.core.schema import NodeWithScore, QueryBundle
   from typing import List

   class HybridRetriever(BaseRetriever):
       """混合检索器：结合向量检索和关键词检索"""

       def __init__(self, vector_retriever, keyword_retriever, alpha=0.5):
           self.vector_retriever = vector_retriever
           self.keyword_retriever = keyword_retriever
           self.alpha = alpha  # 向量检索权重

       def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
           # 向量检索
           vector_nodes = self.vector_retriever.retrieve(query_bundle)

           # 关键词检索
           keyword_nodes = self.keyword_retriever.retrieve(query_bundle)

           # 合并并重新评分
           all_nodes = {}
           for node in vector_nodes:
               all_nodes[node.node.id_] = node
               node.score = node.score * self.alpha

           for node in keyword_nodes:
               if node.node.id_ in all_nodes:
                   all_nodes[node.node.id_].score += node.score * (1 - self.alpha)
               else:
                   node.score = node.score * (1 - self.alpha)
                   all_nodes[node.node.id_] = node

           # 按分数排序
           sorted_nodes = sorted(
               all_nodes.values(),
               key=lambda x: x.score,
               reverse=True
           )

           return sorted_nodes[:10]

响应合成器
==========

自定义响应生成过程。

.. code-block:: python

   from llama_index.core.response_synthesizers import (
       get_response_synthesizer,
       ResponseMode
   )

   # 获取响应合成器
   synthesizer = get_response_synthesizer(
       response_mode=ResponseMode.COMPACT,
       verbose=True
   )

   # 使用自定义提示
   from llama_index.core.prompts import PromptTemplate

   custom_prompt = PromptTemplate(
       """基于以下上下文回答问题。
       如果无法从上下文中找到答案，请说"我不知道"。

       上下文：
       {context_str}

       问题：{query_str}

       回答："""
   )

   synthesizer = get_response_synthesizer(
       response_mode=ResponseMode.COMPACT,
       text_qa_template=custom_prompt
   )

自定义查询引擎
==============

完全自定义查询流程。

.. code-block:: python

   from llama_index.core.query_engine import CustomQueryEngine
   from llama_index.core.retrievers import BaseRetriever
   from llama_index.core.response_synthesizers import BaseSynthesizer
   from llama_index.core import Settings
   from llama_index.llms.openai import OpenAI

   class RAGQueryEngine(CustomQueryEngine):
       """自定义 RAG 查询引擎"""

       retriever: BaseRetriever
       llm: OpenAI

       def custom_query(self, query_str: str) -> str:
           # 1. 检索相关节点
           nodes = self.retriever.retrieve(query_str)

           # 2. 构建上下文
           context_parts = []
           for i, node in enumerate(nodes):
               context_parts.append(f"[{i+1}] {node.text}")
           context = "\n\n".join(context_parts)

           # 3. 构建提示
           prompt = f"""基于以下参考资料回答问题：

参考资料：
{context}

问题：{query_str}

请提供详细的回答，并引用相关的参考资料编号。"""

           # 4. 调用 LLM 生成回答
           response = self.llm.complete(prompt)

           return str(response)

   # 使用自定义查询引擎
   retriever = index.as_retriever(similarity_top_k=5)
   llm = OpenAI(model="gpt-4o-mini")

   custom_engine = RAGQueryEngine(
       retriever=retriever,
       llm=llm
   )

   response = custom_engine.query("LlamaIndex 有什么特点？")
   print(response)

查询转换
========

在查询前对问题进行转换以提高检索效果。

HyDE（假设性文档嵌入）
----------------------

.. code-block:: python

   from llama_index.core.indices.query.query_transform import HyDEQueryTransform
   from llama_index.core.query_engine import TransformQueryEngine

   # 创建 HyDE 转换器
   hyde = HyDEQueryTransform(include_original=True)

   # 基础查询引擎
   base_engine = index.as_query_engine()

   # 包装为转换查询引擎
   hyde_engine = TransformQueryEngine(
       query_engine=base_engine,
       query_transform=hyde
   )

   # HyDE 工作流程：
   # 1. 用 LLM 生成一个假设性的答案
   # 2. 用假设性答案进行检索（可能更精确）
   # 3. 用检索结果生成最终答案

   response = hyde_engine.query("什么是知识图谱？")

查询分解
--------

.. code-block:: python

   from llama_index.core.indices.query.query_transform import (
       StepDecomposeQueryTransform
   )

   # 创建查询分解转换器
   step_decompose = StepDecomposeQueryTransform(verbose=True)

   # 包装查询引擎
   decompose_engine = TransformQueryEngine(
       query_engine=base_engine,
       query_transform=step_decompose
   )

   # 分解复杂查询
   response = decompose_engine.query(
       "比较 LlamaIndex 和 LangChain 的区别，以及各自的优势"
   )

子问题查询引擎
==============

将复杂问题分解为子问题，分别查询后合并。

.. code-block:: python

   from llama_index.core.query_engine import SubQuestionQueryEngine
   from llama_index.core.tools import QueryEngineTool, ToolMetadata

   # 创建多个专题索引
   ai_index = VectorStoreIndex.from_documents([
       Document(text="人工智能相关内容...")
   ])
   ml_index = VectorStoreIndex.from_documents([
       Document(text="机器学习相关内容...")
   ])

   # 定义查询工具
   query_engine_tools = [
       QueryEngineTool(
           query_engine=ai_index.as_query_engine(),
           metadata=ToolMetadata(
               name="ai_knowledge",
               description="包含人工智能基础知识"
           )
       ),
       QueryEngineTool(
           query_engine=ml_index.as_query_engine(),
           metadata=ToolMetadata(
               name="ml_knowledge",
               description="包含机器学习详细内容"
           )
       ),
   ]

   # 创建子问题查询引擎
   sub_question_engine = SubQuestionQueryEngine.from_defaults(
       query_engine_tools=query_engine_tools
   )

   # 复杂问题会被分解
   response = sub_question_engine.query(
       "人工智能和机器学习有什么关系？各自有什么应用？"
   )

查询路由
========

根据查询内容路由到不同的引擎。

.. code-block:: python

   from llama_index.core.query_engine import RouterQueryEngine
   from llama_index.core.selectors import LLMSingleSelector

   # 创建不同用途的查询引擎
   summary_engine = summary_index.as_query_engine(
       response_mode="tree_summarize"
   )
   detail_engine = vector_index.as_query_engine(
       similarity_top_k=10
   )

   # 定义工具
   tools = [
       QueryEngineTool.from_defaults(
           query_engine=summary_engine,
           description="用于获取文档的总体摘要和概述"
       ),
       QueryEngineTool.from_defaults(
           query_engine=detail_engine,
           description="用于查找具体细节和回答详细问题"
       ),
   ]

   # 创建路由查询引擎
   router_engine = RouterQueryEngine(
       selector=LLMSingleSelector.from_defaults(),
       query_engine_tools=tools
   )

   # 自动路由到合适的引擎
   response = router_engine.query("总结这篇文档的主要内容")  # -> summary_engine
   response = router_engine.query("什么是向量嵌入？")        # -> detail_engine

实战示例
========

构建一个功能完整的查询系统。

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Document, Settings
   from llama_index.core.query_engine import RouterQueryEngine
   from llama_index.core.tools import QueryEngineTool, ToolMetadata
   from llama_index.core.selectors import LLMSingleSelector
   from llama_index.llms.openai import OpenAI
   from typing import Optional

   class IntelligentQuerySystem:
       """智能查询系统"""

       def __init__(self):
           Settings.llm = OpenAI(model="gpt-4o-mini")
           self.indexes = {}
           self.router_engine = None

       def add_knowledge_domain(self, name: str, documents: list, description: str):
           """添加知识领域"""
           index = VectorStoreIndex.from_documents(documents)
           self.indexes[name] = {
               "index": index,
               "description": description
           }
           self._rebuild_router()

       def _rebuild_router(self):
           """重建路由引擎"""
           tools = []
           for name, data in self.indexes.items():
               engine = data["index"].as_query_engine(
                   similarity_top_k=5,
                   response_mode="compact"
               )
               tools.append(
                   QueryEngineTool.from_defaults(
                       query_engine=engine,
                       description=data["description"]
                   )
               )

           if tools:
               self.router_engine = RouterQueryEngine(
                   selector=LLMSingleSelector.from_defaults(),
                   query_engine_tools=tools,
                   verbose=True
               )

       def query(self, question: str, domain: Optional[str] = None) -> str:
           """执行查询"""
           if domain and domain in self.indexes:
               # 查询特定领域
               engine = self.indexes[domain]["index"].as_query_engine()
               response = engine.query(question)
           elif self.router_engine:
               # 自动路由
               response = self.router_engine.query(question)
           else:
               return "没有可用的知识库"

           return str(response)

       def query_with_sources(self, question: str) -> dict:
           """查询并返回来源"""
           if not self.indexes:
               return {"answer": "没有可用的知识库", "sources": []}

           # 使用第一个索引演示
           name, data = list(self.indexes.items())[0]
           engine = data["index"].as_query_engine(similarity_top_k=5)
           response = engine.query(question)

           sources = []
           for node in response.source_nodes:
               sources.append({
                   "text": node.text[:200],
                   "score": node.score,
                   "metadata": node.metadata
               })

           return {
               "answer": str(response),
               "sources": sources
           }

   # 使用示例
   system = IntelligentQuerySystem()

   # 添加知识领域
   system.add_knowledge_domain(
       name="ai_basics",
       documents=[
           Document(text="人工智能是让机器展现智能行为的技术..."),
           Document(text="机器学习是 AI 的核心方法...")
       ],
       description="人工智能和机器学习的基础知识"
   )

   system.add_knowledge_domain(
       name="programming",
       documents=[
           Document(text="Python 是流行的编程语言..."),
           Document(text="代码质量很重要...")
       ],
       description="编程和软件开发相关知识"
   )

   # 查询
   result = system.query("什么是机器学习？")
   print(result)

   # 带来源的查询
   result = system.query_with_sources("AI 有什么应用？")
   print(f"答案: {result['answer']}")
   print(f"来源数: {len(result['sources'])}")

小结
====

本教程介绍了：

- 查询引擎的基本概念和工作流程
- 不同的响应模式：refine、compact、tree_summarize 等
- 流式输出和异步查询
- 检索器的使用和自定义
- 响应合成器的配置
- 查询转换：HyDE、查询分解
- 子问题查询引擎和路由查询引擎
- 完整的智能查询系统实现

下一步
------

在下一个教程中，我们将学习更多高级检索策略，
包括混合检索、重排序、多跳检索等。

练习
====

1. 比较不同响应模式的输出效果
2. 实现一个带查询历史的会话式查询引擎
3. 使用 HyDE 转换提高检索准确度
4. 构建一个多领域知识的路由查询系统
