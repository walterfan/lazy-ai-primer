################################
Tutorial 5: 索引类型
################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

.. contents:: 目录
   :local:
   :depth: 2

索引类型概述
============

LlamaIndex 提供多种索引类型，每种都有其特定的使用场景和优势。

.. code-block:: text

   索引类型概览：

   ┌─────────────────────────────────────────────────────────────┐
   │                    LlamaIndex 索引类型                       │
   ├─────────────────────────────────────────────────────────────┤
   │                                                              │
   │   ┌───────────────┐    ┌───────────────┐    ┌────────────┐  │
   │   │ VectorStore   │    │   Summary     │    │  Keyword   │  │
   │   │    Index      │    │    Index      │    │   Index    │  │
   │   │   向量索引    │    │   摘要索引    │    │  关键词索引│  │
   │   └───────────────┘    └───────────────┘    └────────────┘  │
   │                                                              │
   │   ┌───────────────┐    ┌───────────────┐    ┌────────────┐  │
   │   │    Tree       │    │ Knowledge     │    │  Document  │  │
   │   │    Index      │    │ Graph Index   │    │  Summary   │  │
   │   │   树形索引    │    │  知识图谱索引 │    │   Index    │  │
   │   └───────────────┘    └───────────────┘    └────────────┘  │
   │                                                              │
   └─────────────────────────────────────────────────────────────┘

索引类型对比
------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - 索引类型
     - 特点
     - 适用场景
     - 查询效率
   * - VectorStoreIndex
     - 语义相似度检索
     - 问答、相似内容查找
     - O(log n) - O(n)
   * - SummaryIndex
     - 顺序遍历所有节点
     - 摘要生成、全文理解
     - O(n)
   * - KeywordTableIndex
     - 关键词匹配
     - 精确搜索、术语查找
     - O(1) - O(log n)
   * - TreeIndex
     - 层次化摘要
     - 大文档摘要、概览
     - O(log n)
   * - KnowledgeGraphIndex
     - 实体关系图
     - 结构化知识查询
     - 取决于图结构

VectorStoreIndex
================

最常用的索引类型，基于向量相似度进行检索。

基本用法
--------

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Document

   documents = [
       Document(text="人工智能是计算机科学的分支，研究如何让机器具有智能。"),
       Document(text="机器学习是人工智能的核心技术，通过数据训练模型。"),
       Document(text="深度学习使用多层神经网络，在图像和语言处理中表现出色。"),
   ]

   # 创建向量索引
   index = VectorStoreIndex.from_documents(documents)

   # 查询
   query_engine = index.as_query_engine(similarity_top_k=2)
   response = query_engine.query("什么是机器学习？")
   print(response)

配置选项
--------

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Settings
   from llama_index.embeddings.openai import OpenAIEmbedding

   # 配置嵌入模型
   Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

   # 创建索引时指定参数
   index = VectorStoreIndex.from_documents(
       documents,
       show_progress=True,      # 显示进度
       use_async=True           # 异步处理
   )

   # 查询时配置
   query_engine = index.as_query_engine(
       similarity_top_k=5,       # 返回前5个相似结果
       response_mode="compact",  # 紧凑模式
       streaming=True            # 流式输出
   )

SummaryIndex
============

遍历所有节点生成响应，适合需要全面理解文档的场景。

基本用法
--------

.. code-block:: python

   from llama_index.core import SummaryIndex, Document

   documents = [
       Document(text="第一章：介绍了项目背景和目标。"),
       Document(text="第二章：详细描述了技术方案。"),
       Document(text="第三章：展示了实验结果和分析。"),
   ]

   # 创建摘要索引
   index = SummaryIndex.from_documents(documents)

   # 查询会遍历所有节点
   query_engine = index.as_query_engine(
       response_mode="tree_summarize"  # 树形摘要模式
   )
   response = query_engine.query("总结这份文档的主要内容")
   print(response)

响应模式
--------

.. code-block:: python

   # 不同的响应模式
   # 1. refine - 逐步精炼答案
   query_engine = index.as_query_engine(response_mode="refine")

   # 2. compact - 合并上下文后生成
   query_engine = index.as_query_engine(response_mode="compact")

   # 3. tree_summarize - 层次化摘要
   query_engine = index.as_query_engine(response_mode="tree_summarize")

   # 4. simple_summarize - 简单合并
   query_engine = index.as_query_engine(response_mode="simple_summarize")

KeywordTableIndex
=================

基于关键词的精确匹配，适合需要精确搜索的场景。

基本用法
--------

.. code-block:: python

   from llama_index.core import KeywordTableIndex, Document

   documents = [
       Document(text="Python 是一种流行的编程语言，广泛用于数据科学。"),
       Document(text="JavaScript 主要用于前端开发和 Node.js 后端。"),
       Document(text="Go 语言以其高并发性能著称，适合系统编程。"),
   ]

   # 创建关键词索引
   index = KeywordTableIndex.from_documents(documents)

   # 查询
   query_engine = index.as_query_engine()
   response = query_engine.query("Python 有什么特点？")
   print(response)

自定义关键词提取
----------------

.. code-block:: python

   from llama_index.core import KeywordTableIndex
   from typing import Set

   def custom_keyword_extract(text: str) -> Set[str]:
       """自定义关键词提取函数"""
       # 简单示例：提取所有首字母大写的单词
       import re
       words = re.findall(r'\b[A-Z][a-z]+\b', text)
       return set(words)

   # 使用自定义提取器
   index = KeywordTableIndex.from_documents(
       documents,
       keyword_extract_fn=custom_keyword_extract
   )

TreeIndex
=========

构建层次化的树形结构，适合大文档的摘要和导航。

基本用法
--------

.. code-block:: python

   from llama_index.core import TreeIndex, Document

   # 准备长文档
   long_document = Document(text="""
   第一部分：项目概述
   本项目旨在构建智能知识管理系统...

   第二部分：技术架构
   系统采用微服务架构...

   第三部分：实现细节
   核心模块包括数据处理、索引构建...

   第四部分：测试与部署
   测试覆盖率达到90%...
   """)

   # 创建树形索引
   index = TreeIndex.from_documents(
       [long_document],
       num_children=3  # 每个节点的子节点数
   )

   # 查询
   query_engine = index.as_query_engine()
   response = query_engine.query("这个项目的技术架构是什么？")
   print(response)

树形结构说明
------------

.. code-block:: text

   TreeIndex 结构：

                     [Root 摘要]
                    /     |     \
                   /      |      \
            [摘要1]    [摘要2]   [摘要3]
            /    \      /  \       |
           /      \    /    \      |
        [叶1]   [叶2] [叶3] [叶4] [叶5]

   - 叶节点：原始文档片段
   - 中间节点：子节点的摘要
   - 根节点：整体摘要

KnowledgeGraphIndex
===================

构建知识图谱，支持实体和关系的查询。

基本用法
--------

.. code-block:: python

   from llama_index.core import KnowledgeGraphIndex, Document
   from llama_index.core import Settings
   from llama_index.llms.openai import OpenAI

   # 配置 LLM（用于提取实体和关系）
   Settings.llm = OpenAI(model="gpt-4o-mini")

   documents = [
       Document(text="张三是ABC公司的CEO，该公司位于北京。"),
       Document(text="ABC公司的主要产品是智能软件，李四是CTO。"),
       Document(text="李四毕业于清华大学，专注于人工智能研究。"),
   ]

   # 创建知识图谱索引
   index = KnowledgeGraphIndex.from_documents(
       documents,
       max_triplets_per_chunk=5,  # 每个文本块最多提取5个三元组
       include_embeddings=True     # 包含嵌入
   )

   # 查询
   query_engine = index.as_query_engine(
       include_text=True  # 包含原文
   )
   response = query_engine.query("张三和ABC公司是什么关系？")
   print(response)

可视化知识图谱
--------------

.. code-block:: python

   from pyvis.network import Network

   # 获取知识图谱数据
   g = index.get_networkx_graph()

   # 使用 pyvis 可视化
   net = Network(notebook=True, directed=True)
   net.from_nx(g)
   net.show("knowledge_graph.html")

DocumentSummaryIndex
====================

为每个文档生成摘要，支持基于摘要的检索。

.. code-block:: python

   from llama_index.core import DocumentSummaryIndex, Document

   documents = [
       Document(
           text="详细介绍了机器学习的基本概念、算法类型和应用场景...",
           metadata={"title": "机器学习入门"}
       ),
       Document(
           text="深度学习是机器学习的子集，使用神经网络进行学习...",
           metadata={"title": "深度学习概述"}
       ),
   ]

   # 创建文档摘要索引
   index = DocumentSummaryIndex.from_documents(documents)

   # 查看生成的摘要
   for doc_id, summary in index.get_document_summary_map().items():
       print(f"文档: {doc_id}")
       print(f"摘要: {summary}")
       print("---")

   # 查询
   query_engine = index.as_query_engine()
   response = query_engine.query("什么是深度学习？")
   print(response)

组合索引
========

ComposableGraph
---------------

将多个索引组合成一个可查询的图。

.. code-block:: python

   from llama_index.core import (
       VectorStoreIndex,
       SummaryIndex,
       Document,
       ServiceContext,
   )
   from llama_index.core.composability import ComposableGraph

   # 创建不同主题的索引
   ai_docs = [Document(text="人工智能相关内容...")]
   ml_docs = [Document(text="机器学习相关内容...")]

   ai_index = VectorStoreIndex.from_documents(ai_docs)
   ml_index = VectorStoreIndex.from_documents(ml_docs)

   # 创建根摘要
   ai_summary = "人工智能索引包含AI基础概念"
   ml_summary = "机器学习索引包含ML算法详解"

   # 组合成图
   graph = ComposableGraph.from_indices(
       SummaryIndex,
       [ai_index, ml_index],
       index_summaries=[ai_summary, ml_summary],
   )

   # 查询会自动路由到相关索引
   query_engine = graph.as_query_engine()
   response = query_engine.query("解释机器学习的基本原理")

RouterQueryEngine
-----------------

根据查询内容路由到不同的索引。

.. code-block:: python

   from llama_index.core.query_engine import RouterQueryEngine
   from llama_index.core.selectors import LLMSingleSelector
   from llama_index.core.tools import QueryEngineTool

   # 创建不同用途的查询引擎
   vector_engine = vector_index.as_query_engine()
   summary_engine = summary_index.as_query_engine()

   # 定义工具
   tools = [
       QueryEngineTool.from_defaults(
           query_engine=vector_engine,
           description="用于查找特定信息和回答具体问题"
       ),
       QueryEngineTool.from_defaults(
           query_engine=summary_engine,
           description="用于总结和概述整体内容"
       ),
   ]

   # 创建路由查询引擎
   router_engine = RouterQueryEngine(
       selector=LLMSingleSelector.from_defaults(),
       query_engine_tools=tools
   )

   # 查询会自动选择合适的引擎
   response = router_engine.query("请总结整个文档的内容")

索引选择指南
============

决策流程
--------

.. code-block:: text

   选择索引类型的决策树：

   开始
     │
     ▼
   需要语义搜索？ ──是──► VectorStoreIndex
     │
     否
     │
     ▼
   需要全文摘要？ ──是──► SummaryIndex
     │
     否
     │
     ▼
   需要精确匹配？ ──是──► KeywordTableIndex
     │
     否
     │
     ▼
   处理超大文档？ ──是──► TreeIndex
     │
     否
     │
     ▼
   需要关系查询？ ──是──► KnowledgeGraphIndex
     │
     否
     │
     ▼
   多源多类型？ ──是──► ComposableGraph + Router

实战示例
========

根据文档类型选择合适的索引。

.. code-block:: python

   from llama_index.core import (
       VectorStoreIndex,
       SummaryIndex,
       KeywordTableIndex,
       Document,
       Settings
   )
   from typing import List

   class SmartIndexManager:
       """智能索引管理器"""

       def __init__(self):
           self.indexes = {}

       def create_index(self, documents: List[Document], index_type: str = "auto"):
           """根据文档特征创建合适的索引"""
           if index_type == "auto":
               index_type = self._detect_best_index_type(documents)

           if index_type == "vector":
               index = VectorStoreIndex.from_documents(documents)
           elif index_type == "summary":
               index = SummaryIndex.from_documents(documents)
           elif index_type == "keyword":
               index = KeywordTableIndex.from_documents(documents)
           else:
               raise ValueError(f"Unknown index type: {index_type}")

           self.indexes[index_type] = index
           return index

       def _detect_best_index_type(self, documents: List[Document]) -> str:
           """自动检测最佳索引类型"""
           total_length = sum(len(doc.text) for doc in documents)
           num_docs = len(documents)

           # 简单启发式规则
           if num_docs == 1 and total_length > 10000:
               return "summary"  # 单个长文档用摘要索引
           elif total_length < 1000:
               return "keyword"  # 短文档用关键词索引
           else:
               return "vector"   # 默认使用向量索引

       def query(self, question: str, index_type: str = None):
           """查询指定或所有索引"""
           if index_type and index_type in self.indexes:
               engine = self.indexes[index_type].as_query_engine()
               return engine.query(question)

           # 查询所有索引
           results = {}
           for idx_type, index in self.indexes.items():
               engine = index.as_query_engine()
               results[idx_type] = engine.query(question)

           return results

   # 使用示例
   manager = SmartIndexManager()

   # 短文档
   short_docs = [Document(text="Python 是编程语言")]
   manager.create_index(short_docs)

   # 长文档
   long_docs = [Document(text="很长的技术文档..." * 100)]
   manager.create_index(long_docs)

   # 查询
   response = manager.query("什么是 Python？", "keyword")
   print(response)

小结
====

本教程介绍了：

- LlamaIndex 的各种索引类型及其特点
- VectorStoreIndex：语义相似度检索
- SummaryIndex：全文摘要
- KeywordTableIndex：关键词匹配
- TreeIndex：层次化结构
- KnowledgeGraphIndex：知识图谱
- 组合索引和路由查询
- 索引选择的最佳实践

下一步
------

在下一个教程中，我们将学习查询引擎（Query Engine），
了解如何优化查询过程以获得更好的回答。

练习
====

1. 为同一组文档创建不同类型的索引，比较查询效果
2. 使用 KnowledgeGraphIndex 构建一个小型知识图谱
3. 实现一个自动选择索引类型的系统
4. 使用 ComposableGraph 组合多个主题的索引
