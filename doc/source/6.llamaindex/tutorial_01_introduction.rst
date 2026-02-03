################################
Tutorial 1: LlamaIndex 入门
################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

.. contents:: 目录
   :local:
   :depth: 2

什么是 LlamaIndex
=================

LlamaIndex（原名 GPT Index）是一个专为大语言模型设计的数据框架，
它的核心目标是让 LLM 能够高效地访问和利用外部数据。

核心理念
--------

.. code-block:: text

   传统 LLM 应用的问题：

   ┌─────────────┐          ┌─────────────┐
   │   用户问题   │ ───────► │     LLM     │ ───► 可能产生幻觉
   └─────────────┘          └─────────────┘

   LlamaIndex 的解决方案：

   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │   用户问题   │───►│  检索相关   │───►│     LLM     │───► 基于事实的回答
   └─────────────┘    │  知识/文档  │    └─────────────┘
                      └─────────────┘
                            ▲
                            │
                      ┌─────────────┐
                      │  索引化的   │
                      │  知识库     │
                      └─────────────┘

LlamaIndex 的核心优势
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 优势
     - 说明
     - 应用场景
   * - 数据连接
     - 支持 100+ 种数据源
     - PDF、数据库、API、网页等
   * - 智能索引
     - 多种索引类型和策略
     - 向量索引、关键词索引、知识图谱
   * - 灵活查询
     - 可定制的查询引擎
     - 问答、摘要、对话
   * - 生产就绪
     - 完善的部署支持
     - 流式输出、可观测性

安装与配置
==========

基础安装
--------

.. code-block:: bash

   # 安装核心包
   pip install llama-index

   # 安装常用扩展
   pip install llama-index-llms-openai
   pip install llama-index-embeddings-openai
   pip install llama-index-vector-stores-chroma

环境配置
--------

.. code-block:: python

   import os
   from dotenv import load_dotenv

   # 加载环境变量
   load_dotenv()

   # 设置 OpenAI API Key
   os.environ["OPENAI_API_KEY"] = "your-api-key"

   # 或者使用其他 LLM
   os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

核心概念
========

Document（文档）
----------------

Document 是 LlamaIndex 中最基础的数据单元，代表一个完整的文档。

.. code-block:: python

   from llama_index.core import Document

   # 创建文档
   doc = Document(
       text="LlamaIndex 是一个强大的数据框架...",
       metadata={
           "source": "tutorial",
           "author": "Walter Fan",
           "date": "2024-01-01"
       }
   )

   print(f"文档内容: {doc.text[:50]}...")
   print(f"元数据: {doc.metadata}")

Node（节点）
------------

Node 是 Document 被分割后的更小单元，是索引和检索的基本单位。

.. code-block:: python

   from llama_index.core.node_parser import SentenceSplitter

   # 将文档分割为节点
   parser = SentenceSplitter(
       chunk_size=256,      # 每个节点的目标大小
       chunk_overlap=20     # 节点之间的重叠
   )

   nodes = parser.get_nodes_from_documents([doc])

   for i, node in enumerate(nodes):
       print(f"Node {i}: {node.text[:50]}...")

Index（索引）
-------------

Index 是组织和存储节点的数据结构，支持高效检索。

.. code-block:: python

   from llama_index.core import VectorStoreIndex

   # 从文档创建向量索引
   index = VectorStoreIndex.from_documents([doc])

   # 索引会自动：
   # 1. 将文档分割为节点
   # 2. 为每个节点生成嵌入向量
   # 3. 存储到向量数据库

Query Engine（查询引擎）
------------------------

Query Engine 负责处理用户查询，检索相关内容并生成回答。

.. code-block:: python

   # 创建查询引擎
   query_engine = index.as_query_engine()

   # 执行查询
   response = query_engine.query("LlamaIndex 有什么优势？")

   print(response)

第一个示例
==========

让我们创建一个完整的示例，从加载文档到回答问题。

准备数据
--------

.. code-block:: python

   # 准备示例文档
   documents = [
       Document(
           text="""
           LlamaIndex 是一个专为大语言模型设计的数据框架。
           它提供了数据连接、索引构建、查询优化等功能。
           主要特点包括：
           1. 支持多种数据源：PDF、Word、数据库、API等
           2. 灵活的索引类型：向量索引、列表索引、关键词索引等
           3. 强大的查询引擎：支持问答、摘要、对话等模式
           4. 生产就绪：支持流式输出、可观测性、缓存等
           """,
           metadata={"source": "introduction", "topic": "overview"}
       ),
       Document(
           text="""
           RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
           它的工作流程是：
           1. 接收用户问题
           2. 从知识库检索相关文档
           3. 将检索结果作为上下文提供给 LLM
           4. LLM 基于上下文生成回答
           LlamaIndex 是构建 RAG 应用的理想框架。
           """,
           metadata={"source": "rag", "topic": "technique"}
       )
   ]

构建索引
--------

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Settings
   from llama_index.llms.openai import OpenAI
   from llama_index.embeddings.openai import OpenAIEmbedding

   # 配置 LLM 和嵌入模型
   Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
   Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

   # 构建向量索引
   index = VectorStoreIndex.from_documents(documents)

   print("索引构建完成！")

执行查询
--------

.. code-block:: python

   # 创建查询引擎
   query_engine = index.as_query_engine(
       similarity_top_k=2,  # 检索 top 2 相关文档
       response_mode="compact"  # 紧凑模式生成回答
   )

   # 提问
   questions = [
       "LlamaIndex 的主要特点是什么？",
       "什么是 RAG？它是如何工作的？",
       "LlamaIndex 支持哪些数据源？"
   ]

   for q in questions:
       print(f"\n问题: {q}")
       response = query_engine.query(q)
       print(f"回答: {response}")
       print("-" * 50)

流式输出
--------

.. code-block:: python

   # 创建支持流式输出的查询引擎
   query_engine = index.as_query_engine(
       streaming=True,
       similarity_top_k=2
   )

   # 流式查询
   response = query_engine.query("详细介绍 LlamaIndex 的特点")

   # 逐步输出响应
   for text in response.response_gen:
       print(text, end="", flush=True)

查看检索结果
------------

.. code-block:: python

   # 获取详细的查询结果
   response = query_engine.query("LlamaIndex 的优势")

   # 查看来源节点
   print("检索到的相关内容:")
   for i, node in enumerate(response.source_nodes):
       print(f"\n--- 来源 {i+1} ---")
       print(f"内容: {node.text[:100]}...")
       print(f"相似度分数: {node.score:.4f}")
       print(f"元数据: {node.metadata}")

LlamaIndex vs LangChain
=======================

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 方面
     - LlamaIndex
     - LangChain
   * - 定位
     - 数据框架，专注 RAG
     - 通用 LLM 应用框架
   * - 数据处理
     - 强大的数据连接和索引
     - 基础的文档加载
   * - 检索策略
     - 丰富的检索和重排序
     - 需要额外配置
   * - Agent
     - 基础支持
     - 强大的 Agent 生态
   * - 学习曲线
     - 中等
     - 中等偏高
   * - 适用场景
     - 知识库问答、文档分析
     - 复杂 Agent、工作流

何时选择 LlamaIndex
-------------------

- 构建 RAG / 知识库问答系统
- 需要处理大量文档
- 需要高级检索策略
- 需要结构化数据提取

何时选择 LangChain
------------------

- 构建复杂 Agent 系统
- 需要灵活的工作流编排
- 需要丰富的工具集成
- 需要与多种 LLM 交互

小结
====

本教程介绍了：

- LlamaIndex 的核心理念和优势
- 基本安装和环境配置
- 核心概念：Document、Node、Index、Query Engine
- 第一个完整的问答示例
- LlamaIndex 与 LangChain 的对比

下一步
------

在下一个教程中，我们将深入学习 LlamaIndex 的数据加载功能，
了解如何从各种数据源（PDF、数据库、API 等）加载和处理文档。

练习
====

1. 尝试修改示例中的文档内容，观察查询结果的变化
2. 调整 ``similarity_top_k`` 参数，比较检索结果
3. 使用不同的 LLM（如 Anthropic Claude）运行示例
4. 尝试添加更多文档，构建一个小型知识库
