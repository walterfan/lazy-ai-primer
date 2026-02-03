################################
Tutorial 3: 节点解析
################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

.. contents:: 目录
   :local:
   :depth: 2

节点解析概述
============

节点解析（Node Parsing）是将文档分割成更小单元的过程。
好的分割策略直接影响检索质量和最终回答的准确性。

.. code-block:: text

   节点解析流程：

   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  Document   │───►│  Parser     │───►│   Nodes     │
   │  (完整文档) │    │  (解析器)   │    │  (文本块)   │
   └─────────────┘    └─────────────┘    └─────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │  关系建立   │
                      │  (前后节点) │
                      └─────────────┘

为什么需要节点解析
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 原因
     - 说明
   * - 嵌入限制
     - 嵌入模型有最大 token 限制（如 8192）
   * - 上下文限制
     - LLM 上下文窗口有限
   * - 检索精度
     - 小块更容易精确匹配查询
   * - 成本控制
     - 减少不必要的 token 消耗

节点（Node）结构
================

基本属性
--------

.. code-block:: python

   from llama_index.core.schema import TextNode

   # 创建节点
   node = TextNode(
       text="这是节点的文本内容...",
       id_="node_001",
       metadata={
           "source": "document.pdf",
           "page": 1,
           "chunk_index": 0
       }
   )

   print(f"节点 ID: {node.id_}")
   print(f"文本长度: {len(node.text)}")
   print(f"元数据: {node.metadata}")

节点关系
--------

.. code-block:: python

   from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

   # 节点之间可以有父子、前后关系
   node1 = TextNode(text="第一段内容...", id_="node_1")
   node2 = TextNode(text="第二段内容...", id_="node_2")

   # 建立前后关系
   node1.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
       node_id=node2.id_
   )
   node2.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
       node_id=node1.id_
   )

   print(f"Node1 的下一个节点: {node1.relationships.get(NodeRelationship.NEXT)}")

SentenceSplitter
================

最常用的解析器，基于句子边界分割文本。

基本用法
--------

.. code-block:: python

   from llama_index.core.node_parser import SentenceSplitter

   # 创建解析器
   splitter = SentenceSplitter(
       chunk_size=256,      # 目标块大小（字符数）
       chunk_overlap=20     # 块之间的重叠
   )

   # 解析文档
   from llama_index.core import Document

   doc = Document(text="""
   人工智能是计算机科学的一个分支。它试图理解智能的实质。
   机器学习是人工智能的核心。深度学习是机器学习的子集。
   自然语言处理让机器理解人类语言。计算机视觉让机器看懂图像。
   这些技术正在改变我们的生活方式。未来将有更多应用场景。
   """)

   nodes = splitter.get_nodes_from_documents([doc])

   for i, node in enumerate(nodes):
       print(f"\n--- Node {i} ---")
       print(f"内容: {node.text}")
       print(f"长度: {len(node.text)}")

配置选项
--------

.. code-block:: python

   splitter = SentenceSplitter(
       chunk_size=512,              # 块大小
       chunk_overlap=50,            # 重叠大小
       separator=" ",               # 主要分隔符
       paragraph_separator="\n\n",  # 段落分隔符
       secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",  # 二级分割正则
   )

TokenTextSplitter
=================

基于 token 数量分割，更精确控制大小。

.. code-block:: python

   from llama_index.core.node_parser import TokenTextSplitter

   # 基于 token 分割
   token_splitter = TokenTextSplitter(
       chunk_size=256,       # token 数量
       chunk_overlap=20,     # 重叠 token 数
       separator=" "
   )

   nodes = token_splitter.get_nodes_from_documents([doc])

   for i, node in enumerate(nodes):
       print(f"Node {i}: {len(node.text)} chars")

SemanticSplitter
================

基于语义相似度分割，保持语义完整性。

.. code-block:: python

   # pip install llama-index-embeddings-openai

   from llama_index.core.node_parser import SemanticSplitterNodeParser
   from llama_index.embeddings.openai import OpenAIEmbedding

   # 创建语义分割器
   embed_model = OpenAIEmbedding()
   semantic_splitter = SemanticSplitterNodeParser(
       buffer_size=1,           # 上下文缓冲
       breakpoint_percentile_threshold=95,  # 分割阈值
       embed_model=embed_model
   )

   nodes = semantic_splitter.get_nodes_from_documents([doc])

   print(f"语义分割产生 {len(nodes)} 个节点")

工作原理
--------

.. code-block:: text

   语义分割原理：

   1. 将文本按句子切分
   2. 计算相邻句子的嵌入向量
   3. 计算相邻句子的相似度
   4. 在相似度低于阈值的位置分割

   ┌────────────────────────────────────────────────┐
   │  句子1  │  句子2  │  句子3  │  句子4  │  句子5  │
   └────────────────────────────────────────────────┘
        ↓         ↓         ↓         ↓
      0.95      0.92      0.45      0.91    相似度
                           ↓
                      低于阈值，分割点

MarkdownNodeParser
==================

专门为 Markdown 文档设计的解析器。

.. code-block:: python

   from llama_index.core.node_parser import MarkdownNodeParser

   md_doc = Document(text="""
   # 第一章：简介

   这是简介部分的内容。介绍了基本概念。

   ## 1.1 背景

   背景信息说明。

   ## 1.2 目标

   目标描述。

   # 第二章：方法

   方法论述。
   """)

   md_parser = MarkdownNodeParser()
   nodes = md_parser.get_nodes_from_documents([md_doc])

   for node in nodes:
       print(f"标题级别: {node.metadata.get('header_path', 'N/A')}")
       print(f"内容: {node.text[:50]}...")
       print("---")

HTMLNodeParser
==============

解析 HTML 文档，保留结构信息。

.. code-block:: python

   from llama_index.core.node_parser import HTMLNodeParser

   html_doc = Document(text="""
   <html>
   <body>
   <h1>主标题</h1>
   <p>这是第一段内容。</p>
   <h2>子标题</h2>
   <p>这是第二段内容。</p>
   <ul>
       <li>列表项1</li>
       <li>列表项2</li>
   </ul>
   </body>
   </html>
   """)

   html_parser = HTMLNodeParser(
       tags=["p", "h1", "h2", "li"]  # 要提取的标签
   )
   nodes = html_parser.get_nodes_from_documents([html_doc])

   for node in nodes:
       print(f"标签: {node.metadata.get('tag', 'N/A')}")
       print(f"内容: {node.text}")

CodeSplitter
============

专门处理代码文件的解析器。

.. code-block:: python

   from llama_index.core.node_parser import CodeSplitter

   code_doc = Document(text='''
   def hello_world():
       """打印问候语"""
       print("Hello, World!")

   class Calculator:
       """简单计算器类"""

       def add(self, a, b):
           """加法"""
           return a + b

       def subtract(self, a, b):
           """减法"""
           return a - b

   def main():
       calc = Calculator()
       print(calc.add(1, 2))
   ''')

   code_splitter = CodeSplitter(
       language="python",
       chunk_lines=10,        # 每块的行数
       chunk_lines_overlap=2, # 重叠行数
       max_chars=1000         # 最大字符数
   )

   nodes = code_splitter.get_nodes_from_documents([code_doc])

   for i, node in enumerate(nodes):
       print(f"\n--- Code Block {i} ---")
       print(node.text)

HierarchicalNodeParser
======================

创建层次化的节点结构，支持父子关系。

.. code-block:: python

   from llama_index.core.node_parser import HierarchicalNodeParser

   # 创建层次化解析器
   hierarchical_parser = HierarchicalNodeParser.from_defaults(
       chunk_sizes=[2048, 512, 128]  # 从大到小的层次
   )

   nodes = hierarchical_parser.get_nodes_from_documents([doc])

   # 查看节点层次
   for node in nodes:
       parent_ref = node.relationships.get(NodeRelationship.PARENT)
       child_refs = node.relationships.get(NodeRelationship.CHILD, [])
       print(f"Node: {node.id_[:8]}...")
       print(f"  Parent: {parent_ref.node_id[:8] if parent_ref else 'None'}...")
       print(f"  Children: {len(child_refs) if isinstance(child_refs, list) else 0}")

组合使用
========

.. code-block:: python

   from llama_index.core.node_parser import (
       SentenceSplitter,
       MarkdownNodeParser,
       CodeSplitter
   )
   from llama_index.core import Document

   class SmartNodeParser:
       """根据文档类型选择合适的解析器"""

       def __init__(self):
           self.parsers = {
               "markdown": MarkdownNodeParser(),
               "code": CodeSplitter(language="python"),
               "default": SentenceSplitter(chunk_size=512)
           }

       def parse(self, doc: Document) -> list:
           doc_type = doc.metadata.get("type", "default")
           parser = self.parsers.get(doc_type, self.parsers["default"])
           return parser.get_nodes_from_documents([doc])

   # 使用示例
   smart_parser = SmartNodeParser()

   # Markdown 文档
   md_doc = Document(
       text="# Title\n\nContent here.",
       metadata={"type": "markdown"}
   )

   # 代码文档
   code_doc = Document(
       text="def foo(): pass",
       metadata={"type": "code"}
   )

   md_nodes = smart_parser.parse(md_doc)
   code_nodes = smart_parser.parse(code_doc)

最佳实践
========

分块大小选择
------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - 场景
     - 推荐大小
     - 说明
   * - 问答系统
     - 256-512
     - 精确检索，减少噪音
   * - 摘要生成
     - 1024-2048
     - 保持上下文完整性
   * - 代码分析
     - 按函数/类分割
     - 保持代码逻辑完整
   * - 长文档
     - 层次化分割
     - 多粒度检索

重叠策略
--------

.. code-block:: python

   # 不同重叠策略的效果

   # 1. 无重叠 - 可能丢失边界信息
   no_overlap = SentenceSplitter(chunk_size=256, chunk_overlap=0)

   # 2. 小重叠 - 基本连续性
   small_overlap = SentenceSplitter(chunk_size=256, chunk_overlap=20)

   # 3. 大重叠 - 更好的上下文，但增加冗余
   large_overlap = SentenceSplitter(chunk_size=256, chunk_overlap=50)

元数据保留
----------

.. code-block:: python

   from llama_index.core.node_parser import SentenceSplitter

   splitter = SentenceSplitter(
       chunk_size=512,
       include_metadata=True,     # 保留文档元数据
       include_prev_next_rel=True # 保留前后关系
   )

   # 自定义元数据处理
   def process_nodes(nodes, additional_metadata):
       for i, node in enumerate(nodes):
           node.metadata.update(additional_metadata)
           node.metadata["chunk_index"] = i
           node.metadata["total_chunks"] = len(nodes)
       return nodes

实战示例
========

构建一个完整的文档处理管道。

.. code-block:: python

   from llama_index.core import Document, VectorStoreIndex, Settings
   from llama_index.core.node_parser import (
       SentenceSplitter,
       MarkdownNodeParser,
   )
   from typing import List

   class DocumentProcessor:
       """文档处理器"""

       def __init__(self):
           self.sentence_splitter = SentenceSplitter(
               chunk_size=512,
               chunk_overlap=50
           )
           self.md_parser = MarkdownNodeParser()

       def detect_doc_type(self, doc: Document) -> str:
           """检测文档类型"""
           filename = doc.metadata.get("file_name", "")
           if filename.endswith(".md"):
               return "markdown"
           elif filename.endswith((".py", ".js", ".java")):
               return "code"
           return "text"

       def process(self, documents: List[Document]) -> List:
           """处理文档列表"""
           all_nodes = []

           for doc in documents:
               doc_type = self.detect_doc_type(doc)

               if doc_type == "markdown":
                   nodes = self.md_parser.get_nodes_from_documents([doc])
               else:
                   nodes = self.sentence_splitter.get_nodes_from_documents([doc])

               # 添加处理元数据
               for node in nodes:
                   node.metadata["doc_type"] = doc_type

               all_nodes.extend(nodes)

           return all_nodes

   # 使用处理器
   processor = DocumentProcessor()

   documents = [
       Document(
           text="# Guide\n\nThis is a guide.",
           metadata={"file_name": "guide.md"}
       ),
       Document(
           text="Regular text content here.",
           metadata={"file_name": "readme.txt"}
       )
   ]

   nodes = processor.process(documents)
   print(f"处理后得到 {len(nodes)} 个节点")

   # 构建索引
   index = VectorStoreIndex(nodes)

小结
====

本教程介绍了：

- 节点解析的概念和重要性
- 各种解析器：SentenceSplitter、TokenTextSplitter、SemanticSplitter 等
- 特定格式解析器：Markdown、HTML、Code
- 层次化解析器的使用
- 分块大小和重叠策略的选择
- 完整的文档处理管道

下一步
------

在下一个教程中，我们将学习嵌入和向量存储，
了解如何将节点转换为向量并高效存储。

练习
====

1. 比较 SentenceSplitter 和 SemanticSplitter 的分割效果
2. 为不同类型的文档选择合适的解析器
3. 实现一个自定义的解析器
4. 测试不同 chunk_size 对检索效果的影响
