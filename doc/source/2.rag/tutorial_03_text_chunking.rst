####################################
Tutorial 3: 文本分块策略
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

为什么需要分块？
================

大语言模型有上下文长度限制，且过长的文本会影响检索效果。

.. code-block:: text

   问题:
   ┌─────────────────────────────────────────────────────────────┐
   │  一篇 10000 字的文档                                        │
   │                                                              │
   │  · 超出 LLM 上下文限制                                      │
   │  · 检索时整篇匹配，精度低                                   │
   │  · 包含太多无关信息                                         │
   └─────────────────────────────────────────────────────────────┘

   解决方案:
   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
   │ Chunk 1│  │ Chunk 2│  │ Chunk 3│  │ Chunk 4│  ...
   │ 500字  │  │ 500字  │  │ 500字  │  │ 500字  │
   └────────┘  └────────┘  └────────┘  └────────┘
   
   · 每个块独立索引
   · 精确检索相关段落
   · 适配 LLM 上下文

分块的关键参数
==============

1. **chunk_size**: 每个块的最大长度
2. **chunk_overlap**: 相邻块之间的重叠长度
3. **分隔符**: 在哪里切分文本

.. code-block:: text

   chunk_size = 100, chunk_overlap = 20

   原文: "这是一段很长的文本，需要被分割成多个小块来处理。每个块都有固定的大小。"

   Chunk 1: "这是一段很长的文本，需要被分割成多个小块来处理。"
   Chunk 2: "成多个小块来处理。每个块都有固定的大小。"
                 ↑
              重叠部分

基本分块方法
============

1. 按字符分块
-------------

.. code-block:: python

   from langchain.text_splitter import CharacterTextSplitter

   text = """
   人工智能（AI）是计算机科学的一个分支，它企图了解智能的实质，
   并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
   
   机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，
   而不需要被明确编程。深度学习是机器学习的一个子集，使用多层神经网络。
   """

   # 按字符分块
   splitter = CharacterTextSplitter(
       separator="\n\n",     # 首选分隔符
       chunk_size=100,       # 每块最大100字符
       chunk_overlap=20,     # 重叠20字符
       length_function=len,  # 长度计算函数
   )

   chunks = splitter.split_text(text)

   for i, chunk in enumerate(chunks):
       print(f"Chunk {i+1} ({len(chunk)} 字符):")
       print(f"  {chunk[:50]}...")
       print()

2. 递归字符分块（推荐）
-----------------------

.. code-block:: python

   from langchain.text_splitter import RecursiveCharacterTextSplitter

   # 递归分块：按优先级尝试不同分隔符
   splitter = RecursiveCharacterTextSplitter(
       separators=["\n\n", "\n", "。", "，", " ", ""],  # 分隔符优先级
       chunk_size=200,
       chunk_overlap=30,
       length_function=len,
   )

   chunks = splitter.split_text(text)

   print(f"分成了 {len(chunks)} 个块")
   for i, chunk in enumerate(chunks):
       print(f"\nChunk {i+1}:")
       print(chunk)

3. 按 Token 分块
----------------

.. code-block:: python

   from langchain.text_splitter import TokenTextSplitter

   # 按 Token 分块（更精确控制 LLM 输入长度）
   splitter = TokenTextSplitter(
       chunk_size=100,      # 100 tokens
       chunk_overlap=20,    # 20 tokens 重叠
   )

   chunks = splitter.split_text(text)

语义分块
========

基于语义边界进行分块，保持内容完整性。

1. 按句子分块
-------------

.. code-block:: python

   import re

   def split_by_sentences(text, max_sentences=5):
       """按句子分块"""
       # 中文句子分割
       sentences = re.split(r'([。！？])', text)
       
       # 重组句子（保留标点）
       full_sentences = []
       for i in range(0, len(sentences)-1, 2):
           if i+1 < len(sentences):
               full_sentences.append(sentences[i] + sentences[i+1])
           else:
               full_sentences.append(sentences[i])
       
       # 按固定句子数分块
       chunks = []
       for i in range(0, len(full_sentences), max_sentences):
           chunk = ''.join(full_sentences[i:i+max_sentences])
           if chunk.strip():
               chunks.append(chunk)
       
       return chunks

   chunks = split_by_sentences(text, max_sentences=3)

2. 按段落分块
-------------

.. code-block:: python

   def split_by_paragraphs(text, max_paragraphs=2):
       """按段落分块"""
       paragraphs = text.split('\n\n')
       paragraphs = [p.strip() for p in paragraphs if p.strip()]
       
       chunks = []
       for i in range(0, len(paragraphs), max_paragraphs):
           chunk = '\n\n'.join(paragraphs[i:i+max_paragraphs])
           chunks.append(chunk)
       
       return chunks

3. 基于嵌入的语义分块
---------------------

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   import numpy as np

   def semantic_chunking(text, similarity_threshold=0.7):
       """基于语义相似度的分块"""
       # 先按句子分割
       sentences = re.split(r'([。！？])', text)
       full_sentences = []
       for i in range(0, len(sentences)-1, 2):
           if i+1 < len(sentences):
               full_sentences.append(sentences[i] + sentences[i+1])
       
       if not full_sentences:
           return [text]
       
       # 计算句子嵌入
       model = SentenceTransformer('all-MiniLM-L6-v2')
       embeddings = model.encode(full_sentences)
       
       # 基于相似度分块
       chunks = []
       current_chunk = [full_sentences[0]]
       
       for i in range(1, len(full_sentences)):
           # 计算与当前块最后一句的相似度
           sim = np.dot(embeddings[i], embeddings[i-1]) / (
               np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
           )
           
           if sim >= similarity_threshold:
               current_chunk.append(full_sentences[i])
           else:
               chunks.append(''.join(current_chunk))
               current_chunk = [full_sentences[i]]
       
       if current_chunk:
           chunks.append(''.join(current_chunk))
       
       return chunks

特定格式分块
============

1. Markdown 分块
----------------

.. code-block:: python

   from langchain.text_splitter import MarkdownTextSplitter

   markdown_text = """
   # 第一章 人工智能简介

   人工智能是一门研究如何使计算机能够完成智能任务的学科。

   ## 1.1 什么是人工智能

   人工智能（AI）是指由人制造的机器所表现出来的智能。

   ## 1.2 AI 的历史

   AI 的概念最早在 1956 年达特茅斯会议上提出。

   # 第二章 机器学习

   机器学习是 AI 的一个重要分支。
   """

   splitter = MarkdownTextSplitter(
       chunk_size=200,
       chunk_overlap=20
   )

   chunks = splitter.split_text(markdown_text)

   for i, chunk in enumerate(chunks):
       print(f"Chunk {i+1}:")
       print(chunk)
       print("-" * 40)

2. 代码分块
-----------

.. code-block:: python

   from langchain.text_splitter import (
       RecursiveCharacterTextSplitter,
       Language
   )

   python_code = '''
   def hello_world():
       """打印 Hello World"""
       print("Hello, World!")

   class Calculator:
       """简单计算器"""
       
       def add(self, a, b):
           return a + b
       
       def subtract(self, a, b):
           return a - b
   '''

   # Python 代码分块
   splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.PYTHON,
       chunk_size=200,
       chunk_overlap=20
   )

   chunks = splitter.split_text(python_code)

3. HTML 分块
------------

.. code-block:: python

   from langchain.text_splitter import HTMLHeaderTextSplitter

   html_content = """
   <html>
   <body>
   <h1>人工智能</h1>
   <p>人工智能是计算机科学的一个分支。</p>
   <h2>机器学习</h2>
   <p>机器学习是 AI 的子领域。</p>
   <h2>深度学习</h2>
   <p>深度学习使用神经网络。</p>
   </body>
   </html>
   """

   headers_to_split_on = [
       ("h1", "Header 1"),
       ("h2", "Header 2"),
   ]

   splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
   chunks = splitter.split_text(html_content)

实战：智能分块器
================

.. code-block:: python

   from langchain.text_splitter import RecursiveCharacterTextSplitter
   from langchain.schema import Document
   from typing import List
   import re

   class SmartChunker:
       """智能文本分块器"""
       
       def __init__(
           self,
           chunk_size: int = 500,
           chunk_overlap: int = 50,
           min_chunk_size: int = 100
       ):
           self.chunk_size = chunk_size
           self.chunk_overlap = chunk_overlap
           self.min_chunk_size = min_chunk_size
           
           # 中文分隔符
           self.chinese_separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]
           
           # 英文分隔符
           self.english_separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
       
       def _detect_language(self, text: str) -> str:
           """检测文本语言"""
           chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
           total_chars = len(text.replace(" ", ""))
           
           if total_chars == 0:
               return "english"
           
           return "chinese" if chinese_chars / total_chars > 0.3 else "english"
       
       def _get_separators(self, language: str) -> List[str]:
           """获取分隔符"""
           return self.chinese_separators if language == "chinese" else self.english_separators
       
       def chunk_text(self, text: str) -> List[str]:
           """分块文本"""
           language = self._detect_language(text)
           separators = self._get_separators(language)
           
           splitter = RecursiveCharacterTextSplitter(
               separators=separators,
               chunk_size=self.chunk_size,
               chunk_overlap=self.chunk_overlap,
               length_function=len
           )
           
           chunks = splitter.split_text(text)
           
           # 过滤太短的块
           chunks = [c for c in chunks if len(c) >= self.min_chunk_size]
           
           return chunks
       
       def chunk_documents(self, documents: List[Document]) -> List[Document]:
           """分块文档列表"""
           chunked_docs = []
           
           for doc in documents:
               chunks = self.chunk_text(doc.page_content)
               
               for i, chunk in enumerate(chunks):
                   chunked_doc = Document(
                       page_content=chunk,
                       metadata={
                           **doc.metadata,
                           "chunk_index": i,
                           "total_chunks": len(chunks),
                           "chunk_size": len(chunk)
                       }
                   )
                   chunked_docs.append(chunked_doc)
           
           return chunked_docs

   # 使用智能分块器
   chunker = SmartChunker(chunk_size=300, chunk_overlap=30)

   # 测试中文
   chinese_text = """
   人工智能是计算机科学的一个重要分支，它致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。
   
   机器学习是人工智能的核心技术之一，它使计算机系统能够从数据中自动学习和改进，而无需进行明确的编程。
   
   深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的复杂表示和特征。
   """

   chunks = chunker.chunk_text(chinese_text)
   print(f"中文文本分成 {len(chunks)} 个块")
   for i, chunk in enumerate(chunks):
       print(f"\nChunk {i+1} ({len(chunk)} 字符):")
       print(chunk)

分块策略选择指南
================

.. csv-table::
   :header: "场景", "推荐策略", "chunk_size", "overlap"
   :widths: 25, 25, 25, 25

   "通用问答", "递归字符分块", "500-1000", "50-100"
   "代码检索", "语言感知分块", "1000-2000", "100-200"
   "文档摘要", "段落分块", "1000-2000", "100"
   "精确检索", "句子分块", "200-500", "20-50"
   "长文档", "语义分块", "500-1000", "50-100"

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "Chunk Size", "每个文本块的最大长度"
   "Overlap", "相邻块之间的重叠部分"
   "Separator", "用于分割文本的分隔符"
   "递归分块", "按优先级尝试不同分隔符"
   "语义分块", "基于语义边界进行分块"

下一步
======

在下一个教程中，我们将学习向量嵌入技术。

:doc:`tutorial_04_embeddings`
