####################################
Tutorial 4: 嵌入与向量存储
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

.. contents:: 目录
   :local:
   :depth: 2

嵌入（Embeddings）概述
======================

嵌入是将文本转换为高维向量的过程，使得语义相似的文本在向量空间中距离更近。

.. code-block:: text

   嵌入过程：

   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │    文本     │───►│  嵌入模型   │───►│   向量      │
   │  "AI很棒"   │    │  (Encoder)  │    │ [0.1, 0.3,  │
   └─────────────┘    └─────────────┘    │  -0.2, ...] │
                                         └─────────────┘

   相似文本产生相近的向量：
   "AI很棒" ──► [0.1, 0.3, -0.2]
   "人工智能很好" ──► [0.12, 0.28, -0.18]  ← 向量接近
   "今天天气好" ──► [0.8, -0.1, 0.5]       ← 向量远离

嵌入模型选择
============

OpenAI Embeddings
-----------------

.. code-block:: python

   # pip install llama-index-embeddings-openai

   from llama_index.embeddings.openai import OpenAIEmbedding

   # 创建嵌入模型
   embed_model = OpenAIEmbedding(
       model="text-embedding-3-small",  # 或 text-embedding-3-large
       api_key="your-api-key"           # 可选，默认使用环境变量
   )

   # 生成嵌入
   text = "LlamaIndex 是一个强大的数据框架"
   embedding = embed_model.get_text_embedding(text)

   print(f"嵌入维度: {len(embedding)}")  # 1536 或 3072
   print(f"向量前5维: {embedding[:5]}")

HuggingFace Embeddings
----------------------

.. code-block:: python

   # pip install llama-index-embeddings-huggingface

   from llama_index.embeddings.huggingface import HuggingFaceEmbedding

   # 使用开源模型（本地运行）
   embed_model = HuggingFaceEmbedding(
       model_name="BAAI/bge-small-zh-v1.5"  # 中文模型
   )

   embedding = embed_model.get_text_embedding("测试文本")
   print(f"嵌入维度: {len(embedding)}")

Ollama Embeddings
-----------------

.. code-block:: python

   # pip install llama-index-embeddings-ollama

   from llama_index.embeddings.ollama import OllamaEmbedding

   # 使用本地 Ollama 服务
   embed_model = OllamaEmbedding(
       model_name="nomic-embed-text",
       base_url="http://localhost:11434"
   )

   embedding = embed_model.get_text_embedding("本地嵌入测试")

嵌入模型对比
------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - 模型
     - 维度
     - 语言支持
     - 速度
     - 成本
   * - text-embedding-3-small
     - 1536
     - 多语言
     - 快
     - 低
   * - text-embedding-3-large
     - 3072
     - 多语言
     - 中
     - 中
   * - bge-small-zh
     - 512
     - 中文
     - 很快
     - 免费
   * - bge-large-zh
     - 1024
     - 中文
     - 中
     - 免费

全局配置嵌入模型
----------------

.. code-block:: python

   from llama_index.core import Settings
   from llama_index.embeddings.openai import OpenAIEmbedding

   # 全局设置嵌入模型
   Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

   # 之后创建的所有索引都会使用这个嵌入模型

向量存储（Vector Store）
========================

向量存储用于高效存储和检索向量。LlamaIndex 支持多种向量数据库。

内存存储（默认）
----------------

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Document

   # 默认使用内存存储
   documents = [Document(text="示例文档内容")]
   index = VectorStoreIndex.from_documents(documents)

   # 适合：原型开发、小数据量
   # 不适合：生产环境、大数据量

Chroma
------

.. code-block:: python

   # pip install llama-index-vector-stores-chroma chromadb

   import chromadb
   from llama_index.vector_stores.chroma import ChromaVectorStore
   from llama_index.core import VectorStoreIndex, StorageContext

   # 创建 Chroma 客户端
   chroma_client = chromadb.PersistentClient(path="./chroma_db")

   # 创建或获取集合
   chroma_collection = chroma_client.get_or_create_collection("my_collection")

   # 创建向量存储
   vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

   # 创建存储上下文
   storage_context = StorageContext.from_defaults(vector_store=vector_store)

   # 创建索引
   index = VectorStoreIndex.from_documents(
       documents,
       storage_context=storage_context
   )

   # 持久化后，下次可以直接加载
   # index = VectorStoreIndex.from_vector_store(vector_store)

FAISS
-----

.. code-block:: python

   # pip install llama-index-vector-stores-faiss faiss-cpu

   import faiss
   from llama_index.vector_stores.faiss import FaissVectorStore
   from llama_index.core import VectorStoreIndex, StorageContext

   # 创建 FAISS 索引
   d = 1536  # 向量维度（取决于嵌入模型）
   faiss_index = faiss.IndexFlatL2(d)

   # 创建向量存储
   vector_store = FaissVectorStore(faiss_index=faiss_index)
   storage_context = StorageContext.from_defaults(vector_store=vector_store)

   # 创建索引
   index = VectorStoreIndex.from_documents(
       documents,
       storage_context=storage_context
   )

   # 保存到磁盘
   index.storage_context.persist("./faiss_storage")

   # 加载
   # storage_context = StorageContext.from_defaults(persist_dir="./faiss_storage")
   # index = load_index_from_storage(storage_context)

Pinecone
--------

.. code-block:: python

   # pip install llama-index-vector-stores-pinecone pinecone-client

   from pinecone import Pinecone
   from llama_index.vector_stores.pinecone import PineconeVectorStore
   from llama_index.core import VectorStoreIndex, StorageContext

   # 初始化 Pinecone
   pc = Pinecone(api_key="your-pinecone-api-key")
   pinecone_index = pc.Index("your-index-name")

   # 创建向量存储
   vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
   storage_context = StorageContext.from_defaults(vector_store=vector_store)

   # 创建索引
   index = VectorStoreIndex.from_documents(
       documents,
       storage_context=storage_context
   )

Milvus
------

.. code-block:: python

   # pip install llama-index-vector-stores-milvus pymilvus

   from llama_index.vector_stores.milvus import MilvusVectorStore
   from llama_index.core import VectorStoreIndex, StorageContext

   # 连接 Milvus
   vector_store = MilvusVectorStore(
       uri="http://localhost:19530",
       collection_name="my_collection",
       dim=1536
   )

   storage_context = StorageContext.from_defaults(vector_store=vector_store)

   index = VectorStoreIndex.from_documents(
       documents,
       storage_context=storage_context
   )

向量数据库对比
--------------

.. list-table::
   :header-rows: 1
   :widths: 15 17 17 17 17 17

   * - 数据库
     - 部署
     - 扩展性
     - 特点
     - 适用场景
     - 成本
   * - Chroma
     - 简单
     - 中
     - 轻量易用
     - 原型/中小项目
     - 免费
   * - FAISS
     - 简单
     - 高
     - 高性能
     - 大规模检索
     - 免费
   * - Pinecone
     - 云服务
     - 很高
     - 托管服务
     - 生产环境
     - 按用量
   * - Milvus
     - 中等
     - 很高
     - 分布式
     - 企业级
     - 免费/企业版

相似度计算
==========

相似度类型
----------

.. code-block:: python

   from llama_index.core.vector_stores.types import VectorStoreQueryMode

   # 不同的相似度度量
   # 1. 余弦相似度（默认）
   # 2. 欧氏距离
   # 3. 点积

   # 手动计算相似度
   import numpy as np

   def cosine_similarity(v1, v2):
       """余弦相似度"""
       return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

   def euclidean_distance(v1, v2):
       """欧氏距离"""
       return np.linalg.norm(np.array(v1) - np.array(v2))

   def dot_product(v1, v2):
       """点积"""
       return np.dot(v1, v2)

   # 示例
   v1 = [0.1, 0.2, 0.3]
   v2 = [0.15, 0.25, 0.35]

   print(f"余弦相似度: {cosine_similarity(v1, v2):.4f}")
   print(f"欧氏距离: {euclidean_distance(v1, v2):.4f}")
   print(f"点积: {dot_product(v1, v2):.4f}")

批量嵌入
========

处理大量文档时，批量嵌入更高效。

.. code-block:: python

   from llama_index.embeddings.openai import OpenAIEmbedding

   embed_model = OpenAIEmbedding()

   # 批量嵌入
   texts = [
       "第一段文本",
       "第二段文本",
       "第三段文本"
   ]

   embeddings = embed_model.get_text_embedding_batch(texts)

   for i, emb in enumerate(embeddings):
       print(f"文本 {i+1} 嵌入维度: {len(emb)}")

嵌入缓存
========

避免重复计算相同文本的嵌入。

.. code-block:: python

   from llama_index.core.embeddings import resolve_embed_model
   from llama_index.core import Settings
   import hashlib
   import json
   import os

   class CachedEmbedding:
       """带缓存的嵌入模型包装器"""

       def __init__(self, embed_model, cache_dir="./embed_cache"):
           self.embed_model = embed_model
           self.cache_dir = cache_dir
           os.makedirs(cache_dir, exist_ok=True)

       def _get_cache_key(self, text: str) -> str:
           return hashlib.md5(text.encode()).hexdigest()

       def _get_cache_path(self, key: str) -> str:
           return os.path.join(self.cache_dir, f"{key}.json")

       def get_text_embedding(self, text: str) -> list:
           cache_key = self._get_cache_key(text)
           cache_path = self._get_cache_path(cache_key)

           # 检查缓存
           if os.path.exists(cache_path):
               with open(cache_path, 'r') as f:
                   return json.load(f)

           # 计算嵌入
           embedding = self.embed_model.get_text_embedding(text)

           # 保存缓存
           with open(cache_path, 'w') as f:
               json.dump(embedding, f)

           return embedding

   # 使用缓存嵌入
   cached_embed = CachedEmbedding(OpenAIEmbedding())
   emb1 = cached_embed.get_text_embedding("测试文本")  # 首次计算
   emb2 = cached_embed.get_text_embedding("测试文本")  # 使用缓存

索引持久化
==========

保存和加载索引。

保存索引
--------

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Document, StorageContext

   # 创建索引
   documents = [
       Document(text="文档1内容"),
       Document(text="文档2内容")
   ]
   index = VectorStoreIndex.from_documents(documents)

   # 持久化到磁盘
   index.storage_context.persist(persist_dir="./storage")

加载索引
--------

.. code-block:: python

   from llama_index.core import StorageContext, load_index_from_storage

   # 从磁盘加载
   storage_context = StorageContext.from_defaults(persist_dir="./storage")
   index = load_index_from_storage(storage_context)

   # 使用索引
   query_engine = index.as_query_engine()
   response = query_engine.query("查询问题")

增量更新
--------

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Document

   # 加载现有索引
   storage_context = StorageContext.from_defaults(persist_dir="./storage")
   index = load_index_from_storage(storage_context)

   # 添加新文档
   new_doc = Document(text="新文档内容")
   index.insert(new_doc)

   # 保存更新
   index.storage_context.persist(persist_dir="./storage")

实战示例
========

构建一个完整的向量知识库。

.. code-block:: python

   import os
   from llama_index.core import (
       VectorStoreIndex,
       SimpleDirectoryReader,
       StorageContext,
       Settings
   )
   from llama_index.embeddings.openai import OpenAIEmbedding
   from llama_index.vector_stores.chroma import ChromaVectorStore
   import chromadb

   class VectorKnowledgeBase:
       """向量知识库"""

       def __init__(self, persist_dir: str = "./kb_storage"):
           self.persist_dir = persist_dir

           # 配置嵌入模型
           Settings.embed_model = OpenAIEmbedding(
               model="text-embedding-3-small"
           )

           # 初始化 Chroma
           self.chroma_client = chromadb.PersistentClient(
               path=os.path.join(persist_dir, "chroma")
           )

           self.index = None

       def create_index(self, documents_dir: str, collection_name: str = "default"):
           """从目录创建索引"""
           # 加载文档
           reader = SimpleDirectoryReader(documents_dir, recursive=True)
           documents = reader.load_data()
           print(f"加载了 {len(documents)} 个文档")

           # 创建向量存储
           collection = self.chroma_client.get_or_create_collection(collection_name)
           vector_store = ChromaVectorStore(chroma_collection=collection)
           storage_context = StorageContext.from_defaults(vector_store=vector_store)

           # 创建索引
           self.index = VectorStoreIndex.from_documents(
               documents,
               storage_context=storage_context,
               show_progress=True
           )

           print("索引创建完成")
           return self.index

       def load_index(self, collection_name: str = "default"):
           """加载现有索引"""
           collection = self.chroma_client.get_collection(collection_name)
           vector_store = ChromaVectorStore(chroma_collection=collection)
           self.index = VectorStoreIndex.from_vector_store(vector_store)
           print("索引加载完成")
           return self.index

       def add_documents(self, documents):
           """添加文档到索引"""
           if self.index is None:
               raise ValueError("请先创建或加载索引")

           for doc in documents:
               self.index.insert(doc)

           print(f"添加了 {len(documents)} 个文档")

       def query(self, question: str, top_k: int = 5):
           """查询知识库"""
           if self.index is None:
               raise ValueError("请先创建或加载索引")

           query_engine = self.index.as_query_engine(
               similarity_top_k=top_k
           )

           response = query_engine.query(question)
           return response

       def search(self, query: str, top_k: int = 5):
           """搜索相关文档（不生成回答）"""
           if self.index is None:
               raise ValueError("请先创建或加载索引")

           retriever = self.index.as_retriever(similarity_top_k=top_k)
           nodes = retriever.retrieve(query)

           results = []
           for node in nodes:
               results.append({
                   "text": node.text,
                   "score": node.score,
                   "metadata": node.metadata
               })

           return results

   # 使用示例
   kb = VectorKnowledgeBase("./my_knowledge_base")

   # 创建索引
   kb.create_index("./documents")

   # 查询
   response = kb.query("什么是人工智能？")
   print(response)

   # 搜索
   results = kb.search("机器学习", top_k=3)
   for r in results:
       print(f"Score: {r['score']:.4f}")
       print(f"Text: {r['text'][:100]}...")

小结
====

本教程介绍了：

- 嵌入的概念和各种嵌入模型
- 向量存储的选择：Chroma、FAISS、Pinecone、Milvus
- 相似度计算方法
- 批量嵌入和嵌入缓存
- 索引的持久化和增量更新
- 完整的向量知识库实现

下一步
------

在下一个教程中，我们将学习 LlamaIndex 的各种索引类型，
了解向量索引之外的其他索引选项。

练习
====

1. 比较不同嵌入模型的效果
2. 使用 Chroma 构建持久化的知识库
3. 实现一个带缓存的嵌入系统
4. 测试不同向量数据库的性能
