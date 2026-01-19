####################################
Tutorial 5: 向量数据库
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是向量数据库？
==================

**向量数据库** 是专门用于存储和检索向量的数据库，支持高效的相似性搜索。

.. code-block:: text

   传统数据库 vs 向量数据库
   
   传统数据库:
   SELECT * FROM documents WHERE title = '机器学习'
   → 精确匹配
   
   向量数据库:
   SEARCH vectors SIMILAR TO query_vector LIMIT 5
   → 语义相似匹配

为什么需要向量数据库？
----------------------

- **高效搜索**: 支持亿级向量的毫秒级检索
- **近似最近邻**: ANN 算法实现快速相似搜索
- **可扩展性**: 支持分布式部署
- **元数据过滤**: 结合向量搜索和传统过滤

主流向量数据库
==============

.. csv-table::
   :header: "数据库", "类型", "特点", "适用场景"
   :widths: 15, 15, 40, 30

   "Chroma", "嵌入式", "轻量、易用、开源", "开发测试、小规模"
   "FAISS", "库", "Meta开源、高性能", "大规模检索"
   "Pinecone", "云服务", "全托管、易扩展", "生产环境"
   "Milvus", "分布式", "开源、高可用", "企业级部署"
   "Weaviate", "云/自托管", "支持多模态", "复杂应用"
   "Qdrant", "云/自托管", "Rust实现、高性能", "生产环境"

使用 Chroma
===========

Chroma 是最简单易用的向量数据库，适合学习和小规模应用。

.. code-block:: python

   # pip install chromadb

   import chromadb
   from chromadb.utils import embedding_functions

   # 1. 创建客户端
   # 内存模式（数据不持久化）
   client = chromadb.Client()

   # 持久化模式
   client = chromadb.PersistentClient(path="./chroma_db")

   # 2. 创建/获取集合
   # 使用默认嵌入函数
   collection = client.create_collection(
       name="my_collection",
       metadata={"description": "测试集合"}
   )

   # 使用自定义嵌入函数
   sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
       model_name="all-MiniLM-L6-v2"
   )

   collection = client.create_collection(
       name="custom_embedding_collection",
       embedding_function=sentence_transformer_ef
   )

   # 3. 添加文档
   collection.add(
       documents=[
           "机器学习是人工智能的分支",
           "深度学习使用神经网络",
           "自然语言处理处理文本数据"
       ],
       metadatas=[
           {"category": "AI", "source": "教材"},
           {"category": "AI", "source": "论文"},
           {"category": "NLP", "source": "教材"}
       ],
       ids=["doc1", "doc2", "doc3"]
   )

   print(f"集合中有 {collection.count()} 个文档")

   # 4. 查询
   results = collection.query(
       query_texts=["什么是神经网络？"],
       n_results=2
   )

   print("\n查询结果:")
   for i, (doc, metadata, distance) in enumerate(zip(
       results['documents'][0],
       results['metadatas'][0],
       results['distances'][0]
   )):
       print(f"  {i+1}. [{distance:.4f}] {doc}")
       print(f"     元数据: {metadata}")

   # 5. 带过滤的查询
   results = collection.query(
       query_texts=["AI技术"],
       n_results=2,
       where={"category": "AI"}  # 只搜索 AI 类别
   )

   # 6. 更新文档
   collection.update(
       ids=["doc1"],
       documents=["机器学习是人工智能的重要分支，让计算机从数据中学习"],
       metadatas=[{"category": "AI", "source": "教材", "updated": True}]
   )

   # 7. 删除文档
   collection.delete(ids=["doc3"])

使用 FAISS
==========

FAISS 是 Meta 开源的高性能向量检索库。

.. code-block:: python

   # pip install faiss-cpu  # 或 faiss-gpu

   import faiss
   import numpy as np
   from sentence_transformers import SentenceTransformer

   # 1. 准备数据
   model = SentenceTransformer('all-MiniLM-L6-v2')

   documents = [
       "Python是一种编程语言",
       "机器学习是AI的分支",
       "深度学习使用神经网络",
       "自然语言处理理解文本",
       "计算机视觉处理图像"
   ]

   # 生成嵌入
   embeddings = model.encode(documents)
   embeddings = np.array(embeddings).astype('float32')

   # 2. 创建索引
   dimension = embeddings.shape[1]  # 向量维度

   # Flat 索引（精确搜索，适合小数据集）
   index = faiss.IndexFlatL2(dimension)

   # 添加向量
   index.add(embeddings)

   print(f"索引中有 {index.ntotal} 个向量")

   # 3. 搜索
   query = "什么是神经网络？"
   query_embedding = model.encode([query]).astype('float32')

   k = 3  # 返回 top 3
   distances, indices = index.search(query_embedding, k)

   print(f"\n查询: {query}")
   for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
       print(f"  {i+1}. [{dist:.4f}] {documents[idx]}")

   # 4. 使用 IVF 索引（适合大数据集）
   nlist = 2  # 聚类数量
   quantizer = faiss.IndexFlatL2(dimension)
   index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)

   # 训练索引
   index_ivf.train(embeddings)
   index_ivf.add(embeddings)

   # 设置搜索参数
   index_ivf.nprobe = 1  # 搜索的聚类数量

   distances, indices = index_ivf.search(query_embedding, k)

   # 5. 保存和加载索引
   faiss.write_index(index, "my_index.faiss")
   loaded_index = faiss.read_index("my_index.faiss")

使用 LangChain 集成
===================

LangChain 提供了统一的向量数据库接口。

.. code-block:: python

   from langchain_community.vectorstores import Chroma, FAISS
   from langchain_community.embeddings import HuggingFaceEmbeddings
   from langchain.schema import Document

   # 准备嵌入模型
   embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
   )

   # 准备文档
   documents = [
       Document(page_content="机器学习是AI的分支", metadata={"source": "教材"}),
       Document(page_content="深度学习使用神经网络", metadata={"source": "论文"}),
       Document(page_content="NLP处理自然语言", metadata={"source": "教材"}),
   ]

   # 使用 Chroma
   vectorstore_chroma = Chroma.from_documents(
       documents=documents,
       embedding=embeddings,
       persist_directory="./chroma_langchain"
   )

   # 使用 FAISS
   vectorstore_faiss = FAISS.from_documents(
       documents=documents,
       embedding=embeddings
   )

   # 统一的搜索接口
   query = "什么是神经网络？"

   # 相似度搜索
   results = vectorstore_chroma.similarity_search(query, k=2)
   for doc in results:
       print(f"内容: {doc.page_content}")
       print(f"元数据: {doc.metadata}")

   # 带分数的搜索
   results_with_scores = vectorstore_chroma.similarity_search_with_score(query, k=2)
   for doc, score in results_with_scores:
       print(f"[{score:.4f}] {doc.page_content}")

   # 转换为检索器
   retriever = vectorstore_chroma.as_retriever(
       search_type="similarity",
       search_kwargs={"k": 3}
   )

   docs = retriever.get_relevant_documents(query)

实战：构建知识库
================

.. code-block:: python

   import chromadb
   from sentence_transformers import SentenceTransformer
   from typing import List, Dict, Optional
   import uuid

   class KnowledgeBase:
       """知识库管理器"""
       
       def __init__(self, persist_path: str = "./knowledge_db"):
           self.client = chromadb.PersistentClient(path=persist_path)
           self.model = SentenceTransformer('all-MiniLM-L6-v2')
           self.collection = self.client.get_or_create_collection(
               name="knowledge_base",
               metadata={"description": "RAG知识库"}
           )
       
       def add_documents(
           self,
           documents: List[str],
           metadatas: Optional[List[Dict]] = None,
           ids: Optional[List[str]] = None
       ):
           """添加文档"""
           if ids is None:
               ids = [str(uuid.uuid4()) for _ in documents]
           
           if metadatas is None:
               metadatas = [{} for _ in documents]
           
           # 生成嵌入
           embeddings = self.model.encode(documents).tolist()
           
           # 添加到集合
           self.collection.add(
               documents=documents,
               embeddings=embeddings,
               metadatas=metadatas,
               ids=ids
           )
           
           print(f"添加了 {len(documents)} 个文档")
       
       def search(
           self,
           query: str,
           top_k: int = 5,
           filter_dict: Optional[Dict] = None
       ) -> List[Dict]:
           """搜索相关文档"""
           query_embedding = self.model.encode([query]).tolist()
           
           results = self.collection.query(
               query_embeddings=query_embedding,
               n_results=top_k,
               where=filter_dict
           )
           
           # 格式化结果
           formatted_results = []
           for i in range(len(results['documents'][0])):
               formatted_results.append({
                   'document': results['documents'][0][i],
                   'metadata': results['metadatas'][0][i],
                   'distance': results['distances'][0][i],
                   'id': results['ids'][0][i]
               })
           
           return formatted_results
       
       def delete(self, ids: List[str]):
           """删除文档"""
           self.collection.delete(ids=ids)
           print(f"删除了 {len(ids)} 个文档")
       
       def get_stats(self) -> Dict:
           """获取统计信息"""
           return {
               "total_documents": self.collection.count(),
               "collection_name": self.collection.name
           }

   # 使用知识库
   kb = KnowledgeBase()

   # 添加文档
   documents = [
       "RAG是检索增强生成的缩写，结合了检索和生成技术。",
       "向量数据库用于存储和检索向量嵌入。",
       "Chroma是一个轻量级的向量数据库。",
       "FAISS是Meta开发的高性能向量检索库。",
       "嵌入模型将文本转换为向量表示。"
   ]

   metadatas = [
       {"topic": "RAG", "level": "基础"},
       {"topic": "向量数据库", "level": "基础"},
       {"topic": "向量数据库", "level": "入门"},
       {"topic": "向量数据库", "level": "进阶"},
       {"topic": "嵌入", "level": "基础"}
   ]

   kb.add_documents(documents, metadatas)

   # 搜索
   print("\n搜索: '什么是向量数据库？'")
   results = kb.search("什么是向量数据库？", top_k=3)
   for r in results:
       print(f"  [{r['distance']:.4f}] {r['document']}")
       print(f"    元数据: {r['metadata']}")

   # 带过滤的搜索
   print("\n搜索（只看基础级别）:")
   results = kb.search(
       "向量数据库",
       top_k=3,
       filter_dict={"level": "基础"}
   )
   for r in results:
       print(f"  [{r['distance']:.4f}] {r['document']}")

   # 统计
   print(f"\n知识库统计: {kb.get_stats()}")

索引优化
========

.. code-block:: python

   import faiss
   import numpy as np

   # 不同索引类型的选择

   dimension = 384
   n_vectors = 100000

   # 1. Flat（精确搜索）- 小数据集
   index_flat = faiss.IndexFlatL2(dimension)

   # 2. IVF（倒排索引）- 中等数据集
   nlist = 100  # 聚类数
   quantizer = faiss.IndexFlatL2(dimension)
   index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)

   # 3. HNSW（图索引）- 高召回率
   index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # 32是连接数

   # 4. PQ（乘积量化）- 大数据集，节省内存
   m = 8  # 子向量数
   index_pq = faiss.IndexPQ(dimension, m, 8)

   # 5. IVF + PQ - 大规模数据集
   index_ivfpq = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "向量数据库", "专门存储和检索向量的数据库"
   "ANN", "近似最近邻搜索算法"
   "索引", "加速搜索的数据结构"
   "Collection", "向量数据库中的集合/表"
   "元数据过滤", "结合向量搜索和属性过滤"

下一步
======

在下一个教程中，我们将学习各种检索策略。

:doc:`tutorial_06_retrieval_strategies`
