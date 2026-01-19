####################################
Tutorial 4: 向量嵌入
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是向量嵌入？
================

**向量嵌入（Embedding）** 是将文本转换为数值向量的技术，使计算机能够理解文本的语义。

.. code-block:: text

   文本                          向量 (简化示意)
   
   "猫"    ──────────────────►   [0.2, 0.8, 0.1, ...]
   "狗"    ──────────────────►   [0.3, 0.7, 0.2, ...]
   "汽车"  ──────────────────►   [0.9, 0.1, 0.8, ...]
   
   语义相似的文本，向量也相似：
   · "猫" 和 "狗" 的向量接近（都是动物）
   · "猫" 和 "汽车" 的向量较远

为什么需要向量嵌入？
--------------------

- **语义理解**: 捕捉文本的含义，而不仅仅是字面匹配
- **相似度计算**: 通过向量距离衡量文本相似度
- **高效检索**: 向量数据库支持快速近似搜索

使用嵌入模型
============

1. Sentence Transformers
------------------------

.. code-block:: python

   # pip install sentence-transformers

   from sentence_transformers import SentenceTransformer

   # 加载模型
   model = SentenceTransformer('all-MiniLM-L6-v2')

   # 单个文本嵌入
   text = "人工智能正在改变世界"
   embedding = model.encode(text)

   print(f"文本: {text}")
   print(f"向量维度: {len(embedding)}")
   print(f"向量前5个值: {embedding[:5]}")

   # 批量嵌入
   texts = [
       "机器学习是人工智能的分支",
       "深度学习使用神经网络",
       "今天天气很好"
   ]
   embeddings = model.encode(texts)

   print(f"\n批量嵌入: {len(embeddings)} 个向量")

2. OpenAI Embeddings
--------------------

.. code-block:: python

   # pip install openai

   from openai import OpenAI

   client = OpenAI()

   def get_openai_embedding(text, model="text-embedding-3-small"):
       """获取 OpenAI 嵌入"""
       response = client.embeddings.create(
           input=text,
           model=model
       )
       return response.data[0].embedding

   # 单个文本
   text = "人工智能正在改变世界"
   embedding = get_openai_embedding(text)

   print(f"向量维度: {len(embedding)}")

   # 批量嵌入
   def get_openai_embeddings(texts, model="text-embedding-3-small"):
       response = client.embeddings.create(
           input=texts,
           model=model
       )
       return [item.embedding for item in response.data]

3. LangChain Embeddings
-----------------------

.. code-block:: python

   from langchain_community.embeddings import HuggingFaceEmbeddings
   from langchain_openai import OpenAIEmbeddings

   # HuggingFace 嵌入
   hf_embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
   )

   # OpenAI 嵌入
   openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

   # 使用
   text = "这是一段测试文本"
   
   vector_hf = hf_embeddings.embed_query(text)
   vector_openai = openai_embeddings.embed_query(text)

   print(f"HuggingFace 向量维度: {len(vector_hf)}")
   print(f"OpenAI 向量维度: {len(vector_openai)}")

   # 批量嵌入
   texts = ["文本1", "文本2", "文本3"]
   vectors = hf_embeddings.embed_documents(texts)

计算相似度
==========

1. 余弦相似度
-------------

.. code-block:: python

   import numpy as np

   def cosine_similarity(v1, v2):
       """计算余弦相似度"""
       dot_product = np.dot(v1, v2)
       norm1 = np.linalg.norm(v1)
       norm2 = np.linalg.norm(v2)
       return dot_product / (norm1 * norm2)

   # 示例
   model = SentenceTransformer('all-MiniLM-L6-v2')

   texts = [
       "猫是一种可爱的宠物",
       "狗是人类最好的朋友",
       "Python是一种编程语言"
   ]

   embeddings = model.encode(texts)

   # 计算相似度矩阵
   print("相似度矩阵:")
   for i, text1 in enumerate(texts):
       for j, text2 in enumerate(texts):
           sim = cosine_similarity(embeddings[i], embeddings[j])
           print(f"  '{text1[:10]}...' vs '{text2[:10]}...': {sim:.3f}")

2. 欧氏距离
-----------

.. code-block:: python

   def euclidean_distance(v1, v2):
       """计算欧氏距离"""
       return np.linalg.norm(np.array(v1) - np.array(v2))

   # 距离越小，越相似
   dist = euclidean_distance(embeddings[0], embeddings[1])
   print(f"欧氏距离: {dist:.4f}")

3. 点积相似度
-------------

.. code-block:: python

   def dot_product_similarity(v1, v2):
       """计算点积相似度（适用于归一化向量）"""
       return np.dot(v1, v2)

   # 先归一化
   def normalize(v):
       return v / np.linalg.norm(v)

   v1_norm = normalize(embeddings[0])
   v2_norm = normalize(embeddings[1])
   sim = dot_product_similarity(v1_norm, v2_norm)

嵌入模型选择
============

.. csv-table::
   :header: "模型", "维度", "语言", "特点", "适用场景"
   :widths: 25, 10, 15, 25, 25

   "all-MiniLM-L6-v2", "384", "英文", "快速、轻量", "通用检索"
   "all-mpnet-base-v2", "768", "英文", "高质量", "高精度检索"
   "text-embedding-3-small", "1536", "多语言", "OpenAI 最新", "生产环境"
   "text-embedding-3-large", "3072", "多语言", "最高质量", "高要求场景"
   "bge-large-zh", "1024", "中文", "中文优化", "中文检索"
   "m3e-base", "768", "中文", "中文优化", "中文检索"

中文嵌入模型
============

.. code-block:: python

   # 使用中文优化的嵌入模型

   # 方式1: 使用 sentence-transformers
   from sentence_transformers import SentenceTransformer

   # BGE 中文模型
   model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

   texts = [
       "机器学习是人工智能的重要分支",
       "深度学习在图像识别中应用广泛",
       "今天的天气非常晴朗"
   ]

   embeddings = model.encode(texts)

   # 计算相似度
   for i in range(len(texts)):
       for j in range(i+1, len(texts)):
           sim = cosine_similarity(embeddings[i], embeddings[j])
           print(f"'{texts[i][:15]}...' vs '{texts[j][:15]}...': {sim:.3f}")

   # 方式2: 使用 LangChain
   from langchain_community.embeddings import HuggingFaceEmbeddings

   embeddings_model = HuggingFaceEmbeddings(
       model_name="BAAI/bge-large-zh-v1.5",
       model_kwargs={'device': 'cpu'},
       encode_kwargs={'normalize_embeddings': True}
   )

实战：语义搜索
==============

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   import numpy as np

   class SemanticSearch:
       """语义搜索引擎"""
       
       def __init__(self, model_name='all-MiniLM-L6-v2'):
           self.model = SentenceTransformer(model_name)
           self.documents = []
           self.embeddings = None
       
       def index(self, documents):
           """索引文档"""
           self.documents = documents
           self.embeddings = self.model.encode(documents)
           print(f"索引了 {len(documents)} 个文档")
       
       def search(self, query, top_k=3):
           """搜索相关文档"""
           query_embedding = self.model.encode([query])[0]
           
           # 计算相似度
           similarities = []
           for i, doc_embedding in enumerate(self.embeddings):
               sim = np.dot(query_embedding, doc_embedding) / (
                   np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
               )
               similarities.append((i, sim))
           
           # 排序
           similarities.sort(key=lambda x: x[1], reverse=True)
           
           # 返回 top_k 结果
           results = []
           for idx, sim in similarities[:top_k]:
               results.append({
                   'document': self.documents[idx],
                   'similarity': sim,
                   'index': idx
               })
           
           return results

   # 使用示例
   search_engine = SemanticSearch()

   # 知识库
   documents = [
       "Python是一种高级编程语言，以简洁易读著称。",
       "机器学习是让计算机从数据中学习的技术。",
       "深度学习使用多层神经网络处理复杂数据。",
       "自然语言处理让计算机理解人类语言。",
       "计算机视觉让机器能够理解图像和视频。",
       "强化学习通过奖励机制训练智能体。",
       "RAG结合检索和生成来增强大语言模型。",
   ]

   # 索引
   search_engine.index(documents)

   # 搜索
   queries = [
       "如何让机器理解图片？",
       "什么是神经网络？",
       "Python语言有什么特点？"
   ]

   for query in queries:
       print(f"\n查询: {query}")
       results = search_engine.search(query, top_k=2)
       for r in results:
           print(f"  [{r['similarity']:.3f}] {r['document']}")

嵌入缓存
========

.. code-block:: python

   import hashlib
   import json
   import os

   class EmbeddingCache:
       """嵌入缓存"""
       
       def __init__(self, model, cache_dir="./embedding_cache"):
           self.model = model
           self.cache_dir = cache_dir
           os.makedirs(cache_dir, exist_ok=True)
       
       def _get_cache_key(self, text):
           """生成缓存键"""
           return hashlib.md5(text.encode()).hexdigest()
       
       def _get_cache_path(self, key):
           """获取缓存文件路径"""
           return os.path.join(self.cache_dir, f"{key}.json")
       
       def encode(self, texts):
           """编码文本（带缓存）"""
           if isinstance(texts, str):
               texts = [texts]
           
           embeddings = []
           texts_to_encode = []
           cache_indices = {}
           
           for i, text in enumerate(texts):
               key = self._get_cache_key(text)
               cache_path = self._get_cache_path(key)
               
               if os.path.exists(cache_path):
                   # 从缓存加载
                   with open(cache_path, 'r') as f:
                       embedding = json.load(f)
                   embeddings.append((i, embedding))
               else:
                   texts_to_encode.append(text)
                   cache_indices[len(texts_to_encode) - 1] = i
           
           # 编码未缓存的文本
           if texts_to_encode:
               new_embeddings = self.model.encode(texts_to_encode).tolist()
               
               for j, embedding in enumerate(new_embeddings):
                   original_idx = cache_indices[j]
                   embeddings.append((original_idx, embedding))
                   
                   # 保存到缓存
                   key = self._get_cache_key(texts_to_encode[j])
                   cache_path = self._get_cache_path(key)
                   with open(cache_path, 'w') as f:
                       json.dump(embedding, f)
           
           # 按原始顺序排序
           embeddings.sort(key=lambda x: x[0])
           return [e[1] for e in embeddings]

   # 使用缓存
   model = SentenceTransformer('all-MiniLM-L6-v2')
   cached_model = EmbeddingCache(model)

   # 第一次编码（会保存缓存）
   embeddings1 = cached_model.encode(["测试文本1", "测试文本2"])

   # 第二次编码（从缓存加载）
   embeddings2 = cached_model.encode(["测试文本1", "测试文本2"])

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "Embedding", "将文本映射到向量空间的表示"
   "向量维度", "嵌入向量的长度"
   "余弦相似度", "衡量两个向量方向的相似程度"
   "语义搜索", "基于语义相似度的搜索"
   "归一化", "将向量缩放到单位长度"

下一步
======

在下一个教程中，我们将学习向量数据库的使用。

:doc:`tutorial_05_vector_database`
