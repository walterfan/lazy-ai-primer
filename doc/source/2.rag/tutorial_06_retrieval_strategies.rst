####################################
Tutorial 6: 检索策略
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

检索策略概述
============

检索是 RAG 系统的核心，好的检索策略能显著提升回答质量。

.. code-block:: text

   检索策略
   ├── 基础检索
   │   ├── 相似度搜索
   │   └── MMR（最大边际相关性）
   ├── 混合检索
   │   ├── 关键词 + 语义
   │   └── 多路召回
   ├── 高级检索
   │   ├── 重排序
   │   ├── 查询扩展
   │   └── 假设文档嵌入
   └── 自适应检索
       └── 根据查询类型选择策略

基础检索
========

1. 相似度搜索
-------------

.. code-block:: python

   from langchain_community.vectorstores import Chroma
   from langchain_community.embeddings import HuggingFaceEmbeddings

   # 创建向量存储
   embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   vectorstore = Chroma.from_texts(
       texts=[
           "Python是一种高级编程语言",
           "机器学习让计算机从数据中学习",
           "深度学习使用神经网络",
           "RAG结合检索和生成技术"
       ],
       embedding=embeddings
   )

   # 相似度搜索
   query = "什么是深度学习？"
   docs = vectorstore.similarity_search(query, k=2)

   for doc in docs:
       print(f"内容: {doc.page_content}")

   # 带分数的搜索
   docs_with_scores = vectorstore.similarity_search_with_score(query, k=2)
   for doc, score in docs_with_scores:
       print(f"[{score:.4f}] {doc.page_content}")

2. MMR（最大边际相关性）
------------------------

MMR 在保证相关性的同时，增加结果的多样性。

.. code-block:: python

   # MMR 搜索
   # 平衡相关性和多样性
   docs = vectorstore.max_marginal_relevance_search(
       query,
       k=4,                    # 返回数量
       fetch_k=10,             # 初始检索数量
       lambda_mult=0.5         # 多样性参数 (0=最大多样性, 1=最大相关性)
   )

   print("MMR 搜索结果:")
   for doc in docs:
       print(f"  - {doc.page_content}")

混合检索
========

结合关键词搜索和语义搜索的优势。

.. code-block:: python

   from langchain.retrievers import BM25Retriever, EnsembleRetriever
   from langchain_community.vectorstores import Chroma
   from langchain_community.embeddings import HuggingFaceEmbeddings

   # 准备文档
   texts = [
       "Python编程语言简洁易学",
       "Java是面向对象的编程语言",
       "机器学习是AI的重要分支",
       "深度学习基于神经网络",
       "自然语言处理处理文本"
   ]

   # 1. 创建 BM25 检索器（关键词）
   bm25_retriever = BM25Retriever.from_texts(texts)
   bm25_retriever.k = 3

   # 2. 创建向量检索器（语义）
   embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   vectorstore = Chroma.from_texts(texts, embeddings)
   vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

   # 3. 创建混合检索器
   ensemble_retriever = EnsembleRetriever(
       retrievers=[bm25_retriever, vector_retriever],
       weights=[0.4, 0.6]  # BM25 40%, 向量 60%
   )

   # 搜索
   query = "Python编程"
   docs = ensemble_retriever.get_relevant_documents(query)

   print("混合检索结果:")
   for doc in docs:
       print(f"  - {doc.page_content}")

重排序
======

使用更强大的模型对初步检索结果进行重新排序。

.. code-block:: python

   # pip install sentence-transformers

   from sentence_transformers import CrossEncoder
   from typing import List, Tuple

   class Reranker:
       """重排序器"""
       
       def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
           self.model = CrossEncoder(model_name)
       
       def rerank(
           self,
           query: str,
           documents: List[str],
           top_k: int = 5
       ) -> List[Tuple[str, float]]:
           """重排序文档"""
           # 构建查询-文档对
           pairs = [[query, doc] for doc in documents]
           
           # 计算相关性分数
           scores = self.model.predict(pairs)
           
           # 排序
           doc_scores = list(zip(documents, scores))
           doc_scores.sort(key=lambda x: x[1], reverse=True)
           
           return doc_scores[:top_k]

   # 使用重排序
   reranker = Reranker()

   query = "深度学习的原理"
   initial_docs = [
       "深度学习使用多层神经网络",
       "Python是编程语言",
       "神经网络模拟人脑结构",
       "机器学习包含深度学习",
       "今天天气很好"
   ]

   reranked = reranker.rerank(query, initial_docs, top_k=3)

   print("重排序结果:")
   for doc, score in reranked:
       print(f"  [{score:.4f}] {doc}")

查询扩展
========

扩展原始查询以提高召回率。

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain.prompts import PromptTemplate

   class QueryExpander:
       """查询扩展器"""
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
           self.prompt = PromptTemplate(
               input_variables=["query"],
               template="""给定用户查询，生成3个相关的扩展查询，用于提高搜索召回率。

   原始查询: {query}

   扩展查询（每行一个）:"""
           )
       
       def expand(self, query: str) -> List[str]:
           """扩展查询"""
           response = self.llm.invoke(self.prompt.format(query=query))
           expanded = response.content.strip().split('\n')
           expanded = [q.strip().lstrip('0123456789.-) ') for q in expanded if q.strip()]
           return [query] + expanded  # 包含原始查询

   # 使用
   expander = QueryExpander()
   original_query = "Python机器学习"
   expanded_queries = expander.expand(original_query)

   print("扩展后的查询:")
   for q in expanded_queries:
       print(f"  - {q}")

   # 对每个扩展查询进行检索，然后合并结果
   all_docs = []
   for q in expanded_queries:
       docs = vectorstore.similarity_search(q, k=2)
       all_docs.extend(docs)

   # 去重
   unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

假设文档嵌入（HyDE）
====================

先生成假设性答案，再用答案进行检索。

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain.prompts import PromptTemplate

   class HyDERetriever:
       """假设文档嵌入检索器"""
       
       def __init__(self, vectorstore, embeddings):
           self.vectorstore = vectorstore
           self.embeddings = embeddings
           self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
           
           self.prompt = PromptTemplate(
               input_variables=["question"],
               template="""请写一段文字来回答以下问题。不需要完全准确，只需要包含相关的关键信息。

   问题: {question}

   回答:"""
           )
       
       def retrieve(self, question: str, k: int = 3):
           """检索相关文档"""
           # 1. 生成假设性答案
           hypothetical_answer = self.llm.invoke(
               self.prompt.format(question=question)
           ).content
           
           print(f"假设性答案: {hypothetical_answer[:100]}...")
           
           # 2. 用假设性答案进行检索
           docs = self.vectorstore.similarity_search(
               hypothetical_answer,
               k=k
           )
           
           return docs

   # 使用 HyDE
   hyde_retriever = HyDERetriever(vectorstore, embeddings)
   docs = hyde_retriever.retrieve("深度学习如何工作？")

   print("\nHyDE 检索结果:")
   for doc in docs:
       print(f"  - {doc.page_content}")

多路召回
========

.. code-block:: python

   from typing import List, Dict
   from langchain.schema import Document

   class MultiRouteRetriever:
       """多路召回检索器"""
       
       def __init__(self, retrievers: Dict[str, any], weights: Dict[str, float] = None):
           self.retrievers = retrievers
           self.weights = weights or {name: 1.0 for name in retrievers}
       
       def retrieve(self, query: str, k: int = 5) -> List[Document]:
           """多路召回"""
           all_docs = {}
           
           for name, retriever in self.retrievers.items():
               weight = self.weights.get(name, 1.0)
               docs = retriever.get_relevant_documents(query)
               
               for i, doc in enumerate(docs):
                   content = doc.page_content
                   # 根据排名和权重计算分数
                   score = weight * (1.0 / (i + 1))
                   
                   if content in all_docs:
                       all_docs[content]['score'] += score
                   else:
                       all_docs[content] = {
                           'doc': doc,
                           'score': score
                       }
           
           # 按分数排序
           sorted_docs = sorted(
               all_docs.values(),
               key=lambda x: x['score'],
               reverse=True
           )
           
           return [item['doc'] for item in sorted_docs[:k]]

   # 使用多路召回
   multi_retriever = MultiRouteRetriever(
       retrievers={
           "semantic": vector_retriever,
           "keyword": bm25_retriever
       },
       weights={
           "semantic": 0.6,
           "keyword": 0.4
       }
   )

   docs = multi_retriever.retrieve("Python编程入门")

自适应检索
==========

根据查询类型选择不同的检索策略。

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from enum import Enum

   class QueryType(Enum):
       FACTUAL = "factual"      # 事实性问题
       CONCEPTUAL = "conceptual"  # 概念性问题
       PROCEDURAL = "procedural"  # 过程性问题

   class AdaptiveRetriever:
       """自适应检索器"""
       
       def __init__(self, vectorstore):
           self.vectorstore = vectorstore
           self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
       
       def classify_query(self, query: str) -> QueryType:
           """分类查询类型"""
           prompt = f"""将以下查询分类为以下类型之一:
   - factual: 寻找具体事实或数据
   - conceptual: 理解概念或原理
   - procedural: 了解如何做某事

   查询: {query}

   类型（只输出一个词）:"""
           
           response = self.llm.invoke(prompt).content.strip().lower()
           
           if "factual" in response:
               return QueryType.FACTUAL
           elif "procedural" in response:
               return QueryType.PROCEDURAL
           else:
               return QueryType.CONCEPTUAL
       
       def retrieve(self, query: str, k: int = 5):
           """自适应检索"""
           query_type = self.classify_query(query)
           print(f"查询类型: {query_type.value}")
           
           if query_type == QueryType.FACTUAL:
               # 事实性问题：精确匹配
               return self.vectorstore.similarity_search(query, k=k)
           
           elif query_type == QueryType.CONCEPTUAL:
               # 概念性问题：增加多样性
               return self.vectorstore.max_marginal_relevance_search(
                   query, k=k, lambda_mult=0.5
               )
           
           else:  # PROCEDURAL
               # 过程性问题：检索更多上下文
               return self.vectorstore.similarity_search(query, k=k*2)[:k]

检索策略选择指南
================

.. csv-table::
   :header: "场景", "推荐策略", "原因"
   :widths: 30, 30, 40

   "通用问答", "相似度搜索", "简单有效"
   "需要多样性", "MMR", "避免结果重复"
   "专业术语多", "混合检索", "结合关键词和语义"
   "高精度要求", "重排序", "二次精排提升质量"
   "复杂问题", "HyDE", "生成假设答案辅助检索"

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "相似度搜索", "基于向量距离的搜索"
   "MMR", "最大边际相关性，平衡相关性和多样性"
   "混合检索", "结合多种检索方法"
   "重排序", "用更强模型对结果重新排序"
   "HyDE", "假设文档嵌入，生成假设答案辅助检索"

下一步
======

在下一个教程中，我们将学习 RAG 的 Prompt 工程。

:doc:`tutorial_07_prompt_engineering`
