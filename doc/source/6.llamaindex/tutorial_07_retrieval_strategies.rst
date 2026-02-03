################################
Tutorial 7: 检索策略
################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

.. contents:: 目录
   :local:
   :depth: 2

检索策略概述
============

检索策略决定了如何从索引中找到与查询最相关的内容。
好的检索策略直接影响 RAG 系统的质量。

.. code-block:: text

   检索策略选择：

   ┌─────────────────────────────────────────────────────────────┐
   │                      检索策略全景                           │
   ├─────────────────────────────────────────────────────────────┤
   │                                                              │
   │   基础策略                     高级策略                      │
   │   ┌─────────────┐             ┌─────────────┐               │
   │   │ 相似度搜索  │             │ 混合检索    │               │
   │   └─────────────┘             └─────────────┘               │
   │   ┌─────────────┐             ┌─────────────┐               │
   │   │  MMR 检索   │             │ 重排序      │               │
   │   └─────────────┘             └─────────────┘               │
   │   ┌─────────────┐             ┌─────────────┐               │
   │   │ 关键词检索  │             │ 多跳检索    │               │
   │   └─────────────┘             └─────────────┘               │
   │                                                              │
   └─────────────────────────────────────────────────────────────┘

相似度搜索
==========

最基础的检索方式，基于向量相似度。

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Document

   documents = [
       Document(text="机器学习是人工智能的核心技术。"),
       Document(text="深度学习使用多层神经网络。"),
       Document(text="自然语言处理让机器理解人类语言。"),
       Document(text="计算机视觉让机器看懂图像。"),
   ]

   index = VectorStoreIndex.from_documents(documents)

   # 创建检索器
   retriever = index.as_retriever(
       similarity_top_k=3  # 返回相似度最高的3个结果
   )

   # 检索
   nodes = retriever.retrieve("什么是机器学习？")

   for node in nodes:
       print(f"Score: {node.score:.4f} | {node.text}")

MMR 检索
========

最大边际相关性（Maximal Marginal Relevance），平衡相关性和多样性。

.. code-block:: python

   from llama_index.core.postprocessor import (
       SimilarityPostprocessor,
       MMRReranker
   )

   # 创建基础检索器
   retriever = index.as_retriever(similarity_top_k=10)

   # 使用 MMR 重排序
   nodes = retriever.retrieve("机器学习的应用")

   # 应用 MMR
   mmr_reranker = MMRReranker(
       top_n=5,
       diversity_bias=0.3  # 多样性偏好，0-1之间
   )
   reranked_nodes = mmr_reranker.postprocess_nodes(nodes)

   print("MMR 重排序后的结果：")
   for node in reranked_nodes:
       print(f"Score: {node.score:.4f} | {node.text[:50]}...")

关键词检索
==========

基于 BM25 的关键词检索。

.. code-block:: python

   # pip install llama-index-retrievers-bm25

   from llama_index.retrievers.bm25 import BM25Retriever
   from llama_index.core.node_parser import SentenceSplitter

   # 准备节点
   splitter = SentenceSplitter(chunk_size=256)
   nodes = splitter.get_nodes_from_documents(documents)

   # 创建 BM25 检索器
   bm25_retriever = BM25Retriever.from_defaults(
       nodes=nodes,
       similarity_top_k=5
   )

   # 检索
   results = bm25_retriever.retrieve("深度学习 神经网络")

   for node in results:
       print(f"Score: {node.score:.4f} | {node.text}")

混合检索
========

结合向量检索和关键词检索的优势。

.. code-block:: python

   from llama_index.core.retrievers import QueryFusionRetriever

   # 创建向量检索器
   vector_retriever = index.as_retriever(similarity_top_k=5)

   # 创建 BM25 检索器
   bm25_retriever = BM25Retriever.from_defaults(
       nodes=nodes,
       similarity_top_k=5
   )

   # 融合检索器
   fusion_retriever = QueryFusionRetriever(
       [vector_retriever, bm25_retriever],
       num_queries=1,
       use_async=False,
       similarity_top_k=5
   )

   # 混合检索
   results = fusion_retriever.retrieve("机器学习的基本概念")

   for node in results:
       print(f"Score: {node.score:.4f} | {node.text}")

自定义混合检索
--------------

.. code-block:: python

   from llama_index.core.retrievers import BaseRetriever
   from llama_index.core.schema import NodeWithScore, QueryBundle
   from typing import List

   class CustomHybridRetriever(BaseRetriever):
       """自定义混合检索器"""

       def __init__(
           self,
           vector_retriever,
           bm25_retriever,
           vector_weight: float = 0.6
       ):
           self.vector_retriever = vector_retriever
           self.bm25_retriever = bm25_retriever
           self.vector_weight = vector_weight
           self.bm25_weight = 1 - vector_weight

       def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
           # 向量检索
           vector_nodes = self.vector_retriever.retrieve(query_bundle)

           # BM25 检索
           bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

           # 归一化分数
           vector_scores = self._normalize_scores(vector_nodes)
           bm25_scores = self._normalize_scores(bm25_nodes)

           # 合并结果
           all_nodes = {}
           for node, score in vector_scores:
               all_nodes[node.node.id_] = {
                   "node": node,
                   "vector_score": score,
                   "bm25_score": 0
               }

           for node, score in bm25_scores:
               if node.node.id_ in all_nodes:
                   all_nodes[node.node.id_]["bm25_score"] = score
               else:
                   all_nodes[node.node.id_] = {
                       "node": node,
                       "vector_score": 0,
                       "bm25_score": score
                   }

           # 计算最终分数
           results = []
           for data in all_nodes.values():
               final_score = (
                   data["vector_score"] * self.vector_weight +
                   data["bm25_score"] * self.bm25_weight
               )
               node = data["node"]
               node.score = final_score
               results.append(node)

           # 排序
           results.sort(key=lambda x: x.score, reverse=True)
           return results[:10]

       def _normalize_scores(self, nodes):
           """归一化分数到 0-1"""
           if not nodes:
               return []

           scores = [n.score for n in nodes]
           min_score = min(scores)
           max_score = max(scores)

           if max_score == min_score:
               return [(n, 1.0) for n in nodes]

           return [
               (n, (n.score - min_score) / (max_score - min_score))
               for n in nodes
           ]

   # 使用
   hybrid_retriever = CustomHybridRetriever(
       vector_retriever=vector_retriever,
       bm25_retriever=bm25_retriever,
       vector_weight=0.7
   )

重排序（Reranking）
===================

对初始检索结果进行重新排序以提高质量。

LLM 重排序
----------

.. code-block:: python

   from llama_index.core.postprocessor import LLMRerank

   # 创建 LLM 重排序器
   reranker = LLMRerank(
       top_n=3,                    # 返回前3个
       choice_batch_size=5         # 每批处理5个
   )

   # 检索后重排序
   retriever = index.as_retriever(similarity_top_k=10)
   nodes = retriever.retrieve("机器学习如何工作？")

   reranked_nodes = reranker.postprocess_nodes(
       nodes,
       query_str="机器学习如何工作？"
   )

   for node in reranked_nodes:
       print(f"Score: {node.score:.4f} | {node.text[:50]}...")

Sentence Transformers 重排序
----------------------------

.. code-block:: python

   # pip install llama-index-postprocessor-sentencetransformers-rerank

   from llama_index.postprocessor.sentencetransformers_rerank import (
       SentenceTransformersRerank
   )

   # 使用 cross-encoder 模型重排序
   reranker = SentenceTransformersRerank(
       model="cross-encoder/ms-marco-MiniLM-L-12-v2",
       top_n=5
   )

   nodes = retriever.retrieve("深度学习的优势")
   reranked_nodes = reranker.postprocess_nodes(
       nodes,
       query_str="深度学习的优势"
   )

Cohere 重排序
-------------

.. code-block:: python

   # pip install llama-index-postprocessor-cohere-rerank

   from llama_index.postprocessor.cohere_rerank import CohereRerank

   reranker = CohereRerank(
       api_key="your-cohere-api-key",
       top_n=5,
       model="rerank-english-v2.0"
   )

   nodes = retriever.retrieve("query")
   reranked_nodes = reranker.postprocess_nodes(nodes, query_str="query")

自动融合检索
============

生成多个查询变体并融合结果。

.. code-block:: python

   from llama_index.core.retrievers import QueryFusionRetriever
   from llama_index.core import Settings
   from llama_index.llms.openai import OpenAI

   Settings.llm = OpenAI(model="gpt-4o-mini")

   # 创建融合检索器
   fusion_retriever = QueryFusionRetriever(
       [vector_retriever],
       num_queries=4,           # 生成4个查询变体
       similarity_top_k=5,
       use_async=True,
       verbose=True
   )

   # 检索
   # 系统会自动生成多个查询变体：
   # 原始查询: "机器学习应用"
   # 变体1: "机器学习的实际应用场景"
   # 变体2: "ML 在工业中的应用"
   # 变体3: "机器学习技术的使用案例"
   nodes = fusion_retriever.retrieve("机器学习应用")

递归检索
========

从摘要到详细内容的递归检索。

.. code-block:: python

   from llama_index.core import SummaryIndex, VectorStoreIndex
   from llama_index.core.retrievers import RecursiveRetriever
   from llama_index.core.query_engine import RetrieverQueryEngine

   # 创建层次化索引
   # 1. 摘要索引（粗粒度）
   summary_index = SummaryIndex.from_documents(documents)

   # 2. 向量索引（细粒度）
   vector_index = VectorStoreIndex.from_documents(documents)

   # 创建递归检索器
   retriever = RecursiveRetriever(
       root_id="summary",
       retriever_dict={
           "summary": summary_index.as_retriever(),
           "detail": vector_index.as_retriever()
       },
       # 定义如何从摘要导航到详细内容
       query_engine_dict={
           "summary": summary_index.as_query_engine()
       }
   )

句子窗口检索
============

检索句子并扩展到周围上下文。

.. code-block:: python

   from llama_index.core.node_parser import SentenceWindowNodeParser
   from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor

   # 创建句子窗口解析器
   node_parser = SentenceWindowNodeParser.from_defaults(
       window_size=3,  # 前后各3个句子作为窗口
       window_metadata_key="window",
       original_text_metadata_key="original_text"
   )

   # 解析文档
   nodes = node_parser.get_nodes_from_documents(documents)

   # 创建索引
   sentence_index = VectorStoreIndex(nodes)

   # 创建后处理器，用窗口内容替换原始文本
   postprocessor = MetadataReplacementPostProcessor(
       target_metadata_key="window"
   )

   # 查询引擎
   query_engine = sentence_index.as_query_engine(
       similarity_top_k=2,
       node_postprocessors=[postprocessor]
   )

   response = query_engine.query("什么是深度学习？")

自动合并检索
============

检索小块并自动合并为更大的上下文。

.. code-block:: python

   from llama_index.core.node_parser import HierarchicalNodeParser
   from llama_index.core.indices.postprocessor import AutoMergingRetriever
   from llama_index.core import StorageContext

   # 创建层次化解析器
   node_parser = HierarchicalNodeParser.from_defaults(
       chunk_sizes=[2048, 512, 128]  # 三个层次
   )

   # 解析得到层次化节点
   nodes = node_parser.get_nodes_from_documents(documents)

   # 存储上下文
   storage_context = StorageContext.from_defaults()
   storage_context.docstore.add_documents(nodes)

   # 只用最小的块构建索引
   leaf_nodes = [n for n in nodes if n.child_nodes is None]
   index = VectorStoreIndex(
       leaf_nodes,
       storage_context=storage_context
   )

   # 基础检索器
   base_retriever = index.as_retriever(similarity_top_k=6)

   # 自动合并检索器
   retriever = AutoMergingRetriever(
       base_retriever,
       storage_context,
       simple_ratio_thresh=0.4  # 当子节点被检索超过40%时，合并为父节点
   )

   nodes = retriever.retrieve("机器学习的基础知识")

小父文档检索
============

存储小块但检索时返回父文档。

.. code-block:: python

   from llama_index.core.retrievers import ParentDocumentRetriever
   from llama_index.core.node_parser import SentenceSplitter

   # 两级分割
   parent_splitter = SentenceSplitter(chunk_size=1024)
   child_splitter = SentenceSplitter(chunk_size=256)

   # 创建父子关系
   parent_nodes = parent_splitter.get_nodes_from_documents(documents)

   # 为每个父节点创建子节点
   for parent in parent_nodes:
       child_doc = Document(text=parent.text)
       children = child_splitter.get_nodes_from_documents([child_doc])
       for child in children:
           child.relationships["parent"] = parent.id_

   # 用子节点建立索引，检索时返回父节点

过滤检索
========

基于元数据过滤检索结果。

.. code-block:: python

   from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter
   from llama_index.core import Document

   # 带元数据的文档
   documents = [
       Document(
           text="Python 机器学习教程",
           metadata={"category": "programming", "level": "beginner"}
       ),
       Document(
           text="高级深度学习技术",
           metadata={"category": "ai", "level": "advanced"}
       ),
       Document(
           text="Python 数据分析入门",
           metadata={"category": "programming", "level": "beginner"}
       ),
   ]

   index = VectorStoreIndex.from_documents(documents)

   # 创建元数据过滤器
   filters = MetadataFilters(
       filters=[
           MetadataFilter(key="category", value="programming"),
           MetadataFilter(key="level", value="beginner"),
       ]
   )

   # 带过滤的检索器
   retriever = index.as_retriever(
       similarity_top_k=5,
       filters=filters
   )

   nodes = retriever.retrieve("编程教程")

实战示例
========

构建一个完整的高级检索系统。

.. code-block:: python

   from llama_index.core import VectorStoreIndex, Document, Settings
   from llama_index.core.node_parser import SentenceSplitter
   from llama_index.core.postprocessor import SimilarityPostprocessor
   from llama_index.retrievers.bm25 import BM25Retriever
   from llama_index.embeddings.openai import OpenAIEmbedding
   from typing import List, Optional

   class AdvancedRetrievalSystem:
       """高级检索系统"""

       def __init__(self):
           Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
           self.documents = []
           self.nodes = []
           self.vector_index = None
           self.bm25_retriever = None

       def add_documents(self, documents: List[Document]):
           """添加文档"""
           self.documents.extend(documents)

           # 解析为节点
           splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
           new_nodes = splitter.get_nodes_from_documents(documents)
           self.nodes.extend(new_nodes)

           # 重建索引
           self._rebuild_indexes()

       def _rebuild_indexes(self):
           """重建所有索引"""
           # 向量索引
           self.vector_index = VectorStoreIndex(self.nodes)

           # BM25 索引
           self.bm25_retriever = BM25Retriever.from_defaults(
               nodes=self.nodes,
               similarity_top_k=10
           )

       def retrieve(
           self,
           query: str,
           strategy: str = "hybrid",
           top_k: int = 5,
           min_similarity: float = 0.5
       ) -> List:
           """执行检索"""
           if strategy == "vector":
               nodes = self._vector_retrieve(query, top_k * 2)
           elif strategy == "keyword":
               nodes = self._keyword_retrieve(query, top_k * 2)
           elif strategy == "hybrid":
               nodes = self._hybrid_retrieve(query, top_k * 2)
           else:
               raise ValueError(f"Unknown strategy: {strategy}")

           # 过滤低相似度结果
           filtered = [n for n in nodes if n.score >= min_similarity]

           # 返回 top_k
           return filtered[:top_k]

       def _vector_retrieve(self, query: str, top_k: int) -> List:
           """向量检索"""
           retriever = self.vector_index.as_retriever(similarity_top_k=top_k)
           return retriever.retrieve(query)

       def _keyword_retrieve(self, query: str, top_k: int) -> List:
           """关键词检索"""
           return self.bm25_retriever.retrieve(query)[:top_k]

       def _hybrid_retrieve(self, query: str, top_k: int) -> List:
           """混合检索"""
           vector_nodes = self._vector_retrieve(query, top_k)
           keyword_nodes = self._keyword_retrieve(query, top_k)

           # 归一化并合并
           all_nodes = {}

           for node in vector_nodes:
               all_nodes[node.node.id_] = {
                   "node": node,
                   "vector_score": node.score,
                   "keyword_score": 0
               }

           for node in keyword_nodes:
               if node.node.id_ in all_nodes:
                   all_nodes[node.node.id_]["keyword_score"] = node.score
               else:
                   all_nodes[node.node.id_] = {
                       "node": node,
                       "vector_score": 0,
                       "keyword_score": node.score
                   }

           # 计算混合分数
           results = []
           for data in all_nodes.values():
               final_score = 0.6 * data["vector_score"] + 0.4 * data["keyword_score"]
               node = data["node"]
               node.score = final_score
               results.append(node)

           results.sort(key=lambda x: x.score, reverse=True)
           return results

       def retrieve_with_rerank(
           self,
           query: str,
           top_k: int = 5,
           rerank_top_n: int = 3
       ) -> List:
           """检索并重排序"""
           # 初始检索更多结果
           nodes = self._hybrid_retrieve(query, top_k * 2)

           # 简单的基于查询词重叠的重排序
           query_words = set(query.lower().split())
           for node in nodes:
               text_words = set(node.text.lower().split())
               overlap = len(query_words & text_words)
               node.score = node.score * (1 + 0.1 * overlap)

           nodes.sort(key=lambda x: x.score, reverse=True)
           return nodes[:rerank_top_n]

   # 使用示例
   system = AdvancedRetrievalSystem()

   # 添加文档
   system.add_documents([
       Document(text="机器学习是人工智能的核心分支，通过算法从数据中学习模式。"),
       Document(text="深度学习是机器学习的子集，使用多层神经网络进行特征学习。"),
       Document(text="自然语言处理使计算机能够理解和生成人类语言。"),
       Document(text="计算机视觉让机器能够从图像和视频中提取信息。"),
   ])

   # 不同策略检索
   print("向量检索:")
   for node in system.retrieve("机器学习", strategy="vector"):
       print(f"  {node.score:.4f}: {node.text[:50]}...")

   print("\n关键词检索:")
   for node in system.retrieve("机器学习", strategy="keyword"):
       print(f"  {node.score:.4f}: {node.text[:50]}...")

   print("\n混合检索:")
   for node in system.retrieve("机器学习", strategy="hybrid"):
       print(f"  {node.score:.4f}: {node.text[:50]}...")

   print("\n带重排序:")
   for node in system.retrieve_with_rerank("机器学习算法"):
       print(f"  {node.score:.4f}: {node.text[:50]}...")

小结
====

本教程介绍了：

- 基础检索策略：相似度搜索、MMR、关键词检索
- 混合检索：结合向量和关键词的优势
- 重排序技术：LLM、Cross-encoder、Cohere
- 高级检索：自动融合、递归检索、句子窗口、自动合并
- 过滤检索：基于元数据筛选
- 完整的高级检索系统实现

下一步
------

在下一个教程中，我们将学习 LlamaIndex 的 Agent 和工具系统，
了解如何构建能够使用工具的智能 Agent。

练习
====

1. 比较不同检索策略在同一数据集上的效果
2. 实现一个带 Cross-encoder 重排序的检索系统
3. 使用句子窗口检索提高上下文完整性
4. 构建一个支持元数据过滤的知识库检索系统
