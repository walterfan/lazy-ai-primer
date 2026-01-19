####################################
Tutorial 9: 高级 RAG 技术
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

高级 RAG 概述
=============

基础 RAG 存在一些局限性，高级技术可以显著提升性能。

.. code-block:: text

   高级 RAG 技术
   
   ├── 检索前优化
   │   ├── 查询改写
   │   ├── 查询分解
   │   └── HyDE
   │
   ├── 检索优化
   │   ├── 混合检索
   │   ├── 递归检索
   │   └── 自适应检索
   │
   ├── 检索后优化
   │   ├── 重排序
   │   ├── 上下文压缩
   │   └── 去重与融合
   │
   └── 生成优化
       ├── 思维链
       ├── 自我反思
       └── 多轮对话

查询改写
========

将用户的原始查询改写为更适合检索的形式。

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain.prompts import PromptTemplate

   class QueryRewriter:
       """查询改写器"""
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
       
       def rewrite(self, query: str) -> str:
           """改写查询"""
           prompt = f"""将以下用户查询改写为更适合搜索的形式。

   原始查询: {query}

   改写要求:
   - 去除口语化表达
   - 补充隐含的关键词
   - 保持原意

   改写后的查询:"""
           
           response = self.llm.invoke(prompt)
           return response.content.strip()
       
       def expand(self, query: str, n: int = 3) -> list:
           """扩展查询"""
           prompt = f"""为以下查询生成{n}个相关的搜索查询。

   原始查询: {query}

   生成的查询（每行一个）:"""
           
           response = self.llm.invoke(prompt)
           queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
           return [query] + queries[:n]

   # 使用
   rewriter = QueryRewriter()

   original = "Python怎么学啊"
   rewritten = rewriter.rewrite(original)
   expanded = rewriter.expand(original)

   print(f"原始: {original}")
   print(f"改写: {rewritten}")
   print(f"扩展: {expanded}")

查询分解
========

将复杂问题分解为多个子问题。

.. code-block:: python

   class QueryDecomposer:
       """查询分解器"""
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
       
       def decompose(self, query: str) -> list:
           """分解复杂查询"""
           prompt = f"""将以下复杂问题分解为多个简单的子问题。

   问题: {query}

   分解要求:
   - 每个子问题应该是独立可回答的
   - 子问题的答案组合起来能回答原问题
   - 按逻辑顺序排列

   子问题（每行一个）:"""
           
           response = self.llm.invoke(prompt)
           sub_queries = [q.strip().lstrip('0123456789.-) ') 
                         for q in response.content.strip().split('\n') 
                         if q.strip()]
           return sub_queries
       
       def answer_with_decomposition(self, query: str, retriever, llm) -> str:
           """分解后逐个回答"""
           sub_queries = self.decompose(query)
           
           sub_answers = []
           for sq in sub_queries:
               docs = retriever.get_relevant_documents(sq)
               context = "\n".join([d.page_content for d in docs])
               
               answer_prompt = f"""基于以下信息回答问题。

   信息: {context}
   问题: {sq}
   回答:"""
               
               answer = llm.invoke(answer_prompt).content
               sub_answers.append({"question": sq, "answer": answer})
           
           # 综合回答
           synthesis_prompt = f"""基于以下子问题的答案，综合回答原始问题。

   原始问题: {query}

   子问题和答案:
   {chr(10).join([f"Q: {sa['question']}{chr(10)}A: {sa['answer']}" for sa in sub_answers])}

   综合回答:"""
           
           final_answer = llm.invoke(synthesis_prompt).content
           return final_answer

   # 使用
   decomposer = QueryDecomposer()
   complex_query = "比较Python和Java在机器学习领域的应用，哪个更适合初学者？"
   sub_queries = decomposer.decompose(complex_query)

   print("子问题:")
   for i, sq in enumerate(sub_queries, 1):
       print(f"  {i}. {sq}")

上下文压缩
==========

压缩检索到的文档，只保留与问题相关的部分。

.. code-block:: python

   from langchain.retrievers import ContextualCompressionRetriever
   from langchain.retrievers.document_compressors import LLMChainExtractor

   class ContextCompressor:
       """上下文压缩器"""
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
       
       def compress(self, documents: list, query: str) -> list:
           """压缩文档"""
           compressed = []
           
           for doc in documents:
               prompt = f"""从以下文档中提取与问题相关的信息。

   文档: {doc.page_content}

   问题: {query}

   提取的相关信息（如果没有相关信息，输出"无相关信息"）:"""
               
               response = self.llm.invoke(prompt).content.strip()
               
               if response and response != "无相关信息":
                   compressed.append({
                       "content": response,
                       "metadata": doc.metadata
                   })
           
           return compressed

   # 使用 LangChain 的压缩检索器
   def create_compression_retriever(base_retriever):
       compressor = LLMChainExtractor.from_llm(
           ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
       )
       
       return ContextualCompressionRetriever(
           base_compressor=compressor,
           base_retriever=base_retriever
       )

递归检索
========

多层次检索，先检索摘要，再检索详细内容。

.. code-block:: python

   class RecursiveRetriever:
       """递归检索器"""
       
       def __init__(self, summary_store, detail_store):
           self.summary_store = summary_store  # 摘要向量库
           self.detail_store = detail_store    # 详细内容向量库
       
       def retrieve(self, query: str, top_k: int = 3) -> list:
           """递归检索"""
           # 1. 先从摘要中检索
           summaries = self.summary_store.similarity_search(query, k=top_k)
           
           # 2. 获取相关文档的ID
           doc_ids = [s.metadata.get("doc_id") for s in summaries]
           
           # 3. 从详细内容中检索
           all_details = []
           for doc_id in doc_ids:
               details = self.detail_store.similarity_search(
                   query,
                   k=2,
                   filter={"doc_id": doc_id}
               )
               all_details.extend(details)
           
           return all_details

   # 构建层次化索引
   def build_hierarchical_index(documents, embeddings):
       """构建层次化索引"""
       from langchain_community.vectorstores import Chroma
       
       # 生成摘要
       llm = ChatOpenAI(model="gpt-3.5-turbo")
       summaries = []
       
       for i, doc in enumerate(documents):
           summary_prompt = f"用一句话总结以下内容：\n{doc.page_content[:500]}"
           summary = llm.invoke(summary_prompt).content
           
           summaries.append({
               "content": summary,
               "doc_id": f"doc_{i}",
               "metadata": doc.metadata
           })
       
       # 创建摘要向量库
       summary_store = Chroma.from_texts(
           texts=[s["content"] for s in summaries],
           embedding=embeddings,
           metadatas=[{"doc_id": s["doc_id"]} for s in summaries]
       )
       
       # 创建详细内容向量库
       detail_store = Chroma.from_documents(
           documents=documents,
           embedding=embeddings
       )
       
       return summary_store, detail_store

自我反思 RAG
============

让 LLM 评估自己的回答，必要时进行修正。

.. code-block:: python

   class SelfReflectiveRAG:
       """自我反思 RAG"""
       
       def __init__(self, retriever, llm):
           self.retriever = retriever
           self.llm = llm
       
       def query(self, question: str, max_iterations: int = 3) -> dict:
           """带自我反思的查询"""
           docs = self.retriever.get_relevant_documents(question)
           context = "\n".join([d.page_content for d in docs])
           
           # 初始回答
           answer = self._generate_answer(question, context)
           
           for i in range(max_iterations):
               # 自我评估
               evaluation = self._evaluate_answer(question, context, answer)
               
               if evaluation["is_satisfactory"]:
                   break
               
               # 根据反馈改进
               answer = self._improve_answer(
                   question, context, answer, evaluation["feedback"]
               )
           
           return {
               "answer": answer,
               "iterations": i + 1,
               "context": context
           }
       
       def _generate_answer(self, question: str, context: str) -> str:
           prompt = f"""基于以下信息回答问题。

   信息: {context}
   问题: {question}
   回答:"""
           return self.llm.invoke(prompt).content
       
       def _evaluate_answer(self, question: str, context: str, answer: str) -> dict:
           prompt = f"""评估以下回答的质量。

   问题: {question}
   上下文: {context}
   回答: {answer}

   评估标准:
   1. 回答是否准确？
   2. 回答是否完整？
   3. 回答是否基于上下文？

   输出格式:
   满意: 是/否
   反馈: [改进建议]"""
           
           response = self.llm.invoke(prompt).content
           
           is_satisfactory = "满意: 是" in response or "是" in response.split('\n')[0]
           feedback = response.split("反馈:")[-1].strip() if "反馈:" in response else ""
           
           return {
               "is_satisfactory": is_satisfactory,
               "feedback": feedback
           }
       
       def _improve_answer(self, question: str, context: str, 
                          answer: str, feedback: str) -> str:
           prompt = f"""根据反馈改进回答。

   问题: {question}
   上下文: {context}
   原回答: {answer}
   反馈: {feedback}

   改进后的回答:"""
           return self.llm.invoke(prompt).content

多模态 RAG
==========

支持图像和文本的混合检索。

.. code-block:: python

   # 概念示例
   class MultiModalRAG:
       """多模态 RAG"""
       
       def __init__(self, text_store, image_store, vision_model):
           self.text_store = text_store
           self.image_store = image_store
           self.vision_model = vision_model
       
       def query(self, question: str, image=None):
           """多模态查询"""
           results = {
               "text_docs": [],
               "images": []
           }
           
           # 文本检索
           results["text_docs"] = self.text_store.similarity_search(question)
           
           # 图像检索（如果有图像查询）
           if image:
               image_embedding = self.vision_model.encode_image(image)
               results["images"] = self.image_store.search(image_embedding)
           
           # 也可以用文本检索相关图像
           text_embedding = self.vision_model.encode_text(question)
           results["related_images"] = self.image_store.search(text_embedding)
           
           return results

实战：构建高级 RAG 系统
=======================

.. code-block:: python

   from typing import List, Dict

   class AdvancedRAG:
       """高级 RAG 系统"""
       
       def __init__(self, retriever, llm):
           self.retriever = retriever
           self.llm = llm
           self.query_rewriter = QueryRewriter()
           self.context_compressor = ContextCompressor()
       
       def query(
           self,
           question: str,
           use_rewrite: bool = True,
           use_compression: bool = True,
           use_reflection: bool = False
       ) -> Dict:
           """高级查询"""
           
           # 1. 查询改写
           if use_rewrite:
               rewritten_query = self.query_rewriter.rewrite(question)
           else:
               rewritten_query = question
           
           # 2. 检索
           docs = self.retriever.get_relevant_documents(rewritten_query)
           
           # 3. 上下文压缩
           if use_compression:
               compressed = self.context_compressor.compress(docs, question)
               context = "\n".join([c["content"] for c in compressed])
           else:
               context = "\n".join([d.page_content for d in docs])
           
           # 4. 生成回答
           answer = self._generate_answer(question, context)
           
           # 5. 自我反思（可选）
           if use_reflection:
               answer = self._reflect_and_improve(question, context, answer)
           
           return {
               "question": question,
               "rewritten_query": rewritten_query,
               "answer": answer,
               "sources": [d.metadata for d in docs]
           }
       
       def _generate_answer(self, question: str, context: str) -> str:
           prompt = f"""基于以下信息回答问题。如果信息不足，请说明。

   信息:
   {context}

   问题: {question}

   回答:"""
           return self.llm.invoke(prompt).content
       
       def _reflect_and_improve(self, question: str, context: str, answer: str) -> str:
           # 简化的自我反思
           check_prompt = f"""检查以下回答是否准确完整。

   问题: {question}
   上下文: {context}
   回答: {answer}

   如果需要改进，输出改进后的回答；否则输出"OK"。"""
           
           response = self.llm.invoke(check_prompt).content
           
           if response.strip() != "OK":
               return response
           return answer

关键概念总结
============

.. csv-table::
   :header: "技术", "作用", "适用场景"
   :widths: 25, 40, 35

   "查询改写", "优化查询表达", "口语化查询"
   "查询分解", "处理复杂问题", "多步骤问题"
   "上下文压缩", "提取关键信息", "长文档"
   "递归检索", "层次化检索", "大型知识库"
   "自我反思", "提升回答质量", "高精度要求"

下一步
======

在最后一个教程中，我们将学习 RAG 生产部署。

:doc:`tutorial_10_production`
