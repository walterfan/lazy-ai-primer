####################################
Tutorial 8: RAG 系统评估
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

为什么需要评估？
================

评估是改进 RAG 系统的基础，帮助我们：

- 量化系统性能
- 发现问题和瓶颈
- 比较不同方案
- 指导优化方向

RAG 评估维度
============

.. code-block:: text

   RAG 评估框架
   
   ┌─────────────────────────────────────────────────────────────┐
   │                     检索评估                                 │
   │  · 召回率：找到了多少相关文档？                             │
   │  · 精确率：返回的文档有多少是相关的？                       │
   │  · MRR：第一个相关文档的排名                                │
   ├─────────────────────────────────────────────────────────────┤
   │                     生成评估                                 │
   │  · 准确性：回答是否正确？                                   │
   │  · 完整性：回答是否全面？                                   │
   │  · 相关性：回答是否切题？                                   │
   ├─────────────────────────────────────────────────────────────┤
   │                     端到端评估                               │
   │  · 用户满意度                                               │
   │  · 任务完成率                                               │
   │  · 响应时间                                                 │
   └─────────────────────────────────────────────────────────────┘

检索评估指标
============

.. code-block:: python

   from typing import List, Set
   import numpy as np

   class RetrievalMetrics:
       """检索评估指标"""
       
       @staticmethod
       def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
           """Precision@K: 前K个结果中相关的比例"""
           retrieved_k = retrieved[:k]
           relevant_count = sum(1 for doc in retrieved_k if doc in relevant)
           return relevant_count / k if k > 0 else 0.0
       
       @staticmethod
       def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
           """Recall@K: 找到的相关文档占所有相关文档的比例"""
           retrieved_k = set(retrieved[:k])
           found = len(retrieved_k & relevant)
           return found / len(relevant) if relevant else 0.0
       
       @staticmethod
       def mrr(retrieved: List[str], relevant: Set[str]) -> float:
           """MRR: 第一个相关文档的排名倒数"""
           for i, doc in enumerate(retrieved):
               if doc in relevant:
                   return 1.0 / (i + 1)
           return 0.0
       
       @staticmethod
       def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
           """NDCG@K: 归一化折损累积增益"""
           dcg = 0.0
           for i, doc in enumerate(retrieved[:k]):
               if doc in relevant:
                   dcg += 1.0 / np.log2(i + 2)  # i+2 因为 log2(1) = 0
           
           # 理想DCG
           ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
           
           return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

   # 使用示例
   metrics = RetrievalMetrics()

   # 假设检索结果和真实相关文档
   retrieved = ["doc1", "doc3", "doc5", "doc2", "doc4"]
   relevant = {"doc1", "doc2", "doc6"}

   print(f"Precision@3: {metrics.precision_at_k(retrieved, relevant, 3):.3f}")
   print(f"Recall@3: {metrics.recall_at_k(retrieved, relevant, 3):.3f}")
   print(f"MRR: {metrics.mrr(retrieved, relevant):.3f}")
   print(f"NDCG@5: {metrics.ndcg_at_k(retrieved, relevant, 5):.3f}")

生成评估指标
============

1. 基于参考答案的评估
---------------------

.. code-block:: python

   from collections import Counter
   import re

   class GenerationMetrics:
       """生成评估指标"""
       
       @staticmethod
       def exact_match(prediction: str, reference: str) -> float:
           """精确匹配"""
           return 1.0 if prediction.strip() == reference.strip() else 0.0
       
       @staticmethod
       def f1_score(prediction: str, reference: str) -> float:
           """F1 分数（基于词重叠）"""
           pred_tokens = set(prediction.lower().split())
           ref_tokens = set(reference.lower().split())
           
           if not pred_tokens or not ref_tokens:
               return 0.0
           
           common = pred_tokens & ref_tokens
           precision = len(common) / len(pred_tokens)
           recall = len(common) / len(ref_tokens)
           
           if precision + recall == 0:
               return 0.0
           
           return 2 * precision * recall / (precision + recall)
       
       @staticmethod
       def bleu_score(prediction: str, reference: str, n: int = 4) -> float:
           """简化版 BLEU 分数"""
           pred_tokens = prediction.lower().split()
           ref_tokens = reference.lower().split()
           
           if len(pred_tokens) == 0:
               return 0.0
           
           # 计算 n-gram 精确率
           scores = []
           for i in range(1, n + 1):
               pred_ngrams = Counter(
                   tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens) - i + 1)
               )
               ref_ngrams = Counter(
                   tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1)
               )
               
               overlap = sum((pred_ngrams & ref_ngrams).values())
               total = sum(pred_ngrams.values())
               
               scores.append(overlap / total if total > 0 else 0)
           
           # 几何平均
           if all(s > 0 for s in scores):
               return np.exp(np.mean(np.log(scores)))
           return 0.0

   # 使用
   gen_metrics = GenerationMetrics()

   prediction = "Python是一种高级编程语言"
   reference = "Python是一种简洁易读的高级编程语言"

   print(f"Exact Match: {gen_metrics.exact_match(prediction, reference)}")
   print(f"F1 Score: {gen_metrics.f1_score(prediction, reference):.3f}")
   print(f"BLEU Score: {gen_metrics.bleu_score(prediction, reference):.3f}")

2. 基于 LLM 的评估
------------------

.. code-block:: python

   from langchain_openai import ChatOpenAI

   class LLMEvaluator:
       """基于 LLM 的评估器"""
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
       
       def evaluate_faithfulness(self, answer: str, context: str) -> dict:
           """评估答案的忠实度（是否基于上下文）"""
           prompt = f"""评估以下回答是否忠实于给定的上下文。

   上下文：
   {context}

   回答：
   {answer}

   请评估：
   1. 回答中的信息是否都来自上下文？
   2. 是否有编造的信息？

   输出格式：
   忠实度分数（0-10）：
   原因："""
           
           response = self.llm.invoke(prompt).content
           # 解析分数（简化处理）
           try:
               score = int(re.search(r'(\d+)', response.split('\n')[0]).group(1))
           except:
               score = 5
           
           return {"score": score, "explanation": response}
       
       def evaluate_relevance(self, answer: str, question: str) -> dict:
           """评估答案的相关性"""
           prompt = f"""评估以下回答与问题的相关性。

   问题：{question}

   回答：{answer}

   请评估：
   1. 回答是否直接回答了问题？
   2. 是否有无关的信息？

   输出格式：
   相关性分数（0-10）：
   原因："""
           
           response = self.llm.invoke(prompt).content
           try:
               score = int(re.search(r'(\d+)', response.split('\n')[0]).group(1))
           except:
               score = 5
           
           return {"score": score, "explanation": response}
       
       def evaluate_completeness(self, answer: str, question: str, context: str) -> dict:
           """评估答案的完整性"""
           prompt = f"""评估回答是否完整地回答了问题。

   问题：{question}
   上下文：{context}
   回答：{answer}

   请评估：
   1. 回答是否涵盖了问题的所有方面？
   2. 是否遗漏了重要信息？

   输出格式：
   完整性分数（0-10）：
   原因："""
           
           response = self.llm.invoke(prompt).content
           try:
               score = int(re.search(r'(\d+)', response.split('\n')[0]).group(1))
           except:
               score = 5
           
           return {"score": score, "explanation": response}

   # 使用
   evaluator = LLMEvaluator()

   context = "Python是一种高级编程语言，以简洁易读著称。Python支持多种编程范式。"
   question = "Python有什么特点？"
   answer = "Python是一种简洁易读的高级编程语言，支持多种编程范式。"

   faithfulness = evaluator.evaluate_faithfulness(answer, context)
   relevance = evaluator.evaluate_relevance(answer, question)

   print(f"忠实度: {faithfulness['score']}/10")
   print(f"相关性: {relevance['score']}/10")

端到端评估
==========

.. code-block:: python

   from typing import List, Dict
   from dataclasses import dataclass
   import time

   @dataclass
   class TestCase:
       question: str
       expected_answer: str
       relevant_docs: List[str]

   class RAGEvaluator:
       """RAG 系统端到端评估器"""
       
       def __init__(self, rag_system):
           self.rag_system = rag_system
           self.retrieval_metrics = RetrievalMetrics()
           self.generation_metrics = GenerationMetrics()
       
       def evaluate_single(self, test_case: TestCase) -> Dict:
           """评估单个测试用例"""
           start_time = time.time()
           
           # 运行 RAG 系统
           result = self.rag_system.query(test_case.question)
           
           latency = time.time() - start_time
           
           # 检索评估
           retrieved_docs = [doc.page_content for doc in result['retrieved_docs']]
           relevant_set = set(test_case.relevant_docs)
           
           retrieval_scores = {
               "precision@3": self.retrieval_metrics.precision_at_k(
                   retrieved_docs, relevant_set, 3
               ),
               "recall@3": self.retrieval_metrics.recall_at_k(
                   retrieved_docs, relevant_set, 3
               ),
               "mrr": self.retrieval_metrics.mrr(retrieved_docs, relevant_set)
           }
           
           # 生成评估
           generation_scores = {
               "f1": self.generation_metrics.f1_score(
                   result['answer'], test_case.expected_answer
               ),
               "bleu": self.generation_metrics.bleu_score(
                   result['answer'], test_case.expected_answer
               )
           }
           
           return {
               "question": test_case.question,
               "answer": result['answer'],
               "expected": test_case.expected_answer,
               "retrieval": retrieval_scores,
               "generation": generation_scores,
               "latency": latency
           }
       
       def evaluate_batch(self, test_cases: List[TestCase]) -> Dict:
           """批量评估"""
           results = []
           
           for tc in test_cases:
               result = self.evaluate_single(tc)
               results.append(result)
           
           # 聚合指标
           avg_metrics = {
               "retrieval": {
                   "precision@3": np.mean([r["retrieval"]["precision@3"] for r in results]),
                   "recall@3": np.mean([r["retrieval"]["recall@3"] for r in results]),
                   "mrr": np.mean([r["retrieval"]["mrr"] for r in results])
               },
               "generation": {
                   "f1": np.mean([r["generation"]["f1"] for r in results]),
                   "bleu": np.mean([r["generation"]["bleu"] for r in results])
               },
               "latency": {
                   "mean": np.mean([r["latency"] for r in results]),
                   "p95": np.percentile([r["latency"] for r in results], 95)
               }
           }
           
           return {
               "individual_results": results,
               "aggregate_metrics": avg_metrics
           }

构建评估数据集
==============

.. code-block:: python

   import json

   def create_evaluation_dataset():
       """创建评估数据集"""
       dataset = [
           {
               "question": "什么是Python？",
               "expected_answer": "Python是一种高级编程语言，以简洁易读著称。",
               "relevant_docs": ["doc_python_intro", "doc_python_features"]
           },
           {
               "question": "机器学习有哪些类型？",
               "expected_answer": "机器学习主要分为监督学习、无监督学习和强化学习。",
               "relevant_docs": ["doc_ml_types", "doc_ml_intro"]
           },
           # ... 更多测试用例
       ]
       
       return dataset

   def save_dataset(dataset, filepath):
       """保存数据集"""
       with open(filepath, 'w', encoding='utf-8') as f:
           json.dump(dataset, f, ensure_ascii=False, indent=2)

   def load_dataset(filepath):
       """加载数据集"""
       with open(filepath, 'r', encoding='utf-8') as f:
           return json.load(f)

评估报告
========

.. code-block:: python

   def generate_evaluation_report(evaluation_results: Dict) -> str:
       """生成评估报告"""
       agg = evaluation_results["aggregate_metrics"]
       
       report = f"""
   ========================================
   RAG 系统评估报告
   ========================================

   检索性能
   --------
   - Precision@3: {agg['retrieval']['precision@3']:.3f}
   - Recall@3: {agg['retrieval']['recall@3']:.3f}
   - MRR: {agg['retrieval']['mrr']:.3f}

   生成性能
   --------
   - F1 Score: {agg['generation']['f1']:.3f}
   - BLEU Score: {agg['generation']['bleu']:.3f}

   延迟性能
   --------
   - 平均延迟: {agg['latency']['mean']:.2f}s
   - P95 延迟: {agg['latency']['p95']:.2f}s

   测试用例数: {len(evaluation_results['individual_results'])}
   ========================================
   """
       return report

关键概念总结
============

.. csv-table::
   :header: "指标", "解释", "适用场景"
   :widths: 20, 50, 30

   "Precision@K", "前K个结果中相关的比例", "检索精度"
   "Recall@K", "找到的相关文档比例", "检索召回"
   "MRR", "第一个相关结果的排名倒数", "排序质量"
   "F1 Score", "词重叠的调和平均", "生成质量"
   "忠实度", "答案是否基于上下文", "幻觉检测"

下一步
======

在下一个教程中，我们将学习高级 RAG 技术。

:doc:`tutorial_09_advanced_rag`
