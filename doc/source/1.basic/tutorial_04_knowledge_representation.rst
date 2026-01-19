####################################
Tutorial 4: 知识表示与推理
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是知识表示？
================

**知识表示** 是将现实世界的知识编码成计算机可以处理的形式。

为什么重要？

- 让 AI 能够"理解"世界
- 支持推理和问题求解
- 实现知识的存储和检索

.. code-block:: text

   现实世界的知识         知识表示          推理系统
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ 苏格拉底是人 │ ──► │ human(苏格拉底)│ ──► │ 苏格拉底会死 │
   │ 人都会死    │     │ ∀x human(x)→ │     │             │
   │             │     │   mortal(x) │     │             │
   └─────────────┘     └─────────────┘     └─────────────┘

知识表示方法
============

1. 命题逻辑
-----------

最简单的逻辑系统，使用命题和逻辑连接词。

.. code-block:: python

   class PropositionalLogic:
       """命题逻辑"""
       
       def __init__(self):
           self.knowledge_base = set()
       
       def tell(self, proposition):
           """添加知识"""
           self.knowledge_base.add(proposition)
       
       def ask(self, query):
           """查询"""
           return query in self.knowledge_base

   # 示例
   kb = PropositionalLogic()
   kb.tell("下雨")
   kb.tell("下雨 → 带伞")

   print(kb.ask("下雨"))  # True

**逻辑运算符**:

- ¬ (NOT): 非
- ∧ (AND): 与
- ∨ (OR): 或
- → (IMPLIES): 蕴含
- ↔ (IFF): 当且仅当

.. code-block:: python

   # 简单的命题逻辑推理
   class SimpleReasoner:
       def __init__(self):
           self.facts = {}
           self.rules = []
       
       def add_fact(self, name, value):
           self.facts[name] = value
       
       def add_rule(self, condition, conclusion):
           """添加规则: condition → conclusion"""
           self.rules.append((condition, conclusion))
       
       def infer(self):
           """前向推理"""
           changed = True
           while changed:
               changed = False
               for condition, conclusion in self.rules:
                   if self.evaluate(condition) and conclusion not in self.facts:
                       self.facts[conclusion] = True
                       changed = True
                       print(f"推理得出: {conclusion}")
       
       def evaluate(self, condition):
           """评估条件"""
           if isinstance(condition, str):
               return self.facts.get(condition, False)
           elif condition[0] == 'AND':
               return all(self.evaluate(c) for c in condition[1:])
           elif condition[0] == 'OR':
               return any(self.evaluate(c) for c in condition[1:])
           return False

   # 使用示例
   reasoner = SimpleReasoner()
   reasoner.add_fact("下雨", True)
   reasoner.add_rule("下雨", "地湿")
   reasoner.add_rule(("AND", "地湿", "没带伞"), "会淋湿")
   reasoner.add_fact("没带伞", True)

   reasoner.infer()
   # 输出:
   # 推理得出: 地湿
   # 推理得出: 会淋湿

2. 一阶谓词逻辑
---------------

比命题逻辑更强大，支持变量、量词和谓词。

.. code-block:: python

   class FirstOrderLogic:
       """一阶谓词逻辑（简化版）"""
       
       def __init__(self):
           self.facts = []      # 事实
           self.rules = []      # 规则
           self.constants = set()  # 常量
       
       def add_fact(self, predicate, *args):
           """添加事实: predicate(arg1, arg2, ...)"""
           self.facts.append((predicate, args))
           self.constants.update(args)
       
       def add_rule(self, conditions, conclusion):
           """添加规则: conditions → conclusion"""
           self.rules.append((conditions, conclusion))
       
       def query(self, predicate, *args):
           """查询事实"""
           for fact_pred, fact_args in self.facts:
               if fact_pred == predicate:
                   if self.unify(args, fact_args):
                       return True
           return False
       
       def unify(self, args1, args2):
           """简单的合一算法"""
           if len(args1) != len(args2):
               return False
           for a1, a2 in zip(args1, args2):
               if a1.startswith('?'):  # 变量
                   continue
               if a2.startswith('?'):  # 变量
                   continue
               if a1 != a2:
                   return False
           return True
       
       def forward_chain(self):
           """前向链推理"""
           changed = True
           while changed:
               changed = False
               for conditions, (pred, args) in self.rules:
                   # 尝试找到满足条件的绑定
                   bindings = self.find_bindings(conditions)
                   for binding in bindings:
                       new_args = tuple(binding.get(a, a) for a in args)
                       if not self.query(pred, *new_args):
                           self.add_fact(pred, *new_args)
                           changed = True
                           print(f"推理: {pred}{new_args}")
       
       def find_bindings(self, conditions):
           """找到满足条件的变量绑定"""
           if not conditions:
               return [{}]
           
           pred, args = conditions[0]
           rest = conditions[1:]
           
           results = []
           for fact_pred, fact_args in self.facts:
               if fact_pred == pred and len(args) == len(fact_args):
                   binding = {}
                   match = True
                   for a, fa in zip(args, fact_args):
                       if a.startswith('?'):
                           binding[a] = fa
                       elif a != fa:
                           match = False
                           break
                   
                   if match:
                       # 递归处理剩余条件
                       for rest_binding in self.find_bindings(rest):
                           combined = {**binding, **rest_binding}
                           results.append(combined)
           
           return results

   # 经典推理示例
   fol = FirstOrderLogic()

   # 事实
   fol.add_fact("human", "苏格拉底")
   fol.add_fact("human", "柏拉图")

   # 规则: ∀x human(x) → mortal(x)
   fol.add_rule(
       [("human", ("?x",))],
       ("mortal", ("?x",))
   )

   print("初始事实:", fol.facts)
   fol.forward_chain()
   print("推理后:", fol.facts)

3. 语义网络
-----------

用图结构表示知识，节点表示概念，边表示关系。

.. code-block:: python

   class SemanticNetwork:
       """语义网络"""
       
       def __init__(self):
           self.nodes = {}  # 节点
           self.edges = []  # 边: (源, 关系, 目标)
       
       def add_node(self, name, properties=None):
           self.nodes[name] = properties or {}
       
       def add_edge(self, source, relation, target):
           self.edges.append((source, relation, target))
       
       def get_relations(self, node, relation=None):
           """获取节点的关系"""
           results = []
           for s, r, t in self.edges:
               if s == node:
                   if relation is None or r == relation:
                       results.append((r, t))
           return results
       
       def is_a(self, node, category):
           """检查继承关系"""
           # 直接检查
           for r, t in self.get_relations(node, "is_a"):
               if t == category:
                   return True
               # 递归检查
               if self.is_a(t, category):
                   return True
           return False
       
       def get_property(self, node, property_name):
           """获取属性（支持继承）"""
           # 先检查自身
           if node in self.nodes and property_name in self.nodes[node]:
               return self.nodes[node][property_name]
           
           # 检查父类
           for r, parent in self.get_relations(node, "is_a"):
               result = self.get_property(parent, property_name)
               if result is not None:
                   return result
           
           return None

   # 示例：动物分类
   net = SemanticNetwork()

   # 添加概念
   net.add_node("动物", {"能移动": True})
   net.add_node("鸟", {"有翅膀": True, "能飞": True})
   net.add_node("企鹅", {"能飞": False, "生活环境": "南极"})
   net.add_node("麻雀")

   # 添加关系
   net.add_edge("鸟", "is_a", "动物")
   net.add_edge("企鹅", "is_a", "鸟")
   net.add_edge("麻雀", "is_a", "鸟")

   # 查询
   print(f"企鹅是动物吗? {net.is_a('企鹅', '动物')}")  # True
   print(f"企鹅能飞吗? {net.get_property('企鹅', '能飞')}")  # False
   print(f"麻雀能飞吗? {net.get_property('麻雀', '能飞')}")  # True (继承)

4. 知识图谱
-----------

现代的知识表示方法，使用三元组 (实体, 关系, 实体)。

.. code-block:: python

   class KnowledgeGraph:
       """简单的知识图谱"""
       
       def __init__(self):
           self.triples = []  # (头实体, 关系, 尾实体)
           self.entity_index = {}  # 实体索引
           self.relation_index = {}  # 关系索引
       
       def add_triple(self, head, relation, tail):
           """添加三元组"""
           triple = (head, relation, tail)
           self.triples.append(triple)
           
           # 建立索引
           if head not in self.entity_index:
               self.entity_index[head] = []
           self.entity_index[head].append(triple)
           
           if relation not in self.relation_index:
               self.relation_index[relation] = []
           self.relation_index[relation].append(triple)
       
       def query(self, head=None, relation=None, tail=None):
           """查询三元组"""
           results = []
           for h, r, t in self.triples:
               if (head is None or h == head) and \
                  (relation is None or r == relation) and \
                  (tail is None or t == tail):
                   results.append((h, r, t))
           return results
       
       def get_neighbors(self, entity):
           """获取实体的邻居"""
           neighbors = []
           for h, r, t in self.triples:
               if h == entity:
                   neighbors.append((r, t, "out"))
               if t == entity:
                   neighbors.append((r, h, "in"))
           return neighbors
       
       def find_path(self, start, end, max_depth=3):
           """寻找两个实体之间的路径"""
           from collections import deque
           
           queue = deque([(start, [start])])
           visited = {start}
           
           while queue:
               current, path = queue.popleft()
               
               if len(path) > max_depth:
                   continue
               
               for relation, neighbor, direction in self.get_neighbors(current):
                   if neighbor == end:
                       return path + [(relation, direction), neighbor]
                   
                   if neighbor not in visited:
                       visited.add(neighbor)
                       queue.append((neighbor, path + [(relation, direction), neighbor]))
           
           return None

   # 示例：构建简单的知识图谱
   kg = KnowledgeGraph()

   # 添加知识
   kg.add_triple("北京", "是首都", "中国")
   kg.add_triple("上海", "位于", "中国")
   kg.add_triple("中国", "是", "国家")
   kg.add_triple("李白", "出生于", "中国")
   kg.add_triple("李白", "职业", "诗人")
   kg.add_triple("杜甫", "职业", "诗人")
   kg.add_triple("李白", "朋友", "杜甫")

   # 查询
   print("中国的相关知识:")
   for triple in kg.query(head="中国"):
       print(f"  {triple}")

   print("\n所有诗人:")
   for h, r, t in kg.query(relation="职业", tail="诗人"):
       print(f"  {h}")

   print("\n李白到中国的路径:")
   path = kg.find_path("李白", "中国")
   print(f"  {path}")

用 PyTorch 实现知识图谱嵌入
===========================

知识图谱嵌入将实体和关系映射到向量空间：

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim

   class TransE(nn.Module):
       """TransE 知识图谱嵌入模型
       
       核心思想: head + relation ≈ tail
       """
       
       def __init__(self, num_entities, num_relations, embedding_dim=50):
           super().__init__()
           self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
           self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
           
           # 初始化
           nn.init.xavier_uniform_(self.entity_embeddings.weight)
           nn.init.xavier_uniform_(self.relation_embeddings.weight)
       
       def forward(self, heads, relations, tails):
           """计算得分（距离越小越好）"""
           h = self.entity_embeddings(heads)
           r = self.relation_embeddings(relations)
           t = self.entity_embeddings(tails)
           
           # TransE: ||h + r - t||
           score = torch.norm(h + r - t, p=2, dim=1)
           return score
       
       def predict(self, head, relation, top_k=5):
           """预测尾实体"""
           h = self.entity_embeddings(head)
           r = self.relation_embeddings(relation)
           
           # 计算与所有实体的距离
           all_entities = self.entity_embeddings.weight
           scores = torch.norm(h + r - all_entities, p=2, dim=1)
           
           # 返回得分最低的k个
           _, indices = torch.topk(scores, top_k, largest=False)
           return indices

   def train_transe(triples, num_entities, num_relations, epochs=100):
       """训练 TransE 模型"""
       model = TransE(num_entities, num_relations)
       optimizer = optim.Adam(model.parameters(), lr=0.01)
       margin = 1.0
       
       # 转换为张量
       heads = torch.tensor([t[0] for t in triples])
       relations = torch.tensor([t[1] for t in triples])
       tails = torch.tensor([t[2] for t in triples])
       
       for epoch in range(epochs):
           # 正样本得分
           pos_scores = model(heads, relations, tails)
           
           # 负采样（随机替换尾实体）
           neg_tails = torch.randint(0, num_entities, (len(triples),))
           neg_scores = model(heads, relations, neg_tails)
           
           # Margin-based loss
           loss = torch.mean(torch.relu(pos_scores - neg_scores + margin))
           
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           # 归一化实体嵌入
           with torch.no_grad():
               model.entity_embeddings.weight.data = \
                   nn.functional.normalize(model.entity_embeddings.weight.data, dim=1)
           
           if (epoch + 1) % 20 == 0:
               print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
       
       return model

   # 示例
   if __name__ == "__main__":
       # 简单的知识图谱
       # 实体: 0=北京, 1=中国, 2=上海, 3=亚洲
       # 关系: 0=首都, 1=位于
       triples = [
           (0, 0, 1),  # 北京 是首都 中国
           (0, 1, 1),  # 北京 位于 中国
           (2, 1, 1),  # 上海 位于 中国
           (1, 1, 3),  # 中国 位于 亚洲
       ]
       
       model = train_transe(triples, num_entities=4, num_relations=2, epochs=100)
       
       # 预测：上海 位于 ?
       predictions = model.predict(
           torch.tensor([2]),  # 上海
           torch.tensor([1]),  # 位于
           top_k=2
       )
       print(f"\n上海 位于 ? 预测: {predictions.tolist()}")
       # 期望输出 1 (中国) 排名靠前

关键概念总结
============

.. csv-table::
   :header: "概念", "解释", "应用"
   :widths: 20, 45, 35

   "命题逻辑", "使用命题和逻辑连接词", "简单推理"
   "一阶逻辑", "支持变量和量词", "复杂推理"
   "语义网络", "图结构表示概念关系", "知识组织"
   "知识图谱", "三元组存储知识", "搜索引擎、问答"
   "知识嵌入", "将知识映射到向量", "知识补全、推荐"

下一步
======

在下一个教程中，我们将学习机器学习的基础知识。

:doc:`tutorial_05_machine_learning`
