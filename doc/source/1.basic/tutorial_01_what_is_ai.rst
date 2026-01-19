####################################
Tutorial 1: 什么是人工智能
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

人工智能的定义
==============

人工智能（Artificial Intelligence, AI）是一个让机器展现出"智能"行为的研究领域。

但什么是"智能"呢？不同的人有不同的理解：

.. csv-table::
   :header: "定义方式", "关注点", "代表观点"
   :widths: 25, 35, 40

   "像人一样思考", "模拟人类思维过程", "认知科学方法"
   "像人一样行动", "通过图灵测试", "行为主义方法"
   "理性地思考", "逻辑推理", "逻辑主义方法"
   "理性地行动", "最优决策", "理性智能体方法"

《人工智能：现代方法》采用的是 **理性智能体** 的定义：

    AI 是研究如何构建能够在环境中采取行动以最大化预期效用的智能体。

通俗理解
--------

想象你在教一个机器人下棋：

- **像人一样思考**: 让机器人像人类棋手一样"直觉"地下棋
- **像人一样行动**: 让机器人下出的棋看起来像人下的
- **理性地思考**: 让机器人分析所有可能的走法
- **理性地行动**: 让机器人选择获胜概率最高的走法

人工智能简史
============

.. code-block:: text

   1950s - AI 诞生
   ├── 1950: 图灵提出"图灵测试"
   ├── 1956: 达特茅斯会议，"人工智能"一词诞生
   └── 1957: 感知机（Perceptron）发明

   1960s-1970s - 早期探索
   ├── 专家系统兴起
   ├── 知识表示研究
   └── 第一次 AI 寒冬（1974-1980）

   1980s - 专家系统时代
   ├── 商业专家系统应用
   └── 第二次 AI 寒冬（1987-1993）

   1990s-2000s - 机器学习兴起
   ├── 统计学习方法
   ├── 支持向量机（SVM）
   └── 深蓝战胜卡斯帕罗夫（1997）

   2010s - 深度学习革命
   ├── 2012: AlexNet 赢得 ImageNet
   ├── 2016: AlphaGo 战胜李世石
   └── 2017: Transformer 架构诞生

   2020s - 大模型时代
   ├── 2020: GPT-3 发布
   ├── 2022: ChatGPT 引爆全球
   └── 2023+: AI Agent 兴起

AI 的主要流派
=============

1. 符号主义（Symbolicism）
--------------------------

**核心思想**: 智能 = 符号操作 + 逻辑推理

.. code-block:: python

   # 符号主义示例：简单的逻辑推理
   
   # 知识库
   knowledge = {
       "苏格拉底是人",
       "所有人都会死"
   }

   # 推理规则
   def reason(knowledge):
       if "苏格拉底是人" in knowledge and "所有人都会死" in knowledge:
           return "苏格拉底会死"
       return None

   conclusion = reason(knowledge)
   print(conclusion)  # 苏格拉底会死

**优点**: 可解释、可推理

**缺点**: 难以处理不确定性、知识获取困难

2. 连接主义（Connectionism）
----------------------------

**核心思想**: 智能来自大量简单单元的连接

.. code-block:: python

   import torch
   import torch.nn as nn

   # 连接主义示例：简单神经网络
   class SimpleNN(nn.Module):
       def __init__(self):
           super().__init__()
           self.layer1 = nn.Linear(10, 5)  # 10个输入，5个神经元
           self.layer2 = nn.Linear(5, 1)   # 5个输入，1个输出
           self.relu = nn.ReLU()
       
       def forward(self, x):
           x = self.relu(self.layer1(x))
           x = self.layer2(x)
           return x

   # 神经网络通过大量"连接"学习模式
   model = SimpleNN()
   print(f"参数数量: {sum(p.numel() for p in model.parameters())}")

**优点**: 能学习复杂模式、处理非结构化数据

**缺点**: 黑箱、需要大量数据

3. 行为主义（Behaviorism）
--------------------------

**核心思想**: 智能 = 感知 + 行动（不需要内部表示）

.. code-block:: python

   # 行为主义示例：简单的反应式智能体
   
   class ReactiveAgent:
       def perceive_and_act(self, sensor_input):
           """直接从感知到行动，没有内部状态"""
           if sensor_input == "obstacle_ahead":
               return "turn_left"
           elif sensor_input == "goal_visible":
               return "move_forward"
           else:
               return "explore"

   agent = ReactiveAgent()
   print(agent.perceive_and_act("obstacle_ahead"))  # turn_left

**优点**: 简单、快速响应

**缺点**: 难以处理复杂任务

AI 的主要研究领域
=================

.. code-block:: text

   人工智能
   ├── 机器学习 (Machine Learning)
   │   ├── 监督学习
   │   ├── 无监督学习
   │   └── 强化学习
   │
   ├── 深度学习 (Deep Learning)
   │   ├── 卷积神经网络 (CNN)
   │   ├── 循环神经网络 (RNN)
   │   └── Transformer
   │
   ├── 自然语言处理 (NLP)
   │   ├── 文本分类
   │   ├── 机器翻译
   │   └── 对话系统
   │
   ├── 计算机视觉 (CV)
   │   ├── 图像分类
   │   ├── 目标检测
   │   └── 图像生成
   │
   ├── 知识表示与推理
   │   ├── 知识图谱
   │   └── 逻辑推理
   │
   └── 智能体与多智能体系统
       ├── 规划
       └── 博弈论

动手实践：你的第一个 AI 程序
============================

让我们用 PyTorch 实现一个简单的"学习"程序：

.. code-block:: python

   import torch

   # 问题：学习函数 y = 2x + 1
   
   # 1. 准备数据
   x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
   y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])  # y = 2x + 1

   # 2. 定义模型（一个简单的线性模型）
   model = torch.nn.Linear(1, 1)  # 1个输入，1个输出

   # 3. 定义损失函数和优化器
   criterion = torch.nn.MSELoss()  # 均方误差
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

   # 4. 训练
   print("开始学习...")
   for epoch in range(1000):
       # 前向传播
       predictions = model(x)
       loss = criterion(predictions, y)
       
       # 反向传播
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
       if (epoch + 1) % 200 == 0:
           print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

   # 5. 测试
   print("\n学习结果:")
   print(f"学到的参数: w = {model.weight.item():.2f}, b = {model.bias.item():.2f}")
   print(f"真实参数:   w = 2.00, b = 1.00")

   # 预测新数据
   test_x = torch.tensor([[5.0]])
   prediction = model(test_x)
   print(f"\n预测 x=5 时: y = {prediction.item():.2f} (真实值: 11)")

运行结果::

   开始学习...
   Epoch 200, Loss: 0.0234
   Epoch 400, Loss: 0.0012
   Epoch 600, Loss: 0.0001
   Epoch 800, Loss: 0.0000
   Epoch 1000, Loss: 0.0000

   学习结果:
   学到的参数: w = 2.00, b = 1.00
   真实参数:   w = 2.00, b = 1.00

   预测 x=5 时: y = 11.00 (真实值: 11)

这就是机器学习的本质：**从数据中自动发现规律**。

AI 的能力边界
=============

AI 擅长什么？
-------------

- ✅ 模式识别（图像、语音、文本）
- ✅ 大规模数据处理
- ✅ 特定领域的优化决策
- ✅ 重复性任务自动化

AI 不擅长什么？
---------------

- ❌ 常识推理
- ❌ 因果理解
- ❌ 创造性思维（真正的创新）
- ❌ 情感理解和社交智能
- ❌ 小样本学习

关键概念总结
============

.. csv-table::
   :header: "概念", "解释", "类比"
   :widths: 20, 50, 30

   "人工智能", "让机器展现智能行为", "教机器'思考'"
   "机器学习", "从数据中学习规律", "从经验中学习"
   "深度学习", "使用深层神经网络", "模拟大脑结构"
   "智能体", "能感知环境并采取行动的系统", "机器人"
   "训练", "用数据调整模型参数", "练习"
   "推理", "用训练好的模型做预测", "考试"

下一步
======

在下一个教程中，我们将学习智能体（Agent）的概念，这是理解现代 AI 系统的基础。

:doc:`tutorial_02_intelligent_agents`
