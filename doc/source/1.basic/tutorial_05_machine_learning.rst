####################################
Tutorial 5: 机器学习基础
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是机器学习？
================

**机器学习** 是让计算机从数据中自动学习规律的方法，而不需要显式编程。

.. code-block:: text

   传统编程:
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │   数据     │ + │   规则     │ ──► │   结果     │
   └────────────┘     └────────────┘     └────────────┘

   机器学习:
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │   数据     │ + │   结果     │ ──► │   规则     │
   └────────────┘     └────────────┘     └────────────┘

通俗理解
--------

想象你在教小孩认识猫：

- **传统编程**: "猫有四条腿、有尾巴、会喵喵叫..."（写规则）
- **机器学习**: 给小孩看很多猫的图片，让他自己总结特征（从数据学习）

机器学习的类型
==============

.. code-block:: text

   机器学习
   ├── 监督学习 (Supervised Learning)
   │   ├── 分类 (Classification)
   │   └── 回归 (Regression)
   │
   ├── 无监督学习 (Unsupervised Learning)
   │   ├── 聚类 (Clustering)
   │   └── 降维 (Dimensionality Reduction)
   │
   ├── 强化学习 (Reinforcement Learning)
   │   └── 通过奖励学习策略
   │
   └── 半监督学习 / 自监督学习
       └── 结合标注和未标注数据

1. 监督学习
-----------

有标签的数据，学习输入到输出的映射。

.. code-block:: python

   import torch
   import torch.nn as nn

   # 分类示例：判断邮件是否为垃圾邮件
   # 输入: 邮件特征向量
   # 输出: 0 (正常) 或 1 (垃圾)

   # 回归示例：预测房价
   # 输入: 房屋特征 (面积、位置、房龄等)
   # 输出: 价格 (连续值)

2. 无监督学习
-------------

没有标签，发现数据中的结构。

.. code-block:: python

   # 聚类示例：客户分群
   # 输入: 客户行为数据
   # 输出: 将客户分成若干群组

   # 降维示例：数据可视化
   # 输入: 高维数据
   # 输出: 低维表示

3. 强化学习
-----------

通过与环境交互，学习最优策略。

.. code-block:: python

   # 示例：学习下棋
   # 状态: 棋盘局面
   # 动作: 走哪一步
   # 奖励: 赢了+1, 输了-1

核心概念
========

1. 数据集
---------

.. code-block:: python

   import torch
   from torch.utils.data import Dataset, DataLoader

   class SimpleDataset(Dataset):
       """简单数据集"""
       
       def __init__(self, X, y):
           self.X = torch.tensor(X, dtype=torch.float32)
           self.y = torch.tensor(y, dtype=torch.float32)
       
       def __len__(self):
           return len(self.X)
       
       def __getitem__(self, idx):
           return self.X[idx], self.y[idx]

   # 数据集划分
   def train_test_split(X, y, test_ratio=0.2):
       n = len(X)
       indices = torch.randperm(n)
       split = int(n * (1 - test_ratio))
       
       train_idx = indices[:split]
       test_idx = indices[split:]
       
       return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

2. 模型
-------

.. code-block:: python

   # 线性模型
   class LinearModel(nn.Module):
       def __init__(self, input_dim, output_dim):
           super().__init__()
           self.linear = nn.Linear(input_dim, output_dim)
       
       def forward(self, x):
           return self.linear(x)

   # 多层感知机
   class MLP(nn.Module):
       def __init__(self, input_dim, hidden_dim, output_dim):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(input_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, output_dim)
           )
       
       def forward(self, x):
           return self.layers(x)

3. 损失函数
-----------

衡量预测与真实值的差距。

.. code-block:: python

   # 回归任务：均方误差
   mse_loss = nn.MSELoss()

   # 分类任务：交叉熵
   ce_loss = nn.CrossEntropyLoss()

   # 二分类：二元交叉熵
   bce_loss = nn.BCEWithLogitsLoss()

   # 示例
   predictions = torch.tensor([2.5, 0.0, 2.1])
   targets = torch.tensor([3.0, -0.5, 2.0])
   
   loss = mse_loss(predictions, targets)
   print(f"MSE Loss: {loss.item():.4f}")

4. 优化器
---------

更新模型参数以最小化损失。

.. code-block:: python

   model = LinearModel(10, 1)

   # 随机梯度下降
   sgd = torch.optim.SGD(model.parameters(), lr=0.01)

   # Adam（常用）
   adam = torch.optim.Adam(model.parameters(), lr=0.001)

   # 学习率调度
   scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=10, gamma=0.1)

实战：线性回归
==============

.. code-block:: python

   import torch
   import torch.nn as nn
   import matplotlib.pyplot as plt

   # 1. 生成数据
   torch.manual_seed(42)
   
   # 真实关系: y = 3x + 2 + 噪声
   X = torch.randn(100, 1)
   y = 3 * X + 2 + torch.randn(100, 1) * 0.5

   # 2. 定义模型
   model = nn.Linear(1, 1)

   # 3. 定义损失和优化器
   criterion = nn.MSELoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

   # 4. 训练
   losses = []
   for epoch in range(100):
       # 前向传播
       predictions = model(X)
       loss = criterion(predictions, y)
       
       # 反向传播
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
       losses.append(loss.item())
       
       if (epoch + 1) % 20 == 0:
           print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

   # 5. 查看学到的参数
   print(f"\n学到的参数:")
   print(f"  权重 w = {model.weight.item():.2f} (真实值: 3)")
   print(f"  偏置 b = {model.bias.item():.2f} (真实值: 2)")

   # 6. 可视化
   plt.figure(figsize=(12, 4))

   plt.subplot(1, 2, 1)
   plt.scatter(X.numpy(), y.numpy(), alpha=0.5, label='数据点')
   plt.plot(X.numpy(), model(X).detach().numpy(), 'r-', label='学到的直线')
   plt.xlabel('X')
   plt.ylabel('y')
   plt.legend()
   plt.title('线性回归结果')

   plt.subplot(1, 2, 2)
   plt.plot(losses)
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('训练损失曲线')

   plt.tight_layout()
   plt.savefig('linear_regression.png')
   print("\n图表已保存到 linear_regression.png")

实战：分类问题
==============

.. code-block:: python

   import torch
   import torch.nn as nn
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   # 1. 生成分类数据
   X, y = make_classification(
       n_samples=1000,
       n_features=20,
       n_informative=10,
       n_classes=2,
       random_state=42
   )

   # 转换为 PyTorch 张量
   X = torch.tensor(X, dtype=torch.float32)
   y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

   # 划分训练集和测试集
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # 2. 定义模型
   class Classifier(nn.Module):
       def __init__(self, input_dim):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(input_dim, 64),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(64, 32),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(32, 1)
           )
       
       def forward(self, x):
           return self.layers(x)

   model = Classifier(20)

   # 3. 训练
   criterion = nn.BCEWithLogitsLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

   for epoch in range(100):
       model.train()
       
       # 前向传播
       outputs = model(X_train)
       loss = criterion(outputs, y_train)
       
       # 反向传播
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
       if (epoch + 1) % 20 == 0:
           # 评估
           model.eval()
           with torch.no_grad():
               train_pred = (torch.sigmoid(model(X_train)) > 0.5).float()
               test_pred = (torch.sigmoid(model(X_test)) > 0.5).float()
               
               train_acc = (train_pred == y_train).float().mean()
               test_acc = (test_pred == y_test).float().mean()
           
           print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, "
                 f"Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}")

过拟合与正则化
==============

**过拟合**: 模型在训练数据上表现好，但在新数据上表现差。

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                                                          │
   │  欠拟合          恰好          过拟合                    │
   │                                                          │
   │    ──────       ~~~~~~       ~∿~∿~∿~                    │
   │   ·  ·  ·     ·  ·  ·     ·  ·  ·                       │
   │  ·    ·  ·   ·    ·  ·   ·    ·  ·                      │
   │                                                          │
   │  训练误差高    训练误差低    训练误差很低                │
   │  测试误差高    测试误差低    测试误差高                  │
   │                                                          │
   └─────────────────────────────────────────────────────────┘

防止过拟合的方法
----------------

.. code-block:: python

   # 1. L2 正则化（权重衰减）
   optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

   # 2. Dropout
   class ModelWithDropout(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(100, 50)
           self.dropout = nn.Dropout(0.5)  # 50% 的神经元随机失活
           self.fc2 = nn.Linear(50, 10)
       
       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = self.dropout(x)  # 训练时随机失活
           x = self.fc2(x)
           return x

   # 3. 早停（Early Stopping）
   best_loss = float('inf')
   patience = 10
   counter = 0

   for epoch in range(1000):
       train_loss = train_one_epoch()
       val_loss = evaluate()
       
       if val_loss < best_loss:
           best_loss = val_loss
           counter = 0
           # 保存最佳模型
           torch.save(model.state_dict(), 'best_model.pt')
       else:
           counter += 1
           if counter >= patience:
               print("Early stopping!")
               break

   # 4. 数据增强（后续教程详细介绍）

模型评估指标
============

分类任务
--------

.. code-block:: python

   def evaluate_classification(y_true, y_pred):
       """分类评估指标"""
       # 混淆矩阵
       TP = ((y_pred == 1) & (y_true == 1)).sum().item()
       TN = ((y_pred == 0) & (y_true == 0)).sum().item()
       FP = ((y_pred == 1) & (y_true == 0)).sum().item()
       FN = ((y_pred == 0) & (y_true == 1)).sum().item()
       
       # 准确率
       accuracy = (TP + TN) / (TP + TN + FP + FN)
       
       # 精确率
       precision = TP / (TP + FP) if (TP + FP) > 0 else 0
       
       # 召回率
       recall = TP / (TP + FN) if (TP + FN) > 0 else 0
       
       # F1 分数
       f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
       
       return {
           'accuracy': accuracy,
           'precision': precision,
           'recall': recall,
           'f1': f1
       }

回归任务
--------

.. code-block:: python

   def evaluate_regression(y_true, y_pred):
       """回归评估指标"""
       # 均方误差
       mse = ((y_pred - y_true) ** 2).mean().item()
       
       # 均方根误差
       rmse = mse ** 0.5
       
       # 平均绝对误差
       mae = (y_pred - y_true).abs().mean().item()
       
       # R² 决定系数
       ss_res = ((y_true - y_pred) ** 2).sum()
       ss_tot = ((y_true - y_true.mean()) ** 2).sum()
       r2 = 1 - ss_res / ss_tot
       
       return {
           'mse': mse,
           'rmse': rmse,
           'mae': mae,
           'r2': r2.item()
       }

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "监督学习", "从标注数据学习输入到输出的映射"
   "无监督学习", "从无标注数据发现结构"
   "损失函数", "衡量预测与真实值的差距"
   "优化器", "更新模型参数的算法"
   "过拟合", "模型过度拟合训练数据"
   "正则化", "防止过拟合的技术"
   "验证集", "用于调参和早停的数据"

下一步
======

在下一个教程中，我们将深入学习神经网络的原理。

:doc:`tutorial_06_neural_networks`
