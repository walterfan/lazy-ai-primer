####################################
Tutorial 6: 神经网络
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是神经网络？
================

**神经网络** 是一种受生物神经系统启发的计算模型，由大量相互连接的"神经元"组成。

.. code-block:: text

   生物神经元                    人工神经元
   
   ┌─────────────┐              ┌─────────────┐
   │  树突       │              │  输入 x₁    │──┐
   │  (接收信号) │              │  输入 x₂    │──┼──► Σ ──► f(·) ──► 输出
   └──────┬──────┘              │  输入 x₃    │──┘
          │                     └─────────────┘
          ▼                     
   ┌─────────────┐              加权求和 + 激活函数
   │  细胞体     │              
   │  (处理信号) │              y = f(w₁x₁ + w₂x₂ + w₃x₃ + b)
   └──────┬──────┘              
          │                     
          ▼                     
   ┌─────────────┐              
   │  轴突       │              
   │  (传递信号) │              
   └─────────────┘              

感知机：最简单的神经网络
========================

.. code-block:: python

   import torch

   class Perceptron:
       """感知机"""
       
       def __init__(self, input_size):
           # 随机初始化权重和偏置
           self.weights = torch.randn(input_size)
           self.bias = torch.randn(1)
       
       def forward(self, x):
           """前向传播"""
           # 加权求和
           z = torch.dot(x, self.weights) + self.bias
           # 阶跃激活函数
           return 1 if z > 0 else 0
       
       def train(self, X, y, epochs=100, lr=0.1):
           """训练"""
           for epoch in range(epochs):
               errors = 0
               for xi, yi in zip(X, y):
                   prediction = self.forward(xi)
                   error = yi - prediction
                   
                   if error != 0:
                       # 更新权重
                       self.weights += lr * error * xi
                       self.bias += lr * error
                       errors += 1
               
               if errors == 0:
                   print(f"Epoch {epoch}: 收敛!")
                   break

   # 示例：学习 AND 门
   X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
   y = torch.tensor([0, 0, 0, 1])  # AND 门的输出

   perceptron = Perceptron(2)
   perceptron.train(X, y)

   print("\nAND 门测试:")
   for xi in X:
       print(f"  {xi.tolist()} -> {perceptron.forward(xi)}")

感知机的局限
------------

感知机只能解决 **线性可分** 问题，无法解决 XOR 问题：

.. code-block:: text

   AND (线性可分)        XOR (非线性可分)
   
   1 ┼─────────────      1 ┼─────────────
     │     ·  /            │  ·        ·
     │       /             │     ?????
     │      /              │
   0 ┼──·──/──·──         0 ┼──·────────·──
     0     1               0     1

多层感知机（MLP）
=================

通过增加 **隐藏层**，神经网络可以学习非线性函数。

.. code-block:: python

   import torch
   import torch.nn as nn

   class MLP(nn.Module):
       """多层感知机"""
       
       def __init__(self, input_size, hidden_size, output_size):
           super().__init__()
           self.layer1 = nn.Linear(input_size, hidden_size)
           self.layer2 = nn.Linear(hidden_size, output_size)
           self.activation = nn.ReLU()
       
       def forward(self, x):
           x = self.layer1(x)
           x = self.activation(x)
           x = self.layer2(x)
           return x

   # 解决 XOR 问题
   X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
   y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # XOR

   model = MLP(2, 4, 1)
   criterion = nn.MSELoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

   for epoch in range(1000):
       outputs = model(X)
       loss = criterion(outputs, y)
       
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
       if (epoch + 1) % 200 == 0:
           print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

   print("\nXOR 测试:")
   with torch.no_grad():
       predictions = model(X)
       for xi, pred in zip(X, predictions):
           print(f"  {xi.tolist()} -> {pred.item():.2f}")

激活函数
========

激活函数引入非线性，让神经网络能学习复杂函数。

.. code-block:: python

   import torch
   import torch.nn as nn
   import matplotlib.pyplot as plt

   # 常用激活函数
   activations = {
       'Sigmoid': nn.Sigmoid(),
       'Tanh': nn.Tanh(),
       'ReLU': nn.ReLU(),
       'LeakyReLU': nn.LeakyReLU(0.1),
       'GELU': nn.GELU()
   }

   x = torch.linspace(-5, 5, 100)

   fig, axes = plt.subplots(1, 5, figsize=(15, 3))

   for ax, (name, func) in zip(axes, activations.items()):
       y = func(x)
       ax.plot(x.numpy(), y.numpy())
       ax.set_title(name)
       ax.grid(True)
       ax.axhline(y=0, color='k', linewidth=0.5)
       ax.axvline(x=0, color='k', linewidth=0.5)

   plt.tight_layout()
   plt.savefig('activations.png')

激活函数比较
------------

.. csv-table::
   :header: "激活函数", "公式", "优点", "缺点"
   :widths: 15, 25, 30, 30

   "Sigmoid", "1/(1+e^(-x))", "输出在(0,1)", "梯度消失"
   "Tanh", "(e^x-e^(-x))/(e^x+e^(-x))", "零中心", "梯度消失"
   "ReLU", "max(0,x)", "计算简单，缓解梯度消失", "死亡ReLU"
   "LeakyReLU", "max(αx,x)", "解决死亡ReLU", "需要调参α"
   "GELU", "x·Φ(x)", "平滑，Transformer常用", "计算稍复杂"

反向传播算法
============

反向传播是训练神经网络的核心算法，利用链式法则计算梯度。

.. code-block:: text

   前向传播:
   输入 x ──► 隐藏层 ──► 输出 y ──► 损失 L
   
   反向传播:
   ∂L/∂w ◄── ∂L/∂y ◄── ∂L/∂L = 1

.. code-block:: python

   import torch

   # 手动实现反向传播（理解原理）
   class ManualMLP:
       def __init__(self, input_size, hidden_size, output_size):
           # 初始化权重
           self.W1 = torch.randn(input_size, hidden_size, requires_grad=False) * 0.1
           self.b1 = torch.zeros(hidden_size, requires_grad=False)
           self.W2 = torch.randn(hidden_size, output_size, requires_grad=False) * 0.1
           self.b2 = torch.zeros(output_size, requires_grad=False)
       
       def forward(self, x):
           """前向传播"""
           # 第一层
           self.z1 = x @ self.W1 + self.b1
           self.a1 = torch.relu(self.z1)
           
           # 第二层
           self.z2 = self.a1 @ self.W2 + self.b2
           self.a2 = self.z2  # 线性输出
           
           return self.a2
       
       def backward(self, x, y, y_pred, lr=0.01):
           """反向传播"""
           batch_size = x.shape[0]
           
           # 输出层梯度
           dL_dz2 = (y_pred - y) / batch_size
           dL_dW2 = self.a1.T @ dL_dz2
           dL_db2 = dL_dz2.sum(dim=0)
           
           # 隐藏层梯度
           dL_da1 = dL_dz2 @ self.W2.T
           dL_dz1 = dL_da1 * (self.z1 > 0).float()  # ReLU 导数
           dL_dW1 = x.T @ dL_dz1
           dL_db1 = dL_dz1.sum(dim=0)
           
           # 更新权重
           self.W2 -= lr * dL_dW2
           self.b2 -= lr * dL_db2
           self.W1 -= lr * dL_dW1
           self.b1 -= lr * dL_db1

   # 使用 PyTorch 自动微分（实际使用）
   class AutogradMLP(nn.Module):
       def __init__(self, input_size, hidden_size, output_size):
           super().__init__()
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.fc2 = nn.Linear(hidden_size, output_size)
       
       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = self.fc2(x)
           return x

   # PyTorch 自动计算梯度
   model = AutogradMLP(2, 4, 1)
   x = torch.randn(10, 2)
   y = torch.randn(10, 1)

   # 前向传播
   y_pred = model(x)
   loss = ((y_pred - y) ** 2).mean()

   # 反向传播（自动）
   loss.backward()

   # 查看梯度
   print("fc1 权重梯度形状:", model.fc1.weight.grad.shape)
   print("fc2 权重梯度形状:", model.fc2.weight.grad.shape)

实战：手写数字识别
==================

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms

   # 1. 数据准备
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ])

   train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
   test_dataset = datasets.MNIST('./data', train=False, transform=transform)

   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=1000)

   # 2. 定义网络
   class DigitClassifier(nn.Module):
       def __init__(self):
           super().__init__()
           self.flatten = nn.Flatten()
           self.layers = nn.Sequential(
               nn.Linear(28 * 28, 256),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(256, 128),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(128, 10)
           )
       
       def forward(self, x):
           x = self.flatten(x)
           return self.layers(x)

   model = DigitClassifier()

   # 3. 训练配置
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # 4. 训练循环
   def train_epoch(model, loader, criterion, optimizer):
       model.train()
       total_loss = 0
       correct = 0
       total = 0
       
       for data, target in loader:
           optimizer.zero_grad()
           output = model(data)
           loss = criterion(output, target)
           loss.backward()
           optimizer.step()
           
           total_loss += loss.item()
           pred = output.argmax(dim=1)
           correct += (pred == target).sum().item()
           total += target.size(0)
       
       return total_loss / len(loader), correct / total

   def test(model, loader):
       model.eval()
       correct = 0
       total = 0
       
       with torch.no_grad():
           for data, target in loader:
               output = model(data)
               pred = output.argmax(dim=1)
               correct += (pred == target).sum().item()
               total += target.size(0)
       
       return correct / total

   # 5. 训练
   print("开始训练 MNIST 分类器...")
   for epoch in range(10):
       train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
       test_acc = test(model, test_loader)
       print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, "
             f"Train Acc={train_acc:.2%}, Test Acc={test_acc:.2%}")

   # 6. 保存模型
   torch.save(model.state_dict(), 'mnist_classifier.pt')
   print("\n模型已保存!")

网络结构设计
============

.. code-block:: python

   # 常见的网络结构模式

   # 1. 逐渐缩小（分类常用）
   class Encoder(nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(784, 256),
               nn.ReLU(),
               nn.Linear(256, 64),
               nn.ReLU(),
               nn.Linear(64, 10)
           )

   # 2. 逐渐扩大（生成常用）
   class Decoder(nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(10, 64),
               nn.ReLU(),
               nn.Linear(64, 256),
               nn.ReLU(),
               nn.Linear(256, 784)
           )

   # 3. 残差连接（深层网络）
   class ResidualBlock(nn.Module):
       def __init__(self, dim):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(dim, dim),
               nn.ReLU(),
               nn.Linear(dim, dim)
           )
       
       def forward(self, x):
           return x + self.layers(x)  # 残差连接

   # 4. 批归一化（加速训练）
   class NormalizedMLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(784, 256),
               nn.BatchNorm1d(256),
               nn.ReLU(),
               nn.Linear(256, 10)
           )

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "神经元", "接收输入、加权求和、激活输出的基本单元"
   "层", "多个神经元的集合"
   "激活函数", "引入非线性的函数"
   "前向传播", "输入到输出的计算过程"
   "反向传播", "计算梯度的算法"
   "梯度下降", "沿梯度方向更新参数"
   "损失函数", "衡量预测误差的函数"

下一步
======

在下一个教程中，我们将学习如何用 PyTorch 构建深度学习模型。

:doc:`tutorial_07_pytorch_deep_learning`
