####################################
Tutorial 7: PyTorch 深度学习实战
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

PyTorch 简介
============

PyTorch 是一个开源的深度学习框架，以其灵活性和易用性著称。

核心特点：

- **动态计算图**: 即时执行，方便调试
- **Pythonic**: 符合 Python 编程习惯
- **强大的 GPU 支持**: 轻松加速计算
- **丰富的生态**: torchvision, torchaudio, torchtext 等

张量（Tensor）基础
==================

.. code-block:: python

   import torch

   # 1. 创建张量
   # 从列表创建
   a = torch.tensor([1, 2, 3])
   print(f"从列表: {a}")

   # 特殊张量
   zeros = torch.zeros(3, 4)      # 全零
   ones = torch.ones(3, 4)        # 全一
   rand = torch.rand(3, 4)        # 均匀分布 [0, 1)
   randn = torch.randn(3, 4)      # 标准正态分布
   eye = torch.eye(3)             # 单位矩阵
   arange = torch.arange(0, 10, 2)  # 等差序列

   # 2. 张量属性
   x = torch.randn(3, 4, 5)
   print(f"形状: {x.shape}")
   print(f"维度: {x.ndim}")
   print(f"数据类型: {x.dtype}")
   print(f"设备: {x.device}")
   print(f"元素数量: {x.numel()}")

   # 3. 张量运算
   a = torch.tensor([1.0, 2.0, 3.0])
   b = torch.tensor([4.0, 5.0, 6.0])

   print(f"加法: {a + b}")
   print(f"乘法: {a * b}")
   print(f"点积: {torch.dot(a, b)}")
   print(f"矩阵乘法: {torch.randn(2, 3) @ torch.randn(3, 4)}")

   # 4. 形状操作
   x = torch.randn(2, 3, 4)
   print(f"原形状: {x.shape}")
   print(f"reshape: {x.reshape(6, 4).shape}")
   print(f"view: {x.view(2, 12).shape}")
   print(f"transpose: {x.transpose(0, 1).shape}")
   print(f"permute: {x.permute(2, 0, 1).shape}")

   # 5. GPU 加速
   if torch.cuda.is_available():
       device = torch.device("cuda")
       x_gpu = x.to(device)
       print(f"GPU 张量: {x_gpu.device}")

自动微分
========

.. code-block:: python

   import torch

   # 启用梯度追踪
   x = torch.tensor([2.0, 3.0], requires_grad=True)
   y = torch.tensor([4.0, 5.0], requires_grad=True)

   # 前向计算
   z = x * y + x ** 2
   loss = z.sum()

   # 反向传播
   loss.backward()

   print(f"x 的梯度: {x.grad}")  # dL/dx = y + 2x
   print(f"y 的梯度: {y.grad}")  # dL/dy = x

   # 不追踪梯度（推理时）
   with torch.no_grad():
       z = x * y  # 不会记录梯度

   # 分离计算图
   z_detached = z.detach()

构建神经网络
============

.. code-block:: python

   import torch
   import torch.nn as nn

   # 方式1：使用 nn.Sequential
   simple_net = nn.Sequential(
       nn.Linear(784, 256),
       nn.ReLU(),
       nn.Linear(256, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )

   # 方式2：继承 nn.Module（推荐）
   class CustomNet(nn.Module):
       def __init__(self, input_dim, hidden_dim, output_dim):
           super().__init__()
           self.fc1 = nn.Linear(input_dim, hidden_dim)
           self.fc2 = nn.Linear(hidden_dim, hidden_dim)
           self.fc3 = nn.Linear(hidden_dim, output_dim)
           self.relu = nn.ReLU()
           self.dropout = nn.Dropout(0.2)
       
       def forward(self, x):
           x = self.relu(self.fc1(x))
           x = self.dropout(x)
           x = self.relu(self.fc2(x))
           x = self.dropout(x)
           x = self.fc3(x)
           return x

   model = CustomNet(784, 256, 10)

   # 查看模型结构
   print(model)

   # 查看参数
   for name, param in model.named_parameters():
       print(f"{name}: {param.shape}")

   # 参数总数
   total_params = sum(p.numel() for p in model.parameters())
   print(f"总参数量: {total_params:,}")

数据加载
========

.. code-block:: python

   from torch.utils.data import Dataset, DataLoader
   import numpy as np

   # 自定义数据集
   class CustomDataset(Dataset):
       def __init__(self, X, y, transform=None):
           self.X = torch.tensor(X, dtype=torch.float32)
           self.y = torch.tensor(y, dtype=torch.long)
           self.transform = transform
       
       def __len__(self):
           return len(self.X)
       
       def __getitem__(self, idx):
           x = self.X[idx]
           y = self.y[idx]
           
           if self.transform:
               x = self.transform(x)
           
           return x, y

   # 创建数据
   X = np.random.randn(1000, 20)
   y = np.random.randint(0, 3, 1000)

   dataset = CustomDataset(X, y)

   # 创建 DataLoader
   train_loader = DataLoader(
       dataset,
       batch_size=32,
       shuffle=True,
       num_workers=0,  # 多进程加载
       pin_memory=True  # GPU 优化
   )

   # 使用
   for batch_x, batch_y in train_loader:
       print(f"批次形状: X={batch_x.shape}, y={batch_y.shape}")
       break

完整训练流程
============

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader, TensorDataset
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   import matplotlib.pyplot as plt

   # 1. 准备数据
   X, y = make_classification(
       n_samples=5000,
       n_features=20,
       n_informative=15,
       n_classes=3,
       random_state=42
   )

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   train_dataset = TensorDataset(
       torch.tensor(X_train, dtype=torch.float32),
       torch.tensor(y_train, dtype=torch.long)
   )
   test_dataset = TensorDataset(
       torch.tensor(X_test, dtype=torch.float32),
       torch.tensor(y_test, dtype=torch.long)
   )

   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=256)

   # 2. 定义模型
   class Classifier(nn.Module):
       def __init__(self):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(20, 64),
               nn.BatchNorm1d(64),
               nn.ReLU(),
               nn.Dropout(0.3),
               nn.Linear(64, 32),
               nn.BatchNorm1d(32),
               nn.ReLU(),
               nn.Dropout(0.3),
               nn.Linear(32, 3)
           )
       
       def forward(self, x):
           return self.net(x)

   # 3. 训练配置
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = Classifier().to(device)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

   # 4. 训练函数
   def train_epoch(model, loader, criterion, optimizer, device):
       model.train()
       total_loss = 0
       correct = 0
       total = 0
       
       for X, y in loader:
           X, y = X.to(device), y.to(device)
           
           optimizer.zero_grad()
           outputs = model(X)
           loss = criterion(outputs, y)
           loss.backward()
           optimizer.step()
           
           total_loss += loss.item() * X.size(0)
           _, predicted = outputs.max(1)
           correct += predicted.eq(y).sum().item()
           total += y.size(0)
       
       return total_loss / total, correct / total

   def evaluate(model, loader, criterion, device):
       model.eval()
       total_loss = 0
       correct = 0
       total = 0
       
       with torch.no_grad():
           for X, y in loader:
               X, y = X.to(device), y.to(device)
               outputs = model(X)
               loss = criterion(outputs, y)
               
               total_loss += loss.item() * X.size(0)
               _, predicted = outputs.max(1)
               correct += predicted.eq(y).sum().item()
               total += y.size(0)
       
       return total_loss / total, correct / total

   # 5. 训练循环
   history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
   best_acc = 0

   print("开始训练...")
   for epoch in range(50):
       train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
       val_loss, val_acc = evaluate(model, test_loader, criterion, device)
       
       scheduler.step(val_loss)
       
       history['train_loss'].append(train_loss)
       history['train_acc'].append(train_acc)
       history['val_loss'].append(val_loss)
       history['val_acc'].append(val_acc)
       
       if val_acc > best_acc:
           best_acc = val_acc
           torch.save(model.state_dict(), 'best_model.pt')
       
       if (epoch + 1) % 10 == 0:
           print(f"Epoch {epoch+1}: "
                 f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%}, "
                 f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}")

   print(f"\n最佳验证准确率: {best_acc:.2%}")

   # 6. 可视化训练过程
   fig, axes = plt.subplots(1, 2, figsize=(12, 4))

   axes[0].plot(history['train_loss'], label='Train')
   axes[0].plot(history['val_loss'], label='Validation')
   axes[0].set_xlabel('Epoch')
   axes[0].set_ylabel('Loss')
   axes[0].legend()
   axes[0].set_title('Loss Curve')

   axes[1].plot(history['train_acc'], label='Train')
   axes[1].plot(history['val_acc'], label='Validation')
   axes[1].set_xlabel('Epoch')
   axes[1].set_ylabel('Accuracy')
   axes[1].legend()
   axes[1].set_title('Accuracy Curve')

   plt.tight_layout()
   plt.savefig('training_history.png')

模型保存与加载
==============

.. code-block:: python

   # 保存整个模型
   torch.save(model, 'model_complete.pt')

   # 只保存参数（推荐）
   torch.save(model.state_dict(), 'model_params.pt')

   # 保存检查点（包含优化器状态）
   checkpoint = {
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
   }
   torch.save(checkpoint, 'checkpoint.pt')

   # 加载模型
   model = Classifier()
   model.load_state_dict(torch.load('model_params.pt'))
   model.eval()

   # 加载检查点
   checkpoint = torch.load('checkpoint.pt')
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   start_epoch = checkpoint['epoch']

常用技巧
========

.. code-block:: python

   # 1. 梯度裁剪（防止梯度爆炸）
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

   # 2. 学习率预热
   def warmup_lr(epoch, warmup_epochs=5, base_lr=0.001):
       if epoch < warmup_epochs:
           return base_lr * (epoch + 1) / warmup_epochs
       return base_lr

   # 3. 混合精度训练（加速）
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()

   for X, y in train_loader:
       optimizer.zero_grad()
       
       with autocast():
           outputs = model(X)
           loss = criterion(outputs, y)
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()

   # 4. 模型集成
   def ensemble_predict(models, x):
       predictions = []
       for model in models:
           model.eval()
           with torch.no_grad():
               pred = model(x)
               predictions.append(pred)
       return torch.stack(predictions).mean(dim=0)

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "Tensor", "PyTorch 的基本数据结构，类似 NumPy 数组"
   "autograd", "自动微分系统"
   "nn.Module", "神经网络模块的基类"
   "DataLoader", "批量加载数据的工具"
   "optimizer", "更新模型参数的优化器"
   "scheduler", "学习率调度器"

下一步
======

在下一个教程中，我们将学习自然语言处理的基础知识。

:doc:`tutorial_08_nlp_fundamentals`
