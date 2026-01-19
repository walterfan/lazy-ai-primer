####################################
Tutorial 9: 计算机视觉基础
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是计算机视觉？
==================

**计算机视觉（CV）** 是让计算机"看懂"图像和视频的技术。

.. code-block:: text

   计算机视觉任务
   ├── 图像分类（这是猫还是狗？）
   ├── 目标检测（图中有什么？在哪里？）
   ├── 语义分割（每个像素属于什么类别？）
   ├── 实例分割（区分不同的物体实例）
   ├── 姿态估计（人体关键点检测）
   └── 图像生成（GAN、扩散模型）

图像基础
========

.. code-block:: python

   import torch
   import numpy as np
   from PIL import Image
   import matplotlib.pyplot as plt

   # 图像是一个三维数组: [高度, 宽度, 通道]
   # 通道通常是 RGB (红、绿、蓝)

   # 创建一个简单的图像
   img = np.zeros((100, 100, 3), dtype=np.uint8)
   img[20:80, 20:80, 0] = 255  # 红色方块

   # 显示
   plt.imshow(img)
   plt.title("简单图像")
   plt.savefig('simple_image.png')

   # PyTorch 中的图像格式: [通道, 高度, 宽度]
   # 需要转换: [H, W, C] -> [C, H, W]
   tensor_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
   print(f"张量形状: {tensor_img.shape}")  # [3, 100, 100]

图像预处理
==========

.. code-block:: python

   from torchvision import transforms

   # 常用的图像变换
   transform = transforms.Compose([
       transforms.Resize((224, 224)),      # 调整大小
       transforms.ToTensor(),              # 转为张量 [0, 1]
       transforms.Normalize(               # 标准化
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       )
   ])

   # 数据增强（训练时）
   train_transform = transforms.Compose([
       transforms.RandomResizedCrop(224),  # 随机裁剪
       transforms.RandomHorizontalFlip(),  # 随机水平翻转
       transforms.ColorJitter(             # 颜色抖动
           brightness=0.2,
           contrast=0.2,
           saturation=0.2
       ),
       transforms.RandomRotation(15),      # 随机旋转
       transforms.ToTensor(),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       )
   ])

卷积神经网络（CNN）
===================

CNN 是处理图像的核心架构。

卷积操作
--------

.. code-block:: python

   import torch
   import torch.nn as nn

   # 卷积层
   # in_channels: 输入通道数
   # out_channels: 输出通道数（卷积核数量）
   # kernel_size: 卷积核大小
   conv = nn.Conv2d(
       in_channels=3,
       out_channels=16,
       kernel_size=3,
       stride=1,
       padding=1
   )

   # 输入: [batch, channels, height, width]
   x = torch.randn(1, 3, 224, 224)
   output = conv(x)
   print(f"输出形状: {output.shape}")  # [1, 16, 224, 224]

   # 池化层
   maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
   pooled = maxpool(output)
   print(f"池化后: {pooled.shape}")  # [1, 16, 112, 112]

简单的 CNN
----------

.. code-block:: python

   class SimpleCNN(nn.Module):
       """简单的卷积神经网络"""
       
       def __init__(self, num_classes=10):
           super().__init__()
           
           # 卷积层
           self.features = nn.Sequential(
               # 第一个卷积块
               nn.Conv2d(3, 32, kernel_size=3, padding=1),
               nn.BatchNorm2d(32),
               nn.ReLU(),
               nn.MaxPool2d(2),  # 224 -> 112
               
               # 第二个卷积块
               nn.Conv2d(32, 64, kernel_size=3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.MaxPool2d(2),  # 112 -> 56
               
               # 第三个卷积块
               nn.Conv2d(64, 128, kernel_size=3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(),
               nn.MaxPool2d(2),  # 56 -> 28
           )
           
           # 分类器
           self.classifier = nn.Sequential(
               nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
               nn.Flatten(),
               nn.Linear(128, 256),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(256, num_classes)
           )
       
       def forward(self, x):
           x = self.features(x)
           x = self.classifier(x)
           return x

   model = SimpleCNN(num_classes=10)
   print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

经典 CNN 架构
=============

.. code-block:: python

   # VGG 风格的块
   def make_vgg_block(in_channels, out_channels, num_convs):
       layers = []
       for _ in range(num_convs):
           layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
           layers.append(nn.ReLU())
           in_channels = out_channels
       layers.append(nn.MaxPool2d(2, 2))
       return nn.Sequential(*layers)

   # ResNet 风格的残差块
   class ResidualBlock(nn.Module):
       def __init__(self, in_channels, out_channels, stride=1):
           super().__init__()
           self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
           self.bn1 = nn.BatchNorm2d(out_channels)
           self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
           self.bn2 = nn.BatchNorm2d(out_channels)
           
           # 捷径连接
           self.shortcut = nn.Sequential()
           if stride != 1 or in_channels != out_channels:
               self.shortcut = nn.Sequential(
                   nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                   nn.BatchNorm2d(out_channels)
               )
       
       def forward(self, x):
           out = torch.relu(self.bn1(self.conv1(x)))
           out = self.bn2(self.conv2(out))
           out += self.shortcut(x)  # 残差连接
           return torch.relu(out)

实战：CIFAR-10 图像分类
=======================

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms

   # 1. 数据准备
   transform_train = transforms.Compose([
       transforms.RandomCrop(32, padding=4),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
   ])

   transform_test = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
   ])

   train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
   test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)

   train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
   test_loader = DataLoader(test_dataset, batch_size=256)

   # 类别名称
   classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

   # 2. 定义网络
   class CIFAR10Net(nn.Module):
       def __init__(self):
           super().__init__()
           self.features = nn.Sequential(
               # Block 1
               nn.Conv2d(3, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.Conv2d(64, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Dropout(0.25),
               
               # Block 2
               nn.Conv2d(64, 128, 3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(),
               nn.Conv2d(128, 128, 3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Dropout(0.25),
               
               # Block 3
               nn.Conv2d(128, 256, 3, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(),
               nn.Conv2d(256, 256, 3, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Dropout(0.25),
           )
           
           self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(256 * 4 * 4, 512),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(512, 10)
           )
       
       def forward(self, x):
           x = self.features(x)
           x = self.classifier(x)
           return x

   # 3. 训练
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = CIFAR10Net().to(device)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

   def train_epoch(model, loader, criterion, optimizer, device):
       model.train()
       total_loss = 0
       correct = 0
       total = 0
       
       for images, labels in loader:
           images, labels = images.to(device), labels.to(device)
           
           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           
           total_loss += loss.item()
           _, predicted = outputs.max(1)
           correct += predicted.eq(labels).sum().item()
           total += labels.size(0)
       
       return total_loss / len(loader), correct / total

   def test(model, loader, device):
       model.eval()
       correct = 0
       total = 0
       
       with torch.no_grad():
           for images, labels in loader:
               images, labels = images.to(device), labels.to(device)
               outputs = model(images)
               _, predicted = outputs.max(1)
               correct += predicted.eq(labels).sum().item()
               total += labels.size(0)
       
       return correct / total

   print("开始训练 CIFAR-10 分类器...")
   for epoch in range(20):
       train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
       test_acc = test(model, test_loader, device)
       scheduler.step(train_loss)
       
       if (epoch + 1) % 5 == 0:
           print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, "
                 f"Train Acc={train_acc:.2%}, Test Acc={test_acc:.2%}")

使用预训练模型
==============

.. code-block:: python

   from torchvision import models

   # 加载预训练的 ResNet
   resnet = models.resnet18(pretrained=True)

   # 冻结特征提取层
   for param in resnet.parameters():
       param.requires_grad = False

   # 替换分类头
   num_classes = 10
   resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

   # 只训练新的分类头
   optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

   # 微调（解冻部分层）
   for param in resnet.layer4.parameters():
       param.requires_grad = True

目标检测简介
============

.. code-block:: python

   # 目标检测输出: 边界框 + 类别
   # 边界框格式: [x_min, y_min, x_max, y_max] 或 [x_center, y_center, width, height]

   class SimpleDetector(nn.Module):
       """简化的目标检测器"""
       
       def __init__(self, num_classes):
           super().__init__()
           # 特征提取
           self.backbone = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Conv2d(64, 128, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
           )
           
           # 预测头
           self.classifier = nn.Conv2d(128, num_classes, 1)  # 类别
           self.regressor = nn.Conv2d(128, 4, 1)             # 边界框
       
       def forward(self, x):
           features = self.backbone(x)
           class_pred = self.classifier(features)
           bbox_pred = self.regressor(features)
           return class_pred, bbox_pred

   # 计算 IoU (Intersection over Union)
   def iou(box1, box2):
       """计算两个边界框的 IoU"""
       x1 = max(box1[0], box2[0])
       y1 = max(box1[1], box2[1])
       x2 = min(box1[2], box2[2])
       y2 = min(box1[3], box2[3])
       
       intersection = max(0, x2 - x1) * max(0, y2 - y1)
       area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
       area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
       union = area1 + area2 - intersection
       
       return intersection / union if union > 0 else 0

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "卷积", "滑动窗口提取局部特征"
   "池化", "降低空间分辨率，增加感受野"
   "特征图", "卷积层的输出"
   "感受野", "输出像素对应的输入区域"
   "残差连接", "跳跃连接，解决梯度消失"
   "迁移学习", "使用预训练模型"

下一步
======

在最后一个教程中，我们将学习强化学习的基础知识。

:doc:`tutorial_10_reinforcement_learning`
