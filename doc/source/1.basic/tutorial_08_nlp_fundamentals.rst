####################################
Tutorial 8: 自然语言处理基础
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是 NLP？
============

**自然语言处理（NLP）** 是让计算机理解、处理和生成人类语言的技术。

.. code-block:: text

   NLP 应用领域
   ├── 文本分类（垃圾邮件检测、情感分析）
   ├── 命名实体识别（识别人名、地名等）
   ├── 机器翻译（中文→英文）
   ├── 问答系统（ChatGPT）
   ├── 文本摘要
   └── 文本生成

NLP 的挑战
----------

- **歧义性**: "苹果很好吃" vs "苹果股价上涨"
- **上下文依赖**: "它" 指代什么？
- **语言多样性**: 方言、俚语、新词
- **常识推理**: 需要背景知识

文本预处理
==========

.. code-block:: python

   import re
   from collections import Counter

   class TextPreprocessor:
       """文本预处理器"""
       
       def __init__(self):
           pass
       
       def clean_text(self, text):
           """清洗文本"""
           # 转小写
           text = text.lower()
           # 去除特殊字符
           text = re.sub(r'[^\w\s]', '', text)
           # 去除多余空格
           text = re.sub(r'\s+', ' ', text).strip()
           return text
       
       def tokenize(self, text):
           """分词"""
           return text.split()
       
       def remove_stopwords(self, tokens, stopwords):
           """去除停用词"""
           return [t for t in tokens if t not in stopwords]

   # 示例
   preprocessor = TextPreprocessor()

   text = "Hello, World! This is a NLP tutorial. NLP is amazing!"
   clean = preprocessor.clean_text(text)
   tokens = preprocessor.tokenize(clean)

   print(f"原文: {text}")
   print(f"清洗后: {clean}")
   print(f"分词: {tokens}")

中文分词
--------

.. code-block:: python

   # 中文需要特殊的分词工具
   # pip install jieba

   import jieba

   text = "自然语言处理是人工智能的重要分支"
   tokens = list(jieba.cut(text))
   print(f"分词结果: {tokens}")
   # ['自然语言', '处理', '是', '人工智能', '的', '重要', '分支']

文本表示
========

1. 词袋模型（Bag of Words）
---------------------------

.. code-block:: python

   from collections import Counter
   import numpy as np

   class BagOfWords:
       """词袋模型"""
       
       def __init__(self):
           self.vocabulary = {}
       
       def fit(self, documents):
           """构建词汇表"""
           all_words = set()
           for doc in documents:
               all_words.update(doc.split())
           
           self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
           return self
       
       def transform(self, documents):
           """转换为向量"""
           vectors = []
           for doc in documents:
               vector = np.zeros(len(self.vocabulary))
               for word in doc.split():
                   if word in self.vocabulary:
                       vector[self.vocabulary[word]] += 1
               vectors.append(vector)
           return np.array(vectors)

   # 示例
   docs = [
       "i love machine learning",
       "machine learning is great",
       "deep learning is a subset of machine learning"
   ]

   bow = BagOfWords()
   bow.fit(docs)
   vectors = bow.transform(docs)

   print(f"词汇表: {bow.vocabulary}")
   print(f"向量形状: {vectors.shape}")

2. TF-IDF
---------

.. code-block:: python

   import numpy as np
   from collections import Counter

   class TFIDF:
       """TF-IDF 向量化"""
       
       def __init__(self):
           self.vocabulary = {}
           self.idf = {}
       
       def fit(self, documents):
           # 构建词汇表
           all_words = set()
           for doc in documents:
               all_words.update(doc.split())
           self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
           
           # 计算 IDF
           n_docs = len(documents)
           doc_freq = Counter()
           for doc in documents:
               unique_words = set(doc.split())
               doc_freq.update(unique_words)
           
           self.idf = {
               word: np.log(n_docs / (df + 1)) + 1
               for word, df in doc_freq.items()
           }
           return self
       
       def transform(self, documents):
           vectors = []
           for doc in documents:
               # 计算 TF
               word_counts = Counter(doc.split())
               total_words = len(doc.split())
               
               vector = np.zeros(len(self.vocabulary))
               for word, count in word_counts.items():
                   if word in self.vocabulary:
                       tf = count / total_words
                       idf = self.idf.get(word, 1)
                       vector[self.vocabulary[word]] = tf * idf
               
               vectors.append(vector)
           return np.array(vectors)

3. 词嵌入（Word Embeddings）
----------------------------

词嵌入将词映射到稠密的向量空间。

.. code-block:: python

   import torch
   import torch.nn as nn

   # 使用 PyTorch 的 Embedding 层
   vocab_size = 10000
   embedding_dim = 100

   embedding = nn.Embedding(vocab_size, embedding_dim)

   # 词索引
   word_indices = torch.tensor([1, 5, 100, 999])

   # 获取词向量
   word_vectors = embedding(word_indices)
   print(f"词向量形状: {word_vectors.shape}")  # [4, 100]

   # 计算词相似度
   def cosine_similarity(v1, v2):
       return torch.dot(v1, v2) / (v1.norm() * v2.norm())

Word2Vec 简化实现
-----------------

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim

   class SkipGram(nn.Module):
       """Skip-gram 模型"""
       
       def __init__(self, vocab_size, embedding_dim):
           super().__init__()
           self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
           self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
       
       def forward(self, center, context, negative):
           """
           center: 中心词索引 [batch]
           context: 上下文词索引 [batch]
           negative: 负采样词索引 [batch, num_neg]
           """
           # 获取嵌入
           center_emb = self.center_embeddings(center)      # [batch, dim]
           context_emb = self.context_embeddings(context)   # [batch, dim]
           neg_emb = self.context_embeddings(negative)      # [batch, num_neg, dim]
           
           # 正样本得分
           pos_score = torch.sum(center_emb * context_emb, dim=1)  # [batch]
           pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)
           
           # 负样本得分
           neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()  # [batch, num_neg]
           neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-10).sum(dim=1)
           
           return (pos_loss + neg_loss).mean()
       
       def get_embedding(self, word_idx):
           return self.center_embeddings(torch.tensor([word_idx]))

循环神经网络（RNN）
===================

RNN 能处理序列数据，记住之前的信息。

.. code-block:: python

   import torch
   import torch.nn as nn

   # 基本 RNN
   class SimpleRNN(nn.Module):
       def __init__(self, input_size, hidden_size, output_size):
           super().__init__()
           self.hidden_size = hidden_size
           self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
           self.fc = nn.Linear(hidden_size, output_size)
       
       def forward(self, x):
           # x: [batch, seq_len, input_size]
           output, hidden = self.rnn(x)
           # output: [batch, seq_len, hidden_size]
           # hidden: [1, batch, hidden_size]
           
           # 使用最后一个时间步的输出
           last_output = output[:, -1, :]
           return self.fc(last_output)

   # LSTM（解决梯度消失问题）
   class LSTMClassifier(nn.Module):
       def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, embedding_dim)
           self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
           self.fc = nn.Linear(hidden_size * 2, output_size)
           self.dropout = nn.Dropout(0.5)
       
       def forward(self, x):
           # x: [batch, seq_len]
           embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]
           output, (hidden, cell) = self.lstm(embedded)
           
           # 拼接双向 LSTM 的最后隐藏状态
           hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
           return self.fc(self.dropout(hidden_cat))

实战：情感分析
==============

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import Dataset, DataLoader
   from collections import Counter

   # 1. 数据准备
   # 模拟数据
   texts = [
       "this movie is great",
       "i love this film",
       "amazing performance",
       "terrible movie",
       "i hate this",
       "worst film ever",
       "excellent story",
       "boring and slow"
   ]
   labels = [1, 1, 1, 0, 0, 0, 1, 0]  # 1: 正面, 0: 负面

   # 构建词汇表
   all_words = []
   for text in texts:
       all_words.extend(text.lower().split())

   word_counts = Counter(all_words)
   vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.most_common())}
   vocab['<PAD>'] = 0
   vocab['<UNK>'] = 1

   # 文本转索引
   def text_to_indices(text, vocab, max_len=10):
       indices = [vocab.get(word, vocab['<UNK>']) for word in text.lower().split()]
       # 填充或截断
       if len(indices) < max_len:
           indices += [vocab['<PAD>']] * (max_len - len(indices))
       else:
           indices = indices[:max_len]
       return indices

   # 2. 数据集
   class SentimentDataset(Dataset):
       def __init__(self, texts, labels, vocab, max_len=10):
           self.data = [
               (torch.tensor(text_to_indices(text, vocab, max_len)), label)
               for text, label in zip(texts, labels)
           ]
       
       def __len__(self):
           return len(self.data)
       
       def __getitem__(self, idx):
           return self.data[idx]

   dataset = SentimentDataset(texts, labels, vocab)
   loader = DataLoader(dataset, batch_size=4, shuffle=True)

   # 3. 模型
   class SentimentLSTM(nn.Module):
       def __init__(self, vocab_size, embedding_dim=32, hidden_size=64):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
           self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
           self.fc = nn.Linear(hidden_size, 1)
       
       def forward(self, x):
           embedded = self.embedding(x)
           _, (hidden, _) = self.lstm(embedded)
           return self.fc(hidden.squeeze(0))

   model = SentimentLSTM(len(vocab))
   criterion = nn.BCEWithLogitsLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.01)

   # 4. 训练
   print("训练情感分析模型...")
   for epoch in range(100):
       total_loss = 0
       for batch_x, batch_y in loader:
           optimizer.zero_grad()
           outputs = model(batch_x).squeeze()
           loss = criterion(outputs, batch_y.float())
           loss.backward()
           optimizer.step()
           total_loss += loss.item()
       
       if (epoch + 1) % 20 == 0:
           print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

   # 5. 测试
   model.eval()
   test_texts = ["this is amazing", "i hate it"]
   for text in test_texts:
       indices = torch.tensor([text_to_indices(text, vocab)])
       with torch.no_grad():
           output = torch.sigmoid(model(indices))
           sentiment = "正面" if output > 0.5 else "负面"
           print(f"'{text}' -> {sentiment} ({output.item():.2f})")

Transformer 简介
================

Transformer 是现代 NLP 的基础架构（GPT、BERT 等都基于它）。

.. code-block:: python

   import torch
   import torch.nn as nn
   import math

   class SelfAttention(nn.Module):
       """自注意力机制"""
       
       def __init__(self, embed_dim, num_heads):
           super().__init__()
           self.num_heads = num_heads
           self.head_dim = embed_dim // num_heads
           
           self.query = nn.Linear(embed_dim, embed_dim)
           self.key = nn.Linear(embed_dim, embed_dim)
           self.value = nn.Linear(embed_dim, embed_dim)
           self.out = nn.Linear(embed_dim, embed_dim)
       
       def forward(self, x):
           batch_size, seq_len, embed_dim = x.shape
           
           # 计算 Q, K, V
           Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
           K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
           V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
           
           # 注意力分数
           scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
           attention = torch.softmax(scores, dim=-1)
           
           # 加权求和
           out = torch.matmul(attention, V)
           out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
           
           return self.out(out)

   class TransformerBlock(nn.Module):
       """Transformer 块"""
       
       def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
           super().__init__()
           self.attention = SelfAttention(embed_dim, num_heads)
           self.norm1 = nn.LayerNorm(embed_dim)
           self.norm2 = nn.LayerNorm(embed_dim)
           self.ff = nn.Sequential(
               nn.Linear(embed_dim, ff_dim),
               nn.GELU(),
               nn.Linear(ff_dim, embed_dim)
           )
           self.dropout = nn.Dropout(dropout)
       
       def forward(self, x):
           # 自注意力 + 残差连接
           x = x + self.dropout(self.attention(self.norm1(x)))
           # 前馈网络 + 残差连接
           x = x + self.dropout(self.ff(self.norm2(x)))
           return x

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "分词", "将文本切分成词或子词"
   "词嵌入", "将词映射到稠密向量"
   "RNN/LSTM", "处理序列的循环神经网络"
   "注意力机制", "让模型关注重要部分"
   "Transformer", "基于注意力的现代架构"

下一步
======

在下一个教程中，我们将学习计算机视觉的基础知识。

:doc:`tutorial_09_computer_vision`
