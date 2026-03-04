(chapter07)=
# 第七章：大语言模型基础 — 开发者需要知道的

```{mermaid}
mindmap
  root((大语言模型基础))
    Transformer架构
      注意力机制
      Multi-Head
      位置编码
    主流模型
      GPT-4o
      Claude
      Gemini
      DeepSeek
      Llama
    核心概念
      Token
      上下文窗口
      Temperature
    Prompt Engineering
      明确角色
      Few-shot
      Chain-of-Thought
      ReAct
    能力边界
      能做的
      不能做的
      幻觉问题
    本地vs云端
      隐私考量
      成本对比
      性能差异
```

> "大语言模型不是魔法，但理解它的工作原理，能让你更好地驾驭它。"

## 7.1 Transformer 架构：直觉理解

2017 年，Google 发表了划时代的论文《Attention Is All You Need》，提出了 **Transformer** 架构。这是当今所有大语言模型的基础。

### 注意力机制的直觉

想象你在阅读一句话："**小猫**坐在垫子上，**它**正在打盹。"

当你读到"它"时，你的大脑会自动"注意"到前面的"小猫"，而不是"垫子"。这就是**注意力机制（Attention Mechanism）**的核心思想——让模型在处理每个词时，能够"关注"到输入中最相关的部分。

```
输入: "The cat sat on the mat, it was sleeping"

处理 "it" 时的注意力权重:
  The  → 0.05
  cat  → 0.60  ← 高注意力！
  sat  → 0.08
  on   → 0.02
  the  → 0.03
  mat  → 0.15
  it   → 0.05
  was  → 0.02
```

### Transformer 的核心组件

```
输入文本 → Tokenizer → Embedding → [Transformer Blocks × N] → 输出概率

Transformer Block:
┌─────────────────────────┐
│  Multi-Head Attention    │  ← 让每个 token 关注其他 token
│  ↓                      │
│  Feed-Forward Network    │  ← 非线性变换
│  ↓                      │
│  Layer Normalization     │  ← 稳定训练
│  + Residual Connection   │  ← 防止梯度消失
└─────────────────────────┘
```

关键概念：
- **Self-Attention**：每个词都能"看到"序列中的所有其他词
- **Multi-Head**：多个注意力头并行工作，捕捉不同类型的关系
- **位置编码**：因为 Transformer 没有循环结构，需要额外编码词的位置信息

## 7.2 主流大语言模型对比

| 模型 | 公司 | 参数量 | 上下文窗口 | 特点 |
|------|------|--------|-----------|------|
| **GPT-4o** | OpenAI | 未公开 | 128K | 多模态，综合能力强 |
| **Claude 3.5 Sonnet** | Anthropic | 未公开 | 200K | 代码能力突出，安全性好 |
| **Gemini 2.0** | Google | 未公开 | 2M | 超长上下文，多模态 |
| **DeepSeek-V3** | DeepSeek | 671B(MoE) | 128K | 开源，性价比极高 |
| **Llama 3.1** | Meta | 405B | 128K | 开源，社区生态丰富 |
| **Qwen 2.5** | 阿里 | 72B | 128K | 中文能力强，开源 |

## 7.3 核心概念

### Token

Token 是 LLM 处理文本的基本单位，不等于"字"或"词"：

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hello, world! 你好，世界！"
tokens = enc.encode(text)
print(f"文本: {text}")
print(f"Token 数: {len(tokens)}")
print(f"Tokens: {tokens}")

# 英文大约 1 token ≈ 4 个字符 ≈ 0.75 个单词
# 中文大约 1 token ≈ 1-2 个汉字
```

### 上下文窗口（Context Window）

上下文窗口是模型一次能处理的最大 Token 数：

```
GPT-4o:           128,000 tokens ≈ 一本 300 页的书
Claude 3.5:       200,000 tokens ≈ 一本 500 页的书
Gemini 2.0:     2,000,000 tokens ≈ 多本书
```

### Temperature

Temperature 控制输出的随机性：

```python
# Temperature = 0: 确定性输出，总是选择概率最高的 token
# Temperature = 0.7: 适度随机，平衡创造性和准确性
# Temperature = 1.0: 高随机性，更有创造力但可能不准确

from openai import OpenAI
client = OpenAI()

# 代码生成用低 temperature
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0,  # 确定性，适合代码
    messages=[{"role": "user", "content": "Write a Python sort function"}]
)

# 创意写作用高 temperature
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.9,  # 高随机性，适合创意
    messages=[{"role": "user", "content": "Write a poem about coding"}]
)
```

## 7.4 Prompt Engineering 基本技巧

### 1. 明确角色和任务

```python
# ❌ 模糊的 Prompt
"帮我写代码"

# ✅ 明确的 Prompt
"""你是一位资深 Python 后端工程师。
请用 FastAPI 实现一个用户注册 API，要求：
1. 接收 email 和 password
2. 验证 email 格式
3. 密码至少 8 位，包含大小写和数字
4. 返回 JWT token
5. 包含完整的错误处理
6. 添加 type hints 和 docstring"""
```

### 2. 提供示例（Few-shot）

```python
prompt = """将以下自然语言转换为 SQL 查询。

示例 1:
输入: "查找所有年龄大于 30 的用户"
输出: SELECT * FROM users WHERE age > 30;

示例 2:
输入: "统计每个部门的平均工资"
输出: SELECT department, AVG(salary) FROM employees GROUP BY department;

现在请转换:
输入: "查找订单金额最高的前 10 个客户"
输出: """
```

### 3. 思维链（Chain-of-Thought）

```python
prompt = """请一步一步分析这段代码的时间复杂度：

```python
def find_pairs(arr, target):
    seen = set()
    pairs = []
    for num in arr:
        complement = target - num
        if complement in seen:
            pairs.append((complement, num))
        seen.add(num)
    return pairs
```

请按以下步骤分析：
1. 识别所有循环和嵌套
2. 分析每个操作的复杂度
3. 计算总体时间复杂度
4. 给出最终结论"""
```

### 4. ReAct 模式（推理 + 行动）

```
问题：北京今天的天气适合跑步吗？

思考：我需要先查询北京今天的天气信息。
行动：调用天气 API 查询北京天气
观察：北京今天晴，气温 22°C，PM2.5 35，风力 2 级

思考：气温适宜，空气质量良好，风力不大，适合户外运动。
回答：北京今天天气非常适合跑步！晴天，22°C，空气质量良好。
```

## 7.5 模型的能力边界

### 能做的 ✅

- 代码生成、补全、解释、重构
- 文本摘要、翻译、写作
- 数据分析和可视化代码
- API 文档生成
- Bug 定位和修复建议
- 架构设计建议

### 不能做的 ❌

- **精确数学计算**：大数乘法可能出错
- **实时信息**：训练数据有截止日期
- **确定性推理**：复杂逻辑推理可能出错
- **记忆持久化**：每次对话独立（除非外部记忆）
- **执行代码**：只能生成，不能运行（除非有工具）
- **保证正确性**：可能产生"幻觉"（看起来正确但实际错误的输出）

## 7.6 本地模型 vs 云端 API

| 维度 | 本地模型 | 云端 API |
|------|---------|---------|
| **隐私** | ✅ 数据不出本地 | ⚠️ 数据发送到云端 |
| **成本** | 一次性硬件投入 | 按 Token 计费 |
| **性能** | 受限于本地硬件 | 强大的云端算力 |
| **模型质量** | 开源模型（稍弱） | 最强闭源模型 |
| **延迟** | 低（本地推理） | 取决于网络 |
| **维护** | 需要自己管理 | 无需维护 |

```python
# 本地模型：使用 Ollama
import ollama

response = ollama.chat(model='deepseek-coder-v2', messages=[
    {'role': 'user', 'content': 'Write a Python quicksort'}
])
print(response['message']['content'])

# 云端 API：使用 OpenAI
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a Python quicksort"}]
)
print(response.choices[0].message.content)
```

## 7.7 本章小结

作为开发者，你不需要成为机器学习专家，但理解 LLM 的基本原理和使用技巧，能让你更高效地利用 AI 工具。关键要点：

1. **Transformer 的注意力机制**是 LLM 的核心
2. **Token 和上下文窗口**决定了模型能处理多少信息
3. **Prompt Engineering**是与 LLM 高效沟通的关键技能
4. **了解模型的边界**，避免在不适合的场景使用

```{admonition} 思考题
:class: hint
1. 为什么 Chain-of-Thought 提示能提高模型的推理能力？
2. 在你的工作中，哪些任务适合用本地模型，哪些适合用云端 API？
3. 如何验证 LLM 生成内容的正确性？
```
