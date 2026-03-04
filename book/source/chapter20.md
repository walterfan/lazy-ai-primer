(chapter20)=
# 第二十章：Agent 的记忆与状态管理

```{mermaid}
mindmap
  root((Agent记忆与状态))
    为什么需要记忆
      对话连贯性
      经验积累
      个性化服务
    记忆分类
      感知记忆
      短期记忆
      长期记忆
      情景记忆
      语义记忆
      程序记忆
    向量数据库
      Chroma
      Pinecone
      Weaviate
      Milvus
    记忆生命周期
      存储
      检索
      遗忘
      整合
    状态管理
      有限状态机
      状态图
      检查点
```

> "没有记忆的 Agent 就像没有笔记本的侦探——每次都要从头开始。"

## 20.1 为什么 Agent 需要记忆

LLM 本身是**无状态**的——每次调用都是独立的，不记得之前的对话。Agent 需要记忆系统来：

- **维持对话连贯性**：记住用户之前说了什么
- **积累经验**：从过去的成功和失败中学习
- **个性化服务**：记住用户的偏好和习惯
- **长期任务**：跨多次交互完成复杂任务

## 20.2 记忆的分类

```
Agent 记忆系统
├── 感知记忆（Sensory Memory）
│   └── 当前输入的原始信息
├── 短期记忆（Short-term / Working Memory）
│   └── 当前对话的上下文窗口
├── 长期记忆（Long-term Memory）
│   ├── 情景记忆（Episodic Memory）
│   │   └── 过去的具体经历和对话
│   ├── 语义记忆（Semantic Memory）
│   │   └── 知识、事实、概念
│   └── 程序记忆（Procedural Memory）
│       └── 技能、习惯、工作流程
```

### 各类记忆的实现

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Memory:
    content: str
    memory_type: str  # "episodic", "semantic", "procedural"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    importance: float = 0.5  # 0-1
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

class MemorySystem:
    def __init__(self):
        self.short_term: list[dict] = []  # 当前对话
        self.long_term: list[Memory] = []  # 持久化记忆
        self.max_short_term = 20  # 短期记忆容量
    
    def add_to_short_term(self, message: dict):
        """添加到短期记忆（对话历史）"""
        self.short_term.append(message)
        if len(self.short_term) > self.max_short_term:
            # 溢出的内容总结后存入长期记忆
            overflow = self.short_term[:5]
            self.consolidate(overflow)
            self.short_term = self.short_term[5:]
    
    def add_to_long_term(self, content: str, memory_type: str, importance: float = 0.5):
        """添加到长期记忆"""
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance
        )
        self.long_term.append(memory)
    
    def consolidate(self, messages: list[dict]):
        """将短期记忆整合为长期记忆"""
        summary = self.summarize(messages)
        self.add_to_long_term(summary, "episodic", importance=0.3)
    
    def recall(self, query: str, top_k: int = 5) -> list[Memory]:
        """根据查询检索相关记忆"""
        # 使用向量相似度搜索
        relevant = self.vector_search(query, top_k)
        # 更新访问信息
        for memory in relevant:
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
        return relevant
    
    def forget(self, threshold: float = 0.1):
        """遗忘不重要的记忆"""
        self.long_term = [
            m for m in self.long_term
            if m.importance > threshold or m.access_count > 3
        ]
```

## 20.3 向量数据库在记忆系统中的应用

### 主流向量数据库对比

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|---------|
| **Chroma** | 嵌入式 | 轻量、Python 原生 | 原型、小项目 |
| **Pinecone** | 云服务 | 全托管、高性能 | 生产环境 |
| **Weaviate** | 自托管/云 | GraphQL API、多模态 | 企业级 |
| **Milvus** | 自托管 | 高性能、分布式 | 大规模部署 |
| **pgvector** | PostgreSQL 扩展 | 与现有 PG 集成 | 已有 PG 的项目 |

### 使用 Chroma 构建记忆

```python
import chromadb
from chromadb.utils import embedding_functions

# 初始化
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(path="./agent_memory")

# 创建不同类型的记忆集合
episodic = client.get_or_create_collection(
    name="episodic_memory",
    embedding_function=ef
)

semantic = client.get_or_create_collection(
    name="semantic_memory",
    embedding_function=ef
)

# 存储记忆
episodic.add(
    documents=["用户喜欢用 Python 写后端，偏好 FastAPI 框架"],
    metadatas=[{"type": "preference", "importance": 0.8, "date": "2026-03-01"}],
    ids=["mem_001"]
)

# 检索记忆
results = episodic.query(
    query_texts=["用户的技术偏好是什么？"],
    n_results=5
)
print(results["documents"])
```

## 20.4 记忆的存储、检索、遗忘与整合

### 记忆生命周期

```
新信息 → 评估重要性 → 存储
                         ↓
                    定期整合（合并相似记忆）
                         ↓
                    检索使用（更新访问频率）
                         ↓
                    遗忘（清理低价值记忆）
```

### 重要性评估

```python
async def assess_importance(content: str, context: str) -> float:
    """使用 LLM 评估记忆的重要性"""
    prompt = f"""
    请评估以下信息的重要性（0-1分）：
    
    信息：{content}
    上下文：{context}
    
    评估标准：
    - 用户明确表达的偏好：0.8-1.0
    - 项目相关的技术决策：0.7-0.9
    - 一般性对话内容：0.2-0.4
    - 临时性信息：0.1-0.2
    
    只返回一个数字。
    """
    score = await llm.generate(prompt)
    return float(score.strip())
```

## 20.5 状态管理

### 有限状态机（FSM）

```python
from enum import Enum

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    USING_TOOL = "using_tool"
    WAITING_FOR_USER = "waiting_for_user"
    ERROR = "error"

class AgentFSM:
    def __init__(self):
        self.state = AgentState.IDLE
        self.transitions = {
            AgentState.IDLE: [AgentState.THINKING],
            AgentState.THINKING: [AgentState.USING_TOOL, AgentState.WAITING_FOR_USER, AgentState.IDLE],
            AgentState.USING_TOOL: [AgentState.THINKING, AgentState.ERROR],
            AgentState.WAITING_FOR_USER: [AgentState.THINKING],
            AgentState.ERROR: [AgentState.IDLE, AgentState.THINKING],
        }
    
    def transition(self, new_state: AgentState):
        if new_state in self.transitions.get(self.state, []):
            self.state = new_state
        else:
            raise ValueError(f"Invalid transition: {self.state} → {new_state}")
```

### LangGraph 检查点

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

# 创建带检查点的图
memory = MemorySaver()
graph = StateGraph(AgentState)

# ... 定义节点和边 ...

app = graph.compile(checkpointer=memory)

# 运行时指定 thread_id，自动保存和恢复状态
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [user_message]}, config)

# 下次调用同一 thread_id，自动恢复之前的状态
result2 = app.invoke({"messages": [another_message]}, config)
```

## 20.6 实战：构建带长期记忆的 Agent

```python
"""完整示例：带长期记忆的个人助手 Agent"""
import chromadb
from openai import OpenAI
from datetime import datetime

class MemoryAgent:
    def __init__(self):
        self.llm = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./memory_db")
        self.memories = self.chroma.get_or_create_collection("agent_memories")
        self.conversation_history = []
    
    def remember(self, content: str, importance: float = 0.5):
        """存储新记忆"""
        self.memories.add(
            documents=[content],
            metadatas=[{
                "importance": importance,
                "timestamp": datetime.now().isoformat(),
                "access_count": 0
            }],
            ids=[f"mem_{datetime.now().timestamp()}"]
        )
    
    def recall(self, query: str, n: int = 5) -> list[str]:
        """检索相关记忆"""
        if self.memories.count() == 0:
            return []
        results = self.memories.query(query_texts=[query], n_results=min(n, self.memories.count()))
        return results["documents"][0] if results["documents"] else []
    
    def chat(self, user_message: str) -> str:
        """与用户对话，自动使用记忆"""
        # 1. 检索相关记忆
        relevant_memories = self.recall(user_message)
        memory_context = "\n".join(f"- {m}" for m in relevant_memories) if relevant_memories else "无相关记忆"
        
        # 2. 构建 Prompt
        system_prompt = f"""你是一个有记忆的个人助手。

相关记忆：
{memory_context}

请基于记忆和当前对话回答用户。如果用户分享了重要信息（偏好、事实、决定），
请在回答末尾用 [REMEMBER: ...] 标记需要记住的内容。"""
        
        self.conversation_history.append({"role": "user", "content": user_message})
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                *self.conversation_history[-10:]  # 最近10轮对话
            ]
        )
        
        reply = response.choices[0].message.content
        
        # 3. 提取需要记忆的内容
        if "[REMEMBER:" in reply:
            memory_content = reply.split("[REMEMBER:")[1].split("]")[0].strip()
            self.remember(memory_content, importance=0.8)
            reply = reply.split("[REMEMBER:")[0].strip()
        
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

# 使用
agent = MemoryAgent()
print(agent.chat("我叫 Walter，是一名 Python 开发者"))
# → "你好 Walter！很高兴认识你..." [自动记住名字和职业]

print(agent.chat("我最近在学 Rust"))
# → "学 Rust 是个好选择！作为 Python 开发者..." [记住了之前的信息]
```

## 20.7 本章小结

记忆系统是 Agent 从"一次性工具"进化为"持续助手"的关键。通过合理设计短期记忆、长期记忆和状态管理，Agent 可以提供更个性化、更连贯的服务。

核心要点：
1. **分层记忆**：不同类型的信息用不同的记忆机制
2. **向量检索**：用语义相似度而非关键词匹配来检索记忆
3. **遗忘机制**：不是所有信息都值得记住
4. **状态持久化**：使用检查点保存和恢复 Agent 状态

```{admonition} 思考题
:class: hint
1. Agent 的记忆系统和人类的记忆有什么相似和不同？
2. 如何处理记忆中的错误信息（用户纠正了之前的说法）？
3. 记忆系统的隐私问题如何解决？
```
