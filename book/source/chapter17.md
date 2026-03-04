(chapter17)=
# 第十七章：Agent 架构模式深度解析

```{mermaid}
mindmap
  root((Agent架构模式))
    ReAct模式
      推理+行动
      交替执行
    Plan-and-Execute
      先规划后执行
      任务分解
    Reflexion模式
      自我反思
      经验学习
    Tool Use模式
      工具选择
      工具调用
    RAG增强
      知识检索
      上下文注入
    记忆系统
      短期记忆
      长期记忆
      工作记忆
    多Agent协作
      层级式
      对等式
      竞争式
```

## 17.1 Agent 架构模式概览

Agent 的架构模式决定了它如何思考、决策和行动。不同的架构模式适用于不同类型的任务，理解这些模式是构建高效 Agent 系统的基础。本章将深入解析六种核心架构模式，并为每种模式提供完整的 Python 代码实现。

```
┌─────────────────────────────────────────────────────┐
│                Agent 架构模式谱系                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ReAct ──── 推理与行动交替，最基础的 Agent 模式        │
│    │                                                │
│  Plan-and-Execute ── 先规划后执行，适合复杂任务        │
│    │                                                │
│  Reflexion ── 加入自我反思，从失败中学习               │
│    │                                                │
│  Tool Use ── 专注于工具选择与编排                      │
│    │                                                │
│  RAG-Agent ── 检索增强的 Agent                        │
│    │                                                │
│  Multi-Agent ── 多个 Agent 协作                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 17.2 ReAct 模式：推理与行动的交替

ReAct（Reasoning + Acting）是最经典的 Agent 架构模式，由 Yao et al. (2022) 提出。其核心思想是让 LLM 在推理（Thought）和行动（Action）之间交替进行，每次行动后观察结果（Observation），再进行下一轮推理。

```python
import json
from openai import OpenAI
from typing import Dict, Any, Callable, Optional

class ReActAgent:
    """ReAct 模式 Agent 实现"""

    SYSTEM_PROMPT = """你是一个遵循 ReAct 模式的 AI Agent。
对于每个问题，你需要交替进行思考和行动。

可用工具:
{tools_description}

请严格按照以下 JSON 格式输出:
{{
    "thought": "你的思考过程",
    "action": "工具名称（如果需要调用工具）或 'finish'（如果任务完成）",
    "action_input": "工具的输入参数或最终答案"
}}
"""

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.tools: Dict[str, Callable] = {}
        self.tools_description: Dict[str, str] = {}

    def register_tool(self, name: str, func: Callable, description: str):
        """注册工具"""
        self.tools[name] = func
        self.tools_description[name] = description

    def run(self, query: str, max_steps: int = 8) -> str:
        """执行 ReAct 循环"""
        tools_desc = "\n".join(
            f"- {name}: {desc}" for name, desc in self.tools_description.items()
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(
                tools_description=tools_desc
            )},
            {"role": "user", "content": query}
        ]

        for step in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            result = json.loads(response.choices[0].message.content)
            thought = result.get("thought", "")
            action = result.get("action", "")
            action_input = result.get("action_input", "")

            print(f"\n--- Step {step + 1} ---")
            print(f"Thought: {thought}")
            print(f"Action: {action}")
            print(f"Action Input: {action_input}")

            if action == "finish":
                return action_input

            # 执行工具调用
            if action in self.tools:
                try:
                    observation = self.tools[action](action_input)
                except Exception as e:
                    observation = f"工具执行错误: {str(e)}"
            else:
                observation = f"未知工具: {action}"

            print(f"Observation: {observation}")

            # 将结果加入对话历史
            messages.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "达到最大步骤数，任务未完成"


# 使用示例
def search_web(query: str) -> str:
    """模拟网络搜索"""
    mock_results = {
        "Python 3.12 新特性": "Python 3.12 引入了改进的错误消息、f-string 改进、类型参数语法等",
        "LangChain 最新版本": "LangChain 0.3.x 已发布，重构了核心架构",
    }
    for key, value in mock_results.items():
        if key in query:
            return value
    return f"搜索 '{query}' 的结果：未找到相关信息"

def calculator(expression: str) -> str:
    """安全计算器"""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            return str(eval(expression))
        return "不安全的表达式"
    except Exception as e:
        return f"计算错误: {e}"

# agent = ReActAgent()
# agent.register_tool("search", search_web, "搜索网络信息")
# agent.register_tool("calculator", calculator, "数学计算")
# result = agent.run("Python 3.12 有哪些新特性？请列出至少3个。")
```

## 17.3 Plan-and-Execute 模式：先规划后执行

Plan-and-Execute 模式将任务处理分为两个明确的阶段：首先由 Planner 制定完整的执行计划，然后由 Executor 逐步执行。这种模式特别适合需要多步骤协调的复杂任务。

```python
from dataclasses import dataclass, field
from typing import List
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PlanStep:
    """计划中的单个步骤"""
    step_id: int
    description: str
    tool: str
    tool_input: str
    dependencies: List[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None

@dataclass
class ExecutionPlan:
    """执行计划"""
    goal: str
    steps: List[PlanStep] = field(default_factory=list)

class PlanAndExecuteAgent:
    """Plan-and-Execute 模式 Agent"""

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.tools: Dict[str, Callable] = {}

    def plan(self, goal: str) -> ExecutionPlan:
        """第一阶段：制定计划"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """你是一个任务规划专家。
请将用户的目标分解为具体的执行步骤。
输出 JSON 格式:
{
    "steps": [
        {
            "step_id": 1,
            "description": "步骤描述",
            "tool": "工具名称",
            "tool_input": "工具输入",
            "dependencies": []
        }
    ]
}"""},
                {"role": "user", "content": f"目标: {goal}\n可用工具: {list(self.tools.keys())}"}
            ],
            response_format={"type": "json_object"}
        )
        plan_data = json.loads(response.choices[0].message.content)
        steps = [PlanStep(**s) for s in plan_data["steps"]]
        return ExecutionPlan(goal=goal, steps=steps)

    def execute(self, plan: ExecutionPlan) -> str:
        """第二阶段：执行计划"""
        results = {}
        for step in plan.steps:
            # 检查依赖是否满足
            deps_met = all(
                plan.steps[d - 1].status == StepStatus.COMPLETED
                for d in step.dependencies
            )
            if not deps_met:
                step.status = StepStatus.SKIPPED
                continue

            step.status = StepStatus.RUNNING
            print(f"执行步骤 {step.step_id}: {step.description}")

            try:
                if step.tool in self.tools:
                    # 替换输入中的变量引用
                    tool_input = step.tool_input
                    for sid, res in results.items():
                        tool_input = tool_input.replace(f"{{step_{sid}_result}}", str(res))
                    step.result = self.tools[step.tool](tool_input)
                    step.status = StepStatus.COMPLETED
                    results[step.step_id] = step.result
                else:
                    step.status = StepStatus.FAILED
                    step.result = f"工具 {step.tool} 不存在"
            except Exception as e:
                step.status = StepStatus.FAILED
                step.result = str(e)

        return self._synthesize_results(plan, results)

    def _synthesize_results(self, plan: ExecutionPlan, results: dict) -> str:
        """综合所有步骤结果生成最终答案"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "根据执行结果生成最终答案。"},
                {"role": "user", "content": f"目标: {plan.goal}\n执行结果: {json.dumps(results, ensure_ascii=False)}"}
            ]
        )
        return response.choices[0].message.content

    def run(self, goal: str) -> str:
        """完整的 Plan-and-Execute 流程"""
        plan = self.plan(goal)
        return self.execute(plan)
```

## 17.4 Reflexion 模式：自我反思与改进

Reflexion 模式（Shinn et al., 2023）在 ReAct 基础上增加了自我反思机制。Agent 在执行失败或结果不理想时，会分析失败原因并生成改进策略，然后在下一次尝试中应用这些经验。

```python
@dataclass
class ReflexionMemory:
    """反思记忆"""
    attempt: int
    task: str
    trajectory: List[Dict[str, Any]]
    outcome: str
    success: bool
    reflection: str  # 反思总结

class ReflexionAgent:
    """Reflexion 模式 Agent"""

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.tools: Dict[str, Callable] = {}
        self.reflections: List[ReflexionMemory] = []

    def reflect(self, task: str, trajectory: list, outcome: str) -> str:
        """对执行过程进行反思"""
        past_reflections = "\n".join(
            f"尝试 {r.attempt}: {r.reflection}" for r in self.reflections[-3:]
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """你是一个善于反思的 AI。
分析之前的执行过程，找出失败原因，并提出具体的改进策略。
重点关注：
1. 哪些步骤是正确的？
2. 哪些步骤导致了错误？
3. 下次应该如何改进？"""},
                {"role": "user", "content": f"""
任务: {task}
执行轨迹: {json.dumps(trajectory, ensure_ascii=False)}
结果: {outcome}
之前的反思: {past_reflections if past_reflections else '无'}
"""}
            ]
        )
        return response.choices[0].message.content

    def evaluate(self, task: str, result: str) -> bool:
        """评估结果是否满足任务要求"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "评估结果是否正确完成了任务。只回答 true 或 false。"},
                {"role": "user", "content": f"任务: {task}\n结果: {result}"}
            ]
        )
        return "true" in response.choices[0].message.content.lower()

    def run(self, task: str, max_attempts: int = 3) -> str:
        """带反思的执行循环"""
        for attempt in range(max_attempts):
            # 构建包含历史反思的提示
            reflections_context = ""
            if self.reflections:
                reflections_context = "从之前的尝试中学到的经验:\n" + "\n".join(
                    f"- {r.reflection}" for r in self.reflections[-3:]
                )

            # 执行任务（使用 ReAct 内循环）
            trajectory = []
            result = self._execute_with_react(task, reflections_context, trajectory)

            # 评估结果
            success = self.evaluate(task, result)

            if success:
                print(f"✅ 第 {attempt + 1} 次尝试成功!")
                return result

            # 反思失败原因
            reflection = self.reflect(task, trajectory, result)
            self.reflections.append(ReflexionMemory(
                attempt=attempt + 1,
                task=task,
                trajectory=trajectory,
                outcome=result,
                success=False,
                reflection=reflection
            ))
            print(f"❌ 第 {attempt + 1} 次尝试失败，反思: {reflection[:100]}...")

        return f"经过 {max_attempts} 次尝试仍未成功"

    def _execute_with_react(self, task, context, trajectory) -> str:
        """内部 ReAct 执行循环"""
        # 简化实现，实际中复用 ReActAgent 逻辑
        messages = [
            {"role": "system", "content": f"完成以下任务。{context}"},
            {"role": "user", "content": task}
        ]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        result = response.choices[0].message.content
        trajectory.append({"action": "generate", "result": result})
        return result
```

## 17.5 Tool Use 模式：工具选择与编排

Tool Use 模式专注于如何让 Agent 高效地选择和使用工具。OpenAI 的 Function Calling 是这一模式的典型实现：

```python
class ToolUseAgent:
    """Tool Use 模式 Agent - 基于 OpenAI Function Calling"""

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.tool_functions: Dict[str, Callable] = {}
        self.tool_schemas: List[Dict] = []

    def register_tool(self, name: str, func: Callable, schema: dict):
        """注册工具及其 JSON Schema"""
        self.tool_functions[name] = func
        self.tool_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                **schema
            }
        })

    def run(self, query: str) -> str:
        """执行工具调用循环"""
        messages = [{"role": "user", "content": query}]

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schemas if self.tool_schemas else None,
            )
            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    print(f"调用工具: {func_name}({func_args})")

                    if func_name in self.tool_functions:
                        result = self.tool_functions[func_name](**func_args)
                    else:
                        result = f"未知工具: {func_name}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            else:
                return msg.content
```

## 17.6 RAG 深度解析：从原理到 LlamaIndex 实战

RAG（Retrieval-Augmented Generation，检索增强生成）是 Agent 系统中最重要的架构模式之一。它解决了 LLM 的核心痛点：**知识截止日期**和**幻觉问题**。通过在生成前检索相关文档，RAG 让 Agent 能够基于真实数据回答问题。

### 17.6.1 为什么需要 RAG

LLM 有三个根本性限制：

1. **知识截止**：训练数据有截止日期，无法回答最新问题
2. **幻觉问题**：对不确定的问题会"编造"看似合理的答案
3. **私有数据**：无法访问企业内部文档、代码库、数据库

RAG 的核心思想很简单：**先搜索，再回答**。

```
传统 LLM 流程：
  用户提问 → LLM 生成答案（可能幻觉）

RAG 流程：
  用户提问 → 检索相关文档 → 将文档作为上下文 → LLM 基于文档生成答案
```

### 17.6.2 RAG 的完整架构

一个生产级 RAG 系统包含两个阶段：

**离线阶段（Indexing）**：

```
原始文档 → 文档加载 → 文本分块 → 向量化（Embedding）→ 存入向量数据库
   │          │          │              │                    │
   PDF      Loader    Chunking     Embedding Model      Vector Store
   Word                Strategy     (text-embedding-3)   (Chroma/Pinecone)
   HTML
   Code
   Markdown
```

**在线阶段（Querying）**：

```
用户查询 → 查询改写 → 向量化 → 相似度检索 → 重排序 → 构建 Prompt → LLM 生成
   │          │         │          │           │          │            │
 原始问题  Query     Embedding  Top-K       Reranker   Context     Response
          Rewrite              Retrieval              Assembly
```

### 17.6.3 LlamaIndex 框架介绍

LlamaIndex（原 GPT Index）是最流行的 RAG 框架，专注于将私有数据与 LLM 连接。它提供了从数据加载到查询的完整工具链。

**为什么选择 LlamaIndex**：
- **数据连接器丰富**：支持 PDF、Word、HTML、数据库、API 等 160+ 数据源
- **索引策略多样**：向量索引、关键词索引、知识图谱索引、树索引
- **查询引擎强大**：支持子问题分解、多步推理、SQL 查询
- **与 Agent 深度集成**：可作为 Agent 的工具使用
- **生产就绪**：支持流式输出、异步、可观测性

安装：

```bash
pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
pip install llama-index-vector-stores-chroma
pip install llama-index-readers-file
```

### 17.6.4 实战一：基础 RAG — 30 行代码构建文档问答

```python
"""
最简 RAG 示例：加载文档 → 建索引 → 查询
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. 配置 LLM 和 Embedding 模型
Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 2. 加载文档（支持 PDF、TXT、MD、DOCX 等）
documents = SimpleDirectoryReader("./docs").load_data()
print(f"加载了 {len(documents)} 个文档")

# 3. 构建向量索引（自动分块 + 向量化 + 存储）
index = VectorStoreIndex.from_documents(documents)

# 4. 创建查询引擎
query_engine = index.as_query_engine(similarity_top_k=5)

# 5. 查询
response = query_engine.query("项目的技术架构是什么？")
print(response)

# 查看检索到的源文档
for node in response.source_nodes:
    print(f"  来源: {node.metadata.get('file_name', '未知')}")
    print(f"  相关度: {node.score:.4f}")
    print(f"  内容片段: {node.text[:100]}...")
```

这 30 行代码背后，LlamaIndex 自动完成了：
- 文档解析（识别文件格式，提取文本）
- 文本分块（默认 1024 token，重叠 200 token）
- 向量化（调用 Embedding API）
- 存储（默认内存存储）
- 检索（余弦相似度搜索）
- Prompt 构建（将检索结果注入 Prompt）
- LLM 调用（生成最终答案）

### 17.6.5 实战二：生产级 RAG — 精细控制每个环节

```python
"""
生产级 RAG：精细控制分块、索引、检索、生成的每个环节
"""
from llama_index.core import (
    VectorStoreIndex, StorageContext, Settings, PromptTemplate,
)
from llama_index.core.node_parser import (
    SentenceSplitter, SemanticSplitterNodeParser,
)
from llama_index.core.extractors import (
    TitleExtractor, SummaryExtractor, QuestionsAnsweredExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.postprocessor import (
    SimilarityPostprocessor, KeywordNodePostprocessor,
)
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# ── 第一步：配置 ──────────────────────────────
Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    dimensions=512,  # 降维以节省存储和提高速度
)

# ── 第二步：文档加载与预处理 ──────────────────
from llama_index.readers.file import PDFReader, DocxReader, MarkdownReader

documents = []
documents.extend(PDFReader().load_data("./docs/architecture.pdf"))
documents.extend(MarkdownReader().load_data("./docs/api-guide.md"))

# 为文档添加元数据
for doc in documents:
    doc.metadata["project"] = "my-project"
    doc.metadata["indexed_at"] = "2026-03-04"

# ── 第三步：文本分块策略（关键！）────────────
# 策略 A：固定大小分块（简单高效）
sentence_splitter = SentenceSplitter(
    chunk_size=512,       # 每块最大 512 token
    chunk_overlap=50,     # 块间重叠 50 token
    paragraph_separator="\n\n",
)

# 策略 B：语义分块（按语义边界切分，效果更好但更慢）
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=Settings.embed_model,
)

nodes = sentence_splitter.get_nodes_from_documents(documents)
print(f"分块后得到 {len(nodes)} 个节点")

# ── 第四步：元数据增强（提高检索质量的关键）──
pipeline = IngestionPipeline(
    transformations=[
        sentence_splitter,
        TitleExtractor(nodes=3),
        SummaryExtractor(summaries=["self"]),
        QuestionsAnsweredExtractor(questions=3),
        Settings.embed_model,
    ]
)
enriched_nodes = pipeline.run(documents=documents)

# ── 第五步：持久化向量存储（Chroma）──────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(enriched_nodes, storage_context=storage_context)

# ── 第六步：高级检索 + 后处理 ────────────────
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=0.7),
    KeywordNodePostprocessor(
        required_keywords=["架构"],
        exclude_keywords=["废弃"],
    ),
]

# ── 第七步：自定义 Prompt ────────────────────
qa_prompt = PromptTemplate(
    """你是一个专业的技术文档助手。请基于以下参考资料回答用户的问题。

要求：
1. 只基于提供的参考资料回答，不要编造信息
2. 如果参考资料不足以回答问题，请明确说明
3. 引用具体的来源文档
4. 使用清晰的结构化格式

参考资料：
{context_str}

用户问题：{query_str}

回答："""
)

response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    text_qa_template=qa_prompt,
)

# ── 第八步：组装查询引擎 ────────────────────
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=postprocessors,
)

response = query_engine.query("系统的微服务架构是如何设计的？")
print("回答:", response)
for i, node in enumerate(response.source_nodes):
    print(f"  [{i+1}] {node.metadata.get('file_name', '未知')} "
          f"(相关度: {node.score:.3f})")
```

### 17.6.6 分块策略深度对比

分块（Chunking）是 RAG 中**最影响效果的环节**。错误的分块策略会导致检索到不相关或不完整的内容。

```{list-table} 分块策略对比
:header-rows: 1
:widths: 20 25 25 30

* - 策略
  - 原理
  - 优势
  - 适用场景
* - 固定大小
  - 按 token 数切分
  - 简单快速，可预测
  - 通用场景，快速原型
* - 句子分割
  - 按句子边界切分
  - 保持语义完整性
  - 自然语言文档
* - 语义分块
  - 按语义相似度切分
  - 语义边界最准确
  - 高质量要求场景
* - 递归分割
  - 按层级分隔符递归切分
  - 保持文档结构
  - 结构化文档（Markdown/代码）
* - 文档特定
  - 按文档结构切分（标题/章节）
  - 保持逻辑完整性
  - 技术文档、论文
```

```python
from llama_index.core.node_parser import (
    SentenceSplitter, SemanticSplitterNodeParser,
    MarkdownNodeParser, CodeSplitter,
)

# 代码文件专用分块器
code_splitter = CodeSplitter(
    language="python", chunk_lines=40, chunk_lines_overlap=5, max_chars=1500,
)

# Markdown 文件专用分块器（按标题层级切分）
md_splitter = MarkdownNodeParser()

# 根据文件类型自动选择分块器
def get_splitter_for_file(file_path: str):
    if file_path.endswith('.py'):
        return CodeSplitter(language="python", chunk_lines=40)
    elif file_path.endswith('.md'):
        return MarkdownNodeParser()
    elif file_path.endswith('.pdf'):
        return SentenceSplitter(chunk_size=512, chunk_overlap=50)
    else:
        return SentenceSplitter(chunk_size=256, chunk_overlap=30)
```

### 17.6.7 高级 RAG 技术

基础 RAG 的检索质量往往不够理想。以下是提升效果的关键技术：

**1. 查询改写（HyDE）**

```python
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

# HyDE：先让 LLM 生成"假设性答案"，用答案去检索
# 效果比直接用问题检索好 20-30%
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

response = hyde_query_engine.query("如何优化数据库查询性能？")
```

**2. 子问题分解**

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

tools = [
    QueryEngineTool(
        query_engine=architecture_engine,
        metadata=ToolMetadata(
            name="architecture_docs",
            description="系统架构设计文档"
        ),
    ),
    QueryEngineTool(
        query_engine=api_engine,
        metadata=ToolMetadata(name="api_docs", description="API 接口文档"),
    ),
]

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
)
# 复杂问题自动分解为子问题，分别检索后综合回答
response = sub_question_engine.query(
    "系统的用户认证是如何设计的？包括架构层面和 API 层面"
)
```

**3. 混合检索（Hybrid Search）**

```python
from llama_index.core.retrievers import QueryFusionRetriever

# 融合向量检索（语义匹配）和关键词检索（精确匹配）
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    similarity_top_k=5,
    num_queries=4,
    mode="reciprocal_rerank",
)
```

**4. 重排序（Reranking）**

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(api_key="your-key", top_n=5)
query_engine = index.as_query_engine(
    similarity_top_k=20,           # 先粗检索 20 个
    node_postprocessors=[reranker], # 再精排到 5 个
)
```

### 17.6.8 RAG 与 Agent 的结合

RAG 可以作为 Agent 的工具，让 Agent 在需要时主动检索知识：

```python
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent

tools = [
    QueryEngineTool.from_defaults(
        query_engine=code_index.as_query_engine(similarity_top_k=5),
        name="search_codebase",
        description="搜索项目代码库，了解代码实现细节",
    ),
    QueryEngineTool.from_defaults(
        query_engine=docs_index.as_query_engine(similarity_top_k=5),
        name="search_docs",
        description="搜索项目文档，了解架构设计和业务逻辑",
    ),
]

agent = ReActAgent.from_tools(
    tools=tools, llm=OpenAI(model="gpt-4o"),
    verbose=True, max_iterations=10,
)

# Agent 自动决定何时使用哪个知识库
response = agent.chat("用户登录接口的实现逻辑是什么？")
```

### 17.6.9 RAG 评估

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator, RelevancyEvaluator,
)

faithfulness_eval = FaithfulnessEvaluator(llm=OpenAI(model="gpt-4o"))
relevancy_eval = RelevancyEvaluator(llm=OpenAI(model="gpt-4o"))

query = "系统支持哪些认证方式？"
response = query_engine.query(query)

# 忠实度：答案是否基于检索到的文档（防幻觉）
faith = faithfulness_eval.evaluate_response(query=query, response=response)
print(f"忠实度: {'通过' if faith.passing else '不通过'}")

# 批量评估
eval_questions = [
    "系统的技术栈是什么？",
    "如何部署到生产环境？",
    "API 的认证方式有哪些？",
]
results = []
for q in eval_questions:
    resp = query_engine.query(q)
    faith = faithfulness_eval.evaluate_response(query=q, response=resp)
    results.append(faith.passing)

print(f"整体忠实度: {sum(results)/len(results):.1%}")
```

### 17.6.10 RAG 常见问题与优化清单

```{list-table} RAG 问题诊断与优化
:header-rows: 1
:widths: 25 35 40

* - 问题
  - 原因
  - 解决方案
* - 检索不到相关文档
  - 分块太大/太小；Embedding 不匹配
  - 调整 chunk_size；换 Embedding 模型；用 HyDE
* - 检索到但答案不对
  - Prompt 模板不好；LLM 忽略上下文
  - 优化 Prompt；加"只基于文档回答"指令
* - 答案包含幻觉
  - 检索结果不够相关
  - 添加 Reranker；降低 temperature
* - 跨文档问题回答差
  - 单次检索无法覆盖多文档
  - 用 SubQuestionQueryEngine 分解问题
* - 速度太慢
  - Embedding 调用多；检索范围大
  - 降维；缓存；减少 top_k
* - 成本太高
  - 每次查询都调 LLM
  - 缓存常见查询；小模型初筛
```

```{admonition} RAG 优化黄金法则
:class: tip
1. **分块是基础**：花 80% 的优化时间在分块策略上
2. **检索比生成重要**：检索不到正确文档，再好的 LLM 也无法回答
3. **评估驱动优化**：先建立评估基准，再做优化，用数据说话
4. **混合检索**：向量 + 关键词几乎总是比单一方式好
5. **Reranker 是性价比最高的优化**：通常能提升 10-20% 准确率
```

---

## 17.7 记忆系统设计

Agent 的记忆系统是其保持上下文连贯性和从经验中学习的关键：

```python
from datetime import datetime
import hashlib

class MemoryType(Enum):
    SHORT_TERM = "short_term"     # 短期记忆：当前对话上下文
    LONG_TERM = "long_term"       # 长期记忆：持久化的知识和经验
    WORKING = "working"           # 工作记忆：当前任务的中间状态
    EPISODIC = "episodic"         # 情景记忆：具体事件的记录
    SEMANTIC = "semantic"         # 语义记忆：抽象知识和概念

@dataclass
class MemoryEntry:
    content: str
    memory_type: MemoryType
    timestamp: datetime
    importance: float  # 0.0 - 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentMemorySystem:
    """Agent 记忆系统"""

    def __init__(self, vector_store, embedding_model, max_short_term: int = 20):
        self.short_term: List[MemoryEntry] = []
        self.working: Dict[str, Any] = {}
        self.vector_store = vector_store  # 用于长期记忆
        self.embedding_model = embedding_model
        self.max_short_term = max_short_term

    def store(self, content: str, memory_type: MemoryType, importance: float = 0.5, **metadata):
        """存储记忆"""
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata
        )

        if memory_type == MemoryType.SHORT_TERM:
            self.short_term.append(entry)
            if len(self.short_term) > self.max_short_term:
                self._consolidate_short_term()
        elif memory_type in (MemoryType.LONG_TERM, MemoryType.EPISODIC, MemoryType.SEMANTIC):
            entry.embedding = self.embedding_model.encode(content)
            self.vector_store.add(entry)

    def recall(self, query: str, memory_type: Optional[MemoryType] = None, top_k: int = 5) -> List[MemoryEntry]:
        """检索记忆"""
        results = []

        # 搜索短期记忆
        if memory_type in (None, MemoryType.SHORT_TERM):
            for entry in reversed(self.short_term):
                if query.lower() in entry.content.lower():
                    results.append(entry)

        # 搜索长期记忆（向量检索）
        if memory_type in (None, MemoryType.LONG_TERM, MemoryType.EPISODIC):
            query_embedding = self.embedding_model.encode(query)
            long_term_results = self.vector_store.similarity_search(
                query_embedding, top_k=top_k
            )
            results.extend(long_term_results)

        # 按重要性和时间排序
        results.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        return results[:top_k]

    def _consolidate_short_term(self):
        """将短期记忆整合到长期记忆"""
        # 保留最近和最重要的记忆
        self.short_term.sort(key=lambda x: x.importance, reverse=True)
        to_consolidate = self.short_term[self.max_short_term // 2:]
        self.short_term = self.short_term[:self.max_short_term // 2]

        for entry in to_consolidate:
            if entry.importance > 0.3:
                entry.memory_type = MemoryType.LONG_TERM
                entry.embedding = self.embedding_model.encode(entry.content)
                self.vector_store.add(entry)

    def forget(self, threshold: float = 0.1):
        """遗忘不重要的记忆"""
        self.short_term = [
            m for m in self.short_term if m.importance > threshold
        ]
```

## 17.8 多 Agent 协作架构

多 Agent 系统有三种主要的协作架构：

### 层级式（Hierarchical）
一个 Supervisor Agent 负责任务分配和结果整合，Worker Agent 负责具体执行。

### 对等式（Peer-to-Peer）
所有 Agent 地位平等，通过消息传递协作，适合需要讨论和辩论的场景。

### 竞争式（Competitive）
多个 Agent 独立完成同一任务，选择最优结果，适合需要多样性的创意任务。

```python
class CollaborationPattern(Enum):
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    COMPETITIVE = "competitive"

class SupervisorAgent:
    """层级式协作中的 Supervisor"""

    def __init__(self, workers: Dict[str, Any], llm_client):
        self.workers = workers
        self.llm = llm_client

    def delegate(self, task: str) -> str:
        """分析任务并委派给合适的 Worker"""
        # 1. 分析任务需要哪些能力
        analysis = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"分析任务并选择合适的执行者。\n任务: {task}\n可用执行者: {list(self.workers.keys())}"
            }],
            response_format={"type": "json_object"}
        )
        assignment = json.loads(analysis.choices[0].message.content)

        # 2. 分发子任务
        results = {}
        for subtask in assignment.get("subtasks", []):
            worker_name = subtask["worker"]
            if worker_name in self.workers:
                results[worker_name] = self.workers[worker_name].run(subtask["task"])

        # 3. 整合结果
        return self._synthesize(task, results)

    def _synthesize(self, original_task: str, results: dict) -> str:
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"原始任务: {original_task}\n各执行者结果: {json.dumps(results, ensure_ascii=False)}\n请整合为最终答案。"
            }]
        )
        return response.choices[0].message.content
```

## 17.9 架构模式选型指南

| 模式 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| ReAct | 通用问答、简单任务 | 实现简单、灵活 | 长任务易迷失方向 |
| Plan-and-Execute | 复杂多步骤任务 | 结构清晰、可追踪 | 规划阶段耗时、难以应对意外 |
| Reflexion | 需要高准确率的任务 | 能从失败中学习 | 多次尝试增加成本 |
| Tool Use | 工具密集型任务 | 工具调用高效 | 依赖工具质量 |
| RAG-Agent | 知识密集型任务 | 减少幻觉 | 检索质量影响大 |
| Multi-Agent | 大型复杂项目 | 专业化分工 | 协调开销大 |

## 17.10 本章小结

本章深入解析了 Agent 的六种核心架构模式。每种模式都有其适用场景和权衡：

1. **ReAct** 是最基础的模式，适合快速原型开发
2. **Plan-and-Execute** 适合需要系统性规划的复杂任务
3. **Reflexion** 通过自我反思提高任务成功率
4. **Tool Use** 专注于高效的工具选择和编排
5. **RAG-Agent** 将检索增强与 Agent 能力结合
6. **记忆系统**是所有模式的基础设施，决定了 Agent 的长期表现

在实际项目中，这些模式往往需要组合使用。下一章我们将介绍主流的 Agent 开发框架，它们封装了这些模式的实现细节，让开发者能够更高效地构建 Agent 系统。
