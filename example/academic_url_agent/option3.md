
# 从零构建 AI Agent：用 LangGraph 实现 ReAct 决策图

> **TL;DR** — 本文用 Python + LangGraph，把决策层从手写 while 循环升级为**状态图（StateGraph）**。你会看到 ReAct 的 Reason → Act → Observe 循环如何被建模为图中的**节点 + 条件边 + 循环边**，更加直观、可控、可扩展。

---

## 一、为什么用 LangGraph 替代手写循环？

LangGraph 是一个基于图的框架，专为构建复杂的 LLM 应用和有状态工作流而设计，它让构建复杂的 Agent 架构变得更加容易。 ([Building ReAct agents with (and without) LangGraph – Dylan Castillo](https://dylancastillo.co/posts/react-agent-langgraph.html))

图（Graph）由节点（Nodes）、边（Edges）、状态（State）和 Reducer 组成。节点是工作单元（函数、工具），边定义节点之间的路径。状态是在节点之间传递并通过 Reducer 更新的持久化数据。 ([Building ReAct agents with (and without) LangGraph – Dylan Castillo](https://dylancastillo.co/posts/react-agent-langgraph.html))

| | 手写 while 循环 | LangGraph StateGraph |
|---|---|---|
| 控制流 | 隐藏在 while + if 里 | **图结构可视化** |
| 状态管理 | 自己维护 messages 列表 | **内置 State + Reducer** |
| 可观测性 | print 大法 | **集成 LangSmith 追踪** |
| 可扩展性 | 改代码加逻辑 | **加节点加边即可** |
| 持久化 | 自己写 | **内置 Checkpointer** |

---

## 二、架构回顾：两层分离

```
┌─────────────────────────────────────────────────────────────┐
│                        用户输入 URL                          │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  决策层 (Decision Layer) ── LangGraph ReAct 图               │
│                                                              │
│   ┌─────────┐   有 tool_calls   ┌────────────┐              │
│   │  agent  │ ───────────────▶ │   tools    │              │
│   │ (LLM)   │ ◀─────────────── │ (执行工具)  │              │
│   └────┬────┘   tool 结果返回    └────────────┘              │
│        │ 无 tool_calls                                       │
│        ▼                                                     │
│      END → 输出抓取到的正文                                    │
└─────────────────────┬───────────────────────────────────────┘
                      ▼  英文正文
┌─────────────────────────────────────────────────────────────┐
│  生成层 (Generation Layer) ── LLM Chains 管线                │
│                                                              │
│  ① 分块翻译  ──▶  ② 质量自检纠错  ──▶  ③ 要点总结            │
│       │                                       │              │
│       ▼                                       ▼              │
│  ④ 难点解释                            ⑤ PlantUML 思维导图    │
│                                              │               │
│                                        ⑥ 调 PlantUML → PNG   │
└─────────────────────────────────────────────────────────────┘
```

**决策层**：用 LangGraph StateGraph 建模 ReAct 循环——Agent 不确定哪种抓取方法有效，需自主试错。
**生成层**：确定性管线，拿到正文后按固定步骤处理，不需要 Agent 决策。

---

## 三、LangGraph 如何表达 ReAct？

在基本的 ReAct Agent 中只有两个节点，一个用于调用模型，一个用于使用工具——但你可以修改这个基本结构以更好地适配你的场景。 ([How to create a ReAct agent from scratch](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/))

```
                     ┌─────────────────────────────┐
                     │       LangGraph 状态图        │
                     │                              │
  START ──▶  ┌──────────────┐                       │
             │  agent 节点   │  🧠 Reason            │
             │  (调用 LLM)   │  LLM 看消息历史，       │
             └──────┬───────┘  决定是否调用工具        │
                    │                               │
              ┌─────┴──────┐                        │
              ▼            ▼                        │
        有 tool_calls   无 tool_calls               │
              │            │                        │
              ▼            ▼                        │
        ┌──────────┐    END                         │
        │ tools 节点│  👁 Observe                    │
        │ (执行工具) │  🔧 Act                       │
        └─────┬────┘                                │
              │  结果写回 messages                    │
              └──────────▶ agent 节点 (循环)          │
                                                    │
                     └─────────────────────────────┘
```

有一条条件边连接 assistant 节点到 tools 节点和 END 节点。条件边基于 assistant 是否想调用工具来路由请求。tools 节点的输出连接回 assistant 节点，形成循环。 ([Building a ReAct Agent with Langgraph: A Step-by-Step Guide | by Umang | Medium](https://medium.com/@umang91999/building-a-react-agent-with-langgraph-a-step-by-step-guide-812d02bafefa))

这就是 ReAct 的全部：**agent → (条件判断) → tools → agent → ... → END**。

---

## 四、环境搭建

### 4.1 用 Poetry 初始化

```bash
mkdir url-translator && cd url-translator
poetry init --name url-translator --python "^3.11" -n
```

### 4.2 安装依赖

```bash
poetry add langchain-core langchain-openai langgraph \
           requests beautifulsoup4 readability-lxml \
           playwright python-dotenv

# Playwright 需额外安装浏览器
poetry run playwright install chromium
```

完整 `pyproject.toml`：

```toml
[tool.poetry]
name = "url-translator"
version = "0.2.0"
description = "LangGraph ReAct Agent: fetch, translate, summarize, mindmap"

[tool.poetry.dependencies]
python = "^3.11"
langchain-core = ">=0.3"
langchain-openai = ">=0.3"
langgraph = ">=0.2"
requests = "^2.31"
beautifulsoup4 = "^4.12"
readability-lxml = "^0.8"
playwright = "^1.40"
python-dotenv = "^1.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### 4.3 环境变量

```bash
# .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
```

---

## 五、代码实现

### 项目结构

```
url-translator/
├── pyproject.toml
├── .env
├── tools.py          # 🔧 抓取工具定义（不变）
├── graph.py          # 🆕 LangGraph 决策层（替代手写循环）
├── pipeline.py       # ⚙️  生成层管线（不变）
└── main.py           # 🚀 入口
```

---

### 5.1 `tools.py` —— 抓取工具（与之前相同）

```python
"""tools.py — Agent 可用的抓取工具。

关键点：
  1. @tool 装饰器：函数 → LangChain Tool
  2. docstring = 工具说明书，LLM 靠它决定何时调用
  3. 返回值中的 [WARN]/[ERROR] 前缀 = 给 Agent 的信号
"""

import requests
from bs4 import BeautifulSoup
from readability import Document
from langchain_core.tools import tool


@tool
def fetch_static(url: str) -> str:
    """用 HTTP GET 静态抓取网页正文。适用于博客、文档等服务端渲染页面。
    如果返回内容包含 [WARN]，说明抓取质量不佳。"""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TranslatorBot/1.0)"},
            timeout=15,
        )
        resp.raise_for_status()
        doc = Document(resp.text)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        if len(text) < 200:
            return (
                f"[WARN] 正文仅 {len(text)} 字符，可能抓取失败。"
                "建议改用 fetch_dynamic 工具进行 JS 渲染抓取。"
            )
        return text[:15000]
    except Exception as e:
        return f"[ERROR] 静态抓取失败: {e}。建议改用 fetch_dynamic。"


@tool
def fetch_dynamic(url: str) -> str:
    """用 Playwright 启动无头浏览器渲染页面后抓取正文。
    适用于 SPA、JS 渲染等静态抓取无法获取内容的页面。
    注意：速度较慢，仅在 fetch_static 失败时使用。"""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            html = page.content()
            browser.close()

        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        if len(text) < 200:
            return (
                f"[WARN] Playwright 渲染后正文仅 {len(text)} 字符，"
                "页面可能需要登录或有反爬。"
            )
        return text[:15000]
    except Exception as e:
        return f"[ERROR] 动态抓取失败: {e}"


ALL_TOOLS = [fetch_static, fetch_dynamic]
```

---

### 5.2 `graph.py` —— ⭐ LangGraph 决策层（核心改动）

这是**最核心**的新文件。用 `StateGraph` 把手写的 while 循环变成一张**可视化的图**。

```python
"""graph.py — 用 LangGraph StateGraph 实现 ReAct 决策层。

=== LangGraph 版 ReAct 的核心思路 ===

把手写的 while 循环拆解成图的三要素：
  1. 节点 (Nodes): agent（调用 LLM）、tools（执行工具）
  2. 条件边 (Conditional Edge): should_continue 判断走哪条路
  3. 循环边 (Loop Edge): tools → agent，把工具结果反馈给 LLM

                   ┌──────┐
     START ──────▶│ agent │
                   └──┬───┘
                      │
              ┌───────┴────────┐
              ▼                ▼
       has tool_calls     no tool_calls
              │                │
              ▼                ▼
         ┌────────┐          END
         │ tools  │
         └───┬────┘
             │
             └──────▶ agent (循环)
"""

import os
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from tools import ALL_TOOLS

load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: 定义状态 (State)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# State 是图中所有节点共享的"黑板"。
# add_messages 是 LangGraph 提供的 reducer —— 它把新消息追加到列表，
# 而不是覆盖，这样对话历史就能自动累积。


class AgentState(TypedDict):
    """ReAct Agent 的状态。

    - messages: 对话消息列表（含 Human / AI / Tool 消息）
    - iteration: 当前循环次数（安全阀）
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: 初始化 LLM + 绑定工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENT_SYSTEM_PROMPT = """你是一个网页内容抓取专家。你的任务是从给定的 URL 获取高质量的英文正文。

## 策略（请严格按顺序执行）
1. 先用 `fetch_static` 工具尝试静态抓取
2. 如果返回内容包含 [WARN] 或 [ERROR]，改用 `fetch_dynamic` 进行浏览器渲染抓取
3. 如果两者都失败，直接说明原因

## 注意
- 最终必须返回抓取到的纯文本正文
- 不要编造内容
- 最多尝试 3 次工具调用
"""

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

# bind_tools: 让 LLM 知道自己有哪些工具可用
# LLM 返回的 AIMessage 可能包含 tool_calls 字段
llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: 定义节点 (Nodes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def agent_node(state: AgentState) -> dict:
    """🧠 Agent 节点 = ReAct 中的 Reason 步骤。

    读取消息历史 → 调用 LLM → 返回决策结果。
    LLM 可能返回：
      - 带 tool_calls 的 AIMessage → 需要执行工具
      - 纯文本的 AIMessage → 已有结论，可以结束
    """
    messages = state["messages"]

    # 确保系统提示在最前面
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + list(messages)

    # 调用 LLM（这里就是 Reason！）
    response = llm_with_tools.invoke(messages)

    # 更新迭代计数
    iteration = state.get("iteration", 0) + 1
    print(f"  🧠 [agent 节点] 第 {iteration} 轮推理完成")

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"     → 决定调用工具: {tc['name']}({tc['args']})")
    else:
        print(f"     → 得出结论，准备结束")

    return {
        "messages": [response],
        "iteration": iteration,
    }


# ToolNode: LangGraph 预构建的工具执行节点。
# 它自动解析 AIMessage 中的 tool_calls，执行对应工具，
# 然后把结果包装成 ToolMessage 返回。
# 这就是 ReAct 中的 Act + Observe！
tools_node = ToolNode(ALL_TOOLS)


def tool_node_with_logging(state: AgentState) -> dict:
    """🔧 工具节点 + 日志。包装 ToolNode 添加打印。

    这里展示了如何在 LangGraph 的预构建节点外包一层自定义逻辑。
    """
    result = tools_node.invoke(state)

    # 打印 Observation 预览
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            preview = msg.content[:150].replace("\n", " ")
            print(f"  👁️ [tools 节点] Observation: {preview}...")

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: 定义条件边 (Conditional Edge)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_ITERATIONS = 5


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """条件路由函数：决定下一步走哪个节点。

    这个函数就是 ReAct 循环的"岔路口"：
      - 如果 LLM 返回了 tool_calls → 走 tools 节点（继续循环）
      - 如果没有 tool_calls → 走 END（结束循环）
      - 如果超过最大迭代次数 → 强制 END（安全阀）
    """
    messages = state["messages"]
    last_message = messages[-1]
    iteration = state.get("iteration", 0)

    # 安全阀：防止无限循环
    if iteration >= MAX_ITERATIONS:
        print(f"  ⚠️  达到最大迭代次数 {MAX_ITERATIONS}，强制结束")
        return "__end__"

    # 如果最后一条消息有 tool_calls，走 tools 节点
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 否则走 END
    return "__end__"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: 组装图 (Build the Graph)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_fetch_graph():
    """构建 ReAct 抓取图。

    图的结构：
        START → agent → (should_continue?) → tools → agent → ... → END
    """
    # 创建状态图
    workflow = StateGraph(AgentState)

    # ── 添加节点 ──────────────────────────────────────────
    workflow.add_node("agent", agent_node)       # 🧠 推理节点
    workflow.add_node("tools", tool_node_with_logging)  # 🔧 工具节点

    # ── 添加边 ────────────────────────────────────────────
    # 入口边：START → agent
    workflow.add_edge(START, "agent")

    # 条件边：agent 之后根据 should_continue 的返回值路由
    workflow.add_conditional_edges(
        "agent",              # 源节点
        should_continue,      # 路由函数
        {                     # 路由映射
            "tools": "tools",      # 有 tool_calls → 去执行工具
            "__end__": END,        # 无 tool_calls → 结束
        },
    )

    # 循环边：tools → agent（把工具结果反馈给 LLM 继续推理）
    workflow.add_edge("tools", "agent")

    # ── 编译 ──────────────────────────────────────────────
    graph = workflow.compile()

    return graph


# ── 构建全局实例 ─────────────────────────────────────────────

fetch_graph = build_fetch_graph()
```

> **要点解析**：
>
> 图由节点、边、状态和 Reducer 组成。节点是工作单元（函数、工具），边定义节点之间的路径。状态是在节点之间传递并通过 Reducer 更新的持久化数据。 ([Building ReAct agents with (and without) LangGraph – Dylan Castillo](https://dylancastillo.co/posts/react-agent-langgraph.html))
>
> "agent" 节点调用语言模型处理消息列表。如果返回的 AIMessage 包含 tool_calls，图就会调用 "tools" 节点。"tools" 节点执行工具（每个 tool_call 一个工具）并将响应作为 ToolMessage 对象添加到消息列表中。 ([Agents (LangGraph) | LangChain Reference](https://reference.langchain.com/python/langgraph/agents/))

---

### 5.3 `pipeline.py` —— 生成层管线（不变）

```python
"""pipeline.py — 生成层：翻译 → 自检 → 总结 → 难点 → 思维导图。

与上一版完全相同——生成层不需要 Agent 决策，是确定性管线。
"""

import os
import zlib
import requests as http_requests
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.3,
)
parser = StrOutputParser()

# ── Prompt 模板 ─────────────────────────────────────────────

TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是专业英中翻译。保持段落结构，专业术语在括号中保留英文原文。"),
    ("human", "请将以下英文翻译成流畅准确的中文：\n\n{text}"),
])

QUALITY_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是翻译质量审核专家。对比原文和译文，检查：漏译、误译、术语准确性、通顺度。"
     "如有问题就修正后返回完整译文；如无问题就原样返回译文。只返回最终译文，不要解释。"),
    ("human", "原文：\n{original}\n\n译文：\n{translation}"),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位善于提炼要点的阅读助手。"),
    ("human", "请用中文总结以下文章的 5-10 个关键要点，用编号列表：\n\n{text}"),
])

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位耐心的技术导师，擅长用简单的类比解释复杂概念。"),
    ("human",
     "以下是一篇技术文章的翻译。请识别其中 3-5 个最难理解的概念或术语，"
     "用简单易懂的中文逐一解释（可用类比）：\n\n{text}"),
])

MINDMAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是信息架构专家。根据内容生成 PlantUML 思维导图脚本。"
     "格式要求：\n"
     "- 用 @startmindmap 和 @endmindmap 包裹\n"
     "- 用 * 表示层级（* 一级, ** 二级, *** 三级）\n"
     "- 节点文字简洁，中文\n"
     "- 只输出 PlantUML 代码，不要其他文字"),
    ("human", "请根据以下文章要点生成思维导图：\n\n{text}"),
])

# ── LCEL Chains ─────────────────────────────────────────────

translate_chain = TRANSLATE_PROMPT | llm | parser
quality_chain   = QUALITY_CHECK_PROMPT | llm | parser
summary_chain   = SUMMARY_PROMPT | llm | parser
explain_chain   = EXPLAIN_PROMPT | llm | parser
mindmap_chain   = MINDMAP_PROMPT | llm | parser


# ── 工具函数 ────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = p
        else:
            current = current + "\n\n" + p if current else p
    if current.strip():
        chunks.append(current.strip())
    return chunks


def translate_with_quality_check(text: str) -> str:
    chunks = chunk_text(text)
    results = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  📝 翻译第 {i}/{len(chunks)} 块...")
        raw = translate_chain.invoke({"text": chunk})
        checked = quality_chain.invoke({
            "original": chunk,
            "translation": raw,
        })
        results.append(checked)
    return "\n\n".join(results)


# ── PlantUML 编码 & 渲染 ────────────────────────────────────

_PLANTUML_ALPHABET = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
)

def _encode_plantuml(text: str) -> str:
    data = zlib.compress(text.encode("utf-8"))[2:-4]
    res = []
    for i in range(0, len(data), 3):
        b = [data[i]] + [data[i+j] if i+j < len(data) else 0 for j in (1, 2)]
        res.append(_PLANTUML_ALPHABET[b[0] >> 2])
        res.append(_PLANTUML_ALPHABET[((b[0] & 0x3) << 4) | (b[1] >> 4)])
        res.append(_PLANTUML_ALPHABET[((b[1] & 0xF) << 2) | (b[2] >> 6)])
        res.append(_PLANTUML_ALPHABET[b[2] & 0x3F])
    return "".join(res)


def render_plantuml_png(script: str, output_path: str = "mindmap.png") -> str:
    encoded = _encode_plantuml(script)
    url = f"https://www.plantuml.com/plantuml/png/{encoded}"
    try:
        resp = http_requests.get(url, timeout=15)
        if resp.status_code == 200 and resp.headers.get(
            "content-type", ""
        ).startswith("image"):
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return output_path
    except Exception as e:
        print(f"  ⚠️  PNG 渲染失败: {e}")
    return ""


# ── 主管线 ──────────────────────────────────────────────────

def run_pipeline(original_text: str) -> dict:
    print("\n🔄 [生成层] 开始处理...\n")

    print("① 翻译中...")
    translation = translate_with_quality_check(original_text)

    print("② 总结要点...")
    summary = summary_chain.invoke({"text": translation})

    print("③ 解释难点...")
    explanation = explain_chain.invoke({"text": translation})

    print("④ 生成思维导图...")
    mindmap_script = mindmap_chain.invoke({"text": summary})

    if "```" in mindmap_script:
        lines = mindmap_script.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        mindmap_script = "\n".join(lines)

    print("⑤ 渲染思维导图 PNG...")
    png_path = render_plantuml_png(mindmap_script)

    with open("mindmap.puml", "w", encoding="utf-8") as f:
        f.write(mindmap_script)

    return {
        "translation": translation,
        "summary": summary,
        "explanation": explanation,
        "mindmap_script": mindmap_script,
        "mindmap_png": png_path,
    }
```

---

### 5.4 `main.py` —— 入口（调用 LangGraph 图）

```python
"""main.py — 程序入口：调用 LangGraph 图 + 生成层管线。

对比之前的手写 while 循环版本：
  Before: react_fetch() 函数里手写 for step in range(MAX_ITERATIONS)
  After:  graph.invoke() 一行搞定，图自己循环

关键区别：
  - 循环逻辑从"代码控制"变成了"图结构控制"
  - 状态管理从"手动 append"变成了"Reducer 自动处理"
  - 可以轻松添加新节点（如 human-in-the-loop 审批节点）
"""

import sys
from langchain_core.messages import HumanMessage

from graph import fetch_graph       # 导入编译好的 LangGraph 图
from pipeline import run_pipeline


def react_fetch(url: str) -> str:
    """用 LangGraph 图执行 ReAct 抓取。

    只需要:
      1. 构造初始状态
      2. 调用 graph.invoke()
      3. 从最终状态中提取结果

    图内部的 ReAct 循环 (agent → tools → agent → ...) 完全自动。
    """
    print("🤖 [决策层] LangGraph ReAct 图启动\n")

    # 构造初始状态
    initial_state = {
        "messages": [
            HumanMessage(
                content=f"请抓取以下 URL 的英文正文内容：{url}"
            ),
        ],
        "iteration": 0,
    }

    # ⭐ 一行调用 —— 图自动执行 ReAct 循环
    final_state = fetch_graph.invoke(initial_state)

    # 从最终状态提取结果
    last_message = final_state["messages"][-1]
    return last_message.content


def main():
    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else input("请输入英文网页 URL: ").strip()
    )
    if not url:
        print("❌ URL 不能为空")
        return

    # ========== 第一阶段：决策层（LangGraph 图）==========
    content = react_fetch(url)

    if "[ERROR]" in content or len(content) < 100:
        print(f"❌ 抓取失败：{content[:300]}")
        return

    print(f"\n✅ 成功获取正文（{len(content)} 字符）")

    # ========== 第二阶段：生成层（LLM 管线）==========
    result = run_pipeline(content)

    # ========== 输出 ==========
    print("\n" + "=" * 60)
    print("📖 中文翻译")
    print("=" * 60)
    print(result["translation"][:2000] + "\n...")

    print("\n" + "=" * 60)
    print("📌 要点总结")
    print("=" * 60)
    print(result["summary"])

    print("\n" + "=" * 60)
    print("🔍 难点解释")
    print("=" * 60)
    print(result["explanation"])

    print("\n" + "=" * 60)
    print("🗺️  PlantUML 思维导图脚本")
    print("=" * 60)
    print(result["mindmap_script"])

    if result["mindmap_png"]:
        print(f"\n✅ 思维导图已保存为: {result['mindmap_png']}")
    print(f"✅ PlantUML 脚本已保存为: mindmap.puml")


if __name__ == "__main__":
    main()
```

---

## 六、运行

```bash
poetry shell
python main.py "https://lilianweng.github.io/posts/2023-06-23-agent/"
```

### 运行效果

```
🤖 [决策层] LangGraph ReAct 图启动

  🧠 [agent 节点] 第 1 轮推理完成
     → 决定调用工具: fetch_static({"url": "https://lilianweng.github.io/..."})
  👁️ [tools 节点] Observation: LLM Powered Autonomous Agents  June 23, 2023...
  🧠 [agent 节点] 第 2 轮推理完成
     → 得出结论，准备结束

✅ 成功获取正文（12836 字符）

🔄 [生成层] 开始处理...
① 翻译中...
  📝 翻译第 1/5 块...
  📝 翻译第 2/5 块...
  ...
② 总结要点...
③ 解释难点...
④ 生成思维导图...
⑤ 渲染思维导图 PNG...

✅ 思维导图已保存为: mindmap.png
✅ PlantUML 脚本已保存为: mindmap.puml
```

---

## 七、手写循环 vs LangGraph 对比

来看 `react_fetch()` 函数的变化：

### Before：手写 while 循环（~40 行）

```python
def react_fetch(url: str) -> str:
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_map = {t.name: t for t in ALL_TOOLS}
    messages = [SystemMessage(...), HumanMessage(...)]

    for step in range(1, MAX_ITERATIONS + 1):     # 手动循环
        response = llm_with_tools.invoke(messages) # 手动调 LLM
        messages.append(response)                  # 手动追加

        if not response.tool_calls:                # 手动判断
            return response.content

        for tc in response.tool_calls:             # 手动执行工具
            result = tool_map[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(...))       # 手动追加结果

    return messages[-1].content
```

### After：LangGraph 图（~5 行调用）

```python
def react_fetch(url: str) -> str:
    initial_state = {
        "messages": [HumanMessage(content=f"请抓取：{url}")],
        "iteration": 0,
    }
    final_state = fetch_graph.invoke(initial_state)  # ⭐ 一行搞定
    return final_state["messages"][-1].content
```

循环逻辑在哪？在 **`graph.py` 的图结构里**——节点 + 条件边 + 循环边定义了一切。

---

## 八、LangGraph ReAct 图的核心概念

### 8.1 State + Reducer：自动管理对话历史

```python
class AgentState(TypedDict):
    # add_messages 是关键 —— 它是一个 reducer
    # 新消息自动追加，而不是覆盖
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int
```

LangGraph 提供了一个便捷的辅助函数 `add_messages` 用于更新状态中的消息列表。它作为 Reducer 工作，接收当前列表和新消息，然后返回合并后的列表。 ([ReAct agent from scratch with Gemini 2.5 and LangGraph | Gemini API | Google AI for Developers](https://ai.google.dev/gemini-api/docs/langgraph-example))

### 8.2 条件边：图的"岔路口"

```python
workflow.add_conditional_edges(
    "agent",           # 从 agent 节点出发
    should_continue,   # 调用这个函数决定走哪条路
    {
        "tools": "tools",   # 返回 "tools" → 去 tools 节点
        "__end__": END,      # 返回 "__end__" → 结束
    },
)
```

这个工具函数实现了 ReAct 风格 Agent 的标准条件逻辑：如果最后一条 AIMessage 包含 tool calls，就路由到工具执行节点；否则结束工作流。这个模式是大多数工具调用 Agent 架构的基础。 ([Agents (LangGraph) | LangChain Reference](https://reference.langchain.com/python/langgraph/agents/))

### 8.3 循环边：形成 ReAct 反馈循环

```python
workflow.add_edge("tools", "agent")  # tools 执行完 → 回到 agent 继续推理
```

条件边基于 assistant 是否想调用工具来路由请求到 tools 节点或 END 节点。tools 节点的输出连接回 assistant 节点，形成循环。 ([Building a ReAct Agent with Langgraph: A Step-by-Step Guide | by Umang | Medium](https://medium.com/@umang91999/building-a-react-agent-with-langgraph-a-step-by-step-guide-812d02bafefa))

### 8.4 ToolNode：预构建的工具执行器

```python
tools_node = ToolNode(ALL_TOOLS)
```

ToolNode 是 LangGraph 预构建的类，用于包装外部工具或函数。当检测到工具调用时触发，使 Agent 能够执行适当的函数并将结果返回到工作流中。 ([Getting Started with LangGraph: A Beginner’s Guide to Building Intelligent Workflows](https://medium.com/@ashutoshsharmaengg/getting-started-with-langgraph-a-beginners-guide-to-building-intelligent-workflows-67eeee0899d0))

### 8.5 完整映射：ReAct 三步 → 图节点/边

```
┌──────────────────────────────────────────────────────────┐
│  ReAct 步骤          LangGraph 对应组件                    │
│──────────────────────────────────────────────────────────│
│  🧠 Reason          agent_node (调用 LLM)                │
│  🔧 Act             tools_node (执行 ToolNode)            │
│  👁️ Observe          ToolMessage 写回 State.messages      │
│  🔄 Loop            tools → agent 循环边                  │
│  ✅ Stop             should_continue 返回 END             │
│  🛡️ Safety Valve    iteration 计数 + 最大迭代检查           │
└──────────────────────────────────────────────────────────┘
```

---

## 九、进阶：使用预构建 `create_react_agent`

如果你不需要自定义图结构，LangGraph 也提供了开箱即用的高层 API：

```python
from langgraph.prebuilt import create_react_agent

graph = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=ALL_TOOLS,
    prompt=AGENT_SYSTEM_PROMPT,
)

result = graph.invoke({
    "messages": [{"role": "user", "content": f"请抓取：{url}"}]
})
```

预构建的 `create_react_agent` 是快速入门的好方式，但当你需要更多控制和定制时，可以创建自定义的 ReAct Agent。 ([How to create a ReAct agent from scratch](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/))我们本文选择从头构建，就是为了让你**看清图的内部结构**。

---

## 十、扩展方向

LangGraph 的图结构让扩展变得非常容易——只需要**加节点、加边**：

```
                     ┌──────────┐
      START ──────▶ │  agent   │
                     └────┬─────┘
                          │
                  ┌───────┴────────┐
                  ▼                ▼
            has tool_calls    no tool_calls
                  │                │
                  ▼                ▼
            ┌──────────┐   ┌────────────┐
            │  tools   │   │  validate  │ ← 🆕 新增验证节点
            └────┬─────┘   └─────┬──────┘
                 │               │
                 └──▶ agent      └──▶ END
```

| 扩展方向 | 做法 |
|---|---|
| **Human-in-the-loop** | 在 tools 前加 `interrupt` 节点，人工审批后再执行 |
| **更多工具** | 加 `fetch_pdf`、`fetch_with_jina`，只需扩充 `ALL_TOOLS` 列表 |
| **内容验证节点** | 加 `validate` 节点，在 END 前检查内容质量，不达标则回退 |
| **持久化** | 编译时传入 `InMemorySaver`：`workflow.compile(checkpointer=saver)` |
| **可视化** | `fetch_graph.get_graph().draw_mermaid_png()` 生成流程图 |

---

## 十一、总结

### LangGraph 做了什么？

`create_agent` 使用 LangGraph 构建基于图的 Agent 运行时。图由节点（步骤）和边（连接）组成，定义了 Agent 如何处理信息。Agent 在图中移动，执行模型节点（调用模型）、工具节点（执行工具）等。 ([Agents - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/agents))

### 一句话总结变化

```
Before:  while True → if tool_calls → execute → append   (命令式)
After:   Node → Edge → Node → Edge → ...                 (声明式)
```

本质没变：还是 **Reason → Act → Observe** 循环。但 LangGraph 让你把**循环逻辑建模为图结构**，获得了：

1. **可视化**：图结构可以直接生成流程图
2. **可扩展**：加节点加边，不用改循环逻辑
3. **可观测**：集成 LangSmith，追踪每个节点的输入输出
4. **可持久化**：内置 Checkpointer，支持断点续传
5. **可中断**：原生支持 Human-in-the-loop

LangGraph 为任何长时间运行的有状态工作流或 Agent 提供低层级支持基础设施，核心优势包括：持久化执行（Agent 可从失败中恢复）、Human-in-the-loop（在执行的任何时刻检查和修改 Agent 状态）、以及全面的记忆管理（短期工作记忆和长期持久化记忆）。 ([GitHub - langchain-ai/langgraph: Build resilient language agents as graphs.](https://github.com/langchain-ai/langgraph))
