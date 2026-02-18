
# 从零构建 AI Agent：用 ReAct 模式打造「英文网页翻译助手」

> **TL;DR** — 本文用 Python + LangChain，从零搭建一个可运行的 ReAct Agent。输入一个英文网页 URL，它会**自主决策**如何抓取正文，然后做 **分块翻译 → 质量自检 → 要点总结 → 难点解释 → 生成 PlantUML 思维导图**。全部代码约 300 行。

---

## 一、什么是 AI Agent？

Anthropic 把 Agent 描述为 "systems where LLMs dynamically direct their own processes and tool usage"，OpenAI 则称其为 "systems that independently accomplish tasks on behalf of users"。本质上，Agent 是**能自主决策、使用工具、采取行动**来完成目标的系统。 ([Building ReAct agents with (and without) LangGraph – Dylan Castillo](https://dylancastillo.co/posts/react-agent-langgraph.html))

对比一下：

| | 普通 LLM 调用 | AI Agent |
|---|---|---|
| 控制流 | 你写死的 if/else | **LLM 自己决定**下一步 |
| 工具使用 | 不用 / 你手动调 | Agent **自主选择**工具 |
| 错误恢复 | 你写 try/except | Agent **观察结果**并重试 |
| 复杂任务 | 一次 prompt | **多轮循环**直到完成 |

一句话：**Agent = LLM（大脑）+ Tools（手脚）+ Loop（行动循环）**。

---

## 二、ReAct 模式：让 AI 学会"想—做—看"

ReAct 基于论文 "ReAct: Synergizing Reasoning and Acting in Language Models"（https://arxiv.org/abs/2210.03629）。 ([create_react_agent — 🦜🔗 LangChain documentation](https://api.python.langchain.com/en/latest/langchain/agents/langchain.agents.react.agent.create_react_agent.html))

核心思想是：模型先做 **Reasoning（推理）**——即 Re 部分，然后基于推理采取 **Action（行动）**——即 Act 部分；接着根据行动的结果（Observation），再次推理。 ([GitHub - botextractai/ai-langchain-react-agent: Create a LangChain ReAct agent with multiple tools (Python REPL and DuckDuckGo Search)](https://github.com/botextractai/ai-langchain-react-agent))

```
┌─────────────────────────────────────────────┐
│              ReAct 循环                       │
│                                              │
│   ┌──────────┐                               │
│   │  Reason  │  "这个 URL 看起来是 SPA，       │
│   │  (推理)   │   静态抓取可能不行"              │
│   └────┬─────┘                               │
│        ▼                                     │
│   ┌──────────┐                               │
│   │   Act    │  调用 fetch_dynamic(url)       │
│   │  (行动)   │                               │
│   └────┬─────┘                               │
│        ▼                                     │
│   ┌──────────┐                               │
│   │ Observe  │  "拿到了 3200 字的正文，        │
│   │  (观察)   │   内容有效！"                   │
│   └────┬─────┘                               │
│        │                                     │
│        ▼                                     │
│   有答案了？──Yes──▶ 返回结果                   │
│        │No                                   │
│        └──── 继续循环 ───────────┘             │
└─────────────────────────────────────────────┘
```

---

## 三、架构设计：两层分离

```
┌─────────────────────────────────────────────────────────────┐
│                        用户输入 URL                          │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  决策层 (Decision Layer) ── ReAct Agent                      │
│                                                              │
│  Agent 自主选择工具：                                         │
│    🔧 fetch_static  ── requests + readability (静态页面)      │
│    🔧 fetch_dynamic ── Playwright 渲染 (SPA/JS 页面)         │
│                                                              │
│  循环：Reason → Act → Observe → ... → 拿到高质量英文正文       │
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

**为什么分两层？**

- **决策层**的核心是"不确定性"——不知道哪种抓取方法能拿到好内容，所以需要 Agent 自主试错。
- **生成层**的核心是"确定性"——拿到正文后，流程是固定的管线，一步步走就行。

---

## 四、环境搭建

### 4.1 用 Poetry 初始化项目

```bash
mkdir url-translator && cd url-translator
poetry init --name url-translator --python "^3.11" -n
```

### 4.2 安装依赖

```bash
poetry add langchain-core langchain-openai requests beautifulsoup4 \
           readability-lxml playwright python-dotenv

# Playwright 需要额外安装浏览器
poetry run playwright install chromium
```

完整 `pyproject.toml`：

```toml
[tool.poetry]
name = "url-translator"
version = "0.1.0"
description = "ReAct Agent: fetch, translate, summarize, mindmap"

[tool.poetry.dependencies]
python = "^3.11"
langchain-core = ">=0.3"
langchain-openai = ">=0.3"
requests = "^2.31"
beautifulsoup4 = "^4.12"
readability-lxml = "^0.8"
playwright = "^1.40"
python-dotenv = "^1.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### 4.3 配置环境变量

```bash
# .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini          # 可选，默认 gpt-4o-mini
```

---

## 五、代码实现

### 项目结构

```
url-translator/
├── pyproject.toml
├── .env
├── tools.py          # 🔧 抓取工具定义
├── pipeline.py       # ⚙️  生成层管线
└── main.py           # 🚀 ReAct Agent + 入口
```

---

### 5.1 `tools.py` —— 抓取工具

```python
"""tools.py — 定义 Agent 可用的工具。

关键点：
  1. 用 @tool 装饰器让普通函数变成 LangChain Tool
  2. docstring 会作为工具描述发送给 LLM，所以要写清楚"什么时候该用这个工具"
  3. 返回值是字符串，Agent 会把它当做 Observation 来推理
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
        return text[:15000]  # 截断防止超 token
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
            return f"[WARN] Playwright 渲染后正文仅 {len(text)} 字符，页面可能需要登录或有反爬。"
        return text[:15000]
    except Exception as e:
        return f"[ERROR] 动态抓取失败: {e}"


# 把所有工具放到列表里，方便 Agent 使用
ALL_TOOLS = [fetch_static, fetch_dynamic]
```

> **要点**：`@tool` 装饰器做了三件事：① 把函数签名提取为 JSON Schema（让 LLM 知道参数）；② 把 docstring 作为工具描述；③ 把返回值包装成 `ToolMessage`。

---

### 5.2 `pipeline.py` —— 生成层管线

```python
"""pipeline.py — 生成层：翻译 → 自检 → 总结 → 难点 → 思维导图。

关键点：
  1. 用 LCEL (LangChain Expression Language) 的 `prompt | llm | parser` 语法构建链
  2. 长文本分块翻译，避免超 token
  3. 翻译后让 LLM 自检质量（self-reflection 模式）
  4. PlantUML 脚本可直接粘贴到 plantuml.com 渲染
"""

import os
import zlib
import requests as http_requests
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ── 初始化 LLM ──────────────────────────────────────────────
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

# ── 构建 LCEL Chains ────────────────────────────────────────

translate_chain = TRANSLATE_PROMPT | llm | parser
quality_chain   = QUALITY_CHECK_PROMPT | llm | parser
summary_chain   = SUMMARY_PROMPT | llm | parser
explain_chain   = EXPLAIN_PROMPT | llm | parser
mindmap_chain   = MINDMAP_PROMPT | llm | parser


# ── 工具函数 ────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    """按段落分块，每块不超过 max_chars 字符。"""
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
    """分块翻译 + 质量自检。这体现了 'self-reflection' 模式。"""
    chunks = chunk_text(text)
    results = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  📝 翻译第 {i}/{len(chunks)} 块...")
        # Step 1: 初翻
        raw = translate_chain.invoke({"text": chunk})
        # Step 2: 自检纠错 —— LLM 审视自己的翻译
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
    """把 PlantUML 脚本编码为 URL 安全字符串（deflate + 自定义 base64）。"""
    data = zlib.compress(text.encode("utf-8"))[2:-4]  # raw deflate
    res = []
    for i in range(0, len(data), 3):
        b = [data[i]] + [data[i+j] if i+j < len(data) else 0 for j in (1, 2)]
        res.append(_PLANTUML_ALPHABET[b[0] >> 2])
        res.append(_PLANTUML_ALPHABET[((b[0] & 0x3) << 4) | (b[1] >> 4)])
        res.append(_PLANTUML_ALPHABET[((b[1] & 0xF) << 2) | (b[2] >> 6)])
        res.append(_PLANTUML_ALPHABET[b[2] & 0x3F])
    return "".join(res)


def render_plantuml_png(script: str, output_path: str = "mindmap.png") -> str:
    """调用 PlantUML 在线服务，把脚本渲染成 PNG 图片。"""
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
    """
    生成层主函数：接收英文正文，输出翻译、总结、难点解释、思维导图。
    """
    print("\n🔄 [生成层] 开始处理...\n")

    # 1️⃣ 分块翻译 + 质量自检
    print("① 翻译中...")
    translation = translate_with_quality_check(original_text)

    # 2️⃣ 要点总结
    print("② 总结要点...")
    summary = summary_chain.invoke({"text": translation})

    # 3️⃣ 难点解释
    print("③ 解释难点...")
    explanation = explain_chain.invoke({"text": translation})

    # 4️⃣ 生成 PlantUML 思维导图
    print("④ 生成思维导图...")
    mindmap_script = mindmap_chain.invoke({"text": summary})

    # 清理：确保脚本格式正确
    if "```" in mindmap_script:
        # 去掉 markdown 代码块标记
        lines = mindmap_script.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        mindmap_script = "\n".join(lines)

    # 5️⃣ 渲染 PNG
    print("⑤ 渲染思维导图 PNG...")
    png_path = render_plantuml_png(mindmap_script)

    # 保存 .puml 文件
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

### 5.3 `main.py` —— ReAct Agent + 入口

这是**最核心**的文件。手动实现 ReAct 循环，让你清楚看到 Reason → Act → Observe 是怎么运作的。

```python
"""main.py — ReAct Agent 决策层 + 程序入口。

=== ReAct 循环的核心实现 ===

关键点：
  1. llm.bind_tools() 让 LLM 知道有哪些工具可用
  2. LLM 返回的 AIMessage 可能包含 tool_calls（即 Act）
  3. 我们执行工具，把结果作为 ToolMessage 追加（即 Observe）
  4. 循环直到 LLM 不再调用工具（即得出 Final Answer）
"""

import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from tools import ALL_TOOLS, fetch_static, fetch_dynamic
from pipeline import run_pipeline

load_dotenv()

# ── Agent 系统提示词 ────────────────────────────────────────

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

# ── 构建 Agent ──────────────────────────────────────────────

MAX_ITERATIONS = 5  # 安全阀：防止无限循环


def react_fetch(url: str) -> str:
    """
    ReAct Agent 决策层：自主决定用什么工具抓取网页内容。

    这就是 ReAct 的核心循环：
      while True:
          response = LLM.think(messages)       # 🧠 Reason
          if no tool_calls: break              # ✅ 有答案了
          for each tool_call:                  # 🔧 Act
              result = execute_tool(tool_call)
              messages.append(Observation)      # 👁️ Observe
    """
    # 初始化 LLM，绑定工具
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # 工具映射表：name → function
    tool_map = {t.name: t for t in ALL_TOOLS}

    # 消息列表（这就是 Agent 的"记忆"）
    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"请抓取以下 URL 的英文正文内容：{url}"),
    ]

    print("🤖 [决策层] ReAct Agent 启动\n")

    for step in range(1, MAX_ITERATIONS + 1):
        print(f"── 第 {step} 轮 ──")

        # ============================
        # 🧠 REASON: LLM 思考下一步
        # ============================
        response = llm_with_tools.invoke(messages)
        messages.append(response)  # 追加 AI 的回复

        # 如果 LLM 没有调用工具，说明已有结论
        if not response.tool_calls:
            print("  💡 Agent 得出结论，结束循环\n")
            return response.content

        # ============================
        # 🔧 ACT + 👁️ OBSERVE
        # ============================
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"  🔧 Action: {tool_name}({tool_args})")

            # 执行工具
            result = tool_map[tool_name].invoke(tool_args)

            # 截取前 200 字符显示
            preview = result[:200].replace("\n", " ")
            print(f"  👁️ Observation: {preview}...")

            # 把观察结果追加到消息列表
            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tc["id"],
                )
            )

    # 安全阀触发
    return messages[-1].content if messages else "[ERROR] Agent 未能获取内容"


# ── 主入口 ──────────────────────────────────────────────────

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else input("请输入英文网页 URL: ").strip()
    if not url:
        print("❌ URL 不能为空")
        return

    # ========== 第一阶段：决策层 ==========
    content = react_fetch(url)

    if "[ERROR]" in content or len(content) < 100:
        print(f"❌ 抓取失败：{content[:300]}")
        return

    print(f"✅ 成功获取正文（{len(content)} 字符）\n")

    # ========== 第二阶段：生成层 ==========
    result = run_pipeline(content)

    # ========== 输出结果 ==========
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
# 激活虚拟环境
poetry shell

# 运行（方式一：命令行参数）
python main.py "https://lilianweng.github.io/posts/2023-06-23-agent/"

# 运行（方式二：交互式输入）
python main.py
```

### 运行效果示例

```
🤖 [决策层] ReAct Agent 启动

── 第 1 轮 ──
  🔧 Action: fetch_static({"url": "https://lilianweng.github.io/posts/2023-06-23-agent/"})
  👁️ Observation: LLM Powered Autonomous Agents  June 23, 2023 · 40 min rea...
  💡 Agent 得出结论，结束循环

✅ 成功获取正文（12836 字符）

🔄 [生成层] 开始处理...

① 翻译中...
  📝 翻译第 1/5 块...
  📝 翻译第 2/5 块...
  📝 翻译第 3/5 块...
  📝 翻译第 4/5 块...
  📝 翻译第 5/5 块...
② 总结要点...
③ 解释难点...
④ 生成思维导图...
⑤ 渲染思维导图 PNG...

============================================================
📖 中文翻译
============================================================
LLM 驱动的自主智能体（Autonomous Agents）...

============================================================
📌 要点总结
============================================================
1. Agent 系统由 LLM 作为核心控制器...
2. 规划能力包括任务分解和自我反思...
...

============================================================
🗺️  PlantUML 思维导图脚本
============================================================
@startmindmap
* LLM 自主智能体
** 规划
*** 任务分解
*** 自我反思
** 记忆
*** 短期记忆
*** 长期记忆
** 工具使用
*** API 调用
*** 代码执行
@endmindmap

✅ 思维导图已保存为: mindmap.png
✅ PlantUML 脚本已保存为: mindmap.puml
```

---

## 七、核心要点回顾

### 7.1 ReAct 循环的本质就是一个 while 循环

```python
# 伪代码 —— 这就是 ReAct 的全部秘密
messages = [system_prompt, user_input]

while True:
    ai_response = llm.invoke(messages)   # 🧠 Reason
    messages.append(ai_response)

    if not ai_response.tool_calls:       # ✅ 完成
        break

    for tool_call in ai_response.tool_calls:
        result = execute(tool_call)      # 🔧 Act
        messages.append(Observation)     # 👁️ Observe
```

Agent 遵循 ReAct（"Reasoning + Acting"）模式，在简短的推理步骤与有针对性的工具调用之间交替，并将观察结果反馈给后续决策，直到得出最终答案。 ([Agents - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/agents))

### 7.2 Tool 的定义决定了 Agent 的能力边界

- **docstring 即说明书**：LLM 通过工具的描述决定何时调用它
- **返回值即反馈**：`[WARN]`/`[ERROR]` 前缀让 Agent 知道需要重试
- 要精确描述工具用途——避免模糊指令如 "search tool"，而应提供清晰指引如 "Use WebSearch tool only for questions requiring current information"，以确保 Agent 选择正确工具。 ([LangChain ReAct Agent: Complete Implementation Guide + Working Examples 2025](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-react-agent-complete-implementation-guide-working-examples-2025))

### 7.3 生成层用"自检"提升质量

```python
# 翻译 → 自检 是一种 Self-Reflection 模式
raw_translation  = translate_chain.invoke({"text": chunk})
final_translation = quality_chain.invoke({
    "original":    chunk,
    "translation": raw_translation,   # LLM 审视自己的输出
})
```

这和 ReAct 的"Observe"异曲同工——让 LLM 观察自己的输出，然后改进。

### 7.4 安全阀不可少

- 设置 `max_iterations` 限制（例如 5 次），防止 Agent 进入无限推理循环，这不仅避免过度 API 调用，也能控制成本。 ([LangChain ReAct Agent: Complete Implementation Guide + Working Examples 2025](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-react-agent-complete-implementation-guide-working-examples-2025))

---

## 八、扩展方向

| 方向 | 做法 |
|---|---|
| **更多抓取工具** | 加 `fetch_pdf` (用 PyMuPDF)、`fetch_with_jina` (Jina Reader API) |
| **并行翻译** | 用 `asyncio` + `chain.abatch()` 并行处理多个 chunk |
| **持久化记忆** | 加入 LangGraph 的 `InMemorySaver` 或数据库 checkpointer |
| **流式输出** | 用 `chain.astream()` 实现打字机效果 |
| **支持更多语言** | 让用户指定目标语言，prompt 加一个 `{target_lang}` 变量 |

---

## 九、总结

构建一个 AI Agent 其实就三步：

```
1. 定义工具（Tools）   → 给 AI 装上手脚
2. 编写循环（ReAct Loop）→ 让 AI 自主思考-行动-观察
3. 设置护栏（Guardrails）→ 安全阀 + 错误处理
```

ReAct Agent 是将大语言模型的推理能力与执行行动的能力相融合的 AI 系统。 ([Building ReAct agents with (and without) LangGraph – Dylan Castillo](https://dylancastillo.co/posts/react-agent-langgraph.html))它的强大之处不在于任何单个步骤有多复杂，而在于**把 LLM 的判断力放进了循环**，让系统有了"随机应变"的能力。

希望这篇文章能帮你理解 Agent 的本质，并动手构建属于自己的 Agent。完整代码已在文中给出，`poetry install` 后即可运行。Happy hacking! 🚀