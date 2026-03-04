(chapter18)=
# 第十八章：Agent 开发框架对比与实战

```{mermaid}
mindmap
  root((Agent开发框架))
    LangChain/LangGraph
      状态图
      条件路由
      人机交互
    CrewAI
      角色扮演
      任务编排
      Sequential/Hierarchical
      技术博客写作团队
    AutoGen
      多Agent对话
      微软生态
      RoundRobinGroupChat
      代码开发团队
    Semantic Kernel
      .NET/Python
      插件系统
    OpenAI Agents SDK
      官方框架
      Handoff转交
      Guardrails防护
      客服系统示例
    Dify/Coze
      低代码平台
      可视化编排
      RAG引擎
      API发布
    框架选型
      场景匹配
      团队技能
```

## 18.1 Agent 开发框架全景

随着 AI Agent 技术的快速发展，涌现出了众多开发框架。这些框架封装了 Agent 的核心模式（如 ReAct、Plan-and-Execute），提供了工具集成、记忆管理、多 Agent 协作等开箱即用的能力。选择合适的框架对项目的成功至关重要。

```
┌─────────────────────────────────────────────────────────────┐
│                  Agent 开发框架全景图                          │
├──────────────┬──────────────────────────────────────────────┤
│  代码优先     │  LangChain/LangGraph, AutoGen, CrewAI,       │
│  (Code-first) │  Semantic Kernel, OpenAI Agents SDK          │
├──────────────┼──────────────────────────────────────────────┤
│  低代码/无代码 │  Dify, Coze, Flowise, LangFlow               │
├──────────────┼──────────────────────────────────────────────┤
│  企业级平台   │  AWS Bedrock Agents, Azure AI Agent Service,  │
│              │  Google Vertex AI Agent Builder               │
└──────────────┴──────────────────────────────────────────────┘
```

## 18.2 主流框架深度对比

### 18.2.1 LangChain / LangGraph

LangChain 是最早也是最流行的 LLM 应用开发框架。LangGraph 是其子项目，专注于构建有状态的、多步骤的 Agent 工作流。

**核心特点：**
- 丰富的组件生态（LLM、向量数据库、工具集成）
- LangGraph 提供基于图的状态机，支持复杂的控制流
- LangSmith 提供可观测性和评估能力
- 社区活跃，文档完善

```python
# LangGraph 核心概念示例
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[list, operator.add]  # 消息累加
    next_action: str
    iteration: int
```

### 18.2.2 CrewAI

CrewAI 专注于多 Agent 协作，以角色扮演为核心理念。

**核心特点：**
- 直观的角色定义（Agent = Role + Goal + Backstory）
- 内置多种协作模式（Sequential、Hierarchical）
- 简洁的 API，学习曲线平缓
- 适合快速构建多 Agent 原型

### 18.2.3 AutoGen (Microsoft)

AutoGen 是微软推出的多 Agent 对话框架，强调 Agent 之间的对话式协作。

**核心特点：**
- 基于对话的 Agent 交互模型
- 支持人机混合协作
- 灵活的 Agent 配置
- 强大的代码执行能力

### 18.2.4 Semantic Kernel (Microsoft)

Semantic Kernel 是微软的另一个 AI 编排框架，更偏向企业级应用。

**核心特点：**
- 支持 C#、Python、Java 多语言
- 与 Azure 生态深度集成
- Plugin 架构，易于扩展
- 企业级安全和治理

### 18.2.5 OpenAI Agents SDK

OpenAI 官方推出的轻量级 Agent 框架，与 OpenAI API 深度集成。

**核心特点：**
- 原生支持 OpenAI 模型和工具
- 内置 Handoff 机制实现多 Agent 协作
- Guardrails 安全防护
- 追踪和可观测性

### 18.2.6 Dify / Coze

低代码 Agent 构建平台，适合非技术用户。

**核心特点：**
- 可视化工作流编排
- 拖拽式工具集成
- 内置 RAG 管道
- 一键部署

## 18.3 CrewAI：角色扮演的多 Agent 框架

CrewAI 的核心理念是**角色扮演**——每个 Agent 都有明确的角色、目标和背景故事，像一个真实的团队一样协作。

### 安装与基本概念

```bash
pip install crewai crewai-tools
```

CrewAI 的四个核心概念：
- **Agent**：有角色、目标、背景的智能体
- **Task**：分配给 Agent 的具体任务
- **Tool**：Agent 可以使用的工具
- **Crew**：Agent 团队，定义协作流程

### 完整示例：技术博客写作团队

```python
"""
用 CrewAI 构建一个技术博客写作团队：
- 研究员：搜索和整理技术资料
- 作者：撰写博客文章
- 编辑：审查和优化文章
"""
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool

# ── 定义工具 ──
search_tool = SerperDevTool()  # Google 搜索
web_tool = WebsiteSearchTool()  # 网页内容提取

# ── 定义 Agent ──
researcher = Agent(
    role="技术研究员",
    goal="深入研究指定技术主题，收集最新、最准确的信息",
    backstory="""你是一位资深的技术研究员，擅长从海量信息中
    提取关键洞察。你总是追求信息的准确性和时效性，
    会交叉验证多个来源。""",
    tools=[search_tool, web_tool],
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="技术博客作者",
    goal="基于研究资料撰写高质量、易读的技术博客文章",
    backstory="""你是一位经验丰富的技术写作者，擅长将复杂的
    技术概念用通俗易懂的语言解释。你的文章结构清晰，
    代码示例实用，读者反馈一直很好。""",
    verbose=True,
    allow_delegation=False,
)

editor = Agent(
    role="技术编辑",
    goal="审查文章的技术准确性、可读性和结构完整性",
    backstory="""你是一位严格的技术编辑，有 10 年的出版经验。
    你关注技术准确性、逻辑连贯性、语言流畅性，
    同时确保文章对目标读者友好。""",
    verbose=True,
    allow_delegation=False,
)

# ── 定义任务 ──
research_task = Task(
    description="""研究 "{topic}" 这个主题：
    1. 搜索最新的技术文章和官方文档
    2. 整理核心概念、优缺点、使用场景
    3. 收集 2-3 个实际应用案例
    4. 记录关键数据和引用来源""",
    expected_output="一份结构化的研究报告，包含核心概念、优缺点、案例和引用",
    agent=researcher,
)

writing_task = Task(
    description="""基于研究报告撰写一篇技术博客文章：
    1. 引人入胜的开头
    2. 清晰的概念解释（配代码示例）
    3. 实际应用场景和案例
    4. 优缺点分析
    5. 总结和建议
    文章长度：2000-3000 字""",
    expected_output="一篇完整的技术博客文章（Markdown 格式）",
    agent=writer,
    context=[research_task],  # 依赖研究任务的输出
)

editing_task = Task(
    description="""审查和优化博客文章：
    1. 检查技术准确性
    2. 优化文章结构和逻辑流
    3. 改进语言表达
    4. 确保代码示例可运行
    5. 添加必要的注释和说明""",
    expected_output="审查后的最终版本文章",
    agent=editor,
    context=[writing_task],
)

# ── 组建团队并执行 ──
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # 顺序执行
    verbose=True,
)

# 执行
result = crew.kickoff(inputs={"topic": "RAG 在企业中的最佳实践"})
print(result)
```

### CrewAI 的协作模式

```python
# 顺序执行：A → B → C
crew = Crew(process=Process.sequential, ...)

# 层级执行：Manager 分配任务给 Agent
crew = Crew(
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4o"),
    ...
)
```

## 18.4 OpenAI Agents SDK：官方轻量级框架

OpenAI 在 2025 年发布了 Agents SDK（前身是 Swarm），提供了一个轻量但强大的 Agent 开发框架。

### 安装

```bash
pip install openai-agents
```

### 核心概念

```python
from agents import Agent, Runner, function_tool, handoff

# ── 定义工具 ──
@function_tool
def search_database(query: str, table: str) -> str:
    """搜索数据库中的记录。
    
    Args:
        query: 搜索关键词
        table: 要搜索的表名（users, orders, products）
    """
    # 实际的数据库查询逻辑
    return f"在 {table} 表中找到 3 条匹配 '{query}' 的记录"

@function_tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件给指定用户。"""
    return f"邮件已发送给 {to}"

# ── 定义 Agent ──
customer_support = Agent(
    name="客服助手",
    instructions="""你是一个友好的客服助手。
    帮助用户查询订单、解答问题。
    如果遇到技术问题，转交给技术支持。
    如果遇到退款请求，转交给退款专员。""",
    tools=[search_database, send_email],
    handoffs=[],  # 稍后设置
)

tech_support = Agent(
    name="技术支持",
    instructions="""你是技术支持专员。
    帮助用户解决技术问题，如登录失败、功能异常等。
    可以查询数据库获取用户信息。""",
    tools=[search_database],
)

refund_agent = Agent(
    name="退款专员",
    instructions="""你是退款专员。
    处理用户的退款请求。
    需要验证订单信息后才能处理退款。""",
    tools=[search_database],
)

# 设置 Agent 间的转交关系
customer_support.handoffs = [
    handoff(tech_support, description="技术问题转交给技术支持"),
    handoff(refund_agent, description="退款请求转交给退款专员"),
]

# ── 运行 ──
async def main():
    result = await Runner.run(
        customer_support,
        input="我的订单 #12345 一直显示处理中，已经三天了",
    )
    print(result.final_output)

import asyncio
asyncio.run(main())
```

### Handoff 模式：Agent 间的智能转交

```python
# OpenAI Agents SDK 的核心创新是 Handoff（转交）
# Agent 可以根据对话内容，自动将用户转交给更合适的 Agent

# 转交是单向的：A → B，B 接管后 A 不再参与
# 这模拟了真实客服中的"转接"场景

# 也可以用 Guardrails 做输入/输出检查
from agents import GuardrailFunctionOutput, InputGuardrail

@InputGuardrail
async def check_sensitive_info(input: str) -> GuardrailFunctionOutput:
    """检查用户输入是否包含敏感信息"""
    sensitive_patterns = ["身份证", "银行卡", "密码"]
    for pattern in sensitive_patterns:
        if pattern in input:
            return GuardrailFunctionOutput(
                should_block=True,
                message="请不要在对话中分享敏感个人信息。"
            )
    return GuardrailFunctionOutput(should_block=False)
```

## 18.5 AutoGen：微软的多 Agent 对话框架

AutoGen 是微软研究院开发的多 Agent 框架，核心理念是**Agent 之间通过对话协作**。

### 安装

```bash
pip install autogen-agentchat autogen-ext
```

### 完整示例：代码开发团队

```python
"""
用 AutoGen 构建代码开发团队：
- 产品经理：定义需求
- 开发者：编写代码
- 测试员：编写和运行测试
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")

# 定义 Agent
product_manager = AssistantAgent(
    name="ProductManager",
    model_client=model_client,
    system_message="""你是产品经理。你的职责是：
    1. 将用户需求转化为清晰的技术需求
    2. 定义验收标准
    3. 审查最终结果是否满足需求
    当你认为任务完成时，回复 'APPROVE'。""",
)

developer = AssistantAgent(
    name="Developer",
    model_client=model_client,
    system_message="""你是高级开发者。你的职责是：
    1. 根据需求编写高质量的 Python 代码
    2. 遵循最佳实践（类型注解、错误处理、文档）
    3. 根据测试反馈修复问题""",
)

tester = AssistantAgent(
    name="Tester",
    model_client=model_client,
    system_message="""你是测试工程师。你的职责是：
    1. 为开发者的代码编写 pytest 测试
    2. 覆盖正常流程、边界条件和异常情况
    3. 报告发现的问题""",
)

# 组建团队（轮流发言）
termination = TextMentionTermination("APPROVE")
team = RoundRobinGroupChat(
    participants=[product_manager, developer, tester],
    termination_condition=termination,
    max_turns=12,
)

# 执行
import asyncio

async def main():
    result = await team.run(
        task="开发一个 Python 函数，实现 LRU 缓存，支持过期时间和最大容量限制"
    )
    for message in result.messages:
        print(f"\n{'='*50}")
        print(f"[{message.source}]:")
        print(message.content[:500])

asyncio.run(main())
```

## 18.6 Dify / Coze：低代码 Agent 平台

对于不想写代码的团队，Dify 和 Coze 提供了**可视化的 Agent 编排平台**。

### Dify 核心特性

```yaml
# Dify 的优势：
开源自部署:
  - Docker 一键部署
  - 数据完全自控
  - 支持私有化 LLM

可视化编排:
  - 拖拽式工作流设计
  - 条件分支、循环、并行
  - 实时调试和预览

RAG 引擎:
  - 内置文档管理
  - 多种分块策略
  - 向量数据库集成

API 发布:
  - 一键生成 REST API
  - WebSocket 支持
  - 嵌入式 Widget
```

### Dify API 调用示例

```python
"""通过 API 调用 Dify 上部署的 Agent"""
import requests

DIFY_API_URL = "https://api.dify.ai/v1"
API_KEY = "app-xxxxxxxxxxxx"

# 发送对话消息
response = requests.post(
    f"{DIFY_API_URL}/chat-messages",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "inputs": {},
        "query": "帮我分析最近一周的销售数据趋势",
        "user": "user-123",
        "conversation_id": "",  # 空字符串表示新对话
        "response_mode": "streaming",
    },
    stream=True,
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

## 18.7 框架选型指南

```python
# 框架选型决策矩阵
framework_comparison = {
    "框架": [
        "LangGraph", "CrewAI", "AutoGen",
        "Semantic Kernel", "OpenAI Agents SDK", "Dify"
    ],
    "学习曲线": ["陡峭", "平缓", "中等", "中等", "平缓", "极低"],
    "灵活性":   ["极高", "中等", "高",   "高",   "中等", "低"],
    "多Agent":  ["强",   "极强", "极强", "中等", "强",   "弱"],
    "生态":     ["极丰富","中等", "中等", "丰富", "有限", "丰富"],
    "生产就绪": ["高",   "中等", "中等", "高",   "高",   "高"],
    "适用场景": [
        "复杂工作流、需要精细控制",
        "多角色协作、快速原型",
        "研究探索、对话式协作",
        "企业级应用、.NET生态",
        "OpenAI生态、快速开发",
        "非技术用户、快速上线"
    ]
}
```

**选型建议：**
- **需要精细控制流** → LangGraph
- **多角色协作为主** → CrewAI
- **微软技术栈** → Semantic Kernel
- **快速原型验证** → OpenAI Agents SDK
- **非技术团队** → Dify / Coze

## 18.8 实战：用 LangGraph 构建完整 Agent

接下来我们用 LangGraph 构建一个具备工具调用、条件路由和人机交互能力的完整 Agent。这个 Agent 能够帮助用户进行信息检索、数据分析和内容生成。

### 18.8.1 环境准备

```bash
pip install langgraph langchain-openai langchain-community tavily-python
```

### 18.8.2 定义状态和工具

```python
"""
完整的 LangGraph Agent 实现
功能：信息检索、数据分析、内容生成
"""
import json
import operator
from typing import TypedDict, Annotated, Literal, Sequence
from datetime import datetime

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ============================================================
# 1. 定义 Agent 状态
# ============================================================
class AgentState(TypedDict):
    """Agent 的状态定义 - 贯穿整个工作流"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str                    # 下一个节点
    task_type: str               # 任务类型
    iteration_count: int         # 迭代计数
    human_feedback: str          # 人类反馈
    final_output: str            # 最终输出

# ============================================================
# 2. 定义工具
# ============================================================
@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息。适用于需要实时数据或最新资讯的场景。"""
    # 实际项目中使用 Tavily、Serper 等搜索 API
    # 这里用模拟数据演示
    mock_results = {
        "AI Agent 最新进展": "2025年，AI Agent 领域取得重大突破：OpenAI 发布了 Agents SDK，"
                           "Anthropic 推出了 Claude Agent，Google 发布了 Gemini Agent...",
        "Python 性能优化": "Python 3.13 引入了 JIT 编译器，性能提升约 30%。"
                         "此外，使用 asyncio、multiprocessing 等也能显著提升性能。",
    }
    for key, value in mock_results.items():
        if any(word in query for word in key.split()):
            return value
    return f"搜索 '{query}' 的结果：找到了相关信息，包括最新的技术动态和行业分析。"

@tool
def analyze_data(data_description: str) -> str:
    """分析数据并生成洞察。输入数据描述，返回分析结果。"""
    return f"数据分析完成。基于 '{data_description}' 的分析：\n" \
           f"1. 数据趋势：呈上升趋势\n" \
           f"2. 关键指标：均值偏高，方差较小\n" \
           f"3. 建议：关注异常值，考虑季节性因素"

@tool
def generate_report(topic: str, format: str = "markdown") -> str:
    """生成结构化报告。指定主题和格式（markdown/html/json）。"""
    return f"# {topic} 报告\n\n" \
           f"## 概述\n关于 {topic} 的综合分析报告。\n\n" \
           f"## 关键发现\n- 发现1: 市场增长显著\n- 发现2: 技术成熟度提升\n\n" \
           f"## 建议\n1. 加大投入\n2. 关注竞争态势\n\n" \
           f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"

@tool
def execute_code(code: str) -> str:
    """执行 Python 代码并返回结果。仅支持安全的计算操作。"""
    try:
        # 安全沙箱（简化版，生产环境需要更严格的沙箱）
        allowed_builtins = {"abs", "len", "max", "min", "sum", "round", "sorted", "range", "list", "dict", "str", "int", "float"}
        result = eval(code, {"__builtins__": {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in allowed_builtins if hasattr(__builtins__, k) or (isinstance(__builtins__, dict) and k in __builtins__)}})
        return f"执行结果: {result}"
    except Exception as e:
        return f"执行错误: {str(e)}"

tools = [search_web, analyze_data, generate_report, execute_code]

# ============================================================
# 3. 定义 LLM
# ============================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ============================================================
# 4. 定义图节点（Node Functions）
# ============================================================
def classify_task(state: AgentState) -> AgentState:
    """任务分类节点：分析用户意图，确定任务类型"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    classification_prompt = SystemMessage(content="""
    分析用户的请求，将其分类为以下类型之一：
    - search: 需要搜索信息
    - analysis: 需要数据分析
    - report: 需要生成报告
    - code: 需要执行代码
    - general: 一般对话

    只回复类型名称。
    """)

    response = llm.invoke([classification_prompt, messages[-1]])
    task_type = response.content.strip().lower()

    return {
        "messages": [AIMessage(content=f"[系统] 任务分类: {task_type}")],
        "task_type": task_type,
        "next": "agent",
        "iteration_count": state.get("iteration_count", 0),
        "human_feedback": "",
        "final_output": ""
    }

def agent_node(state: AgentState) -> AgentState:
    """核心 Agent 节点：推理并决定行动"""
    messages = state["messages"]
    iteration = state.get("iteration_count", 0)

    system_msg = SystemMessage(content=f"""你是一个智能助手 Agent。
当前任务类型: {state.get('task_type', 'general')}
当前迭代: {iteration}
人类反馈: {state.get('human_feedback', '无')}

请根据用户需求，选择合适的工具完成任务。
如果任务已完成，直接给出最终答案。""")

    # 过滤掉系统标记消息，保留有效对话
    effective_messages = [
        m for m in messages if not (isinstance(m, AIMessage) and m.content.startswith("[系统]"))
    ]

    response = llm_with_tools.invoke([system_msg] + effective_messages)

    return {
        "messages": [response],
        "next": "tools" if response.tool_calls else "review",
        "task_type": state.get("task_type", "general"),
        "iteration_count": iteration + 1,
        "human_feedback": state.get("human_feedback", ""),
        "final_output": response.content if not response.tool_calls else ""
    }

def review_node(state: AgentState) -> AgentState:
    """审查节点：评估 Agent 输出质量"""
    messages = state["messages"]
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.content.startswith("[系统]"):
            last_ai_message = msg
            break

    if not last_ai_message:
        return {**state, "messages": [], "next": "end"}

    review_prompt = f"""评估以下回答的质量（1-10分）：
回答: {last_ai_message.content[:500]}

如果分数 >= 7，回复 "APPROVED"。
否则回复 "NEEDS_IMPROVEMENT: <改进建议>"。"""

    response = llm.invoke([HumanMessage(content=review_prompt)])
    review_result = response.content

    if "APPROVED" in review_result:
        return {
            "messages": [AIMessage(content=f"[系统] 审查通过")],
            "next": "end",
            "task_type": state.get("task_type", ""),
            "iteration_count": state.get("iteration_count", 0),
            "human_feedback": state.get("human_feedback", ""),
            "final_output": last_ai_message.content
        }
    else:
        return {
            "messages": [AIMessage(content=f"[系统] 需要改进: {review_result}")],
            "next": "agent",
            "task_type": state.get("task_type", ""),
            "iteration_count": state.get("iteration_count", 0),
            "human_feedback": review_result,
            "final_output": ""
        }

def human_review_node(state: AgentState) -> AgentState:
    """人机交互节点：等待人类确认（使用 LangGraph interrupt）"""
    # 在实际部署中，这里会暂停执行等待人类输入
    # LangGraph 支持通过 interrupt 机制实现
    return {
        "messages": [AIMessage(content="[系统] 等待人类确认...")],
        "next": "agent",
        "task_type": state.get("task_type", ""),
        "iteration_count": state.get("iteration_count", 0),
        "human_feedback": state.get("human_feedback", ""),
        "final_output": state.get("final_output", "")
    }

# ============================================================
# 5. 定义条件路由
# ============================================================
def route_after_agent(state: AgentState) -> Literal["tools", "review", "end"]:
    """Agent 节点后的路由逻辑"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # 超过最大迭代次数，强制结束
    if state.get("iteration_count", 0) > 5:
        return "end"

    # 如果有工具调用，路由到工具节点
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 否则进入审查
    return "review"

def route_after_review(state: AgentState) -> Literal["agent", "end"]:
    """审查节点后的路由逻辑"""
    if state.get("next") == "end" or state.get("iteration_count", 0) > 5:
        return "end"
    return "agent"

# ============================================================
# 6. 构建图
# ============================================================
def build_agent_graph():
    """构建完整的 Agent 工作流图"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("classify", classify_task)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("review", review_node)

    # 添加边
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "agent")
    workflow.add_conditional_edges("agent", route_after_agent)
    workflow.add_edge("tools", "agent")  # 工具执行后回到 Agent
    workflow.add_conditional_edges("review", route_after_review)

    # 设置检查点（支持状态持久化和恢复）
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph

# ============================================================
# 7. 运行 Agent
# ============================================================
def run_agent(query: str):
    """运行 Agent 处理用户查询"""
    graph = build_agent_graph()

    config = {"configurable": {"thread_id": "user-session-001"}}

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next": "",
        "task_type": "",
        "iteration_count": 0,
        "human_feedback": "",
        "final_output": ""
    }

    print(f"🚀 用户查询: {query}\n")
    print("=" * 60)

    for event in graph.stream(initial_state, config):
        for node_name, node_output in event.items():
            print(f"\n📍 节点: {node_name}")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if isinstance(msg, AIMessage):
                        if msg.content and not msg.content.startswith("[系统]"):
                            print(f"   🤖 Agent: {msg.content[:200]}...")
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"   🔧 调用工具: {tc['name']}({tc['args']})")
                    elif isinstance(msg, ToolMessage):
                        print(f"   📋 工具结果: {msg.content[:150]}...")

    print("\n" + "=" * 60)
    print("✅ Agent 执行完成")

# 使用示例
# run_agent("请搜索 AI Agent 的最新进展，并生成一份简要报告")
```

### 18.8.3 可视化工作流

```python
# LangGraph 支持将工作流导出为 Mermaid 图
def visualize_graph():
    graph = build_agent_graph()
    # 生成 Mermaid 格式的流程图
    print(graph.get_graph().draw_mermaid())

# 输出类似：
# graph TD
#     __start__ --> classify
#     classify --> agent
#     agent -->|has_tool_calls| tools
#     agent -->|no_tool_calls| review
#     tools --> agent
#     review -->|approved| __end__
#     review -->|needs_improvement| agent
```

### 18.8.4 添加人机交互（Human-in-the-Loop）

```python
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt, Command

def human_approval_tool(action_description: str) -> str:
    """需要人类审批的操作"""
    # interrupt 会暂停图的执行，等待外部输入
    decision = interrupt({
        "question": f"是否批准以下操作？\n{action_description}",
        "options": ["approve", "reject", "modify"]
    })

    if decision == "approve":
        return "操作已批准，继续执行。"
    elif decision == "reject":
        return "操作已拒绝，请选择替代方案。"
    else:
        return f"操作需要修改: {decision}"

# 在实际应用中，通过 API 恢复执行：
# graph.invoke(Command(resume="approve"), config)
```

## 18.9 框架集成最佳实践

### 18.9.1 错误处理与重试

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustAgent:
    """带有健壮错误处理的 Agent"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def call_llm(self, messages):
        """带重试的 LLM 调用"""
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            print(f"LLM 调用失败: {e}, 重试中...")
            raise

    def safe_tool_call(self, tool_name, tool_input, timeout=30):
        """带超时和错误处理的工具调用"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"工具 {tool_name} 执行超时")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            result = self.tools[tool_name](tool_input)
            return {"success": True, "result": result}
        except TimeoutError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            signal.alarm(0)
```

### 18.9.2 成本控制

```python
class CostTracker:
    """LLM 调用成本追踪"""

    PRICING = {
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    }

    def __init__(self, budget_limit: float = 1.0):
        self.total_cost = 0.0
        self.budget_limit = budget_limit
        self.call_log = []

    def track(self, model: str, input_tokens: int, output_tokens: int):
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = input_tokens * pricing["input"] + output_tokens * pricing["output"]
        self.total_cost += cost
        self.call_log.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "cumulative": self.total_cost
        })

        if self.total_cost > self.budget_limit:
            raise Exception(f"预算超限! 已花费: ${self.total_cost:.4f}, 限额: ${self.budget_limit:.2f}")

        return cost
```

## 18.10 本章小结

本章全面对比了主流 Agent 开发框架，并通过 LangGraph 实战展示了如何构建一个完整的 Agent 系统。关键要点：

1. **框架选型**应基于团队技术栈、项目复杂度和控制需求
2. **LangGraph** 提供了最灵活的图式工作流编排能力
3. **状态管理**是 Agent 系统的核心，LangGraph 的 TypedDict + Checkpointer 模式值得借鉴
4. **条件路由**使 Agent 能够根据运行时状态动态调整执行路径
5. **人机交互**（Human-in-the-Loop）是生产级 Agent 的必备能力
6. **错误处理和成本控制**是工程化落地的关键

下一章我们将深入探讨 Agent 的工具系统设计，包括 Function Calling 和 MCP 协议。
