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

from .tools import ALL_TOOLS

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

## 策略（请根据 URL 类型选择）

### RFC 文档
如果 URL 包含 "rfc" 和数字（如 rfc7519、#rfc7519）：
1. **优先使用 `fetch_rfc_direct`**，直接从 IETF 官方源抓取
2. 从 URL 中提取 RFC 编号（如从 "#rfc7519" 提取 "7519"）
3. 如果失败，再尝试其他方法

### 普通网页
1. 先用 `fetch_static` 工具尝试静态抓取
2. 如果返回内容包含 [WARN] 或 [ERROR]，改用 `fetch_dynamic` 进行浏览器渲染抓取
3. 如果抓取到的是 JavaScript 代码而非正文，说明页面需要特殊处理

## 注意
- 最终必须返回抓取到的纯文本正文
- 不要编造内容
- 最多尝试 3 次工具调用
- 如果抓取到的内容包含大量 JavaScript 函数或代码，说明抓取策略不对
"""


def create_llm():
    """创建 LLM 实例，支持自签名证书的本地部署。"""
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    disable_ssl = os.getenv("DISABLE_SSL_VERIFY", "false").lower() == "true"

    # 配置 httpx 客户端以支持自签名证书
    import httpx
    http_client = httpx.Client(verify=not disable_ssl) if disable_ssl else None
    http_async_client = httpx.AsyncClient(verify=not disable_ssl) if disable_ssl else None

    kwargs = {
        "model": model,
        "temperature": 0,
    }

    if api_key:
        kwargs["openai_api_key"] = api_key

    if base_url:
        kwargs["openai_api_base"] = base_url

    if http_client:
        kwargs["http_client"] = http_client

    if http_async_client:
        kwargs["http_async_client"] = http_async_client

    return ChatOpenAI(**kwargs)


# 延迟初始化 LLM，避免在导入时就要求 API key
_llm = None
_llm_with_tools = None


def get_llm():
    """获取 LLM 实例（延迟初始化）"""
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm


def get_llm_with_tools():
    """获取绑定工具的 LLM 实例（延迟初始化）"""
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm().bind_tools(ALL_TOOLS)
    return _llm_with_tools


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
    response = get_llm_with_tools().invoke(messages)

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
            print(f"  👁️  [tools 节点] Observation: {preview}...")

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
