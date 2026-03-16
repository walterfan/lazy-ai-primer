# 数据模型与 API

## 概述

nanobot 不使用传统数据库。所有持久化数据都以**文件**形式存储在本地文件系统中，主要位于 `~/.nanobot/` 目录下。本文档描述了数据模型、存储格式、配置 schema 以及内部/外部 API。

## 数据存储

### 存储位置

| 数据 | 路径 | 格式 |
|------|------|--------|
| 配置 | `~/.nanobot/config.json` | JSON |
| 会话 | `~/.nanobot/workspace/sessions/{key}.jsonl` | JSONL（每行一个 JSON 对象） |
| 长期记忆 | `~/.nanobot/workspace/memory/MEMORY.md` | Markdown |
| 历史日志 | `~/.nanobot/workspace/memory/HISTORY.md` | Markdown（仅追加） |
| 心跳任务 | `~/.nanobot/workspace/HEARTBEAT.md` | Markdown（复选框列表） |
| 定时任务 | 内存中（由 `CronService` 管理） | 仅运行时 |
| Skills（工作区） | `~/.nanobot/workspace/skills/{name}/SKILL.md` | Markdown |
| Skills（内置） | `nanobot/skills/{name}/SKILL.md` | Markdown |
| 引导文件 | `~/.nanobot/workspace/AGENTS.md`、`SOUL.md`、`USER.md`、`TOOLS.md`、`IDENTITY.md` | Markdown |

### 会话存储（JSONL）

每个会话存储为一个 `.jsonl` 文件。第一行始终是元数据记录，后续各行为消息。

**元数据记录**（第 1 行）：

```json
{
  "_type": "metadata",
  "key": "telegram:12345",
  "created_at": "2026-03-01T10:00:00",
  "updated_at": "2026-03-15T14:30:00",
  "metadata": {},
  "last_consolidated": 42
}
```

**消息记录**（第 2 行起）：

```json
{
  "role": "user",
  "content": "What is the weather today?",
  "timestamp": "2026-03-15T14:30:00.123456"
}
```

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{"id": "call_abc", "function": {"name": "web_search", "arguments": "{\"query\": \"weather today\"}"}}],
  "timestamp": "2026-03-15T14:30:01.234567"
}
```

```json
{
  "role": "tool",
  "content": "Sunny, 22°C",
  "tool_call_id": "call_abc",
  "name": "web_search",
  "timestamp": "2026-03-15T14:30:02.345678"
}
```

**关键设计决策**：
- **仅追加**：消息不会被修改或删除——只有 `last_consolidated` 会向前推进
- **JSONL 格式**：每条消息占一行，便于流式处理和 grep 检索
- **Session key**：`{channel}:{chat_id}`（例如 `telegram:12345`、`cli:default`）

### 记忆存储

`~/.nanobot/workspace/memory/` 下的两层记忆系统：

| 文件 | 用途 | 更新方式 |
|------|---------|---------------|
| `MEMORY.md` | 关于用户的长期事实 | 在整合时覆盖写入（LLM 重写全部内容） |
| `HISTORY.md` | 按时间顺序的事件日志 | 仅追加（新条目添加到末尾） |

**整合过程**：当会话超过阈值（`memory_window`）时，较早的消息会由 LLM 通过 `save_memory` 工具调用进行摘要，生成：
- `history_entry`：带时间戳的段落，追加到 `HISTORY.md`
- `memory_update`：完整更新后的 `MEMORY.md` 内容

## 核心数据模型

### Event 类型（`nanobot/bus/events.py`）

```python
@dataclass
class InboundMessage:
    channel: str              # "telegram", "discord", "slack", ...
    sender_id: str            # 用户标识
    chat_id: str              # 聊天/频道标识
    content: str              # 消息文本
    timestamp: datetime       # 接收时间
    media: list[str]          # 媒体 URL
    metadata: dict[str, Any]  # 渠道特定数据
    session_key_override: str | None  # 线程级 session key

@dataclass
class OutboundMessage:
    channel: str              # 目标渠道
    chat_id: str              # 目标聊天
    content: str              # 响应文本
    reply_to: str | None      # 回复的消息 ID
    media: list[str]          # 媒体附件
    metadata: dict[str, Any]  # 渠道特定数据
```

### Provider 类型（`nanobot/providers/base.py`）

```python
@dataclass
class ToolCallRequest:
    id: str                       # 唯一调用 ID
    name: str                     # 工具名称
    arguments: dict[str, Any]     # 工具参数

@dataclass
class LLMResponse:
    content: str | None           # 文本响应
    tool_calls: list[ToolCallRequest]  # 工具调用请求
    finish_reason: str            # "stop"、"tool_calls" 等
    usage: dict[str, int]         # Token 使用统计
    reasoning_content: str | None # DeepSeek-R1、Kimi 推理内容
    thinking_blocks: list[dict] | None  # Anthropic 扩展思考
```

### Session 模型（`nanobot/session/manager.py`）

```python
@dataclass
class Session:
    key: str                           # "channel:chat_id"
    messages: list[dict[str, Any]]     # 仅追加的消息列表
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
    last_consolidated: int             # 最后一条已整合消息的索引
```

### Cron 类型（`nanobot/cron/types.py`）

```python
@dataclass
class CronSchedule:
    kind: str          # "at" | "every" | "cron"
    at_ms: int | None  # 绝对时间戳（用于 "at"）
    every_ms: int | None  # 间隔时间（用于 "every"）
    expr: str | None   # Cron 表达式（用于 "cron"）
    tz: str | None     # 时区（用于 "cron"）

@dataclass
class CronJob:
    id: str
    description: str
    schedule: CronSchedule
    state: CronJobState
    payload: CronPayload
```

## 配置 Schema

根 `Config` 对象（`nanobot/config/schema.py`）是一个 Pydantic `BaseSettings` 模型：

```
Config
├── agents: AgentsConfig
│   └── defaults: AgentDefaults
│       ├── workspace: str = "~/.nanobot/workspace"
│       ├── model: str = "anthropic/claude-opus-4-5"
│       ├── provider: str = "auto"
│       ├── max_tokens: int = 8192
│       ├── temperature: float = 0.1
│       ├── max_tool_iterations: int = 40
│       ├── memory_window: int = 100
│       └── reasoning_effort: str | None
├── channels: ChannelsConfig
│   ├── send_progress: bool = True
│   ├── send_tool_hints: bool = False
│   ├── telegram: TelegramConfig
│   ├── discord: DiscordConfig
│   ├── whatsapp: WhatsAppConfig
│   ├── slack: SlackConfig
│   ├── feishu: FeishuConfig
│   ├── dingtalk: DingTalkConfig
│   ├── qq: QQConfig
│   ├── email: EmailConfig
│   ├── matrix: MatrixConfig
│   └── mochat: MochatConfig
├── providers: ProvidersConfig
│   ├── custom: ProviderConfig
│   ├── anthropic: ProviderConfig
│   ├── openai: ProviderConfig
│   ├── openrouter: ProviderConfig
│   ├── deepseek: ProviderConfig
│   ├── groq: ProviderConfig
│   ├── gemini: ProviderConfig
│   ├── moonshot: ProviderConfig
│   ├── zhipu: ProviderConfig
│   ├── dashscope: ProviderConfig
│   ├── minimax: ProviderConfig
│   ├── aihubmix: ProviderConfig
│   ├── siliconflow: ProviderConfig
│   ├── volcengine: ProviderConfig
│   ├── vllm: ProviderConfig
│   ├── openai_codex: ProviderConfig
│   └── github_copilot: ProviderConfig
├── gateway: GatewayConfig
│   ├── host: str = "0.0.0.0"
│   ├── port: int = 18790
│   └── heartbeat: HeartbeatConfig
│       ├── enabled: bool = True
│       └── interval_s: int = 1800
└── tools: ToolsConfig
    ├── restrict_to_workspace: bool = False
    ├── web: WebToolsConfig
    │   ├── proxy: str | None
    │   └── search: WebSearchConfig
    │       ├── api_key: str
    │       └── max_results: int = 5
    ├── exec: ExecToolConfig
    │   ├── timeout: int = 60
    │   └── path_append: str
    └── mcp_servers: dict[str, MCPServerConfig]
```

**配置加载流程**：`~/.nanobot/config.json` → `json.load()` → `_migrate_config()` → `Config.model_validate()`。

**关键特性**：`Base` 模型上的 `alias_generator=to_camel` 意味着 JSON 配置中同时接受 `"apiKey"`（camelCase）和 `"api_key"`（snake_case），从而兼容 Claude Desktop / Cursor MCP 的配置格式。

## Tool API（OpenAI Function Calling 格式）

所有工具都以 OpenAI 格式的 function schema 暴露给 LLM。示例：

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read the contents of a file.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "The file path to read."
        }
      },
      "required": ["path"]
    }
  }
}
```

### Tool 参数汇总

| Tool | 必需参数 | 可选参数 |
|------|-------------------|-------------------|
| `read_file` | `path: string` | — |
| `write_file` | `path: string`、`content: string` | — |
| `edit_file` | `path: string`、`old_text: string`、`new_text: string` | — |
| `list_dir` | `path: string` | — |
| `exec` | `command: string` | — |
| `web_search` | `query: string` | — |
| `web_fetch` | `url: string` | — |
| `message_user` | `content: string` | — |
| `spawn` | `task: string` | — |
| `cron` | 因 action 而异 | `action`、`id`、`description`、`schedule` |

## 外部 API 集成

nanobot 不对外暴露自己的 HTTP API，而是连接外部服务：

| 服务 | 协议 | 用途 |
|---------|----------|---------|
| LLM providers | HTTPS（OpenAI 兼容） | 带工具调用的聊天补全 |
| Brave Search | HTTPS | 网页搜索结果 |
| MCP servers | stdio / HTTP (SSE) | 外部工具执行 |
| 聊天平台 | WebSocket / HTTP long-poll / IMAP | 消息收发 |

## 相关文档

- [架构设计](02-architecture.md) — 组件设计
- [工作流](03-workflows.md) — 数据在系统中的流转
- [编码规范](05-conventions.md) — 代码规范

---

**最后更新**：2026-03-15
**版本**：1.0
