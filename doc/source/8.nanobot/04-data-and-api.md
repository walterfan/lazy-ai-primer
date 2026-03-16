# Data Model and API

## Overview

nanobot does not use a traditional database. All persistent data is stored as **files** on the local filesystem, primarily under `~/.nanobot/`. This document describes the data models, storage formats, configuration schema, and the internal/external APIs.

## Data Storage

### Storage Locations

| Data | Path | Format |
|------|------|--------|
| Configuration | `~/.nanobot/config.json` | JSON |
| Sessions | `~/.nanobot/workspace/sessions/{key}.jsonl` | JSONL (one JSON object per line) |
| Long-term memory | `~/.nanobot/workspace/memory/MEMORY.md` | Markdown |
| History log | `~/.nanobot/workspace/memory/HISTORY.md` | Markdown (append-only) |
| Heartbeat tasks | `~/.nanobot/workspace/HEARTBEAT.md` | Markdown (checkbox list) |
| Cron jobs | In-memory (managed by `CronService`) | Runtime only |
| Skills (workspace) | `~/.nanobot/workspace/skills/{name}/SKILL.md` | Markdown |
| Skills (built-in) | `nanobot/skills/{name}/SKILL.md` | Markdown |
| Bootstrap files | `~/.nanobot/workspace/AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md`, `IDENTITY.md` | Markdown |

### Session Storage (JSONL)

Each session is stored as a `.jsonl` file. The first line is always a metadata record; subsequent lines are messages.

**Metadata record** (line 1):

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

**Message records** (lines 2+):

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

**Key design decisions**:
- **Append-only**: Messages are never modified or deleted from the list — only `last_consolidated` advances
- **JSONL format**: Each message is a single line for easy streaming and grep-ability
- **Session key**: `{channel}:{chat_id}` (e.g., `telegram:12345`, `cli:default`)

### Memory Storage

Two-layer memory system in `~/.nanobot/workspace/memory/`:

| File | Purpose | Update Pattern |
|------|---------|---------------|
| `MEMORY.md` | Long-term facts about the user | Overwritten during consolidation (LLM rewrites entire content) |
| `HISTORY.md` | Chronological event log | Append-only (new entries added at end) |

**Consolidation process**: When the session exceeds a threshold (`memory_window`), older messages are summarized by the LLM via the `save_memory` tool call, producing:
- A `history_entry`: timestamped paragraph appended to `HISTORY.md`
- A `memory_update`: full updated `MEMORY.md` content

## Core Data Models

### Event Types (`nanobot/bus/events.py`)

```python
@dataclass
class InboundMessage:
    channel: str              # "telegram", "discord", "slack", ...
    sender_id: str            # User identifier
    chat_id: str              # Chat/channel identifier
    content: str              # Message text
    timestamp: datetime       # When received
    media: list[str]          # Media URLs
    metadata: dict[str, Any]  # Channel-specific data
    session_key_override: str | None  # Thread-scoped session key

@dataclass
class OutboundMessage:
    channel: str              # Target channel
    chat_id: str              # Target chat
    content: str              # Response text
    reply_to: str | None      # Reply-to message ID
    media: list[str]          # Media attachments
    metadata: dict[str, Any]  # Channel-specific data
```

### Provider Types (`nanobot/providers/base.py`)

```python
@dataclass
class ToolCallRequest:
    id: str                       # Unique call ID
    name: str                     # Tool name
    arguments: dict[str, Any]     # Tool parameters

@dataclass
class LLMResponse:
    content: str | None           # Text response
    tool_calls: list[ToolCallRequest]  # Tool call requests
    finish_reason: str            # "stop", "tool_calls", etc.
    usage: dict[str, int]         # Token usage stats
    reasoning_content: str | None # DeepSeek-R1, Kimi reasoning
    thinking_blocks: list[dict] | None  # Anthropic extended thinking
```

### Session Model (`nanobot/session/manager.py`)

```python
@dataclass
class Session:
    key: str                           # "channel:chat_id"
    messages: list[dict[str, Any]]     # Append-only message list
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
    last_consolidated: int             # Index of last consolidated message
```

### Cron Types (`nanobot/cron/types.py`)

```python
@dataclass
class CronSchedule:
    kind: str          # "at" | "every" | "cron"
    at_ms: int | None  # Absolute timestamp (for "at")
    every_ms: int | None  # Interval (for "every")
    expr: str | None   # Cron expression (for "cron")
    tz: str | None     # Timezone (for "cron")

@dataclass
class CronJob:
    id: str
    description: str
    schedule: CronSchedule
    state: CronJobState
    payload: CronPayload
```

## Configuration Schema

The root `Config` object (`nanobot/config/schema.py`) is a Pydantic `BaseSettings` model:

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

**Config loading**: `~/.nanobot/config.json` → `json.load()` → `_migrate_config()` → `Config.model_validate()`.

**Key feature**: The `alias_generator=to_camel` on the `Base` model means both `"apiKey"` (camelCase) and `"api_key"` (snake_case) are accepted in JSON config, enabling compatibility with Claude Desktop / Cursor MCP configs.

## Tool API (OpenAI Function Calling Format)

All tools are exposed to the LLM as OpenAI-format function schemas. Example:

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

### Tool Parameters Summary

| Tool | Required Parameters | Optional Parameters |
|------|-------------------|-------------------|
| `read_file` | `path: string` | — |
| `write_file` | `path: string`, `content: string` | — |
| `edit_file` | `path: string`, `old_text: string`, `new_text: string` | — |
| `list_dir` | `path: string` | — |
| `exec` | `command: string` | — |
| `web_search` | `query: string` | — |
| `web_fetch` | `url: string` | — |
| `message_user` | `content: string` | — |
| `spawn` | `task: string` | — |
| `cron` | varies by action | `action`, `id`, `description`, `schedule` |

## External API Integrations

nanobot does not expose its own HTTP API. It connects to external services:

| Service | Protocol | Purpose |
|---------|----------|---------|
| LLM providers | HTTPS (OpenAI-compatible) | Chat completions with tool calling |
| Brave Search | HTTPS | Web search results |
| MCP servers | stdio / HTTP (SSE) | External tool execution |
| Chat platforms | WebSocket / HTTP long-poll / IMAP | Message exchange |

## Related Documentation

- [Architecture](02-architecture.md) — Component design
- [Workflows](03-workflows.md) — Data flow through the system
- [Conventions](05-conventions.md) — Code conventions

---

**Last Updated**: 2026-03-15
**Version**: 1.0
