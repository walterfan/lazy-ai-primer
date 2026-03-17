# PicoClaw 数据模型与 API

## 核心数据模型

### 配置模型（config.json）

PicoClaw 使用单一 JSON 配置文件，结构如下：

```text
{
  "agents": {
    "defaults": {
      "workspace": "~/.picoclaw/workspace",
      "restrict_to_workspace": true,
      "model_name": "gpt-5.4",
      "max_tokens": 8192,
      "temperature": 0.7,
      "max_tool_iterations": 20,
      "summarize_message_threshold": 20,
      "summarize_token_percent": 75
    }
  },
  "model_list": [...],
  "channels": {...},
  "providers": {...},
  "tools": {...}
}
```

### 模型列表（model_list）

新的统一模型配置方式，替代旧的 `providers` 字段：

```json
{
  "model_name": "claude-sonnet-4.6",
  "model": "anthropic/claude-sonnet-4.6",
  "api_key": "sk-ant-xxx",
  "api_base": "https://api.anthropic.com/v1",
  "thinking_level": "high"
}
```

`model` 字段格式：`{protocol}/{model-id}`，支持的协议前缀：

| 前缀 | 协议 | 说明 |
|------|------|------|
| `openai/` | OpenAI Compatible | 通用 OpenAI 兼容 API |
| `anthropic/` | Anthropic SDK | Anthropic Go SDK |
| `anthropic-messages/` | Anthropic Messages | 原生 Messages API |
| `azure/` | Azure OpenAI | Azure 部署 |
| `antigravity/` | Antigravity | OAuth 认证的 Google 模型 |
| `claude-cli/` | Claude CLI | 本地 Claude CLI |
| `codex-cli/` | Codex CLI | 本地 Codex CLI |
| `deepseek/` | OpenAI Compatible | DeepSeek API |
| `ollama/` | OpenAI Compatible | 本地 Ollama |
| `groq/` | OpenAI Compatible | Groq API |
| `longcat/` | OpenAI Compatible | LongCat API |
| `modelscope/` | OpenAI Compatible | ModelScope API |

### 消息模型

```go
// LLM 消息
type Message struct {
    Role       string     `json:"role"`        // system | user | assistant | tool
    Content    string     `json:"content"`
    ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
    ToolCallID string     `json:"tool_call_id,omitempty"`
}

// 工具调用
type ToolCall struct {
    ID        string         `json:"id"`
    Type      string         `json:"type"`       // "function"
    Function  *FunctionCall  `json:"function,omitempty"`
    Name      string         `json:"name,omitempty"`
    Arguments map[string]any `json:"arguments,omitempty"`
}

// LLM 响应
type LLMResponse struct {
    Content      string     `json:"content"`
    ToolCalls    []ToolCall `json:"tool_calls,omitempty"`
    FinishReason string     `json:"finish_reason"`
    Usage        *UsageInfo `json:"usage,omitempty"`
}
```

### 工具定义

```go
type ToolDefinition struct {
    Type     string                 `json:"type"`     // "function"
    Function ToolFunctionDefinition `json:"function"`
}

type ToolFunctionDefinition struct {
    Name        string         `json:"name"`
    Description string         `json:"description"`
    Parameters  map[string]any `json:"parameters"` // JSON Schema
}
```

## 持久化存储

PicoClaw 不使用数据库，所有数据以文件形式存储：

| 数据 | 格式 | 路径 | 说明 |
|------|------|------|------|
| 配置 | JSON | `~/.picoclaw/config.json` | 主配置文件 |
| 会话历史 | JSONL | `~/.picoclaw/sessions/{key}.jsonl` | 每行一条消息 |
| 长期记忆 | Markdown | `workspace/memory/MEMORY.md` | 结构化事实 |
| 历史日志 | Markdown | `workspace/memory/HISTORY.md` | 追加式事件日志 |
| 定时任务 | JSON | `~/.picoclaw/cron.json` | Cron 任务列表 |
| 凭证 | 加密文件 | `~/.picoclaw/credentials/` | ChaCha20-Poly1305 加密 |
| 技能 | 目录 | `workspace/skills/{name}/SKILL.md` | 已安装技能 |
| 状态 | JSON | `~/.picoclaw/state.json` | 运行时状态 |

### 会话存储格式（JSONL）

每个会话一个文件，每行是一条 JSON 消息：

```json
{"role":"user","content":"Hello","timestamp":"2026-03-17T10:00:00Z"}
{"role":"assistant","content":"Hi! How can I help?","timestamp":"2026-03-17T10:00:01Z"}
{"role":"user","content":"What's the weather?","timestamp":"2026-03-17T10:01:00Z"}
{"role":"assistant","content":"","tool_calls":[{"id":"tc_1","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"weather today\"}"}}],"timestamp":"2026-03-17T10:01:01Z"}
{"role":"tool","content":"Sunny, 25°C","tool_call_id":"tc_1","timestamp":"2026-03-17T10:01:02Z"}
{"role":"assistant","content":"It's sunny and 25°C today!","timestamp":"2026-03-17T10:01:03Z"}
```

## 内置工具 API

| 工具名 | 参数 | 说明 |
|--------|------|------|
| `read_file` | `path: string` | 读取文件内容 |
| `write_file` | `path: string, content: string` | 写入文件 |
| `edit_file` | `path: string, old_text: string, new_text: string` | 编辑文件（精确替换） |
| `exec` | `command: string, working_dir?: string` | 执行 Shell 命令 |
| `web_search` | `query: string, count?: int` | Web 搜索 |
| `web_fetch` | `url: string, extractMode?: string, maxChars?: int` | 抓取网页内容 |
| `cron` | `action: string, ...` | 定时任务管理 |
| `spawn` | `task: string, label?: string` | 创建子 Agent |
| `message` | `content: string, channel?: string, chat_id?: string, media?: []string` | 发送消息 |
| `send_file` | `path: string, channel?: string, chat_id?: string` | 发送文件 |
| `mcp_tool` | (动态) | MCP 服务器提供的工具 |
| `i2c` | `bus: int, addr: int, ...` | I2C 硬件操作（Linux） |
| `spi` | `bus: int, chip: int, ...` | SPI 硬件操作（Linux） |

## Web API（Launcher）

Web 后端（`web/backend/`）提供 REST API：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/gateway/status` | GET | 网关运行状态 |
| `/api/gateway/start` | POST | 启动网关 |
| `/api/gateway/stop` | POST | 停止网关 |
| `/api/channels` | GET | 渠道列表和状态 |
| `/api/models` | GET | 模型列表 |
| `/api/sessions` | GET | 会话列表 |
| `/api/sessions/:id/history` | GET | 会话历史 |
| `/api/skills` | GET | 已安装技能 |
| `/api/tools` | GET | 可用工具列表 |
| `/api/system/config` | GET/PUT | 配置读写 |
| `/api/pico/chat` | POST/WS | Pico 聊天接口 |

## 健康检查

`pkg/health/` 提供 HTTP 健康检查端点：

```
GET /health → 200 OK
GET /health/ready → 200 OK (所有服务就绪)
```
