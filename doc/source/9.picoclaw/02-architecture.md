# PicoClaw 架构

## 整体架构

```{mermaid}
flowchart TB
    subgraph Channels["消息渠道层"]
        TG[Telegram]
        DC[Discord]
        FS[飞书]
        DT[钉钉]
        SK[Slack]
        QQ[QQ]
        WC[企业微信]
        WA[WhatsApp]
        LN[LINE]
        MX[Matrix]
        IRC[IRC]
        MC[MaixCAM]
    end

    subgraph Bus["消息总线"]
        IB[InboundMessage]
        OB[OutboundMessage]
    end

    subgraph Core["Agent 核心"]
        AL[AgentLoop]
        AI[AgentInstance]
        CB[ContextBuilder]
        RT[Router]
        SM[SessionManager]
        MM[Memory]
    end

    subgraph Tools["工具层"]
        RF[read_file]
        WF[write_file]
        EF[edit_file]
        EX[exec]
        WS[web_search]
        WFE[web_fetch]
        CR[cron]
        SP[spawn]
        MSG[message]
        MCP[mcp_tool]
        HW[i2c / spi]
    end

    subgraph Providers["LLM Provider 层"]
        OAI[OpenAI Compatible]
        ANT[Anthropic Native]
        AZR[Azure OpenAI]
        OLL[Ollama / vLLM]
        FB[FallbackChain]
    end

    subgraph Services["后台服务"]
        CRON[CronService]
        HB[HeartbeatService]
        HL[HealthServer]
        DEV[DeviceService]
        MS[MediaStore]
    end

    Channels -->|InboundMessage| IB
    IB --> AL
    AL --> AI
    AI --> CB
    AI --> RT
    AI --> SM
    AI --> MM
    AL -->|Tool Calls| Tools
    AL -->|LLM Chat| FB
    FB --> OAI & ANT & AZR & OLL
    AL -->|OutboundMessage| OB
    OB --> Channels
    CRON & HB & HL & DEV & MS -.-> AL
```

## 核心组件

### 1. Gateway（网关）

`pkg/gateway/gateway.go` 是整个系统的组装点，负责：

- 加载配置（`config.Config`）
- 创建 LLM Provider 和 FallbackChain
- 初始化 AgentLoop
- 启动所有后台服务（Cron、Heartbeat、Health、Device）
- 启动所有消息渠道
- 处理优雅关闭（graceful shutdown）

```go
type services struct {
    CronService      *cron.CronService
    HeartbeatService *heartbeat.HeartbeatService
    MediaStore       media.MediaStore
    ChannelManager   *channels.Manager
    DeviceService    *devices.Service
    HealthServer     *health.Server
}
```

### 2. AgentLoop（Agent 循环）

`pkg/agent/loop.go` 是核心处理引擎，实现了完整的 Agent 循环：

```
接收消息 → 路由到 Agent → 构建上下文 → 调用 LLM → 解析工具调用 → 执行工具 → 循环直到完成 → 发送响应
```

关键方法：

| 方法 | 职责 |
|------|------|
| `Run()` | 启动消息监听循环 |
| `processMessage()` | 处理单条消息的完整流程 |
| `resolveMessageRoute()` | 根据消息内容路由到目标 Agent |
| `runAgentLoop()` | 执行 LLM ↔ Tool 迭代循环 |
| `runLLMIteration()` | 单次 LLM 调用 + 工具执行 |
| `selectCandidates()` | 选择 Provider 候选（支持 Smart Routing） |
| `maybeSummarize()` | 上下文过长时触发摘要 |

### 3. AgentInstance（Agent 实例）

`pkg/agent/instance.go` 封装了单个 Agent 的完整配置：

- 模型选择（主模型 + 备选模型列表）
- 工作空间路径
- 工具注册表
- 会话管理器
- 上下文构建器
- 温度、最大 Token、思考级别等参数
- Smart Router（轻量模型路由）

### 4. 消息总线（Message Bus）

`pkg/bus/bus.go` 实现了进程内的发布-订阅消息总线：

```go
// 入站消息（渠道 → Agent）
type InboundMessage struct {
    Channel    string
    SenderID   string
    Sender     SenderInfo
    ChatID     string
    Content    string
    Media      []string
    Peer       Peer        // direct | group | channel
    SessionKey string
    Metadata   map[string]string
}

// 出站消息（Agent → 渠道）
type OutboundMessage struct {
    Channel          string
    ChatID           string
    Content          string
    ReplyToMessageID string
}
```

### 5. Provider 层

Provider 层采用**协议优先**的架构（正在从 Vendor-based 重构为 Protocol-based）：

```{mermaid}
flowchart LR
    subgraph Factory["Provider Factory"]
        F[factory.go]
    end

    subgraph Protocols["协议适配"]
        OC[OpenAI Compatible]
        AM[Anthropic Messages]
        AS[Anthropic SDK]
        AZ[Azure OpenAI]
        AG[Antigravity/OAuth]
        CC[Claude CLI]
        CX[Codex CLI]
    end

    subgraph Fallback["故障转移"]
        FC[FallbackChain]
        EC[ErrorClassifier]
        CD[Cooldown]
    end

    F -->|"openai/*"| OC
    F -->|"anthropic-messages/*"| AM
    F -->|"anthropic/*"| AS
    F -->|"azure/*"| AZ
    F -->|"antigravity/*"| AG
    F -->|"claude-cli/*"| CC
    F -->|"codex-cli/*"| CX

    OC & AM & AS & AZ & AG --> FC
    FC --> EC --> CD
```

**故障转移机制**：
- `FallbackChain` 按优先级尝试多个 Provider
- `ErrorClassifier` 将错误分类为 auth / rate_limit / billing / timeout / overloaded / format
- `Cooldown` 对失败的 Provider 实施冷却期
- `FailoverFormat` 类型的错误不触发重试（请求本身有问题）

### 6. 渠道层

每个渠道通过 `init()` 自注册到全局 Registry：

```go
// pkg/channels/telegram/init.go
func init() {
    channels.Register("telegram", func(cfg json.RawMessage, bus *bus.MessageBus) (channels.Channel, error) {
        return NewTelegramChannel(cfg, bus)
    })
}
```

渠道管理器（`channels/Manager`）负责：
- 根据配置启动/停止渠道
- 消息分发（长消息自动分片 `split.go`）
- 错误处理和重试

### 7. 工具系统

工具通过 `ToolRegistry` 注册，每个工具实现 `Tool` 接口：

```go
type Tool interface {
    Name() string
    Description() string
    Parameters() map[string]any
    Execute(ctx context.Context, args map[string]any) (*ToolResult, error)
}
```

工具在 `AgentInstance` 创建时根据配置选择性注册。特殊工具：
- **spawn**：创建子 Agent 异步执行任务
- **mcp_tool**：桥接 MCP 服务器提供的工具
- **i2c / spi**：直接操作嵌入式硬件（Linux only）

### 8. 技能系统

技能（Skills）是可安装的能力扩展包：

```
workspace/skills/{skill-name}/
├── SKILL.md          # 技能定义（名称、描述、指令）
├── scripts/          # 可执行脚本
└── references/       # 参考文档
```

- `skills/loader.go`：从工作空间加载已安装技能
- `skills/installer.go`：从 Git 仓库安装技能
- `skills/clawhub_registry.go`：ClawHub 公共技能市场搜索

## 关键调用链

### 消息处理主流程

```
Channel.OnMessage()
  → bus.Publish(InboundMessage)
    → AgentLoop.processMessage()
      → resolveMessageRoute()          // 路由到目标 Agent
      → agent.Sessions.Load()          // 加载会话历史
      → agent.ContextBuilder.Build()   // 构建系统提示词
      → runAgentLoop()                 // 进入 LLM 循环
        → selectCandidates()           // 选择 Provider（Smart Routing）
        → runLLMIteration()            // 调用 LLM
          → FallbackChain.Chat()       // 带故障转移的 LLM 调用
          → parseToolCalls()           // 解析工具调用
          → tools.Execute()            // 执行工具
          → (循环直到无工具调用或达到最大迭代)
      → maybeSummarize()               // 可能触发上下文摘要
      → bus.Publish(OutboundMessage)   // 发送响应
```

### 上下文构建

```
ContextBuilder.Build()
  → 读取 AGENTS.md（Agent 指令）
  → 读取 SOUL.md（人格定义）
  → 读取 USER.md（用户画像）
  → 读取 TOOLS.md（工具使用说明）
  → 读取 MEMORY.md（长期记忆）
  → 加载已安装 Skills 的 SKILL.md
  → 注入运行时上下文（时间、渠道、ChatID）
  → 组装为 system prompt
```

## 跨切面关注点

### 认证

- OAuth 2.0 + PKCE 流程（`pkg/auth/`）
- 凭证加密存储（`pkg/credential/`，支持 ChaCha20-Poly1305）
- 每个渠道独立的 `allow_from` 白名单

### 日志

- 结构化日志（`pkg/logger/`）
- 可配置日志级别

### 错误处理

- Provider 层：`FailoverError` 分类 + 自动重试
- 渠道层：`pkg/channels/errors.go` 统一错误类型
- 工具层：`ToolResult` 包含 success/error 状态

### 配置管理

- JSON 配置文件（`config.json`）
- 环境变量覆盖（`caarlos0/env`）
- 配置版本迁移（`pkg/config/migration.go`）
- 运行时热重载（Provider + 配置）

### 安全

- 文件系统沙箱（`restrict_to_workspace`）
- 路径白名单（`allow_read_paths` / `allow_write_paths`）
- Shell 命令危险操作拦截
- SSRF 防护（网络工具内置黑名单）
- 会话隔离（不同用户/渠道独立会话）
