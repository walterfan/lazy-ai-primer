# PicoClaw 核心工作流

## 工作流 1：消息处理（Message Processing）

### 概述

用户通过任意渠道（Telegram、飞书、Discord 等）发送消息，PicoClaw 接收后路由到对应 Agent，调用 LLM 生成回复，可能执行工具调用，最终将结果返回给用户。

### 流程图

```{mermaid}
sequenceDiagram
    participant U as 用户
    participant CH as Channel
    participant BUS as MessageBus
    participant AL as AgentLoop
    participant RT as Router
    participant AI as AgentInstance
    participant LLM as LLM Provider
    participant T as Tools

    U->>CH: 发送消息
    CH->>BUS: Publish(InboundMessage)
    BUS->>AL: processMessage()
    AL->>RT: resolveMessageRoute()
    RT-->>AL: AgentInstance + Route
    AL->>AI: Sessions.Load(sessionKey)
    AL->>AI: ContextBuilder.Build()
    loop LLM 迭代循环
        AL->>LLM: FallbackChain.Chat()
        LLM-->>AL: LLMResponse
        alt 有工具调用
            AL->>T: tool.Execute()
            T-->>AL: ToolResult
        else 无工具调用
            AL-->>AL: 循环结束
        end
    end
    AL->>AI: Sessions.Save()
    AL->>BUS: Publish(OutboundMessage)
    BUS->>CH: SendMessage()
    CH->>U: 回复消息
```

### 关键代码入口

| 步骤 | 文件 | 函数 |
|------|------|------|
| 消息接收 | `pkg/channels/{platform}/{platform}.go` | `handleMessage()` |
| 消息路由 | `pkg/agent/loop.go` | `resolveMessageRoute()` |
| Agent 循环 | `pkg/agent/loop.go` | `runAgentLoop()` |
| LLM 调用 | `pkg/providers/fallback.go` | `FallbackChain.Chat()` |
| 工具执行 | `pkg/tools/toolloop.go` | `ExecuteToolCalls()` |
| 响应发送 | `pkg/channels/manager.go` | `SendMessage()` |

### 错误分支

- **Provider 失败**：FallbackChain 自动切换到下一个候选 Provider
- **工具执行失败**：返回错误信息给 LLM，LLM 决定是否重试
- **超过最大迭代**：返回默认提示消息
- **上下文过长**：触发 `maybeSummarize()` 压缩历史

## 工作流 2：Gateway 启动

### 概述

`picoclaw gateway` 命令启动完整的消息网关服务，包括 Agent 初始化、渠道连接、后台服务启动。

### 流程

```{mermaid}
flowchart TD
    A[加载 config.json] --> B[创建 LLM Provider]
    B --> C[构建 FallbackChain]
    C --> D[创建 AgentLoop]
    D --> E[注册内置工具]
    E --> F[初始化 MCP 客户端]
    F --> G[启动后台服务]
    G --> G1[CronService]
    G --> G2[HeartbeatService]
    G --> G3[HealthServer]
    G --> G4[DeviceService]
    G --> G5[MediaStore]
    G --> H[启动消息渠道]
    H --> H1[Telegram Bot]
    H --> H2[Discord Bot]
    H --> H3[飞书 Webhook]
    H --> H4[其他渠道...]
    H --> I[监听信号]
    I --> J{收到 SIGTERM/SIGINT?}
    J -->|是| K[优雅关闭]
    K --> K1[停止渠道]
    K --> K2[停止后台服务]
    K --> K3[关闭 Provider]
```

### 关键代码入口

| 步骤 | 文件 | 函数 |
|------|------|------|
| 命令入口 | `cmd/picoclaw/internal/gateway/command.go` | `NewGatewayCommand()` |
| 网关启动 | `pkg/gateway/gateway.go` | `Run()` |
| Agent 创建 | `pkg/agent/instance.go` | `NewAgentInstance()` |
| 渠道启动 | `pkg/channels/manager.go` | `StartAll()` |

## 工作流 3：子 Agent（Spawn）

### 概述

主 Agent 可以通过 `spawn` 工具创建子 Agent 异步执行耗时任务。子 Agent 有独立的会话和上下文，完成后通过消息总线报告结果。

### 流程

```{mermaid}
sequenceDiagram
    participant MA as 主 Agent
    participant SP as Spawn Tool
    participant SA as 子 Agent
    participant LLM as LLM Provider
    participant BUS as MessageBus

    MA->>SP: spawn(task="翻译文档")
    SP->>SA: 创建子 AgentInstance
    SP-->>MA: "子 Agent 已启动 (id: xxx)"
    Note over MA: 主 Agent 继续处理其他事务

    par 子 Agent 异步执行
        SA->>LLM: Chat(task context)
        LLM-->>SA: Response + ToolCalls
        SA->>SA: 执行工具...
        SA->>SA: 迭代直到完成
    end

    SA->>BUS: Publish(完成通知)
    BUS->>MA: "[Subagent 'xxx' completed]"
```

### 关键代码入口

| 步骤 | 文件 |
|------|------|
| Spawn 工具 | `pkg/tools/spawn.go` |
| 状态查询 | `pkg/tools/spawn_status.go` |
| 子 Agent 工具 | `pkg/tools/subagent.go` |

## 工作流 4：定时任务（Cron）

### 概述

用户可以通过 `cron` 工具创建定时提醒和周期性任务。CronService 在后台运行，到时间时触发 Agent 处理。

### 流程

```{mermaid}
flowchart LR
    A[用户请求创建定时任务] --> B[cron 工具]
    B --> C[CronService.Add]
    C --> D[持久化到 cron.json]
    D --> E[gronx 调度器]
    E -->|到达触发时间| F[AgentLoop.ProcessDirect]
    F --> G[Agent 处理任务]
    G --> H[通过渠道发送结果]
```

### 关键代码入口

| 步骤 | 文件 |
|------|------|
| Cron 工具 | `pkg/tools/cron.go` |
| Cron 服务 | `pkg/cron/service.go` |
| 心跳服务 | `pkg/heartbeat/service.go` |

## 工作流 5：Smart Routing（智能路由）

### 概述

当配置了 `model_routing` 时，AgentLoop 会根据消息复杂度自动选择轻量模型（快速/便宜）或重量模型（智能/贵）。

### 流程

```{mermaid}
flowchart TD
    A[收到消息] --> B{配置了 Router?}
    B -->|否| C[使用默认 Candidates]
    B -->|是| D[Router.Score 消息]
    D --> E{复杂度评分}
    E -->|低| F[使用 LightCandidates]
    E -->|高| G[使用主 Candidates]
    F --> H[调用轻量模型]
    G --> I[调用重量模型]
```

### 关键代码入口

| 步骤 | 文件 |
|------|------|
| 路由器 | `pkg/routing/router.go` |
| 特征提取 | `pkg/routing/features.go` |
| 分类器 | `pkg/routing/classifier.go` |
| 候选选择 | `pkg/agent/loop.go` → `selectCandidates()` |
