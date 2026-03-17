# PicoClaw 仓库地图

## 目录结构

```
picoclaw/
├── cmd/                          # 可执行入口
│   ├── picoclaw/                 # 主 CLI 程序
│   │   ├── main.go              # 入口：cobra 命令注册
│   │   └── internal/            # CLI 子命令实现
│   │       ├── agent/           #   picoclaw agent — 交互式 Agent
│   │       ├── auth/            #   picoclaw auth — OAuth 认证
│   │       ├── cron/            #   picoclaw cron — 定时任务管理
│   │       ├── gateway/         #   picoclaw gateway — 消息网关
│   │       ├── migrate/         #   picoclaw migrate — 配置迁移
│   │       ├── model/           #   picoclaw model — 模型管理
│   │       ├── onboard/         #   picoclaw onboard — 首次配置向导
│   │       ├── skills/          #   picoclaw skills — 技能管理
│   │       ├── status/          #   picoclaw status — 状态查看
│   │       └── version/         #   picoclaw version — 版本信息
│   └── picoclaw-launcher-tui/   # TUI 启动器（Bubble Tea）
│       ├── main.go
│       └── internal/ui/         # TUI 界面组件
├── pkg/                          # 核心库（28 个包）
│   ├── agent/                   # Agent 核心：实例、循环、上下文、记忆、思考
│   ├── auth/                    # OAuth 2.0 + PKCE 认证
│   ├── bus/                     # 消息总线（InboundMessage ↔ OutboundMessage）
│   ├── channels/                # 消息渠道适配器（15 个平台）
│   │   ├── base.go              # 基础渠道接口
│   │   ├── manager.go           # 渠道管理器
│   │   ├── telegram/            # Telegram Bot
│   │   ├── discord/             # Discord Bot
│   │   ├── feishu/              # 飞书/Lark
│   │   ├── dingtalk/            # 钉钉
│   │   ├── slack/               # Slack
│   │   ├── qq/                  # QQ
│   │   ├── wecom/               # 企业微信（Bot/App/AIBot 三种模式）
│   │   ├── whatsapp/            # WhatsApp（Bridge 模式）
│   │   ├── whatsapp_native/     # WhatsApp（原生模式）
│   │   ├── line/                # LINE
│   │   ├── matrix/              # Matrix
│   │   ├── irc/                 # IRC
│   │   ├── onebot/              # OneBot 协议
│   │   ├── maixcam/             # MaixCAM 硬件
│   │   └── pico/                # Pico 协议
│   ├── commands/                # 命令注册表
│   ├── config/                  # 配置加载、迁移、版本管理
│   ├── constants/               # 全局常量
│   ├── credential/              # 凭证加密存储
│   ├── cron/                    # 定时任务服务
│   ├── devices/                 # 硬件设备事件（IoT）
│   ├── fileutil/                # 文件工具函数
│   ├── gateway/                 # 网关服务（组装所有组件）
│   ├── health/                  # 健康检查 HTTP 服务
│   ├── heartbeat/               # 心跳服务（周期性任务）
│   ├── identity/                # Agent 身份标识
│   ├── logger/                  # 日志
│   ├── mcp/                     # MCP（Model Context Protocol）客户端
│   ├── media/                   # 媒体文件存储
│   ├── memory/                  # 记忆系统（JSONL 存储 + 迁移）
│   ├── migrate/                 # 配置迁移工具
│   ├── providers/               # LLM Provider 适配器
│   │   ├── factory.go           # Provider 工厂
│   │   ├── fallback.go          # 故障转移链
│   │   ├── anthropic/           # Anthropic SDK
│   │   ├── anthropic_messages/  # Anthropic Messages API
│   │   ├── azure/               # Azure OpenAI
│   │   ├── openai_compat/       # OpenAI 兼容协议（通用）
│   │   ├── claude_cli_provider/ # Claude CLI
│   │   ├── codex_cli_provider/  # Codex CLI
│   │   └── common/              # 公共工具
│   ├── routing/                 # 消息路由（Agent 选择、模型路由）
│   ├── session/                 # 会话管理（JSONL 后端）
│   ├── skills/                  # 技能系统（加载、安装、搜索、ClawHub 注册表）
│   ├── state/                   # 全局状态管理
│   ├── tools/                   # 工具注册表 + 内置工具
│   │   ├── registry.go          # 工具注册表
│   │   ├── shell.go             # exec 工具
│   │   ├── filesystem.go        # read_file 工具
│   │   ├── edit.go              # edit_file / write_file 工具
│   │   ├── web.go               # web_search / web_fetch 工具
│   │   ├── search_tool.go       # 搜索引擎适配
│   │   ├── message.go           # message 工具
│   │   ├── cron.go              # cron 工具
│   │   ├── spawn.go             # spawn 子 Agent 工具
│   │   ├── mcp_tool.go          # MCP 工具桥接
│   │   ├── i2c.go               # I2C 硬件接口
│   │   ├── spi.go               # SPI 硬件接口
│   │   └── send_file.go         # 文件发送工具
│   ├── utils/                   # 通用工具函数
│   └── voice/                   # 语音转写接口
├── web/                          # Web UI
│   ├── backend/                 # Go HTTP 后端（嵌入前端静态文件）
│   │   ├── main.go              # Web 服务入口
│   │   ├── api/                 # REST API 路由
│   │   ├── middleware/          # HTTP 中间件
│   │   └── model/               # 数据模型
│   └── frontend/                # Vue 3 + TypeScript 前端
│       ├── src/
│       │   ├── api/             # API 客户端
│       │   ├── components/      # UI 组件
│       │   ├── hooks/           # React-style hooks
│       │   └── i18n/            # 国际化
│       └── vite.config.ts
├── workspace/                    # 默认 Agent 工作空间模板
│   ├── AGENTS.md                # Agent 指令
│   ├── IDENTITY.md              # 身份定义
│   ├── SOUL.md                  # 人格定义
│   ├── USER.md                  # 用户画像
│   ├── memory/MEMORY.md         # 长期记忆
│   └── skills/                  # 内置技能
├── config/                       # 配置示例
│   └── config.example.json
├── docker/                       # Docker 部署
│   ├── Dockerfile               # 标准镜像
│   ├── Dockerfile.full          # 完整镜像（含 Web UI）
│   └── docker-compose.yml
├── assets/                       # 静态资源（Logo、截图、GIF）
├── docs/                         # 项目文档
│   ├── channels/                # 各渠道配置指南
│   ├── design/                  # 设计文档
│   └── migration/               # 迁移指南
├── scripts/                      # 构建脚本
├── .github/workflows/            # CI/CD（build, pr, release, nightly, docker）
├── .golangci.yaml                # Lint 配置
├── .goreleaser.yaml              # 发布配置
├── Makefile                      # 构建入口
├── go.mod / go.sum               # Go 依赖
└── README.md                     # 项目说明（多语言版本）
```

## 关键入口点

| 入口 | 文件 | 说明 |
|------|------|------|
| CLI 主入口 | `cmd/picoclaw/main.go` | cobra 命令树根节点 |
| Gateway 启动 | `pkg/gateway/gateway.go` | 组装所有服务并启动 |
| Agent 循环 | `pkg/agent/loop.go` | 核心消息处理循环 |
| Agent 实例 | `pkg/agent/instance.go` | Agent 配置与工具注册 |
| Provider 工厂 | `pkg/providers/factory.go` | LLM Provider 创建 |
| 工具注册 | `pkg/tools/registry.go` | 内置工具注册表 |
| 渠道管理 | `pkg/channels/manager.go` | 消息渠道生命周期 |
| 配置加载 | `pkg/config/config.go` | JSON 配置解析 |
| 消息总线 | `pkg/bus/bus.go` | 进程内消息路由 |

## 命名与分层约定

### 命名规范

- **包名**：小写单词，如 `agent`, `channels`, `providers`
- **文件名**：snake_case，如 `context_cache_test.go`
- **接口**：动词/名词，如 `LLMProvider`, `Tool`, `Transcriber`
- **平台特定**：`_linux.go`, `_windows.go`, `_unix.go` 后缀

### 分层模式

```
cmd/          → CLI 入口层（cobra 命令）
pkg/gateway/  → 组装层（依赖注入、服务编排）
pkg/agent/    → 业务核心层（Agent 循环、上下文构建）
pkg/tools/    → 工具层（LLM 可调用的工具）
pkg/channels/ → 适配层（外部平台对接）
pkg/providers/→ 适配层（LLM API 对接）
pkg/bus/      → 基础设施层（消息传递）
pkg/config/   → 基础设施层（配置管理）
pkg/session/  → 基础设施层（会话持久化）
```

### 设计模式

- **注册表模式**：渠道（`channels/registry.go`）、工具（`tools/registry.go`）、技能（`skills/registry.go`）均通过 `init()` 自注册
- **Provider 工厂**：`providers/factory.go` 根据 model 前缀（`openai/`, `anthropic/`, `azure/` 等）创建对应 Provider
- **故障转移链**：`providers/fallback.go` 实现多 Provider 自动切换
- **消息总线**：`bus/bus.go` 解耦渠道与 Agent，支持 InboundMessage → Agent → OutboundMessage 流转
- **插件式渠道**：每个渠道包通过 `init.go` 中的 `init()` 函数自动注册到全局 Registry
