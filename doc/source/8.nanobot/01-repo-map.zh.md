# 仓库地图

## 目录结构

```
nanobot/                          # 项目根目录
├── nanobot/                      # 主 Python 包
│   ├── __init__.py               # 版本号、Logo 常量
│   ├── __main__.py               # `python -m nanobot` 入口
│   ├── agent/                    # 核心 Agent 逻辑
│   │   ├── loop.py               # Agent 循环（LLM ↔ 工具执行）
│   │   ├── context.py            # 系统提示词与上下文构建
│   │   ├── memory.py             # 持久化记忆（MEMORY.md、HISTORY.md）
│   │   ├── skills.py             # Skills 加载器（读取 SKILL.md 文件）
│   │   ├── subagent.py           # 后台 subagent 执行
│   │   └── tools/                # 内置工具实现
│   │       ├── base.py           # 抽象 Tool 基类
│   │       ├── registry.py       # 工具注册表
│   │       ├── filesystem.py     # read_file、write_file、edit_file、list_dir
│   │       ├── shell.py          # exec（Shell 命令执行）
│   │       ├── web.py            # web_search、web_fetch
│   │       ├── message.py        # message_user（发送消息）
│   │       ├── spawn.py          # spawn（subagent 任务）
│   │       ├── cron.py           # cron（定时任务）
│   │       └── mcp.py            # MCP 服务器工具桥接
│   ├── bus/                      # 消息路由
│   │   ├── events.py             # InboundMessage、OutboundMessage 数据类
│   │   └── queue.py              # MessageBus（异步队列）
│   ├── channels/                 # 聊天平台集成
│   │   ├── base.py               # BaseChannel ABC
│   │   ├── manager.py            # ChannelManager（初始化、路由、分发）
│   │   ├── telegram.py           # Telegram 频道
│   │   ├── discord.py            # Discord 频道
│   │   ├── whatsapp.py           # WhatsApp 频道（通过 Node.js 桥接）
│   │   ├── slack.py              # Slack 频道（Socket Mode）
│   │   ├── feishu.py             # 飞书/Lark 频道（WebSocket）
│   │   ├── dingtalk.py           # 钉钉频道（Stream Mode）
│   │   ├── qq.py                 # QQ 频道（botpy SDK）
│   │   ├── email.py              # 邮件频道（IMAP/SMTP）
│   │   ├── matrix.py             # Matrix/Element 频道（可选，E2EE）
│   │   └── mochat.py             # Mochat/Claw IM 频道（Socket.IO）
│   ├── cli/                      # CLI 接口
│   │   └── commands.py           # Typer 命令（agent、gateway、onboard、status）
│   ├── config/                   # 配置
│   │   ├── schema.py             # Pydantic 模型（Config、ProvidersConfig、ChannelsConfig 等）
│   │   └── loader.py             # 配置文件加载与合并
│   ├── cron/                     # 定时任务
│   │   ├── service.py            # CronService（基于 croniter 的调度器）
│   │   └── types.py              # CronTask 数据类
│   ├── heartbeat/                # 周期性唤醒
│   │   └── service.py            # HeartbeatService（读取 HEARTBEAT.md）
│   ├── providers/                # LLM 提供商集成
│   │   ├── base.py               # LLMProvider ABC、LLMResponse、ToolCallRequest
│   │   ├── registry.py           # ProviderSpec + PROVIDERS 注册表（唯一事实来源）
│   │   ├── litellm_provider.py   # 基于 LiteLLM 的提供商（覆盖大多数提供商）
│   │   ├── custom_provider.py    # 直连 OpenAI 兼容提供商（绕过 LiteLLM）
│   │   ├── openai_codex_provider.py  # 基于 OAuth 的 OpenAI Codex 提供商
│   │   └── transcription.py      # 语音转写（Groq Whisper）
│   ├── session/                  # 会话管理
│   │   └── manager.py            # Session、SessionManager（JSONL 持久化）
│   ├── skills/                   # 内置 Agent Skills
│   │   ├── clawhub/SKILL.md      # ClawHub Skill 搜索/安装
│   │   ├── cron/SKILL.md         # Cron 任务管理
│   │   ├── github/SKILL.md       # GitHub 操作
│   │   ├── memory/SKILL.md       # 记忆管理
│   │   ├── skill-creator/SKILL.md # 创建新 Skill
│   │   ├── summarize/SKILL.md    # 文本摘要
│   │   ├── tmux/SKILL.md         # Tmux 会话管理
│   │   └── weather/SKILL.md      # 天气查询
│   ├── templates/                # 工作区模板
│   │   └── memory/               # 记忆模板文件
│   └── utils/                    # 公共工具函数
│       └── helpers.py            # ensure_dir、safe_filename 等
├── bridge/                       # WhatsApp Node.js 桥接
├── tests/                        # 测试套件（pytest）
│   ├── test_commands.py
│   ├── test_cron_service.py
│   ├── test_heartbeat_service.py
│   ├── test_context_prompt_cache.py
│   ├── test_loop_save_turn.py
│   ├── test_tool_validation.py
│   └── ...（18+ 个测试文件）
├── case/                         # README 演示 GIF
├── pyproject.toml                # 构建配置、依赖、脚本
├── Dockerfile                    # 容器构建
├── docker-compose.yml            # 多容器编排
├── core_agent_lines.sh           # 代码行数验证脚本
├── README.md                     # 主文档
├── SECURITY.md                   # 安全策略
├── COMMUNICATION.md              # 社区频道
└── LICENSE                       # MIT License
```

## 关键入口

### 应用启动

- **CLI 入口**：`nanobot/cli/commands.py` → Typer `app`
  - `nanobot onboard` → 初始化配置和工作区
  - `nanobot agent` → 启动交互式聊天（单次或 REPL 模式）
  - `nanobot gateway` → 启动长驻守护进程，开启所有已启用的频道
  - `nanobot status` → 显示提供商/频道状态
- **包入口**：`nanobot/__main__.py` → `python -m nanobot`
- **脚本入口**：`pyproject.toml` → `[project.scripts] nanobot = "nanobot.cli.commands:app"`

### 核心处理循环

- **Agent Loop**：`nanobot/agent/loop.py:AgentLoop`
  1. 从 `MessageBus` 接收 `InboundMessage`
  2. 通过 `SessionManager` 加载/创建 `Session`
  3. 通过 `ContextBuilder` 构建上下文（系统提示词 + 历史记录 + 记忆 + Skills）
  4. 调用 `LLMProvider.chat()`，传入消息和工具定义
  5. 通过 `ToolRegistry` 执行 `LLMResponse` 中的工具调用
  6. 将 `OutboundMessage` 发布回 `MessageBus`

### 消息路由

- **MessageBus**：`nanobot/bus/queue.py:MessageBus`
  - `inbound` 队列：频道 → Agent
  - `outbound` 队列：Agent → 频道
- **ChannelManager**：`nanobot/channels/manager.py:ChannelManager`
  - 根据配置初始化已启用的频道
  - 将出站消息分发到对应的频道

### 配置加载

- **Schema**：`nanobot/config/schema.py:Config`
  - 根 Pydantic 模型，包含嵌套配置：`ProvidersConfig`、`ChannelsConfig`、`AgentsConfig`、`ToolsConfig`
  - 同时支持 camelCase 和 snake_case 键名
- **Loader**：`nanobot/config/loader.py`
  - 从 `~/.nanobot/config.json` 加载配置，合并默认值

## 约定

### 命名

#### 文件
- Python：`snake_case.py`
- 测试文件：`tests/` 目录下的 `test_*.py`
- Skills：每个 Skill 子目录中的 `SKILL.md`
- 配置：`config.json`（camelCase 键名，Pydantic 别名）

#### 类
- 所有类使用 `PascalCase`
- 不使用 ABC 后缀——基类采用描述性命名（`BaseChannel`、`LLMProvider`、`Tool`）

#### 函数
- 所有函数和方法使用 `snake_case`
- 异步方法遵循标准模式（`async def start`、`async def send`）
- 私有方法以 `_` 为前缀

### 分层架构

项目遵循清晰的分层架构，以消息总线作为解耦边界：

```
CLI / Gateway
     ↓
 Channels (inbound) → MessageBus → AgentLoop → LLM Provider
                                        ↓
                                  ToolRegistry → Tools (fs, shell, web, mcp, ...)
                                        ↓
                                  MessageBus → Channels (outbound)
```

#### 表示层
- **位置**：`nanobot/cli/`、`nanobot/channels/`
- **职责**：用户交互（CLI 提示、聊天平台 API）
- **依赖**：可依赖 bus、config

#### Agent/领域层
- **位置**：`nanobot/agent/`
- **职责**：核心逻辑——LLM 交互、工具执行、上下文构建、记忆管理
- **依赖**：bus、providers、session、config

#### 基础设施层
- **位置**：`nanobot/providers/`、`nanobot/session/`、`nanobot/bus/`
- **职责**：外部服务集成（LLM API、会话存储、消息队列）
- **依赖**：config、基础抽象

### Import 组织

```python
# 标准库
from __future__ import annotations
import asyncio
from pathlib import Path

# 外部依赖
from loguru import logger
from pydantic import BaseModel

# 内部包
from nanobot.bus.events import InboundMessage
from nanobot.agent.tools.base import Tool
```

## 相关文档

- [架构设计](02-architecture.md) — 组件设计与数据流
- [工作流](03-workflows.md) — 关键业务流程

---

**最后更新**：2026-03-15
**版本**：1.0
