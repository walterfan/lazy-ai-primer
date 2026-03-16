# 项目概览

## 目标

**nanobot** 是一个超轻量级的个人 AI 助手框架。它用约 4,000 行 Python 代码实现了核心 agent 功能——比同类项目 Clawdbot（43 万行以上）小 99%。该项目面向开发者和研究人员，旨在提供一个简洁、可读、可扩展的 AI agent，能够对接多种聊天平台和 LLM 提供商。

## 业务边界

### 我们做什么

- 提供多渠道 AI 助手（Telegram、Discord、WhatsApp、Slack、Feishu、DingTalk、QQ、Email、Matrix、Mochat）
- 通过统一的 Provider Registry 支持 15+ 家 LLM 提供商（OpenRouter、Anthropic、OpenAI、DeepSeek、Gemini、Groq 等）
- 代替用户执行工具操作：文件读写、Shell 命令、网页搜索/抓取、定时任务调度以及 MCP 服务器
- 维护对话会话，支持持久化记忆和历史消息合并
- 提供 skills 系统，实现可扩展、可插拔的 agent 能力（天气、GitHub、tmux、cron、memory 等）
- 支持 heartbeat 驱动的周期性任务和 cron 定时任务
- 同时提供 CLI 交互模式和 gateway（长驻守护进程）模式

### 我们不做什么

- 不做通用 SaaS 产品——nanobot 是个人助手，不是多租户服务
- 不提供 Web UI——交互通过 CLI 或聊天平台进行
- 不训练或微调 LLM——仅调用现有提供商的 API
- 不管理基础设施或 Kubernetes——部署方式就是单个 Python 进程或 Docker 容器

## 关键用户角色

- **终端用户**：通过聊天平台（Telegram、Discord 等）或 CLI 与机器人交互，完成各种任务
- **开发者 / 研究人员**：阅读、修改或扩展代码库——添加新的 provider、channel、tool 或 skill
- **自托管者**：通过 pip、Docker 或 systemd 服务在自己的机器上部署 nanobot

## 核心使用场景

1. **交互式 AI 对话**
   - 参与者：终端用户
   - 目标：用自然语言提问、获取回答、执行任务
   - 结果：agent 回复文本、运行工具并返回结果

2. **多平台网关**
   - 参与者：终端用户 / 自托管者
   - 目标：将机器人接入 Telegram、Discord、Slack 等平台，从任意平台进行交互
   - 结果：gateway 守护进程在各 channel 和 agent 循环之间路由消息

3. **自动化周期性任务**
   - 参与者：终端用户
   - 目标：调度周期性任务（如每日天气摘要、收件箱扫描）
   - 结果：heartbeat 服务每 30 分钟唤醒一次，检查 `HEARTBEAT.md` 并执行待处理任务

4. **通过 Skills 和 MCP 扩展能力**
   - 参与者：开发者
   - 目标：添加新能力（通过 Markdown 定义 skill，通过 MCP 服务器定义 tool）
   - 结果：agent 在运行时自动发现并使用新工具，无需修改核心代码

## 技术栈

### 语言与运行时

- 主要语言：**Python ≥ 3.11**
- 包管理器：pip / uv
- 构建系统：Hatchling（`pyproject.toml`）

### 核心依赖

- **typer** — CLI 框架
- **litellm** — 统一 LLM API 抽象层（用于非直连 provider）
- **pydantic / pydantic-settings** — 配置 schema 与校验
- **httpx** — 异步 HTTP 客户端
- **loguru** — 结构化日志
- **rich** — 终端渲染（Markdown、表格）
- **prompt-toolkit** — 支持历史记录的交互式 CLI 输入
- **mcp** — Model Context Protocol 客户端

### Channel SDK

- python-telegram-bot、slack-sdk、qq-botpy、lark-oapi、dingtalk-stream、python-socketio、websockets、matrix-nio（可选）

### 测试

- **pytest** + **pytest-asyncio** — 单元测试与异步测试支持
- **ruff** — 代码检查（E、F、I、N、W 规则）

## 部署模型

nanobot 以**单进程单体架构**运行，搭配可插拔的异步 channel 适配器。

### 部署方式

| 方式 | 命令 | 适用场景 |
|--------|---------|----------|
| 本地 CLI | `nanobot agent` | 交互式单次对话 |
| 本地 Gateway | `nanobot gateway` | 长驻守护进程，接入聊天渠道 |
| Docker | `docker run nanobot gateway` | 容器化部署 |
| Docker Compose | `docker compose up -d nanobot-gateway` | 多容器编排 |
| systemd 用户服务 | `systemctl --user start nanobot-gateway` | Linux 后台服务 |

### 配置

- 单一配置文件：`~/.nanobot/config.json`
- 工作区目录：`~/.nanobot/workspace/`
- 会话存储：`~/.nanobot/sessions/`
- 记忆：`~/.nanobot/workspace/MEMORY.md`、`HISTORY.md`

## 质量目标

- **代码规模**：核心 agent 代码 ≤ 4,000 行（通过 `core_agent_lines.sh` 验证）
- **启动时间**：CLI 模式亚秒级启动
- **可扩展性**：添加新的 LLM provider 只需 2 步（注册 registry 条目 + 添加 config 字段）
- **测试覆盖**：`tests/` 目录下的单元测试覆盖核心功能

## 合规与安全

- 每个 channel 支持 `allowFrom` 白名单——控制谁可以与机器人交互
- `restrictToWorkspace` 标志将所有文件/Shell 工具沙箱化到工作区目录内
- 按 channel/chat 进行会话隔离，防止跨对话数据泄露
- 代码中不存储凭据——所有密钥保存在 `~/.nanobot/config.json` 中
- MIT License

## 团队与联系方式

- **组织**：HKUDS（香港大学数据科学实验室）
- **代码仓库**：[github.com/HKUDS/nanobot](https://github.com/HKUDS/nanobot)
- **社区**：Discord、微信、飞书群组（详见 `COMMUNICATION.md`）
- **包**：[pypi.org/project/nanobot-ai](https://pypi.org/project/nanobot-ai/)

## 相关文档

- [仓库结构](01-repo-map.md)
- [架构设计](02-architecture.md)
- [工作流](03-workflows.md)
- 安全策略

---

**最后更新**：2026-03-15
**版本**：0.1.4.post3
