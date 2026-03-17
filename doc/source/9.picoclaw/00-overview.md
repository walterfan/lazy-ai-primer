# PicoClaw 项目概览

## 项目简介

**PicoClaw** 是由 [Sipeed](https://sipeed.com) 发起的开源项目，用 Go 语言从零实现的超轻量级个人 AI 助手（Agent）。灵感来自 Python 版的 [NanoBot](https://github.com/HKUDS/nanobot)，但并非 fork，而是通过 AI 自举（AI-Bootstrapped）方式完成的 Go 原生重写——95% 的核心代码由 AI Agent 生成，人类做审查和微调。

**核心卖点**：在 $10 的硬件上跑 AI Agent，内存占用 <10MB，启动时间 <1 秒。

> 🦐 皮皮虾，我们走！

## 定位与边界

### 它是什么

- 超轻量级个人 AI 助手，可部署在嵌入式设备、旧手机、树莓派等低成本硬件上
- 多渠道消息网关（Telegram、Discord、飞书、钉钉、Slack、微信企业版、QQ、LINE、Matrix、IRC、WhatsApp 等 15+ 平台）
- 支持多 LLM Provider 的统一接口（OpenAI、Anthropic、DeepSeek、Gemini、Ollama、vLLM 等 20+ 提供商）
- 工具调用（Tool Use）框架：文件读写、Shell 执行、Web 搜索、定时任务、子 Agent 等
- MCP（Model Context Protocol）客户端
- 技能（Skills）市场与插件系统

### 它不是什么

- 不是 LLM 推理引擎（它调用外部 LLM API）
- 不是通用 Web 应用框架
- 不是企业级多租户 SaaS 平台
- 不是 NanoBot 或 OpenClaw 的 fork

## 关键数据

| 指标 | 数值 |
|------|------|
| 语言 | Go 1.25+ |
| 代码规模 | 421 个 Go 文件，~93,600 行 |
| 测试文件 | 162 个 `_test.go` |
| 内存占用 | <10MB（核心功能） |
| 启动时间 | <1s（0.6GHz 单核） |
| 二进制大小 | 单文件，CGO_ENABLED=0 |
| 支持架构 | x86_64, ARM64, ARM, RISC-V, MIPS, LoongArch, s390x |
| 支持 OS | Linux, macOS, Windows, FreeBSD, NetBSD |
| GitHub Stars | 12K+（一周内） |
| 贡献者 | 40+ |
| 最新版本 | v0.2.3 |
| 许可证 | MIT |

## 技术栈

| 层次 | 技术 |
|------|------|
| 语言 | Go 1.25+ |
| CLI 框架 | cobra |
| 消息总线 | 自研 `pkg/bus`（InboundMessage / OutboundMessage） |
| LLM Provider | OpenAI-compatible, Anthropic (native + messages), Azure, Ollama, vLLM, DeepSeek, Gemini, Groq, Cerebras, Mistral, 智谱, 月之暗面, 通义千问, 火山引擎, LongCat, ModelScope 等 |
| 消息渠道 | Telegram, Discord, 飞书, 钉钉, Slack, QQ, 企业微信, LINE, Matrix, IRC, OneBot, WhatsApp, MaixCAM |
| 工具系统 | 自研 `pkg/tools`（read_file, write_file, edit_file, exec, web_search, web_fetch, cron, spawn, message, send_file, mcp_tool, i2c, spi） |
| 会话管理 | JSONL 文件存储 |
| 内存/记忆 | JSONL + Markdown（MEMORY.md / HISTORY.md） |
| 定时任务 | gronx（cron 表达式解析） |
| 语音转写 | 可插拔 Transcriber 接口 |
| Web UI | Go 后端 + Vue 3 / TypeScript 前端（Vite） |
| TUI 启动器 | Bubble Tea（Go TUI 框架） |
| 系统托盘 | fyne.io/systray |
| 构建发布 | GoReleaser, GitHub Actions, Docker |
| Lint | golangci-lint |
| 硬件接口 | I2C, SPI（Linux 嵌入式设备） |

## 部署模型

- **单二进制部署**：编译为单个静态链接二进制，`CGO_ENABLED=0`
- **多平台交叉编译**：通过 GoReleaser 一次构建 15+ 平台/架构组合
- **Docker Compose**：提供标准版和完整版（含 Web UI）两种 Docker 部署方式
- **嵌入式设备**：可直接部署到 LicheeRV-Nano ($10)、NanoKVM、MaixCAM 等 RISC-V/ARM 开发板
- **旧手机**：通过 Termux 在 Android 手机上运行

## 质量目标

| 维度 | 目标 |
|------|------|
| 内存 | 核心进程 <20MB（目标 64MB RAM 设备可用） |
| 启动 | <1s（0.6GHz 单核） |
| 可用性 | 7×24 无人值守运行 |
| 安全 | Sandbox 文件系统、会话隔离、凭证加密、SSRF 防护 |
