# Project Overview

## Purpose

**nanobot** is an ultra-lightweight personal AI assistant framework. It delivers core agent functionality in ~4,000 lines of Python — 99% smaller than comparable projects like Clawdbot (430k+ lines). The project targets developers and researchers who want a clean, readable, and extensible AI agent that can connect to multiple chat platforms and LLM providers.

## Business Boundaries

### What We Do

- Provide a multi-channel AI assistant (Telegram, Discord, WhatsApp, Slack, Feishu, DingTalk, QQ, Email, Matrix, Mochat)
- Support 15+ LLM providers via a unified Provider Registry (OpenRouter, Anthropic, OpenAI, DeepSeek, Gemini, Groq, etc.)
- Execute tools on behalf of the user: file I/O, shell commands, web search/fetch, cron scheduling, and MCP servers
- Maintain conversation sessions with persistent memory and history consolidation
- Offer a skills system for extensible, pluggable agent capabilities (weather, GitHub, tmux, cron, memory, etc.)
- Support heartbeat-driven periodic tasks and scheduled cron jobs
- Provide both CLI interactive mode and gateway (long-running daemon) mode

### What We Don't Do

- Serve as a general-purpose SaaS product — nanobot is a personal assistant, not a multi-tenant service
- Provide a web UI — interaction is through CLI or chat platforms
- Train or fine-tune LLMs — it uses existing provider APIs
- Manage infrastructure or Kubernetes — deployment is a single Python process or Docker container

## Key User Roles

- **End User**: Interacts with the bot via chat platforms (Telegram, Discord, etc.) or CLI to get tasks done
- **Developer / Researcher**: Reads, modifies, or extends the codebase — adds new providers, channels, tools, or skills
- **Self-Hoster**: Deploys nanobot on their own machine via pip, Docker, or systemd service

## Core Use Cases

1. **Interactive AI Chat**
   - Actor: End User
   - Goal: Ask questions, get answers, execute tasks via natural language
   - Outcome: The agent replies with text, runs tools, and delivers results

2. **Multi-Platform Gateway**
   - Actor: End User / Self-Hoster
   - Goal: Connect the bot to Telegram, Discord, Slack, etc. and interact from any platform
   - Outcome: The gateway daemon routes messages between channels and the agent loop

3. **Automated Periodic Tasks**
   - Actor: End User
   - Goal: Schedule recurring tasks (e.g., daily weather summary, inbox scan)
   - Outcome: The heartbeat service wakes up every 30 minutes, checks `HEARTBEAT.md`, and executes pending tasks

4. **Extend with Skills & MCP**
   - Actor: Developer
   - Goal: Add new capabilities (skills via Markdown, tools via MCP servers)
   - Outcome: The agent discovers and uses new tools at runtime without core code changes

## Technology Stack

### Language & Runtime

- Primary language: **Python ≥ 3.11**
- Package manager: pip / uv
- Build system: Hatchling (`pyproject.toml`)

### Core Dependencies

- **typer** — CLI framework
- **litellm** — Unified LLM API abstraction (for non-direct providers)
- **pydantic / pydantic-settings** — Configuration schema & validation
- **httpx** — Async HTTP client
- **loguru** — Structured logging
- **rich** — Terminal rendering (Markdown, tables)
- **prompt-toolkit** — Interactive CLI input with history
- **mcp** — Model Context Protocol client

### Channel SDKs

- python-telegram-bot, slack-sdk, qq-botpy, lark-oapi, dingtalk-stream, python-socketio, websockets, matrix-nio (optional)

### Testing

- **pytest** + **pytest-asyncio** — Unit and async test support
- **ruff** — Linting (E, F, I, N, W rules)

## Deployment Model

nanobot runs as a **single-process monolith** with pluggable async channel adapters.

### Deployment Options

| Method | Command | Use Case |
|--------|---------|----------|
| Local CLI | `nanobot agent` | Interactive one-off chat |
| Local Gateway | `nanobot gateway` | Long-running daemon with chat channels |
| Docker | `docker run nanobot gateway` | Containerized deployment |
| Docker Compose | `docker compose up -d nanobot-gateway` | Multi-container orchestration |
| systemd user service | `systemctl --user start nanobot-gateway` | Linux background service |

### Configuration

- Single config file: `~/.nanobot/config.json`
- Workspace directory: `~/.nanobot/workspace/`
- Session storage: `~/.nanobot/sessions/`
- Memory: `~/.nanobot/workspace/MEMORY.md`, `HISTORY.md`

## Quality Targets

- **Code size**: ≤ 4,000 lines of core agent code (verified via `core_agent_lines.sh`)
- **Startup time**: Sub-second for CLI mode
- **Extensibility**: Adding a new LLM provider requires only 2 steps (registry entry + config field)
- **Test coverage**: Unit tests in `tests/` directory covering core functionality

## Compliance & Security

- `allowFrom` whitelist per channel — controls who can interact with the bot
- `restrictToWorkspace` flag sandboxes all file/shell tools to the workspace directory
- Session isolation per channel/chat to prevent cross-conversation data leakage
- No credentials stored in code — all secrets in `~/.nanobot/config.json`
- MIT License

## Team & Contacts

- **Organization**: HKUDS (The University of Hong Kong, Data Science Lab)
- **Repository**: [github.com/HKUDS/nanobot](https://github.com/HKUDS/nanobot)
- **Community**: Discord, WeChat, Feishu groups (see `COMMUNICATION.md`)
- **Package**: [pypi.org/project/nanobot-ai](https://pypi.org/project/nanobot-ai/)

## Related Documentation

- [Repository Map](01-repo-map.md)
- [Architecture](02-architecture.md)
- [Workflows](03-workflows.md)
- Security Policy

---

**Last Updated**: 2026-03-15
**Version**: 0.1.4.post3
