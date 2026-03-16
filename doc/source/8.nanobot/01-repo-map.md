# Repository Map

## Directory Structure

```
nanobot/                          # Project root
├── nanobot/                      # Main Python package
│   ├── __init__.py               # Version, logo constants
│   ├── __main__.py               # `python -m nanobot` entry
│   ├── agent/                    # Core agent logic
│   │   ├── loop.py               # Agent loop (LLM ↔ tool execution)
│   │   ├── context.py            # System prompt & context builder
│   │   ├── memory.py             # Persistent memory (MEMORY.md, HISTORY.md)
│   │   ├── skills.py             # Skills loader (reads SKILL.md files)
│   │   ├── subagent.py           # Background subagent execution
│   │   └── tools/                # Built-in tool implementations
│   │       ├── base.py           # Abstract Tool base class
│   │       ├── registry.py       # Tool registry
│   │       ├── filesystem.py     # read_file, write_file, edit_file, list_dir
│   │       ├── shell.py          # exec (shell command execution)
│   │       ├── web.py            # web_search, web_fetch
│   │       ├── message.py        # message_user (send messages)
│   │       ├── spawn.py          # spawn (subagent tasks)
│   │       ├── cron.py           # cron (scheduled tasks)
│   │       └── mcp.py            # MCP server tool bridge
│   ├── bus/                      # Message routing
│   │   ├── events.py             # InboundMessage, OutboundMessage dataclasses
│   │   └── queue.py              # MessageBus (async queues)
│   ├── channels/                 # Chat platform integrations
│   │   ├── base.py               # BaseChannel ABC
│   │   ├── manager.py            # ChannelManager (init, route, dispatch)
│   │   ├── telegram.py           # Telegram channel
│   │   ├── discord.py            # Discord channel
│   │   ├── whatsapp.py           # WhatsApp channel (via Node.js bridge)
│   │   ├── slack.py              # Slack channel (Socket Mode)
│   │   ├── feishu.py             # Feishu/Lark channel (WebSocket)
│   │   ├── dingtalk.py           # DingTalk channel (Stream Mode)
│   │   ├── qq.py                 # QQ channel (botpy SDK)
│   │   ├── email.py              # Email channel (IMAP/SMTP)
│   │   ├── matrix.py             # Matrix/Element channel (optional, E2EE)
│   │   └── mochat.py             # Mochat/Claw IM channel (Socket.IO)
│   ├── cli/                      # CLI interface
│   │   └── commands.py           # Typer commands (agent, gateway, onboard, status)
│   ├── config/                   # Configuration
│   │   ├── schema.py             # Pydantic models (Config, ProvidersConfig, ChannelsConfig, etc.)
│   │   └── loader.py             # Config file loading and merging
│   ├── cron/                     # Scheduled tasks
│   │   ├── service.py            # CronService (croniter-based scheduler)
│   │   └── types.py              # CronTask dataclass
│   ├── heartbeat/                # Periodic wake-up
│   │   └── service.py            # HeartbeatService (reads HEARTBEAT.md)
│   ├── providers/                # LLM provider integrations
│   │   ├── base.py               # LLMProvider ABC, LLMResponse, ToolCallRequest
│   │   ├── registry.py           # ProviderSpec + PROVIDERS registry (single source of truth)
│   │   ├── litellm_provider.py   # LiteLLM-based provider (covers most providers)
│   │   ├── custom_provider.py    # Direct OpenAI-compatible provider (bypasses LiteLLM)
│   │   ├── openai_codex_provider.py  # OAuth-based OpenAI Codex provider
│   │   └── transcription.py      # Voice transcription (Groq Whisper)
│   ├── session/                  # Conversation management
│   │   └── manager.py            # Session, SessionManager (JSONL persistence)
│   ├── skills/                   # Bundled agent skills
│   │   ├── clawhub/SKILL.md      # ClawHub skill search/install
│   │   ├── cron/SKILL.md         # Cron task management
│   │   ├── github/SKILL.md       # GitHub operations
│   │   ├── memory/SKILL.md       # Memory management
│   │   ├── skill-creator/SKILL.md # Create new skills
│   │   ├── summarize/SKILL.md    # Text summarization
│   │   ├── tmux/SKILL.md         # Tmux session management
│   │   └── weather/SKILL.md      # Weather queries
│   ├── templates/                # Workspace templates
│   │   └── memory/               # Memory template files
│   └── utils/                    # Shared utilities
│       └── helpers.py            # ensure_dir, safe_filename, etc.
├── bridge/                       # WhatsApp Node.js bridge
├── tests/                        # Test suite (pytest)
│   ├── test_commands.py
│   ├── test_cron_service.py
│   ├── test_heartbeat_service.py
│   ├── test_context_prompt_cache.py
│   ├── test_loop_save_turn.py
│   ├── test_tool_validation.py
│   └── ... (18+ test files)
├── case/                         # Demo GIFs for README
├── pyproject.toml                # Build config, dependencies, scripts
├── Dockerfile                    # Container build
├── docker-compose.yml            # Multi-container setup
├── core_agent_lines.sh           # Line count verification script
├── README.md                     # Main documentation
├── SECURITY.md                   # Security policy
├── COMMUNICATION.md              # Community channels
└── LICENSE                       # MIT License
```

## Key Entry Points

### Application Startup

- **CLI entry**: `nanobot/cli/commands.py` → Typer `app`
  - `nanobot onboard` → initializes config and workspace
  - `nanobot agent` → starts interactive chat (single-shot or REPL)
  - `nanobot gateway` → starts long-running daemon with all enabled channels
  - `nanobot status` → shows provider/channel status
- **Package entry**: `nanobot/__main__.py` → `python -m nanobot`
- **Script entry**: `pyproject.toml` → `[project.scripts] nanobot = "nanobot.cli.commands:app"`

### Core Processing Loop

- **Agent Loop**: `nanobot/agent/loop.py:AgentLoop`
  1. Receives `InboundMessage` from `MessageBus`
  2. Loads/creates `Session` via `SessionManager`
  3. Builds context via `ContextBuilder` (system prompt + history + memory + skills)
  4. Calls `LLMProvider.chat()` with messages and tool definitions
  5. Executes tool calls from `LLMResponse` via `ToolRegistry`
  6. Publishes `OutboundMessage` back to `MessageBus`

### Message Routing

- **MessageBus**: `nanobot/bus/queue.py:MessageBus`
  - `inbound` queue: channels → agent
  - `outbound` queue: agent → channels
- **ChannelManager**: `nanobot/channels/manager.py:ChannelManager`
  - Initializes enabled channels from config
  - Dispatches outbound messages to the correct channel

### Configuration Loading

- **Schema**: `nanobot/config/schema.py:Config`
  - Root Pydantic model with nested configs: `ProvidersConfig`, `ChannelsConfig`, `AgentsConfig`, `ToolsConfig`
  - Accepts both camelCase and snake_case keys
- **Loader**: `nanobot/config/loader.py`
  - Loads from `~/.nanobot/config.json`, merges defaults

## Conventions

### Naming

#### Files
- Python: `snake_case.py`
- Test files: `test_*.py` in `tests/` directory
- Skills: `SKILL.md` in each skill subdirectory
- Config: `config.json` (camelCase keys, Pydantic aliases)

#### Classes
- `PascalCase` for all classes
- ABC suffix not used — base classes are named descriptively (`BaseChannel`, `LLMProvider`, `Tool`)

#### Functions
- `snake_case` for all functions and methods
- Async methods prefixed with standard patterns (`async def start`, `async def send`)
- Private methods prefixed with `_`

### Layering

The project follows a clean layered architecture with the message bus as the decoupling boundary:

```
CLI / Gateway
     ↓
 Channels (inbound) → MessageBus → AgentLoop → LLM Provider
                                        ↓
                                  ToolRegistry → Tools (fs, shell, web, mcp, ...)
                                        ↓
                                  MessageBus → Channels (outbound)
```

#### Presentation Layer
- **Location**: `nanobot/cli/`, `nanobot/channels/`
- **Responsibility**: User interaction (CLI prompts, chat platform APIs)
- **Dependencies**: Can depend on bus, config

#### Agent/Domain Layer
- **Location**: `nanobot/agent/`
- **Responsibility**: Core logic — LLM interaction, tool execution, context building, memory
- **Dependencies**: Bus, providers, session, config

#### Infrastructure Layer
- **Location**: `nanobot/providers/`, `nanobot/session/`, `nanobot/bus/`
- **Responsibility**: External service integration (LLM APIs, session storage, message queuing)
- **Dependencies**: Config, base abstractions

### Import Organization

```python
# Standard library
from __future__ import annotations
import asyncio
from pathlib import Path

# External dependencies
from loguru import logger
from pydantic import BaseModel

# Internal packages
from nanobot.bus.events import InboundMessage
from nanobot.agent.tools.base import Tool
```

## Related Documentation

- [Architecture](02-architecture.md) — Component design and data flow
- [Workflows](03-workflows.md) — Key business processes

---

**Last Updated**: 2026-03-15
**Version**: 1.0
