# Engineering Conventions

## Code Style

### Language & Version

- **Python ≥ 3.11** — uses `X | Y` union syntax, `match` statements, `StrEnum`, etc.
- **Line length**: 100 characters (`pyproject.toml: [tool.ruff] line-length = 100`)
- **Target version**: py311

### Linting & Formatting

nanobot uses **ruff** for linting with the following rule sets:

| Code | Rules |
|------|-------|
| E | pycodestyle errors |
| F | Pyflakes |
| I | isort (import sorting) |
| N | pep8-naming |
| W | pycodestyle warnings |

**Ignored**: `E501` (line length) — the 100-char limit is a guideline, not enforced.

Run linter:
```bash
ruff check nanobot/
```

### Import Organization

Imports follow this order (enforced by ruff's `I` rules):

```python
# 1. Future annotations (always first when used)
from __future__ import annotations

# 2. Standard library
import asyncio
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

# 3. Third-party libraries
from loguru import logger
from pydantic import BaseModel

# 4. Local imports
from nanobot.bus.events import InboundMessage
from nanobot.agent.tools.base import Tool
```

**TYPE_CHECKING guard**: Heavy imports used only for type hints go inside `if TYPE_CHECKING:` blocks to avoid circular imports and speed up module loading.

### Type Annotations

- All public function signatures must have type annotations
- Use `str | None` syntax (not `Optional[str]`)
- Use `list[str]` (not `List[str]`)
- Use `dict[str, Any]` (not `Dict[str, Any]`)

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Modules | `snake_case.py` | `litellm_provider.py` |
| Classes | `PascalCase` | `AgentLoop`, `MessageBus` |
| Functions / Methods | `snake_case` | `build_system_prompt()` |
| Private methods | `_snake_case` | `_handle_message()` |
| Constants | `UPPER_SNAKE_CASE` | `BUILTIN_SKILLS_DIR` |
| Config fields (Python) | `snake_case` | `api_key`, `allow_from` |
| Config fields (JSON) | `camelCase` | `apiKey`, `allowFrom` |
| Dataclass fields | `snake_case` | `sender_id`, `chat_id` |
| Test functions | `test_descriptive_name` | `test_validate_params_missing_required` |

### Configuration Naming

The Pydantic `Base` model uses `alias_generator=to_camel` so that:
- Python code uses `snake_case`: `config.providers.openrouter.api_key`
- JSON config accepts both: `"apiKey"` or `"api_key"`

This dual-naming is intentional — it maintains Python idiom in code while being compatible with Claude Desktop / Cursor config formats that use camelCase.

## Architecture Patterns

### Abstract Base Classes

Core abstractions use Python ABCs:
- `LLMProvider` — abstract `chat()` and `get_default_model()`
- `BaseChannel` — abstract `start()`, `stop()`, `send()`
- `Tool` — abstract `name`, `description`, `parameters`, `execute()`

### Registry Pattern

The provider system uses a declarative registry (`ProviderSpec` entries in a tuple) instead of if-elif chains. Adding a new provider means adding data, not modifying control flow.

### Message Bus (Pub/Sub)

Channels and the agent loop are decoupled through `asyncio.Queue`. This means:
- Channels don't know about the agent
- The agent doesn't know about channels
- All coordination happens through `InboundMessage` / `OutboundMessage` dataclasses

### Lazy Imports

Channel SDKs are imported lazily inside `ChannelManager._init_channels()` — only when the channel is enabled. This avoids loading heavy SDKs (telegram, discord, slack, etc.) for channels the user hasn't configured.

### Append-Only Sessions

Session messages are never modified or deleted. Consolidation advances the `last_consolidated` pointer but leaves old messages intact. This design:
- Preserves LLM prompt cache efficiency (prefixes don't change)
- Simplifies persistence (append to JSONL)
- Avoids data loss from failed consolidation

## Error Handling

### Tool Execution

The `ToolRegistry.execute()` method wraps all tool calls in try/except:
- Validation errors → return error string with `[Analyze the error above and try a different approach.]` hint
- Execution errors → return error string with the same hint
- The LLM receives error strings as tool results and can self-correct

### Provider Errors

Provider exceptions (API errors, rate limits, timeouts) are:
- Logged via `loguru`
- Propagated to the agent loop
- Communicated to the user as error messages

### Channel Errors

Channel connection failures are:
- Logged as warnings
- The channel is marked as unavailable
- Other channels continue to operate independently

## Logging

- **Library**: loguru (structured logging with colors)
- **Default output**: stderr
- **Log format**: timestamp, level, module, message
- **CLI flag**: `--logs` shows logs alongside chat output in interactive mode
- **Best practice**: Use `logger.info()` for key events, `logger.warning()` for recoverable issues, `logger.exception()` for unexpected failures

## Testing

### Framework

- **pytest** + **pytest-asyncio** (mode: `auto`)
- Tests in `tests/` directory
- Test files: `test_*.py`

### Patterns

- **Unit tests**: Test individual functions/methods with no external dependencies
- **Mocking**: Use `DummyProvider`, `SampleTool` etc. — lightweight test doubles defined in test files
- **Async tests**: Decorated with `@pytest.mark.asyncio`, use `tmp_path` fixture for temp files
- **Fixtures**: `tmp_path` (pytest built-in) for isolated filesystem tests

### Running Tests

```bash
# Run all tests
pytest -s tests/

# Run a specific test file
pytest -s tests/test_tool_validation.py

# Run with verbose output
pytest -v tests/

# Run tests matching a pattern
pytest -k "test_heartbeat" tests/
```

## Git & Contribution

### Branch Strategy

- PRs welcome against the main branch
- The codebase is intentionally small and readable

### Commit Style

- Descriptive commit messages
- No enforced commit message format

### Code Size Budget

The project maintains a **~4,000 line** code size target for core agent code. The script `core_agent_lines.sh` verifies this.

## Dependency Management

### Adding Dependencies

- Add to `pyproject.toml` under `[project.dependencies]`
- Pin major version ranges: `"typer>=0.20.0,<1.0.0"`
- Optional dependencies go in `[project.optional-dependencies]`

### Current Dependency Groups

| Group | Purpose | Example Packages |
|-------|---------|-----------------|
| Core | Always installed | typer, litellm, pydantic, httpx, loguru, rich |
| Channels | Chat platform SDKs | python-telegram-bot, slack-sdk, qq-botpy |
| Matrix | Optional Matrix support | matrix-nio, mistune, nh3 |
| Dev | Development tools | pytest, pytest-asyncio, ruff |

## Skills Convention

### Structure

Each skill lives in a directory with a `SKILL.md` file:

```
skills/
└── my-skill/
    └── SKILL.md
```

### Frontmatter

Skills use YAML frontmatter for metadata:

```yaml
---
name: weather
description: Query weather information for any location
metadata: '{"nanobot": {"requires": {"bins": ["curl"]}, "always": false}}'
---
```

### Priority

Workspace skills (`~/.nanobot/workspace/skills/`) override built-in skills (`nanobot/skills/`) when names collide.

## Related Documentation

- [Repository Map](01-repo-map.md) — File structure and naming
- [Architecture](02-architecture.md) — Design patterns
- [Testing](07-testing.md) — Detailed testing strategy

---

**Last Updated**: 2026-03-15
**Version**: 1.0
