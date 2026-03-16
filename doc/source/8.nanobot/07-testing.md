# Testing Strategy

## Overview

nanobot uses **pytest** with **pytest-asyncio** for testing. The test suite focuses on unit testing of core components with lightweight test doubles (no heavy mocking frameworks). Tests are designed to run fast without external services.

## Test Infrastructure

### Framework

| Tool | Version | Purpose |
|------|---------|---------|
| pytest | ≥ 9.0 | Test runner and assertions |
| pytest-asyncio | ≥ 1.3 | Async test support |
| ruff | ≥ 0.1 | Linting (not testing, but part of CI quality) |

### Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- `asyncio_mode = "auto"`: All async test functions are automatically treated as async tests (no need for `@pytest.mark.asyncio` on each test, though it's still used explicitly in the codebase for clarity).
- `testpaths = ["tests"]`: Test discovery starts from the `tests/` directory.

### Running Tests

```bash
# Run all tests
pytest -s tests/

# Run a specific test file
pytest -s tests/test_tool_validation.py

# Run tests matching a pattern
pytest -k "test_heartbeat" tests/

# Run with verbose output
pytest -v tests/

# Run with coverage (if coverage is installed)
pytest --cov=nanobot tests/
```

## Test Organization

### Test Files

| File | Tests | Component |
|------|-------|-----------|
| `test_tool_validation.py` | Tool parameter validation, JSON Schema checks | `agent/tools/base.py`, `agent/tools/registry.py` |
| `test_heartbeat_service.py` | Heartbeat idempotency, LLM decision handling | `heartbeat/service.py` |
| `test_cron_service.py` | Cron job scheduling, firing, cancellation | `cron/service.py` |
| `test_cron_commands.py` | Cron command parsing and dispatch | `cron/` |
| `test_commands.py` | CLI command parsing | `cli/commands.py` |
| `test_cli_input.py` | CLI input handling | `cli/commands.py` |
| `test_context_prompt_cache.py` | Prompt cache optimization | `agent/context.py` |
| `test_loop_save_turn.py` | Agent loop turn saving | `agent/loop.py` |
| `test_message_tool.py` | Message tool behavior | `agent/tools/message.py` |
| `test_message_tool_suppress.py` | Message suppression logic | `agent/tools/message.py` |
| `test_email_channel.py` | Email channel parsing | `channels/email.py` |
| `test_matrix_channel.py` | Matrix channel integration | `channels/matrix.py` |
| `test_feishu_post_content.py` | Feishu message formatting | `channels/feishu.py` |
| `test_memory_consolidation_types.py` | Memory consolidation edge cases | `agent/memory.py` |
| `test_consolidate_offset.py` | Consolidation offset tracking | `agent/memory.py`, `session/manager.py` |
| `test_task_cancel.py` | Task cancellation logic | `cron/service.py` |

## Testing Patterns

### Test Doubles

The codebase uses simple, custom test doubles rather than heavy mocking:

**DummyProvider** — A minimal LLM provider that returns pre-configured responses:

```python
class DummyProvider:
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)

    async def chat(self, *args, **kwargs) -> LLMResponse:
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])
```

**SampleTool** — A minimal tool for testing parameter validation:

```python
class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"
```

### Async Testing

All async tests use the `@pytest.mark.asyncio` decorator and can use `tmp_path` for isolated filesystem:

```python
@pytest.mark.asyncio
async def test_heartbeat_decide_skip(tmp_path) -> None:
    provider = DummyProvider([LLMResponse(content="no tool call", tool_calls=[])])
    service = HeartbeatService(workspace=tmp_path, provider=provider, model="test")
    action, tasks = await service._decide("heartbeat content")
    assert action == "skip"
```

### Filesystem Isolation

Tests that touch the filesystem use pytest's `tmp_path` fixture:

```python
@pytest.mark.asyncio
async def test_session_persistence(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("test:123")
    session.add_message("user", "Hello")
    manager.save(session)

    # Reload from disk
    manager2 = SessionManager(tmp_path)
    session2 = manager2.get_or_create("test:123")
    assert len(session2.messages) == 1
```

### Validation Testing

Tool parameter validation is tested thoroughly with edge cases:

```python
def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)

def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params({
        "query": "hi", "count": 2,
        "meta": {"flags": [1, "ok"]},
    })
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)
```

## What Is Tested

### Core Agent

- **Tool parameter validation**: JSON Schema validation (types, ranges, enums, nested objects, arrays, required fields)
- **Tool registry**: Tool lookup, error messages for missing tools, validation error propagation
- **Agent loop**: Turn saving, session persistence
- **Context building**: Prompt cache optimization

### Memory & Sessions

- **Memory consolidation**: LLM-driven consolidation, offset tracking, edge cases with different argument types
- **Session management**: Message append, history retrieval, orphaned tool result cleanup

### Channels

- **Email**: IMAP message parsing, SMTP formatting
- **Matrix**: Room event handling
- **Feishu**: Post content formatting (rich text)

### Services

- **Heartbeat**: Idempotent start, skip/run decisions, LLM tool call handling
- **Cron**: Job scheduling (at/every/cron), firing, cancellation, store persistence

### CLI

- **Command parsing**: Typer command registration
- **Input handling**: Terminal input, exit commands

## What Is NOT Tested (Known Gaps)

- **End-to-end flows**: No integration tests that run the full gateway with real LLM providers
- **Channel connectivity**: No tests for actual WebSocket/API connections (would require external services)
- **LLM response quality**: No tests for actual LLM output (would require API keys)
- **Concurrent access**: No stress tests for simultaneous messages from multiple channels
- **Provider-specific behavior**: No tests for LiteLLM prefixing, env var injection, or gateway detection

## Adding New Tests

### Naming Convention

```
tests/test_{module_or_feature}.py
```

### Template

```python
"""Tests for {feature}."""

import pytest

from nanobot.module import SomeClass


def test_descriptive_behavior_name() -> None:
    """Test that SomeClass does X when Y."""
    obj = SomeClass()
    result = obj.method()
    assert result == expected


@pytest.mark.asyncio
async def test_async_behavior(tmp_path) -> None:
    """Test async operation with filesystem isolation."""
    obj = SomeClass(workspace=tmp_path)
    result = await obj.async_method()
    assert result is not None
```

### Best Practices

1. **Keep tests fast**: No network calls, no sleeping, no heavy I/O
2. **Use `tmp_path`**: Never write to real filesystem paths
3. **Minimal test doubles**: Simple classes that return pre-configured data
4. **One assertion focus**: Each test should verify one specific behavior
5. **Descriptive names**: `test_{what_is_tested}_{condition}_{expected_result}`

## Related Documentation

- [Conventions](05-conventions.md) — Code style and development practices
- [Repository Map](01-repo-map.md) — File locations
- [Architecture](02-architecture.md) — Component design

---

**Last Updated**: 2026-03-15
**Version**: 1.0
