# 测试策略

## 概述

nanobot 使用 **pytest** 和 **pytest-asyncio** 进行测试。测试套件侧重于对核心组件进行单元测试，采用轻量级的测试替身（不依赖重型 mock 框架）。所有测试都可以在无需外部服务的情况下快速运行。

## 测试基础设施

### 框架

| 工具 | 版本 | 用途 |
|------|---------|---------|
| pytest | ≥ 9.0 | 测试运行器和断言 |
| pytest-asyncio | ≥ 1.3 | 异步测试支持 |
| ruff | ≥ 0.1 | 代码检查（非测试工具，但属于 CI 质量保障的一部分） |

### 配置（`pyproject.toml`）

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- `asyncio_mode = "auto"`：所有 async 测试函数会自动被识别为异步测试（无需在每个测试上添加 `@pytest.mark.asyncio`，不过代码库中为了清晰起见仍然显式使用了该装饰器）。
- `testpaths = ["tests"]`：测试发现从 `tests/` 目录开始。

### 运行测试

```bash
# 运行所有测试
pytest -s tests/

# 运行指定测试文件
pytest -s tests/test_tool_validation.py

# 运行匹配特定模式的测试
pytest -k "test_heartbeat" tests/

# 以详细模式运行
pytest -v tests/

# 运行并生成覆盖率报告（需安装 coverage）
pytest --cov=nanobot tests/
```

## 测试组织结构

### 测试文件

| 文件 | 测试内容 | 对应组件 |
|------|-------|-----------|
| `test_tool_validation.py` | 工具参数校验、JSON Schema 检查 | `agent/tools/base.py`、`agent/tools/registry.py` |
| `test_heartbeat_service.py` | Heartbeat 幂等性、LLM 决策处理 | `heartbeat/service.py` |
| `test_cron_service.py` | 定时任务调度、触发、取消 | `cron/service.py` |
| `test_cron_commands.py` | 定时任务命令解析与分发 | `cron/` |
| `test_commands.py` | CLI 命令解析 | `cli/commands.py` |
| `test_cli_input.py` | CLI 输入处理 | `cli/commands.py` |
| `test_context_prompt_cache.py` | Prompt 缓存优化 | `agent/context.py` |
| `test_loop_save_turn.py` | Agent 循环轮次保存 | `agent/loop.py` |
| `test_message_tool.py` | 消息工具行为 | `agent/tools/message.py` |
| `test_message_tool_suppress.py` | 消息抑制逻辑 | `agent/tools/message.py` |
| `test_email_channel.py` | 邮件通道解析 | `channels/email.py` |
| `test_matrix_channel.py` | Matrix 通道集成 | `channels/matrix.py` |
| `test_feishu_post_content.py` | 飞书消息格式化 | `channels/feishu.py` |
| `test_memory_consolidation_types.py` | 记忆整合边界情况 | `agent/memory.py` |
| `test_consolidate_offset.py` | 整合偏移量追踪 | `agent/memory.py`、`session/manager.py` |
| `test_task_cancel.py` | 任务取消逻辑 | `cron/service.py` |

## 测试模式

### 测试替身

代码库使用简单的自定义测试替身，而非重型 mock 框架：

**DummyProvider** — 一个返回预配置响应的最小化 LLM provider：

```python
class DummyProvider:
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)

    async def chat(self, *args, **kwargs) -> LLMResponse:
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])
```

**SampleTool** — 一个用于测试参数校验的最小化工具：

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

### 异步测试

所有异步测试使用 `@pytest.mark.asyncio` 装饰器，并可通过 `tmp_path` 实现文件系统隔离：

```python
@pytest.mark.asyncio
async def test_heartbeat_decide_skip(tmp_path) -> None:
    provider = DummyProvider([LLMResponse(content="no tool call", tool_calls=[])])
    service = HeartbeatService(workspace=tmp_path, provider=provider, model="test")
    action, tasks = await service._decide("heartbeat content")
    assert action == "skip"
```

### 文件系统隔离

涉及文件系统操作的测试使用 pytest 的 `tmp_path` fixture：

```python
@pytest.mark.asyncio
async def test_session_persistence(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("test:123")
    session.add_message("user", "Hello")
    manager.save(session)

    # 从磁盘重新加载
    manager2 = SessionManager(tmp_path)
    session2 = manager2.get_or_create("test:123")
    assert len(session2.messages) == 1
```

### 校验测试

工具参数校验通过各种边界情况进行了充分测试：

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

## 已覆盖的测试内容

### 核心 Agent

- **工具参数校验**：JSON Schema 校验（类型、范围、枚举、嵌套对象、数组、必填字段）
- **工具注册表**：工具查找、缺失工具的错误信息、校验错误传播
- **Agent 循环**：轮次保存、会话持久化
- **上下文构建**：Prompt 缓存优化

### 记忆与会话

- **记忆整合**：LLM 驱动的整合、偏移量追踪、不同参数类型的边界情况
- **会话管理**：消息追加、历史记录检索、孤立工具结果清理

### 通道

- **Email**：IMAP 消息解析、SMTP 格式化
- **Matrix**：房间事件处理
- **飞书**：富文本消息格式化

### 服务

- **Heartbeat**：幂等启动、跳过/执行决策、LLM 工具调用处理
- **Cron**：任务调度（at/every/cron）、触发、取消、存储持久化

### CLI

- **命令解析**：Typer 命令注册
- **输入处理**：终端输入、退出命令

## 未覆盖的测试内容（已知空白）

- **端到端流程**：没有使用真实 LLM provider 运行完整 gateway 的集成测试
- **通道连接**：没有测试实际的 WebSocket/API 连接（需要外部服务）
- **LLM 响应质量**：没有测试实际的 LLM 输出（需要 API key）
- **并发访问**：没有针对多通道同时发送消息的压力测试
- **Provider 特定行为**：没有测试 LiteLLM 前缀、环境变量注入或 gateway 检测

## 添加新测试

### 命名规范

```
tests/test_{module_or_feature}.py
```

### 模板

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

### 最佳实践

1. **保持测试快速**：不发起网络请求、不使用 sleep、不做重量级 I/O
2. **使用 `tmp_path`**：绝不写入真实的文件系统路径
3. **最小化测试替身**：使用返回预配置数据的简单类
4. **聚焦单一断言**：每个测试只验证一个特定行为
5. **描述性命名**：`test_{被测内容}_{条件}_{预期结果}`

## 相关文档

- [编码规范](05-conventions.md) — 代码风格与开发实践
- [仓库结构](01-repo-map.md) — 文件位置
- [架构设计](02-architecture.md) — 组件设计

---

**最后更新**：2026-03-15
**版本**：1.0
