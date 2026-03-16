# 工程规范

## 代码风格

### 语言与版本

- **Python ≥ 3.11** — 使用 `X | Y` 联合类型语法、`match` 语句、`StrEnum` 等特性。
- **行宽限制**：100 字符（`pyproject.toml: [tool.ruff] line-length = 100`）
- **目标版本**：py311

### 代码检查与格式化

nanobot 使用 **ruff** 进行代码检查，启用了以下规则集：

| Code | Rules |
|------|-------|
| E | pycodestyle errors |
| F | Pyflakes |
| I | isort（import 排序） |
| N | pep8-naming |
| W | pycodestyle warnings |

**已忽略**：`E501`（行宽限制）— 100 字符的限制只是建议，不强制执行。

运行代码检查：
```bash
ruff check nanobot/
```

### Import 组织

Import 遵循以下顺序（由 ruff 的 `I` 规则强制执行）：

```python
# 1. Future annotations（使用时始终放在最前面）
from __future__ import annotations

# 2. 标准库
import asyncio
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

# 3. 第三方库
from loguru import logger
from pydantic import BaseModel

# 4. 本地 import
from nanobot.bus.events import InboundMessage
from nanobot.agent.tools.base import Tool
```

**TYPE_CHECKING 守卫**：仅用于类型提示的重量级 import 应放在 `if TYPE_CHECKING:` 块中，以避免循环导入并加快模块加载速度。

### 类型注解

- 所有公开函数签名必须有类型注解
- 使用 `str | None` 语法（而非 `Optional[str]`）
- 使用 `list[str]`（而非 `List[str]`）
- 使用 `dict[str, Any]`（而非 `Dict[str, Any]`）

### 命名规范

| 元素 | 规范 | 示例 |
|---------|-----------|---------|
| 模块 | `snake_case.py` | `litellm_provider.py` |
| 类 | `PascalCase` | `AgentLoop`, `MessageBus` |
| 函数 / 方法 | `snake_case` | `build_system_prompt()` |
| 私有方法 | `_snake_case` | `_handle_message()` |
| 常量 | `UPPER_SNAKE_CASE` | `BUILTIN_SKILLS_DIR` |
| 配置字段（Python） | `snake_case` | `api_key`, `allow_from` |
| 配置字段（JSON） | `camelCase` | `apiKey`, `allowFrom` |
| Dataclass 字段 | `snake_case` | `sender_id`, `chat_id` |
| 测试函数 | `test_descriptive_name` | `test_validate_params_missing_required` |

### 配置命名

Pydantic 的 `Base` model 使用了 `alias_generator=to_camel`，因此：
- Python 代码中使用 `snake_case`：`config.providers.openrouter.api_key`
- JSON 配置中两种写法都接受：`"apiKey"` 或 `"api_key"`

这种双重命名是有意为之的 — 既保持了 Python 代码中的惯用风格，又兼容了 Claude Desktop / Cursor 等使用 camelCase 的配置格式。

## 架构模式

### 抽象基类

核心抽象使用 Python ABC：
- `LLMProvider` — 抽象方法 `chat()` 和 `get_default_model()`
- `BaseChannel` — 抽象方法 `start()`、`stop()`、`send()`
- `Tool` — 抽象属性 `name`、`description`、`parameters`，抽象方法 `execute()`

### 注册表模式

Provider 系统使用声明式注册表（一个包含 `ProviderSpec` 条目的 tuple），而非 if-elif 链。添加新 provider 只需添加数据，无需修改控制流。

### 消息总线（发布/订阅）

Channel 和 agent loop 通过 `asyncio.Queue` 解耦。这意味着：
- Channel 不知道 agent 的存在
- Agent 不知道 channel 的存在
- 所有协调都通过 `InboundMessage` / `OutboundMessage` dataclass 完成

### 延迟导入

Channel SDK 在 `ChannelManager._init_channels()` 内部延迟导入 — 只有当该 channel 被启用时才会加载。这样可以避免为用户未配置的 channel 加载重量级 SDK（telegram、discord、slack 等）。

### 仅追加会话

会话消息永远不会被修改或删除。Consolidation 会推进 `last_consolidated` 指针，但旧消息保持不变。这种设计：
- 保持了 LLM prompt cache 的效率（前缀不会变化）
- 简化了持久化（追加写入 JSONL）
- 避免了 consolidation 失败导致的数据丢失

## 错误处理

### 工具执行

`ToolRegistry.execute()` 方法将所有工具调用包裹在 try/except 中：
- 校验错误 → 返回错误字符串，附带 `[Analyze the error above and try a different approach.]` 提示
- 执行错误 → 返回错误字符串，附带同样的提示
- LLM 会收到错误字符串作为工具结果，并可以自行纠正

### Provider 错误

Provider 异常（API 错误、速率限制、超时）的处理方式：
- 通过 `loguru` 记录日志
- 向上传播到 agent loop
- 以错误消息的形式通知用户

### Channel 错误

Channel 连接失败的处理方式：
- 记录为 warning 级别日志
- 将该 channel 标记为不可用
- 其他 channel 继续独立运行

## 日志

- **日志库**：loguru（带颜色的结构化日志）
- **默认输出**：stderr
- **日志格式**：时间戳、级别、模块、消息
- **CLI 参数**：`--logs` 在交互模式下同时显示日志和聊天输出
- **最佳实践**：关键事件用 `logger.info()`，可恢复的问题用 `logger.warning()`，意外故障用 `logger.exception()`

## 测试

### 框架

- **pytest** + **pytest-asyncio**（mode: `auto`）
- 测试位于 `tests/` 目录
- 测试文件：`test_*.py`

### 模式

- **单元测试**：测试单个函数/方法，不依赖外部服务
- **Mock**：使用 `DummyProvider`、`SampleTool` 等轻量级测试替身，定义在测试文件中
- **异步测试**：使用 `@pytest.mark.asyncio` 装饰器，用 `tmp_path` fixture 处理临时文件
- **Fixture**：`tmp_path`（pytest 内置）用于隔离的文件系统测试

### 运行测试

```bash
# 运行所有测试
pytest -s tests/

# 运行指定测试文件
pytest -s tests/test_tool_validation.py

# 详细输出
pytest -v tests/

# 运行匹配特定模式的测试
pytest -k "test_heartbeat" tests/
```

## Git 与贡献

### 分支策略

- 欢迎向 main 分支提交 PR
- 代码库刻意保持精简和可读

### Commit 风格

- 使用描述性的 commit 消息
- 不强制要求特定的 commit 消息格式

### 代码规模预算

项目将核心 agent 代码的规模目标维持在 **约 4,000 行**。脚本 `core_agent_lines.sh` 用于验证这一点。

## 依赖管理

### 添加依赖

- 添加到 `pyproject.toml` 的 `[project.dependencies]` 下
- 锁定主版本范围：`"typer>=0.20.0,<1.0.0"`
- 可选依赖放在 `[project.optional-dependencies]` 中

### 当前依赖分组

| 分组 | 用途 | 示例包 |
|-------|---------|-----------------|
| Core | 始终安装 | typer, litellm, pydantic, httpx, loguru, rich |
| Channels | 聊天平台 SDK | python-telegram-bot, slack-sdk, qq-botpy |
| Matrix | 可选的 Matrix 支持 | matrix-nio, mistune, nh3 |
| Dev | 开发工具 | pytest, pytest-asyncio, ruff |

## Skill 规范

### 目录结构

每个 skill 位于一个包含 `SKILL.md` 文件的目录中：

```
skills/
└── my-skill/
    └── SKILL.md
```

### Frontmatter

Skill 使用 YAML frontmatter 存放元数据：

```yaml
---
name: weather
description: Query weather information for any location
metadata: '{"nanobot": {"requires": {"bins": ["curl"]}, "always": false}}'
---
```

### 优先级

工作区 skill（`~/.nanobot/workspace/skills/`）在名称冲突时会覆盖内置 skill（`nanobot/skills/`）。

## 相关文档

- [仓库地图](01-repo-map.md) — 文件结构与命名
- [架构](02-architecture.md) — 设计模式
- [测试](07-testing.md) — 详细的测试策略

---

**最后更新**：2026-03-15
**版本**：1.0
