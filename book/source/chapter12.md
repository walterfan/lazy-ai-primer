(chapter12)=
# 第十二章：Cursor — 重新定义代码编辑器

```{mermaid}
mindmap
  root((Cursor 深度解析))
    产品定位
      AI-First IDE
      基于 VS Code
      现象级增长
    核心模式
      Tab 智能补全
      Cmd-K 内联编辑
      Chat 对话模式
      Composer 多文件编辑
      Agent 自主模式
    项目级配置
      .cursorrules
      .cursor/mcp.json
      上下文管理
    MCP 集成
      工具扩展
      外部数据源
      自定义 Server
    实战技巧
      高效 Prompt
      上下文控制
      多文件协作
    与竞品对比
      vs Copilot
      vs Windsurf
      vs Augment Code
    最佳实践
      项目配置模板
      团队协作规范
      安全注意事项
```

> "Cursor 不只是一个带 AI 的编辑器，它是第一个真正以 AI 为核心设计的开发环境。"

## 12.1 Cursor 的崛起

### 12.1.1 从 VS Code 到 AI-First IDE

2023 年初，Anysphere 公司发布了 Cursor——一个基于 VS Code 的 AI 代码编辑器。短短两年内，它从一个小众工具成长为现象级产品：

- **2024 年 ARR 突破 1 亿美元**，成为增长最快的开发者工具之一
- **用户覆盖**从独立开发者到 Fortune 500 企业
- **融资超过 9 亿美元**，估值数十亿

Cursor 的成功不是偶然的。它抓住了一个关键洞察：**AI 不应该是编辑器的插件，而应该是编辑器的核心**。

```{list-table} Cursor 发展里程碑
:header-rows: 1
:widths: 20 80

* - 时间
  - 事件
* - 2023 Q1
  - Cursor 0.1 发布，基于 VS Code fork
* - 2023 Q3
  - 推出 Composer 多文件编辑模式
* - 2024 Q1
  - Agent 模式上线，支持自主执行任务
* - 2024 Q3
  - 集成 MCP 协议，支持外部工具
* - 2025 Q1
  - 推出 Background Agent（后台 Agent）
* - 2025 Q2
  - Bug Finder 功能，主动发现代码问题
```

### 12.1.2 为什么开发者选择 Cursor

```
GitHub Copilot：在你的编辑器里加了一个 AI 助手
Cursor：围绕 AI 重新设计了整个编辑器

这个区别至关重要。
```

Cursor 的核心优势：
1. **深度上下文理解**：自动索引整个代码库，理解文件间的依赖关系
2. **多模式交互**：从行级补全到项目级重构，覆盖所有场景
3. **VS Code 兼容**：继承了 VS Code 的全部生态（扩展、快捷键、设置）
4. **模型灵活性**：支持 GPT-4o、Claude 3.5 Sonnet、Gemini 等多种模型
5. **MCP 集成**：通过 MCP 协议连接外部工具和数据源

## 12.2 五大核心模式详解

### 12.2.1 Tab 补全：最自然的 AI 交互

Tab 补全是 Cursor 最基础也最高频的功能。它不只是简单的代码补全，而是基于整个项目上下文的**智能预测**。

```python
# 你正在写一个 FastAPI 路由
# 当你输入函数签名时，Cursor 会预测整个函数体

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    # 按 Tab，Cursor 自动补全：
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.model_validate(user)
```

Tab 补全的智能之处：
- **多行预测**：不只补全当前行，而是预测接下来的多行代码
- **上下文感知**：根据项目中已有的模式和风格来补全
- **跨文件理解**：知道 `User` 模型的字段、`UserResponse` 的结构
- **光标预测**：补全后自动将光标移到下一个需要编辑的位置

```{admonition} 高效使用 Tab 补全的技巧
:class: tip
1. **写好注释再写代码**：先写注释描述意图，Tab 补全会更准确
2. **写好函数签名**：类型注解越完整，补全质量越高
3. **保持代码风格一致**：Cursor 会学习你的风格
4. **善用 Escape**：不满意时按 Escape 拒绝，重新输入引导方向
```

### 12.2.2 Cmd-K：内联编辑的利器

`Cmd-K`（macOS）或 `Ctrl-K`（Windows/Linux）是 Cursor 的内联编辑模式。选中一段代码，按 Cmd-K，输入指令，AI 直接在原位修改代码。

```python
# 选中以下代码，按 Cmd-K，输入 "添加错误处理和日志记录"

# 修改前：
def process_payment(order_id: str, amount: float):
    order = db.get_order(order_id)
    payment = stripe.charge(amount, order.customer_id)
    db.update_order(order_id, status="paid", payment_id=payment.id)
    return payment

# 修改后（AI 自动生成）：
import logging
from stripe.error import StripeError

logger = logging.getLogger(__name__)

def process_payment(order_id: str, amount: float) -> Payment:
    """处理订单支付"""
    logger.info(f"Processing payment for order {order_id}, amount: {amount}")
    
    order = db.get_order(order_id)
    if not order:
        logger.error(f"Order {order_id} not found")
        raise OrderNotFoundError(f"Order {order_id} not found")
    
    if order.status == "paid":
        logger.warning(f"Order {order_id} already paid")
        raise DuplicatePaymentError(f"Order {order_id} already paid")
    
    try:
        payment = stripe.charge(amount, order.customer_id)
        logger.info(f"Payment successful: {payment.id}")
    except StripeError as e:
        logger.error(f"Payment failed for order {order_id}: {e}")
        db.update_order(order_id, status="payment_failed")
        raise PaymentFailedError(f"Payment failed: {e}") from e
    
    db.update_order(order_id, status="paid", payment_id=payment.id)
    logger.info(f"Order {order_id} updated to paid status")
    return payment
```

Cmd-K 的典型用途：
- **重构代码**："将这个类拆分为两个"
- **添加功能**："给这个函数添加缓存"
- **修复问题**："修复这个函数的并发安全问题"
- **转换风格**："将回调风格改为 async/await"
- **添加类型**："给所有参数和返回值添加类型注解"

### 12.2.3 Chat：对话式编程

Chat 模式（`Cmd-L`）打开侧边栏对话窗口，适合需要讨论和探索的场景。

```markdown
👤 用户：
@codebase 我们的认证系统是怎么实现的？有哪些安全隐患？

🤖 Cursor：
根据代码库分析，你们的认证系统实现如下：

**架构**：
- `auth/jwt_handler.py` — JWT 令牌生成和验证
- `auth/middleware.py` — 认证中间件
- `auth/oauth.py` — Google/GitHub OAuth 集成
- `models/user.py` — 用户模型

**潜在安全隐患**：
1. ⚠️ `jwt_handler.py:23` — JWT secret 从环境变量读取，但没有检查是否为空
2. ⚠️ `middleware.py:45` — 令牌过期时间设为 30 天，建议缩短到 1 小时
3. 🔴 `oauth.py:67` — OAuth state 参数未验证，存在 CSRF 风险
4. ⚠️ 缺少 rate limiting，登录接口可能被暴力破解

需要我逐个修复这些问题吗？
```

Chat 模式的关键特性：
- **`@codebase`**：引用整个代码库作为上下文
- **`@file`**：引用特定文件
- **`@folder`**：引用特定目录
- **`@web`**：搜索网络获取最新信息
- **`@docs`**：引用指定文档
- **`@git`**：引用 Git 历史

### 12.2.4 Composer：多文件编辑的革命

Composer（`Cmd-I`）是 Cursor 最强大的模式，可以同时创建和修改多个文件。这是 Cursor 与其他 AI 编辑器拉开差距的核心功能。

```markdown
👤 用户：
给项目添加一个完整的用户通知系统：
1. 支持邮件、站内信、WebSocket 实时推送三种通知方式
2. 通知模板支持变量替换
3. 用户可以配置通知偏好（哪些事件通过哪种方式通知）
4. 包含数据库迁移和 API 接口
5. 包含单元测试

🤖 Cursor Composer 同时创建/修改：
📁 新建文件：
  ├── models/notification.py          — 通知模型和模板模型
  ├── schemas/notification.py         — Pydantic schemas
  ├── services/notification_service.py — 通知发送逻辑
  ├── services/email_sender.py        — 邮件发送
  ├── services/websocket_manager.py   — WebSocket 管理
  ├── routes/notifications.py         — API 路由
  ├── migrations/003_notifications.py — 数据库迁移
  └── tests/test_notifications.py     — 测试

📝 修改文件：
  ├── main.py                         — 注册新路由和 WebSocket
  ├── models/__init__.py              — 导出新模型
  └── requirements.txt                — 添加新依赖
```

Composer 的工作流程：
1. **分析需求**：理解你要什么，分析现有代码结构
2. **制定计划**：列出需要创建和修改的文件
3. **逐文件生成**：按依赖顺序生成代码
4. **差异展示**：以 diff 形式展示所有变更
5. **一键应用**：确认后一次性应用所有变更

```{admonition} Composer 最佳实践
:class: tip
1. **需求要具体**：越具体，生成质量越高
2. **分步进行**：复杂功能分 2-3 轮 Composer 完成
3. **先审后用**：仔细审查 diff，特别是对现有文件的修改
4. **及时提交**：每轮 Composer 后 git commit，方便回滚
```

### 12.2.5 Agent 模式：自主完成复杂任务

Agent 模式是 Cursor 的最高级形态。在这个模式下，Cursor 不仅生成代码，还能**自主执行命令、运行测试、修复错误**，形成完整的开发闭环。

```markdown
👤 用户（在 Composer 中开启 Agent 模式）：
请给项目添加 Docker 支持，包括：
- 多阶段构建的 Dockerfile
- docker-compose.yml（包含 app、postgres、redis）
- 健康检查
- 确保能正常启动

🤖 Cursor Agent 自主执行：
[1/6] 分析项目结构和依赖...
[2/6] 创建 Dockerfile（多阶段构建）
[3/6] 创建 docker-compose.yml
[4/6] 创建 .dockerignore
[5/6] 运行 `docker-compose build`
  ❌ 构建失败：缺少 libpq-dev
  → 自动修复 Dockerfile，添加系统依赖
  → 重新运行 `docker-compose build` ✅
[6/6] 运行 `docker-compose up -d`
  → 等待健康检查通过...
  → 运行 `curl http://localhost:8000/health` ✅

所有服务已启动并通过健康检查。
```

Agent 模式的能力：
- **读取文件**：自动读取相关文件了解上下文
- **执行命令**：运行 shell 命令（需要用户确认）
- **安装依赖**：自动 `pip install`、`npm install`
- **运行测试**：执行测试并分析失败原因
- **自我修复**：遇到错误自动分析并修复
- **迭代直到成功**：持续尝试直到任务完成

### 12.2.6 Background Agent（2025 新功能）

Background Agent 是 Cursor 2025 年推出的重磅功能，允许 Agent 在**后台独立运行**，不阻塞你的编辑工作。

```
使用场景：
1. 你在 Agent 中提交一个任务："重构整个测试套件，从 unittest 迁移到 pytest"
2. Agent 在后台的云端沙箱中工作
3. 你继续在编辑器中做其他事情
4. Agent 完成后通知你，创建一个 PR
5. 你审查 PR，合并或要求修改
```

这本质上是一个**异步的 AI 开发者**——你分配任务，它独立完成，你审查结果。

## 12.3 项目级配置：.cursorrules

`.cursorrules` 是 Cursor 最重要的配置文件之一。它定义了 AI 在你的项目中应该遵循的规则和约定，相当于给 AI 一份"项目手册"。

### 12.3.1 完整的 .cursorrules 模板

```markdown
# Project: E-Commerce Platform API

## Tech Stack
- Python 3.12 with FastAPI
- SQLAlchemy 2.0 (async mode) with PostgreSQL
- Pydantic v2 for data validation
- Redis for caching
- Celery for background tasks
- pytest + pytest-asyncio for testing
- Alembic for database migrations

## Architecture
This is a clean architecture project:
- `app/api/` — FastAPI route handlers (thin layer, delegate to services)
- `app/services/` — Business logic (pure Python, no framework dependencies)
- `app/repositories/` — Data access layer (SQLAlchemy queries)
- `app/models/` — SQLAlchemy ORM models
- `app/schemas/` — Pydantic request/response schemas
- `app/core/` — Configuration, security, dependencies
- `tests/` — Mirror the app structure

## Code Conventions
- All functions must have type hints for parameters and return values
- Use `async def` for all route handlers and database operations
- Use dependency injection via FastAPI's `Depends()`
- Prefer raising custom exceptions over returning error dicts
- Use `structlog` for logging, not `print()` or stdlib `logging`
- Database queries go in repositories, never in route handlers
- All money amounts use `Decimal`, never `float`

## Naming Conventions
- Files: snake_case (e.g., `user_service.py`)
- Classes: PascalCase (e.g., `UserService`)
- Functions: snake_case (e.g., `get_user_by_id`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_RETRY_COUNT`)
- API routes: kebab-case (e.g., `/api/v1/user-profiles`)

## Error Handling
- Define custom exceptions in `app/core/exceptions.py`
- Use exception handlers in `app/core/exception_handlers.py`
- Always return structured error responses:
  ```json
  {"error": {"code": "USER_NOT_FOUND", "message": "...", "details": {}}}
  ```

## Testing
- Every new feature must have tests
- Use `pytest.fixture` for test setup
- Use `httpx.AsyncClient` for API tests
- Mock external services (Stripe, email, etc.)
- Test file naming: `test_{module_name}.py`
- Minimum coverage: 80% for new code

## Security Rules
- NEVER hardcode secrets, API keys, or passwords
- ALWAYS use parameterized queries (no f-strings in SQL)
- ALWAYS validate and sanitize user input
- Use bcrypt for password hashing
- JWT tokens expire in 1 hour, refresh tokens in 7 days

## Git Conventions
- Commit messages: `type(scope): description`
- Types: feat, fix, refactor, test, docs, chore
- Example: `feat(auth): add OAuth2 Google login`
```

### 12.3.2 .cursorrules 的进阶技巧

**1. 引用示例代码**

```markdown
## Code Examples

When creating a new API endpoint, follow this pattern:

```python
# app/api/v1/users.py
from fastapi import APIRouter, Depends, status
from app.schemas.user import UserCreate, UserResponse
from app.services.user_service import UserService
from app.core.dependencies import get_user_service

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    data: UserCreate,
    service: UserService = Depends(get_user_service),
):
    """Create a new user account."""
    return await service.create_user(data)
```
```

**2. 定义禁止事项**

```markdown
## DO NOT
- Do NOT use `datetime.now()`, use `datetime.utcnow()` or `datetime.now(UTC)`
- Do NOT use `requests` library, use `httpx` with async
- Do NOT write raw SQL, use SQLAlchemy ORM
- Do NOT catch bare `Exception`, always catch specific exceptions
- Do NOT use `*` imports
```

**3. 项目特定知识**

```markdown
## Domain Knowledge
- A "workspace" contains multiple "projects"
- A "project" contains multiple "tasks"
- Users have roles: "owner", "admin", "member", "viewer"
- Billing is per-workspace, not per-user
- Free tier: 3 projects, 100 tasks per project
- Pro tier: unlimited projects and tasks
```

## 12.4 MCP 集成：扩展 Cursor 的能力边界

Cursor 通过 MCP（Model Context Protocol）协议连接外部工具和数据源，大幅扩展了 AI 的能力。

### 12.4.1 配置 MCP Server

```json
// .cursor/mcp.json — 项目级 MCP 配置
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/docs"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxx"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/mydb"
      }
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

### 12.4.2 MCP 在 Cursor 中的实际应用

```markdown
👤 用户：
@github 查看最近的 PR 中有没有关于性能问题的讨论

🤖 Cursor（通过 MCP 调用 GitHub API）：
找到 3 个相关 PR：
1. PR #234 "Optimize database queries" — 讨论了 N+1 查询问题
2. PR #241 "Add Redis caching" — 讨论了缓存策略
3. PR #245 "Fix memory leak in WebSocket" — 报告了内存泄漏

👤 用户：
@postgres 查看 users 表的索引情况，有没有缺失的索引

🤖 Cursor（通过 MCP 查询数据库）：
users 表当前索引：
- PRIMARY KEY (id)
- UNIQUE INDEX (email)
- INDEX (created_at)

建议添加的索引：
- INDEX (workspace_id) — 按工作空间查询用户时会用到
- INDEX (role, workspace_id) — 按角色筛选时会用到
```

## 12.5 Cursor Rules：新一代项目规则系统

2025 年，Cursor 推出了全新的 Rules 系统，取代了旧的 `.cursorrules` 单文件方案。新系统更灵活、更强大，支持**多规则文件**、**条件触发**和**角色分离**。

### 12.5.1 Rules 系统架构

```
项目根目录/
├── .cursor/
│   ├── rules/                    ← 新版规则目录
│   │   ├── general.mdc           ← 通用规则
│   │   ├── python-style.mdc      ← Python 代码风格
│   │   ├── testing.mdc           ← 测试规范
│   │   ├── security.mdc          ← 安全规则
│   │   ├── api-design.mdc        ← API 设计规范
│   │   └── react-components.mdc  ← React 组件规范
│   └── mcp.json                  ← MCP 配置
└── .cursorrules                  ← 旧版（仍兼容）
```

### 12.5.2 Rule 文件格式（.mdc）

每个 `.mdc` 文件包含 frontmatter 元数据和规则正文：

```yaml
---
description: "Python 代码风格和最佳实践"
globs: "**/*.py"
alwaysApply: false
---

# Python Code Style Rules

## Type Hints
- ALL functions must have type hints for parameters and return values
- Use `from __future__ import annotations` for forward references
- Prefer `list[str]` over `List[str]` (Python 3.10+)

## Naming
- Variables and functions: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE
- Private methods: prefix with underscore `_method_name`

## Error Handling
- Never catch bare `Exception`, always catch specific exceptions
- Use custom exception classes defined in `app/exceptions.py`
- Always log exceptions with context:
  ```python
  try:
      result = await service.process(data)
  except ValidationError as e:
      logger.error("Validation failed", extra={"data": data, "error": str(e)})
      raise
  ```

## Async Patterns
- Use `async def` for all I/O operations
- Use `asyncio.gather()` for concurrent operations
- Never use `time.sleep()`, use `asyncio.sleep()`

## Imports
- Group imports: stdlib → third-party → local
- Use absolute imports, not relative
- No wildcard imports (`from module import *`)
```

### 12.5.3 四种触发模式

```{list-table} Rule 触发模式
:header-rows: 1
:widths: 20 30 50

* - 模式
  - 配置
  - 说明
* - **Always**
  - `alwaysApply: true`
  - 每次 AI 交互都加载，适合通用规则
* - **Glob 匹配**
  - `globs: "**/*.py"`
  - 编辑匹配文件时自动加载
* - **手动引用**
  - `alwaysApply: false`
  - 在 Chat/Composer 中用 `@rules` 手动引用
* - **Agent 自选**
  - `description` 写清楚
  - Agent 模式下根据描述自动选择相关规则
```

### 12.5.4 实用 Rules 示例集

**安全规则（security.mdc）**：

```yaml
---
description: "安全编码规范，防止常见漏洞"
globs: "**/*.py,**/*.ts,**/*.js"
alwaysApply: true
---

# Security Rules — 违反任何一条都必须修复

## CRITICAL — 绝对禁止
- NEVER hardcode secrets, API keys, passwords, or tokens in code
- NEVER use string concatenation/f-strings for SQL queries
- NEVER disable SSL verification (`verify=False`)
- NEVER use `eval()` or `exec()` with user input
- NEVER log sensitive data (passwords, tokens, PII)

## Authentication
- Passwords: bcrypt with cost factor ≥ 12
- JWT: RS256 algorithm, 1-hour expiry for access tokens
- Always validate JWT signature AND expiry AND issuer

## Input Validation
- Validate ALL user input at API boundary
- Use Pydantic models for request validation
- Sanitize file names before filesystem operations
- Limit request body size (default: 10MB)

## Dependencies
- No dependencies with known critical CVEs
- Pin all dependency versions
- Use `pip-audit` or `safety` for vulnerability scanning
```

**API 设计规则（api-design.mdc）**：

```yaml
---
description: "RESTful API 设计规范"
globs: "app/api/**/*.py,app/routes/**/*.py"
alwaysApply: false
---

# API Design Rules

## URL Conventions
- Use kebab-case: `/user-profiles` not `/userProfiles`
- Use plural nouns: `/users` not `/user`
- Nest resources: `/users/{id}/orders`
- Version prefix: `/api/v1/`

## Response Format
All responses must follow this structure:
```json
{
  "data": { ... },
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "total_pages": 5
  }
}
```

Error responses:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable message",
    "details": { "field": "email", "reason": "invalid format" }
  }
}
```

## Status Codes
- 200: Success (GET, PUT, PATCH)
- 201: Created (POST)
- 204: No Content (DELETE)
- 400: Bad Request (validation error)
- 401: Unauthorized (not authenticated)
- 403: Forbidden (not authorized)
- 404: Not Found
- 409: Conflict (duplicate resource)
- 422: Unprocessable Entity
- 429: Too Many Requests
- 500: Internal Server Error (never expose details)
```

**测试规则（testing.mdc）**：

```yaml
---
description: "测试编写规范"
globs: "tests/**/*.py"
alwaysApply: false
---

# Testing Rules

## Structure
- Test file mirrors source: `app/services/user.py` → `tests/services/test_user.py`
- Use pytest, not unittest
- Use pytest-asyncio for async tests

## Naming
- Test functions: `test_{method}_{scenario}_{expected_result}`
- Example: `test_create_user_with_duplicate_email_raises_conflict`

## Patterns
```python
# Good test structure (Arrange-Act-Assert)
async def test_create_user_success(db_session, user_factory):
    # Arrange
    user_data = UserCreate(email="test@example.com", name="Test")
    
    # Act
    user = await user_service.create_user(db_session, user_data)
    
    # Assert
    assert user.id is not None
    assert user.email == "test@example.com"
```

## Coverage
- New features: minimum 80% line coverage
- Bug fixes: must include regression test
- Use `pytest-cov` to measure coverage
```

### 12.5.5 从 .cursorrules 迁移到 Rules 系统

```bash
# 迁移步骤
mkdir -p .cursor/rules

# 将旧的 .cursorrules 按职责拆分
# 通用规则 → general.mdc
# 代码风格 → code-style.mdc
# 测试规范 → testing.mdc
# 安全规则 → security.mdc

# 旧文件可以保留（向后兼容），但建议迁移
```

```{admonition} 迁移建议
:class: tip
1. **按职责拆分**：一个 Rule 文件只管一件事
2. **善用 globs**：让规则只在相关文件中生效
3. **通用规则用 alwaysApply**：项目级约定、安全规则
4. **特定规则用 glob 触发**：代码风格、测试规范
5. **写好 description**：Agent 模式依赖描述来选择规则
```

## 12.6 自定义 Commands：可复用的 Prompt 模板

Cursor 支持自定义斜杠命令（Slash Commands），将常用的 Prompt 模式封装为可复用的命令。

### 12.6.1 创建自定义 Command

```
.cursor/
└── commands/
    ├── review.md          ← /review 命令
    ├── refactor.md        ← /refactor 命令
    ├── test.md            ← /test 命令
    ├── explain.md         ← /explain 命令
    └── document.md        ← /document 命令
```

### 12.6.2 实用 Command 示例

**代码审查命令（review.md）**：

```markdown
请审查以下代码，从这些维度评估：

## 安全性
- SQL 注入、XSS、CSRF 风险
- 硬编码的密钥或敏感信息
- 输入验证是否充分

## 代码质量
- 命名是否清晰
- 函数是否过长（>30行需要拆分）
- 是否遵循单一职责原则
- 错误处理是否完善

## 性能
- N+1 查询问题
- 不必要的内存分配
- 可以缓存的重复计算

## 可维护性
- 是否有足够的类型注解
- 是否需要添加注释
- 是否有重复代码可以提取

请对每个问题给出具体的代码位置和修复建议。
用 ✅ 标记没问题的项，用 ⚠️ 标记需要注意的项，用 🔴 标记必须修复的项。
```

**生成测试命令（test.md）**：

```markdown
为选中的代码生成完整的测试用例：

1. **正常流程测试**：覆盖主要的成功路径
2. **边界条件测试**：空值、极大值、极小值
3. **异常流程测试**：无效输入、网络错误、超时
4. **并发测试**（如适用）：竞态条件

要求：
- 使用 pytest 框架
- 使用 Arrange-Act-Assert 模式
- 测试函数命名：`test_{method}_{scenario}_{expected}`
- Mock 外部依赖（数据库、API、文件系统）
- 每个测试只验证一个行为
```

**重构命令（refactor.md）**：

```markdown
请重构选中的代码，目标：

1. **提高可读性**：清晰的命名、合理的结构
2. **减少复杂度**：拆分长函数、消除嵌套
3. **遵循 SOLID 原则**：
   - S: 单一职责
   - O: 开闭原则
   - L: 里氏替换
   - I: 接口隔离
   - D: 依赖倒置

约束：
- 保持外部接口不变（函数签名、返回类型）
- 保持现有测试通过
- 每次只做一种重构，不要同时改太多
- 解释每个重构步骤的原因
```

**文档生成命令（document.md）**：

```markdown
为选中的代码生成文档：

1. **模块级文档**：描述模块的职责和使用方式
2. **类文档**：描述类的用途、属性、使用示例
3. **函数文档**：Google 风格 docstring，包含：
   - 功能描述
   - Args（参数说明和类型）
   - Returns（返回值说明）
   - Raises（可能抛出的异常）
   - Example（使用示例）

格式要求：
```python
def function_name(param1: str, param2: int = 0) -> dict:
    """一句话描述函数功能。

    更详细的说明（如果需要）。

    Args:
        param1: 参数1的说明。
        param2: 参数2的说明。默认为 0。

    Returns:
        返回值的说明。例如：
        {"status": "ok", "data": [...]}

    Raises:
        ValueError: 当 param1 为空时。
        ConnectionError: 当数据库连接失败时。

    Example:
        >>> result = function_name("hello", 42)
        >>> print(result["status"])
        "ok"
    """
```
```

### 12.6.3 带参数的 Command

Command 可以使用 `$ARGUMENTS` 占位符接收用户输入：

```markdown
<!-- .cursor/commands/convert.md -->
将选中的代码转换为 $ARGUMENTS 风格：

保持功能不变，只改变代码风格和模式。
如果转换后需要额外的导入或依赖，请一并添加。
```

使用方式：
```
/convert async/await 风格
/convert 函数式编程风格
/convert 面向对象风格
```

```{admonition} Command 设计原则
:class: tip
1. **一个 Command 做一件事**：不要把审查、重构、测试混在一起
2. **输出格式要明确**：告诉 AI 用什么格式输出（表格、列表、代码块）
3. **约束要具体**：不要说"写好的代码"，要说"函数不超过 30 行"
4. **包含示例**：给 AI 一个输出示例，效果会好很多
5. **团队共享**：将 `.cursor/commands/` 纳入版本控制
```

## 12.7 Cursor 实战：从零构建完整项目

### 12.7.1 实战：30 分钟构建 API 服务

以下是一个真实的 Cursor 开发流程记录：

**第 1 轮 Composer（5 分钟）— 项目脚手架**

```
请创建一个 FastAPI 项目，结构如下：
- 使用 Poetry 管理依赖
- SQLAlchemy 2.0 async + PostgreSQL
- Alembic 数据库迁移
- pytest 测试框架
- Docker + docker-compose
- 包含健康检查端点
- 包含基本的项目配置（.env, .gitignore, pyproject.toml）
```

**第 2 轮 Composer（10 分钟）— 核心功能**

```
实现用户管理模块：
1. 用户注册（邮箱+密码，密码用 bcrypt 哈希）
2. 用户登录（返回 JWT access_token + refresh_token）
3. 获取当前用户信息
4. 更新用户资料
5. 包含完整的 Pydantic schemas 和错误处理
6. 遵循 .cursorrules 中的架构规范
```

**第 3 轮 Agent（10 分钟）— 运行和调试**

```
请：
1. 创建数据库迁移并执行
2. 启动 docker-compose（postgres + redis）
3. 运行应用
4. 用 httpx 测试所有 API 端点
5. 修复发现的任何问题
```

**第 4 轮 Composer（5 分钟）— 测试**

```
为用户管理模块编写完整的测试：
- API 集成测试（使用 httpx.AsyncClient）
- Service 层单元测试
- 包含正常流程和异常流程
- 使用 pytest fixtures 管理测试数据库
```

## 12.8 Cursor vs 竞品对比

```{list-table} AI 代码编辑器深度对比（2026 年）
:header-rows: 1
:widths: 18 18 18 18 14 14

* - 特性
  - Cursor
  - GitHub Copilot
  - Windsurf
  - Augment Code
  - Cline
* - 基础
  - VS Code fork
  - VS Code 插件
  - VS Code fork
  - VS Code 插件
  - VS Code 插件
* - 代码补全
  - ⭐⭐⭐⭐⭐
  - ⭐⭐⭐⭐
  - ⭐⭐⭐⭐
  - ⭐⭐⭐⭐
  - ⭐⭐⭐
* - 多文件编辑
  - ⭐⭐⭐⭐⭐
  - ⭐⭐⭐
  - ⭐⭐⭐⭐
  - ⭐⭐⭐⭐
  - ⭐⭐⭐⭐
* - Agent 模式
  - ✅ 强
  - ✅ 中
  - ✅ 中
  - ❌
  - ✅ 强
* - MCP 支持
  - ✅
  - ❌
  - ✅
  - ❌
  - ✅
* - 后台 Agent
  - ✅
  - ❌
  - ❌
  - ❌
  - ❌
* - 代码库索引
  - ✅ 全量
  - ✅ 部分
  - ✅ 全量
  - ✅ 全量
  - ❌
* - 模型选择
  - 多模型
  - GPT 系列
  - 多模型
  - 自有模型
  - 多模型
* - 价格/月
  - $20
  - $10
  - $15
  - $30
  - 免费(API费)
```

## 12.9 团队使用 Cursor 的最佳实践

### 12.9.1 团队规范

```markdown
## 团队 Cursor 使用规范

### 1. 统一配置
- 所有项目必须包含 .cursorrules 文件
- .cursorrules 纳入版本控制
- MCP 配置中不包含敏感信息（使用环境变量）

### 2. 代码审查
- AI 生成的代码与手写代码同等对待
- PR 描述中标注 AI 辅助的部分
- 安全敏感代码必须人工审查

### 3. 安全
- 使用 Cursor Business 版（数据不用于训练）
- 不在 Prompt 中包含生产环境密钥
- 定期审查 .cursor/mcp.json 的权限配置

### 4. 效率
- 鼓励使用 Agent 模式处理重复性任务
- 建立团队 Prompt 模板库
- 定期分享 Cursor 使用技巧
```

## 12.10 本章小结

Cursor 代表了代码编辑器的未来方向——**AI 不是附加功能，而是核心体验**。从 Tab 补全到 Background Agent，Cursor 提供了一个完整的 AI 编程工作流。

```{admonition} 关键要点
:class: tip
1. **五大模式**各有适用场景：Tab（行级）→ Cmd-K（块级）→ Chat（探索）→ Composer（功能级）→ Agent（项目级）
2. **.cursorrules** 是提升 AI 输出质量的关键，值得花时间精心编写
3. **MCP 集成**让 Cursor 从代码编辑器进化为开发平台
4. **Agent 模式**是 Vibe Coding 的最佳实践载体
5. **团队使用**需要统一规范，特别是安全和代码审查方面
```

在下一章中，我们将深入探索另一个现象级产品——Claude Code，它代表了一种完全不同的 AI 编程哲学：**终端优先，Agent 原生**。
