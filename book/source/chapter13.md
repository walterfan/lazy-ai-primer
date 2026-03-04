(chapter13)=
# 第十三章：Claude Code — 终端中的 AI 软件工程师

```{mermaid}
mindmap
  root((Claude Code 深度解析))
    产品定位
      终端原生 Agent
      Agentic Coding
      Anthropic 官方
    核心能力
      代码理解与生成
      自主执行命令
      多文件编辑
      Git 操作
      测试驱动开发
    工作模式
      交互式对话
      一次性命令
      管道模式
      Headless 模式
    配置系统
      CLAUDE.md
      权限管理
      MCP 集成
      自定义命令
    高级特性
      多 Claude 并行
      子 Agent 模式
      GitHub 集成
      CI/CD 集成
    实战场景
      新项目开发
      遗留系统重构
      Bug 修复
      代码审查
    与 Cursor 对比
      终端 vs IDE
      适用场景
      互补使用
```

> "Claude Code 不是一个代码编辑器的插件，它是一个住在你终端里的 AI 软件工程师。"

## 13.1 Claude Code 的诞生与定位

### 13.1.1 为什么 Anthropic 要做终端工具

2025 年 2 月，Anthropic 发布了 Claude Code——一个运行在终端中的 AI 编程 Agent。这个选择看似反直觉：在 GUI 编辑器大行其道的时代，为什么要做一个命令行工具？

答案在于 Anthropic 对 AI 编程未来的独特理解：

```
Cursor 的哲学：AI 增强编辑器 → 人类主导，AI 辅助
Claude Code 的哲学：AI 就是开发者 → AI 主导，人类监督

Cursor：给你一个更聪明的编辑器
Claude Code：给你一个 AI 同事
```

Claude Code 的设计理念：
1. **Agent-First**：不是代码补全工具，而是能独立完成任务的 Agent
2. **终端原生**：终端是最灵活的开发环境，没有 GUI 的限制
3. **全栈能力**：读文件、写文件、执行命令、Git 操作、网络请求
4. **安全可控**：所有操作透明可见，危险操作需要确认

### 13.1.2 Claude Code 的爆发式增长

```{list-table} Claude Code 关键数据
:header-rows: 1
:widths: 30 70

* - 指标
  - 数据
* - 发布时间
  - 2025 年 2 月（研究预览版）
* - 正式发布
  - 2025 年 5 月
* - 日活跃用户
  - 数十万开发者（2025 Q4）
* - 代码生成量
  - Anthropic 内部 >80% 代码由 Claude Code 辅助完成
* - SWE-bench 成绩
  - 72.7%（2025 年 10 月，业界最高）
* - 支持平台
  - macOS, Linux, Windows (WSL)
```

一个令人震惊的事实：**Anthropic 自己的工程师大量使用 Claude Code 来开发 Claude Code**——这是一个 AI 工具用自己来改进自己的递归过程。

## 13.2 安装与基本使用

### 13.2.1 安装

```bash
# 使用 npm 全局安装
npm install -g @anthropic-ai/claude-code

# 验证安装
claude --version

# 首次运行，需要登录 Anthropic 账户
claude
```

### 13.2.2 基本交互

```bash
# 启动交互式会话
$ claude

╭─────────────────────────────────────────╮
│ Claude Code v1.x                        │
│ Model: claude-sonnet-4-20250514         │
│ Working directory: /Users/you/project   │
╰─────────────────────────────────────────╯

> 这个项目是做什么的？

Claude: 让我看看项目结构...

[Read] package.json
[Read] README.md
[Read] src/index.ts

这是一个 TypeScript 编写的 REST API 服务，使用 Express.js 框架，
主要功能包括：
1. 用户认证（JWT）
2. 博客文章 CRUD
3. 评论系统
4. 文件上传（S3）

技术栈：TypeScript + Express + Prisma + PostgreSQL + Redis
```

### 13.2.3 四种使用模式

**1. 交互式模式（最常用）**

```bash
$ claude
> 帮我修复 issue #42 中报告的分页 bug
```

**2. 一次性命令模式**

```bash
# 直接传入 prompt，执行完退出
$ claude -p "解释 src/auth/jwt.ts 中的令牌刷新逻辑"
```

**3. 管道模式**

```bash
# 将其他命令的输出传给 Claude 分析
$ cat error.log | claude -p "分析这些错误日志，找出根因"

$ git diff HEAD~5 | claude -p "审查这些代码变更，找出潜在问题"

$ pytest --tb=long 2>&1 | claude -p "分析测试失败原因并修复"
```

**4. Headless 模式（CI/CD 集成）**

```bash
# 在 CI 中使用，无需人工交互
$ claude -p "运行所有测试，如果有失败的，修复它们" \
    --allowedTools "Read,Write,Bash" \
    --max-turns 20
```

## 13.3 核心能力深度解析

### 13.3.1 代码理解：整个项目的心智模型

Claude Code 最强大的能力之一是**深度理解整个代码库**。它不只是搜索关键词，而是构建项目的"心智模型"。

```bash
> 画出这个项目的架构图，包括服务间的依赖关系

Claude:
[Read] src/ (递归读取 47 个文件)
[Read] docker-compose.yml
[Read] prisma/schema.prisma

项目架构：

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Nginx     │────→│   Express   │────→│  PostgreSQL  │
│  (反向代理)  │     │   API Server │     │  (Prisma ORM)│
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────┴─────┐ ┌────┴────┐
              │   Redis    │ │   S3    │
              │  (缓存/会话)│ │ (文件存储)│
              └───────────┘ └─────────┘

关键依赖链：
- routes/ → services/ → repositories/ → prisma client
- middleware/auth.ts → services/jwt.ts → Redis (session store)
- services/upload.ts → AWS S3 SDK
```

### 13.3.2 自主执行：从理解到行动

Claude Code 不只是建议你做什么，它**直接帮你做**：

```bash
> 项目的 Node.js 版本太旧了，请升级到 Node 20，
  并修复所有兼容性问题

Claude:
[Read] package.json — 当前 Node 16
[Read] .nvmrc — 指定 v16.20.0
[Read] Dockerfile — FROM node:16-alpine
[Read] .github/workflows/ci.yml — node-version: 16

我需要做以下修改：

1. 更新 .nvmrc
[Write] .nvmrc → v20.11.0

2. 更新 package.json engines
[Edit] package.json — "node": ">=20.0.0"

3. 更新 Dockerfile
[Edit] Dockerfile — FROM node:20-alpine

4. 更新 CI 配置
[Edit] .github/workflows/ci.yml — node-version: 20

5. 检查依赖兼容性
[Run] npm install
⚠️ 发现 3 个依赖需要更新：
  - node-sass → 已废弃，替换为 sass
  - uuid@3 → 升级到 uuid@9
  - bcrypt@5.0 → 升级到 bcrypt@5.1

[Edit] package.json — 更新依赖版本
[Run] npm install
[Run] npm run build
✅ 构建成功

6. 运行测试
[Run] npm test
⚠️ 2 个测试失败：
  - test/crypto.test.ts — crypto.createCipher 已废弃
  → 修改为 crypto.createCipheriv
  
[Edit] src/utils/crypto.ts
[Edit] test/crypto.test.ts
[Run] npm test
✅ 所有 127 个测试通过

升级完成！主要变更：
- Node.js 16 → 20
- 替换了 node-sass 为 sass
- 升级了 uuid 和 bcrypt
- 修复了 crypto API 废弃问题
```

### 13.3.3 Git 深度集成

```bash
> 帮我创建一个 PR，修复 issue #87 中报告的内存泄漏问题

Claude:
[Read] GitHub Issue #87 — "WebSocket 连接不断增长导致内存泄漏"

让我分析 WebSocket 相关代码...
[Read] src/websocket/manager.ts
[Read] src/websocket/handlers.ts

找到问题了！在 `manager.ts:45` 中，当客户端断开连接时，
事件监听器没有被正确清理。

[Run] git checkout -b fix/websocket-memory-leak

[Edit] src/websocket/manager.ts
  - 添加 disconnect 事件处理
  - 在断开时清理所有监听器
  - 添加连接池大小限制

[Edit] src/websocket/handlers.ts
  - 添加心跳检测，清理僵尸连接

[Write] tests/websocket/memory-leak.test.ts
  - 添加内存泄漏回归测试

[Run] npm test
✅ 所有测试通过

[Run] git add -A
[Run] git commit -m "fix(websocket): clean up listeners on disconnect to prevent memory leak

- Add proper cleanup in WebSocket manager disconnect handler
- Add heartbeat mechanism to detect and remove zombie connections
- Add connection pool size limit (default: 10000)
- Add regression test for memory leak scenario

Fixes #87"

[Run] git push origin fix/websocket-memory-leak

要我创建 PR 吗？

> 是的

[Run] gh pr create --title "fix(websocket): prevent memory leak on disconnect" \
  --body "## Problem\nWebSocket connections were not properly cleaned up...\n\n## Solution\n..." \
  --assignee @me

PR #92 已创建：https://github.com/org/repo/pull/92
```

## 13.4 CLAUDE.md：项目记忆文件

`CLAUDE.md` 是 Claude Code 的项目配置文件，类似于 Cursor 的 `.cursorrules`，但更强调**项目知识和记忆**。

### 13.4.1 完整的 CLAUDE.md 示例

```markdown
# CLAUDE.md

## Project Overview
This is the backend API for "TaskFlow" — a project management SaaS.
Production URL: https://api.taskflow.io
Staging URL: https://api-staging.taskflow.io

## Tech Stack
- Python 3.12, FastAPI, SQLAlchemy 2.0 (async), PostgreSQL 16
- Redis 7 for caching and rate limiting
- Celery + RabbitMQ for background tasks
- MinIO for file storage (S3-compatible)

## Development Setup
```bash
# Start dependencies
docker-compose up -d postgres redis rabbitmq minio

# Install Python dependencies
poetry install

# Run migrations
alembic upgrade head

# Start dev server
uvicorn app.main:app --reload --port 8000

# Run tests
pytest -x --tb=short
```

## Architecture Decisions
- We use the Repository pattern: routes → services → repositories
- All database operations are async
- We use Pydantic v2 discriminated unions for polymorphic responses
- Background tasks (email, notifications) go through Celery, not FastAPI BackgroundTasks
- We use structlog for structured logging

## Important Conventions
- All API responses follow: `{"data": ..., "meta": {"page": ..., "total": ...}}`
- Error responses follow: `{"error": {"code": "...", "message": "...", "details": {...}}}`
- Database migrations must be reversible (include downgrade)
- All new endpoints need OpenAPI docs and integration tests

## Known Issues
- The search endpoint is slow for >10k results (needs Elasticsearch)
- File upload has a 50MB limit (MinIO config, not code)
- OAuth refresh token rotation is not yet implemented

## Testing
- Use `pytest-asyncio` with `asyncio_mode = "auto"`
- Test database is created/destroyed per test session
- Use `factory_boy` for test data generation
- Mocking: use `unittest.mock.AsyncMock` for async functions

## Deployment
- CI/CD: GitHub Actions → Docker build → AWS ECS
- Database migrations run automatically in CI before deployment
- Feature flags managed via LaunchDarkly
```

### 13.4.2 CLAUDE.md 的层级系统

Claude Code 支持多层级的 CLAUDE.md：

```
~/.claude/CLAUDE.md          ← 全局配置（个人偏好）
~/projects/CLAUDE.md         ← 组织级配置
~/projects/myapp/CLAUDE.md   ← 项目级配置（最常用）
~/projects/myapp/src/CLAUDE.md ← 目录级配置（特定模块）
```

全局 CLAUDE.md 示例：

```markdown
# ~/.claude/CLAUDE.md

## Personal Preferences
- I prefer Python with type hints everywhere
- I like concise code with clear variable names
- Always use f-strings, not .format() or %
- I prefer pytest over unittest
- Use pathlib instead of os.path
- Commit messages in English, code comments in Chinese when needed
```

## 13.5 权限与安全模型

Claude Code 的权限系统是其安全设计的核心：

```bash
# 权限分级
┌─────────────────────────────────────────┐
│ 无需确认（自动执行）                      │
│ - 读取文件                               │
│ - 列出目录                               │
│ - 搜索代码（grep/ripgrep）               │
│ - 查看 Git 状态/日志                     │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ 需要确认（显示操作，等待 y/n）            │
│ - 写入/修改文件                          │
│ - 执行 Shell 命令                        │
│ - Git commit/push                        │
│ - 安装依赖                               │
│ - 网络请求                               │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ 始终阻止                                 │
│ - rm -rf /                               │
│ - 修改系统文件                            │
│ - 访问 ~/.ssh 等敏感目录                  │
└─────────────────────────────────────────┘
```

你可以通过 `/allowed-tools` 命令自定义权限：

```bash
> /allowed-tools

当前允许的工具：
✅ Read — 读取文件（自动）
✅ List — 列出目录（自动）
✅ Grep — 搜索代码（自动）
⚠️ Write — 写入文件（需确认）
⚠️ Bash — 执行命令（需确认）
⚠️ WebSearch — 网络搜索（需确认）

# 信任当前项目的写入操作
> /allowed-tools Write(~/projects/myapp/**)

# 信任特定命令
> /allowed-tools Bash(npm test, npm run build, pytest)
```

## 13.6 自定义 Commands：封装可复用的工作流

Claude Code 支持自定义斜杠命令，将常用的工作流封装为一键触发的命令。

### 13.6.1 Command 文件结构

```
项目根目录/
├── .claude/
│   └── commands/
│       ├── review.md        ← /project:review
│       ├── fix-issue.md     ← /project:fix-issue
│       ├── add-tests.md     ← /project:add-tests
│       ├── refactor.md      ← /project:refactor
│       └── deploy-check.md  ← /project:deploy-check
└── ~/.claude/
    └── commands/
        ├── daily-standup.md ← /user:daily-standup（个人命令）
        └── learn.md         ← /user:learn
```

命令分两级：
- **项目级**（`.claude/commands/`）：团队共享，纳入版本控制
- **用户级**（`~/.claude/commands/`）：个人使用，不入版本控制

### 13.6.2 Command 文件格式

````markdown
<!-- .claude/commands/review.md -->
---
description: "审查当前分支的所有代码变更"
---

请审查当前分支相对于 main 的所有代码变更：

1. 运行 `git diff main...HEAD` 查看所有变更
2. 逐文件分析：
   - 🔴 安全隐患（SQL 注入、硬编码密钥、XSS）
   - ⚠️ 代码质量（命名、复杂度、重复代码）
   - 💡 性能问题（N+1 查询、不必要的循环）
   - 📝 测试覆盖（新代码是否有测试）
3. 给出总体评价（1-5 分）和改进建议
4. 如果发现必须修复的问题，列出具体的修复步骤
````

### 13.6.3 带参数的 Command

使用 `$ARGUMENTS` 接收用户输入：

````markdown
<!-- .claude/commands/fix-issue.md -->
---
description: "从 GitHub Issue 自动修复 Bug"
---

请完成以下任务：

1. 运行 `gh issue view $ARGUMENTS` 查看 Issue 详情
2. 分析问题根因
3. 创建修复分支：`fix/issue-$ARGUMENTS`
4. 编写修复代码
5. 编写回归测试
6. 运行全部测试确保没有回归
7. 提交代码（commit message 引用 Issue 编号）
8. 创建 PR

每一步都要告诉我你在做什么和为什么这样做。
````

使用方式：
```bash
> /project:fix-issue 156
```

### 13.6.4 实用 Command 集合

**添加测试（add-tests.md）**：

````markdown
---
description: "为指定文件或最近修改的代码添加测试"
---

请为 $ARGUMENTS 添加完整的测试：

1. 分析代码，识别所有需要测试的路径
2. 编写测试用例：
   - 正常流程（Happy Path）
   - 边界条件（空值、极值、特殊字符）
   - 异常流程（无效输入、网络错误、权限不足）
3. 使用项目现有的测试框架和 fixtures
4. 运行测试确保全部通过
5. 检查覆盖率，确保新代码 ≥ 80%

测试命名规范：test_{函数名}_{场景}_{预期结果}
````

**部署前检查（deploy-check.md）**：

````markdown
---
description: "部署前的全面检查"
---

请执行部署前检查清单：

1. **代码质量**
   - 运行 linter（ruff/eslint）
   - 检查类型错误（mypy/tsc）
   - 检查是否有 TODO/FIXME/HACK 注释

2. **测试**
   - 运行全部测试
   - 检查测试覆盖率
   - 检查是否有跳过的测试

3. **安全**
   - 检查依赖漏洞（pip-audit/npm audit）
   - 搜索硬编码的密钥或密码
   - 检查 .env.example 是否与 .env 同步

4. **数据库**
   - 检查是否有未执行的迁移
   - 检查迁移是否可回滚

5. **配置**
   - 检查环境变量是否完整
   - 检查 Docker 配置是否最新

输出格式：每项用 ✅/⚠️/🔴 标记状态，最后给出"可以部署"或"需要修复"的结论。
````

**每日站会（个人命令，daily-standup.md）**：

````markdown
---
description: "生成每日站会报告"
---

请基于 Git 历史生成我的每日站会报告：

1. 运行 `git log --author="$(git config user.name)" --since="yesterday" --oneline`
2. 总结昨天完成的工作
3. 查看当前分支和未提交的变更，推断今天的计划
4. 检查是否有分配给我的 Issue：`gh issue list --assignee @me`

输出格式：
**昨天完成**：
- ...

**今天计划**：
- ...

**阻塞问题**：
- （如果有的话）
````

## 13.7 Hooks：自动化的事件触发器

Hooks 是 Claude Code 2025 年推出的强大特性，允许在特定事件发生时**自动执行脚本**，实现工作流自动化。

### 13.7.1 Hook 的工作原理

```
Claude Code 事件 → 匹配 Hook 规则 → 执行脚本 → 结果反馈给 Claude

支持的事件：
├── PreToolUse    — 工具调用前（可拦截/修改）
├── PostToolUse   — 工具调用后（可检查结果）
├── Notification  — Claude 发送通知时
└── Stop          — Claude 完成任务时
```

### 13.7.2 配置 Hooks

在 `.claude/settings.json` 中配置：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/pre-write-check.py \"$FILE_PATH\""
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/post-write-lint.py \"$FILE_PATH\""
          }
        ]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/log-commands.py \"$COMMAND\""
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/on-complete.py"
          }
        ]
      }
    ]
  }
}
```

### 13.7.3 实用 Hook 示例

**写入前自动检查（防止写入敏感文件）**：

```python
#!/usr/bin/env python3
# .claude/hooks/pre-write-check.py
"""在 Claude 写入文件前检查是否安全"""
import sys
import json

PROTECTED_FILES = {
    ".env", ".env.production", "secrets.yaml",
    "id_rsa", "id_ed25519", ".ssh/config",
}

PROTECTED_PATTERNS = [
    "migrations/",  # 数据库迁移需要人工审查
    "deploy/",      # 部署配置需要人工审查
]

def check_file(file_path: str) -> dict:
    # 检查是否是受保护文件
    for protected in PROTECTED_FILES:
        if file_path.endswith(protected):
            return {
                "decision": "block",
                "reason": f"🔴 受保护文件：{protected}，需要人工修改"
            }
    
    # 检查是否匹配受保护路径
    for pattern in PROTECTED_PATTERNS:
        if pattern in file_path:
            return {
                "decision": "block",
                "reason": f"⚠️ 受保护目录：{pattern}，需要人工审查"
            }
    
    return {"decision": "allow"}

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else ""
    result = check_file(file_path)
    print(json.dumps(result))
```

**写入后自动 lint**：

```python
#!/usr/bin/env python3
# .claude/hooks/post-write-lint.py
"""在 Claude 写入 Python 文件后自动运行 linter"""
import sys
import subprocess

file_path = sys.argv[1] if len(sys.argv) > 1 else ""

if file_path.endswith('.py'):
    # 运行 ruff 自动修复
    result = subprocess.run(
        ["ruff", "check", "--fix", file_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"⚠️ Lint 问题：\n{result.stdout}")
    
    # 运行 ruff format
    subprocess.run(["ruff", "format", file_path], capture_output=True)

elif file_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
    subprocess.run(["npx", "prettier", "--write", file_path], capture_output=True)
```

**任务完成时自动通知**：

```python
#!/usr/bin/env python3
# .claude/hooks/on-complete.py
"""Claude 完成任务时发送通知"""
import subprocess
import sys

# macOS 通知
subprocess.run([
    "osascript", "-e",
    'display notification "Claude Code 任务已完成" with title "Claude Code"'
])

# 或者发送到 Slack
# import requests
# requests.post(SLACK_WEBHOOK, json={"text": "Claude Code 任务已完成 ✅"})
```

### 13.7.4 Hook 最佳实践

```{admonition} Hook 设计原则
:class: tip
1. **快速执行**：Hook 脚本应在几秒内完成，不要阻塞 Claude 的工作流
2. **幂等性**：同一个 Hook 多次执行应该产生相同结果
3. **优雅失败**：Hook 失败不应该中断 Claude 的工作
4. **日志记录**：记录 Hook 的执行情况，方便调试
5. **版本控制**：将 Hook 脚本纳入 `.claude/hooks/` 目录，团队共享
```

## 13.8 CLAUDE.md 高级编写技巧

CLAUDE.md 的质量直接决定了 Claude Code 的工作效果。以下是经过实践验证的高级技巧。

### 13.8.1 结构化模板

```markdown
# CLAUDE.md

## 项目概述
[一段话描述项目是什么、做什么、给谁用]

## 技术栈
[列出所有技术选型，版本号很重要]

## 开发环境
[完整的启动命令，Claude 可以直接执行]

## 架构决策
[为什么选择这个架构？有什么约束？]
[这部分帮助 Claude 理解"为什么"，而不只是"是什么"]

## 代码规范
[命名、风格、模式——越具体越好]

## 常见陷阱
[项目中容易踩的坑，提前告诉 Claude]

## 当前状态
[正在进行的工作、已知问题、技术债务]
```

### 13.8.2 写好 CLAUDE.md 的关键原则

**1. 写"为什么"而不只是"是什么"**

```markdown
# ❌ 不好的写法
## 架构
使用微服务架构。

# ✅ 好的写法
## 架构
使用微服务架构，因为：
1. 团队有 5 个小组，每组负责独立的业务域
2. 不同服务有不同的扩展需求（搜索服务需要更多 CPU）
3. 我们需要独立部署能力（每周发布 20+ 次）

但要注意：服务间通信使用 gRPC 而非 REST，因为内部调用需要低延迟。
```

**2. 包含"不要做什么"**

```markdown
## 常见陷阱 — 请特别注意
- ❌ 不要直接修改 `generated/` 目录下的文件，它们是自动生成的
- ❌ 不要在 service 层直接导入 FastAPI 的类（保持框架无关）
- ❌ 不要用 `datetime.now()`，统一用 `datetime.now(UTC)`
- ❌ 不要在测试中使用真实的外部 API，必须 mock
- ❌ 不要手动修改 Alembic 迁移文件的 revision ID
```

**3. 给出可执行的命令**

````markdown
## 常用命令
```bash
# 运行所有测试
pytest -x --tb=short

# 只运行某个模块的测试
pytest tests/services/test_user.py -v

# 生成数据库迁移
alembic revision --autogenerate -m "description"

# 运行迁移
alembic upgrade head

# 启动开发服务器
uvicorn app.main:app --reload --port 8000

# 代码格式化
ruff check --fix . && ruff format .
```
````

## 13.9 高级特性

### 13.9.1 多 Claude 并行工作

```bash
# 终端 1：Claude 在修复 Bug
$ claude
> 修复 issue #42 的分页问题

# 终端 2：另一个 Claude 在写文档
$ claude
> 为所有 API 端点生成 OpenAPI 文档

# 终端 3：又一个 Claude 在重构
$ claude
> 将 utils.py 中的函数按职责拆分到不同模块

# 三个 Claude 同时工作，互不干扰
```

### 13.9.2 与 GitHub Actions 集成

```yaml
# .github/workflows/claude-fix.yml
name: Claude Auto-Fix
on:
  issues:
    types: [labeled]

jobs:
  auto-fix:
    if: github.event.label.name == 'claude-fix'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Claude Code
        run: npm install -g @anthropic-ai/claude-code
      
      - name: Run Claude to fix issue
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          claude -p "Read issue #${{ github.event.issue.number }} and fix it. \
            Create a branch, make the fix, write tests, and create a PR." \
            --allowedTools "Read,Write,Bash" \
            --max-turns 30
```

### 13.9.3 自定义斜杠命令

```markdown
# .claude/commands/review.md
---
description: "审查当前分支的所有变更"
---

请审查当前分支相对于 main 的所有代码变更：

1. 运行 `git diff main...HEAD` 查看所有变更
2. 逐文件分析：
   - 代码质量问题
   - 潜在的 Bug
   - 安全隐患
   - 性能问题
   - 测试覆盖
3. 给出总体评价和改进建议
```

使用：
```bash
> /review
```

## 13.10 Claude Code vs Cursor：如何选择

```{list-table} Claude Code vs Cursor 深度对比
:header-rows: 1
:widths: 25 37 38

* - 维度
  - Cursor
  - Claude Code
* - **交互方式**
  - GUI 编辑器（可视化）
  - 终端命令行（文本）
* - **核心理念**
  - AI 增强的编辑器
  - 终端中的 AI 工程师
* - **最佳场景**
  - 日常编码、UI 开发、实时预览
  - 重构、Bug 修复、自动化任务
* - **上下文理解**
  - 通过索引 + @引用
  - 通过主动读取文件
* - **代码编辑**
  - 可视化 diff，逐行审查
  - 直接修改，终端显示变更
* - **执行能力**
  - Agent 模式可执行命令
  - 原生支持所有终端操作
* - **Git 集成**
  - 基础 Git 操作
  - 深度 Git 操作（创建 PR 等）
* - **CI/CD 集成**
  - 不支持
  - 原生支持 Headless 模式
* - **多实例**
  - 一个编辑器窗口
  - 多终端并行
* - **远程开发**
  - 需要 Remote SSH
  - 天然支持（SSH + claude）
* - **学习曲线**
  - 低（VS Code 用户秒上手）
  - 中（需要熟悉终端）
* - **价格**
  - $20/月（Pro）
  - API 按量计费（约 $20-50/月）
```

### 13.10.1 互补使用的最佳实践

```
最佳工作流：Cursor + Claude Code 互补

Cursor 负责：
├── 日常编码（Tab 补全 + Cmd-K）
├── UI 开发（实时预览）
├── 代码浏览和理解（Chat + @codebase）
└── 小范围修改（Composer）

Claude Code 负责：
├── 大规模重构（跨多文件的结构性变更）
├── Bug 修复（从 Issue 到 PR 的完整流程）
├── 自动化任务（CI/CD、脚本编写）
├── 代码审查（git diff | claude）
└── 远程服务器操作（SSH 环境）
```

## 13.11 实战：用 Claude Code 完成完整开发任务

### 13.11.1 场景：从 Issue 到 PR

```bash
$ claude

> 请查看 GitHub Issue #156，理解问题，修复它，写测试，创建 PR

Claude:
[Run] gh issue view 156

Issue #156: "用户搜索 API 在特殊字符输入时返回 500 错误"
报告者描述：搜索 "O'Brien" 时服务器返回 Internal Server Error

让我复现这个问题...
[Read] src/services/user_service.py
[Read] src/repositories/user_repository.py

找到了！在 user_repository.py:34，搜索查询使用了字符串拼接：
```python
query = f"SELECT * FROM users WHERE name LIKE '%{search_term}%'"
```
这不仅有 SQL 注入风险，还会在包含单引号的输入时报错。

修复方案：使用 SQLAlchemy 的参数化查询。

[Run] git checkout -b fix/user-search-special-chars

[Edit] src/repositories/user_repository.py
  - 替换字符串拼接为参数化查询
  - 添加输入清理

[Write] tests/test_user_search.py
  - 测试正常搜索
  - 测试特殊字符（单引号、双引号、百分号、反斜杠）
  - 测试 SQL 注入防护
  - 测试空字符串和超长输入

[Run] pytest tests/test_user_search.py -v
✅ 6/6 测试通过

[Run] pytest  # 运行全部测试确保没有回归
✅ 234/234 测试通过

[Run] git add -A && git commit -m "fix(search): use parameterized queries for user search

- Replace string concatenation with SQLAlchemy bindparams
- Add input sanitization for special characters
- Add comprehensive test coverage for edge cases

Fixes #156"

[Run] git push origin fix/user-search-special-chars
[Run] gh pr create \
  --title "fix(search): handle special characters in user search" \
  --body "## Problem
User search API returns 500 when input contains special characters like quotes.

## Root Cause
String concatenation in SQL query (SQL injection vulnerability).

## Solution
- Use SQLAlchemy parameterized queries
- Add input sanitization
- Add test coverage for special characters

## Testing
- Added 6 new test cases covering special characters, SQL injection, edge cases
- All 234 existing tests pass

Fixes #156"

PR #162 已创建 ✅
```

## 13.12 本章小结

Claude Code 代表了 AI 编程工具的另一种哲学——**不是让编辑器更智能，而是让 AI 成为你的编程搭档**。它在终端中运行，拥有完整的系统访问能力，可以独立完成从理解问题到提交代码的全流程。

```{admonition} 关键要点
:class: tip
1. **Claude Code 是 Agent，不是编辑器插件**——它能自主读取、修改、执行、测试
2. **CLAUDE.md** 是项目的"大脑"，好的 CLAUDE.md 能显著提升 AI 的工作质量
3. **权限系统**确保安全：读取自动执行，写入需确认，危险操作被阻止
4. **管道模式**是杀手级特性：`git diff | claude` 让 AI 融入现有工作流
5. **与 Cursor 互补**而非替代：Cursor 做日常编码，Claude Code 做重构和自动化
6. **CI/CD 集成**让 Claude Code 成为团队的"AI 开发者"
```

```{admonition} 思考题
:class: hint
1. 你更喜欢 GUI 编辑器还是终端工具？为什么？
2. 让 AI 自主执行命令（如 `git push`），你觉得安全吗？需要什么样的保障？
3. 如果 Claude Code 可以在 CI 中自动修复 Bug，这对团队的工作流程有什么影响？
4. Cursor 和 Claude Code 的互补使用，你会如何设计自己的工作流？
```

在下一章中，我们将探讨氛围编程的边界与风险——当 AI 生成了大量代码，我们如何确保质量、安全和可维护性。
