####################################
Tutorial 4: Rules 与配置
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

============ ==========================
**Abstract** Cursor Rules 配置与最佳实践
**Authors**  Walter Fan
**Status**   WIP
**Updated**  |date|
============ ==========================

.. contents::
   :local:

什么是 Cursor Rules
===================

Cursor Rules 是一种让 AI 遵循特定规范的机制。通过配置 Rules，你可以：

- 定义代码风格和规范
- 指定技术栈和框架
- 设置项目特定的约定
- 提供领域知识和上下文

为什么需要 Rules
----------------

没有 Rules 时：

- AI 可能使用你不喜欢的代码风格
- AI 可能不了解你的项目结构
- AI 可能使用过时的 API
- 每次都需要重复说明要求

有了 Rules：

- AI 自动遵循你的规范
- 代码风格一致
- 减少重复沟通
- 提高生成质量

Rules 的类型
============

Cursor 支持多种类型的 Rules：

.. code-block:: text

   Rules 类型

   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                 │
   │   ┌─────────────────┐   ┌─────────────────┐                    │
   │   │   Global Rules  │   │  Project Rules  │                    │
   │   │   全局规则       │   │   项目规则       │                    │
   │   ├─────────────────┤   ├─────────────────┤                    │
   │   │ ~/.cursor/rules │   │ .cursor/rules   │                    │
   │   │ 所有项目生效     │   │ 仅当前项目生效   │                    │
   │   └─────────────────┘   └─────────────────┘                    │
   │                                                                 │
   │   ┌─────────────────┐   ┌─────────────────┐                    │
   │   │  .cursorrules   │   │ Cursor Settings │                    │
   │   │   项目根目录     │   │   设置中配置     │                    │
   │   ├─────────────────┤   ├─────────────────┤                    │
   │   │ 简单配置        │   │ 图形界面配置     │                    │
   │   │ 向后兼容        │   │ 用户级别        │                    │
   │   └─────────────────┘   └─────────────────┘                    │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘

1. .cursorrules 文件
--------------------

最简单的方式，在项目根目录创建 ``.cursorrules`` 文件：

.. code-block:: text

    # .cursorrules

    You are an expert Python developer.

    ## Code Style
    - Use Python 3.11+ features
    - Follow PEP 8 style guide
    - Use type hints for all functions
    - Write docstrings in Google style

    ## Project Structure
    - Use src/ layout
    - Tests go in tests/ directory
    - Use pytest for testing

    ## Dependencies
    - Use Poetry for dependency management
    - Prefer standard library over third-party

2. .cursor/rules 目录
---------------------

更灵活的方式，可以按文件类型或模块配置不同规则：

.. code-block:: text

    .cursor/
    └── rules/
        ├── global.md        # 全局规则
        ├── python.md        # Python 特定规则
        ├── typescript.md    # TypeScript 特定规则
        └── testing.md       # 测试相关规则

**global.md**:

.. code-block:: markdown

    # Global Rules

    ## General Principles
    - Write clean, readable code
    - Prefer simplicity over cleverness
    - Always handle errors appropriately

    ## Documentation
    - Add comments for complex logic
    - Keep README up to date

**python.md**:

.. code-block:: markdown

    ---
    description: Rules for Python files
    globs: ["**/*.py"]
    ---

    # Python Rules

    ## Style
    - Use Black for formatting
    - Use isort for imports
    - Maximum line length: 88 characters

    ## Type Hints
    - All functions must have type hints
    - Use `from __future__ import annotations`

    ## Error Handling
    - Use specific exception types
    - Always log exceptions

3. Cursor Settings
------------------

在 Settings → Rules 中配置用户级别的规则：

1. 打开 Settings（Cmd+,）
2. 搜索 "Rules"
3. 在文本框中输入规则

这些规则会应用到所有项目。

Rules 语法详解
==============

基本结构
--------

Rules 使用 Markdown 格式，支持 YAML front matter：

.. code-block:: markdown

    ---
    description: 规则描述
    globs: ["**/*.py", "**/*.pyi"]
    alwaysApply: false
    ---

    # 规则标题

    ## 分类 1
    - 规则内容

    ## 分类 2
    - 规则内容

Front Matter 选项
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 选项
     - 类型
     - 说明
   * - description
     - string
     - 规则的简短描述
   * - globs
     - string[]
     - 文件匹配模式
   * - alwaysApply
     - boolean
     - 是否总是应用（默认 false）

Globs 模式示例
--------------

.. code-block:: yaml

    # 所有 Python 文件
    globs: ["**/*.py"]

    # 特定目录
    globs: ["src/**/*.py"]

    # 多种文件类型
    globs: ["**/*.ts", "**/*.tsx"]

    # 排除测试文件
    globs: ["**/*.py", "!**/test_*.py"]

实用 Rules 示例
===============

Python 项目 Rules
-----------------

.. code-block:: markdown

    ---
    description: Python project rules
    globs: ["**/*.py"]
    ---

    # Python Development Rules

    ## Environment
    - Python 3.11+
    - Use Poetry for dependency management
    - Use pyproject.toml for configuration

    ## Code Style
    - Follow PEP 8
    - Use Black formatter (line length 88)
    - Use isort for import sorting
    - Use Ruff for linting

    ## Type Hints
    - All functions must have type hints
    - Use `from __future__ import annotations`
    - Use `typing` module for complex types

    ## Documentation
    - Use Google-style docstrings
    - Include type info in docstrings
    - Add examples for complex functions

    ## Error Handling
    - Use specific exception types
    - Create custom exceptions when needed
    - Always log exceptions with context

    ## Testing
    - Use pytest
    - Test file naming: test_*.py
    - Aim for 80%+ coverage
    - Use fixtures for common setup

    ## Example Function

    ```python
    from __future__ import annotations

    def calculate_total(
        items: list[dict[str, float]],
        tax_rate: float = 0.1,
    ) -> float:
        """Calculate total price with tax.

        Args:
            items: List of items with 'price' key.
            tax_rate: Tax rate as decimal. Defaults to 0.1.

        Returns:
            Total price including tax.

        Raises:
            ValueError: If items list is empty.

        Example:
            >>> items = [{'price': 10.0}, {'price': 20.0}]
            >>> calculate_total(items)
            33.0
        """
        if not items:
            raise ValueError("Items list cannot be empty")

        subtotal = sum(item['price'] for item in items)
        return subtotal * (1 + tax_rate)
    ```

TypeScript/React 项目 Rules
----------------------------

.. code-block:: markdown

    ---
    description: TypeScript React project rules
    globs: ["**/*.ts", "**/*.tsx"]
    ---

    # TypeScript React Rules

    ## Environment
    - TypeScript 5.x
    - React 18.x
    - Next.js 14.x (App Router)

    ## Code Style
    - Use ESLint + Prettier
    - Use functional components
    - Use hooks, no class components

    ## Naming Conventions
    - Components: PascalCase (UserProfile.tsx)
    - Hooks: camelCase with 'use' prefix (useAuth.ts)
    - Utils: camelCase (formatDate.ts)
    - Types: PascalCase with suffix (UserType, UserProps)

    ## Component Structure
    ```tsx
    // 1. Imports
    import { useState } from 'react';

    // 2. Types
    interface Props {
      title: string;
      onSubmit: (data: FormData) => void;
    }

    // 3. Component
    export function MyComponent({ title, onSubmit }: Props) {
      // 3.1 Hooks
      const [state, setState] = useState('');

      // 3.2 Handlers
      const handleClick = () => {};

      // 3.3 Render
      return <div>{title}</div>;
    }
    ```

    ## State Management
    - Local state: useState
    - Complex state: useReducer
    - Global state: Zustand or Context
    - Server state: TanStack Query

    ## Styling
    - Use Tailwind CSS
    - Use cn() for conditional classes
    - Prefer utility classes over custom CSS

API 开发 Rules
--------------

.. code-block:: markdown

    ---
    description: REST API development rules
    globs: ["**/api/**/*.py", "**/routes/**/*.py"]
    ---

    # API Development Rules

    ## Design Principles
    - RESTful design
    - Use proper HTTP methods
    - Use proper status codes
    - Version APIs (v1, v2)

    ## Response Format
    ```json
    {
      "success": true,
      "data": {},
      "message": "Success",
      "timestamp": "2024-01-01T00:00:00Z"
    }
    ```

    ## Error Response
    ```json
    {
      "success": false,
      "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input",
        "details": []
      },
      "timestamp": "2024-01-01T00:00:00Z"
    }
    ```

    ## Status Codes
    - 200: Success
    - 201: Created
    - 400: Bad Request
    - 401: Unauthorized
    - 403: Forbidden
    - 404: Not Found
    - 500: Internal Server Error

    ## Validation
    - Validate all input
    - Use Pydantic models
    - Return detailed error messages

    ## Security
    - Use JWT for authentication
    - Rate limiting on all endpoints
    - Input sanitization
    - CORS configuration

Rules 最佳实践
==============

1. 保持简洁
-----------

不要写太长的 Rules，AI 可能无法完全遵循：

.. code-block:: markdown

    # 好的做法
    - Use Python 3.11+
    - Follow PEP 8
    - Use type hints

    # 不好的做法（太详细）
    - Use Python 3.11 or higher versions...
    - Follow PEP 8 style guide which includes...
    - Use type hints according to PEP 484...

2. 提供示例
-----------

示例比规则描述更有效：

.. code-block:: markdown

    ## Error Handling

    Good:
    ```python
    try:
        result = process(data)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise
    ```

    Bad:
    ```python
    try:
        result = process(data)
    except:
        pass
    ```

3. 分层组织
-----------

使用多个 Rules 文件，按职责分离：

.. code-block:: text

    .cursor/rules/
    ├── global.md       # 通用规则
    ├── python.md       # Python 规则
    ├── frontend.md     # 前端规则
    ├── api.md          # API 规则
    └── testing.md      # 测试规则

4. 版本控制
-----------

将 Rules 文件纳入版本控制，团队共享：

.. code-block:: bash

    git add .cursorrules
    git add .cursor/rules/
    git commit -m "Add Cursor rules for project"

5. 定期更新
-----------

随着项目演进，更新 Rules：

- 技术栈升级
- 新的代码规范
- 团队约定变化

调试 Rules
==========

检查 Rules 是否生效
-------------------

1. 打开 Chat
2. 输入：``@rules 显示当前生效的规则``
3. 或者在对话中观察 AI 是否遵循规则

常见问题
--------

**Rules 没有生效**

- 检查文件位置是否正确
- 检查 globs 模式是否匹配
- 重启 Cursor

**Rules 冲突**

- 项目规则优先于全局规则
- 更具体的 globs 优先
- 后定义的规则优先

**Rules 太长**

- 精简规则内容
- 只保留最重要的规则
- 使用示例代替详细描述

实战：为项目配置 Rules
======================

步骤 1：分析项目
----------------

首先了解项目的技术栈和规范：

- 使用什么语言和框架？
- 有什么代码规范？
- 项目结构是怎样的？

步骤 2：创建基础 Rules
----------------------

创建 ``.cursorrules`` 文件：

.. code-block:: markdown

    # Project: My Awesome App

    ## Tech Stack
    - Python 3.11 + FastAPI
    - PostgreSQL + SQLAlchemy
    - React + TypeScript

    ## Code Style
    - Backend: PEP 8 + Black
    - Frontend: ESLint + Prettier

    ## Project Structure
    - backend/: Python API
    - frontend/: React app
    - docs/: Documentation

步骤 3：添加特定规则
--------------------

为不同模块创建专门的规则：

.. code-block:: bash

    mkdir -p .cursor/rules
    touch .cursor/rules/backend.md
    touch .cursor/rules/frontend.md

步骤 4：测试和迭代
------------------

1. 使用 AI 生成代码
2. 检查是否符合规则
3. 根据结果调整规则

小结
====

本教程介绍了 Cursor Rules 的配置方法：

- **Rules 类型**: .cursorrules、.cursor/rules、Settings
- **语法**: Markdown + YAML front matter
- **最佳实践**: 简洁、示例、分层、版本控制

关键要点：

1. Rules 让 AI 遵循你的规范
2. 好的 Rules 需要示例
3. 定期更新 Rules

下一步
------

在下一个教程中，我们将学习提示词（Prompting）技巧，让 AI 更好地理解你的意图。

练习
====

1. 为你的项目创建一个 .cursorrules 文件
2. 创建至少 3 个不同的规则文件
3. 测试规则是否生效
4. 与团队分享你的规则配置

参考资源
========

- `Cursor Rules 官方文档 <https://docs.cursor.com/context/rules-for-ai>`_
- `Cursor Directory - 社区规则 <https://cursor.directory/>`_
- `Awesome Cursor Rules <https://github.com/PatrickJS/awesome-cursorrules>`_
