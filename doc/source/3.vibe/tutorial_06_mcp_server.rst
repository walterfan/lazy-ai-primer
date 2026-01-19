####################################
Tutorial 6: MCP Server
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

============ ==========================
**Abstract** Model Context Protocol 扩展 AI 能力
**Authors**  Walter Fan
**Status**   WIP
**Updated**  |date|
============ ==========================

.. contents::
   :local:

什么是 MCP
==========

MCP（Model Context Protocol）是 Anthropic 推出的开放协议，用于连接 AI 模型与外部数据源和工具。

.. code-block:: text

   MCP 架构

   ┌─────────────────────────────────────────────────────────────────┐
   │                        AI 应用 (Cursor)                         │
   │                              │                                  │
   │                        MCP Client                               │
   │                              │                                  │
   └──────────────────────────────┼──────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │ MCP Server│ │ MCP Server│ │ MCP Server│
              │ (文件系统) │ │ (数据库)  │ │ (API)     │
              └───────────┘ └───────────┘ └───────────┘

MCP 的核心概念
--------------

1. **Resources（资源）**: 数据源，如文件、数据库记录
2. **Tools（工具）**: AI 可以调用的函数
3. **Prompts（提示）**: 预定义的提示模板
4. **Sampling（采样）**: 让服务器请求 AI 补全

为什么需要 MCP
--------------

- **扩展能力**: 让 AI 访问外部系统
- **标准化**: 统一的协议，一次开发到处使用
- **安全**: 受控的访问方式
- **灵活**: 可以连接任何数据源

Cursor 中使用 MCP
=================

配置 MCP Server
---------------

在 Cursor 中配置 MCP Server：

1. 打开 Settings
2. 找到 MCP 配置部分
3. 添加 Server 配置

配置文件位置：``~/.cursor/mcp.json``

.. code-block:: json

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
        },
        "github": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-github"],
          "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
          }
        }
      }
    }

常用 MCP Server
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Server
     - 功能
     - 使用场景
   * - filesystem
     - 文件系统访问
     - 读写项目外的文件
   * - github
     - GitHub API
     - 管理仓库、Issues、PR
   * - postgres
     - PostgreSQL 数据库
     - 查询和修改数据库
   * - sqlite
     - SQLite 数据库
     - 本地数据库操作
   * - brave-search
     - 网络搜索
     - 搜索最新信息
   * - memory
     - 持久化记忆
     - 跨会话保存信息

安装和使用 MCP Server
=====================

Filesystem Server
-----------------

让 AI 访问指定目录的文件：

**安装**:

.. code-block:: bash

    npm install -g @modelcontextprotocol/server-filesystem

**配置**:

.. code-block:: json

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/Users/yourname/Documents",
            "/Users/yourname/Projects"
          ]
        }
      }
    }

**使用**:

在 Cursor Chat 中::

    @mcp 读取 /Users/yourname/Documents/notes.txt 的内容

GitHub Server
-------------

管理 GitHub 仓库：

**配置**:

.. code-block:: json

    {
      "mcpServers": {
        "github": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-github"],
          "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxx"
          }
        }
      }
    }

**使用**:

::

    @mcp 列出我的 GitHub 仓库

    @mcp 创建一个新的 Issue：
    仓库：myrepo
    标题：Bug: 登录失败
    内容：用户反馈登录时出现 500 错误

PostgreSQL Server
-----------------

连接 PostgreSQL 数据库：

**配置**:

.. code-block:: json

    {
      "mcpServers": {
        "postgres": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-postgres"],
          "env": {
            "POSTGRES_CONNECTION_STRING": "postgresql://user:pass@localhost:5432/mydb"
          }
        }
      }
    }

**使用**:

::

    @mcp 查询 users 表中最近注册的 10 个用户

    @mcp 统计每个月的订单数量

Memory Server
-------------

跨会话保存信息：

**配置**:

.. code-block:: json

    {
      "mcpServers": {
        "memory": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-memory"]
        }
      }
    }

**使用**:

::

    @mcp 记住：项目使用 Python 3.11 和 FastAPI

    @mcp 我们项目使用什么技术栈？

开发自定义 MCP Server
=====================

MCP Server 可以用 Python 或 TypeScript 开发。

Python MCP Server
-----------------

**安装依赖**:

.. code-block:: bash

    pip install mcp

**基本结构**:

.. code-block:: python

    # my_server.py
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Resource, Tool, TextContent

    # 创建服务器
    server = Server("my-server")

    # 定义资源
    @server.list_resources()
    async def list_resources():
        return [
            Resource(
                uri="myapp://config",
                name="Application Config",
                description="Current application configuration"
            )
        ]

    @server.read_resource()
    async def read_resource(uri: str):
        if uri == "myapp://config":
            return TextContent(
                type="text",
                text='{"debug": true, "version": "1.0.0"}'
            )

    # 定义工具
    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="get_weather",
                description="Get weather for a city",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["city"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "get_weather":
            city = arguments["city"]
            # 实际实现中调用天气 API
            return TextContent(
                type="text",
                text=f"Weather in {city}: Sunny, 25°C"
            )

    # 运行服务器
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)

    if __name__ == "__main__":
        import asyncio
        asyncio.run(main())

**配置使用**:

.. code-block:: json

    {
      "mcpServers": {
        "my-server": {
          "command": "python",
          "args": ["/path/to/my_server.py"]
        }
      }
    }

TypeScript MCP Server
---------------------

**安装依赖**:

.. code-block:: bash

    npm init -y
    npm install @modelcontextprotocol/sdk

**基本结构**:

.. code-block:: typescript

    // src/index.ts
    import { Server } from "@modelcontextprotocol/sdk/server/index.js";
    import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

    const server = new Server(
      { name: "my-server", version: "1.0.0" },
      { capabilities: { tools: {}, resources: {} } }
    );

    // 定义工具
    server.setRequestHandler("tools/list", async () => ({
      tools: [
        {
          name: "calculate",
          description: "Perform a calculation",
          inputSchema: {
            type: "object",
            properties: {
              expression: { type: "string" }
            },
            required: ["expression"]
          }
        }
      ]
    }));

    server.setRequestHandler("tools/call", async (request) => {
      if (request.params.name === "calculate") {
        const expr = request.params.arguments.expression;
        const result = eval(expr); // 注意：实际使用中要安全处理
        return { content: [{ type: "text", text: String(result) }] };
      }
    });

    // 启动服务器
    const transport = new StdioServerTransport();
    server.connect(transport);

实用 MCP Server 示例
====================

项目文档 Server
---------------

让 AI 访问项目文档：

.. code-block:: python

    # docs_server.py
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Resource, TextContent
    import os
    from pathlib import Path

    server = Server("docs-server")
    DOCS_DIR = Path("/path/to/project/docs")

    @server.list_resources()
    async def list_resources():
        resources = []
        for doc_file in DOCS_DIR.rglob("*.md"):
            rel_path = doc_file.relative_to(DOCS_DIR)
            resources.append(
                Resource(
                    uri=f"docs://{rel_path}",
                    name=str(rel_path),
                    description=f"Documentation: {rel_path}"
                )
            )
        return resources

    @server.read_resource()
    async def read_resource(uri: str):
        if uri.startswith("docs://"):
            path = uri.replace("docs://", "")
            file_path = DOCS_DIR / path
            if file_path.exists():
                content = file_path.read_text()
                return TextContent(type="text", text=content)
        return TextContent(type="text", text="Resource not found")

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)

    if __name__ == "__main__":
        import asyncio
        asyncio.run(main())

API 测试 Server
---------------

让 AI 测试 API：

.. code-block:: python

    # api_tester_server.py
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    import httpx
    import json

    server = Server("api-tester")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="http_request",
                description="Make an HTTP request",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE"]
                        },
                        "url": {"type": "string"},
                        "headers": {"type": "object"},
                        "body": {"type": "object"}
                    },
                    "required": ["method", "url"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "http_request":
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=arguments["method"],
                    url=arguments["url"],
                    headers=arguments.get("headers", {}),
                    json=arguments.get("body")
                )
                result = {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.text
                }
                return TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)

    if __name__ == "__main__":
        import asyncio
        asyncio.run(main())

MCP 最佳实践
============

1. 安全考虑
-----------

- 限制文件系统访问范围
- 使用环境变量存储敏感信息
- 验证工具输入
- 记录操作日志

2. 性能优化
-----------

- 缓存频繁访问的资源
- 使用连接池
- 异步处理 I/O 操作

3. 错误处理
-----------

.. code-block:: python

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            # 工具逻辑
            pass
        except ValueError as e:
            return TextContent(
                type="text",
                text=f"Invalid input: {e}"
            )
        except Exception as e:
            return TextContent(
                type="text",
                text=f"Error: {e}"
            )

4. 文档和描述
-------------

为工具和资源提供清晰的描述：

.. code-block:: python

    Tool(
        name="search_logs",
        description="""
        Search application logs.

        Examples:
        - Search for errors: {"query": "ERROR", "limit": 100}
        - Search by time: {"query": "timeout", "start": "2024-01-01"}
        """,
        inputSchema={...}
    )

调试 MCP Server
===============

1. 查看日志
-----------

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)

2. 测试工具
-----------

使用 MCP Inspector：

.. code-block:: bash

    npx @modelcontextprotocol/inspector python my_server.py

3. 在 Cursor 中调试
-------------------

- 检查 MCP 配置是否正确
- 查看 Cursor 的开发者工具
- 检查 Server 进程是否运行

小结
====

本教程介绍了 MCP Server：

- **MCP 概念**: 连接 AI 与外部系统的协议
- **常用 Server**: filesystem、github、postgres
- **自定义开发**: Python 和 TypeScript 实现
- **最佳实践**: 安全、性能、错误处理

关键要点：

1. MCP 扩展了 AI 的能力边界
2. 可以连接任何数据源和服务
3. 开发自定义 Server 很简单

下一步
------

在下一个教程中，我们将学习 Spec-Driven Development（规格驱动开发）。

练习
====

1. 配置 filesystem MCP Server
2. 配置 GitHub MCP Server 并管理 Issues
3. 开发一个简单的自定义 MCP Server
4. 将 MCP Server 集成到你的工作流程中

参考资源
========

- `MCP 官方文档 <https://modelcontextprotocol.io/>`_
- `MCP GitHub 仓库 <https://github.com/modelcontextprotocol>`_
- `MCP Server 列表 <https://github.com/modelcontextprotocol/servers>`_
- `Cursor MCP 配置 <https://docs.cursor.com/context/model-context-protocol>`_
