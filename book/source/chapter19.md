(chapter19)=
# 第十九章：Agent 的工具系统设计

```{mermaid}
mindmap
  root((Agent工具系统))
    Function Calling
      工具定义
      参数解析
      结果返回
    MCP协议
      Host/Client/Server
      Resources
      Tools
      Prompts
      Sampling
    传输层
      stdio
      SSE
      Streamable HTTP
    FC vs MCP对比
      标准化程度
      生态系统
    工具安全
      沙箱执行
      权限控制
      输入验证
    工具编排
      串行调用
      并行调用
      条件调用
```

> "Agent 的能力边界由它能使用的工具决定。"

## 19.1 Function Calling 的原理

Function Calling 是让 LLM 调用外部工具的核心机制：

```python
from openai import OpenAI

client = OpenAI()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# LLM 决定是否调用工具
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto"
)

# 如果 LLM 决定调用工具
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name  # "get_weather"
    arguments = json.loads(tool_call.function.arguments)  # {"city": "北京"}
    
    # 执行实际的工具调用
    result = get_weather(**arguments)
    
    # 将结果返回给 LLM
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })
    
    # LLM 基于工具结果生成最终回答
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
```

## 19.2 MCP（Model Context Protocol）详解

MCP 是 Anthropic 于 2024 年底发布的开放协议，旨在标准化 LLM 与外部工具/数据源的交互方式。

### MCP 架构

```
┌─────────────────────────────────────────────┐
│                  MCP Host                    │
│  (Claude Desktop / Cursor / IDE)            │
│                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │MCP      │  │MCP      │  │MCP      │    │
│  │Client 1 │  │Client 2 │  │Client 3 │    │
│  └────┬────┘  └────┬────┘  └────┬────┘    │
└───────┼─────────────┼─────────────┼─────────┘
        │             │             │
   ┌────┴────┐   ┌────┴────┐  ┌────┴────┐
   │MCP      │   │MCP      │  │MCP      │
   │Server   │   │Server   │  │Server   │
   │(文件系统)│   │(GitHub) │  │(数据库) │
   └─────────┘   └─────────┘  └─────────┘
```

### MCP 的四大能力

**1. Resources（资源）**：向 LLM 暴露数据
```python
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="file:///project/README.md",
            name="Project README",
            mimeType="text/markdown"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "file:///project/README.md":
        content = Path("README.md").read_text()
        return content
```

**2. Tools（工具）**：让 LLM 执行操作
```python
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_code",
            description="在代码库中搜索指定模式",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "搜索模式（正则表达式）"},
                    "file_type": {"type": "string", "description": "文件类型过滤，如 .py"}
                },
                "required": ["pattern"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_code":
        results = search_codebase(arguments["pattern"], arguments.get("file_type"))
        return [TextContent(type="text", text=json.dumps(results))]
```

**3. Prompts（提示模板）**：预定义的交互模板
```python
@server.list_prompts()
async def list_prompts():
    return [
        Prompt(
            name="code_review",
            description="代码审查提示模板",
            arguments=[
                PromptArgument(name="code", description="要审查的代码", required=True)
            ]
        )
    ]
```

**4. Sampling（采样）**：让 Server 请求 LLM 生成内容（反向调用）

### MCP 传输层

```python
# 1. stdio 传输（本地进程）
# 最简单，适合本地工具
import asyncio
from mcp.server.stdio import stdio_server

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

# 2. SSE 传输（HTTP Server-Sent Events）
# 适合远程服务
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette

app = Starlette()
sse = SseServerTransport("/messages")

@app.route("/sse")
async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1])

# 3. Streamable HTTP（最新）
# 支持双向流式通信
from mcp.server.streamable_http import StreamableHTTPServerTransport
```

## 19.3 完整 MCP Server 示例

```python
"""一个完整的 MCP Server：项目管理工具"""
import json
from datetime import datetime
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource

server = Server("project-manager")

# 内存中的任务存储
tasks = []

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="create_task",
            description="创建新任务",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "任务标题"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "assignee": {"type": "string", "description": "负责人"}
                },
                "required": ["title"]
            }
        ),
        Tool(
            name="list_tasks",
            description="列出所有任务，可按状态筛选",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["todo", "in_progress", "done"]}
                }
            }
        ),
        Tool(
            name="update_task_status",
            description="更新任务状态",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "status": {"type": "string", "enum": ["todo", "in_progress", "done"]}
                },
                "required": ["task_id", "status"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "create_task":
        task = {
            "id": len(tasks) + 1,
            "title": arguments["title"],
            "priority": arguments.get("priority", "medium"),
            "assignee": arguments.get("assignee", "unassigned"),
            "status": "todo",
            "created_at": datetime.now().isoformat()
        }
        tasks.append(task)
        return [TextContent(type="text", text=f"任务创建成功: {json.dumps(task, ensure_ascii=False)}")]
    
    elif name == "list_tasks":
        filtered = tasks
        if "status" in arguments:
            filtered = [t for t in tasks if t["status"] == arguments["status"]]
        return [TextContent(type="text", text=json.dumps(filtered, ensure_ascii=False))]
    
    elif name == "update_task_status":
        task_id = arguments["task_id"]
        for task in tasks:
            if task["id"] == task_id:
                task["status"] = arguments["status"]
                return [TextContent(type="text", text=f"任务 {task_id} 状态已更新为 {arguments['status']}")]
        return [TextContent(type="text", text=f"未找到任务 {task_id}")]

# 运行服务器
if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    
    asyncio.run(main())
```

## 19.4 OpenAI Function Calling vs MCP 对比

| 维度 | OpenAI Function Calling | MCP |
|------|------------------------|-----|
| 标准化 | OpenAI 私有 API | 开放协议 |
| 工具发现 | 每次请求传递工具定义 | 动态发现（list_tools） |
| 数据访问 | 需要自己实现 | Resources 原生支持 |
| 传输方式 | HTTP API | stdio / SSE / Streamable HTTP |
| 生态系统 | OpenAI 生态 | 跨平台、跨模型 |
| 适用场景 | 简单工具调用 | 复杂的工具生态系统 |

## 19.5 工具安全

```python
# 工具安全最佳实践

# 1. 输入验证
def validate_tool_input(arguments: dict, schema: dict) -> bool:
    """严格验证工具输入"""
    from jsonschema import validate, ValidationError
    try:
        validate(instance=arguments, schema=schema)
        return True
    except ValidationError:
        return False

# 2. 沙箱执行
import subprocess
def execute_in_sandbox(command: str, timeout: int = 30) -> str:
    """在沙箱中执行命令"""
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        timeout=timeout,
        # 限制资源
        env={"PATH": "/usr/bin:/bin"}  # 限制可用命令
    )
    return result.stdout[:10000]  # 限制输出大小

# 3. 权限控制
TOOL_PERMISSIONS = {
    "read_file": {"allowed_paths": ["/project/"]},
    "write_file": {"allowed_paths": ["/project/output/"]},
    "execute_command": {"blocked_commands": ["rm", "dd", "format"]},
}
```

## 19.6 本章小结

工具系统是 Agent 能力的关键扩展点。从 OpenAI 的 Function Calling 到 Anthropic 的 MCP 协议，工具标准化正在快速发展。

核心要点：
1. **好的工具描述**是 LLM 正确使用工具的前提
2. **MCP 协议**正在成为工具标准化的事实标准
3. **安全第一**：工具执行必须有沙箱、权限控制和输入验证
4. **工具编排**：复杂任务需要串行、并行、条件调用的组合

```{admonition} 思考题
:class: hint
1. MCP 协议会成为 AI 工具的"HTTP"吗？
2. 如何设计工具描述，让 LLM 更准确地选择和使用工具？
3. 工具安全的最大挑战是什么？
```
