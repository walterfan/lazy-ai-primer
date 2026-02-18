# Bug Fix: 中文翻译未保存到 Markdown 文件

## 问题描述

用户报告生成的 `report.md` 文件中，"中文翻译"部分显示的是工具调用的 JSON 格式，而不是实际的翻译内容：

```markdown
## 中文翻译

{"name": "fetch_static", "arguments": {"url": "https://example.com"}}
```

## 根本原因

在 `main.py` 的 `react_fetch()` 函数中，代码从 LangGraph 的最终状态中提取结果时使用了：

```python
last_message = final_state["messages"][-1]
return last_message.content
```

问题：
1. 最后一条消息是 `AIMessage`（Agent 的回复），而不是 `ToolMessage`（工具的执行结果）
2. `AIMessage` 的 content 可能只是 Agent 的总结或确认信息，不包含实际抓取的网页内容
3. 实际的抓取内容存储在 `ToolMessage` 中

## 解决方案

修改 `react_fetch()` 函数，从消息历史中正确提取 `ToolMessage` 的内容：

```python
def react_fetch(url: str) -> str:
    from langchain_core.messages import ToolMessage

    # ... [初始化和图调用代码] ...

    final_state = fetch_graph.invoke(initial_state)

    # 从后向前查找最后一个成功的 ToolMessage
    messages = final_state["messages"]

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            # 检查是否是成功的抓取结果（不是错误或警告）
            if not content.startswith("[ERROR]") and len(content) > 100:
                return content

    # 兜底：如果没有找到有效的 ToolMessage，返回最后一条消息
    return messages[-1].content
```

## 修复逻辑

1. **从后向前遍历**：从最新的消息开始查找
2. **类型检查**：只查找 `ToolMessage` 类型的消息
3. **内容验证**：
   - 不是错误消息（不以 `[ERROR]` 开头）
   - 内容长度 > 100 字符（确保不是警告或空内容）
4. **兜底机制**：如果没找到有效的 `ToolMessage`，返回最后一条消息

## 测试

创建了 `test_extraction.py` 测试两种场景：

### 场景 1：正常抓取
```
HumanMessage → AIMessage (tool_calls) → ToolMessage (内容) → AIMessage (总结)
```
✅ 正确提取 `ToolMessage` 中的内容

### 场景 2：错误重试
```
HumanMessage
→ AIMessage (fetch_static)
→ ToolMessage ([ERROR]...)    ← 跳过
→ AIMessage (fetch_dynamic)
→ ToolMessage (实际内容)      ← 提取这个
→ AIMessage (总结)
```
✅ 跳过错误消息，提取成功的内容

## 验证

运行测试：
```bash
poetry run python test_extraction.py
```

预期输出：
```
✅ 找到有效的 ToolMessage:
   内容: This is the content fetched...
```

## 影响范围

**修改的文件**：
- `src/academic_url_agent/main.py` - `react_fetch()` 函数

**新增的文件**：
- `test_extraction.py` - 消息提取测试

**不影响的部分**：
- `pipeline.py` - 翻译和报告生成逻辑正确
- `graph.py` - LangGraph 图结构正确
- `tools.py` - 工具实现正确

## 下一步

重新运行程序生成报告：

```bash
poetry run python -m academic_url_agent.main "https://example.com/article"
```

检查 `report.md` 的"中文翻译"部分应该包含完整的翻译内容。

## 版本

- **Fixed in**: v0.2.1
- **Reported**: 2026-02-15
- **Fixed**: 2026-02-15

## 相关问题

这个问题的根源是 LangGraph 的消息历史结构理解不够深入：

```
messages = [
    HumanMessage(...),      # 用户输入
    AIMessage(...),         # Agent 决策（包含 tool_calls）
    ToolMessage(...),       # 工具执行结果 ← 这才是我们需要的！
    AIMessage(...),         # Agent 总结
]
```

需要提取的是 `ToolMessage`，而不是最后的 `AIMessage`。
