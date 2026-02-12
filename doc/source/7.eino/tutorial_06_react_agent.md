# Tutorial 6: ReAct Agent

## ReAct 模式简介

**ReAct**（Reasoning + Acting）是让 LLM 交替进行「推理」与「行动」的 Agent 模式：根据当前上下文决定下一步是生成回答，还是调用某个工具，并根据工具结果继续推理，直到得出最终答案。

Eino 在 **Flow** 层提供 ReAct 的预置实现，并与 **ToolsNode**、**ToolCallingChatModel** 配合使用。

参考：[Eino: ReAct Agent Manual](https://www.cloudwego.io/docs/eino/)、[DeepWiki ADK](https://deepwiki.com/cloudwego/eino)。

## 图中结构概览

典型 ReAct 图包含：

- **模型节点**：使用 `ToolCallingChatModel`，输出中可包含 `tool_calls`。  
- **工具节点（ToolsNode）**：根据 `tool_calls` 调用对应 `BaseTool`，将结果写回消息（如 assistant message + tool results）。  
- **边与循环**：  
  - 模型 → 若存在 tool_calls → 工具节点 → 将结果追加到对话 → 再回到模型；  
  - 模型 → 若无 tool_calls（或已满足终止条件）→ 结束，输出最终回答。

这样形成「思考 → 选工具 → 执行 → 再思考」的循环，直到模型选择不再调用工具。

## 配置要点

1. **工具注册**：为 ToolsNode 提供一组 `BaseTool`（名称、描述、参数 schema），与模型 function calling 对齐。  
2. **模型**：使用 eino-ext 中支持 function calling 的 ChatModel（如 OpenAI、Claude）。  
3. **终止条件**：图中需定义「何时结束循环」（例如无 tool_calls、或达到最大步数）。  
4. **流式**：若需流式输出，使用图的 `Stream` 执行；工具调用回合仍由框架按需拼接/收集。

## 与 ADK 的关系

**ChatModelAgent**（ADK）是对 ReAct 模式的高层封装：内部已包含「模型 + 工具循环」的图结构，你主要配置 ChatModel、Tools 和可选中间件即可，无需手写图的每条边。  
因此：

- 快速上手：用 **ChatModelAgent** 即可得到 ReAct 行为。  
- 需要自定义图结构、分支或状态时：使用 **Graph** + 模型节点 + ToolsNode 自行编排。

文档：[Eino ReAct Agent Manual](https://www.cloudwego.io/docs/eino/)、[ChatModelAgent](https://deepwiki.com/cloudwego/eino)。

## 小结

- ReAct = 推理 + 行动：模型决定是否调用工具，工具结果再回馈模型，循环直到结束。  
- Eino 通过 Graph（模型节点 + ToolsNode + 循环边）或 ADK 的 ChatModelAgent 实现 ReAct。  
- 配置要点：ToolCallingChatModel、Tools 注册、终止条件与可选流式。

下一节将系统介绍 **ADK**：Agent 接口、Runner、ChatModelAgent 及扩展方式。
