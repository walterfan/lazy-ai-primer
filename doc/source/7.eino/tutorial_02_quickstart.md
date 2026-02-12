# Tutorial 2: 快速开始

## 环境要求

- **Go**：1.18 及以上  
- 代码风格由 `golangci-lint` 约束；导出符号需 GoDoc 注释，格式使用 `gofmt -s`。

参考：[Eino Quick Start](https://www.cloudwego.io/docs/eino/)。

## 最小 LLM 应用

在 Eino 中，一个最简单的「提示 → 模型 → 输出」管道可以用 **Chain** 表示：把 `ChatTemplate` 和 `ChatModel` 顺序连接。

概念流程：

```text
用户输入 → ChatTemplate（组消息）→ ChatModel（调用 LLM）→ 输出
```

实现步骤概要（具体 API 以当前 eino/eino-ext 为准）：

1. **引入依赖**  
   - `github.com/cloudwego/eino`（核心）  
   - `github.com/cloudwego/eino-ext` 中对应模型实现（如 OpenAI）

2. **创建 ChatTemplate**  
   - 定义系统提示与用户占位符，将用户输入填入模板得到消息列表。

3. **创建 ChatModel**  
   - 使用 eino-ext 中的 OpenAI（或其它）实现，配置 API Key 等。

4. **用 Chain 串联**  
   - 将 Template 与 Model 组成一条 Chain，调用 `Invoke` 或 `Stream`。

5. **执行**  
   - `Invoke(ctx, input)`：同步得到最终结果。  
   - `Stream(ctx, input)`：得到流式输出，按 token 消费。

完整可运行示例见官方 [Build a Minimal LLM Application](https://www.cloudwego.io/docs/eino/) 与 [EinoExamples](https://github.com/cloudwego/eino-examples) 仓库。

## 第一个 Agent：给 LLM「双手」

仅靠 Chain 只能做单轮调用。若要「让 LLM 决定是否调用工具、再根据工具结果继续」，需要 **Agent** 模式。

Eino 的 **ReAct Agent**（见 [Agent — Give Your LLM Hands](https://www.cloudwego.io/docs/eino/)）典型流程：

1. **定义工具（Tools）**  
   - 实现 `BaseTool`（或 eino-ext 中的具体 Tool 类型），每个工具有名称、描述、参数 schema。  
   - 例如：搜索、查数据库、调用 API 等。

2. **使用 ToolCallingChatModel**  
   - 使用支持工具调用的模型（如 OpenAI function calling），并注册上述工具。

3. **构建 ReAct Agent 图**  
   - 图中包含「模型节点」与「工具执行节点」，边表示「模型输出 → 工具调用」或「工具结果 → 再次进入模型」。  
   - 使用 ADK 中的 **ChatModelAgent** 或 Flow 中的 ReAct 预置流程，可自动完成「思考 → 选工具 → 执行 → 再思考」的循环。

4. **运行**  
   - 通过 Runner 或 Graph 的 `Invoke`/`Stream` 执行，输入用户问题，得到最终回答或流式输出。

这样 LLM 就具备了「动手能力」：在需要时调用外部工具并基于结果继续推理。

## 小结

- 最小应用：用 **Chain** 连接 ChatTemplate 与 ChatModel，通过 `Invoke` 或 `Stream` 调用。  
- 第一个 Agent：定义 **Tools**，使用 **ToolCallingChatModel** 和 ReAct 图（或 ChatModelAgent），实现「推理 + 工具调用」循环。  
- 更多示例见 [EinoExamples](https://github.com/cloudwego/eino-examples) 与 [Eino Quick Start](https://www.cloudwego.io/docs/eino/)。

下一节将介绍 Eino 的**组件系统**：ChatModel、ChatTemplate、Tool、Retriever 等接口与用法。
