# Tutorial 8: 多 Agent 系统

## 多 Agent 场景

当任务需要**分工协作**（如规划、执行、审核、专家子任务）时，可以用多个 Agent 配合完成：  
一个 Agent 负责任务分解与调度，其他 Agent 负责具体执行或专业领域，彼此通过**上下文传递**或**工具封装**（Agent 当 Tool）协作。

Eino ADK 与 Flow 提供多种多 Agent 模式。  
参考：[Eino ADK: Agent Collaboration](https://www.cloudwego.io/docs/eino/)、[DeepWiki Multi-Agent](https://deepwiki.com/cloudwego/eino)。

## MultiAgent 与 Host-Specialist

- **MultiAgent**：在一个「主机」协调下，将请求路由到多个**专家 Agent** 之一或按流程依次经过多个 Agent。  
- **Host**：负责理解意图、选择专家或编排步骤。  
- **Specialist**：各司其职（如搜索、写代码、查数据库），可复用 ChatModelAgent、Workflow Agent 等。

文档：[Eino Tutorial: Host Multi-Agent](https://www.cloudwego.io/docs/eino/)、[Multi-Agent Transfers](https://deepwiki.com/cloudwego/eino)。

## 实现方式

1. **Agent 作为节点**：在 Graph 中，每个 Agent 是一个节点；边表示「谁把上下文传给谁」。  
2. **AgentTool**：将某个 Agent 包装成 `BaseTool`，这样另一个 Agent（如 Host）在需要时「调用」该 Tool，即把子任务交给对应 Agent，并拿到结果继续推理。  
3. **上下文与状态**：通过图的 state/channel 在 Agent 之间传递消息、工具结果或自定义结构。

## DeepAgent

- **DeepAgent** 是一种「任务编排 + 子 Agent 委托」的模式：  
  - 顶层 Agent 负责拆解任务、选择执行者；  
  - 子 Agent（可嵌套）负责具体执行；  
  - 内置 **Task Tool** 与子 Agent 管理，便于实现「规划 → 分配 → 执行 → 汇总」的流程。  

文档：[Eino ADK MultiAgent: DeepAgents](https://www.cloudwego.io/docs/eino/)、[DeepWiki DeepAgent](https://deepwiki.com/cloudwego/eino)。

## 其他模式

- **Plan-Execute Agent**：一个 Agent 做规划（输出步骤），另一个按步骤执行并回报。  
- **Supervisor Agent**：单一 Supervisor 将请求分发给多个 Worker Agent，并汇总结果。  

文档：[Eino ADK MultiAgent: Plan-Execute](https://www.cloudwego.io/docs/eino/)、[Supervisor Agent](https://www.cloudwego.io/docs/eino/)。

## 小结

- 多 Agent 通过 Host-Specialist、AgentTool、Graph 中的 Agent 节点协作。  
- DeepAgent 提供任务编排与子 Agent 委托；Plan-Execute、Supervisor 等是常见模式。  
- 上下文与状态由图的 state/channel 和 Runner 的检查点统一管理。

下一节将介绍 **Human-in-the-Loop**：中断、检查点与恢复。
