# Tutorial 4: Chain 与 Graph 编排

## 编排框架概览

Eino 的 **Pillar 2** 提供三种编排 API，在「简单」与「灵活」之间分层：

| API | 特点 | 适用场景 |
|-----|------|----------|
| **Chain** | 线性、顺序 | 简单管道：Template → Model → Parser |
| **Graph** | 有环/无环 DAG，分支与状态 | ReAct 循环、条件分支、复杂工作流 |
| **Workflow** | 无环 DAG + 字段级映射 | 精确数据变换、并行 fan-out/fan-in |

参考：[Eino Chain/Graph Orchestration](https://www.cloudwego.io/docs/eino/)、[DeepWiki Orchestration](https://deepwiki.com/cloudwego/eino)。

## Chain

- **顺序流**：节点 A → B → C，数据依次传递。  
- 典型用法：`ChatTemplate` → `ChatModel` → 解析/后处理（如 Lambda）。  
- 编译后仍由统一运行时执行，支持流式与回调。

文档：[Eino: Chain/Graph Orchestration Introduction](https://www.cloudwego.io/docs/eino/)。

## Graph

- **有向图**：支持环（如 Agent 的「思考 → 工具 → 再思考」）、分支和状态。  
- 核心能力：
  - **图构建**：定义节点与边，可条件边（根据状态选下一节点）。  
  - **状态与通道**：节点间通过 channel/state 传递数据。  
  - **编译与类型系统**：编译期校验连接与类型。  
  - **执行引擎**：runner、channelManager、taskManager、checkPointer。  
  - **分支与条件执行**：根据当前状态选择下一跳。  

文档：[Orchestration Design Principles](https://www.cloudwego.io/docs/eino/)、[Graph Construction](https://deepwiki.com/cloudwego/eino)。

## Workflow

- **无环 DAG** + **字段级映射**：每个节点声明输入/输出字段，边描述字段如何从一节点映射到下一节点。  
- 适合：
  - 多输入、多输出的数据流；  
  - 并行 fan-out（一路输入多路处理）与 fan-in（多路汇总）；  
  - 需要明确「哪个字段到哪个字段」的 ETL 式流水线。  

文档：[Eino: Workflow Orchestration Framework](https://www.cloudwego.io/docs/eino/)、[Workflow Dependencies](https://deepwiki.com/cloudwego/eino)。

## 统一执行运行时

三种 API 编译到同一套引擎（`compose/graph_run.go`）：

- **类型检查**：编译时校验节点间数据类型与连接。  
- **流式**：自动处理流的拼接、装箱、复制、合并。  
- **并发**：taskManager 管理并发任务。  
- **状态与检查点**：channelManager 管理数据流；checkPointer 支持暂停/恢复。  

因此，即使用 Chain 写的简单管道，也享有与复杂 Graph 相同的运行时能力（流式、回调、检查点等）。

## 小结

- **Chain**：线性顺序，适合简单管道。  
- **Graph**：有环/无环 DAG，支持状态、分支与循环，适合 Agent 与复杂工作流。  
- **Workflow**：无环 DAG + 字段映射，适合结构化数据流与并行。  
- 三者共享**统一运行时**，具备类型安全、流式、并发与检查点。

下一节将介绍 **流式四种范式**：Invoke、Stream、Collect、Transform。
