# Tutorial 7: ADK — Agent 开发套件

## ADK 概述

**Agent Development Kit（ADK）** 是 Eino 的 **Pillar 3**：在编排框架之上提供高层 Agent 抽象，便于实现多步推理、工具调用、多 Agent 协作和人工介入。

参考：[Eino ADK Overview](https://www.cloudwego.io/docs/eino/)、[DeepWiki ADK](https://deepwiki.com/cloudwego/eino)。

## 核心概念

### Agent 接口

- ADK 定义统一的 **Agent 接口**（输入/输出、执行方式），不同实现（如 ChatModelAgent、Workflow Agents、MultiAgent）都遵循该接口。  
- 便于在图中将「整个 Agent」当作一个节点使用（例如通过 **AgentTool** 把 Agent 包装成 Tool，被其他 Agent 调用）。

文档：[Eino ADK: Agent Interface](https://www.cloudwego.io/docs/eino/)。

### Runner 与扩展

- **Runner** 负责 Agent 的**生命周期**：启动运行、推进步骤、处理中断与恢复。  
- 支持 **检查点**：将状态持久化，之后可从某一步恢复（用于 Human-in-the-Loop 或故障恢复）。  
- 通过 **扩展**（如中间件、回调）可以注入日志、监控、限流、权限等。

文档：[Eino ADK: Agent Runner & Extensions](https://www.cloudwego.io/docs/eino/)。

### ChatModelAgent

- **ChatModelAgent** 是 ReAct 模式的「开箱即用」实现：  
  - 内部是「ChatModel + 工具循环」的图；  
  - 支持中间件（如注入工具、修改上下文）；  
  - 支持中断与恢复（与 Runner、CheckPoint 配合）。  

文档：[Eino ADK: ChatModelAgent](https://www.cloudwego.io/docs/eino/)。

### Workflow Agents

- 基于 **Workflow** 编排的 Agent：用字段级映射描述数据流，适合「多步骤、多输入输出」的固定流程。  
- 可与 ChatModel、Tool、Retriever 等组件组合。

文档：[Eino ADK: Workflow Agents](https://www.cloudwego.io/docs/eino/)。

## 执行模式与中断

- **正常执行**：Runner 驱动 Agent 一步步执行直到结束。  
- **中断（Interrupt）**：在配置的节点（如「需要人工确认」）暂停，将当前状态写入 CheckPoint；恢复时从该状态继续。  
- 用于审核、人工修正、多轮确认等 Human-in-the-Loop 场景。

文档：[Eino ADK: Interrupt & CheckPoint Manual](https://www.cloudwego.io/docs/eino/)、[Interrupt System Architecture](https://deepwiki.com/cloudwego/eino)。

## 小结

- ADK 提供 Agent 接口、Runner、ChatModelAgent、Workflow Agents 等高层抽象。  
- Runner 管理生命周期与检查点；ChatModelAgent 实现 ReAct；AgentTool 支持 Agent 嵌套与组合。  
- 中断与检查点为实现 Human-in-the-Loop 提供基础。

下一节将介绍 **多 Agent**：MultiAgent、DeepAgent、Host-Specialist 等模式。
