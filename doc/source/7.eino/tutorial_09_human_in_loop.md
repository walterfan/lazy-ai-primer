# Tutorial 9: Human-in-the-Loop

## 为什么需要人工介入

在敏感操作、合规审核、创造性决策等场景下，需要**在关键步骤暂停**，由人确认、修改或批准后再继续执行。Eino 通过 **Interrupt（中断）** 与 **CheckPoint（检查点）** 支持这类 Human-in-the-Loop 流程。

参考：[Eino Human-in-the-Loop Framework](https://www.cloudwego.io/docs/eino/)、[DeepWiki Interrupt](https://deepwiki.com/cloudwego/eino)。

## 中断与检查点

- **Interrupt**：在图或 Agent 的某个节点执行后，不继续到下一节点，而是**暂停**并返回当前状态（及可选提示信息）给调用方。  
- **CheckPoint**：将当前执行状态（如对话历史、工具结果、图状态）**持久化**，以便：  
  - 稍后从该点 **恢复** 执行；  
  - 或供人工界面展示、编辑后再触发恢复。  

这样，一次「运行」可以分成多段：运行 → 中断 → 人工处理 → 从检查点恢复 → 继续运行。

## 在 Agent 中的使用

- **ChatModelAgent** 等可与 Runner 配合：在指定步骤（如「调用某工具之后」「生成最终回答之前」）注册中断。  
- Runner 执行到该步骤时触发 Interrupt，将 CheckPoint 保存；外部系统（如 Web 后端）可：  
  - 读取状态并展示给用户；  
  - 接收用户输入（确认/修改/驳回）；  
  - 调用恢复 API，传入 CheckPoint 与可选新输入，继续执行。  

文档：[Eino ADK: Interrupt & CheckPoint Manual](https://www.cloudwego.io/docs/eino/)、[Interrupt System Architecture](https://deepwiki.com/cloudwego/eino)。

## 设计要点

1. **中断点**：明确哪些节点后需要人工介入（例如「发布前」「支付前」）。  
2. **状态暴露**：CheckPoint 中保存的内容要足够「可展示、可编辑」，便于前端或审核界面使用。  
3. **恢复策略**：恢复时是否允许修改消息、跳过某步、重试等，需与业务约定。  
4. **超时与取消**：长时间未恢复的会话可超时关闭或取消，避免资源占用。

## 小结

- Human-in-the-Loop 依赖 **Interrupt**（暂停）与 **CheckPoint**（持久化状态）。  
- Runner 与 Agent 在配置的中断点暂停并写入 CheckPoint；恢复时从 CheckPoint 继续。  
- 适合审核、确认、人工修正等需要「人参与决策」的流程。

下一节将介绍 **生态与生产**：eino-ext 集成、回调与可观测性。
