# 7. Eino

| | |
|---|---|
| **Abstract** | Eino 入门教程 — Go 语言 LLM 应用框架 |
| **Authors** | Walter Fan |
| **Status** | WIP as draft |
| **Updated** | today |

[Eino](https://github.com/cloudwego/eino)（发音 "I know"）是 CloudWeGo 开源的 **Go 语言 LLM 应用开发框架**，强调类型安全、流式支持与可组合性。其设计借鉴 LangChain 与 Google ADK，同时提供 Go 原生的开发体验。

官方资源：

- 用户手册：[Eino User Manual \| CloudWeGo](https://www.cloudwego.io/docs/eino/)
- 源码与架构：[cloudwego/eino \| DeepWiki](https://deepwiki.com/cloudwego/eino)

## 为什么需要 Eino？

在 Go 生态中构建 LLM 应用时，通常需要：

- **统一的组件抽象** — ChatModel、Tool、Retriever、Embedder 等可替换实现
- **编排能力** — 顺序链、有状态图、工作流
- **Agent 模式** — ReAct、多 Agent、人工介入与断点恢复
- **类型安全与流式** — 编译期校验、原生流式 I/O

Eino 围绕三大支柱满足上述需求：组件抽象层、编排框架、Agent 开发套件（ADK）。

## 三支柱架构

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Eino Three-Pillar Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Pillar 1: Component Abstractions                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │ ChatModel   │ │ BaseTool    │ │ Retriever   │  ...            │
│  │ ChatTemplate│ │ Embedder   │ │ Indexer     │                │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  Pillar 2: Orchestration Framework                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │   Chain     │ │   Graph     │ │  Workflow  │                 │
│  │ (sequential)│ │ (DAG+state) │ │ (field map)│                 │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│  Unified Runtime: runner, channelManager, taskManager, checkPointer │
│                          │                                       │
│                          ▼                                       │
│  Pillar 3: Agent Development Kit (ADK)                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │ChatModelAgent│ │ DeepAgent   │ │ MultiAgent  │                │
│  │ (ReAct)     │ │ (subagents) │ │ (host/spec) │                │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| 支柱 | 主要包/实体 | 作用 |
|------|-------------|------|
| **Component Abstractions** | ChatModel, BaseTool, Retriever, Embedder, Indexer, DocumentLoader | 可复用的可替换组件 |
| **Orchestration** | Chain, Graph, Workflow, runner, channelManager, taskManager | 类型安全的编排与统一执行 |
| **ADK** | ChatModelAgent, Runner, AgentTool, DeepAgent, MultiAgent | 多步推理与工具调用、多 Agent |

## 设计原则

| 原则 | 说明 |
|------|------|
| **Type Safety** | 强类型与编译期校验组件连接 |
| **Stream-First** | 流式优先，支持自动拼接、装箱、合并、复制 |
| **Composition over Configuration** | 通过组合简单组件构建复杂行为 |
| **Transparency** | 实现可替换，用户面向抽象编程 |
| **Observability** | 内置切面与回调，便于观测 |
| **Go Idioms** | 遵循 Go 惯例而非照搬其他语言 |

## 学习目标

完成本教程后，你将能够：

- 理解 Eino 的三支柱与核心概念
- 使用 Chain / Graph / Workflow 编排组件
- 掌握流式四种范式：Invoke、Stream、Collect、Transform
- 构建 ReAct Agent 与多 Agent 系统
- 使用 ADK Runner、中断与检查点实现 Human-in-the-Loop
- 对接 Eino 生态（eino-ext、eino-examples）

## 仓库生态

- **Eino Core**（本仓库）：类型与 schema、组件接口、编排、回调、ADK、预置 Flow
- **EinoExt**：组件具体实现（OpenAI/Anthropic、向量库等）、回调实现、可视化与调试
- **EinoExamples**：示例应用与端到端用法

---

```{toctree}
---
maxdepth: 1
caption: 教程目录
---

tutorial_01_introduction
tutorial_02_quickstart
tutorial_03_components
tutorial_04_chain_graph
tutorial_05_streaming
tutorial_06_react_agent
tutorial_07_adk
tutorial_08_multi_agent
tutorial_09_human_in_loop
tutorial_10_ecosystem
```
