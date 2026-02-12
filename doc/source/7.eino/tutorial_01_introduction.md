# Tutorial 1: Eino 简介

## 什么是 Eino？

**Eino**（发音 "I know"）是 [CloudWeGo](https://www.cloudwego.io/) 开源的、用 **Go 语言** 编写的 LLM 应用开发框架。它围绕三个核心支柱组织：

1. **组件抽象层（Component Abstractions）** — 可复用的构建块与清晰接口  
2. **编排框架（Orchestration Framework）** — 将组件连接起来的三种方式：Chain、Graph、Workflow  
3. **Agent 开发套件（ADK）** — 基于编排的高层 Agent 模式

框架强调 **类型安全**、**流式优先** 和 **可组合性**，设计上借鉴 [LangChain](https://python.langchain.com/) 与 [Google ADK](https://github.com/google/adk)，同时保持 Go 惯用写法。

参考：[Eino Overview (DeepWiki)](https://deepwiki.com/cloudwego/eino)、[Eino User Manual](https://www.cloudwego.io/docs/eino/)。

## 三大支柱概览

### 1. 组件抽象层

可替换的构建块，每种组件都有明确的 **Input / Output / Option** 与执行方式（如 `Generate`/`Invoke`、`Stream`）：

| 组件类型 | 作用 |
|----------|------|
| `ChatModel` / `ToolCallingChatModel` | 与 LLM 交互 |
| `BaseTool`（`InvokableTool`、`StreamableTool`） | 工具调用 |
| `ChatTemplate` | 消息格式化 |
| `Retriever` | 文档检索 |
| `Embedder` / `Indexer` | 向量与索引 |
| `DocumentLoader` / `DocumentTransformer` | 文档加载与转换 |
| `Lambda` | 自定义函数节点 |

实现对用户透明：例如任何需要 `Retriever` 的地方都可以使用 `MultiQueryRetriever` 等具体实现。

### 2. 编排框架

三种编排 API，在「简单」与「强大」之间权衡：

| API | 特点 | 典型用法 |
|-----|------|----------|
| **Chain** | 线性顺序 | 简单管道：Template → Model → Parser |
| **Graph** | 有环/无环 DAG，分支与状态 | ReAct 循环、条件路由、复杂工作流 |
| **Workflow** | 无环 DAG + 字段级映射 | 精确数据变换、并行 fan-out/fan-in |

三者最终都编译到 **同一套执行运行时**（`compose/graph_run.go`），包括：

- **runner** — 主执行循环  
- **channelManager** — 数据流  
- **taskManager** — 并发任务  
- **checkPointer** — 状态持久化与恢复  

因此具备统一的类型检查、流式、并发与检查点能力。

### 3. Agent 开发套件（ADK）

在编排之上提供高层 Agent 抽象：

- **ChatModelAgent** — ReAct 模式，自动工具调用循环  
- **DeepAgent** — 任务编排与子 Agent 委托  
- **MultiAgent** — 主机-专家等多 Agent 模式  
- **Runner** — Agent 生命周期与检查点  
- **AgentTool** — 将 Agent 封装为工具以便组合  

## 设计哲学

| 原则 | 描述 |
|------|------|
| **Type Safety** | 全链路强类型，组件连接在编译期校验 |
| **Stream-First** | 原生流式，支持自动拼接、装箱、合并、复制 |
| **Composition over Configuration** | 用组合简单组件代替复杂配置 |
| **Transparency** | 面向接口编程，实现可替换 |
| **Observability** | 内置切面与回调，便于观测与调试 |
| **Go Idioms** | 遵循 Go 习惯，而非直接移植其他语言模式 |

## 仓库生态

- **Eino Core**（[github.com/cloudwego/eino](https://github.com/cloudwego/eino)）：类型与 schema、组件接口、编排、回调、ADK、预置 Flow  
- **EinoExt**（[github.com/cloudwego/eino-ext](https://github.com/cloudwego/eino-ext)）：OpenAI/Anthropic 等模型、向量库、回调实现、可视化与调试工具  
- **EinoExamples**（[github.com/cloudwego/eino-examples](https://github.com/cloudwego/eino-examples)）：示例应用与最佳实践  

## 小结

- Eino 是 Go 语言的 LLM 应用框架，强调类型安全、流式与可组合。  
- 三支柱：组件抽象、编排（Chain/Graph/Workflow）、ADK。  
- 编排统一到同一运行时，支持流式、并发与检查点。  
- 官方文档：[CloudWeGo Eino](https://www.cloudwego.io/docs/eino/)，架构细节见 [DeepWiki Eino](https://deepwiki.com/cloudwego/eino)。

下一节将介绍如何快速搭建一个最小 LLM 应用与第一个 Agent。
