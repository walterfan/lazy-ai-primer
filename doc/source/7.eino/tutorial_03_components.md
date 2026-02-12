# Tutorial 3: 组件系统

## 组件概述

Eino 的 **Pillar 1** 是组件抽象层：每种组件都有清晰的 **接口**（Input / Output / Option）、**执行方式**（如 `Generate`/`Invoke`、`Stream`）以及可选的 **回调** 支持。实现可替换，调用方只依赖接口。

参考：[Eino Components](https://www.cloudwego.io/docs/eino/)、[DeepWiki Core Concepts - Components](https://deepwiki.com/cloudwego/eino)。

## 主要组件类型

### ChatModel / ToolCallingChatModel

- **ChatModel**：与 LLM 交互，输入消息列表，输出回复（或流）。  
- **ToolCallingChatModel**：在对话之外支持 **工具调用**（function calling），用于 ReAct 等 Agent。

实现通常来自 eino-ext，例如：

- OpenAI、Claude、Gemini、DeepSeek、Qwen、Ollama、Qianfan 等  
- AgenticModel（如 ARK）用于更复杂的 Agent 能力  

文档：[Eino: ChatModel Guide](https://www.cloudwego.io/docs/eino/)、Ecosystem 中的 ChatModel 集成。

### ChatTemplate

- 将「用户输入 + 可选的系统提示、历史等」格式化为 LLM 所需的消息列表（如 system/user/assistant）。  
- 保证与各模型的消息格式（如 OpenAI、Claude）一致。

文档：[Eino: ChatTemplate Guide](https://www.cloudwego.io/docs/eino/)。

### BaseTool、ToolsNode

- **BaseTool**：单个工具的抽象，包含名称、描述、参数定义；支持 `InvokableTool`（同步）或 `StreamableTool`（流式）。  
- **ToolsNode**：在 Graph 中执行一组工具的节点，根据模型输出的 tool_calls 调用对应 Tool，并把结果填回消息。

创建自定义工具见 [How to Create a Tool](https://www.cloudwego.io/docs/eino/)，配置与执行见 [ToolsNode & Tool Guide](https://www.cloudwego.io/docs/eino/)。  
Eino-ext 提供多种现成 Tool：DuckDuckGo/Bing/Google 搜索、Wikipedia、HTTP、MCP、Commandline、Browseruse 等。

### Retriever

- 给定查询，返回一组相关文档或片段（通常带分数或元数据）。  
- 用于 RAG：先检索再拼进 Prompt 或上下文。

eino-ext 提供多种 Retriever：Elasticsearch 7/8/9、Milvus v1/v2、OpenSearch、Redis、Volc VikingDB、Dify、Volc Knowledge 等。  
文档：[Eino: Retriever Guide](https://www.cloudwego.io/docs/eino/).

### Embedder / Indexer

- **Embedder**：文本 → 向量。  
- **Indexer**：写入/查询向量索引（如 Milvus、Elasticsearch、Redis、VikingDB）。

用于 RAG 的「建库」与「检索」链路。  
文档：[Eino: Indexer Guide](https://www.cloudwego.io/docs/eino/)、[Eino: Embedding Guide](https://www.cloudwego.io/docs/eino/)。  
Ecosystem 中有 ARK、OpenAI、DashScope、Ollama、Qianfan、TencentCloud 等 Embedding 集成。

### DocumentLoader / DocumentTransformer

- **DocumentLoader**：从本地文件、S3、URL 等加载原始文档。  
- **DocumentTransformer**：对文档做切分、过滤等（如 RecursiveSplitter、SemanticSplitter、MarkdownSplitter）。  
- **Document Parser**：解析 HTML、PDF 等格式。

文档：[Eino: Document Loader Guide](https://www.cloudwego.io/docs/eino/)、[Document Parser](https://www.cloudwego.io/docs/eino/)、[Document Transformer](https://www.cloudwego.io/docs/eino/)。

### Lambda

- 在编排图中插入 **自定义函数** 节点：输入 → 业务逻辑 → 输出。  
- 用于简单变换、过滤、聚合等，无需单独实现某类组件接口。

文档：[Eino: Lambda Guide](https://www.cloudwego.io/docs/eino/)。

## 组件在编排中的使用

- **Chain**：按顺序连接若干组件（如 Template → Model → Lambda 解析）。  
- **Graph**：将组件作为节点，用边和状态连接；例如 Model 节点 + ToolsNode + Lambda。  
- **Workflow**：无环 DAG + 字段级映射，适合「多输入多输出」的数据流。

下一节将介绍 **Chain、Graph、Workflow** 的编排能力与统一运行时。

## 小结

- Eino 组件包括：ChatModel、ChatTemplate、BaseTool/ToolsNode、Retriever、Embedder/Indexer、DocumentLoader/Transformer、Lambda。  
- 实现集中在 eino-ext，通过接口可替换。  
- 组件通过 Chain/Graph/Workflow 编排，形成完整应用。
