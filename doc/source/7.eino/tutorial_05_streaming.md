# Tutorial 5: 流式处理

## 流式范式概览

Eino 的编排图支持四种执行范式，覆盖「输入/输出是否为流」的四种组合：

| 范式 | 输入 | 输出 | 说明 |
|------|------|------|------|
| **Invoke** | I | O | 非流输入 → 非流输出 |
| **Stream** | I | StreamReader[O] | 非流输入 → 流式输出 |
| **Collect** | StreamReader[I] | O | 流式输入 → 非流输出（收集后一次返回） |
| **Transform** | StreamReader[I] | StreamReader[O] | 流式输入 → 流式输出 |

参考：[Eino Streaming Essentials](https://www.cloudwego.io/docs/eino/)、[DeepWiki Streaming Paradigms](https://deepwiki.com/cloudwego/eino)。

## 框架的流式行为

运行时会对流做自动处理，使组件可以按「非流」接口编写，仍参与流式图：

- **Concatenate**：下游需要「完整数据」时才消费（如 ToolsNode 需收齐 tool_calls 再执行），流会被拼接后再传。  
- **Box**：非流输出被「装箱」成单元素流，以便与流式节点衔接。  
- **Copy**：同一流可被复制给多个消费者（如回调与下游节点）。  
- **Merge**：多路流汇入同一节点时合并。

因此你可以混合「流式节点」与「非流式节点」，由框架保证数据流正确。

## 使用场景

- **Invoke**：同步调用，适合简单查询、批量任务。  
- **Stream**：LLM 逐 token 输出、日志/监控实时展示。  
- **Collect**：从外部流式源（如 SSE）读入，聚合成一条消息再进模型。  
- **Transform**：管道式处理流（如解码 → 过滤 → 再编码）。

文档：[Stream Processing Internals](https://deepwiki.com/cloudwego/eino)。

## 小结

- 四种范式：Invoke / Stream / Collect / Transform，覆盖输入输出是否流式。  
- 运行时负责拼接、装箱、复制、合并，便于混合流式与非流式节点。  
- 详细实现见 [Eino Streaming Essentials](https://www.cloudwego.io/docs/eino/) 与 DeepWiki。

下一节将介绍 **ReAct Agent**：在图中实现「推理 + 工具调用」循环。
