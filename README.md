# Artificial Intelligence Primer

## 🎯 What's AI

> ``The exciting new effort to make computers think ... machines with minds, in the full and literal sense'' (Haugeland, 1985)

> ``The automation of activities that we associate with human thinking, activities such as decision-making, problem solving, learning ...'' (Bellman, 1978)

> ``The study of mental faculties through the use of computational models'' (Charniak and McDermott, 1985)

> ``The study of the computations that make it possible to perceive, reason, and act'' (Winston, 1992)

> ``The art of creating machines that perform functions that require intelligence when performed by people'' (Kurzweil, 1990)

> ``The study of how to make computers do things at which, at the moment, people are better'' (Rich and Knight, 1991)

> ``A field of study that seeks to explain and emulate intelligent behavior in terms of computational processes'' (Schalkoff, 1990)

> ``The branch of computer science that is concerned with the automation of intelligent behavior'' (Luger and Stubblefield, 1993)


- 从智能的角度来看

  人工智能使机器变得 "智能" -- 按照我们所期望的以人类的方式行事。

  无法区分计算机响应和人类响应的情况被称为图灵测试。

- 从研究角度来看

  "人工智能是研究如何让计算机去做目前人类做得更好的事情"[Rich and Knight，1991，p.3]

  人工智能始于 20 世纪 60 年代初——最初的尝试是玩游戏（跳棋）、定理证明（一些简单的定理）和一般问题解决（仅非常简单的任务）

  其研究领域包括：

    - 正式任务（数学、游戏），
    - 日常任务（感知、机器人、自然语言、常识推理）
    - 专家任务（财务分析、医学诊断、工程、科学分析和其他领域）
    - 等等

- 从商业角度来看

    人工智能是一套非常强大的工具，以及使用这些工具解决业务问题的方法。

- 从编程的角度来看

    人工智能包括符号编程、问题解决和搜索的研究, 包括图像与音视频的识别与生成, 自然语言处理, 机器学习, 神经网络, 机器人, 等等

---

## 📚 人工智能入门教程 - 从基础概念到实战应用

### 1. 人工智能基础 (AI Fundamentals)

参考《人工智能：现代方法》，介绍 AI 基本概念，使用 PyTorch 实践。

| 教程 | 主题 | 内容 |
|------|------|------|
| [Tutorial 1](doc/source/1.basic/tutorial_01_what_is_ai.rst) | 什么是人工智能 | AI 定义、历史、主要流派 |
| [Tutorial 2](doc/source/1.basic/tutorial_02_intelligent_agents.rst) | 智能体 | Agent 概念、类型、设计 |
| [Tutorial 3](doc/source/1.basic/tutorial_03_search_algorithms.rst) | 搜索算法 | BFS、DFS、A* 算法 |
| [Tutorial 4](doc/source/1.basic/tutorial_04_knowledge_representation.rst) | 知识表示与推理 | 逻辑、语义网络、知识图谱 |
| [Tutorial 5](doc/source/1.basic/tutorial_05_machine_learning.rst) | 机器学习基础 | 监督/无监督/强化学习 |
| [Tutorial 6](doc/source/1.basic/tutorial_06_neural_networks.rst) | 神经网络 | 感知机、MLP、反向传播 |
| [Tutorial 7](doc/source/1.basic/tutorial_07_pytorch_deep_learning.rst) | PyTorch 深度学习 | 张量、自动微分、训练流程 |
| [Tutorial 8](doc/source/1.basic/tutorial_08_nlp_fundamentals.rst) | 自然语言处理 | 词嵌入、RNN、Transformer |
| [Tutorial 9](doc/source/1.basic/tutorial_09_computer_vision.rst) | 计算机视觉 | CNN、图像分类、目标检测 |
| [Tutorial 10](doc/source/1.basic/tutorial_10_reinforcement_learning.rst) | 强化学习 | Q-Learning、DQN、策略梯度 |

### 2. RAG 检索增强生成 (Retrieval-Augmented Generation)

让大语言模型利用外部知识库生成更准确的回答。

| 教程 | 主题 | 内容 |
|------|------|------|
| [Tutorial 1](doc/source/2.rag/tutorial_01_introduction.rst) | RAG 入门 | 概念、原理、与微调对比 |
| [Tutorial 2](doc/source/2.rag/tutorial_02_document_loading.rst) | 文档加载 | PDF、Word、网页等格式 |
| [Tutorial 3](doc/source/2.rag/tutorial_03_text_chunking.rst) | 文本分块 | 字符、语义、特定格式分块 |
| [Tutorial 4](doc/source/2.rag/tutorial_04_embeddings.rst) | 向量嵌入 | 嵌入模型、相似度计算 |
| [Tutorial 5](doc/source/2.rag/tutorial_05_vector_database.rst) | 向量数据库 | Chroma、FAISS、索引优化 |
| [Tutorial 6](doc/source/2.rag/tutorial_06_retrieval_strategies.rst) | 检索策略 | MMR、混合检索、重排序 |
| [Tutorial 7](doc/source/2.rag/tutorial_07_prompt_engineering.rst) | Prompt 工程 | RAG 专用 Prompt 设计 |
| [Tutorial 8](doc/source/2.rag/tutorial_08_evaluation.rst) | RAG 评估 | 检索/生成指标、端到端评估 |
| [Tutorial 9](doc/source/2.rag/tutorial_09_advanced_rag.rst) | 高级 RAG | 查询改写、分解、自我反思 |
| [Tutorial 10](doc/source/2.rag/tutorial_10_production.rst) | 生产部署 | FastAPI、Docker、监控 |

### 3. 氛围编程 (Vibe Coding)

AI 辅助编程的新范式，与 AI 协作的高效开发方法。

| 教程 | 主题 | 内容 |
|------|------|------|
| [Tutorial 1](doc/source/3.vibe/tutorial_01_introduction.rst) | 氛围编程入门 | 概念、核心理念、能力模型 |
| [Tutorial 2](doc/source/3.vibe/tutorial_02_ai_assistants.rst) | AI 编程助手 | Cursor、Copilot、Aider 对比 |
| [Tutorial 3](doc/source/3.vibe/tutorial_03_cursor_commands.rst) | Cursor 命令 | Chat、Inline Edit、Composer |
| [Tutorial 4](doc/source/3.vibe/tutorial_04_rules_config.rst) | Rules 配置 | .cursorrules、项目规则 |
| [Tutorial 5](doc/source/3.vibe/tutorial_05_prompting_skills.rst) | 提示词技巧 | RACE 框架、高级技巧 |
| [Tutorial 6](doc/source/3.vibe/tutorial_06_mcp_server.rst) | MCP Server | 扩展 AI 能力、自定义开发 |
| [Tutorial 7](doc/source/3.vibe/tutorial_07_spec_driven_dev.rst) | 规格驱动开发 | SDD 方法论、规格模板 |
| [Tutorial 8](doc/source/3.vibe/tutorial_08_code_review.rst) | AI 代码审查 | 审查维度、自动化集成 |
| [Tutorial 9](doc/source/3.vibe/tutorial_09_testing_debugging.rst) | 测试与调试 | 测试生成、错误分析 |
| [Tutorial 10](doc/source/3.vibe/tutorial_10_best_practices.rst) | 最佳实践 | 工作流、协作、持续改进 |

### 4. LangChain 入门

构建 LLM 应用的框架。

| 教程 | 主题 | 内容 |
|------|------|------|
| [Tutorial 1](doc/source/4.langchain/tutorial_01_introduction.rst) | LangChain 入门 | 概念、安装、基本用法 |
| [Tutorial 2](doc/source/4.langchain/tutorial_02_llm_chat_models.rst) | LLM 与 Chat Models | 模型调用、参数配置 |
| [Tutorial 3](doc/source/4.langchain/tutorial_03_prompt_templates.rst) | Prompt Templates | 模板设计、变量注入 |
| [Tutorial 4](doc/source/4.langchain/tutorial_04_chains.rst) | Chains | 链式调用、组合链 |
| [Tutorial 5](doc/source/4.langchain/tutorial_05_memory.rst) | Memory | 对话记忆、上下文管理 |
| [Tutorial 6](doc/source/4.langchain/tutorial_06_agents_tools.rst) | Agents & Tools | 智能体、工具调用 |
| [Tutorial 7](doc/source/4.langchain/tutorial_07_rag.rst) | RAG 实现 | 检索增强生成实战 |
| [Tutorial 8](doc/source/4.langchain/tutorial_08_content_agent.rst) | 内容创作 Agent | 自媒体内容生成 |
| [Tutorial 9](doc/source/4.langchain/tutorial_09_publishing_agent.rst) | 发布 Agent | 多平台发布自动化 |
| [Tutorial 10](doc/source/4.langchain/tutorial_10_complete_workflow.rst) | 完整工作流 | 端到端自媒体系统 |

### 5. LangGraph 入门

构建有状态、多步骤 AI 应用的框架。

| 教程 | 主题 | 内容 |
|------|------|------|
| [Tutorial 1](doc/source/5.langgraph/tutorial_01_introduction.rst) | LangGraph 入门 | 核心概念、与 LangChain 对比 |
| [Tutorial 2](doc/source/5.langgraph/tutorial_02_state_graph.rst) | State 与 Graph | 状态定义、更新机制 |
| [Tutorial 3](doc/source/5.langgraph/tutorial_03_nodes_edges.rst) | Nodes 与 Edges | 节点类型、边的定义 |
| [Tutorial 4](doc/source/5.langgraph/tutorial_04_conditional_routing.rst) | 条件路由 | 动态路径选择、LLM 路由 |
| [Tutorial 5](doc/source/5.langgraph/tutorial_05_cycles_loops.rst) | 循环与迭代 | 迭代优化、重试机制 |
| [Tutorial 6](doc/source/5.langgraph/tutorial_06_human_in_loop.rst) | Human-in-the-Loop | 人工干预、审核流程 |
| [Tutorial 7](doc/source/5.langgraph/tutorial_07_persistence.rst) | 持久化与检查点 | SQLite/PostgreSQL、状态恢复 |
| [Tutorial 8](doc/source/5.langgraph/tutorial_08_multi_agent.rst) | 多 Agent 协作 | Supervisor 模式、团队协作 |
| [Tutorial 9](doc/source/5.langgraph/tutorial_09_content_workflow.rst) | 自媒体内容工作流 | 完整实战案例 |
| [Tutorial 10](doc/source/5.langgraph/tutorial_10_production.rst) | 生产部署 | FastAPI、Docker、监控 |

### 6. LlamaIndex 入门

连接 LLM 与外部数据的强大框架，专注于 RAG 和知识库构建。

| 教程 | 主题 | 内容 |
|------|------|------|
| [Tutorial 1](doc/source/6.llamaindex/tutorial_01_introduction.rst) | LlamaIndex 入门 | 核心概念、Document/Node/Index |
| [Tutorial 2](doc/source/6.llamaindex/tutorial_02_data_loading.rst) | 数据加载 | PDF、数据库、Web、云存储 |
| [Tutorial 3](doc/source/6.llamaindex/tutorial_03_node_parsing.rst) | 节点解析 | 文本分割、语义分块、层次化 |
| [Tutorial 4](doc/source/6.llamaindex/tutorial_04_embeddings_vectorstore.rst) | 嵌入与向量存储 | Chroma、FAISS、Milvus |
| [Tutorial 5](doc/source/6.llamaindex/tutorial_05_index_types.rst) | 索引类型 | Vector、Summary、Keyword、Tree |
| [Tutorial 6](doc/source/6.llamaindex/tutorial_06_query_engine.rst) | 查询引擎 | 响应模式、流式输出、自定义 |
| [Tutorial 7](doc/source/6.llamaindex/tutorial_07_retrieval_strategies.rst) | 检索策略 | 混合检索、重排序、句子窗口 |
| [Tutorial 8](doc/source/6.llamaindex/tutorial_08_agents_tools.rst) | Agents 与 Tools | ReAct Agent、工具开发 |
| [Tutorial 9](doc/source/6.llamaindex/tutorial_09_knowledge_base.rst) | 构建知识库系统 | 企业级知识库完整实现 |
| [Tutorial 10](doc/source/6.llamaindex/tutorial_10_production.rst) | 生产部署 | Docker、缓存、监控、安全 |

---

## 📖 推荐阅读

- 《人工智能：现代方法》- Stuart Russell, Peter Norvig
- 《深度学习》- Ian Goodfellow, Yoshua Bengio, Aaron Courville
- 《动手学深度学习》- 李沐等

## 🔗 相关链接

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [Hugging Face](https://huggingface.co/)

## 📝 License

CC0-1.0 license
