################
5. LangGraph
################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

============ ==========================
**Abstract** LangGraph 入门教程
**Authors**  Walter Fan
**Status**   WIP as draft
**Updated**  |date|
============ ==========================

LangGraph 是 LangChain 团队开发的用于构建有状态、多步骤 AI 应用的框架。
它基于图（Graph）的概念，让你能够构建复杂的 Agent 工作流。

为什么需要 LangGraph？
======================

LangChain 的 AgentExecutor 虽然强大，但在以下场景中有局限性：

- 需要精确控制 Agent 执行流程
- 需要实现复杂的多 Agent 协作
- 需要支持人工干预（Human-in-the-Loop）
- 需要持久化状态和断点恢复

LangGraph 通过图结构解决了这些问题，提供了：

- **显式的流程控制** - 用节点和边定义工作流
- **状态管理** - 在节点之间传递和修改状态
- **条件路由** - 根据状态动态选择下一步
- **循环支持** - 支持迭代和重试逻辑
- **持久化** - 支持状态保存和恢复

学习目标
========

完成本教程后，你将能够：

- 理解 LangGraph 的核心概念（State、Node、Edge）
- 构建有状态的 Agent 工作流
- 实现条件路由和循环逻辑
- 添加人工干预点
- 实现多 Agent 协作系统
- 部署生产级 LangGraph 应用

实战项目：自媒体 AI 工作流
==========================

我们将构建一个完整的自媒体内容生产系统：

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                 Self-Media Content Workflow                  │
   │                                                              │
   │   [话题研究] ──► [内容策划] ──► [内容创作] ──► [审核修改]    │
   │        │              │              │              │        │
   │        ▼              ▼              ▼              ▼        │
   │   [热点分析]    [大纲生成]    [文章撰写]    [人工审核]      │
   │                                              │              │
   │                                              ▼              │
   │                                        [多平台发布]         │
   │                                              │              │
   │                                              ▼              │
   │                                        [数据追踪]          │
   └─────────────────────────────────────────────────────────────┘

.. toctree::
   :maxdepth: 1
   :caption: 教程目录:

   tutorial_01_introduction
   tutorial_02_state_graph
   tutorial_03_nodes_edges
   tutorial_04_conditional_routing
   tutorial_05_cycles_loops
   tutorial_06_human_in_loop
   tutorial_07_persistence
   tutorial_08_multi_agent
   tutorial_09_content_workflow
   tutorial_10_production
