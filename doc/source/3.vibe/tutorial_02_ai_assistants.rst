####################################
Tutorial 2: AI 编程助手
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

============ ==========================
**Abstract** AI 编程助手的类型与选择
**Authors**  Walter Fan
**Status**   WIP
**Updated**  |date|
============ ==========================

.. contents::
   :local:

AI 编程助手概览
===============

AI 编程助手是氛围编程的核心工具。它们可以分为几大类：

.. code-block:: text

   AI 编程助手分类

   ┌─────────────────────────────────────────────────────────────────┐
   │                        AI 编程助手                               │
   ├─────────────────┬─────────────────┬─────────────────────────────┤
   │   AI-Native IDE │   IDE 插件      │      命令行 / Web 工具       │
   ├─────────────────┼─────────────────┼─────────────────────────────┤
   │ • Cursor        │ • GitHub Copilot│ • Aider                     │
   │ • Windsurf      │ • Codeium       │ • Claude CLI                │
   │ • Zed           │ • Tabnine       │ • ChatGPT / Claude Web      │
   │ • Void          │ • Amazon Q      │ • Replit Agent              │
   └─────────────────┴─────────────────┴─────────────────────────────┘

AI-Native IDE
=============

这类工具从设计之初就以 AI 为核心，提供最深度的 AI 集成体验。

Cursor
------

**官网**: https://cursor.sh

Cursor 是目前最流行的 AI-Native IDE，基于 VS Code 构建。

**核心特性**：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 特性
     - 说明
   * - Chat
     - 与 AI 对话，支持 @ 引用文件、符号、文档
   * - Composer
     - 多文件编辑，AI 可同时修改多个文件
   * - Inline Edit (Cmd+K)
     - 在代码中直接编辑，无需切换窗口
   * - Tab 补全
     - 智能代码补全，支持多行补全
   * - Rules
     - 自定义规则，让 AI 遵循你的编码规范
   * - MCP
     - 支持 Model Context Protocol 扩展

**适用场景**：

- 日常开发工作
- 快速原型开发
- 代码重构
- 学习新技术

**示例：使用 Cursor Chat**

::

    @main.py @utils.py
    请帮我重构这两个文件：
    1. 将重复的数据库连接代码提取到 utils.py
    2. 添加连接池支持
    3. 添加重试机制

Windsurf
--------

**官网**: https://codeium.com/windsurf

Windsurf 是 Codeium 推出的 AI-Native IDE。

**核心特性**：

- **Cascade**: 多步骤自动化，AI 可以自主完成复杂任务
- **Flows**: 预定义的工作流模板
- **深度代码理解**: 强大的代码索引和搜索

**特点**：

- 更强调 AI 的自主性
- 适合需要 AI 完成大量自动化工作的场景

Zed
---

**官网**: https://zed.dev

Zed 是一个高性能的代码编辑器，由 Atom 的创始团队开发。

**核心特性**：

- **极致性能**: 使用 Rust 编写，启动和响应极快
- **协作编辑**: 原生支持多人实时协作
- **AI 集成**: 内置 AI 助手

**特点**：

- 性能优先
- 适合追求速度的开发者

IDE 插件
========

这类工具作为插件集成到现有 IDE 中，让你在熟悉的环境中使用 AI。

GitHub Copilot
--------------

**官网**: https://github.com/features/copilot

GitHub Copilot 是最早的 AI 编程助手之一，由 GitHub 和 OpenAI 合作开发。

**核心特性**：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 特性
     - 说明
   * - 代码补全
     - 实时的代码建议
   * - Copilot Chat
     - 在 IDE 中与 AI 对话
   * - Copilot Edits
     - 多文件编辑能力
   * - CLI
     - 命令行中使用 Copilot

**支持的 IDE**：

- VS Code
- Visual Studio
- JetBrains IDEs
- Neovim
- Xcode

**示例：使用 Copilot Chat**

.. code-block:: text

    /explain 这段代码的作用是什么？

    /fix 修复这个 bug

    /tests 为这个函数生成测试

Codeium
-------

**官网**: https://codeium.com

Codeium 提供免费的 AI 代码补全服务。

**核心特性**：

- **免费使用**: 个人用户完全免费
- **广泛支持**: 支持 70+ 编程语言
- **多 IDE 支持**: VS Code、JetBrains、Vim 等

**特点**：

- 对个人开发者友好
- 补全质量不错

Amazon Q Developer
------------------

**官网**: https://aws.amazon.com/q/developer/

Amazon 推出的 AI 编程助手，特别擅长 AWS 相关开发。

**核心特性**：

- **AWS 专家**: 深度理解 AWS 服务
- **安全扫描**: 自动检测安全漏洞
- **代码转换**: 支持 Java 版本升级等

**适用场景**：

- AWS 云开发
- Java 项目现代化

命令行工具
==========

对于喜欢终端的开发者，命令行 AI 工具是很好的选择。

Aider
-----

**官网**: https://aider.chat

Aider 是一个强大的命令行 AI 编程工具。

**核心特性**：

.. code-block:: bash

    # 安装
    pip install aider-chat

    # 基本使用
    aider

    # 指定文件
    aider main.py utils.py

    # 使用特定模型
    aider --model claude-3-5-sonnet

**常用命令**：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 命令
     - 说明
   * - ``/add <file>``
     - 添加文件到对话上下文
   * - ``/drop <file>``
     - 从上下文移除文件
   * - ``/diff``
     - 显示待应用的更改
   * - ``/commit``
     - 提交更改到 Git
   * - ``/undo``
     - 撤销上一次更改

**特点**：

- Git 集成良好
- 支持多种 AI 模型
- 适合终端爱好者

Claude CLI
----------

Anthropic 官方的命令行工具。

.. code-block:: bash

    # 安装
    npm install -g @anthropic-ai/claude-cli

    # 使用
    claude "帮我写一个 Python 函数计算斐波那契数列"

选择合适的工具
==============

选择 AI 编程助手时，考虑以下因素：

决策矩阵
--------

.. csv-table::
   :header: "考虑因素", "Cursor", "Copilot", "Codeium", "Aider"
   :widths: 20, 20, 20, 20, 20

   "价格", "$20/月", "$10-19/月", "免费", "免费（需 API）"
   "IDE 绑定", "Cursor", "多 IDE", "多 IDE", "终端"
   "上下文管理", "★★★★★", "★★★☆☆", "★★★☆☆", "★★★★☆"
   "多文件编辑", "★★★★★", "★★★★☆", "★★★☆☆", "★★★★☆"
   "自定义规则", "★★★★★", "★★☆☆☆", "★★☆☆☆", "★★★☆☆"
   "学习曲线", "低", "低", "低", "中"

推荐场景
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 场景
     - 推荐工具
   * - 全职开发者，追求效率
     - Cursor
   * - 已有 IDE 习惯，不想换
     - GitHub Copilot
   * - 预算有限的个人开发者
     - Codeium
   * - 终端爱好者
     - Aider
   * - AWS 开发
     - Amazon Q Developer

AI 模型选择
===========

不同的 AI 编程助手使用不同的底层模型：

主流模型对比
------------

.. csv-table::
   :header: "模型", "提供商", "特点", "适用场景"
   :widths: 20, 20, 30, 30

   "Claude 3.5 Sonnet", "Anthropic", "代码能力强，上下文长", "复杂编程任务"
   "GPT-4o", "OpenAI", "通用能力强", "多样化任务"
   "Claude 3 Opus", "Anthropic", "推理能力最强", "架构设计、复杂问题"
   "GPT-4o-mini", "OpenAI", "快速、便宜", "简单任务、快速迭代"
   "DeepSeek Coder", "DeepSeek", "开源、代码专精", "代码生成、补全"

模型选择建议
------------

1. **日常开发**: Claude 3.5 Sonnet 或 GPT-4o
2. **复杂架构设计**: Claude 3 Opus
3. **快速迭代**: GPT-4o-mini
4. **预算有限**: DeepSeek Coder

实践：配置你的开发环境
======================

步骤 1：安装 Cursor
-------------------

1. 访问 https://cursor.sh
2. 下载对应系统的安装包
3. 安装并启动

步骤 2：配置模型
----------------

1. 打开 Settings（Cmd/Ctrl + ,）
2. 选择 Models
3. 配置你偏好的模型：

   - 默认模型：Claude 3.5 Sonnet（推荐）
   - 快速模型：GPT-4o-mini
   - 高级模型：Claude 3 Opus

步骤 3：导入 VS Code 配置
-------------------------

如果你之前使用 VS Code：

1. Cursor 会自动检测并提示导入
2. 或者手动：Settings → General → Import VS Code Settings

步骤 4：安装必要扩展
--------------------

推荐安装的扩展：

- Python（如果使用 Python）
- ESLint（JavaScript/TypeScript）
- Prettier（代码格式化）
- GitLens（Git 增强）

小结
====

本教程介绍了主流的 AI 编程助手：

- **AI-Native IDE**: Cursor、Windsurf、Zed
- **IDE 插件**: GitHub Copilot、Codeium、Amazon Q
- **命令行工具**: Aider、Claude CLI

关键要点：

1. 选择工具要考虑：价格、功能、学习曲线
2. Cursor 是目前最全面的 AI 编程工具
3. 模型选择影响输出质量

下一步
------

在下一个教程中，我们将深入学习 Cursor 的各种命令和快捷键。

练习
====

1. 安装至少两种 AI 编程助手，对比它们的体验
2. 在 Cursor 中尝试不同的 AI 模型，感受差异
3. 使用 Aider 完成一个简单的编程任务
4. 记录你最喜欢的工具和原因

参考资源
========

- `Cursor 官方文档 <https://docs.cursor.com/>`_
- `GitHub Copilot 文档 <https://docs.github.com/en/copilot>`_
- `Aider 文档 <https://aider.chat/docs/>`_
- `AI 模型对比 <https://artificialanalysis.ai/>`_
