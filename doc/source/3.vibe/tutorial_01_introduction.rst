####################################
Tutorial 1: 氛围编程入门
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

============ ==========================
**Abstract** 氛围编程概念与核心理念
**Authors**  Walter Fan
**Status**   WIP
**Updated**  |date|
============ ==========================

.. contents::
   :local:

什么是氛围编程
==============

氛围编程（Vibe Coding）是 AI 时代的一种新编程范式。这个概念由 Andrej Karpathy 在 2025 年提出：

    "There's a new kind of coding I call 'vibe coding', where you fully give in to the vibes,
    embrace exponentials, and forget that the code even exists."

简单来说，氛围编程就是：

1. **与 AI 协作编程**：不再是独自敲代码，而是与 AI 助手对话式开发
2. **关注意图而非实现**：你描述想要什么，AI 帮你实现
3. **快速迭代验证**：快速生成、测试、修正，形成闭环

核心理念
========

氛围编程的核心理念可以用一句话概括：

    **"Create the right atmosphere for AI collaboration"**
    （为 AI 协作创造合适的氛围）

这意味着：

- 不是让 AI 适应你
- 也不是你迁就 AI
- 而是建立一个清晰、系统化的沟通框架

类比：带实习生
--------------

想象你在带一个聪明但没有上下文的实习生。你不会只说"写个登录功能"就走开，对吧？

你会：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 沟通内容
     - 示例
   * - 解释业务背景
     - "我们是一个电商平台，用户需要登录后才能下单"
   * - 说明技术栈
     - "我们用 Spring Boot + React，数据库是 MySQL"
   * - 指出安全要求
     - "密码需要加密存储，支持 OAuth2 登录"
   * - 告诉代码规范
     - "我们用 RESTful API，返回统一的 JSON 格式"
   * - 定义验收标准
     - "需要支持手机号和邮箱两种登录方式"

AI 也需要同样的上下文！

AI 时代的角色转变
=================

氛围编程时代，开发者的角色发生了根本性转变：

.. csv-table::
   :header: "角色", "过去的职责", "AI 时代的职责", "变化本质"
   :widths: 15, 25, 30, 30

   "开发者", "逐行编写代码", "维护和执行规格说明", "从打字员到架构师"
   "QA", "手写测试用例", "定义验收标准", "从执行者到守护者"
   "AI", "不存在", "代码生成与实现", "新的团队成员"

关键洞察
--------

.. note::

   **AI 是生产力放大器——它既放大你的清晰度，也放大你的困惑。**

   - 当你的需求清晰、规格明确时，AI 能以 10 倍速度帮你实现
   - 但如果你自己都不清楚要什么，AI 只会用 10 倍速度生成垃圾代码

氛围编程所需能力
================

氛围编程需要一套新的能力模型：

.. code-block:: text

   氛围编程能力金字塔

                    ┌─────────────┐
                    │   创新能力   │  ← 发现新问题，提出新方案
                    ├─────────────┤
                    │   决策能力   │  ← 在多方案中做出选择
                    ├─────────────┤
                    │   判断能力   │  ← 辨别 AI 输出质量
                    ├─────────────┤
                    │   提问能力   │  ← 问出好问题（最重要！）
                    ├─────────────┤
                    │   表达能力   │  ← 清晰传达技术想法
                    ├─────────────┤
                    │   编码能力   │  ← 理解和修改代码
                    └─────────────┘

核心能力详解
------------

**1. 提问能力（极高重要性）**

AI 给出什么样的答案，很大程度上取决于你问什么样的问题。

糟糕的问题::

    帮我写一个查询订单的接口

好的问题::

    我需要实现一个订单查询的 REST API，具体要求：
    1. 使用 Spring Boot 3.x + Java 17
    2. 支持按订单ID、用户ID、订单状态等多条件查询
    3. 需要分页查询，支持排序
    4. 使用 Spring Data JPA 访问 MySQL
    5. 需要参数校验
    6. 返回统一的响应格式

**2. 判断能力（极高重要性）**

AI 生成的代码不一定是最优的，你需要能够识别：

- 代码质量问题（命名、结构、可读性）
- 安全漏洞（SQL 注入、XSS 等）
- 性能问题（N+1 查询、内存泄漏）
- 架构合理性（是否符合项目规范）

**3. 决策能力（高重要性）**

AI 可能给出多种方案，你需要根据实际情况选择：

- 适用性：哪个方案更适合当前项目？
- 权衡：性能、可维护性、开发成本如何平衡？
- 风险：引入新技术的风险是否可控？

氛围编程 vs 传统编程
====================

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 维度
     - 传统编程
     - 氛围编程
   * - 编码方式
     - 手动逐行编写
     - 描述意图，AI 生成
   * - 关注点
     - 语法、API、实现细节
     - 需求、架构、验收标准
   * - 迭代速度
     - 较慢，需要手动修改
     - 快速，对话式迭代
   * - 学习曲线
     - 需要掌握语言和框架
     - 需要掌握提问和判断
   * - 适用场景
     - 所有场景
     - 快速原型、CRUD、重复性工作

第一个氛围编程示例
==================

让我们通过一个简单的例子体验氛围编程：

**场景**：创建一个 Python 脚本，统计目录下的文件类型分布

**传统方式**：你需要自己编写代码

.. code-block:: python

    import os
    from collections import Counter

    def count_file_types(directory):
        extensions = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                extensions.append(ext if ext else 'no_extension')
        return Counter(extensions)

    if __name__ == '__main__':
        result = count_file_types('.')
        for ext, count in result.most_common():
            print(f'{ext}: {count}')

**氛围编程方式**：你描述需求，AI 生成代码

::

    我需要一个 Python 脚本，功能如下：
    1. 统计指定目录下所有文件的扩展名分布
    2. 递归遍历子目录
    3. 按数量从多到少排序输出
    4. 支持命令行参数指定目录
    5. 输出格式美观，使用表格形式

AI 会生成更完善的代码，包含命令行参数解析、表格输出、错误处理等。

氛围编程工具链
==============

要进行高效的氛围编程，你需要合适的工具：

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 工具类型
     - 代表产品
     - 特点
   * - AI IDE
     - Cursor, Windsurf
     - 深度集成 AI，支持上下文感知
   * - IDE 插件
     - GitHub Copilot, Codeium
     - 在现有 IDE 中添加 AI 能力
   * - 命令行工具
     - Aider, Claude CLI
     - 终端中进行 AI 编程
   * - Web 工具
     - ChatGPT, Claude
     - 通用对话，可用于编程

本教程将以 **Cursor** 为主要示例工具，因为它：

- 专为 AI 编程设计
- 支持丰富的上下文管理
- 有强大的规则系统
- 支持 MCP 扩展

开始你的氛围编程之旅
====================

准备工作
--------

1. **安装 Cursor**

   访问 https://cursor.sh 下载并安装

2. **配置 AI 模型**

   - 打开 Settings → Models
   - 选择你偏好的模型（Claude、GPT-4 等）
   - 配置 API Key（如果使用自己的 key）

3. **熟悉基本快捷键**

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - 快捷键
        - 功能
      * - ``Cmd/Ctrl + K``
        - 内联编辑（在代码中直接修改）
      * - ``Cmd/Ctrl + L``
        - 打开聊天面板
      * - ``Cmd/Ctrl + I``
        - 打开 Composer（多文件编辑）
      * - ``Tab``
        - 接受 AI 建议

小试牛刀
--------

打开 Cursor，创建一个新文件，然后：

1. 按 ``Cmd/Ctrl + L`` 打开聊天
2. 输入：

   ::

       创建一个 Python 函数，实现斐波那契数列的第 n 项计算，
       要求：
       1. 使用递归 + 记忆化
       2. 添加类型注解
       3. 添加文档字符串
       4. 添加单元测试

3. 观察 AI 生成的代码
4. 点击 "Apply" 应用到文件

恭喜！你已经完成了第一次氛围编程。

小结
====

本教程介绍了氛围编程的基本概念：

- **氛围编程是什么**：与 AI 协作的新编程范式
- **核心理念**：为 AI 协作创造合适的氛围
- **所需能力**：提问、判断、决策、创新
- **工具选择**：Cursor 是理想的入门工具

下一步
------

在下一个教程中，我们将深入了解各种 AI 编程助手的特点和选择。

练习
====

1. 安装 Cursor 并完成基本配置
2. 使用 AI 生成一个简单的 TODO 应用
3. 对比你手写代码和 AI 生成代码的差异
4. 尝试用不同的提示词描述同一个需求，观察结果差异

参考资源
========

- `Cursor 官方文档 <https://docs.cursor.com/>`_
- `Andrej Karpathy - Vibe Coding <https://twitter.com/karpathy/status/1886192184808149383>`_
- `AI-Assisted Programming Best Practices <https://github.com/anthropics/courses>`_
