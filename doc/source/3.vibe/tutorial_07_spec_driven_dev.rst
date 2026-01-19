####################################
Tutorial 7: 规格驱动开发
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

============ ==========================
**Abstract** Spec-Driven Development 方法论
**Authors**  Walter Fan
**Status**   WIP
**Updated**  |date|
============ ==========================

.. contents::
   :local:

什么是规格驱动开发
==================

规格驱动开发（Spec-Driven Development，SDD）是氛围编程时代的一种结构化开发方法论。

核心思想：**规格（Spec）是信任之源，先写规格，再让 AI 实现**。

.. code-block:: text

   传统开发 vs 规格驱动开发

   传统开发:
   需求 → 设计 → 编码 → 测试 → 部署
              ↑
           开发者手写代码

   规格驱动开发:
   需求 → 规格 → 计划 → 任务 → 实现 → 验证
              ↑
           开发者写规格，AI 生成代码

在 SDD 中，规格不只是参考文档，而是实现和测试的**唯一依据**。所有代码、测试、文档都围绕规格展开。

为什么需要 SDD
--------------

1. **避免 Vibe 编程的常见问题**: 功能模糊、架构偏差、安全缺失
2. **提高一致性**: 跨团队和 AI Agent 协作中保持可追溯性
3. **保证代码质量**: 在大型项目中保持长期可维护性
4. **规格即文档**: 规格永远不会过时，因为它就是实现的依据

SDD 四阶段工作流
================

SDD 采用四阶段工作流：**Specify → Plan → Tasks → Implement**

.. code-block:: text

   SDD 四阶段工作流

   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                 │
   │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
   │   │  Specify    │ ──▶ │    Plan     │ ──▶ │   Tasks     │      │
   │   │  定义规格   │     │  技术计划   │     │  拆分任务   │      │
   │   └─────────────┘     └─────────────┘     └─────────────┘      │
   │                                                  │              │
   │                                                  ▼              │
   │                                           ┌─────────────┐      │
   │                                           │  Implement  │      │
   │                                           │  逐步实现   │      │
   │                                           └─────────────┘      │
   │                                                  │              │
   │         ┌────────────────────────────────────────┘              │
   │         │ 发现问题                                              │
   │         ▼                                                       │
   │   ┌─────────────┐                                              │
   │   │   更新规格  │ ◀── 反馈循环                                  │
   │   └─────────────┘                                              │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘

阶段 1：Specify（定义规格）
---------------------------

**目的**: 明确 **什么** 要做（What）和 **为什么** 做（Why）

规格文档应包含：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 内容
     - 说明
   * - 目标（Goal）
     - 这个功能要解决什么问题？
   * - 功能需求（FR）
     - 系统必须做什么？
   * - 非功能需求（NFR）
     - 性能、安全性、可用性等约束
   * - 禁止项（Do Not）
     - 明确不要做什么，避免歧义
   * - 验收标准（AC）
     - 如何判断功能完成？

**Specify 的关键原则**:

- **具体而非模糊**: "用户可以登录" → "用户使用邮箱和密码登录，成功返回 JWT"
- **包含禁止项**: "不要使用第三方 OAuth"、"不要存储明文密码"
- **可验证**: 每个需求都应该能转化为测试

阶段 2：Plan（技术计划）
------------------------

**目的**: 确定 **怎么** 做（How）

技术计划应包含：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 内容
     - 说明
   * - 技术栈
     - 语言、框架、数据库等选择
   * - 架构设计
     - 模块划分、组件关系
   * - 接口契约
     - API 设计、数据格式
   * - 数据模型
     - 数据库 Schema 设计
   * - 风险分析
     - 技术风险和应对策略

**Plan 的关键原则**:

- **尽早确定技术决策**: 不要在实现阶段才考虑架构
- **定义清晰的接口**: 模块间的契约要明确
- **考虑约束**: 性能、安全、成本等限制

阶段 3：Tasks（拆分任务）
-------------------------

**目的**: 将计划拆分为 **可执行、可验证** 的小任务

任务拆分原则：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 原则
     - 说明
   * - 小而独立
     - 每个任务对应一个 PR 或功能片段
   * - 可验证
     - 每个任务有明确的验收标准
   * - 依赖清晰
     - 任务间的依赖关系和优先级明确
   * - 可估时
     - 每个任务的工作量可预估

**任务示例**::

    Task 1: 创建用户数据模型
    - 验收标准: User 模型包含 email, password_hash, created_at 字段
    - 依赖: 无
    - 预估: 0.5h

    Task 2: 实现注册 API
    - 验收标准: POST /register 返回 201 和用户信息
    - 依赖: Task 1
    - 预估: 1h

    Task 3: 添加密码验证
    - 验收标准: 密码少于 8 位返回 400 错误
    - 依赖: Task 2
    - 预估: 0.5h

阶段 4：Implement（逐步实现）
-----------------------------

**目的**: 按任务顺序实现，每步都验证

实现原则：

1. **严格依照规格**: 不要偏离规格文档
2. **测试驱动**: 每个验收标准对应测试用例
3. **小步提交**: 每次提交可审查，不大幅偏离规格
4. **反馈循环**: 发现规格问题，及时更新规格

SDD 核心命令
============

在 Cursor 或其他 AI 编程工具中，可以使用以下命令模式：

.. csv-table::
   :header: "命令", "作用", "使用时机"
   :widths: 20, 40, 40

   "/specify", "撰写规格文档：定义目标、需求、验收标准", "项目/功能启动阶段"
   "/plan", "从规格生成技术计划：架构、接口、数据模型", "规格审阅完毕后"
   "/tasks", "把计划拆分为小任务，带验收标准和依赖", "设计完成后"
   "/implement", "按任务实现，参照标准进行测试验证", "任务分配后"

命令使用示例
------------

**/specify 命令**::

    /specify

    我要实现一个用户通知系统，请帮我写规格文档：
    - 目标：用户可以接收系统通知
    - 渠道：站内信、邮件、短信
    - 约束：每分钟最多发送 100 条通知

**/plan 命令**::

    /plan

    @specs/notification.md

    请根据这个规格生成技术计划：
    - 技术栈选择
    - 架构设计
    - 数据库设计
    - API 设计

**/tasks 命令**::

    /tasks

    @specs/notification.md @plans/notification-plan.md

    请将计划拆分为可执行的任务列表，每个任务包含：
    - 任务描述
    - 验收标准
    - 依赖关系
    - 预估时间

**/implement 命令**::

    /implement

    @tasks/notification-tasks.md

    请实现 Task 1: 创建通知数据模型
    - 遵循规格中的字段定义
    - 添加类型注解
    - 编写单元测试

SDD vs Vibe Coding
==================

.. csv-table::
   :header: "对比维度", "Vibe Coding", "Spec-Driven Development"
   :widths: 20, 40, 40

   "起点", "Prompt 或即兴需求", "规格（Spec）先行"
   "结构性", "灵活/松散，经常重构", "严格结构：规格→计划→任务→实现"
   "适合场景", "快速原型、小功能探索", "生产环境、大项目、团队协作"
   "质量与可预测性", "不稳定，依赖经验", "稳定、明确、容易验证"
   "可维护性", "随意，技术债易积累", "高，可追踪、可维护"
   "AI 协作效率", "需要多轮修正", "一次生成质量更高"

何时使用 SDD
------------

**适合 SDD 的场景**:

- 生产环境的功能开发
- 团队协作项目
- 需要长期维护的系统
- 有明确验收标准的需求
- 安全性要求高的功能

**适合 Vibe Coding 的场景**:

- 快速原型验证
- 探索性开发
- 个人小工具
- 一次性脚本

实战：用户认证功能
==================

让我们通过一个完整的例子来演示 SDD 四阶段工作流。

Step 1：Specify（定义规格）
---------------------------

创建 ``specs/auth.md``:

.. code-block:: markdown

    # Feature: User Authentication

    ## 1. Goal（目标）
    实现安全可靠的用户认证系统，支持注册、登录、登出功能。

    ## 2. Functional Requirements（功能需求）

    ### 2.1 Registration (FR-REG)
    - FR-REG-001: User can register with email and password
    - FR-REG-002: Email must be unique
    - FR-REG-003: Password must be at least 8 characters with letters and numbers

    ### 2.2 Login (FR-LOGIN)
    - FR-LOGIN-001: User can login with email and password
    - FR-LOGIN-002: System returns JWT token on successful login
    - FR-LOGIN-003: Account locks after 5 failed attempts

    ### 2.3 Logout (FR-LOGOUT)
    - FR-LOGOUT-001: User can logout and token is invalidated

    ## 3. Non-Functional Requirements（非功能需求）
    - NFR-001: Login response time < 200ms
    - NFR-002: Passwords stored with bcrypt (cost factor 12)
    - NFR-003: JWT expires in 1 hour

    ## 4. Do Not（禁止项）
    - DO-NOT-001: Do not store plain text passwords
    - DO-NOT-002: Do not use third-party OAuth (phase 1)
    - DO-NOT-003: Do not log sensitive data (passwords, tokens)

    ## 5. Acceptance Criteria（验收标准）
    - AC-REG-001: Given valid email/password, When register, Then account created
    - AC-REG-002: Given existing email, When register, Then error returned
    - AC-LOGIN-001: Given valid credentials, When login, Then JWT returned
    - AC-LOGIN-002: Given 5 failed attempts, When login again, Then account locked

Step 2：Plan（技术计划）
------------------------

创建 ``plans/auth-plan.md``:

.. code-block:: markdown

    # Technical Plan: User Authentication

    ## 1. Tech Stack
    - Framework: FastAPI
    - Database: PostgreSQL
    - ORM: SQLAlchemy 2.0
    - JWT: python-jose
    - Password: bcrypt

    ## 2. Architecture

    ```
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Router    │ ──▶ │   Service   │ ──▶ │ Repository  │
    │ (auth.py)   │     │ (auth_svc)  │     │  (user_repo)│
    └─────────────┘     └─────────────┘     └─────────────┘
                              │
                              ▼
                        ┌─────────────┐
                        │   Utils     │
                        │ (jwt, hash) │
                        └─────────────┘
    ```

    ## 3. API Design

    ### POST /api/v1/auth/register
    - Request: { email, password, name }
    - Response: { user_id, email, name }

    ### POST /api/v1/auth/login
    - Request: { email, password }
    - Response: { access_token, refresh_token, expires_in }

    ### POST /api/v1/auth/logout
    - Headers: Authorization: Bearer <token>
    - Response: { message }

    ## 4. Data Model

    ### User Table
    - id: UUID (PK)
    - email: String (unique, indexed)
    - password_hash: String
    - name: String
    - failed_login_attempts: Integer (default: 0)
    - locked_until: DateTime (nullable)
    - created_at: DateTime
    - updated_at: DateTime

Step 3：Tasks（拆分任务）
-------------------------

创建 ``tasks/auth-tasks.md``:

.. code-block:: markdown

    # Tasks: User Authentication

    ## Task 1: Create User Model
    - Description: Create SQLAlchemy User model
    - Acceptance: Model has all fields from plan
    - Dependencies: None
    - Estimate: 0.5h

    ## Task 2: Implement Password Utils
    - Description: Create hash and verify functions
    - Acceptance: bcrypt with cost factor 12
    - Dependencies: None
    - Estimate: 0.5h

    ## Task 3: Implement JWT Utils
    - Description: Create token generation and validation
    - Acceptance: 1 hour expiry, proper claims
    - Dependencies: None
    - Estimate: 0.5h

    ## Task 4: Create User Repository
    - Description: CRUD operations for User
    - Acceptance: create, get_by_email, update methods
    - Dependencies: Task 1
    - Estimate: 1h

    ## Task 5: Implement Auth Service
    - Description: Business logic for auth
    - Acceptance: register, login, logout methods
    - Dependencies: Task 2, 3, 4
    - Estimate: 2h

    ## Task 6: Create Auth Router
    - Description: FastAPI endpoints
    - Acceptance: All 3 endpoints working
    - Dependencies: Task 5
    - Estimate: 1h

    ## Task 7: Write Tests
    - Description: Unit and integration tests
    - Acceptance: All AC-* covered
    - Dependencies: Task 6
    - Estimate: 2h

Step 4：Implement（逐步实现）
-----------------------------

按任务顺序，让 AI 逐个实现：

**实现 Task 1**::

    @specs/auth.md @plans/auth-plan.md @tasks/auth-tasks.md

    请实现 Task 1: Create User Model

    要求：
    - 使用 SQLAlchemy 2.0 风格
    - 包含所有计划中的字段
    - 添加类型注解

**验证并继续 Task 2**::

    Task 1 完成，请继续实现 Task 2: Implement Password Utils

    要求：
    - 使用 bcrypt
    - cost factor = 12
    - 包含 hash_password 和 verify_password 函数

常见陷阱与解决方案
==================

.. csv-table::
   :header: "陷阱", "问题", "解决方案"
   :widths: 25, 35, 40

   "规格太模糊", "AI 生成的代码不符合预期", "在 Specify 阶段多问 Why/Do Not"
   "技术决策太晚", "实现阶段频繁返工", "在 Plan 阶段尽早确定技术栈"
   "任务太大", "难以验证和审查", "拆分到每个任务 < 2 小时"
   "实现偏离规格", "最终产物与需求不符", "每个任务完成后对照验收标准"
   "规格不更新", "规格与代码脱节", "发现问题及时更新规格"

SDD 最佳实践
============

1. 规格是"活"的文档
--------------------

当需求或环境变化时，应同步更新规格/计划/任务：

::

    # 规格变更记录
    ## [v1.1] - 2024-01-15
    - Changed: FR-REG-003 密码要求从 8 位改为 10 位
    - Added: FR-LOGIN-004 支持验证码登录

2. 验收标准落到测试
--------------------

每个验收标准都应该有对应的测试用例：

.. code-block:: python

    # AC-LOGIN-001: Given valid credentials, When login, Then JWT returned
    def test_login_with_valid_credentials():
        # Given
        user = create_user(email="test@example.com", password="Pass123!")

        # When
        response = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "Pass123!"
        })

        # Then
        assert response.status_code == 200
        assert "access_token" in response.json()["data"]

3. 建立团队规范（Constitution）
-------------------------------

定义团队的"宪法"，作为所有规格的基础约束：

.. code-block:: markdown

    # Team Constitution

    ## Code Style
    - Python: PEP 8 + Black
    - Type hints required

    ## Security
    - No plain text passwords
    - All inputs validated
    - SQL injection prevention

    ## Testing
    - Minimum 80% coverage
    - All AC must have tests

    ## Documentation
    - Public APIs documented
    - Complex logic commented

4. 使用工具辅助
---------------

推荐使用以下工具管理 SDD 流程：

- **Spec-Kit**: 规格模板和管理工具
- **Cursor Rules**: 将规范写入 .cursorrules
- **GitHub Issues**: 任务跟踪
- **pytest**: 验收测试

小结
====

本教程介绍了规格驱动开发（SDD）的完整方法论：

- **四阶段工作流**: Specify → Plan → Tasks → Implement
- **核心命令**: /specify, /plan, /tasks, /implement
- **与 Vibe Coding 对比**: SDD 更适合生产环境和团队协作
- **最佳实践**: 活文档、测试驱动、团队规范

关键要点：

1. **规格是信任之源**: 所有实现都围绕规格展开
2. **小步快跑**: 任务要小、可验证、可追踪
3. **持续反馈**: 发现问题及时更新规格
4. **好的规格 = 好的代码**: 投入在规格上的时间会在实现阶段收回

下一步
------

在下一个教程中，我们将学习如何使用 AI 进行代码审查。

练习
====

1. 为一个简单功能完成 SDD 四阶段
2. 使用 /specify 命令生成规格文档
3. 将规格中的验收标准转化为测试
4. 建立你的团队 Constitution

参考资源
========

- `OpenAPI Specification <https://swagger.io/specification/>`_
- `Gherkin Syntax <https://cucumber.io/docs/gherkin/>`_
- `API Design Guidelines <https://github.com/microsoft/api-guidelines>`_
- `Spec-Kit <https://github.com/spec-kit/spec-kit>`_
