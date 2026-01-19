####################################
Tutorial 8: AI 辅助代码审查
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

============ ==========================
**Abstract** 使用 AI 进行代码审查
**Authors**  Walter Fan
**Status**   WIP
**Updated**  |date|
============ ==========================

.. contents::
   :local:

为什么用 AI 做代码审查
======================

代码审查是软件开发的重要环节，但传统方式有一些问题：

- **耗时**: 审查者需要花大量时间理解代码
- **主观**: 不同审查者标准不一
- **疲劳**: 审查质量随时间下降
- **延迟**: 等待审查者有空

AI 辅助代码审查可以：

- **即时反馈**: 提交后立即获得反馈
- **一致标准**: 始终按照相同标准审查
- **全面覆盖**: 检查安全、性能、风格等多个维度
- **学习工具**: 从 AI 的建议中学习

AI 代码审查的维度
=================

.. code-block:: text

   AI 代码审查维度

   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                 │
   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
   │   │  代码质量   │  │   安全性    │  │   性能      │            │
   │   ├─────────────┤  ├─────────────┤  ├─────────────┤            │
   │   │ • 可读性    │  │ • SQL 注入  │  │ • 时间复杂度│            │
   │   │ • 命名规范  │  │ • XSS       │  │ • 空间复杂度│            │
   │   │ • 代码重复  │  │ • 认证授权  │  │ • 数据库查询│            │
   │   │ • 复杂度    │  │ • 敏感数据  │  │ • 内存泄漏  │            │
   │   └─────────────┘  └─────────────┘  └─────────────┘            │
   │                                                                 │
   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
   │   │  最佳实践   │  │   测试覆盖  │  │   文档      │            │
   │   ├─────────────┤  ├─────────────┤  ├─────────────┤            │
   │   │ • 设计模式  │  │ • 单元测试  │  │ • 注释      │            │
   │   │ • 错误处理  │  │ • 边界测试  │  │ • Docstring │            │
   │   │ • 日志记录  │  │ • 异常测试  │  │ • README    │            │
   │   └─────────────┘  └─────────────┘  └─────────────┘            │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘

在 Cursor 中进行代码审查
========================

基本审查
--------

选中代码或打开文件，使用 Chat 进行审查：

::

    @current_file

    请审查这段代码，关注以下方面：
    1. 代码质量和可读性
    2. 潜在的 bug
    3. 性能问题
    4. 安全漏洞
    5. 最佳实践

专项审查
--------

**安全审查**::

    @current_file

    作为安全专家，请审查这段代码的安全问题：
    1. SQL 注入
    2. XSS 攻击
    3. 认证/授权问题
    4. 敏感数据处理
    5. 输入验证

**性能审查**::

    @current_file

    作为性能优化专家，请审查这段代码：
    1. 时间复杂度
    2. 空间复杂度
    3. 数据库查询效率
    4. 缓存使用
    5. 并发处理

**架构审查**::

    @folder/

    作为架构师，请审查这个模块的设计：
    1. 职责是否单一
    2. 依赖是否合理
    3. 接口设计是否清晰
    4. 是否易于测试
    5. 是否易于扩展

代码审查 Checklist
==================

通用 Checklist
--------------

.. code-block:: markdown

    ## 代码审查 Checklist

    ### 功能正确性
    - [ ] 代码实现了预期功能
    - [ ] 边界条件处理正确
    - [ ] 错误处理完善

    ### 代码质量
    - [ ] 命名清晰有意义
    - [ ] 函数/方法长度适中（< 50 行）
    - [ ] 圈复杂度合理（< 10）
    - [ ] 没有重复代码
    - [ ] 注释适当

    ### 安全性
    - [ ] 输入已验证
    - [ ] 无 SQL 注入风险
    - [ ] 无 XSS 风险
    - [ ] 敏感数据已加密
    - [ ] 权限检查正确

    ### 性能
    - [ ] 无 N+1 查询
    - [ ] 适当使用缓存
    - [ ] 无内存泄漏风险
    - [ ] 大数据集有分页

    ### 测试
    - [ ] 有单元测试
    - [ ] 测试覆盖关键路径
    - [ ] 测试边界条件

    ### 文档
    - [ ] 公共 API 有文档
    - [ ] 复杂逻辑有注释
    - [ ] README 已更新

Python 专项 Checklist
---------------------

::

    请根据以下 checklist 审查 Python 代码：

    ### Python 特定
    - [ ] 使用类型注解
    - [ ] 遵循 PEP 8
    - [ ] 使用 f-string 而非 % 或 format
    - [ ] 使用 pathlib 而非 os.path
    - [ ] 使用 context manager 管理资源
    - [ ] 异常处理具体（不用 bare except）
    - [ ] 使用 dataclass 或 Pydantic
    - [ ] import 顺序正确

JavaScript/TypeScript 专项 Checklist
------------------------------------

::

    请根据以下 checklist 审查 TypeScript 代码：

    ### TypeScript 特定
    - [ ] 类型定义完整（无 any）
    - [ ] 使用 const/let，不用 var
    - [ ] 使用 async/await，不用回调
    - [ ] 使用可选链 ?.
    - [ ] 使用空值合并 ??
    - [ ] 组件 props 有类型定义
    - [ ] 正确处理 Promise 错误
    - [ ] 避免不必要的 re-render

实战：审查示例代码
==================

示例 1：发现安全问题
--------------------

**待审查代码**:

.. code-block:: python

    def get_user(user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        result = db.execute(query)
        return result.fetchone()

    def login(request):
        username = request.form['username']
        password = request.form['password']
        user = get_user_by_name(username)
        if user and user.password == password:
            session['user_id'] = user.id
            return redirect('/dashboard')
        return "Login failed"

**审查提示词**::

    请审查这段代码的安全问题

**AI 审查结果**:

.. code-block:: text

    发现以下安全问题：

    1. SQL 注入（严重）
       位置：get_user 函数
       问题：直接拼接 SQL 字符串
       修复：使用参数化查询

    2. 明文密码比较（严重）
       位置：login 函数
       问题：密码没有哈希
       修复：使用 bcrypt 比较

    3. 会话固定攻击风险
       位置：login 函数
       问题：登录成功后没有重新生成 session
       修复：登录后调用 session.regenerate()

    4. 信息泄露
       位置：login 函数
       问题："Login failed" 没有区分用户不存在和密码错误
       建议：统一返回 "Invalid credentials"

**修复后的代码**:

.. code-block:: python

    from werkzeug.security import check_password_hash

    def get_user(user_id: int) -> User | None:
        query = "SELECT * FROM users WHERE id = :id"
        result = db.execute(query, {"id": user_id})
        return result.fetchone()

    def login(request):
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        if not username or not password:
            return "Invalid credentials", 401

        user = get_user_by_name(username)
        if user and check_password_hash(user.password_hash, password):
            session.regenerate()
            session['user_id'] = user.id
            return redirect('/dashboard')

        return "Invalid credentials", 401

示例 2：发现性能问题
--------------------

**待审查代码**:

.. code-block:: python

    def get_order_details(order_ids: list[int]) -> list[dict]:
        results = []
        for order_id in order_ids:
            order = Order.query.get(order_id)
            user = User.query.get(order.user_id)
            items = OrderItem.query.filter_by(order_id=order_id).all()
            products = []
            for item in items:
                product = Product.query.get(item.product_id)
                products.append({
                    'name': product.name,
                    'quantity': item.quantity
                })
            results.append({
                'order_id': order.id,
                'user_name': user.name,
                'products': products
            })
        return results

**审查提示词**::

    请审查这段代码的性能问题，假设 order_ids 可能有 1000 个

**AI 审查结果**:

.. code-block:: text

    发现严重的 N+1 查询问题：

    对于 1000 个订单，这段代码会执行：
    - 1000 次 Order 查询
    - 1000 次 User 查询
    - 1000 次 OrderItem 查询
    - N 次 Product 查询（N = 订单项总数）

    总计可能超过 5000 次数据库查询！

    优化建议：
    1. 使用 eager loading 一次性加载关联数据
    2. 使用 IN 查询批量获取
    3. 考虑使用 joinedload 或 selectinload

**修复后的代码**:

.. code-block:: python

    from sqlalchemy.orm import joinedload, selectinload

    def get_order_details(order_ids: list[int]) -> list[dict]:
        # 一次查询获取所有数据
        orders = (
            Order.query
            .filter(Order.id.in_(order_ids))
            .options(
                joinedload(Order.user),
                selectinload(Order.items).joinedload(OrderItem.product)
            )
            .all()
        )

        return [
            {
                'order_id': order.id,
                'user_name': order.user.name,
                'products': [
                    {
                        'name': item.product.name,
                        'quantity': item.quantity
                    }
                    for item in order.items
                ]
            }
            for order in orders
        ]

    # 现在只需要 3 次查询：
    # 1. Orders
    # 2. Users (joinedload)
    # 3. OrderItems + Products (selectinload + joinedload)

自动化代码审查
==============

Git Hook 集成
-------------

创建 pre-commit hook，在提交前进行 AI 审查：

.. code-block:: bash

    #!/bin/bash
    # .git/hooks/pre-commit

    # 获取暂存的文件
    FILES=$(git diff --cached --name-only --diff-filter=ACMR | grep -E '\.(py|js|ts)$')

    if [ -z "$FILES" ]; then
        exit 0
    fi

    echo "Running AI code review..."

    # 使用 aider 或其他工具进行审查
    for FILE in $FILES; do
        echo "Reviewing $FILE..."
        # 这里可以调用 AI API 进行审查
    done

CI/CD 集成
----------

在 GitHub Actions 中集成 AI 审查：

.. code-block:: yaml

    # .github/workflows/ai-review.yml
    name: AI Code Review

    on:
      pull_request:
        types: [opened, synchronize]

    jobs:
      review:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v3

          - name: Get changed files
            id: changed-files
            uses: tj-actions/changed-files@v35

          - name: AI Review
            env:
              OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
            run: |
              for file in ${{ steps.changed-files.outputs.all_changed_files }}; do
                echo "Reviewing $file"
                # 调用 AI API 进行审查
              done

          - name: Post Review Comments
            uses: actions/github-script@v6
            with:
              script: |
                // 将审查结果作为 PR 评论发布

审查反馈处理
============

理解 AI 建议
------------

AI 的建议分为几个级别：

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 级别
     - 描述
     - 处理方式
   * - 🔴 Critical
     - 安全漏洞、严重 bug
     - 必须修复
   * - 🟠 Major
     - 性能问题、设计缺陷
     - 应该修复
   * - 🟡 Minor
     - 代码风格、命名问题
     - 建议修复
   * - 🟢 Suggestion
     - 改进建议
     - 可选

判断建议质量
------------

AI 的建议不一定都是正确的，需要判断：

1. **理解上下文**: AI 可能不了解完整的业务背景
2. **权衡利弊**: 有些建议可能引入新的复杂性
3. **验证建议**: 对于重要修改，要验证正确性
4. **保持批判**: 不要盲目接受所有建议

常见误判
--------

- **过度优化**: 建议优化不是瓶颈的代码
- **过度抽象**: 建议创建不必要的抽象
- **忽略约束**: 建议使用项目中不允许的库
- **风格偏好**: 将个人偏好作为问题

代码审查对话
============

多轮对话进行深入审查：

**第一轮**::

    @file.py
    请审查这段代码

**第二轮**::

    你提到了 SQL 注入问题，能详细解释一下攻击场景吗？

**第三轮**::

    修复后的代码如下，请再次审查：
    [粘贴修复后的代码]

**第四轮**::

    除了安全问题，这段代码的测试应该怎么写？

小结
====

本教程介绍了 AI 辅助代码审查：

- **审查维度**: 质量、安全、性能、最佳实践
- **审查方法**: 基本审查、专项审查、Checklist
- **实战案例**: 安全问题、性能问题
- **自动化**: Git Hook、CI/CD 集成

关键要点：

1. AI 审查是补充，不是替代人工审查
2. 要判断 AI 建议的质量
3. 建立团队的审查 Checklist

下一步
------

在下一个教程中，我们将学习 AI 辅助测试和调试。

练习
====

1. 用 AI 审查你最近写的代码
2. 创建你的代码审查 Checklist
3. 设置 pre-commit hook 进行自动审查
4. 对比 AI 审查和人工审查的结果

参考资源
========

- `Google Code Review Guidelines <https://google.github.io/eng-practices/review/>`_
- `OWASP Code Review Guide <https://owasp.org/www-project-code-review-guide/>`_
- `Clean Code by Robert C. Martin <https://www.oreilly.com/library/view/clean-code-a/9780136083238/>`_
