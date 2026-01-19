####################################
Tutorial 9: AI 辅助测试与调试
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

============ ==========================
**Abstract** 使用 AI 进行测试和调试
**Authors**  Walter Fan
**Status**   WIP
**Updated**  |date|
============ ==========================

.. contents::
   :local:

AI 辅助测试
===========

测试是软件质量的保障，AI 可以帮助：

- **生成测试用例**: 根据代码自动生成测试
- **发现边界情况**: 识别可能遗漏的测试场景
- **提高覆盖率**: 针对未覆盖代码生成测试
- **维护测试**: 代码变更后更新测试

生成单元测试
============

基本用法
--------

选中函数，让 AI 生成测试：

::

    @function

    请为这个函数生成 pytest 测试用例：
    1. 正常情况
    2. 边界条件
    3. 异常情况
    4. 使用参数化测试

示例：测试计算函数
------------------

**待测试代码**:

.. code-block:: python

    def calculate_discount(price: float, discount_percent: float) -> float:
        """Calculate discounted price.

        Args:
            price: Original price (must be positive)
            discount_percent: Discount percentage (0-100)

        Returns:
            Discounted price

        Raises:
            ValueError: If price is negative or discount is out of range
        """
        if price < 0:
            raise ValueError("Price cannot be negative")
        if not 0 <= discount_percent <= 100:
            raise ValueError("Discount must be between 0 and 100")

        return price * (1 - discount_percent / 100)

**提示词**::

    请为 calculate_discount 函数生成完整的 pytest 测试

**生成的测试**:

.. code-block:: python

    import pytest
    from mymodule import calculate_discount


    class TestCalculateDiscount:
        """Tests for calculate_discount function."""

        # 正常情况测试
        @pytest.mark.parametrize("price,discount,expected", [
            (100, 10, 90),      # 10% off
            (100, 0, 100),      # No discount
            (100, 100, 0),      # 100% off
            (50, 25, 37.5),     # 25% off
            (99.99, 10, 89.991),  # Decimal price
        ])
        def test_valid_inputs(self, price, discount, expected):
            """Test with valid inputs."""
            result = calculate_discount(price, discount)
            assert result == pytest.approx(expected)

        # 边界条件测试
        def test_zero_price(self):
            """Test with zero price."""
            assert calculate_discount(0, 50) == 0

        def test_boundary_discount_zero(self):
            """Test with 0% discount (boundary)."""
            assert calculate_discount(100, 0) == 100

        def test_boundary_discount_hundred(self):
            """Test with 100% discount (boundary)."""
            assert calculate_discount(100, 100) == 0

        # 异常情况测试
        def test_negative_price_raises_error(self):
            """Test that negative price raises ValueError."""
            with pytest.raises(ValueError, match="Price cannot be negative"):
                calculate_discount(-10, 10)

        def test_discount_below_zero_raises_error(self):
            """Test that discount below 0 raises ValueError."""
            with pytest.raises(ValueError, match="Discount must be between"):
                calculate_discount(100, -1)

        def test_discount_above_hundred_raises_error(self):
            """Test that discount above 100 raises ValueError."""
            with pytest.raises(ValueError, match="Discount must be between"):
                calculate_discount(100, 101)

        # 类型测试（如果需要）
        def test_float_precision(self):
            """Test floating point precision."""
            result = calculate_discount(100, 33.33)
            assert result == pytest.approx(66.67, rel=1e-2)

测试 API 端点
=============

**待测试代码**:

.. code-block:: python

    # routes/users.py
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    router = APIRouter()

    class UserCreate(BaseModel):
        email: str
        name: str

    @router.post("/users", status_code=201)
    async def create_user(user: UserCreate):
        if not user.email or "@" not in user.email:
            raise HTTPException(status_code=400, detail="Invalid email")
        # 创建用户逻辑
        return {"id": 1, "email": user.email, "name": user.name}

**提示词**::

    请为这个 FastAPI 端点生成测试，使用 httpx 和 pytest

**生成的测试**:

.. code-block:: python

    import pytest
    from httpx import AsyncClient
    from main import app


    @pytest.fixture
    async def client():
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac


    class TestCreateUser:
        """Tests for POST /users endpoint."""

        @pytest.mark.asyncio
        async def test_create_user_success(self, client):
            """Test successful user creation."""
            response = await client.post(
                "/users",
                json={"email": "test@example.com", "name": "Test User"}
            )

            assert response.status_code == 201
            data = response.json()
            assert data["email"] == "test@example.com"
            assert data["name"] == "Test User"
            assert "id" in data

        @pytest.mark.asyncio
        async def test_create_user_invalid_email(self, client):
            """Test with invalid email format."""
            response = await client.post(
                "/users",
                json={"email": "invalid-email", "name": "Test"}
            )

            assert response.status_code == 400
            assert "Invalid email" in response.json()["detail"]

        @pytest.mark.asyncio
        async def test_create_user_empty_email(self, client):
            """Test with empty email."""
            response = await client.post(
                "/users",
                json={"email": "", "name": "Test"}
            )

            assert response.status_code == 400

        @pytest.mark.asyncio
        async def test_create_user_missing_fields(self, client):
            """Test with missing required fields."""
            response = await client.post("/users", json={})

            assert response.status_code == 422  # Validation error

提高测试覆盖率
==============

查找未覆盖代码
--------------

::

    @coverage_report.html

    根据覆盖率报告，以下代码行未被测试覆盖：
    - line 45-48: 错误处理分支
    - line 67: 边界条件

    请生成测试用例覆盖这些代码

针对性生成测试
--------------

::

    @mymodule.py

    这个模块的测试覆盖率是 75%，请分析未覆盖的代码，
    并生成测试用例将覆盖率提高到 90% 以上

AI 辅助调试
===========

调试是找出和修复 bug 的过程，AI 可以帮助：

- **分析错误信息**: 解释复杂的错误堆栈
- **定位问题**: 根据症状找到可能的原因
- **提供修复建议**: 给出修复方案
- **解释代码行为**: 帮助理解代码执行流程

分析错误信息
============

基本用法
--------

::

    我遇到了这个错误，请帮我分析：

    ```
    Traceback (most recent call last):
      File "main.py", line 42, in process_data
        result = data['key']['nested']
    KeyError: 'nested'
    ```

    相关代码：
    @main.py:40-50

示例：分析复杂错误
------------------

**错误信息**:

.. code-block:: text

    sqlalchemy.exc.IntegrityError: (psycopg2.errors.UniqueViolation)
    duplicate key value violates unique constraint "users_email_key"
    DETAIL:  Key (email)=(test@example.com) already exists.

    [SQL: INSERT INTO users (email, name, created_at)
    VALUES (%(email)s, %(name)s, %(created_at)s) RETURNING users.id]
    [parameters: {'email': 'test@example.com', 'name': 'Test', 'created_at': ...}]

**提示词**::

    请分析这个错误并给出解决方案

**AI 分析**:

.. code-block:: text

    错误分析：

    1. 错误类型：IntegrityError - 数据库完整性约束违反
    2. 具体原因：尝试插入的 email 'test@example.com' 已存在
    3. 约束名称：users_email_key（email 字段的唯一约束）

    可能的场景：
    - 用户重复注册
    - 数据迁移时有重复数据
    - 并发请求导致竞态条件

    解决方案：

    方案 1：在插入前检查是否存在
    ```python
    existing = db.query(User).filter_by(email=email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")
    ```

    方案 2：使用 upsert（更新或插入）
    ```python
    from sqlalchemy.dialects.postgresql import insert

    stmt = insert(User).values(email=email, name=name)
    stmt = stmt.on_conflict_do_update(
        index_elements=['email'],
        set_={'name': name, 'updated_at': datetime.now()}
    )
    ```

    方案 3：捕获异常并处理
    ```python
    try:
        db.add(user)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already exists")
    ```

    推荐：方案 1 + 方案 3 结合使用，提供更好的用户体验和错误处理。

调试复杂问题
============

问题描述模板
------------

::

    【问题描述】
    [描述问题现象]

    【期望行为】
    [描述正确的行为]

    【实际行为】
    [描述实际发生的情况]

    【复现步骤】
    1. [步骤 1]
    2. [步骤 2]
    3. [步骤 3]

    【相关代码】
    @file.py

    【错误日志】
    [粘贴错误日志]

    【已尝试的解决方案】
    - [方案 1]：[结果]
    - [方案 2]：[结果]

示例：调试异步问题
------------------

::

    【问题描述】
    异步任务偶尔会卡住，不返回结果

    【期望行为】
    所有任务在 5 秒内完成

    【实际行为】
    大约 10% 的请求会超时

    【相关代码】
    ```python
    async def fetch_all(urls):
        tasks = [fetch(url) for url in urls]
        return await asyncio.gather(*tasks)
    ```

    【已尝试】
    - 增加超时时间：问题依然存在
    - 减少并发数：问题减少但未消除

**AI 分析**:

.. code-block:: text

    可能的原因：

    1. 连接池耗尽
       - asyncio.gather 同时发起所有请求
       - 如果 urls 很多，可能超过连接池限制

    2. 某些 URL 响应慢
       - gather 等待所有任务完成
       - 一个慢任务会阻塞整体

    3. 资源竞争
       - 共享资源（如数据库连接）可能造成死锁

    解决方案：

    1. 使用信号量限制并发
    ```python
    async def fetch_all(urls, max_concurrent=10):
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_limit(url):
            async with semaphore:
                return await fetch(url)

        tasks = [fetch_with_limit(url) for url in urls]
        return await asyncio.gather(*tasks)
    ```

    2. 添加超时处理
    ```python
    async def fetch_all(urls, timeout=5):
        async def fetch_with_timeout(url):
            try:
                return await asyncio.wait_for(fetch(url), timeout=timeout)
            except asyncio.TimeoutError:
                return None  # 或者返回错误标记

        tasks = [fetch_with_timeout(url) for url in urls]
        return await asyncio.gather(*tasks)
    ```

    3. 使用 return_exceptions 避免单个失败影响全部
    ```python
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed: {result}")
    ```

调试策略
========

二分法调试
----------

::

    这段代码有 bug，但我不确定问题在哪里。
    请帮我设计一个调试策略，使用二分法定位问题。

    @problematic_code.py

日志分析
--------

::

    以下是错误发生时的日志，请帮我分析问题：

    ```
    2024-01-15 10:23:45 INFO Starting request processing
    2024-01-15 10:23:45 DEBUG Fetching user data for id=123
    2024-01-15 10:23:46 DEBUG User data fetched successfully
    2024-01-15 10:23:46 DEBUG Processing order items
    2024-01-15 10:23:47 ERROR Unexpected error in process_order
    2024-01-15 10:23:47 ERROR Traceback: ...
    ```

性能调试
--------

::

    这个函数运行很慢，请帮我分析可能的性能瓶颈：

    @slow_function.py

    运行时间：平均 5 秒
    期望时间：< 500ms

测试驱动调试
============

用测试来验证修复：

::

    我修复了这个 bug：
    [描述修复]

    请帮我写一个测试用例来验证：
    1. 修复是有效的
    2. 没有引入新的问题

.. code-block:: python

    def test_bug_fix_issue_123():
        """Regression test for issue #123.

        Bug: User creation failed when email contained '+' character.
        Fix: Updated email validation regex to allow '+'.
        """
        # Arrange
        email = "test+label@example.com"

        # Act
        user = create_user(email=email, name="Test")

        # Assert
        assert user is not None
        assert user.email == email

调试工具集成
============

使用 AI 解释调试工具输出：

pdb 调试
--------

::

    我在使用 pdb 调试，当前状态：

    ```
    > /app/main.py(42)process_data()
    -> result = calculate(data)
    (Pdb) p data
    {'items': [1, 2, 3], 'config': None}
    (Pdb) p self.cache
    {}
    ```

    问题是 result 返回 None，请帮我分析可能的原因

性能分析
--------

::

    以下是 cProfile 的输出，请帮我分析性能瓶颈：

    ```
             ncalls  tottime  percall  cumtime  percall filename:lineno(function)
               1000    2.345    0.002    5.678    0.006 utils.py:42(process)
              10000    1.234    0.000    1.234    0.000 {method 'encode' of 'str'}
            ```

    @utils.py:42

小结
====

本教程介绍了 AI 辅助测试与调试：

- **测试生成**: 单元测试、API 测试、覆盖率提升
- **错误分析**: 解读错误信息、定位问题
- **调试策略**: 二分法、日志分析、性能调试
- **工具集成**: pdb、profiler 等

关键要点：

1. AI 可以大幅提高测试效率
2. 提供完整的上下文有助于准确调试
3. 用测试验证修复是最佳实践

下一步
------

在下一个教程中，我们将总结氛围编程的最佳实践。

练习
====

1. 为你的项目生成测试，提高覆盖率到 80%
2. 使用 AI 分析一个你遇到过的复杂 bug
3. 建立你的调试问题描述模板
4. 尝试使用 AI 解释 profiler 输出

参考资源
========

- `pytest 官方文档 <https://docs.pytest.org/>`_
- `Python 调试技巧 <https://realpython.com/python-debugging-pdb/>`_
- `Testing Best Practices <https://martinfowler.com/articles/practical-test-pyramid.html>`_
