(chapter05)=
# 第五章：软件质量与工程实践的变迁

```{mermaid}
mindmap
  root((软件质量与工程实践))
    代码质量
      静态分析
      代码审查
      Ruff/SonarQube
    测试金字塔
      单元测试
      集成测试
      E2E测试
    TDD/BDD/ATDD
      红绿重构
      行为驱动
      验收驱动
    技术债务
      分类管理
      质量门禁
    代码重构
      提取方法
      单一职责
    文档即代码
      Sphinx/MkDocs
      自动生成
    AI辅助质量
      AI测试生成
      AI代码审查
```

> "质量不是一种行为，而是一种习惯。" — 亚里士多德

## 5.1 代码质量：从人工审查到自动化分析

### 代码审查的演进

```
1970s: 代码走查（Code Walkthrough）— 打印代码，围坐讨论
1990s: 正式代码审查（Fagan Inspection）— 严格流程，角色分工
2000s: 轻量级代码审查 — Pull Request 模式
2010s: 自动化静态分析 — SonarQube、ESLint
2020s: AI 代码审查 — CodeRabbit、PR-Agent
```

### 静态分析工具

```python
# ruff — 极速 Python linter（Rust 编写）
# pyproject.toml 配置
"""
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "S", "B", "A", "C4"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101"]  # 允许测试中使用 assert
"""

# 运行检查
# ruff check .
# ruff format .
```

## 5.2 测试金字塔

```
          ╱╲
         ╱  ╲         E2E 测试（少量）
        ╱ E2E╲        - Playwright / Selenium
       ╱──────╲       - 慢、脆弱、但最接近用户
      ╱        ╲
     ╱ 集成测试  ╲     集成测试（适量）
    ╱────────────╲    - 测试服务间交互
   ╱              ╲   - 需要外部依赖
  ╱   单元测试     ╲   单元测试（大量）
 ╱──────────────────╲  - 快速、稳定、独立
╱                    ╲ - pytest / Jest / JUnit
```

### 单元测试示例

```python
# 被测代码
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, name: str, price: float, quantity: int = 1):
        if price < 0:
            raise ValueError("Price cannot be negative")
        if quantity < 1:
            raise ValueError("Quantity must be at least 1")
        self.items.append({"name": name, "price": price, "quantity": quantity})
    
    @property
    def total(self) -> float:
        return sum(item["price"] * item["quantity"] for item in self.items)
    
    def apply_discount(self, percentage: float) -> float:
        if not 0 <= percentage <= 100:
            raise ValueError("Discount must be between 0 and 100")
        return self.total * (1 - percentage / 100)

# 测试代码
import pytest

class TestShoppingCart:
    def setup_method(self):
        self.cart = ShoppingCart()
    
    def test_empty_cart_total_is_zero(self):
        assert self.cart.total == 0
    
    def test_add_single_item(self):
        self.cart.add_item("Python Book", 49.99)
        assert self.cart.total == 49.99
    
    def test_add_multiple_items(self):
        self.cart.add_item("Python Book", 49.99, quantity=2)
        self.cart.add_item("Coffee", 4.99)
        assert self.cart.total == pytest.approx(104.97)
    
    def test_negative_price_raises_error(self):
        with pytest.raises(ValueError, match="Price cannot be negative"):
            self.cart.add_item("Bad Item", -10)
    
    def test_apply_discount(self):
        self.cart.add_item("Item", 100)
        assert self.cart.apply_discount(20) == 80.0
    
    @pytest.mark.parametrize("discount", [-1, 101, 150])
    def test_invalid_discount_raises_error(self, discount):
        self.cart.add_item("Item", 100)
        with pytest.raises(ValueError):
            self.cart.apply_discount(discount)
```

## 5.3 TDD、BDD、ATDD

### TDD（测试驱动开发）

```
Red → Green → Refactor 循环：

1. Red:    写一个失败的测试
2. Green:  写最少的代码让测试通过
3. Refactor: 重构代码，保持测试通过
```

```python
# TDD 示例：实现一个密码验证器

# Step 1: Red — 写失败的测试
def test_password_must_be_at_least_8_chars():
    assert validate_password("short") == False

# Step 2: Green — 最简实现
def validate_password(password: str) -> bool:
    return len(password) >= 8

# Step 3: 继续添加测试
def test_password_must_contain_uppercase():
    assert validate_password("alllowercase123") == False

# Step 4: 扩展实现
def validate_password(password: str) -> bool:
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    return True
```

### BDD（行为驱动开发）

```gherkin
# features/login.feature
Feature: User Login
  As a registered user
  I want to log in to my account
  So that I can access my dashboard

  Scenario: Successful login
    Given I am on the login page
    When I enter valid credentials
    And I click the login button
    Then I should be redirected to the dashboard
    And I should see a welcome message

  Scenario: Failed login with wrong password
    Given I am on the login page
    When I enter an invalid password
    And I click the login button
    Then I should see an error message "Invalid credentials"
    And I should remain on the login page
```

## 5.4 技术债务管理

### 技术债务的分类

```
                    有意的                    无意的
              ┌─────────────────┬─────────────────┐
   鲁莽的     │ "我们没时间做    │ "什么是分层      │
              │  设计，先上线"   │  架构？"         │
              ├─────────────────┼─────────────────┤
   谨慎的     │ "我们知道后果，  │ "现在我们知道    │
              │  但现在必须发布" │  应该怎么做了"   │
              └─────────────────┴─────────────────┘
```

### 管理策略

```python
# 在代码中标记技术债务
# TODO: 这里需要重构，当前实现不支持并发
# FIXME: 临时方案，需要在 Q2 替换为消息队列
# HACK: 绕过第三方库的 bug，等升级后移除
# DEBT: 缺少输入验证，安全风险

# 使用工具追踪
# SonarQube 可以自动检测和量化技术债务
# 配置质量门禁（Quality Gate）
# - 新代码覆盖率 > 80%
# - 新代码无严重问题
# - 技术债务比率 < 5%
```

## 5.5 代码重构的艺术

### 常见重构手法

```python
# 重构前：过长的函数
def process_order(order_data):
    # 验证（20行）
    if not order_data.get('user_id'):
        raise ValueError("Missing user_id")
    if not order_data.get('items'):
        raise ValueError("Missing items")
    for item in order_data['items']:
        if item['quantity'] < 1:
            raise ValueError("Invalid quantity")
    
    # 计算价格（15行）
    subtotal = sum(i['price'] * i['quantity'] for i in order_data['items'])
    tax = subtotal * 0.1
    total = subtotal + tax
    
    # 保存到数据库（10行）
    order = Order(user_id=order_data['user_id'], total=total)
    db.session.add(order)
    db.session.commit()
    
    # 发送通知（10行）
    send_email(order_data['user_id'], f"Order {order.id} confirmed")
    return order

# 重构后：提取方法，单一职责
def process_order(order_data):
    validate_order(order_data)
    total = calculate_total(order_data['items'])
    order = save_order(order_data['user_id'], total)
    notify_user(order)
    return order

def validate_order(order_data):
    if not order_data.get('user_id'):
        raise ValueError("Missing user_id")
    if not order_data.get('items'):
        raise ValueError("Missing items")
    for item in order_data['items']:
        if item['quantity'] < 1:
            raise ValueError("Invalid quantity")

def calculate_total(items) -> float:
    subtotal = sum(i['price'] * i['quantity'] for i in items)
    tax = subtotal * 0.1
    return subtotal + tax

def save_order(user_id, total):
    order = Order(user_id=user_id, total=total)
    db.session.add(order)
    db.session.commit()
    return order

def notify_user(order):
    send_email(order.user_id, f"Order {order.id} confirmed")
```

## 5.6 文档即代码（Docs as Code）

```
# 文档与代码同仓库、同流程
project/
├── src/              # 源代码
├── tests/            # 测试
├── docs/             # 文档（Markdown/RST）
│   ├── conf.py       # Sphinx 配置
│   ├── index.md      # 首页
│   ├── api/          # API 文档（自动生成）
│   └── guides/       # 使用指南
├── .github/
│   └── workflows/
│       └── docs.yml  # 文档 CI/CD
└── README.md
```

## 5.7 从人工到 AI 辅助质量保证

| 传统方式 | AI 辅助方式 |
|----------|------------|
| 人工编写测试用例 | AI 自动生成测试用例 |
| 人工代码审查 | AI 预审 + 人工复核 |
| 手动编写文档 | AI 生成文档草稿 |
| 规则式静态分析 | AI 语义级代码分析 |
| 手动回归测试 | AI 智能选择回归范围 |

```python
# AI 辅助生成测试的 Prompt 示例
"""
请为以下 Python 函数生成全面的 pytest 测试用例，包括：
1. 正常路径测试
2. 边界条件测试
3. 异常处理测试
4. 参数化测试

函数代码：
{paste your function here}
"""
```

## 5.8 本章小结

软件质量不是测试出来的，而是设计和构建出来的。从 TDD 到 AI 辅助测试，从人工代码审查到 AI 代码审查，工具在变，但核心理念不变：**尽早发现问题，持续改进质量**。

在 AI 时代，质量保证的重点正在从"人写的代码有没有 Bug"转向"AI 生成的代码是否安全、正确、可维护"。这是一个全新的挑战，我们将在第九章深入探讨。

```{admonition} 思考题
:class: hint
1. 你的项目测试覆盖率是多少？你觉得够吗？
2. TDD 在 AI 辅助编程时代还有意义吗？
3. 如何建立对 AI 生成代码的质量信心？
```
