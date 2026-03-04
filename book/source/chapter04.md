(chapter04)=
# 第四章：软件架构的演进 — 从单体到智能体

```{mermaid}
mindmap
  root((软件架构演进))
    单体架构
      简单直接
      单体优先策略
    SOA
      企业服务总线
      服务解耦
    微服务
      独立部署
      去中心化
    Serverless
      按需计算
      零运维
    事件驱动架构
      松耦合
      Kafka
    领域驱动设计
      限界上下文
      聚合根
    API First
      OpenAPI规范
    Agent架构
      LLM驱动路由
      智能协调
```

> "架构是关于重要的东西的决策，不管那是什么。" — Ralph Johnson

## 4.1 架构演进的全景图

```
1990s        2000s        2010s        2015s        2020s        2025s
单体架构  →  SOA  →  微服务  →  Serverless  →  事件驱动  →  Agent 架构
  │           │         │           │              │            │
 简单        解耦      独立部署    按需计费      实时响应     自主决策
 紧耦合      ESB       API网关     冷启动       消息队列     LLM驱动
```

## 4.2 单体架构：起点与回归

单体架构将所有功能打包在一个部署单元中：

```python
# 典型的单体 Django 应用
# myapp/
# ├── users/          # 用户模块
# ├── orders/         # 订单模块
# ├── payments/       # 支付模块
# ├── notifications/  # 通知模块
# └── settings.py     # 共享配置

# 所有模块共享同一个数据库
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'myapp_db',
    }
}
```

**单体的优势**（常被忽视）：
- ✅ 开发简单，一个代码库
- ✅ 部署简单，一个部署包
- ✅ 调试简单，单进程调试
- ✅ 事务简单，数据库事务即可
- ✅ 适合小团队和早期项目

**"单体优先"策略**：Martin Fowler 建议新项目从单体开始，等到确实需要时再拆分微服务。

## 4.3 SOA 与微服务

### SOA（面向服务架构）

SOA 通过企业服务总线（ESB）连接各个服务：

```
┌──────┐  ┌──────┐  ┌──────┐
│服务A  │  │服务B  │  │服务C  │
└──┬───┘  └──┬───┘  └──┬───┘
   │         │         │
┌──┴─────────┴─────────┴──┐
│     企业服务总线（ESB）    │
└─────────────────────────┘
```

SOA 的问题：ESB 成为单点瓶颈和复杂性的集中地。

### 微服务的核心原则

1. **单一职责**：每个服务只做一件事
2. **独立部署**：服务可以独立发布
3. **去中心化**：没有中央 ESB
4. **容错设计**：服务间通过断路器等模式处理失败

```python
# 微服务间通信示例（使用 FastAPI + httpx）
from fastapi import FastAPI
import httpx

app = FastAPI()

# 订单服务调用用户服务
@app.post("/orders")
async def create_order(user_id: str, items: list):
    # 调用用户服务验证用户
    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            f"http://user-service:8001/users/{user_id}"
        )
        if user_resp.status_code != 200:
            raise HTTPException(status_code=404, detail="User not found")
    
    # 调用支付服务
    async with httpx.AsyncClient() as client:
        payment_resp = await client.post(
            "http://payment-service:8002/payments",
            json={"user_id": user_id, "amount": calculate_total(items)}
        )
    
    return {"order_id": "...", "status": "created"}
```

## 4.4 Serverless：按需计算

```python
# AWS Lambda 函数示例
import json
import boto3

def handler(event, context):
    """处理 API Gateway 请求"""
    body = json.loads(event['body'])
    
    # 业务逻辑
    result = process_data(body)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

Serverless 的权衡：
- ✅ 零运维，按调用计费
- ✅ 自动扩缩容
- ⚠️ 冷启动延迟
- ⚠️ 执行时间限制
- ⚠️ 供应商锁定

## 4.5 事件驱动架构（EDA）

```python
# 事件驱动示例（使用 Kafka）
from confluent_kafka import Producer, Consumer

# 生产者：发布订单创建事件
producer = Producer({'bootstrap.servers': 'kafka:9092'})

def publish_order_event(order):
    event = {
        "type": "OrderCreated",
        "data": {
            "order_id": order.id,
            "user_id": order.user_id,
            "total": order.total,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    producer.produce('orders', json.dumps(event).encode())
    producer.flush()

# 消费者：库存服务监听订单事件
consumer = Consumer({
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'inventory-service',
    'auto.offset.reset': 'earliest'
})
consumer.subscribe(['orders'])

while True:
    msg = consumer.poll(1.0)
    if msg and not msg.error():
        event = json.loads(msg.value())
        if event['type'] == 'OrderCreated':
            reserve_inventory(event['data'])
```

EDA 的优势：
- **松耦合**：生产者不需要知道消费者
- **可扩展**：新增消费者不影响现有系统
- **弹性**：消费者可以按自己的速度处理
- **审计**：事件流天然形成审计日志

## 4.6 领域驱动设计（DDD）

### 核心概念

```
┌─────────────────────────────────────┐
│           限界上下文                  │
│      (Bounded Context)              │
│                                     │
│  ┌─────────┐    ┌─────────────┐    │
│  │ 实体     │    │ 值对象       │    │
│  │(Entity)  │    │(Value Object)│    │
│  └─────────┘    └─────────────┘    │
│                                     │
│  ┌─────────┐    ┌─────────────┐    │
│  │ 聚合根   │    │ 领域事件     │    │
│  │(Aggregate│    │(Domain Event)│    │
│  │  Root)   │    │             │    │
│  └─────────┘    └─────────────┘    │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 仓储 (Repository)           │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

```python
# DDD 示例：订单聚合根
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

@dataclass(frozen=True)
class Money:
    """值对象"""
    amount: float
    currency: str = "USD"

@dataclass
class OrderItem:
    """实体"""
    product_id: str
    quantity: int
    unit_price: Money

@dataclass
class Order:
    """聚合根"""
    id: str
    customer_id: str
    items: List[OrderItem] = field(default_factory=list)
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.utcnow)
    _events: List[dict] = field(default_factory=list, repr=False)

    @property
    def total(self) -> Money:
        total = sum(item.unit_price.amount * item.quantity for item in self.items)
        return Money(amount=total)

    def place(self):
        """下单 — 业务规则在聚合根中"""
        if not self.items:
            raise ValueError("Cannot place an empty order")
        if self.status != "draft":
            raise ValueError(f"Cannot place order in {self.status} status")
        self.status = "placed"
        self._events.append({
            "type": "OrderPlaced",
            "order_id": self.id,
            "total": self.total.amount
        })
```

## 4.7 API First 设计

```yaml
# OpenAPI 规范示例
openapi: 3.0.3
info:
  title: Order Service API
  version: 1.0.0
paths:
  /orders:
    post:
      summary: Create a new order
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateOrderRequest'
      responses:
        '201':
          description: Order created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Order'
```

## 4.8 架构决策记录（ADR）

```markdown
# ADR-001: 使用事件驱动架构处理订单流程

## 状态：已接受

## 背景
订单处理涉及多个服务（库存、支付、通知），需要一种松耦合的通信方式。

## 决策
采用事件驱动架构，使用 Kafka 作为消息中间件。

## 后果
- ✅ 服务间松耦合
- ✅ 易于添加新的消费者
- ⚠️ 需要处理最终一致性
- ⚠️ 增加了调试复杂度
```

## 4.9 从微服务到 AI Agent 架构

```
微服务架构：                    Agent 架构：
┌──────┐ ┌──────┐             ┌──────────┐ ┌──────────┐
│用户   │ │订单   │             │ Planning │ │ Coding   │
│服务   │ │服务   │      →      │ Agent    │ │ Agent    │
└──┬───┘ └──┬───┘             └────┬─────┘ └────┬─────┘
   │        │                      │             │
   API 网关                    Orchestrator Agent
                              (LLM 驱动的路由与协调)
```

关键区别：
- 微服务：**确定性**路由，API 网关按规则转发
- Agent 架构：**智能**路由，LLM 根据意图决定调用哪些服务
- 微服务：服务间通过 API 契约通信
- Agent 架构：Agent 间通过自然语言 + 结构化数据通信

## 4.10 本章小结

软件架构的演进反映了我们对复杂性管理的不断探索。从单体到微服务，我们学会了分而治之；从同步到事件驱动，我们学会了松耦合；从 API 网关到 AI Agent，我们正在学习让系统自主决策。

架构没有银弹，选择取决于团队规模、业务复杂度、技术能力等多种因素。在 AI 时代，架构师的角色不是消失，而是从"设计系统结构"升级为"设计人机协作的系统结构"。

```{admonition} 思考题
:class: hint
1. 你当前的项目适合什么架构？为什么？
2. 事件驱动架构和 AI Agent 架构有什么相似之处？
3. 如果让 AI 来做架构决策，你觉得可行吗？
```
