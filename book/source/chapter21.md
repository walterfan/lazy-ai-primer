(chapter21)=
# 第二十一章：多 Agent 系统与协作模式

```{mermaid}
mindmap
  root((多Agent系统))
    为什么多Agent
      上下文限制
      专业化需求
      并行处理
    架构模式
      Supervisor
      Hierarchical
      Peer-to-Peer
      Debate
      Assembly Line
    Agent间通信
      消息总线
      结构化消息
    任务分解
      分配策略
      依赖管理
    冲突解决
      裁判机制
      投票机制
    实战案例
      软件开发团队
      CrewAI实现
```

> "一个人可以走得很快，一群人可以走得很远。Agent 也是如此。"

## 21.1 为什么需要多 Agent

单个 Agent 的局限：
- **上下文窗口限制**：一个 Agent 无法同时处理所有信息
- **专业化需求**：不同任务需要不同的专业知识和工具
- **并行处理**：某些任务可以并行执行以提高效率
- **质量保证**：多个 Agent 交叉验证可以提高输出质量

## 21.2 多 Agent 架构模式

### 1. Supervisor 模式

```
                ┌──────────────┐
                │  Supervisor  │
                │   Agent      │
                └──────┬───────┘
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Worker A │ │ Worker B │ │ Worker C │
    │ (搜索)   │ │ (分析)   │ │ (写作)   │
    └──────────┘ └──────────┘ └──────────┘
```

Supervisor 负责：任务分配、进度监控、结果整合、冲突解决。

### 2. Hierarchical 模式

```
              ┌─────────────┐
              │  总监 Agent  │
              └──────┬──────┘
                     │
         ┌───────────┼───────────┐
         ↓                       ↓
   ┌───────────┐          ┌───────────┐
   │ 前端经理   │          │ 后端经理   │
   └─────┬─────┘          └─────┬─────┘
         │                      │
    ┌────┼────┐            ┌────┼────┐
    ↓         ↓            ↓         ↓
  ┌────┐  ┌────┐        ┌────┐  ┌────┐
  │UI  │  │测试│        │API │  │DB  │
  └────┘  └────┘        └────┘  └────┘
```

### 3. Peer-to-Peer 模式

```
    ┌──────┐ ←──→ ┌──────┐
    │Agent │      │Agent │
    │  A   │      │  B   │
    └──┬───┘      └───┬──┘
       │    ╲    ╱    │
       │     ╲  ╱     │
       │      ╲╱      │
       │      ╱╲      │
       │     ╱  ╲     │
    ┌──┴───┐      ┌───┴──┐
    │Agent │ ←──→ │Agent │
    │  C   │      │  D   │
    └──────┘      └──────┘
```

### 4. Debate 模式

```
    ┌──────────┐     ┌──────────┐
    │ Agent A  │ ←→  │ Agent B  │
    │ (正方)   │     │ (反方)   │
    └────┬─────┘     └─────┬────┘
         │                 │
         └────────┬────────┘
                  ↓
           ┌───────────┐
           │ Judge Agent│
           │ (裁判)     │
           └───────────┘
```

### 5. Assembly Line 模式

```
输入 → [Agent 1: 需求分析] → [Agent 2: 设计] → [Agent 3: 编码] → [Agent 4: 测试] → 输出
```

## 21.3 Agent 间通信

```python
from dataclasses import dataclass
from typing import Any
from enum import Enum

class MessageType(Enum):
    TASK = "task"           # 任务分配
    RESULT = "result"       # 任务结果
    QUESTION = "question"   # 提问
    FEEDBACK = "feedback"   # 反馈
    STATUS = "status"       # 状态更新

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    msg_type: MessageType
    content: Any
    metadata: dict = None

class MessageBus:
    """Agent 间的消息总线"""
    def __init__(self):
        self.subscribers: dict[str, list] = {}
        self.message_log: list[AgentMessage] = []
    
    def subscribe(self, agent_id: str, callback):
        self.subscribers.setdefault(agent_id, []).append(callback)
    
    async def send(self, message: AgentMessage):
        self.message_log.append(message)
        for callback in self.subscribers.get(message.receiver, []):
            await callback(message)
    
    async def broadcast(self, sender: str, msg_type: MessageType, content: Any):
        for agent_id in self.subscribers:
            if agent_id != sender:
                await self.send(AgentMessage(sender, agent_id, msg_type, content))
```

## 21.4 实战：构建软件开发团队 Agent 系统

```python
"""使用 CrewAI 构建软件开发团队"""
from crewai import Agent, Task, Crew, Process

# 定义 Agent 角色
product_manager = Agent(
    role="产品经理",
    goal="将用户需求转化为清晰的产品规格",
    backstory="你是一位经验丰富的产品经理，擅长理解用户需求并编写清晰的PRD。",
    verbose=True,
    allow_delegation=False
)

architect = Agent(
    role="软件架构师",
    goal="设计可扩展、可维护的系统架构",
    backstory="你是一位资深架构师，精通微服务、云原生和分布式系统设计。",
    verbose=True,
    allow_delegation=False
)

developer = Agent(
    role="高级开发工程师",
    goal="编写高质量、可测试的代码",
    backstory="你是一位全栈开发工程师，精通 Python、TypeScript，注重代码质量。",
    verbose=True,
    allow_delegation=False
)

tester = Agent(
    role="QA 工程师",
    goal="确保软件质量，发现潜在问题",
    backstory="你是一位严谨的测试工程师，擅长编写测试用例和发现边界情况。",
    verbose=True,
    allow_delegation=False
)

reviewer = Agent(
    role="代码审查员",
    goal="审查代码质量、安全性和最佳实践",
    backstory="你是一位代码审查专家，关注安全、性能和可维护性。",
    verbose=True,
    allow_delegation=False
)

# 定义任务流水线
def build_software(requirement: str):
    # 任务 1：需求分析
    task_requirements = Task(
        description=f"分析以下需求并编写产品规格文档：\n{requirement}",
        expected_output="详细的产品规格文档，包含用户故事和验收标准",
        agent=product_manager
    )
    
    # 任务 2：架构设计
    task_architecture = Task(
        description="基于产品规格设计系统架构",
        expected_output="架构设计文档，包含服务划分、技术选型、API 设计",
        agent=architect,
        context=[task_requirements]
    )
    
    # 任务 3：编码实现
    task_coding = Task(
        description="根据架构设计实现核心功能代码",
        expected_output="完整的、可运行的代码实现",
        agent=developer,
        context=[task_requirements, task_architecture]
    )
    
    # 任务 4：测试
    task_testing = Task(
        description="为实现的代码编写全面的测试用例",
        expected_output="测试代码和测试报告",
        agent=tester,
        context=[task_coding]
    )
    
    # 任务 5：代码审查
    task_review = Task(
        description="审查代码质量、安全性和最佳实践合规性",
        expected_output="代码审查报告，包含改进建议",
        agent=reviewer,
        context=[task_coding, task_testing]
    )
    
    # 组建团队
    crew = Crew(
        agents=[product_manager, architect, developer, tester, reviewer],
        tasks=[task_requirements, task_architecture, task_coding, task_testing, task_review],
        process=Process.sequential,  # 顺序执行
        verbose=True
    )
    
    result = crew.kickoff()
    return result

# 运行
result = build_software(
    "构建一个 RESTful API 服务，支持用户注册、登录和个人资料管理"
)
print(result)
```

## 21.5 冲突解决机制

当多个 Agent 意见不一致时：

```python
async def resolve_conflict(opinions: list[dict], judge_agent) -> str:
    """多 Agent 冲突解决"""
    conflict_prompt = f"""
    以下是不同 Agent 对同一问题的不同意见：
    
    {chr(10).join(f"- {o['agent']}: {o['opinion']}" for o in opinions)}
    
    请作为裁判：
    1. 分析每个意见的优缺点
    2. 综合考虑后给出最终决策
    3. 解释决策理由
    """
    return await judge_agent.generate(conflict_prompt)
```

## 21.6 本章小结

多 Agent 系统是 AI 应用的重要发展方向。通过合理的架构设计和协作模式，多个专业化的 Agent 可以协同完成复杂任务。

核心要点：
1. **选择合适的架构模式**：根据任务特点选择 Supervisor、P2P 或 Assembly Line
2. **明确角色和职责**：每个 Agent 有清晰的专业领域
3. **设计通信协议**：结构化的消息格式和路由机制
4. **冲突解决**：当 Agent 意见不一致时有明确的解决机制

```{admonition} 思考题
:class: hint
1. 多 Agent 系统和微服务架构有什么相似之处？
2. 如何评估多 Agent 系统是否比单 Agent 更有效？
3. 多 Agent 系统的调试和监控有什么特殊挑战？
```
