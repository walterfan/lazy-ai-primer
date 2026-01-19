####################################
Tutorial 2: 智能体（Agent）
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是智能体？
==============

**智能体（Agent）** 是能够感知环境并在环境中采取行动的系统。

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                        环境 (Environment)                    │
   │                                                              │
   │    ┌─────────────┐                    ┌─────────────┐       │
   │    │   感知器     │◄────── 感知 ──────│             │       │
   │    │  (Sensors)  │                    │    环境     │       │
   │    └──────┬──────┘                    │   状态      │       │
   │           │                           │             │       │
   │           ▼                           │             │       │
   │    ┌─────────────┐                    │             │       │
   │    │   智能体    │                    │             │       │
   │    │  (Agent)    │                    │             │       │
   │    └──────┬──────┘                    │             │       │
   │           │                           │             │       │
   │           ▼                           │             │       │
   │    ┌─────────────┐                    │             │       │
   │    │   执行器     │─────── 行动 ──────►│             │       │
   │    │ (Actuators) │                    └─────────────┘       │
   │    └─────────────┘                                          │
   │                                                              │
   └─────────────────────────────────────────────────────────────┘

用生活中的例子来理解：

- **自动驾驶汽车**: 感知（摄像头、雷达）→ 决策（AI）→ 行动（转向、加速）
- **智能客服**: 感知（用户输入）→ 决策（NLP）→ 行动（回复）
- **扫地机器人**: 感知（传感器）→ 决策（路径规划）→ 行动（移动、清扫）

智能体的组成
============

.. code-block:: python

   class Agent:
       """智能体的基本结构"""
       
       def __init__(self):
           self.state = None  # 内部状态
       
       def perceive(self, environment):
           """感知环境"""
           return environment.get_percept()
       
       def think(self, percept):
           """决策/思考"""
           # 根据感知决定行动
           action = self.decide(percept)
           return action
       
       def act(self, action, environment):
           """执行行动"""
           environment.execute(action)
       
       def run(self, environment):
           """智能体主循环"""
           while True:
               percept = self.perceive(environment)
               action = self.think(percept)
               self.act(action, environment)

理性智能体
==========

**理性智能体** 是指在给定感知序列的情况下，能够采取使预期效用最大化的行动的智能体。

关键概念：

- **性能度量**: 评估智能体行为好坏的标准
- **先验知识**: 智能体事先知道的关于环境的信息
- **感知序列**: 智能体到目前为止接收到的所有感知
- **可执行动作**: 智能体可以采取的行动集合

.. code-block:: python

   class RationalAgent(Agent):
       """理性智能体"""
       
       def __init__(self, performance_measure):
           super().__init__()
           self.performance_measure = performance_measure
           self.percept_history = []
       
       def think(self, percept):
           # 记录感知历史
           self.percept_history.append(percept)
           
           # 选择能最大化预期效用的行动
           best_action = None
           best_utility = float('-inf')
           
           for action in self.get_possible_actions():
               expected_utility = self.estimate_utility(action, percept)
               if expected_utility > best_utility:
                   best_utility = expected_utility
                   best_action = action
           
           return best_action

智能体的类型
============

1. 简单反射型智能体
-------------------

**特点**: 只根据当前感知做决策，不考虑历史

.. code-block:: python

   class SimpleReflexAgent(Agent):
       """简单反射型智能体"""
       
       def __init__(self, rules):
           super().__init__()
           self.rules = rules  # 条件-动作规则
       
       def think(self, percept):
           """直接根据规则做出反应"""
           for condition, action in self.rules:
               if condition(percept):
                   return action
           return None

   # 示例：简单的恒温器
   thermostat_rules = [
       (lambda p: p['temperature'] < 20, 'turn_on_heater'),
       (lambda p: p['temperature'] > 25, 'turn_off_heater'),
       (lambda p: True, 'do_nothing')
   ]

   thermostat = SimpleReflexAgent(thermostat_rules)
   action = thermostat.think({'temperature': 18})
   print(f"行动: {action}")  # turn_on_heater

**优点**: 简单、快速

**缺点**: 无法处理需要记忆的情况

2. 基于模型的反射型智能体
-------------------------

**特点**: 维护环境的内部模型，考虑历史状态

.. code-block:: python

   class ModelBasedAgent(Agent):
       """基于模型的反射型智能体"""
       
       def __init__(self, transition_model, sensor_model, rules):
           super().__init__()
           self.transition_model = transition_model  # 状态转移模型
           self.sensor_model = sensor_model          # 感知模型
           self.rules = rules
           self.state = None
           self.last_action = None
       
       def think(self, percept):
           # 更新内部状态
           self.state = self.update_state(
               self.state,
               self.last_action,
               percept
           )
           
           # 根据状态选择行动
           for condition, action in self.rules:
               if condition(self.state):
                   self.last_action = action
                   return action
           
           return None
       
       def update_state(self, state, action, percept):
           """根据转移模型和感知更新状态"""
           if state is None:
               return self.sensor_model(percept)
           
           predicted_state = self.transition_model(state, action)
           return self.sensor_model(percept, predicted_state)

3. 基于目标的智能体
-------------------

**特点**: 有明确的目标，选择能达成目标的行动

.. code-block:: python

   class GoalBasedAgent(Agent):
       """基于目标的智能体"""
       
       def __init__(self, goal):
           super().__init__()
           self.goal = goal
           self.state = None
       
       def think(self, percept):
           self.state = self.update_state(percept)
           
           # 搜索达成目标的行动序列
           plan = self.search_for_goal(self.state, self.goal)
           
           if plan:
               return plan[0]  # 返回第一个行动
           return None
       
       def search_for_goal(self, state, goal):
           """搜索达成目标的路径"""
           # 使用搜索算法（下一章详细介绍）
           pass

   # 示例：迷宫寻路智能体
   class MazeAgent(GoalBasedAgent):
       def __init__(self, goal_position):
           super().__init__(goal_position)
       
       def search_for_goal(self, current_pos, goal_pos):
           # 简化版：贪心选择最近的方向
           dx = goal_pos[0] - current_pos[0]
           dy = goal_pos[1] - current_pos[1]
           
           actions = []
           if dx > 0: actions.append('right')
           if dx < 0: actions.append('left')
           if dy > 0: actions.append('down')
           if dy < 0: actions.append('up')
           
           return actions if actions else ['stay']

4. 基于效用的智能体
-------------------

**特点**: 不仅考虑是否达成目标，还考虑"有多好"

.. code-block:: python

   class UtilityBasedAgent(Agent):
       """基于效用的智能体"""
       
       def __init__(self, utility_function):
           super().__init__()
           self.utility = utility_function
           self.state = None
       
       def think(self, percept):
           self.state = self.update_state(percept)
           
           # 评估每个行动的预期效用
           best_action = None
           best_expected_utility = float('-inf')
           
           for action in self.get_possible_actions():
               # 计算行动的预期效用
               expected_utility = self.expected_utility(action)
               
               if expected_utility > best_expected_utility:
                   best_expected_utility = expected_utility
                   best_action = action
           
           return best_action
       
       def expected_utility(self, action):
           """计算行动的预期效用"""
           total = 0
           for outcome, probability in self.get_outcomes(action):
               total += probability * self.utility(outcome)
           return total

5. 学习型智能体
---------------

**特点**: 能够从经验中学习，不断改进

.. code-block:: python

   class LearningAgent(Agent):
       """学习型智能体"""
       
       def __init__(self):
           super().__init__()
           self.performance_element = None  # 决策组件
           self.learning_element = None     # 学习组件
           self.critic = None               # 评价组件
           self.problem_generator = None    # 探索组件
       
       def think(self, percept):
           # 1. 根据当前知识选择行动
           action = self.performance_element.select_action(percept)
           
           # 2. 评价行动效果
           feedback = self.critic.evaluate(percept, action)
           
           # 3. 学习改进
           self.learning_element.learn(feedback)
           
           # 4. 探索新可能
           if self.should_explore():
               action = self.problem_generator.suggest_exploration()
           
           return action

实战：用 PyTorch 实现学习型智能体
=================================

让我们实现一个简单的学习型智能体，学习在网格世界中寻找宝藏：

.. code-block:: python

   import torch
   import torch.nn as nn
   import random

   class GridWorld:
       """简单的网格世界环境"""
       
       def __init__(self, size=5):
           self.size = size
           self.agent_pos = [0, 0]
           self.goal_pos = [size-1, size-1]
       
       def reset(self):
           self.agent_pos = [0, 0]
           return self.get_state()
       
       def get_state(self):
           """返回状态向量"""
           return torch.tensor([
               self.agent_pos[0] / self.size,
               self.agent_pos[1] / self.size,
               self.goal_pos[0] / self.size,
               self.goal_pos[1] / self.size
           ], dtype=torch.float32)
       
       def step(self, action):
           """执行行动，返回新状态和奖励"""
           # 行动: 0=上, 1=下, 2=左, 3=右
           moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
           dx, dy = moves[action]
           
           # 更新位置（边界检查）
           new_x = max(0, min(self.size-1, self.agent_pos[0] + dx))
           new_y = max(0, min(self.size-1, self.agent_pos[1] + dy))
           self.agent_pos = [new_x, new_y]
           
           # 计算奖励
           if self.agent_pos == self.goal_pos:
               reward = 10.0  # 到达目标
               done = True
           else:
               reward = -0.1  # 每步小惩罚，鼓励快速到达
               done = False
           
           return self.get_state(), reward, done

   class SimpleQNetwork(nn.Module):
       """简单的 Q 网络"""
       
       def __init__(self, state_size=4, action_size=4):
           super().__init__()
           self.fc1 = nn.Linear(state_size, 32)
           self.fc2 = nn.Linear(32, 32)
           self.fc3 = nn.Linear(32, action_size)
           self.relu = nn.ReLU()
       
       def forward(self, x):
           x = self.relu(self.fc1(x))
           x = self.relu(self.fc2(x))
           return self.fc3(x)

   class QLearningAgent:
       """Q-Learning 智能体"""
       
       def __init__(self, state_size=4, action_size=4):
           self.action_size = action_size
           self.q_network = SimpleQNetwork(state_size, action_size)
           self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.01)
           self.epsilon = 1.0  # 探索率
           self.epsilon_decay = 0.995
           self.epsilon_min = 0.01
           self.gamma = 0.99  # 折扣因子
       
       def select_action(self, state):
           """选择行动（ε-贪婪策略）"""
           if random.random() < self.epsilon:
               return random.randint(0, self.action_size - 1)
           
           with torch.no_grad():
               q_values = self.q_network(state)
               return q_values.argmax().item()
       
       def learn(self, state, action, reward, next_state, done):
           """学习更新"""
           # 计算目标 Q 值
           with torch.no_grad():
               if done:
                   target = reward
               else:
                   next_q = self.q_network(next_state).max()
                   target = reward + self.gamma * next_q
           
           # 计算当前 Q 值
           current_q = self.q_network(state)[action]
           
           # 计算损失并更新
           loss = nn.MSELoss()(current_q, torch.tensor(target))
           
           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()
           
           # 衰减探索率
           self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
           
           return loss.item()

   # 训练智能体
   def train_agent(episodes=500):
       env = GridWorld(size=5)
       agent = QLearningAgent()
       
       rewards_history = []
       
       for episode in range(episodes):
           state = env.reset()
           total_reward = 0
           
           for step in range(100):  # 最多100步
               action = agent.select_action(state)
               next_state, reward, done = env.step(action)
               
               agent.learn(state, action, reward, next_state, done)
               
               state = next_state
               total_reward += reward
               
               if done:
                   break
           
           rewards_history.append(total_reward)
           
           if (episode + 1) % 100 == 0:
               avg_reward = sum(rewards_history[-100:]) / 100
               print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
       
       return agent, rewards_history

   # 运行训练
   if __name__ == "__main__":
       print("开始训练智能体...")
       agent, rewards = train_agent(500)
       
       print("\n测试训练好的智能体:")
       env = GridWorld(size=5)
       state = env.reset()
       
       print(f"起点: {env.agent_pos}, 目标: {env.goal_pos}")
       
       for step in range(20):
           action = agent.select_action(state)
           action_names = ['上', '下', '左', '右']
           state, reward, done = env.step(action)
           print(f"步骤 {step+1}: {action_names[action]} -> 位置 {env.agent_pos}")
           
           if done:
               print("到达目标！")
               break

环境的分类
==========

.. csv-table::
   :header: "维度", "类型", "说明", "例子"
   :widths: 15, 20, 35, 30

   "可观察性", "完全可观察", "智能体能感知完整环境状态", "国际象棋"
   "", "部分可观察", "只能感知部分状态", "扑克牌"
   "确定性", "确定性", "行动结果完全确定", "国际象棋"
   "", "随机性", "行动结果有不确定性", "掷骰子游戏"
   "连续性", "离散", "有限的状态和行动", "棋类游戏"
   "", "连续", "无限的状态或行动", "自动驾驶"
   "智能体数", "单智能体", "只有一个智能体", "数独"
   "", "多智能体", "多个智能体交互", "围棋"

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "智能体", "能感知环境并采取行动的系统"
   "感知", "智能体获取环境信息的过程"
   "行动", "智能体对环境产生影响的过程"
   "理性", "选择能最大化预期效用的行动"
   "效用", "衡量状态或结果好坏的数值"
   "学习", "从经验中改进行为的能力"

下一步
======

在下一个教程中，我们将学习搜索算法，这是智能体解决问题的核心方法。

:doc:`tutorial_03_search_algorithms`
