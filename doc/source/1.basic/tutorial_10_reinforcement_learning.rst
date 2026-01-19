####################################
Tutorial 10: 强化学习基础
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是强化学习？
================

**强化学习（RL）** 是让智能体通过与环境交互，学习最优行为策略的方法。

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    强化学习循环                              │
   │                                                              │
   │         状态 s_t                    奖励 r_t                 │
   │            │                           ▲                     │
   │            ▼                           │                     │
   │     ┌─────────────┐             ┌─────────────┐             │
   │     │   智能体    │   动作 a_t   │    环境     │             │
   │     │   (Agent)   │────────────►│ (Environment)│             │
   │     └─────────────┘             └─────────────┘             │
   │                                                              │
   │   目标: 最大化累积奖励 Σ γ^t × r_t                          │
   │                                                              │
   └─────────────────────────────────────────────────────────────┘

关键概念
--------

- **状态（State）**: 环境的当前情况
- **动作（Action）**: 智能体可以采取的操作
- **奖励（Reward）**: 环境对动作的反馈
- **策略（Policy）**: 状态到动作的映射
- **价值函数（Value）**: 状态或动作的长期价值

强化学习 vs 其他学习
--------------------

.. csv-table::
   :header: "学习类型", "数据来源", "反馈形式", "例子"
   :widths: 20, 25, 25, 30

   "监督学习", "标注数据", "正确答案", "图像分类"
   "无监督学习", "无标注数据", "无反馈", "聚类"
   "强化学习", "交互产生", "奖励信号", "游戏、机器人"

马尔可夫决策过程（MDP）
=======================

强化学习问题通常建模为 MDP：

- **S**: 状态空间
- **A**: 动作空间
- **P(s'|s,a)**: 状态转移概率
- **R(s,a,s')**: 奖励函数
- **γ**: 折扣因子（0 < γ ≤ 1）

.. code-block:: python

   import numpy as np

   class SimpleMDP:
       """简单的 MDP 环境"""
       
       def __init__(self):
           # 状态: 0, 1, 2, 3, 4 (4 是终止状态)
           self.n_states = 5
           self.n_actions = 2  # 0: 左, 1: 右
           self.terminal_state = 4
           
           # 奖励: 到达终止状态得 +10
           self.rewards = {4: 10}
       
       def step(self, state, action):
           """执行动作，返回 (新状态, 奖励, 是否结束)"""
           if state == self.terminal_state:
               return state, 0, True
           
           # 状态转移
           if action == 0:  # 左
               new_state = max(0, state - 1)
           else:  # 右
               new_state = min(self.terminal_state, state + 1)
           
           reward = self.rewards.get(new_state, -1)  # 每步 -1 鼓励快速到达
           done = new_state == self.terminal_state
           
           return new_state, reward, done

价值函数
========

1. 状态价值函数 V(s)
--------------------

从状态 s 开始，遵循策略 π 的期望累积奖励。

.. code-block:: python

   def compute_state_values(mdp, policy, gamma=0.9, iterations=100):
       """策略评估: 计算状态价值"""
       V = np.zeros(mdp.n_states)
       
       for _ in range(iterations):
           V_new = np.zeros(mdp.n_states)
           
           for s in range(mdp.n_states):
               if s == mdp.terminal_state:
                   continue
               
               action = policy[s]
               next_state, reward, done = mdp.step(s, action)
               
               if done:
                   V_new[s] = reward
               else:
                   V_new[s] = reward + gamma * V[next_state]
           
           V = V_new
       
       return V

2. 动作价值函数 Q(s, a)
-----------------------

从状态 s 执行动作 a 后的期望累积奖励。

.. code-block:: python

   def compute_q_values(mdp, V, gamma=0.9):
       """计算 Q 值"""
       Q = np.zeros((mdp.n_states, mdp.n_actions))
       
       for s in range(mdp.n_states):
           for a in range(mdp.n_actions):
               next_state, reward, done = mdp.step(s, a)
               
               if done:
                   Q[s, a] = reward
               else:
                   Q[s, a] = reward + gamma * V[next_state]
       
       return Q

Q-Learning
==========

Q-Learning 是一种无模型的强化学习算法。

.. code-block:: python

   import numpy as np
   import random

   class QLearning:
       """Q-Learning 算法"""
       
       def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=1.0):
           self.n_states = n_states
           self.n_actions = n_actions
           self.lr = lr  # 学习率
           self.gamma = gamma  # 折扣因子
           self.epsilon = epsilon  # 探索率
           self.epsilon_decay = 0.995
           self.epsilon_min = 0.01
           
           # Q 表
           self.Q = np.zeros((n_states, n_actions))
       
       def select_action(self, state):
           """ε-贪婪策略选择动作"""
           if random.random() < self.epsilon:
               return random.randint(0, self.n_actions - 1)
           return np.argmax(self.Q[state])
       
       def learn(self, state, action, reward, next_state, done):
           """更新 Q 值"""
           if done:
               target = reward
           else:
               target = reward + self.gamma * np.max(self.Q[next_state])
           
           # Q-Learning 更新公式
           self.Q[state, action] += self.lr * (target - self.Q[state, action])
           
           # 衰减探索率
           self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

   # 训练
   def train_q_learning(env, agent, episodes=1000):
       rewards_history = []
       
       for episode in range(episodes):
           state = 0  # 初始状态
           total_reward = 0
           
           for step in range(100):
               action = agent.select_action(state)
               next_state, reward, done = env.step(state, action)
               
               agent.learn(state, action, reward, next_state, done)
               
               state = next_state
               total_reward += reward
               
               if done:
                   break
           
           rewards_history.append(total_reward)
           
           if (episode + 1) % 100 == 0:
               avg_reward = np.mean(rewards_history[-100:])
               print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
       
       return rewards_history

   # 运行
   env = SimpleMDP()
   agent = QLearning(env.n_states, env.n_actions)
   rewards = train_q_learning(env, agent)

   print("\n学到的 Q 表:")
   print(agent.Q)
   print("\n最优策略:")
   for s in range(env.n_states - 1):
       action = "右" if np.argmax(agent.Q[s]) == 1 else "左"
       print(f"  状态 {s}: {action}")

深度 Q 网络（DQN）
==================

用神经网络近似 Q 函数，处理大状态空间。

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from collections import deque
   import random

   class DQN(nn.Module):
       """深度 Q 网络"""
       
       def __init__(self, state_size, action_size):
           super().__init__()
           self.network = nn.Sequential(
               nn.Linear(state_size, 64),
               nn.ReLU(),
               nn.Linear(64, 64),
               nn.ReLU(),
               nn.Linear(64, action_size)
           )
       
       def forward(self, x):
           return self.network(x)

   class DQNAgent:
       """DQN 智能体"""
       
       def __init__(self, state_size, action_size):
           self.state_size = state_size
           self.action_size = action_size
           
           # 超参数
           self.gamma = 0.99
           self.epsilon = 1.0
           self.epsilon_decay = 0.995
           self.epsilon_min = 0.01
           self.batch_size = 32
           
           # 经验回放
           self.memory = deque(maxlen=10000)
           
           # 网络
           self.q_network = DQN(state_size, action_size)
           self.target_network = DQN(state_size, action_size)
           self.update_target_network()
           
           self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
       
       def update_target_network(self):
           """更新目标网络"""
           self.target_network.load_state_dict(self.q_network.state_dict())
       
       def remember(self, state, action, reward, next_state, done):
           """存储经验"""
           self.memory.append((state, action, reward, next_state, done))
       
       def select_action(self, state):
           """选择动作"""
           if random.random() < self.epsilon:
               return random.randint(0, self.action_size - 1)
           
           with torch.no_grad():
               state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
               q_values = self.q_network(state_tensor)
               return q_values.argmax().item()
       
       def replay(self):
           """经验回放学习"""
           if len(self.memory) < self.batch_size:
               return
           
           # 采样
           batch = random.sample(self.memory, self.batch_size)
           states, actions, rewards, next_states, dones = zip(*batch)
           
           states = torch.tensor(states, dtype=torch.float32)
           actions = torch.tensor(actions, dtype=torch.long)
           rewards = torch.tensor(rewards, dtype=torch.float32)
           next_states = torch.tensor(next_states, dtype=torch.float32)
           dones = torch.tensor(dones, dtype=torch.float32)
           
           # 计算目标 Q 值
           with torch.no_grad():
               next_q_values = self.target_network(next_states).max(1)[0]
               targets = rewards + (1 - dones) * self.gamma * next_q_values
           
           # 计算当前 Q 值
           current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
           
           # 更新网络
           loss = nn.MSELoss()(current_q_values, targets)
           
           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()
           
           # 衰减探索率
           self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
           
           return loss.item()

策略梯度方法
============

直接优化策略，而不是学习价值函数。

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.distributions import Categorical

   class PolicyNetwork(nn.Module):
       """策略网络"""
       
       def __init__(self, state_size, action_size):
           super().__init__()
           self.network = nn.Sequential(
               nn.Linear(state_size, 64),
               nn.ReLU(),
               nn.Linear(64, action_size),
               nn.Softmax(dim=-1)
           )
       
       def forward(self, x):
           return self.network(x)

   class REINFORCEAgent:
       """REINFORCE 算法"""
       
       def __init__(self, state_size, action_size):
           self.policy = PolicyNetwork(state_size, action_size)
           self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
           self.gamma = 0.99
           
           # 存储一个回合的数据
           self.log_probs = []
           self.rewards = []
       
       def select_action(self, state):
           """根据策略选择动作"""
           state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
           probs = self.policy(state_tensor)
           
           dist = Categorical(probs)
           action = dist.sample()
           
           self.log_probs.append(dist.log_prob(action))
           
           return action.item()
       
       def store_reward(self, reward):
           """存储奖励"""
           self.rewards.append(reward)
       
       def learn(self):
           """回合结束后学习"""
           # 计算折扣回报
           returns = []
           G = 0
           for r in reversed(self.rewards):
               G = r + self.gamma * G
               returns.insert(0, G)
           
           returns = torch.tensor(returns)
           returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化
           
           # 计算策略梯度损失
           policy_loss = []
           for log_prob, G in zip(self.log_probs, returns):
               policy_loss.append(-log_prob * G)
           
           loss = torch.stack(policy_loss).sum()
           
           # 更新策略
           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()
           
           # 清空缓存
           self.log_probs = []
           self.rewards = []
           
           return loss.item()

实战：CartPole 游戏
===================

.. code-block:: python

   import gym
   import torch
   import numpy as np

   # 创建环境
   # pip install gym
   env = gym.make('CartPole-v1')

   state_size = env.observation_space.shape[0]  # 4
   action_size = env.action_space.n  # 2

   # 使用 DQN 训练
   agent = DQNAgent(state_size, action_size)

   episodes = 500
   target_update_freq = 10

   for episode in range(episodes):
       state = env.reset()[0]
       total_reward = 0
       
       for step in range(500):
           action = agent.select_action(state)
           next_state, reward, done, truncated, info = env.step(action)
           
           agent.remember(state, action, reward, next_state, done)
           agent.replay()
           
           state = next_state
           total_reward += reward
           
           if done or truncated:
               break
       
       # 更新目标网络
       if (episode + 1) % target_update_freq == 0:
           agent.update_target_network()
       
       if (episode + 1) % 50 == 0:
           print(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

   env.close()

强化学习算法总结
================

.. csv-table::
   :header: "算法", "类型", "特点", "适用场景"
   :widths: 15, 20, 35, 30

   "Q-Learning", "值函数", "简单，离线学习", "小状态空间"
   "DQN", "值函数+深度学习", "处理大状态空间", "Atari 游戏"
   "REINFORCE", "策略梯度", "直接优化策略", "连续动作"
   "A2C/A3C", "Actor-Critic", "结合值和策略", "通用"
   "PPO", "策略梯度", "稳定，广泛使用", "机器人、游戏"

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "MDP", "马尔可夫决策过程，强化学习的数学框架"
   "策略", "状态到动作的映射"
   "价值函数", "状态或动作的长期价值"
   "探索与利用", "尝试新动作 vs 使用已知最优"
   "经验回放", "存储和重用过去的经验"
   "目标网络", "稳定 DQN 训练的技术"

总结
====

恭喜你完成了人工智能基础教程！

你已经学习了：

1. ✅ 人工智能的定义和历史
2. ✅ 智能体的概念和类型
3. ✅ 搜索算法（BFS、DFS、A*）
4. ✅ 知识表示与推理
5. ✅ 机器学习基础
6. ✅ 神经网络原理
7. ✅ PyTorch 深度学习实战
8. ✅ 自然语言处理基础
9. ✅ 计算机视觉基础
10. ✅ 强化学习基础

下一步建议
==========

1. **深入学习**: 选择感兴趣的方向深入研究
2. **实践项目**: 动手实现完整的 AI 项目
3. **阅读论文**: 了解最新研究进展
4. **参与社区**: 加入 AI 学习社区交流

🎉 祝你在 AI 学习之路上不断进步！
