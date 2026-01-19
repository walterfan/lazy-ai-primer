####################################
Tutorial 3: 搜索算法
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是搜索问题？
================

很多 AI 问题可以建模为 **搜索问题**：从初始状态出发，通过一系列行动，到达目标状态。

搜索问题的组成：

- **初始状态**: 起点
- **行动**: 可以采取的操作
- **转移模型**: 行动如何改变状态
- **目标测试**: 判断是否到达目标
- **路径代价**: 行动的代价

.. code-block:: python

   class SearchProblem:
       """搜索问题的抽象定义"""
       
       def __init__(self, initial_state, goal_state):
           self.initial_state = initial_state
           self.goal_state = goal_state
       
       def get_actions(self, state):
           """返回当前状态可用的行动"""
           raise NotImplementedError
       
       def get_result(self, state, action):
           """返回执行行动后的新状态"""
           raise NotImplementedError
       
       def is_goal(self, state):
           """判断是否到达目标"""
           return state == self.goal_state
       
       def get_cost(self, state, action, next_state):
           """返回行动的代价"""
           return 1  # 默认代价为1

经典例子：八数码问题
--------------------

.. code-block:: text

   初始状态:        目标状态:
   ┌───┬───┬───┐   ┌───┬───┬───┐
   │ 1 │ 2 │ 3 │   │ 1 │ 2 │ 3 │
   ├───┼───┼───┤   ├───┼───┼───┤
   │ 4 │   │ 5 │ → │ 4 │ 5 │ 6 │
   ├───┼───┼───┤   ├───┼───┼───┤
   │ 6 │ 7 │ 8 │   │ 7 │ 8 │   │
   └───┴───┴───┘   └───┴───┴───┘

无信息搜索（盲目搜索）
======================

1. 广度优先搜索（BFS）
----------------------

**策略**: 先扩展浅层节点，逐层搜索

.. code-block:: python

   from collections import deque

   def bfs(problem):
       """广度优先搜索"""
       # 边界：待探索的节点队列
       frontier = deque([(problem.initial_state, [])])
       # 已探索的状态
       explored = set()
       
       while frontier:
           state, path = frontier.popleft()
           
           # 目标测试
           if problem.is_goal(state):
               return path
           
           # 标记为已探索
           explored.add(state)
           
           # 扩展节点
           for action in problem.get_actions(state):
               next_state = problem.get_result(state, action)
               
               if next_state not in explored:
                   frontier.append((next_state, path + [action]))
       
       return None  # 无解

   # 示例：迷宫问题
   class MazeProblem(SearchProblem):
       def __init__(self, maze, start, goal):
           super().__init__(start, goal)
           self.maze = maze
           self.rows = len(maze)
           self.cols = len(maze[0])
       
       def get_actions(self, state):
           actions = []
           x, y = state
           moves = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]
           
           for name, dx, dy in moves:
               nx, ny = x + dx, y + dy
               if 0 <= nx < self.rows and 0 <= ny < self.cols:
                   if self.maze[nx][ny] != 1:  # 1 表示墙
                       actions.append(name)
           return actions
       
       def get_result(self, state, action):
           x, y = state
           moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
           dx, dy = moves[action]
           return (x + dx, y + dy)

   # 测试
   maze = [
       [0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 1, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0]
   ]

   problem = MazeProblem(maze, (0, 0), (4, 4))
   path = bfs(problem)
   print(f"BFS 找到的路径: {path}")

**特点**:

- ✅ 完备性：如果有解，一定能找到
- ✅ 最优性：找到的是最短路径（代价相同时）
- ❌ 空间复杂度高：O(b^d)，b是分支因子，d是深度

2. 深度优先搜索（DFS）
----------------------

**策略**: 先扩展最深的节点

.. code-block:: python

   def dfs(problem, max_depth=float('inf')):
       """深度优先搜索"""
       
       def recursive_dfs(state, path, depth):
           if problem.is_goal(state):
               return path
           
           if depth >= max_depth:
               return None
           
           for action in problem.get_actions(state):
               next_state = problem.get_result(state, action)
               
               if next_state not in path:  # 避免循环
                   result = recursive_dfs(
                       next_state,
                       path + [action],
                       depth + 1
                   )
                   if result is not None:
                       return result
           
           return None
       
       return recursive_dfs(problem.initial_state, [], 0)

   # 迭代加深搜索：结合 BFS 和 DFS 的优点
   def iterative_deepening_dfs(problem, max_depth=100):
       """迭代加深搜索"""
       for depth in range(max_depth):
           result = dfs(problem, depth)
           if result is not None:
               return result
       return None

**特点**:

- ✅ 空间复杂度低：O(bd)
- ❌ 不完备：可能陷入无限循环
- ❌ 不最优：可能找到较长的路径

有信息搜索（启发式搜索）
========================

启发式搜索使用 **启发函数 h(n)** 来估计从节点 n 到目标的代价。

1. 贪婪最佳优先搜索
-------------------

**策略**: 总是扩展 h(n) 最小的节点

.. code-block:: python

   import heapq

   def greedy_best_first(problem, heuristic):
       """贪婪最佳优先搜索"""
       frontier = [(heuristic(problem.initial_state), problem.initial_state, [])]
       explored = set()
       
       while frontier:
           _, state, path = heapq.heappop(frontier)
           
           if problem.is_goal(state):
               return path
           
           if state in explored:
               continue
           explored.add(state)
           
           for action in problem.get_actions(state):
               next_state = problem.get_result(state, action)
               if next_state not in explored:
                   h = heuristic(next_state)
                   heapq.heappush(frontier, (h, next_state, path + [action]))
       
       return None

2. A* 搜索
----------

**策略**: 使用 f(n) = g(n) + h(n)

- g(n): 从起点到 n 的实际代价
- h(n): 从 n 到目标的估计代价
- f(n): 估计的总代价

.. code-block:: python

   def astar(problem, heuristic):
       """A* 搜索算法"""
       # (f值, g值, 状态, 路径)
       frontier = [(heuristic(problem.initial_state), 0, problem.initial_state, [])]
       explored = {}  # 状态 -> 最小g值
       
       while frontier:
           f, g, state, path = heapq.heappop(frontier)
           
           if problem.is_goal(state):
               return path, g  # 返回路径和代价
           
           # 如果已经用更小的代价访问过，跳过
           if state in explored and explored[state] <= g:
               continue
           explored[state] = g
           
           for action in problem.get_actions(state):
               next_state = problem.get_result(state, action)
               next_g = g + problem.get_cost(state, action, next_state)
               
               if next_state not in explored or explored[next_state] > next_g:
                   h = heuristic(next_state)
                   f = next_g + h
                   heapq.heappush(frontier, (f, next_g, next_state, path + [action]))
       
       return None, float('inf')

   # 曼哈顿距离启发函数
   def manhattan_distance(state, goal):
       """曼哈顿距离：适用于网格世界"""
       return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

   # 测试 A*
   problem = MazeProblem(maze, (0, 0), (4, 4))
   goal = (4, 4)
   heuristic = lambda s: manhattan_distance(s, goal)

   path, cost = astar(problem, heuristic)
   print(f"A* 找到的路径: {path}")
   print(f"路径代价: {cost}")

启发函数的性质
--------------

**可采纳性（Admissibility）**: h(n) ≤ h*(n)

- h(n) 从不高估到目标的真实代价
- 保证 A* 找到最优解

**一致性（Consistency）**: h(n) ≤ c(n, a, n') + h(n')

- 满足三角不等式
- 一致性蕴含可采纳性

实战：用 PyTorch 学习启发函数
=============================

我们可以用神经网络来学习启发函数：

.. code-block:: python

   import torch
   import torch.nn as nn
   import random

   class LearnedHeuristic(nn.Module):
       """学习的启发函数"""
       
       def __init__(self, state_size):
           super().__init__()
           self.network = nn.Sequential(
               nn.Linear(state_size, 64),
               nn.ReLU(),
               nn.Linear(64, 32),
               nn.ReLU(),
               nn.Linear(32, 1),
               nn.ReLU()  # 启发值应该非负
           )
       
       def forward(self, state):
           return self.network(state)

   def generate_training_data(maze, num_samples=1000):
       """生成训练数据：(状态, 到目标的真实距离)"""
       rows, cols = len(maze), len(maze[0])
       goal = (rows-1, cols-1)
       
       data = []
       
       for _ in range(num_samples):
           # 随机选择一个非墙壁的位置
           while True:
               x = random.randint(0, rows-1)
               y = random.randint(0, cols-1)
               if maze[x][y] != 1:
                   break
           
           # 用 BFS 计算真实距离
           problem = MazeProblem(maze, (x, y), goal)
           path = bfs(problem)
           
           if path is not None:
               true_distance = len(path)
               state = torch.tensor([x/rows, y/cols, goal[0]/rows, goal[1]/cols], 
                                   dtype=torch.float32)
               data.append((state, true_distance))
       
       return data

   def train_heuristic(maze, epochs=100):
       """训练启发函数"""
       model = LearnedHeuristic(state_size=4)
       optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
       criterion = nn.MSELoss()
       
       # 生成训练数据
       print("生成训练数据...")
       data = generate_training_data(maze, num_samples=500)
       
       print(f"训练数据量: {len(data)}")
       
       # 训练
       for epoch in range(epochs):
           total_loss = 0
           random.shuffle(data)
           
           for state, true_dist in data:
               predicted = model(state)
               loss = criterion(predicted, torch.tensor([[true_dist]], dtype=torch.float32))
               
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           if (epoch + 1) % 20 == 0:
               print(f"Epoch {epoch+1}, Loss: {total_loss/len(data):.4f}")
       
       return model

   # 训练并使用学习的启发函数
   if __name__ == "__main__":
       maze = [
           [0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0]
       ]
       
       print("训练启发函数...")
       learned_h = train_heuristic(maze)
       
       # 测试
       print("\n测试学习的启发函数:")
       test_states = [(0, 0), (2, 2), (3, 3)]
       goal = (4, 4)
       
       for state in test_states:
           x, y = state
           state_tensor = torch.tensor([x/5, y/5, goal[0]/5, goal[1]/5], dtype=torch.float32)
           
           learned_h_value = learned_h(state_tensor).item()
           true_h_value = manhattan_distance(state, goal)
           
           print(f"状态 {state}: 学习的h={learned_h_value:.2f}, 曼哈顿距离={true_h_value}")

搜索算法比较
============

.. csv-table::
   :header: "算法", "完备性", "最优性", "时间复杂度", "空间复杂度"
   :widths: 20, 15, 15, 25, 25

   "BFS", "是", "是*", "O(b^d)", "O(b^d)"
   "DFS", "否", "否", "O(b^m)", "O(bm)"
   "迭代加深", "是", "是*", "O(b^d)", "O(bd)"
   "贪婪最佳优先", "否", "否", "O(b^m)", "O(b^m)"
   "A*", "是", "是", "O(b^d)", "O(b^d)"

*注：当所有边代价相同时*

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "搜索问题", "从初始状态通过行动到达目标状态"
   "状态空间", "所有可能状态的集合"
   "搜索树", "搜索过程中生成的树结构"
   "启发函数", "估计到目标代价的函数"
   "可采纳性", "启发函数不高估真实代价"
   "A* 算法", "使用 f(n)=g(n)+h(n) 的最优搜索"

下一步
======

在下一个教程中，我们将学习知识表示与推理。

:doc:`tutorial_04_knowledge_representation`
