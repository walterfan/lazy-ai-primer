####################################
Tutorial 5: 循环与迭代
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

循环的重要性
============

在 AI Agent 中，循环是实现以下功能的关键：

- **迭代优化**: 反复改进内容直到满意
- **重试机制**: 失败后自动重试
- **对话循环**: 持续的人机交互
- **自我纠错**: Agent 检查并修正自己的输出

LangGraph 中的循环
==================

LangGraph 原生支持循环，通过条件边实现：

.. code-block:: text

   ┌─────────┐
   │  START  │
   └────┬────┘
        │
        ▼
   ┌─────────┐     ┌─────────┐
   │ Process │────►│ Evaluate│
   └─────────┘     └────┬────┘
        ▲               │
        │    ┌──────────┴──────────┐
        │    │                     │
        │    ▼                     ▼
        │  不满意               满意
        │    │                     │
        └────┘                     ▼
                              ┌─────────┐
                              │   END   │
                              └─────────┘

基本循环模式
============

.. code-block:: python

   from typing import TypedDict, Literal
   from langgraph.graph import StateGraph, START, END

   class State(TypedDict):
       content: str
       iteration: int
       max_iterations: int
       is_satisfied: bool

   def process(state: State) -> dict:
       """处理节点"""
       content = state["content"]
       iteration = state.get("iteration", 0) + 1
       
       # 模拟处理
       improved_content = f"{content} [改进 #{iteration}]"
       
       return {
           "content": improved_content,
           "iteration": iteration
       }

   def evaluate(state: State) -> dict:
       """评估节点"""
       iteration = state["iteration"]
       max_iter = state.get("max_iterations", 3)
       
       # 模拟评估：达到最大迭代次数则满意
       is_satisfied = iteration >= max_iter
       
       return {"is_satisfied": is_satisfied}

   def should_continue(state: State) -> Literal["continue", "end"]:
       """路由函数"""
       if state["is_satisfied"]:
           return "end"
       return "continue"

   # 构建图
   graph = StateGraph(State)

   graph.add_node("process", process)
   graph.add_node("evaluate", evaluate)

   graph.add_edge(START, "process")
   graph.add_edge("process", "evaluate")

   graph.add_conditional_edges(
       "evaluate",
       should_continue,
       {
           "continue": "process",  # 循环回到 process
           "end": END
       }
   )

   app = graph.compile()

   result = app.invoke({
       "content": "初始内容",
       "iteration": 0,
       "max_iterations": 3,
       "is_satisfied": False
   })

   print(f"最终内容: {result['content']}")
   print(f"迭代次数: {result['iteration']}")

防止无限循环
============

1. **最大迭代次数限制**

.. code-block:: python

   def should_continue(state: State) -> str:
       max_iterations = state.get("max_iterations", 5)
       current = state.get("iteration", 0)
       
       if current >= max_iterations:
           return "end"  # 强制结束
       
       if state["is_satisfied"]:
           return "end"
       
       return "continue"

2. **使用图的 recursion_limit**

.. code-block:: python

   # 编译时设置递归限制
   app = graph.compile()
   
   # 运行时设置
   result = app.invoke(
       initial_state,
       {"recursion_limit": 10}  # 最多执行10步
   )

实战：内容迭代优化系统
======================

.. code-block:: python

   from typing import TypedDict, List, Literal, Annotated
   from operator import add
   from langgraph.graph import StateGraph, START, END
   from langchain_openai import ChatOpenAI

   # ========== 状态定义 ==========

   class ContentState(TypedDict):
       topic: str
       platform: str
       current_content: str
       quality_score: int
       target_score: int
       iteration: int
       max_iterations: int
       improvement_history: Annotated[List[dict], add]

   # ========== 节点定义 ==========

   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

   def generate_initial(state: ContentState) -> dict:
       """生成初始内容"""
       topic = state["topic"]
       platform = state["platform"]
       
       response = llm.invoke(f"""
   为「{platform}」写一篇关于「{topic}」的短文（300字左右）。
   """)
       
       return {
           "current_content": response.content,
           "iteration": 1
       }

   def evaluate_content(state: ContentState) -> dict:
       """评估内容质量"""
       content = state["current_content"]
       
       response = llm.invoke(f"""
   评估以下内容的质量（0-100分）：

   {content}

   评分标准：
   - 内容深度 (40分)
   - 结构清晰 (30分)
   - 语言表达 (30分)

   输出格式：
   分数: [数字]
   问题: [主要问题，一句话]
   """)
       
       result = response.content
       
       # 解析分数
       try:
           score_line = [l for l in result.split("\n") if "分数" in l][0]
           score = int(''.join(filter(str.isdigit, score_line)))
       except:
           score = 70
       
       # 解析问题
       try:
           issue_line = [l for l in result.split("\n") if "问题" in l][0]
           issue = issue_line.split(":")[-1].strip()
       except:
           issue = "需要整体优化"
       
       return {
           "quality_score": score,
           "improvement_history": [{
               "iteration": state["iteration"],
               "score": score,
               "issue": issue
           }]
       }

   def improve_content(state: ContentState) -> dict:
       """改进内容"""
       content = state["current_content"]
       history = state["improvement_history"]
       
       # 获取最新的问题
       latest_issue = history[-1]["issue"] if history else "整体优化"
       
       response = llm.invoke(f"""
   请改进以下内容：

   原文：
   {content}

   需要改进的问题：{latest_issue}

   要求：
   1. 针对问题进行改进
   2. 保持原有优点
   3. 输出完整的改进后内容
   """)
       
       return {
           "current_content": response.content,
           "iteration": state["iteration"] + 1
       }

   # ========== 路由函数 ==========

   def should_continue(state: ContentState) -> Literal["improve", "done"]:
       """决定是否继续优化"""
       score = state["quality_score"]
       target = state.get("target_score", 85)
       iteration = state["iteration"]
       max_iter = state.get("max_iterations", 5)
       
       # 达到目标分数或最大迭代次数
       if score >= target:
           return "done"
       if iteration >= max_iter:
           return "done"
       
       return "improve"

   # ========== 构建图 ==========

   def create_improvement_workflow():
       graph = StateGraph(ContentState)
       
       graph.add_node("generate", generate_initial)
       graph.add_node("evaluate", evaluate_content)
       graph.add_node("improve", improve_content)
       
       # 流程：生成 -> 评估 -> (改进 -> 评估) 循环
       graph.add_edge(START, "generate")
       graph.add_edge("generate", "evaluate")
       
       graph.add_conditional_edges(
           "evaluate",
           should_continue,
           {
               "improve": "improve",
               "done": END
           }
       )
       
       graph.add_edge("improve", "evaluate")
       
       return graph.compile()

   # ========== 运行 ==========

   workflow = create_improvement_workflow()

   # 流式查看每步结果
   print("开始内容优化流程...")
   print("=" * 50)

   for event in workflow.stream({
       "topic": "AI编程入门",
       "platform": "微信公众号",
       "target_score": 85,
       "max_iterations": 3,
       "iteration": 0,
       "improvement_history": []
   }):
       for node, output in event.items():
           if node == "evaluate":
               print(f"\n[评估] 第 {output.get('improvement_history', [{}])[-1].get('iteration', '?')} 轮")
               print(f"  分数: {output.get('quality_score', 'N/A')}")
               if output.get('improvement_history'):
                   print(f"  问题: {output['improvement_history'][-1].get('issue', 'N/A')}")
           elif node == "improve":
               print(f"\n[改进] 第 {output.get('iteration', '?')} 轮优化完成")

   print("\n" + "=" * 50)
   print("优化完成！")

带重试的循环
============

.. code-block:: python

   class RetryState(TypedDict):
       task: str
       result: str
       error: str
       retry_count: int
       max_retries: int
       success: bool

   def execute_task(state: RetryState) -> dict:
       """执行任务（可能失败）"""
       try:
           # 模拟可能失败的操作
           result = risky_operation(state["task"])
           return {"result": result, "success": True, "error": ""}
       except Exception as e:
           return {
               "error": str(e),
               "success": False,
               "retry_count": state.get("retry_count", 0) + 1
           }

   def should_retry(state: RetryState) -> Literal["retry", "success", "fail"]:
       """决定是否重试"""
       if state["success"]:
           return "success"
       
       if state["retry_count"] >= state.get("max_retries", 3):
           return "fail"
       
       return "retry"

   graph.add_conditional_edges(
       "execute",
       should_retry,
       {
           "retry": "execute",  # 重试
           "success": "complete",
           "fail": "handle_error"
       }
   )

循环最佳实践
============

1. **始终设置终止条件**

.. code-block:: python

   def should_continue(state):
       # 多重终止条件
       if state["iteration"] >= state["max_iterations"]:
           return "end"
       if state["is_complete"]:
           return "end"
       if state.get("error_count", 0) >= 3:
           return "end"
       return "continue"

2. **记录迭代历史**

.. code-block:: python

   class State(TypedDict):
       history: Annotated[List[dict], add]  # 追加模式

   def process(state):
       return {
           "history": [{
               "iteration": state["iteration"],
               "timestamp": datetime.now().isoformat(),
               "action": "processed"
           }]
       }

3. **提供进度反馈**

.. code-block:: python

   for event in workflow.stream(initial_state):
       for node, output in event.items():
           print(f"[{node}] 完成 - 迭代 {output.get('iteration', '?')}")

下一步
======

在下一个教程中，我们将学习如何实现人工干预（Human-in-the-Loop）。

:doc:`tutorial_06_human_in_loop`
