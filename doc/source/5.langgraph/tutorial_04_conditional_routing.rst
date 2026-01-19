####################################
Tutorial 4: 条件路由
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

条件路由概述
============

条件路由让你的工作流能够根据状态动态选择执行路径。这是构建智能 Agent 的关键能力。

.. code-block:: text

   ┌─────────┐
   │  Node A │
   └────┬────┘
        │
        ▼
   ┌─────────┐      ┌─────────┐
   │ Router  │─────►│ Node B  │  (条件1)
   └────┬────┘      └─────────┘
        │
        │           ┌─────────┐
        └──────────►│ Node C  │  (条件2)
                    └─────────┘

基本条件路由
============

.. code-block:: python

   from langgraph.graph import StateGraph, START, END
   from typing import TypedDict

   class State(TypedDict):
       input: str
       category: str
       result: str

   def classify(state: State) -> dict:
       """分类节点"""
       input_text = state["input"]
       if "技术" in input_text or "编程" in input_text:
           category = "tech"
       elif "生活" in input_text:
           category = "life"
       else:
           category = "general"
       return {"category": category}

   def route_by_category(state: State) -> str:
       """路由函数：返回下一个节点名称"""
       return state["category"]

   def process_tech(state: State) -> dict:
       return {"result": "技术内容处理完成"}

   def process_life(state: State) -> dict:
       return {"result": "生活内容处理完成"}

   def process_general(state: State) -> dict:
       return {"result": "通用内容处理完成"}

   # 构建图
   graph = StateGraph(State)

   graph.add_node("classify", classify)
   graph.add_node("tech", process_tech)
   graph.add_node("life", process_life)
   graph.add_node("general", process_general)

   graph.add_edge(START, "classify")

   # 条件路由
   graph.add_conditional_edges(
       "classify",
       route_by_category,
       {
           "tech": "tech",
           "life": "life",
           "general": "general"
       }
   )

   graph.add_edge("tech", END)
   graph.add_edge("life", END)
   graph.add_edge("general", END)

   app = graph.compile()

使用 Literal 类型提示
=====================

.. code-block:: python

   from typing import Literal

   def route_by_score(state: State) -> Literal["pass", "fail", "review"]:
       """带类型提示的路由函数"""
       score = state["score"]
       if score >= 80:
           return "pass"
       elif score >= 60:
           return "review"
       else:
           return "fail"

   graph.add_conditional_edges(
       "evaluate",
       route_by_score,
       {
           "pass": "publish",
           "fail": "rewrite",
           "review": "human_review"
       }
   )

多条件组合路由
==============

.. code-block:: python

   from typing import TypedDict, Literal

   class ContentState(TypedDict):
       content_type: str
       quality_score: int
       platform: str
       is_urgent: bool

   def complex_router(state: ContentState) -> str:
       """复杂路由逻辑"""
       content_type = state["content_type"]
       score = state["quality_score"]
       is_urgent = state["is_urgent"]
       
       # 紧急内容直接发布（如果质量达标）
       if is_urgent and score >= 70:
           return "quick_publish"
       
       # 高质量内容
       if score >= 90:
           return "premium_publish"
       
       # 需要改进
       if score < 60:
           return "major_revision"
       
       # 一般内容
       return "standard_publish"

   graph.add_conditional_edges(
       "evaluate",
       complex_router,
       {
           "quick_publish": "publish_now",
           "premium_publish": "feature_content",
           "major_revision": "rewrite",
           "standard_publish": "schedule_publish"
       }
   )

使用 LLM 进行路由决策
=====================

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate

   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

   def llm_router(state: ContentState) -> str:
       """使用 LLM 进行智能路由"""
       content = state["content"]
       
       prompt = ChatPromptTemplate.from_template("""
   分析以下内容，决定最适合的处理方式：

   内容：{content}

   选项：
   - "publish": 内容质量高，可以直接发布
   - "edit": 内容需要小幅修改
   - "rewrite": 内容需要重写
   - "reject": 内容不适合发布

   只输出一个选项词。
   """)
       
       chain = prompt | llm
       response = chain.invoke({"content": content})
       decision = response.content.strip().lower()
       
       # 确保返回有效选项
       valid_options = ["publish", "edit", "rewrite", "reject"]
       if decision not in valid_options:
           decision = "edit"
       
       return decision

   graph.add_conditional_edges(
       "analyze",
       llm_router,
       {
           "publish": "publish_node",
           "edit": "edit_node",
           "rewrite": "rewrite_node",
           "reject": "reject_node"
       }
   )

实战：自媒体内容审核流程
========================

.. code-block:: python

   from typing import TypedDict, List, Literal, Optional
   from langgraph.graph import StateGraph, START, END
   from langchain_openai import ChatOpenAI

   # ========== 状态定义 ==========

   class ReviewState(TypedDict):
       content: str
       title: str
       platform: str
       
       # 审核结果
       quality_score: int
       originality_score: int
       compliance_check: bool
       issues: List[str]
       
       # 决策
       decision: str
       final_content: str

   # ========== 节点定义 ==========

   llm = ChatOpenAI(model="gpt-4o-mini")

   def check_quality(state: ReviewState) -> dict:
       """检查内容质量"""
       content = state["content"]
       
       response = llm.invoke(f"""
   评估以下内容的质量（0-100分）：

   {content[:1000]}

   评分维度：结构、深度、可读性
   只输出数字。
   """)
       
       try:
           score = int(response.content.strip())
       except:
           score = 70
       
       return {"quality_score": score}

   def check_originality(state: ReviewState) -> dict:
       """检查原创性"""
       content = state["content"]
       
       response = llm.invoke(f"""
   评估以下内容的原创性（0-100分）：

   {content[:1000]}

   考虑：是否有独特观点、是否像是抄袭
   只输出数字。
   """)
       
       try:
           score = int(response.content.strip())
       except:
           score = 80
       
       return {"originality_score": score}

   def check_compliance(state: ReviewState) -> dict:
       """检查合规性"""
       content = state["content"]
       
       response = llm.invoke(f"""
   检查以下内容是否有合规问题：

   {content[:1000]}

   检查项：敏感词、虚假信息、违规内容
   
   如果没有问题输出 "PASS"
   如果有问题输出问题列表，每行一个
   """)
       
       result = response.content.strip()
       if result.upper() == "PASS":
           return {"compliance_check": True, "issues": []}
       else:
           issues = [line.strip() for line in result.split("\n") if line.strip()]
           return {"compliance_check": False, "issues": issues}

   def make_decision(state: ReviewState) -> dict:
       """综合决策"""
       quality = state["quality_score"]
       originality = state["originality_score"]
       compliance = state["compliance_check"]
       
       if not compliance:
           decision = "reject"
       elif quality >= 80 and originality >= 70:
           decision = "approve"
       elif quality >= 60 and originality >= 50:
           decision = "revise"
       else:
           decision = "reject"
       
       return {"decision": decision}

   def approve_content(state: ReviewState) -> dict:
       """批准发布"""
       return {"final_content": state["content"]}

   def request_revision(state: ReviewState) -> dict:
       """请求修改"""
       issues = state.get("issues", [])
       quality = state["quality_score"]
       
       feedback = f"质量分数：{quality}/100\n"
       if issues:
           feedback += f"问题：{', '.join(issues)}\n"
       feedback += "请修改后重新提交。"
       
       return {"final_content": f"[需要修改]\n{feedback}"}

   def reject_content(state: ReviewState) -> dict:
       """拒绝发布"""
       issues = state.get("issues", [])
       reason = ", ".join(issues) if issues else "质量不达标"
       return {"final_content": f"[已拒绝] 原因：{reason}"}

   # ========== 路由函数 ==========

   def decision_router(state: ReviewState) -> Literal["approve", "revise", "reject"]:
       return state["decision"]

   # ========== 构建图 ==========

   def create_review_workflow():
       graph = StateGraph(ReviewState)
       
       # 添加审核节点
       graph.add_node("check_quality", check_quality)
       graph.add_node("check_originality", check_originality)
       graph.add_node("check_compliance", check_compliance)
       graph.add_node("make_decision", make_decision)
       
       # 添加处理节点
       graph.add_node("approve", approve_content)
       graph.add_node("revise", request_revision)
       graph.add_node("reject", reject_content)
       
       # 并行审核流程
       graph.add_edge(START, "check_quality")
       graph.add_edge(START, "check_originality")
       graph.add_edge(START, "check_compliance")
       
       # 汇总决策
       graph.add_edge("check_quality", "make_decision")
       graph.add_edge("check_originality", "make_decision")
       graph.add_edge("check_compliance", "make_decision")
       
       # 条件路由
       graph.add_conditional_edges(
           "make_decision",
           decision_router,
           {
               "approve": "approve",
               "revise": "revise",
               "reject": "reject"
           }
       )
       
       # 结束
       graph.add_edge("approve", END)
       graph.add_edge("revise", END)
       graph.add_edge("reject", END)
       
       return graph.compile()

   # ========== 运行 ==========

   workflow = create_review_workflow()

   result = workflow.invoke({
       "content": """
       AI编程正在改变软件开发的方式。通过使用AI辅助工具，
       开发者可以更快地编写代码、发现bug、优化性能。
       本文将介绍几个实用的AI编程技巧，帮助你提升开发效率。
       """,
       "title": "AI编程技巧分享",
       "platform": "微信公众号"
   })

   print(f"质量分数: {result['quality_score']}")
   print(f"原创性分数: {result['originality_score']}")
   print(f"合规检查: {'通过' if result['compliance_check'] else '未通过'}")
   print(f"决策: {result['decision']}")
   print(f"结果: {result['final_content']}")

路由最佳实践
============

1. **明确的路由条件**

.. code-block:: python

   # ✅ 好：条件清晰
   def router(state):
       if state["score"] >= 80:
           return "high"
       return "low"

   # ❌ 差：条件模糊
   def router(state):
       if state["score"] > 75 and state["score"] < 85:
           return "maybe_high"  # 边界不清

2. **处理所有情况**

.. code-block:: python

   # ✅ 好：有默认处理
   def router(state):
       mapping = {"a": "node_a", "b": "node_b"}
       return mapping.get(state["type"], "default_node")

3. **类型安全**

.. code-block:: python

   from typing import Literal

   # ✅ 好：使用 Literal 类型
   def router(state) -> Literal["approve", "reject", "review"]:
       ...

下一步
======

在下一个教程中，我们将学习如何实现循环和迭代逻辑。

:doc:`tutorial_05_cycles_loops`
