####################################
Tutorial 6: Agents 与 Tools
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是 Agent？
==============

Agent 是一种让 LLM 能够自主决策并执行操作的架构。与普通 Chain 不同：

- **Chain**: 预定义的固定流程
- **Agent**: LLM 自主决定下一步做什么

Agent 的核心能力：

1. **推理** - 分析任务，制定计划
2. **行动** - 调用 Tools 执行操作
3. **观察** - 获取执行结果
4. **迭代** - 根据结果调整策略

什么是 Tool？
=============

Tool 是 Agent 可以调用的外部功能，例如：

- 搜索引擎
- 数据库查询
- API 调用
- 文件操作
- 代码执行

创建自定义 Tool
===============

使用装饰器
----------

.. code-block:: python

   from langchain_core.tools import tool

   @tool
   def search_trending_topics(category: str) -> str:
       """搜索指定分类的热门话题
       
       Args:
           category: 话题分类，如 "科技"、"生活"、"职场"
       """
       # 模拟搜索结果
       topics = {
           "科技": ["AI编程入门", "ChatGPT使用技巧", "Python自动化"],
           "生活": ["高效学习方法", "时间管理", "健康饮食"],
           "职场": ["面试技巧", "职场沟通", "副业赚钱"]
       }
       return f"热门话题: {', '.join(topics.get(category, ['暂无数据']))}"

   @tool
   def get_platform_requirements(platform: str) -> str:
       """获取自媒体平台的内容要求
       
       Args:
           platform: 平台名称，如 "微信公众号"、"知乎"、"小红书"
       """
       requirements = {
           "微信公众号": "字数: 1500-3000字，风格: 深度、有温度，支持图文混排",
           "知乎": "字数: 不限，风格: 专业、有理有据，适合深度分析",
           "小红书": "字数: 500-1000字，风格: 活泼、口语化，多用emoji和图片",
       }
       return requirements.get(platform, "未找到该平台信息")

   @tool
   def calculate_reading_time(word_count: int) -> str:
       """计算文章预计阅读时间
       
       Args:
           word_count: 文章字数
       """
       minutes = word_count / 300  # 假设每分钟阅读300字
       return f"预计阅读时间: {minutes:.1f} 分钟"

使用 StructuredTool
-------------------

.. code-block:: python

   from langchain_core.tools import StructuredTool
   from pydantic import BaseModel, Field

   class ArticleInput(BaseModel):
       title: str = Field(description="文章标题")
       platform: str = Field(description="目标平台")
       word_count: int = Field(description="目标字数")

   def generate_article_outline(title: str, platform: str, word_count: int) -> str:
       """根据参数生成文章大纲"""
       sections = word_count // 500  # 每500字一个章节
       return f"""
   文章大纲 - {title}
   平台: {platform}
   预计章节数: {sections}
   
   1. 引言 - 引出话题
   2. 主体部分 ({sections-2} 个章节)
   3. 总结与行动号召
   """

   outline_tool = StructuredTool.from_function(
       func=generate_article_outline,
       name="generate_outline",
       description="根据标题、平台和字数生成文章大纲",
       args_schema=ArticleInput
   )

创建 Agent
==========

使用 create_react_agent
-----------------------

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain.agents import create_react_agent, AgentExecutor
   from langchain_core.prompts import ChatPromptTemplate
   from langchain import hub

   # 准备工具
   tools = [search_trending_topics, get_platform_requirements, calculate_reading_time]

   # 获取 ReAct prompt
   prompt = hub.pull("hwchase17/react")

   # 创建 LLM
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

   # 创建 Agent
   agent = create_react_agent(llm, tools, prompt)

   # 创建执行器
   agent_executor = AgentExecutor(
       agent=agent,
       tools=tools,
       verbose=True,  # 显示推理过程
       handle_parsing_errors=True
   )

   # 运行
   result = agent_executor.invoke({
       "input": "帮我找一个科技类的热门话题，然后告诉我在微信公众号发布需要注意什么"
   })

   print(result["output"])

使用 Tool Calling Agent（推荐）
-------------------------------

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain.agents import create_tool_calling_agent, AgentExecutor
   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

   # 创建 prompt
   prompt = ChatPromptTemplate.from_messages([
       ("system", """你是自媒体内容策划助手。
   你可以使用工具来帮助用户：
   1. 搜索热门话题
   2. 查询平台要求
   3. 计算阅读时间

   请根据用户需求，合理使用工具完成任务。"""),
       MessagesPlaceholder(variable_name="chat_history", optional=True),
       ("human", "{input}"),
       MessagesPlaceholder(variable_name="agent_scratchpad")
   ])

   # 创建支持 tool calling 的 LLM
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

   # 创建 Agent
   agent = create_tool_calling_agent(llm, tools, prompt)

   # 创建执行器
   agent_executor = AgentExecutor(
       agent=agent,
       tools=tools,
       verbose=True
   )

   # 运行
   result = agent_executor.invoke({
       "input": "我想在知乎写一篇2000字的科技文章，帮我规划一下"
   })

实战：自媒体内容规划 Agent
==========================

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain_core.tools import tool
   from langchain.agents import create_tool_calling_agent, AgentExecutor
   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
   from datetime import datetime
   import json

   # ========== 定义工具 ==========

   @tool
   def search_hot_topics(category: str, count: int = 5) -> str:
       """搜索热门话题
       
       Args:
           category: 话题分类
           count: 返回数量
       """
       # 实际应用中可以调用真实的热搜API
       topics = {
           "科技": [
               "AI Agent 开发入门",
               "Claude 3.5 vs GPT-4o 对比",
               "Cursor IDE 使用技巧",
               "Python 自动化办公",
               "向量数据库选型指南"
           ],
           "职场": [
               "35岁程序员转型方向",
               "远程办公效率提升",
               "技术面试准备指南",
               "副业变现实战",
               "职场沟通技巧"
           ]
       }
       result = topics.get(category, ["暂无数据"])[:count]
       return json.dumps({"category": category, "topics": result}, ensure_ascii=False)

   @tool
   def analyze_topic_potential(topic: str) -> str:
       """分析话题的内容潜力
       
       Args:
           topic: 话题名称
       """
       # 模拟分析结果
       return json.dumps({
           "topic": topic,
           "search_volume": "高",
           "competition": "中等",
           "trend": "上升",
           "recommended_platforms": ["知乎", "微信公众号"],
           "content_angles": [
               "入门教程",
               "实战案例",
               "避坑指南",
               "工具推荐"
           ]
       }, ensure_ascii=False)

   @tool
   def create_content_calendar(topics: str, days: int = 7) -> str:
       """创建内容发布日历
       
       Args:
           topics: 逗号分隔的话题列表
           days: 规划天数
       """
       topic_list = [t.strip() for t in topics.split(",")]
       calendar = []
       
       for i in range(min(days, len(topic_list))):
           calendar.append({
               "day": i + 1,
               "topic": topic_list[i % len(topic_list)],
               "platform": ["微信公众号", "知乎", "小红书"][i % 3],
               "content_type": ["深度文章", "教程", "短内容"][i % 3]
           })
       
       return json.dumps({"calendar": calendar}, ensure_ascii=False, indent=2)

   @tool
   def get_writing_tips(platform: str, content_type: str) -> str:
       """获取特定平台和内容类型的写作建议
       
       Args:
           platform: 目标平台
           content_type: 内容类型（文章/视频脚本/短内容）
       """
       tips = {
           ("微信公众号", "文章"): [
               "开头要有钩子，3秒内抓住读者",
               "使用小标题分段，提升可读性",
               "每段不超过3行，适合手机阅读",
               "结尾加入互动问题和关注引导"
           ],
           ("知乎", "文章"): [
               "开头直接给出核心观点",
               "使用数据和案例支撑论点",
               "适当引用权威来源",
               "回答要有独特视角"
           ],
           ("小红书", "短内容"): [
               "标题要有emoji，吸引眼球",
               "图片比文字更重要",
               "口语化表达，像朋友聊天",
               "加入实用的干货清单"
           ]
       }
       
       key = (platform, content_type)
       result = tips.get(key, ["暂无针对性建议"])
       return json.dumps({"tips": result}, ensure_ascii=False)

   # ========== 创建 Agent ==========

   class ContentPlanningAgent:
       def __init__(self):
           self.tools = [
               search_hot_topics,
               analyze_topic_potential,
               create_content_calendar,
               get_writing_tips
           ]
           
           self.prompt = ChatPromptTemplate.from_messages([
               ("system", """你是专业的自媒体内容策划师。你的职责是：

   1. 帮助用户发现热门话题
   2. 分析话题的内容潜力
   3. 制定内容发布计划
   4. 提供写作建议

   工作流程：
   - 先了解用户的领域和目标平台
   - 搜索相关热门话题
   - 分析最有潜力的话题
   - 制定内容日历
   - 给出具体的写作建议

   请充分利用工具，给出专业、可执行的建议。"""),
               MessagesPlaceholder(variable_name="chat_history", optional=True),
               ("human", "{input}"),
               MessagesPlaceholder(variable_name="agent_scratchpad")
           ])
           
           self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
           self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
           self.executor = AgentExecutor(
               agent=self.agent,
               tools=self.tools,
               verbose=True,
               max_iterations=10
           )
       
       def plan(self, request: str) -> str:
           result = self.executor.invoke({"input": request})
           return result["output"]

   # ========== 使用示例 ==========

   agent = ContentPlanningAgent()

   # 场景1：寻找话题
   result = agent.plan("我是一个科技博主，帮我找一些最近的热门话题")
   print(result)

   # 场景2：完整规划
   result = agent.plan("""
   我想开始做自媒体，主要方向是科技和编程。
   目标平台是知乎和微信公众号。
   请帮我：
   1. 找5个适合的话题
   2. 分析最有潜力的话题
   3. 制定一周的内容计划
   4. 给出写作建议
   """)
   print(result)

Agent 执行流程
==============

.. code-block:: text

   用户输入
      │
      ▼
   ┌─────────────────────────────────────┐
   │              Agent                   │
   │  ┌─────────────────────────────┐    │
   │  │     LLM (思考/决策)          │    │
   │  └──────────────┬──────────────┘    │
   │                 │                    │
   │    ┌────────────┴────────────┐      │
   │    ▼                         ▼      │
   │  调用工具                 直接回答   │
   │    │                                │
   │    ▼                                │
   │  ┌─────────┐                        │
   │  │  Tools  │                        │
   │  └────┬────┘                        │
   │       │                             │
   │       ▼                             │
   │  观察结果 ──────► 继续思考           │
   │                                     │
   └─────────────────────────────────────┘
      │
      ▼
   最终输出

下一步
======

在下一个教程中，我们将学习 RAG（检索增强生成）技术，让 Agent 能够利用外部知识库。

:doc:`tutorial_07_rag`
