####################################
Tutorial 8: 自媒体内容创作 Agent
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

项目概述
========

本教程将构建一个完整的自媒体内容创作 Agent，它能够：

1. 根据热点话题生成内容创意
2. 撰写高质量的文章
3. 生成配套的标题、摘要、标签
4. 适配不同平台的风格要求

系统架构
========

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                  Self-Media Content Agent                    │
   │                                                              │
   │  ┌──────────────────────────────────────────────────────┐   │
   │  │                    Agent Core                         │   │
   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
   │  │  │ Planner │  │ Writer  │  │ Editor  │  │Optimizer│  │   │
   │  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  │   │
   │  └───────┼────────────┼────────────┼────────────┼───────┘   │
   │          │            │            │            │            │
   │  ┌───────┴────────────┴────────────┴────────────┴───────┐   │
   │  │                      Tools                            │   │
   │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────────┐ │   │
   │  │  │ 热点API │ │知识库  │ │ SEO工具│ │ 平台适配器     │ │   │
   │  │  └────────┘ └────────┘ └────────┘ └────────────────┘ │   │
   │  └──────────────────────────────────────────────────────┘   │
   │                                                              │
   └─────────────────────────────────────────────────────────────┘

完整实现
========

.. code-block:: python

   from langchain_openai import ChatOpenAI, OpenAIEmbeddings
   from langchain_core.tools import tool
   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
   from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
   from langchain.agents import create_tool_calling_agent, AgentExecutor
   from langchain_community.vectorstores import FAISS
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from pydantic import BaseModel, Field
   from typing import List, Optional
   from dataclasses import dataclass
   from enum import Enum
   import json

   # ========== 数据模型 ==========

   class Platform(str, Enum):
       WECHAT = "微信公众号"
       ZHIHU = "知乎"
       XIAOHONGSHU = "小红书"
       TOUTIAO = "今日头条"
       WEIBO = "微博"

   @dataclass
   class PlatformConfig:
       name: str
       max_length: int
       style: str
       features: List[str]
       title_max_length: int

   PLATFORM_CONFIGS = {
       Platform.WECHAT: PlatformConfig(
           name="微信公众号",
           max_length=3000,
           style="深度、有温度、引发共鸣",
           features=["图文混排", "长文阅读", "深度内容"],
           title_max_length=64
       ),
       Platform.ZHIHU: PlatformConfig(
           name="知乎",
           max_length=5000,
           style="专业、有理有据、数据支撑",
           features=["专业分析", "深度讨论", "引用来源"],
           title_max_length=50
       ),
       Platform.XIAOHONGSHU: PlatformConfig(
           name="小红书",
           max_length=1000,
           style="活泼、口语化、emoji丰富",
           features=["图片为主", "种草分享", "生活方式"],
           title_max_length=20
       ),
       Platform.TOUTIAO: PlatformConfig(
           name="今日头条",
           max_length=2000,
           style="通俗易懂、信息量大、吸引眼球",
           features=["资讯热点", "大众化", "快速阅读"],
           title_max_length=30
       ),
   }

   class ContentIdea(BaseModel):
       title: str = Field(description="内容标题")
       angle: str = Field(description="切入角度")
       target_audience: str = Field(description="目标受众")
       key_points: List[str] = Field(description="核心要点")
       estimated_engagement: str = Field(description="预估互动程度：高/中/低")

   class ArticleOutline(BaseModel):
       title: str = Field(description="文章标题")
       hook: str = Field(description="开头钩子")
       sections: List[dict] = Field(description="章节列表，每个包含title和points")
       conclusion: str = Field(description="结论要点")
       cta: str = Field(description="行动号召")

   class ArticleContent(BaseModel):
       title: str = Field(description="标题")
       content: str = Field(description="正文内容")
       summary: str = Field(description="摘要")
       tags: List[str] = Field(description="标签")
       platform: str = Field(description="目标平台")

   # ========== 工具定义 ==========

   @tool
   def search_trending_topics(category: str, count: int = 5) -> str:
       """搜索指定分类的热门话题
       
       Args:
           category: 话题分类（科技/职场/生活/教育/财经）
           count: 返回话题数量
       """
       # 实际应用中可接入真实的热搜API
       trending = {
           "科技": [
               {"topic": "AI Agent开发实战", "heat": 95, "trend": "上升"},
               {"topic": "Claude vs GPT对比评测", "heat": 88, "trend": "稳定"},
               {"topic": "Cursor IDE效率提升技巧", "heat": 82, "trend": "上升"},
               {"topic": "RAG技术最佳实践", "heat": 78, "trend": "上升"},
               {"topic": "LangChain入门教程", "heat": 75, "trend": "稳定"},
           ],
           "职场": [
               {"topic": "AI时代的职业转型", "heat": 92, "trend": "上升"},
               {"topic": "远程办公效率指南", "heat": 85, "trend": "稳定"},
               {"topic": "程序员副业变现", "heat": 80, "trend": "上升"},
               {"topic": "技术面试通关秘籍", "heat": 76, "trend": "稳定"},
               {"topic": "35岁程序员的出路", "heat": 90, "trend": "上升"},
           ],
           "生活": [
               {"topic": "高效学习方法论", "heat": 88, "trend": "稳定"},
               {"topic": "数字极简主义", "heat": 75, "trend": "上升"},
               {"topic": "个人知识管理", "heat": 82, "trend": "上升"},
               {"topic": "时间管理实战", "heat": 79, "trend": "稳定"},
               {"topic": "阅读习惯养成", "heat": 72, "trend": "稳定"},
           ]
       }
       
       topics = trending.get(category, [])[:count]
       return json.dumps({"category": category, "topics": topics}, ensure_ascii=False)

   @tool
   def get_platform_guidelines(platform: str) -> str:
       """获取指定平台的内容创作指南
       
       Args:
           platform: 平台名称（微信公众号/知乎/小红书/今日头条）
       """
       try:
           p = Platform(platform)
           config = PLATFORM_CONFIGS[p]
           return json.dumps({
               "platform": config.name,
               "max_length": config.max_length,
               "style": config.style,
               "features": config.features,
               "title_max_length": config.title_max_length,
               "tips": [
                   f"字数建议: {config.max_length//2}-{config.max_length}字",
                   f"标题不超过{config.title_max_length}字",
                   f"风格: {config.style}"
               ]
           }, ensure_ascii=False)
       except ValueError:
           return json.dumps({"error": f"不支持的平台: {platform}"}, ensure_ascii=False)

   @tool
   def analyze_content_potential(topic: str, platform: str) -> str:
       """分析话题在特定平台的内容潜力
       
       Args:
           topic: 话题名称
           platform: 目标平台
       """
       # 模拟分析结果
       analysis = {
           "topic": topic,
           "platform": platform,
           "potential_score": 85,
           "competition_level": "中等",
           "recommended_angles": [
               "入门教程型 - 适合新手",
               "实战案例型 - 展示具体应用",
               "避坑指南型 - 分享踩坑经验",
               "对比评测型 - 横向比较"
           ],
           "best_publish_time": "工作日早8点或晚8点",
           "estimated_read_time": "5-8分钟"
       }
       return json.dumps(analysis, ensure_ascii=False)

   @tool
   def generate_title_variants(topic: str, style: str, count: int = 5) -> str:
       """生成多个标题变体
       
       Args:
           topic: 文章主题
           style: 标题风格（吸引眼球/专业严谨/悬念式/数字式）
           count: 生成数量
       """
       # 实际应用中可以用LLM生成
       templates = {
           "吸引眼球": [
               f"震惊！{topic}竟然可以这样用",
               f"99%的人不知道的{topic}技巧",
               f"{topic}，看完这篇就够了",
           ],
           "专业严谨": [
               f"{topic}技术深度解析",
               f"从原理到实践：{topic}完全指南",
               f"{topic}：架构设计与最佳实践",
           ],
           "悬念式": [
               f"学会{topic}后，我的工作效率提升了10倍",
               f"为什么大厂都在用{topic}？",
               f"{topic}的秘密，今天终于公开了",
           ],
           "数字式": [
               f"5分钟掌握{topic}核心要点",
               f"{topic}入门：7个必知技巧",
               f"3步搞定{topic}，小白也能学会",
           ]
       }
       
       titles = templates.get(style, templates["数字式"])[:count]
       return json.dumps({"topic": topic, "style": style, "titles": titles}, ensure_ascii=False)

   @tool  
   def check_content_quality(content: str) -> str:
       """检查内容质量并给出优化建议
       
       Args:
           content: 待检查的内容
       """
       word_count = len(content)
       paragraphs = content.split('\n\n')
       
       issues = []
       suggestions = []
       
       # 检查字数
       if word_count < 500:
           issues.append("内容过短")
           suggestions.append("建议扩充到800字以上")
       elif word_count > 5000:
           issues.append("内容过长")
           suggestions.append("建议精简或分成系列文章")
       
       # 检查段落
       if len(paragraphs) < 3:
           issues.append("段落过少")
           suggestions.append("建议增加分段，提升可读性")
       
       # 检查是否有小标题
       if '##' not in content and '**' not in content:
           suggestions.append("建议添加小标题或加粗重点")
       
       quality_score = 100 - len(issues) * 15
       
       return json.dumps({
           "word_count": word_count,
           "paragraph_count": len(paragraphs),
           "quality_score": max(quality_score, 60),
           "issues": issues,
           "suggestions": suggestions
       }, ensure_ascii=False)

   # ========== 内容创作 Agent ==========

   class ContentCreationAgent:
       """自媒体内容创作 Agent"""
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
           self.tools = [
               search_trending_topics,
               get_platform_guidelines,
               analyze_content_potential,
               generate_title_variants,
               check_content_quality
           ]
           
           self.agent_prompt = ChatPromptTemplate.from_messages([
               ("system", """你是专业的自媒体内容创作专家。你的职责是：

   1. 发现热门话题和内容机会
   2. 分析话题潜力和最佳切入角度
   3. 根据平台特点创作高质量内容
   4. 优化标题、摘要和标签

   工作原则：
   - 内容要有价值，不做标题党
   - 适配不同平台的风格和用户习惯
   - 注重可读性和互动性
   - 保持专业性和原创性

   请充分利用工具完成任务，给出专业、可执行的内容方案。"""),
               MessagesPlaceholder(variable_name="chat_history", optional=True),
               ("human", "{input}"),
               MessagesPlaceholder(variable_name="agent_scratchpad")
           ])
           
           self.agent = create_tool_calling_agent(
               self.llm, 
               self.tools, 
               self.agent_prompt
           )
           
           self.executor = AgentExecutor(
               agent=self.agent,
               tools=self.tools,
               verbose=True,
               max_iterations=15
           )
       
       def find_topics(self, category: str, platform: str) -> str:
           """发现适合的话题"""
           return self.executor.invoke({
               "input": f"""
               请帮我在「{category}」分类下找到适合在「{platform}」发布的话题。
               
               要求：
               1. 搜索当前热门话题
               2. 获取平台的内容指南
               3. 分析每个话题的潜力
               4. 推荐最适合的2-3个话题，并说明理由
               """
           })["output"]
       
       def create_outline(self, topic: str, platform: str) -> str:
           """创建文章大纲"""
           return self.executor.invoke({
               "input": f"""
               请为话题「{topic}」创建一份适合「{platform}」的文章大纲。
               
               要求：
               1. 获取平台指南，了解风格要求
               2. 设计吸引人的开头钩子
               3. 规划3-5个章节
               4. 设计有力的结尾和行动号召
               5. 生成多个标题备选
               """
           })["output"]
       
       def write_article(self, topic: str, platform: str, outline: str = None) -> str:
           """撰写完整文章"""
           outline_info = f"\n参考大纲：{outline}" if outline else ""
           
           return self.executor.invoke({
               "input": f"""
               请为「{platform}」撰写一篇关于「{topic}」的完整文章。
               {outline_info}
               
               要求：
               1. 获取平台指南，严格遵循风格要求
               2. 撰写完整的文章内容
               3. 检查内容质量并优化
               4. 生成标题、摘要和标签
               5. 确保内容有价值、可读性强
               """
           })["output"]

   # ========== 使用示例 ==========

   if __name__ == "__main__":
       agent = ContentCreationAgent()
       
       # 场景1：发现话题
       print("=" * 50)
       print("场景1：发现话题")
       print("=" * 50)
       topics = agent.find_topics("科技", "微信公众号")
       print(topics)
       
       # 场景2：创建大纲
       print("\n" + "=" * 50)
       print("场景2：创建大纲")
       print("=" * 50)
       outline = agent.create_outline("AI Agent开发实战", "知乎")
       print(outline)
       
       # 场景3：撰写文章
       print("\n" + "=" * 50)
       print("场景3：撰写文章")
       print("=" * 50)
       article = agent.write_article("LangChain入门", "小红书")
       print(article)

运行效果
========

运行上述代码，Agent 会：

1. 调用 ``search_trending_topics`` 获取热门话题
2. 调用 ``get_platform_guidelines`` 了解平台要求
3. 调用 ``analyze_content_potential`` 分析话题潜力
4. 调用 ``generate_title_variants`` 生成标题
5. 调用 ``check_content_quality`` 检查并优化内容

最终输出一份完整的内容方案，包括：

- 推荐话题及理由
- 文章大纲
- 完整内容
- 标题、摘要、标签

下一步
======

在下一个教程中，我们将实现多平台发布 Agent，让内容能够自动发布到各个平台。

:doc:`tutorial_09_publishing_agent`
