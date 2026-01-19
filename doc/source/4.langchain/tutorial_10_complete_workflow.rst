####################################
Tutorial 10: 完整自媒体工作流
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

项目概述
========

本教程将整合前面所有知识，构建一个完整的自媒体 AI Agent 系统，实现：

1. **内容策划** - 发现热点、分析潜力、制定计划
2. **内容创作** - 生成大纲、撰写文章、优化内容
3. **多平台适配** - 自动适配不同平台风格
4. **自动发布** - 一键发布到多个平台
5. **数据追踪** - 监控发布效果、生成报告

系统架构
========

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────┐
   │                    Self-Media AI Agent System                        │
   │                                                                      │
   │  ┌─────────────────────────────────────────────────────────────┐    │
   │  │                      Orchestrator                            │    │
   │  │          (工作流编排器 - 协调各个子Agent)                    │    │
   │  └──────────────────────────┬──────────────────────────────────┘    │
   │                             │                                        │
   │    ┌────────────────────────┼────────────────────────┐              │
   │    │                        │                        │              │
   │    ▼                        ▼                        ▼              │
   │  ┌──────────┐        ┌──────────┐        ┌──────────────────┐      │
   │  │ Planner  │        │ Creator  │        │    Publisher     │      │
   │  │  Agent   │───────►│  Agent   │───────►│      Agent       │      │
   │  │ 内容策划  │        │ 内容创作  │        │    多平台发布     │      │
   │  └──────────┘        └──────────┘        └──────────────────┘      │
   │       │                   │                       │                 │
   │       ▼                   ▼                       ▼                 │
   │  ┌──────────────────────────────────────────────────────────────┐  │
   │  │                    Shared Resources                           │  │
   │  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────────────────┐  │  │
   │  │  │知识库   │  │ 模板库  │  │ 配置   │  │    状态存储        │  │  │
   │  │  │  RAG   │  │Prompts │  │ Config │  │   State Store     │  │  │
   │  │  └────────┘  └────────┘  └────────┘  └────────────────────┘  │  │
   │  └──────────────────────────────────────────────────────────────┘  │
   │                                                                      │
   └─────────────────────────────────────────────────────────────────────┘

完整实现
========

项目结构
--------

.. code-block:: text

   selfmedia_agent/
   ├── __init__.py
   ├── config.py           # 配置管理
   ├── models.py           # 数据模型
   ├── tools/              # 工具集
   │   ├── __init__.py
   │   ├── research.py     # 话题研究工具
   │   ├── writing.py      # 写作工具
   │   └── publishing.py   # 发布工具
   ├── agents/             # Agent 定义
   │   ├── __init__.py
   │   ├── planner.py      # 策划 Agent
   │   ├── creator.py      # 创作 Agent
   │   └── publisher.py    # 发布 Agent
   ├── knowledge/          # 知识库
   │   └── writing_tips.txt
   └── main.py             # 主入口

配置管理 (config.py)
--------------------

.. code-block:: python

   from dataclasses import dataclass, field
   from typing import Dict, List, Optional
   from enum import Enum
   import os
   from dotenv import load_dotenv

   load_dotenv()

   class Platform(str, Enum):
       WECHAT = "微信公众号"
       ZHIHU = "知乎"
       XIAOHONGSHU = "小红书"
       TOUTIAO = "今日头条"

   @dataclass
   class PlatformConfig:
       name: str
       max_length: int
       style: str
       title_max_length: int
       best_publish_times: List[str]
       credentials: Dict = field(default_factory=dict)

   @dataclass
   class AgentConfig:
       model_name: str = "gpt-4o-mini"
       temperature: float = 0.7
       max_iterations: int = 15
       verbose: bool = True

   class Config:
       """全局配置"""
       
       # API Keys
       OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
       
       # Agent 配置
       AGENT = AgentConfig()
       
       # 平台配置
       PLATFORMS = {
           Platform.WECHAT: PlatformConfig(
               name="微信公众号",
               max_length=3000,
               style="深度、有温度、引发共鸣",
               title_max_length=64,
               best_publish_times=["08:00", "12:00", "20:00"]
           ),
           Platform.ZHIHU: PlatformConfig(
               name="知乎",
               max_length=5000,
               style="专业、有理有据",
               title_max_length=50,
               best_publish_times=["10:00", "21:00"]
           ),
           Platform.XIAOHONGSHU: PlatformConfig(
               name="小红书",
               max_length=1000,
               style="活泼、口语化、emoji",
               title_max_length=20,
               best_publish_times=["12:00", "19:00"]
           ),
       }

数据模型 (models.py)
--------------------

.. code-block:: python

   from dataclasses import dataclass, field
   from typing import List, Dict, Optional
   from datetime import datetime
   from enum import Enum

   class ContentStatus(str, Enum):
       DRAFT = "draft"
       READY = "ready"
       PUBLISHED = "published"
       FAILED = "failed"

   @dataclass
   class ContentIdea:
       topic: str
       angle: str
       target_audience: str
       platforms: List[str]
       priority: int = 1
       created_at: datetime = field(default_factory=datetime.now)

   @dataclass
   class ContentOutline:
       title: str
       hook: str
       sections: List[Dict[str, str]]
       conclusion: str
       cta: str

   @dataclass
   class Article:
       id: str
       title: str
       content: str
       summary: str
       tags: List[str]
       outline: Optional[ContentOutline] = None
       status: ContentStatus = ContentStatus.DRAFT
       created_at: datetime = field(default_factory=datetime.now)
       
   @dataclass
   class AdaptedArticle:
       original_id: str
       platform: str
       title: str
       content: str
       summary: str
       tags: List[str]
       cover_suggestion: str
       status: ContentStatus = ContentStatus.READY

   @dataclass
   class PublishRecord:
       article_id: str
       platform: str
       success: bool
       post_url: Optional[str] = None
       error_message: Optional[str] = None
       published_at: Optional[datetime] = None
       stats: Dict = field(default_factory=dict)

主系统实现 (main.py)
--------------------

.. code-block:: python

   from langchain_openai import ChatOpenAI, OpenAIEmbeddings
   from langchain_core.tools import tool
   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
   from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
   from langchain.agents import create_tool_calling_agent, AgentExecutor
   from langchain_community.vectorstores import FAISS
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from langchain.memory import ConversationBufferWindowMemory
   from typing import List, Dict, Optional
   from datetime import datetime
   import json
   import uuid

   # ========== 工具定义 ==========

   @tool
   def search_hot_topics(category: str, count: int = 5) -> str:
       """搜索热门话题
       
       Args:
           category: 分类（科技/职场/生活/教育）
           count: 数量
       """
       topics = {
           "科技": [
               {"topic": "AI Agent 开发", "heat": 95, "trend": "🔥上升"},
               {"topic": "Cursor IDE 技巧", "heat": 88, "trend": "📈上升"},
               {"topic": "RAG 最佳实践", "heat": 82, "trend": "➡️稳定"},
           ],
           "职场": [
               {"topic": "AI 时代职业转型", "heat": 92, "trend": "🔥上升"},
               {"topic": "远程办公效率", "heat": 78, "trend": "➡️稳定"},
           ]
       }
       return json.dumps(topics.get(category, [])[:count], ensure_ascii=False)

   @tool
   def analyze_topic(topic: str) -> str:
       """分析话题潜力
       
       Args:
           topic: 话题名称
       """
       return json.dumps({
           "topic": topic,
           "potential": "高",
           "competition": "中等",
           "recommended_angles": ["入门教程", "实战案例", "避坑指南"],
           "best_platforms": ["知乎", "微信公众号"]
       }, ensure_ascii=False)

   @tool
   def create_outline(topic: str, platform: str, angle: str) -> str:
       """创建文章大纲
       
       Args:
           topic: 话题
           platform: 目标平台
           angle: 切入角度
       """
       return json.dumps({
           "title": f"{topic}：{angle}",
           "hook": "你是否也遇到过这样的困扰...",
           "sections": [
               {"title": "背景介绍", "points": ["问题描述", "为什么重要"]},
               {"title": "核心方法", "points": ["方法1", "方法2", "方法3"]},
               {"title": "实战案例", "points": ["案例分析", "代码演示"]},
               {"title": "常见问题", "points": ["FAQ1", "FAQ2"]},
           ],
           "conclusion": "总结核心要点",
           "cta": "关注获取更多干货"
       }, ensure_ascii=False)

   @tool
   def write_section(section_title: str, points: str, style: str) -> str:
       """撰写文章章节
       
       Args:
           section_title: 章节标题
           points: 要点（逗号分隔）
           style: 写作风格
       """
       # 实际应用中这里会调用 LLM 生成内容
       return f"""
   ## {section_title}

   {points}

   这是一段根据 {style} 风格生成的内容示例...
   """

   @tool
   def adapt_for_platform(content: str, source_platform: str, target_platform: str) -> str:
       """将内容适配到目标平台
       
       Args:
           content: 原始内容
           source_platform: 来源平台
           target_platform: 目标平台
       """
       adaptations = {
           "小红书": "已将内容改为口语化风格，添加了emoji 📚✨",
           "知乎": "已将内容改为专业风格，添加了数据引用",
           "微信公众号": "已优化排版，添加了引导关注"
       }
       return json.dumps({
           "target_platform": target_platform,
           "adaptation_note": adaptations.get(target_platform, "已完成适配"),
           "content_preview": content[:100] + "..."
       }, ensure_ascii=False)

   @tool
   def publish_content(platform: str, title: str, content: str) -> str:
       """发布内容到平台
       
       Args:
           platform: 目标平台
           title: 标题
           content: 内容
       """
       # 模拟发布
       post_id = str(uuid.uuid4())[:8]
       return json.dumps({
           "success": True,
           "platform": platform,
           "post_id": post_id,
           "url": f"https://{platform.lower()}.com/p/{post_id}",
           "published_at": datetime.now().isoformat()
       }, ensure_ascii=False)

   @tool
   def get_content_stats(platform: str, post_id: str) -> str:
       """获取内容数据统计
       
       Args:
           platform: 平台
           post_id: 文章ID
       """
       import random
       return json.dumps({
           "platform": platform,
           "post_id": post_id,
           "views": random.randint(1000, 10000),
           "likes": random.randint(50, 500),
           "comments": random.randint(10, 100),
           "shares": random.randint(5, 50)
       }, ensure_ascii=False)

   # ========== 主 Agent 系统 ==========

   class SelfMediaAgentSystem:
       """自媒体 AI Agent 系统"""
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
           
           # 所有工具
           self.tools = [
               search_hot_topics,
               analyze_topic,
               create_outline,
               write_section,
               adapt_for_platform,
               publish_content,
               get_content_stats
           ]
           
           # 对话记忆
           self.memory = ConversationBufferWindowMemory(
               k=10,
               return_messages=True,
               memory_key="chat_history"
           )
           
           # 系统提示
           self.system_prompt = """你是专业的自媒体 AI 助手，具备以下能力：

   🎯 **内容策划**
   - 发现热门话题和内容机会
   - 分析话题潜力和最佳切入角度
   - 制定内容发布计划

   ✍️ **内容创作**
   - 创建结构化的文章大纲
   - 撰写高质量的文章内容
   - 优化标题、摘要和标签

   📱 **多平台运营**
   - 适配不同平台的内容风格
   - 自动发布到多个平台
   - 追踪内容表现数据

   工作原则：
   1. 内容要有价值，不做标题党
   2. 适配平台特点，尊重用户习惯
   3. 数据驱动，持续优化
   4. 保持专业性和原创性

   请根据用户需求，合理使用工具完成任务。每次任务完成后，给出清晰的总结和下一步建议。
   """
           
           self.prompt = ChatPromptTemplate.from_messages([
               ("system", self.system_prompt),
               MessagesPlaceholder(variable_name="chat_history", optional=True),
               ("human", "{input}"),
               MessagesPlaceholder(variable_name="agent_scratchpad")
           ])
           
           self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
           self.executor = AgentExecutor(
               agent=self.agent,
               tools=self.tools,
               verbose=True,
               max_iterations=20,
               handle_parsing_errors=True
           )
       
       def chat(self, message: str) -> str:
           """与 Agent 对话"""
           history = self.memory.load_memory_variables({})["chat_history"]
           
           result = self.executor.invoke({
               "input": message,
               "chat_history": history
           })
           
           # 保存对话历史
           self.memory.save_context(
               {"input": message},
               {"output": result["output"]}
           )
           
           return result["output"]
       
       def plan_content(self, category: str, platforms: List[str], days: int = 7) -> str:
           """制定内容计划"""
           platforms_str = "、".join(platforms)
           return self.chat(f"""
           请帮我制定一个{days}天的内容计划：
           
           - 内容领域：{category}
           - 目标平台：{platforms_str}
           
           要求：
           1. 搜索当前热门话题
           2. 分析每个话题的潜力
           3. 为每天安排一个内容主题
           4. 说明每个内容的切入角度和目标平台
           """)
       
       def create_content(self, topic: str, platform: str, style: str = None) -> str:
           """创作内容"""
           style_info = f"，风格要求：{style}" if style else ""
           return self.chat(f"""
           请帮我创作一篇关于「{topic}」的文章，发布到「{platform}」{style_info}。
           
           请执行以下步骤：
           1. 分析话题，确定最佳切入角度
           2. 创建详细的文章大纲
           3. 逐章节撰写内容
           4. 生成标题、摘要和标签
           """)
       
       def publish_to_platforms(self, content: str, title: str, platforms: List[str]) -> str:
           """发布到多个平台"""
           platforms_str = "、".join(platforms)
           return self.chat(f"""
           请将以下内容发布到这些平台：{platforms_str}
           
           标题：{title}
           
           内容：
           {content}
           
           请执行以下步骤：
           1. 为每个平台适配内容风格
           2. 发布到各平台
           3. 汇报发布结果
           """)
       
       def get_performance_report(self, post_ids: Dict[str, str]) -> str:
           """获取发布效果报告"""
           posts_info = "\n".join([f"- {p}: {pid}" for p, pid in post_ids.items()])
           return self.chat(f"""
           请获取以下文章的数据表现：
           
           {posts_info}
           
           请生成一份数据报告，包括：
           1. 各平台的阅读量、点赞、评论、分享数据
           2. 数据对比分析
           3. 优化建议
           """)
       
       def full_workflow(self, topic: str, platforms: List[str]) -> str:
           """执行完整工作流"""
           platforms_str = "、".join(platforms)
           return self.chat(f"""
           请执行完整的自媒体内容工作流：
           
           话题：{topic}
           目标平台：{platforms_str}
           
           完整流程：
           1. 分析话题潜力和最佳切入角度
           2. 创建文章大纲
           3. 撰写完整文章
           4. 为每个平台适配内容
           5. 发布到所有平台
           6. 生成发布报告
           
           请逐步执行，每一步都给出详细输出。
           """)

   # ========== 使用示例 ==========

   def main():
       print("=" * 60)
       print("自媒体 AI Agent 系统")
       print("=" * 60)
       
       agent = SelfMediaAgentSystem()
       
       # 交互式对话
       print("\n欢迎使用自媒体 AI Agent！")
       print("你可以：")
       print("1. 让我帮你发现热门话题")
       print("2. 让我帮你创作文章")
       print("3. 让我帮你发布到多个平台")
       print("4. 输入 'quit' 退出")
       print("-" * 60)
       
       while True:
           user_input = input("\n你: ").strip()
           
           if user_input.lower() in ['quit', 'exit', 'q']:
               print("再见！")
               break
           
           if not user_input:
               continue
           
           print("\nAgent 正在处理...")
           response = agent.chat(user_input)
           print(f"\nAgent: {response}")

   if __name__ == "__main__":
       main()

使用示例
========

.. code-block:: python

   # 创建 Agent 系统
   agent = SelfMediaAgentSystem()

   # 场景1：制定内容计划
   plan = agent.plan_content(
       category="科技",
       platforms=["微信公众号", "知乎"],
       days=7
   )
   print(plan)

   # 场景2：创作内容
   content = agent.create_content(
       topic="AI Agent 开发入门",
       platform="知乎",
       style="专业但易懂"
   )
   print(content)

   # 场景3：多平台发布
   result = agent.publish_to_platforms(
       content="...",
       title="AI Agent 开发入门指南",
       platforms=["微信公众号", "知乎", "小红书"]
   )
   print(result)

   # 场景4：完整工作流
   full_result = agent.full_workflow(
       topic="LangChain 实战教程",
       platforms=["微信公众号", "知乎"]
   )
   print(full_result)

部署建议
========

1. **环境配置**

   .. code-block:: bash

      # 创建 .env 文件
      OPENAI_API_KEY=your-api-key
      
      # 安装依赖
      pip install langchain langchain-openai python-dotenv

2. **生产环境优化**

   - 使用 Redis 存储对话历史
   - 添加请求限流和重试机制
   - 实现异步发布队列
   - 添加监控和告警

3. **扩展方向**

   - 接入真实的平台 API
   - 添加图片生成能力
   - 实现定时发布功能
   - 添加数据分析仪表板

总结
====

通过本系列教程，我们学习了：

1. **LangChain 基础** - LLM、Prompts、Chains
2. **高级特性** - Memory、Agents、Tools
3. **RAG 技术** - 知识库构建与检索
4. **实战应用** - 自媒体内容创作与发布

现在你已经掌握了使用 LangChain 构建 AI Agent 的核心技能，可以将这些知识应用到更多场景中！

🎉 恭喜完成全部教程！
