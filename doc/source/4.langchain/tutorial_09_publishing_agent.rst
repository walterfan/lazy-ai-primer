####################################
Tutorial 9: 多平台发布 Agent
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

项目概述
========

本教程将构建一个多平台发布 Agent，它能够：

1. 将内容自动适配不同平台的格式
2. 管理多平台账号和发布配置
3. 自动化发布流程
4. 追踪发布状态和数据

系统架构
========

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                Multi-Platform Publishing Agent              │
   │                                                              │
   │  ┌────────────────────────────────────────────────────────┐ │
   │  │                   Content Adapter                       │ │
   │  │  原始内容 ──► 微信版本 / 知乎版本 / 小红书版本 / ...    │ │
   │  └────────────────────────────────────────────────────────┘ │
   │                            │                                 │
   │  ┌────────────────────────┴───────────────────────────────┐ │
   │  │                   Platform Connectors                   │ │
   │  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐     │ │
   │  │  │微信  │  │知乎  │  │小红书│  │头条  │  │微博  │     │ │
   │  │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘     │ │
   │  └─────┼─────────┼─────────┼─────────┼─────────┼─────────┘ │
   │        │         │         │         │         │            │
   │  ┌─────┴─────────┴─────────┴─────────┴─────────┴─────────┐ │
   │  │                   Status Tracker                       │ │
   │  │          发布状态追踪 / 数据统计 / 错误处理             │ │
   │  └────────────────────────────────────────────────────────┘ │
   │                                                              │
   └─────────────────────────────────────────────────────────────┘

内容适配器
==========

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate
   from langchain_core.output_parsers import StrOutputParser
   from dataclasses import dataclass
   from typing import Dict, List, Optional
   from enum import Enum
   import json

   class Platform(str, Enum):
       WECHAT = "微信公众号"
       ZHIHU = "知乎"
       XIAOHONGSHU = "小红书"
       TOUTIAO = "今日头条"
       WEIBO = "微博"

   @dataclass
   class AdaptedContent:
       platform: Platform
       title: str
       content: str
       summary: str
       tags: List[str]
       cover_suggestion: str
       publish_time_suggestion: str

   class ContentAdapter:
       """内容适配器：将原始内容适配到不同平台"""
       
       PLATFORM_PROMPTS = {
           Platform.WECHAT: """
   将以下内容改写为微信公众号风格：
   - 标题：有深度，可以稍长，引发思考
   - 开头：要有钩子，3秒抓住读者
   - 正文：分段清晰，每段不超过3行
   - 使用小标题分隔章节
   - 结尾：引导关注和互动
   - 字数：1500-3000字
   """,
           Platform.ZHIHU: """
   将以下内容改写为知乎风格：
   - 标题：专业、直接，可以是问题形式
   - 开头：直接给出核心观点
   - 正文：逻辑严密，数据支撑
   - 适当引用来源增加可信度
   - 结尾：总结观点，欢迎讨论
   - 字数：不限，但要言之有物
   """,
           Platform.XIAOHONGSHU: """
   将以下内容改写为小红书风格：
   - 标题：简短有力，带emoji 📚
   - 开头：亲切口语化，像朋友聊天
   - 正文：分点列出，多用emoji
   - 加入个人体验和感受
   - 结尾：互动提问
   - 字数：500-1000字
   """,
           Platform.TOUTIAO: """
   将以下内容改写为今日头条风格：
   - 标题：吸引眼球，信息量大
   - 开头：直入主题，抓住注意力
   - 正文：通俗易懂，案例丰富
   - 适合快速阅读
   - 结尾：引导评论
   - 字数：1000-2000字
   """,
           Platform.WEIBO: """
   将以下内容改写为微博风格：
   - 标题：简短有力
   - 正文：精炼，140字左右核心观点
   - 可以是长微博（1000字以内）
   - 话题标签：#话题#格式
   - 适合传播和讨论
   """
       }
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
           
       def adapt(self, original_content: str, original_title: str, 
                 target_platform: Platform) -> AdaptedContent:
           """将内容适配到目标平台"""
           
           prompt = ChatPromptTemplate.from_template("""
   {platform_instruction}

   原始标题：{original_title}

   原始内容：
   {original_content}

   请输出适配后的内容，格式如下：
   【标题】
   （适配后的标题）

   【正文】
   （适配后的正文）

   【摘要】
   （50字以内的摘要）

   【标签】
   （5个相关标签，逗号分隔）

   【封面建议】
   （封面图片创意描述）

   【发布时间建议】
   （最佳发布时间）
   """)
           
           chain = prompt | self.llm | StrOutputParser()
           
           result = chain.invoke({
               "platform_instruction": self.PLATFORM_PROMPTS[target_platform],
               "original_title": original_title,
               "original_content": original_content
           })
           
           # 解析结果
           return self._parse_result(result, target_platform)
       
       def _parse_result(self, result: str, platform: Platform) -> AdaptedContent:
           """解析LLM输出"""
           sections = {}
           current_section = None
           current_content = []
           
           for line in result.split('\n'):
               if line.startswith('【') and line.endswith('】'):
                   if current_section:
                       sections[current_section] = '\n'.join(current_content).strip()
                   current_section = line[1:-1]
                   current_content = []
               else:
                   current_content.append(line)
           
           if current_section:
               sections[current_section] = '\n'.join(current_content).strip()
           
           return AdaptedContent(
               platform=platform,
               title=sections.get('标题', ''),
               content=sections.get('正文', ''),
               summary=sections.get('摘要', ''),
               tags=[t.strip() for t in sections.get('标签', '').split(',')],
               cover_suggestion=sections.get('封面建议', ''),
               publish_time_suggestion=sections.get('发布时间建议', '')
           )
       
       def adapt_to_all(self, original_content: str, original_title: str,
                        platforms: List[Platform] = None) -> Dict[Platform, AdaptedContent]:
           """适配到多个平台"""
           if platforms is None:
               platforms = list(Platform)
           
           results = {}
           for platform in platforms:
               print(f"正在适配到 {platform.value}...")
               results[platform] = self.adapt(original_content, original_title, platform)
           
           return results

平台连接器
==========

.. code-block:: python

   from abc import ABC, abstractmethod
   from dataclasses import dataclass
   from typing import Optional
   from datetime import datetime
   import json

   @dataclass
   class PublishResult:
       success: bool
       platform: str
       post_id: Optional[str] = None
       post_url: Optional[str] = None
       error_message: Optional[str] = None
       publish_time: Optional[datetime] = None

   class PlatformConnector(ABC):
       """平台连接器基类"""
       
       @abstractmethod
       def authenticate(self, credentials: dict) -> bool:
           """认证"""
           pass
       
       @abstractmethod
       def publish(self, content: AdaptedContent) -> PublishResult:
           """发布内容"""
           pass
       
       @abstractmethod
       def get_post_stats(self, post_id: str) -> dict:
           """获取文章数据"""
           pass

   class WeChatConnector(PlatformConnector):
       """微信公众号连接器"""
       
       def __init__(self):
           self.access_token = None
           self.app_id = None
       
       def authenticate(self, credentials: dict) -> bool:
           """微信公众号认证
           
           实际应用需要：
           1. 使用 app_id 和 app_secret 获取 access_token
           2. access_token 有效期2小时，需要刷新
           """
           self.app_id = credentials.get('app_id')
           app_secret = credentials.get('app_secret')
           
           # 模拟认证
           # 实际代码：
           # url = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={self.app_id}&secret={app_secret}"
           # response = requests.get(url)
           # self.access_token = response.json()['access_token']
           
           self.access_token = "mock_access_token"
           return True
       
       def publish(self, content: AdaptedContent) -> PublishResult:
           """发布到微信公众号
           
           实际流程：
           1. 上传图片素材获取 media_id
           2. 创建图文消息
           3. 发布或群发
           """
           try:
               # 模拟发布
               # 实际代码需要调用微信API
               print(f"[微信公众号] 发布文章: {content.title}")
               
               return PublishResult(
                   success=True,
                   platform="微信公众号",
                   post_id="wechat_" + datetime.now().strftime("%Y%m%d%H%M%S"),
                   post_url="https://mp.weixin.qq.com/s/xxxxx",
                   publish_time=datetime.now()
               )
           except Exception as e:
               return PublishResult(
                   success=False,
                   platform="微信公众号",
                   error_message=str(e)
               )
       
       def get_post_stats(self, post_id: str) -> dict:
           """获取文章数据"""
           # 模拟数据
           return {
               "read_count": 1234,
               "like_count": 56,
               "share_count": 23,
               "comment_count": 12
           }

   class ZhihuConnector(PlatformConnector):
       """知乎连接器"""
       
       def __init__(self):
           self.cookies = None
       
       def authenticate(self, credentials: dict) -> bool:
           """知乎认证（通常使用cookies）"""
           self.cookies = credentials.get('cookies')
           return self.cookies is not None
       
       def publish(self, content: AdaptedContent) -> PublishResult:
           """发布到知乎"""
           try:
               print(f"[知乎] 发布文章: {content.title}")
               
               return PublishResult(
                   success=True,
                   platform="知乎",
                   post_id="zhihu_" + datetime.now().strftime("%Y%m%d%H%M%S"),
                   post_url="https://zhuanlan.zhihu.com/p/xxxxx",
                   publish_time=datetime.now()
               )
           except Exception as e:
               return PublishResult(
                   success=False,
                   platform="知乎",
                   error_message=str(e)
               )
       
       def get_post_stats(self, post_id: str) -> dict:
           return {
               "view_count": 5678,
               "upvote_count": 234,
               "comment_count": 45,
               "collect_count": 67
           }

   class XiaohongshuConnector(PlatformConnector):
       """小红书连接器"""
       
       def authenticate(self, credentials: dict) -> bool:
           return True
       
       def publish(self, content: AdaptedContent) -> PublishResult:
           try:
               print(f"[小红书] 发布笔记: {content.title}")
               
               return PublishResult(
                   success=True,
                   platform="小红书",
                   post_id="xhs_" + datetime.now().strftime("%Y%m%d%H%M%S"),
                   post_url="https://www.xiaohongshu.com/explore/xxxxx",
                   publish_time=datetime.now()
               )
           except Exception as e:
               return PublishResult(
                   success=False,
                   platform="小红书",
                   error_message=str(e)
               )
       
       def get_post_stats(self, post_id: str) -> dict:
           return {
               "like_count": 890,
               "collect_count": 123,
               "comment_count": 34,
               "share_count": 12
           }

发布管理器
==========

.. code-block:: python

   from typing import Dict, List
   from datetime import datetime
   import json

   class PublishManager:
       """发布管理器：协调多平台发布"""
       
       def __init__(self):
           self.connectors: Dict[Platform, PlatformConnector] = {
               Platform.WECHAT: WeChatConnector(),
               Platform.ZHIHU: ZhihuConnector(),
               Platform.XIAOHONGSHU: XiaohongshuConnector(),
           }
           self.publish_history: List[PublishResult] = []
       
       def setup_credentials(self, platform: Platform, credentials: dict) -> bool:
           """配置平台凭证"""
           if platform in self.connectors:
               return self.connectors[platform].authenticate(credentials)
           return False
       
       def publish_to_platform(self, content: AdaptedContent) -> PublishResult:
           """发布到单个平台"""
           connector = self.connectors.get(content.platform)
           if not connector:
               return PublishResult(
                   success=False,
                   platform=content.platform.value,
                   error_message=f"不支持的平台: {content.platform.value}"
               )
           
           result = connector.publish(content)
           self.publish_history.append(result)
           return result
       
       def publish_to_all(self, adapted_contents: Dict[Platform, AdaptedContent]) -> List[PublishResult]:
           """发布到所有已适配的平台"""
           results = []
           for platform, content in adapted_contents.items():
               print(f"\n正在发布到 {platform.value}...")
               result = self.publish_to_platform(content)
               results.append(result)
               
               if result.success:
                   print(f"✅ 发布成功: {result.post_url}")
               else:
                   print(f"❌ 发布失败: {result.error_message}")
           
           return results
       
       def get_publish_report(self) -> str:
           """生成发布报告"""
           report = []
           report.append("=" * 50)
           report.append("发布报告")
           report.append("=" * 50)
           
           success_count = sum(1 for r in self.publish_history if r.success)
           fail_count = len(self.publish_history) - success_count
           
           report.append(f"总计: {len(self.publish_history)} 篇")
           report.append(f"成功: {success_count} 篇")
           report.append(f"失败: {fail_count} 篇")
           report.append("")
           
           for result in self.publish_history:
               status = "✅" if result.success else "❌"
               report.append(f"{status} [{result.platform}]")
               if result.success:
                   report.append(f"   链接: {result.post_url}")
                   report.append(f"   时间: {result.publish_time}")
               else:
                   report.append(f"   错误: {result.error_message}")
           
           return "\n".join(report)

完整的发布 Agent
================

.. code-block:: python

   from langchain_core.tools import tool
   from langchain.agents import create_tool_calling_agent, AgentExecutor
   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

   # 工具定义
   @tool
   def adapt_content_for_platform(content: str, title: str, platform: str) -> str:
       """将内容适配到指定平台
       
       Args:
           content: 原始内容
           title: 原始标题
           platform: 目标平台（微信公众号/知乎/小红书）
       """
       adapter = ContentAdapter()
       try:
           p = Platform(platform)
           result = adapter.adapt(content, title, p)
           return json.dumps({
               "platform": platform,
               "title": result.title,
               "content_preview": result.content[:200] + "...",
               "summary": result.summary,
               "tags": result.tags,
               "cover_suggestion": result.cover_suggestion,
               "publish_time_suggestion": result.publish_time_suggestion
           }, ensure_ascii=False)
       except ValueError:
           return json.dumps({"error": f"不支持的平台: {platform}"}, ensure_ascii=False)

   @tool
   def publish_to_platform(platform: str, title: str, content: str) -> str:
       """发布内容到指定平台
       
       Args:
           platform: 目标平台
           title: 文章标题
           content: 文章内容
       """
       manager = PublishManager()
       
       try:
           p = Platform(platform)
           adapted = AdaptedContent(
               platform=p,
               title=title,
               content=content,
               summary="",
               tags=[],
               cover_suggestion="",
               publish_time_suggestion=""
           )
           result = manager.publish_to_platform(adapted)
           
           return json.dumps({
               "success": result.success,
               "platform": result.platform,
               "post_url": result.post_url,
               "error": result.error_message
           }, ensure_ascii=False)
       except Exception as e:
           return json.dumps({"error": str(e)}, ensure_ascii=False)

   @tool
   def get_best_publish_time(platform: str) -> str:
       """获取平台最佳发布时间
       
       Args:
           platform: 平台名称
       """
       times = {
           "微信公众号": {
               "best_times": ["早8:00-9:00", "中午12:00-13:00", "晚20:00-22:00"],
               "best_days": ["周二", "周三", "周四"],
               "avoid": "周末下午"
           },
           "知乎": {
               "best_times": ["上午10:00-11:00", "晚21:00-23:00"],
               "best_days": ["工作日"],
               "avoid": "凌晨时段"
           },
           "小红书": {
               "best_times": ["中午12:00-14:00", "晚19:00-22:00"],
               "best_days": ["周末"],
               "avoid": "工作日上午"
           }
       }
       
       return json.dumps(times.get(platform, {"error": "未知平台"}), ensure_ascii=False)

   class PublishingAgent:
       """多平台发布 Agent"""
       
       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
           self.tools = [
               adapt_content_for_platform,
               publish_to_platform,
               get_best_publish_time
           ]
           
           self.prompt = ChatPromptTemplate.from_messages([
               ("system", """你是专业的自媒体运营助手，负责多平台内容发布。

   你的职责：
   1. 将内容适配到不同平台的风格
   2. 选择最佳发布时间
   3. 执行发布操作
   4. 追踪发布状态

   工作原则：
   - 确保内容适配平台特点
   - 选择最佳发布时间
   - 发布前确认内容质量
   - 记录发布结果"""),
               MessagesPlaceholder(variable_name="chat_history", optional=True),
               ("human", "{input}"),
               MessagesPlaceholder(variable_name="agent_scratchpad")
           ])
           
           self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
           self.executor = AgentExecutor(
               agent=self.agent,
               tools=self.tools,
               verbose=True
           )
       
       def publish(self, content: str, title: str, platforms: List[str]) -> str:
           """发布内容到多个平台"""
           platforms_str = "、".join(platforms)
           
           return self.executor.invoke({
               "input": f"""
               请将以下内容发布到这些平台: {platforms_str}
               
               标题: {title}
               
               内容: {content}
               
               请执行以下步骤：
               1. 为每个平台适配内容
               2. 查询每个平台的最佳发布时间
               3. 执行发布
               4. 汇报发布结果
               """
           })["output"]

   # 使用示例
   if __name__ == "__main__":
       agent = PublishingAgent()
       
       result = agent.publish(
           content="""
           AI编程正在改变软件开发的方式。通过使用AI辅助工具，
           开发者可以更快地编写代码、发现bug、优化性能。
           本文将介绍几个实用的AI编程技巧...
           """,
           title="AI编程入门指南",
           platforms=["微信公众号", "知乎", "小红书"]
       )
       
       print(result)

下一步
======

在下一个教程中，我们将整合所有组件，构建完整的自媒体工作流系统。

:doc:`tutorial_10_complete_workflow`
