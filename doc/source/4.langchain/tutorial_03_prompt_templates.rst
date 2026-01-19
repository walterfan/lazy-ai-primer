####################################
Tutorial 3: Prompt Templates
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是 Prompt Template？
========================

Prompt Template 是一种可重用的提示词模板，允许你：

- 定义带有变量占位符的模板
- 动态填充变量生成最终提示词
- 保持提示词的一致性和可维护性

基础用法
========

PromptTemplate
--------------

.. code-block:: python

   from langchain_core.prompts import PromptTemplate

   # 创建模板
   template = PromptTemplate.from_template(
       "请为{platform}平台写一篇关于{topic}的{content_type}，字数约{word_count}字"
   )

   # 填充变量
   prompt = template.format(
       platform="微信公众号",
       topic="Python编程",
       content_type="技术文章",
       word_count=1500
   )

   print(prompt)
   # 输出: 请为微信公众号平台写一篇关于Python编程的技术文章，字数约1500字

ChatPromptTemplate
------------------

.. code-block:: python

   from langchain_core.prompts import ChatPromptTemplate

   # 创建聊天模板
   chat_template = ChatPromptTemplate.from_messages([
       ("system", "你是一个专业的{role}，擅长{skill}"),
       ("human", "{user_input}")
   ])

   # 生成消息
   messages = chat_template.format_messages(
       role="自媒体内容创作者",
       skill="撰写爆款文章",
       user_input="帮我写一个关于AI的开头"
   )

高级模板技巧
============

使用部分变量
------------

.. code-block:: python

   from langchain_core.prompts import PromptTemplate

   # 预填充部分变量
   template = PromptTemplate.from_template(
       "为{platform}写一篇{style}风格的{topic}文章"
   )

   # 创建部分填充的模板
   wechat_template = template.partial(platform="微信公众号")
   zhihu_template = template.partial(platform="知乎")

   # 使用时只需填充剩余变量
   prompt1 = wechat_template.format(style="轻松幽默", topic="编程入门")
   prompt2 = zhihu_template.format(style="专业严谨", topic="编程入门")

组合模板
--------

.. code-block:: python

   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

   # 带有历史消息占位符的模板
   template = ChatPromptTemplate.from_messages([
       ("system", "你是自媒体写作助手"),
       MessagesPlaceholder(variable_name="history"),  # 历史消息
       ("human", "{input}")
   ])

使用 Few-Shot 示例
------------------

.. code-block:: python

   from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

   # 定义示例
   examples = [
       {"topic": "Python", "title": "5分钟学会Python，小白也能看懂"},
       {"topic": "AI", "title": "AI时代来临，普通人如何抓住机遇？"},
       {"topic": "职场", "title": "35岁程序员的出路在哪里？这3点很重要"},
   ]

   # 示例模板
   example_template = PromptTemplate.from_template(
       "主题: {topic}\n标题: {title}"
   )

   # Few-Shot 模板
   few_shot_template = FewShotPromptTemplate(
       examples=examples,
       example_prompt=example_template,
       prefix="你是爆款标题生成专家，以下是一些优秀标题示例：\n",
       suffix="\n现在请为主题「{input_topic}」生成一个类似风格的标题：",
       input_variables=["input_topic"]
   )

   prompt = few_shot_template.format(input_topic="区块链")
   print(prompt)

实战：多平台内容模板系统
========================

.. code-block:: python

   from langchain_core.prompts import ChatPromptTemplate
   from langchain_openai import ChatOpenAI
   from dataclasses import dataclass

   @dataclass
   class PlatformConfig:
       name: str
       style: str
       max_length: int
       features: str

   class ContentTemplateSystem:
       # 平台配置
       PLATFORMS = {
           "微信公众号": PlatformConfig(
               name="微信公众号",
               style="温暖、有深度、引发共鸣",
               max_length=3000,
               features="支持图文混排，适合深度内容"
           ),
           "知乎": PlatformConfig(
               name="知乎",
               style="专业、有理有据、引用数据",
               max_length=5000,
               features="适合专业分析和深度讨论"
           ),
           "小红书": PlatformConfig(
               name="小红书",
               style="活泼、口语化、多用emoji",
               max_length=1000,
               features="适合种草、分享、生活方式"
           ),
           "头条": PlatformConfig(
               name="今日头条",
               style="通俗易懂、吸引眼球、信息量大",
               max_length=2000,
               features="适合新闻、资讯、热点"
           ),
       }

       def __init__(self):
           self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
           
           self.base_template = ChatPromptTemplate.from_messages([
               ("system", """你是专业的自媒体内容创作者。
               
   目标平台: {platform_name}
   平台特点: {platform_features}
   写作风格: {platform_style}
   字数限制: {max_length}字以内

   请严格按照平台特点和风格要求创作内容。"""),
               ("human", "请为主题「{topic}」创作一篇{content_type}")
           ])

       def create_content(self, platform: str, topic: str, content_type: str = "文章") -> str:
           config = self.PLATFORMS.get(platform)
           if not config:
               raise ValueError(f"不支持的平台: {platform}")
           
           messages = self.base_template.format_messages(
               platform_name=config.name,
               platform_features=config.features,
               platform_style=config.style,
               max_length=config.max_length,
               topic=topic,
               content_type=content_type
           )
           
           response = self.llm.invoke(messages)
           return response.content

   # 使用示例
   system = ContentTemplateSystem()

   # 为不同平台生成内容
   wechat_content = system.create_content("微信公众号", "AI编程入门")
   zhihu_content = system.create_content("知乎", "AI编程入门")
   xiaohongshu_content = system.create_content("小红书", "AI编程入门")

   print("=== 微信公众号版本 ===")
   print(wechat_content[:500])
   print("\n=== 知乎版本 ===")
   print(zhihu_content[:500])

输出解析器
==========

将 LLM 输出解析为结构化数据：

.. code-block:: python

   from langchain_core.output_parsers import JsonOutputParser
   from langchain_core.prompts import PromptTemplate
   from pydantic import BaseModel, Field

   # 定义输出结构
   class ArticleOutline(BaseModel):
       title: str = Field(description="文章标题")
       sections: list[str] = Field(description="章节列表")
       key_points: list[str] = Field(description="核心要点")

   # 创建解析器
   parser = JsonOutputParser(pydantic_object=ArticleOutline)

   # 创建带解析指令的模板
   template = PromptTemplate(
       template="为主题「{topic}」生成文章大纲\n{format_instructions}",
       input_variables=["topic"],
       partial_variables={"format_instructions": parser.get_format_instructions()}
   )

   # 使用
   from langchain_openai import ChatOpenAI
   
   chain = template | ChatOpenAI(model="gpt-4o-mini") | parser
   outline = chain.invoke({"topic": "Python自动化办公"})
   
   print(f"标题: {outline['title']}")
   print(f"章节: {outline['sections']}")

下一步
======

在下一个教程中，我们将学习如何使用 Chains 将多个组件串联成复杂的工作流。

:doc:`tutorial_04_chains`
