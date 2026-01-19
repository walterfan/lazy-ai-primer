####################################
Tutorial 4: Chains 链式调用
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是 Chain？
==============

Chain 是 LangChain 的核心概念，它将多个组件串联成一个处理管道。
通过 Chain，你可以：

- 组合多个 LLM 调用
- 添加数据预处理和后处理
- 构建复杂的多步骤工作流

LCEL: LangChain Expression Language
===================================

LangChain 推荐使用 LCEL（管道符 ``|``）来构建 Chain：

.. code-block:: python

   from langchain_core.prompts import ChatPromptTemplate
   from langchain_openai import ChatOpenAI
   from langchain_core.output_parsers import StrOutputParser

   # 使用 | 操作符构建 Chain
   prompt = ChatPromptTemplate.from_template("写一个关于{topic}的笑话")
   model = ChatOpenAI(model="gpt-4o-mini")
   parser = StrOutputParser()

   # 构建 Chain: prompt -> model -> parser
   chain = prompt | model | parser

   # 调用
   result = chain.invoke({"topic": "程序员"})
   print(result)

Chain 的基本模式
================

顺序执行
--------

.. code-block:: python

   from langchain_core.prompts import ChatPromptTemplate
   from langchain_openai import ChatOpenAI
   from langchain_core.output_parsers import StrOutputParser

   # Chain 1: 生成标题
   title_prompt = ChatPromptTemplate.from_template(
       "为主题「{topic}」生成一个吸引人的标题，只输出标题"
   )

   # Chain 2: 根据标题生成大纲
   outline_prompt = ChatPromptTemplate.from_template(
       "为标题「{title}」生成文章大纲，包含3-5个章节"
   )

   model = ChatOpenAI(model="gpt-4o-mini")
   parser = StrOutputParser()

   # 组合成完整流程
   title_chain = title_prompt | model | parser
   outline_chain = outline_prompt | model | parser

   # 顺序执行
   title = title_chain.invoke({"topic": "AI编程"})
   outline = outline_chain.invoke({"title": title})

   print(f"标题: {title}")
   print(f"大纲:\n{outline}")

使用 RunnablePassthrough
------------------------

.. code-block:: python

   from langchain_core.runnables import RunnablePassthrough, RunnableParallel

   # 保留原始输入并添加新字段
   chain = (
       RunnablePassthrough.assign(
           title=title_prompt | model | parser
       )
       | RunnablePassthrough.assign(
           outline=outline_prompt | model | parser
       )
   )

   result = chain.invoke({"topic": "Python自动化"})
   # result = {"topic": "Python自动化", "title": "...", "outline": "..."}

并行执行
--------

.. code-block:: python

   from langchain_core.runnables import RunnableParallel

   # 同时生成多个版本
   parallel_chain = RunnableParallel(
       wechat=ChatPromptTemplate.from_template(
           "用微信公众号风格写一段关于{topic}的介绍"
       ) | model | parser,
       zhihu=ChatPromptTemplate.from_template(
           "用知乎专业风格写一段关于{topic}的介绍"
       ) | model | parser,
       xiaohongshu=ChatPromptTemplate.from_template(
           "用小红书活泼风格写一段关于{topic}的介绍"
       ) | model | parser,
   )

   results = parallel_chain.invoke({"topic": "学习编程"})
   print("微信版:", results["wechat"][:100])
   print("知乎版:", results["zhihu"][:100])
   print("小红书版:", results["xiaohongshu"][:100])

条件分支
--------

.. code-block:: python

   from langchain_core.runnables import RunnableBranch

   # 根据内容类型选择不同的处理链
   branch_chain = RunnableBranch(
       (
           lambda x: x["content_type"] == "技术文章",
           ChatPromptTemplate.from_template(
               "用专业技术风格写关于{topic}的文章"
           ) | model | parser
       ),
       (
           lambda x: x["content_type"] == "生活分享",
           ChatPromptTemplate.from_template(
               "用轻松口语风格写关于{topic}的分享"
           ) | model | parser
       ),
       # 默认分支
       ChatPromptTemplate.from_template(
           "写一篇关于{topic}的通用文章"
       ) | model | parser
   )

   tech_article = branch_chain.invoke({"topic": "Python", "content_type": "技术文章"})
   life_share = branch_chain.invoke({"topic": "学习", "content_type": "生活分享"})

实战：自媒体内容生产流水线
==========================

.. code-block:: python

   from langchain_core.prompts import ChatPromptTemplate
   from langchain_openai import ChatOpenAI
   from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
   from langchain_core.runnables import RunnablePassthrough, RunnableParallel
   from pydantic import BaseModel, Field

   class ContentPlan(BaseModel):
       title: str = Field(description="文章标题")
       hook: str = Field(description="开头钩子，吸引读者")
       sections: list[str] = Field(description="章节标题列表")
       cta: str = Field(description="结尾行动号召")

   class ContentPipeline:
       def __init__(self):
           self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
           self.creative_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
           
       def build_pipeline(self):
           # Step 1: 生成内容规划
           plan_prompt = ChatPromptTemplate.from_template("""
   你是自媒体内容策划专家。为主题「{topic}」创建内容规划。
   目标平台: {platform}
   目标受众: {audience}

   请输出JSON格式的内容规划。
   {format_instructions}
   """)
           plan_parser = JsonOutputParser(pydantic_object=ContentPlan)
           
           plan_chain = (
               plan_prompt.partial(
                   format_instructions=plan_parser.get_format_instructions()
               )
               | self.model
               | plan_parser
           )

           # Step 2: 生成完整文章
           content_prompt = ChatPromptTemplate.from_template("""
   根据以下规划撰写完整文章：

   标题: {title}
   开头钩子: {hook}
   章节: {sections}
   结尾CTA: {cta}

   要求：
   - 每个章节300-500字
   - 语言生动，案例丰富
   - 适合{platform}平台风格
   """)

           content_chain = content_prompt | self.creative_model | StrOutputParser()

           # Step 3: 生成配套内容（并行）
           extras_chain = RunnableParallel(
               summary=ChatPromptTemplate.from_template(
                   "用50字总结这篇文章的核心观点：\n{content}"
               ) | self.model | StrOutputParser(),
               
               tags=ChatPromptTemplate.from_template(
                   "为这篇文章生成5个相关标签，用逗号分隔：\n{content}"
               ) | self.model | StrOutputParser(),
               
               cover_idea=ChatPromptTemplate.from_template(
                   "为这篇文章设计封面图创意，描述画面内容：\n标题：{title}"
               ) | self.model | StrOutputParser(),
           )

           # 组合完整流水线
           def run_pipeline(inputs: dict) -> dict:
               # 生成规划
               plan = plan_chain.invoke(inputs)
               
               # 生成文章
               content = content_chain.invoke({
                   **plan,
                   "platform": inputs["platform"]
               })
               
               # 生成配套内容
               extras = extras_chain.invoke({
                   "content": content,
                   "title": plan["title"]
               })
               
               return {
                   "plan": plan,
                   "content": content,
                   **extras
               }
           
           return run_pipeline

   # 使用示例
   pipeline = ContentPipeline()
   run = pipeline.build_pipeline()

   result = run({
       "topic": "AI时代的学习方法",
       "platform": "微信公众号",
       "audience": "职场人士"
   })

   print("=== 内容规划 ===")
   print(f"标题: {result['plan']['title']}")
   print(f"章节: {result['plan']['sections']}")

   print("\n=== 文章内容（前500字）===")
   print(result['content'][:500])

   print("\n=== 配套内容 ===")
   print(f"摘要: {result['summary']}")
   print(f"标签: {result['tags']}")
   print(f"封面创意: {result['cover_idea']}")

流式输出 Chain
==============

.. code-block:: python

   # 流式输出完整 Chain
   chain = prompt | model | parser

   for chunk in chain.stream({"topic": "编程学习"}):
       print(chunk, end="", flush=True)

下一步
======

在下一个教程中，我们将学习如何使用 Memory 来保持对话上下文。

:doc:`tutorial_05_memory`
