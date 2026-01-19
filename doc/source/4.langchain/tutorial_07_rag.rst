####################################
Tutorial 7: RAG 检索增强生成
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是 RAG？
============

RAG（Retrieval Augmented Generation）是一种结合检索和生成的技术：

1. **检索（Retrieval）**: 从知识库中找到相关文档
2. **增强（Augmented）**: 将检索结果作为上下文
3. **生成（Generation）**: LLM 基于上下文生成回答

RAG 的优势：

- 减少幻觉，提高准确性
- 可以使用最新数据
- 知识可追溯、可更新
- 降低 token 成本

RAG 架构
========

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                      RAG Pipeline                        │
   │                                                          │
   │  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
   │  │ 文档加载  │───►│ 文本分割  │───►│ 向量化 & 存储    │   │
   │  │ Loaders  │    │ Splitters│    │ Embeddings+Store │   │
   │  └──────────┘    └──────────┘    └────────┬─────────┘   │
   │                                           │              │
   │                                           ▼              │
   │  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
   │  │ 用户问题  │───►│ 向量检索  │───►│  上下文 + LLM    │   │
   │  │  Query   │    │ Retriever│    │   生成回答       │   │
   │  └──────────┘    └──────────┘    └──────────────────┘   │
   │                                                          │
   └─────────────────────────────────────────────────────────┘

Step 1: 文档加载
================

.. code-block:: python

   from langchain_community.document_loaders import (
       TextLoader,
       PyPDFLoader,
       WebBaseLoader,
       DirectoryLoader
   )

   # 加载文本文件
   text_loader = TextLoader("article.txt", encoding="utf-8")
   docs = text_loader.load()

   # 加载 PDF
   pdf_loader = PyPDFLoader("document.pdf")
   docs = pdf_loader.load()

   # 加载网页
   web_loader = WebBaseLoader("https://example.com/article")
   docs = web_loader.load()

   # 加载目录下所有文件
   dir_loader = DirectoryLoader(
       "knowledge_base/",
       glob="**/*.txt",
       loader_cls=TextLoader
   )
   docs = dir_loader.load()

Step 2: 文本分割
================

.. code-block:: python

   from langchain_text_splitters import (
       RecursiveCharacterTextSplitter,
       CharacterTextSplitter
   )

   # 推荐：递归字符分割器
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,      # 每块最大字符数
       chunk_overlap=200,    # 块之间重叠字符数
       separators=["\n\n", "\n", "。", "！", "？", "，", " "]
   )

   chunks = splitter.split_documents(docs)

   print(f"原始文档数: {len(docs)}")
   print(f"分割后块数: {len(chunks)}")

Step 3: 向量化与存储
====================

.. code-block:: python

   from langchain_openai import OpenAIEmbeddings
   from langchain_community.vectorstores import FAISS, Chroma

   # 创建 Embedding 模型
   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

   # 使用 FAISS（内存向量库）
   vectorstore = FAISS.from_documents(chunks, embeddings)

   # 或使用 Chroma（持久化向量库）
   vectorstore = Chroma.from_documents(
       chunks,
       embeddings,
       persist_directory="./chroma_db"
   )

   # 保存和加载
   vectorstore.save_local("faiss_index")
   loaded_vectorstore = FAISS.load_local("faiss_index", embeddings)

Step 4: 检索与生成
==================

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate
   from langchain_core.runnables import RunnablePassthrough
   from langchain_core.output_parsers import StrOutputParser

   # 创建检索器
   retriever = vectorstore.as_retriever(
       search_type="similarity",  # 或 "mmr" (最大边际相关性)
       search_kwargs={"k": 4}     # 返回最相关的4个文档
   )

   # RAG Prompt
   rag_prompt = ChatPromptTemplate.from_template("""
   基于以下上下文回答问题。如果上下文中没有相关信息，请说明。

   上下文:
   {context}

   问题: {question}

   回答:
   """)

   # 创建 RAG Chain
   llm = ChatOpenAI(model="gpt-4o-mini")

   def format_docs(docs):
       return "\n\n".join(doc.page_content for doc in docs)

   rag_chain = (
       {"context": retriever | format_docs, "question": RunnablePassthrough()}
       | rag_prompt
       | llm
       | StrOutputParser()
   )

   # 使用
   answer = rag_chain.invoke("什么是LangChain？")
   print(answer)

实战：自媒体知识库助手
======================

.. code-block:: python

   from langchain_openai import ChatOpenAI, OpenAIEmbeddings
   from langchain_community.vectorstores import FAISS
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from langchain_core.prompts import ChatPromptTemplate
   from langchain_core.runnables import RunnablePassthrough
   from langchain_core.output_parsers import StrOutputParser
   from langchain_community.document_loaders import TextLoader
   from typing import List
   import os

   class SelfMediaKnowledgeBase:
       """自媒体写作知识库"""
       
       def __init__(self, knowledge_dir: str = "knowledge"):
           self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
           self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
           self.vectorstore = None
           self.knowledge_dir = knowledge_dir
           
       def add_knowledge(self, texts: List[str], metadatas: List[dict] = None):
           """添加知识到向量库"""
           splitter = RecursiveCharacterTextSplitter(
               chunk_size=500,
               chunk_overlap=100
           )
           
           chunks = splitter.create_documents(texts, metadatas)
           
           if self.vectorstore is None:
               self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
           else:
               self.vectorstore.add_documents(chunks)
       
       def load_from_files(self, directory: str):
           """从文件目录加载知识"""
           all_docs = []
           
           for filename in os.listdir(directory):
               if filename.endswith('.txt'):
                   filepath = os.path.join(directory, filename)
                   loader = TextLoader(filepath, encoding='utf-8')
                   docs = loader.load()
                   
                   # 添加元数据
                   for doc in docs:
                       doc.metadata["source"] = filename
                   all_docs.extend(docs)
           
           splitter = RecursiveCharacterTextSplitter(
               chunk_size=500,
               chunk_overlap=100
           )
           chunks = splitter.split_documents(all_docs)
           
           self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
           print(f"已加载 {len(chunks)} 个知识块")
       
       def query(self, question: str) -> str:
           """查询知识库"""
           if self.vectorstore is None:
               return "知识库为空，请先添加知识"
           
           retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
           
           prompt = ChatPromptTemplate.from_template("""
   你是自媒体写作专家。基于以下知识库内容回答问题。

   知识库内容:
   {context}

   用户问题: {question}

   请给出专业、实用的建议：
   """)
           
           def format_docs(docs):
               return "\n\n---\n\n".join(
                   f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
                   for doc in docs
               )
           
           chain = (
               {"context": retriever | format_docs, "question": RunnablePassthrough()}
               | prompt
               | self.llm
               | StrOutputParser()
           )
           
           return chain.invoke(question)
       
       def save(self, path: str = "selfmedia_kb"):
           """保存向量库"""
           if self.vectorstore:
               self.vectorstore.save_local(path)
               print(f"知识库已保存到 {path}")
       
       def load(self, path: str = "selfmedia_kb"):
           """加载向量库"""
           self.vectorstore = FAISS.load_local(
               path, 
               self.embeddings,
               allow_dangerous_deserialization=True
           )
           print(f"知识库已从 {path} 加载")

   # ========== 使用示例 ==========

   # 创建知识库
   kb = SelfMediaKnowledgeBase()

   # 添加自媒体写作知识
   knowledge_texts = [
       """
       微信公众号写作技巧：
       1. 标题要有数字或疑问句，提高点击率
       2. 开头3秒定生死，要有钩子
       3. 使用小标题分段，每段不超过3行
       4. 结尾要有互动问题和关注引导
       5. 最佳发布时间：早8点、中午12点、晚8点
       """,
       """
       知乎写作技巧：
       1. 回答开头直接给出核心观点
       2. 使用数据和案例支撑论点
       3. 适当引用权威来源增加可信度
       4. 回答要有独特视角，避免千篇一律
       5. 长回答分段清晰，使用加粗突出重点
       """,
       """
       小红书写作技巧：
       1. 标题要有emoji，视觉吸引力强
       2. 封面图比内容更重要
       3. 口语化表达，像朋友聊天
       4. 加入实用的干货清单
       5. 适当使用热门话题标签
       """,
       """
       爆款标题公式：
       1. 数字法：5个技巧、3分钟学会
       2. 疑问法：为什么...？如何...？
       3. 对比法：从小白到大神
       4. 痛点法：还在为...烦恼？
       5. 悬念法：99%的人不知道的...
       """
   ]

   kb.add_knowledge(knowledge_texts)

   # 查询
   print(kb.query("如何写一个吸引人的标题？"))
   print("\n" + "="*50 + "\n")
   print(kb.query("微信公众号发布文章的最佳时间是什么？"))

   # 保存知识库
   kb.save("selfmedia_kb")

高级 RAG 技术
=============

多查询检索
----------

.. code-block:: python

   from langchain.retrievers.multi_query import MultiQueryRetriever

   # 使用 LLM 生成多个查询变体
   multi_retriever = MultiQueryRetriever.from_llm(
       retriever=vectorstore.as_retriever(),
       llm=ChatOpenAI(model="gpt-4o-mini")
   )

   # 会自动生成多个相关查询，合并检索结果
   docs = multi_retriever.invoke("如何写爆款标题？")

上下文压缩
----------

.. code-block:: python

   from langchain.retrievers import ContextualCompressionRetriever
   from langchain.retrievers.document_compressors import LLMChainExtractor

   # 使用 LLM 压缩/提取相关内容
   compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4o-mini"))

   compression_retriever = ContextualCompressionRetriever(
       base_compressor=compressor,
       base_retriever=vectorstore.as_retriever()
   )

   # 返回的文档只包含与查询相关的部分
   docs = compression_retriever.invoke("标题写作技巧")

混合检索
--------

.. code-block:: python

   from langchain.retrievers import EnsembleRetriever
   from langchain_community.retrievers import BM25Retriever

   # 关键词检索器
   bm25_retriever = BM25Retriever.from_documents(chunks)
   bm25_retriever.k = 3

   # 向量检索器
   vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

   # 混合检索（结合关键词和语义）
   ensemble_retriever = EnsembleRetriever(
       retrievers=[bm25_retriever, vector_retriever],
       weights=[0.4, 0.6]  # 权重分配
   )

下一步
======

在下一个教程中，我们将综合运用所学知识，构建一个完整的自媒体内容创作 Agent。

:doc:`tutorial_08_content_agent`
