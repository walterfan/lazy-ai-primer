####################################
Tutorial 1: RAG 入门
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

什么是 RAG？
============

**RAG（Retrieval-Augmented Generation）** 是一种将信息检索与文本生成相结合的技术，
让大语言模型能够利用外部知识来生成更准确的回答。

通俗理解
--------

想象你在回答一个专业问题：

- **不用 RAG**: 凭记忆回答，可能不准确或过时
- **使用 RAG**: 先查阅相关资料，然后基于资料回答

.. code-block:: text

   传统 LLM 对话:
   
   用户: "公司最新的请假政策是什么？"
   LLM:  "抱歉，我没有您公司的内部政策信息..."

   RAG 增强对话:
   
   用户: "公司最新的请假政策是什么？"
   [检索公司内部文档...]
   [找到: 员工手册2024版-请假制度.pdf]
   LLM:  "根据公司最新政策，年假天数为：工作满1年10天，满5年15天..."

RAG 的核心组件
==============

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                     RAG 系统架构                             │
   │                                                              │
   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
   │  │  文档处理    │    │  向量数据库  │    │  大语言模型  │     │
   │  │             │    │             │    │             │     │
   │  │ · 加载文档  │    │ · 存储向量  │    │ · 理解问题  │     │
   │  │ · 文本分块  │    │ · 相似搜索  │    │ · 生成回答  │     │
   │  │ · 向量化    │    │ · 索引管理  │    │ · 引用来源  │     │
   │  └─────────────┘    └─────────────┘    └─────────────┘     │
   │                                                              │
   └─────────────────────────────────────────────────────────────┘

1. **文档处理**: 将原始文档转换为可检索的格式
2. **向量数据库**: 存储文档向量，支持高效相似性搜索
3. **大语言模型**: 基于检索到的内容生成回答

RAG 的工作流程
==============

索引阶段（离线）
----------------

.. code-block:: python

   # 伪代码展示 RAG 索引流程

   # 1. 加载文档
   documents = load_documents("./knowledge_base/")

   # 2. 文本分块
   chunks = split_into_chunks(documents, chunk_size=500)

   # 3. 生成向量
   embeddings = embedding_model.encode(chunks)

   # 4. 存入向量数据库
   vector_db.add(chunks, embeddings)

查询阶段（在线）
----------------

.. code-block:: python

   # 伪代码展示 RAG 查询流程

   # 1. 用户提问
   question = "什么是机器学习？"

   # 2. 问题向量化
   query_embedding = embedding_model.encode(question)

   # 3. 检索相关文档
   relevant_docs = vector_db.search(query_embedding, top_k=3)

   # 4. 构建 Prompt
   prompt = f"""
   基于以下参考资料回答问题：

   参考资料：
   {relevant_docs}

   问题：{question}

   请基于参考资料回答，如果资料中没有相关信息，请说明。
   """

   # 5. 生成回答
   answer = llm.generate(prompt)

动手实践：第一个 RAG 系统
=========================

让我们用 Python 构建一个简单的 RAG 系统：

.. code-block:: python

   # 安装依赖
   # pip install openai chromadb sentence-transformers

   from sentence_transformers import SentenceTransformer
   import chromadb

   # 1. 准备知识库
   knowledge_base = [
       "Python是一种高级编程语言，以简洁易读著称。",
       "机器学习是人工智能的一个分支，让计算机从数据中学习。",
       "深度学习使用多层神经网络来学习数据的复杂模式。",
       "自然语言处理(NLP)是让计算机理解人类语言的技术。",
       "RAG是检索增强生成的缩写，结合了检索和生成技术。",
   ]

   # 2. 初始化嵌入模型
   embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

   # 3. 初始化向量数据库
   client = chromadb.Client()
   collection = client.create_collection("knowledge_base")

   # 4. 索引文档
   embeddings = embedding_model.encode(knowledge_base).tolist()
   collection.add(
       documents=knowledge_base,
       embeddings=embeddings,
       ids=[f"doc_{i}" for i in range(len(knowledge_base))]
   )

   print("知识库索引完成！")

   # 5. 检索函数
   def retrieve(query, top_k=2):
       query_embedding = embedding_model.encode([query]).tolist()
       results = collection.query(
           query_embeddings=query_embedding,
           n_results=top_k
       )
       return results['documents'][0]

   # 6. 测试检索
   query = "什么是深度学习？"
   relevant_docs = retrieve(query)

   print(f"\n问题: {query}")
   print(f"检索到的相关文档:")
   for i, doc in enumerate(relevant_docs, 1):
       print(f"  {i}. {doc}")

   # 7. 构建 Prompt（实际使用时调用 LLM）
   def build_prompt(question, context_docs):
       context = "\n".join([f"- {doc}" for doc in context_docs])
       prompt = f"""基于以下参考资料回答问题。

   参考资料：
   {context}

   问题：{question}

   回答："""
       return prompt

   prompt = build_prompt(query, relevant_docs)
   print(f"\n生成的 Prompt:\n{prompt}")

使用 LangChain 简化 RAG
=======================

LangChain 提供了更简洁的 RAG 实现：

.. code-block:: python

   from langchain_community.document_loaders import TextLoader
   from langchain.text_splitter import CharacterTextSplitter
   from langchain_community.embeddings import HuggingFaceEmbeddings
   from langchain_community.vectorstores import Chroma
   from langchain_openai import ChatOpenAI
   from langchain.chains import RetrievalQA

   # 1. 加载文档
   loader = TextLoader("./knowledge.txt")
   documents = loader.load()

   # 2. 分割文档
   text_splitter = CharacterTextSplitter(
       chunk_size=500,
       chunk_overlap=50
   )
   docs = text_splitter.split_documents(documents)

   # 3. 创建向量存储
   embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
   )
   vectorstore = Chroma.from_documents(docs, embeddings)

   # 4. 创建检索 QA 链
   llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
   qa_chain = RetrievalQA.from_chain_type(
       llm=llm,
       chain_type="stuff",
       retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
   )

   # 5. 提问
   question = "什么是机器学习？"
   answer = qa_chain.invoke(question)
   print(f"问题: {question}")
   print(f"回答: {answer['result']}")

RAG vs 微调
===========

.. csv-table::
   :header: "特性", "RAG", "微调（Fine-tuning）"
   :widths: 20, 40, 40

   "知识更新", "实时更新，只需更新知识库", "需要重新训练模型"
   "成本", "较低，无需训练", "较高，需要GPU和数据"
   "可解释性", "高，可追溯来源", "低，黑箱生成"
   "适用场景", "知识密集型问答", "风格/行为调整"
   "数据需求", "只需文档", "需要训练数据"
   "实现难度", "较简单", "较复杂"

RAG 的应用场景
==============

1. **企业知识库问答**

.. code-block:: text

   场景: 员工查询公司政策
   知识库: 员工手册、规章制度、FAQ
   效果: 准确回答公司内部问题

2. **客服机器人**

.. code-block:: text

   场景: 回答产品相关问题
   知识库: 产品文档、使用说明、常见问题
   效果: 减少人工客服压力

3. **法律/医疗咨询**

.. code-block:: text

   场景: 专业领域问答
   知识库: 法律条文、医学文献
   效果: 提供专业参考，降低幻觉风险

4. **代码助手**

.. code-block:: text

   场景: 回答代码相关问题
   知识库: API文档、代码示例
   效果: 提供准确的技术指导

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "RAG", "检索增强生成，结合检索和生成的技术"
   "向量嵌入", "将文本转换为数值向量表示"
   "向量数据库", "存储和检索向量的数据库"
   "语义搜索", "基于语义相似度的搜索"
   "Chunk", "文档分割后的文本块"
   "Top-K", "返回最相似的K个结果"

下一步
======

在下一个教程中，我们将学习如何加载和处理各种格式的文档。

:doc:`tutorial_02_document_loading`
