################################
Tutorial 2: 数据加载
################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

.. contents:: 目录
   :local:
   :depth: 2

数据加载概述
============

LlamaIndex 提供了强大的数据加载能力，支持 100+ 种数据源。
数据加载是 RAG 管道的第一步，也是最关键的一步。

.. code-block:: text

   数据加载流程：

   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  原始数据   │───►│  Reader     │───►│  Document   │
   │  (各种格式) │    │  (加载器)   │    │  (标准格式) │
   └─────────────┘    └─────────────┘    └─────────────┘

数据连接器类型
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 类型
     - 示例
     - 说明
   * - 本地文件
     - PDF, Word, Markdown
     - 最常用的数据源
   * - 数据库
     - MySQL, PostgreSQL
     - 结构化数据
   * - 云存储
     - S3, Google Drive
     - 远程文件
   * - Web
     - 网页, API
     - 在线数据
   * - 知识库
     - Notion, Confluence
     - 协作平台

SimpleDirectoryReader
=====================

最常用的文件加载器，支持多种文件格式。

基本用法
--------

.. code-block:: python

   from llama_index.core import SimpleDirectoryReader

   # 加载整个目录
   reader = SimpleDirectoryReader(input_dir="./data")
   documents = reader.load_data()

   print(f"加载了 {len(documents)} 个文档")
   for doc in documents:
       print(f"- {doc.metadata.get('file_name', 'unknown')}")

指定文件类型
------------

.. code-block:: python

   # 只加载特定类型的文件
   reader = SimpleDirectoryReader(
       input_dir="./data",
       required_exts=[".pdf", ".md", ".txt"]  # 只加载这些类型
   )
   documents = reader.load_data()

递归加载子目录
--------------

.. code-block:: python

   # 递归加载所有子目录
   reader = SimpleDirectoryReader(
       input_dir="./data",
       recursive=True,  # 包含子目录
       exclude=["*.log", "temp/*"]  # 排除模式
   )
   documents = reader.load_data()

自定义元数据
------------

.. code-block:: python

   def custom_metadata_func(file_path: str) -> dict:
       """自定义元数据提取函数"""
       import os
       from datetime import datetime

       stat = os.stat(file_path)
       return {
           "file_name": os.path.basename(file_path),
           "file_path": file_path,
           "file_size": stat.st_size,
           "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
           "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
       }

   reader = SimpleDirectoryReader(
       input_dir="./data",
       file_metadata=custom_metadata_func
   )
   documents = reader.load_data()

PDF 文件加载
============

基础 PDF 加载
-------------

.. code-block:: python

   from llama_index.readers.file import PDFReader

   # 使用专门的 PDF 读取器
   pdf_reader = PDFReader()
   documents = pdf_reader.load_data(file="./documents/report.pdf")

   for doc in documents:
       print(f"页码: {doc.metadata.get('page_label', 'N/A')}")
       print(f"内容预览: {doc.text[:200]}...")

高级 PDF 处理
-------------

.. code-block:: python

   # 安装高级 PDF 处理器
   # pip install llama-index-readers-llama-parse

   from llama_parse import LlamaParse

   # LlamaParse 提供更好的 PDF 解析
   parser = LlamaParse(
       api_key="your-llama-cloud-api-key",
       result_type="markdown",  # 输出 Markdown 格式
       language="zh",  # 支持中文
       verbose=True
   )

   documents = parser.load_data("./documents/complex_report.pdf")

处理扫描 PDF
------------

.. code-block:: python

   # 对于扫描的 PDF，需要 OCR 支持
   # pip install pytesseract pdf2image

   from llama_index.readers.file import ImageReader

   # 先将 PDF 转换为图片，再进行 OCR
   image_reader = ImageReader(
       text_type="plain_text",
       parser_config={"lang": "chi_sim+eng"}  # 中英文混合
   )

数据库加载
==========

SQL 数据库
----------

.. code-block:: python

   from llama_index.readers.database import DatabaseReader

   # 连接数据库
   db_reader = DatabaseReader(
       uri="mysql://user:password@localhost:3306/mydb"
   )

   # 执行 SQL 查询并加载结果
   documents = db_reader.load_data(
       query="SELECT id, title, content FROM articles WHERE status = 'published'"
   )

   print(f"从数据库加载了 {len(documents)} 条记录")

自定义数据库加载
----------------

.. code-block:: python

   from sqlalchemy import create_engine, text
   from llama_index.core import Document

   def load_from_database(connection_string: str, query: str) -> list:
       """自定义数据库加载函数"""
       engine = create_engine(connection_string)
       documents = []

       with engine.connect() as conn:
           result = conn.execute(text(query))
           for row in result:
               doc = Document(
                   text=str(row.content),
                   metadata={
                       "id": row.id,
                       "title": row.title,
                       "source": "database",
                       "table": "articles"
                   }
               )
               documents.append(doc)

       return documents

   # 使用自定义加载器
   docs = load_from_database(
       "postgresql://user:pass@localhost/db",
       "SELECT * FROM knowledge_base"
   )

Web 数据加载
============

网页爬取
--------

.. code-block:: python

   from llama_index.readers.web import SimpleWebPageReader

   # 加载单个网页
   web_reader = SimpleWebPageReader(html_to_text=True)
   documents = web_reader.load_data(
       urls=["https://docs.llamaindex.ai/en/stable/"]
   )

   print(f"网页内容长度: {len(documents[0].text)}")

递归网页爬取
------------

.. code-block:: python

   from llama_index.readers.web import WholeSiteReader

   # 爬取整个网站（谨慎使用）
   site_reader = WholeSiteReader(
       prefix="https://docs.llamaindex.ai",
       max_depth=2  # 最大深度
   )

   documents = site_reader.load_data(
       base_url="https://docs.llamaindex.ai/en/stable/"
   )

API 数据加载
------------

.. code-block:: python

   import requests
   from llama_index.core import Document

   def load_from_api(api_url: str, headers: dict = None) -> list:
       """从 API 加载数据"""
       response = requests.get(api_url, headers=headers)
       data = response.json()

       documents = []
       for item in data.get("items", []):
           doc = Document(
               text=item.get("content", ""),
               metadata={
                   "id": item.get("id"),
                   "title": item.get("title"),
                   "source": "api",
                   "url": api_url
               }
           )
           documents.append(doc)

       return documents

   # 示例：从 GitHub API 加载
   docs = load_from_api(
       "https://api.github.com/repos/run-llama/llama_index/readme",
       headers={"Accept": "application/vnd.github.v3+json"}
   )

云存储加载
==========

AWS S3
------

.. code-block:: python

   # pip install llama-index-readers-s3

   from llama_index.readers.s3 import S3Reader

   s3_reader = S3Reader(
       bucket="my-bucket",
       prefix="documents/",  # 可选前缀
       aws_access_id="your-access-key",
       aws_access_secret="your-secret-key"
   )

   documents = s3_reader.load_data()

Google Drive
------------

.. code-block:: python

   # pip install llama-index-readers-google

   from llama_index.readers.google import GoogleDriveReader

   drive_reader = GoogleDriveReader(
       credentials_path="./credentials.json"
   )

   # 加载特定文件夹
   documents = drive_reader.load_data(folder_id="your-folder-id")

知识库平台
==========

Notion
------

.. code-block:: python

   # pip install llama-index-readers-notion

   from llama_index.readers.notion import NotionPageReader

   notion_reader = NotionPageReader(
       integration_token="your-notion-token"
   )

   # 加载特定页面
   documents = notion_reader.load_data(
       page_ids=["page-id-1", "page-id-2"]
   )

   # 或加载整个数据库
   documents = notion_reader.load_data(
       database_id="your-database-id"
   )

Confluence
----------

.. code-block:: python

   # pip install llama-index-readers-confluence

   from llama_index.readers.confluence import ConfluenceReader

   confluence_reader = ConfluenceReader(
       base_url="https://your-domain.atlassian.net/wiki",
       user="your-email@example.com",
       api_token="your-api-token"
   )

   # 加载特定空间
   documents = confluence_reader.load_data(
       space_key="MYSPACE",
       include_attachments=True
   )

自定义 Reader
=============

创建自定义数据加载器
--------------------

.. code-block:: python

   from llama_index.core.readers.base import BaseReader
   from llama_index.core import Document
   from typing import List
   import json

   class CustomJSONReader(BaseReader):
       """自定义 JSON 文件读取器"""

       def __init__(self, text_field: str = "content"):
           self.text_field = text_field

       def load_data(self, file_path: str) -> List[Document]:
           with open(file_path, 'r', encoding='utf-8') as f:
               data = json.load(f)

           documents = []
           items = data if isinstance(data, list) else [data]

           for item in items:
               text = item.get(self.text_field, "")
               metadata = {k: v for k, v in item.items() if k != self.text_field}

               doc = Document(text=text, metadata=metadata)
               documents.append(doc)

           return documents

   # 使用自定义读取器
   reader = CustomJSONReader(text_field="body")
   documents = reader.load_data("./data/articles.json")

批量加载多种格式
----------------

.. code-block:: python

   from pathlib import Path
   from llama_index.core import Document

   class MultiFormatLoader:
       """支持多种格式的批量加载器"""

       def __init__(self):
           self.readers = {}

       def register_reader(self, extension: str, reader):
           """注册文件格式对应的读取器"""
           self.readers[extension] = reader

       def load_directory(self, directory: str) -> List[Document]:
           """加载目录下所有支持的文件"""
           documents = []
           dir_path = Path(directory)

           for file_path in dir_path.rglob("*"):
               if file_path.is_file():
                   ext = file_path.suffix.lower()
                   if ext in self.readers:
                       docs = self.readers[ext].load_data(str(file_path))
                       documents.extend(docs)

           return documents

   # 使用示例
   loader = MultiFormatLoader()
   loader.register_reader(".pdf", PDFReader())
   loader.register_reader(".json", CustomJSONReader())

   all_docs = loader.load_directory("./data")

数据预处理
==========

文档清洗
--------

.. code-block:: python

   import re
   from llama_index.core import Document

   def clean_document(doc: Document) -> Document:
       """清洗文档内容"""
       text = doc.text

       # 移除多余空白
       text = re.sub(r'\s+', ' ', text)

       # 移除特殊字符
       text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:\'\"()-]', '', text)

       # 规范化标点
       text = text.strip()

       return Document(text=text, metadata=doc.metadata)

   # 批量清洗
   cleaned_docs = [clean_document(doc) for doc in documents]

添加全局元数据
--------------

.. code-block:: python

   def add_global_metadata(documents: List[Document], metadata: dict) -> List[Document]:
       """为所有文档添加全局元数据"""
       for doc in documents:
           doc.metadata.update(metadata)
       return documents

   # 添加项目信息
   documents = add_global_metadata(documents, {
       "project": "knowledge-base",
       "version": "1.0",
       "indexed_at": "2024-01-01"
   })

实战示例：多源数据整合
======================

.. code-block:: python

   from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
   from llama_index.readers.database import DatabaseReader
   from llama_index.readers.web import SimpleWebPageReader

   def build_unified_knowledge_base():
       """构建统一的知识库"""
       all_documents = []

       # 1. 加载本地文档
       print("加载本地文档...")
       local_reader = SimpleDirectoryReader(
           input_dir="./documents",
           recursive=True,
           required_exts=[".pdf", ".md", ".txt"]
       )
       local_docs = local_reader.load_data()
       for doc in local_docs:
           doc.metadata["source_type"] = "local_file"
       all_documents.extend(local_docs)
       print(f"  - 加载了 {len(local_docs)} 个本地文档")

       # 2. 加载数据库记录
       print("加载数据库记录...")
       db_reader = DatabaseReader(uri="sqlite:///knowledge.db")
       db_docs = db_reader.load_data(
           query="SELECT title, content FROM knowledge WHERE active = 1"
       )
       for doc in db_docs:
           doc.metadata["source_type"] = "database"
       all_documents.extend(db_docs)
       print(f"  - 加载了 {len(db_docs)} 条数据库记录")

       # 3. 加载网页内容
       print("加载网页内容...")
       web_reader = SimpleWebPageReader(html_to_text=True)
       web_docs = web_reader.load_data(urls=[
           "https://example.com/doc1",
           "https://example.com/doc2"
       ])
       for doc in web_docs:
           doc.metadata["source_type"] = "web"
       all_documents.extend(web_docs)
       print(f"  - 加载了 {len(web_docs)} 个网页")

       # 4. 构建统一索引
       print(f"\n总共加载 {len(all_documents)} 个文档")
       print("构建向量索引...")
       index = VectorStoreIndex.from_documents(all_documents)

       return index

   # 构建知识库
   knowledge_index = build_unified_knowledge_base()

   # 查询
   query_engine = knowledge_index.as_query_engine()
   response = query_engine.query("总结我们的产品特点")
   print(response)

小结
====

本教程介绍了：

- SimpleDirectoryReader 的使用方法
- 各种文件格式的加载：PDF、数据库、Web 等
- 云存储和知识库平台的数据加载
- 自定义 Reader 的创建
- 数据预处理和清洗
- 多源数据整合的最佳实践

下一步
------

在下一个教程中，我们将学习节点解析（Node Parsing），
了解如何将文档智能地分割为更小的、可索引的单元。

练习
====

1. 使用 SimpleDirectoryReader 加载你的本地文档
2. 创建一个自定义 Reader 加载特定格式的数据
3. 尝试从数据库或 API 加载数据
4. 实现一个多源数据整合的知识库
