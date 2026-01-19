####################################
Tutorial 2: 文档加载与处理
####################################

.. include:: ../links.ref
.. include:: ../tags.ref
.. include:: ../abbrs.ref

文档加载概述
============

RAG 系统的第一步是将各种格式的文档加载到系统中。

支持的文档格式
--------------

.. code-block:: text

   常见文档格式
   ├── 文本文件 (.txt, .md, .rst)
   ├── PDF 文档 (.pdf)
   ├── Office 文档 (.docx, .xlsx, .pptx)
   ├── 网页 (.html, URL)
   ├── 代码文件 (.py, .js, .java)
   └── 结构化数据 (.json, .csv)

使用 LangChain 加载文档
=======================

1. 加载文本文件
---------------

.. code-block:: python

   from langchain_community.document_loaders import TextLoader

   # 加载单个文本文件
   loader = TextLoader("./docs/readme.txt", encoding="utf-8")
   documents = loader.load()

   print(f"加载了 {len(documents)} 个文档")
   print(f"内容预览: {documents[0].page_content[:200]}...")
   print(f"元数据: {documents[0].metadata}")

2. 加载 PDF 文件
----------------

.. code-block:: python

   # pip install pypdf

   from langchain_community.document_loaders import PyPDFLoader

   # 加载 PDF
   loader = PyPDFLoader("./docs/manual.pdf")
   pages = loader.load()

   print(f"PDF 共 {len(pages)} 页")
   for i, page in enumerate(pages[:3]):
       print(f"\n第 {i+1} 页:")
       print(f"  内容: {page.page_content[:100]}...")
       print(f"  元数据: {page.metadata}")

   # 或者加载为单个文档
   loader = PyPDFLoader("./docs/manual.pdf")
   documents = loader.load_and_split()

3. 加载 Word 文档
-----------------

.. code-block:: python

   # pip install python-docx

   from langchain_community.document_loaders import Docx2txtLoader

   loader = Docx2txtLoader("./docs/report.docx")
   documents = loader.load()

   print(f"Word 文档内容: {documents[0].page_content[:200]}...")

4. 加载网页
-----------

.. code-block:: python

   from langchain_community.document_loaders import WebBaseLoader

   # 加载单个网页
   loader = WebBaseLoader("https://example.com/article")
   documents = loader.load()

   print(f"网页标题: {documents[0].metadata.get('title', 'N/A')}")
   print(f"内容: {documents[0].page_content[:300]}...")

   # 加载多个网页
   urls = [
       "https://example.com/page1",
       "https://example.com/page2",
   ]
   loader = WebBaseLoader(urls)
   documents = loader.load()

5. 加载 CSV 文件
----------------

.. code-block:: python

   from langchain_community.document_loaders import CSVLoader

   # 加载 CSV
   loader = CSVLoader(
       file_path="./data/products.csv",
       csv_args={
           "delimiter": ",",
           "quotechar": '"',
       }
   )
   documents = loader.load()

   print(f"加载了 {len(documents)} 行数据")
   print(f"示例: {documents[0].page_content}")

6. 加载目录
-----------

.. code-block:: python

   from langchain_community.document_loaders import DirectoryLoader

   # 加载目录下所有 txt 文件
   loader = DirectoryLoader(
       "./docs/",
       glob="**/*.txt",
       show_progress=True
   )
   documents = loader.load()

   print(f"从目录加载了 {len(documents)} 个文档")

   # 加载多种格式
   from langchain_community.document_loaders import (
       DirectoryLoader,
       TextLoader,
       PyPDFLoader
   )

   # 加载 txt 文件
   txt_loader = DirectoryLoader("./docs/", glob="**/*.txt", loader_cls=TextLoader)
   txt_docs = txt_loader.load()

   # 加载 pdf 文件
   pdf_loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
   pdf_docs = pdf_loader.load()

   all_docs = txt_docs + pdf_docs

自定义文档加载器
================

.. code-block:: python

   from langchain.schema import Document
   from typing import List
   import json

   class CustomJSONLoader:
       """自定义 JSON 加载器"""
       
       def __init__(self, file_path: str, content_key: str = "content"):
           self.file_path = file_path
           self.content_key = content_key
       
       def load(self) -> List[Document]:
           documents = []
           
           with open(self.file_path, 'r', encoding='utf-8') as f:
               data = json.load(f)
           
           # 假设 JSON 是一个列表
           if isinstance(data, list):
               for i, item in enumerate(data):
                   content = item.get(self.content_key, str(item))
                   metadata = {
                       "source": self.file_path,
                       "index": i,
                       **{k: v for k, v in item.items() if k != self.content_key}
                   }
                   documents.append(Document(
                       page_content=content,
                       metadata=metadata
                   ))
           
           return documents

   # 使用自定义加载器
   # 假设 JSON 格式: [{"title": "...", "content": "..."}, ...]
   loader = CustomJSONLoader("./data/articles.json", content_key="content")
   documents = loader.load()

文档预处理
==========

1. 清洗文本
-----------

.. code-block:: python

   import re

   def clean_text(text: str) -> str:
       """清洗文本"""
       # 去除多余空白
       text = re.sub(r'\s+', ' ', text)
       
       # 去除特殊字符
       text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:，。！？；：]', '', text)
       
       # 去除首尾空白
       text = text.strip()
       
       return text

   def clean_documents(documents):
       """清洗文档列表"""
       for doc in documents:
           doc.page_content = clean_text(doc.page_content)
       return documents

   # 使用
   documents = clean_documents(documents)

2. 过滤文档
-----------

.. code-block:: python

   def filter_documents(documents, min_length=50, max_length=10000):
       """过滤文档"""
       filtered = []
       
       for doc in documents:
           length = len(doc.page_content)
           
           # 过滤太短或太长的文档
           if min_length <= length <= max_length:
               filtered.append(doc)
           else:
               print(f"过滤文档: 长度={length}, 来源={doc.metadata.get('source', 'unknown')}")
       
       return filtered

   documents = filter_documents(documents)
   print(f"过滤后剩余 {len(documents)} 个文档")

3. 添加元数据
-------------

.. code-block:: python

   from datetime import datetime

   def enrich_metadata(documents, source_name: str):
       """丰富文档元数据"""
       for doc in documents:
           doc.metadata.update({
               "source_name": source_name,
               "indexed_at": datetime.now().isoformat(),
               "char_count": len(doc.page_content),
               "word_count": len(doc.page_content.split())
           })
       return documents

   documents = enrich_metadata(documents, "公司知识库")

实战：构建文档加载管道
======================

.. code-block:: python

   from langchain_community.document_loaders import (
       DirectoryLoader,
       TextLoader,
       PyPDFLoader,
       Docx2txtLoader
   )
   from langchain.schema import Document
   from typing import List
   import os
   import re
   from datetime import datetime

   class DocumentPipeline:
       """文档加载管道"""
       
       def __init__(self, base_path: str):
           self.base_path = base_path
           self.documents = []
       
       def load_all(self) -> List[Document]:
           """加载所有支持的文档"""
           
           # 加载 txt 文件
           txt_path = os.path.join(self.base_path, "**/*.txt")
           if self._has_files(txt_path):
               self._load_txt()
           
           # 加载 pdf 文件
           pdf_path = os.path.join(self.base_path, "**/*.pdf")
           if self._has_files(pdf_path):
               self._load_pdf()
           
           # 加载 docx 文件
           docx_path = os.path.join(self.base_path, "**/*.docx")
           if self._has_files(docx_path):
               self._load_docx()
           
           return self.documents
       
       def _has_files(self, pattern: str) -> bool:
           import glob
           return len(glob.glob(pattern, recursive=True)) > 0
       
       def _load_txt(self):
           loader = DirectoryLoader(
               self.base_path,
               glob="**/*.txt",
               loader_cls=TextLoader,
               loader_kwargs={"encoding": "utf-8"}
           )
           docs = loader.load()
           self._add_documents(docs, "txt")
       
       def _load_pdf(self):
           loader = DirectoryLoader(
               self.base_path,
               glob="**/*.pdf",
               loader_cls=PyPDFLoader
           )
           docs = loader.load()
           self._add_documents(docs, "pdf")
       
       def _load_docx(self):
           loader = DirectoryLoader(
               self.base_path,
               glob="**/*.docx",
               loader_cls=Docx2txtLoader
           )
           docs = loader.load()
           self._add_documents(docs, "docx")
       
       def _add_documents(self, docs: List[Document], doc_type: str):
           for doc in docs:
               doc.metadata["doc_type"] = doc_type
               doc.metadata["indexed_at"] = datetime.now().isoformat()
           self.documents.extend(docs)
       
       def clean(self) -> 'DocumentPipeline':
           """清洗文档"""
           for doc in self.documents:
               # 去除多余空白
               doc.page_content = re.sub(r'\s+', ' ', doc.page_content)
               doc.page_content = doc.page_content.strip()
           return self
       
       def filter(self, min_length: int = 50) -> 'DocumentPipeline':
           """过滤短文档"""
           self.documents = [
               doc for doc in self.documents
               if len(doc.page_content) >= min_length
           ]
           return self
       
       def get_stats(self) -> dict:
           """获取统计信息"""
           stats = {
               "total_documents": len(self.documents),
               "total_characters": sum(len(d.page_content) for d in self.documents),
               "by_type": {}
           }
           
           for doc in self.documents:
               doc_type = doc.metadata.get("doc_type", "unknown")
               stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
           
           return stats

   # 使用文档管道
   pipeline = DocumentPipeline("./knowledge_base/")
   documents = pipeline.load_all().clean().filter(min_length=100).documents

   stats = pipeline.get_stats()
   print(f"加载统计:")
   print(f"  总文档数: {stats['total_documents']}")
   print(f"  总字符数: {stats['total_characters']}")
   print(f"  按类型: {stats['by_type']}")

处理大文件
==========

.. code-block:: python

   from langchain_community.document_loaders import PyPDFLoader
   from typing import Iterator
   from langchain.schema import Document

   def load_large_pdf_lazy(file_path: str) -> Iterator[Document]:
       """惰性加载大型 PDF"""
       loader = PyPDFLoader(file_path)
       
       for page in loader.lazy_load():
           # 处理每一页后立即返回，不占用过多内存
           yield page

   # 使用生成器处理大文件
   for i, page in enumerate(load_large_pdf_lazy("large_document.pdf")):
       # 处理每一页
       print(f"处理第 {i+1} 页: {len(page.page_content)} 字符")
       
       # 可以在这里直接进行分块和索引
       # chunks = text_splitter.split_documents([page])
       # vectorstore.add_documents(chunks)

关键概念总结
============

.. csv-table::
   :header: "概念", "解释"
   :widths: 25, 75

   "Document", "LangChain 中的文档对象，包含内容和元数据"
   "Loader", "文档加载器，负责读取特定格式的文件"
   "Metadata", "文档的附加信息，如来源、页码等"
   "Lazy Loading", "惰性加载，按需读取，节省内存"
   "Pipeline", "文档处理管道，串联多个处理步骤"

下一步
======

在下一个教程中，我们将学习文本分块策略。

:doc:`tutorial_03_text_chunking`
