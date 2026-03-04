# ✏️ AI Editor — 虚拟书稿编辑

一个基于 AI Agent 的书稿编写、修改和校对系统。这是《AI 时代的软件工程》一书的配套实战项目。

## ✨ 功能特性

### 📖 章节管理
- 创建、编辑、删除书稿章节
- Markdown 实时预览
- 版本追踪

### ✏️ AI 编辑（7 种操作）
- **🔍 校对**：修正错别字、语法、标点
- **✨ 润色**：改善表达、提升可读性
- **📝 扩写**：补充内容、增加细节和代码示例
- **📐 缩写**：精简内容、去除冗余
- **🔄 重构**：调整结构、重新组织
- **🌐 翻译**：中英互译，保持技术术语准确
- **📋 审查**：给出修改建议（不直接修改）

### ✍️ AI 写作
- 输入主题和大纲，AI 生成完整章节
- 支持技术书籍、博客、学术三种风格
- 可设置目标字数和上下文

### 💬 编辑对话
- 与 AI 编辑讨论书稿内容和结构
- 头脑风暴、解决写作瓶颈
- 获取写作技巧建议

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────┐
│                Vue.js 3 前端                 │
│         Vite + Tailwind CSS + Pinia          │
├─────────────────────────────────────────────┤
│                FastAPI 后端                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ 章节 API │ │ 编辑 API │ │  写作 API    │ │
│  └────┬─────┘ └────┬─────┘ └──────┬───────┘ │
│       │            │              │          │
│  ┌────▼─────┐ ┌────▼─────┐ ┌─────▼──────┐  │
│  │ Chapter  │ │ Editor   │ │  Writer    │  │
│  │ CRUD     │ │ Agent    │ │  Agent     │  │
│  └────┬─────┘ └────┬─────┘ └─────┬──────┘  │
│       │            │              │          │
│  ┌────▼─────┐ ┌────▼─────┐ ┌─────▼──────┐  │
│  │ SQLite   │ │ OpenAI   │ │ diff-match  │  │
│  │ 数据库   │ │ GPT-4o   │ │ -patch     │  │
│  └──────────┘ └──────────┘ └────────────┘  │
└─────────────────────────────────────────────┘
```

## 🚀 快速开始

### 前置要求
- Python 3.11+
- Node.js 18+
- OpenAI API Key

### 本地运行

```bash
# 1. 进入项目
cd examples/ai-editor

# 2. 启动后端
cd backend
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python run.py
# 后端运行在 http://localhost:8001
# API 文档：http://localhost:8001/docs

# 3. 启动前端（新终端）
cd frontend
npm install
npm run dev
# 前端运行在 http://localhost:5174
```

### Docker Compose

```bash
cd examples/ai-editor
cp backend/.env.example backend/.env
# 编辑 .env，填入 OPENAI_API_KEY

docker compose up -d
# 访问 http://localhost:5174
```

## 📁 项目结构

```
ai-editor/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── api/
│   │   │   ├── chapters.py      # 章节 CRUD
│   │   │   ├── editor.py        # AI 编辑接口
│   │   │   ├── writer.py        # AI 写作接口
│   │   │   └── chat.py          # 编辑对话
│   │   ├── agents/
│   │   │   ├── editor_agent.py  # 编辑 + 写作 Agent
│   │   │   └── chat_agent.py    # 对话 Agent
│   │   ├── models/
│   │   │   ├── schemas.py       # Pydantic 模型
│   │   │   └── db_models.py     # ORM 模型
│   │   └── core/
│   │       ├── config.py        # 配置
│   │       └── database.py      # 数据库
│   ├── requirements.txt
│   └── run.py
├── frontend/
│   ├── src/
│   │   ├── App.vue
│   │   ├── views/
│   │   │   ├── ChaptersView.vue # 章节管理
│   │   │   ├── EditorView.vue   # AI 编辑器
│   │   │   ├── WriterView.vue   # AI 写作
│   │   │   └── ChatView.vue     # 编辑对话
│   │   └── utils/
│   │       ├── api.js           # API 客户端
│   │       └── markdown.js      # Markdown 渲染
│   └── package.json
├── docker-compose.yml
└── README.md
```

## 🔧 AI 编辑操作详解

| 操作 | 说明 | 温度 | 适用场景 |
|------|------|------|---------|
| 校对 | 修正错误，不改风格 | 0.3 | 最终发布前 |
| 润色 | 改善表达，保持原意 | 0.7 | 初稿完成后 |
| 扩写 | 补充细节和示例 | 0.7 | 内容不够充实 |
| 缩写 | 精简 30-50% | 0.7 | 内容过于冗长 |
| 重构 | 调整结构和逻辑 | 0.7 | 结构混乱时 |
| 翻译 | 中英互译 | 0.3 | 多语言版本 |
| 审查 | 只给建议不修改 | 0.7 | 需要第二意见 |

## 📖 涉及的书中知识点

| 章节 | 知识点 | 在项目中的体现 |
|------|--------|---------------|
| 第 6 章 | Prompt Engineering | 7 种编辑操作的 Prompt 设计 |
| 第 17 章 | Agent 架构 | EditorAgent / WriterAgent / ChatAgent |
| 第 18 章 | 多 Agent 协作 | 编辑、写作、对话三个 Agent |
| 第 21 章 | FastAPI | 后端 API 设计 |
| 第 22 章 | 前端开发 | Vue.js 3 + Tailwind CSS |

## 📝 License

MIT — 仅供学习参考
