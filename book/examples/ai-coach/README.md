# 🎓 AI Coach — 虚拟学习教练

一个基于 RAG + Agent 的个人知识库与学习督导系统。这是《AI 时代的软件工程》一书的配套实战项目。

## ✨ 功能特性

### 💬 AI 教练对话
- **教练模式**：督促学习进度，制定计划，给予鼓励
- **导师模式**：深入讲解技术概念，循序渐进
- **测验模式**：出题考核，评判答案，难度递进

### 📚 个人知识库
- 上传文档（TXT、MD、PDF）构建个人知识库
- 基于 LlamaIndex + ChromaDB 的 RAG 检索
- 知识库驱动的智能问答

### 📊 学习计划追踪
- 设定学习目标和每日时长
- 记录学习会话和笔记
- 连续学习天数、完成度统计
- AI 教练基于数据给出个性化反馈

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────┐
│                  Vue.js 3 前端               │
│         Vite + Tailwind CSS + Pinia          │
├─────────────────────────────────────────────┤
│                FastAPI 后端                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ 知识库API │ │ 对话 API │ │ 学习计划 API │ │
│  └────┬─────┘ └────┬─────┘ └──────┬───────┘ │
│       │            │              │          │
│  ┌────▼─────┐ ┌────▼─────┐ ┌─────▼──────┐  │
│  │RAG Engine│ │CoachAgent│ │ Progress   │  │
│  │LlamaIndex│ │ 3 Modes  │ │ Tracker    │  │
│  └────┬─────┘ └────┬─────┘ └─────┬──────┘  │
│       │            │              │          │
│  ┌────▼─────┐ ┌────▼─────┐ ┌─────▼──────┐  │
│  │ ChromaDB │ │ OpenAI   │ │  SQLite    │  │
│  │向量数据库 │ │ GPT-4o   │ │  数据库    │  │
│  └──────────┘ └──────────┘ └────────────┘  │
└─────────────────────────────────────────────┘
```

## 🚀 快速开始

### 前置要求
- Python 3.11+
- Node.js 18+
- OpenAI API Key

### 方式一：本地运行

```bash
# 1. 克隆项目
cd examples/ai-coach

# 2. 启动后端
cd backend
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

python run.py
# 后端运行在 http://localhost:8000
# API 文档：http://localhost:8000/docs

# 3. 启动前端（新终端）
cd frontend
npm install
npm run dev
# 前端运行在 http://localhost:5173
```

### 方式二：Docker Compose

```bash
cd examples/ai-coach
cp backend/.env.example backend/.env
# 编辑 backend/.env，填入 OPENAI_API_KEY

docker compose up -d
# 访问 http://localhost:5173
```

## 📁 项目结构

```
ai-coach/
├── backend/                    # Python 后端
│   ├── app/
│   │   ├── main.py            # FastAPI 入口
│   │   ├── api/               # API 路由
│   │   │   ├── chat.py        # 对话接口
│   │   │   ├── knowledge.py   # 知识库接口
│   │   │   ├── learning.py    # 学习计划接口
│   │   │   └── health.py      # 健康检查
│   │   ├── agents/
│   │   │   └── coach.py       # AI 教练 Agent
│   │   ├── rag/
│   │   │   └── engine.py      # RAG 引擎
│   │   ├── models/
│   │   │   ├── schemas.py     # Pydantic 模型
│   │   │   └── db_models.py   # ORM 模型
│   │   ├── core/
│   │   │   ├── config.py      # 配置管理
│   │   │   └── database.py    # 数据库连接
│   │   └── services/          # 业务逻辑
│   ├── requirements.txt
│   ├── run.py
│   └── .env.example
├── frontend/                   # Vue.js 前端
│   ├── src/
│   │   ├── App.vue            # 根组件
│   │   ├── main.js            # 入口
│   │   ├── router.js          # 路由
│   │   ├── views/             # 页面
│   │   │   ├── ChatView.vue   # 对话页
│   │   │   ├── KnowledgeView.vue  # 知识库页
│   │   │   └── LearningView.vue   # 学习计划页
│   │   └── utils/
│   │       ├── api.js         # API 客户端
│   │       └── markdown.js    # Markdown 渲染
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml
└── README.md
```

## 📖 涉及的书中知识点

| 章节 | 知识点 | 在项目中的体现 |
|------|--------|---------------|
| 第 6 章 | Prompt Engineering | CoachAgent 的系统提示词设计 |
| 第 12 章 | Cursor 开发 | 项目可用 Cursor 辅助开发 |
| 第 17 章 | RAG 架构 | LlamaIndex + ChromaDB 知识库 |
| 第 17 章 | 分块策略 | SentenceSplitter 配置 |
| 第 18 章 | Agent 框架 | CoachAgent 多模式智能体 |
| 第 21 章 | FastAPI | 后端 API 设计 |
| 第 22 章 | 前端开发 | Vue.js 3 + Tailwind CSS |

## 📝 License

MIT — 仅供学习参考
