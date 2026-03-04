(appendix_a)=
# 附录 A：实战项目 — AI Coach 虚拟学习教练

```{admonition} 项目概述
:class: tip
AI Coach 是一个基于 RAG + Agent 的个人知识库与学习督导系统。它展示了如何将本书中讲解的 Prompt Engineering、RAG 架构、Agent 框架、FastAPI 后端和 Vue.js 前端等知识点融合到一个完整的全栈应用中。

📁 源码位置：`examples/ai-coach/`
```

## A.1 项目背景与设计思路

### 为什么做这个项目？

在学习技术的过程中，我们常常面临三个痛点：

1. **知识碎片化**：学习资料散落在各处，需要时找不到
2. **缺乏督促**：自学容易三天打鱼两天晒网
3. **反馈不足**：不知道自己学得怎么样，哪里需要加强

AI Coach 正是为了解决这三个问题而设计的。它不是一个简单的聊天机器人，而是一个**有记忆、有知识、有策略**的学习伙伴。

### 设计原则

在动手编码之前，我们确立了几个设计原则：

- **知识驱动**：所有回答都基于用户上传的知识库，而非 AI 的通用知识
- **多模式交互**：教练督促、导师讲解、测验考核，满足不同学习场景
- **数据追踪**：记录学习时长、连续天数、难度感受，用数据驱动反馈
- **流式体验**：SSE 流式输出，像真人对话一样逐字显示

## A.2 技术架构

### 整体架构图

```
┌─────────────────────────────────────────────────────┐
│                    Vue.js 3 前端                     │
│            Vite + Tailwind CSS + Pinia               │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ 对话页面  │  │  知识库页面   │  │  学习计划页面  │  │
│  │ SSE 流式  │  │  上传/检索   │  │  目标/进度    │  │
│  └─────┬────┘  └──────┬───────┘  └───────┬───────┘  │
│        │              │                  │           │
│  ┌─────▼──────────────▼──────────────────▼────────┐  │
│  │          Axios + Token 拦截器 + SSE 客户端      │  │
│  └─────────────────────┬──────────────────────────┘  │
├────────────────────────┼─────────────────────────────┤
│                   FastAPI 后端                        │
│  ┌─────────┐  ┌────────▼────────┐  ┌──────────────┐ │
│  │Auth API │  │  Chat API       │  │ Learning API │ │
│  │JWT+RBAC │  │  普通 + SSE     │  │ 目标+记录    │ │
│  └────┬────┘  └────────┬────────┘  └──────┬───────┘ │
│       │               │                   │          │
│  ┌────▼────┐  ┌───────▼────────┐  ┌──────▼───────┐  │
│  │  User   │  │  CoachAgent    │  │  Progress    │  │
│  │  Model  │  │  3 种模式      │  │  Tracker     │  │
│  └────┬────┘  └───────┬────────┘  └──────┬───────┘  │
│       │               │                   │          │
│  ┌────▼────┐  ┌───────▼────────┐  ┌──────▼───────┐  │
│  │ SQLite  │  │  RAG Engine    │  │   SQLite     │  │
│  │ (users) │  │  LlamaIndex    │  │  (goals)     │  │
│  └─────────┘  └───────┬────────┘  └──────────────┘  │
│                       │                              │
│               ┌───────▼────────┐                     │
│               │   ChromaDB     │                     │
│               │   向量数据库    │                     │
│               └────────────────┘                     │
└──────────────────────────────────────────────────────┘
```

### 技术选型理由

| 技术 | 选择 | 理由 |
|------|------|------|
| 后端框架 | FastAPI | 异步原生、自动 API 文档、类型安全 |
| ORM | SQLAlchemy 2.0 async | 异步支持好、生态成熟 |
| 向量数据库 | ChromaDB | 轻量级、嵌入式、适合单机部署 |
| RAG 框架 | LlamaIndex | 抽象层次高、开箱即用 |
| LLM | OpenAI GPT-4o | 中文能力强、指令遵循好 |
| 前端框架 | Vue.js 3 + Vite | 轻量、响应式、开发体验好 |
| CSS | Tailwind CSS | 原子化、快速原型、一致性好 |
| 认证 | JWT + bcrypt | 无状态、前后端分离友好 |

## A.3 核心实现详解

### A.3.1 RAG 引擎：让 AI 基于你的知识回答

RAG（Retrieval-Augmented Generation）是本项目的核心。它让 AI 不再"胡说八道"，而是基于你上传的学习资料来回答问题。

**工作流程：**

```
用户提问 → 向量检索（ChromaDB）→ 取回相关文档片段 → 注入 Prompt → LLM 生成回答
```

**关键代码解析：**

```python
# app/rag/engine.py — 文档添加
async def add_document(self, title, content, tags, doc_id):
    doc = LlamaDocument(text=content, metadata={"title": title, "tags": ",".join(tags)})
    
    # 分块：将长文档切成小段，每段 512 个 token，重叠 50 个
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents([doc])
    
    # 向量化并存入 ChromaDB
    self.index.insert_nodes(nodes)
```

```{admonition} 分块策略的选择
:class: note
分块大小（chunk_size）是 RAG 系统最重要的超参数之一：
- **太大**（>1024）：检索精度下降，可能引入无关内容
- **太小**（<256）：丢失上下文，回答不完整
- **推荐**：技术文档用 512-1024，对话记录用 256-512

重叠（chunk_overlap）确保跨块的句子不会被截断，通常设为 chunk_size 的 10-20%。
```

**查询流程：**

```python
# app/rag/engine.py — 知识检索
async def query(self, question, top_k=5):
    retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
    
    # 过滤低相关度结果（相似度 < 0.5 的丢弃）
    postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.5)]
    
    # 用检索到的内容生成回答
    synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=QA_PROMPT,  # 自定义 Prompt
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=postprocessors,
    )
    response = query_engine.query(question)
```

### A.3.2 CoachAgent：三种模式的智能体

CoachAgent 是项目的"灵魂"。它不是一个简单的 API 调用封装，而是一个有角色、有策略、有记忆的智能体。

**三种模式的 Prompt 设计：**

| 模式 | 角色定位 | 温度 | 核心指令 |
|------|---------|------|---------|
| 🎯 教练 | 亲切但有要求的导师 | 0.7 | 督促进度、制定计划、鼓励激励 |
| 📖 导师 | 耐心的大学教授 | 0.7 | 深入讲解、举例说明、循序渐进 |
| ✍️ 测验 | 严格的考试官 | 0.7 | 出题考核、评判答案、难度递进 |

```{admonition} Prompt 设计心得
:class: tip
好的系统 Prompt 应该包含四个要素：
1. **角色定义**：你是谁？（"你是一位经验丰富的学习教练"）
2. **职责清单**：你要做什么？（编号列出 5 个具体职责）
3. **风格约束**：怎么做？（"像一位亲切但专业的导师"）
4. **格式要求**：输出什么？（"使用中文回答，适当使用 emoji"）
```

**流式输出的实现：**

```python
# app/agents/coach.py — 流式生成
async def respond_stream(self, message, history, mode):
    # 1. 先做 RAG 检索（非流式，很快）
    rag_context = await self.rag_engine.query(message)
    
    # 2. 构建消息列表
    messages = [system_prompt + rag_context] + history + [user_message]
    
    # 3. 流式调用 LLM
    response_gen = self.llm.stream_chat(messages)
    for chunk in response_gen:
        yield {"token": chunk.delta}  # 逐 token 返回
```

### A.3.3 SSE 流式对话

传统的 HTTP 请求-响应模式需要等 AI 生成完整回复才能显示，用户体验差。SSE（Server-Sent Events）让我们可以逐字推送，实现"打字机效果"。

**后端 SSE 端点：**

```python
# app/api/chat.py
@router.post("/message/stream")
async def send_message_stream(chat_req: ChatRequest):
    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'session_id': sid})}\n\n"
        
        async for chunk in coach.respond_stream(message, history, mode):
            yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**前端 SSE 客户端：**

```javascript
// src/utils/api.js
export function streamChat(message, sessionId, mode, onToken, onDone, onError) {
  fetch('/api/chat/message/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ message, session_id: sessionId, mode }),
  }).then(async (response) => {
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      // 解析 SSE 数据，逐 token 回调
      const data = JSON.parse(line.slice(6))
      if (data.type === 'token') onToken(data.token)
    }
  })
}
```

```{admonition} 为什么选 SSE 而不是 WebSocket？
:class: note
- **SSE** 是单向的（服务器→客户端），基于 HTTP，更简单
- **WebSocket** 是双向的，适合实时聊天、游戏等场景
- 对于 AI 对话，用户发一条消息、AI 回一条消息，SSE 完全够用
- SSE 天然支持自动重连，且不需要额外的协议升级
```

### A.3.4 JWT 认证与角色授权

**认证流程：**

```
注册/登录 → 服务器返回 access_token + refresh_token
    ↓
前端存入 localStorage
    ↓
每次请求自动附加 Authorization: Bearer <token>
    ↓
后端解码 JWT，验证用户身份
    ↓
token 过期 → 用 refresh_token 换新 token
```

**角色授权：**

```python
# 三级角色体系
role_hierarchy = {"admin": 3, "editor": 2, "user": 1}

# 使用方式：保护需要 admin 权限的端点
@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user: dict = Depends(require_role("admin"))):
    ...
```

**前端路由守卫：**

```javascript
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  if (!to.meta.public && !token) {
    next('/login')  // 未登录，跳转登录页
  } else {
    next()
  }
})
```

## A.4 学习计划追踪

学习计划模块不仅记录数据，还会基于数据生成 AI 教练反馈：

```python
# 计算统计数据
total_hours = sum(s.duration_minutes for s in sessions) / 60
streak_days = calculate_streak(sessions)  # 连续学习天数
completion_pct = total_minutes / expected_minutes * 100

# AI 教练基于数据给反馈
coach_prompt = f"""
- 学习主题：{goal.topic}
- 已学习：{total_hours} 小时
- 连续学习：{streak_days} 天
- 完成度：{completion_pct}%
- 平均难度：{avg_difficulty}/5
请给出：1) 一句鼓励的话 2) 2-3 条具体建议
"""
```

这种"数据 + AI"的模式，让反馈不再是泛泛而谈，而是基于用户真实学习数据的个性化建议。

## A.5 项目运行指南

### 环境准备

```bash
# 前置要求
- Python 3.11+
- Node.js 18+
- OpenAI API Key
```

### 快速启动

```bash
# 1. 启动后端
cd examples/ai-coach/backend
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
# → http://localhost:8000 (API 文档: /docs)

# 2. 启动前端（新终端）
cd examples/ai-coach/frontend
npm install
npm run dev
# → http://localhost:5173
```

### Docker 一键部署

```bash
cd examples/ai-coach
cp backend/.env.example backend/.env
# 编辑 .env
docker compose up -d
# → http://localhost:5173
```

### 使用流程

1. **注册账号** → 进入对话页面
2. **上传知识** → 知识库页面，上传你的学习资料（Markdown、TXT）
3. **开始对话** → 选择教练/导师/测验模式，与 AI 交流
4. **设定目标** → 学习计划页面，创建学习目标
5. **记录学习** → 每次学习后记录时长和笔记
6. **查看进度** → 获取 AI 教练的个性化反馈

## A.6 涉及的书中知识点

| 章节 | 知识点 | 在项目中的体现 |
|------|--------|---------------|
| 第 6 章 | Prompt Engineering | CoachAgent 三种模式的系统提示词设计 |
| 第 17 章 | RAG 架构 | LlamaIndex + ChromaDB 知识库检索增强 |
| 第 17 章 | 分块策略 | SentenceSplitter 的 chunk_size/overlap 配置 |
| 第 18 章 | Agent 框架 | CoachAgent 多模式智能体实现 |
| 第 21 章 | 多 Agent 协作 | 教练、导师、测验三种角色的切换 |
| 第 22 章 | Agent 评估 | 学习进度追踪作为间接评估手段 |

## A.7 启发与思考

### 对读者的启发

1. **RAG 不只是"检索+生成"**：好的 RAG 系统需要精心设计分块策略、相似度阈值、Prompt 模板。本项目展示了一个完整的 RAG 工程实践。

2. **Agent 的核心是 Prompt**：CoachAgent 的三种模式，本质上是三套不同的系统 Prompt。好的 Prompt 设计能让同一个 LLM 表现出截然不同的"人格"。

3. **数据驱动的 AI 反馈**：不要让 AI 凭空给建议。把用户的真实数据（学习时长、难度感受、连续天数）注入 Prompt，AI 的反馈会更有针对性。

4. **SSE 是 AI 应用的标配**：流式输出不是锦上添花，而是必需品。用户等待 10 秒看到完整回复 vs 立即看到逐字输出，体验天差地别。

### 读者可以怎么做

```{admonition} 动手实践建议
:class: tip

**初级挑战：**
- 修改 CoachAgent 的 Prompt，添加一个新模式（如"面试官模式"）
- 在学习计划中添加"每周总结"功能
- 支持上传 PDF 文件（提示：使用 `pypdf` 库）

**中级挑战：**
- 将 ChromaDB 替换为 Milvus 或 Qdrant，对比性能
- 添加多用户支持：每个用户有独立的知识库
- 实现"学习小组"功能：多个用户共享知识库

**高级挑战：**
- 将 CoachAgent 改造为 LangGraph 状态机，实现更复杂的对话流程
- 添加语音输入/输出（Whisper + TTS）
- 部署到云端，添加 CI/CD 流水线
- 实现 Agent 自主学习：根据用户反馈自动优化 Prompt
```

## A.8 项目文件清单

```
ai-coach/                          # 43 个文件
├── README.md                      # 项目文档
├── docker-compose.yml             # Docker 编排
├── backend/                       # Python 后端
│   ├── Dockerfile
│   ├── .env.example
│   ├── requirements.txt
│   ├── run.py
│   └── app/
│       ├── main.py                # FastAPI 入口
│       ├── core/
│       │   ├── config.py          # 配置（JWT_SECRET, OPENAI_API_KEY...）
│       │   ├── database.py        # SQLAlchemy async
│       │   └── auth.py            # JWT 认证 + 角色授权
│       ├── models/
│       │   ├── schemas.py         # Pydantic 模型
│       │   └── db_models.py       # ORM（User, Document, Goal, Session...）
│       ├── api/
│       │   ├── auth.py            # 注册/登录/刷新
│       │   ├── chat.py            # 对话（普通 + SSE 流式）
│       │   ├── knowledge.py       # 知识库 CRUD + RAG 查询
│       │   ├── learning.py        # 学习目标 + 进度追踪
│       │   └── health.py          # 健康检查
│       ├── agents/
│       │   └── coach.py           # CoachAgent（教练/导师/测验）
│       └── rag/
│           └── engine.py          # RAG 引擎（LlamaIndex + ChromaDB）
└── frontend/                      # Vue.js 前端
    ├── Dockerfile
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── src/
        ├── main.js
        ├── App.vue                # 侧边栏布局
        ├── router.js              # 路由 + 守卫
        ├── stores/
        │   └── auth.js            # Pinia 认证状态
        ├── views/
        │   ├── LoginView.vue      # 登录/注册
        │   ├── ChatView.vue       # 对话（SSE 流式）
        │   ├── KnowledgeView.vue  # 知识库管理
        │   └── LearningView.vue   # 学习计划
        └── utils/
            ├── api.js             # Axios + Token 拦截器 + SSE
            └── markdown.js        # Markdown 渲染
```
