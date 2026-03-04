(appendix_b)=
# 附录 B：实战项目 — AI Editor 虚拟书稿编辑

```{admonition} 项目概述
:class: tip
AI Editor 是一个基于多 Agent 的书稿编写、修改和校对系统。它展示了如何用 Prompt Engineering 实现 7 种专业编辑操作，如何设计多 Agent 协作架构，以及如何用 diff 算法可视化 AI 的修改。

📁 源码位置：`examples/ai-editor/`
```

## B.1 项目背景与设计思路

### 为什么做这个项目？

写一本技术书是一个漫长的过程。作者常常面临这些挑战：

1. **校对繁琐**：几万字的书稿，逐字检查错别字和语法错误，耗时且容易遗漏
2. **润色困难**：知道哪里写得不好，但不知道怎么改更好
3. **结构混乱**：写着写着就跑题了，段落之间缺乏逻辑连贯性
4. **翻译需求**：技术书籍常需要中英双语版本

AI Editor 就是为了解决这些问题而设计的。它不是要替代人类编辑，而是做一个**7×24 小时在线的编辑助手**，随时可以帮你校对、润色、扩写、缩写、重构、翻译和审查。

### 与 AI Coach 的区别

| 维度 | AI Coach | AI Editor |
|------|----------|-----------|
| 核心技术 | RAG + 知识检索 | Prompt Engineering + Diff |
| Agent 数量 | 1 个（多模式） | 3 个（编辑/写作/对话） |
| 数据流 | 用户上传知识 → AI 检索回答 | 用户提交书稿 → AI 编辑修改 |
| 输出形式 | 对话回复 | 修改后的文本 + Diff 对比 |
| 适用场景 | 学习辅导 | 内容创作 |

这两个项目互补：AI Coach 帮你**学习知识**，AI Editor 帮你**输出知识**。

## B.2 技术架构

### 整体架构图

```
┌─────────────────────────────────────────────────────┐
│                    Vue.js 3 前端                     │
│            Vite + Tailwind CSS + Pinia               │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐ ┌───────┐ │
│  │ 章节管理  │  │ AI 编辑器 │  │ AI 写作  │ │ 对话  │ │
│  │ CRUD     │  │ Diff 对比 │  │ 生成内容 │ │ SSE  │ │
│  └────┬─────┘  └────┬─────┘  └────┬────┘ └───┬───┘ │
│       │             │             │           │      │
│  ┌────▼─────────────▼─────────────▼───────────▼───┐  │
│  │       Axios + Token 拦截器 + SSE 客户端         │  │
│  └────────────────────┬───────────────────────────┘  │
├───────────────────────┼──────────────────────────────┤
│                  FastAPI 后端                         │
│  ┌─────────┐  ┌───────▼───────┐  ┌───────────────┐  │
│  │Auth API │  │ Editor API    │  │  Writer API   │  │
│  │JWT+RBAC │  │ 7 种编辑操作  │  │  内容生成     │  │
│  └────┬────┘  └───────┬───────┘  └───────┬───────┘  │
│       │               │                  │           │
│  ┌────▼────┐  ┌───────▼───────┐  ┌──────▼────────┐  │
│  │  User   │  │ EditorAgent   │  │ WriterAgent   │  │
│  │  Model  │  │ + ChatAgent   │  │               │  │
│  └────┬────┘  └───────┬───────┘  └───────┬───────┘  │
│       │               │                  │           │
│  ┌────▼────┐  ┌───────▼───────┐  ┌──────▼────────┐  │
│  │ SQLite  │  │ OpenAI GPT-4o │  │ diff-match    │  │
│  │         │  │               │  │ -patch        │  │
│  └─────────┘  └───────────────┘  └───────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 三个 Agent 的分工

```
EditorAgent ─── 负责 7 种编辑操作（校对/润色/扩写/缩写/重构/翻译/审查）
WriterAgent ─── 负责从零生成书稿内容
ChatAgent   ─── 负责与作者讨论书稿、提供建议
```

这种设计体现了**单一职责原则**：每个 Agent 专注做一件事，做到极致。

## B.3 核心实现详解

### B.3.1 七种编辑操作的 Prompt 设计

这是本项目最核心的部分。7 种编辑操作，本质上是 7 套精心设计的 Prompt。

**设计方法论：**

每个 Prompt 都遵循相同的结构：

```
1. 角色定义（你是谁）
2. 任务清单（做什么，5 条具体指令）
3. 约束条件（不要做什么）
4. 输出格式（怎么输出）
5. 输入文本（原文）
```

**七种操作对比：**

| 操作 | 温度 | 核心指令 | 输出 |
|------|------|---------|------|
| 🔍 校对 | 0.3 | 修正错误，不改风格 | 修正后全文 |
| ✨ 润色 | 0.7 | 改善表达，保持原意 | 润色后全文 |
| 📝 扩写 | 0.7 | 补充细节和代码示例 | 扩写后全文 |
| 📐 缩写 | 0.7 | 精简 30-50%，保留核心 | 缩写后全文 |
| 🔄 重构 | 0.7 | 调整结构，添加过渡句 | 重构后全文 |
| 🌐 翻译 | 0.3 | 技术术语准确，保持格式 | 翻译后全文 |
| 📋 审查 | 0.7 | 从 5 个维度审查 | 建议列表 |

```{admonition} 温度参数的选择
:class: note
- **低温度（0.3）**：校对和翻译需要确定性，不能"创造性地"修改原文
- **高温度（0.7）**：润色、扩写等需要一定的创造性
- **不要用 0**：即使是校对，也需要一点灵活性来处理歧义
```

**校对 Prompt 示例：**

```python
PROOFREAD_PROMPT = """你是一位专业的中文校对编辑。请校对以下文本：

任务：
1. 修正错别字和错误用词
2. 修正语法错误
3. 修正标点符号错误
4. 保持原文风格和语气不变
5. 不要改变原文的意思和结构

只输出修正后的完整文本，不要添加任何解释。

原文：
{text}"""
```

**审查 Prompt 示例（最复杂的一个）：**

```python
REVIEW_PROMPT = """你是一位严格的技术书籍审稿人。请审查以下文本：

审查维度：
1. **技术准确性**：概念是否正确？代码示例是否可运行？
2. **逻辑连贯性**：段落之间是否有清晰的逻辑关系？
3. **可读性**：语言是否清晰？是否有难以理解的表述？
4. **完整性**：是否有遗漏的重要内容？
5. **一致性**：术语使用是否一致？风格是否统一？

请用以下格式输出：
## 总体评价
## 具体问题
1. 🔴 [必须修改] ...
2. ⚠️ [建议修改] ...
3. 💡 [可以改进] ...
## 修改建议"""
```

```{admonition} Prompt 设计的关键技巧
:class: tip
1. **明确的否定指令**："不要改变原文的意思"比"保持原意"更有效
2. **结构化输出**：用 emoji + 分级（🔴⚠️💡）让 AI 的输出更有层次
3. **五维审查框架**：给 AI 一个清晰的评估框架，避免泛泛而谈
4. **"只输出...不要添加"**：防止 AI 在修改后的文本前后加解释
```

### B.3.2 Diff 可视化：看清 AI 改了什么

编辑操作的一个关键问题是：**AI 到底改了什么？** 如果只给你一个修改后的版本，你很难判断哪些是好的修改、哪些是不必要的。

我们使用 `diff-match-patch` 库来生成可视化的差异对比：

```python
from diff_match_patch import diff_match_patch

class EditorAgent:
    def __init__(self):
        self.dmp = diff_match_patch()
    
    def edit(self, text, action):
        # 1. 调用 LLM 获取编辑结果
        edited = self.call_llm(text, action)
        
        # 2. 生成 diff
        diffs = self.dmp.diff_main(text, edited)
        self.dmp.diff_cleanupSemantic(diffs)  # 语义级别清理
        diff_html = self.dmp.diff_prettyHtml(diffs)
        
        # 3. 统计变化
        stats = {
            "original_chars": len(text),
            "edited_chars": len(edited),
            "change_ratio": abs(len(edited) - len(text)) / len(text) * 100,
        }
        
        return {"edited": edited, "diff_html": diff_html, "stats": stats}
```

前端用 CSS 渲染 diff：

```css
.diff-view ins { background: #d4edda; color: #155724; }  /* 新增：绿色 */
.diff-view del { background: #f8d7da; color: #721c24; }  /* 删除：红色 */
```

这样，用户可以清楚地看到：
- 🟢 绿色高亮 = AI 新增的内容
- 🔴 红色删除线 = AI 删除的内容
- 黑色 = 未修改的内容

### B.3.3 WriterAgent：AI 辅助写作

WriterAgent 可以根据主题和大纲生成完整的章节内容：

```python
class WriterAgent:
    def write(self, topic, outline, style, word_count, context):
        style_desc = {
            "technical": "专业严谨的技术写作风格",
            "casual": "轻松易读的风格，适合技术博客",
            "academic": "学术论文风格，严谨规范",
        }
        
        prompt = f"""你是一位资深的技术书籍作者。请撰写以下内容：
        
        主题：{topic}
        风格：{style_desc[style]}
        目标字数：约 {word_count} 字
        大纲：{outline}
        上下文：{context}
        
        要求：
        1. 使用 Markdown 格式
        2. 技术概念要有代码示例
        3. 适当使用列表、表格增强可读性
        """
```

**上下文注入**是一个重要的设计：通过传入前后章节的摘要，AI 可以保持内容的连贯性，避免重复或遗漏。

### B.3.4 多 Agent 协作模式

三个 Agent 虽然独立工作，但通过共享数据实现协作：

```
用户写初稿 → EditorAgent 校对 → EditorAgent 润色 → ChatAgent 讨论修改
     ↑                                                      │
     └──────────── 用户采纳修改，继续迭代 ←─────────────────┘
```

```
WriterAgent 生成初稿 → 保存为章节 → EditorAgent 审查 → 用户修改 → EditorAgent 校对
```

这种模式体现了本书第 21 章讨论的**流水线式多 Agent 协作**：每个 Agent 处理一个环节，输出传递给下一个。

## B.4 编辑器界面设计

### 双栏布局

编辑器采用经典的双栏布局：

```
┌──────────────────┬──────────────────┐
│   📝 编辑器       │   👁 预览/Diff    │
│   (Markdown)     │   (渲染结果)      │
│                  │                  │
│  用户在这里编辑   │  实时预览         │
│  支持选中部分文本  │  或查看 AI 修改   │
│                  │  的 Diff 对比     │
│                  │                  │
│  [💾 保存]       │  [✅ 应用修改]    │
└──────────────────┴──────────────────┘
```

**选中编辑**是一个贴心的功能：用户可以选中一段文本，只对选中部分执行编辑操作，而不是处理整篇文章。这在处理长文档时非常实用。

```javascript
// 获取用户选中的文本
const textarea = document.querySelector('textarea')
const selection = textarea.value.substring(
  textarea.selectionStart, 
  textarea.selectionEnd
)
// 如果没有选中，则处理全文
const text = selection || chapter.content
```

### 右侧面板的三种模式

| 模式 | 说明 | 何时显示 |
|------|------|---------|
| 👁 预览 | Markdown 实时渲染 | 默认 |
| 📊 Diff | 红绿对比显示修改 | 编辑操作后 |
| ✨ 编辑结果 | 修改后的完整文本 + 统计 | 编辑操作后 |

## B.5 项目运行指南

### 快速启动

```bash
# 1. 启动后端
cd examples/ai-editor/backend
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
# → http://localhost:8001 (API 文档: /docs)

# 2. 启动前端（新终端）
cd examples/ai-editor/frontend
npm install
npm run dev
# → http://localhost:5174
```

### Docker 一键部署

```bash
cd examples/ai-editor
cp backend/.env.example backend/.env
docker compose up -d
# → http://localhost:5174
```

### 使用流程

1. **注册登录** → 进入章节管理页面
2. **创建章节** → 输入章节号、标题和内容（Markdown）
3. **AI 编辑** → 点击编辑器顶部的操作按钮（校对/润色/扩写...）
4. **查看 Diff** → 在右侧面板查看 AI 的修改对比
5. **应用修改** → 满意则点击"应用修改"，不满意则忽略
6. **AI 写作** → 在写作页面输入主题和大纲，AI 生成内容
7. **编辑对话** → 与 AI 编辑讨论书稿结构和内容

## B.6 涉及的书中知识点

| 章节 | 知识点 | 在项目中的体现 |
|------|--------|---------------|
| 第 6 章 | Prompt Engineering | 7 种编辑操作的 Prompt 设计（角色/任务/约束/格式） |
| 第 6 章 | 温度参数 | 校对(0.3) vs 润色(0.7) 的温度选择 |
| 第 17 章 | Agent 架构 | EditorAgent / WriterAgent / ChatAgent 三个独立 Agent |
| 第 21 章 | 多 Agent 协作 | 流水线式协作：写作→审查→校对→润色 |
| 第 22 章 | Agent 评估 | Diff 可视化作为编辑质量的评估手段 |

## B.7 启发与思考

### 对读者的启发

1. **Prompt 就是产品**：7 种编辑操作的差异，100% 来自 Prompt 的不同。同一个 GPT-4o，通过不同的 Prompt 可以变成校对员、润色师、翻译官、审稿人。这就是 Prompt Engineering 的威力。

2. **温度是一个被低估的参数**：校对用 0.3（确定性高），润色用 0.7（创造性强）。很多开发者忽略了温度调优，但它对输出质量的影响巨大。

3. **Diff 是 AI 编辑的必需品**：如果 AI 直接给你一个修改后的版本，你无法判断它改了什么。Diff 可视化让用户有**审查和控制权**，这是人机协作的关键。

4. **多 Agent 不一定要复杂**：本项目的三个 Agent 没有用 LangGraph 或 CrewAI，就是简单的 Python 类。关键是**职责清晰**，而不是框架花哨。

5. **选中编辑是杀手级功能**：处理长文档时，全文编辑太慢且不精确。让用户选中一段文本单独处理，既快又准。

### 读者可以怎么做

```{admonition} 动手实践建议
:class: tip

**初级挑战：**
- 添加一个新的编辑操作（如"添加代码注释"或"生成摘要"）
- 在编辑历史中添加"撤销"功能（恢复到上一个版本）
- 支持导出为 Word 文档（提示：使用 `python-docx` 库）

**中级挑战：**
- 实现"批量编辑"：一键对所有章节执行校对
- 添加"风格一致性检查"：确保全书术语统一
- 实现编辑操作的 A/B 测试：同一段文本用不同 Prompt 编辑，让用户选择更好的

**高级挑战：**
- 将 EditorAgent 改造为 LangGraph 工作流：校对→润色→审查 自动串联
- 添加 RAG 支持：上传风格指南，让 AI 按照指南编辑
- 实现多人协作编辑（类似 Google Docs）
- 训练一个专门的编辑模型（Fine-tuning），替代通用 GPT-4o
```

### 两个项目的对比与互补

```{admonition} 从两个项目中学到的架构模式
:class: important

| 模式 | AI Coach | AI Editor | 适用场景 |
|------|----------|-----------|---------|
| RAG | ✅ 核心 | ❌ 未使用 | 需要基于特定知识回答 |
| 多模式 Agent | ✅ 1 个 Agent 3 种模式 | ✅ 3 个独立 Agent | 功能差异大时用多 Agent |
| SSE 流式 | ✅ 对话 | ✅ 对话 | 所有 AI 生成场景 |
| Diff 可视化 | ❌ 不需要 | ✅ 核心 | 需要展示修改的场景 |
| 数据驱动反馈 | ✅ 学习数据 | ❌ 未使用 | 需要个性化建议 |

选择哪种模式，取决于你的应用场景。没有银弹，只有合适的工具。
```

## B.8 项目文件清单

```
ai-editor/                         # 45 个文件
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
│       │   ├── config.py          # 配置
│       │   ├── database.py        # SQLAlchemy async
│       │   └── auth.py            # JWT 认证 + 角色授权
│       ├── models/
│       │   ├── schemas.py         # Pydantic（含 7 种 EditAction 枚举）
│       │   └── db_models.py       # ORM（User, Book, Chapter, EditHistory...）
│       ├── api/
│       │   ├── auth.py            # 注册/登录/刷新
│       │   ├── chapters.py        # 章节 CRUD
│       │   ├── editor.py          # AI 编辑（7 种操作 + 应用 + 历史）
│       │   ├── writer.py          # AI 写作
│       │   ├── chat.py            # 编辑对话（普通 + SSE）
│       │   └── health.py          # 健康检查
│       └── agents/
│           ├── editor_agent.py    # EditorAgent + WriterAgent
│           └── chat_agent.py      # ChatAgent（流式）
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
        │   ├── ChaptersView.vue   # 章节管理
        │   ├── EditorView.vue     # AI 编辑器（双栏 + Diff）
        │   ├── WriterView.vue     # AI 写作助手
        │   └── ChatView.vue       # 编辑对话（SSE）
        └── utils/
            ├── api.js             # Axios + Token + SSE
            └── markdown.js        # Markdown 渲染
```
