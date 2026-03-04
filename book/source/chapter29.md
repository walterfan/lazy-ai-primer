(chapter29)=
# 第二十九章：构建你的 AI 工具箱

```{mermaid}
mindmap
  root((构建AI工具箱))
    编码工具
      Cursor
      Claude Code
      Copilot
    设计工具
      v0
      Figma AI
    写作工具
      Claude
      Notion AI
    学习工具
      NotebookLM
      Perplexity
    自动化
      n8n
      Zapier AI
    Agent开发
      LangGraph
      MCP SDK
    成本控制
      模型分级
      缓存策略
      API优化
```

> "工匠的价值不在于他拥有多少工具，而在于他知道何时使用哪个工具。"

## 29.1 个人 AI 工具栈推荐（2026 版）

### 编码工具

| 工具 | 用途 | 月费 | 推荐指数 |
|------|------|------|---------|
| **Cursor** | 主力 AI IDE | $20 | ⭐⭐⭐⭐⭐ |
| **Claude Code** | 终端 AI 编程 | $20 (API) | ⭐⭐⭐⭐⭐ |
| **GitHub Copilot** | VS Code 补全 | $10 | ⭐⭐⭐⭐ |
| **Augment Code** | 企业级代码理解 | 联系销售 | ⭐⭐⭐⭐ |

### 设计工具

| 工具 | 用途 | 月费 |
|------|------|------|
| **v0** | UI 组件生成 | 免费/付费 |
| **Figma AI** | 设计稿生成 | Figma 订阅内 |
| **Midjourney** | 图片素材生成 | $10 |

### 写作与文档

| 工具 | 用途 | 月费 |
|------|------|------|
| **Claude** | 长文写作、分析 | $20 |
| **Notion AI** | 团队文档 | $10 |
| **Mintlify** | API 文档生成 | 免费/付费 |

### 学习与研究

| 工具 | 用途 | 月费 |
|------|------|------|
| **NotebookLM** | 文档分析、播客生成 | 免费 |
| **Perplexity** | AI 搜索引擎 | 免费/$20 |
| **Elicit** | 学术论文研究 | 免费/付费 |

### 自动化

| 工具 | 用途 | 月费 |
|------|------|------|
| **n8n** | 工作流自动化（可自托管） | 免费/付费 |
| **Zapier AI** | 无代码自动化 | $20+ |
| **Make** | 可视化自动化 | 免费/付费 |

### Agent 开发

| 工具 | 用途 | 月费 |
|------|------|------|
| **LangGraph** | Agent 框架 | 开源免费 |
| **LangSmith** | Agent 追踪调试 | 免费/付费 |
| **MCP SDK** | 工具协议开发 | 开源免费 |

## 29.2 如何评估和选择 AI 工具

### 评估框架（PRICE）

```
P — Performance（性能）：工具的输出质量如何？
R — Reliability（可靠性）：稳定性如何？宕机频率？
I — Integration（集成）：能否融入现有工作流？
C — Cost（成本）：总拥有成本（订阅 + API + 时间）
E — Evolution（演进）：团队是否活跃？更新频率？
```

### 选择决策矩阵

```python
# 工具评估打分示例
tools = {
    "Cursor": {"performance": 9, "reliability": 8, "integration": 9, "cost": 7, "evolution": 9},
    "Copilot": {"performance": 7, "reliability": 9, "integration": 10, "cost": 9, "evolution": 8},
    "Claude Code": {"performance": 9, "reliability": 8, "integration": 7, "cost": 7, "evolution": 9},
}

weights = {"performance": 0.3, "reliability": 0.2, "integration": 0.2, "cost": 0.15, "evolution": 0.15}

for tool, scores in tools.items():
    total = sum(scores[k] * weights[k] for k in weights)
    print(f"{tool}: {total:.1f}/10")
```

## 29.3 自建 AI 工具：打造个人 AI 助手

```python
"""个人 AI 助手框架（简化版）"""
import os
from openai import OpenAI
from datetime import datetime

class PersonalAssistant:
    def __init__(self):
        self.client = OpenAI()
        self.memory_file = "~/.assistant/memory.json"
        self.tools = self._register_tools()
    
    def _register_tools(self):
        return {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "web_search": self._web_search,
            "run_command": self._run_command,
            "manage_calendar": self._manage_calendar,
        }
    
    async def _read_file(self, path: str) -> str:
        with open(os.path.expanduser(path)) as f:
            return f.read()
    
    async def _write_file(self, path: str, content: str):
        with open(os.path.expanduser(path), 'w') as f:
            f.write(content)
        return f"Written to {path}"
    
    async def chat(self, message: str) -> str:
        """与助手对话"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": message}
            ],
            tools=self._tool_definitions()
        )
        return self._process_response(response)
    
    def _system_prompt(self):
        return f"""你是我的个人 AI 助手。
当前时间：{datetime.now().isoformat()}
你可以读写文件、搜索网络、执行命令、管理日历。
请简洁、准确地回答问题。"""

# 使用
assistant = PersonalAssistant()
# await assistant.chat("帮我整理今天的待办事项")
```

## 29.4 成本控制策略

### API 费用优化

```python
# 策略 1：模型分级
def choose_model(task_complexity: str) -> str:
    """根据任务复杂度选择模型"""
    model_map = {
        "simple": "gpt-4o-mini",      # $0.15/1M input
        "medium": "gpt-4o",            # $2.50/1M input
        "complex": "claude-3.5-sonnet", # $3.00/1M input
    }
    return model_map.get(task_complexity, "gpt-4o-mini")

# 策略 2：缓存
from functools import lru_cache
import hashlib

response_cache = {}

def cached_llm_call(prompt: str, model: str) -> str:
    cache_key = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
    if cache_key in response_cache:
        return response_cache[cache_key]
    response = call_llm(prompt, model)
    response_cache[cache_key] = response
    return response

# 策略 3：Prompt 压缩
def compress_context(context: str, max_tokens: int = 2000) -> str:
    """压缩上下文以减少 Token 消耗"""
    if estimate_tokens(context) <= max_tokens:
        return context
    # 使用小模型总结
    summary = call_llm(
        f"请用不超过{max_tokens}个token总结以下内容：\n{context}",
        model="gpt-4o-mini"
    )
    return summary
```

### 月度成本预算

```
个人开发者月度 AI 工具预算参考：
├── Cursor Pro: $20
├── Claude Pro: $20
├── API 调用: $30-50
├── 其他工具: $10-20
└── 总计: $80-110/月

团队（每人）：
├── Cursor Business: $40
├── GitHub Copilot Business: $19
├── LangSmith: $39
├── 其他: $20
└── 总计: ~$120/人/月
```

## 29.5 实战：搭建完整 AI 开发环境

```bash
# 1. 安装核心工具
brew install --cask cursor
npm install -g @anthropic-ai/claude-code

# 2. 配置 Cursor
# 安装扩展：Python, Ruff, GitLens, Docker

# 3. 配置 Claude Code
export ANTHROPIC_API_KEY="your-key"

# 4. 配置项目级 AI 规则
cat > .cursorrules << 'EOF'
You are working on a Python FastAPI project.
Use Python 3.12, async/await, type hints.
Follow Google Python Style Guide.
Write tests for all new code.
EOF

# 5. 安装 Agent 开发工具
pip install langchain langgraph langsmith
pip install mcp chromadb

# 6. 配置 LangSmith 追踪
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-key"

# 7. 配置 MCP
cat > ~/.cursor/mcp.json << 'EOF'
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/project"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_TOKEN": "your-token"}
    }
  }
}
EOF

echo "✅ AI 开发环境搭建完成！"
```

## 29.6 保持工具箱更新

```markdown
## 季度工具评估清单

### 每季度做一次：
- [ ] 检查现有工具是否有重大更新
- [ ] 评估是否有更好的替代品
- [ ] 审查 API 费用趋势
- [ ] 测试 1-2 个新工具
- [ ] 更新团队工具推荐列表

### 信息来源：
- Hacker News / Reddit r/LocalLLaMA
- AI 工具评测博客
- 同行推荐
- 官方更新日志
```

## 29.7 本章小结

构建个人 AI 工具箱不是一次性的事情，而是一个持续优化的过程。关键原则：

1. **少即是多**：不要追求工具数量，深度使用 2-3 个核心工具
2. **成本意识**：监控 API 费用，使用模型分级策略
3. **定期评估**：每季度审查工具效果
4. **团队统一**：核心工具团队统一，减少协作摩擦
5. **自建补充**：对于特定需求，自建工具可能更合适

```{admonition} 思考题
:class: hint
1. 你目前每月在 AI 工具上花费多少？值得吗？
2. 如果只能选择一个 AI 编程工具，你会选哪个？
3. 自建 AI 助手和使用现成工具，各有什么优缺点？
```
