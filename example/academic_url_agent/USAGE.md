# Academic URL Agent - Usage Guide

LangGraph ReAct Agent 用于抓取、翻译和总结学术网页内容。

## 快速开始

### 1. 安装依赖

```bash
cd example/academic_url_agent
poetry install
poetry run playwright install chromium
```

### 2. 配置环境变量

创建 `.env` 文件（参考 `.env.example`）：

```bash
# LLM Configuration
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://localhost:8000/v1
LLM_MODEL=gpt-4o-mini

# 如果使用自签名证书的本地 LLM，设置为 true
DISABLE_SSL_VERIFY=true
```

**说明：**
- `LLM_API_KEY`: 你的 API 密钥
- `LLM_BASE_URL`: OpenAI 兼容 API 的基础 URL
- `LLM_MODEL`: 模型名称
- `DISABLE_SSL_VERIFY`: 对于自签名证书设为 `true`

### 3. 运行

```bash
# 方式 1：使用 poetry run
poetry run python -m academic_url_agent.main "https://example.com/article"

# 方式 2：进入 poetry shell
poetry shell
python -m academic_url_agent.main "https://example.com/article"

# 方式 3：交互式输入 URL
poetry run python -m academic_url_agent.main
```

## 示例

```bash
poetry run python -m academic_url_agent.main \
  "https://lilianweng.github.io/posts/2023-06-23-agent/"
```

## 输出

程序会生成：

1. **控制台输出**：
   - 抓取进度
   - 翻译进度
   - 要点总结（完整）
   - 难点解释（完整）
   - PlantUML 思维导图脚本

2. **文件输出**：
   - `report.md` - **完整 Markdown 报告**（包含所有内容）
   - `mindmap.puml` - PlantUML 思维导图源码
   - `mindmap.png` - 思维导图图片（如果 PlantUML 在线服务可用）

### Markdown 报告内容

`report.md` 包含：

- ✅ 完整中文翻译
- ✅ 要点总结
- ✅ 难点解释
- ✅ PlantUML 思维导图脚本
- ✅ 在线思维导图查看链接
- ✅ 原文（折叠显示）
- ✅ 生成时间和原文链接

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户输入 URL                          │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  决策层 (Decision Layer) ── LangGraph ReAct 图               │
│                                                              │
│   ┌─────────┐   有 tool_calls   ┌────────────┐              │
│   │  agent  │ ───────────────▶ │   tools    │              │
│   │ (LLM)   │ ◀─────────────── │ (执行工具)  │              │
│   └────┬────┘   tool 结果返回    └────────────┘              │
│        │ 无 tool_calls                                       │
│        ▼                                                     │
│      END → 输出抓取到的正文                                    │
└─────────────────────┬───────────────────────────────────────┘
                      ▼  英文正文
┌─────────────────────────────────────────────────────────────┐
│  生成层 (Generation Layer) ── LLM Chains 管线                │
│                                                              │
│  ① 分块翻译  ──▶  ② 质量自检纠错  ──▶  ③ 要点总结            │
│       │                                       │              │
│       ▼                                       ▼              │
│  ④ 难点解释                            ⑤ PlantUML 思维导图    │
└─────────────────────────────────────────────────────────────┘
```

## 工具说明

### fetch_static
- 使用 HTTP GET 进行静态抓取
- 适用于服务端渲染的博客、文档等
- 速度快，但无法处理 JS 渲染的页面

### fetch_dynamic
- 使用 Playwright 启动无头浏览器
- 适用于 SPA、需要 JS 渲染的页面
- 速度较慢，仅在静态抓取失败时使用

## 本地 LLM 配置

如果你使用本地部署的 LLM（如 Ollama、vLLM、LocalAI 等），需要：

1. **设置环境变量**：
   ```bash
   LLM_BASE_URL=http://localhost:11434/v1  # Ollama
   # 或
   LLM_BASE_URL=https://localhost:8000/v1  # vLLM with SSL
   ```

2. **自签名证书处理**：
   ```bash
   DISABLE_SSL_VERIFY=true
   ```

3. **模型名称**：
   ```bash
   LLM_MODEL=qwen2.5:14b  # Ollama 格式
   # 或
   LLM_MODEL=Qwen/Qwen2.5-14B-Instruct  # vLLM 格式
   ```

## 故障排除

### SSL 证书错误
```
DISABLE_SSL_VERIFY=true
```

### 连接超时
检查 `LLM_BASE_URL` 是否正确，端口是否开放。

### Playwright 浏览器未安装
```bash
poetry run playwright install chromium
```

### 工具调用不工作
确保你的 LLM 支持 OpenAI 兼容的函数调用（Function Calling / Tool Calling）。

## 扩展

### 添加新工具

在 `tools.py` 中添加：

```python
@tool
def fetch_pdf(url: str) -> str:
    """从 PDF URL 提取文本内容。"""
    # 实现代码
    pass

ALL_TOOLS = [fetch_static, fetch_dynamic, fetch_pdf]
```

### 自定义 Prompt

修改 `graph.py` 中的 `AGENT_SYSTEM_PROMPT` 或 `pipeline.py` 中的各个 Prompt 模板。

### 添加新节点

在 `graph.py` 的 `build_fetch_graph()` 中添加节点和边。
