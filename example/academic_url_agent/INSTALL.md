# Installation Guide

## Prerequisites

- Python 3.11 or higher
- Poetry (Python dependency management tool)

## Installation Steps

### 1. Install Dependencies

```bash
cd example/academic_url_agent
poetry install
```

### 2. Install Playwright Browser

Playwright 需要下载浏览器引擎（用于动态网页抓取）：

```bash
poetry run playwright install chromium
```

### 3. Configure Environment Variables

复制示例配置文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置你的 LLM：

```bash
# 必填
LLM_API_KEY=your-api-key-here

# OpenAI 兼容 API 的基础 URL（可选，默认使用 OpenAI）
LLM_BASE_URL=http://localhost:11434/v1

# 模型名称
LLM_MODEL=gpt-4o-mini

# 如果使用自签名证书，设置为 true
DISABLE_SSL_VERIFY=true
```

### 4. Test Installation

```bash
poetry run python test_setup.py
```

预期输出：

```
============================================================
Academic URL Agent - 环境测试
============================================================
🔍 检查环境变量配置...

✓ LLM_API_KEY: 已设置 (sk-xxxxxxxx...)
✓ LLM_BASE_URL: http://localhost:11434/v1
✓ LLM_MODEL: gpt-4o-mini
✓ DISABLE_SSL_VERIFY: true

🔍 测试依赖导入...

✓ langchain_core
✓ langchain_openai
✓ langgraph
✓ requests
✓ beautifulsoup4
✓ readability-lxml

🔍 测试工具模块...

✓ 成功导入工具: ['fetch_static', 'fetch_dynamic']

🔍 测试 LangGraph 模块...

✓ 成功创建 ReAct 图

============================================================
测试结果
============================================================
环境变量: ✓ 通过
依赖导入: ✓ 通过
工具模块: ✓ 通过
LangGraph 图: ✓ 通过

🎉 所有测试通过！

运行示例:
  poetry run python -m academic_url_agent.main "https://example.com"
```

## Local LLM Setup Examples

### Ollama

```bash
# .env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:14b
DISABLE_SSL_VERIFY=false
```

### vLLM

```bash
# .env
LLM_API_KEY=your-token
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
DISABLE_SSL_VERIFY=false
```

### vLLM with Self-Signed SSL

```bash
# .env
LLM_API_KEY=your-token
LLM_BASE_URL=https://localhost:8443/v1
LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
DISABLE_SSL_VERIFY=true
```

### OpenAI

```bash
# .env
LLM_API_KEY=sk-...
# LLM_BASE_URL 留空使用默认
LLM_MODEL=gpt-4o-mini
DISABLE_SSL_VERIFY=false
```

## Troubleshooting

### Poetry not found

Install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### SSL Certificate Error

```bash
# 在 .env 中设置
DISABLE_SSL_VERIFY=true
```

### Playwright Browser Not Found

```bash
poetry run playwright install chromium
```

### ImportError

确保在 poetry 环境中运行：

```bash
poetry shell
python test_setup.py
```

或使用 `poetry run`:

```bash
poetry run python test_setup.py
```

## Next Steps

参考 `USAGE.md` 了解如何使用。
