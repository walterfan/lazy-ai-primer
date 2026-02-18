# Quick Start Guide

## 3 分钟快速开始

### 1️⃣ 安装依赖

```bash
cd example/academic_url_agent
poetry install
poetry run playwright install chromium
```

### 2️⃣ 配置环境

创建 `.env` 文件：

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env，填入你的配置
nano .env
```

**最小配置（使用 OpenAI）：**
```bash
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
```

**本地 LLM 配置（Ollama）：**
```bash
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:14b
```

**本地 LLM with 自签名证书：**
```bash
LLM_API_KEY=your-key
LLM_BASE_URL=https://localhost:8443/v1
LLM_MODEL=your-model
DISABLE_SSL_VERIFY=true
```

### 3️⃣ 测试安装

```bash
poetry run python test_setup.py
```

✅ 看到这个输出说明配置成功：
```
🎉 所有测试通过！
```

### 4️⃣ 运行示例

```bash
# 示例 1: 技术博客
poetry run python -m academic_url_agent.main \
  "https://lilianweng.github.io/posts/2023-06-23-agent/"

# 示例 2: RFC 文档（自动使用官方源）
poetry run python -m academic_url_agent.main \
  "http://www.rfcreader.com/#rfc7519"
```

## 运行流程

程序会执行以下步骤：

```
🤖 [决策层] LangGraph ReAct 图启动
  🧠 [agent 节点] 第 1 轮推理完成
     → 决定调用工具: fetch_static(...)
  👁️  [tools 节点] Observation: LLM Powered Autonomous Agents...
  🧠 [agent 节点] 第 2 轮推理完成
     → 得出结论，准备结束

✅ 成功获取正文（12836 字符）

🔄 [生成层] 开始处理...
① 翻译中...
  📝 翻译第 1/5 块...
  📝 翻译第 2/5 块...
  ...
② 总结要点...
③ 解释难点...
④ 生成思维导图...
⑤ 渲染思维导图 PNG...

✅ 思维导图已保存为: mindmap.png
✅ PlantUML 脚本已保存为: mindmap.puml
```

## 其他运行方式

### 方式 1: 命令行参数
```bash
poetry run python -m academic_url_agent.main "https://example.com/article"
```

### 方式 2: 交互式输入
```bash
poetry run python -m academic_url_agent.main
# 然后粘贴 URL
```

### 方式 3: 进入 Poetry Shell
```bash
poetry shell
python -m academic_url_agent.main "https://example.com/article"
```

## 输出文件

- **`report.md`** - 完整 Markdown 报告（包含翻译、总结、解释、思维导图）
- `mindmap.puml` - PlantUML 思维导图源码
- `mindmap.png` - 思维导图图片（如果 PlantUML 在线服务可用）

**重要**: 完整翻译内容保存在 `report.md` 中，控制台只显示前 2000 字符预览。

## 常见问题

### ❓ API Key 错误
```
✗ The api_key client option must be set
```

**解决：** 检查 `.env` 文件中的 `LLM_API_KEY`

### ❓ SSL 证书错误
```
✗ SSLError: certificate verify failed
```

**解决：** 在 `.env` 中设置 `DISABLE_SSL_VERIFY=true`

### ❓ 连接超时
```
✗ Connection timeout
```

**解决：** 检查 `LLM_BASE_URL` 是否正确，端口是否开放

### ❓ Playwright 错误
```
✗ Executable doesn't exist
```

**解决：** 运行 `poetry run playwright install chromium`

### ❓ 工具调用失败
```
✗ Tool calling not supported
```

**解决：** 确保你的 LLM 支持 OpenAI 兼容的函数调用（Function Calling）。

推荐的支持函数调用的模型：
- OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo
- Anthropic: claude-3-opus, claude-3-sonnet
- 本地: Qwen2.5, Yi, DeepSeek

## 下一步

- 📖 阅读 [USAGE.md](USAGE.md) 了解详细用法
- 🔧 阅读 [INSTALL.md](INSTALL.md) 了解详细安装说明
- 📚 阅读 [README.md](README.md) 了解架构原理

## 支持的 URL 类型

✅ 静态网页（博客、文档等）
✅ 动态网页（SPA、JS 渲染）
✅ 学术论文（arXiv、研究博客）
✅ 技术文章（Medium、Dev.to 等）

❌ 需要登录的页面
❌ 付费墙内容
❌ 严格反爬的网站
