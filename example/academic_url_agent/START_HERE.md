# 🚀 START HERE

## Academic URL Agent - LangGraph ReAct Agent

从零构建 AI Agent，用 LangGraph 实现 ReAct 决策图。

### 🎯 这是什么？

一个基于 **LangGraph** 的智能网页抓取和翻译 Agent：
- ✅ 智能选择抓取策略（静态 or 浏览器渲染）
- ✅ 英文 → 中文翻译 + 质量检查
- ✅ 自动生成要点总结、难点解释、思维导图
- ✅ 支持本地 LLM（Ollama、vLLM、自签名证书）

---

## 📖 文档导航

| 文档 | 用途 | 适合人群 |
|------|------|----------|
| **[QUICKSTART.md](QUICKSTART.md)** | 3 分钟快速开始 | 想立即运行的人 ⚡ |
| **[INSTALL.md](INSTALL.md)** | 详细安装说明 | 遇到安装问题的人 🔧 |
| **[USAGE.md](USAGE.md)** | 详细使用说明 | 想深入了解用法的人 📚 |
| **[README.md](README.md)** | 架构和原理教程 | 想理解实现原理的人 🧠 |
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | 项目总览 | 开发者和贡献者 👨‍💻 |

---

## ⚡ 最快开始方式

### 1. 安装
```bash
cd example/academic_url_agent
poetry install
poetry run playwright install chromium
```

### 2. 配置
创建 `.env` 文件：
```bash
LLM_API_KEY=your-api-key
# 如果使用本地 LLM，还需要：
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=qwen2.5:14b
# DISABLE_SSL_VERIFY=true  # 自签名证书
```

### 3. 测试
```bash
poetry run python test_setup.py
```

### 4. 运行
```bash
poetry run python -m academic_url_agent.main \
  "https://lilianweng.github.io/posts/2023-06-23-agent/"
```

✅ **看到这样的输出说明成功了：**
```
🤖 [决策层] LangGraph ReAct 图启动
  🧠 [agent 节点] 第 1 轮推理完成
     → 决定调用工具: fetch_static(...)
  👁️  [tools 节点] Observation: ...
✅ 成功获取正文（12836 字符）
```

---

## 🏗️ 架构速览

```
用户 URL
   ↓
┌─────────────────────────────┐
│ 决策层 (LangGraph ReAct)    │  ← Agent 智能决策
│   agent ⇄ tools             │
└──────────┬──────────────────┘
           ↓ 英文正文
┌─────────────────────────────┐
│ 生成层 (LLM Chains)         │  ← 确定性管线
│  翻译 → 总结 → 解释 → 导图   │
└─────────────────────────────┘
```

**关键创新：**
- **决策层**用图（StateGraph）替代手写循环
- **生成层**用链（LCEL）实现固定流程

---

## 🎓 学习路径

### 如果你是...

**🏃 赶时间的人**
1. 看 [QUICKSTART.md](QUICKSTART.md)
2. 运行 → 改配置 → 再运行
3. 完成 ✅

**🔧 遇到问题的人**
1. 看 [INSTALL.md](INSTALL.md) 的故障排除
2. 运行 `poetry run python test_setup.py` 诊断问题
3. 根据错误信息对应解决

**📚 想深入学习的人**
1. 阅读 [README.md](README.md) 理解原理
2. 阅读源码：`graph.py` → `tools.py` → `pipeline.py`
3. 尝试扩展：添加新工具、新节点

**👨‍💻 想贡献代码的人**
1. 看 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) 了解全局
2. Fork → 改代码 → 测试 → PR
3. 欢迎提交 Issue 和建议

---

## 🌟 关键特性

| 特性 | 说明 |
|------|------|
| **智能抓取** | 先静态抓取，失败自动切换浏览器渲染 |
| **本地 LLM** | 支持 Ollama、vLLM 等本地部署 |
| **自签名证书** | 一行配置解决 SSL 问题 |
| **延迟初始化** | 模块可以导入但不立即需要 API key |
| **可视化图** | ReAct 循环用图结构表达，易懂易改 |
| **全中文输出** | 翻译、总结、解释都是中文 |
| **Markdown 报告** | 自动生成完整报告，包含所有内容 |

---

## 🔥 使用场景

✅ 阅读英文技术博客/论文
✅ 快速理解学术文章
✅ **RFC 标准文档**（如 RFC 7519 JWT）
✅ 生成文章思维导图
✅ 学习 LangGraph 和 ReAct 架构

❌ 需要登录的页面
❌ 付费墙内容

---

## 🆘 快速排错

| 问题 | 解决方案 |
|------|----------|
| API Key 错误 | 检查 `.env` 中 `LLM_API_KEY` |
| SSL 证书错误 | 设置 `DISABLE_SSL_VERIFY=true` |
| 连接超时 | 检查 `LLM_BASE_URL` 和端口 |
| Playwright 错误 | 运行 `poetry run playwright install chromium` |
| 工具调用失败 | 确保 LLM 支持函数调用 |

详细故障排除见 [INSTALL.md](INSTALL.md)。

---

## 📦 项目结构

```
academic_url_agent/
├── START_HERE.md          ← 你在这里 👋
├── QUICKSTART.md          ← 3 分钟快速开始
├── INSTALL.md             ← 安装指南
├── USAGE.md               ← 使用说明
├── README.md              ← 架构教程（原教程）
├── PROJECT_SUMMARY.md     ← 项目总览
│
├── .env.example           ← 环境变量示例
├── pyproject.toml         ← Poetry 配置
├── test_setup.py          ← 测试脚本
│
└── src/academic_url_agent/
    ├── main.py            ← 入口
    ├── graph.py           ← LangGraph 决策层
    ├── pipeline.py        ← 生成层
    └── tools.py           ← 抓取工具
```

---

## 💡 下一步

完成快速开始后，你可以：

1. **尝试不同的 URL**
   ```bash
   poetry run python -m academic_url_agent.main "https://arxiv.org/abs/..."
   poetry run python -m academic_url_agent.main "https://your-blog.com/post"
   ```

2. **自定义 Prompt**
   - 编辑 `graph.py` 中的 `AGENT_SYSTEM_PROMPT`
   - 编辑 `pipeline.py` 中的 Prompt 模板

3. **添加新工具**
   - 在 `tools.py` 中添加 `@tool` 装饰的函数
   - 加入 `ALL_TOOLS` 列表

4. **扩展决策图**
   - 在 `graph.py` 中添加新节点
   - 添加条件边和循环边

5. **集成到你的项目**
   ```python
   from academic_url_agent.graph import fetch_graph
   result = fetch_graph.invoke(...)
   ```

---

## 🤝 贡献

欢迎：
- 🐛 提交 Bug 报告
- 💡 建议新功能
- 📝 改进文档
- 🔧 提交 PR

---

## 📞 获取帮助

1. 查看相关文档（见上方导航表）
2. 运行 `poetry run python test_setup.py` 诊断
3. 提交 Issue（附上错误信息和环境）

---

## ⭐ 快速链接

- [立即开始 ⚡](QUICKSTART.md)
- [遇到问题？🔧](INSTALL.md)
- [深入学习 📚](README.md)
- [开发者文档 👨‍💻](PROJECT_SUMMARY.md)

---

**祝你使用愉快！🎉**

如果觉得有帮助，别忘了给项目点个 ⭐
