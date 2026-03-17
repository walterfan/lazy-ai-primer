# PicoClaw 项目知识库

```{mermaid}
mindmap
  root((PicoClaw))
    核心特性
      超轻量 <10MB RAM
      单二进制部署
      多架构支持
      1 秒启动
    架构
      Agent Loop
      Message Bus
      Provider Chain
      Tool Registry
    渠道
      Telegram
      Discord
      飞书
      钉钉
      Slack
      QQ
      企业微信
      WhatsApp
      LINE
      Matrix
      IRC
    Provider
      OpenAI Compatible
      Anthropic Native
      Azure OpenAI
      Ollama / vLLM
      DeepSeek
      Gemini
    工具
      文件操作
      Shell 执行
      Web 搜索
      定时任务
      子 Agent
      MCP
      硬件 I2C/SPI
```

**PicoClaw** 是由 Sipeed 发起的开源项目，用 Go 语言实现的超轻量级个人 AI 助手。它可以在 $10 的嵌入式硬件上运行，内存占用 <10MB，启动时间 <1 秒。

- **GitHub**: [sipeed/picoclaw](https://github.com/sipeed/picoclaw)
- **官网**: [picoclaw.io](https://picoclaw.io)
- **许可证**: MIT
- **代码规模**: 421 Go 文件，~93,600 行，162 个测试文件

```{toctree}
:maxdepth: 2

00-overview
01-repo-map
02-architecture
03-workflows
04-data-and-api
05-conventions
06-runbook
07-testing
```
