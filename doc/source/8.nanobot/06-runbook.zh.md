# 运维手册

## 安装

### 从源码安装（推荐用于开发）

```bash
git clone https://github.com/HKUDS/nanobot.git
cd nanobot
pip install -e .
```

### 从 PyPI 安装

```bash
pip install nanobot-ai
```

### 使用 uv 安装

```bash
uv tool install nanobot-ai
```

### 安装 Matrix 支持

```bash
pip install nanobot-ai[matrix]
```

## 首次配置

```bash
# 1. 初始化配置和工作区
nanobot onboard

# 2. 编辑配置文件，添加 API 密钥
# macOS/Linux:
vim ~/.nanobot/config.json
# 或使用任意编辑器

# 3. 验证配置
nanobot status

# 4. 发送一条消息测试
nanobot agent -m "Hello!"
```

### 最小可用配置

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-your-key-here"
    }
  }
}
```

## 运行

### CLI 模式（交互式）

```bash
nanobot agent              # 交互式 REPL
nanobot agent -m "Hello"   # 单条消息
nanobot agent --no-markdown # 纯文本输出
nanobot agent --logs       # 显示调试日志
```

### Gateway 模式（守护进程）

```bash
nanobot gateway            # 前台运行
```

### Docker

```bash
# 构建镜像
docker build -t nanobot .

# 首次配置
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot onboard

# 运行 gateway
docker run -v ~/.nanobot:/root/.nanobot -p 18790:18790 nanobot gateway

# 执行单条命令
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot agent -m "Hello!"
```

### Docker Compose

```bash
docker compose run --rm nanobot-cli onboard   # 首次配置
docker compose up -d nanobot-gateway           # 启动 gateway
docker compose logs -f nanobot-gateway         # 查看日志
docker compose down                            # 停止服务
```

### systemd 用户服务（Linux）

创建 `~/.config/systemd/user/nanobot-gateway.service`：

```ini
[Unit]
Description=Nanobot Gateway
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/nanobot gateway
Restart=always
RestartSec=10
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=%h

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now nanobot-gateway
loginctl enable-linger $USER   # 注销后保持服务运行
```

## 常用操作

### 查看状态

```bash
nanobot status
```

显示内容包括：已配置的 provider、已启用的 channel、工作区路径、模型选择。

### 重置会话

删除对应的会话文件：

```bash
rm ~/.nanobot/workspace/sessions/telegram_12345.jsonl
```

或清除所有会话：

```bash
rm ~/.nanobot/workspace/sessions/*.jsonl
```

### 查看记忆

```bash
cat ~/.nanobot/workspace/memory/MEMORY.md    # 长期记忆
cat ~/.nanobot/workspace/memory/HISTORY.md   # 按时间排列的日志
```

### 编辑心跳任务

```bash
vim ~/.nanobot/workspace/HEARTBEAT.md
```

格式：
```markdown
## Periodic Tasks
- [ ] Check weather forecast and send a summary
- [ ] Scan inbox for urgent emails
```

### 添加/更新 Provider 凭据

编辑 `~/.nanobot/config.json` 并重启服务：

```bash
vim ~/.nanobot/config.json
# 如果以 systemd 服务运行：
systemctl --user restart nanobot-gateway
```

### OAuth Provider 登录

```bash
nanobot provider login openai-codex
nanobot provider login github-copilot
```

### WhatsApp 二维码配置

```bash
# 终端 1：启动 WhatsApp 桥接
nanobot channels login

# 终端 2：启动 gateway
nanobot gateway
```

## 故障排查

### 问题："No provider configured"

**症状**：`nanobot agent` 返回缺少 API 密钥的错误。

**解决方案**：
1. 运行 `nanobot status` 检查 provider 配置
2. 确保 `~/.nanobot/config.json` 中至少有一个 provider 配置了有效的 `apiKey`
3. 检查模型名称是否与已配置的 provider 匹配

### 问题："Access denied for sender"

**症状**：Bot 忽略来自聊天平台的消息。

**解决方案**：
1. 检查 channel 配置中的 `allowFrom` 列表
2. 将发送者的 ID 添加到 `allowFrom`
3. 使用 `["*"]` 允许所有发送者（仅用于测试）
4. 查看日志，确认被拒绝的实际发送者 ID

### 问题：Channel 连接失败

**症状**：日志中出现 "Channel not available"。

**解决方案**：
1. 检查对应的 channel SDK 是否已安装（部分为可选依赖）
2. 验证凭据（bot token、app secret 等）
3. 检查网络连接和代理设置
4. 对于 WebSocket 类型的 channel（飞书、钉钉、Slack）：确保防火墙允许出站 WebSocket 连接

### 问题：工具执行失败

**症状**：Agent 报告工具错误。

**解决方案**：
1. 如果启用了工作区限制（`restrictToWorkspace: true`），确保文件/命令在 `~/.nanobot/workspace/` 范围内
2. 对于 shell 命令：检查 `tools.exec.timeout`（默认：60 秒）
3. 对于 MCP 工具：检查 MCP server 配置中的 `toolTimeout`（默认：30 秒）
4. 检查文件权限

### 问题：记忆/会话数据损坏

**症状**：回复内容混乱、上下文重复，或出现 "tool_call_id not found" 错误。

**解决方案**：
1. 删除受影响的会话文件：`rm ~/.nanobot/workspace/sessions/{key}.jsonl`
2. 记忆文件可以手动编辑：`~/.nanobot/workspace/memory/MEMORY.md`
3. 重启 gateway

### 问题：内存占用过高

**症状**：Python 进程消耗过多 RAM。

**解决方案**：
1. 降低 `agents.defaults.memory_window`（默认：100），使其更积极地进行上下文压缩
2. 删除旧的会话文件
3. 减小 `agents.defaults.max_tool_iterations`（默认：40）

## 配置参考

### 环境变量

配置支持通过环境变量覆盖，前缀为 `NANOBOT_`，嵌套分隔符为 `__`：

```bash
export NANOBOT_AGENTS__DEFAULTS__MODEL="openai/gpt-4o"
export NANOBOT_PROVIDERS__OPENAI__API_KEY="sk-xxx"
```

### 关键配置项

| 配置项 | JSON 路径 | 默认值 | 说明 |
|--------|----------|--------|------|
| 模型 | `agents.defaults.model` | `anthropic/claude-opus-4-5` | LLM 模型标识符 |
| Provider | `agents.defaults.provider` | `auto` | Provider 名称或 `auto` |
| 最大 token 数 | `agents.defaults.maxTokens` | `8192` | 最大响应 token 数 |
| 温度 | `agents.defaults.temperature` | `0.1` | 采样温度 |
| 工作区 | `agents.defaults.workspace` | `~/.nanobot/workspace` | 工作目录 |
| 记忆窗口 | `agents.defaults.memoryWindow` | `100` | 触发上下文压缩的消息数 |
| 工具迭代次数 | `agents.defaults.maxToolIterations` | `40` | 最大工具调用轮数 |
| 工作区沙箱 | `tools.restrictToWorkspace` | `false` | 对所有工具启用沙箱限制 |
| Shell 超时 | `tools.exec.timeout` | `60` | Shell 命令超时时间（秒） |
| 心跳间隔 | `gateway.heartbeat.intervalS` | `1800` | 心跳唤醒间隔（秒） |
| 进度流式推送 | `channels.sendProgress` | `true` | 流式推送部分响应 |

## 监控

### 日志输出

- **CLI 模式**：日志输出到 stderr（使用 `--logs` 参数可见）
- **Gateway 模式**：日志输出到 stderr（可重定向到文件，或使用 journalctl 查看 systemd 日志）
- **Docker**：`docker compose logs -f nanobot-gateway`
- **systemd**：`journalctl --user -u nanobot-gateway -f`

### 健康指标

- Gateway 进程正在运行
- `nanobot status` 中 channel 显示为 "enabled"
- 心跳每 30 分钟触发一次（可在日志中确认）
- 会话文件持续更新（检查文件时间戳）

## 备份与恢复

### 需要备份的内容

| 路径 | 优先级 | 内容 |
|------|--------|------|
| `~/.nanobot/config.json` | 关键 | 所有凭据和配置 |
| `~/.nanobot/workspace/memory/` | 高 | 长期记忆和历史记录 |
| `~/.nanobot/workspace/sessions/` | 中 | 对话历史 |
| `~/.nanobot/workspace/skills/` | 中 | 用户自定义 skill |
| `~/.nanobot/workspace/HEARTBEAT.md` | 低 | 周期性任务定义 |

### 恢复步骤

1. 重新安装 nanobot
2. 恢复 `~/.nanobot/config.json`
3. 恢复 `~/.nanobot/workspace/` 目录
4. 运行 `nanobot status` 验证配置

## 相关文档

- [架构设计](02-architecture.md) — 系统设计
- [数据模型](04-data-and-api.md) — 存储格式
- [开发规范](05-conventions.md) — 开发约定

---

**最后更新**：2026-03-15
**版本**：1.0
