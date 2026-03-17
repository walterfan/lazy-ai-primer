# PicoClaw 测试策略

## 测试金字塔

```{mermaid}
graph TB
    E2E["E2E 测试<br/>(手动 + CI)"]
    INT["集成测试<br/>(*_integration_test.go)"]
    UNIT["单元测试<br/>(*_test.go × 162)"]

    style E2E fill:#f96,stroke:#333
    style INT fill:#ff9,stroke:#333
    style UNIT fill:#9f9,stroke:#333

    E2E --> INT --> UNIT
```

| 层次 | 数量 | 范围 | 运行方式 |
|------|------|------|----------|
| 单元测试 | 162 文件 | 函数/方法级别 | `make test` |
| 集成测试 | ~5 文件 | Provider API 调用 | 需要 API Key |
| E2E 测试 | 手动 | 完整消息流程 | Docker Compose |

## 单元测试分布

| 包 | 测试文件数 | 关键测试 |
|----|-----------|----------|
| `pkg/agent/` | 7 | context_cache, loop, thinking, registry |
| `pkg/auth/` | 5 | oauth, pkce, token, store, anthropic_usage |
| `pkg/bus/` | 1 | bus 消息发布/订阅 |
| `pkg/channels/` | 6 | base, manager, split, errors, errutil, command |
| `pkg/config/` | 3 | config, migration, model_config, version |
| `pkg/credential/` | 3 | credential, keygen, store |
| `pkg/cron/` | 1 | service |
| `pkg/heartbeat/` | 1 | service |
| `pkg/identity/` | 1 | identity |
| `pkg/mcp/` | 1 | manager |
| `pkg/memory/` | 2 | jsonl, migration |
| `pkg/providers/` | ~15 | factory, fallback, cooldown, error_classifier, 各 provider |
| `pkg/routing/` | 4 | agent_id, route, router, session_key |
| `pkg/session/` | 2 | jsonl_backend, manager |
| `pkg/skills/` | 5 | clawhub_registry, installer, loader, registry, search_cache |
| `pkg/state/` | 1 | state |
| `pkg/tools/` | ~15 | shell, edit, filesystem, web, cron, spawn, message, mcp_tool 等 |
| `pkg/voice/` | 1 | transcriber |
| `cmd/picoclaw/` | ~20 | 各子命令的单元测试 |

## 测试运行

```bash
# 运行所有单元测试
make test

# 运行特定包的测试
go test ./pkg/agent/...
go test ./pkg/providers/...

# 运行特定测试
go test ./pkg/agent/ -run TestAgentLoop

# 带覆盖率
go test -cover ./pkg/...

# 生成覆盖率报告
go test -coverprofile=coverage.out ./pkg/...
go tool cover -html=coverage.out

# 竞态检测
go test -race ./pkg/...
```

## 集成测试

集成测试文件以 `_integration_test.go` 结尾，需要真实 API Key：

```bash
# Claude CLI Provider 集成测试
ANTHROPIC_API_KEY=sk-ant-xxx go test ./pkg/providers/ -run Integration -v

# Codex CLI Provider 集成测试
OPENAI_API_KEY=sk-xxx go test ./pkg/providers/ -run Integration -v
```

## Mock 策略

- **Provider Mock**：`pkg/agent/mock_provider_test.go` 提供 `MockProvider` 实现
- **接口驱动**：所有外部依赖通过接口抽象，便于 Mock
- **无外部 Mock 框架**：使用手写 Mock（Go 标准实践）

```go
// mock_provider_test.go
type MockProvider struct {
    ChatFunc func(ctx context.Context, messages []Message, ...) (*LLMResponse, error)
}

func (m *MockProvider) Chat(ctx context.Context, messages []Message, ...) (*LLMResponse, error) {
    return m.ChatFunc(ctx, messages, ...)
}
```

## CI/CD 测试

### PR 检查（`.github/workflows/pr.yml`）

```yaml
jobs:
  lint:     # golangci-lint
  test:     # go test ./...
  build:    # make build-all
```

### 主分支构建（`.github/workflows/build.yml`）

```yaml
jobs:
  build:    # make build-all（所有平台交叉编译）
```

### Nightly 构建（`.github/workflows/nightly.yml`）

- 每日 UTC 0:00 自动构建
- 发布 nightly 预发布版本

## 关键测试场景

### Agent 循环

- 正常消息处理（无工具调用）
- 单次工具调用
- 多次工具调用迭代
- 达到最大迭代限制
- 上下文摘要触发
- 子 Agent spawn 和完成

### Provider 故障转移

- 主 Provider 认证失败 → 切换备选
- 主 Provider 限流 → 切换备选 + 冷却
- 所有 Provider 失败 → 返回错误
- Format 错误 → 不重试

### 渠道消息

- 长消息自动分片
- 媒体附件处理
- 群组消息过滤（mention_only）
- allow_from 白名单

### 工具安全

- 文件系统沙箱（restrict_to_workspace）
- 路径白名单验证
- Shell 命令危险操作拦截
- 符号链接路径规范化
