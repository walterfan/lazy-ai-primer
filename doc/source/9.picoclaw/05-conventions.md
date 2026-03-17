# PicoClaw 工程规范

## 代码风格

### Go 规范

- 遵循标准 Go 代码风格（`gofmt`）
- 包名：小写单词，不用下划线
- 文件名：snake_case（如 `context_cache_test.go`）
- 平台特定代码：`_linux.go`, `_windows.go`, `_unix.go`, `_other.go` 后缀
- 构建标签：`//go:build !cgo` 等用于条件编译

### 项目约定

- `CGO_ENABLED=0`：默认禁用 CGO，确保静态链接
- 构建标签 `stdjson`：使用标准库 JSON（避免依赖 CGO JSON 库）
- 版本信息通过 `-ldflags` 注入：`config.Version`, `config.GitCommit`, `config.BuildTime`

## Lint 配置

使用 `golangci-lint`，配置在 `.golangci.yaml`：

```yaml
# 关键 linter
linters:
  enable:
    - errcheck
    - govet
    - ineffassign
    - staticcheck
    - unused
    - gosec
    - errorlint
    - bodyclose
```

运行方式：

```bash
make lint    # 运行 lint
make fix     # 自动修复
make fmt     # 格式化
make vet     # go vet
```

## 错误处理

### Provider 层

```go
// FailoverError 分类错误类型
type FailoverError struct {
    Reason   FailoverReason  // auth | rate_limit | billing | timeout | format | overloaded | unknown
    Provider string
    Model    string
    Status   int
    Wrapped  error
}
```

- `auth`：认证失败，切换 Provider
- `rate_limit`：限流，切换 Provider + 冷却
- `billing`：余额不足，切换 Provider
- `timeout`：超时，切换 Provider
- `format`：请求格式错误，**不重试**（问题在请求本身）
- `overloaded`：服务过载，切换 Provider + 冷却

### 渠道层

```go
// pkg/channels/errors.go
type ChannelError struct {
    Channel string
    Op      string
    Err     error
}
```

### 工具层

```go
// pkg/tools/result.go
type ToolResult struct {
    Content string
    IsError bool
}
```

## 日志规范

使用 `pkg/logger/` 包，基于标准库 `log/slog`：

```go
logger.Info("message processed", "channel", channel, "duration", elapsed)
logger.Error("provider failed", "error", err, "model", model)
```

## 配置管理

### 配置来源优先级

1. 命令行参数
2. 环境变量（`caarlos0/env` 库）
3. `config.json` 文件
4. 默认值（`pkg/config/defaults.go`）

### 配置迁移

`pkg/config/migration.go` 处理配置版本升级：

```go
// 自动检测旧格式并迁移
func MigrateConfig(cfg *Config) error {
    // v1 → v2: providers → model_list
    // v2 → v3: ...
}
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `PICOCLAW_CONFIG` | 配置文件路径 |
| `PICOCLAW_GATEWAY_HOST` | 网关监听地址 |
| `PICOCLAW_LOG_LEVEL` | 日志级别 |

## 渠道开发规范

新增渠道需要：

1. 在 `pkg/channels/` 下创建子目录
2. 实现 `channels.Channel` 接口
3. 在 `init.go` 中通过 `channels.Register()` 自注册
4. 在 `pkg/gateway/gateway.go` 中添加 `import _` 引入
5. 在 `config.example.json` 中添加配置模板
6. 在 `docs/channels/` 下添加配置文档

### Channel 接口

```go
type Channel interface {
    Start(ctx context.Context) error
    Stop() error
    SendMessage(msg OutboundMessage) error
    Name() string
}
```

## 工具开发规范

新增工具需要：

1. 在 `pkg/tools/` 下创建文件
2. 实现 `Tool` 接口
3. 在 `pkg/agent/instance.go` 的 `NewAgentInstance()` 中注册
4. 在 `config.example.json` 的 `tools` 中添加配置

## 测试规范

- 测试文件与源文件同目录：`xxx.go` → `xxx_test.go`
- 使用标准库 `testing` 包
- Mock 通过接口实现（如 `mock_provider_test.go`）
- 集成测试文件名：`xxx_integration_test.go`

```bash
make test     # 运行所有测试
make check    # deps + fmt + vet + test
```

## 禁止事项（Anti-Patterns）

- ❌ 不要使用 CGO（除非有明确的硬件需求且提供 fallback）
- ❌ 不要在 `pkg/` 中引入 `cmd/` 的依赖
- ❌ 不要硬编码 API Key（使用配置或 OAuth）
- ❌ 不要在渠道层直接调用 LLM（必须通过 MessageBus → AgentLoop）
- ❌ 不要忽略 `context.Context` 的取消信号
- ❌ 不要在工具执行中阻塞超过配置的超时时间
