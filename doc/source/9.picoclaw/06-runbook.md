# PicoClaw 运维手册

## 快速启动

### 从源码构建

```bash
# 克隆仓库
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw

# 安装依赖
make deps

# 构建
make build
# 输出: build/picoclaw

# 构建并安装到 $GOPATH/bin
make install
```

### 首次配置

```bash
# 交互式配置向导
picoclaw onboard

# 或手动创建配置
cp config/config.example.json ~/.picoclaw/config.json
vim ~/.picoclaw/config.json  # 填入 API Key 和渠道 Token
```

### 启动网关

```bash
# 前台运行
picoclaw gateway

# 查看状态
picoclaw status
```

### 交互式 Agent（CLI 模式）

```bash
picoclaw agent
```

## Docker 部署

```bash
# 首次运行（生成配置文件）
docker compose -f docker/docker-compose.yml --profile gateway up
# 容器会打印 "First-run setup complete." 然后退出

# 编辑配置
vim docker/data/config.json

# 启动
docker compose -f docker/docker-compose.yml --profile gateway up -d

# 查看日志
docker compose -f docker/docker-compose.yml logs -f picoclaw-gateway

# 停止
docker compose -f docker/docker-compose.yml --profile gateway down
```

### 完整版（含 Web UI）

```bash
docker compose -f docker/docker-compose.full.yml up -d
```

## 交叉编译

```bash
# 构建所有平台
make build-all

# 特定平台
make build-linux-arm       # ARM 32-bit
make build-linux-arm64     # ARM 64-bit (树莓派 4/5)
make build-linux-mipsle    # MIPS Little-Endian
make build-pi-zero         # 树莓派 Zero 2 W (ARM + ARM64)
```

## 嵌入式设备部署

### LicheeRV-Nano ($10 RISC-V)

```bash
# 在开发机上交叉编译
GOOS=linux GOARCH=riscv64 CGO_ENABLED=0 go build -o picoclaw-riscv64 ./cmd/picoclaw

# 传输到设备
scp picoclaw-riscv64 root@licheerv:/usr/local/bin/picoclaw

# 在设备上运行
ssh root@licheerv
picoclaw onboard
picoclaw gateway
```

### Android 手机（Termux）

```bash
# 在 Termux 中
wget https://github.com/sipeed/picoclaw/releases/download/v0.2.3/picoclaw-linux-arm64
chmod +x picoclaw-linux-arm64
pkg install proot
termux-chroot ./picoclaw-linux-arm64 onboard
```

## 测试

```bash
# 运行所有测试
make test

# 完整检查（依赖 + 格式 + vet + 测试）
make check

# Lint
make lint

# Docker 中测试
make docker-test
```

## 常见问题排查

### 1. Gateway 启动失败

```bash
# 检查配置文件语法
python3 -c "import json; json.load(open('~/.picoclaw/config.json'))"

# 检查端口占用
lsof -i :18790

# 查看详细日志
PICOCLAW_LOG_LEVEL=debug picoclaw gateway
```

### 2. Provider 连接失败

```bash
# 测试 API 连通性
curl -H "Authorization: Bearer sk-xxx" https://api.openai.com/v1/models

# 检查代理设置
echo $HTTP_PROXY $HTTPS_PROXY

# 使用 model 命令测试
picoclaw model list
```

### 3. 渠道连接问题

```bash
# Telegram: 检查 Bot Token
curl https://api.telegram.org/bot<TOKEN>/getMe

# 飞书: 检查 App ID 和 Secret
# 确保 Webhook URL 可达

# 查看渠道状态
picoclaw status
```

### 4. 内存占用过高

```bash
# 查看进程内存
ps aux | grep picoclaw

# 使用 pprof（如果启用了 health server）
go tool pprof http://localhost:18790/debug/pprof/heap
```

### 5. 配置迁移

```bash
# 从旧版本迁移配置
picoclaw migrate

# 手动迁移: providers → model_list
# 参考 docs/migration/model-list-migration.md
```

## 技能管理

```bash
# 列出已安装技能
picoclaw skills list

# 列出内置技能
picoclaw skills list-builtin

# 搜索技能
picoclaw skills search weather

# 安装技能
picoclaw skills install https://github.com/user/skill-repo

# 安装内置技能
picoclaw skills install-builtin weather

# 删除技能
picoclaw skills remove weather
```

## 定时任务管理

```bash
# 列出定时任务
picoclaw cron list

# 添加定时任务
picoclaw cron add --cron "0 9 * * *" --message "早安提醒"

# 禁用任务
picoclaw cron disable <job-id>

# 启用任务
picoclaw cron enable <job-id>

# 删除任务
picoclaw cron remove <job-id>
```

## 发布流程

```bash
# 使用 GoReleaser 发布
# 1. 创建 tag
git tag v0.2.4

# 2. 推送 tag（触发 GitHub Actions release workflow）
git push origin v0.2.4

# 或手动触发 workflow
gh workflow run release.yml -f tag=v0.2.4
```

GoReleaser 会自动：
- 交叉编译所有平台二进制
- 构建 Docker 镜像（推送到 GHCR + DockerHub）
- 创建 GitHub Release
- 上传到火山引擎 TOS（可选）
