(chapter03)=
# 第三章：DevOps 与云原生 — 打破开发与运维的壁垒

```{mermaid}
mindmap
  root((DevOps与云原生))
    DevOps文化
      2009 DevOpsDays
      CALMS模型
      打破壁垒
    基础设施即代码
      Terraform
      Ansible
      Pulumi
    容器化
      Docker
      Kubernetes
      容器编排
    微服务
      服务拆分
      API网关
      分布式挑战
    GitOps
      Git为真实来源
      ArgoCD/Flux
      Platform Engineering
    可观测性
      Metrics
      Logs
      Traces
```

> "DevOps 不是一个团队、一个工具或一个职位，它是一种文化。" — Gene Kim

## 3.1 DevOps 的起源

2009 年，比利时根特市，Patrick Debois 组织了第一届 **DevOpsDays** 大会。这个名字来源于 "Development"（开发）和 "Operations"（运维）的组合，代表着一场打破两个阵营之间壁垒的运动。

### 开发与运维的传统矛盾

```
开发团队的目标：快速交付新功能
    "我们需要更快地发布！"
         ↕ 冲突 ↕
运维团队的目标：保持系统稳定
    "别动生产环境！"
```

这种矛盾导致了：
- 发布周期长（每月甚至每季度一次）
- "扔过墙"式的交接（开发写完代码扔给运维部署）
- 出了问题互相推诿
- 运维手动操作，容易出错

### DevOps 的核心理念（CALMS）

- **C**ulture（文化）：协作、信任、共担责任
- **A**utomation（自动化）：一切可自动化的都应该自动化
- **L**ean（精益）：消除浪费，持续改进
- **M**easurement（度量）：用数据驱动决策
- **S**haring（分享）：知识共享，打破信息孤岛

## 3.2 基础设施即代码（IaC）

### 从手动运维到代码化

传统运维靠的是"运维手册"和"SSH 上去改"，IaC 将基础设施的管理变成了编写代码：

```hcl
# Terraform 示例：在 AWS 上创建一个 EC2 实例
resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.medium"

  tags = {
    Name        = "web-server"
    Environment = "production"
    ManagedBy   = "terraform"
  }

  user_data = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y nginx
    systemctl start nginx
  EOF
}

resource "aws_security_group" "web_sg" {
  name = "web-server-sg"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

### 主流 IaC 工具对比

| 工具 | 类型 | 语言 | 特点 |
|------|------|------|------|
| **Terraform** | 声明式 | HCL | 多云支持，生态最丰富 |
| **Pulumi** | 命令式 | Python/Go/TS | 用编程语言写基础设施 |
| **Ansible** | 过程式 | YAML | 无 Agent，简单易用 |
| **CloudFormation** | 声明式 | JSON/YAML | AWS 原生 |

## 3.3 容器化革命

### Docker：一次构建，到处运行

2013 年，Docker 的发布彻底改变了软件的打包和分发方式：

```dockerfile
# 多阶段构建示例
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Docker 解决的核心问题：
- ✅ "在我机器上能跑" → 环境一致性
- ✅ 依赖隔离，不同应用互不干扰
- ✅ 秒级启动，轻量级虚拟化
- ✅ 镜像分层，高效分发

### Kubernetes：容器编排的事实标准

```yaml
# Kubernetes Deployment 示例
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web-app
        image: myregistry/web-app:v1.2.3
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "128Mi"
            cpu: "250m"
          limits:
            memory: "256Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: web-app-service
spec:
  selector:
    app: web-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Kubernetes 的核心能力：
- **自动扩缩容**：根据负载自动调整 Pod 数量
- **自愈**：容器崩溃自动重启
- **滚动更新**：零停机部署
- **服务发现**：内置 DNS 和负载均衡
- **配置管理**：ConfigMap 和 Secret

## 3.4 微服务架构

### 从单体到微服务

```
单体架构：                    微服务架构：
┌─────────────────┐          ┌──────┐ ┌──────┐ ┌──────┐
│   用户模块       │          │ 用户  │ │ 订单  │ │ 支付  │
│   订单模块       │    →     │ 服务  │ │ 服务  │ │ 服务  │
│   支付模块       │          └──┬───┘ └──┬───┘ └──┬───┘
│   通知模块       │             │        │        │
│   ─────────     │          ┌──┴────────┴────────┴──┐
│   共享数据库     │          │    消息队列 / API 网关   │
└─────────────────┘          └────────────────────────┘
```

### 微服务的挑战

微服务不是银弹，它带来了新的复杂性：

- **分布式系统的固有复杂性**：网络延迟、部分失败、数据一致性
- **运维复杂度**：几十上百个服务的部署、监控、调试
- **服务间通信**：同步（REST/gRPC）vs 异步（消息队列）
- **数据管理**：每个服务独立数据库，跨服务查询困难

## 3.5 GitOps 与 Platform Engineering

### GitOps

GitOps 将 Git 作为基础设施和应用配置的唯一真实来源：

```
开发者 → Git Push → Git Repository → ArgoCD/Flux → Kubernetes
                         ↑                              │
                         └──── 自动同步 ─────────────────┘
```

核心原则：
1. **声明式**：所有配置都是声明式的
2. **版本化**：所有变更都通过 Git 提交
3. **自动化**：变更自动应用到目标环境
4. **自愈**：系统自动纠正偏差

### Platform Engineering

2023-2024 年，Platform Engineering 成为新趋势：

```
开发者 → 内部开发者平台（IDP）→ 基础设施
         ┌─────────────────┐
         │  自助服务门户     │
         │  ├── 创建环境     │
         │  ├── 部署应用     │
         │  ├── 查看日志     │
         │  └── 管理配置     │
         └─────────────────┘
```

Platform Engineering 的理念是：**不要让每个开发者都成为 Kubernetes 专家，而是提供一个简单的平台让他们自助服务**。

## 3.6 可观测性三支柱

### Metrics、Logs、Traces

```
┌─────────────────────────────────────────┐
│              可观测性                     │
│                                         │
│  Metrics          Logs         Traces   │
│  (指标)           (日志)       (追踪)    │
│                                         │
│  Prometheus      ELK Stack    Jaeger    │
│  Grafana         Loki         Zipkin    │
│  Datadog         Fluentd      Tempo     │
│                                         │
│  "系统现在        "发生了       "请求经    │
│   怎么样？"       什么？"       过了哪     │
│                                里？"     │
└─────────────────────────────────────────┘
```

```python
# OpenTelemetry 示例：自动追踪
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 初始化追踪
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# 使用追踪
@tracer.start_as_current_span("process_order")
def process_order(order_id: str):
    span = trace.get_current_span()
    span.set_attribute("order.id", order_id)
    
    with tracer.start_as_current_span("validate_order"):
        validate(order_id)
    
    with tracer.start_as_current_span("charge_payment"):
        charge(order_id)
    
    with tracer.start_as_current_span("send_notification"):
        notify(order_id)
```

## 3.7 云原生时代的交付流水线

一个完整的云原生交付流水线：

```
代码提交 → 代码扫描 → 构建镜像 → 安全扫描 → 测试 → 部署到 Staging → 集成测试 → 部署到 Production
   │         │          │          │        │          │              │              │
   Git     SonarQube  Docker    Trivy    pytest    ArgoCD        Selenium       ArgoCD
           Semgrep    Buildkit  Snyk     Jest      Helm          Playwright     Canary
```

## 3.8 本章小结

DevOps 和云原生技术从根本上改变了软件的构建、交付和运维方式。从手动部署到自动化流水线，从物理服务器到容器编排，从运维手册到基础设施即代码——每一步都在提高效率、降低风险。

这些技术为 AI 时代的软件开发奠定了坚实的基础。AI Agent 的部署和运维同样需要容器化、编排、可观测性等能力。在后续章节中，我们将看到 AI 如何进一步改变这些实践。

```{admonition} 思考题
:class: hint
1. 你的团队目前的部署频率是多少？有哪些瓶颈？
2. 微服务是否适合所有项目？什么时候应该选择单体架构？
3. AI 能否帮助解决微服务带来的运维复杂性？
```
