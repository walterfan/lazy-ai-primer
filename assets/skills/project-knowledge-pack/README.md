# Project Knowledge Pack (PKP) Skill

让 AI 真正理解并参与软件项目开发的完整工具包。

## 概述

Project Knowledge Pack（PKP）是一套系统化的方法论和工具，用于创建 AI 可读的项目文档。它使 AI 助手能够像人类工程师一样：

- **定位代码**：快速找到代码、配置和脚本
- **理解架构**：掌握架构、依赖、数据流和业务逻辑
- **执行开发**：搭建环境、运行测试、修改代码、创建 PR
- **验证质量**：设计测试、定位 bug、执行回归测试

## 快速开始

### 1. 初始化项目知识包

```
/pkp-init           # 标准 Markdown 格式
/pkp-init --sphinx  # 支持 Sphinx + MyST 构建
```

这将在项目根目录创建 `man/` 目录结构和模板文件。

如果使用 `--sphinx` 选项，还会创建 Sphinx 配置文件，支持：
- 使用 Sphinx + MyST 构建精美的 HTML 文档
- 生成 PDF、ePub 等多种格式
- 支持 MyST 增强语法（admonitions、directives、cross-references）
- 自动重载开发服务器

### 2. 生成核心文档

```
/pkp-overview     # 生成项目概览
/pkp-repo-map     # 生成仓库地图
```

这两个文档是基础，是 AI 理解项目的起点。

### 3. 完善其他文档

根据需要生成其他文档：

```
/pkp-architecture         # 架构文档
/pkp-workflow user-login  # 业务流程文档
/pkp-data-api            # 数据模型和 API
/pkp-conventions         # 工程规范
/pkp-runbook             # 运维手册
/pkp-testing             # 测试策略
```

### 4. 记录重要决策

```
/pkp-adr "使用 PostgreSQL 而不是 MongoDB"
```

### 5. 管理变更提案

```
/pkp-change add-payment-gateway
```

### 6. 让 AI 理解项目

使用分三轮的方式让 AI 理解项目：

```
/pkp-feed-ai 1  # 第一轮：建立地图
/pkp-feed-ai 2  # 第二轮：深入业务流程
/pkp-feed-ai 3  # 第三轮：模块验收
```

### 7. 验证完整性

```
/pkp-verify
```

## 目录结构

### 标准 Markdown 结构

```
man/
├── 00-overview.md          # 项目总览
├── 01-repo-map.md          # 仓库地图
├── 02-architecture.md      # 架构文档
├── 03-workflows.md         # 业务流程
├── 04-data-and-api.md      # 数据模型和 API
├── 05-conventions.md       # 工程规范
├── 06-runbook.md           # 运维手册
├── 07-testing.md           # 测试策略
├── adr/                    # 架构决策记录
│   ├── README.md
│   ├── template.md
│   └── 0001-xxx.md
└── changes/                # 变更提案
    ├── README.md
    ├── _template/
    └── add-xxx/
        ├── proposal.md
        ├── design.md
        ├── tasks.md
        └── specs/
```

### Sphinx + MyST 结构（使用 --sphinx）

在标准结构基础上增加：

```
man/
├── index.md                # Sphinx 文档入口（含 toctree）
├── ai-guide.md             # AI 使用指南
├── conf.py                 # Sphinx 配置
├── requirements.txt        # Python 依赖
├── Makefile                # Unix/macOS 构建脚本
├── make.bat                # Windows 构建脚本
├── _static/                # 静态资源
│   └── custom.css          # 自定义样式
├── _build/                 # 构建输出（自动生成）
│   └── html/
│       └── index.html
├── adr/
│   └── index.md            # ADR 索引（含 toctree）
└── changes/
    └── index.md            # Changes 索引（含 toctree）
```

## 核心命令

### 初始化与生成

- `/pkp-init [project-root] [--sphinx]` - 初始化知识包结构（可选 Sphinx 支持）
- `/pkp-overview [project-root]` - 生成项目概览
- `/pkp-repo-map [project-root]` - 生成仓库地图
- `/pkp-architecture [project-root]` - 生成架构文档
- `/pkp-workflow <name> [project-root]` - 生成业务流程文档
- `/pkp-data-api [project-root]` - 生成数据模型和 API 文档
- `/pkp-conventions [project-root]` - 生成工程规范文档
- `/pkp-runbook [project-root]` - 生成运维手册
- `/pkp-testing [project-root]` - 生成测试策略文档

### 决策与变更管理

- `/pkp-adr <title>` - 创建架构决策记录
- `/pkp-change <change-id> [type]` - 创建变更提案

### AI 理解与验证

- `/pkp-feed-ai [round]` - 生成 AI 理解项目的提示词（分三轮）
- `/pkp-verify [project-root]` - 验证知识包的完整性
- `/pkp-help` - 显示帮助信息

### Sphinx 构建（需要 --sphinx 初始化）

- `/pkp-build` - 构建 HTML 文档
- `/pkp-build html` - 构建 HTML 文档
- `/pkp-build pdf` - 构建 PDF 文档
- `/pkp-build serve` - 构建并启动自动重载服务器

## 工作流程

### 新项目启动

1. 初始化：
   ```bash
   /pkp-init              # 标准版
   /pkp-init --sphinx     # Sphinx 版（推荐）
   ```

2. 生成基础文档：
   ```bash
   /pkp-overview
   /pkp-repo-map
   ```

3. 补充架构和流程：
   ```bash
   /pkp-architecture
   /pkp-workflow user-login
   ```

4. （可选）构建 Sphinx 文档：
   ```bash
   cd man/
   pip install -r requirements.txt
   make html        # 构建 HTML
   make serve       # 构建并启动开发服务器
   ```

5. 让 AI 理解：
   ```bash
   /pkp-feed-ai 1
   ```

### 日常开发

1. 记录决策：`/pkp-adr "决策标题"`
2. 提交变更提案：`/pkp-change change-id`
3. 更新文档：修改相应的 md 文件
4. 验证完整性：`/pkp-verify`

### AI 协作

1. 准备上下文：选择相关文档
2. 使用预设提示词：`/pkp-feed-ai [round]`
3. 让 AI 输出：理解总结、改进建议、具体改动
4. 迭代优化：根据 AI 反馈完善文档

## 图表支持

PKP 完全支持丰富的 UML 图表和示意图：

### 支持的图表类型

**Mermaid（优先推荐）**：
- ✅ 流程图（Flowchart）
- ✅ 时序图（Sequence Diagram）
- ✅ 类图（Class Diagram）
- ✅ 状态图（State Diagram）
- ✅ 实体关系图（ER Diagram）
- ✅ 甘特图（Gantt Chart）
- ✅ 饼图（Pie Chart）
- ✅ Git 图（Git Graph）
- ✅ 用户旅程图（Journey Diagram）

**PlantUML（备用）**：
- ✅ 组件图（Component Diagram）
- ✅ 部署图（Deployment Diagram）
- ✅ 活动图（Activity Diagram）
- ✅ 用例图（Use Case Diagram）

### 使用示例

在文档中直接使用 Mermaid 代码块：

````markdown
```{mermaid}
sequenceDiagram
    participant User
    participant API
    participant DB

    User->>API: Login Request
    API->>DB: Verify Credentials
    DB-->>API: User Data
    API-->>User: Access Token
```
````

参见完整的[图表指南](templates/docs/diagrams-guide.md)。

## 方法论

PKP 基于以下核心原则：

### 1. 四类能力

AI 要像工程师一样工作，需要四类能力：

- **定位**：知道代码在哪里
- **理解**：懂架构和业务逻辑
- **执行**：能改代码、跑测试
- **验证**：能设计测试、定位问题

### 2. 分层喂料

不要一次把所有代码给 AI，而是分三轮：

1. **第一轮**：建立地图（overview + repo-map + 架构）
2. **第二轮**：深入关键业务链路（workflows + 代码）
3. **第三轮**：逐个模块验收（模块 + 测试）

### 3. 工程化文档

文档要：

- **结构化**：清晰的章节和格式
- **可链接**：引用具体代码位置
- **可执行**：包含可运行的命令
- **活文档**：随代码一起更新

### 4. 决策可追溯

通过 ADR 记录关键决策：

- 为什么这样设计
- 考虑了哪些方案
- 做了什么取舍
- 有什么后果

## 最佳实践

### 文档编写

1. **先易后难**：先写 overview 和 repo-map
2. **具体引用**：使用 `file:line` 格式引用代码
3. **记录原因**：不只写"是什么"，更要写"为什么"
4. **包含示例**：提供可运行的命令和代码示例

### 变更管理

1. **小步提交**：每次变更要小、可测试、可回滚
2. **文档先行**：先写 proposal 和 design，再写代码
3. **同步更新**：代码和文档在同一个 PR 中更新
4. **版本控制**：所有文档都纳入 git 管理

### AI 协作

1. **渐进式**：不要一次给所有代码
2. **验证式**：让 AI 输出理解，验证是否正确
3. **任务式**：给 AI 明确的任务和验收标准
4. **迭代式**：根据 AI 反馈不断完善文档

## 工具集成

PKP 可以与以下工具集成：

- **OpenSpec**: 变更管理工作流
- **SpecKit**: 软件项目规划
- **Living Documentation**: 从代码自动生成文档
- **TDD/MDD**: 测试驱动和指标驱动开发

## 适用场景

### 适合使用 PKP 的场景

- ✅ 希望 AI 长期参与开发、做 Code Review、接小需求
- ✅ 中大型项目，多模块/多服务
- ✅ 需要频繁 onboard 新人
- ✅ 有技术债需要梳理和重构

### 不适合的场景

- ❌ 一次性问一句就跑的临时问题
- ❌ 没有文档、没有测试、环境都拉不起来的"祖传仓库"
- ❌ 极小的个人项目（不到 1000 行代码）

## 参考资料

- **原始文章**：[如何让 AI 真正"懂"你的项目？](../../content/journal/journal_20260216_ai-project-knowledge-pack.md)
- **OpenSpec**：https://github.com/jpoehnelt/openspec
- **ADR**：https://adr.github.io/
- **DevContainer**：https://containers.dev/

## 贡献

欢迎提出改进建议！

- Issue：在项目仓库提交问题
- Pull Request：直接提交改进
- 讨论：在团队频道讨论使用心得

## 许可

本 skill 基于 CC-BY-NC-ND 4.0 许可发布。

---

**版本**：1.0
**作者**：Walter Fan
**最后更新**：2026-02-18
