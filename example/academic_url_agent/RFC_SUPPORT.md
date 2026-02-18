# RFC 文档支持

## 问题描述

使用 `http://www.rfcreader.com/#rfc7519` 这类 URL 时，程序可能抓取到的是 JavaScript 函数或页面框架代码，而不是 RFC 文档的实际正文内容。

**原因**：rfcreader.com 是一个单页应用（SPA），内容通过 JavaScript 动态加载，Readability 库可能无法正确提取主要内容。

## 解决方案

新增了 `fetch_rfc_direct` 工具，可以直接从 IETF 官方源抓取 RFC 文档的纯文本版本。

### 工作原理

1. 从 URL 中自动提取 RFC 编号
2. 直接访问 IETF 官方 RFC 仓库
3. 下载纯文本格式（.txt）的 RFC 文档

### 支持的 URL 格式

```python
# 以下所有格式都能正确识别 RFC 7519：
"http://www.rfcreader.com/#rfc7519"
"https://www.rfc-editor.org/rfc/rfc7519.txt"
"https://tools.ietf.org/html/rfc7519"
"rfc7519"
"RFC 7519"
"7519"
```

### 官方源

程序会将所有 RFC 请求转换为访问官方源：
```
https://www.rfc-editor.org/rfc/rfc{编号}.txt
```

例如：
- RFC 7519 → `https://www.rfc-editor.org/rfc/rfc7519.txt`
- RFC 2616 → `https://www.rfc-editor.org/rfc/rfc2616.txt`

## 使用方法

### 自动模式（推荐）

Agent 会自动检测 RFC URL 并选择正确的工具：

```bash
poetry run python -m academic_url_agent.main \
  "http://www.rfcreader.com/#rfc7519"
```

**工作流程**：
1. Agent 识别 URL 包含 "rfc" 和数字
2. 自动调用 `fetch_rfc_direct` 工具
3. 从 IETF 官方源抓取纯文本版本
4. 翻译并生成报告

### 直接指定 RFC 编号

如果你知道 RFC 编号，可以直接提供：

```bash
poetry run python -m academic_url_agent.main "rfc7519"
```

## Agent 策略更新

Agent 的系统提示已更新，现在会：

### RFC 文档优先使用 fetch_rfc_direct

```
如果 URL 包含 "rfc" 和数字：
1. 优先使用 fetch_rfc_direct
2. 自动从 URL 提取 RFC 编号
3. 从 IETF 官方源抓取
```

### 普通网页使用原有策略

```
1. 先用 fetch_static（静态抓取）
2. 失败后用 fetch_dynamic（浏览器渲染）
3. 检测 JavaScript 代码并给出警告
```

## 示例

### 示例 1: RFC 7519 (JSON Web Token)

```bash
poetry run python -m academic_url_agent.main \
  "http://www.rfcreader.com/#rfc7519"
```

**输出**：
```
🤖 [决策层] LangGraph ReAct 图启动

  🧠 [agent 节点] 第 1 轮推理完成
     → 决定调用工具: fetch_rfc_direct({"url_or_number": "http://..."})
  📡 从 IETF 官方源抓取 RFC 7519: https://www.rfc-editor.org/rfc/rfc7519.txt
  ✅ 成功抓取 RFC 7519（63039 字符）
  👁️  [tools 节点] Observation: Internet Engineering Task Force...

✅ 成功获取正文（20000 字符）

🔄 [生成层] 开始处理...
① 翻译中...
...
```

### 示例 2: RFC 2616 (HTTP/1.1)

```bash
poetry run python -m academic_url_agent.main "rfc2616"
```

### 示例 3: RFC 9110 (HTTP Semantics)

```bash
poetry run python -m academic_url_agent.main \
  "https://www.rfc-editor.org/rfc/rfc9110.txt"
```

## 技术细节

### 工具定义

```python
@tool
def fetch_rfc_direct(url_or_number: str) -> str:
    """直接从 IETF 官方源抓取 RFC 文档纯文本版本。

    参数:
        url_or_number: 可以是完整 URL 或 RFC 编号

    返回:
        RFC 文档的纯文本内容（前 20000 字符）
    """
```

### RFC 编号提取

```python
def extract_rfc_number(url: str) -> str:
    """从 URL 中提取 RFC 编号。

    支持的格式：
    - http://www.rfcreader.com/#rfc7519 → "7519"
    - https://www.rfc-editor.org/rfc/rfc7519.txt → "7519"
    - https://tools.ietf.org/html/rfc7519 → "7519"
    """
    match = re.search(r'rfc[\s#/]*(\d+)', url.lower())
    if match:
        return match.group(1)
    return ""
```

### JavaScript 代码检测

`fetch_dynamic` 工具现在会检测是否抓取到 JavaScript 代码：

```python
# 检查是否抓取到的主要是 JavaScript 代码
if "function(" in text or "=>{" in text or text.count("{") > len(text) / 50:
    return (
        f"[WARN] 检测到大量 JavaScript 代码，"
        "可能未正确提取正文。建议尝试 fetch_rfc_direct 工具。"
    )
```

## 测试

### 测试 RFC 功能

```bash
poetry run python test_rfc.py
```

**预期输出**：
```
✅ RFC 编号提取: 通过
✅ RFC 抓取: 通过
🎉 所有测试通过！
```

### 完整集成测试

```bash
# 测试 RFC 7519 (JWT)
poetry run python -m academic_url_agent.main \
  "http://www.rfcreader.com/#rfc7519"

# 检查 report.md
head -n 30 report.md
```

## 已知限制

### 1. RFC 文档长度限制

- **问题**: RFC 文档可能非常长（50000+ 字符）
- **解决**: 目前只返回前 20000 字符
- **建议**: 如果需要完整文档，考虑增加限制或分段处理

### 2. 非 RFC 文档

- **问题**: 该工具仅适用于 IETF RFC 文档
- **不适用**: 其他标准文档（如 W3C、IEEE）
- **建议**: 为其他标准文档类型添加专门的工具

### 3. 网络依赖

- **问题**: 需要访问 rfc-editor.org
- **解决**: 确保网络连接正常
- **超时**: 设置为 30 秒

## 故障排除

### 问题 1: 无法提取 RFC 编号

```
[ERROR] 无法从输入中提取 RFC 编号
```

**解决**：
- 确保 URL 包含 "rfc" 和数字
- 尝试直接提供 RFC 编号（如 "7519"）
- 检查 URL 格式是否正确

### 问题 2: 抓取超时

```
[ERROR] RFC 直接抓取失败: timeout
```

**解决**：
- 检查网络连接
- 尝试访问 https://www.rfc-editor.org 确认可达
- 稍后重试

### 问题 3: 找不到 RFC

```
[ERROR] RFC 直接抓取失败: 404
```

**解决**：
- 确认 RFC 编号存在
- 访问 https://www.rfc-editor.org/rfc-index.html 查看有效的 RFC

## 更新记录

### v0.2.1 - 2026-02-15

- ✅ 新增 `fetch_rfc_direct` 工具
- ✅ 新增 `extract_rfc_number` 辅助函数
- ✅ 更新 Agent 系统提示支持 RFC 优先策略
- ✅ `fetch_dynamic` 增加 JavaScript 代码检测
- ✅ 添加 RFC 抓取测试 (`test_rfc.py`)

## 相关资源

- **IETF RFC Editor**: https://www.rfc-editor.org/
- **RFC Index**: https://www.rfc-editor.org/rfc-index.html
- **RFC Search**: https://www.rfc-editor.org/search/rfc_search.php

## 贡献

欢迎提交其他文档类型的支持请求：
- W3C 规范
- IEEE 标准
- arXiv 论文
- 等等...
