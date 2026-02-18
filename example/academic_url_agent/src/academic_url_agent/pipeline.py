"""pipeline.py — 生成层：翻译 → 自检 → 总结 → 难点 → 思维导图。

与上一版完全相同——生成层不需要 Agent 决策，是确定性管线。
"""

import os
import zlib
import requests as http_requests
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def create_llm():
    """创建 LLM 实例，支持自签名证书的本地部署。"""
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    disable_ssl = os.getenv("DISABLE_SSL_VERIFY", "false").lower() == "true"

    # 配置 httpx 客户端以支持自签名证书
    import httpx
    http_client = httpx.Client(verify=not disable_ssl) if disable_ssl else None
    http_async_client = httpx.AsyncClient(verify=not disable_ssl) if disable_ssl else None

    kwargs = {
        "model": model,
        "temperature": 0.3,
    }

    if api_key:
        kwargs["openai_api_key"] = api_key

    if base_url:
        kwargs["openai_api_base"] = base_url

    if http_client:
        kwargs["http_client"] = http_client

    if http_async_client:
        kwargs["http_async_client"] = http_async_client

    return ChatOpenAI(**kwargs)


# 延迟初始化 LLM，避免在导入时就要求 API key
_llm = None


def get_llm():
    """获取 LLM 实例（延迟初始化）"""
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm


parser = StrOutputParser()

# ── Prompt 模板 ─────────────────────────────────────────────

TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是专业英中翻译。保持段落结构，专业术语在括号中保留英文原文。"),
    ("human", "请将以下英文翻译成流畅准确的中文：\n\n{text}"),
])

QUALITY_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是翻译质量审核专家。对比原文和译文，检查：漏译、误译、术语准确性、通顺度。"
     "如有问题就修正后返回完整译文；如无问题就原样返回译文。只返回最终译文，不要解释。"),
    ("human", "原文：\n{original}\n\n译文：\n{translation}"),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位善于提炼要点的阅读助手。"),
    ("human", "请用中文总结以下文章的 5-10 个关键要点，用编号列表：\n\n{text}"),
])

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位耐心的技术导师，擅长用简单的类比解释复杂概念。"),
    ("human",
     "以下是一篇技术文章的翻译。请识别其中 3-5 个最难理解的概念或术语，"
     "用简单易懂的中文逐一解释（可用类比）：\n\n{text}"),
])

MINDMAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是信息架构专家。根据内容生成 PlantUML 思维导图脚本。"
     "格式要求：\n"
     "- 用 @startmindmap 和 @endmindmap 包裹\n"
     "- 用 * 表示层级（* 一级, ** 二级, *** 三级）\n"
     "- 节点文字简洁，中文\n"
     "- 只输出 PlantUML 代码，不要其他文字"),
    ("human", "请根据以下文章要点生成思维导图：\n\n{text}"),
])

# ── LCEL Chains ─────────────────────────────────────────────
# 链在第一次使用时动态创建，避免在导入时就需要 API key

def get_translate_chain():
    return TRANSLATE_PROMPT | get_llm() | parser


def get_quality_chain():
    return QUALITY_CHECK_PROMPT | get_llm() | parser


def get_summary_chain():
    return SUMMARY_PROMPT | get_llm() | parser


def get_explain_chain():
    return EXPLAIN_PROMPT | get_llm() | parser


def get_mindmap_chain():
    return MINDMAP_PROMPT | get_llm() | parser


# ── 工具函数 ────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = p
        else:
            current = current + "\n\n" + p if current else p
    if current.strip():
        chunks.append(current.strip())
    return chunks


def translate_with_quality_check(text: str) -> str:
    chunks = chunk_text(text)
    results = []
    translate_chain = get_translate_chain()
    quality_chain = get_quality_chain()

    for i, chunk in enumerate(chunks, 1):
        print(f"  📝 翻译第 {i}/{len(chunks)} 块...")
        raw = translate_chain.invoke({"text": chunk})
        checked = quality_chain.invoke({
            "original": chunk,
            "translation": raw,
        })
        results.append(checked)
    return "\n\n".join(results)


# ── PlantUML 编码 & 渲染 ────────────────────────────────────

_PLANTUML_ALPHABET = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
)

def _encode_plantuml(text: str) -> str:
    data = zlib.compress(text.encode("utf-8"))[2:-4]
    res = []
    for i in range(0, len(data), 3):
        b = [data[i]] + [data[i+j] if i+j < len(data) else 0 for j in (1, 2)]
        res.append(_PLANTUML_ALPHABET[b[0] >> 2])
        res.append(_PLANTUML_ALPHABET[((b[0] & 0x3) << 4) | (b[1] >> 4)])
        res.append(_PLANTUML_ALPHABET[((b[1] & 0xF) << 2) | (b[2] >> 6)])
        res.append(_PLANTUML_ALPHABET[b[2] & 0x3F])
    return "".join(res)


def render_plantuml_png(script: str, output_path: str = "mindmap.png") -> str:
    encoded = _encode_plantuml(script)
    url = f"https://www.plantuml.com/plantuml/png/{encoded}"
    try:
        resp = http_requests.get(url, timeout=15)
        if resp.status_code == 200 and resp.headers.get(
            "content-type", ""
        ).startswith("image"):
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return output_path
    except Exception as e:
        print(f"  ⚠️  PNG 渲染失败: {e}")
    return ""


# ── 主管线 ──────────────────────────────────────────────────

def run_pipeline(original_text: str, url: str = "", output_path: str = "report.md") -> dict:
    print("\n🔄 [生成层] 开始处理...\n")

    print("① 翻译中...")
    translation = translate_with_quality_check(original_text)

    print("② 总结要点...")
    summary = get_summary_chain().invoke({"text": translation})

    print("③ 解释难点...")
    explanation = get_explain_chain().invoke({"text": translation})

    print("④ 生成思维导图...")
    mindmap_script = get_mindmap_chain().invoke({"text": summary})

    if "```" in mindmap_script:
        lines = mindmap_script.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        mindmap_script = "\n".join(lines)

    print("⑤ 渲染思维导图 PNG...")
    png_path = render_plantuml_png(mindmap_script)

    print("⑥ 保存 Markdown 文档...")
    markdown_path = save_markdown_report(
        url=url,
        original_text=original_text,
        translation=translation,
        summary=summary,
        explanation=explanation,
        mindmap_script=mindmap_script,
        mindmap_png=png_path,
        output_path=output_path,
    )

    with open("mindmap.puml", "w", encoding="utf-8") as f:
        f.write(mindmap_script)

    return {
        "translation": translation,
        "summary": summary,
        "explanation": explanation,
        "mindmap_script": mindmap_script,
        "mindmap_png": png_path,
        "markdown_report": markdown_path,
    }


def save_markdown_report(
    url: str,
    original_text: str,
    translation: str,
    summary: str,
    explanation: str,
    mindmap_script: str,
    mindmap_png: str,
    output_path: str = "report.md",
) -> str:
    """保存所有生成内容到 Markdown 文件。

    Args:
        url: 原始 URL
        original_text: 原始英文正文
        translation: 中文翻译
        summary: 要点总结
        explanation: 难点解释
        mindmap_script: PlantUML 思维导图脚本
        mindmap_png: 思维导图 PNG 路径
        output_path: 输出文件路径

    Returns:
        保存的文件路径
    """
    from datetime import datetime

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构建 Markdown 内容
    markdown_content = f"""# 文章分析报告

**生成时间**: {timestamp}

**原文链接**: {url if url else "N/A"}

---

## 目录

- [中文翻译](#中文翻译)
- [要点总结](#要点总结)
- [难点解释](#难点解释)
- [思维导图](#思维导图)
- [原文](#原文)

---

## 中文翻译

{translation}

---

## 要点总结

{summary}

---

## 难点解释

{explanation}

---

## 思维导图

### PlantUML 脚本

```plantuml
{mindmap_script}
```

### 在线查看

[在线渲染思维导图](https://www.plantuml.com/plantuml/uml/{_encode_plantuml(mindmap_script)})

"""

    # 如果有本地 PNG 图片，添加图片引用
    if mindmap_png:
        markdown_content += f"""### 本地图片

![思维导图]({mindmap_png})

"""

    # 添加原文（折叠）
    markdown_content += f"""---

## 原文

<details>
<summary>点击展开原文</summary>

```text
{original_text}
```

</details>

---

**报告生成完成** ✅

> 由 [Academic URL Agent](https://github.com/your-repo/academic-url-agent) 自动生成
"""

    # 保存到文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return output_path
