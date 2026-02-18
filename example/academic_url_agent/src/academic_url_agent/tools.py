"""tools.py — Agent 可用的抓取工具。

关键点：
  1. @tool 装饰器：函数 → LangChain Tool
  2. docstring = 工具说明书，LLM 靠它决定何时调用
  3. 返回值中的 [WARN]/[ERROR] 前缀 = 给 Agent 的信号
"""

import re
import requests
from bs4 import BeautifulSoup
from readability import Document
from langchain_core.tools import tool


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


@tool
def fetch_static(url: str) -> str:
    """用 HTTP GET 静态抓取网页正文。适用于博客、文档等服务端渲染页面。
    如果返回内容包含 [WARN]，说明抓取质量不佳。"""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TranslatorBot/1.0)"},
            timeout=15,
        )
        resp.raise_for_status()
        doc = Document(resp.text)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        if len(text) < 200:
            return (
                f"[WARN] 正文仅 {len(text)} 字符，可能抓取失败。"
                "建议改用 fetch_dynamic 工具进行 JS 渲染抓取。"
            )
        return text[:15000]
    except Exception as e:
        return f"[ERROR] 静态抓取失败: {e}。建议改用 fetch_dynamic。"


@tool
def fetch_dynamic(url: str) -> str:
    """用 Playwright 启动无头浏览器渲染页面后抓取正文。
    适用于 SPA、JS 渲染等静态抓取无法获取内容的页面。
    注意：速度较慢，仅在 fetch_static 失败时使用。"""
    try:
        from playwright.sync_api import sync_playwright
        import time

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # 访问页面
            page.goto(url, wait_until="networkidle", timeout=30000)

            # 额外等待以确保动态内容加载完成
            time.sleep(2)

            # 尝试滚动页面触发懒加载
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1)

            html = page.content()
            browser.close()

        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        # 检查是否抓取到的主要是 JavaScript 代码
        if "function(" in text or "=>{" in text or text.count("{") > len(text) / 50:
            return (
                f"[WARN] 检测到大量 JavaScript 代码（{len(text)} 字符），"
                "可能未正确提取正文。建议尝试 fetch_rfc_direct 工具。"
            )

        if len(text) < 200:
            return (
                f"[WARN] Playwright 渲染后正文仅 {len(text)} 字符，"
                "页面可能需要登录或有反爬。"
            )
        return text[:15000]
    except Exception as e:
        return f"[ERROR] 动态抓取失败: {e}"


@tool
def fetch_rfc_direct(url_or_number: str) -> str:
    """直接从 IETF 官方源抓取 RFC 文档纯文本版本。
    适用于 RFC 文档（如 RFC 7519）。

    参数:
        url_or_number: 可以是完整 URL 或 RFC 编号

    支持的输入:
        - 完整 URL: "http://www.rfcreader.com/#rfc7519"
        - 完整 URL: "https://www.rfc-editor.org/rfc/rfc7519.txt"
        - RFC 编号: "7519" 或 "rfc7519"

    工具会自动从输入中提取 RFC 编号，然后从 IETF 官方源抓取纯文本版本。
    """
    try:
        # 尝试从输入中提取 RFC 编号
        rfc_num = extract_rfc_number(url_or_number)

        if not rfc_num:
            # 如果无法从 URL 提取，尝试直接解析数字
            rfc_num = url_or_number.lower().replace("rfc", "").strip()
            # 只保留数字
            rfc_num = ''.join(filter(str.isdigit, rfc_num))

        if not rfc_num:
            return "[ERROR] 无法从输入中提取 RFC 编号。请提供有效的 RFC URL 或编号。"

        # 构建 IETF 官方 RFC 文本 URL
        official_url = f"https://www.rfc-editor.org/rfc/rfc{rfc_num}.txt"

        print(f"  📡 从 IETF 官方源抓取 RFC {rfc_num}: {official_url}")

        resp = requests.get(
            official_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TranslatorBot/1.0)"},
            timeout=30,
        )
        resp.raise_for_status()

        text = resp.text

        # 验证是否是 RFC 文档
        if "Request for Comments" not in text and "RFC" not in text[:500]:
            return f"[WARN] 抓取的内容不像是 RFC 文档。请检查 RFC 编号 {rfc_num} 是否正确。"

        if len(text) < 500:
            return f"[WARN] RFC 文档内容过短（{len(text)} 字符），可能抓取失败。"

        # RFC 文档通常很长，返回前 20000 字符
        print(f"  ✅ 成功抓取 RFC {rfc_num}（{len(text)} 字符）")
        return text[:20000]

    except Exception as e:
        return f"[ERROR] RFC 直接抓取失败: {e}。请确认 RFC 编号或 URL 正确。"


ALL_TOOLS = [fetch_static, fetch_dynamic, fetch_rfc_direct]
