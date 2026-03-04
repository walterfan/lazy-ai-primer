"""编辑对话 Agent — 与用户讨论书稿内容"""
from openai import OpenAI
from loguru import logger
from app.core.config import get_settings

SYSTEM_PROMPT = """你是一位经验丰富的书籍编辑助手（AI Editor）。你的职责是：

1. **讨论书稿**：与作者讨论章节内容、结构、风格
2. **提供建议**：给出具体的修改建议和改进方向
3. **回答问题**：解答关于写作技巧、排版规范的问题
4. **头脑风暴**：帮助作者构思新内容、解决写作瓶颈

风格：专业、有建设性、尊重作者的创作意图。
使用中文回答。"""


class ChatAgent:
    """编辑对话智能体"""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)

    def respond(
        self,
        message: str,
        history: list[dict] = None,
        chapter_content: str = "",
    ) -> str:
        """生成回复"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chapter_content:
            messages.append({
                "role": "system",
                "content": f"当前讨论的章节内容（摘要）：\n{chapter_content[:3000]}",
            })

        for msg in (history or [])[-20:]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"对话失败: {e}")
            return f"抱歉，出了点问题：{str(e)}"

    def respond_stream(
        self,
        message: str,
        history: list[dict] = None,
        chapter_content: str = "",
    ):
        """流式生成回复（SSE）"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chapter_content:
            messages.append({
                "role": "system",
                "content": f"当前讨论的章节内容（摘要）：\n{chapter_content[:3000]}",
            })

        for msg in (history or [])[-20:]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            yield f"抱歉，出了点问题：{str(e)}"
