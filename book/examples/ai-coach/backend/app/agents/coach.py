"""AI 教练 Agent — 核心智能体，支持多种模式"""
from loguru import logger
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from app.core.config import get_settings
from app.core.openai_client import get_openai_client_kwargs


# 不同模式的系统提示词
SYSTEM_PROMPTS = {
    "coach": """你是一位经验丰富的学习教练（AI Coach）。你的职责是：

1. **督促学习**：关心用户的学习进度，温和但坚定地督促
2. **制定计划**：帮助用户制定合理的学习计划和里程碑
3. **答疑解惑**：基于知识库中的资料回答技术问题
4. **鼓励激励**：在用户遇到困难时给予鼓励和支持
5. **反馈建议**：基于学习数据给出个性化建议

风格：像一位亲切但专业的导师，既有耐心又有要求。
使用中文回答。适当使用 emoji 增加亲和力。""",

    "tutor": """你是一位专业的技术导师（AI Tutor）。你的职责是：

1. **深入讲解**：用通俗易懂的语言解释技术概念
2. **举例说明**：用实际代码和类比帮助理解
3. **循序渐进**：从基础到高级，逐步引导
4. **检查理解**：适时提问确认用户是否理解

风格：像一位耐心的大学教授，善于用简单的方式解释复杂概念。
使用中文回答。代码示例要完整可运行。""",

    "quiz": """你是一位严格的考试官（AI Quiz Master）。你的职责是：

1. **出题考核**：根据用户的学习内容出题
2. **评判答案**：客观评价用户的回答
3. **解析错误**：详细解释错误原因和正确答案
4. **难度递进**：根据用户表现调整题目难度

风格：严格但公正，注重考核理解深度而非死记硬背。
使用中文。每次只出一道题，等用户回答后再出下一题。""",
}


class CoachAgent:
    """AI 教练智能体"""

    def __init__(self, rag_engine=None):
        self.settings = get_settings()
        self.rag_engine = rag_engine
        self.llm = OpenAI(
            model=self.settings.OPENAI_MODEL,
            temperature=0.7,
            api_key=self.settings.OPENAI_API_KEY,
            **get_openai_client_kwargs(self.settings),
        )

    async def respond(
        self,
        message: str,
        history: list[dict] = None,
        mode: str = "coach",
    ) -> dict:
        """生成教练回复"""
        history = history or []
        sources = []

        # 1. 如果是 coach 或 tutor 模式，先检索知识库
        rag_context = ""
        if self.rag_engine and mode in ("coach", "tutor"):
            try:
                rag_result = await self.rag_engine.query(message, top_k=3)
                if rag_result["sources"]:
                    sources = rag_result["sources"]
                    rag_context = "\n\n📚 相关知识库内容：\n"
                    for s in sources:
                        rag_context += f"- [{s['title']}] {s['text']}\n"
            except Exception as e:
                logger.warning(f"RAG 检索失败: {e}")

        # 2. 构建消息列表
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["coach"])
        if rag_context:
            system_prompt += f"\n\n{rag_context}"

        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)]

        # 添加历史消息（最近 20 条）
        for msg in history[-20:]:
            role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg["content"]))

        # 3. 调用 LLM
        try:
            response = self.llm.chat(messages)
            reply = response.message.content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            reply = "抱歉，我暂时无法回答。请稍后再试。🙏"

        return {
            "reply": reply,
            "sources": sources,
        }

    async def respond_stream(
        self,
        message: str,
        history: list[dict] = None,
        mode: str = "coach",
    ):
        """流式生成教练回复（SSE）"""
        history = history or []
        sources = []

        # 检索知识库
        rag_context = ""
        if self.rag_engine and mode in ("coach", "tutor"):
            try:
                rag_result = await self.rag_engine.query(message, top_k=3)
                if rag_result["sources"]:
                    sources = rag_result["sources"]
                    rag_context = "\n\n📚 相关知识库内容：\n"
                    for s in sources:
                        rag_context += f"- [{s['title']}] {s['text']}\n"
            except Exception as e:
                logger.warning(f"RAG 检索失败: {e}")

        # 构建消息
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["coach"])
        if rag_context:
            system_prompt += f"\n\n{rag_context}"

        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)]
        for msg in history[-20:]:
            role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg["content"]))

        # 流式调用 LLM
        try:
            response_gen = self.llm.stream_chat(messages)
            for chunk in response_gen:
                yield {
                    "token": chunk.delta,
                    "sources": sources if chunk.delta else [],
                }
        except Exception as e:
            logger.error(f"流式 LLM 调用失败: {e}")
            yield {"token": f"抱歉，出了点问题：{str(e)}", "sources": []}
