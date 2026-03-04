"""AI 编辑 Agent — 核心智能体，支持多种编辑操作"""
from loguru import logger
from openai import OpenAI
from app.core.config import get_settings
from diff_match_patch import diff_match_patch


EDIT_PROMPTS = {
    "proofread": """你是一位专业的中文校对编辑。请校对以下文本：

任务：
1. 修正错别字和错误用词
2. 修正语法错误
3. 修正标点符号错误
4. 保持原文风格和语气不变
5. 不要改变原文的意思和结构

只输出修正后的完整文本，不要添加任何解释。

原文：
{text}""",

    "polish": """你是一位资深的技术书籍编辑。请润色以下文本：

任务：
1. 改善语言表达，使其更流畅自然
2. 增强可读性，适当调整句式
3. 确保技术术语使用准确
4. 保持原文的核心意思不变
5. 保持技术文档的专业性

只输出润色后的完整文本，不要添加任何解释。

原文：
{text}""",

    "expand": """你是一位经验丰富的技术作者。请扩写以下文本：

任务：
1. 补充更多细节和解释
2. 添加实际的代码示例（如适用）
3. 增加类比和比喻帮助理解
4. 补充注意事项和最佳实践
5. 保持原文的风格和结构

{instruction}

只输出扩写后的完整文本。

原文：
{text}""",

    "condense": """你是一位精炼的技术编辑。请缩写以下文本：

任务：
1. 去除冗余和重复的内容
2. 精简表达，保留核心信息
3. 合并相似的段落
4. 删除不必要的修饰语
5. 目标：缩减 30-50% 的篇幅

只输出缩写后的完整文本，不要添加任何解释。

原文：
{text}""",

    "restructure": """你是一位资深的技术书籍架构师。请重构以下文本的结构：

任务：
1. 重新组织段落顺序，使逻辑更清晰
2. 添加或调整标题层级
3. 将长段落拆分为更易读的小段
4. 确保内容的递进关系合理
5. 添加过渡句使段落衔接自然

{instruction}

只输出重构后的完整文本。

原文：
{text}""",

    "translate": """你是一位专业的技术文档翻译。请将以下文本翻译为{target_language}：

要求：
1. 技术术语翻译准确
2. 保持原文的结构和格式（Markdown）
3. 代码块中的注释也要翻译
4. 保持专业但易读的语气
5. 对于约定俗成的术语保留英文（如 API、SDK）

只输出翻译后的完整文本。

原文：
{text}""",

    "review": """你是一位严格的技术书籍审稿人。请审查以下文本并给出修改建议：

审查维度：
1. **技术准确性**：概念是否正确？代码示例是否可运行？
2. **逻辑连贯性**：段落之间是否有清晰的逻辑关系？
3. **可读性**：语言是否清晰？是否有难以理解的表述？
4. **完整性**：是否有遗漏的重要内容？
5. **一致性**：术语使用是否一致？风格是否统一？

请用以下格式输出：

## 总体评价
（1-2 句话概括）

## 具体问题
1. 🔴 [必须修改] ...
2. ⚠️ [建议修改] ...
3. 💡 [可以改进] ...

## 修改建议
（具体的修改方案）

原文：
{text}""",
}


class EditorAgent:
    """AI 编辑智能体"""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.dmp = diff_match_patch()

    def edit(
        self,
        text: str,
        action: str,
        instruction: str = "",
        target_language: str = "English",
    ) -> dict:
        """执行编辑操作"""
        prompt_template = EDIT_PROMPTS.get(action)
        if not prompt_template:
            raise ValueError(f"不支持的编辑操作: {action}")

        prompt = prompt_template.format(
            text=text,
            instruction=instruction,
            target_language=target_language,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "你是一位专业的书籍编辑助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3 if action in ("proofread", "translate") else 0.7,
                max_tokens=4096,
            )
            edited = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"编辑操作失败: {e}")
            raise

        # 生成 diff
        diff_html = ""
        suggestions = []
        if action != "review":
            diffs = self.dmp.diff_main(text, edited)
            self.dmp.diff_cleanupSemantic(diffs)
            diff_html = self.dmp.diff_prettyHtml(diffs)
        else:
            suggestions = self._extract_suggestions(edited)

        # 统计
        original_words = len(text)
        edited_words = len(edited)
        stats = {
            "original_chars": original_words,
            "edited_chars": edited_words,
            "change_ratio": round(abs(edited_words - original_words) / max(original_words, 1) * 100, 1),
        }

        return {
            "original": text,
            "edited": edited,
            "action": action,
            "diff_html": diff_html,
            "suggestions": suggestions,
            "stats": stats,
        }

    def _extract_suggestions(self, review_text: str) -> list[str]:
        """从审查结果中提取建议列表"""
        suggestions = []
        for line in review_text.split("\n"):
            line = line.strip()
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                suggestions.append(line)
            elif line.startswith(("🔴", "⚠️", "💡", "-")):
                suggestions.append(line)
        return suggestions


class WriterAgent:
    """AI 写作智能体"""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)

    def write(
        self,
        topic: str,
        outline: str = "",
        style: str = "technical",
        word_count: int = 2000,
        context: str = "",
    ) -> dict:
        """生成书稿内容"""
        style_desc = {
            "technical": "专业严谨的技术写作风格，适合技术书籍",
            "casual": "轻松易读的风格，适合技术博客",
            "academic": "学术论文风格，严谨规范",
        }.get(style, "专业技术写作风格")

        prompt = f"""你是一位资深的技术书籍作者。请撰写以下内容：

主题：{topic}
风格：{style_desc}
目标字数：约 {word_count} 字

{f"大纲要求：{chr(10)}{outline}" if outline else "请先构思合理的大纲，再按大纲撰写。"}

{f"上下文（前后章节摘要）：{chr(10)}{context}" if context else ""}

要求：
1. 使用 Markdown 格式
2. 包含清晰的标题层级（##、###）
3. 技术概念要有代码示例
4. 语言清晰、逻辑连贯
5. 适当使用列表、表格增强可读性
6. 在关键概念处添加提示框（用 > 引用块）

请直接输出完整的章节内容。"""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "你是一位专业的技术书籍作者，擅长将复杂概念讲解清楚。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=8192,
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"写作失败: {e}")
            raise

        return {
            "content": content,
            "word_count": len(content),
            "outline_used": outline or "（自动生成）",
        }
