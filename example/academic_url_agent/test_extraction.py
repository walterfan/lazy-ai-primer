"""测试内容提取功能"""

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def test_message_extraction():
    """测试从消息历史中提取正确的内容"""

    # 模拟 LangGraph 的消息历史
    messages = [
        HumanMessage(content="请抓取以下 URL 的英文正文内容：https://example.com"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "fetch_static",
                "args": {"url": "https://example.com"},
                "id": "call_1"
            }]
        ),
        ToolMessage(
            content="This is the actual fetched content from the website...",
            tool_call_id="call_1"
        ),
        AIMessage(content="我已经成功抓取到了网页内容。"),
    ]

    print("🧪 测试消息提取...\n")
    print(f"消息历史中共有 {len(messages)} 条消息：")
    for i, msg in enumerate(messages):
        print(f"  {i+1}. {type(msg).__name__}")

    # 从后向前查找 ToolMessage
    print("\n🔍 从后向前查找 ToolMessage...")
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            print(f"\n✅ 找到 ToolMessage:")
            print(f"   内容: {msg.content[:100]}...")
            return msg.content

    print("\n❌ 未找到 ToolMessage")
    return messages[-1].content


def test_error_case():
    """测试错误情况下的提取"""

    messages = [
        HumanMessage(content="请抓取以下 URL 的英文正文内容：https://example.com"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "fetch_static",
                "args": {"url": "https://example.com"},
                "id": "call_1"
            }]
        ),
        ToolMessage(
            content="[ERROR] 静态抓取失败: Connection timeout",
            tool_call_id="call_1"
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "fetch_dynamic",
                "args": {"url": "https://example.com"},
                "id": "call_2"
            }]
        ),
        ToolMessage(
            content="This is the content fetched by dynamic method. It contains much more text than the error message. " * 3,
            tool_call_id="call_2"
        ),
        AIMessage(content="我使用动态抓取方法成功获取了内容。"),
    ]

    print("\n\n🧪 测试错误重试场景...\n")
    print(f"消息历史中共有 {len(messages)} 条消息")

    # 从后向前查找有效的 ToolMessage
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            print(f"\n检查 ToolMessage: {content[:50]}...")

            if not content.startswith("[ERROR]") and len(content) > 100:
                print(f"✅ 找到有效的 ToolMessage:")
                print(f"   内容: {content[:100]}...")
                return content
            else:
                print(f"⏭️  跳过（错误或内容太短）")

    print("\n❌ 未找到有效的 ToolMessage")
    return messages[-1].content


if __name__ == "__main__":
    print("=" * 60)
    print("消息内容提取测试")
    print("=" * 60)

    result1 = test_message_extraction()
    result2 = test_error_case()

    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)
