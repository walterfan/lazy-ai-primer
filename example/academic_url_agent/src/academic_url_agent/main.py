"""main.py — 程序入口：调用 LangGraph 图 + 生成层管线。

对比之前的手写 while 循环版本：
  Before: react_fetch() 函数里手写 for step in range(MAX_ITERATIONS)
  After:  graph.invoke() 一行搞定，图自己循环

关键区别：
  - 循环逻辑从"代码控制"变成了"图结构控制"
  - 状态管理从"手动 append"变成了"Reducer 自动处理"
  - 可以轻松添加新节点（如 human-in-the-loop 审批节点）
"""

import sys
from langchain_core.messages import HumanMessage

from .graph import fetch_graph
from .pipeline import run_pipeline


def react_fetch(url: str) -> str:
    """用 LangGraph 图执行 ReAct 抓取。

    只需要:
      1. 构造初始状态
      2. 调用 graph.invoke()
      3. 从最终状态中提取结果

    图内部的 ReAct 循环 (agent → tools → agent → ...) 完全自动。
    """
    from langchain_core.messages import ToolMessage

    print("🤖 [决策层] LangGraph ReAct 图启动\n")

    # 构造初始状态
    initial_state = {
        "messages": [
            HumanMessage(
                content=f"请抓取以下 URL 的英文正文内容：{url}"
            ),
        ],
        "iteration": 0,
    }

    # ⭐ 一行调用 —— 图自动执行 ReAct 循环
    final_state = fetch_graph.invoke(initial_state)

    # 从最终状态提取结果
    # 需要找到最后一个成功的 ToolMessage（即抓取结果）
    messages = final_state["messages"]

    # 从后向前查找最后一个 ToolMessage
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            # 检查是否是成功的抓取结果（不是错误或警告）
            if not content.startswith("[ERROR]") and len(content) > 100:
                return content

    # 如果没有找到有效的 ToolMessage，返回最后一条消息的内容
    last_message = messages[-1]
    return last_message.content


def main():
    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else input("请输入英文网页 URL: ").strip()
    )
    if not url:
        print("❌ URL 不能为空")
        return

    # ========== 第一阶段：决策层（LangGraph 图）==========
    content = react_fetch(url)

    if "[ERROR]" in content or len(content) < 100:
        print(f"❌ 抓取失败：{content[:300]}")
        return

    print(f"\n✅ 成功获取正文（{len(content)} 字符）")

    # ========== 第二阶段：生成层（LLM 管线）==========
    result = run_pipeline(content, url=url)

    # ========== 输出 ==========
    print("\n" + "=" * 60)
    print("📖 中文翻译（预览）")
    print("=" * 60)
    print(result["translation"][:2000] + "\n...")

    print("\n" + "=" * 60)
    print("📌 要点总结")
    print("=" * 60)
    print(result["summary"])

    print("\n" + "=" * 60)
    print("🔍 难点解释")
    print("=" * 60)
    print(result["explanation"])

    print("\n" + "=" * 60)
    print("🗺️  PlantUML 思维导图脚本")
    print("=" * 60)
    print(result["mindmap_script"])

    print("\n" + "=" * 60)
    print("📁 输出文件")
    print("=" * 60)
    if result["mindmap_png"]:
        print(f"✅ 思维导图图片: {result['mindmap_png']}")
    print(f"✅ PlantUML 脚本: mindmap.puml")
    print(f"✅ 完整报告: {result['markdown_report']}")

    print("\n💡 提示: 查看完整内容请打开 " + result['markdown_report'])


if __name__ == "__main__":
    main()
