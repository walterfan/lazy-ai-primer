"""测试 Markdown 报告生成功能"""

from academic_url_agent.pipeline import save_markdown_report


def test_markdown_generation():
    """测试 Markdown 报告生成"""
    print("🧪 测试 Markdown 报告生成...\n")

    # 测试数据
    test_data = {
        "url": "https://example.com/test-article",
        "original_text": """# Test Article

This is a test article with some content.

## Section 1
Content of section 1.

## Section 2
Content of section 2.""",
        "translation": """# 测试文章

这是一篇带有一些内容的测试文章。

## 第一节
第一节的内容。

## 第二节
第二节的内容。""",
        "summary": """1. 这是一个测试文章
2. 包含两个主要部分
3. 用于验证 Markdown 生成功能""",
        "explanation": """### 概念 1: Markdown
Markdown 是一种轻量级标记语言，用于格式化文本。

### 概念 2: 测试
测试是验证功能正确性的重要步骤。""",
        "mindmap_script": """@startmindmap
* 测试文章
** 第一节
*** 内容点 1
*** 内容点 2
** 第二节
*** 内容点 3
@endmindmap""",
        "mindmap_png": "test_mindmap.png",
    }

    # 生成报告
    output_path = "test_report.md"
    try:
        result_path = save_markdown_report(
            url=test_data["url"],
            original_text=test_data["original_text"],
            translation=test_data["translation"],
            summary=test_data["summary"],
            explanation=test_data["explanation"],
            mindmap_script=test_data["mindmap_script"],
            mindmap_png=test_data["mindmap_png"],
            output_path=output_path,
        )

        print(f"✅ Markdown 报告已生成: {result_path}")

        # 读取并显示前几行
        with open(result_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"\n📄 报告预览（前 10 行）:\n")
            for line in lines[:10]:
                print(line, end="")

        print(f"\n\n✅ 测试通过！完整报告已保存到: {result_path}")
        print(f"📝 使用 Markdown 查看器打开查看完整内容")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Markdown 报告生成测试")
    print("=" * 60)

    success = test_markdown_generation()

    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  测试失败，请检查错误信息")
    print("=" * 60)
