"""测试 RFC 抓取功能"""

from academic_url_agent.tools import extract_rfc_number, fetch_rfc_direct


def test_rfc_extraction():
    """测试 RFC 编号提取"""
    print("🧪 测试 RFC 编号提取...\n")

    test_cases = [
        ("http://www.rfcreader.com/#rfc7519", "7519"),
        ("https://www.rfc-editor.org/rfc/rfc7519.txt", "7519"),
        ("https://tools.ietf.org/html/rfc7519", "7519"),
        ("rfc7519", "7519"),
        ("RFC 7519", "7519"),
        ("7519", ""),  # 纯数字无 rfc 前缀
    ]

    all_passed = True
    for url, expected in test_cases:
        result = extract_rfc_number(url)
        status = "✅" if result == expected or (expected == "" and result.isdigit()) else "❌"
        print(f"{status} {url:50} → {result:10} (期望: {expected or '数字'})")
        if status == "❌":
            all_passed = False

    return all_passed


def test_rfc_fetch():
    """测试 RFC 抓取（使用真实 API）"""
    print("\n\n🧪 测试 RFC 抓取（真实 API 调用）...\n")

    # 使用一个较小的 RFC 进行测试
    test_url = "http://www.rfcreader.com/#rfc7519"

    print(f"测试 URL: {test_url}")
    print("调用 fetch_rfc_direct...\n")

    try:
        result = fetch_rfc_direct.invoke({"url_or_number": test_url})

        if result.startswith("[ERROR]"):
            print(f"❌ 抓取失败: {result}")
            return False
        elif result.startswith("[WARN]"):
            print(f"⚠️  警告: {result}")
            return False
        else:
            print(f"✅ 成功抓取 ({len(result)} 字符)")
            print(f"\n预览前 500 字符:")
            print("-" * 60)
            print(result[:500])
            print("-" * 60)

            # 验证是否是 RFC 7519 (JWT)
            if "JSON Web Token" in result or "JWT" in result:
                print("\n✅ 验证通过：内容包含 'JSON Web Token' 或 'JWT'")
                return True
            else:
                print("\n⚠️  警告：内容不包含预期的关键词")
                return False

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("RFC 抓取功能测试")
    print("=" * 60)

    # 测试 1: RFC 编号提取
    extraction_passed = test_rfc_extraction()

    # 测试 2: RFC 抓取（需要网络）
    fetch_passed = test_rfc_fetch()

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"RFC 编号提取: {'✅ 通过' if extraction_passed else '❌ 失败'}")
    print(f"RFC 抓取: {'✅ 通过' if fetch_passed else '❌ 失败'}")

    if extraction_passed and fetch_passed:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败")
