"""测试脚本 - 验证安装和配置"""

import os
from dotenv import load_dotenv

def test_env():
    """测试环境变量配置"""
    load_dotenv()

    print("🔍 检查环境变量配置...\n")

    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    disable_ssl = os.getenv("DISABLE_SSL_VERIFY", "false")

    print(f"✓ LLM_API_KEY: {'已设置 (' + api_key[:10] + '...)' if api_key else '未设置'}")
    print(f"✓ LLM_BASE_URL: {base_url if base_url else '使用默认 OpenAI'}")
    print(f"✓ LLM_MODEL: {model}")
    print(f"✓ DISABLE_SSL_VERIFY: {disable_ssl}")

    if not api_key:
        print("\n⚠️  警告: LLM_API_KEY 未设置")
        print("   请创建 .env 文件并配置 API 密钥")
        return False

    return True


def test_imports():
    """测试依赖导入"""
    print("\n🔍 测试依赖导入...\n")

    try:
        import langchain_core
        print("✓ langchain_core")
    except ImportError as e:
        print(f"✗ langchain_core: {e}")
        return False

    try:
        import langchain_openai
        print("✓ langchain_openai")
    except ImportError as e:
        print(f"✗ langchain_openai: {e}")
        return False

    try:
        import langgraph
        print("✓ langgraph")
    except ImportError as e:
        print(f"✗ langgraph: {e}")
        return False

    try:
        import requests
        print("✓ requests")
    except ImportError as e:
        print(f"✗ requests: {e}")
        return False

    try:
        from bs4 import BeautifulSoup
        print("✓ beautifulsoup4")
    except ImportError as e:
        print(f"✗ beautifulsoup4: {e}")
        return False

    try:
        from readability import Document
        print("✓ readability-lxml")
    except ImportError as e:
        print(f"✗ readability-lxml: {e}")
        return False

    return True


def test_tools():
    """测试工具模块"""
    print("\n🔍 测试工具模块...\n")

    try:
        from academic_url_agent.tools import ALL_TOOLS, fetch_static, fetch_dynamic
        print(f"✓ 成功导入工具: {[t.name for t in ALL_TOOLS]}")
        return True
    except Exception as e:
        print(f"✗ 工具导入失败: {e}")
        return False


def test_graph():
    """测试图模块"""
    print("\n🔍 测试 LangGraph 模块...\n")

    try:
        from academic_url_agent.graph import fetch_graph
        print("✓ 成功创建 ReAct 图")
        return True
    except Exception as e:
        print(f"✗ 图创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Academic URL Agent - 环境测试")
    print("=" * 60)

    results = []

    results.append(("环境变量", test_env()))
    results.append(("依赖导入", test_imports()))
    results.append(("工具模块", test_tools()))
    results.append(("LangGraph 图", test_graph()))

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 所有测试通过！")
        print("\n运行示例:")
        print('  poetry run python -m academic_url_agent.main "https://example.com"')
    else:
        print("\n⚠️  部分测试失败，请检查配置")
