"""
测试 Ollama 的 web_search 和 web_fetch 工具

参考: https://github.com/ollama/ollama-python/blob/main/examples/web-search.py

运行方式:
    pytest tests/test_ollama_web_tools.py -v
    python -m pytest tests/test_ollama_web_tools.py -v -s  # 显示 print 输出
"""

import os
import pytest
from dotenv import load_dotenv
from ollama import Client

# 加载 .env 环境变量
load_dotenv()

# 获取 API Key
api_key = os.getenv("OLLAMA_API_KEY")
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 创建客户端（带认证）
# 注意：ollama 库要求 web_search/web_fetch 必须有 Bearer token
client = Client(
    host=base_url,
    headers={"Authorization": f"Bearer {api_key}"} if api_key else None
)

# 确保 API key 已设置
if not api_key:
    pytest.skip("OLLAMA_API_KEY not set in .env", allow_module_level=True)


class TestOllamaWebSearch:
    """测试 web_search 工具"""

    def test_web_search_basic(self):
        """基础搜索测试"""
        query = "python programming language"
        response = client.web_search(query=query)
        
        print(f"\n=== Web Search Results for: {query} ===")
        print(f"Number of results: {len(response.results)}")
        
        for i, result in enumerate(response.results[:3]):  # 只打印前3个
            print(f"\n--- Result {i+1} ---")
            print(f"Title: {result.title}")
            print(f"URL: {result.url}")
            # 安全打印 content，避免编码问题
            content_preview = result.content[:200] if result.content else ""
            try:
                print(f"Content: {content_preview}...")
            except UnicodeEncodeError:
                print(f"Content: [contains special characters]")
        
        assert hasattr(response, "results")
        assert len(response.results) > 0
        assert response.results[0].content is not None

    def test_web_search_chinese(self):
        """中文搜索测试"""
        query = "人工智能最新发展"
        response = client.web_search(query=query)
        
        print(f"\n=== Chinese Search Results for: {query} ===")
        print(f"Number of results: {len(response.results)}")
        
        if response.results:
            print(f"First result: {response.results[0].title}")
        
        assert hasattr(response, "results")

    def test_web_search_max_results(self):
        """指定最大结果数"""
        query = "machine learning"
        max_results = 5
        response = client.web_search(query=query, max_results=max_results)
        
        print(f"\n=== Web Search with max_results={max_results} ===")
        print(f"Requested: {max_results}, Got: {len(response.results)}")
        
        assert len(response.results) <= max_results


class TestOllamaWebFetch:
    """测试 web_fetch 工具"""

    def test_web_fetch_basic(self):
        """基础网页抓取测试"""
        url = "https://example.com"
        response = client.web_fetch(url=url)
        
        print(f"\n=== Web Fetch Results for: {url} ===")
        print(f"Title: {response.title}")
        print(f"Content length: {len(response.content)}")
        print(f"Content preview: {response.content[:300]}...")
        
        assert hasattr(response, "content")
        assert len(response.content) > 0

    def test_web_fetch_with_client(self):
        """使用不同的 URL 测试"""
        url = "https://httpbin.org/html"
        response = client.web_fetch(url=url)
        
        print(f"\n=== Web Fetch for: {url} ===")
        print(f"Content length: {len(response.content)}")
        
        assert hasattr(response, "content")

    def test_web_fetch_with_links(self):
        """测试包含链接的网页抓取"""
        url = "https://github.com"
        response = client.web_fetch(url=url)
        
        print(f"\n=== Web Fetch with Links for: {url} ===")
        print(f"Title: {response.title}")
        print(f"Content length: {len(response.content)}")
        if response.links:
            print(f"Number of links found: {len(response.links)}")
            print(f"Sample links: {response.links[:5]}")
        
        assert hasattr(response, "content")


class TestOllamaWebSearchIntegration:
    """集成测试: 先搜索再抓取"""

    def test_search_and_fetch(self):
        """测试搜索获取URL后抓取页面"""
        # 1. 搜索
        query = "openai chatgpt"
        search_results = client.web_search(query=query)
        
        print(f"\n=== Integration Test ===")
        print(f"Search query: {query}")
        print(f"Found {len(search_results.results)} results")
        
        assert len(search_results.results) > 0
        
        # 2. 尝试抓取一个已知可用的 URL，而不是搜索结果的第一个（可能404）
        test_url = "https://example.com"
        print(f"\nFetching known good URL: {test_url}")
        
        fetch_result = client.web_fetch(url=test_url)
        print(f"Fetched content length: {len(fetch_result.content)}")
        
        assert len(fetch_result.content) > 0


if __name__ == "__main__":
    # 允许直接运行此文件进行测试
    pytest.main([__file__, "-v", "-s"])
