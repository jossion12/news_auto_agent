"""LangChain tool：Ollama web_search / web_fetch 包装 + 输出路径规划。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from langchain.tools import tool

from src.config import Settings
from src.output import build_output_path

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 4000
_TRUNCATION_MARKER = "\n...[结果已截断]"
MAX_SEARCH_RESULTS = 10  # 官方文档：max_results 最大 10
_FALLBACK_TIMEOUT = 60.0


def _serialize_result(data: Any) -> str:
    """Pydantic 模型/字典序列化为字符串并截断到 <=4000 字符。"""
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json", exclude_none=True)
    text = json.dumps(data, ensure_ascii=False)
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    keep = MAX_TOOL_RESULT_CHARS - len(_TRUNCATION_MARKER)
    return text[:keep] + _TRUNCATION_MARKER


def _create_ollama_client(settings: Settings):
    """创建带认证的 Ollama 客户端。

    参考: https://github.com/ollama/ollama-python/blob/main/examples/web-search.py

    注意事项:
    - web_search/web_fetch 必须使用 Bearer token 认证
    - OLLAMA_API_KEY 来自 .env 文件
    """
    from ollama import Client  # 延迟导入，避免模块顶部循环依赖

    headers: dict[str, str] = {}
    if settings.ollama_api_key:
        headers["Authorization"] = f"Bearer {settings.ollama_api_key}"

    return Client(
        host=settings.ollama_base_url,
        headers=headers if headers else None,
    )


def _experimental_fallback(base_url: str, endpoint: str, payload: dict) -> dict:
    """调用本地 Ollama 服务的 /api/experimental 端点（使用本机已登录的 Ollama 账号）。

    注意：此降级机制已过时，仅作向后兼容保留。
    新代码建议直接使用 ollama.Client + Bearer token。
    """
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/experimental/{endpoint}",
        json=payload,
        timeout=_FALLBACK_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def make_web_tools(settings: Settings, web_state: dict | None = None):
    """把 ollama 包的 web_search/web_fetch 包装为两个 LangChain tool。

    优先走官方 Python SDK（Bearer key 直连 ollama.com，见
    https://docs.ollama.com/capabilities/web-search）；失败时降级到本地 Ollama
    服务的 /api/experimental 端点（依赖本机已登录的 Ollama 账号）。

    web_state（可选）：{"ok": bool, "failures": int}，由调用方传入以追踪联网可用性，
    用于运行后按 FR3 在代码层补齐降级标记（警告行/参考链接）。

    用法示例（参考 tests/test_ollama_web_tools.py）:
        client = _create_ollama_client(settings)
        response = client.web_search(query="python", max_results=3)
        response = client.web_fetch(url="https://example.com")
    """
    client = _create_ollama_client(settings)
    base_url = settings.ollama_base_url

    @tool
    def web_search(query: str, max_results: int = 3) -> str:
        """联网搜索（Ollama web search）。

        返回 JSON 字符串，单次 <=4000 字符；失败时返回以 [web_search 失败] 开头的提示。

        示例:
            client.web_search(query="python programming", max_results=5)

        返回结构 (WebSearchResponse):
            - results: List[SearchResult]，每个包含 title, url, content
        """
        max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
        try:
            result = client.web_search(query=query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001 - 按 FR3 统一降级
            logger.warning("web_search SDK 调用失败（原因：%s），尝试本地服务端点", exc)
            try:
                result = _experimental_fallback(
                    base_url, "web_search", {"query": query, "max_results": max_results}
                )
            except Exception as fb_exc:  # noqa: BLE001
                logger.warning("web_search 本地服务端点也失败（原因：%s）", fb_exc)
                if web_state is not None:
                    web_state["failures"] = int(web_state.get("failures", 0)) + 1
                return f"[web_search 失败: {type(exc).__name__}: {exc}]"
        if web_state is not None:
            web_state["ok"] = True
        return _serialize_result(result)

    @tool
    def web_fetch(url: str) -> str:
        """抓取网页正文（Ollama web fetch）。

        返回 JSON 字符串，单次 <=4000 字符；失败时返回以 [web_fetch 失败] 开头的提示。

        示例:
            client.web_fetch(url="https://example.com")

        返回结构 (WebFetchResponse):
            - title: str
            - content: str（网页正文）
            - links: List[str]（可选，提取的链接）
        """
        try:
            result = client.web_fetch(url=url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("web_fetch SDK 调用失败（原因：%s），尝试本地服务端点", exc)
            try:
                result = _experimental_fallback(base_url, "web_fetch", {"url": url})
            except Exception as fb_exc:  # noqa: BLE001
                logger.warning("web_fetch 本地服务端点也失败（原因：%s）", fb_exc)
                if web_state is not None:
                    web_state["failures"] = int(web_state.get("failures", 0)) + 1
                return f"[web_fetch 失败: {type(exc).__name__}: {exc}]"
        if web_state is not None:
            web_state["ok"] = True
        return _serialize_result(result)

    return [web_search, web_fetch]


def make_output_path_tool(
    project_root: Path,
    output_dir: Path,
    rss_date: str,
    existing_names: set[str],
):
    """供写作 SubAgent 计算最终文件路径，保证 slug/冲突规则在代码层执行。"""

    @tool
    def resolve_output_path(topic_title: str, english_slug_hint: str = "") -> str:
        """计算文章输出文件路径并返回虚拟路径（如 /output/2026-09-03-xxx.md）。

        写作前必须调用一次，并把返回值原样作为 write_file 的 file_path 参数。
        已有同名文件时自动追加 -2/-3。
        """
        _, virtual_path = build_output_path(
            project_root=project_root,
            output_dir=output_dir,
            rss_date=rss_date,
            topic_title=topic_title,
            english_slug_hint=english_slug_hint,
            existing_names=existing_names,
        )
        return virtual_path

    return resolve_output_path
