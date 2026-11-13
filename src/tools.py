"""LangChain tool：本地 web_search / web_fetch + 输出路径规划。

联网取材已从 ollama.com 的 web_search/web_fetch 切换到本地
SearXNG + Crawl4AI 封装层（见 `src/local_web.py`）：
- 默认 `http://127.0.0.1:8000`，docker 容器由 cli.py 自动启停
- 返回结构与原 Ollama 工具一致（results[].title/url/content、title/content/links），
  因此 prompts 里的取材策略不用改
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain.tools import tool

from src import local_web
from src.config import Settings
from src.output import build_output_path

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 4000
_TRUNCATION_MARKER = "\n...[结果已截断]"
MAX_SEARCH_RESULTS = 10  # 官方文档：max_results 最大 10
# SDK 直连 ollama.com 的读写超时：不设超时时，网络中断会让请求永久挂起
_SDK_TIMEOUT = 90.0


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
    """创建带认证的 Ollama 客户端（仅作可选兜底，默认不走）。"""
    from ollama import Client  # 延迟导入

    headers: dict[str, str] = {}
    if settings.ollama_api_key:
        headers["Authorization"] = f"Bearer {settings.ollama_api_key}"

    return Client(
        host=settings.ollama_base_url,
        headers=headers if headers else None,
        timeout=_SDK_TIMEOUT,
    )


def _ollama_fallback_enabled() -> bool:
    """OLLAMA 云端联网兜底：默认关闭（额度已用完）。

    需要时在 .env 设 WEB_TOOLS_OLLAMA_FALLBACK=1。
    """
    return os.getenv("WEB_TOOLS_OLLAMA_FALLBACK", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _ollama_search(settings: Settings, query: str, max_results: int) -> Any:
    client = _create_ollama_client(settings)
    try:
        return client.web_search(query=query, max_results=max_results)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _ollama_fetch(settings: Settings, url: str) -> Any:
    client = _create_ollama_client(settings)
    try:
        return client.web_fetch(url=url)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def make_web_tools(settings: Settings, web_state: dict | None = None):
    """把本地联网服务包装为两个 LangChain tool。

    web_state（可选）：{"ok": bool, "failures": int}，由调用方传入以追踪联网可用性，
    用于运行后按 FR3 在代码层补齐降级标记（警告行/参考链接）。
    """

    @tool
    def web_search(query: str, max_results: int = 3) -> str:
        """联网搜索（本地 SearXNG）。

        返回 JSON 字符串（results[].title/url/content），单次 <=4000 字符；
        失败时返回以 [web_search 失败] 开头的提示。

        示例:
            web_search(query="RAG 检索增强生成", max_results=5)
        """
        max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
        try:
            result = local_web.search(query, max_results)
        except Exception as exc:  # noqa: BLE001 - 按 FR3 统一降级
            logger.warning("web_search 本地服务失败（原因：%s）", exc)
            if _ollama_fallback_enabled():
                try:
                    logger.info("web_search 尝试 Ollama 云端兜底")
                    result = _ollama_search(settings, query, max_results)
                except Exception as fb_exc:  # noqa: BLE001
                    logger.warning("web_search Ollama 兜底也失败（原因：%s）", fb_exc)
                    if web_state is not None:
                        web_state["failures"] = int(web_state.get("failures", 0)) + 1
                    return f"[web_search 失败: {type(fb_exc).__name__}: {fb_exc}]"
            else:
                if web_state is not None:
                    web_state["failures"] = int(web_state.get("failures", 0)) + 1
                return f"[web_search 失败: {type(exc).__name__}: {exc}]"

        if web_state is not None:
            web_state["ok"] = True
        return _serialize_result(result)

    @tool
    def web_fetch(url: str) -> str:
        """抓取网页正文（本地 Crawl4AI，支持 JS 渲染页）。

        返回 JSON 字符串（title/content/links），单次 <=4000 字符；
        失败时返回以 [web_fetch 失败] 开头的提示。

        示例:
            web_fetch(url="https://example.com/article")
        """
        try:
            result = local_web.fetch(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("web_fetch 本地服务失败（原因：%s）", exc)
            if _ollama_fallback_enabled():
                try:
                    logger.info("web_fetch 尝试 Ollama 云端兜底")
                    result = _ollama_fetch(settings, url)
                except Exception as fb_exc:  # noqa: BLE001
                    logger.warning("web_fetch Ollama 兜底也失败（原因：%s）", fb_exc)
                    if web_state is not None:
                        web_state["failures"] = int(web_state.get("failures", 0)) + 1
                    return f"[web_fetch 失败: {type(fb_exc).__name__}: {fb_exc}]"
            else:
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
