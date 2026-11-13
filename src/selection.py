"""DeepSeek 直连选题：单次结构化输出，不走 deepagents。

原来的选题由 deepagents 主 Agent（DeepSeek）完成，一次选题要带：
deepagents 系统提示词 + 文件系统/task 工具 schema + 30 条候选 + 多轮 tool 往返，
实测一次技术选题约 1.6 万输入 token（2 次调用）。

这里改成「一次 ChatOpenAI + with_structured_output」：
- 系统提示词只有两三句话；
- 只发候选列表，不带任何工具；
- 结果由 Pydantic 直接校验，代码层按链接/标题回填候选元数据，避免模型编字段导致重试。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Sequence, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from src.config import Settings
from src.models import BlogSelectionResult, HotspotSelectionResult, NewsItem
from src.prompts import hotspot_selection_system_prompt, selection_system_prompt
from src.usage import UsageTracker

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# 选题输出很短：picks 1-2 条 + 理由 + 大纲，1400 tokens 足够，顺便封顶成本。
_SELECT_MAX_TOKENS = 1400
_SELECT_TIMEOUT = 120.0


def _selector_model(settings: Settings) -> ChatOpenAI:
    # 选题是「排序 + 写理由」，不需要长思维链；默认关闭 thinking，输出 token 约降 60%。
    # 想恢复深度思考：.env 里设 SELECT_REASONING_EFFORT=medium（或 high）。
    effort = (os.getenv("SELECT_REASONING_EFFORT") or "none").strip().lower()
    extra: dict[str, Any] = {"reasoning_effort": effort} if effort else {}
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.select_temperature,
        max_tokens=_SELECT_MAX_TOKENS,
        timeout=_SELECT_TIMEOUT,
        max_retries=1,
        **extra,
    )


def _parse_json_object(text: str) -> Any | None:
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def _text_of(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _validate(text: str, schema: type[SchemaT], count: int) -> tuple[SchemaT | None, str]:
    obj = _parse_json_object(text)
    if obj is None:
        return None, "没有解析到 JSON 对象"
    try:
        result = schema.model_validate(obj)
    except (ValidationError, TypeError, ValueError) as exc:
        return None, f"schema 校验失败：{str(exc)[:300]}"
    if len(result.picks) != count:  # type: ignore[attr-defined]
        return None, f"picks 数量为 {len(result.picks)}，要求 {count}"  # type: ignore[attr-defined]
    return result, ""


def _select(
    settings: Settings,
    prompt: str,
    count: int,
    schema: type[SchemaT],
    system_prompt: str,
    label: str,
) -> tuple[SchemaT, dict[str, int]]:
    tracker = UsageTracker()
    schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    # 不用 function calling / json_schema：deepseek-v4-flash 的 thinking 模式不支持
    # tool_choice，只支持 response_format={"type":"json_object"}。
    model = _selector_model(settings)
    messages: list[Any] = [
        SystemMessage(
            content=(
                f"{system_prompt}\n\n只输出一个 JSON 对象，不要 Markdown 代码块、不要解释文字。"
                f"JSON 必须符合以下 schema：\n{schema_hint}"
            )
        ),
        HumanMessage(content=prompt),
    ]

    response = model.invoke(
        messages, config={"callbacks": [tracker]}, response_format={"type": "json_object"}
    )
    result, error = _validate(_text_of(response), schema, count)
    if result is None:
        logger.warning("%s 选题首次输出未通过校验（%s），追加一次纠正请求", label, error)
        messages = [
            *messages,
            AIMessage(content=_text_of(response)),
            HumanMessage(
                content=(
                    f"上面的 JSON 未通过校验：{error}。"
                    f"请重新输出恰好 {count} 个 picks 的完整 JSON，字段严格照 schema，不要解释。"
                )
            ),
        ]
        response = model.invoke(
            messages, config={"callbacks": [tracker]}, response_format={"type": "json_object"}
        )
        result, error = _validate(_text_of(response), schema, count)
    if result is None:
        raise ValueError(f"DeepSeek 选题结果不合法：{error}")

    usage = tracker.log(f"deepseek-{label}-selection")
    return result, usage


def select_blog_topics(
    settings: Settings, prompt: str, count: int = 1
) -> tuple[BlogSelectionResult, dict[str, int]]:
    """选技术博客主题（DeepSeek 单次结构化输出）。"""
    return _select(
        settings,
        prompt,
        count,
        BlogSelectionResult,
        selection_system_prompt(count),
        "tech",
    )


def select_hotspot_topics(
    settings: Settings, prompt: str, count: int = 2
) -> tuple[HotspotSelectionResult, dict[str, int]]:
    """选热点新闻（DeepSeek 单次结构化输出）。"""
    return _select(
        settings,
        prompt,
        count,
        HotspotSelectionResult,
        hotspot_selection_system_prompt(count),
        "hotspot",
    )


def ground_picks(picks: Sequence[Any], items: Sequence[NewsItem]) -> None:
    """用候选列表回填 topic_title/link/source/published_at，防止模型改写出错。"""
    by_link = {item.link: item for item in items if item.link}
    by_title = {item.title: item for item in items if item.title}

    for pick in picks:
        item = by_link.get(getattr(pick, "link", "")) or by_title.get(
            getattr(pick, "topic_title", "")
        )
        if item is None:
            title = getattr(pick, "topic_title", "")
            item = next(
                (
                    candidate
                    for candidate in items
                    if title and (title in candidate.title or candidate.title in title)
                ),
                None,
            )
        if item is None:
            logger.warning("选题 %r 未在候选中匹配到，保留模型原始字段", getattr(pick, "topic_title", ""))
            continue
        pick.topic_title = item.title or pick.topic_title
        pick.link = item.link
        pick.source = item.source
        pick.published_at = item.published_at
