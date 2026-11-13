"""Deep Agents 主 Agent/写作 SubAgent 构建与结构化选题执行（含一次重试）。"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ValidationError

from src.config import PROJECT_ROOT, Settings
from src.models import BlogSelectionResult, HotspotSelectionResult
from src.usage import UsageTracker
from src.prompts import (
    HOTSPOT_WRITER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    hotspot_main_system_prompt,
    main_system_prompt,
)

logger = logging.getLogger(__name__)


def _make_chat_model(settings: Settings, *, writer: bool) -> BaseChatModel:
    """按 provider 构建模型：ollama（本地）或 deepseek（OpenAI 兼容 API）。"""
    provider = settings.write_provider if writer else settings.select_provider
    temperature = settings.write_temperature if writer else settings.select_temperature
    if provider == "deepseek":
        # DeepSeek 上下文窗口远大于本地模型，num_ctx 不适用；NUM_PREDICT 映射为 max_tokens。
        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,
            max_tokens=settings.num_predict,
        )
    if writer:
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.write_model,
            num_ctx=settings.write_num_ctx,
            temperature=temperature,
            num_predict=settings.num_predict,
        )
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.select_model,
        num_ctx=settings.select_num_ctx,
        temperature=temperature,
        num_predict=settings.num_predict,
    )


def build_main_agent(
    settings: Settings, writer_tools: Sequence[BaseTool], count: int = 1
):
    """创建 1 主 Agent（选题、结构化输出）+ 1 写作 SubAgent；count 为要选的主题数。"""
    writer_spec: dict[str, Any] = {
        "name": "writer",
        "description": (
            "把主编选定的主题扩写成 >=1500 字的中文技术博客；"
            "拥有 web_fetch/web_search/resolve_output_path/write_file 工具，负责联网取材并落盘 Markdown。"
        ),
        "model": _make_chat_model(settings, writer=True),
        "system_prompt": WRITER_SYSTEM_PROMPT,
        "tools": list(writer_tools),
    }
    return create_deep_agent(
        name="news-selector",
        model=_make_chat_model(settings, writer=False),
        system_prompt=main_system_prompt(count),
        backend=FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True),
        checkpointer=MemorySaver(),
        subagents=[writer_spec],
        # 注意：本地小模型对 response_format（结构化输出工具）支持不稳定，
        # 选题结论改为由提示词约束最终消息输出 JSON，代码层校验+重试（见 run_selection）。
    )


def build_hotspot_agent(
    settings: Settings, writer_tools: Sequence[BaseTool], count: int = 2
):
    """创建热点主 Agent（选 count 则 + 结构化输出）+ 热点写作 SubAgent。"""
    writer_spec: dict[str, Any] = {
        "name": "hotspot_writer",
        "description": (
            "把主编选定的一则热点新闻写成中文热点解读摘要 md；"
            "有原文链接时用 web_fetch 抓取总结，无链接时用 web_search 搜索总结；"
            "拥有 web_fetch/web_search/resolve_output_path/write_file 工具，负责联网取材并落盘 Markdown。"
        ),
        "model": _make_chat_model(settings, writer=True),
        "system_prompt": HOTSPOT_WRITER_SYSTEM_PROMPT,
        "tools": list(writer_tools),
    }
    return create_deep_agent(
        name="hotspot-selector",
        model=_make_chat_model(settings, writer=False),
        system_prompt=hotspot_main_system_prompt(count),
        backend=FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True),
        checkpointer=MemorySaver(),
        subagents=[writer_spec],
    )


def build_writer_agent(
    settings: Settings, writer_tools: Sequence[BaseTool], *, hotspot: bool = False
):
    """独立写作 Agent：只跑 writer，不经过 DeepSeek 主 Agent，省掉选题后的多轮往返。"""
    return create_deep_agent(
        name="hotspot_writer" if hotspot else "writer",
        model=_make_chat_model(settings, writer=True),
        system_prompt=HOTSPOT_WRITER_SYSTEM_PROMPT if hotspot else WRITER_SYSTEM_PROMPT,
        tools=list(writer_tools),
        backend=FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True),
        checkpointer=MemorySaver(),
    )


def run_writer(
    agent,
    pick: Any,
    item: Any,
    *,
    hotspot: bool = False,
    thread_id: str = "writer",
) -> None:
    """把一条选题直接交给写作 Agent（省去 DeepSeek 主 Agent 的 task 往返）。"""
    lines = [
        f"【选题】{getattr(pick, 'topic_title', '')}",
        f"【原文链接】{getattr(pick, 'link', '') or '（无）'}",
        f"【来源】{getattr(pick, 'source', '')}",
        f"【发布时间】{getattr(pick, 'published_at', '')}",
        f"【RSS 摘要】{(getattr(item, 'summary', '') if item is not None else '') or '（无）'}",
        f"【选稿理由】{getattr(pick, 'selection_reason', '')}",
    ]
    outline = getattr(pick, "outline", None)
    if outline:
        lines.append("【大纲】")
        lines.extend(f"- {point}" for point in outline)
    lines.append("请按你的写作规范完成，并落盘到 output 目录（先调用 resolve_output_path，再调 write_file）。")

    agent.invoke(
        {"messages": [{"role": "user", "content": "\n".join(lines)}]},
        config={"configurable": {"thread_id": thread_id}},
    )


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text") or block.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _iter_json_texts(state: Mapping[str, Any]):
    """遍历结构化响应、AIMessage 正文与 tool_call 参数中可能的 JSON。"""
    structured = state.get("structured_response")
    if structured is not None:
        if hasattr(structured, "model_dump_json"):
            yield structured.model_dump_json()
        else:
            yield json.dumps(structured, ensure_ascii=False)

    for message in reversed(state.get("messages") or []):
        if not isinstance(message, AIMessage):
            continue
        yield _message_text(message)
        for tool_call in getattr(message, "tool_calls", None) or []:
            args = tool_call.get("args")
            if args:
                yield json.dumps(args, ensure_ascii=False)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """从文本（可能夹杂说明文字或代码块）中提取首个 JSON 对象/数组。"""
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
            if isinstance(obj, (dict, list)):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _validate_selection(
    raw: Any, model_cls: type[BaseModel] = BlogSelectionResult
) -> tuple[BaseModel | None, str | None]:
    try:
        if isinstance(raw, model_cls):
            return raw, None
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        selection = model_cls.model_validate(raw)
        return selection, None
    except (ValidationError, TypeError, ValueError) as exc:
        return None, str(exc)


def check_pick_count(selection: BaseModel, expected: int) -> str | None:
    """校验结论中的 picks 数量与期望值一致；一致或模型无 picks 字段时返回 None。"""
    picks = getattr(selection, "picks", None)
    if not isinstance(picks, list):
        return None
    if len(picks) != expected:
        return f"picks 数量 {len(picks)} 与要求 {expected} 不一致"
    return None


def _extract_selection(
    state: Mapping[str, Any],
    model_cls: type[BaseModel] = BlogSelectionResult,
    expected_count: int | None = None,
) -> tuple[BaseModel | None, list[str]]:
    errors: list[str] = []
    structured = state.get("structured_response")
    if structured is not None:
        selection, error = _validate_selection(structured, model_cls)
        if selection is not None:
            count_error = (
                check_pick_count(selection, expected_count)
                if expected_count is not None
                else None
            )
            if count_error is None:
                return selection, errors
            errors.append(f"structured_response {count_error}")
        else:
            errors.append(f"structured_response 校验失败：{error}")

    for text in _iter_json_texts(state):
        obj = _parse_json_object(text)
        if obj is None:
            continue
        selection, error = _validate_selection(obj, model_cls)
        if selection is not None:
            count_error = (
                check_pick_count(selection, expected_count)
                if expected_count is not None
                else None
            )
            if count_error is None:
                return selection, errors
            errors.append(count_error)
        elif error:
            errors.append(error)
    return None, errors


def _retry_prompt(errors: list[str], *, hotspot: bool = False, count: int = 1) -> str:
    detail = "；".join(errors[:3]) if errors else "最终 JSON 缺失"
    if hotspot:
        return (
            "你上次的最终输出没有通过选稿结论校验（错误：" + detail + "）。"
            "请基于同一候选继续主线流程：若 hotspot_writer 尚未全部确认写入文件，"
            "先调用 task（subagent_type='hotspot_writer'）委托或催办；"
            "若都已完成，不要重复调用或写文件。"
            "你的最后一条消息必须【只包含一个 JSON 对象】，不要 Markdown 代码块、不要解释文字，格式："
            "{\"picks\": [{\"topic_title\": \"...\", \"link\": \"...\", \"source\": \"...\", "
            "\"published_at\": \"...\", \"selection_reason\": \"...\"}, {...}]}，"
            f"picks 恰好 {count} 个，topic_title/link/source/published_at 照抄候选列表。"
        )
    return (
        "你上次的最终输出没有通过选题结论校验（错误：" + detail + "）。"
        "请基于同一候选继续主线流程：若 writer 尚未全部确认写入文件，"
        "请先调用 task（subagent_type='writer'）委托或催办写作；"
        "若都已完成，不要重复调用或写文件。"
        "你的最后一条消息必须【只包含一个 JSON 对象】，不要 Markdown 代码块、不要解释文字，格式："
        "{\"picks\": [{\"topic_title\": \"...\", \"link\": \"...\", \"source\": \"...\", "
        "\"published_at\": \"...\", \"selection_reason\": \"...\", "
        "\"outline\": [\"...\", \"...\"]}, {...}]}，"
        f"picks 恰好 {count} 个，outline 为 5-8 项的字符串数组，"
        "topic_title/link/source/published_at 照抄候选列表。"
    )


def run_selection(
    agent, user_prompt: str, thread_id: str, count: int = 1
) -> tuple[Mapping[str, Any], BlogSelectionResult]:
    """运行主 Agent（含委派 writer），并对 BlogSelectionResult 解析失败重试一次。"""
    tracker = UsageTracker()
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [tracker]}
    logger.info("启动主 Agent：选 %d 题 -> task(writer) x%d -> 结构化输出……", count, count)
    state = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]}, config=config)
    tracker.log("tech-selection")

    selection, errors = _extract_selection(state, BlogSelectionResult, count)
    if selection is not None:
        return state, selection  # type: ignore[return-value]

    logger.warning("主 Agent 输出未通过 BlogSelectionResult 校验，按 FR1 重试一次。")
    state = agent.invoke(
        {"messages": [{"role": "user", "content": _retry_prompt(errors, count=count)}]},
        config=config,
    )
    selection, retry_errors = _extract_selection(state, BlogSelectionResult, count)
    if selection is not None:
        return state, selection  # type: ignore[return-value]

    all_errors = (errors + retry_errors) or ["两次均未返回 BlogSelectionResult"]
    _dump_transcript(state, selection=None)
    raise RuntimeError("两次尝试均未获得合法 BlogSelectionResult：" + "；".join(all_errors[:5]))


def run_hotspot_selection(
    agent, user_prompt: str, thread_id: str, count: int = 2
) -> tuple[Mapping[str, Any], HotspotSelectionResult]:
    """运行热点主 Agent（含 count 次委派 hotspot_writer），并重试一次。"""
    tracker = UsageTracker()
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [tracker]}
    logger.info("启动热点主 Agent：选 %d 则 -> task(hotspot_writer) x%d -> 结构化输出……", count, count)
    state = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]}, config=config)
    tracker.log("hotspot-selection")

    selection, errors = _extract_selection(state, HotspotSelectionResult, count)
    if selection is not None:
        return state, selection  # type: ignore[return-value]

    logger.warning("热点主 Agent 输出未通过 HotspotSelectionResult 校验，重试一次。")
    state = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": _retry_prompt(errors, hotspot=True, count=count)}
            ]
        },
        config=config,
    )
    selection, retry_errors = _extract_selection(state, HotspotSelectionResult, count)
    if selection is not None:
        return state, selection  # type: ignore[return-value]

    all_errors = (errors + retry_errors) or ["两次均未返回 HotspotSelectionResult"]
    _dump_transcript(state, selection=None)
    raise RuntimeError("两次尝试均未获得合法 HotspotSelectionResult：" + "；".join(all_errors[:5]))


def _dump_transcript(state: Mapping[str, Any], selection: BaseModel | None) -> Path:
    """把主 Agent 的消息记录转储到 output/debug_transcript.txt，便于排查本地模型输出。"""
    target = PROJECT_ROOT / "output" / "debug_transcript.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, message in enumerate(state.get("messages") or []):
        role = getattr(message, "type", "?")
        lines.append(f"\n===== [{i}] {role} =====")
        lines.append(_message_text(message))
        for tool_call in getattr(message, "tool_calls", None) or []:
            lines.append(f"-- tool_call: {json.dumps(tool_call, ensure_ascii=False)}")
    if selection is not None:
        lines.append("\n===== [final] SelectionResult =====")
        lines.append(selection.model_dump_json(indent=2))
    target.write_text("\n".join(lines), encoding="utf-8")
    logger.warning("已转储对话记录到 %s（排查用）", target)
    return target
