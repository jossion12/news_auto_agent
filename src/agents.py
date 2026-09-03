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
from src.models import SelectionResult
from src.prompts import MAIN_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT

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


def build_main_agent(settings: Settings, writer_tools: Sequence[BaseTool]):
    """创建 1 主 Agent（选题、结构化输出）+ 1 写作 SubAgent。"""
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
        system_prompt=MAIN_SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True),
        checkpointer=MemorySaver(),
        subagents=[writer_spec],
        # 注意：本地小模型对 response_format（结构化输出工具）支持不稳定，
        # 选题结论改为由提示词约束最终消息输出 JSON，代码层校验+重试（见 run_selection）。
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


def _validate_selection(raw: Any) -> tuple[SelectionResult | None, str | None]:
    try:
        if isinstance(raw, SelectionResult):
            return raw, None
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        selection = SelectionResult.model_validate(raw)
        return selection, None
    except (ValidationError, TypeError, ValueError) as exc:
        return None, str(exc)


def _extract_selection(state: Mapping[str, Any]) -> tuple[SelectionResult | None, list[str]]:
    errors: list[str] = []
    structured = state.get("structured_response")
    if structured is not None:
        selection, error = _validate_selection(structured)
        if selection is not None:
            return selection, errors
        errors.append(f"structured_response 校验失败：{error}")

    for text in _iter_json_texts(state):
        obj = _parse_json_object(text)
        if obj is None:
            continue
        selection, error = _validate_selection(obj)
        if selection is not None:
            return selection, errors
        if error:
            errors.append(error)
    return None, errors


def _retry_prompt(errors: list[str]) -> str:
    detail = "；".join(errors[:3]) if errors else "最终 JSON 缺失"
    return (
        "你上次的最终输出没有通过选题结论校验（错误：" + detail + "）。"
        "请基于同一候选继续主线流程：若尚未确认 writer 已写入文件，请先调用 task（subagent_type='writer'）"
        "委托或催办写作；若 writer 已完成，不要重复调用或写文件。"
        "你的最后一条消息必须【只包含一个 JSON 对象】，不要 Markdown 代码块、不要解释文字，字段："
        "topic_title/link/source/published_at/selection_reason/outline"
        "（outline 为 5-8 项的字符串数组，topic_title/link/source/published_at 照抄候选列表）。"
    )


def run_selection(agent, user_prompt: str, thread_id: str) -> tuple[Mapping[str, Any], SelectionResult]:
    """运行主 Agent（含委派 writer），并对 SelectionResult 解析失败重试一次。"""
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("启动主 Agent：选题 -> task(writer) -> 结构化输出……")
    state = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]}, config=config)

    selection, errors = _extract_selection(state)
    if selection is not None:
        return state, selection

    logger.warning("主 Agent 输出未通过 SelectionResult 校验，按 FR1 重试一次。")
    state = agent.invoke(
        {"messages": [{"role": "user", "content": _retry_prompt(errors)}]},
        config=config,
    )
    selection, retry_errors = _extract_selection(state)
    if selection is not None:
        return state, selection

    all_errors = (errors + retry_errors) or ["两次均未返回 SelectionResult"]
    _dump_transcript(state, selection=None)
    raise RuntimeError("两次尝试均未获得合法 SelectionResult：" + "；".join(all_errors[:5]))


def _dump_transcript(state: Mapping[str, Any], selection: SelectionResult | None) -> Path:
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
