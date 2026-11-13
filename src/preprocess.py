"""代码层 RSS 预处理：元数据剥离、HN 热度解析、截断、top-30 排序。"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path

from src.models import HotspotData, NewsItem, RssData

SUMMARY_LIMIT = 120
TOP_N = 20
HOTSPOT_PER_SOURCE = 5
HOTSPOT_TOP_N = 18


def _env_int(name: str, default: int) -> int:
    """运行时读 env（.env 在 load_settings 时才加载，不能在模块导入时读）。"""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default

_HN_POINTS_RE = re.compile(r"Points:\s*([\d,]+)", re.IGNORECASE)
_HN_COMMENTS_RE = re.compile(r"#?\s*Comments:\s*([\d,]+)", re.IGNORECASE)


def _compact(value: str | None) -> str:
    return " ".join((value or "").split())


def _is_arxiv(source: str, link: str, summary: str) -> bool:
    return "arxiv" in f"{source} {link} {summary}".lower()


def _strip_arxiv_prefix(summary: str) -> str:
    """剥离 'arXiv:... Abstract:' 元数据前缀，保留论文摘要正文。"""
    match = re.search(r"(?:^|\s)Abstract:\s*", summary, flags=re.IGNORECASE)
    if match:
        return summary[match.end() :].strip()
    # 兼容没有 "Abstract:" 前缀的条目：剥掉开头的 arXiv 编号/声明行。
    stripped = re.sub(
        r"^\s*arXiv:\S+(?:\s+Announce\s+Type:[^\n]*)?\s*",
        "",
        summary,
        flags=re.IGNORECASE,
    )
    return stripped.strip()


def _parse_int(value: re.Match[str] | None) -> int:
    if value is None:
        return 0
    try:
        return int(value.group(1).replace(",", ""))
    except ValueError:
        return 0


def _extract_item(index: int, item: dict, summary_limit: int = SUMMARY_LIMIT) -> NewsItem:
    raw_summary = item.get("summary") or ""
    source = _compact(item.get("source"))
    link = _compact(item.get("link"))

    arxiv = _is_arxiv(source, link, raw_summary)
    summary = _strip_arxiv_prefix(raw_summary) if arxiv else raw_summary
    summary = _compact(summary)[:summary_limit]

    points = _parse_int(_HN_POINTS_RE.search(raw_summary))
    comments = _parse_int(_HN_COMMENTS_RE.search(raw_summary))

    return NewsItem(
        index=index,
        title=_compact(item.get("title")),
        link=link,
        source=source or "(未知来源)",
        published_at=_compact(item.get("published_at")),
        summary=summary,
        hn_points=points,
        hn_comments=comments,
        category=_compact(item.get("category")),
    )


def load_rss(rss_file: Path, count: int = 1, json_hint: bool = True) -> RssData:
    """加载 RSS JSON 并返回预处理后的 top 候选与 prompt；count 为要选的主题数。"""
    if not rss_file.is_file():
        raise FileNotFoundError(f"RSS 文件不存在：{rss_file}")
    try:
        with rss_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"RSS JSON 解析失败（{rss_file}）：{exc}") from exc

    try:
        sources = payload["sources"]
        raw_items = [
            item
            for source in sources
            for item in (source.get("items") or [])
        ]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("RSS JSON 缺少顶层 sources/sources[].items 结构") from exc

    summary_limit = _env_int("SELECT_SUMMARY_LIMIT", SUMMARY_LIMIT)
    top_n = _env_int("SELECT_TOP_N", TOP_N)
    items = [_extract_item(idx, item, summary_limit) for idx, item in enumerate(raw_items)]
    # 按 HN Points 降序；无 Points 视为 0。同分时 Comments 高的在前，其余保持原始顺序。
    items.sort(key=lambda it: (-it.hn_points, -it.hn_comments, it.index))
    top_items = items[:top_n]

    rss_date = str(payload.get("date") or date.today().isoformat())
    lines = [f"今日 RSS 日期：{rss_date}"]
    lines.append(f"候选共 {len(top_items)} 条（按 HN Points 降序取 top-{top_n}，无热度计 0）：")
    lines.append("")
    for rank, item in enumerate(top_items, start=1):
        lines.append(f"{rank}. {item.to_candidate()}")
        lines.append("")
    if json_hint:
        lines.append(
            f"最后提醒（务必遵守）：确认全部 writer 都已写入文件后，你的最后一条回复必须只包含一个 "
            f"JSON 对象：{{\"picks\": [{{\"topic_title\": \"...\", \"link\": \"...\", \"source\": \"...\", "
            f"\"published_at\": \"...\", \"selection_reason\": \"...\", \"outline\": [\"...\"]}}, {{...}}]}}，"
            f"picks 必须恰好 {count} 个，outline 为 5-8 项字符串数组，"
            f"topic_title/link/source/published_at 照抄候选列表对应条目，"
            f"不要输出 Markdown 代码块或任何解释文字。"
        )
    return RssData(date=rss_date, top_items=top_items, prompt="\n".join(lines))


def _load_items_payload(data_file: Path, label: str) -> tuple[str, list[dict]]:
    """加载 sources/sources[].items 结构的 JSON，返回 (date, 全部原始条目)。"""
    if not data_file.is_file():
        raise FileNotFoundError(f"{label}文件不存在：{data_file}")
    try:
        with data_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"{label} JSON 解析失败（{data_file}）：{exc}") from exc

    try:
        sources = payload["sources"]
        raw_items = [
            item
            for source in sources
            for item in (source.get("items") or [])
        ]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"{label} JSON 缺少顶层 sources/sources[].items 结构") from exc

    data_date = str(payload.get("date") or date.today().isoformat())
    return data_date, raw_items


def load_hotspots(hotspots_file: Path, count: int = 2, json_hint: bool = True) -> HotspotData:
    """加载热点新闻 JSON，按来源限流（每源至多 HOTSPOT_PER_SOURCE 条）取 top 候选；
    count 为要选的新闻数。"""
    hotspots_date, raw_items = _load_items_payload(hotspots_file, "热点")
    summary_limit = _env_int("SELECT_SUMMARY_LIMIT", SUMMARY_LIMIT)
    hotspot_top_n = _env_int("SELECT_HOTSPOT_TOP_N", HOTSPOT_TOP_N)

    items: list[NewsItem] = []
    per_source_count: dict[str, int] = {}
    for idx, raw in enumerate(raw_items):
        item = _extract_item(idx, raw, summary_limit)
        if not item.title:
            continue
        source_count = per_source_count.get(item.source, 0)
        if source_count >= HOTSPOT_PER_SOURCE:
            continue
        per_source_count[item.source] = source_count + 1
        items.append(item)
        if len(items) >= hotspot_top_n:
            break

    lines = [f"今日热点日期：{hotspots_date}"]
    lines.append(
        f"候选共 {len(items)} 条（每来源至多 {HOTSPOT_PER_SOURCE} 条、"
        f"合计至多 {hotspot_top_n} 条；link 为空的条目无原文链接，writer 需用搜索补素材）："
    )
    lines.append("")
    for rank, item in enumerate(items, start=1):
        lines.append(f"{rank}. {item.to_candidate()}")
        lines.append("")
    if json_hint:
        lines.append(
            f"最后提醒（务必遵守）：确认全部 hotspot_writer 都已写入文件后，你的最后一条回复必须只包含一个 "
            f"JSON 对象：{{\"picks\": [{{\"topic_title\": \"...\", \"link\": \"...\", \"source\": \"...\", "
            f"\"published_at\": \"...\", \"selection_reason\": \"...\"}}, {{...}}]}}，"
            f"picks 必须恰好 {count} 个，"
            f"topic_title/link/source/published_at 照抄候选列表对应条目（link 可为空字符串），"
            f"不要输出 Markdown 代码块或任何解释文字。"
        )
    return HotspotData(date=hotspots_date, items=items, prompt="\n".join(lines))
