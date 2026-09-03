"""代码层 RSS 预处理：元数据剥离、HN 热度解析、截断、top-30 排序。"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from src.models import NewsItem, RssData

SUMMARY_LIMIT = 200
TOP_N = 30

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


def _extract_item(index: int, item: dict) -> NewsItem:
    raw_summary = item.get("summary") or ""
    source = _compact(item.get("source"))
    link = _compact(item.get("link"))

    arxiv = _is_arxiv(source, link, raw_summary)
    summary = _strip_arxiv_prefix(raw_summary) if arxiv else raw_summary
    summary = _compact(summary)[:SUMMARY_LIMIT]

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


def load_rss(rss_file: Path) -> RssData:
    """加载 RSS JSON 并返回预处理后的 top-30 候选与 prompt。"""
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

    items = [_extract_item(idx, item) for idx, item in enumerate(raw_items)]
    # 按 HN Points 降序；无 Points 视为 0。同分时 Comments 高的在前，其余保持原始顺序。
    items.sort(key=lambda it: (-it.hn_points, -it.hn_comments, it.index))
    top_items = items[:TOP_N]

    rss_date = str(payload.get("date") or date.today().isoformat())
    lines = [f"今日 RSS 日期：{rss_date}"]
    lines.append(f"候选共 {len(top_items)} 条（按 HN Points 降序取 top-{TOP_N}，无热度计 0）：")
    lines.append("")
    for rank, item in enumerate(top_items, start=1):
        lines.append(f"{rank}. {item.to_candidate()}")
        lines.append("")
    lines.append(
        "最后提醒（务必遵守）：确认 writer 已写入文件后，你的最后一条回复必须只包含一个 "
        "JSON 对象（字段 topic_title/link/source/published_at/selection_reason/outline，"
        "outline 为 5-8 项字符串数组，topic_title/link/source/published_at 照抄候选列表对应条目），"
        "不要输出 Markdown 代码块或任何解释文字。"
    )
    return RssData(date=rss_date, top_items=top_items, prompt="\n".join(lines))
