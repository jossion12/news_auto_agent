"""统一 CLI：按 文件 + 类型（tech 技术 / news 新闻）+ 生成文章数 运行选题流水线。"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path

from src import local_web
from src.agents import (
    build_hotspot_agent,
    build_main_agent,
    build_writer_agent,
    run_hotspot_selection,
    run_selection,
    run_writer,
)
from src.selection import ground_picks, select_blog_topics, select_hotspot_topics
from src.config import PROJECT_ROOT, Settings, configure_langsmith, load_settings
from src.output import (
    apply_degraded_markers,
    enforce_date_prefix,
    ensure_frontmatter,
    snapshot_markdown_files,
    strip_body_h1,
)
from src.preprocess import load_hotspots, load_rss
from src.tools import make_output_path_tool, make_web_tools

logger = logging.getLogger("news_auto_agent")

KIND_TECH = "tech"
KIND_NEWS = "news"
KIND_ALIASES = {
    "tech": KIND_TECH,
    "技术": KIND_TECH,
    "blog": KIND_TECH,
    "news": KIND_NEWS,
    "新闻": KIND_NEWS,
    "hotspot": KIND_NEWS,
    "热点": KIND_NEWS,
}
DEFAULT_COUNT = {KIND_TECH: 1, KIND_NEWS: 2}


def parse_kind(value: str) -> str:
    """解析类型参数：tech/技术 -> tech；news/新闻/热点 -> news。"""
    kind = KIND_ALIASES.get(value.strip().lower())
    if kind is None:
        raise argparse.ArgumentTypeError(
            f"类型必须是 tech（技术）或 news（新闻/热点），当前为 {value!r}"
        )
    return kind


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"生成文章数必须是整数：{value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"生成文章数必须 >= 1：{value!r}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-agent",
        description="每日新闻自动选题并生成中文 Markdown：tech=技术博客，news=热点解读。",
    )
    parser.add_argument(
        "--type",
        required=True,
        type=parse_kind,
        help="流水线类型：tech/技术（RSS 技术博客）或 news/新闻/热点（热点解读）",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="输入 JSON 文件（相对项目根目录或绝对路径）；缺省用 RSS_FILE / HOTSPOTS_FILE",
    )
    parser.add_argument(
        "--count",
        type=positive_int,
        default=None,
        help="生成文章数（tech 默认 1，news 默认 2）",
    )
    return parser


def _resolve_data_file(settings: Settings, kind: str, file_arg: str | None) -> Path:
    if file_arg:
        path = Path(file_arg)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return settings.rss_file if kind == KIND_TECH else settings.hotspots_file


def _find_written_files(output_dir: Path, before: dict[str, str]) -> list[Path]:
    """从新增/内容变化的 md 中挑出本次运行的产物，按 mtime 新->旧排序。"""
    candidates: list[Path] = []
    for path in output_dir.glob("*.md"):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.name not in before or before[path.name] != digest:
            candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def _match_pick(picks: list, text: str, index: int):
    """按原文链接出现在文件内容中匹配 pick；匹配不到则按顺序占位。"""
    for pick in picks:
        if pick.link and pick.link in text:
            return pick
    return picks[min(index, len(picks) - 1)]


def _print_report(
    kind: str,
    picks: list,
    written_paths: list[Path],
    deepseek_usage: dict | None = None,
) -> None:
    label = "选题" if kind == KIND_TECH else "热点选稿"
    print()
    print(f"===== {label}结论 =====")
    for i, pick in enumerate(picks, start=1):
        print(f"[{i}] {pick.topic_title}")
        print(f"    链接：{pick.link or '(无原文链接)'}")
        print(f"    来源：{pick.source} | 发布：{pick.published_at}")
        print(f"    理由：{pick.selection_reason}")
        outline = getattr(pick, "outline", None)
        if outline:
            print(f"    大纲（{len(outline)} 点）：")
            for point in outline:
                print(f"      - {point}")
    for path in written_paths:
        try:
            relative = path.relative_to(PROJECT_ROOT)
        except ValueError:
            relative = path
        print(f"文章文件：{relative}")
    if deepseek_usage and deepseek_usage.get("calls"):
        total = deepseek_usage.get("prompt", 0) + deepseek_usage.get("completion", 0)
        print(
            f"DeepSeek 用量：{deepseek_usage['calls']} 次调用，"
            f"输入 {deepseek_usage.get('prompt', 0)}（缓存 {deepseek_usage.get('cached', 0)}）"
            f" / 输出 {deepseek_usage.get('completion', 0)} = {total} tokens"
        )


def _find_item(items, pick):
    """按链接（其次标题，最后模糊包含）在候选里找原条目，用于给 writer 传 RSS 摘要。

    选题模型常把标题润色/改写，精确匹配会失败；此时 writer 拿到空摘要会按
    「摘要为空则不写文件」的规则拒绝产出，因此这里必须补一层模糊兜底。
    """
    by_link = {item.link: item for item in items if item.link}
    item = by_link.get(getattr(pick, "link", ""))
    if item is not None:
        return item
    title = getattr(pick, "topic_title", "") or ""
    item = next((candidate for candidate in items if candidate.title == title), None)
    if item is not None:
        return item

    def normalize(text: str) -> str:
        return re.sub(r"[\s\W_]+", "", text or "").lower()

    target = normalize(title)
    if not target:
        return None
    best = None
    best_score = 0.0
    for candidate in items:
        current = normalize(candidate.title)
        if not current:
            continue
        if target in current or current in target:
            score = min(len(target), len(current)) / max(len(target), len(current))
        else:
            score = SequenceMatcher(None, target, current).ratio()
        if score > best_score:
            best_score = score
            best = candidate
    return best if best_score >= 0.6 else None


def run_pipeline(kind: str, data_file: Path, count: int) -> int:
    """执行一条流水线：预处理 -> 启动联网容器 -> 选题 -> 写作 -> 兜底 -> 报告 -> 停容器。"""
    settings = load_settings()
    configure_langsmith(settings)
    # DeepSeek 只负责选题：单次结构化输出，写作仍走本地 Ollama。
    direct_select = settings.select_provider == "deepseek"
    logger.info(
        "配置完成：OLLAMA=%s SELECT=%s（%s） WRITE=%s 联网=%s",
        settings.ollama_base_url,
        settings.select_model,
        "DeepSeek 直连选题" if direct_select else "本地 Agent 选题",
        settings.write_model,
        local_web.base_url(),
    )

    if kind == KIND_TECH:
        data = load_rss(data_file, count, json_hint=not direct_select)
        logger.info("RSS 预处理完成：date=%s，候选=%d 条，选 %d 题", data.date, len(data.top_items), count)
    else:
        data = load_hotspots(data_file, count, json_hint=not direct_select)
        logger.info("热点预处理完成：date=%s，候选=%d 条，选 %d 则", data.date, len(data.items), count)

    # 输出目录 mkdir；快照用于运行后定位 writer 落盘的文件。
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    before = snapshot_markdown_files(settings.output_dir)

    web_state = {"ok": False, "failures": 0}
    existing_names = set(before)
    deepseek_usage: dict[str, int] = {}

    # 本地联网服务（SearXNG + Crawl4AI）：按需启动，跑完停掉。
    with local_web.session():
        writer_tools = [
            *make_web_tools(settings, web_state),
            make_output_path_tool(
                project_root=PROJECT_ROOT,
                output_dir=settings.output_dir,
                rss_date=data.date,
                existing_names=existing_names,
            ),
        ]

        selection = None
        if direct_select:
            try:
                if kind == KIND_TECH:
                    selection, deepseek_usage = select_blog_topics(settings, data.prompt, count)
                    ground_picks(selection.picks, data.top_items)
                    items = data.top_items
                else:
                    selection, deepseek_usage = select_hotspot_topics(settings, data.prompt, count)
                    ground_picks(selection.picks, data.items)
                    items = data.items
            except Exception as exc:  # noqa: BLE001 - 选题失败时回退本地模型，保证日出稿
                logger.error("DeepSeek 直连选题失败：%s；回退本地模型 Agent 流程", exc)
                selection = None

        if selection is not None:
            writer_agent = build_writer_agent(
                settings, writer_tools, hotspot=(kind != KIND_TECH)
            )
            for index, pick in enumerate(selection.picks):
                item = _find_item(items, pick)
                run_writer(
                    writer_agent,
                    pick,
                    item,
                    hotspot=(kind != KIND_TECH),
                    thread_id=f"{settings.thread_id}-{data.date}-{index}",
                )
                existing_names.update(path.name for path in settings.output_dir.glob("*.md"))
            picks = selection.picks
        else:
            # 兜底：原来的 deepagents 主 Agent；DeepSeek 不可用时改为本地模型选题。
            local_settings = (
                replace(settings, select_provider="ollama")
                if settings.select_provider == "deepseek"
                else settings
            )
            if kind == KIND_TECH:
                agent = build_main_agent(local_settings, writer_tools, count)
                _, selection = run_selection(agent, data.prompt, settings.thread_id, count)
                items = data.top_items
            else:
                agent = build_hotspot_agent(local_settings, writer_tools, count)
                _, selection = run_hotspot_selection(
                    agent, data.prompt, settings.hotspot_thread_id, count
                )
                items = data.items
            picks = selection.picks

        written_paths = _find_written_files(settings.output_dir, before)
        if len(written_paths) < count:
            raise RuntimeError(
                f"writer 应产出 {count} 个 Markdown 文件，实际找到 {len(written_paths)} 个，"
                "请检查上文 warning 日志"
            )
        written_paths = written_paths[:count]

        for index, path in enumerate(written_paths):
            path = enforce_date_prefix(path, data.date)
            written_paths[index] = path
            text = path.read_text(encoding="utf-8")
            pick = _match_pick(picks, text, index)
            selected_item = next(
                (it for it in items if it.link and it.link == pick.link),
                None,
            )
            # content schema 兜底：文件开头保证 title/date/tags/summary frontmatter。
            if ensure_frontmatter(path, data.date, pick, selected_item):
                logger.info("已补全 frontmatter：%s", path)
            # FR3 兜底：联网工具全部失败时，代码层补警告行与参考链接段。
            if not web_state["ok"]:
                apply_degraded_markers(path, pick, selected_item)
                logger.warning("联网工具不可用，已按 FR3 降级处理：%s", path)
            # SEO 兜底：正文里不要再写 H1（页面模板会用 title 渲染 h1）
            if strip_body_h1(path):
                logger.info("已去除正文重复 H1：%s", path)

        for path in written_paths:
            logger.info("文章已生成：%s", path)
        _print_report(kind, picks, written_paths, deepseek_usage)
    return 0

def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，打印报告时遇到 ö/・ 等字符会 UnicodeEncodeError，
    # 导致文章已生成却以退出码 1 结束。保留原编码，仅把不可编码字符替换掉。
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (ValueError, OSError):  # 非交互流（如管道）不支持时忽略
            pass
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    count = args.count if args.count is not None else DEFAULT_COUNT[args.type]
    data_file = None
    try:
        settings = load_settings()
        data_file = _resolve_data_file(settings, args.type, args.file)
        if not data_file.is_file():
            raise FileNotFoundError(f"输入文件不存在：{data_file}")
        return run_pipeline(args.type, data_file, count)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
    except Exception:  # noqa: BLE001 - 入口兜底：任何异常均以非零退出
        logger.exception("程序异常退出")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
