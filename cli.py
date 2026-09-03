"""统一 CLI：按 文件 + 类型（tech 技术 / news 新闻）+ 生成文章数 运行选题流水线。"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from src.agents import (
    build_hotspot_agent,
    build_main_agent,
    run_hotspot_selection,
    run_selection,
)
from src.config import PROJECT_ROOT, Settings, configure_langsmith, load_settings
from src.output import (
    apply_degraded_markers,
    enforce_date_prefix,
    ensure_frontmatter,
    snapshot_markdown_files,
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


def _print_report(kind: str, picks: list, written_paths: list[Path]) -> None:
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


def run_pipeline(kind: str, data_file: Path, count: int) -> int:
    """执行一条流水线：预处理 -> Agent 运行 -> 兜底 -> 报告。返回退出码。"""
    settings = load_settings()
    configure_langsmith(settings)
    logger.info(
        "配置完成：OLLAMA=%s SELECT=%s WRITE=%s",
        settings.ollama_base_url,
        settings.select_model,
        settings.write_model,
    )

    if kind == KIND_TECH:
        data = load_rss(data_file, count)
        logger.info("RSS 预处理完成：date=%s，候选=%d 条，选 %d 题", data.date, len(data.top_items), count)
    else:
        data = load_hotspots(data_file, count)
        logger.info("热点预处理完成：date=%s，候选=%d 条，选 %d 则", data.date, len(data.items), count)

    # 输出目录 mkdir；快照用于运行后定位 writer 落盘的文件。
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    before = snapshot_markdown_files(settings.output_dir)

    web_state = {"ok": False, "failures": 0}
    writer_tools = [
        *make_web_tools(settings, web_state),
        make_output_path_tool(
            project_root=PROJECT_ROOT,
            output_dir=settings.output_dir,
            rss_date=data.date,
            existing_names=set(before),
        ),
    ]

    if kind == KIND_TECH:
        agent = build_main_agent(settings, writer_tools, count)
        _, selection = run_selection(agent, data.prompt, settings.thread_id, count)
        items = data.top_items
    else:
        agent = build_hotspot_agent(settings, writer_tools, count)
        _, selection = run_hotspot_selection(
            agent, data.prompt, settings.hotspot_thread_id, count
        )
        items = data.items

    written_paths = _find_written_files(settings.output_dir, before)
    if len(written_paths) < count:
        raise RuntimeError(
            f"writer 应产出 {count} 个 Markdown 文件，实际找到 {len(written_paths)} 个，"
            "请检查上文 warning 日志"
        )
    written_paths = written_paths[:count]

    picks = selection.picks
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

    for path in written_paths:
        logger.info("文章已生成：%s", path)
    _print_report(kind, picks, written_paths)
    return 0


def main(argv: list[str] | None = None) -> int:
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
