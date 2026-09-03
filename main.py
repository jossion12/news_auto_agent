"""news-auto-agent 入口：每日 RSS 自动选题并生成中文技术博客。"""

from __future__ import annotations

import logging
import sys

from src.agents import build_main_agent, run_selection
from src.config import PROJECT_ROOT, Settings, configure_langsmith, load_settings
from src.output import (
    apply_degraded_markers,
    enforce_date_prefix,
    ensure_frontmatter,
    find_written_file,
    snapshot_markdown_files,
)
from src.preprocess import load_rss
from src.tools import make_output_path_tool, make_web_tools

logger = logging.getLogger("news_auto_agent")


def _print_report(settings: Settings, selection, written_path) -> None:
    print()
    print("===== 选题结论 =====")
    print(f"主题：{selection.topic_title}")
    print(f"链接：{selection.link}")
    print(f"来源：{selection.source}")
    print(f"发布时间：{selection.published_at}")
    print(f"选题理由：{selection.selection_reason}")
    print(f"大纲（{len(selection.outline)} 点）：")
    for point in selection.outline:
        print(f"  - {point}")
    try:
        relative = written_path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = written_path
    print(f"文章文件：{written_path}")
    print(f"（相对项目根目录：{relative}）")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        settings = load_settings()
        configure_langsmith(settings)
        logger.info(
            "配置完成：OLLAMA=%s SELECT=%s WRITE=%s",
            settings.ollama_base_url,
            settings.select_model,
            settings.write_model,
        )

        rss = load_rss(settings.rss_file)
        logger.info("RSS 预处理完成：date=%s，候选=%d 条", rss.date, len(rss.top_items))

        # FR4：输出目录 mkdir；快照用于运行后定位 writer 落盘的文件。
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        before = snapshot_markdown_files(settings.output_dir)

        web_state = {"ok": False, "failures": 0}
        writer_tools = [
            *make_web_tools(settings, web_state),
            make_output_path_tool(
                project_root=PROJECT_ROOT,
                output_dir=settings.output_dir,
                rss_date=rss.date,
                existing_names=set(before),
            ),
        ]
        agent = build_main_agent(settings, writer_tools)

        _, selection = run_selection(agent, rss.prompt, settings.thread_id)

        written_path = find_written_file(settings.output_dir, before)
        if written_path is None:
            raise RuntimeError("writer 未产出新的 Markdown 文件，请检查上文 warning 日志")

        # FR4 兜底：强制 <date>- 文件名前缀；writer 自命名也能保证规范。
        written_path = enforce_date_prefix(written_path, rss.date)

        selected_item = next(
            (it for it in rss.top_items if it.link == selection.link),
            None,
        )

        # content schema 兜底：文件开头保证 title/date/tags/summary frontmatter。
        if ensure_frontmatter(written_path, rss.date, selection, selected_item):
            logger.info("已补全 frontmatter：%s", written_path)

        # FR3 兜底：联网工具全部失败时，代码层补警告行与参考链接段。
        if not web_state["ok"]:
            apply_degraded_markers(written_path, selection, selected_item)
            logger.warning("联网工具不可用，已按 FR3 降级处理：%s", written_path)

        logger.info("文章已生成：%s", written_path)
        _print_report(settings, selection, written_path)
        return 0

    except FileNotFoundError as exc:
        logger.error("%s", exc)
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
    except Exception:  # noqa: BLE001 - 入口兜底：任何异常均以非零退出
        logger.exception("程序异常退出")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
