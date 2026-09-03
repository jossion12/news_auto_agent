"""RSS 技术博客入口（等价于 cli.py --type tech --count 1）。"""

from __future__ import annotations

import logging
import sys

from cli import KIND_TECH, run_pipeline
from src.config import load_settings

logger = logging.getLogger("news_auto_agent")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        settings = load_settings()
        return run_pipeline(KIND_TECH, settings.rss_file, count=1)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
    except Exception:  # noqa: BLE001 - 入口兜底：任何异常均以非零退出
        logger.exception("程序异常退出")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
