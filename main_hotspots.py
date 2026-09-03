"""热点解读入口（等价于 cli.py --type news --count 2）。"""

from __future__ import annotations

import logging
import sys

from cli import KIND_NEWS, run_pipeline
from src.config import load_settings

logger = logging.getLogger("news_hotspots_agent")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        settings = load_settings()
        return run_pipeline(KIND_NEWS, settings.hotspots_file, count=2)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
    except Exception:  # noqa: BLE001 - 入口兜底：任何异常均以非零退出
        logger.exception("程序异常退出")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
