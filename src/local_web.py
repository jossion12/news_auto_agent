"""本地联网服务（SearXNG + Crawl4AI 封装层）客户端 + Docker 生命周期管理。

替代原来 ollama.com 的 web_search / web_fetch：
- 检索与正文抓取都打到本地 `http://127.0.0.1:8000`（docker compose 里的 api 服务）
- 运行前自动 `docker compose up -d`，运行结束自动 `docker compose stop`
  （只有“本次运行把它拉起来”才会停，本来就开着的不动）

环境变量（都能在 .env 里配）：
    LOCAL_WEB_BASE_URL      默认 http://127.0.0.1:8000
    LOCAL_WEB_COMPOSE_FILE  默认 searxng-crawl4ai 的 docker-compose.yml 绝对路径
    LOCAL_WEB_AUTOSTART     1=自动启停容器，0=只调用不启停（默认 1）
    LOCAL_WEB_KEEPALIVE     1=运行结束不停容器（默认 0）
    LOCAL_WEB_API_TOKEN     封装层若设了 API_TOKEN，这里填同值
    LOCAL_WEB_START_TIMEOUT 启动等待秒数，默认 150
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Iterator

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_COMPOSE_FILE = (
    r"D:\project\private\new_life\code\local_flow\searxng-crawl4ai\docker-compose.yml"
)

_SEARCH_TIMEOUT = 120.0
_FETCH_TIMEOUT = 180.0
_HEALTH_TIMEOUT = 5.0


class LocalWebError(RuntimeError):
    """本地联网服务不可用 / 返回异常。"""


def _env(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


def base_url() -> str:
    return _env("LOCAL_WEB_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def compose_file() -> str:
    return _env("LOCAL_WEB_COMPOSE_FILE", DEFAULT_COMPOSE_FILE)


def _autostart() -> bool:
    return _env("LOCAL_WEB_AUTOSTART", "1").lower() in ("1", "true", "yes", "on")


def _keepalive() -> bool:
    return _env("LOCAL_WEB_KEEPALIVE", "0").lower() in ("1", "true", "yes", "on")


def _start_timeout() -> float:
    try:
        return float(_env("LOCAL_WEB_START_TIMEOUT", "150"))
    except ValueError:
        return 150.0


def _headers() -> dict[str, str]:
    token = _env("LOCAL_WEB_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def is_healthy(timeout: float = _HEALTH_TIMEOUT) -> bool:
    """GET /health 返回 200 且 status=ok 才算就绪。"""
    try:
        resp = httpx.get(
            f"{base_url()}/health", headers=_headers(), timeout=timeout, trust_env=False
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        return isinstance(data, dict) and data.get("status") == "ok"
    except Exception:  # noqa: BLE001 - 健康检查失败就是不健康
        return False


def _run_docker(args: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    docker = shutil.which("docker")
    if not docker:
        raise LocalWebError("找不到 docker 命令：请确认 Docker Desktop 已启动并在 PATH 中")
    cmd = [docker, *args]
    logger.info("执行：%s", " ".join(cmd))
    try:
        return subprocess.run(
            cmd,
            cwd=os.path.dirname(compose_file()) or None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:  # docker 存在但无法执行
        raise LocalWebError(f"无法执行 docker：{exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise LocalWebError(f"docker 命令超时（{timeout:.0f}s）：{' '.join(cmd)}") from exc


def _wait_until_healthy() -> None:
    deadline = time.monotonic() + _start_timeout()
    while time.monotonic() < deadline:
        if is_healthy():
            logger.info("本地联网服务已就绪：%s", base_url())
            return
        time.sleep(2)
    raise LocalWebError(
        f"容器已启动，但 {_start_timeout():.0f}s 内 {base_url()}/health 未就绪；"
        f"请检查：docker compose -f \"{compose_file()}\" logs api"
    )


def start() -> bool:
    """确保容器在跑。返回 True 表示“本次启动的”，False 表示原本就健康。"""
    if is_healthy():
        logger.info("本地联网服务已在运行：%s", base_url())
        return False

    cf = compose_file()
    if not os.path.isfile(cf):
        raise LocalWebError(f"找不到 compose 文件：{cf}（检查 LOCAL_WEB_COMPOSE_FILE）")

    proc = _run_docker(["compose", "-f", cf, "up", "-d"], timeout=420)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise LocalWebError(
            f"docker compose up 失败（exit={proc.returncode}）；"
            f"请确认 Docker Desktop 已运行。输出：{detail}"
        )
    _wait_until_healthy()
    return True


def stop() -> None:
    """停掉容器（保留容器，下次 up 很快）。失败只记 warning，不影响主流程。"""
    cf = compose_file()
    try:
        proc = _run_docker(["compose", "-f", cf, "stop"], timeout=180)
        if proc.returncode != 0:
            logger.warning("docker compose stop 返回 %s：%s", proc.returncode, (proc.stderr or "").strip()[-400:])
        else:
            logger.info("本地联网服务已停止（docker compose stop）")
    except Exception as exc:  # noqa: BLE001 - 停容器失败不应让整条流水线失败
        logger.warning("停止本地联网服务失败：%s", exc)


@contextmanager
def session() -> Iterator[None]:
    """在 with 块内保证容器可用，退出时按需停止。"""
    if not _autostart():
        logger.info("LOCAL_WEB_AUTOSTART=0，跳过容器启停（直接用现有服务）")
        yield
        return

    started = False
    try:
        started = start()
    except Exception as exc:  # noqa: BLE001 - 起不来时按 FR3 降级，不中断生成
        logger.error("本地联网服务启动失败：%s", exc)
        logger.error("本次 web_search/web_fetch 将不可用，文章会按 FR3 降级生成")

    try:
        yield
    finally:
        if started and not _keepalive():
            stop()


def _post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        resp = httpx.post(
            f"{base_url()}{path}",
            json=payload,
            headers=_headers(),
            timeout=timeout,
            trust_env=False,
        )
    except httpx.HTTPError as exc:
        raise LocalWebError(f"连不上本地联网服务 {base_url()}：{exc}") from exc

    if resp.status_code >= 400:
        raise LocalWebError(f"{path} 返回 HTTP {resp.status_code}：{resp.text[:300]}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise LocalWebError(f"{path} 返回的不是 JSON：{resp.text[:200]}") from exc
    if not isinstance(data, dict):
        raise LocalWebError(f"{path} 返回结构异常：{type(data).__name__}")
    return data


def search(query: str, max_results: int = 3) -> dict[str, Any]:
    """搜索，返回 {"results": [{title, url, content}]}（与 Ollama web_search 同形）。"""
    return _post(
        "/api/web_search",
        {"query": query, "max_results": max_results},
        timeout=_SEARCH_TIMEOUT,
    )


def fetch(url: str) -> dict[str, Any]:
    """抓取单页正文，返回 {title, content, links, success, error, ...}。"""
    data = _post("/api/web_fetch", {"url": url}, timeout=_FETCH_TIMEOUT)
    if not data.get("success", True):
        raise LocalWebError(data.get("error") or "抓取失败（无正文）")
    return data
