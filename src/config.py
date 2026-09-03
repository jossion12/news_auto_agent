"""配置加载：优先读取项目根目录的 `.env`，缺失时使用与 `.env.example` 一致的内置默认值。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 = 本文件所在目录的上一级（src/config.py -> 项目根）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 内置默认值必须与 .env.example 保持一致；只允许读取 .env，禁止读取 .env.example 本身。
ENV_DEFAULTS: dict[str, str] = {
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_API_KEY": "",
    "SELECT_MODEL": "qwen3.5:9b",
    "WRITE_MODEL": "qwen3.6:27b",
    "SELECT_PROVIDER": "ollama",
    "WRITE_PROVIDER": "ollama",
    "DEEPSEEK_API_KEY": "",
    "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
    "DEEPSEEK_MODEL": "deepseek-v4-flash",
    "SELECT_NUM_CTX": "16384",
    "WRITE_NUM_CTX": "32768",
    "SELECT_TEMPERATURE": "0.2",
    "WRITE_TEMPERATURE": "0.7",
    "NUM_PREDICT": "4096",
    "RSS_FILE": "sample/2026-09-03_tech_rss.json",
    "OUTPUT_DIR": "output",
    "THREAD_ID": "news-blog",
    "LANGSMITH_API_KEY": "",
    "LANGSMITH_PROJECT": "",
}

SUPPORTED_PROVIDERS = ("ollama", "deepseek")


@dataclass(frozen=True)
class Settings:
    """运行时配置（路径均已在项目根目录基准下解析）。"""

    ollama_base_url: str
    ollama_api_key: str
    select_model: str
    write_model: str
    select_provider: str
    write_provider: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    select_num_ctx: int
    write_num_ctx: int
    select_temperature: float
    write_temperature: float
    num_predict: int
    rss_file: Path
    output_dir: Path
    thread_id: str
    langsmith_api_key: str
    langsmith_project: str | None


def _env_str(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        value = ENV_DEFAULTS[name]
    return value.strip()


def _env_int(name: str) -> int:
    try:
        return int(_env_str(name))
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 不是合法整数：{os.getenv(name)!r}") from exc


def _env_float(name: str) -> float:
    try:
        return float(_env_str(name))
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 不是合法浮点数：{os.getenv(name)!r}") from exc


def _resolve_project_path(value: str) -> Path:
    """所有相对路径一律以项目根目录为基准；绝对路径按原样使用。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _provider(name: str) -> str:
    value = _env_str(name).lower()
    if value not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"环境变量 {name} 必须是 {SUPPORTED_PROVIDERS} 之一，当前为 {value!r}"
        )
    return value


def load_settings() -> Settings:
    """加载 `.env`（存在时覆盖默认值），并解析配置。"""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    langsmith_api_key = _env_str("LANGSMITH_API_KEY")
    langsmith_project = _env_str("LANGSMITH_PROJECT")
    select_provider = _provider("SELECT_PROVIDER")
    write_provider = _provider("WRITE_PROVIDER")
    deepseek_api_key = _env_str("DEEPSEEK_API_KEY")
    if (select_provider == "deepseek" or write_provider == "deepseek") and not deepseek_api_key:
        raise ValueError(
            "SELECT_PROVIDER/WRITE_PROVIDER 为 deepseek 但未设置 DEEPSEEK_API_KEY，"
            "请在 .env 中填写"
        )

    return Settings(
        ollama_base_url=_env_str("OLLAMA_BASE_URL"),
        ollama_api_key=_env_str("OLLAMA_API_KEY"),
        select_model=_env_str("SELECT_MODEL"),
        write_model=_env_str("WRITE_MODEL"),
        select_provider=select_provider,
        write_provider=write_provider,
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=_env_str("DEEPSEEK_BASE_URL"),
        deepseek_model=_env_str("DEEPSEEK_MODEL"),
        select_num_ctx=_env_int("SELECT_NUM_CTX"),
        write_num_ctx=_env_int("WRITE_NUM_CTX"),
        select_temperature=_env_float("SELECT_TEMPERATURE"),
        write_temperature=_env_float("WRITE_TEMPERATURE"),
        num_predict=_env_int("NUM_PREDICT"),
        rss_file=_resolve_project_path(_env_str("RSS_FILE")),
        output_dir=_resolve_project_path(_env_str("OUTPUT_DIR")),
        thread_id=_env_str("THREAD_ID"),
        langsmith_api_key=langsmith_api_key,
        langsmith_project=langsmith_project or None,
    )


def configure_langsmith(settings: Settings) -> None:
    """仅当 LANGSMITH_API_KEY 有值时启用 LangSmith 追踪。"""
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_TRACING"] = "true"
        if settings.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ.pop("LANGSMITH_API_KEY", None)
