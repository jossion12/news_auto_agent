"""LLM token 用量统计（LangChain callback）。

只做一件事：把每次 LLM 调用的 prompt / completion / cached tokens 累计起来，
按模型分桶打印，方便对比优化前后的 DeepSeek 消耗。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class UsageTracker(BaseCallbackHandler):
    """累计 token 用量；按模型名分桶（Ollama / DeepSeek 分开）。"""

    def __init__(self) -> None:
        super().__init__()
        # model -> {"prompt": int, "completion": int, "cached": int, "calls": int}
        self.by_model: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "cached": 0, "calls": 0}
        )

    # LangChain 的 on_llm_end 是同步回调，内部用同步方法即可。
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ANN401
        model = "unknown"
        usage: dict[str, Any] | None = None

        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            model = str(llm_output.get("model_name") or llm_output.get("model") or model)
            raw = llm_output.get("token_usage") or llm_output.get("usage")
            if isinstance(raw, dict):
                usage = raw

        if usage is None:
            for generation_list in getattr(response, "generations", []) or []:
                for generation in generation_list:
                    message = getattr(generation, "message", None)
                    meta = getattr(message, "usage_metadata", None)
                    if isinstance(meta, dict):
                        usage = {
                            "prompt_tokens": meta.get("input_tokens", 0),
                            "completion_tokens": meta.get("output_tokens", 0),
                            "total_tokens": meta.get("total_tokens", 0),
                        }
                        model = str(
                            getattr(message, "response_metadata", {}).get("model_name")
                            or model
                        )
                        break
                if usage is not None:
                    break

        if not usage:
            return

        bucket = self.by_model[model or "unknown"]
        bucket["calls"] += 1
        bucket["prompt"] += int(usage.get("prompt_tokens") or 0)
        bucket["completion"] += int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            bucket["cached"] += int(details.get("cached_tokens") or 0)

    def summary(self) -> dict[str, dict[str, int]]:
        return {model: dict(values) for model, values in self.by_model.items()}

    def deepseek_summary(self) -> dict[str, int]:
        total = {"prompt": 0, "completion": 0, "cached": 0, "calls": 0}
        for model, values in self.by_model.items():
            if "deepseek" in model.lower():
                for key in total:
                    total[key] += values.get(key, 0)
        return total

    def log(self, label: str) -> dict[str, int]:
        """打印各模型用量，返回 DeepSeek 汇总。"""
        for model, values in self.by_model.items():
            logger.info(
                "LLM 用量[%s] %s：调用 %d 次，输入 %d（缓存命中 %d）/ 输出 %d tokens",
                label,
                model or "unknown",
                values["calls"],
                values["prompt"],
                values["cached"],
                values["completion"],
            )
        ds = self.deepseek_summary()
        if ds["calls"]:
            logger.info(
                "DeepSeek 合计[%s]：%d 次调用，输入 %d（缓存 %d）/ 输出 %d = %d tokens",
                label,
                ds["calls"],
                ds["prompt"],
                ds["cached"],
                ds["completion"],
                ds["prompt"] + ds["completion"],
            )
        return ds
