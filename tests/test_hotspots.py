"""load_hotspots / HotspotSelectionResult 的代码层测试（纯本地，不联网）。"""

from __future__ import annotations

import json

import pytest

from src.models import HotspotSelectionResult
from src.preprocess import HOTSPOT_PER_SOURCE, load_hotspots


def _item(index: int, source: str, title: str = "", link: str = "") -> dict:
    return {
        "title": title or f"{source} 新闻 {index}",
        "link": link,
        "source": source,
        "published_at": "Thu, 03 Sep 2026 08:47:29 +0000",
        "summary": f"摘要 {index}",
        "category": "AI",
    }


def _write_payload(tmp_path, sources, data_date="2026-09-03") -> object:
    path = tmp_path / "hotspots.json"
    path.write_text(
        json.dumps({"date": data_date, "sources": sources}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_load_hotspots_caps_per_source(tmp_path):
    items = [_item(i, "IT之家") for i in range(HOTSPOT_PER_SOURCE + 3)]
    path = _write_payload(tmp_path, [{"source": "IT之家", "items": items}])
    data = load_hotspots(path)
    assert data.date == "2026-09-03"
    assert len(data.items) == HOTSPOT_PER_SOURCE
    assert "候选共 5 条" in data.prompt
    assert '"picks"' in data.prompt


def test_load_hotspots_keeps_empty_link_items(tmp_path):
    sources = [
        {"source": "源A", "items": [_item(1, "源A", link="https://a.com/1")]},
        {"source": "源B", "items": [_item(2, "源B", link="")]},
    ]
    data = load_hotspots(_write_payload(tmp_path, sources))
    assert [it.link for it in data.items] == ["https://a.com/1", ""]
    assert "无原文链接" in data.prompt


def test_load_hotspots_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hotspots(tmp_path / "nope.json")


def test_load_hotspots_bad_structure(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"date": "2026-09-03"}', encoding="utf-8")
    with pytest.raises(ValueError, match="sources"):
        load_hotspots(path)


def test_hotspot_selection_result_min_one_pick():
    pick = {
        "topic_title": "标题",
        "link": "https://a.com/1",
        "source": "源A",
        "published_at": "Thu, 03 Sep 2026 08:47:29 +0000",
        "selection_reason": "理由",
    }
    # 数量约束在运行层 check_pick_count 按 count 校验，模型层只要求 >= 1。
    assert len(HotspotSelectionResult.model_validate({"picks": [pick]}).picks) == 1
    assert len(HotspotSelectionResult.model_validate({"picks": [pick, pick, pick]}).picks) == 3
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        HotspotSelectionResult.model_validate({"picks": []})


def test_check_pick_count():
    from src.agents import check_pick_count

    pick = {
        "topic_title": "标题",
        "link": "https://a.com/1",
        "source": "源A",
        "published_at": "Thu, 03 Sep 2026 08:47:29 +0000",
        "selection_reason": "理由",
    }
    two = HotspotSelectionResult.model_validate({"picks": [pick, pick]})
    assert check_pick_count(two, 2) is None
    assert check_pick_count(two, 3) is not None


def test_load_hotspots_prompt_uses_count(tmp_path):
    sources = [{"source": "源A", "items": [_item(1, "源A", link="https://a.com/1")]}]
    data = load_hotspots(_write_payload(tmp_path, sources), count=3)
    assert "picks 必须恰好 3 个" in data.prompt


def test_cli_parse_kind_aliases():
    from cli import KIND_NEWS, KIND_TECH, parse_kind, positive_int

    assert parse_kind("tech") == KIND_TECH
    assert parse_kind("技术") == KIND_TECH
    assert parse_kind("news") == KIND_NEWS
    assert parse_kind("新闻") == KIND_NEWS
    assert parse_kind("热点") == KIND_NEWS
    with pytest.raises(Exception):  # noqa: B017 - argparse.ArgumentTypeError
        parse_kind("sports")
    assert positive_int("2") == 2
    with pytest.raises(Exception):  # noqa: B017 - argparse.ArgumentTypeError
        positive_int("0")
