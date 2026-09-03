"""ensure_frontmatter / apply_degraded_markers 的代码层兜底测试（纯本地，不联网）。"""

from __future__ import annotations

from types import SimpleNamespace

from src.output import (
    DEGRADED_WARNING,
    apply_degraded_markers,
    ensure_frontmatter,
)

SELECTION = SimpleNamespace(
    topic_title="Gemini 3.8 Flash 发布",
    link="https://example.com/gemini",
    source="Example Source",
    selection_reason="热度断层第一",
)
ITEM = SimpleNamespace(summary="summary https://example.com/gemini", category="AI")


def _write(tmp_path, text: str):
    path = tmp_path / "2026-09-03-test.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_prepends_frontmatter_when_missing(tmp_path):
    path = _write(tmp_path, "# 标题\n\n第一段摘要内容。\n\n## 章节\n")
    assert ensure_frontmatter(path, "2026-09-03", SELECTION, ITEM) is True
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "title: 标题\n" in text
    assert "date: 2026-09-03\n" in text
    assert "tags: ['AI']\n" in text
    assert "summary: 第一段摘要内容。\n" in text
    assert "# 标题" in text


def test_keeps_existing_complete_frontmatter(tmp_path):
    original = (
        "---\ntitle: T\ndate: 2026-09-03\ntags: ['AI']\nsummary: S\n---\n\n# 标题\n"
    )
    path = _write(tmp_path, original)
    assert ensure_frontmatter(path, "2026-09-03", SELECTION, ITEM) is False
    assert path.read_text(encoding="utf-8") == original


def test_fills_only_missing_keys(tmp_path):
    path = _write(tmp_path, "---\ntitle: T\n---\n\n# 标题\n\n正文。\n")
    assert ensure_frontmatter(path, "2026-09-03", SELECTION, ITEM) is True
    text = path.read_text(encoding="utf-8")
    assert "title: T\n" in text
    assert "date: 2026-09-03\n" in text
    assert "tags: ['AI']\n" in text
    assert "summary:" in text


def test_fallback_uses_category_and_h1(tmp_path):
    item = SimpleNamespace(summary="", category="大模型")
    path = _write(tmp_path, "# 正式标题\n\n> 引用不算摘要。\n\n真正摘要段落。\n")
    ensure_frontmatter(path, "2026-09-03", SELECTION, item)
    text = path.read_text(encoding="utf-8")
    assert "title: 正式标题\n" in text
    assert "tags: ['大模型']\n" in text
    assert "summary: 真正摘要段落。\n" in text


def test_yaml_quotes_special_chars(tmp_path):
    path = _write(tmp_path, '# 标题: 带冒号 "引号"\n\n摘要。\n')
    ensure_frontmatter(path, "2026-09-03", SELECTION, ITEM)
    text = path.read_text(encoding="utf-8")
    assert 'title: "标题: 带冒号 \\"引号\\""' in text


def test_degraded_warning_goes_after_frontmatter(tmp_path):
    path = _write(tmp_path, "---\ntitle: T\ndate: 2026-09-03\ntags: ['AI']\nsummary: S\n---\n\n# 标题\n\n正文 https://example.com/gemini。\n")
    apply_degraded_markers(path, SELECTION, ITEM)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert text.index(DEGRADED_WARNING) > text.index("\n---")
    assert "## 参考链接" in text


def test_degraded_warning_at_top_without_frontmatter(tmp_path):
    path = _write(tmp_path, "# 标题\n\n正文 https://example.com/gemini。\n")
    apply_degraded_markers(path, SELECTION, ITEM)
    text = path.read_text(encoding="utf-8")
    assert text.startswith(DEGRADED_WARNING)
