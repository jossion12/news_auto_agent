"""输出路径/文件名规则：slug、冲突追加、项目根目录路径解析。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

SLUG_MIN_LENGTH = 3
SLUG_MAX_LENGTH = 40
FALLBACK_SLUG = "tech-blog"

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
_VALID_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def make_slug(text: str) -> str:
    """小写字母/数字/短横线，3-40 字符；中文或非法文本回退为固定英文 slug。"""
    slug = _SLUG_RE.sub("-", text.strip()).lower().strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if len(slug) > SLUG_MAX_LENGTH:
        head = slug[:SLUG_MAX_LENGTH].rstrip("-")
        cut = head.rfind("-")
        if cut >= SLUG_MIN_LENGTH - 1:
            head = head[:cut]
        slug = head.strip("-")
    if len(slug) < SLUG_MIN_LENGTH or not _VALID_SLUG_RE.fullmatch(slug):
        return FALLBACK_SLUG
    return slug


def build_output_path(
    project_root: Path,
    output_dir: Path,
    rss_date: str,
    topic_title: str,
    english_slug_hint: str = "",
    existing_names: set[str] | None = None,
) -> tuple[Path, str]:
    """返回 (磁盘绝对路径, 面向 deepagents 的虚拟路径 /output/<date>-<slug>.md)。

    已存在的同名文件视为冲突并追加 -2、-3；目录自动创建。
    """
    slug = make_slug(english_slug_hint or topic_title)
    stem = f"{rss_date}-{slug}"

    if existing_names is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_names = {path.name for path in output_dir.glob("*.md")}

    filename = f"{stem}.md"
    counter = 2
    while filename in existing_names:
        filename = f"{stem}-{counter}.md"
        counter += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    abs_path = output_dir / filename
    relative = abs_path.relative_to(project_root).as_posix()
    return abs_path, f"/{relative}"


def snapshot_markdown_files(output_dir: Path) -> dict[str, str]:
    """快照输出目录中的 md 文件名 -> sha256，用于识别本次运行新增/被覆盖的文件。"""
    if not output_dir.is_dir():
        return {}
    result: dict[str, str] = {}
    for path in output_dir.glob("*.md"):
        if path.is_file():
            result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def find_written_file(
    output_dir: Path,
    before: dict[str, str],
) -> Path | None:
    """从新增/内容变化的 md 中挑选本次运行的产物（优先日期前缀，其次最新 mtime）。"""
    if not output_dir.is_dir():
        return None
    candidates: list[Path] = []
    for path in output_dir.glob("*.md"):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.name not in before or before[path.name] != digest:
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


DEGRADED_WARNING = "[警告：联网工具不可用，本文仅基于 RSS 摘要生成]"

FRONTMATTER_KEYS = ("title", "date", "tags", "summary")


def ensure_frontmatter(path: Path, rss_date: str, selection, item) -> bool:
    """content schema 兜底：文件开头保证有 title/date/tags/summary frontmatter。

    writer 已写全则不动；frontmatter 存在但缺字段时只补缺失项；完全没有时按选题信息
    整体前置生成。返回文件是否有改动。
    """
    text = path.read_text(encoding="utf-8")
    fallback = _fallback_frontmatter(text, rss_date, selection, item)

    stripped = text.lstrip()
    if stripped.startswith("---"):
        end = stripped.find("\n---", 3)
        if end != -1:
            block = stripped[3:end]
            rest = stripped[end + len("\n---"):]
            missing = [
                key for key in FRONTMATTER_KEYS
                if not re.search(rf"^\s*{key}\s*:", block, re.M)
            ]
            if not missing:
                return False
            lines = [line for line in block.strip("\n").splitlines()]
            lines += [f"{key}: {fallback[key]}" for key in missing]
            text = "---\n" + "\n".join(lines) + "\n---" + rest
            path.write_text(text, encoding="utf-8")
            return True

    frontmatter = (
        "---\n"
        + "\n".join(f"{key}: {fallback[key]}" for key in FRONTMATTER_KEYS)
        + "\n---\n\n"
    )
    path.write_text(frontmatter + stripped, encoding="utf-8")
    return True


def _fallback_frontmatter(text: str, rss_date: str, selection, item) -> dict[str, str]:
    """writer 没写 frontmatter 时，从正文与选题信息推导四个字段。"""
    title = selection.topic_title
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip() or title
            break

    summary = ""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block.startswith(("#", ">", "---")):
            continue
        summary = re.sub(r"!?\[([^]]*)\]\([^)]*\)", r"\1", block)
        summary = re.sub(r"[*_`>]+", "", summary)
        summary = re.sub(r"\s+", " ", summary).strip()
        break
    if not summary:
        summary = selection.selection_reason
    if len(summary) > 120:
        summary = summary[:120].rstrip() + "……"

    tags = [item.category] if item is not None and getattr(item, "category", "") else []
    if not tags:
        tags = ["科技"]

    return {
        "title": _yaml_scalar(title),
        "date": rss_date,
        "tags": str(tags),
        "summary": _yaml_scalar(summary),
    }


def _yaml_scalar(value: str) -> str:
    """含 YAML 特殊字符时转双引号字符串，避免 content schema 解析失败。"""
    if re.search(r"[:#'\"\[\]{},&*!|>%@`]|^[-?]|\s$", value):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def _insert_degraded_warning(text: str) -> str:
    """把降级警告行插到文首；已有 frontmatter 时插到 frontmatter 之后。"""
    if text.lstrip().startswith(DEGRADED_WARNING):
        return text
    stripped = text.lstrip()
    if stripped.startswith("---"):
        end = stripped.find("\n---", 3)
        if end != -1:
            close_end = stripped.find("\n", end + 1)
            head = stripped[:close_end].rstrip("\n")
            tail = stripped[close_end:].lstrip("\n")
            return head + "\n\n" + DEGRADED_WARNING + "\n\n" + tail
    return DEGRADED_WARNING + "\n\n" + stripped


def enforce_date_prefix(path: Path, rss_date: str) -> Path:
    """FR4 兜底：文件名强制 <date>- 前缀；writer 自命名时也能保证规范。"""
    if path.name.startswith(f"{rss_date}-"):
        return path
    target = path.with_name(f"{rss_date}-{path.name}")
    counter = 2
    while target.exists():
        target = path.with_name(f"{rss_date}-{counter}-{path.name}")
        counter += 1
    path.rename(target)
    return target


def apply_degraded_markers(path: Path, selection, item) -> None:
    """FR3 降级兜底（代码层）：联网工具全部失败时，补警告行与参考链接段。

    - 首行补 DEGRADED_WARNING（writer 已写则不重复）；
    - 文末缺「参考链接」段时，用选题原文链接 + RSS 摘要中的 URL 补齐（去重，至多 5 条）；
    - 文中已有「参考链接」段时，给无法在 RSS 素材中核实的链接标注「（未能核实）」
      （降级模式下 writer 可能编造 URL，如 news.ycombinator.com/item?id=gemini-3-8-flash）。
    """
    text = path.read_text(encoding="utf-8")
    verified = _verified_urls(selection, item)
    changed = False
    marked_text = _insert_degraded_warning(text)
    if marked_text != text:
        text = marked_text
        changed = True

    has_refs = "## 参考链接" in text or "## 参考资料" in text
    if not has_refs:
        if verified:
            refs = "\n\n## 参考链接\n\n" + "\n".join(f"- {url}" for url in verified)
            text = text.rstrip() + refs + "\n"
            changed = True
    else:
        marked = _mark_unverified_ref_links(text, verified)
        if marked != text:
            text = marked
            changed = True

    if changed:
        path.write_text(text, encoding="utf-8")


def _verified_urls(selection, item, limit: int = 5) -> list[str]:
    """从选题原文链接 + RSS 摘要中提取的可信 URL（去重）。"""
    urls: list[str] = []
    for candidate in [selection.link, *(item.summary.split() if item else [])]:
        for raw in re.findall(r"https?://[^\s,，)）]+", str(candidate)):
            url = raw.rstrip(".。")
            if url not in urls:
                urls.append(url)
    return urls[:limit]


def _mark_unverified_ref_links(text: str, verified: list[str]) -> str:
    """给参考链接段中无法在 RSS 素材里核实的 URL 追加「（未能核实）」标注。"""
    def repl(match: re.Match[str]) -> str:
        url = match.group(1)
        if url in verified or any(url.startswith(v) for v in verified):
            return match.group(0)
        if "未能核实" in match.group(0):
            return match.group(0)
        return match.group(0) + "（未能核实）"

    return re.sub(r"\]\((https?://[^)]+)\)", repl, text)
