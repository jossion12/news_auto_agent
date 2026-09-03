"""Pydantic / 数据模型定义。"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class TopicPick(BaseModel):
    """博客选题：主 Agent 从 RSS 候选中选出的单个主题。"""

    topic_title: str = Field(..., description="选中条目的标题")
    link: str = Field(..., description="原文链接")
    source: str = Field(..., description="来源名称")
    published_at: str = Field(..., description="发布时间")
    selection_reason: str = Field(..., description="选题理由")
    outline: list[str] = Field(
        ...,
        min_length=5,
        max_length=8,
        description="文章大纲，5-8 个要点",
    )


class BlogSelectionResult(BaseModel):
    """主 Agent 的结构化选题结论：picks 数量由运行时 count 决定（FR1）。"""

    picks: list[TopicPick] = Field(..., min_length=1)


class HotspotPick(BaseModel):
    """热点选稿：主 Agent 从候选中选出的单则新闻。"""

    topic_title: str = Field(..., description="选中条目的标题")
    link: str = Field(..., description="原文链接，可能为空字符串（无原文链接）")
    source: str = Field(..., description="来源名称")
    published_at: str = Field(..., description="发布时间")
    selection_reason: str = Field(..., description="选稿理由")


class HotspotSelectionResult(BaseModel):
    """热点主 Agent 的结构化选稿结论：picks 数量由运行时 count 决定。"""

    picks: list[HotspotPick] = Field(..., min_length=1)


@dataclass(frozen=True)
class NewsItem:
    """预处理后的一条候选新闻。"""

    index: int
    title: str
    link: str
    source: str
    published_at: str
    summary: str
    hn_points: int = 0
    hn_comments: int = 0
    category: str = ""

    def to_candidate(self) -> str:
        """渲染成供主 Agent 阅读的紧凑候选（约 400 字符/条）。"""
        title = self.title.strip()[:160] or "(无标题)"
        heat = f"Points={self.hn_points or 'N/A'}, Comments={self.hn_comments or 'N/A'}"
        return (
            f"标题: {title}\n"
            f"链接: {self.link.strip()[:140]}\n"
            f"来源: {self.source} | 发布: {self.published_at.strip()[:60]}\n"
            f"热度: {heat}\n"
            f"摘要: {self.summary[:200]}"
        )


@dataclass(frozen=True)
class RssData:
    """RSS 预处理结果。"""

    date: str
    top_items: list[NewsItem]
    prompt: str


@dataclass(frozen=True)
class HotspotData:
    """热点新闻预处理结果。"""

    date: str
    items: list[NewsItem]
    prompt: str
