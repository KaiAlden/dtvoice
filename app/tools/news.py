"""基于多个 RSS 分类源的新闻工具。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping
from xml.etree import ElementTree

import httpx

from app.cache import TTLCache
from app.settings import NewsSettings, NewsSource
from app.tools.base import Tool, ToolDefinition, ToolExecutionContext, ToolResult, ToolWarmupResult

SOURCE_ALIASES = {
    "社会": "society",
    "社会新闻": "society",
    "国际": "world",
    "国际新闻": "world",
    "世界": "world",
    "财经": "finance",
    "财经新闻": "finance",
    "金融": "finance",
}


@dataclass(frozen=True, slots=True)
class NewsItem:
    title: str
    url: str
    published_at: str


class NewsTool(Tool):
    definition = ToolDefinition(
        name="get_news",
        description="获取社会、国际或财经分类的最新新闻；未指定分类时汇总默认分类。",
        parameters={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "可选分类：社会、国际、财经。未填写时汇总默认分类。",
                }
            },
            "additionalProperties": False,
        },
    )

    def __init__(self, settings: NewsSettings, http_client: httpx.AsyncClient):
        self._settings = settings
        self._http_client = http_client
        self._cache: TTLCache[tuple[NewsItem, ...]] = TTLCache(settings.cache_ttl_seconds)

    async def warmup(self) -> ToolWarmupResult:
        result = await self.execute(ToolExecutionContext(session_id="startup-news"), {})
        return ToolWarmupResult(result.ok, "默认新闻已预取" if result.ok else result.content)

    async def execute(
        self, context: ToolExecutionContext, arguments: Mapping[str, Any]
    ) -> ToolResult:
        requested = str(arguments.get("source") or "").strip()
        source_name = SOURCE_ALIASES.get(requested, requested)
        if source_name:
            sources = (source_name,)
        else:
            sources = self._settings.default_sources

        unknown_sources = [name for name in sources if name not in self._settings.sources]
        if unknown_sources:
            return ToolResult(False, f"未配置新闻分类: {', '.join(unknown_sources)}")
        if not sources:
            return ToolResult(False, "未配置可用的新闻 RSS 源")

        reports = await asyncio.gather(
            *(self._fetch_source(self._settings.sources[name]) for name in sources)
        )
        available = [report for report in reports if report is not None]
        if not available:
            return ToolResult(False, "新闻服务暂时不可用，请稍后重试")
        return ToolResult(True, _format_news(available))

    # 从指定的 RSS 源获取新闻列表，返回源和新闻列表的元组，或 None 表示获取失败。
    async def _fetch_source(self, source: NewsSource) -> tuple[NewsSource, tuple[NewsItem, ...]] | None:
        cached = self._cache.get(source.name)
        if cached is not None:
            return source, cached

        try:
            response = await asyncio.wait_for(
                self._http_client.get(
                    source.url,
                    headers={"User-Agent": "aibot-core/0.1"},
                ),
                timeout=self._settings.source_timeout_seconds,
            )
            response.raise_for_status()
            items = _parse_rss(response.content, self._settings.max_items_per_source)
            self._cache.set(source.name, items)
            return source, items
        except (TimeoutError, httpx.HTTPError, ElementTree.ParseError):
            return None


def _parse_rss(content: bytes, max_items: int) -> tuple[NewsItem, ...]:
    """只解析 RSS 标准字段，避免为新闻列表引入额外依赖。"""

    root = ElementTree.fromstring(content)
    items: list[NewsItem] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "无标题").strip()
        url = (item.findtext("link") or "").strip()
        published_at = (item.findtext("pubDate") or "").strip()
        if title:
            items.append(NewsItem(title=title, url=url, published_at=published_at))
    minimum = datetime.min.replace(tzinfo=timezone.utc)
    items.sort(
        key=lambda item: _parse_published_at(item.published_at) or minimum,
        reverse=True,
    )
    return tuple(items[:max_items])


def _format_news(reports: list[tuple[NewsSource, tuple[NewsItem, ...]]]) -> str:
    lines = ["最新新闻："]
    for source, items in reports:
        if not items:
            continue
        lines.append(f"【{source.label}】")
        for index, item in enumerate(items, start=1):
            lines.append(f"{index}. {item.title}")
            if item.published_at:
                lines.append(f"时间：{item.published_at}")
            if item.url:
                lines.append(f"链接：{item.url}")
    return "\n".join(lines)


def _parse_published_at(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
