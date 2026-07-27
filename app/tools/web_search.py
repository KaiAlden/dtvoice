"""联网搜索工具，兼容 Tavily、秘塔与 SearXNG。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import httpx

from app.settings import WebSearchSettings
from app.tools.base import Tool, ToolDefinition, ToolExecutionContext, ToolResult, ToolWarmupResult

CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")
HISTORICAL_PATTERNS = (
    "过去",
    "历史",
    "当时",
    "此前",
    "曾经",
    "去年",
    "前年",
    "回顾",
    "旧闻",
)
NEWS_KEYWORDS = ("新闻", "消息", "动态", "进展", "热点", "头条", "发布", "宣布")


class TemporalIntent(str, Enum):
    LATEST = "latest"
    HISTORICAL = "historical"


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    summary: str
    url: str
    published_text: str
    published_at: datetime | None
    score: float
    original_index: int


class WebSearchTool(Tool):
    definition = ToolDefinition(
        name="web_search",
        description="当问题需要实时互联网信息、公开网页资料或最新动态时使用。",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索问题或关键词"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    def __init__(self, settings: WebSearchSettings, http_client: httpx.AsyncClient):
        self._settings = settings
        self._http_client = http_client

    async def warmup(self) -> ToolWarmupResult:
        """使用 OPTIONS 建立搜索服务连接，不执行会消耗额度的搜索。"""

        if self._settings.provider == "searxng":
            url = f"{self._settings.base_url}/search"
        elif self._settings.provider == "tavily":
            url = self._settings.base_url or "https://api.tavily.com/search"
        else:
            url = self._settings.base_url or "https://metaso.cn/api/v1/search"
        try:
            await self._http_client.options(
                url, headers={"Authorization": f"Bearer {self._settings.api_key}"}
            )
            return ToolWarmupResult(True, "搜索服务连接已预热")
        except httpx.HTTPError as exc:
            return ToolWarmupResult(False, str(exc))

    async def execute(
        self, context: ToolExecutionContext, arguments: Mapping[str, Any]
    ) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult(False, "联网搜索缺少 query 参数")
        temporal_intent = _temporal_intent(query)
        try:
            if self._settings.provider == "tavily":
                results = await self._search_tavily(query, temporal_intent)
            elif self._settings.provider == "metaso":
                results = await self._search_metaso(query, temporal_intent)
            elif self._settings.provider == "searxng":
                results = await self._search_searxng(query, temporal_intent)
            else:
                return ToolResult(False, "联网搜索服务未配置")
            normalized = _normalize_results(results)
            if temporal_intent == TemporalIntent.LATEST:
                normalized = _sort_latest(normalized)
            return ToolResult(
                True,
                _format_results(normalized, temporal_intent),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return ToolResult(False, "联网搜索服务暂时不可用，请稍后重试")

    async def _search_tavily(
        self, query: str, temporal_intent: TemporalIntent
    ) -> list[Mapping[str, Any]]:
        body: dict[str, Any] = {
            "query": query,
            "max_results": self._settings.max_results,
            "search_depth": "basic",
            "include_answer": False,
        }
        if temporal_intent == TemporalIntent.LATEST:
            body["time_range"] = self._settings.latest_time_range
            if any(keyword in query for keyword in NEWS_KEYWORDS):
                body["topic"] = "news"
        response = await self._http_client.post(
            self._settings.base_url or "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self._settings.api_key}"},
            json=body,
        )
        response.raise_for_status()
        return list(response.json().get("results") or [])

    async def _search_metaso(
        self, query: str, temporal_intent: TemporalIntent
    ) -> list[Mapping[str, Any]]:
        effective_query = query
        if temporal_intent == TemporalIntent.LATEST:
            today = datetime.now(CHINA_TIMEZONE).strftime("%Y-%m-%d")
            effective_query = f"{query} 最新信息 截至{today}"
        response = await self._http_client.post(
            self._settings.base_url or "https://metaso.cn/api/v1/search",
            headers={"Authorization": f"Bearer {self._settings.api_key}"},
            json={
                "q": effective_query,
                "size": self._settings.max_results,
                "stream": False,
                "scope": "webpage",
                "includeSummary": True,
            },
        )
        response.raise_for_status()
        return list(response.json().get("webpages") or [])

    async def _search_searxng(
        self, query: str, temporal_intent: TemporalIntent
    ) -> list[Mapping[str, Any]]:
        params: dict[str, Any] = {"q": query, "format": "json", "pageno": 1}
        if temporal_intent == TemporalIntent.LATEST:
            params["time_range"] = self._settings.latest_time_range
        response = await self._http_client.get(
            f"{self._settings.base_url}/search",
            params=params,
        )
        response.raise_for_status()
        return list(response.json().get("results") or [])[: self._settings.max_results]


def _temporal_intent(query: str) -> TemporalIntent:
    if re.search(r"(?:19|20)\d{2}\s*年?", query):
        return TemporalIntent.HISTORICAL
    if any(pattern in query for pattern in HISTORICAL_PATTERNS):
        return TemporalIntent.HISTORICAL
    return TemporalIntent.LATEST


def _normalize_results(results: list[Mapping[str, Any]]) -> list[SearchResult]:
    normalized: list[SearchResult] = []
    for index, item in enumerate(results):
        published_value = _first_value(
            item,
            (
                "published_date",
                "publishedDate",
                "publish_time",
                "publishTime",
                "pubDate",
                "date",
                "time",
            ),
        )
        published_text = str(published_value or "").strip()
        normalized.append(
            SearchResult(
                title=str(item.get("title") or "无标题"),
                summary=str(item.get("content") or item.get("summary") or "").strip(),
                url=str(item.get("url") or item.get("link") or "").strip(),
                published_text=published_text,
                published_at=_parse_datetime(published_value),
                score=_float_or_zero(item.get("score")),
                original_index=index,
            )
        )
    return normalized


def _sort_latest(results: list[SearchResult]) -> list[SearchResult]:
    minimum = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(
        results,
        key=lambda item: (
            item.published_at is not None,
            item.published_at or minimum,
            item.score,
            -item.original_index,
        ),
        reverse=True,
    )


def _format_results(
    results: list[SearchResult], temporal_intent: TemporalIntent
) -> str:
    if not results:
        return "未找到相关联网搜索结果"
    heading = (
        "联网搜索结果（已优先按发布时间排序）："
        if temporal_intent == TemporalIntent.LATEST
        else "联网搜索结果："
    )
    lines = [heading]
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item.title}")
        if item.published_text:
            lines.append(f"发布时间：{item.published_text}")
        if item.summary:
            lines.append(f"摘要：{item.summary[:1_000]}")
        if item.url:
            lines.append(f"链接：{item.url}")
    return "\n".join(lines)


def _first_value(item: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1_000
        try:
            parsed = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError):
                return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=CHINA_TIMEZONE)
    return parsed.astimezone(timezone.utc)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
