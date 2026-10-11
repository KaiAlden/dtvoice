"""确定性获取时间、位置、天气和可选外部动态上下文。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Mapping

import httpx

from app.cache import TTLCache
from app.prompting.models import ContextSnapshot
from app.prompting.time_info import TimeInfoProvider
from app.settings import ContextSettings
from app.tools.base import ToolExecutionContext
from app.tools.location import LocationResolver
from app.tools.weather import WeatherTool

logger = logging.getLogger(__name__)


# 上下文数据提供者：按请求生成时间、位置、天气等可注入 Prompt 的事实快照
class ContextDataProvider:

    # 初始化提供者，注入配置、HTTP 客户端与解析/天气工具
    def __init__(
        self,
        settings: ContextSettings,
        http_client: httpx.AsyncClient,
        location_resolver: LocationResolver,
        weather_tool: WeatherTool,
    ) -> None:
        self._settings = settings
        self._http_client = http_client
        self._location_resolver = location_resolver
        self._weather_tool = weather_tool
        self._time = TimeInfoProvider(settings.timezone)
        self._provider_cache: TTLCache[str] = TTLCache(
            settings.provider_cache_ttl_seconds
        )
        self._startup_snapshot: ContextSnapshot | None = None
        self._startup_local_facts: tuple[str, str] | None = None

    @property
    def startup_snapshot(self) -> ContextSnapshot | None:
        return self._startup_snapshot

    # 启动预热：生成并缓存启动时的上下文快照，错误不会阻止服务启动
    async def warmup(self) -> ContextSnapshot:

        context = ToolExecutionContext(session_id="startup-context")
        self._startup_snapshot = await self.snapshot(context)
        self._startup_local_facts = (
            self._startup_snapshot.local_address,
            self._startup_snapshot.weather_info,
        )
        return self._startup_snapshot

    # 生成完整上下文快照：时间、本地事实与外部动态上下文
    async def snapshot(
        self,
        context: ToolExecutionContext,
        *,
        include_weather: bool = True,
    ) -> ContextSnapshot:
        temporal = self._time.snapshot()
        local_facts, dynamic_context = await asyncio.gather(
            self._local_facts(context, include_weather=include_weather),
            self._external_context(context.device_id),
        )
        local_address, weather_info = local_facts
        return ContextSnapshot(
            current_time=temporal.current_time,
            today_date=temporal.today_date,
            today_weekday=temporal.today_weekday,
            lunar_date=temporal.lunar_date,
            local_address=local_address,
            weather_info=weather_info,
            dynamic_context=dynamic_context,
        )

    # 获取本地事实：解析位置并通过天气工具获取天气信息描述
    async def _local_facts(
        self,
        context: ToolExecutionContext,
        *,
        include_weather: bool,
    ) -> tuple[str, str]:
        if context.device_area and not (
            context.latitude is not None and context.longitude is not None
        ):
            local_address = context.device_area.strip()
            if not include_weather:
                return local_address, "未主动查询；天气问题请调用 get_weather"
            weather = await self._weather_tool.execute(
                context, {"location": local_address}
            )
            return local_address, (
                weather.content if weather.ok else "天气信息暂不可用"
            )
        if (
            context.latitude is None
            and context.longitude is None
            and not context.client_ip
            and self._startup_local_facts is not None
        ):
            return self._startup_local_facts
        location = await self._location_resolver.resolve(context)
        requested_location = ""
        if location is not None:
            local_address = location.city or "当前位置"
        else:
            local_address = self._settings.default_location or "未知位置"
            requested_location = (
                self._settings.weather_location or self._settings.default_location
            )

        if not include_weather:
            return local_address, "未主动查询；天气问题请调用 get_weather"

        weather = await self._weather_tool.execute(
            context, {"location": requested_location} if requested_location else {}
        )
        weather_info = weather.content if weather.ok else "天气信息暂不可用"
        return local_address, weather_info

    # 获取并合并外部动态上下文源（并使用本地缓存），结果受长度限制
    async def _external_context(self, device_id: str | None) -> str:
        if not self._settings.providers:
            return ""
        cache_identity = device_id or "anonymous"
        results = await asyncio.gather(
            *(
                self._fetch_provider(index, provider.url, provider.headers, cache_identity)
                for index, provider in enumerate(self._settings.providers)
            )
        )
        combined = "\n".join(item for item in results if item)
        return combined[: self._settings.max_dynamic_chars]

    # 从单个提供者拉取数据，格式化并缓存；出错时返回空字符串
    async def _fetch_provider(
        self,
        index: int,
        url: str,
        configured_headers: Mapping[str, str],
        device_id: str,
    ) -> str:
        cache_key = f"{index}:{url}:{device_id}"
        cached = self._provider_cache.get(cache_key)
        if cached is not None:
            return cached

        headers = dict(configured_headers)
        if device_id != "anonymous":
            headers["device-id"] = device_id
        try:
            response = await self._http_client.get(
                url,
                headers=headers,
                timeout=self._settings.provider_timeout_seconds,
            )
            response.raise_for_status()
            formatted = _format_provider_payload(response.json())
            self._provider_cache.set(cache_key, formatted)
            return formatted
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning("动态上下文源获取失败 url=%s error=%s", url, exc)
            return ""


# 将外部提供者的响应 payload 格式化为可读的文本列表
def _format_provider_payload(payload: Any) -> str:
    if isinstance(payload, Mapping) and payload.get("code") == 0 and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, Mapping):
        return "\n".join(f"- {key}：{_compact(value)}" for key, value in payload.items())
    if isinstance(payload, list):
        return "\n".join(f"- {_compact(item)}" for item in payload)
    return f"- {_compact(payload)}" if payload is not None else ""


# 将任意值压缩为字符串：字符串去首尾空白，非字符串使用紧凑 JSON 序列化
def _compact(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
