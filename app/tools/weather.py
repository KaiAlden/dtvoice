"""基于 Open-Meteo JSON API 的低延迟天气工具。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping

import httpx

from app.cache import StaleTTLCache, TTLCache
from app.settings import WeatherSettings
from app.tools.base import Tool, ToolDefinition, ToolExecutionContext, ToolResult
from app.tools.location import LocationResolver, ResolvedLocation

WEATHER_LABELS = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "毛毛雨",
    53: "毛毛雨",
    55: "强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "强阵雨",
    82: "暴雨",
    95: "雷暴",
    96: "冰雹雷暴",
    99: "强冰雹雷暴",
}

logger = logging.getLogger(__name__)


class WeatherTool(Tool):
    definition = ToolDefinition(
        name="get_weather",
        description=(
            "查询指定城市或用户当前位置的天气。用户没有指定城市时不传 location，"
            "系统会优先使用用户坐标，再按请求 IP 推断城市。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "可选的城市或地区名称，例如北京、杭州。",
                }
            },
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        settings: WeatherSettings,
        http_client: httpx.AsyncClient,
        location_resolver: LocationResolver,
    ) -> None:
        self._settings = settings
        self._http_client = http_client
        self._location_resolver = location_resolver
        self._cache: StaleTTLCache[str] = StaleTTLCache(
            settings.cache_ttl_seconds, settings.stale_cache_seconds
        )
        self._geocode_cache: TTLCache[ResolvedLocation] = TTLCache(
            settings.cache_ttl_seconds
        )
        self._refresh_tasks: dict[str, asyncio.Task[None]] = {}

    # 解释：该方法用于执行天气查询工具，根据用户提供的参数和上下文信息，返回天气查询结果。
    async def execute(
        self, context: ToolExecutionContext, arguments: Mapping[str, Any]
    ) -> ToolResult:
        requested_location = str(arguments.get("location") or "").strip()
        target = await self._resolve_target(requested_location, context)
        if target is None or target.latitude is None or target.longitude is None:
            return ToolResult(False, "无法确定查询天气的位置，请提供城市名称或经纬度")

        cache_key = f"{target.latitude:.3f},{target.longitude:.3f}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            if cached.is_stale:
                self._refresh_in_background(cache_key, target)
                return ToolResult(True, f"{cached.value}（天气数据刷新中）")
            return ToolResult(True, cached.value)

        return await self._fetch_forecast(cache_key, target)

    async def _fetch_forecast(
        self, cache_key: str, target: ResolvedLocation
    ) -> ToolResult:

        try:
            response = await self._http_client.get(
                self._settings.forecast_url,
                params={
                    "latitude": target.latitude,
                    "longitude": target.longitude,
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                    "timezone": "auto",
                    "forecast_days": self._settings.forecast_days,
                },
            )
            response.raise_for_status()
            report = _format_weather(target, response.json())
            self._cache.set(cache_key, report)
            return ToolResult(True, report)
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return ToolResult(False, "天气服务暂时不可用，请稍后重试")

    def _refresh_in_background(self, cache_key: str, target: ResolvedLocation) -> None:
        existing = self._refresh_tasks.get(cache_key)
        if existing is not None and not existing.done():
            return

        async def refresh() -> None:
            try:
                result = await self._fetch_forecast(cache_key, target)
                if not result.ok:
                    logger.warning("天气后台刷新失败 key=%s", cache_key)
            finally:
                self._refresh_tasks.pop(cache_key, None)

        self._refresh_tasks[cache_key] = asyncio.create_task(refresh())

    # 解释：该方法用于解析目标位置，根据用户请求的位置信息或上下文中的坐标信息进行经纬度解析，并返回解析后的经纬度。
    async def _resolve_target(
        self, requested_location: str, context: ToolExecutionContext
    ) -> ResolvedLocation | None:
        if not requested_location:
            return await self._location_resolver.resolve(context)

        geocode_key = requested_location.casefold()
        cached = self._geocode_cache.get(geocode_key)
        if cached is not None:
            return cached

        try:
            response = await self._http_client.get(
                self._settings.geocoding_url,
                params={"name": requested_location, "count": 1, "language": "zh", "format": "json"},
            )
            response.raise_for_status()
            results = response.json().get("results") or []
            if not results:
                return None
            first = results[0]
            resolved = ResolvedLocation(
                city=str(first.get("name") or requested_location),
                latitude=float(first["latitude"]),
                longitude=float(first["longitude"]),
                source="geocoding",
            )
            self._geocode_cache.set(geocode_key, resolved)
            return resolved
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return None

# 解释：该函数用于格式化天气信息，将天气数据转换为可读的文本描述，包括当前天气和未来几天的天气预报。
def _format_weather(target: ResolvedLocation, payload: Mapping[str, Any]) -> str:
    current = payload["current"]
    daily = payload.get("daily") or {}
    city = target.city or "当前位置"
    weather = WEATHER_LABELS.get(int(current.get("weather_code", -1)), "未知")
    current_text = (
        f"{city}当前天气：{weather}，气温 {current.get('temperature_2m')}℃，"
        f"体感 {current.get('apparent_temperature')}℃，风速 {current.get('wind_speed_10m')} km/h。"
    )

    dates = daily.get("time") or []
    codes = daily.get("weather_code") or []
    max_temps = daily.get("temperature_2m_max") or []
    min_temps = daily.get("temperature_2m_min") or []
    forecasts: list[str] = []
    for date, code, high, low in zip(dates, codes, max_temps, min_temps):
        forecasts.append(
            f"{date}：{WEATHER_LABELS.get(int(code), '未知')}，{low}~{high}℃"
        )
    return current_text + ("未来天气：" + "；".join(forecasts) if forecasts else "")
