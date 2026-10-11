"""城市级 IP 定位工具。

本模块提供客户端经纬度优先、公网 IP 备选的位置信息解析逻辑，以及对应的工具接口实现。
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote

import httpx

from app.cache import TTLCache
from app.settings import LocationSettings
from app.tools.base import Tool, ToolDefinition, ToolExecutionContext, ToolResult, ToolWarmupResult


@dataclass(frozen=True, slots=True)
class ResolvedLocation:
    """解析后的位置信息。

    Attributes:
        city: 城市名，若仅有坐标则为 None。
        latitude: 纬度。
        longitude: 经度。
        source: 位置来源标识。
    """

    city: str | None
    latitude: float | None
    longitude: float | None
    source: str


class LocationResolver:
    """城市定位解析器，支持缓存并按优先级取坐标或 IP。"""

    def __init__(self, settings: LocationSettings, http_client: httpx.AsyncClient):
        self._settings = settings
        self._http_client = http_client
        self._cache: TTLCache[ResolvedLocation] = TTLCache(settings.cache_ttl_seconds)

    async def resolve(self, context: ToolExecutionContext) -> ResolvedLocation | None:
        """解析工具调用上下文中的位置信息。

        优先返回客户端提供的经纬度；否则使用公网 IP 请求第三方服务获取城市级位置信息。
        如果 IP 无效或查询失败，则返回 None。
        """
        if context.latitude is not None and context.longitude is not None:
            return ResolvedLocation(
                city=None,
                latitude=context.latitude,
                longitude=context.longitude,
                source="client_coordinates",
            )

        client_ip = context.client_ip
        if not client_ip or _is_non_public_ip(client_ip):
            return None
        cached = self._cache.get(client_ip)
        if cached is not None:
            return cached

        try:
            response = await self._http_client.get(
                self._settings.ip_lookup_url.format(ip=quote(client_ip, safe=""))
            )
            response.raise_for_status()
            data = response.json()
            if data.get("success") is False:
                return None
            city = _optional_text(data.get("city"))
            location = ResolvedLocation(
                city=city,
                latitude=_optional_float(data.get("latitude")),
                longitude=_optional_float(data.get("longitude")),
                source="request_ip",
            )
            if not city and location.latitude is None:
                return None
            self._cache.set(client_ip, location)
            return location
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    async def warmup(self) -> ToolWarmupResult:
        """用公开保留 IP 预热定位服务连接，不写入用户位置缓存。"""

        try:
            response = await self._http_client.get(
                self._settings.ip_lookup_url.format(ip="8.8.8.8")
            )
            response.raise_for_status()
            return ToolWarmupResult(True, "定位服务连接已预热")
        except httpx.HTTPError as exc:
            return ToolWarmupResult(False, str(exc))


class GetLocationTool(Tool):
    """对外暴露的获取位置信息工具。"""

    definition = ToolDefinition(
        name="get_location",
        description="获取用户当前的城市级位置。优先使用用户提供的经纬度，否则根据请求公网 IP 推断。",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )

    def __init__(self, resolver: LocationResolver):
        self._resolver = resolver

    async def warmup(self) -> ToolWarmupResult:
        return await self._resolver.warmup()

    async def execute(
        self, context: ToolExecutionContext, arguments: Mapping[str, Any]
    ) -> ToolResult:
        """执行位置获取请求。

        返回推断的城市或客户端提供的坐标信息。
        """
        if context.device_area and not (
            context.latitude is not None and context.longitude is not None
        ):
            return ToolResult(
                True, f"当前设备所在地：{context.device_area.strip()}（来源：设备）"
            )
        location = await self._resolver.resolve(context)
        if location is None:
            return ToolResult(False, "无法从当前请求确定用户所在城市")
        if location.city:
            return ToolResult(
                True,
                f"当前推断城市：{location.city}（来源：{location.source}）",
            )
        return ToolResult(
            True,
            f"当前坐标：纬度 {location.latitude}，经度 {location.longitude}（来源：客户端提供）",
        )


def _is_non_public_ip(value: str) -> bool:
    """判断字符串是否表示非公网 IP。"""
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return True
    return ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_unspecified


def _optional_text(value: object) -> str | None:
    """提取非空字符串，否则返回 None。"""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_float(value: object) -> float | None:
    """尝试将值转换为浮点数，失败时返回 None。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
