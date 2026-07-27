"""工具内部使用的轻量级进程内 TTL 缓存。"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class _CacheItem(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int, max_items: int = 1_000) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_items = max_items
        self._items: dict[str, _CacheItem[T]] = {}

    def get(self, key: str) -> T | None:
        item = self._items.get(key)
        if item is None:
            return None
        if item.expires_at <= monotonic():
            self._items.pop(key, None)
            return None
        return item.value

    def set(self, key: str, value: T) -> None:
        if key not in self._items and len(self._items) >= self._max_items:
            self._items.pop(next(iter(self._items)))
        self._items[key] = _CacheItem(value, monotonic() + self._ttl_seconds)


@dataclass(frozen=True, slots=True)
class StaleCacheValue(Generic[T]):
    value: T
    is_stale: bool


@dataclass(slots=True)
class _StaleCacheItem(Generic[T]):
    value: T
    fresh_until: float
    expires_at: float


class StaleTTLCache(Generic[T]):
    """支持过期值快速返回、后台刷新的进程内缓存。"""

    def __init__(self, fresh_seconds: int, stale_seconds: int) -> None:
        self._fresh_seconds = fresh_seconds
        self._stale_seconds = stale_seconds
        self._items: dict[str, _StaleCacheItem[T]] = {}

    def get(self, key: str) -> StaleCacheValue[T] | None:
        item = self._items.get(key)
        if item is None:
            return None
        now = monotonic()
        if item.expires_at <= now:
            self._items.pop(key, None)
            return None
        return StaleCacheValue(item.value, item.fresh_until <= now)

    def set(self, key: str, value: T) -> None:
        now = monotonic()
        self._items[key] = _StaleCacheItem(
            value=value,
            fresh_until=now + self._fresh_seconds,
            expires_at=now + self._fresh_seconds + self._stale_seconds,
        )
