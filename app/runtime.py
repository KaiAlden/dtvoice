"""应用生命周期内复用的运行时资源。"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

import httpx

from app.settings import Settings


@dataclass(slots=True)
class Runtime:
    """集中管理后续 LLM 与工具调用共用的异步连接资源。"""

    settings: Settings
    started_at: float = field(default_factory=monotonic)
    http_client: httpx.AsyncClient | None = None
    agent_ready: bool = False
    component_statuses: dict[str, str] = field(default_factory=dict)

    async def start(self) -> None:
        # 连接池在应用生命周期内复用，避免后续每次聊天请求重新建立连接。
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.http_timeout_seconds),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                # 覆盖 HTTPX 默认 5 秒，减少用户间隔聊天后的 DNS/TCP/TLS 重建。
                keepalive_expiry=600.0,
            ),
        )

    async def stop(self) -> None:
        if self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None

    def set_component_status(self, component: str, ready: bool) -> None:
        self.component_statuses[component] = "ready" if ready else "not_ready"

    def health_snapshot(self) -> dict[str, object]:
        llm_status = "ready" if self.settings.is_llm_configured else "not_configured"
        status = "ok" if llm_status == "ready" and self.agent_ready else "degraded"
        components: dict[str, dict[str, str]] = {
            "configuration": {"status": "ready"},
            "http_client": {"status": "ready" if self.http_client is not None else "not_ready"},
            "llm": {"status": llm_status, "default_profile": self.settings.default_profile},
            "agent": {"status": "ready" if self.agent_ready else "not_ready"},
        }
        components.update(
            {name: {"status": value} for name, value in self.component_statuses.items()}
        )
        return {
            "status": status,
            "service": self.settings.service_name,
            "version": "0.1.0",
            "environment": self.settings.environment,
            "uptime_seconds": round(monotonic() - self.started_at, 3),
            "components": components,
        }
