"""对外接口的数据模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ComponentHealth(BaseModel):
    status: Literal["ready", "not_ready", "not_configured"]
    default_profile: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    environment: str
    uptime_seconds: float = Field(ge=0)
    components: dict[str, ComponentHealth]


class SessionResponse(BaseModel):
    session_id: str


class LocationOverride(BaseModel):
    """未来客户端提供的精确位置，优先于请求 IP。"""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class ChatStreamRequest(BaseModel):
    """与 ai_bot 对齐的必填字段和 dtvoice 原有可选能力。"""

    user_id: str = Field(min_length=1, max_length=128)
    device_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=8_000)
    model: str | None = Field(default=None, min_length=1, max_length=64)
    location: LocationOverride | None = None
    constitution: str | None = Field(default=None, max_length=128)
    device_area: str | None = Field(default=None, max_length=128)

    @field_validator("user_id", "device_id", "prompt")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("字段不能为空白字符")
        return value

    @field_validator("constitution", "device_area")
    @classmethod
    def trim_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None
