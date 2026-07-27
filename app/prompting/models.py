"""Prompt 动态上下文的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    current_time: str
    today_date: str
    today_weekday: str
    lunar_date: str
    local_address: str
    weather_info: str
    dynamic_context: str = ""
