"""按配置时区生成公历、星期与农历事实。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


@dataclass(frozen=True, slots=True)
# 时间快照
class TimeSnapshot:
    current_time: str
    today_date: str
    today_weekday: str
    lunar_date: str


class TimeInfoProvider:
    """时间按分钟刷新，日期和农历按自然日缓存。"""

    def __init__(self, timezone_name: str) -> None:
        try:
            self._timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"未知时区: {timezone_name}") from exc
        self._minute_key = ""
        self._current_time = ""
        self._date_key = ""
        self._today_weekday = ""
        self._lunar_date = ""

    def snapshot(self, now: datetime | None = None) -> TimeSnapshot:
        current = now.astimezone(self._timezone) if now else datetime.now(self._timezone)
        # 一分钟刷新一次时间
        minute_key = current.strftime("%Y-%m-%d %H:%M")
        if minute_key != self._minute_key:
            self._minute_key = minute_key
            self._current_time = current.strftime("%H:%M")

        # 一天刷新一次日期和农历
        date_key = current.strftime("%Y-%m-%d")
        if date_key != self._date_key:
            self._date_key = date_key
            self._today_weekday = WEEKDAYS[current.weekday()]
            self._lunar_date = _lunar_date(current)

        return TimeSnapshot(
            current_time=self._current_time,
            today_date=self._date_key,
            today_weekday=self._today_weekday,
            lunar_date=self._lunar_date,
        )


def _lunar_date(now: datetime) -> str:
    try:
        import cnlunar

        lunar = cnlunar.Lunar(now.replace(tzinfo=None), godType="8char")
        return f"{lunar.lunarYearCn}年{lunar.lunarMonthCn[:-1]}{lunar.lunarDayCn}"
    except Exception:
        return "农历暂不可用"
