"""Agent 内部的受控动作协议。"""

from __future__ import annotations

from enum import Enum


class ClientAction(str, Enum):
    """API 将这些动作转换成对外的 SSE 数据帧。"""

    RESPONSE = "RESPONSE"
    END_SESSION = "END_SESSION"
    COMPLETE = "COMPLETE"
