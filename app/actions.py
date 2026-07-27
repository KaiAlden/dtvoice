"""发送给客户端的受控动作协议。"""

from __future__ import annotations

from enum import Enum


class ClientAction(str, Enum):
    """SSE action 事件允许发送给客户端的动作。"""

    RESPONSE = "RESPONSE"
    END_SESSION = "END_SESSION"
