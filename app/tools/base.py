"""工具定义与执行协议。

本模块定义工具的元信息、调用结构、执行上下文与执行结果，以及具体工具必须实现的抽象接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """工具元信息定义。

    Attributes:
        name: 工具名称。
        description: 工具描述。
        parameters: 工具参数的 JSON schema。
    """

    name: str
    description: str
    parameters: Mapping[str, Any]

    # 将工具定义转换为 OpenAI 函数调用的 schema。
    def to_openai_schema(self) -> dict[str, Any]:
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """表示一次工具调用请求。

    Attributes:
        call_id: 调用唯一标识。
        name: 工具名称。
        arguments_json: JSON 格式的参数字符串。
    """

    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """工具执行时的上下文信息。

    Attributes:
        session_id: 当前会话 ID。
        device_id: 设备 ID，可选。
        client_ip: 客户端 IP，可选。
        latitude: 纬度，可选。
        longitude: 经度，可选。
    """

    session_id: str
    device_id: str | None = None
    client_ip: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class ToolResultAction(str, Enum):
    """工具执行完成后的 Agent 编排策略。"""

    RESPONSE = "RESPONSE"
    RECORD = "RECORD"
    REQLLM = "REQLLM"
    END_SESSION = "END_SESSION"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """工具执行结果封装。

    Attributes:
        ok: 表示执行是否成功。
        content: 返回内容或错误信息。
        action: 工具执行完成后的 Agent 编排策略。
        response: RESPONSE/RECORD/END_SESSION 场景下直接发送给用户的固定回复。
    """

    ok: bool
    content: str
    action: ToolResultAction = ToolResultAction.REQLLM
    response: str | None = None


@dataclass(frozen=True, slots=True)
class ToolWarmupResult:
    """工具预热结果。"""

    ok: bool
    detail: str = ""


class Tool(ABC):
    """所有具体业务工具必须实现的异步接口。"""

    definition: ToolDefinition

    @abstractmethod
    async def execute(
        self, context: ToolExecutionContext, arguments: Mapping[str, Any]
    ) -> ToolResult:
        """执行一次受控工具调用。

        Args:
            context: 当前工具调用上下文。
            arguments: 已解析的工具参数。

        Returns:
            工具执行结果。
        """

    async def warmup(self) -> ToolWarmupResult:
        """可选启动预热；默认工具不需要业务预热。"""

        return ToolWarmupResult(True, "无需业务预热")
