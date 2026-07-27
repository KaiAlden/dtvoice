"""上层 Agent 使用的统一聊天模型接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping


ChatMessage = Mapping[str, Any]  # 规定字典格式的聊天消息。


@dataclass(frozen=True, slots=True)   # 冻结数据，节省内存
class ToolCallDelta:
    """流式响应中某一个工具调用的增量字段。"""

    index: int
    call_id: str | None = None     # 可为空
    name: str | None = None
    arguments_delta: str | None = None


@dataclass(frozen=True, slots=True)
class ChatStreamChunk:
    """兼容文本增量与 OpenAI 流式 Tool Call 增量。"""

    text: str | None = None
    tool_call_deltas: tuple[ToolCallDelta, ...] = field(default_factory=tuple)   # 装着 ToolCallDelta 工具调用碎片的元组，不填默认是空元组


class LLMError(RuntimeError):
    """模型配置或调用失败时抛出的安全异常。继承父类即可"""


class ChatModel(ABC):
    """屏蔽具体供应商差异的流式聊天接口。"""

    @abstractmethod
    def validate_profile(self, profile_name: str) -> None:
        """在开始流式响应前校验模型档案是否可用。"""

    @abstractmethod
    async def stream_chat(
        self,
        profile_name: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> AsyncIterator[ChatStreamChunk]:
        """按流式分片返回文本增量与工具调用增量。"""
