"""结束当前匿名会话的内部工具与高置信度意图识别。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Mapping

from app.tools.base import (
    Tool,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
    ToolResultAction,
)

_IGNORED_SEPARATORS = re.compile(r"[\s\u3000，。！？!?、,~～…·]+")
_EXPLICIT_EXIT_PHRASES = frozenset(
    {
        "再见",
        "拜拜",
        "拜",
        "再见啦",
        "拜拜啦",
        "再见喽",
        "拜拜喽",
        "再见吧",
        "退下",
        "退下吧",
        "bye",
        "byebye",
        "好的再见",
        "好再见",
        "那再见",
        "好的拜拜",
        "好拜拜",
        "那拜拜",
        "下次聊",
        "下次再聊",
        "回头聊",
        "回头再聊",
        "晚点聊",
        "晚点再聊",
        "先这样",
        "先这样吧",
        "今天先这样",
        "今天就到这里",
        "今天先聊到这里",
        "结束对话",
        "结束",
        "结束会话",
        "退出对话",
        "退出会话",
        "我先走了",
        "我先去忙了",
        "不聊了",
    }
)
_FAREWELL_RESPONSES = (
    "好嘞，下次见呦～",
    "好呀，那我们下次再聊哦。",
    "好的，回头见，祝您顺顺利利，天天开心呀。",
)


def is_explicit_exit_intent(message: str) -> bool:
    """只匹配独立、明确的告别短句，避免歌曲名和讨论语境误触发。"""

    normalized = unicodedata.normalize("NFKC", message).casefold()
    compact = _IGNORED_SEPARATORS.sub("", normalized)
    return compact in _EXPLICIT_EXIT_PHRASES


def farewell_response(session_id: str) -> str:
    """按会话稳定选择一条自然、简短的告别语。"""

    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    return _FAREWELL_RESPONSES[
        int.from_bytes(digest[:2], "big") % len(_FAREWELL_RESPONSES)
    ]


class ExitIntentTool(Tool):
    """让模型把语义化告别转换成受控的会话结束动作。"""

    definition = ToolDefinition(
        name="handle_exit_intent",
        description=(
            "仅当用户明确表示要结束当前对话、告别或稍后再聊时调用。"
            "如果用户只是在讨论“再见”这个词、歌曲名称或他人的告别，不得调用。"
            "调用时不要同时调用其他工具。"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        return ToolResult(
            True,
            "当前会话应立即结束",
            action=ToolResultAction.END_SESSION,
            response=farewell_response(context.session_id),
        )
