"""工具注册与执行调度模块。

负责维护工具白名单、暴露工具 JSON 模式、并发执行工具调用、参数解析、超时与错误处理。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping

from app.tools.base import Tool, ToolCall, ToolExecutionContext, ToolResult, ToolWarmupResult


@dataclass(frozen=True, slots=True)
class ExecutedToolCall:
    """封装单次工具调用及其结果。

    Attributes:
        call: 原始工具调用信息。
        result: 工具执行结果。
    """

    call: ToolCall
    result: ToolResult


class ToolRegistry:
    """管理工具注册与调用调度的注册表。"""

    def __init__(self, timeout_seconds: float = 12.0) -> None:
        """初始化工具注册表。

        Args:
            timeout_seconds: 单个工具调用的超时时间（秒）。
        """
        self._tools: dict[str, Tool] = {}
        self._timeout_seconds = timeout_seconds
        self._schema_cache: tuple[dict[str, Any], ...] | None = None

    def register(self, tool: Tool) -> None:
        """注册一个工具到白名单中。

        如果工具名称已存在，则抛出 ValueError。
        """
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"工具已注册: {name}")
        self._tools[name] = tool
        self._schema_cache = None

    def schemas(self, names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """返回缓存后的 OpenAI JSON schema；可按工具名称筛选。"""

        if self._schema_cache is None:
            self._schema_cache = tuple(
                tool.definition.to_openai_schema() for tool in self._tools.values()
            )
        if names is None:
            return list(self._schema_cache)
        selected = set(names)
        return [
            schema
            for schema in self._schema_cache
            if schema["function"]["name"] in selected
        ]

    async def warmup_all(self, timeout_seconds: float) -> dict[str, ToolWarmupResult]:
        """并行完成已注册工具预热；单项失败不影响其他工具。"""

        async def warm(name: str, tool: Tool) -> tuple[str, ToolWarmupResult]:
            try:
                result = await asyncio.wait_for(tool.warmup(), timeout=timeout_seconds)
                return name, result
            except TimeoutError:
                return name, ToolWarmupResult(False, "预热超时")
            except Exception as exc:
                return name, ToolWarmupResult(False, str(exc))

        pairs = await asyncio.gather(
            *(warm(name, tool) for name, tool in self._tools.items())
        )
        return dict(pairs)

    async def execute_many(
        self, calls: list[ToolCall], context: ToolExecutionContext
    ) -> list[ExecutedToolCall]:
        """并发执行多个工具调用。

        Args:
            calls: 要执行的工具调用列表。
            context: 工具执行上下文。

        Returns:
            每次工具执行封装后的结果列表。
        """
        return list(await asyncio.gather(*(self._execute_one(call, context) for call in calls)))

    async def _execute_one(
        self, call: ToolCall, context: ToolExecutionContext
    ) -> ExecutedToolCall:
        """执行单个工具调用并返回封装结果。"""
        tool = self._tools.get(call.name)
        if tool is None:
            return ExecutedToolCall(call, ToolResult(False, f"未注册的工具: {call.name}"))

        try:
            arguments = json.loads(call.arguments_json or "{}")
            if not isinstance(arguments, Mapping):
                raise ValueError("工具参数必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as exc:
            return ExecutedToolCall(call, ToolResult(False, f"工具参数格式错误: {exc}"))

        try:
            result = await asyncio.wait_for(
                tool.execute(context, arguments), timeout=self._timeout_seconds
            )
            return ExecutedToolCall(call, result)
        except TimeoutError:
            return ExecutedToolCall(call, ToolResult(False, "工具调用超时，请稍后重试"))
        except Exception:
            return ExecutedToolCall(call, ToolResult(False, "工具执行失败，请稍后重试"))
