"""OpenAI 兼容协议的异步流式聊天实现。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Mapping

import httpx
from openai import APIError, AsyncOpenAI

from app.llm.base import ChatMessage, ChatModel, ChatStreamChunk, LLMError, ToolCallDelta
from app.settings import LLMProfile, Settings


class OpenAICompatibleChatModel(ChatModel):
    """为每个已配置模型档案复用一个异步 OpenAI 客户端。"""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient):
        self._profiles = settings.llm_profiles
        self._clients: dict[str, AsyncOpenAI] = {}
        for name, profile in self._profiles.items():
            if profile.is_configured:
                self._clients[name] = self._create_client(profile, http_client)

    @staticmethod    # 	住在类里的独立工具函数，不依赖任何实例
    def _create_client(profile: LLMProfile, http_client: httpx.AsyncClient) -> AsyncOpenAI:
        # 创建一个异步 OpenAI 客户端实例，使用指定的 API 密钥和基础 URL，并设置最大重试次数为 0。
        return AsyncOpenAI(
            api_key=profile.api_key,
            base_url=profile.base_url,
            http_client=http_client,
            max_retries=0,
        )

    def validate_profile(self, profile_name: str) -> None:
        if profile_name not in self._profiles:
            raise LLMError(f"未找到模型档案: {profile_name}")
        if profile_name not in self._clients:
            raise LLMError(f"模型档案未完成配置: {profile_name}")

    async def warmup(self, profile_name: str) -> None:
        """发送最小请求以消除模型服务首轮推理冷启动。"""

        received = False
        async for _ in self.stream_chat(
            profile_name,
            [
                {"role": "system", "content": "你是服务预热任务。"},
                {"role": "user", "content": "仅回复 OK。"},
            ],
        ):
            received = True
        if not received:
            raise LLMError("模型预热未返回有效内容")

    # 流式聊天：按分片返回文本增量与工具调用增量
    async def stream_chat(
        self,
        profile_name: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> AsyncIterator[ChatStreamChunk]:
        self.validate_profile(profile_name)
        profile = self._profiles[profile_name]
        client = self._clients[profile_name]

        try:
            request_data: dict[str, Any] = {
                "model": profile.model,
                "messages": list(messages),
                "stream": True,
            }
            extra_body = _request_extra_body(profile)
            if extra_body:
                request_data["extra_body"] = extra_body
            if tools:
                request_data["tools"] = list(tools)
                request_data["tool_choice"] = "auto"
            stream = await client.chat.completions.create(**request_data)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                tool_deltas: list[ToolCallDelta] = []
                for tool_call in delta.tool_calls or ():
                    function = tool_call.function
                    tool_deltas.append(
                        ToolCallDelta(
                            index=tool_call.index,
                            call_id=tool_call.id,
                            name=function.name if function else None,
                            arguments_delta=function.arguments if function else None,
                        )
                    )
                if delta.content or tool_deltas:
                    yield ChatStreamChunk(
                        text=delta.content,
                        tool_call_deltas=tuple(tool_deltas),
                    )
        except APIError as exc:
            raise LLMError("模型服务调用失败，请稍后重试") from exc
        except httpx.HTTPError as exc:
            raise LLMError("模型服务网络连接失败，请稍后重试") from exc


def _request_extra_body(profile: LLMProfile) -> dict[str, Any]:
    """把统一的思考模式配置转换为各 OpenAI 兼容端的请求字段。"""

    extra_body = _copy_mapping(profile.request_extra_body)
    if profile.reasoning_mode != "disabled":
        return extra_body

    disabled_payloads: dict[str, dict[str, Any]] = {
        "none": {},
        "vllm_qwen": {"chat_template_kwargs": {"enable_thinking": False}},
        "siliconflow": {"enable_thinking": False},
        "openai_thinking": {"thinking": {"type": "disabled"}},
    }
    _deep_merge(extra_body, disabled_payloads[profile.reasoning_adapter])
    return extra_body


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """复制配置，避免合并嵌套参数时修改 Settings 中的原始对象。"""

    copied: dict[str, Any] = {}
    for key, item in value.items():
        copied[key] = _copy_mapping(item) if isinstance(item, Mapping) else item
    return copied


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """递归覆盖目标字段，确保 disabled 配置不会被自定义参数反向打开。"""

    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = _copy_mapping(value) if isinstance(value, Mapping) else value
