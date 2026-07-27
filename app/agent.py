"""具备 Function Calling 循环的流式 Agent。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from time import perf_counter

from app.actions import ClientAction
from app.llm.base import ChatModel, ChatStreamChunk, LLMError, ToolCallDelta
from app.memory.long import LongMemoryService
from app.prompting.context_provider import ContextDataProvider
from app.prompting.manager import PromptManager
from app.session_store import InMemorySessionStore, StoredMessage
from app.session_summary import SessionSummaryService
from app.settings import Settings
from app.tools.base import ToolCall, ToolExecutionContext, ToolResultAction
from app.tools.exit_intent import farewell_response, is_explicit_exit_intent
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "你是一个专业、友好、简洁的中医养生智能聊天小助手，喜欢用亲和、幽默、活泼积极的语气和用户聊天，同时擅长为用户提供养生建议、普及中医养生知识。你来自安徽省亳州市的亳州智慧养生有限公司，你叫大头，平时对话时爱玩梗，喜欢帮助他人。"
MAX_MODEL_CALLS = 2
MAX_TOOL_RESULT_CHARS = 12_000
TRANSITION_TOOL_NAMES = frozenset(
    {
        "get_weather",
        "web_search",
        "get_news",
        "knowledge_search",
        "get_location",
    }
)
TOOL_RESULT_RESPONSE_RULES = """
当前工具已经执行完成。请结合完整会话上下文和工具结果生成最终回复：
1. 保持“大头”的角色身份，表达亲和、自然、简洁，不使用机械客服腔。
2. 只使用工具返回的事实，不得猜测、篡改数字、日期、地点或来源。
3. 提取与用户当前问题最相关的主要内容，不要完整复述工具结果。
4. 不得提及工具、函数、数据库、检索过程或内部调用。
5. 天气突出天气变化和实用建议；知识库突出针对当前体质的问题重点；
   新闻和网络搜索优先表达最新且最相关的内容，并保留必要时间和来源。
6. 当前阶段只输出最终回复，禁止再次调用任何工具。
7. 如果上一条 assistant 消息已有“我看看/我查一下”等简短过渡语，说明用户已经看到它；
   直接承接并给出答案，不要重复过渡语，也不要说“查询完毕”“结果如下”。
""".strip()


# 临时存储工具调用片段，直到完整
@dataclass(slots=True)
class _PendingToolCall:    
    """暂存单个工具调用片段，直到完整。"""
    call_id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ActionOutput:
    """发送给终端的受控输出帧；action 从不由模型文本决定。"""

    action: ClientAction
    response: str


class _ToolCallCollector:
    """将 OpenAI 流式碎片拼装为可执行的完整工具调用。"""

    # 初始化工具调用接收器
    def __init__(self) -> None:
        self._calls: dict[int, _PendingToolCall] = {}

    # 递进单个工具调用片段
    def add(self, delta: ToolCallDelta) -> None:
        call = self._calls.setdefault(delta.index, _PendingToolCall())
        if delta.call_id:
            call.call_id = delta.call_id
        if delta.name:
            call.name = delta.name
        if delta.arguments_delta:
            call.arguments.append(delta.arguments_delta)

    # 整合碎片为完整工具调用，验证并返回
    def build(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(self._calls):
            pending = self._calls[index]
            if not pending.call_id or not pending.name:
                raise LLMError("模型返回了不完整的工具调用")
            calls.append(
                ToolCall(
                    call_id=pending.call_id,
                    name=pending.name,
                    arguments_json="".join(pending.arguments) or "{}",
                )
            )
        return calls


class AgentRunner:
    """直答优先，工具调用时在结果回填后再流式输出最终答案。"""

    # 初始化 Agent，注入配置、模型、窗口、工具与可选的管理服务
    def __init__(
        self,
        settings: Settings,
        model: ChatModel,
        sessions: InMemorySessionStore,
        tools: ToolRegistry,
        prompt_manager: PromptManager | None = None,
        context_provider: ContextDataProvider | None = None,
        memory: LongMemoryService | None = None,
        session_summary: SessionSummaryService | None = None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._sessions = sessions
        self._tools = tools
        self._prompt_manager = prompt_manager
        self._context_provider = context_provider
        self._memory = memory
        self._session_summary = session_summary

    # 验证模型配置，不存在时报错
    def validate_model(self, requested_profile: str | None) -> str:
        profile = requested_profile or self._settings.default_profile
        self._model.validate_profile(profile)
        return profile

    # 流式回复：一次或多个工具调用执行后最常规回复
    async def stream_reply(
        self,
        session_id: str,
        user_message: str,
        requested_profile: str | None,
        tool_context: ToolExecutionContext | None = None,
    ) -> AsyncIterator[ActionOutput]:
        profile = self.validate_model(requested_profile)
        started_at = perf_counter()
        first_token_sent = False
        execution_context = tool_context or ToolExecutionContext(session_id=session_id)

        # 明确、独立的告别短句走确定性快路径，不调用模型，也不写入任何记忆。
        if is_explicit_exit_intent(user_message):
            yield ActionOutput(
                action=ClientAction.END_SESSION,
                response=farewell_response(session_id),
            )
            return

        # 主 Prompt 只包含已完成的会话摘要；待摘要原文仅供后台摘要任务使用。
        async with self._sessions.locked_session(session_id) as session_context:
            system_prompt = await self._build_system_prompt(
                execution_context,
                session_context.summary,
            )
            messages: list[dict[str, object]] = [
                {"role": "system", "content": system_prompt},
                *session_context.history,
                {"role": "user", "content": user_message},
            ]
            # 第一轮由模型判断是否调用工具；工具完成后最多再调用同一模型一次组织结果。
            for round_index in range(MAX_MODEL_CALLS):
                schemas = self._tools.schemas() if round_index == 0 else []
                messages[0] = {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n"
                        "当需要工具时，只能返回工具调用，不能同时输出回答正文。"
                        "获得工具结果后，请基于结果自然地回答用户。"
                        if schemas
                        else f"{system_prompt}\n{TOOL_RESULT_RESPONSE_RULES}"
                    ),
                }
                # 处理模型流式输出，收集工具调用或直接输出最终答案
                collector = _ToolCallCollector()
                saw_tool_call = False
                final_text: list[str] = []

                try:
                    async for chunk in self._model.stream_chat(profile, messages, schemas):
                        if chunk.tool_call_deltas:
                            if final_text:
                                logger.warning(
                                    "模型先返回正文后返回工具调用，继续执行工具 session_id=%s",
                                    session_id,
                                )
                            saw_tool_call = True
                            for delta in chunk.tool_call_deltas:
                                collector.add(delta)

                        if chunk.text:
                            if saw_tool_call:
                                logger.warning(
                                    "模型在工具调用中夹带正文，已忽略该正文 session_id=%s",
                                    session_id,
                                )
                                continue
                            final_text.append(chunk.text)
                            if not first_token_sent:
                                first_token_sent = True
                                logger.info(
                                    "chat 首 token 已输出 session_id=%s profile=%s latency_ms=%.1f",
                                    session_id,
                                    profile,
                                    (perf_counter() - started_at) * 1000,
                                )
                            yield ActionOutput(
                                action=ClientAction.RESPONSE,
                                response=chunk.text,
                            )
                except asyncio.CancelledError:
                    logger.info("chat 客户端已断开 session_id=%s", session_id)
                    raise
                except LLMError:
                    raise
                except Exception as exc:
                    logger.exception("chat 流式响应异常 session_id=%s", session_id)
                    raise LLMError("模型响应异常，请稍后重试") from exc
                # 模型返回了工具调用，执行并整合结果
                if not saw_tool_call:
                    answer = "".join(final_text).strip()
                    if not answer:
                        raise LLMError("模型未返回有效内容，请稍后重试")
                    self._record_answer(
                        session_id, user_message, answer, execution_context
                    )
                    return

                if not schemas:
                    raise LLMError("工具结果整理阶段不允许再次调用工具")

                calls = collector.build()
                if not calls:
                    raise LLMError("模型未提供可执行的工具调用")

                tool_started_at = perf_counter()
                visible_preamble = "".join(final_text).strip()
                transition = (
                    None
                    if visible_preamble
                    else _build_tool_transition(calls, session_id)
                )
                messages.append(
                    _assistant_tool_message(
                        calls,
                        content=visible_preamble or transition,
                    )
                )

                # 先启动工具，再向前端发送过渡语。生成器在 yield 后让出执行权，
                # 工具可在前端播放过渡语期间继续运行，不增加工具自身耗时。
                tool_task = asyncio.create_task(
                    self._tools.execute_many(calls, execution_context),
                    name=f"chat-tools:{session_id}",
                )
                try:
                    if transition:
                        if not first_token_sent:
                            first_token_sent = True
                            logger.info(
                                "chat 工具过渡语已输出 session_id=%s profile=%s latency_ms=%.1f",
                                session_id,
                                profile,
                                (perf_counter() - started_at) * 1000,
                            )
                        yield ActionOutput(
                            action=ClientAction.RESPONSE,
                            response=transition,
                        )
                    results = await tool_task
                finally:
                    if not tool_task.done():
                        tool_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await tool_task
                logger.info(
                    "chat 工具执行完成 session_id=%s tools=%s latency_ms=%.1f",
                    session_id,
                    ",".join(item.call.name for item in results),
                    (perf_counter() - tool_started_at) * 1000,
                )

                exit_results = [
                    item
                    for item in results
                    if item.result.action == ToolResultAction.END_SESSION
                ]
                if exit_results:
                    answer = (
                        exit_results[0].result.response
                        or exit_results[0].result.content
                    ).strip()
                    if not answer:
                        answer = farewell_response(session_id)
                    yield ActionOutput(
                        action=ClientAction.END_SESSION,
                        response=answer,
                    )
                    return

                terminal_actions = {
                    ToolResultAction.RESPONSE,
                    ToolResultAction.RECORD,
                }
                if results and all(
                    item.result.action in terminal_actions for item in results
                ):
                    answer = "\n".join(
                        filter(
                            None,
                            (
                                (item.result.response or item.result.content).strip()
                                for item in results
                            ),
                        )
                    ).strip()
                    if not answer:
                        raise LLMError("工具未返回有效内容，请稍后重试")
                    self._record_answer(
                        session_id, user_message, answer, execution_context
                    )
                    if not first_token_sent:
                        logger.info(
                            "chat 首响应已输出 session_id=%s profile=%s latency_ms=%.1f",
                            session_id,
                            profile,
                            (perf_counter() - started_at) * 1000,
                        )
                    yield ActionOutput(
                        action=ClientAction.RESPONSE,
                        response=answer,
                    )
                    return

                for executed in results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": executed.call.call_id,
                            "content": _tool_result_content(
                                executed.result.ok, executed.result.content
                            ),
                        }
                    )

            raise LLMError("模型未能基于工具结果生成有效回复")

    def _record_answer(
        self,
        session_id: str,
        user_message: str,
        answer: str,
        execution_context: ToolExecutionContext,
    ) -> None:
        """统一记录最终用户可见回复及其会话摘要、长期记忆任务。"""

        overflow = self._sessions.append_turn(session_id, user_message, answer)
        if self._session_summary is not None and overflow:
            self._session_summary.enqueue(session_id, overflow)
        if self._memory is not None:
            memory_id = execution_context.device_id or session_id
            self._memory.record_turn(memory_id, user_message, answer)

    # 构造系统提示词：融合静态提示、上下文、记忆和工具定义
    async def _build_system_prompt(
        self,
        context: ToolExecutionContext,
        session_summary: str,
    ) -> str:
        if self._prompt_manager is None or self._context_provider is None:
            return SYSTEM_PROMPT

        memory_id = context.device_id or context.session_id
        try:
            session_memory = (
                f"本会话摘要（仅在 session {context.session_id} 存活期间有效）：\n"
                f"{session_summary}"
                if session_summary
                else "本会话暂无摘要；仅使用近期原始对话。"
            )
            if self._memory is None:
                snapshot = await self._context_provider.snapshot(
                    context, include_weather=False
                )
                memory = session_memory
            else:
                snapshot, long_memory = await asyncio.gather(
                    self._context_provider.snapshot(context, include_weather=False),
                    self._memory.query(memory_id),
                )
                memory = session_memory
                if long_memory:
                    memory += f"\n\n授权长期记忆：\n{long_memory}"
            return self._prompt_manager.build(snapshot, memory, self._tools.schemas())
        except Exception:
            logger.exception("动态系统提示词构建失败 session_id=%s", context.session_id)
            return SYSTEM_PROMPT


# 构造 Assistant 工具调用消息：封装完整的工具调用列表
def _assistant_tool_message(
    calls: list[ToolCall],
    content: str | None = None,
) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in calls
        ],
    }


def _build_tool_transition(calls: list[ToolCall], session_id: str) -> str | None:
    """为耗时工具生成简短受控过渡语，不额外调用模型。"""

    relevant_calls = [call for call in calls if call.name in TRANSITION_TOOL_NAMES]
    if not relevant_calls:
        return None

    if len(relevant_calls) > 1:
        return _choose_transition(
            session_id,
            relevant_calls,
            (
                "我一起帮你查一下。",
                "我把这几项一起看一下。",
                "我一起核实下这些信息。",
            ),
        )

    call = relevant_calls[0]
    arguments = _parse_tool_arguments(call.arguments_json)
    if call.name == "get_weather":
        location = _compact_transition_value(
            arguments.get("location") or arguments.get("city")
        )
        phrases = (
            (
                f"我看看{location}的天气信息。",
                f"我帮你查下{location}的天气。",
                f"我来看下{location}天气情况。",
            )
            if location
            else (
                "我看看你这里的天气信息。",
                "我帮你查下当地的天气。",
                "我看下你这边的天气情况。",
            )
        )
    elif call.name == "web_search":
        phrases = (
            "我查一下这件事的最新信息。",
            "我看看这件事最近有什么新进展。",
            "我帮你核实下相关信息。",
        )
    elif call.name == "get_news":
        phrases = (
            "我看看今天有什么值得关注的消息。",
            "我帮你看看最新的消息。",
            "我看下最近有哪些新动态。",
        )
    elif call.name == "knowledge_search":
        phrases = (
            "我结合你的情况查一下相关建议。",
            "我帮你找找更合适的建议。",
            "我结合你的情况仔细看一下。",
        )
    else:
        phrases = (
            "我确认一下你现在的位置。",
            "我看下你当前所在的位置。",
            "我帮你确认下当前位置。",
        )

    return _choose_transition(session_id, relevant_calls, phrases)


def _choose_transition(
    session_id: str,
    calls: list[ToolCall],
    phrases: tuple[str, ...],
) -> str:
    """稳定地分散短句，避免同一请求重试时文案来回变化。"""

    seed = "|".join(
        [
            session_id,
            *(f"{call.name}:{call.arguments_json}" for call in calls),
        ]
    )
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return phrases[int.from_bytes(digest[:2], "big") % len(phrases)]


def _parse_tool_arguments(arguments_json: str) -> dict[str, object]:
    try:
        arguments = json.loads(arguments_json)
    except (TypeError, ValueError):
        return {}
    return arguments if isinstance(arguments, dict) else {}


def _compact_transition_value(value: object, max_chars: int = 20) -> str:
    if not isinstance(value, str):
        return ""
    compact = " ".join(value.split()).strip("，。！？；：,.!?;:")
    return compact[:max_chars]


# 格式化工具结果：超长度限制并序列化为 JSON
def _tool_result_content(ok: bool, content: str) -> str:
    safe_content = content[:MAX_TOOL_RESULT_CHARS]
    return json.dumps({"ok": ok, "content": safe_content}, ensure_ascii=False)
