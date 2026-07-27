"""会话摘要服务：发现原始对话窗口溢出后，调用模型将旧消息压缩为 JSON 摘要，再写回会话仓库。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.llm.base import ChatModel, LLMError
from app.prompting.template import StrictPromptTemplate
from app.session_store import InMemorySessionStore, StoredMessage
from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SummaryJob:
    session_id: str


class SessionSummaryService:
    """窗口溢出时压缩旧消息；结果只保存在 InMemorySessionStore。"""

    def __init__(
        self, settings: Settings, model: ChatModel, sessions: InMemorySessionStore
    ) -> None:
        self._settings = settings.session
        self._model = model
        self._sessions = sessions
        self._profile = self._settings.summary_profile or settings.default_profile
        prompt_path = _resolve_path(settings, self._settings.summary_prompt_path)
        if not prompt_path.is_file():
            raise ValueError(f"会话摘要 Prompt 不存在: {prompt_path}")
        self._template = StrictPromptTemplate(prompt_path.read_text(encoding="utf-8"))
        self._queue: asyncio.Queue[_SummaryJob | None] = asyncio.Queue(
            maxsize=self._settings.summary_queue_max_size
        )
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._model.validate_profile(self._profile)
        self._worker = asyncio.create_task(self._run(), name="session-summary-worker")

    async def stop(self) -> None:
        await self._queue.join()
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker
            self._worker = None

    def enqueue(self, session_id: str, messages: Sequence[StoredMessage]) -> None:
        if not messages:
            return
        try:
            self._queue.put_nowait(_SummaryJob(session_id))
        except asyncio.QueueFull:
            logger.warning("会话摘要队列已满，跳过 session_id=%s", session_id)

    # 解释：异步运行摘要任务，持续从队列中获取任务并处理，直到收到 None 作为终止信号。
    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                await self._summarize(job)
            except Exception:
                logger.exception("会话摘要更新失败 session_id=%s", getattr(job, "session_id", ""))
            finally:
                self._queue.task_done()

    # 异步执行会话摘要任务，并更新会话摘要。
    async def _summarize(self, job: _SummaryJob) -> None:
        snapshot = await self._sessions.get_summary_snapshot(job.session_id)
        if snapshot is None:
            return
        previous_summary, version, pending_messages = snapshot
        if not pending_messages:
            return
        prompt = self._template.render(
            max_chars=self._settings.summary_max_chars,
            previous_summary=previous_summary or "{}",
            conversation=_format_messages(pending_messages),
        )
        chunks: list[str] = []
        async for chunk in self._model.stream_chat(
            self._profile,
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "请按系统规则生成当前会话摘要，只输出 JSON 对象。",
                },
            ],
            (),
        ):
            if chunk.tool_call_deltas:
                raise LLMError("会话摘要模型返回了工具调用")
            if chunk.text:
                chunks.append(chunk.text)
        # 将摘要文本规范化为 JSON 对象，并检查长度限制，然后尝试将其应用到会话中。
        try:
            summary = _normalize_summary("".join(chunks), self._settings.summary_max_chars)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            # 摘要是可丢弃的短期工作状态：模型异常时保留上一版并消费旧消息，
            # 不能让同一批 pending 消息无限注入后续摘要任务。
            logger.warning("会话摘要无效，丢弃本批旧消息 session_id=%s error=%s", job.session_id, exc)
            summary = _normalize_summary(
                previous_summary or "{}", self._settings.summary_max_chars
            )
        if not await self._sessions.apply_summary(
            job.session_id, version, summary, pending_messages
        ):
            logger.info("会话已结束或摘要版本已更新，丢弃旧摘要 session_id=%s", job.session_id)


# 格式化消息为字符串，用于会话摘要。
def _format_messages(messages: Sequence[StoredMessage]) -> str:
    return "\n".join(
        f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
        for message in messages
    )

# 将原始摘要文本解析为受限的工作状态 JSON，并按优先级裁剪到长度预算内。
def _normalize_summary(raw: str, max_chars: int) -> str:
    value = raw.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise ValueError("会话摘要必须是 JSON 对象")
    task = parsed.get("task") if isinstance(parsed.get("task"), Mapping) else {}
    handoff = (
        parsed.get("handoff") if isinstance(parsed.get("handoff"), Mapping) else {}
    )
    status = task.get("status", "active")
    if status not in {"active", "waiting_user", "waiting_tool", "completed"}:
        status = "active"
    normalized: dict[str, Any] = {
        "v": 1,
        "task": {
            "goal": _trim_text(task.get("goal"), 100),
            "status": status,
            "constraints": _string_list(task.get("constraints"), 3, 40),
        },
        "facts": _string_list(parsed.get("facts"), 5, 80),
        "decisions": _string_list(parsed.get("decisions"), 3, 80),
        "open_loops": _string_list(parsed.get("open_loops"), 3, 80),
        "handoff": {
            "last_result": _trim_text(handoff.get("last_result"), 100),
            "next_action": _trim_text(handoff.get("next_action"), 100),
        },
    }
    return _fit_summary_budget(normalized, max_chars)


def _string_list(value: object, limit: int, item_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        text
        for item in value[:limit]
        if (text := _trim_text(item, item_chars))
    ]


def _trim_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _fit_summary_budget(summary: dict[str, Any], max_chars: int) -> str:
    """优先保留目标和待办，依次删除低优先级的事实与已决事项。"""

    def dump() -> str:
        return json.dumps(summary, ensure_ascii=False, separators=(",", ":"))

    normalized = dump()
    for key in ("facts", "decisions"):
        while len(normalized) > max_chars and summary[key]:
            summary[key].pop()
            normalized = dump()
    for key in ("last_result", "next_action"):
        while len(normalized) > max_chars and summary["handoff"][key]:
            value = summary["handoff"][key]
            summary["handoff"][key] = (
                "" if len(value) == 1 else _trim_text(value, len(value) // 2)
            )
            normalized = dump()
    while len(normalized) > max_chars and summary["task"]["constraints"]:
        summary["task"]["constraints"].pop()
        normalized = dump()
    while len(normalized) > max_chars and summary["open_loops"]:
        summary["open_loops"].pop()
        normalized = dump()
    while len(normalized) > max_chars and summary["task"]["goal"]:
        goal = summary["task"]["goal"]
        summary["task"]["goal"] = "" if len(goal) == 1 else _trim_text(goal, len(goal) // 2)
        normalized = dump()
    if len(normalized) > max_chars:
        raise ValueError("会话摘要长度上限过小，无法保存最小结构")
    return normalized

# 解析配置路径：绝对路径直接返回，相对路径相对于配置文件所在目录
def _resolve_path(settings: Settings, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path
    base = settings.config_path.parent if settings.config_path else Path.cwd()
    return (base / path).resolve()
