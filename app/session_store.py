"""会话状态仓库：保存近期原始对话、已生成摘要、待摘要的旧消息，并处理会话锁、过期和清理。。"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import monotonic
from typing import TypedDict

# 存储在会话中的消息结构
class StoredMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class SessionContext:
    """当前 LLM 请求可见的、严格限定在一个 session 内的上下文。"""

    history: tuple[StoredMessage, ...]
    summary: str
    pending_summary: tuple[StoredMessage, ...]


@dataclass(slots=True)
class _SessionState:
    history: deque[StoredMessage] = field(default_factory=deque)
    summary: str = ""
    summary_version: int = 0
    pending_summary: deque[StoredMessage] = field(default_factory=deque)
    created_at: float = 0.0
    last_active_at: float = 0.0


class InMemorySessionStore:
    """按 session 隔离原始上下文和可过期的会话摘要，不保存到磁盘。"""

    def __init__(
        self,
        max_turns: int = 8,
        resume_turns: int | None = None,
        history_max_chars: int = 5_000,
        history_resume_chars: int = 3_500,
        pending_hard_max_chars: int = 6_000,
        idle_ttl_seconds: int = 900,
        max_sessions: int = 1_000,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._max_messages = max_turns * 2
        self._resume_messages = (resume_turns or max_turns) * 2
        self._history_max_chars = history_max_chars
        self._history_resume_chars = history_resume_chars
        self._pending_hard_max_chars = pending_hard_max_chars
        if not 0 < self._resume_messages <= self._max_messages:
            raise ValueError("resume_turns 必须大于 0 且不大于 max_turns")
        if not 0 < history_resume_chars < history_max_chars:
            raise ValueError("history_resume_chars 必须大于 0 且小于 history_max_chars")
        if pending_hard_max_chars <= 0:
            raise ValueError("pending_hard_max_chars 必须大于 0")
        self._idle_ttl_seconds = idle_ttl_seconds
        self._max_sessions = max_sessions
        self._clock = clock
        self._sessions: dict[str, _SessionState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def locked_session(self, session_id: str) -> AsyncIterator[SessionContext]:
        """串行化同一会话请求，并在进入时执行过期会话清理。"""

        self._cleanup_expired()
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            state = self._get_or_create(session_id)
            state.last_active_at = self._clock()
            yield SessionContext(
                tuple(state.history), state.summary, tuple(state.pending_summary)
            )

    @asynccontextmanager
    async def locked_history(self, session_id: str) -> AsyncIterator[list[StoredMessage]]:
        """兼容旧调用方：只返回当前窗口内的原始对话。"""

        async with self.locked_session(session_id) as context:
            yield list(context.history)

    def append_turn(
        self, session_id: str, user_content: str, assistant_content: str
    ) -> tuple[StoredMessage, ...]:
        """调用方需持有该 session 的锁；返回被移出窗口、待摘要的旧消息。"""

        state = self._get_or_create(session_id)
        state.history.append({"role": "user", "content": user_content})
        state.history.append({"role": "assistant", "content": assistant_content})
        state.last_active_at = self._clock()
        overflow: list[StoredMessage] = []
        if self._history_needs_compaction(state.history):
            while (
                len(state.history) > 2
                and (
                    len(state.history) > self._resume_messages
                    or _messages_chars(state.history) > self._history_resume_chars
                )
            ):
                overflow.append(state.history.popleft())
                overflow.append(state.history.popleft())
            state.pending_summary.extend(overflow)
            self._trim_pending(state.pending_summary)
        return tuple(overflow)

    async def get_summary_snapshot(
        self, session_id: str
    ) -> tuple[str, int, tuple[StoredMessage, ...]] | None:
        self._cleanup_expired()
        lock = self._locks.get(session_id)
        if lock is None:
            return None
        async with lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return state.summary, state.summary_version, tuple(state.pending_summary)

    async def apply_summary(
        self,
        session_id: str,
        expected_version: int,
        summary: str,
        consumed_messages: Sequence[StoredMessage],
    ) -> bool:
        lock = self._locks.get(session_id)
        if lock is None:
            return False
        async with lock:
            state = self._sessions.get(session_id)
            if state is None or state.summary_version != expected_version:
                return False
            consumed = tuple(consumed_messages)
            if tuple(state.pending_summary)[: len(consumed)] != consumed:
                return False
            state.summary = summary
            state.summary_version += 1
            for _ in consumed:
                state.pending_summary.popleft()
            state.last_active_at = self._clock()
            return True

    async def delete(self, session_id: str) -> bool:
        lock = self._locks.get(session_id)
        if lock is None:
            return False
        async with lock:
            existed = self._sessions.pop(session_id, None) is not None
        self._locks.pop(session_id, None)
        return existed

    def get_history(self, session_id: str) -> list[StoredMessage]:
        """仅用于测试或不需要并发保证的只读场景。"""

        state = self._sessions.get(session_id)
        return list(state.history) if state else []

    def get_session_count(self) -> int:
        self._cleanup_expired()
        return len(self._sessions)

    def _history_needs_compaction(self, history: Sequence[StoredMessage]) -> bool:
        return (
            len(history) > self._max_messages
            or _messages_chars(history) > self._history_max_chars
        )

    def _trim_pending(self, pending: deque[StoredMessage]) -> None:
        """摘要服务不可用时，宁可遗忘最早的旧消息，也不能无限积压。"""

        while len(pending) > 2 and _messages_chars(pending) > self._pending_hard_max_chars:
            pending.popleft()
            pending.popleft()

    def cleanup_expired(self) -> int:
        return self._cleanup_expired()

    def _get_or_create(self, session_id: str) -> _SessionState:
        state = self._sessions.get(session_id)
        if state is not None:
            return state
        self._evict_for_capacity()
        now = self._clock()
        state = _SessionState(created_at=now, last_active_at=now)
        self._sessions[session_id] = state
        return state

    def _cleanup_expired(self) -> int:
        now = self._clock()
        expired = [
            session_id
            for session_id, state in self._sessions.items()
            if now - state.last_active_at >= self._idle_ttl_seconds
            and not _lock_is_held(self._locks.get(session_id))
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)
        return len(expired)

    def _evict_for_capacity(self) -> None:
        while len(self._sessions) >= self._max_sessions:
            candidates = [
                (state.last_active_at, session_id)
                for session_id, state in self._sessions.items()
                if not _lock_is_held(self._locks.get(session_id))
            ]
            if not candidates:
                return
            _, oldest_id = min(candidates)
            self._sessions.pop(oldest_id, None)
            self._locks.pop(oldest_id, None)


def _lock_is_held(lock: asyncio.Lock | None) -> bool:
    return lock.locked() if lock is not None else False


def _messages_chars(messages: Sequence[StoredMessage]) -> int:
    return sum(len(message["content"]) for message in messages)
