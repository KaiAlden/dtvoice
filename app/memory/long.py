"""三维度评估、动态更新并压缩空间的长期记忆。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from app.llm.base import ChatModel, LLMError
from app.mysql_store import MySQLStore
from app.prompting.template import StrictPromptTemplate
from app.settings import MemorySettings, Settings

logger = logging.getLogger(__name__)


# 记忆更新任务单元
@dataclass(frozen=True, slots=True)
class _MemoryJob:
    memory_id: str
    user_message: str
    assistant_message: str


# 长期记忆服务，负责三维度评估与动态压缩
class LongMemoryService:
    # 初始化服务，加载配置、模型和数据库
    def __init__(self, settings: Settings, model: ChatModel, store: MySQLStore) -> None:
        self._settings = settings.memory
        self._model = model
        self._profile = self._settings.profile or settings.default_profile
        self._store = store
        prompt_path = _resolve_path(settings, self._settings.prompt_path)
        if not prompt_path.is_file():
            raise ValueError(f"记忆 Prompt 不存在: {prompt_path}")
        self._template = StrictPromptTemplate(prompt_path.read_text(encoding="utf-8"))
        self._queue: asyncio.Queue[_MemoryJob | None] = asyncio.Queue(
            maxsize=self._settings.queue_max_size
        )
        self._worker: asyncio.Task[None] | None = None
        self._pending: dict[str, list[tuple[str, str]]] = {}

    # 启动内存服务和后台更新工作线程
    async def start(self) -> None:
        self._model.validate_profile(self._profile)
        self._worker = asyncio.create_task(self._run(), name="memory-update-worker")

    # 停止服务，刷新待处理任务并等待工作线程完成
    async def stop(self) -> None:
        self._flush_pending()
        await self._queue.join()
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker
            self._worker = None

    # 查询指定ID的记忆内容
    async def query(self, memory_id: str) -> str:
        record = await self._store.get_memory(memory_id)
        return record.content if record else ""

    # 累积记录会话轮次，达到阈值时触发更新
    def record_turn(self, memory_id: str, user_message: str, assistant_message: str) -> None:
        turns = self._pending.setdefault(memory_id, [])
        turns.append((user_message, assistant_message))
        if len(turns) < self._settings.update_every_turns:
            return
        self._enqueue(memory_id, turns)
        self._pending.pop(memory_id, None)

    # 刷新所有待处理的记忆更新任务
    def _flush_pending(self) -> None:
        for memory_id, turns in list(self._pending.items()):
            self._enqueue(memory_id, turns)
        self._pending.clear()

    # 将记忆更新任务入队，处理队列满的情况
    def _enqueue(self, memory_id: str, turns: list[tuple[str, str]]) -> None:
        user_message = "\n".join(user for user, _ in turns)
        assistant_message = "\n".join(assistant for _, assistant in turns)
        try:
            self._queue.put_nowait(_MemoryJob(memory_id, user_message, assistant_message))
        except asyncio.QueueFull:
            logger.warning("记忆更新队列已满，跳过 memory_id=%s", memory_id)

    # 后台工作线程，持续处理记忆更新队列
    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                await self._update(job)
            except Exception:
                logger.exception("长期记忆更新失败 memory_id=%s", getattr(job, "memory_id", ""))
            finally:
                self._queue.task_done()

    # 执行单个记忆更新：融合新内容并重新评分
    async def _update(self, job: _MemoryJob) -> None:
        record = await self._store.get_memory(job.memory_id)
        revision = record.revision if record else 0
        previous = record.content if record else "{}"
        prompt = self._template.render(
            max_chars=self._settings.max_chars,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            revision=revision + 1,
            previous_memory=previous,
            user_message=job.user_message[:8_000],
            assistant_message=job.assistant_message[:8_000],
        )
        chunks: list[str] = []
        async for chunk in self._model.stream_chat(
            self._profile,
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "请按系统规则更新长期记忆，只输出 JSON 对象。",
                },
            ],
            (),
        ):
            if chunk.tool_call_deltas:
                raise LLMError("记忆模型返回了工具调用")
            if chunk.text:
                chunks.append(chunk.text)
        normalized = _normalize_memory("".join(chunks), revision + 1, self._settings)
        if not await self._store.save_memory(job.memory_id, normalized, revision):
            logger.warning("记忆版本发生冲突，等待后续轮次重试 memory_id=%s", job.memory_id)


# 规范化记忆数据：提取事件、计算综合评分、压缩超限内容
def _normalize_memory(raw: str, revision: int, settings: MemorySettings) -> str:
    data = json.loads(_extract_json(raw))
    if not isinstance(data, dict):
        raise ValueError("记忆输出必须是 JSON 对象")

    archive = data.get("时空档案")
    if not isinstance(archive, dict):
        archive = {}
        data["时空档案"] = archive
    events = archive.get("记忆立方")
    if not isinstance(events, list):
        events = []
        archive["记忆立方"] = events

    normalized_events: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict) or not str(item.get("事件") or "").strip():
            continue
        recency = _score(item.get("时效性"), 100)
        emotion = _score(item.get("情感强度"), 50)
        links = item.get("关联项") if isinstance(item.get("关联项"), list) else []
        density = _score(item.get("关联密度"), min(100, len(links) * 25))
        item["时效性"] = recency
        item["情感强度"] = emotion
        item["关联密度"] = density
        weighted_score = recency * 0.40 + emotion * 0.35 + density * 0.25
        item["综合评分"] = int(weighted_score + 0.5)
        last_seen = _integer(item.get("最后提及轮次"), revision)
        item["最后提及轮次"] = last_seen
        item["提及次数"] = max(1, _integer(item.get("提及次数"), 1))
        if item["综合评分"] < 60 and revision - last_seen >= 3:
            continue
        normalized_events.append(item)

    normalized_events.sort(key=lambda item: item["综合评分"], reverse=True)
    archive["记忆立方"] = normalized_events
    compact = _dump(data)
    while len(compact) > settings.max_chars and normalized_events:
        normalized_events.pop()
        compact = _dump(data)

    for key in ("高光语录",):
        values = data.get(key)
        while len(compact) > settings.max_chars and isinstance(values, list) and values:
            values.pop()
            compact = _dump(data)

    if len(compact) > settings.max_chars:
        raise ValueError("记忆在裁剪低权重条目后仍超过空间上限")
    return compact


# 从原始文本中提取JSON块（兼容 Markdown 代码块）
def _extract_json(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    return stripped


# 将值转换为 0-100 的评分，无效值使用默认值
def _score(value: Any, default: int) -> int:
    return max(0, min(100, _integer(value, default)))


# 安全地将值转换为整数，失败时返回默认值
def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

# 将字典序列化为紧凑 JSON 字符串（无空格、中文）
def _dump(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# 解析提示词路径：绝对路径直接返回，相对路径相对于配置文件所在目录
def _resolve_path(settings: Settings, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path
    # 确定路径基准：配置文件目录优先，其次为当前工作目录
    base = settings.config_path.parent if settings.config_path else Path.cwd()
    return (base / path).resolve()
