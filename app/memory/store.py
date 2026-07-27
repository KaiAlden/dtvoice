"""使用 SQLite 原子保存按设备隔离的结构化记忆。"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path


# 记忆记录单据：包含ID、内容、版本和更新时间
@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    content: str
    revision: int
    updated_at: str


# SQLite 记忆存储引擎，通过乐观锁实现并发安全
class SQLiteMemoryStore:
    # 初始化存储，指定数据库文件路径
    def __init__(self, path: Path) -> None:
        self._path = path

    # 异步初始化数据库表结构
    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    # 异步获取指定ID的记忆记录
    async def get(self, memory_id: str) -> MemoryRecord | None:
        return await asyncio.to_thread(self._get_sync, memory_id)

    # 异步保存记忆，基于版本号实现乐观锁并发控制
    async def save(self, memory_id: str, content: str, expected_revision: int) -> bool:
        return await asyncio.to_thread(
            self._save_sync, memory_id, content, expected_revision
        )

    # 创建 SQLite 连接，启用 WAL 模式和自动重试
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    # 同步初始化：创建目录并建表
    def _initialize_sync(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    # 同步查询：获取单条记忆记录
    def _get_sync(self, memory_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT memory_id, content, revision, updated_at "
                "FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return MemoryRecord(*row) if row else None

    # 同步保存：新建或更新记录，版本冲突时安全返回失败
    def _save_sync(self, memory_id: str, content: str, expected_revision: int) -> bool:
        next_revision = expected_revision + 1
        with self._connect() as connection:
            if expected_revision == 0:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO memories(memory_id, content, revision) "
                    "VALUES (?, ?, ?)",
                    (memory_id, content, next_revision),
                )
            else:
                cursor = connection.execute(
                    "UPDATE memories SET content = ?, revision = ?, "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE memory_id = ? AND revision = ?",
                    (content, next_revision, memory_id, expected_revision),
                )
        return cursor.rowcount == 1
