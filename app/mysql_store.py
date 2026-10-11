"""MySQL persistence for chat turns and optional long-term memory."""

# 本模块负责两类持久化数据：逐轮聊天记录，以及启用长期记忆时的记忆内容。

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.settings import MySQLSettings


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    # 一条长期记忆记录；revision 用于避免并发更新覆盖较新的内容。
    memory_id: str
    content: str
    revision: int
    updated_at: str


class MySQLStore:
    """Share a connection pool across chat history and long-term memory."""

    def __init__(self, settings: MySQLSettings) -> None:
        # 连接池供聊天记录和长期记忆共用；信号量限制同时占用的连接数。
        self._settings = settings
        self._pool: Any = None
        self._semaphore = asyncio.Semaphore(settings.pool_size)

    async def initialize(self) -> None:
        # 启动时先检查必需的连接配置，再在线程中建立连接池和数据表。
        if not all(
            (self._settings.host, self._settings.user, self._settings.database)
        ):
            raise ValueError("MySQL 配置不完整：需要 host、user 和 database")
        await asyncio.to_thread(self._initialize_sync)

    async def save_turn(self, user_id: str, prompt: str, answer: str) -> tuple[int, int]:
        # 同步数据库操作放入工作线程，避免阻塞流式回复使用的事件循环。
        async with self._semaphore:
            return await asyncio.to_thread(self._save_turn_sync, user_id, prompt, answer)

    async def get_memory(self, memory_id: str) -> MemoryRecord | None:
        # 长期记忆关闭时，上层不会调用此方法。
        async with self._semaphore:
            return await asyncio.to_thread(self._get_memory_sync, memory_id)

    async def save_memory(
        self, memory_id: str, content: str, expected_revision: int
    ) -> bool:
        # expected_revision 表示调用方读取到的版本，用于检查更新冲突。
        async with self._semaphore:
            return await asyncio.to_thread(
                self._save_memory_sync, memory_id, content, expected_revision
            )

    def _initialize_sync(self) -> None:
        from mysql.connector.pooling import MySQLConnectionPool

        # 目标数据库需要预先存在；这里只负责连接并自动建立所需的表。
        self._pool = MySQLConnectionPool(
            pool_size=self._settings.pool_size,
            pool_reset_session=True,
            host=self._settings.host,
            port=self._settings.port,
            user=self._settings.user,
            password=self._settings.password,
            database=self._settings.database,
            charset="utf8mb4",
            connection_timeout=self._settings.connect_timeout,
        )
        connection = self._pool.get_connection()
        cursor = connection.cursor()
        try:
            # 一轮对话写两行：role=user 保存提问，role=assistant 保存可见回复。
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(128) NOT NULL,
                    role ENUM('user', 'assistant') NOT NULL,
                    content LONGTEXT NOT NULL,
                    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                    INDEX idx_chat_history_user_id (user_id, id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            # 长期记忆按 memory_id 保存一份当前内容及其修订版本。
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id VARCHAR(128) NOT NULL PRIMARY KEY,
                    content TEXT NOT NULL,
                    revision INT NOT NULL,
                    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            connection.commit()
        except Exception:
            # 建表失败时丢弃连接池状态，让应用启动直接报错。
            connection.rollback()
            self._pool = None
            raise
        finally:
            cursor.close()
            connection.close()

    def _connection(self) -> Any:
        # 每次操作从已初始化的连接池借用连接，结束时归还。
        if self._pool is None:
            raise RuntimeError("MySQL 尚未初始化")
        return self._pool.get_connection()

    def _save_turn_sync(
        self, user_id: str, prompt: str, answer: str
    ) -> tuple[int, int]:
        connection = self._connection()
        cursor = connection.cursor()
        try:
            # 两行放在同一事务中，避免只保存用户提问或只保存助手回复。
            connection.start_transaction()
            cursor.execute(
                "INSERT INTO chat_history (user_id, role, content) VALUES (%s, %s, %s)",
                (user_id, "user", prompt),
            )
            user_row_id = int(cursor.lastrowid)
            cursor.execute(
                "INSERT INTO chat_history (user_id, role, content) VALUES (%s, %s, %s)",
                (user_id, "assistant", answer),
            )
            # 返回两条记录真实的自增 ID，供 SSE 的完成帧使用。
            assistant_row_id = int(cursor.lastrowid)
            connection.commit()
            return user_row_id, assistant_row_id
        except Exception:
            # 任一插入失败就回滚整轮聊天记录。
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def _get_memory_sync(self, memory_id: str) -> MemoryRecord | None:
        # 按记忆标识读取当前内容；没有记录时返回 None。
        connection = self._connection()
        cursor = connection.cursor()
        try:
            cursor.execute(
                "SELECT memory_id, content, revision, updated_at "
                "FROM memories WHERE memory_id = %s",
                (memory_id,),
            )
            row = cursor.fetchone()
            return MemoryRecord(*row[:3], str(row[3])) if row else None
        finally:
            cursor.close()
            connection.close()

    def _save_memory_sync(
        self, memory_id: str, content: str, expected_revision: int
    ) -> bool:
        connection = self._connection()
        cursor = connection.cursor()
        try:
            # 首次写入使用版本 1；已有记录只在版本未变化时更新。
            if expected_revision == 0:
                cursor.execute(
                    "INSERT IGNORE INTO memories(memory_id, content, revision) "
                    "VALUES (%s, %s, %s)",
                    (memory_id, content, 1),
                )
            else:
                cursor.execute(
                    "UPDATE memories SET content = %s, revision = %s, "
                    "updated_at = CURRENT_TIMESTAMP(6) "
                    "WHERE memory_id = %s AND revision = %s",
                    (content, expected_revision + 1, memory_id, expected_revision),
                )
            # 影响行数为 0 表示版本冲突或记录已被其他任务先行创建。
            saved = cursor.rowcount == 1
            connection.commit()
            return saved
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()
