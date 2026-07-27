"""本地曲库匹配工具；只返回文件标识，不播放音频。"""

from __future__ import annotations

import asyncio
import difflib
from pathlib import Path
from typing import Any, Mapping

from app.settings import MusicSettings
from app.tools.base import (
    Tool,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
    ToolResultAction,
    ToolWarmupResult,
)


class MusicTool(Tool):
    definition = ToolDefinition(
        name="find_music",
        description="在本地曲库中按歌曲名称匹配音乐，返回歌曲名称和文件标识，不播放音频。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "歌曲名、歌手名或音乐关键词，可选"}
            },
            "additionalProperties": False,
        },
    )

    def __init__(self, settings: MusicSettings, music_dir: Path):
        self._settings = settings
        self._music_dir = music_dir
        self._cached_tracks: list[Path] | None = None

    async def warmup(self) -> ToolWarmupResult:
        if not self._music_dir.is_dir():
            return ToolWarmupResult(False, "本地曲库目录不存在")
        self._cached_tracks = await asyncio.to_thread(self._scan_tracks)
        return ToolWarmupResult(True, f"已建立 {len(self._cached_tracks)} 首歌曲索引")

    async def execute(
        self, context: ToolExecutionContext, arguments: Mapping[str, Any]
    ) -> ToolResult:
        if not self._music_dir.is_dir():
            return ToolResult(
                False,
                "本地曲库目录不存在",
                action=ToolResultAction.RESPONSE,
            )
        query = str(arguments.get("query") or "").strip().lower()
        tracks = self._tracks()
        if not tracks:
            return ToolResult(
                True,
                "本地曲库中暂无可匹配的音乐文件",
                action=ToolResultAction.RESPONSE,
            )
        matches = _rank_tracks(query, tracks, self._settings.max_results)
        if not matches:
            return ToolResult(
                True,
                "本地曲库中没有匹配的音乐",
                action=ToolResultAction.RESPONSE,
            )

        lines = ["匹配到的本地音乐："]
        for path in matches:
            identifier = path.relative_to(self._music_dir).as_posix()
            lines.append(f"歌曲名称：{path.stem}\n文件标识：{identifier}")
        content = "\n".join(lines)
        return ToolResult(
            True,
            content,
            action=ToolResultAction.RECORD,
            response=f"好嘞，已经为你找到音乐。\n{content}",
        )

    def _tracks(self) -> list[Path]:
        if self._cached_tracks is not None:
            return self._cached_tracks
        return self._scan_tracks()

    def _scan_tracks(self) -> list[Path]:
        return sorted(
            (
                path
                for path in self._music_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in self._settings.extensions
            ),
            key=lambda path: path.as_posix().lower(),
        )


def _rank_tracks(query: str, tracks: list[Path], limit: int) -> list[Path]:
    if not query:
        return tracks[:limit]
    scored: list[tuple[float, Path]] = []
    for path in tracks:
        name = path.stem.lower()
        score = 1.0 if query in name else difflib.SequenceMatcher(None, query, name).ratio()
        if score >= 0.35:
            scored.append((score, path))
    return [path for _, path in sorted(scored, key=lambda item: (-item[0], item[1].name))[:limit]]
