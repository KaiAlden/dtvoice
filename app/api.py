"""System health and ai_bot-compatible streaming chat API."""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.actions import ClientAction
from app.llm.base import LLMError
from app.schemas import ChatStreamRequest, HealthResponse, SessionResponse
from app.tools.base import ToolExecutionContext

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    tags=["system"],
)
async def health(request: Request) -> HealthResponse:
    """Return local component status without probing external services."""
    snapshot = request.app.state.runtime.health_snapshot()
    return HealthResponse.model_validate(snapshot)


@router.post("/api/chatstream", tags=["chat"], response_model=None)
async def chatstream(request: Request) -> Response:
    """Stream text and finish with the two committed MySQL row IDs."""
    try:
        data = await request.json()
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("user_id"), str) or not data["user_id"].strip():
        return _json_result(False, "用户ID不能为空", None, 400)
    if not isinstance(data.get("prompt"), str) or not data["prompt"].strip():
        return _json_result(False, "请输入您的问题", None, 400)
    try:
        payload = ChatStreamRequest.model_validate(data)
    except ValidationError:
        return _json_result(False, "输入参数格式错误", None, 400)

    agent = request.app.state.agent
    user_id = payload.user_id
    try:
        agent.validate_model(payload.model)
    except LLMError as exc:
        return _json_result(False, str(exc), None, 503)

    async def event_stream():
        pending_end_response: str | None = None
        session_cleanup_completed = False
        record_ids: tuple[int, int] | None = None
        streamed_text: list[str] = []
        try:
            context = ToolExecutionContext(
                session_id=user_id,
                device_id=payload.device_id,
                client_ip=_client_ip(request),
                latitude=payload.location.latitude if payload.location else None,
                longitude=payload.location.longitude if payload.location else None,
                constitution=payload.constitution,
                device_area=payload.device_area,
            )
            async for delta in agent.stream_reply(
                user_id, payload.prompt, payload.model, context
            ):
                if delta.action == ClientAction.COMPLETE:
                    record_ids = delta.record_ids
                    continue
                if delta.action == ClientAction.END_SESSION:
                    # Agent must release its session lock before cleanup.
                    pending_end_response = delta.response
                    continue
                if await request.is_disconnected():
                    logger.info("SSE 客户端已断开 user_id=%s", user_id)
                    return
                streamed_text.append(delta.response)
                yield _sse_result(True, "false", delta.response)

            if pending_end_response is not None:
                streamed_text.append(pending_end_response)
                record_ids = await request.app.state.mysql_store.save_turn(
                    user_id, payload.prompt, "".join(streamed_text)
                )
                await request.app.state.sessions.delete(user_id)
                session_cleanup_completed = True
                yield _sse_result(True, "false", pending_end_response)
            if record_ids is None:
                raise RuntimeError("回答未完成或未保存")
            yield _sse_result(True, "true", f"{record_ids[0]},{record_ids[1]}")
        except LLMError as exc:
            yield _sse_result(False, str(exc), None)
        except Exception:
            logger.exception("流式回答或 MySQL 保存失败 user_id=%s", user_id)
            yield _sse_result(False, "回答暂时无法完成，请稍后再试。", None)
        finally:
            if pending_end_response is not None and not session_cleanup_completed:
                await asyncio.shield(request.app.state.sessions.delete(user_id))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/sessions", response_model=SessionResponse, tags=["chat"])
async def create_session() -> SessionResponse:
    """Generate an ID that clients may use as a chatstream user_id."""
    return SessionResponse(session_id=uuid4().hex)


@router.delete("/sessions/{session_id}", status_code=204, tags=["chat"])
async def end_session(request: Request, session_id: str) -> Response:
    """Explicitly clear the in-memory context for an ID."""
    await request.app.state.sessions.delete(session_id)
    return Response(status_code=204)


def _sse_result(success: bool, message: str, data: str | None) -> str:
    return "data: " + json.dumps(
        {"success": success, "message": message, "data": data},
        ensure_ascii=False,
    ) + "\n\n"


def _json_result(
    success: bool, message: str, data: str | None, status_code: int
) -> JSONResponse:
    return JSONResponse(
        {"success": success, "message": message, "data": data},
        status_code=status_code,
    )


def _client_ip(request: Request) -> str | None:
    """Read proxy headers only when explicitly trusted."""
    if request.app.state.runtime.settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",", maxsplit=1)[0].strip() or None
    return request.client.host if request.client else None
