"""系统状态与流式聊天 HTTP 接口。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

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
    """返回本地资源状态，不对外部模型或工具发起探测请求。"""

    snapshot = request.app.state.runtime.health_snapshot()
    return HealthResponse.model_validate(snapshot)


@router.post("/chatstream", tags=["chat"])
async def chatstream(
    request: Request, payload: ChatStreamRequest
) -> StreamingResponse:
    """以 SSE 输出受控回复动作，并在告别时结束当前会话。"""

    agent = request.app.state.agent
    session_id = payload.session_id or uuid4().hex
    try:
        agent.validate_model(payload.model)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def event_stream():
        pending_end_response: str | None = None
        session_cleanup_completed = False
        try:
            context = ToolExecutionContext(
                session_id=session_id,
                device_id=payload.device_id,
                client_ip=_client_ip(request),
                latitude=payload.location.latitude if payload.location else None,
                longitude=payload.location.longitude if payload.location else None,
            )
            async for delta in agent.stream_reply(
                session_id, payload.message, payload.model, context
            ):
                if delta.action == ClientAction.END_SESSION:
                    # 先让 Agent 结束并释放会话锁，再统一删除，避免锁内自删除死锁。
                    pending_end_response = delta.response
                    continue
                if await request.is_disconnected():
                    logger.info("SSE 客户端已断开 session_id=%s", session_id)
                    return
                yield _sse_event(
                    "action", {"action": delta.action, "response": delta.response}
                )

            if pending_end_response is not None:
                await request.app.state.sessions.delete(session_id)
                session_cleanup_completed = True
                yield _sse_event(
                    "action",
                    {
                        "action": ClientAction.END_SESSION,
                        "response": pending_end_response,
                        "ended_session_id": session_id,
                    },
                )
                yield _sse_event(
                    "done",
                    {
                        "ended_session_id": session_id,
                        "session_ended": True,
                    },
                )
                return

            yield _sse_event("done", {"session_id": session_id})
        except LLMError as exc:
            yield _sse_event("error", {"code": "llm_error", "message": str(exc)})
        finally:
            if pending_end_response is not None and not session_cleanup_completed:
                # 即使客户端恰好在告别后断开，也应完成会话清理。
                await asyncio.shield(request.app.state.sessions.delete(session_id))

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
    """为客户端创建随机会话 ID；也可省略 chatstream 的 session_id 自动生成。"""

    return SessionResponse(session_id=uuid4().hex)


@router.delete("/sessions/{session_id}", status_code=204, tags=["chat"])
async def end_session(request: Request, session_id: str) -> Response:
    """显式结束匿名会话并删除其原始窗口与会话摘要。"""

    await request.app.state.sessions.delete(session_id)
    return Response(status_code=204)


def _sse_event(event: str, data: Mapping[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _client_ip(request: Request) -> str | None:
    """只有显式信任反向代理时才读取可伪造的转发头。"""

    if request.app.state.runtime.settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",", maxsplit=1)[0].strip() or None
    return request.client.host if request.client else None
