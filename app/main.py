"""HTTP 服务入口与应用生命周期。"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI

from app import __version__
from app.agent import AgentRunner
from app.api import router
from app.llm.openai_compatible import OpenAICompatibleChatModel
from app.logging_config import configure_logging
from app.memory.long import LongMemoryService
from app.prompting.context_provider import ContextDataProvider
from app.prompting.manager import PromptManager
from app.runtime import Runtime
from app.session_store import InMemorySessionStore
from app.session_summary import SessionSummaryService
from app.settings import Settings, load_settings
from app.tools.registry import ToolRegistry
from app.tools.knowledge import KnowledgeSearchTool
from app.tools.exit_intent import ExitIntentTool
from app.tools.location import GetLocationTool, LocationResolver
from app.tools.weather import WeatherTool
from app.tools.web_search import WebSearchTool
from app.tools.news import NewsTool
from app.tools.music import MusicTool

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建应用实例；测试可传入内存配置以避免依赖本地文件。"""

    active_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(active_settings.log_level)
        runtime = Runtime(active_settings)
        await runtime.start()
        app.state.runtime = runtime
        sessions = InMemorySessionStore(
            max_turns=active_settings.session.max_turns,
            resume_turns=active_settings.session.resume_turns,
            history_max_chars=active_settings.session.history_max_chars,
            history_resume_chars=active_settings.session.history_resume_chars,
            pending_hard_max_chars=active_settings.session.pending_hard_max_chars,
            idle_ttl_seconds=active_settings.session.idle_ttl_seconds,
            max_sessions=active_settings.session.max_sessions,
        )
        model = OpenAICompatibleChatModel(active_settings, runtime.http_client)
        tools = ToolRegistry()
        location_resolver = LocationResolver(active_settings.location, runtime.http_client)
        tools.register(GetLocationTool(location_resolver))
        weather_tool = WeatherTool(
            active_settings.weather, runtime.http_client, location_resolver
        )
        tools.register(weather_tool)
        if active_settings.knowledge.is_configured:
            tools.register(KnowledgeSearchTool(active_settings.knowledge))
        if active_settings.web_search.is_configured:
            tools.register(WebSearchTool(active_settings.web_search, runtime.http_client))
        tools.register(NewsTool(active_settings.news, runtime.http_client))
        tools.register(MusicTool(active_settings.music, _music_directory(active_settings)))
        tools.register(ExitIntentTool())
        prompt_manager = PromptManager(active_settings)
        context_provider = ContextDataProvider(
            active_settings.context,
            runtime.http_client,
            location_resolver,
            weather_tool,
        )
        session_summary = None
        if active_settings.session.summary_enabled and active_settings.is_llm_configured:
            session_summary = SessionSummaryService(active_settings, model, sessions)
            await session_summary.start()
        memory = None
        if active_settings.memory.enabled:
            if active_settings.memory.provider != "long":
                raise ValueError(
                    f"不支持的记忆 provider: {active_settings.memory.provider}"
                )
            memory = LongMemoryService(active_settings, model)
            await memory.start()
        app.state.sessions = sessions
        app.state.tools = tools
        app.state.prompt_manager = prompt_manager
        app.state.context_provider = context_provider
        app.state.session_summary = session_summary
        app.state.memory = memory
        # 配置启用时，所有预热并行执行；严格模式下失败不会进入服务阶段。
        if active_settings.startup.warmup_enabled:
            startup = active_settings.startup
            tool_task = asyncio.create_task(
                tools.warmup_all(startup.tool_warmup_timeout_seconds),
                name="tools-startup-warmup",
            )
            context_task = asyncio.create_task(
                asyncio.wait_for(
                    context_provider.warmup(), timeout=startup.weather_warmup_timeout_seconds
                ),
                name="context-startup-warmup",
            )
            model_task = (
                asyncio.create_task(
                    asyncio.wait_for(
                        model.warmup(active_settings.default_profile),
                        timeout=startup.model_warmup_timeout_seconds,
                    ),
                    name="model-startup-warmup",
                )
                if active_settings.is_llm_configured
                else None
            )
            model_waitable = model_task or asyncio.create_task(asyncio.sleep(0))
            try:
                tool_results, context_result, model_result = await asyncio.wait_for(
                    asyncio.gather(
                        tool_task, context_task, model_waitable, return_exceptions=True
                    ),
                    timeout=startup.warmup_timeout_seconds,
                )
            except TimeoutError as exc:
                tool_results = context_result = model_result = exc
                logger.error(
                    "启动预热超过总预算 timeout_seconds=%s",
                    startup.warmup_timeout_seconds,
                )

            if isinstance(tool_results, dict):
                for name, result in tool_results.items():
                    runtime.set_component_status(name, result.ok)
            else:
                logger.error("工具预热失败 error=%s", tool_results)
            weather_ready = not isinstance(context_result, Exception)
            model_ready = not isinstance(model_result, Exception)
            knowledge_result = (
                tool_results.get("knowledge_search") if isinstance(tool_results, dict) else None
            )
            knowledge_ready = (
                not active_settings.knowledge.is_configured
                or (knowledge_result is not None and knowledge_result.ok)
            )
            runtime.set_component_status("weather_context", weather_ready)
            runtime.set_component_status("get_weather", weather_ready)
            if active_settings.is_llm_configured:
                runtime.set_component_status("model_warmup", model_ready)
            tools_ready = isinstance(tool_results, dict) and all(
                result.ok for result in tool_results.values()
            )
            runtime.agent_ready = weather_ready and model_ready and knowledge_ready and tools_ready
            if startup.strict_warmup and not runtime.agent_ready:
                logger.error(
                    "核心预热未完成，拒绝启动 weather=%s model=%s knowledge=%s tools=%s",
                    weather_ready,
                    model_ready,
                    knowledge_ready,
                    tools_ready,
                )
                if session_summary is not None:
                    await session_summary.stop()
                if memory is not None:
                    await memory.stop()
                await runtime.stop()
                raise RuntimeError("启动预热失败，服务未就绪")
        else:
            runtime.agent_ready = True

        # 在应用启动时创建 AgentRunner 实例，并将其存储在 app.state.agent 中。
        app.state.agent = AgentRunner(
            active_settings,
            model,
            sessions,
            tools,
            prompt_manager,
            context_provider,
            memory,
            session_summary,
        )
        # 使用 yield 暂停上下文管理器的执行，允许应用继续运行。
        try:
            yield
        # 在应用关闭时执行清理操作，包括停止会话摘要服务、记忆服务、取消上下文预热任务，并停止运行时。
        finally:
            if session_summary is not None:
                await session_summary.stop()
            if memory is not None:
                await memory.stop()
            await runtime.stop()

    app = FastAPI(
        title=active_settings.service_name,
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


def _music_directory(settings: Settings) -> Path:
    configured = Path(settings.music.directory)
    if configured.is_absolute():
        return configured
    base_directory = settings.config_path.parent if settings.config_path else Path.cwd()
    return (base_directory / configured).resolve()


app = create_app()


if __name__ == "__main__":
    settings = load_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
