"""应用配置读取与本地校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """配置文件格式或关键字段不合法。"""


@dataclass(frozen=True, slots=True)
class LLMProfile:
    """一个 OpenAI 兼容模型的连接配置。"""

    name: str
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    reasoning_mode: str = "disabled"
    reasoning_adapter: str = "none"
    request_extra_body: Mapping[str, Any] = field(default_factory=dict)

    # 装饰器：只读模式
    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)  # 判断是否配置了必要的字段


# 查ip地址并缓存
@dataclass(frozen=True, slots=True)
class LocationSettings:
    ip_lookup_url: str = "https://ipwho.is/{ip}"
    cache_ttl_seconds: int = 600


@dataclass(frozen=True, slots=True)
class WeatherSettings:
    geocoding_url: str = "https://geocoding-api.open-meteo.com/v1/search"    # 地名--》经纬度
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"   # 经纬度--》天气预报
    cache_ttl_seconds: int = 300  # 缓存时间，单位秒
    stale_cache_seconds: int = 600
    forecast_days: int = 7  # 预报天数


@dataclass(frozen=True, slots=True)
class StartupSettings:
    warmup_enabled: bool = False
    strict_warmup: bool = True
    warmup_timeout_seconds: float = 10.0
    model_warmup_timeout_seconds: float = 8.0
    weather_warmup_timeout_seconds: float = 7.0
    tool_warmup_timeout_seconds: float = 6.0


@dataclass(frozen=True, slots=True)
class PromptSettings:
    template_path: str = "prompts/agent-prompt.txt"
    base_prompt: str = "你叫大头，来自安徽亳州智慧养生科技公司，是一个专业、自然、简洁的中医养生型智能聊天助手，目标是根据中医九大体质观念，结合中医养生知识，提供个性化的养生建议和指导。"


@dataclass(frozen=True, slots=True)
# 接入第三方上下文信息提供器的配置信息
class ContextProviderSettings:
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextSettings:
    timezone: str = "Asia/Shanghai"
    default_location: str = ""
    weather_location: str = ""
    provider_cache_ttl_seconds: int = 300   
    provider_timeout_seconds: float = 3.0   
    max_dynamic_chars: int = 4_000  # 动态上下文最大字符数
    providers: tuple[ContextProviderSettings, ...] = ()


@dataclass(frozen=True, slots=True)

class MemorySettings:
    """跨会话持久化的长期记忆配置。"""

    enabled: bool = False
    provider: str = "long"
    profile: str = ""
    prompt_path: str = "prompts/memory-long.txt"
    max_chars: int = 900
    update_every_turns: int = 1   # 每多少轮更新一次长期记忆
    queue_max_size: int = 100   # 队列最大大小


@dataclass(frozen=True, slots=True)
class MySQLSettings:
    """聊天记录和长期记忆共用的 MySQL 连接配置。"""

    host: str = ""
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = ""
    pool_size: int = 5
    connect_timeout: int = 10


@dataclass(frozen=True, slots=True)
class SessionSettings:
    """会话短期记忆策略。"""

    max_turns: int = 4
    resume_turns: int = 4
    history_max_chars: int = 5_000
    history_resume_chars: int = 3_500
    pending_hard_max_chars: int = 6_000
    idle_ttl_seconds: int = 900
    max_sessions: int = 1_000  
    summary_enabled: bool = True
    summary_profile: str = ""
    summary_prompt_path: str = "prompts/session-summary.txt"
    summary_max_chars: int = 3_000
    summary_queue_max_size: int = 100  # 摘要队列最大大小


@dataclass(frozen=True, slots=True)
class KnowledgeSettings:
    milvus_uri: str = ""
    milvus_token: str = ""
    milvus_connect_timeout: float = 10.0
    milvus_database: str = "default"
    milvus_diet_collection: str = "diet_public"
    milvus_regimen_collection: str = "regimen_public"
    cache_ttl_seconds: int = 3_600
    cache_max_entries: int = 1_000
    cache_version: str = "v1"

    # 把方法伪装成属性
    @property
    def is_configured(self) -> bool:
        return bool(
            self.milvus_uri
            and self.milvus_diet_collection
            and self.milvus_regimen_collection
        )


@dataclass(frozen=True, slots=True)
class WebSearchSettings:
    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    max_results: int = 3
    latest_time_range: str = "month"

    @property
    def is_configured(self) -> bool:
        if self.provider in {"tavily", "metaso"}:
            return bool(self.api_key)
        if self.provider == "searxng":
            return bool(self.base_url)
        return False


@dataclass(frozen=True, slots=True)
class NewsSource:
    """一个可由新闻工具读取的 RSS 分类源。"""

    name: str
    label: str
    url: str


def _default_news_sources() -> dict[str, NewsSource]:
    """提供可直接运行的三个默认分类源，兼容旧版单源配置。"""

    return {
        "society": NewsSource(
            name="society",
            label="社会",
            url="https://www.chinanews.com.cn/rss/society.xml",
        ),
        "world": NewsSource(
            name="world",
            label="国际",
            url="https://www.chinanews.com.cn/rss/world.xml",
        ),
        "finance": NewsSource(
            name="finance",
            label="财经",
            url="https://www.chinanews.com.cn/rss/finance.xml",
        ),
    }


@dataclass(frozen=True, slots=True)
class NewsSettings:
    sources: Mapping[str, NewsSource] = field(default_factory=_default_news_sources)
    default_sources: tuple[str, ...] = ("society", "world", "finance")
    max_items_per_source: int = 3
    cache_ttl_seconds: int = 120
    source_timeout_seconds: float = 3.0


@dataclass(frozen=True, slots=True)
class MusicSettings:
    directory: str = "music"
    extensions: tuple[str, ...] = (".mp3", ".wav", ".flac", ".m4a")
    max_results: int = 5


@dataclass(frozen=True, slots=True)
class Settings:
    """第一批所需的最小运行配置。"""

    service_name: str = "aibot-core"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    http_timeout_seconds: float = 15.0
    trust_proxy_headers: bool = False  
    startup: StartupSettings = field(default_factory=StartupSettings)
    default_profile: str = "default"
    llm_profiles: Mapping[str, LLMProfile] = field(default_factory=dict)
    prompt: PromptSettings = field(default_factory=PromptSettings)
    context: ContextSettings = field(default_factory=ContextSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    mysql: MySQLSettings = field(default_factory=MySQLSettings)
    session: SessionSettings = field(default_factory=SessionSettings)
    location: LocationSettings = field(default_factory=LocationSettings)
    weather: WeatherSettings = field(default_factory=WeatherSettings)
    knowledge: KnowledgeSettings = field(default_factory=KnowledgeSettings)
    web_search: WebSearchSettings = field(default_factory=WebSearchSettings)
    news: NewsSettings = field(default_factory=NewsSettings)
    music: MusicSettings = field(default_factory=MusicSettings)
    config_path: Path | None = None

    @property
    def selected_llm_profile(self) -> LLMProfile | None:
        return self.llm_profiles.get(self.default_profile)

    @property
    def is_llm_configured(self) -> bool:
        profile = self.selected_llm_profile
        return profile is not None and profile.is_configured


def load_settings(config_path: str | Path | None = None) -> Settings:
    """读取 YAML 配置；未提供配置文件时使用可启动的安全默认值。"""

    resolved_path = _resolve_config_path(config_path)   # 解析配置文件路径
    raw_config = _read_yaml(resolved_path) if resolved_path else {}
    return _build_settings(raw_config, resolved_path)


# 解析配置文件路径
def _resolve_config_path(config_path: str | Path | None) -> Path | None:
    requested_path = config_path or os.getenv("AIBOT_CONFIG")
    if requested_path:
        path = Path(requested_path).expanduser().resolve()
        if not path.is_file():
            raise ConfigurationError(f"配置文件不存在: {path}")
        return path

    default_path = Path.cwd() / "config.yaml"
    return default_path.resolve() if default_path.is_file() else None


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"配置文件 YAML 格式错误: {path}") from exc

    if not isinstance(loaded, Mapping):     # 检查load导入的是否为类字典对象
        raise ConfigurationError("配置文件根节点必须是对象")
    return loaded


def _build_settings(raw: Mapping[str, Any], config_path: Path | None) -> Settings:
    application = _mapping(raw.get("application"), "application")
    server = _mapping(raw.get("server"), "server")
    startup = _mapping(raw.get("startup"), "startup")
    llm = _mapping(raw.get("llm"), "llm")
    prompt = _mapping(raw.get("prompt"), "prompt")
    context = _mapping(raw.get("context"), "context")
    memory = _mapping(raw.get("memory"), "memory")
    mysql = _mapping(raw.get("mysql"), "mysql")
    session = _mapping(raw.get("session"), "session")
    location = _mapping(raw.get("location"), "location")
    weather = _mapping(raw.get("weather"), "weather")
    knowledge = _mapping(raw.get("knowledge"), "knowledge")
    web_search = _mapping(raw.get("web_search"), "web_search")
    news = _mapping(raw.get("news"), "news")
    music = _mapping(raw.get("music"), "music")

    port = _positive_int(server.get("port", 8080), "server.port", maximum=65535)
    timeout = _positive_float(
        server.get("http_timeout_seconds", 15.0), "server.http_timeout_seconds"
    )
    profiles = _parse_profiles(llm.get("profiles", {}))
    default_profile = _string(llm.get("default_profile", "default"), "llm.default_profile")

    if profiles and default_profile not in profiles:
        raise ConfigurationError(
            f"llm.default_profile 未在 llm.profiles 中定义: {default_profile}"
        )

    return Settings(
        service_name=_string(application.get("name", "aibot-core"), "application.name"),
        environment=_string(application.get("environment", "development"), "application.environment"),
        host=_string(server.get("host", "0.0.0.0"), "server.host"),
        port=port,
        log_level=_string(server.get("log_level", "INFO"), "server.log_level").upper(),
        http_timeout_seconds=timeout,
        trust_proxy_headers=_bool(
            server.get("trust_proxy_headers", False), "server.trust_proxy_headers"
        ),
        startup=StartupSettings(
            warmup_enabled=_bool(
                startup.get("warmup_enabled", False), "startup.warmup_enabled"
            ),
            warmup_timeout_seconds=_positive_float(
                startup.get("warmup_timeout_seconds", 10),
                "startup.warmup_timeout_seconds",
                maximum=60,
            ),
            strict_warmup=_bool(
                startup.get("strict_warmup", True), "startup.strict_warmup"
            ),
            model_warmup_timeout_seconds=_positive_float(
                startup.get("model_warmup_timeout_seconds", 8),
                "startup.model_warmup_timeout_seconds",
                maximum=60,
            ),
            weather_warmup_timeout_seconds=_positive_float(
                startup.get("weather_warmup_timeout_seconds", 7),
                "startup.weather_warmup_timeout_seconds",
                maximum=60,
            ),
            tool_warmup_timeout_seconds=_positive_float(
                startup.get("tool_warmup_timeout_seconds", 6),
                "startup.tool_warmup_timeout_seconds",
                maximum=60,
            ),
        ),
        default_profile=default_profile,
        llm_profiles=profiles,
        prompt=PromptSettings(
            template_path=_string(
                prompt.get("template_path", "prompts/agent-prompt.txt"),
                "prompt.template_path",
            ),
            base_prompt=_string(
                prompt.get("base_prompt", "你是一个专业、自然、简洁的中文 AI 助手。"),
                "prompt.base_prompt",
            ),
        ),
        context=ContextSettings(
            timezone=_string(context.get("timezone", "Asia/Shanghai"), "context.timezone"),
            default_location=_string(
                context.get("default_location", ""), "context.default_location"
            ),
            weather_location=_string(
                context.get("weather_location", ""), "context.weather_location"
            ),
            provider_cache_ttl_seconds=_positive_int(
                context.get("provider_cache_ttl_seconds", 300),
                "context.provider_cache_ttl_seconds",
                maximum=86_400,
            ),
            provider_timeout_seconds=_positive_float(
                context.get("provider_timeout_seconds", 3),
                "context.provider_timeout_seconds",
            ),
            max_dynamic_chars=_positive_int(
                context.get("max_dynamic_chars", 4_000),
                "context.max_dynamic_chars",
                maximum=50_000,
            ),
            providers=_parse_context_providers(context.get("providers", [])),
        ),
        memory=MemorySettings(
            enabled=_bool(memory.get("enabled", False), "memory.enabled"),
            provider=_string(memory.get("provider", "long"), "memory.provider"),
            profile=_string(memory.get("profile", ""), "memory.profile"),
            prompt_path=_string(
                memory.get("prompt_path", "prompts/memory-long.txt"),
                "memory.prompt_path",
            ),
            max_chars=_positive_int(
                memory.get("max_chars", 900), "memory.max_chars", maximum=20_000
            ),
            update_every_turns=_positive_int(
                memory.get("update_every_turns", 1),
                "memory.update_every_turns",
                maximum=100,
            ),
            queue_max_size=_positive_int(
                memory.get("queue_max_size", 100),
                "memory.queue_max_size",
                maximum=10_000,
            ),
        ),
        mysql=MySQLSettings(
            host=_string(
                os.getenv("DTVOICE_MYSQL_HOST", mysql.get("host", "")), "mysql.host"
            ),
            port=_positive_int(
                os.getenv("DTVOICE_MYSQL_PORT", mysql.get("port", 3306)),
                "mysql.port",
                maximum=65535,
            ),
            user=_string(
                os.getenv("DTVOICE_MYSQL_USER", mysql.get("user", "")), "mysql.user"
            ),
            password=_string(
                os.getenv("DTVOICE_MYSQL_PASSWORD", mysql.get("password", "")),
                "mysql.password",
            ),
            database=_string(
                os.getenv("DTVOICE_MYSQL_DATABASE", mysql.get("database", "")),
                "mysql.database",
            ),
            pool_size=_positive_int(
                mysql.get("pool_size", 5), "mysql.pool_size", maximum=32
            ),
            connect_timeout=_positive_int(
                mysql.get("connect_timeout", 10), "mysql.connect_timeout", maximum=60
            ),
        ),
        session=_parse_session_settings(session),
        location=LocationSettings(
            ip_lookup_url=_string(
                location.get("ip_lookup_url", "https://ipwho.is/{ip}"),
                "location.ip_lookup_url",
            ),
            cache_ttl_seconds=_positive_int(
                location.get("cache_ttl_seconds", 600),
                "location.cache_ttl_seconds",
                maximum=86_400,
            ),
        ),
        weather=WeatherSettings(
            geocoding_url=_string(
                weather.get("geocoding_url", "https://geocoding-api.open-meteo.com/v1/search"),
                "weather.geocoding_url",
            ),
            forecast_url=_string(
                weather.get("forecast_url", "https://api.open-meteo.com/v1/forecast"),
                "weather.forecast_url",
            ),
            cache_ttl_seconds=_positive_int(
                weather.get("cache_ttl_seconds", 300),
                "weather.cache_ttl_seconds",
                maximum=86_400,
            ),
            stale_cache_seconds=_positive_int(
                weather.get("stale_cache_seconds", 600),
                "weather.stale_cache_seconds",
                maximum=86_400,
            ),
            forecast_days=_positive_int(
                weather.get("forecast_days", 7), "weather.forecast_days", maximum=16
            ),
        ),
        knowledge=KnowledgeSettings(
            milvus_uri=_string(
                knowledge.get("milvus_uri", ""), "knowledge.milvus_uri"
            ).rstrip("/"),
            milvus_token=_string(knowledge.get("milvus_token", ""), "knowledge.milvus_token"),
            milvus_connect_timeout=_positive_float(
                knowledge.get("milvus_connect_timeout", 10),
                "knowledge.milvus_connect_timeout",
                maximum=120,
            ),
            milvus_database=_string(
                knowledge.get("milvus_database", "default"), "knowledge.milvus_database"
            ),
            milvus_diet_collection=_string(
                knowledge.get("milvus_diet_collection", "diet_public"),
                "knowledge.milvus_diet_collection",
            ),
            milvus_regimen_collection=_string(
                knowledge.get("milvus_regimen_collection", "regimen_public"),
                "knowledge.milvus_regimen_collection",
            ),
            cache_ttl_seconds=_positive_int(
                knowledge.get("cache_ttl_seconds", 3_600),
                "knowledge.cache_ttl_seconds",
                maximum=86_400,
            ),
            cache_max_entries=_positive_int(
                knowledge.get("cache_max_entries", 1_000),
                "knowledge.cache_max_entries",
                maximum=100_000,
            ),
            cache_version=_string(
                knowledge.get("cache_version", "v1"), "knowledge.cache_version"
            ),
        ),
        web_search=WebSearchSettings(
            provider=_string(web_search.get("provider", ""), "web_search.provider").lower(),
            api_key=_string(web_search.get("api_key", ""), "web_search.api_key"),
            base_url=_string(web_search.get("base_url", ""), "web_search.base_url").rstrip("/"),
            max_results=_positive_int(
                web_search.get("max_results", 3), "web_search.max_results", maximum=10
            ),
            latest_time_range=_choice(
                web_search.get("latest_time_range", "month"),
                "web_search.latest_time_range",
                {"day", "week", "month", "year"},
            ),
        ),
        news=_parse_news_settings(news),
        music=MusicSettings(
            directory=_string(music.get("directory", "music"), "music.directory"),
            extensions=_string_tuple(
                music.get("extensions", [".mp3", ".wav", ".flac", ".m4a"]),
                "music.extensions",
            ),
            max_results=_positive_int(
                music.get("max_results", 5), "music.max_results", maximum=20
            ),
        ),
        config_path=config_path,
    )


def _parse_session_settings(session: Mapping[str, Any]) -> SessionSettings:
    max_turns = _positive_int(session.get("max_turns", 4), "session.max_turns", maximum=20)
    resume_turns = _positive_int(
        session.get("resume_turns", max_turns), "session.resume_turns", maximum=20
    )
    history_max_chars = _positive_int(
        session.get("history_max_chars", 5_000),
        "session.history_max_chars",
        maximum=100_000,
    )
    history_resume_chars = _positive_int(
        session.get("history_resume_chars", 3_500),
        "session.history_resume_chars",
        maximum=100_000,
    )
    pending_hard_max_chars = _positive_int(
        session.get("pending_hard_max_chars", 6_000),
        "session.pending_hard_max_chars",
        maximum=100_000,
    )
    if resume_turns > max_turns:
        raise ConfigurationError("session.resume_turns 不得大于 session.max_turns")
    if history_resume_chars >= history_max_chars:
        raise ConfigurationError(
            "session.history_resume_chars 必须小于 session.history_max_chars"
        )
    return SessionSettings(
        max_turns=max_turns,
        resume_turns=resume_turns,
        history_max_chars=history_max_chars,
        history_resume_chars=history_resume_chars,
        pending_hard_max_chars=pending_hard_max_chars,
        idle_ttl_seconds=_positive_int(
            session.get("idle_ttl_seconds", 900),
            "session.idle_ttl_seconds",
            maximum=86_400,
        ),
        max_sessions=_positive_int(
            session.get("max_sessions", 1_000),
            "session.max_sessions",
            maximum=100_000,
        ),
        summary_enabled=_bool(
            session.get("summary_enabled", True), "session.summary_enabled"
        ),
        summary_profile=_string(session.get("summary_profile", ""), "session.summary_profile"),
        summary_prompt_path=_string(
            session.get("summary_prompt_path", "prompts/session-summary.txt"),
            "session.summary_prompt_path",
        ),
        summary_max_chars=_positive_int(
            session.get("summary_max_chars", 3_000),
            "session.summary_max_chars",
            maximum=20_000,
        ),
        summary_queue_max_size=_positive_int(
            session.get("summary_queue_max_size", 100),
            "session.summary_queue_max_size",
            maximum=10_000,
        ),
    )

# 将yaml文件中的大模型配置解析为LLMProfile对象的字典
def _parse_profiles(value: Any) -> Mapping[str, LLMProfile]:
    profiles = _mapping(value, "llm.profiles")
    parsed: dict[str, LLMProfile] = {}
    for name, profile_config in profiles.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("llm.profiles 的模型名称必须是非空字符串")
        config = _mapping(profile_config, f"llm.profiles.{name}")  # 字典取值工具
        reasoning = _mapping(config.get("reasoning"), f"llm.profiles.{name}.reasoning")
        reasoning_mode = _string(
            reasoning.get("mode", "disabled"), f"llm.profiles.{name}.reasoning.mode"
        ).lower()   # 字符串取值工具
        if reasoning_mode not in {"disabled", "enabled"}:
            raise ConfigurationError(
                f"llm.profiles.{name}.reasoning.mode 仅支持 disabled 或 enabled"
            )
        reasoning_adapter = _string(
            reasoning.get("adapter", "none"), f"llm.profiles.{name}.reasoning.adapter"
        ).lower()
        if reasoning_adapter not in {"none", "vllm_qwen", "siliconflow", "openai_thinking"}:
            raise ConfigurationError(
                f"llm.profiles.{name}.reasoning.adapter 不受支持: {reasoning_adapter}"
            )
        parsed[name] = LLMProfile(
            name=name,
            base_url=_string(config.get("base_url", ""), f"llm.profiles.{name}.base_url"),
            api_key=_string(config.get("api_key", ""), f"llm.profiles.{name}.api_key"),
            model=_string(config.get("model", ""), f"llm.profiles.{name}.model"),
            reasoning_mode=reasoning_mode,
            reasoning_adapter=reasoning_adapter,
            request_extra_body=_mapping(
                config.get("request_extra_body"), f"llm.profiles.{name}.request_extra_body"
            ),
        )
    return parsed


# 将yaml文件中的第三方上下文中的context配置解析为ContextProviderSettings对象的元组
def _parse_context_providers(value: Any) -> tuple[ContextProviderSettings, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigurationError("context.providers 必须是数组")
    providers: list[ContextProviderSettings] = []
    for index, item in enumerate(value):
        config = _mapping(item, f"context.providers.{index}")
        url = _string(config.get("url", ""), f"context.providers.{index}.url")
        if not url:
            continue
        if not url.startswith(("https://", "http://")):
            raise ConfigurationError(f"context.providers.{index}.url 必须是 HTTP(S) 地址")
        raw_headers = _mapping(config.get("headers"), f"context.providers.{index}.headers")
        headers = {
            _string(key, f"context.providers.{index}.headers.key"): _string(
                header_value, f"context.providers.{index}.headers.{key}"
            )
            for key, header_value in raw_headers.items()
        }
        providers.append(ContextProviderSettings(url=url, headers=headers))
    return tuple(providers)


def _parse_news_settings(value: Mapping[str, Any]) -> NewsSettings:
    """读取分类 RSS 源，并保证默认分类均有对应配置。"""

    source_configs = _mapping(value.get("sources"), "news.sources")
    if not source_configs:
        # 旧版的 url_template/default_source 会被安全忽略，改用默认 RSS 分类源。
        return NewsSettings(
            max_items_per_source=_positive_int(
                value.get("max_items_per_source", value.get("max_items", 3)),
                "news.max_items_per_source",
                maximum=10,
            ),
            cache_ttl_seconds=_positive_int(
                value.get("cache_ttl_seconds", 120), "news.cache_ttl_seconds", maximum=86_400
            ),
            source_timeout_seconds=_positive_float(
                value.get("source_timeout_seconds", 3), "news.source_timeout_seconds"
            ),
        )
    sources: dict[str, NewsSource] = {}
    for name, source_config in source_configs.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("news.sources 的名称必须是非空字符串")
        item = _mapping(source_config, f"news.sources.{name}")
        source_type = _string(item.get("type", "rss"), f"news.sources.{name}.type").lower()
        if source_type != "rss":
            raise ConfigurationError(f"news.sources.{name}.type 当前仅支持 rss")
        url = _string(item.get("url", ""), f"news.sources.{name}.url")
        if not url.startswith(("https://", "http://")):
            raise ConfigurationError(f"news.sources.{name}.url 必须是 HTTP(S) 地址")
        sources[name] = NewsSource(
            name=name,
            label=_string(item.get("label", name), f"news.sources.{name}.label"),
            url=url,
        )

    default_sources = _string_list(
        value.get("default_sources", list(sources)), "news.default_sources"
    )
    unknown_sources = [name for name in default_sources if name not in sources]
    if unknown_sources:
        raise ConfigurationError(
            f"news.default_sources 包含未配置分类: {', '.join(unknown_sources)}"
        )

    return NewsSettings(
        sources=sources,
        default_sources=default_sources,
        max_items_per_source=_positive_int(
            value.get("max_items_per_source", 3), "news.max_items_per_source", maximum=10
        ),
        cache_ttl_seconds=_positive_int(
            value.get("cache_ttl_seconds", 120), "news.cache_ttl_seconds", maximum=86_400
        ),
        source_timeout_seconds=_positive_float(
            value.get("source_timeout_seconds", 3), "news.source_timeout_seconds"
        ),
    )

# 将yaml文件中的配置解析为Settings对象的字典
def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field_name} 必须是对象")
    return value

# 字符串取值工具
def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{field_name} 必须是字符串")
    return value.strip()


def _choice(value: Any, field_name: str, choices: set[str]) -> str:
    converted = _string(value, field_name).lower()
    if converted not in choices:
        raise ConfigurationError(
            f"{field_name} 仅支持: {', '.join(sorted(choices))}"
        )
    return converted


# 整数取值工具
def _positive_int(value: Any, field_name: str, maximum: int) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} 必须是整数") from exc
    if not 1 <= converted <= maximum:
        raise ConfigurationError(f"{field_name} 必须在 1 到 {maximum} 之间")
    return converted


# 浮点数取值工具
def _positive_float(
    value: Any, field_name: str, maximum: float | None = None
) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} 必须是数字") from exc
    if converted <= 0 or (maximum is not None and converted > maximum):
        range_hint = f"，且不大于 {maximum}" if maximum is not None else ""
        raise ConfigurationError(f"{field_name} 必须大于 0{range_hint}")
    return converted

# 布尔值取值工具
def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field_name} 必须是布尔值")
    return value

# 把非空字符串数组逐项校验、转小写、冻成元组；并以 . 开头
def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    result = _string_list(value, field_name)
    if any(not item.startswith(".") for item in result):
        raise ConfigurationError(f"{field_name} 中的扩展名必须以 . 开头")
    return result

# 把非空字符串数组逐项校验、转小写、冻成元组；
def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{field_name} 必须是非空字符串数组")
    return tuple(_string(item, field_name).lower() for item in value)
