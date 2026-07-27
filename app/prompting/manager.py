"""加载、校验并组装静态 Prompt 与动态上下文。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from app.prompting.models import ContextSnapshot
from app.prompting.template import StrictPromptTemplate
from app.settings import Settings

REQUIRED_TAGS = (
    "core_rules",
    "anti_ai_smell",
    "tool_and_knowledge",
    "safety_compliance",
    "context",
    "memory",
)


# Prompt 模板缺失或结构不合法异常
class PromptTemplateError(ValueError):
    """Prompt 模板缺失或结构不合法。"""


# Prompt 管理器：加载、校验并组装静态 Prompt 与动态上下文
class PromptManager:
    # 初始化：加载并校验 Prompt 模板
    def __init__(self, settings: Settings) -> None:
        self._settings = settings.prompt
        self._template_path = _resolve_path(settings, self._settings.template_path)
        self._source = self._load_and_validate(self._template_path)
        self._template = StrictPromptTemplate(self._source)

    # 获取 Prompt 模板文件路径
    @property
    def template_path(self) -> Path:
        return self._template_path

    # 组装完整 Prompt：融合静态模板、动态上下文、记忆和工具定义
    def build(
        self,
        context: ContextSnapshot,
        memory: str,
        tools: Sequence[Mapping[str, Any]],
    ) -> str:
        return self._template.render(
            base_prompt=self._settings.base_prompt,
            current_time=_safe_dynamic(context.current_time),
            today_date=_safe_dynamic(context.today_date),
            today_weekday=_safe_dynamic(context.today_weekday),
            lunar_date=_safe_dynamic(context.lunar_date),
            local_address=_safe_dynamic(context.local_address or "未知位置"),
            weather_info=_safe_dynamic(context.weather_info or "天气信息暂不可用"),
            dynamic_context=_safe_dynamic(context.dynamic_context),
            memory=_safe_dynamic(memory or "暂无长期记忆"),
            available_tools=_format_tools(tools),
        ).strip()

    # 加载并校验 Prompt 模板：检查必需标签并验证顺序
    @staticmethod
    def _load_and_validate(path: Path) -> str:
        if not path.is_file():
            raise PromptTemplateError(f"Prompt 模板不存在: {path}")
        source = path.read_text(encoding="utf-8")
        previous_index = -1
        for tag in REQUIRED_TAGS:
            opening = f"<{tag}>"
            closing = f"</{tag}>"
            if source.count(opening) != 1 or source.count(closing) != 1:
                raise PromptTemplateError(f"Prompt 模板必须且只能包含一组 {opening}")
            current_index = source.index(opening)
            if current_index <= previous_index or source.index(closing) <= current_index:
                raise PromptTemplateError(f"Prompt 标签顺序或闭合错误: {tag}")
            previous_index = current_index
        return source


# 解析配置路径：绝对路径直接返回，相对路径相对于配置文件所在目录
def _resolve_path(settings: Settings, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path
    base = settings.config_path.parent if settings.config_path else Path.cwd()
    return (base / path).resolve()


# 安全转义动态值：防止 HTML/XML 注入
def _safe_dynamic(value: str) -> str:
    return str(value).replace("<", "&lt;").replace(">", "&gt;")


# 格式化工具列表：生成 Markdown 格式的工具说明
def _format_tools(tools: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for schema in tools:
        function = schema.get("function")
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name") or "").strip()
        description = str(function.get("description") or "").strip()
        if name:
            lines.append(f"- {name}：{description}" if description else f"- {name}")
    return "\n".join(lines) if lines else "- 当前没有可用工具"
