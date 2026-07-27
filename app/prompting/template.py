"""只支持简单变量的严格 Prompt 模板渲染器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")  # 匹配{{变量名}}


@dataclass(frozen=True, slots=True)
class StrictPromptTemplate:
    source: str

    def render(self, **values: Any) -> str:
        required = set(_VARIABLE.findall(self.source))   # 抽出变量名集合
        missing = sorted(required.difference(values))  # 找出缺失的变量名
        if missing:
            raise ValueError(f"Prompt 模板缺少变量: {', '.join(missing)}")
        return _VARIABLE.sub(lambda match: str(values[match.group(1)]), self.source)   # sub替换变量名
