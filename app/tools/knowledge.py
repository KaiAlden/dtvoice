"""按体质、地区、季节和建议类别精确查询 Milvus 养生知识库。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.cache import TTLCache
from app.settings import KnowledgeSettings
from app.tools.base import Tool, ToolDefinition, ToolExecutionContext, ToolResult, ToolWarmupResult

logger = logging.getLogger(__name__)

PROVINCE_TO_AREA = {
    "北京": "华北", "天津": "华北", "河北": "华北", "山西": "华北", "内蒙古": "华北",
    "辽宁": "东北", "吉林": "东北", "黑龙江": "东北",
    "上海": "华东长江以南", "浙江": "华东长江以南", "福建": "华东长江以南", "台湾": "华东长江以南",
    "江苏": "华东长江以北", "山东": "华东长江以北", "安徽": "华东长江以北",
    "湖北": "华中", "湖南": "华中", "河南": "华中", "江西": "华中",
    "广东": "华南", "广西": "华南", "海南": "华南", "香港": "华南", "澳门": "华南",
    "陕西": "西北", "甘肃": "西北", "青海": "西北", "宁夏": "西北", "新疆": "西北",
    "四川": "西南", "云南": "西南", "贵州": "西南", "西藏": "西南", "重庆": "西南",
}
CITY_TO_AREA = {
    **dict.fromkeys("北京 天津 石家庄 唐山 秦皇岛 邯郸 邢台 保定 张家口 承德 沧州 廊坊 衡水 太原 大同 阳泉 长治 晋城 朔州 晋中 运城 忻州 临汾 吕梁 呼和浩特 包头 乌海 赤峰 通辽 鄂尔多斯 呼伦贝尔 巴彦淖尔 乌兰察布".split(), "华北"),
    **dict.fromkeys("沈阳 大连 鞍山 抚顺 本溪 丹东 锦州 营口 阜新 辽阳 盘锦 铁岭 朝阳 葫芦岛 长春 吉林 四平 辽源 通化 白山 松原 白城 延边 哈尔滨 齐齐哈尔 鸡西 鹤岗 双鸭山 大庆 伊春 佳木斯 七台河 牡丹江 黑河 绥化 大兴安岭".split(), "东北"),
    **dict.fromkeys("南京 无锡 徐州 常州 苏州 南通 连云港 淮安 盐城 扬州 镇江 泰州 宿迁 济南 青岛 淄博 枣庄 东营 烟台 潍坊 济宁 泰安 威海 日照 临沂 德州 聊城 滨州 菏泽 合肥 芜湖 蚌埠 淮南 马鞍山 淮北 铜陵 安庆 黄山 滁州 阜阳 宿州 六安 亳州 池州 宣城".split(), "华东长江以北"),
    **dict.fromkeys("上海 杭州 宁波 温州 嘉兴 湖州 绍兴 金华 衢州 舟山 台州 丽水 福州 厦门 莆田 三明 泉州 漳州 南平 龙岩 宁德 台湾 台北".split(), "华东长江以南"),
    **dict.fromkeys("郑州 开封 洛阳 平顶山 安阳 鹤壁 新乡 焦作 濮阳 许昌 漯河 三门峡 南阳 商丘 信阳 周口 驻马店 济源 武汉 黄石 十堰 宜昌 襄阳 鄂州 荆门 孝感 荆州 黄冈 咸宁 随州 恩施 长沙 株洲 湘潭 衡阳 邵阳 岳阳 常德 张家界 益阳 郴州 永州 怀化 娄底 湘西 南昌 景德镇 萍乡 九江 新余 鹰潭 赣州 吉安 宜春 抚州 上饶".split(), "华中"),
    **dict.fromkeys("广州 韶关 深圳 珠海 汕头 佛山 江门 湛江 茂名 肇庆 惠州 梅州 汕尾 河源 阳江 清远 东莞 中山 潮州 揭阳 云浮 南宁 柳州 桂林 梧州 北海 防城港 钦州 贵港 玉林 百色 贺州 河池 来宾 崇左 海口 三亚 三沙 儋州 香港 澳门".split(), "华南"),
    **dict.fromkeys("西安 铜川 宝鸡 咸阳 渭南 延安 汉中 榆林 安康 商洛 兰州 嘉峪关 金昌 白银 天水 武威 张掖 平凉 酒泉 庆阳 定西 陇南 临夏 甘南 西宁 海东 海北 黄南 海南州 果洛 玉树 海西 银川 石嘴山 吴忠 固原 中卫 乌鲁木齐 克拉玛依 吐鲁番 哈密 昌吉 博尔塔拉 巴音郭楞 阿克苏 克孜勒苏 喀什 和田 伊犁 塔城 阿勒泰".split(), "西北"),
    **dict.fromkeys("成都 自贡 攀枝花 泸州 德阳 绵阳 广元 遂宁 内江 乐山 南充 眉山 宜宾 广安 达州 雅安 巴中 资阳 阿坝 甘孜 凉山 贵阳 六盘水 遵义 安顺 毕节 铜仁 黔西南 黔东南 黔南 昆明 曲靖 玉溪 保山 昭通 丽江 普洱 临沧 楚雄 红河 文山 西双版纳 大理 德宏 怒江 迪庆 拉萨 日喀则 昌都 林芝 山南 那曲 阿里 重庆".split(), "西南"),
}
VALID_AREAS = frozenset({"华北", "东北", "华东长江以南", "华东长江以北", "华中", "华南", "西北", "西南"})
DIET_SUGGESTIONS = frozenset({"季节饮食原则", "忌食清单"})
REGIMEN_SUGGESTIONS = frozenset({"起居建议", "情绪调节", "药浴调理", "穴位保健", "运动推荐"})


def normalize_area(area: str) -> str:
    """将省、市或已有大区转换成知识库使用的八大区域。"""

    value = area.strip()
    normalized = PROVINCE_TO_AREA.get(value, CITY_TO_AREA.get(value, value))
    if normalized == value and value.endswith("市"):
        short_name = value[:-1]
        normalized = PROVINCE_TO_AREA.get(short_name, CITY_TO_AREA.get(short_name, short_name))
    if normalized == value:
        # 动态上下文可能给出“安徽亳州”或更详细地址；优先匹配更长的城市名，
        # 避免“海南州”被“海南”这一省级名称错误覆盖。
        for city in sorted(CITY_TO_AREA, key=len, reverse=True):
            if city in value:
                normalized = CITY_TO_AREA[city]
                break
    if normalized == value:
        for province in sorted(PROVINCE_TO_AREA, key=len, reverse=True):
            if province in value:
                normalized = PROVINCE_TO_AREA[province]
                break
    return normalized if normalized in VALID_AREAS else ""


def current_season(now: datetime | None = None) -> str:
    month = (now or datetime.now(ZoneInfo("Asia/Shanghai"))).month
    if month in (3, 4, 5):
        return "春"
    if month in (6, 7, 8):
        return "夏"
    if month in (9, 10, 11):
        return "秋"
    return "冬"


class KnowledgeSearchTool(Tool):

    definition = ToolDefinition(
        name="knowledge_search",
        description=(
            "查询养生知识库。仅用于饮食和调理问题；必须传入用户明确的体质、"
            "设备所在地或用户提供的地区，以及规定的建议类别。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "constitution": {"type": "string", "description": "用户明确提供的体质名称"},
                "area": {"type": "string", "description": "设备所在地或用户提供的省、市、大区"},
                "season": {"type": "string", "enum": ["春", "夏", "秋", "冬"], "description": "可省略，服务端会按当前季节补齐"},
                "suggestion_name": {
                    "type": "string",
                    "enum": ["季节饮食原则", "忌食清单", "起居建议", "情绪调节", "药浴调理", "穴位保健", "运动推荐"],
                    "description": "根据用户问题选择的知识类别",
                },
            },
            "required": ["constitution", "area", "suggestion_name"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        settings: KnowledgeSettings,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory or self._create_client
        self._client: Any | None = None
        self._cache: TTLCache[str] = TTLCache(
            settings.cache_ttl_seconds, settings.cache_max_entries
        )
        self._query_locks: dict[str, asyncio.Lock] = {}

    async def warmup(self) -> ToolWarmupResult:
        """提前完成 PyMilvus 导入、建连和轻量 RPC。"""

        try:
            await asyncio.to_thread(self._warmup_client)
            return ToolWarmupResult(True, "Milvus 已连接")
        except Exception as exc:
            logger.warning("Milvus 预热失败: %s", exc)
            return ToolWarmupResult(False, str(exc))

    async def execute(
        self, context: ToolExecutionContext, arguments: Mapping[str, Any]
    ) -> ToolResult:
        constitution = str(arguments.get("constitution") or "").strip()
        area = str(arguments.get("area") or "").strip()
        suggestion_name = str(arguments.get("suggestion_name") or "").strip()
        season = str(arguments.get("season") or "").strip() or current_season()

        if not constitution:
            return self._response("为了根据您的体质提供针对性的养生建议，请先进行体质检测。")
        if not area or area in {"未知位置", "未知", "unknown", "Unknown"}:
            return self._response("为了提供更有针对性的建议，请先告诉我您所在的省份或地区。")
        if not suggestion_name:
            return self._response("请先明确您想了解饮食建议，还是起居、情绪、药浴、穴位、运动中的某一类调理建议。")
        if suggestion_name in DIET_SUGGESTIONS:
            collection_name = self._settings.milvus_diet_collection
        elif suggestion_name in REGIMEN_SUGGESTIONS:
            collection_name = self._settings.milvus_regimen_collection
        else:
            return self._response(f"不支持的建议分类「{suggestion_name}」。")

        normalized_area = normalize_area(area)
        if not normalized_area:
            return self._response("暂未识别该城市对应的知识库地区，请提供省份或地区名称。")
        if season not in {"春", "夏", "秋", "冬"}:
            return self._response("季节参数无效，请使用春、夏、秋或冬。")

        expression = " and ".join(
            (
                f"constitution == {json.dumps(constitution, ensure_ascii=False)}",
                f"area == {json.dumps(normalized_area, ensure_ascii=False)}",
                f"season == {json.dumps(season, ensure_ascii=False)}",
                f"suggestion_name == {json.dumps(suggestion_name, ensure_ascii=False)}",
            )
        )
        cache_key = "|".join(
            (
                self._settings.cache_version,
                collection_name,
                constitution,
                normalized_area,
                season,
                suggestion_name,
            )
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._response(cached)

        lock = self._query_locks.setdefault(cache_key, asyncio.Lock())
        try:
            async with lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return self._response(cached)
                rows = await asyncio.to_thread(self._query, collection_name, expression)
        except Exception:
            logger.exception("Milvus 精确查询失败")
            return self._response("知识库检索服务暂时不可用，请稍后重试。", ok=False)

        content = str(rows[0].get("content") or "").strip() if rows else ""
        self._cache.set(cache_key, content)
        if not content:
            return self._response("暂未找到与您当前体质、地区和季节相匹配的建议。")
        return self._response(content)

    def _query(self, collection_name: str, expression: str) -> list[Mapping[str, Any]]:
        if self._client is None:
            self._client = self._client_factory()
        rows = self._client.query(
            collection_name=collection_name,
            filter=expression,
            output_fields=["content"],
            limit=1,
        )
        return [row for row in rows if isinstance(row, Mapping)]

    def _warmup_client(self) -> None:
        if self._client is None:
            self._client = self._client_factory()
        self._client.list_collections()

    def _create_client(self) -> Any:
        from pymilvus import MilvusClient

        kwargs: dict[str, Any] = {
            "uri": self._settings.milvus_uri,
            "db_name": self._settings.milvus_database,
            "timeout": self._settings.milvus_connect_timeout,
        }
        if self._settings.milvus_token:
            kwargs["token"] = self._settings.milvus_token
        return MilvusClient(**kwargs)

    @staticmethod
    def _response(content: str, ok: bool = True) -> ToolResult:
        # 知识库内容需要回填给模型，结合用户问题整理为自然语言后再回复。
        return ToolResult(ok=ok, content=content)
