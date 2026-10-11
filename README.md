# AI 核心处理后端

当前包含项目骨架、健康检查、低延迟 SSE 聊天、统一 OpenAI 兼容模型接口、四轮近期上下文、结构化长期记忆、静态/动态 Prompt 组装、Function Calling 工具调度，以及知识库、定位、天气、联网搜索、分类 RSS 新闻和本地音乐匹配工具。

## 运行

```powershell
cd D:\project\dtvoice\dtvoice
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item config.example.yaml config.yaml
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

在 `config.yaml` 中填写默认模型的 `base_url`、`api_key` 和 `model`。未填写模型配置时，服务仍可启动，但健康状态会显示为 `degraded`。MySQL 是必需依赖：先创建独立的 dtvoice 数据库，再填写 `mysql.host`、`mysql.port`、`mysql.user`、`mysql.database`；密码可通过 `DTVOICE_MYSQL_PASSWORD` 环境变量提供。其他连接项也可用 `DTVOICE_MYSQL_HOST`、`DTVOICE_MYSQL_PORT`、`DTVOICE_MYSQL_USER`、`DTVOICE_MYSQL_DATABASE` 覆盖。启动时自动创建 `chat_history` 和 `memories` 表；账号需要建表权限。

每个模型档案可用统一的 `reasoning.mode: disabled` 表示关闭思考，再用 `reasoning.adapter` 适配实际服务：本地 Qwen/vLLM 使用 `vllm_qwen`，SiliconFlow 使用 `siliconflow`，直连支持 `thinking.type` 的服务使用 `openai_thinking`。必要时可通过 `request_extra_body` 补充供应商字段；关闭思考的适配字段会优先覆盖冲突配置。

## 健康检查

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

## 流式聊天

`POST /api/chatstream` 返回 `text/event-stream`。必须填写 `user_id` 和 `prompt`；`device_id`、`model`、`location`、`constitution` 和 `device_area` 可选。相同 `user_id` 在内存会话有效期间沿用短期上下文；调用方应为不同用户提供不同 ID。

```json
{
  "user_id": "user-001",
  "device_id": "device-001",
  "prompt": "讲一个简短的笑话",
  "model": "default"
}
```

每个 SSE 帧只包含 `data:`。文字片段为 `{"success":true,"message":"false","data":"回复片段"}`；MySQL 成功提交用户和助手两条记录后，结束帧为 `{"success":true,"message":"true","data":"用户记录ID,助手记录ID"}`。流中失败返回 `{"success":false,"message":"错误说明","data":null}`，且不会发送成功结束帧。请求参数错误在流开始前返回 HTTP 400 JSON。用户明确告别时，服务端仍会清空内存会话，并把告别文本按相同格式输出及保存。公共场所模式下，短期记忆空闲 15 分钟、显式结束或服务重启后清除；MySQL 聊天记录不随内存会话清除。

可主动创建或结束会话：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8080/sessions
Invoke-RestMethod -Method Delete http://127.0.0.1:8080/sessions/<session_id>
```

`POST /sessions` 返回的随机 `session_id` 可作为新请求的 `user_id`；`DELETE /sessions/{session_id}` 只清除内存上下文，不删除 MySQL 聊天记录。

客户端已知精确坐标时可附带 `location`，它会覆盖 IP 定位：

```json
{
  "user_id": "user-001",
  "prompt": "我这里天气如何？",
  "location": {"latitude": 31.2304, "longitude": 121.4737}
}
```

## Function Calling

Agent 已支持 OpenAI 兼容的流式 Function Calling：聚合工具参数分片、并发执行白名单工具、将结果写回 `tool` 消息，并由模型生成最终回答。天气、联网搜索、新闻、知识库和定位等耗时工具会先启动后台任务，再由程序立即流式发送一条受控的自然过渡语；该过程不额外调用模型，也不会等待过渡语发送完成后才启动工具。每个请求最多调用模型两次、执行一轮工具；工具超时、未知工具、参数错误和执行异常都会作为受控结果回填给模型。

系统提示词会根据当前真实注册工具生成简要能力列表；OpenAI 原生 Function Calling schema 仍通过独立的 `tools` 参数传递。

## Prompt、动态上下文与缓存

- `prompts/agent-prompt.txt` 保存稳定人设、核心规则、自然表达约束、工具策略、安全边界、动态上下文与长期记忆分区。
- 服务启动时严格校验并缓存模板，预热当前时间、日期、星期、农历，以及配置的默认地点“安徽亳州”和未来天气。
- 聊天请求构建系统提示词时不会为设备新位置主动查询天气；天气问题统一按需调用 `get_weather`，避免无关问题被天气网络请求阻塞。
- 当前时间按分钟刷新；日期、星期和农历按自然日刷新；位置、天气和外部上下文按各自 TTL 缓存。
- 设备传入坐标或公网 IP 时优先生成设备本地上下文，否则使用 `context.default_location`。
- 事实由程序确定性获取，正常回复仍由 LLM 自然组织；询问本地事实时不重复调用工具。

## 会话短期记忆

- `user_id` 是唯一短期会话隔离边界，`device_id` 仅代表公共设备，不能替代用户 ID。
- 最近 6 轮对话保留原始用户/助手消息，满足连续追问。
- 原始窗口溢出时，`prompts/session-summary.txt` 才会异步压缩旧消息，摘要只存在进程内，随会话一起删除。
- `session.idle_ttl_seconds` 默认 900 秒；`session.max_sessions` 限制内存中的匿名会话总数。

## 长期记忆（可选）

- `prompts/memory-long.txt` 定义时效性 40%、情感强度 35%、关联密度 25% 的三维度评估。
- 默认 `memory.enabled: false`。只有在用户明确身份、授权与隐私策略齐备时才应启用。
- 启用后每轮成功回复异步更新记忆，不阻塞 SSE；无效 JSON、超长输出或版本冲突不会覆盖上一版。
- 记忆默认限制为 900 字，合并重复信息，并清理低于 60 分且连续三次未提及的条目。
- `memory.profile` 可指定独立总结模型；留空时使用 `llm.default_profile`。

## 已接入工具

- `get_location`：客户端经纬度优先，否则按请求公网 IP 推断城市级位置；私网、本地 Postman 请求无法可靠定位。
- `get_weather`：可查指定城市，或基于用户坐标 / IP 城市查询；使用 JSON 天气接口并带短 TTL 缓存。
- `knowledge_search`：仅当 `knowledge` 配置中的 Embedding 与 Milvus 参数完整时才注册给模型。Milvus 使用 REST 搜索接口，与原项目的检索格式兼容。
- `web_search`：支持 Tavily、秘塔或 SearXNG；只有对应配置有效时才注册。
- `get_news`：并发读取配置中的 RSS 分类源。可查“社会新闻”“国际新闻”“财经新闻”；未指定分类时汇总 `default_sources`。不抓取新闻详情页。
- `find_music`：匹配本地 `music` 目录中的音频文件，返回歌曲名称和相对文件标识，不播放音频。

请将 `config.example.yaml` 复制为本地 `config.yaml`，填入模型与 Milvus 的实际参数；真实密钥不会写入仓库。

## 测试

```powershell
.\.venv\Scripts\python -m pytest -q
```
