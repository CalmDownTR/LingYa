# 项目架构

> 由 Architect 维护。描述实际代码结构，不是设计文档里的理想状态。
> 输入：`lingya/` 代码 + `plan/roadmap.md` + `decisions/ADR-*.md` + `specs/capabilities.md` + `reference/glossary.md`
> 更新时机：每个版本交付后，由 TR 触发。

---

## 1. 架构风格

模块化单体（Modular Monolith）。EventBus 异步解耦模块，Protocol 定义接口契约。

**决策依据**：单人开发、单用户单 Agent、MindEngine 有状态连续演化。详见 ADR-002。

---

## 2. 模块拓扑

```
lingya/
├── mind/              — 心智引擎（纯计算层，零框架依赖）
│   ├── engine.py          MindEngine 7 步事件管道 + build_static_prompt() + reload_config() 热重载 + TONE_PRESETS 5 档预设
│   ├── config.py          OCEAN/PAD/ToneMatrix/IdentityAnchor 配置模型
│   ├── affect.py          OCC 22 情绪分类 + PAD 弹簧演化 + OCEAN 漂移
│   ├── dynamics.py        IPC 5 状态状态机（agency/communion 双轴）
│   ├── tone.py            PAD→ToneMatrix（warmth/formality/humor）+ 阶段检测
│   ├── guard.py           身份漂移守卫（embedding 余弦相似度 + 重锚提示）
│   ├── state.py           MindState / PADPoint 运行时状态模型（pydantic）
│   └── __init__.py        公开导出：MindEngine, MindState, MindConfig 等
│
├── memory/            — 记忆系统
│   ├── store.py           EnhancedMemoryStore（ChromaDB PersistentClient 后端）
│   ├── reflection.py      反思树（importance 阈值触发，fire-and-forget）
│   └── __init__.py        公开导出：EnhancedMemoryStore, MemoryStore
│
├── gateway/           — 网关与通信
│   ├── daemon.py          GatewayDaemon（纯生命周期；Traceloop.init() 接入点；启动时检查 web/dist/ 并打印访问 URL）
│   ├── server.py          FastAPI app（create_app 创建；POST /chat SSE 流 + GET/PUT /settings/* + Session CRUD + StaticFiles mount web/dist/ + /shutdown）
│   ├── router.py          MessageRouter（dict-in/dict-out 路由；_handle_chat_streaming 公开生成器 + _extract_text_content 静态方法 + _handle_chat_invoke 向后兼容 + _handle_settings 热重载 + _handle_session CRUD；astream_events 前过滤 _subagent_factory）
│   ├── auth.py            HTTPBearer dependency（LINGYA_API_KEY 校验 + auth_enabled 开关）
│   └── background.py      BackgroundRunner（heartbeat/diary/decay 三循环）
│
├── tools/             — Agent 工具注册
│   ├── memory_tools.py    create_memory_tools(memory_store) → remember/recall 工具列表
│   └── __init__.py
│
├── storage/           — 数据库
│   ├── db.py              Database（aiosqlite 封装，WAL 模式，迁移管理；会话数据存 LangGraph checkpoints）
│   └── migrations.py      SQL 迁移（v1 → v5）
│
├── protocols.py       — IMemoryStore / ILLMBackend Protocol 接口
├── events.py          — EventBus（asyncio pub/sub，3 个预定义事件）
├── app.py             — ApplicationBuilder（链式组装）+ Application dataclass
├── llm.py             — LiteLLMModel(BaseChatModel) 适配 litellm.completion()，替代 ChatOpenAI 直接使用；支持 100+ provider
├── transformers.py    — LingYaInnerProcessTransformer（通过 astream_events(transformers=[...]) 内联传递到 LangGraph v3 StreamMux）
├── config.py          — Config / LLMConfig / OtelConfig（yaml → pydantic）
├── reflection.py      — generate_opening_line()（"我刚才在想..."开场白）
├── diary.py           — 日记生成/列表/读取
└── __main__.py         — 控制台入口点（lingya start/stop/status）
```

### 已删除模块

| 文件 | 删除版本 | 原因 |
|------|----------|------|
| `gateway/protocol.py` | v0.8.1 | RFC 6455 帧编解码不再需要（WebSocket → HTTP+SSE） |
| `observability.py` | v0.8.3 | 手动 `init_observability()` 被 `Traceloop.init()` 替代（ADR-007） |
| `storage/conversations.py` `turns.py` | v0.9.0 | 废弃的 conversations/turns 表删除；会话数据唯一来源为 LangGraph checkpoints |
| `cli.py` | v0.9.4 | CLI 交互式聊天删除；Web UI 为唯一聊天入口（ADR-009） |
| `gateway/client.py` | v0.9.4 | GatewayClient 的唯一消费者是 CLI；Web UI 通过 fetch + ReadableStream 直接消费 SSE（ADR-009） |

### Web UI 前端（v0.9.0 新增）

```
web/
├── package.json        Vite 6 + React 19 + TypeScript + Tailwind CSS 4
├── vite.config.ts      dev proxy → FastAPI:8765
└── src/
    ├── App.tsx              React Router（SPA 路由，chat/settings 单页）
    ├── main.tsx             入口，挂载到 #root
    ├── types.ts             TypeScript 类型定义
    ├── lib/
    │   └── api.ts           TanStack Query 5 查询/变更（fetch + Bearer token）
    ├── hooks/
    │   ├── useSSE.ts        fetch + ReadableStream 消费 SSE（POST /chat）
    │   └── useApi.ts        fetch 封装（Bearer auth）
    └── components/
        ├── chat/
        │   ├── ChatWindow.tsx    聊天主窗口（路由 "/"）
        │   ├── MessageList.tsx   消息列表
        │   ├── MessageBubble.tsx 消息气泡（react-markdown 渲染 + 流式光标动画 + ContentBlock 防御性解析）
        │   ├── ChatInput.tsx     输入框
        │   └── PhaseIndicator.tsx 过程阶段指示器（recalling/thinking/generating 三阶段动画 + 记忆召回计数）
        ├── settings/
        │   ├── SettingsPanel.tsx     设置主面板
        │   ├── OCEANSliders.tsx      OCEAN 五维度滑块
        │   ├── IdentityEditor.tsx    身份名 + 个性简述编辑
        │   └── TonePresetPicker.tsx  语气预设选档（5 档）
        └── sessions/
            ├── SessionDrawer.tsx     会话抽屉
            └── SessionItem.tsx       会话条目
```

构建产物 `web/dist/` 不入 git，daemon 启动后通过 FastAPI StaticFiles mount 到 "/"。

### 依赖方向

```
Traceloop.init()（Traceloop SDK → 全局 OTel TracerProvider）
    └── 自动 instrument: LangChain + OpenAI + ChromaDB + FastAPI

Web UI（web/dist/ 构建产物）
    └── FastAPI StaticFiles mount at "/"（html=True，SPA fallback）
        ├── 显式路由优先匹配（/chat /mind /memory /diary /session /settings /health /shutdown /stats）
        └── 未匹配路径 → index.html → React Router 接管

ApplicationBuilder
    ├── Config (yaml)          ← lingya/config.py
    ├── MindConfig (yaml)      ← lingya/mind/config.py
    ├── Database               ← lingya/storage/db.py (aiosqlite)
    ├── ChatOpenAI             ← langchain-openai
    ├── EnhancedMemoryStore    ← IMemoryStore (Protocol) ← ChromaDB
    ├── MindEngine             ← IMemoryStore + llm_call inject + EventBus
    ├── EventBus               ← lingya/events.py (asyncio)
    └── DeepAgent              ← DeepAgents + Checkpointer + memory tools

GatewayDaemon
    ├── Traceloop.init()       ← gated by config.otel.enabled
    └── ApplicationBuilder → Application → MessageRouter → FastAPI app
        └── BackgroundRunner
                ├── _heartbeat_loop      → MindEngine.idle_tick()
                ├── _diary_scheduler     → diary.py (via engine.config)
                └── _decay_loop          → MemoryStore.apply_decay()
```

---

## 3. 数据流

### 对话路径（HTTP + SSE — v0.8.1+）

```
HTTP Client (httpx / fetch)
    → POST /chat {"text": "..."}, Authorization: Bearer
    → FastAPI endpoint → sse_generator() → router._handle_chat_streaming()
        ├── MindEngine.get_prompt_fragment()
        ├── asyncio.create_task(MindEngine.process_event(event))  [并发执行，不等 LLM 流结束]
        ├── agent.astream_events(input, config, version="v3")
        │   │   # transformers 通过 astream_events(transformers=[create_lingya_transformer]) 内联传递。
        │   │   # 调用前过滤 agent.stream_transformers 中的 _subagent_factory（按 __name__ 匹配）
        │   │   # 以避免 LangGraph v3 StreamMux key 冲突。finally 块中恢复原始 tuple。
        │   ├── method="messages" + text-delta → emit chat.delta
        │   ├── method="lingya_inner" → emit process.phase / memory.recall
        │   └── method="messages" + message-finish → 收集完整文本
        │
        ├── asyncio.wait_for(engine_task, timeout=0.5)  [流结束后最多等 0.5s]
        ├── asyncio.create_task(check_response_alignment(text))  [fire-and-forget]
        └── emit mind.transition  [含 engine_ms 计时]

SSE 事件帧 → ReadableStream (Web) / httpx aiter_lines() (运维命令 `--diary`)
Web UI 通过 fetch + ReadableStream 消费相同 SSE 帧（EventSource 不支持 POST），
帧格式统一，复用同一套 router._handle_chat_streaming() 生成器。
```

### Settings 热重载路径（v0.9.0 新增）

```
PUT /settings/ocean|identity|tone  /  POST /settings/reset
    → router._handle_settings(payload)
        → engine.reload_config(config_partial)
            ├── 更新 self.config.ocean / identity / tone_matrix（内存）
            ├── 重新计算 PAD baseline（OCEAN 变更时）
            ├── 重建 self._static_prompt（build_static_prompt）
            ├── save_state(db) → SQLite mind_state 表
            └── 200 { ok: true }

走直接调用路径，不走 EventBus（同步需求，用户改完立即确认生效）。
```

### Session 管理路径（v0.9.0 新增）

```
POST /session (new|switch|delete)  /  GET /session/list|current|history
    → router._handle_session(payload)
        ├── new:  生成 thread_id → 文件持久化（current_session.txt）
        ├── switch: 校验 thread_id 存在 → 切换 → 文件持久化
        ├── delete: 删除 checkpoints 行
        ├── list:   SELECT checkpoints GROUP BY thread_id ORDER BY MAX(checkpoint_id)
        ├── current: 返回当前 session info
        └── history: agent.aget_state() → 提取 HumanMessage/AIMessage → JSON

current_session.txt（data_dir 下）
    → daemon 重启后恢复上次 thread_id，无需手动选择会话
```

### OTel Trace 路径（v0.8.3）

```
Traceloop.init() 一行启动
    ├── LangChain (agent/chains/tools)     → span 自动
    ├── OpenAI / ChatOpenAI               → span（token/cost/latency）
    ├── ChromaDB (memory query/insert)     → span 自动
    ├── FastAPI (HTTP request)            → span（FastAPIInstrumentor 单独一行）
    └── mind.process_event                → 手动 span（PAD/OCEAN/IPC attributes）

OTel Span → OTLP Exporter → Console (v0.8.3) / Langfuse (v1.0)
    ↑ 直连后端，不走 Gateway 中继
```

### MindEngine.process_event() 管道（实际 7 步）

```
1. OCC + IPC 合并评估    occ_ipc_process() → 1 次 LLM 调用（1.5s 超时）
2. PAD 弹簧演化          evolve_pad(current_pad, occ_pull, ocean_baseline)
3. IPC 状态转移          ipc_to_state() + next_ipc_state()（纯计算）
4. 阶段检测 + 动态语气   detect_stage() + compute_dynamic_tone()（纯计算）
5. 重要性预评分 + 存储    rule_based_importance() → store_with_importance()
                         └→ asyncio.create_task(_deferred_importance_score) [bg]
6. 反思检查              cumulative_importance ≥ threshold?
                         └→ asyncio.create_task(check_and_reflect) [fire-and-forget]
7. OCEAN 渐进漂移         ocean_drift() 每 10 轮执行（纯计算）
   Auto-persist           save_state(db) → SQLite
```

> **注意**：记忆召回（检索）发生在 agent 内部（DeepAgents 的 tool calling），
> 不属于 MindEngine 管道。MindEngine 只负责心智状态演化。

### 后台路径

```
BackgroundRunner（daemon 内，3 个 asyncio Task）
    ├── _heartbeat_loop (60s)  → MindEngine.idle_tick()     [PAD 空闲漂移]
    ├── _diary_scheduler (1h)  → diary.py                   [stub — transcript source not yet wired]
    │   └── should_generate_diary() → _try_generate_diary() logs stub message
    │       (未来：从 LangGraph checkpoint 读历史，重新接线)
    └── _decay_loop (24h)      → MemoryStore.apply_decay()  [记忆衰减]
```

### EventBus 事件流（v0.7 当前）

```
MindEngine                          BackgroundRunner
    │                                    │
    ├── publish(MIND_STATE_CHANGED)      │   [内部使用]
    │                                    │
    │                                    ├── publish(DIARY_READY)
    │                                    │   [内部使用]
    │                                    │
    │                                    └── publish(MEMORY_DECAYED)
    │                                        [内部使用]
    │
    └── v0.9.0 当前: EventBus 已就绪，但无消费者订阅
        Settings 热重载走直接调用路径（ADR-008），不经过 EventBus。
        EventBus 保留给 v0.10 Dreaming 整合的异步场景（fire-and-forget）。
```

### 流式事件帧（v0.8）

5 种事件经由 SSE 推送至客户端：

| 事件 | 来源 | 负责方 | 帧格式 |
|------|------|--------|--------|
| process.phase | Transformer 检测 node 切换 | `transformers.py` | `{"type":"event","event":"process.phase","payload":{"phase":"thinking"}}` |
| memory.recall | Transformer 检测 memory tool call | `transformers.py` | `{"type":"event","event":"memory.recall","payload":{"count":3,"top_match":"..."}}` |
| chat.delta | stream.messages → text-delta | `router.py` | `{"type":"event","event":"chat.delta","payload":{"content":"我记"}}` |
| mind.transition | MindEngine.process_event() 后（与 LLM 流并发执行，最晚 0.5s 超时） | `router.py` | `{"type":"event","event":"mind.transition","payload":{"pad":{...},"occ_emotion":"...","ipc":"..."}}` |
| chat_response | _handle_chat 返回值 | `server.py` | `{"type":"chat_response","payload":{"meta":{"engine_ms":...},"text":"..."}}` |

旧客户端忽略 `type: "event"` 帧，仍能收到 `chat_response`。

---

## 4. 关键接口

| 接口 | 类型 | 位置 | 关键方法 |
|------|------|------|----------|
| IMemoryStore | `@runtime_checkable` Protocol | `protocols.py` | warmup / store / search / list_all / delete / recover / store_with_importance / score_importance / update_importance / search_weighted / apply_decay / get_cumulative_importance |
| ILLMBackend | `@runtime_checkable` Protocol | `protocols.py` | ainvoke(messages) |
| EventBus | 具体类 | `events.py` | subscribe(event_type, handler) / publish(event_type, **kwargs) |
| Application | dataclass | `app.py` | config, model, memory, engine, agent, checkpointer, event_bus, teardown() |
| ApplicationBuilder | 具体类 | `app.py` | with_database / with_model / with_memory / with_event_bus / with_engine / with_agent / build() |
| GatewayDaemon | 具体类 | `gateway/daemon.py` | start() / shutdown() / is_running() |
| MessageRouter | 具体类 | `gateway/router.py` | route(dict) → dict；`_handle_chat_streaming(messages, config, user_text)` 公开生成器；`_handle_chat_invoke(payload, emit)` 向后兼容；`_handle_settings(payload)`；`_handle_session(payload)` 含文件持久化 |
| BackgroundRunner | 具体类 | `gateway/background.py` | start() / stop() / is_running |
| Database | 具体类 | `storage/db.py` | initialize() / close() / conn |
| MindEngine | 具体类 | `mind/engine.py` | process_event(event) / idle_tick() / get_tone_params() / get_prompt_fragment() / check_response_alignment() / load_state() / save_state() / reload_config(config_partial) / TONE_PRESETS（类常量，5 档预设） |

### v0.8.3 变更

| 变更 | 详情 |
|------|------|
| 删除 `ApplicationBuilder.with_observability()` | 被 `daemon.py` 内 `Traceloop.init()` 替代 |
| 删除 `Application.tracer` 属性 | OTel tracer 通过 `trace.get_tracer()` 全局获取，不再构造传参 |
| 删除 `MindEngine.__init__(tracer=)` | 同上 |
| 删除 `BackgroundRunner.__init__(tracer=)` | 同上 |
| 新增 `FastAPIInstrumentor.instrument_app(app)` | 在 `create_app()` 内，自动 instrument HTTP 请求 |
| 删除 `observability.py` | 整个文件删除（ADR-007） |

### IMemoryStore 异步化状态

当前 v0.9.0：12 个方法中 **11 个是同步**（def），仅 `score_importance` 是 `async def`。
ChromaDB PersistentClient 无 async API，调用方直接同步调用。

v0.10 计划：全部改为 `async def`，同步 ChromaDB 调用包 `asyncio.to_thread()`。

### 预定义事件

```python
MIND_STATE_CHANGED = "mind_state_changed"   # MindEngine 状态变更后
DIARY_READY = "diary_ready"                 # 日记生成后
MEMORY_DECAYED = "memory_decayed"           # 记忆衰减后
```

---

## 5. 技术栈

| 层 | 技术 | 版本/备注 |
|----|------|-----------|
| 语言 | Python | ≥3.12 |
| Agent 框架 | DeepAgents | create_deep_agent + StateBackend + SummarizationToolMiddleware；基于 LangGraph (>=1.2.7)，支持 stream_events(v3)；langchain>=1.3.1, langchain-core>=1.4.8 |
| LLM | DeepSeek（默认）/ OpenAI / Ollama / 100+ provider | LiteLLMModel(BaseChatModel) 适配 litellm.completion()；model name 格式 `provider/model_name` |
| 对话持久化 | LangGraph Checkpointer | AsyncSqliteSaver（SQLite） |
| 向量存储 | ChromaDB | PersistentClient，内置 embedding |
| 状态持久化 | aiosqlite | MindEngine 状态 + schema 迁移（v1→v5） |
| 传输层 | FastAPI + uvicorn | HTTP + SSE；REST 端点（/chat /mind /memory /diary /session /settings /shutdown /health /stats）+ StaticFiles mount web/dist/（SPA fallback） |
| HTTP 客户端 | httpx（运维命令）+ fetch（Web UI） | useSSE fetch + ReadableStream；EventSource 不支持 POST |
| 可观测 | OpenLLMetry (Traceloop SDK) | `Traceloop.init()` 一行覆盖 LangChain/OpenAI/ChromaDB；`FastAPIInstrumentor` 单独一行 |
| 可观测（用户面向） | `time.monotonic()` 掐点 | engine_ms + sse_hop_ms，注入响应 meta 字段；与 OTel trace 并存 |
| 控制台入口 | `[project.scripts]` + argparse | `lingya start/stop/status`；`lingya/__main__.py` |
| 前端框架 | React 19 + TypeScript | Vite 6 构建，SPA |
| 前端样式 | Tailwind CSS 4 | CSS-first 配置，dark-only（v0.9） |
| 前端数据 | TanStack Query 5 + React Router 7 | Server state + SPA 路由 |
| 前端测试 | Vitest + @testing-library/react | happy-dom 环境 |
| 前端包管理 | npm | 独立 package.json，web/ 目录 |
| 配置 | pydantic + PyYAML | config.yaml（系统）+ agent_config.yaml（人格） |
| 包管理 | uv + pyproject.toml | traceloop-sdk>=0.61.0（v0.8.3 新增） |
| 测试 | pytest + pytest-asyncio | |
| 评估 | deepeval | 依赖已声明，尚未集成到 CI |

---

## 6. ADR 引用

| 决策 | ADR | 版本 |
|------|-----|------|
| 架构风格（模块化单体 + EventBus + Protocol） | ADR-002 | v0.7.0 |
| 可观测性（OTel → Langfuse v1.0） | ADR-003 | v0.7.0 |
| Inner Process Streaming（LangChain stream_events v3 + Custom Transformer，内联传递 + _subagent_factory 过滤；MindEngine 并发执行） | ADR-004 (Amendment 1, Amendment 2) | v0.8.0, v0.9.2, v0.9.3 |
| HTTP + SSE 传输层（WebSocket → FastAPI + uvicorn） | ADR-005 | v0.8.1 |
| 进程生命周期管理（CLI `/stop` + `main.py --stop` 优雅关闭入口） | ADR-006 | v0.8.2 |
| 自动 Instrumentation（OpenLLMetry 替代手动 OTel span） | ADR-007 | v0.8.3 |
| 前端技术栈与 Settings/Session API 契约 | ADR-008 | v0.9.0 |
| 抛弃 CLI 聊天交互，专注 Web UI，重构运维命令 | ADR-009 | v0.9.4 |

---

## 7. 当前版本

v0.9.4（最后更新：2026-07-06）

### 代码与架构文档一致性

| 检查项 | 状态 |
|--------|------|
| 模块拓扑与 `lingya/` 目录一致 | ✅ `observability.py` / `protocol.py` / `cli.py` / `gateway/client.py` 已删除；`llm.py` / `__main__.py` 新增 |
| MindEngine 管道步数与实际代码一致 | ✅ 7 步 |
| IPC 位置标注正确 | ✅ dynamics.py |
| MemoryStore 接口异步化状态正确 | ✅ 当前仅 score_importance 是 async |
| 传输层：FastAPI + uvicorn | ✅ ADR-005 已落地 |
| 流式：SSE 事件帧 + Custom Transformer（astream_events inline 传递 + _subagent_factory 过滤） | ✅ ADR-004 Amendment 1 + 2 已落地；v0.9.2 并发执行 + v0.9.3 inline 回退 |
| 进程生命周期：`--stop` + `--status` | ✅ ADR-006 + ADR-009 已落地 |
| OTel：OpenLLMetry `Traceloop.init()` 替代手动脚手架 | ✅ ADR-007 已落地 |
| `self._tracer` 传播链已清除 | ✅ `app.py` / `engine.py` / `background.py` 无 tracer 引用 |
| 回归测试全绿 | ✅ 345 passed |
| web/ 模块与 ADR-008 一致 | ✅ Vite 6 + React 19 + TypeScript + Tailwind 4 + TanStack Query 5 + React Router 7 |
| Settings CRUD 端点与 ADR-008 一致 | ✅ GET/PUT /settings/* + POST /settings/reset |
| Session CRUD 端点与 ADR-008 一致 | ✅ POST /session + GET /session/list\|current\|history |
| SPA fallback 与 ADR-008 一致 | ✅ StaticFiles mount "/" + html=True，显式路由优先 |
| Settings 热重载走直接调用路径（非 EventBus） | ✅ 与 ADR-008 决策一致 |
| engine.reload_config() 已实现 | ✅ 支持 ocean/identity/tone_preset/reset 四种操作 |
| TONE_PRESETS 5 档预设已定义 | ✅ warm/neutral/cool/passionate/gentle |
| conversations/turns 表已删除 | ✅ 会话数据唯一来源为 LangGraph checkpoints |
| LiteLLM 适配层 | ✅ `lingya/llm.py` LiteLLMModel(BaseChatModel) 通过 litellm 支持 100+ provider；ApplicationBuilder.with_model() 已切换 |
| SubagentTransformer 冲突修复 | ✅ router.py 调用 astream_events 前过滤 _subagent_factory，finally 块恢复 |
| ContentBlock 格式修复 | ✅ _extract_text_content / _extract_text_content_from_value 覆盖三条路径；前端 MessageBubble 防御性解析 |
| Web UI PhaseIndicator | ✅ PhaseIndicator.tsx 三阶段动画 + 记忆召回计数 |
| CLI 聊天已删除 | ✅ `cli.py` / `gateway/client.py` 已删除；Web UI 为唯一聊天入口（ADR-009） |
| 运维命令已就绪 | ✅ `main.py` start/stop/status/diary；`lingya` 入口点（ADR-009） |
| Dockerfile 已添加 | ✅ 最简 Dockerfile，python:3.12-slim + Node.js + uv sync + web build（ADR-009） |
| rich 依赖已移除 | ✅ pyproject.toml 中已删除 |
| MindEngine embedding_fn 已注入 | ✅ `EnhancedMemoryStore.get_embedding_fn()` → `check_response_alignment()` 真正运行 |
| GatewayClient 引用已清除 | ✅ `gateway/__init__.py` / `tests/` / `main.py` 中无残留 |

### v0.9.0 已完成

- 新增 web/ 顶层目录（Vite + React SPA）
- 新增 gateway 端点：GET/PUT /settings/*（5 个）+ POST /settings/reset + Session CRUD 增强（list/current/history）
- 新增 engine.reload_config() + TONE_PRESETS
- 新增 gateway server StaticFiles mount web/dist/（SPA fallback）
- 新增 MessageRouter._handle_chat_streaming() 公开为生成器（server.py 直接调用）
- 删除废弃的 storage/conversations.py + turns.py
- 新增 ADR-008

### v0.9.2 已完成（2026-07-03）

- P1: MindEngine.process_event() 改为与 LLM streaming 并发执行（`asyncio.create_task`），消除 token 结束到 "complete" 信号的可见延迟。流结束后最多等 0.5s。
- P2: LingYaInnerProcessTransformer 注册到 `LingYaStreamingMiddleware(AgentMiddleware)`（langchain>=1.3.2 最佳实践）。**注：此变更在 v0.9.3 被回退（参见 ADR-004 Amendment 2）。**
- P3: 清理 `transformers.py` 死代码（`_phase_emitted` 属性）。
- P4: Web SSE parser 健壮化（CRLF 行尾 + TypeScript 类型断言）。
- 依赖升级: `langchain>=1.3.2`, `langgraph>=1.2.7`, `langchain-core>=1.4.8`。

### v0.9.3 已完成（2026-07-04）

- 新增 `lingya/llm.py` — `LiteLLMModel(BaseChatModel)` 适配 litellm.completion()，支持 100+ provider。`ApplicationBuilder.with_model()` 切换为 `LiteLLMModel`。
- `LiteLLMModel.bind_tools()` 实现（满足 `create_deep_agent` 要求）+ `_stream()` 返回类型修正（`ChatGenerationChunk`）。
- LingYaStreamingMiddleware 回退（ADR-004 Amendment 2）：transformer 恢复为 `astream_events(transformers=[...])` 内联传递。新增 `_subagent_factory` 过滤 workaround（router.py save/restore 模式）。
- `MessageRouter` 新增 `_extract_text_content()` / `_extract_text_content_from_value()` 静态方法，处理 LangChain AIMessage.content 返回 ContentBlock list 的情况。覆盖 `_handle_chat_invoke` / `_handle_chat_streaming` / `_load_history` 三条路径。
- 新增 Web UI `PhaseIndicator.tsx` 组件 — recalling/thinking/generating 三阶段动画 + 记忆召回计数。
- 前端 `MessageBubble.tsx` 防御性解析 ContentBlock 格式。
- React hooks purity + set-state-in-effect lint 修复。

### v0.9.4 已完成（2026-07-06）

- **Blocker 修复**：pyproject.toml 版本号同步到 0.9.4；`.env.example` 新增 `LINGYA_API_KEY`；`MindEngine` 构造注入 `embedding_fn`（`EnhancedMemoryStore.get_embedding_fn()` → ChromaDB DefaultEmbeddingFunction），`check_response_alignment()` 真正运行身份守卫。
- **删除 CLI 聊天**：删除 `lingya/cli.py`（~415 行）和 `lingya/gateway/client.py`（~245 行）；移除 `rich` 依赖；删除 CLI 相关测试（`test_client.py` / `test_client_streaming.py` / `test_start.py`）；`gateway/__init__.py` 移除 `GatewayClient` 导出。Web UI 成为唯一聊天入口（ADR-009）。
- **重构运维命令**：`main.py` 重写：`daemon_main()` 前台启动、`stop_daemon()` SIGTERM 优雅关闭、`status()` 5 种状态判断（本地检查，不调 HTTP）、`diary()` HTTP 查询日记。
- **入口点 + Docker**：新增 `lingya/__main__.py`（argparse 子命令 start/stop/status）；`pyproject.toml` 新增 `[project.scripts]` 入口点（`lingya` 命令）；新增最简 `Dockerfile`（python:3.12-slim + Node.js + uv sync + web build）。
- **文档重写**：README.md 覆盖 daemon 启动、Web UI 访问、LiteLLM 配置、运维命令、Docker；`web/README.md` 定制为 LingYa 前端架构文档；`architecture.md` 同步模块变更（删除 cli.py/gateway/client.py，新增 __main__.py）。

### 待 v0.10 交付后更新

- 新增 `memory/dream.py`、`memory/health.py`
- 更新 IMemoryStore：全部方法 async 化完成
- 更新数据流：dream_job 路径 + BackgroundRunner 新增循环
- 更新当前版本号到 v0.10.0
- Web UI 心智状态可视化、日记面板、记忆侧栏（v0.9 不做 → v0.10）

---

## 8. 已知缺口

| 缺口 | 详情 |
|------|------|
| Diary transcript source | `_try_generate_diary` 是 stub — transcript 来源未接通。未来：从 LangGraph checkpoint 读对话历史（替代已删除的 turns 表） |
| MCP tools | 配置驱动的 MCP server 连接未接通（main.py 内 TODO） |
| Belief update | `belief.py` 已实现但未接入主管道 |
| CLI diary/memory/mind 查询 | CLI 聊天已删除（ADR-009），`--diary` 运维命令只能查看最新日记。memory/mind 查询需通过 `curl GET /memory` 和 `curl GET /mind` 或 Web UI |
| 最后一公里差异化 | 引擎产出方向性语气 Δ（A=10/A=90 之间 7-9 warmth 点），但最终中文文本差异对 LLM-judge 分类太微弱（pairwise accuracy ~40-60%） |
| Web UI light mode | dark-only 设计已交付。light-mode fallback 未实现 |
| 多用户隔离 | Session CRUD 支持多会话切换 + current_session.txt 持久化。仍为单用户 — 多用户 auth/数据隔离推迟到 v1.0 |
| Settings YAML 回写 | Web UI 设置变更热更新 MindEngine 内存状态（engine.reload_config()）+ 持久化到 SQLite。YAML 文件不会自动回写 — 重启后从 SQLite 恢复 |
| Web UI 心智状态可视化 | PAD/情绪/OCEAN/IPC 快照已在 GET /mind 可用，但前端未消费——设置面板仅展示可编辑项，无可视化仪表盘（v0.10） |
| Web UI 日记翻阅 | GET /diary 端点可用（CLI 可读），但前端日记面板未实现（v0.10） |
| Web UI 记忆侧栏 | GET /memory 端点可用，但前端无记忆浏览 UI（v0.10） |
| MindEngine 并发超时边缘情况 | 如果 engine_task LLM 调用 > 0.5s（当前 1.5s 超时，极少发生），mind.transition 帧可能在 engine_task 实际完成前推送。engine_task 仍在后台执行到完成，不影响状态正确性，仅该帧 engine_ms 指标不反映实际耗时。 |
