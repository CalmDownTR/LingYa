# LingYa Roadmap

## 版本总览

| 版本 | 主题 | 核心交付 | 状态 |
|------|------|----------|------|
| v0.1.0 | 基础骨架 | CLI + 人格系统 + 会话持久化 | ✅ 已完成 |
| v0.2.0 | 她记得你 | 结构化记忆 + 语义召回 + CLI 记忆管理 | ✅ 已完成 |
| v0.3.0 | 她是谁 | 机器人三原则 + "我刚才在想..."开场白 | ✅ 已完成 |
| v0.4.0 | 她的日记 | 定时周期写日记 + `/diary` 翻阅 + 成长支撑 | ✅ 已完成 |
| v0.5.0 | 她有心智 | mind 引擎：OCEAN+PAD+OCC 情感动力 + IPC 人际状态机 + 动态语气 + 信念锚定 + 身份守卫 + 反思树 | ✅ 已完成 |
| v0.6.0 | 多入口架构 | Gateway 常驻 daemon + WebSocket 协议 + 记忆衰减 + 后台心跳 | ✅ 已完成 |
| v0.6.1 | 可观测 | 管道 hook + 6 项耗时指标 + `lingya stats` + reply meta（-v 模式） | ✅ 已完成 |
| v0.7.0 | **架构瘦身** | ApplicationBuilder + EventBus + Protocol 接口 + OTel tracing | ✅ 已完成 |
| v0.7.1 | 整理 | pyproject.toml 版本号同步 + RELEASE_CHECKLIST + test 修复 + lint | ✅ 已完成 |
| v0.8.0 | 记忆进化 + 流式输出 | Dreaming 记忆整合 + 异步 MemoryStore + 记忆层级 + EventEmitter + 流式 chat | **📋 下一个** |
| v0.9.0 | 她的面孔 | Web UI 对话窗口 + 人格状态可视化 | 📋 计划中 | |
| v1.0.0 | 她是她自己 | 质量闭环 + 语音 + 安装体验 + Langfuse + 行为指令层 | 📋 远期 |
| v2.0.0 | 社区与生态 | 插件系统 + 多人格 + 多模态 | 📋 远期 |

## v0.7.0 — 架构瘦身

> 解决 GatewayDaemon God-class 问题，引入 EventBus 解耦模块，用 Protocol 接口打开存储替换空间，同时接入 OTel 为后续可观测性打基础。

### 架构方案

- [ADR-002: 可扩展架构设计](./ADR-002-scalable-architecture.md)
- [ADR-003: 可观测性架构设计](./ADR-003-observability.md)
- [ADR-004: Inner Process Event Streaming](./ADR-004-inner-process-streaming.md)

### 关键交付

| # | 交付物 | 改动 | 验收标准 | 状态 |
|---|--------|------|----------|------|
| 1 | **ApplicationBuilder** | `lingya/app.py` 新增 | ① `ApplicationBuilder().with_config().with_database().with_model().with_memory().with_engine().with_event_bus().with_agent().build()` 可组装完整 Application ② Builder 每步校验前置条件，缺失时抛明确异常 ③ CLI 直连模式复用同一 Builder | ✅ |
| 2 | **EventBus** | `lingya/events.py` 新增 | ① MindEngine 处理事件后发布 `MIND_STATE_CHANGED`，BackgroundRunner 通过 `subscribe` 接收 ② `DIARY_READY` 事件发布后，WebSocket Server 收到并推送给客户端 ③ handler 异常不阻塞其他 handler，被 logger.exception 捕获 ④ `EventBus` 可独立单元测试 | ✅ |
| 3 | **Protocol 接口** | `lingya/protocols.py` 新增 | ① `MindEngine.__init__` 签名接受 `IMemoryStore` 而非 `EnhancedMemoryStore` ② `EnhancedMemoryStore` 实现了 `IMemoryStore` Protocol（`@runtime_checkable` 通过） ③ 可用 Mock 实现替换 MemoryStore 进行测试 ④ `ILLMBackend` Protocol 允许未来接入多模型 failover | ✅ |
| 4 | **工具注册解耦** | `lingya/tools/memory_tools.py` 新增 | ① `@tool` 装饰器不在 daemon.py 内定义 ② `create_memory_tools(memory_store)` 返回可注册的工具列表 ③ daemon.py 不 import `langchain_core.tools` ④ 工具注册逻辑可独立单元测试 | ✅ |
| 5 | **WebSocket 认证** | `lingya/gateway/auth.py` 新增 | ① 连接后 5s 内未发送 `{"type": "auth", "payload": {"key": "..."}}` 则断开 ② key 匹配环境变量 `LINGYA_API_KEY` ③ `auth_enabled: false` 时跳过认证（开发模式） ④ 认证失败日志记录来源 IP | ✅ |
| 6 | **Daemon 瘦身** | `lingya/gateway/daemon.py` 修改 | ① daemon.py ≤ 100 行 ② Daemon 只负责生命周期（start/shutdown/signal），不包含任何组件组装逻辑 ③ 所有组装逻辑在 ApplicationBuilder ④ 删除 `_init_database/_init_model/_init_memory/_init_engine/_init_agent` 5 个方法 | ✅ |
| 7 | **OTel + OpenLLMetry** | `lingya/observability.py` 新增 | ① `Traceloop.init()` 自动 instrument LangChain/OpenAI/ChromaDB 调用 ② `MindEngine.process_event()` 内手动 span，记录 PAD/OCEAN/IPC 变化为 attributes ③ BackgroundRunner 3 个循环各有 span ④ `config.yaml` 新增 `otel.enabled` 开关，关闭时零开销 ⑤ 删除 `MindEngine._stats` 和 `MessageRouter._route_timings`（迁移到 OTel） ⑥ console exporter 模式下 `lingya start` 终端输出 trace 结构 | ✅ |

### 依赖关系

```
1. ApplicationBuilder ← 无依赖，最先做
2. Protocol 接口     ← 无依赖，可与 1 并行
3. EventBus          ← 依赖 ApplicationBuilder（Builder 中 with_event_bus()）
4. 工具注册解耦      ← 依赖 Protocol（IMemoryStore 参数）
5. WS 认证           ← 无依赖，可与 3-4 并行
6. Daemon 瘦身       ← 依赖 1+2+3+4 全部完成
7. OTel              ← 依赖 ApplicationBuilder（with_observability()），可与 5 并行
```

### 不动

- `mind/*` — 零改动（MindEngine 只改签名接受 Protocol）
- `memory/store.py` — 只加 Protocol 兼容（实现 IMemoryStore）
- `diary/*` — 零改动

### 不做

- 不引入微服务
- 不引入外部消息队列
- 不引入 Langfuse（v1.0 再做）
- 不做性能优化
- 不改 MindEngine 核心管道逻辑

---

## v0.8.0 — 记忆进化 + 流式输出

> 借鉴 OpenClaw 的 Dreaming 机制和 DeepAgents 的 consolidation 模式，让记忆不只是衰减，还会整合。同时引入 Inner Process Event Streaming，让交互不再是黑箱——用户能看到"她在回忆"、"她在思考"，而不是干等 10 秒。

### 架构决策

**为什么自建记忆而不使用 DeepAgents 原生 Store？**

DeepAgents 的长期记忆是 **Passive Memory**（文件加载到 system prompt，始终可见），LingYa 需要 **Active Memory**（向量检索 + 重要性评分 + 衰减，按需 recall）。两者解决不同范式：

| | DeepAgents Store (Passive) | LingYa Memory (Active) |
|---|---|---|
| 记忆位置 | 始终在 system prompt 里 | 向量数据库，按需检索 |
| 检索方式 | 不需要——已经在上下文 | 余弦相似度 + 加权排序 |
| 适合记忆量 | 少量结构化文档（几 KB） | 大量碎片化事实（成百上千条） |
| 衰减/遗忘 | 无——只增不减 | 三级衰减 + Dreaming 整合 |
| token 成本 | 每次对话付全量记忆 token | 只付检索结果 token |

LingYa 的 VISION 是"她是一个真实的人"——真人不会每次对话把所有记忆摊在面前，而是被线索触发才想起。Active Memory 更符合这个行为。

**不做底层 IVectorBackend 抽象**——IMemoryStore 已是正确的抽象层。每个 DB 一个实现类（ChromaDBMemoryStore / QdrantMemoryStore），共享加权/衰减逻辑在基类中。不需要更底层的向量存储接口。

**为什么用 Inner Process Streaming 而非 Deep Agents subagent streaming？**

LingYa 是单智能体系统，没有 subagent。Deep Agents 的 `stream.subagents` 投影的是多智能体结构（谁在干活），LingYa 需要的是单智能体内心过程（她在想什么）。详见 [ADR-004](./ADR-004-inner-process-streaming.md)。

### 架构方案

- [ADR-002 附录: OpenClaw 借鉴分析](./ADR-002-scalable-architecture.md)
- [ADR-004: Inner Process Event Streaming](./ADR-004-inner-process-streaming.md)
- IMemoryStore Protocol 异步化（v0.7 预留）

### 关键交付

| # | 交付物 | 改动 | 验收标准 |
|---|--------|------|----------|
| 1 | **IMemoryStore 异步化** | `lingya/protocols.py` + `lingya/memory/store.py` 修改 | ① Protocol 12 个方法全部改为 `async def` ② 当前 `EnhancedMemoryStore` 用 `asyncio.to_thread()` 包装同步 ChromaDB 调用（ChromaDB PersistentClient 无 async API） ③ MindEngine 调用 `await memory.store_with_importance()` 而非同步调用 ④ 所有调用方（engine/router/reflection/tools）加 `await` ⑤ 未来换 Qdrant（原生 AsyncQdrantClient）或远程 DB 时不需要改 Protocol |
| 2 | **Dreaming 记忆整合** | `lingya/memory/dream.py` 新增 | ① BackgroundRunner 新增 `dream_job()` 循环（默认每 7 天执行一次） ② 找出语义相似度 > 0.85 的记忆对 ③ 用 LLM 合并为更精炼的条目（保留原始时间线信息） ④ 删除原始条目，保留合并结果 ⑤ 发布 `MEMORY_DREAMT` 事件 ⑥ 合并前后记忆总数只减不增（不引入新信息） |
| 3 | **记忆层级** | `lingya/memory/store.py` 修改 | ① 新增 `memory_tier` 元数据：episodic / semantic / working ② `store_with_importance` 自动分级：importance > 0.8 → semantic，其余 → episodic ③ 搜索优先返回 semantic 层 ④ working memory 保留最近 5 轮对话上下文（不存 ChromaDB，纯内存） |
| 4 | **记忆健康度** | `lingya/memory/health.py` 新增 | ① WebSocket `/memory health` 路由返回：总条目数、各层级占比、平均重要性、最旧条目日期 ② 记忆总数 > 500 时在日记中提及"最近记了太多东西" ③ decay + dream 执行后发布 `MEMORY_HEALTH_CHANGED` 事件 |
| 5 | **EventEmitter** | `lingya/gateway/emitter.py` 新增 | ① `EventEmitter` 封装 StreamWriter，提供 `emit(event_type, **data)` 方法 ② 推送 `{"type": "event", "event": "...", "payload": {...}}` 结构化帧 ③ 旧客户端忽略 `type: event` 帧，向后兼容 ④ emitter.emit() 异常不阻塞主流程（日志记录后继续） |
| 6 | **流式 chat 输出** | `lingya/gateway/router.py` + `server.py` 修改 | ① `_handle_chat` 接受 `emitter` 参数 ② 推送 `process.phase` 事件（recalling → thinking → generating） ③ agent.ainvoke → astream_events，推送 `chat.delta` token 流 ④ 最终推送 `chat.complete` + 完整消息 ⑤ CLI 客户端实时打印 token（打字效果） ⑥ 不支持事件流的客户端仍收到完整 `chat` 响应帧 |

### 依赖

- v0.7 的 IMemoryStore Protocol
- v0.7 的 EventBus（事件发布）
- v0.7 的 BackgroundRunner + OTel span
- v0.7 的 WebSocket Server（提供 StreamWriter）

---

## v0.9.0 — 她的面孔

> 第一个面向用户的 UI。不只是对话窗口，还是她存在的证据——你可以看到她的心情在变，看到她在回忆什么。

### 架构方案

- [ADR-004: Inner Process Event Streaming](./ADR-004-inner-process-streaming.md) — Phase 2

### 关键交付

| # | 交付物 | 改动 | 验收标准 |
|---|--------|------|----------|
| 1 | **Web UI 对话窗口** | `lingya/web/` 新增 | ① 浏览器打开 `http://localhost:8765` 可对话 ② 支持 Markdown 渲染 ③ WebSocket 实时收发消息 + 消费事件流（process.phase / chat.delta / mind.transition） ④ 认证通过 URL 参数传入 |
| 2 | **人格状态可视化** | `lingya/web/` 新增 | ① PAD 三维坐标实时显示（随 mind.transition 事件更新） ② 当前 OCC 情绪 + IPC 状态显示 ③ OCEAN 五维雷达图 ④ 日记时间线浏览 ⑤ 数据通过 WebSocket `/mind` 路由获取 |
| 3 | **EventBus→WS Bridge** | `lingya/gateway/bridge.py` 新增 | ① `EventBusWSBridge` 订阅 EventBus 事件，通过 EventEmitter 推送到 WS 客户端 ② 连接断开时自动 unsubscribe ③ MindEngine publish MIND_STATE_CHANGED → 客户端收到 mind.transition 事件 ④ Memory 检索 → 客户端收到 memory.recall 事件 |
| 4 | **多模型 failover** | `lingya/model/` 新增 | ① 配置文件定义 failover chain：`[deepseek, openai, ollama]` ② 首选模型超时 30s 或返回 5xx → 自动切下一个 ③ 切换时日志记录，不丢弃用户消息 ④ 通过 ILLMBackend Protocol 实现 |

### 依赖

- v0.7 的 EventBus（推送通知给 Web UI）
- v0.7 的 Protocol 接口（ILLMBackend 支持多模型）
- v0.7 的 WS 认证（Web UI 连接需要认证）
- v0.8 的 EventEmitter + 流式 chat 输出（Web UI 消费事件流）
- v0.8 的记忆健康度（Web UI 展示）

---

## v1.0.0 — 她是她自己

> LingYa 1.0：质量可度量、安装即用、有声音、可审查。

### 关键交付

| # | 交付物 | 改动 | 验收标准 |
|---|--------|------|----------|
| 1 | **质量闭环** | CI + eval | ① 记忆召回准确率 > 85%（deepeval automated test） ② 人格一致性 > 90%（同一 OCEAN 配置下 5 轮对话语气评分标准差 < 0.5） ③ 核心路径端到端测试覆盖：chat → engine → memory → diary ④ CI 每次提交自动跑 eval，回归时阻断合并 |
| 2 | **Langfuse 集成** | `lingya/observability.py` 修改 | ① OTel exporter 从 console 切到 OTLP → Langfuse（自托管 Docker） ② LLM 调用 token/cost/latency 在 Langfuse Dashboard 可查 ③ Session 级别追踪（一次对话所有 LLM 调用聚合） ④ 与 deepeval 评分关联（trace → eval score） |
| 3 | **语音 I/O** | `lingya/voice/` 新增 | ① Web UI 支持语音输入（STT → 文本 → 管道） ② 支持语音输出（文本 → TTS → 播放） ③ 人格音色可配置（TTS speaker_id） ④ 延迟 < 3s（STT+LLM+TTS 端到端） |
| 4 | **安装体验** | pyproject.toml + CLI | ① `pip install lingya && lingya init` 5 分钟内完成首次对话 ② `lingya init` 交互式引导：选模型 → 填 API key → 命名她 ③ 默认配置开箱即用（DeepSeek + 默认人格） ④ Docker 一键部署（`docker compose up`） |
| 5 | **演化可审查** | `lingya/mind/audit.py` 新增 | ① 每次信念变更写入 `audit/` 目录（JSON Lines） ② OCEAN 漂移历史可查询（`/mind audit` 路由） ③ 身份重锚事件可查询 ④ 审计日志不可篡改（append-only） |
| 6 | **行为指令层** | DeepAgents Store 集成 | ① 启用 `use_longterm_memory=True` + `StoreBackend` ② agent 通过 `edit_file` 自编辑 `/memories/instructions.md`——可写的行为指令 ③ `/identity/*` 路径 read-only（底层原则和身份不可改） ④ 第三层记忆：结构化文档（Store）+ 碎片化事实（ChromaDB）+ 对话历史（Checkpointer） |

### 依赖

- v0.7 的 OTel（Langfuse 作为后端）
- v0.8 的记忆层级 + Dreaming（eval 目标）
- v0.9 的 Web UI（语音 I/O 的载体）

### 不做

- 移动端 App
- 商业化/付费功能
- 多人在线

---

## v2.0.0 — 社区与生态

> 开放 LingYa 的人格和工具生态，让每个人都能定义自己的"她"。

### 关键交付

| # | 交付物 | 验收标准 |
|---|--------|----------|
| 1 | **Plugin 系统** | ① `lingya plugin install <name>` 安装第三方工具/连接器 ② Plugin 通过 PluginAPI 注册（registerTool / registerHook） ③ Plugin 隔离：异常不崩溃主进程 ④ 至少 3 个示例 Plugin（天气/新闻/日历） |
| 2 | **多人格 marketplace** | ① 人格配置文件可导出/导入 ② 社区分享页面 ③ `lingya personality import <url>` 一键加载 ④ 多人格切换不丢失各自记忆 ⑤ CompositeBackend 路由按 `persona_id` 隔离（借鉴 DeepAgents 的 namespace 模式） |
| 3 | **多模态** | ① 图片输入（vision）——她能看懂你发的图 ② 手势/表情识别（可选） ③ 多模态输出（图片生成，可选） |

### 依赖

- v0.7 的 Protocol + EventBus（Plugin 注册）
- v0.7 的 AppBuilder（多实例组装）
- v1.0 的语音基础设施

---

## 已完成版本

### v0.2.0 — 她记得你

> 结构化语义记忆系统，用户在第 5 次对话时 LingYa 能展现对前序对话的记忆。

**实际交付**：ChromaDB PersistentClient + sentence-transformers 语义检索，`memory_store` / `memory_search` agent 工具，CLI `/memories` `/forget` `/remember` 管理。偏差：计划 `store.py` + `search.py` 分离，实际合并为 `MemoryStore` 类。2026-05-26 单日交付。

### v0.3.0 — 种子

> 种下两颗种子：她的底层边界（阿西莫夫三原则）、她的内在生活（"我刚才在想..."开场白）。

**实际交付**：三原则在 system prompt 根部注入，不可覆盖；开场白基于上次对话转录生成，不再说"你好"。偏差：原计划"无反思锚点则跳过"，实际改为"普通对话就说符合性格的开场白"——更自然。2026-05-26 同日交付。

### v0.4.0 — 她的日记

> TR 不在的时候 LingYa 留下了自己的痕迹——不是数值面板，是她写的、TR 能读到的日记。

**实际交付**：每次会话结束后 LingYa 以自己的身份写一篇内心独白（非对话摘要），`/diary` 翻阅，启动时提示"📔 我写了一篇新日记"。日记质量标准：有她的观察、没说出口的话、自我怀疑、性格痕迹。成长引擎推后到 v0.5。2026-05-27 交付。

### v0.5.0 — 她有心智

> 引入完整的心智引擎（mind），从 OCEAN 人格模型到 PAD 情感空间到 OCC 情绪决策树到 IPC 人际状态机——她的人格从静态配置变成动态运转的系统。

**实际交付**：8 个子系统——OCEAN 大五人格参数化 + PAD 三维情感弹簧演化 + OCC 22 情绪分类决策树 + IPC 5 状态状态机 + PAD→ToneMatrix 动态语气映射 + 信念锚定 + 身份漂移守卫 + 反思树 + OCEAN 渐进漂移。MindEngine 8 步事件管道 + SQLite 状态持久化。偏差：原计划只做 OCEAN→语言行为映射，实际引入了完整的 ALMA-PAD-OCC 情感三层模型——因为"人格→输出"的单层映射太薄，需要情感和心境作为中间层才能产生有动态感的语气。2026-05-30 交付。

**v0.5 验证（2026-06-01）**：引擎层区分度达标（tone Δ=7.9），但 DeepSeek native helpful style 使文本层 pairwise 准确率仅 40-60%。接受当前天花板，不再调参。

### v0.6.0 — 多入口架构

> 将 LingYa 从单进程 CLI 脚本进化为常驻 daemon + 多客户端架构。

**实际交付**：GatewayDaemon + asyncio WebSocket + `lingya start` + BackgroundRunner + 记忆三级衰减。2026-06-01 交付。架构方案：[ADR-001](./ADR-001-architecture-v06.md)。

### v0.6.1 — 可观测

> v0.6 把直连变成 `CLI → WebSocket → Engine`，但这条新链路的延迟没有度量。

**实际交付**：Engine 观测 hook + 6 项耗时指标 + `lingya stats` + reply meta（-v 模式）+ stats WebSocket 路由。

### v0.7.0 — 架构瘦身

> 解决 GatewayDaemon God-class 问题，引入 EventBus 解耦模块，用 Protocol 接口打开存储替换空间，接入 OTel 为后续可观测性打基础。

**实际交付**：
1. **ApplicationBuilder**（`lingya/app.py`）— 链式组装替代 Daemon 内 5 个 `_init_*` 方法，每步校验前置条件，`Application.teardown()` 统一异步资源清理
2. **EventBus**（`lingya/events.py`）— 纯 asyncio pub/sub，handler 异常不阻塞，3 个预定义事件（MIND_STATE_CHANGED / DIARY_READY / MEMORY_DECAYED）
3. **Protocol 接口**（`lingya/protocols.py`）— IMemoryStore 12 方法全覆盖 + ILLMBackend，MindEngine 依赖 Protocol 不依赖具体实现
4. **工具注册解耦**（`lingya/tools/memory_tools.py`）— `remember` / `recall` 工具从 daemon 闭包提取为独立模块，接受 IMemoryStore 参数
5. **WebSocket 认证**（`lingya/gateway/auth.py`）— WSAuth dataclass，5s 超时，LINGYA_API_KEY 校验，auth_enabled 开关
6. **Daemon 瘦身**（`lingya/gateway/daemon.py`）— 319→100 行，只管生命周期，组装全部委托 ApplicationBuilder
7. **OTel 可观测**（`lingya/observability.py`）— init_observability() 可开关，关闭时零开销，ConsoleSpanExporter 输出 trace

**计划外交付**：
- `/new` session 命令 — Router 新增 `session` 消息类型，`/new` 生成新 thread_id 清零对话历史（MindEngine 情感状态不受影响）
- `_handle_stats` deprecation notice — 旧 stats 端点返回 `deprecated: true` 提示迁移到 OTel
- 删除 `MindEngine._stats` deque 和 `MessageRouter._route_timings` — 迁移到 OTel metrics

**架构方案**：[ADR-002](./ADR-002-scalable-architecture.md) + [ADR-003](./ADR-003-observability.md) + [ADR-004](./ADR-004-inner-process-streaming.md)

### v0.7.1 — 整理

> v0.7.0 后的 housekeeping：补齐 pyproject.toml 版本号（0.2.0→0.7.0）、建立发布流程、修复测试和 lint。

**实际交付**：
1. **pyproject.toml 版本号同步** — 从 0.2.0 修正到 0.7.0（历史欠债：tag 版本与 pyproject.toml 不一致）
2. **RELEASE_CHECKLIST.md** — `.github/RELEASE_CHECKLIST.md` 固化发布前检查项和步骤
3. **test_start 修复** — 3 个测试因 `_find_port_owner` 新增调用导致 Popen mock 多了一次 lsof 调用，补充 mock `_find_port_owner` 修复
4. **lint 修复** — `client.py` 删除未使用的 `Any` import

---

## 决策点

| 节点 | 时机 | Go 条件 | No-Go 处理 |
|------|------|---------|------------|
| v0.7 重构完成 | Daemon 瘦身后 | ① 所有 v0.6 功能不退化 ② daemon.py < 100 行 ③ OTel console exporter 输出 trace | 检查 Builder 组装链，逐步回退 |
| v0.8 记忆进化 | Dreaming 上线后 | ① Dreaming 合并不引入虚假记忆（人工检查 5 次合并结果） ② IMemoryStore 异步化不引入性能退化 ③ 流式输出旧客户端仍能正常收到完整响应 | 缩小 Dreaming 范围，先做层级再做整合；流式降级为非流式 |
| v0.9 Web UI | UI 可交互后 | ① WebSocket 连接稳定 ② 人格状态实时更新 ③ 多模型 failover 至少测试 1 次切换 | Web UI 不阻塞 v1.0，可降级为纯 CLI |
| v1.0 就绪 | 全部交付后 | ① eval 通过 + 安装体验流畅 + Langfuse 可查 | 继续迭代，不赶版本 |

## 节奏原则

- **功能驱动发布**，Ready when ready
- **质量标准是"0 数据丢失"**，不是"0 bug"
- **不预估时间** — AI 编程很快，方案文档只写做什么、怎么做、验收标准
- **先改产品文档，再写执行计划** — 所有代码改动前，先确保 `product/` 下的 VISION/CAPABILITIES/ROADMAP 对齐

## 停车场

好想法，但不在当前范围：

- 人格社区分享/存储（v2.0）
- 多人格切换（v2.0）
- MCP 工具生态深度集成（v2.0 Plugin 系统）
- 记忆加密与零知识证明
- 群组对话/多人格协作
- 移动端 App
