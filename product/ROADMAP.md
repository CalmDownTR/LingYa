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
| v1.0.0 | 她是她自己 | 独立运行 + 质量闭环 + 社区就绪 | 📋 远期 |

## v0.2.0 — 她记得你

> 结构化语义记忆系统，用户在第 5 次对话时 LingYa 能展现对前序对话的记忆。

**实际交付**：ChromaDB PersistentClient + sentence-transformers 语义检索，`memory_store` / `memory_search` agent 工具，CLI `/memories` `/forget` `/remember` 管理。偏差：计划 `store.py` + `search.py` 分离，实际合并为 `MemoryStore` 类。2026-05-26 单日交付。

## v0.3.0 — 种子

> 种下两颗种子：她的底层边界（阿西莫夫三原则）、她的内在生活（"我刚才在想..."开场白）。

**实际交付**：三原则在 system prompt 根部注入，不可覆盖；开场白基于上次对话转录生成，不再说"你好"。偏差：原计划"无反思锚点则跳过"，实际改为"普通对话就说符合性格的开场白"——更自然。2026-05-26 同日交付。

## v0.4.0 — 她的日记

> TR 不在的时候 LingYa 留下了自己的痕迹——不是数值面板，是她写的、TR 能读到的日记。

**实际交付**：每次会话结束后 LingYa 以自己的身份写一篇内心独白（非对话摘要），`/diary` 翻阅，启动时提示"📔 我写了一篇新日记"。日记质量标准：有她的观察、没说出口的话、自我怀疑、性格痕迹。成长引擎推后到 v0.5。2026-05-27 交付。

## v0.5.0 — 她有心智

> 引入完整的心智引擎（mind），从 OCEAN 人格模型到 PAD 情感空间到 OCC 情绪决策树到 IPC 人际状态机——她的人格从静态配置变成动态运转的系统。

**实际交付**：8 个子系统——OCEAN 大五人格参数化 + PAD 三维情感弹簧演化 + OCC 22 情绪分类决策树 + IPC 5 状态状态机 + PAD→ToneMatrix 动态语气映射 + 信念锚定 + 身份漂移守卫 + 反思树 + OCEAN 渐进漂移。MindEngine 8 步事件管道 + SQLite 状态持久化。偏差：原计划只做 OCEAN→语言行为映射，实际引入了完整的 ALMA-PAD-OCC 情感三层模型——因为"人格→输出"的单层映射太薄，需要情感和心境作为中间层才能产生有动态感的语气。2026-05-30 交付。

**v0.5 验证（2026-06-01）**：引擎层区分度达标（tone Δ=7.9），但 DeepSeek native helpful style 使文本层 pairwise 准确率仅 40-60%。接受当前天花板，不再调参。


## v0.6.0 — 多入口架构

> 目标：将 LingYa 从单进程 CLI 脚本进化为常驻 daemon + 多客户端架构。CLI、Web UI、飞书 bot 等多个入口共享同一个 MindEngine 实例。架构方案详见 [ADR-001](./ADR-001-architecture-v06.md)。

### 为什么是现在

v0.5 交付了完整的 mind 引擎，但引擎的生命周期绑定在 CLI 进程上——启动、对话、退出、销毁。这不是一个"系统"，这是一个"脚本"。多入口架构是 v1.0 "独立运行"的基础设施前提——引擎必须先成为常驻服务，才能有不依附于会话时间的后台心跳、定时日记、PAD 自然漂移。

v0.5 验证（2026-06-01）结论：引擎层区分度存在（tone Δ=7.9），但 DeepSeek 的 native helpful style 使文本层差异不够稳定。当前阶段继续调参 ROI 低，架构演进优先级更高。

### 关键交付

| # | 交付物 | 说明 | 验收标准 |
|---|--------|------|----------|
| 1 | **Gateway Daemon** | 常驻进程，创建并持有 MindEngine 单例，管理 graceful shutdown | `lingya daemon` 启动后进程常驻，PID 文件 + SIGTERM 优雅退出 |
| 2 | **WebSocket 协议层** | asyncio WebSocket，消息路由（chat/mind/diary/memory） | curl 能通过 WebSocket 查询心智状态、触发对话 |
| 3 | **`lingya start` 一键启动** | 自动拉起 daemon（如未运行）+ attach CLI，掩盖底层复杂度 | 用户体验与 `uv run main.py` 一致，感知不到 daemon 的存在 |
| 4 | **BackgroundRunner** | PAD idle drift + 定时日记调度 | 无交互时 PAD 向基线缓漫游、日记定时触发 |
| 5 | **记忆衰减** | 三级衰减机制（见下方设计），BackgroundRunner 定时执行 | 该忘的淡出、不该忘的保留 |

### 记忆衰减设计

衰减的不是 `importance`（记录存入时的原始重要性，永久不变），而是独立的 `retrieval_weight`（检索权重）。`/memory recover` 恢复时重置 `retrieval_weight = importance`，不需要猜原始值。

**衰减公式（线性）**：

```
retrieval_weight = importance × max(0, 1 - days/180)
```

第 90 天权重减半，第 180 天归零。

**三级策略**：

| 级别 | importance | 衰减行为 | 理由 |
|------|-----------|---------|------|
| 🔒 关键记忆 | > 0.8 | `retrieval_weight` 锁定 = `importance`，**永不衰减** | 她理解你的基石（"我最怕孤独"），不应被时间冲淡 |
| 📉 普通记忆 | 0.3-0.8 | `retrieval_weight` 线性衰减（90 天半衰期） | 日常语境信息，逐渐淡出是自然的 |
| 🗑️ 微记忆 | < 0.3 | 30 天后软删除（标记 `archived`，不参与检索） | 闲聊、天气——不值得占语义空间 |

**衰减不是真删除**：archived 的记忆数据保留 90 天，用户可通过 `/memory recover <id>` 恢复（`retrieval_weight` 重置为 `importance`）。到期后硬删除。

**验证方式**：不是验证"衰减有没有跑"，是验证区分度——90 天前的"今天天气不错"检索不到，90 天前的"我小时候被霸凌过"（importance 高）仍能召回。

### 不做

- 不做 Web UI 仪表盘（自省面板/演化时间线）——数字面板与日记的"人性"感冲突，且对用户优先级低；`/mind` 命令保留给开发者
- 不做信念变更日志——信念锚定已有变更逻辑，日志写入是附属品，v0.7 补
- 不迁移 LangGraph agent loop 到 WebSocket——agent 保持在进程内运行，Gateway 在应用层转发消息
- 不引入 FastAPI / React / npm——WebSocket asyncio 零外部依赖，Web UI 用纯 HTML+JS

### 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Gateway 崩溃影响所有客户端 | 单点故障 | 崩溃不丢状态——每轮对话后 `save_state(db)` 已持久化至 SQLite。daemon 重启 = `load_state` + 继续运行。PID 文件残留用于崩溃检测（`lingya status` 告知用户上次异常退出时间）。不做自动重启——那是 OS 的事 |
| 部署复杂度退步 | 从 `uv run main.py` 变成两个进程，用户体感变差 | `lingya start` 自动拉起 daemon + attach CLI，用户感知不到底层变化 |
| 记忆衰减误伤关键记忆 | 不该忘的内容消失了 | importance > 0.8 永不衰减；软删除保留 90 天可恢复 |
| WebSocket 连接中断 | 对话体验断裂 | CLI 检测断连自动重试；短断连不丢消息 |
| 对话状态管理复杂化 | agent loop 与 CLI 紧耦合，引入 Gateway 后需处理中间层 | chat 路由仅在文本层转发，不重写 agent loop |

### 迁移路径（对齐 ADR-001）

```
Phase A: Gateway Daemon + lingya start
  产物：daemon + WebSocket server + mind/diary/memory 只读路由 + 一键启动
  用户体验：lingya start（自动拉起 daemon + attach CLI），与当前 uv run main.py 无差异

Phase B: chat 路由 + 记忆衰减
  产物：chat 路由 + BackgroundRunner（PAD idle drift + 日记调度 + memory decay）
  引擎拥有不依附于会话的生命节律，记忆开始自然淡出
```

## v1.0.0 — 她是她自己

> 目标：LingYa 是一个完整的人——独立运行、有自己的内在生活、质量闭环、社区就绪。

### 关键交付

| # | 交付物 | 验收标准 |
|---|--------|----------|
| 1 | 长期自主运行 | 她不只是会话驱动的——有自己的内部节奏、独立行动、不依附于用户的时间线 |
| 2 | 记忆衰减与层级 | 短期/中期/长期记忆分层，90 天未触及的日常记忆自动降级，关键记忆永不衰减 |
| 3 | 语音输入/输出 | 支持语音对话，人格音色可配置 |
| 4 | 质量闭环 | 记忆召回准确率 > 85%，人格一致性 > 90%，端到端测试覆盖核心路径 |
| 5 | 安装体验 | `pip install lingya && lingya init` 5 分钟内完成首次对话 |
| 6 | 文档与贡献指南 | API 文档、人格配置指南、贡献者指南齐备 |

### 不做

- 移动端 App（非核心，社区可自建）
- 商业化/付费功能（开源优先）

## 时间线与节奏

| 版本 | 状态 |
|------|------|
| v0.2.0 | ✅ 已完成（2026-05-26） |
| v0.3.0 | ✅ 已完成（2026-05-26） |
| v0.4.0 | ✅ 已完成（2026-05-27） |
| v0.5.0 | ✅ 已完成（2026-05-30） |
| v0.6.0 | 🔜 当前迭代 |
| v1.0.0 | 📋 远期 |

> **v0.5 验证记录（2026-06-01）**：引擎层区分度达标（tone Δ=7.9），但 DeepSeek native style 使文本层 pairwise 准确率仅 40-60%。接受当前天花板，不继续调参。详细记录见 `product/v0.5-verification-plan.md`。

> **节奏原则**：功能驱动发布，Ready when ready。质量标准是"0 数据丢失"，不是"0 bug"。

## 决策点

| 节点 | 时机 | Go 条件 | No-Go 处理 |
|------|------|---------|------------|
| OCEAN 区分度验证 | v0.5 交付后 | ✅ 引擎层达标（tone Δ=7.9），文本层接受天花板 | 不继续调参，转向架构演进 |
| Gateway 连通性 | Phase A 交付后 | `curl` 通过 WebSocket 查询心智状态成功 | 检查 asyncio server 实现 |
| v1.0 就绪 | v0.6 完成后 | 多入口架构稳定 + 记忆衰减正常 + 其他准备就绪 | 继续迭代 |

## 优先级排序理由

```
Phase A: Gateway Daemon + lingya start (基础设施 + 用户体验零退步)
    →
Phase B: chat 路由 + BackgroundRunner + 记忆衰减 (独立生命)
```

1. **`lingya start` 必须在 Phase A 就位**：没有它，Phase A 对用户是体验退步。一键启动让底层架构演进对用户完全透明。
2. **Phase B 落地"独立存在"**：chat 路由打通多入口，BackgroundRunner 让她不依附于会话时间线，记忆衰减让记忆系统自然代谢。

## 停车场

好想法，但不在当前范围：

- 人格社区分享/存储——玩家可以分享或存储自己培养出的人格（远期可探索）
- 多人格切换
- MCP 工具生态深度集成
- 记忆加密与零知识证明
- 群组对话（多人格协作）
