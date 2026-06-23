# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 核心原则

- **KISS (Keep It Simple, Stupid)**: 优先选择最简单的方案。能用现有模块解决的，不引入新模块；能用标准库的，不加依赖；能用一个函数写完的，不拆成三个。简单即正确。
- **YAGNI (You Aren't Gonna Need It)**: 不为假设的需求写代码。不要预设计未来的扩展点、不要提前抽象、不要给还没出现的配置项留坑。需求到了再改，那时候你更清楚该怎么做。
- **先想后写**: 有歧义或不确定时，先说出来，不要默默选一个方案。方案过度设计时，主动指出更简单的替代方案。不理解的地方停下来问，别猜。
- **精准修改**: 只改和当前任务直接相关的代码。不顺手重构、不改格式、不删无关代码。每个改动的行都应该能追溯到用户的需求。自己的改动产生的 orphan（引入的 import、变量等）要清理。
- **测试驱动**: 先写测试，再写实现。测试即规格——它定义了"什么叫做完"，也是你自检的唯一标准：红灯（写测试）→ 绿灯（写实现）→ 重构。不是"看上去应该可以了"，而是"测试过了"。

## Commands

```bash
uv sync                          # Install all dependencies
uv add <package>                 # Add a runtime dependency
uv add --dev <package>           # Add a dev dependency
uv run python main.py            # Run the app
uv run pytest -s                 # Run tests
uv run ruff check lingya/        # Lint
uv run mypy lingya/              # Type check
```

## Architecture

完整架构详细记录在 [.claude/specs/architecture.md](.claude/specs/architecture.md)，包含文件树、依赖图、已知 gap。**改动涉及架构变化时，必须同步更新 spec 和本节。**

LingYa 基于 **deepagents** (`create_deep_agent`) 构建，支持两种运行模式：

**Direct mode** (`python main.py`): CLI 进程内直连 agent
**Gateway mode** (`python main.py start`): 常驻 daemon + HTTP/SSE 多客户端

```
main.py
  ├── [--daemon] → GatewayDaemon (常驻进程)
  │     ├── MindEngine (单例, 有状态连续演化)
  │     ├── create_deep_agent() (agent 实例)
  │     │     ├── model: ChatOpenAI (DeepSeek API)
  │     │     ├── tools: [memory_store, memory_search]
  │     │     ├── middleware: [SummarizationToolMiddleware]
  │     │     ├── system_prompt: build_static_prompt()
  │     │     ├── backend: StateBackend
  │     │     └── checkpointer: AsyncSqliteSaver
  │     ├── FastAPI Server (HTTP + SSE, uvicorn)
  │     ├── MessageRouter (mind/diary/memory/chat 路由)
  │     └── BackgroundRunner (PAD idle drift + diary scheduler + memory decay)
  │
  ├── [start] → 自动拉起 daemon + HTTP CLI 客户端
  │
  └── [default] → LingYaCLI (进程内直连 agent + MindEngine)

MindEngine (pure computation, zero framework dependency)
  ├── OCC 22-emotion classification (deterministic)
  ├── PAD evolution (pleasure-arousal-dominance)
  ├── IPC state machine (agency/communion)
  ├── Dynamic tone (continuous PAD→tone + OCEAN modulation)
  ├── OCEAN drift (every 10 turns, max 0.005/step)
  ├── Reflection tree (importance-threshold triggered)
  └── idle_tick (PAD spring-restore toward baseline, spring_k=0.01)
```

### Request lifecycle

**Direct mode** (unchanged):
1. CLI calls `agent.ainvoke({"messages": [SystemMessage(fragment), HumanMessage(msg)]}, config)` with thread_id
2. LangGraph checkpoint loads conversation state
3. deepagents middleware pipeline
4. LLM called with all tools available
5. Tool calls executed, response extracted
6. MindEngine.process_event() runs post-response: OCC+IPC → PAD → tone → importance → reflection → drift → save
7. State checkpointed by LangGraph

**Gateway mode** (HTTP + SSE, FastAPI):
1. Client sends `POST /chat` with `{"text": "..."}` via HTTP
2. FastAPI endpoint → router._handle_chat_streaming() async generator
3. Agent astream_events → SSE event frames pushed to client
4. MindEngine.process_event() runs post-response: OCC+IPC → PAD → tone → importance → reflection → drift → save
4. Response `{"type": "chat_response", "payload": {"text": "...", "tone": {...}}}` sent back
5. BackgroundRunner maintains independent life rhythm: PAD idle drift, diary scheduling, memory decay

### Modules at a glance

| Module | Role | Key detail |
|--------|------|------------|
| `main.py` | Assembly | Wires model + tools + middleware + MindEngine; 3 entry points: --daemon / start / default |
| `lingya/cli.py` | Terminal UI | Rich-based, dual mode: direct (`run()`) + HTTP/SSE client |
| `lingya/config.py` | Config | Pydantic + YAML + env overlay |
| `lingya/gateway/` | Multi-entry | Daemon, FastAPI SSE server, message router, HTTP client, auth, BackgroundRunner |
| `lingya/mind/` | Personality | Dynamic engine: OCC emotion → PAD → tone → OCEAN drift → reflection → idle_tick |
| `lingya/memory/` | Memory | ChromaDB-backed, importance-weighted, three-level decay (retrieval_weight), recover |
| `lingya/storage/` | Persistence | SQLite via aiosqlite, tables: conversations, turns, mind_state |
| `lingya/diary.py` | Diary | Markdown diary generation in LingYa's voice, one per day |
| `lingya/reflection.py` | Opening | Generates context-aware opening line for returning users |

### Configuration
- `config.yaml` — runtime settings (safe to commit): LLM, db_path, memory_path, data_dir
- `agent_config.yaml` — mind config: identity, OCEAN traits, tone_matrix, behavior_guardrails
- `agent_config.example.yaml` — template for new setups
- `.env` — secrets: `DEEPSEEK_API_KEY`, `LINGYA_API_KEY`

## 协作流程

**用户掌方向，Claude 执引擎。**

| | 用户 | Claude |
|---|---|---|
| 角色 | 决定做什么、方案行不行 | 想清楚怎么做、主动暴露歧义 |
| 简单任务 | 一句话指令 | 直接改，改完一句话告知 |
| 非平凡任务 | 确认或调整方案方向 | 先说明改哪些文件、为什么这样改、有什么取舍需要拍板，得到确认后再动手 |
| 架构变化 | — | 改完后同步更新 CLAUDE.md + `.claude/specs/architecture.md` |

核心：**动手前你说了算，动手后让你知道。**
