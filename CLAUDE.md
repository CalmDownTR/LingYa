# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 核心原则

- **KISS (Keep It Simple, Stupid)**: 优先选择最简单的方案。能用现有模块解决的，不引入新模块；能用标准库的，不加依赖；能用一个函数写完的，不拆成三个。简单即正确。
- **YAGNI (You Aren't Gonna Need It)**: 不为假设的需求写代码。不要预设计未来的扩展点、不要提前抽象、不要给还没出现的配置项留坑。需求到了再改，那时候你更清楚该怎么做。
- **先想后写**: 有歧义或不确定时，先说出来，不要默默选一个方案。方案过度设计时，主动指出更简单的替代方案。不理解的地方停下来问，别猜。
- **精准修改**: 只改和当前任务直接相关的代码。不顺手重构、不改格式、不删无关代码。每个改动的行都应该能追溯到用户的需求。自己的改动产生的 orphan（引入的 import、变量等）要清理。
- **目标驱动**: 每个任务先明确可验证的成功标准，再开始写代码。完成时对照标准自检——不是"看上去应该可以了"，而是"验证过了"。复杂任务先列出步骤，逐步闭环。

## Commands

```bash
uv sync                    # Install all dependencies
uv run python main.py      # Run the app
uv run pytest -s           # Run tests
uv run ruff check lingya/  # Lint
uv run mypy lingya/        # Type check
```

## Architecture

完整架构详细记录在 [.claude/specs/architecture.md](.claude/specs/architecture.md)，包含文件树、依赖图、已知 gap。**改动涉及架构变化时，必须同步更新 spec 和本节。**

```
main.py → CLI → Agent.handle_input() → LangGraph create_react_agent
                    │   ├── ChatOpenAI (langchain-openai)
                    │   ├── MemoryManager
                    │   │     ├── ShortTermMemory (deque, token-based compression)
                    │   │     └── LongTermMemory (ChromaDB + BGE embeddings)
                    │   ├── PersonalityEngine (Genome → Adapter → Active mask → system prompt)
                    │   ├── Agent Tools (create_agent_tools closure factory)
                    │   │     ├── search_memory, save_memory, fetch_url
                    │   └── Database (SQLite via aiosqlite)
```

### Request lifecycle
1. Store user message in short-term memory + SQLite
2. Pre-flight token budget check — compress if needed (token-based, summary in deque, NOT in ChromaDB)
3. Build system prompt: personality (behavioral auth language) + compression summaries + tool guidance
4. LangGraph ReAct agent loop: LLM with tools (search_memory, save_memory, fetch_url), up to N iterations
5. Extract final AI response, store in deque + SQLite
6. Personality maybe_evolve() (stub)

### Modules at a glance

| Module | Role | Key detail |
|--------|------|------------|
| `lingya/agent.py` | Orchestrator | LangGraph create_react_agent, stateless snapshot mode |
| `lingya/cli.py` | Terminal UI | Rich-based, `/fetch` `/personality` `/reflect` etc. |
| `lingya/config.py` | Config | Pydantic + YAML + env overlay |
| `lingya/tools.py` | Agent tools | Closure factory: search_memory, save_memory, fetch_url |
| `lingya/memory/` | Memory | Short-term (deque, hard-cap safety guard), Long-term (ChromaDB), Manager (token-based compression, save_to_long_term) |
| `lingya/personality/` | Personality | Genome (persistent, 6 traits + 4 switches) + Active mask (runtime, behavioral auth language) + situation detection + perturbation; evolution is a stub |
| `lingya/ingestion/` | Content ingestion | Chunker, embedder (BGE), loader |
| `lingya/storage/` | Persistence | SQLite via aiosqlite, 5 tables |

### Configuration
- `config.yaml` — runtime settings (safe to commit)
- `.env` — secrets: `DEEPSEEK_API_KEY`, `LINGYA_API_KEY`, `HF_ENDPOINT`

## 协作流程

**用户掌方向，Claude 执引擎。**

| | 用户 | Claude |
|---|---|---|
| 角色 | 决定做什么、方案行不行 | 想清楚怎么做、主动暴露歧义 |
| 简单任务 | 一句话指令 | 直接改，改完一句话告知 |
| 非平凡任务 | 确认或调整方案方向 | 先说明改哪些文件、为什么这样改、有什么取舍需要拍板，得到确认后再动手 |
| 架构变化 | — | 改完后同步更新 CLAUDE.md + `.claude/specs/architecture.md` |

核心：**动手前你说了算，动手后让你知道。**
