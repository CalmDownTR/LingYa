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

LingYa 基于 **deepagents** (`create_deep_agent`) 构建。

```
main.py → create_deep_agent()
            ├── model: ChatOpenAI (DeepSeek API)
            ├── tools: [memory_store, memory_search] + MCP tools (optional)
            ├── middleware: [SummarizationToolMiddleware]
            ├── system_prompt: base agent prompt
            └── backend: StateBackend (shared instance)
```

### Request lifecycle
1. CLI calls `agent.ainvoke({"messages": [user_msg]}, config)` with thread_id
2. LangGraph checkpoint loads conversation state
3. deepagents middleware pipeline:
   - Built-in: TodoList, Filesystem, SubAgent, Summarization
   - **SummarizationToolMiddleware**: optional `compact_conversation` tool
4. LLM called with all tools available (MCP)
5. Tool calls executed, response extracted, state checkpointed

### Modules at a glance

| Module | Role | Key detail |
|--------|------|------------|
| `main.py` | Assembly | Wires model + tools + middleware + CLI |
| `lingya/cli.py` | Terminal UI | Rich-based, `/sessions` `/new` `/switch` |
| `lingya/config.py` | Config | Pydantic + YAML + env overlay, slimmed down |
| `lingya/persona/` | Persona | Dynamic prompt assembly via mind_core + tone_matrix |
| `lingya/memory/` | Memory | ChromaDB-backed semantic memory, `store` + `search` + `list_all` + `delete` |
| `lingya/storage/` | Persistence | SQLite via aiosqlite, 2 tables (conversations, schema_version) |

### Configuration
- `config.yaml` — runtime settings (safe to commit)
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
