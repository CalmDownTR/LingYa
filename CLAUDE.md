# CLAUDE.md

This file provides guidance to Claude Code in this repository.
完整架构详见 @product/reference/architecture.md。Git 纪律见 @.claude/git-disciplines.md。

## 核心原则
- **KISS & YAGNI**: 优先最简单的方案。不为假设需求写代码，不提前抽象。简单即正确。
- **先想后写**: 有歧义或过度设计时停下来主动指出，不理解的地方直接问，别猜。
- **精准修改**: 只改和当前任务直接相关的代码。及时清理自己改动产生的孤立变量/Import。
- **测试驱动 (TDD)**: 先写测试，再写实现。红灯（写测试）→ 绿灯（写实现）→ 重构。

## Commands
```bash
uv sync                          # Install dependencies
uv add <package>                 # Add a runtime dependency
uv run python main.py            # Run direct mode
uv run python main.py start      # Run gateway daemon
uv run pytest -s                 # Run test suite
uv run ruff check lingya/        # Lint code
uv run mypy lingya/              # Type check
```

## Architecture

LingYa 基于 **deepagents** 构建，两种运行模式：
- **Direct mode** (`python main.py`): CLI 进程内直连 agent
- **Gateway mode** (`python main.py start`): 常驻 daemon + HTTP/SSE 多客户端

### Modules at a glance

| Module | Role |
|--------|------|
| `main.py` | Assembly: model + tools + middleware + MindEngine |
| `lingya/cli.py` | Terminal UI (Rich), dual mode: direct + HTTP client |
| `lingya/config.py` | Config (Pydantic + YAML + env overlay) |
| `lingya/gateway/` | Multi-entry: daemon, FastAPI SSE server, router, auth, BackgroundRunner, Settings API |
| `web/` | Web UI: Vite 6 + React 19 + TypeScript + Tailwind 4 |
| `lingya/mind/` | Personality: OCC→PAD→tone→OCEAN drift→reflection→idle_tick |
| `lingya/memory/` | ChromaDB-backed, importance-weighted, three-level decay |
| `lingya/storage/` | SQLite via aiosqlite (mind_state + LangGraph checkpoints) |
| `lingya/diary.py` | Markdown diary in LingYa's voice |
| `lingya/reflection.py` | Context-aware opening line |

完整拓扑、数据流、接口契约见 @product/reference/architecture.md。架构变化时同步更新该文档。

### Configuration
- `config.yaml` — runtime settings (LLM, db_path, memory_path, otel)
- `agent_config.yaml` — mind config (identity, OCEAN, tone, guardrails)
- `.env` — secrets: `DEEPSEEK_API_KEY`, `LINGYA_API_KEY`

## 🤝 Domain Boundaries (CRITICAL)

三方协作，各管一层。`product/context/` 全员只读。

| Role | Owns (writes) | Reads |
|------|---------------|-------|
| **PM** (via @product-agent) | — | context/, plan/, specs/, capabilities |
| **Architect** (via @architect-agent) | — | roadmap, ADRs, architecture.md, lingya/ code |
| **Coding** (主 agent = you) | `lingya/` `tests/` `product/plan/` `product/specs/` `product/decisions/` `product/reference/architecture.md` | roadmap (current version), PRD, ADRs |

**关键约束**：subagent（PM/Architect）只做只读分析，返回建议。所有 `product/` 文档写入由主 agent（你）在用户确认后执行。这保证用户确认前 roadmap/PRD/ADR 不被污染。

## 🔄 Multi-Agent Pipelines

你（主 agent）是**编排器 + 对话主体 + 编码执行者 + 唯一的 `product/` 文档写入者**。

subagent 是只读分析工具，**无持续上下文**——它们看不到对话历史，每次 spawn 都是全新开始。所以讨论在你这里发生，subagent 只做单次分析后返回结论。

### Pipeline A: 新功能与产品演进

当用户提出产品想法或功能需求时：

1. **对齐理解**：先和用户快速确认范围（"这个需求我理解是 XX，建议放 v0.x？"）。不急着 spawn。
2. **产品分析**：spawn `@product-agent`，传递原始想法 + 对齐结果。PM 只读分析，返回方案建议（含 user story + 验收标准 + 不做清单 + 优先级理由）。
3. **方案讨论**：把 PM 建议总结给用户讨论。可能多轮——用户反馈后可再 spawn PM 带着反馈重新分析。
4. **写入规划**（用户确认后）：你将定稿方案写入 `product/plan/roadmap.md` 对应版本 + 创建 `product/specs/PRD-NNN-*.md`。
5. **架构评审**：spawn `@architect-agent`，传递定稿方案。Architect 只读评审，返回 ADR 草稿 + 破坏性变更提示。
6. **ADR 确认**：把 ADR 草稿总结给用户确认。确认后你写入 `product/decisions/ADR-NNN-*.md`。
7. **编码执行**：用户确认后，按 roadmap 当前版本的实现步骤 TDD 编码。每个 Step 完成立即 commit。

### Pipeline B: 纯架构改进

当用户有单纯的技术/架构改进想法（不涉及产品功能）：

1. **对齐理解**：和用户确认改进目标。
2. **架构分析**：spawn `@architect-agent`，传递改进想法。Architect 只读分析，返回 ADR 草稿。
3. **讨论确认**：把 ADR 草稿总结给用户讨论。确认后你写入 `product/decisions/ADR-NNN-*.md`。
4. **编码执行**：按 ADR 在 `lingya/` 中执行代码修改。

### Pipeline C: 版本交付闭环

编码完成后（所有 Step 完成 + 测试通过）：

1. **产品验收**：spawn `@product-agent`，传递 roadmap 当前版本的验收标准。PM 对照检查，返回验收结果（PASS/FAIL + 证据）。
2. **更新文档**（验收通过后）：你更新 `product/specs/capabilities.md`（新功能状态）+ `product/plan/roadmap.md`（版本状态改为已完成）。
3. **架构同步**：spawn `@architect-agent`，传递"版本交付，请更新架构文档"。Architect 读 `lingya/` 真实代码，返回 architecture.md 更新建议。你确认后写入 `product/reference/architecture.md`。

## 协作流程总则

**用户掌方向，你执引擎。**

| | 用户 | 主 agent (you) |
|---|---|---|
| 角色 | 决定做什么、方案行不行 | 想清楚怎么做、主动暴露歧义 |
| 简单任务 | 一句话指令 | 直接改，改完一句话告知 |
| 非平凡任务 | 确认或调整方案方向 | 先说明改哪些文件、为什么、有什么取舍，得到确认后再动手 |
| 架构变化 | — | 改完后同步更新 architecture.md |

核心：**动手前你说了算，动手后让你知道。**
