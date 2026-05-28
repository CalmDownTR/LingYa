# Architecture Spec (2026-05-28)

## File Tree

```
main.py                    # Entry point: loads .env, wires model + agent + mind engine + CLI
config.yaml                # Runtime config (safe to commit)

agent_config.yaml          # Mind config: identity, OCEAN, tone_matrix, behavior_guardrails
agent_config.example.yaml  # Template for new setups

lingya/
├── __init__.py
├── cli.py                 # LingYaCLI — Rich interactive terminal loop + /commands
├── config.py              # Pydantic models: Config, LLMConfig + YAML loader
├── diary.py               # Diary generation: prompt, format, save/list/read Markdown diaries
├── reflection.py          # Opening-line generation for returning users
├── memory/
│   ├── __init__.py        # Exports MemoryStore, EnhancedMemoryStore
│   ├── store.py           # ChromaDB PersistentClient, importance scoring, weighted search
│   └── reflection.py      # Reflection tree: importance-threshold → self-notion extraction
├── mind/
│   ├── __init__.py        # Exports MindEngine, MindConfig, MindState, PADPoint, etc.
│   ├── config.py          # Pydantic models: MindConfig, BigFiveTraits, IdentityAnchor, ToneMatrix
│   ├── engine.py          # MindEngine: per-turn pipeline (OCC+IPC → PAD → tone → reflection → drift)
│   ├── state.py           # MindState, PADPoint — fully serializable personality state
│   ├── affect.py          # OCC 22-emotion decision tree, PAD evolution, OCEAN→PAD baseline, OCEAN drift
│   ├── dynamics.py        # IPC dual-axis state machine (agency/communion)
│   ├── tone.py            # Stage-aware tone mapping: PAD→tone + OCEAN modulation
│   ├── guard.py           # Identity re-anchoring via cosine similarity monitoring
│   └── belief.py          # Belief anchoring with OCEAN-modulated update probability
└── storage/
    ├── __init__.py
    ├── db.py              # SQLite via aiosqlite, 3+ tables, CRUD + mind_state persistence
    └── migrations.py      # 7 ordered SQL migration statements

tests/
├── __init__.py
├── conftest.py            # pytest fixtures (DB + mind)
├── test_config.py         # Config loading tests
├── test_session.py        # Session management tests
├── test_diary.py          # Diary generation and I/O tests
├── test_memory.py         # Memory store tests
├── test_reflection.py     # Reflection + opening line tests
├── test_mind_config.py    # Mind config loading tests
├── test_mind_affect.py    # OCC classification, PAD evolution, OCEAN drift tests
├── test_mind_engine.py    # MindEngine pipeline integration tests
├── test_mind_tone.py      # Tone mapping and stage detection tests
└── eval_runner.py         # E2E runner: LLM calls + pass/fail checks
```

## Dependency Graph

```
main.py → config, cli, memory, mind/engine, mind/config, storage/db

cli.py → storage/db, reflection, diary, mind/config

memory/store.py → (chromadb)
memory/reflection.py → memory/store.py

mind/engine.py → mind/affect, mind/config, mind/dynamics, mind/guard, mind/state, mind/tone, memory/store, memory/reflection
mind/affect.py → mind/config, mind/state
mind/dynamics.py → (stdlib only)
mind/tone.py → mind/config, mind/state
mind/guard.py → (stdlib only)
mind/belief.py → mind/config

diary.py → mind/config

reflection.py → mind/config

storage/db.py → storage/migrations
```

No circular dependencies.

## Agent Architecture

LingYa uses **deepagents** (`create_deep_agent`) as its agent harness, built on LangGraph. The agent is assembled in `main.py`:

```
create_deep_agent()
  ├── model: ChatOpenAI (DeepSeek API)
  ├── tools: [memory_store, memory_search] + MCP tools (optional)
  ├── middleware: [SummarizationToolMiddleware]
  ├── system_prompt: build_static_prompt() — identity + guardrails (static portion)
  ├── backend: StateBackend (shared with summarization middleware)
  └── checkpointer: AsyncSqliteSaver (persists conversation state across restarts)
```

### Mind Engine (Phase 2+)

The mind module is a **pure computational layer** with zero dependency on the agent framework. The agent consumes its output (dynamic tone fragment per-turn).

`agent_config.yaml` defines:
- **identity**: immutable identity anchor — who the AI is, core belief
- **ocean**: Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism), each 0.0–1.0. Seed for first startup; thereafter DB is sole source of truth.
- **tone_matrix**: base warmth (0–100), formality (0–100), humor (0–1)
- **behavior_guardrails**: blacklist rules with highest execution priority

`MindEngine.process_event()` pipeline per turn:
1. **OCC + IPC** (1 LLM call, 1.5s timeout) — cognitive appraisal + agency/communion estimation
2. **OCC classify** — deterministic 22-emotion decision tree → PAD pull vector
3. **PAD evolve** — apply emotion pull + spring-restore toward OCEAN-derived baseline
4. **IPC transition** — state machine enforces valid agency/communion state changes
5. **Stage detect** → **dynamic tone** — continuous PAD→tone mapping with OCEAN modulation
6. **Importance pre-score** — rule-based (keyword matching), LLM refinement in background
7. **Reflection check** — if cumulative importance ≥ threshold, fire-and-forget reflection tree
8. **OCEAN drift** (every 10 turns) — tiny PAD deviation → OCEAN adjustment (max 0.005/step)
9. **Auto-persist** — mind state saved to SQLite

### System Prompt Assembly

The system prompt has two parts:
1. **Static** (`build_static_prompt`): identity + core belief + guardrails + memory behavior instructions
2. **Dynamic** (`MindEngine.get_prompt_fragment()`): injected as a SystemMessage each turn, carrying current PAD-derived mood, warmth/formality descriptors, and stage-specific hints

### Request Lifecycle (deepagents)

```
1. CLI calls agent.ainvoke({"messages": [SystemMessage(fragment), HumanMessage(msg)]}, config)
2. LangGraph checkpoint loads conversation state by thread_id
3. Middleware pipeline executes:
   a. TodoListMiddleware — task planning
   b. FilesystemMiddleware — virtual filesystem tools (ls, read, write, edit)
   c. SubAgentMiddleware — sub-agent spawning
   d. SummarizationMiddleware — auto context compression at 85% token threshold
   e. SummarizationToolMiddleware — optional `compact_conversation` tool for manual summarization
4. LLM called with tools available
5. Tool calls executed (MCP tools)
6. Response extracted, displayed
7. MindEngine.process_event() runs post-response (fire-and-forget for scoring/reflection)
8. State checkpointed by LangGraph
```

### Persistence

- **SQLite** (aiosqlite): conversations, turns, mind_state tables. MindState serialized as JSON in a singleton row.
- **ChromaDB**: semantic memory vector store with importance-weighted retrieval.
- **Diary**: Markdown files under `data/diary/`, one per day.

## Known Gaps

| Gap | Detail |
|-----|--------|
| MCP tools | Config-driven MCP server connection not yet wired (TODO in main.py) |
| Embedding fn | `MindEngine.check_response_alignment()` needs an embedding function to enable identity guard; currently always returns True |
| Belief update | `belief.py` is implemented but not yet wired into the main pipeline |
