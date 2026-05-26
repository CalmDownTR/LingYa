# Architecture Spec (2026-05-26)

## File Tree

```
main.py                    # Entry point: loads .env, wires agent + CLI
config.yaml                # Runtime config (safe to commit)

agent_config.yaml          # Persona config: mind_core, tone_matrix, behavior_guardrails

lingya/
├── cli.py                 # LingYaCLI — Rich interactive terminal loop
├── config.py              # Pydantic models: Config, LLMConfig
├── persona/
│   ├── config.py          # Pydantic models: PersonaConfig, MindCore, ToneMatrix + YAML loader
│   ├── bucketing.py       # Interval bucketing: map_warmth(), map_formality() (if-elif)
│   └── assembler.py       # PromptAssembler: config + bucketing → system prompt
└── storage/
    ├── db.py              # SQLite via aiosqlite, 2 tables, CRUD methods
    └── migrations.py      # 2 ordered SQL migration statements

tests/
├── cases.json             # Persona eval Corner Case definitions (mock history for long-context test)
├── conftest.py            # pytest fixtures (DB + persona)
├── test_persona_bucketing.py  # Bucketing boundary value tests
├── test_persona_assembler.py  # Prompt structure verification
├── eval_runner.py         # E2E runner: LLM calls + pass/fail checks
├── test_config.py         # Config loading tests
└── test_session.py        # Session management tests
```

## Dependency Graph

```
main.py → config, cli, persona/assembler, storage/db

cli.py → storage/db

persona/assembler.py → persona/config.py, persona/bucketing.py

storage/db.py → storage/migrations
```

No circular dependencies.

## Agent Architecture

LingYa uses **deepagents** (`create_deep_agent`) as its agent harness, built on LangGraph. The agent is assembled in `main.py`:

```
create_deep_agent()
  ├── model: ChatOpenAI (DeepSeek API)
  ├── tools: MCP tools (optional)
  ├── middleware: [SummarizationToolMiddleware]
  ├── system_prompt: PromptAssembler.assemble() — persona-driven dynamic prompt
  └── backend: StateBackend (shared with summarization middleware)
```

### Persona System (Phase 1)

`agent_config.yaml` defines three orthogonal concerns:
- **mind_core**: immutable identity — who the AI is
- **tone_matrix**: warmth (0-100) and formality (0-100) — how the AI speaks
- **behavior_guardrails**: blacklist rules with highest execution priority

`PromptAssembler` assembles the final system prompt in fixed order:
1. `# ROLE IDENTITY` — mind_core.identity + core_belief
2. `# INTERACTION STYLE` — bucketed warmth + formality instructions
3. `# STRICT NEGATIVE BOUNDARIES` — guardrails at bottom (recency bias)

### Request Lifecycle (deepagents)

```
1. CLI calls agent.ainvoke({"messages": [user_msg]}, config)
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
7. State checkpointed by LangGraph
```

## Known Gaps

| Gap | Detail |
|-----|--------|
| MCP tools | Config-driven MCP server connection not yet wired (TODO in main.py) |
| Persistent checkpointing | ✅ Resolved — SqliteSaver checkpointer added (a62efaf) |
