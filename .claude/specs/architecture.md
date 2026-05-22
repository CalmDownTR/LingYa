# Architecture Spec (2026-05-21)

## File Tree

```
main.py                    # Entry point: loads .env, wires agent + CLI
config.yaml                # Runtime config (safe to commit)

lingya/
├── cli.py                 # LingYaCLI — Rich interactive terminal loop
├── config.py              # Pydantic models: Config, LLMConfig, PersonalityConfig
├── embedder.py            # SentenceTransformer wrapper, LRU-cached, async
├── middleware.py           # PersonalityMiddleware — injects behavioral auth language
├── memory/
│   ├── long_term.py       # ChromaDB PersistentClient, BGE embeddings, cosine search
│   └── tools.py           # search_memory / save_memory langchain tools
├── personality/
│   ├── model.py           # PersonalityGenome (persistent) + ActivePersonality (runtime) + Adapter
│   ├── engine.py          # PersonalityEngine — load/save genome, get_system_prompt()
│   └── templates.py       # LLM prompt templates (REFLECTION_SYSTEM_PROMPT, PERSONALITY_CONTEXT_TEMPLATE)
├── ingestion/
│   └── chunker.py         # Recursive token-based text splitter (tiktoken cl100k_base)
└── storage/
    ├── db.py              # SQLite via aiosqlite, 3 tables, CRUD methods
    └── migrations.py      # 6 ordered SQL migration statements
```

## Dependency Graph

```
main.py → config, cli, middleware, memory/long_term, memory/tools, personality/engine, storage/db

cli.py → personality/engine, storage/db
middleware.py → personality/engine

memory/tools.py → memory/long_term, ingestion/chunker
memory/long_term.py → embedder

personality/engine.py → config, personality/model, personality/templates, storage/db

storage/db.py → storage/migrations
```

No circular dependencies.

## Agent Architecture

LingYa uses **deepagents** (`create_deep_agent`) as its agent harness, built on LangGraph. The agent is assembled in `main.py`:

```
create_deep_agent()
  ├── model: ChatOpenAI (DeepSeek API)
  ├── tools: memory tools (ChromaDB) + MCP tools (optional)
  ├── middleware: [PersonalityMiddleware]   ← LingYa's unique differentiator
  ├── system_prompt: base agent prompt
  └── backend: StateBackend (in-memory virtual filesystem)
```

### Request Lifecycle (deepagents)

```
1. CLI calls agent.ainvoke({"messages": [user_msg]}, config)
2. LangGraph checkpoint loads conversation state by thread_id
3. Middleware pipeline executes:
   a. TodoListMiddleware — task planning
   b. FilesystemMiddleware — virtual filesystem tools (ls, read, write, edit)
   c. SubAgentMiddleware — sub-agent spawning
   d. SummarizationMiddleware — auto context compression
   e. PersonalityMiddleware — injects behavioral authorization language
4. LLM called with tools available
5. Tool calls executed (memory tools, filesystem, MCP tools)
6. Response extracted, displayed
7. State checkpointed by LangGraph
```

### Personality Middleware

```
user_input → PersonalityMiddleware.awrap_model_call()
                ├── _extract_last_user_text(request.messages)
                ├── PersonalityEngine.get_system_prompt(user_input)
                │     ├── detect_situation(user_input) → Situation enum
                │     └── Adapter.activate(genome, situation) → ActivePersonality.to_system_prompt()
                └── request.override(system_message=injected_prompt)
```

The middleware is the sole custom component in the deepagents pipeline. It intercepts each model call and prepends LingYa's behavioral authorization language to the system prompt.

## Personality Architecture

Unchanged from the original design. See git history for full details.

### Trait model (Big Five-aligned)

| Trait | Role | Big Five mapping |
|-------|------|-----------------|
| `exploration` (0–1) | Novelty-seeking vs risk-aversion | Openness |
| `analytical_depth` (0–1) | Cognitive need / depth of thought | Openness |
| `playfulness` (0–1) | Humor and levity | Extraversion |
| `empathy` (0–1) | Emotional attunement vs objectivity | Agreeableness |
| `directness` (0–1) | Frankness vs diplomacy | (behavioral auth) |
| `adaptability` (0–1) | Stress tolerance / reaction to criticism | Neuroticism (inverse) |

### Behavior switches (metacognitive)

- `asks_clarifying_questions` — pause on ambiguity vs guess
- `admits_uncertainty` — state ignorance vs feign confidence
- `offers_unsolicited_insights` — proactive observations vs just answer
- `matches_user_tone` — mirror user's length/tone/punctuation vs fixed voice

### Situational perturbation

```
Genome → Adapter.activate(genome, situation) → ActivePersonality (traits perturbed ±0.3)
                                           │
                                           ▼
                                  to_system_prompt()
```

- Keyword-based detection (no LLM call)
- Perturbations clamped to [0, 1], never persisted

## Long-Term Memory

ChromaDB with BGE embeddings, exposed as two langchain tools:
- `search_memory(query)` — semantic search across stored memories
- `save_memory(content, source)` — chunk and store new memories

Independent from deepagents' LangGraph Memory Store. They serve different purposes:
- ChromaDB: semantic search over ingested content and saved facts
- LangGraph Store: cross-session agent state (future use)

## Known Gaps

| Gap | Detail |
|-----|--------|
| Personality evolution | `maybe_evolve()` is a stub — always returns False |
| MCP tools | Config-driven MCP server connection not yet wired (TODO in main.py) |
| Persistent checkpointing | ✅ Resolved — SqliteSaver checkpointer added (a62efaf) |
| PERSONALITY_CONTEXT_TEMPLATE | Defined in templates.py but unused |
