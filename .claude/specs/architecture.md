# Architecture Spec (2026-05-19)

## File Tree

```
main.py                    # Entry point: loads .env, runs agent + CLI
config.yaml                # Runtime config (safe to commit)

lingya/
├── agent.py               # LingYaAgent — core orchestrator
├── cli.py                 # LingYaCLI — Rich interactive terminal loop
├── config.py              # Pydantic models: Config, LLMConfig, MemoryConfig, PersonalityConfig
├── llm/
│   ├── base.py            # BaseLLMBackend (ABC), ToolDefinition, LLMResponse
│   ├── factory.py         # create_backend() — routes provider name to OpenAICompatBackend
│   └── openai_compat.py   # AsyncOpenAI wrapper, lazy init, tool call support
├── memory/
│   ├── short_term.py      # deque[Message], sliding window, compression trigger
│   ├── long_term.py       # ChromaDB PersistenClient, BGE embeddings, cosine search
│   └── manager.py         # MemoryManager — bridges short/long term, compression, context builder
├── personality/
│   ├── model.py           # Personality Pydantic model (5 traits + style)
│   ├── engine.py          # PersonalityEngine — load/save, maybe_evolve() is a stub
│   └── templates.py       # LLM prompt templates (PERSONALITY_CONTEXT_TEMPLATE unused)
├── ingestion/
│   ├── chunker.py         # Recursive token-based text splitter (tiktoken cl100k_base)
│   ├── embedder.py        # SentenceTransformer wrapper, LRU-cached, async
│   ├── loader.py          # ingest_text/ingest_file/ingest_url (NOT called by agent or CLI)
│   └── tools.py           # LLM tool definitions for fetch/ingest (NOT wired to agent)
└── storage/
    ├── db.py              # SQLite via aiosqlite, 5 tables, CRUD methods
    └── migrations.py      # 6 ordered SQL migration statements
```

## Dependency Graph

```
main.py → config, agent, cli

agent.py → config, llm/factory, memory/manager, memory/short_term, personality/engine, storage/db
cli.py → agent

llm/factory.py → config, llm/base, llm/openai_compat
llm/openai_compat.py → config, llm/base

memory/manager.py → config, ingestion/chunker, memory/short_term, memory/long_term
memory/long_term.py → ingestion/embedder

personality/engine.py → config, personality/model, personality/templates

ingestion/loader.py → ingestion/chunker, memory/manager (TYPE_CHECKING only)
ingestion/tools.py → llm/base, memory/manager (TYPE_CHECKING only)

storage/db.py → storage/migrations
```

No circular dependencies.

## Agent Request Lifecycle

```
1. CLI passes user input → agent.handle_input()
2. Message stored in ShortTermMemory (deque)
3. LongTermMemory searched via ChromaDB vector similarity
4. System prompt built: personality + retrieved memories + compressed summary
5. LLM called via OpenAICompatBackend
6. Response stored in short-term memory
7. Compression triggered if msg count > threshold
8. Personality maybe_evolve() called (always no-op currently)
9. Turn logged to SQLite
```

## Known Gaps

| Gap | Detail |
|-----|--------|
| Personality evolution | `maybe_evolve()` is a stub — always returns False |
| ingestion/loader.py | Ingest functions defined but not called by agent or CLI |
| ingestion/tools.py | LLM tool defs defined but not wired to agent |
| PERSONALITY_CONTEXT_TEMPLATE | Defined in templates.py but agent builds context inline instead |
| CLI /fetch | CLI has its own inline URL fetch logic, bypasses loader.ingest_url |
