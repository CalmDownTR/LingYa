# Architecture Spec (2026-05-19)

## File Tree

```
main.py                    # Entry point: loads .env, runs agent + CLI
config.yaml                # Runtime config (safe to commit)

lingya/
├── agent.py               # LingYaAgent — core orchestrator
├── cli.py                 # LingYaCLI — Rich interactive terminal loop
├── config.py              # Pydantic models: Config, LLMConfig, MemoryConfig, PersonalityConfig
├── embedder.py            # SentenceTransformer wrapper, LRU-cached, async
├── memory/
│   ├── short_term.py      # deque[Message], sliding window, compression trigger
│   ├── long_term.py       # ChromaDB PersistenClient, BGE embeddings, cosine search
│   └── manager.py         # MemoryManager — bridges short/long term, compression, context builder
├── personality/
│   ├── model.py           # PersonalityGenome (persistent) + ActivePersonality (runtime mask) + Adapter
│   ├── engine.py          # PersonalityEngine — load/save genome, get_system_prompt() via mask
│   └── templates.py       # LLM prompt templates (PERSONALITY_CONTEXT_TEMPLATE unused)
├── ingestion/
│   ├── chunker.py         # Recursive token-based text splitter (tiktoken cl100k_base)
│   └── loader.py          # ingest_text/ingest_file/ingest_url (NOT called by agent or CLI)
└── storage/
    ├── db.py              # SQLite via aiosqlite, 5 tables, CRUD methods
    └── migrations.py      # 6 ordered SQL migration statements
```

## Dependency Graph

```
main.py → config, agent, cli

agent.py → config, memory/manager, memory/short_term, personality/engine, storage/db, ingestion/loader
cli.py → agent


memory/manager.py → config, ingestion/chunker, memory/short_term, memory/long_term
memory/long_term.py → embedder

personality/engine.py → config, personality/model, personality/templates

ingestion/loader.py → ingestion/chunker, memory/manager (TYPE_CHECKING only)

storage/db.py → storage/migrations
```

No circular dependencies.

## Personality Architecture

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

### Style preferences

- `verbosity_preference`: concise / balanced / verbose
- `reflex_mode`: instant (conversational flow) / deliberate (step-by-step reasoning)
- `preferred_formats`: e.g. paragraphs, bullet-points

### Prompt rendering

Traits are rendered as **behavioral authorization language** (not adjectives):
- High directness: "You have permission to skip pleasantries..."
- Low adaptability: "When challenged, stand your ground..."

This counters LLM alignment's tendency to drift toward bland, polite responses.

### Situational perturbation

```
user_input → detect_situation() → Situation enum (CRISIS/DEBATE/CASUAL/TECHNICAL/DEFAULT)
                                       │
                                       ▼
Genome → Adapter.activate(genome, situation) → ActivePersonality (traits perturbed ±0.3)
                                       │
                                       ▼
                              to_system_prompt()
```

- Keyword-based detection (no LLM call), fast and cheap
- Perturbations are clamped to [0, 1], never persisted to DB
- `SITUATION_MODIFIERS` dict maps each situation to trait deltas

```
PersonalityGenome (DB, persistent)          # source of truth
  └─ PersonalityAdapter.activate(situation) # pure function + situational perturbation
       └─ ActivePersonality (memory-only)   # transient, destroyed after request
            └─ to_system_prompt()           # renders behavioral authorization language
```

- **Genome**: stores baseline traits, behavior switches, style prefs. Loaded from DB on startup, saved on change.
- **Active mask**: flattened view of the genome with situational perturbations applied. Created fresh per request. No state, no side effects, concurrency-safe.
- **Adapter**: stateless bridge — copies genome values, applies situational deltas, renders behavioral authorization language.

## Agent Request Lifecycle

```
1. CLI passes user input → agent.handle_input()
2. Message stored in ShortTermMemory (deque)
3. LongTermMemory searched via ChromaDB vector similarity
4. Situation detected from user input → adapter applies trait perturbations
5. System prompt built: genome → adapter(situation) → active mask + retrieved memories + compressed summary
6. LLM called via langchain-openai ChatOpenAI (DeepSeek API)
7. Response stored in short-term memory
8. Compression triggered if msg count > threshold
9. Personality maybe_evolve() called (always no-op currently)
10. Turn logged to SQLite
```

## Known Gaps

| Gap | Detail |
|-----|--------|
| Personality evolution | `maybe_evolve()` is a stub — always returns False |
| Situational detection | Keyword-based; LLM-based classifier would be more accurate |
| ingestion/loader.py | Ingest functions defined but not called by agent or CLI |
| PERSONALITY_CONTEXT_TEMPLATE | Defined in templates.py but agent builds context inline instead |
| CLI /fetch | CLI has its own inline URL fetch logic, bypasses loader.ingest_url |
