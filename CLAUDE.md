# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                    # Install all dependencies
uv run python main.py      # Run the app
uv run pytest -s           # Run tests
uv run ruff check lingya/  # Lint
uv run mypy lingya/        # Type check
```

## Architecture

```
main.py → CLI → Agent.handle_input() → LLM response
                    ├── MemoryManager
                    │     ├── ShortTermMemory (deque, sliding window)
                    │     └── LongTermMemory (ChromaDB + BGE embeddings)
                    ├── PersonalityEngine (Pydantic model → system prompt)
                    └── Database (SQLite via aiosqlite)
```

### Agent flow (`lingya/agent.py`)
1. Add user message to short-term memory
2. Retrieve relevant long-term memories via vector search
3. Build system prompt from personality + memories + compressed context
4. Call LLM, store response, trigger compression/evolution as needed

### Memory system (`lingya/memory/`)
- **Short-term**: `deque[Message]` with max capacity. When messages exceed `compression_trigger_messages`, the oldest messages are popped and sent to the LLM for summarization. The summary is stored in long-term memory and injected into the system prompt via `build_context_for_llm()`.
- **Long-term**: ChromaDB with cosine similarity. Uses `BAAI/bge-small-zh-v1.5` (512-dim) for embeddings. Content ingested via `/fetch` is chunked, embedded, and stored here.

### LLM backend (`lingya/llm/`)
Single `OpenAICompatBackend` serves DeepSeek, OpenAI, and Ollama through the same interface. Config controls provider/model/base_url. API key is read from `DEEPSEEK_API_KEY` or `LINGYA_API_KEY` env vars.

### Personality (`lingya/personality/`)
`Personality` is a Pydantic model with 5 scalar traits (curiosity, analytical_depth, playfulness, empathy, directness) plus communication style fields. `to_system_prompt()` serializes it into the system prompt. Evolution is a stub — `maybe_evolve()` increments a counter but never triggers actual changes.

### Embedding model
The model `BAAI/bge-small-zh-v1.5` is cloned from ModelScope (HuggingFace is blocked from China). It lives at `data/models/bge-small-zh-v1.5/` (gitignored). If you need a different model, clone it from ModelScope with `git lfs` and update `config.yaml`.

### Configuration
- `config.yaml` — all settings (safe to commit)
- `.env` — secrets: `DEEPSEEK_API_KEY`, `HF_ENDPOINT`
- `lingya/config.py` — Pydantic models for config, loads YAML + env overlays. `api_key_env` is resolved from whichever env var actually contains the key.
