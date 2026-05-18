# LingYa

An autonomous AI agent with persistent memory and an evolving personality. As LingYa consumes more information through conversations and ingested content, it develops a unique character.

## Quick Start

```bash
git clone <repo> && cd LingYa

# Set up secrets
cp .env.example .env
# Edit .env and fill in your DEEPSEEK_API_KEY

# Install dependencies
uv sync

# Run
uv run python main.py
```

Supports any OpenAI-compatible API — edit `config.yaml` to switch between DeepSeek, OpenAI, or Ollama.

## Commands

| Command | Description |
|---------|-------------|
| `/fetch <url>` | Ingest a web page into long-term memory |
| `/personality` | View current personality traits |
| `/reflect` | Analyze the current conversation |
| `/history` | Show conversation history |
| `/clear` | Clear short-term memory |
| `/help` | Show available commands |
| `/exit` | Quit |

## Architecture

```
User Input → CLI (Rich)
    ↓
Agent.handle_input()
    ├── ShortTermMemory   (deque, sliding window)
    ├── LongTermMemory    (ChromaDB, vector search)
    ├── PersonalityEngine (Pydantic traits → system prompt)
    ├── LLM Backend       (OpenAI-compatible, lazy init)
    └── SQLite            (conversation persistence)
    ↓
Rich Markdown Response
```

### Key Modules

- **Memory** — Short-term with automatic LLM compression + long-term vector store with semantic retrieval
- **Personality** — 5 configurable traits (curiosity, analytical depth, playfulness, empathy, directness) serialized as a system prompt
- **Ingestion** — Recursive text chunking, local embedding model, URL fetcher
- **LLM Abstraction** — Single backend serves DeepSeek, OpenAI, and Ollama through the same interface

## Configuration

- `config.yaml` — LLM, memory, personality settings (committed, safe to share)
- `.env` — Secrets like API keys (never committed, use `.env.example` as template)

## Roadmap

- **v0.1.0** (current) — CLI chat, memory system, static personality, web ingestion
- **v0.2.0** — Automatic personality evolution, web search, file ingestion, autonomous tool use

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
