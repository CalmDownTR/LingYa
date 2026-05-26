# LingYa

An autonomous AI agent with persistent memory. As LingYa consumes more information through conversations and ingested content, it builds up useful context.

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
| `/reflect` | Analyze the current conversation |
| `/memories` | List all stored memories |
| `/forget <index>` | Delete a memory by its index |
| `/remember <text>` | Manually add a memory |
| `/sessions` | List all sessions |
| `/new` | Start a new session |
| `/switch <id>` | Switch to a session by ID |
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
    ├── LLM Backend       (OpenAI-compatible, lazy init)
    └── SQLite            (conversation persistence)
    ↓
Rich Markdown Response
```

### Key Modules

- **Memory** — Short-term with automatic LLM compression + long-term ChromaDB vector store with semantic retrieval and `memory_store`/`memory_search` agent tools
- **CLI Commands** — `/memories`, `/forget`, `/remember` for manual memory management
- **Ingestion** — Recursive text chunking, local embedding model, URL fetcher
- **LLM Abstraction** — Single backend serves DeepSeek, OpenAI, and Ollama through the same interface

## Configuration

- `config.yaml` — LLM, memory settings (committed, safe to share)
- `.env` — Secrets like API keys (never committed, use `.env.example` as template)

## Roadmap

- **v0.1.0** — CLI chat, memory system, web ingestion
- **v0.2.0** (current) — Semantic memory (store + search), agent tool integration, memory management CLI
- **v0.3.0** — Emotional engine, persona perceptibility
- **v0.4.0** — Proactive interaction, context reinforcement
- **v1.0.0** — Quality closed-loop, documentation, community-ready

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
