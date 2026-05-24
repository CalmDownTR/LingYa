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

- **Memory** — Short-term with automatic LLM compression + long-term vector store with semantic retrieval
- **Ingestion** — Recursive text chunking, local embedding model, URL fetcher
- **LLM Abstraction** — Single backend serves DeepSeek, OpenAI, and Ollama through the same interface

## Configuration

- `config.yaml` — LLM, memory settings (committed, safe to share)
- `.env` — Secrets like API keys (never committed, use `.env.example` as template)

## Roadmap

- **v0.1.0** (current) — CLI chat, memory system, web ingestion
- **v0.2.0** — Web search, file ingestion, autonomous tool use

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
